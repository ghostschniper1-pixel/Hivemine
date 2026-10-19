"""A static, word-level automaton of the grammar for a GPU or C++ mask builder (DESIGN.md 5,
"Where the masks come from").

The Python ``Decoder`` asks a ``SlotProvider`` at every word. The fast path cannot: the C++ core
exports, per ready agent, only *option bitsets* (for each slot, which of its values are
allowed, plus small tables for slots that depend on an earlier choice), and the actor builds
each word's mask on the GPU from the automaton state and those bitsets, with no trip back to
the CPU between words. This module compiles the templates, once, into the tables that needs:

``Automaton.domains[name]`` (``DomainTrie``), one per value domain, over the *full* value set
(``values``, the bitset order):
    ``children[node]``  word id -> child node (node 0 is the root)
    ``terminal[node]``  index of the value whose phrase ends at the node, or -1
    ``incidence``       bool [n_nodes, n_values]: value's phrase passes through (or ends at) node
                        -- the "prefix-node x value incidence"
    ``below``           bool [n_nodes, n_values]: value's phrase goes on past node (passes
                        through one of its children); ``incidence[node]`` is ``below[node]``
                        plus the value ending at the node. A phrase that has reached ``node``
                        can go on iff ``(below[node] & bitset).any()``, and can end there iff
                        ``bitset[terminal[node]]``.
    The entity slot is split in two, as DESIGN.md 5 describes: ``ordinal`` (1-16) and
    ``entity_target`` (entity classes, then ``("dropped", item)``), whose bitset depends on
    the ordinal chosen ("the entity given the ordinal").

``Automaton.states`` / ``transition``: a deterministic automaton over words.
    ``transition``      int32 [n_states, n_words], next state or -1 (dense)
    ``states[s]``       the NFA items the state stands for (several where templates share a
                        prefix: after ``go to the nearest iron`` both ``iron ore`` (block) and
                        ``iron golem`` (entity) are live, and after ``tell team where is iron``
                        both forms of ``where``, ``iron ingot`` (item) and ``iron ore`` (block))
    ``moves[(s, w)]``   tuple of ``Move(src, dst, atoms)``: item ``src`` of state ``s`` goes,
                        on word ``w``, to item ``dst`` of state ``transition[s, w]`` if every
                        atom holds
    ``start``, ``start_entries``  the start state and its items' atoms (verb and template)
    ``final_items``     the items that mean "complete" (after ``.``)

Atoms are conditions on the bitsets, evaluated along each live item's own chosen values:

    ("verb", v)                         v is offered (verbs whose feature flag is off are not
                                        in the automaton at all: ``Automaton.flags``)
    ("choice", slot, id)                template / intent / goal id is offered
    ("enter", slot, domain)             the slot has at least one value (one-word lookahead)
    ("below", slot, domain, node)       some offered value of the slot goes on past node
                                        (``below[node]``): the phrase may stay in the trie
    ("value", slot, domain, index)      the value ending here is offered; it becomes chosen

A word that moves a phrase into trie node ``n`` is allowed through a target item only if that
item survives: the item that stays in the trie needs ``("below", .., n)``, the item that ends
the value at ``n`` and moves on needs ``("value", .., terminal[n])`` and then whatever the next
element needs (``enter`` for a next slot). That is the Decoder's one-word lookahead: a value
whose next slot is empty is masked out even when, over the full value set, it is the prefix of
a longer phrase (``stone`` of ``stone pickaxe``) that the provider does not offer.

The runtime keeps, per agent, the state id plus a bitmask of live items (at most
``max_items`` per state) and their chosen values (registers: only the few dependent slots
need them). The mask of word w is the OR, over moves (s, w) whose source item is live, of the
AND of the move's atoms; feeding w sets the live items of the next state to the move targets
whose atoms held. ``AutomatonWalker`` is the reference implementation of exactly that, driven
by a ``SlotProvider``; the tests check it against the ``Decoder`` word by word, including the
``DeadEndError`` a provider with a dead end deeper than one word causes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hivemind.lang import grammar as G
from hivemind.lang.decoder import SlotProvider, choice_ids
from hivemind.lang.grammar import Lit, Slot
from hivemind.lang.language import Language, Mark, _expand, assemble
from hivemind.lang.types import DeadEndError, EntityRef, Flags

Atom = tuple


@dataclass
class DomainTrie:
    """A phrase trie over the full value set of one domain (see the module docstring)."""

    name: str
    values: tuple
    children: list[dict[int, int]]
    terminal: np.ndarray
    incidence: np.ndarray
    below: np.ndarray
    index: dict[Any, int] = field(repr=False, default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.children)


@dataclass(frozen=True)
class Move:
    src: int  # index into states[s]
    dst: int  # index into states[transition[s, w]]
    atoms: tuple[Atom, ...]


@dataclass
class Automaton:
    words: list[str]
    flags: Flags
    domains: dict[str, DomainTrie]
    items: list[tuple]  # ("t", template node) or ("s", template edge, trie node)
    states: list[tuple[int, ...]]
    transition: np.ndarray
    moves: dict[tuple[int, int], tuple[Move, ...]]
    start: int
    start_entries: tuple[tuple[int, tuple[Atom, ...]], ...]
    final_items: frozenset[int]
    edges: list[tuple[int, Slot, int]]  # template edges into slots: (from node, slot, to node)

    @property
    def max_items(self) -> int:
        return max(len(s) for s in self.states)

    def summary(self) -> str:
        n_moves = sum(len(m) for m in self.moves.values())
        sizes = ", ".join(f"{k} {d.n_nodes}x{len(d.values)}" for k, d in self.domains.items())
        return (f"{len(self.states)} states x {len(self.words)} words ({self.transition.nbytes / 1e6:.1f} MB dense), "
                f"{len(self.items)} NFA items, {n_moves} moves, at most {self.max_items} items per state; "
                f"tries (nodes x values): {sizes}")


def _domain_trie(name: str, values: Iterable[Any], phrase_ids) -> DomainTrie:
    values = tuple(values)
    children: list[dict[int, int]] = [{}]
    terminal = [-1]
    paths: list[list[int]] = []
    for i, v in enumerate(values):
        node = 0
        path = [0]
        for w in phrase_ids(v):
            nxt = children[node].get(w)
            if nxt is None:
                nxt = len(children)
                children[node][w] = nxt
                children.append({})
                terminal.append(-1)
            node = nxt
            path.append(node)
        if terminal[node] != -1:
            raise ValueError(f"{name}: two values with the same phrase")
        terminal[node] = i
        paths.append(path)
    inc = np.zeros((len(children), len(values)), bool)
    below = np.zeros((len(children), len(values)), bool)
    for i, path in enumerate(paths):
        inc[path, i] = True
        below[path[:-1], i] = True  # every node of the path but the one the phrase ends at
    return DomainTrie(name, values, children, np.array(terminal, np.int32), inc, below,
                      {v: i for i, v in enumerate(values)})


def _domains(lang: Language) -> dict[str, DomainTrie]:
    out: dict[str, DomainTrie] = {}
    for dom in sorted(lang.domains):
        if dom == "entity_ref" or dom in ("event_entity", "event_item", "count", "station", "container", "cause"):
            continue
        out[dom] = _domain_trie(dom, lang.domains[dom], lambda v, d=dom: lang.value_ids(d, v))
    out["ordinal"] = _domain_trie("ordinal", range(1, len(G.ORDINALS) + 1), lambda k: (lang.word_id(G.ORDINALS[k - 1]),))
    ent = lang.raw["domains"]["entity_ref"]
    targets = [*ent["entities"], *((G.DROPPED, i) for i in ent["dropped"])]

    def target_ids(t: Any) -> tuple[int, ...]:
        return lang.entity_ref_words(EntityRef(1, t))[1:]

    out["entity_target"] = _domain_trie("entity_target", targets, lambda t: tuple(lang.word_id(w) for w in target_ids(t)))
    return out


def _flat_paths(flags: Flags) -> list[tuple]:
    paths = []
    for verb in G.VERBS:
        if not flags.enabled(G.flag_of(verb)):
            continue
        for tid, els in G.COMMAND_ELEMENTS[verb].items():
            for flat in _expand(els):
                seq: list = [Mark("verb", verb)]
                if G.multi_template(verb):
                    seq.append(Mark("template", tid))
                for e in flat:
                    if isinstance(e, Slot) and e.domain == "entity_ref":
                        seq += [Slot("entity", "ordinal"), Slot("entity", "entity_target")]
                    else:
                        seq.append(e)
                seq.append(Lit(G.END))
                paths.append(tuple(seq))
    return paths


def build_automaton(lang: Language, flags: Flags | None = None) -> Automaton:
    """Compile the grammar over the full value sets. Only verbs whose feature flag is on in
    ``flags`` (default: all off, as for the decoder) can start a command, so the static table
    never lets a disabled verb through, whatever verb bitsets are exported."""
    flags = Flags() if flags is None else flags
    doms = _domains(lang)
    # template trie: node -> list of (element, child)
    tchildren: list[dict] = [{}]
    for seq in _flat_paths(flags):
        node = 0
        for e in seq:
            nxt = tchildren[node].get(e)
            if nxt is None:
                nxt = tchildren[node][e] = len(tchildren)
                tchildren.append({})
            node = nxt
    edges: list[tuple[int, Slot, int]] = []
    edge_of: dict[tuple[int, Slot], int] = {}
    for tn, ch in enumerate(tchildren):
        for e, child in ch.items():
            if isinstance(e, Slot):
                edge_of[(tn, e)] = len(edges)
                edges.append((tn, e, child))

    items: list[tuple] = []
    item_id: dict[tuple, int] = {}

    def iid(it: tuple) -> int:
        i = item_id.get(it)
        if i is None:
            i = item_id[it] = len(items)
            items.append(it)
        return i

    closure_cache: dict[tuple, list[tuple[tuple, tuple]]] = {}

    def closure(it: tuple) -> list[tuple[tuple, tuple]]:
        hit = closure_cache.get(it)
        if hit is not None:
            return hit
        out: list[tuple[tuple, tuple]] = []
        if it[0] == "t":
            tn = it[1]
            ch = tchildren[tn]
            if not ch or any(isinstance(e, Lit) for e in ch):
                out.append((it, ()))
            for e, child in ch.items():
                if isinstance(e, Mark):
                    atom = ("verb", e.value) if e.slot == "verb" else ("choice", e.slot, e.value)
                    out += [(x, (atom, *a)) for x, a in closure(("t", child))]
                elif isinstance(e, Slot):
                    out.append((("s", edge_of[(tn, e)], 0), (("enter", e.name, e.domain),)))
        else:
            _, ei, node = it
            _, slot, child = edges[ei]
            dt = doms[slot.domain]
            if dt.children[node]:  # stay in the trie, if an offered phrase goes on past node
                out.append((it, (("below", slot.name, slot.domain, node),)))
            v = int(dt.terminal[node])
            if v >= 0:
                out += [(x, (("value", slot.name, slot.domain, v), *a)) for x, a in closure(("t", child))]
        closure_cache[it] = out
        return out

    def word_moves(it: tuple) -> list[tuple[int, tuple, tuple]]:
        """(word, target item, atoms) for each word the item can take. A trie edge needs no atom
        of its own: every item of the child's closure needs ``below`` or ``value`` there, each
        of which implies that some offered value passes through the child."""
        if it[0] == "t":
            return [(lang.word_id(e.word), ("t", child), ()) for e, child in tchildren[it[1]].items() if isinstance(e, Lit)]
        _, ei, node = it
        _, slot, _ = edges[ei]
        dt = doms[slot.domain]
        return [(w, ("s", ei, n2), ()) for w, n2 in dt.children[node].items()]

    start_cl = closure(("t", 0))
    start_items = tuple(sorted({iid(x) for x, _ in start_cl}))
    states: list[tuple[int, ...]] = [start_items]
    state_id = {start_items: 0}
    start_entries = tuple((start_items.index(iid(x)), a) for x, a in start_cl)
    moves: dict[tuple[int, int], tuple[Move, ...]] = {}
    trans_rows: list[dict[int, int]] = []
    s = 0
    while s < len(states):
        cur = states[s]
        by_word: dict[int, list[tuple[int, int, tuple]]] = {}
        for si, i in enumerate(cur):
            for w, target, atoms in word_moves(items[i]):
                for x, a in closure(target):
                    by_word.setdefault(w, []).append((si, iid(x), (*atoms, *a)))
        row: dict[int, int] = {}
        for w in sorted(by_word):
            lst = by_word[w]
            nxt = tuple(sorted({d for _, d, _ in lst}))
            ns = state_id.get(nxt)
            if ns is None:
                ns = state_id[nxt] = len(states)
                states.append(nxt)
            row[w] = ns
            moves[(s, w)] = tuple(Move(si, nxt.index(d), a) for si, d, a in lst)
        trans_rows.append(row)
        s += 1
    transition = np.full((len(states), len(lang.words)), -1, np.int32)
    for s, row in enumerate(trans_rows):
        for w, ns in row.items():
            transition[s, w] = ns
    final = frozenset(i for i, it in enumerate(items) if it[0] == "t" and not tchildren[it[1]])
    return Automaton(list(lang.words), flags, doms, items, states, transition, moves, 0, start_entries, final, edges)


@dataclass(frozen=True)
class _Live:
    item: int  # index into the current state's items
    verb: str | None
    chosen: tuple[tuple[str, Any], ...]
    ordinal: int | None  # register for the split entity slot


class AutomatonWalker:
    """Reference mask builder over the static automaton: what the GPU does, driven by a
    SlotProvider instead of exported bitsets. Same interface as ``Decoder``."""

    def __init__(self, auto: Automaton, lang: Language, provider: SlotProvider,
                 flags: Flags | Mapping[str, bool] | Iterable[str] | None = None):
        self.auto = auto
        self.lang = lang
        self.provider = provider
        self.flags = Flags.coerce(flags)
        self._verbs = {v for v in provider.verbs() if self.flags.enabled(G.flag_of(v))}
        self._cache: dict[tuple, Any] = {}
        self.state = auto.start
        live0 = _Live(-1, None, (), None)
        self.live = self._dedupe(x for idx, atoms in auto.start_entries
                                 if (x := self._apply(live0, idx, atoms)) is not None)
        self.ids: list[int] = []

    @staticmethod
    def _dedupe(lives: Iterable[_Live]) -> list[_Live]:
        out, seen = [], set()
        for x in lives:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _bits(self, verb: str, slot: str, dom: str, chosen: tuple, ordinal: int | None) -> np.ndarray:
        key = (verb, slot, dom, chosen, ordinal)
        hit = self._cache.get(key)
        if hit is None:
            opts = self.provider.options(verb, slot, dict(chosen))
            dt = self.auto.domains[dom]
            hit = np.zeros(len(dt.values), bool)
            for v in opts:
                if dom == "ordinal":
                    hit[dt.index[v.ordinal]] = True
                elif dom == "entity_target":
                    if v.ordinal == ordinal:
                        hit[dt.index[v.entity]] = True
                else:
                    hit[dt.index[self.lang.domain_value(dom, v)]] = True
            self._cache[key] = hit
        return hit

    def _apply(self, live: _Live, dst: int, atoms: tuple[Atom, ...]) -> _Live | None:
        verb, chosen, ordinal = live.verb, live.chosen, live.ordinal
        for a in atoms:
            kind = a[0]
            if kind == "verb":
                if a[1] not in self._verbs:
                    return None
                verb = a[1]
            elif kind == "choice":
                _, slot, cid = a
                if cid not in set(self.provider.options(verb, slot, dict(chosen))) or cid not in choice_ids(verb, slot):
                    return None
                chosen = (*chosen, (slot, cid))
            elif kind == "enter":
                if not self._bits(verb, a[1], a[2], chosen, ordinal).any():
                    return None
            elif kind == "below":
                _, slot, dom, node = a
                if not (self.auto.domains[dom].below[node] & self._bits(verb, slot, dom, chosen, ordinal)).any():
                    return None
            elif kind == "value":
                _, slot, dom, idx = a
                if not self._bits(verb, slot, dom, chosen, ordinal)[idx]:
                    return None
                v = self.auto.domains[dom].values[idx]
                if dom == "ordinal":
                    ordinal = v
                elif dom == "entity_target":
                    chosen, ordinal = (*chosen, (slot, EntityRef(ordinal, v))), None
                else:
                    chosen = (*chosen, (slot, v))
            else:  # pragma: no cover
                raise ValueError(a)
        return _Live(dst, verb, chosen, ordinal)

    def _step(self, w: int) -> list[_Live]:
        moves = self.auto.moves.get((self.state, w), ())
        by_src: dict[int, list[_Live]] = {}
        for x in self.live:
            by_src.setdefault(x.item, []).append(x)
        out = []
        for m in moves:
            for x in by_src.get(m.src, ()):
                y = self._apply(x, m.dst, m.atoms)
                if y is not None:
                    out.append(y)
        return self._dedupe(out)

    @property
    def done(self) -> bool:
        items = self.auto.states[self.state]
        return bool(self.live) and all(items[x.item] in self.auto.final_items for x in self.live)

    def allowed(self) -> np.ndarray:
        """As ``Decoder.allowed``: the words that continue some live item; ``DeadEndError``
        when the command is not complete and no word is (the provider offered a dead end
        deeper than the one-word lookahead, or allows no command)."""
        mask = np.zeros(len(self.auto.words), bool)
        if self.done:
            return mask
        for w in np.flatnonzero(self.auto.transition[self.state] >= 0):
            if self._step(int(w)):
                mask[w] = True
        if not mask.any():
            so_far = " ".join(self.auto.words[i] for i in self.ids)
            raise DeadEndError(f"no valid continuation after {so_far!r} (the provider offered a dead end)"
                               if self.ids else "the provider allows no command")
        return mask

    def feed(self, w: int) -> None:
        nxt = self._step(int(w))
        if not nxt:
            raise ValueError(f"word {self.auto.words[int(w)]!r} not allowed")
        self.state = int(self.auto.transition[self.state, int(w)])
        self.live = nxt
        self.ids.append(int(w))

    def command(self):
        cmds = {assemble(x.verb, x.chosen) for x in self.live}
        if not self.done or len(cmds) != 1:
            raise ValueError(f"not one complete command: {cmds}")
        return cmds.pop()

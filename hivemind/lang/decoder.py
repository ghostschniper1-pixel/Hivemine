"""Masked, word-by-word command generation (contracts 4.5, DESIGN.md 5 "Writing a command").

    dec = lang.decoder(provider, flags)          # flags: containers / goals, both off by default
    while not dec.done:
        mask = dec.allowed()                     # np.ndarray[bool], len(lang.words)
        if dec.forced: dec.feed(int(mask.argmax()))
        else: dec.feed(policy_pick(mask))
    cmd = dec.command()

The decoder walks the templates as a nondeterministic automaton. Its state is a small set of
*threads*, each a partial reading of the words so far: a verb, the slot values chosen so far
(left to right, ``template``/``intent``/``goal`` included), the template elements still to come,
and, inside a slot, the node reached in a trie of the phrases the provider offered for it. A
word is allowed if at least one thread can take it and something can still follow it.

When a thread reaches a slot it asks the provider ``options(verb, slot, chosen)``; when it
reaches a choice (``template`` of a verb with several templates, ``intent`` of ``chat``,
``goal`` of ``plan``) it asks for the allowed ids and splits into one thread per id; the next
words decide which survive (``go to the nearest`` is both ``go to the nearest {block}`` and
``go to the {ordinal} {entity}`` until the phrase shows which). A phrase that is a prefix of
another (``zombie`` / ``zombie villager``, ``stone`` / ``stone pickaxe``) does not end at its
last word: the thread both stays in the trie and, tentatively, finishes the slot with that value
and moves on, asking the provider for the next slot as if the value were chosen. Whichever word
comes next (``villager``, or the literal or ``.`` after the slot) settles it, so a phrase ends
only when the next literal word or ``.`` is chosen. ``options`` must therefore be a pure
function of its arguments.

Provider contract (beyond contracts 4.5): every value offered must belong to the slot's full
value set (``Language.domains``; fallback classes never do, so offering one raises
``ValueError``), and must have at least one complete command after it ("no dead ends").
Numbers may be NumPy integers (they become ints); a bool is never a value. The slot ``entity``
takes ``EntityRef``s in commands but entity-class names inside ``plan to kill {mine_count}
{entity}`` (a goal names a kind of mob, not one visible mob; ``slot_domain`` tells which). The
decoder looks one word ahead, so a value whose next slot has no options is masked out, but a
provider that offers a dead end deeper than that makes ``allowed()`` raise ``DeadEndError``.
Verbs whose feature flag is off (``store``, ``take``: ``containers``; ``plan``: ``goals``) are
never allowed, whatever the provider says.

``enumerate_commands`` lists every command a provider allows (for tests and for checking a
provider); ``FullProvider`` offers every value of every slot.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from hivemind.lang import grammar as G
from hivemind.lang.grammar import Choice, Element, Lit, Slot
from hivemind.lang.language import Language, TrieNode, assemble, build_trie
from hivemind.lang.types import Command, DeadEndError, EntityRef, Flags


class SlotProvider(Protocol):
    """Implemented by the simulator (and later the bridge), contracts 4.5."""

    def verbs(self) -> Collection[str]:
        """Verbs with at least one valid command."""
        ...

    def options(self, verb: str, slot: str, chosen: Mapping[str, Any]) -> Collection[Any]:
        """Valid values for one slot, given the verb and the slots already chosen (left to right)."""
        ...


@dataclass(frozen=True)
class TemplateChoice:
    """The ``template`` choice of a verb with several templates (an element of the decoder)."""

    verb: str


def _verb_slot_domains() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for verb, templates in G.COMMAND_ELEMENTS.items():
        doms: dict[str, str] = {}
        stack: list[Element] = [e for els in templates.values() for e in els]
        while stack:
            e = stack.pop()
            if isinstance(e, Slot):
                if doms.setdefault(e.name, e.domain) != e.domain:
                    raise ValueError(f"{verb}: slot {e.name!r} has two domains")
            elif isinstance(e, Choice):
                stack.extend(x for els in G.CHOICE_ELEMENTS[e.slot].values() for x in els)
        out[verb] = doms
    return out


VERB_SLOT_DOMAINS: dict[str, dict[str, str]] = _verb_slot_domains()


def slot_domain(verb: str, slot: str) -> str:
    """The value domain of a slot of a verb (``item`` of ``chat`` is ``chat_item``, ...)."""
    try:
        return VERB_SLOT_DOMAINS[verb][slot]
    except KeyError:
        raise KeyError(f"verb {verb!r} has no slot {slot!r}") from None


def choice_ids(verb: str, slot: str) -> tuple[str, ...]:
    if slot == "template":
        return tuple(G.COMMANDS[verb])
    if slot == "intent":
        return G.INTENTS
    if slot == "goal":
        return G.GOAL_KINDS
    raise KeyError(slot)


def start_elements(verb: str) -> tuple:
    """The elements a command of ``verb`` starts with (a template choice, or its one template)."""
    if G.multi_template(verb):
        return (TemplateChoice(verb), Lit(G.END))
    (els,) = G.COMMAND_ELEMENTS[verb].values()
    return (*els, Lit(G.END))


class FullProvider:
    """A SlotProvider offering every value of every slot (the full v1 sets), for tests and the
    parser's view of the grammar. ``flags`` limits ``verbs()`` (default: everything on)."""

    def __init__(self, lang: Language, flags: Flags | Mapping[str, bool] | Iterable[str] | None = None):
        self.lang = lang
        self.flags = Flags.all() if flags is None else Flags.coerce(flags)

    def verbs(self) -> tuple[str, ...]:
        return tuple(v for v in G.VERBS if self.flags.enabled(G.flag_of(v)))

    def options(self, verb: str, slot: str, chosen: Mapping[str, Any]) -> tuple:
        if slot in G.CHOICE_SLOTS:
            return choice_ids(verb, slot)
        return self.lang.domains[slot_domain(verb, slot)]


@dataclass(frozen=True)
class _Thread:
    verb: str
    chosen: tuple[tuple[str, Any], ...]
    rest: tuple  # elements still to come (Lit, Slot, Choice, TemplateChoice)
    node: TrieNode | None = None  # inside a slot: the trie node reached
    slot: Slot | None = None

    @property
    def waiting(self) -> bool:
        return self.node is not None or bool(self.rest)


class _Options:
    """Provider queries, cached per (verb, slot, chosen) and checked against the grammar."""

    def __init__(self, lang: Language, provider: SlotProvider):
        self.lang = lang
        self.provider = provider
        self._cache: dict[tuple, Any] = {}

    def choices(self, verb: str, slot: str, chosen: tuple) -> tuple[str, ...]:
        key = (verb, slot, chosen)
        hit = self._cache.get(key)
        if hit is None:
            known = choice_ids(verb, slot)
            got = list(self.provider.options(verb, slot, dict(chosen)))
            bad = [x for x in got if x not in known]
            if bad:
                raise ValueError(f"provider offered unknown {slot} {bad} for {verb!r}; known: {list(known)}")
            offered = set(got)
            hit = self._cache[key] = tuple(x for x in known if x in offered)
        return hit

    def values(self, verb: str, slot: Slot, chosen: tuple) -> tuple:
        """The provider's values for a slot, as plain values (NumPy integers become ints)."""
        key = (verb, slot.name, chosen)
        hit = self._cache.get(key)
        if hit is None:
            out = []
            for v in self.provider.options(verb, slot.name, dict(chosen)):
                try:
                    out.append(self.lang.domain_value(slot.domain, v))
                except ValueError:
                    hint = ""
                    if slot.domain == "entity_class" and isinstance(v, EntityRef):
                        hint = (" (the entity of the goal 'kill' is an entity-class name such as 'zombie', "
                                "not an EntityRef)")
                    raise ValueError(f"provider offered {v!r} for {verb} {slot.name}: "
                                     f"not a valid {slot.domain}{hint}") from None
            hit = self._cache[key] = tuple(out)
        return hit

    def trie(self, verb: str, slot: Slot, chosen: tuple) -> TrieNode | None:
        key = ("trie", verb, slot.name, chosen)
        if key not in self._cache:
            vals = self.values(verb, slot, chosen)
            self._cache[key] = build_trie(((v, self.lang.value_ids(slot.domain, v)) for v in vals),
                                          f"{verb} {slot.name}") if vals else None
        return self._cache[key]


class Decoder:
    """Masked word-by-word decoder (see the module docstring)."""

    def __init__(self, lang: Language, provider: SlotProvider,
                 flags: Flags | Mapping[str, bool] | Iterable[str] | None = None):
        self.lang = lang
        self.flags = Flags.coerce(flags)
        self._opts = _Options(lang, provider)
        self.ids: list[int] = []
        verbs = set(provider.verbs())
        unknown = verbs - set(G.VERBS)
        if unknown:
            raise ValueError(f"provider offered unknown verbs {sorted(unknown)}")
        starts = [_Thread(v, (), start_elements(v)) for v in G.VERBS if v in verbs and self.flags.enabled(G.flag_of(v))]
        self._threads = _dedupe(x for t in starts for x in self._expand(t))
        self._succ: dict[int, list[_Thread]] | None = None
        self._mask: np.ndarray | None = None

    # -- automaton -----------------------------------------------------------------------

    def _expand(self, t: _Thread) -> Iterator[_Thread]:
        """Epsilon closure: threads waiting for a word, or complete."""
        if t.node is not None:
            if t.node.children:
                yield t
            if t.node.has_value:
                yield from self._expand(_Thread(t.verb, (*t.chosen, (t.slot.name, t.node.value)), t.rest))
            return
        if not t.rest:
            yield t
            return
        e = t.rest[0]
        if isinstance(e, Lit):
            yield t
        elif isinstance(e, Slot):
            root = self._opts.trie(t.verb, e, t.chosen)
            if root is not None:
                yield from self._expand(_Thread(t.verb, t.chosen, t.rest[1:], root, e))
        elif isinstance(e, TemplateChoice):
            for tid in self._opts.choices(t.verb, "template", t.chosen):
                els = G.COMMAND_ELEMENTS[t.verb][tid]
                yield from self._expand(_Thread(t.verb, (*t.chosen, ("template", tid)), (*els, *t.rest[1:])))
        elif isinstance(e, Choice):
            for cid in self._opts.choices(t.verb, e.slot, t.chosen):
                els = G.CHOICE_ELEMENTS[e.slot][cid]
                yield from self._expand(_Thread(t.verb, (*t.chosen, (e.slot, cid)), (*els, *t.rest[1:])))
        else:  # pragma: no cover
            raise TypeError(e)

    def _successors(self) -> dict[int, list[_Thread]]:
        if self._succ is None:
            raw: dict[int, list[_Thread]] = {}
            for t in self._threads:
                if t.node is not None:
                    for w, child in t.node.children.items():
                        raw.setdefault(w, []).append(_Thread(t.verb, t.chosen, t.rest, child, t.slot))
                elif t.rest:
                    e = t.rest[0]
                    raw.setdefault(self.lang.word_id(e.word), []).append(_Thread(t.verb, t.chosen, t.rest[1:]))
            succ = {}
            for w in sorted(raw):
                nxt = _dedupe(x for t in raw[w] for x in self._expand(t))
                if nxt:
                    succ[w] = nxt
            self._succ = succ
        return self._succ

    # -- public API ----------------------------------------------------------------------

    @property
    def done(self) -> bool:
        """True once a complete command has been written (after its final '.')."""
        return bool(self._threads) and not any(t.waiting for t in self._threads)

    def allowed(self) -> np.ndarray:
        """Bool mask over the vocabulary: the words that continue some valid command."""
        if self._mask is None:
            mask = np.zeros(len(self.lang.words), bool)
            if not self.done:
                succ = self._successors()
                if not succ:
                    so_far = " ".join(self.lang.words[i] for i in self.ids)
                    raise DeadEndError(f"no valid continuation after {so_far!r} (the provider offered a dead end)"
                                       if self.ids else "the provider allows no command")
                mask[list(succ)] = True
            self._mask = mask
        return self._mask

    @property
    def forced(self) -> bool:
        """True when exactly one word is allowed (fed in without a policy decision)."""
        return int(self.allowed().sum()) == 1

    def allowed_words(self) -> list[str]:
        return [self.lang.words[i] for i in np.flatnonzero(self.allowed())]

    def feed(self, word_id: int) -> None:
        """Advance by one word; ValueError if the word is not allowed."""
        word_id = int(word_id)
        if self.done:
            raise ValueError("the command is complete")
        self.allowed()
        succ = self._successors()
        if word_id not in succ:
            w = self.lang.words[word_id] if 0 <= word_id < len(self.lang.words) else word_id
            so_far = " ".join(self.lang.words[i] for i in self.ids)
            raise ValueError(f"{w!r} is not allowed after {so_far!r}; allowed: {self.allowed_words()[:12]}")
        self._threads = succ[word_id]
        self.ids.append(word_id)
        self._succ = None
        self._mask = None

    def feed_word(self, word: str) -> None:
        self.feed(self.lang.word_id(word))

    def command(self) -> Command:
        """The finished command (after ``done``)."""
        if not self.done:
            raise ValueError("the command is not complete")
        cmds = {assemble(t.verb, t.chosen) for t in self._threads}
        if len(cmds) != 1:
            raise ValueError(f"ambiguous command {self.text()!r}: {sorted(map(repr, cmds))}")
        return cmds.pop()

    def text(self) -> str:
        from hivemind.lang.language import join_words

        return join_words([self.lang.words[i] for i in self.ids])

    def fork(self) -> Decoder:
        """An independent copy at the same point (sharing the provider cache), for search."""
        twin = object.__new__(Decoder)
        twin.lang, twin.flags, twin._opts = self.lang, self.flags, self._opts
        twin.ids = list(self.ids)
        twin._threads, twin._succ, twin._mask = self._threads, self._succ, self._mask
        return twin

    @property
    def n_threads(self) -> int:
        return len(self._threads)


def _dedupe(threads: Iterable[_Thread]) -> list[_Thread]:
    seen: set = set()
    out = []
    for t in threads:
        key = (t.verb, t.chosen, t.rest, id(t.node), t.slot)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def enumerate_commands(lang: Language, provider: SlotProvider,
                       flags: Flags | Mapping[str, bool] | Iterable[str] | None = None,
                       verbs: Iterable[str] | None = None) -> Iterator[Command]:
    """Every command the provider allows, slot by slot left to right (no words involved).

    With ``FullProvider`` and all flags on this is the whole v1 command space (about 330,000
    commands, most of them ``tell ...`` and ``give ...``). ``verbs`` restricts the verbs."""
    flags = Flags.coerce(flags)
    offered = set(provider.verbs())
    wanted = set(verbs) if verbs is not None else None
    opts = _Options(lang, provider)

    def rec(verb: str, rest: tuple, chosen: tuple) -> Iterator[Command]:
        while rest and isinstance(rest[0], Lit):
            rest = rest[1:]
        if not rest:
            yield assemble(verb, chosen)
            return
        e = rest[0]
        if isinstance(e, Slot):
            for v in opts.values(verb, e, chosen):
                yield from rec(verb, rest[1:], (*chosen, (e.name, v)))
        elif isinstance(e, TemplateChoice):
            for tid in opts.choices(verb, "template", chosen):
                yield from rec(verb, (*G.COMMAND_ELEMENTS[verb][tid], *rest[1:]), (*chosen, ("template", tid)))
        else:
            for cid in opts.choices(verb, e.slot, chosen):
                yield from rec(verb, (*G.CHOICE_ELEMENTS[e.slot][cid], *rest[1:]), (*chosen, (e.slot, cid)))

    for verb in G.VERBS:
        if verb in offered and flags.enabled(G.flag_of(verb)) and (wanted is None or verb in wanted):
            yield from rec(verb, start_elements(verb), ())

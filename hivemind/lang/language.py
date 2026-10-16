"""The ``Language``: words, phrases, rendering and strict parsing (contracts 4.4).

    from hivemind.lang import Language, Command
    lang = Language.load()                         # data/vocab-1.21.1.json (cached, hash-checked)
    cmd = lang.parse("mine 4 iron ore.")           # Command('mine', (('block', 'iron_ore'), ('mine_count', 4)))
    lang.render(cmd)                               # 'mine 4 iron ore.'
    lang.encode(cmd); lang.decode(ids)             # word ids, '.' included, no <bos>
    lang.phrase("item", "iron_ingot")              # ('iron', 'ingot')
    lang.result_words("INTERRUPTED", cause="creeper")   # ['interrupted', 'by', 'creeper']
    lang.render_event("player_gave_me", player="blue_7", n=26, item="bread")  # 'blue 7 gave me 20 bread'

Rendering joins words with single spaces and attaches the final ``.`` to the last word. Parsing
is strict: it accepts exactly what ``render`` writes, over the full v1 value sets of every slot
(``domain``), with every verb available whatever the feature flags. It is a backtracking match
of the words against the compiled templates, trying every way a phrase can end (``zombie`` /
``zombie villager``), and it fails if the words have no reading or (a grammar bug) several.
The decoder (``hivemind.lang.decoder``) is a separate implementation of the same grammar; the
tests check that the two agree on every command.

Values: block, item and entity classes by their class names; players as ``red_0`` .. ``blue_19``
and ``human`` (phrase ``red 0``); addressees also ``everyone`` and ``team``; nonce words by
themselves; directions by name; patterns ``here``, ``in_front``, ``under_me``, ``around_me``,
``for_light``; numbers as ints.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from hivemind.lang import grammar as G
from hivemind.lang.grammar import Choice, Element, Lit, Slot
from hivemind.lang.types import Command, EntityRef, Event, Flags, Goal, ParseError, Sentence, _Frame, plain_value
from hivemind.rules.classes import canonical_json
from hivemind.rules.loader import GAME_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOCAB_PATH = REPO_ROOT / "data" / f"vocab-{GAME_VERSION}.json"

_NO_VALUE = object()
_TEMPLATE_KINDS = {"verb": "command", "intent": "sentence", "goal": "goal", "result": "result", "event": "event"}


def vocab_hash(vocab: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of a vocabulary without its ``hash`` field (build provenance)."""
    body = {k: v for k, v in vocab.items() if k != "hash"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


# What the network's input and output layers and the grammar masks depend on: the words and
# their order, the concept rows (class ids and phrases), the slot value lists (bitset order) and
# the templates. Not the concepts hash, the word sources or the human aliases.
WORDS_HASH_FIELDS = ("version", "words", "concepts", "domains", "templates")


def words_hash(vocab: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of the ``WORDS_HASH_FIELDS`` of a vocabulary: equal for two
    vocabularies a checkpoint can use alike, even when the rules or concepts were rebuilt."""
    body = {k: vocab[k] for k in WORDS_HASH_FIELDS}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


class TrieNode:
    """A node of a phrase trie over word ids; ``value`` is set where a phrase ends."""

    __slots__ = ("children", "value")

    def __init__(self) -> None:
        self.children: dict[int, TrieNode] = {}
        self.value: Any = _NO_VALUE

    @property
    def has_value(self) -> bool:
        return self.value is not _NO_VALUE


def build_trie(pairs: Iterable[tuple[Any, Sequence[int]]], what: str = "slot") -> TrieNode:
    """A trie of (value, phrase ids) pairs. Two values with the same phrase raise ValueError."""
    root = TrieNode()
    for value, ids in pairs:
        node = root
        for w in ids:
            nxt = node.children.get(w)
            if nxt is None:
                nxt = node.children[w] = TrieNode()
            node = nxt
        if node.has_value and node.value != value:
            raise ValueError(f"{what}: {node.value!r} and {value!r} have the same phrase")
        node.value = value
    return root


@dataclass(frozen=True)
class Mark:
    """A zero-width element of a flattened path: records a choice (template, intent or goal)."""

    slot: str
    value: str


@dataclass(frozen=True)
class FlatTemplate:
    """One complete template, choices expanded, as a flat element list for matching."""

    verb: str | None  # None for sentences, goals, events and results
    head: str  # template id, intent, goal kind, event kind or result code
    elements: tuple[Lit | Slot | Mark, ...]


def assemble(verb: str, pairs: Iterable[tuple[str, Any]]) -> Command:
    """Command from the (slot, value) pairs chosen left to right, nested sentence or goal included."""
    nest = G.NESTED.get(verb)
    if nest is None:
        return Command(verb, tuple(pairs))
    outer: list[tuple[str, Any]] = []
    inner: list[tuple[str, Any]] = []
    head = None
    for k, v in pairs:
        if head is None and k == nest.slot:
            head = v
        elif head is None:
            outer.append((k, v))
        else:
            inner.append((k, v))
    if head is None:
        raise ValueError(f"{verb}: no {nest.slot} chosen")
    frame = Sentence(head, **dict(inner)) if nest.slot == "intent" else Goal(head, **dict(inner))
    return Command(verb, (*outer, (nest.field, frame)))


def split_text(text: str) -> list[str]:
    """Canonical text -> words: split on whitespace, the final '.' as its own word."""
    words = text.split()
    if words and words[-1] != "." and words[-1].endswith("."):
        words[-1] = words[-1][:-1]
        words.append(".")
    return words


def join_words(words: Sequence[str]) -> str:
    """Words -> text: single spaces, a final '.' attached to the word before it."""
    if words and words[-1] == "." and len(words) > 1:
        return " ".join(words[:-1]) + "."
    return " ".join(words)


class Language:
    """The shared vocabulary and grammar (see the module docstring)."""

    def __init__(self, vocab: Mapping[str, Any], path: Path | None = None):
        if vocab.get("version") != GAME_VERSION:
            raise ValueError(f"vocabulary is for version {vocab.get('version')}, expected {GAME_VERSION}")
        expected = {"command": G.COMMANDS, "sentence": G.SENTENCES, "goal": G.GOALS, "result": G.RESULTS,
                    "event": G.EVENTS, "verb_flags": G.VERB_FLAG}
        if json.loads(json.dumps(vocab["templates"])) != json.loads(json.dumps(expected)):
            raise ValueError("vocabulary file was built from different templates: rebuild it "
                             "(python -m hivemind.lang.vocab)")
        self.raw = vocab
        self.path = path
        self.version: str = vocab["version"]
        self.hash: str = vocab["hash"]  # the whole file (provenance: record with runs and replays)
        self.words_hash: str = vocab["words_hash"]  # what a checkpoint depends on (hivemind.lang.vocab)
        self.concepts_hash: str = vocab["concepts_hash"]
        self.words: list[str] = list(vocab["words"])
        if tuple(self.words[:4]) != G.RESERVED:
            raise ValueError("vocabulary must start with <pad>, <unk>, <bos>, '.'")
        self._ids: dict[str, int] = {w: i for i, w in enumerate(self.words)}
        if len(self._ids) != len(self.words):
            raise ValueError("vocabulary has repeated words")

        # concept tables
        self._phrase: dict[str, dict[Any, tuple[str, ...]]] = {}
        self._rows: dict[str, dict[Any, dict]] = {}
        for r in vocab["concepts"]:
            self._phrase.setdefault(r["kind"], {})[r["name"]] = tuple(r["phrase"])
            self._rows.setdefault(r["kind"], {})[r["name"]] = r
        self._concept: dict[str, dict[tuple[str, ...], Any]] = {}
        for kind, table in self._phrase.items():
            inv: dict[tuple[str, ...], Any] = {}
            for name, ph in table.items():
                inv[ph] = _NO_VALUE if ph in inv else name  # a repeated (fallback) phrase names nothing
            self._concept[kind] = inv

        # slot domains: values in a fixed order and their phrases
        d = vocab["domains"]
        self.domains: dict[str, tuple] = {}
        self._dphrase: dict[str, dict[Any, tuple[str, ...]]] = {}
        for name, values in d.items():
            if name == "entity_ref":
                continue
            self.domains[name] = tuple(values)
            self._dphrase[name] = {v: self._value_phrase_raw(name, v) for v in values}
        ent = d["entity_ref"]
        refs = [EntityRef(k, e) for k in ent["ordinals"] for e in ent["entities"]]
        refs += [EntityRef(k, (G.DROPPED, i)) for k in ent["ordinals"] for i in ent["dropped"]]
        self.domains["entity_ref"] = tuple(refs)
        self._dphrase["entity_ref"] = {r: self.entity_ref_words(r) for r in refs}
        self._dids: dict[str, dict[Any, tuple[int, ...]]] = {
            dom: {v: tuple(self._ids[w] for w in ph) for v, ph in table.items()} for dom, table in self._dphrase.items()
        }
        self._tries: dict[str, TrieNode] = {}
        self._paths: dict[str, Any] = {}
        self._human = None
        self.aliases: dict[str, dict[str, str]] = {k: dict(v) for k, v in vocab.get("human_aliases", {}).items()}

    # -- construction ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None, verify: bool = True) -> Language:
        """Load (once, cached per path) the vocabulary file; ``verify`` checks its hash."""
        p = Path(path) if path is not None else DEFAULT_VOCAB_PATH
        return _load_cached(str(p.resolve()), verify)

    @classmethod
    def build(cls, concepts=None) -> Language:
        """Build the vocabulary in memory from the class tables (default: the committed file)."""
        from hivemind.lang.vocab import build_vocab

        return cls(build_vocab(concepts))

    def __repr__(self) -> str:
        return f"Language(version={self.version!r}, words={len(self.words)}, hash={self.hash[:12]})"

    def __len__(self) -> int:
        return len(self.words)

    def compute_hash(self) -> str:
        return vocab_hash(dict(self.raw))

    def compute_words_hash(self) -> str:
        return words_hash(self.raw)

    @property
    def templates(self) -> Mapping[str, Any]:
        """The template text by kind: ``command`` (verb -> template id -> text), ``sentence``,
        ``goal``, ``result``, ``event`` (contracts 4.3), and ``verb_flags``. Verbs, intents,
        goals and result codes are named by these templates rather than by a phrase of their own."""
        return self.raw["templates"]

    # -- words -------------------------------------------------------------------------

    def word_id(self, word: str) -> int:
        try:
            return self._ids[word]
        except KeyError:
            raise KeyError(f"{word!r} is not in the vocabulary") from None

    def word(self, i: int) -> str:
        return self.words[i]

    def has_word(self, word: str) -> bool:
        return word in self._ids

    def ids(self, words: Iterable[str]) -> list[int]:
        return [self.word_id(w) for w in words]

    # -- phrases -----------------------------------------------------------------------

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(self._phrase)

    def phrase(self, kind: str, name: Any) -> tuple[str, ...]:
        """The words of a concept: kind is block, item, entity, player, setter, addressee,
        nonce, number, ordinal, dir, pattern, station, container, cause, relation, place_kind
        or equip_slot. Verbs, result codes, chat intents and goals have no phrase of their
        own: they are named by their templates (``templates``, ``result_words``)."""
        try:
            return self._phrase[kind][name]
        except KeyError:
            if kind in _TEMPLATE_KINDS:
                raise KeyError(f"a {kind} has no phrase of its own: it is named by its template "
                               f"(Language.templates[{_TEMPLATE_KINDS[kind]!r}], result_words)") from None
            raise KeyError(f"no {kind} named {name!r}") from None

    def concept(self, kind: str, words: Sequence[str]) -> Any:
        """The inverse of ``phrase``. A phrase shared by several fallback classes names none of
        them and raises KeyError, as does an unknown phrase."""
        name = self._concept.get(kind, {}).get(tuple(words), _NO_VALUE)
        if name is _NO_VALUE:
            raise KeyError(f"no single {kind} with phrase {' '.join(words)!r}")
        return name

    def row(self, kind: str, name: Any) -> dict:
        """The concept row of the vocabulary file (class id, fallback, resource, v1, ...)."""
        return self._rows[kind][name]

    def entity_ref_words(self, ref: EntityRef) -> tuple[str, ...]:
        """``EntityRef(2, "zombie")`` -> ('second', 'zombie'); dropped items -> ('nearest', 'dropped', 'bread')."""
        item = ref.dropped_item
        tail = (G.DROPPED, *self._phrase["item"][item]) if item is not None else self._phrase["entity"][ref.entity]
        return (G.ORDINALS[ref.ordinal - 1], *tail)

    def _value_phrase_raw(self, domain: str, v: Any) -> tuple[str, ...]:
        if domain in ("block", "station"):
            return self._phrase["block"][v]
        if domain in ("item", "event_item"):
            return self._phrase["item"][v]
        if domain == "chat_item":
            return self._phrase["nonce"][v] if v in self._phrase["nonce"] else self._phrase["item"][v]
        if domain in ("entity_class", "event_entity"):
            return self._phrase["entity"][v]
        if domain == "player":
            return self._phrase["player"][v]
        if domain == "addressee":
            return self._phrase["player"][v] if v in self._phrase["player"] else self._phrase["addressee"][v]
        if domain in ("dir8", "dir4"):
            return self._phrase["dir"][v]
        if domain == "pattern":
            return self._phrase["pattern"][v]
        if domain == "container":
            return self._phrase["container"][v]
        if domain == "cause":
            return self._phrase["cause"][v] if v in self._phrase["cause"] else self._phrase["entity"][v]
        if domain in G.INT_DOMAINS:
            return (str(v),)
        raise KeyError(f"unknown domain {domain!r}")

    def domain_value(self, domain: str, value: Any) -> Any:
        """``value`` as the plain value of a domain (a NumPy integer becomes an int); ValueError
        if it is not in the domain. A bool or a float is never a number (``True == 1``)."""
        try:
            v = plain_value(value)
        except ValueError:
            raise ValueError(f"{value!r} is not a valid {domain}") from None
        if domain in G.INT_DOMAINS and type(v) is not int:
            raise ValueError(f"{value!r} is not a valid {domain} (not an int)")
        table = self._dphrase.get(domain)
        if table is None:
            raise ValueError(f"unknown domain {domain!r}")
        try:
            if v in table:
                return v
        except TypeError:  # unhashable
            pass
        raise ValueError(f"{value!r} is not a valid {domain}")

    def value_words(self, domain: str, value: Any) -> tuple[str, ...]:
        """The words of a slot value in a domain; ValueError if the value is not in the domain
        (a fallback class, a non-v1 item, a count that is not a bin, ...)."""
        return self._dphrase[domain][self.domain_value(domain, value)]

    def value_ids(self, domain: str, value: Any) -> tuple[int, ...]:
        return self._dids[domain][self.domain_value(domain, value)]

    def in_domain(self, domain: str, value: Any) -> bool:
        try:
            self.domain_value(domain, value)
        except ValueError:
            return False
        return True

    def trie(self, domain: str) -> TrieNode:
        """The phrase trie over the full value set of a domain (cached)."""
        t = self._tries.get(domain)
        if t is None:
            t = self._tries[domain] = build_trie(self._dids[domain].items(), domain)
        return t

    def class_phrase_ids(self, kind: str) -> np.ndarray:
        """int32 [number of class ids, longest phrase] of word ids, padded with <pad> (0), indexed
        by class id (blocks: 256 rows, buried at 255): the phrase of every class for the
        observation encoder (DESIGN.md 6.4)."""
        rows = [r for r in self.raw["concepts"] if r["kind"] == kind]
        n = 256 if kind == "block" else 1 + max(r["id"] for r in rows)
        width = max(len(r["phrase"]) for r in rows)
        out = np.zeros((n, width), np.int32)
        for r in rows:
            out[r["id"], : len(r["phrase"])] = [self._ids[w] for w in r["phrase"]]
        return out

    # -- rendering ---------------------------------------------------------------------

    def _render_elements(self, elements: Sequence[Element], values: Mapping[str, Any], what: str) -> list[str]:
        expected = set(G.slot_names(elements)) | {e.field for e in elements if isinstance(e, Choice)}
        given = set(values) - {"template"}
        if given != expected:
            raise ValueError(f"{what}: slots {sorted(given)} do not match the template's {sorted(expected)}")
        out: list[str] = []
        for e in elements:
            if isinstance(e, Lit):
                out.append(e.word)
            elif isinstance(e, Slot):
                v = values[e.name]
                try:
                    out.extend(self.value_words(e.domain, v))
                except ValueError:
                    raise ValueError(f"{what}: {v!r} is not a valid {e.name} ({e.domain})") from None
            else:
                frame = values[e.field]
                cls = Sentence if e.slot == "intent" else Goal
                if type(frame) is not cls:  # a Goal in a chat command is an invalid command (ValueError)
                    raise ValueError(f"{what}: {e.field} must be a {cls.__name__}, got {frame!r}")
                out.extend(self._frame_words(frame))
        return out

    def _frame_words(self, frame: _Frame) -> list[str]:
        table = {Sentence: G.SENTENCE_ELEMENTS, Goal: G.GOAL_ELEMENTS, Event: G.EVENT_ELEMENTS}[type(frame)]
        els = table.get(frame._head)
        if els is None:
            raise ValueError(f"unknown {type(frame).__name__} {frame._head!r}")
        values = frame.slots
        if isinstance(frame, Event) and "n" in values:
            n = values["n"]
            if isinstance(n, bool) or not isinstance(n, int):
                raise ValueError(f"event count must be an int, got {n!r}")
            values["n"] = G.snap_count(n)
        return self._render_elements(els, values, f"{type(frame).__name__} {frame._head!r}")

    def to_words(self, x: Command | Sentence | Goal | Event) -> list[str]:
        """A command's words, final '.' included; a sentence's, goal's or event's words (no '.')."""
        if isinstance(x, _Frame):
            return self._frame_words(x)
        if not isinstance(x, Command):
            raise TypeError(f"cannot render {x!r}")
        templates = G.COMMAND_ELEMENTS.get(x.verb)
        if templates is None:
            raise ValueError(f"unknown verb {x.verb!r}")
        if G.multi_template(x.verb):
            tid = x.get("template")
            if tid not in templates:
                raise ValueError(f"{x.verb}: template must be one of {list(templates)}, got {tid!r}")
        else:
            if "template" in x.slots:
                raise ValueError(f"{x.verb} has one template; Command.args must not have 'template'")
            (tid,) = templates
        return [*self._render_elements(templates[tid], x.slots, f"{x.verb} ({tid})"), G.END]

    def render(self, x: Command | Sentence | Goal | Event) -> str:
        """``"mine 4 iron ore."``: words joined by spaces, the final '.' attached."""
        return join_words(self.to_words(x))

    def encode(self, x: Command | Sentence | Goal | Event) -> list[int]:
        return [self._ids[w] for w in self.to_words(x)]

    def decode(self, ids: Sequence[int]) -> Command:
        """Word ids -> Command (a leading <bos> and trailing <pad>s are ignored); ParseError if invalid."""
        ids = list(ids)
        if ids and ids[0] == G.BOS_ID:
            ids = ids[1:]
        while ids and ids[-1] == G.PAD_ID:
            ids.pop()
        bad = [i for i in ids if not 0 <= int(i) < len(self.words)]
        if bad:
            raise ParseError(f"word ids out of range: {bad}")
        return self.parse([self.words[int(i)] for i in ids])

    def chat_text(self, addressee: str, sentence: Sentence) -> str:
        """In-game chat text, addressee first (DESIGN.md 9): ``everyone, i need 3 iron ingot.``
        The exact format is fixed in M2."""
        who = " ".join(self.value_words("addressee", addressee))
        return f"{who}, {' '.join(self.to_words(sentence))}."

    # -- results and events --------------------------------------------------------------

    def result_words(self, code: str, cause: str | None = None, station: str | None = None,
                     container: str | None = None) -> list[str]:
        """The result words of a failure code (DESIGN.md 5): ``no crafting table nearby``,
        ``chest full``, ``interrupted by creeper``. ``station`` is a block-class name
        (crafting_table, furnace, bed); ``container`` is ``inventory`` (the default) or
        ``chest``; ``cause`` is ``damage``, ``message`` or a hostile entity class."""
        els = G.RESULT_ELEMENTS.get(code)
        if els is None:
            raise ValueError(f"unknown result code {code!r}; one of {list(G.RESULT_CODES)}")
        given = {"cause": cause, "station": station, "container": container}
        names = set(G.slot_names(els))
        if code == "INV_FULL" and container is None:
            given["container"] = "inventory"
        extra = [k for k, v in given.items() if v is not None and k not in names]
        if extra:
            raise ValueError(f"{code} takes no {', '.join(extra)}")
        missing = [k for k in names if given[k] is None]
        if missing:
            raise ValueError(f"{code} needs {', '.join(missing)}")
        return self._render_elements(els, {k: given[k] for k in names}, f"result {code}")

    def event_words(self, kind: str, **fields: Any) -> list[str]:
        """The words of an event sentence (contracts 4.3); ``n`` is snapped to the nearest
        number token (ties down, above 64 is 64)."""
        return self._frame_words(Event(kind, **fields))

    def render_event(self, kind: str, **fields: Any) -> str:
        return " ".join(self.event_words(kind, **fields))

    # -- strict parsing ------------------------------------------------------------------

    def _command_paths(self) -> dict[int, list[FlatTemplate]]:
        """First word id -> the flattened paths of the verb it starts (cached)."""
        cached = self._paths.get("command")
        if cached is None:
            cached = {}
            for verb, templates in G.COMMAND_ELEMENTS.items():
                for tid, els in templates.items():
                    head: tuple = (Mark("template", tid),) if G.multi_template(verb) else ()
                    for flat in _expand(els):
                        path = FlatTemplate(verb, tid, (*head, *flat, Lit(G.END)))
                        first = self._ids[els[0].word]  # type: ignore[union-attr]
                        cached.setdefault(first, []).append(path)
            self._paths["command"] = cached
        return cached

    def _frame_paths(self, kind: str) -> list[FlatTemplate]:
        cached = self._paths.get(kind)
        if cached is None:
            table = {"sentence": G.SENTENCE_ELEMENTS, "goal": G.GOAL_ELEMENTS, "event": G.EVENT_ELEMENTS,
                     "result": G.RESULT_ELEMENTS}[kind]
            cached = self._paths[kind] = [FlatTemplate(None, head, tuple(els)) for head, els in table.items()]  # type: ignore[arg-type]
        return cached

    def _to_ids(self, words: Sequence[str]) -> list[int]:
        for i, w in enumerate(words):
            if w not in self._ids:
                raise ParseError(f"unknown word {w!r} at position {i} in {join_words(words)!r}",
                                 words=tuple(words), position=i)
        return [self._ids[w] for w in words]

    def _match(self, paths: Sequence[FlatTemplate], ids: Sequence[int]) -> tuple[list[tuple[FlatTemplate, tuple]], int, set[str]]:
        """All complete readings of ``ids`` by the paths, plus the furthest failure position and
        what was expected there (for error messages)."""
        n = len(ids)
        found: list[tuple[FlatTemplate, tuple]] = []
        far = [-1, set()]

        def note(pos: int, what: str) -> None:
            if pos > far[0]:
                far[0], far[1] = pos, {what}
            elif pos == far[0]:
                far[1].add(what)

        def rec(path: FlatTemplate, ei: int, wi: int, pairs: tuple) -> None:
            els = path.elements
            while ei < len(els):
                e = els[ei]
                if isinstance(e, Mark):
                    pairs = (*pairs, (e.slot, e.value))
                    ei += 1
                elif isinstance(e, Lit):
                    if wi < n and self.words[ids[wi]] == e.word:
                        ei += 1
                        wi += 1
                    else:
                        note(wi, repr(e.word))
                        return
                else:
                    node = self.trie(e.domain)
                    j = wi
                    while j < n:
                        child = node.children.get(ids[j])
                        if child is None:
                            break
                        node = child
                        j += 1
                        if node.has_value:
                            rec(path, ei + 1, j, (*pairs, (e.name, node.value)))
                    if j == wi:
                        note(wi, f"{{{e.name}}}")
                    else:
                        nxt = sorted(self.words[w] for w in node.children)
                        if len(nxt) <= 6:
                            for w in nxt:
                                note(j, repr(w))
                        else:
                            note(j, f"the rest of a {{{e.name}}} ({', '.join(map(repr, nxt[:4]))}, ...)")
                    return
            if wi == n:
                found.append((path, pairs))
            else:
                note(wi, "the end")

        for p in paths:
            rec(p, 0, 0, ())
        return found, far[0], far[1]

    def _fail(self, what: str, words: Sequence[str], pos: int, expected: set[str]) -> ParseError:
        exp = tuple(sorted(expected))
        got = repr(words[pos]) if 0 <= pos < len(words) else "the end"
        before = " ".join(words[:pos]) if pos > 0 else ""
        where = f"after {before!r}" if before else "at the start"
        return ParseError(f"not a valid {what}: {join_words(words)!r}: {where} expected {' or '.join(exp)}, got {got}",
                          words=tuple(words), position=pos, expected=exp)

    def parse(self, text_or_words: str | Sequence[str]) -> Command:
        """Strict canonical parse of a command (``render``'s output); raises ParseError."""
        words = split_text(text_or_words) if isinstance(text_or_words, str) else list(text_or_words)
        if not words:
            raise ParseError("empty command", words=(), position=0)
        ids = self._to_ids(words)
        paths = self._command_paths().get(ids[0])
        if paths is None:
            firsts = sorted({self.words[i] for i in self._command_paths()})
            raise ParseError(f"not a command: {join_words(words)!r} does not start with a verb "
                             f"({', '.join(firsts)})", words=tuple(words), position=0,
                             expected=tuple(repr(f) for f in firsts))
        found, pos, expected = self._match(paths, ids)
        cmds = {assemble(p.verb, pairs) for p, pairs in found}  # type: ignore[arg-type]
        if not cmds:
            raise self._fail("command", words, pos, expected)
        if len(cmds) > 1:
            raise ParseError(f"ambiguous command {' '.join(words)!r}: {sorted(map(repr, cmds))}", words=tuple(words))
        return cmds.pop()

    def _parse_frame(self, kind: str, words: str | Sequence[str]) -> tuple[str, dict]:
        ws = words.split() if isinstance(words, str) else list(words)
        if ws and ws[-1] == G.END:
            ws = ws[:-1]
        ids = self._to_ids(ws)
        found, pos, expected = self._match(self._frame_paths(kind), ids)
        readings = {(p.head, pairs) for p, pairs in found}
        if not readings:
            raise self._fail(kind, ws, pos, expected)
        if len(readings) > 1:
            raise ParseError(f"ambiguous {kind} {' '.join(ws)!r}: {sorted(readings)}", words=tuple(ws))
        head, pairs = readings.pop()
        return head, dict(pairs)

    def parse_sentence(self, words: str | Sequence[str]) -> Sentence:
        head, slots = self._parse_frame("sentence", words)
        return Sentence(head, **slots)

    def parse_goal(self, words: str | Sequence[str]) -> Goal:
        head, slots = self._parse_frame("goal", words)
        return Goal(head, **slots)

    def parse_event(self, words: str | Sequence[str]) -> Event:
        """Event sentence -> Event (``n`` is the snapped count that the words show)."""
        head, slots = self._parse_frame("event", words)
        return Event(head, **slots)

    def parse_result(self, words: str | Sequence[str]) -> tuple[str, dict[str, Any]]:
        """Result words -> (code, {cause | station | container: value})."""
        return self._parse_frame("result", words)

    # -- decoding, enumeration, automaton, human input (other modules) -------------------

    def decoder(self, provider, flags: Flags | Mapping[str, bool] | Iterable[str] | None = None):
        """A masked word-by-word decoder over ``provider``'s options (contracts 4.5)."""
        from hivemind.lang.decoder import Decoder

        return Decoder(self, provider, flags)

    def enumerate_commands(self, provider, flags: Flags | Mapping[str, bool] | Iterable[str] | None = None,
                           verbs: Iterable[str] | None = None):
        """Every command ``provider`` allows (slot by slot, left to right)."""
        from hivemind.lang.decoder import enumerate_commands

        return enumerate_commands(self, provider, flags, verbs)

    def full_provider(self, flags: Flags | Mapping[str, bool] | Iterable[str] | None = None):
        """A SlotProvider that offers every value of every slot (the full v1 sets)."""
        from hivemind.lang.decoder import FullProvider

        return FullProvider(self, flags if flags is not None else Flags.all())

    def automaton(self, flags: Flags | Mapping[str, bool] | Iterable[str] | None = None):
        """The static word-level automaton over the full value sets (hivemind.lang.automaton).
        Only verbs whose feature flag is on can start a command (default: all flags off, as
        for ``decoder``)."""
        from hivemind.lang.automaton import build_automaton

        return build_automaton(self, Flags.coerce(flags))

    def _human_reader(self):
        if self._human is None:
            from hivemind.lang.human import HumanReader

            self._human = HumanReader(self)
        return self._human

    def normalize_human(self, text: str) -> list[str]:
        """Typed text -> vocabulary words (contracts 4.4; see hivemind.lang.human)."""
        return self._human_reader().normalize(text)

    def parse_human(self, text: str, speaker: str = G.HUMAN) -> Command | Sentence | None:
        """Lenient parse of typed text into a Command or a chat Sentence, or None (contracts 4.4)."""
        return self._human_reader().read(text, speaker).content

    def read_human(self, text: str, speaker: str = G.HUMAN):
        """Like ``parse_human`` but returns a HumanMessage with the addressee and the words."""
        return self._human_reader().read(text, speaker)


def _expand(elements: Sequence[Element]) -> list[tuple[Lit | Slot | Mark, ...]]:
    """Flatten nested choices: one element tuple per intent or goal, with a Mark for the choice."""
    out: list[tuple] = [()]
    for e in elements:
        if isinstance(e, Choice):
            nxt = []
            for prefix in out:
                for head, sub in G.CHOICE_ELEMENTS[e.slot].items():
                    nxt.append((*prefix, Mark(e.slot, head), *sub))
            out = nxt
        else:
            out = [(*prefix, e) for prefix in out]
    return out


@lru_cache(maxsize=4)
def _load_cached(path: str, verify: bool) -> Language:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if verify and (vocab_hash(raw) != raw.get("hash") or words_hash(raw) != raw.get("words_hash")):
        raise ValueError(f"{path}: content does not match its hash (edited by hand? rebuild it)")
    return Language(raw, Path(path))

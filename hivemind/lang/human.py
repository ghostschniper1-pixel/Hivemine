"""Human input: the tokenizer and the lenient parser (contracts 4.4, DESIGN.md 9).

    lang.normalize_human("Can you give me 3 breads?")    # ['<unk>', '<unk>', 'give', 'me', '3', 'bread']
    lang.parse_human("can you give me 3 breads?", "human")  # Sentence('give', item='bread', item_count=3)
    lang.parse_human("red 0, mine 4 iron ore", "human")     # Command('mine', ...)
    lang.read_human("red 0, mine 4 iron ore", "human").addressee   # 'red_0'

``normalize_human(text)`` turns typed text into vocabulary words, word by word:

1. lower case; contractions (``where's`` -> ``where is``, ``i've`` -> ``i have``, ``gimme`` ->
   ``give me``); punctuation removed (a leading minus on a number is kept).
2. Numbers: digits, number words (``one`` .. ``twenty``, ``thirty-two``, ``sixty four``),
   ``a dozen`` (12), ``a couple`` (2), ``a stack`` (64), ``half a stack`` (32), ``2 stacks``
   (128). A number that is not a number token becomes the nearest one (ties down, above 64 is
   64). ``a``/``an`` become ``1`` only before an item (or a resource block: ``mine an iron
   ore``); elsewhere they are unknown words. ``1st`` .. ``16th`` and ``first``/``closest`` are
   the ordinals (``first`` and ``closest`` are ``nearest``: rank 1 by distance).
3. Synonyms, each true of what it maps to (``SYNONYMS``, ``PHRASE_SYNONYMS``): ``pick`` ->
   ``pickaxe``, ``workbench`` -> ``crafting table``, ``wood`` -> ``log`` (the log class is
   every log and six-sided wood block; only as a noun: in ``wood planks`` it stays unknown),
   ``wood``/``gold`` before a tool or armor piece -> ``wooden``/``golden``, ``cobble`` ->
   ``cobblestone``, ``yeah`` -> ``yes``, ``retreat`` -> ``flee``, ``thank you`` ->
   ``thanks``, ``run away`` -> ``flee``, ``can i have`` -> ``give me``, ``i want`` ->
   ``i need``, ``where are`` -> ``where is``, ``offhand`` -> ``off hand``, ...
   Deliberately *not* mapped, because the word alone does not say which class is meant:
   ``iron``, ``gold``, ``copper`` (raw metal, ingot, nugget or ore?), ``grass`` (grass block or
   short grass?), ``sword``, ``ingot``. A sentence that needs them fails to parse rather than
   naming the wrong thing. (Member names that are another kind's phrase, such as ``chicken``
   for raw chicken, are left out of ``human_aliases`` for the same reason.)
4. Plurals and singulars map to whichever form the vocabulary has (``sticks`` -> ``stick``,
   ``plank`` -> ``planks``, ``ores`` -> ``ore``, ``berries`` stays): every vocabulary word gets
   its regular English plural and singular forms (``variants``), kept only where they name
   exactly one vocabulary word. A vocabulary word is kept as typed, so normalizing is
   context-free and can differ from the parse: ``carrots`` is the carrot crop's phrase and stays
   ``carrots`` in ``i need 3 carrots``, which parses as the carrot item. Wherever a parse
   exists, show its words (``HumanMessage.content_words``), not the normalized ``words``.
5. Names of class members, from the vanilla ids (``spruce log`` -> ``log``, ``white wool`` ->
   ``wool``, the vocabulary's ``human_aliases``), become the class phrase when they contain a
   word outside the vocabulary.
6. Anything else is ``<unk>``.

``parse_human(text, speaker)`` reads the same tokens leniently and returns a ``Command``, a chat
``Sentence``, or None:

- An optional addressee first (``red 0, ...``, ``everyone ...``, ``team ...``); ``read_human``
  returns it with the content.
- Filler words are dropped (``please``, ``can``, ``could``, ``you``, ``pls``, ``hey``,
  ``some``, ``just``, ...); ``the``, ``a``, ``of``, ``for``, ``and`` are skipped where the
  grammar has no place for them.
- Counts snap to the nearest bin of their slot (ties down: ``mine 6`` -> 4, ``craft 5`` -> 4);
  a missing count is 1. A number outside what the slot can mean gives no reading
  (``NUMBER_RANGES``): a count, distance or wait below 1 (``i have 0 sticks`` is not
  ``i have 1 stick``), a y level outside the world; furnace and chest numbers must be exact.
  ``nearest`` and ``the`` may be left out (``attack the zombie``, ``go to iron ore``), as may
  the unit words ``blocks`` and ``seconds``, ``y`` and ``my``, and the ``i`` of ``i need``.
- Phrases match regardless of number (``3 iron ingots``, ``sweet berry``) and by member names.
- ``me`` in a player slot is the speaker (``give 3 bread to me``); in a sentence it stays
  ``me`` (``give me 3 bread``, ``follow me``), since a chat message carries its speaker.
- Verb words a person uses for the same action are accepted at the start of a command:
  ``kill``/``fight``/``hit`` (attack: it hits until the target is dead), ``make`` (craft),
  ``cook`` (smelt), ``dig``/``break``/``chop`` (mine), ``put`` (place or store), ``equip``
  (wear for armor, hold for anything else), ``walk``/``run`` (go), ``grab``/``get`` (take from
  a chest), ``run``/``escape`` (flee; ``retreat`` is a synonym of ``flee`` everywhere).
- Any word that is neither in the vocabulary, a member name, a filler nor a number makes the
  parse fail: an unknown word may change the meaning (``netherite pickaxe``).
- Readings are scored (a skipped grammar word, a left-out optional word, a verb synonym, a
  default count each cost a little) and the cheapest wins; on a tie chat sentences come
  before commands. ``where is the iron ore`` gives None: ``where is {item}`` takes items, and
  ``iron ore`` is a block, not an item (mapping it to ``raw iron`` would be false).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hivemind.lang import grammar as G
from hivemind.lang.grammar import Lit, Slot
from hivemind.lang.language import FlatTemplate, Language, Mark, _expand, assemble
from hivemind.lang.types import Command, EntityRef, Sentence

# --- tables -------------------------------------------------------------------------------

CONTRACTIONS = (  # applied to the lower-cased text, in order
    (r"\bgimme\b", "give me"), (r"\blemme\b", "let me"), (r"\bwanna\b", "want to"),
    (r"\b(where|what|it|that|there|here|who)'s\b", r"\1 is"), (r"\bi'm\b", "i am"), (r"\bi've\b", "i have"),
    (r"\bi'd\b", "i would"), (r"\bi'll\b", "i will"), (r"\bcan't\b", "can not"), (r"\bwon't\b", "will not"),
    (r"n't\b", " not"), (r"'re\b", " are"), (r"'ll\b", " will"), (r"'ve\b", " have"), (r"'s\b", ""),
)
UNITS = {w: i for i, w in enumerate((
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"))}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
ORDINAL_DIGITS = re.compile(r"^(\d+)(st|nd|rd|th)$")
ORDINAL_SYNONYMS = {"first": "nearest", "closest": "nearest"}  # rank 1 by distance is the nearest

# Multi-word rewrites (longest first), before single words. Each is true of what it maps to.
PHRASE_SYNONYMS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = tuple(sorted((
    (("work", "bench"), ("crafting", "table")),
    (("crafting", "bench"), ("crafting", "table")),
    (("thank", "you"), ("thanks",)),
    (("thank", "u"), ("thanks",)),
    (("run", "away"), ("flee",)),
    (("come", "over", "here"), ("come", "here")),
    (("over", "here"), ("here",)),
    (("come", "to", "me"), ("come", "here")),
    (("can", "i", "have"), ("give", "me")),
    (("could", "i", "have"), ("give", "me")),
    (("may", "i", "have"), ("give", "me")),
    (("can", "i", "get"), ("give", "me")),
    (("could", "i", "get"), ("give", "me")),
    (("i", "want"), ("i", "need")),
    (("i", "would", "like"), ("i", "need")),
    (("i", "got"), ("i", "have")),
    (("where", "are"), ("where", "is")),
    (("y", "level"), ("y",)),
    (("every", "one"), ("everyone",)),
    (("all", "of", "you"), ("everyone",)),
    (("go", "to", "sleep"), ("sleep",)),
    (("put", "on"), ("wear",)),
    (("in", "front", "of", "me"), ("in", "front")),
    (("in", "front", "of", "you"), ("in", "front")),
    (("below", "me"), ("under", "me")),
    (("beneath", "me"), ("under", "me")),
    (("north", "east"), ("northeast",)),
    (("north", "west"), ("northwest",)),
    (("south", "east"), ("southeast",)),
    (("south", "west"), ("southwest",)),
    (("pick", "up"), ("pick_up",)),  # picking up is automatic: no command; stays unknown (not "pickaxe up")
), key=lambda kv: -len(kv[0])))

SYNONYMS: dict[str, tuple[str, ...]] = {  # single words not in the vocabulary -> vocabulary words
    # (checked before member-name words and verb alternatives, so each entry takes effect)
    "pick": ("pickaxe",), "pickax": ("pickaxe",),
    "workbench": ("crafting", "table"),
    "wood": ("log",),  # the log class is every log and six-sided wood block (a noun only: NOUN_SYNONYMS)
    "cobble": ("cobblestone",),
    "bonemeal": ("bone", "meal"),
    "offhand": ("off", "hand"),
    "everybody": ("everyone",), "everyones": ("everyone",),
    "yeah": ("yes",), "yep": ("yes",), "yup": ("yes",), "yea": ("yes",), "sure": ("yes",),
    "nope": ("no",), "nah": ("no",),
    "thx": ("thanks",), "ty": ("thanks",), "thanx": ("thanks",), "thnx": ("thanks",), "tysm": ("thanks",),
    "retreat": ("flee",),
    "are": ("is",),  # only "is" is in the grammar ("where is")
    **{k: (v,) for k, v in ORDINAL_SYNONYMS.items()},
}
# Synonyms of a noun that do not hold for the same word as a modifier: "wood" is logs, but in
# "wood planks" or "wood door" it says what the thing is made of. Before the last word of a
# class phrase or member name, such a word is left as it is (an unknown or member-name word).
NOUN_SYNONYMS = frozenset({"wood"})
TOOL_ARMOR_WORDS = frozenset({"pickaxe", "pick", "axe", "shovel", "hoe", "sword", "helmet", "chestplate", "leggings",
                              "boots", "pickaxes", "axes", "shovels", "hoes", "swords", "picks"})
MATERIAL_ADJECTIVES = {"wood": "wooden", "gold": "golden"}  # before a tool or armor word

FILLERS = frozenset((
    "please", "pls", "plz", "plox", "can", "could", "would", "will", "you", "u", "ya", "hey", "hi", "hello", "yo",
    "some", "just", "kindly", "ok", "okay", "now", "any", "maybe", "also", "quickly", "actually", "do", "does"))
SKIPPABLE = {"the": 0.1, "of": 0.1, "for": 0.1, "and": 0.1, "a": 0.1, "an": 0.1}  # vocabulary words with no place
OPTIONAL_LITERALS = {"the", "nearest", "my", "blocks", "seconds", "y", "down"}  # may be left out of commands
SENTENCE_OPTIONAL = {"need": {"i"}, "come": {"here"}, "follow": {"me"}, "help": {"me"}}  # "need 3 bread"
VERB_ALTERNATIVES: dict[str, tuple[str, ...]] = {  # first word of a template -> words people use for it
    "go": ("walk", "run", "head"),
    "attack": ("kill", "fight", "hit"),
    "flee": ("run", "escape"),  # ("retreat" is a synonym: SYNONYMS)
    "mine": ("dig", "break", "chop"),
    "craft": ("make",),
    "smelt": ("cook",),
    "place": ("put",),
    "store": ("put", "deposit"),
    "take": ("grab", "get"),
    "hold": ("equip",),  # "equip" a tool or weapon: hold it ...
    "wear": ("equip",),  # ... and a piece of armor: wear it (checked against the item's equipment slot)
    "eat": ("consume",),
}
VERB_ALTERNATIVE_WORDS = frozenset(w for ws in VERB_ALTERNATIVES.values() for w in ws)
COST_OPTIONAL = 0.5
COST_VERB_ALT = 0.2
COST_DEFAULT_COUNT = 1.0
COST_ALIAS = 0.1
COST_VARIANT_TEMPLATE = 0.5
ARMOR_SLOTS = frozenset({"head", "chest", "legs", "feet"})
DEFAULT_COUNT_SLOTS = ("item_count", "mine_count")  # a missing count is 1 ("craft a torch", "mine iron ore")
# Typed numbers a number slot can read, as (lowest, highest or None); a number outside gives no
# reading, so "i have 0 sticks" fails to parse instead of becoming "i have 1 stick". Inside the
# range a number snaps to the slot's nearest bin. Furnace and chest numbers must be exact.
NUMBER_RANGES: dict[str, tuple[int, int | None]] = {
    "item_count": (1, None), "mine_count": (1, None), "wait_seconds": (1, None),
    "explore_dist": (1, None), "strip_dist": (1, None),
    "y_band": (-64, 319),  # [mem] the 1.21.1 overworld build range, y -64..319
}

# Human-only word orders (cost COST_VARIANT_TEMPLATE): "give red 2 3 bread".
VARIANT_TEMPLATES = {("give", "give"): ("give {player} {item_count} {item}",)}


@dataclass(frozen=True)
class Tok:
    """A token of typed text: ``word`` is the vocabulary form where there is one."""

    raw: str
    word: str
    kind: str  # "word" (vocabulary), "alias" (only in member names), "num", "article", "filler", "unknown"
    num: int | None = None


@dataclass(frozen=True)
class HumanMessage:
    """What a person typed, read: who to (None if not said), the parsed content (None if it
    could not be read), the normalized words, and the canonical words of the content.

    ``words`` is context-free (step 4 of the module docstring), so it can differ from the
    parse: show ``content_words`` wherever it is not None, and ``words`` only for text that did
    not parse."""

    speaker: str
    addressee: str | None
    content: Command | Sentence | None
    words: tuple[str, ...]
    content_words: tuple[str, ...] | None = None  # lang.to_words(content) ('.' ends a command)


# --- English word forms ---------------------------------------------------------------------

_IRREGULAR = {"foot": "feet", "man": "men", "mouse": "mice", "tooth": "teeth", "goose": "geese", "cactus": "cacti",
              "child": "children", "person": "people"}
_VOWELS = set("aeiou")


def plurals(w: str) -> set[str]:
    """Regular (and a few irregular) plural forms of an English noun."""
    out = set()
    if w in _IRREGULAR:
        out.add(_IRREGULAR[w])
    if w.endswith("man"):
        out.add(w[:-3] + "men")
    if w.endswith(("s", "x", "z", "ch", "sh")):
        out.add(w + "es")
    elif w.endswith("y") and len(w) > 1 and w[-2] not in _VOWELS:
        out.add(w[:-1] + "ies")
    elif w.endswith("fe"):
        out |= {w[:-2] + "ves", w + "s"}
    elif w.endswith("f") and not w.endswith("ff"):
        out |= {w[:-1] + "ves", w + "s"}
    elif w.endswith("o"):
        out |= {w + "es", w + "s"}
    else:
        out.add(w + "s")
    return out


def singulars(w: str) -> set[str]:
    """Regular (and a few irregular) singular forms of an English plural."""
    out = {k for k, v in _IRREGULAR.items() if v == w}
    if w.endswith("men"):
        out.add(w[:-3] + "man")
    if w.endswith("ies"):
        out.add(w[:-3] + "y")
    elif w.endswith("ves"):
        out |= {w[:-3] + "f", w[:-3] + "fe"}
    elif w.endswith(("oes", "ches", "shes", "xes", "sses", "zes")):
        out.add(w[:-2])
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        out.add(w[:-1])
    return out


# --- the reader ---------------------------------------------------------------------------


class HumanReader:
    """Tokenizer and lenient parser bound to one Language (see the module docstring)."""

    def __init__(self, lang: Language):
        self.lang = lang
        vocab = {w for w in lang.words[len(G.RESERVED):]}
        self.vocab = vocab
        # words that occur only in member names (spruce, stripped, white, ...)
        alias_words = {w for table in lang.aliases.values() for a in table for w in a.split()}
        self.alias_words = alias_words - vocab
        known = vocab | self.alias_words
        # plural/singular variants -> the known word; groups of known words that are variants of each other
        found: dict[str, set[str]] = {}
        parent: dict[str, str] = {w: w for w in known}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for w in sorted(known):
            if len(w) < 3 or not w.isalpha():
                continue
            for f in plurals(w) | singulars(w):
                if f in known:
                    a, b = find(w), find(f)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
                else:
                    found.setdefault(f, set()).add(w)
        self.variants: dict[str, str] = {f: next(iter(ws)) for f, ws in found.items() if len(ws) == 1}
        self.key_of: dict[str, str] = {w: find(w) for w in known}
        # the last words of class phrases and member names: nouns a NOUN_SYNONYMS word can modify
        names = [r["phrase"] for r in lang.raw["concepts"] if r["kind"] in ("block", "item", "entity")]
        names += [a.split() for table in lang.aliases.values() for a in table]
        self._noun_ends = {self.key(ws[-1]) for ws in names if ws}
        self._indexes: dict[str, dict[tuple[str, ...], list[tuple[Any, bool]]]] = {}
        self._norm_aliases = self._context_free_aliases()
        self._paths: list[tuple[float, str, FlatTemplate]] | None = None
        self._count_first_keys: set[str] | None = None

    # -- keys and phrase indexes ------------------------------------------------------------

    def key(self, word: str) -> str:
        return self.key_of.get(word, word)

    def keys(self, words: Sequence[str]) -> tuple[str, ...]:
        return tuple(self.key(w) for w in words)

    def index(self, domain: str) -> dict[tuple[str, ...], list[tuple[Any, bool]]]:
        """Number-insensitive phrase index of a domain: key tuple -> [(value, is_alias)]."""
        idx = self._indexes.get(domain)
        if idx is None:
            idx = {}
            values = self.lang.domains[domain] if domain != "entity_target" else None
            if domain == "entity_target":
                ent = self.lang.raw["domains"]["entity_ref"]
                pairs = [(e, self.lang.phrase("entity", e)) for e in ent["entities"]]
                pairs += [((G.DROPPED, i), (G.DROPPED, *self.lang.phrase("item", i))) for i in ent["dropped"]]
                pairs += [((G.DROPPED, i), (G.DROPPED, *a.split()))
                          for a, i in self.lang.aliases.get("item", {}).items()]
                alias_src = "entity_class"
            else:
                pairs = [(v, self.lang.value_words(domain, v)) for v in values]
                alias_src = {"block": "block", "item": "item", "chat_item": "item", "entity_class": "entity_class"}.get(domain)
            for v, ph in pairs:
                idx.setdefault(self.keys(ph), []).append((v, False))
            if alias_src:
                canonical = set(idx)
                for a, name in self.lang.aliases.get(alias_src, {}).items():
                    k = self.keys(a.split())
                    if k not in canonical:
                        idx.setdefault(k, []).append((name, True))
            self._indexes[domain] = idx
        return idx

    def _context_free_aliases(self) -> dict[tuple[str, ...], tuple[str, ...]]:
        """Member names with a word outside the vocabulary -> the class phrase, where every
        domain that knows the name agrees on the phrase (for normalize)."""
        out: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
        kinds = {"block": "block", "item": "item", "entity_class": "entity"}
        for dom, table in self.lang.aliases.items():
            for a, name in table.items():
                ws = tuple(a.split())
                if any(w not in self.vocab for w in ws):
                    out.setdefault(ws, set()).add(self.lang.phrase(kinds[dom], name))
        return {k: next(iter(v)) for k, v in out.items() if len(v) == 1}

    # -- tokenizing -------------------------------------------------------------------------

    def _raw_tokens(self, text: str) -> list[str]:
        t = text.lower().replace("’", "'").replace("`", "'")
        for pat, rep in CONTRACTIONS:
            t = re.sub(pat, rep, t)
        out: list[str] = []
        for m in re.finditer(r"-?\d+(?:st|nd|rd|th|x)?|x\d+|[a-z]+(?:[-'][a-z]+)*", t):
            s = m.group(0)
            if "-" in s and not s.lstrip("-").isdigit() and not s[0].isdigit():
                parts = s.split("-")
                if all(p in UNITS or p in TENS for p in parts):
                    out.extend(parts)
                else:
                    out.extend(p for p in parts if p)
            else:
                out.append(s.replace("'", ""))
        return out

    def tokens(self, text: str) -> list[Tok]:
        """Typed text -> tokens (steps 1-4 of the module docstring; aliases untouched)."""
        raw = self._raw_tokens(text)
        # numbers
        toks: list[Tok] = []
        i = 0
        while i < len(raw):
            w = raw[i]
            nxt = raw[i + 1] if i + 1 < len(raw) else None
            m = ORDINAL_DIGITS.match(w)
            if m:
                k = int(m.group(1))
                word = G.ORDINALS[k - 1] if 1 <= k <= len(G.ORDINALS) else w
                toks.append(Tok(w, word, "word" if word in self.vocab else "unknown"))
                i += 1
                continue
            if re.fullmatch(r"-?\d+x|x\d+", w):
                toks.append(Tok(w, w.strip("x"), "num", int(w.strip("x"))))
                i += 1
                continue
            if re.fullmatch(r"-?\d+", w):
                n = int(w)
                if nxt in ("stack", "stacks"):
                    toks.append(Tok(f"{w} {nxt}", str(n * 64), "num", n * 64))
                    i += 2
                    continue
                toks.append(Tok(w, w, "num", n))
                i += 1
                continue
            if w == "half" and raw[i + 1:i + 3] == ["a", "stack"]:
                toks.append(Tok("half a stack", "32", "num", 32))
                i += 3
                continue
            if w in ("a", "an") and nxt in ("dozen", "couple", "stack"):
                n = {"dozen": 12, "couple": 2, "stack": 64}[nxt]
                skip = 3 if nxt == "couple" and raw[i + 2:i + 3] == ["of"] else 2
                toks.append(Tok(f"{w} {nxt}", str(n), "num", n))
                i += skip
                continue
            if w == "dozen":
                toks.append(Tok(w, "12", "num", 12))
                i += 1
                continue
            if w in UNITS or w in TENS:
                n, j = self._number_words(raw, i)
                toks.append(Tok(" ".join(raw[i:j]), str(n), "num", n))
                i = j
                continue
            toks.append(Tok(w, w, "word"))
            i += 1
        # phrase synonyms
        toks = self._rewrite_phrases(toks)
        # single words
        out: list[Tok] = []
        for k, t in enumerate(toks):
            if t.kind != "word":
                out.append(t)
                continue
            w = t.word
            nxt = toks[k + 1].word if k + 1 < len(toks) else None
            if w in MATERIAL_ADJECTIVES and nxt in TOOL_ARMOR_WORDS:
                out.append(Tok(t.raw, MATERIAL_ADJECTIVES[w], "word"))
                continue
            if w in ("a", "an"):
                out.append(Tok(t.raw, w, "article", 1))
                continue
            if w in self.vocab:
                out.append(t)
                continue
            syn = self._synonym(w, nxt)
            if syn is not None:
                out.extend(Tok(t.raw, s, "word") for s in syn)
                continue
            if w in FILLERS:
                out.append(Tok(t.raw, w, "filler"))
                continue
            if w in VERB_ALTERNATIVE_WORDS:  # known to the parser (first word of a command), <unk> in normalize
                out.append(t)
                continue
            if w in self.alias_words:
                out.append(Tok(t.raw, w, "alias"))
                continue
            v = self.variants.get(w)
            if v is not None:
                out.append(Tok(t.raw, v, "alias" if v in self.alias_words else "word"))
                continue
            out.append(Tok(t.raw, w, "unknown"))
        return out

    def _synonym(self, w: str, nxt: str | None) -> tuple[str, ...] | None:
        """The vocabulary words ``SYNONYMS`` gives for ``w`` (or its singular), if any."""
        for s in (w, *sorted(singulars(w))):
            syn = SYNONYMS.get(s)
            if syn is not None:
                modifies = nxt is not None and self.key(self.variants.get(nxt, nxt)) in self._noun_ends
                if s in NOUN_SYNONYMS and modifies:
                    return None  # a modifier here ("wood planks"), not the noun
                return syn
        return None

    @staticmethod
    def _number_words(raw: list[str], i: int) -> tuple[int, int]:
        n = 0
        j = i
        if raw[j] in TENS:
            n = TENS[raw[j]]
            j += 1
            if j < len(raw) and raw[j] in UNITS and 0 < UNITS[raw[j]] < 10:
                n += UNITS[raw[j]]
                j += 1
        else:
            n = UNITS[raw[j]]
            j += 1
        if j < len(raw) and raw[j] == "hundred":
            n *= 100
            j += 1
        return n, j

    def _rewrite_phrases(self, toks: list[Tok]) -> list[Tok]:
        out: list[Tok] = []
        i = 0
        def is_word(t: Tok, s: str) -> bool:  # a number word is matched as typed ("every one")
            return (t.kind == "word" and t.word == s) or (t.kind == "num" and t.raw == s)

        while i < len(toks):
            for src, dst in PHRASE_SYNONYMS:
                if i + len(src) <= len(toks) and all(is_word(toks[i + k], s) for k, s in enumerate(src)):
                    raw = " ".join(t.raw for t in toks[i:i + len(src)])
                    out.extend(Tok(raw, d, "word") for d in dst)
                    i += len(src)
                    break
            else:
                out.append(toks[i])
                i += 1
        return out

    # -- normalize --------------------------------------------------------------------------

    def _starts_count_phrase(self, toks: Sequence[Tok], i: int) -> bool:
        """Does a v1 item or resource block phrase (or member name) start at token i?"""
        while i < len(toks) and toks[i].kind == "filler":
            i += 1
        if i >= len(toks) or toks[i].kind not in ("word", "alias"):
            return False
        if self._count_first_keys is None:
            self._count_first_keys = {ph[0] for dom in ("item", "block") for ph in self.index(dom) if ph}
        return self.key(toks[i].word) in self._count_first_keys

    def normalize(self, text: str) -> list[str]:
        """Typed text -> vocabulary words (``<unk>`` for anything else)."""
        toks = self.tokens(text)
        out: list[str] = []
        i = 0
        while i < len(toks):
            t = toks[i]
            if t.kind in ("word", "alias"):
                hit = None
                for n in range(min(6, len(toks) - i), 0, -1):
                    span = toks[i:i + n]
                    if all(s.kind in ("word", "alias") for s in span):
                        ph = self._norm_aliases.get(tuple(s.word for s in span))
                        if ph is not None:
                            hit = (n, ph)
                            break
                if hit:
                    out.extend(hit[1])
                    i += hit[0]
                    continue
                out.append(t.word if t.word in self.vocab else G.UNK)
            elif t.kind == "num":
                out.append(str(_snap_number(t.num)))
            elif t.kind == "article":
                out.append("1" if self._starts_count_phrase(toks, i + 1) else G.UNK)
            else:
                out.append(G.UNK)
            i += 1
        return out

    # -- lenient parsing --------------------------------------------------------------------

    def _templates(self) -> list[tuple[float, str, FlatTemplate]]:
        """(base cost, kind, flat template) for every chat sentence and command (sentences first)."""
        if self._paths is None:
            paths: list[tuple[float, str, FlatTemplate]] = []
            for intent, els in G.SENTENCE_ELEMENTS.items():
                paths.append((0.0, "sentence", FlatTemplate(None, intent, tuple(els))))
            for verb, templates in G.COMMAND_ELEMENTS.items():
                for tid, els in templates.items():
                    head = (Mark("template", tid),) if G.multi_template(verb) else ()
                    texts = [(0.0, els)]
                    for v in VARIANT_TEMPLATES.get((verb, tid), ()):
                        texts.append((COST_VARIANT_TEMPLATE, G.compile_template(v)))
                    for base, e in texts:
                        for flat in _expand(e):
                            paths.append((base, "command", FlatTemplate(verb, tid, (*head, *flat))))
            self._paths = paths
        return self._paths

    def _match(self, path: FlatTemplate, toks: Sequence[Tok], speaker: str | None) -> list[tuple[float, tuple]]:
        """Every lenient reading of all of ``toks`` by ``path``: (cost, chosen pairs)."""
        els = path.elements
        n = len(toks)
        out: list[tuple[float, tuple]] = []
        first_lit = next((i for i, e in enumerate(els) if isinstance(e, Lit)), None)
        intent = path.head if path.verb is None else None

        def rec(ei: int, ti: int, cost: float, pairs: tuple) -> None:
            if cost > 3.0:
                return
            if ti < n and toks[ti].word in SKIPPABLE and toks[ti].kind in ("word", "article"):
                rec(ei, ti + 1, cost + SKIPPABLE[toks[ti].word], pairs)
            if ei == len(els):
                if ti == n:
                    out.append((cost, pairs))
                return
            e = els[ei]
            if isinstance(e, Mark):
                rec(ei + 1, ti, cost, (*pairs, (e.slot, e.value)))
                return
            if isinstance(e, Lit):
                key = self.key(e.word)
                if ti < n and toks[ti].kind == "word" and self.key(toks[ti].word) == key:
                    rec(ei + 1, ti + 1, cost, pairs)
                if ei == first_lit and path.verb is not None and ti < n and toks[ti].kind == "word" \
                        and toks[ti].word in VERB_ALTERNATIVES.get(e.word, ()):
                    rec(ei + 1, ti + 1, cost + COST_VERB_ALT, (*pairs, ("_alt", toks[ti].word)))
                # A command's first word picks its verb and is never left out; a sentence's
                # optional words are listed one by one (the "i" of "i need" included).
                optional = SENTENCE_OPTIONAL.get(intent, set()) if intent else OPTIONAL_LITERALS
                if e.word in optional and (intent is not None or ei != first_lit):
                    rec(ei + 1, ti, cost + COST_OPTIONAL, pairs)
                return
            for value, used, c in self._slot_readings(e, toks, ti, speaker):
                rec(ei + 1, ti + used, cost + c, (*pairs, (e.name, value)))

        rec(0, 0, 0.0, ())
        return out

    def _slot_readings(self, slot: Slot, toks: Sequence[Tok], ti: int, speaker: str | None):
        """(value, tokens used, cost) for each way the slot can read tokens from ti."""
        dom = slot.domain
        n = len(toks)
        t = toks[ti] if ti < n else None
        if dom in G.INT_DOMAINS:
            bins = G.INT_DOMAINS[dom]
            if t is not None and t.kind == "num":
                rng = NUMBER_RANGES.get(dom)
                if rng is None:  # furnace and chest numbers: exact
                    if t.num in bins:
                        yield t.num, 1, 0.0
                elif t.num >= rng[0] and (rng[1] is None or t.num <= rng[1]):
                    yield G.snap(t.num, bins), 1, 0.0
            if t is not None and t.kind == "article" and dom in DEFAULT_COUNT_SLOTS + ("wait_seconds",):
                yield G.snap(1, bins), 1, 0.0
            if dom in DEFAULT_COUNT_SLOTS:
                yield 1, 0, COST_DEFAULT_COUNT
            return
        if dom in ("player", "addressee") and t is not None and t.kind == "word" and t.word == "me" \
                and speaker is not None:
            yield speaker, 1, 0.0
        if dom == "entity_ref":
            starts = [(1, ti, 0.0)]
            if t is not None and t.kind == "word" and t.word in G.ORDINALS:
                starts.append((G.ORDINALS.index(t.word) + 1, ti + 1, 0.0))
            for k, j, c in starts:
                for v, used, c2 in self._phrase_readings("entity_target", toks, j):
                    yield EntityRef(k, v), (j - ti) + used, c + c2
            return
        yield from self._phrase_readings(dom, toks, ti)

    def _phrase_readings(self, dom: str, toks: Sequence[Tok], ti: int):
        idx = self.index(dom)
        span: list[str] = []
        for j in range(ti, min(len(toks), ti + 7)):
            t = toks[j]
            if t.kind not in ("word", "alias", "num"):
                break
            span.append(self.key(t.word))
            for v, alias in idx.get(tuple(span), ()):
                yield v, j - ti + 1, COST_ALIAS if alias else 0.0

    def read(self, text: str, speaker: str = G.HUMAN) -> HumanMessage:
        """Read typed text (see the module docstring)."""
        speaker_name = self._player_name(speaker)
        words = tuple(self.normalize(text))
        all_toks = self.tokens(text)
        toks = [t for t in all_toks if t.kind != "filler"]
        best: tuple[float, int, Any] | None = None
        best_addr = None
        unknown = any(t.kind == "unknown" for t in all_toks)  # an unknown word may change the meaning
        for addr, rest, addr_cost in ([] if unknown else self._addressee_splits(toks, speaker_name)):
            for order, (base, kind, path) in enumerate(self._templates()):
                for cost, pairs in self._match(path, rest, speaker_name):
                    total = base + cost + addr_cost
                    alt = dict(pairs).get("_alt")
                    pairs = tuple(p for p in pairs if p[0] != "_alt")
                    if alt == "equip" and (dict(pairs).get("template") == "wear") != self._is_armor(dict(pairs)["item"]):
                        continue  # "equip" means wear for armor and hold for anything else
                    if best is None or (total, order) < best[:2]:
                        content = (Sentence(path.head, **dict(pairs)) if kind == "sentence"
                                   else assemble(path.verb, pairs))  # type: ignore[arg-type]
                        best = (total, order, content)
                        best_addr = addr
        if best is None:
            prefix = [v for v, used, _ in self._phrase_readings("addressee", toks, 0) if used < len(toks)]
            return HumanMessage(speaker_name or speaker, prefix[-1] if prefix else None, None, words)
        content = best[2]
        return HumanMessage(speaker_name or speaker, best_addr, content, words, tuple(self.lang.to_words(content)))

    def _is_armor(self, item: str) -> bool:
        return self.lang.row("item", item).get("equip_slot") in ARMOR_SLOTS

    def _addressee_splits(self, toks: list[Tok], speaker: str | None):
        """(addressee or None, remaining tokens, cost): the whole text, and every addressee prefix."""
        yield None, toks, 0.0
        for v, used, c in self._phrase_readings("addressee", toks, 0):
            if used < len(toks):
                yield v, toks[used:], c

    def _player_name(self, speaker: str) -> str | None:
        name = speaker.replace(" ", "_")
        return name if name in self.lang.domains["player"] else None


def _snap_number(n: int) -> int:
    """Any integer -> the nearest number token (ties down; above 64 is 64; below -54 is -54)."""
    if n > 64:
        return 64
    return G.snap(max(n, -54), G.NUMBER_TOKENS)

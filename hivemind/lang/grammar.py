"""The static English grammar: verbs, templates, sentences, goals, results and events (contracts 4.3).

Everything here is fixed text, independent of the class tables; the concept phrases that fill
the slots come from ``data/concepts-1.21.1.json`` through ``hivemind.lang.vocab``.

Templates are written exactly as in docs/contracts.md 4.3. Slots are in braces; every other
word is literal. Two slot spellings are special:

- ``{ordinal} {entity}`` is one slot ``entity`` whose value is an ``EntityRef`` (contracts 4.5
  has no separate ``ordinal`` slot): ``the second zombie`` is ``EntityRef(2, "zombie")``.
- ``{sentence}`` in ``tell {addressee} {sentence}`` and ``{goal}`` in ``plan to {goal}`` are
  choices among the chat sentences and goals below, expanded inline: the decoder asks for the
  ``intent`` or ``goal`` slot and then for that sentence's or goal's own slots.

Which values fill a slot is the slot's *domain* (``SLOT_DOMAINS``). A slot name can mean
different domains in different places: ``item`` is a v1 item class in commands and goals, but
also a nonce word in chat sentences (``chat_item``); ``entity`` is an ``EntityRef`` in commands
but an entity class in the goal ``kill {mine_count} {entity}`` and in event sentences.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tokens and fixed word lists
# ---------------------------------------------------------------------------

PAD, UNK, BOS, END = "<pad>", "<unk>", "<bos>", "."
RESERVED = (PAD, UNK, BOS, END)  # contracts 4.1: ids 0, 1, 2, 3
PAD_ID, UNK_ID, BOS_ID, END_ID = 0, 1, 2, 3

NUMBER_TOKENS: tuple[int, ...] = (*range(21), 32, 48, 64, -16, -32, -54)  # contracts 4.1
COUNT_TOKENS: tuple[int, ...] = (*range(21), 32, 48, 64)  # the non-negative ones, for counts (contracts 4.1)
NONCE_WORDS = ("dax", "wug", "blicket", "fep", "toma", "zib", "kiki", "bouba")  # contracts 4.1, DESIGN.md 9

TEAMS = ("red", "blue")  # contracts 4.2: team words (configurable: changing them rebuilds the vocabulary)
TEAM_SIZE = 20  # contracts 4.2: red 0 .. red 19
HUMAN = "human"  # contracts 4.2: the owner's player name (also a relation word)
LLM = "llm"  # contracts 4.2: a coupled LLM as a goal setter
SETTERS = ("me", LLM)  # DESIGN.md 6.2: who set a goal (besides player names and human)
ADDRESSEE_GROUPS = ("everyone", "team")  # contracts 4.3: tell {addressee}

ORDINALS = ("nearest", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
            "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth")  # contracts 4.3
DIR8 = ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")  # contracts 4.3
DIR4 = ("north", "east", "south", "west")  # contracts 4.3
PATTERNS: dict[str, tuple[str, ...]] = {  # contracts 4.3: place {item} {pattern}
    "here": ("here",),
    "in_front": ("in", "front"),
    "under_me": ("under", "me"),  # pillar up
    "around_me": ("around", "me"),  # enclose
    "for_light": ("for", "light"),
}
EXPLORE_DISTS = (16, 32, 64)  # contracts 4.3
STRIP_DISTS = (16, 32)  # contracts 4.3
Y_BANDS = (48, 16, 0, -16, -32, -54)  # contracts 4.3
MINE_COUNTS = (1, 2, 4, 8, 16)  # contracts 4.3
ITEM_COUNTS = (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)  # contracts 4.3
WAIT_SECONDS = (1, 5, 20)  # contracts 4.3: 20, 100, 400 ticks
FURNACES = (1, 2, 3, 4)  # contracts 4.3: the agent's own furnaces in placement order
CHESTS = (1, 2, 3, 4, 5, 6, 7, 8)  # contracts 4.3: the agent's own chests

STATIONS = ("crafting_table", "furnace", "bed")  # contracts 4.3: block-class names, named by their phrases
CONTAINERS = ("inventory", "chest")  # contracts 4.3: {container} full
CAUSE_WORDS = ("damage", "message")  # contracts 4.3: interrupted by {cause}; or a hostile's entity phrase
RELATIONS = ("teammate", "enemy", "human", "hostile", "passive", "object")  # contracts 4.2, DESIGN.md 6.2
PLACE_KINDS = ("resource", "station", "danger", "home")  # contracts 4.2, 6.1
EQUIP_SLOTS: dict[str, tuple[str, ...]] = {  # contracts 4.2: equipment slots (the game's names for them)
    "mainhand": ("main", "hand"),
    "offhand": ("off", "hand"),
    "head": ("head",),
    "chest": ("chest",),
    "legs": ("legs",),
    "feet": ("feet",),
}

# ---------------------------------------------------------------------------
# Templates (contracts 4.3)
# ---------------------------------------------------------------------------

# Verb ids in id order: the design's 17 verbs keep ids 0-16, then store, take, plan.
COMMANDS: dict[str, dict[str, str]] = {
    "goto": {
        "block": "go to the nearest {block}",
        "entity": "go to the {ordinal} {entity}",
        "player": "go to {player}",
        "furnace": "go to my furnace {furnace}",
        "chest": "go to my chest {chest}",
        "bed": "go to my bed",
        "home": "go home",
    },
    "explore": {"explore": "explore {dir8} {explore_dist} blocks"},
    "descend": {"descend": "dig down to y {y_band}"},
    "strip_mine": {"strip_mine": "strip mine {dir4} {strip_dist} blocks"},
    "mine": {"mine": "mine {mine_count} {block}"},
    "craft": {"craft": "craft {item_count} {item}"},
    "smelt_start": {"smelt_start": "smelt {item_count} {item}"},
    "collect": {"collect": "collect from my furnace {furnace}"},
    "place": {"place": "place {item} {pattern}"},
    "attack": {"entity": "attack the {ordinal} {entity}", "player": "attack {player}"},
    "flee": {"entity": "flee from the {ordinal} {entity}", "player": "flee from {player}"},
    "eat": {"eat": "eat {item}"},
    "equip": {"hold": "hold {item}", "off_hand": "hold {item} in off hand", "wear": "wear {item}"},
    "sleep": {"sleep": "sleep"},
    "give": {"give": "give {item_count} {item} to {player}"},
    "chat": {"chat": "tell {addressee} {sentence}"},
    "wait": {"wait": "wait {wait_seconds} seconds"},
    "store": {"store": "store {item_count} {item} in my chest {chest}"},
    "take": {"take": "take {item_count} {item} from my chest {chest}"},
    "plan": {"plan": "plan to {goal}"},
}
VERBS: tuple[str, ...] = tuple(COMMANDS)
VERB_ID: dict[str, int] = {v: i for i, v in enumerate(VERBS)}
VERB_FLAG: dict[str, str] = {"store": "containers", "take": "containers", "plan": "goals"}  # contracts 4.3

SENTENCES: dict[str, str] = {  # contracts 4.3: the 14 chat intents (DESIGN.md 9)
    "need": "i need {item_count} {item}",
    "have": "i have {item_count} {item}",
    "give": "give me {item_count} {item}",
    "where": "where is {item}",
    "come": "come here",
    "follow": "follow me",
    "help": "help me",
    "attack": "attack",
    "flee": "flee",
    "stop": "stop",
    "wait": "wait",
    "yes": "yes",
    "no": "no",
    "thanks": "thanks",
}
INTENTS: tuple[str, ...] = tuple(SENTENCES)

GOALS: dict[str, str] = {  # contracts 4.3, DESIGN.md 6.6
    "get": "get {item_count} {item}",
    "kill": "kill {mine_count} {entity}",
    "home": "go home",
    "to_player": "go to {player}",
    "to_block": "go to the nearest {block}",
    "explore": "explore {dir8}",
    "survive": "survive until morning",
    "help": "help {player}",
    "protect": "protect {player}",
}
GOAL_KINDS: tuple[str, ...] = tuple(GOALS)

RESULTS: dict[str, str] = {  # DESIGN.md 5 (authoritative), contracts 4.3
    "OK": "done",
    "NO_TARGET": "no target",
    "NO_PATH": "no path",
    "PATH_TIMEOUT": "path too slow",
    "STUCK": "stuck",
    "DIG_FAILED": "dig failed",
    "MISSING_INGREDIENT": "missing ingredient",
    "NO_STATION": "no {station} nearby",
    "INV_FULL": "{container} full",
    "TARGET_LOST": "target lost",
    "HAZARD": "danger ahead",
    "INTERRUPTED": "interrupted by {cause}",
    "DIED": "died",
    "TIMEOUT": "too slow",
}
RESULT_CODES: tuple[str, ...] = tuple(RESULTS)

EVENTS: dict[str, str] = {  # contracts 4.3, DESIGN.md 6.3
    "player_attacked_me": "{player} attacked me",
    "entity_attacked_me": "{entity} attacked me",
    "i_attacked_player": "i attacked {player}",
    "player_killed_me": "{player} killed me",
    "entity_killed_me": "{entity} killed me",
    "i_killed_player": "i killed {player}",
    "i_killed_entity": "i killed {entity}",
    "i_died": "i died",
    "player_gave_me": "{player} gave me {n} {item}",
    "i_gave": "i gave {n} {item} to {player}",
    "player_opened_chest": "{player} opened my chest {chest}",
    "someone_opened_chest": "someone opened my chest {chest}",
    "missing_from_chest": "{n} {item} missing from my chest {chest}",
    "i_found": "i found {block}",
}
EVENT_KINDS: tuple[str, ...] = tuple(EVENTS)

# Entity phrases (contracts 4.3): an entity class, or "dropped {item}" for an item on the ground.
DROPPED = "dropped"

# ---------------------------------------------------------------------------
# Slot domains
# ---------------------------------------------------------------------------

# Domain of each slot name in commands; CONTEXT_DOMAINS overrides it inside sentences, goals,
# events and results. Domains name value sets built by hivemind.lang.vocab:
#   block         resource block classes (curated; fallbacks never fill a slot)
#   item          v1 item classes
#   chat_item     v1 item classes and the nonce words (chat item slots only, contracts 4.1)
#   entity_ref    EntityRef(ordinal, entity class | ("dropped", v1 item class))
#   entity_class  curated entity classes that can be named (not none, player or item)
#   player        player names (red_0 .. blue_19, human)
#   addressee     players, everyone, team
#   event_entity  every entity class a sentence can name (curated and fallback)
#   event_item    every item class except none (curated and fallback)
#   count         the non-negative number tokens (0-20, 32, 48, 64)
#   station, container, cause   result-word slots
#   and the fixed int or word sets named after their slot.
SLOT_DOMAINS: dict[str, str] = {
    "block": "block", "item": "item", "entity": "entity_ref", "player": "player", "furnace": "furnace",
    "chest": "chest", "dir8": "dir8", "dir4": "dir4", "explore_dist": "explore_dist",
    "strip_dist": "strip_dist", "y_band": "y_band", "mine_count": "mine_count", "item_count": "item_count",
    "pattern": "pattern", "wait_seconds": "wait_seconds", "addressee": "addressee",
}
CONTEXT_DOMAINS: dict[str, dict[str, str]] = {
    "sentence": {"item": "chat_item"},
    "goal": {"entity": "entity_class"},
    "event": {"entity": "event_entity", "item": "event_item", "n": "count"},
    "result": {"station": "station", "container": "container", "cause": "cause"},
}
INT_DOMAINS: dict[str, tuple[int, ...]] = {
    "furnace": FURNACES, "chest": CHESTS, "explore_dist": EXPLORE_DISTS, "strip_dist": STRIP_DISTS,
    "y_band": Y_BANDS, "mine_count": MINE_COUNTS, "item_count": ITEM_COUNTS, "wait_seconds": WAIT_SECONDS,
    "count": COUNT_TOKENS,
}
# Choice slots pick a template (contracts 4.5: "template", "intent", "goal").
CHOICE_SLOTS = ("template", "intent", "goal")


def domain_of(slot: str, context: str = "command") -> str:
    """The domain of a slot name in a context ("command", "sentence", "goal", "event", "result")."""
    over = CONTEXT_DOMAINS.get(context, {})
    if slot in over:
        return over[slot]
    if slot in SLOT_DOMAINS:
        return SLOT_DOMAINS[slot]
    raise KeyError(f"unknown slot {slot!r} in a {context}")


# ---------------------------------------------------------------------------
# Compiled templates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lit:
    """A literal word."""

    word: str


@dataclass(frozen=True)
class Slot:
    """A slot: its name (as in Command.args and SlotProvider.options) and its value domain."""

    name: str
    domain: str


@dataclass(frozen=True)
class Choice:
    """A nested sentence or goal: ``slot`` is "intent" or "goal"; ``field`` is the Command arg
    that holds the resulting Sentence or Goal."""

    slot: str
    field: str


Element = Lit | Slot | Choice


def compile_template(text: str, context: str = "command") -> tuple[Element, ...]:
    """Template text -> elements. ``{ordinal} {entity}`` becomes the single slot ``entity``."""
    parts = text.split()
    out: list[Element] = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p == "{ordinal}":
            if i + 1 >= len(parts) or parts[i + 1] != "{entity}":
                raise ValueError(f"{{ordinal}} must be followed by {{entity}} in {text!r}")
            out.append(Slot("entity", "entity_ref"))
            i += 2
            continue
        if p.startswith("{") and p.endswith("}"):
            name = p[1:-1]
            if name == "sentence":
                out.append(Choice("intent", "sentence"))
            elif name == "goal":
                out.append(Choice("goal", "goal"))
            else:
                out.append(Slot(name, domain_of(name, context)))
        else:
            if not p.islower() and not p.lstrip("-").isdigit():
                raise ValueError(f"template words must be lower case: {p!r} in {text!r}")
            out.append(Lit(p))
        i += 1
    return tuple(out)


COMMAND_ELEMENTS: dict[str, dict[str, tuple[Element, ...]]] = {
    verb: {tid: compile_template(t) for tid, t in templates.items()} for verb, templates in COMMANDS.items()
}
SENTENCE_ELEMENTS: dict[str, tuple[Element, ...]] = {k: compile_template(t, "sentence") for k, t in SENTENCES.items()}
GOAL_ELEMENTS: dict[str, tuple[Element, ...]] = {k: compile_template(t, "goal") for k, t in GOALS.items()}
RESULT_ELEMENTS: dict[str, tuple[Element, ...]] = {k: compile_template(t, "result") for k, t in RESULTS.items()}
EVENT_ELEMENTS: dict[str, tuple[Element, ...]] = {k: compile_template(t, "event") for k, t in EVENTS.items()}
NESTED: dict[str, Choice] = {  # verbs whose last slot is a nested sentence or goal
    verb: e for verb, ts in COMMAND_ELEMENTS.items() for els in ts.values() for e in els if isinstance(e, Choice)
}
CHOICE_ELEMENTS: dict[str, dict[str, tuple[Element, ...]]] = {"intent": SENTENCE_ELEMENTS, "goal": GOAL_ELEMENTS}


def multi_template(verb: str) -> bool:
    """True for the verbs whose Command.args carry a ``template`` slot."""
    return len(COMMANDS[verb]) > 1


def slot_names(elements: Iterable[Element]) -> tuple[str, ...]:
    return tuple(e.name for e in elements if isinstance(e, Slot))


def command_slot_names(verb: str, tid: str) -> frozenset[str]:
    """The Command.args keys of a verb's template (``template`` included where it applies)."""
    names = set(slot_names(COMMAND_ELEMENTS[verb][tid]))
    names.update(e.field for e in COMMAND_ELEMENTS[verb][tid] if isinstance(e, Choice))
    if multi_template(verb):
        names.add("template")
    return frozenset(names)


def infer_template(verb: str, given: set[str]) -> str | None:
    """The template id of ``verb`` whose slots are exactly ``given`` (``template`` excluded), or
    None for a single-template verb. Raises ValueError if no template or several match."""
    if verb not in COMMANDS:
        raise ValueError(f"unknown verb {verb!r}")
    if not multi_template(verb):
        return None
    fits = [tid for tid in COMMANDS[verb] if command_slot_names(verb, tid) - {"template"} == given]
    if len(fits) != 1:
        raise ValueError(f"cannot tell which {verb!r} template has slots {sorted(given)}: "
                         f"{'none' if not fits else ', '.join(fits)}; pass template=...")
    return fits[0]


def literal_words() -> dict[str, list[str]]:
    """Every literal word, by source, in template order (for the vocabulary build)."""
    def lits(texts: Iterable[str]) -> list[str]:
        out: list[str] = []
        for t in texts:
            for p in t.split():
                if not p.startswith("{") and p not in out:
                    out.append(p)
        return out

    return {
        "command templates": lits(t for ts in COMMANDS.values() for t in ts.values()),
        "chat sentences": lits(SENTENCES.values()),
        "goal templates": lits(GOALS.values()),
        "result words": lits(RESULTS.values()),
        "event sentences": lits(EVENTS.values()),
    }


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def snap(n: float, bins: Sequence[int]) -> int:
    """The bin nearest to ``n``; a tie goes to the smaller bin (contracts 4.1)."""
    best = None
    for b in sorted(bins):
        d = abs(n - b)
        if best is None or d < best[0]:
            best = (d, b)
    if best is None:
        raise ValueError("no bins")
    return best[1]


def snap_count(n: float) -> int:
    """A count as a number token: nearest of 0-20, 32, 48, 64, ties down, anything above 64 is
    64 (contracts 4.1, DESIGN.md 6.3)."""
    if n > 64:
        return 64
    return snap(max(n, 0), COUNT_TOKENS)


def number_word(n: int) -> str:
    if n not in NUMBER_TOKENS:
        raise ValueError(f"{n} is not a number token (snap it first)")
    return str(n)


def flag_of(verb: str) -> str | None:
    return VERB_FLAG.get(verb)


def check_static_grammar() -> None:
    """Sanity checks on the fixed text (run by the tests and the vocabulary build): every
    template, sentence and goal starts with a literal word, and each verb's templates start
    with words no other verb starts with, so the first word of a command picks its verb."""
    owner: dict[str, str] = {}
    for verb, ts in COMMAND_ELEMENTS.items():
        for els in ts.values():
            if not els or not isinstance(els[0], Lit):
                raise ValueError(f"{verb}: every template must start with a literal word")
            if owner.setdefault(els[0].word, verb) != verb:
                raise ValueError(f"first word {els[0].word!r} starts both {owner[els[0].word]!r} and {verb!r}")
    for mapping in (SENTENCE_ELEMENTS, GOAL_ELEMENTS):
        for els in mapping.values():
            if not els or not isinstance(els[0], Lit):
                raise ValueError("sentences and goals must start with a literal word")


def first_words() -> Mapping[str, str]:
    """First word -> verb id (the first word of a command picks its verb)."""
    return {els[0].word: verb for verb, ts in COMMAND_ELEMENTS.items() for els in ts.values()}  # type: ignore[union-attr]

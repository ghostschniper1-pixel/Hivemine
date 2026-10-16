"""Tests for the grammar, rendering and the strict parser (hivemind.lang, docs/contracts.md 4.3, 4.4)."""

from __future__ import annotations

import json
import pickle
from collections import Counter

import numpy as np
import pytest

from hivemind.lang import Command, EntityRef, Event, Flags, Goal, Language, ParseError, Sentence
from hivemind.lang import grammar as G


@pytest.fixture(scope="module")
def lang() -> Language:
    return Language.load()


def _keep(c: Command) -> bool:
    """Dense sample of the command space: everything except the large products of give, store,
    take and chat, which are cut to full slices (every item and count for one player or chest,
    every player or chest for one item; every sentence for one addressee, every addressee for a
    few sentences)."""
    if c.verb == "give":
        return c["player"] == "red_0" or (c["item"] == "bread" and c["item_count"] == 3)
    if c.verb in ("store", "take"):
        return c["chest"] == 1 or (c["item"] == "bread" and c["item_count"] == 3)
    if c.verb == "chat":
        s = c["sentence"]
        return c["addressee"] == "blue_19" or (s.get("item") in (None, "iron_ingot", "dax")
                                               and s.get("item_count") in (None, 3))
    return True


@pytest.fixture(scope="module")
def sample(lang: Language) -> list[Command]:
    return [c for c in lang.enumerate_commands(lang.full_provider(), Flags.all()) if _keep(c)]


# -- examples ---------------------------------------------------------------------------------


EXAMPLES = [
    ("mine 4 iron ore.", Command("mine", (("block", "iron_ore"), ("mine_count", 4)))),
    ("craft 4 stick.", Command("craft", (("item", "stick"), ("item_count", 4)))),
    ("craft 4 planks.", Command("craft", (("item", "planks"), ("item_count", 4)))),
    ("go to the nearest deepslate iron ore.", Command.of("goto", block="deepslate_iron_ore")),
    ("go to the second zombie villager.", Command.of("goto", entity=EntityRef(2, "zombie_villager"))),
    ("go to the nearest dropped raw iron.", Command.of("goto", entity=EntityRef(1, ("dropped", "raw_iron")))),
    ("go to red 3.", Command.of("goto", player="red_3")),
    ("go to my furnace 2.", Command.of("goto", furnace=2)),
    ("go to my chest 8.", Command.of("goto", chest=8)),
    ("go to my bed.", Command("goto", (("template", "bed"),))),
    ("go home.", Command("goto", (("template", "home"),))),
    ("explore northwest 64 blocks.", Command("explore", (("dir8", "northwest"), ("explore_dist", 64)))),
    ("dig down to y -54.", Command("descend", (("y_band", -54),))),
    ("strip mine east 32 blocks.", Command("strip_mine", (("dir4", "east"), ("strip_dist", 32)))),
    ("smelt 8 raw iron.", Command("smelt_start", (("item", "raw_iron"), ("item_count", 8)))),
    ("collect from my furnace 1.", Command("collect", (("furnace", 1),))),
    ("place cobblestone under me.", Command("place", (("item", "cobblestone"), ("pattern", "under_me")))),
    ("place torch for light.", Command("place", (("item", "torch"), ("pattern", "for_light")))),
    ("attack the sixteenth cave spider.", Command.of("attack", entity=EntityRef(16, "cave_spider"))),
    ("attack blue 7.", Command.of("attack", player="blue_7")),
    ("flee from the nearest creeper.", Command.of("flee", entity=EntityRef(1, "creeper"))),
    ("flee from human.", Command.of("flee", player="human")),
    ("eat bread.", Command("eat", (("item", "bread"),))),
    ("hold iron pickaxe.", Command("equip", (("item", "iron_pickaxe"), ("template", "hold")))),
    ("hold shield in off hand.", Command("equip", (("item", "shield"), ("template", "off_hand")))),
    ("wear iron helmet.", Command("equip", (("item", "iron_helmet"), ("template", "wear")))),
    ("sleep.", Command("sleep")),
    ("give 3 bread to red 7.", Command("give", (("item", "bread"), ("item_count", 3), ("player", "red_7")))),
    ("tell everyone i need 3 iron ingot.",
     Command("chat", (("addressee", "everyone"), ("sentence", Sentence("need", item_count=3, item="iron_ingot"))))),
    ("tell team where is dax.", Command("chat", (("addressee", "team"), ("sentence", Sentence("where", item="dax"))))),
    ("tell human thanks.", Command("chat", (("addressee", "human"), ("sentence", Sentence("thanks"))))),
    ("wait 20 seconds.", Command("wait", (("wait_seconds", 20),))),
    ("store 64 cobblestone in my chest 1.",
     Command("store", (("chest", 1), ("item", "cobblestone"), ("item_count", 64)))),
    ("take 2 iron ingot from my chest 3.", Command("take", (("chest", 3), ("item", "iron_ingot"), ("item_count", 2)))),
    ("plan to get 3 iron ingot.", Command("plan", (("goal", Goal("get", item_count=3, item="iron_ingot")),))),
    ("plan to kill 4 zombie.", Command("plan", (("goal", Goal("kill", mine_count=4, entity="zombie")),))),
    ("plan to go to the nearest log.", Command("plan", (("goal", Goal("to_block", block="log")),))),
    ("plan to survive until morning.", Command("plan", (("goal", Goal("survive")),))),
    ("plan to protect red 3.", Command("plan", (("goal", Goal("protect", player="red_3")),))),
]


@pytest.mark.parametrize("text,cmd", EXAMPLES, ids=[t for t, _ in EXAMPLES])
def test_examples(lang: Language, text: str, cmd: Command) -> None:
    assert lang.render(cmd) == text
    assert lang.parse(text) == cmd
    assert lang.to_words(cmd) == text[:-1].split() + ["."]
    assert lang.decode(lang.encode(cmd)) == cmd


def test_contract_api_example(lang: Language) -> None:
    cmd = Command(verb="mine", args=(("block", "iron_ore"), ("mine_count", 4)))
    assert lang.render(cmd) == "mine 4 iron ore."
    assert lang.to_words(cmd) == ["mine", "4", "iron", "ore", "."]
    ids = lang.encode(cmd)
    assert ids[-1] == G.END_ID and lang.decode(ids) == cmd
    assert lang.decode([G.BOS_ID, *ids, G.PAD_ID, G.PAD_ID]) == cmd
    assert lang.phrase("block", "iron_ore") == ("iron", "ore")
    assert lang.concept("item", ("iron", "ingot")) == "iron_ingot"


# -- the full command space -------------------------------------------------------------------


def test_round_trip_over_a_dense_sample(lang: Language, sample: list[Command]) -> None:
    verbs = Counter(c.verb for c in sample)
    assert set(verbs) == set(G.VERBS)
    assert len(sample) > 25_000
    for c in sample:
        text = lang.render(c)
        assert lang.parse(text) == c, text
        assert Command.from_json(c.to_json()) == c


def test_every_verb_slot_value_appears_in_the_sample(lang: Language, sample: list[Command]) -> None:
    """The sample covers every value of every slot of every verb (and of nested sentences and goals)."""
    seen: dict[tuple[str, str], set] = {}
    for c in sample:
        for k, v in c.args:
            if isinstance(v, (Sentence, Goal)):
                for k2, v2 in v.args:
                    seen.setdefault((c.verb, k2), set()).add(v2)
            else:
                seen.setdefault((c.verb, k), set()).add(v)
    from hivemind.lang.decoder import VERB_SLOT_DOMAINS

    for verb, slots in VERB_SLOT_DOMAINS.items():
        for slot, dom in slots.items():
            assert seen[(verb, slot)] == set(lang.domains[dom]), (verb, slot)


def run_slow(request: pytest.FixtureRequest) -> None:
    """Exhaustive checks run only when selected with ``-m slow`` (no conftest skips them by default)."""
    if "slow" not in (request.config.getoption("markexpr") or "") or "not slow" in request.config.getoption("markexpr"):
        pytest.skip("exhaustive check: run with -m slow")


@pytest.mark.slow
def test_round_trip_over_every_command(lang: Language, request: pytest.FixtureRequest) -> None:
    run_slow(request)
    n = 0
    for c in lang.enumerate_commands(lang.full_provider(), Flags.all()):
        assert lang.parse(lang.render(c)) == c
        n += 1
    assert n == 331_407  # recompute if the class tables change


def test_command_space_size(lang: Language) -> None:
    n_items, n_ents, n_players = len(lang.domains["item"]), len(lang.domains["entity_class"]), 41
    expected = {
        "give": len(G.ITEM_COUNTS) * n_items * n_players,
        "attack": 16 * (n_ents + n_items) + n_players,
        "store": len(G.ITEM_COUNTS) * n_items * 8,
        "equip": 3 * n_items,
    }
    counts = Counter(c.verb for c in lang.enumerate_commands(lang.full_provider(), Flags.all(),
                                                             verbs=expected))
    assert dict(counts) == expected


# -- strictness -------------------------------------------------------------------------------


@pytest.mark.parametrize("text,fragment", [
    ("mine 4 iron ingot.", "expected 'ore'"),
    ("mine 3 iron ore.", "{mine_count}"),
    ("mine 4 iron ore", "expected '.'"),
    ("stop.", "does not start with a verb"),
    ("dance.", "unknown word 'dance'"),
    ("mine 4 netherite ore.", "unknown word 'netherite'"),
    ("mine 4 other block.", "{block}"),  # fallback classes never fill a slot
    ("mine 4 slab.", "{block}"),
    ("go to the nearest monster.", "monster"),  # a fallback entity class
    ("craft 1 other item.", "{item}"),
    ("craft 1 dax.", "{item}"),  # nonce words only in chat item slots
    ("plan to get 1 dax.", "{item}"),
    ("go to the nearest dropped item.", "expected"),  # "dropped {item}" names the item
    ("attack the nearest player.", "expected"),  # players are named by name
    ("mine 4 stone . stone .", "expected the end"),
    ("give 3 bread to red 20.", "expected"),
    ("go to my furnace 5.", "{furnace}"),
    ("Mine 4 iron ore.", "unknown word"),
])
def test_parse_errors(lang: Language, text: str, fragment: str) -> None:
    with pytest.raises(ParseError) as err:
        lang.parse(text)
    assert fragment in str(err.value)


def test_parse_error_details(lang: Language) -> None:
    with pytest.raises(ParseError) as err:
        lang.parse("mine 4 iron ingot.")
    e = err.value
    assert e.position == 3 and e.words[3] == "ingot" and "'ore'" in e.expected
    with pytest.raises(ParseError):
        lang.decode([G.BOS_ID, 9999])
    with pytest.raises(ParseError):
        lang.parse("")


def test_render_rejects_invalid_commands(lang: Language) -> None:
    bad = [
        Command("mine", (("block", "other_other_block"), ("mine_count", 4))),  # fallback block
        Command("mine", (("block", "cobblestone"), ("mine_count", 4))),  # not a resource class
        Command("craft", (("item", "other_item_64"), ("item_count", 1))),  # fallback item
        Command("mine", (("block", "stone"), ("mine_count", 3))),  # 3 is not a mine_count bin
        Command("mine", (("block", "stone"),)),  # missing slot
        Command("mine", (("block", "stone"), ("mine_count", 4), ("item", "stick"))),  # extra slot
        Command("goto", (("block", "stone"),)),  # template missing
        Command("goto", (("block", "stone"), ("template", "entity"))),  # wrong template
        Command("mine", (("block", "stone"), ("mine_count", 4), ("template", "mine"))),
        Command("attack", (("entity", EntityRef(1, "monster")), ("template", "entity"))),  # fallback entity
        Command("attack", (("entity", EntityRef(1, ("dropped", "other_item_1"))), ("template", "entity"))),
        Command("chat", (("addressee", "everyone"), ("sentence", Goal("home")))),
        Command("chat", (("addressee", "red_0"), ("sentence", Sentence("need", item_count=3)))),
        Command("dance"),
    ]
    for c in bad:
        with pytest.raises(ValueError):
            lang.render(c)


def test_command_construction() -> None:
    a = Command("give", (("player", "red_7"), ("item_count", 3), ("item", "bread")))
    b = Command("give", [["item", "bread"], ("item_count", 3), ("player", "red_7")])
    assert a == b and hash(a) == hash(b) and a.args[0][0] == "item"
    assert a["item"] == "bread" and a.get("template") is None and a.slots["player"] == "red_7"
    assert Command.of("goto", block="log") == Command("goto", (("block", "log"), ("template", "block")))
    assert Command.of("attack", player="red_1")["template"] == "player"
    assert Command.of("mine", block="log", mine_count=1).args == (("block", "log"), ("mine_count", 1))
    for verb, slots in (("goto", {}), ("equip", {"item": "shield"}), ("goto", {"item": "stick"})):
        with pytest.raises(ValueError):
            Command.of(verb, **slots)
    with pytest.raises(ValueError):
        Command("give", (("item", "bread"), ("item", "stick")))
    with pytest.raises(ValueError):
        EntityRef(0, "zombie")
    with pytest.raises(ValueError):
        EntityRef(1, ("thrown", "bread"))
    assert EntityRef(1, ["dropped", "bread"]).dropped_item == "bread"
    s = Sentence("need", item="bread", item_count=2)
    assert s == Sentence("need", item_count=2, item="bread") and s.intent == "need" and s["item"] == "bread"
    assert s != Goal("need", item="bread", item_count=2)
    assert pickle.loads(pickle.dumps(s)) == s and pickle.loads(pickle.dumps(a)) == a
    with pytest.raises(AttributeError):
        s.args = ()  # type: ignore[misc]
    assert repr(s) == "Sentence('need', item='bread', item_count=2)"


def test_json_form(lang: Language) -> None:
    c = lang.parse("attack the second zombie.")
    assert c.to_json() == {"verb": "attack", "args": {"entity": {"ordinal": 2, "entity": "zombie"}, "template": "entity"}}
    d = lang.parse("go to the nearest dropped bread.")
    assert d.to_json()["args"]["entity"] == {"ordinal": 1, "dropped": "bread"}
    e = lang.parse("tell everyone i need 3 iron ingot.")
    assert e.to_json()["args"]["sentence"] == {"intent": "need", "item": "iron_ingot", "item_count": 3}
    for x in (c, d, e, lang.parse("plan to help red 2.")):
        assert Command.from_json(x.to_json()) == x


def test_values_are_plain_python(lang: Language) -> None:
    """NumPy numbers and strings become int and str (JSON-safe, equal to the plain command); a
    bool is never a value; a float is never a number."""
    plain = Command("mine", (("block", "iron_ore"), ("mine_count", 4)))
    c = Command("mine", (("block", np.str_("iron_ore")), ("mine_count", np.int64(4))))
    assert c == plain and hash(c) == hash(plain) and type(c["mine_count"]) is int and type(c["block"]) is str
    assert json.loads(json.dumps(c.to_json())) == plain.to_json() and lang.render(c) == "mine 4 iron ore."
    r = EntityRef(np.int32(2), (np.str_("dropped"), np.str_("bread")))
    assert r == EntityRef(2, ("dropped", "bread")) and type(r.ordinal) is int and json.dumps(r.to_json())
    assert Sentence("need", item_count=np.int64(3), item="bread") == Sentence("need", item_count=3, item="bread")
    assert lang.render_event("player_gave_me", player="blue_7", n=np.int64(3), item="bread") == "blue 7 gave me 3 bread"
    for bad in (True, np.bool_(True)):
        with pytest.raises(ValueError, match="bool"):
            Command("collect", (("furnace", bad),))
        with pytest.raises(ValueError):
            EntityRef(bad, "zombie")
    with pytest.raises(ValueError):
        lang.render(Command("mine", (("block", "iron_ore"), ("mine_count", 4.0))))
    assert not lang.in_domain("mine_count", 4.0) and not lang.in_domain("furnace", True)
    assert lang.domain_value("mine_count", np.int64(4)) == 4 and lang.in_domain("mine_count", np.int8(4))


def test_flags_do_not_limit_the_parser(lang: Language) -> None:
    for text in ("store 1 bread in my chest 1.", "take 1 bread from my chest 1.", "plan to go home."):
        assert lang.render(lang.parse(text)) == text
    assert Flags.coerce(None) == Flags() and Flags.coerce(["goals"]) == Flags(goals=True)
    assert Flags.coerce({"containers": True}).containers and Flags.all().goals
    with pytest.raises(ValueError):
        Flags.coerce(["teleport"])


# -- sentences, goals, results, events ----------------------------------------------------------


def test_sentences_and_goals(lang: Language) -> None:
    assert len(G.INTENTS) == 14 and len(G.GOAL_KINDS) == 9 and len(G.VERBS) == 20
    s = Sentence("need", item_count=3, item="iron_ingot")
    assert lang.to_words(s) == ["i", "need", "3", "iron", "ingot"]
    assert lang.parse_sentence("i need 3 iron ingot") == s
    assert lang.chat_text("everyone", s) == "everyone, i need 3 iron ingot."
    assert lang.chat_text("red_0", Sentence("follow")) == "red 0, follow me."
    for intent in G.INTENTS:
        slots = {k: v for k, v in (("item_count", 2), ("item", "wug")) if k in G.slot_names(G.SENTENCE_ELEMENTS[intent])}
        x = Sentence(intent, **slots)
        assert lang.parse_sentence(lang.to_words(x)) == x
    g = Goal("explore", dir8="south")
    assert lang.render(g) == "explore south" and lang.parse_goal("explore south") == g
    assert lang.parse_goal("go to the nearest iron ore") == Goal("to_block", block="iron_ore")
    assert lang.parse_goal("help human") == Goal("help", player="human")
    # verbs, intents, goals and result codes are named by their templates, not by phrases
    for kind, name in (("verb", "strip_mine"), ("intent", "need"), ("goal", "survive"), ("result", "NO_PATH")):
        with pytest.raises(KeyError, match="named by its template"):
            lang.phrase(kind, name)
    assert lang.templates["command"]["strip_mine"]["strip_mine"] == "strip mine {dir4} {strip_dist} blocks"


RESULT_CASES = [
    ("OK", {}, "done"),
    ("NO_TARGET", {}, "no target"),
    ("NO_PATH", {}, "no path"),
    ("PATH_TIMEOUT", {}, "path too slow"),
    ("STUCK", {}, "stuck"),
    ("DIG_FAILED", {}, "dig failed"),
    ("MISSING_INGREDIENT", {}, "missing ingredient"),
    ("NO_STATION", {"station": "crafting_table"}, "no crafting table nearby"),
    ("NO_STATION", {"station": "furnace"}, "no furnace nearby"),
    ("NO_STATION", {"station": "bed"}, "no bed nearby"),
    ("INV_FULL", {}, "inventory full"),
    ("INV_FULL", {"container": "chest"}, "chest full"),
    ("TARGET_LOST", {}, "target lost"),
    ("HAZARD", {}, "danger ahead"),
    ("INTERRUPTED", {"cause": "damage"}, "interrupted by damage"),
    ("INTERRUPTED", {"cause": "message"}, "interrupted by message"),
    ("INTERRUPTED", {"cause": "creeper"}, "interrupted by creeper"),
    ("INTERRUPTED", {"cause": "cave_spider"}, "interrupted by cave spider"),
    ("INTERRUPTED", {"cause": "monster"}, "interrupted by monster"),  # a hostile outside the curated list
    ("DIED", {}, "died"),
    ("TIMEOUT", {}, "too slow"),
]


@pytest.mark.parametrize("code,extra,text", RESULT_CASES, ids=[f"{c}-{t}" for c, _, t in RESULT_CASES])
def test_result_words(lang: Language, code: str, extra: dict, text: str) -> None:
    words = lang.result_words(code, **extra)
    assert " ".join(words) == text
    parsed_code, parsed_extra = lang.parse_result(words)
    assert parsed_code == code
    assert parsed_extra == ({"container": "inventory"} if code == "INV_FULL" and not extra else extra)


def test_every_failure_code_has_result_words(lang: Language) -> None:
    codes = ["OK", "NO_TARGET", "NO_PATH", "PATH_TIMEOUT", "STUCK", "DIG_FAILED", "MISSING_INGREDIENT", "NO_STATION",
             "INV_FULL", "TARGET_LOST", "HAZARD", "INTERRUPTED", "DIED", "TIMEOUT"]  # DESIGN.md 5
    assert list(G.RESULT_CODES) == codes
    assert {c for c, _, _ in RESULT_CASES} == set(codes)
    texts = {" ".join(lang.result_words(c, **e)) for c, e, _ in RESULT_CASES}
    assert len(texts) == len(RESULT_CASES)  # one-to-one


def test_result_word_errors(lang: Language) -> None:
    for code, extra in (("NO_STATION", {}), ("INTERRUPTED", {}), ("OK", {"cause": "damage"}),
                        ("NO_STATION", {"station": "chest"}), ("INTERRUPTED", {"cause": "cow"}),
                        ("INV_FULL", {"container": "barrel"}), ("TELEPORTED", {})):
        with pytest.raises(ValueError):
            lang.result_words(code, **extra)


EVENT_CASES = [
    (Event("player_attacked_me", player="blue_7"), "blue 7 attacked me"),
    (Event("entity_attacked_me", entity="zombie"), "zombie attacked me"),
    (Event("i_attacked_player", player="red_2"), "i attacked red 2"),
    (Event("player_killed_me", player="human"), "human killed me"),
    (Event("entity_killed_me", entity="monster"), "monster killed me"),
    (Event("i_killed_player", player="blue_19"), "i killed blue 19"),
    (Event("i_killed_entity", entity="cave_spider"), "i killed cave spider"),
    (Event("i_died"), "i died"),
    (Event("player_gave_me", player="blue_7", n=3, item="bread"), "blue 7 gave me 3 bread"),
    (Event("i_gave", n=12, item="iron_ingot", player="red_3"), "i gave 12 iron ingot to red 3"),
    (Event("player_opened_chest", player="blue_7", chest=1), "blue 7 opened my chest 1"),
    (Event("someone_opened_chest", chest=8), "someone opened my chest 8"),
    (Event("missing_from_chest", n=3, item="iron_ingot", chest=1), "3 iron ingot missing from my chest 1"),
    (Event("i_found", block="diamond_ore"), "i found diamond ore"),
]


@pytest.mark.parametrize("event,text", EVENT_CASES, ids=[t for _, t in EVENT_CASES])
def test_event_sentences(lang: Language, event: Event, text: str) -> None:
    assert lang.render(event) == text
    assert lang.render_event(event.kind, **event.slots) == text
    assert lang.parse_event(text) == event


def test_every_event_kind_is_covered() -> None:
    assert [e.kind for e, _ in EVENT_CASES] == list(G.EVENT_KINDS)


@pytest.mark.parametrize("n,word", [(0, "0"), (5, "5"), (20, "20"), (21, "20"), (26, "20"), (27, "32"), (40, "32"),
                                    (41, "48"), (56, "48"), (57, "64"), (64, "64"), (65, "64"), (1000, "64"),
                                    (-3, "0")])
def test_event_counts_snap_to_number_tokens(lang: Language, n: int, word: str) -> None:
    words = lang.event_words("player_gave_me", player="blue_7", n=n, item="other_item_64")
    assert words == ["blue", "7", "gave", "me", word, "other", "item"]
    assert G.snap_count(n) == int(word)


def test_event_errors(lang: Language) -> None:
    with pytest.raises(ValueError):
        lang.event_words("i_found", block="cobblestone")  # not a resource class
    with pytest.raises(ValueError):
        lang.event_words("player_gave_me", player="blue_7", n=2.5, item="bread")
    with pytest.raises(ValueError):
        lang.event_words("i_teleported")
    with pytest.raises(ValueError):
        lang.event_words("i_died", player="red_1")


def test_snap() -> None:
    assert G.snap(5, G.ITEM_COUNTS) == 4  # tie 4/6 goes down
    assert G.snap(7, G.ITEM_COUNTS) == 6  # tie 6/8 goes down
    assert G.snap(100, G.MINE_COUNTS) == 16
    assert G.snap(32, G.Y_BANDS) == 16  # tie 48/16 goes to the smaller
    assert G.snap(-60, G.Y_BANDS) == -54


def test_static_grammar() -> None:
    G.check_static_grammar()
    assert G.first_words()["hold"] == "equip" and G.first_words()["wear"] == "equip"
    assert G.compile_template("go to the {ordinal} {entity}")[-1] == G.Slot("entity", "entity_ref")
    assert G.domain_of("item", "sentence") == "chat_item" and G.domain_of("entity", "goal") == "entity_class"
    with pytest.raises(ValueError):
        G.compile_template("go to the {ordinal}")

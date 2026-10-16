"""Tests for the human tokenizer and lenient parser (hivemind.lang.human, contracts 4.4, DESIGN.md 9)."""

from __future__ import annotations

import pytest

from hivemind.lang import Command, Language, Sentence
from hivemind.lang import grammar as G
from hivemind.lang.human import PHRASE_SYNONYMS, SENTENCE_OPTIONAL, SYNONYMS, HumanReader, plurals, singulars


@pytest.fixture(scope="module")
def lang() -> Language:
    return Language.load()


@pytest.fixture(scope="module")
def reader(lang: Language) -> HumanReader:
    return lang._human_reader()


SLOT_VALUES = {"item_count": 2, "item": "bread", "block": "iron_ore"}  # a value per sentence slot


def show(lang: Language, x: Command | Sentence | None) -> str | None:
    if x is None:
        return None
    return lang.render(x) if isinstance(x, Command) else " ".join(lang.to_words(x))


# (text, speaker, expected content as canonical text or None, expected addressee)
CASES = [
    # the owner's examples
    ("red 0, mine 4 iron ore", "human", "mine 4 iron ore.", "red_0"),
    ("can you give me 3 breads?", "human", "give me 3 bread", None),
    ("i need three iron ingots", "human", "i need 3 iron ingot", None),
    ("where is the iron ore", "human", "where is iron ore", None),  # draft 0.5: the block form
    ("red 0, where is the iron ore", "human", "where is iron ore", "red_0"),
    ("where are the sticks?", "human", "where is stick", None),
    ("where's the stone", "human", "where is stone", None),  # the block (test_where_reads_blocks_as_blocks)
    ("where is the oak log", "human", "where is log", None),  # a member name: the block log
    ("where is some wood", "human", "where is log", None),
    ("where is the coal", "human", "where is coal", None),  # the item
    ("where is the coal ore", "human", "where is coal ore", None),  # the block
    ("where are the diamonds", "human", "where is diamond", None),
    ("where is the iron", "human", None, None),  # raw iron, iron ingot, iron ore? no guess
    # addressees
    ("everyone, i need 3 iron ingot", "human", "i need 3 iron ingot", "everyone"),
    ("Team, follow me!", "human", "follow me", "team"),
    ("hey red 0 come here", "human", "come here", "red_0"),
    ("blue 7 go home", "human", "go home.", "blue_7"),
    ("tell everyone i need 3 iron ingot", "human", "tell everyone i need 3 iron ingot.", None),
    # "me" in a player slot is the speaker; in a sentence it stays "me"
    ("give 3 bread to me", "human", "give 3 bread to human.", None),
    ("give 3 bread to me", "blue 4", "give 3 bread to blue 4.", None),
    ("give 3 bread to me", "llm", None, None),  # an LLM is not a player
    ("red 0 go to me", "human", "go to human.", "red_0"),
    ("red 0 attack me", "red_5", "attack red 5.", "red_0"),
    ("give red 2 three bread", "human", "give 3 bread to red 2.", None),
    # counts: number words, articles, bins (ties down), defaults
    ("craft 5 torches", "human", "craft 4 torch.", None),
    ("i need 7 sticks", "human", "i need 6 stick", None),
    ("mine 6 iron ores", "human", "mine 4 iron ore.", None),
    ("mine twenty stone", "human", "mine 16 stone.", None),
    ("i have 100 sticks", "human", "i have 64 stick", None),
    ("Please craft a stone pick", "human", "craft 1 stone pickaxe.", None),
    ("i need some sweet berry", "human", "i need 1 sweet berries", None),
    ("mine iron ore", "human", "mine 1 iron ore.", None),
    ("can i have a stack of planks", "human", "give me 64 planks", None),
    ("i need half a stack of torches", "human", "i need 32 torch", None),
    ("explore north east 50 blocks", "human", "explore northeast 64 blocks.", None),
    ("dig down to y -20", "human", "dig down to y -16.", None),
    ("dig down to y -64", "human", "dig down to y -54.", None),  # the deepest band
    ("wait a second", "human", "wait 1 seconds.", None),
    # a number the slot cannot mean gives no parse, never a different number ("none" is not "one")
    ("i have 0 sticks", "human", None, None),
    ("i have zero bread", "human", None, None),
    ("i need 0 torches", "human", None, None),
    ("give me 0 bread", "human", None, None),
    ("mine 0 iron ore", "human", None, None),
    ("craft -3 torches", "human", None, None),
    ("give 0 bread to red 1", "human", None, None),
    ("wait 0 seconds", "human", None, None),
    ("explore north 0 blocks", "human", None, None),
    ("strip mine north 0 blocks", "human", None, None),
    ("dig down to y -200", "human", None, None),  # below the world
    ("collect from my furnace 5", "human", None, None),  # furnace and chest numbers are exact
    # plurals, synonyms, member names
    ("i need a wooden pick", "human", "i need 1 wooden pickaxe", None),
    ("craft a gold sword", "human", "craft 1 golden sword.", None),
    ("craft a wood pickaxe", "human", "craft 1 wooden pickaxe.", None),
    ("craft a workbench", "human", "craft 1 crafting table.", None),
    ("i got 4 cobble", "human", "i have 4 cobblestone", None),
    ("i need 4 wood", "human", "i need 4 log", None),  # "wood" is a member-name word and a synonym
    ("mine 4 oak wood", "human", "mine 4 log.", None),
    ("i need 4 wood planks", "human", None, None),  # a modifier here: not "log planks"
    ("need 4 bread", "human", "i need 4 bread", None),  # the "i" of "i need" may be left out
    ("retreat", "human", "flee", None),
    ("retreat from the creeper", "human", "flee from the nearest creeper.", None),
    ("i need 3 carrots", "human", "i need 3 carrot", None),  # "carrots" (the crop's phrase) as an item
    ("chop 4 spruce logs", "human", "mine 4 log.", None),
    ("eat cooked beef", "human", "eat steak.", None),
    ("cook 3 beef", "human", "smelt 3 raw beef.", None),
    ("mine 4 lapis ore", "human", "mine 4 lapis lazuli ore.", None),
    ("i need 2 daxes", "human", "i need 2 dax", None),  # nonce words in chat item slots
    ("hold shield in offhand", "human", "hold shield in off hand.", None),
    ("put on iron helmet", "human", "wear iron helmet.", None),
    ("equip iron helmet", "human", "wear iron helmet.", None),  # "equip" armor: wear it
    ("equip a diamond sword", "human", "hold diamond sword.", None),  # anything else: hold it
    ("equip shield in off hand", "human", "hold shield in off hand.", None),
    ("place a torch in front of me", "human", "place torch in front.", None),
    ("put 3 bread in chest 2", "human", "store 3 bread in my chest 2.", None),
    # entities and ordinals
    ("go to the iron ore", "human", "go to the nearest iron ore.", None),
    ("attack the 2nd zombie", "human", "attack the second zombie.", None),
    ("kill the zombie villager", "human", "attack the nearest zombie villager.", None),
    ("go to the closest cow", "human", "go to the nearest cow.", None),
    ("run from the creeper", "human", "flee from the nearest creeper.", None),
    ("go to the nearest dropped iron ingot", "human", "go to the nearest dropped iron ingot.", None),
    ("attack the 17th zombie", "human", None, None),
    # one-word sentences
    ("thank you!", "human", "thanks", None),
    ("yeah", "human", "yes", None),
    ("nope", "human", "no", None),
    ("help!", "human", "help me", None),
    ("run away", "human", "flee", None),
    ("wait", "human", "wait", None),
    ("attack", "human", "attack", None),
    ("stop", "human", "stop", None),
    # unknown or ambiguous words give no parse rather than a wrong one
    ("i need 3 netherite ingots", "human", None, None),
    ("i need 3 iron", "human", None, None),  # raw iron, iron ingot or iron nugget?
    ("got any iron?", "human", None, None),
    ("pick up the iron ingot", "human", None, None),  # not "pickaxe up"
    # a member name that is another kind's phrase is not an alias: "chicken" is the mob
    ("where is the chicken", "human", None, None),  # not "where is raw chicken"
    ("where is the rabbit", "human", None, None),
    ("cook 3 chicken", "human", None, None),
    ("cook 3 raw chicken", "human", "smelt 3 raw chicken.", None),
    ("go to the chicken", "human", "go to the nearest chicken.", None),
    ("go to the tnt", "human", None, None),  # "tnt" is the block, not primed tnt
    ("red 0, flee from the tnt", "human", None, "red_0"),
    ("flee from the primed tnt", "human", "flee from the nearest primed tnt.", None),
    ("mine 4 wheat", "human", None, None),  # "wheat" is the item, not the wheat crops
    ("mine 4 blorp", "human", None, None),
    ("red 0, mine 4 blorp", "human", None, "red_0"),
    ("", "human", None, None),
]


@pytest.mark.parametrize("text,speaker,expected,addressee", CASES, ids=[f"{t}|{s}" for t, s, _, _ in CASES])
def test_parse_human(lang: Language, text: str, speaker: str, expected: str | None, addressee: str | None) -> None:
    msg = lang.read_human(text, speaker)
    assert show(lang, msg.content) == expected
    assert msg.addressee == addressee
    assert lang.parse_human(text, speaker) == msg.content
    if isinstance(msg.content, Command):  # whatever a person types, the result is a valid command
        assert lang.parse(lang.render(msg.content)) == msg.content
    assert msg.content_words == (None if msg.content is None else tuple(lang.to_words(msg.content)))


NORMALIZE = [
    ("Can you give me 3 breads?", "<unk> <unk> give me 3 bread"),
    ("I need THIRTY-TWO sticks", "i need 32 stick"),
    ("I need thirty two sticks", "i need 32 stick"),
    ("mine twenty five stone", "mine 20 stone"),  # 25 -> nearest number token, tie down
    ("i need 100 sticks", "i need 64 stick"),
    ("dig down to y -20", "dig down to y -16"),
    ("dig to -100", "dig to -54"),
    ("craft a crafting table", "craft 1 crafting table"),  # a/an -> 1 only before an item ...
    ("mine an iron ore", "mine 1 iron ore"),  # ... or a resource block
    ("attack a zombie", "attack <unk> zombie"),
    ("a workbench", "1 crafting table"),
    ("stripped spruce logs and oak planks", "log and oak planks"),  # member names with a word outside the vocabulary
    ("spruce logs", "spruce log"),  # only vocabulary words ("spruce" is a biome word): kept as typed
    ("white wool", "wool"),
    ("plank", "planks"),
    ("ores", "ore"),
    ("sticks", "stick"),
    ("berries", "berries"),
    ("sweet berry", "sweet berry"),
    ("zombies", "zombie"),
    ("wolves", "wolf"),
    ("leaf", "leaves"),
    ("potatos", "potato"),
    ("2nd zombie", "second zombie"),
    ("first zombie", "nearest zombie"),
    ("blah", "<unk>"),
    ("where's my pick?", "where is my pickaxe"),
    ("thank you", "thanks"),
    ("gimme a dozen apples", "give me 12 apple"),
    ("a couple of torches", "2 torch"),
    ("3x bread", "3 bread"),
    ("i have 0 sticks", "i have 0 stick"),  # the true number, though it does not parse
    ("wood", "log"),
    ("stripped dark oak wood", "log"),
    ("dark oak wood", "dark oak log"),  # "dark" is a biome word now: vocabulary words are kept, "wood" -> "log"
    ("wood planks", "<unk> planks"),  # a modifier: not "log planks"
    ("retreat", "flee"),
    ("i need 3 carrots", "i need 3 carrots"),  # a vocabulary word is kept (context-free; see content_words)
]


@pytest.mark.parametrize("text,words", NORMALIZE, ids=[t for t, _ in NORMALIZE])
def test_normalize_human(lang: Language, text: str, words: str) -> None:
    assert " ".join(lang.normalize_human(text)) == words
    assert all(lang.has_word(w) for w in lang.normalize_human(text))


def test_word_forms() -> None:
    assert plurals("berry") == {"berries"} and plurals("torch") == {"torches"} and plurals("leaf") >= {"leaves"}
    assert "zombies" in plurals("zombie") and "potatoes" in plurals("potato") and "endermen" in plurals("enderman")
    assert singulars("berries") == {"berry"} and singulars("planks") == {"plank"} and "leaf" in singulars("leaves")
    assert singulars("lapis") == set() and singulars("glass") == set()


def test_number_variants_share_a_key(reader: HumanReader) -> None:
    for a, b in (("berry", "berries"), ("carrot", "carrots"), ("potato", "potatoes"), ("brick", "bricks"),
                 ("second", "seconds"), ("block", "blocks")):
        assert reader.key(a) == reader.key(b), (a, b)
    assert reader.key("stone") != reader.key("stick")


def test_synonym_targets_are_vocabulary_words(lang: Language) -> None:
    for target in [*SYNONYMS.values(), *(dst for _, dst in PHRASE_SYNONYMS)]:
        for w in target:
            assert lang.has_word(w) or w == "pick_up", w
    # deliberately unmapped: bare metal and other words that name several classes
    for w in ("iron", "gold", "copper", "grass", "sword", "ingot"):
        assert w not in SYNONYMS


def test_every_synonym_takes_effect(lang: Language) -> None:
    """No synonym is shadowed by an earlier rule (a member-name word, a verb alternative, ...)."""
    for word, target in SYNONYMS.items():
        assert lang.normalize_human(word) == list(target), word
    for src, dst in PHRASE_SYNONYMS:
        if dst != ("pick_up",):  # picking up is not a command: stays unknown
            assert lang.normalize_human(" ".join(src)) == list(dst), src
    for intent, words in SENTENCE_OPTIONAL.items():  # every optional sentence word can be left out
        for els in G.SENTENCE_ELEMENTS[intent]:
            s = Sentence(intent, **{k: SLOT_VALUES[k] for k in G.slot_names(els)})
            for w in words:
                short = " ".join(x for x in lang.to_words(s) if x != w)
                assert lang.parse_human(short) == s, (intent, short)


def test_variants_point_to_known_words(reader: HumanReader) -> None:
    for form, word in reader.variants.items():
        assert word in reader.vocab or word in reader.alias_words
        assert form in plurals(word) | singulars(word)


def test_human_message_fields(lang: Language) -> None:
    msg = lang.read_human("red 0, mine 4 iron ore", "human")
    assert msg.speaker == "human" and msg.addressee == "red_0"
    assert msg.words == ("red", "0", "mine", "4", "iron", "ore")
    assert msg.content_words == ("mine", "4", "iron", "ore", ".")
    # normalized words are context-free; the parse's words say what was understood
    carrots = lang.read_human("i need 3 carrots")
    assert carrots.words == ("i", "need", "3", "carrots") and lang.phrase("block", "carrots") == ("carrots",)
    assert carrots.content == Sentence("need", item_count=3, item="carrot")
    assert carrots.content_words == ("i", "need", "3", *lang.phrase("item", "carrot")) == ("i", "need", "3", "carrot")
    assert lang.read_human("mine 4 blorp").content_words is None
    assert lang.read_human("i need 3 bread", "red 5").speaker == "red_5"
    assert lang.parse_human("mine 4 iron ore.") == lang.parse("mine 4 iron ore.")  # canonical text reads as itself


def test_canonical_text_reads_as_itself(lang: Language) -> None:
    """Every canonical command of a sample is read back unchanged by the lenient parser."""
    sample = ["go to the nearest iron ore.", "go to the second zombie villager.", "hold shield in off hand.",
              "tell red 1 i have 12 raw iron.", "plan to protect red 3.", "explore northwest 64 blocks.",
              "wait 5 seconds.", "collect from my furnace 4.", "take 2 iron ingot from my chest 3.",
              "attack the nearest dropped flint and steel.", "place cobblestone under me."]
    for text in sample:
        assert lang.parse_human(text, "human") == lang.parse(text), text
    assert lang.parse_human("i need 3 iron ingot", "human") == Sentence("need", item_count=3, item="iron_ingot")
    for intent, forms in G.SENTENCE_ELEMENTS.items():
        for els in forms:
            s = Sentence(intent, **{k: SLOT_VALUES[k] for k in G.slot_names(els)})
            assert lang.parse_human(" ".join(lang.to_words(s)), "human") == s


def test_where_reads_blocks_as_blocks(lang: Language) -> None:
    """Draft 0.5: "where is ..." names a resource block with the block form and an item with the
    item form, from the canonical phrase or a member name, and never maps a word to another
    class. The 17 resource blocks that share their phrase with an item are read as the block,
    the place to go (contracts 4.3)."""
    blocks, where_items = lang.domains["block"], lang.domains["where_item"]
    for b in blocks:
        phrase = " ".join(lang.phrase("block", b))
        assert lang.parse_human(f"where is the {phrase}") == Sentence("where", block=b), phrase
    for i in where_items:
        phrase = " ".join(lang.value_words("where_item", i))
        assert lang.parse_human(f"where is {phrase}") == Sentence("where", item=i), phrase
    shared = [i for i in lang.domains["item"] if i not in where_items]
    assert len(shared) == 17
    for i in shared:  # the item log is named "log", and so is the block: read as the block
        b = lang.concept("block", lang.phrase("item", i))
        assert lang.parse_human(f"where is the {' '.join(lang.phrase('item', i))}") == Sentence("where", block=b)
    # member names: a block member's name gives that block class, an item member's that item class
    for a, b in lang.aliases["block"].items():
        assert lang.parse_human(f"where is the {a}") == Sentence("where", block=b), a
    for a, i in lang.aliases["item"].items():
        got = lang.parse_human(f"where is the {a}")
        if i in where_items:
            assert got == Sentence("where", item=i), a
        else:  # an item named like a block: its member names read as the block they also name, or not at all
            assert got is None or (got.get("block") is not None and a in lang.aliases["block"]), (a, got)

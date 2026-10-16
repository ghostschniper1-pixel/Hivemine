"""Tests for the shared vocabulary (hivemind.lang.vocab, docs/contracts.md 4.1, 4.2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hivemind.lang import Language
from hivemind.lang import grammar as G
from hivemind.lang import vocab as V
from hivemind.lang import world_words as W
from hivemind.lang.language import DEFAULT_VOCAB_PATH, vocab_hash, words_hash
from hivemind.rules.classes import Concepts, concepts_hash, load_concepts


@pytest.fixture(scope="module")
def lang() -> Language:
    return Language.load()


@pytest.fixture(scope="module")
def fresh() -> dict:
    return V.build_vocab()


def test_file_is_up_to_date_and_deterministic(fresh: dict) -> None:
    data = V.render(fresh)
    assert data == DEFAULT_VOCAB_PATH.read_bytes(), "rebuild: python -m hivemind.lang.vocab"
    assert V.render(V.build_vocab()) == data  # same bytes on a second build


def test_check_mode(tmp_path: Path, capsys) -> None:
    assert V.main(["--check"]) == 0
    stale = tmp_path / "vocab.json"
    stale.write_bytes(DEFAULT_VOCAB_PATH.read_bytes().replace(b'"dax"', b'"dux"'))
    assert V.main(["--check", "--out", str(stale)]) == 1
    out = tmp_path / "new.json"
    assert V.main(["--out", str(out)]) == 0
    assert out.read_bytes() == DEFAULT_VOCAB_PATH.read_bytes()
    assert "words" in capsys.readouterr().out


def test_load_is_cached_and_verified(lang: Language) -> None:
    assert Language.load() is lang
    assert lang.compute_hash() == lang.hash == vocab_hash(json.loads(DEFAULT_VOCAB_PATH.read_text()))
    assert lang.compute_words_hash() == lang.words_hash
    assert lang.concepts_hash == load_concepts().hash


def test_words_hash_ignores_provenance(fresh: dict) -> None:
    """A rules-only rebuild changes the concepts hash, hence ``hash``, but not ``words_hash``."""
    assert fresh["words_hash"] == words_hash(fresh)
    raw = dict(load_concepts().raw)
    raw["rules_hash"] = "0" * 64
    raw["hash"] = concepts_hash(raw)
    other = V.build_vocab(Concepts(raw))
    assert other["concepts_hash"] != fresh["concepts_hash"] and other["hash"] != fresh["hash"]
    assert other["words_hash"] == fresh["words_hash"]
    changed = dict(fresh, words=[*fresh["words"][:-1], "dux"])  # a different word changes it
    assert words_hash(changed) != fresh["words_hash"]


def test_tampered_file_is_rejected(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_VOCAB_PATH.read_text())
    raw["words"][10], raw["words"][11] = raw["words"][11], raw["words"][10]
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="hash"):
        Language.load(path)
    assert Language.load(path, verify=False).words[10] == raw["words"][10]
    raw["templates"]["command"]["mine"]["mine"] = "dig {mine_count} {block}"
    with pytest.raises(ValueError, match="templates"):
        Language(raw)
    raw = json.loads(DEFAULT_VOCAB_PATH.read_text())  # a stale words_hash under a matching hash
    raw["words_hash"] = "0" * 64
    raw["hash"] = vocab_hash(raw)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="hash"):
        Language.load(path)


def test_reserved_ids_and_no_duplicates(lang: Language) -> None:
    assert lang.words[:4] == ["<pad>", "<unk>", "<bos>", "."]
    assert [lang.word_id(w) for w in G.RESERVED] == [G.PAD_ID, G.UNK_ID, G.BOS_ID, G.END_ID]
    assert len(set(lang.words)) == len(lang.words) == len(lang)
    for w in lang.words[4:]:
        assert w == w.lower() and " " not in w and w, w
    with pytest.raises(KeyError):
        lang.word_id("netherite")


def test_every_phrase_and_grammar_word_is_in_the_vocabulary(lang: Language) -> None:
    concepts = load_concepts()
    for classes in (concepts.block_classes, concepts.item_classes, concepts.entity_classes, concepts.biomes):
        for c in classes:  # curated and fallback classes alike, and every biome
            for w in c.phrase:
                assert lang.has_word(w), (c.name, w)
    for r in lang.raw["concepts"]:
        assert all(lang.has_word(w) for w in r["phrase"]), r
    for ws in G.literal_words().values():
        assert all(lang.has_word(w) for w in ws)
    for words in [G.ORDINALS, G.DIR8, G.NONCE_WORDS, G.RELATIONS, G.PLACE_KINDS, G.CAUSE_WORDS, G.CONTAINERS,
                  W.FACE_WORDS, W.TIME_WORDS,
                  [str(n) for n in G.NUMBER_TOKENS], [G.LLM, "me", "everyone", "team", "dropped", "someone"]]:
        assert all(lang.has_word(w) for w in words), words


def test_number_tokens_and_players(lang: Language) -> None:
    assert [lang.phrase("number", n) for n in (0, 20, 32, 48, 64, -16, -32, -54)] == [
        ("0",), ("20",), ("32",), ("48",), ("64",), ("-16",), ("-32",), ("-54",)]
    for bad in ("21", "63", "100", "-1"):
        assert not lang.has_word(bad)
    assert lang.phrase("player", "red_0") == ("red", "0")
    assert lang.phrase("player", "blue_19") == ("blue", "19")
    assert lang.phrase("player", "human") == ("human",)
    assert len(lang.domains["player"]) == 41
    assert lang.concept("player", ("blue", "7")) == "blue_7"


def test_word_order_follows_the_sources(lang: Language, fresh: dict) -> None:
    order = []
    for ws in fresh["sources"].values():
        order += [w for w in ws if w not in order and w not in G.RESERVED]
    assert lang.words[4:] == order
    # concept words come first, the nonce words last
    assert lang.words[4] == "air"
    assert lang.words[-8:] == list(G.NONCE_WORDS)
    assert list(fresh["sources"])[5:7] == ["fallback entity classes", "biomes"]  # biomes follow the entity classes
    i = list(fresh["sources"]).index("directions")
    assert list(fresh["sources"])[i:i + 3] == ["directions", "face words", "time words"]
    # the new time words sit together in their fixed order
    ids = [lang.word_id(w) for w in W.TIME_WORDS]
    assert ids == list(range(ids[0], ids[0] + 4))


def test_concept_tables(lang: Language) -> None:
    concepts = load_concepts()
    for c in concepts.block_classes:
        assert lang.phrase("block", c.name) == c.phrase
        row = lang.row("block", c.name)
        assert (row["id"], row["fallback"], row["resource"]) == (c.id, c.fallback, c.resource)
        if not c.fallback:
            assert lang.concept("block", c.phrase) == c.name
    for c in concepts.item_classes:
        assert lang.phrase("item", c.name) == c.phrase
    for c in concepts.entity_classes:
        assert lang.phrase("entity", c.name) == c.phrase
        assert lang.row("entity", c.name)["entity_kind"] == c.kind
    with pytest.raises(KeyError):
        lang.concept("block", ("other", "block"))  # shared by several fallbacks: names none of them
    assert lang.phrase("station", "crafting_table") == ("crafting", "table")
    assert lang.phrase("pattern", "under_me") == ("under", "me")
    assert lang.phrase("ordinal", 16) == ("sixteenth",)
    assert lang.phrase("equip_slot", "offhand") == ("off", "hand")
    assert lang.phrase("container", "furnace") == ("furnace",)  # draft 0.5: furnace full
    assert lang.concept("container", ("furnace",)) == "furnace"


def test_biome_phrases(lang: Language) -> None:
    """Draft 0.5: every 1.21.1 biome is a concept named by its English phrase (contracts 3)."""
    concepts = load_concepts()
    assert len(concepts.biomes) == 64 and sum(b.overworld for b in concepts.biomes) == 53
    for b in concepts.biomes:
        assert lang.phrase("biome", b.name) == b.phrase
        assert lang.concept("biome", b.phrase) == b.name  # biome phrases are unique
        row = lang.row("biome", b.name)
        assert (row["id"], row["overworld"], row["dimension"]) == (b.id, b.overworld, b.dimension)
    assert lang.phrase("biome", "birch_forest") == ("birch", "forest")
    assert lang.phrase("biome", "plains") == ("plains",)
    assert lang.phrase("biome", "old_growth_spruce_taiga") == ("old", "growth", "spruce", "taiga")
    assert lang.concept("biome", ("windswept", "gravelly", "hills")) == "windswept_gravelly_hills"
    assert not lang.row("biome", "the_end")["overworld"] and lang.row("biome", "deep_dark")["overworld"]
    with pytest.raises(KeyError):
        lang.concept("biome", ("birch",))
    with pytest.raises(KeyError):
        lang.phrase("biome", "birch")
    table = lang.class_phrase_ids("biome")
    assert table.shape == (64, 4)
    assert [lang.word(i) for i in table[concepts.biome("birch_forest").id] if i] == ["birch", "forest"]
    assert lang.phrase_ids("biome", "plains") == (lang.word_id("plains"),)
    # biome phrases name no block, item or entity class
    things = {tuple(r["phrase"]) for r in lang.raw["concepts"] if r["kind"] in ("block", "item", "entity")}
    assert not things & {b.phrase for b in concepts.biomes}


def test_draft_05_words(lang: Language) -> None:
    """unseen replaces buried; the biome, time and face words are in the vocabulary."""
    assert lang.has_word("unseen") and not lang.has_word("buried")
    assert lang.phrase("block", "unseen") == ("unseen",) and lang.row("block", "unseen")["id"] == 255
    for w in ("day", "dusk", "night", "dawn", "top", "bottom", "north", "east", "south", "west", "birch", "forest",
              "plains", "taiga", "windswept", "lukewarm"):
        assert lang.has_word(w), w
    assert [lang.phrase("time", t) for t in W.TIME_WORDS] == [("day",), ("dusk",), ("night",), ("dawn",)]
    assert [lang.phrase("face", f) for f in W.FACE_WORDS] == [(f,) for f in ("top", "bottom", "north", "south",
                                                                             "east", "west")]
    assert lang.concept("face", ("top",)) == "top" and lang.concept("time", ("dawn",)) == "dawn"
    assert lang.phrase_ids("time", "dusk") == (lang.word_id("dusk"),)
    # face and time words are the observation's words only: no command, sentence or goal uses them
    literal = {w for ws in G.literal_words().values() for w in ws}
    assert not literal & {"day", "dusk", "night", "dawn", "top", "bottom", "unseen"}


def test_where_item_domain(lang: Language) -> None:
    """``where is {item}`` leaves out exactly the items whose phrase is a resource block's: those
    words name the block (where is {block}), so the chat sentence has one reading (contracts 4.3)."""
    blocks = {lang.phrase("block", b) for b in lang.domains["block"]}
    named_like_block = [i for i in lang.domains["item"] if lang.phrase("item", i) in blocks]
    assert len(named_like_block) == 17 and {"log", "stone", "sand", "dirt", "obsidian"} <= set(named_like_block)
    assert list(lang.domains["where_item"]) == [i for i in lang.domains["chat_item"] if i not in named_like_block]
    where = {lang.value_words("where_item", v) for v in lang.domains["where_item"]}
    assert not where & blocks
    assert "dax" in lang.domains["where_item"] and "iron_ingot" in lang.domains["where_item"]


def test_domains_never_hold_fallbacks(lang: Language) -> None:
    concepts = load_concepts()
    blocks = lang.domains["block"]
    assert blocks and all(concepts.block_class(b).resource and not concepts.block_class(b).fallback for b in blocks)
    assert set(blocks) == {c.name for c in concepts.block_classes if c.resource}
    items = lang.domains["item"]
    assert len(items) == 159 and all(concepts.item_class(i).v1 and not concepts.item_class(i).fallback for i in items)
    assert set(lang.domains["chat_item"]) == set(items) | set(G.NONCE_WORDS)
    ents = lang.domains["entity_class"]
    assert all(not concepts.entity_class(e).fallback for e in ents)
    assert not {"none", "player", "item", "monster", "animal", "object"} & set(ents)
    refs = lang.domains["entity_ref"]
    assert len(refs) == 16 * (len(ents) + len(items))
    for r in refs:
        assert (r.dropped_item in items) if r.dropped_item else (r.entity in ents)
    # observation-only domains do name fallbacks (their phrases are true of every member)
    assert "monster" in lang.domains["event_entity"] and "other_item_64" in lang.domains["event_item"]


def test_phrases_are_unique_within_each_slot_domain(lang: Language) -> None:
    for dom, values in lang.domains.items():
        phrases = [lang.value_words(dom, v) for v in values]
        assert len(set(phrases)) == len(phrases), dom


def test_human_aliases_are_true_and_unambiguous(lang: Language) -> None:
    concepts = load_concepts()
    aliases = lang.aliases
    assert aliases["block"]["oak log"] == "log" and aliases["item"]["white wool"] == "wool"
    assert aliases["item"]["beef"] == "beef" and lang.phrase("item", "beef") == ("raw", "beef")
    assert aliases["item"]["cooked beef"] == "cooked_beef"  # phrase "steak"
    # no alias is the phrase of any class of any kind: "chicken" names the mob, never raw chicken
    phrases = {" ".join(r["phrase"]) for r in lang.raw["concepts"] if r["kind"] in ("block", "item", "entity")}
    for name in ("chicken", "cod", "rabbit", "salmon"):
        assert name not in aliases["item"] and lang.phrase("entity", name) == (name,)
    assert "tnt" not in aliases["entity_class"] and "wheat" not in aliases["block"]
    for dom, getter in (("block", concepts.block_class), ("item", concepts.item_class),
                        ("entity_class", concepts.entity_class)):
        for alias, name in aliases[dom].items():
            assert alias not in phrases, (dom, alias)
            assert alias.replace(" ", "_") in getter(name).members  # a member's own vanilla name


def test_class_phrase_ids(lang: Language) -> None:
    concepts = load_concepts()
    table = lang.class_phrase_ids("block")
    assert table.shape[0] == 256 and table.dtype == np.int32
    iron = concepts.block_class("iron_ore")
    assert [lang.word(i) for i in table[iron.id] if i] == ["iron", "ore"]
    assert [lang.word(i) for i in table[255] if i] == ["unseen"]  # draft 0.5 (draft 0.4: buried)
    items = lang.class_phrase_ids("item")
    assert [lang.word(i) for i in items[concepts.item_class("raw_iron").id] if i] == ["raw", "iron"]


def test_vocabulary_size_report(fresh: dict) -> None:
    text = V.report(fresh)
    assert f"{len(fresh['words'])} words" in text
    assert 450 <= len(fresh["words"]) <= 700  # DESIGN.md 6.4: about 600 words [est]
    assert "biomes" in text and "time words" in text and "face words" in text

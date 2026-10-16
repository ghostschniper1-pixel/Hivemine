"""Tests for the rules build (docs/contracts.md sections 1 and 2): the committed file's
invariants, spot checks against known 1.21.1 values, the build helpers, and a rebuild from
the source cache that must reproduce the committed file byte for byte."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from hivemind.rules import build_rules as B
from hivemind.rules import constants as C
from hivemind.rules.fetch_sources import verify_all

REPO = Path(__file__).resolve().parents[1]
RULES_PATH = REPO / "data" / "rules-1.21.1.json"
SOURCES_JSON = REPO / "data" / "sources.json"
SOURCES_DIR = REPO / "data" / "sources"

BLOCK_KEYS = {"id", "display_name", "hardness", "resistance", "diggable", "transparent", "emit_light",
              "filter_light", "default_state", "min_state", "max_state", "states", "mineable", "harvest_tools",
              "requires_tool", "needs_tier", "loot_table", "collision"}
ITEM_KEYS = {"id", "display_name", "stack_size", "max_damage", "food", "tool", "attack_damage", "attack_speed",
             "armor", "fuel_ticks", "equip_slot"}
RECIPE_KEYS = {"id", "type", "category", "group", "pattern", "key", "ingredients", "result", "cookingtime",
               "experience", "special"}
ENTITY_KEYS = {"id", "display_name", "width", "height", "category", "type", "max_health", "armor",
               "attack_damage", "speed"}
PICKAXES = ["wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "golden_pickaxe", "diamond_pickaxe", "netherite_pickaxe"]


@pytest.fixture(scope="module")
def raw_bytes() -> bytes:
    return RULES_PATH.read_bytes()


@pytest.fixture(scope="module")
def rules(raw_bytes: bytes) -> dict:
    return json.loads(raw_bytes)


# ---------------------------------------------------------------------------
# File-level invariants
# ---------------------------------------------------------------------------


def test_top_level_layout(rules: dict) -> None:
    assert set(rules) == {"version", "build", "blocks", "items", "tags", "recipes", "loot_tables", "entities",
                          "constants"}
    assert rules["version"] == "1.21.1"
    assert set(rules["build"]) == {"sources", "hash"}
    assert set(rules["tags"]) == {"block", "item"}


def test_file_is_canonical_and_hash_matches(rules: dict, raw_bytes: bytes) -> None:
    assert B.render(rules) == raw_bytes  # sorted keys, compact separators, trailing newline
    assert B.rules_hash(rules) == rules["build"]["hash"]


def test_build_records_the_pins(rules: dict) -> None:
    pins = json.loads(SOURCES_JSON.read_text())
    assert rules["build"]["sources"] == pins["sources"]
    s = pins["sources"]
    assert s["minecraft-data"]["commit"] == "201ca802a1b2c968b02e7fa5b31dd74e8bff3d8b"
    assert s["mcmeta-data"]["commit"] == "aacd2a457333b258d044a642f38270c8cddbc628"
    assert s["mcmeta-data"]["tag"] == "1.21.1-data"
    assert s["mcmeta-summary"]["commit"] == "7cffbcdad5fa36f22acef6bd4b869ecad4ca3fd9"
    assert s["mcmeta-summary"]["tag"] == "1.21.1-summary"
    for name in ("item_components/data.json", "blocks/data.json", "registries/data.json"):
        assert re.fullmatch(r"[0-9a-f]{64}", s["mcmeta-summary"]["files"][name])
    for path, digest in s["minecraft-data"]["files"].items():
        assert path.startswith(("data/pc/1.21.1/", "data/pc/1.20.5/", "data/pc/1.20/"))
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_blocks_schema_and_state_ids(rules: dict) -> None:
    blocks = rules["blocks"]
    assert len(blocks) > 1000
    next_state = 0
    for name, b in sorted(blocks.items(), key=lambda kv: kv[1]["min_state"]):
        assert set(b) == BLOCK_KEYS, name
        assert b["min_state"] == next_state, name  # state ids are contiguous across blocks
        n = b["max_state"] - b["min_state"] + 1
        assert math.prod(len(s["values"]) for s in b["states"]) == n, name
        assert b["min_state"] <= b["default_state"] <= b["max_state"], name
        col = b["collision"]
        assert set(col) == {"shapes", "by_state"} and col["shapes"], name
        if col["by_state"] is not None:
            assert len(col["by_state"]) == n and max(col["by_state"]) < len(col["shapes"]), name
        for shape in col["shapes"]:
            assert all(len(box) == 6 for box in shape), name
        next_state = b["max_state"] + 1
    assert sorted(b["id"] for b in blocks.values()) == list(range(len(blocks)))


def test_block_harvest_fields_are_consistent(rules: dict) -> None:
    items = rules["items"]
    for name, b in rules["blocks"].items():
        assert set(b["mineable"]) <= {"axe", "hoe", "pickaxe", "shovel"}, name
        assert b["requires_tool"] == (b["harvest_tools"] is not None), name
        for tool in b["harvest_tools"] or []:
            assert items[tool]["tool"] is not None, (name, tool)
        assert b["needs_tier"] in (None, "stone", "iron", "diamond"), name
        if b["needs_tier"] is not None and name not in B.TOLERATED_HARVEST:
            assert b["requires_tool"], name
        if b["loot_table"] is not None:
            assert b["loot_table"] in rules["loot_tables"], name


def test_items_schema(rules: dict) -> None:
    for name, it in rules["items"].items():
        assert set(it) == ITEM_KEYS, name
        assert it["stack_size"] in (1, 16, 64), name
        if it["max_damage"] is not None:
            assert it["stack_size"] == 1, name
        if it["tool"] is not None:
            assert set(it["tool"]) == {"kind", "tier", "speed", "rules", "damage_per_block"}, name
            for rule in it["tool"]["rules"]:
                blocks = rule["blocks"]
                if isinstance(blocks, str):
                    assert blocks.startswith("#") and blocks[1:] in rules["tags"]["block"], name
                else:
                    assert all(b in rules["blocks"] for b in blocks), name
        if it["food"] is not None:
            assert it["food"]["nutrition"] > 0 and it["food"]["saturation"] >= 0, name
        if it["armor"] is not None:
            assert it["equip_slot"] == it["armor"]["slot"], name


def test_tags_are_fully_resolved(rules: dict) -> None:
    for kind, universe in (("block", rules["blocks"]), ("item", rules["items"])):
        tags = rules["tags"][kind]
        assert len(tags) > 100
        for tag, members in tags.items():
            assert members == sorted(set(members)), tag
            assert all(m in universe for m in members), tag  # no '#refs' and no unknown names
    assert "oak_log" in rules["tags"]["block"]["logs"]  # logs -> #oak_logs -> oak_log (nested)
    assert len(rules["tags"]["item"]["planks"]) == 11


def test_recipes(rules: dict) -> None:
    recipes = rules["recipes"]
    ids = [r["id"] for r in recipes]
    assert ids == sorted(set(ids))
    for r in recipes:
        assert set(r) == RECIPE_KEYS, r["id"]
        if r["result"] is not None:
            assert r["result"]["item"] in rules["items"] and r["result"]["count"] >= 1, r["id"]
        slots = list((r["key"] or {}).values()) + list(r["ingredients"] or [])
        for alts in slots:
            assert alts and all(a in rules["items"] for a in alts), r["id"]
        if r["type"] in B.COOKING_TYPES:
            assert r["cookingtime"] > 0 and r["experience"] is not None, r["id"]
        if r["type"] == "crafting_shaped":
            chars = {ch for row in r["pattern"] for ch in row} - {" "}
            assert chars == set(r["key"]), r["id"]
        assert r["special"] == (r["type"].startswith("crafting_special_") or r["type"] in B.SPECIAL_RECIPE_TYPES)
    assert sum(r["type"] == "crafting_shaped" for r in recipes) > 600


def test_loot_tables_are_verbatim_vanilla(rules: dict) -> None:
    loot = rules["loot_tables"]
    assert all(k.startswith(("blocks/", "entities/")) for k in loot)
    assert loot["blocks/stone"]["type"] == "minecraft:block"
    assert loot["entities/zombie"]["type"] == "minecraft:entity"
    assert "entities/sheep/white" in loot


def test_entities(rules: dict) -> None:
    entities = rules["entities"]
    assert len(entities) == 130
    for name, e in entities.items():
        assert set(e) == ENTITY_KEYS, name
    for mob in C.MOBS:
        assert entities[mob]["max_health"] is not None, mob


def test_constants_carry_source_labels(rules: dict) -> None:
    pattern = re.compile(C.SOURCE_LABELS_PATTERN)
    for name, c in rules["constants"].items():
        assert set(c) == {"value", "source"}, name
        assert pattern.match(c["source"]), name
    required = [
        "exhaustion_sprint_per_m", "exhaustion_jump", "exhaustion_break_block", "exhaustion_attack",
        "exhaustion_threshold", "regen_saturated_interval_ticks", "regen_interval_ticks", "regen_min_food",
        "starvation_interval_ticks", "day_length_ticks", "night_start_tick", "bed_use_start_tick",
        "hostile_surface_spawn_start_tick", "attack_cooldown_numerator_ticks", "attack_multiplier_min",
        "critical_multiplier", "invulnerability_ticks", "spawn_cap_monster", "spawn_cap_creature",
        "spawn_min_player_distance", "despawn_instant_distance", "despawn_idle_ticks", "reach_block",
        "reach_entity", "walk_speed_m_per_s", "sprint_speed_m_per_s", "smelt_ticks", "eat_ticks",
        "break_delay_ticks", "chat_min_interval_ticks",
    ]
    for name in required:
        assert name in rules["constants"], name
    k = rules["constants"]
    assert (k["reach_block"]["value"], k["reach_entity"]["value"]) == (4.5, 3.0)
    assert (k["walk_speed_m_per_s"]["value"], k["sprint_speed_m_per_s"]["value"]) == (4.317, 5.612)
    assert (k["smelt_ticks"]["value"], k["eat_ticks"]["value"], k["break_delay_ticks"]["value"]) == (200, 32, 6)
    assert k["day_length_ticks"] == {"value": 24000, "source": "[15]"}


# ---------------------------------------------------------------------------
# Spot checks against known 1.21.1 values
# ---------------------------------------------------------------------------


def test_stone(rules: dict) -> None:
    stone = rules["blocks"]["stone"]
    assert stone["hardness"] == 1.5
    assert stone["mineable"] == ["pickaxe"]
    assert stone["requires_tool"] and stone["needs_tier"] is None
    assert stone["harvest_tools"] == sorted(PICKAXES)


def test_tier_gated_ores(rules: dict) -> None:
    iron_ore = rules["blocks"]["iron_ore"]
    assert iron_ore["needs_tier"] == "stone"
    expected = ["stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe"]
    assert iron_ore["harvest_tools"] == sorted(expected)
    obsidian = rules["blocks"]["obsidian"]
    assert obsidian["hardness"] == 50.0
    assert obsidian["needs_tier"] == "diamond"
    assert obsidian["harvest_tools"] == ["diamond_pickaxe", "netherite_pickaxe"]
    assert rules["blocks"]["bedrock"]["hardness"] == -1.0


def test_oak_log(rules: dict) -> None:
    log = rules["blocks"]["oak_log"]
    assert log["hardness"] == 2.0
    assert log["mineable"] == ["axe"]
    assert log["requires_tool"] is False and log["harvest_tools"] is None


@pytest.mark.parametrize("tier,speed", [("wooden", 2.0), ("stone", 4.0), ("iron", 6.0), ("diamond", 8.0),
                                        ("netherite", 9.0), ("golden", 12.0)])
def test_tool_speeds(rules: dict, tier: str, speed: float) -> None:
    for kind in ("pickaxe", "axe", "shovel", "hoe"):
        tool = rules["items"][f"{tier}_{kind}"]["tool"]
        assert tool["kind"] == kind and tool["speed"] == speed


def test_attack_modifiers(rules: dict) -> None:
    pick = rules["items"]["iron_pickaxe"]
    assert pick["attack_damage"] == 3.0
    assert pick["attack_speed"] == pytest.approx(-2.8, abs=1e-6)
    swords = [rules["items"][f"{t}_sword"]["attack_damage"] for t in ("wooden", "stone", "iron", "diamond")]
    assert swords == [3.0, 4.0, 5.0, 6.0]  # + base 1 = DESIGN 4.3's 4, 5, 6, 7
    assert rules["items"]["diamond_sword"]["attack_speed"] == pytest.approx(-2.4, abs=1e-6)  # 1.6 attacks/s


def test_food_and_stacks(rules: dict) -> None:
    bread = rules["items"]["bread"]
    assert (bread["food"]["nutrition"], bread["food"]["saturation"]) == (5, 6.0)
    assert rules["items"]["dried_kelp"]["food"]["eat_seconds"] == 0.8
    assert rules["items"]["mushroom_stew"]["food"]["converts_to"] == "bowl"
    assert rules["items"]["ender_pearl"]["stack_size"] == 16
    assert rules["items"]["iron_pickaxe"]["max_damage"] == 250


def test_smelting_raw_iron(rules: dict) -> None:
    smelt = [r for r in rules["recipes"] if r["type"] == "smelting" and r["result"]["item"] == "iron_ingot"
             and r["ingredients"] == [["raw_iron"]]]
    assert len(smelt) == 1 and smelt[0]["cookingtime"] == 200 and smelt[0]["experience"] == 0.7


def test_stick_recipe_resolves_planks_tag(rules: dict) -> None:
    stick = next(r for r in rules["recipes"] if r["id"] == "stick")
    assert stick["pattern"] == ["#", "#"] and stick["result"] == {"item": "stick", "count": 4}
    assert stick["key"]["#"] == rules["tags"]["item"]["planks"]


def _find(node, pred):
    if isinstance(node, dict):
        if pred(node):
            yield node
        for v in node.values():
            yield from _find(v, pred)
    elif isinstance(node, list):
        for v in node:
            yield from _find(v, pred)


def test_gravel_and_coal_ore_loot(rules: dict) -> None:
    gravel = rules["loot_tables"]["blocks/gravel"]
    flint = list(_find(gravel, lambda n: n.get("name") == "minecraft:flint"))
    assert len(flint) == 1
    bonus = [c for c in flint[0]["conditions"] if c["condition"] == "minecraft:table_bonus"]
    assert bonus[0]["chances"][0] == 0.1  # 10% without Fortune, not minecraft-data's 1/3
    coal = rules["loot_tables"]["blocks/coal_ore"]
    assert list(_find(coal, lambda n: n.get("name") == "minecraft:coal"))
    assert rules["blocks"]["coal_ore"]["loot_table"] == "blocks/coal_ore"
    assert rules["blocks"]["wall_torch"]["loot_table"] == "blocks/torch"


def test_fuel_and_armor(rules: dict) -> None:
    items = rules["items"]
    fuel = {n: items[n]["fuel_ticks"] for n in ("coal", "charcoal", "coal_block", "lava_bucket", "oak_log",
                                                 "oak_planks", "stick", "cobblestone", "crimson_planks")}
    assert fuel == {"coal": 1600, "charcoal": 1600, "coal_block": 16000, "lava_bucket": 20000, "oak_log": 300,
                    "oak_planks": 300, "stick": 100, "cobblestone": None, "crimson_planks": None}
    points = {m: [items[f"{m}_{p}"]["armor"]["points"] for p in ("helmet", "chestplate", "leggings", "boots")]
              for m in ("leather", "golden", "chainmail", "iron", "diamond", "netherite")}
    assert points == {"leather": [1, 3, 2, 1], "golden": [2, 5, 3, 1], "chainmail": [2, 5, 4, 1],
                      "iron": [2, 6, 5, 2], "diamond": [3, 8, 6, 3], "netherite": [3, 8, 6, 3]}
    assert items["diamond_chestplate"]["armor"]["toughness"] == 2.0
    assert items["netherite_boots"]["armor"]["toughness"] == 3.0
    assert items["netherite_boots"]["armor"]["knockback_resistance"] == 0.1
    assert items["iron_helmet"]["equip_slot"] == "head"
    k = rules["constants"]
    assert k["fuel:coal"] == {"value": 1600, "source": "[20]"}
    assert k["armor:iron"]["source"] == "[17]" and k["armor:iron"]["value"]["points"] == [2, 6, 5, 2]


def test_zombie(rules: dict) -> None:
    z = rules["entities"]["zombie"]
    assert (z["max_health"], z["armor"]) == (20.0, 2.0)
    assert (z["width"], z["height"]) == (0.6, 1.95)
    assert rules["constants"]["mob:zombie:max_health"]["source"] == "[19]"
    assert rules["constants"]["mob:zombie:attack_damage"]["source"] == "[mem]"


# ---------------------------------------------------------------------------
# Build helpers (no sources needed)
# ---------------------------------------------------------------------------


def test_trim_pattern() -> None:
    assert B.trim_pattern([" # ", " I "]) == ["#", "I"]
    assert B.trim_pattern(["   ", "#X ", "   "]) == ["#X"]
    assert B.trim_pattern(["# #", "###"]) == ["# #", "###"]  # interior gaps stay


def test_resolve_tags_nested_and_errors() -> None:
    raw = {"a": {"values": ["minecraft:x", "#minecraft:b"]}, "b": {"values": ["y", {"id": "#c", "required": False}]}}
    assert B.resolve_tags(raw, {"x", "y"}, "item") == {"a": ["x", "y"], "b": ["y"]}
    with pytest.raises(B.BuildError):
        B.resolve_tags({"a": {"values": ["#a"]}}, set(), "item")  # cycle
    with pytest.raises(B.BuildError):
        B.resolve_tags({"a": {"values": ["z"]}}, {"x"}, "item")  # unknown member


def test_tool_rule_evaluation_first_match_wins() -> None:
    tags = {"incorrect": frozenset({"obsidian"}), "mineable/pickaxe": frozenset({"stone", "obsidian"})}
    rules = [B.normalize_tool_rule({"blocks": "#minecraft:incorrect", "correct_for_drops": False}),
             B.normalize_tool_rule({"blocks": "#minecraft:mineable/pickaxe", "correct_for_drops": True, "speed": 6.0})]
    assert B.tool_speed(rules, "stone", tags) == 6.0
    assert B.tool_speed(rules, "obsidian", tags) == 6.0  # the incorrect rule has no speed
    assert B.tool_speed(rules, "dirt", tags) == 1.0
    assert B.tool_correct_for_drops(rules, "stone", tags) is True
    assert B.tool_correct_for_drops(rules, "obsidian", tags) is False
    assert B.tool_kind_and_tier("golden_axe") == ("axe", "gold")
    assert B.tool_kind_and_tier("shears") == ("shears", None)


def test_first_alternative_follows_definition_order() -> None:
    raw_tags = {"oak_logs": {"values": ["minecraft:oak_log", "minecraft:oak_wood"]},
                "logs": {"values": ["#minecraft:oak_logs", "minecraft:birch_log"]}}
    assert B.first_alternative({"tag": "minecraft:logs"}, raw_tags) == "oak_log"
    assert B.first_alternative([{"item": "minecraft:coal"}, {"item": "minecraft:charcoal"}], raw_tags) == "coal"


def test_hash_ignores_build_field() -> None:
    a = {"version": "1.21.1", "x": {"b": 1, "a": [1.5, None]}}
    b = dict(a, build={"hash": "whatever"})
    assert B.rules_hash(a) == B.rules_hash(b)
    assert B.canonical_json(a["x"]) == '{"a":[1.5,null],"b":1}'


# ---------------------------------------------------------------------------
# Rebuild from the source cache: cross-checks pass and the file is reproduced exactly
# ---------------------------------------------------------------------------


def _sources_available() -> bool:
    return SOURCES_DIR.is_dir() and not verify_all(SOURCES_JSON, SOURCES_DIR)


@pytest.mark.skipif(not _sources_available(),
                    reason="data/sources/ missing or not matching the pins; run python -m hivemind.rules.fetch_sources")
def test_rebuild_from_cache_reproduces_committed_file(raw_bytes: bytes) -> None:
    rules, checks, _notes = B.build(SOURCES_DIR, SOURCES_JSON, verify=False)  # verified by the skipif
    for chk in checks:
        assert chk.compared > 0, chk.name
        if chk.fatal:
            assert chk.mismatches == [], chk.line()
    names = {c.name for c in checks}
    for expected in ("stack sizes", "durability", "food values", "harvest tools vs tool rules + tags",
                     "tool speeds: materials.json vs tool rules", "recipes: minecraft-data vs mcmeta (crafting)"):
        assert expected in names
    assert rules["build"]["hash"] == json.loads(raw_bytes)["build"]["hash"]
    assert B.render(rules) == raw_bytes

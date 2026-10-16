"""Tests for the rules build (docs/contracts.md sections 1 and 2): the committed file's
invariants, spot checks against known 1.21.1 values, the build helpers, every cross-check on
small planted data, and a rebuild from the source cache that must reproduce the committed
file byte for byte. The rebuild tests need data/sources/ (git-ignored; fetched by
``python -m hivemind.rules.fetch_sources``) and are skipped without it; the planted-data
tests run everywhere."""

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
                          "biomes", "constants"}
    assert rules["version"] == "1.21.1"
    assert set(rules["build"]) == {"sources", "hash"}
    assert set(rules["tags"]) == set(B.TAG_KINDS) == {"block", "item", "entity_type", "enchantment", "worldgen/biome"}


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
    for kind, universe, n_min in (("block", rules["blocks"], 100), ("item", rules["items"], 100),
                                  ("entity_type", rules["entities"], 30), ("worldgen/biome", rules["biomes"], 50)):
        tags = rules["tags"][kind]
        assert len(tags) > n_min, kind
        for tag, members in tags.items():
            assert members == sorted(set(members)), tag
            assert all(m in universe for m in members), tag  # no '#refs' and no unknown names
    for tag, members in rules["tags"]["enchantment"].items():  # no enchantment table: names only
        assert members == sorted(set(members)) and all(re.fullmatch(r"[a-z_]+", m) for m in members), tag
    assert "oak_log" in rules["tags"]["block"]["logs"]  # logs -> #oak_logs -> oak_log (nested)
    assert len(rules["tags"]["item"]["planks"]) == 11
    # the tags that loot conditions use (entities/wither_skeleton, every meat drop)
    assert rules["tags"]["entity_type"]["skeletons"] == ["bogged", "skeleton", "skeleton_horse", "stray",
                                                         "wither_skeleton"]
    assert rules["tags"]["enchantment"]["smelts_loot"] == ["fire_aspect"]
    assert "exclusive_set/armor" in rules["tags"]["enchantment"]  # subdirectories keep their path


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
    assert all(k.startswith(B.LOOT_TABLE_PREFIXES) or k in B.LOOT_TABLE_FILES for k in loot)
    assert loot["blocks/stone"]["type"] == "minecraft:block"
    assert loot["entities/zombie"]["type"] == "minecraft:entity"
    assert "entities/sheep/white" in loot
    fishing = sorted(k for k in loot if k.startswith("gameplay/"))
    assert fishing == ["gameplay/fishing", "gameplay/fishing/fish", "gameplay/fishing/junk", "gameplay/fishing/treasure"]
    assert loot["gameplay/fishing/fish"]["type"] == "minecraft:fishing"


def test_loot_table_references_resolve_inside_the_file(rules: dict) -> None:
    loot = rules["loot_tables"]
    refs = {(path, ref) for path, table in loot.items() for ref in B.loot_references(table)}
    assert all(ref in loot for _, ref in refs), sorted(r for r in refs if r[1] not in loot)
    assert ("entities/guardian", "gameplay/fishing/fish") in refs
    assert ("entities/elder_guardian", "gameplay/fishing/fish") in refs
    assert ("gameplay/fishing", "gameplay/fishing/junk") in refs


def test_biomes(rules: dict) -> None:
    biomes = rules["biomes"]
    assert len(biomes) == 64
    assert sorted(b["id"] for b in biomes.values()) == list(range(64))
    assert [n for n, _ in sorted(biomes.items(), key=lambda kv: kv[1]["id"])] == sorted(biomes)  # registry order
    for name, b in biomes.items():
        assert set(b) == {"id", "display_name", "dimension"}, name
        assert b["dimension"] in ("overworld", "nether", "end", None), name
        assert (b["dimension"] == "overworld") == (name in rules["tags"]["worldgen/biome"]["is_overworld"]), name
    assert biomes["birch_forest"] == {"id": 4, "display_name": "Birch Forest", "dimension": "overworld"}
    assert biomes["plains"]["display_name"] == "Plains"
    assert biomes["nether_wastes"]["dimension"] == "nether" and biomes["the_end"]["dimension"] == "end"
    assert biomes["the_void"]["dimension"] is None  # in no dimension tag (B.TOLERATED_BIOME_DIMENSION)
    dims = [b["dimension"] for b in biomes.values()]
    assert (dims.count("overworld"), dims.count("nether"), dims.count("end")) == (53, 5, 5)
    assert {"deep_dark", "lush_caves", "dripstone_caves"} <= set(rules["tags"]["worldgen/biome"]["is_overworld"])


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


def test_mechanics_constants_from_memory(rules: dict) -> None:
    """The five values mechanics.py once kept as code defaults are in the file, labelled [mem];
    the draft-0.3 fatigue formula keeps its [13] label (DESIGN 4.3) under names that say it is
    the draft's, not the game's."""
    k = rules["constants"]
    assert k["exhaustion_cap"] == {"value": 40.0, "source": "[mem]"}
    assert k["break_mining_fatigue_table"] == {"value": [0.3, 0.09, 0.0027, 0.00081], "source": "[mem]"}
    assert k["invulnerability_reset_ticks"] == {"value": 20, "source": "[mem]"}
    assert k["max_sky_light"] == {"value": 15, "source": "[mem]"}
    assert k["sky_darken_not_day_min"] == {"value": 4, "source": "[mem]"}
    assert k["draft03_mining_fatigue_base"] == {"value": 0.3, "source": "[13]"}
    assert k["draft03_mining_fatigue_max_level"] == {"value": 4, "source": "[13]"}
    # the table agrees with the draft formula at levels 1-2 only
    table, base = k["break_mining_fatigue_table"]["value"], k["draft03_mining_fatigue_base"]["value"]
    assert [math.isclose(t, base ** (i + 1)) for i, t in enumerate(table)] == [True, True, False, False]
    # the only fatigue values under game-sounding names are the game's table
    assert sorted(n for n in k if "fatigue" in n and not n.startswith("draft03_")) == ["break_mining_fatigue_table"]
    # the old names, which made a wrong formula and a non-night threshold look like game rules, are gone
    for old in ("break_mining_fatigue_base", "break_mining_fatigue_max_level", "sky_darken_night_min"):
        assert old not in k, old
    # the reset time leaves [18]'s 10 ticks of invulnerability
    assert k["invulnerability_reset_ticks"]["value"] - k["invulnerability_ticks"]["value"] == 10


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
    # each armor field has its own label: DESIGN 4.3 quotes [17] only for iron and diamond
    # points and diamond's toughness
    assert k["armor:iron:points"] == {"value": [2, 6, 5, 2], "source": "[17]"}
    assert k["armor:diamond:points"] == {"value": [3, 8, 6, 3], "source": "[17]"}
    assert k["armor:diamond:toughness"] == {"value": 2.0, "source": "[17]"}
    for name in ("armor:iron:toughness", "armor:iron:knockback_resistance", "armor:diamond:knockback_resistance",
                 "armor:leather:points", "armor:netherite:toughness"):
        assert k[name]["source"] == "[mem]", name
    assert not any(n.startswith("armor:") and n.count(":") == 1 and n != "armor:turtle_helmet" for n in k)


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
# Cross-checks on small planted data (no sources needed): each check passes on agreeing
# sources and reports a planted disagreement
# ---------------------------------------------------------------------------


def _src(md: dict | None = None, **parts) -> B.Sources:
    base = {"items": [], "foods": [], "blocks": [], "materials": {}, "recipes": {}, "entities": [], "language": {},
            "biomes": [], "block_loot": [], "entity_loot": []}
    return B.Sources(pins={}, md={**base, **(md or {})}, tags=parts.get("tags", {k: {} for k in B.TAG_KINDS}),
                     recipes=parts.get("recipes", {}), loot=parts.get("loot", {}),
                     components=parts.get("components", {}), summary_blocks=parts.get("summary_blocks", {}),
                     entity_registry=parts.get("entity_registry", []),
                     enchantment_registry=parts.get("enchantment_registry", []),
                     biome_registry=parts.get("biome_registry", []))


def test_check_items_detects_planted_mismatches() -> None:
    def checks(md_stack: int, md_durability: int | None, md_food: int) -> dict[str, list[str]]:
        md = {"items": [{"name": "bread", "stackSize": 64}, {"name": "iron_pickaxe", "stackSize": 1,
                                                               "maxDurability": md_durability}],
              "foods": [{"name": "bread", "foodPoints": md_food, "saturation": 6.0}]}
        md["items"][0]["stackSize"] = md_stack
        comps = {"bread": {"minecraft:max_stack_size": 64, "minecraft:food": {"nutrition": 5, "saturation": 6.0}},
                 "iron_pickaxe": {"minecraft:max_stack_size": 1, "minecraft:max_damage": 250}}
        return {c.name: c.mismatches for c in B.check_items(_src(md, components=comps))}

    assert all(not m for m in checks(64, 250, 5).values())
    bad = checks(16, 251, 4)
    assert bad["stack sizes"] and bad["durability"] and bad["food values"]


def test_check_block_states_detects_a_wrong_default() -> None:
    states = [{"name": "face", "type": "enum", "values": ["floor", "wall", "ceiling"]},
              {"name": "powered", "type": "bool", "values": ["true", "false"]}]
    summary = {"lever": [{"face": ["floor", "wall", "ceiling"], "powered": ["true", "false"]},
                         {"face": "wall", "powered": "false"}]}
    for default, ok in ((13, True), (12, False)):  # wall = 1, false = 1: offset 1 * 2 + 1 = 3
        blocks = {"lever": {"states": states, "min_state": 10, "default_state": default}}
        assert (not B.check_block_states(_src(summary_blocks=summary), blocks).mismatches) is ok


def _pickaxe(tier_tag: str, speed: float) -> dict:
    rules = [B.normalize_tool_rule({"blocks": f"#minecraft:{tier_tag}", "correct_for_drops": False}),
             B.normalize_tool_rule({"blocks": "#minecraft:mineable/pickaxe", "correct_for_drops": True,
                                    "speed": speed})]
    return {"tool": {"kind": "pickaxe", "rules": rules}}


def test_check_harvest_tools_detects_planted_mismatch() -> None:
    items = {"wooden_pickaxe": _pickaxe("incorrect_for_wooden_tool", 2.0),
             "stone_pickaxe": _pickaxe("incorrect_for_stone_tool", 4.0)}
    tags = {"incorrect_for_wooden_tool": ["iron_ore"], "incorrect_for_stone_tool": [], "mineable/pickaxe": ["iron_ore"]}
    for harvest, ok in ((["stone_pickaxe"], True), (["stone_pickaxe", "wooden_pickaxe"], False)):
        blocks = {"iron_ore": {"requires_tool": True, "harvest_tools": harvest, "needs_tier": "stone",
                               "mineable": ["pickaxe"]}}
        assert (not B.check_harvest_tools(blocks, items, tags).mismatches) is ok


def test_check_tool_speeds_detects_planted_mismatch() -> None:
    items = {"wooden_pickaxe": _pickaxe("incorrect_for_wooden_tool", 2.0)}
    tags = {"incorrect_for_wooden_tool": [], "mineable/pickaxe": ["stone"]}
    for md_speed, ok in ((2.0, True), (4.0, False)):
        md = {"blocks": [{"name": "stone", "material": "mineable/pickaxe"}],
              "items": [{"id": 1, "name": "wooden_pickaxe"}], "materials": {"mineable/pickaxe": {"1": md_speed}}}
        chk = B.check_tool_speeds(_src(md), items, tags, {"stone": {"mineable": ["pickaxe"]}})
        assert chk.compared == 1 and (not chk.mismatches) is ok


def test_check_recipes_detects_planted_mismatches() -> None:
    md_items = [{"id": 1, "name": "stick"}, {"id": 2, "name": "oak_planks"}, {"id": 3, "name": "birch_planks"},
                {"id": 4, "name": "flint_and_steel"}, {"id": 5, "name": "iron_ingot"}, {"id": 6, "name": "flint"},
                {"id": 7, "name": "bone"}]
    recipes = [{"id": "stick", "type": "crafting_shaped", "pattern": ["#", "#"],
                "key": {"#": ["birch_planks", "oak_planks"]}, "ingredients": None,
                "result": {"item": "stick", "count": 4}},
               {"id": "flint_and_steel", "type": "crafting_shapeless", "pattern": None, "key": None,
                "ingredients": [["iron_ingot"], ["bone", "flint"]], "result": {"item": "flint_and_steel", "count": 1}}]
    raw = {"flint_and_steel": {"ingredients": [{"item": "minecraft:iron_ingot"},
                                               [{"item": "minecraft:flint"}, {"item": "minecraft:bone"}]]}}

    def check(sticks: list[list[list[int]]], fs_count: int) -> B.CrossCheck:
        md_recipes = {"1": [{"result": {"id": 1, "count": 4}, "inShape": shape} for shape in sticks],
                      "4": [{"result": {"id": 4, "count": fs_count}, "ingredients": [5, 6]}]}
        return B.check_recipes(_src({"items": md_items, "recipes": md_recipes}, recipes=raw), recipes)

    good = check([[[2], [2]], [[3], [3]]], 1)
    assert not good.mismatches
    assert any("1 non-first ingredient alternatives in 1 recipes are NOT compared" in t for t in good.tolerated)
    assert any("lacks ['birch_planks']" in m for m in check([[[2], [2]]], 1).mismatches)  # a tag member missing
    assert any("shapeless" in m for m in check([[[2], [2]], [[3], [3]]], 2).mismatches)  # result count


def test_check_entities_detects_a_planted_name() -> None:
    for name, ok in (("Zombie", True), ("Zombi", False)):
        src = _src({"entities": [{"name": "zombie", "displayName": name}],
                    "language": {"entity.minecraft.zombie": "Zombie"}}, entity_registry=["zombie"])
        assert (not B.check_entities(src).mismatches) is ok


def test_build_biomes_ids_names_and_dimensions() -> None:
    lang = {"biome.minecraft.beach": "Beach", "biome.minecraft.the_void": "The Void"}
    src = _src({"language": lang}, biome_registry=["beach", "the_void"])
    tags = {"is_overworld": ["beach"], "is_nether": [], "is_end": []}
    notes: list[str] = []
    assert B.build_biomes(src, tags, notes) == {"beach": {"id": 0, "display_name": "Beach", "dimension": "overworld"},
                                                "the_void": {"id": 1, "display_name": "The Void", "dimension": None}}
    with pytest.raises(B.BuildError, match="several dimension tags"):
        B.build_biomes(src, {**tags, "is_end": ["beach"]}, notes)
    with pytest.raises(B.BuildError, match="no English name"):
        B.build_biomes(_src({"language": {}}, biome_registry=["beach"]), tags, notes)


def test_check_biomes_detects_planted_mismatches() -> None:
    biomes = {"beach": {"id": 0, "display_name": "Beach", "dimension": "overworld"},
              "the_void": {"id": 1, "display_name": "The Void", "dimension": None}}
    md = [{"id": 0, "name": "beach", "displayName": "Beach", "dimension": "overworld"},
          {"id": 1, "name": "the_void", "displayName": "The Void", "dimension": "overworld"}]
    chk = B.check_biomes(_src({"biomes": md}), biomes)
    assert chk.compared == 2 and not chk.mismatches and len(chk.tolerated) == 1  # the_void's dimension
    for i, field, value in ((0, "id", 5), (0, "displayName", "Shore"), (0, "dimension", "nether"), (1, "name", "x")):
        bad = [dict(b) for b in md]
        bad[i][field] = value
        assert B.check_biomes(_src({"biomes": bad}), biomes).mismatches, (field, value)


def test_select_loot_tables_follows_references() -> None:
    item = {"type": "minecraft:item", "name": "minecraft:cod"}
    loot = {"blocks/stone": {"pools": []},
            "entities/guardian": {"pools": [{"entries": [{"type": "minecraft:loot_table",
                                                          "value": "minecraft:gameplay/fishing/fish"}]}]},
            "gameplay/fishing": {"pools": [{"entries": [{"type": "minecraft:loot_table",
                                                         "value": "minecraft:gameplay/other"}]}]},
            "gameplay/fishing/fish": {"pools": [{"entries": [item]}]},
            "gameplay/other": {"pools": [{"entries": [{"type": "minecraft:loot_table", "value": {"pools": []}}]}]},
            "gameplay/unused": {"pools": []},
            "chests/village": {"pools": []}}
    notes: list[str] = []
    kept = B.select_loot_tables(loot, notes)
    assert list(kept) == ["blocks/stone", "entities/guardian", "gameplay/fishing", "gameplay/fishing/fish",
                          "gameplay/other"]  # gameplay/other only because gameplay/fishing references it
    assert "['gameplay/other']" in notes[-1]
    broken = {**loot, "entities/guardian": {"pools": [{"entries": [{"type": "minecraft:loot_table",
                                                                     "value": "minecraft:gameplay/missing"}]}]}}
    with pytest.raises(B.BuildError, match="missing table gameplay/missing"):
        B.select_loot_tables(broken, notes)
    with pytest.raises(B.BuildError, match="missing loot table gameplay/fishing"):
        B.select_loot_tables({"blocks/stone": {"pools": []}}, notes)


def test_check_display_names_detects_a_planted_name() -> None:
    for name, ok in (("Stone", True), ("Rock", False)):
        src = _src({"blocks": [{"name": "stone", "displayName": name}], "language": {"block.minecraft.stone": "Stone"}})
        assert (not B.check_display_names(src).mismatches) is ok


def test_check_derived_loot_detects_planted_mismatch() -> None:
    blocks = {"torch": {"loot_table": "blocks/torch"}, "wall_torch": {"loot_table": "blocks/torch"}}
    for wall_drops, ok in (([10], True), ([11], False)):
        src = _src({"blocks": [{"name": "torch", "drops": [10]}, {"name": "wall_torch", "drops": wall_drops}]})
        assert (not B.check_derived_loot(src, blocks).mismatches) is ok


def test_entity_loot_check_follows_table_references() -> None:
    loot = {"entities/sheep": {"pools": [{"entries": [{"type": "minecraft:item", "name": "minecraft:mutton"}]}]},
            "entities/sheep/white": {"pools": [{"entries": [{"type": "minecraft:item", "name": "minecraft:white_wool"},
                                                            {"type": "minecraft:loot_table",
                                                             "value": "minecraft:entities/sheep"}]}]}}
    src = _src({"entity_loot": [{"entity": "sheep", "drops": [{"item": "mutton"}, {"item": "white_wool"}]}]},
               loot=loot, entity_registry=["sheep"])
    chk = B.check_stale_entity_loot(src)
    assert chk.compared == 1 and not chk.mismatches and not chk.fatal


def test_fatal_mismatches_raise() -> None:
    ok, info, bad = B.CrossCheck("a"), B.CrossCheck("b", fatal=False), B.CrossCheck("c")
    info.mismatches.append("stale, info only")
    assert B.fatal_failures([ok, info]) == []
    bad.mismatches.append("x: 1 vs 2")
    assert B.fatal_failures([ok, info, bad]) == [bad]
    err = B.CrossCheckError([ok, info, bad])
    assert isinstance(err, B.BuildError) and err.checks == [ok, info, bad] and "x: 1 vs 2" in str(err)


# ---------------------------------------------------------------------------
# Rebuild from the source cache: cross-checks pass and the file is reproduced exactly
# ---------------------------------------------------------------------------


def _sources_available() -> bool:
    return SOURCES_DIR.is_dir() and not verify_all(SOURCES_JSON, SOURCES_DIR)


needs_sources = pytest.mark.skipif(
    not _sources_available(),
    reason="data/sources/ missing or not matching the pins; run python -m hivemind.rules.fetch_sources")


@needs_sources
def test_rebuild_from_cache_reproduces_committed_file(raw_bytes: bytes) -> None:
    rules, checks, _notes = B.build(SOURCES_DIR, SOURCES_JSON, verify=False)  # verified by the skipif
    for chk in checks:
        assert chk.compared > 0, chk.name
        if chk.fatal:
            assert chk.mismatches == [], chk.line()
    names = {c.name for c in checks}
    for expected in ("stack sizes", "durability", "food values", "harvest tools vs tool rules + tags",
                     "tool speeds: materials.json vs tool rules", "recipes: minecraft-data vs mcmeta (crafting)",
                     "biomes: 1.21.1 registry and tags vs minecraft-data 1.20.5"):
        assert expected in names
    assert rules["build"]["hash"] == json.loads(raw_bytes)["build"]["hash"]
    assert B.render(rules) == raw_bytes


@needs_sources
def test_build_fails_on_a_real_disagreement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without its tolerance entry, minecraft-data's map-extending recipe is a recipe mismatch."""
    monkeypatch.setattr(B, "TOLERATED_RECIPES_ONLY_IN_MD", {})
    with pytest.raises(B.CrossCheckError) as exc:
        B.build(SOURCES_DIR, SOURCES_JSON, verify=False)
    assert [c.name for c in B.fatal_failures(exc.value.checks)] == ["recipes: minecraft-data vs mcmeta (crafting)"]
    _rules, checks, _notes = B.build(SOURCES_DIR, SOURCES_JSON, verify=False, strict=False)  # main's mode
    assert B.fatal_failures(checks)

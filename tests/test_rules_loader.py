"""Tests for hivemind.rules.load_rules and the typed Rules wrapper."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from hivemind.rules import Biome, Block, Item, Rules, load_rules
from hivemind.rules.loader import DEFAULT_RULES_PATH


@pytest.fixture(scope="module")
def rules() -> Rules:
    return load_rules()


def break_ticks(rules: Rules, tool: str | None, block: str) -> int:
    """DESIGN 4.3 breaking formula without enchantments or effects (for the check values)."""
    b = rules.block(block)
    speed = rules.mining_speed(tool, block)
    divisor = 30 if rules.can_harvest(tool, block) else 100
    dmg = speed / b.hardness / divisor
    return 0 if dmg >= 1 else math.ceil(1 / dmg - 1e-9)


def test_load_is_cached_and_verified(rules: Rules) -> None:
    assert load_rules() is rules
    assert load_rules(DEFAULT_RULES_PATH) is rules
    assert rules.version == "1.21.1"
    assert rules.compute_hash() == rules.hash


def test_tampered_file_is_rejected(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_RULES_PATH.read_text())
    raw["blocks"]["stone"]["hardness"] = 0.0
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_rules(path)
    assert load_rules(path, verify=False).block("stone").hardness == 0.0


def test_copy_loads_from_explicit_path(tmp_path: Path) -> None:
    path = tmp_path / "copy.json"
    shutil.copy(DEFAULT_RULES_PATH, path)
    r = load_rules(path)
    assert isinstance(r, Rules) and r.hash == load_rules().hash


def test_typed_lookups(rules: Rules) -> None:
    stone = rules.block("minecraft:stone")
    assert isinstance(stone, Block)
    assert (stone.hardness, stone.mineable, stone.requires_tool) == (1.5, ("pickaxe",), True)
    pick = rules.item("iron_pickaxe")
    assert isinstance(pick, Item)
    assert (pick.tool.kind, pick.tool.tier, pick.tool.speed) == ("pickaxe", "iron", 6.0)
    assert pick.attack_damage == 3.0 and pick.attack_speed == pytest.approx(-2.8, abs=1e-6)
    assert rules.item("bread").food.nutrition == 5 and rules.item("bread").food.saturation == 6.0
    assert rules.item("coal").fuel_ticks == 1600
    assert rules.item("diamond_helmet").armor.points == 3
    z = rules.entity("zombie")
    assert (z.max_health, z.armor) == (20.0, 2.0)
    with pytest.raises(KeyError):
        rules.block("not_a_block")
    with pytest.raises(KeyError):
        rules.item("stone_pick")


def test_tags(rules: Rules) -> None:
    pick = rules.tag("block", "mineable/pickaxe")
    assert pick == rules.tag("block", "#minecraft:mineable/pickaxe")
    assert "stone" in pick and "oak_log" not in pick
    assert rules.in_tag("item", "oak_planks", "planks")
    assert rules.tag("block", "needs_diamond_tool") == tuple(sorted(rules.tag("block", "needs_diamond_tool")))
    with pytest.raises(KeyError):
        rules.tag("block", "no_such_tag")
    assert set(rules.tag_kinds) == {"block", "item", "entity_type", "enchantment", "worldgen/biome"}
    assert rules.in_tag("entity_type", "stray", "#minecraft:skeletons")
    assert not rules.in_tag("entity_type", "zombie", "skeletons")
    assert rules.tag("enchantment", "smelts_loot") == ("fire_aspect",)
    assert rules.in_tag("worldgen/biome", "plains", "is_overworld")
    with pytest.raises(KeyError):
        rules.tag("enchantments", "smelts_loot")  # unknown kind


def test_biomes(rules: Rules) -> None:
    b = rules.biome("minecraft:birch_forest")
    assert isinstance(b, Biome) and b == Biome("birch_forest", 4, "Birch Forest", "overworld")
    assert [x.id for x in rules.biomes.values()] == list(range(64))  # iteration in id order
    assert rules.biome("the_void").dimension is None and rules.biome("warped_forest").dimension == "nether"
    with pytest.raises(KeyError, match="pale_garden"):
        rules.biome("pale_garden")  # added in 1.21.4, not in 1.21.1


def test_recipes_for(rules: Rules) -> None:
    sticks = rules.recipes_for("stick")
    assert {r.id for r in sticks} == {"stick", "stick_from_bamboo_item"}
    plank_stick = next(r for r in sticks if r.id == "stick")
    assert plank_stick.result_count == 4 and len(plank_stick.key["#"]) == 11
    smelt = [r for r in rules.recipes_for("iron_ingot") if r.type == "smelting" and r.ingredients == (("raw_iron",),)]
    assert len(smelt) == 1 and smelt[0].cookingtime == 200
    table = rules.recipes_for("crafting_table")
    assert table[0].pattern == ("##", "##")
    assert rules.recipes_for("bedrock") == ()


def test_loot_tables(rules: Rules) -> None:
    gravel = rules.loot_table(rules.block("gravel").loot_table)
    assert gravel["type"] == "minecraft:block"
    assert "minecraft:flint" in json.dumps(gravel)
    assert "minecraft:coal" in json.dumps(rules.loot_table("blocks/coal_ore"))
    assert rules.loot_table("minecraft:entities/zombie")["type"] == "minecraft:entity"
    assert rules.loot_table("gameplay/fishing/fish")["type"] == "minecraft:fishing"  # the guardians' fish
    with pytest.raises(KeyError):
        rules.loot_table("blocks/nothing")


def test_constants(rules: Rules) -> None:
    assert rules.constant("smelt_ticks") == 200
    assert rules.constant_entry("smelt_ticks").source == "[20]"
    assert rules.constant("reach_block") == 4.5
    assert rules.constant("fuel:#logs") == 300
    with pytest.raises(KeyError):
        rules.constant("warp_speed")


def test_mining_and_harvest(rules: Rules) -> None:
    assert rules.mining_speed(None, "stone") == 1.0
    assert rules.mining_speed("wooden_pickaxe", "stone") == 2.0
    assert rules.mining_speed("golden_pickaxe", "stone") == 12.0
    assert rules.mining_speed("diamond_pickaxe", "oak_log") == 1.0
    assert rules.mining_speed("stick", "stone") == 1.0  # not a tool
    for pick in ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "golden_pickaxe", "diamond_pickaxe",
                 "netherite_pickaxe"):
        assert rules.can_harvest(pick, "stone")
    assert not rules.can_harvest(None, "stone")
    assert not rules.can_harvest("wooden_pickaxe", "iron_ore") and rules.can_harvest("stone_pickaxe", "iron_ore")
    assert not rules.can_harvest("golden_pickaxe", "iron_ore")  # gold is wood tier for drops
    assert not rules.can_harvest("iron_pickaxe", "obsidian") and rules.can_harvest("diamond_pickaxe", "obsidian")
    assert rules.can_harvest(None, "oak_log") and rules.can_harvest(None, "dirt")
    assert rules.mining_speed("iron_pickaxe", "minecraft:stone") == 6.0


def test_unknown_names_raise(rules: Rules) -> None:
    """A typo must not silently mine at bare-hand speed."""
    for call in (rules.mining_speed, rules.can_harvest):
        with pytest.raises(KeyError, match="stone_brick"):
            call("iron_pickaxe", "stone_brick")
        with pytest.raises(KeyError, match="iron_pick"):
            call("iron_pick", "stone")


def test_design_break_time_check_values(rules: Rules) -> None:
    # DESIGN.md 4.3: stone with a wooden pickaxe 23 ticks; stone by hand 150; log by hand 60,
    # with a wooden axe 30; obsidian with a diamond pickaxe 188.
    assert break_ticks(rules, "wooden_pickaxe", "stone") == 23
    assert break_ticks(rules, None, "stone") == 150
    assert break_ticks(rules, None, "oak_log") == 60
    assert break_ticks(rules, "wooden_axe", "oak_log") == 30
    assert break_ticks(rules, "diamond_pickaxe", "obsidian") == 188


def test_block_states_and_collision(rules: Rules) -> None:
    slab = rules.block("oak_slab")
    assert slab.state_values(slab.default_state) == {"type": "bottom", "waterlogged": "false"}
    assert slab.collision_boxes() == ((0.0, 0.0, 0.0, 1.0, 0.5, 1.0),)
    top = next(s for s in range(slab.min_state, slab.max_state + 1) if slab.state_values(s)["type"] == "top")
    assert slab.collision_boxes(top) == ((0.0, 0.5, 0.0, 1.0, 1.0, 1.0),)
    assert max(box[4] for box in rules.block("oak_fence").collision_boxes()) == 1.5
    assert rules.block("air").collision_boxes() == ()
    assert rules.block("short_grass").collision_boxes() == ()
    assert rules.block_of_state(rules.block("stone").default_state).name == "stone"
    grass = rules.block("grass_block")
    assert grass.state_values(grass.default_state) == {"snowy": "false"}
    with pytest.raises(ValueError):
        slab.collision_boxes(rules.block("stone").min_state)

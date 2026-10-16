"""Tests for hivemind.sim.ref.mechanics against the check values of DESIGN.md 4.3."""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import MISSING, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hivemind.rules import Rules, load_rules
from hivemind.rules import build_rules as B
from hivemind.rules.fetch_sources import verify_all
from hivemind.sim.ref import mechanics as m
from hivemind.sim.ref.mechanics import Difficulty, HungerState, MechanicsConstants, f32

REPO = Path(__file__).resolve().parents[1]
SOURCES_JSON = REPO / "data" / "sources.json"
SOURCES_DIR = REPO / "data" / "sources"


@pytest.fixture(scope="module")
def rules() -> Rules:
    return load_rules()


@pytest.fixture(scope="module")
def c(rules: Rules) -> MechanicsConstants:
    return m.constants_for(rules)


def python_float32_sum_ticks(progress: float) -> int:
    """Reference for ``client_break_ticks``: a plain loop of float32 additions."""
    total, n = 0.0, 0
    while total < 1.0:
        total = f32(total + progress)
        n += 1
    return n


# =============================================================================
# Constants
# =============================================================================


def test_constants_come_from_the_rules_file(rules: Rules, c: MechanicsConstants) -> None:
    assert c is m.default_constants()
    assert c.break_delay_ticks == rules.constant("break_delay_ticks") == 6
    assert c.smelt_ticks == rules.constant("smelt_ticks") == 200
    # The five [mem] values that used to be code defaults now come from the rules file.
    assert c.exhaustion_cap == rules.constant("exhaustion_cap") == 40.0
    assert c.break_mining_fatigue_table == tuple(rules.constant("break_mining_fatigue_table")) == \
        (0.3, 0.09, 0.0027, 0.00081)
    assert c.invulnerability_reset_ticks == rules.constant("invulnerability_reset_ticks") == 20
    assert c.max_sky_light == rules.constant("max_sky_light") == 15
    assert c.sky_darken_not_day_min == rules.constant("sky_darken_not_day_min") == 4
    assert (c.draft03_mining_fatigue_base, c.draft03_mining_fatigue_max_level) == \
        (rules.constant("draft03_mining_fatigue_base"), rules.constant("draft03_mining_fatigue_max_level")) == (0.3, 4)


def test_mechanics_constants_have_no_code_defaults() -> None:
    assert all(f.default is MISSING and f.default_factory is MISSING for f in fields(MechanicsConstants))


def test_old_constant_names_are_only_aliases(c: MechanicsConstants) -> None:
    # Kept until mining_fatigue_multiplier and sky_is_dark read the new names; never fields.
    names = {f.name for f in fields(MechanicsConstants)}
    for old, new in [("break_mining_fatigue_base", "draft03_mining_fatigue_base"),
                     ("break_mining_fatigue_max_level", "draft03_mining_fatigue_max_level"),
                     ("sky_darken_night_min", "sky_darken_not_day_min")]:
        assert old not in names and new in names
        assert getattr(c, old) == getattr(c, new)


def test_sky_darken_not_day_is_not_the_night_of_the_clock(c: MechanicsConstants) -> None:
    # The rename's reason: the threshold is Level.isDay()'s (the bed window), which is 918 ticks a
    # day longer than [15]'s night (night_start_tick..night_end_tick).
    not_day = [t for t in range(24000) if m.sky_darken(t) >= c.sky_darken_not_day_min]
    night = [t for t in range(24000) if m.is_night(t)]
    assert (not_day[0], not_day[-1], len(not_day)) == (c.bed_use_start_tick, c.bed_use_end_tick - 1, 10918)
    assert (night[0], night[-1], len(night)) == (c.night_start_tick, c.night_end_tick - 1, 10000)
    assert set(night) < set(not_day) and len(set(not_day) - set(night)) == 918


def _stub_rules(rules: Rules) -> tuple[dict, SimpleNamespace]:
    consts = dict(rules.constants)
    return consts, SimpleNamespace(constants=consts, constant=lambda name: consts[name].value)


def test_from_rules_takes_rules_values(rules: Rules) -> None:
    consts, stub = _stub_rules(rules)
    consts["exhaustion_cap"] = SimpleNamespace(value=12.5, source="[est]")
    consts["break_mining_fatigue_table"] = SimpleNamespace(value=[0.5, 0.25], source="[est]")
    c2 = MechanicsConstants.from_rules(stub)
    assert c2.exhaustion_cap == 12.5 and c2.break_mining_fatigue_table == (0.5, 0.25)


@pytest.mark.parametrize("name", ["smelt_ticks", "exhaustion_cap", "break_mining_fatigue_table",
                                  "invulnerability_reset_ticks", "max_sky_light", "sky_darken_not_day_min",
                                  "draft03_mining_fatigue_base"])
def test_from_rules_fails_loudly_on_a_missing_constant(rules: Rules, name: str) -> None:
    consts, stub = _stub_rules(rules)
    del consts[name]
    with pytest.raises(KeyError, match=name):
        MechanicsConstants.from_rules(stub)


# =============================================================================
# Breaking [13]
# =============================================================================

# block, held item, DESIGN.md 4.3 check value (ceil(1/dmg) with the game's tool speeds),
# vanilla client (float32 accumulation), unpatched Mineflayer's wait (prismarine-block digTime).
BREAK_CHECKS = [
    ("stone", "wooden_pickaxe", 23, 23, 23),
    # 1/dmg is exactly 150, but 150 float32 additions of 0.0066666668 sum to 0.99999923 < 1, so
    # the vanilla client breaks on tick 151. The check value 150 is ceil(1/dmg) in exact
    # arithmetic (the server's 70% check accepts "finished" well before either).
    ("stone", None, 150, 151, 150),
    # Same effect: 60 additions of 0.016666668 sum to 0.9999997 in float32.
    ("oak_log", None, 60, 61, 60),
    # Here the float32 sum overshoots: 30 additions of 0.033333335 give 1.0000004.
    ("oak_log", "wooden_axe", 30, 30, 30),
    # Obsidian is tier-gated: minecraft-data's material gives a diamond pickaxe speed 1, not 8,
    # so Mineflayer waits 1 / 50 / 30 -> 1500 ticks for what the game breaks in 188.
    ("obsidian", "diamond_pickaxe", 188, 188, 1500),
]


@pytest.mark.parametrize("block,held,design,client,mineflayer", BREAK_CHECKS)
def test_break_check_values(rules: Rules, block: str, held: str | None, design: int, client: int,
                            mineflayer: int) -> None:
    t = m.break_ticks_for(rules, block, held)
    assert t.design_ticks == design
    assert t.ticks == client
    assert t.mineflayer_ticks == mineflayer
    assert t.ticks == python_float32_sum_ticks(t.progress_per_tick)
    # The server accepts "finished" at progress * (k + 1) >= 0.7 (float32), and not one tick earlier.
    k = t.server_accept_ticks
    assert f32(t.progress_per_tick * (k + 1)) >= f32(0.7) > f32(t.progress_per_tick * k)
    assert k < t.ticks


def test_mineflayer_double_rounding(rules: Rules) -> None:
    # 2 / 3 / 30 is 1/45 exactly, but in doubles 1 / dmg is just above 45, so the design formula
    # (and Mineflayer, which agrees on coal ore) gives 46 ticks while the vanilla client's
    # float32 sum reaches 1.0 after 45.
    t = m.break_ticks_for(rules, "coal_ore", "wooden_pickaxe")
    assert (t.ticks, t.design_ticks, t.mineflayer_ticks) == (45, 46, 46)
    assert 1.0 / (2.0 / 3.0 / 30.0) > 45.0


def test_stone_by_hand_drops_nothing(rules: Rules) -> None:
    assert not m.can_harvest(rules, "stone", None)
    assert m.can_harvest(rules, "stone", "wooden_pickaxe")
    assert m.can_harvest(rules, "oak_log", None)
    assert not m.can_harvest(rules, "obsidian", "iron_pickaxe")
    # Wrong tier still mines at the tool's speed, but at the /100 divisor: 6 / 50 / 100.
    t = m.break_ticks_for(rules, "obsidian", "iron_pickaxe")
    assert t.progress_per_tick == f32(f32(6.0 / 50.0) / 100.0)
    assert t.design_ticks == 834
    assert t.mineflayer_ticks == 5000  # minecraft-data's material: speed 1, 1 / 50 / 100


# Unpatched Mineflayer's dig wait: prismarine-block 1.23.0 Block.digTime(itemId, false, inWater,
# notOnGround, enchantments, effects) / 50, with prismarine-registry 1.12.0 and minecraft-data
# 3.117.0 (the versions bridge/package-lock.json locks), run under Node 22 on 2026-10-16 for every
# 1.21.1 block x 32 tools x {ground, water, air}; all 101,760 values equal mineflayer_ticks. A sample:
# (block, held, in_water, on_ground, efficiency, haste, mining_fatigue) -> ticks
MINEFLAYER_DIGTIME = [
    ("iron_ore", "stone_pickaxe", False, True, 0, 0, 0, 91),  # the game: 23
    ("iron_ore", "iron_pickaxe", False, True, 0, 0, 0, 91),
    ("iron_ore", "wooden_pickaxe", False, True, 0, 0, 0, 150),  # wooden is the one md speed 2
    ("iron_ore", "wooden_axe", False, True, 0, 0, 0, 150),  # the game: 300 (axe speed 1)
    ("iron_ore", "golden_pickaxe", False, True, 0, 0, 0, 300),
    ("iron_ore", None, False, True, 0, 0, 0, 300),
    ("diamond_ore", "iron_pickaxe", False, True, 0, 0, 0, 91),  # the game: 15
    ("deepslate_iron_ore", "stone_pickaxe", False, True, 0, 0, 0, 135),  # the game: 34
    ("iron_block", "stone_pickaxe", False, True, 0, 0, 0, 150),
    ("obsidian", "diamond_pickaxe", False, True, 0, 0, 0, 1500),
    ("obsidian", "iron_pickaxe", False, True, 0, 0, 0, 5000),
    ("moss_carpet", "wooden_hoe", False, True, 0, 0, 0, 3),  # md material "plant" has no hoes
    ("coal_ore", "wooden_pickaxe", False, True, 0, 0, 0, 46),
    ("stone", "wooden_pickaxe", True, True, 0, 0, 0, 113),
    ("stone", "wooden_pickaxe", False, False, 0, 0, 0, 113),
    ("stone", "wooden_pickaxe", False, True, 3, 0, 0, 4),
    ("stone", "wooden_pickaxe", False, True, 0, 1, 0, 19),
    ("stone", "wooden_pickaxe", False, True, 0, 0, 3, 8334),
    ("stone", "iron_pickaxe", False, True, 0, 0, 4, 9260),
    ("iron_ore", "stone_pickaxe", False, True, 5, 0, 0, 91),  # md speed 1: Efficiency does nothing
    ("iron_ore", "stone_pickaxe", False, True, 0, 2, 0, 65),
    ("obsidian", "diamond_pickaxe", False, True, 0, 2, 0, 1072),
    ("oak_log", "iron_axe", False, True, 2, 0, 0, 6),
    ("oak_log", "iron_axe", False, True, 0, 0, 2, 112),
]


@pytest.mark.parametrize("block,held,in_water,on_ground,eff,haste,fatigue,expected", MINEFLAYER_DIGTIME)
def test_mineflayer_ticks_match_prismarine_digtime(rules: Rules, block: str, held: str | None, in_water: bool,
                                                   on_ground: bool, eff: int, haste: int, fatigue: int,
                                                   expected: int) -> None:
    t = m.break_ticks_for(rules, block, held, in_water=in_water, on_ground=on_ground, efficiency=eff,
                          haste=haste, mining_fatigue=fatigue)
    assert t.mineflayer_ticks == expected


def test_mineflayer_tool_speed_differs_only_on_md_quirks(rules: Rules) -> None:
    assert m.mineflayer_tool_speed(rules, "stone_pickaxe", "iron_ore") == 1.0
    assert rules.mining_speed("stone_pickaxe", "iron_ore") == 4.0
    assert m.mineflayer_tool_speed(rules, "wooden_shovel", "obsidian") == 2.0
    assert rules.mining_speed("wooden_shovel", "obsidian") == 1.0
    assert m.mineflayer_tool_speed(rules, "iron_hoe", "moss_carpet") == 1.0
    assert m.mineflayer_tool_speed(rules, "iron_pickaxe", "stone") == rules.mining_speed("iron_pickaxe", "stone")
    assert m.mineflayer_tool_speed(rules, None, "iron_ore") == 1.0
    # The exceptions are exactly the pairs the rules build tolerates in its materials check.
    pairs = {(b, k) for b, kinds in m.MINEFLAYER_MISSING_TOOL_KINDS.items() for k in kinds}
    assert pairs == set(B.TOLERATED_TOOL_SPEED)
    assert m.MINEFLAYER_TIER_GATED_TAG in B.TOLERATED_MATERIAL_SUBSTITUTION
    # The design formula with Mineflayer's speed is the primitive behind both counts.
    assert m.break_ticks(3.0, 4.0, True, mineflayer_speed=1.0).mineflayer_ticks == \
        m.design_break_ticks(3.0, 1.0, True) == 91
    assert m.break_ticks(3.0, 4.0, True).mineflayer_ticks == m.break_ticks(3.0, 4.0, True).design_ticks == 23


def test_mineflayer_refused_with_wrong_wooden_tools(rules: Rules) -> None:
    # Iron ore with a wooden axe: Mineflayer sends "finished" after 150 ticks (50% progress); the
    # server needs 70%, (k + 1) / 300 >= 0.7 from k = 209, so it refuses.
    t = m.break_ticks_for(rules, "iron_ore", "wooden_axe")
    assert (t.mineflayer_ticks, t.server_accept_ticks) == (150, 209)
    assert t.mineflayer_refused
    for block, held in (("iron_ore", "stone_pickaxe"), ("iron_ore", "wooden_pickaxe"), ("stone", None),
                        ("coal_ore", "wooden_pickaxe"), ("short_grass", None), ("bedrock", "wooden_axe")):
        assert not m.break_ticks_for(rules, block, held).mineflayer_refused, (block, held)


def _md_cache_ok() -> bool:
    return SOURCES_DIR.is_dir() and not verify_all(SOURCES_JSON, SOURCES_DIR)


@pytest.mark.skipif(not _md_cache_ok(), reason="data/sources/ missing or not matching the pins")
def test_mineflayer_tool_speed_matches_minecraft_data_everywhere(rules: Rules) -> None:
    # prismarine-block's lookups, ported: speed = materials[block.material][itemId] or 1, and
    # canHarvest = no harvestTools, or the held item's id is in them.
    md = SOURCES_DIR / "minecraft-data" / "data" / "pc" / "1.21.1"
    blocks = json.loads((md / "blocks.json").read_text())
    materials = json.loads((md / "materials.json").read_text())
    item_id = {i["name"]: i["id"] for i in json.loads((md / "items.json").read_text())}
    tools = [None] + sorted(n for n, it in rules.items.items() if it.tool is not None)
    assert {int(i) for mat in materials.values() for i in mat} <= {item_id[t] for t in tools[1:]}
    for b in blocks:
        speeds, harvest = materials.get(b["material"], {}), b.get("harvestTools")
        for tool in tools:
            md_speed = float(speeds.get(str(item_id[tool]), 1.0)) if tool else 1.0
            assert m.mineflayer_tool_speed(rules, tool, b["name"]) == md_speed, (b["name"], tool)
            md_harvest = not harvest or (tool is not None and str(item_id[tool]) in harvest)
            assert rules.can_harvest(tool, b["name"]) == md_harvest, (b["name"], tool)


DIG_FIXTURE_SCRIPT = REPO / "bridge" / "node" / "make_dig_fixture.py"


@pytest.fixture(scope="module")
def dig_fixture_script():
    spec = importlib.util.spec_from_file_location("make_dig_fixture", DIG_FIXTURE_SCRIPT)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


def test_bridge_dig_fixture_is_current(rules: Rules, dig_fixture_script) -> None:
    # bridge/node/test/patches.test.js checks patched Mineflayer against this fixture, so it must
    # follow the mechanics; regenerate with .venv/bin/python bridge/node/make_dig_fixture.py
    gen = dig_fixture_script
    fixture = gen.build_fixture(rules)
    assert gen.render(fixture) == gen.FIXTURE_PATH.read_text(encoding="utf-8"), "rerun make_dig_fixture.py"
    held, obsidian = fixture["held"], fixture["blocks"]["obsidian"]
    assert len(fixture["tier_gated"]) == 101 and held[0] is None
    assert obsidian["design"]["ground"][held.index("diamond_pickaxe")] == 188
    assert obsidian["unpatched"]["ground"][held.index("diamond_pickaxe")] == 1500
    assert {(e["block"], e["held"].endswith("_hoe")) for e in fixture["exceptions"]} == {("moss_carpet", True)}


def test_patched_mineflayer_waits_design_ticks_on_every_block(rules: Rules, dig_fixture_script) -> None:
    # DESIGN.md 5: with the bridge's dig-time patch, Mineflayer waits BreakTime.design_ticks for every
    # 1.21.1 block, held tool and condition except moss carpet with a hoe. The fixture covers the v1
    # blocks; this covers all of them (deduplicated by the inputs of the formula). It is why the
    # simulator's count for the bridge is design_ticks, not the unpatched mineflayer_ticks.
    c = m.constants_for(rules)
    tier_gated = set(rules.tag("block", m.MINEFLAYER_TIER_GATED_TAG))
    held_items = [None, *sorted(n for n, it in rules.items.items() if it.tool is not None)]
    conditions = ((False, True), (True, True), (False, False))  # (in_water, on_ground)
    differs: dict[tuple, bool] = {}  # (hardness, game speed, can harvest, patched speed) -> any tick differs
    culprits, other_speed = set(), 0
    for block, b in rules.blocks.items():
        for held in held_items:
            speed = rules.mining_speed(held, block)
            patched = dig_fixture_script.patched_speed(rules, held, block, tier_gated)
            if patched == speed:
                continue  # same inputs, same ticks
            other_speed += 1
            key = (b.hardness, speed, rules.can_harvest(held, block), patched)
            if key not in differs:
                differs[key] = any(
                    m.design_break_ticks(key[0], patched, key[2], in_water=w, on_ground=g, c=c)
                    != m.design_break_ticks(key[0], speed, key[2], in_water=w, on_ground=g, c=c)
                    for w, g in conditions)
            if differs[key]:
                culprits.add((block, rules.item(held).tool.kind if held else None))
    assert culprits == {("moss_carpet", "hoe")}
    # moss carpet and pink petals with each of the 6 hoes (pink petals break instantly anyway)
    assert other_speed == 12 and len(rules.blocks) > 1000 and len(held_items) > 30


def test_server_accept_example(rules: Rules) -> None:
    # 16 * 0.044444446 = 0.7111 >= 0.7, 15 * 0.044444446 = 0.667 < 0.7
    assert m.break_ticks_for(rules, "stone", "wooden_pickaxe").server_accept_ticks == 15


def test_instant_and_unbreakable(rules: Rules) -> None:
    grass = m.break_ticks_for(rules, "short_grass")  # hardness 0
    assert grass.instant and grass.breakable
    assert (grass.ticks, grass.design_ticks, grass.mineflayer_ticks, grass.server_accept_ticks) == (0, 0, 0, 0)
    fast = m.break_ticks(0.5, 8.0, True)  # 8 / 0.5 / 30 = 0.53: two ticks, not instant
    assert fast.ticks == 2 and not fast.instant
    insta = m.break_ticks(0.2, 8.0, True)  # 1.33 >= 1
    assert insta.instant and insta.design_ticks == insta.mineflayer_ticks == 0
    bedrock = m.break_ticks_for(rules, "bedrock", "netherite_pickaxe")
    assert not bedrock.breakable and bedrock.progress_per_tick == 0.0
    assert (bedrock.ticks, bedrock.design_ticks, bedrock.mineflayer_ticks, bedrock.server_accept_ticks) == \
        (None, None, None, None)


def test_known_tool_times(rules: Rules) -> None:
    assert m.break_ticks_for(rules, "stone", "diamond_pickaxe").ticks == 6  # 8 / 1.5 / 30
    assert m.break_ticks_for(rules, "cobweb", "wooden_sword").ticks == 8  # sword rule: 15 / 4 / 30
    assert m.break_ticks_for(rules, "dirt", None).ticks == 15  # 1 / 0.5 / 30
    assert m.break_ticks_for(rules, "iron_ore", "stone_pickaxe").ticks == 23  # 4 / 3 / 30


def test_efficiency_applies_only_above_speed_one(rules: Rules) -> None:
    hand = m.break_ticks_for(rules, "stone", None)
    assert m.break_ticks_for(rules, "stone", None, efficiency=5) == hand
    assert m.destroy_speed(6.0, efficiency=5) == 32.0  # 6 + 5^2 + 1
    assert m.break_ticks_for(rules, "stone", "iron_pickaxe", efficiency=5).ticks == 2  # 32 / 1.5 / 30 = 0.71


def test_haste_water_and_air(rules: Rules) -> None:
    assert m.destroy_speed(2.0, haste=2) == f32(2.0 * f32(1.4))
    assert m.break_ticks_for(rules, "stone", "wooden_pickaxe", haste=2).design_ticks == 17  # 2.8/1.5/30
    water = m.break_ticks_for(rules, "stone", "wooden_pickaxe", in_water=True)
    air = m.break_ticks_for(rules, "stone", "wooden_pickaxe", on_ground=False)
    assert water.design_ticks == air.design_ticks == 113  # 0.4 / 1.5 / 30 -> 112.5
    assert m.break_ticks_for(rules, "stone", "wooden_pickaxe", in_water=True, aqua_affinity=True) == \
        m.break_ticks_for(rules, "stone", "wooden_pickaxe")
    both = m.break_ticks_for(rules, "stone", "wooden_pickaxe", in_water=True, on_ground=False)
    assert both.design_ticks == both.mineflayer_ticks == 563  # 0.08 / 1.5 / 30 -> 562.5


def test_mining_fatigue_table_vs_design_formula() -> None:
    vanilla = [m.mining_fatigue_multiplier(i) for i in range(6)]
    design = [m.mining_fatigue_multiplier(i, design_formula=True) for i in range(6)]
    assert vanilla == [1.0, f32(0.3), f32(0.09), f32(0.0027), f32(0.00081), f32(0.00081)]
    assert design[:3] == pytest.approx(vanilla[:3], rel=1e-6)
    assert design[3] == pytest.approx(10 * vanilla[3], rel=1e-6)  # 0.027 vs 0.0027 [mem]
    assert design[4] == design[5] == pytest.approx(10 * vanilla[4], rel=1e-6)


def test_client_ticks_match_a_plain_float32_loop() -> None:
    for progress in (0.3, 0.1, 1 / 7, 0.0123, 0.003, 1 / 300, 0.000913):
        p = f32(progress)
        assert m.client_break_ticks(p) == python_float32_sum_ticks(p)


def test_tiny_progress_never_reaches_one() -> None:
    assert m.client_break_ticks(f32(2.0**-26)) is None  # below half an ulp of 1 - 2^-24
    slow = m.client_break_ticks(f32(1e-6))  # crosses chunk boundaries
    assert slow is not None and abs(slow - 1_000_000) < 0.1 * 1_000_000


def test_break_delay(c: MechanicsConstants) -> None:
    assert m.break_delay_ticks() == m.break_delay_ticks(c) == 6


# =============================================================================
# Hunger [14]
# =============================================================================


def test_spawn_state_and_sprinting() -> None:
    h = HungerState()
    assert (h.food, h.saturation, h.exhaustion, h.timer) == (20, 5.0, 0.0, 0)
    assert h.can_sprint() and not h.needs_food()
    assert HungerState(food=7).can_sprint()
    assert not HungerState(food=6).can_sprint()


def test_exhaustion_cap_and_costs() -> None:
    h = HungerState()
    h.add_exhaustion(100.0)
    assert h.exhaustion == 40.0
    assert m.exhaustion_cost("sprint", 10.0) == f32(f32(0.1) * 10.0)
    assert m.exhaustion_cost("break_block") == f32(0.005)
    assert m.exhaustion_cost("jump") == f32(0.05) and m.exhaustion_cost("sprint_jump") == f32(0.2)
    with pytest.raises(ValueError):
        m.exhaustion_cost("dance")


def test_exhaustion_threshold_is_strict() -> None:
    h = HungerState(food=20, saturation=3.0, exhaustion=4.0)
    h.tick(20.0)
    assert (h.exhaustion, h.saturation) == (4.0, 3.0)  # exactly 4 does nothing [mem]
    h.add_exhaustion(0.125)
    h.tick(20.0)
    assert (h.exhaustion, h.saturation, h.food) == (0.125, 2.0, 20)
    h = HungerState(food=9, saturation=0.0, exhaustion=4.5)
    h.tick(20.0)
    assert (h.exhaustion, h.food) == (0.5, 8)


def starve(difficulty: Difficulty, rate: float = 0.125, limit: int = 5000) -> tuple[int, list[int], float]:
    """Full health, spawn hunger, a fixed exhaustion per tick: (tick food hits 0, damage ticks, health)."""
    h, health, food_zero, damage = HungerState(), 20.0, -1, []
    for t in range(1, limit + 1):
        h.add_exhaustion(rate)
        delta = h.tick(health, difficulty=difficulty)
        if h.food == 0 and food_zero < 0:
            food_zero = t
        if delta:
            health += delta
            damage.append(t)
        if health <= 0:
            break
    return food_zero, damage, health


@pytest.mark.parametrize("difficulty,hits,final", [
    (Difficulty.EASY, 10, 10.0), (Difficulty.NORMAL, 19, 1.0), (Difficulty.HARD, 20, 0.0)])
def test_starvation_sequence(difficulty: Difficulty, hits: int, final: float) -> None:
    # 0.125 per tick is exact in float32: one unit of saturation or food per 32 ticks once past 4
    # (strictly). The 5 saturation and 20 food go at ticks 33, 65, ..., 33 + 32 * 24 = 801.
    # Starvation then hurts every 80 ticks: 880, 960, ...; it stops at 10 HP on Easy, 1 HP on
    # Normal, and kills on Hard.
    food_zero, damage, health = starve(difficulty)
    assert food_zero == 801
    assert damage == [880 + 80 * i for i in range(hits)]
    assert health == final


def test_regeneration_sequence() -> None:
    # Health 10, spawn hunger (20 / 5.0), no other exhaustion. Saturated regeneration heals
    # min(sat, 6) / 6 every 10 ticks and costs min(sat, 6) exhaustion; each time exhaustion
    # exceeds 4, one saturation goes. At tick 30 the exhaustion is exactly 4.0, so it is not spent
    # until it reaches 7.0 at tick 40. Saturation runs out at tick 91; then food >= 18 heals 1 HP
    # per 80 ticks (ticks 170 and 250) for 6 exhaustion each, until food falls to 17.
    h, health, events = HungerState(), 10.0, []
    for t in range(1, 400):
        delta = h.tick(health)
        if delta:
            health = f32(health + delta)
            events.append((t, delta))
    assert [t for t, _ in events] == [10, 20, 30, 40, 50, 60, 70, 80, 90, 170, 250]
    heals = [f32(5 / 6), f32(4 / 6), 0.5, 0.5, f32(2 / 6)] + [f32(1 / 6)] * 4 + [1.0, 1.0]
    assert [d for _, d in events] == pytest.approx(heals, abs=1e-6)
    assert health == pytest.approx(10 + sum(heals), abs=1e-5)
    assert (h.food, h.saturation, h.exhaustion) == (17, 0.0, 1.0)


def test_no_regeneration_at_full_health_or_when_disabled() -> None:
    h = HungerState(timer=5)
    for _ in range(200):
        assert h.tick(20.0) == 0.0
    assert (h.food, h.saturation, h.timer) == (20, 5.0, 0)
    h = HungerState()
    assert all(h.tick(10.0, natural_regeneration=False) == 0.0 for _ in range(200))
    assert h.tick(19.5) == 0.0 and h.timer == 1


def test_heal_is_capped_at_max_health() -> None:
    h = HungerState(timer=9)
    assert h.tick(19.5) == 0.5  # 5/6 would overshoot
    assert h.exhaustion == 5.0  # but the cost is the full min(sat, 6)


def test_peaceful_never_loses_food() -> None:
    h = HungerState(food=10, saturation=0.0)
    for _ in range(100):
        h.add_exhaustion(1.0)
        h.tick(20.0, difficulty=Difficulty.PEACEFUL)
    assert h.food == 10


def test_eating(rules: Rules) -> None:
    h = HungerState(food=15, saturation=0.0)
    h.eat_item(rules, "bread")  # nutrition 5, saturation 6.0 (absolute, 1.20.5+)
    assert (h.food, h.saturation) == (20, 6.0)
    h = HungerState(food=18, saturation=2.0)
    h.eat_item(rules, "cooked_beef")  # 8 and 12.8: food caps at 20
    assert (h.food, h.saturation) == (20, f32(14.8))
    h = HungerState(food=3, saturation=0.0)
    h.eat_item(rules, "golden_apple")  # 4 and 9.6: saturation caps at the new food level
    assert (h.food, h.saturation) == (7, 7.0)
    with pytest.raises(ValueError):
        h.eat_item(rules, "stone")


def test_eat_ticks(rules: Rules) -> None:
    assert m.eat_ticks() == 32
    assert m.eat_ticks_for(rules, "bread") == 32  # eat_seconds 1.6
    assert m.eat_ticks_for(rules, "dried_kelp") == 16  # eat_seconds 0.8


# =============================================================================
# Clock [15]
# =============================================================================


def test_time_of_day_and_night() -> None:
    assert m.time_of_day(24000 * 3 + 13000) == 13000
    assert not m.is_night(12999) and m.is_night(13000) and m.is_night(22999) and not m.is_night(23000)
    assert m.is_night(24000 + 18000)
    assert m.ticks_until(1000, 13000) == 12000
    assert m.ticks_until(14000, 13000) == 23000
    assert m.ticks_until(13000, 13000) == 0


def test_clock_windows_edges() -> None:
    assert not m.surface_hostile_spawn_time(13187)
    assert m.surface_hostile_spawn_time(13188) and m.surface_hostile_spawn_time(22812)
    assert not m.surface_hostile_spawn_time(22813)
    assert not m.can_sleep_time(12541) and m.can_sleep_time(12542)
    assert m.can_sleep_time(23459) and not m.can_sleep_time(23460)


def test_clock_windows_match_the_sky_darken_formula() -> None:
    # Two independent sources agree: the wiki's tick windows [15] and the game's sky formula [mem].
    for t in range(24000):
        darken = m.sky_darken(t)
        assert m.surface_hostile_spawn_time(t) == (darken >= 8) == m.sky_allows_surface_spawns(t), t
        assert m.can_sleep_time(t) == (darken >= 4) == m.sky_is_dark(t), t
    assert m.surface_sky_light(6000) == 15 and m.surface_sky_light(18000) == 4
    assert m.sky_darken(18000, rain=1.0) == 11 and m.sky_darken(6000, rain=1.0) == 3
    assert not m.sky_is_dark(6000, rain=1.0) and m.sky_is_dark(6000, rain=1.0, thunder=1.0)  # 5 in a storm


# =============================================================================
# Combat [18]
# =============================================================================


def test_attack_stats(rules: Rules) -> None:
    fist = m.attack_stats_for(rules, None)
    assert (fist.damage, fist.attack_speed, fist.cooldown) == (1.0, 4.0, 5.0)
    assert m.attack_stats_for(rules, "stick") == fist  # no attribute modifiers
    for sword, dmg in (("wooden_sword", 4.0), ("stone_sword", 5.0), ("iron_sword", 6.0), ("diamond_sword", 7.0)):
        s = m.attack_stats_for(rules, sword)
        assert s.damage == dmg
        assert s.attack_speed == pytest.approx(1.6, abs=1e-6)
        assert s.cooldown == pytest.approx(12.5, abs=1e-5)  # 12.500001 in float32
    assert m.attack_stats_for(rules, "wooden_axe").cooldown == pytest.approx(25.0, abs=1e-5)


def test_damage_multiplier_endpoints() -> None:
    assert m.attack_damage_multiplier(0.0) == f32(0.2)
    assert m.attack_damage_multiplier(1.0) == 1.0
    assert m.attack_damage_multiplier(f32(0.9)) == pytest.approx(0.848, abs=1e-6)  # the wiki's "84.8%"
    assert m.attack_charge(0, 5.0) == f32(0.1)
    assert m.attack_charge(100, 5.0) == 1.0  # clamped
    assert m.melee_damage(1.0, 0, 5.0) == pytest.approx(0.208, abs=1e-6)  # fist right after a swing
    assert m.melee_damage(1.0, 4, 5.0) == pytest.approx(0.848, abs=1e-6)
    assert m.melee_damage(1.0, 5, 5.0) == 1.0


def test_sword_full_charge_needs_13_ticks(rules: Rules) -> None:
    # T = (float)(1 / 1.5999999 * 20) = 12.500001, so (12 + 0.5) / T = 0.99999994 < 1:
    # a float32 effect. The hit after 12 ticks deals 6.999999, after 13 ticks the full 7.
    assert m.melee_damage_for(rules, "diamond_sword", 12) == pytest.approx(7.0, abs=2e-6)
    assert m.melee_damage_for(rules, "diamond_sword", 12) < 7.0
    assert m.melee_damage_for(rules, "diamond_sword", 13) == 7.0
    assert m.melee_damage_for(rules, None, 5) == 1.0


def test_critical_hits(rules: Rules) -> None:
    assert not m.is_critical(f32(0.9), True)  # needs charge > 0.9
    assert m.is_critical(0.92, True) and not m.is_critical(1.0, False)
    assert m.melee_damage_for(rules, "diamond_sword", 13, falling=True) == 10.5
    s = m.attack_stats_for(rules, "diamond_sword")
    # (10 + 0.5) / 12.5 = 0.84: no critical; (11 + 0.5) / 12.5 = 0.92: critical
    assert m.melee_damage(s.damage, 10, s.cooldown, falling=True) == m.melee_damage(s.damage, 10, s.cooldown)
    assert m.melee_damage(s.damage, 11, s.cooldown, falling=True) == \
        f32(m.melee_damage(s.damage, 11, s.cooldown) * 1.5)


def test_difficulty_scaling_of_mob_damage() -> None:
    assert [m.scale_damage_for_difficulty(3.0, d) for d in Difficulty] == [0.0, 2.5, 3.0, 4.5]
    assert m.scale_damage_for_difficulty(1.0, Difficulty.EASY) == 1.0  # min(1.5, 1)


def test_fall_damage() -> None:
    assert [m.fall_damage(x) for x in (0.0, 2.0, 3.0, 3.5, 4.0, 10.0, 23.0)] == [0, 0, 0, 1, 1, 7, 20]


def test_hurt_cooldown() -> None:
    hc = m.HurtCooldown()
    assert hc.hurt(5.0) == 5.0
    for _ in range(9):
        hc.tick()
    assert hc.hurt(3.0) == 0.0  # still invulnerable, weaker hit
    assert hc.hurt(7.0) == 2.0  # stronger hit: only the difference
    hc.tick()  # 10 ticks after the first hit
    assert hc.hurt(1.0) == 1.0 and hc.invulnerable_time == 20


# =============================================================================
# Armor [17]
# =============================================================================

IRON = ["iron_helmet", "iron_chestplate", "iron_leggings", "iron_boots"]
DIAMOND = ["diamond_helmet", "diamond_chestplate", "diamond_leggings", "diamond_boots"]


def test_armor_examples(rules: Rules) -> None:
    assert m.armor_totals(rules, IRON) == (15, 0.0)
    assert m.armor_totals(rules, DIAMOND) == (20, 8.0)
    # Iron: A = 15, 15 - 4 * 10 / 8 = 10 > 15 / 5, so 10 * (1 - 10 / 25) = 6.
    assert m.armor_damage_for(rules, 10.0, IRON) == 6.0
    # Diamond: A = 20, T = 8: 20 - 40 / 16 = 17.5, so 10 * (1 - 0.7) = 3.
    assert m.armor_damage_for(rules, 10.0, DIAMOND) == 3.0
    assert m.armor_damage(10.0, 0, 0.0) == 10.0
    assert m.armor_damage(100.0, 15, 0.0) == 88.0  # the A/5 floor: 100 * (1 - 3/25)
    assert m.armor_damage(1.0, 30, 0.0) == pytest.approx(0.2, abs=1e-7)  # the cap of 20 points
    assert m.armor_damage(7.0, 2, 0.0) == pytest.approx(6.888, abs=1e-6)  # diamond sword vs zombie
    with pytest.raises(ValueError):
        m.armor_totals(rules, ["diamond_sword"])


def test_armor_matches_the_design_formula() -> None:
    for dmg in (0.5, 1.0, 3.0, 7.0, 12.5, 40.0):
        for armor in range(0, 31, 3):
            for tough in (0.0, 2.0, 8.0, 12.0):
                expected = dmg * (1 - min(20, max(armor / 5, armor - 4 * dmg / (tough + 8))) / 25)
                assert m.armor_damage(dmg, armor, tough) == pytest.approx(expected, rel=1e-6, abs=1e-6)


# =============================================================================
# Smelting and fuel [20]
# =============================================================================


def test_smelting_and_fuel(rules: Rules) -> None:
    assert m.smelt_ticks() == 200
    assert m.fuel_ticks(rules, "coal") == 1600 and m.items_per_fuel(rules, "coal") == 8.0
    assert m.fuel_ticks(rules, "oak_log") == 300 and m.items_per_fuel(rules, "oak_log") == 1.5
    assert m.fuel_ticks(rules, "oak_planks") == 300
    assert m.fuel_ticks(rules, "stick") == 100 and m.items_per_fuel(rules, "stick") == 0.5
    assert m.fuel_ticks(rules, "lava_bucket") == 20000 and m.fuel_ticks(rules, "coal_block") == 16000
    assert m.fuel_ticks(rules, "stone") is None and m.items_per_fuel(rules, "stone") == 0.0


def test_f32_rounds_like_java() -> None:
    assert f32(0.1) == float(np.float32(0.1)) != 0.1
    assert f32(1.0) == 1.0 and f32(2.0**-149) == 2.0**-149  # smallest float32 subnormal
    assert math.isinf(m.progress_per_tick(0.0, 1.0, True))  # Java: x / 0.0F = +inf

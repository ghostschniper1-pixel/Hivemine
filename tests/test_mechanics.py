"""Tests for hivemind.sim.ref.mechanics against the check values of DESIGN.md 4.3."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from hivemind.rules import Rules, load_rules
from hivemind.sim.ref import mechanics as m
from hivemind.sim.ref.mechanics import Difficulty, HungerState, MechanicsConstants, f32


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
    assert c.exhaustion_cap == 40.0  # [mem] default: not in the rules file yet


def test_from_rules_requires_rules_constants_and_prefers_them(rules: Rules) -> None:
    consts = dict(rules.constants)
    consts["exhaustion_cap"] = SimpleNamespace(value=12.5, source="[est]")
    consts["break_mining_fatigue_table"] = SimpleNamespace(value=[0.5, 0.25], source="[est]")
    stub = SimpleNamespace(constants=consts, constant=lambda name: consts[name].value)
    c2 = MechanicsConstants.from_rules(stub)
    assert c2.exhaustion_cap == 12.5 and c2.break_mining_fatigue_table == (0.5, 0.25)
    del consts["smelt_ticks"]
    with pytest.raises(KeyError):
        MechanicsConstants.from_rules(stub)


# =============================================================================
# Breaking [13]
# =============================================================================

# block, held item, DESIGN.md 4.3 check value, vanilla client (float32 accumulation)
BREAK_CHECKS = [
    ("stone", "wooden_pickaxe", 23, 23),
    # 1/dmg is exactly 150, but 150 float32 additions of 0.0066666668 sum to 0.99999923 < 1, so
    # the vanilla client breaks on tick 151. The check value 150 is ceil(1/dmg) in exact
    # arithmetic, which is also what Mineflayer waits (and what the server's 70% check accepts).
    ("stone", None, 150, 151),
    # Same effect: 60 additions of 0.016666668 sum to 0.9999997 in float32.
    ("oak_log", None, 60, 61),
    # Here the float32 sum overshoots: 30 additions of 0.033333335 give 1.0000004.
    ("oak_log", "wooden_axe", 30, 30),
    ("obsidian", "diamond_pickaxe", 188, 188),
]


@pytest.mark.parametrize("block,held,design,client", BREAK_CHECKS)
def test_break_check_values(rules: Rules, block: str, held: str | None, design: int, client: int) -> None:
    t = m.break_ticks_for(rules, block, held)
    assert t.mineflayer_ticks == design
    assert t.ticks == client
    assert t.ticks == python_float32_sum_ticks(t.progress_per_tick)
    # The server accepts "finished" at progress * (k + 1) >= 0.7 (float32), and not one tick earlier.
    k = t.server_accept_ticks
    assert f32(t.progress_per_tick * (k + 1)) >= f32(0.7) > f32(t.progress_per_tick * k)
    assert k < t.ticks


def test_mineflayer_double_rounding(rules: Rules) -> None:
    # 2 / 3 / 30 is 1/45 exactly, but in doubles 1 / dmg is just above 45, so Mineflayer waits
    # 46 ticks while the vanilla client's float32 sum reaches 1.0 after 45.
    t = m.break_ticks_for(rules, "coal_ore", "wooden_pickaxe")
    assert (t.ticks, t.mineflayer_ticks) == (45, 46)
    assert 1.0 / (2.0 / 3.0 / 30.0) > 45.0


def test_stone_by_hand_drops_nothing(rules: Rules) -> None:
    assert not m.can_harvest(rules, "stone", None)
    assert m.can_harvest(rules, "stone", "wooden_pickaxe")
    assert m.can_harvest(rules, "oak_log", None)
    assert not m.can_harvest(rules, "obsidian", "iron_pickaxe")
    # Wrong tier still mines at the tool's speed, but at the /100 divisor: 6 / 50 / 100.
    t = m.break_ticks_for(rules, "obsidian", "iron_pickaxe")
    assert t.progress_per_tick == f32(f32(6.0 / 50.0) / 100.0)
    assert t.mineflayer_ticks == 834


def test_server_accept_example(rules: Rules) -> None:
    # 16 * 0.044444446 = 0.7111 >= 0.7, 15 * 0.044444446 = 0.667 < 0.7
    assert m.break_ticks_for(rules, "stone", "wooden_pickaxe").server_accept_ticks == 15


def test_instant_and_unbreakable(rules: Rules) -> None:
    grass = m.break_ticks_for(rules, "short_grass")  # hardness 0
    assert grass.instant and grass.breakable
    assert (grass.ticks, grass.mineflayer_ticks, grass.server_accept_ticks) == (0, 0, 0)
    fast = m.break_ticks(0.5, 8.0, True)  # 8 / 0.5 / 30 = 0.53: two ticks, not instant
    assert fast.ticks == 2 and not fast.instant
    insta = m.break_ticks(0.2, 8.0, True)  # 1.33 >= 1
    assert insta.instant and insta.mineflayer_ticks == 0
    bedrock = m.break_ticks_for(rules, "bedrock", "netherite_pickaxe")
    assert not bedrock.breakable and bedrock.progress_per_tick == 0.0
    assert (bedrock.ticks, bedrock.mineflayer_ticks, bedrock.server_accept_ticks) == (None, None, None)


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
    assert m.break_ticks_for(rules, "stone", "wooden_pickaxe", haste=2).mineflayer_ticks == 17  # 2.8/1.5/30
    water = m.break_ticks_for(rules, "stone", "wooden_pickaxe", in_water=True)
    air = m.break_ticks_for(rules, "stone", "wooden_pickaxe", on_ground=False)
    assert water.mineflayer_ticks == air.mineflayer_ticks == 113  # 0.4 / 1.5 / 30 -> 112.5
    assert m.break_ticks_for(rules, "stone", "wooden_pickaxe", in_water=True, aqua_affinity=True) == \
        m.break_ticks_for(rules, "stone", "wooden_pickaxe")
    both = m.break_ticks_for(rules, "stone", "wooden_pickaxe", in_water=True, on_ground=False)
    assert both.mineflayer_ticks == 563  # 0.08 / 1.5 / 30 -> 562.5


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

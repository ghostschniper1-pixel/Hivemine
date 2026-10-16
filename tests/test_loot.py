"""Tests for hivemind.sim.ref.loot: block and mob drops, vanilla loot semantics, determinism."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from hivemind.rules import load_concepts, load_rules
from hivemind.sim.ref.loot import (
    ENCHANTMENT_TAGS,
    ENTITY_TYPE_TAGS,
    EXTRA_LOOT_TABLES,
    KillContext,
    LootSampler,
    UnsupportedLoot,
    block_drops,
    entity_drops,
    validate_table,
)
from hivemind.sim.ref.rng import CounterRNG, Purpose

C = load_concepts()
R = load_rules()
SAMPLER = LootSampler()
RNG = CounterRNG(20261016)
MCMETA = Path(__file__).resolve().parents[1] / "data" / "sources" / "mcmeta-1.21.1-data" / "data" / "minecraft"


def iid(name: str) -> int:
    return C.item_class(name).id


def stream(tick: int, entity: int = 0, episode: int = 0):
    return RNG.stream(episode, tick, Purpose.LOOT, entity)


def names(d: dict[int, int]) -> dict[str, int]:
    return {C.item_class(k).name: v for k, v in d.items()}


def sample_many(fn, n: int, episode: int = 1) -> list[dict[str, int]]:
    """``n`` samples, each with its own stream (tick i)."""
    return [names(fn(stream(i, episode=episode))) for i in range(n)]


def assert_rate(hits: int, n: int, p: float, sigmas: float = 5.0) -> None:
    """|hits/n - p| within ``sigmas`` binomial standard deviations (deterministic seeds: no flakes)."""
    sd = math.sqrt(p * (1 - p) / n)
    assert abs(hits / n - p) <= sigmas * sd, (hits / n, p, sd)


# -- blocks ---------------------------------------------------------------------------------


def test_stone_drops_cobblestone_only_when_harvested():
    assert names(SAMPLER.block_drops("stone", "wooden_pickaxe", stream(0))) == {"cobblestone": 1}
    assert SAMPLER.block_drops("stone", None, stream(1)) == {}  # by hand: nothing
    assert SAMPLER.block_drops("stone", "wooden_axe", stream(2)) == {}
    assert names(SAMPLER.block_drops("stone", None, stream(3), harvested=True)) == {"cobblestone": 1}
    assert names(block_drops("stone", "diamond_pickaxe", stream(4))) == {"cobblestone": 1}


def test_ores_need_their_tier():
    assert SAMPLER.block_drops("iron_ore", "wooden_pickaxe", stream(0)) == {}
    assert names(SAMPLER.block_drops("iron_ore", "stone_pickaxe", stream(1))) == {"raw_iron": 1}
    assert names(SAMPLER.block_drops("coal_ore", "wooden_pickaxe", stream(2))) == {"coal": 1}
    assert SAMPLER.block_drops("diamond_ore", "stone_pickaxe", stream(3)) == {}
    assert names(SAMPLER.block_drops("diamond_ore", "iron_pickaxe", stream(4))) == {"diamond": 1}
    assert names(SAMPLER.block_drops("log", None, stream(5))) == {"log": 1}
    assert SAMPLER.block_drops("glass", None, stream(6)) == {}  # no drops without Silk Touch


def test_gravel_flint_rate_is_ten_percent():
    n = 20000
    draws = sample_many(lambda s: SAMPLER.block_drops("gravel", None, s), n)
    flint = sum(1 for d in draws if d == {"flint": 1})
    gravel = sum(1 for d in draws if d == {"gravel": 1})
    assert flint + gravel == n  # exactly one item each time
    assert_rate(flint, n, 0.1)  # vanilla 10%, not minecraft-data's stale 1/3


@pytest.mark.parametrize("block, apple", [("oak_leaves", 0.005), ("leaves", 0.0)])
def test_leaves_sapling_stick_and_apple_rates(block: str, apple: float):
    n = 20000
    draws = sample_many(lambda s: SAMPLER.block_drops(block, None, s), n, episode=2)
    assert_rate(sum(1 for d in draws if "sapling" in d), n, 0.05)  # table_bonus at Fortune 0
    sticks = [d.get("stick", 0) for d in draws]
    assert_rate(sum(1 for k in sticks if k), n, 0.02)
    assert set(sticks) == {0, 1, 2}
    assert_rate(sum(sticks), n, 0.02 * 1.5, sigmas=6.0)  # 1-2 sticks when the 2% roll hits
    apples = sum(1 for d in draws if "apple" in d)
    if apple:
        assert_rate(apples, n, apple)
    else:
        assert apples == 0  # only oak and dark oak leaves drop apples
    assert all(set(d) <= {"sapling", "stick", "apple"} for d in draws)


def test_leaves_with_shears_drop_the_leaves_block():
    for i in range(200):
        assert names(SAMPLER.block_drops("oak_leaves", "shears", stream(i))) == {"other_item_64": 1}


def test_grass_seeds_and_cobweb():
    n = 8000
    draws = sample_many(lambda s: SAMPLER.block_drops("short_grass", None, s), n, episode=3)
    assert_rate(sum(1 for d in draws if d == {"wheat_seeds": 1}), n, 0.125)
    assert names(SAMPLER.block_drops("cobweb", "iron_sword", stream(0))) == {"string": 1}
    assert names(SAMPLER.block_drops("cobweb", "shears", stream(1))) == {"other_item_64": 1}
    assert SAMPLER.block_drops("cobweb", None, stream(2)) == {}


def test_wheat_seeds_draw_three_trials_even_without_fortune():
    # Mature wheat: 1 wheat, and a seed entry of count 1 plus Binomial(0 + 3, 0.5714286): 1-4 seeds.
    n = 6000
    p = 0.5714286
    draws = sample_many(lambda s: SAMPLER.block_drops("wheat", None, s), n, episode=4)
    assert all(d["wheat"] == 1 for d in draws)
    seeds = Counter(d["wheat_seeds"] for d in draws)
    assert set(seeds) == {1, 2, 3, 4}
    for k in range(4):
        assert_rate(seeds[1 + k], n, math.comb(3, k) * p**k * (1 - p) ** (3 - k))


def test_uniform_counts_and_limit_count():
    redstone = Counter(SAMPLER.block_drops("redstone_ore", "iron_pickaxe", stream(i))[iid("redstone")]
                       for i in range(2000))
    assert set(redstone) == {4, 5}
    lapis = Counter(SAMPLER.block_drops("lapis_ore", "stone_pickaxe", stream(i))[iid("lapis_lazuli")]
                    for i in range(3000))
    assert set(lapis) == set(range(4, 10))
    melon = Counter(SAMPLER.block_drops("melon", None, stream(i))[iid("melon_slice")] for i in range(3000))
    assert set(melon) == set(range(3, 8))  # 3-7, limit 9 does not bite at Fortune 0


def test_every_block_class_drop_table_is_supported():
    tools = [None] + [c.name for c in C.item_classes if c.v1 and (c.tool is not None or c.name == "shears")]
    n_tables = 0
    for b in C.block_classes:
        if b.drops is None:
            continue
        validate_table(b.drops["table"], b.name)
        n_tables += 1
        for k, tool in enumerate(tools):
            for d in range(3):
                drops = SAMPLER.block_drops(b.id, tool, stream(1000 * k + d, entity=b.id))
                assert all(n > 0 for n in drops.values())
                assert set(drops) <= {iid(x) for x in b.drops["items"]}
    assert n_tables > 150


# -- mobs ----------------------------------------------------------------------------------------


def v1_mobs() -> list[str]:
    return [m for e in C.entity_classes if e.kind in ("hostile", "passive") and not e.fallback for m in e.members]


def contexts() -> list[KillContext]:
    known = {"slime_size": 1, "is_captain": False}
    return [KillContext(**known), KillContext.by_player(**known), KillContext.by_player(on_fire=True, **known),
            KillContext(attacker="skeleton", direct_attacker="arrow", **known),
            KillContext(attacker="frog", slime_size=1, is_captain=True),
            KillContext(damage_tags=frozenset({"is_lightning"}), **known),
            KillContext.by_player(sheared=True, colour="black", **known)]


def test_every_v1_mob_loot_table_is_supported():
    mobs = v1_mobs()
    assert len(mobs) == 59 and "zombie" in mobs and "guardian" in mobs
    for k, mob in enumerate(mobs):
        validate_table(SAMPLER._table(SAMPLER.entity_table(mob)), mob)
        for j, ctx in enumerate(contexts()):
            for d in range(15):
                drops = SAMPLER.entity_drops(mob, stream(100 * j + d, entity=k), ctx)
                assert all(n > 0 for n in drops.values())


def test_zombie_rotten_flesh_zero_to_two():
    n = 6000
    draws = sample_many(lambda s: SAMPLER.entity_drops("zombie", s, killed_by_player=True), n, episode=5)
    flesh = Counter(d.get("rotten_flesh", 0) for d in draws)
    assert set(flesh) == {0, 1, 2}
    for k in range(3):
        assert_rate(flesh[k], n, 1 / 3)
    rare = sum(1 for d in draws if set(d) & {"iron_ingot", "carrot", "potato"})
    assert_rate(rare, n, 0.025)  # killed by a player: 2.5% for one rare drop
    not_player = sample_many(lambda s: entity_drops("zombie", s), 3000, episode=6)
    assert all(set(d) <= {"rotten_flesh"} for d in not_player)


def test_cooked_meat_when_on_fire():
    for i in range(300):
        raw = names(SAMPLER.entity_drops("cow", stream(i), KillContext.by_player()))
        cooked = names(SAMPLER.entity_drops("cow", stream(i), KillContext.by_player(on_fire=True)))
        assert 1 <= raw["beef"] <= 3 and "cooked_beef" not in raw
        assert cooked.get("cooked_beef") == raw["beef"] and "beef" not in cooked  # same draws, smelted
        assert cooked.get("leather") == raw.get("leather")
        chicken = names(SAMPLER.entity_drops("chicken", stream(i), on_fire=True))
        assert chicken.get("cooked_chicken") == 1 and "chicken" not in chicken
    potato = [names(SAMPLER.entity_drops("zombie", stream(i, episode=7), KillContext.by_player(on_fire=True)))
              for i in range(4000)]
    assert any("baked_potato" in d for d in potato) and not any("potato" in d for d in potato)


def test_sheep_drop_wool_unless_sheared():
    for i in range(200):
        d = names(SAMPLER.entity_drops("sheep", stream(i), KillContext.by_player(colour="red")))
        assert d["wool"] == 1 and 1 <= d["mutton"] <= 2
        sheared = names(SAMPLER.entity_drops("sheep", stream(i), KillContext.by_player(sheared=True)))
        assert "wool" not in sheared and 1 <= sheared["mutton"] <= 2
    assert SAMPLER.entity_table("sheep") == "entities/sheep/white"


def test_conditions_on_the_killer():
    disc = iid("other_item_1")  # music discs are not v1 items
    by_skeleton = KillContext(attacker="skeleton", direct_attacker="arrow")
    assert all(SAMPLER.entity_drops("creeper", stream(i), by_skeleton).get(disc) == 1 for i in range(50))
    assert all(disc not in SAMPLER.entity_drops("creeper", stream(i), killed_by_player=True) for i in range(50))
    bowl = [names(SAMPLER.entity_drops("turtle", stream(i), KillContext(damage_tags=frozenset({"is_lightning"}))))
            for i in range(20)]
    assert all(d.get("bowl") == 1 for d in bowl)
    assert all("bowl" not in names(SAMPLER.entity_drops("turtle", stream(i), killed_by_player=True))
               for i in range(20))


def test_slime_size_and_frog_kills():
    small = [names(SAMPLER.entity_drops("slime", stream(i), KillContext.by_player(slime_size=1))) for i in range(300)]
    assert {d.get("slime_ball", 0) for d in small} == {0, 1, 2}
    big = [SAMPLER.entity_drops("slime", stream(i), KillContext.by_player(slime_size=2)) for i in range(50)]
    assert all(d == {} for d in big)
    eaten = [names(SAMPLER.entity_drops("slime", stream(i), KillContext(attacker="frog", slime_size=1)))
             for i in range(50)]
    assert all(d == {"slime_ball": 1} for d in eaten)


def test_missing_context_raises():
    with pytest.raises(UnsupportedLoot, match="size"):
        SAMPLER.entity_drops("slime", stream(0), KillContext.by_player())
    with pytest.raises(UnsupportedLoot, match="captain"):
        SAMPLER.entity_drops("pillager", stream(0), KillContext.by_player())
    with pytest.raises(UnsupportedLoot):
        SAMPLER.entity_drops("monster", stream(0))  # a fallback class has many members
    by_id = [names(SAMPLER.entity_drops(C.entity_class("skeleton").id, stream(i), KillContext.by_player()))
             for i in range(50)]
    assert all(set(d) <= {"arrow", "bone"} for d in by_id) and any(by_id)


def test_guardian_fish_come_from_the_nested_fishing_table():
    # Killed by a player: 2.5% for one roll of gameplay/fishing/fish (salmon weight 25 of 100).
    n = 12000
    draws = [names(SAMPLER.entity_drops("guardian", stream(i, episode=8), KillContext.by_player())) for i in range(n)]
    assert_rate(sum(1 for d in draws if "salmon" in d), n, 0.025 * 0.25)
    assert_rate(sum(1 for d in draws if "pufferfish" in d), n, 0.025 * 0.13)
    not_player = [SAMPLER.entity_drops("guardian", stream(i, episode=8)) for i in range(2000)]
    assert not any(iid("salmon") in d for d in not_player)


# -- semantics, randomness and validation ----------------------------------------------------------


def test_determinism_and_purpose():
    a = [SAMPLER.entity_drops("witch", stream(i, entity=9), killed_by_player=True) for i in range(200)]
    b = [entity_drops("witch", stream(i, entity=9), killed_by_player=True) for i in range(200)]
    assert a == b
    c = [SAMPLER.entity_drops("witch", stream(i, entity=10), killed_by_player=True) for i in range(200)]
    assert a != c
    with pytest.raises(ValueError):
        SAMPLER.block_drops("gravel", None, RNG.stream(0, 0, Purpose.SPAWN))


def test_draws_are_taken_only_for_random_outcomes():
    s = stream(5)
    SAMPLER.block_drops("stone", "stone_pickaxe", s)
    assert s.draw == 0  # one entry, constant count: no draw
    SAMPLER.block_drops("gravel", None, s)
    assert s.draw == 1  # one chance test
    SAMPLER.block_drops("wheat", None, s)
    assert s.draw == 4  # three binomial trials


def item_table(*functions: dict, conditions: list | None = None, count_entries: int = 1) -> dict:
    entry: dict = {"type": "minecraft:item", "name": "minecraft:stick"}
    if functions:
        entry["functions"] = list(functions)
    pool: dict = {"rolls": 1.0, "bonus_rolls": 0.0, "entries": [entry] * count_entries}
    if conditions:
        pool["conditions"] = conditions
    return {"type": "minecraft:generic", "pools": [pool]}


def test_explosions_decay_counts_and_destroy_blocks():
    t = item_table({"function": "minecraft:set_count", "count": 8.0}, {"function": "minecraft:explosion_decay"})
    assert SAMPLER.sample_table(t, stream(0)) == {iid("stick"): 8}  # no explosion: identity
    total = sum(SAMPLER.sample_table(t, stream(i), explosion_radius=4.0).get(iid("stick"), 0) for i in range(2000))
    assert_rate(total, 16000, 0.25)
    t = item_table(conditions=[{"condition": "minecraft:survives_explosion"}])
    hits = sum(1 for i in range(4000) if SAMPLER.sample_table(t, stream(i), explosion_radius=5.0))
    assert_rate(hits, 4000, 0.2)


def test_fortune_formulas_are_identities_at_level_zero():
    for formula, params in (("minecraft:ore_drops", None), ("minecraft:uniform_bonus_count", {"bonusMultiplier": 1})):
        fn = {"function": "minecraft:apply_bonus", "enchantment": "minecraft:fortune", "formula": formula}
        if params:
            fn["parameters"] = params
        s = stream(1)
        assert SAMPLER.sample_table(item_table({"function": "minecraft:set_count", "count": 2.0}, fn), s) == \
            {iid("stick"): 2}
        assert s.draw == 0


def test_alternatives_group_sequence_and_weights():
    stick = {"type": "minecraft:item", "name": "minecraft:stick"}
    coal = {"type": "minecraft:item", "name": "minecraft:coal"}
    never = {"type": "minecraft:item", "name": "minecraft:diamond",
             "conditions": [{"condition": "minecraft:random_chance", "chance": 0.0}]}

    def one(entry: dict) -> dict:
        return {"type": "minecraft:generic", "pools": [{"rolls": 1.0, "entries": [entry]}]}

    s = stream(0)
    assert SAMPLER.sample_table(one({"type": "minecraft:alternatives", "children": [never, stick, coal]}), s) == \
        {iid("stick"): 1}
    group = {"type": "minecraft:group", "children": [stick, coal]}  # two candidates: one is drawn by weight
    picks = Counter(next(iter(SAMPLER.sample_table(one(group), stream(i)))) for i in range(4000))
    assert_rate(picks[iid("stick")], 4000, 0.5)
    seq = {"type": "minecraft:sequence", "children": [stick, never, coal]}  # stops at the failing child
    assert all(next(iter(SAMPLER.sample_table(one(seq), stream(i)))) == iid("stick") for i in range(50))
    weighted = {"type": "minecraft:generic", "pools": [{"rolls": 1.0, "entries": [
        {**stick, "weight": 3}, {**coal, "weight": 1}, {"type": "minecraft:empty", "weight": 4}]}]}
    got = Counter(tuple(SAMPLER.sample_table(weighted, stream(i, episode=9))) for i in range(8000))
    assert_rate(got[(iid("stick"),)], 8000, 3 / 8)
    assert_rate(got[(iid("coal"),)], 8000, 1 / 8)
    assert_rate(got[()], 8000, 0.5)


def test_unsupported_parts_raise():
    with pytest.raises(UnsupportedLoot, match="function"):
        validate_table(item_table({"function": "minecraft:set_enchantments"}))
    with pytest.raises(UnsupportedLoot, match="condition"):
        validate_table(item_table(conditions=[{"condition": "minecraft:weather_check", "raining": True}]))
    with pytest.raises(UnsupportedLoot, match="keys"):
        validate_table(item_table({"function": "minecraft:set_count", "count": 1.0, "surprise": 1}))
    with pytest.raises(UnsupportedLoot, match="provider"):
        validate_table(item_table({"function": "minecraft:set_count", "count": {"type": "minecraft:score"}}))
    with pytest.raises(UnsupportedLoot):
        validate_table({"pools": [{"rolls": 1.0, "entries": [{"type": "minecraft:dynamic", "name": "contents"}]}]})
    with pytest.raises(UnsupportedLoot, match="frog"):
        validate_table(R.loot_table("entities/magma_cube"))  # not a v1 mob: froglights by frog variant



LOCATION_TABLE = {"type": "minecraft:block", "pools": [{"rolls": 1.0, "conditions": [
    {"condition": "minecraft:location_check", "offsetY": 1, "predicate": {"block": {"blocks": "self"}}}],
    "entries": [{"type": "minecraft:item", "name": "stick"}]}]}
STATE_TABLE = {"type": "minecraft:block", "pools": [{"rolls": 1.0, "entries": [{
    "type": "minecraft:item", "name": "stick",
    "conditions": [{"condition": "minecraft:block_state_property", "block": "self",
                    "properties": {"age": {"min": 5}}}]}]}]}


def test_location_and_state_conditions_need_their_context():
    with pytest.raises(UnsupportedLoot, match="location"):
        SAMPLER.sample_table(LOCATION_TABLE, stream(0), level="class")
    with pytest.raises(UnsupportedLoot, match="state"):
        SAMPLER.sample_table(STATE_TABLE, stream(0), level="class")
    above = SAMPLER.sample_table(LOCATION_TABLE, stream(0), level="class", location_check=lambda c: c["offsetY"] == 1)
    assert above == {iid("stick"): 1}
    assert SAMPLER.sample_table(LOCATION_TABLE, stream(0), level="class", location_check=lambda c: False) == {}
    assert SAMPLER.sample_table(STATE_TABLE, stream(0), level="class", block_state={"age": "7"}) == {iid("stick"): 1}
    assert SAMPLER.sample_table(STATE_TABLE, stream(0), level="class", block_state={"age": "2"}) == {}


def test_embedded_tables_match_the_pinned_sources():
    files = {
        "gameplay/fishing/fish": MCMETA / "loot_table" / "gameplay" / "fishing" / "fish.json",
        "skeletons": MCMETA / "tags" / "entity_type" / "skeletons.json",
        "smelts_loot": MCMETA / "tags" / "enchantment" / "smelts_loot.json",
    }
    if not all(p.exists() for p in files.values()):
        pytest.skip("source cache not present (python -m hivemind.rules.fetch_sources)")
    assert EXTRA_LOOT_TABLES["gameplay/fishing/fish"] == json.loads(files["gameplay/fishing/fish"].read_text())
    tag = [v.removeprefix("minecraft:") for v in json.loads(files["skeletons"].read_text())["values"]]
    assert ENTITY_TYPE_TAGS["skeletons"] == tuple(tag)
    ench = [v.removeprefix("minecraft:") for v in json.loads(files["smelts_loot"].read_text())["values"]]
    assert ENCHANTMENT_TAGS["smelts_loot"] == tuple(ench)

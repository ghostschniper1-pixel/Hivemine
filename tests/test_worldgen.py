"""Tests for the test-world generator (hivemind.sim.ref.worldgen) and its ore table (ore_table)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from hivemind.rules import load_concepts
from hivemind.rules.fetch_sources import verify_all
from hivemind.sim.ref import ore_table as ot
from hivemind.sim.ref import worldgen as wg
from hivemind.sim.ref.rng import CounterRNG
from hivemind.sim.ref.world import MIN_Y

REPO = Path(__file__).resolve().parents[1]
SOURCES_JSON = REPO / "data" / "sources.json"
SOURCES_DIR = REPO / "data" / "sources"
C = load_concepts()
ID = {c.name: c.id for c in C.block_classes}
FAMILIES = {f: (ID[f"{f}_ore"], ID[f"deepslate_{f}_ore"])
            for f in ("coal", "iron", "copper", "gold", "redstone", "lapis", "diamond")}


@pytest.fixture(scope="module")
def flat_worlds():
    """Two flat worlds without stone blobs or caves: every ore block comes from an ore row."""
    cfg = wg.preset("flat", stone_blobs=False)
    return [wg.generate(seed, cfg) for seed in (11, 12)]


@pytest.fixture(scope="module")
def default_world():
    return wg.generate(3, "default")


@pytest.fixture(scope="module")
def forest():
    return wg.generate(5, "forest")


# -- the ore table ------------------------------------------------------------------------


def _sources_ok() -> bool:
    return SOURCES_DIR.is_dir() and not verify_all(SOURCES_JSON, SOURCES_DIR)


@pytest.mark.skipif(not _sources_ok(), reason="data/sources/ missing or not matching the pins")
def test_ore_table_regenerates_from_pinned_sources():
    assert ot.generate_table() == ot.ORE_TABLE
    assert ot.render_rows(ot.ORE_TABLE) in Path(ot.__file__).read_text(encoding="utf-8")


def test_ore_table_contents():
    rows = {r.name: r for r in ot.ORE_TABLE}
    iron = rows["ore_iron_middle"]
    assert (iron.count_min, iron.height, iron.min_y, iron.max_y, iron.size) == (10, "trapezoid", -24, 56, 9)
    assert iron.targets[0].place == "iron_ore" and iron.targets[1].place == "deepslate_iron_ore"
    assert set(iron.targets[1].replaces) == {"deepslate", "tuff"}
    assert set(iron.targets[0].replaces) == {"stone", "granite", "diorite", "andesite"}
    assert rows["ore_diamond_buried"].discard_on_air == 1.0
    assert rows["ore_diamond_large"].rarity == 9
    assert (rows["ore_gold_lower"].count_min, rows["ore_gold_lower"].count_max) == (0, 1)
    assert rows["ore_emerald"].in_biome("windswept_hills") and not rows["ore_emerald"].in_biome("plains")
    assert rows["ore_copper"].in_biome("plains") and not rows["ore_copper"].in_biome("dripstone_caves")
    plains = {r.name for r in ot.ore_table("plains")}
    assert "ore_emerald" not in plains and "ore_coal_lower" in plains and len(plains) == 25
    for r in ot.ORE_TABLE:  # every placed class is a real class with an honest phrase
        for t in r.targets:
            assert C.block_class(t.place).name == t.place


@pytest.mark.parametrize("name", ["ore_iron_middle", "ore_redstone", "ore_diamond", "ore_coal_lower"])
def test_height_pmf_is_a_distribution(name):
    row = {r.name: r for r in ot.ORE_TABLE}[name]
    pmf = ot.height_pmf(row)
    assert math.isclose(sum(pmf.values()), 1.0)
    assert min(pmf) == row.min_y and max(pmf) == row.max_y
    if row.height == "trapezoid":
        mid = (row.min_y + row.max_y) // 2
        assert pmf[mid] > pmf[row.min_y] * 10  # a triangle peaked in the middle


@pytest.mark.parametrize("name", ["ore_iron_middle", "ore_redstone", "ore_diamond", "ore_gold_lower",
                                  "ore_diorite_upper"])
def test_attempts_follow_the_table(name):
    """Counts, rarity and origin heights of the generator's attempts match the table."""
    rows = ot.ORE_TABLE
    fi = next(i for i, r in enumerate(rows) if r.name == name)
    row = rows[fi]
    rng = CounterRNG(99)
    chunks = [(cx, cz) for cx in range(40) for cz in range(40)]
    origins = []
    for chunk in chunks:
        attempts = wg.ore_attempts(rng, row, fi, chunk)
        for x, _, z in attempts:
            assert chunk[0] * 16 <= x < chunk[0] * 16 + 16 and chunk[1] * 16 <= z < chunk[1] * 16 + 16
        origins += [y for _, y, _ in attempts]
    n_expected = row.expected_attempts() * len(chunks)
    sd = math.sqrt(len(chunks) * (row.count_max - row.count_min + 1) ** 2 / 4 + n_expected) + 1
    assert abs(len(origins) - n_expected) < 4 * sd
    ys = np.asarray(origins)
    assert ys.min() >= row.min_y and ys.max() <= row.max_y
    pmf = ot.height_pmf(row)
    support = np.arange(row.min_y, row.max_y + 1)
    cdf = np.cumsum([pmf.get(int(y), 0.0) for y in support])
    emp = np.searchsorted(np.sort(ys), support, side="right") / len(ys)
    assert np.abs(emp - cdf).max() < 1.63 / math.sqrt(len(ys)) * 1.5  # Kolmogorov-Smirnov, ~1% level, slack


def test_ore_density_knob_scales_attempts():
    row = next(r for r in ot.ORE_TABLE if r.name == "ore_iron_middle")
    rng = CounterRNG(5)
    chunks = [(cx, cz) for cx in range(20) for cz in range(20)]
    for density in (0.25, 0.5, 2.0):
        n = sum(len(wg.ore_attempts(rng, row, 12, c, density)) for c in chunks)
        assert abs(n / (10 * density * len(chunks)) - 1.0) < 0.05
    blob = next(r for r in ot.ORE_TABLE if r.name == "ore_granite_lower")
    assert not blob.is_ore
    assert sum(len(wg.ore_attempts(rng, blob, 3, c, 0.25)) for c in chunks) == 2 * len(chunks)  # not scaled


# -- determinism ------------------------------------------------------------------------------


def test_same_seed_same_world_different_seed_differs(forest):
    again = wg.generate(5, "forest")
    assert np.array_equal(forest.blocks, again.blocks)
    other = wg.generate(6, "forest")
    assert (forest.blocks != other.blocks).mean() > 0.01
    assert forest.meta["spawn"] == again.meta["spawn"]


# -- terrain ----------------------------------------------------------------------------------


def test_bedrock_and_deepslate_strata(flat_worlds):
    w = flat_worlds[0]
    b = w.blocks
    assert (b[:, 0, :] == ID["bedrock"]).all()  # y -64
    bedrock_y = np.nonzero(b == ID["bedrock"])[1] + MIN_Y
    assert bedrock_y.max() <= -60
    frac = [(b[:, y - MIN_Y, :] == ID["bedrock"]).mean() for y in (-63, -62, -61, -60)]
    assert frac[0] > frac[1] > frac[2] > frac[3] > 0  # the rough layer thins out upwards
    rock = lambda y: b[:, y - MIN_Y, :]  # noqa: E731
    assert (rock(-30) == ID["deepslate"]).mean() > 0.9
    assert not (rock(20) == ID["deepslate"]).any()
    band = [(rock(y) == ID["deepslate"]).mean() for y in range(0, 9)]
    assert band[0] > 0.9 and band[-1] < 0.1 and band[2] > band[6]


def test_flat_surface_soil(flat_worlds):
    w = flat_worlds[0]
    assert (w.heightmap == 64).all()
    assert (w.blocks[:, 64 - MIN_Y, :] == ID["grass_block"]).all()
    assert (w.blocks[:, 63 - MIN_Y, :] == ID["dirt"]).all() and (w.blocks[:, 61 - MIN_Y, :] == ID["dirt"]).all()
    assert (w.blocks[:, 65 - MIN_Y:, :] == 0).all()
    four_deep = (w.blocks[:, 60 - MIN_Y, :] == ID["dirt"]).mean()  # 3 or 4 dirt layers [est]
    assert 0.3 < four_deep < 0.7 and not (w.blocks[:, 59 - MIN_Y, :] == ID["dirt"]).any()


def test_rolling_surface_range(default_world):
    h = default_world.meta["surface"]
    assert 55 <= h.min() and h.max() <= 90  # [est] mostly 64-80
    assert 64 <= np.median(h) <= 80


def test_water_sand_trees_caves_lava(default_world):
    w = default_world
    b = w.blocks
    y = np.arange(MIN_Y, 192)[None, :, None]
    water = b == ID["water"]
    assert water.any()
    assert (b == ID["sand"]).any()  # beaches and pond floors
    lava_y = np.nonzero(b == ID["lava"])[1] + MIN_Y
    assert lava_y.size == 0 or lava_y.max() <= wg.LAVA_LEVEL
    under = y < w.meta["surface"][:, None, :] - 2
    cave = (b == 0) & under
    assert 0.002 < cave.sum() / under.sum() < 0.1  # [est]
    assert w.meta["trees"] > 20


def test_caves_expose_some_ore(default_world):
    w = default_world
    ores = np.isin(w.blocks, [i for pair in FAMILIES.values() for i in pair])
    air = wg._air_neighbour(w.blocks, 0, MIN_Y, 0, w.blocks.shape)
    exposed = (ores & air).sum() / ores.sum()
    assert 0.002 < exposed < 0.2  # [est] until M3 measures exposure on real chunks


def test_trees_have_trunks_4_to_6_and_honest_leaves(forest):
    b = forest.blocks
    log, oak, birch = ID["log"], ID["oak_leaves"], ID["leaves"]
    assert (b == oak).sum() > (b == birch).sum() > 0
    xs, ys, zs = np.nonzero(b == log)
    columns = {(x, z) for x, z in zip(xs.tolist(), zs.tolist())}
    assert len(columns) == forest.meta["trees"]
    for x, z in columns:
        heights = np.nonzero(b[x, :, z] == log)[0]
        assert 4 <= len(heights) <= 6 and np.all(np.diff(heights) == 1)
        top = heights[-1]
        assert b[x, top + 1, z] in (oak, birch)  # the canopy sits on the trunk
        assert b[x, heights[0] - 1, z] == ID["dirt"]


def test_every_voxel_is_a_world_class(default_world):
    assert C.block_defined[default_world.blocks].all()
    assert not (default_world.blocks == 255).any()


# -- ores -------------------------------------------------------------------------------------


def test_deepslate_variants_replace_deepslate(flat_worlds):
    for w in flat_worlds:
        for stone_ore, deep_ore in FAMILIES.values():
            ys_stone = np.nonzero(w.blocks == stone_ore)[1] + MIN_Y
            ys_deep = np.nonzero(w.blocks == deep_ore)[1] + MIN_Y
            if ys_deep.size:
                assert ys_deep.max() <= 8  # deepslate exists only up to y 8
            if ys_stone.size:
                assert ys_stone.min() >= 0  # no granite blobs here: stone only from y 0


def test_ore_counts_per_y_band_match_the_table(flat_worlds):
    """Ore blocks per Y band agree with count x P(band) x mean vein size from the table."""
    stats = {}
    rows = ot.ore_table("plains")
    for r in rows:
        if r.size not in stats:
            stats[r.size] = wg.vein_size_stats(r.size, n=300)
    chunks = sum(w.chunks_x * w.chunks_z for w in flat_worlds)
    bands = [(-58, -30), (-29, -1), (0, 30), (31, 55)]  # inside the stone of a flat world (bedrock < -59, dirt > 60)
    checked = 0
    total_exp = total_act = 0.0
    for family, ids in FAMILIES.items():
        family_rows = [r for r in rows if r.targets[0].place == f"{family}_ore"]
        ys = np.concatenate([np.nonzero(np.isin(w.blocks, ids))[1] + MIN_Y for w in flat_worlds])
        for lo, hi in bands:
            exp = var = 0.0
            for r in family_rows:
                n = chunks * r.expected_attempts() * ot.band_probability(r, lo, hi)
                exp += n * stats[r.size][0]
                var += n * stats[r.size][1]  # sum of n Bernoulli-thinned vein sizes (an upper bound)
            act = int(((ys >= lo) & (ys <= hi)).sum())
            total_exp += exp
            total_act += act
            if exp < 50:
                assert act < 50 + 4 * math.sqrt(var + 1)
                continue
            checked += 1
            assert abs(act - exp) < 4 * math.sqrt(var), (family, lo, hi, act, exp)
    assert checked >= 14
    assert abs(total_act / total_exp - 1) < 0.1


def test_ore_density_knob_scales_blocks():
    base = wg.preset("flat", size_x=32, size_z=32, stone_blobs=False)
    one = wg.generate(4, base)
    two = wg.generate(4, wg.preset("flat", size_x=32, size_z=32, stone_blobs=False, ore_density=2.0))
    quarter = wg.generate(4, wg.preset("flat", size_x=32, size_z=32, stone_blobs=False, ore_density=0.25))
    ores = np.array([i for pair in FAMILIES.values() for i in pair])
    n = [int(np.isin(w.blocks, ores).sum()) for w in (quarter, one, two)]
    assert 0.15 < n[0] / n[1] < 0.35 and 1.7 < n[2] / n[1] < 2.3


def test_buried_diamonds_avoid_air():
    """discard_chance_on_air_exposure 1.0 never places next to air; 0.0 ignores air."""
    rng = CounterRNG(1)
    stone, ore = ID["stone"], ID["diamond_ore"]
    targets = wg.OreTargets((np.isin(np.arange(256), [stone]),), (ore,))
    for discard, may_touch in ((1.0, False), (0.0, True)):
        touching = 0
        for i in range(40):
            blocks = np.full((24, 256, 24), stone, np.uint8)
            blocks[:, 0 - MIN_Y:, :] = 0  # air from y 0 up
            wg.place_vein(blocks, (12, 0, 12), 8, discard, targets, rng, wg.vein_key(0, i, (0, 0)))
            ys = np.nonzero(blocks == ore)[1] + MIN_Y
            touching += int((ys == -1).sum())
        assert (touching > 0) == may_touch


# -- biomes per column (draft 0.5) ----------------------------------------------------------------


def _ids_of(names):
    return {C.biome(n).id for n in names}


def test_biome_bands_are_deterministic_and_seeded():
    cfg = wg.WorldGenConfig()
    a, b = wg.biome_indices(CounterRNG(3), cfg), wg.biome_indices(CounterRNG(3), cfg)
    assert np.array_equal(a, b) and a.shape == (128, 128)
    assert (wg.biome_indices(CounterRNG(4), cfg) != a).mean() > 0.2
    small = wg.preset("default", size_x=48, size_z=48, caves_per_chunk=0.0, ores=False)
    assert np.array_equal(wg.generate(9, small).biome_map, wg.generate(9, small).biome_map)


def test_biome_regions_are_smooth():
    """Regions of one biome are large and contiguous, and neighbouring bands border each other."""
    cfg = wg.WorldGenConfig()
    for seed in range(10):
        b = wg.biome_indices(CounterRNG(seed), cfg)
        _, sizes = wg.column_regions(b)
        assert min(sizes) >= wg.MIN_BIOME_COLUMNS == 24, seed
        assert len(sizes) <= 16, seed
        jumps = np.concatenate([np.abs(np.diff(b, axis=0)).ravel(), np.abs(np.diff(b, axis=1)).ravel()])
        assert jumps.max() <= 1, seed  # a desert borders plains, not a forest
        assert (jumps > 0).mean() < 0.05, seed  # under 5% of neighbouring columns differ


def test_merge_small_regions():
    b = np.ones((10, 10), np.int64)
    b[4, 4] = 0  # a single-column speck
    b[0:2, 7:9] = 3  # a 4-column island in the corner
    out = wg.merge_small_regions(b, 5)
    assert (out == 1).all()
    tie = np.zeros((6, 6), np.int64)
    tie[:, 3:] = 2
    tie[2, 2:4] = 1  # two columns, bordered by three 0s and three 2s: the lower band wins
    merged = wg.merge_small_regions(tie, 4)
    assert merged[2, 2] == 0 and merged[2, 3] == 0
    one = np.full((5, 5), 2, np.int64)
    assert np.array_equal(wg.merge_small_regions(one, 100), one)  # the whole world: nothing to join


def test_biomes_are_true_of_the_terrain(default_world):
    w = default_world
    b = w.blocks
    names = {C.biome(i).name for i in np.unique(w.biome_map).tolist()}
    assert names <= set(wg.biome_palette(wg.PRESETS["default"])) and len(names) >= 3
    assert not names & wg.UNDERGROUND_BIOMES and all(C.biome(n).overworld for n in names)
    h = w.meta["surface"]
    xs, zs = np.meshgrid(np.arange(w.size_x), np.arange(w.size_z), indexing="ij")
    top = b[xs, h - MIN_Y, zs]
    desert = w.biome_map == C.biome("desert").id
    green = np.isin(w.biome_map, list(_ids_of(("plains", "forest", "birch_forest"))))
    assert desert.sum() > 500 and green.sum() > 500
    assert np.isin(top[desert], [ID["sand"], ID["gravel"]]).all()  # sand on top (gravel only deep under water)
    assert (top[green] == ID["grass_block"]).mean() > 0.95  # the rest: beaches, ponds, dirt under trunks
    assert not (top[green] == ID["sandstone"]).any()
    ys = np.arange(MIN_Y, MIN_Y + b.shape[1])[None, :, None]
    sandstone = (b == ID["sandstone"]).any(axis=1)
    near_top = ((b == ID["sandstone"]) & (ys >= h[:, None, :] - 8) & (ys < h[:, None, :] - 2)).any(axis=1)
    assert (near_top[desert]).mean() > 0.95  # sandstone under the desert's sand [mem]
    assert not sandstone[green].any()
    # Plants and trees follow the column's biome.
    above = b[xs, np.minimum(h + 1, 191) - MIN_Y, zs]
    assert not np.isin(above[desert], [ID["short_grass"], ID["flower"], ID["log"]]).any()
    assert (above[desert] == ID["dead_bush"]).any() and not (above[~desert] == ID["dead_bush"]).any()
    trunks = (b[xs, h + 1 - MIN_Y, zs] == ID["log"]) & (b[xs, h - MIN_Y, zs] == ID["dirt"])
    assert not trunks[desert].any()
    forest = w.biome_map == C.biome("forest").id
    plains = w.biome_map == C.biome("plains").id
    assert trunks[forest].mean() > 5 * trunks[plains].mean() > 0  # forests are dense, plains sparse


def test_birch_forest_grows_only_birch():
    w = wg.generate(5, wg.preset("forest", biomes=("birch_forest",)))
    assert (w.biome_map == C.biome("birch_forest").id).all() and w.meta["trees"] > 10
    assert (w.blocks == ID["leaves"]).sum() > 0 and not (w.blocks == ID["oak_leaves"]).any()


def test_presets_name_one_biome(flat_worlds, forest):
    assert (flat_worlds[0].biome_map == C.biome("plains").id).all()
    assert (forest.biome_map == C.biome("forest").id).all()


@pytest.mark.parametrize("overrides", [
    {"biomes": ("lush_caves",)},  # underground: names a cave, not a column
    {"biomes": ("plains", "nether_wastes")},  # not an overworld biome
    {"biomes": ("badlands",)},  # no surface for it, and it lists extra gold rows
    {"biomes": ("plains", "plains")},
    {"biomes": ()},
    {"biome": "badlands", "biomes": ("plains",)},  # badlands rows placed under a plains name
    {"biome": "windswept_hills"},  # emerald rows under plains, desert and forest names
    {"sea_biome": "deep_dark"},
])
def test_check_biomes_rejects_false_names(overrides):
    with pytest.raises(ValueError):
        wg.check_biomes(wg.preset("default", **overrides), C)


def test_sea_columns_are_ocean_ponds_keep_their_land_biome():
    cfg = wg.preset("default", size_x=64, size_z=64, surface_y=59.0, caves_per_chunk=0.0, ores=False,
                    stone_blobs=False, pools_per_chunk=0.0)
    w = wg.generate(2, cfg)
    h = w.meta["surface"]
    sea = h < wg.WATER_TOP
    ocean = w.biome_map == C.biome("ocean").id
    labels, sizes = wg.column_regions(sea)
    big = np.isin(labels, [r for r, n in enumerate(sizes) if n >= wg.SEA_COLUMNS]) & sea
    assert big.sum() > 500 and np.array_equal(ocean, big)  # every sea, and only seas
    assert (w.blocks[:, wg.WATER_TOP - MIN_Y, :][ocean] == ID["water"]).all()
    xs, zs = np.nonzero(ocean)
    assert np.isin(w.blocks[xs, h[xs, zs] - MIN_Y, zs], [ID["sand"], ID["gravel"]]).all()
    small = sea & ~big
    if small.any():
        assert not ocean[small].any()  # small lakes keep the land biome
    dry = wg.generate(2, wg.preset("default", size_x=64, size_z=64, surface_y=59.0, caves_per_chunk=0.0,
                                   ores=False, stone_blobs=False, pools_per_chunk=0.0, sea_biome=None))
    assert not (dry.biome_map == C.biome("ocean").id).any()
    # The seas cut the land: the pieces left smaller than MIN_BIOME_COLUMNS that touch other land
    # join a neighbouring land biome, and no other land column changes its biome.
    cut = np.where(ocean, C.biome("ocean").id, dry.biome_map)
    labels, sizes = wg.column_regions(cut)
    specks = np.zeros_like(ocean)
    for r, n in enumerate(sizes):
        piece = labels == r
        if n < wg.MIN_BIOME_COLUMNS and not ocean[piece].any() and (wg._neighbours(piece) & ~ocean).any():
            specks |= piece
    assert np.array_equal((dry.biome_map != w.biome_map) & ~ocean, specks)
    differs = (dry.blocks != w.blocks).any(axis=1)  # e.g. sandstone under a sea floor in a desert band
    assert not differs[~ocean & ~specks].any()  # nothing else changes when the seas keep their land biome


@pytest.mark.parametrize("seed", [0, 1])
def test_no_land_biome_is_a_speck_beside_the_sea(seed):
    """A low world, whose seas cut the land bands into pieces: every land region of the final
    biome map has at least MIN_BIOME_COLUMNS columns, except an island with no land around it,
    and each column's surface is still true of its biome. (Seeds 0 and 1 left desert and forest
    regions of 1 to 13 columns along the coasts before the land was merged again.)"""
    cfg = wg.preset("default", surface_y=62.0, caves_per_chunk=0.0, ores=False, stone_blobs=False, vegetation=0.0)
    w = wg.generate(seed, cfg)
    ocean = w.biome_map == C.biome("ocean").id
    assert ocean.sum() > 1000 and (~ocean).sum() > 1000
    labels, sizes = wg.column_regions(w.biome_map)
    for r, n in enumerate(sizes):
        piece = labels == r
        if n < wg.MIN_BIOME_COLUMNS:
            assert not ocean[piece].any() and not (wg._neighbours(piece) & ~ocean).any(), (seed, n)  # an island
    h = w.meta["surface"]
    xs, zs = np.meshgrid(np.arange(w.size_x), np.arange(w.size_z), indexing="ij")
    top = w.blocks[xs, h - MIN_Y, zs]
    desert = w.biome_map == C.biome("desert").id
    assert np.isin(top[desert], [ID["sand"], ID["gravel"]]).all()
    green = np.isin(w.biome_map, list(_ids_of(("plains", "forest", "birch_forest"))))
    assert not np.isin(top[green], [ID["sandstone"]]).any()
    assert np.isin(top[green], [ID["grass_block"], ID["sand"], ID["gravel"]]).all()  # sand on beaches


def test_merge_small_regions_keeps_fixed_columns():
    b = np.zeros((8, 8), np.int64)
    b[:, 4:] = 1
    sea = np.zeros((8, 8), bool)
    sea[3:5, :] = True  # a strait across the middle
    sea[2, 0] = sea[2, 2] = True  # two inlets of it
    b[sea] = 9
    b[2, 1] = 1  # a speck of band 1 between the inlets: three sea neighbours, one land
    out = wg.merge_small_regions(b, 4, fixed=sea)
    assert (out[sea] == 9).all()  # fixed columns keep their band
    assert out[2, 1] == 0  # the speck joined its land neighbour, not the sea's band 9
    assert wg.merge_small_regions(b, 4)[2, 1] == 9  # without ``fixed`` the sea's band would win
    assert (out[~sea] == np.where(np.arange(8)[None, :] < 4, 0, 1).repeat(8, 0)[~sea]).all()
    island = np.full((6, 6), 9, np.int64)
    island[2:4, 2:4] = 1  # a 4-column island with only fixed columns around it
    assert np.array_equal(wg.merge_small_regions(island, 10, fixed=island == 9), island)


# -- spawn ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["flat", "forest", "default"])
def test_spawn_on_surface(which, flat_worlds, forest, default_world):
    w = {"flat": flat_worlds[0], "forest": forest, "default": default_world}[which]
    x, y, z = w.meta["spawn"]
    assert x % 1 == 0.5 and z % 1 == 0.5 and y == int(y)
    xi, yi, zi = int(x), int(y), int(z)
    assert yi == w.heightmap[xi, zi] + 1
    ground = w.get(xi, yi - 1, zi)
    assert C.solid[ground] and not C.hazard[ground] and not C.liquid[ground]
    for dy in (0, 1):
        cell = w.get(xi, yi + dy, zi)
        assert C.collision_height[cell] == 0 and not C.liquid[cell]
    assert wg.spawn_point(w) == (x, y, z)

"""Deterministic test worlds from a seed, until M3 replaces them with real chunks (contracts 5).

DESIGN.md 4.4 ("world sources") plans two sources: real chunks pregenerated on a 1.21.1 server,
and a statistical generator fitted to them (M3). Until then this module makes plausible test
worlds so the rest of the simulator can be built and tested. It does **not** reimplement the
game's noise worldgen (DESIGN.md 4.2). What it takes from the game is the ore placement of the
pinned 1.21.1 data (``ore_table.py``: counts, height ranges and distributions, vein sizes, air
exposure, stone and deepslate targets) and a few constants of the game's surface rules
(labelled [3] or [mem]). Every other number is an estimate ([est]) to be replaced by the
statistics measured in M3.

    from hivemind.sim.ref.worldgen import generate, spawn_point
    world = generate(seed=7)                    # the 128 x 128 default preset
    world = generate(seed=7, config="forest")   # or "flat", or a WorldGenConfig
    x, y, z = spawn_point(world)                # feet position on the surface
    world.biome_map[x, z]                       # the column's biome id (draft 0.5)

Steps, in order (all randomness through ``CounterRNG`` with ``Purpose.WORLDGEN``, keyed by the
step, the chunk or lattice point, and a draw index, so the result does not depend on the order
of evaluation and two seeds never share numbers):

1. **Height:** value noise (four octaves, smoothstep interpolation of lattice values) around
   y 70 [est], so the surface lies mostly between 64 and 80 [est]; ``flat`` uses one height.
2. **Water:** columns below sea level fill with water up to y 62 (sea level 63 [3]); a few dug
   pools [est] add small ponds above sea level.
3. **Biomes** (draft 0.5; DESIGN.md 4.4): one per column from a smooth value-noise field [est]
   cut into bands along ``WorldGenConfig.biomes`` (default desert, plains, forest, birch forest,
   so a desert borders plains and a birch forest borders a forest); a region smaller than
   ``MIN_BIOME_COLUMNS`` = 24 columns [est] joins its neighbours' biome, so no biome is a speck.
   Each biome must be true of its column: ``BIOME_SURFACES`` decides the column's surface and
   plants (below), and every biome of the palette must list exactly the ore-table rows that
   ``WorldGenConfig.biome`` places (true of plains, forest, birch forest, desert and ocean;
   ``generate`` checks it). Only overworld biomes, and never an underground one (``lush caves``,
   ``dripstone caves``, ``deep dark``). A sea-level water body of at least ``SEA_COLUMNS`` = 256
   columns [est] is a sea and takes ``sea_biome`` (``ocean``); ponds and small sea-level lakes
   keep their land biome, as a pond in a plain is plains in the game. The default terrain seldom
   dips below sea level, so default worlds rarely hold a sea.
4. **Strata:** bedrock at y -64 with the vanilla rough layer (a gradient to y -59 [mem]),
   deepslate at y <= 0 with a transition band up to y 8 [mem], stone above, then 3-4 layers [est]
   of the biome's soil under its top block: dirt under a grass block, or in a desert sand under
   sand over 3 layers of sandstone [mem: the desert surface rule puts sandstone under the sand;
   depths est]; sand within 2 blocks [est] of water near water level, sand or gravel [est] on the
   floor under water.
5. **Caves:** random worms [est] (a Poisson number per chunk) carve tunnels so ores get
   exposed; carved cells at y <= -56 hold lava (the cave carver's ``lava_level`` [3]); cells next
   to water are left alone.
6. **Ores:** per chunk, every row of the ore table for the biome (``plains`` by default): count,
   rarity, origin height (uniform or trapezoid), then a vein shaped as the game's ``OreFeature``
   [mem] that replaces only its targets (``deepslate iron ore`` replaces deepslate and tuff),
   with ``discard_chance_on_air_exposure`` against cave air. ``ore_density`` scales the number
   of ore attempts (DESIGN.md 10.2 knob x{0.25, 0.5, 1, 2}); stone blobs (granite, dirt, ...)
   are not scaled.
7. **Trees:** log trunks 4-6 high [est] under a leaf canopy [est] (oak with ``oak leaves``, birch
   with ``leaves``, so each leaf class's words stay true of it), as many per chunk as the column's
   biome has [est] (forests dense, plains rare, deserts none; a birch forest only birch, a forest
   mostly oak [mem]); then short grass and flowers on grass blocks and dead bushes on desert sand.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from hivemind.rules import load_concepts, load_rules
from hivemind.sim.ref.ore_table import OreFeature, ore_table
from hivemind.sim.ref.rng import CounterRNG, DrawKey, DrawStream, Purpose
from hivemind.sim.ref.world import CHUNK_SIZE, HEIGHT, MAX_Y, MIN_Y, World

if TYPE_CHECKING:
    from hivemind.rules.classes import Concepts

SEA_LEVEL = 63  # [3] noise_settings/overworld.json "sea_level": 63
WATER_TOP = SEA_LEVEL - 1  # [mem] the sea's top water block is at y 62
LAVA_LEVEL = -56  # [3] configured_carver/cave.json lava_level {"above_bottom": 8}; cells at or below hold lava [mem]
BEDROCK_ALWAYS_Y = -64  # [mem] surface rule bedrock_floor: always bedrock at and below y -64 ...
BEDROCK_NEVER_Y = -59  # [mem] ... never at and above y -59, a linear gradient between
DEEPSLATE_ALWAYS_Y = 0  # [mem] surface rule "deepslate": always at and below y 0 ...
DEEPSLATE_NEVER_Y = 8  # [mem] ... never at and above y 8, a linear gradient between

# RNG layout: episode = step, tick = sub-index (octave, ore row, ...), entity = chunk or lattice key.
STEP_HEIGHT, STEP_SOIL, STEP_POOLS, STEP_CAVES, STEP_ORES, STEP_VEINS, STEP_TREES, STEP_PLANTS, STEP_BIOMES = range(1, 10)
_DISCARD_DRAW0 = 1 << 20  # draw indices of a vein's per-cell air-exposure draws start here

# DESIGN.md 4.4 [design]: biomes a column biome never names (they are caves, in the game's 3-D cells).
UNDERGROUND_BIOMES = frozenset({"lush_caves", "dripstone_caves", "deep_dark"})
MIN_BIOME_COLUMNS = 24  # [est] a land biome region (4-connected columns) is never smaller than this
SEA_COLUMNS = 256  # [est] a sea-level water body of at least this many columns is a sea (``sea_biome``)


@dataclass(frozen=True)
class BiomeSurface:
    """What a test-world biome builds on its columns, so that its name is true of them (DESIGN.md
    4.4). Densities are per chunk or per column, all [est] until M3 measures real chunks."""

    top: str  # block on top of a dry column
    soil: str  # the dirt_depth layers under it
    under: str | None  # below the soil, ``under_depth`` layers
    under_depth: int
    trees_per_chunk: float  # mean tree attempts per chunk where the whole chunk has this biome
    birch_fraction: float  # birch trees among them (the rest oak)
    grass_chance: float  # short grass per grass-block column with air above
    flower_chance: float
    dead_bush_chance: float  # dead bushes per sand column with air above


BIOME_SURFACES: dict[str, BiomeSurface] = {
    # [est] vanilla plains have a few oaks [mem]
    "plains": BiomeSurface("grass_block", "dirt", None, 0, 0.1, 0.0, 0.3, 0.02, 0.0),
    # [est] vanilla forests mix oak and birch [mem]
    "forest": BiomeSurface("grass_block", "dirt", None, 0, 5.0, 0.2, 0.15, 0.01, 0.0),
    # [est] birch only [mem]
    "birch_forest": BiomeSurface("grass_block", "dirt", None, 0, 5.0, 1.0, 0.15, 0.01, 0.0),
    # [mem] sand over sandstone, no trees or grass, dead bushes; depths and density [est]
    "desert": BiomeSurface("sand", "sand", "sandstone", 3, 0.0, 0.0, 0.0, 0.0, 0.01),
    # Only on sea columns, which are always under water: their floor is sand, or gravel in deeper
    # water (``_strata``) [mem: ocean floors are sand and gravel]; nothing grows there.
    "ocean": BiomeSurface("sand", "sand", None, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class WorldGenConfig:
    """Generator settings. Every default is an estimate [est] unless noted."""

    size_x: int = 128  # DESIGN.md 4.4: single-agent world
    size_z: int = 128
    surface_y: float = 70.0  # [est] mean surface height (surface mostly 64-80)
    octaves: tuple[tuple[int, float], ...] = ((64, 6.0), (32, 3.0), (16, 1.5), (8, 0.75))  # [est] (spacing, amplitude)
    flat: bool = False  # a flat world at round(surface_y)
    dirt_depth: tuple[int, int] = (3, 4)  # [est] soil layers under the top block
    beach_distance: int = 2  # [est] sand within this many blocks of water ...
    beach_height: int = 2  # [est] ... on columns at most this far above the water
    pools_per_chunk: float = 0.05  # [est] dug ponds above sea level
    caves_per_chunk: float = 1.0  # [est] mean number of cave worms starting in a chunk (Poisson)
    cave_length: tuple[int, int] = (40, 100)  # [est] worm steps (blocks)
    cave_radius: tuple[float, float] = (1.5, 3.0)  # [est] worm radius range
    cave_y: tuple[int, int] = (-60, 56)  # [est] worm start heights
    ores: bool = True  # place ore rows of the table
    stone_blobs: bool = True  # place the table's granite, diorite, andesite, tuff, dirt and gravel blobs
    ore_density: float = 1.0  # DESIGN.md 10.2 scarcity knob
    biome: str = "plains"  # which rows of the ore table apply (ore_table.OreFeature.in_biome)
    # The land biomes of the columns, in band order along the biome noise (keys of BIOME_SURFACES) [est].
    biomes: tuple[str, ...] = ("desert", "plains", "forest", "birch_forest")
    biome_octaves: tuple[tuple[int, float], ...] = ((64, 1.0), (32, 0.25))  # [est] (spacing, amplitude)
    biome_spread: float = 0.5  # [est] the bands split [-spread, spread] of the normalized noise evenly
    sea_biome: str | None = "ocean"  # the biome of sea columns (None: they keep their land biome)
    vegetation: float = 1.0  # scales every biome's trees, grass, flowers and dead bushes
    trunk_height: tuple[int, int] = (4, 6)  # [est]


PRESETS: dict[str, WorldGenConfig] = {
    "default": WorldGenConfig(),
    "forest": WorldGenConfig(size_x=64, size_z=64, biomes=("forest",), pools_per_chunk=0.1),  # [est]
    "flat": WorldGenConfig(size_x=64, size_z=64, flat=True, surface_y=64.0, pools_per_chunk=0.0, caves_per_chunk=0.0,
                           biomes=("plains",), vegetation=0.0),  # [est]
}


def preset(name: str, **overrides) -> WorldGenConfig:
    """A named preset (``default``, ``forest``, ``flat``), optionally with fields replaced."""
    return replace(PRESETS[name], **overrides)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _key(a: np.ndarray | int, b: np.ndarray | int) -> np.ndarray | int:
    """A 24-bit entity key from two coordinates (12 bits each, wrapping)."""
    return (np.asarray(a) & 0xFFF) | ((np.asarray(b) & 0xFFF) << 12)


def _ids(concepts: Concepts) -> dict[str, int]:
    return {c.name: c.id for c in concepts.block_classes}


def _class_set(concepts: Concepts, names: Sequence[str]) -> np.ndarray:
    mask = np.zeros(256, bool)
    for n in names:
        mask[concepts.block_class(n).id] = True
    return mask


def _dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    """Grow a 2-D mask by ``steps`` cells (Chebyshev)."""
    out = mask.copy()
    for _ in range(steps):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        grown[1:, 1:] |= out[:-1, :-1]
        grown[:-1, :-1] |= out[1:, 1:]
        grown[1:, :-1] |= out[:-1, 1:]
        grown[:-1, 1:] |= out[1:, :-1]
        out = grown
    return out


def _grow_max(a: np.ndarray, steps: int) -> np.ndarray:
    """Max filter of a 2-D array over a (2 * steps + 1)^2 square."""
    out = a.copy()
    for _ in range(steps):
        g = out.copy()
        g[1:, :] = np.maximum(g[1:, :], out[:-1, :])
        g[:-1, :] = np.maximum(g[:-1, :], out[1:, :])
        out = g
    for _ in range(steps):
        g = out.copy()
        g[:, 1:] = np.maximum(g[:, 1:], out[:, :-1])
        g[:, :-1] = np.maximum(g[:, :-1], out[:, 1:])
        out = g
    return out


def _value_noise(rng: CounterRNG, sx: int, sz: int, spacing: int, octave: int, step: int = STEP_HEIGHT) -> np.ndarray:
    """Smoothly interpolated lattice noise in [-1, 1] on an sx x sz grid (lattice draws keyed by
    ``step`` and ``octave``)."""
    gx = np.arange(sx // spacing + 2)
    gz = np.arange(sz // spacing + 2)
    lattice = rng.uniform_array(step, octave, Purpose.WORLDGEN, _key(gx[:, None], gz[None, :]), 0) * 2.0 - 1.0

    def axis(n: int) -> tuple[np.ndarray, np.ndarray]:
        f = np.arange(n) / spacing
        i = np.floor(f).astype(int)
        t = f - i
        return i, t * t * (3.0 - 2.0 * t)

    ix, tx = axis(sx)
    iz, tz = axis(sz)
    a = lattice[ix][:, iz]
    b = lattice[ix + 1][:, iz]
    c = lattice[ix][:, iz + 1]
    d = lattice[ix + 1][:, iz + 1]
    tx, tz = tx[:, None], tz[None, :]
    return (a * (1 - tx) + b * tx) * (1 - tz) + (c * (1 - tx) + d * tx) * tz


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------


def _heights(rng: CounterRNG, cfg: WorldGenConfig) -> np.ndarray:
    if cfg.flat:
        return np.full((cfg.size_x, cfg.size_z), round(cfg.surface_y), np.int32)
    h = np.full((cfg.size_x, cfg.size_z), cfg.surface_y)
    for octave, (spacing, amplitude) in enumerate(cfg.octaves):
        h += amplitude * _value_noise(rng, cfg.size_x, cfg.size_z, spacing, octave)
    return np.clip(np.rint(h), MIN_Y + 8, MAX_Y - 16).astype(np.int32)  # [est] room for bedrock and trees


def _pools(rng: CounterRNG, cfg: WorldGenConfig, h: np.ndarray, water_top: np.ndarray) -> None:
    """Dig small ponds [est] into ``h`` and raise ``water_top`` inside them (in place)."""
    xs, zs = np.arange(cfg.size_x)[:, None], np.arange(cfg.size_z)[None, :]
    for cx in range(-(-cfg.size_x // CHUNK_SIZE)):
        for cz in range(-(-cfg.size_z // CHUNK_SIZE)):
            s = rng.stream(STEP_POOLS, 0, Purpose.WORLDGEN, _key(cx, cz))
            if not s.bernoulli(cfg.pools_per_chunk):
                continue
            px, pz = cx * CHUNK_SIZE + s.below(CHUNK_SIZE), cz * CHUNK_SIZE + s.below(CHUNK_SIZE)
            radius, depth = s.randint(2, 4), s.randint(1, 3)  # [est]
            dist = np.sqrt((xs - px) ** 2 + (zs - pz) ** 2)
            disk = dist <= radius
            rim = (dist <= radius + 1) & ~disk
            if not disk.any() or (water_top[disk | rim] >= h[disk | rim]).any():
                continue  # touches existing water: leave it
            level = int(h[disk | rim].min())  # the lowest rim keeps the water in
            if level <= WATER_TOP:
                continue
            bowl = np.maximum(1, np.rint(depth * (1.0 - (dist / (radius + 0.5)) ** 2))).astype(np.int32)
            h[disk] = np.minimum(h[disk], level - bowl[disk])
            water_top[disk] = level


def biome_palette(cfg: WorldGenConfig) -> tuple[str, ...]:
    """The biomes a world of ``cfg`` may hold, as indexed by ``column_biomes``: the land biomes
    in band order, then the sea biome if there is one."""
    return cfg.biomes + ((cfg.sea_biome,) if cfg.sea_biome is not None else ())


def check_biomes(cfg: WorldGenConfig, concepts: Concepts) -> None:
    """Raise ValueError unless every biome of ``biome_palette(cfg)`` can be true of its columns
    (module docstring, step 3): an overworld biome that is not underground, with a
    ``BIOME_SURFACES`` entry, listing exactly the ore-table rows of ``cfg.biome``."""
    if not cfg.biomes:
        raise ValueError("a test world needs at least one land biome")
    if len(set(biome_palette(cfg))) != len(biome_palette(cfg)):
        raise ValueError(f"a biome is listed twice in {biome_palette(cfg)}")
    rows = {r.name for r in ore_table(cfg.biome)}
    for name in biome_palette(cfg):
        b = concepts.biome(name)
        if not b.overworld or name in UNDERGROUND_BIOMES:
            raise ValueError(f"{name}: test-world columns take overworld surface biomes only")
        if name not in BIOME_SURFACES:
            raise ValueError(f"{name}: no BIOME_SURFACES entry says what this biome builds")
        if {r.name for r in ore_table(name)} != rows:
            raise ValueError(f"{name} lists other ore rows than {cfg.biome}, whose rows are placed everywhere")


def biome_indices(rng: CounterRNG, cfg: WorldGenConfig) -> np.ndarray:
    """int [size_x, size_z]: each column's land biome, an index into ``cfg.biomes``. The octaves
    of ``cfg.biome_octaves`` are summed and normalized by their amplitudes, then cut into
    ``len(cfg.biomes)`` bands of equal width over [-spread, spread] [est] (the outer bands extend
    to the ends); then every region smaller than ``MIN_BIOME_COLUMNS`` joins a neighbour
    (``merge_small_regions``)."""
    n = len(cfg.biomes)
    if n == 1:
        return np.zeros((cfg.size_x, cfg.size_z), np.int64)
    field = np.zeros((cfg.size_x, cfg.size_z))
    for octave, (spacing, amplitude) in enumerate(cfg.biome_octaves):
        field += amplitude * _value_noise(rng, cfg.size_x, cfg.size_z, spacing, octave, STEP_BIOMES)
    field /= sum(a for _, a in cfg.biome_octaves)
    edges = np.linspace(-cfg.biome_spread, cfg.biome_spread, n + 1)[1:-1]
    return merge_small_regions(np.searchsorted(edges, field, side="right").astype(np.int64), MIN_BIOME_COLUMNS)


def column_regions(values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """The 4-connected regions of equal values of a 2-D array: a label per cell (0, 1, ... in the
    order of each region's first cell in row-major order) and each region's size."""
    sx, sz = values.shape
    labels = np.full((sx, sz), -1, np.int64)
    vals = values.tolist()
    lab = labels.tolist()
    sizes: list[int] = []
    for x0 in range(sx):
        for z0 in range(sz):
            if lab[x0][z0] >= 0:
                continue
            r, v = len(sizes), vals[x0][z0]
            lab[x0][z0] = r
            stack, n = [(x0, z0)], 0
            while stack:
                x, z = stack.pop()
                n += 1
                for a, b in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
                    if 0 <= a < sx and 0 <= b < sz and lab[a][b] < 0 and vals[a][b] == v:
                        lab[a][b] = r
                        stack.append((a, b))
            sizes.append(n)
    return np.asarray(lab, np.int64).reshape(sx, sz), sizes


def _neighbours(mask: np.ndarray) -> np.ndarray:
    """The cells 4-adjacent to a 2-D mask, outside it."""
    out = np.zeros_like(mask)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out & ~mask


def merge_small_regions(bands: np.ndarray, min_columns: int) -> np.ndarray:
    """Give every 4-connected region of fewer than ``min_columns`` columns the band its
    neighbouring columns hold most often (ties: the lower band), smallest region first (ties: the
    first in row-major order), until none is left; a region with no neighbour (the whole world)
    stays. Each pass relabels the regions and merges only regions that touch no region merged
    earlier in the pass, so every merge sees the regions as they are. Deterministic: it reads
    only the bands."""
    bands = bands.copy()
    while True:
        labels, sizes = column_regions(bands)
        small = sorted((n, r) for r, n in enumerate(sizes) if n < min_columns)
        if not small or len(sizes) == 1:
            return bands
        changed = np.zeros(bands.shape, bool)
        for _, r in small:
            mask = labels == r
            around = _neighbours(mask)
            if changed[mask | around].any():
                continue  # next pass, after relabelling
            values, counts = np.unique(bands[around], return_counts=True)
            best = int(values[np.argmax(counts)])  # np.unique sorts: the lower band wins ties
            bands[mask] = best
            changed |= mask


def column_biomes(rng: CounterRNG, cfg: WorldGenConfig, water_top: np.ndarray) -> np.ndarray:
    """int [size_x, size_z]: each column's biome, an index into ``biome_palette(cfg)``: the land
    bands of ``biome_indices``, except that every sea-level water body (4-connected columns whose
    water stands at ``WATER_TOP``, not the dug ponds above it) of at least ``SEA_COLUMNS``
    columns takes ``cfg.sea_biome``. Smaller ones keep their land biome, as a pond in a plain is
    plains in the game."""
    biome = biome_indices(rng, cfg)
    if cfg.sea_biome is None:
        return biome
    sea = water_top == WATER_TOP
    labels, sizes = column_regions(sea)
    big = [r for r, n in enumerate(sizes) if n >= SEA_COLUMNS]
    if big:
        is_sea = np.isin(labels, big) & sea
        biome = np.where(is_sea, len(cfg.biomes), biome)
    return biome


def _strata(rng: CounterRNG, cfg: WorldGenConfig, ids: dict[str, int], h: np.ndarray,
            water_top: np.ndarray, biome: np.ndarray) -> np.ndarray:
    """Fill the columns: bedrock, deepslate, stone, the biome's soil and top block (sand on beaches
    and under water) and water. ``biome`` holds each column's index into ``biome_palette(cfg)``."""
    sx, sz = h.shape
    y = np.arange(MIN_Y, MAX_Y + 1)[None, :, None]
    cols = _key(np.arange(sx)[:, None], np.arange(sz)[None, :])
    blocks = np.zeros((sx, HEIGHT, sz), np.uint8)
    hh = h[:, None, :]

    # Rock: deepslate below the gradient, stone above; bedrock at the bottom. Random draws only
    # in the two gradient bands (one per cell, keyed by column and y).
    rock = np.where(y <= DEEPSLATE_ALWAYS_Y, ids["deepslate"], ids["stone"]).astype(np.uint8)
    rock = np.broadcast_to(rock, blocks.shape).copy()
    for step, (always, never), cls in ((0, (DEEPSLATE_ALWAYS_Y, DEEPSLATE_NEVER_Y), "deepslate"),
                                       (1, (BEDROCK_ALWAYS_Y, BEDROCK_NEVER_Y), "bedrock")):
        band = np.arange(always, never + 1)
        u = rng.uniform_array(STEP_SOIL, step, Purpose.WORLDGEN, cols[:, None, :], (band - MIN_Y)[None, :, None])
        p = (never - band) / (never - always)  # 1 at "always", 0 at "never"
        rows = rock[:, band - MIN_Y, :]
        rock[:, band - MIN_Y, :] = np.where(u < p[None, :, None], ids[cls], rows)
    rock[:, :BEDROCK_ALWAYS_Y - MIN_Y + 1, :] = ids["bedrock"]
    blocks[:] = np.where(y <= hh, rock, 0)

    # Soil: depth per column; the biome's soil and top; sand on beaches and under water.
    surfaces = [BIOME_SURFACES[b] for b in biome_palette(cfg)]
    depth = cfg.dirt_depth[0] + (rng.uniform_array(STEP_SOIL, 2, Purpose.WORLDGEN, cols, 0)
                                 * (cfg.dirt_depth[1] - cfg.dirt_depth[0] + 1)).astype(np.int32)
    under_cls = np.array([ids[s.under] if s.under else 0 for s in surfaces], np.uint8)[biome]
    under_depth = np.array([s.under_depth if s.under else 0 for s in surfaces], np.int32)[biome]
    is_under = ((y >= hh - (depth + under_depth)[:, None, :]) & (y < hh - depth[:, None, :])
                & (y > BEDROCK_NEVER_Y) & (under_depth[:, None, :] > 0))
    blocks[:] = np.where(is_under, under_cls[:, None, :], blocks)
    wet = water_top >= h
    near = _dilate(wet, cfg.beach_distance)
    near_level = _grow_max(np.where(wet, water_top, MIN_Y - 1), cfg.beach_distance)
    beach = ~wet & near & (h <= near_level + cfg.beach_height)
    deep_water = wet & (water_top - h >= 3)  # [est] gravel floors in deeper water
    soil_top = np.where(wet | beach, ids["sand"], np.array([ids[s.soil] for s in surfaces], np.int64)[biome])
    soil_top = np.where(deep_water, ids["gravel"], soil_top)
    top = np.where(wet | beach, soil_top, np.array([ids[s.top] for s in surfaces], np.int64)[biome])
    is_soil = (y >= hh - depth[:, None, :]) & (y < hh) & (y > BEDROCK_NEVER_Y)
    blocks[:] = np.where(is_soil, soil_top[:, None, :], blocks)
    blocks[:] = np.where((y == hh) & (y > BEDROCK_NEVER_Y), top[:, None, :], blocks)
    blocks[:] = np.where((y > hh) & (y <= water_top[:, None, :]), ids["water"], blocks)
    return blocks


# ---------------------------------------------------------------------------
# Caves [est]
# ---------------------------------------------------------------------------


def _caves(rng: CounterRNG, cfg: WorldGenConfig, concepts: Concepts, blocks: np.ndarray) -> None:
    """Carve worms [est]: air, or lava at and below ``LAVA_LEVEL`` [3]; never next to water."""
    sx, _, sz = blocks.shape
    carve = np.zeros(blocks.shape, bool)
    for cx in range(-(-sx // CHUNK_SIZE)):
        for cz in range(-(-sz // CHUNK_SIZE)):
            s = rng.stream(STEP_CAVES, 0, Purpose.WORLDGEN, _key(cx, cz))
            for _ in range(s.poisson(cfg.caves_per_chunk)):
                _worm(s, cfg, carve, cx * CHUNK_SIZE + s.uniform() * CHUNK_SIZE, s.randint(*cfg.cave_y) + 0.5,
                      cz * CHUNK_SIZE + s.uniform() * CHUNK_SIZE)
    rules = load_rules()
    carvable = np.zeros(256, bool)
    for name in rules.tag("block", "overworld_carver_replaceables"):  # [3] the cave carver's targets
        carvable[concepts.block_class_of_state(rules.block(name).default_state).id] = True
    water = concepts.block_class("water").id
    carvable[water] = False
    wet = blocks == water
    near_water = wet.copy()
    near_water[1:] |= wet[:-1]
    near_water[:-1] |= wet[1:]
    near_water[:, :-1] |= wet[:, 1:]  # water above
    near_water[:, :, 1:] |= wet[:, :, :-1]
    near_water[:, :, :-1] |= wet[:, :, 1:]
    carve &= carvable[blocks] & ~near_water
    lava_rows = np.arange(MIN_Y, MAX_Y + 1)[None, :, None] <= LAVA_LEVEL
    blocks[carve & lava_rows] = concepts.block_class("lava").id
    blocks[carve & ~lava_rows] = 0


def _worm(s: DrawStream, cfg: WorldGenConfig, carve: np.ndarray, x: float, y: float, z: float) -> None:
    """One cave worm [est]: a wandering tube of spheres (vertical radius 0.8x [est])."""
    length = s.randint(*cfg.cave_length)
    yaw = s.uniform() * 2.0 * math.pi
    pitch = (s.uniform() - 0.5) * 0.5
    r0 = cfg.cave_radius[0] + s.uniform() * (cfg.cave_radius[1] - cfg.cave_radius[0])
    sx, _, sz = carve.shape
    for k in range(length):
        r = r0 * (0.7 + 0.5 * math.sin(math.pi * k / length))
        ry = 0.8 * r
        x0, x1 = max(math.floor(x - r), 0), min(math.floor(x + r) + 1, sx)
        y0, y1 = max(math.floor(y - ry), MIN_Y), min(math.floor(y + ry) + 1, MAX_Y + 1)
        z0, z1 = max(math.floor(z - r), 0), min(math.floor(z + r) + 1, sz)
        if x0 < x1 and y0 < y1 and z0 < z1:
            dx = ((np.arange(x0, x1) + 0.5 - x) / r)[:, None, None]
            dy = ((np.arange(y0, y1) + 0.5 - y) / ry)[None, :, None]
            dz = ((np.arange(z0, z1) + 0.5 - z) / r)[None, None, :]
            carve[x0:x1, y0 - MIN_Y:y1 - MIN_Y, z0:z1] |= dx * dx + dy * dy + dz * dz < 1.0
        x += math.cos(pitch) * math.cos(yaw)
        y += math.sin(pitch)
        z += math.cos(pitch) * math.sin(yaw)
        yaw += (s.uniform() - 0.5) * 0.5
        pitch = pitch * 0.7 + (s.uniform() - 0.5) * 0.4
        if y < LAVA_LEVEL - 4 or y > MAX_Y:
            break


# ---------------------------------------------------------------------------
# Ores
# ---------------------------------------------------------------------------


class _Draws:
    """Consecutive raw draws of one key prefix, fetched in one vectorized call.

    ``uniform``, ``below`` and ``randint`` use exactly the formulas of ``CounterRNG`` (rng.py), so
    draw i equals ``rng.uniform(DrawKey(episode, tick, purpose, entity, start + i))`` and so on.
    """

    def __init__(self, rng: CounterRNG, key: DrawKey, n: int):
        self.key = key
        self.raw = rng.u64_array(key.episode, key.tick, key.purpose, key.entity,
                                 key.draw + np.arange(n, dtype=np.int64)).tolist()
        self.i = 0

    def u64(self) -> int:
        v = self.raw[self.i]
        self.i += 1
        return v

    def uniform(self) -> float:
        return (self.u64() >> 11) * (1.0 / (1 << 53))

    def below(self, n: int) -> int:
        return (self.u64() * n) >> 64

    def randint(self, lo: int, hi: int) -> int:
        return lo + self.below(hi - lo + 1)

    def bernoulli(self, p: float) -> bool:
        return self.uniform() < p


def sample_height(row: OreFeature, d: _Draws | DrawStream) -> int:
    """One origin height of an ore attempt, as ``ore_table.height_pmf`` describes [mem]."""
    lo, hi = row.min_y, row.max_y
    if lo > hi:
        return lo
    k = hi - lo
    if row.height == "uniform" or row.plateau >= k:
        return d.randint(lo, hi)
    small = (k - row.plateau) // 2
    return lo + d.randint(0, k - small) + d.randint(0, small)


@dataclass(frozen=True)
class OreTargets:
    """Class-id form of an ore row's targets: a bool[256] replace mask and a class per target."""

    replace: tuple[np.ndarray, ...]
    place: tuple[int, ...]

    @classmethod
    def of(cls, concepts: Concepts, row: OreFeature) -> OreTargets:
        return cls(tuple(_class_set(concepts, t.replaces) for t in row.targets),
                   tuple(concepts.block_class(t.place).id for t in row.targets))


def vein_spheres(origin: tuple[int, int, int], size: int, rng: CounterRNG, key: DrawKey) -> np.ndarray:
    """The spheres (n, 4: x, y, z, radius) of one ore vein, as the game's ``OreFeature`` [mem].

    angle = U[0,1) * pi; the vein runs from origin + (sin, cos)(angle) * size/8 to origin minus that,
    with each end's y = origin y + randint(0, 2) - 2. Sphere k of ``size`` sits at t = k / size along
    it with radius ((sin(pi t) + 1) * (U[0,1) * size / 16) + 1) / 2; a sphere inside a bigger one
    is dropped (pairs in order, as the game does). Draws: key.draw + 0 angle, + 1 and + 2 the y
    offsets, + 3 + k the radius of sphere k.
    """
    d = _Draws(rng, key, size + 3)
    ox, oy, oz = origin
    angle = d.uniform() * math.pi
    g = size / 8.0
    x0, x1 = ox + math.sin(angle) * g, ox - math.sin(angle) * g
    z0, z1 = oz + math.cos(angle) * g, oz - math.cos(angle) * g
    y0, y1 = oy + d.below(3) - 2, oy + d.below(3) - 2
    t = np.arange(size) / size
    rr = np.array([d.uniform() for _ in range(size)]) * size / 16.0
    out = np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0 + (z1 - z0) * t,
                    ((np.sin(np.pi * t) + 1.0) * rr + 1.0) / 2.0], axis=1)
    c = out[:, :3]
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(axis=2).tolist()
    r = out[:, 3].tolist()
    for a in range(size - 1):
        if r[a] <= 0:
            continue
        row = d2[a]
        for b in range(a + 1, size):
            rb = r[b]
            if rb <= 0:
                continue
            dr = r[a] - rb
            if dr * dr > row[b]:
                if dr > 0:
                    r[b] = -1.0
                else:
                    r[a] = -1.0
                    break  # a sphere already dropped cannot drop another
    keep = np.asarray(r) > 0
    return out[keep]


def place_vein(blocks: np.ndarray, origin: tuple[int, int, int], size: int, discard: float, targets: OreTargets,
               rng: CounterRNG, key: DrawKey) -> int:
    """Place one vein into ``blocks`` (indexed [x, y - MIN_Y, z]); returns the blocks placed.

    A cell is placed when its centre lies inside some sphere, it is inside the world, its class
    matches a target (the first matching target wins), and it passes the air check [mem]: with
    discard chance c, c <= 0 always places; c >= 1 places only cells with no air neighbour; else
    a draw u >= c places, and u < c places only cells with no air neighbour. The per-cell draws
    are keyed by the cell's index in the vein's bounding box, so they do not depend on order.
    """
    spheres = vein_spheres(origin, size, rng, key)
    lo = np.floor(spheres[:, :3] - spheres[:, 3:]).min(axis=0).astype(int)
    hi = np.floor(spheres[:, :3] + spheres[:, 3:]).max(axis=0).astype(int)
    sx, _, sz = blocks.shape
    xs = np.arange(max(lo[0], 0), min(hi[0], sx - 1) + 1)
    ys = np.arange(max(lo[1], MIN_Y), min(hi[1], MAX_Y) + 1)
    zs = np.arange(max(lo[2], 0), min(hi[2], sz - 1) + 1)
    if not (xs.size and ys.size and zs.size):
        return 0
    box = blocks[xs[0]:xs[-1] + 1, ys[0] - MIN_Y:ys[-1] - MIN_Y + 1, zs[0]:zs[-1] + 1]
    candidates = np.zeros(box.shape, bool)
    for replace_mask in targets.replace:
        candidates |= replace_mask[box]
    if not candidates.any():
        return 0
    r = spheres[:, 3]
    dx = ((xs[:, None] + 0.5 - spheres[None, :, 0]) / r) ** 2
    dy = ((ys[:, None] + 0.5 - spheres[None, :, 1]) / r) ** 2
    dz = ((zs[:, None] + 0.5 - spheres[None, :, 2]) / r) ** 2
    inside = (dx[:, None, None, :] + dy[None, :, None, :] + dz[None, None, :, :] < 1.0).any(axis=3)
    ok = inside & candidates
    if discard > 0.0 and ok.any():
        air = _air_neighbour(blocks, xs[0], ys[0], zs[0], box.shape)
        if discard >= 1.0:
            ok &= ~air
        else:
            idx = np.arange(box.size, dtype=np.int64).reshape(box.shape)
            u = rng.uniform_array(key.episode, key.tick, key.purpose, key.entity, key.draw + _DISCARD_DRAW0 + idx)
            ok &= (u >= discard) | ~air
    new = box.copy()
    done = np.zeros(box.shape, bool)
    for replace_mask, place in zip(targets.replace, targets.place):
        hit = ok & replace_mask[box] & ~done
        new[hit] = place
        done |= hit
    box[...] = new
    return int(done.sum())


def _air_neighbour(blocks: np.ndarray, x0: int, y0: int, z0: int, shape: tuple[int, ...]) -> np.ndarray:
    """For each cell of a box, whether one of its six neighbours is air (outside: not air)."""
    sx, sy, sz = blocks.shape
    i0 = y0 - MIN_Y
    pad = np.ones((shape[0] + 2, shape[1] + 2, shape[2] + 2), np.uint8)  # 1: not air
    ax0, ax1 = max(x0 - 1, 0), min(x0 + shape[0] + 1, sx)
    ay0, ay1 = max(i0 - 1, 0), min(i0 + shape[1] + 1, sy)
    az0, az1 = max(z0 - 1, 0), min(z0 + shape[2] + 1, sz)
    pad[ax0 - x0 + 1:ax1 - x0 + 1, ay0 - i0 + 1:ay1 - i0 + 1, az0 - z0 + 1:az1 - z0 + 1] = \
        blocks[ax0:ax1, ay0:ay1, az0:az1] != 0
    a = pad == 0
    return (a[:-2, 1:-1, 1:-1] | a[2:, 1:-1, 1:-1] | a[1:-1, :-2, 1:-1] | a[1:-1, 2:, 1:-1]
            | a[1:-1, 1:-1, :-2] | a[1:-1, 1:-1, 2:])


def ore_attempts(rng: CounterRNG, row: OreFeature, row_index: int, chunk: tuple[int, int],
                 density: float = 1.0) -> list[tuple[int, int, int]]:
    """The vein origins of one ore row in one chunk: rarity, count (ores scaled by ``density``),
    then per attempt x and z uniform in the chunk (``in_square``) and y from the height provider.

    Draws (key: STEP_ORES, row index, WORLDGEN, chunk): 0 rarity, 1 count, 2 density remainder;
    attempt a uses draws 1024 + 4a ... 1024 + 4a + 3.
    """
    cx, cz = chunk
    key = DrawKey(STEP_ORES, row_index, Purpose.WORLDGEN, int(_key(cx, cz)))
    head = _Draws(rng, key, 3)
    if row.rarity > 1 and not head.bernoulli(1.0 / row.rarity):
        return []
    head.i = 1
    count = head.randint(row.count_min, row.count_max)
    if row.is_ore and density != 1.0:
        scaled = count * density
        head.i = 2
        count = int(scaled) + int(head.bernoulli(scaled - int(scaled)))
    if count == 0:
        return []
    d = _Draws(rng, key.at(1024), 4 * count)
    out = []
    for a in range(count):
        d.i = 4 * a
        x = cx * CHUNK_SIZE + d.below(CHUNK_SIZE)
        z = cz * CHUNK_SIZE + d.below(CHUNK_SIZE)
        out.append((x, sample_height(row, d), z))
    return out


def vein_key(row_index: int, attempt: int, chunk: tuple[int, int]) -> DrawKey:
    """The draw key of one vein (its spheres and its per-cell air-exposure draws)."""
    return DrawKey(STEP_VEINS, row_index * 65536 + attempt, Purpose.WORLDGEN, int(_key(*chunk)))


def _ores(rng: CounterRNG, cfg: WorldGenConfig, concepts: Concepts, blocks: np.ndarray) -> dict[str, int]:
    """Place the ore table's rows chunk by chunk; returns blocks placed per row."""
    rows = [r for r in ore_table(cfg.biome) if (cfg.ores if r.is_ore else cfg.stone_blobs)]
    targets = {r.name: OreTargets.of(concepts, r) for r in rows}
    row_index = {r.name: i for i, r in enumerate(ore_table())}  # stable keys whatever the biome
    sx, _, sz = blocks.shape
    filled = (blocks != 0).any(axis=(0, 2))
    solid_top = MIN_Y + int(np.flatnonzero(filled)[-1]) if filled.any() else MIN_Y
    placed = {r.name: 0 for r in rows}
    for cx in range(-(-sx // CHUNK_SIZE)):
        for cz in range(-(-sz // CHUNK_SIZE)):
            for row in rows:
                fi = row_index[row.name]
                reach = row.size / 16.0 + 4.0  # how far a vein reaches below its origin (offset, radius, rounding)
                for attempt, origin in enumerate(ore_attempts(rng, row, fi, (cx, cz), cfg.ore_density)):
                    if origin[1] - reach > solid_top or origin[1] + reach < MIN_Y:
                        continue  # nothing to replace there (the draws are keyed, so skipping is safe)
                    placed[row.name] += place_vein(blocks, origin, row.size, row.discard_on_air, targets[row.name],
                                                   rng, vein_key(fi, attempt, (cx, cz)))
    return placed


def vein_size_stats(size: int, n: int = 400, seed: int = 0) -> tuple[float, float]:
    """Mean and mean square of the blocks one vein of ``size`` places in solid stone (for tests
    and the expected-count model)."""
    concepts = load_concepts()
    stone = concepts.block_class("stone").id
    rng = CounterRNG(seed)
    span = size // 4 + 8
    counts = []
    targets = OreTargets((_class_set(concepts, ("stone",)),), (concepts.block_class("iron_ore").id,))
    blocks = np.full((2 * span + 1, HEIGHT, 2 * span + 1), stone, np.uint8)
    for i in range(n):
        blocks[...] = stone
        counts.append(place_vein(blocks, (span, 0, span), size, 0.0, targets, rng,
                                 DrawKey(STEP_VEINS, size, Purpose.GENERIC, i)))
    c = np.asarray(counts, float)
    return float(c.mean()), float((c * c).mean())


# ---------------------------------------------------------------------------
# Trees and plants [est]
# ---------------------------------------------------------------------------


def _trees(rng: CounterRNG, cfg: WorldGenConfig, ids: dict[str, int], concepts: Concepts, blocks: np.ndarray,
           h: np.ndarray, biome: np.ndarray) -> int:
    """Oak and birch trees on grass blocks [est], as dense as each column's biome says; returns
    the number placed. Per chunk, Poisson(the palette's densest biome x vegetation) attempts,
    each kept with probability (its column biome's density) / (the densest) and birch with the
    column biome's birch fraction (draws: x, z, height, keep, birch)."""
    sx, _, sz = blocks.shape
    surfaces = [BIOME_SURFACES[b] for b in biome_palette(cfg)]
    density = np.array([s.trees_per_chunk for s in surfaces])
    densest = float(density.max())
    if densest <= 0.0 or cfg.vegetation <= 0.0:
        return 0
    keep_p = density / densest
    birch_p = np.array([s.birch_fraction for s in surfaces])
    soft = concepts.replaceable | (np.arange(256) == 0)
    trunks: list[tuple[int, int]] = []
    for cx in range(-(-sx // CHUNK_SIZE)):
        for cz in range(-(-sz // CHUNK_SIZE)):
            s = rng.stream(STEP_TREES, 0, Purpose.WORLDGEN, _key(cx, cz))
            for _ in range(s.poisson(densest * cfg.vegetation)):
                x, z = cx * CHUNK_SIZE + s.below(CHUNK_SIZE), cz * CHUNK_SIZE + s.below(CHUNK_SIZE)
                height = s.randint(*cfg.trunk_height)
                u_keep, u_birch = s.uniform(), s.uniform()
                t = rng.stream(STEP_TREES, 1, Purpose.WORLDGEN, _key(x, z))  # canopy corner draws
                if x >= sx or z >= sz:
                    continue
                b = int(biome[x, z])
                if not u_keep < keep_p[b]:
                    continue
                birch = u_birch < birch_p[b]
                g = int(h[x, z])
                if blocks[x, g - MIN_Y, z] != ids["grass_block"] or g + height + 2 > MAX_Y:
                    continue
                if any(max(abs(x - a), abs(z - b)) < 3 for a, b in trunks):  # [est] spacing
                    continue
                trunks.append((x, z))
                leaves = ids["leaves"] if birch else ids["oak_leaves"]  # birch: "leaves"; oak: "oak leaves"
                top = g + height
                # Canopy [est] (roughly the vanilla blob foliage of radius 2 [mem]): two layers of
                # radius 2 below the trunk top, radius 1 at the top, a plus shape above it;
                # corners are dropped at random.
                for dy, radius, plus in ((-2, 2, False), (-1, 2, False), (0, 1, False), (1, 1, True)):
                    yy = top + dy
                    for dx in range(-radius, radius + 1):
                        for dz in range(-radius, radius + 1):
                            corner = abs(dx) == radius and abs(dz) == radius
                            if corner and (plus or not t.bernoulli(0.5)):
                                continue
                            xx, zz = x + dx, z + dz
                            if 0 <= xx < sx and 0 <= zz < sz and soft[blocks[xx, yy - MIN_Y, zz]]:
                                blocks[xx, yy - MIN_Y, zz] = leaves
                for yy in range(g + 1, top + 1):
                    blocks[x, yy - MIN_Y, z] = ids["log"]
                blocks[x, g - MIN_Y, z] = ids["dirt"]  # [mem] trees turn the block under the trunk to dirt
    return len(trunks)


def _plants(rng: CounterRNG, cfg: WorldGenConfig, ids: dict[str, int], blocks: np.ndarray, h: np.ndarray,
            biome: np.ndarray) -> None:
    """Short grass and flowers [est] on grass blocks with air above, dead bushes on sand, at each
    column biome's rates (scaled by ``vegetation``)."""
    sx, _, sz = blocks.shape
    surfaces = [BIOME_SURFACES[b] for b in biome_palette(cfg)]
    v = cfg.vegetation
    grass_p = np.array([s.grass_chance * v for s in surfaces])[biome]
    flower_p = np.array([s.flower_chance * v for s in surfaces])[biome]
    bush_p = np.array([s.dead_bush_chance * v for s in surfaces])[biome]
    xs, zs = np.meshgrid(np.arange(sx), np.arange(sz), indexing="ij")
    u = rng.uniform_array(STEP_PLANTS, 0, Purpose.WORLDGEN, _key(xs, zs), 0)
    ok = (h + 1 <= MAX_Y)
    top = blocks[xs, h - MIN_Y, zs]
    above = blocks[xs, np.minimum(h + 1, MAX_Y) - MIN_Y, zs]
    ok &= above == 0
    on_grass = ok & (top == ids["grass_block"])
    grass = on_grass & (u < grass_p)
    flower = on_grass & (u >= grass_p) & (u < grass_p + flower_p)
    bush = ok & (top == ids["sand"]) & (u < bush_p)
    blocks[xs[grass], h[grass] + 1 - MIN_Y, zs[grass]] = ids["short_grass"]
    blocks[xs[flower], h[flower] + 1 - MIN_Y, zs[flower]] = ids["flower"]
    blocks[xs[bush], h[bush] + 1 - MIN_Y, zs[bush]] = ids["dead_bush"]


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def generate(seed: int, config: WorldGenConfig | str = "default", *, concepts: Concepts | None = None) -> World:
    """A deterministic test world: the same seed and config give identical voxels."""
    cfg = PRESETS[config] if isinstance(config, str) else config
    concepts = concepts if concepts is not None else load_concepts()
    check_biomes(cfg, concepts)
    ids = _ids(concepts)
    rng = CounterRNG(seed)
    h = _heights(rng, cfg)
    water_top = np.where(h < WATER_TOP, WATER_TOP, MIN_Y - 1).astype(np.int32)
    if cfg.pools_per_chunk > 0:
        _pools(rng, cfg, h, water_top)
    biome = column_biomes(rng, cfg, water_top)
    blocks = _strata(rng, cfg, ids, h, water_top, biome)
    if cfg.caves_per_chunk > 0:
        _caves(rng, cfg, concepts, blocks)
    ore_blocks = _ores(rng, cfg, concepts, blocks) if (cfg.ores or cfg.stone_blobs) else {}
    trees = _trees(rng, cfg, ids, concepts, blocks, h, biome)
    if cfg.vegetation > 0:
        _plants(rng, cfg, ids, blocks, h, biome)
    world = World(cfg.size_x, cfg.size_z, concepts=concepts)
    world.load(blocks)
    world.set_biome_map(np.array([concepts.biome(b).id for b in biome_palette(cfg)], np.uint8)[biome])
    world.meta.update(seed=seed, preset=config if isinstance(config, str) else None, config=asdict(cfg),
                      ore_blocks=ore_blocks, trees=trees, surface=h)
    world.meta["spawn"] = spawn_point(world)
    return world


def standable_surface(world: World, x: int, z: int) -> int | None:
    """The feet y of a safe surface spot in column (x, z), or None.

    Safe [est]: the column's highest light-blocking block (``world.heightmap``: water, leaves,
    plants and snow layers do not count) is solid, not a hazard, not a log (no spawning on a
    tree), and the two cells above it are free of collision, liquid and hazards.
    """
    c = world.concepts
    y = int(world.heightmap[x, z])
    if y < MIN_Y or y + 2 > MAX_Y:
        return None
    ground = world.get(x, y, z)
    if not c.solid[ground] or c.hazard[ground] or ground == c.block_class("log").id:
        return None
    for cell in (world.get(x, y + 1, z), world.get(x, y + 2, z)):
        if c.collision_height[cell] > 0 or c.liquid[cell] or c.hazard[cell]:
            return None
    return y + 1


def spawn_point(world: World, near: tuple[int, int] | None = None) -> tuple[float, float, float]:
    """A feet position (x + 0.5, y, z + 0.5) on the surface: the first safe column on a square
    spiral out from ``near`` (default the world's centre). Deterministic."""
    cx, cz = near if near is not None else (world.size_x // 2, world.size_z // 2)
    for radius in range(max(world.size_x, world.size_z)):
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if max(abs(dx), abs(dz)) != radius:
                    continue
                x, z = cx + dx, cz + dz
                if 0 <= x < world.size_x and 0 <= z < world.size_z:
                    y = standable_surface(world, x, z)
                    if y is not None:
                        return (x + 0.5, float(y), z + 0.5)
    raise ValueError("no safe surface column in this world")

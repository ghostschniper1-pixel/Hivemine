"""Tests for hivemind.sim.ref.world: voxels, change log, light, slots, clock, chunk ticking."""

from __future__ import annotations

import numpy as np
import pytest

from hivemind.rules import load_concepts
from hivemind.sim.ref.world import (
    CHESTS_PER_AGENT,
    DEFAULT_BIOME,
    ENTITY_CAPACITY,
    MAX_Y,
    MIN_Y,
    SIMULATION_DISTANCE,
    UNSEEN,
    BlockChange,
    SlotsFull,
    World,
    flood_light_reference,
    light_filter_table,
    ticking_chunks,
)

C = load_concepts()
STONE = C.block_class("stone").id
GLASS = C.block_class("glass").id
TORCH = C.block_class("torch").id
LAVA = C.block_class("lava").id
MAGMA = C.block_class("magma_block").id
LEAVES = C.block_class("oak_leaves").id
BARRIER = C.block_class("barrier").id
WATER = C.block_class("water").id
SNOW = C.block_class("snow").id  # one snow layer
GRASS = C.block_class("grass_block").id
TINTED = C.block_class("other_tinted_glass").id
SLAB = C.block_class("other_slab").id


def small_world() -> World:
    w = World(24, 24)
    blocks = w.blocks.copy()
    blocks[:, : 60 - MIN_Y + 1, :] = STONE  # stone up to y 60
    w.load(blocks)
    return w


# -- voxels and the edge ---------------------------------------------------------------


def test_shape_and_y_range():
    w = World(16, 20)
    assert w.blocks.shape == (16, 256, 20)
    assert (MIN_Y, MAX_Y) == (-64, 191)
    assert w.blocks.dtype == np.uint8


def test_get_set_and_world_edge():
    w = small_world()
    assert w.get(3, 60, 3) == STONE and w.get(3, 61, 3) == 0
    assert w.set(3, 61, 3, GLASS) == 0
    assert w.get(3, 61, 3) == GLASS
    for x, y, z in ((-1, 10, 0), (24, 10, 0), (0, 10, -1), (0, 10, 24), (0, MIN_Y - 1, 0), (-1, MAX_Y + 1, 0)):
        assert w.get(x, y, z) == BARRIER
        with pytest.raises(IndexError):
            w.set(x, y, z, STONE)
    assert w.get(0, MAX_Y + 1, 0) == 0 and w.get(23, 319, 23) == 0  # above the voxels: air, as in the game
    with pytest.raises(IndexError):
        w.set(0, MAX_Y + 1, 0, STONE)  # but nothing can be placed there
    assert C.hardness[BARRIER] < 0  # unbreakable
    with pytest.raises(ValueError):
        w.set(1, 1, 1, UNSEEN)  # "unseen" is a perception class, never a voxel
    assert UNSEEN == 255 and C.block_class(UNSEEN).name == "unseen"
    bad = w.blocks.copy()
    bad[0, 0, 0] = UNSEEN
    with pytest.raises(ValueError):
        w.load(bad)


def test_region_pads_with_barrier():
    w = small_world()
    r = w.region(-2, 59, -2, 3, 62, 3)
    assert r.shape == (5, 3, 5)
    assert (r[:2] == BARRIER).all() and (r[:, :, :2] == BARRIER).all()
    assert (r[2:, 0, 2:] == STONE).all() and (r[2:, 2, 2:] == 0).all()
    top = w.region(-1, MAX_Y - 1, 0, 2, MAX_Y + 3, 2)  # rows 190..193
    assert (top[0] == BARRIER).all() and (top[1:, 2:] == 0).all()


def test_change_log_cursor_and_trim():
    w = small_world()
    cursor = w.change_count
    w.set(1, 61, 1, GLASS)
    w.advance(5)
    w.set(1, 61, 1, 0)
    w.set(1, 61, 1, 0)  # no change: not logged
    changes = w.changes_since(cursor)
    assert changes == [BlockChange(1, 61, 1, 0, GLASS, 0), BlockChange(1, 61, 1, GLASS, 0, 5)]
    assert w.change_count == cursor + 2
    w.trim_changes(cursor + 1)
    assert w.changes_since(cursor + 1) == changes[1:]
    with pytest.raises(ValueError):
        w.changes_since(cursor)
    assert w.changes_since(w.change_count) == []
    with pytest.raises(ValueError):  # a cursor from the future (another world) is not "no changes"
        w.changes_since(w.change_count + 1)
    gen = w.generation
    w.load(w.blocks)
    assert w.generation == gen + 1 and w.change_count == 0
    with pytest.raises(ValueError):  # the cursor of the previous generation
        w.changes_since(cursor + 2)


def test_every_world_object_has_its_own_serial():
    import copy
    import pickle

    w = small_world()
    others = [small_world(), w.copy(), copy.copy(w), copy.deepcopy(w), pickle.loads(pickle.dumps(w))]
    serials = [w.serial, *(o.serial for o in others)]
    assert len(set(serials)) == len(serials)
    assert others[1].generation == w.generation  # a copy keeps the generation, so caches need the serial
    gen = w.generation
    w.load(w.blocks)
    assert w.serial == serials[0] and w.generation == gen + 1  # a load keeps the object, bumps the generation


# -- heightmap and sky light ---------------------------------------------------------------


def test_light_filters_come_from_the_game_data():
    f = light_filter_table(C)
    for name, value in (("air", 0), ("glass", 0), ("torch", 0), ("short_grass", 0), ("snow", 0),
                        ("other_slab", 0), ("fence", 0), ("other_fence_gate", 0), ("chest", 0), ("farmland", 0),
                        ("water", 1), ("lava", 1), ("oak_leaves", 1), ("ice", 1), ("cobweb", 1),
                        ("stone", 15), ("dirt", 15), ("log", 15), ("other_tinted_glass", 15)):
        assert f[C.block_class(name).id] == value, name
    assert not C.transparent[SNOW] and not C.transparent[SLAB] and C.transparent[TINTED]  # not the sight flag


def test_heightmap_and_sky_light():
    w = small_world()
    assert (w.heightmap == 60).all()
    assert w.sky_light(5, 61, 5) == 15 and w.sky_light(5, 60, 5) == 0
    w.set(5, 70, 5, STONE)  # a roof
    assert w.heightmap[5, 5] == 70
    assert w.sky_light(5, 65, 5) == 0 and w.sky_light(5, 71, 5) == 15
    w.set(5, 70, 5, GLASS)  # glass filters nothing: sky light passes
    assert w.heightmap[5, 5] == 60 and w.sky_light(5, 65, 5) == 15
    w.set(5, 75, 5, LEAVES)  # leaves dim sky light by 1 per block
    assert w.heightmap[5, 5] == 60
    assert w.sky_light(5, 76, 5) == 15 and w.sky_light(5, 75, 5) == 14 and w.sky_light(5, 65, 5) == 14
    w.set(5, 74, 5, LEAVES)
    assert w.sky_light(5, 65, 5) == 13
    w.set(5, 60, 5, 0)  # dig the top: the next light-blocking cell below
    assert w.heightmap[5, 5] == 59
    region = w.sky_light_region(4, 58, 4, 7, 62, 7)
    assert region[1, :, 1].tolist() == [0, 0, 13, 13]  # y 58, 59 dark; 60, 61 under two leaves
    assert region[0, :, 0].tolist() == [0, 0, 0, 15]
    w.set(6, 80, 6, TINTED)  # tinted glass blocks light although it can be seen through
    assert w.heightmap[6, 6] == 80 and w.sky_light(6, 79, 6) == 0


def test_sky_light_region_equals_pointwise():
    w = small_world()
    for x, y, z, cls in ((3, 70, 3, LEAVES), (3, 66, 3, WATER), (4, 64, 4, STONE), (5, 61, 5, SNOW),
                         (5, 62, 5, LEAVES), (6, 63, 6, TINTED)):
        w.set(x, y, z, cls)
    box = (-2, 55, -1, 9, MAX_Y + 3, 8)
    region = w.sky_light_region(*box)
    expected = np.array([[[w.sky_light(x, y, z) for z in range(box[2], box[5])] for y in range(box[1], box[4])]
                         for x in range(box[0], box[3])])
    assert np.array_equal(region, expected)
    assert region[3, MAX_Y + 2 - box[1], 3] == 15 and region[0, 10, 3] == 0  # above: sky; beside: 0


def test_snow_layer_lets_light_through():
    """A snow layer is not transparent (sight) but filters no light: sky and torch light reach the
    agent standing in it, and it does not raise the heightmap."""
    w = World(16, 16)
    b = w.blocks.copy()
    b[:, : 64 - MIN_Y + 1, :] = GRASS
    b[:, 65 - MIN_Y, :] = SNOW
    w.load(b)
    w.set(8, 65, 8, TORCH)
    assert (w.heightmap == 64).all()
    assert w.sky_light(9, 65, 8) == 15 and w.block_light_at(9, 65, 8) == 13 and w.light(9, 65, 8) == 15
    assert w.block_light_at(12, 65, 8) == 10
    w.set(8, 66, 8, SLAB)  # a slab above passes light too (shape occlusion is not modelled)
    assert w.sky_light(8, 65, 8) == 15 and w.heightmap[8, 8] == 64


def test_water_dims_sky_light_by_depth_and_passes_block_light():
    w = small_world()
    for x in range(4, 9):
        for z in range(4, 9):
            for y in (58, 59, 60):
                w.set(x, y, z, WATER)  # a 3-deep pond
    assert [w.sky_light(6, y, 6) for y in (61, 60, 59, 58, 57)] == [15, 14, 13, 12, 0]
    w.set(3, 61, 6, TORCH)
    assert w.block_light_at(5, 60, 6) == 14 - 3  # three steps, water costs 1 like air


# -- block light ----------------------------------------------------------------------------


def test_torch_light_falls_off_by_one_per_block():
    w = small_world()
    w.set(10, 61, 10, TORCH)
    assert C.light[TORCH] == 14
    assert w.block_light_at(10, 61, 10) == 14
    assert w.block_light_at(11, 61, 10) == 13
    assert w.block_light_at(13, 62, 10) == 10  # Manhattan distance 4
    assert w.block_light_at(10, 60, 10) == 0  # stone is opaque
    assert w.light(10, 61, 10) == 15  # sky light wins in the open


def test_light_blocked_by_wall_and_passes_glass():
    w = small_world()
    for z in range(24):
        for y in range(61, 70):
            w.set(12, y, z, STONE)
    w.set(10, 61, 10, TORCH)
    assert w.block_light_at(11, 61, 10) == 13
    assert w.block_light_at(13, 61, 10) == 0  # behind the wall (going over it is > 14 steps)
    w.set(12, 61, 10, GLASS)
    assert w.block_light_at(13, 61, 10) == 11
    w.set(12, 61, 10, STONE)
    assert w.block_light_at(13, 61, 10) == 0  # removal relights


def test_opaque_light_sources_emit():
    w = small_world()
    w.set(5, 60, 5, MAGMA)
    assert C.light[MAGMA] == 3
    assert w.block_light_at(5, 60, 5) == 3 and w.block_light_at(5, 61, 5) == 2
    assert w.block_light_at(5, 59, 5) == 0


def test_incremental_light_equals_full_recompute_and_bfs():
    """Random edits: the incremental relight equals a full recomputation and a plain BFS."""
    w = small_world()
    cells = [(x, y, z) for x in range(2, 22, 3) for y in (58, 60, 61, 63) for z in range(2, 22, 4)]
    choices = (0, STONE, GLASS, TORCH, LAVA, LEAVES, MAGMA, WATER, SNOW, TINTED, SLAB)
    for i, (x, y, z) in enumerate(cells):
        w.set(x, y, z, choices[(i * 7 + x + z) % len(choices)])
    for x, y, z in cells[::3]:
        w.set(x, y, z, 0)
    incremental = w.block_light.copy()
    w.recompute_light()
    assert np.array_equal(incremental, w.block_light)
    assert np.array_equal(incremental, flood_light_reference(w))


# -- entity slots ---------------------------------------------------------------------------


def test_entity_slot_capacities():
    w = World(16, 16)
    assert CHESTS_PER_AGENT == 8  # DESIGN.md 4.4: my chest 1-8 per agent
    assert ENTITY_CAPACITY == {"agents": 40, "mobs": 32, "items": 64, "furnaces": 16, "chests": 320}
    for kind, cap in ENTITY_CAPACITY.items():
        table = getattr(w, kind)
        assert table.capacity == cap and len(table) == 0 and table.in_use.shape == (cap,)
    assert World(16, 16, entity_capacity={"agents": 4}).chests.capacity == 32  # scales with agents
    assert World(16, 16, entity_capacity={"agents": 4, "chests": 5}).chests.capacity == 5


def test_slot_alloc_free_and_fields():
    w = World(16, 16)
    zombie = C.entity_class("zombie").id
    a = w.mobs.alloc(cls=zombie, pos=(1.5, 64.0, 2.5), health=20.0)
    b = w.mobs.alloc(cls=zombie)
    assert (a, b) == (0, 1)
    assert w.mobs.pos[a].tolist() == [1.5, 64.0, 2.5] and w.mobs.target[a] == -1
    w.mobs.free(a)
    assert w.mobs.active().tolist() == [1]
    c = w.mobs.alloc(cls=zombie)
    assert c == 0 and w.mobs.health[c] == 0.0  # reset to the fill value
    with pytest.raises(KeyError):
        w.mobs.alloc(colour=3)
    w.mobs.add_field("fuse", np.int16, (), -1)
    assert w.mobs.fuse.tolist() == [-1] * 32
    for _ in range(30):
        w.mobs.alloc()
    with pytest.raises(SlotsFull):
        w.mobs.alloc()
    assert w.chests.slots.shape == (320, 27, 2)


def test_copy_is_deep():
    w = small_world()
    w.agents.alloc(pos=(3.5, 61.0, 3.5))
    v = w.copy()
    v.set(1, 61, 1, STONE)
    v.agents.pos[0] = (0, 0, 0)
    v.biome_map[0, 0] = C.biome("desert").id
    assert w.get(1, 61, 1) == 0 and w.agents.pos[0].tolist() == [3.5, 61.0, 3.5]
    assert v.change_count == 1 and w.change_count == 0
    assert w.biome_map[0, 0] == C.biome(DEFAULT_BIOME).id


def test_light_points_equal_pointwise():
    w = small_world()
    for x, y, z, cls in ((3, 70, 3, LEAVES), (3, 66, 3, WATER), (4, 64, 4, STONE), (8, 61, 8, TORCH),
                         (12, 62, 12, LAVA), (6, 63, 6, TINTED)):
        w.set(x, y, z, cls)
    cells = np.array([(x, y, z) for x in (-1, 0, 3, 4, 8, 9, 12, 23, 24) for y in (MIN_Y - 1, 59, 60, 61, 62, 65,
                                                                                    69, 70, 71, MAX_Y, MAX_Y + 1)
                      for z in (-2, 3, 4, 8, 12, 23)])
    sky, block = w.light_points(cells)
    assert sky.dtype == np.uint8 and block.dtype == np.uint8
    assert sky.tolist() == [w.sky_light(*c) for c in cells.tolist()]
    assert block.tolist() == [w.block_light_at(*c) for c in cells.tolist()]
    assert w.light_points(np.zeros((0, 3), int))[0].shape == (0,)


# -- agents: facing and home ------------------------------------------------------------------


def test_alloc_agent_facing_and_home():
    w = World(16, 16)
    a = w.alloc_agent((3.5, 64.0, 5.5), yaw=-90.0, pitch=30.0)
    assert w.agents.pos[a].tolist() == [3.5, 64.0, 5.5] and w.agents.home[a].tolist() == [3.5, 64.0, 5.5]
    assert w.agents.yaw.dtype == np.float32 and (w.agents.yaw[a], w.agents.pitch[a]) == (-90.0, 30.0)
    b = w.alloc_agent((8.5, 64.0, 8.5), home=(1.0, 2.0, 3.0), team=1, number=7)
    assert w.agents.home[b].tolist() == [1.0, 2.0, 3.0] and (w.agents.team[b], w.agents.number[b]) == (1, 7)
    assert w.agents.alive[b] and w.agents.health[b] == 20.0
    raw = w.agents.alloc(pos=(1.5, 64.0, 1.5))  # a slot whose home was never set: NaN, never 0
    assert np.isnan(w.agents.home[raw]).all()
    for pitch in (-90.5, 91.0):
        with pytest.raises(ValueError):
            w.alloc_agent((1.5, 64.0, 1.5), pitch=pitch)


# -- biomes -----------------------------------------------------------------------------------


def test_biome_map_default_set_and_lookup():
    w = World(16, 8)
    plains, desert = C.biome("plains").id, C.biome("desert").id
    assert DEFAULT_BIOME == "plains" and w.biome_map.shape == (16, 8) and (w.biome_map == plains).all()
    biomes = np.full((16, 8), plains, np.uint8)
    biomes[8:, :] = desert
    w.set_biome_map(biomes)
    biomes[:] = plains  # the world keeps its own copy
    assert w.biome_at(7.99, 3.0) == plains and w.biome_at(8.0, 3.0) == desert
    assert w.biome_at(-5.0, 3.0) == plains and w.biome_at(40.0, -3.0) == desert  # outside: the nearest column
    with pytest.raises(ValueError):
        w.set_biome_map(np.zeros((8, 16), np.uint8))
    with pytest.raises(TypeError):
        w.set_biome_map(np.zeros((16, 8), np.int64))
    with pytest.raises(ValueError):
        w.set_biome_map(np.full((16, 8), len(C.biomes), np.uint8))


# -- clock ---------------------------------------------------------------------------------


def test_clock():
    w = World(16, 16, day_time=23_990)
    w.advance(20)
    assert w.tick == 20 and w.day_time == 24_010 and w.time_of_day == 10
    with pytest.raises(ValueError):
        w.advance(-1)


# -- chunk ticking --------------------------------------------------------------------------


def test_ticking_square_is_chebyshev_radius_10():
    assert SIMULATION_DISTANCE == 10
    mask = ticking_chunks(40, 40, [(20 * 16 + 5.5, 64.0, 20 * 16 + 0.2)])
    assert mask.sum() == 21 * 21 == 441  # DESIGN.md 11.3
    assert mask[10, 10] and mask[30, 30] and mask[10, 30]  # corners of the square tick
    assert not mask[9, 20] and not mask[31, 20] and not mask[20, 31]


def test_ticking_clipped_to_world_and_union_of_agents():
    mask = ticking_chunks(32, 32, [(8.0, 64.0, 8.0)])
    assert mask.sum() == 11 * 11  # chunks 0..10 in x and z
    two = ticking_chunks(64, 4, [(8.0, 8.0), (60 * 16.0, 8.0)])  # (x, z) pairs work too
    assert two[:11].all() and two[50:].all() and not two[11:50].any()
    assert not ticking_chunks(8, 8, []).any()


def test_ticking_uses_floor_for_chunk_of_position():
    assert World.chunk_of(15.99, -0.01) == (0, -1)
    assert World.chunk_of(16.0, 31.9) == (1, 1)
    # An agent just past a chunk border moves the square by one chunk.
    a = ticking_chunks(40, 40, [(20 * 16 - 0.01, 0, 100.0)])
    b = ticking_chunks(40, 40, [(20 * 16 + 0.01, 0, 100.0)])
    assert a[9].any() and not b[9].any() and b[30].any() and not a[30].any()


def test_world_ticking_uses_agent_slots_dead_or_alive():
    w = World(16 * 30, 16)
    assert not w.ticking_chunks().any()
    i = w.agents.alloc(pos=(5.0, 64.0, 5.0))
    assert w.ticking_chunks()[:11].all() and not w.ticking_chunks()[11:].any()
    assert w.is_ticking(10 * 16 + 1, 3) and not w.is_ticking(11 * 16 + 1, 3)
    w.agents.alive[i] = False  # a dead player stays in the level until it respawns [mem]
    assert w.ticking_chunks()[:11].all() and len(w.agent_positions(alive_only=True)) == 0
    w.agents.free(i)
    assert not w.ticking_chunks().any()

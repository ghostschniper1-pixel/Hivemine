"""Tests for hivemind.sim.ref.world: voxels, change log, light, slots, clock, chunk ticking."""

from __future__ import annotations

import numpy as np
import pytest

from hivemind.rules import load_concepts
from hivemind.sim.ref.world import (
    ENTITY_CAPACITY,
    MAX_Y,
    MIN_Y,
    SIMULATION_DISTANCE,
    BlockChange,
    SlotsFull,
    World,
    flood_light_reference,
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
    for x, y, z in ((-1, 10, 0), (24, 10, 0), (0, 10, -1), (0, 10, 24), (0, MIN_Y - 1, 0), (0, MAX_Y + 1, 0)):
        assert w.get(x, y, z) == BARRIER
        with pytest.raises(IndexError):
            w.set(x, y, z, STONE)
    assert C.hardness[BARRIER] < 0  # unbreakable
    with pytest.raises(ValueError):
        w.set(1, 1, 1, 255)  # "buried" is a perception class, never a voxel


def test_region_pads_with_barrier():
    w = small_world()
    r = w.region(-2, 59, -2, 3, 62, 3)
    assert r.shape == (5, 3, 5)
    assert (r[:2] == BARRIER).all() and (r[:, :, :2] == BARRIER).all()
    assert (r[2:, 0, 2:] == STONE).all() and (r[2:, 2, 2:] == 0).all()


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
    gen = w.generation
    w.load(w.blocks)
    assert w.generation == gen + 1 and w.change_count == 0


# -- heightmap and sky light ---------------------------------------------------------------


def test_heightmap_and_sky_light():
    w = small_world()
    assert (w.heightmap == 60).all()
    assert w.sky_light(5, 61, 5) == 15 and w.sky_light(5, 60, 5) == 0
    w.set(5, 70, 5, STONE)  # a roof
    assert w.heightmap[5, 5] == 70
    assert w.sky_light(5, 65, 5) == 0 and w.sky_light(5, 71, 5) == 15
    w.set(5, 70, 5, GLASS)  # transparent roof: sky light passes
    assert w.heightmap[5, 5] == 60 and w.sky_light(5, 65, 5) == 15
    w.set(5, 75, 5, LEAVES)  # leaves are transparent
    assert w.heightmap[5, 5] == 60
    w.set(5, 60, 5, 0)  # dig the top: the next opaque cell below
    assert w.heightmap[5, 5] == 59
    region = w.sky_light_region(4, 58, 4, 7, 62, 7)
    assert region[1, :, 1].tolist() == [0, 0, 15, 15]  # y 58, 59 dark; 60, 61 lit
    assert region[0, :, 0].tolist() == [0, 0, 0, 15]


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
    choices = (0, STONE, GLASS, TORCH, LAVA, LEAVES, MAGMA)
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
    assert ENTITY_CAPACITY == {"agents": 40, "mobs": 32, "items": 64, "furnaces": 16, "chests": 16}
    for kind, cap in ENTITY_CAPACITY.items():
        table = getattr(w, kind)
        assert table.capacity == cap and len(table) == 0 and table.in_use.shape == (cap,)


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
    assert w.chests.slots.shape == (16, 27, 2)


def test_copy_is_deep():
    w = small_world()
    w.agents.alloc(pos=(3.5, 61.0, 3.5))
    v = w.copy()
    v.set(1, 61, 1, STONE)
    v.agents.pos[0] = (0, 0, 0)
    assert w.get(1, 61, 1) == 0 and w.agents.pos[0].tolist() == [3.5, 61.0, 3.5]
    assert v.change_count == 1 and w.change_count == 0


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


def test_world_ticking_uses_live_agents():
    w = World(16 * 30, 16)
    assert not w.ticking_chunks().any()
    i = w.agents.alloc(pos=(5.0, 64.0, 5.0))
    assert w.ticking_chunks()[:11].all() and not w.ticking_chunks()[11:].any()
    assert w.is_ticking(10 * 16 + 1, 3) and not w.is_ticking(11 * 16 + 1, 3)
    w.agents.alive[i] = False
    assert not w.ticking_chunks().any()

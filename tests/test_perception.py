"""Tests for hivemind.sim.ref.perception: exposure, window, line of sight, sightings, honesty."""

from __future__ import annotations

import numpy as np
import pytest

from hivemind.rules import load_concepts
from hivemind.sim.ref import perception as P
from hivemind.sim.ref import worldgen as wg
from hivemind.sim.ref.rng import CounterRNG, Purpose
from hivemind.sim.ref.world import MAX_Y, MIN_Y, World

C = load_concepts()
ID = {c.name: c.id for c in C.block_classes}
STONE, GLASS, IRON, DIAMOND = ID["stone"], ID["glass"], ID["iron_ore"], ID["diamond_ore"]
FLOOR = 60  # stone up to y 60, air above


def room(size: int = 40) -> World:
    w = World(size, size)
    b = w.blocks.copy()
    b[:, : FLOOR - MIN_Y + 1, :] = STONE
    w.load(b)
    return w


def wall(w: World, x: int, cls: int, ys=range(FLOOR + 1, FLOOR + 8)) -> None:
    for z in range(w.size_z):
        for y in ys:
            w.set(x, y, z, cls)


FEET = (10.5, FLOOR + 1.0, 10.5)
EYE = P.eye_of(FEET)


# -- exposure and the window ---------------------------------------------------------------


def test_eye_height():
    assert P.EYE_HEIGHT == 1.62
    assert EYE == (10.5, FLOOR + 2.62, 10.5)


def test_exposure_rules():
    w = room()
    assert P.is_exposed(w, 5, FLOOR, 5)  # top face touches air
    assert not P.is_exposed(w, 5, FLOOR - 1, 5)  # surrounded by stone
    w.set(5, FLOOR, 5, GLASS)
    assert P.is_exposed(w, 5, FLOOR - 1, 5)  # glass is transparent
    assert P.exposed_faces(w, 5, FLOOR - 1, 5) == (3,)  # +y only
    assert not P.is_exposed(w, 0, 10, 5)  # the barrier at x = -1 is the world edge, not a window
    w.set(6, 40, 6, 0)  # a one-cell pocket deep down
    assert P.is_exposed(w, 6, 41, 6) and not P.is_exposed(w, 6, 40, 6)  # air with no open face is buried


def test_window_shape_buried_and_light():
    w = room()
    win = P.voxel_window(w, FEET)
    assert win.classes.shape == (15, 7, 15) == P.WINDOW_SHAPE
    assert win.origin == (3, FLOOR + 1 - 3, 3)
    # y rows: FLOOR-2 (buried), FLOOR-1 (buried), FLOOR (stone, exposed), FLOOR+1.. (air)
    assert (win.classes[:, 0, :] == P.BURIED).all() and (win.classes[:, 1, :] == P.BURIED).all()
    assert (win.classes[:, 2, :] == STONE).all() and (win.classes[:, 3:, :] == 0).all()
    assert (win.light[:, 3:, :] == 15).all() and (win.light[:, :3, :] == 0).all()
    w.set(10, FLOOR - 1, 10, IRON)  # hidden under the floor: still buried
    assert np.array_equal(P.voxel_window(w, FEET).classes, win.classes)
    w.set(10, FLOOR, 10, 0)  # dig the floor: the ore is now exposed
    win2 = P.voxel_window(w, FEET)
    assert win2.classes[7, 1, 7] == IRON and win2.classes[7, 2, 7] == 0


def test_window_at_world_edge_shows_barrier_only_where_seen():
    w = room()
    win = P.voxel_window(w, (1.5, FLOOR + 1.0, 1.5))
    assert win.origin[0] == -6
    row = win.classes[:, 4, 7]  # y = FLOOR + 2, z = 1: x from -6 to 8
    assert row[5] == ID["barrier"]  # x = -1 touches the air at x = 0: the wall is seen
    assert (row[:5] == P.BURIED).all()  # beyond it nothing is seen
    assert (row[6:] == 0).all()


def test_window_at_the_top_shows_air():
    """Above y 191 the world is air, as in the game (no barrier ceiling)."""
    w = room(20)
    for y in range(FLOOR + 1, 189):
        w.set(10, y, 10, STONE)  # a pillar; the agent stands on it with its feet at 189
    win = P.voxel_window(w, (10.5, 189.0, 10.5))
    assert win.origin[1] + 6 == MAX_Y + 1
    assert (win.classes[:, 6, :] == 0).all() and (win.light[:, 6, :] == 15).all()


# -- line of sight --------------------------------------------------------------------------


def place_ore(w: World) -> tuple[int, int, int]:
    pos = (20, FLOOR + 1, 10)  # an iron ore block standing on the floor, 9.5 blocks away
    w.set(*pos, IRON)
    return pos


def test_sighting_in_open_air():
    w = room()
    pos = place_ore(w)
    s = P.sightings(w, EYE)
    hit = s.nearest(IRON)
    assert hit is not None and hit.pos == pos and abs(hit.distance - np.linalg.norm(np.add(pos, 0.5) - EYE)) < 1e-9
    assert s.counts[IRON] == 1 and not s.xray
    assert s.counts[STONE] == P.PER_CLASS  # the exposed floor: the 8 nearest of its blocks
    assert P.sightings(w, EYE, per_class=None).counts[STONE] > 100


def test_los_blocked_by_stone_but_not_glass():
    w = room()
    place_ore(w)
    wall(w, 15, STONE)
    assert P.sightings(w, EYE).nearest(IRON) is None
    wall(w, 15, GLASS)
    assert P.sightings(w, EYE).nearest(IRON) is not None
    wall(w, 15, ID["oak_leaves"])  # leaves are transparent too
    assert P.sightings(w, EYE).nearest(IRON) is not None
    wall(w, 15, ID["water"])  # and water
    assert P.sightings(w, EYE).nearest(IRON) is not None


def test_lava_blocks_sight():
    w = room()
    place_ore(w)
    wall(w, 15, ID["lava"], ys=range(FLOOR + 1, FLOOR + 10))
    assert C.transparent[ID["lava"]] and not P.see_through_table(C)[ID["lava"]]
    assert P.sightings(w, EYE).nearest(IRON) is None
    assert not P.line_of_sight(w, EYE, (20.5, FLOOR + 1.5, 10.5))


def test_only_faces_that_face_the_eye_count():
    w = room()
    w.set(20, FLOOR, 10, IRON)  # in the floor, top face open
    assert P.sightings(w, EYE).nearest(IRON) is not None
    under = room()
    under.set(20, FLOOR, 10, IRON)
    under_feet = (20.5, 30.0, 10.5)  # in a cave below the floor, sealed off from the ore
    for x in range(17, 24):
        for z in range(7, 14):
            for y in range(30, 34):
                under.set(x, y, z, 0)
    assert P.sightings(under, P.eye_of(under_feet)).nearest(IRON) is None


def test_sight_range_24():
    w = room(60)
    near, far = (10 + 22, FLOOR + 2, 10), (10, FLOOR + 2, 10 + 26)
    w.set(*near, IRON)
    w.set(*far, DIAMOND)
    s = P.sightings(w, EYE)
    assert s.nearest(IRON) is not None and s.nearest(DIAMOND) is None
    assert P.sightings(w, EYE, sight_range=30).nearest(DIAMOND) is not None


def test_counts_and_sorting():
    w = room()
    for x in (14, 16, 18):
        w.set(x, FLOOR + 1, 10, IRON)
    s = P.sightings(w, EYE)
    irons = s.of(IRON)
    assert [h.pos[0] for h in irons] == [14, 16, 18] and s.counts[IRON] == 3
    assert all(a.distance <= b.distance for a, b in zip(s.items, s.items[1:]))


def test_per_class_cap_keeps_the_nearest():
    w = room()
    for z in range(0, 34, 2):
        w.set(14, FLOOR + 1, z, IRON)  # 17 ores in a row, all within 24 blocks
    assert P.PER_CLASS == 8
    s = P.sightings(w, EYE)
    full = P.sightings(w, EYE, per_class=None)
    assert s.counts[IRON] == 8 and s.counts[STONE] == 8 and full.counts[IRON] == 17
    assert s.of(IRON) == full.of(IRON)[:8]  # the 8 nearest visible ones
    assert s.rays < full.rays / 10  # the floor's hundreds of stone blocks cost no rays once 8 are seen


def test_sighting_cache_is_exact():
    w = room()
    place_ore(w)
    cache = P.SightingCache()
    first = cache.get(w, EYE, 0)
    again = cache.get(w, EYE, 5)
    assert cache.hits == 1 and again == P.sightings(w, EYE, 5) and again.tick == 5
    w.set(30, FLOOR + 1, 30, IRON)  # inside the sight box: recomputed
    assert cache.get(w, EYE, 10) == P.sightings(w, EYE, 10) and cache.hits == 1
    big = World(80, 80)
    b = big.blocks.copy()
    b[:, : FLOOR - MIN_Y + 1, :] = STONE
    big.load(b)
    cache = P.SightingCache()
    cache.get(big, EYE, 0)
    big.set(70, FLOOR + 1, 70, IRON)  # far outside the box: the cached result stays valid
    assert cache.get(big, EYE, 5) == P.sightings(big, EYE, 5) and cache.hits == 1
    big.set(20, FLOOR + 1, 10, IRON)  # inside: recomputed, and the new ore is seen
    assert cache.get(big, EYE, 10).nearest(IRON) is not None and cache.hits == 1
    moved = P.eye_of((11.5, FLOOR + 1.0, 10.5))
    assert cache.get(big, moved, 15) == P.sightings(big, moved, 15) and cache.hits == 1
    assert first.nearest(IRON) is not None
    p = P.perceive(big, (11.5, FLOOR + 1.0, 10.5), 20, cache=cache)
    assert cache.hits == 2 and p.sightings == P.sightings(big, moved, 20)


def test_xray_is_flagged_and_sees_buried_blocks():
    w = room()
    w.set(12, FLOOR - 5, 12, DIAMOND)
    honest = P.sightings(w, EYE)
    xray = P.sightings(w, EYE, xray=True)
    assert honest.nearest(DIAMOND) is None and not honest.xray
    assert xray.xray and xray.nearest(DIAMOND).pos == (12, FLOOR - 5, 12) and xray.rays == 0
    nearest_stone = P.sightings(w, EYE, xray=True, per_class=1).nearest(STONE)
    assert nearest_stone.pos == (10, FLOOR, 10)  # x-ray: the nearest, exposed or not
    p = P.perceive(w, FEET, 0, xray=True)
    assert p.xray and p.sightings.xray


def test_line_of_sight_and_entities():
    w = room()
    target = (20.5, FLOOR + 1.0, 10.5)
    assert P.line_of_sight(w, EYE, (20.5, FLOOR + 1.5, 10.5))
    assert P.entity_visible(w, EYE, target, 1.95)
    wall(w, 15, STONE)
    assert not P.line_of_sight(w, EYE, (20.5, FLOOR + 1.5, 10.5))
    assert not P.entity_visible(w, EYE, target, 1.95)
    wall(w, 15, GLASS)
    assert P.entity_visible(w, EYE, target, 1.95)
    assert not P.entity_visible(w, EYE, (10.5, FLOOR + 1.0, 39.5), 1.95, max_distance=20)


def test_corner_ties_are_deterministic():
    """A ray through an edge between two cells steps x first: the x-side cell decides."""
    w = room()
    eye = (10.5, FLOOR + 2.5, 10.5)
    end = (12.5, FLOOR + 2.5, 12.5)  # passes exactly through the edge at (11, 11)
    w.set(10, FLOOR + 2, 11, STONE)  # the z-side cell: never visited
    assert P.line_of_sight(w, eye, end)
    w.set(10, FLOOR + 2, 11, 0)
    w.set(11, FLOOR + 2, 10, STONE)  # the x-side cell: visited first
    assert not P.line_of_sight(w, eye, end)


# -- the schedule ----------------------------------------------------------------------------


def test_ray_schedule_every_5_ticks():
    sched = P.RaySchedule()
    assert sched.period == 5
    assert [t for t in range(12) if sched.due(t)] == [0, 5, 10]
    assert [t for t in range(12) if P.RaySchedule(phase=3).due(t)] == [3, 8]
    w = room()
    assert P.perceive(w, FEET, 5).sightings is not None
    p = P.perceive(w, FEET, 6)
    assert p.sightings is None and p.window.classes.shape == P.WINDOW_SHAPE
    assert P.perceive(w, FEET, 6, schedule=None).sightings is not None


# -- honesty ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cave_world() -> World:
    return wg.generate(3, wg.preset("forest", size_x=48, size_z=48, caves_per_chunk=2.0))


def cave_feet(w: World) -> tuple[float, float, float]:
    """A feet spot inside a carved cave: air with a solid floor and air at head height, at least
    5 below the surface; the one nearest the world's centre (then lowest x, y, z)."""
    air = w.blocks == 0
    ok = air.copy()
    ok[:, 1:, :] &= C.solid[w.blocks[:, :-1, :]]
    ok[:, :-1, :] &= air[:, 1:, :]
    ys = np.arange(w.blocks.shape[1])[None, :, None] + MIN_Y
    ok &= ys < np.asarray(w.meta["surface"])[:, None, :] - 4
    pts = np.argwhere(ok)
    assert len(pts), "the cave world has no cave"
    d = np.abs(pts[:, 0] - w.size_x // 2) + np.abs(pts[:, 2] - w.size_z // 2)
    x, i, z = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0], d))[0]]
    return (x + 0.5, float(i + MIN_Y), z + 0.5)


def _hidden_cells(w: World, feet, radius: int = 28) -> np.ndarray:
    """World cells near the agent that are not exposed (no see-through neighbour)."""
    see = P.see_through_table(C)
    fx, fy, fz = P.feet_cell(feet)
    base = np.array([fx - radius - 1, fy - radius - 1, fz - radius - 1])
    region = w.region(*base, *(base + 2 * radius + 3))
    hidden = ~P.exposed_mask(see[region]) & ~see[region[1:-1, 1:-1, 1:-1]]
    hidden &= C.block_defined[region[1:-1, 1:-1, 1:-1]] & (region[1:-1, 1:-1, 1:-1] != ID["barrier"])
    return np.argwhere(hidden) + base + 1


def _same(a: P.Perception, b: P.Perception) -> bool:
    return (np.array_equal(a.window.classes, b.window.classes) and np.array_equal(a.window.light, b.window.light)
            and a.sightings == b.sightings)


@pytest.mark.parametrize("where", ["spawn", "cave"])
def test_honest_mode_ignores_hidden_changes(cave_world, where):
    feet = cave_world.meta["spawn"] if where == "spawn" else cave_feet(cave_world)
    w = cave_world.copy()
    before = P.perceive(w, feet, 0)
    if where == "cave":
        assert feet[1] < w.meta["surface"][int(feet[0])][int(feet[2])] - 4 and w.get(*map(int, feet)) == 0
        assert before.sightings.items and (before.window.classes == P.BURIED).any()
    cells = _hidden_cells(w, feet)
    assert len(cells) > 10_000
    rng = CounterRNG(8)
    choices = [ID[n] for n in ("stone", "deepslate", "iron_ore", "diamond_ore", "coal_ore", "dirt", "granite",
                               "deepslate_iron_ore", "gold_ore", "obsidian")]
    for i, (x, y, z) in enumerate(cells[:: 7].tolist()):
        if w.in_bounds(x, y, z):
            w.set(x, y, z, choices[rng.below((0, 0, Purpose.GENERIC, 0, i), len(choices))])
    after = P.perceive(w, feet, 0)
    assert _same(before, after)
    assert P.sightings(w, P.eye_of(feet), per_class=None) == P.sightings(cave_world, P.eye_of(feet), per_class=None)
    xray_after = P.sightings(w, P.eye_of(feet), xray=True, per_class=None)
    assert xray_after != P.sightings(cave_world, P.eye_of(feet), xray=True, per_class=None)  # x-ray sees them


def test_exposed_but_out_of_sight_blocks_are_not_reported():
    """A sealed cave next door: its walls are exposed (to the cave air) but not in sight."""
    w = room()
    for x in range(20, 25):
        for z in range(5, 10):
            for y in range(50, 53):
                w.set(x, y, z, 0)
    w.set(22, 49, 7, IRON)  # the cave floor: exposed, out of sight
    before = P.perceive(w, FEET, 0)
    assert before.sightings.nearest(IRON) is None
    w.set(22, 49, 7, DIAMOND)
    assert _same(before, P.perceive(w, FEET, 0))
    w.set(22, 50, 7, ID["lava"])  # a light source in the sealed cave changes nothing visible either
    assert _same(before, P.perceive(w, FEET, 0))

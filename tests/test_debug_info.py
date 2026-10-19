"""Tests for hivemind.sim.ref.debug_info: the F3 debug group (DESIGN.md 6.2; contracts 6).

Position relative to home with y absolute, Minecraft's yaw and pitch conventions and the facing
word, the three look-ray targets (and that none of them is ever reported through or behind the
targeted block), the time words and day number, ticks since the episode began, the biome and the
light at the feet."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from hivemind.rules import load_concepts
from hivemind.sim.ref import debug_info as D
from hivemind.sim.ref import perception as P
from hivemind.sim.ref.world import MIN_Y, World

C = load_concepts()
ID = {c.name: c.id for c in C.block_classes}
AIR, STONE, GLASS, WATER, LAVA, TORCH, LEAVES = (0, ID["stone"], ID["glass"], ID["water"], ID["lava"], ID["torch"],
                                                 ID["oak_leaves"])
ZOMBIE = C.entity_class("zombie").id
PLAYER = C.entity_class("player").id
RAW_IRON = C.item_class("raw_iron").id
FLOOR = 60
FEET = (10.5, 61.0, 10.5)  # eye at y 62.62
EAST, WEST, NORTH, SOUTH = -90.0, 90.0, 180.0, 0.0  # yaws [mem]


def world_with(fills=(), size: int = 40, day_time: int = 0) -> World:
    """Stone up to y 60, air above, then each fill (class, (x0, x1), (y0, y1), (z0, z1)), inclusive."""
    w = World(size, size, day_time=day_time)
    b = w.blocks.copy()
    b[:, : FLOOR - MIN_Y + 1, :] = STONE
    for cls, (x0, x1), (y0, y1), (z0, z1) in fills:
        b[x0:x1 + 1, y0 - MIN_Y:y1 - MIN_Y + 1, z0:z1 + 1] = cls
    w.load(b)
    return w


def wall(cls: int, x: int) -> tuple:
    """A wall across the world at column x, from the floor up 4 blocks."""
    return (cls, (x, x), (61, 64), (0, 39))


def f3(w: World, yaw: float = EAST, pitch: float = 0.0, feet=FEET) -> D.DebugInfo:
    slot = w.alloc_agent(feet, yaw=yaw, pitch=pitch)
    return D.debug_info(w, slot)


def zombie(w: World, x: float, y: float = 61.0, z: float = 10.5) -> int:
    return w.mobs.alloc(cls=ZOMBIE, pos=(x, y, z))


# ---------------------------------------------------------------------------
# Position and facing
# ---------------------------------------------------------------------------


def test_position_relative_to_home_y_absolute():
    w = world_with()
    slot = w.alloc_agent((5.25, 70.0, 7.75))  # home defaults to the spawn position
    g = D.debug_info(w, slot)
    assert (g.x, g.y, g.z) == (0.0, 70.0, 0.0)
    w.agents.pos[slot] = (8.25, 71.5, 3.75)
    g = D.debug_info(w, slot)
    assert (g.x, g.y, g.z) == (3.0, 71.5, -4.0)  # x and z relative to home, y absolute
    other = w.alloc_agent((30.5, 61.0, 30.5), home=(20.0, 0.0, 40.0))  # from M9: the team's base
    g = D.debug_info(w, other)
    assert (g.x, g.y, g.z) == (10.5, 61.0, -9.5)
    names = {f.name for f in dataclasses.fields(D.DebugInfo)}
    assert not names & {"abs_x", "abs_z", "pos", "feet"}  # absolute x and z are never given
    raw = w.agents.alloc(pos=(1.5, 61.0, 1.5))  # a slot whose home was never set
    with pytest.raises(ValueError):
        D.debug_info(w, raw)
    with pytest.raises(ValueError):
        D.debug_info(w, 39)  # not in use


@pytest.mark.parametrize("yaw, pitch, expected", [
    (SOUTH, 0.0, (0.0, 0.0, 1.0)), (WEST, 0.0, (-1.0, 0.0, 0.0)), (NORTH, 0.0, (0.0, 0.0, -1.0)),
    (EAST, 0.0, (1.0, 0.0, 0.0)), (0.0, -90.0, (0.0, 1.0, 0.0)), (0.0, 90.0, (0.0, -1.0, 0.0)),
    (45.0, 0.0, (-math.sqrt(0.5), 0.0, math.sqrt(0.5))), (0.0, 30.0, (0.0, -0.5, math.sqrt(0.75))),
])
def test_view_vector_conventions(yaw, pitch, expected):
    """Yaw 0 faces south (+z), 90 west, 180 north, -90 east; pitch -90 up, +90 down [mem]."""
    v = D.view_vector(yaw, pitch)
    assert np.allclose(v, expected, atol=1e-4)  # the game's 65,536-entry sine table: steps of 9.6e-5 rad
    assert abs(math.sqrt(sum(c * c for c in v)) - 1.0) < 1e-6
    assert all(float(np.float32(c)) == c for c in v)  # float32 values: the game's own arithmetic


def test_view_vector_uses_the_games_sine_table():
    """Mth.sin truncates the angle to a table entry [mem]: 30 degrees is entry 5461, not 5461.33."""
    v = D.view_vector(0.0, 30.0)
    assert v[1] == -float(np.float32(math.sin(5461 * 2.0 * math.pi / 65536.0)))
    assert v[1] != -0.5 and abs(v[1] + 0.5) < 3e-5


def test_facing_word_and_wrapped_yaw():
    for yaw, word in ((0.0, "south"), (44.9, "south"), (45.0, "west"), (90.0, "west"), (134.9, "west"),
                      (135.0, "north"), (180.0, "north"), (-180.0, "north"), (-135.1, "north"),
                      (-135.0, "east"), (-90.0, "east"), (-45.1, "east"), (-45.0, "south"), (270.0, "east"),
                      (450.0, "west"), (-360.0, "south"), (720.0 + 180.0, "north")):
        assert D.facing_word(yaw) == word, yaw
    for angle, wrapped in ((270.0, -90.0), (180.0, -180.0), (-180.0, -180.0), (540.0, -180.0), (-190.0, 170.0),
                           (359.5, -0.5), (0.0, 0.0), (179.5, 179.5), (-720.0, 0.0)):
        assert D.wrap_degrees(angle) == wrapped, angle
    g = f3(world_with(), yaw=270.0, pitch=-12.5)
    assert (g.yaw, g.pitch, g.facing) == (-90.0, -12.5, "east")


# ---------------------------------------------------------------------------
# The targeted block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yaw, pitch, fills, face, distance", [
    (EAST, 0.0, [wall(STONE, 13)], "west", 2.5),
    (WEST, 0.0, [wall(STONE, 8)], "east", 1.5),
    (NORTH, 0.0, [(STONE, (0, 39), (61, 64), (8, 8))], "south", 1.5),
    (SOUTH, 0.0, [(STONE, (0, 39), (61, 64), (13, 13))], "north", 2.5),
    (0.0, 90.0, [], "top", 1.62),  # the floor under the feet
    (0.0, -90.0, [(STONE, (0, 39), (64, 64), (0, 39))], "bottom", 64 - 62.62),  # a ceiling
])
def test_targeted_block_face_and_distance(yaw, pitch, fills, face, distance):
    g = f3(world_with(fills), yaw, pitch)
    assert g.block == STONE and g.block_face == face
    assert g.block_distance == pytest.approx(distance, abs=1e-6)
    assert g.fluid is None and g.entity is None


def test_block_reach_is_4_5_strict():
    assert D.BLOCK_REACH == 4.5 and D.ENTITY_REACH == 3.0
    assert f3(world_with([wall(STONE, 14)])).block_distance == pytest.approx(3.5)
    assert f3(world_with([wall(STONE, 15)])).block is None  # exactly 4.5: out of reach (closerThan is strict)
    out = f3(world_with([wall(STONE, 16)]))
    assert out.block is None and out.block_face is None and out.block_distance is None


def test_glass_and_leaves_are_targeted_blocks():
    for cls in (GLASS, LEAVES, TORCH):
        g = f3(world_with([wall(cls, 12), wall(STONE, 14)]))
        assert g.block == cls and g.block_distance == pytest.approx(1.5) and g.block_face == "west"


def test_head_inside_a_block():
    w = world_with([(ID["sand"], (10, 10), (62, 62), (10, 10))])  # a fallen block at head height
    zombie(w, 11.5)
    for yaw, pitch, face in ((EAST, 0.0, "west"), (0.0, 90.0, "top"), (0.0, -90.0, "bottom"), (NORTH, 0.0, "south")):
        g = f3(w.copy(), yaw, pitch)
        assert (g.block, g.block_distance, g.block_face) == (ID["sand"], 0.0, face)  # getNearest(d).getOpposite()
        assert g.fluid is None and g.entity is None


# ---------------------------------------------------------------------------
# The targeted fluid
# ---------------------------------------------------------------------------


def test_water_is_passed_and_named():
    g = f3(world_with([wall(WATER, 12), wall(STONE, 14)]))
    assert (g.fluid, g.fluid_distance) == ("water", pytest.approx(1.5))
    assert g.block == STONE and g.block_distance == pytest.approx(3.5)
    only = f3(world_with([wall(WATER, 12)]))
    assert only.fluid == "water" and only.block is None
    far = f3(world_with([wall(WATER, 16)]))  # 5.5 away: beyond the reach
    assert far.fluid is None
    behind = f3(world_with([wall(STONE, 12), wall(WATER, 13)]))  # behind the targeted block
    assert behind.block == STONE and behind.fluid is None


def test_lava_stops_the_ray():
    g = f3(world_with([wall(LAVA, 12), wall(STONE, 14)]))
    assert (g.fluid, g.fluid_distance) == ("lava", pytest.approx(1.5)) and g.block is None
    first_water = f3(world_with([wall(WATER, 12), wall(LAVA, 13), wall(STONE, 14)]))
    assert first_water.fluid == "water" and first_water.block is None  # the fluid line shows the first
    behind = f3(world_with([wall(STONE, 12), wall(LAVA, 13)]))
    assert behind.block == STONE and behind.fluid is None  # lava behind a block is not named


def test_head_in_water():
    g = f3(world_with([(WATER, (5, 15), (61, 64), (5, 15))]))
    assert (g.fluid, g.fluid_distance) == ("water", 0.0)


# ---------------------------------------------------------------------------
# The targeted entity
# ---------------------------------------------------------------------------


def test_entity_within_reach_is_targeted():
    w = world_with()
    zombie(w, 12.5)  # box from x 12.2: 1.7 from the eye
    g = f3(w)
    assert (g.entity, g.entity_agent) == (ZOMBIE, None) and g.entity_distance == pytest.approx(1.7)
    far = world_with()
    zombie(far, 14.0)  # 3.2 away: beyond the entity reach
    assert f3(far).entity is None
    exact = world_with()
    zombie(exact, 13.8)  # its box starts exactly 3.0 away: closerThan is strict
    assert f3(exact).entity is None


def test_entity_nearer_than_the_block_is_targeted_with_it():
    w = world_with([wall(STONE, 14)])
    zombie(w, 12.5)
    g = f3(w)
    assert g.entity == ZOMBIE and g.block == STONE and g.entity_distance < g.block_distance


@pytest.mark.parametrize("cls", [STONE, GLASS, LEAVES, LAVA])
def test_entity_behind_the_targeted_block_or_lava_is_never_named(cls):
    """A zombie within 3.0 blocks but behind the block (or lava) the ray stops at."""
    w = world_with([wall(cls, 12)])
    base = f3(w.copy())
    zombie(w, 13.5)  # box from x 13.2: 2.7 away, behind the wall at 1.5
    g = f3(w)
    assert g.entity is None and g == base  # the whole group is unchanged
    if cls in (GLASS, LEAVES):
        assert g.block == cls  # seen through, but the crosshair stops at it
        assert P.entity_visible(w, P.eye_of(FEET), (13.5, 61.0, 10.5), 1.95)


def test_a_wall_beyond_the_block_reach_still_stops_the_entity_pick():
    """With an entity reach larger than the block reach (both are parameters), a wall between the
    two reaches is not the targeted block, yet the entity pick stops at it, as the game's pick
    casts its block ray to the larger reach [mem]: nothing behind a wall is named."""
    for cls in (STONE, GLASS, LEAVES, LAVA):
        w = world_with([wall(cls, 13)])  # west face 2.5 east of the eye
        zombie(w, 14.5)  # box from x 14.2: 3.7 away, behind the wall
        g = D.debug_info(w, w.alloc_agent(FEET, yaw=EAST), reach=2.0, entity_reach=4.0)
        assert g.entity is None and g.block is None and g.fluid is None, cls  # lava at 2.5: beyond the reach too
    open_ground = world_with()
    zombie(open_ground, 14.5)
    g = D.debug_info(open_ground, open_ground.alloc_agent(FEET, yaw=EAST), reach=2.0, entity_reach=4.0)
    assert g.entity == ZOMBIE and g.entity_distance == pytest.approx(3.7)  # positive control: no wall
    front = world_with([wall(STONE, 13)])
    zombie(front, 12.5)  # box from x 12.2: 1.7 away, in front of the wall
    g = D.debug_info(front, front.alloc_agent(FEET, yaw=EAST), reach=1.0, entity_reach=4.0)
    assert g.entity == ZOMBIE and g.block is None  # the wall at 2.5 is beyond the block reach of 1.0
    water = world_with([wall(WATER, 12), wall(STONE, 14)])
    zombie(water, 13.5)  # in the water behind the first water cell
    g = D.debug_info(water, water.alloc_agent(FEET, yaw=EAST), reach=1.0, entity_reach=4.0)
    assert g.fluid is None and g.block is None and g.entity == ZOMBIE  # water at 1.5: passed, not named


@pytest.mark.parametrize("plant", ["kelp", "seagrass", "tall_seagrass"])
def test_a_plant_in_water_is_the_targeted_block_and_its_water_the_fluid(plant):
    """Kelp and seagrass always stand in water: F3's fluid line names the water of the first cell its
    fluid pick hits [mem], so the plant is the targeted block and ``water`` the targeted fluid."""
    cls = ID[plant]
    g = f3(world_with([wall(cls, 12), wall(STONE, 14)]))  # from the air
    assert (g.block, g.block_face, g.fluid) == (cls, "west", "water")
    assert g.block_distance == pytest.approx(1.5) and g.fluid_distance == pytest.approx(1.5)
    pool = (WATER, (5, 15), (61, 64), (5, 15))
    head = f3(world_with([pool, (cls, (10, 10), (61, 64), (10, 10))]))  # the head inside the plant
    assert (head.block, head.block_distance, head.fluid, head.fluid_distance) == (cls, 0.0, "water", 0.0)
    swim = f3(world_with([pool, wall(cls, 12)]))  # the head in open water, a plant ahead
    assert (swim.fluid, swim.fluid_distance) == ("water", 0.0) and swim.block == cls
    assert swim.block_distance == pytest.approx(1.5)
    behind = world_with([pool, wall(cls, 12)])
    zombie(behind, 13.5)  # 2.7 away, behind the plant: the plant is a targeted block, so it stops the pick
    assert f3(behind).entity is None


def test_entity_in_water_is_targeted():
    w = world_with([(WATER, (11, 14), (61, 64), (0, 39))])
    zombie(w, 12.5)
    g = f3(w)
    assert g.fluid == "water" and g.fluid_distance == pytest.approx(0.5) and g.entity == ZOMBIE


def test_players_items_dead_and_self():
    w = world_with()
    me = w.alloc_agent(FEET, yaw=EAST)
    other = w.alloc_agent((12.5, 61.0, 10.5))
    g = D.debug_info(w, me)
    assert (g.entity, g.entity_agent) == (PLAYER, other)  # the builder names it (e.g. blue 7)
    w.agents.alive[other] = False
    assert D.debug_info(w, me).entity is None  # the dead leave no body [mem]
    w.items.alloc(item=RAW_IRON, count=1, pos=(11.5, 62.5, 10.5))  # a dropped item right at eye height
    assert D.debug_info(w, me).entity is None  # dropped items are not pickable [mem]
    boxes = P.entity_boxes(w, exclude_agent=me)  # passing boxes with items in them changes nothing
    eye, d = P.eye_of(FEET), D.view_vector(EAST, 0.0)
    assert D.look_targets(w, eye, d, boxes=boxes).entity is None


def test_a_box_that_contains_the_eye_is_targeted_at_distance_0():
    w = world_with([wall(STONE, 12)])
    zombie(w, 10.6)  # its box holds the observer's eye
    g = f3(w)
    assert (g.entity, g.entity_distance) == (ZOMBIE, 0.0) and g.block == STONE


# ---------------------------------------------------------------------------
# Time, biome, light
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("day_time, word, tick, day", [
    (0, "day", 0, 0), (11_999, "day", 11_999, 0), (12_000, "dusk", 12_000, 0), (12_999, "dusk", 12_999, 0),
    (13_000, "night", 13_000, 0), (22_999, "night", 22_999, 0), (23_000, "dawn", 23_000, 0),
    (23_999, "dawn", 23_999, 0), (24_000, "day", 0, 1), (24_000 * 5 + 12_500, "dusk", 12_500, 5),
])
def test_time_word_and_day_number(day_time, word, tick, day):
    g = f3(world_with(day_time=day_time))
    assert (g.time_word, g.tick_of_day, g.day, g.ticks) == (word, tick, day, 0)
    assert D.time_word(day_time) == word and D.day_number(day_time) == day and D.tick_of_day(day_time) == tick


def test_ticks_since_the_episode_began_and_the_night_skip():
    w = world_with(day_time=23_990)
    w.advance(20)
    slot = w.alloc_agent(FEET)
    g = D.debug_info(w, slot)
    assert (g.ticks, g.tick_of_day, g.day, g.time_word) == (20, 10, 1, "day")
    w.day_time += 13_000  # a night skip moves the clock but no ticks pass
    g = D.debug_info(w, slot)
    assert (g.ticks, g.tick_of_day, g.day, g.time_word) == (20, 13_010, 1, "night")


def test_biome_at_the_feet():
    w = world_with()
    biomes = np.full((40, 40), C.biome("plains").id, np.uint8)
    biomes[20:, :] = C.biome("desert").id
    w.set_biome_map(biomes)
    assert f3(w.copy()).biome_name == "plains" and f3(w.copy()).biome == C.biome("plains").id
    east = f3(w.copy(), feet=(25.5, 61.0, 10.5))
    assert east.biome_name == "desert" and C.biome(east.biome).phrase == ("desert",)
    assert f3(w.copy(), feet=(19.99, 61.0, 10.5)).biome_name == "plains"  # the feet's column, floor(x)


def test_light_at_the_feet():
    g = f3(world_with())
    assert (g.sky_light, g.block_light) == (15, 0)
    lit = f3(world_with([(TORCH, (12, 12), (61, 61), (10, 10))]))
    assert (lit.sky_light, lit.block_light) == (15, 12)  # 14 at the torch, 2 cells away
    roofed = f3(world_with([(STONE, (0, 39), (66, 66), (0, 39))]))
    assert roofed.sky_light == 0


# ---------------------------------------------------------------------------
# One shared function
# ---------------------------------------------------------------------------


def test_perceive_carries_the_same_group_and_is_deterministic():
    w = world_with([wall(STONE, 13), (TORCH, (9, 9), (61, 61), (12, 12))], day_time=13_500)
    slot = w.alloc_agent(FEET, yaw=-80.0, pitch=10.0)
    zombie(w, 12.0, z=10.9)
    g = D.debug_info(w, slot)
    assert P.perceive(w, agent=slot).f3 == g == D.debug_info(w.copy(), slot)
    assert P.perceive(w, agent=slot, preset="S").f3 == g  # both presets have the F3 group
    assert g.entity == ZOMBIE and g.block == STONE and g.time_word == "night"

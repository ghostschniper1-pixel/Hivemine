"""Tests for hivemind.sim.ref.perception (v2, draft 0.5): surface-only vision, the S and M windows,
the semantic panorama, entity boxes, resource sightings, the x-ray ablation, the schedule, and
honesty (DESIGN.md 6.1, 6.2; contracts 6).

The honesty tests build adversarial scenes (a cave behind a one-block wall, ores under the floor,
an entity behind a block, a lava wall, glass, leaves, a sealed room beside the agent) and
generated cave worlds, change what the eye cannot see, and require the window, the panorama, the
sightings and the F3 group to stay exactly the same."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hivemind.lang.world_words import FACE_WORDS as WORLD_FACE_WORDS
from hivemind.rules import load_concepts, load_rules
from hivemind.sim.ref import debug_info as D
from hivemind.sim.ref import perception as P
from hivemind.sim.ref import worldgen as wg
from hivemind.sim.ref.rng import CounterRNG, Purpose
from hivemind.sim.ref.world import MAX_Y, MIN_Y, UNSEEN, World, light_filter_table

C = load_concepts()
R = load_rules()
ID = {c.name: c.id for c in C.block_classes}

# The oracle for sight, written by hand and independent of perception's table: the vanilla blocks a
# player sees through or past [mem]. A class is see-through when every member is on this list.
_COLOURS = ("white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray", "light_gray", "cyan",
            "purple", "blue", "brown", "green", "red", "black")
_WOODS = ("oak", "spruce", "birch", "jungle", "acacia", "cherry", "dark_oak", "mangrove", "bamboo", "crimson", "warped")
_CORALS = ("tube", "brain", "bubble", "fire", "horn")
KNOWN_SEE_THROUGH = frozenset({
    # nothing there, or an invisible block
    "air", "void_air", "cave_air", "light", "structure_void",
    # glass, ice and see-through solids
    "glass", "tinted_glass", "glass_pane", *(f"{c}_stained_glass" for c in _COLOURS),
    *(f"{c}_stained_glass_pane" for c in _COLOURS), "ice", "frosted_ice", "slime_block", "honey_block",
    "iron_bars", "chain", *(f"{w}{a}copper_grate" for w in ("", "waxed_") for a in ("", "exposed_", "weathered_",
                                                                                   "oxidized_")),
    "spawner", "trial_spawner", "scaffolding", "mangrove_roots",
    "oak_leaves", "spruce_leaves", "birch_leaves", "jungle_leaves", "acacia_leaves", "cherry_leaves",
    "dark_oak_leaves", "mangrove_leaves", "azalea_leaves", "flowering_azalea_leaves",
    # fluids a player sees through (not lava)
    "water", "bubble_column",
    # small or thin things
    "torch", "wall_torch", "redstone_torch", "redstone_wall_torch", "soul_torch", "soul_wall_torch", "lantern",
    "soul_lantern", "end_rod", "lightning_rod", "ladder", "rail", "powered_rail", "detector_rail", "activator_rail",
    "lever", "tripwire", "tripwire_hook", "redstone_wire", "stone_button", "polished_blackstone_button",
    *(f"{w}_button" for w in _WOODS), *(f"{w}_pressure_plate" for w in _WOODS), "stone_pressure_plate",
    "polished_blackstone_pressure_plate", "light_weighted_pressure_plate", "heavy_weighted_pressure_plate",
    "candle", *(f"{c}_candle" for c in _COLOURS), "sea_pickle", "conduit", "brewing_stand", "cocoa", "frogspawn",
    "lily_pad", "flower_pot", "fire", "soul_fire", "nether_portal", "cobweb",
    "amethyst_cluster", "large_amethyst_bud", "medium_amethyst_bud", "small_amethyst_bud",
    # plants
    *(f"{w}_sapling" for w in ("oak", "spruce", "birch", "jungle", "acacia", "cherry", "dark_oak")),
    "mangrove_propagule", "short_grass", "tall_grass", "fern", "large_fern", "dead_bush", "dandelion", "torchflower",
    "poppy", "blue_orchid", "allium", "azure_bluet", "red_tulip", "orange_tulip", "white_tulip", "pink_tulip",
    "oxeye_daisy", "cornflower", "lily_of_the_valley", "wither_rose", "sunflower", "lilac", "rose_bush", "peony",
    "brown_mushroom", "red_mushroom", "sugar_cane", "sweet_berry_bush", "vine", "bamboo", "bamboo_sapling",
    "kelp", "kelp_plant", "seagrass", "tall_seagrass", "wheat", "carrots", "potatoes", "beetroots", "melon_stem",
    "attached_melon_stem", "pumpkin_stem", "attached_pumpkin_stem", "torchflower_crop", "pitcher_crop",
    "pitcher_plant", "cave_vines", "cave_vines_plant", "weeping_vines", "weeping_vines_plant", "twisting_vines",
    "twisting_vines_plant", "hanging_roots", "glow_lichen", "sculk_vein", "pink_petals", "big_dripleaf_stem",
    "small_dripleaf", "spore_blossom", "nether_wart", "crimson_fungus", "warped_fungus", "crimson_roots",
    "warped_roots", "nether_sprouts",
    *(f"{d}{c}_coral{k}" for d in ("", "dead_") for c in _CORALS for k in ("", "_fan", "_wall_fan")),
    *(n for n in R.blocks if n.startswith("potted_")),  # a small pot with a plant in it
})
AIR, STONE, GLASS, IRON, DIAMOND = 0, ID["stone"], ID["glass"], ID["iron_ore"], ID["diamond_ore"]
LAVA, WATER, LEAVES, TORCH, OBSIDIAN, BARRIER = (ID[n] for n in ("lava", "water", "oak_leaves", "torch", "obsidian",
                                                                 "barrier"))
ZOMBIE = C.entity_class("zombie").id
PLAYER = C.entity_class("player").id
ITEM = C.entity_class("item").id
RAW_IRON = C.item_class("raw_iron").id
SEE = np.zeros(256, bool)  # the oracle's table (KNOWN_SEE_THROUGH), never perception's
for _c in C.block_classes:
    SEE[_c.id] = bool(_c.members) and all(m in KNOWN_SEE_THROUGH for m in _c.members)
FILTER = light_filter_table(C)
FLOOR = 60  # stone up to y 60, air above
FEET = (10.5, FLOOR + 1.0, 10.5)
EYE = P.eye_of(FEET)
PANO = P.PRESETS["M"].panorama
# Opaque, light-blocking, non-emitting classes: what the honesty tests may turn hidden cells into.
SWAPS = tuple(ID[n] for n in ("stone", "deepslate", "iron_ore", "diamond_ore", "coal_ore", "dirt", "granite",
                              "deepslate_iron_ore", "gold_ore", "obsidian"))


def world_with(size: int = 40, fills=(), solid_to: int = FLOOR) -> World:
    """Stone up to y ``solid_to`` and air above, then each fill (class, (x0, x1), (y0, y1), (z0, z1)),
    inclusive ranges, written before the light is computed."""
    w = World(size, size)
    b = w.blocks.copy()
    b[:, : solid_to - MIN_Y + 1, :] = STONE
    for cls, (x0, x1), (y0, y1), (z0, z1) in fills:
        b[x0:x1 + 1, y0 - MIN_Y:y1 - MIN_Y + 1, z0:z1 + 1] = cls
    w.load(b)
    return w


def room(size: int = 40) -> World:
    return world_with(size)


@pytest.fixture(scope="module")
def open_world() -> World:
    """A 140 x 140 stone floor: from its centre nothing but the floor lies within 64 blocks."""
    return world_with(140)


OPEN_FEET = (70.5, FLOOR + 1.0, 70.5)
OPEN_EYE = P.eye_of(OPEN_FEET)


def pano_at(w: World, feet=OPEN_FEET, **kw) -> P.Panorama:
    return P.panorama(w, P.eye_of(feet), PANO, **kw)


def column(azimuth: float) -> int:
    """The panorama column containing an azimuth (degrees clockwise from north)."""
    return math.floor((azimuth % 360.0) / PANO.column_step)


def row(elevation: float) -> int:
    """The panorama row containing an elevation (degrees, up positive)."""
    return math.floor((PANO.top - elevation) / PANO.row_step)


# ---------------------------------------------------------------------------
# Presets, constants, tables
# ---------------------------------------------------------------------------


def test_presets_s_and_m():
    s, m = P.PRESETS["S"], P.PRESETS["M"]
    assert (s.window_shape, s.half_xz, s.half_y, s.panorama) == ((15, 7, 15), 7, 3, None)  # DESIGN.md 6.1
    assert (m.window_shape, m.half_xz, m.half_y) == ((21, 9, 21), 10, 4)
    assert P.DEFAULT_PRESET == "M" and P.config_of("S") is s and P.config_of(m) is m
    pc = m.panorama
    assert (pc.columns, pc.rows, pc.range, pc.top, pc.bottom) == (96, 24, 64.0, 30.0, -60.0)
    assert pc.column_step == 3.75 and pc.row_step == 3.75
    with pytest.raises(ValueError):
        P.PerceptionConfig("bad", 5, 1, None)  # the eye's cell must lie inside the window
    with pytest.raises(ValueError):
        P.PanoramaConfig(rows=0)
    with pytest.raises(ValueError):
        P.PanoramaConfig(top=-70.0)
    big = P.PerceptionConfig("XL", 12, 5, None)  # the size is configurable
    assert P.voxel_window(room(), FEET, big).classes.shape == (25, 11, 25)


def test_eye_and_tables():
    assert P.EYE_HEIGHT == 1.62 and EYE == (10.5, FLOOR + 2.62, 10.5)
    assert UNSEEN == 255 and C.block_class(255).phrase == ("unseen",)
    table = P.see_through_table(C)
    for name in ("air", "glass", "oak_leaves", "leaves", "water", "torch", "short_grass"):
        assert table[ID[name]], name
    assert C.transparent[LAVA] and not table[LAVA]  # flagged transparent in the data, not see-through here
    assert C.transparent[BARRIER] and not table[BARRIER] and not table[UNSEEN]
    for name in ("stone", "dirt", "iron_ore", "sand", "log"):
        assert not table[ID[name]], name
    table[ID["stone"]] = True  # a copy: the cached table perception uses is not changed
    assert not P.see_through_table(C)[ID["stone"]]


def class_of(member: str) -> int:
    """The block class holding a vanilla block (by its default state)."""
    return C.block_class_of_state(R.blocks[member].default_state).id


def test_see_through_table_is_the_hand_written_oracle():
    """Perception's table (the data's render flag minus SIGHT_BLOCKING) is exactly the classes
    whose members are all on the hand-written list of blocks a player sees through."""
    table = P.see_through_table(C)
    assert set(np.flatnonzero(table).tolist()) == set(np.flatnonzero(SEE).tolist())
    assert all(m in R.blocks for m in KNOWN_SEE_THROUGH | P.SIGHT_BLOCKING)  # no misspelt names
    assert not KNOWN_SEE_THROUGH & P.SIGHT_BLOCKING
    assert all(R.blocks[m].transparent for m in P.SIGHT_BLOCKING)  # only the render flag needs them
    for member in ("oak_door", "iron_door", "oak_trapdoor", "iron_trapdoor", "white_bed", "cauldron",
                   "shulker_box", "decorated_pot", "sniffer_egg", "oak_sign", "white_banner", "hopper"):
        assert C.transparent[class_of(member)] and not table[class_of(member)], member
    for member in ("iron_bars", "slime_block", "frosted_ice", "copper_grate"):  # merged with a blocking member
        assert not table[class_of(member)], member


def test_shapes_that_cover_the_cell_block_sight():
    """A cross-check from the data's collision shapes: a member flagged transparent whose shape, in
    some state, spans a whole cross-section of the cell or fills 40% of it is opaque to sight,
    unless it is glass, ice, leaves or another block known to be see-through."""
    table = P.see_through_table(C)
    flagged = []
    for name, b in R.blocks.items():
        if not b.transparent or name in ("barrier",):
            continue
        covers = False
        for shape in b.collision_shapes:
            vol = sum((x1 - x0) * (y1 - y0) * (z1 - z0) for x0, y0, z0, x1, y1, z1 in shape)
            plane = any(sum(hi - lo >= 0.999 for lo, hi in ((box[0], box[3]), (box[1], box[4]), (box[2], box[5]))) >= 2
                        for box in shape)
            covers |= plane or vol >= 0.4
        if covers:
            flagged.append(name)
            assert name in KNOWN_SEE_THROUGH or not table[class_of(name)], name
    assert {"oak_door", "shulker_box", "cauldron", "decorated_pot", "glass", "oak_leaves"} <= set(flagged)


def test_face_words_are_the_shared_table():
    assert P.FACE_WORDS == WORLD_FACE_WORDS == ("top", "bottom", "north", "south", "east", "west")
    words = {(axis, step): P.FACE_WORDS[P.entered_face(axis, step)] for axis in range(3) for step in (-1, 1)}
    assert words == {(0, 1): "west", (0, -1): "east", (1, -1): "top", (1, 1): "bottom",
                     (2, 1): "north", (2, -1): "south"}  # a ray travelling south (+z) enters a north face


# ---------------------------------------------------------------------------
# The voxel window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["S", "M"])
def test_window_on_open_ground(preset):
    cfg = P.PRESETS[preset]
    win = P.voxel_window(room(), FEET, preset)
    h = cfg.half_y
    assert win.classes.shape == cfg.window_shape and win.classes.dtype == np.uint8
    assert win.origin == (10 - cfg.half_xz, FLOOR + 1 - h, 10 - cfg.half_xz) and not win.xray
    floor = h - 1  # the window row of y = FLOOR
    assert (win.classes[:, :floor, :] == UNSEEN).all() and not win.seen[:, :floor, :].any()  # under the floor
    assert (win.classes[:, floor, :] == STONE).all()  # every top face of the floor faces the eye
    assert (win.classes[:, floor + 1:, :] == AIR).all() and win.seen[:, floor:, :].all()  # seen air shows air
    assert (win.light[:, floor + 1:, :] == 15).all() and (win.light[:, :floor + 1, :] == 0).all()
    assert win.rays > 0


def test_ore_under_the_floor_is_unseen_until_dug_open():
    w = room()
    win = P.voxel_window(w, FEET, "M")  # origin (0, 57, 0)
    w.set(10, FLOOR - 1, 10, IRON)  # under the feet, every face covered
    assert P.voxel_window(w, FEET, "M") == win
    w.set(10, FLOOR, 10, AIR)  # dig the floor cell: the ore's top face now faces the eye
    dug = P.voxel_window(w, FEET, "M")
    assert dug.classes[10, FLOOR - 1 - 57, 10] == IRON and dug.classes[10, FLOOR - 57, 10] == AIR
    assert dug.classes[10, FLOOR - 2 - 57, 10] == UNSEEN


def test_only_faces_turned_toward_the_eye_count():
    """An ore whose one open face points away from the eye is unseen, though exposed."""
    w = world_with(fills=[(STONE, (13, 13), (61, 62), (9, 11)), (STONE, (14, 14), (62, 62), (9, 11)),
                          (STONE, (14, 14), (61, 61), (9, 9)), (STONE, (14, 14), (61, 61), (11, 11)),
                          (IRON, (14, 14), (61, 61), (10, 10))])
    assert P.exposed_faces(w, 14, 61, 10) == (1,)  # +x only: it faces away from the eye at x 10.5
    win = P.voxel_window(w, FEET, "M")
    assert win.classes[14, 61 - 57, 10] == UNSEEN
    w.set(13, 61, 10, AIR)  # open its west face, which faces the eye
    assert P.voxel_window(w, FEET, "M").classes[14, 61 - 57, 10] == IRON


def test_window_glass_leaves_water_pass_sight_lava_does_not():
    for wall, seen in ((GLASS, True), (LEAVES, True), (WATER, True), (LAVA, False), (STONE, False)):
        w = world_with(fills=[(wall, (12, 12), (61, 64), (0, 39)), (IRON, (15, 15), (61, 61), (10, 10))])
        win = P.voxel_window(w, FEET, "M")
        assert (win.classes[15, 61 - 57, 10] == IRON) == seen, C.block_class(wall).name
        assert win.classes[12, 61 - 57, 10] == wall  # the wall itself is seen, lava included


def test_window_at_world_edge_shows_barrier_only_where_seen():
    w = room()
    win = P.voxel_window(w, (1.5, FLOOR + 1.0, 1.5), "M")
    assert win.origin[0] == -9
    line = win.classes[:, FLOOR + 2 - win.origin[1], 11]  # y = FLOOR + 2, z = 1: x from -9 to 11
    assert line[8] == BARRIER  # x = -1: its east face touches the air at x = 0 and faces the eye
    assert (line[:8] == UNSEEN).all()  # beyond the edge nothing is seen
    assert (line[9:] == AIR).all()


def test_window_at_the_top_shows_air():
    """Above y 191 the world is air, as in the game (no barrier ceiling)."""
    w = world_with(20, fills=[(STONE, (10, 10), (FLOOR + 1, 188), (10, 10))])
    win = P.voxel_window(w, (10.5, 189.0, 10.5), "S")
    assert win.origin[1] + 6 == MAX_Y + 1
    assert (win.classes[:, 6, :] == AIR).all() and (win.light[:, 6, :] == 15).all()


def test_window_light_is_the_cells_own_light_where_seen():
    w = world_with(fills=[(AIR, (4, 16), (61, 64), (4, 16)), (TORCH, (12, 12), (61, 61), (10, 10))], solid_to=70)
    win = P.voxel_window(w, FEET, "M")
    x0, y0, z0 = win.origin
    for (i, j, k) in np.argwhere(np.ones(win.classes.shape, bool))[::7]:
        x, y, z = x0 + i, y0 + j, z0 + k
        expected = w.light(x, y, z) if win.seen[i, j, k] else 0
        assert win.light[i, j, k] == expected
    assert win.light[11 - x0, 61 - y0, 10 - z0] == 13  # beside the torch, in a dark room
    assert win.classes[12 - x0, 61 - y0, 10 - z0] == TORCH


def test_head_inside_a_block_sees_nothing():
    """A cell with collision (a fallen block) or lava at the eye blinds it: the game draws the
    block's in-wall overlay or lava's fog over the screen [mem]."""
    for cls in (ID["sand"], LAVA):
        w = room()
        w.set(10, FLOOR + 2, 10, cls)  # a fallen block (or lava) at head height
        win = P.voxel_window(w, FEET, "M")
        assert not win.seen.any() and (win.classes == UNSEEN).all() and (win.light == 0).all()
        pano = P.panorama(w, EYE, PANO)
        assert (pano.block == cls).all() and (pano.distance == 0).all() and (pano.face == P.NONE).all()
        assert not P.entity_visible(w, EYE, (12.5, FLOOR + 1.0, 10.5), 1.95)
    t = P.tables(w)
    assert not t.eye_clear[LAVA] and not t.eye_clear[ID["sand"]] and not t.eye_clear[BARRIER]
    assert not t.eye_clear[UNSEEN]


def test_a_sign_or_banner_at_head_height_does_not_blind_the_eye():
    """An opaque class without collision (signs, banners, the portal class) can hold the head while
    the body stands in a free cell. The eye looks out of it: the window and the panorama are those
    of an air cell there, except that the window shows the banner in the eye's own cell."""
    banner, sign, portal = class_of("white_wall_banner"), class_of("oak_sign"), class_of("nether_portal")
    t = P.tables(room())
    for cls in (banner, sign, portal):
        assert not t.see[cls] and C.collision_height[cls] == 0 and t.eye_clear[cls], C.block_class(cls).name
    plain = world_with(fills=[(STONE, (11, 11), (61, 64), (0, 39))])  # a wall 0.5 east of the eye
    hung = plain.copy()
    hung.set(10, FLOOR + 2, 10, banner)  # a wall banner on it, at head height in the agent's cell
    zombie = (8.5, FLOOR + 1.0, 10.5)
    for w in (plain, hung):
        w.mobs.alloc(cls=ZOMBIE, pos=zombie)
    a, b = P.voxel_window(plain, FEET, "M"), P.voxel_window(hung, FEET, "M")
    assert b.seen.sum() > 1000 and np.array_equal(a.seen, b.seen)
    eye_cell = (10 - b.origin[0], FLOOR + 2 - b.origin[1], 10 - b.origin[2])
    assert b.classes[eye_cell] == banner and a.classes[eye_cell] == AIR
    other = np.ones(a.classes.shape, bool)
    other[eye_cell] = False
    assert np.array_equal(a.classes[other], b.classes[other])
    pa, pb = P.panorama(plain, EYE, PANO), P.panorama(hung, EYE, PANO)
    for name in ("block", "entity", "face", "distance"):
        assert np.array_equal(getattr(pa, name), getattr(pb, name)), name  # the eye's cell is never a hit
    assert (pb.entity == ZOMBIE).any() and P.entity_visible(hung, EYE, zombie, 1.95)
    # The wall it hangs on still hides what lies behind it.
    behind = hung.copy()
    behind.set(12, FLOOR + 2, 10, DIAMOND)
    assert P.voxel_window(behind, FEET, "M") == b and P.panorama(behind, EYE, PANO) == pb


def test_corner_ties_are_deterministic():
    """A segment through an edge between two cells steps x first: the x-side cell decides."""
    w = room()
    eye = (10.5, FLOOR + 2.5, 10.5)
    end = (12.5, FLOOR + 2.5, 12.5)  # passes exactly through the edge at (11, 11)
    w.set(10, FLOOR + 2, 11, STONE)  # the z-side cell: never visited
    assert P.line_of_sight(w, eye, end)
    w.set(10, FLOOR + 2, 11, AIR)
    w.set(11, FLOOR + 2, 10, STONE)  # the x-side cell: visited first
    assert not P.line_of_sight(w, eye, end)


# ---------------------------------------------------------------------------
# The panorama: geometry
# ---------------------------------------------------------------------------


def test_panorama_directions():
    d = P.panorama_directions(PANO)
    assert d.shape == (24, 96, 3) and not d.flags.writeable
    assert np.allclose(np.linalg.norm(d, axis=2), 1.0)
    el = np.degrees(np.arcsin(d[:, 0, 1]))
    assert np.allclose(el, 30.0 - (np.arange(24) + 0.5) * 3.75)  # row 0 looks up, row 23 down
    assert (d[:8, :, 1] > 0).all() and (d[8:, :, 1] < 0).all()  # 8 rows above the horizon, 16 below
    az = np.degrees(np.arctan2(d[7, :, 0], -d[7, :, 2])) % 360.0  # clockwise from north (-z)
    assert np.allclose(az, (np.arange(96) + 0.5) * 3.75)
    for c, (dx, dz) in ((0, (0, -1)), (24, (1, 0)), (48, (0, 1)), (72, (-1, 0))):  # north, east, south, west
        assert d[7, c, 0] * dx + d[7, c, 2] * dz > 0.99


def test_open_sky_and_floor(open_world):
    p = pano_at(open_world)
    assert p.block.shape == (24, 96) and p.block.dtype == np.int16
    up = slice(0, 8)
    assert (p.block[up] == AIR).all() and (p.face[up] == P.NONE).all() and (p.distance[up] == 1.0).all()
    assert (p.entity == P.NONE).all() and (p.agent == P.NONE).all()
    assert (p.sky_light[up] == 1.0).all() and (p.block_light == 0).all()  # where the ray ends: open sky
    down = slice(8, 24)
    assert (p.block[down] == STONE).all() and (p.face[down] == P.FACE_WORDS.index("top")).all()
    el = np.radians(30.0 - (np.arange(8, 24) + 0.5) * 3.75)
    expected = (OPEN_EYE[1] - (FLOOR + 1)) / -np.sin(el) / 64.0  # the floor's top plane
    assert np.allclose(p.distance[down, 5], expected, rtol=1e-5)
    assert (p.sky_light[down] == 1.0).all()  # the air in front of the floor's top face


def test_pillar_north_shows_in_the_north_columns(open_world):
    w = open_world.copy()
    for y in range(FLOOR + 1, FLOOR + 21):
        w.set(70, y, 62, OBSIDIAN)  # 8 cells north of the eye's cell; its south face is 7.5 away
    p = pano_at(w)
    shown = p.block == OBSIDIAN
    cols = sorted(set(np.nonzero(shown)[1].tolist()))
    assert cols == [0, 95]  # azimuth 1.875 and 358.125: the two columns either side of north
    assert not shown[:, 40:56].any()  # nothing in the south columns
    rows = sorted(set(np.nonzero(shown[:, 0])[0].tolist()))
    assert rows[0] == 0 and 7 in rows and 8 in rows  # from the top row down past the horizon
    assert p.face[7, 0] == P.FACE_WORDS.index("south")  # a ray travelling north enters a south face
    e = a = math.radians(1.875)
    assert p.distance[7, 0] == pytest.approx(7.5 / (math.cos(e) * math.cos(a)) / 64.0, rel=1e-6)


def test_pillar_east_and_faces_of_walls(open_world):
    w = open_world.copy()
    for y in range(FLOOR + 1, FLOOR + 21):
        w.set(78, y, 70, OBSIDIAN)  # east
        w.set(62, y, 70, GLASS)  # west
        w.set(70, y, 78, ID["dirt"])  # south
    p = pano_at(w)
    assert set(np.nonzero(p.block[7] == OBSIDIAN)[0].tolist()) == {23, 24}
    assert p.face[7, 24] == P.FACE_WORDS.index("west")
    assert set(np.nonzero(p.block[7] == GLASS)[0].tolist()) == {71, 72} and p.face[7, 72] == P.FACE_WORDS.index("east")
    assert set(np.nonzero(p.block[7] == ID["dirt"])[0].tolist()) == {47, 48}
    assert p.face[7, 48] == P.FACE_WORDS.index("north")


def test_block_below_the_eye_shows_in_lower_rows_only(open_world):
    w = open_world.copy()
    w.set(70, FLOOR + 1, 62, OBSIDIAN)  # one block on the floor, 7.5 north: below the eye (1.62 up)
    p = pano_at(w)
    rows = sorted(set(np.nonzero((p.block == OBSIDIAN)[:, 0])[0].tolist()))
    assert rows == [row(-5.625), row(-9.375)] == [9, 10]  # elevations whose ray meets its south face
    assert not (p.block[:9] == OBSIDIAN).any()


def test_ceiling_shows_a_bottom_face():
    w = world_with(fills=[(STONE, (0, 39), (FLOOR + 4, FLOOR + 4), (0, 39))])  # a roof 1.38 above the eye
    p = pano_at(w, FEET)
    assert p.block[0, 0] == STONE and p.face[0, 0] == P.FACE_WORDS.index("bottom")
    assert p.distance[0, 0] == pytest.approx((FLOOR + 4 - EYE[1]) / math.sin(math.radians(28.125)) / 64.0, rel=1e-6)


def test_range_cut_off_at_64(open_world):
    near, far = open_world.copy(), open_world.copy()
    for x in range(66, 75):
        for y in range(FLOOR + 1, FLOOR + 8):
            near.set(x, y, 10, OBSIDIAN)  # south face at z 11: 59.5 north of the eye
            far.set(x, y, 4, OBSIDIAN)  # south face at z 5: 65.5, beyond 64
    pn, pf = pano_at(near), pano_at(far)
    assert pn.block[7, 0] == OBSIDIAN and pn.distance[7, 0] < 1.0
    assert pf.block[7, 0] == AIR and pf.distance[7, 0] == 1.0 and pf.face[7, 0] == P.NONE
    assert not (pf.block == OBSIDIAN).any()


def test_pixel_light_is_the_cell_in_front_of_the_face():
    w = world_with(fills=[(AIR, (4, 16), (61, 64), (4, 16)), (TORCH, (10, 10), (61, 61), (5, 5))], solid_to=70)
    p = pano_at(w, FEET)
    i, j = 7, 0  # looking north at the wall z = 3: the cell in front of it is z = 4, near the torch
    assert p.block[i, j] == STONE and p.face[i, j] == P.FACE_WORDS.index("south")
    x, y, z = (int(v) for v in p.cell[i, j])
    assert z == 3
    front = (x, y, z + 1)
    assert p.block_light[i, j] == pytest.approx(w.block_light_at(*front) / 15.0)
    assert p.block_light[i, j] > 0 and p.sky_light[i, j] == 0.0  # a dark room lit by the torch
    assert (p.block == TORCH).any()  # the torch itself is a hit: a non-air block


def test_water_lava_glass_leaves_are_hits():
    for cls in (WATER, LAVA, GLASS, LEAVES):
        w = world_with(fills=[(cls, (10, 10), (61, 70), (6, 6)), (IRON, (10, 10), (61, 70), (4, 4))])
        p = pano_at(w, FEET)
        assert p.block[7, 0] == cls and p.face[7, 0] == P.FACE_WORDS.index("south")
        assert not (p.block == IRON).any()  # what lies straight behind is not in the panorama


def test_a_ray_that_starts_in_water_sees_the_pool():
    w = world_with(fills=[(WATER, (4, 16), (57, 64), (4, 16))], solid_to=64)
    feet = (10.5, 58.0, 10.5)  # the head (eye 59.62) is under water
    p = P.panorama(w, P.eye_of(feet), PANO)
    assert not (p.block == WATER).any() and (p.distance > 0).all()  # never water at distance 0
    assert p.block[23, 0] == STONE and p.face[23, 0] == P.FACE_WORDS.index("top")  # the pool's floor
    assert p.block[7, 0] == STONE and p.face[7, 0] == P.FACE_WORDS.index("south")  # its north wall
    above = P.panorama(w, P.eye_of((10.5, 65.0, 10.5)), PANO)  # from the air above, water is a hit
    assert above.block[23, 0] == WATER and above.face[23, 0] == P.FACE_WORDS.index("top")


@pytest.mark.parametrize("plant", ["kelp", "seagrass", "tall_seagrass"])
def test_a_head_in_a_plant_under_water_starts_in_water(plant):
    """Kelp and seagrass always stand in water (the data flags them liquid): a head in one is under
    water, so its rays pass the water around it and see the pool and the plants, never water at
    distance 0. A swimming ray that meets a plant cell stops there: the plant is a hit."""
    cls = ID[plant]
    pool = [(WATER, (4, 16), (57, 64), (4, 16))]
    water = world_with(fills=pool, solid_to=64)
    grown = world_with(fills=[*pool, (cls, (10, 10), (57, 64), (10, 10))], solid_to=64)  # a column through the head
    feet = (10.5, 58.0, 10.5)  # eye 59.62, in cell (10, 59, 10)
    eye = P.eye_of(feet)
    assert grown.get(10, 59, 10) == cls and P.tables(grown).water[cls] and not P.tables(grown).open_water[cls]
    p = P.panorama(grown, eye, PANO)
    assert not (p.block == WATER).any() and (p.distance > 0).all()
    assert p.block[23, 0] == cls and p.face[23, 0] == P.FACE_WORDS.index("top")  # the plant cell below the head
    pw = P.panorama(water, eye, PANO)
    away = p.block != cls  # the pixels whose rays leave the plant column: as from open water
    assert away.sum() > 1500 and np.array_equal(p.block[away], pw.block[away])
    assert np.array_equal(p.distance[away], pw.distance[away])
    beside = world_with(fills=[*pool, (cls, (12, 12), (57, 64), (4, 16))], solid_to=64)  # a wall of plants east
    pb = P.panorama(beside, eye, PANO)  # from open water
    assert pb.block[7, 24] == cls and pb.face[7, 24] == P.FACE_WORDS.index("west")
    assert pb.distance[7, 24] == pytest.approx(1.5 / math.cos(math.radians(1.875)) / math.cos(math.radians(1.875))
                                               / 64.0, rel=1e-6)


# ---------------------------------------------------------------------------
# The panorama: entities
# ---------------------------------------------------------------------------


def test_entity_boxes():
    w = room()
    me = w.alloc_agent(FEET)
    other = w.alloc_agent((14.5, 61.0, 10.5))
    dead = w.alloc_agent((6.5, 61.0, 10.5))
    w.agents.alive[dead] = False
    z = w.mobs.alloc(cls=ZOMBIE, pos=(10.5, 61.0, 6.5))
    it = w.items.alloc(item=RAW_IRON, count=1, pos=(12.5, 61.0, 12.5))
    b = P.entity_boxes(w, exclude_agent=me)
    assert len(b) == 3 and b.source.tolist() == [0, 1, 2] and b.slot.tolist() == [other, z, it]
    assert b.cls.tolist() == [PLAYER, ZOMBIE, ITEM] and b.agent.tolist() == [other, -1, -1]
    assert np.allclose(b.lo[0], (14.2, 61.0, 10.2)) and np.allclose(b.hi[0], (14.8, 62.8, 10.8))  # 0.6 x 1.8
    assert np.allclose(b.hi[1] - b.lo[1], (0.6, 1.95, 0.6)) and np.allclose(b.hi[2] - b.lo[2], (0.25, 0.25, 0.25))
    assert len(P.entity_boxes(w, exclude_agent=me, items=False)) == 2
    assert len(P.entity_boxes(w)) == 4  # the observer too when not excluded; never the dead


def test_entities_in_the_panorama(open_world):
    w = open_world.copy()
    me = w.alloc_agent(OPEN_FEET)
    z = w.mobs.alloc(cls=ZOMBIE, pos=(70.5, 61.0, 65.5))  # 5 north: its box's south side is 4.7 away
    other = w.alloc_agent((75.5, 61.0, 70.5))  # 5 east
    # On the floor 4 south, a little east, onto the ray of column 47 (a 0.25 box centred due south
    # would fall between the rays of columns 47 and 48, which pass 0.127 to either side).
    w.items.alloc(item=RAW_IRON, count=1, pos=(70.6, 61.0, 74.5))
    p = P.panorama(w, OPEN_EYE, PANO, exclude_agent=me)
    assert p.entity[7, 0] == ZOMBIE and p.block[7, 0] == P.NONE and p.agent[7, 0] == P.NONE
    assert p.face[7, 0] == P.FACE_WORDS.index("south")
    e = math.radians(1.875)
    assert p.distance[7, 0] == pytest.approx(4.7 / (math.cos(e) * math.cos(e)) / 64.0, rel=1e-6)
    assert p.entity[7, 24] == PLAYER and p.agent[7, 24] == other and p.face[7, 24] == P.FACE_WORDS.index("west")
    shown = p.entity == ITEM
    assert shown.any() and (np.nonzero(shown)[0] >= 8).all()  # a small box on the floor, below the horizon
    assert not (p.agent == me).any()  # never the observer's own box
    assert z == 0


def test_panorama_entities_reach_64_not_beyond(open_world):
    w = open_world.copy()
    # A 0.6-wide box 40 blocks away spans 0.9 degrees, so it is placed on a pixel's ray: the ray of
    # row 8 (1.875 degrees down) and column 0 (1.875 east of north) passes 1.30 blocks east of the
    # eye's line 39.7 blocks north, at y 61.32, before it would reach the floor (49.5 away).
    off = 39.7 * math.tan(math.radians(1.875))
    w.mobs.alloc(cls=ZOMBIE, pos=(70.5 + off, 61.0, 30.5))  # 40 north: beyond the entity table's 32 blocks
    w.mobs.alloc(cls=ZOMBIE, pos=(110.5, 61.0, 70.5 + off))  # 40 east, on the ray of column 24
    w.mobs.alloc(cls=ZOMBIE, pos=(70.5, 61.0, 136.5))  # 66 south: beyond the panorama's 64
    p = pano_at(w)
    assert p.entity[8, 0] == ZOMBIE and p.entity[8, 24] == ZOMBIE
    assert p.distance[8, 0] == pytest.approx(39.7 / math.cos(math.radians(1.875)) ** 2 / 64.0, rel=1e-6)
    assert not (p.entity[:, 40:56] == ZOMBIE).any()
    assert not P.entity_visible(w, OPEN_EYE, (70.5 + off, 61.0, 30.5), 1.95)  # the entity table stops at 32


def test_entity_behind_a_block_or_glass_is_not_a_pixel():
    w = world_with(fills=[(STONE, (12, 12), (61, 62), (10, 10)), (GLASS, (8, 8), (61, 64), (0, 39))])
    behind_stone = (13.5, 61.0, 10.5)
    behind_glass = (6.5, 61.0, 10.5)
    w.mobs.alloc(cls=ZOMBIE, pos=behind_stone)
    w.mobs.alloc(cls=ZOMBIE, pos=behind_glass)
    p = pano_at(w, FEET)
    assert not (p.entity == ZOMBIE).any()  # the stone hides one, the glass is hit before the other
    assert p.block[7, column(270.0)] == GLASS
    assert not P.entity_visible(w, EYE, behind_stone, 1.95)
    assert P.entity_visible(w, EYE, behind_glass, 1.95)  # glass is see-through for the entity table


def test_entity_visible_range_and_walls():
    w = room()
    target = (20.5, FLOOR + 1.0, 10.5)
    assert P.entity_visible(w, EYE, target, 1.95)
    w2 = world_with(fills=[(STONE, (15, 15), (61, 68), (0, 39))])
    assert not P.entity_visible(w2, EYE, target, 1.95)
    assert not P.entity_visible(room(), EYE, (10.5, FLOOR + 1.0, 39.5), 1.95, max_distance=20)
    assert P.ENTITY_RANGE == 32.0


# ---------------------------------------------------------------------------
# Sightings
# ---------------------------------------------------------------------------


def test_sightings_from_the_window():
    w = room()
    for x in (14, 16):
        w.set(x, FLOOR + 1, 10, IRON)
    w.set(10, FLOOR - 1, 12, DIAMOND)  # buried: not sighted
    win = P.voxel_window(w, FEET, "M")
    s = P.sightings(C, EYE, 5, win)
    assert [h.pos for h in s.of(IRON)] == [(14, FLOOR + 1, 10), (16, FLOOR + 1, 10)] and s.counts[IRON] == 2
    assert s.nearest(DIAMOND) is None and s.tick == 5 and not s.xray
    assert s.of(IRON)[0].distance == pytest.approx(float(np.linalg.norm(np.add((14, FLOOR + 1, 10), 0.5) - EYE)))
    assert all((a.distance, a.pos) <= (b.distance, b.pos) for a, b in zip(s.items, s.items[1:]))
    # Every seen resource block is a sighting (DESIGN.md 6.1: it enters memory when it is seen);
    # which places to keep is the notebook's eviction, not perception's.
    seen_stone = int((win.seen & (win.classes == STONE)).sum())
    assert s.counts[STONE] == seen_stone and 400 < seen_stone < 21 * 21  # the ores hide a few floor cells
    assert {h.pos for h in s.of(STONE)} == {tuple(int(v) for v in c) for c in
                                            np.argwhere(win.seen & (win.classes == STONE)) + np.asarray(win.origin)}
    capped = P.sightings(C, EYE, 5, win, per_class=8)  # an optional bound, used by nothing in the design
    assert capped.counts[STONE] == 8 and capped.of(STONE) == s.of(STONE)[:8] and capped.of(IRON) == s.of(IRON)
    assert P.perceive(w, FEET).sightings == P.sightings(C, EYE, w.tick, win, P.panorama(w, EYE, PANO))


def test_sightings_s_window_only_m_window_and_panorama(open_world):
    w = open_world.copy()
    for x in range(66, 75):
        for y in range(FLOOR + 1, FLOOR + 8):
            w.set(x, y, 10, IRON)  # a wall of iron ore 59.5 north
    w.set(70, FLOOR + 1, 66, DIAMOND)  # inside both windows
    s = P.perceive(w, OPEN_FEET, preset="S")
    m = P.perceive(w, OPEN_FEET, preset="M")
    assert s.panorama is None and s.sightings.nearest(IRON) is None and s.sightings.nearest(DIAMOND) is not None
    assert m.panorama is not None and m.sightings.nearest(IRON) is not None
    far = m.sightings.nearest(IRON)
    assert far.pos[2] == 10 and 59 < far.distance < 64.5
    assert len(m.sightings.of(DIAMOND)) == 1  # seen in the window and the panorama: one sighting per cell
    beyond = open_world.copy()
    for x in range(66, 75):
        for y in range(FLOOR + 1, FLOOR + 8):
            beyond.set(x, y, 4, IRON)  # 65.5 north: out of range
    assert P.perceive(beyond, OPEN_FEET, preset="M").sightings.nearest(IRON) is None


def test_sightings_refuse_an_xray_window():
    win = P.voxel_window(room(), FEET, "M", xray=True)
    with pytest.raises(ValueError):
        P.sightings(C, EYE, 0, win)


# ---------------------------------------------------------------------------
# The x-ray ablation
# ---------------------------------------------------------------------------


def test_xray_is_flagged_and_sees_what_is_hidden():
    w = room()
    w.set(12, FLOOR - 2, 12, DIAMOND)  # buried, inside the M window (y 57..65)
    honest = P.perceive(w, FEET)
    xray = P.perceive(w, FEET, xray=True)
    assert not honest.xray and not honest.window.xray and not honest.sightings.xray
    assert xray.xray and xray.window.xray and xray.sightings.xray
    assert honest.sightings.nearest(DIAMOND) is None and honest.window.classes[12, FLOOR - 2 - 57, 12] == UNSEEN
    assert xray.window.classes[12, FLOOR - 2 - 57, 12] == DIAMOND and xray.window.seen.all()
    assert np.array_equal(xray.window.classes, w.region(0, 57, 0, 21, 66, 21))
    assert xray.sightings.nearest(DIAMOND).pos == (12, FLOOR - 2, 12)
    assert xray.sightings.counts[STONE] == P.XRAY_PER_CLASS == 8  # findBlocks' count: not 300,000 stone blocks
    assert xray.panorama == honest.panorama  # the panorama stays honest
    assert P.xray_sightings(w, EYE, per_class=1).nearest(STONE).pos == (10, FLOOR, 10)  # nearest, seen or not


def test_xray_range_is_64():
    w = room()
    w.set(10, -1, 10, DIAMOND)  # centre 63.12 below the eye
    w.set(11, -4, 10, IRON)  # centre 66.1 below
    s = P.xray_sightings(w, EYE, 0, per_class=None)
    assert s.nearest(DIAMOND) is not None and s.nearest(IRON) is None and s.xray and s.rays == 0
    assert P.XRAY_RANGE == 64.0


# ---------------------------------------------------------------------------
# perceive, the cache, the schedule, determinism
# ---------------------------------------------------------------------------


def test_perceive_parts_and_f3():
    w = room()
    slot = w.alloc_agent(FEET, yaw=-90.0)
    w.advance(37)
    p = P.perceive(w, agent=slot)
    assert p.tick == 37 and p.feet == FEET and p.eye == EYE and p.preset == "M"
    assert p.panorama is not None and p.f3 is not None and p.f3.ticks == 37 and p.f3.facing == "east"
    assert p.window == P.voxel_window(w, FEET, "M") and p.panorama == P.panorama(w, EYE, PANO, exclude_agent=slot)
    assert P.perceive(w, FEET, preset="S").panorama is None and P.perceive(w, FEET).f3 is None
    assert P.perceive(w, agent=slot, panorama=False).panorama is None  # in-macro recasts
    assert P.perceive(w, agent=slot, f3=False).f3 is None
    with pytest.raises(ValueError):
        P.perceive(w)  # neither feet nor agent
    with pytest.raises(ValueError):
        P.perceive(w, (11.5, 61.0, 10.5), agent=slot)  # the F3 group and the window must share a viewpoint


def test_window_cache_is_exact():
    big = world_with(80)
    cache = P.WindowCache()
    first = cache.get(big, FEET, "M")
    again = cache.get(big, FEET, "M")
    assert cache.hits == 1 and again == first == P.voxel_window(big, FEET, "M")
    big.set(70, FLOOR + 1, 70, IRON)  # far outside the window and its light margin: still valid
    assert cache.get(big, FEET, "M") == P.voxel_window(big, FEET, "M") and cache.hits == 2
    big.set(25, FLOOR + 1, 10, TORCH)  # outside the window, inside the light margin: recomputed
    assert cache.get(big, FEET, "M") == P.voxel_window(big, FEET, "M") and cache.hits == 2
    big.set(12, FLOOR + 1, 10, IRON)  # inside: recomputed, and the new ore is seen
    win = cache.get(big, FEET, "M")
    assert cache.hits == 2 and win.classes[12, FLOOR + 1 - 57, 10] == IRON
    moved = (11.5, FLOOR + 1.0, 10.5)
    assert cache.get(big, moved, "M") == P.voxel_window(big, moved, "M") and cache.hits == 2
    assert cache.get(big, moved, "S") == P.voxel_window(big, moved, "S") and cache.hits == 2  # another preset
    assert cache.get(big, moved, "S", xray=True).xray and cache.hits == 2
    big.trim_changes(big.change_count)
    p = P.perceive(big, moved, preset="S", xray=True, cache=cache)
    assert cache.hits == 3 and p.window.xray


def test_window_cache_never_takes_one_world_for_another():
    """Resets copy a pristine world. A copy is another world, even when the old one was freed and
    CPython hands its id() to the copy, and even though a copy keeps the generation."""
    import gc

    pristine = room()
    cache = P.WindowCache()
    stale = 0
    for _ in range(40):
        w = pristine.copy()
        stale += cache.get(w, FEET, "M") != P.voxel_window(w, FEET, "M")
        w.set(12, FLOOR + 1, 10, IRON)  # the episode changes the world in view, the cache follows
        assert cache.get(w, FEET, "M").classes[12, FLOOR + 1 - 57, 10] == IRON
        del w
        gc.collect()
    assert stale == 0 and cache.hits == 0
    w = pristine.copy()
    cache.get(w, FEET, "M")
    w.load(w.blocks.copy())  # the same object, a new generation
    w.set(12, FLOOR + 1, 10, IRON)
    assert cache.get(w, FEET, "M").classes[12, FLOOR + 1 - 57, 10] == IRON


def test_ray_schedule_every_20_ticks():
    sched = P.RaySchedule()
    assert sched.period == P.RECAST_PERIOD == 20
    assert [t for t in range(50) if sched.due(t)] == [0, 20, 40]
    assert [t for t in range(50) if P.RaySchedule(phase=3).due(t)] == [3, 23, 43]
    assert [t for t in range(50) if P.RaySchedule(phase=23).due(t)] == [3, 23, 43]


def test_deterministic():
    a = wg.generate(4, wg.preset("forest", size_x=40, size_z=40))
    b = wg.generate(4, wg.preset("forest", size_x=40, size_z=40))
    feet = a.meta["spawn"]
    for w in (a, b):
        w.alloc_agent(feet, yaw=33.0, pitch=12.0)
        w.mobs.alloc(cls=ZOMBIE, pos=(feet[0] + 3.0, feet[1], feet[2] - 2.0))
    pa, pb = P.perceive(a, agent=0), P.perceive(b, agent=0)
    assert pa == pb and P.perceive(a, agent=0) == pa and P.perceive(a.copy(), agent=0) == pa
    assert pa.panorama.steps == pb.panorama.steps and pa.window.rays == pb.window.rays


# ---------------------------------------------------------------------------
# Honesty: adversarial scenes
# ---------------------------------------------------------------------------


CAVE_VIEW = (11.5, 61.0, 10.5)  # in the cave scene's room, 1.5 west of its east wall (x 13)


def _cave_behind_wall():
    """A dark room with a torch; one stone block east of it (x 13), a sealed cave (x 14-20)."""
    w = world_with(fills=[(AIR, (4, 12), (61, 64), (4, 16)), (AIR, (14, 20), (61, 64), (4, 16)),
                          (TORCH, (5, 5), (61, 61), (5, 5))], solid_to=75)
    slot = w.alloc_agent(CAVE_VIEW, yaw=-90.0)  # facing the wall, 1.5 away

    def hide(w: World) -> None:
        for y in range(61, 65):
            for z in range(4, 17):
                w.set(21, y, z, IRON)  # the cave's far wall
        for x in range(14, 21):
            for z in range(4, 17, 3):
                w.set(x, 60, z, DIAMOND)  # its floor
                w.set(x, 65, z, ID["coal_ore"])  # its ceiling
        w.set(18, 61, 14, LAVA)  # a light source in the sealed cave
        w.mobs.alloc(cls=ZOMBIE, pos=(14.5, 61.0, 10.5))  # just behind the wall: its box 2.7 away, within reach

    def reveal(w: World) -> None:
        w.set(13, 62, 10, AIR)  # a hole in the wall

    return w, slot, hide, reveal, [(14.5, 61.0, 10.5)]


def _ores_under_floor():
    w = room()
    slot = w.alloc_agent(FEET, yaw=10.0, pitch=70.0)  # looking down at the floor

    def hide(w: World) -> None:
        for x in range(6, 15):
            for z in range(6, 15):
                w.set(x, FLOOR - 1, z, SWAPS[(x + z) % len(SWAPS)])
                w.set(x, FLOOR - 2, z, DIAMOND)
        w.set(10, 40, 10, ID["emerald_ore"])

    def reveal(w: World) -> None:
        w.set(10, FLOOR, 11, AIR)  # a hole beside the feet: the eye sees down onto (10, 59, 11)

    return w, slot, hide, reveal, []


def _entity_behind_block():
    w = world_with(fills=[(STONE, (12, 12), (61, 62), (10, 10))])
    slot = w.alloc_agent(FEET, yaw=-90.0)  # the pillar is 1.5 east

    def hide(w: World) -> None:
        w.mobs.alloc(cls=ZOMBIE, pos=(13.5, 61.0, 10.5))  # 2.7 away: within the entity reach
        w.items.alloc(item=RAW_IRON, count=1, pos=(13.5, 61.0, 10.5))

    def reveal(w: World) -> None:
        w.set(12, 62, 10, AIR)
        w.set(12, 61, 10, AIR)

    return w, slot, hide, reveal, [(13.5, 61.0, 10.5)]


def _lava_wall():
    w = world_with(fills=[(LAVA, (12, 12), (61, 64), (0, 39))])
    slot = w.alloc_agent(FEET, yaw=-90.0)

    def hide(w: World) -> None:
        for x in range(13, 17):
            for z in range(8, 13):
                w.set(x, FLOOR, z, SWAPS[(x * 3 + z) % len(SWAPS)])
        w.mobs.alloc(cls=ZOMBIE, pos=(13.5, 61.0, 10.5))

    def reveal(w: World) -> None:
        for y in range(61, 65):
            w.set(12, y, 10, AIR)

    return w, slot, hide, reveal, [(13.5, 61.0, 10.5)]


def _see_through_then_stone(cls: int):
    def scene():
        w = world_with(fills=[(cls, (12, 12), (61, 64), (0, 39)), (STONE, (14, 14), (61, 64), (0, 39))])
        slot = w.alloc_agent(FEET, yaw=-90.0)

        def hide(w: World) -> None:
            for x in range(15, 19):
                for z in range(8, 13):
                    w.set(x, FLOOR, z, SWAPS[(x + 2 * z) % len(SWAPS)])
            w.mobs.alloc(cls=ZOMBIE, pos=(15.5, 61.0, 10.5))

        def reveal(w: World) -> None:
            w.set(14, 61, 10, AIR)  # a hole in the stone wall behind the see-through one
            w.set(14, 62, 10, AIR)

        return w, slot, hide, reveal, [(15.5, 61.0, 10.5)]

    return scene


def _sealed_room_beside():
    w = world_with(fills=[(STONE, (11, 17), (61, 67), (7, 13)), (AIR, (13, 15), (63, 65), (9, 11))])
    slot = w.alloc_agent((9.5, 61.0, 10.5), yaw=-90.0)  # its west wall is 1.5 east

    def hide(w: World) -> None:
        for x in range(12, 17):
            for y in range(62, 67):
                for z in range(8, 13):
                    if w.get(x, y, z) == STONE:
                        w.set(x, y, z, SWAPS[(x + y + z) % len(SWAPS)])  # the inner shell
        w.set(13, 63, 9, LAVA)  # light that cannot leave the room
        w.mobs.alloc(cls=ZOMBIE, pos=(14.5, 63.0, 10.5))

    def reveal(w: World) -> None:
        for x in (11, 12):  # a tunnel two high into the room
            for y in (62, 63):
                w.set(x, y, 10, AIR)

    return w, slot, hide, reveal, [(14.5, 63.0, 10.5)]


ROOM_VIEW = (9.5, 61.0, 10.5)  # 2.5 west of the room's west wall (x 12), looking east


def room_behind_a_wall_of(cls: int) -> World:
    """A sealed stone room (air x 13-15, y 61-64, z 7-13) whose west wall (x 12) is made of ``cls``,
    with an iron ore on its floor at (14, 61, 10)."""
    return world_with(fills=[(STONE, (12, 16), (61, 65), (6, 14)), (AIR, (13, 15), (61, 64), (7, 13)),
                             (cls, (12, 12), (61, 64), (7, 13)), (IRON, (14, 14), (61, 61), (10, 10))])


def _behind_a_wall_of(member: str):
    """A room behind a wall of a block the data flags transparent but that hides what is behind it."""

    def scene():
        w = room_behind_a_wall_of(class_of(member))
        slot = w.alloc_agent(ROOM_VIEW, yaw=-90.0)

        def hide(w: World) -> None:
            w.set(14, 61, 10, DIAMOND)
            for x in range(13, 16):
                for z in range(7, 14):
                    w.set(x, 60, z, SWAPS[(x + z) % len(SWAPS)])  # the room's floor
            w.mobs.alloc(cls=ZOMBIE, pos=(14.5, 61.0, 11.5))

        def reveal(w: World) -> None:
            w.set(12, 61, 10, AIR)  # a hole in the wall
            w.set(12, 62, 10, AIR)

        return w, slot, hide, reveal, [(14.5, 61.0, 11.5)]

    return scene


def _banner_at_head_height():
    """A wall banner in the agent's own cell at head height, hung on a one-block wall (x 11) with a
    sealed cave behind it (x 12-16): the eye looks out of the banner's cell, not through the wall.
    F3 names the banner itself at distance 0 (the head inside a block, debug_info), so here the
    window, the panorama and entity_visible are what can change."""
    w = world_with(fills=[(AIR, (4, 10), (61, 64), (4, 16)), (AIR, (12, 16), (61, 64), (4, 16)),
                          (class_of("white_wall_banner"), (10, 10), (62, 62), (10, 10))], solid_to=75)
    slot = w.alloc_agent(FEET, yaw=-90.0)  # eye in the banner's cell, facing the wall 0.5 away

    def hide(w: World) -> None:
        for y in range(61, 65):
            for z in range(4, 17):
                w.set(17, y, z, IRON)  # the cave's far wall
        for x in range(12, 17):
            w.set(x, 60, 10, DIAMOND)  # its floor
        w.mobs.alloc(cls=ZOMBIE, pos=(12.5, 61.0, 10.5))  # right behind the wall

    def reveal(w: World) -> None:
        w.set(11, 62, 10, AIR)  # a hole in the wall behind the banner

    return w, slot, hide, reveal, [(12.5, 61.0, 10.5)]


SCENES = {
    "cave behind a 1-block wall": _cave_behind_wall,
    "cave behind the wall a banner at head height hangs on": _banner_at_head_height,
    "ores under the floor": _ores_under_floor,
    "entity behind a block": _entity_behind_block,
    "lava wall": _lava_wall,
    "glass": _see_through_then_stone(GLASS),
    "leaves": _see_through_then_stone(LEAVES),
    "sealed room beside the agent": _sealed_room_beside,
    "behind closed doors": _behind_a_wall_of("oak_door"),
    "behind iron doors (merged with iron bars)": _behind_a_wall_of("iron_door"),
    "behind beds": _behind_a_wall_of("white_bed"),
    "behind shulker boxes": _behind_a_wall_of("shulker_box"),
}


def test_every_sight_blocking_class_hides_what_is_behind_it():
    """For every class the data flags transparent that perception makes opaque, a wall of it hides
    the room behind: the ore is neither in the window nor sighted, and the zombie is not visible.
    The positive control: a wall of glass shows both."""
    blocking = [c.id for c in C.block_classes if C.transparent[c.id] and not P.see_through_table(C)[c.id]
                and c.id not in (LAVA, BARRIER)]
    assert len(blocking) >= 20
    eye = P.eye_of(ROOM_VIEW)
    zombie = (14.5, 61.0, 11.5)
    for cls in [*blocking, GLASS]:
        w = room_behind_a_wall_of(cls)
        w.mobs.alloc(cls=ZOMBIE, pos=zombie)
        shown = []
        for preset in ("S", "M"):
            win = P.voxel_window(w, ROOM_VIEW, preset)
            x0, y0, z0 = win.origin
            assert win.classes[12 - x0, 61 - y0, 10 - z0] == cls  # the wall itself is seen
            shown.append(win.classes[14 - x0, 61 - y0, 10 - z0] == IRON)
            shown.append(P.sightings(C, eye, 0, win).nearest(IRON) is not None)
        shown.append(P.entity_visible(w, eye, zombie, 1.95))
        assert all(shown) if cls == GLASS else not any(shown), C.block_class(cls).name


@pytest.mark.parametrize("name", list(SCENES))
def test_honesty_scene(name):
    w, slot, hide, reveal, hidden_entities = SCENES[name]()
    before = {preset: P.perceive(w, agent=slot, preset=preset) for preset in ("S", "M")}
    changed = w.copy()  # World.copy copies the entity tables too
    hide(changed)
    assert not np.array_equal(changed.blocks, w.blocks) or len(changed.mobs) > len(w.mobs)
    for preset in ("S", "M"):
        after = P.perceive(changed, agent=slot, preset=preset)
        assert after == before[preset], preset  # window, panorama, sightings and the F3 group
    eye = P.eye_of(tuple(float(v) for v in w.agents.pos[slot]))
    for pos in hidden_entities:
        assert not P.entity_visible(changed, eye, pos, 1.95)
    if not np.array_equal(changed.blocks, w.blocks):  # the change was real: x-ray sees it
        xa, xb = P.perceive(changed, agent=slot, xray=True), P.perceive(w, agent=slot, xray=True)
        assert xa.window != xb.window or xa.sightings != xb.sightings
    # Positive control: open a way for the eye in both worlds; now the hidden changes show.
    reveal(w)
    reveal(changed)
    assert P.perceive(changed, agent=slot) != P.perceive(w, agent=slot)


def test_f3_is_unchanged_by_a_zombie_or_lava_behind_the_wall():
    w, slot, *_ = _cave_behind_wall()
    base = P.perceive(w, agent=slot).f3
    assert base.block == STONE and base.block_face == "west" and base.entity is None and base.fluid is None
    zombie = w.copy()
    zombie.mobs = w.mobs.copy()
    zombie.mobs.alloc(cls=ZOMBIE, pos=(14.5, 61.0, 10.5))  # 2.7 from the eye, within reach: behind the wall
    zombie.mobs.alloc(cls=ZOMBIE, pos=(13.5, 61.0, 10.5))  # inside the wall's cell, 1.7 away
    assert P.perceive(zombie, agent=slot).f3 == base
    lava = w.copy()
    lava.set(14, 62, 10, LAVA)  # right behind the wall in the look direction, sealed in the cave
    assert P.perceive(lava, agent=slot).f3 == base
    for world in (zombie, lava):  # positive control: without the wall block both are named
        world.set(13, 61, 10, AIR)
        world.set(13, 62, 10, AIR)
    assert P.perceive(zombie, agent=slot).f3.entity == ZOMBIE
    opened = P.perceive(lava, agent=slot).f3
    assert opened.fluid == "lava" and opened.block is None


def test_only_the_wall_keeps_f3_from_naming_the_hidden_zombie():
    """The cave scene's hidden zombie is within the entity reach in the look direction, so the
    scene's F3 check can fail: once the hole is open, F3 names the zombie."""
    w, slot, hide, reveal, _ = _cave_behind_wall()
    hide(w)
    assert P.perceive(w, agent=slot).f3.entity is None
    reveal(w)
    g = P.perceive(w, agent=slot).f3
    assert g.entity == ZOMBIE and g.entity_distance < D.ENTITY_REACH


# ---------------------------------------------------------------------------
# Honesty: generated cave worlds
# ---------------------------------------------------------------------------


def _sweep(see: np.ndarray, reach: np.ndarray, axis: int) -> np.ndarray:
    """Spread ``reach`` along runs of see-through cells on one axis."""
    s = np.moveaxis(see, axis, -1)
    r = np.moveaxis(reach, axis, -1)
    length = s.shape[-1]
    run = np.cumsum(~s, axis=-1)  # a new run id after every blocking cell
    line = np.arange(int(np.prod(s.shape[:-1]))).reshape(s.shape[:-1])[..., None] * (length + 1)
    ids = run + line
    hit = np.zeros(int(ids.max()) + 1, bool)
    hit[ids[r]] = True
    return np.moveaxis(s & hit[ids], -1, axis)


def eye_component(see: np.ndarray, start: tuple[int, int, int]) -> np.ndarray:
    """The 6-connected component of see-through cells containing ``start``: every cell a straight
    segment from an eye there can cross. Every cell such a segment reaches is in it or beside it."""
    reach = np.zeros(see.shape, bool)
    if not see[start]:
        return reach
    reach[start] = True
    count = 1
    while True:
        for axis in range(3):
            reach = _sweep(see, reach, axis)
        n = int(reach.sum())
        if n == count:
            return reach
        count = n


def beside(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


class Sight:
    """What an eye can possibly reach in a world: the eye's see-through component (within 70
    blocks vertically, the whole world sideways) and the cells beside it."""

    def __init__(self, w: World, eye):
        ec = np.floor(np.asarray(eye)).astype(np.int64)
        self.lo = np.array([-1, max(MIN_Y - 1, int(ec[1]) - 70), -1])
        hi = np.array([w.size_x + 1, min(MAX_Y + 2, int(ec[1]) + 71), w.size_z + 1])
        self.region = w.region(*(int(v) for v in self.lo), *(int(v) for v in hi))
        see = SEE[self.region]
        self.comp = eye_component(see, tuple(int(v) for v in ec - self.lo))
        self.near = beside(self.comp)
        inside = np.zeros(see.shape, bool)
        inside[1:-1, max(0, MIN_Y - int(self.lo[1])):, 1:-1] = True
        inside[:, max(0, MAX_Y + 1 - int(self.lo[1])):, :] = False
        light_safe = (FILTER[self.region] == 15) & (C.light[self.region] == 0)
        self.hidden = ~see & ~self.near & light_safe & inside  # cells no segment from the eye reaches
        self.sealed = see & ~self.comp & inside & (self.region == AIR)  # air the eye cannot reach

    def cells(self, mask: np.ndarray) -> np.ndarray:
        return np.argwhere(mask) + self.lo[None, :]

    def _lookup(self, mask: np.ndarray, cells: np.ndarray) -> np.ndarray:
        local = np.asarray(cells).reshape(-1, 3) - self.lo[None, :]
        ok = np.all((local >= 0) & (local < np.array(mask.shape)[None, :]), axis=1)
        out = np.zeros(len(local), bool)
        out[ok] = mask[local[ok, 0], local[ok, 1], local[ok, 2]]
        return out

    def is_near(self, cells: np.ndarray) -> np.ndarray:
        return self._lookup(self.near, cells)

    def in_comp(self, cells: np.ndarray) -> np.ndarray:
        return self._lookup(self.comp, cells)


@pytest.fixture(scope="module")
def cave_world() -> World:
    return wg.generate(3, wg.preset("forest", size_x=48, size_z=48, caves_per_chunk=2.0))


def cave_feet(w: World) -> tuple[float, float, float]:
    """A feet spot inside a carved cave: air with a solid floor and air at head height, at least 5
    below the surface; the one nearest the world's centre (then lowest x, y, z)."""
    air = w.blocks == AIR
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


@pytest.mark.parametrize("where", ["surface", "cave"])
def test_seen_cells_and_pixels_lie_within_sight(cave_world, where):
    """Structural honesty: every seen window cell, every panorama hit and every cell whose light a
    pixel shows is reachable from the eye through see-through cells."""
    feet = cave_world.meta["spawn"] if where == "surface" else cave_feet(cave_world)
    eye = P.eye_of(feet)
    sight = Sight(cave_world, eye)
    win = P.voxel_window(cave_world, feet, "M")
    seen = np.argwhere(win.seen) + np.asarray(win.origin)[None, :]
    assert len(seen) > 50 and sight.is_near(seen).all()
    p = P.panorama(cave_world, eye, PANO)
    hit = p.block != AIR
    assert hit.any() and sight.is_near(p.cell[hit].reshape(-1, 3)).all()
    ec = np.floor(np.asarray(eye)).astype(np.int64)
    fronts = []
    for i, j in np.argwhere(hit):
        c = p.cell[i, j].astype(np.int64)
        n = P.FACE_NORMALS[p.face[i, j]] if p.face[i, j] >= 0 else np.zeros(3, np.int64)
        fronts.append(c + n)
    fronts = np.asarray(fronts)
    assert (sight.in_comp(fronts) | np.all(fronts == ec[None, :], axis=1)).all()


@pytest.mark.parametrize("where", ["surface", "cave"])
def test_hidden_changes_leave_the_observation_unchanged(cave_world, where):
    feet = cave_world.meta["spawn"] if where == "surface" else cave_feet(cave_world)
    w = cave_world.copy()
    slot = w.alloc_agent(feet, yaw=-135.0, pitch=25.0)
    if where == "cave":
        assert feet[1] < w.meta["surface"][int(feet[0])][int(feet[2])] - 4 and w.get(*map(math.floor, feet)) == AIR
    before = {preset: P.perceive(w, agent=slot, preset=preset) for preset in ("S", "M")}
    if where == "cave":
        assert (before["M"].window.classes == UNSEEN).any() and before["M"].sightings.items
    sight = Sight(w, P.eye_of(feet))
    cells = sight.cells(sight.hidden)
    assert len(cells) > 10_000
    win = before["M"].window
    in_window = np.all((cells >= win.origin) & (cells < np.add(win.origin, win.classes.shape)), axis=1)
    assert in_window[::5].sum() > 100  # many of the changed cells lie inside the window
    xray_before = P.perceive(w, agent=slot, xray=True)
    rng = CounterRNG(8)
    for i, (x, y, z) in enumerate(cells[::5].tolist()):
        old = w.get(x, y, z)
        new = SWAPS[rng.below((0, 0, Purpose.GENERIC, 0, i), len(SWAPS))]
        w.set(x, y, z, new if new != old else SWAPS[(SWAPS.index(new) + 1) % len(SWAPS)])
    sealed = sight.cells(sight.sealed)
    sealed = sealed[(sealed[:, 1] < MAX_Y - 2)]
    placed = 0
    for x, y, z in sealed[:: max(1, len(sealed) // 20)].tolist():  # zombies in caves the eye cannot reach
        if sight.in_comp(np.array([[x, y + 1, z]]))[0] or placed >= 20:
            continue
        w.mobs.alloc(cls=ZOMBIE, pos=(x + 0.5, float(y), z + 0.5))
        placed += 1
    assert placed > 0
    for preset in ("S", "M"):
        assert P.perceive(w, agent=slot, preset=preset) == before[preset], preset
    xray_after = P.perceive(w, agent=slot, xray=True)
    assert xray_after.window != xray_before.window and xray_after.sightings != xray_before.sightings


def test_honest_sightings_ignore_hidden_ores(cave_world):
    """The sightings of every pixel and seen cell, without the per-class cap, are unchanged."""
    feet = cave_feet(cave_world)
    eye = P.eye_of(feet)
    w = cave_world.copy()

    def all_sightings(world):
        return P.sightings(C, eye, 0, P.voxel_window(world, feet, "M"), P.panorama(world, eye, PANO), per_class=None)

    base = all_sightings(w)
    sight = Sight(w, eye)
    for x, y, z in sight.cells(sight.hidden)[::3].tolist():
        w.set(x, y, z, DIAMOND)
    assert all_sightings(w) == base
    assert P.xray_sightings(w, eye, per_class=None) != P.xray_sightings(cave_world, eye, per_class=None)

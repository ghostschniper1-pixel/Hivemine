"""Tests for hivemind.sim.ref.nav: the port of mineflayer-pathfinder's rules, goals, failure codes
and T_goto. ``test_same_as_mineflayer_pathfinder`` runs the installed package (bridge/node_modules)
on the same scenes and requires identical results; it is skipped where Node is missing."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hivemind.rules import load_concepts
from hivemind.sim.ref import nav as N
from hivemind.sim.ref import worldgen as wg
from hivemind.sim.ref.mechanics import break_ticks
from hivemind.sim.ref.world import MAX_Y, MIN_Y, World

C = load_concepts()
ID = {c.name: c.id for c in C.block_classes}
AIR, STONE, BEDROCK = 0, ID["stone"], ID["bedrock"]
WOOD_PICK = C.item_class("wooden_pickaxe").id
STONE_PICK = C.item_class("stone_pickaxe").id
AXE = C.item_class("iron_axe").id
FLOOR = 60  # the floor's top block; feet stand at y 61
Y = FLOOR + 1
REPO = Path(__file__).resolve().parents[1]
NODE_MODULES = REPO / "bridge" / "node_modules"


def arena(sx: int = 32, sz: int = 32, floor: int = BEDROCK) -> World:
    w = World(sx, sz)
    b = w.blocks.copy()
    b[:, : FLOOR - MIN_Y + 1, :] = floor
    w.load(b)
    return w


def fill(w: World, lo, hi, cls: int) -> None:
    """Fill the inclusive box lo..hi (world coordinates)."""
    b = w.blocks.copy()
    b[lo[0]:hi[0] + 1, lo[1] - MIN_Y:hi[1] - MIN_Y + 1, lo[2]:hi[2] + 1] = cls
    w.load(b)


def corridor(length: int = 20, open_height: int = 2) -> World:
    """A 1-wide corridor along x at z = 2, ``open_height`` cells high, in solid bedrock."""
    w = arena(length, 5)
    fill(w, (0, Y, 0), (length - 1, Y + 5, 4), BEDROCK)
    fill(w, (0, Y, 2), (length - 1, Y + open_height - 1, 2), AIR)
    return w


def feet(x: int, y: int, z: int) -> tuple[float, float, float]:
    return (x + 0.5, float(y), z + 0.5)


def passable_path(w: World, path: N.Path) -> bool:
    """Every cell of the path holds a body without digging (after the path's own digs)."""
    grid = N.NavGrid(w)
    dug = {d.pos for d in path.digs}
    return all((grid.passable(*c) or c in dug) and (grid.passable(c[0], c[1] + 1, c[2]) or (c[0], c[1] + 1, c[2]) in dug)
               for c in path.cells[1:])


# -- the pathfinder's view of blocks ------------------------------------------------------------


def test_block_view_of_common_blocks():
    v = N.block_view(C, N.never_break_classes(C), N.avoid_classes(C))

    def of(name: str) -> dict[str, bool]:
        i = ID[name]
        return {k: getattr(v, k)[i] for k in ("safe", "physical", "liquid", "climbable", "replaceable", "falls",
                                              "cant_break")}

    assert of("air") == {"safe": True, "physical": False, "liquid": False, "climbable": False, "replaceable": True,
                         "falls": False, "cant_break": False}
    assert of("water")["safe"] and of("water")["liquid"] and of("water")["replaceable"]
    assert not of("lava")["safe"] and of("lava")["liquid"]  # avoided
    assert of("ladder")["climbable"] and of("ladder")["safe"] and of("ladder")["physical"]
    assert of("vine")["safe"] and not of("vine")["climbable"]  # the package leaves vines out
    assert of("snow")["safe"] and of("short_grass")["safe"] and of("other_carpet")["safe"]
    assert not of("fence")["physical"] and not of("fence")["safe"]  # taller than 1: neither
    for gate in ("other_fence_gate", "other_open_fence_gate"):  # a gate's block has a bounding box
        assert not of(gate)["safe"] and not of(gate)["physical"] and of(gate)["cant_break"]
    assert of("sand")["falls"] and of("gravel")["falls"] and not of("red_sand")["falls"]
    assert of("bedrock")["cant_break"] and of("chest")["cant_break"] and of("barrier")["cant_break"]
    for name in ("sweet_berry_bush", "powder_snow", "wither_rose", "cobweb", "fire"):
        assert not of(name)["safe"] and not of(name)["cant_break"]  # avoided, so dug when in the way
    assert of("magma_block")["physical"]  # the package cannot avoid standing on it
    assert v.height[ID["stone"]] == 1.0 and v.height[ID["air"]] == 0.0 and v.height[ID["fence"]] == 1.5
    assert [C.block_class(i).name for i in range(256) if C.block_defined[i] and not v.uniform[i]] \
        == ["other_other_block_7"]


def test_bridge_block_lists():
    lists = N.bridge_block_lists(C)
    cant, avoid = set(lists["blocksCantBreak"]), set(lists["blocksToAvoid"])
    assert {"chest", "crafting_table", "furnace", "red_bed", "oak_door", "iron_door", "copper_door",
            "oak_fence_gate", "warped_fence_gate"} <= cant
    assert {"lava", "fire", "soul_fire", "cobweb", "magma_block", "cactus", "powder_snow", "sweet_berry_bush",
            "wither_rose", "campfire", "soul_campfire"} <= avoid
    assert lists["blocksCantBreak"] == sorted(cant) and "stone" not in cant and "water" not in avoid


def test_dig_ticks_use_exact_hardness():
    table = N.dig_ticks_table(C, (0,))
    assert table[ID["oak_leaves"]] == break_ticks(0.2, 1.0, True).design_ticks == 6  # not 7 (float32 0.2)
    assert table[ID["stone"]] == 150 and table[ID["bedrock"]] == -1  # DESIGN.md 4.3: stone by hand 150


# -- walking ------------------------------------------------------------------------------------


def test_walk_straight_and_t_goto():
    w = arena()
    res = N.find_path(w, feet(2, Y, 5), N.GoalBlock(12, Y, 5))
    assert res.ok and res.code == N.OK
    p = res.path
    assert p.end == (12, Y, 5) and len(p.steps) == 10 and all(s.move == N.Move.WALK for s in p.steps)
    assert p.length == 10 and p.turns == 0 and p.jumps == 0 and p.dig_ticks == 0 and p.pillars == 0
    assert p.cost == 10.0 and res.nodes == 10
    m = p.model
    assert N.WALK_SPEED == 4.317
    expected = m.search_ticks(p.nodes) + 10 / 4.317 * 20 + m.epsilon
    assert math.isclose(p.t_goto, expected)
    assert math.isclose(sum(p.step_ticks()), p.t_goto - p.t_search - m.epsilon)


def test_walks_around_an_unbreakable_wall():
    w = arena()
    fill(w, (10, Y, 0), (10, Y + 3, 25), BEDROCK)  # a wall with a gap at z 26..31
    res = N.find_path(w, feet(5, Y, 5), N.GoalBlock(15, Y, 5))
    assert res.ok
    p = res.path
    assert max(c[2] for c in p.cells) >= 26  # through the gap
    assert p.length > 30 and p.turns >= 2
    assert passable_path(w, p) and p.dig_ticks == 0


def test_diagonal_needs_one_clear_side():
    """getMoveDiagonal: one side column clear is enough; both blocked and unbreakable: no diagonal."""
    w = arena(12, 12)
    fill(w, (6, Y, 5), (6, Y + 1, 5), BEDROCK)  # the +x side of (5,5) -> (6,6)
    res = N.find_path(w, feet(5, Y, 5), N.GoalBlock(6, Y, 6))
    assert res.ok and [s.move for s in res.path.steps] == [N.Move.DIAGONAL] and res.path.cost == math.sqrt(2)
    fill(w, (5, Y, 6), (5, Y + 1, 6), BEDROCK)  # and the +z side
    res = N.find_path(w, feet(5, Y, 5), N.GoalBlock(6, Y, 6))
    assert res.ok and all(s.move != N.Move.DIAGONAL or (a[0] == s.pos[0] or a[2] == s.pos[2])
                          for a, s in zip(res.path.cells, res.path.steps))
    assert res.path.cells[1] not in ((6, Y, 6),)


# -- step-ups and drops -----------------------------------------------------------------------


def test_step_up_one_block_but_not_two():
    w = corridor(open_height=3)
    fill(w, (8, Y, 2), (19, Y, 2), BEDROCK)  # a 1-block step up at x 8
    res = N.find_path(w, feet(2, Y, 2), N.GoalBlock(12, Y + 1, 2), allow_dig=False)
    assert res.ok and [s.move for s in res.path.steps].count(N.Move.ASCEND) == 1 and res.path.jumps == 1
    w2 = corridor(open_height=3)
    fill(w2, (8, Y, 2), (19, Y + 1, 2), BEDROCK)  # a 2-block step, with headroom above it
    fill(w2, (8, Y + 2, 2), (19, Y + 4, 2), AIR)
    assert N.find_path(w2, feet(2, Y, 2), N.GoalBlock(12, Y + 2, 2), allow_dig=False).code == N.NO_PATH
    w3 = corridor()  # a 1-block step without headroom over the start: no jump
    fill(w3, (8, Y, 2), (19, Y, 2), BEDROCK)
    assert N.find_path(w3, feet(2, Y, 2), N.GoalBlock(12, Y + 1, 2), allow_dig=False).code == N.NO_PATH


@pytest.mark.parametrize("height,ok", [(1, True), (3, True), (4, False)])
def test_drop_lands_at_most_three_below(height, ok):
    """maxDropDown = 4 counts to the landing block, so the feet fall at most 3 (getLandingBlock)."""
    assert N.MAX_DROP_DOWN == 4 and N.MAX_SAFE_FALL == 3
    w = arena(9, 9)
    top = FLOOR + height  # a 1x1 bedrock tower; the agent stands on it
    fill(w, (4, Y, 4), (4, top, 4), BEDROCK)
    res = N.find_path(w, feet(4, top + 1, 4), N.GoalBlock(1, Y, 1))
    assert res.ok == ok
    if ok:
        falls = [s.fall for s in res.path.steps if s.fall]
        assert falls == [height]


def test_drop_into_water_at_any_depth():
    w = arena(9, 9)
    fill(w, (0, FLOOR - 9, 0), (8, FLOOR, 8), ID["water"])  # a 10-deep pool around a tall tower
    fill(w, (4, FLOOR - 9, 4), (4, FLOOR + 8, 4), BEDROCK)
    fill(w, (0, FLOOR - 9, 0), (0, Y, 0), BEDROCK)  # a pillar to climb out on, 1 above the water
    res = N.find_path(w, feet(4, FLOOR + 9, 4), N.GoalBlock(0, Y + 1, 0))
    assert res.ok and res.path.steps[0].move == N.Move.DROP and res.path.steps[0].fall == 9
    assert res.path.steps[0].pos[1] == FLOOR and res.path.blocks_used == 0


# -- water and lava --------------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 3])
def test_wades_and_swims_across_water_without_blocks(depth):
    w = arena(25, 5)
    fill(w, (0, Y, 0), (24, 62, 4), BEDROCK)  # banks with the top block at y 62
    fill(w, (10, 63 - depth, 0), (14, 62, 4), ID["water"])
    if depth > 1:
        fill(w, (15, 62, 0), (15, 62, 4), AIR)  # a step on the far bank
    res = N.find_path(w, feet(5, 63, 2), N.GoalBlock(20, 63, 2))
    assert res.ok and res.path.blocks_used == 0 and res.path.dig_ticks == 0
    in_water = [c for c in res.path.cells if w.get(*c) == ID["water"]]
    assert in_water and all(c[1] == 62 for c in in_water)  # swims at the surface level


def test_never_swims_up():
    """getMoveUp refuses liquid feet cells: a water column cannot be climbed."""
    w = arena(9, 9)
    fill(w, (0, Y, 0), (8, Y + 4, 8), BEDROCK)
    fill(w, (4, Y, 4), (4, Y + 4, 4), ID["water"])  # a 1x1 shaft of water
    assert N.find_path(w, feet(4, Y, 4), N.GoalBlock(1, Y + 5, 1), blocks=10).code == N.NO_PATH


def test_lava_is_never_entered_or_crossed():
    w = arena()
    fill(w, (10, FLOOR, 0), (12, FLOOR, 31), ID["lava"])  # a lava channel across the whole floor
    assert N.find_path(w, feet(5, Y, 5), N.GoalBlock(16, Y, 5)).code == N.NO_PATH
    res = N.find_path(w, feet(5, Y, 5), N.GoalBlock(16, Y, 5), blocks=3)  # bridging over it works
    assert res.ok and all(w.get(*c) != ID["lava"] for c in res.path.cells)


def test_lava_pool_is_avoided_when_a_way_around_exists():
    w = arena()
    fill(w, (10, FLOOR, 0), (12, FLOOR, 20), ID["lava"])  # lava in the floor
    res = N.find_path(w, feet(5, Y, 5), N.GoalBlock(16, Y, 5))
    assert res.ok
    assert all(w.get(c[0], c[1] - 1, c[2]) != ID["lava"] for c in res.path.cells)
    assert max(c[2] for c in res.path.cells) >= 21


# -- digging ---------------------------------------------------------------------------------------


def test_dig_through_a_one_block_wall_with_exact_ticks():
    w = corridor()
    fill(w, (10, Y, 2), (10, Y + 1, 2), STONE)  # the wall: feet and head cells
    assert break_ticks(1.5, 2.0, True).design_ticks == 23  # DESIGN.md 4.3 check value
    res = N.find_path(w, feet(2, Y, 2), N.GoalBlock(15, Y, 2), held=WOOD_PICK)
    assert res.ok
    digs = res.path.digs
    assert [d.pos for d in digs] == [(10, Y + 1, 2), (10, Y, 2)]  # head first, then feet
    assert all(d.cls == STONE and d.ticks == 23 for d in digs) and res.path.dig_ticks == 46
    assert res.path.cost == 13 + 2 * (1 + 3 * 23 * 50 / 1000)  # (1 + 3 * seconds) * digCost per block
    by_hand = N.find_path(w, feet(2, Y, 2), N.GoalBlock(15, Y, 2))
    assert by_hand.ok and by_hand.path.dig_ticks == 2 * 150  # DESIGN.md 4.3: stone by hand 150
    hand, pick = by_hand.path, res.path
    assert (hand.t_goto - hand.t_search) - (pick.t_goto - pick.t_search) == pytest.approx(2 * (150 - 23))
    best = N.find_path(w, feet(2, Y, 2), N.GoalBlock(15, Y, 2), held=0, tools=(WOOD_PICK, STONE_PICK))
    assert all(d.ticks == break_ticks(1.5, 4.0, True).design_ticks for d in best.path.digs)
    assert N.find_path(w, feet(2, Y, 2), N.GoalBlock(15, Y, 2), allow_dig=False).code == N.NO_PATH


def test_never_digs_next_to_liquid():
    w = corridor()
    fill(w, (10, Y, 2), (10, Y + 1, 2), STONE)
    w.set(10, Y + 2, 2, ID["water"])  # water above the wall's head block
    res = N.find_path(w, feet(2, Y, 2), N.GoalBlock(15, Y, 2), held=STONE_PICK)
    assert res.code == N.NO_PATH
    grid = N.NavGrid(w, held=STONE_PICK)
    assert not grid.safe_to_break(10, Y + 1, 2) and grid.safe_to_break(10, Y, 2)
    w.set(10, Y + 2, 2, BEDROCK)
    w.set(10, Y + 1, 1, ID["water"])  # water beside the head block
    assert N.find_path(w, feet(2, Y, 2), N.GoalBlock(15, Y, 2), held=STONE_PICK).code == N.NO_PATH
    w.set(10, Y + 1, 1, BEDROCK)
    assert N.find_path(w, feet(2, Y, 2), N.GoalBlock(15, Y, 2), held=STONE_PICK).ok


def test_dig_down_and_falling_blocks():
    w = arena(9, 9, floor=STONE)
    res = N.find_path(w, feet(4, Y, 4), N.GoalBlock(4, Y - 3, 4), held=STONE_PICK)
    assert res.ok and [s.move for s in res.path.steps] == [N.Move.DOWN] * 3
    assert [d.pos for d in res.path.digs] == [(4, FLOOR, 4), (4, FLOOR - 1, 4), (4, FLOOR - 2, 4)]
    w2 = corridor()
    fill(w2, (10, Y, 2), (10, Y + 1, 2), STONE)
    w2.set(10, Y + 2, 2, ID["sand"])  # sand would fall into the dug gap
    assert N.find_path(w2, feet(2, Y, 2), N.GoalBlock(15, Y, 2), held=STONE_PICK).code == N.NO_PATH


# -- pillars and bridges -------------------------------------------------------------------------


def test_pillar_up_out_of_a_pit():
    w = arena(9, 9)
    fill(w, (0, Y, 0), (8, Y + 2, 8), BEDROCK)  # solid up to Y + 2 ...
    fill(w, (4, Y, 4), (4, Y + 2, 4), AIR)  # ... with a 1x1 pit, 3 deep
    goal = N.GoalBlock(1, Y + 3, 1)
    assert N.find_path(w, feet(4, Y, 4), goal).code == N.NO_PATH
    res = N.find_path(w, feet(4, Y, 4), goal, blocks=5)
    assert res.ok
    p = res.path
    # Two pillar steps lift the feet to Y + 2, level with the rim's top block; one jump gets out.
    assert [s.move for s in p.steps[:3]] == [N.Move.PILLAR, N.Move.PILLAR, N.Move.ASCEND]
    assert p.pillars == 2 and p.blocks_used == 2 and p.jumps == 3
    assert [s.places for s in p.steps if s.move == N.Move.PILLAR] == [((4, Y, 4),), ((4, Y + 1, 4),)]
    assert N.find_path(w, feet(4, Y, 4), goal, blocks=1).code == N.NO_PATH  # one block per step
    assert N.find_path(w, feet(4, Y, 4), goal, blocks=5, allow_pillar=False).code == N.NO_PATH


def test_bridge_over_a_gap_with_blocks():
    w = arena(20, 5)
    fill(w, (0, Y, 0), (19, Y + 3, 4), BEDROCK)
    fill(w, (0, Y, 2), (19, Y + 1, 2), AIR)
    fill(w, (8, FLOOR - 5, 2), (10, FLOOR, 2), AIR)  # a 3-wide gap, 6 deep
    goal = N.GoalBlock(15, Y, 2)
    assert N.find_path(w, feet(2, Y, 2), goal).code == N.NO_PATH
    res = N.find_path(w, feet(2, Y, 2), goal, blocks=4)
    assert res.ok and res.path.bridges == 3 and res.path.blocks_used == 3
    assert [s.places for s in res.path.steps if s.places] == [((x, FLOOR, 2),) for x in (8, 9, 10)]


def test_blocks_are_placed_only_inside_the_world():
    """The voxels end at y 191 (sim-only limit): a pillar cannot place above it; the agent may
    stand with its head above it, where the world reads as air."""
    w = World(20, 20)
    b = w.blocks.copy()
    b[:, : 60 - MIN_Y + 1, :] = STONE
    b[10, : 188 - MIN_Y + 1, 10] = STONE  # a stone column; feet on top at 189
    w.load(b)
    res = N.find_path(w, feet(10, 189, 10), N.GoalBlock(10, 192, 10), blocks=5)
    assert res.ok and [s.move for s in res.path.steps] == [N.Move.PILLAR] * 3
    assert max(p[1] for s in res.path.steps for p in s.places) == MAX_Y
    assert N.find_path(w, feet(10, 189, 10), N.GoalBlock(10, 193, 10), blocks=5).code == N.NO_PATH


# -- never broken, avoided, climbable ---------------------------------------------------------------


@pytest.mark.parametrize("block", ["door", "other_fence_gate", "other_open_fence_gate", "crafting_table",
                                   "furnace", "bed", "chest", "other_other_pickaxe_block_8"])
def test_doors_gates_and_stations_are_never_broken(block):
    w = corridor()
    w.set(10, Y, 2, ID[block])
    assert N.find_path(w, feet(2, Y, 2), N.GoalBlock(15, Y, 2), held=AXE, tools=(STONE_PICK,)).code == N.NO_PATH
    plain = N.Settings(never_break=(), avoid=())  # the package's own lists: only chests are kept
    res = N.find_path(w, feet(2, Y, 2), N.GoalBlock(15, Y, 2), held=AXE, tools=(STONE_PICK,), settings=plain)
    assert res.ok == (block != "chest")
    if res.ok:
        assert (10, Y, 2) in {d.pos for d in res.path.digs}


@pytest.mark.parametrize("hazard,dug", [("fire", True), ("cobweb", True), ("sweet_berry_bush", True),
                                        ("powder_snow", True), ("wither_rose", True), ("cactus", True),
                                        ("other_campfire", True)])
def test_avoided_blocks_are_dug_not_entered(hazard, dug):
    w = corridor()
    w.set(10, Y, 2, ID[hazard])
    res = N.find_path(w, feet(2, Y, 2), N.GoalBlock(15, Y, 2), held=AXE, tools=(STONE_PICK,))
    assert res.ok and ((10, Y, 2) in {d.pos for d in res.path.digs}) == dug


def test_lava_in_the_way_is_never_dug():
    w = corridor()
    w.set(10, Y, 2, ID["lava"])
    assert N.find_path(w, feet(2, Y, 2), N.GoalBlock(15, Y, 2), held=AXE, tools=(STONE_PICK,)).code == N.NO_PATH


def test_magma_floor_is_walked_over():
    """The package cannot avoid standing on a block (blocksToAvoid only stops entering it)."""
    w = corridor()
    w.set(10, FLOOR, 2, ID["magma_block"])
    res = N.find_path(w, feet(2, Y, 2), N.GoalBlock(15, Y, 2), allow_dig=False)
    assert res.ok and (10, Y, 2) in res.path.cells


def test_ladders_are_climbed_and_vines_passed():
    w = arena(9, 9)
    fill(w, (0, Y, 0), (8, Y + 4, 8), BEDROCK)
    fill(w, (4, Y, 4), (4, Y + 5, 4), ID["ladder"])  # a 1x1 shaft lined with ladder
    res = N.find_path(w, feet(4, Y, 4), N.GoalBlock(1, Y + 5, 1), allow_dig=False)
    assert res.ok and res.path.blocks_used == 0
    assert [s.move for s in res.path.steps].count(N.Move.CLIMB) == 4
    down = N.find_path(w, feet(4, Y + 5, 4), N.GoalBlock(4, Y, 4), allow_dig=False)
    assert down.ok and all(s.move == N.Move.DOWN and s.fall == 1 for s in down.path.steps)
    fill(w, (4, Y, 4), (4, Y + 5, 4), ID["vine"])  # vines: passable, not climbable
    assert N.find_path(w, feet(4, Y, 4), N.GoalBlock(1, Y + 5, 1), allow_dig=False).code == N.NO_PATH
    c = corridor()
    c.set(10, Y, 2, ID["vine"])
    res = N.find_path(c, feet(2, Y, 2), N.GoalBlock(15, Y, 2), allow_dig=False)
    assert res.ok and res.path.dig_ticks == 0


# -- goals and limits ----------------------------------------------------------------------------------


def test_reach_goals():
    w = arena()
    target = (20, Y + 2, 20)
    w.set(*target, STONE)
    res = N.find_path(w, feet(2, Y, 2), N.near_block(target))
    assert res.ok and N.REACH_BLOCK == 4.5
    e = res.path.end
    assert sum((a - b) ** 2 for a, b in zip(e, target)) <= 4.5 ** 2
    ent = N.find_path(w, feet(2, Y, 2), N.near_entity((10.3, float(Y), 25.7)))
    assert ent.ok and N.REACH_ENTITY == 3.0
    e = ent.path.end
    assert (e[0] - 10) ** 2 + (e[1] - Y) ** 2 + (e[2] - 25) ** 2 <= 9


def test_xz_goal():
    w = arena()
    w.set(20, Y, 20, BEDROCK)
    res = N.find_path(w, feet(2, Y, 2), N.GoalXZ(20, 20))
    assert res.ok and res.path.end == (20, Y + 1, 20)  # any height: here on top of the block
    near = N.find_path(w, feet(2, Y, 2), N.GoalXZ(20, 20, range=5))
    e = near.path.end
    assert near.ok and (e[0] - 20) ** 2 + (e[2] - 20) ** 2 <= 25 and near.path.length < res.path.length
    w.set(20, Y + 1, 20, BEDROCK)  # a 2-high pillar: its top needs blocks
    assert N.find_path(w, feet(2, Y, 2), N.GoalXZ(20, 20)).code == N.NO_PATH
    assert N.find_path(w, feet(2, Y, 2), N.GoalXZ(20, 20), blocks=2).ok


def test_flee_reaches_16_blocks():
    w = arena(48, 48)
    threat = (24.5, float(Y), 24.5)
    res = N.find_path(w, feet(26, Y, 24), N.flee_from(threat))
    assert res.ok
    e = res.path.end
    assert math.hypot(e[0] + 0.5 - 24.5, e[2] + 0.5 - 24.5) >= 16
    assert res.path.length < 20  # straight away, not a detour


def test_search_radius_bounds_cost_not_distance():
    """searchRadius = 96 caps g + h at h(start) + 96 (astar.js): far targets on a cheap path are
    reachable; a near target behind a long detour is not."""
    assert N.SEARCH_RADIUS == 96
    w = arena(130, 3)
    fill(w, (0, Y, 0), (129, Y + 3, 0), BEDROCK)
    fill(w, (0, Y, 2), (129, Y + 3, 2), BEDROCK)
    res = N.find_path(w, feet(1, Y, 1), N.GoalBlock(128, Y, 1))
    assert res.ok and res.path.length == 127
    d = arena(70, 70)
    fill(d, (0, Y, 10), (63, Y + 3, 10), BEDROCK)  # a wall; the way round is about 2 x 60 blocks
    assert N.find_path(d, feet(3, Y, 7), N.GoalBlock(3, Y, 13)).code == N.NO_PATH
    far = N.find_path(d, feet(3, Y, 7), N.GoalBlock(3, Y, 13), search_radius=-1)
    assert far.ok and far.path.cost > 6 + 96


def serpentine(size: int) -> World:
    """A maze of bedrock rows at odd z, each with a single gap alternating between the ends."""
    w = arena(size, size)
    for i, z in enumerate(range(1, size - 1, 2)):
        fill(w, (0, Y, z), (size - 1, Y + 2, z), BEDROCK)
        gap = size - 1 if i % 2 == 0 else 0
        fill(w, (gap, Y, z), (gap, Y + 1, z), AIR)
    return w


def test_maze_solved_within_budget_and_timeout_on_small_budget():
    w = serpentine(64)
    goal = N.GoalBlock(0, Y, 62)
    assert N.find_path(w, feet(0, Y, 0), goal).code == N.NO_PATH  # far beyond the cost bound
    res = N.find_path(w, feet(0, Y, 0), goal, search_radius=-1)
    assert res.ok and res.path.length > 32 * 62 and res.path.dig_ticks == 0
    small = N.find_path(w, feet(0, Y, 0), goal, node_budget=500, search_radius=-1)
    assert small.code == N.PATH_TIMEOUT and small.path is None and small.nodes == 500


def test_path_timeout_on_huge_open_search():
    """An unreachable goal in a big field: the search runs out of budget (pathfinder's think
    timeout), while in a small field it runs out of nodes (NO_PATH)."""
    big = arena(128, 128)
    fill(big, (100, Y, 100), (104, Y + 4, 104), BEDROCK)
    fill(big, (102, Y, 102), (102, Y + 1, 102), AIR)  # an enclosed cell
    res = N.find_path(big, feet(30, Y, 30), N.GoalBlock(102, Y, 102), node_budget=3000)
    assert res.code == N.PATH_TIMEOUT and res.nodes == 3000
    small = arena(24, 24)
    fill(small, (10, Y, 10), (14, Y + 4, 14), BEDROCK)
    fill(small, (12, Y, 12), (12, Y + 1, 12), AIR)
    assert N.find_path(small, feet(2, Y, 2), N.GoalBlock(12, Y, 12)).code == N.NO_PATH


def test_deterministic():
    w = serpentine(24)
    a = N.find_path(w, feet(0, Y, 0), N.GoalBlock(23, Y, 22), search_radius=-1)
    b = N.find_path(w, feet(0, Y, 0), N.GoalBlock(23, Y, 22), search_radius=-1)
    assert a == b and a.ok


def test_nav_sees_world_changes_immediately():
    w = corridor()
    assert N.find_path(w, feet(2, Y, 2), N.GoalBlock(15, Y, 2), allow_dig=False).ok
    w.set(10, Y, 2, BEDROCK)
    assert N.find_path(w, feet(2, Y, 2), N.GoalBlock(15, Y, 2), allow_dig=False).code == N.NO_PATH


# -- the same results as mineflayer-pathfinder 2.4.5 ---------------------------------------------------

_HARNESS = r"""
// Run mineflayer-pathfinder's Movements + AStar on class scenes exported by tests/test_nav.py, with
// the bridge's settings (bridge/PROTOCOL.md 4.4) and nav.py's never-break and avoid lists.
const NM = process.argv[2]
const fs = require('fs')
const { Vec3 } = require(NM + '/vec3')
const registry = require(NM + '/prismarine-registry')('1.21.1')
const Block = require(NM + '/prismarine-block')(registry)
const Movements = require(NM + '/mineflayer-pathfinder/lib/movements')
const AStar = require(NM + '/mineflayer-pathfinder/lib/astar')
const Move = require(NM + '/mineflayer-pathfinder/lib/move')
const { goals } = require(NM + '/mineflayer-pathfinder')
const out = []
for (const c of JSON.parse(fs.readFileSync(process.argv[3]))) {
  const grid = new Map()
  for (const [x, y, z, s] of c.cells) grid.set(`${x},${y},${z}`, s)
  const bot = {
    registry, game: { minY: -64 }, entity: { effects: {} }, inventory: { items: () => [] }, entities: {},
    pathfinder: { bestHarvestTool: () => c.tool ? { type: registry.itemsByName[c.tool].id } : null },
    blockAt (p) {
      let s
      if (p.x < 0 || p.x >= c.sx || p.z < 0 || p.z >= c.sz || p.y < -64) s = c.edge
      else if (p.y < c.ylo) s = c.below
      else s = grid.get(`${p.x},${p.y},${p.z}`) ?? c.air
      const b = Block.fromStateId(s, 0)
      b.position = new Vec3(p.x, p.y, p.z)
      return b
    }
  }
  const m = new Movements(bot)
  m.allowParkour = false
  m.canOpenDoors = false
  m.allowEntityDetection = false
  m.canDig = c.canDig
  m.allow1by1towers = c.towers
  for (const n of c.cantBreak) m.blocksCantBreak.add(registry.blocksByName[n].id)
  for (const n of c.avoid) m.blocksToAvoid.add(registry.blocksByName[n].id)
  const g = c.goal
  const goal = g[0] === 'block' ? new goals.GoalBlock(g[1], g[2], g[3]) : new goals.GoalNear(g[1], g[2], g[3], g[4])
  const astar = new AStar(new Move(c.start[0], c.start[1], c.start[2], c.blocks, 0), m, goal, 1e9, 1e9, c.radius)
  let r = astar.compute()
  while (r.status === 'partial') r = astar.compute()
  out.push({ status: r.status, visited: r.visitedNodes, cost: r.cost,
    path: r.status === 'success' ? r.path.map(p => [[p.x, p.y, p.z], p.toBreak.map(b => [b.x, b.y, b.z]),
      p.toPlace.map(q => [q.x + q.dx, q.y + q.dy, q.z + q.dz])]) : null })
}
fs.writeFileSync(process.argv[4], JSON.stringify(out))
"""


def _scenes() -> list[dict]:
    """(world, start, goal, ...) cases covering every move, the lists, water and real terrain."""
    out = []

    def case(w, start, goal, blocks=0, tool=None, ylo=FLOOR - 6, settings=N.DEFAULT_SETTINGS):
        out.append(dict(w=w, start=start, goal=goal, blocks=blocks, tool=tool, ylo=ylo, settings=settings))

    for h in (3, 4):
        w = arena(9, 9)
        fill(w, (4, Y, 4), (4, FLOOR + h, 4), BEDROCK)
        case(w, (4, FLOOR + h + 1, 4), ("block", 1, Y, 1))
    w = arena(130, 3)
    fill(w, (0, Y, 0), (129, Y + 3, 0), BEDROCK)
    fill(w, (0, Y, 2), (129, Y + 3, 2), BEDROCK)
    case(w, (1, Y, 1), ("block", 110, Y, 1))
    w = arena(70, 70)
    fill(w, (0, Y, 10), (63, Y + 3, 10), BEDROCK)
    case(w, (3, Y, 7), ("block", 3, Y, 13))
    for depth in (1, 3):
        w = arena(25, 5)
        fill(w, (0, Y, 0), (24, 62, 4), BEDROCK)
        fill(w, (10, 63 - depth, 0), (14, 62, 4), ID["water"])
        fill(w, (15, 62, 0), (15, 62, 4), AIR)
        for blocks in (0, 16):
            case(w, (5, 63, 2), ("block", 20, 63, 2), blocks=blocks)
    plain = N.Settings(never_break=(), avoid=())
    for name in ("door", "crafting_table", "bed", "other_open_fence_gate", "chest", "sweet_berry_bush",
                 "powder_snow", "cactus", "cobweb", "fire", "vine", "oak_leaves", "fence", "snow", "other_carpet",
                 "other_slab", "torch"):
        w = corridor()
        w.set(10, Y, 2, ID[name])
        case(w, (2, Y, 2), ("block", 15, Y, 2))
        case(w, (2, Y, 2), ("block", 15, Y, 2), settings=plain)
    w = corridor()
    w.set(10, FLOOR, 2, ID["magma_block"])
    case(w, (2, Y, 2), ("block", 15, Y, 2))
    w = arena(9, 9)
    fill(w, (0, Y, 0), (8, Y + 4, 8), BEDROCK)
    fill(w, (4, Y, 4), (4, Y + 5, 4), ID["ladder"])
    case(w, (4, Y, 4), ("block", 1, Y + 5, 1))
    w = arena(12, 12)
    fill(w, (6, Y, 5), (6, Y + 1, 5), BEDROCK)
    case(w, (5, Y, 5), ("block", 6, Y, 6))
    w = arena(9, 9)
    fill(w, (0, Y, 0), (8, Y + 2, 8), BEDROCK)
    fill(w, (4, Y, 4), (4, Y + 2, 4), AIR)
    case(w, (4, Y, 4), ("block", 1, Y + 3, 1), blocks=5)
    w = arena(20, 5)
    fill(w, (0, Y, 0), (19, Y + 3, 4), BEDROCK)
    fill(w, (0, Y, 2), (19, Y + 1, 2), AIR)
    fill(w, (8, FLOOR - 5, 2), (10, FLOOR, 2), AIR)
    case(w, (2, Y, 2), ("block", 15, Y, 2), blocks=4)
    w = corridor()
    fill(w, (10, Y, 2), (10, Y + 1, 2), STONE)
    case(w, (2, Y, 2), ("block", 15, Y, 2))
    case(w, (2, Y, 2), ("block", 15, Y, 2), tool="wooden_pickaxe")
    w = corridor()
    fill(w, (10, Y, 2), (10, Y + 1, 2), STONE)
    w.set(10, Y + 2, 2, ID["sand"])
    case(w, (2, Y, 2), ("block", 15, Y, 2))
    w = arena(9, 9, floor=STONE)
    case(w, (4, Y, 4), ("block", 4, Y - 3, 4), ylo=FLOOR - 8)
    for seed, preset in ((3, "forest"), (5, "default")):  # natural terrain: trees, grass, water, caves
        g = wg.generate(seed, wg.preset(preset, size_x=32, size_z=32))
        sx, sy, sz = (int(v) for v in g.meta["spawn"])
        lo = int(np.min(g.meta["surface"])) - 6
        b = g.blocks.copy()
        b[:, : lo - MIN_Y, :] = STONE  # the harness reads everything below ``lo`` as stone
        g.load(b)
        for k, (x, z) in enumerate(((3, 28), (27, 4), (30, 30))):
            case(g, (sx, sy, sz), ("near", x, int(g.heightmap[x, z]) + 1, z, 2.0), blocks=2 * k, ylo=lo,
                 tool="wooden_pickaxe" if k == 1 else None)
    return out


def _export(c: dict) -> dict:
    w, s = c["w"], c["settings"]
    view = N.block_view(C, *N._lists(C, s))
    ylo = c["ylo"]
    yhi = min(MAX_Y, int(w.heightmap.max()) + 16)
    region = w.blocks[:, ylo - MIN_Y: yhi - MIN_Y + 1, :]
    cells = [[int(x), int(i) + ylo, int(z), view.state[region[x, i, z]]] for x, i, z in np.argwhere(region != 0)]
    lists = N.bridge_block_lists(C, s)
    return dict(sx=w.size_x, sz=w.size_z, ylo=ylo, cells=cells, below=view.state[int(w.blocks[0, ylo - 1 - MIN_Y, 0])],
                edge=view.state[ID["barrier"]], air=view.state[AIR], start=c["start"], goal=c["goal"],
                blocks=c["blocks"], tool=c["tool"], canDig=s.can_dig, towers=s.allow_1by1_towers,
                radius=s.search_radius, cantBreak=lists["blocksCantBreak"], avoid=lists["blocksToAvoid"])


@pytest.mark.skipif(shutil.which("node") is None or not (NODE_MODULES / "mineflayer-pathfinder").is_dir(),
                    reason="needs Node and the bridge's node_modules (npm ci in bridge/)")
def test_same_as_mineflayer_pathfinder(tmp_path):
    """Status, visited nodes, cost, every path cell, every break and every placement match the
    installed package on the same voxels (each class drawn as its ``BlockView.state``)."""
    scenes = _scenes()
    (tmp_path / "harness.js").write_text(_HARNESS)
    (tmp_path / "cases.json").write_text(json.dumps([_export(c) for c in scenes]))
    subprocess.run(["node", str(tmp_path / "harness.js"), str(NODE_MODULES), str(tmp_path / "cases.json"),
                    str(tmp_path / "out.json")], check=True, timeout=120)
    results = json.loads((tmp_path / "out.json").read_text())
    assert len(results) == len(scenes)
    statuses = set()
    for c, pf in zip(scenes, results):
        st, g = c["start"], c["goal"]
        goal = N.GoalBlock(*g[1:]) if g[0] == "block" else N.GoalNear(*g[1:])
        tool = C.item_class(c["tool"]).id if c["tool"] else 0
        res = N.find_path(c["w"], feet(*st), goal, held=tool, blocks=c["blocks"], settings=c["settings"],
                          node_budget=10 ** 6)
        mine = None if not res.ok else [[list(s.pos), [list(d.pos) for d in s.digs], [list(p) for p in s.places]]
                                        for s in res.path.steps]
        assert (res.code, res.nodes) == ({"success": N.OK, "noPath": N.NO_PATH}[pf["status"]], pf["visited"]), c
        assert mine == pf["path"], c
        if res.ok:
            assert res.path.cost == pytest.approx(pf["cost"], abs=1e-9)
        statuses.add(res.code)
    assert statuses == {N.OK, N.NO_PATH}

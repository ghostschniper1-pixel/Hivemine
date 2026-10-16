"""Navigation: mineflayer-pathfinder's movement rules and A* on the class grid (DESIGN.md 4.5).

The bridge moves agents with mineflayer-pathfinder 2.4.5 [8] (bridge/PROTOCOL.md 4.4). This module
is a port of that package's ``lib/movements.js`` (moves and costs) and ``lib/astar.js`` with
``lib/heap.js`` (search order, cost bound), run on the simulator's block classes, so ``goto``,
``mine``, ``explore``, ``give`` and ``flee`` find the same paths and fail in the same places on
both sides. ``tests/test_nav.py`` runs the installed package on the same scenes and compares
status, path, breaks, placements and visited-node counts (skipped where Node is missing)::

    from hivemind.sim.ref.nav import find_path, GoalNear, near_block
    res = find_path(world, feet, near_block((20, 61, 10)), held=stone_pickaxe_id, blocks=12)
    if res.ok:
        res.path.steps, res.path.digs, res.path.t_goto     # what to do and how long it takes
    else:
        res.code                                            # "NO_PATH" or "PATH_TIMEOUT"

**Blocks as the pathfinder sees them** (``block_view``). A node is the agent's feet cell; the body
fills it and the cell above. Each block class gets the flags ``Movements.getBlock`` computes for
its vanilla members, from the same minecraft-data fields the rules file holds:

- *safe* (the body may be in it): no bounding box, or a ladder, or a carpet (collision top
  < 0.1), and not avoided. Water, short grass, flowers, vines and snow layers (minecraft-data
  gives snow no bounding box) are safe; a fence gate is not, open or closed (its block has one).
- *physical* (may be stood on): a bounding box and not fence-like (the lowest state's first
  collision box taller than 1: fences and walls). Fence gates count as physical, because their
  lowest state is an open gate with no box.
- *liquid* (water, lava), *climbable* (ladder only; the package leaves vines out), *replaceable*
  (air, water, lava), *falls* (sand, gravel), and per state the collision *height*.
- *never broken*: the package's own list (chest, and every block minecraft-data marks not
  diggable) plus ``never_break_classes`` (stations, beds, doors, fence gates); *avoided* (never
  safe, so broken if it is in the way): the package's (fire, cobweb, lava) plus every hazard class
  (``avoid_classes``). The bridge must load the same additions into
  ``Movements.blocksCantBreak`` and ``blocksToAvoid``: ``bridge_block_lists`` gives the vanilla
  names (bridge/PROTOCOL.md 4.4).

A class takes the view of most of its block states (ties: the lowest state id). The flags are
uniform in every class except the fallback ``other_other_block_7`` (torchflower crop, pitcher
crop, pitcher plant: safe, while the package sees the pitcher crop as a solid block). Heights
differ inside 25 classes, all fallbacks or partial blocks (snow layers 2-8, soul sand and soil,
walls, amethyst buds, chests with barrels); the height of most states is used, which matters
only to the 1.2-block jump test and the half-block pillar test.

**Moves** (``getNeighbors``), in this order: for west, east, north, south: *forward* (dig the head
and feet cells if needed; with no support below, place one block there unless the target is
water: swimming); *jump up* one block (placing up to two blocks under the target if it has no
support); *drop down* (dig through the target column if needed, then fall onto the first solid
block at most ``maxDropDown`` = 4 below the current feet cell, so the feet fall at most 3, or
into water at any depth); then the four *diagonals* (one side column clear, the cheaper one
dug if needed; level, one up, or one down); then *down* (dig the block below and fall, or climb
down a ladder) and *up* (climb a ladder, or jump and place a block under the feet:
``allow1by1towers``). No parkour (``allowParkour = false``); doors and fence gates are never
opened (``canOpenDoors = false``) and, never being broken, are walls whether open or closed.
Nothing is dug next to a liquid (``dontCreateFlow``) or under sand or gravel
(``dontMineUnderFallingBlock``). In water the agent swims at the surface level: it may move
across and step out onto a ledge just under the surface, never up through water (``getMoveUp``
refuses liquid feet cells); from deep water a block level with the surface is too high to jump on
(the floor below is water). Like the package, a plan never sees its own placed blocks: after a
pillar step the next jump is judged from the empty cell below the feet.

**Costs** are the package's: forward 1, jump up 2, drop down 1, diagonal sqrt(2) (+1 up), up and
down 1; each placed block ``placeCost`` 1; each block dug in the way (1 + 3 * seconds) *
``digCost`` 1 (blocks removed only to place a block there cost nothing extra, as in the
package); a move starting in water adds ``liquidCost`` 1; a move costing more than 100 is dropped.
Entities are ignored (the package adds 1 per entity in the way [8]; the sim has none in v1 [est]).

**Search** (``astar.js``): A* with the goal's heuristic, the package's binary heap (ties and its
off-by-one sift included) and its open/closed bookkeeping. ``searchRadius`` = 96 bounds the
cost, not the distance: a neighbour is skipped when g + h > h(start) + 96, so a target any
distance away is reachable along a path costing at most 96 more than the straight-line estimate,
and a target 6 blocks away behind a long wall is not. More than ``NODE_BUDGET`` visited nodes
gives ``PATH_TIMEOUT``, standing in for the 2 s ``thinkTimeout`` [8]: 10,000 nodes at an assumed
25 nodes per ms of search [est] is 400 ms of search, i.e. 40 physics ticks of ``tickTimeout`` =
10 ms (DESIGN.md 11.1) or 2 s of wall time. An exhausted search gives ``NO_PATH``.

**Dig time** of a block is ``mechanics.break_ticks(...).design_ticks`` with the best of the held
tool and ``tools`` (the package's ``bestHarvestTool``): Mineflayer's wait with the bridge's
tool-speed patch (DESIGN.md 5, wrapper rules), on the ground and dry. ``blocks`` is the number of
scaffolding items the package counts: dirt and cobblestone only (``SCAFFOLDING_ITEMS``).

**Sim-only limits.** Blocks are placed only inside the world (y <= 191): the voxels end there,
while the real game allows building to 319. Half blocks are approximated: a node is always a whole
cell (the package's start offset for slabs is not modelled) [est].

**Duration** (DESIGN.md 4.5)::

    T_goto = T_search + L/v + k_turn*turns + k_jump*jumps + sum(dig ticks) + k_pillar*(blocks placed) + eps
    T_search = search_ms / tickTimeout

in ticks, with v = 4.317 m/s walking [mem] (also used for swimming and climbing [est]), L the
length of the path through the node centres, and the coefficients ``GotoModel`` placeholders [est]
until M2 fits them to Mineflayer telemetry. Every placed block is charged like a pillar [est]
(DESIGN.md 4.5 names only pillars).

The grid reads the world's live voxels, so block changes apply to the next search at once; the
block views and dig-time tables are cached per concept set, list and tool set.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

from hivemind.rules import load_rules
from hivemind.sim.ref.mechanics import break_ticks
from hivemind.sim.ref.world import HEIGHT, MAX_Y, MIN_Y, World

if TYPE_CHECKING:
    from hivemind.rules.classes import Concepts

OK = "OK"
NO_PATH = "NO_PATH"  # DESIGN.md 5 failure codes
PATH_TIMEOUT = "PATH_TIMEOUT"

_RULES = load_rules()
WALK_SPEED = float(_RULES.constant("walk_speed_m_per_s"))  # [mem] 4.317 m/s (DESIGN.md 4.5)
TICKS_PER_SECOND = int(_RULES.constant("ticks_per_second"))  # [mem] 20
REACH_BLOCK = float(_RULES.constant("reach_block"))  # [mem] 4.5 (DESIGN.md 4.5)
REACH_ENTITY = float(_RULES.constant("reach_entity"))  # [mem] 3.0
MAX_DROP_DOWN = int(_RULES.constant("nav_max_drop_blocks"))  # [8] maxDropDown = 4: landing block <= 4 below
MAX_SAFE_FALL = MAX_DROP_DOWN - 1  # feet fall at most 3 onto a solid block (getLandingBlock, checked)
SEARCH_RADIUS = 96.0  # bridge/PROTOCOL.md 4.4: searchRadius = 96, a cost bound (astar.js, checked)
NODE_BUDGET = 10_000  # [est] stands in for thinkTimeout = 2000 ms (see the module docstring)
TICK_TIMEOUT_MS = 10.0  # DESIGN.md 11.1: pathfinder tickTimeout = 10 ms per physics tick
NODES_PER_MS = 25.0  # [est] Mineflayer A* speed, measured in M2
FLEE_DISTANCE = 16.0  # DESIGN.md 5: flee moves 16-24 blocks away; the goal's minimum [est]
MS_PER_TICK = 50  # prismarine-block digTime returns ticks * 50 ms (checked: index.js)
MAX_MOVE_COST = 100.0  # movements.js: a move costing more is dropped; 100 also means "cannot break"
JUMP_HEIGHT = 1.2  # movements.js: a step up of more than 1.2 blocks is too high
HALF_BLOCK_JUMP = -0.2  # movements.js getMoveUp: no jump-place from a floor more than 0.2 below the cell

# mineflayer-pathfinder 2.4.5 Movements constructor (checked: lib/movements.js)
PF_CANT_BREAK = ("chest",)  # plus every block minecraft-data marks not diggable
PF_AVOID = ("fire", "cobweb", "lava")  # blocksToAvoid
PF_LIQUIDS = ("water", "lava")
PF_CLIMBABLES = ("ladder",)  # vine is commented out in the package
PF_FALLING = ("sand", "gravel")  # gravityBlocks
PF_REPLACEABLE = ("air", "cave_air", "void_air", "water", "lava")
SCAFFOLDING_ITEMS = ("dirt", "cobblestone")  # scafoldingBlocks: the items countScaffoldingItems counts
# minecraft-data pc/1.21.1 blocks.json gives snow the bounding box "empty" although layers 2-8
# collide; every other block's bounding box is "block" exactly when some state has a collision
# box (checked against the installed package for all 1,060 blocks).
EMPTY_BOX_DESPITE_SHAPES = ("snow",)

# Additions the bridge loads into Movements.blocksCantBreak and blocksToAvoid (bridge_block_lists).
# Never broken: the agent's stations and bed, chests, and every class holding a door or a fence
# gate (DESIGN.md 4.5: doors and gates are walls; canOpenDoors = false).
NEVER_BREAK = ("chest", "crafting_table", "furnace", "bed")
NEVER_BREAK_MEMBER_SUFFIXES = ("_door", "_fence_gate")
# Avoided: every hazard class (DESIGN.md 4.4 flags: lava, fire, magma, cactus, cobweb, powder
# snow, sweet berry bush, wither rose, campfires). The package then breaks an avoided block that is
# in the way (a berry bush) instead of walking through it; it cannot avoid standing on magma.

CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # movements.js order: west, east, north, south
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass(frozen=True)
class Settings:
    """The bridge's ``Movements`` and search settings (bridge/PROTOCOL.md 4.4; package defaults
    where the protocol sets nothing). ``never_break`` and ``avoid`` are block-class names; None
    means the default lists (``never_break_classes``, ``avoid_classes``)."""

    can_dig: bool = True
    dig_cost: float = 1.0
    place_cost: float = 1.0
    liquid_cost: float = 1.0
    dont_create_flow: bool = True
    dont_mine_under_falling_block: bool = True
    allow_1by1_towers: bool = True
    max_drop_down: int = MAX_DROP_DOWN
    infinite_liquid_dropdown_distance: bool = True
    search_radius: float = SEARCH_RADIUS
    never_break: tuple[str, ...] | None = None
    avoid: tuple[str, ...] | None = None


DEFAULT_SETTINGS = Settings()


@lru_cache(maxsize=8)
def never_break_classes(concepts: Concepts) -> tuple[str, ...]:
    """``NEVER_BREAK`` plus every class with a door or fence-gate member (iron and copper doors
    sit in fallback classes; all of their members are then never broken)."""
    out = list(NEVER_BREAK)
    for c in concepts.block_classes:
        if c.name not in out and any(m.endswith(NEVER_BREAK_MEMBER_SUFFIXES) for m in c.members):
            out.append(c.name)
    return tuple(out)


@lru_cache(maxsize=8)
def avoid_classes(concepts: Concepts) -> tuple[str, ...]:
    """Every block class with the hazard flag."""
    return tuple(c.name for c in concepts.block_classes if concepts.hazard[c.id])


def _lists(concepts: Concepts, settings: Settings) -> tuple[tuple[str, ...], tuple[str, ...]]:
    nb = settings.never_break if settings.never_break is not None else never_break_classes(concepts)
    av = settings.avoid if settings.avoid is not None else avoid_classes(concepts)
    return tuple(nb), tuple(av)


def bridge_block_lists(concepts: Concepts, settings: Settings = DEFAULT_SETTINGS) -> dict[str, list[str]]:
    """Vanilla block names the bridge adds to ``Movements.blocksCantBreak`` and ``blocksToAvoid``:
    every member of the listed classes, sorted."""
    nb, av = _lists(concepts, settings)

    def members(names: Sequence[str]) -> list[str]:
        return sorted({m for n in names for m in concepts.block_class(n).members})

    return {"blocksCantBreak": members(nb), "blocksToAvoid": members(av)}


@dataclass(frozen=True)
class GotoModel:
    """Coefficients of T_goto in ticks (DESIGN.md 4.5); all placeholders [est] until M2."""

    walk_speed: float = WALK_SPEED  # m/s [mem]
    k_turn: float = 1.0  # [est] ticks per change of direction
    k_jump: float = 2.0  # [est] ticks per jump (jump up, pillar, diagonal up)
    k_pillar: float = 8.0  # [est] ticks per placed block (pillar or bridge)
    epsilon: float = 2.0  # [est] fixed overhead
    nodes_per_ms: float = NODES_PER_MS  # [est]
    tick_timeout_ms: float = TICK_TIMEOUT_MS  # DESIGN.md 11.1

    def search_ms(self, nodes: int) -> float:
        return nodes / self.nodes_per_ms

    def search_ticks(self, nodes: int) -> float:
        """T_search = search_ms / tickTimeout."""
        return self.search_ms(nodes) / self.tick_timeout_ms

    def walk_ticks(self, metres: float) -> float:
        return metres / self.walk_speed * TICKS_PER_SECOND


# ---------------------------------------------------------------------------
# Goals (pathfinder's goals [8]; heuristics as in lib/goals.js)
# ---------------------------------------------------------------------------


def _xz(dx: float, dz: float) -> float:
    """Pathfinder's distanceXZ (octile distance), in the package's operation order."""
    dx, dz = abs(dx), abs(dz)
    return abs(dx - dz) + min(dx, dz) * math.sqrt(2)


class Goal(Protocol):
    def is_end(self, x: int, y: int, z: int) -> bool: ...

    def heuristic(self, x: int, y: int, z: int) -> float: ...


@dataclass(frozen=True)
class GoalBlock:
    """Feet exactly in a cell."""

    x: int
    y: int
    z: int

    def is_end(self, x: int, y: int, z: int) -> bool:
        return (x, y, z) == (self.x, self.y, self.z)

    def heuristic(self, x: int, y: int, z: int) -> float:
        return _xz(self.x - x, self.z - z) + abs(self.y - y)


@dataclass(frozen=True)
class GoalNear:
    """Feet cell within ``range`` of a point, measured between integer cells (pathfinder's
    ``GoalNear``: dx^2 + dy^2 + dz^2 <= range^2 [8])."""

    x: int
    y: int
    z: int
    range: float

    def is_end(self, x: int, y: int, z: int) -> bool:
        return (self.x - x) ** 2 + (self.y - y) ** 2 + (self.z - z) ** 2 <= self.range ** 2

    def heuristic(self, x: int, y: int, z: int) -> float:
        return _xz(self.x - x, self.z - z) + abs(self.y - y)


@dataclass(frozen=True)
class GoalXZ:
    """Any y in column (x, z) (pathfinder's ``GoalXZ``), or within ``range`` of it (``GoalNearXZ``)."""

    x: int
    z: int
    range: float = 0.0

    def is_end(self, x: int, y: int, z: int) -> bool:
        return (self.x - x) ** 2 + (self.z - z) ** 2 <= self.range ** 2

    def heuristic(self, x: int, y: int, z: int) -> float:
        return _xz(self.x - x, self.z - z)


@dataclass(frozen=True)
class GoalFlee:
    """Get at least ``distance`` blocks (horizontally) from a point (DESIGN.md 5 ``flee``). The
    bridge's flee goal is not fixed yet [est]; it must use this same test and heuristic."""

    x: float
    z: float
    distance: float = FLEE_DISTANCE

    def is_end(self, x: int, y: int, z: int) -> bool:
        return math.hypot(x + 0.5 - self.x, z + 0.5 - self.z) >= self.distance

    def heuristic(self, x: int, y: int, z: int) -> float:
        return max(0.0, self.distance - math.hypot(x + 0.5 - self.x, z + 0.5 - self.z))


def near_block(pos: Sequence[int], reach: float = REACH_BLOCK) -> GoalNear:
    """Within block reach (4.5 [mem]) of a block's cell."""
    return GoalNear(int(pos[0]), int(pos[1]), int(pos[2]), reach)


def near_entity(pos: Sequence[float], reach: float = REACH_ENTITY) -> GoalNear:
    """Within entity reach (3.0 [mem]) of an entity's feet cell (pathfinder's ``GoalFollow``)."""
    return GoalNear(math.floor(pos[0]), math.floor(pos[1]), math.floor(pos[2]), reach)


def flee_from(pos: Sequence[float], distance: float = FLEE_DISTANCE) -> GoalFlee:
    return GoalFlee(float(pos[0]), float(pos[2]), distance)


# ---------------------------------------------------------------------------
# Paths and results
# ---------------------------------------------------------------------------


class Move(str, Enum):
    WALK = "walk"  # forward (also swimming)
    BRIDGE = "bridge"  # forward, placing a block under the target
    ASCEND = "ascend"  # jump up one block (possibly placing under the target)
    DROP = "drop"  # step off and fall (possibly digging the target column)
    DIAGONAL = "diagonal"  # level, one up or one down
    DOWN = "down"  # through the block below (dug, or a ladder) onto the next support
    PILLAR = "pillar"  # jump and place a block under the feet
    CLIMB = "climb"  # up one cell of ladder


@dataclass(frozen=True)
class Dig:
    pos: tuple[int, int, int]
    cls: int
    ticks: int


@dataclass(frozen=True)
class Step:
    """One move: where the feet end up, what is dug (in the package's order) and placed, and how
    far the feet fall."""

    move: Move
    pos: tuple[int, int, int]
    digs: tuple[Dig, ...] = ()
    places: tuple[tuple[int, int, int], ...] = ()
    fall: int = 0


@dataclass(frozen=True)
class Path:
    start: tuple[int, int, int]
    steps: tuple[Step, ...]
    cost: float  # A* cost (pathfinder units)
    nodes: int  # nodes visited by the search (the package's visitedNodes)
    model: GotoModel = field(default=GotoModel(), repr=False)

    @property
    def end(self) -> tuple[int, int, int]:
        return self.steps[-1].pos if self.steps else self.start

    @property
    def cells(self) -> tuple[tuple[int, int, int], ...]:
        return (self.start, *(s.pos for s in self.steps))

    @property
    def length(self) -> float:
        """L: the length through the node centres (3-D)."""
        return sum(math.dist(a, b) for a, b in zip(self.cells, self.cells[1:]))

    @property
    def digs(self) -> tuple[Dig, ...]:
        return tuple(d for s in self.steps for d in s.digs)

    @property
    def dig_ticks(self) -> int:
        return sum(d.ticks for d in self.digs)

    @property
    def jumps(self) -> int:
        return sum(_is_jump(a, s) for a, s in zip(self.cells, self.steps))

    @property
    def pillars(self) -> int:
        return sum(s.move == Move.PILLAR for s in self.steps)

    @property
    def bridges(self) -> int:
        """Blocks placed by moves other than pillars (under a forward or jump-up target)."""
        return sum(len(s.places) for s in self.steps if s.move != Move.PILLAR)

    @property
    def blocks_used(self) -> int:
        return sum(len(s.places) for s in self.steps)

    @property
    def turns(self) -> int:
        """Changes of horizontal direction between consecutive horizontal moves."""
        turns, last = 0, None
        for a, b in zip(self.cells, self.cells[1:]):
            d = (b[0] - a[0], b[2] - a[2])
            if d == (0, 0):
                continue
            if last is not None and d != last:
                turns += 1
            last = d
        return turns

    @property
    def search_ms(self) -> float:
        return self.model.search_ms(self.nodes)

    @property
    def t_search(self) -> float:
        return self.model.search_ticks(self.nodes)

    @property
    def t_goto(self) -> float:
        """T_goto in ticks (DESIGN.md 4.5), with this path's ``GotoModel``."""
        m = self.model
        return (self.t_search + m.walk_ticks(self.length) + m.k_turn * self.turns + m.k_jump * self.jumps
                + self.dig_ticks + m.k_pillar * self.blocks_used + m.epsilon)

    def step_ticks(self) -> tuple[float, ...]:
        """Ticks spent on each step (walking, its digs, jump and placing; no search or eps), for
        executing the path tick by tick. Turns are charged to the step that turns."""
        m = self.model
        out, last = [], None
        for a, s in zip(self.cells, self.steps):
            b = s.pos
            t = m.walk_ticks(math.dist(a, b)) + sum(d.ticks for d in s.digs)
            t += m.k_jump * _is_jump(a, s) + m.k_pillar * len(s.places)
            d = (b[0] - a[0], b[2] - a[2])
            if d != (0, 0):
                if last is not None and d != last:
                    t += m.k_turn
                last = d
            out.append(t)
        return tuple(out)


def _is_jump(a: tuple[int, int, int], s: Step) -> bool:
    return s.move in (Move.ASCEND, Move.PILLAR) or (s.move == Move.DIAGONAL and s.pos[1] > a[1])


@dataclass(frozen=True)
class NavResult:
    code: str  # OK, NO_PATH or PATH_TIMEOUT
    path: Path | None
    nodes: int  # nodes visited (closed), as the package's visitedNodes

    @property
    def ok(self) -> bool:
        return self.code == OK


# ---------------------------------------------------------------------------
# Block classes as the pathfinder sees them
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockView:
    """Per block-class id (256 entries): the flags of ``Movements.getBlock`` (module docstring)."""

    safe: tuple[bool, ...]
    physical: tuple[bool, ...]
    liquid: tuple[bool, ...]
    climbable: tuple[bool, ...]
    replaceable: tuple[bool, ...]
    falls: tuple[bool, ...]
    cant_break: tuple[bool, ...]
    height: tuple[float, ...]  # collision top above the cell's bottom; 0 without collision
    state: tuple[int, ...]  # the lowest vanilla state with exactly this view (-1: none)
    uniform: tuple[bool, ...]  # every state of the class has these flags (heights may differ)


def _block_flags(b, never: frozenset[str], avoid: frozenset[str]) -> tuple[bool, ...]:
    """(safe, physical, liquid, climbable, replaceable, falls, cant_break) of a vanilla block."""
    shapes = b.collision_shapes
    by_state = b.collision_by_state
    box = any(len(shapes[i]) for i in (set(by_state) if by_state else {0}))
    box = box and b.name not in EMPTY_BOX_DESPITE_SHAPES
    first = shapes[by_state[0] if by_state else 0]
    fence = bool(first) and first[0][4] > 1  # shapes[0][4] of the lowest state, as the package
    carpet = bool(first) and first[0][4] < 0.1
    climbable = b.name in PF_CLIMBABLES
    avoided = b.name in PF_AVOID or b.name in avoid
    safe = (not box or climbable or carpet) and not avoided
    physical = box and not fence
    return (safe, physical, b.name in PF_LIQUIDS, climbable, b.name in PF_REPLACEABLE and not physical,
            b.name in PF_FALLING, b.name in PF_CANT_BREAK or not b.diggable or b.name in never)


@lru_cache(maxsize=8)
def block_view(concepts: Concepts, never_break: tuple[str, ...], avoid: tuple[str, ...]) -> BlockView:
    """How the pathfinder sees each class, with these never-break and avoid class lists."""
    rules = load_rules()
    never = frozenset(m for n in never_break for m in concepts.block_class(n).members)
    avoided = frozenset(m for n in avoid for m in concepts.block_class(n).members)
    views: dict[int, Counter] = {}
    first_state: dict[tuple[int, tuple], int] = {}
    for b in rules.blocks.values():
        flags = _block_flags(b, never, avoided)
        for s in range(b.min_state, b.max_state + 1):
            boxes = b.collision_boxes(s)
            view = (*flags, max(box[4] for box in boxes) if boxes else 0.0)
            cls = int(concepts.state_class[s])
            views.setdefault(cls, Counter())[view] += 1
            first_state.setdefault((cls, view), s)
    wall = (False, True, False, False, False, False, True, 1.0)  # unused ids: an unbreakable wall
    cols: list[list] = [[] for _ in range(10)]
    for cls in range(256):
        counts = views.get(cls)
        if counts is None:
            view, state, uniform = wall, -1, True
        else:
            view = max(counts, key=lambda v: (counts[v], -first_state[(cls, v)]))
            state, uniform = first_state[(cls, view)], len({v[:-1] for v in counts}) == 1
        for i, v in enumerate((*view, state, uniform)):
            cols[i].append(v)
    return BlockView(*(tuple(c) for c in cols))


@lru_cache(maxsize=64)
def dig_ticks_table(concepts: Concepts, tools: tuple[int, ...]) -> tuple[int, ...]:
    """Ticks to break each block class with the best of ``tools`` (item-class ids; 0 = hand):
    ``BreakTime.design_ticks`` (Mineflayer's wait with the bridge's patch). -1: unbreakable.

    Hardness is read from the class (a double, as in the rules), not from the float32 array
    ``concepts.hardness``: 0.2 in float32 makes leaves by hand 7 ticks instead of 6. Tool speeds
    (1, 1.5, 2, 4, 5, 6, 8, 12, 15) are exact in float32."""
    out = []
    for cls in range(256):
        best = None
        hardness = concepts.block_class(cls).hardness if concepts.block_defined[cls] else -1.0
        if hardness >= 0:
            for tool in tools:
                t = break_ticks(float(hardness), float(concepts.tool_speed[tool, cls]),
                                bool(concepts.can_harvest[tool, cls])).design_ticks
                if t is not None and (best is None or t < best):
                    best = t
        out.append(-1 if best is None else int(best))
    return tuple(out)


# ---------------------------------------------------------------------------
# The grid and its moves (lib/movements.js)
# ---------------------------------------------------------------------------

Neighbour = tuple[float, Step, int]  # (cost, step, scaffolding blocks left)


class NavGrid:
    """The package's moves over a world's live voxels for one tool set and settings.

    It holds a view of ``world.blocks``: ``World.set`` shows at once, but after ``World.load``
    (which replaces the array) make a new grid. ``find_path`` makes one per search.
    """

    def __init__(self, world: World, *, held: int = 0, tools: Sequence[int] = (),
                 settings: Settings = DEFAULT_SETTINGS):
        self.world = world
        self.settings = settings
        self.view = block_view(world.concepts, *_lists(world.concepts, settings))
        v = self.view
        self._safe, self._phys, self._liq = v.safe, v.physical, v.liquid
        self._climb, self._repl, self._falls, self._cant = v.climbable, v.replaceable, v.falls, v.cant_break
        self._h = v.height
        self.dig = dig_ticks_table(world.concepts, tuple(sorted({0, int(held), *map(int, tools)})))
        self._flat = world.blocks.reshape(-1)  # a view: changes to the world show at once
        self._sx, self._sz = world.size_x, world.size_z
        self._barrier = world.barrier

    # -- cells -----------------------------------------------------------------------

    def cls(self, x: int, y: int, z: int) -> int:
        """The class at a cell, as ``World.get`` (barrier at the sides and below, air above)."""
        if 0 <= x < self._sx and 0 <= z < self._sz:
            if 0 <= y - MIN_Y < HEIGHT:
                return int(self._flat[(x * HEIGHT + (y - MIN_Y)) * self._sz + z])
            if y > MAX_Y:
                return 0
        return self._barrier

    def passable(self, x: int, y: int, z: int) -> bool:
        """The body may be in this cell without digging (the package's ``safe``)."""
        return self._safe[self.cls(x, y, z)]

    def solid(self, x: int, y: int, z: int) -> bool:
        """The cell may be stood on (the package's ``physical``)."""
        return self._phys[self.cls(x, y, z)]

    def _height(self, x: int, y: int, z: int) -> float:
        return y + self._h[self.cls(x, y, z)]

    def _placeable(self, x: int, y: int, z: int) -> bool:
        """The sim can hold a placed block there (inside the voxels)."""
        return 0 <= x < self._sx and 0 <= z < self._sz and MIN_Y <= y <= MAX_Y

    def safe_to_break(self, x: int, y: int, z: int) -> bool:
        """``Movements.safeToBreak``: digging allowed, no liquid above or beside
        (``dontCreateFlow``), no sand or gravel above, not a never-break block."""
        s = self.settings
        if not s.can_dig:
            return False
        liq, cls = self._liq, self.cls
        if s.dont_create_flow and (liq[cls(x, y + 1, z)] or liq[cls(x - 1, y, z)] or liq[cls(x + 1, y, z)]
                                   or liq[cls(x, y, z - 1)] or liq[cls(x, y, z + 1)]):
            return False
        if s.dont_mine_under_falling_block and self._falls[cls(x, y + 1, z)]:
            return False
        return not self._cant[cls(x, y, z)]

    def _dig(self, x: int, y: int, z: int) -> Dig:
        c = self.cls(x, y, z)
        return Dig((x, y, z), c, self.dig[c])

    def _safe_or_break(self, x: int, y: int, z: int, digs: list[Dig]) -> float:
        """``Movements.safeOrBreak``: 0 if passable, the dig cost if it may be dug, else 100."""
        c = self.cls(x, y, z)
        if self._safe[c]:
            return 0.0
        if not self.safe_to_break(x, y, z):
            return MAX_MOVE_COST
        ticks = self.dig[c]
        digs.append(Dig((x, y, z), c, ticks))
        ms = ticks * MS_PER_TICK if ticks >= 0 else math.inf
        return (1 + 3 * ms / 1000) * self.settings.dig_cost

    def _landing(self, x: int, y: int, z: int, dx: int, dz: int) -> tuple[int, int, int] | None:
        """``getLandingBlock``: the feet cell after falling from (x + dx, y - 1, z + dz): above the
        first solid block if it is at most ``maxDropDown`` below y, or the first water cell."""
        lx, ly, lz = x + dx, y - 2, z + dz
        while ly > MIN_Y:
            c = self.cls(lx, ly, lz)
            if self._liq[c] and self._safe[c]:
                return (lx, ly, lz)
            if self._phys[c]:
                return (lx, ly + 1, lz) if y - ly <= self.settings.max_drop_down else None
            if not self._safe[c]:
                return None
            ly -= 1
        return None

    def neighbours(self, x: int, y: int, z: int, blocks: int) -> list[Neighbour]:
        """``getNeighbors``: the moves from feet cell (x, y, z) with ``blocks`` scaffolding items."""
        out: list[Neighbour] = []
        for dx, dz in CARDINALS:
            self._forward(x, y, z, dx, dz, blocks, out)
            self._jump_up(x, y, z, dx, dz, blocks, out)
            self._drop_down(x, y, z, dx, dz, blocks, out)
        for dx, dz in DIAGONALS:
            self._diagonal(x, y, z, dx, dz, blocks, out)
        self._down(x, y, z, blocks, out)
        self._up(x, y, z, blocks, out)
        return out

    def _forward(self, x: int, y: int, z: int, dx: int, dz: int, blocks: int, out: list[Neighbour]) -> None:
        tx, tz = x + dx, z + dz
        cost, digs, places = 1.0, [], []
        cd = self.cls(tx, y - 1, tz)
        if not self._phys[cd] and not self._liq[self.cls(tx, y, tz)]:
            if blocks == 0 or not self._placeable(tx, y - 1, tz):
                return
            if not self._repl[cd]:
                if not self.safe_to_break(tx, y - 1, tz):
                    return
                digs.append(self._dig(tx, y - 1, tz))
            places.append((tx, y - 1, tz))
            cost += self.settings.place_cost
        cost += self._safe_or_break(tx, y + 1, tz, digs)
        if cost > MAX_MOVE_COST:
            return
        cost += self._safe_or_break(tx, y, tz, digs)
        if cost > MAX_MOVE_COST:
            return
        if self._liq[self.cls(x, y, z)]:
            cost += self.settings.liquid_cost
        move = Move.BRIDGE if places else Move.WALK
        out.append((cost, Step(move, (tx, y, tz), tuple(digs), tuple(places)), blocks - len(places)))

    def _jump_up(self, x: int, y: int, z: int, dx: int, dz: int, blocks: int, out: list[Neighbour]) -> None:
        tx, tz = x + dx, z + dz
        cost, digs, places = 2.0, [], []
        cc = self.cls(tx, y, tz)
        c_height = y + self._h[cc]
        if not self._phys[cc]:
            if blocks == 0:
                return
            cd = self.cls(tx, y - 1, tz)
            if not self._phys[cd]:
                if blocks == 1 or not self._placeable(tx, y - 1, tz):
                    return
                if not self._repl[cd]:
                    if not self.safe_to_break(tx, y - 1, tz):
                        return
                    digs.append(self._dig(tx, y - 1, tz))
                places.append((tx, y - 1, tz))
                cost += self.settings.place_cost
            if not self._placeable(tx, y, tz):
                return
            if not self._repl[cc]:
                if not self.safe_to_break(tx, y, tz):
                    return
                digs.append(self._dig(tx, y, tz))
            places.append((tx, y, tz))
            cost += self.settings.place_cost
            c_height += 1
        if c_height - self._height(x, y - 1, z) > JUMP_HEIGHT:
            return
        for cx, cy, cz in ((x, y + 2, z), (tx, y + 2, tz), (tx, y + 1, tz)):  # blocks A, H, B
            cost += self._safe_or_break(cx, cy, cz, digs)
            if cost > MAX_MOVE_COST:
                return
        out.append((cost, Step(Move.ASCEND, (tx, y + 1, tz), tuple(digs), tuple(places)), blocks - len(places)))

    def _drop_down(self, x: int, y: int, z: int, dx: int, dz: int, blocks: int, out: list[Neighbour]) -> None:
        tx, tz = x + dx, z + dz
        land = self._landing(x, y, z, dx, dz)
        if land is None:
            return
        if not self.settings.infinite_liquid_dropdown_distance and y - land[1] > self.settings.max_drop_down:
            return
        cost, digs = 1.0, []
        for cy in (y + 1, y, y - 1):  # blocks B, C, D
            cost += self._safe_or_break(tx, cy, tz, digs)
            if cost > MAX_MOVE_COST:
                return
        if self._liq[self.cls(tx, y, tz)]:
            return  # the package does not step off into water at head level
        out.append((cost, Step(Move.DROP, land, tuple(digs), fall=y - land[1]), blocks))

    def _diagonal(self, x: int, y: int, z: int, dx: int, dz: int, blocks: int, out: list[Neighbour]) -> None:
        tx, tz = x + dx, z + dz
        cost = math.sqrt(2)
        up = 1 if self._phys[self.cls(tx, y, tz)] else 0
        floor0 = self._height(x, y - 1, z)
        sides = []
        for sx, sz in ((x, tz), (tx, z)):  # the z side first, as the package
            c, digs = 0.0, []
            c += self._safe_or_break(sx, y + up + 1, sz, digs)
            c += self._safe_or_break(sx, y + up, sz, digs)
            if self._height(sx, y + up - 1, sz) - floor0 > JUMP_HEIGHT:
                c += self._safe_or_break(sx, y + up - 1, sz, digs)
            sides.append((c, digs))
        (c1, d1), (c2, d2) = sides
        digs = list(d1) if c1 < c2 else list(d2)
        cost += c1 if c1 < c2 else c2
        if cost > MAX_MOVE_COST:
            return
        for cy in (y + up, y + up + 1):
            cost += self._safe_or_break(tx, cy, tz, digs)
            if cost > MAX_MOVE_COST:
                return
        if self._liq[self.cls(x, y, z)]:
            cost += self.settings.liquid_cost
        if up:
            if self._height(tx, y, tz) - floor0 > JUMP_HEIGHT:
                return
            cost += self._safe_or_break(x, y + 2, z, digs)
            if cost > MAX_MOVE_COST:
                return
            cost += 1
            out.append((cost, Step(Move.DIAGONAL, (tx, y + 1, tz), tuple(digs)), blocks))
            return
        cd = self.cls(tx, y - 1, tz)
        if self._phys[cd] or self._liq[self.cls(tx, y, tz)]:
            out.append((cost, Step(Move.DIAGONAL, (tx, y, tz), tuple(digs)), blocks))
        elif self._phys[self.cls(tx, y - 2, tz)] or self._liq[cd]:
            if not self._safe[cd]:
                return
            out.append((cost, Step(Move.DIAGONAL, (tx, y - 1, tz), tuple(digs), fall=1), blocks))

    def _down(self, x: int, y: int, z: int, blocks: int, out: list[Neighbour]) -> None:
        land = self._landing(x, y, z, 0, 0)
        if land is None:
            return
        digs: list[Dig] = []
        cost = 1.0 + self._safe_or_break(x, y - 1, z, digs)
        if cost > MAX_MOVE_COST:
            return
        if self._liq[self.cls(x, y, z)]:
            return
        out.append((cost, Step(Move.DOWN, land, tuple(digs), fall=y - land[1]), blocks))

    def _up(self, x: int, y: int, z: int, blocks: int, out: list[Neighbour]) -> None:
        c1 = self.cls(x, y, z)
        if self._liq[c1]:
            return
        digs: list[Dig] = []
        places: list[tuple[int, int, int]] = []
        cost = 1.0 + self._safe_or_break(x, y + 2, z, digs)
        if cost > MAX_MOVE_COST:
            return
        if not self._climb[c1]:
            if not self.settings.allow_1by1_towers or blocks == 0 or not self._placeable(x, y, z):
                return
            if not self._repl[c1]:
                if not self.safe_to_break(x, y, z):
                    return
                digs.append(self._dig(x, y, z))
            if self._phys[self.cls(x, y - 1, z)] and self._height(x, y - 1, z) - y < HALF_BLOCK_JUMP:
                return
            places.append((x, y, z))
            cost += self.settings.place_cost
        if cost > MAX_MOVE_COST:
            return
        move = Move.PILLAR if places else Move.CLIMB
        out.append((cost, Step(move, (x, y + 1, z), tuple(digs), tuple(places)), blocks - len(places)))


# ---------------------------------------------------------------------------
# A* (lib/astar.js with lib/heap.js)
# ---------------------------------------------------------------------------


class _Node:
    __slots__ = ("pos", "g", "h", "f", "parent", "step", "blocks", "index")

    def __init__(self, pos: tuple[int, int, int], g: float, h: float, parent: _Node | None, step: Step | None,
                 blocks: int):
        self.pos, self.g, self.h, self.f = pos, g, h, g + h
        self.parent, self.step, self.blocks = parent, step, blocks
        self.index = 0


class _Heap:
    """The package's ``BinaryHeapOpenSet`` (1-based, ordered by f), kept exactly so that ties are
    broken the same way, including ``pop``'s ``smallerChild < size - 1`` test."""

    def __init__(self) -> None:
        self.a: list[_Node | None] = [None]

    def __len__(self) -> int:
        return len(self.a) - 1

    def _sift_up(self, cur: int) -> None:
        a = self.a
        par = cur >> 1
        while cur > 1 and a[par].f > a[cur].f:
            a[par], a[cur] = a[cur], a[par]
            a[par].index, a[cur].index = par, cur
            cur, par = par, par >> 1

    def push(self, node: _Node) -> None:
        self.a.append(node)
        node.index = len(self.a) - 1
        self._sift_up(node.index)

    def update(self, node: _Node) -> None:
        self._sift_up(node.index)

    def pop(self) -> _Node:
        a = self.a
        smallest = a[1]
        a[1] = a[-1]
        a.pop()
        size = len(a) - 1
        if size >= 1:
            a[1].index = 1
        if size < 2:
            return smallest
        val, index, child = a[1], 1, 2
        while True:
            node = a[child]
            if child < size - 1:
                right = a[child + 1]
                if node.f > right.f:
                    child += 1
                    node = right
            if val.f <= node.f:
                break
            a[index], a[child] = node, val
            node.index, val.index = index, child
            index = child
            child *= 2
            if child > size:
                break
        return smallest


def find_path(world: World, start: Sequence[float], goal: Goal, *, held: int = 0, tools: Sequence[int] = (),
              blocks: int = 0, allow_dig: bool = True, allow_pillar: bool = True, node_budget: int = NODE_BUDGET,
              search_radius: float | None = None, settings: Settings = DEFAULT_SETTINGS,
              model: GotoModel = GotoModel()) -> NavResult:
    """A* from the feet cell of ``start`` to ``goal`` (``astar.js``).

    held, tools: item-class ids the agent can dig with (0 = hand); blocks: scaffolding items it
    carries (``SCAFFOLDING_ITEMS``); allow_dig and allow_pillar override ``canDig`` and
    ``allow1by1towers``, search_radius ``searchRadius`` (a negative value means unbounded).
    Returns ``OK`` with a ``Path``, ``NO_PATH`` when every node within the cost bound is
    exhausted, or ``PATH_TIMEOUT`` once ``node_budget`` nodes are visited without reaching the goal.
    """
    settings = replace(settings, can_dig=settings.can_dig and allow_dig,
                       allow_1by1_towers=settings.allow_1by1_towers and allow_pillar,
                       search_radius=settings.search_radius if search_radius is None else search_radius)
    grid = NavGrid(world, held=held, tools=tools, settings=settings)
    s = (math.floor(start[0]), math.floor(start[1]), math.floor(start[2]))
    first = _Node(s, 0.0, goal.heuristic(*s), None, None, int(blocks))
    max_cost = first.h + settings.search_radius if settings.search_radius >= 0 else -1.0
    heap = _Heap()
    heap.push(first)
    open_nodes: dict[tuple[int, int, int], _Node] = {s: first}
    closed: set[tuple[int, int, int]] = set()
    while len(heap):
        if len(closed) >= node_budget:
            return NavResult(PATH_TIMEOUT, None, len(closed))
        node = heap.pop()
        if goal.is_end(*node.pos):
            return NavResult(OK, _build(s, node, len(closed), model), len(closed))
        del open_nodes[node.pos]
        closed.add(node.pos)
        for cost, step, left in grid.neighbours(*node.pos, node.blocks):
            pos = step.pos
            if pos in closed:
                continue
            g = node.g + cost
            h = goal.heuristic(*pos)
            if max_cost > 0 and g + h > max_cost:
                continue
            other = open_nodes.get(pos)
            if other is None:
                other = _Node(pos, g, h, node, step, left)
                open_nodes[pos] = other
                heap.push(other)
            elif other.g < g:
                continue
            else:
                other.g, other.h, other.f = g, h, g + h
                other.parent, other.step, other.blocks = node, step, left
                heap.update(other)
    return NavResult(NO_PATH, None, len(closed))


def _build(start: tuple[int, int, int], end: _Node, visited: int, model: GotoModel) -> Path:
    steps = []
    node: _Node | None = end
    while node is not None and node.step is not None:
        steps.append(node.step)
        node = node.parent
    return Path(start, tuple(reversed(steps)), end.g, visited, model)

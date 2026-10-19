"""The reference world: voxels, light, biomes, entity slots, clock and chunk ticking (contracts 5).

DESIGN.md 4.4 (world model), 4.2 (what is exact) and 4.5 (tick model). Usage::

    from hivemind.sim.ref.world import World
    world = World(128, 128)                     # all air; worldgen.generate() fills one
    world.set(10, 64, 10, stone_id)             # tracked: world.changes_since(cursor)
    world.get(-1, 64, 0)                        # outside the edge: the barrier class
    world.block_light_at(x, y, z), world.sky_light(x, y, z)
    world.biome_at(10.5, 7.2)                   # biome id of the column (a biome class, contracts 3)
    slot = world.alloc_agent((5.5, 64.0, 7.5), yaw=180.0)  # home defaults to the spawn position
    slot = world.mobs.alloc(cls=zombie_id, pos=(5.5, 64.0, 7.5))
    world.ticking_chunks()                      # bool[chunks_x, chunks_z]

**Voxels.** One byte per cell holding a block-class id (contracts 3): ``blocks[x, y - MIN_Y, z]``
for x in [0, size_x), y in [-64, 191] (256 levels, DESIGN.md 4.4 D3), z in [0, size_z). Cells
beside the box (x or z outside) and below it read as ``barrier`` (hardness -1, unbreakable): the
world edge, matching the barrier walls of the real-game evaluation areas (DESIGN.md 11.3). Cells
above it (y > 191, inside x and z) read as air: start areas keep their terrain at or below y 191
(DESIGN.md 4.4) and the game has air above, up to y 319; nothing can be placed there. The class
flags (solid, transparent, liquid, hazard, collision height, climbable, gravity, replaceable,
light) come from ``load_concepts()`` and are shared by perception and the nav grid.

**Change tracking.** ``set`` appends a ``BlockChange`` to a log. Consumers (nav caches,
perception caches, the viewer) keep a cursor: ``changes = world.changes_since(cursor)``, then
``cursor = world.change_count``. ``load`` (bulk replace, used by worldgen and resets) clears the
log and bumps ``generation`` so consumers know to rebuild. A cursor is only meaningful for the
world (and generation) it came from: ``changes_since`` raises for a cursor past ``change_count``
rather than reading it as "no changes". Caches tell worlds apart by ``serial``, a number unique
to each ``World`` object in the process (a copy gets a new one), never by ``id()``, which
CPython reuses once an object is freed.

**Light** (DESIGN.md 4.2: approximate) uses each class's *light filter*: minecraft-data's
``filter_light`` of its members (``light_filter_table``), the game's own light data. It is 0 for
air, glass, torches, plants, snow layers, slabs, stairs, fences, carpets, chests, farmland and dirt
paths; 1 for water, lava, leaves, ice and cobwebs; 15 for full blocks and tinted glass. The
rendering flag ``transparent`` is not used for light (perception builds its see-through table
from it, ``perception.see_through_table``, which also takes out doors, beds and other blocks
that hide what is behind them): a snow layer is not transparent but lets all light through. Fallback classes whose members differ take
the filter of most of their states.

- Block light is a 4-bit flood from light-emitting classes: light entering a cell with filter
  f < 15 loses max(1, f) levels, and filter-15 cells stop it (the light engine's
  ``level - max(1, lightBlock)`` [mem]); so water and leaves pass block light like air.
- Sky light falls straight down each column: a cell has 15 minus the filters of every cell from
  the top of the world down to and including it, and 0 at or below the column's highest filter-15
  cell (the heightmap). Water and leaves thus dim it by 1 per block [mem]. There is no sideways
  spread under overhangs (the real game spreads it [mem]).
- Blocks the game occludes by shape (a bottom slab's floor face) pass light like air here [mem].

The day-night level of sky light is ``mechanics.surface_sky_light``; the arrays here hold the raw
value. ``set`` relights only the box of cells within 15 of the change (a change cannot alter light
farther away), which gives exactly the result of a full recomputation (tested).
``light_points`` reads both kinds of light at many cells at once (the panorama and the F3 group).

**Biomes** (draft 0.5; DESIGN.md 4.4, contracts 3): one biome id per column,
``biome_map[x, z]`` (uint8, the vanilla registry id of a biome class), ``plains`` unless set
[design] (``set_biome_map``; worldgen assigns a true biome per column). The M3 importer will keep
the game's 3-D biome cells instead.

**Entities** live in fixed slot tables with in-use flags (DESIGN.md 4.4; Craftax caps entity
counts the same way [40]): agents <= 40, mobs <= 32, dropped items <= 64, furnaces <= 16, and
chests <= 8 per agent slot (320; contracts 4.3: ``my chest 1``-``8``; the chest table is sized
from the agent count [design], used from M9). Each table is a structure of arrays; the modules
that own a kind (mobs, inventory, actions) may add fields with ``SlotTable.add_field``.

The agent table holds, besides position, health, team, number and the alive flag, what the F3
group reads (``debug_info.py``; DESIGN.md 6.2): ``yaw`` and ``pitch`` (float32 degrees in
Minecraft's convention [mem]: yaw 0 faces south (+z), 90 west, 180 north, -90 east; pitch -90
looks straight up, +90 straight down; the look vector is (-sin yaw cos pitch, -sin pitch,
cos yaw cos pitch)), and ``home`` (float64 x, y, z: the agent's spawn point, or from M9 its
team's base, where ``go home`` leads). ``home`` starts as NaN, so a slot whose home was never set
cannot leak absolute coordinates; ``alloc_agent`` sets it to the spawn position by default.

**Clock.** ``tick`` counts game ticks since the episode began; ``day_time`` is the game's day
time (``/time query daytime`` [mem]), which the night skip can move forward without ticking.

**Chunk ticking** (DESIGN.md 4.2, 11.3; [72] and [mem]). Mobs, dropped items and (probably [mem])
furnaces advance only in ticking chunks. The rule: with simulation distance d (10 on the frozen
server, DESIGN.md 11.2), the 16x16 chunk column (cx, cz) ticks when
``max(|cx - px|, |cz - pz|) <= d`` for the chunk (px, pz) that contains some agent, i.e. a square
of (2d + 1)^2 = 441 chunks, +-160 blocks around the agent's chunk [72]. The Chebyshev square
comes from 1.18+ chunk-ticket propagation, which spreads to all 8 neighbours [mem]; whether the
outermost ring also ticks entities is checked by the harness's chunk-ticking report in M2
(DESIGN.md 11.3). Every agent slot in use holds a ticket, dead or alive: a dead player stays in
the level, at the place it died, until it respawns [mem]; the M2 report checks this too. A chunk
index is ``floor(x / 16)``; the world's chunks start at x = z = 0.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from hivemind.rules import load_concepts, load_rules
from hivemind.sim.ref.mechanics import time_of_day

if TYPE_CHECKING:
    from hivemind.rules.classes import Concepts

MIN_Y = -64  # DESIGN.md 4.4 (D3): y trimmed to -64..191
HEIGHT = 256  # DESIGN.md 4.4: 256 levels
MAX_Y = MIN_Y + HEIGHT - 1  # 191
CHUNK_SIZE = 16  # [mem] chunk columns are 16 x 16 blocks
SIMULATION_DISTANCE = 10  # DESIGN.md 11.2: simulation-distance 10, the Java multiplayer default [72]
MAX_LIGHT = 15  # [mem] light levels 0..15 (4 bits, DESIGN.md 4.4)
DEFAULT_SIZE = 128  # DESIGN.md 4.4: a 128 x 128 single-agent world
AIR = 0  # contracts 3: class 0 is air
UNSEEN = 255  # contracts 3: class 255 is "unseen" (perception only, never stored in a world)
DEFAULT_BIOME = "plains"  # [design] the biome of every column until a biome map is set

CHESTS_PER_AGENT = 8  # [design] DESIGN.md 4.4: from M9 chests <= 8 per agent (contracts 4.3: my chest 1-8)
# Fixed entity slot capacities (DESIGN.md 4.4); chests scale with the agent slots.
ENTITY_CAPACITY: dict[str, int] = {"agents": 40, "mobs": 32, "items": 64, "furnaces": 16,
                                   "chests": CHESTS_PER_AGENT * 40}
CHEST_SLOTS = 27  # [mem] a single chest holds 27 stacks
_WORLD_SERIALS = itertools.count(1)  # World.serial: one number per World object in the process


class SlotsFull(RuntimeError):
    """Raised by ``SlotTable.alloc`` when every slot is in use."""


@dataclass(frozen=True)
class FieldSpec:
    dtype: Any
    shape: tuple[int, ...] = ()
    fill: Any = 0


class SlotTable:
    """A fixed-capacity entity table: one NumPy array per field, plus ``in_use`` flags.

    ``alloc`` takes the lowest free slot (deterministic), resets every field to its fill value
    and sets the given ones; ``free`` clears the flag. Fields are attributes: ``table.pos[i]``.
    """

    def __init__(self, name: str, capacity: int, fields: Mapping[str, FieldSpec]):
        self.name = name
        self.capacity = capacity
        self.in_use = np.zeros(capacity, bool)
        self._fields: dict[str, FieldSpec] = {}
        for field_name, spec in fields.items():
            self.add_field(field_name, spec.dtype, spec.shape, spec.fill)

    def __repr__(self) -> str:
        return f"SlotTable({self.name!r}, {len(self)}/{self.capacity}, fields={list(self._fields)})"

    def __len__(self) -> int:
        return int(self.in_use.sum())

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self._fields)

    def add_field(self, name: str, dtype: Any, shape: tuple[int, ...] = (), fill: Any = 0) -> None:
        """Add a field (an array of ``capacity`` rows of ``shape``) filled with ``fill``."""
        if name in self._fields or name in ("name", "capacity", "in_use"):
            raise ValueError(f"{self.name}: field {name!r} exists")
        self._fields[name] = FieldSpec(dtype, tuple(shape), fill)
        setattr(self, name, np.full((self.capacity, *shape), fill, dtype))

    def alloc(self, **values: Any) -> int:
        """Use the lowest free slot; returns its index. Raises ``SlotsFull``."""
        free = np.flatnonzero(~self.in_use)
        if free.size == 0:
            raise SlotsFull(f"all {self.capacity} {self.name} slots are in use")
        i = int(free[0])
        for field_name, spec in self._fields.items():
            getattr(self, field_name)[i] = spec.fill
        for field_name, value in values.items():
            if field_name not in self._fields:
                raise KeyError(f"{self.name} has no field {field_name!r}")
            getattr(self, field_name)[i] = value
        self.in_use[i] = True
        return i

    def free(self, i: int) -> None:
        if not self.in_use[i]:
            raise ValueError(f"{self.name} slot {i} is not in use")
        self.in_use[i] = False

    def active(self) -> np.ndarray:
        """Indices of the slots in use, ascending."""
        return np.flatnonzero(self.in_use)

    def copy(self) -> SlotTable:
        t = SlotTable(self.name, self.capacity, {})
        t._fields = dict(self._fields)
        t.in_use = self.in_use.copy()
        for field_name in self._fields:
            setattr(t, field_name, getattr(self, field_name).copy())
        return t


def _entity_tables(capacity: Mapping[str, int]) -> dict[str, SlotTable]:
    """The five slot tables with their basic fields (other modules may add more)."""
    pos = FieldSpec(np.float64, (3,), 0.0)
    ipos = FieldSpec(np.int32, (3,), 0)
    specs: dict[str, dict[str, FieldSpec]] = {
        # yaw and pitch: Minecraft's convention (module docstring); home: NaN until set.
        "agents": {"pos": pos, "yaw": FieldSpec(np.float32), "pitch": FieldSpec(np.float32),
                   "home": FieldSpec(np.float64, (3,), np.nan),
                   "health": FieldSpec(np.float32, (), 20.0),  # [mem] max health 20
                   "team": FieldSpec(np.int8), "number": FieldSpec(np.int16), "alive": FieldSpec(bool, (), True)},
        "mobs": {"cls": FieldSpec(np.int16), "pos": pos, "health": FieldSpec(np.float32),
                 "target": FieldSpec(np.int16, (), -1), "age": FieldSpec(np.int32)},
        "items": {"item": FieldSpec(np.int16), "count": FieldSpec(np.int16), "pos": pos, "age": FieldSpec(np.int32),
                  "pickup_delay": FieldSpec(np.int16), "thrower": FieldSpec(np.int16, (), -1)},
        "furnaces": {"pos": ipos, "owner": FieldSpec(np.int16, (), -1),
                     "input": FieldSpec(np.int16), "input_count": FieldSpec(np.int16),
                     "fuel": FieldSpec(np.int16), "fuel_count": FieldSpec(np.int16),
                     "output": FieldSpec(np.int16), "output_count": FieldSpec(np.int16),
                     "burn_left": FieldSpec(np.int32), "burn_total": FieldSpec(np.int32),
                     "cook_ticks": FieldSpec(np.int32)},
        "chests": {"pos": ipos, "owner": FieldSpec(np.int16, (), -1),
                   "slots": FieldSpec(np.int16, (CHEST_SLOTS, 2), 0)},  # (item class, count) per slot
    }
    return {kind: SlotTable(kind, capacity[kind], specs[kind]) for kind in specs}


@lru_cache(maxsize=8)
def light_filter_table(concepts: Concepts) -> np.ndarray:
    """uint8[256]: each block class's light filter, minecraft-data's ``filter_light`` (0..15) of
    most of its block states (ties: the lowest state id). Curated classes are uniform; 17 fallback
    classes mix 0 with 1 or 15. Ids that are not block classes get 15."""
    rules = load_rules()
    votes: dict[int, dict[int, int]] = {}
    first: dict[tuple[int, int], int] = {}
    for b in rules.blocks.values():
        for state in range(b.min_state, b.max_state + 1):
            cls = int(concepts.state_class[state])
            votes.setdefault(cls, {})
            votes[cls][b.filter_light] = votes[cls].get(b.filter_light, 0) + 1
            first.setdefault((cls, b.filter_light), state)
    table = np.full(UNSEEN + 1, MAX_LIGHT, np.uint8)
    for cls, counts in votes.items():
        table[cls] = max(counts, key=lambda f: (counts[f], -first[(cls, f)]))
    table.setflags(write=False)
    return table


@dataclass(frozen=True)
class BlockChange:
    """One tracked ``set``: position, old and new class, and the world tick."""

    x: int
    y: int
    z: int
    old: int
    new: int
    tick: int


class World:
    """Voxels, light, heightmap, entity slots, clock and chunk ticking (see the module docstring)."""

    def __init__(self, size_x: int = DEFAULT_SIZE, size_z: int = DEFAULT_SIZE, *, concepts: Concepts | None = None,
                 day_time: int = 0, entity_capacity: Mapping[str, int] | None = None):
        if size_x <= 0 or size_z <= 0:
            raise ValueError("world size must be positive")
        self.concepts = concepts if concepts is not None else load_concepts()
        c = self.concepts
        self.size_x, self.size_z = int(size_x), int(size_z)
        self.barrier = c.block_class("barrier").id
        # Class property tables (indexed by class id), shared with perception and nav.
        self.transparent: np.ndarray = c.transparent  # rendering flag, not used for light; not sight either
        # (perception.see_through_table is the sight table)
        self.light_filter: np.ndarray = light_filter_table(c)
        self.emission: np.ndarray = c.light
        self.blocks = np.zeros((self.size_x, HEIGHT, self.size_z), np.uint8)
        self.block_light = np.zeros_like(self.blocks)
        self.heightmap = np.full((self.size_x, self.size_z), MIN_Y - 1, np.int16)
        self.biome_map = np.full((self.size_x, self.size_z), c.biome(DEFAULT_BIOME).id, np.uint8)
        self.tick = 0
        self.day_time = int(day_time)
        self.serial = next(_WORLD_SERIALS)  # unique per World object (module docstring)
        self.generation = 0
        self._log: list[BlockChange] = []
        self._log_offset = 0
        cap = dict(ENTITY_CAPACITY)
        cap.update(entity_capacity or {})
        if "chests" not in (entity_capacity or {}):
            cap["chests"] = CHESTS_PER_AGENT * cap["agents"]
        tables = _entity_tables(cap)
        self.agents: SlotTable = tables["agents"]
        self.mobs: SlotTable = tables["mobs"]
        self.items: SlotTable = tables["items"]
        self.furnaces: SlotTable = tables["furnaces"]
        self.chests: SlotTable = tables["chests"]
        self.meta: dict[str, Any] = {}  # free-form provenance (worldgen seed, preset, spawn point)

    def __repr__(self) -> str:
        return f"World({self.size_x}x{self.size_z}, tick={self.tick}, day_time={self.day_time})"

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Unpickling, ``copy.copy`` and ``copy.deepcopy`` make another object: a new ``serial``."""
        self.__dict__.update(state)
        self.serial = next(_WORLD_SERIALS)

    # -- geometry ---------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.blocks.shape  # type: ignore[return-value]

    @property
    def chunks_x(self) -> int:
        return -(-self.size_x // CHUNK_SIZE)

    @property
    def chunks_z(self) -> int:
        return -(-self.size_z // CHUNK_SIZE)

    def in_bounds(self, x: int, y: int, z: int) -> bool:
        return 0 <= x < self.size_x and MIN_Y <= y <= MAX_Y and 0 <= z < self.size_z

    # -- voxels -----------------------------------------------------------------------

    def get(self, x: int, y: int, z: int) -> int:
        """The block class at a cell; ``barrier`` beside and below the world, air above it."""
        if 0 <= x < self.size_x and 0 <= z < self.size_z:
            if MIN_Y <= y <= MAX_Y:
                return int(self.blocks[x, y - MIN_Y, z])
            if y > MAX_Y:
                return AIR
        return self.barrier

    def set(self, x: int, y: int, z: int, cls: int) -> int:
        """Set a cell's class (tracked); returns the old class. The edge cannot be changed."""
        if not self.in_bounds(x, y, z):
            raise IndexError(f"({x}, {y}, {z}) is outside the world")
        if not 0 <= cls < UNSEEN or not self.concepts.block_defined[cls]:
            raise ValueError(f"{cls} is not a block class a world can hold")
        old = int(self.blocks[x, y - MIN_Y, z])
        if old == cls:
            return old
        self.blocks[x, y - MIN_Y, z] = cls
        self._log.append(BlockChange(x, y, z, old, int(cls), self.tick))
        self._update_heightmap(x, y, z, cls)
        if self.emission[old] != self.emission[cls] or self.light_filter[old] != self.light_filter[cls]:
            self._relight_box(x - MAX_LIGHT, y - MAX_LIGHT, z - MAX_LIGHT,
                              x + MAX_LIGHT + 1, y + MAX_LIGHT + 1, z + MAX_LIGHT + 1)
        return old

    def region(self, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int, *, light: bool = False) -> np.ndarray:
        """Classes of the half-open box [x0, x1) x [y0, y1) x [z0, z1), indexed [x, y - y0, z],
        read as ``get`` does (barrier beside and below the world, air above it), or block light
        (0 outside the world) with ``light=True``."""
        src = self.block_light if light else self.blocks
        out = np.full((x1 - x0, y1 - y0, z1 - z0), 0 if light else self.barrier, np.uint8)
        ax0, ax1 = max(x0, 0), min(x1, self.size_x)
        ay0, ay1 = max(y0, MIN_Y), min(y1, MAX_Y + 1)
        az0, az1 = max(z0, 0), min(z1, self.size_z)
        if ax0 < ax1 and az0 < az1:
            if ay0 < ay1:
                out[ax0 - x0:ax1 - x0, ay0 - y0:ay1 - y0, az0 - z0:az1 - z0] = \
                    src[ax0:ax1, ay0 - MIN_Y:ay1 - MIN_Y, az0:az1]
            if not light and y1 > MAX_Y + 1:
                out[ax0 - x0:ax1 - x0, max(MAX_Y + 1, y0) - y0:, az0 - z0:az1 - z0] = AIR
        return out

    def load(self, blocks: np.ndarray) -> None:
        """Replace every voxel at once (worldgen, resets): clears the change log, recomputes the
        heightmap and all light, and bumps ``generation``."""
        if blocks.shape != self.blocks.shape:
            raise ValueError(f"expected shape {self.blocks.shape}, got {blocks.shape}")
        if blocks.dtype != np.uint8:
            raise TypeError("voxels must be uint8")
        if not self.concepts.block_defined[blocks].all() or (blocks == UNSEEN).any():
            raise ValueError("voxels hold ids that are not block classes")
        self.blocks = np.ascontiguousarray(blocks).copy()
        self._log.clear()
        self._log_offset = 0
        self.generation += 1
        self.recompute_heightmap()
        self.recompute_light()

    # -- change tracking ----------------------------------------------------------------

    @property
    def change_count(self) -> int:
        """The cursor after the latest change (changes are numbered from 0)."""
        return self._log_offset + len(self._log)

    def changes_since(self, cursor: int) -> list[BlockChange]:
        """Changes numbered ``cursor`` and later. Raises if they were trimmed away, or if the
        cursor lies past ``change_count`` (it came from another world or generation)."""
        if cursor < self._log_offset:
            raise ValueError(f"changes before {self._log_offset} were trimmed (asked for {cursor})")
        if cursor > self.change_count:
            raise ValueError(f"cursor {cursor} is past this world's {self.change_count} changes")
        return self._log[cursor - self._log_offset:]

    def trim_changes(self, cursor: int) -> None:
        """Forget changes numbered below ``cursor`` (every consumer has read them)."""
        cut = max(0, min(cursor, self.change_count) - self._log_offset)
        del self._log[:cut]
        self._log_offset += cut

    # -- heightmap and sky light ----------------------------------------------------------

    def _blocks_sky(self, cls: np.ndarray | int) -> np.ndarray | bool:
        return self.light_filter[cls] >= MAX_LIGHT

    def recompute_heightmap(self) -> None:
        """Highest cell per column whose light filter is 15 (MIN_Y - 1 when there is none)."""
        opaque = self._blocks_sky(self.blocks)
        any_opaque = opaque.any(axis=1)
        top = HEIGHT - 1 - np.argmax(opaque[:, ::-1, :], axis=1)
        self.heightmap = np.where(any_opaque, top + MIN_Y, MIN_Y - 1).astype(np.int16)

    def _update_heightmap(self, x: int, y: int, z: int, cls: int) -> None:
        h = int(self.heightmap[x, z])
        if self._blocks_sky(cls):
            if y > h:
                self.heightmap[x, z] = y
        elif y == h:
            below = np.flatnonzero(self._blocks_sky(self.blocks[x, :y - MIN_Y, z]))
            self.heightmap[x, z] = (below[-1] + MIN_Y) if below.size else MIN_Y - 1

    def sky_light(self, x: int, y: int, z: int) -> int:
        """Raw sky light (module docstring): 15 minus the light filters of the column's cells from
        the top down to this one; 0 at or below the heightmap and outside the world's sides."""
        if not (0 <= x < self.size_x and 0 <= z < self.size_z) or y < MIN_Y:
            return 0
        if y > MAX_Y:
            return MAX_LIGHT
        if y <= self.heightmap[x, z]:
            return 0
        dim = int(self.light_filter[self.blocks[x, y - MIN_Y:, z]].sum(dtype=np.int64))
        return max(MAX_LIGHT - dim, 0)

    def sky_light_region(self, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> np.ndarray:
        """``sky_light`` over a half-open box, indexed like ``region``."""
        out = np.zeros((x1 - x0, y1 - y0, z1 - z0), np.uint8)
        ax0, ax1, az0, az1 = max(x0, 0), min(x1, self.size_x), max(z0, 0), min(z1, self.size_z)
        if ax0 >= ax1 or az0 >= az1:
            return out
        xs, zs = slice(ax0 - x0, ax1 - x0), slice(az0 - z0, az1 - z0)
        lo, hi = max(y0, MIN_Y), min(y1, MAX_Y + 1)
        if lo < hi:
            f = self.light_filter[self.blocks[ax0:ax1, lo - MIN_Y:, az0:az1]].astype(np.int32)
            dim = np.cumsum(f[:, ::-1, :], axis=1)[:, ::-1, :]  # filters from the top down to each cell
            out[xs, lo - y0:hi - y0, zs] = np.clip(MAX_LIGHT - dim[:, :hi - lo, :], 0, MAX_LIGHT)
        if y1 > MAX_Y + 1:
            out[xs, max(MAX_Y + 1, y0) - y0:, zs] = MAX_LIGHT
        return out

    # -- block light -------------------------------------------------------------------------

    def block_light_at(self, x: int, y: int, z: int) -> int:
        if self.in_bounds(x, y, z):
            return int(self.block_light[x, y - MIN_Y, z])
        return 0

    def light(self, x: int, y: int, z: int) -> int:
        """max(block light, raw sky light) of a cell."""
        return max(self.block_light_at(x, y, z), self.sky_light(x, y, z))

    def light_points(self, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(raw sky light, block light) as uint8 arrays for integer cells (n, 3), each exactly
        ``sky_light`` and ``block_light_at`` of its cell, computed for all cells at once."""
        cells = np.asarray(cells, np.int64).reshape(-1, 3)
        x, y, z = cells[:, 0], cells[:, 1], cells[:, 2]
        sky = np.zeros(len(cells), np.uint8)
        block = np.zeros(len(cells), np.uint8)
        inside = (x >= 0) & (x < self.size_x) & (z >= 0) & (z < self.size_z)
        sky[inside & (y > MAX_Y)] = MAX_LIGHT
        col = np.flatnonzero(inside & (y >= MIN_Y) & (y <= MAX_Y))
        if col.size:
            cx, cy, cz = x[col], y[col] - MIN_Y, z[col]
            block[col] = self.block_light[cx, cy, cz]
            lit = cy + MIN_Y > self.heightmap[cx, cz]  # at or below the heightmap: 0
            if lit.any():
                cx, cy, cz, idx = cx[lit], cy[lit], cz[lit], col[lit]
                # Filters from each cell up to the top of its column: a masked row sum.
                f = self.light_filter[self.blocks[cx, :, cz]].astype(np.int32)
                f[np.arange(HEIGHT)[None, :] < cy[:, None]] = 0
                sky[idx] = np.clip(MAX_LIGHT - f.sum(axis=1), 0, MAX_LIGHT)
        return sky, block

    def recompute_light(self) -> None:
        """Block light of the whole world from scratch (the bounding box of all sources)."""
        self.block_light = np.zeros_like(self.blocks)
        src = np.argwhere(self.emission[self.blocks] > 0)
        if src.size == 0:
            return
        lo = src.min(axis=0) - MAX_LIGHT
        hi = src.max(axis=0) + MAX_LIGHT + 1
        self._relight_box(int(lo[0]), int(lo[1]) + MIN_Y, int(lo[2]), int(hi[0]), int(hi[1]) + MIN_Y, int(hi[2]))

    def _relight_box(self, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> None:
        """Recompute block light inside a box (world coordinates, half-open), keeping the light
        of the one-cell ring around it fixed as the boundary condition."""
        x0, x1 = max(x0, 0), min(x1, self.size_x)
        i0, i1 = max(y0 - MIN_Y, 0), min(y1 - MIN_Y, HEIGHT)
        z0, z1 = max(z0, 0), min(z1, self.size_z)
        if x0 >= x1 or i0 >= i1 or z0 >= z1:
            return
        # Padded box: the ring cells keep their current light and are never updated.
        px0, px1 = max(x0 - 1, 0), min(x1 + 1, self.size_x)
        pi0, pi1 = max(i0 - 1, 0), min(i1 + 1, HEIGHT)
        pz0, pz1 = max(z0 - 1, 0), min(z1 + 1, self.size_z)
        cls = self.blocks[px0:px1, pi0:pi1, pz0:pz1]
        inner = np.zeros(cls.shape, bool)
        inner[x0 - px0:x1 - px0, i0 - pi0:i1 - pi0, z0 - pz0:z1 - pz0] = True
        lit = np.where(inner, self.emission[cls], self.block_light[px0:px1, pi0:pi1, pz0:pz1]).astype(np.int16)
        if lit.max() > 0:  # no source and no light at the boundary: the box is dark
            f = self.light_filter[cls]
            lit = propagate_light(lit, inner & (f < MAX_LIGHT), np.maximum(f, 1))
        self.block_light[x0:x1, i0:i1, z0:z1] = lit[x0 - px0:x1 - px0, i0 - pi0:i1 - pi0, z0 - pz0:z1 - pz0]

    # -- clock --------------------------------------------------------------------------------

    def advance(self, ticks: int = 1) -> None:
        """Advance the clock (``tick`` and ``day_time``) by ``ticks``."""
        if ticks < 0:
            raise ValueError("time does not run backwards")
        self.tick += ticks
        self.day_time += ticks

    @property
    def time_of_day(self) -> int:
        """``day_time`` within the day, 0..23999 [15]."""
        return time_of_day(self.day_time)

    # -- biomes ---------------------------------------------------------------------------------

    def biome_at(self, x: float, z: float) -> int:
        """The biome id of the column containing (x, z); outside the world's sides, the nearest
        column's (the barrier edge has no biome of its own)."""
        ix = min(max(math.floor(x), 0), self.size_x - 1)
        iz = min(max(math.floor(z), 0), self.size_z - 1)
        return int(self.biome_map[ix, iz])

    def set_biome_map(self, biomes: np.ndarray) -> None:
        """Replace the biome of every column (uint8 [size_x, size_z] of biome-class ids)."""
        biomes = np.asarray(biomes)
        if biomes.shape != (self.size_x, self.size_z):
            raise ValueError(f"expected shape {(self.size_x, self.size_z)}, got {biomes.shape}")
        if biomes.dtype != np.uint8:
            raise TypeError("biome ids must be uint8")
        if (biomes >= len(self.concepts.biomes)).any():
            raise ValueError("biome map holds ids that are not biome classes")
        self.biome_map = np.ascontiguousarray(biomes).copy()

    # -- agents ---------------------------------------------------------------------------------

    def alloc_agent(self, pos: Sequence[float], *, yaw: float = 0.0, pitch: float = 0.0,
                    home: Sequence[float] | None = None, **fields: Any) -> int:
        """Take the lowest free agent slot at feet position ``pos``, facing ``yaw``/``pitch``
        (degrees, Minecraft's convention), with ``home`` (default: ``pos``, the spawn point).
        Other agent fields (team, number, health, ...) may be given too."""
        if not -90.0 <= pitch <= 90.0:
            raise ValueError(f"pitch must lie in [-90, 90], got {pitch}")
        return self.agents.alloc(pos=pos, yaw=yaw, pitch=pitch, home=pos if home is None else home, **fields)

    # -- chunk ticking ------------------------------------------------------------------------

    @staticmethod
    def chunk_of(x: float, z: float) -> tuple[int, int]:
        """The chunk column containing a position (``floor(x / 16)``)."""
        return math.floor(x) // CHUNK_SIZE, math.floor(z) // CHUNK_SIZE

    def agent_positions(self, alive_only: bool = False) -> np.ndarray:
        """Positions (n, 3) of the agent slots in use (only the live ones with ``alive_only``).
        A dead agent keeps its chunks ticking until it respawns (module docstring, [mem])."""
        idx = self.agents.active()
        if alive_only:
            idx = idx[self.agents.alive[idx]]
        return self.agents.pos[idx]

    def ticking_chunks(self, positions: Iterable[Sequence[float]] | np.ndarray | None = None,
                       distance: int = SIMULATION_DISTANCE) -> np.ndarray:
        """bool[chunks_x, chunks_z]: which of the world's chunk columns tick (``ticking_chunks``),
        for the given (x, y, z) or (x, z) positions, by default every agent slot in use."""
        pts = self.agent_positions() if positions is None else positions
        return ticking_chunks(self.chunks_x, self.chunks_z, pts, distance)

    def is_ticking(self, x: float, z: float, mask: np.ndarray | None = None) -> bool:
        """Whether the chunk containing (x, z) ticks (``mask`` from ``ticking_chunks``)."""
        mask = self.ticking_chunks() if mask is None else mask
        cx, cz = self.chunk_of(x, z)
        return bool(0 <= cx < mask.shape[0] and 0 <= cz < mask.shape[1] and mask[cx, cz])

    # -- copies ----------------------------------------------------------------------------------

    def copy(self) -> World:
        """A deep copy (resets copy a pristine world, DESIGN.md 4.4); the change log starts empty."""
        w = World.__new__(World)
        w.__dict__.update(self.__dict__)
        w.serial = next(_WORLD_SERIALS)  # another object: caches must not take it for this one
        w.blocks = self.blocks.copy()
        w.block_light = self.block_light.copy()
        w.heightmap = self.heightmap.copy()
        w.biome_map = self.biome_map.copy()
        w._log = []
        w._log_offset = 0
        for kind in ("agents", "mobs", "items", "furnaces", "chests"):
            setattr(w, kind, getattr(self, kind).copy())
        w.meta = dict(self.meta)
        return w


def ticking_chunks(chunks_x: int, chunks_z: int, positions: Iterable[Sequence[float]] | np.ndarray,
                   distance: int = SIMULATION_DISTANCE) -> np.ndarray:
    """bool[chunks_x, chunks_z] for a world whose chunks start at x = z = 0.

    A chunk ticks when its Chebyshev chunk distance to the chunk of some position is <=
    ``distance`` ([72][mem], see the module docstring). Positions are (x, y, z) or (x, z).
    """
    pts = np.asarray(list(positions) if not isinstance(positions, np.ndarray) else positions, float)
    mask = np.zeros((chunks_x, chunks_z), bool)
    if pts.size == 0:
        return mask
    for p in pts.reshape(len(pts), -1):
        px, pz = (p[0], p[2]) if len(p) == 3 else (p[0], p[1])
        cx, cz = World.chunk_of(px, pz)
        ax0, ax1 = max(cx - distance, 0), min(cx + distance + 1, chunks_x)
        az0, az1 = max(cz - distance, 0), min(cz + distance + 1, chunks_z)
        if ax0 < ax1 and az0 < az1:
            mask[ax0:ax1, az0:az1] = True
    return mask


def propagate_light(lit: np.ndarray, update: np.ndarray, cost: np.ndarray | int = 1) -> np.ndarray:
    """Spread light: repeat ``lit = max(lit, neighbour - cost)`` on ``update`` cells until stable.

    ``lit`` holds each cell's fixed value (emission, or the boundary light); ``cost`` is the
    loss on entering each cell (>= 1: max(1, light filter)). After at most 14 rounds (light 15
    crosses at most 14 cells) every update cell holds the brightest path value, which is what a
    breadth-first flood gives.
    """
    lit = lit.astype(np.int16)
    cost = np.asarray(cost, np.int16)
    for _ in range(MAX_LIGHT):
        nb = np.zeros_like(lit)
        np.maximum(nb[1:], lit[:-1], out=nb[1:])
        np.maximum(nb[:-1], lit[1:], out=nb[:-1])
        np.maximum(nb[:, 1:], lit[:, :-1], out=nb[:, 1:])
        np.maximum(nb[:, :-1], lit[:, 1:], out=nb[:, :-1])
        np.maximum(nb[:, :, 1:], lit[:, :, :-1], out=nb[:, :, 1:])
        np.maximum(nb[:, :, :-1], lit[:, :, 1:], out=nb[:, :, :-1])
        new = np.where(update, np.maximum(lit, nb - cost), lit)
        if np.array_equal(new, lit):
            break
        lit = new
    return lit.astype(np.uint8)


def flood_light_reference(world: World) -> np.ndarray:
    """Block light by a plain flood from every source (slow; for tests): light entering a cell
    with filter f < 15 loses max(1, f)."""
    from collections import deque

    out = np.zeros_like(world.blocks)
    queue: deque[tuple[int, int, int]] = deque()
    for x, i, z in np.argwhere(world.emission[world.blocks] > 0):
        out[x, i, z] = world.emission[world.blocks[x, i, z]]
        queue.append((int(x), int(i), int(z)))
    sx, sy, sz = out.shape
    while queue:
        x, i, z = queue.popleft()
        level = int(out[x, i, z])
        for dx, di, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nx, ni, nz = x + dx, i + di, z + dz
            if not (0 <= nx < sx and 0 <= ni < sy and 0 <= nz < sz):
                continue
            f = int(world.light_filter[world.blocks[nx, ni, nz]])
            v = level - max(1, f)
            if f < MAX_LIGHT and v > 0 and out[nx, ni, nz] < v:
                out[nx, ni, nz] = v
                queue.append((nx, ni, nz))
    return out

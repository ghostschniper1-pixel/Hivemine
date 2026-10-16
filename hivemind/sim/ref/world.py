"""The reference world: voxels, light, entity slots, clock and chunk ticking (contracts 5).

DESIGN.md 4.4 (world model), 4.2 (what is exact) and 4.5 (tick model). Usage::

    from hivemind.sim.ref.world import World
    world = World(128, 128)                     # all air; worldgen.generate() fills one
    world.set(10, 64, 10, stone_id)             # tracked: world.changes_since(cursor)
    world.get(-1, 64, 0)                        # outside the edge: the barrier class
    world.block_light_at(x, y, z), world.sky_light(x, y, z)
    slot = world.mobs.alloc(cls=zombie_id, pos=(5.5, 64.0, 7.5))
    world.ticking_chunks()                      # bool[chunks_x, chunks_z]

**Voxels.** One byte per cell holding a block-class id (contracts 3): ``blocks[x, y - MIN_Y, z]``
for x in [0, size_x), y in [-64, 191] (256 levels, DESIGN.md 4.4 D3), z in [0, size_z). Every cell
outside that box reads as ``barrier`` (hardness -1, unbreakable): the world edge, matching the
barrier walls of the real-game evaluation areas (DESIGN.md 11.3). The class flags (solid,
transparent, liquid, hazard, collision height, climbable, gravity, replaceable, light) come from
``load_concepts()`` and are shared by perception and the nav grid.

**Change tracking.** ``set`` appends a ``BlockChange`` to a log. Consumers (nav caches,
perception caches, the viewer) keep a cursor: ``changes = world.changes_since(cursor)``, then
``cursor = world.change_count``. ``load`` (bulk replace, used by worldgen and resets) clears the
log and bumps ``generation`` so consumers know to rebuild.

**Light** (DESIGN.md 4.2: approximate). Block light is a 4-bit BFS from light-emitting classes:
a cell's light is max(own emission, brightest transparent-path neighbour - 1), and light passes
only through transparent classes (air, glass, leaves, water, ...). Water and leaves dim light by
1 per block like air [mem: in the game they dim sky light more]. Sky light comes from a heightmap
of the highest non-transparent cell per column: 15 above it, 0 at or below it (no sideways spread
under overhangs; the real game spreads it [mem]). The day-night level of sky light is
``mechanics.surface_sky_light``; the arrays here hold the raw value. ``set`` relights only the box
of cells within 15 of the change (a change cannot alter light farther away), which gives exactly
the result of a full recomputation (tested).

**Entities** live in fixed slot tables with in-use flags (DESIGN.md 4.4; Craftax caps entity
counts the same way [40]): agents <= 40, mobs <= 32, dropped items <= 64, furnaces <= 16, chests
<= 16. Each table is a structure of arrays; the modules that own a kind (mobs, inventory,
actions) may add fields with ``SlotTable.add_field``.

**Clock.** ``tick`` counts game ticks since the episode began; ``day_time`` is the game's day
time (``/time query daytime`` [mem]), which the night skip can move forward without ticking.

**Chunk ticking** (DESIGN.md 4.2, 11.3; [72] and [mem]). Mobs, dropped items and (probably [mem])
furnaces advance only in ticking chunks. The rule: with simulation distance d (10 on the frozen
server, DESIGN.md 11.2), the 16x16 chunk column (cx, cz) ticks when
``max(|cx - px|, |cz - pz|) <= d`` for the chunk (px, pz) that contains some agent, i.e. a square
of (2d + 1)^2 = 441 chunks, +-160 blocks around the agent's chunk [72]. The Chebyshev square
comes from 1.18+ chunk-ticket propagation, which spreads to all 8 neighbours [mem]; whether the
outermost ring also ticks entities is checked by the harness's chunk-ticking report in M2
(DESIGN.md 11.3). A chunk index is ``floor(x / 16)``; the world's chunks start at x = z = 0.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from hivemind.rules import load_concepts
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
BURIED = 255  # contracts 3: class 255 is "buried" (perception only, never stored in a world)

# Fixed entity slot capacities (DESIGN.md 4.4; chests: 16 per world for now, the task's cap).
ENTITY_CAPACITY: dict[str, int] = {"agents": 40, "mobs": 32, "items": 64, "furnaces": 16, "chests": 16}
CHEST_SLOTS = 27  # [mem] a single chest holds 27 stacks


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
        "agents": {"pos": pos, "yaw": FieldSpec(np.float32), "pitch": FieldSpec(np.float32),
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
        self.transparent: np.ndarray = c.transparent
        self.emission: np.ndarray = c.light
        self.blocks = np.zeros((self.size_x, HEIGHT, self.size_z), np.uint8)
        self.block_light = np.zeros_like(self.blocks)
        self.heightmap = np.full((self.size_x, self.size_z), MIN_Y - 1, np.int16)
        self.tick = 0
        self.day_time = int(day_time)
        self.generation = 0
        self._log: list[BlockChange] = []
        self._log_offset = 0
        cap = dict(ENTITY_CAPACITY)
        cap.update(entity_capacity or {})
        tables = _entity_tables(cap)
        self.agents: SlotTable = tables["agents"]
        self.mobs: SlotTable = tables["mobs"]
        self.items: SlotTable = tables["items"]
        self.furnaces: SlotTable = tables["furnaces"]
        self.chests: SlotTable = tables["chests"]
        self.meta: dict[str, Any] = {}  # free-form provenance (worldgen seed, preset, spawn point)

    def __repr__(self) -> str:
        return f"World({self.size_x}x{self.size_z}, tick={self.tick}, day_time={self.day_time})"

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
        """The block class at a cell; ``barrier`` outside the world."""
        if 0 <= x < self.size_x and MIN_Y <= y <= MAX_Y and 0 <= z < self.size_z:
            return int(self.blocks[x, y - MIN_Y, z])
        return self.barrier

    def set(self, x: int, y: int, z: int, cls: int) -> int:
        """Set a cell's class (tracked); returns the old class. The edge cannot be changed."""
        if not self.in_bounds(x, y, z):
            raise IndexError(f"({x}, {y}, {z}) is outside the world")
        if not 0 <= cls < BURIED or not self.concepts.block_defined[cls]:
            raise ValueError(f"{cls} is not a block class a world can hold")
        old = int(self.blocks[x, y - MIN_Y, z])
        if old == cls:
            return old
        self.blocks[x, y - MIN_Y, z] = cls
        self._log.append(BlockChange(x, y, z, old, int(cls), self.tick))
        self._update_heightmap(x, y, z, cls)
        if self.emission[old] != self.emission[cls] or self.transparent[old] != self.transparent[cls]:
            self._relight_box(x - MAX_LIGHT, y - MAX_LIGHT, z - MAX_LIGHT,
                              x + MAX_LIGHT + 1, y + MAX_LIGHT + 1, z + MAX_LIGHT + 1)
        return old

    def region(self, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int, *, light: bool = False) -> np.ndarray:
        """Classes of the half-open box [x0, x1) x [y0, y1) x [z0, z1), indexed [x, y - y0, z],
        with ``barrier`` outside the world (or block light 0 with ``light=True``)."""
        src = self.block_light if light else self.blocks
        out = np.full((x1 - x0, y1 - y0, z1 - z0), 0 if light else self.barrier, np.uint8)
        ax0, ax1 = max(x0, 0), min(x1, self.size_x)
        ay0, ay1 = max(y0, MIN_Y), min(y1, MAX_Y + 1)
        az0, az1 = max(z0, 0), min(z1, self.size_z)
        if ax0 < ax1 and ay0 < ay1 and az0 < az1:
            out[ax0 - x0:ax1 - x0, ay0 - y0:ay1 - y0, az0 - z0:az1 - z0] = \
                src[ax0:ax1, ay0 - MIN_Y:ay1 - MIN_Y, az0:az1]
        return out

    def load(self, blocks: np.ndarray) -> None:
        """Replace every voxel at once (worldgen, resets): clears the change log, recomputes the
        heightmap and all light, and bumps ``generation``."""
        if blocks.shape != self.blocks.shape:
            raise ValueError(f"expected shape {self.blocks.shape}, got {blocks.shape}")
        if blocks.dtype != np.uint8:
            raise TypeError("voxels must be uint8")
        if not self.concepts.block_defined[blocks].all() or (blocks == BURIED).any():
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
        """Changes numbered ``cursor`` and later. Raises if they were trimmed away."""
        if cursor < self._log_offset:
            raise ValueError(f"changes before {self._log_offset} were trimmed (asked for {cursor})")
        return self._log[cursor - self._log_offset:]

    def trim_changes(self, cursor: int) -> None:
        """Forget changes numbered below ``cursor`` (every consumer has read them)."""
        cut = max(0, min(cursor, self.change_count) - self._log_offset)
        del self._log[:cut]
        self._log_offset += cut

    # -- heightmap and sky light ----------------------------------------------------------

    def recompute_heightmap(self) -> None:
        """Highest non-transparent cell per column (MIN_Y - 1 when the column is all transparent)."""
        opaque = ~self.transparent[self.blocks]
        any_opaque = opaque.any(axis=1)
        top = HEIGHT - 1 - np.argmax(opaque[:, ::-1, :], axis=1)
        self.heightmap = np.where(any_opaque, top + MIN_Y, MIN_Y - 1).astype(np.int16)

    def _update_heightmap(self, x: int, y: int, z: int, cls: int) -> None:
        h = int(self.heightmap[x, z])
        if not self.transparent[cls]:
            if y > h:
                self.heightmap[x, z] = y
        elif y == h:
            column = ~self.transparent[self.blocks[x, :y - MIN_Y, z]]
            below = np.flatnonzero(column)
            self.heightmap[x, z] = (below[-1] + MIN_Y) if below.size else MIN_Y - 1

    def sky_light(self, x: int, y: int, z: int) -> int:
        """Raw sky light: 15 above the column's highest non-transparent cell, else 0."""
        if not (0 <= x < self.size_x and 0 <= z < self.size_z):
            return 0
        return MAX_LIGHT if y > self.heightmap[x, z] else 0

    def sky_light_region(self, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> np.ndarray:
        """``sky_light`` over a half-open box, indexed like ``region`` (0 outside the world)."""
        hm = np.full((x1 - x0, z1 - z0), MAX_Y + 1, np.int32)  # outside: never lit
        ax0, ax1, az0, az1 = max(x0, 0), min(x1, self.size_x), max(z0, 0), min(z1, self.size_z)
        if ax0 < ax1 and az0 < az1:
            hm[ax0 - x0:ax1 - x0, az0 - z0:az1 - z0] = self.heightmap[ax0:ax1, az0:az1]
        ys = np.arange(y0, y1)[None, :, None]
        return np.where(ys > hm[:, None, :], MAX_LIGHT, 0).astype(np.uint8)

    # -- block light -------------------------------------------------------------------------

    def block_light_at(self, x: int, y: int, z: int) -> int:
        if self.in_bounds(x, y, z):
            return int(self.block_light[x, y - MIN_Y, z])
        return 0

    def light(self, x: int, y: int, z: int) -> int:
        """max(block light, raw sky light) of a cell."""
        return max(self.block_light_at(x, y, z), self.sky_light(x, y, z))

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
            lit = propagate_light(lit, inner & self.transparent[cls])
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

    # -- chunk ticking ------------------------------------------------------------------------

    @staticmethod
    def chunk_of(x: float, z: float) -> tuple[int, int]:
        """The chunk column containing a position (``floor(x / 16)``)."""
        return math.floor(x) // CHUNK_SIZE, math.floor(z) // CHUNK_SIZE

    def agent_positions(self) -> np.ndarray:
        """Positions (n, 3) of the agents in use and alive."""
        idx = self.agents.active()
        idx = idx[self.agents.alive[idx]]
        return self.agents.pos[idx]

    def ticking_chunks(self, positions: Iterable[Sequence[float]] | np.ndarray | None = None,
                       distance: int = SIMULATION_DISTANCE) -> np.ndarray:
        """bool[chunks_x, chunks_z]: which of the world's chunk columns tick (``ticking_chunks``),
        for the given (x, y, z) or (x, z) positions, by default the live agents'."""
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
        w.blocks = self.blocks.copy()
        w.block_light = self.block_light.copy()
        w.heightmap = self.heightmap.copy()
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


def propagate_light(lit: np.ndarray, update: np.ndarray) -> np.ndarray:
    """Spread light: repeat ``lit = max(lit, neighbour - 1)`` on ``update`` cells until stable.

    ``lit`` holds each cell's fixed value (emission, or the boundary light); after at most 14
    rounds (light 15 reaches 14 cells) every update cell holds the brightest path value, which is
    what a breadth-first flood gives.
    """
    lit = lit.astype(np.int16)
    for _ in range(MAX_LIGHT):
        nb = np.zeros_like(lit)
        np.maximum(nb[1:], lit[:-1], out=nb[1:])
        np.maximum(nb[:-1], lit[1:], out=nb[:-1])
        np.maximum(nb[:, 1:], lit[:, :-1], out=nb[:, 1:])
        np.maximum(nb[:, :-1], lit[:, 1:], out=nb[:, :-1])
        np.maximum(nb[:, :, 1:], lit[:, :, :-1], out=nb[:, :, 1:])
        np.maximum(nb[:, :, :-1], lit[:, :, 1:], out=nb[:, :, :-1])
        new = np.where(update, np.maximum(lit, nb - 1), lit)
        if np.array_equal(new, lit):
            break
        lit = new
    return lit.astype(np.uint8)


def flood_light_reference(world: World) -> np.ndarray:
    """Block light by a plain breadth-first flood from every source (slow; for tests)."""
    from collections import deque

    out = np.zeros_like(world.blocks)
    queue: deque[tuple[int, int, int]] = deque()
    for x, i, z in np.argwhere(world.emission[world.blocks] > 0):
        out[x, i, z] = world.emission[world.blocks[x, i, z]]
        queue.append((int(x), int(i), int(z)))
    sx, sy, sz = out.shape
    while queue:
        x, i, z = queue.popleft()
        v = int(out[x, i, z]) - 1
        if v <= 0:
            continue
        for dx, di, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nx, ni, nz = x + dx, i + di, z + dz
            if 0 <= nx < sx and 0 <= ni < sy and 0 <= nz < sz and world.transparent[world.blocks[nx, ni, nz]] \
                    and out[nx, ni, nz] < v:
                out[nx, ni, nz] = v
                queue.append((nx, ni, nz))
    return out

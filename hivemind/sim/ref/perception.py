"""Perception: what an agent may know about the blocks around it (DESIGN.md 6.1; contracts 5, 6.1).

This is the **only** source the notebook and the action masks may use for blocks. The bridge
runs the same functions on the blocks Mineflayer holds (bridge/PROTOCOL.md 7.2), so the
simulator and the real game perceive alike. Usage::

    from hivemind.sim.ref.perception import perceive, voxel_window, sightings, SightingCache
    cache = SightingCache()                                  # one per agent (optional, exact)
    p = perceive(world, feet=(x, y, z), tick=world.tick, cache=cache)  # window always, rays when due
    p.window.classes            # uint8 (15, 7, 15) [x, y, z]: class ids, 255 = buried
    p.sightings.counts          # {class id: sighted blocks}; None when the rays were not due
    sightings(world, eye, xray=True)    # ablation only: flagged ``xray`` in the result

Definitions (DESIGN.md 6.1):

- **See-through cells:** classes with the ``transparent`` flag (air, glass, leaves, water,
  torches, short grass, ...), except ``barrier`` and ``lava``. The barrier stands for the world
  edge (DESIGN.md 11.3 walls the evaluation areas with it), and nothing beyond it can look back
  in; lava is flagged transparent in the data, but no player sees through a lava pool [mem]. The
  bridge runs this same function, so it applies the same rule; its world mirror must be clipped to
  the evaluation area with barrier outside (bridge/PROTOCOL.md 7.2), or the window would show
  real terrain beyond the wall where the simulator shows ``buried``. Above y 191 the world reads
  as air (``World.get``), as in the game.
- **Exposed:** a cell is exposed when at least one of its six faces touches a see-through cell.
- **Voxel window:** the 15 x 15 x 7 cells around the agent's feet cell (x and z -7..+7, y -3..+3),
  indexed [x, y, z] like the world. Exposed cells show their class, others the reserved class
  255 ``buried``. This lets a little knowledge through walls at short range (a stone next to a
  hidden cave shows as stone), the same on both sides. Light per cell: max(block light, raw sky
  light) for exposed cells, 0 for buried ones.
- **Line of sight:** a ray from the eye (feet + 1.62 [mem]) to the centre of an exposed face that
  faces the eye; every cell the ray crosses before reaching the face must be see-through. The
  cells are walked with a 3-D DDA (Amanatides and Woo); when the ray crosses two cell planes at
  once (an edge or a corner), it steps x before y before z, a fixed rule both sides share. A cube
  has at most 3 faces facing a point outside it, so a block costs at most 3 rays.
- **Resource sightings** (bounded; the definition DESIGN.md 6.1 should state in place of "a
  fixed set of rays [est]"): the candidates are the exposed blocks of resource classes
  (``concepts.resource``) whose centres lie within 24 blocks [est] of the eye, taken nearest
  first (ties: x, then y, then z). A candidate whose class already has ``PER_CLASS`` = 8 [est]
  sightings is skipped without rays. Otherwise its facing open faces are traced (at most 3 rays
  of at most ~43 cell steps), and it is sighted if any ray is clear. The notebook keeps 48 places
  (DESIGN.md 6.3), so 8 positions per class are plenty. Without the cap (``per_class=None``)
  every sighted block is reported: at the spawn of a 128 x 128 test world that is 1,700-2,600
  blocks and 4,400-6,500 rays per call, too much for the M4 budget of <= 800 us per decision
  (DESIGN.md 4.6). With the cap, spawns cost 8-310 rays, and 75 random standing spots in generated
  test worlds (surface and caves) a median of 210 rays (90th percentile 500, maximum 989). The
  cost is not capped further: blocks exposed only to a sealed cave cost rays without being seen,
  so any ray budget would let hidden cells decide which visible blocks are reported. ``rays``
  in the result is a cost figure for profiling, never an input to the notebook or observation.
- **Ray schedule:** sightings are computed every 5 ticks [est] (``RaySchedule``); the window is
  cheap and is computed on every call. ``SightingCache`` reuses the last result while the eye has
  not moved and no block within the sight box has changed, which gives exactly the same output
  (an agent digging in place casts no rays between block changes).
- **X-ray ablation** (DESIGN.md 6.1): resource blocks within 24, exposed or not, which is what
  Mineflayer's ``findBlocks`` gives, with the same per-class cap and no rays. Results carry
  ``xray=True`` and never count toward exit tests.

Honesty: in normal mode nothing here reads a cell that is not exposed, except to learn that it
is not see-through. Changing hidden cells to other opaque classes therefore never changes the
output, the ``rays`` cost figure aside (``tests/test_perception.py`` checks it, from the surface
and from inside a cave).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from hivemind.sim.ref.world import BURIED, World

if TYPE_CHECKING:
    from hivemind.rules.classes import Concepts

EYE_HEIGHT = 1.62  # [mem] standing player eye height
SIGHT_RANGE = 24.0  # DESIGN.md 6.1 [est] resource sightings
ENTITY_RANGE = 32.0  # DESIGN.md 6.2: entities within 32 blocks
WINDOW_HALF_XZ = 7  # DESIGN.md 6.2: 15 x 15 (x, z)
WINDOW_HALF_Y = 3  # DESIGN.md 6.2: 7 levels, y -3..+3
WINDOW_SHAPE = (2 * WINDOW_HALF_XZ + 1, 2 * WINDOW_HALF_Y + 1, 2 * WINDOW_HALF_XZ + 1)
RAY_PERIOD = 5  # DESIGN.md 6.1 [est] sightings checked every 5 ticks
PER_CLASS = 8  # [est] sightings kept per resource class per call (the notebook holds 48 places)
_BATCH = 128  # candidates traced per vectorized batch (does not change the result)

# Faces in a fixed order: -x, +x, -y, +y, -z, +z.
FACES = np.array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)], np.int64)


def see_through_table(concepts: Concepts) -> np.ndarray:
    """bool[256]: the classes a line of sight passes (transparent, except barrier, lava and buried)."""
    see = concepts.transparent.copy()
    see[concepts.block_class("barrier").id] = False
    see[concepts.block_class("lava").id] = False  # [mem] nobody sees through lava
    see[BURIED] = False
    return see


@lru_cache(maxsize=8)
def _tables_for(concepts: Concepts) -> tuple[np.ndarray, np.ndarray]:
    return see_through_table(concepts), concepts.resource.copy()


def _tables(world: World) -> tuple[np.ndarray, np.ndarray]:
    """(see-through, resource) bool[256] tables of the world's concepts."""
    return _tables_for(world.concepts)


def feet_cell(feet: Sequence[float]) -> tuple[int, int, int]:
    return math.floor(feet[0]), math.floor(feet[1]), math.floor(feet[2])


def eye_of(feet: Sequence[float]) -> tuple[float, float, float]:
    """The eye position of an agent standing at ``feet``."""
    return float(feet[0]), float(feet[1]) + EYE_HEIGHT, float(feet[2])


# ---------------------------------------------------------------------------
# Exposure and the voxel window
# ---------------------------------------------------------------------------


def exposed_mask(see: np.ndarray) -> np.ndarray:
    """For a see-through mask with a one-cell margin, whether each interior cell is exposed."""
    return (see[:-2, 1:-1, 1:-1] | see[2:, 1:-1, 1:-1] | see[1:-1, :-2, 1:-1] | see[1:-1, 2:, 1:-1]
            | see[1:-1, 1:-1, :-2] | see[1:-1, 1:-1, 2:])


def is_exposed(world: World, x: int, y: int, z: int) -> bool:
    see, _ = _tables(world)
    return any(see[world.get(x + int(dx), y + int(dy), z + int(dz))] for dx, dy, dz in FACES)


def exposed_faces(world: World, x: int, y: int, z: int) -> tuple[int, ...]:
    """Indices into ``FACES`` of the faces of a cell that touch a see-through cell."""
    see, _ = _tables(world)
    return tuple(i for i, (dx, dy, dz) in enumerate(FACES)
                 if see[world.get(x + int(dx), y + int(dy), z + int(dz))])


@dataclass(frozen=True)
class VoxelWindow:
    """The 15 x 15 x 7 window: ``classes[i, j, k]`` is the cell at ``origin + (i, j, k)``."""

    origin: tuple[int, int, int]
    classes: np.ndarray  # uint8, BURIED (255) where not exposed
    light: np.ndarray  # uint8, 0..15; 0 where buried


def voxel_window(world: World, feet: Sequence[float]) -> VoxelWindow:
    """The window around the feet cell (see the module docstring)."""
    see_table, _ = _tables(world)
    fx, fy, fz = feet_cell(feet)
    x0, y0, z0 = fx - WINDOW_HALF_XZ, fy - WINDOW_HALF_Y, fz - WINDOW_HALF_XZ
    sx, sy, sz = WINDOW_SHAPE
    region = world.region(x0 - 1, y0 - 1, z0 - 1, x0 + sx + 1, y0 + sy + 1, z0 + sz + 1)
    exposed = exposed_mask(see_table[region])
    inner = region[1:-1, 1:-1, 1:-1]
    classes = np.where(exposed, inner, BURIED).astype(np.uint8)
    light = np.maximum(world.region(x0, y0, z0, x0 + sx, y0 + sy, z0 + sz, light=True),
                       world.sky_light_region(x0, y0, z0, x0 + sx, y0 + sy, z0 + sz))
    light = np.where(exposed, light, 0).astype(np.uint8)
    return VoxelWindow((x0, y0, z0), classes, light)


# ---------------------------------------------------------------------------
# Line of sight (3-D DDA)
# ---------------------------------------------------------------------------


def trace_rays(see: np.ndarray, base: np.ndarray, eye: np.ndarray, ends: np.ndarray,
               last_cells: np.ndarray) -> np.ndarray:
    """Vectorized DDA: for each ray from ``eye`` towards ``ends[i]``, whether it reaches
    ``last_cells[i]`` (the cell just before the end point) crossing only see-through cells, the
    eye's cell and the last cell included.

    ``see`` is a see-through mask whose cell [0, 0, 0] is world cell ``base``; cells outside it
    count as not see-through. With t running from 0 at the eye to 1 at the end point, a ray
    steps into the neighbour across the nearest cell plane (smallest t; ties: x, then y, then z)
    until it enters the last cell; it fails if it is blocked first or passes t = 1 without
    entering it. The arithmetic is IEEE doubles in the order written, so the C++ core can repeat
    it exactly: t_max = (boundary - eye) * (1 / d), then t_max += |1 / d| per step.
    """
    n = len(ends)
    if n == 0:
        return np.zeros(0, bool)
    eye = np.asarray(eye, float)
    d = ends - eye[None, :]
    cell = np.broadcast_to(np.floor(eye).astype(np.int64), (n, 3)).copy()
    step = np.sign(d).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(d != 0, 1.0 / d, np.inf)
        t_delta = np.abs(inv)
        boundary = np.where(d > 0, cell + 1, cell).astype(float)
        t_max = np.where(d != 0, (boundary - eye[None, :]) * inv, np.inf)
    shape = np.array(see.shape)

    def lookup(c: np.ndarray) -> np.ndarray:
        local = c - base[None, :]
        ok = np.all((local >= 0) & (local < shape[None, :]), axis=1)
        out = np.zeros(len(c), bool)
        out[ok] = see[local[ok, 0], local[ok, 1], local[ok, 2]]
        return out

    clear = lookup(cell)
    active = clear & ~np.all(cell == last_cells, axis=1)
    max_steps = int(np.abs(last_cells - cell).sum(axis=1).max()) + 3
    for _ in range(max_steps):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        ax = np.argmin(t_max[idx], axis=1)  # first minimum: x before y before z on ties
        beyond = t_max[idx, ax] > 1.0 + 1e-9  # passed the end point without entering the last cell
        active[idx[beyond]] = False
        idx, ax = idx[~beyond], ax[~beyond]
        cell[idx, ax] += step[idx, ax]
        t_max[idx, ax] += t_delta[idx, ax]
        ok = lookup(cell[idx])
        clear[idx] &= ok
        active[idx] = ok & ~np.all(cell[idx] == last_cells[idx], axis=1)
    return clear & np.all(cell == last_cells, axis=1)


def line_of_sight(world: World, eye: Sequence[float], point: Sequence[float],
                  last_cell: Sequence[int] | None = None) -> bool:
    """Whether the segment from ``eye`` to ``point`` crosses only see-through cells (the cell
    containing ``point`` excluded when ``last_cell`` names the cell just before it)."""
    see_table, _ = _tables(world)
    eye = np.asarray(eye, float)
    point = np.asarray(point, float)
    lo = np.floor(np.minimum(eye, point)).astype(np.int64) - 1
    hi = np.floor(np.maximum(eye, point)).astype(np.int64) + 2
    region = world.region(int(lo[0]), int(lo[1]), int(lo[2]), int(hi[0]), int(hi[1]), int(hi[2]))
    see = see_table[region]
    if last_cell is None:
        # Walk all the way: the cell containing the point must be see-through too.
        nudged = point + (point - eye) * 1e-9
        last = np.floor(nudged).astype(np.int64)
        return bool(trace_rays(see, lo, eye, nudged[None, :], last[None, :])[0])
    return bool(trace_rays(see, lo, eye, point[None, :], np.asarray(last_cell, np.int64)[None, :])[0])


def entity_visible(world: World, eye: Sequence[float], pos: Sequence[float], height: float,
                   max_distance: float = ENTITY_RANGE) -> bool:
    """Whether an entity (feet at ``pos``, ``height`` tall) is within range and in sight: a ray to
    its feet + 0.1, its middle or its top - 0.1 crosses only see-through cells [est]."""
    eye = np.asarray(eye, float)
    pos = np.asarray(pos, float)
    if np.linalg.norm(pos + (0.0, height / 2.0, 0.0) - eye) > max_distance:
        return False
    for dy in (min(0.1, height / 2), height / 2, max(height - 0.1, height / 2)):
        if line_of_sight(world, eye, pos + (0.0, dy, 0.0)):
            return True
    return False


# ---------------------------------------------------------------------------
# Resource sightings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sighting:
    """One sighted block: its class id, cell and the distance from the eye to its centre."""

    cls: int
    pos: tuple[int, int, int]
    distance: float


@dataclass(frozen=True)
class Sightings:
    """The resource blocks sighted at one tick (sorted by distance, then x, y, z).

    ``counts`` is the number sighted per class (at most ``per_class``). ``rays`` is the number
    of rays traced, a cost figure that depends on hidden cells (sealed caves): it is left out of
    equality and must never reach the notebook or the observation.
    """

    tick: int
    eye: tuple[float, float, float]
    xray: bool
    items: tuple[Sighting, ...]
    counts: dict[int, int] = field(default_factory=dict)
    rays: int = field(default=0, compare=False)

    def of(self, cls: int) -> tuple[Sighting, ...]:
        return tuple(s for s in self.items if s.cls == cls)

    def nearest(self, cls: int) -> Sighting | None:
        return next((s for s in self.items if s.cls == cls), None)


def _sight_box(eye: np.ndarray, sight_range: float) -> tuple[np.ndarray, np.ndarray]:
    """(base, top): the half-open world box ``sightings`` reads (one cell of margin)."""
    ec = np.floor(eye).astype(np.int64)
    r = int(math.ceil(sight_range)) + 1
    return ec - r - 1, ec + r + 2


def sightings(world: World, eye: Sequence[float], tick: int = 0, *, xray: bool = False,
              sight_range: float = SIGHT_RANGE, per_class: int | None = PER_CLASS) -> Sightings:
    """Resource blocks within ``sight_range`` of the eye that have a face in line of sight, the
    nearest ``per_class`` of each class (module docstring; None: all). ``xray``: resource blocks
    in range whether exposed or not, no rays."""
    see_table, resource = _tables(world)
    eye_arr = np.asarray(eye, float)
    base, top = _sight_box(eye_arr, sight_range)
    region = world.region(int(base[0]), int(base[1]), int(base[2]), int(top[0]), int(top[1]), int(top[2]))
    see = see_table[region]
    inner = region[1:-1, 1:-1, 1:-1]
    centres = [np.arange(base[i] + 1, top[i] - 1) + 0.5 - eye_arr[i] for i in range(3)]
    dist2 = centres[0][:, None, None] ** 2 + centres[1][None, :, None] ** 2 + centres[2][None, None, :] ** 2
    candidate = resource[inner] & (dist2 <= sight_range * sight_range)
    if not xray:
        candidate &= exposed_mask(see)
    cells = np.argwhere(candidate)  # interior indices
    world_cells = cells + base[None, :] + 1
    dists = np.sqrt(dist2[tuple(cells.T)])
    order = np.lexsort((world_cells[:, 2], world_cells[:, 1], world_cells[:, 0], dists))
    cells, world_cells, dists = cells[order], world_cells[order], dists[order]
    classes = inner[tuple(cells.T)].astype(np.int64)
    cap = len(cells) + 1 if per_class is None else per_class
    counts: dict[int, int] = {}
    chosen: list[int] = []
    rays = 0
    for b0 in range(0, len(cells), _BATCH):
        idx = np.arange(b0, min(b0 + _BATCH, len(cells)))
        idx = idx[[counts.get(int(c), 0) < cap for c in classes[idx]]]  # classes already full: skip
        if idx.size == 0:
            continue
        if xray:
            n_rays, visible = np.zeros(len(idx), np.int64), np.ones(len(idx), bool)
        else:
            n_rays, visible = _faces_in_sight(see, base, eye_arr, cells[idx])
        # The definition is sequential (nearest first); the batch only computes ahead.
        for i, n, vis in zip(idx.tolist(), n_rays.tolist(), visible.tolist()):
            c = int(classes[i])
            if counts.get(c, 0) >= cap:
                continue
            rays += n
            if vis:
                counts[c] = counts.get(c, 0) + 1
                chosen.append(i)
    items = tuple(Sighting(int(classes[i]), (int(world_cells[i, 0]), int(world_cells[i, 1]), int(world_cells[i, 2])),
                           float(dists[i])) for i in chosen)
    return Sightings(tick, (float(eye_arr[0]), float(eye_arr[1]), float(eye_arr[2])), xray, items,
                     dict(sorted(counts.items())), rays)


def _faces_in_sight(see: np.ndarray, base: np.ndarray, eye: np.ndarray,
                    cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For interior cells (indices into ``see[1:-1, 1:-1, 1:-1]``): how many open faces face the
    eye (the rays a cell costs, at most 3), and whether one of them is in line of sight."""
    world_cells = cells + base[None, :] + 1
    ray_cell, ray_end, ray_last = [], [], []
    for normal in FACES:
        nb = cells + 1 + normal[None, :]  # the neighbour's index in ``see``
        open_face = see[nb[:, 0], nb[:, 1], nb[:, 2]]
        centre = world_cells + 0.5 + 0.5 * normal[None, :]
        facing = ((eye[None, :] - centre) @ normal) > 0
        use = np.flatnonzero(open_face & facing)
        ray_cell.append(use)
        ray_end.append(centre[use])
        ray_last.append(world_cells[use] + normal[None, :])
    which = np.concatenate(ray_cell)
    ok = trace_rays(see, base, eye, np.concatenate(ray_end), np.concatenate(ray_last))
    n_rays = np.bincount(which, minlength=len(cells))
    visible = np.zeros(len(cells), bool)
    visible[which[ok]] = True
    return n_rays, visible


class SightingCache:
    """Reuses an agent's last ``Sightings`` while its eye is exactly where it was and no block in
    the box ``sightings`` reads has changed (``World.changes_since``); the output is the same as a
    fresh call, with the new tick. Keep one per agent."""

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._cursor = 0
        self._result: Sightings | None = None
        self.hits = 0

    def get(self, world: World, eye: Sequence[float], tick: int, *, xray: bool = False,
            sight_range: float = SIGHT_RANGE, per_class: int | None = PER_CLASS) -> Sightings:
        eye_t = (float(eye[0]), float(eye[1]), float(eye[2]))
        key = (id(world), world.generation, eye_t, xray, sight_range, per_class)
        if key == self._key and self._result is not None and self._unchanged(world, eye_t, sight_range):
            self._cursor = world.change_count
            self.hits += 1
            return replace(self._result, tick=tick)
        self._result = sightings(world, eye_t, tick, xray=xray, sight_range=sight_range, per_class=per_class)
        self._key, self._cursor = key, world.change_count
        return self._result

    def _unchanged(self, world: World, eye: tuple[float, float, float], sight_range: float) -> bool:
        try:
            changes = world.changes_since(self._cursor)
        except ValueError:  # the log was trimmed past our cursor
            return False
        base, top = _sight_box(np.asarray(eye), sight_range)
        return not any(base[0] <= c.x < top[0] and base[1] <= c.y < top[1] and base[2] <= c.z < top[2]
                       for c in changes)


# ---------------------------------------------------------------------------
# The schedule and the combined call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RaySchedule:
    """When sightings are recomputed: ticks with ``tick % period == phase`` (DESIGN.md 6.1 [est]).

    Give agents different phases (for example their slot number) to spread the work; the bridge
    uses the same schedule on its own tick counter.
    """

    period: int = RAY_PERIOD
    phase: int = 0

    def due(self, tick: int) -> bool:
        return tick % self.period == self.phase % self.period


@dataclass(frozen=True)
class Perception:
    """One perception call: the window always, the sightings only on ray ticks (else None)."""

    tick: int
    feet: tuple[float, float, float]
    eye: tuple[float, float, float]
    window: VoxelWindow
    sightings: Sightings | None
    xray: bool


def perceive(world: World, feet: Sequence[float], tick: int | None = None, *, xray: bool = False,
             schedule: RaySchedule | None = RaySchedule(), sight_range: float = SIGHT_RANGE,
             cache: SightingCache | None = None) -> Perception:
    """Window and (on scheduled ticks, or always with ``schedule=None``) resource sightings,
    through ``cache`` when one is given."""
    tick = world.tick if tick is None else tick
    eye = eye_of(feet)
    due = schedule is None or schedule.due(tick)
    seen = None
    if due:
        seen = (cache.get(world, eye, tick, xray=xray, sight_range=sight_range) if cache is not None
                else sightings(world, eye, tick, xray=xray, sight_range=sight_range))
    return Perception(tick, (float(feet[0]), float(feet[1]), float(feet[2])), eye, voxel_window(world, feet), seen,
                      xray)

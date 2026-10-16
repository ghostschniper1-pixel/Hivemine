"""Perception: what an agent may know about the blocks around it (DESIGN.md 6.1; contracts 5, 6.1).

This is the **only** source the notebook and the action masks may use for blocks. The bridge
runs the same functions on the blocks Mineflayer holds (bridge/PROTOCOL.md 7.2), so the
simulator and the real game perceive alike. Usage::

    from hivemind.sim.ref.perception import perceive, voxel_window, sightings
    p = perceive(world, feet=(x, y, z), tick=world.tick)   # window every call, rays when due
    p.window.classes            # uint8 (15, 7, 15) [x, y, z]: class ids, 255 = buried
    p.sightings.counts          # {class id: visible blocks}; None when the rays were not due
    sightings(world, eye, xray=True)    # ablation only: flagged ``xray`` in the result

Definitions (DESIGN.md 6.1):

- **See-through cells:** classes with the ``transparent`` flag (air, glass, leaves, water,
  torches, short grass, ...), except ``barrier``: the barrier stands for the world edge
  (DESIGN.md 11.3 walls the evaluation areas with it), and nothing beyond it can look back in.
  The bridge applies the same rule to barrier blocks.
- **Exposed:** a cell is exposed when at least one of its six faces touches a see-through cell.
- **Voxel window:** the 15 x 15 x 7 cells around the agent's feet cell (x and z -7..+7, y -3..+3),
  indexed [x, y, z] like the world. Exposed cells show their class, others the reserved class
  255 ``buried``. This lets a little knowledge through walls at short range (a stone next to a
  hidden cave shows as stone), the same on both sides. Light per cell: max(block light, raw sky
  light) for exposed cells, 0 for buried ones.
- **Line of sight:** a ray from the eye (feet + 1.62 [mem]) to the centre of an exposed face that
  faces the eye; every cell the ray crosses before reaching the face must be see-through. The
  cells are walked with a 3-D DDA (Amanatides and Woo); when the ray crosses two cell planes at
  once (an edge or a corner), it steps x before y before z, a fixed rule both sides share.
- **Resource sightings:** every block of a resource class (``concepts.resource``) whose centre is
  within 24 blocks [est] of the eye and that has a face in line of sight. The result lists each
  sighted block (class, position, distance) and the number of visible blocks per class.
- **Ray schedule:** sightings are computed every 5 ticks [est] (``RaySchedule``); the window is
  cheap and is computed on every call.
- **X-ray ablation** (DESIGN.md 6.1): every resource block within 24, exposed or not, which is what
  Mineflayer's ``findBlocks`` gives. Results carry ``xray=True`` and never count toward exit tests.

Honesty: in normal mode nothing here reads a cell that is not exposed, except to learn that it
is not see-through. Changing hidden cells to other opaque classes therefore never changes the
output (``tests/test_perception.py`` checks it).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
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

# Faces in a fixed order: -x, +x, -y, +y, -z, +z.
FACES = np.array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)], np.int64)


def see_through_table(concepts: Concepts) -> np.ndarray:
    """bool[256]: the classes a line of sight passes (transparent, except barrier and buried)."""
    see = concepts.transparent.copy()
    see[concepts.block_class("barrier").id] = False
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
    """The resource blocks sighted at one tick (sorted by distance, then x, y, z)."""

    tick: int
    eye: tuple[float, float, float]
    xray: bool
    items: tuple[Sighting, ...]
    counts: dict[int, int] = field(default_factory=dict)

    def of(self, cls: int) -> tuple[Sighting, ...]:
        return tuple(s for s in self.items if s.cls == cls)

    def nearest(self, cls: int) -> Sighting | None:
        return next((s for s in self.items if s.cls == cls), None)


def sightings(world: World, eye: Sequence[float], tick: int = 0, *, xray: bool = False,
              sight_range: float = SIGHT_RANGE) -> Sightings:
    """Resource blocks within ``sight_range`` of the eye that have a face in line of sight
    (``xray``: every resource block in range, flagged as the ablation)."""
    see_table, resource = _tables(world)
    eye_arr = np.asarray(eye, float)
    ec = np.floor(eye_arr).astype(np.int64)
    r = int(math.ceil(sight_range)) + 1
    base = ec - r - 1  # world cell of region[0, 0, 0]; one extra cell of margin
    top = ec + r + 2
    region = world.region(int(base[0]), int(base[1]), int(base[2]), int(top[0]), int(top[1]), int(top[2]))
    see = see_table[region]
    inner = region[1:-1, 1:-1, 1:-1]
    centres = [np.arange(base[i] + 1, top[i] - 1) + 0.5 - eye_arr[i] for i in range(3)]
    dist2 = centres[0][:, None, None] ** 2 + centres[1][None, :, None] ** 2 + centres[2][None, None, :] ** 2
    candidate = resource[inner] & (dist2 <= sight_range * sight_range)
    if not xray:
        candidate &= exposed_mask(see)
    cells = np.argwhere(candidate)  # interior indices
    if xray or cells.size == 0:
        visible = np.ones(len(cells), bool)
    else:
        visible = _cells_in_sight(see, base, eye_arr, cells)
    world_cells = cells[visible] + base[None, :] + 1
    classes = inner[tuple(cells[visible].T)] if len(world_cells) else np.zeros(0, np.uint8)
    dists = np.sqrt(dist2[tuple(cells[visible].T)]) if len(world_cells) else np.zeros(0)
    order = np.lexsort((world_cells[:, 2], world_cells[:, 1], world_cells[:, 0], dists)) if len(world_cells) else []
    items = tuple(Sighting(int(classes[i]), (int(world_cells[i, 0]), int(world_cells[i, 1]), int(world_cells[i, 2])),
                           float(dists[i])) for i in order)
    counts: dict[int, int] = {}
    for s in items:
        counts[s.cls] = counts.get(s.cls, 0) + 1
    return Sightings(tick, (float(eye_arr[0]), float(eye_arr[1]), float(eye_arr[2])), xray, items,
                     dict(sorted(counts.items())))


def _cells_in_sight(see: np.ndarray, base: np.ndarray, eye: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """For interior cells (indices into ``see[1:-1, 1:-1, 1:-1]``), whether some exposed face that
    faces the eye is in line of sight."""
    world_cells = cells + base[None, :] + 1
    ray_cell, ray_end, ray_last = [], [], []
    for f, normal in enumerate(FACES):
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
    visible = np.zeros(len(cells), bool)
    visible[which[ok]] = True
    return visible


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
             schedule: RaySchedule | None = RaySchedule(), sight_range: float = SIGHT_RANGE) -> Perception:
    """Window and (on scheduled ticks, or always with ``schedule=None``) resource sightings."""
    tick = world.tick if tick is None else tick
    eye = eye_of(feet)
    due = schedule is None or schedule.due(tick)
    seen = sightings(world, eye, tick, xray=xray, sight_range=sight_range) if due else None
    return Perception(tick, (float(feet[0]), float(feet[1]), float(feet[2])), eye, voxel_window(world, feet), seen,
                      xray)

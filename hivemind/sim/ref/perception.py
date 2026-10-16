"""Perception: what an agent may know about the world around it (DESIGN.md 6.1, 6.2; contracts 5, 6).

This is the **only** source the notebook, the observation and the action masks may use for
blocks. The bridge runs the same functions on the blocks Mineflayer holds (bridge/PROTOCOL.md
7.2), so the simulator and the real game perceive alike. Usage::

    from hivemind.sim.ref.perception import perceive, voxel_window, panorama, PRESETS, WindowCache
    p = perceive(world, feet, tick, preset="M", agent=slot)   # at a decision: window + panorama
    p.window.classes        # uint8 (21, 9, 21) [x, y, z]: class ids, 255 = unseen
    p.panorama.block        # int16 (24, 96) [row, column]: class of the block hit (M only)
    p.sightings.items       # resource blocks seen, nearest 8 per class (for the notebook)
    perceive(world, feet, tick, panorama=False, cache=cache)  # in-macro recasts (RaySchedule)
    perceive(world, feet, tick, xray=True)                    # ablation only: flagged ``xray``

Definitions (DESIGN.md 6.1, draft 0.5: surface-only vision):

- **See-through cells:** classes with the ``transparent`` flag (air, glass, leaves, water,
  torches, short grass, ...), except ``barrier`` and ``lava`` (and ``unseen``). The barrier
  stands for the world edge (DESIGN.md 11.3 walls the evaluation areas with it), and nothing
  beyond it can look back in; lava is flagged transparent in the data, but no player sees
  through a lava pool [mem]. The bridge's world mirror must be clipped to the evaluation area
  with barrier outside (bridge/PROTOCOL.md 7.2). Above y 191 the world reads as air
  (``World.get``), as in the game. Every other class is *opaque* here.
- **Seen:** a cell is seen when a straight segment from the eye (feet + 1.62 [mem]) to it crosses
  only see-through cells before reaching it: to the centre of a see-through cell, or to the
  centre of a face of an opaque cell that points toward the eye (a face whose outward normal n
  has (eye - face centre) . n > 0; such a segment ends inside the neighbour across the face, so
  that neighbour must be see-through too; a cube has at most three such faces, and one clear
  segment is enough). The eye's own cell is crossed by every segment, so with the head inside an
  opaque cell (a fallen block, lava) nothing is seen. The test is conservative: a cell whose face
  centres are hidden but whose corner shows is ``unseen`` in the window, while a panorama ray or
  the look ray may still hit it, which is honest, because such a ray proves the eye sees it.
- **Segments** are walked cell by cell with a 3-D DDA (Amanatides and Woo; ``trace_rays``): when a
  segment crosses two cell planes at once (an edge or a corner), it steps x before y before z, a
  fixed rule both sides share. The arithmetic is IEEE doubles in the order written.
- **Voxel window** (``voxel_window``): the cells around the feet cell, indexed [x, y, z] like the
  world: preset S 15 x 15 x 7 (x and z -7..+7, y -3..+3), preset M 21 x 21 x 9 (x and z
  -10..+10, y -4..+4), or any ``PerceptionConfig``. Seen cells show their class (seen air shows
  ``air``), every other cell the reserved class 255 ``unseen``. Light per seen cell is
  max(block light, raw sky light) of the cell itself; 0 for unseen cells. Every segment of the
  window lies inside the window's own box (the eye is inside it), so the window reads nothing
  outside it but the one-cell ring of neighbours.
- **Semantic panorama** (``panorama``; preset M only): 96 columns x 24 rows [est] of rays from the
  eye. Column c points at azimuth (c + 0.5) x 3.75 degrees clockwise from north (north, east,
  south, west), row r at elevation 30 - (r + 0.5) x 3.75 degrees (up positive, so row 0 looks
  28.1 degrees up and row 23 58.1 degrees down) [est]: the pixel's centre ray, direction
  (cos e sin a, sin e, -cos e cos a) (north is -z, east +x [mem]). ``panorama_directions`` is the
  fixed table of these unit vectors; the C++ core loads the same table, so no libm call decides a
  pixel. Each ray walks cells from the eye's cell (the DDA above, t in blocks) and stops at the
  first thing it reaches within 64 blocks [est]:

  * a block: the first cell that is not air, entered at t <= 64 (water, lava, leaves, glass,
    plants and the world-edge barrier count). The eye's own cell is never a hit; a ray that starts
    in water (``water`` or ``bubble_column``) passes such cells until it first leaves them
    [design]. Distance = t at which the ray enters the cell; the face is the side it enters
    (a ray travelling south enters a ``north`` face [mem]).
  * an entity's box (``entity_boxes``: other live agents, mobs, dropped items), when the ray
    enters the box at a t with 0 <= t <= 64, strictly before it leaves the air cell it is in (so a
    block entered at the same t wins). Boxes that contain the eye are skipped. The face is the
    box side the ray enters (the slab axis with the largest entry t; ties x, y, z).
  * nothing: ``air`` at distance 1 (64/64), no face.

  The light of a pixel is the raw sky light and block light of the cell *in front of the face*:
  the last cell the ray crossed before the block (or the cell where it met the entity's box, or
  where it ended when nothing was hit). With the head inside an opaque cell, every pixel shows
  that cell at distance 0 with no face and that cell's light: the eye-cell exception, the
  in-wall view of the game, and the honest case (nothing beyond is seen).
- **Resource sightings** (``sightings``): the resource blocks (``concepts.resource``) among the
  seen window cells and the panorama's block hits (up to 64 blocks [est]), by cell, sorted by
  distance from the eye to the cell centre (ties: x, y, z), the nearest ``PER_CLASS`` = 8 [est] of
  each class (the notebook keeps 48 places, DESIGN.md 6.3). Preset S has no panorama, so its
  sightings come from the window alone.
- **Schedule** (DESIGN.md 6.1): the panorama is cast once per decision; during a macro the
  window (and its sightings) is recast every ``RECAST_PERIOD`` = 20 ticks [est] while the eye
  moves (``RaySchedule``). ``WindowCache`` reuses a window while the feet have not moved and no
  block that could change it has changed, giving exactly the same output.
- **X-ray ablation** (DESIGN.md 6.1): the window shows every cell's true class, and sightings are
  every resource block within 64 blocks [est] (exposed or not, no rays; what Mineflayer's
  ``findBlocks`` gives), nearest 8 per class. The panorama stays as it is. Results carry
  ``xray=True``; never used for an agent that is evaluated or deployed.

Honesty: in normal mode a ray stops at the first opaque cell, so nothing here reads a cell no
straight segment from the eye reaches through see-through cells, except to learn that it is not
see-through. Changing such cells into other opaque classes that block light and emit none
therefore never changes the window, the panorama or the sightings (``tests/test_perception.py``
checks it from the surface and from inside caves, around walls, sealed caves and floors). Light
is the exception the design states: a light source out of sight changes the light of seen cells.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from hivemind.sim.ref.world import AIR, MAX_LIGHT, UNSEEN, World

if TYPE_CHECKING:
    from hivemind.rules.classes import Concepts

EYE_HEIGHT = 1.62  # [mem] standing player eye height
ENTITY_RANGE = 32.0  # DESIGN.md 6.2: the entity table lists visible entities within 32 blocks
PER_CLASS = 8  # [est] sightings kept per resource class per call (the notebook holds 48 places)
XRAY_RANGE = 64.0  # DESIGN.md 6.1 [est]: x-ray memory from every resource block within 64 blocks
RECAST_PERIOD = 20  # DESIGN.md 6.1 [est]: in-macro window recasts every 20 ticks while the eye moves
NONE = -1  # "none" in the panorama's int arrays (no block, no entity, no agent, no face)
DEFAULT_BOX = (0.6, 1.8)  # [est] width, height for entity classes whose size the data lacks (fallbacks)

# Neighbour offsets in a fixed order: -x, +x, -y, +y, -z, +z (``exposed_faces`` indexes these).
FACES = np.array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)], np.int64)

# The face words (DESIGN.md 6.2 order; the same as hivemind.lang.world_words.FACE_WORDS) and their
# outward normals: north is -z, east +x [mem].
FACE_WORDS: tuple[str, ...] = ("top", "bottom", "north", "south", "east", "west")
FACE_NORMALS = np.array([(0, 1, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1), (1, 0, 0), (-1, 0, 0)], np.int64)
# _ENTERED[axis][step > 0]: the face a ray enters crossing into a cell along axis, normal = -step.
_ENTERED = np.array([[4, 5], [0, 1], [3, 2]], np.int8)  # x: east, west; y: top, bottom; z: south, north


def entered_face(axis: int, step: int) -> int:
    """Index into ``FACE_WORDS`` of the face a ray enters when it crosses into a cell along
    ``axis`` (0 x, 1 y, 2 z) moving in direction ``step`` (+1 or -1): a ray travelling south (+z)
    enters a ``north`` face, a falling ray (-y) a ``top`` face."""
    return int(_ENTERED[axis][1 if step > 0 else 0])


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanoramaConfig:
    """The panorama's size, range and pitch span (DESIGN.md 6.1 [est])."""

    columns: int = 96  # [est] 3.75 degrees each, column 0 starting at north, clockwise
    rows: int = 24  # [est]
    range: float = 64.0  # [est] blocks
    top: float = 30.0  # [est] elevation of the top edge, degrees above level
    bottom: float = -60.0  # [est] elevation of the bottom edge

    def __post_init__(self) -> None:
        if self.columns < 1 or self.rows < 1 or self.range <= 0 or not -90 <= self.bottom < self.top <= 90:
            raise ValueError(f"bad panorama config {self}")

    @property
    def column_step(self) -> float:
        return 360.0 / self.columns

    @property
    def row_step(self) -> float:
        return (self.top - self.bottom) / self.rows


@dataclass(frozen=True)
class PerceptionConfig:
    """A window size (half extents around the feet cell) and the panorama, if any."""

    name: str
    half_xz: int
    half_y: int
    panorama: PanoramaConfig | None

    def __post_init__(self) -> None:
        if self.half_xz < 1 or self.half_y < 2:  # half_y >= 2 keeps the eye's cell inside the window
            raise ValueError(f"bad window half extents {self.half_xz}, {self.half_y}")

    @property
    def window_shape(self) -> tuple[int, int, int]:
        return (2 * self.half_xz + 1, 2 * self.half_y + 1, 2 * self.half_xz + 1)


PRESETS: dict[str, PerceptionConfig] = {
    "S": PerceptionConfig("S", 7, 3, None),  # DESIGN.md 6.1: 15 x 15 x 7, no panorama
    "M": PerceptionConfig("M", 10, 4, PanoramaConfig()),  # DESIGN.md 6.1: 21 x 21 x 9 and the panorama
}
DEFAULT_PRESET = "M"  # DESIGN.md 6.4: M is the default


def config_of(preset: str | PerceptionConfig) -> PerceptionConfig:
    """A preset by name ("S", "M"), or a config as given."""
    return PRESETS[preset] if isinstance(preset, str) else preset


# ---------------------------------------------------------------------------
# Tables and small helpers
# ---------------------------------------------------------------------------


def see_through_table(concepts: Concepts) -> np.ndarray:
    """bool[256]: the classes a line of sight passes (transparent, except barrier, lava and unseen)."""
    see = concepts.transparent.copy()
    see[concepts.block_class("barrier").id] = False
    see[concepts.block_class("lava").id] = False  # [mem] nobody sees through lava
    see[UNSEEN] = False
    return see


@dataclass(frozen=True)
class _Tables:
    see: np.ndarray  # bool[256] see-through
    resource: np.ndarray  # bool[256]
    water: np.ndarray  # bool[256]: water and bubble columns (what a swimmer sees through, and F3's "water")
    lava: int
    player: int  # entity class of agents
    item: int  # entity class of dropped items
    width: np.ndarray  # float64 per entity class (DEFAULT_BOX where the data has none)
    height: np.ndarray


@lru_cache(maxsize=8)
def _tables_for(concepts: Concepts) -> _Tables:
    water = np.zeros(UNSEEN + 1, bool)
    for name in ("water", "bubble_column"):
        water[concepts.block_class(name).id] = True
    n = 1 + max(c.id for c in concepts.entity_classes)
    width = np.full(n, DEFAULT_BOX[0])
    height = np.full(n, DEFAULT_BOX[1])
    for c in concepts.entity_classes:
        if c.width is not None and c.height is not None:
            width[c.id], height[c.id] = c.width, c.height
    return _Tables(see_through_table(concepts), concepts.resource.copy(), water, concepts.block_class("lava").id,
                   concepts.entity_class("player").id, concepts.entity_class("item").id, width, height)


def tables(world: World) -> _Tables:
    """The class tables perception uses, for the world's concepts (cached)."""
    return _tables_for(world.concepts)


def feet_cell(feet: Sequence[float]) -> tuple[int, int, int]:
    return math.floor(feet[0]), math.floor(feet[1]), math.floor(feet[2])


def eye_of(feet: Sequence[float]) -> tuple[float, float, float]:
    """The eye position of an agent standing at ``feet``."""
    return float(feet[0]), float(feet[1]) + EYE_HEIGHT, float(feet[2])


class _ArrayEq:
    """Equality by value for frozen dataclasses holding NumPy arrays (fields with compare=False
    are skipped)."""

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        for name, f in self.__dataclass_fields__.items():  # type: ignore[attr-defined]
            if not f.compare:
                continue
            a, b = getattr(self, name), getattr(other, name)
            if isinstance(a, np.ndarray):
                if not (isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)):
                    return False
            elif a != b:
                return False
        return True

    __hash__ = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------


def exposed_mask(see: np.ndarray) -> np.ndarray:
    """For a see-through mask with a one-cell margin, whether each interior cell is exposed (a face
    touches a see-through cell). Only exposed opaque cells can be seen."""
    return (see[:-2, 1:-1, 1:-1] | see[2:, 1:-1, 1:-1] | see[1:-1, :-2, 1:-1] | see[1:-1, 2:, 1:-1]
            | see[1:-1, 1:-1, :-2] | see[1:-1, 1:-1, 2:])


def is_exposed(world: World, x: int, y: int, z: int) -> bool:
    see = tables(world).see
    return any(see[world.get(x + int(dx), y + int(dy), z + int(dz))] for dx, dy, dz in FACES)


def exposed_faces(world: World, x: int, y: int, z: int) -> tuple[int, ...]:
    """Indices into ``FACES`` of the faces of a cell that touch a see-through cell."""
    see = tables(world).see
    return tuple(i for i, (dx, dy, dz) in enumerate(FACES)
                 if see[world.get(x + int(dx), y + int(dy), z + int(dz))])


# ---------------------------------------------------------------------------
# Line of sight (3-D DDA)
# ---------------------------------------------------------------------------


def trace_rays(see: np.ndarray, base: np.ndarray, eye: np.ndarray, ends: np.ndarray,
               last_cells: np.ndarray) -> np.ndarray:
    """Vectorized DDA: for each ray from ``eye`` towards ``ends[i]``, whether it reaches
    ``last_cells[i]`` (the cell just before the end point, or the cell containing it) crossing
    only see-through cells, the eye's cell and the last cell included.

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
    see_table = tables(world).see
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
# Seen cells and the voxel window
# ---------------------------------------------------------------------------


def seen_cells(world: World, eye: Sequence[float], origin: Sequence[int],
               shape: Sequence[int]) -> tuple[np.ndarray, int]:
    """Which cells of the box ``origin + [0, shape)`` are seen from ``eye`` (module docstring), as
    bool[shape], and the number of segments traced (a cost figure)."""
    see_table = tables(world).see
    eye_arr = np.asarray(eye, float)
    o = np.asarray(origin, np.int64)
    s = np.asarray(shape, np.int64)
    ec = np.floor(eye_arr).astype(np.int64)
    lo = np.minimum(o, ec) - 1  # the box and the eye's cell, with a ring of neighbours
    hi = np.maximum(o + s, ec + 1) + 1
    see = see_table[world.region(int(lo[0]), int(lo[1]), int(lo[2]), int(hi[0]), int(hi[1]), int(hi[2]))]
    off = o - lo
    box_see = see[off[0]:off[0] + s[0], off[1]:off[1] + s[1], off[2]:off[2] + s[2]].reshape(-1)
    cells = np.indices(tuple(int(v) for v in s)).reshape(3, -1).T + o[None, :]
    owners, ends, lasts = [], [], []
    st = np.flatnonzero(box_see)  # see-through cells: one segment to the centre
    owners.append(st)
    ends.append(cells[st] + 0.5)
    lasts.append(cells[st])
    op = np.flatnonzero(~box_see)  # opaque cells: one segment per open face turned toward the eye
    oc = cells[op]
    for normal in FACES:
        nb = oc + normal[None, :] - lo[None, :]
        open_face = see[nb[:, 0], nb[:, 1], nb[:, 2]]
        centre = oc + 0.5 + 0.5 * normal[None, :]
        facing = ((eye_arr[None, :] - centre) @ normal) > 0
        use = np.flatnonzero(open_face & facing)
        owners.append(op[use])
        ends.append(centre[use])
        lasts.append(oc[use] + normal[None, :])
    owner = np.concatenate(owners)
    ok = trace_rays(see, lo, eye_arr, np.concatenate(ends), np.concatenate(lasts))
    seen = np.zeros(len(cells), bool)
    seen[owner[ok]] = True
    return seen.reshape(tuple(int(v) for v in s)), int(len(owner))


@dataclass(frozen=True, eq=False)
class VoxelWindow(_ArrayEq):
    """The window around the feet cell: ``classes[i, j, k]`` is the cell at ``origin + (i, j, k)``."""

    origin: tuple[int, int, int]
    classes: np.ndarray  # uint8, UNSEEN (255) where not seen (x-ray: every cell's class)
    light: np.ndarray  # uint8 0..15, max(block light, raw sky light); 0 where unseen
    seen: np.ndarray  # bool (x-ray: all True)
    xray: bool = False
    rays: int = field(default=0, compare=False)  # segments traced: a cost figure, never an input


def voxel_window(world: World, feet: Sequence[float], config: str | PerceptionConfig = DEFAULT_PRESET, *,
                 xray: bool = False) -> VoxelWindow:
    """The window of a preset around the feet cell (module docstring); ``xray``: the ablation."""
    cfg = config_of(config)
    fx, fy, fz = feet_cell(feet)
    x0, y0, z0 = fx - cfg.half_xz, fy - cfg.half_y, fz - cfg.half_xz
    sx, sy, sz = cfg.window_shape
    classes = world.region(x0, y0, z0, x0 + sx, y0 + sy, z0 + sz)
    light = np.maximum(world.region(x0, y0, z0, x0 + sx, y0 + sy, z0 + sz, light=True),
                       world.sky_light_region(x0, y0, z0, x0 + sx, y0 + sy, z0 + sz))
    if xray:
        return VoxelWindow((x0, y0, z0), classes, light.astype(np.uint8), np.ones(classes.shape, bool), True, 0)
    seen, rays = seen_cells(world, eye_of(feet), (x0, y0, z0), (sx, sy, sz))
    return VoxelWindow((x0, y0, z0), np.where(seen, classes, UNSEEN).astype(np.uint8),
                       np.where(seen, light, 0).astype(np.uint8), seen, False, rays)


class WindowCache:
    """Reuses an agent's last window while its feet are exactly where they were and no block that
    could change the window has changed (``World.changes_since``): any cell with x and z within
    16 of the window and y at most 16 below it. That box holds every cell the window's classes,
    segments and light read (block light travels at most 15 cells, sky light comes from above).
    The output equals a fresh call. Keep one per agent."""

    LIGHT_MARGIN = 16  # [design] MAX_LIGHT + 1 cells around the window

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._cursor = 0
        self._result: VoxelWindow | None = None
        self.hits = 0

    def get(self, world: World, feet: Sequence[float], config: str | PerceptionConfig = DEFAULT_PRESET, *,
            xray: bool = False) -> VoxelWindow:
        cfg = config_of(config)
        feet_t = (float(feet[0]), float(feet[1]), float(feet[2]))
        key = (id(world), world.generation, feet_t, cfg, xray)
        if key == self._key and self._result is not None and self._unchanged(world, self._result, cfg):
            self._cursor = world.change_count
            self.hits += 1
            return self._result
        self._result = voxel_window(world, feet_t, cfg, xray=xray)
        self._key, self._cursor = key, world.change_count
        return self._result

    def _unchanged(self, world: World, win: VoxelWindow, cfg: PerceptionConfig) -> bool:
        try:
            changes = world.changes_since(self._cursor)
        except ValueError:  # the log was trimmed past our cursor
            return False
        m = self.LIGHT_MARGIN
        x0, y0, z0 = win.origin
        sx, _, sz = cfg.window_shape
        return not any(x0 - m <= c.x < x0 + sx + m and z0 - m <= c.z < z0 + sz + m and c.y >= y0 - m
                       for c in changes)


# ---------------------------------------------------------------------------
# Entity boxes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EntityBoxes(_ArrayEq):
    """Axis-aligned boxes of the entities a ray can hit, in a fixed order: live agents (slot order),
    mobs, then dropped items. ``lo``/``hi`` (n, 3) float64; ``cls`` the entity class; ``agent`` the
    agent slot of a player (else -1); ``source`` 0 agent, 1 mob, 2 item; ``slot`` the slot index."""

    lo: np.ndarray
    hi: np.ndarray
    cls: np.ndarray
    agent: np.ndarray
    source: np.ndarray
    slot: np.ndarray

    def __len__(self) -> int:
        return len(self.cls)


def entity_boxes(world: World, *, exclude_agent: int | None = None, items: bool = True) -> EntityBoxes:
    """The boxes of every entity in the world's slot tables: x and z +-width/2 around the position,
    y from the feet up by the height (the entity class's size, contracts 3; ``DEFAULT_BOX`` [est]
    where the data has none). Dead agents are left out (the game removes the body [mem]), and so is
    ``exclude_agent`` (the observer itself)."""
    t = tables(world)
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []  # pos, cls, slots, source
    a = world.agents.active()
    a = a[world.agents.alive[a]]
    if exclude_agent is not None:
        a = a[a != exclude_agent]
    parts.append((world.agents.pos[a], np.full(len(a), t.player, np.int64), a, 0))
    m = world.mobs.active()
    parts.append((world.mobs.pos[m], world.mobs.cls[m].astype(np.int64), m, 1))
    if items:
        it = world.items.active()
        parts.append((world.items.pos[it], np.full(len(it), t.item, np.int64), it, 2))
    pos = np.concatenate([p[0] for p in parts]).reshape(-1, 3).astype(float)
    cls = np.concatenate([p[1] for p in parts]).astype(np.int64)
    slot = np.concatenate([p[2] for p in parts]).astype(np.int64)
    source = np.concatenate([np.full(len(p[2]), p[3], np.int64) for p in parts])
    half = t.width[cls] / 2.0
    lo = np.stack([pos[:, 0] - half, pos[:, 1], pos[:, 2] - half], axis=1)
    hi = np.stack([pos[:, 0] + half, pos[:, 1] + t.height[cls], pos[:, 2] + half], axis=1)
    agent = np.where(source == 0, slot, NONE)
    return EntityBoxes(lo, hi, cls.astype(np.int16), agent.astype(np.int16), source.astype(np.int8),
                       slot.astype(np.int16))


def ray_box_entry(eye: Sequence[float], dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Slab test: for rays from ``eye`` along ``dirs`` (n, 3) and boxes ``lo``/``hi`` (m, 3), the t
    at which each ray enters each box (n, m; inf when it misses, when the box is behind the eye or
    when the box contains the eye) and the slab axis of that entry (largest entry t; ties x, y,
    z). t is in units of the direction's length."""
    eye = np.asarray(eye, float)
    n, m = len(dirs), len(lo)
    if m == 0 or n == 0:
        return np.full((n, m), np.inf), np.zeros((n, m), np.int8)
    near = np.empty((n, m, 3))
    far = np.empty((n, m, 3))
    for k in range(3):
        d = dirs[:, k][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(d != 0, 1.0 / d, 0.0)
            t1 = (lo[None, :, k] - eye[k]) * inv
            t2 = (hi[None, :, k] - eye[k]) * inv
        inside = (lo[None, :, k] <= eye[k]) & (eye[k] <= hi[None, :, k])
        near[:, :, k] = np.where(d != 0, np.minimum(t1, t2), np.where(inside, -np.inf, np.inf))
        far[:, :, k] = np.where(d != 0, np.maximum(t1, t2), np.where(inside, np.inf, -np.inf))
    axis = np.argmax(near, axis=2).astype(np.int8)
    t_near = near.max(axis=2)
    t_far = far.min(axis=2)
    hit = (t_near <= t_far) & (t_near >= 0.0)
    return np.where(hit, t_near, np.inf), axis


# ---------------------------------------------------------------------------
# The semantic panorama
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def panorama_directions(cfg: PanoramaConfig = PanoramaConfig()) -> np.ndarray:
    """float64 (rows, columns, 3) unit vectors of the pixels' centre rays (module docstring). This
    table is the definition; the C++ core loads it rather than recomputing sines."""
    az = np.radians((np.arange(cfg.columns) + 0.5) * cfg.column_step)  # clockwise from north
    el = np.radians(cfg.top - (np.arange(cfg.rows) + 0.5) * cfg.row_step)  # up positive
    ce, se = np.cos(el)[:, None], np.sin(el)[:, None]
    dirs = np.stack([ce * np.sin(az)[None, :], np.broadcast_to(se, (cfg.rows, cfg.columns)),
                     -ce * np.cos(az)[None, :]], axis=2)
    dirs.setflags(write=False)
    return dirs


@dataclass(frozen=True, eq=False)
class Panorama(_ArrayEq):
    """The semantic panorama: arrays of shape (rows, columns), row 0 at the top, column 0 at north.

    ``block`` int16: class of the block cell hit; ``AIR`` (0) when nothing is hit within range;
    ``NONE`` (-1) when an entity's box is hit first. ``entity`` int16: entity class of the box hit,
    else -1. ``agent`` int16: the agent slot when that entity is a player, else -1. ``face`` int8:
    index into ``FACE_WORDS`` of the side entered, -1 when nothing is hit (or the head is inside
    an opaque cell). ``distance`` float32: distance / range, 1 when nothing is hit.
    ``sky_light``, ``block_light`` float32: raw light / 15 of the cell in front of the face.
    ``cell`` int32 (rows, columns, 3): the block cell hit, else the cell where the ray stopped (for
    the notebook's sightings; never an input, since it holds absolute coordinates).
    """

    eye: tuple[float, float, float]
    config: PanoramaConfig
    block: np.ndarray
    entity: np.ndarray
    agent: np.ndarray
    face: np.ndarray
    distance: np.ndarray
    sky_light: np.ndarray
    block_light: np.ndarray
    cell: np.ndarray
    steps: int = field(default=0, compare=False)  # cells walked: a cost figure, never an input

    def hits(self, cls: int) -> np.ndarray:
        """bool (rows, columns): the pixels showing block class ``cls``."""
        return self.block == cls


def panorama(world: World, eye: Sequence[float], config: PanoramaConfig | None = None, *,
             exclude_agent: int | None = None, boxes: EntityBoxes | None = None) -> Panorama:
    """Cast the panorama from ``eye`` (module docstring). ``exclude_agent``: the observer's own
    agent slot (its box is never a hit); ``boxes``: precomputed ``entity_boxes``."""
    cfg = config if config is not None else PanoramaConfig()
    t = tables(world)
    eye_arr = np.asarray(eye, float)
    dirs = panorama_directions(cfg).reshape(-1, 3)
    n = len(dirs)
    shape = (cfg.rows, cfg.columns)
    ec = np.floor(eye_arr).astype(np.int64)
    eye_cls = world.get(int(ec[0]), int(ec[1]), int(ec[2]))
    if not t.see[eye_cls]:
        # The head is inside an opaque cell: it is all the eye sees (the eye-cell exception).
        sky, blk = world.light_points(ec[None, :])
        return Panorama(tuple(float(v) for v in eye_arr), cfg, np.full(shape, eye_cls, np.int16),
                        np.full(shape, NONE, np.int16), np.full(shape, NONE, np.int16),
                        np.full(shape, NONE, np.int8), np.zeros(shape, np.float32),
                        np.full(shape, np.float32(sky[0]) / np.float32(MAX_LIGHT), np.float32),
                        np.full(shape, np.float32(blk[0]) / np.float32(MAX_LIGHT), np.float32),
                        np.broadcast_to(ec.astype(np.int32), (*shape, 3)).copy(), 0)
    max_t = float(cfg.range)
    reach = int(math.ceil(max_t)) + 2
    y_lo = int(math.floor(eye_arr[1] + min(0.0, float(dirs[:, 1].min())) * max_t)) - 2
    y_hi = int(math.floor(eye_arr[1] + max(0.0, float(dirs[:, 1].max())) * max_t)) + 3
    base = np.array([ec[0] - reach, y_lo, ec[2] - reach], np.int64)
    top = np.array([ec[0] + reach + 1, y_hi, ec[2] + reach + 1], np.int64)
    classes = world.region(int(base[0]), int(base[1]), int(base[2]), int(top[0]), int(top[1]), int(top[2]))

    # Entities: the nearest box entry per ray, within range.
    if boxes is None:
        boxes = entity_boxes(world, exclude_agent=exclude_agent)
    t_box, box_axis = ray_box_entry(eye_arr, dirs, boxes.lo, boxes.hi)
    if len(boxes):
        first = np.argmin(t_box, axis=1)  # ties: the first box in ``entity_boxes`` order
        ent_t = t_box[np.arange(n), first]
        ent_axis = box_axis[np.arange(n), first]
    else:
        first = np.zeros(n, np.int64)
        ent_t = np.full(n, np.inf)
        ent_axis = np.zeros(n, np.int8)
    ent_t = np.where(ent_t <= max_t, ent_t, np.inf)

    # The DDA (same arithmetic as ``trace_rays``, t in blocks along unit directions).
    cell = np.broadcast_to(ec, (n, 3)).copy()
    step = np.sign(dirs).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(dirs != 0, 1.0 / dirs, np.inf)
        t_delta = np.abs(inv)
        boundary = np.where(dirs > 0, cell + 1, cell).astype(float)
        t_max = np.where(dirs != 0, (boundary - eye_arr[None, :]) * inv, np.inf)
    kind = np.zeros(n, np.int8)  # 0 nothing, 1 block, 2 entity
    t_hit = np.full(n, max_t)
    hit_axis = np.zeros(n, np.int64)
    front = cell.copy()  # the cell in front of the face (whose light the pixel shows)
    hit_cell = cell.copy()
    swim = np.full(n, bool(t.water[eye_cls]))
    active = np.ones(n, bool)
    steps = 0
    rows = np.arange(n)
    while True:
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        steps += int(idx.size)
        tm = t_max[idx]
        ax = np.argmin(tm, axis=1)  # first minimum: x before y before z on ties
        exit_t = tm[np.arange(idx.size), ax]
        ent = ent_t[idx] < exit_t  # the ray meets the box inside the cell it is in
        end = ~ent & (exit_t > max_t)  # the range ends inside this cell
        done = idx[ent | end]
        kind[idx[ent]] = 2
        t_hit[idx[ent]] = ent_t[idx[ent]]
        front[done] = cell[done]
        hit_cell[done] = cell[done]
        active[done] = False
        go = ~(ent | end)
        i2, ax2, t2 = idx[go], ax[go], exit_t[go]
        prev = cell[i2].copy()
        cell[i2, ax2] += step[i2, ax2]
        t_max[i2, ax2] += t_delta[i2, ax2]
        local = cell[i2] - base[None, :]
        c = classes[local[:, 0], local[:, 1], local[:, 2]]
        sw = swim[i2]
        wet = t.water[c]
        swim[i2[sw & ~wet]] = False  # left the water the ray started in
        hit = ~(sw & wet) & (c != AIR)
        hi = i2[hit]
        kind[hi] = 1
        t_hit[hi] = t2[hit]
        hit_axis[hi] = ax2[hit]
        front[hi] = prev[hit]
        hit_cell[hi] = cell[hi]
        active[hi] = False

    block = np.zeros(n, np.int16)  # AIR when nothing is hit
    entity = np.full(n, NONE, np.int16)
    agent = np.full(n, NONE, np.int16)
    face = np.full(n, NONE, np.int8)
    b = kind == 1
    local = hit_cell[b] - base[None, :]
    block[b] = classes[local[:, 0], local[:, 1], local[:, 2]]
    face[b] = _ENTERED[hit_axis[b], (step[b, hit_axis[b]] > 0).astype(np.int64)]
    e = kind == 2
    block[e] = NONE
    if e.any():
        which = first[e]
        entity[e] = boxes.cls[which]
        agent[e] = boxes.agent[which]
        eax = ent_axis[e].astype(np.int64)
        face[e] = _ENTERED[eax, (step[rows[e], eax] > 0).astype(np.int64)]
    sky, blk = world.light_points(front)
    scale = np.float32(MAX_LIGHT)
    return Panorama(
        tuple(float(v) for v in eye_arr), cfg, block.reshape(shape), entity.reshape(shape), agent.reshape(shape),
        face.reshape(shape), (t_hit / max_t).astype(np.float32).reshape(shape),
        (sky.astype(np.float32) / scale).reshape(shape), (blk.astype(np.float32) / scale).reshape(shape),
        hit_cell.astype(np.int32).reshape(*shape, 3), steps)


_cast_panorama = panorama  # ``perceive`` has a flag of the same name


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

    ``counts`` is the number sighted per class (at most ``per_class``). ``rays`` is a cost figure
    (segments traced and panorama cells walked); it is left out of equality and must never reach
    the notebook or the observation.
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


def _select(cells: np.ndarray, classes: np.ndarray, eye: np.ndarray, per_class: int | None
            ) -> tuple[tuple[Sighting, ...], dict[int, int]]:
    """The nearest ``per_class`` cells of each class (distance to the centre, then x, y, z)."""
    if len(cells) == 0:
        return (), {}
    dist = np.sqrt((((cells + 0.5) - eye[None, :]) ** 2).sum(axis=1))
    order = np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0], dist))
    cap = len(cells) + 1 if per_class is None else per_class
    counts: dict[int, int] = {}
    items = []
    for i in order.tolist():
        c = int(classes[i])
        if counts.get(c, 0) >= cap:
            continue
        counts[c] = counts.get(c, 0) + 1
        items.append(Sighting(c, (int(cells[i, 0]), int(cells[i, 1]), int(cells[i, 2])), float(dist[i])))
    return tuple(items), dict(sorted(counts.items()))


def sightings(concepts: Concepts, eye: Sequence[float], tick: int, window: VoxelWindow | None,
              pano: Panorama | None = None, *, per_class: int | None = PER_CLASS) -> Sightings:
    """Resource blocks among the seen window cells and the panorama's block hits (module
    docstring), the nearest ``per_class`` of each class (None: all)."""
    resource = _tables_for(concepts).resource
    eye_arr = np.asarray(eye, float)
    parts_cells, parts_cls = [np.zeros((0, 3), np.int64)], [np.zeros(0, np.int64)]
    rays = 0
    if window is not None:
        if window.xray:
            raise ValueError("an x-ray window is not a source of honest sightings (use xray_sightings)")
        hit = window.seen & resource[window.classes]
        idx = np.argwhere(hit)
        parts_cells.append(idx + np.asarray(window.origin, np.int64)[None, :])
        parts_cls.append(window.classes[hit].astype(np.int64))
        rays += window.rays
    if pano is not None:
        hit = (pano.block > 0) & resource[np.maximum(pano.block, 0)]
        parts_cells.append(pano.cell[hit].astype(np.int64))
        parts_cls.append(pano.block[hit].astype(np.int64))
        rays += pano.steps
    cells = np.concatenate(parts_cells)
    classes = np.concatenate(parts_cls)
    if len(cells):
        cells, first = np.unique(cells, axis=0, return_index=True)  # one sighting per cell
        classes = classes[first]
    items, counts = _select(cells, classes, eye_arr, per_class)
    return Sightings(int(tick), (float(eye_arr[0]), float(eye_arr[1]), float(eye_arr[2])), False, items, counts,
                     rays)


def xray_sightings(world: World, eye: Sequence[float], tick: int = 0, *, sight_range: float = XRAY_RANGE,
                   per_class: int | None = PER_CLASS) -> Sightings:
    """Ablation only: every resource block whose centre lies within ``sight_range`` of the eye,
    seen or not (no rays), the nearest ``per_class`` of each class; flagged ``xray``."""
    resource = tables(world).resource
    eye_arr = np.asarray(eye, float)
    ec = np.floor(eye_arr).astype(np.int64)
    r = int(math.ceil(sight_range)) + 1
    base = ec - r
    region = world.region(int(base[0]), int(base[1]), int(base[2]),
                          int(ec[0] + r + 1), int(ec[1] + r + 1), int(ec[2] + r + 1))
    centres = [np.arange(base[i], base[i] + region.shape[i]) + 0.5 - eye_arr[i] for i in range(3)]
    dist2 = centres[0][:, None, None] ** 2 + centres[1][None, :, None] ** 2 + centres[2][None, None, :] ** 2
    hit = resource[region] & (dist2 <= sight_range * sight_range)
    cells = np.argwhere(hit) + base[None, :]
    items, counts = _select(cells, region[hit].astype(np.int64), eye_arr, per_class)
    return Sightings(int(tick), (float(eye_arr[0]), float(eye_arr[1]), float(eye_arr[2])), True, items, counts, 0)


# ---------------------------------------------------------------------------
# The schedule and the combined call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RaySchedule:
    """When a macro recasts the window: ticks with ``tick % period == phase`` (DESIGN.md 6.1:
    every 20 ticks [est] while the eye moves; with a ``WindowCache`` a still eye costs nothing).

    Give agents different phases (for example their slot number) to spread the work; the bridge
    uses the same schedule on its own tick counter. The panorama is cast once per decision.
    """

    period: int = RECAST_PERIOD
    phase: int = 0

    def due(self, tick: int) -> bool:
        return tick % self.period == self.phase % self.period


@dataclass(frozen=True)
class Perception:
    """One perception call: the window, the panorama (preset M at a decision, else None) and the
    sightings for the notebook."""

    tick: int
    feet: tuple[float, float, float]
    eye: tuple[float, float, float]
    preset: str
    window: VoxelWindow
    panorama: Panorama | None
    sightings: Sightings
    xray: bool


def perceive(world: World, feet: Sequence[float], tick: int | None = None, *,
             preset: str | PerceptionConfig = DEFAULT_PRESET, agent: int | None = None, panorama: bool = True,
             xray: bool = False, cache: WindowCache | None = None, per_class: int | None = PER_CLASS) -> Perception:
    """The window (through ``cache`` when one is given), the panorama when ``panorama`` is set and
    the preset has one (pass False for in-macro recasts), and the sightings from both.
    ``agent``: the observer's slot, whose own box the panorama skips. ``xray``: the ablation
    (module docstring), flagged in every part of the result."""
    cfg = config_of(preset)
    tick = world.tick if tick is None else tick
    feet_t = (float(feet[0]), float(feet[1]), float(feet[2]))
    eye = eye_of(feet_t)
    window = (cache.get(world, feet_t, cfg, xray=xray) if cache is not None
              else voxel_window(world, feet_t, cfg, xray=xray))
    pano = None
    if panorama and cfg.panorama is not None:
        pano = _cast_panorama(world, eye, cfg.panorama, exclude_agent=agent)
    if xray:
        seen = xray_sightings(world, eye, tick, per_class=per_class)
    else:
        seen = sightings(world.concepts, eye, tick, window, pano, per_class=per_class)
    return Perception(int(tick), feet_t, eye, cfg.name, window, pano, seen, xray)

"""Perception: what an agent may know about the world around it (DESIGN.md 6.1, 6.2; contracts 5, 6).

This is the **only** source the notebook, the observation and the action masks may use for
blocks. The bridge runs the same functions on the blocks Mineflayer holds (bridge/PROTOCOL.md
7.2), so the simulator and the real game perceive alike. Usage::

    from hivemind.sim.ref.perception import perceive, voxel_window, panorama, PRESETS, WindowCache
    p = perceive(world, agent=slot, preset="M")   # at a decision: window, panorama, F3 group
    p.window.classes        # uint8 (21, 9, 21) [x, y, z]: class ids, 255 = unseen
    p.panorama.block        # int16 (24, 96) [row, column]: class of the block hit (M only)
    p.sightings.items       # every resource block seen, nearest first (for the notebook)
    p.f3.block, p.f3.time_word                                # the F3 group (debug_info.py)
    perceive(world, feet, tick, panorama=False, cache=cache)  # in-macro recasts (RaySchedule)
    perceive(world, feet, tick, xray=True)                    # ablation only: flagged ``xray``

Definitions (DESIGN.md 6.1, draft 0.5: surface-only vision):

- **See-through cells** (``see_through_table``, perception's own table): classes with the data's
  ``transparent`` flag (air, glass, leaves, water, torches, short grass, ...), except ``barrier``,
  ``lava``, ``unseen`` and every class with a member in ``SIGHT_BLOCKING``. The flag is a
  rendering flag (the block does not occlude a whole cube), so it also marks doors, trapdoors,
  beds, cauldrons, shulker boxes and other blocks that hide what lies behind them; those are
  opaque here, and so is a class that merges them with see-through blocks (iron bars with iron
  doors, slime blocks with decorated pots), which is always honest. The barrier stands for the
  world edge (DESIGN.md 11.3 walls the evaluation areas with it), and nothing beyond it can look
  back in; lava is flagged transparent in the data, but no player sees through a lava pool
  [mem]. The bridge's world mirror must be clipped to the evaluation area with barrier outside
  (bridge/PROTOCOL.md 7.2). Above y 191 the world reads as air (``World.get``), as in the game.
  Every other class is *opaque* here.
- **Seen:** a cell is seen when a straight segment from the eye (feet + 1.62 [mem]) to it crosses
  only see-through cells before reaching it: to the centre of a see-through cell, or to the
  centre of a face of an opaque cell that points toward the eye (a face whose outward normal n
  has (eye - face centre) . n > 0; such a segment ends inside the neighbour across the face, so
  that neighbour must be see-through too; a cube has at most three such faces, and one clear
  segment is enough). The test is conservative: a cell whose face centres are hidden but whose
  corner shows is ``unseen`` in the window, while a panorama ray or the look ray may still hit
  it, which is honest, because such a ray proves the eye sees it.
- **The eye's own cell:** every segment and every ray starts in it, so it decides whether the eye
  sees anything. The eye looks out of it (``eye_clear``) when it is see-through, or opaque but
  without collision (signs, banners, the portal class: the head can be inside one while the body
  stands in a free cell); such a cell is then passed like a see-through one, so a wall banner at
  head height does not blind the agent [design]. What its board or cloth hides in the game is
  not modelled (block shapes are not); from inside the cell that is part of the wall a sign or
  banner hangs on, or a slice behind a hanging sign's board. Every other opaque cell blinds the
  eye [design]: lava, whose fog hides everything a block away [mem], and a cell with collision
  (a block fallen or placed into the head), over which the game draws its in-wall overlay [mem:
  the overlay is drawn for a full collision cube; any collision counts here, which is
  conservative for classes that merge shapes]. With the head in such a cell nothing is seen.
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
    plants and the world-edge barrier count). The eye's own cell is never a hit. A ray that starts
    in water passes water cells (``water`` and ``bubble_column``) until it first leaves them
    [design]; it starts in water when the eye's cell holds water, which includes the plants that
    always stand in it (``kelp``, ``seagrass``, ``tall_seagrass``, flagged liquid in the data), so
    a swimmer in a kelp forest sees the kelp and the pool, not the water around its head. Another
    plant cell met by the swimming ray is a hit. Distance = t at which the ray enters the cell;
    the face is the side it enters (a ray travelling south enters a ``north`` face [mem]).
  * an entity's box (``entity_boxes``: other live agents, mobs, dropped items), when the ray
    enters the box at a t with 0 <= t <= 64, strictly before it leaves the air cell it is in (so a
    block entered at the same t wins). Boxes that contain the eye are skipped. The face is the
    box side the ray enters (the slab axis with the largest entry t; ties x, y, z).
  * nothing: ``air`` at distance 1 (64/64), no face.

  The light of a pixel is the raw sky light and block light of the cell *in front of the face*:
  the last cell the ray crossed before the block (or the cell where it met the entity's box, or
  where it ended when nothing was hit).

  The rule "the eye's own cell is never a hit" (DESIGN.md 6.1) holds for every cell the eye
  looks out of (``eye_clear``, above): the ray starts inside it and looks out. DESIGN.md does
  not say what a head inside a cell that blinds it (a fallen block of sand, lava) sees. Here
  every pixel then shows that cell at distance 0 with no face and that cell's light [design]: it
  is what the game draws over the whole screen, and the window shows nothing either, since no
  segment leaves the eye's cell. Casting past it would show what the eye cannot see.
- **Resource sightings** (``sightings``): every resource block (``concepts.resource``) among the
  seen window cells and the panorama's block hits (entered within 64 blocks [est]), one per cell,
  sorted by distance from the eye to the cell centre (ties: x, y, z). All of them go to the
  notebook, as DESIGN.md 6.1 says ("a resource block enters memory when it is seen"); which
  places it keeps is the notebook's eviction (DESIGN.md 6.3), not perception's. ``per_class``
  can keep only the nearest n of each class, for callers that want a bound; nothing in the
  design uses it. Preset S has no panorama, so its sightings come from the window alone.
- **Schedule** (DESIGN.md 6.1): the panorama is cast once per decision; during a macro the
  window (and its sightings) is recast every ``RECAST_PERIOD`` = 20 ticks [est] while the eye
  moves (``RaySchedule``). ``WindowCache`` reuses a window while the feet have not moved and no
  block that could change it has changed, giving exactly the same output.
- **X-ray ablation** (DESIGN.md 6.1): the window shows every cell's true class, and sightings are
  the resource blocks within 64 blocks [est] (exposed or not, no rays; what Mineflayer's
  ``findBlocks`` gives), the nearest ``XRAY_PER_CLASS`` = 8 [est] of each class, because
  ``findBlocks`` returns at most its ``count`` [mem] and every block within 64 is half a million
  blocks of stone and dirt. The panorama stays as it is. Results carry ``xray=True``; never used
  for an agent that is evaluated or deployed.

Honesty: in normal mode a ray stops at the first opaque cell, so nothing here reads a cell no
straight segment from the eye reaches through see-through cells (the eye's own cell passable
when the eye looks out of it), except to learn that it is not see-through, and no entity box is
entered beyond the first block a ray meets. Changing such cells into other opaque classes that
block light and emit none, or putting entities where no ray reaches, therefore never changes
the window, the panorama, the sightings or the F3 group
(``tests/test_perception.py`` checks it in adversarial scenes and in generated cave worlds, from
the surface and from inside caves; ``tests/test_debug_info.py`` checks the look ray). Light is
the exception the design states: a light source out of sight changes the light of seen cells.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from hivemind.lang.world_words import FACES as WORLD_FACES
from hivemind.sim.ref.world import AIR, MAX_LIGHT, UNSEEN, World

if TYPE_CHECKING:
    from hivemind.rules.classes import Concepts
    from hivemind.sim.ref.debug_info import DebugInfo

EYE_HEIGHT = 1.62  # [mem] standing player eye height
ENTITY_RANGE = 32.0  # DESIGN.md 6.2: the entity table lists visible entities within 32 blocks
XRAY_RANGE = 64.0  # DESIGN.md 6.1 [est]: x-ray memory from every resource block within 64 blocks
XRAY_PER_CLASS = 8  # [est] the x-ray's nearest blocks per class (``findBlocks`` returns at most ``count`` [mem])
RECAST_PERIOD = 20  # DESIGN.md 6.1 [est]: in-macro window recasts every 20 ticks while the eye moves
NONE = -1  # "none" in the panorama's int arrays (no block, no entity, no agent, no face)
DEFAULT_BOX = (0.6, 1.8)  # [est] width, height for entity classes whose size the data lacks (fallbacks)

# Neighbour offsets in a fixed order: -x, +x, -y, +y, -z, +z (``exposed_faces`` indexes these).
FACES = np.array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)], np.int64)

# The face words and their outward normals, from the one table the observation builder also
# reads (hivemind.lang.world_words: top, bottom, north, south, east, west; north is -z [mem]).
FACE_WORDS: tuple[str, ...] = tuple(f.name for f in WORLD_FACES)
FACE_NORMALS = np.array([f.normal for f in WORLD_FACES], np.int64)


def _entered_table() -> np.ndarray:
    """``[axis][step > 0]``: the index of the face a ray enters crossing into a cell along
    ``axis`` in direction ``step`` (the face whose outward normal is -step along that axis)."""
    out = np.zeros((3, 2), np.int8)
    for axis in range(3):
        for positive in (0, 1):
            normal = [0, 0, 0]
            normal[axis] = -1 if positive else 1
            out[axis, positive] = [tuple(n) for n in FACE_NORMALS.tolist()].index(tuple(normal))
    out.setflags(write=False)
    return out


_ENTERED = _entered_table()  # x: east, west; y: top, bottom; z: south, north


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


DEFAULT_PANORAMA = PanoramaConfig()
PRESETS: dict[str, PerceptionConfig] = {
    "S": PerceptionConfig("S", 7, 3, None),  # DESIGN.md 6.1: 15 x 15 x 7, no panorama
    "M": PerceptionConfig("M", 10, 4, DEFAULT_PANORAMA),  # DESIGN.md 6.1: 21 x 21 x 9 and the panorama
}
DEFAULT_PRESET = "M"  # DESIGN.md 6.4: M is the default


def config_of(preset: str | PerceptionConfig) -> PerceptionConfig:
    """A preset by name ("S", "M"), or a config as given."""
    return PRESETS[preset] if isinstance(preset, str) else preset


# ---------------------------------------------------------------------------
# Tables and small helpers
# ---------------------------------------------------------------------------


_WOODS = ("oak", "spruce", "birch", "jungle", "acacia", "cherry", "dark_oak", "mangrove", "bamboo", "crimson",
          "warped")  # [mem] the wood sets with doors, trapdoors and signs
_COLOURS = ("white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray", "light_gray", "cyan",
            "purple", "blue", "brown", "green", "red", "black")  # [mem] the 16 dye colours
_COPPER = tuple(f"{wax}{age}copper" for wax in ("", "waxed_") for age in ("", "exposed_", "weathered_", "oxidized_"))

# Vanilla blocks that minecraft-data flags ``transparent`` but that hide what lies behind them.
# That flag is a rendering flag: the block does not occlude a whole cube, so the renderer draws
# its neighbours' faces. It is not "a player sees through it". These members block sight here
# [mem], and a class with any of them is opaque (treating a mixed class as opaque is always
# honest):
# - panels across a whole side or a whole floor or ceiling: doors and trapdoors, open or closed
#   (one class holds both states), of wood, iron and copper; signs and banners (a board or cloth
#   over at least half of a side);
# - basins: cauldrons and hoppers (walls on every side, a floor across the cell);
# - bulky shapes over most of the cell: shulker boxes, decorated pots, sniffer, dragon and turtle
#   eggs, beds (9/16 high over the whole footprint), campfires (7/16), the beacon (core and base),
#   chorus plants and flowers, the vault, pointed dripstone (up to 12/16 wide), azaleas (a slab
#   of leaves over the top half);
# - cells drawn opaque: the end portal and end gateway, and a moving piston (a block in motion).
SIGHT_BLOCKING: frozenset[str] = frozenset({
    *(f"{w}_door" for w in _WOODS), "iron_door", *(f"{c}_door" for c in _COPPER),
    *(f"{w}_trapdoor" for w in _WOODS), "iron_trapdoor", *(f"{c}_trapdoor" for c in _COPPER),
    *(f"{w}_{kind}" for w in _WOODS for kind in ("sign", "wall_sign", "hanging_sign", "wall_hanging_sign")),
    *(f"{c}_{kind}" for c in _COLOURS for kind in ("banner", "wall_banner")),
    "cauldron", "water_cauldron", "lava_cauldron", "powder_snow_cauldron", "hopper",
    "shulker_box", *(f"{c}_shulker_box" for c in _COLOURS), *(f"{c}_bed" for c in _COLOURS),
    "decorated_pot", "sniffer_egg", "dragon_egg", "turtle_egg", "campfire", "soul_campfire", "beacon",
    "chorus_plant", "chorus_flower", "vault", "pointed_dripstone", "azalea", "flowering_azalea",
    "end_portal", "end_gateway", "moving_piston",
})


@lru_cache(maxsize=8)
def _see_through_for(concepts: Concepts) -> np.ndarray:
    see = concepts.transparent.copy()
    for c in concepts.block_classes:
        if see[c.id] and any(m in SIGHT_BLOCKING for m in c.members):
            see[c.id] = False
    see[concepts.block_class("barrier").id] = False  # the world edge: nothing beyond it looks back in
    see[concepts.block_class("lava").id] = False  # [mem] nobody sees through lava
    see[UNSEEN] = False
    see.setflags(write=False)
    return see


def see_through_table(concepts: Concepts) -> np.ndarray:
    """bool[256]: the classes a line of sight passes (module docstring): ``transparent`` in the
    data, except classes with a ``SIGHT_BLOCKING`` member, barrier, lava and unseen."""
    return _see_through_for(concepts).copy()


@dataclass(frozen=True)
class _Tables:
    see: np.ndarray  # bool[256] see-through
    eye_clear: np.ndarray  # bool[256]: an eye inside a cell of this class looks out of it (module docstring)
    resource: np.ndarray  # bool[256]
    water: np.ndarray  # bool[256]: cells that hold water (open water and the plants that stand in it)
    open_water: np.ndarray  # bool[256]: water and bubble columns, nothing else (what a swimmer's ray passes)
    lava: int
    player: int  # entity class of agents
    item: int  # entity class of dropped items
    width: np.ndarray  # float64 per entity class (DEFAULT_BOX where the data has none)
    height: np.ndarray


@lru_cache(maxsize=8)
def _tables_for(concepts: Concepts) -> _Tables:
    see = _see_through_for(concepts)
    lava = concepts.block_class("lava").id
    open_water = np.zeros(UNSEEN + 1, bool)
    for name in ("water", "bubble_column"):
        open_water[concepts.block_class(name).id] = True
    # Water and the see-through classes the data flags liquid other than lava: kelp, seagrass and
    # tall seagrass, which are always water-filled in the game [mem].
    water = open_water | (concepts.liquid & see)
    water[lava] = False
    # An eye inside a see-through cell looks out of it, and so does one inside a cell without
    # collision (signs, banners, portals [design]): the game draws its in-wall overlay over the
    # screen only for a cell with a full collision cube [mem], and any collision counts here.
    eye_clear = see | (concepts.collision_height == 0)
    eye_clear[lava] = False  # [mem] lava's fog hides everything a block away
    eye_clear[UNSEEN] = False
    for a in (water, open_water, eye_clear):
        a.setflags(write=False)
    n = 1 + max(c.id for c in concepts.entity_classes)
    width = np.full(n, DEFAULT_BOX[0])
    height = np.full(n, DEFAULT_BOX[1])
    for c in concepts.entity_classes:
        if c.width is not None and c.height is not None:
            width[c.id], height[c.id] = c.width, c.height
    return _Tables(see, eye_clear, concepts.resource.copy(), water, open_water, lava,
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


def _sight_mask(t: _Tables, region: np.ndarray, eye_local: np.ndarray) -> np.ndarray:
    """The see-through mask of a region of classes for segments from an eye in cell ``eye_local``
    of it: perception's table, except that the eye's own cell is passable when ``eye_clear`` says
    the eye looks out of it (module docstring)."""
    see = t.see[region]
    i, j, k = (int(v) for v in eye_local)
    see[i, j, k] = t.eye_clear[region[i, j, k]]
    return see


def line_of_sight(world: World, eye: Sequence[float], point: Sequence[float],
                  last_cell: Sequence[int] | None = None) -> bool:
    """Whether the segment from ``eye`` to ``point`` crosses only see-through cells (the cell
    containing ``point`` excluded when ``last_cell`` names the cell just before it)."""
    eye = np.asarray(eye, float)
    point = np.asarray(point, float)
    lo = np.floor(np.minimum(eye, point)).astype(np.int64) - 1
    hi = np.floor(np.maximum(eye, point)).astype(np.int64) + 2
    region = world.region(int(lo[0]), int(lo[1]), int(lo[2]), int(hi[0]), int(hi[1]), int(hi[2]))
    see = _sight_mask(tables(world), region, np.floor(eye).astype(np.int64) - lo)
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
    eye_arr = np.asarray(eye, float)
    o = np.asarray(origin, np.int64)
    s = np.asarray(shape, np.int64)
    ec = np.floor(eye_arr).astype(np.int64)
    lo = np.minimum(o, ec) - 1  # the box and the eye's cell, with a ring of neighbours
    hi = np.maximum(o + s, ec + 1) + 1
    region = world.region(int(lo[0]), int(lo[1]), int(lo[2]), int(hi[0]), int(hi[1]), int(hi[2]))
    see = _sight_mask(tables(world), region, ec - lo)
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
    return seen.reshape(tuple(int(v) for v in s)), len(owner)


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
    The output equals a fresh call. Keep one per agent. Worlds are told apart by
    ``World.serial`` (a copy, such as a reset's copy of the pristine world, is another world) and
    ``World.generation`` (a ``load``), never by ``id()``, which CPython reuses."""

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
        key = (world.serial, world.generation, feet_t, cfg, xray)
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
        except ValueError:  # the log was trimmed past our cursor (or the cursor is not this world's)
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

    def without_agent(self, agent: int | None) -> EntityBoxes:
        """These boxes without agent slot ``agent``'s (the observer's own box), in the same order."""
        if agent is None:
            return self
        keep = self.agent != agent
        if keep.all():
            return self
        return EntityBoxes(self.lo[keep], self.hi[keep], self.cls[keep], self.agent[keep], self.source[keep],
                           self.slot[keep])


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


def ray_box_entry(eye: Sequence[float], dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray, *,
                  inside_hits: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Slab test: for rays from ``eye`` along ``dirs`` (n, 3) and boxes ``lo``/``hi`` (m, 3), the t
    at which each ray enters each box (n, m; inf when it misses, when the box is behind the eye or
    when the box contains the eye) and the slab axis of that entry (largest entry t; ties x, y,
    z). t is in units of the direction's length.

    ``inside_hits``: a box that contains the eye (``lo <= eye < hi`` on every axis, the game's
    ``AABB.contains`` [mem]) counts as entered at t = 0 instead, as the game's crosshair pick
    does (``ProjectileUtil.getEntityHitResult`` [mem]); the F3 group uses it, the panorama not."""
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
    t = np.where(hit, t_near, np.inf)
    if inside_hits:
        contains = np.all((lo <= eye[None, :]) & (eye[None, :] < hi), axis=1)
        t[:, contains] = 0.0
    return t, axis


# ---------------------------------------------------------------------------
# The semantic panorama
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def panorama_directions(cfg: PanoramaConfig = DEFAULT_PANORAMA) -> np.ndarray:
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
    agent slot (its box is never a hit, also when it is among ``boxes``); ``boxes``: precomputed
    ``entity_boxes``, which may be shared by every observer."""
    cfg = config if config is not None else DEFAULT_PANORAMA
    t = tables(world)
    eye_arr = np.asarray(eye, float)
    dirs = panorama_directions(cfg).reshape(-1, 3)
    n = len(dirs)
    shape = (cfg.rows, cfg.columns)
    ec = np.floor(eye_arr).astype(np.int64)
    eye_cls = world.get(int(ec[0]), int(ec[1]), int(ec[2]))
    if not t.eye_clear[eye_cls]:
        # The head is inside lava or a block: it is all the eye sees (the eye-cell exception).
        sky, blk = world.light_points(ec[None, :])
        return Panorama(tuple(float(v) for v in eye_arr), cfg, np.full(shape, eye_cls, np.int16),
                        np.full(shape, NONE, np.int16), np.full(shape, NONE, np.int16),
                        np.full(shape, NONE, np.int8), np.zeros(shape, np.float32),
                        np.full(shape, np.float32(sky[0]) / np.float32(MAX_LIGHT), np.float32),
                        np.full(shape, np.float32(blk[0]) / np.float32(MAX_LIGHT), np.float32),
                        np.broadcast_to(ec.astype(np.int32), (*shape, 3)).copy(), 0)
    max_t = float(cfg.range)
    reach = math.ceil(max_t) + 2
    y_lo = math.floor(eye_arr[1] + min(0.0, float(dirs[:, 1].min())) * max_t) - 2
    y_hi = math.floor(eye_arr[1] + max(0.0, float(dirs[:, 1].max())) * max_t) + 3
    base = np.array([ec[0] - reach, y_lo, ec[2] - reach], np.int64)
    top = np.array([ec[0] + reach + 1, y_hi, ec[2] + reach + 1], np.int64)
    classes = world.region(int(base[0]), int(base[1]), int(base[2]), int(top[0]), int(top[1]), int(top[2]))

    # Entities: the nearest box entry per ray, within range.
    boxes = entity_boxes(world, exclude_agent=exclude_agent) if boxes is None else boxes.without_agent(exclude_agent)
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
    swim = np.full(n, bool(t.water[eye_cls]))  # the head is in water, or in a plant standing in it
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
        wet = t.open_water[c]  # a plant in the water is a hit, as the pool's floor is
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

    ``counts`` is the number sighted per class (at most ``per_class`` when one is given).
    ``rays`` is a cost figure (segments traced and panorama cells walked); it is left out of
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
              pano: Panorama | None = None, *, per_class: int | None = None) -> Sightings:
    """Resource blocks among the seen window cells and the panorama's block hits (module
    docstring): all of them, or the nearest ``per_class`` of each class when one is given."""
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
                   per_class: int | None = XRAY_PER_CLASS) -> Sightings:
    """Ablation only: the resource blocks whose centres lie within ``sight_range`` of the eye, seen
    or not (no rays), the nearest ``per_class`` of each class (None: all of them, which is half a
    million blocks of stone and dirt in a test world); flagged ``xray``."""
    resource = tables(world).resource
    eye_arr = np.asarray(eye, float)
    ec = np.floor(eye_arr).astype(np.int64)
    r = math.ceil(sight_range) + 1
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
    """One perception call: the window, the panorama (preset M at a decision, else None), the
    sightings for the notebook and the F3 group (``debug_info.DebugInfo``, when the call names
    the observer's agent slot, else None). ``xray`` marks the ablation; such a result must never
    reach an agent that is evaluated or deployed (DESIGN.md 6.1)."""

    tick: int
    feet: tuple[float, float, float]
    eye: tuple[float, float, float]
    preset: str
    window: VoxelWindow
    panorama: Panorama | None
    sightings: Sightings
    xray: bool
    f3: DebugInfo | None = None


def perceive(world: World, feet: Sequence[float] | None = None, tick: int | None = None, *,
             preset: str | PerceptionConfig = DEFAULT_PRESET, agent: int | None = None, panorama: bool = True,
             f3: bool = True, xray: bool = False, cache: WindowCache | None = None,
             per_class: int | None = None) -> Perception:
    """The window (through ``cache`` when one is given), the panorama when ``panorama`` is set and
    the preset has one (pass False for in-macro recasts), the sightings from both, and the F3
    group when ``agent`` is given and ``f3`` is set.

    ``agent``: the observer's slot. Its own box is never a panorama hit, its feet are the default
    for ``feet``, and the F3 group is computed for it (``feet`` must then be where the slot
    stands, so that the window and the F3 group describe one viewpoint). ``tick`` defaults to
    ``world.tick``. ``xray``: the ablation (module docstring), flagged in the result, its window
    and its sightings; the panorama and the F3 group are the same as without it. ``per_class``:
    keep only the nearest n sightings of each class (honest sightings keep all by default, the
    x-ray its ``XRAY_PER_CLASS``)."""
    cfg = config_of(preset)
    tick = world.tick if tick is None else tick
    if feet is None:
        if agent is None:
            raise ValueError("perceive needs the feet position or the observer's agent slot")
        feet = world.agents.pos[agent]
    feet_t = (float(feet[0]), float(feet[1]), float(feet[2]))
    group = None
    if agent is not None and f3:
        from hivemind.sim.ref.debug_info import debug_info  # debug_info imports this module

        if feet_t != tuple(float(v) for v in world.agents.pos[agent]):
            raise ValueError(f"feet {feet_t} are not where agent slot {agent} stands")
        group = debug_info(world, agent)
    eye = eye_of(feet_t)
    window = (cache.get(world, feet_t, cfg, xray=xray) if cache is not None
              else voxel_window(world, feet_t, cfg, xray=xray))
    pano = None
    if panorama and cfg.panorama is not None:
        pano = _cast_panorama(world, eye, cfg.panorama, exclude_agent=agent)
    if xray:
        seen = xray_sightings(world, eye, tick, per_class=XRAY_PER_CLASS if per_class is None else per_class)
    else:
        seen = sightings(world.concepts, eye, tick, window, pano, per_class=per_class)
    return Perception(int(tick), feet_t, eye, cfg.name, window, pano, seen, xray, group)

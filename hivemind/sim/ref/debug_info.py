"""The F3 debug group: what the vanilla debug screen shows, for one agent (DESIGN.md 6.2; contracts 6).

The owner asked for the debug screen's inputs (coordinates, facing, the block looked at, time of
day, time passed). ``debug_info`` computes them from the world, and the bridge runs it on its
world mirror, never on the client's debug screen. Usage::

    from hivemind.sim.ref.debug_info import debug_info
    f3 = debug_info(world, agent)          # agent: slot in world.agents
    f3.x, f3.y, f3.z                       # x and z relative to home, y absolute
    f3.facing, f3.yaw, f3.pitch            # 'north', yaw wrapped to [-180, 180), pitch
    f3.block, f3.block_face                # targeted block class id and face word, or None
    f3.fluid, f3.entity                    # 'water' / 'lava' / None; entity class id or None
    f3.time_word, f3.day, f3.ticks         # 'dusk', day number, ticks since the episode began
    f3.biome_name, f3.sky_light            # 'birch_forest' (lang.phrase("biome", ...)), 0..15

Fields (DESIGN.md 6.2, draft 0.5). Words are returned as concept names, which the observation
builder turns into words: the facing and face words are direction and face concepts
(``north``, ``top``), the time word a time concept (``dusk``), the biome a biome class
(``birch_forest`` -> ``birch forest``), the fluid a block class (``water``).

- **Position:** feet x and z minus the agent's home (``world.agents.home``: its spawn point or,
  from M9, its team's base), y absolute (ore heights depend on it). Absolute x and z are never
  given (evaluation areas sit hundreds of blocks apart). A slot without a home raises.
- **Facing:** yaw wrapped to [-180, 180) and pitch, in Minecraft's convention [mem] (yaw 0 south,
  90 west, 180 north, -90 east; pitch -90 up, +90 down; ``Mth.wrapDegrees`` [mem]), and F3's
  4-way word, ``Direction.fromYRot``: south, west, north, east for floor(yaw / 90 + 0.5) & 3 [mem].
- **Look ray:** from the eye (feet + 1.62 [mem]) along the view vector the game computes,
  ``Entity.calculateViewVector`` with float32 ``Mth.sin``/``Mth.cos`` (a 65,536-entry sine table)
  [mem], so the ray is bit-exact and needs no libm. Cells are walked with perception's DDA (ties
  x, y, z), t in blocks, starting with the eye's own cell (t = 0):

  * ``water`` (or a bubble column) is passed; the first one met is the **targeted fluid**
    ``water``.
  * ``lava`` stops the ray (it is not see-through, DESIGN.md 6.1: a stated deviation from F3); if
    no fluid came first it is the targeted fluid ``lava``.
  * any other cell that is not air is the **targeted block**: class, distance t (where the ray
    enters it) and the face word of the side entered. If it is the eye's own cell (the head inside
    a block), t = 0 and the face is the game's choice for a ray that starts inside a shape,
    ``Direction.getNearest(d).getOpposite()`` (ties: down, up, north, south, west, east) [mem].
  * a cell entered beyond 4.5 blocks (the block reach [mem]; F3 itself picks to 20 [mem]) ends
    the ray with no block. A fluid counts only if it comes before the targeted block, or within
    4.5 when there is none.
  * the **targeted entity** is the first entity box the ray enters within 3.0 blocks (the entity
    reach [mem]) and strictly nearer than the targeted block's entry point or a lava cell: other
    live agents (players) and mobs; dropped items are not pickable in the game [mem] and the
    agent's own box and boxes containing the eye are skipped.

  None of it can name something behind a wall: the ray stops at the first opaque cell.
- **Time:** the tick within the day (``world.time_of_day``) and its word: ``day`` 0-11999,
  ``dusk`` 12000-12999, ``night`` 13000-22999, ``dawn`` 23000-23999 [design, from [15]]; the day
  number floor(day time / 24000) [mem: F3's Local Difficulty "Day N"], where the day time is the
  game's clock (a night skip moves it); ``ticks``, the game ticks since the episode began
  (``world.tick``).
- **Biome** of the feet's column (``world.biome_at``; test worlds assign one per column).
- **Light** at the feet cell: raw sky light and block light, F3's "Client Light" parts [mem].

Honest by construction: it reads the agent's own slot, the world clock, the biome and light of
the agent's own cell, and cells along a ray that stops where sight stops.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from hivemind.sim.ref.perception import (
    FACE_WORDS,
    EntityBoxes,
    entered_face,
    entity_boxes,
    eye_of,
    ray_box_entry,
    tables,
)
from hivemind.sim.ref.world import AIR, World

BLOCK_REACH = 4.5  # [mem] survival block interaction range (DESIGN.md 4.5)
ENTITY_REACH = 3.0  # [mem] survival entity interaction range (DESIGN.md 4.5)
TICKS_PER_DAY = 24_000  # [mem] 20 minutes at 20 ticks per second

# DESIGN.md 6.2 [design, from [15]]: (word, first tick, last tick + 1); together they cover the day.
TIME_WORDS: tuple[tuple[str, int, int], ...] = (("day", 0, 12_000), ("dusk", 12_000, 13_000),
                                                ("night", 13_000, 23_000), ("dawn", 23_000, TICKS_PER_DAY))
_FACING = ("south", "west", "north", "east")  # [mem] Direction.from2DDataValue order


def time_word(day_time: int) -> str:
    """``day``, ``dusk``, ``night`` or ``dawn`` for a day-time counter (any day)."""
    t = int(day_time) % TICKS_PER_DAY
    return next(w for w, lo, hi in TIME_WORDS if lo <= t < hi)


def day_number(day_time: int) -> int:
    """floor(day time / 24000) [mem]."""
    return int(day_time) // TICKS_PER_DAY


def facing_word(yaw: float) -> str:
    """F3's "Facing" word: ``Direction.fromYRot``, floor(yaw / 90 + 0.5) & 3 -> south, west,
    north, east [mem]."""
    return _FACING[math.floor(float(yaw) / 90.0 + 0.5) & 3]


def wrap_degrees(angle: float) -> float:
    """``Mth.wrapDegrees`` for a float angle: into [-180, 180) [mem]."""
    a = math.fmod(float(np.float32(angle)), 360.0)
    if a >= 180.0:
        a -= 360.0
    if a < -180.0:
        a += 360.0
    return float(np.float32(a))


# -- the game's view vector -----------------------------------------------------------------


@lru_cache(maxsize=1)
def _sin_table() -> np.ndarray:
    """``Mth.SIN[i] = (float)Math.sin(i * PI * 2.0 / 65536.0)`` [mem]."""
    return np.sin(np.arange(65536, dtype=np.float64) * math.pi * 2.0 / 65536.0).astype(np.float32)


def _f32(x: float) -> np.float32:
    return np.float32(x)


def _mth_sin(x: np.float32) -> np.float32:
    """``Mth.sin``: SIN[(int)(x * 10430.378F) & 65535] [mem]."""
    return _sin_table()[int(_f32(x * _f32(10430.378))) & 65535]


def _mth_cos(x: np.float32) -> np.float32:
    """``Mth.cos``: SIN[(int)(x * 10430.378F + 16384.0F) & 65535] [mem]."""
    return _sin_table()[int(_f32(_f32(x * _f32(10430.378)) + _f32(16384.0))) & 65535]


def view_vector(yaw: float, pitch: float) -> tuple[float, float, float]:
    """``Entity.calculateViewVector(xRot = pitch, yRot = yaw)`` [mem]: f = pitch * (float)(pi/180),
    g = -yaw * (float)(pi/180); (sin g cos f, -sin f, cos g cos f) in float32, widened to double."""
    rad = _f32(math.pi / 180.0)
    f = _f32(_f32(pitch) * rad)
    g = _f32(-_f32(yaw) * rad)
    h, i, j, k = _mth_cos(g), _mth_sin(g), _mth_cos(f), _mth_sin(f)
    return float(_f32(i * j)), float(-k), float(_f32(h * j))


def _inside_face(d: Sequence[float]) -> int:
    """``Direction.getNearest(d).getOpposite()`` as a face index [mem]: the direction with the largest
    d . normal in the order down, up, north, south, west, east (first wins), turned around."""
    order = ((0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0))
    best, best_dot = order[0], -math.inf
    for n in order:
        dot = d[0] * n[0] + d[1] * n[1] + d[2] * n[2]
        if dot > best_dot:
            best, best_dot = n, dot
    opposite = tuple(-v for v in best)
    names = {(0, 1, 0): "top", (0, -1, 0): "bottom", (0, 0, -1): "north", (0, 0, 1): "south",
             (1, 0, 0): "east", (-1, 0, 0): "west"}
    return FACE_WORDS.index(names[opposite])


# -- the result ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class DebugInfo:
    """The F3 group of one agent at one tick (module docstring). Absent targets are None."""

    x: float  # feet x - home x
    y: float  # feet y (absolute)
    z: float  # feet z - home z
    yaw: float  # degrees, wrapped to [-180, 180)
    pitch: float  # degrees, -90 up .. 90 down
    facing: str  # north, east, south, west
    block: int | None  # targeted block class id
    block_distance: float | None
    block_face: str | None  # top, bottom, north, south, east, west
    fluid: str | None  # water, lava
    fluid_distance: float | None
    entity: int | None  # targeted entity class id
    entity_agent: int | None  # its agent slot when it is a player (the builder names it, e.g. blue 7)
    entity_distance: float | None
    tick_of_day: int  # 0..23999
    time_word: str  # day, dusk, night, dawn
    day: int  # floor(day time / 24000)
    ticks: int  # game ticks since the episode began
    biome: int  # biome class id (vanilla registry id)
    biome_name: str  # its concept name (lang.phrase("biome", name))
    sky_light: int  # raw sky light at the feet, 0..15
    block_light: int  # block light at the feet, 0..15


@dataclass(frozen=True)
class LookTargets:
    """What the look ray meets (module docstring); distances in blocks from the eye."""

    block: int | None = None
    block_distance: float | None = None
    block_face: str | None = None
    fluid: str | None = None
    fluid_distance: float | None = None
    entity: int | None = None
    entity_agent: int | None = None
    entity_distance: float | None = None


def look_targets(world: World, eye: Sequence[float], direction: Sequence[float], *,
                 boxes: EntityBoxes | None = None, exclude_agent: int | None = None,
                 reach: float = BLOCK_REACH, entity_reach: float = ENTITY_REACH) -> LookTargets:
    """The targeted block, fluid and entity along the ray from ``eye`` along the unit vector
    ``direction`` (module docstring). ``boxes``: precomputed ``entity_boxes`` (dropped items are
    not targets; they are left out here)."""
    t = tables(world)
    e = [float(v) for v in eye]
    d = [float(v) for v in direction]
    cell = [math.floor(v) for v in e]
    step = [(v > 0) - (v < 0) for v in d]
    inv = [1.0 / v if v != 0 else math.inf for v in d]
    t_delta = [abs(v) for v in inv]
    t_max = [((cell[k] + 1 if d[k] > 0 else cell[k]) - e[k]) * inv[k] if d[k] != 0 else math.inf for k in range(3)]
    block = block_t = block_face = fluid = fluid_t = None
    stop_t = math.inf  # where the ray stops: the targeted block's entry point or a lava cell
    t_in, axis = 0.0, -1  # the eye's own cell first
    while t_in <= reach:
        c = world.get(cell[0], cell[1], cell[2])
        if t.water[c]:
            if fluid is None:
                fluid, fluid_t = "water", t_in
        elif c == t.lava:
            if fluid is None:
                fluid, fluid_t = "lava", t_in
            stop_t = t_in
            break
        elif c != AIR:
            block, block_t = c, t_in
            block_face = FACE_WORDS[_inside_face(d) if axis < 0 else entered_face(axis, step[axis])]
            stop_t = t_in
            break
        axis = min(range(3), key=lambda k: t_max[k])  # first minimum: x before y before z
        t_in = t_max[axis]
        cell[axis] += step[axis]
        t_max[axis] += t_delta[axis]

    entity = entity_agent = entity_t = None
    if boxes is None:
        boxes = entity_boxes(world, exclude_agent=exclude_agent, items=False)
    keep = np.flatnonzero(boxes.source != 2)  # dropped items cannot be targeted [mem]
    if keep.size:
        tb, _ = ray_box_entry(e, np.asarray([d]), boxes.lo[keep], boxes.hi[keep])
        k = int(np.argmin(tb[0]))  # ties: the first box in ``entity_boxes`` order
        tk = float(tb[0, k])
        if tk <= entity_reach and tk < stop_t:
            i = int(keep[k])
            entity, entity_t = int(boxes.cls[i]), tk
            entity_agent = int(boxes.agent[i]) if boxes.agent[i] >= 0 else None
    return LookTargets(block, block_t, block_face, fluid, fluid_t, entity, entity_agent, entity_t)


def debug_info(world: World, agent: int, *, reach: float = BLOCK_REACH,
               entity_reach: float = ENTITY_REACH) -> DebugInfo:
    """The F3 group of agent slot ``agent`` (module docstring)."""
    a = world.agents
    if not a.in_use[agent]:
        raise ValueError(f"agent slot {agent} is not in use")
    home = a.home[agent]
    if not np.isfinite(home).all():
        raise ValueError(f"agent slot {agent} has no home (World.alloc_agent sets it)")
    feet = [float(v) for v in a.pos[agent]]
    yaw, pitch = float(a.yaw[agent]), float(a.pitch[agent])
    look = look_targets(world, eye_of(feet), view_vector(yaw, pitch), exclude_agent=agent, reach=reach,
                        entity_reach=entity_reach)
    fx, fy, fz = (math.floor(v) for v in feet)
    sky, blk = world.light_points(np.array([[fx, fy, fz]]))
    biome = world.biome_at(feet[0], feet[2])
    return DebugInfo(
        x=feet[0] - float(home[0]), y=feet[1], z=feet[2] - float(home[2]),
        yaw=wrap_degrees(yaw), pitch=pitch, facing=facing_word(yaw),
        block=look.block, block_distance=look.block_distance, block_face=look.block_face,
        fluid=look.fluid, fluid_distance=look.fluid_distance,
        entity=look.entity, entity_agent=look.entity_agent, entity_distance=look.entity_distance,
        tick_of_day=world.time_of_day, time_word=time_word(world.day_time), day=day_number(world.day_time),
        ticks=world.tick, biome=biome, biome_name=world.concepts.biome(biome).name,
        sky_light=int(sky[0]), block_light=int(blk[0]))

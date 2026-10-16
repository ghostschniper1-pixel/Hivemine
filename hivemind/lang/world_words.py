"""Words for where and when: directions, block faces and times of day (DESIGN.md 6.1, 6.2;
docs/contracts.md 4.2, 6).

Small typed tables for the observation builder (the F3 debug group and the semantic panorama)
and for the macros that turn a direction word into a heading. Each entry's ``name`` is a word of
the vocabulary and the name of a concept with that one-word phrase (``lang.phrase("face", "top")``
is ``("top",)``, ``lang.phrase("time", "dusk")`` is ``("dusk",)``, ``lang.phrase("dir",
"north")`` is ``("north",)``); the numbers next to it are what make the word true.

    from hivemind.lang.world_words import facing_word, time_word, entered_face, FACES
    facing_word(170.0)          # 'north' (F3's "Facing" from the yaw)
    time_word(12_500)           # 'dusk'
    day_number(30_000)          # 1
    entered_face(2, +1)         # 'north': a ray travelling south (+z) enters a cell's north face

Coordinates are Minecraft's [mem]: x grows to the east, y up and z to the south, so north is -z.
Yaw is in degrees: 0 faces south, 90 west, 180 (or -180) north and -90 east; pitch -90 looks
straight up and +90 straight down [mem]. The direction a yaw faces is (-sin yaw, cos yaw) in
(x, z) [mem].

Biomes are concepts of the class tables, not of this module: ``lang.phrase("biome", "birch_forest")``
is ``("birch", "forest")`` and ``lang.class_phrase_ids("biome")`` gives every biome's word ids by
its registry id (contracts 3).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from hivemind.lang import grammar as G

# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Direction:
    """A compass direction word with its step in the world and the yaw that faces it."""

    name: str
    dx: int  # step in x (east is +1) [mem]
    dz: int  # step in z (south is +1) [mem]
    yaw: float  # Minecraft yaw facing this way, degrees in (-180, 180] [mem]

    @property
    def unit(self) -> tuple[float, float]:
        """(x, z) of length 1 (diagonals are 1/sqrt(2) on each axis)."""
        n = math.hypot(self.dx, self.dz)
        return (self.dx / n, self.dz / n)


# The eight directions of explore {dir8}, in G.DIR8 order (north first, clockwise).
DIRECTIONS: tuple[Direction, ...] = (
    Direction("north", 0, -1, 180.0),
    Direction("northeast", 1, -1, -135.0),
    Direction("east", 1, 0, -90.0),
    Direction("southeast", 1, 1, -45.0),
    Direction("south", 0, 1, 0.0),
    Direction("southwest", -1, 1, 45.0),
    Direction("west", -1, 0, 90.0),
    Direction("northwest", -1, -1, 135.0),
)
DIRECTION: dict[str, Direction] = {d.name: d for d in DIRECTIONS}
DIRECTIONS4: tuple[Direction, ...] = tuple(DIRECTION[n] for n in G.DIR4)  # strip mine {dir4}, F3's facing
DIRECTION_WORDS: tuple[str, ...] = tuple(d.name for d in DIRECTIONS)

# F3's "Facing" word: the game's horizontal direction of the yaw, Direction.fromYRot, whose 2-D
# data values are 0 south, 1 west, 2 north, 3 east [mem].
_FACING_BY_2D_VALUE = ("south", "west", "north", "east")


def facing_word(yaw: float) -> str:
    """The 4-way direction word F3 shows for a yaw in degrees ("Facing: north"): the nearest of
    south (0), west (90), north (180) and east (-90); exactly halfway rounds clockwise
    (``floor(yaw / 90 + 0.5) & 3``, as the game does [mem])."""
    if not math.isfinite(yaw):
        raise ValueError(f"yaw must be finite, got {yaw!r}")
    return _FACING_BY_2D_VALUE[math.floor(yaw / 90.0 + 0.5) & 3]


# ---------------------------------------------------------------------------
# Block faces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Face:
    """A side of a block cell (or of an entity's box), named by the way it points."""

    name: str
    normal: tuple[int, int, int]  # outward unit normal (x, y, z) [mem: north is -z]


# The face words of the panorama and the F3 targeted block, in the order DESIGN.md 6.2 lists them.
FACES: tuple[Face, ...] = (
    Face("top", (0, 1, 0)),
    Face("bottom", (0, -1, 0)),
    Face("north", (0, 0, -1)),
    Face("south", (0, 0, 1)),
    Face("east", (1, 0, 0)),
    Face("west", (-1, 0, 0)),
)
FACE: dict[str, Face] = {f.name: f for f in FACES}
FACE_WORDS: tuple[str, ...] = tuple(f.name for f in FACES)
_FACE_BY_NORMAL: dict[tuple[int, int, int], str] = {f.normal: f.name for f in FACES}


def face_of_normal(normal: Sequence[int]) -> str:
    """The face word of an outward unit normal: ``(0, 1, 0)`` is ``top``."""
    key = tuple(int(c) for c in normal)
    try:
        return _FACE_BY_NORMAL[key]  # type: ignore[index]
    except KeyError:
        raise ValueError(f"not an axis-aligned unit normal: {tuple(normal)!r}") from None


def entered_face(axis: int, step: int) -> str:
    """The face a ray enters when it crosses into a cell along ``axis`` (0 x, 1 y, 2 z) moving in
    the ``step`` direction (+1 or -1): the face whose normal points back along the ray, so a ray
    travelling south (+z) hits a ``north`` face and a ray falling (-y) hits a ``top`` face."""
    if axis not in (0, 1, 2) or step not in (1, -1):
        raise ValueError(f"axis must be 0, 1 or 2 and step +1 or -1, got {axis}, {step}")
    normal = [0, 0, 0]
    normal[axis] = -step
    return _FACE_BY_NORMAL[tuple(normal)]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Times of day
# ---------------------------------------------------------------------------

TICKS_PER_DAY = 24_000  # [mem] one Minecraft day: 20 minutes at 20 ticks per second


@dataclass(frozen=True)
class TimeOfDay:
    """A time word and the ticks within the day it is true of: ``start <= tick < stop``."""

    name: str
    start: int
    stop: int


# DESIGN.md 6.2 [design, from [15]]: day 0-11999, dusk 12000-12999, night 13000-22999,
# dawn 23000-23999. Together they cover the day exactly once, in order.
TIMES_OF_DAY: tuple[TimeOfDay, ...] = (
    TimeOfDay("day", 0, 12_000),
    TimeOfDay("dusk", 12_000, 13_000),
    TimeOfDay("night", 13_000, 23_000),
    TimeOfDay("dawn", 23_000, TICKS_PER_DAY),
)
TIME_OF_DAY: dict[str, TimeOfDay] = {t.name: t for t in TIMES_OF_DAY}
TIME_WORDS: tuple[str, ...] = tuple(t.name for t in TIMES_OF_DAY)


def tick_of_day(day_time: int) -> int:
    """The tick within the day (0..23999) of the game's day-time counter."""
    return int(day_time) % TICKS_PER_DAY


def time_word(day_time: int) -> str:
    """The time word of a day-time counter: ``day``, ``dusk``, ``night`` or ``dawn``."""
    t = tick_of_day(day_time)
    for w in TIMES_OF_DAY:
        if w.start <= t < w.stop:
            return w.name
    raise AssertionError(t)  # pragma: no cover: TIMES_OF_DAY covers the day


def day_number(day_time: int) -> int:
    """floor(day time / 24000): F3's "Day N" [mem: Local Difficulty line]. The day time is the
    game's clock, which a night skip moves forward (DESIGN.md 6.2)."""
    return int(day_time) // TICKS_PER_DAY


def check_tables() -> None:
    """The tables agree with the grammar and with themselves (run by the tests and the
    vocabulary build)."""
    if DIRECTION_WORDS != G.DIR8:
        raise ValueError("DIRECTIONS must list G.DIR8 in order")
    for d in DIRECTIONS:
        x, z = -math.sin(math.radians(d.yaw)), math.cos(math.radians(d.yaw))
        ux, uz = d.unit
        if abs(x - ux) > 1e-9 or abs(z - uz) > 1e-9:
            raise ValueError(f"{d.name}: yaw {d.yaw} does not face ({d.dx}, {d.dz})")
    for d in DIRECTIONS4:
        if facing_word(d.yaw) != d.name:
            raise ValueError(f"{d.name}: facing_word({d.yaw}) is {facing_word(d.yaw)}")
    if {FACE[n].normal for n in G.DIR4} != {(d.dx, 0, d.dz) for d in DIRECTIONS4}:
        raise ValueError("the side faces must point the way their direction words do")
    if len(set(_FACE_BY_NORMAL)) != 6:
        raise ValueError("six faces with six different normals")
    edges = [t.start for t in TIMES_OF_DAY] + [TIMES_OF_DAY[-1].stop]
    if edges[0] != 0 or edges[-1] != TICKS_PER_DAY or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("TIMES_OF_DAY must cover 0..23999 in order")
    for t, nxt in zip(TIMES_OF_DAY, TIMES_OF_DAY[1:]):
        if t.stop != nxt.start:
            raise ValueError(f"{t.name} must end where {nxt.name} starts")

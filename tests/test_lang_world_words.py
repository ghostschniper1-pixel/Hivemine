"""Tests for the typed tables of direction, face and time words (hivemind.lang.world_words;
DESIGN.md 6.1, 6.2)."""

from __future__ import annotations

import math

import pytest

from hivemind.lang import Language
from hivemind.lang import grammar as G
from hivemind.lang import world_words as W


@pytest.fixture(scope="module")
def lang() -> Language:
    return Language.load()


def test_tables_are_consistent() -> None:
    W.check_tables()
    assert W.DIRECTION_WORDS == G.DIR8
    assert tuple(d.name for d in W.DIRECTIONS4) == G.DIR4
    assert W.FACE_WORDS == ("top", "bottom", "north", "south", "east", "west")  # DESIGN.md 6.2 order
    assert W.TIME_WORDS == ("day", "dusk", "night", "dawn")


def test_every_word_is_a_concept_of_the_vocabulary(lang: Language) -> None:
    for d in W.DIRECTIONS:
        assert lang.phrase("dir", d.name) == (d.name,)
    for f in W.FACES:
        assert lang.phrase("face", f.name) == (f.name,) and lang.concept("face", (f.name,)) == f.name
    for t in W.TIMES_OF_DAY:
        assert lang.phrase("time", t.name) == (t.name,) and lang.concept("time", (t.name,)) == t.name


def test_directions_point_the_minecraft_way() -> None:
    """North is -z and east +x; yaw 0 faces south, 90 west, 180 north, -90 east [mem]."""
    d = W.DIRECTION
    assert (d["north"].dx, d["north"].dz) == (0, -1) and (d["east"].dx, d["east"].dz) == (1, 0)
    assert (d["southwest"].dx, d["southwest"].dz) == (-1, 1)
    assert [d[n].yaw for n in ("south", "west", "north", "east")] == [0.0, 90.0, 180.0, -90.0]
    for x in W.DIRECTIONS:
        ux, uz = x.unit
        assert math.isclose(math.hypot(ux, uz), 1.0)


@pytest.mark.parametrize("yaw,word", [
    (0.0, "south"), (44.9, "south"), (45.0, "west"), (90.0, "west"), (134.9, "west"), (135.0, "north"),
    (180.0, "north"), (-180.0, "north"), (-135.1, "north"), (-135.0, "east"), (-90.0, "east"), (-45.1, "east"),
    (-45.0, "south"), (-44.9, "south"), (270.0, "east"), (360.0, "south"), (-360.0, "south"), (720.0 + 91, "west"),
])
def test_facing_word(yaw: float, word: str) -> None:
    """F3's "Facing": floor(yaw / 90 + 0.5) & 3 over (south, west, north, east) [mem]."""
    assert W.facing_word(yaw) == word


def test_facing_word_rejects_nan() -> None:
    with pytest.raises(ValueError):
        W.facing_word(float("nan"))


def test_faces() -> None:
    assert W.face_of_normal((0, 1, 0)) == "top" and W.face_of_normal([0, -1, 0]) == "bottom"
    assert W.face_of_normal((0, 0, -1)) == "north" and W.face_of_normal((1, 0, 0)) == "east"
    for f in W.FACES:
        assert W.face_of_normal(f.normal) == f.name
    # the face a ray enters points back along the ray (DESIGN.md 6.2: a ray travelling south hits a north face)
    assert W.entered_face(2, +1) == "north" and W.entered_face(2, -1) == "south"
    assert W.entered_face(0, +1) == "west" and W.entered_face(0, -1) == "east"
    assert W.entered_face(1, -1) == "top" and W.entered_face(1, +1) == "bottom"
    for bad in ((1, 1, 0), (0, 0, 0), (0, 2, 0)):
        with pytest.raises(ValueError):
            W.face_of_normal(bad)
    with pytest.raises(ValueError):
        W.entered_face(3, 1)
    with pytest.raises(ValueError):
        W.entered_face(0, 0)


@pytest.mark.parametrize("tick,word", [
    (0, "day"), (11_999, "day"), (12_000, "dusk"), (12_999, "dusk"), (13_000, "night"), (22_999, "night"),
    (23_000, "dawn"), (23_999, "dawn"), (24_000, "day"), (24_000 * 5 + 12_500, "dusk"),
])
def test_time_word(tick: int, word: str) -> None:
    """day 0-11999, dusk 12000-12999, night 13000-22999, dawn 23000-23999 (DESIGN.md 6.2)."""
    assert W.time_word(tick) == word


def test_time_of_day_and_day_number() -> None:
    assert W.TICKS_PER_DAY == 24_000
    assert [W.tick_of_day(t) for t in (0, 23_999, 24_000, 30_000)] == [0, 23_999, 0, 6_000]
    assert [W.day_number(t) for t in (0, 23_999, 24_000, 30_000, 240_000)] == [0, 0, 1, 1, 10]
    covered = sorted(t for w in W.TIMES_OF_DAY for t in range(w.start, w.stop))
    assert covered == list(range(W.TICKS_PER_DAY))  # every tick of the day has exactly one word
    # the day turns from dawn to day at tick 0 of the next day (contracts 4.3, survive until morning)
    assert W.time_word(23_999) == "dawn" and W.time_word(24_000) == "day" and W.day_number(24_000) == 1

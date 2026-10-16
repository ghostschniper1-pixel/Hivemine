"""Tests for hivemind.sim.ref.rng: the counter-based RNG (DESIGN.md 4.7).

The golden values pin the exact algorithm; a C++ port must reproduce every one of them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hivemind.sim.ref.rng import (
    GAMMA,
    MASK64,
    CounterRNG,
    DrawKey,
    Purpose,
    exp_neg,
    mix64,
    pack_key,
    poisson_draws,
)

SEED = 20261016


@pytest.fixture(scope="module")
def rng() -> CounterRNG:
    return CounterRNG(SEED)


# -- golden values ------------------------------------------------------------------


def test_mix64_matches_splitmix64() -> None:
    # The first output of SplitMix64 seeded with 0 is mix64(0 + GAMMA) = 0xE220A8397B1DCDAF,
    # the widely published reference value [mem]; mix64(0) = 0 follows from the formula.
    assert mix64((0 + GAMMA) & MASK64) == 0xE220A8397B1DCDAF
    assert mix64(0) == 0
    assert mix64(1) == 0x5692161D100B05E5


def test_pack_key_bit_layout() -> None:
    w0, w1 = pack_key(DrawKey(episode=3, tick=1200, purpose=Purpose.LOOT, entity=17, draw=2))
    assert w0 == 3 << 32 | 1200
    assert w1 == 7 << 56 | 17 << 32 | 2
    top = pack_key(DrawKey(2**32 - 1, 2**32 - 1, 255, 2**24 - 1, 2**32 - 1))
    assert top == (MASK64, MASK64)


GOLDEN_U64 = [
    (DrawKey(0, 0, Purpose.GENERIC, 0, 0), 0x4EC2D9484B3263E4),
    (DrawKey(0, 0, Purpose.GENERIC, 0, 1), 0x2041FDBED03B3676),
    (DrawKey(0, 1, Purpose.GENERIC, 0, 0), 0x53313B20EEAC7901),
    (DrawKey(1, 0, Purpose.GENERIC, 0, 0), 0x9DC2554920AEF224),
    (DrawKey(3, 1200, Purpose.LOOT, 17, 2), 0x0BC3F073506EB2AC),
    (DrawKey(2**32 - 1, 2**32 - 1, 255, 2**24 - 1, 2**32 - 1), 0xDAB705ACAA1AE1B8),
]


@pytest.mark.parametrize("key,expected", GOLDEN_U64)
def test_golden_u64(rng: CounterRNG, key: DrawKey, expected: int) -> None:
    assert rng.u64(key) == expected


def test_golden_seeds() -> None:
    key = DrawKey(0, 0, Purpose.GENERIC)
    assert CounterRNG(0).u64(key) == 0x238275BC38FCBE91
    assert CounterRNG(1).u64(key) == 0xB18A02F46D8D86C3
    assert CounterRNG(2**64 - 1).u64(key) == 0xFC042709560421DA


def test_golden_derived_draws(rng: CounterRNG) -> None:
    assert rng.uniform(DrawKey(3, 1200, Purpose.LOOT, 17, 2)) == 0.04595854583061043
    s = rng.stream(7, 6000, Purpose.SPAWN, 2)
    assert [s.randint(1, 6) for _ in range(10)] == [2, 5, 2, 5, 1, 2, 3, 1, 2, 2]
    assert [s.poisson(2.5) for _ in range(10)] == [2, 3, 5, 1, 4, 7, 1, 1, 3, 2]
    assert s.draw == 20
    assert [s.poisson(40.0) for _ in range(5)] == [50, 39, 36, 45, 36]  # 3 draws each
    assert s.draw == 35
    assert "".join(s.choice("abcde") for _ in range(10)) == "daabeeddba"
    assert [s.bernoulli(0.3) for _ in range(10)] == [True, False, False, True, False,
                                                      False, False, True, False, False]


# -- determinism and key independence ---------------------------------------------------


def test_deterministic_across_instances(rng: CounterRNG) -> None:
    other = CounterRNG(SEED)
    for key, _ in GOLDEN_U64:
        assert rng.u64(key) == other.u64(key) == rng.u64(key)
    # Order of evaluation does not matter: a stream equals explicit keys.
    s = rng.stream(5, 99, Purpose.MOB_AI, 3, start=10)
    assert [s.u64() for _ in range(4)] == [rng.u64(DrawKey(5, 99, Purpose.MOB_AI, 3, d)) for d in range(10, 14)]


def test_plain_tuples_work_as_keys(rng: CounterRNG) -> None:
    assert rng.u64((3, 1200, 7, 17, 2)) == 0x0BC3F073506EB2AC
    assert rng.uniform((0, 0, 0)) == rng.uniform(DrawKey(0, 0, Purpose.GENERIC, 0, 0))
    assert rng.poisson((7, 6000, Purpose.SPAWN, 2, 20), 40.0) == 50  # as in the golden stream


def test_every_key_field_changes_the_output(rng: CounterRNG) -> None:
    base = DrawKey(2, 300, Purpose.SPAWN, 4, 5)
    v = rng.u64(base)
    for field in DrawKey._fields:
        changed = base._replace(**{field: getattr(base, field) + 1})
        assert rng.u64(changed) != v, field
    assert CounterRNG(SEED + 1).u64(base) != v


def test_no_collisions_on_a_key_grid(rng: CounterRNG) -> None:
    values = set()
    for ep in range(3):
        for tick in range(4):
            for purpose in (Purpose.SPAWN, Purpose.LOOT, Purpose.HIT):
                u = rng.u64_array(ep, tick, purpose, np.arange(64)[:, None], np.arange(64)[None, :])
                values.update(int(x) for x in u.ravel())
    assert len(values) == 3 * 4 * 3 * 64 * 64


def test_u64_array_equals_scalar(rng: CounterRNG) -> None:
    ent = np.array([0, 1, 7, 2**24 - 1])
    draws = np.array([0, 3, 2**32 - 1])
    arr = rng.u64_array(9, 77, Purpose.LAG, ent[:, None], draws[None, :])
    assert arr.dtype == np.uint64 and arr.shape == (4, 3)
    for i, e in enumerate(ent):
        for j, d in enumerate(draws):
            assert int(arr[i, j]) == rng.u64(DrawKey(9, 77, Purpose.LAG, int(e), int(d)))
    uni = rng.uniform_array(9, 77, Purpose.LAG, 7, draws)
    assert [float(x) for x in uni] == [rng.uniform(DrawKey(9, 77, Purpose.LAG, 7, int(d))) for d in draws]


@pytest.mark.parametrize("key", [
    DrawKey(-1, 0, 0), DrawKey(2**32, 0, 0), DrawKey(0, 2**32, 0), DrawKey(0, 0, 256),
    DrawKey(0, 0, 0, 2**24), DrawKey(0, 0, 0, -1), DrawKey(0, 0, 0, 0, 2**32),
])
def test_out_of_range_fields_raise(rng: CounterRNG, key: DrawKey) -> None:
    with pytest.raises(ValueError):
        rng.u64(key)


@pytest.mark.parametrize("key", [
    DrawKey(0, 600.5, 0), DrawKey(0, 600.0, 0), (0, 600.9, 3), DrawKey(0.0, 0, 0), DrawKey(0, 0, 1.0),
    DrawKey(0, 0, 0, 2.0), DrawKey(0, 0, 0, 0, 1.5), DrawKey(0, True, 0), DrawKey(0, 0, 0, np.float64(3)),
    DrawKey(0, 0, 0, np.True_), DrawKey(0, "1", 0),
])
def test_non_integer_fields_raise_instead_of_truncating(rng: CounterRNG, key: tuple) -> None:
    with pytest.raises(TypeError):
        rng.u64(key)


def test_integer_like_fields_are_accepted(rng: CounterRNG) -> None:
    expected = rng.u64(DrawKey(3, 1200, 7, 17, 2))
    assert rng.u64(DrawKey(np.int32(3), np.uint64(1200), Purpose.LOOT, np.int64(17), np.uint8(2))) == expected


def test_u64_array_rejects_non_integer_arrays(rng: CounterRNG) -> None:
    with pytest.raises(TypeError):
        rng.u64_array(0, 600, 3, np.array([1.7, 2.2]), 0)
    with pytest.raises(TypeError):
        rng.u64_array(0, 600, 3, 1, np.array([0.0, 1.0]))
    with pytest.raises(TypeError):
        rng.u64_array(0, 600.5, 3, 1, 0)
    with pytest.raises(TypeError):
        rng.u64_array(0, 600, 3.0, 1, 0)
    with pytest.raises(TypeError):
        rng.u64_array(0, 600, 3, np.array([True, False]), 0)
    ok = rng.u64_array(0, 600, 3, np.array([1, 2], dtype=np.uint32), np.uint8(4))
    assert [int(x) for x in ok] == [rng.u64(DrawKey(0, 600, 3, e, 4)) for e in (1, 2)]
    assert rng.u64_array(0, 600, 3, np.array([], dtype=np.int64), 0).shape == (0,)


def test_bad_seed_and_arguments_raise(rng: CounterRNG) -> None:
    with pytest.raises(TypeError):
        CounterRNG(1.9)
    with pytest.raises(TypeError):
        CounterRNG(True)
    key = DrawKey(0, 0, 0)
    with pytest.raises(ValueError):
        CounterRNG(-1)
    with pytest.raises(ValueError):
        CounterRNG(2**64)
    with pytest.raises(ValueError):
        rng.randint(key, 3, 2)
    with pytest.raises(ValueError):
        rng.choice(key, [])
    with pytest.raises(ValueError):
        rng.below(key, 0)
    with pytest.raises(ValueError):
        rng.poisson(key, -1.0)
    with pytest.raises(ValueError):
        rng.poisson(key, math.inf)


def test_purpose_values_are_frozen() -> None:
    # Replays depend on these numbers: append new purposes, never renumber.
    assert [(p.name, p.value) for p in Purpose] == [
        ("GENERIC", 0), ("WORLDGEN", 1), ("RESET", 2), ("SPAWN", 3), ("DESPAWN", 4), ("MOB_AI", 5),
        ("HIT", 6), ("LOOT", 7), ("LAG", 8), ("CHAT", 9), ("ORDER", 10)]


# -- statistical sanity (rough, fixed seed so never flaky) -------------------------------


def test_uniform_is_roughly_uniform(rng: CounterRNG) -> None:
    n = 200_000
    u = rng.uniform_array(0, 0, Purpose.GENERIC, np.arange(n) // 1000, np.arange(n) % 1000)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.005
    counts = np.bincount((u * 64).astype(int), minlength=64)
    chi2 = float(((counts - n / 64) ** 2 / (n / 64)).sum())
    assert chi2 < 120  # 63 degrees of freedom: the 99.99% quantile is about 112


def test_bits_are_balanced(rng: CounterRNG) -> None:
    u = rng.u64_array(1, 2, Purpose.WORLDGEN, 0, np.arange(50_000))
    bits = (u[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & np.uint64(1)
    frac = bits.mean(axis=0)
    assert np.all(np.abs(frac - 0.5) < 0.012)


@pytest.mark.parametrize("axis", ["draw", "entity", "tick", "episode"])
def test_neighbouring_keys_are_uncorrelated(rng: CounterRNG, axis: str) -> None:
    n = 20_000
    if axis == "draw":
        a = rng.uniform_array(0, 0, Purpose.SPAWN, np.arange(n), 0)
        b = rng.uniform_array(0, 0, Purpose.SPAWN, np.arange(n), 1)
    elif axis == "entity":
        a = rng.uniform_array(0, 0, Purpose.SPAWN, 0, np.arange(n))
        b = rng.uniform_array(0, 0, Purpose.SPAWN, 1, np.arange(n))
    elif axis == "tick":
        a = rng.uniform_array(0, 10, Purpose.SPAWN, 0, np.arange(n))
        b = rng.uniform_array(0, 11, Purpose.SPAWN, 0, np.arange(n))
    else:
        a = rng.uniform_array(4, 10, Purpose.SPAWN, 0, np.arange(n))
        b = rng.uniform_array(5, 10, Purpose.SPAWN, 0, np.arange(n))
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.03
    assert abs(np.corrcoef(a[:-1], a[1:])[0, 1]) < 0.03


def test_randint_choice_bernoulli(rng: CounterRNG) -> None:
    s = rng.stream(0, 0, Purpose.LOOT)
    rolls = [s.randint(1, 6) for _ in range(6000)]
    assert set(rolls) == {1, 2, 3, 4, 5, 6}
    assert all(abs(rolls.count(k) - 1000) < 120 for k in range(1, 7))
    assert s.draw == 6000
    picks = [s.choice(("a", "b", "c")) for _ in range(3000)]
    assert all(abs(picks.count(x) - 1000) < 110 for x in "abc")
    hits = sum(s.bernoulli(0.25) for _ in range(8000))
    assert abs(hits - 2000) < 150
    assert not any(s.bernoulli(0.0) for _ in range(100))
    assert all(s.bernoulli(1.0) for _ in range(100))
    assert s.randint(5, 5) == 5


def test_below_uses_the_high_word(rng: CounterRNG) -> None:
    key = DrawKey(1, 1, Purpose.GENERIC)
    assert rng.below(key, 1 << 64) == rng.u64(key)
    assert rng.below(key, 2) == rng.u64(key) >> 63
    assert rng.below(key, 1) == 0


# -- Poisson ------------------------------------------------------------------------------


def test_exp_neg_is_accurate() -> None:
    for x in (0.0, 1e-9, 0.1, 0.5, 1.0, 3.7, 8.0, 15.9, 16.0):
        assert exp_neg(x) == pytest.approx(math.exp(-x), rel=1e-13, abs=0.0)
    with pytest.raises(ValueError):
        exp_neg(16.5)


def test_poisson_draw_counts() -> None:
    assert [poisson_draws(x) for x in (0.0, 0.3, 16.0, 16.01, 32.0, 100.0)] == [1, 1, 1, 2, 2, 7]


@pytest.mark.parametrize("lam", [0.05, 0.7, 3.0, 16.0, 45.0])
def test_poisson_matches_the_distribution(rng: CounterRNG, lam: float) -> None:
    n = 6000
    s = rng.stream(1, 1, Purpose.SPAWN, entity=int(lam * 100))
    xs = np.array([s.poisson(lam) for _ in range(n)])
    assert s.draw == n * poisson_draws(lam)
    assert abs(xs.mean() - lam) < 5 * math.sqrt(lam / n)
    assert abs(xs.var() - lam) < 0.15 * lam + 0.01
    if lam <= 16:
        # Total-variation distance to the exact pmf.
        k = np.arange(int(lam + 12 * math.sqrt(lam) + 12))
        pmf = np.array([math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1)) for i in k])
        emp = np.bincount(xs, minlength=len(k))[: len(k)] / n
        assert 0.5 * np.abs(emp - pmf).sum() < 0.03


def test_poisson_zero_mean(rng: CounterRNG) -> None:
    s = rng.stream(0, 0, Purpose.SPAWN)
    assert s.poisson(0.0) == 0 and s.draw == 1

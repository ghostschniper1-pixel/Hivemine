"""Counter-based random numbers for the reference simulator (DESIGN.md 4.7, contracts section 5).

Every draw is a pure function of ``(seed, episode, tick, purpose, entity, draw)``. Nothing is
stateful, so the reference simulator and the C++ core draw identical numbers in any order, and a
replay needs only the seed and the actions.

    rng = CounterRNG(seed=12345)
    key = DrawKey(episode=0, tick=600, purpose=Purpose.LOOT, entity=7)
    rng.uniform(key)                         # float in [0, 1), 53 bits
    rng.randint(key.at(1), 1, 3)             # int in [1, 3]
    s = rng.stream(0, 600, Purpose.SPAWN)    # draw index advances by itself
    n = s.poisson(0.35); x = s.below(16)

Exact algorithm (the C++ core must reproduce it bit for bit; all arithmetic on unsigned 64-bit
integers, wrapping mod 2^64):

    GAMMA  = 0x9E3779B97F4A7C15                  SplitMix64's increment (odd)
    mix64(z):                                    SplitMix64 output function (Stafford's Mix13)
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    w0 = episode << 32 | tick                    episode: 32 bits, tick: 32 bits
    w1 = purpose << 56 | entity << 32 | draw     purpose: 8 bits, entity: 24 bits, draw: 32 bits

    seed_state = mix64(seed + GAMMA)                         first SplitMix64 output of `seed`
    tick_state = mix64(seed_state + (w0 + 1) * GAMMA)        output w0 of the stream seeded by seed_state
    u64        = mix64(tick_state + (w1 + 1) * GAMMA)        output w1 of the stream seeded by tick_state

That is two nested SplitMix64 streams: the seed's stream indexed by (episode, tick) gives a state,
and that state's stream indexed by (purpose, entity, draw) gives the number. Fields out of range
raise ``ValueError`` rather than wrap, so two keys can never alias.

Derived draws (each uses exactly the draws stated, so a stream's draw index stays predictable):

    uniform(k)      = (u64(k) >> 11) * 2^-53                              1 draw
    below(k, n)     = (u64(k) * n) >> 64     (the high word; n >= 1)      1 draw
                      No rejection step: the bias is below n / 2^64. MSVC: __umulh; GCC: __int128.
    randint(k, a, b)= a + below(k, b - a + 1)                             1 draw
    choice(k, seq)  = seq[below(k, len(seq))]                             1 draw
    bernoulli(k, p) = uniform(k) < p                                      1 draw
    poisson(k, lam) = see ``poisson``; uses poisson_draws(lam) draws k.draw, k.draw + 1, ...

Floating point in ``poisson`` uses only IEEE-754 double +, -, *, / in the order written, so it is
reproducible in C++ as long as the compiler does not contract a*b+c into FMA
(GCC/Clang: -ffp-contract=off; MSVC: the default /fp:precise is fine).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum
from typing import NamedTuple, TypeVar

import numpy as np

T = TypeVar("T")

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
MIX_M1 = 0xBF58476D1CE4E5B9
MIX_M2 = 0x94D049BB133111EB

EPISODE_BITS = 32
TICK_BITS = 32
PURPOSE_BITS = 8
ENTITY_BITS = 24
DRAW_BITS = 32

POISSON_CHUNK = 16.0  # largest mean drawn with one uniform; bigger means are split into equal chunks
POISSON_MAX_MEAN = 1.0e6  # guard against runaway loops
_EXP_TERMS = 20  # Taylor terms for exp(-y), y <= 0.5
_EXP_HALVINGS = 5  # exp(-x) = exp(-x / 32) ** 32


class Purpose(IntEnum):
    """What a draw is for. Part of the key, so purposes never share numbers.

    The values are part of the replay format: never renumber, only append.
    """

    GENERIC = 0  # tests and one-off draws
    WORLDGEN = 1  # statistical terrain, ore and tree placement
    RESET = 2  # start areas, world-pool picks, initial placement
    SPAWN = 3  # mob spawning: counts, positions, types
    DESPAWN = 4  # random despawning
    MOB_AI = 5  # wandering, target choice, tie-breaks in mob movement
    HIT = 6  # measured hit chance of mob attacks
    LOOT = 7  # loot-table sampling (blocks and mobs)
    LAG = 8  # lag and robustness model (DESIGN.md 8)
    CHAT = 9  # nonce-word meanings and other chat draws (DESIGN.md 9)
    ORDER = 10  # tie-breaking the order of ready agents


class DrawKey(NamedTuple):
    """The counter of one draw. ``entity`` is any stable id below 2^24 (0 for world-level draws).

    Every method that takes a key also accepts a plain tuple ``(episode, tick, purpose[, entity[, draw]])``.
    """

    episode: int
    tick: int
    purpose: int
    entity: int = 0
    draw: int = 0

    def at(self, draw: int) -> DrawKey:
        """The same key with another draw index."""
        return self._replace(draw=draw)


KeyLike = DrawKey | tuple[int, ...]


def mix64(z: int) -> int:
    """SplitMix64's output function on an unsigned 64-bit integer."""
    z = ((z ^ (z >> 30)) * MIX_M1) & MASK64
    z = ((z ^ (z >> 27)) * MIX_M2) & MASK64
    return z ^ (z >> 31)


def _check(name: str, value: int, bits: int) -> int:
    v = int(value)
    if not 0 <= v < (1 << bits):
        raise ValueError(f"{name}={value} is outside [0, 2^{bits})")
    return v


def pack_key(key: KeyLike) -> tuple[int, int]:
    """The two 64-bit words ``(w0, w1)`` of a key (see the module docstring)."""
    key = DrawKey(*key)
    w0 = _check("episode", key.episode, EPISODE_BITS) << TICK_BITS | _check("tick", key.tick, TICK_BITS)
    w1 = (_check("purpose", key.purpose, PURPOSE_BITS) << (ENTITY_BITS + DRAW_BITS)
          | _check("entity", key.entity, ENTITY_BITS) << DRAW_BITS
          | _check("draw", key.draw, DRAW_BITS))
    return w0, w1


def exp_neg(x: float) -> float:
    """e^-x for 0 <= x <= 16, using only + - * / in a fixed order (bit-reproducible in C++).

    y = x / 32 (exact); s = 1; for n = 20 .. 1: s = 1 - (y * s) / n; then square s five times.
    Relative error about 1e-14, which is irrelevant: what matters is that both simulators agree.
    """
    if not 0.0 <= x <= POISSON_CHUNK:
        raise ValueError(f"exp_neg needs 0 <= x <= {POISSON_CHUNK}, got {x}")
    y = x / float(1 << _EXP_HALVINGS)
    s = 1.0
    for n in range(_EXP_TERMS, 0, -1):
        s = 1.0 - (y * s) / float(n)
    for _ in range(_EXP_HALVINGS):
        s = s * s
    return s


def poisson_draws(lam: float) -> int:
    """How many draws ``poisson(lam)`` consumes: max(1, ceil(lam / 16))."""
    if not (lam >= 0.0 and math.isfinite(lam)) or lam > POISSON_MAX_MEAN:
        raise ValueError(f"poisson mean must be in [0, {POISSON_MAX_MEAN}], got {lam}")
    return max(1, math.ceil(lam / POISSON_CHUNK))


def _poisson_inversion(u: float, lam: float, p0: float) -> int:
    """Smallest k with u < F(k) for Poisson(lam), F summed term by term (p0 = e^-lam)."""
    k, p, cdf = 0, p0, p0
    while u >= cdf:
        k += 1
        p = (p * lam) / float(k)
        nxt = cdf + p
        if nxt == cdf:  # the sum stopped growing (u within rounding of 1): deterministic stop
            break
        cdf = nxt
    return k


class CounterRNG:
    """Counter-based generator: every number is a pure function of the seed and a ``DrawKey``."""

    def __init__(self, seed: int):
        self.seed = _check("seed", seed, 64)
        self._seed_state = mix64((self.seed + GAMMA) & MASK64)

    def __repr__(self) -> str:
        return f"CounterRNG(seed={self.seed})"

    # -- raw bits -------------------------------------------------------------

    def u64(self, key: KeyLike) -> int:
        """64 random bits for ``key``."""
        w0, w1 = pack_key(key)
        tick_state = mix64((self._seed_state + (w0 + 1) * GAMMA) & MASK64)
        return mix64((tick_state + (w1 + 1) * GAMMA) & MASK64)

    def u64_array(self, episode: int, tick: int, purpose: int, entity: np.ndarray | int,
                  draw: np.ndarray | int) -> np.ndarray:
        """Vectorized ``u64`` over arrays of entities and/or draw indices (broadcast); uint64 array.

        Equal element by element to ``u64(DrawKey(episode, tick, purpose, e, d))``.
        """
        ent = np.asarray(entity, dtype=np.int64)
        drw = np.asarray(draw, dtype=np.int64)
        if ent.size and (ent.min() < 0 or ent.max() >= 1 << ENTITY_BITS):
            raise ValueError("entity outside [0, 2^24)")
        if drw.size and (drw.min() < 0 or drw.max() >= 1 << DRAW_BITS):
            raise ValueError("draw outside [0, 2^32)")
        w0, _ = pack_key(DrawKey(episode, tick, purpose))
        tick_state = mix64((self._seed_state + (w0 + 1) * GAMMA) & MASK64)
        w1 = (np.uint64(int(purpose) << (ENTITY_BITS + DRAW_BITS))
              | (ent.astype(np.uint64) << np.uint64(DRAW_BITS)) | drw.astype(np.uint64))
        with np.errstate(over="ignore"):
            z = np.uint64(tick_state) + (w1 + np.uint64(1)) * np.uint64(GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(MIX_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(MIX_M2)
            return z ^ (z >> np.uint64(31))

    # -- distributions ----------------------------------------------------------

    def uniform(self, key: KeyLike) -> float:
        """Uniform float in [0, 1) with 53 random bits."""
        return (self.u64(key) >> 11) * (1.0 / (1 << 53))

    def uniform_array(self, episode: int, tick: int, purpose: int, entity: np.ndarray | int,
                      draw: np.ndarray | int) -> np.ndarray:
        """Vectorized ``uniform``; float64 array."""
        bits = self.u64_array(episode, tick, purpose, entity, draw) >> np.uint64(11)
        return bits.astype(np.float64) * (1.0 / (1 << 53))

    def below(self, key: KeyLike, n: int) -> int:
        """Integer in [0, n), 1 <= n <= 2^64, by the high word of u64 * n (no rejection)."""
        if not 1 <= n <= 1 << 64:
            raise ValueError(f"below needs 1 <= n <= 2^64, got {n}")
        return (self.u64(key) * n) >> 64

    def randint(self, key: KeyLike, lo: int, hi: int) -> int:
        """Integer in [lo, hi], both ends included (like ``random.randint``)."""
        if hi < lo:
            raise ValueError(f"empty range [{lo}, {hi}]")
        return lo + self.below(key, hi - lo + 1)

    def choice(self, key: KeyLike, seq: Sequence[T]) -> T:
        """One element of a non-empty sequence."""
        if len(seq) == 0:
            raise ValueError("choice from an empty sequence")
        return seq[self.below(key, len(seq))]

    def bernoulli(self, key: KeyLike, p: float) -> bool:
        """True with probability p (p <= 0: never; p >= 1: always)."""
        return self.uniform(key) < p

    def poisson(self, key: KeyLike, lam: float) -> int:
        """Poisson(lam) by inversion, deterministic and reproducible in C++.

        The mean is split into c = max(1, ceil(lam / 16)) equal chunks lam_c = lam / c (lam = 0
        returns 0 and still counts one draw). Chunk i takes u = uniform at draw index key.draw + i
        and searches the CDF (the sum of the chunks is Poisson(lam)):

            k = 0; p = F = exp_neg(lam_c)
            while u >= F:
                k += 1; p = (p * lam_c) / k
                if F + p == F: break          # F stopped growing: deterministic stop
                F = F + p
        """
        key = DrawKey(*key)
        chunks = poisson_draws(lam)
        if lam == 0.0:
            return 0
        lam_c = lam / float(chunks)
        p0 = exp_neg(lam_c)
        return sum(_poisson_inversion(self.uniform(key.at(key.draw + i)), lam_c, p0) for i in range(chunks))

    def stream(self, episode: int, tick: int, purpose: int, entity: int = 0, start: int = 0) -> DrawStream:
        """A cursor over consecutive draw indices of one (episode, tick, purpose, entity)."""
        return DrawStream(self, DrawKey(episode, tick, purpose, entity, start))


class DrawStream:
    """Consecutive draws of one key prefix; each call advances the draw index by the draws it used."""

    def __init__(self, rng: CounterRNG, key: DrawKey):
        self.rng = rng
        self.key = key

    def __repr__(self) -> str:
        return f"DrawStream({self.rng!r}, {self.key!r})"

    @property
    def draw(self) -> int:
        """The draw index the next call will use."""
        return self.key.draw

    def _take(self, n: int = 1) -> DrawKey:
        key = self.key
        self.key = key.at(key.draw + n)
        return key

    def u64(self) -> int:
        return self.rng.u64(self._take())

    def uniform(self) -> float:
        return self.rng.uniform(self._take())

    def below(self, n: int) -> int:
        return self.rng.below(self._take(), n)

    def randint(self, lo: int, hi: int) -> int:
        return self.rng.randint(self._take(), lo, hi)

    def choice(self, seq: Sequence[T]) -> T:
        return self.rng.choice(self._take(), seq)

    def bernoulli(self, p: float) -> bool:
        return self.rng.bernoulli(self._take(), p)

    def poisson(self, lam: float) -> int:
        return self.rng.poisson(self._take(poisson_draws(lam)), lam)

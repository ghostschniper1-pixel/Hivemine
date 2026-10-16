"""Exact game formulas of the reference simulator (DESIGN.md 4.3; contracts section 5).

Pure functions over primitive inputs (hardness, tool speed, flags, ...), small state classes, and
thin helpers that take vanilla block and item names plus the ``Rules`` object. Nothing here
depends on the class tables (``hivemind.rules.classes``); the simulator passes class properties,
which equal the properties of every member (contracts section 3).

**Float32.** Java Edition 1.21.1 computes breaking progress, hunger, attack charge, armor and fall
damage in 32-bit floats. Each function rounds every operation to float32 (``f32``) in the order
the game evaluates it, so the C++ core can use plain ``float`` (no FMA contraction) and agree bit
for bit. A double result rounded to float32 equals the float32 operation for + - * / (double has
more than 2 * 24 + 2 bits), so this emulation is exact.

**Sources.** Constants come from the rules file (``data/rules-1.21.1.json``, built from
``hivemind/rules/constants.py``) through ``MechanicsConstants``; each keeps the label recorded
there ([13]-[20] are minecraft.wiki pages quoted by DESIGN.md). The evaluation order of the game
code and the few values not yet in the rules file are from memory of the 1.21.1 code ([mem])
unless a comment cites a data file. Data files from the pinned mcmeta ``1.21.1-data`` [3] confirm
some of it: ``enchantment/efficiency.json`` (mining_efficiency += level^2 + 1),
``enchantment/aqua_affinity.json`` (submerged_mining_speed x5 from its 0.2 base),
``damage_type/*.json`` (exhaustion 0.1 for mob and player attacks, 0 for starve and fall;
mob attacks scale with difficulty) and ``tags/damage_type/bypasses_armor.json`` (starve, fall).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import MISSING, dataclass, field, fields
from enum import IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from hivemind.rules import load_rules

if TYPE_CHECKING:
    from hivemind.rules import Rules


def f32(x: float) -> float:
    """Round a Python float to the nearest float32 value (Java's ``(float)`` cast)."""
    return float(np.float32(x))


# =============================================================================
# Constants
# =============================================================================


@dataclass(frozen=True)
class MechanicsConstants:
    """The constants the formulas use, named as in ``rules.constants``.

    Fields without a default must exist in the rules file. Fields with a default are not in the
    rules file yet (all [mem]); ``from_rules`` takes the rules value as soon as one exists.
    """

    # Breaking [13]
    break_damage_divisor_harvestable: int
    break_damage_divisor_not_harvestable: int
    break_delay_ticks: int
    break_haste_per_level: float
    break_mining_fatigue_base: float
    break_mining_fatigue_max_level: int
    break_underwater_multiplier: float
    break_airborne_multiplier: float
    break_server_accept_fraction: float
    # Hunger [14]
    max_food: int
    max_health: float
    spawn_food: int
    spawn_saturation: float
    exhaustion_sprint_per_m: float
    exhaustion_swim_per_m: float
    exhaustion_jump: float
    exhaustion_sprint_jump: float
    exhaustion_break_block: float
    exhaustion_attack: float
    exhaustion_damage: float
    exhaustion_threshold: float
    regen_saturated_interval_ticks: int
    regen_saturated_max_saturation: float
    regen_interval_ticks: int
    regen_min_food: int
    regen_exhaustion: float
    starvation_interval_ticks: int
    starvation_floor_easy: float
    starvation_floor_normal: float
    starvation_floor_hard: float
    sprint_requires_food_above: int
    eat_ticks: int
    ticks_per_second: int
    # Clock [15]
    day_length_ticks: int
    night_start_tick: int
    night_end_tick: int
    hostile_surface_spawn_start_tick: int
    hostile_surface_spawn_end_tick: int
    bed_use_start_tick: int
    bed_use_end_tick: int
    hostile_spawn_max_sky_light: int
    # Combat [18]
    player_base_attack_damage: float
    player_base_attack_speed: float
    attack_cooldown_numerator_ticks: float
    attack_charge_offset_ticks: float
    attack_multiplier_min: float
    attack_multiplier_scale: float
    critical_multiplier: float
    critical_min_charge: float
    invulnerability_ticks: int
    fall_damage_safe_blocks: float
    # Armor [17]
    armor_cap: float
    armor_floor_divisor: float
    armor_damage_coefficient: float
    armor_toughness_offset: float
    armor_divisor: float
    # Smelting [20]
    smelt_ticks: int
    # --- Not in the rules file yet (requested); from memory of the 1.21.1 code -------------
    exhaustion_cap: float = 40.0  # [mem] FoodData.addExhaustion: min(exhaustion + x, 40)
    # [mem] Player.getDestroySpeed: Mining Fatigue I-IV multiply by 0.3, 0.09, 0.0027, 0.00081
    # (a switch on the amplifier, not 0.3^n). prismarine-block 1.23.0 (npm [9]) has the same table.
    break_mining_fatigue_table: tuple[float, ...] = (0.3, 0.09, 0.0027, 0.00081)
    invulnerability_reset_ticks: int = 20  # [mem] a hit sets invulnerableTime = 20; blocks while > 10
    max_sky_light: int = 15  # [mem]
    sky_darken_night_min: int = 4  # [mem] Level.isDay() is skyDarken < 4; beds work only when not isDay

    @classmethod
    def from_rules(cls, rules: Rules) -> MechanicsConstants:
        values = {}
        for f in fields(cls):
            if f.name in rules.constants:
                v = rules.constant(f.name)
                values[f.name] = tuple(v) if isinstance(v, list) else v
            elif f.default is MISSING:
                raise KeyError(f"rules file has no constant {f.name!r}")
        return cls(**values)


_CONSTANTS: dict[str, MechanicsConstants] = {}


def constants_for(rules: Rules) -> MechanicsConstants:
    """``MechanicsConstants`` of a rules object (cached by the rules hash)."""
    c = _CONSTANTS.get(rules.hash)
    if c is None:
        c = _CONSTANTS[rules.hash] = MechanicsConstants.from_rules(rules)
    return c


@lru_cache(maxsize=1)
def default_constants() -> MechanicsConstants:
    """Constants of the default rules file (``load_rules()``)."""
    return constants_for(load_rules())


def _c(c: MechanicsConstants | None) -> MechanicsConstants:
    return c if c is not None else default_constants()


# =============================================================================
# Breaking [13]
# =============================================================================


def can_harvest(rules: Rules, block: str, held: str | None) -> bool:
    """Whether breaking ``block`` while holding ``held`` (None = empty hand) drops anything [1][3]."""
    return rules.can_harvest(held, block)


def mining_fatigue_multiplier(level: int, *, design_formula: bool = False,
                              c: MechanicsConstants | None = None) -> float:
    """Speed multiplier of Mining Fatigue ``level`` (1 = Mining Fatigue I; 0 = none), as float32.

    Vanilla uses the table 0.3, 0.09, 0.0027, 0.00081 [mem]; DESIGN.md 4.3 writes 0.3^min(level, 4)
    [13], which agrees for levels 1-2 and is 10x larger at levels 3-4. ``design_formula`` selects
    the design's version for comparison. Irrelevant in v1 (only elder guardians give the effect).
    """
    c = _c(c)
    if level <= 0:
        return 1.0
    if design_formula:
        return f32(c.break_mining_fatigue_base ** min(level, c.break_mining_fatigue_max_level))
    table = c.break_mining_fatigue_table
    return f32(table[min(level, len(table)) - 1])


def destroy_speed(tool_speed: float, *, efficiency: int = 0, haste: int = 0, mining_fatigue: int = 0,
                  in_water: bool = False, aqua_affinity: bool = False, on_ground: bool = True,
                  c: MechanicsConstants | None = None) -> float:
    """``Player.getDestroySpeed`` in float32 [13] (order of operations [mem]).

    tool_speed: the held item's tool-rule speed on the block, else 1 (``Rules.mining_speed``).
    efficiency, haste, mining_fatigue: levels (1 = level I; 0 = none).
    in_water: the player's eyes are in water; aqua_affinity: worn on the helmet.
    """
    c = _c(c)
    f = f32(tool_speed)
    if f > 1.0 and efficiency > 0:  # [3] mining_efficiency attribute += level^2 + 1
        f = f32(f + f32(efficiency * efficiency + 1))
    if haste > 0:  # [13] * (1 + 0.2 * level), float32 as 1.0F + (float)level * 0.2F [mem]
        f = f32(f * f32(1.0 + f32(f32(haste) * f32(c.break_haste_per_level))))
    if mining_fatigue > 0:
        f = f32(f * mining_fatigue_multiplier(mining_fatigue, c=c))
    if in_water and not aqua_affinity:  # [13] *0.2; 1.21 multiplies by the submerged_mining_speed attribute [mem]
        f = f32(f * f32(c.break_underwater_multiplier))
    if not on_ground:  # [13] /5: the game divides by 5.0F [mem], so divide by 1/0.2
        f = f32(f / f32(1.0 / c.break_airborne_multiplier))
    return f


def progress_per_tick(hardness: float, speed: float, can_harvest: bool,
                      c: MechanicsConstants | None = None) -> float:
    """``BlockBehaviour.getDestroyProgress`` in float32: speed / hardness / (30 or 100) [13].

    Hardness -1 (unbreakable) gives 0; hardness 0 gives +inf (Java's float division by zero).
    """
    c = _c(c)
    if hardness < 0:
        return 0.0
    if hardness == 0:
        return math.inf
    divisor = c.break_damage_divisor_harvestable if can_harvest else c.break_damage_divisor_not_harvestable
    return f32(f32(f32(speed) / f32(hardness)) / f32(divisor))


_ACCUMULATE_CHUNK = 1 << 16


def client_break_ticks(progress: float) -> int | None:
    """Ticks until the vanilla client's float32 progress sum reaches 1.0; 0 = instant, None = never.

    ``MultiPlayerGameMode.destroyProgress`` is a float field that gains ``progress`` once per tick
    and breaks the block when it is >= 1.0F [mem]. The sum is float32, so it can fall short of the
    exact product: stone by hand (1/150 per tick) sums to 0.99999923 after 150 ticks and breaks on
    the 151st. A progress below about 2^-25 per tick stops adding up before 1.0 and never breaks.
    """
    if progress <= 0.0:
        return None
    if progress >= 1.0:
        return 0
    p = np.float32(progress)
    total, done = np.float32(0.0), 0
    n = min(_ACCUMULATE_CHUNK, int(1.0 / progress) + 4)
    while True:
        # np.cumsum (add.accumulate) adds strictly left to right, so this is the sequential float32 sum.
        chunk = np.full(n + 1, p, dtype=np.float32)
        chunk[0] = total
        sums = np.cumsum(chunk, dtype=np.float32)[1:]
        hit = int(np.argmax(sums >= np.float32(1.0)))
        if sums[hit] >= np.float32(1.0):
            return done + hit + 1
        if sums[-1] == total:  # the sum stopped growing
            return None
        total, done, n = sums[-1], done + n, _ACCUMULATE_CHUNK


def server_accept_ticks(progress: float, c: MechanicsConstants | None = None) -> int | None:
    """Earliest server ticks k after "start digging" at which "finished" is accepted.

    ``ServerPlayerGameMode`` accepts when progress * (float)(k + 1) >= 0.7F in float32 [mem]; an
    earlier "finished" is remembered and the server breaks the block itself once the same product
    reaches 1.0 [mem]. 0 = instant (accepted with the start packet); None = never.
    """
    c = _c(c)
    if progress <= 0.0:
        return None
    if progress >= 1.0:
        return 0
    threshold = f32(c.break_server_accept_fraction)
    k = max(0, math.floor(threshold / progress * (1.0 - 1e-6)) - 2)
    while f32(progress * f32(k + 1)) < threshold:
        k += 1
    return k


def design_break_ticks(hardness: float, tool_speed: float, can_harvest: bool, *, efficiency: int = 0,
                       haste: int = 0, mining_fatigue: int = 0, in_water: bool = False,
                       aqua_affinity: bool = False, on_ground: bool = True,
                       c: MechanicsConstants | None = None) -> int | None:
    """DESIGN.md 4.3's ``ticks = ceil(1/dmg)``, in double precision. 0 = instant; None = never.

    The arithmetic is that of prismarine-block's ``Block.digTime`` (JavaScript doubles, same order;
    read in prismarine-block 1.23.0 from npm [9]). What it means depends on ``tool_speed``:

    - the game's tool-rule speed (``Rules.mining_speed``): the design formula and its check
      values (``BreakTime.design_ticks``);
    - the speed prismarine-block looks up (``mineflayer_tool_speed``): what Mineflayer waits
      between "start" and "finished" (``BreakTime.mineflayer_ticks``). The two speeds differ on
      tier-gated blocks (see ``mineflayer_tool_speed``).

    Double rounding can land just above an integer: coal ore with a wooden pickaxe is
    2 / 3 / 30 = 1/45 exactly, but 1 / dmg evaluates to 45.000000000000007, so this gives 46.
    """
    c = _c(c)
    if hardness < 0:
        return None
    speed = float(tool_speed)
    if efficiency > 0 and speed > 1.0:
        speed += efficiency * efficiency + 1
    if haste > 0:
        speed *= 1 + c.break_haste_per_level * haste
    if mining_fatigue > 0:
        table = c.break_mining_fatigue_table
        speed *= table[min(mining_fatigue, len(table)) - 1]
    if in_water and not aqua_affinity:
        speed /= 1.0 / c.break_underwater_multiplier
    if not on_ground:
        speed /= 1.0 / c.break_airborne_multiplier
    if hardness == 0:
        return 0
    divisor = c.break_damage_divisor_harvestable if can_harvest else c.break_damage_divisor_not_harvestable
    delta = speed / hardness / divisor
    if delta >= 1.0:
        return 0
    return math.ceil(1.0 / delta)


# Mineflayer's ``bot.dig`` sends "start", waits ``bot.digTime(block)`` ms and sends "finished"
# (mineflayer 4.39.0, lib/plugins/digging.js [74], read from npm [9]). ``digTime`` is
# prismarine-block 1.23.0's ``Block.digTime``, which takes the tool speed from minecraft-data's
# materials table, ``registry.materials[block.material][heldItemId]`` (default 1), not from the
# game's tool rules. Both lookups below are data of the pinned minecraft-data [1] (pc/1.21.1
# blocks.json and materials.json; the same bytes as npm minecraft-data 3.117.0, which
# bridge/package-lock.json locks). tests/test_mechanics.py checks them against the source cache
# for every block and tool, and against digTime outputs of the real packages.
#
# Every tier-gated block (exactly the members of the block tag ``incorrect_for_wooden_tool``:
# iron, gold, diamond, redstone, lapis, copper and emerald ores, obsidian, iron block, ...) has
# material "incorrect_for_wooden_tool", and that material lists only these four tools:
MINEFLAYER_TIER_GATED_TAG = "incorrect_for_wooden_tool"
MINEFLAYER_TIER_GATED_SPEEDS: dict[str, float] = {  # [1] materials.json "incorrect_for_wooden_tool"
    "wooden_shovel": 2.0, "wooden_pickaxe": 2.0, "wooden_axe": 2.0, "wooden_hoe": 2.0}
# Blocks whose minecraft-data material lacks a tool kind that the 1.21.1 mineable tags give them
# (the same pairs as build_rules.TOLERATED_TOOL_SPEED): Mineflayer digs them at speed 1 with it.
MINEFLAYER_MISSING_TOOL_KINDS: dict[str, frozenset[str]] = {  # [1] material "plant" has no hoes
    "moss_carpet": frozenset({"hoe"}),
    "pink_petals": frozenset({"hoe"}),
}
_MINEFLAYER_DEFAULT_SPEED = 1.0  # prismarine-block digTime: blockBreakingSpeed = 1 without a match [9]


def mineflayer_tool_speed(rules: Rules, held: str | None, block: str) -> float:
    """The tool speed Mineflayer's dig wait uses for ``block`` with ``held`` (None = empty hand).

    Equal to ``rules.mining_speed`` (the game's tool rules) except where minecraft-data's
    material disagrees with the tags (see the tables above; the rules build checks every other
    block and tool). On tier-gated blocks every stone, iron, gold, diamond and netherite tool
    gets 1 (the game: the tool's tier speed), and a wooden axe, shovel or hoe gets 2 (the game: 1).
    So unpatched Mineflayer waits too long on the whole ore tech tree, by the tool's speed factor
    (iron ore with a stone pickaxe: 91 ticks, not 23; obsidian with a diamond pickaxe: 1500, not
    188), and with a wooden axe, shovel or hoe it sends "finished" before the server accepts it
    (``BreakTime.mineflayer_refused``).
    """
    if held is None:
        return _MINEFLAYER_DEFAULT_SPEED
    if rules.in_tag("block", block, MINEFLAYER_TIER_GATED_TAG):
        return MINEFLAYER_TIER_GATED_SPEEDS.get(held, _MINEFLAYER_DEFAULT_SPEED)
    tool = rules.item(held).tool
    if tool is not None and tool.kind in MINEFLAYER_MISSING_TOOL_KINDS.get(block, ()):
        return _MINEFLAYER_DEFAULT_SPEED
    return rules.mining_speed(held, block)


@dataclass(frozen=True)
class BreakTime:
    """How long a block takes to break. Every count is 0 for instant and None for never.

    ``design_ticks`` is also what Mineflayer would wait if the bridge made ``bot.digTime`` use
    the game's tool rules; ``mineflayer_ticks`` is what Mineflayer 4.39.0 waits as shipped.
    """

    progress_per_tick: float  # float32 progress per tick (0 unbreakable, inf for hardness 0)
    ticks: int | None  # vanilla client: ticks of float32 accumulation until >= 1.0
    design_ticks: int | None  # DESIGN.md 4.3: ceil(1/dmg) in doubles with the game's tool speeds
    mineflayer_ticks: int | None  # unpatched Mineflayer's wait: ceil(1/dmg) with minecraft-data's speeds
    server_accept_ticks: int | None  # earliest server ticks after "start" that accept "finished" (70%)

    @property
    def breakable(self) -> bool:
        return self.ticks is not None or self.design_ticks is not None

    @property
    def instant(self) -> bool:
        return self.ticks == 0

    @property
    def mineflayer_refused(self) -> bool:
        """Unpatched Mineflayer sends "finished" before the server's 70% check passes.

        Compares tick counts and ignores the phase of the server tick and network jitter, so it
        is exact only away from the boundary. The server then refuses the "finished" packet and
        breaks the block by itself once progress reaches 100% [mem], long after Mineflayer's
        promise resolved; DESIGN.md 5's dig check sees the block still there (``DIG_FAILED``).
        """
        if self.mineflayer_ticks is None or self.server_accept_ticks is None:
            return False
        return self.mineflayer_ticks < self.server_accept_ticks


def break_ticks(hardness: float, tool_speed: float, can_harvest: bool, *, mineflayer_speed: float | None = None,
                efficiency: int = 0, haste: int = 0, mining_fatigue: int = 0, in_water: bool = False,
                aqua_affinity: bool = False, on_ground: bool = True,
                c: MechanicsConstants | None = None) -> BreakTime:
    """Break time of a block from primitive inputs [13] (see ``BreakTime`` for the four counts).

    tool_speed: the game's tool-rule speed (``Rules.mining_speed``); mineflayer_speed: the speed
    prismarine-block uses (``mineflayer_tool_speed``), None = the same as ``tool_speed``.

    ``ticks`` emulates the vanilla client exactly. It differs from ``design_ticks`` (the DESIGN.md
    check values) when 1/dmg is an integer or nearly one, where float32 rounding can add a tick
    (stone or logs by hand: 151 and 61 instead of 150 and 60) or double rounding can (coal ore
    with a wooden pickaxe: 45 vs 46), and on very slow breaks, where the float32 sum drifts. The
    vanilla client then waits ``break_delay_ticks()`` before progress starts on the next block.
    """
    c = _c(c)
    effects = {"efficiency": efficiency, "haste": haste, "mining_fatigue": mining_fatigue, "in_water": in_water,
               "aqua_affinity": aqua_affinity, "on_ground": on_ground}
    speed = destroy_speed(tool_speed, c=c, **effects)
    progress = progress_per_tick(hardness, speed, can_harvest, c)
    design = design_break_ticks(hardness, tool_speed, can_harvest, c=c, **effects)
    if mineflayer_speed is None or mineflayer_speed == tool_speed:
        mineflayer = design
    else:
        mineflayer = design_break_ticks(hardness, mineflayer_speed, can_harvest, c=c, **effects)
    return BreakTime(
        progress_per_tick=progress,
        ticks=client_break_ticks(progress),
        design_ticks=design,
        mineflayer_ticks=mineflayer,
        server_accept_ticks=server_accept_ticks(progress, c),
    )


def break_ticks_for(rules: Rules, block: str, held: str | None = None, *, efficiency: int = 0, haste: int = 0,
                    mining_fatigue: int = 0, in_water: bool = False, aqua_affinity: bool = False,
                    on_ground: bool = True) -> BreakTime:
    """Break time of a vanilla block with a vanilla held item (None = empty hand)."""
    return break_ticks(rules.block(block).hardness, rules.mining_speed(held, block), rules.can_harvest(held, block),
                       mineflayer_speed=mineflayer_tool_speed(rules, held, block), efficiency=efficiency,
                       haste=haste, mining_fatigue=mining_fatigue, in_water=in_water,
                       aqua_affinity=aqua_affinity, on_ground=on_ground, c=constants_for(rules))


def break_delay_ticks(c: MechanicsConstants | None = None) -> int:
    """Ticks after a break before the next block starts (rules ``break_delay_ticks``, [13]: 6).

    The vanilla client sets ``destroyDelay = 5`` after a break [mem], then spends one more tick
    starting the next block without progress, so 6 progress-free ticks separate two blocks [mem].
    Mineflayer has no such delay; the bridge's own pacing applies (to be measured in M2).
    """
    return int(_c(c).break_delay_ticks)


# =============================================================================
# Hunger [14]
# =============================================================================


class Difficulty(IntEnum):
    """Game difficulty, with the game's ids [mem]."""

    PEACEFUL = 0
    EASY = 1
    NORMAL = 2
    HARD = 3


_EXHAUSTION_KINDS = {
    "sprint": "exhaustion_sprint_per_m",
    "swim": "exhaustion_swim_per_m",
    "jump": "exhaustion_jump",
    "sprint_jump": "exhaustion_sprint_jump",
    "break_block": "exhaustion_break_block",
    "attack": "exhaustion_attack",
    "damage": "exhaustion_damage",
}


def exhaustion_cost(kind: str, amount: float = 1.0, c: MechanicsConstants | None = None) -> float:
    """Exhaustion of an activity in float32 [14]: ``sprint`` and ``swim`` per metre (``amount``
    metres), ``jump``, ``sprint_jump``, ``break_block``, ``attack``, ``damage`` (per hit taken;
    [3]: mob and player attacks 0.1, starvation and falls 0) per event (``amount`` events)."""
    try:
        name = _EXHAUSTION_KINDS[kind]
    except KeyError:
        raise ValueError(f"unknown exhaustion kind {kind!r}; one of {sorted(_EXHAUSTION_KINDS)}") from None
    return f32(f32(getattr(_c(c), name)) * amount)


def _heal(health: float, amount: float, max_health: float) -> float:
    """``LivingEntity.heal``: health + amount clamped to [0, max] in float32; returns the change."""
    if health <= 0.0:
        return 0.0
    new = min(max(f32(health + amount), 0.0), f32(max_health))
    return new - health


def _spawn_food() -> int:
    return default_constants().spawn_food


def _spawn_saturation() -> float:
    return default_constants().spawn_saturation


@dataclass
class HungerState:
    """``FoodData``: food level, saturation, exhaustion and the regeneration/starvation timer [14].

    The defaults are the spawn state from the rules file (food 20, saturation 5 [14]).
    Saturation and exhaustion are float32 values, as in the game.
    """

    food: int = field(default_factory=_spawn_food)
    saturation: float = field(default_factory=_spawn_saturation)
    exhaustion: float = 0.0
    timer: int = 0

    def add_exhaustion(self, amount: float, c: MechanicsConstants | None = None) -> None:
        """``FoodData.addExhaustion``: exhaustion = min(exhaustion + amount, 40) in float32 [mem]."""
        c = _c(c)
        self.exhaustion = min(f32(self.exhaustion + f32(amount)), f32(c.exhaustion_cap))

    def tick(self, health: float, max_health: float | None = None, difficulty: Difficulty = Difficulty.NORMAL,
             natural_regeneration: bool = True, c: MechanicsConstants | None = None) -> float:
        """One game tick of ``FoodData.tick``; returns the health change (+ heal, -1 starvation).

        Order [14], with the comparisons of the code [mem]:
        1. exhaustion > 4 (strictly): subtract 4, then saturation - 1 (floored at 0), or, if
           saturation is 0, food - 1 (not on Peaceful).
        2. Hurt, food 20 and saturation > 0: every 10 ticks heal min(sat, 6)/6, add min(sat, 6) exhaustion.
        3. Else hurt and food >= 18: every 80 ticks heal 1, add 6 exhaustion.
        4. Else food 0: every 80 ticks take 1 starvation damage while health > 10 (Easy, Peaceful),
           > 1 (Normal) or always (Hard).
        5. Else the timer resets.
        "Hurt" means 0 < health < max_health. Starvation damage goes through ``hurt`` in the game
        (it bypasses armor [3], but invulnerability ticks can swallow it [mem]); the caller applies it.
        Not modelled: Peaceful's extra regeneration in ``Player.aiStep`` (the server runs Normal).
        """
        c = _c(c)
        max_health = c.max_health if max_health is None else max_health
        threshold = f32(c.exhaustion_threshold)
        if self.exhaustion > threshold:
            self.exhaustion = f32(self.exhaustion - threshold)
            if self.saturation > 0.0:
                self.saturation = max(f32(self.saturation - 1.0), 0.0)
            elif difficulty != Difficulty.PEACEFUL:
                self.food = max(self.food - 1, 0)

        hurt = 0.0 < health < max_health
        if natural_regeneration and self.saturation > 0.0 and hurt and self.food >= c.max_food:
            self.timer += 1
            if self.timer >= c.regen_saturated_interval_ticks:
                amount = min(self.saturation, f32(c.regen_saturated_max_saturation))
                delta = _heal(health, f32(amount / f32(c.regen_saturated_max_saturation)), max_health)
                self.add_exhaustion(amount, c)
                self.timer = 0
                return delta
        elif natural_regeneration and self.food >= c.regen_min_food and hurt:
            self.timer += 1
            if self.timer >= c.regen_interval_ticks:
                delta = _heal(health, 1.0, max_health)
                self.add_exhaustion(c.regen_exhaustion, c)
                self.timer = 0
                return delta
        elif self.food <= 0:
            self.timer += 1
            if self.timer >= c.starvation_interval_ticks:
                self.timer = 0
                if health > _starvation_floor(difficulty, c):
                    return -1.0
        else:
            self.timer = 0
        return 0.0

    def eat(self, nutrition: int, saturation: float, c: MechanicsConstants | None = None) -> None:
        """``FoodData.eat`` with 1.20.5+ food components [14][mem]: ``saturation`` is the absolute
        amount the component stores (nutrition * ratio * 2; bread 5 and 6.0). Food is capped at
        20, then saturation at the new food level."""
        c = _c(c)
        self.food = min(max(self.food + nutrition, 0), c.max_food)
        self.saturation = min(max(f32(f32(saturation) + self.saturation), 0.0), float(self.food))

    def eat_item(self, rules: Rules, item: str) -> None:
        """Eat a vanilla food item (nutrition and saturation from its food component [3])."""
        food = rules.item(item).food
        if food is None:
            raise ValueError(f"{item!r} is not food")
        self.eat(food.nutrition, food.saturation, constants_for(rules))

    def needs_food(self, c: MechanicsConstants | None = None) -> bool:
        """Food below the maximum (only then can ordinary food be eaten) [14]."""
        return self.food < _c(c).max_food

    def can_sprint(self, c: MechanicsConstants | None = None) -> bool:
        """Sprinting needs food > 6 [14]."""
        return self.food > _c(c).sprint_requires_food_above


def _starvation_floor(difficulty: Difficulty, c: MechanicsConstants) -> float:
    if difficulty == Difficulty.HARD:
        return c.starvation_floor_hard
    if difficulty == Difficulty.NORMAL:
        return c.starvation_floor_normal
    return c.starvation_floor_easy  # Easy; Peaceful follows the same "health > 10" branch [mem]


def eat_ticks(eat_seconds: float | None = None, c: MechanicsConstants | None = None) -> int:
    """Ticks to eat: (int)(eat_seconds * 20.0F) as ``FoodProperties.eatDurationTicks`` [mem];
    None gives the rules default (32 [mem], from the usual eat_seconds 1.6 [3])."""
    c = _c(c)
    if eat_seconds is None:
        return int(c.eat_ticks)
    return int(f32(f32(eat_seconds) * f32(c.ticks_per_second)))


def eat_ticks_for(rules: Rules, item: str) -> int:
    """Ticks to eat a vanilla food item (dried kelp 16, most food 32)."""
    food = rules.item(item).food
    if food is None:
        raise ValueError(f"{item!r} is not food")
    return eat_ticks(food.eat_seconds, constants_for(rules))


# =============================================================================
# Clock [15]
# =============================================================================


def time_of_day(day_time: int, c: MechanicsConstants | None = None) -> int:
    """Ticks into the current day, 0..23999 [15]."""
    return day_time % _c(c).day_length_ticks


def is_night(day_time: int, c: MechanicsConstants | None = None) -> bool:
    """Night as [15] defines it: 13000 <= time < 23000."""
    c = _c(c)
    return c.night_start_tick <= time_of_day(day_time, c) < c.night_end_tick


def surface_hostile_spawn_time(day_time: int, c: MechanicsConstants | None = None) -> bool:
    """Hostiles can spawn in full sky light, clear weather: 13188 <= time <= 22812 [15].

    Both ends are included: that is exactly where ``sky_darken`` >= 8, i.e. surface light
    15 - 8 = 7 <= the overworld's monster spawn light level, uniform 0..7 [3][16].
    """
    c = _c(c)
    return c.hostile_surface_spawn_start_tick <= time_of_day(day_time, c) <= c.hostile_surface_spawn_end_tick


def can_sleep_time(day_time: int, c: MechanicsConstants | None = None) -> bool:
    """Beds work 12542 <= time < 23460 [15], clear weather.

    The end is excluded: that is exactly where ``sky_darken`` >= 4 (``Level.isDay`` is false) [mem].
    """
    c = _c(c)
    return c.bed_use_start_tick <= time_of_day(day_time, c) < c.bed_use_end_tick


def ticks_until(day_time: int, target_time_of_day: int, c: MechanicsConstants | None = None) -> int:
    """Ticks from ``day_time`` until the clock next shows ``target_time_of_day`` (0 if it does now)."""
    c = _c(c)
    return (target_time_of_day - time_of_day(day_time, c)) % c.day_length_ticks


@lru_cache(maxsize=1)
def _sin_table() -> tuple[float, ...]:
    """``Mth.SIN[i] = (float)Math.sin(i * PI * 2.0 / 65536.0)`` [mem]."""
    x = np.arange(65536, dtype=np.float64) * math.pi * 2.0 / 65536.0
    return tuple(np.sin(x).astype(np.float32).tolist())


def _mth_cos(x: float) -> float:
    """``Mth.cos``: SIN[(int)(x * 10430.378F + 16384.0F) & 65535] [mem]."""
    return _sin_table()[int(f32(f32(x * f32(10430.378)) + f32(16384.0))) & 65535]


def celestial_angle(day_time: int, c: MechanicsConstants | None = None) -> float:
    """Sun angle as a fraction of a turn, ``DimensionType.timeOfDay`` in the overworld [mem]."""
    d = (day_time / float(_c(c).day_length_ticks) - 0.25) % 1.0
    e = 0.5 - math.cos(d * math.pi) / 2.0
    return f32(f32(d * 2.0 + e) / f32(3.0))


def sky_darken(day_time: int, rain: float = 0.0, thunder: float = 0.0,
               c: MechanicsConstants | None = None) -> int:
    """``Level.skyDarken`` (0 at noon, 11 at midnight in clear weather) [mem].

    The formula reproduces [15]'s windows exactly (tests): surface hostiles need skyDarken >= 8
    (13188..22812), beds need skyDarken >= 4 (12542..23459). Rain and thunder levels in [0, 1]
    darken the sky further (weather is skipped in v1).
    """
    angle = celestial_angle(day_time, c)
    cos = _mth_cos(f32(angle * f32(math.pi * 2.0)))
    d = 1.0 - f32(f32(rain) * f32(5.0)) / 16.0
    e = 1.0 - f32(f32(thunder) * f32(5.0)) / 16.0
    f = 0.5 + 2.0 * min(max(cos, -0.25), 0.25)
    return int((1.0 - f * d * e) * 11.0)


def surface_sky_light(day_time: int, rain: float = 0.0, thunder: float = 0.0,
                      c: MechanicsConstants | None = None) -> int:
    """Effective sky light under open sky, 15 - skyDarken [mem]."""
    return _c(c).max_sky_light - sky_darken(day_time, rain, thunder, c)


def sky_is_dark(day_time: int, rain: float = 0.0, thunder: float = 0.0,
                c: MechanicsConstants | None = None) -> bool:
    """Not ``Level.isDay()``: skyDarken >= 4 [mem]. Beds work only then (weather-aware ``can_sleep_time``)."""
    return sky_darken(day_time, rain, thunder, c) >= _c(c).sky_darken_night_min


def sky_allows_surface_spawns(day_time: int, rain: float = 0.0, c: MechanicsConstants | None = None) -> bool:
    """Surface sky light <= 7, the most the monster spawn light test can pass [3][16] (weather-aware
    ``surface_hostile_spawn_time``; thunder uses a different light rule [mem] and is not covered)."""
    return surface_sky_light(day_time, rain, 0.0, c) <= _c(c).hostile_spawn_max_sky_light


# =============================================================================
# Combat [18]
# =============================================================================


@dataclass(frozen=True)
class AttackStats:
    """A player's melee attributes with one held item."""

    damage: float  # float32 attack damage: base 1 + the item's modifier [18]
    attack_speed: float  # attacks per second (double attribute): base 4 + the item's modifier [mem]
    cooldown: float  # float32 ticks to full charge, T = 20 / attack_speed [18]


def attack_cooldown_ticks(attack_speed: float, c: MechanicsConstants | None = None) -> float:
    """``getCurrentItemAttackStrengthDelay``: (float)(1.0 / attack_speed * 20.0) [18][mem]."""
    return f32(1.0 / attack_speed * _c(c).attack_cooldown_numerator_ticks)


def attack_stats(damage_modifier: float = 0.0, speed_modifier: float = 0.0,
                 c: MechanicsConstants | None = None) -> AttackStats:
    """Attack attributes from the item's attribute modifiers (0, 0 for an empty hand)."""
    c = _c(c)
    speed = c.player_base_attack_speed + speed_modifier
    return AttackStats(damage=f32(c.player_base_attack_damage + damage_modifier), attack_speed=speed,
                       cooldown=attack_cooldown_ticks(speed, c))


def attack_stats_for(rules: Rules, held: str | None) -> AttackStats:
    """Attack attributes with a vanilla held item (None = fist: 1 damage, 4 attacks/s)."""
    item = rules.item(held) if held else None
    dmg = item.attack_damage if item and item.attack_damage is not None else 0.0
    spd = item.attack_speed if item and item.attack_speed is not None else 0.0
    return attack_stats(dmg, spd, constants_for(rules))


def attack_charge(ticks_since_attack: int, cooldown: float, c: MechanicsConstants | None = None) -> float:
    """``getAttackStrengthScale(0.5)``: clamp((t + 0.5) / T, 0, 1) in float32 [18].

    t counts ticks since the last swing or item switch.
    """
    c = _c(c)
    h = f32(f32(f32(ticks_since_attack) + f32(c.attack_charge_offset_ticks)) / f32(cooldown))
    return min(max(h, 0.0), 1.0)


def attack_damage_multiplier(charge: float, c: MechanicsConstants | None = None) -> float:
    """0.2 + charge^2 * 0.8 in float32 [18]; in [0.2, 1] because the charge is clamped."""
    c = _c(c)
    return f32(f32(c.attack_multiplier_min) + f32(f32(charge * charge) * f32(c.attack_multiplier_scale)))


def is_critical(charge: float, falling: bool, c: MechanicsConstants | None = None) -> bool:
    """A critical hit needs charge > 0.9 [mem] and ``falling``.

    ``falling`` stands for the game's full condition [mem]: fall distance > 0, not on the ground,
    not climbing, not in water, not blind, not riding, not sprinting. [18]'s "84.8%" is the damage
    multiplier at charge 0.9 (0.2 + 0.81 * 0.8), not a charge threshold.
    """
    return falling and charge > f32(_c(c).critical_min_charge)


def melee_damage(damage: float, ticks_since_attack: int, cooldown: float, *, falling: bool = False,
                 c: MechanicsConstants | None = None) -> float:
    """Damage of a player's melee hit before the target's armor, in float32 [18].

    damage * (0.2 + charge^2 * 0.8), then * 1.5 if critical. No enchantments (v1 skips them).
    """
    c = _c(c)
    charge = attack_charge(ticks_since_attack, cooldown, c)
    out = f32(f32(damage) * attack_damage_multiplier(charge, c))
    if is_critical(charge, falling, c):
        out = f32(out * f32(c.critical_multiplier))
    return out


def melee_damage_for(rules: Rules, held: str | None, ticks_since_attack: int, *, falling: bool = False) -> float:
    """``melee_damage`` with a vanilla held item (None = fist)."""
    s = attack_stats_for(rules, held)
    return melee_damage(s.damage, ticks_since_attack, s.cooldown, falling=falling, c=constants_for(rules))


def scale_damage_for_difficulty(amount: float, difficulty: Difficulty) -> float:
    """``Player.hurt`` scaling of damage from mobs (damage types with scaling
    ``when_caused_by_living_non_player`` [3] whose source is a non-player mob) [mem]:
    Peaceful 0, Easy min(x / 2 + 1, x), Normal x, Hard x * 3 / 2, in float32."""
    x = f32(amount)
    if difficulty == Difficulty.PEACEFUL:
        return 0.0  # [mem]
    if difficulty == Difficulty.EASY:
        return min(f32(f32(x / 2.0) + 1.0), x)  # [mem]
    if difficulty == Difficulty.HARD:
        return f32(f32(x * 3.0) / 2.0)  # [mem]
    return x


def fall_damage(fall_distance: float, c: MechanicsConstants | None = None) -> int:
    """Fall damage, ceil(distance - 3) and at least 0, in float32 [mem] (no armor reduction [3])."""
    g = f32(f32(fall_distance) - f32(_c(c).fall_damage_safe_blocks))
    return max(0, math.ceil(g))


@dataclass
class HurtCooldown:
    """Invulnerability after a hit (``LivingEntity.hurt``), [18] 10 ticks, details [mem].

    A hit that lands sets ``invulnerable_time = 20``; it counts down once per tick. While it is
    above 10, a new hit only deals the amount by which it exceeds the last one, and nothing if
    it does not.
    """

    invulnerable_time: int = 0
    last_hurt: float = 0.0

    def hurt(self, amount: float, c: MechanicsConstants | None = None) -> float:
        """Register a hit of ``amount`` (before armor); returns the amount that gets through."""
        c = _c(c)
        amount = f32(amount)
        if self.invulnerable_time > c.invulnerability_reset_ticks - c.invulnerability_ticks:
            if amount <= self.last_hurt:
                return 0.0
            through = f32(amount - self.last_hurt)
            self.last_hurt = amount
            return through
        self.last_hurt = amount
        self.invulnerable_time = c.invulnerability_reset_ticks
        return amount

    def tick(self) -> None:
        if self.invulnerable_time > 0:
            self.invulnerable_time -= 1


# =============================================================================
# Armor [17]
# =============================================================================


def armor_damage(damage: float, armor: float, toughness: float = 0.0, c: MechanicsConstants | None = None) -> float:
    """Damage after armor: dmg * (1 - min(20, max(A/5, A - 4*dmg/(T+8))) / 25) [17], in float32.

    Evaluated as ``CombatRules.getDamageAfterAbsorb`` does [mem]: f = 2 + T/4,
    g = clamp(A - dmg/f, A * 0.2, 20), result dmg * (1 - g/25).
    """
    c = _c(c)
    d, a, t = f32(damage), f32(armor), f32(toughness)
    coef = f32(c.armor_damage_coefficient)
    f = f32(f32(c.armor_toughness_offset / c.armor_damage_coefficient) + f32(t / coef))
    g = f32(a - f32(d / f))
    floor = f32(a * f32(1.0 / c.armor_floor_divisor))
    g = floor if g < floor else min(g, f32(c.armor_cap))
    h = f32(g / f32(c.armor_divisor))
    return f32(d * f32(1.0 - h))


def armor_totals(rules: Rules, worn: Iterable[str]) -> tuple[int, float]:
    """Total armor points and toughness of worn vanilla items (points floored, as the game does [mem])."""
    points, toughness = 0.0, 0.0
    for name in worn:
        a = rules.item(name).armor
        if a is None:
            raise ValueError(f"{name!r} is not armor")
        points += a.points
        toughness += a.toughness
    return math.floor(points), f32(toughness)


def armor_damage_for(rules: Rules, damage: float, worn: Iterable[str]) -> float:
    """``armor_damage`` for a set of worn vanilla armor items."""
    points, toughness = armor_totals(rules, worn)
    return armor_damage(damage, points, toughness, constants_for(rules))


# =============================================================================
# Smelting and fuel [20]
# =============================================================================


def smelt_ticks(c: MechanicsConstants | None = None) -> int:
    """Ticks to smelt one item in a furnace (200 [20])."""
    return int(_c(c).smelt_ticks)


def fuel_ticks(rules: Rules, item: str) -> int | None:
    """Burn time of a vanilla item in ticks, None if it is not fuel (rules file; [20] and [mem])."""
    return rules.item(item).fuel_ticks


def items_per_fuel(rules: Rules, item: str) -> float:
    """Items one unit of fuel smelts (coal 8, log 1.5, stick 0.5); 0 if it is not fuel."""
    ticks = fuel_ticks(rules, item)
    return 0.0 if ticks is None else ticks / smelt_ticks(constants_for(rules))

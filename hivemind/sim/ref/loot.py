"""Loot-table sampler: block drops and mob drops as item-class counts (contracts 5; DESIGN.md 4.2 "Drops").

Usage::

    from hivemind.sim.ref.loot import KillContext, LootSampler
    from hivemind.sim.ref.rng import CounterRNG, Purpose
    rng = CounterRNG(seed=1)
    loot = LootSampler()
    loot.block_drops("stone", "wooden_pickaxe", rng.stream(0, 600, Purpose.LOOT, entity=3))  # {cobblestone: 1}
    loot.block_drops("stone", None, rng.stream(0, 601, Purpose.LOOT, entity=3))              # {} (not harvested)
    loot.entity_drops("zombie", rng.stream(0, 602, Purpose.LOOT, entity=40), KillContext.by_player())

**Tables.** Blocks use the class-level drop tables of ``data/concepts-1.21.1.json``
(``block_classes[].drops.table``): the vanilla table already specialised to a v1 player break (no
Fortune, no Silk Touch, no explosion, block-state conditions resolved) with item names replaced by
item-class names. Mobs use the vanilla tables of the rules file (``entities/<name>``), evaluated
with vanilla item names (so ``furnace_smelt`` uses the vanilla smelting recipes exactly) and mapped
to item classes at the end. An unsheared sheep uses ``entities/sheep/<colour>`` (one wool plus the
plain sheep table), as the game does [mem]; all colours give the class ``wool``. Tables referenced
from other tables but missing from the rules file are in ``EXTRA_LOOT_TABLES`` (from the pinned
mcmeta ``1.21.1-data`` [3]).

**Harvesting.** A block gives drops only when broken with a tool that can harvest it
(``concepts.can_harvest``; the server checks ``hasCorrectToolForDrops`` before rolling the table
[mem]). Breaking stone by hand gives nothing.

**Semantics** (vanilla 1.21.1 ``LootTable``/``LootPool`` [mem]). Table functions, then pools in
order. A pool's conditions are tested once; then ``rolls`` (+ bonus_rolls x luck; luck is 0 in v1)
times: every entry is expanded, testing its conditions (alternatives: the first child that expands;
group: all children; sequence: children until one fails), candidates with weight 0 are dropped
(weight + quality x luck), one candidate is drawn by weight (no draw when only one is left), and
its stacks pass through the entry's functions, then the pool's, then the table's. A function with
conditions runs only when they hold. Conditions are tested left to right and stop at the first
deciding term. Stacks with count 0 vanish. Supported:

- entries ``item``, ``tag`` (``expand``: one candidate per tag member; else every member),
  ``loot_table`` (nested, by id or inline), ``empty``, ``alternatives``, ``group``, ``sequence``;
- number providers: constants (``Math.round`` to ints), ``constant``, ``uniform`` (ints:
  ``Mth.nextInt``, floats: ``Mth.nextFloat``), ``binomial`` (n trials with probability p);
  level-based values: constants and ``linear``;
- functions ``set_count`` (with ``add``), ``limit_count``, ``apply_bonus`` (``ore_drops``,
  ``uniform_bonus_count``, ``binomial_with_bonus_count``: the last draws ``extra`` trials even at
  Fortune 0), ``explosion_decay``, ``furnace_smelt``, ``enchanted_count_increase`` (Looting); and
  ``COMPONENT_ONLY_FUNCTIONS``, which change item components the class level does not see
  (potion, bottle amplifier, names) and are identities here;
- conditions ``random_chance``, ``random_chance_with_enchanted_bonus``, ``table_bonus``,
  ``survives_explosion``, ``match_tool`` (item lists and tags, enchantment predicates),
  ``inverted``, ``any_of``, ``all_of``, ``killed_by_player``, ``entity_properties`` (``flags``
  is_on_fire, ``type`` with entity tags, ``type_specific`` slime size and raider captain,
  ``equipment`` main-hand enchantments), ``damage_source_properties`` (``tags``,
  ``source_entity``, ``direct_entity``, ``is_direct``), ``block_state_property`` and
  ``location_check`` when the caller provides the state or a location test.

Anything else raises ``UnsupportedLoot``: an unknown type, function, condition, provider or key
(``validate_table`` checks a whole table statically), and a condition whose context is missing
(a slime without a size, a location check without a test). Nothing is silently ignored.
Enchantments never apply in v1 (DESIGN.md 4.2): tools and weapons carry none, so Silk Touch,
Fortune and Looting are level 0. Luck is 0.

**Randomness.** Every draw comes from a ``DrawStream`` with purpose ``Purpose.LOOT``
(``rng.stream(episode, tick, Purpose.LOOT, entity)``), one stream per loot event: two events of the
same tick need different ``entity`` ids (e.g. the breaking agent's id for a block, a mob-slot based
id for a death). Draws, in the evaluation order above: a chance test takes ``uniform() < p``; an
integer uniform takes ``randint(lo, hi)`` and no draw when lo >= hi; a float uniform
``lo + uniform() * (hi - lo)``; a binomial one ``uniform()`` per trial; a weighted choice
``below(total weight)``; ``explosion_decay`` one ``uniform() <= 1/radius`` per item. A draw is
taken only when its outcome is random (so the C++ core can reproduce the stream exactly).
Not covered here: mobs' equipment drops (a skeleton's bow, a zombie's tools), which are not loot
tables, and experience.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from hivemind.rules import load_concepts, load_rules
from hivemind.sim.ref.inventory import ItemRef, item_id
from hivemind.sim.ref.rng import DrawStream, Purpose

if TYPE_CHECKING:
    from hivemind.rules import Rules
    from hivemind.rules.classes import Concepts

Drops = dict[int, int]  # item-class id -> count

MAX_NESTING = 8  # nested loot_table entries deeper than this raise (the game stops recursion too [mem])

# Tables that rules-file tables reference but the rules file does not carry; verbatim from the
# pinned mcmeta 1.21.1-data, data/minecraft/loot_table/gameplay/fishing/fish.json [3].
EXTRA_LOOT_TABLES: dict[str, dict] = {
    "gameplay/fishing/fish": {
        "type": "minecraft:fishing",
        "pools": [{"bonus_rolls": 0.0, "rolls": 1.0, "entries": [
            {"type": "minecraft:item", "name": "minecraft:cod", "weight": 60},
            {"type": "minecraft:item", "name": "minecraft:salmon", "weight": 25},
            {"type": "minecraft:item", "name": "minecraft:tropical_fish", "weight": 2},
            {"type": "minecraft:item", "name": "minecraft:pufferfish", "weight": 13},
        ]}],
        "random_sequence": "minecraft:gameplay/fishing/fish",
    },
}
# Non-item tags the tables use, from the pinned mcmeta 1.21.1-data [3]:
# tags/entity_type/skeletons.json and tags/enchantment/smelts_loot.json.
ENTITY_TYPE_TAGS: dict[str, tuple[str, ...]] = {
    "skeletons": ("skeleton", "stray", "wither_skeleton", "skeleton_horse", "bogged"),
}
ENCHANTMENT_TAGS: dict[str, tuple[str, ...]] = {"smelts_loot": ("fire_aspect",)}
# Functions that only set item components (no change of item or count): identities at class level.
COMPONENT_ONLY_FUNCTIONS = frozenset({
    "set_potion",  # tipped arrows of strays and bogged
    "set_ominous_bottle_amplifier",  # pillager captains' bottle (its random amplifier is not drawn)
    "copy_components", "copy_name",  # block-entity names and contents
})
# Entity names in loot contexts ("this", "attacker", ...); 1.21 names [mem].
ENTITY_TARGETS = ("this", "attacker", "direct_attacker", "attacking_player")


class UnsupportedLoot(ValueError):
    """A loot table uses something this sampler does not implement, or a context it was not given."""


def _ns(name: str) -> str:
    return name.removeprefix("#").removeprefix("minecraft:")


# =============================================================================
# Contexts
# =============================================================================


@dataclass(frozen=True)
class KillContext:
    """How a mob died. Entity types are vanilla names (``"player"``, ``"skeleton"``, ``"arrow"``).

    killed_by_player: a player hurt the mob recently (the game's lastHurtByPlayer, within 100
        ticks [mem]), whatever finally killed it; ``killed_by_player`` conditions test this.
    on_fire: the mob was burning when it died (meat drops cooked).
    attacker: the entity that caused the killing damage, None for environmental deaths
        (sunlight fire, falls, lava).
    direct_attacker: the entity that dealt it directly (an arrow); None = the attacker.
    damage_tags: damage-type tags of the killing damage (only tags a table tests matter,
        e.g. ``is_lightning``).
    weapon_enchantments: enchantments of the attacker's main-hand item (empty in v1).
    slime_size: the size of a slime or magma cube (1 = smallest); None = unknown.
    is_captain: a raider is a patrol or raid captain; None = unknown.
    sheared: a sheep has been sheared; colour: a sheep's wool colour (any colour gives ``wool``).
    """

    killed_by_player: bool = False
    on_fire: bool = False
    attacker: str | None = None
    direct_attacker: str | None = None
    damage_tags: frozenset[str] = frozenset()
    weapon_enchantments: Mapping[str, int] = field(default_factory=dict)
    slime_size: int | None = None
    is_captain: bool | None = None
    sheared: bool = False
    colour: str = "white"

    @classmethod
    def by_player(cls, on_fire: bool = False, **kw: Any) -> KillContext:
        """A player's melee kill (the usual case in v1)."""
        return cls(killed_by_player=True, on_fire=on_fire, attacker="player", **kw)


@dataclass
class _Ctx:
    """One loot evaluation: the random stream, what the conditions may ask, how names map to classes."""

    stream: DrawStream
    level: str  # "class": names are item-class names; "vanilla": vanilla item names
    tool: str | None = None  # the breaking tool (class or vanilla name per level); None = hand
    has_tool: bool = False  # the TOOL parameter exists (block breaks), even for an empty hand
    tool_enchantments: Mapping[str, int] = field(default_factory=dict)
    this: str | None = None  # vanilla type of the killed entity
    kill: KillContext | None = None
    explosion_radius: float | None = None
    block_state: Mapping[str, str] | None = None
    location_check: Callable[[Mapping[str, Any]], bool] | None = None
    depth: int = 0


# =============================================================================
# Static validation
# =============================================================================

_TABLE_KEYS = {"type", "pools", "functions", "random_sequence"}
_POOL_KEYS = {"rolls", "bonus_rolls", "entries", "conditions", "functions"}
_ENTRY_KEYS = {
    "item": {"type", "name", "weight", "quality", "functions", "conditions"},
    "tag": {"type", "name", "expand", "weight", "quality", "functions", "conditions"},
    "loot_table": {"type", "value", "weight", "quality", "functions", "conditions"},
    "empty": {"type", "weight", "quality", "functions", "conditions"},
    "alternatives": {"type", "children", "conditions"},
    "group": {"type", "children", "conditions"},
    "sequence": {"type", "children", "conditions"},
}
_FUNCTION_KEYS = {
    "set_count": {"count", "add"},
    "limit_count": {"limit"},
    "apply_bonus": {"enchantment", "formula", "parameters"},
    "explosion_decay": set(),
    "furnace_smelt": set(),
    "enchanted_count_increase": {"enchantment", "count", "limit"},
    "set_potion": {"id"},
    "set_ominous_bottle_amplifier": {"amplifier"},
    "copy_components": {"source", "include", "exclude"},
    "copy_name": {"source"},
}
_CONDITION_KEYS = {
    "random_chance": {"chance"},
    "random_chance_with_enchanted_bonus": {"enchantment", "unenchanted_chance", "enchanted_chance"},
    "table_bonus": {"enchantment", "chances"},
    "survives_explosion": set(),
    "match_tool": {"predicate"},
    "inverted": {"term"},
    "any_of": {"terms"},
    "all_of": {"terms"},
    "killed_by_player": set(),
    "entity_properties": {"entity", "predicate"},
    "damage_source_properties": {"predicate"},
    "block_state_property": {"block", "properties"},
    "location_check": {"predicate", "offsetX", "offsetY", "offsetZ"},
}
_PROVIDER_KEYS = {"constant": {"value"}, "uniform": {"min", "max"}, "binomial": {"n", "p"}}
_BONUS_FORMULAS = {"ore_drops": set(), "uniform_bonus_count": {"bonusMultiplier"},
                   "binomial_with_bonus_count": {"extra", "probability"}}
_ENTITY_PREDICATE_KEYS = {"flags", "type", "type_specific", "equipment"}
_FLAGS = {"is_on_fire"}
_TYPE_SPECIFIC = {"slime": {"size"}, "raider": {"is_captain", "has_raid"}}
_DAMAGE_PREDICATE_KEYS = {"tags", "source_entity", "direct_entity", "is_direct"}
_ITEM_PREDICATE_KEYS = {"items", "predicates"}


def _check_keys(node: Mapping[str, Any], allowed: set[str], what: str, path: str) -> None:
    extra = set(node) - allowed
    if extra:
        raise UnsupportedLoot(f"{path}: {what} has unsupported keys {sorted(extra)}")


def _validate_provider(p: Any, path: str) -> None:
    if isinstance(p, (int, float)) and not isinstance(p, bool):
        return
    if not isinstance(p, dict) or "type" not in p:
        raise UnsupportedLoot(f"{path}: unsupported number provider {p!r}")
    kind = _ns(p["type"])
    if kind not in _PROVIDER_KEYS:
        raise UnsupportedLoot(f"{path}: unsupported number provider type {kind!r}")
    _check_keys(p, _PROVIDER_KEYS[kind] | {"type"}, f"provider {kind}", path)
    for k in _PROVIDER_KEYS[kind]:
        if k in p:
            _validate_provider(p[k], f"{path}.{k}")


def _validate_level_value(v: Any, path: str) -> None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return
    if isinstance(v, dict) and _ns(v.get("type", "")) == "linear":
        _check_keys(v, {"type", "base", "per_level_above_first"}, "linear level value", path)
        return
    raise UnsupportedLoot(f"{path}: unsupported level-based value {v!r}")


def _validate_entity_predicate(p: Any, path: str) -> None:
    if not isinstance(p, dict):
        raise UnsupportedLoot(f"{path}: entity predicate must be an object")
    _check_keys(p, _ENTITY_PREDICATE_KEYS, "entity predicate", path)
    if "flags" in p:
        _check_keys(p["flags"], _FLAGS, "entity flags", path)
    if "type_specific" in p:
        kind = _ns(p["type_specific"].get("type", ""))
        if kind not in _TYPE_SPECIFIC:
            raise UnsupportedLoot(f"{path}: unsupported type_specific predicate {kind!r}")
        _check_keys(p["type_specific"], _TYPE_SPECIFIC[kind] | {"type"}, f"type_specific {kind}", path)
    if "equipment" in p:
        _check_keys(p["equipment"], {"mainhand"}, "equipment predicate", path)
        _validate_item_predicate(p["equipment"]["mainhand"], f"{path}.equipment.mainhand")


def _validate_item_predicate(p: Any, path: str) -> None:
    if not isinstance(p, dict):
        raise UnsupportedLoot(f"{path}: item predicate must be an object")
    _check_keys(p, _ITEM_PREDICATE_KEYS, "item predicate", path)
    if "predicates" in p:
        _check_keys(p["predicates"], {"minecraft:enchantments"}, "item sub-predicates", path)
        for e in p["predicates"]["minecraft:enchantments"]:
            _check_keys(e, {"enchantments", "levels"}, "enchantment predicate", path)


def _validate_condition(c: Any, path: str) -> None:
    kind = _ns(c.get("condition", "")) if isinstance(c, dict) else ""
    if kind not in _CONDITION_KEYS:
        raise UnsupportedLoot(f"{path}: unsupported condition {kind or c!r}")
    _check_keys(c, _CONDITION_KEYS[kind] | {"condition"}, f"condition {kind}", path)
    if kind == "random_chance":
        _validate_provider(c["chance"], path)
    elif kind == "random_chance_with_enchanted_bonus":
        _validate_level_value(c["enchanted_chance"], path)
    elif kind == "inverted":
        _validate_condition(c["term"], f"{path}.term")
    elif kind in ("any_of", "all_of"):
        for i, t in enumerate(c["terms"]):
            _validate_condition(t, f"{path}.terms[{i}]")
    elif kind == "match_tool":
        _validate_item_predicate(c["predicate"], path)
    elif kind == "entity_properties":
        if c["entity"] not in ENTITY_TARGETS:
            raise UnsupportedLoot(f"{path}: unsupported entity target {c['entity']!r}")
        _validate_entity_predicate(c["predicate"], path)
    elif kind == "damage_source_properties":
        p = c["predicate"]
        _check_keys(p, _DAMAGE_PREDICATE_KEYS, "damage source predicate", path)
        for k in ("source_entity", "direct_entity"):
            if k in p:
                _validate_entity_predicate(p[k], f"{path}.{k}")
        for t in p.get("tags", ()):
            _check_keys(t, {"id", "expected"}, "damage tag predicate", path)


def _validate_function(f: Any, path: str) -> None:
    kind = _ns(f.get("function", "")) if isinstance(f, dict) else ""
    if kind not in _FUNCTION_KEYS:
        raise UnsupportedLoot(f"{path}: unsupported function {kind or f!r}")
    _check_keys(f, _FUNCTION_KEYS[kind] | {"function", "conditions"}, f"function {kind}", path)
    for i, c in enumerate(f.get("conditions", ())):
        _validate_condition(c, f"{path}.conditions[{i}]")
    if kind in ("set_count", "enchanted_count_increase"):
        _validate_provider(f["count"], path)
    elif kind == "limit_count":
        _check_keys(f["limit"], {"min", "max"}, "limit", path)
        for k in ("min", "max"):
            if k in f["limit"]:
                _validate_provider(f["limit"][k], path)
    elif kind == "apply_bonus":
        formula = _ns(f["formula"])
        if formula not in _BONUS_FORMULAS:
            raise UnsupportedLoot(f"{path}: unsupported apply_bonus formula {formula!r}")
        _check_keys(f.get("parameters", {}), _BONUS_FORMULAS[formula], f"apply_bonus {formula}", path)


def _validate_entry(e: Any, path: str) -> None:
    kind = _ns(e.get("type", "")) if isinstance(e, dict) else ""
    if kind not in _ENTRY_KEYS:
        raise UnsupportedLoot(f"{path}: unsupported entry type {kind or e!r}")
    _check_keys(e, _ENTRY_KEYS[kind], f"entry {kind}", path)
    for i, c in enumerate(e.get("conditions", ())):
        _validate_condition(c, f"{path}.conditions[{i}]")
    for i, f in enumerate(e.get("functions", ())):
        _validate_function(f, f"{path}.functions[{i}]")
    for i, child in enumerate(e.get("children", ())):
        _validate_entry(child, f"{path}.children[{i}]")
    if kind == "loot_table" and isinstance(e["value"], dict):
        validate_table(e["value"], f"{path}.value")


def validate_table(table: Mapping[str, Any], path: str = "table") -> None:
    """Raise ``UnsupportedLoot`` if anything in the table is not implemented (module docstring)."""
    _check_keys(table, _TABLE_KEYS, "table", path)
    for i, f in enumerate(table.get("functions", ())):
        _validate_function(f, f"{path}.functions[{i}]")
    for i, pool in enumerate(table.get("pools", ())):
        p = f"{path}.pools[{i}]"
        _check_keys(pool, _POOL_KEYS, "pool", p)
        _validate_provider(pool["rolls"], f"{p}.rolls")
        _validate_provider(pool.get("bonus_rolls", 0.0), f"{p}.bonus_rolls")
        for j, c in enumerate(pool.get("conditions", ())):
            _validate_condition(c, f"{p}.conditions[{j}]")
        for j, fn in enumerate(pool.get("functions", ())):
            _validate_function(fn, f"{p}.functions[{j}]")
        for j, e in enumerate(pool["entries"]):
            _validate_entry(e, f"{p}.entries[{j}]")


# =============================================================================
# Sampler
# =============================================================================


@dataclass
class _Stack:
    name: str  # item-class name (class level) or vanilla item name
    count: int


class LootSampler:
    """Samples block and mob drops as item-class counts (see the module docstring)."""

    def __init__(self, concepts: Concepts | None = None, rules: Rules | None = None):
        self.concepts: Concepts = concepts if concepts is not None else load_concepts()
        self.rules: Rules = rules if rules is not None else load_rules()
        # furnace_smelt: vanilla input -> vanilla result, from the rules' smelting recipes
        smelt: dict[str, tuple[str, int]] = {}
        ambiguous: set[str] = set()
        for r in self.rules.recipes:
            if r.type != "smelting" or r.result_item is None or not r.ingredients:
                continue
            for name in r.ingredients[0]:
                if name in smelt and smelt[name] != (r.result_item, r.result_count):
                    ambiguous.add(name)
                smelt.setdefault(name, (r.result_item, r.result_count))
        self._smelt_vanilla = {k: v for k, v in smelt.items() if k not in ambiguous}
        self._smelt_ambiguous = frozenset(ambiguous)
        # furnace_smelt at class level: class-level smelting recipes
        smelt_cls: dict[str, str] = {}
        for r in self.concepts.recipes:
            if r["type"] == "smelting":
                for name in r["ingredients"][0]:
                    smelt_cls[name] = r["result"]
        self._smelt_class = smelt_cls
        self._validated: set[str] = set()

    # -- public API --------------------------------------------------------------------

    def block_drops(self, block: str | int, tool: ItemRef | None, stream: DrawStream, *,
                    harvested: bool | None = None, block_state: Mapping[str, str] | None = None,
                    location_check: Callable[[Mapping[str, Any]], bool] | None = None) -> Drops:
        """Drops of one block of a class broken by a player holding ``tool`` (an item class; None =
        empty hand). ``harvested`` overrides the harvest check (default: ``concepts.can_harvest``)."""
        bc = self.concepts.block_class(block)
        tool_id = 0 if tool is None else item_id(self.concepts, tool)
        if harvested is None:
            harvested = bool(self.concepts.can_harvest[tool_id, bc.id])
        if not harvested or bc.drops is None:
            return {}
        ctx = _Ctx(stream=_loot_stream(stream), level="class", has_tool=True,
                   tool=None if tool_id == 0 else self.concepts.item_class(tool_id).name,
                   block_state=block_state, location_check=location_check)
        return self._sample(bc.drops["table"], ctx, key=f"block class {bc.id}")

    def entity_table(self, entity: str | int, kill: KillContext | None = None) -> str:
        """The loot-table path a dead mob uses (vanilla entity name or entity class)."""
        name = self._entity_name(entity)
        if name == "sheep" and not (kill and kill.sheared):
            return f"entities/sheep/{kill.colour if kill else 'white'}"
        return f"entities/{name}"

    def entity_drops(self, entity: str | int, stream: DrawStream, kill: KillContext | None = None, *,
                     killed_by_player: bool | None = None, on_fire: bool | None = None) -> Drops:
        """Drops of a dead mob. ``kill`` describes the death; ``killed_by_player`` and ``on_fire``
        are shortcuts (a player kill sets the attacker to ``player``)."""
        if kill is None:
            kill = KillContext.by_player() if killed_by_player else KillContext()
        elif killed_by_player is not None and killed_by_player != kill.killed_by_player:
            kill = replace(kill, killed_by_player=killed_by_player,
                           attacker=kill.attacker or ("player" if killed_by_player else None))
        if on_fire is not None:
            kill = replace(kill, on_fire=on_fire)
        name = self._entity_name(entity)
        path = self.entity_table(name, kill)
        ctx = _Ctx(stream=_loot_stream(stream), level="vanilla", this=name, kill=kill)
        return self._sample(self._table(path), ctx, key=path)

    def sample_table(self, table: Mapping[str, Any], stream: DrawStream, *, level: str = "vanilla",
                     tool: str | None = None, kill: KillContext | None = None, this: str | None = None,
                     explosion_radius: float | None = None, block_state: Mapping[str, str] | None = None,
                     location_check: Callable[[Mapping[str, Any]], bool] | None = None) -> Drops:
        """Roll any table (``level``: "vanilla" or "class" item names) in a given context: a block
        break with ``tool`` when ``kill`` is None, else a mob death."""
        if level not in ("vanilla", "class"):
            raise ValueError(f"level must be 'vanilla' or 'class', got {level!r}")
        ctx = _Ctx(stream=_loot_stream(stream), level=level, tool=tool, has_tool=kill is None, this=this,
                   kill=kill, explosion_radius=explosion_radius, block_state=block_state,
                   location_check=location_check)
        return self._sample(table, ctx)

    # -- tables -----------------------------------------------------------------------------

    def _entity_name(self, entity: str | int) -> str:
        if isinstance(entity, str) and f"entities/{_ns(entity)}" in self.rules.raw["loot_tables"]:
            return _ns(entity)
        cls = self.concepts.entity_class(entity)
        if len(cls.members) != 1:
            raise UnsupportedLoot(f"entity class {cls.name} has {len(cls.members)} members; name the vanilla entity")
        return cls.members[0]

    def _table(self, path: str) -> dict:
        path = _ns(path)
        if path in EXTRA_LOOT_TABLES:
            return EXTRA_LOOT_TABLES[path]
        try:
            return self.rules.loot_table(path)
        except KeyError:
            raise UnsupportedLoot(f"loot table {path!r} is not in the rules file") from None

    def _validate(self, table: Mapping[str, Any], key: str | None) -> None:
        """``validate_table``, once per persistent table (``key``: a table path or block class)."""
        if key is None or key not in self._validated:
            validate_table(table, key or "table")
            if key is not None:
                self._validated.add(key)

    def _sample(self, table: Mapping[str, Any], ctx: _Ctx, key: str | None = None) -> Drops:
        self._validate(table, key)
        out: Drops = {}

        def emit(stack: _Stack) -> None:
            if stack.count > 0:
                cid = self._class_id(stack.name, ctx.level)
                out[cid] = out.get(cid, 0) + stack.count

        self._roll_table(table, ctx, emit, [])
        return out

    def _class_id(self, name: str, level: str) -> int:
        if level == "class":
            return self.concepts.item_class(name).id
        return self.concepts.item_class_of(_ns(name)).id

    # -- evaluation ---------------------------------------------------------------------------

    def _roll_table(self, table: Mapping[str, Any], ctx: _Ctx, emit: Callable[[_Stack], None],
                    outer: list[list[Mapping[str, Any]]]) -> None:
        chain = [list(table.get("functions", ())), *outer]
        for pool in table.get("pools", ()):
            if not self._all(pool.get("conditions", ()), ctx):
                continue
            pool_chain = [list(pool.get("functions", ())), *chain]
            rolls = self._int(pool["rolls"], ctx)  # + floor(bonus_rolls * luck): luck is 0 in v1
            for _ in range(rolls):
                self._roll_pool(pool, ctx, emit, pool_chain)

    def _roll_pool(self, pool: Mapping[str, Any], ctx: _Ctx, emit: Callable[[_Stack], None],
                   chain: list[list[Mapping[str, Any]]]) -> None:
        candidates: list[tuple[Mapping[str, Any], str | None]] = []  # (entry, tag member or None)
        for e in pool["entries"]:
            self._expand(e, ctx, candidates)
        weighted = [(e, m, w) for e, m in candidates if (w := max(math.floor(e.get("weight", 1)), 0)) > 0]
        if not weighted:
            return
        if len(weighted) == 1:
            entry, member, _ = weighted[0]
        else:
            j = ctx.stream.below(sum(w for _, _, w in weighted))
            for entry, member, w in weighted:
                j -= w
                if j < 0:
                    break
        self._create(entry, member, ctx, emit, chain)

    def _expand(self, e: Mapping[str, Any], ctx: _Ctx, out: list[tuple[Mapping[str, Any], str | None]]) -> bool:
        """``LootPoolEntryContainer.expand``: True if the entry (or a child) produced candidates."""
        kind = _ns(e["type"])
        if not self._all(e.get("conditions", ()), ctx):
            return False
        if kind == "alternatives":
            return any(self._expand(c, ctx, out) for c in e["children"])  # stops at the first success
        if kind == "group":
            for c in e["children"]:
                self._expand(c, ctx, out)
            return True
        if kind == "sequence":
            return all(self._expand(c, ctx, out) for c in e["children"])  # stops at the first failure
        if kind == "tag" and e.get("expand", False):
            out.extend((e, m) for m in self._item_tag(e["name"], ctx))
            return True
        out.append((e, None))
        return True

    def _create(self, e: Mapping[str, Any], member: str | None, ctx: _Ctx, emit: Callable[[_Stack], None],
                chain: list[list[Mapping[str, Any]]]) -> None:
        """``createItemStack`` of a chosen candidate, passing stacks through the function chain."""
        kind = _ns(e["type"])
        full = [list(e.get("functions", ())), *chain]

        def finish(stack: _Stack) -> None:
            for fns in full:
                for fn in fns:
                    stack = self._apply(fn, stack, ctx)
            emit(stack)

        if kind == "item":
            finish(_Stack(e["name"], 1))
        elif kind == "tag":
            for m in [member] if member is not None else self._item_tag(e["name"], ctx):
                finish(_Stack(m, 1))
        elif kind == "loot_table":
            if ctx.depth >= MAX_NESTING:
                raise UnsupportedLoot(f"loot tables nested deeper than {MAX_NESTING}")
            value = e["value"]
            if isinstance(value, dict):
                table = value
                self._validate(table, None)
            else:
                table = self._table(value)
                self._validate(table, _ns(value))
            ctx.depth += 1
            try:
                self._roll_table(table, ctx, finish, [])
            finally:
                ctx.depth -= 1
        elif kind != "empty":
            raise UnsupportedLoot(f"entry type {kind!r} cannot be created")

    def _item_tag(self, name: str, ctx: _Ctx) -> list[str]:
        if ctx.level != "vanilla":
            raise UnsupportedLoot(f"item tag {name!r} in a class-level table")
        return list(self.rules.tag("item", _ns(name)))

    # -- numbers ---------------------------------------------------------------------------------

    def _float(self, p: Any, ctx: _Ctx) -> float:
        if isinstance(p, (int, float)):
            return float(p)
        kind = _ns(p["type"])
        if kind == "constant":
            return float(p["value"])
        if kind == "uniform":  # Mth.nextFloat(random, min, max)
            lo, hi = self._float(p["min"], ctx), self._float(p["max"], ctx)
            return lo if lo >= hi else lo + ctx.stream.uniform() * (hi - lo)
        if kind == "binomial":
            return float(self._int(p, ctx))
        raise UnsupportedLoot(f"number provider {kind!r}")

    def _int(self, p: Any, ctx: _Ctx) -> int:
        if isinstance(p, (int, float)):
            return math.floor(p + 0.5)  # NumberProvider.getInt: Math.round(getFloat) [mem]
        kind = _ns(p["type"])
        if kind == "constant":
            return math.floor(float(p["value"]) + 0.5)
        if kind == "uniform":  # Mth.nextInt(random, min, max), both ends included
            lo, hi = self._int(p["min"], ctx), self._int(p["max"], ctx)
            return lo if lo >= hi else ctx.stream.randint(lo, hi)
        if kind == "binomial":
            n, prob = self._int(p["n"], ctx), self._float(p["p"], ctx)
            return sum(1 for _ in range(n) if ctx.stream.uniform() < prob)
        raise UnsupportedLoot(f"number provider {kind!r}")

    @staticmethod
    def _level_value(v: Any, level: int) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if _ns(v["type"]) == "linear":
            return float(v["base"]) + float(v["per_level_above_first"]) * (level - 1)
        raise UnsupportedLoot(f"level-based value {v!r}")

    # -- conditions ------------------------------------------------------------------------------

    def _all(self, conds: Any, ctx: _Ctx) -> bool:
        return all(self._test(c, ctx) for c in conds)  # stops at the first false

    def _test(self, c: Mapping[str, Any], ctx: _Ctx) -> bool:
        kind = _ns(c["condition"])
        if kind == "random_chance":
            return ctx.stream.uniform() < self._float(c["chance"], ctx)
        if kind == "random_chance_with_enchanted_bonus":
            lvl = self._weapon_level(c["enchantment"], ctx)
            p = self._level_value(c["enchanted_chance"], lvl) if lvl > 0 else float(c["unenchanted_chance"])
            return ctx.stream.uniform() < p
        if kind == "table_bonus":
            chances = c["chances"]
            return ctx.stream.uniform() < float(chances[min(self._tool_level(c["enchantment"], ctx), len(chances) - 1)])
        if kind == "survives_explosion":
            if ctx.explosion_radius is None:
                return True
            return ctx.stream.uniform() <= 1.0 / ctx.explosion_radius
        if kind == "match_tool":
            return ctx.has_tool and self._item_matches(c["predicate"], ctx.tool, ctx.tool_enchantments, ctx)
        if kind == "inverted":
            return not self._test(c["term"], ctx)
        if kind == "any_of":
            return any(self._test(t, ctx) for t in c["terms"])
        if kind == "all_of":
            return all(self._test(t, ctx) for t in c["terms"])
        if kind == "killed_by_player":
            return ctx.kill is not None and ctx.kill.killed_by_player
        if kind == "entity_properties":
            return self._entity_matches(c["predicate"], c["entity"], ctx)
        if kind == "damage_source_properties":
            return self._damage_matches(c["predicate"], ctx)
        if kind == "block_state_property":
            if ctx.block_state is None:
                raise UnsupportedLoot("block_state_property needs the block state (block_state=...)")
            return _state_matches(c.get("properties", {}), ctx.block_state)
        if kind == "location_check":
            if ctx.location_check is None:
                raise UnsupportedLoot("location_check needs a location test (location_check=...)")
            return bool(ctx.location_check(c))
        raise UnsupportedLoot(f"condition {kind!r}")

    def _tool_level(self, enchantment: str, ctx: _Ctx) -> int:
        return int(ctx.tool_enchantments.get(_ns(enchantment), 0))

    def _weapon_level(self, enchantment: str, ctx: _Ctx) -> int:
        if ctx.kill is None or ctx.kill.attacker is None:
            return 0
        return int(ctx.kill.weapon_enchantments.get(_ns(enchantment), 0))

    def _item_matches(self, pred: Mapping[str, Any], item: str | None, enchants: Mapping[str, int], ctx: _Ctx) -> bool:
        """``ItemPredicate.test`` on the tool or weapon (None = an empty stack)."""
        if "items" in pred:
            items = pred["items"]
            names = [items] if isinstance(items, str) else list(items)
            allowed: set[str] = set()
            for n in names:
                if n.startswith("#"):
                    allowed |= set(self._item_tag(n, ctx))
                else:
                    allowed.add(_ns(n))
            if item is None or _ns(item) not in allowed:
                return False
        for e in pred.get("predicates", {}).get("minecraft:enchantments", ()):
            if not _enchantment_matches(e, enchants):
                return False
        return True

    def _entity_type(self, target: str, ctx: _Ctx) -> str | None:
        """Vanilla type of a context entity, None if absent."""
        if target == "this":
            return ctx.this if ctx.kill is not None else None
        k = ctx.kill
        if k is None:
            return None
        if target == "attacker":
            return k.attacker
        if target == "direct_attacker":
            return k.direct_attacker or k.attacker
        if target == "attacking_player":
            return "player" if k.killed_by_player else None
        raise UnsupportedLoot(f"entity target {target!r}")

    def _entity_matches(self, pred: Mapping[str, Any], target: str, ctx: _Ctx) -> bool:
        """``EntityPredicate`` on a context entity; absent entities match nothing."""
        if ctx.kill is None and target == "this" and pred == {}:
            return True  # a block break: the breaking player exists
        etype = self._entity_type(target, ctx)
        if etype is None:
            if ctx.kill is None and target == "this":
                raise UnsupportedLoot(f"entity predicate {pred} on the breaking player is not supported")
            return False
        return self._entity_type_matches(pred, etype, target, ctx)

    def _entity_type_matches(self, pred: Mapping[str, Any], etype: str, target: str, ctx: _Ctx) -> bool:
        kill = ctx.kill
        if "type" in pred and not _type_matches(pred["type"], etype):
            return False
        if "flags" in pred:
            for flag, want in pred["flags"].items():
                if target != "this" or kill is None:
                    raise UnsupportedLoot(f"entity flag {flag} of {target!r} is not known")
                if flag == "is_on_fire" and kill.on_fire != want:
                    return False
        if "type_specific" in pred:
            ts = pred["type_specific"]
            kind = _ns(ts["type"])
            if target != "this" or kill is None:
                raise UnsupportedLoot(f"type_specific {kind} of {target!r} is not known")
            if kind == "slime":
                if etype not in ("slime", "magma_cube"):
                    return False
                if kill.slime_size is None:
                    raise UnsupportedLoot("a slime's loot needs its size (KillContext.slime_size)")
                if not _int_range(ts.get("size"), kill.slime_size):
                    return False
            elif kind == "raider":
                if "has_raid" in ts:
                    raise UnsupportedLoot("raider has_raid is not known in v1")
                if "is_captain" in ts:
                    if kill.is_captain is None:
                        raise UnsupportedLoot("a raider's loot needs KillContext.is_captain")
                    if kill.is_captain != ts["is_captain"]:
                        return False
        if "equipment" in pred:
            if target not in ("attacker", "direct_attacker") or kill is None or etype != kill.attacker:
                return False  # only the attacker's own weapon is known; an arrow holds nothing
            hand = pred["equipment"]["mainhand"]
            if "items" in hand:
                raise UnsupportedLoot("the attacker's weapon item is not part of KillContext")
            for e in hand.get("predicates", {}).get("minecraft:enchantments", ()):
                if not _enchantment_matches(e, kill.weapon_enchantments):
                    return False
        return True

    def _damage_matches(self, pred: Mapping[str, Any], ctx: _Ctx) -> bool:
        kill = ctx.kill
        if kill is None:
            return False  # no damage source in a block break
        for t in pred.get("tags", ()):
            if (_ns(t["id"]) in kill.damage_tags) != bool(t["expected"]):
                return False
        if "is_direct" in pred:
            direct = kill.direct_attacker is None or kill.direct_attacker == kill.attacker
            if direct != pred["is_direct"]:
                return False
        for key, target in (("source_entity", "attacker"), ("direct_entity", "direct_attacker")):
            if key in pred:
                etype = self._entity_type(target, ctx)
                if etype is None or not self._entity_type_matches(pred[key], etype, target, ctx):
                    return False
        return True

    # -- functions -----------------------------------------------------------------------------

    def _apply(self, fn: Mapping[str, Any], stack: _Stack, ctx: _Ctx) -> _Stack:
        if not self._all(fn.get("conditions", ()), ctx):
            return stack
        kind = _ns(fn["function"])
        if kind == "set_count":
            base = stack.count if fn.get("add", False) else 0
            return _Stack(stack.name, max(0, base + self._int(fn["count"], ctx)))
        if kind == "limit_count":
            lim = fn["limit"]
            n = stack.count
            if "min" in lim:
                n = max(n, self._int(lim["min"], ctx))
            if "max" in lim:
                n = min(n, self._int(lim["max"], ctx))
            return _Stack(stack.name, n)
        if kind == "apply_bonus":
            return _Stack(stack.name, self._bonus(fn, stack.count, ctx))
        if kind == "explosion_decay":
            if ctx.explosion_radius is None:
                return stack
            keep = 1.0 / ctx.explosion_radius
            return _Stack(stack.name, sum(1 for _ in range(stack.count) if ctx.stream.uniform() <= keep))
        if kind == "furnace_smelt":
            return self._furnace_smelt(stack, ctx)
        if kind == "enchanted_count_increase":
            lvl = self._weapon_level(fn["enchantment"], ctx)
            if lvl == 0:
                return stack
            n = stack.count + math.floor(lvl * self._float(fn["count"], ctx) + 0.5)
            limit = int(fn.get("limit", 0))
            return _Stack(stack.name, min(n, limit) if limit > 0 else n)
        if kind in COMPONENT_ONLY_FUNCTIONS:
            return stack
        raise UnsupportedLoot(f"function {kind!r}")

    def _bonus(self, fn: Mapping[str, Any], count: int, ctx: _Ctx) -> int:
        """``ApplyBonusCount`` with the tool's enchantment level (0 in v1)."""
        lvl = self._tool_level(fn["enchantment"], ctx)
        formula = _ns(fn["formula"])
        params = fn.get("parameters", {})
        if formula == "ore_drops":
            if lvl <= 0:
                return count
            return count * (max(ctx.stream.below(lvl + 2) - 1, 0) + 1)
        if formula == "uniform_bonus_count":
            bound = int(params["bonusMultiplier"]) * lvl + 1
            return count + (ctx.stream.below(bound) if bound > 1 else 0)
        if formula == "binomial_with_bonus_count":
            p = float(params["probability"])
            return count + sum(1 for _ in range(lvl + int(params["extra"])) if ctx.stream.uniform() < p)
        raise UnsupportedLoot(f"apply_bonus formula {formula!r}")

    def _furnace_smelt(self, stack: _Stack, ctx: _Ctx) -> _Stack:
        """``SmeltItemFunction``: the smelting result, same count; unchanged without a recipe."""
        name = _ns(stack.name)
        if ctx.level == "class":
            result = self._smelt_class.get(name)
            return stack if result is None else _Stack(result, stack.count)
        if name in self._smelt_ambiguous:
            raise UnsupportedLoot(f"several smelting recipes take {name}")
        hit = self._smelt_vanilla.get(name)
        return stack if hit is None else _Stack(hit[0], stack.count)


# =============================================================================
# Helpers
# =============================================================================


def _loot_stream(stream: DrawStream) -> DrawStream:
    if stream.key.purpose != Purpose.LOOT:
        raise ValueError(f"loot draws need purpose LOOT, got {Purpose(stream.key.purpose).name}")
    return stream


def _type_matches(want: Any, etype: str) -> bool:
    """``EntityTypePredicate``: an id, an ``#tag`` or a list of ids."""
    names = [want] if isinstance(want, str) else list(want)
    for n in names:
        if n.startswith("#"):
            tag = _ns(n)
            if tag not in ENTITY_TYPE_TAGS:
                raise UnsupportedLoot(f"entity type tag {n!r}")
            if etype in ENTITY_TYPE_TAGS[tag]:
                return True
        elif _ns(n) == etype:
            return True
    return False


def _int_range(want: Any, value: int) -> bool:
    """``MinMaxBounds.Ints``: an exact number or {min, max}."""
    if want is None:
        return True
    if isinstance(want, (int, float)):
        return value == int(want)
    lo, hi = want.get("min"), want.get("max")
    return (lo is None or value >= lo) and (hi is None or value <= hi)


def _enchantment_matches(e: Mapping[str, Any], enchants: Mapping[str, int]) -> bool:
    """One ``minecraft:enchantments`` sub-predicate: some listed enchantment present at a level in range."""
    names = e.get("enchantments")
    if names is None:
        candidates = list(enchants)
    else:
        listed = [names] if isinstance(names, str) else list(names)
        candidates = []
        for n in listed:
            if n.startswith("#"):
                tag = _ns(n)
                if tag not in ENCHANTMENT_TAGS:
                    raise UnsupportedLoot(f"enchantment tag {n!r}")
                candidates += ENCHANTMENT_TAGS[tag]
            else:
                candidates.append(_ns(n))
    levels = e.get("levels")
    for c in candidates:
        lvl = int(enchants.get(c, 0))
        if lvl > 0 and _int_range(levels, lvl):
            return True
    return False


def _state_matches(props: Mapping[str, Any], state: Mapping[str, str]) -> bool:
    for prop, want in props.items():
        have = state.get(prop)
        if have is None:
            return False
        if isinstance(want, dict):
            v = int(have)
            if ("min" in want and v < int(want["min"])) or ("max" in want and v > int(want["max"])):
                return False
        elif str(want) != str(have):
            return False
    return True


@lru_cache(maxsize=1)
def default_sampler() -> LootSampler:
    """The sampler of the default concepts and rules files (cached)."""
    return LootSampler()


def block_drops(block: str | int, tool: ItemRef | None, stream: DrawStream, **kw: Any) -> Drops:
    """``LootSampler.block_drops`` with the default sampler."""
    return default_sampler().block_drops(block, tool, stream, **kw)


def entity_drops(entity: str | int, stream: DrawStream, kill: KillContext | None = None, **kw: Any) -> Drops:
    """``LootSampler.entity_drops`` with the default sampler."""
    return default_sampler().entity_drops(entity, stream, kill, **kw)

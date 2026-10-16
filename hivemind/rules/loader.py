"""Typed, read-only access to ``data/rules-1.21.1.json`` (docs/contracts.md section 2).

    from hivemind.rules import load_rules
    rules = load_rules()
    rules.block("stone").hardness                 # 1.5
    rules.item("iron_pickaxe").tool.speed         # 6.0
    rules.tag("block", "mineable/pickaxe")        # ('acacia_button', ...) also "#minecraft:..."
    rules.recipes_for("stick")                    # recipes whose result is stick
    rules.loot_table("blocks/gravel")             # vanilla loot table JSON
    rules.entity("zombie").max_health             # 20.0
    rules.constant("smelt_ticks")                 # 200 (rules.constant_entry(...) has the source)
    rules.mining_speed("wooden_pickaxe", "stone") # 2.0; tool None = bare hand
    rules.can_harvest("wooden_pickaxe", "iron_ore")  # False

``load_rules`` caches one ``Rules`` per path. The wrapper keeps the raw JSON in ``Rules.raw``;
the typed objects are frozen views of it (lists are tuples). Treat everything as read-only.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

GAME_VERSION = "1.21.1"
DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "data" / f"rules-{GAME_VERSION}.json"

Box = tuple[float, float, float, float, float, float]


def _name(name: str) -> str:
    """Accept ``stone``, ``minecraft:stone`` and, for tags, ``#minecraft:logs``."""
    name = name.lstrip("#")
    return name.removeprefix("minecraft:")


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class BlockState:
    name: str
    type: str  # "bool" | "int" | "enum" | "direction"
    values: tuple[str, ...]


@dataclass(frozen=True)
class Block:
    name: str
    id: int
    display_name: str
    hardness: float  # -1 = unbreakable
    resistance: float
    diggable: bool
    transparent: bool
    emit_light: int
    filter_light: int
    default_state: int
    min_state: int
    max_state: int
    states: tuple[BlockState, ...]
    mineable: tuple[str, ...]  # tool kinds: "axe", "hoe", "pickaxe", "shovel"
    harvest_tools: tuple[str, ...] | None  # items that get drops; None = any tool or hand
    requires_tool: bool
    needs_tier: str | None  # None | "stone" | "iron" | "diamond"
    loot_table: str | None  # "blocks/<name>" (or the standing variant's for wall blocks)
    collision_shapes: tuple[tuple[Box, ...], ...]
    collision_by_state: tuple[int, ...] | None

    @property
    def unbreakable(self) -> bool:
        return self.hardness < 0

    @property
    def state_count(self) -> int:
        return self.max_state - self.min_state + 1

    def collision_boxes(self, state: int | None = None) -> tuple[Box, ...]:
        """Collision boxes of a state id (default state if None), in block units."""
        state = self.default_state if state is None else state
        if not self.min_state <= state <= self.max_state:
            raise ValueError(f"state {state} is not a state of {self.name}")
        if self.collision_by_state is None:
            return self.collision_shapes[0]
        return self.collision_shapes[self.collision_by_state[state - self.min_state]]

    def state_values(self, state: int) -> dict[str, str]:
        """Property values of a state id (the first property varies slowest)."""
        if not self.min_state <= state <= self.max_state:
            raise ValueError(f"state {state} is not a state of {self.name}")
        offset, out = state - self.min_state, {}
        for s in reversed(self.states):
            offset, i = divmod(offset, len(s.values))
            out[s.name] = s.values[i]
        return dict(reversed(out.items()))


@dataclass(frozen=True)
class ToolRule:
    blocks: str | tuple[str, ...]  # "#tag" or block names
    speed: float | None
    correct_for_drops: bool | None


@dataclass(frozen=True)
class Tool:
    kind: str  # "pickaxe", "axe", "shovel", "hoe", "sword", "shears", "mace", "trident"
    tier: str | None  # "wood", "stone", "iron", "gold", "diamond", "netherite"
    speed: float | None  # speed on #mineable/<kind> (digging tools only)
    rules: tuple[ToolRule, ...]
    damage_per_block: int


@dataclass(frozen=True)
class Food:
    nutrition: int
    saturation: float
    eat_seconds: float
    can_always_eat: bool
    converts_to: str | None
    effects: tuple[dict, ...]


@dataclass(frozen=True)
class Armor:
    slot: str
    points: int
    toughness: float
    knockback_resistance: float


@dataclass(frozen=True)
class Item:
    name: str
    id: int
    display_name: str
    stack_size: int
    max_damage: int | None
    food: Food | None
    tool: Tool | None
    attack_damage: float | None  # attribute modifier; the player's base 1 is added by the simulator
    attack_speed: float | None  # attribute modifier; the player's base 4 is added by the simulator
    armor: Armor | None
    fuel_ticks: int | None
    equip_slot: str | None


@dataclass(frozen=True)
class Recipe:
    id: str
    type: str  # "crafting_shaped", "crafting_shapeless", "smelting", ...
    category: str | None
    group: str | None
    pattern: tuple[str, ...] | None
    key: dict[str, tuple[str, ...]] | None
    ingredients: tuple[tuple[str, ...], ...] | None  # each slot: alternative item names
    result_item: str | None
    result_count: int
    cookingtime: int | None
    experience: float | None
    special: bool


@dataclass(frozen=True)
class Entity:
    name: str
    id: int | None
    display_name: str
    width: float | None
    height: float | None
    category: str | None
    type: str | None
    max_health: float | None
    armor: float | None
    attack_damage: float | None
    speed: float | None


@dataclass(frozen=True)
class Constant:
    value: Any
    source: str  # "[n]" | "[mem]" | "[est]"


def _block(name: str, d: dict) -> Block:
    c = d["collision"]
    return Block(
        name=name, id=d["id"], display_name=d["display_name"], hardness=d["hardness"],
        resistance=d["resistance"], diggable=d["diggable"], transparent=d["transparent"],
        emit_light=d["emit_light"], filter_light=d["filter_light"], default_state=d["default_state"],
        min_state=d["min_state"], max_state=d["max_state"],
        states=tuple(BlockState(s["name"], s["type"], tuple(s["values"])) for s in d["states"]),
        mineable=tuple(d["mineable"]),
        harvest_tools=None if d["harvest_tools"] is None else tuple(d["harvest_tools"]),
        requires_tool=d["requires_tool"], needs_tier=d["needs_tier"], loot_table=d["loot_table"],
        collision_shapes=tuple(tuple(tuple(box) for box in shape) for shape in c["shapes"]),
        collision_by_state=None if c["by_state"] is None else tuple(c["by_state"]),
    )


def _item(name: str, d: dict) -> Item:
    f, t, a = d["food"], d["tool"], d["armor"]
    food = None if f is None else Food(f["nutrition"], f["saturation"], f["eat_seconds"], f["can_always_eat"],
                                       f["converts_to"], tuple(f["effects"]))
    tool = None if t is None else Tool(
        t["kind"], t["tier"], t["speed"],
        tuple(ToolRule(_freeze(r["blocks"]), r["speed"], r["correct_for_drops"]) for r in t["rules"]),
        t["damage_per_block"])
    armor = None if a is None else Armor(a["slot"], a["points"], a["toughness"], a["knockback_resistance"])
    return Item(name=name, id=d["id"], display_name=d["display_name"], stack_size=d["stack_size"],
                max_damage=d["max_damage"], food=food, tool=tool, attack_damage=d["attack_damage"],
                attack_speed=d["attack_speed"], armor=armor, fuel_ticks=d["fuel_ticks"], equip_slot=d["equip_slot"])


def _recipe(d: dict) -> Recipe:
    res = d["result"]
    return Recipe(
        id=d["id"], type=d["type"], category=d["category"], group=d["group"], pattern=_freeze(d["pattern"]),
        key=None if d["key"] is None else {k: tuple(v) for k, v in d["key"].items()},
        ingredients=_freeze(d["ingredients"]), result_item=res["item"] if res else None,
        result_count=res["count"] if res else 0, cookingtime=d["cookingtime"], experience=d["experience"],
        special=d["special"])


def _entity(name: str, d: dict) -> Entity:
    return Entity(name=name, **{k: d[k] for k in ("id", "display_name", "width", "height", "category", "type",
                                                   "max_health", "armor", "attack_damage", "speed")})


class Rules:
    """A thin typed wrapper over the rules JSON."""

    def __init__(self, raw: dict, path: Path | None = None):
        if raw.get("version") != GAME_VERSION:
            raise ValueError(f"rules file is for version {raw.get('version')}, expected {GAME_VERSION}")
        self.raw = raw
        self.path = path
        self.version: str = raw["version"]
        self.hash: str = raw["build"]["hash"]
        self.blocks: dict[str, Block] = {n: _block(n, d) for n, d in raw["blocks"].items()}
        self.items: dict[str, Item] = {n: _item(n, d) for n, d in raw["items"].items()}
        self.entities: dict[str, Entity] = {n: _entity(n, d) for n, d in raw["entities"].items()}
        self.recipes: tuple[Recipe, ...] = tuple(_recipe(d) for d in raw["recipes"])
        self.constants: dict[str, Constant] = {n: Constant(d["value"], d["source"])
                                               for n, d in raw["constants"].items()}
        self._tags = {kind: {n: tuple(v) for n, v in tags.items()} for kind, tags in raw["tags"].items()}
        self._tag_sets = {kind: {n: frozenset(v) for n, v in tags.items()} for kind, tags in raw["tags"].items()}
        by_result: dict[str, list[Recipe]] = defaultdict(list)
        for r in self.recipes:
            if r.result_item is not None:
                by_result[r.result_item].append(r)
        self._by_result = {k: tuple(v) for k, v in by_result.items()}
        self._block_by_state: dict[int, Block] = {}

    def __repr__(self) -> str:
        return (f"Rules(version={self.version!r}, hash={self.hash[:12]}, "
                f"blocks={len(self.blocks)}, items={len(self.items)})")

    # -- lookups ------------------------------------------------------------

    def block(self, name: str) -> Block:
        try:
            return self.blocks[_name(name)]
        except KeyError:
            raise KeyError(f"unknown block {name!r}") from None

    def item(self, name: str) -> Item:
        try:
            return self.items[_name(name)]
        except KeyError:
            raise KeyError(f"unknown item {name!r}") from None

    def entity(self, name: str) -> Entity:
        try:
            return self.entities[_name(name)]
        except KeyError:
            raise KeyError(f"unknown entity {name!r}") from None

    def tag(self, kind: str, name: str) -> tuple[str, ...]:
        """Members of a block or item tag, fully resolved, sorted."""
        try:
            return self._tags[kind][_name(name)]
        except KeyError:
            raise KeyError(f"unknown {kind} tag {name!r}") from None

    def in_tag(self, kind: str, name: str, tag: str) -> bool:
        try:
            return _name(name) in self._tag_sets[kind][_name(tag)]
        except KeyError:
            raise KeyError(f"unknown {kind} tag {tag!r}") from None

    def recipes_for(self, item: str) -> tuple[Recipe, ...]:
        """All recipes whose result is ``item`` (any type, special ones included)."""
        return self._by_result.get(_name(item), ())

    def loot_table(self, path: str) -> dict:
        """Vanilla loot table JSON by path, e.g. ``blocks/gravel`` or ``entities/zombie``."""
        try:
            return self.raw["loot_tables"][_name(path)]
        except KeyError:
            raise KeyError(f"unknown loot table {path!r}") from None

    def constant(self, name: str) -> Any:
        """The value of a hard-coded constant (see ``constant_entry`` for its source label)."""
        return self.constant_entry(name).value

    def constant_entry(self, name: str) -> Constant:
        try:
            return self.constants[name]
        except KeyError:
            raise KeyError(f"unknown constant {name!r}") from None

    def block_of_state(self, state: int) -> Block:
        """The block that owns a global block-state id."""
        if not self._block_by_state:
            for b in self.blocks.values():
                for s in range(b.min_state, b.max_state + 1):
                    self._block_by_state[s] = b
        try:
            return self._block_by_state[state]
        except KeyError:
            raise KeyError(f"unknown block state {state}") from None

    # -- tool rules -----------------------------------------------------------

    def _matches(self, rule: ToolRule, block: str) -> bool:
        if isinstance(rule.blocks, str):
            return block in self._tag_sets["block"][rule.blocks[1:]]
        return block in rule.blocks

    def mining_speed(self, tool: str | None, block: str) -> float:
        """Tool speed on a block from the 1.21.1 tool rules (no enchantments or effects).

        ``tool`` None (or an item without a tool component) is the bare hand: 1.0.
        """
        block = _name(block)
        t = self.item(tool).tool if tool else None
        if t is not None:
            for rule in t.rules:
                if rule.speed is not None and self._matches(rule, block):
                    return float(rule.speed)
        return float(self.constant("break_wrong_tool_speed"))

    def can_harvest(self, tool: str | None, block: str) -> bool:
        """Whether breaking ``block`` with ``tool`` (None = hand) gives drops."""
        b = self.block(block)
        if not b.requires_tool:
            return True
        t = self.item(tool).tool if tool else None
        if t is None:
            return False
        for rule in t.rules:
            if rule.correct_for_drops is not None and self._matches(rule, b.name):
                return bool(rule.correct_for_drops)
        return False

    # -- integrity --------------------------------------------------------------

    def compute_hash(self) -> str:
        """sha256 of the canonical JSON without ``build`` (must equal ``self.hash``)."""
        body = {k: v for k, v in self.raw.items() if k != "build"}
        text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=4)
def _load_cached(path: str, verify: bool) -> Rules:
    with open(path, encoding="utf-8") as f:
        rules = Rules(json.load(f), Path(path))
    if verify and rules.compute_hash() != rules.hash:
        raise ValueError(f"{path}: content does not match build.hash (edited by hand? rebuild it)")
    return rules


def load_rules(path: str | Path | None = None, verify: bool = True) -> Rules:
    """Load (once, cached per path) the rules file; ``verify`` checks ``build.hash``."""
    p = Path(path) if path is not None else DEFAULT_RULES_PATH
    return _load_cached(str(p.resolve()), verify)

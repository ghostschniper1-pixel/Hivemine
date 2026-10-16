"""Build ``data/rules-1.21.1.json`` from the pinned sources (docs/contracts.md section 2).

Usage::

    python -m hivemind.rules.fetch_sources          # once, fills data/sources/
    python -m hivemind.rules.build_rules [--sources DIR] [--out FILE] [--check]

Where each part comes from:

- blocks: minecraft-data ``pc/1.21.1/blocks.json`` (hardness, states, light, harvest tools),
  mcmeta tags (``mineable/*``, ``needs_*_tool``), collision shapes from minecraft-data
  ``pc/1.20.5/blockCollisionShapes.json``, loot table paths from mcmeta.
  minecraft-data's ``material`` field is not used (it is wrong for tier-gated blocks, e.g.
  iron_ore shows ``incorrect_for_wooden_tool``).
- items: mcmeta item components (stack size, durability, food, tool rules, attack
  modifiers), names and ids from minecraft-data; armor, fuel and equipment slots from
  ``constants.py`` (not in the data dumps).
- tags: every mcmeta tag of the kinds in ``TAG_KINDS`` (block, item, entity_type,
  enchantment, worldgen/biome), resolved recursively to sorted name lists. Loot conditions use
  entity_type tags (``#skeletons``) and enchantment tags (``#smelts_loot``); the biome tags
  give each biome its dimension.
- recipes: every mcmeta recipe, tag ingredients resolved to name lists. Special recipes
  (``crafting_special_*``, decorated pot, smithing) are kept and flagged ``special``.
- loot_tables: mcmeta ``loot_table/blocks/*``, ``loot_table/entities/**`` and the fishing
  tables (``gameplay/fishing`` and ``gameplay/fishing/*``; the guardians' drops roll
  ``gameplay/fishing/fish``) verbatim, plus every table an included table references through a
  ``minecraft:loot_table`` entry, transitively (``select_loot_tables``; a reference to a
  missing table fails the build).
- entities: the mcmeta ``entity_type`` registry; ids, sizes and categories from
  minecraft-data ``pc/1.20.5/entities.json``; stats from ``constants.py``.
- biomes: the mcmeta ``worldgen/biome`` registry (64 biomes in 1.21.1); ``id`` is the index in
  the registry (alphabetical, the order in which the game numbers a data-driven registry [mem]),
  ``display_name`` from minecraft-data ``language.json`` (``biome.minecraft.<name>``),
  ``dimension`` from the biome tags ``is_overworld``, ``is_nether`` and ``is_end`` (null when
  the biome is in none of them: ``the_void``). Cross-checked against minecraft-data
  ``pc/1.20.5/biomes.json`` (1.21 added no biome, so the 1.20.5 file still applies).
- constants: ``constants.all_constants()`` as ``{"value", "source"}``.

Representation choices not fixed by the contract:

- Names carry no ``minecraft:`` namespace, except inside the verbatim loot tables.
- ``states``: ``[{name, type, values}]`` in minecraft-data's order (the first property
  varies slowest); boolean values are ``["true", "false"]`` (index 0 = true).
- ``collision``: ``{"shapes": [boxes, ...], "by_state": [i, ...] | null}``: the block's
  distinct collision shapes, and for each state (index ``state - min_state``) the index of
  its shape; ``by_state`` is null when every state uses ``shapes[0]``. A shape is a list of
  boxes ``[x0, y0, z0, x1, y1, z1]`` in block units (empty = no collision).
- ``tool.rules``: ``[{blocks, speed, correct_for_drops}]`` in game order; ``blocks`` is
  ``"#tag"`` or a list of block names; ``speed`` is the speed on ``#mineable/<kind>`` for
  digging tools and null for swords, shears, mace and trident. ``tool.damage_per_block``
  is added (durability lost per block mined: 1, or 2 for swords, mace and trident).
- ``food`` adds ``eat_seconds``, ``can_always_eat``, ``converts_to`` and ``effects``.
- recipes add ``special`` (bool).
- ``tags`` is keyed by tag kind (``TAG_KINDS``); tags in subdirectories keep their path
  (``mineable/pickaxe``, ``exclusive_set/armor``, ``has_structure/village_plains``).
- ``biomes``: ``{name: {id, display_name, dimension}}`` (additive; see above).

The build cross-checks overlapping sources and fails on any disagreement that is not
listed in a ``TOLERATED_*`` table below with its reason (``build`` raises
``CrossCheckError``; ``main`` prints every check first). minecraft-data's 1.20 ``blockLoot``
and ``entityLoot`` are compared with the 1.21.1 loot tables for information only (known to
be stale).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hivemind.rules import constants as C
from hivemind.rules.fetch_sources import DEFAULT_DEST, DEFAULT_SOURCES_JSON, load_pins, verify_all

GAME_VERSION = "1.21.1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / f"rules-{GAME_VERSION}.json"

TOOL_KINDS = ("axe", "hoe", "pickaxe", "shovel")
NEEDS_TIER_TAGS = (("stone", "needs_stone_tool"), ("iron", "needs_iron_tool"), ("diamond", "needs_diamond_tool"))
# Item-name prefix -> tool tier, and the tier's incorrect_for_* block tag.
TIER_PREFIXES = {"wooden": "wood", "stone": "stone", "iron": "iron", "golden": "gold",
                 "diamond": "diamond", "netherite": "netherite"}
TIER_TAGS = {"wood": "incorrect_for_wooden_tool", "stone": "incorrect_for_stone_tool",
             "iron": "incorrect_for_iron_tool", "gold": "incorrect_for_gold_tool",
             "diamond": "incorrect_for_diamond_tool", "netherite": "incorrect_for_netherite_tool"}
TOOL_SUFFIXES = ("pickaxe", "shovel", "axe", "hoe", "sword")
SPECIAL_RECIPE_TYPES = ("crafting_decorated_pot", "smithing_transform", "smithing_trim")
COOKING_TYPES = ("smelting", "blasting", "smoking", "campfire_cooking")
# Tag kinds kept in the rules file: directory under data/minecraft/tags -> what the names are.
TAG_KINDS = ("block", "item", "entity_type", "enchantment", "worldgen/biome")
# Loot tables kept: path prefixes and single files (plus every table they reference).
LOOT_TABLE_PREFIXES = ("blocks/", "entities/", "gameplay/fishing/")
LOOT_TABLE_FILES = ("gameplay/fishing",)
# Biome tags that name a biome's dimension (a biome is in at most one).
BIOME_DIMENSION_TAGS = (("overworld", "is_overworld"), ("nether", "is_nether"), ("end", "is_end"))
FOOD_DEFAULT_EAT_SECONDS = 1.6  # [mem] food component default; = constants eat_ticks 32 / 20
TOOL_DEFAULT_DAMAGE_PER_BLOCK = 1  # [mem] tool component default
FLOAT_TOLERANCE = 1e-5  # float32 vs decimal printing (e.g. 7.2000003 vs 7.2)

# ---------------------------------------------------------------------------
# Tolerated differences between sources. Each entry needs a reason; anything else fails.
# ---------------------------------------------------------------------------

# minecraft-data block "material" values that do not describe the block's tool tag. For
# these the per-block tool-speed check substitutes the material implied by the mcmeta
# mineable tags (e.g. iron_ore: "incorrect_for_wooden_tool" -> "mineable/pickaxe").
TOLERATED_MATERIAL_SUBSTITUTION = {
    "incorrect_for_wooden_tool": "minecraft-data stores the tier tag instead of the mineable tag "
                                 "for tier-gated blocks; materials.json maps it to wooden tools only",
}

# (block, tool kind) pairs whose minecraft-data material omits a mineable tag that the
# 1.21.1 tags have. The rules file uses the tags, so these blocks mine fast with that tool.
TOLERATED_TOOL_SPEED = {
    ("moss_carpet", "hoe"): "minecraft-data material 'plant' omits mineable/hoe; the 1.21.1 "
                            "mineable/hoe tag lists moss_carpet (the tags are used)",
    ("pink_petals", "hoe"): "minecraft-data material 'plant' omits mineable/hoe; the 1.21.1 "
                            "mineable/hoe tag lists pink_petals (the tags are used)",
}

# Blocks where minecraft-data's harvestTools and the needs_*_tool tags disagree.
TOLERATED_HARVEST = {
    "crafter": "in needs_stone_tool, but minecraft-data lists no harvestTools (no tool required for "
               "drops), in which case the tier tag has no effect on drops. Kept as requires_tool=false "
               "(minecraft-data); UNVERIFIED, check in the real game (M2)",
}

# Display names: minecraft-data names wall-mounted blocks after the item that places them
# ("Torch" for wall_torch), while the language file has "Wall Torch" or no key at all.
# Tolerated when the minecraft-data name equals the standing variant's language name.
TOLERATED_WALL_DISPLAY_NAME = "minecraft-data uses the standing variant's (item's) name for wall-mounted blocks"

# Entity display names that differ between minecraft-data 1.20.5 and the 1.21.1 language
# file (none for the pinned sources).
TOLERATED_ENTITY_DISPLAY_NAMES: dict[str, str] = {}

# Biomes whose minecraft-data dimension differs from the one the 1.21.1 biome tags give.
TOLERATED_BIOME_DIMENSION = {
    "the_void": "the superflat 'void' preset's biome is in none of the is_overworld, is_nether and is_end "
                "tags (dimension null); minecraft-data files it under overworld",
}

# Recipes present in only one source. minecraft-data recipes are keyed by
# (result, sorted ingredient names); mcmeta recipes by id.
TOLERATED_RECIPES_ONLY_IN_MCMETA: dict[str, str] = {}
TOLERATED_RECIPES_ONLY_IN_MD: dict[tuple[str, tuple[str, ...]], str] = {
    ("map", ("filled_map", "paper")): "minecraft-data lists the map-extending special recipe "
                                      "(mcmeta crafting_special_mapextending) as a plain shaped recipe",
}
# Representation differences in the recipe check (not data disagreements):
# - shaped: the game trims empty edge rows and columns of a pattern (" # " -> "#");
#   minecraft-data stores the trimmed grid, so mcmeta patterns are trimmed before comparing.
# - shaped: minecraft-data expands each tag or alternative key into one recipe per member,
#   so the union of its recipes must cover every alternative of every slot.
# - shapeless: minecraft-data keeps only the FIRST alternative of each ingredient (in
#   definition order, tags expanded in order), so it must equal that one instance exactly.


class BuildError(RuntimeError):
    """A source is missing, malformed, or a cross-check failed."""


class CrossCheckError(BuildError):
    """Fatal cross-check mismatches; ``checks`` holds every check (for the report)."""

    def __init__(self, checks: list[CrossCheck]):
        failed = fatal_failures(checks)
        super().__init__("cross-check mismatches: " + "; ".join(
            f"{c.name} ({len(c.mismatches)}, first: {c.mismatches[0]})" for c in failed))
        self.checks = checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def strip_ns(name: str) -> str:
    """``minecraft:stone`` -> ``stone`` (other namespaces are kept)."""
    return name.removeprefix("minecraft:")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, UTF-8 text."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def rules_hash(rules: dict) -> str:
    """sha256 of the canonical JSON of ``rules`` without its ``build`` field."""
    body = {k: v for k, v in rules.items() if k != "build"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def floats_equal(a: float | None, b: float | None, tol: float = FLOAT_TOLERANCE) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_tree(root: Path) -> dict[str, Any]:
    """All ``*.json`` under ``root`` keyed by relative path without suffix (posix)."""
    if not root.is_dir():
        raise BuildError(f"missing source directory {root}")
    return {p.relative_to(root).with_suffix("").as_posix(): _read_json(p)
            for p in sorted(root.rglob("*.json"))}


@dataclass
class CrossCheck:
    """Result of one cross-check between sources."""

    name: str
    fatal: bool = True  # informational checks (stale sources) never fail the build
    compared: int = 0
    mismatches: list[str] = field(default_factory=list)
    tolerated: list[str] = field(default_factory=list)

    def line(self) -> str:
        status = "ok" if not self.mismatches else ("FAIL" if self.fatal else "info")
        return (f"[{status:4}] {self.name}: {self.compared} compared, {len(self.mismatches)} mismatches, "
                f"{len(self.tolerated)} tolerated")


def fatal_failures(checks: Iterable[CrossCheck]) -> list[CrossCheck]:
    """The fatal checks that found mismatches (the build fails if there is any)."""
    return [c for c in checks if c.fatal and c.mismatches]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@dataclass
class Sources:
    """The raw upstream data, as loaded from ``data/sources/``."""

    pins: dict
    md: dict[str, Any]  # minecraft-data files by short name, e.g. "blocks", "entities"
    tags: dict[str, dict[str, Any]]  # tag kind (TAG_KINDS) -> tag name -> raw tag JSON
    recipes: dict[str, Any]  # recipe id -> raw JSON
    loot: dict[str, Any]  # every mcmeta loot table: "blocks/stone" -> raw JSON (select_loot_tables picks)
    components: dict[str, Any]  # item name -> item components
    summary_blocks: dict[str, Any]  # block name -> [properties, default]
    entity_registry: list[str]
    enchantment_registry: list[str]
    biome_registry: list[str]  # in registry order


MD_FILES = {
    "version": "data/pc/1.21.1/version.json",
    "blocks": "data/pc/1.21.1/blocks.json",
    "items": "data/pc/1.21.1/items.json",
    "foods": "data/pc/1.21.1/foods.json",
    "materials": "data/pc/1.21.1/materials.json",
    "recipes": "data/pc/1.21.1/recipes.json",
    "language": "data/pc/1.21.1/language.json",
    "entities": "data/pc/1.20.5/entities.json",
    "shapes": "data/pc/1.20.5/blockCollisionShapes.json",
    "biomes": "data/pc/1.20.5/biomes.json",
    "block_loot": "data/pc/1.20/blockLoot.json",
    "entity_loot": "data/pc/1.20/entityLoot.json",
}


def load_sources(sources_dir: Path, pins: dict) -> Sources:
    specs = pins["sources"]
    md_root = sources_dir / specs["minecraft-data"]["dir"]
    md = {key: _read_json(md_root / rel) for key, rel in MD_FILES.items()}
    if md["version"]["minecraftVersion"] != GAME_VERSION:
        raise BuildError(f"minecraft-data version is {md['version']['minecraftVersion']}")

    data_root = sources_dir / specs["mcmeta-data"]["dir"]
    if _read_json(data_root / "version.json")["id"] != GAME_VERSION:
        raise BuildError("mcmeta data checkout is not 1.21.1")
    mc = data_root / "data" / "minecraft"
    tags = {kind: _read_tree(mc / "tags" / kind) for kind in TAG_KINDS}
    recipes = _read_tree(mc / "recipe")
    loot = _read_tree(mc / "loot_table")

    summary_root = sources_dir / specs["mcmeta-summary"]["dir"]
    if _read_json(summary_root / "version.json")["id"] != GAME_VERSION:
        raise BuildError("mcmeta summary is not 1.21.1")
    components = {strip_ns(k): v for k, v in _read_json(summary_root / "item_components" / "data.json").items()}
    summary_blocks = {strip_ns(k): v for k, v in _read_json(summary_root / "blocks" / "data.json").items()}
    registries = _read_json(summary_root / "registries" / "data.json")
    entity_registry, enchantment_registry, biome_registry = (
        [strip_ns(x) for x in registries[key]] for key in ("entity_type", "enchantment", "worldgen/biome"))
    return Sources(pins, md, tags, recipes, loot, components, summary_blocks, entity_registry,
                   enchantment_registry, biome_registry)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def resolve_tags(raw: dict[str, Any], known: set[str], kind: str) -> dict[str, list[str]]:
    """Resolve tags that reference other tags ('#name') into sorted name lists."""
    resolved: dict[str, list[str]] = {}

    def visit(name: str, stack: tuple[str, ...]) -> list[str]:
        if name in resolved:
            return resolved[name]
        if name in stack:
            raise BuildError(f"{kind} tag cycle: {' -> '.join(stack + (name,))}")
        out: set[str] = set()
        for entry in raw[name].get("values", []):
            ref, required = (entry["id"], entry.get("required", True)) if isinstance(entry, dict) else (entry, True)
            if ref.startswith("#"):
                sub = strip_ns(ref[1:])
                if sub in raw:
                    out.update(visit(sub, stack + (name,)))
                elif required:
                    raise BuildError(f"{kind} tag {name} references unknown tag {sub}")
            else:
                member = strip_ns(ref)
                if member in known:
                    out.add(member)
                elif required:
                    raise BuildError(f"{kind} tag {name} references unknown {kind} {member}")
        resolved[name] = sorted(out)
        return resolved[name]

    for tag_name in sorted(raw):
        visit(tag_name, ())
    return resolved


def first_alternative(ing: Any, raw_item_tags: dict[str, Any]) -> str:
    """The first item of an ingredient in definition order (tags expanded in order)."""
    if isinstance(ing, list):
        return first_alternative(ing[0], raw_item_tags)
    if "item" in ing:
        return strip_ns(ing["item"])
    entry = raw_item_tags[strip_ns(ing["tag"])]["values"][0]
    ref = entry["id"] if isinstance(entry, dict) else entry
    return first_alternative({"tag": ref[1:]} if ref.startswith("#") else {"item": ref}, raw_item_tags)


def ingredient_names(ing: Any, item_tags: dict[str, list[str]]) -> list[str]:
    """A recipe ingredient (item, tag, or list of alternatives) -> sorted item names."""
    if isinstance(ing, list):
        names: set[str] = set()
        for alt in ing:
            names.update(ingredient_names(alt, item_tags))
        return sorted(names)
    if "item" in ing:
        return [strip_ns(ing["item"])]
    if "tag" in ing:
        tag = strip_ns(ing["tag"])
        if tag not in item_tags:
            raise BuildError(f"recipe ingredient uses unknown item tag {tag}")
        return list(item_tags[tag])
    raise BuildError(f"unknown ingredient form {ing!r}")


# ---------------------------------------------------------------------------
# Tools: evaluation of the 1.21.1 tool component
# ---------------------------------------------------------------------------


def normalize_tool_rule(rule: dict) -> dict:
    blocks = rule["blocks"]
    if isinstance(blocks, str):
        blocks = "#" + strip_ns(blocks[1:]) if blocks.startswith("#") else [strip_ns(blocks)]
    else:
        blocks = [strip_ns(b) for b in blocks]
    return {"blocks": blocks, "speed": rule.get("speed"), "correct_for_drops": rule.get("correct_for_drops")}


def _rule_matches(rule: dict, block: str, block_tag_sets: dict[str, frozenset[str]]) -> bool:
    blocks = rule["blocks"]
    if isinstance(blocks, str):
        return block in block_tag_sets[blocks[1:]]
    return block in blocks


def tool_speed(rules: list[dict], block: str, block_tag_sets: dict[str, frozenset[str]]) -> float:
    """Mining speed of a tool on a block: the first rule with a speed that matches, else 1."""
    for rule in rules:
        if rule["speed"] is not None and _rule_matches(rule, block, block_tag_sets):
            return float(rule["speed"])
    return float(C.GAME["break_wrong_tool_speed"].value)


def tool_correct_for_drops(rules: list[dict], block: str, block_tag_sets: dict[str, frozenset[str]]) -> bool:
    """Whether a tool harvests a block: the first rule with correct_for_drops that matches."""
    for rule in rules:
        if rule["correct_for_drops"] is not None and _rule_matches(rule, block, block_tag_sets):
            return bool(rule["correct_for_drops"])
    return False


def tool_kind_and_tier(name: str) -> tuple[str, str | None]:
    for suffix in TOOL_SUFFIXES:
        if name.endswith("_" + suffix):
            prefix = name[: -len(suffix) - 1]
            if prefix not in TIER_PREFIXES:
                raise BuildError(f"tool {name} has an unknown tier prefix {prefix!r}")
            return suffix, TIER_PREFIXES[prefix]
    return name, None  # shears, mace, trident


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def _states(md_states: list[dict]) -> list[dict]:
    out = []
    for s in md_states:
        values = s.get("values") or (["true", "false"] if s["type"] == "bool" else None)
        if values is None or len(values) != s["num_values"]:
            raise BuildError(f"state {s['name']} has no usable value list")
        out.append({"name": s["name"], "type": s["type"], "values": list(values)})
    return out


def _collision(name: str, n_states: int, shapes: dict) -> dict:
    ref = shapes["blocks"][name]
    if isinstance(ref, int):
        return {"shapes": [shapes["shapes"][str(ref)]], "by_state": None}
    if len(ref) != n_states:
        raise BuildError(f"{name}: {len(ref)} collision shapes for {n_states} states")
    distinct = list(dict.fromkeys(ref))  # shape ids in order of first appearance
    if len(distinct) == 1:
        return {"shapes": [shapes["shapes"][str(ref[0])]], "by_state": None}
    return {"shapes": [shapes["shapes"][str(i)] for i in distinct], "by_state": [distinct.index(i) for i in ref]}


# Blocks whose loot table is another block's ("dropsLike" in the game code) [mem]: wall
# variants of torches, signs, banners, heads and coral fans have no loot table file of
# their own. Derived by removing "wall_" from the name; listed in the build output.
def _loot_table_for(name: str, loot: dict[str, Any]) -> tuple[str | None, bool]:
    if f"blocks/{name}" in loot:
        return f"blocks/{name}", False
    if "wall_" in name:
        standing = name.replace("wall_", "", 1)
        if f"blocks/{standing}" in loot:
            return f"blocks/{standing}", True
    return None, False


# Default collision for 1.21.1 blocks without a 1.20.5 shape: a full cube [est]. The
# build lists such blocks; for the pinned sources there are none.
DEFAULT_COLLISION = {"shapes": [[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]], "by_state": None}  # [est]


def build_blocks(src: Sources, block_tags: dict[str, list[str]], item_names: dict[int, str],
                 notes: list[str]) -> dict[str, dict]:
    tag_sets = {k: frozenset(v) for k, v in block_tags.items()}
    shapes = src.md["shapes"]
    missing_shapes: list[str] = []
    derived_loot: list[str] = []
    no_loot: list[str] = []
    blocks: dict[str, dict] = {}
    for b in src.md["blocks"]:
        name = b["name"]
        tiers = [tier for tier, tag in NEEDS_TIER_TAGS if name in tag_sets[tag]]
        if len(tiers) > 1:
            raise BuildError(f"{name} is in several needs_*_tool tags: {tiers}")
        harvest = b.get("harvestTools")
        harvest_tools = sorted(item_names[int(i)] for i in harvest) if harvest else None
        n_states = b["maxStateId"] - b["minStateId"] + 1
        if name in shapes["blocks"]:
            collision = _collision(name, n_states, shapes)
        else:
            missing_shapes.append(name)
            collision = DEFAULT_COLLISION
        loot_table, derived = _loot_table_for(name, src.loot)
        if derived:
            derived_loot.append(f"{name}->{loot_table}")
        if loot_table is None:
            no_loot.append(name)
        blocks[name] = {
            "id": b["id"],
            "display_name": b["displayName"],
            "hardness": b["hardness"] if b["hardness"] is not None else -1.0,
            "resistance": b["resistance"],
            "diggable": b["diggable"],
            "transparent": b["transparent"],
            "emit_light": b["emitLight"],
            "filter_light": b["filterLight"],
            "default_state": b["defaultState"],
            "min_state": b["minStateId"],
            "max_state": b["maxStateId"],
            "states": _states(b["states"]),
            "mineable": [k for k in TOOL_KINDS if name in tag_sets[f"mineable/{k}"]],
            "harvest_tools": harvest_tools,
            "requires_tool": harvest_tools is not None,
            "needs_tier": tiers[0] if tiers else None,
            "loot_table": loot_table,
            "collision": collision,
        }
    notes.append(f"blocks: {len(blocks)}; without a 1.20.5 collision shape (default full cube [est]): "
                 f"{len(missing_shapes)} {missing_shapes}")
    notes.append(f"blocks: loot table taken from the standing variant (dropsLike [mem]): {len(derived_loot)}")
    notes.append(f"blocks: no loot table: {len(no_loot)} {no_loot}")
    return blocks


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def _attack_modifiers(name: str, comp: dict) -> tuple[float | None, float | None]:
    damage = speed = None
    for m in comp.get("minecraft:attribute_modifiers", {}).get("modifiers", []):
        if m["operation"] != "add_value" or m["slot"] != "mainhand":
            raise BuildError(f"{name}: unexpected attribute modifier {m}")
        if m["type"] == "minecraft:generic.attack_damage":
            damage = m["amount"]
        elif m["type"] == "minecraft:generic.attack_speed":
            speed = m["amount"]
        else:
            raise BuildError(f"{name}: unexpected attribute {m['type']}")
    return damage, speed


def _food(comp: dict) -> dict | None:
    f = comp.get("minecraft:food")
    if f is None:
        return None
    effects = [
        {"effect": strip_ns(e["effect"]["id"]), "duration": e["effect"].get("duration", 0),
         "amplifier": e["effect"].get("amplifier", 0), "probability": e.get("probability", 1.0)}
        for e in f.get("effects", [])
    ]
    converts = f.get("using_converts_to")
    return {
        "nutrition": f["nutrition"],
        "saturation": f["saturation"],
        "eat_seconds": f.get("eat_seconds", FOOD_DEFAULT_EAT_SECONDS),
        "can_always_eat": f.get("can_always_eat", False),
        "converts_to": strip_ns(converts["id"]) if converts else None,
        "effects": effects,
    }


def _tool(name: str, comp: dict) -> dict | None:
    t = comp.get("minecraft:tool")
    if t is None:
        return None
    rules = [normalize_tool_rule(r) for r in t["rules"]]
    kind, tier = tool_kind_and_tier(name)
    speed = None
    if kind in TOOL_KINDS:
        own = [r["speed"] for r in rules if r["blocks"] == f"#mineable/{kind}"]
        if len(own) != 1:
            raise BuildError(f"{name}: expected one rule on #mineable/{kind}, found {len(own)}")
        speed = own[0]
        tier_rules = [r for r in rules if isinstance(r["blocks"], str) and r["blocks"].startswith("#incorrect_for_")]
        if len(tier_rules) != 1 or tier_rules[0]["blocks"] != "#" + TIER_TAGS[tier] \
                or tier_rules[0]["correct_for_drops"] is not False:
            raise BuildError(f"{name}: tier {tier} from the name disagrees with its tool rules {rules}")
    return {"kind": kind, "tier": tier, "speed": speed, "rules": rules,
            "damage_per_block": t.get("damage_per_block", TOOL_DEFAULT_DAMAGE_PER_BLOCK)}


def _armor_table(known_items: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for material, a in C.ARMOR.items():
        for piece, slot in C.ARMOR_PIECES.items():
            item = f"{material}_{piece}"
            if item not in known_items:
                raise BuildError(f"armor constant names an unknown item {item}")
            out[item] = {"slot": slot, "points": a.points.value[C.ARMOR_SLOTS.index(slot)],
                         "toughness": a.toughness.value, "knockback_resistance": a.knockback_resistance.value}
    for item, (slot, points, toughness, kb, _src) in C.ARMOR_EXTRA.items():
        if item not in known_items:
            raise BuildError(f"armor constant names an unknown item {item}")
        out[item] = {"slot": slot, "points": points, "toughness": toughness, "knockback_resistance": kb}
    missing = [n for n in known_items if n.endswith(tuple("_" + p for p in C.ARMOR_PIECES)) and n not in out]
    if missing:
        raise BuildError(f"armor items without hard-coded values: {sorted(missing)}")
    return out


def _fuel_table(known_items: set[str], item_tags: dict[str, list[str]], notes: list[str]) -> dict[str, int]:
    fuel: dict[str, int] = {}
    overrides: list[str] = []
    for ref, ticks, _src in C.FUEL:
        if ref.startswith("#"):
            if ref[1:] not in item_tags:
                raise BuildError(f"fuel constant names an unknown item tag {ref}")
            members = item_tags[ref[1:]]
        else:
            if ref not in known_items:
                raise BuildError(f"fuel constant names an unknown item {ref}")
            members = [ref]
        for m in members:
            if m in fuel and fuel[m] != ticks:
                overrides.append(f"{m}: {fuel[m]}->{ticks} ({ref})")
            fuel[m] = ticks
    excluded = set(item_tags[C.FUEL_EXCLUDED_TAG.value]) & set(fuel)
    for m in excluded:
        del fuel[m]
    notes.append(f"items: {len(fuel)} fuels; {len(excluded)} excluded as #{C.FUEL_EXCLUDED_TAG.value}; "
                 f"overrides by later entries: {overrides}")
    return fuel


def _equip_slot(name: str, armor: dict | None) -> str | None:
    if armor is not None:
        return armor["slot"]
    if name in C.EQUIP_SLOT_EXTRA:
        return C.EQUIP_SLOT_EXTRA[name]
    if name.endswith(C.EQUIP_HEAD_SUFFIXES):
        return "head"
    return None


def build_items(src: Sources, item_tags: dict[str, list[str]], notes: list[str]) -> dict[str, dict]:
    known = {it["name"] for it in src.md["items"]}
    armor = _armor_table(known)
    fuel = _fuel_table(known, item_tags, notes)
    for name in C.EQUIP_SLOT_EXTRA:
        if name not in known:
            raise BuildError(f"equip-slot constant names an unknown item {name}")
    items: dict[str, dict] = {}
    for it in src.md["items"]:
        name = it["name"]
        comp = src.components.get(name)
        if comp is None:
            raise BuildError(f"item {name} has no item components")  # also checked in check_items
        damage, speed = _attack_modifiers(name, comp)
        items[name] = {
            "id": it["id"],
            "display_name": it["displayName"],
            "stack_size": comp["minecraft:max_stack_size"],
            "max_damage": comp.get("minecraft:max_damage"),
            "food": _food(comp),
            "tool": _tool(name, comp),
            "attack_damage": damage,
            "attack_speed": speed,
            "armor": armor.get(name),
            "fuel_ticks": fuel.get(name),
            "equip_slot": _equip_slot(name, armor.get(name)),
        }
    return items


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------


def build_recipes(src: Sources, item_tags: dict[str, list[str]]) -> list[dict]:
    out = []
    for rid in sorted(src.recipes):
        r = src.recipes[rid]
        rtype = strip_ns(r["type"])
        pattern = key = ingredients = result = cookingtime = experience = None
        if rtype == "crafting_shaped":
            pattern = list(r["pattern"])
            key = {ch: ingredient_names(ing, item_tags) for ch, ing in r["key"].items()}
        elif rtype == "crafting_shapeless":
            ingredients = [ingredient_names(ing, item_tags) for ing in r["ingredients"]]
        elif rtype in COOKING_TYPES:
            ingredients = [ingredient_names(r["ingredient"], item_tags)]
            cookingtime = r["cookingtime"]
            experience = r["experience"]
        elif rtype == "stonecutting":
            ingredients = [ingredient_names(r["ingredient"], item_tags)]
        elif rtype in ("smithing_transform", "smithing_trim"):
            ingredients = [ingredient_names(r[k], item_tags) for k in ("template", "base", "addition")]
        if "result" in r:
            res = r["result"]
            if set(res) - {"id", "count"}:
                raise BuildError(f"recipe {rid}: result with components is not supported: {res}")
            result = {"item": strip_ns(res["id"]), "count": res.get("count", 1)}
        out.append({
            "id": rid,
            "type": rtype,
            "category": r.get("category"),
            "group": r.get("group") or None,
            "pattern": pattern,
            "key": key,
            "ingredients": ingredients,
            "result": result,
            "cookingtime": cookingtime,
            "experience": experience,
            "special": rtype.startswith("crafting_special_") or rtype in SPECIAL_RECIPE_TYPES,
        })
    return out


# ---------------------------------------------------------------------------
# Entities and constants
# ---------------------------------------------------------------------------


def build_entities(src: Sources) -> dict[str, dict]:
    md = {e["name"]: e for e in src.md["entities"]}
    lang = src.md["language"]
    for mob in C.MOBS:
        if mob not in src.entity_registry:
            raise BuildError(f"mob constant names an unknown entity {mob}")
    out: dict[str, dict] = {}
    for name in src.entity_registry:
        e = md.get(name)
        stats = C.MOBS.get(name)
        out[name] = {
            "id": e["id"] if e else None,
            "display_name": lang.get(f"entity.minecraft.{name}", e["displayName"] if e else name),
            "width": e["width"] if e else None,
            "height": e["height"] if e else None,
            "category": e["category"] if e else None,
            "type": e["type"] if e else None,
            "max_health": stats.max_health.value if stats else None,
            "armor": stats.armor.value if stats else None,
            "attack_damage": stats.attack_damage.value if stats else None,
            "speed": stats.speed.value if stats else None,
        }
    return out


def build_biomes(src: Sources, biome_tags: dict[str, list[str]], notes: list[str]) -> dict[str, dict]:
    """The worldgen/biome registry: id (registry index), English name and dimension."""
    lang = src.md["language"]
    dim_sets = {dim: frozenset(biome_tags[tag]) for dim, tag in BIOME_DIMENSION_TAGS}
    out: dict[str, dict] = {}
    for i, name in enumerate(src.biome_registry):
        dims = [dim for dim, members in dim_sets.items() if name in members]
        if len(dims) > 1:
            raise BuildError(f"biome {name} is in several dimension tags: {dims}")
        key = f"biome.minecraft.{name}"
        if key not in lang:
            raise BuildError(f"biome {name} has no English name ({key}) in language.json")
        out[name] = {"id": i, "display_name": lang[key], "dimension": dims[0] if dims else None}
    by_dim = Counter(str(b["dimension"]) for b in out.values())
    notes.append(f"biomes: {len(out)} ({dict(sorted(by_dim.items()))}); without a dimension tag: "
                 f"{sorted(n for n, b in out.items() if b['dimension'] is None)}")
    return out


def loot_references(node: Any) -> set[str]:
    """Paths of the tables that ``minecraft:loot_table`` entries reference by id (inline tables
    are part of the node and need nothing)."""
    out: set[str] = set()
    if isinstance(node, dict):
        ref = node.get("value") if node.get("type") == "minecraft:loot_table" else None
        if isinstance(ref, str):
            out.add(strip_ns(ref))
        for v in node.values():
            out |= loot_references(v)
    elif isinstance(node, list):
        for v in node:
            out |= loot_references(v)
    return out


def select_loot_tables(all_loot: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    """The loot tables of the rules file: LOOT_TABLE_PREFIXES and LOOT_TABLE_FILES, plus every
    table they reference, transitively. Fails on a reference to a table that does not exist."""
    for path in LOOT_TABLE_FILES:
        if path not in all_loot:
            raise BuildError(f"missing loot table {path}")
    roots = {k for k in all_loot if k.startswith(LOOT_TABLE_PREFIXES) or k in LOOT_TABLE_FILES}
    keep, todo = set(roots), sorted(roots)
    while todo:
        path = todo.pop()
        for ref in sorted(loot_references(all_loot[path])):
            if ref not in all_loot:
                raise BuildError(f"loot table {path} references the missing table {ref}")
            if ref not in keep:
                keep.add(ref)
                todo.append(ref)
    notes.append(f"loot tables: {len(keep)} of {len(all_loot)} kept; added only because another table "
                 f"references them: {sorted(keep - roots)}")
    return {k: all_loot[k] for k in sorted(keep)}


def build_constants() -> dict[str, dict]:
    pattern = re.compile(C.SOURCE_LABELS_PATTERN)
    out = {}
    for name, const in sorted(C.all_constants().items()):
        if not pattern.match(const.source):
            raise BuildError(f"constant {name} has a bad source label {const.source!r}")
        out[name] = {"value": const.value, "source": const.source}
    return out


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------


def check_items(src: Sources) -> list[CrossCheck]:
    """Stack sizes, durability and food: minecraft-data items/foods vs item components."""
    stack, dur, food = CrossCheck("stack sizes"), CrossCheck("durability"), CrossCheck("food values")
    md_names = {it["name"] for it in src.md["items"]}
    for name in sorted(md_names ^ set(src.components)):
        stack.mismatches.append(f"{name}: only in {'minecraft-data' if name in md_names else 'item components'}")
    for it in src.md["items"]:
        comp = src.components.get(it["name"])
        if comp is None:
            continue
        stack.compared += 1
        if it["stackSize"] != comp["minecraft:max_stack_size"]:
            stack.mismatches.append(f"{it['name']}: {it['stackSize']} vs {comp['minecraft:max_stack_size']}")
        dur.compared += 1
        if it.get("maxDurability") != comp.get("minecraft:max_damage"):
            dur.mismatches.append(f"{it['name']}: {it.get('maxDurability')} vs {comp.get('minecraft:max_damage')}")
    md_foods = {f["name"]: f for f in src.md["foods"]}
    comp_foods = {n: c["minecraft:food"] for n, c in src.components.items() if "minecraft:food" in c}
    for name in sorted(set(md_foods) ^ set(comp_foods)):
        food.mismatches.append(f"{name}: food only in {'foods.json' if name in md_foods else 'item components'}")
    for name in sorted(set(md_foods) & set(comp_foods)):
        food.compared += 1
        f, c = md_foods[name], comp_foods[name]
        if f["foodPoints"] != c["nutrition"] or not floats_equal(f["saturation"], c["saturation"]):
            food.mismatches.append(f"{name}: {f['foodPoints']}/{f['saturation']} vs {c['nutrition']}/{c['saturation']}")
        elif f["saturation"] != c["saturation"]:
            food.tolerated.append(f"{name}: saturation {f['saturation']} vs {c['saturation']} (float32 printing)")
    return [stack, dur, food]


def check_block_states(src: Sources, blocks: dict[str, dict]) -> CrossCheck:
    """minecraft-data block states vs the mcmeta block-state summary (names, values, default)."""
    chk = CrossCheck("block states")
    for name in sorted(set(blocks) ^ set(src.summary_blocks)):
        chk.mismatches.append(f"{name}: only in {'minecraft-data' if name in blocks else 'summary'}")
    for name in sorted(set(blocks) & set(src.summary_blocks)):
        chk.compared += 1
        props, default = src.summary_blocks[name]
        states = blocks[name]["states"]
        if {s["name"]: s["values"] for s in states} != props:
            chk.mismatches.append(f"{name}: states {states} vs {props}")
            continue
        offset = 0  # the first property varies slowest
        for s in states:
            offset = offset * len(s["values"]) + s["values"].index(default[s["name"]])
        b = blocks[name]
        if b["min_state"] + offset != b["default_state"]:
            chk.mismatches.append(f"{name}: default state {b['default_state']} vs {b['min_state'] + offset}")
    return chk


def check_harvest_tools(blocks: dict[str, dict], items: dict[str, dict],
                        block_tags: dict[str, list[str]]) -> CrossCheck:
    """minecraft-data harvestTools vs the item tool rules evaluated on the mcmeta tags."""
    chk = CrossCheck("harvest tools vs tool rules + tags")
    tag_sets = {k: frozenset(v) for k, v in block_tags.items()}
    tools = {n: it["tool"]["rules"] for n, it in items.items() if it["tool"]}
    for name, b in sorted(blocks.items()):
        if b["requires_tool"]:
            chk.compared += 1
            correct = sorted(t for t, rules in tools.items() if tool_correct_for_drops(rules, name, tag_sets))
            if correct != b["harvest_tools"]:
                chk.mismatches.append(f"{name}: minecraft-data {b['harvest_tools']} vs tool rules {correct}")
            if b["needs_tier"] and not b["mineable"]:
                chk.mismatches.append(f"{name}: needs {b['needs_tier']} tool but is in no mineable tag")
        elif b["needs_tier"] is not None:
            chk.compared += 1
            if name in TOLERATED_HARVEST:
                chk.tolerated.append(f"{name}: {TOLERATED_HARVEST[name]}")
            else:
                chk.mismatches.append(f"{name}: in needs_{b['needs_tier']}_tool but minecraft-data "
                                      "has no harvest tools")
    return chk


def check_tool_speeds(src: Sources, items: dict[str, dict], block_tags: dict[str, list[str]],
                      blocks: dict[str, dict]) -> CrossCheck:
    """materials.json speeds (via each block's material) vs the tool rules, for every block x tool."""
    chk = CrossCheck("tool speeds: materials.json vs tool rules")
    tag_sets = {k: frozenset(v) for k, v in block_tags.items()}
    materials = src.md["materials"]
    id_to_name = {it["id"]: it["name"] for it in src.md["items"]}
    tools = {n: it["tool"]["rules"] for n, it in items.items() if it["tool"]}
    substituted = Counter()
    for b in src.md["blocks"]:
        name, material = b["name"], b["material"]
        if material in TOLERATED_MATERIAL_SUBSTITUTION:
            kinds = blocks[name]["mineable"]
            if len(kinds) != 1:
                chk.mismatches.append(f"{name}: material {material} but mineable kinds {kinds}")
                continue
            material = f"mineable/{kinds[0]}"
            substituted[b["material"]] += 1
        md_speeds = {id_to_name[int(i)]: s for i, s in materials[material].items()}
        for tool_name, rules in sorted(tools.items()):
            chk.compared += 1
            real = tool_speed(rules, name, tag_sets)
            expected = md_speeds.get(tool_name, 1.0)
            if floats_equal(real, expected):
                continue
            kind = items[tool_name]["tool"]["kind"]
            if (name, kind) in TOLERATED_TOOL_SPEED:
                chk.tolerated.append(f"{name} with {tool_name}: {expected} vs {real}: "
                                     f"{TOLERATED_TOOL_SPEED[(name, kind)]}")
            else:
                chk.mismatches.append(f"{name} ({material}) with {tool_name}: "
                                      f"materials.json {expected} vs rules {real}")
    for material, n in sorted(substituted.items()):
        chk.tolerated.append(f"{n} blocks with material {material!r} compared as their mineable tag: "
                             f"{TOLERATED_MATERIAL_SUBSTITUTION[material]}")
    return chk


def _md_recipe_forms(src: Sources) -> list[tuple[str, int, str, Any]]:
    """minecraft-data recipes as (result, count, form, data); form 'shaped' or 'shapeless'."""
    names = {it["id"]: it["name"] for it in src.md["items"]}
    out = []
    for result_id, recipes in src.md["recipes"].items():
        for r in recipes:
            res = names[r["result"]["id"]]
            if int(result_id) != r["result"]["id"]:
                raise BuildError(f"minecraft-data recipe filed under {result_id} produces {res}")
            if "inShape" in r:
                grid = tuple(tuple(names[c] if c is not None else None for c in row) for row in r["inShape"])
                out.append((res, r["result"]["count"], "shaped", grid))
            else:
                out.append((res, r["result"]["count"], "shapeless", tuple(names[c] for c in r["ingredients"])))
    return out


def trim_pattern(pattern: list[str]) -> list[str]:
    """Remove empty edge rows and columns, as the game does when it loads a shaped recipe."""
    width = max(len(row) for row in pattern)
    rows = [row.ljust(width) for row in pattern]
    filled = [i for i, row in enumerate(rows) if row.strip()]
    cols = [c for c in range(width) if any(row[c] != " " for row in rows)]
    if not filled:
        return []
    return [row[cols[0]: cols[-1] + 1] for row in rows[filled[0]: filled[-1] + 1]]


def _shaped_grid(r: dict) -> tuple[tuple[frozenset[str] | None, ...], ...]:
    return tuple(tuple(frozenset(r["key"][ch]) if ch != " " else None for ch in row)
                 for row in trim_pattern(r["pattern"]))


def check_recipes(src: Sources, recipes: list[dict]) -> CrossCheck:
    """Crafting recipes in both sources: result, count and ingredients (see the notes at
    TOLERATED_RECIPES_ONLY_IN_MD for how minecraft-data represents alternatives)."""
    chk = CrossCheck("recipes: minecraft-data vs mcmeta (crafting)")
    shaped: dict[tuple[str, int], list[tuple[str, Any]]] = defaultdict(list)
    for r in recipes:
        if r["type"] == "crafting_shaped":
            shaped[(r["result"]["item"], r["result"]["count"])].append((r["id"], _shaped_grid(r)))

    def tolerate_md(res: str, names: Iterable[str | None]) -> bool:
        key = (res, tuple(sorted({n for n in names if n is not None})))
        if key in TOLERATED_RECIPES_ONLY_IN_MD:
            chk.tolerated.append(f"minecraft-data {key}: {TOLERATED_RECIPES_ONLY_IN_MD[key]}")
            return True
        return False

    # Shaped: every minecraft-data recipe is an instance of an mcmeta recipe, and the
    # instances cover every alternative of every slot.
    seen: dict[str, list[set[str]]] = {}
    md_shapeless: Counter = Counter()
    for res, count, form, data in _md_recipe_forms(src):
        if form == "shapeless":
            md_shapeless[(res, count, tuple(sorted(data)))] += 1
            continue
        chk.compared += 1
        matched = False
        for rid, grid in shaped.get((res, count), []):
            if len(grid) != len(data) or any(len(a) != len(b) for a, b in zip(grid, data, strict=True)):
                continue
            cells = [(a, b) for ra, rb in zip(grid, data, strict=True) for a, b in zip(ra, rb, strict=True)]
            if all((a is None and b is None) or (a is not None and b in a) for a, b in cells):
                slots = seen.setdefault(rid, [set() for _ in cells])
                for k, (_a, b) in enumerate(cells):
                    if b is not None:
                        slots[k].add(b)
                matched = True
        if not matched and not tolerate_md(res, (c for row in data for c in row)):
            chk.mismatches.append(f"minecraft-data shaped recipe for {count} {res} {data} has no mcmeta match")
    for (res, count), entries in sorted(shaped.items()):
        for rid, grid in entries:
            chk.compared += 1
            if rid not in seen:
                if rid in TOLERATED_RECIPES_ONLY_IN_MCMETA:
                    chk.tolerated.append(f"{rid}: {TOLERATED_RECIPES_ONLY_IN_MCMETA[rid]}")
                else:
                    chk.mismatches.append(f"mcmeta recipe {rid} ({count} {res}) has no minecraft-data counterpart")
                continue
            cells = [a for row in grid for a in row]
            for k, (alts, got) in enumerate(zip(cells, seen[rid], strict=True)):
                if alts is not None and set(alts) != got:
                    missing = sorted(set(alts) - got)
                    chk.mismatches.append(f"mcmeta recipe {rid} slot {k}: minecraft-data lacks {missing}")

    # Shapeless: minecraft-data holds exactly the first-alternative instance of each recipe, so
    # the other alternatives of a slot (tag members, listed alternatives) cannot be compared.
    raw_item_tags = src.tags["item"]
    mc_shapeless: Counter = Counter()
    unchecked_alts, unchecked_recipes = 0, 0
    for r in recipes:
        if r["type"] == "crafting_shapeless":
            first = tuple(sorted(first_alternative(ing, raw_item_tags) for ing in src.recipes[r["id"]]["ingredients"]))
            mc_shapeless[(r["result"]["item"], r["result"]["count"], first)] += 1
            extra = sum(len(alts) - 1 for alts in r["ingredients"])
            unchecked_alts += extra
            unchecked_recipes += extra > 0
    if unchecked_alts:
        chk.tolerated.append(f"shapeless: {unchecked_alts} non-first ingredient alternatives in {unchecked_recipes} "
                             "recipes are NOT compared (minecraft-data keeps only the first alternative)")
    for key in sorted(set(md_shapeless) | set(mc_shapeless)):
        chk.compared += 1
        n_md, n_mc = md_shapeless[key], mc_shapeless[key]
        if n_md != n_mc and not (n_mc == 0 and tolerate_md(key[0], key[2])):
            chk.mismatches.append(f"shapeless {key[1]} {key[0]} from {list(key[2])}: "
                                  f"{n_md} in minecraft-data, {n_mc} in mcmeta")
    return chk


def check_entities(src: Sources) -> CrossCheck:
    chk = CrossCheck("entities: 1.21.1 registry vs minecraft-data 1.20.5")
    md = {e["name"]: e for e in src.md["entities"]}
    lang = src.md["language"]
    for name in sorted(set(src.entity_registry) ^ set(md)):
        chk.mismatches.append(f"{name}: only in {'1.21.1 registry' if name in src.entity_registry else '1.20.5 data'}")
    for name in sorted(set(src.entity_registry) & set(md)):
        chk.compared += 1
        lang_name = lang.get(f"entity.minecraft.{name}")
        if lang_name is not None and lang_name != md[name]["displayName"]:
            if name in TOLERATED_ENTITY_DISPLAY_NAMES:
                chk.tolerated.append(f"{name}: {TOLERATED_ENTITY_DISPLAY_NAMES[name]}")
            else:
                chk.mismatches.append(f"{name}: display name {md[name]['displayName']!r} vs language {lang_name!r}")
    return chk


def check_biomes(src: Sources, biomes: dict[str, dict]) -> CrossCheck:
    """The 1.21.1 biome registry, language names and dimension tags vs minecraft-data 1.20.5
    biomes.json (names, ids, display names, dimension)."""
    chk = CrossCheck("biomes: 1.21.1 registry and tags vs minecraft-data 1.20.5")
    md = {b["name"]: b for b in src.md["biomes"]}
    for name in sorted(set(biomes) ^ set(md)):
        chk.mismatches.append(f"{name}: only in {'1.21.1 registry' if name in biomes else '1.20.5 data'}")
    for name in sorted(set(biomes) & set(md)):
        chk.compared += 1
        ours, theirs = biomes[name], md[name]
        if ours["id"] != theirs["id"]:
            chk.mismatches.append(f"{name}: id {ours['id']} (registry order) vs minecraft-data {theirs['id']}")
        if ours["display_name"] != theirs["displayName"]:
            chk.mismatches.append(f"{name}: display name {ours['display_name']!r} vs {theirs['displayName']!r}")
        if ours["dimension"] != theirs["dimension"]:
            if name in TOLERATED_BIOME_DIMENSION:
                chk.tolerated.append(f"{name}: {TOLERATED_BIOME_DIMENSION[name]}")
            else:
                chk.mismatches.append(f"{name}: dimension {ours['dimension']} (tags) vs {theirs['dimension']}")
    return chk


def check_display_names(src: Sources) -> CrossCheck:
    """minecraft-data display names vs the 1.21.1 language file (blocks and items)."""
    chk = CrossCheck("display names vs language.json")
    lang = src.md["language"]
    for kind, entries in (("block", src.md["blocks"]), ("item", src.md["items"])):
        for e in entries:
            keys = [f"{kind}.minecraft.{e['name']}"] + ([f"block.minecraft.{e['name']}"] if kind == "item" else [])
            found = [lang[k] for k in keys if k in lang]
            chk.compared += 1
            if e["displayName"] in found:
                continue
            standing = e["name"].replace("wall_", "", 1)
            if kind == "block" and "wall_" in e["name"] and lang.get(f"block.minecraft.{standing}") == e["displayName"]:
                chk.tolerated.append(f"{e['name']} {e['displayName']!r} vs {found}: {TOLERATED_WALL_DISPLAY_NAME}")
            else:
                chk.mismatches.append(f"{kind} {e['name']}: {e['displayName']!r} vs {found}")
    return chk


def check_derived_loot(src: Sources, blocks: dict[str, dict]) -> CrossCheck:
    """Loot tables taken from the standing variant vs minecraft-data's per-block drops."""
    chk = CrossCheck("derived (wall) loot tables vs minecraft-data drops")
    md = {b["name"]: b for b in src.md["blocks"]}
    for name, b in sorted(blocks.items()):
        if b["loot_table"] and b["loot_table"] != f"blocks/{name}":
            chk.compared += 1
            standing = b["loot_table"].split("/", 1)[1]
            if sorted(md[name]["drops"]) != sorted(md[standing]["drops"]):
                chk.mismatches.append(f"{name}: drops {md[name]['drops']} vs {standing} {md[standing]['drops']}")
    return chk


def _loot_items(node: Any, loot: dict[str, Any] | None = None, seen: frozenset[str] = frozenset()) -> set[str]:
    """Item names an (mcmeta) loot table can produce, ignoring conditions. With ``loot``,
    ``minecraft:loot_table`` entries are followed into the referenced tables."""
    out: set[str] = set()
    if isinstance(node, dict):
        if node.get("type") == "minecraft:item" and "name" in node:
            out.add(strip_ns(node["name"]))
        ref = node.get("value") if node.get("type") == "minecraft:loot_table" else None
        if loot is not None and isinstance(ref, str) and strip_ns(ref) in loot and strip_ns(ref) not in seen:
            out |= _loot_items(loot[strip_ns(ref)], loot, seen | {strip_ns(ref)})
        for v in node.values():
            out |= _loot_items(v, loot, seen)
    elif isinstance(node, list):
        for v in node:
            out |= _loot_items(v, loot, seen)
    return out


def check_stale_loot(src: Sources, blocks: dict[str, dict]) -> CrossCheck:
    """Informational: minecraft-data 1.20 blockLoot vs mcmeta 1.21.1 loot (known to be stale)."""
    chk = CrossCheck("block loot: minecraft-data 1.20 vs mcmeta 1.21.1 (info only)", fatal=False)
    for entry in src.md["block_loot"]:
        name = entry["block"]
        if name not in blocks:
            continue
        chk.compared += 1
        table = blocks[name]["loot_table"]
        mc = _loot_items(src.loot[table]) if table else set()
        md = {d["item"] for d in entry["drops"]}
        if md != mc:
            chk.mismatches.append(f"{name}: 1.20 {sorted(md)} vs 1.21.1 {sorted(mc)}")
    return chk


def check_stale_entity_loot(src: Sources) -> CrossCheck:
    """Informational: minecraft-data 1.20 entityLoot vs mcmeta 1.21.1 entity loot (known to be
    stale). The 1.21.1 side is the union of ``entities/<name>`` and its sub-tables (sheep
    colours), following ``minecraft:loot_table`` references."""
    chk = CrossCheck("entity loot: minecraft-data 1.20 vs mcmeta 1.21.1 (info only)", fatal=False)
    for entry in src.md["entity_loot"]:
        name = entry["entity"]
        if name not in src.entity_registry:
            chk.mismatches.append(f"{name}: not a 1.21.1 entity")
            continue
        chk.compared += 1
        tables = [k for k in src.loot if k == f"entities/{name}" or k.startswith(f"entities/{name}/")]
        mc = set().union(*(_loot_items(src.loot[k], src.loot) for k in tables))
        md = {d["item"] for d in entry["drops"]}
        if md != mc:
            chk.mismatches.append(f"{name}: 1.20 {sorted(md)} vs 1.21.1 {sorted(mc)}")
    return chk


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(sources_dir: Path = DEFAULT_DEST, sources_json: Path = DEFAULT_SOURCES_JSON,
          verify: bool = True, strict: bool = True) -> tuple[dict, list[CrossCheck], list[str]]:
    """Build the rules dict. Returns (rules, cross-checks, notes); raises BuildError on failure.

    ``strict`` (the default) raises ``CrossCheckError`` when a fatal cross-check finds a
    mismatch (contracts 2: the build fails on disagreement); ``strict=False`` returns the
    checks so the caller can report them (``main`` does, then fails).
    """
    pins = load_pins(sources_json)
    if verify:
        problems = verify_all(sources_json, sources_dir)
        if problems:
            raise BuildError("sources do not match data/sources.json (run fetch_sources): " + "; ".join(problems))
    src = load_sources(sources_dir, pins)
    notes: list[str] = []

    block_names = {b["name"] for b in src.md["blocks"]}
    item_names = {it["name"] for it in src.md["items"]}
    known = {"block": block_names, "item": item_names, "entity_type": set(src.entity_registry),
             "enchantment": set(src.enchantment_registry), "worldgen/biome": set(src.biome_registry)}
    tags = {kind: resolve_tags(src.tags[kind], known[kind], kind) for kind in TAG_KINDS}
    block_tags, item_tags = tags["block"], tags["item"]
    for tag in [f"mineable/{k}" for k in TOOL_KINDS] + [t for _, t in NEEDS_TIER_TAGS]:
        if tag not in block_tags:
            raise BuildError(f"missing block tag {tag}")
    for _dim, tag in BIOME_DIMENSION_TAGS:
        if tag not in tags["worldgen/biome"]:
            raise BuildError(f"missing biome tag {tag}")

    blocks = build_blocks(src, block_tags, {it["id"]: it["name"] for it in src.md["items"]}, notes)
    items = build_items(src, item_tags, notes)
    recipes = build_recipes(src, item_tags)
    for r in recipes:
        if r["result"] and r["result"]["item"] not in item_names:
            raise BuildError(f"recipe {r['id']} produces unknown item {r['result']['item']}")
    loot_tables = select_loot_tables(src.loot, notes)
    entities = build_entities(src)
    biomes = build_biomes(src, tags["worldgen/biome"], notes)
    constants = build_constants()

    checks = [
        *check_items(src),
        check_block_states(src, blocks),
        check_harvest_tools(blocks, items, block_tags),
        check_tool_speeds(src, items, block_tags, blocks),
        check_recipes(src, recipes),
        check_entities(src),
        check_biomes(src, biomes),
        check_display_names(src),
        check_derived_loot(src, blocks),
        check_stale_loot(src, blocks),
        check_stale_entity_loot(src),
    ]
    if strict and fatal_failures(checks):
        raise CrossCheckError(checks)
    notes.append(f"recipes: {len(recipes)} ({sum(r['special'] for r in recipes)} flagged special); "
                 f"types {dict(sorted(Counter(r['type'] for r in recipes).items()))}")
    notes.append(f"loot tables: {sum(k.startswith('blocks/') for k in loot_tables)} block, "
                 f"{sum(k.startswith('entities/') for k in loot_tables)} entity, "
                 f"{sum(k.startswith('gameplay/') for k in loot_tables)} gameplay")
    notes.append(f"tags: {', '.join(f'{len(tags[k])} {k}' for k in TAG_KINDS)}")
    notes.append(f"entities: {len(entities)} (registry 1.21.1; ids and sizes from minecraft-data 1.20.5); "
                 f"with hard-coded stats: {sorted(C.MOBS)}")

    rules = {
        "version": GAME_VERSION,
        "blocks": blocks,
        "items": items,
        "tags": tags,
        "recipes": recipes,
        "loot_tables": loot_tables,
        "entities": entities,
        "biomes": biomes,
        "constants": constants,
    }
    rules["build"] = {"sources": pins["sources"], "hash": rules_hash(rules)}
    return rules, checks, notes


def render(rules: dict) -> bytes:
    """The exact bytes written to the rules file."""
    return (canonical_json(rules) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build data/rules-1.21.1.json from the pinned sources.")
    ap.add_argument("--sources", type=Path, default=DEFAULT_DEST, help="source cache (data/sources)")
    ap.add_argument("--sources-json", type=Path, default=DEFAULT_SOURCES_JSON)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="build in memory and compare with --out; write nothing")
    ap.add_argument("--verbose", "-v", action="store_true", help="print every mismatch and tolerated difference")
    args = ap.parse_args(argv)

    try:
        rules, checks, notes = build(args.sources, args.sources_json, strict=False)  # report, then fail below
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    for note in notes:
        print(note)
    print("cross-checks:")
    for chk in checks:
        print("  " + chk.line())
        limit = None if args.verbose else 10
        for m in chk.mismatches[:limit]:
            print(f"      mismatch: {m}")
        if limit is not None and len(chk.mismatches) > limit:
            print(f"      ... {len(chk.mismatches) - limit} more (use -v)")
        if args.verbose:
            for t in chk.tolerated:
                print(f"      tolerated: {t}")
    if fatal_failures(checks):
        print("build failed: cross-check mismatches (see above)", file=sys.stderr)
        return 1

    data = render(rules)
    print(f"hash {rules['build']['hash']}, {len(data):,} bytes")
    if args.check:
        same = args.out.exists() and args.out.read_bytes() == data
        print(f"{args.out}: {'up to date' if same else 'DIFFERS from a fresh build'}")
        return 0 if same else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

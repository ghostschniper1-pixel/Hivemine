"""Agent-level class tables: block, item and entity classes with English phrases (contracts 3).

Usage::

    python -m hivemind.rules.classes [--rules FILE] [--out FILE] [--csv FILE] [--check] [-v]

    from hivemind.rules.classes import load_concepts
    concepts = load_concepts()
    concepts.block_class("iron_ore").phrase        # ('iron', 'ore')
    concepts.block_class_of_state(1)               # the class of vanilla state 1 (stone)
    concepts.item_class_of("oak_log").name         # 'log'
    concepts.entity_class_of("cave_spider").phrase # ('cave', 'spider')
    concepts.class_by_phrase("block", ("hanging", "lantern")).name  # 'other_hanging_lantern'
    concepts.hardness[concepts.block_class("stone").id]   # 1.5 (NumPy arrays for the simulator)

The build reads only ``data/rules-1.21.1.json`` (no network, no raw sources) and writes
``data/concepts-1.21.1.json`` and ``data/block_classes-1.21.1.csv`` (one row per block state).

What a class is. The agent does not see 1,060 blocks and 1,333 items; it sees classes, sets of
vanilla blocks (per state), items or entities that behave the same in v1. Each class has a
phrase: plain English words taken from the 1.21.1 display names (the rules file's names, which
the rules build checks against minecraft-data ``language.json``), lower-cased and split on
spaces and punctuation, with overrides only where a class merges differently named members or
splits one block by state. Phrases keep the game's number ("carrots", "stone bricks", "shears").

How the build guards the classes (it fails on any violation that is not listed in
``EXCEPTIONS`` with a justification and the gameplay approximation it causes; an exception that
no longer matches a violation also fails, so the list stays exact):

- Coverage: every block state, item and entity is in exactly one class.
- Block homogeneity (curated classes): all members share hardness, tool speeds and harvest
  rules for every tool item (so break times stay exact), mineable kinds, needs_tier,
  transparency, light, the liquid/climbable/hazard/gravity/replaceable flags; and per state the
  collision height class, the solid flag and the drops. Drops are the vanilla loot table
  specialised to the state (``block_state_property`` conditions resolved) and to a v1 player
  break (no Fortune or Silk Touch, DESIGN.md 4.2; no explosion), with item names mapped to item
  classes, the block's own name mapped to ``self``, and ``random_sequence`` dropped.
- Item homogeneity (v1 classes): stack size, durability, food, tool, attack damage and speed,
  armor, fuel ticks, equipment slot and the block class an item places.
- Recipe roles: vanilla recipes whose result is a v1 class are mapped to class level (each
  ingredient slot becomes the set of v1 classes it accepts; non-v1 alternatives are dropped).
  Recipes with the same class-level form are a family, and the family must accept *every*
  combination of class members (e.g. ``planks <- log`` holds for every log, because each wood
  type has its own recipe). Correlated slots fail this check: a bed needs three wool of one
  colour, which the class ``wool`` cannot express.
- Phrases: unique among the curated classes of each kind (``none``, air and buried included),
  so ``phrase`` is invertible for everything a command can name (contracts 4.4); no block-class
  phrase equals an entity-class phrase (they meet in ``go to the nearest {block}`` vs
  ``go to the nearest {entity}``). Fallback phrases must be true of every member, because their
  words are shared with curated classes and the agent learns word meanings from them; they may
  repeat (fallbacks never appear in commands) but never equal a curated phrase.

Fallbacks. Block states that are not curated fall into classes keyed by the exact break
inputs, the collision height class, the transparent/liquid/climbable/hazard/gravity/replaceable
flags and the v1 item classes they can drop. Keying on everything would need 235 fallback
classes next to 114 curated ones, more than a byte holds, so light, the solid flag and drop details other than
which v1 items can drop may vary inside a fallback class; each such class is listed in
``exceptions`` with the value it takes (most common light, solid only if all states are full
cubes, the most common drop table) and, where the v1 drop counts differ by state, every
variant (e.g. 2-5 snowballs for snow layers). A fallback's phrase is, in this order: a listed
hint (``FALLBACK_PHRASE_HINTS``: "infested block", and state words for blocks split by state:
"hanging lantern", "open fence gate"); the words all members' names end with (the whole name
for a single member: "beacon"; "glazed terracotta", "carpet", "slab"), prefixed by ``other``
if a curated class already has those words; else ``other <tool> block`` (or ``other block``).
A member's name is never used for a class it does not describe: naming coal blocks "bell"
would teach the agent a wrong meaning. Items outside v1 fall into ``other_item_{1,16,64}`` keyed by stack
size only, with distinct phrases. Entities outside the curated list fall into ``monster``,
``animal`` or ``object`` by minecraft-data's entity type and category; where the two disagree,
``ENTITY_KIND_OVERRIDES`` decides (skeleton horse: animal, evoker fangs: object). Fallbacks
are never resources and never v1 items.

Representation choices not fixed by the contract (all additive):

- ``collision_height`` is a height class: 0 (no collision), 0.5 (anything up to the 0.6-block
  step height: carpet, slab, bed, snow layers), 1 (up to a full block: also chests, farmland,
  top slabs), 1.5 (fences and walls, cannot be jumped). ``solid`` means every state's collision
  box is exactly the full cube.
- Block classes carry ``state_ranges`` (inclusive state-id ranges), ``tool_speeds`` (v1 tool
  class -> speed where it differs from 1), ``harvest_tools`` (v1 item classes that get drops;
  null when no tool is needed), and ``drops`` = ``{source, state, table, items}``: the class-level
  loot table (item names replaced by item-class names) evaluated at ``state``.
- The file also lists the class-level ``recipes`` (the recipe book the checks verified) and the
  ``exceptions`` with their approximations, plus ``rules_hash`` and ``csv_sha256``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from hivemind.rules.loader import GAME_VERSION, Entity, Rules, load_rules

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONCEPTS_PATH = REPO_ROOT / "data" / f"concepts-{GAME_VERSION}.json"
DEFAULT_CSV_PATH = REPO_ROOT / "data" / f"block_classes-{GAME_VERSION}.csv"

AIR_ID = 0
BURIED_ID = 255
MAX_BLOCK_CLASS_ID = 254  # ids 1..254 for curated and fallback classes; 255 is "buried"
NONE_ITEM_ID = 0
NONE_ENTITY_ID = 0

STEP_HEIGHT = 0.6  # [mem] player step height: a collision top up to this is walked over
COLLISION_CLASSES = (0.0, 0.5, 1.0, 1.5)
TIER_LEVEL = {None: 0, "stone": 1, "iron": 2, "diamond": 3}
MINEABLE_BITS = {"axe": 1, "hoe": 2, "pickaxe": 4, "shovel": 8}
RECIPE_TYPES = ("crafting_shaped", "crafting_shapeless", "smelting", "blasting", "smoking",
                "campfire_cooking", "stonecutting")

COLOURS = ("white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
           "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black")
# The eight overworld trees with logs (bamboo has a "block", crimson and warped have "stems").
WOODS = ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry")
WOOD_SETS = ("bamboo", "crimson", "warped")  # wood sets without logs, for doors and fences

# Block flags that the data dumps do not carry. DESIGN.md 4.4 lists the hazards lava, fire,
# magma, cactus, cobweb, powder snow and sweet berry bush; the rest are [mem].
HAZARD_BLOCKS = frozenset({
    "lava", "fire", "magma_block", "cactus", "cobweb", "powder_snow", "sweet_berry_bush",  # DESIGN.md 4.4
    "soul_fire", "campfire", "soul_campfire", "wither_rose",  # [mem] damage on contact
})
GRAVITY_BLOCKS = frozenset({
    "sand", "red_sand", "gravel", "suspicious_sand", "suspicious_gravel",  # [mem]
    "anvil", "chipped_anvil", "damaged_anvil", "dragon_egg",  # [mem]
    *(f"{c}_concrete_powder" for c in COLOURS),  # [mem]
})
# Cells that are always filled with fluid (kelp and seagrass are always waterlogged) [mem].
LIQUID_BLOCKS = frozenset({"water", "lava", "bubble_column", "kelp", "kelp_plant", "seagrass", "tall_seagrass"})

# Fallback block classes key on these (exact); FALLBACK_APPROXIMATED may vary inside one.
FALLBACK_BLOCK_KEYS = ("hardness", "mineable", "requires_tool", "needs_tier", "tool_speeds", "harvest", "transparent",
                       "liquid", "climbable", "hazard", "gravity")
FALLBACK_STATE_KEYS = ("collision_height", "replaceable")
FALLBACK_APPROXIMATED = ("light", "solid", "drops other than which v1 items")


# ---------------------------------------------------------------------------
# Curated classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockSpec:
    """A curated block class: member blocks, optional state filter, optional phrase override."""

    name: str
    members: tuple[str, ...]
    phrase: tuple[str, ...] | None = None
    resource: bool = False
    states: Mapping[str, tuple[str, ...]] | None = None  # property -> allowed values


@dataclass(frozen=True)
class ItemSpec:
    name: str
    members: tuple[str, ...]
    phrase: tuple[str, ...] | None = None


@dataclass(frozen=True)
class EntitySpec:
    name: str
    kind: str  # "hostile" | "passive" | "player" | "item" | "other"
    members: tuple[str, ...]
    phrase: tuple[str, ...] | None = None


def B(name: str, *members: str, phrase: str | None = None, resource: bool = False,
      states: Mapping[str, tuple[str, ...]] | None = None) -> BlockSpec:
    return BlockSpec(name, members or (name,), tuple(phrase.split()) if phrase else None, resource, states)


def I(name: str, *members: str, phrase: str | None = None) -> ItemSpec:
    return ItemSpec(name, members or (name,), tuple(phrase.split()) if phrase else None)


def E(name: str, kind: str, *members: str, phrase: str | None = None) -> EntitySpec:
    return EntitySpec(name, kind, members or (name,), tuple(phrase.split()) if phrase else None)


LOGS = tuple(f"{p}{w}_{k}" for w in WOODS for k in ("log", "wood") for p in ("", "stripped_"))
PLANKS = tuple(f"{w}_planks" for w in (*WOODS, "bamboo"))
SAPLINGS = tuple(f"{w}_sapling" for w in WOODS if w != "mangrove")  # mangrove has a propagule
SMALL_FLOWERS = ("dandelion", "poppy", "blue_orchid", "allium", "azure_bluet", "red_tulip", "orange_tulip",
                 "white_tulip", "pink_tulip", "oxeye_daisy", "cornflower", "lily_of_the_valley", "torchflower")
WOODEN_DOORS = tuple(f"{w}_door" for w in (*WOODS, *WOOD_SETS))
WOODEN_FENCES = tuple(f"{w}_fence" for w in (*WOODS, *WOOD_SETS))

# Block classes in id order (air is 0, fallbacks follow, buried is 255). resource=True marks
# useful mine/goto targets (the notebook's resource memory).
BLOCK_SPECS: tuple[BlockSpec, ...] = (
    B("air", "air", "cave_air", "void_air", phrase="air"),
    # stone and deep terrain
    B("stone", resource=True),
    B("deepslate", resource=True),
    B("granite", resource=True),
    B("diorite", resource=True),
    B("andesite", resource=True),
    B("tuff", resource=True),
    B("calcite", resource=True),
    B("cobblestone"),
    B("cobbled_deepslate"),
    B("mossy_cobblestone"),
    B("stone_bricks", "stone_bricks", "mossy_stone_bricks", "cracked_stone_bricks", "chiseled_stone_bricks",
      phrase="stone bricks"),
    B("dripstone_block"),
    B("pointed_dripstone"),
    B("obsidian", resource=True),
    B("bedrock"),
    B("barrier"),
    B("magma_block"),
    # soil
    B("dirt", resource=True),
    B("coarse_dirt"),
    B("rooted_dirt"),
    B("podzol"),
    B("mycelium"),
    B("grass_block", resource=True),
    B("dirt_path"),
    B("farmland"),
    B("mud"),
    B("clay", resource=True),
    B("moss_block"),
    B("moss_carpet"),
    # sand and desert
    B("sand", resource=True),
    B("red_sand", resource=True),
    B("gravel", resource=True),
    B("sandstone", resource=True),
    B("red_sandstone", resource=True),
    B("terracotta", "terracotta", *(f"{c}_terracotta" for c in COLOURS), phrase="terracotta"),
    # snow and ice
    B("snow", states={"layers": ("1",)}),
    B("snow_block"),
    B("powder_snow"),
    B("ice"),
    B("packed_ice"),
    B("blue_ice"),
    # fluids
    B("water"),
    B("lava"),
    B("bubble_column"),
    # trees
    B("log", *LOGS, phrase="log", resource=True),
    B("planks", *PLANKS, phrase="planks"),
    B("leaves", "spruce_leaves", "birch_leaves", "acacia_leaves", "cherry_leaves", phrase="leaves", resource=True),
    B("oak_leaves", "oak_leaves", "dark_oak_leaves", phrase="oak leaves", resource=True),
    B("jungle_leaves"),
    B("mangrove_leaves"),
    B("azalea_leaves", "azalea_leaves", "flowering_azalea_leaves", phrase="azalea leaves"),
    B("sapling", *SAPLINGS, phrase="sapling"),
    B("mangrove_roots"),
    B("bamboo"),
    # plants
    B("short_grass", resource=True),
    B("tall_grass"),
    B("fern"),
    B("large_fern"),
    B("dead_bush"),
    B("flower", *SMALL_FLOWERS, phrase="flower"),
    B("tall_flower", "sunflower", "lilac", "rose_bush", "peony", phrase="tall flower"),
    B("wither_rose"),
    B("brown_mushroom"),
    B("red_mushroom"),
    B("sugar_cane", resource=True),
    B("cactus"),
    B("pumpkin", resource=True),
    B("melon", resource=True),
    B("sweet_berry_bush", resource=True),
    B("vine"),
    B("lily_pad"),
    B("azalea", "azalea", "flowering_azalea", phrase="azalea"),
    B("kelp", "kelp", "kelp_plant", phrase="kelp"),
    B("seagrass"),
    B("tall_seagrass"),
    # crops (village farms)
    B("wheat", resource=True),
    B("carrots"),
    B("potatoes"),
    B("beetroots"),
    # ores: stone and deepslate variants differ in hardness (3 vs 4.5)
    B("coal_ore", resource=True),
    B("deepslate_coal_ore", resource=True),
    B("iron_ore", resource=True),
    B("deepslate_iron_ore", resource=True),
    B("copper_ore", resource=True),
    B("deepslate_copper_ore", resource=True),
    B("gold_ore", resource=True),
    B("deepslate_gold_ore", resource=True),
    B("redstone_ore", resource=True),
    B("deepslate_redstone_ore", resource=True),
    B("lapis_ore", resource=True),
    B("deepslate_lapis_ore", resource=True),
    B("diamond_ore", resource=True),
    B("deepslate_diamond_ore", resource=True),
    B("emerald_ore", resource=True),
    B("deepslate_emerald_ore", resource=True),
    B("raw_iron_block", resource=True),
    B("raw_copper_block"),
    # placed blocks and early structures
    B("crafting_table"),
    B("furnace"),
    B("chest"),
    B("bed", *(f"{c}_bed" for c in COLOURS), phrase="bed"),
    B("torch", "torch", "wall_torch"),
    B("ladder"),
    B("door", *WOODEN_DOORS, phrase="door"),
    B("fence", *WOODEN_FENCES, phrase="fence"),
    B("glass", "glass", *(f"{c}_stained_glass" for c in COLOURS), phrase="glass"),
    B("glass_pane", "glass_pane", *(f"{c}_stained_glass_pane" for c in COLOURS), phrase="glass pane"),
    B("wool", *(f"{c}_wool" for c in COLOURS), phrase="wool"),
    B("cobweb", resource=True),
    B("spawner"),
    B("rail", "rail", "powered_rail", "detector_rail", "activator_rail", phrase="rail"),
    B("fire"),
    B("tnt", states={"unstable": ("false",)}),  # unstable TNT (no drops) falls back
)

# v1 item classes in id order (0 is "none"; non-v1 fallbacks follow).
ITEM_SPECS: tuple[ItemSpec, ...] = (
    # wood
    I("log", *LOGS, phrase="log"),
    I("planks", *PLANKS, phrase="planks"),
    I("stick"),
    I("sapling", *SAPLINGS, phrase="sapling"),
    I("crafting_table"),
    I("bowl"),
    I("ladder"),
    I("chest"),
    I("furnace"),
    I("torch"),
    # tools and weapons
    *(I(f"{tier}_{kind}") for tier in ("wooden", "stone", "iron", "golden", "diamond")
      for kind in ("pickaxe", "axe", "shovel", "hoe", "sword")),
    I("shears"),
    I("bow"),
    I("arrow"),
    I("shield"),
    I("flint_and_steel"),
    I("bucket"),
    I("water_bucket"),
    I("lava_bucket"),
    I("milk_bucket"),
    # armor
    *(I(f"{material}_{piece}") for material in ("leather", "golden", "chainmail", "iron", "diamond")
      for piece in ("helmet", "chestplate", "leggings", "boots")),
    # stone-type and terrain blocks
    I("cobblestone"),
    I("cobbled_deepslate"),
    I("stone"),
    I("granite"),
    I("diorite"),
    I("andesite"),
    I("tuff"),
    I("calcite"),
    I("obsidian"),
    I("dirt"),
    I("sand"),
    I("red_sand"),
    I("gravel"),
    I("sandstone"),
    I("red_sandstone"),
    I("glass"),
    I("glass_bottle"),
    I("flint"),
    I("clay_ball"),
    I("snowball"),
    I("snow_block"),
    # ores and metals
    I("coal"),
    I("charcoal"),
    I("raw_iron"),
    I("raw_gold"),
    I("raw_copper"),
    I("iron_ingot"),
    I("gold_ingot"),
    I("copper_ingot"),
    I("iron_nugget"),
    I("gold_nugget"),
    I("diamond"),
    I("emerald"),
    I("redstone"),
    I("lapis_lazuli"),
    I("raw_iron_block"),
    I("raw_copper_block"),
    # mob drops and animal products
    I("rotten_flesh"),
    I("bone"),
    I("bone_meal"),
    I("string"),
    I("spider_eye"),
    I("gunpowder"),
    I("ender_pearl"),
    I("slime_ball"),
    I("feather"),
    I("egg"),
    I("leather"),
    I("rabbit_hide"),
    I("rabbit_foot"),
    I("ink_sac"),
    I("phantom_membrane"),
    I("glowstone_dust"),
    I("wool", *(f"{c}_wool" for c in COLOURS), phrase="wool"),
    I("bed", *(f"{c}_bed" for c in COLOURS), phrase="bed"),
    # food and farming
    I("apple"),
    I("bread"),
    I("wheat"),
    I("wheat_seeds"),
    I("carrot"),
    I("potato"),
    I("baked_potato"),
    I("poisonous_potato"),
    I("beetroot"),
    I("beetroot_seeds"),
    I("melon_slice"),
    I("melon_seeds"),
    I("pumpkin"),
    I("pumpkin_seeds"),
    I("sweet_berries"),
    I("glow_berries"),
    I("brown_mushroom"),
    I("red_mushroom"),
    I("mushroom_stew"),
    I("sugar_cane"),
    I("sugar"),
    I("paper"),
    I("kelp"),
    I("dried_kelp"),
    I("beef"),
    I("cooked_beef"),
    I("porkchop"),
    I("cooked_porkchop"),
    I("chicken"),
    I("cooked_chicken"),
    I("mutton"),
    I("cooked_mutton"),
    I("rabbit"),
    I("cooked_rabbit"),
    I("cod"),
    I("cooked_cod"),
    I("salmon"),
    I("cooked_salmon"),
    I("tropical_fish"),
    I("pufferfish"),
)

ENTITY_SPECS: tuple[EntitySpec, ...] = (
    E("player", "player"),
    E("item", "item", phrase="dropped item"),
    E("experience_orb", "other"),
    E("arrow", "other"),
    E("tnt", "other"),  # "Primed TNT": about to explode
    # hostile
    E("zombie", "hostile"),
    E("husk", "hostile"),
    E("drowned", "hostile"),
    E("zombie_villager", "hostile"),
    E("skeleton", "hostile"),
    E("stray", "hostile"),
    E("bogged", "hostile"),
    E("creeper", "hostile"),
    E("spider", "hostile"),
    E("cave_spider", "hostile"),
    E("enderman", "hostile"),
    E("witch", "hostile"),
    E("slime", "hostile"),
    E("phantom", "hostile"),
    E("silverfish", "hostile"),
    E("pillager", "hostile"),
    E("vindicator", "hostile"),
    E("evoker", "hostile"),
    E("vex", "hostile"),
    E("ravager", "hostile"),
    E("guardian", "hostile"),
    E("elder_guardian", "hostile"),
    E("warden", "hostile"),
    E("breeze", "hostile"),
    E("zombified_piglin", "hostile"),
    # passive (neutral mobs such as wolves and iron golems included)
    E("cow", "passive"),
    E("pig", "passive"),
    E("sheep", "passive"),
    E("chicken", "passive"),
    E("rabbit", "passive"),
    E("horse", "passive"),
    E("donkey", "passive"),
    E("mule", "passive"),
    E("llama", "passive"),
    E("wolf", "passive"),
    E("fox", "passive"),
    E("cat", "passive"),
    E("parrot", "passive"),
    E("goat", "passive"),
    E("bee", "passive"),
    E("turtle", "passive"),
    E("frog", "passive"),
    E("axolotl", "passive"),
    E("armadillo", "passive"),
    E("polar_bear", "passive"),
    E("panda", "passive"),
    E("mooshroom", "passive"),
    E("villager", "passive"),
    E("wandering_trader", "passive"),
    E("iron_golem", "passive"),
    E("snow_golem", "passive"),
    E("bat", "passive"),
    E("squid", "passive"),
    E("glow_squid", "passive"),
    E("dolphin", "passive"),
    E("cod", "passive"),
    E("salmon", "passive"),
    E("tropical_fish", "passive"),
    E("pufferfish", "passive"),
)

# Fallback entity classes by kind (names, phrases).
ENTITY_FALLBACKS = (("monster", "hostile", ("monster",)), ("animal", "passive", ("animal",)),
                    ("object", "other", ("object",)))
# The kind of a fallback entity comes from minecraft-data 1.20.5's entity type and category.
# Where both say something and disagree, ENTITY_KIND_OVERRIDES must decide (the build fails
# otherwise, and on an override that no longer settles a disagreement).
ENTITY_TYPE_KIND: dict[str, str] = {  # "mob" and missing types say nothing
    "hostile": "hostile", "animal": "passive", "passive": "passive", "water_creature": "passive",
    "ambient": "passive", "player": "other", "projectile": "other", "other": "other", "living": "other",
}
ENTITY_CATEGORY_KIND: dict[str, str] = {  # missing categories say nothing
    "Hostile mobs": "hostile", "Passive mobs": "passive", "Projectiles": "other", "Immobile": "other",
    "Vehicles": "other", "UNKNOWN": "other",
}
ENTITY_KIND_OVERRIDES: dict[str, str] = {  # all listed under "Hostile mobs"
    "skeleton_horse": "passive",  # [mem] type animal: passive (also when it brings skeleton riders)
    "zombie_horse": "passive",  # [mem] type animal: passive
    "evoker_fangs": "other",  # [mem] type other: a spell effect, not a mob
    "hoglin": "hostile",  # [mem] type animal (it can be bred) but attacks players
}

# Phrase hints for fallback classes (see the module docstring), as (member block, fallback-key
# values the class must have, phrase). For each member in state-id order, that block's hints are
# tried in table order; the first match wins. The key values let a block that is split by state
# name each part. Every hint must name some class (a stale hint fails the build).
FALLBACK_PHRASE_HINTS: tuple[tuple[str, Mapping[str, Any], str], ...] = (
    # blocks split by state (collision class or drops)
    ("snow", {"collision_height": 0.5}, "snow layers"),  # 2-5 layers (1 layer is the curated "snow")
    ("snow", {"collision_height": 1.0}, "deep snow"),  # 6-8 layers: a full step up
    ("lantern", {"collision_height": 1.0}, "hanging lantern"),
    ("lantern", {"collision_height": 0.5}, "lantern"),
    ("oak_fence_gate", {"collision_height": 0.0}, "open fence gate"),
    ("oak_fence_gate", {"collision_height": 1.5}, "fence gate"),
    ("oak_trapdoor", {"collision_height": 0.5}, "trapdoor"),  # closed in the bottom half
    ("oak_trapdoor", {"collision_height": 1.0}, "raised trapdoor"),  # open, or in the top half
    ("big_dripleaf", {"collision_height": 1.0}, "big dripleaf"),
    ("big_dripleaf", {"collision_height": 0.0}, "tilted big dripleaf"),  # fully tilted: no collision
    ("amethyst_cluster", {"collision_height": 0.5}, "amethyst cluster"),  # clusters and buds facing up
    ("amethyst_cluster", {"collision_height": 1.0}, "hanging amethyst cluster"),  # on walls and ceilings
    ("cave_vines", {"v1_drops": ["glow_berries"]}, "cave vines with glow berries"),
    # (cave vines without berries share a class with weeping and twisting vines: a generic phrase)
    ("candle_cake", {}, "cake with candle"),  # names end "... candle", but the thing is a cake
    ("tnt", {}, "unstable tnt"),
    # members without a common name, or whose typical member would mislead
    ("infested_stone", {}, "infested block"),
    ("suspicious_sand", {}, "suspicious block"),
    ("potted_oak_sapling", {}, "potted sapling"),
    ("mangrove_propagule", {}, "small plant"),  # propagule, nether wart, fungi, spore blossom, small dripleaf
    ("flower_pot", {}, "flower pot"),  # empty and potted plants without v1 drops
    ("skeleton_skull", {}, "skull"),  # mob heads and skulls on the ground
    ("skeleton_wall_skull", {}, "wall skull"),
)
FALLBACK_JUSTIFICATION = ("block ids must fit one byte (contracts 3); fallback classes key on exact break "
                          "inputs, collision class, nav and perception flags and the v1 items they can drop")
ITEM_FALLBACK_PHRASES = {1: ("other", "unstackable", "item"), 16: ("other", "small", "stack", "item"),
                         64: ("other", "item")}


# ---------------------------------------------------------------------------
# Exceptions: each tolerated violation needs a justification and the approximation it causes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Exception_:
    kind: str  # "recipe" | "drops" | "block" | "item"
    cls: str  # class name
    check: str  # the check or field that differs
    justification: str
    approximation: str
    drop_state: Mapping[str, str] | None = None  # for "drops": property values the class drops as
    keep_recipe: bool = False  # for "recipe": keep the family in the class-level book anyway
    two_cell: bool = False  # for "drops": a two-cell object (location checks on the other half hold)


TWO_CELL = ("two-cell object: one half drops nothing and breaking either half removes both",
            ("the class drops once per object (as the half named in drop_state); the simulator must "
             "break both cells together, so a loot condition that tests the other half (tall grass, "
             "large fern) is taken as true"))
CROP_AGE = ("crop age is not tracked by the class (voxels hold class ids only); crops do not grow in v1",
            "a crop drops as a fully grown one (drop_state); in the real game unripe crops give only seeds")

EXCEPTIONS: tuple[Exception_, ...] = (
    Exception_("recipe", "bed", "correlated slots",
               "each bed recipe needs three wool of one colour (and dyeing recipes need dyes, not v1)",
               "the simulator crafts a bed from any three wool; in the real game the three wool must "
               "share a colour (sim-to-real check in M7)", keep_recipe=True),
    Exception_("recipe", "log", "correlated slots",
               "wood and stripped-wood recipes turn four logs of one wood type into three logs of the "
               "same class",
               "these recipes are left out of the class-level recipe book (they only lose material)"),
    Exception_("drops", "bed", "state-dependent drops", *TWO_CELL, drop_state={"part": "head"}, two_cell=True),
    Exception_("drops", "door", "state-dependent drops", *TWO_CELL, drop_state={"half": "lower"}, two_cell=True),
    Exception_("drops", "tall_grass", "state-dependent drops", *TWO_CELL, drop_state={"half": "lower"},
               two_cell=True),
    Exception_("drops", "large_fern", "state-dependent drops", *TWO_CELL, drop_state={"half": "lower"},
               two_cell=True),
    Exception_("drops", "tall_flower", "state-dependent drops", *TWO_CELL, drop_state={"half": "lower"},
               two_cell=True),
    Exception_("drops", "wheat", "state-dependent drops", *CROP_AGE, drop_state={"age": "7"}),
    Exception_("drops", "carrots", "state-dependent drops", *CROP_AGE, drop_state={"age": "7"}),
    Exception_("drops", "potatoes", "state-dependent drops", *CROP_AGE, drop_state={"age": "7"}),
    Exception_("drops", "beetroots", "state-dependent drops", *CROP_AGE, drop_state={"age": "3"}),
    Exception_("drops", "sweet_berry_bush", "state-dependent drops",
               "berry bush age is not tracked by the class; bushes do not grow in v1",
               "a bush drops as a ripe one (age 3: 2-3 berries); in the real game younger bushes give "
               "fewer or none", drop_state={"age": "3"}),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class BuildError(Exception):
    pass


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, UTF-8 text (as the rules file)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def concepts_hash(concepts: dict) -> str:
    """sha256 of the canonical JSON of ``concepts`` without its ``hash`` field."""
    body = {k: v for k, v in concepts.items() if k != "hash"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def words(display_name: str) -> tuple[str, ...]:
    """Display name -> phrase: lower-case words split on spaces and punctuation.

    "Jack o'Lantern" -> ("jack", "o", "lantern"). A possessive 's is dropped so the owner's word
    is reused: "Rabbit's Foot" -> ("rabbit", "foot").
    """
    return tuple(re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", display_name.lower())))


def _ns(name: str) -> str:
    return name.removeprefix("minecraft:")


def collision_class(boxes: Sequence[Sequence[float]]) -> float:
    """Height class of a collision shape (see the module docstring)."""
    if not boxes:
        return 0.0
    top = max(b[4] for b in boxes)
    if top <= STEP_HEIGHT + 1e-9:
        return 0.5
    if top <= 1.0 + 1e-9:
        return 1.0
    return 1.5


def is_full_cube(boxes: Sequence[Sequence[float]]) -> bool:
    return len(boxes) == 1 and tuple(boxes[0]) == (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)


def fallback_entity_kind(e: Entity) -> tuple[str, str | None]:
    """(kind, problem) of an entity outside the curated list: kind is "hostile", "passive" or
    "other", from minecraft-data's type and category or ENTITY_KIND_OVERRIDES; problem is a
    message when the type and category disagree and no override decides."""
    by_type = ENTITY_TYPE_KIND.get(e.type or "")
    by_category = ENTITY_CATEGORY_KIND.get(e.category or "")
    if e.name in ENTITY_KIND_OVERRIDES:
        return ENTITY_KIND_OVERRIDES[e.name], None
    if by_type and by_category and by_type != by_category:
        return by_type, (f"type {e.type!r} says {by_type}, category {e.category!r} says {by_category}; "
                         "add it to ENTITY_KIND_OVERRIDES")
    return by_type or by_category or "other", None


def _ranges(ids: Iterable[int]) -> list[list[int]]:
    """Sorted ints -> inclusive [lo, hi] runs."""
    out: list[list[int]] = []
    for i in sorted(ids):
        if out and out[-1][1] == i - 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


# ---------------------------------------------------------------------------
# Loot tables: specialise to a block state, map items to classes
# ---------------------------------------------------------------------------


def _state_matches(props: Mapping[str, Any], state: Mapping[str, str]) -> bool | None:
    """Whether state values match a block_state_property test; None when undecidable."""
    result: bool | None = True
    for prop, want in props.items():
        if prop not in state:
            result = None
            continue
        have = state[prop]
        if isinstance(want, dict):  # {"min": .., "max": ..}
            lo, hi = want.get("min"), want.get("max")
            try:
                v = int(have)
            except ValueError:
                result = None
                continue
            if (lo is not None and v < int(lo)) or (hi is not None and v > int(hi)):
                return False
        elif str(want) != have:
            return False
    return result


def _is_enchantment_predicate(pred: Any) -> bool:
    """A match_tool predicate that tests only enchantments (Silk Touch)."""
    return isinstance(pred, dict) and set(pred) == {"predicates"} and set(pred["predicates"]) == {
        "minecraft:enchantments"}


def _tests_other_half(cond: Mapping[str, Any], block: str) -> bool:
    """A location_check that tests this same block one cell above or below: the other half of
    a two-cell object."""
    pred = cond.get("predicate")
    if not isinstance(pred, dict) or set(pred) != {"block"} or not isinstance(pred["block"], dict):
        return False
    blocks = pred["block"].get("blocks")
    return (isinstance(blocks, str) and _ns(blocks) == block and cond.get("offsetY") in (1, -1)
            and not cond.get("offsetX") and not cond.get("offsetZ"))


def simplify_condition(cond: Mapping[str, Any], block: str, state: Mapping[str, str],
                       two_cell: bool = False) -> bool | dict:
    """A loot condition specialised to a v1 player break of this block state: True, False, or the
    (partly simplified) condition when it still depends on the tool, chance or surroundings.

    v1 player breaks: no enchantments (DESIGN.md 4.2: no Fortune or Silk Touch), no explosion,
    and the breaking entity exists. ``two_cell``: the block is one half of a two-cell object
    that the simulator breaks as a whole (``TWO_CELL``), so a check that the other half is there
    holds.
    """
    kind = cond.get("condition")
    if kind == "minecraft:block_state_property" and _ns(cond.get("block", "")) == block:
        v = _state_matches(cond.get("properties", {}), state)
        return dict(cond) if v is None else v
    if kind == "minecraft:location_check" and two_cell and _tests_other_half(cond, block):
        return True
    if kind == "minecraft:survives_explosion":
        return True  # no explosion: always survives
    if kind == "minecraft:entity_properties" and cond.get("predicate") == {}:
        return True  # "the breaking entity exists"
    if kind == "minecraft:match_tool" and _is_enchantment_predicate(cond.get("predicate")):
        return False  # no enchanted tools in v1
    if kind == "minecraft:inverted":
        inner = simplify_condition(cond["term"], block, state, two_cell)
        return (not inner) if isinstance(inner, bool) else {"condition": kind, "term": inner}
    if kind in ("minecraft:any_of", "minecraft:all_of"):
        decisive = kind == "minecraft:any_of"  # the value that decides the whole condition
        terms = []
        for t in cond["terms"]:
            v = simplify_condition(t, block, state, two_cell)
            if v is decisive:
                return decisive
            if not isinstance(v, bool):
                terms.append(v)
        if not terms:
            return not decisive
        return terms[0] if len(terms) == 1 else {"condition": kind, "terms": terms}
    return dict(cond)


def _resolve_conditions(node: dict, block: str, state: Mapping[str, str], two_cell: bool = False) -> bool:
    """Simplify ``node['conditions']`` in place; False when the node can never apply."""
    conds = node.get("conditions")
    if not conds:
        return True
    kept = []
    for c in conds:
        v = simplify_condition(c, block, state, two_cell)
        if v is False:
            return False
        if v is not True:
            kept.append(v)
    if kept:
        node["conditions"] = kept
    else:
        del node["conditions"]
    return True


# Loot functions that do nothing in a v1 player break (no explosion, no enchantment, no names).
_IDENTITY_FUNCTIONS = frozenset({"minecraft:explosion_decay", "minecraft:copy_components", "minecraft:copy_name"})
_FORTUNE_IDENTITY_FORMULAS = frozenset({"minecraft:ore_drops", "minecraft:uniform_bonus_count"})


def _is_identity_function(fn: Mapping[str, Any]) -> bool:
    name = fn.get("function")
    if name in _IDENTITY_FUNCTIONS:
        return True
    # Fortune level 0: ore_drops and uniform_bonus_count add nothing (binomial_with_bonus_count
    # still draws ``extra`` trials, so it stays).
    return (name == "minecraft:apply_bonus" and fn.get("enchantment") == "minecraft:fortune"
            and fn.get("formula") in _FORTUNE_IDENTITY_FORMULAS)


def specialise_loot(table: Any, block: str, state: Mapping[str, str], two_cell: bool = False) -> Any:
    """A copy of a loot table specialised to a v1 player break of this block state.

    Resolves ``block_state_property`` conditions for the state; removes what cannot matter in
    v1 (Silk Touch branches, explosion survival and decay, copied block-entity names, Fortune
    bonuses that are identities at level 0); turns Fortune ``table_bonus`` into its level-0
    ``random_chance``. Item tests (shears), random chances and location checks are kept,
    except that with ``two_cell`` a location check for the object's other half (the same block
    one cell above or below) holds, because the simulator breaks both cells together.
    """

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            out = []
            for x in node:
                if isinstance(x, dict):
                    if "function" in x and _is_identity_function(x):
                        continue
                    if x.get("condition") == "minecraft:table_bonus" and x.get("enchantment") == "minecraft:fortune":
                        x = {"chance": x["chances"][0], "condition": "minecraft:random_chance"}
                    if "conditions" in x and ("type" in x or "function" in x or "rolls" in x):
                        x = dict(x)
                        if not _resolve_conditions(x, block, state, two_cell):
                            continue
                out.append(walk(x))
            return out
        if isinstance(node, dict):
            out = {k: walk(v) for k, v in node.items()}
            if out.get("functions") == []:
                del out["functions"]
            return out
        return node

    top = dict(table)
    top.pop("random_sequence", None)
    return simplify_loot(walk(top))


def _simplify_entry(e: dict) -> dict | None:
    """Exact simplifications of one loot entry; None when it can never produce anything."""
    e = dict(e)
    if e.get("type") == "minecraft:alternatives":
        children = []
        for c in e.get("children", []):
            c = _simplify_entry(c)
            if c is None:
                continue
            if c.get("type") == "minecraft:alternatives" and "conditions" not in c:
                children.extend(c["children"])  # nested alternatives flatten
            else:
                children.append(c)
            if "conditions" not in children[-1]:
                break  # an unconditional child always wins: later ones never run
        if not children:
            return None
        if len(children) == 1 and "conditions" not in e:
            return children[0]
        e["children"] = children
        return e
    if e.get("type") == "minecraft:item":
        fns = list(e.get("functions", []))
        if fns and fns[0].get("function") == "minecraft:set_count" and fns[0].get("count") == 1.0 \
                and not fns[0].get("add") and "conditions" not in fns[0]:
            fns = fns[1:]  # setting the default count first does nothing
        if fns:
            e["functions"] = fns
        else:
            e.pop("functions", None)
    return e


def simplify_loot(table: dict) -> dict | None:
    """Drop empty pools and entries; None when the table can never drop anything."""
    pools = []
    for pool in table.get("pools", []):
        entries = [x for x in (_simplify_entry(e) for e in pool.get("entries", [])) if x is not None]
        if entries:
            pools.append({**pool, "entries": entries})
    if not pools:
        return None
    return {**table, "pools": pools}


def normalise_loot(table: Any, block: str, item_class: Mapping[str, str],
                   item_tags: Mapping[str, Sequence[str]] | None = None) -> Any:
    """Map item names (and item tags in tool predicates) to item-class names, the block's own
    name to ``self``."""

    def classes(name: str) -> list[str]:
        if name.startswith("#"):
            if item_tags is None:
                raise BuildError(f"item tag {name} in a loot table needs item_tags")
            return sorted({item_class[m] for m in item_tags[_ns(name[1:])]})
        return [item_class[_ns(name)]]

    def item(name: str) -> str:
        return item_class[_ns(name)]

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k == "name" and node.get("type") == "minecraft:item":
                    out[k] = item(v)
                elif k == "items" and isinstance(v, (str, list)):  # match_tool predicate
                    names = [v] if isinstance(v, str) else v
                    out[k] = sorted({c for n in names for c in classes(n)})
                elif k in ("block", "blocks") and isinstance(v, str) and _ns(v) == block:
                    out[k] = "self"
                else:
                    out[k] = walk(v)
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(table)


def loot_items(table: Any) -> set[str]:
    """Item-class names a normalised table can produce."""
    out: set[str] = set()
    if isinstance(table, dict):
        if table.get("type") == "minecraft:item" and "name" in table:
            out.add(table["name"])
        for v in table.values():
            out |= loot_items(v)
    elif isinstance(table, list):
        for v in table:
            out |= loot_items(v)
    return out


def _count_text(count: Any) -> str:
    """A set_count number provider as short text: ``2``, ``1-3``, ``B(3, 0.533)``."""
    if isinstance(count, (int, float)):
        return f"{count:g}"
    kind = _ns(count.get("type", "")) if isinstance(count, dict) else ""
    try:
        if kind == "uniform":
            return f"{count['min']:g}-{count['max']:g}"
        if kind == "binomial":
            return f"B({count['n']:g}, {count['p']:.3g})"
    except (KeyError, TypeError, ValueError):  # nested number providers
        pass
    return canonical_json(count)


def _condition_text(cond: Mapping[str, Any]) -> str:
    kind = _ns(cond.get("condition", ""))
    if kind == "random_chance":
        return f"p {cond['chance']:.3g}"
    if kind == "match_tool":
        return "with " + "/".join(cond.get("predicate", {}).get("items", ["tool"]))
    return kind


def v1_drop_terms(table: Any, is_v1: Callable[[str], bool]) -> list[str]:
    """How a normalised table produces v1 item classes: one short term per item entry, such as
    ``snowball x3``, ``melon_seeds xB(3, 0.533)`` or ``wheat_seeds p 0.125``, with the
    conditions of the entry and its enclosing entries and pool. For reports only."""
    out: list[str] = []

    def walk(node: Any, conds: list[Mapping[str, Any]]) -> None:
        if isinstance(node, list):
            for x in node:
                walk(x, conds)
            return
        if not isinstance(node, dict):
            return
        here = [*conds, *node.get("conditions", ())] if ("rolls" in node or "type" in node) else conds
        if node.get("type") == "minecraft:item" and is_v1(node.get("name", "")):
            parts = [node["name"]]
            for fn in node.get("functions", ()):
                if fn.get("function") == "minecraft:set_count":
                    parts.append(("+" if fn.get("add") else "x") + _count_text(fn["count"]))
                else:
                    parts.append(_ns(fn.get("function", "")))
                parts += [f"({_condition_text(c)})" for c in fn.get("conditions", ())]
            parts += [_condition_text(c) for c in here]
            out.append(" ".join(parts))
        for k, v in node.items():
            if k != "conditions":
                walk(v, here)

    walk(table, [])
    return sorted(out)


def loot_state_properties(table: Any, block: str) -> set[str]:
    """Properties of this block that the table's conditions test."""
    out: set[str] = set()
    if isinstance(table, dict):
        if table.get("condition") == "minecraft:block_state_property" and _ns(table.get("block", "")) == block:
            out |= set(table.get("properties", {}))
        for v in table.values():
            out |= loot_state_properties(v, block)
    elif isinstance(table, list):
        for v in table:
            out |= loot_state_properties(v, block)
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    kind: str
    cls: str
    check: str
    detail: str


@dataclass
class _Fallback:
    """A fallback block class while it is being named: its key (``fallback_key``), its state ids
    and its member blocks in state-id order."""

    key: dict
    states: list[int]
    members: list[str]


def _common_tail(phrases: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    """The words all phrases end with (empty if their last words differ)."""
    tail: list[str] = []
    for i in range(1, min(len(p) for p in phrases) + 1):
        if len({p[-i] for p in phrases}) != 1:
            break
        tail.insert(0, phrases[0][-i])
    return tuple(tail)


@dataclass
class BuildResult:
    concepts: dict
    csv_text: str
    violations: list[Violation] = field(default_factory=list)  # unexcused: the build fails
    tolerated: list[tuple[Exception_, list[str]]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


class _Builder:
    """One build of the class tables from a Rules object."""

    def __init__(self, rules: Rules, block_specs: Sequence[BlockSpec], item_specs: Sequence[ItemSpec],
                 entity_specs: Sequence[EntitySpec], exceptions: Sequence[Exception_]):
        self.r = rules
        self.block_specs = tuple(block_specs)
        self.item_specs = tuple(item_specs)
        self.entity_specs = tuple(entity_specs)
        self.exceptions = tuple(exceptions)
        self.violations: list[Violation] = []
        self.notes: list[str] = []
        self.tool_items = sorted(n for n, it in rules.items.items() if it.tool is not None)
        self.climbable = frozenset(rules.tag("block", "climbable"))
        self.replaceable = frozenset(rules.tag("block", "replaceable"))
        self.walls = frozenset(rules.tag("block", "walls"))
        self._block_props: dict[str, dict] = {}
        self._loot_props: dict[str, tuple[str, ...]] = {}
        self._drop_info_cache: dict[int, tuple[str, list[str], Any]] = {}  # keeps the table alive
        self._drops_cache: dict[tuple[str, tuple[str, ...], bool], Any] = {}

    def fail(self, kind: str, cls: str, check: str, detail: str) -> None:
        self.violations.append(Violation(kind, cls, check, detail))

    # -- items ------------------------------------------------------------------

    def assign_items(self) -> None:
        r = self.r
        self.item_class: dict[str, str] = {"air": "none"}
        self.item_members: dict[str, list[str]] = {"none": ["air"]}
        self.item_order: list[str] = ["none"]
        self.item_v1: dict[str, bool] = {"none": False}
        self.item_phrase: dict[str, tuple[str, ...]] = {"none": ("nothing",)}
        for spec in self.item_specs:
            if spec.name in self.item_members:
                raise BuildError(f"duplicate item class {spec.name}")
            for m in spec.members:
                if m not in r.items:
                    raise BuildError(f"item class {spec.name}: unknown item {m}")
                if m in self.item_class:
                    raise BuildError(f"item {m} is in classes {self.item_class[m]} and {spec.name}")
                self.item_class[m] = spec.name
            self.item_members[spec.name] = sorted(spec.members)
            self.item_order.append(spec.name)
            self.item_v1[spec.name] = True
            names = [r.items[m].display_name for m in spec.members]
            self.item_phrase[spec.name] = spec.phrase or self._single_phrase("item", spec.name, names)
        fallback: dict[int, list[str]] = defaultdict(list)
        for name in sorted(r.items):
            if name not in self.item_class:
                fallback[r.items[name].stack_size].append(name)
        for size in sorted(fallback):
            cname = f"other_item_{size}"
            if size not in ITEM_FALLBACK_PHRASES:
                raise BuildError(f"no fallback phrase for stack size {size}")
            for m in fallback[size]:
                self.item_class[m] = cname
            self.item_members[cname] = fallback[size]
            self.item_order.append(cname)
            self.item_v1[cname] = False
            self.item_phrase[cname] = ITEM_FALLBACK_PHRASES[size]
        self.item_id = {n: i for i, n in enumerate(self.item_order)}

    def _single_phrase(self, kind: str, cls: str, names: Sequence[str]) -> tuple[str, ...]:
        """The phrase of a class without an override: all members must share one display name."""
        phrases = {words(n) for n in names}
        if len(phrases) != 1:
            self.fail(kind, cls, "phrase", f"merged members have different names {sorted(set(names))}; "
                                           "give the class a phrase override")
        return min(phrases)

    # -- blocks -----------------------------------------------------------------

    def block_props(self, name: str) -> dict:
        """Block-level properties that every member of a class must share (cached)."""
        if name in self._block_props:
            return self._block_props[name]
        b = self.r.blocks[name]
        props = {
            "hardness": b.hardness,
            "mineable": list(b.mineable),
            "requires_tool": b.requires_tool,
            "needs_tier": b.needs_tier,
            "tool_speeds": {t: s for t in self.tool_items if (s := self.r.mining_speed(t, name)) != 1.0},
            "harvest": (sorted(t for t in self.tool_items if self.r.can_harvest(t, name))
                        if b.requires_tool else None),
            "transparent": b.transparent,
            "light": b.emit_light,
            "liquid": name in LIQUID_BLOCKS,
            "climbable": name in self.climbable,
            "hazard": name in HAZARD_BLOCKS,
            "gravity": name in GRAVITY_BLOCKS,
        }
        self._block_props[name] = props
        return props

    def state_props(self, name: str, state: int, drop_state: Mapping[str, str] | None = None,
                    two_cell: bool = False) -> dict:
        """State-level properties: collision class, solid, replaceable and specialised drops."""
        b = self.r.blocks[name]
        boxes = b.collision_boxes(state)
        values = b.state_values(state)
        height = collision_class(boxes)
        if not boxes and name in self.walls and values.get("up") == "false":
            height = 1.5  # unreachable post-less, side-less wall state: keeps the wall's class (APPROXIMATIONS)
        replaceable = name in self.replaceable and not (name == "snow" and values.get("layers") != "1")  # [mem]
        if drop_state:
            values = {**values, **{k: v for k, v in drop_state.items() if k in values}}
        return {"collision_height": height, "solid": is_full_cube(boxes),
                "replaceable": replaceable, "drops": self._drops(name, values, two_cell)}

    def _drops(self, name: str, values: Mapping[str, str], two_cell: bool = False) -> Any:
        """Specialised, normalised drops (cached by the state properties the table tests)."""
        b = self.r.blocks[name]
        if b.loot_table is None:
            return None
        table = self.r.loot_table(b.loot_table)
        if name not in self._loot_props:
            self._loot_props[name] = tuple(sorted(loot_state_properties(table, name)))
        key = (name, tuple(values.get(p, "") for p in self._loot_props[name]), two_cell)
        if key not in self._drops_cache:
            self._drops_cache[key] = normalise_loot(specialise_loot(table, name, values, two_cell), name,
                                                    self.item_class, self.r.raw["tags"]["item"])
        return self._drops_cache[key]

    def _state_label(self, state: int) -> str:
        """``block[prop=value,...]`` with the properties the block's loot table tests."""
        b = self.r.block_of_state(state)
        values = b.state_values(state)
        inner = ",".join(f"{p}={values[p]}" for p in self._loot_props.get(b.name, ()) if p in values)
        return f"{b.name}[{inner}]" if inner else b.name

    def fallback_key(self, name: str, state: int) -> dict:
        """What a fallback class keys on: exact break inputs, collision class, nav and perception
        flags, and the v1 item classes it can drop (FALLBACK_APPROXIMATED lists the rest)."""
        bp, sp = self.block_props(name), self.state_props(name, state)
        key = {k: bp[k] for k in FALLBACK_BLOCK_KEYS}
        key.update({k: sp[k] for k in FALLBACK_STATE_KEYS})
        key["v1_drops"] = self._drop_info(sp["drops"])[1]
        return key

    def _drop_info(self, drops: Any) -> tuple[str, list[str]]:
        """(canonical JSON, sorted v1 item classes) of a cached drops table."""
        k = id(drops)
        if k not in self._drop_info_cache:
            self._drop_info_cache[k] = (canonical_json(drops),
                                        sorted(i for i in loot_items(drops) if self.item_v1[i]), drops)
        info = self._drop_info_cache[k]
        return info[0], info[1]

    def assign_blocks(self) -> None:
        r = self.r
        self.state_class: dict[int, str] = {}
        self.block_order: list[str] = []
        self.block_spec: dict[str, BlockSpec | None] = {}
        self.block_states: dict[str, list[int]] = defaultdict(list)
        for spec in self.block_specs:
            if spec.name in self.block_spec or spec.name == "buried":
                raise BuildError(f"duplicate or reserved block class {spec.name}")
            if (spec.name == "air") != (not self.block_order):
                raise BuildError("air must be the first block class")
            self.block_spec[spec.name] = spec
            self.block_order.append(spec.name)
            for m in spec.members:
                if m not in r.blocks:
                    raise BuildError(f"block class {spec.name}: unknown block {m}")
                b = r.blocks[m]
                for s in range(b.min_state, b.max_state + 1):
                    if spec.states is not None:
                        vals = b.state_values(s)
                        if any(vals.get(p) not in allowed for p, allowed in spec.states.items()):
                            continue
                    if s in self.state_class:
                        raise BuildError(f"state {s} of {m} is in classes {self.state_class[s]} and {spec.name}")
                    self.state_class[s] = spec.name
                    self.block_states[spec.name].append(s)
                if not any(self.state_class.get(s) == spec.name for s in range(b.min_state, b.max_state + 1)):
                    raise BuildError(f"block class {spec.name}: no state of {m} passes the state filter")
        self.curated_count = len(self.block_order)

        # Fallbacks: group the remaining states by fallback_key.
        groups: dict[str, list[int]] = defaultdict(list)
        for name in sorted(r.blocks, key=lambda n: r.blocks[n].min_state):
            b = r.blocks[name]
            for s in range(b.min_state, b.max_state + 1):
                if s not in self.state_class:
                    groups[canonical_json(self.fallback_key(name, s))].append(s)
        fallbacks = [_Fallback(json.loads(key), states, sorted({r.block_of_state(s).name for s in states},
                                                               key=lambda n: r.blocks[n].min_state))
                     for key, states in groups.items()]
        fallbacks.sort(key=lambda f: (-len(f.states), min(f.states)))  # larger classes pick names first
        curated_phrases = {self.block_phrase_of(n) for n in self.block_order}
        named = sorted(zip(self._fallback_phrases(fallbacks, curated_phrases), fallbacks),
                       key=lambda pf: (pf[0], min(pf[1].states)))
        used: dict[str, int] = defaultdict(int)
        self.fallback_phrase: dict[str, tuple[str, ...]] = {}
        for phrase, fb in named:
            base = "other_" + "_".join(phrase)
            used[base] += 1  # fallback phrases may repeat; the suffix keeps the class names apart
            cname = base if used[base] == 1 else f"{base}_{used[base]}"
            self.block_order.append(cname)
            self.block_spec[cname] = None
            self.fallback_phrase[cname] = phrase
            for s in fb.states:
                self.state_class[s] = cname
                self.block_states[cname].append(s)
        if len(self.block_order) > MAX_BLOCK_CLASS_ID + 1:
            raise BuildError(f"{len(self.block_order)} block classes ({self.curated_count} curated incl. air) do "
                             f"not fit ids 0..{MAX_BLOCK_CLASS_ID}")
        self.block_id = {n: i for i, n in enumerate(self.block_order)}
        self.block_members = {c: sorted({r.block_of_state(s).name for s in st}, key=lambda n: r.blocks[n].min_state)
                              for c, st in self.block_states.items()}

    def block_phrase_of(self, cls: str) -> tuple[str, ...]:
        spec = self.block_spec.get(cls)
        if spec is not None:
            if spec.phrase:
                return spec.phrase
            return self._single_phrase("block", cls, [self.r.blocks[m].display_name for m in spec.members])
        return self.fallback_phrase[cls]

    def _fallback_phrases(self, fallbacks: Sequence[_Fallback],
                          curated: set[tuple[str, ...]]) -> list[tuple[str, ...]]:
        """One phrase per fallback class, in the order given (see the module docstring): a hint,
        else the words every member's name ends with (``other`` first if a curated class has
        them), else ``other <tool> block``. Every phrase is true of every member; phrases may
        repeat across fallback classes."""
        names = {m: words(self.r.blocks[m].display_name) for f in fallbacks for m in f.members}
        used_hints: set[int] = set()
        out: list[tuple[str, ...]] = []
        for f in fallbacks:
            phrase = self._fallback_hint(f, used_hints)
            if phrase is None:
                tail = _common_tail([names[m] for m in f.members])
                if tail:
                    phrase = ("other", *tail) if tail in curated else tail
                else:
                    tools = f.key.get("mineable") or []
                    phrase = ("other", tools[0], "block") if len(tools) == 1 else ("other", "block")
            out.append(phrase)
        for i, (hint_block, want, _p) in enumerate(FALLBACK_PHRASE_HINTS):
            if i not in used_hints:
                self.fail("block", hint_block, "stale phrase hint", f"{want}: matches no fallback class; remove it")
        return out

    def _fallback_hint(self, f: _Fallback, used: set[int]) -> tuple[str, ...] | None:
        """The phrase of the first FALLBACK_PHRASE_HINTS entry that matches this class."""
        for m in f.members:
            for i, (block, want, phrase) in enumerate(FALLBACK_PHRASE_HINTS):
                if block == m and all(f.key.get(k) == v for k, v in want.items()):
                    used.add(i)
                    return tuple(phrase.split())
        return None

    # -- block homogeneity --------------------------------------------------------

    def check_blocks(self) -> None:
        """Curated classes: every field must agree (or an exception applies). Fallback classes
        agree on fallback_key by construction; the rest is approximated and reported."""
        r = self.r
        exc = {e.cls: e for e in self.exceptions if e.kind == "drops"}
        self.block_cls_props: dict[str, dict] = {}
        self.fallback_approx: dict[str, list[str]] = {}
        for cls in self.block_order:
            states = self.block_states.get(cls, [])
            if not states:
                raise BuildError(f"block class {cls} has no states")
            members = self.block_members[cls]
            if self.block_spec[cls] is None:
                self._fallback_props(cls, states, members)
                continue
            bprops = {m: self.block_props(m) for m in members}
            if len({canonical_json(p) for p in bprops.values()}) > 1:
                self._report_diff("block", cls, bprops)
            e = exc.get(cls)
            drop_state = e.drop_state if e else None
            two_cell = bool(e and e.two_cell)
            sprops: dict[str, list[int]] = defaultdict(list)
            changed = False
            for s in states:
                name = r.block_of_state(s).name
                p = self.state_props(name, s, drop_state, two_cell)
                sprops[canonical_json(p)].append(s)
                if drop_state and not changed and canonical_json(self.state_props(name, s)) != canonical_json(p):
                    changed = True
            if drop_state and changed:
                self.used_exceptions[("drops", cls, e.check)].append(f"drops as {dict(drop_state)}")
            if len(sprops) > 1:
                detail = {f"{r.block_of_state(ss[0]).name}#{ss[0]} (+{len(ss) - 1})": json.loads(key)
                          for key, ss in sprops.items()}
                self._report_diff("block", cls, detail)
            rep_block = members[0]
            b = r.blocks[rep_block]
            own = [s for s in states if b.min_state <= s <= b.max_state]
            rep_state = b.default_state if b.default_state in own else own[0]
            if drop_state:
                rep_state = next((s for s in own if all(b.state_values(s).get(k, v) == v
                                                        for k, v in drop_state.items())), rep_state)
            sp = self.state_props(rep_block, rep_state, drop_state, two_cell)
            self.block_cls_props[cls] = {"rep_block": rep_block, "rep_state": rep_state,
                                         "block": self.block_props(rep_block), "solid": sp["solid"],
                                         "collision_height": sp["collision_height"],
                                         "replaceable": sp["replaceable"], "drops": sp["drops"],
                                         "light": self.block_props(rep_block)["light"]}

    def _fallback_props(self, cls: str, states: Sequence[int], members: Sequence[str]) -> None:
        """Class-level values of a fallback class; fields that vary are recorded as approximations."""
        r = self.r
        per_state = [(s, r.block_of_state(s).name) for s in states]
        sps = {s: self.state_props(n, s) for s, n in per_state}
        lights = Counter(self.block_props(n)["light"] for _, n in per_state)
        light = max(lights, key=lambda v: (lights[v], v))  # most common, ties -> brighter
        drop_json = {s: self._drop_info(sps[s]["drops"])[0] for s in states}
        drops = Counter(drop_json.values())
        first = {}
        for st in sorted(states):
            first.setdefault(drop_json[st], st)
        top = max(drops, key=lambda d: (drops[d], -first[d]))  # most common, ties -> lowest state id
        rep_state = first[top]
        rep_block = r.block_of_state(rep_state).name
        solid = all(sps[s]["solid"] for s in states)
        varies = []
        if len(lights) > 1:
            varies.append(f"light {sorted(lights)} -> {light}")
        if len({sps[s]["solid"] for s in states}) > 1:
            varies.append(f"solid mixed -> {solid}")
        if len(drops) > 1:
            varies.append(self._drops_variation(drops, first, sps, rep_state))
        if varies:
            self.fallback_approx[cls] = varies
        self.block_cls_props[cls] = {"rep_block": rep_block, "rep_state": rep_state,
                                     "block": self.block_props(rep_block), "solid": solid,
                                     "collision_height": sps[rep_state]["collision_height"],
                                     "replaceable": sps[rep_state]["replaceable"], "drops": sps[rep_state]["drops"],
                                     "light": light}

    def _drops_variation(self, drops: Mapping[str, int], first: Mapping[str, int], sps: Mapping[int, dict],
                         rep_state: int) -> str:
        """Report text for a fallback class whose states have several drop tables: the table the
        class uses and, where the tables differ in how v1 items drop (snow layers, stems), every
        variant, so the loot and equivalence tests know the exact error."""
        terms = {d: v1_drop_terms(sps[first[d]]["drops"], lambda c: self.item_v1.get(c, False)) for d in drops}
        head = f"drops: {len(drops)} tables -> those of {self._state_label(rep_state)}"
        if len({tuple(t) for t in terms.values()}) == 1:
            return f"{head} (v1 drops the same in all)"
        parts = [f"{self._state_label(first[d])}: {', '.join(terms[d]) or 'no v1 items'}"
                 for d in sorted(drops, key=lambda d: first[d])]
        return f"{head}; v1 drops differ by state: " + " | ".join(parts)

    def _report_diff(self, kind: str, cls: str, by_member: Mapping[str, dict]) -> None:
        """Record one violation per differing field."""
        fields = sorted({k for p in by_member.values() for k in p})
        for f in fields:
            vals = defaultdict(list)
            for m, p in by_member.items():
                vals[canonical_json(p.get(f))].append(m)
            if len(vals) > 1:
                parts = [f"{', '.join(ms[:4])}{' ...' if len(ms) > 4 else ''}: {v[:160]}" for v, ms in vals.items()]
                self.fail(kind, cls, f, " | ".join(parts))

    # -- item homogeneity -----------------------------------------------------------

    def places_block(self, item: str) -> str | None:
        """The block class an item places (block items by name, plus a few named exceptions)."""
        target = ITEM_PLACES_BLOCK.get(item, item)
        b = None if target is None else self.r.blocks.get(target)
        if b is None:
            return None
        return self.state_class[b.default_state]

    def item_props(self, name: str) -> dict:
        it = self.r.items[name]
        f, t, a = it.food, it.tool, it.armor
        return {
            "stack_size": it.stack_size,
            "max_damage": it.max_damage,
            "food": None if f is None else {
                "nutrition": f.nutrition, "saturation": f.saturation, "eat_seconds": f.eat_seconds,
                "can_always_eat": f.can_always_eat,
                "converts_to": None if f.converts_to is None else self.item_class[f.converts_to],
                "effects": list(f.effects)},
            "tool": None if t is None else {"kind": t.kind, "tier": t.tier, "speed": t.speed,
                                            "damage_per_block": t.damage_per_block},
            "attack_damage": it.attack_damage,
            "attack_speed": it.attack_speed,
            "armor": None if a is None else {"slot": a.slot, "points": a.points, "toughness": a.toughness,
                                             "knockback_resistance": a.knockback_resistance},
            "fuel_ticks": it.fuel_ticks,
            "equip_slot": it.equip_slot,
            "places_block_class": self.places_block(name),
        }

    def check_items(self) -> None:
        for cls in self.item_order:
            if not self.item_v1[cls]:
                continue
            members = self.item_members[cls]
            props = {m: self.item_props(m) for m in members}
            if len({canonical_json(p) for p in props.values()}) > 1:
                self._report_diff("item", cls, props)

    # -- recipes ----------------------------------------------------------------------

    @staticmethod
    def _slots(rec) -> tuple[str, list[tuple[str, ...]]]:
        """(shape id, slot alternatives): shaped slots in row-major order, others as listed."""
        if rec.type == "crafting_shaped":
            slots = [rec.key[ch] for row in rec.pattern for ch in row if ch != " "]
            shape = "|".join("".join("x" if c != " " else " " for c in row) for row in rec.pattern)
            return f"shaped:{shape}", slots
        return "unordered", [tuple(s) for s in rec.ingredients]

    def class_recipes(self) -> list[dict]:
        """Class-level recipe book with the family coverage check (see the module docstring)."""
        families: dict[str, list[tuple[Any, list[tuple[str, ...]], list[frozenset[str]]]]] = defaultdict(list)
        for rec in self.r.recipes:
            if rec.special or rec.type not in RECIPE_TYPES or rec.result_item is None:
                continue
            result = self.item_class[rec.result_item]
            if not self.item_v1[result]:
                continue
            shape, slots = self._slots(rec)
            v1_slots = [tuple(n for n in alts if self.item_v1[self.item_class[n]]) for alts in slots]
            if any(not s for s in v1_slots):
                continue  # needs a non-v1 ingredient
            class_sets = [frozenset(self.item_class[n] for n in s) for s in v1_slots]
            order = list(range(len(slots)))
            if shape == "unordered":
                order.sort(key=lambda i: sorted(class_sets[i]))
            v1_slots = [v1_slots[i] for i in order]
            class_sets = [class_sets[i] for i in order]
            sig = canonical_json({"type": rec.type, "result": result, "count": rec.result_count, "shape": shape,
                                  "slots": [sorted(c) for c in class_sets], "cookingtime": rec.cookingtime})
            families[sig].append((rec, v1_slots, class_sets))
        recipes = []
        exc = {(e.cls, e.check): e for e in self.exceptions if e.kind == "recipe"}
        for sig, fam in sorted(families.items()):
            s = json.loads(sig)
            result = s["result"]
            universe = [frozenset(m for c in cs for m in self.item_members[c]) for cs in fam[0][2]]
            accepts = [[frozenset(sl) for sl in v1_slots] for _, v1_slots, _ in fam]
            missing = _uncovered(universe, accepts)
            ids = sorted(rec.id for rec, _, _ in fam)
            if missing is not None:
                detail = f"{ids[:3]}{' ...' if len(ids) > 3 else ''} reject {missing}"
                e = exc.get((result, "correlated slots"))
                if e is None:
                    self.fail("recipe", result, "correlated slots", detail)
                    continue
                self.used_exceptions[("recipe", result, "correlated slots")].append(detail)
                if not e.keep_recipe:
                    continue  # left out of the class-level book
            rec0, _, class_sets = fam[0]
            entry: dict[str, Any] = {"type": s["type"], "result": result, "count": s["count"], "sources": ids}
            if s["type"] in ("crafting_shaped",):
                letters: dict[frozenset[str], str] = {}
                pattern, k = [], 0
                for row in rec0.pattern:
                    line = ""
                    for ch in row:
                        if ch == " ":
                            line += " "
                            continue
                        cs = class_sets[k]
                        k += 1
                        if cs not in letters:
                            letters[cs] = "ABCDEFGHI"[len(letters)]
                        line += letters[cs]
                    pattern.append(line)
                entry["pattern"] = pattern
                entry["key"] = {v: sorted(c) for c, v in letters.items()}
            else:
                entry["ingredients"] = [sorted(c) for c in class_sets]
            if rec0.cookingtime is not None:
                entry["cookingtime"] = rec0.cookingtime
                entry["experience"] = rec0.experience
            recipes.append(entry)
        return recipes

    # -- entities -----------------------------------------------------------------------

    def assign_entities(self) -> None:
        r = self.r
        self.entity_class: dict[str, str] = {}
        self.entity_order: list[str] = ["none"]
        self.entity_members: dict[str, list[str]] = {"none": []}
        self.entity_kind: dict[str, str] = {"none": "other"}
        self.entity_phrase: dict[str, tuple[str, ...]] = {"none": ("nothing",)}
        for spec in self.entity_specs:
            if spec.name in self.entity_members:
                raise BuildError(f"duplicate entity class {spec.name}")
            for m in spec.members:
                if m not in r.entities:
                    raise BuildError(f"entity class {spec.name}: unknown entity {m}")
                if m in self.entity_class:
                    raise BuildError(f"entity {m} is in classes {self.entity_class[m]} and {spec.name}")
                self.entity_class[m] = spec.name
            self.entity_order.append(spec.name)
            self.entity_members[spec.name] = sorted(spec.members)
            self.entity_kind[spec.name] = spec.kind
            self.entity_phrase[spec.name] = spec.phrase or self._single_phrase(
                "entity", spec.name, [r.entities[m].display_name for m in spec.members])
        for cname, kind, phrase in ENTITY_FALLBACKS:
            self.entity_order.append(cname)
            self.entity_members[cname] = []
            self.entity_kind[cname] = kind
            self.entity_phrase[cname] = phrase
        by_kind = {kind: cname for cname, kind, _ in ENTITY_FALLBACKS}
        for name in sorted(r.entities):
            if name in self.entity_class:
                continue
            kind, problem = fallback_entity_kind(r.entities[name])
            if problem:
                self.fail("entity", name, "kind", problem)
            cname = by_kind[kind]
            self.entity_class[name] = cname
            self.entity_members[cname].append(name)
        for name in ENTITY_KIND_OVERRIDES:
            e = r.entities.get(name)
            votes = set() if e is None else {ENTITY_TYPE_KIND.get(e.type or ""),
                                             ENTITY_CATEGORY_KIND.get(e.category or "")}
            if self.entity_class.get(name) not in by_kind.values() or len(votes - {None}) < 2:
                self.fail("entity", name, "stale kind override", "not a fallback entity whose type and category "
                                                                 "disagree; remove it from ENTITY_KIND_OVERRIDES")
        self.entity_id = {n: i for i, n in enumerate(self.entity_order)}

    # -- phrases ---------------------------------------------------------------------------

    def check_phrases(self) -> None:
        """Contracts 3: phrases are lower-case words, unique among the curated classes of each
        kind; a fallback phrase may repeat but never equals a curated phrase of its kind; air =
        ``air`` and buried = ``buried``."""

        def unique(kind: str, phrases: Mapping[str, tuple[str, ...]], fallback: Callable[[str], bool]) -> None:
            curated: dict[tuple[str, ...], str] = {}
            for cls, p in phrases.items():
                if not p or not all(re.fullmatch(r"[a-z0-9]+", w) for w in p):
                    self.fail(kind, cls, "phrase", f"phrase {p!r} is not a sequence of lower-case words")
                if not fallback(cls):
                    if p in curated:
                        self.fail(kind, cls, "phrase", f"phrase {' '.join(p)!r} also used by {curated[p]}")
                    curated[p] = cls
            for cls, p in phrases.items():
                if fallback(cls) and p in curated:
                    self.fail(kind, cls, "phrase", f"fallback phrase {' '.join(p)!r} equals curated class {curated[p]}")

        unique("block", self.block_phrase, lambda c: c in self.fallback_phrase)
        if self.block_phrase["air"] != ("air",) or self.block_phrase["buried"] != ("buried",):
            self.fail("block", "air", "phrase", "air must be 'air' and buried 'buried'")
        unique("item", self.item_phrase, lambda c: c.startswith("other_item_"))
        unique("entity", self.entity_phrase, lambda c: False)
        entity_phrases = {p: c for c, p in self.entity_phrase.items()}
        for c, p in self.block_phrase.items():
            if p in entity_phrases:
                self.fail("block", c, "phrase", f"block phrase {' '.join(p)!r} equals entity class "
                                                f"{entity_phrases[p]}")

    # -- assemble ---------------------------------------------------------------------------

    def build(self) -> BuildResult:
        r = self.r
        self.used_exceptions = defaultdict(list)
        self.assign_items()
        self.assign_blocks()
        self.assign_entities()
        self.block_phrase = {c: self.block_phrase_of(c) for c in self.block_order}
        self.block_phrase["buried"] = ("buried",)
        self.check_blocks()
        self.check_items()
        recipes = self.class_recipes()
        self.check_phrases()
        for spec in self.block_specs:
            if spec.resource and spec.name == "air":
                self.fail("block", "air", "resource", "air cannot be a resource")

        # exceptions: each must match a violation; matched ones are tolerated
        tolerated = []
        for e in self.exceptions:
            hits = self.used_exceptions.get((e.kind, e.cls, e.check))
            if not hits:
                self.fail(e.kind, e.cls, "stale exception", f"{e.check}: no longer matches anything; remove it")
            else:
                tolerated.append((e, hits))

        block_classes = [self._block_entry(c) for c in self.block_order]
        block_classes.append({
            "id": BURIED_ID, "name": "buried", "phrase": ["buried"], "members": [], "fallback": False,
            "resource": False, "state_ranges": [], "n_states": 0, "hardness": -1.0, "mineable": [],
            "requires_tool": False, "needs_tier": None, "tool_speeds": {}, "harvest_tools": None, "drops": None,
            "flags": {"solid": True, "collision_height": 1.0, "transparent": False, "liquid": False,
                      "climbable": False, "hazard": False, "gravity": False, "replaceable": False, "light": 0},
            "note": "perception only: an unexposed cell in the voxel window (DESIGN.md 6.1); never in a world"})
        item_classes = [self._item_entry(c) for c in self.item_order]
        entity_classes = [self._entity_entry(c) for c in self.entity_order]

        rows = ["state_id,block,class_id,class"]
        for s in sorted(self.state_class):
            cls = self.state_class[s]
            rows.append(f"{s},{r.block_of_state(s).name},{self.block_id[cls]},{cls}")
        csv_text = "\n".join(rows) + "\n"
        n_states = sum(b.state_count for b in r.blocks.values())
        if len(self.state_class) != n_states:
            raise BuildError(f"{len(self.state_class)} of {n_states} states assigned")

        concepts = {
            "version": GAME_VERSION,
            "rules_hash": r.hash,
            "csv_sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
            "block_classes": block_classes,
            "item_classes": item_classes,
            "entity_classes": entity_classes,
            "recipes": recipes,
            "exceptions": [{"kind": e.kind, "class": e.cls, "check": e.check, "justification": e.justification,
                            "approximation": e.approximation,
                            **({"drop_state": dict(e.drop_state)} if e.drop_state else {})}
                           for e in self.exceptions]
                          + [{"kind": "block", "class": c, "check": "fallback", "justification": FALLBACK_JUSTIFICATION,
                              "approximation": "; ".join(v)} for c, v in self.fallback_approx.items()]
                          + self._fixed_approximations(),
        }
        concepts["hash"] = concepts_hash(concepts)
        n_res = sum(1 for c in block_classes if c["resource"])
        self.notes.append(f"block classes: {len(block_classes)} ({self.curated_count} curated incl. air, "
                          f"{len(self.block_order) - self.curated_count} fallback, buried); {n_res} resources")
        n_v1 = sum(1 for c in item_classes if c["v1"])
        self.notes.append(f"item classes: {len(item_classes)} ({n_v1} v1, none, "
                          f"{len(item_classes) - n_v1 - 1} fallback)")
        self.notes.append(f"entity classes: {len(entity_classes)} ({len(self.entity_specs)} curated, none, "
                          f"{len(ENTITY_FALLBACKS)} fallback)")
        self.notes.append(f"class-level recipes: {len(recipes)}")
        self.notes.append(f"fallback block classes with approximated light, solid or drops: "
                          f"{len(self.fallback_approx)} (listed in exceptions)")
        return BuildResult(concepts, csv_text, self.violations, tolerated, self.notes)

    @staticmethod
    def _fixed_approximations() -> list[dict]:
        """Approximations built into the class model itself (not tied to one check)."""
        return [
            {"kind": "item", "class": "other_item_*", "check": "fallback",
             "justification": "items outside v1 are grouped by stack size only",
             "approximation": "non-v1 items cannot be crafted with, eaten, worn, placed or used as tools in the "
                              "simulator; a non-v1 tool (netherite, mace, trident) mines like an empty hand"},
            {"kind": "block", "class": "*", "check": "unreachable wall states",
             "justification": "wall states with no post and no sides have no collision box but cannot occur "
                              "in normal play",
             "approximation": "they keep the wall's 1.5 collision class"},
            {"kind": "block", "class": "*", "check": "v1 player breaks",
             "justification": "drops are specialised to a player breaking the block without enchantments "
                              "(DESIGN.md 4.2: no Fortune or Silk Touch) and without an explosion",
             "approximation": "blocks destroyed by explosions drop as if broken by a player"},
            {"kind": "block", "class": "*", "check": "waterlogged",
             "justification": "waterlogged states keep the class of the dry state",
             "approximation": "water inside slabs, stairs, fences and similar blocks is ignored"},
            {"kind": "block", "class": "*", "check": "light",
             "justification": "light emission is per block in the data (lit furnaces, lit redstone ore and "
                              "redstone lamps are not distinguished)",
             "approximation": "block light from state-dependent sources is taken as the block's data value"},
        ]

    def _block_entry(self, cls: str) -> dict:
        cp = self.block_cls_props[cls]
        bp = cp["block"]
        spec = self.block_spec[cls]
        drops = None
        if cp["drops"] is not None:
            rb = self.r.blocks[cp["rep_block"]]
            values = rb.state_values(cp["rep_state"])
            used = loot_state_properties(self.r.loot_table(rb.loot_table), rb.name)
            drops = {"source": rb.loot_table, "state": {k: values[k] for k in sorted(used) if k in values},
                     "table": cp["drops"], "items": sorted(loot_items(cp["drops"]))}
        harvest = None
        if bp["requires_tool"]:
            harvest = sorted({self.item_class[t] for t in bp["harvest"] if self.item_v1[self.item_class[t]]},
                             key=lambda c: self.item_id[c])
        speeds = {self.item_class[t]: s for t, s in bp["tool_speeds"].items() if self.item_v1[self.item_class[t]]}
        return {
            "id": self.block_id[cls],
            "name": cls,
            "phrase": list(self.block_phrase[cls]),
            "members": self.block_members[cls],
            "fallback": spec is None,
            "resource": bool(spec and spec.resource),
            "state_ranges": _ranges(self.block_states[cls]),
            "n_states": len(self.block_states[cls]),
            "hardness": bp["hardness"],
            "mineable": bp["mineable"],
            "requires_tool": bp["requires_tool"],
            "needs_tier": bp["needs_tier"],
            "tool_speeds": dict(sorted(speeds.items(), key=lambda kv: self.item_id[kv[0]])),
            "harvest_tools": harvest,
            "drops": drops,
            "flags": {"solid": cp["solid"], "collision_height": cp["collision_height"],
                      "transparent": bp["transparent"], "liquid": bp["liquid"], "climbable": bp["climbable"],
                      "hazard": bp["hazard"], "gravity": bp["gravity"], "replaceable": cp["replaceable"],
                      "light": cp["light"]},
        }

    def _item_entry(self, cls: str) -> dict:
        members = self.item_members[cls]
        entry: dict[str, Any] = {"id": self.item_id[cls], "name": cls, "phrase": list(self.item_phrase[cls]),
                                 "members": members, "v1": self.item_v1[cls],
                                 "fallback": cls.startswith("other_item_")}
        if cls == "none" or not self.item_v1[cls]:
            entry.update({"stack_size": 64 if cls == "none" else self.r.items[members[0]].stack_size,
                          "max_damage": None, "food": None, "tool": None, "attack_damage": None,
                          "attack_speed": None, "armor": None, "fuel_ticks": None, "places_block_class": None,
                          "equip_slot": None})
            return entry
        entry.update(self.item_props(members[0]))
        return entry

    def _entity_entry(self, cls: str) -> dict:
        members = self.entity_members[cls]
        entry: dict[str, Any] = {"id": self.entity_id[cls], "name": cls, "phrase": list(self.entity_phrase[cls]),
                                 "members": members, "kind": self.entity_kind[cls],
                                 "fallback": cls in {c for c, _, _ in ENTITY_FALLBACKS}}
        stats = ("max_health", "armor", "attack_damage", "speed", "width", "height")
        if entry["fallback"] or not members:
            entry.update({k: None for k in stats})
        else:
            e = self.r.entities[members[0]]
            entry.update({k: getattr(e, k) for k in stats})
        return entry


# Items that place a block of another name, or (None) share a block's name but place nothing [mem].
ITEM_PLACES_BLOCK: dict[str, str | None] = {
    "wheat_seeds": "wheat", "carrot": "carrots", "potato": "potatoes", "beetroot_seeds": "beetroots",
    "sweet_berries": "sweet_berry_bush", "glow_berries": "cave_vines", "melon_seeds": "melon_stem",
    "pumpkin_seeds": "pumpkin_stem", "redstone": "redstone_wire", "string": "tripwire",
    "cocoa_beans": "cocoa", "torchflower_seeds": "torchflower_crop", "pitcher_pod": "pitcher_crop",
    "water_bucket": "water", "lava_bucket": "lava", "powder_snow_bucket": "powder_snow",
    "bamboo": "bamboo_sapling",
    "wheat": None,  # the wheat item is not the wheat crop block
}


def _uncovered(universe: Sequence[frozenset[str]], accepts: Sequence[Sequence[frozenset[str]]]) -> list[str] | None:
    """A combination of slot members that no recipe of the family accepts, or None."""
    n = len(universe)
    memo: dict[tuple[int, tuple[int, ...]], list[str] | None] = {}

    def rec(i: int, alive: tuple[int, ...]) -> list[str] | None:
        if i == n:
            return None
        key = (i, alive)
        if key in memo:
            return memo[key]
        out = None
        for m in sorted(universe[i]):
            nxt = tuple(r for r in alive if m in accepts[r][i])
            if not nxt:
                out = [m]
                break
            sub = rec(i + 1, nxt)
            if sub is not None:
                out = [m, *sub]
                break
        memo[key] = out
        return out

    return rec(0, tuple(range(len(accepts))))


def build(rules: Rules | None = None, *, block_specs: Sequence[BlockSpec] = BLOCK_SPECS,
          item_specs: Sequence[ItemSpec] = ITEM_SPECS, entity_specs: Sequence[EntitySpec] = ENTITY_SPECS,
          exceptions: Sequence[Exception_] = EXCEPTIONS) -> BuildResult:
    """Build the class tables. Inspect ``result.violations`` (empty when the build passes)."""
    return _Builder(rules or load_rules(), block_specs, item_specs, entity_specs, exceptions).build()


def render(result: BuildResult) -> tuple[bytes, bytes]:
    """The exact bytes of the concepts JSON and the CSV."""
    return (canonical_json(result.concepts) + "\n").encode("utf-8"), result.csv_text.encode("utf-8")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockClass:
    id: int
    name: str
    phrase: tuple[str, ...]
    members: tuple[str, ...]
    fallback: bool
    resource: bool
    hardness: float
    mineable: tuple[str, ...]
    requires_tool: bool
    needs_tier: str | None
    tool_speeds: Mapping[str, float]
    harvest_tools: tuple[str, ...] | None
    drops: Mapping[str, Any] | None
    solid: bool
    collision_height: float
    transparent: bool
    liquid: bool
    climbable: bool
    hazard: bool
    gravity: bool
    replaceable: bool
    light: int
    state_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ItemClass:
    id: int
    name: str
    phrase: tuple[str, ...]
    members: tuple[str, ...]
    v1: bool
    fallback: bool
    stack_size: int
    max_damage: int | None
    food: Mapping[str, Any] | None
    tool: Mapping[str, Any] | None
    attack_damage: float | None
    attack_speed: float | None
    armor: Mapping[str, Any] | None
    fuel_ticks: int | None
    places_block_class: str | None
    equip_slot: str | None


@dataclass(frozen=True)
class EntityClass:
    id: int
    name: str
    phrase: tuple[str, ...]
    members: tuple[str, ...]
    kind: str
    fallback: bool
    max_health: float | None
    armor: float | None
    attack_damage: float | None
    speed: float | None
    width: float | None
    height: float | None


def _block_class(d: dict) -> BlockClass:
    f = d["flags"]
    return BlockClass(
        id=d["id"], name=d["name"], phrase=tuple(d["phrase"]), members=tuple(d["members"]), fallback=d["fallback"],
        resource=d["resource"], hardness=d["hardness"], mineable=tuple(d["mineable"]),
        requires_tool=d["requires_tool"], needs_tier=d["needs_tier"], tool_speeds=dict(d["tool_speeds"]),
        harvest_tools=None if d["harvest_tools"] is None else tuple(d["harvest_tools"]), drops=d["drops"],
        solid=f["solid"], collision_height=f["collision_height"], transparent=f["transparent"],
        liquid=f["liquid"], climbable=f["climbable"], hazard=f["hazard"], gravity=f["gravity"],
        replaceable=f["replaceable"], light=f["light"],
        state_ranges=tuple((lo, hi) for lo, hi in d["state_ranges"]))


def _item_class(d: dict) -> ItemClass:
    return ItemClass(**{k: d[k] for k in ("id", "name", "v1", "fallback", "stack_size", "max_damage", "food", "tool",
                                          "attack_damage", "attack_speed", "armor", "fuel_ticks",
                                          "places_block_class", "equip_slot")},
                     phrase=tuple(d["phrase"]), members=tuple(d["members"]))


def _entity_class(d: dict) -> EntityClass:
    return EntityClass(**{k: d[k] for k in ("id", "name", "kind", "fallback", "max_health", "armor",
                                            "attack_damage", "speed", "width", "height")},
                       phrase=tuple(d["phrase"]), members=tuple(d["members"]))


class Concepts:
    """Typed, read-only access to the concepts file, with NumPy arrays for the simulator.

    Block arrays are indexed by block-class id (length 256): ``hardness`` (float32, -1 =
    unbreakable), ``solid``, ``transparent``, ``liquid``, ``climbable``, ``hazard``,
    ``gravity``, ``replaceable``, ``resource``, ``requires_tool``, ``fallback`` (bool),
    ``collision_height`` (float32), ``light`` (uint8), ``needs_tier`` (int8: 0 none, 1 stone,
    2 iron, 3 diamond), ``mineable`` (uint8 bits: axe 1, hoe 2, pickaxe 4, shovel 8).
    ``state_class[state_id]`` (uint8) maps every vanilla block state to its class.
    ``tool_speed[item_class, block_class]`` (float32) and ``can_harvest[item_class,
    block_class]`` (bool) give the break inputs for every held item class (0 = empty hand;
    non-v1 items mine like a hand). Item arrays by item-class id: ``stack_size``,
    ``max_damage``, ``fuel_ticks``, ``nutrition`` (int32, 0 = none), ``saturation``,
    ``attack_damage``, ``attack_speed`` (float32 modifiers, 0 = none), ``armor_points`` (int32),
    ``v1`` (bool). Entity arrays by entity-class id: ``max_health``, ``entity_armor``,
    ``entity_attack_damage`` (float32, NaN = unknown). ``class_by_phrase(kind, words)`` inverts
    ``phrase`` (phrases are unique within each kind).
    """

    def __init__(self, raw: dict, path: Path | None = None):
        if raw.get("version") != GAME_VERSION:
            raise ValueError(f"concepts file is for version {raw.get('version')}, expected {GAME_VERSION}")
        self.raw = raw
        self.path = path
        self.version: str = raw["version"]
        self.hash: str = raw["hash"]
        self.rules_hash: str = raw["rules_hash"]
        self.block_classes: tuple[BlockClass, ...] = tuple(_block_class(d) for d in raw["block_classes"])
        self.item_classes: tuple[ItemClass, ...] = tuple(_item_class(d) for d in raw["item_classes"])
        self.entity_classes: tuple[EntityClass, ...] = tuple(_entity_class(d) for d in raw["entity_classes"])
        self.recipes: tuple[dict, ...] = tuple(raw["recipes"])
        self.exceptions: tuple[dict, ...] = tuple(raw["exceptions"])
        self._blocks = {**{c.name: c for c in self.block_classes}, **{c.id: c for c in self.block_classes}}
        self._items = {**{c.name: c for c in self.item_classes}, **{c.id: c for c in self.item_classes}}
        self._entities = {**{c.name: c for c in self.entity_classes}, **{c.id: c for c in self.entity_classes}}
        self._item_of = {m: c for c in self.item_classes for m in c.members}
        self._entity_of = {m: c for c in self.entity_classes for m in c.members}
        self._by_phrase: dict[str, dict[tuple[str, ...], Any]] = {}
        for kind, classes in (("block", self.block_classes), ("item", self.item_classes),
                              ("entity", self.entity_classes)):
            counts = Counter(c.phrase for c in classes)
            if any(counts[c.phrase] > 1 for c in classes if not c.fallback):
                raise ValueError(f"concepts file has repeated curated {kind} phrases (rebuild it)")
            self._by_phrase[kind] = {c.phrase: c for c in classes if counts[c.phrase] == 1}
        self._build_arrays()

    def __repr__(self) -> str:
        return (f"Concepts(version={self.version!r}, hash={self.hash[:12]}, blocks={len(self.block_classes)}, "
                f"items={len(self.item_classes)}, entities={len(self.entity_classes)})")

    # -- lookups --------------------------------------------------------------------

    def block_class(self, key: str | int) -> BlockClass:
        try:
            return self._blocks[key]
        except KeyError:
            raise KeyError(f"unknown block class {key!r}") from None

    def item_class(self, key: str | int) -> ItemClass:
        try:
            return self._items[key]
        except KeyError:
            raise KeyError(f"unknown item class {key!r}") from None

    def entity_class(self, key: str | int) -> EntityClass:
        try:
            return self._entities[key]
        except KeyError:
            raise KeyError(f"unknown entity class {key!r}") from None

    def block_class_of_state(self, state_id: int) -> BlockClass:
        if not 0 <= state_id < len(self.state_class):
            raise KeyError(f"unknown block state {state_id}")
        return self.block_classes_by_id[int(self.state_class[state_id])]

    def item_class_of(self, vanilla_item: str) -> ItemClass:
        try:
            return self._item_of[vanilla_item.removeprefix("minecraft:")]
        except KeyError:
            raise KeyError(f"unknown item {vanilla_item!r}") from None

    def entity_class_of(self, vanilla_entity: str) -> EntityClass:
        try:
            return self._entity_of[vanilla_entity.removeprefix("minecraft:")]
        except KeyError:
            raise KeyError(f"unknown entity {vanilla_entity!r}") from None

    def class_by_phrase(self, kind: str, phrase: Sequence[str]) -> BlockClass | ItemClass | EntityClass:
        """The class of a kind ("block", "item" or "entity") with this phrase (the inverse of
        ``phrase``). Defined for every curated class; a phrase shared by several fallback
        classes names none of them and raises KeyError."""
        try:
            return self._by_phrase[kind][tuple(phrase)]
        except KeyError:
            raise KeyError(f"no single {kind} class with phrase {' '.join(phrase)!r}") from None

    # -- arrays ------------------------------------------------------------------------

    def _build_arrays(self) -> None:
        n = BURIED_ID + 1
        self.block_classes_by_id: dict[int, BlockClass] = {c.id: c for c in self.block_classes}
        self.hardness = np.full(n, -1.0, np.float32)  # unused ids stay unbreakable
        self.collision_height = np.zeros(n, np.float32)
        self.light = np.zeros(n, np.uint8)
        self.needs_tier = np.zeros(n, np.int8)
        self.mineable = np.zeros(n, np.uint8)
        flags = ("solid", "transparent", "liquid", "climbable", "hazard", "gravity", "replaceable", "resource",
                 "requires_tool", "fallback")
        for f in flags:
            setattr(self, f, np.zeros(n, bool))
        self.block_defined = np.zeros(n, bool)
        n_states = 1 + max(hi for c in self.block_classes for _, hi in c.state_ranges)
        self.state_class = np.full(n_states, BURIED_ID, np.uint8)
        assigned = np.zeros(n_states, bool)
        for c in self.block_classes:
            i = c.id
            self.block_defined[i] = True
            self.hardness[i] = c.hardness
            self.collision_height[i] = c.collision_height
            self.light[i] = c.light
            self.needs_tier[i] = TIER_LEVEL[c.needs_tier]
            self.mineable[i] = sum(MINEABLE_BITS[m] for m in c.mineable)
            for f in flags:
                getattr(self, f)[i] = getattr(c, f)
            for lo, hi in c.state_ranges:
                if assigned[lo:hi + 1].any():
                    raise ValueError(f"block states {lo}..{hi} assigned twice")
                self.state_class[lo:hi + 1] = i
                assigned[lo:hi + 1] = True
        if not assigned.all():
            raise ValueError("some block states have no class")

        m = len(self.item_classes)
        self.stack_size = np.zeros(m, np.int32)
        self.max_damage = np.zeros(m, np.int32)
        self.fuel_ticks = np.zeros(m, np.int32)
        self.nutrition = np.zeros(m, np.int32)
        self.saturation = np.zeros(m, np.float32)
        self.attack_damage = np.zeros(m, np.float32)
        self.attack_speed = np.zeros(m, np.float32)
        self.armor_points = np.zeros(m, np.int32)
        self.v1 = np.zeros(m, bool)
        for c in self.item_classes:
            i = c.id
            self.stack_size[i] = c.stack_size
            self.max_damage[i] = c.max_damage or 0
            self.fuel_ticks[i] = c.fuel_ticks or 0
            if c.food:
                self.nutrition[i] = c.food["nutrition"]
                self.saturation[i] = c.food["saturation"]
            self.attack_damage[i] = c.attack_damage or 0.0
            self.attack_speed[i] = c.attack_speed or 0.0
            self.armor_points[i] = c.armor["points"] if c.armor else 0
            self.v1[i] = c.v1

        self.tool_speed = np.ones((m, n), np.float32)
        self.can_harvest = np.zeros((m, n), bool)
        for b in self.block_classes:
            self.can_harvest[:, b.id] = not b.requires_tool
            for item, speed in b.tool_speeds.items():
                self.tool_speed[self._items[item].id, b.id] = speed
            for item in b.harvest_tools or ():
                self.can_harvest[self._items[item].id, b.id] = True

        k = len(self.entity_classes)
        self.max_health = np.full(k, np.nan, np.float32)
        self.entity_armor = np.full(k, np.nan, np.float32)
        self.entity_attack_damage = np.full(k, np.nan, np.float32)
        for c in self.entity_classes:
            for arr, v in ((self.max_health, c.max_health), (self.entity_armor, c.armor),
                           (self.entity_attack_damage, c.attack_damage)):
                if v is not None:
                    arr[c.id] = v

    # -- integrity -------------------------------------------------------------------------

    def compute_hash(self) -> str:
        return concepts_hash(self.raw)


@lru_cache(maxsize=4)
def _load_cached(path: str, verify: bool) -> Concepts:
    with open(path, encoding="utf-8") as f:
        concepts = Concepts(json.load(f), Path(path))
    if verify and concepts.compute_hash() != concepts.hash:
        raise ValueError(f"{path}: content does not match its hash (edited by hand? rebuild it)")
    return concepts


def load_concepts(path: str | Path | None = None, verify: bool = True) -> Concepts:
    """Load (once, cached per path) the concepts file; ``verify`` checks its hash."""
    p = Path(path) if path is not None else DEFAULT_CONCEPTS_PATH
    return _load_cached(str(p.resolve()), verify)


def read_block_csv(path: str | Path | None = None) -> list[tuple[int, str, int, str]]:
    """Rows of the block-state CSV: (state_id, block, class_id, class)."""
    p = Path(path) if path is not None else DEFAULT_CSV_PATH
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if rows[0] != ["state_id", "block", "class_id", "class"]:
        raise ValueError(f"{p}: unexpected header {rows[0]}")
    return [(int(s), b, int(i), c) for s, b, i, c in rows[1:]]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def report(result: BuildResult, verbose: bool = False) -> str:
    out = io.StringIO()
    for note in result.notes:
        print(note, file=out)
    print(f"tolerated exceptions: {len(result.tolerated)}", file=out)
    for e, hits in result.tolerated:
        print(f"  [{e.kind}] {e.cls}: {e.check} ({len(hits)}x)", file=out)
        if verbose:
            for h in hits[:5]:
                print(f"      {h}", file=out)
    print(f"violations: {len(result.violations)}", file=out)
    for v in result.violations[: None if verbose else 40]:
        print(f"  [{v.kind}] {v.cls}: {v.check}: {v.detail[:400]}", file=out)
    return out.getvalue()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the block, item and entity class tables.")
    ap.add_argument("--rules", type=Path, default=None, help="rules file (default data/rules-1.21.1.json)")
    ap.add_argument("--out", type=Path, default=DEFAULT_CONCEPTS_PATH)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    ap.add_argument("--check", action="store_true", help="build in memory and compare with the files")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)
    try:
        result = build(load_rules(args.rules))
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    print(report(result, args.verbose), end="")
    if not result.ok:
        print("build failed: homogeneity or phrase violations (see above)", file=sys.stderr)
        return 1
    data, csv_bytes = render(result)
    print(f"hash {result.concepts['hash']}, {len(data):,} bytes; csv {len(csv_bytes):,} bytes")
    if args.check:
        same = (args.out.exists() and args.out.read_bytes() == data
                and args.csv.exists() and args.csv.read_bytes() == csv_bytes)
        print(f"{args.out}, {args.csv}: {'up to date' if same else 'DIFFER from a fresh build'}")
        return 0 if same else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    args.csv.write_bytes(csv_bytes)
    print(f"wrote {args.out} and {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

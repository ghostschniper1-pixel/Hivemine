"""Hard-coded game constants for Java Edition 1.21.1 (docs/contracts.md section 2).

Everything here is a number the pinned data dumps do not contain: formula constants
(DESIGN.md 4.3, 4.5, 9), armor values, fuel times and mob stats. Every value carries a
source label, both in the data (``Const.source``) and in a comment:

- ``[n]``: DESIGN.md reference n, where DESIGN.md quotes the value from that source;
- ``[mem]``: from memory, not verified (to be checked against the real game in M2);
- ``[est]``: an estimate or design choice to be measured.

``build_rules`` copies all of it into the ``constants`` section of the rules file as
``{"value": ..., "source": ...}``, and resolves the armor, fuel and mob tables onto
items and entities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SOURCE_LABELS_PATTERN = r"^\[(\d+|mem|est)\]$"


@dataclass(frozen=True)
class Const:
    """A hard-coded value and the label of where it comes from."""

    value: Any
    source: str


# ---------------------------------------------------------------------------
# Game constants used by DESIGN.md 4.3, 4.5, 5 and 9
# ---------------------------------------------------------------------------

GAME: dict[str, Const] = {
    # Ticks
    "ticks_per_second": Const(20, "[mem]"),  # [mem] 20 game ticks per second at full rate
    # Breaking [13] (DESIGN 4.3)
    "break_wrong_tool_speed": Const(1.0, "[13]"),  # [13] speed when the tool's tag does not match
    "break_damage_divisor_harvestable": Const(30, "[13]"),  # [13] dmg = speed/hardness/30
    "break_damage_divisor_not_harvestable": Const(100, "[13]"),  # [13] dmg = speed/hardness/100
    "break_delay_ticks": Const(6, "[13]"),  # [13] ticks before the next block can be started
    "break_haste_per_level": Const(0.2, "[13]"),  # [13] speed *= 1 + 0.2*haste
    # [13] speed *= 0.3^min(fatigue, 4). [mem] suspicion: the game code uses 0.3, 0.09, 0.0027,
    # 0.00081 for levels 1-4, which differs from 0.3^n at levels 3-4. Irrelevant in v1.
    "break_mining_fatigue_base": Const(0.3, "[13]"),  # [13]
    "break_mining_fatigue_max_level": Const(4, "[13]"),  # [13]
    "break_underwater_multiplier": Const(0.2, "[13]"),  # [13] without Aqua Affinity
    "break_airborne_multiplier": Const(0.2, "[13]"),  # [13] not on the ground: /5
    "break_server_accept_fraction": Const(0.7, "[mem]"),  # [mem] server accepts "finished" at 70%
    # Hunger [14] (DESIGN 4.3)
    "max_food": Const(20, "[14]"),  # [14]
    "max_health": Const(20.0, "[mem]"),  # [mem] player health, 10 hearts
    "spawn_food": Const(20, "[14]"),  # [14]
    "spawn_saturation": Const(5.0, "[14]"),  # [14]
    "exhaustion_sprint_per_m": Const(0.1, "[14]"),  # [14]
    "exhaustion_swim_per_m": Const(0.01, "[14]"),  # [14]
    "exhaustion_jump": Const(0.05, "[14]"),  # [14]
    "exhaustion_sprint_jump": Const(0.2, "[14]"),  # [14]
    "exhaustion_break_block": Const(0.005, "[14]"),  # [14]
    "exhaustion_attack": Const(0.1, "[14]"),  # [14]
    "exhaustion_damage": Const(0.1, "[14]"),  # [14] per damage event
    "exhaustion_threshold": Const(4.0, "[14]"),  # [14] at 4: subtract 4, lose 1 saturation or food
    "regen_saturated_interval_ticks": Const(10, "[14]"),  # [14] food 20 and saturation > 0
    "regen_saturated_max_saturation": Const(6.0, "[14]"),  # [14] heal min(sat,6)/6 HP for min(sat,6) exhaustion
    "regen_interval_ticks": Const(80, "[14]"),  # [14] food >= 18: 1 HP per 80 ticks
    "regen_min_food": Const(18, "[14]"),  # [14]
    "regen_exhaustion": Const(6.0, "[14]"),  # [14] exhaustion per regenerated HP
    "starvation_interval_ticks": Const(80, "[14]"),  # [14] food 0: 1 damage per 80 ticks
    "starvation_floor_easy": Const(10.0, "[14]"),  # [14] starvation stops at 10 HP on Easy
    "starvation_floor_normal": Const(1.0, "[14]"),  # [14] stops at 1 HP on Normal
    "starvation_floor_hard": Const(0.0, "[14]"),  # [14] does not stop on Hard
    "sprint_requires_food_above": Const(6, "[14]"),  # [14] sprinting needs food > 6
    # [14] saturation += nutrition * ratio * 2, capped at the food level. The 1.21.1 food
    # component already stores nutrition * ratio * 2 as "saturation" (bread: 5 * 0.6 * 2 = 6).
    "eat_saturation_factor": Const(2.0, "[14]"),  # [14]
    "eat_ticks": Const(32, "[mem]"),  # [mem] default eating time (food component eat_seconds 1.6)
    # Clock [15], [73] (DESIGN 4.2, 4.3)
    "day_length_ticks": Const(24000, "[15]"),  # [15]
    "day_start_tick": Const(0, "[15]"),  # [15]
    "day_end_tick": Const(12000, "[15]"),  # [15]
    "night_start_tick": Const(13000, "[15]"),  # [15]
    "night_end_tick": Const(23000, "[15]"),  # [15]
    "hostile_surface_spawn_start_tick": Const(13188, "[15]"),  # [15] clear weather
    "hostile_surface_spawn_end_tick": Const(22812, "[15]"),  # [15]
    "bed_use_start_tick": Const(12542, "[15]"),  # [15]
    "bed_use_end_tick": Const(23460, "[15]"),  # [15]
    "players_sleeping_percentage": Const(100, "[73]"),  # [73] gamerule default
    "bed_monster_check_horizontal": Const(8.0, "[mem]"),  # [mem] no sleep with a monster within 8 blocks
    "bed_monster_check_vertical": Const(5.0, "[mem]"),  # [mem] ... and 5 blocks up or down
    # Combat [18] (DESIGN 4.3)
    "player_base_attack_damage": Const(1.0, "[18]"),  # [18] fist
    "player_base_attack_speed": Const(4.0, "[mem]"),  # [mem] attacks per second without an item
    "attack_cooldown_numerator_ticks": Const(20.0, "[18]"),  # [18] T = 20 / attackSpeed
    "attack_charge_offset_ticks": Const(0.5, "[18]"),  # [18] multiplier uses (t + 0.5) / T
    "attack_multiplier_min": Const(0.2, "[18]"),  # [18] 0.2 + charge^2 * 0.8, clamped to [0.2, 1]
    "attack_multiplier_scale": Const(0.8, "[18]"),  # [18]
    "critical_multiplier": Const(1.5, "[18]"),  # [18]
    "critical_min_charge": Const(0.9, "[mem]"),  # [mem] attack charge (not damage multiplier) above 0.9
    "invulnerability_ticks": Const(10, "[18]"),  # [18] after a hit
    "fall_damage_safe_blocks": Const(3.0, "[mem]"),  # [mem] damage = blocks fallen - 3
    # Armor formula [17] (DESIGN 4.3): dmg *= 1 - min(cap, max(A/floor_div, A - k*dmg/(T+t0)))/divisor
    "armor_cap": Const(20.0, "[17]"),  # [17]
    "armor_floor_divisor": Const(5.0, "[17]"),  # [17] A/5
    "armor_damage_coefficient": Const(4.0, "[17]"),  # [17] A - 4*dmg/(T+8)
    "armor_toughness_offset": Const(8.0, "[17]"),  # [17]
    "armor_divisor": Const(25.0, "[17]"),  # [17]
    # Spawning and despawning [16] (DESIGN 4.3)
    "spawn_cap_monster": Const(70, "[16]"),  # [16] x loaded chunks / 289
    "spawn_cap_creature": Const(10, "[16]"),  # [16]
    "spawn_cap_chunk_divisor": Const(289, "[16]"),  # [16] 17 x 17 chunks
    "hostile_spawn_max_block_light": Const(0, "[16]"),  # [16]
    "hostile_spawn_max_sky_light": Const(7, "[16]"),  # [16] sky light <= random 0..7
    "spawn_min_player_distance": Const(24.0, "[16]"),  # [16]
    "despawn_instant_distance": Const(128.0, "[16]"),  # [16]
    "despawn_random_distance": Const(32.0, "[16]"),  # [16]
    "despawn_idle_ticks": Const(600, "[16]"),  # [16]
    "despawn_random_chance_per_tick": Const(1.0 / 800.0, "[16]"),  # [16]
    "creeper_fuse_ticks": Const(30, "[mem]"),  # [mem] DESIGN 4.2 "creeper fuse exact"
    "creeper_explosion_power": Const(3.0, "[mem]"),  # [mem]
    # Movement and reach (DESIGN 4.5)
    "walk_speed_m_per_s": Const(4.317, "[mem]"),  # [mem]
    "sprint_speed_m_per_s": Const(5.612, "[mem]"),  # [mem]
    "reach_block": Const(4.5, "[mem]"),  # [mem] survival block reach
    "reach_entity": Const(3.0, "[mem]"),  # [mem] survival entity reach
    "nav_max_drop_blocks": Const(4, "[8]"),  # [8] pathfinder maxDropDown default
    # Smelting [20] (DESIGN 4.2)
    "smelt_ticks": Const(200, "[20]"),  # [20] per item in a furnace
    # Items on the ground (DESIGN 5)
    "item_toss_pickup_delay_ticks": Const(40, "[mem]"),  # [mem] after a player throws an item
    "item_despawn_ticks": Const(6000, "[mem]"),  # [mem] dropped items vanish after 5 minutes of ticking
    "item_pickup_reach_horizontal": Const(1.0, "[mem]"),  # [mem] player box grown by 1 block in x and z
    "item_pickup_reach_vertical": Const(0.5, "[mem]"),  # [mem] ... and 0.5 in y
    # Chat (DESIGN 9)
    "chat_min_interval_ticks": Const(40, "[est]"),  # [est] design limit, below vanilla's spam kick
    "chat_message_cost": Const(-0.001, "[est]"),  # [est] reward per message
    "vanilla_chat_spam_increment": Const(20, "[mem]"),  # [mem] spam counter += 20 per message
    "vanilla_chat_spam_kick_threshold": Const(200, "[mem]"),  # [mem] kick when the counter exceeds 200
}


# ---------------------------------------------------------------------------
# Armor (not in the 1.21.1 data dumps). Points per slot: head, chest, legs, feet.
# DESIGN 4.3 quotes only iron 2/6/5/2 and diamond 3/8/6/3 points and diamond's toughness 2
# from [17]; every other field is from memory and must be checked against [17]. Each field
# carries its own label, so the rules file never presents an unquoted value as sourced.
# ---------------------------------------------------------------------------

ARMOR_SLOTS = ("head", "chest", "legs", "feet")
ARMOR_PIECES = {"helmet": "head", "chestplate": "chest", "leggings": "legs", "boots": "feet"}


@dataclass(frozen=True)
class ArmorMaterial:
    points: Const  # (head, chest, legs, feet)
    toughness: Const  # per piece
    knockback_resistance: Const  # per piece


def _armor(points: tuple[int, int, int, int], toughness: float, knockback_resistance: float,
           points_src: str = "[mem]", toughness_src: str = "[mem]", kb_src: str = "[mem]") -> ArmorMaterial:
    return ArmorMaterial(Const(points, points_src), Const(toughness, toughness_src),
                         Const(knockback_resistance, kb_src))


ARMOR: dict[str, ArmorMaterial] = {
    "leather": _armor((1, 3, 2, 1), 0.0, 0.0),  # [mem]
    "chainmail": _armor((2, 5, 4, 1), 0.0, 0.0),  # [mem]
    "golden": _armor((2, 5, 3, 1), 0.0, 0.0),  # [mem]
    "iron": _armor((2, 6, 5, 2), 0.0, 0.0, points_src="[17]"),  # [17] points; [mem] toughness, kb resistance
    "diamond": _armor((3, 8, 6, 3), 2.0, 0.0, points_src="[17]", toughness_src="[17]"),  # [17]; [mem] kb resistance
    "netherite": _armor((3, 8, 6, 3), 3.0, 0.1),  # [mem] toughness 3, kb resistance 0.1 per piece
}

# Single armor items that are not part of a material set.
ARMOR_EXTRA: dict[str, tuple[str, int, float, float, str]] = {
    "turtle_helmet": ("head", 2, 0.0, 0.0, "[mem]"),  # [mem]
}

# Other items with a fixed equipment slot (no armor value) [mem]. In 1.21.1 these come from
# Java code (Equipable), not from the item components.
EQUIP_SLOT_EXTRA: dict[str, str] = {
    "elytra": "chest",  # [mem]
    "carved_pumpkin": "head",  # [mem]
    "shield": "offhand",  # [mem] ShieldItem's equipment slot
}
EQUIP_HEAD_SUFFIXES = ("_head", "_skull")  # [mem] mob heads and skulls go on the head


# ---------------------------------------------------------------------------
# Furnace fuel (not in the 1.21.1 data dumps): burn time in ticks, in the order of
# vanilla's fuel table (a later entry overrides an earlier one for the same item).
# A name starting with '#' is an item tag. DESIGN 4.2 quotes coal 1600, coal block
# 16000, lava 20000 and log 300 from [20]; everything else is [mem].
# ---------------------------------------------------------------------------

FUEL: list[tuple[str, int, str]] = [
    ("lava_bucket", 20000, "[20]"),  # [20]
    ("coal_block", 16000, "[20]"),  # [20]
    ("blaze_rod", 2400, "[mem]"),  # [mem]
    ("coal", 1600, "[20]"),  # [20]
    ("charcoal", 1600, "[mem]"),  # [mem] same as coal
    ("#logs", 300, "[20]"),  # [20]
    ("#bamboo_blocks", 300, "[mem]"),  # [mem]
    ("#planks", 300, "[mem]"),  # [mem] DESIGN 4.2
    ("bamboo_mosaic", 300, "[mem]"),  # [mem]
    ("#wooden_stairs", 300, "[mem]"),  # [mem]
    ("bamboo_mosaic_stairs", 300, "[mem]"),  # [mem]
    ("#wooden_slabs", 150, "[mem]"),  # [mem]
    ("bamboo_mosaic_slab", 150, "[mem]"),  # [mem]
    ("#wooden_trapdoors", 300, "[mem]"),  # [mem]
    ("#wooden_pressure_plates", 300, "[mem]"),  # [mem]
    ("#wooden_fences", 300, "[mem]"),  # [mem]
    ("#fence_gates", 300, "[mem]"),  # [mem]
    ("note_block", 300, "[mem]"),  # [mem]
    ("bookshelf", 300, "[mem]"),  # [mem]
    ("chiseled_bookshelf", 300, "[mem]"),  # [mem]
    ("lectern", 300, "[mem]"),  # [mem]
    ("jukebox", 300, "[mem]"),  # [mem]
    ("chest", 300, "[mem]"),  # [mem]
    ("trapped_chest", 300, "[mem]"),  # [mem]
    ("crafting_table", 300, "[mem]"),  # [mem]
    ("daylight_detector", 300, "[mem]"),  # [mem]
    ("#banners", 300, "[mem]"),  # [mem]
    ("bow", 300, "[mem]"),  # [mem]
    ("fishing_rod", 300, "[mem]"),  # [mem]
    ("ladder", 300, "[mem]"),  # [mem]
    ("#signs", 200, "[mem]"),  # [mem]
    ("#hanging_signs", 800, "[mem]"),  # [mem]
    ("wooden_shovel", 200, "[mem]"),  # [mem]
    ("wooden_sword", 200, "[mem]"),  # [mem]
    ("wooden_hoe", 200, "[mem]"),  # [mem]
    ("wooden_axe", 200, "[mem]"),  # [mem]
    ("wooden_pickaxe", 200, "[mem]"),  # [mem]
    ("#wooden_doors", 200, "[mem]"),  # [mem]
    ("#boats", 1200, "[mem]"),  # [mem]
    ("#wool", 100, "[mem]"),  # [mem]
    ("#wooden_buttons", 100, "[mem]"),  # [mem]
    ("stick", 100, "[mem]"),  # [mem] DESIGN 4.2
    ("#saplings", 100, "[mem]"),  # [mem]
    ("bowl", 100, "[mem]"),  # [mem]
    ("#wool_carpets", 67, "[mem]"),  # [mem]
    ("dried_kelp_block", 4001, "[mem]"),  # [mem]
    ("crossbow", 300, "[mem]"),  # [mem]
    ("bamboo", 50, "[mem]"),  # [mem]
    ("dead_bush", 100, "[mem]"),  # [mem]
    ("scaffolding", 50, "[mem]"),  # [mem]
    ("loom", 300, "[mem]"),  # [mem]
    ("barrel", 300, "[mem]"),  # [mem]
    ("cartography_table", 300, "[mem]"),  # [mem]
    ("fletching_table", 300, "[mem]"),  # [mem]
    ("smithing_table", 300, "[mem]"),  # [mem]
    ("composter", 300, "[mem]"),  # [mem]
    ("azalea", 100, "[mem]"),  # [mem]
    ("flowering_azalea", 100, "[mem]"),  # [mem]
    ("mangrove_roots", 300, "[mem]"),  # [mem]
]
# Items in this item tag never burn, even if a tag above contains them (crimson and warped wood).
FUEL_EXCLUDED_TAG = Const("non_flammable_wood", "[mem]")  # [mem]


# ---------------------------------------------------------------------------
# Mob stats (not in the data files). attack_damage is the Normal-difficulty damage of
# one hit; speed is the movement_speed attribute (not m/s). None = not applicable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MobStats:
    max_health: Const
    armor: Const
    attack_damage: Const
    speed: Const


def _m(hp: float, armor: float, dmg: float | None, speed: float | None, hp_src: str = "[mem]",
       armor_src: str = "[mem]", dmg_src: str = "[mem]", speed_src: str = "[mem]") -> MobStats:
    return MobStats(Const(hp, hp_src), Const(armor, armor_src), Const(dmg, dmg_src), Const(speed, speed_src))


MOBS: dict[str, MobStats] = {
    "player": _m(20.0, 0.0, 1.0, 0.1, dmg_src="[18]"),  # [mem] 20 HP and speed; [18] fist 1
    "zombie": _m(20.0, 2.0, 3.0, 0.23, hp_src="[19]", armor_src="[19]"),  # [19] 20 HP, 2 armor; [mem] rest
    "husk": _m(20.0, 2.0, 3.0, 0.23),  # [mem]
    "drowned": _m(20.0, 2.0, 3.0, 0.23),  # [mem] melee; trident throwers differ
    "zombie_villager": _m(20.0, 2.0, 3.0, 0.23),  # [mem]
    "skeleton": _m(20.0, 0.0, 3.0, 0.25, dmg_src="[est]"),  # [mem]; [est] arrow damage varies with velocity
    "stray": _m(20.0, 0.0, 3.0, 0.25, dmg_src="[est]"),  # [mem]; [est] arrow damage
    "creeper": _m(20.0, 0.0, None, 0.25),  # [mem] explodes instead (creeper_* constants)
    "spider": _m(16.0, 0.0, 2.0, 0.3),  # [mem]
    "cave_spider": _m(12.0, 0.0, 2.0, 0.3),  # [mem] plus poison on Normal
    "enderman": _m(40.0, 0.0, 7.0, 0.3),  # [mem]
    "witch": _m(26.0, 0.0, None, 0.25),  # [mem] throws potions
    "silverfish": _m(8.0, 0.0, 1.0, 0.25),  # [mem]
    # Overworld hostiles that v1 episodes can meet: slimes (swamps, slime chunks below y 40),
    # phantoms (nights without sleep), bogged (swamps). Other curated hostiles (illagers,
    # guardians, warden, breeze, zombified piglin) have no stats yet (NaN in the concepts).
    # Slime stats depend on its size s (1, 2 or 4): health s^2, attack s (none at s = 1),
    # speed 0.2 + 0.1 s [mem]; the class carries the big slime (s = 4).
    "slime": _m(16.0, 0.0, 4.0, 0.6),  # [mem] big slime
    "phantom": _m(20.0, 0.0, 6.0, None),  # [mem] attack 6 + size (0); flies, no walking speed
    "bogged": _m(16.0, 0.0, 3.0, 0.25, dmg_src="[est]"),  # [mem]; [est] arrow damage as the skeleton's
    "cow": _m(10.0, 0.0, None, 0.2),  # [mem]
    "pig": _m(10.0, 0.0, None, 0.25),  # [mem]
    "sheep": _m(8.0, 0.0, None, 0.23),  # [mem]
    "chicken": _m(4.0, 0.0, None, 0.25),  # [mem]
    "rabbit": _m(3.0, 0.0, None, 0.3),  # [mem]
}


def all_constants() -> dict[str, Const]:
    """Every hard-coded value as a flat name -> Const map (the rules file's ``constants``)."""
    out: dict[str, Const] = dict(GAME)
    for material, a in ARMOR.items():
        out[f"armor:{material}:points"] = Const(list(a.points.value), a.points.source)
        out[f"armor:{material}:toughness"] = a.toughness
        out[f"armor:{material}:knockback_resistance"] = a.knockback_resistance
    for item, (slot, points, toughness, kb, src) in ARMOR_EXTRA.items():
        out[f"armor:{item}"] = Const(
            {"slot": slot, "points": points, "toughness": toughness, "knockback_resistance": kb}, src)
    for name, ticks, src in FUEL:
        out[f"fuel:{name}"] = Const(ticks, src)
    out["fuel_excluded_tag"] = FUEL_EXCLUDED_TAG
    for mob, s in MOBS.items():
        for field_name in ("max_health", "armor", "attack_damage", "speed"):
            c: Const = getattr(s, field_name)
            out[f"mob:{mob}:{field_name}"] = c
    return out

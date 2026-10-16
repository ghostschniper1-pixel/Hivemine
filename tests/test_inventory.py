"""Tests for hivemind.sim.ref.inventory: stacking, overflow, hands, armor, durability, snapshots."""

from __future__ import annotations

import numpy as np
import pytest

from hivemind.rules import load_concepts
from hivemind.sim.ref import mechanics
from hivemind.sim.ref.inventory import (
    ARMOR_INDEX,
    FIND_ORDER,
    MAIN_SIZE,
    N_SLOTS,
    OFFHAND,
    Inventory,
    Stack,
)
from hivemind.sim.ref.rng import CounterRNG, Purpose

C = load_concepts()


def iid(name: str) -> int:
    return C.item_class(name).id


# -- stacking and overflow -------------------------------------------------------------


def test_add_merges_then_fills_hotbar_first():
    inv = Inventory()
    assert inv.add("log", 10) == 0
    assert inv.slots[0] == Stack(iid("log"), 10)
    assert inv.add("log", 60) == 0  # 54 merge into slot 0, the rest opens slot 1
    assert inv.slots[0] == Stack(iid("log"), 64) and inv.slots[1] == Stack(iid("log"), 6)
    assert inv.add("cobblestone", 3) == 0
    assert inv.slots[2] == Stack(iid("cobblestone"), 3)
    assert inv.count("log") == 70 and inv.count("cobblestone") == 3


def test_add_order_selected_slot_then_offhand_then_main():
    inv = Inventory()
    inv.slots[5] = Stack(iid("torch"), 10)
    inv.slots[OFFHAND] = Stack(iid("torch"), 60)
    inv.slots[2] = Stack(iid("torch"), 63)
    inv.select(2)
    assert inv.add("torch", 1) == 0
    assert inv.slots[2].count == 64  # the selected slot first
    assert inv.add("torch", 5) == 0
    assert inv.slots[OFFHAND].count == 64 and inv.slots[5].count == 11  # then the off hand, then slot 5


def test_empty_offhand_and_armor_are_not_filled():
    inv = Inventory()
    for i in range(MAIN_SIZE):
        inv.slots[i] = Stack(iid("dirt"), 64)
    assert inv.add("stick", 3) == 3
    assert inv.slots[OFFHAND] is None and all(inv.slots[i] is None for i in ARMOR_INDEX.values())


def test_overflow_returns_what_did_not_fit():
    inv = Inventory()
    assert inv.free_capacity("log") == 36 * 64
    assert inv.add("log", 36 * 64 + 5) == 5
    assert inv.count("log") == 36 * 64
    assert inv.free_capacity("log") == 0 and inv.free_capacity("stick") == 0
    assert inv.add("stick") == 1


def test_stack_sizes_from_the_classes():
    inv = Inventory()
    assert inv.add("snowball", 20) == 0
    assert [s.count for s in inv.slots[:2]] == [16, 4]  # snowballs stack to 16
    assert inv.add("wooden_pickaxe", 2) == 0
    picks = [s for s in inv.slots if s is not None and s.item == iid("wooden_pickaxe")]
    assert len(picks) == 2 and all(s.count == 1 and s.durability == 59 for s in picks)


def test_free_capacity_matches_add():
    rng = CounterRNG(3)
    for trial in range(30):
        s = rng.stream(0, trial, Purpose.GENERIC)
        inv = Inventory()
        for _ in range(s.randint(0, 40)):
            inv.add(("log", "stick", "snowball", "iron_sword", "torch")[s.below(5)], s.randint(1, 64))
        for item in ("log", "snowball", "iron_sword"):
            cap = inv.free_capacity(item)
            probe = inv.copy()
            assert probe.add(item, cap + 3) == 3


def test_class_members_share_a_stack_known_approximation():
    # Documented approximation (module docstring): a slot holds an item class, so 20 oak and 20
    # birch logs share one slot here (two in the game). The count per class is exact.
    assert len(C.item_class("log").members) > 1
    inv = Inventory()
    inv.add("log", 20)
    inv.add("log", 20)
    assert inv.slots[0] == Stack(iid("log"), 40) and inv.slots[1] is None
    assert inv.count("log") == 40


def test_moved_stacks_keep_their_durability():
    inv = Inventory()
    inv.add("iron_pickaxe", durability=3)
    inv.add("torch", 10)
    (taken,) = inv.remove("iron_pickaxe")
    assert taken == Stack(iid("iron_pickaxe"), 1, 3) and inv.count("iron_pickaxe") == 0
    other = Inventory()
    assert other.add_stack(taken) == 0
    assert other.slots[0] == Stack(iid("iron_pickaxe"), 1, 3)  # not repaired to 250
    assert other.add_stack(Stack(iid("torch"), 70)) == 0 and other.count("torch") == 70
    with pytest.raises(ValueError):
        other.add_stack(Stack(iid("iron_pickaxe"), 1))  # a moved tool must carry its durability
    # remove returns one part per slot touched, in Mineflayer's order
    inv.slots[20] = Stack(iid("torch"), 3)
    assert inv.remove("torch", 5) == [Stack(iid("torch"), 3), Stack(iid("torch"), 2)]


def test_world_tables_get_durability_fields():
    from hivemind.sim.ref.inventory import NO_DURABILITY, add_durability_fields
    from hivemind.sim.ref.world import CHEST_SLOTS, World

    w = World(16, 16)
    add_durability_fields(w)
    add_durability_fields(w)  # a second call changes nothing
    assert "durability" in w.items.fields and "slot_durability" in w.chests.fields
    i = w.items.alloc(item=iid("iron_pickaxe"), count=1, durability=3)
    assert w.items.durability[i] == 3
    j = w.items.alloc(item=iid("torch"), count=4)
    assert w.items.durability[j] == NO_DURABILITY
    assert w.chests.slot_durability.shape == (w.chests.capacity, CHEST_SLOTS)
    assert (w.chests.slot_durability == NO_DURABILITY).all()


def test_add_validates_arguments():
    inv = Inventory()
    with pytest.raises(ValueError):
        inv.add("none")
    with pytest.raises(ValueError):
        inv.add("log", durability=5)
    with pytest.raises(ValueError):
        inv.add("iron_pickaxe", durability=251)
    with pytest.raises(KeyError):
        inv.add("not_an_item")


# -- removing and counting -----------------------------------------------------------------


def test_remove_uses_mineflayer_order_main_then_hotbar():
    assert FIND_ORDER[:2] == (9, 10) and FIND_ORDER[-9:] == tuple(range(9))
    inv = Inventory()
    inv.slots[0] = Stack(iid("planks"), 10)
    inv.slots[20] = Stack(iid("planks"), 3)
    inv.remove("planks", 5)
    assert inv.slots[20] is None and inv.slots[0] == Stack(iid("planks"), 8)


def test_remove_too_many_raises_and_changes_nothing():
    inv = Inventory()
    inv.add("coal", 3)
    before = inv.snapshot()
    with pytest.raises(ValueError):
        inv.remove("coal", 4)
    assert inv.snapshot() == before


def test_offhand_counts_but_is_only_removed_on_request():
    inv = Inventory()
    inv.slots[OFFHAND] = Stack(iid("torch"), 5)
    inv.add("torch", 2)  # merges into the off hand
    assert inv.count("torch") == 7 and inv.count_main("torch") == 0
    with pytest.raises(ValueError):
        inv.remove("torch", 1)
    inv.remove("torch", 3, offhand=True)
    assert inv.count("torch") == 4
    assert inv.counts() == {iid("torch"): 4}


# -- hands and armor ------------------------------------------------------------------------


def test_hold_selects_moves_or_swaps():
    inv = Inventory()
    inv.add("dirt", 64)
    inv.slots[3] = Stack(iid("stone_pickaxe"), 1, 131)
    assert inv.hold("stone_pickaxe") == 3 and inv.held_item() == iid("stone_pickaxe")
    inv.slots[15] = Stack(iid("iron_sword"), 1, 250)
    slot = inv.hold("iron_sword")  # moved from the main inventory to the first empty hotbar slot
    assert slot == 1 and inv.slots[15] is None and inv.held() == Stack(iid("iron_sword"), 1, 250)
    for i in range(9):
        inv.slots[i] = inv.slots[i] or Stack(iid("dirt"), 1)
    inv.slots[30] = Stack(iid("bread"), 4)
    inv.select(4)
    old = inv.slots[4]
    assert inv.hold("bread") == 4  # hotbar full: swapped into the selected slot
    assert inv.slots[4] == Stack(iid("bread"), 4) and inv.slots[30] == old
    with pytest.raises(ValueError):
        inv.hold("diamond")
    with pytest.raises(ValueError):
        inv.select(9)


def test_wear_and_swap_armor():
    inv = Inventory()
    for piece in ("helmet", "chestplate", "leggings", "boots"):
        inv.add(f"iron_{piece}")
    for piece in ("helmet", "chestplate", "leggings", "boots"):
        assert inv.wear(f"iron_{piece}") == 0
    assert inv.slots[ARMOR_INDEX["head"]].item == iid("iron_helmet")
    assert inv.slots[ARMOR_INDEX["feet"]].item == iid("iron_boots")
    assert inv.armor_totals() == (15, 0.0)  # iron 2/6/5/2 [17]
    assert inv.count("iron_helmet") == 0  # worn armor is not "held"
    inv.add("diamond_helmet")
    assert inv.wear("diamond_helmet") == iid("iron_helmet")  # swapped: the iron helmet comes back
    assert inv.count("iron_helmet") == 1
    assert inv.armor_totals() == (16, 2.0)
    with pytest.raises(ValueError):
        inv.wear("bread")  # not held
    inv.add("bread")
    with pytest.raises(ValueError):
        inv.wear("bread")


def test_full_diamond_armor_and_damage_after_armor():
    inv = Inventory()
    for piece in ("helmet", "chestplate", "leggings", "boots"):
        inv.add(f"diamond_{piece}")
        inv.wear(f"diamond_{piece}")
    assert inv.armor_totals() == (20, 8.0)  # diamond 3/8/6/3 plus toughness 2 each [17]
    assert inv.damage_after_armor(10.0) == mechanics.armor_damage(10.0, 20, 8.0)
    assert Inventory().damage_after_armor(7.0) == 7.0


def test_offhand_swap():
    inv = Inventory()
    inv.add("shield")
    inv.add("torch", 10)
    inv.hold_offhand("shield")
    assert inv.offhand().item == iid("shield") and inv.count_main("shield") == 0
    inv.hold_offhand("torch")  # the shield goes back to where the torches were
    assert inv.offhand() == Stack(iid("torch"), 10) and inv.count_main("shield") == 1


def test_attack_stats_of_held_item():
    inv = Inventory()
    fist = inv.attack_stats()
    assert (fist.damage, fist.attack_speed) == (1.0, 4.0)
    inv.add("wooden_sword")
    inv.hold("wooden_sword")
    s = inv.attack_stats()
    assert s.damage == 4.0 and abs(s.attack_speed - 1.6) < 1e-6  # [18] wood sword 4 at 1.6/s


# -- durability -------------------------------------------------------------------------------


def test_wooden_pickaxe_breaks_after_59_stone():
    inv = Inventory()
    inv.add("wooden_pickaxe")
    inv.hold("wooden_pickaxe")
    for i in range(58):
        assert not inv.mine_with_held("stone")
    assert inv.held().durability == 1
    assert inv.mine_with_held("stone")  # damage reaches max_damage: it breaks
    assert inv.held() is None and inv.count("wooden_pickaxe") == 0


def test_instant_blocks_cost_no_durability_except_for_shears():
    inv = Inventory()
    inv.add("iron_pickaxe")
    inv.hold("iron_pickaxe")
    inv.mine_with_held("short_grass")
    inv.mine_with_held("torch")
    assert inv.held().durability == 250
    inv.add("shears")
    inv.hold("shears")
    inv.mine_with_held("short_grass")
    assert inv.held().durability == 237
    inv.mine_with_held("fire")
    assert inv.held().durability == 237


def test_sword_and_tool_wear_per_block_and_per_hit():
    inv = Inventory()
    inv.add("stone_sword")
    inv.hold("stone_sword")
    inv.mine_with_held("dirt", 3)  # damage_per_block 2 for swords
    assert inv.held().durability == 131 - 6
    inv.hit_with_held(4)  # 1 per hit
    assert inv.held().durability == 131 - 10
    inv.add("stone_axe")
    inv.hold("stone_axe")
    inv.hit_with_held(2)  # digging tools: 2 per hit [mem]
    assert inv.held().durability == 131 - 4
    inv.add("bread")
    inv.hold("bread")
    assert not inv.hit_with_held() and not inv.mine_with_held("stone")
    inv.select(8)
    assert inv.held() is None and not inv.hit_with_held()


def test_use_dispatches_per_block_or_per_hit():
    inv = Inventory()
    inv.add("iron_sword")
    inv.hold("iron_sword")
    assert not inv.use(3)  # 3 blocks x 2
    assert not inv.use(2, hit=True)  # 2 hits x 1
    assert not inv.use(5, block="torch")  # instant: nothing
    assert inv.held().durability == 250 - 8
    assert inv.use(250)
    assert inv.held() is None and not inv.use()


def test_armor_wear_from_hits():
    inv = Inventory()
    inv.add("leather_helmet")
    inv.add("iron_chestplate")
    inv.wear("leather_helmet")
    inv.wear("iron_chestplate")
    assert inv.wear_armor(10.0) == []  # floor(10 / 4) = 2 each
    assert inv.worn()["head"].durability == 55 - 2 and inv.worn()["chest"].durability == 240 - 2
    inv.wear_armor(1.0)  # at least 1
    assert inv.worn()["head"].durability == 52
    assert inv.wear_armor(4.0 * 52) == [iid("leather_helmet")]  # 52 each: the helmet (52 left) breaks
    assert "head" not in inv.worn() and inv.worn()["chest"].durability == 237 - 52


# -- snapshots and invariants -------------------------------------------------------------------


def test_snapshot_restore_equality_and_array():
    inv = Inventory()
    inv.add("log", 5)
    inv.add("iron_pickaxe")
    a = inv.snapshot()
    other = inv.copy()
    assert other == inv and other is not inv
    inv.add("dirt", 3)
    inv.select(3)
    assert inv != other and inv.snapshot() != a
    inv.restore(a)
    assert inv == other and inv.snapshot() == a
    arr = inv.to_array()
    assert arr.shape == (N_SLOTS, 3) and arr.dtype == np.int32
    assert tuple(arr[1]) == (iid("iron_pickaxe"), 1, 250) and tuple(arr[2]) == (0, 0, -1)


def test_random_operations_keep_counts_within_stack_limits():
    rng = CounterRNG(11)
    items = ("log", "stick", "snowball", "iron_pickaxe", "bread", "iron_helmet", "torch")
    for trial in range(20):
        s = rng.stream(0, trial, Purpose.GENERIC)
        inv = Inventory()
        total: dict[str, int] = {}
        for _ in range(60):
            item = items[s.below(len(items))]
            op = s.below(4)
            if op == 0:
                n = s.randint(1, 40)
                total[item] = total.get(item, 0) + n - inv.add(item, n)
            elif op == 1 and inv.count_main(item):
                n = s.randint(1, inv.count_main(item))
                inv.remove(item, n)
                total[item] -= n
            elif op == 2 and inv.count_main(item):
                inv.hold(item)
            elif op == 3 and inv.count_main(item) and C.item_class(item).equip_slot == "head":
                inv.wear(item)
            for i, st in inv.stacks():
                assert 1 <= st.count <= C.item_class(st.item).stack_size
            worn = sum(1 for st in inv.worn().values() if st.item == iid("iron_helmet"))
            for name, n in total.items():
                assert inv.count(name) + (worn if name == "iron_helmet" else 0) == n

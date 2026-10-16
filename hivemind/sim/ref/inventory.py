"""A player's inventory over item classes (contracts 5; DESIGN.md 4.2 "combat, armor, durability, stacks").

Usage::

    from hivemind.sim.ref.inventory import Inventory
    inv = Inventory()                          # empty; item classes from load_concepts()
    inv.add("log", 10)                         # -> 0 (nothing left over)
    inv.add("wooden_pickaxe")                  # tools start at full durability (59)
    inv.count("log"), inv.free_capacity("log") # 10, 64 * 34 + 54
    inv.hold("wooden_pickaxe"); inv.mine_with_held("stone")   # durability 59 -> 58
    inv.wear("iron_helmet"); inv.armor_totals()                # (2, 0.0)
    state = inv.snapshot(); ...; inv.restore(state)            # replays, differential tests

**Slots** use the game's indices for the player inventory [mem]: 0-8 hotbar, 9-35 the rest of the
main inventory, 36-39 armor (36 feet, 37 legs, 38 chest, 39 head), 40 the off hand. Each slot is
empty (``None``) or a ``Stack(item, count, durability)``: an item-class id from ``load_concepts()``,
a count from 1 to the class's stack size, and the remaining durability (``max_damage - damage``)
for damageable items, else ``None``. ``selected`` is the hotbar slot in the main hand (0-8).
Public methods take an item class as its name (``"iron_ingot"``) or its id.

**Adding** follows ``Inventory.add`` of the game [mem]: each pass puts as many items as fit into
the first slot that already holds the same item with room left, searched in the order selected
hotbar slot, off hand, then slots 0-35; failing that, into the first empty slot of 0-35 (hotbar
first). An empty off hand or armor slot is never filled by picking up. Damageable items never
merge (in v1 they all have stack size 1). ``add`` returns what did not fit.

**Removing** and finding a stack for an action (``remove``, ``wear``, ``hold_offhand``) search the
main inventory in Mineflayer's order, slots 9-35 and then the hotbar 0-8: the order of
``Window.findInventoryItem``, which the bridge's crafting, tossing and equipping use [mem]. Items
in the off hand and in armor slots are not taken by ``remove`` unless asked (``offhand=True``),
because Mineflayer's window counts and searches skip them too [mem]; ``count`` includes the off
hand, ``count_main`` does not. Counts per class are exact; which slot an item ends up in after a
craft can differ from the real game (Mineflayer puts leftovers into the first empty slot).

**Durability** [mem]: an item breaks when its damage reaches ``max_damage`` (``ItemStack.
hurtAndBreak``: damage >= max), i.e. when the remaining durability here reaches 0, so a wooden
pickaxe (59) breaks on its 59th block. Mining a block with a tool costs the tool component's
``damage_per_block`` (rules ``tool.damage_per_block``: 1 for digging tools and shears, 2 for
swords) unless the block breaks instantly (hardness 0); shears lose 1 on every block except fire.
A melee hit costs swords 1 and axes, pickaxes, shovels and hoes 2 (``SwordItem`` and
``DiggerItem.postHurtEnemy``); other items nothing. A hit that armor can reduce costs each worn
armor piece max(1, floor(damage / 4)) of the damage before armor (``LivingEntity.doHurtEquipment``).
No Unbreaking in v1.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hivemind.rules import load_concepts
from hivemind.sim.ref import mechanics

if TYPE_CHECKING:
    from hivemind.rules.classes import Concepts, ItemClass

ItemRef = str | int  # an item class by name or id

HOTBAR_SIZE = 9  # [mem] slots 0-8
MAIN_SIZE = 36  # [mem] slots 0-35: hotbar and the 27 slots above it
ARMOR_SLOTS = ("feet", "legs", "chest", "head")  # [mem] armor slots 36-39, in the game's order
ARMOR_INDEX = {name: MAIN_SIZE + i for i, name in enumerate(ARMOR_SLOTS)}  # feet 36 ... head 39
OFFHAND = 40  # [mem] the off-hand slot
N_SLOTS = 41
# Mineflayer's search order over the main inventory: window slots 9-35, then the hotbar 36-44,
# i.e. game slots 9..35 then 0..8 (Window.findInventoryItem) [mem].
FIND_ORDER: tuple[int, ...] = (*range(HOTBAR_SIZE, MAIN_SIZE), *range(HOTBAR_SIZE))
# Durability lost per melee hit by tool kind (SwordItem / DiggerItem.postHurtEnemy) [mem].
HIT_DAMAGE: dict[str, int] = {"sword": 1, "pickaxe": 2, "axe": 2, "shovel": 2, "hoe": 2}
ARMOR_WEAR_DIVISOR = 4.0  # [mem] each armor piece loses max(1, floor(damage / 4)) per hit
FIRE_BLOCKS = frozenset({"fire", "soul_fire"})  # [mem] BlockTags.FIRE: shears take no damage on these


@dataclass(frozen=True)
class Stack:
    """One occupied slot: item-class id, count, remaining durability (None if not damageable)."""

    item: int
    count: int
    durability: int | None = None


@dataclass(frozen=True)
class InventoryState:
    """An immutable copy of an inventory (``Inventory.snapshot``); equal states compare equal."""

    slots: tuple[Stack | None, ...]
    selected: int


def item_id(concepts: Concepts, item: ItemRef) -> int:
    """The id of an item class given by name or id (KeyError if unknown)."""
    if isinstance(item, str):
        return concepts.item_class(item).id
    return concepts.item_class(int(item)).id


class Inventory:
    """36 main slots, 4 armor slots, the off hand and the selected hotbar slot (see the module docstring)."""

    def __init__(self, concepts: Concepts | None = None):
        self.concepts: Concepts = concepts if concepts is not None else load_concepts()
        self.slots: list[Stack | None] = [None] * N_SLOTS
        self.selected: int = 0

    # -- basics -------------------------------------------------------------------------

    def __repr__(self) -> str:
        held = ", ".join(f"{self.concepts.item_class(i).name} x{n}" for i, n in self.counts().items())
        return f"Inventory({held or 'empty'}; selected {self.selected})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Inventory):
            return NotImplemented
        return self.slots == other.slots and self.selected == other.selected

    __hash__ = None  # mutable

    def item_class(self, item: ItemRef) -> ItemClass:
        return self.concepts.item_class(item_id(self.concepts, item))

    def id_of(self, item: ItemRef) -> int:
        return item_id(self.concepts, item)

    def stack_size(self, item: ItemRef) -> int:
        return int(self.concepts.item_class(item_id(self.concepts, item)).stack_size)

    def snapshot(self) -> InventoryState:
        return InventoryState(tuple(self.slots), self.selected)

    def restore(self, state: InventoryState) -> None:
        if len(state.slots) != N_SLOTS:
            raise ValueError(f"an inventory state has {N_SLOTS} slots, got {len(state.slots)}")
        self.slots = list(state.slots)
        self.selected = state.selected

    def copy(self) -> Inventory:
        inv = Inventory(self.concepts)
        inv.restore(self.snapshot())
        return inv

    def to_array(self) -> np.ndarray:
        """int32 array (41, 3): item id, count, durability (-1 if none); empty slots are (0, 0, -1)."""
        out = np.zeros((N_SLOTS, 3), np.int32)
        out[:, 2] = -1
        for i, s in enumerate(self.slots):
            if s is not None:
                out[i] = (s.item, s.count, -1 if s.durability is None else s.durability)
        return out

    def stacks(self) -> Iterator[tuple[int, Stack]]:
        """(slot index, stack) of every occupied slot, armor and off hand included, in index order."""
        for i, s in enumerate(self.slots):
            if s is not None:
                yield i, s

    # -- counting -----------------------------------------------------------------------

    def count_main(self, item: ItemRef) -> int:
        """Items of a class in the 36 main slots (what Mineflayer's window counts; crafting uses this)."""
        iid = item_id(self.concepts, item)
        return sum(s.count for s in self.slots[:MAIN_SIZE] if s is not None and s.item == iid)

    def count(self, item: ItemRef) -> int:
        """Items of a class held: main slots and the off hand (worn armor is not counted)."""
        iid = item_id(self.concepts, item)
        off = self.slots[OFFHAND]
        return self.count_main(iid) + (off.count if off is not None and off.item == iid else 0)

    def counts(self) -> dict[int, int]:
        """Item-class id -> count over the main slots and the off hand, in id order."""
        out: dict[int, int] = {}
        for s in (*self.slots[:MAIN_SIZE], self.slots[OFFHAND]):
            if s is not None:
                out[s.item] = out.get(s.item, 0) + s.count
        return dict(sorted(out.items()))

    def slots_of(self, item: ItemRef) -> list[int]:
        """Main-inventory slots holding the item, in Mineflayer's search order."""
        iid = item_id(self.concepts, item)
        return [i for i in FIND_ORDER if self.slots[i] is not None and self.slots[i].item == iid]

    def free_capacity(self, item: ItemRef) -> int:
        """How many of this item ``add`` would accept now (room in matching stacks of the main slots
        and the off hand, plus a full stack per empty main slot)."""
        iid = item_id(self.concepts, item)
        cls = self.concepts.item_class(iid)
        size = int(cls.stack_size)
        room = 0
        for i in (*range(MAIN_SIZE), OFFHAND):
            s = self.slots[i]
            if s is None:
                room += size if i < MAIN_SIZE else 0
            elif s.item == iid and s.durability is None and cls.max_damage is None:
                room += size - s.count
        return room

    def empty_main_slots(self) -> int:
        return sum(1 for s in self.slots[:MAIN_SIZE] if s is None)

    # -- adding and removing ------------------------------------------------------------

    def add(self, item: ItemRef, n: int = 1, durability: int | None = None) -> int:
        """Add ``n`` items in the game's order (module docstring); returns how many did not fit.

        Damageable items get ``durability`` (default: full, the class's ``max_damage``).
        """
        iid = item_id(self.concepts, item)
        if iid == 0:
            raise ValueError("cannot add the 'none' item class")
        if n < 0:
            raise ValueError(f"cannot add a negative count ({n})")
        cls = self.concepts.item_class(iid)
        size = int(cls.stack_size)
        if cls.max_damage is not None:
            durability = cls.max_damage if durability is None else durability
            if not 0 < durability <= cls.max_damage:
                raise ValueError(f"durability {durability} outside 1..{cls.max_damage} for {cls.name}")
        elif durability is not None:
            raise ValueError(f"{cls.name} has no durability")
        left = n
        while left > 0:
            i = self._slot_with_room(iid, size, durability)
            if i is None:
                i = self._first_empty_main()
            if i is None:
                break
            have = self.slots[i].count if self.slots[i] is not None else 0
            put = min(left, size - have)
            self.slots[i] = Stack(iid, have + put, durability)
            left -= put
        return left

    def _slot_with_room(self, iid: int, size: int, durability: int | None) -> int | None:
        """``Inventory.getSlotWithRemainingSpace`` [mem]: selected slot, off hand, then 0..35."""
        if durability is not None:
            return None  # damageable items never merge
        for i in (self.selected, OFFHAND, *range(MAIN_SIZE)):
            s = self.slots[i]
            if s is not None and s.item == iid and s.durability is None and s.count < size:
                return i
        return None

    def _first_empty_main(self) -> int | None:
        """``Inventory.getFreeSlot`` [mem]: the first empty slot of 0..35 (hotbar first)."""
        for i in range(MAIN_SIZE):
            if self.slots[i] is None:
                return i
        return None

    def remove(self, item: ItemRef, n: int = 1, *, offhand: bool = False) -> None:
        """Remove ``n`` items from the main slots in Mineflayer's search order (then the off hand if
        ``offhand``). Raises ValueError, changing nothing, if fewer are held."""
        iid = item_id(self.concepts, item)
        have = self.count(iid) if offhand else self.count_main(iid)
        if n < 0 or n > have:
            raise ValueError(f"cannot remove {n} {self.concepts.item_class(iid).name}: {have} held")
        left = n
        for i in (*FIND_ORDER, OFFHAND) if offhand else FIND_ORDER:
            if left == 0:
                break
            s = self.slots[i]
            if s is None or s.item != iid:
                continue
            take = min(left, s.count)
            self.slots[i] = None if take == s.count else Stack(iid, s.count - take, s.durability)
            left -= take

    def take_slot(self, index: int) -> Stack | None:
        """Empty one slot and return what it held."""
        s = self.slots[index]
        self.slots[index] = None
        return s

    # -- hands and armor --------------------------------------------------------------------

    def select(self, slot: int) -> None:
        """Select hotbar slot 0-8 (the main hand)."""
        if not 0 <= slot < HOTBAR_SIZE:
            raise ValueError(f"hotbar slot must be 0-8, got {slot}")
        self.selected = slot

    def held(self) -> Stack | None:
        """The stack in the main hand (the selected hotbar slot)."""
        return self.slots[self.selected]

    def held_item(self) -> int:
        """Item-class id in the main hand, 0 (``none``) for an empty hand."""
        s = self.slots[self.selected]
        return 0 if s is None else s.item

    def offhand(self) -> Stack | None:
        return self.slots[OFFHAND]

    def hold(self, item: ItemRef) -> int:
        """Put the item in the main hand; returns the hotbar slot now selected. ValueError if not held.

        Rule [design]: if the selected slot holds it, nothing changes; else select the first
        hotbar slot that holds it; else move the first stack found (Mineflayer's search order) to
        the first empty hotbar slot and select it; else swap it with the selected slot's stack.
        """
        iid = item_id(self.concepts, item)
        if self.held_item() == iid:
            return self.selected
        for i in range(HOTBAR_SIZE):
            s = self.slots[i]
            if s is not None and s.item == iid:
                self.selected = i
                return i
        src = self._find(iid)
        dst = next((i for i in range(HOTBAR_SIZE) if self.slots[i] is None), self.selected)
        self._swap(src, dst)
        self.selected = dst
        return dst

    def hold_offhand(self, item: ItemRef) -> None:
        """Swap the first stack of the item (Mineflayer's search order) with the off hand."""
        iid = item_id(self.concepts, item)
        off = self.slots[OFFHAND]
        if off is not None and off.item == iid:
            return
        self._swap(self._find(iid), OFFHAND)

    def wear(self, item: ItemRef) -> int:
        """Move an armor item into its slot (by its equip slot), swapping out what was worn there.

        Returns the item-class id previously worn there (0 if none). ValueError if the item is not
        armor or not held.
        """
        iid = item_id(self.concepts, item)
        cls = self.concepts.item_class(iid)
        if cls.equip_slot not in ARMOR_INDEX:
            raise ValueError(f"{cls.name} cannot be worn (equip slot {cls.equip_slot})")
        dst = ARMOR_INDEX[cls.equip_slot]
        old = self.slots[dst]
        self._swap(self._find(iid), dst)
        return 0 if old is None else old.item

    def worn(self) -> dict[str, Stack]:
        """Armor slot name -> stack, for occupied armor slots."""
        return {name: self.slots[i] for name, i in ARMOR_INDEX.items() if self.slots[i] is not None}

    def _find(self, iid: int) -> int:
        for i in FIND_ORDER:
            s = self.slots[i]
            if s is not None and s.item == iid:
                return i
        raise ValueError(f"no {self.concepts.item_class(iid).name} in the main inventory")

    def _swap(self, a: int, b: int) -> None:
        self.slots[a], self.slots[b] = self.slots[b], self.slots[a]

    # -- combat -----------------------------------------------------------------------------

    def armor_totals(self) -> tuple[int, float]:
        """Armor points and toughness of the worn armor (points floored as the game does [mem]),
        the inputs of ``mechanics.armor_damage``."""
        points, toughness = 0.0, 0.0
        for i in ARMOR_INDEX.values():
            s = self.slots[i]
            if s is not None:
                armor = self.concepts.item_class(s.item).armor
                if armor is not None:  # anything else in an armor slot gives no armor
                    points += armor["points"]
                    toughness += armor["toughness"]
        return math.floor(points), mechanics.f32(toughness)

    def damage_after_armor(self, damage: float) -> float:
        """Damage after the worn armor [17] (``mechanics.armor_damage``)."""
        points, toughness = self.armor_totals()
        return mechanics.armor_damage(damage, points, toughness)

    def attack_stats(self) -> mechanics.AttackStats:
        """Melee attributes with the held item (fist: 1 damage, 4 attacks/s) [18]."""
        s = self.slots[self.selected]
        if s is None:
            return mechanics.attack_stats(0.0, 0.0)
        cls = self.concepts.item_class(s.item)
        return mechanics.attack_stats(cls.attack_damage or 0.0, cls.attack_speed or 0.0)

    # -- durability ---------------------------------------------------------------------------

    def damage_slot(self, index: int, amount: int) -> bool:
        """Take ``amount`` durability from the item in a slot; True if it broke (the slot empties).
        Items without durability are unaffected."""
        s = self.slots[index]
        if s is None or s.durability is None or amount <= 0:
            return False
        left = s.durability - amount
        if left <= 0:  # damage >= max_damage: the item breaks [mem]
            self.slots[index] = None
            return True
        self.slots[index] = Stack(s.item, s.count, left)
        return False

    def use(self, n: int = 1, *, hit: bool = False, block: str | int | None = None) -> bool:
        """Durability use of the held item: ``n`` melee hits (``hit``) or ``n`` broken blocks of a
        class (``block``; None = a block that is neither instant nor fire). True if it broke."""
        if hit:
            return self.hit_with_held(n)
        if block is not None:
            return self.mine_with_held(block, n)
        s = self.slots[self.selected]
        tool = None if s is None else self.concepts.item_class(s.item).tool
        return tool is not None and self.damage_slot(self.selected, n * max(tool["damage_per_block"], 0))

    def mine_with_held(self, block: str | int, n: int = 1) -> bool:
        """Wear on the held item from breaking ``n`` blocks of a block class; True if it broke.

        A tool loses its ``damage_per_block`` per block unless the block breaks instantly
        (hardness 0); shears lose it on every block except fire [mem]. Non-tools lose nothing.
        """
        s = self.slots[self.selected]
        if s is None:
            return False
        tool = self.concepts.item_class(s.item).tool
        if tool is None or tool["damage_per_block"] <= 0:
            return False
        bc = self.concepts.block_class(block)
        if tool["kind"] == "shears":
            if FIRE_BLOCKS & set(bc.members):
                return False
        elif bc.hardness == 0:
            return False
        return self.damage_slot(self.selected, n * tool["damage_per_block"])

    def hit_with_held(self, n: int = 1) -> bool:
        """Wear on the held item from ``n`` melee hits that land; True if it broke."""
        s = self.slots[self.selected]
        if s is None:
            return False
        tool = self.concepts.item_class(s.item).tool
        per_hit = HIT_DAMAGE.get(tool["kind"], 0) if tool is not None else 0
        return self.damage_slot(self.selected, n * per_hit)

    def wear_armor(self, damage: float) -> list[int]:
        """Wear on every worn armor piece from a hit that armor reduces (``damage`` before armor);
        returns the item-class ids that broke."""
        if damage <= 0.0:
            return []
        per_piece = int(max(1.0, mechanics.f32(mechanics.f32(damage) / ARMOR_WEAR_DIVISOR)))
        broken = []
        for i in ARMOR_INDEX.values():
            s = self.slots[i]
            is_armor = s is not None and self.concepts.item_class(s.item).armor is not None
            if is_armor and self.damage_slot(i, per_piece):
                broken.append(s.item)
        return broken

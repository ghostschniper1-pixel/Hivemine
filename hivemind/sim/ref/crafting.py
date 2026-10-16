"""Class-level crafting and smelting (contracts 5; DESIGN.md 4.2 and 5: ``craft``, ``smelt_start``, ``collect``).

Usage::

    from hivemind.sim.ref.crafting import Furnace, RecipeBook, craft, craftable
    book = RecipeBook()                               # the class-level recipes of load_concepts()
    craftable(inv, near_table=False)                  # {item-class id: max crafts}, e.g. {planks: 3}
    r = craft(inv, "stick", 64, near_table=False)     # "up to" 64 sticks; r.code, r.n_crafts, r.clicks
    f = Furnace()
    book.smelt_start(inv, f, "raw_iron", 8)           # puts 8 raw iron and the fuel chosen by the rule below
    f.tick(1600); book.collect(inv, f)                # 8 iron ingots

**Recipes.** The book reads the class-level recipes of ``data/concepts-1.21.1.json`` (built by
``hivemind.rules.classes``: every vanilla recipe whose result is a v1 class, with each ingredient
slot turned into the set of v1 item classes it accepts; the build checked that every combination of
class members is accepted). v1 uses the player's 2x2 grid, the crafting table (3x3) and the furnace
only (DESIGN.md 4.2), so the book keeps ``crafting_shaped`` and ``crafting_shapeless`` recipes and,
of the cooking recipes, type ``smelting`` (smoking, blasting and campfire recipes are left out).
A shaped recipe needs a table when its pattern is wider or taller than 2; a shapeless one when it
has more than 4 ingredients. Known approximation (concepts ``exceptions``): a bed takes any three
wool here; in the game the three wool must share a colour (sim-to-real check in M7).

**Mixing alternatives.** A slot that accepts several classes (cobblestone or cobbled deepslate in a
furnace; coal or charcoal in a torch) can be filled from any of them, as the game's tag ingredients
allow per slot. Recipes whose different slots accept overlapping but unequal class sets would need a
matching search; none exist in the 1.21.1 v1 book, and building a book with one raises.

**Crafting, "up to" counts** (contracts 4.3). ``craft(inv, item, up_to_items)`` crafts as many
times as possible such that the items made do not exceed ``up_to_items``, but at least once:
``n = max(1, min(max_crafts, up_to_items // count))``. ``craft 6 stick.`` makes 4 sticks (one craft;
two would make 8), ``craft 1 planks.`` makes 4 planks. Failure codes (DESIGN.md 5):
``MISSING_INGREDIENT`` when no recipe can be crafted once, ``NO_STATION`` when only recipes that
need a crafting table could be and none is near. A craft whose result would not fit stops the
command there with ``INV_FULL`` (container ``inventory``), whether or not earlier crafts were made
(``n_crafts`` says how many): a full inventory ends an action with INV_FULL (DESIGN.md 4.5,
contracts 4.3), so the result words are ``inventory full``, not ``done``.

**Choices** [design], deterministic:
- Recipe: among the recipes for the item that can be crafted at least once at this station,
  prefer the recipe that uses the item class the agent holds most of; then the one allowing more
  crafts; then book order (sorted by source recipe ids). One command uses one recipe, as one
  Mineflayer ``craft`` call does.
- Ingredients: each craft fills a slot group from the class held most first (ties: lower class id),
  then the next, which keeps mixing to a minimum.

**Clicks** (duration model, DESIGN.md 5: "n_clicks·RT + 2 ticks per item [est]; a pickaxe is about
8-10 clicks"). Mineflayer crafts one result per loop pass [6][75]; per pass it picks up an
ingredient stack whenever the next grid cell needs another class, right-clicks one item into each
cell, puts the rest back, takes the result and puts it away [mem]. So clicks per craft = cells +
pickups + 3 [est]: 10 for a pickaxe, 8 for a crafting table, 6 for sticks. ``smelt_start`` counts 3
clicks per stack put into the furnace (input, fuel) and ``collect`` 3 per stack taken [est],
matching DESIGN.md 5's "about 6" and "about 3".

**Furnace** (``Furnace.tick``) follows ``AbstractFurnaceBlockEntity.serverTick`` of 1.21.1 [mem]
exactly, per tick: if lit, the burn time drops by 1; then, if lit or (fuel and input present): when
not lit and the input can be smelted (a smelting recipe exists and the output slot is empty or holds
the result with room), one fuel item is consumed and the burn time set to its fuel ticks [20]
(lava leaves a bucket in the fuel slot [mem]); if lit and the input can be smelted, the cook
progress rises by 1 and at the recipe's cooking time (200 ticks [20]) one input becomes one output
and progress restarts; otherwise the progress resets to 0. When not lit and nothing can burn, the
progress cools by 2 per tick. So fuel burns on while there is no input, partial fuel carries over
from one item to the next, and 8 raw iron with 1 coal finish exactly at tick 1600. ``tick(n)``
jumps over stretches where nothing but the counters change, with the same result as n single ticks
(tested). Experience is ignored in v1 (the game stores it per recipe and pays it on extraction).

**Fuel choice for smelt_start** [design], deterministic. Candidates are the fuel classes held in
the main slots that can go into the fuel slot (it must be empty or hold the same class) and that,
with the heat already in the furnace, smelt at least one more item. The input class itself
counts, minus the items being smelted (logs can fuel log -> charcoal). Candidates are ranked in
tiers: plain fuel first, then fuels that are stations the agent needs (``PRESERVED_FUELS``:
crafting table, chest), then damageable items (wooden tools, bow): a higher tier is used only
when no lower tier has a candidate. Within a tier the fuel that smelts the most of the requested
items wins, then the least wasted burn time, then the fewest fuel items, then the lower class id.
No fuel is added when the furnace already has heat for every item. The number of items put in is
then what that fuel covers ("up to" semantics), so no input waits for fuel.

**Heat already in the furnace**, counted exactly as ``Furnace.step`` uses it: a lit furnace with
burn time B left cooks B - 1 more ticks (the counter drops before the furnace cooks; when it
reaches 0 the next fuel item lights and cooks in that same tick), and each fuel item in the slot
cooks its fuel ticks. The items already inside need their cook time minus the progress made.

**Output slot.** The items put in are capped so that, with the items already inside, every result
fits the output slot as it is now; then no item waits for a collect and no burning fuel goes to
waste. When the output slot holds another item, or is full, nothing cooks and no fuel lights until
it is taken (``collect``): the burn time left and the cook progress are then not counted (they may
be gone by then), only the fuel in the slot, and the cap is a whole output stack. So every
``smelt_start`` that returns OK smelts all its items, given only that the agent collects the
output when it blocks the furnace (tested by running plans to completion).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from hivemind.rules import load_concepts
from hivemind.sim.ref import mechanics
from hivemind.sim.ref.inventory import Inventory, ItemRef, item_id

if TYPE_CHECKING:
    from hivemind.rules.classes import Concepts

GRID_2X2 = 2  # [mem] the player's crafting grid is 2x2; the crafting table's 3x3
CRAFT_TYPES = ("crafting_shaped", "crafting_shapeless")  # DESIGN.md 4.2: 2x2 grid and table
FURNACE_TYPE = "smelting"  # DESIGN.md 4.2: the furnace only (no smoker, blast furnace or campfire in v1)
CLICKS_AFTER_GRID = 3  # [est] put the rest away, take the result, put it into the inventory [mem]
CLICKS_PER_FURNACE_STACK = 3  # [est] pick up a stack, click the furnace slot, put the rest back
FUEL_REMAINDER = {"lava_bucket": "bucket"}  # [mem] Item.getCraftingRemainingItem: an empty bucket stays
PRESERVED_FUELS = ("crafting_table", "chest")  # [design] stations burned only when nothing else is held
IDLE = 1 << 30  # Furnace._quiet_ticks: nothing happens until a slot changes

OK = "OK"
MISSING_INGREDIENT = "MISSING_INGREDIENT"
NO_STATION = "NO_STATION"
INV_FULL = "INV_FULL"
NO_TARGET = "NO_TARGET"


# =============================================================================
# Recipes
# =============================================================================


@dataclass(frozen=True)
class CraftRecipe:
    """A class-level crafting recipe.

    ``cells`` lists, for every filled grid cell (shaped: row-major; shapeless: as listed), the
    item-class ids it accepts. ``groups`` merges equal cell sets: (accepted ids, cells per craft).
    """

    result: int
    count: int
    shaped: bool
    width: int
    height: int
    cells: tuple[tuple[int, ...], ...]
    sources: tuple[str, ...]

    @property
    def needs_table(self) -> bool:
        """Does not fit the player's 2x2 grid."""
        if self.shaped:
            return self.width > GRID_2X2 or self.height > GRID_2X2
        return len(self.cells) > GRID_2X2 * GRID_2X2

    @property
    def n_ingredients(self) -> int:
        """Items consumed per craft."""
        return len(self.cells)

    @property
    def groups(self) -> tuple[tuple[tuple[int, ...], int], ...]:
        out: dict[tuple[int, ...], int] = {}
        for c in self.cells:
            out[c] = out.get(c, 0) + 1
        return tuple(out.items())


@dataclass(frozen=True)
class SmeltRecipe:
    """A class-level furnace recipe: any of ``inputs`` -> ``count`` x ``result`` in ``cook_ticks``."""

    inputs: tuple[int, ...]
    result: int
    count: int
    cook_ticks: int
    experience: float
    sources: tuple[str, ...]


@dataclass(frozen=True)
class CraftResult:
    """What ``craft`` did. ``consumed`` maps item-class ids to counts; ``clicks`` is the estimated
    number of window clicks (module docstring) for the duration model."""

    code: str
    item: int
    recipe: CraftRecipe | None = None
    n_crafts: int = 0
    n_items: int = 0
    consumed: Mapping[int, int] = field(default_factory=dict)
    clicks: int = 0
    container: str | None = None  # with INV_FULL: what is full (``inventory``), for the result words

    @property
    def n_ingredients(self) -> int:
        """Ingredient items consumed in total."""
        return sum(self.consumed.values())

    def duration_ticks(self, rt_ticks: float) -> float:
        """DESIGN.md 5's craft time without the walk: per craft, clicks x RT + 2 ticks [est]."""
        return self.clicks * rt_ticks + 2 * self.n_crafts


@dataclass(frozen=True)
class SmeltPlan:
    """Items to put into a furnace and the fuel to add (fuel 0 = none needed or none possible)."""

    item: int
    n_items: int
    fuel: int = 0
    fuel_units: int = 0
    waste_ticks: int = 0


@dataclass(frozen=True)
class FurnaceResult:
    """What ``smelt_start`` or ``collect`` did. With ``INV_FULL``, ``container`` says what is full:
    ``inventory`` (collect) or ``furnace`` (smelt_start), the ``{container}`` of the result words."""

    code: str
    item: int = 0
    n_items: int = 0
    fuel: int = 0
    fuel_units: int = 0
    clicks: int = 0
    container: str | None = None


class RecipeBook:
    """The v1 recipe book at class level (see the module docstring).

    ``recipes`` overrides the concepts file's class-level recipe list (same format), for tests.
    """

    def __init__(self, concepts: Concepts | None = None, recipes: Sequence[Mapping[str, Any]] | None = None):
        self.concepts: Concepts = concepts if concepts is not None else load_concepts()
        raw = self.concepts.recipes if recipes is None else recipes
        cid = self._id
        crafting: list[CraftRecipe] = []
        smelting: list[SmeltRecipe] = []
        for r in sorted(raw, key=lambda r: (r["type"], tuple(r.get("sources", ())), r["result"])):
            if r["type"] in CRAFT_TYPES:
                crafting.append(self._craft_recipe(r))
            elif r["type"] == FURNACE_TYPE:
                (alts,) = r["ingredients"]
                smelting.append(SmeltRecipe(tuple(sorted(cid(a) for a in alts)), cid(r["result"]), int(r["count"]),
                                            int(r["cookingtime"]), float(r.get("experience", 0.0)),
                                            tuple(r.get("sources", ()))))
        self.crafting: tuple[CraftRecipe, ...] = tuple(crafting)
        self.smelting: tuple[SmeltRecipe, ...] = tuple(smelting)
        by_result: dict[int, list[CraftRecipe]] = {}
        for rec in self.crafting:
            by_result.setdefault(rec.result, []).append(rec)
        self._by_result = {k: tuple(v) for k, v in by_result.items()}
        self._smelt: dict[int, SmeltRecipe] = {}
        for rec in self.smelting:
            for i in rec.inputs:
                old = self._smelt.get(i)
                if old is not None and (old.result, old.count) != (rec.result, rec.count):
                    raise ValueError(f"two smelting recipes for {self.concepts.item_class(i).name}")
                self._smelt.setdefault(i, rec)
        self._fuel_remainder = {cid(k): cid(v) for k, v in FUEL_REMAINDER.items()}
        self._preserved = frozenset(cid(n) for n in PRESERVED_FUELS)

    def _id(self, item: ItemRef) -> int:
        return item_id(self.concepts, item)

    def _craft_recipe(self, r: Mapping[str, Any]) -> CraftRecipe:
        cid = self._id
        if r["type"] == "crafting_shaped":
            rows = list(r["pattern"])
            key = {k: tuple(sorted(cid(a) for a in v)) for k, v in r["key"].items()}
            cells = tuple(key[ch] for row in rows for ch in row if ch != " ")
            # ShapedRecipePattern shrinks empty outer rows and columns before fitting the grid [mem]
            used_rows = [i for i, row in enumerate(rows) if row.strip()]
            used_cols = [j for row in rows for j, ch in enumerate(row) if ch != " "]
            width, height, shaped = max(used_cols) - min(used_cols) + 1, used_rows[-1] - used_rows[0] + 1, True
        else:
            cells = tuple(tuple(sorted(cid(a) for a in alts)) for alts in r["ingredients"])
            width = height = 0
            shaped = False
        sets = {frozenset(c) for c in cells}
        for a in sets:
            for b in sets:
                if a != b and a & b:
                    raise ValueError(f"recipe for {r['result']}: slots accept overlapping class sets {sorted(a)}, "
                                     f"{sorted(b)} (unsupported, see the module docstring)")
        return CraftRecipe(cid(r["result"]), int(r["count"]), shaped, width, height, cells,
                           tuple(r.get("sources", ())))

    # -- lookups ------------------------------------------------------------------------

    def recipes_for(self, item: ItemRef) -> tuple[CraftRecipe, ...]:
        """Crafting recipes whose result is the item (empty if none)."""
        return self._by_result.get(self._id(item), ())

    def needs_table(self, item: ItemRef) -> bool:
        """True when every crafting recipe for the item needs a crafting table (KeyError if none)."""
        recipes = self.recipes_for(item)
        if not recipes:
            raise KeyError(f"no crafting recipe for {self.concepts.item_class(self._id(item)).name}")
        return all(r.needs_table for r in recipes)

    def smelt_recipe(self, item: ItemRef) -> SmeltRecipe | None:
        """The furnace recipe that takes this input class, or None."""
        return self._smelt.get(self._id(item))

    def cook_ticks(self, item: ItemRef) -> int:
        """Cooking time of an input (``getTotalCookTime``: the recipe's, else 200 [20])."""
        rec = self._smelt.get(self._id(item)) if item else None
        return rec.cook_ticks if rec is not None else mechanics.smelt_ticks()

    def fuel_ticks(self, item: ItemRef) -> int:
        """Burn time of a fuel class (0 if it is not fuel) [20][mem]."""
        return int(self.concepts.item_class(self._id(item)).fuel_ticks or 0) if item else 0

    def fuel_remainder(self, item: int) -> int:
        """What a fuel item leaves in the fuel slot (an empty bucket for lava), 0 for nothing."""
        return self._fuel_remainder.get(item, 0)

    # -- crafting -----------------------------------------------------------------------

    @staticmethod
    def max_crafts(inv: Inventory, recipe: CraftRecipe) -> int:
        """How many times the recipe can be crafted from the main inventory (ingredients only)."""
        best = math.inf
        for alts, per_craft in recipe.groups:
            have = sum(inv.count_main(a) for a in alts)
            best = min(best, have // per_craft)
        return int(best)

    def craftable(self, inv: Inventory, near_table: bool) -> dict[int, int]:
        """{item-class id: max crafts} for every item that can be crafted at least once here."""
        out: dict[int, int] = {}
        for rec in self.crafting:
            if rec.needs_table and not near_table:
                continue
            n = self.max_crafts(inv, rec)
            if n > 0:
                out[rec.result] = max(out.get(rec.result, 0), n)
        return dict(sorted(out.items()))

    def choose_recipe(self, inv: Inventory, item: ItemRef, near_table: bool) -> tuple[str, CraftRecipe | None]:
        """(code, recipe): the recipe ``craft`` would use (module docstring), or the failure code."""
        able = [(rec, n) for rec in self.recipes_for(item) if (n := self.max_crafts(inv, rec)) > 0]
        if not able:
            return MISSING_INGREDIENT, None
        here = [(rec, n) for rec, n in able if near_table or not rec.needs_table]
        if not here:
            return NO_STATION, None

        def most_held(rec: CraftRecipe) -> int:
            return max(inv.count_main(a) for alts, _ in rec.groups for a in alts)

        order = {id(rec): k for k, rec in enumerate(self.crafting)}
        best = min(here, key=lambda rn: (-most_held(rn[0]), -rn[1], order[id(rn[0])]))
        return OK, best[0]

    @staticmethod
    def _fill(inv: Inventory, recipe: CraftRecipe) -> list[int]:
        """The class put into each cell for one craft: each group drawn from the class held most
        first (ties: lower id). Assumes one craft is possible."""
        take: dict[tuple[int, ...], list[int]] = {}
        for alts, need in recipe.groups:
            order = sorted(alts, key=lambda a: (-inv.count_main(a), a))
            seq: list[int] = []
            for a in order:
                k = min(inv.count_main(a), need - len(seq))
                seq += [a] * k
                if len(seq) == need:
                    break
            take[alts] = seq
        pos = {alts: 0 for alts in take}
        out = []
        for c in recipe.cells:
            out.append(take[c][pos[c]])
            pos[c] += 1
        return out

    def craft(self, inv: Inventory, item: ItemRef, up_to_items: int, near_table: bool = False) -> CraftResult:
        """Craft up to ``up_to_items`` of an item (at least one craft); changes ``inv`` (module docstring)."""
        iid = self._id(item)
        if up_to_items < 1:
            raise ValueError(f"item count must be at least 1, got {up_to_items}")
        code, rec = self.choose_recipe(inv, iid, near_table)
        if rec is None:
            return CraftResult(code, iid)
        target = max(1, min(self.max_crafts(inv, rec), up_to_items // rec.count))
        consumed: dict[int, int] = {}
        crafts = clicks = 0
        full = False
        for _ in range(target):
            cells = self._fill(inv, rec)
            before = inv.snapshot()
            used: dict[int, int] = {}
            for c in cells:
                used[c] = used.get(c, 0) + 1
            for c, n in used.items():
                inv.remove(c, n)
            if inv.add(rec.result, rec.count) > 0:  # the result does not fit: stop here
                inv.restore(before)
                full = True
                break
            for c, n in used.items():
                consumed[c] = consumed.get(c, 0) + n
            pickups = 1 + sum(1 for a, b in pairwise(cells) if a != b)
            clicks += len(cells) + pickups + CLICKS_AFTER_GRID
            crafts += 1
        code, container = (INV_FULL, "inventory") if full else (OK, None)
        return CraftResult(code, iid, rec, crafts, crafts * rec.count, dict(sorted(consumed.items())), clicks,
                           container)

    # -- smelting -------------------------------------------------------------------------

    def _fuel_tier(self, fuel: int) -> int:
        if self.concepts.item_class(fuel).max_damage is not None:
            return 2
        return 1 if fuel in self._preserved else 0

    def _output_room(self, f: Furnace, rec: SmeltRecipe) -> int:
        """How many more cooks of this recipe the output slot takes now (``canBurn``'s room test)."""
        size = int(self.concepts.item_class(rec.result).stack_size)
        if f.output_count == 0:
            return size // rec.count
        if f.output != rec.result:
            return 0
        return (size - f.output_count) // rec.count

    def plan_smelt(self, inv: Inventory, item: ItemRef, up_to: int, furnace: Furnace | None = None) -> SmeltPlan:
        """How many items ``smelt_start`` would put in and which fuel it would add (module docstring)."""
        iid = self._id(item)
        f = furnace if furnace is not None else Furnace(book=self)
        rec = self._smelt.get(iid)
        if rec is None or up_to < 1 or (f.input_count > 0 and f.input != iid):
            return SmeltPlan(iid, 0)
        inside = f.input_count
        cook = rec.cook_ticks
        slot_heat = f.fuel_count * self.fuel_ticks(f.fuel)
        room = self._output_room(f, rec)
        if room == 0:
            # Output blocked: nothing cooks and no fuel lights until it is taken. By then the burn
            # time left and the cook progress may be gone, so only the fuel in the slot counts.
            heat, backlog = slot_heat, inside * cook
            room = int(self.concepts.item_class(rec.result).stack_size) // rec.count  # once taken
        else:
            # A lit furnace with burn time B left cooks B - 1 more ticks: step() lowers the counter
            # before it cooks, and at 0 the next fuel item lights (and cooks) in the same tick.
            heat = max(f.burn_left - 1, 0) + slot_heat
            backlog = inside * cook - (f.cook_ticks if inside else 0)  # heat the items inside still need
        held = inv.count_main(iid)
        # room - inside: every item put in has room in the output slot, so none waits for a collect
        n_max = min(up_to, held, int(self.concepts.item_class(iid).stack_size) - inside, room - inside)
        if n_max <= 0:
            return SmeltPlan(iid, 0)

        def covered(extra_heat: int) -> int:
            """Items of the n_max that ``heat + extra_heat`` smelts."""
            return max(0, min(n_max, (heat + extra_heat - backlog) // cook))

        base = covered(0)
        if base >= n_max:
            return SmeltPlan(iid, n_max)
        options: list[tuple[tuple, SmeltPlan]] = []
        fuels = [f.fuel] if f.fuel else sorted(inv.counts())
        for fuel in fuels:
            ticks = self.fuel_ticks(fuel)
            slot_room = self.concepts.item_class(fuel).stack_size - (f.fuel_count if f.fuel == fuel else 0)
            if ticks <= 0 or slot_room <= 0 or inv.count_main(fuel) == 0:
                continue
            best: SmeltPlan | None = None
            for n in range(n_max, base, -1):  # the most items this fuel can cover
                need = backlog + n * cook - heat
                units = -(-need // ticks)
                if units <= slot_room and units + (n if fuel == iid else 0) <= inv.count_main(fuel):
                    best = SmeltPlan(iid, n, fuel, units, units * ticks - need)
                    break
            if best is not None:
                options.append(((self._fuel_tier(fuel), -best.n_items, best.waste_ticks, best.fuel_units, fuel), best))
        if not options:
            return SmeltPlan(iid, base)
        tier = min(k[0] for k, _ in options)
        return min((o for o in options if o[0][0] == tier), key=lambda o: o[0])[1]

    def smeltable(self, inv: Inventory, furnace: Furnace | None = None) -> dict[int, int]:
        """{input class id: items smelt_start could smelt now} for held inputs with enough fuel
        and room in this furnace: the smelt_start mask (``smelt_start`` returns OK exactly for these)."""
        out = {}
        for iid in inv.counts():
            if iid in self._smelt and inv.count_main(iid) > 0:
                p = self.plan_smelt(inv, iid, inv.count_main(iid), furnace)
                if p.n_items > 0:
                    out[iid] = p.n_items
        return out

    def smelt_start(self, inv: Inventory, furnace: Furnace, item: ItemRef, up_to: int) -> FurnaceResult:
        """Put up to ``up_to`` items and the chosen fuel into the furnace (``smelt {n} {item}.``).

        The mask is ``smeltable(inv, furnace)``: it allows the item exactly when this returns OK.
        Otherwise: ``MISSING_INGREDIENT`` when the agent lacks what it needs (the item, a recipe
        for it, or fuel for one item); ``INV_FULL`` with container ``furnace`` when the agent has
        it but the furnace's slots block it (the input slot holds another item or is full, the
        fuel slot holds a fuel the agent cannot add to, or the output slot is too full for the
        items inside). With that mask these failures happen only when the furnace changes during
        the walk to it (another player's doing). ValueError for a count below 1.
        """
        iid = self._id(item)
        if up_to < 1:
            raise ValueError(f"item count must be at least 1, got {up_to}")
        plan = self.plan_smelt(inv, iid, up_to, furnace)
        if plan.n_items == 0:
            code = self._smelt_failure(inv, furnace, iid)
            return FurnaceResult(code, iid, container="furnace" if code == INV_FULL else None)
        inv.remove(iid, plan.n_items)
        furnace.put_input(iid, plan.n_items)
        clicks = CLICKS_PER_FURNACE_STACK
        if plan.fuel_units:
            inv.remove(plan.fuel, plan.fuel_units)
            furnace.put_fuel(plan.fuel, plan.fuel_units)
            clicks += CLICKS_PER_FURNACE_STACK
        return FurnaceResult(OK, iid, plan.n_items, plan.fuel, plan.fuel_units, clicks)

    def _smelt_failure(self, inv: Inventory, furnace: Furnace, iid: int) -> str:
        """Why ``plan_smelt`` finds nothing to smelt: ``INV_FULL`` (the furnace's slots block it)
        or ``MISSING_INGREDIENT`` (the agent lacks the item, a recipe or the fuel)."""
        if iid not in self._smelt or inv.count_main(iid) == 0:
            return MISSING_INGREDIENT
        if furnace.input_count > 0 and (furnace.input != iid
                                        or furnace.input_count >= self.concepts.item_class(iid).stack_size):
            return INV_FULL
        held = inv.count_main(iid)
        for clear_output, clear_fuel in ((True, False), (False, True), (True, True)):
            probe = furnace.copy()
            if clear_output:
                probe.take_output()
            if clear_fuel:
                probe.take_fuel()
            if self.plan_smelt(inv, iid, held, probe).n_items > 0:
                return INV_FULL  # an emptier output or fuel slot would have let it smelt
        return MISSING_INGREDIENT

    def collect(self, inv: Inventory, furnace: Furnace) -> FurnaceResult:
        """Take the output (as much as fits), and an empty bucket left in the fuel slot [design].

        ``NO_TARGET`` when there is no output. ``INV_FULL`` (container ``inventory``) when the
        inventory fills before everything is taken, with ``n_items`` the number taken (0 if none
        fit); a full inventory ends an action with INV_FULL (DESIGN.md 4.5).
        """
        if furnace.output_count == 0:
            return FurnaceResult(NO_TARGET)
        item, n = furnace.output, furnace.output_count
        left = inv.add(item, n)
        furnace.output_count = left
        if left == 0:
            furnace.output = 0
        clicks = CLICKS_PER_FURNACE_STACK if left < n else 0
        bucket_stays = False
        if furnace.fuel_count and self.fuel_ticks(furnace.fuel) == 0:  # a bucket left by lava
            if inv.free_capacity(furnace.fuel) >= furnace.fuel_count:
                inv.add(*furnace.take_fuel())
                clicks += CLICKS_PER_FURNACE_STACK
            else:
                bucket_stays = True
        if left > 0 or bucket_stays:
            return FurnaceResult(INV_FULL, item, n - left, clicks=clicks, container="inventory")
        return FurnaceResult(OK, item, n - left, clicks=clicks)


@lru_cache(maxsize=1)
def default_book() -> RecipeBook:
    """The recipe book of the default concepts file (cached)."""
    return RecipeBook()


def craftable(inv: Inventory, near_table: bool) -> dict[int, int]:
    """``RecipeBook.craftable`` with the default book (contracts 5)."""
    return default_book().craftable(inv, near_table)


def craft(inv: Inventory, item: ItemRef, up_to_items: int, near_table: bool = False) -> CraftResult:
    """``RecipeBook.craft`` with the default book (contracts 5)."""
    return default_book().craft(inv, item, up_to_items, near_table)


def smeltable(inv: Inventory, furnace: Furnace | None = None) -> dict[int, int]:
    """``RecipeBook.smeltable`` with the default book."""
    return default_book().smeltable(inv, furnace)


# =============================================================================
# Furnace
# =============================================================================


@dataclass
class Furnace:
    """One furnace's slots and timers, advanced by ``tick`` (module docstring).

    The fields match the world's furnace slot table (``World.furnaces``): ``as_fields`` and
    ``from_fields`` convert. ``burn_left`` is the game's litTime, ``burn_total`` litDuration,
    ``cook_ticks`` cookingProgress; ``cook_total`` (cookingTotalTime) follows the input.
    """

    input: int = 0
    input_count: int = 0
    fuel: int = 0
    fuel_count: int = 0
    output: int = 0
    output_count: int = 0
    burn_left: int = 0
    burn_total: int = 0
    cook_ticks: int = 0
    cook_total: int = field(default_factory=mechanics.smelt_ticks)
    book: RecipeBook = field(default_factory=default_book, compare=False, repr=False)

    FIELDS = ("input", "input_count", "fuel", "fuel_count", "output", "output_count", "burn_left", "burn_total",
              "cook_ticks")

    # -- state --------------------------------------------------------------------------

    @property
    def lit(self) -> bool:
        return self.burn_left > 0

    def as_fields(self) -> dict[str, int]:
        """The world's furnace slot fields."""
        return {k: getattr(self, k) for k in self.FIELDS}

    @classmethod
    def from_fields(cls, values: Mapping[str, Any], book: RecipeBook | None = None) -> Furnace:
        book = book if book is not None else default_book()
        f = cls(**{k: int(values[k]) for k in cls.FIELDS}, book=book)
        f.cook_total = book.cook_ticks(f.input)
        return f

    def copy(self) -> Furnace:
        return replace(self)

    def state(self) -> tuple[int, ...]:
        """All counters as a tuple (for comparisons and replays)."""
        return tuple(getattr(self, f.name) for f in fields(self) if f.name != "book")

    def _result(self) -> tuple[int, int] | None:
        rec = self.book.smelt_recipe(self.input) if self.input_count > 0 else None
        return None if rec is None else (rec.result, rec.count)

    def can_burn(self) -> bool:
        """``canBurn``: a recipe for the input and room for its result in the output slot."""
        res = self._result()
        if res is None:
            return False
        if self.output_count == 0:
            return True
        size = self.book.concepts.item_class(res[0]).stack_size
        return self.output == res[0] and self.output_count + res[1] <= size

    # -- slots ------------------------------------------------------------------------------

    def put_input(self, item: ItemRef, n: int) -> int:
        """Add up to ``n`` items to the input slot (empty or the same class); returns how many went in.
        A new input class restarts the cook progress (``setItem`` on slot 0 [mem])."""
        iid = self.book._id(item)
        if self.input_count and self.input != iid:
            return 0
        put = min(n, self.book.concepts.item_class(iid).stack_size - self.input_count)
        if put <= 0:
            return 0
        if self.input_count == 0:
            self.cook_total = self.book.cook_ticks(iid)
            self.cook_ticks = 0
        self.input, self.input_count = iid, self.input_count + put
        return put

    def put_fuel(self, item: ItemRef, n: int) -> int:
        """Add up to ``n`` items to the fuel slot (empty or the same class); returns how many went in."""
        iid = self.book._id(item)
        if self.fuel_count and self.fuel != iid:
            return 0
        put = min(n, self.book.concepts.item_class(iid).stack_size - self.fuel_count)
        if put <= 0:
            return 0
        self.fuel, self.fuel_count = iid, self.fuel_count + put
        return put

    def take_output(self) -> tuple[int, int]:
        out = (self.output, self.output_count)
        self.output, self.output_count = 0, 0
        return out

    def take_input(self) -> tuple[int, int]:
        out = (self.input, self.input_count)
        self.input, self.input_count = 0, 0
        return out

    def take_fuel(self) -> tuple[int, int]:
        out = (self.fuel, self.fuel_count)
        self.fuel, self.fuel_count = 0, 0
        return out

    # -- ticking ----------------------------------------------------------------------------

    def step(self) -> None:
        """Exactly one game tick of ``AbstractFurnaceBlockEntity.serverTick`` [mem]."""
        if self.burn_left > 0:
            self.burn_left -= 1
        has_input = self.input_count > 0
        has_fuel = self.fuel_count > 0
        if self.burn_left > 0 or (has_fuel and has_input):
            burnable = self.can_burn()
            if self.burn_left == 0 and burnable:
                self.burn_left = self.burn_total = self.book.fuel_ticks(self.fuel) if has_fuel else 0
                if self.burn_left > 0:
                    self.fuel_count -= 1
                    if self.fuel_count == 0:
                        rem = self.book.fuel_remainder(self.fuel)
                        self.fuel, self.fuel_count = rem, 1 if rem else 0
            if self.burn_left > 0 and burnable:
                self.cook_ticks += 1
                if self.cook_ticks == self.cook_total:
                    self.cook_ticks = 0
                    self.cook_total = self.book.cook_ticks(self.input)  # computed before the burn [mem]
                    self._burn()
            else:
                self.cook_ticks = 0
        elif self.burn_left == 0 and self.cook_ticks > 0:
            self.cook_ticks = min(max(self.cook_ticks - 2, 0), self.cook_total)

    def _burn(self) -> None:
        """One input item becomes the recipe's result in the output slot."""
        result, count = self._result()
        if self.output_count == 0:
            self.output = result
        self.output_count += count
        self.input_count -= 1
        if self.input_count == 0:
            self.input = 0

    def _quiet_ticks(self) -> int:
        """How many coming ticks change only the counters linearly (0 when the next is an event;
        ``IDLE`` when nothing will happen until a slot changes)."""
        if self.burn_left > 0:
            if self.can_burn():  # cooking: the next event is an item finishing or the fuel running out
                return max(0, min(self.cook_total - self.cook_ticks, self.burn_left) - 1)
            return self.burn_left - 1  # burning without cooking: progress stays 0
        if (self.fuel_count > 0 and self.input_count > 0 and self.can_burn()
                and self.book.fuel_ticks(self.fuel) > 0):
            return 0  # the next tick lights the furnace
        return IDLE

    def tick(self, n: int = 1) -> None:
        """Advance ``n`` ticks; the same result as ``n`` calls of ``step`` (tested)."""
        if n < 0:
            raise ValueError(f"cannot tick backwards ({n})")
        while n > 0:
            k = min(self._quiet_ticks(), n)
            if k == 0:
                self.step()
                n -= 1
                continue
            if self.burn_left > 0:
                cooking = self.can_burn()
                self.burn_left -= k
                self.cook_ticks = self.cook_ticks + k if cooking else 0
            elif self.fuel_count > 0 and self.input_count > 0:
                # fuel and input but nothing burns (full output, or a bucket in the fuel slot): progress
                # resets; a relight attempt with a non-fuel item sets litDuration to its 0 burn time
                if self.can_burn():
                    self.burn_total = 0
                self.cook_ticks = 0
            else:
                self.cook_ticks = min(max(self.cook_ticks - 2 * k, 0), self.cook_total)  # cooling
            n -= k

    def next_output_ticks(self, limit: int = 1 << 20) -> int | None:
        """Ticks until the output slot next changes, None if it will not (without new items)."""
        f = self.copy()
        start, t = (f.output, f.output_count), 0
        while t < limit:
            q = f._quiet_ticks()
            if q >= IDLE:
                return None
            k = min(q, limit - t - 1)
            f.tick(k)  # quiet ticks never touch the output
            f.step()
            t += k + 1
            if (f.output, f.output_count) != start:
                return t
        return None

    def finish_ticks(self, limit: int = 1 << 20) -> int | None:
        """Ticks until the input slot is empty (everything smelted), None if it will not empty."""
        if self.input_count == 0:
            return 0
        f = self.copy()
        t = 0
        while f.input_count > 0:
            k = f.next_output_ticks(limit - t)
            if k is None:
                return None
            f.tick(k)
            t += k
        return t

"""Tests for hivemind.sim.ref.crafting: class-level recipes, "up to" crafting, smelting and the furnace."""

from __future__ import annotations

import pytest

from hivemind.rules import load_concepts
from hivemind.sim.ref.crafting import (
    INV_FULL,
    MISSING_INGREDIENT,
    NO_STATION,
    NO_TARGET,
    OK,
    Furnace,
    RecipeBook,
    craft,
    craftable,
    default_book,
    smeltable,
)
from hivemind.sim.ref.inventory import MAIN_SIZE, Inventory, Stack
from hivemind.sim.ref.rng import CounterRNG, Purpose

C = load_concepts()
BOOK = default_book()


def iid(name: str) -> int:
    return C.item_class(name).id


def inv_with(**items: int) -> Inventory:
    inv = Inventory()
    for name, n in items.items():
        assert inv.add(name, n) == 0
    return inv


def names(d: dict[int, int]) -> dict[str, int]:
    return {C.item_class(k).name: v for k, v in d.items()}


# -- the book -----------------------------------------------------------------------------


def test_book_uses_crafting_and_furnace_recipes_only():
    assert BOOK.crafting and BOOK.smelting
    assert all(len(r.inputs) >= 1 and r.cook_ticks == 200 for r in BOOK.smelting)  # [20]
    n_smelting = sum(1 for r in C.recipes if r["type"] == "smelting")
    assert len(BOOK.smelting) == n_smelting  # smoking, blasting, campfire recipes are left out
    assert BOOK.smelt_recipe("raw_iron").result == iid("iron_ingot")
    assert BOOK.smelt_recipe("beef").result == iid("cooked_beef")
    assert BOOK.smelt_recipe("stick") is None


def test_needs_table():
    for item in ("planks", "stick", "crafting_table", "torch", "shears", "flint_and_steel"):
        assert not BOOK.needs_table(item), item
    for item in ("wooden_pickaxe", "furnace", "chest", "bed", "bucket", "shield", "bow", "bread"):
        assert BOOK.needs_table(item), item
    with pytest.raises(KeyError):
        BOOK.needs_table("diamond")


# -- crafting ---------------------------------------------------------------------------------


def test_planks_from_log():
    inv = inv_with(log=1)
    r = craft(inv, "planks", 4)
    assert (r.code, r.n_crafts, r.n_items) == (OK, 1, 4)
    assert inv.counts() == {iid("planks"): 4}


def test_sticks_and_crafting_table():
    inv = inv_with(planks=6)
    r = craft(inv, "stick", 4)
    assert (r.code, r.n_items, r.n_ingredients, r.clicks) == (OK, 4, 2, 6)
    r = craft(inv, "crafting_table", 1)
    assert (r.code, r.n_items, r.n_ingredients, r.clicks) == (OK, 1, 4, 8)
    assert inv.counts() == {iid("stick"): 4, iid("crafting_table"): 1}


TOOL_SHAPES = {"pickaxe": (3, 2), "axe": (3, 2), "shovel": (1, 2), "hoe": (2, 2), "sword": (2, 1)}
TOOL_MATERIALS = {"wooden": "planks", "stone": "cobblestone", "iron": "iron_ingot", "golden": "gold_ingot",
                  "diamond": "diamond"}


@pytest.mark.parametrize("tier", list(TOOL_MATERIALS))
@pytest.mark.parametrize("kind", list(TOOL_SHAPES))
def test_every_v1_tool(tier: str, kind: str):
    material, (n_mat, n_stick) = TOOL_MATERIALS[tier], TOOL_SHAPES[kind]
    tool = f"{tier}_{kind}"
    inv = inv_with(**{material: n_mat, "stick": n_stick})
    assert craft(inv, tool, 1).code == NO_STATION  # 3 tall: needs a table
    r = craft(inv, tool, 1, near_table=True)
    assert (r.code, r.n_items) == (OK, 1)
    assert names(r.consumed) == {material: n_mat, "stick": n_stick}
    assert inv.counts() == {iid(tool): 1}
    assert inv.slots[0].durability == C.item_class(tool).max_damage


def test_wooden_pickaxe_clicks_match_the_duration_model():
    inv = inv_with(planks=3, stick=2)
    r = craft(inv, "wooden_pickaxe", 1, near_table=True)
    assert 8 <= r.clicks <= 10  # DESIGN.md 5: about 8-10 clicks for a pickaxe [est]
    assert r.duration_ticks(2.0) == r.clicks * 2.0 + 2


@pytest.mark.parametrize("item, ingredients, table", [
    ("shears", {"iron_ingot": 2}, False),
    ("flint_and_steel", {"flint": 1, "iron_ingot": 1}, False),
    ("bucket", {"iron_ingot": 3}, True),
    ("shield", {"planks": 6, "iron_ingot": 1}, True),
    ("bow", {"stick": 3, "string": 3}, True),
    ("chest", {"planks": 8}, True),
])
def test_other_v1_tools_and_stations(item: str, ingredients: dict[str, int], table: bool):
    inv = inv_with(**ingredients)
    r = craft(inv, item, 1, near_table=table)
    assert r.code == OK and inv.counts() == {iid(item): 1}
    assert names(r.consumed) == ingredients


@pytest.mark.parametrize("fuel", ["coal", "charcoal"])
def test_torch_from_coal_or_charcoal(fuel: str):
    inv = inv_with(**{fuel: 1, "stick": 1})
    r = craft(inv, "torch", 4)
    assert (r.code, r.n_items) == (OK, 4) and inv.counts() == {iid("torch"): 4}
    assert not r.recipe.needs_table


def test_torch_prefers_the_class_held_most_and_mixes_when_needed():
    inv = inv_with(coal=1, charcoal=2, stick=5)
    r = craft(inv, "torch", 64)
    assert (r.n_crafts, r.n_items) == (3, 12)
    assert names(r.consumed) == {"coal": 1, "charcoal": 2, "stick": 3}
    inv = inv_with(coal=3, charcoal=2, stick=1)
    craft(inv, "torch", 4)
    assert inv.count("coal") == 2 and inv.count("charcoal") == 2  # coal was held most


def test_furnace_from_cobblestone_or_mixed_with_deepslate():
    inv = inv_with(cobblestone=8)
    assert craft(inv, "furnace", 1, near_table=True).code == OK
    inv = inv_with(cobblestone=5, cobbled_deepslate=3)
    r = craft(inv, "furnace", 1, near_table=True)
    assert r.code == OK and names(r.consumed) == {"cobblestone": 5, "cobbled_deepslate": 3}
    assert r.clicks == 8 + 2 + 3  # two pickups: one per class
    assert craft(inv_with(cobblestone=7), "furnace", 1, near_table=True).code == MISSING_INGREDIENT


def test_bed_from_any_three_wool():
    # Known approximation (concepts exceptions): any three wool make a bed; the game needs one colour.
    assert any(e["class"] == "bed" and e["check"] == "correlated slots" for e in C.exceptions)
    inv = inv_with(wool=3, planks=3)
    r = craft(inv, "bed", 1, near_table=True)
    assert r.code == OK and inv.counts() == {iid("bed"): 1}


def test_up_to_counts():
    inv = inv_with(planks=10)
    r = craft(inv, "stick", 6)  # one craft makes 4; two would make 8 > 6
    assert (r.n_crafts, r.n_items) == (1, 4)
    r = craft(inv, "stick", 64)  # as many as possible: 8 planks left -> 4 crafts
    assert (r.n_crafts, r.n_items) == (4, 16)
    inv = inv_with(log=3)
    r = craft(inv, "planks", 1)  # at least one craft
    assert (r.n_crafts, r.n_items) == (1, 4)
    r = craft(inv, "planks", 64)
    assert (r.n_crafts, r.n_items) == (2, 8) and inv.count("planks") == 12
    with pytest.raises(ValueError):
        craft(inv, "stick", 0)


def test_missing_ingredient_and_no_station():
    inv = inv_with(planks=1)
    before = inv.snapshot()
    assert craft(inv, "stick", 4).code == MISSING_INGREDIENT
    assert craft(inv, "diamond", 1).code == MISSING_INGREDIENT  # no recipe at all
    inv = inv_with(planks=8)
    r = craft(inv, "chest", 1, near_table=False)
    assert r.code == NO_STATION and r.n_crafts == 0
    assert inv_with(planks=1).snapshot() == before


def test_craftable_respects_the_table():
    inv = inv_with(planks=9, stick=2)
    small = names(craftable(inv, near_table=False))
    big = names(craftable(inv, near_table=True))
    assert small["crafting_table"] == 2 and small["stick"] == 4
    assert "wooden_pickaxe" not in small and "chest" not in small
    assert big["wooden_pickaxe"] == 1 and big["chest"] == 1 and big["wooden_sword"] == 2
    assert set(small) < set(big)
    assert all(n > 0 for n in big.values())


def test_full_inventory_stops_the_craft():
    inv = Inventory()
    for i in range(MAIN_SIZE - 1):
        inv.slots[i] = Stack(iid("wooden_sword"), 1, 59)
    inv.slots[MAIN_SIZE - 1] = Stack(iid("planks"), 64)
    before = inv.snapshot()
    r = craft(inv, "stick", 4)
    assert (r.code, r.container, r.n_crafts) == (INV_FULL, "inventory", 0) and inv.snapshot() == before
    inv.slots[MAIN_SIZE - 1] = Stack(iid("planks"), 4)
    assert craft(inv, "crafting_table", 1).code == OK  # the planks' slot frees up


def test_a_craft_stopped_by_a_full_inventory_reports_it():
    # The reviewer's case: room for one craft of sticks only. The result is "inventory full", with
    # the one craft that was made, not "done".
    inv = Inventory()
    for i in range(MAIN_SIZE - 2):
        inv.slots[i] = Stack(iid("wooden_sword"), 1, 59)
    inv.slots[MAIN_SIZE - 2] = Stack(iid("planks"), 8)
    inv.slots[MAIN_SIZE - 1] = Stack(iid("stick"), 60)
    r = craft(inv, "stick", 64)
    assert (r.code, r.container, r.n_crafts, r.n_items) == (INV_FULL, "inventory", 1, 4)
    assert inv.count("stick") == 64 and inv.count("planks") == 6 and names(r.consumed) == {"planks": 2}
    assert craft(inv_with(planks=8), "stick", 64).container is None  # OK carries no container


def test_collect_reports_a_full_inventory():
    inv = Inventory()
    for i in range(MAIN_SIZE - 1):
        inv.slots[i] = Stack(iid("wooden_sword"), 1, 59)
    inv.slots[MAIN_SIZE - 1] = Stack(iid("iron_ingot"), 61)
    f = Furnace(output=iid("iron_ingot"), output_count=8)
    r = BOOK.collect(inv, f)
    assert (r.code, r.container, r.n_items, r.clicks) == (INV_FULL, "inventory", 3, 3)
    assert f.output_count == 5 and inv.count("iron_ingot") == 64
    r = BOOK.collect(inv, f)
    assert (r.code, r.n_items, r.clicks) == (INV_FULL, 0, 0) and f.output_count == 5
    # all the output fits but the bucket left by lava does not: the inventory is full
    inv.slots[MAIN_SIZE - 1] = None
    f = Furnace(output=iid("iron_ingot"), output_count=5, fuel=iid("bucket"), fuel_count=1)
    r = BOOK.collect(inv, f)
    assert (r.code, r.n_items, f.output_count, f.fuel) == (INV_FULL, 5, 0, iid("bucket"))


def test_recipe_choice_prefers_the_item_held_most():
    recipes = [
        {"type": "crafting_shaped", "result": "torch", "count": 4, "pattern": ["A", "B"],
         "key": {"A": ["coal"], "B": ["stick"]}, "sources": ["torch_coal"]},
        {"type": "crafting_shaped", "result": "torch", "count": 4, "pattern": ["A", "B"],
         "key": {"A": ["charcoal"], "B": ["stick"]}, "sources": ["torch_charcoal"]},
    ]
    book = RecipeBook(recipes=recipes)
    inv = inv_with(coal=2, charcoal=3, stick=2)
    code, rec = book.choose_recipe(inv, "torch", near_table=False)
    assert code == OK and rec.sources == ("torch_charcoal",)
    r = book.craft(inv, "torch", 64)
    assert names(r.consumed) == {"charcoal": 2, "stick": 2}  # one recipe per command
    inv = inv_with(coal=1, charcoal=1, stick=5)  # tie on the most-held item: book order
    assert book.choose_recipe(inv, "torch", False)[1].sources == ("torch_charcoal",)


def test_patterns_are_shrunk_before_fitting_the_grid():
    def shaped(pattern: list[str]) -> dict:
        return {"type": "crafting_shaped", "result": "stick", "count": 4, "pattern": pattern,
                "key": {"A": ["planks"]}, "sources": ["x"]}

    book = RecipeBook(recipes=[shaped([" A ", " A ", "   "])])
    (rec,) = book.crafting
    assert (rec.width, rec.height, rec.needs_table) == (1, 2, False)
    (rec,) = RecipeBook(recipes=[shaped(["A  ", "A  ", "A  "])]).crafting
    assert (rec.width, rec.height, rec.needs_table) == (1, 3, True)
    (bow,) = BOOK.recipes_for("bow")
    assert (bow.width, bow.height) == (3, 3)


def test_overlapping_slot_sets_are_rejected():
    bad = [{"type": "crafting_shapeless", "result": "torch", "count": 1,
            "ingredients": [["coal", "charcoal"], ["coal"]], "sources": ["x"]}]
    with pytest.raises(ValueError):
        RecipeBook(recipes=bad)


# -- furnace --------------------------------------------------------------------------------


def test_eight_raw_iron_with_one_coal_finish_at_tick_1600():
    inv = inv_with(raw_iron=8, coal=1)
    f = Furnace()
    r = BOOK.smelt_start(inv, f, "raw_iron", 8)
    assert (r.code, r.n_items, r.fuel, r.fuel_units, r.clicks) == (OK, 8, iid("coal"), 1, 6)
    assert inv.counts() == {}
    assert f.next_output_ticks() == 200 and f.finish_ticks() == 1600
    f.tick(200)
    assert f.output_count == 1
    f.tick(1399)
    assert f.output_count == 7 and f.input_count == 1
    f.tick(1)
    assert (f.output, f.output_count, f.input_count) == (iid("iron_ingot"), 8, 0)
    assert f.burn_left == 1 and f.fuel_count == 0
    f.tick(1)
    assert not f.lit
    c = BOOK.collect(inv, f)
    assert (c.code, c.n_items) == (OK, 8) and inv.counts() == {iid("iron_ingot"): 8}
    assert BOOK.collect(inv, f).code == NO_TARGET


def test_partial_fuel_carries_over_between_items():
    f = Furnace()
    f.put_input("raw_iron", 3)
    f.put_fuel("planks", 2)  # 300 ticks each: 1.5 items per plank
    f.tick(200)
    assert f.output_count == 1
    f.tick(100)
    assert f.cook_ticks == 100 and f.burn_left == 1 and f.fuel_count == 1
    f.tick(1)  # the first plank is out; the second lights and progress continues
    assert f.fuel_count == 0 and f.burn_left == 300 and f.cook_ticks == 101
    f.tick(99)
    assert f.output_count == 2
    f.tick(199)
    assert f.output_count == 2
    f.tick(1)
    assert f.output_count == 3 and f.burn_left == 1  # 2 planks smelt 3 items, all at tick 600


def test_fuel_burns_on_without_input_and_progress_cools_without_fuel():
    f = Furnace()
    f.put_input("raw_iron", 2)
    f.put_fuel("coal", 1)
    f.tick(400)
    assert f.output_count == 2 and f.burn_left == 1201  # lit on tick 1 with 1600, then -1 per tick
    f.tick(500)  # no input: the coal keeps burning
    assert f.burn_left == 701 and f.cook_ticks == 0
    g = Furnace()
    g.put_input("raw_iron", 1)
    g.put_fuel("stick", 1)  # 100 ticks: half an item
    g.tick(100)
    assert g.cook_ticks == 100
    g.tick(1)
    assert g.cook_ticks == 98 and not g.lit  # cools by 2 per tick
    g.tick(49)
    assert g.cook_ticks == 0 and g.output_count == 0
    assert g.next_output_ticks() is None and g.finish_ticks() is None


def test_lava_leaves_a_bucket_that_collect_returns():
    inv = inv_with(raw_iron=1, lava_bucket=1)
    f = Furnace()
    assert BOOK.smelt_start(inv, f, "raw_iron", 1).fuel == iid("lava_bucket")
    f.tick(1)
    assert (f.fuel, f.fuel_count, f.burn_left) == (iid("bucket"), 1, 20000)
    f.tick(199)
    c = BOOK.collect(inv, f)
    assert c.code == OK and inv.counts() == {iid("bucket"): 1, iid("iron_ingot"): 1}
    assert f.fuel_count == 0


def test_full_output_stalls_the_furnace():
    f = Furnace(output=iid("iron_ingot"), output_count=64)
    f.put_input("raw_iron", 1)
    f.put_fuel("coal", 1)
    f.tick(1000)
    assert (f.fuel_count, f.input_count, f.cook_ticks, f.burn_left) == (1, 1, 0, 0)
    assert f.next_output_ticks() is None
    f.take_output()
    f.tick(200)
    assert f.output_count == 1


def test_tick_jumps_equal_single_steps():
    rng = CounterRNG(5)
    inputs = ("raw_iron", "log", "sand", "beef", "stick", "iron_pickaxe")
    fuels = ("coal", "planks", "stick", "lava_bucket", "wooden_pickaxe", "bucket")
    for trial in range(150):
        s = rng.stream(0, trial, Purpose.GENERIC)
        a, b = Furnace(), Furnace()
        for _ in range(5):
            op = s.below(4)
            if op == 0:
                item, n = inputs[s.below(len(inputs))], s.randint(1, 20)
                a.put_input(item, n), b.put_input(item, n)
            elif op == 1:
                item, n = fuels[s.below(len(fuels))], s.randint(1, 3)
                a.put_fuel(item, n), b.put_fuel(item, n)
            elif op == 2:
                a.take_output(), b.take_output()
            ticks = s.randint(0, 700)
            a.tick(ticks)
            for _ in range(ticks):
                b.step()
            assert a == b, (trial, a, b)


def test_furnace_fields_match_the_world_slot_table():
    from hivemind.sim.ref.world import World

    assert set(Furnace.FIELDS) <= set(World(16, 16).furnaces.fields)
    f = Furnace()
    f.put_input("raw_iron", 5)
    f.put_fuel("coal", 2)
    f.tick(333)
    g = Furnace.from_fields(f.as_fields())
    assert g == f


# -- smelt_start and the fuel rule ------------------------------------------------------------------


def plan(inv: Inventory, item: str, n: int, furnace: Furnace | None = None) -> tuple[str, int, int]:
    p = BOOK.plan_smelt(inv, item, n, furnace)
    return (C.item_class(p.fuel).name if p.fuel else "", p.fuel_units, p.n_items)


def test_fuel_choice_least_waste():
    assert plan(inv_with(raw_iron=8, coal=1, planks=5), "raw_iron", 8) == ("coal", 1, 8)  # planks cover 7
    assert plan(inv_with(raw_iron=2, coal=1, planks=5), "raw_iron", 2) == ("planks", 2, 2)  # waste 200 < 1400
    assert plan(inv_with(raw_iron=1, coal=1, stick=4), "raw_iron", 1) == ("stick", 2, 1)  # waste 0
    assert plan(inv_with(raw_iron=1, stick=1), "raw_iron", 1) == ("", 0, 0)  # half an item is not an item


def test_fuel_choice_prefers_plain_fuel_over_stations_over_tools():
    assert plan(inv_with(raw_iron=1, planks=1, crafting_table=1), "raw_iron", 1) == ("planks", 1, 1)
    assert plan(inv_with(raw_iron=1, crafting_table=1, wooden_pickaxe=1), "raw_iron", 1) == ("crafting_table", 1, 1)
    assert plan(inv_with(raw_iron=1, wooden_pickaxe=1), "raw_iron", 1) == ("wooden_pickaxe", 1, 1)
    assert plan(inv_with(raw_iron=2, wooden_pickaxe=1, coal=1), "raw_iron", 2) == ("coal", 1, 2)


def test_logs_fuel_their_own_charcoal():
    inv = inv_with(log=5)
    r = BOOK.smelt_start(inv, f := Furnace(), "log", 64)
    assert (r.n_items, r.fuel, r.fuel_units) == (3, iid("log"), 2)  # 3 + ceil(600 / 300) = 5
    assert inv.counts() == {}
    f.tick(600)
    assert (f.output, f.output_count) == (iid("charcoal"), 3)


def test_smelt_up_to_what_the_fuel_covers():
    inv = inv_with(raw_iron=10, coal=1)
    r = BOOK.smelt_start(inv, Furnace(), "raw_iron", 64)
    assert (r.code, r.n_items) == (OK, 8) and inv.count("raw_iron") == 2


def test_existing_heat_is_used_first():
    # A lit furnace with burn time B left cooks B - 1 more ticks (step() lowers it before cooking):
    # 1001 ticks smelt 5 items, 1000 only 4, so a fifth item needs fuel.
    f = Furnace(burn_left=1001)
    inv = inv_with(raw_iron=5, coal=1)
    r = BOOK.smelt_start(inv, f, "raw_iron", 5)
    assert (r.n_items, r.fuel_units, r.clicks) == (5, 0, 3) and inv.count("coal") == 1
    assert f.finish_ticks() == 1000
    f = Furnace(burn_left=1000)
    assert plan(inv_with(raw_iron=5), "raw_iron", 5, f) == ("", 0, 4)
    inv = inv_with(raw_iron=5, coal=1)
    assert BOOK.smelt_start(inv, f, "raw_iron", 5).fuel_units == 1
    assert f.finish_ticks() == 1000  # 999 ticks of the old fire, then the coal lights and cooks
    f2 = Furnace()
    f2.put_input("raw_iron", 2)
    f2.put_fuel("planks", 1)
    inv = inv_with(raw_iron=4, planks=3, coal=1)
    assert plan(inv, "raw_iron", 4, f2) == ("planks", 3, 4)  # same fuel class as the slot holds


def test_burn_time_left_is_counted_exactly():
    # Every burn time left, with an empty furnace: the plan's items are exactly what the fire smelts.
    for burn_left in range(0, 1300):
        f = Furnace(burn_left=burn_left)
        inv = inv_with(raw_iron=64)
        r = BOOK.smelt_start(inv, f, "raw_iron", 64)
        expected = max(burn_left - 1, 0) // 200
        assert r.n_items == expected and r.fuel_units == 0, burn_left
        if expected:
            assert r.code == OK and f.finish_ticks() == expected * 200, burn_left


def test_lit_furnace_after_a_collect_does_not_strand_an_item():
    # The reviewer's sequence: 1 raw iron with 1 coal, collect at tick 201 (1400 ticks of fire left),
    # then 7 more: the fire covers only 6, so one coal is added and all 7 smelt.
    inv = inv_with(raw_iron=8, coal=2)
    f = Furnace()
    BOOK.smelt_start(inv, f, "raw_iron", 1)
    f.tick(201)
    assert BOOK.collect(inv, f).n_items == 1 and f.burn_left == 1400
    r = BOOK.smelt_start(inv, f, "raw_iron", 7)
    assert (r.code, r.n_items, r.fuel_units) == (OK, 7, 1)
    f.tick(20000)
    assert (f.output_count, f.input_count) == (7, 0)


def test_blocked_output_slot_waits_for_a_collect():
    # Iron ingots sit in the output; sand cannot cook until they are taken, and the fire left
    # burns out meanwhile, so the plan does not count it.
    inv = inv_with(raw_iron=1, coal=2, sand=7)
    f = Furnace()
    BOOK.smelt_start(inv, f, "raw_iron", 1)
    f.tick(200)
    assert (f.output, f.output_count, f.burn_left) == (iid("iron_ingot"), 1, 1401)
    r = BOOK.smelt_start(inv, f, "sand", 7)
    assert (r.code, r.n_items, r.fuel, r.fuel_units) == (OK, 7, iid("coal"), 1)
    f.tick(1500)
    assert (f.output, f.input_count, f.fuel_count, f.lit) == (iid("iron_ingot"), 7, 1, False)
    assert f.finish_ticks() is None
    assert BOOK.collect(inv, f).code == OK
    assert f.finish_ticks() == 1400
    f.tick(1400)
    assert (f.output, f.output_count, f.input_count) == (iid("glass"), 7, 0)
    # without fuel held, the output slot is what blocks: the furnace is full, nothing is missing
    g = Furnace(output=iid("iron_ingot"), output_count=1, burn_left=1401)
    r = BOOK.smelt_start(inv_with(sand=7), g, "sand", 7)
    assert (r.code, r.container) == (INV_FULL, "furnace")


def test_items_are_capped_by_the_room_in_the_output_slot():
    f = Furnace(output=iid("iron_ingot"), output_count=60)
    inv = inv_with(raw_iron=8, coal=1)
    r = BOOK.smelt_start(inv, f, "raw_iron", 8)
    assert (r.code, r.n_items) == (OK, 4) and inv.count("raw_iron") == 4
    f.tick(800)
    assert (f.output_count, f.input_count) == (64, 0)
    g = Furnace(output=iid("iron_ingot"), output_count=62)
    g.put_input("raw_iron", 2)  # the output fills exactly with the two inside
    r = BOOK.smelt_start(inv_with(raw_iron=1, coal=1), g, "raw_iron", 1)
    assert (r.code, r.container) == (INV_FULL, "furnace")


def test_smelt_failures():
    inv = inv_with(raw_iron=3)
    before = inv.snapshot()
    assert BOOK.smelt_start(inv, Furnace(), "raw_iron", 3).code == MISSING_INGREDIENT  # no fuel
    assert BOOK.smelt_start(inv_with(coal=1), Furnace(), "raw_iron", 3).code == MISSING_INGREDIENT  # no input
    assert BOOK.smelt_start(inv_with(stick=3, coal=1), Furnace(), "stick", 3).code == MISSING_INGREDIENT  # no recipe
    assert inv.snapshot() == before
    with pytest.raises(ValueError):
        BOOK.smelt_start(inv_with(raw_iron=1, coal=1), Furnace(), "raw_iron", 0)
    # The agent holds input and fuel, but the furnace's slots block it: "furnace full", not
    # "missing ingredient".
    held = inv_with(raw_iron=8, coal=4)
    busy = Furnace()
    busy.put_input("sand", 4)  # another item in the input slot
    full = Furnace()
    full.put_input("raw_iron", 64)  # a full input slot
    stick = Furnace()
    stick.put_fuel("stick", 1)  # another fuel in the fuel slot, worth half an item
    for f in (busy, full, stick):
        before = (held.snapshot(), f.copy())
        r = BOOK.smelt_start(held, f, "raw_iron", 8)
        assert (r.code, r.container, r.n_items) == (INV_FULL, "furnace", 0)
        assert (held.snapshot(), f) == before


def test_smeltable():
    assert names(smeltable(inv_with(raw_iron=10, coal=1))) == {"raw_iron": 8}
    assert smeltable(inv_with(raw_iron=10)) == {}
    assert names(smeltable(inv_with(sand=3, raw_iron=2, coal=1))) == {"raw_iron": 2, "sand": 3}


SMELT_INPUTS = ("raw_iron", "sand", "log", "beef", "cobblestone", "raw_gold")
SMELT_FUELS = ("coal", "charcoal", "planks", "stick", "log", "lava_bucket", "wooden_pickaxe", "crafting_table",
               "bucket")
OUTPUTS = ("iron_ingot", "glass", "charcoal", "cooked_beef", "stone")


def random_furnace(s) -> Furnace:
    """A furnace in a state reached by play: some loads, some ticks, some collects, sometimes an
    output of another item."""
    start = s.below(3)
    if start == 0:
        f = Furnace(output=iid(OUTPUTS[s.below(len(OUTPUTS))]), output_count=s.randint(1, 64))
    elif start == 1:  # a small batch just finished: often still lit, with its output waiting
        f = Furnace()
        k = s.randint(1, 6)
        f.put_input(SMELT_INPUTS[s.below(len(SMELT_INPUTS))], k)
        f.put_fuel(("coal", "planks", "lava_bucket")[s.below(3)], 2)
        f.tick(200 * k + s.randint(0, 400))
        return f
    else:
        f = Furnace()
    for _ in range(s.randint(0, 3)):
        op = s.below(5)
        if op == 0:  # mostly small loads, so many finish and leave a lit furnace with an output
            f.put_input(SMELT_INPUTS[s.below(len(SMELT_INPUTS))], s.randint(1, 64 if s.below(4) == 0 else 6))
        elif op == 1:
            f.put_fuel(SMELT_FUELS[s.below(len(SMELT_FUELS))], s.randint(1, 4))
        elif op == 2 and s.below(2):
            f.take_output()
        elif op == 3:
            f.take_input()
        f.tick(s.randint(0, 2000))
    return f


def random_held(s) -> Inventory:
    inv = Inventory()
    for _ in range(s.randint(1, 5)):
        name = (*SMELT_INPUTS, *SMELT_FUELS)[s.below(len(SMELT_INPUTS) + len(SMELT_FUELS))]
        inv.add(name, s.randint(1, 20) if C.item_class(name).stack_size > 1 else 1)
    return inv


def run_until_empty(f: Furnace, s, early_collects: bool) -> bool:
    """Tick the furnace, taking the output only when it blocks the furnace and only after the fire
    left has burned out (the worst case), or also at random moments; True if the input slot
    empties without adding anything."""
    for _ in range(5000):  # random partial ticks take about log2(200) steps per item
        if f.input_count == 0:
            return True
        k = f.next_output_ticks()
        if k is None:
            if f.output_count == 0:
                return False  # stuck with input left and nothing to collect
            f.tick(f.burn_left + 1)  # nothing changes but the fire burning out
            f.take_output()
            continue
        f.tick(s.randint(1, k) if early_collects else k)
        if early_collects and s.below(3) == 0:
            f.take_output()
    return False


def test_every_smelt_start_that_returns_ok_smelts_all_its_items():
    """Property (review finding): no plan strands an item. Covers lit and cold furnaces, cook
    progress, fuel in the slot, items inside, and output slots that are partly full or hold
    another item; the agent collects only when the output blocks the furnace."""
    rng = CounterRNG(21)
    n_ok = n_full = 0
    for trial in range(1500):
        s = rng.stream(0, trial, Purpose.GENERIC)
        f, inv = random_furnace(s), random_held(s)
        item = SMELT_INPUTS[s.below(len(SMELT_INPUTS))]
        if s.below(4):
            inv.add(item, s.randint(1, 64))
        if s.below(2):
            inv.add(SMELT_FUELS[s.below(3)], s.randint(1, 3))  # coal, charcoal or planks
        able = BOOK.smeltable(inv, f)
        n = (1, 3, 8, 64)[s.below(4)]
        before = (f.copy(), inv.snapshot())
        r = BOOK.smelt_start(inv, f, item, n)
        assert (r.code == OK) == (iid(item) in able), (trial, r, before)  # the mask is smeltable()
        if r.code != OK:
            assert r.code in (MISSING_INGREDIENT, INV_FULL) and (f, inv.snapshot()) == before
            assert (r.container == "furnace") == (r.code == INV_FULL)
            n_full += r.code == INV_FULL
            continue
        n_ok += 1
        assert 1 <= r.n_items <= min(n, able[iid(item)])
        assert run_until_empty(f.copy(), s, early_collects=False), (trial, r, before)
        assert run_until_empty(f.copy(), s, early_collects=True), (trial, r, before)
    assert n_ok > 500 and n_full > 20


def test_plans_at_the_edges_of_the_burn_time_smelt_all_their_items():
    """The burn times where a one-tick error strands an item: burn time left minus the heat the
    items inside still need, at and next to multiples of 200, lit or not, with cook progress, and
    with progress left over after the input was taken out."""
    rng = CounterRNG(22)
    n_ok = 0
    for burn_left in sorted({max(0, 200 * k + d) for k in range(9) for d in (-1, 0, 1, 2)}):
        for inside, progress in ((0, 0), (1, 0), (1, 57), (3, 199), (2, 1)):
            for taken_out in (False, True):
                f = Furnace(burn_left=burn_left, burn_total=1600)
                if inside:
                    f.put_input("raw_iron", inside)
                    f.cook_ticks = progress
                if taken_out:
                    f.take_input()  # progress stays until the next tick
                inv = inv_with(raw_iron=64, planks=64)
                r = BOOK.smelt_start(inv, f, "raw_iron", 64)
                assert r.code == OK
                n_ok += 1
                s = rng.stream(0, burn_left, Purpose.GENERIC)
                assert run_until_empty(f.copy(), s, early_collects=False), (burn_left, inside, progress, r)
                no_fuel = Furnace(burn_left=burn_left, burn_total=1600)
                r = BOOK.smelt_start(inv_with(raw_iron=64), no_fuel, "raw_iron", 64)
                if r.code == OK:
                    assert r.fuel_units == 0 and no_fuel.finish_ticks() is not None, (burn_left, r)
    assert n_ok == 35 * 5 * 2

"""Tests for the agent-level class tables (hivemind.rules.classes, docs/contracts.md section 3)."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from hivemind.rules import classes as C
from hivemind.rules import load_rules
from hivemind.rules.classes import Concepts, load_concepts, read_block_csv


@pytest.fixture(scope="module")
def rules():
    return load_rules()


@pytest.fixture(scope="module")
def concepts() -> Concepts:
    return load_concepts()


@pytest.fixture(scope="module")
def fresh(rules) -> C.BuildResult:
    return C.build(rules)


# -- files, hashes, determinism ---------------------------------------------------------


def test_files_are_up_to_date_and_deterministic(fresh: C.BuildResult) -> None:
    data, csv_bytes = C.render(fresh)
    assert data == C.DEFAULT_CONCEPTS_PATH.read_bytes(), "rebuild: python -m hivemind.rules.classes"
    assert csv_bytes == C.DEFAULT_CSV_PATH.read_bytes()
    assert C.render(C.build())[0] == data  # same bytes on a second build


def test_load_is_cached_and_verified(concepts: Concepts, rules) -> None:
    assert load_concepts() is concepts
    assert concepts.compute_hash() == concepts.hash
    assert concepts.rules_hash == rules.hash
    assert concepts.raw["csv_sha256"] == __import__("hashlib").sha256(C.DEFAULT_CSV_PATH.read_bytes()).hexdigest()


def test_tampered_file_is_rejected(tmp_path: Path) -> None:
    raw = json.loads(C.DEFAULT_CONCEPTS_PATH.read_text())
    raw["block_classes"][1]["hardness"] = 0.0
    path = tmp_path / "concepts.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_concepts(path)
    assert load_concepts(path, verify=False).block_class(1).hardness == 0.0


# -- coverage -------------------------------------------------------------------------------


def test_every_block_state_is_in_exactly_one_class(concepts: Concepts, rules) -> None:
    n_states = sum(b.state_count for b in rules.blocks.values())
    rows = read_block_csv()
    assert [r[0] for r in rows] == list(range(n_states))
    assert len(concepts.state_class) == n_states
    for state, block, class_id, cls in rows:
        assert rules.block_of_state(state).name == block
        assert concepts.block_class(cls).id == class_id == concepts.state_class[state]
    seen = np.zeros(n_states, int)
    for c in concepts.block_classes:
        for lo, hi in c.state_ranges:
            seen[lo:hi + 1] += 1
    assert (seen == 1).all()
    members = {m for c in concepts.block_classes for m in c.members}
    assert members == set(rules.blocks)


def test_every_item_and_entity_is_in_exactly_one_class(concepts: Concepts, rules) -> None:
    items = [m for c in concepts.item_classes for m in c.members]
    assert sorted(items) == sorted(rules.items)
    entities = [m for c in concepts.entity_classes for m in c.members]
    assert sorted(entities) == sorted(rules.entities)
    for name in rules.items:
        assert name in concepts.item_class_of(name).members
    for name in rules.entities:
        assert name in concepts.entity_class_of("minecraft:" + name).members


def test_ids_and_reserved_classes(concepts: Concepts) -> None:
    ids = [c.id for c in concepts.block_classes]
    assert len(ids) == len(set(ids)) and max(ids) == 255
    assert all(0 <= i <= 254 for i in ids[:-1])
    air, buried = concepts.block_class(0), concepts.block_class(255)
    assert air.name == "air" and air.phrase == ("air",)
    assert set(air.members) == {"air", "cave_air", "void_air"}
    assert buried.name == "buried" and buried.phrase == ("buried",) and buried.members == ()
    assert concepts.item_class(0).name == "none" and concepts.item_class_of("air").id == 0
    assert concepts.state_class.dtype == np.uint8
    for kind in ("item_classes", "entity_classes"):
        assert [c["id"] for c in concepts.raw[kind]] == list(range(len(concepts.raw[kind])))


def test_class_counts_match_the_design(concepts: Concepts) -> None:
    curated = [c for c in concepts.block_classes if not c.fallback and c.name not in ("air", "buried")]
    assert 100 <= len(curated) + 1 <= 140
    resources = [c for c in concepts.block_classes if c.resource]
    assert 30 <= len(resources) <= 50  # DESIGN.md resource memory: about 40
    v1 = [c for c in concepts.item_classes if c.v1]
    assert 150 <= len(v1) <= 200


# -- homogeneity ------------------------------------------------------------------------------


def test_build_has_no_violations_and_uses_every_exception(fresh: C.BuildResult) -> None:
    assert fresh.ok, C.report(fresh)
    assert {(e.kind, e.cls, e.check) for e, _ in fresh.tolerated} == {
        (e.kind, e.cls, e.check) for e in C.EXCEPTIONS}
    listed = {(e["kind"], e["class"]) for e in fresh.concepts["exceptions"]}
    assert ("recipe", "bed") in listed and ("drops", "door") in listed


def test_planted_violations_are_detected(rules) -> None:
    by_name = {s.name: s for s in C.BLOCK_SPECS}
    blocks = [s for s in C.BLOCK_SPECS if s.name not in ("deepslate_iron_ore", "oak_leaves")]
    blocks[[s.name for s in blocks].index("iron_ore")] = dataclasses.replace(
        by_name["iron_ore"], members=("iron_ore", "deepslate_iron_ore"), phrase=("iron", "ore"))  # hardness 3 vs 4.5
    blocks[[s.name for s in blocks].index("leaves")] = dataclasses.replace(
        by_name["leaves"], members=(*by_name["leaves"].members, *by_name["oak_leaves"].members))  # apples
    blocks[[s.name for s in blocks].index("cobweb")] = dataclasses.replace(
        by_name["cobweb"], phrase=("spider",))  # equals an entity phrase
    items = [s for s in C.ITEM_SPECS if s.name != "water_bucket"]
    items[[s.name for s in items].index("bucket")] = C.I("bucket", "bucket", "water_bucket", phrase="bucket")
    exceptions = [e for e in C.EXCEPTIONS if not (e.kind == "recipe" and e.cls == "bed")]
    exceptions.append(C.Exception_("drops", "stone", "state-dependent drops", "planted", "planted",
                                   drop_state={"axis": "y"}))
    result = C.build(rules, block_specs=blocks, item_specs=items, exceptions=exceptions)
    found = {(v.kind, v.cls, v.check) for v in result.violations}
    assert ("block", "iron_ore", "hardness") in found
    assert ("block", "leaves", "drops") in found
    assert ("block", "cobweb", "phrase") in found
    assert ("item", "bucket", "stack_size") in found
    assert ("recipe", "bed", "correlated slots") in found
    assert ("drops", "stone", "stale exception") in found
    assert not result.ok


def test_planted_phrase_and_kind_violations_are_detected(rules, monkeypatch: pytest.MonkeyPatch) -> None:
    hints = list(C.FALLBACK_PHRASE_HINTS)
    hints.insert(0, ("oak_slab", {}, "planks"))  # a fallback named like a curated class
    hints.append(("stone", {}, "rock"))  # stone is curated: the hint matches nothing
    monkeypatch.setattr(C, "FALLBACK_PHRASE_HINTS", tuple(hints))
    monkeypatch.setattr(C, "ITEM_FALLBACK_PHRASES", {**C.ITEM_FALLBACK_PHRASES, 16: ("stick",)})
    overrides = {k: v for k, v in C.ENTITY_KIND_OVERRIDES.items() if k != "hoglin"}  # type and category disagree
    monkeypatch.setattr(C, "ENTITY_KIND_OVERRIDES", {**overrides, "camel": "passive"})  # camel: no disagreement
    result = C.build(rules)
    found = {(v.kind, v.cls, v.check) for v in result.violations}
    assert ("block", "other_planks", "phrase") in found  # a fallback phrase may repeat, but not a curated one
    assert ("block", "stone", "stale phrase hint") in found
    assert ("item", "other_item_16", "phrase") in found
    assert ("entity", "hoglin", "kind") in found
    assert ("entity", "camel", "stale kind override") in found
    assert not result.ok


def test_fallback_entity_kinds() -> None:
    from hivemind.rules.loader import Entity

    def kind(name: str, type_: str | None, category: str | None) -> tuple[str, str | None]:
        return C.fallback_entity_kind(Entity(name, 1, name, 1.0, 1.0, category, type_, None, None, None, None))

    assert kind("skeleton_horse", "animal", "Hostile mobs") == ("passive", None)  # override
    assert kind("evoker_fangs", "other", "Hostile mobs") == ("other", None)  # override
    assert kind("ghast", "mob", "Hostile mobs") == ("hostile", None)  # type "mob" says nothing
    assert kind("boat", "other", "Vehicles") == ("other", None)
    assert kind("new_mob", None, None) == ("other", None)
    k, problem = kind("new_beast", "animal", "Hostile mobs")
    assert problem and "ENTITY_KIND_OVERRIDES" in problem


def test_curated_block_classes_are_exact_for_breaking(concepts: Concepts, rules) -> None:
    """Hardness, tool speeds and harvest rules of every block equal its class's, for every v1 tool."""
    tools = [c for c in concepts.item_classes if c.v1 and c.tool] + [concepts.item_class(0)]
    for name, b in rules.blocks.items():
        for state in {b.min_state, b.default_state, b.max_state}:
            c = concepts.block_class_of_state(state)
            assert c.hardness == b.hardness, name
            for t in tools:
                tool = t.members[0] if t.id else None
                assert concepts.tool_speed[t.id, c.id] == pytest.approx(rules.mining_speed(tool, name)), (name, t.name)
                assert concepts.can_harvest[t.id, c.id] == rules.can_harvest(tool, name), (name, t.name)
            assert c.transparent == b.transparent, name


def test_collision_classes_per_state(concepts: Concepts, rules) -> None:
    walls = set(rules.tag("block", "walls"))
    for name, b in rules.blocks.items():
        for state in range(b.min_state, b.max_state + 1):
            c = concepts.block_class_of_state(state)
            boxes = b.collision_boxes(state)
            want = C.collision_class(boxes)
            if not boxes and name in walls:
                want = 1.5  # listed approximation: unreachable post-less wall states
            assert c.collision_height == want, (name, state)
            if not c.fallback:
                assert c.solid == C.is_full_cube(boxes), (name, state)


def test_break_check_values_at_class_level(concepts: Concepts) -> None:
    """DESIGN.md 4.3 check values, computed from the class arrays alone."""

    def ticks(tool: str | None, block: str) -> int:
        t = concepts.item_class(tool).id if tool else 0
        b = concepts.block_class(block).id
        divisor = 30 if concepts.can_harvest[t, b] else 100
        dmg = concepts.tool_speed[t, b] / concepts.hardness[b] / divisor
        return 0 if dmg >= 1 else math.ceil(1 / dmg - 1e-6)

    assert ticks("wooden_pickaxe", "stone") == 23
    assert ticks(None, "stone") == 150
    assert ticks(None, "log") == 60
    assert ticks("wooden_axe", "log") == 30
    assert ticks("diamond_pickaxe", "obsidian") == 188


# -- spot checks ---------------------------------------------------------------------------------


def test_spot_checks(concepts: Concepts, rules) -> None:
    iron, deep = concepts.block_class("iron_ore"), concepts.block_class("deepslate_iron_ore")
    assert iron.id != deep.id and (iron.hardness, deep.hardness) == (3.0, 4.5)
    assert iron.needs_tier == "stone" and iron.resource and deep.resource
    logs = {f"{w}_log" for w in ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry")}
    assert {concepts.block_class_of_state(rules.block(n).default_state).name for n in logs} == {"log"}
    assert {concepts.item_class_of(n).name for n in logs} == {"log"}
    assert concepts.block_class("log").phrase == ("log",) and concepts.item_class("planks").phrase == ("planks",)
    assert concepts.block_class("stone").drops["items"] == ["cobblestone"]
    assert concepts.block_class("grass_block").drops["items"] == ["dirt"]
    assert concepts.block_class("iron_ore").drops["items"] == ["raw_iron"]
    assert concepts.block_class("glass").drops is None  # Silk Touch only: nothing in v1
    assert concepts.block_class_of_state(rules.block("cave_air").default_state).id == 0
    assert concepts.block_class("torch").members == ("torch", "wall_torch")
    assert concepts.item_class_of("white_wool") is concepts.item_class_of("black_wool")
    assert concepts.item_class_of("red_bed").name == "bed" and concepts.item_class("bed").stack_size == 1


@pytest.mark.parametrize("kind,name,phrase", [
    ("block", "iron_ore", "iron ore"),
    ("block", "crafting_table", "crafting table"),
    ("item", "crafting_table", "crafting table"),
    ("item", "wooden_pickaxe", "wooden pickaxe"),
    ("entity", "zombie", "zombie"),
    ("entity", "cave_spider", "cave spider"),
    ("block", "deepslate_iron_ore", "deepslate iron ore"),
    ("item", "raw_iron", "raw iron"),
    ("item", "rabbit_foot", "rabbit foot"),
    ("block", "lapis_ore", "lapis lazuli ore"),
    ("item", "bed", "bed"),
    ("item", "wool", "wool"),
    ("block", "oak_leaves", "oak leaves"),
    ("block", "sapling", "sapling"),
])
def test_phrases(concepts: Concepts, kind: str, name: str, phrase: str) -> None:
    get = {"block": concepts.block_class, "item": concepts.item_class, "entity": concepts.entity_class}[kind]
    assert " ".join(get(name).phrase) == phrase


def test_words_from_display_names() -> None:
    assert C.words("Jack o'Lantern") == ("jack", "o", "lantern")
    assert C.words("Rabbit's Foot") == ("rabbit", "foot")
    assert C.words("Block of Raw Iron") == ("block", "of", "raw", "iron")


def test_word_reuse_for_iron(concepts: Concepts) -> None:
    for kind, name in (("block", "iron_ore"), ("block", "deepslate_iron_ore"), ("item", "raw_iron"),
                       ("item", "iron_ingot"), ("item", "iron_pickaxe"), ("item", "iron_nugget")):
        c = concepts.block_class(name) if kind == "block" else concepts.item_class(name)
        assert "iron" in c.phrase


def test_curated_phrases_are_unique_and_invertible(concepts: Concepts) -> None:
    """Contracts 3: curated classes (everything a command can name) have their own phrase, so
    the phrase is invertible (contracts 4.4); a fallback phrase never equals a curated one."""
    for kind, classes in (("block", concepts.block_classes), ("item", concepts.item_classes),
                          ("entity", concepts.entity_classes)):
        curated = [c.phrase for c in classes if not c.fallback]
        assert len(curated) == len(set(curated)), [p for p in curated if curated.count(p) > 1]
        for c in classes:
            assert c.phrase and all(re.fullmatch(r"[a-z0-9]+", w) for w in c.phrase), c.name
            if not c.fallback:
                assert concepts.class_by_phrase(kind, c.phrase) is c
            else:
                assert c.phrase not in set(curated), c.name
    with pytest.raises(KeyError):
        concepts.class_by_phrase("block", ("block",))
    with pytest.raises(KeyError):  # shared by several fallback classes: names none of them
        concepts.class_by_phrase("block", ("other", "pickaxe", "block"))


def test_fallback_phrases_are_true_of_every_member(concepts: Concepts, rules) -> None:
    """Blocks split by state get state words, and no fallback borrows a member's name for blocks
    it does not describe (coal blocks must not be called "bell": the agent learns from words)."""

    def phrase_of(block: str, **state: str) -> str:
        b = rules.block(block)
        s = next(s for s in range(b.min_state, b.max_state + 1)
                 if all(b.state_values(s)[k] == v for k, v in state.items()))
        return " ".join(concepts.block_class_of_state(s).phrase)

    assert phrase_of("lantern", hanging="true") == "hanging lantern"
    assert phrase_of("lantern", hanging="false") == "lantern"
    assert phrase_of("oak_fence_gate", open="true") == "open fence gate"
    assert phrase_of("birch_fence_gate", open="false") == "fence gate"
    assert phrase_of("oak_trapdoor", open="false", half="bottom") == "trapdoor"
    assert phrase_of("oak_trapdoor", open="true", half="bottom") == "raised trapdoor"
    assert phrase_of("snow", layers="1") == "snow"
    assert phrase_of("snow", layers="3") == "snow layers"
    assert phrase_of("snow", layers="7") == "deep snow"
    assert phrase_of("cave_vines", berries="true", age="0") == "cave vines with glow berries"
    # without berries, cave vines share a class with "weeping vines plant" etc.: no common tail
    assert phrase_of("cave_vines", berries="false", age="0") == "other axe block"
    assert phrase_of("oak_slab", type="bottom", waterlogged="false") == "slab"
    assert phrase_of("stone_slab", type="bottom", waterlogged="false") == "slab"
    for block in ("coal_block", "diamond_block", "netherite_block", "bricks", "polished_granite"):
        assert phrase_of(block) == "other pickaxe block", block
    hints = {tuple(p.split()) for _b, _w, p in C.FALLBACK_PHRASE_HINTS}
    for c in concepts.block_classes:
        if not c.fallback:
            continue
        assert re.fullmatch("other_" + "_".join(c.phrase) + r"(_\d+)?", c.name), c.name
        if c.phrase in hints or c.phrase in {("other", "block")} | {("other", t, "block") for t in c.mineable}:
            continue
        tail = c.phrase[1:] if c.phrase[0] == "other" else c.phrase  # "other" marks a curated name
        for m in c.members:
            assert C.words(rules.block(m).display_name)[-len(tail):] == tail, (c.name, m)
    assert {c.phrase for c in concepts.item_classes if c.fallback} == {
        ("other", "unstackable", "item"), ("other", "small", "stack", "item"), ("other", "item")}


def test_block_and_entity_phrases_are_disjoint(concepts: Concepts) -> None:
    blocks = {c.phrase for c in concepts.block_classes}
    entities = {c.phrase for c in concepts.entity_classes}
    assert not blocks & entities


def test_fallbacks(concepts: Concepts, rules) -> None:
    for c in concepts.block_classes:
        if c.fallback:
            assert not c.resource and c.name.startswith("other_")
    for c in concepts.item_classes:
        if c.fallback:
            assert not c.v1 and c.places_block_class is None and c.food is None
    assert concepts.item_class_of("netherite_pickaxe").name == "other_item_1"
    assert concepts.item_class_of("oak_slab").name == "other_item_64"
    assert concepts.entity_class_of("blaze").name == "monster"
    assert concepts.entity_class_of("boat").name == "object"
    assert concepts.entity_class_of("hoglin").name == "monster"
    assert concepts.entity_class_of("skeleton_horse").name == "animal"  # minecraft-data: "Hostile mobs"
    assert concepts.entity_class_of("zombie_horse").name == "animal"
    assert concepts.entity_class_of("evoker_fangs").name == "object"
    assert concepts.block_class_of_state(rules.block("oak_slab").default_state).fallback


def test_flags(concepts: Concepts) -> None:
    b = concepts.block_class
    assert b("water").liquid and not b("water").hazard and b("water").collision_height == 0.0
    assert b("lava").liquid and b("lava").hazard and b("lava").light == 15
    assert b("ladder").climbable and b("vine").climbable
    assert b("sand").gravity and b("gravel").gravity and not b("dirt").gravity
    assert b("short_grass").replaceable and b("air").replaceable and not b("stone").replaceable
    for hazard in ("lava", "fire", "magma_block", "cactus", "cobweb", "powder_snow", "sweet_berry_bush"):
        assert b(hazard).hazard, hazard
    assert b("fence").collision_height == 1.5 and b("bed").collision_height == 0.5
    assert b("stone").solid and not b("chest").solid and b("chest").collision_height == 1.0
    assert b("bedrock").hardness == -1 and b("barrier").hardness == -1
    assert b("torch").light == 14 and b("torch").collision_height == 0.0
    assert b("leaves").transparent and not b("stone").transparent
    assert concepts.hazard[b("lava").id] and concepts.light[b("lava").id] == 15
    assert concepts.hardness[255] == -1 and concepts.hardness[b("stone").id] == np.float32(1.5)


def test_item_classes(concepts: Concepts) -> None:
    i = concepts.item_class
    assert i("wooden_pickaxe").tool == {"kind": "pickaxe", "tier": "wood", "speed": 2.0, "damage_per_block": 1}
    assert i("bread").food["nutrition"] == 5
    assert i("log").fuel_ticks == 300 and i("planks").fuel_ticks == 300 and i("coal").fuel_ticks == 1600
    assert i("iron_chestplate").armor["points"] == 6
    assert i("log").places_block_class == "log" and i("wheat_seeds").places_block_class == "wheat"
    assert i("wheat").places_block_class is None
    assert i("snowball").stack_size == 16 and concepts.stack_size[i("snowball").id] == 16
    assert i("sapling").members == tuple(sorted(f"{w}_sapling" for w in (
        "oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "cherry")))


def test_entity_classes(concepts: Concepts) -> None:
    e = concepts.entity_class
    assert e("zombie").kind == "hostile" and e("zombie").max_health == 20.0
    assert e("cow").kind == "passive" and e("player").kind == "player" and e("item").kind == "item"
    assert e("item").phrase == ("dropped", "item")
    assert np.isnan(concepts.max_health[e("monster").id])
    assert {"zombie", "creeper", "skeleton", "spider", "enderman", "witch"} <= {
        c.name for c in concepts.entity_classes if c.kind == "hostile"}
    assert (e("slime").max_health, e("phantom").max_health, e("bogged").max_health) == (16.0, 20.0, 16.0)
    assert not np.isnan(concepts.entity_attack_damage[e("phantom").id])


# -- recipes and drops -------------------------------------------------------------------------


def test_class_level_recipes(concepts: Concepts) -> None:
    by_result: dict[str, list[dict]] = {}
    for r in concepts.recipes:
        by_result.setdefault(r["result"], []).append(r)
    planks = [r for r in by_result["planks"] if r["type"] == "crafting_shapeless"]
    assert len(planks) == 1 and planks[0]["ingredients"] == [["log"]] and planks[0]["count"] == 4
    assert len(planks[0]["sources"]) == 8
    (bed,) = by_result["bed"]  # listed approximation: any three wool
    assert sorted(bed["key"].values()) == [["planks"], ["wool"]] and len(bed["sources"]) == 16
    assert "log" not in by_result  # wood-from-logs recipes are left out (listed)
    assert ["cobbled_deepslate", "cobblestone"] in by_result["stone_pickaxe"][0]["key"].values()
    assert any(r["type"] == "smelting" and r["ingredients"] == [["raw_iron"]] for r in by_result["iron_ingot"])
    v1 = {c.name for c in concepts.item_classes if c.v1}
    for r in concepts.recipes:
        slots = list(r.get("key", {}).values()) + list(r.get("ingredients", []))
        assert r["result"] in v1 and all(set(s) <= v1 for s in slots)


def test_drop_tables_name_item_classes(concepts: Concepts) -> None:
    names = {c.name for c in concepts.item_classes}

    def items(node) -> set[str]:
        out = set()
        if isinstance(node, dict):
            if node.get("type") == "minecraft:item":
                out.add(node["name"])
            for v in node.values():
                out |= items(v)
        elif isinstance(node, list):
            for v in node:
                out |= items(v)
        return out

    for c in concepts.block_classes:
        if c.drops:
            assert items(c.drops["table"]) == set(c.drops["items"]) <= names
            assert "silk_touch" not in json.dumps(c.drops["table"])


def test_specialise_loot(rules) -> None:
    slab = rules.loot_table("blocks/oak_slab")
    double = C.specialise_loot(slab, "oak_slab", {"type": "double", "waterlogged": "false"})
    bottom = C.specialise_loot(slab, "oak_slab", {"type": "bottom", "waterlogged": "false"})
    assert '"count": 2.0' in json.dumps(double) and "set_count" not in json.dumps(bottom)
    snow = C.specialise_loot(rules.loot_table("blocks/snow"), "snow", {"layers": "3"})
    assert '"count": 3.0' in json.dumps(snow) and "snow_block" not in json.dumps(snow)
    leaves = C.specialise_loot(rules.loot_table("blocks/oak_leaves"), "oak_leaves", {})
    assert '"chance": 0.05' in json.dumps(leaves) and "table_bonus" not in json.dumps(leaves)
    assert C.specialise_loot(rules.loot_table("blocks/glass"), "glass", {}) is None
    tall = rules.loot_table("blocks/tall_grass")
    assert "location_check" in json.dumps(C.specialise_loot(tall, "tall_grass", {"half": "lower"}))
    paired = C.specialise_loot(tall, "tall_grass", {"half": "lower"}, two_cell=True)
    assert "location_check" not in json.dumps(paired) and len(paired["pools"]) == 1  # the upper pool is gone


def test_class_drops_need_no_world_state(concepts: Concepts) -> None:
    """Voxels hold class ids only, so no class-level table may test block states or neighbours."""
    for c in concepts.block_classes:
        if c.drops:
            text = json.dumps(c.drops["table"])
            assert "location_check" not in text and "block_state_property" not in text, c.name


def test_fallback_drop_count_errors_are_listed(concepts: Concepts) -> None:
    approx = {e["class"]: e["approximation"] for e in concepts.exceptions if e["check"] == "fallback"}
    snow = approx["other_snow_layers"]
    assert "snow[layers=2]" in snow and all(f"snowball x{n}" in snow for n in (2, 3, 4, 5))
    assert "melon_seeds xB(3, 0.0667)" in approx["other_melon_stem"]
    assert concepts.block_class("other_snow_layers").drops["table"]["pools"][0]["entries"][0]["functions"][0][
        "count"] == 2.0  # the class uses the first table: the listed error is what the loot tests must allow

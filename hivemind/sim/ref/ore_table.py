"""Overworld ore placement of Java Edition 1.21.1, as a small committed table (contracts 5).

The test-world generator (``worldgen.py``) places ores per chunk from this table until M3 replaces
the test worlds with real chunks. Every row is one *placed feature* of the pinned mcmeta
``1.21.1-data`` worldgen files [3] that some overworld biome lists, with the numbers the game uses:

- ``count_min..count_max``: attempts per chunk (``minecraft:count``, a constant or a uniform int);
- ``rarity``: ``minecraft:rarity_filter`` chance, one chunk in N on average (1 = every chunk);
- ``height``, ``min_y``, ``max_y``, ``plateau``: the ``height_range`` provider (``uniform`` or
  ``trapezoid``) with its vertical anchors resolved for the overworld (min_y -64, height 384,
  from ``noise_settings/overworld.json``);
- ``size`` and ``discard_on_air``: the configured ``minecraft:ore`` feature's vein size and
  ``discard_chance_on_air_exposure``;
- ``targets``: which block classes a vein replaces (the target's block tag, resolved through the
  rules file) and the class it places there. Ores have a stone and a deepslate target, so
  ``deepslate iron ore`` replaces deepslate and tuff, while ``iron ore`` replaces stone, granite,
  diorite and andesite;
- ``step`` (6 underground ores, 7 underground decoration) and the biomes that list the feature:
  ``only_biomes`` when few do, ``except_biomes`` when most do (both empty: every overworld biome).

Rows are in the game's order within a step [mem]: vanilla sorts all biomes' features into one
global order per step (``FeatureSorter``); here the plains list comes first and features of other
biomes are inserted after their predecessor in that biome's list, which gives the same order for
every feature in this table.

Regenerate (needs ``data/sources/``, fetched by ``python -m hivemind.rules.fetch_sources``)::

    python -m hivemind.sim.ref.ore_table            # print the generated rows
    python -m hivemind.sim.ref.ore_table --check    # exit 1 if the committed table is stale
    python -m hivemind.sim.ref.ore_table --write    # rewrite the committed rows in this file

``tests/test_worldgen.py`` regenerates the table whenever ``data/sources/`` is present and checks
that it equals the committed one. The sampling rules of the height providers (``height_pmf``) are
from memory of the game code ([mem], noted at each function).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCES = REPO_ROOT / "data" / "sources" / "mcmeta-1.21.1-data" / "data" / "minecraft" / "worldgen"

OVERWORLD_MIN_Y = -64  # [3] noise_settings/overworld.json "noise": {"min_y": -64, "height": 384}
OVERWORLD_HEIGHT = 384  # [3] same file
OVERWORLD_TOP_Y = OVERWORLD_MIN_Y + OVERWORLD_HEIGHT - 1  # 319: ``below_top`` counts down from here [mem]
ORE_STEPS = (6, 7)  # [3] biome "features" index: 6 underground_ores, 7 underground_decoration


@dataclass(frozen=True)
class OreTarget:
    """One target of an ore feature: the block classes it may replace and the class it places."""

    replaces: tuple[str, ...]
    place: str


@dataclass(frozen=True)
class OreFeature:
    """One placed ore feature (see the module docstring for the fields)."""

    name: str
    feature: str
    count_min: int
    count_max: int
    rarity: int
    height: str
    min_y: int
    max_y: int
    plateau: int
    size: int
    discard_on_air: float
    targets: tuple[OreTarget, ...]
    step: int
    only_biomes: tuple[str, ...] = ()
    except_biomes: tuple[str, ...] = ()

    @property
    def is_ore(self) -> bool:
        """True for metal, gem and coal ores (scaled by the density knob); False for stone blobs."""
        return any(t.place.endswith("_ore") for t in self.targets)

    @property
    def placed_classes(self) -> tuple[str, ...]:
        return tuple(t.place for t in self.targets)

    def in_biome(self, biome: str) -> bool:
        """Whether an overworld biome lists this feature."""
        if self.only_biomes:
            return biome in self.only_biomes
        return biome not in self.except_biomes

    def expected_attempts(self) -> float:
        """Mean placement attempts per chunk: rarity probability times the mean count."""
        return (self.count_min + self.count_max) / 2.0 / self.rarity


# ---------------------------------------------------------------------------
# The committed table. Generated; do not edit by hand (see the module docstring).
# ---------------------------------------------------------------------------

_STONE = ("andesite", "diorite", "granite", "stone")
_DEEP = ("deepslate", "tuff")
_BASE = ("andesite", "deepslate", "diorite", "granite", "stone", "tuff")


def _ore(stone: str, deep: str) -> tuple[OreTarget, ...]:
    return (OreTarget(_STONE, stone), OreTarget(_DEEP, deep))


def _blob(place: str) -> tuple[OreTarget, ...]:
    return (OreTarget(_BASE, place),)


ORE_TABLE: tuple[OreFeature, ...] = (
    # BEGIN GENERATED (python -m hivemind.sim.ref.ore_table --write)
    OreFeature('ore_dirt', 'ore_dirt', 7, 7, 1, 'uniform', 0, 160, 0, 33, 0.0, _blob('dirt'), 6),  # [3]
    OreFeature('ore_gravel', 'ore_gravel', 14, 14, 1, 'uniform', -64, 319, 0, 33, 0.0, _blob('gravel'), 6),  # [3]
    OreFeature('ore_granite_upper', 'ore_granite', 1, 1, 6, 'uniform', 64, 128, 0, 64, 0.0, _blob('granite'), 6),  # [3]
    OreFeature('ore_granite_lower', 'ore_granite', 2, 2, 1, 'uniform', 0, 60, 0, 64, 0.0, _blob('granite'), 6),  # [3]
    OreFeature('ore_diorite_upper', 'ore_diorite', 1, 1, 6, 'uniform', 64, 128, 0, 64, 0.0, _blob('diorite'), 6),  # [3]
    OreFeature('ore_diorite_lower', 'ore_diorite', 2, 2, 1, 'uniform', 0, 60, 0, 64, 0.0, _blob('diorite'), 6),  # [3]
    OreFeature('ore_andesite_upper', 'ore_andesite', 1, 1, 6, 'uniform', 64, 128, 0, 64, 0.0,
               _blob('andesite'), 6),  # [3]
    OreFeature('ore_andesite_lower', 'ore_andesite', 2, 2, 1, 'uniform', 0, 60, 0, 64, 0.0,
               _blob('andesite'), 6),  # [3]
    OreFeature('ore_tuff', 'ore_tuff', 2, 2, 1, 'uniform', -64, 0, 0, 64, 0.0, _blob('tuff'), 6),  # [3]
    OreFeature('ore_coal_upper', 'ore_coal', 30, 30, 1, 'uniform', 136, 319, 0, 17, 0.0,
               _ore('coal_ore', 'deepslate_coal_ore'), 6),  # [3]
    OreFeature('ore_coal_lower', 'ore_coal_buried', 20, 20, 1, 'trapezoid', 0, 192, 0, 17, 0.5,
               _ore('coal_ore', 'deepslate_coal_ore'), 6),  # [3]
    OreFeature('ore_iron_upper', 'ore_iron', 90, 90, 1, 'trapezoid', 80, 384, 0, 9, 0.0,
               _ore('iron_ore', 'deepslate_iron_ore'), 6),  # [3]
    OreFeature('ore_iron_middle', 'ore_iron', 10, 10, 1, 'trapezoid', -24, 56, 0, 9, 0.0,
               _ore('iron_ore', 'deepslate_iron_ore'), 6),  # [3]
    OreFeature('ore_iron_small', 'ore_iron_small', 10, 10, 1, 'uniform', -64, 72, 0, 4, 0.0,
               _ore('iron_ore', 'deepslate_iron_ore'), 6),  # [3]
    OreFeature('ore_gold', 'ore_gold_buried', 4, 4, 1, 'trapezoid', -64, 32, 0, 9, 0.5,
               _ore('gold_ore', 'deepslate_gold_ore'), 6),  # [3]
    OreFeature('ore_gold_lower', 'ore_gold_buried', 0, 1, 1, 'uniform', -64, -48, 0, 9, 0.5,
               _ore('gold_ore', 'deepslate_gold_ore'), 6),  # [3]
    OreFeature('ore_redstone', 'ore_redstone', 4, 4, 1, 'uniform', -64, 15, 0, 8, 0.0,
               _ore('redstone_ore', 'deepslate_redstone_ore'), 6),  # [3]
    OreFeature('ore_redstone_lower', 'ore_redstone', 8, 8, 1, 'trapezoid', -96, -32, 0, 8, 0.0,
               _ore('redstone_ore', 'deepslate_redstone_ore'), 6),  # [3]
    OreFeature('ore_diamond', 'ore_diamond_small', 7, 7, 1, 'trapezoid', -144, 16, 0, 4, 0.5,
               _ore('diamond_ore', 'deepslate_diamond_ore'), 6),  # [3]
    OreFeature('ore_diamond_medium', 'ore_diamond_medium', 2, 2, 1, 'uniform', -64, -4, 0, 8, 0.5,
               _ore('diamond_ore', 'deepslate_diamond_ore'), 6),  # [3]
    OreFeature('ore_diamond_large', 'ore_diamond_large', 1, 1, 9, 'trapezoid', -144, 16, 0, 12, 0.7,
               _ore('diamond_ore', 'deepslate_diamond_ore'), 6),  # [3]
    OreFeature('ore_diamond_buried', 'ore_diamond_buried', 4, 4, 1, 'trapezoid', -144, 16, 0, 8, 1.0,
               _ore('diamond_ore', 'deepslate_diamond_ore'), 6),  # [3]
    OreFeature('ore_lapis', 'ore_lapis', 2, 2, 1, 'trapezoid', -32, 32, 0, 7, 0.0,
               _ore('lapis_ore', 'deepslate_lapis_ore'), 6),  # [3]
    OreFeature('ore_lapis_buried', 'ore_lapis_buried', 4, 4, 1, 'uniform', -64, 64, 0, 7, 1.0,
               _ore('lapis_ore', 'deepslate_lapis_ore'), 6),  # [3]
    OreFeature('ore_copper_large', 'ore_copper_large', 16, 16, 1, 'trapezoid', -16, 112, 0, 20, 0.0,
               _ore('copper_ore', 'deepslate_copper_ore'), 6,
               only_biomes=('dripstone_caves',)),  # [3]
    OreFeature('ore_copper', 'ore_copper_small', 16, 16, 1, 'trapezoid', -16, 112, 0, 10, 0.0,
               _ore('copper_ore', 'deepslate_copper_ore'), 6,
               except_biomes=('dripstone_caves',)),  # [3]
    OreFeature('ore_clay', 'ore_clay', 46, 46, 1, 'uniform', -64, 256, 0, 33, 0.0,
               _blob('clay'), 6,
               only_biomes=('lush_caves',)),  # [3]
    OreFeature('ore_gold_extra', 'ore_gold', 50, 50, 1, 'uniform', 32, 256, 0, 9, 0.0,
               _ore('gold_ore', 'deepslate_gold_ore'), 6,
               only_biomes=('badlands', 'eroded_badlands', 'wooded_badlands')),  # [3]
    OreFeature('ore_emerald', 'ore_emerald', 100, 100, 1, 'trapezoid', -16, 480, 0, 3, 0.0,
               _ore('emerald_ore', 'deepslate_emerald_ore'), 6,
               only_biomes=('cherry_grove', 'frozen_peaks', 'grove', 'jagged_peaks', 'meadow', 'snowy_slopes',
                            'stony_peaks', 'windswept_forest', 'windswept_gravelly_hills', 'windswept_hills')),  # [3]
    OreFeature('ore_infested', 'ore_infested', 14, 14, 1, 'uniform', -64, 63, 0, 9, 0.0,
               _ore('other_infested_block', 'other_other_pickaxe_block_5'), 7,
               only_biomes=('cherry_grove', 'frozen_peaks', 'grove', 'jagged_peaks', 'meadow', 'snowy_slopes',
                            'stony_peaks', 'windswept_forest', 'windswept_gravelly_hills', 'windswept_hills')),  # [3]
    # END GENERATED
)


def ore_table(biome: str | None = None) -> tuple[OreFeature, ...]:
    """The committed rows, optionally only those an overworld biome lists (e.g. ``"plains"``)."""
    if biome is None:
        return ORE_TABLE
    return tuple(f for f in ORE_TABLE if f.in_biome(biome))


# ---------------------------------------------------------------------------
# Height distributions [mem]
# ---------------------------------------------------------------------------


def height_pmf(feature: OreFeature) -> dict[int, float]:
    """The probability of each origin y of one attempt (before any vein spread).

    [mem] ``UniformHeight``: ``randomBetweenInclusive(min, max)``; if min > max the game logs a
    warning and returns min. ``TrapezoidHeight``: with k = max - min, if plateau >= k it is uniform;
    else l = (k - plateau) // 2, m = k - l and y = min + U{0..m} + U{0..l} (two independent
    uniform integers), a trapezoid (a triangle when plateau = 0).
    """
    lo, hi = feature.min_y, feature.max_y
    if lo > hi:
        return {lo: 1.0}
    if feature.height == "uniform":
        n = hi - lo + 1
        return {y: 1.0 / n for y in range(lo, hi + 1)}
    if feature.height != "trapezoid":
        raise ValueError(f"unknown height provider {feature.height!r}")
    k = hi - lo
    if feature.plateau >= k:
        n = k + 1
        return {y: 1.0 / n for y in range(lo, hi + 1)}
    small = (k - feature.plateau) // 2
    big = k - small
    pmf: dict[int, float] = {}
    p = 1.0 / ((big + 1) * (small + 1))
    for a in range(big + 1):
        for b in range(small + 1):
            pmf[lo + a + b] = pmf.get(lo + a + b, 0.0) + p
    return pmf


def band_probability(feature: OreFeature, y_lo: int, y_hi: int) -> float:
    """Probability that an attempt's origin y lies in [y_lo, y_hi]."""
    return sum(p for y, p in height_pmf(feature).items() if y_lo <= y <= y_hi)


# ---------------------------------------------------------------------------
# Generator (reads the pinned worldgen files)
# ---------------------------------------------------------------------------


def _strip(name: str) -> str:
    return name.removeprefix("minecraft:")


def _anchor(a: Mapping[str, int]) -> int:
    """A vertical anchor resolved for the overworld [mem]: absolute y, or counted from the bottom
    (min_y) or from the top (the highest block, min_y + height - 1)."""
    (kind, value), = a.items()
    if kind == "absolute":
        return int(value)
    if kind == "above_bottom":
        return OVERWORLD_MIN_Y + int(value)
    if kind == "below_top":
        return OVERWORLD_TOP_Y - int(value)
    raise ValueError(f"unknown vertical anchor {a!r}")


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _overworld_biomes(root: Path) -> dict[str, list[list[str]]]:
    """Feature lists of the overworld biomes: those with the ``minecraft:cave`` carver [3]."""
    biomes = {}
    for path in sorted((root / "biome").glob("*.json")):
        data = _read(path)
        carvers = data.get("carvers", {}).get("air", [])
        carvers = [carvers] if isinstance(carvers, str) else carvers
        if "minecraft:cave" in carvers:
            biomes[path.stem] = [[_strip(f) for f in step] for step in data["features"]]
    return biomes


def _feature_order(biomes: Mapping[str, list[list[str]]], step: int, first: str) -> list[str]:
    """Merge the biomes' lists of one step: ``first`` biome's list, then the others in name order,
    each unseen feature inserted after its predecessor in that biome's list."""
    order: list[str] = []
    for name in [first, *sorted(b for b in biomes if b != first)]:
        feats = biomes[name][step] if step < len(biomes[name]) else []
        prev = None
        for f in feats:
            if f not in order:
                order.insert(order.index(prev) + 1 if prev is not None else len(order), f)
            prev = f
    return order


def _class_of(block: str) -> str:
    """The block class of a vanilla block's default state (``hivemind.rules.classes``)."""
    from hivemind.rules import load_concepts, load_rules

    rules, concepts = load_rules(), load_concepts()
    return concepts.block_class_of_state(rules.block(block).default_state).name


def _target_classes(target: Mapping) -> tuple[str, ...]:
    from hivemind.rules import load_rules

    if target.get("predicate_type") != "minecraft:tag_match":
        raise ValueError(f"unsupported ore target predicate {target!r}")
    members = load_rules().tag("block", _strip(target["tag"]))
    return tuple(sorted({_class_of(m) for m in members}))


def _row(name: str, step: int, placed: Mapping, configured: Mapping, biomes: Sequence[str],
         all_biomes: Sequence[str]) -> OreFeature:
    count_min = count_max = 1
    rarity = 1
    height = None
    for p in placed["placement"]:
        kind = _strip(p["type"])
        if kind == "count":
            c = p["count"]
            if isinstance(c, int):
                count_min = count_max = c
            elif _strip(c["type"]) == "uniform":
                count_min, count_max = int(c["min_inclusive"]), int(c["max_inclusive"])
            else:
                raise ValueError(f"{name}: unsupported count {c!r}")
        elif kind == "rarity_filter":
            rarity = int(p["chance"])
        elif kind == "height_range":
            height = p["height"]
        elif kind not in ("in_square", "biome"):
            raise ValueError(f"{name}: unsupported placement modifier {kind!r}")
    if height is None:
        raise ValueError(f"{name}: no height_range")
    if _strip(configured["type"]) != "ore":
        raise ValueError(f"{name}: not an ore feature")
    cfg = configured["config"]
    targets = tuple(OreTarget(_target_classes(t["target"]), _class_of(_strip(t["state"]["Name"])))
                    for t in cfg["targets"])
    only: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    if len(biomes) <= len(all_biomes) // 2:
        only = tuple(sorted(biomes))
    else:
        excluded = tuple(sorted(set(all_biomes) - set(biomes)))
    return OreFeature(
        name=name, feature=_strip(placed["feature"]), count_min=count_min, count_max=count_max, rarity=rarity,
        height=_strip(height["type"]), min_y=_anchor(height["min_inclusive"]), max_y=_anchor(height["max_inclusive"]),
        plateau=int(height.get("plateau", 0)), size=int(cfg["size"]),
        discard_on_air=float(cfg["discard_chance_on_air_exposure"]), targets=targets, step=step,
        only_biomes=only, except_biomes=excluded)


def generate_table(root: Path | str = DEFAULT_SOURCES) -> tuple[OreFeature, ...]:
    """Build the table from the worldgen directory of the pinned mcmeta ``1.21.1-data`` [3]."""
    root = Path(root)
    noise = _read(root / "noise_settings" / "overworld.json")["noise"]
    if (noise["min_y"], noise["height"]) != (OVERWORLD_MIN_Y, OVERWORLD_HEIGHT):
        raise ValueError(f"overworld noise settings changed: {noise}")
    biomes = _overworld_biomes(root)
    rows = []
    for step in ORE_STEPS:
        for name in _feature_order(biomes, step, "plains"):
            if not name.startswith("ore_"):
                continue
            placed = _read(root / "placed_feature" / f"{name}.json")
            configured = _read(root / "configured_feature" / f"{_strip(placed['feature'])}.json")
            users = [b for b, steps in biomes.items() if step < len(steps) and name in steps[step]]
            rows.append(_row(name, step, placed, configured, users, list(biomes)))
    return tuple(rows)


# ---------------------------------------------------------------------------
# Rendering the committed rows
# ---------------------------------------------------------------------------

_BEGIN = "    # BEGIN GENERATED (python -m hivemind.sim.ref.ore_table --write)\n"
_END = "    # END GENERATED\n"


def _targets_source(targets: tuple[OreTarget, ...]) -> str:
    if (len(targets) == 2 and targets[0].replaces == _STONE and targets[1].replaces == _DEEP):
        return f"_ore({targets[0].place!r}, {targets[1].place!r})"
    if len(targets) == 1 and targets[0].replaces == _BASE:
        return f"_blob({targets[0].place!r})"
    return "(" + ", ".join(f"OreTarget({t.replaces!r}, {t.place!r})" for t in targets) + ",)"


def render_rows(rows: Iterable[OreFeature]) -> str:
    """The Python source of the committed rows (one ``OreFeature`` each, lines <= 120 chars)."""
    out = []
    pad = " " * 15
    for r in rows:
        head = (f"    OreFeature({r.name!r}, {r.feature!r}, {r.count_min}, {r.count_max}, {r.rarity}, "
                f"{r.height!r}, {r.min_y}, {r.max_y}, {r.plateau}, {r.size}, {r.discard_on_air!r},")
        body = f"{_targets_source(r.targets)}, {r.step}"
        key, biomes = ("only_biomes", r.only_biomes) if r.only_biomes else ("except_biomes", r.except_biomes)
        if not biomes:
            line = f"{head} {body}),  # [3]"
            out.append(line if len(line) <= 120 else f"{head}\n{pad}{body}),  # [3]")
            continue
        lines = [head, f"{pad}{body},", f"{pad}{key}=("]
        for i, b in enumerate(biomes):
            item = f"{b!r}" + (", " if i < len(biomes) - 1 else ")),  # [3]")
            if len(lines[-1]) + len(item.rstrip()) > 120:
                lines[-1] = lines[-1].rstrip()
                lines.append(pad + " " * (len(key) + 2))
            lines[-1] += item
        if len(biomes) == 1:
            lines[-1] = lines[-1].replace(")),  # [3]", ",)),  # [3]")
        out.append("\n".join(lines))
    return "".join(line + "\n" for line in out)


def write_table(rows: Sequence[OreFeature], path: Path = Path(__file__)) -> None:
    """Replace the generated rows of this module's source."""
    text = path.read_text(encoding="utf-8")
    start, end = text.index(_BEGIN) + len(_BEGIN), text.index(_END)
    path.write_text(text[:start] + render_rows(rows) + text[end:], encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sources", type=Path, default=DEFAULT_SOURCES, help="mcmeta worldgen directory")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="exit 1 if the committed table is stale")
    mode.add_argument("--write", action="store_true", help="rewrite the committed table")
    args = ap.parse_args(argv)
    rows = generate_table(args.sources)
    if args.write:
        write_table(rows)
        print(f"wrote {len(rows)} rows to {Path(__file__).name}")
        return 0
    if args.check:
        if rows != ORE_TABLE:
            print("ore table is stale: run python -m hivemind.sim.ref.ore_table --write", file=sys.stderr)
            return 1
        print(f"ore table is current ({len(rows)} rows)")
        return 0
    sys.stdout.write(render_rows(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())

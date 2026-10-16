"""Write ``bridge/node/test/dig_times.json``: how long Mineflayer must wait to break each v1 block
once the dig-time patch (``bridge/node/patches.js``, DESIGN.md 5) is applied.

Usage, from the repository root::

    .venv/bin/python bridge/node/make_dig_fixture.py           # (re)write the fixture
    .venv/bin/python bridge/node/make_dig_fixture.py --check   # compare with the committed file

The values come from the reference mechanics (``hivemind.sim.ref.mechanics``), never from Node:

- ``design``: DESIGN.md 4.3's ``ceil(1/dmg)`` with the game's tool-rule speeds
  (``break_ticks_for(...).design_ticks``). Patched Mineflayer must wait exactly this.
- ``unpatched``: what Mineflayer 4.39.0 waits as shipped (``mineflayer_ticks``: minecraft-data's
  material speeds, ``mechanics.mineflayer_tool_speed``); present only for blocks where it differs
  from ``design`` somewhere.
- ``exceptions``: the pairs the patch does not fix, with what patched Mineflayer still waits. The
  build asserts these are exactly DESIGN.md 5's documented case, moss carpet with a hoe
  (minecraft-data's material "plant" lists no hoes).

Ticks are integers; 0 = instant, null = never. Each block row holds, per condition (``ground``,
``water``: eyes in water without Aqua Affinity, ``air``: not on the ground), one value per entry
of ``held`` (null = empty hand, then every v1 tool class of ``data/concepts-1.21.1.json``).

Blocks (vanilla names): every member of the block tag ``incorrect_for_wooden_tool`` (the 101
tier-gated blocks the patch changes: all metal and gem ores, their deepslate forms, obsidian,
metal blocks, ...), the coal ores, stone and deepslate and their cobbles, every member of the
block tag ``logs``, dirt, grass block, sand, red sand, gravel, and moss carpet (the exception).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hivemind.rules import Rules, load_rules  # the repository's package, installed by pip install -e .
from hivemind.rules.classes import Concepts, load_concepts
from hivemind.sim.ref import mechanics as m

FIXTURE_PATH = Path(__file__).resolve().parent / "test" / "dig_times.json"

TIER_GATED_TAG = m.MINEFLAYER_TIER_GATED_TAG  # "incorrect_for_wooden_tool"
TIER_GATED_COUNT = 101  # DESIGN.md 5: "the 101 blocks of incorrect_for_wooden_tool" (1.21.1 tags [3])
MINEABLE_KINDS = ("axe", "hoe", "pickaxe", "shovel")

# The v1-relevant blocks besides the tag members (vanilla names; DESIGN.md 5 and the M0 review).
EXTRA_BLOCKS: dict[str, tuple[str, ...]] = {
    "coal_ore": ("coal_ore", "deepslate_coal_ore"),
    "stone": ("stone", "deepslate", "cobblestone", "cobbled_deepslate"),
    "soil": ("dirt", "grass_block", "sand", "red_sand", "gravel"),
    "exception": ("moss_carpet",),
}
LOG_TAG = "logs"

# Conditions: (name, in_water, on_ground). prismarine-block's digTime takes inWater and notOnGround.
CONDITIONS: tuple[tuple[str, bool, bool], ...] = (
    ("ground", False, True),
    ("water", True, True),
    ("air", False, False),
)

# DESIGN.md 5: "With the patch, Mineflayer's wait equals the formula of 4.3 for every block, tool
# and condition but one (moss carpet with a hoe)". The build fails on any other difference.
DOCUMENTED_EXCEPTIONS: dict[str, frozenset[str]] = {"moss_carpet": frozenset({"hoe"})}


class FixtureError(Exception):
    """The data contradicts an assumption of the patch."""


def tier_gated_blocks(rules: Rules) -> list[str]:
    """The tag ``incorrect_for_wooden_tool``, checked to be pickaxe-only (what the patch assumes)."""
    members = sorted(rules.tag("block", TIER_GATED_TAG))
    if len(members) != TIER_GATED_COUNT:
        raise FixtureError(f"{TIER_GATED_TAG} has {len(members)} members, expected {TIER_GATED_COUNT}")
    for name in members:
        kinds = [k for k in MINEABLE_KINDS if rules.in_tag("block", name, f"mineable/{k}")]
        if kinds != ["pickaxe"]:
            raise FixtureError(f"{name} is in mineable tags {kinds}, not pickaxe only")
    return members


def fixture_blocks(rules: Rules) -> dict[str, str]:
    """Block name -> group, in a stable order."""
    out = {name: "tier_gated" for name in tier_gated_blocks(rules)}
    for group, names in EXTRA_BLOCKS.items():
        if group == "exception":
            continue
        out.update({n: group for n in names})
    out.update({n: "log" for n in sorted(rules.tag("block", LOG_TAG))})
    out.update({n: "exception" for n in EXTRA_BLOCKS["exception"]})
    for name in out:
        rules.block(name)  # KeyError on an unknown name
    return out


def v1_held_items(concepts: Concepts) -> list[str | None]:
    """Empty hand, then the members of every v1 tool class (concepts file), sorted."""
    tools = sorted(mem for c in concepts.item_classes if c.v1 and c.tool is not None for mem in c.members)
    return [None, *tools]


def patched_speed(rules: Rules, held: str | None, block: str, tier_gated: set[str]) -> float:
    """The speed prismarine-block uses after the patch: a tier-gated block's material becomes
    "mineable/pickaxe", whose speeds are the pickaxes' tool-rule speeds; nothing else changes."""
    if block in tier_gated:
        return rules.mining_speed(held, block)
    return m.mineflayer_tool_speed(rules, held, block)


def build_fixture(rules: Rules | None = None, concepts: Concepts | None = None) -> dict[str, Any]:
    rules = rules if rules is not None else load_rules()
    concepts = concepts if concepts is not None else load_concepts()
    blocks = fixture_blocks(rules)
    tier_gated = {b for b, g in blocks.items() if g == "tier_gated"}
    held = v1_held_items(concepts)
    c = m.constants_for(rules)

    rows: dict[str, Any] = {}
    exceptions: list[dict[str, Any]] = []
    for block, group in blocks.items():
        hardness = rules.block(block).hardness
        design: dict[str, list[int | None]] = {}
        unpatched: dict[str, list[int | None]] = {}
        for cond, in_water, on_ground in CONDITIONS:
            design[cond], unpatched[cond] = [], []
            for item in held:
                t = m.break_ticks_for(rules, block, item, in_water=in_water, on_ground=on_ground)
                design[cond].append(t.design_ticks)
                unpatched[cond].append(t.mineflayer_ticks)
                patched = m.design_break_ticks(hardness, patched_speed(rules, item, block, tier_gated),
                                               rules.can_harvest(item, block), in_water=in_water,
                                               on_ground=on_ground, c=c)
                if patched != t.design_ticks:
                    kind = rules.item(item).tool.kind if item is not None else None
                    if kind not in DOCUMENTED_EXCEPTIONS.get(block, ()):
                        raise FixtureError(f"patched Mineflayer would differ from DESIGN.md 4.3 on {block} with "
                                           f"{item} ({cond}): {patched} vs {t.design_ticks}")
                    exceptions.append({"block": block, "held": item, "condition": cond,
                                       "design": t.design_ticks, "patched": patched})
        row: dict[str, Any] = {"group": group, "design": design}
        if unpatched != design:
            row["unpatched"] = unpatched
        rows[block] = row
    if not exceptions:
        raise FixtureError("the documented moss carpet + hoe exception no longer appears; update DESIGN.md 5")

    return {
        "version": rules.version,
        "generator": "bridge/node/make_dig_fixture.py",
        "units": "ticks; 0 = instant, null = never",
        "tier_gated_tag": TIER_GATED_TAG,
        "tier_gated": sorted(tier_gated),
        "held": held,
        "conditions": [{"name": n, "in_water": w, "on_ground": g} for n, w, g in CONDITIONS],
        "exceptions": exceptions,
        "blocks": rows,
    }


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def render(fixture: dict[str, Any]) -> str:
    """Deterministic text: one line per block row and per exception, so diffs stay readable."""
    lines = ["{"]
    for key, value in fixture.items():
        if key in ("exceptions", "blocks"):
            continue
        lines.append(f"  {_compact(key)}: {_compact(value)},")
    lines.append('  "exceptions": [')
    exc = fixture["exceptions"]
    lines += [f"    {_compact(e)}{',' if i < len(exc) - 1 else ''}" for i, e in enumerate(exc)]
    lines.append("  ],")
    lines.append('  "blocks": {')
    rows = list(fixture["blocks"].items())
    lines += [f"    {_compact(k)}: {_compact(v)}{',' if i < len(rows) - 1 else ''}" for i, (k, v) in enumerate(rows)]
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=FIXTURE_PATH)
    ap.add_argument("--check", action="store_true", help="compare with --out; write nothing")
    args = ap.parse_args(argv)
    try:
        text = render(build_fixture())
    except FixtureError as exc:
        print(f"fixture failed: {exc}", file=sys.stderr)
        return 2
    if args.check:
        same = args.out.exists() and args.out.read_text(encoding="utf-8") == text
        print(f"{args.out}: {'up to date' if same else 'DIFFERS from a fresh build'}")
        return 0 if same else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

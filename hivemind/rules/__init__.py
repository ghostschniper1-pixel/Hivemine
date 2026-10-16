"""Game rules for Java Edition 1.21.1, built from pinned upstream data (docs/contracts.md 1-2).

- ``python -m hivemind.rules.fetch_sources`` downloads the pinned sources into data/sources/.
- ``python -m hivemind.rules.build_rules`` builds data/rules-1.21.1.json and cross-checks the sources.
- ``load_rules()`` reads the built file (cached) through a thin typed wrapper.
- ``python -m hivemind.rules.classes`` builds the class tables (data/concepts-1.21.1.json and
  data/block_classes-1.21.1.csv); ``load_concepts()`` reads them (contracts 3).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hivemind.rules.loader import (
    Armor,
    Biome,
    Block,
    BlockState,
    Constant,
    Entity,
    Food,
    Item,
    Recipe,
    Rules,
    Tool,
    ToolRule,
    load_rules,
)

__all__ = [
    "Armor",
    "Biome",
    "Block",
    "BlockState",
    "Constant",
    "Entity",
    "Food",
    "Item",
    "Recipe",
    "Rules",
    "Tool",
    "ToolRule",
    "load_concepts",
    "load_rules",
]

if TYPE_CHECKING:
    from hivemind.rules.classes import Concepts


def load_concepts(path: str | Path | None = None, verify: bool = True) -> Concepts:
    """``hivemind.rules.classes.load_concepts`` (cached; ``verify`` checks the file's hash).

    Imported on first use, so that ``python -m hivemind.rules.classes`` does not import the
    module twice.
    """
    from hivemind.rules.classes import load_concepts as _load_concepts

    return _load_concepts(path, verify)

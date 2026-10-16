"""Game rules for Java Edition 1.21.1, built from pinned upstream data (docs/contracts.md 1-2).

- ``python -m hivemind.rules.fetch_sources`` downloads the pinned sources into data/sources/.
- ``python -m hivemind.rules.build_rules`` builds data/rules-1.21.1.json and cross-checks the sources.
- ``load_rules()`` reads the built file (cached) through a thin typed wrapper.
"""

from hivemind.rules.loader import (
    Armor,
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
    "load_rules",
]

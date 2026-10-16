"""The shared English language layer (docs/contracts.md 4, DESIGN.md 5, 6.2, 6.6, 9).

One controlled English vocabulary serves actions, observations, chat and human input, so the
embedding of ``iron`` is trained by ``mine 4 iron ore.``, by ``raw iron`` in the inventory and by
``i need 3 iron ingot`` in chat. Words are always true of what they name: every block, item,
entity and biome is named by its class phrase from ``data/concepts-1.21.1.json``.

Modules:

- ``grammar``: the fixed text: 20 verb ids and their templates, 15 chat sentences of 14
  intents (``where`` has ``where is {item}`` and ``where is {block}``), 9 goals, result words,
  event sentences, slot domains, number tokens.
- ``world_words``: small typed tables of the words for where and when (directions with their
  steps and yaws, block faces with their normals, times of day with their ticks) for the
  observation builder's F3 group and panorama (DESIGN.md 6.1, 6.2).
- ``vocab``: builds ``data/vocab-1.21.1.json`` (``python -m hivemind.lang.vocab [--check]``).
- ``language``: ``Language`` (words, phrases, render, encode/decode, strict parse, results,
  events).
- ``decoder``: masked word-by-word generation over a ``SlotProvider`` (contracts 4.5),
  ``FullProvider`` and ``enumerate_commands``.
- ``automaton``: a static word-level automaton (states x words, plus per-slot trie incidence)
  for a later GPU or C++ mask builder.
- ``human``: the human tokenizer and lenient parser (``normalize_human``, ``parse_human``).
"""

from hivemind.lang.decoder import Decoder, FullProvider, SlotProvider, enumerate_commands
from hivemind.lang.grammar import snap, snap_count
from hivemind.lang.language import Language
from hivemind.lang.types import Command, DeadEndError, EntityRef, Event, Flags, Goal, ParseError, Sentence

__all__ = [
    "Command",
    "DeadEndError",
    "Decoder",
    "EntityRef",
    "Event",
    "Flags",
    "FullProvider",
    "Goal",
    "Language",
    "ParseError",
    "Sentence",
    "SlotProvider",
    "enumerate_commands",
    "snap",
    "snap_count",
]

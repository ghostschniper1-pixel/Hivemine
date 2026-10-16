# Minecraft Hivemind

Two teams of 10–20 learning Minecraft agents in one world: do they fight over resources or work together? See [DESIGN.md](DESIGN.md) for the full design and [docs/contracts.md](docs/contracts.md) for the module interfaces.

Agents act and perceive in plain English words: an action is a command such as `mine 4 iron ore.` or `give 3 bread to red 7.`, and the same words name what they see and what they say in chat.

## Status

Milestone M0 (pin version, build rules, freeze server config) and the start of M1 (reference simulator) are in progress. Nothing is trained yet.

## Layout

| Path | What |
|---|---|
| `DESIGN.md` | Design document |
| `docs/contracts.md` | Interfaces between modules |
| `data/` | Built rules, concept tables and vocabulary (raw sources in `data/sources/`, git-ignored) |
| `hivemind/rules/` | Fetch pinned sources, build `data/rules-1.21.1.json` and the class tables |
| `hivemind/lang/` | The shared English vocabulary, command grammar, parser and masked word-by-word decoding |
| `hivemind/sim/ref/` | Reference simulator (readable Python, the oracle for the later C++ core) |
| `hivemind/features/` | Observation builder and memory, shared by the simulator and the bridge |
| `bridge/` | Real-game side: frozen server config, Mineflayer hosts (later) |
| `tests/` | pytest suite |

## Setup

```
python -m venv .venv
.venv/bin/pip install -e .[dev]          # Windows: .venv\Scripts\pip install -e .[dev]
.venv/bin/python -m hivemind.rules.fetch_sources   # only needed to rebuild data/
.venv/bin/python -m pytest
```

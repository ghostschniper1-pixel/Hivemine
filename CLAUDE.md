# Minecraft Hivemind

Read docs/HANDOFF.md (where things stand and what comes next), DESIGN.md (the design) and docs/contracts.md (the interfaces between modules) before changing code.

- Python 3.11 or newer, NumPy only under `hivemind/`. Tests: `.venv/bin/python -m pytest` (Windows: `.venv\Scripts\python -m pytest`); slow tests run with `-m slow`. Node tests: `cd bridge && npm test` (Node 22 or newer).
- Generated data in `data/` is committed and checked byte for byte. After changing the rules build, the class tables or the vocabulary, rebuild in this order and commit the results together: `python -m hivemind.rules.build_rules`, then `python -m hivemind.rules.classes`, then `python -m hivemind.lang.vocab` (each has `--check`).
- Every hard-coded game constant carries a source label: a DESIGN.md reference such as [13], [mem] (from memory, not verified) or [est] (an estimate to be measured).
- Everything the agent sees or does is named in plain English words, and the words must be true of what they name (DESIGN.md 5 and 6). Perception is honest: the agent never learns what it has not seen (DESIGN.md 6.1).
- Line endings are LF on every OS (`.gitattributes`).
- Models: use Claude Opus 5.5 (`claude-opus-5-5`) for work, subagents and tests. Do not use Claude Fable 5.1 unless it is absolutely needed (owner's instruction).
- Two Claude sessions may work on this repository: a cloud session and one on the owner's PC. Pull before starting, stay on the branch you were given, never force-push, and do not commit half-finished generated data.

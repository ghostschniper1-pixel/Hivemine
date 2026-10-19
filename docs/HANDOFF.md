# Handoff: Minecraft Hivemind moves to JeremyPC

**Date:** 2026-10-19.
**From:** the cloud Claude Code session that started the project (2026-10-16 to 2026-10-19).
**To:** the owner, and the Claude session that continues the work on JeremyPC.
**Branch:** `claude/zen-lamport-2wzdk7` on `ghostschniper1-pixel/Hivemine`.

From now on, the work continues on JeremyPC. The cloud session stops changing the repository once this letter is committed, so only one session writes to the branch.

## 1. Read these first

| File | What it is |
|---|---|
| `CLAUDE.md` | The working rules for any Claude session in this repository (loaded automatically) |
| `DESIGN.md` | The design, draft 0.5: goals, every decision with its reason, milestones M0–M11 |
| `docs/contracts.md` | The interfaces between modules. Change it first if an interface has to change |
| this file | Where things stand and what comes next |

## 2. The owner's decisions so far

Each is written into DESIGN.md with its reason; the decision numbers are in DESIGN.md section 14.

- **English for everything (D18).** Every action is an English command written word by word under a grammar mask (`mine 4 iron ore.`, `give 3 bread to red 7.`). Every symbol the agent sees is named in the same words, and the words must be true of what they name.
- **Explicit memory (D19).** Each agent keeps an honest notebook of places, players, chests and events (DESIGN.md 6.3).
- **Chests (D21).** `store` and `take` are in the vocabulary now, switched on at the team stage (M9).
- **LLM later, as one AI (D20, D22, D23).** Inert LLM ports and a shared goal section are reserved now. Coupling an LLM is milestone M11, kept last so that training starts sooner.
- **Surface-only vision (D15).** The agent sees only blocks in its line of sight. X-ray exists only as a labelled ablation and is never used for agents that are evaluated or deployed.
- **Two network sizes (D24).** S (about 5M parameters) and M (about 25–30M, the default), compared at equal wall-clock time at M5. On 2026-10-19 the owner accepted M's estimated cost: about 150 MFLOP per decision, and about 10 weeks of machine time for the team plan.
- **No panorama in S (D25).** Confirmed on 2026-10-19: S stays the lean baseline.
- **The other inputs.** M gets a 360-degree semantic panorama (D25). Both sizes get the debug-screen (F3) facts (D26) and biomes.
- **Models.** Use Claude Opus 5.5 for work and tests, not Claude Fable 5.1 unless absolutely needed.
- **Own server only.** Agents play only on the owner's firewalled server (DESIGN.md 11.6).

**Open question for the owner.** Two proposed gamerules must be settled before the M3 measurements: `doWeatherCycle false` (the simulator has no rain) and `spawnChunkRadius 0` (otherwise spawn chunks tick with no agent nearby, unlike the simulator). They are in `bridge/server/gamerules-1.21.1.mcfunction`.

## 3. Status by milestone

### M0: pin the version, build the rules, freeze the server config. Done, and checked against the real game

- **Rules (`data/rules-1.21.1.json`).** Exact 1.21.1 data built from pinned sources (`data/sources.json`): 1,060 blocks, 1,333 items, 1,290 recipes, every loot table and tag, and labelled constants.
- **Class tables (`data/concepts-1.21.1.json`).** 254 block classes, 163 item classes, 68 entity classes and 64 biomes. Each has an English phrase that is true of every member.
- **Vocabulary (`data/vocab-1.21.1.json`).** 561 words.
- **Checked on JeremyPC (2026-10-19).** The vanilla data generator agrees with the rules on every loot table, recipe, tag, block state range, item component and protocol id (2,523). The server config, all 51 gamerules and the difficulty were read back from a live server. Versions and hashes are pinned in `bridge/server/mods-1.21.1.json` (DESIGN.md 4.1 and 11.2).

### M1: the reference simulator. About two thirds done

**Done, each reviewed adversarially and fixed:**

| Module | What it does |
|---|---|
| `hivemind/sim/ref/rng.py`, `mechanics.py` | Counter-based randomness; exact formulas. Dig times are given for the client, the design formula and Mineflayer |
| `hivemind/lang/` | Vocabulary, grammar for 20 verbs, chat, goals and events; decoder with grammar masks; a static automaton for a later GPU or C++ mask builder; a parser for typed human text. All 331,407 commands round-trip |
| `hivemind/sim/ref/world.py`, `worldgen.py`, `ore_table.py` | Class-level voxel world, light, chunk ticking; deterministic test worlds with real 1.21.1 ore placement and biomes |
| `hivemind/sim/ref/perception.py`, `debug_info.py` | Surface-only window (S and M presets), semantic panorama, sightings, the F3 facts. Honesty is tested with adversarial worlds |
| `hivemind/sim/ref/nav.py` | Pathfinding with the bridge's mineflayer-pathfinder rules, checked against the real package run through Node |
| `hivemind/sim/ref/inventory.py`, `crafting.py`, `loot.py` | Inventory, class-level crafting, the furnace with vanilla timing, and the loot sampler |

**Not built yet** (contracts section 5):
- `actions.py`: the 20 verbs as interruptible macros, with masks, durations and failure codes;
- `mobs.py`, `rewards.py`, `goals.py`;
- `hivemind/features/memory.py`: the notebook, with the attribution rules of DESIGN.md 6.3;
- `hivemind/features/observation.py`: the observation builder;
- `env.py` and `viewer.py`;
- the M1 end-to-end test: a scripted agent that reaches an iron pickaxe in a test world by writing English commands.

### M2: bridge and calibration. Started

- **Done.**
  - The Mineflayer dig-time patch (`bridge/node/patches.js`) and its Node test. Unpatched, Mineflayer mines every tier-gated ore at hand speed in 1.21.1.
  - The protocol (`bridge/PROTOCOL.md`).
  - The frozen server config and `servertool.py` checks.
  - A pinned, working Fabric 1.21.1 server on JeremyPC (`C:\hivemind\servers\inst-0`).
- **Not built yet:**
  - the Node host;
  - the Python conductor;
  - the harness mod (a Fabric mod that logs break ticks on the server);
  - the M2 measurements.

### M3 to M11

Not started. See DESIGN.md section 12.

## 4. Next steps, in order

1. **Finish M1** (section 3). Build in the order the contracts imply:
   1. `features/memory.py` and `features/observation.py`;
   2. `actions.py` with its `SlotProvider` masks;
   3. `mobs.py`, `rewards.py` and `goals.py`;
   4. `env.py` and `viewer.py`;
   5. the end-to-end test.

   The simulator uses `BreakTime.design_ticks` for digs, which is what Mineflayer waits once the tool-speed patch is applied.
2. **M2 on JeremyPC.** Build the Node host and the conductor, then the harness mod. Then run the rule checks against the live server: server-logged dig ticks within ±1, every recipe, and drop distributions. Fit action durations and lag.
3. **M3:** real worlds from pregenerated chunks, and the measured spawn and ore-exposure rates.

## 5. Known approximations and open items

Each is recorded in the code or in DESIGN.md where it applies.

- **Class merging.** Members of an item class share stacks in the simulator, but not in the game. A bed needs three wool of one colour. Crafting can mix members of a class, so the bridge's `craft` must build a Mineflayer recipe from the items actually held.
- **Biomes.** Test worlds have biomes per column; the game stores them per 4×4×4 cell, so cave biomes cannot appear in test worlds.
- **Light** is per block in the rules data. Crops are treated as fully grown.
- **Moss carpet with a hoe:** patched Mineflayer waits 3 ticks, while the game takes 2 with a wooden hoe and breaks it instantly with a stone or better hoe.
- **`check-mods` reports NOT FROZEN** until the harness mod exists (M2). This is by design.
- **Values marked [mem] in the code** are from memory and not yet verified against the game. The M2 checks are where they get confirmed or corrected.

## 6. Working on JeremyPC

**Update the existing clone** (PowerShell, one line at a time). The two files checked out earlier are replaced by their newer versions; no other local change exists:

```powershell
cd $HOME\Hivemine
git fetch
git reset --hard origin/claude/zen-lamport-2wzdk7
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
cd bridge
npm ci
npm test
cd ..
```

**Rules for the work** (CLAUDE.md has the full list):
- Run the tests before every commit.
- Rebuild generated data in the order given in CLAUDE.md.
- Stay on this branch and never force-push.
- The "WIP checkpoint" commits in the history are snapshots taken while workers were mid-change. Later commits supersede them; ignore them.

**Running Claude on the PC.** There are three ways, which differ in billing:

| Command | Bills | Note |
|---|---|---|
| `claude` (signed in with the claude.ai account) | the claude.ai plan | Same plan as the cloud session; weekly limits apply |
| `.\tools\claude-api.ps1` | the API key's account | Works, but the $200 credit does not cover Claude Code ("Credit balance too low"); needs paid API balance |
| `.\tools\agent.ps1` (Agent SDK runner) | the API key's account | The credit lists the Agent SDK. Run one short prompt first, then check the Console usage page to confirm the credit pays. The credit expires 2026-10-29. Setup: `tools/README.md` |

**The server:** `C:\hivemind\servers\inst-0`, with the RCON password in the `HIVEMIND_RCON_PASSWORD` environment variable. The firewall block rules from `bridge/server/README.md` section 5 are in place.

## 7. How the cloud session worked (worth keeping)

- **Contracts first.** The interfaces in `docs/contracts.md` were fixed before code, so modules could be built in parallel.
- **Build, review, fix.** Every area was built by one agent, reviewed by an independent adversarial agent that had to prove each defect with evidence, and then fixed by a third agent that verified each finding before changing anything. This caught real bugs, including:
  - Mineflayer's wrong dig times;
  - an x-ray leak in the see-through rule;
  - fallback block classes named after the wrong blocks (coal blocks called "bell").
- **Exact checks.** Generated data is rebuilt byte for byte and hashed. Constants carry their source. Honesty is tested by changing hidden parts of the world and checking that nothing the agent sees changes.

## 8. A first prompt for the session on JeremyPC

> Read docs/HANDOFF.md, CLAUDE.md, DESIGN.md (sections 4.5, 5, 6, 7 and 12) and docs/contracts.md (sections 4.5, 5 and 6). Run the test suite and tell me the result. Then plan the rest of M1: `hivemind/features/memory.py`, `hivemind/features/observation.py`, `hivemind/sim/ref/actions.py`, `mobs.py`, `rewards.py`, `goals.py`, `env.py`, `viewer.py` and the M1 end-to-end test. Show me the plan before writing code. Use Claude Opus 5.5.

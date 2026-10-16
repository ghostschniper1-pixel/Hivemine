# Bridge: the real-game side

The bridge lets the same agent that trains in the simulator play real Minecraft (Java Edition 1.21.1), so that every result can be checked in the real game (DESIGN.md 2 principle 4, section 11). Movement and clicks are scripted Mineflayer macros; the agent still writes English commands such as `mine 4 iron ore.`, and sees the same words it sees in the simulator.

## What runs where

All on the owner's PC (Windows 11, Threadripper, 2× RTX 3090), all on loopback:

| Process | Language | Count (evaluation) | Job |
|---|---|---|---|
| **Conductor** | Python (asyncio) | 1 | Runs the policy on GPU 0, batching agents that become ready within 5–10 ms. Holds the one copy of everything shared with the simulator: the language (`hivemind.lang`: vocabulary, parser, masks), the perception function, the notebook memory and the featurizer. Resolves each command into a concrete plan, and drives the servers over RCON (setup, resets, `/tick`) |
| **Node hosts** | Node.js ≥ 22, Mineflayer | 10, with 4 bots each | Run the bots: wrappers per verb with deadlines and cleanup, verified digs, the per-bot lag shim, telemetry. They speak only vanilla names and never parse English |
| **Minecraft servers** | Java 21, Fabric 1.21.1 + Lithium + Carpet + harness mod | 4 instances, 10 bots each | The ground truth. Frozen config in `server/` |

```
Python conductor ── TCP 127.0.0.1:7600, u32 length + msgpack (PROTOCOL.md) ── Node hosts ×10 (4 bots each)
   │ same featurizer, perception, notebook, language and masks as the simulator
   └─ RCON 127.0.0.1:25575+K (/tick, setup, resets) ── Fabric 1.21.1 servers, K = 0..3 (game ports 25565+K)
```

How a decision flows: the policy writes `mine 4 iron ore.` word by word under the grammar mask; the conductor parses it (one parser, `hivemind.lang`), looks up known iron ore in the agent's notebook, and sends `act` with the text, the parsed form and a plan with those block positions. The Node host runs the `mine` wrapper and answers `result` with a failure code and the result words (`done`, `interrupted by zombie`). Between decisions it streams the bot's state, the nearby blocks and events (chat, damage, deaths, kicks, position corrections), from which the conductor updates the notebook and builds the next observation.

## Memory in the real game

Each agent remembers where things were, who is an enemy and what is in its chests through its **notebook** (DESIGN.md 6.3, contracts 6.1), and the same notebook code runs in the simulator and here. In the real game it is filled only from what the bot's own client received: blocks the bot can see (perception runs in Python on a mirror of the nearby chunks), players with their team colour, chat, damage with its source, deaths, pickups, chest lids and the chest contents the bot saw when it stored or took items. The client is told more than the agent may use (entities behind walls, chest lids within about 64 blocks [mem] with a viewer count but no name), so the conductor runs the simulator's perception before the notebook names anyone, exactly as in the simulator. PROTOCOL.md section 9 maps every notebook table to the messages that feed it. Nothing comes from RCON or from searches over all loaded blocks, so the real-game agent knows no more than the simulated one.

## Files

| Path | What | Status |
|---|---|---|
| `PROTOCOL.md` | The conductor ↔ Node protocol, version 1: framing, handshake and versioning, `act`/`cancel`/`result`, state stream, events, timing | Specification; provisional parts marked |
| `package.json`, `package-lock.json` | Node dependencies, pinned exactly (below) | Done |
| `server/` | Frozen `server.properties`, gamerules and the vanilla defaults of the others, team setup, mod pins, `servertool.py` with its offline tests (`test_servertool.py`), Windows setup guide | Done, except the TODO-pin versions and hashes (see `server/README.md`) |
| `.gitattributes` | LF line endings in `bridge/` on every OS, so recorded file hashes match on Windows | Done |
| conductor, Node hosts, wrappers, lag shim | | **Not built** (M2, M7) |
| harness mod | | **Not built** (M2) |

## Node packages

Pinned exactly in `package.json`; the full tree is locked in `package-lock.json`. Install on the owner's PC with `npm ci` inside `bridge/` (Node ≥ 22). `node_modules/` is git-ignored.

| Package | Pinned | npm latest, 2026-10-16 | Engines | Note |
|---|---|---|---|---|
| `mineflayer` | 4.39.0 | 4.39.0 | node ≥ 22 | 1.21.1 is in its tested versions |
| `mineflayer-pathfinder` | 2.4.5 | 2.4.5 | — | last published 2023-09-04; the defaults differ from DESIGN.md 4.5, so the conductor sends the settings (PROTOCOL.md 4.4) |
| `mineflayer-collectblock` | 1.6.0 | 1.6.0 | — | |
| `mineflayer-tool` | 1.2.0 | 1.2.0 | — | last published 2022-05-11 |
| `mineflayer-armor-manager` | 2.0.1 | 2.0.1 | node ≥ 18 | off for learners: equipping is a policy action |
| `mineflayer-auto-eat` | 5.0.3 | 5.0.3 | — | off for learners: eating is a policy action. ES module only (`"type": "module"`); load it with `import`. It pulls in two GPL-3.0 packages (`@nxg-org/mineflayer-util-plugin`, `node-aabb`); fine for private use, check before distributing anything |
| `@msgpack/msgpack` | 3.1.3 | 3.1.3 | node ≥ 18 | |

All seven match the versions DESIGN.md 11.1 recorded on 2026-10-13; there was no newer release by 2026-10-16. Resolved in the lock: `minecraft-data` 3.117.0, `minecraft-protocol` 1.68.0 (needs node ≥ 22), `prismarine-physics` 1.11.1, `prismarine-block` 1.23.0. The plugins ask for `mineflayer` 4.x (`^4.0.0` up to `^4.30.0`), and npm dedupes them all onto the pinned 4.39.0.

**minecraft-data matches the rules build.** minecraft-data 3.117.0 maps 1.21.1 to protocol version 767 (data version 3955), and the eleven minecraft-data files pinned in `data/sources.json` (blocks, items, foods, materials, recipes, language, entities, collision shapes, loot) are byte-identical in the npm package (SHA-256 checked). So the block state ids and names Mineflayer reports are the ones `data/rules-1.21.1.json` and the class tables were built from. The Node host checks this again at every start (`hello`, PROTOCOL.md 4).

Re-pinning: run `npm view <pkg> version engines` for each package, change `package.json`, run `npm install --package-lock-only --ignore-scripts`, check `npm ls --package-lock-only`, and re-run the minecraft-data hash comparison. A new Mineflayer is a bridge change: rerun the M2 rule and action checks (DESIGN.md 11.4).

## Not built yet

- **M2, minimal bridge:** the harness mod (server-side break-tick log, player-state restore, chunk-ticking report); the conductor's protocol server and a Node host with wrappers for `goto`, `mine` and `craft`; deadlines and cleanup; verified digs; the lag shim; telemetry. Exit: server-logged digs within ±1 tick, recipes 100%, duration fits (DESIGN.md 12).
- **M7, full bridge:** all 19 verbs, the world mirror at scale, evaluation areas, resets and the 4-instance orchestration, the 40-bot ramp test (MSPT p95 < 40 ms, Node loop delay p99 < 20 ms) and the calibration thresholds of DESIGN.md 11.4.

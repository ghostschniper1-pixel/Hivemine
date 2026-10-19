# Bridge: the real-game side

The bridge lets the same agent that trains in the simulator play real Minecraft (Java Edition 1.21.1), so that every result can be checked in the real game (DESIGN.md 2 principle 4, section 11). Movement and clicks are scripted Mineflayer macros; the agent still writes English commands such as `mine 4 iron ore.`, and sees the same words it sees in the simulator.

## What runs where

All on the owner's PC (Windows 11, Threadripper, 2× RTX 3090), all on loopback:

| Process | Language | Count (evaluation) | Job |
|---|---|---|---|
| **Conductor** | Python (asyncio) | 1 | Runs the policy on GPU 0, batching agents that become ready within 5–10 ms. Holds the one copy of everything shared with the simulator: the language (`hivemind.lang`: vocabulary, parser, masks), the perception functions (the visible-only window, the panorama, the F3 group), the notebook memory, the resolver and the featurizer. Resolves each command into a concrete plan, and drives the servers over RCON (setup, resets, `/tick`) |
| **Node hosts** | Node.js ≥ 22, Mineflayer | 10, with 4 bots each | Run the bots: wrappers per verb with deadlines and cleanup, verified digs, the per-bot lag shim, telemetry. They speak only vanilla names and never parse English |
| **Minecraft servers** | Java 21, Fabric 1.21.1 + Lithium + Carpet + harness mod | 4 instances, 10 bots each | The ground truth. Frozen config in `server/` |

```
Python conductor ── TCP 127.0.0.1:7600, u32 length + msgpack (PROTOCOL.md) ── Node hosts ×10 (4 bots each)
   │ same featurizer, perception, notebook, language, resolver and masks as the simulator
   └─ RCON 127.0.0.1:25575+K (/tick, setup, resets) ── Fabric 1.21.1 servers, K = 0..3 (game ports 25565+K)
```

How a decision flows: the policy writes `mine 4 iron ore.` word by word under the grammar mask; the conductor parses it (one parser, `hivemind.lang`), looks up known iron ore in the agent's notebook with the shared resolver, and sends `act` with the text, the parsed form and a plan with those block positions. The Node host runs the `mine` wrapper and answers `result` with a failure code and the result words (`done`, `interrupted by zombie`). Between decisions it streams the bot's state, the nearby blocks with their light and biomes, and events (chat, damage, deaths, kicks, position corrections), from which the conductor updates the notebook and builds the next observation.

## What the agent sees in the real game

Exactly what it sees in the simulator, computed by the same code (DESIGN.md 6.1, 6.2). The conductor keeps a mirror of the blocks, light and biomes the bot's client holds, clipped to the bot's evaluation area (barrier outside, as at the simulator's world edge), and runs the shared perception on it: the voxel window shows only cells a straight line from the eye reaches through see-through cells (`unseen` for the rest), preset M adds the 96 × 24 panorama out to 64 blocks, and the F3 group gives position relative to home, facing, the targeted block, fluid and entity, the time, the biome and the light at the feet. Nothing comes from the client's debug screen. PROTOCOL.md 7 says what the mirror holds and how far it reaches (4 chunks around the bot's chunk in preset M, 2 in S).

## Memory in the real game

Each agent remembers where things were, who is an enemy and what is in its chests through its **notebook** (DESIGN.md 6.3, contracts 6.1), and the same notebook code runs in the simulator and here. In the real game it is filled only from what the bot's own client received: blocks the bot can see (perception on the mirror), players with their team colour, chat, damage with its source, deaths, pickups, chest lids and the chest contents the bot saw when it stored or took items. The client is told more than the agent may use (entities behind walls, chest lids within about 64 blocks [mem] with a viewer count but no name), so the conductor runs the simulator's perception before the notebook names anyone, exactly as in the simulator. PROTOCOL.md section 9 maps every notebook table to the messages that feed it. Nothing comes from RCON or from searches over all loaded blocks, so the real-game agent knows no more than the simulated one.

## Files

| Path | What | Status |
|---|---|---|
| `PROTOCOL.md` | The conductor ↔ Node protocol, version 1: framing, handshake and versioning, host startup order, `act`/`cancel`/`result` and the plan per verb, state stream and world mirror, events, timing | Specification, aligned with DESIGN.md draft 0.5; provisional parts marked |
| `package.json`, `package-lock.json` | Node dependencies, pinned exactly (below) | Done |
| `node/` | Node host code. So far `patches.js`, the dig-time patch every host must apply before it creates a bot, with its test (`test/patches.test.js`) and fixture (`test/dig_times.json`, made by `make_dig_fixture.py` from the reference mechanics). `node/README.md` also lists what the pinned packages do that a host must work around. Run the test from `bridge/` with Node 22 or newer: `npm ci --ignore-scripts` once, then `npm test` | Patch done; host not built (M2) |
| `server/` | Frozen `server.properties`, gamerules and the vanilla defaults of the others, team setup, mod pins, `servertool.py` with its offline tests (`test_servertool.py`), Windows setup guide | Done, except the TODO-pin versions and hashes (see `server/README.md`) |
| `.gitattributes` | LF line endings in `bridge/` on every OS, so recorded file hashes match on Windows | Done |
| conductor, Node hosts, wrappers, lag shim | | **Not built** (M2, M7) |
| harness mod | | **Not built** (M2) |

## Node packages

Pinned exactly in `package.json`; the full tree is locked in `package-lock.json`. Install on the owner's PC with `npm ci` inside `bridge/` (Node ≥ 22). `node_modules/` is git-ignored.

| Package | Pinned | npm latest, 2026-10-16 | Engines | Note |
|---|---|---|---|---|
| `mineflayer` | 4.39.0 | 4.39.0 | node ≥ 22 | 1.21.1 is in its tested versions |
| `mineflayer-pathfinder` | 2.4.5 | 2.4.5 | — | last published 2023-09-04; the defaults differ from DESIGN.md 4.5, so the conductor sends the settings (PROTOCOL.md 4.4, 5.1) |
| `mineflayer-collectblock` | 1.6.0 | 1.6.0 | — | |
| `mineflayer-tool` | 1.2.0 | 1.2.0 | — | last published 2022-05-11 |
| `mineflayer-armor-manager` | 2.0.1 | 2.0.1 | node ≥ 18 | off for learners: equipping is a policy action |
| `mineflayer-auto-eat` | 5.0.3 | 5.0.3 | — | off for learners: eating is a policy action. ES module only (`"type": "module"`); load it with `import`. It pulls in two GPL-3.0 packages (`@nxg-org/mineflayer-util-plugin`, `node-aabb`); fine for private use, check before distributing anything |
| `@msgpack/msgpack` | 3.1.3 | 3.1.3 | node ≥ 18 | |
| `prismarine-block` | 1.23.0 | 1.23.0 (checked 2026-10-19) | — | declared because `node/patches.js` and its test load it; the version the lock already resolved under Mineflayer |
| `prismarine-registry` | 1.12.0 | 1.12.0 (checked 2026-10-19) | — | as above |

The first seven match the versions DESIGN.md 11.1 recorded on 2026-10-13; there was no newer release by 2026-10-16. The last two were declared on 2026-10-19 at the versions already in the lock, and regenerating the lock (`npm install --package-lock-only --ignore-scripts`) changed only its root entry. Resolved in the lock: `minecraft-data` 3.117.0, `minecraft-protocol` 1.68.0 (needs node ≥ 22), `prismarine-physics` 1.11.1, `prismarine-chunk` 1.41.0, `prismarine-recipe` 1.5.0. The plugins ask for `mineflayer` 4.x (`^4.0.0` up to `^4.30.0`), and npm dedupes them all onto the pinned 4.39.0.

**minecraft-data matches the rules build.** minecraft-data 3.117.0 maps 1.21.1 to protocol version 767 (data version 3955), and the twelve minecraft-data files pinned in `data/sources.json` (version, blocks, items, foods, materials, recipes, language, entities, collision shapes, biomes, block and entity loot) are byte-identical in the npm package (SHA-256 checked on 2026-10-19). So the block state ids and names Mineflayer reports are the ones `data/rules-1.21.1.json` and the class tables were built from. The Node host checks this again at every start (`hello`, PROTOCOL.md 4).

**Dig times: one known difference.** With the dig-time patch (`node/patches.js`, DESIGN.md 5), Mineflayer's dig wait equals the design formula for every block, tool and condition except **moss carpet with a hoe**: patched Mineflayer waits 3 ticks on the ground with any hoe (minecraft-data's material `plant` lists no hoes), while the game takes 2 ticks with a wooden hoe and breaks the carpet instantly with a stone or better hoe. Mineflayer is only slower there; the server never refuses the break. `node/README.md` has the details.

Re-pinning: run `npm view <pkg> version engines` for each package, change `package.json`, run `npm install --package-lock-only --ignore-scripts`, check `npm ls --package-lock-only`, and re-run the minecraft-data hash comparison. A new Mineflayer is a bridge change: rerun the M2 rule and action checks (DESIGN.md 11.4), both test suites, and the checks of `node/README.md` ("What the pinned packages do").

## Not built yet

- **M2, minimal bridge:** the harness mod (server-side break-tick log, player-state restore, chunk-ticking report); the conductor's protocol server and a Node host with wrappers for `goto`, `mine` and `craft`, set up in the order of PROTOCOL.md 5.1; deadlines and cleanup; verified digs; the lag shim; telemetry. Exit: server-logged digs within ±1 tick, recipes 100%, duration fits (DESIGN.md 12).
- **M7, full bridge:** all 20 verb ids, the world mirror at scale, evaluation areas, resets and the 4-instance orchestration, the 40-bot ramp test (MSPT p95 < 40 ms, Node loop delay p99 < 20 ms) and the calibration thresholds of DESIGN.md 11.4.
- **Shared code the conductor needs from `hivemind/`:** the resolver (`resolve.py`), a world that holds the game's light, and the biome zoom (PROTOCOL.md 13).

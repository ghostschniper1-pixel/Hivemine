# Node host code

Code for the Node hosts that run the Mineflayer bots (bridge/README.md, PROTOCOL.md). So far this folder holds one piece: the dig-time patch. The hosts themselves come in M2.

## The dig-time patch (`patches.js`)

Mineflayer's `bot.dig` sends "start", waits `block.digTime(...)`, then sends "finished". prismarine-block 1.23.0 computes `digTime` with a tool speed looked up through minecraft-data's per-block `material`. In minecraft-data 3.117.0 (pc/1.21.1), all 101 tier-gated blocks (the members of the block tag `incorrect_for_wooden_tool`: every metal and gem ore and its deepslate form, obsidian, the metal blocks, the copper family, ...) have the material `incorrect_for_wooden_tool`, which lists only the four wooden tools. Unpatched, Mineflayer waits:

| Block, tool | Game (DESIGN.md 4.3) | Unpatched Mineflayer |
|---|---|---|
| obsidian, diamond pickaxe | 188 ticks | 1,500 ticks |
| iron ore, stone pickaxe | 23 ticks | 91 ticks |
| iron ore, wooden axe | 300 ticks | 150 ticks: "finished" comes before the server accepts it, so the dig fails |

`applyDigTimePatch(registry)` gives those 101 blocks the material `mineable/pickaxe`. This is correct because in the 1.21.1 tags every one of them is mineable with a pickaxe and with no other tool kind. The patch checks this at runtime against the tag list embedded in `patches.js`, logs the count (101 expected), and throws `DigTimePatchError` without changing anything if the data is not what it was written for (another game version, a changed material set, or `mineable/pickaxe` listing something other than pickaxes). Calling it twice is harmless. Harvest rules are not touched; minecraft-data's `harvestTools` are already right.

With the patch, Mineflayer's wait equals DESIGN.md 4.3's formula for every block, tool and condition, with one exception that stays: **moss carpet with a hoe**. minecraft-data's material `plant` lists no hoes, so Mineflayer waits 3 ticks on the ground (15 in water or in the air) whatever the hoe. The game takes 2 ticks with a wooden hoe and breaks the carpet instantly with a stone or better hoe. Mineflayer is only slower here; the server never refuses the break.

### The Node host (M2) must call it first

A `Block` copies `material` from the registry when it is constructed, so the patch must run **before any Block object exists**, that is, before the first `mineflayer.createBot`. Both lines below are required:

```js
const mineflayer = require('mineflayer')
const { applyDigTimePatch, assertDigTimePatched, mineflayerRegistry } = require('./node/patches') // from bridge/

applyDigTimePatch(mineflayerRegistry()) // 1. first thing at host startup, once per process

const bot = mineflayer.createBot(options)
bot.once('inject_allowed', () => assertDigTimePatched(bot.registry)) // 2. for every bot
```

One call of `applyDigTimePatch` is enough for the whole process. The patch edits the block definitions in place, and they belong to minecraft-data's per-version cache. Every registry for 1.21.1 made by the same copy of prismarine-registry shares that cache, including each bot's `bot.registry`, which Mineflayer makes with `require('prismarine-registry')(version)` from its own folder. `mineflayerRegistry()` therefore loads prismarine-registry from Mineflayer's folder, not by bare name: if a future lock nested a second copy under `mineflayer`, a registry required by bare name would patch a cache that no bot reads.

The per-bot check is mandatory. `assertDigTimePatched(bot.registry)` throws a `DigTimePatchError` unless all 101 tier-gated blocks of that bot's registry have the patched material; thrown from the `inject_allowed` listener, it ends the host process before the bot has loaded a chunk. A bot must never play unpatched, because nothing else would show it: it would simply mine the ore tech tree several times too slowly with every pickaxe but the wooden one (iron ore with a stone pickaxe: 91 ticks, not 23), and fail digs with a wooden axe, shovel or hoe. Mineflayer emits `inject_allowed` after setting `bot.registry` and before any chunk arrives (mineflayer 4.39.0, `lib/loader.js`).

Patching only `bot.digTime` would not be enough: mineflayer-tool and mineflayer-pathfinder call `block.digTime` directly.

## Tests

From `bridge/`, with Node 22:

```sh
npm ci --ignore-scripts   # once; node_modules/ is git-ignored
npm test                  # node --test "node/test/*.test.js"
```

The script names the test files with a glob because Node 22's `--test` takes file patterns, not a directory (`node --test node/test` fails with "Cannot find module").

`test/patches.test.js` loads prismarine-registry and prismarine-block from Mineflayer's own folder (as the bots do) for 1.21.1 and compares `digTime` with `test/dig_times.json` for every fixture pair: the 101 tier-gated blocks, the coal ores, stone, deepslate and their cobbles, all 40 members of the tag `logs`, dirt, grass block, sand, red sand, gravel and moss carpet, each with the empty hand and every v1 tool (26), on the ground, in water and in the air. It checks that:

- the copies Mineflayer resolves are prismarine-block 1.23.0, prismarine-registry 1.12.0 and minecraft-data 3.117.0, and Mineflayer, prismarine-chunk (`bot.blockAt`) and mineflayer-pathfinder all load the same prismarine-block;
- unpatched `digTime` matches the reference model of unpatched Mineflayer for every pair, and is wrong on every tier-gated block;
- the patch changes exactly those 101 blocks and logs the count;
- patched `digTime` equals the design formula for every pair except moss carpet with a hoe;
- a Block made before the patch keeps the old material (why the host must patch first), and a registry made afterwards shares the patched definitions and passes the per-bot check;
- the per-bot check refuses a registry that is unpatched, partly patched or for another version;
- the patch refuses data it was not written for.

The fixture comes from the reference mechanics (`hivemind.sim.ref.mechanics`), never from Node. Regenerate it from the repository root after any change to the mechanics, the rules or the v1 tool classes:

```sh
.venv/bin/python bridge/node/make_dig_fixture.py           # rewrite test/dig_times.json
.venv/bin/python bridge/node/make_dig_fixture.py --check   # compare only
```

The generator fails if a tier-gated block is not pickaxe-only or if patched Mineflayer would differ from the formula anywhere except moss carpet with a hoe. The Python suite (`tests/test_mechanics.py`) fails if the committed fixture is out of date.

Re-pinning minecraft-data or prismarine-block is a bridge change (bridge/README.md): rerun both test suites. If the material set changes, the patch throws at startup rather than mining at the wrong speed.

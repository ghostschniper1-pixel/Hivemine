# Node host code

Code for the Node hosts that run the Mineflayer bots (bridge/README.md, PROTOCOL.md). So far this folder holds one piece, the dig-time patch, with its test. The hosts come in M2: PROTOCOL.md 5.1 fixes their startup order, and the section "What the pinned packages do" below lists what a host must work around.

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

A `Block` copies `material` from the registry when it is constructed, so the patch must run **before any Block object exists**: before the first `mineflayer.createBot`, before any `new Movements(bot)` (its constructor makes a Block for every block type), and before mineflayer-pathfinder is loaded into a bot (its inject makes a `Movements`). Both patch lines below are required; the rest is the bot setup of PROTOCOL.md 5.1:

```js
'use strict'
const mineflayer = require('mineflayer')
const { pathfinder, Movements } = require('mineflayer-pathfinder')
const toolPlugin = require('mineflayer-tool').plugin
const patches = require('./node/patches') // from bridge/

patches.applyDigTimePatch(patches.mineflayerRegistry()) // 1. first thing at host startup, once per process

// ... connect to the conductor and exchange hello (PROTOCOL.md 4); `settings` is the conductor's hello.settings

function startBot (join, settings) {
  const bot = mineflayer.createBot({
    host: join.server.host, port: join.server.port, username: join.username,
    version: '1.21.1', auth: 'offline', respawn: false
  })
  bot.once('inject_allowed', () => patches.assertDigTimePatched(bot.registry)) // 2. for every bot
  bot.loadPlugin(pathfinder) // armor-manager and auto-eat stay unloaded: equipping and eating are actions
  bot.loadPlugin(toolPlugin)
  bot.once('spawn', () => {
    const pf = settings.pathfinder // built from hivemind/sim/ref/nav.py (PROTOCOL.md 5.1)
    const m = new Movements(bot)
    Object.assign(m, pf.movements) // allowParkour false, allowEntityDetection false, ...
    m.scafoldingBlocks = pf.scafoldingBlocks.map(name => bot.registry.itemsByName[name].id) // dirt, cobblestone
    for (const name of pf.blocksCantBreak) m.blocksCantBreak.add(bot.registry.blocksByName[name].id)
    for (const name of pf.blocksToAvoid) m.blocksToAvoid.add(bot.registry.blocksByName[name].id)
    bot.pathfinder.setMovements(m)
    Object.assign(bot.pathfinder, pf.search) // tickTimeout 10, thinkTimeout 2000, searchRadius 96
  })
  return bot
}
```

One call of `applyDigTimePatch` is enough for the whole process. The patch edits the block definitions in place, and they belong to minecraft-data's per-version cache. Every registry for 1.21.1 made by the same copy of prismarine-registry shares that cache, including each bot's `bot.registry`, which Mineflayer makes with `require('prismarine-registry')(version)` from its own folder. `mineflayerRegistry()` therefore loads prismarine-registry from Mineflayer's folder, not by bare name: if a future lock nested a second copy under `mineflayer`, a registry required by bare name would patch a cache that no bot reads.

The per-bot check is mandatory. `assertDigTimePatched(bot.registry)` throws a `DigTimePatchError` unless all 101 tier-gated blocks of that bot's registry have the patched material; thrown from the `inject_allowed` listener, it ends the host process before the bot has loaded a chunk. A bot must never play unpatched, because nothing else would show it: it would simply mine the ore tech tree several times too slowly with every pickaxe but the wooden one (iron ore with a stone pickaxe: 91 ticks, not 23), and fail digs with a wooden axe, shovel or hoe. Mineflayer emits `inject_allowed` after setting `bot.registry` and before any chunk arrives (mineflayer 4.39.0, `lib/loader.js`).

Patching only `bot.digTime` would not be enough: mineflayer-tool and mineflayer-pathfinder call `block.digTime` directly.

## What the pinned packages do

Read from the locked sources on 2026-10-19 and, where a behaviour is stated, run. Each item says where PROTOCOL.md handles it.

- **Light getters read the wrong cell.** prismarine-chunk 1.41.0 reads network light as big-endian 64-bit words, so `column.getSkyLight`, `column.getBlockLight` and `bot.blockAt(pos).light` / `.skyLight` return the light of the cell x' = 14 − (x & ~1) + (x & 1) in the same row, while its disk loader uses the game's nibble layout. The host forwards the light bytes as received (`map_chunk` and `update_light` packets, or `column.dumpLight()`, which returns them unchanged) and never reads light through the getters (PROTOCOL.md 7.2).
- **`bot.time` is wrong above 2^31 and stale between packets.** mineflayer's `longToBigInt` (`lib/plugins/time.js`) turns a value whose low 32 bits have the top bit set negative (2^31 + 5 becomes −2^31 + 5), and nothing advances `bot.time` between `update_time` packets. The host reads `age` and `time` from the raw packet with `BigInt(...)` (protodef's i64 decodes correctly) and estimates the ticks since (PROTOCOL.md 7.1).
- **Angles.** Mineflayer's yaw is in radians, 0 at north, counterclockwise; its pitch is positive upward. The protocol carries Minecraft degrees, converted as Mineflayer converts when it sends the look (PROTOCOL.md section 3).
- **`bot.equip(item, 'hand')` is not the simulator's hold.** It takes the first stack in slots 9–35 even when the hotbar holds the item, and with a full hotbar it rotates through the hotbar slots (`lib/plugins/simple_inventory.js`). The `equip` wrapper implements `Inventory.hold`'s rule with `setQuickBarSlot` and `moveSlotItem` (PROTOCOL.md 6.2).
- **`bot.recipesFor` cannot mix.** minecraft-data's recipes list one recipe per member (11 crafting-table recipes, each with four planks of one kind; no shape among the 1,470 recipes mixes planks), so a mix the game accepts finds no recipe. The `craft` wrapper builds a `Recipe` from the plan's shape with the prismarine-recipe copy Mineflayer loads, `require(patches.packageDir('prismarine-recipe', patches.packageDir('mineflayer', __dirname)))(bot.registry).Recipe`, and calls `bot.craft(recipe, passes, table)`, which finds each slot's item by id (PROTOCOL.md 6.2).
- **The flee goal** is `nav.GoalFlee` ported to a `goals.Goal` subclass, not `GoalInvert` (PROTOCOL.md 5.1).
- **`scafoldingBlocks`** is the package's spelling of the Movements field (`lib/movements.js`).
- **64-bit values.** @msgpack/msgpack 3.1.3 refuses a `BigInt` unless `useBigInt64` is set and encodes integers beyond 2^53 as float64, so the hashed seed travels as a decimal string (PROTOCOL.md section 2, 5).
- **Chat type.** Mineflayer's `message` event passes `'chat'` for every player message; the host reads the raw `playerChat` packet (PROTOCOL.md 8).

## Tests

From `bridge/`, with Node 22 or newer:

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

`bridge/package.json` declares prismarine-block 1.23.0 and prismarine-registry 1.12.0, the versions the lock already resolved under Mineflayer, because `patches.js` and this test load them directly; declaring them changed nothing in the lock but its root entry.

The fixture comes from the reference mechanics (`hivemind.sim.ref.mechanics`), never from Node. Regenerate it from the repository root after any change to the mechanics, the rules or the v1 tool classes:

```sh
.venv/bin/python bridge/node/make_dig_fixture.py           # rewrite test/dig_times.json
.venv/bin/python bridge/node/make_dig_fixture.py --check   # compare only
```

The generator fails if a tier-gated block is not pickaxe-only or if patched Mineflayer would differ from the formula anywhere except moss carpet with a hoe. The Python suite (`tests/test_mechanics.py`) fails if the committed fixture is out of date.

Re-pinning minecraft-data, prismarine-block or prismarine-chunk is a bridge change (bridge/README.md): rerun both test suites and recheck the list above. If the material set changes, the patch throws at startup rather than mining at the wrong speed.

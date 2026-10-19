# Conductor ↔ Node host protocol, version 1

The wire protocol between the Python conductor and the Node hosts that run the Mineflayer bots (DESIGN.md 11.1, contracts section 7).

**Status:** a specification for M2 (minimal bridge) and M7 (full bridge), aligned on 2026-10-19 with DESIGN.md draft 0.5 and with what `hivemind/` builds. Nothing implements the protocol yet; `node/patches.js` is the only host code so far. Parts marked **provisional** are expected to change when the wrappers are written in M2; changes that only add optional fields keep protocol version 1 (section 4.3). Labels as in DESIGN.md: [n] a reference there, [mem] unverified memory, [est] an estimate, [design] a choice made here. Facts marked *(checked)* were read on 2026-10-16 or 2026-10-19 from the pinned package sources (mineflayer 4.39.0, minecraft-protocol 1.68.0, mineflayer-pathfinder 2.4.5, prismarine-registry 1.12.0, prismarine-block 1.23.0, prismarine-chunk 1.41.0, prismarine-recipe 1.5.0, prismarine-windows 2.10.0, protodef 1.19.0, @msgpack/msgpack 3.1.3, installed with `npm ci` from `package-lock.json`), from the pinned mcmeta 1.21.1 data, or from `hivemind/` as built. The behaviours of @msgpack/msgpack, of prismarine-chunk's light, of Mineflayer's time decoding and angle conversions, and the counts of minecraft-data's recipes were also run under Node 22.

Three rules shape everything below:
1. **Node never parses English.** The conductor sends the command text (for logs), the parsed form from `hivemind.lang` and a concrete `plan`, which is what Node executes. There is one parser and one resolver, both on the Python side (DESIGN.md 5; contracts 5 `resolve.py` and 7).
2. **Node never decides what the agent knows.** It forwards what the bot's own client received (blocks, light, biomes, entities, chat, damage, windows). Perception (the visible-only voxel window, the panorama, the F3 group), the notebook, masks and every class or phrase live in Python, shared with the simulator (DESIGN.md 2 principle 6, 6.1, 6.2). Targets come from the notebook, never from `findBlocks` (no X-ray). The bridge never runs the X-ray ablation: it is never used for an agent that is evaluated or deployed (DESIGN.md 6.1).
3. **Node speaks vanilla; machine fields speak `hivemind.lang`.** Blocks, items, entities and biomes are vanilla names without the `minecraft:` prefix (`deepslate_iron_ore`, `raw_iron`, `zombie`, `birch_forest`) and block state ids. Python maps them to classes and words with `data/concepts-1.21.1.json` and `data/block_classes-1.21.1.csv`. Players and parsed commands use `hivemind.lang`'s machine forms: concept names (`red_0`, `human`) and `Command.to_json()`. English phrases (`red 0`) appear only in English text: the command, the chat line and the result words.

## 1. Processes and connections

```
Python conductor (asyncio) ── TCP 127.0.0.1:7600, u32 length + msgpack ── Node hosts ×10 (4 bots each)
   └─ RCON 127.0.0.1:25575+K (setup, resets, /tick) ── Fabric 1.21.1 servers, K = 0..3
Node hosts ── Minecraft protocol 1.21.1 ── 127.0.0.1:25565+K
```

- The conductor listens on `127.0.0.1:${HIVEMIND_BRIDGE_PORT}` (default 7600) and starts the Node hosts itself, passing `HIVEMIND_BRIDGE_PORT`, `HIVEMIND_SESSION` (a random id per conductor run) and `HIVEMIND_HOST_ID`. Each host opens one TCP connection, which carries the traffic of all its bots (at most 4–5 per host, DESIGN.md 11.1).
- Both sides set `TCP_NODELAY`. Without it, Nagle's algorithm can hold back small messages, such as a result, for tens of milliseconds [mem].
- Bots connect to `127.0.0.1` (not `localhost`, which may resolve to `::1` while the server binds IPv4 only) and pass `version: '1.21.1'` explicitly. minecraft-protocol 1.68.0's default version is 26.1 *(checked: `src/version.js`)*; 1.21.1 is in its supported list and in mineflayer 4.39.0's tested versions *(checked)*.
- RCON is the conductor's alone and never feeds observations.

## 2. Framing and encoding

- A frame is a **u32 big-endian length** followed by that many bytes of payload. The length excludes its own 4 bytes. The payload is exactly one msgpack value, always a map.
- Maximum payload: 16 MiB. A larger length is a protocol error: the receiver sends `bye` and closes.
- msgpack subset: nil, bool, int (up to 64-bit), float64, str (UTF-8), bin, array, and map with str keys. No extension types (no msgpack timestamps), no float32.
- @msgpack/msgpack encodes every safe integer (|n| ≤ 2^53 − 1) as an int, including `1.0`, and every other number, 2^53 included, as float64 *(checked)*. **A receiver must accept an int wherever a float is expected.** A `BigInt` throws unless the encoder's `useBigInt64` option is set *(checked)*, so a 64-bit value that can exceed 2^53 travels as a decimal string (`hashed_seed`, section 5). A `Uint8Array` or `Buffer` encodes as bin and decodes as `Uint8Array` *(checked)*.
- Optional fields may be absent or nil; receivers treat both alike and ignore unknown fields. (An `undefined` value in a JS map encodes as nil *(checked)*.)
- The conductor's runtime code may use only the standard library and NumPy if it lives under `hivemind/` (contracts section 0). This subset is small enough for a stdlib codec; otherwise the `msgpack` package can be used outside `hivemind/`. Deflated fields (section 7.2) use zlib, which both standard libraries have.

Example: `{"t": "ping", "nonce": 1}` is the frame `00 00 00 0f 82 a1 74 a4 70 69 6e 67 a5 6e 6f 6e 63 65 01` *(checked with @msgpack/msgpack 3.1.3)*.

## 3. Common fields and conventions

| Field | Type | Meaning |
|---|---|---|
| `t` | str | message type (`hello`, `act`, `result`, ...) |
| `seq` | int | per connection and direction, 1, 2, 3, ...; for logs and gap detection |
| `agent` | str | the agent's concept name: `red_0`, `blue_7` (contracts 4.2) |
| `tick` | int | the bot's client tick counter: Mineflayer's `physicsTick`, one per 50 ms of wall time *(checked: `PHYSICS_INTERVAL_MS = 50`)*, counted from the bot's spawn |
| `age` | int | the server's world age, estimated: the age in the last `update_time` packet plus the client ticks since it arrived, at most 20 ahead. The server sends `update_time` every 20 ticks [mem], so `age` is coarse; exact server-side ticks come from the harness mod's log (M2, DESIGN.md 11.3) |
| `wall_ms` / `*_ms` | float | milliseconds; absolute times are Unix epoch ms (both processes share one clock) |

- **Players are concept names** in every machine field: `red_0` … `red_19`, `blue_0` … `blue_19` and `human`, never phrases (contracts 4.2, 4.4). Their phrases (`red 0`, `blue 19`, `human`) appear only in English text. An agent's in-game username is its concept name (`red_0`, `blue_19`): 3–16 characters from `[A-Za-z0-9_]` [mem]. The human's username comes from the conductor's config. The conductor's `hello` sends every player's username (`players`, 4.2), and Node translates at its edge: usernames from the server become concept names before they reach the conductor, and concept names in a plan become usernames before they reach Mineflayer.
- **Positions** are absolute world coordinates: blocks as `[x, y, z]` ints, entities as `[x, y, z]` floats (feet). The conductor turns them into the world mirror's local coordinates (7.2).
- **Angles:** `yaw` and `pitch` are degrees in Minecraft's convention, the one the F3 group reads (DESIGN.md 6.2): yaw 0 faces south, 90 west, 180 north, −90 east; pitch −90 looks straight up, +90 straight down [mem]. Node converts with the expressions Mineflayer itself uses when it sends the look to the server, `Math.fround(conv.toNotchianYaw(yaw))` = float32(degrees(π − yaw)) and `Math.fround(conv.toNotchianPitch(pitch))` = float32(−degrees(pitch)) *(checked: `lib/conversions.js`, `lib/plugins/physics.js`)*, with `conv = require('mineflayer/lib/conversions')`. Mineflayer's own yaw is 0 at north and grows counterclockwise, and its pitch is positive upward *(checked)*, so raw values would turn the agent around.
- **Entity references** (`EntityRef` below): `{id, type, player?, username?, item?}`. `id` is the protocol entity id; `type` the vanilla entity type (`player`, `zombie`, `item`); for a player, `player` is its concept name and `username` its in-game name; `item` is `{name, count}` for a dropped item. A player whose username is not in the `players` table has `player` nil: on a firewalled offline server that is an infrastructure fault, which the conductor logs (DESIGN.md 11.6).
- **Inventory slots** use Mineflayer's player-window numbering: 5–8 armor (head to feet), 9–35 main, 36–44 hotbar, 45 off hand [mem]. A slot is `[slot, name, count, damage]`, with `damage` nil for items without durability. The simulator's `Inventory` (inventory.py) numbers the same slots as the game's player inventory: hotbar 0–8 is window 36–44, 9–35 is 9–35, armor 36–39 (feet to head) is window 8–5, off hand 40 is window 45.

## 4. Handshake and versioning

### 4.1 `hello` from Node (first message on a connection)

| Field | Type | Meaning |
|---|---|---|
| `protocol` | int | `1` |
| `session` | str | echo of `HIVEMIND_SESSION`; stops a stray host from an earlier run from attaching |
| `host_id` | str | echo of `HIVEMIND_HOST_ID` |
| `game_version` | str | `1.21.1` |
| `node` | str | `process.version` |
| `packages` | map | every installed package: its path as the lock writes it (`node_modules/mineflayer`, `node_modules/@msgpack/msgpack`, `node_modules/prismarine-physics/node_modules/vec3`) → the `version` in that folder's `package.json`. Node visits every folder `node_modules/<name>` and `node_modules/@<scope>/<name>` at any depth under `bridge/` (not `.bin`, not `package.json` files inside a package's own subfolders). An `npm ci` tree gives exactly the lock's 99 entries *(checked)* |
| `lock_sha256` | str | SHA-256 of `bridge/package-lock.json` after turning CRLF into LF; for the record only (the check uses `packages`) |
| `mcdata` | map | `{version, sha256: {path: hash}}`: hashes of the minecraft-data files listed in `data/sources.json` (`data/pc/1.21.1/blocks.json`, ..., `data/pc/1.20.5/biomes.json`), read from `node_modules/minecraft-data` |

### 4.2 `hello` from the conductor (reply)

| Field | Type | Meaning |
|---|---|---|
| `protocol` | int | `1` |
| `session` | str | the session id |
| `game_version` | str | `1.21.1` |
| `rules_hash` | str | `build.hash` of `data/rules-1.21.1.json` |
| `concepts_hash` | str | `hash` of `data/concepts-1.21.1.json` |
| `vocab_hash`, `vocab_words_hash` | str | `Language.hash` and `Language.words_hash` of `data/vocab-1.21.1.json`: the whole file (provenance) and the part a checkpoint depends on (the words, concept rows, domains and templates; contracts 4.1) |
| `players` | map | concept name → in-game username for every player of the session: each agent (`red_0` → `red_0`) and `human` → the owner's username (section 3) |
| `git` | str | the conductor's git commit |
| `settings` | map | section 4.4 |
| `tables` | map | section 4.5 |

The conductor instead answers `bye{reason}` and closes when:
- the protocol number differs, the session differs, or `game_version` is not `1.21.1`;
- any `mcdata` hash differs from the pins in `data/sources.json`. Node reports block state ids, and Python maps them with tables built from the pinned files. In `bridge/package-lock.json` minecraft-data is 3.117.0, and all twelve pinned files are byte-identical to the pins *(checked 2026-10-19)*;
- `packages` differs from the `packages` section of `bridge/package-lock.json`, compared by path and version, not by file bytes: a missing package (other than those the lock marks `optional`), an extra one, or another version. The M2 duration, dig and lag fits hold only for the pinned stack (mineflayer, mineflayer-pathfinder, prismarine-physics, prismarine-block, ...), so a host with any other version must not play (bridge/README.md: "A new Mineflayer is a bridge change").

Committed text files keep LF on every OS (`bridge/.gitattributes`), and every hash of a text file in the bridge is taken after turning CRLF into LF, so a Windows checkout gives the same values.

Node writes all of these values, its own and the conductor's, into each bot's telemetry `session.json` (DESIGN.md 11.4); the conductor writes them into `runs/<id>/run.json`.

### 4.3 Versioning rule

`protocol` is one integer and there is no negotiation. Adding an optional field or a new event kind is compatible (receivers ignore what they do not know). Removing or renaming a field, or changing a meaning or a unit, bumps the number. The rules, concepts and vocabulary hashes version the content: Node holds no vocabulary, so it cannot drift from them, but the hashes in every session record show what the agent was speaking, and `vocab_words_hash` shows which checkpoints fit.

### 4.4 `settings` (one source of truth, on the Python side)

| Key | Value | Source |
|---|---|---|
| `perception` | `M` or `S` | the agents' network preset (DESIGN.md 6.4); it sets the two radii below |
| `state_every_ticks` | 5 | resource sightings are checked every 5 ticks (DESIGN.md 6.1 [est]) |
| `mirror_radius_chunks` | 4 in preset M, 2 in S | covers everything perception reads from anywhere in the bot's chunk; the rule is in 7.2 |
| `entity_radius` | 67 in preset M, 35 in S | blocks from the bot's eye to an entity's position; why in 7.1 |
| `interrupt_hostile_radius` | 8 | DESIGN.md 4.5 |
| `dig_verify_extra_ticks` | 2 | re-read a dug block after RT + 2 ticks (DESIGN.md 5) |
| `chat_min_interval_ticks` | 40 | DESIGN.md 9 |
| `pathfinder` | the map and table in 5.1 | DESIGN.md 4.5, 11.1, built from `hivemind/sim/ref/nav.py` |

### 4.5 `tables`

Generated by the conductor from `hivemind.lang` and the concept tables at start-up, so that Node can render result words and detect hostiles without its own copy of the vocabulary:

| Key | Content |
|---|---|
| `hostile` | vanilla entity types whose entity class has kind `hostile` |
| `result_words` | failure code → words, with `{cause}`, `{station}` and `{container}` placeholders, from `Language.result_words` (DESIGN.md 5): `OK` → `done`, `NO_STATION` → `no {station} nearby`, `INV_FULL` → `{container} full`, `INTERRUPTED` → `interrupted by {cause}`, ... |
| `phrases` | `{entity: {vanilla type: phrase}, station: {crafting_table: "crafting table", furnace: "furnace", bed: "bed"}, container: {inventory: "inventory", chest: "chest", furnace: "furnace"}, cause: {damage: "damage", message: "message"}}` for filling the placeholders |

## 5. Bots

| Message | Direction | Fields |
|---|---|---|
| `join` | conductor → Node | `agent`, `username` (the agent's concept name, section 3), `server: {host, port}`, `area: {x0, z0, x1, z1}` (the bot's evaluation area, absolute block bounds, half-open; DESIGN.md 11.3, bridge/server/README.md section 8), `lag: {profile, seed, ...}` (the per-bot lag shim, DESIGN.md 8, 11.2), `respawn_delay_ticks` (nil: no respawn; episodes end on death in M1–M5), `team` (nil, `red` or `blue`), `telemetry_dir` |
| `joined` | Node → conductor | `agent`, `entity_id`, `pos`, `yaw`, `pitch`, `age`, `day_time`, `tick` (0), `hashed_seed`, `biomes`, `view_distance`, `simulation_distance` |
| `leave` | conductor → Node | `agent` |
| `left` | Node → conductor | `agent`, `reason` |

In `joined`:
- `hashed_seed` is the login packet's `worldState.hashedSeed` *(checked: 1.21.1 `SpawnInfo` in minecraft-data's protocol)* as a signed decimal string, `BigInt(packet.worldState.hashedSeed).toString()` (protodef decodes an i64 correctly, *checked*). F3's biome zoom needs it (7.2). Respawn packets carry the same `SpawnInfo` [mem].
- `biomes` is an array: index = the server's network id of a biome, value = its vanilla name, from `bot.registry.biomes` after login. prismarine-registry fills it from the server's `registry_data` packet for `worldgen/biome`, in the order the server sent *(checked: `lib/pc/index.js`)*. Chunk sections carry those ids (7.2); the conductor maps names to biome classes with `Concepts.biome_of`, so nothing depends on the network order.
- `view_distance` and `simulation_distance` are the login packet's values; the conductor checks them against the frozen 10 (DESIGN.md 11.2) and refuses a bot that disagrees.

Bots are created with Mineflayer's automatic respawn off (`respawn: false`; *checked*: `lib/plugins/health.js` respawns only when that option is set), and Node respawns them itself after `respawn_delay_ticks`. Respawning is automatic, not a policy action (DESIGN.md 5). The conductor puts each bot on its scoreboard team over RCON (`bridge/server/teams-1.21.1.mcfunction`).

### 5.1 Host startup and bot setup

The order is part of the protocol: a host that gets it wrong mines at the wrong speed or paths differently from the simulator, with no other sign.

1. **Patch first.** Call `applyDigTimePatch(mineflayerRegistry())` from `node/patches.js` once per process, before any prismarine-block `Block` object exists: before the first `mineflayer.createBot`, before any `new Movements(bot)` (its constructor makes a Block for every block type, *checked*: `lib/movements.js`), before mineflayer-pathfinder or mineflayer-tool is loaded into a bot (the pathfinder's inject makes a `Movements`, *checked*: `index.js`), and before any `require('prismarine-block')(registry)`. A Block copies its material when it is made, so a Block made earlier keeps the unpatched dig time (node/README.md). Then connect to the conductor and exchange `hello`.
2. **Per `join`:** `mineflayer.createBot({host: '127.0.0.1', port, username, version: '1.21.1', auth: 'offline', respawn: false})`, then at once `bot.once('inject_allowed', () => assertDigTimePatched(bot.registry))`: a bot whose registry is not patched must end the host before it loads a chunk. Load mineflayer-pathfinder and mineflayer-tool; mineflayer-armor-manager and mineflayer-auto-eat stay unloaded, because equipping and eating are policy actions (DESIGN.md 11.1).
3. **Pathfinder settings,** after `spawn`, from `settings.pathfinder` (built by the conductor from `nav.py`, the simulator's port of the package, so both sides find the same paths; `tests/test_nav.py` runs the installed package with exactly these settings and lists against nav.py):

| Object | Setting | Value | Package default *(checked)* |
|---|---|---|---|
| `bot.pathfinder` | `tickTimeout` | 10 (ms of A* per physics tick, DESIGN.md 11.1) | 40 |
| | `thinkTimeout` | 2000 (over 2 s gives `PATH_TIMEOUT`) | 5000 |
| | `searchRadius` | 96 (a cost bound in `astar.js`, nav.py) | −1 (unbounded) |
| `Movements` | `allowParkour` | false | true |
| | `canOpenDoors` | false | false |
| | `dontCreateFlow` | true | true |
| | `maxDropDown` | 4 | 4 |
| | `allow1by1towers` | true | true |
| | `allowEntityDetection` | false (entities in the way are ignored, as in the simulator) | true |
| | `canDig` | true | true |
| | `scafoldingBlocks` (the package's spelling) | the item ids of `nav.SCAFFOLDING_ITEMS`: `dirt`, `cobblestone` | the same two |
| | `blocksCantBreak` | the package's set plus `nav.bridge_block_lists(concepts)["blocksCantBreak"]`: every member of the never-dug classes (stations, beds, chests, doors, fence gates; 70 vanilla names as built) | `chest` and every block minecraft-data marks not diggable |
| | `blocksToAvoid` | the package's set plus `nav.bridge_block_lists(concepts)["blocksToAvoid"]`: every member of the hazard classes (`cactus`, `campfire`, `cobweb`, `fire`, `lava`, `magma_block`, `powder_snow`, `soul_campfire`, `soul_fire`, `sweet_berry_bush`, `wither_rose` as built) | `fire`, `cobweb`, `lava` |

   `settings.pathfinder` is `{search: {tickTimeout, thinkTimeout, searchRadius}, movements: {allowParkour, canOpenDoors, dontCreateFlow, maxDropDown, allow1by1towers, allowEntityDetection, canDig}, scafoldingBlocks, blocksCantBreak, blocksToAvoid}`, the last three as lists of vanilla names. Node assigns `search` to `bot.pathfinder` and `movements` to a `new Movements(bot)`, maps the names with `bot.registry.itemsByName` and `blocksByName`, replaces `scafoldingBlocks` with the mapped ids, and **adds** the two block lists to the Sets the constructor made (the package's own entries stay). Every other setting keeps the package default, which is what `nav.Settings` models (`digCost`, `placeCost`, `liquidCost` 1; `dontMineUnderFallingBlock` and `infiniteLiquidDropdownDistance` true). Then `bot.pathfinder.setMovements(movements)`. `allowSprinting` is left at the package default, true, for now (section 13).
4. **The flee goal** is `nav.GoalFlee` ported exactly: a goal at least `distance` blocks from a fixed point, measured horizontally from the centre of the node's cell. With `goals` from mineflayer-pathfinder:

   ```js
   class GoalFlee extends goals.Goal { // nav.GoalFlee (hivemind/sim/ref/nav.py)
     constructor (x, z, distance) { super(); this.x = x; this.z = z; this.distance = distance }
     isEnd (node) { return Math.hypot(node.x + 0.5 - this.x, node.z + 0.5 - this.z) >= this.distance }
     heuristic (node) { return Math.max(0, this.distance - Math.hypot(node.x + 0.5 - this.x, node.z + 0.5 - this.z)) }
   }
   ```

   Pathfinder nodes have integer `x`, `y`, `z`: the floored feet cell *(checked: `lib/move.js`)*, as nav.py's goals take them. The package's `GoalInvert` is not used: its end test and heuristic are not the simulator's.
5. **First state.** After `spawn`, Node sends `joined` and then a `state` with every mirrored column (7.2). The conductor builds no observation, and sends no `act`, until the mirror holds every column within `mirror_radius_chunks` of the bot's chunk that intersects the area.

## 6. Actions

### 6.1 `act` (conductor → Node)

| Field | Type | Meaning |
|---|---|---|
| `agent` | str | |
| `req` | int | request id, strictly increasing per agent |
| `command` | str | the English command, e.g. `mine 4 iron ore.`; for logs only |
| `parsed` | map | `Command.to_json()` of the `hivemind.lang` command (contracts 4.4): `{verb, args}`; for logs and to pick the wrapper |
| `plan` | map | concrete parameters resolved by the conductor (6.2): what Node executes |
| `deadline_ticks` | int | server ticks (by `age`) from receipt |
| `deadline_ms` | int | wall-clock ms from receipt |
| `sent_ms` | float | when the conductor sent it |

`parsed` is exactly `Command.to_json()` (`hivemind/lang/types.py`), and `Command.from_json` reads it back. `args` maps slot names to values, with exactly the slot names of contracts 4.5 (`block`, `item`, `entity`, `player`, `mine_count`, `item_count`, `addressee`, `furnace`, `dir8`, ...), plus `template` (the template id) for the verbs with several templates: `goto`, `attack`, `flee` and `equip`. Values are concept names (str) or ints, or maps for the structured values: an entity as `{ordinal, entity}`, or `{ordinal, dropped}` for an item on the ground; `chat`'s `sentence` as `{intent, <its slots>}`; `plan`'s `goal` as `{kind, <its slots>}`. Players are concept names. As the parser gives them *(checked)*:

```
mine 4 iron ore.                     {verb: "mine",   args: {block: "iron_ore", mine_count: 4}}
craft 4 planks.                      {verb: "craft",  args: {item: "planks", item_count: 4}}
attack the second zombie.            {verb: "attack", args: {entity: {ordinal: 2, entity: "zombie"}, template: "entity"}}
go to the nearest dropped bread.     {verb: "goto",   args: {entity: {ordinal: 1, dropped: "bread"}, template: "entity"}}
go to red 7.                         {verb: "goto",   args: {player: "red_7", template: "player"}}
go home.                             {verb: "goto",   args: {template: "home"}}
flee from blue 2.                    {verb: "flee",   args: {player: "blue_2", template: "player"}}
hold torch in off hand.              {verb: "equip",  args: {item: "torch", template: "off_hand"}}
give 3 bread to red 7.               {verb: "give",   args: {item: "bread", item_count: 3, player: "red_7"}}
tell everyone i need 3 iron ingot.   {verb: "chat",   args: {addressee: "everyone", sentence: {intent: "need", item: "iron_ingot", item_count: 3}}}
tell red 3 where is iron ore.        {verb: "chat",   args: {addressee: "red_3", sentence: {intent: "where", block: "iron_ore"}}}
plan to get 3 iron ingot.            {verb: "plan",   args: {goal: {kind: "get", item: "iron_ingot", item_count: 3}}}
```

Node uses `parsed.verb` to choose the wrapper and reads everything else from `plan`. It does not interpret `args`: those values are class names (`iron_ore`, `log`), and turning a class into something to act on needs the notebook. A class-level `mine 4 log.` becomes a list of known log positions; `craft 4 planks.` becomes vanilla recipes built from the items the bot holds. This keeps the honesty rule (targets come only from what the agent perceived) on the Python side.

**The resolver.** The plan comes from **one resolver** in `hivemind/`: `hivemind/sim/ref/resolve.py` (contracts 5, owned with `actions.py`), which turns the parsed command, the notebook and the inventory (and the recipe book) into a plan: which known positions in which order, which frontier point or staircase direction, which vanilla recipe and item variants from the items held. The conductor and the simulator's macros both call it, so real-versus-simulator replays from the same snapshot (DESIGN.md 11.4) pick the same targets and recipes. It reads only what the agent perceived, did or was told. Contracts 7 and DESIGN.md 11.1 carry the `plan` field. The resolver is not built yet (section 13).

### 6.2 `plan` per verb (provisional, fixed with the wrappers in M2)

Conventions:
- Positions are absolute (section 3). Goals are those `nav.py` builds, so both sides walk to the same place: `nav.near_block` is `GoalNear` with range 4.5 around a block, `nav.near_entity` `GoalNear` with range 3.0 around an entity's floored feet cell (reaches [mem], DESIGN.md 4.5).
- **Items.** `item` is one vanilla item name where the game takes exactly one (a furnace slot, a recipe slot). `items` is the list of vanilla names that count as the command's item class (its members in the concepts file), so Node needs no class table. Node takes stacks of any of them in Mineflayer's search order, window slots 9–35 and then the hotbar 36–44 (`Window.findInventoryItem`, *checked*: prismarine-windows `lib/Window.js`), which is the order of the simulator's `Inventory.remove` (inventory.py); a count is the total over the names. `equip` and the verbs that first put an item in the hand (`place`, `eat`) follow `Inventory.hold` instead (below).
- Counts are "up to" (contracts 4.3): Node does as many units as it can, up to the count, and reports how many in `units`.

| Verb | `plan` fields | Mineflayer side |
|---|---|---|
| `goto` | `pos` and `range`, or `entity` (id) and `range`, or `player` (concept name) and `range` | pathfinder `GoalNear` as above; for an entity or player, at its feet cell when the action starts (a `retarget` may move it, 6.3) |
| `explore` | `xz`: the frontier point | pathfinder `GoalXZ` |
| `descend` | `y`: target level; `dir`: staircase direction | own loop: `dig` + move |
| `strip_mine` | `dir`, `length` | own loop: `dig` + move |
| `mine` | `targets`: block positions in the conductor's order, `count` | pathfinder + `dig`, best tool chosen automatically (`mineflayer-tool`), digs verified (DESIGN.md 5) |
| `craft` | `steps`: recipes built from the items held (below) | a prismarine-recipe `Recipe` per step, `bot.craft` |
| `smelt_start` | `furnace` (position), `input` (one vanilla item: a furnace slot holds one), `count`, `fuel` (one vanilla item, or nil when the furnace already has the heat), `fuel_count` | `openFurnace`, `putFuel`, `putInput` |
| `collect` | `furnace` (position), `fuel_remainders`: `["bucket"]` (the values of `crafting.FUEL_REMAINDER`) | `openFurnace`, `takeOutput`, then `takeFuel` for an empty bucket (below) |
| `place` | `items`, `cells`: positions in order | hold (below), `placeBlock` |
| `attack` | `entity` (id; a player is resolved to its entity id by the conductor) | own loop on `attack`, at the cooldown |
| `flee` | `from`: `[x, z]`, the threat's position when the action starts; `distance` 16 (`nav.FLEE_DISTANCE`) | pathfinder with `GoalFlee` (5.1) |
| `eat` | `items` | hold (below), `consume` |
| `equip` | `items`, `dest`: `hand`, `off-hand`, `head`, `torso`, `legs`, `feet` | below |
| `sleep` | `bed` (position) | `sleep` |
| `give` | `player` (concept name), `items`, `count` | goto to ≤ 3 blocks + `toss` |
| `chat` | `text`: the rendered line, `Language.chat_text(addressee, sentence)`, e.g. `everyone, i need 3 iron ingot.` (DESIGN.md 9); `channel`: `global`, `team` or `whisper`; `to`: concept name or nil | `chat`, or `/teammsg` and `/msg <username>` [mem] |
| `wait` | `ticks`: 20, 100 or 400 | no-op until the time is up or an interrupt |
| `store` | `chest` (position), `items`, `count` | `openContainer`, `deposit` |
| `take` | `chest`, `items`, `count` | `openContainer`, `withdraw` |
| `plan` | none | Node waits one client tick and answers `OK`; the conductor then writes `parsed.args.goal` into the agent's own goal slot (DESIGN.md 6.6). The verb `plan` has nothing to do with the `act` field `plan` |

**`craft`: a recipe built from the items held.** `steps` is a list of `{result: {name, count}, in_shape, ingredients, passes, table}`: `in_shape` is rows of vanilla item names or nil (a shaped recipe, top row first), `ingredients` a list of vanilla names (a shapeless one); exactly one of the two is given. `result.count` is what one pass makes, `passes` how many passes, `table` the crafting table's position or nil for the 2 × 2 grid. The resolver fills every slot with an item the bot holds, mixing members of one class (two oak and two birch planks for a crafting table) and the alternatives a slot accepts (a tag ingredient takes any of its members, and some slots list alternatives, such as coal or charcoal for a torch); when the slots must change between passes (three oak and one birch planks make two sticks pass by pass), it sends one step per distinct filling. For each step Node maps the names with `bot.registry.itemsByName`, builds `new Recipe({inShape: [[id or null, ...], ...], result: {id, count}})` (or `{ingredients: [id, ...], result}`) with the `Recipe` class of the prismarine-recipe copy Mineflayer loads (`require(patches.packageDir('prismarine-recipe', patches.packageDir('mineflayer', __dirname)))(bot.registry).Recipe`), and calls `bot.craft(recipe, passes, tableBlock)`. Mineflayer's craft finds each slot's item with `findInventoryItem` and clicks it into place *(checked: `lib/plugins/craft.js`)*, so a mixed shape works. `bot.recipesFor` is not used: minecraft-data's `recipes.json` lists one recipe per member, never a mix (eleven crafting-table recipes, each with four planks of one kind, *checked*), so it reports nothing for a mix the game accepts. `units` is the items crafted. A failed pass ends the action with `MISSING_INGREDIENT` (the inventory changed) or `INV_FULL`.

**`collect`: output, then the bucket.** Node opens the furnace and takes the output with `takeOutput`; then, if the fuel slot holds an item named in `fuel_remainders` (the empty bucket a lava bucket leaves, crafting.py [mem]), it takes that too with `takeFuel`. As in `RecipeBook.collect`: no output gives `NO_TARGET`; if the inventory fills before the output is all taken, or the bucket does not fit, the result is `INV_FULL` with container `inventory`; `units` counts output items only. `raw_state.container` shows the furnace's slots as the window left them.

**`equip` and the hand: `Inventory.hold`'s rule.** For `dest` `hand` (and before `place` or `eat`): if the selected hotbar slot (window 36 + `bot.quickBarSlot`) holds one of `items`, nothing changes; else the first hotbar slot 36–44 holding one is selected (`bot.setQuickBarSlot`); else the first stack in window 9–35 (Mineflayer's search order) is moved to the first empty hotbar slot, which is then selected; else it is swapped with the selected slot's stack (`bot.moveSlotItem`, which swaps). Node must not call `bot.equip(item, 'hand')` for this: Mineflayer takes the first stack in 9–35 even when the hotbar holds one, and with a full hotbar it rotates through the slots (`nextQuickBarSlot`) *(checked: `lib/plugins/simple_inventory.js`)*. For `off-hand` (nothing when the off hand already holds one of `items`) and the armor slots, Node finds the first stack of `items` in window 9–44 and calls `bot.equip(stack, dest)`, which swaps it with what is there (`bot.moveSlotItem`, *checked*), as `Inventory.hold_offhand` and `Inventory.wear` do.

### 6.3 `cancel` and `retarget` (conductor → Node)

| Message | Fields | Effect |
|---|---|---|
| `cancel` | `agent`, `req`, `cause`: `message`, `damage`, `{hostile: <type>}` or `episode` | Node stops the action with full cleanup (6.6) and sends its result with code `INTERRUPTED` and that cause. A `cancel` for a request that has already finished is ignored: its result is already on the way |
| `retarget` | `agent`, `req`, `plan`: only the fields to replace | Updates a running action, e.g. more `targets` for `mine` when the notebook sees more of the class mid-action (the simulator's macro also uses what it perceives along the way), or a new `from` for `flee` |

`episode` is for resets; the conductor discards that transition.

### 6.4 Interrupts: who detects what

A running action is interrupted by damage, a new hostile within 8 blocks, a message to the agent from a teammate or the human, a vanished target or a full inventory (DESIGN.md 4.5). Each is detected on the side that can do so without parsing English or holding the vocabulary:

| Trigger | Detected by | Result |
|---|---|---|
| health decreases | Node (`update_health`) | `INTERRUPTED`, cause `damage` |
| a hostile (`tables.hostile`) inside `area` enters `interrupt_hostile_radius` during the action | Node | `INTERRUPTED`, cause `{hostile: <type>}` |
| chat addressed to the agent by a teammate or the human | conductor (needs the human tokenizer and the relation) | `cancel` with cause `message` |
| target entity gone, target block changed | Node | `TARGET_LOST` or `NO_TARGET` |
| no room for the next item | Node | `INV_FULL`, container `inventory` |
| a deadline passes | Node | `TIMEOUT` |
| the bot dies | Node | `DIED` |

Hostiles outside the area are ignored [design]: the simulator has nothing beyond its barrier edge, and the client is told about mobs behind the area's wall. Messages from enemies do not interrupt; they only reach the chat field (DESIGN.md 4.5).

### 6.5 `result` (Node → conductor)

Exactly one per `act`, whatever happens.

| Field | Type | Meaning |
|---|---|---|
| `agent`, `req` | | as in `act` |
| `code` | str | failure code (DESIGN.md 5): `OK`, `NO_TARGET`, `NO_PATH`, `PATH_TIMEOUT`, `STUCK`, `DIG_FAILED`, `MISSING_INGREDIENT`, `NO_STATION`, `INV_FULL`, `TARGET_LOST`, `HAZARD`, `INTERRUPTED`, `DIED`, `TIMEOUT` |
| `cause` | str, map or nil | for `INTERRUPTED`: `damage`, `message` or `{hostile: <vanilla type>}`; the conductor maps the type to its entity class for `Language.result_words` |
| `station` | str or nil | for `NO_STATION`: `crafting_table`, `furnace` or `bed` |
| `container` | str or nil | for `INV_FULL`: `inventory` (a full inventory stops `mine`, `strip_mine`, `craft`, `collect` or `take` partway), `chest` (`store` finds the chest full) or `furnace` (the furnace's own slots block `smelt_start`) (DESIGN.md 5) |
| `result` | str | the result words, e.g. `no path`, `interrupted by zombie`, `inventory full`, rendered from `tables`. The conductor re-renders them from `code`, `cause`, `station` and `container` with `Language.result_words` and logs a protocol error on any mismatch |
| `units` | int | units done: blocks mined or placed, items crafted, stored, taken, given, collected |
| `tick0`, `tick1` | int | client ticks at start and end |
| `age0`, `age1` | int | server age estimates at start and end |
| `wall_ms` | float | wall-clock duration |
| `n_roundtrips` | int | server confirmations waited for (window clicks, dig checks, ...) |
| `inv_delta` | map | vanilla item → signed count change, end minus start |
| `path` | map or nil | `{length, replans, search_ms}` for path-based actions (DESIGN.md 11.4) |
| `corrections` | int | server position corrections during the action |
| `aborted` | str or nil | `kick` or `host_restart` when infrastructure ended the action (6.6) |
| `raw_state` | map | the state at the end (7.1), plus `container` after `store`, `take`, `smelt_start` and `collect`: `{pos, kind: chest or furnace, slots}` exactly as the window showed it |

**Ordering.** Every event caused during an action is sent before its result: the `hurt` event before `INTERRUPTED` by damage, the `death` event before `DIED`. Right before a `result`, Node also sends a `state` with every mirror update gathered so far (7.2), so the mirror is current when the conductor builds the observation for the agent's next decision.

### 6.6 Deadlines, cleanup and faults

- Both deadlines count from the receipt of `act`; whichever passes first ends the action with `TIMEOUT` (`too slow`). `deadline_ticks` is in server ticks, so slow-server profiles (`/tick rate 15`) do not cut digs short; `deadline_ms` catches hangs. The conductor sets them from the simulator's duration model with a margin [est], e.g. `deadline_ms ≥ deadline_ticks × 50 × 20/TPS + 2000` [est].
- **Cleanup** on any early end (deadline, cancel, death): `stopDigging`, stop the pathfinder (`setGoal(null)`), close any open window, clear the control states, stop using an item. Window clicks in 1.21.1 have no transaction packets and wait with no timeout [75], so the deadline is their only exit (DESIGN.md 5).
- **Kicks** are events, not episode ends (DESIGN.md 15, lesson 10). The running action ends with `TIMEOUT` and `aborted: kick`; the conductor counts it as an infrastructure fault, outside the success statistics, and decides whether to `join` the bot again. **Provisional:** whether the agent should see `too slow` here is open until M2 measures kick rates.

## 7. State stream (Node → conductor)

### 7.1 `state`, every `state_every_ticks` client ticks per bot (and before every `result`)

| Field | Type | Meaning |
|---|---|---|
| `agent`, `tick`, `age` | | section 3 |
| `day_time` | int | the full day-time counter (`/time query daytime` [mem]), never taken modulo 24,000, so the day number floor(day_time / 24,000) can be computed (DESIGN.md 6.2). Estimated like `age`: the `time` of the last `update_time` packet plus the client ticks since it arrived, at most 20 ahead, while the daylight cycle runs. When the gamerule `doDaylightCycle` is false the server sends the time negated [mem]; `day_time` is then its absolute value, not advanced |
| `time_tick` | int | the client `tick` at which that `update_time` arrived, so the conductor knows how old `age` and `day_time` are (after a reset's `/time set`, it waits for a newer packet) |
| `pos`, `vel`, `yaw`, `pitch`, `on_ground` | | own movement: feet position; `yaw` and `pitch` in Minecraft degrees (section 3) |
| `health`, `food`, `saturation`, `oxygen`, `xp: {level, progress}` | | own status |
| `held_slot` | int | hotbar slot 0–8 |
| `ping` | int | ms, as Mineflayer reports it [6] |
| `weather` | map | `{rain, thunder}` as the client knows it |
| `entities` | array | every entity whose position is within `entity_radius` of the bot's eye (feet + 1.62 [mem]), visible or not, the bot itself excluded: `EntityRef` plus `pos`, `vel`, `yaw`, `health` (nil if unknown), `held` and `armor` (vanilla items or nil), `team` (scoreboard team for players, nil if none) |
| `inventory` | array or nil | all non-empty slots (section 3), sent only when the inventory changed since the last `state` |
| `chunks`, `unload`, `changes`, `light` | | world mirror (7.2) |

Node reads `age` and `time` from the raw `update_time` packet, as `Number(BigInt(packet.age))` and `Number(BigInt(packet.time))` (protodef decodes an i64 correctly, *checked*), not from `bot.time`: mineflayer 4.39.0's `longToBigInt` (`lib/plugins/time.js`) makes any value whose low 32 bits have the top bit set negative (2^31 + 5 becomes −2^31 + 5, *checked*), and `bot.time` is not advanced between packets.

**Why these entity radii** [design]. The panorama counts any entity box a ray enters within 64 blocks (DESIGN.md 6.1), and the largest box of any entity class reaches 2.97 blocks from its position (the feet point; a warden's 0.9 × 2.9 box in the concepts file), so preset M needs every entity within 64 + 3 = 67 of the eye. Preset S has no panorama: the entity table lists visible entities whose middle is within 32 blocks of the eye (`entity_visible`), and the F3 group's targeted entity lies within 3.0, so 32 plus half the tallest box, 35, is enough. The client is told about entities within their tracking ranges at the frozen `entity-broadcast-range-percentage=100`: 64 blocks horizontally for arrows, more for players, mobs and dropped items [mem], so the panorama's 64 blocks are covered. M2 checks it with a mob at 64 blocks.

`raw_state` in a result has the same fields, without the world-mirror fields and with `inventory` always present.

### 7.2 World mirror

Python runs the shared perception (DESIGN.md 6.1, 6.2: the visible-only window, the panorama in preset M, the F3 group) and the notebook on a mirror of what the bot's client holds, so Node never decides what is visible. The conductor keeps one mirror per bot: a `World` (`hivemind/sim/ref/world.py`) the size of the bot's evaluation area (128 × 128, DESIGN.md 11.3), in local coordinates (local x = x − `area.x0`, local z = z − `area.z0`; y unchanged), and calls the same functions on it as the simulator does (`perception.perceive`, `debug_info`).

**Clipped to the evaluation area.** Every cell outside the area (and below y −64) reads as `barrier`, and every cell above y 191 inside it as air: exactly how `World.get` reads the simulator's world edge (DESIGN.md 4.4, 11.3). The real barrier walls, the terrain beyond them and anything above 191 never reach perception, and perception treats the barrier as not see-through, so both sides see the same edge. Node sends only columns that intersect the area, and only their sections −4..11 (y −64..191; start areas keep their terrain at or below 191, DESIGN.md 4.4). A block placed above 191 in the real game is not mirrored.

**Radius.** Node mirrors the columns within `mirror_radius_chunks` (Chebyshev distance, in chunks) of the bot's chunk, and updates the set when the bot changes chunk. The rule: **r = ceil(R / 16)**, where R is the farthest horizontal distance from the eye at which perception reads a cell. The eye can be anywhere in its chunk c, x ∈ [16c, 16c + 16), so every cell within R of it lies in [16c − R, 16c + 16 + R), inside chunks c − ceil(R/16) … c + ceil(R/16); the same holds for z. The eye's chunk is the feet's chunk.
- **Preset M:** R = 64, the panorama's range. perception.py walks each ray cell by cell and enters no cell beyond t = 64 blocks; its pixel rays are never parallel to x or z (column centres sit at odd multiples of 1.875°), so a ray moves less than 64 along either axis. r = 4: a 9 × 9 square of columns, the bot's chunk ± 64 blocks. Draft 0.4's 2 chunks covered only its 24-block perception.
- **Preset S:** R = 32, the line-of-sight segments of `entity_visible` to entities within 32 blocks; the window needs only ±8 (±7 and its ring of neighbours) and the look ray 4.5. r = 2.
- Light is read from the mirror, never recomputed (below), so no light margin is added. View distance 10 (DESIGN.md 11.2) keeps all of it loaded on the client.
- perception.py fetches slightly larger boxes than it uses (the panorama's `World.region` spans ±66 cells around the eye's cell, `line_of_sight` one cell beyond each segment), so area cells outside the mirrored square must still hold some block class: the last mirrored value, or air where nothing was ever mirrored. They never change an output.

**Fields** in `state`:
- `chunks`: columns that entered the radius, or that the server sent again: `{x, z, sections}` with x, z in chunks. Each section is `{y, ...}` (section index −4..11) with:
  - block states: `single: <state id>` when the section is uniform, otherwise `zids`: 4096 little-endian uint16 state ids, index (y × 16 + z) × 16 + x within the section, deflated with zlib;
  - biomes: the section's 4 × 4 × 4 biome cells (the game stores one biome per 4 × 4 × 4 cell, DESIGN.md 4.4): `biome_single: <id>` when uniform, otherwise `biomes`: 64 bytes, index (y4 × 4 + z4) × 4 + x4, read with `column.getBiome(pos)` (column-local x and z) at each cell's lowest corner. The ids are the server's network ids, named by `joined.biomes`;
  - light: `zsky` and `zblock`, each the section's 2,048 light bytes exactly as the server sent them, deflated, in the game's layout: cell (x, y, z) at index (y × 16 + z) × 16 + x, two cells per byte, low nibble first [mem]. A section whose light arrived as all zero (the packet's empty mask) is sent as 2,048 zero bytes; a section for which the client holds no light data at all (no bit in either mask yet) has no field.
- `unload`: `[[x, z], ...]` for columns that left the radius or that the server unloaded.
- `changes`: `[[x, y, z, state_id], ...]` for block updates in mirrored columns, y −64..191, since the previous `state`, in arrival order.
- `light`: `[{x, z, y, zsky?, zblock?}, ...]`: sections of mirrored columns whose light the server updated since the previous `state` (`update_light` packets, which Mineflayer applies to its column, *checked*: `lib/plugins/blocks.js`), each with its full new contents, applied after `changes` of the same message. Mineflayer does not relight the column itself after a block change, so the mirror's light is the light the server sends [mem: the server sends updated light for changed sections].
- State ids are minecraft-data 1.21.1 ids, which are the protocol ids (verified at `hello` through the file hashes, and against the vanilla report by `servertool.py check-generator`). Python maps them to block classes with `data/block_classes-1.21.1.csv`.

**Reading light without scrambling it.** Node must forward the light bytes as received: from the raw `map_chunk` and `update_light` packets (`skyLight[k]` is the 2,048 bytes of the k-th set bit of `skyLightMask`, a bit of `emptySkyLightMask` means all zero, and the same for block light; 1.21.1 packet layout *checked* in minecraft-data's protocol), or from prismarine-chunk's `column.dumpLight()`, which returns exactly the bytes it read *(checked)*. Light index i of a column covers section y = i − 5: the overworld's light arrays have one extra section below y −64 and one above *(checked: `getLightSectionIndex`)*. It must not use `column.getSkyLight`, `column.getBlockLight`, or `block.light` and `block.skyLight` from `bot.blockAt`: prismarine-chunk 1.41.0 reads network light as big-endian 64-bit words, so these getters return the light of another cell in the same row, x' = 14 − (x & ~1) + (x & 1) *(checked: `src/pc/1.18/ChunkColumn.js` and `src/pc/common/BitArray.js`)*, while the same class loads light from disk in the nibble layout above. Block states and biomes travel as palette indices packed into 64-bit longs [mem], the layout that reader expects, so `getBlockStateId` (which `bot.blockAt` relies on) and `getBiome` are not affected; M2 still compares a biome border with F3.

**What the conductor's mirror holds:**
- **Voxels:** the block class of each state id.
- **Light:** the game's sky and block light as received. It is not provisional: the window's light, the panorama's light fields and the F3 group's light at the feet read it (DESIGN.md 6.2). The simulator's own light is an approximation (DESIGN.md 4.2), so the mirror never recomputes light from its voxels. `World` computes its light from the voxels today, so the mirror needs a way to hold given sky and block light arrays (section 13).
- **Biome:** the F3 group reads one biome per column (`World.biome_at(x, z)` in `debug_info.py`), because test worlds assign one per column (DESIGN.md 4.4). On the mirror that value must be the biome F3 shows: `BiomeManager.getBiome` at the feet block, which picks one of eight 4 × 4 × 4 cells, q or q + 1 on each axis with q = (coordinate − 2) >> 2, by a jittered distance keyed by the hashed world seed [mem]; the chosen cell can lie in a neighbouring section or column, which the mirror holds. Rule [design]: before each F3 computation, the conductor writes into the mirror's `biome_map`, at the feet column, the zoomed biome of the feet block (the floored feet position), computed in absolute coordinates with `joined.hashed_seed` and mapped to its biome class (`Concepts.biome_of` of the vanilla name). The F3 group reads nothing else of `biome_map`, so this per-column value is exactly the game's for the one column it reads. The zoom must be one function in `hivemind/`, shared with the M3 importer, which keeps the 3-D cells (DESIGN.md 4.4: "checked against F3 in M3"); it does not exist yet (section 13).
- **Clock:** `World.day_time` = `state.day_time` (the full counter), `World.tick` = `state.age` − the age at the episode's start.
- **Entities:** the bot itself in an agent slot (its `pos`, `yaw`, `pitch` and home), and every entity of `state.entities` in the slot table of its kind: players as agents, mobs, dropped items. Boxes come from the entity classes' sizes (perception.py), never from Mineflayer.

### 7.3 Where each perceived input comes from

| Input (DESIGN.md 6.1, 6.2) | From |
|---|---|
| Voxel window: classes of seen cells, `unseen` (255) for the rest, light | mirror voxels and light, rays from the eye at `pos` + 1.62 |
| Panorama (preset M): block or entity hit, distance, face, sky and block light in front of the face | mirror voxels, light and entity boxes |
| Entity table: the 16 nearest visible within 32 | `entities`, filtered by `entity_visible` on the mirror |
| F3: x and z relative to home, y absolute | `pos`; home is the conductor's (the agent's spawn point; from M9 its team's base). Absolute x and z never enter the observation |
| F3: yaw, pitch, the 4-way facing word | `yaw`, `pitch` (Minecraft degrees); the word from `hivemind.lang.world_words.facing_word` |
| F3: targeted block, targeted fluid | the look ray (`debug_info.look_targets`) through the mirror's voxels |
| F3: targeted entity | the same ray against the mirror's entity boxes; never through or behind the targeted block or a lava cell (debug_info.py) |
| F3: tick of the day and its word, day number | `day_time` |
| F3: ticks since the episode began | `age` − the age at the episode's start |
| F3: biome at the feet | the mirror's biome cells, zoomed with `hashed_seed` (7.2) |
| F3: sky light and block light at the feet | the mirror's light |

Nothing in this table comes from the client's debug screen, from RCON or from the server's state.

## 8. Events (Node → conductor)

`event{agent, kind, tick, age, ...}`. The first five kinds are those of DESIGN.md 11.1; the others are what the notebook needs from the real game (section 9). Players are concept names (section 3).

| `kind` | Fields | Source |
|---|---|---|
| `chat` | `from` (concept name; nil with `username` set for a stranger), `uuid`, `text`, `chat_type` (registry name without namespace: `chat`, `msg_command_incoming`, `team_msg_command_incoming`, ...; the 1.21.1 names are the files in mcmeta `data/minecraft/chat_type` *(checked)*), `from_self` | player chat only, from the raw `playerChat` event of the client (`bot._client.on('playerChat', data)`): `uuid` = `data.sender`, `text` = `data.plainMessage`, `from` = the concept name of the username of that UUID in `bot.players`. `chat_type` = `bot.registry.chatFormattingById[i].name` with `i = data.type.chatType ?? data.type` (prismarine-registry 1.12.0 already applies 1.21's offset of 1). Mineflayer's own `message` event cannot be used for the type: it passes the constant `'chat'` for every player message *(checked: mineflayer `lib/plugins/chat.js`, prismarine-registry `lib/pc/index.js`)*. A `playerChat` without `sender` (the `profileless_chat` packet, e.g. `/say` over RCON) and system messages are not forwarded. The conductor reads `text` with the human tokenizer (DESIGN.md 9) |
| `hurt` | `victim` and `source` (`EntityRef`, `source` nil if none), `damage_type` (registry name, e.g. `player_attack`, `mob_attack`, `fall`, `lava`), `health` (own health after the hit, when the victim is the bot) | the `damage_event` packet, which names the source entity; Mineflayer emits `entityHurt(entity, source)` from it *(checked: `lib/plugins/entities.js`)*. Sent for the bot and for every tracked entity |
| `death` | `killer` (`EntityRef` or nil), `death_key` (translation key, e.g. `death.attack.player`), `args` (the key's arguments: a player as its concept name, a mob as its vanilla type from the translation key, e.g. `entity.minecraft.zombie` → `zombie`, anything else nil), `pos` | the death message's translation key and arguments: structured data, not English text. `death.attack.player` is `%1$s was slain by %2$s` in the pinned language file, so `args[1]` is the killer. With the gamerule `showDeathMessages` false the message is empty, so `servertool.py check-gamerules` keeps it at the vanilla default, true |
| `kick` | `reason` (text component as JSON), `logged_in` | Mineflayer `kicked(reason, loggedIn)` *(checked)* |
| `correction` | `from`, `to`, `dist` | Mineflayer `forcedMove`: a server position correction or teleport *(checked)* |
| `entity_dead` | `entity` (`EntityRef`), `pos` | entity status 3 [mem] |
| `pickup` | `collector` (`EntityRef`), `item: {name, count}`, `item_entity` (id) | the collect packet (Mineflayer `playerCollect`) *(checked)* |
| `lid` | `pos`, `block` (`chest` or `trapped_chest`), `viewers` (int: players that now have it open; no names) | Mineflayer `chestLidMove(block, viewers, block2)`, for a double chest once, at the right half. It fires for **any** chest block loaded in the bot's client whose open count changes, with no line-of-sight or distance test: Mineflayer checks only that `bot.blockAt(pos)` is loaded *(checked: `lib/plugins/block_actions.js`)*. The server sends block events to every player within 64 blocks of the chest [mem], through walls. So a `lid` event is heard, not seen; Node forwards it only for `chest` and `trapped_chest` (the same `block_action` packet also reaches it for other blocks) |
| `respawn` | `pos` | after Node respawns the bot |

## 9. What the notebook gets from the real game

The owner asked how an agent remembers where things were, who is an enemy and how much is in a chest (DESIGN.md 1, 6.3). In the simulator, the notebook is written from the simulator's perception and the agent's own actions. In the real game, the same `Notebook` code is fed from this protocol, and every input is something the bot's own client received:

| Notebook table | Fed by |
|---|---|
| Places (`iron ore`, `my furnace 2`, `my chest 1`, `my bed`, `lava`) | perception on the world mirror (7.2) every `state`: seen window cells and, in preset M, panorama pixels up to 64 blocks away (DESIGN.md 6.1); own stations from the positions in successful `place` results |
| Players (relation, last seen, last message, damage, kills, gifts, times seen opening my chests) | `state.entities` (`player`, position, `team`, from which the relation word follows, as a player sees name colours), filtered by the shared perception (below); `chat` events (last message words); `hurt` events (damage each way, from `source` and `victim`); `death` and `entity_dead` (kills each way); `pickup` events and `give` results (items given each way, attributed by the fixed rule of DESIGN.md 6.3 from the notebook's own perception history); `lid` events on the agent's own chests (times seen opening my chests, by the rule below) |
| Containers (contents as last seen) | `raw_state.container` after `store` and `take`, passed to `notebook.on_container_view`. A drop since the last look becomes `3 iron ingot missing from my chest 1`. No event ever names a thief: the game does not tell a client who took what (DESIGN.md 6.3) |
| Events (ring buffer of sentences) | written by the conductor from the above, in the vocabulary's event sentences (contracts 4.3) |

**Perception before attribution.** Node forwards what the client received, and the client receives more than the agent may use: `state.entities` holds every tracked entity within `entity_radius`, whether or not it is in line of sight, and `lid` events arrive through walls (section 8). The conductor therefore runs the same perception functions as the simulator (DESIGN.md 6.1) on the world mirror before the notebook uses an entity: a player counts as *visible* only if `entity_visible` says so at that tick. The `{player} opened my chest {chest}` rule of DESIGN.md 6.3 is then applied unchanged: one of the agent's own chests gets a `lid` event with more viewers while the agent is not using it, and the opener is named only if exactly one other visible player is within 6 blocks [est] of the chest; otherwise the event is `someone opened my chest {chest}`. The simulator produces `lid` events with the same range as the server (measured in M7 [est]), so both sides write the same sentences.

Nothing reaches the notebook from RCON, from server-side state or from Mineflayer searches over all loaded blocks.

## 10. Timing and batching

- Each agent is its own semi-MDP. A `result` makes its agent ready.
- **Batching:** the first ready result opens a window of `batch_window_ms` (5–10 ms, default 7 [est]; DESIGN.md 11.1, as in WAI's async actor). The window closes when it expires or when every agent is ready. The conductor then runs one batched forward pass and sends one `act` per ready agent. Node does no batching; it sends every message at once.
- **Liveness:** the conductor sends `ping{nonce, sent_ms}` every second; Node answers `pong{nonce, sent_ms, node_ms, loop_delay_ms}`, where `loop_delay_ms` is the event-loop delay for telemetry (M7 target: p99 < 20 ms). After three missed pongs the conductor restarts the host; its bots' open actions are recorded with `aborted: host_restart`.

## 11. Errors and shutdown

| Message | Fields | When |
|---|---|---|
| `error` | `agent`, `req`, `what`, `detail` | `busy` (an `act` while that agent's action runs), `unknown_agent`, `bad_plan`, `bad_message`. The `act` did not start; this is a bug to fix, not a game outcome, and the transition is not recorded |
| `bye` | `reason` | either side, before closing |

## 12. Example exchange

```
Node  → hello   {protocol: 1, session: "s-3f9c", host_id: "h0", game_version: "1.21.1", packages: {...}, mcdata: {...}}
Cond. → hello   {protocol: 1, session: "s-3f9c", rules_hash: "...", concepts_hash: "...", vocab_hash: "...",
                 vocab_words_hash: "...", players: {red_0: "red_0", human: "<owner's username>"}, settings: {...}, tables: {...}}
Cond. → join    {agent: "red_0", username: "red_0", server: {host: "127.0.0.1", port: 25565},
                 area: {x0: 0, z0: 0, x1: 128, z1: 128}, respawn_delay_ticks: null, ...}
Node  → joined  {agent: "red_0", entity_id: 812, pos: [12.5, 70.0, -3.5], yaw: 180.0, pitch: 0.0, age: 1200,
                 day_time: 1200, tick: 0, hashed_seed: "-4437018226040387823", biomes: ["badlands", ...], ...}
Node  → state   {agent: "red_0", tick: 5, day_time: 1205, chunks: [...], entities: [...], inventory: [...], ...}
Cond. → act     {agent: "red_0", req: 1, command: "mine 4 iron ore.",
                 parsed: {verb: "mine", args: {block: "iron_ore", mine_count: 4}},
                 plan: {targets: [[14, 52, -9], [15, 52, -9], [15, 53, -10]], count: 4}, deadline_ticks: 900, deadline_ms: 47000}
Node  → state   {agent: "red_0", tick: 10, changes: [[14, 52, -9, 0]], light: [...], ...}
Node  → event   {agent: "red_0", kind: "hurt", victim: {id: 812, type: "player", player: "red_0", username: "red_0"},
                 source: {id: 901, type: "zombie"}, damage_type: "mob_attack", health: 17.0}
Node  → state   {agent: "red_0", tick: 61, ...}
Node  → result  {agent: "red_0", req: 1, code: "INTERRUPTED", cause: "damage", result: "interrupted by damage", units: 1,
                 inv_delta: {raw_iron: 1}, tick0: 7, tick1: 61, wall_ms: 2712.4, ...}
```

## 13. Open points

- **`plan` fields per verb** (6.2) are fixed with the wrappers in M2.
- **The resolver** (`hivemind/sim/ref/resolve.py`, contracts 5) is specified but not built; until it exists the conductor cannot produce a `plan`.
- **The mirror's world.** `World` computes light from its voxels and has no way to take the game's light, so the mirror needs one (a light-holding variant or a loader that `light_points`, `region(light=True)` and `sky_light_region` then read); the conductor cannot serve real light to the shared perception until it exists.
- **The biome zoom** (7.2): one shared function in `hivemind/` for the M3 importer and the conductor, with test values checked against F3 in M3. Until it exists the bridge has no F3 biome, and no real-game evaluation may run.
- **Light without data** (7.2): prismarine-chunk reads a section with no light data as 0 *(checked)*; what the game's client shows there, which is the light F3 reports, is [mem]. M2 compares sky light high above the terrain and in a deep cave with F3's "Client Light", together with the nibble layout of 7.2 (one torch), and the conductor follows the game.
- **Sprinting.** mineflayer-pathfinder sprints whenever its path allows (`allowSprinting` true by default, *checked*: `index.js`), while nav.py's `GotoModel` uses the walking speed, 4.317 m/s [mem], and sprinting also costs the game more food [mem]. M2 decides: set `allowSprinting` false, or model sprinting in the simulator (DESIGN.md 4.5 has both speeds).
- **Block-event range** for `lid`: 64 blocks is [mem]; M7 measures it, and the simulator uses the measured value (DESIGN.md 6.3).
- **Kick results:** `TIMEOUT` with `aborted: kick` for now (6.6).
- **Mirror bandwidth:** in preset M a bot mirrors up to 81 columns of 16 sections, each with deflated block states and light; the bytes per bot per second at walking speed, and the cost of the first fill, are measured in M2 [est].

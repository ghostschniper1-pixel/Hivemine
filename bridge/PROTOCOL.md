# Conductor ↔ Node host protocol, version 1

The wire protocol between the Python conductor and the Node hosts that run the Mineflayer bots (DESIGN.md 11.1, contracts section 7).

**Status:** a specification for M2 (minimal bridge) and M7 (full bridge). Nothing implements it yet. Parts marked **provisional** are expected to change when the wrappers are written in M2; changes that only add optional fields keep protocol version 1 (section 4.3). Labels as in DESIGN.md: [n] a reference there, [mem] unverified memory, [est] an estimate. Facts marked *(checked)* were read from the pinned package sources (mineflayer 4.39.0, minecraft-protocol 1.68.0, mineflayer-pathfinder 2.4.5, @msgpack/msgpack 3.1.3) on 2026-10-16.

Three rules shape everything below:
1. **Node never parses English.** The conductor sends the command text (for logs), the parsed form from `hivemind.lang` and a concrete `plan`. There is one parser, on the Python side (DESIGN.md 5).
2. **Node never decides what the agent knows.** It forwards what the bot's own client received (blocks, entities, chat, damage, windows). Perception, the notebook, masks and every class or phrase live in Python, shared with the simulator (DESIGN.md 2, principle 6). Targets come from the notebook, never from `findBlocks` (no X-ray).
3. **Node speaks vanilla.** Blocks, items and entities are vanilla names without the `minecraft:` prefix (`deepslate_iron_ore`, `raw_iron`, `zombie`) and block state ids. Python maps them to classes and words with `data/concepts-1.21.1.json` and `data/block_classes-1.21.1.csv`.

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
- @msgpack/msgpack encodes every integer-valued JS number as an int, including `1.0`, and other numbers as float64 *(checked)*. **A receiver must accept an int wherever a float is expected.** A `Uint8Array` or `Buffer` encodes as bin and decodes as `Uint8Array` *(checked)*.
- Optional fields may be absent or nil; receivers treat both alike and ignore unknown fields. (An `undefined` value in a JS map encodes as nil *(checked)*.)
- The conductor's runtime code may use only the standard library and NumPy if it lives under `hivemind/` (contracts section 0). This subset is small enough for a stdlib codec; otherwise the `msgpack` package can be used outside `hivemind/`.

Example: `{"t": "ping", "nonce": 1}` is the frame `00 00 00 0f 82 a1 74 a4 70 69 6e 67 a5 6e 6f 6e 63 65 01` *(checked with @msgpack/msgpack 3.1.3)*.

## 3. Common fields and conventions

| Field | Type | Meaning |
|---|---|---|
| `t` | str | message type (`hello`, `act`, `result`, ...) |
| `seq` | int | per connection and direction, 1, 2, 3, ...; for logs and gap detection |
| `agent` | str | the agent's name as a phrase: `red 0`, `blue 7` |
| `tick` | int | the bot's client tick counter: Mineflayer's `physicsTick`, one per 50 ms of wall time *(checked: `PHYSICS_INTERVAL_MS = 50`)*, counted from the bot's spawn |
| `age` | int | the server's world age, estimated: the age in the last `update_time` packet plus the client ticks since it arrived, at most 20 ahead. The server sends `update_time` every 20 ticks [mem], so `age` is coarse; exact server-side ticks come from the harness mod's log (M2, DESIGN.md 11.3) |
| `wall_ms` / `*_ms` | float | milliseconds; absolute times are Unix epoch ms (both processes share one clock) |

- **Names.** An agent's in-game username is its phrase with `_` for spaces: `red 0` ↔ `red_0`, `blue 19` ↔ `blue_19`. Usernames must be 3–16 characters from `[A-Za-z0-9_]` [mem]. The human's username comes from the conductor's config and maps to the name `human`.
- **Positions:** blocks as `[x, y, z]` ints; entities as `[x, y, z]` floats (feet); `yaw` and `pitch` in radians as Mineflayer gives them.
- **Entity references** (`EntityRef` below): `{id, type, name?, item?}`. `id` is the protocol entity id; `type` the vanilla entity type (`player`, `zombie`, `item`); `name` the username for players; `item` `{name, count}` for dropped items.
- **Inventory slots** use Mineflayer's player-window numbering: 5–8 armor (head to feet), 9–35 main, 36–44 hotbar, 45 off hand [mem]. A slot is `[slot, name, count, damage]`, with `damage` nil for items without durability.

## 4. Handshake and versioning

### 4.1 `hello` from Node (first message on a connection)

| Field | Type | Meaning |
|---|---|---|
| `protocol` | int | `1` |
| `session` | str | echo of `HIVEMIND_SESSION`; stops a stray host from an earlier run from attaching |
| `host_id` | str | echo of `HIVEMIND_HOST_ID` |
| `game_version` | str | `1.21.1` |
| `node` | str | `process.version` |
| `packages` | map | installed versions: `mineflayer`, `mineflayer-pathfinder`, `@msgpack/msgpack`, `minecraft-data`, `minecraft-protocol`, ... |
| `lock_sha256` | str | SHA-256 of `bridge/package-lock.json` |
| `mcdata` | map | `{version, sha256: {path: hash}}`: hashes of the minecraft-data files listed in `data/sources.json` (`data/pc/1.21.1/blocks.json`, ...), read from `node_modules/minecraft-data` |

### 4.2 `hello` from the conductor (reply)

| Field | Type | Meaning |
|---|---|---|
| `protocol` | int | `1` |
| `session` | str | the session id |
| `game_version` | str | `1.21.1` |
| `rules_hash`, `concepts_hash`, `vocab_hash` | str | `build.hash` of `data/rules-1.21.1.json`, and the hashes of `data/concepts-1.21.1.json` and `data/vocab-1.21.1.json` |
| `git` | str | the conductor's git commit |
| `settings` | map | section 4.4 |
| `tables` | map | section 4.5 |

The conductor instead answers `bye{reason}` and closes when the protocol number differs, the session differs, `game_version` is not `1.21.1`, or any `mcdata` hash differs from the pins in `data/sources.json`. That last check matters: Node reports block state ids, and Python maps them with tables built from the pinned files. In `bridge/package-lock.json` minecraft-data is 3.117.0, and all eleven pinned files are byte-identical to the pins *(checked)*.

Node writes all of these values, its own and the conductor's, into each bot's telemetry `session.json` (DESIGN.md 11.4); the conductor writes them into `runs/<id>/run.json`.

### 4.3 Versioning rule

`protocol` is one integer and there is no negotiation. Adding an optional field or a new event kind is compatible (receivers ignore what they do not know). Removing or renaming a field, or changing a meaning or a unit, bumps the number. The rules, concepts and vocabulary hashes version the content: Node holds no vocabulary, so it cannot drift from them, but the hashes in every session record show what the agent was speaking.

### 4.4 `settings` (one source of truth, on the Python side)

| Key | Value | Source |
|---|---|---|
| `state_every_ticks` | 5 | resource sightings are checked every 5 ticks (DESIGN.md 6.1 [est]) |
| `mirror_radius_chunks` | 2 | covers 24-block perception from anywhere in the bot's chunk [est] |
| `entity_radius` | 32 | entities in the observation (DESIGN.md 6.2) |
| `interrupt_hostile_radius` | 8 | DESIGN.md 4.5 |
| `dig_verify_extra_ticks` | 2 | re-read a dug block after RT + 2 ticks (DESIGN.md 5) |
| `chat_min_interval_ticks` | 40 | DESIGN.md 9 |
| `pathfinder` | `tickTimeout` 10, `thinkTimeout` 2000, `searchRadius` 96, `allowParkour` false, `canOpenDoors` false, `maxDropDown` 4, `allow1by1towers` true, `canDig` true, `dontCreateFlow` true | DESIGN.md 4.5, 11.1. The package defaults differ: `tickTimeout` 40, `thinkTimeout` 5000, `searchRadius` −1, `allowParkour` true *(checked: mineflayer-pathfinder 2.4.5 `index.js`, `lib/movements.js`)* |

### 4.5 `tables`

Generated by the conductor from `hivemind.lang` and the concept tables at start-up, so that Node can render result words and detect hostiles without its own copy of the vocabulary:

| Key | Content |
|---|---|
| `hostile` | vanilla entity types whose entity class has kind `hostile` |
| `result_words` | failure code → words, with `{cause}` and `{station}` placeholders: `OK` → `done`, `NO_STATION` → `no {station} nearby`, `INTERRUPTED` → `interrupted by {cause}`, ... (DESIGN.md 5) |
| `phrases` | `{entity: {vanilla type: phrase}, station: {crafting_table: "crafting table", furnace: "furnace", bed: "bed"}}` for filling the placeholders |

## 5. Bots

| Message | Direction | Fields |
|---|---|---|
| `join` | conductor → Node | `agent`, `username`, `server: {host, port}`, `lag: {profile, seed, ...}` (the per-bot lag shim, DESIGN.md 8, 11.2), `respawn_delay_ticks` (nil: no respawn; episodes end on death in M1–M5), `team` (nil, `red` or `blue`), `telemetry_dir` |
| `joined` | Node → conductor | `agent`, `entity_id`, `pos`, `age`, `tick` (0) |
| `leave` | conductor → Node | `agent` |
| `left` | Node → conductor | `agent`, `reason` |

Bots are created with Mineflayer's automatic respawn off (`respawn: false`; *checked*: `lib/plugins/health.js` respawns only when that option is set), and Node respawns them itself after `respawn_delay_ticks`. Respawning is automatic, not a policy action (DESIGN.md 5). The conductor puts each bot on its scoreboard team over RCON (`bridge/server/teams-1.21.1.mcfunction`).

## 6. Actions

### 6.1 `act` (conductor → Node)

| Field | Type | Meaning |
|---|---|---|
| `agent` | str | |
| `req` | int | request id, strictly increasing per agent |
| `command` | str | the English command, e.g. `mine 4 iron ore.`; for logs only |
| `parsed` | map | `{verb, args}`: the `hivemind.lang` `Command` (contracts 4.4); for logs and to pick the wrapper |
| `plan` | map | concrete parameters resolved by the conductor (6.2): what Node executes |
| `deadline_ticks` | int | server ticks (by `age`) from receipt |
| `deadline_ms` | int | wall-clock ms from receipt |
| `sent_ms` | float | when the conductor sent it |

`parsed.args` is `Command.args` as a map from slot name to value, with exactly the slot names `hivemind.lang` uses. Values are concept names (str), ints, or maps for the two structured values: an `EntityRef` as `{ordinal, entity}` or `{ordinal, dropped: <item class>}`, a `Sentence` as `{intent, count?, item?}`. Examples, following contracts 4.4:

```
mine 4 iron ore.                    {verb: "mine",   args: {block: "iron_ore", count: 4}}
attack the second zombie.           {verb: "attack", args: {entity: {ordinal: 2, entity: "zombie"}}}
tell everyone i need 3 iron ingot.  {verb: "chat",   args: {sentence: {intent: "need", count: 3, item: "iron_ingot"}, to: "everyone"}}
```

Node uses `parsed.verb` to choose the wrapper and reads everything else from `plan`. It does not interpret `args`: those values are class names (`iron_ore`, `log`), and turning a class into something to act on needs the notebook. A class-level `mine 4 log.` becomes a list of known log positions; `craft 4 planks.` becomes a vanilla recipe the inventory can satisfy. This keeps the honesty rule (targets come only from what the agent perceived) on the Python side.

### 6.2 `plan` per verb (provisional, fixed with the wrappers in M2)

| Verb | `plan` fields | Mineflayer side |
|---|---|---|
| `goto` | `pos` and `range`, or `entity` (id) and `range`, or `player` (username) | pathfinder `GoalNear` / `GoalFollow` |
| `explore` | `xz`: the frontier point | pathfinder `GoalXZ` |
| `descend` | `y`: target level; `dir`: staircase direction | own loop: `dig` + move |
| `strip_mine` | `dir`, `length` | own loop: `dig` + move |
| `mine` | `targets`: block positions in the conductor's order, `count` | pathfinder + `dig`, best tool chosen automatically (`mineflayer-tool`), digs verified (DESIGN.md 5) |
| `craft` | `steps`: `[{result, times, table}]` with vanilla result names, `table` a position or nil | `recipesFor` with the inventory, `craft` |
| `smelt_start` | `furnace` (position), `input`, `count`, `fuel`, `fuel_count` | `openFurnace`, `putFuel`, `putInput` |
| `collect` | `furnace` | `openFurnace`, `takeOutput` |
| `place` | `item`, `cells`: positions in order | `placeBlock` |
| `attack` | `entity` (id) | own loop on `attack`, at the cooldown |
| `flee` | `entity` (id) or `from` (position); `min_dist` 16, `max_dist` 24 | pathfinder, inverted goal |
| `eat` | `item` | `consume` |
| `equip` | `item`, `dest`: `hand`, `off-hand`, `head`, `torso`, `legs`, `feet` | `equip` |
| `sleep` | `bed` (position) | `sleep` |
| `give` | `player` (username), `item`, `count` | goto to ≤ 3 blocks + `toss` |
| `chat` | `text`: the rendered line, e.g. `everyone, i need 3 iron ingot.` (DESIGN.md 9); `channel`: `global`, `team` or `whisper`; `to`: username or nil | `chat`, or `/teammsg` and `/msg` [mem] |
| `wait` | `ticks`: 20, 100 or 400 | no-op until the time is up or an interrupt |
| `store` | `chest` (position), `item`, `count` | `openContainer`, `deposit` |
| `take` | `chest`, `item`, `count` | `openContainer`, `withdraw` |

`item`, `input`, `fuel` and `result` are vanilla item names chosen by the conductor from the inventory (for `place cobblestone here`, the vanilla item of class `cobblestone` that the bot holds). Counts are "up to" (contracts 4.3): Node does as many units as it can, up to the count, and reports how many in `units`.

### 6.3 `cancel` and `retarget` (conductor → Node)

| Message | Fields | Effect |
|---|---|---|
| `cancel` | `agent`, `req`, `cause`: `message`, `damage`, `{hostile: <type>}` or `episode` | Node stops the action with full cleanup (6.6) and sends its result with code `INTERRUPTED` and that cause. A `cancel` for a request that has already finished is ignored: its result is already on the way |
| `retarget` | `agent`, `req`, `plan`: only the fields to replace | Updates a running action, e.g. more `targets` for `mine` when the notebook sees more of the class mid-action (the simulator's macro also uses what it perceives along the way) |

`episode` is for resets; the conductor discards that transition.

### 6.4 Interrupts: who detects what

A running action is interrupted by damage, a new hostile within 8 blocks, a message to the agent from a teammate or the human, a vanished target or a full inventory (DESIGN.md 4.5). Each is detected on the side that can do so without parsing English or holding the vocabulary:

| Trigger | Detected by | Result |
|---|---|---|
| health decreases | Node (`update_health`) | `INTERRUPTED`, cause `damage` |
| a hostile (`tables.hostile`) enters `interrupt_hostile_radius` during the action | Node | `INTERRUPTED`, cause `{hostile: <type>}` |
| chat addressed to the agent by a teammate or the human | conductor (needs the human tokenizer and the relation) | `cancel` with cause `message` |
| target entity gone, target block changed | Node | `TARGET_LOST` or `NO_TARGET` |
| no room for the next item | Node | `INV_FULL` |
| a deadline passes | Node | `TIMEOUT` |
| the bot dies | Node | `DIED` |

Messages from enemies do not interrupt; they only reach the chat field (DESIGN.md 4.5).

### 6.5 `result` (Node → conductor)

Exactly one per `act`, whatever happens.

| Field | Type | Meaning |
|---|---|---|
| `agent`, `req` | | as in `act` |
| `code` | str | failure code (DESIGN.md 5): `OK`, `NO_TARGET`, `NO_PATH`, `PATH_TIMEOUT`, `STUCK`, `DIG_FAILED`, `MISSING_INGREDIENT`, `NO_STATION`, `INV_FULL`, `TARGET_LOST`, `HAZARD`, `INTERRUPTED`, `DIED`, `TIMEOUT` |
| `cause` | str, map or nil | for `INTERRUPTED`: `damage`, `message` or `{hostile: <type>}` |
| `station` | str or nil | for `NO_STATION`: `crafting_table`, `furnace` or `bed` |
| `result` | str | the result words, e.g. `no path`, `interrupted by zombie`, rendered from `tables`. The conductor re-renders them from `code`, `cause` and `station` with `lang.result_words` and logs a protocol error on any mismatch |
| `units` | int | units done: blocks mined or placed, items crafted, stored, taken, given |
| `tick0`, `tick1` | int | client ticks at start and end |
| `age0`, `age1` | int | server age estimates at start and end |
| `wall_ms` | float | wall-clock duration |
| `n_roundtrips` | int | server confirmations waited for (window clicks, dig checks, ...) |
| `inv_delta` | map | vanilla item → signed count change, end minus start |
| `path` | map or nil | `{length, replans, search_ms}` for path-based actions (DESIGN.md 11.4) |
| `corrections` | int | server position corrections during the action |
| `aborted` | str or nil | `kick` or `host_restart` when infrastructure ended the action (6.6) |
| `raw_state` | map | the state at the end (7.1), plus `container` after `store`, `take`, `smelt_start` and `collect`: `{pos, kind: chest or furnace, slots}` exactly as the window showed it |

**Ordering.** Every event caused during an action is sent before its result: the `hurt` event before `INTERRUPTED` by damage, the `death` event before `DIED`.

### 6.6 Deadlines, cleanup and faults

- Both deadlines count from the receipt of `act`; whichever passes first ends the action with `TIMEOUT` (`too slow`). `deadline_ticks` is in server ticks, so slow-server profiles (`/tick rate 15`) do not cut digs short; `deadline_ms` catches hangs. The conductor sets them from the simulator's duration model with a margin [est], e.g. `deadline_ms ≥ deadline_ticks × 50 × 20/TPS + 2000` [est].
- **Cleanup** on any early end (deadline, cancel, death): `stopDigging`, stop the pathfinder (`setGoal(null)`), close any open window, clear the control states, stop using an item. Window clicks in 1.21.1 have no transaction packets and wait with no timeout [75], so the deadline is their only exit (DESIGN.md 5).
- **Kicks** are events, not episode ends (DESIGN.md 15, lesson 10). The running action ends with `TIMEOUT` and `aborted: kick`; the conductor counts it as an infrastructure fault, outside the success statistics, and decides whether to `join` the bot again. **Provisional:** whether the agent should see `too slow` here is open until M2 measures kick rates.

## 7. State stream (Node → conductor)

### 7.1 `state`, every `state_every_ticks` client ticks per bot

| Field | Type | Meaning |
|---|---|---|
| `agent`, `tick`, `age`, `day_time` | | `day_time` is the time of day from `update_time` |
| `pos`, `vel`, `yaw`, `pitch`, `on_ground` | | own movement |
| `health`, `food`, `saturation`, `oxygen`, `xp: {level, progress}` | | own status |
| `held_slot` | int | hotbar slot 0–8 |
| `ping` | int | ms, as Mineflayer reports it [6] |
| `weather` | map | `{rain, thunder}` as the client knows it |
| `entities` | array | every entity within `entity_radius`: `EntityRef` plus `pos`, `vel`, `yaw`, `health` (nil if unknown), `held` and `armor` (vanilla items or nil), `team` (scoreboard team for players, nil if none) |
| `inventory` | array or nil | all non-empty slots (section 3), sent only when the inventory changed since the last `state` |
| `chunks`, `unload`, `changes` | | world mirror (7.2) |

`raw_state` in a result has the same fields, without the world-mirror fields and with `inventory` always present.

### 7.2 World mirror

Python runs the shared perception (DESIGN.md 6.1) and the notebook on a mirror of the blocks the bot's client holds, so Node never decides what is visible. Only the part near the bot is mirrored:

- `chunks`: chunk columns that came within `mirror_radius_chunks` (Chebyshev distance, in chunks) of the bot's chunk: `{x, z, sections}`. Sections cover y −64..191 (section indices −4..11; the design's Y range, DESIGN.md 4.4). Each section is `{y, single: <state id>}` when uniform, otherwise `{y, zids: <bin>}`: 4096 little-endian uint16 state ids in the order index = (y × 16 + z) × 16 + x [mem], deflated with zlib.
- `unload`: `[[x, z], ...]` for columns that left the radius or that the server unloaded.
- `changes`: `[[x, y, z, state_id], ...]` for block updates inside the radius since the previous `state`, in arrival order.
- State ids are minecraft-data 1.21.1 ids, which are the protocol ids (verified at `hello` through the file hashes, and against the vanilla report by `servertool.py check-generator`). Python maps them to block classes with `data/block_classes-1.21.1.csv`.
- **Provisional:** block and sky light per section (`light: {block, sky}` as bin). The simulator's light is approximate (DESIGN.md 4.2), so the featurizer may compute light from the mirror instead.

## 8. Events (Node → conductor)

`event{agent, kind, tick, age, ...}`. The first five kinds are those of DESIGN.md 11.1; the others are what the notebook needs from the real game (section 9).

| `kind` | Fields | Source |
|---|---|---|
| `chat` | `from` (username), `uuid`, `text`, `chat_type` (registry name, e.g. `chat`, `msg_command_incoming`, `team_msg_command_incoming`), `from_self` | player chat only, from Mineflayer's `message` event with the sender *(checked: `lib/plugins/chat.js`)*. System messages are not forwarded. The conductor reads `text` with the human tokenizer (DESIGN.md 9) |
| `hurt` | `victim` and `source` (`EntityRef`, `source` nil if none), `damage_type` (registry name, e.g. `player_attack`, `mob_attack`, `fall`, `lava`), `health` (own health after the hit, when the victim is the bot) | the `damage_event` packet, which names the source entity; Mineflayer emits `entityHurt(entity, source)` from it *(checked: `lib/plugins/entities.js`)*. Sent for the bot and for every tracked entity |
| `death` | `killer` (`EntityRef` or nil), `death_key` (translation key, e.g. `death.attack.player`), `args` (the key's arguments as names), `pos` | the death message's translation key and arguments: structured data, not English text. `death.attack.player` is `%1$s was slain by %2$s` in the pinned language file, so `args[1]` is the killer |
| `kick` | `reason` (text component as JSON), `logged_in` | Mineflayer `kicked(reason, loggedIn)` *(checked)* |
| `correction` | `from`, `to`, `dist` | Mineflayer `forcedMove`: a server position correction or teleport *(checked)* |
| `entity_dead` | `entity` (`EntityRef`), `pos` | entity status 3 [mem] |
| `pickup` | `collector` (`EntityRef`), `item: {name, count}`, `item_entity` (id) | the collect packet (Mineflayer `playerCollect`) *(checked)* |
| `lid` | `pos`, `viewers` (int) | Mineflayer `chestLidMove`: a chest the bot can see was opened or closed *(checked: `lib/plugins/block_actions.js`)* |
| `respawn` | `pos` | after Node respawns the bot |

## 9. What the notebook gets from the real game

The owner asked how an agent remembers where things were, who is an enemy and how much is in a chest (DESIGN.md 1, 6.3). In the simulator, the notebook is written from the simulator's perception and the agent's own actions. In the real game, the same `Notebook` code is fed from this protocol, and every input is something the bot's own client received:

| Notebook table | Fed by |
|---|---|
| Places (`iron ore`, `my furnace 2`, `my chest 1`, `my bed`, `lava`) | perception on the world mirror (7.2) every `state`; own stations from the positions in successful `place` results |
| Players (relation, last seen, last message, damage, kills, gifts, thefts) | `state.entities` (name, position, `team`, from which the relation word follows, as a player sees name colours); `chat` events (last message words); `hurt` events (damage each way, from `source` and `victim`); `death` and `entity_dead` (kills each way); `pickup` events and `give` results (items given each way, attributed from the notebook's own perception history); `lid` events and container contents (items taken from my chests) |
| Containers (contents as last seen) | `raw_state.container` after `store` and `take`, passed to `notebook.on_container_view`. A drop since the last look becomes `3 iron ingot missing from my chest 1`; the thief is named only if the agent saw the chest opened (`lid`) with that player beside it |
| Events (ring buffer of sentences) | written by the conductor from the above, in the vocabulary's event sentences (contracts 4.3) |

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
Cond. → hello   {protocol: 1, session: "s-3f9c", rules_hash: "819981de...", vocab_hash: "...", settings: {...}, tables: {...}}
Cond. → join    {agent: "red 0", username: "red_0", server: {host: "127.0.0.1", port: 25565}, respawn_delay_ticks: null, ...}
Node  → joined  {agent: "red 0", entity_id: 812, pos: [12.5, 70.0, -3.5], age: 1200, tick: 0}
Node  → state   {agent: "red 0", tick: 5, chunks: [...], entities: [...], inventory: [...], ...}
Cond. → act     {agent: "red 0", req: 1, command: "mine 4 iron ore.", parsed: {verb: "mine", args: {block: "iron_ore", count: 4}},
                 plan: {targets: [[14, 52, -9], [15, 52, -9], [15, 53, -10]], count: 4}, deadline_ticks: 900, deadline_ms: 47000}
Node  → state   {agent: "red 0", tick: 10, changes: [[14, 52, -9, 0]], ...}
Node  → event   {agent: "red 0", kind: "hurt", victim: {id: 812, type: "player", name: "red_0"},
                 source: {id: 901, type: "zombie"}, damage_type: "mob_attack", health: 17.0}
Node  → result  {agent: "red 0", req: 1, code: "INTERRUPTED", cause: "damage", result: "interrupted by damage", units: 1,
                 inv_delta: {raw_iron: 1}, tick0: 7, tick1: 61, wall_ms: 2712.4, ...}
```

## 13. Open points

- **Slot names in `parsed.args`.** Contracts 4.4 writes `("count", 4)` for `mine`, while 4.5 lists the slot names `mine_count` and `item_count`. The protocol passes whatever `hivemind.lang` settles on; Node does not read `args`.
- **`plan` fields per verb** (6.2) are fixed with the wrappers in M2.
- **Kick results:** `TIMEOUT` with `aborted: kick` for now (6.6).
- **Light** in the world mirror (7.2).
- **Mirror bandwidth:** a column is 16 deflated sections; the bytes per bot per second at walking speed are measured in M2 [est].

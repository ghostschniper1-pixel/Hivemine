# Interface contracts (v1)

This file fixes the interfaces between modules so they can be built in parallel. DESIGN.md says *why*; this file says *what each module must provide*. If a contract proves wrong, change it here first, then in every module that uses it.

Owner requests folded in (2026-10-16):
- **Natural language in all actions.** Every action, and every symbol the agent sees (blocks, items, entities, results, chat), is expressed in plain English words from one shared vocabulary, so the network can learn what the words mean.
- **LLM ports.** Spare, inert layers and slots reserved in the network so an LLM can be coupled in later (DESIGN.md 6.4). They do not affect anything built here.

## 0. Conventions

- Python ≥ 3.11, NumPy only inside `hivemind/` (no other runtime dependencies). Tests use pytest and live in `tests/`.
- Game version is pinned: **Java Edition 1.21.1**. Never mix versions.
- Every hard-coded game constant carries a source label in a comment: a DESIGN.md reference like `# [13]`, `# [mem]` (from memory, unchecked), or `# [est]` (estimate, to be measured).
- Reference code is written to be read: plain Python, small functions, dataclasses. Speed comes later, in the C++ core (M4).
- Determinism: no use of Python's `random`, `time` or hash-randomized iteration in simulation code. All randomness goes through `hivemind.sim.ref.rng`.
- The default test suite must finish in under about 60 s. Mark longer tests `@pytest.mark.slow`.
- Workers never run git commands that change state (no add, commit, stash, checkout, reset).

Run tests with `.venv/bin/python -m pytest`.

## 1. Data sources (pinned)

`data/sources.json` records the exact upstream commits. Raw files are cached in `data/sources/` (git-ignored) by `python -m hivemind.rules.fetch_sources`.

| Source | Pin | Used for |
|---|---|---|
| PrismarineJS/minecraft-data | commit `201ca802a1b2c968b02e7fa5b31dd74e8bff3d8b`; files from `data/pc/1.21.1/` (blocks, items, foods, materials, recipes, language), `data/pc/1.20.5/` (entities, blockCollisionShapes), `data/pc/1.20/` (blockLoot, entityLoot: cross-check only, known to be stale) | hardness, harvest tools, states, display names, collision shapes, tool speeds |
| misode/mcmeta tag `1.21.1-data` | `aacd2a457333b258d044a642f38270c8cddbc628` | recipes, loot tables, tags, worldgen placed/configured features |
| misode/mcmeta tag `1.21.1-summary` | `7cffbcdad5fa36f22acef6bd4b869ecad4ca3fd9` | `item_components/data.json` (stack size, durability, food, tool rules and speeds, attack damage and speed), `blocks/data.json` (block states) |

Not reachable from the cloud container: Mojang downloads, minecraft.wiki, Fabric maven, Modrinth, arXiv. Anything that needs the real server jar runs on the owner's PC.

## 2. Rules (`data/rules-1.21.1.json`, `hivemind.rules`)

Built by `python -m hivemind.rules.build_rules` from the pinned sources. Committed to git so the project builds offline. Exact game data at the **vanilla** level (real block and item names).

```
{
  "version": "1.21.1",
  "build": {"sources": {...}, "hash": "<sha256 of the canonical JSON without this field>"},
  "blocks":   {name: {id, display_name, hardness (-1 = unbreakable), resistance, diggable, transparent,
                      emit_light, filter_light, default_state, min_state, max_state, states,
                      mineable: [tool kinds from tags mineable/*], harvest_tools: [item names] | null,
                      requires_tool: bool, needs_tier: null | "stone" | "iron" | "diamond",
                      loot_table: "blocks/<name>" | null, collision: <shape per state or shared>}},
  "items":    {name: {id, display_name, stack_size, max_damage | null, food: {nutrition, saturation} | null,
                      tool: {kind, tier, speed, rules} | null, attack_damage | null, attack_speed | null,
                      armor: {slot, points, toughness, knockback_resistance} | null,
                      fuel_ticks | null, equip_slot | null}},
  "tags":     {"block": {tag: [names, fully resolved]}, "item": {...}},
  "recipes":  [{id, type, category, group, pattern, key, ingredients, result: {item, count},
                cookingtime, experience}],            # tag ingredients resolved to name lists
  "loot_tables": {"blocks/<name>": <vanilla JSON verbatim>, "entities/<name>": ...},
  "entities": {name: {id, display_name, width, height, category, type, max_health, armor,
                      attack_damage, speed}},          # stats not in data files: hard-coded with labels
  "constants": {name: {"value": ..., "source": "[n]" | "[mem]" | "[est]"}}
}
```

- `attack_damage` and `attack_speed` are the item's attribute modifiers; the player's base (1 damage, 4 attacks/s) is added by the simulator.
- Armor values and fuel times are not in the 1.21.1 data dumps. Hard-code them with labels.
- The build cross-checks overlapping sources and **fails** on disagreement (recipes in both sources; stack sizes, durability and food in items/foods vs item components; harvest tools vs tool rules plus `needs_*_tool` tags; tool speeds in materials vs tool rules). Known representation differences are listed explicitly in the build script, each with a reason.

API: `hivemind.rules.load_rules(path=None) -> Rules` (cached; a thin typed wrapper over the JSON with helpers such as `block(name)`, `item(name)`, `recipes_for(item)`, `tag(kind, name)`).

## 3. Concepts (`data/concepts-1.21.1.json`, `data/block_classes-1.21.1.csv`, `hivemind.rules.classes`)

The agent does not see 1,000+ vanilla blocks and items. It sees **classes**: sets of vanilla blocks or items that behave the same in v1. Each class has a natural-language **phrase**. The simulator works at the class level throughout (voxels hold block-class ids, inventories hold item-class counts). The bridge maps real names to classes.

```
{
  "version": "1.21.1", "hash": "...",
  "block_classes":  [{id (0 = air, 255 = buried), name, phrase: [words], members: [vanilla block names],
                      resource: bool,            # a valid target for mine/goto and resource memory
                      hardness, mineable, requires_tool, needs_tier,
                      drops: <class-level drop rule or loot-table reference>,
                      flags: {solid, collision_height, transparent, liquid, climbable, hazard,
                              gravity, replaceable, light}}],
  "item_classes":   [{id (0 = none), name, phrase: [words], members: [vanilla item names], v1: bool,
                      stack_size, max_damage, food, tool, attack_damage, attack_speed, armor,
                      fuel_ticks, places_block_class: name | null, equip_slot}],
  "entity_classes": [{id, name, phrase: [words], members: [vanilla entity names],
                      kind: "hostile" | "passive" | "player" | "item" | "other",
                      max_health, armor, attack_damage, width, height}]
}
```

Rules:
- Every 1.21.1 block state, item and entity maps to exactly one class. The CSV has one row per block state: `state_id,block,class_id,class`.
- **Homogeneity.** All members of a class share every property the simulator uses: hardness, mineable tool kinds, harvest requirement, collision, transparency, light, hazard, and drops after mapping items to item classes; for items: stack size, durability, food, tool, armor, fuel and recipe roles. The build checks this and fails, except for explicitly listed and justified exceptions. Consequence: `iron ore` and `deepslate iron ore` are different classes (hardness 3 vs 4.5), while all eight log types form one class `log` (identical in v1).
- Block-class ids fit in one byte. About 100–140 curated classes for natural overworld terrain and early-game placed blocks. Rare or out-of-scope blocks fall back to generic classes keyed by their physical signature (so break times stay exact); fallback classes are never resources.
- Item classes: curated for the v1 tech tree and common overworld drops (about 150–200), with `v1: true`. Everything else falls back to non-v1 classes keyed by stack size.
- Phrases come from English display names (minecraft-data `language.json`), lower-cased and split on spaces and punctuation, with explicit overrides where a class merges members (e.g. `log`, `planks`, `wool`, `bed`). Phrases are unique within each kind. Reserved: air = `air`, buried = `buried`.
- Known v1 approximation: class merging hides some real-game details (a bed needs three wool of one colour). List each such case in the class builder.

API: `hivemind.rules.classes.load_concepts(path=None) -> Concepts` with `block_class(name|id)`, `item_class(name|id)`, `entity_class(name|id)`, `block_class_of_state(state_id)`, `item_class_of(vanilla_item)`, and arrays for the simulator (`hardness[class_id]`, flag arrays, etc.).

## 4. Language (`hivemind.lang`, `data/vocab-1.21.1.json`)

One controlled English vocabulary serves actions, observations, chat and human input. Words mean the same thing everywhere, so the embedding of `iron` is trained by `mine 4 iron ore.`, by `raw iron` in the inventory, by `i need 3 iron ingot` in chat, and by the human typing "got any iron?".

### 4.1 Tokens

- Lower-case words. Numbers are single tokens (`0`–`20`, `32`, `48`, `64`, `-16`, `-32`, `-54`). `.` ends every command and chat sentence.
- Reserved ids: `<pad>` = 0, `<unk>` = 1, `<bos>` = 2 (decoder start), `.` = 3.
- Word list order is stable and saved in `data/vocab-1.21.1.json` with a hash. Changing the vocabulary changes the network's input and output layers, so it is versioned.
- Eight **nonce words** for teaching new meanings in chat (DESIGN.md 9): `dax wug blicket fep toma zib kiki bouba`. They are valid only in chat item slots; their meaning is drawn per episode.
- Canonical text uses the singular display-name form (`mine 4 iron ore.`, `craft 4 stick.`). The human tokenizer accepts plurals.

### 4.2 Phrases

Every concept the agent can name has a phrase (a tuple of words): block classes, item classes, entity classes, players, verbs, directions, place patterns, equipment slots, result codes, interrupt causes, chat intents, numbers. Players are named `red 0`–`red 19`, `blue 0`–`blue 19` and `human` (team words are configurable). The single agent of M1–M5 is `red 0`.

### 4.3 Command grammar

The design's 17 verbs keep their ids, and two container verbs are added (19); each has one or more English templates. Slots are in braces; everything else is a literal word. Every command ends with `.`.

| Verb id | Templates | Notes |
|---|---|---|
| `goto` | `go to the nearest {block}` · `go to the {ordinal} {entity}` · `go to {player}` · `go to my furnace {furnace}` · `go home` | block from resource memory; entity from the visible list |
| `explore` | `explore {dir8} {explore_dist} blocks` | dir8: north, northeast, east, southeast, south, southwest, west, northwest; explore_dist: 16, 32, 64 |
| `descend` | `dig down to y {y_band}` | y_band: 48, 16, 0, -16, -32, -54 |
| `strip_mine` | `strip mine {dir4} {strip_dist} blocks` | dir4: north, east, south, west; strip_dist: 16, 32 |
| `mine` | `mine {mine_count} {block}` | mine_count: 1, 2, 4, 8, 16 |
| `craft` | `craft {item_count} {item}` | item_count: 1, 2, 3, 4, 6, 8, 12, 16, 32, 64 |
| `smelt_start` | `smelt {item_count} {item}` | fuel chosen automatically |
| `collect` | `collect from my furnace {furnace}` | furnace: 1–4, the agent's own furnaces in placement order |
| `place` | `place {item} {pattern}` | pattern: `here`, `in front`, `under me` (pillar), `around me` (enclose), `for light` |
| `attack` | `attack the {ordinal} {entity}` · `attack {player}` | |
| `flee` | `flee from the {ordinal} {entity}` · `flee from {player}` | |
| `eat` | `eat {item}` | |
| `equip` | `hold {item}` · `hold {item} in off hand` · `wear {item}` | `wear` for armor; slot follows from the item |
| `sleep` | `sleep` | |
| `give` | `give {item_count} {item} to {player}` | |
| `chat` | `tell {addressee} {sentence}` | addressee: a player, `everyone` or `team` |
| `wait` | `wait {wait_seconds} seconds` | wait_seconds: 1, 5, 20 (20, 100, 400 ticks) |
| `store` | `store {item_count} {item} in my chest {chest}` | **off by default** (feature flag `containers`); chest: 1–8 |
| `take` | `take {item_count} {item} from my chest {chest}` | **off by default**; item: what the agent's memory says is in that chest |

`goto` gains `go to my chest {chest}` and `go to my bed` (both masked unless known). The store and take verbs and their words are in the vocabulary from the start, so enabling them later (team stage, M9) does not change the network's input and output layers. That makes 19 verb ids.

- `ordinal` is `nearest`, `second`, `third`, … `sixteenth`: the rank by distance among visible entities with the same phrase.
- Entity phrases: the entity class phrase (`zombie`, `cave spider`, `cow`), or `dropped {item}` for an item lying on the ground.
- **Counts mean "up to".** `craft 64 stick.` crafts as many as possible up to 64; `give 8 bread to red 3.` gives at most what is held. A count is valid if at least one unit can be done. This keeps masks simple (the count does not depend on the item chosen after it) and matches how people talk.

Chat sentences (the 14 intents of DESIGN.md 9):

| Intent | Sentence |
|---|---|
| need / have / give | `i need {item_count} {item}` · `i have {item_count} {item}` · `give me {item_count} {item}` |
| where | `where is {item}` |
| come, follow, help | `come here` · `follow me` · `help me` |
| attack, flee, stop, wait | `attack` · `flee` · `stop` · `wait` |
| yes, no, thanks | `yes` · `no` · `thanks` |

Example: `tell everyone i need 3 iron ingot.` In chat item slots the nonce words are also allowed.

**Results** (shown to the agent after each action, and logged): `done`, `no target`, `no path`, `path too slow`, `stuck`, `dig failed`, `missing ingredient`, `no {station} nearby` (station: crafting table, furnace, bed), `inventory full`, `target lost`, `danger ahead`, `interrupted by {cause}` (cause: `damage`, `message`, or the entity phrase of the hostile that appeared), `died`, `too slow`. These map one-to-one to the failure codes of DESIGN.md 5 (`OK`, `NO_TARGET`, …).

**Event sentences** (written into the memory's event log, section 6.1; same vocabulary):
`{player} attacked me` · `{entity} attacked me` · `i attacked {player}` · `{player} killed me` · `{entity} killed me` · `i killed {player}` · `i killed {entity}` · `{player} gave me {n} {item}` · `i gave {n} {item} to {player}` · `i died` · `{player} took {n} {item} from my chest {chest}` (seen only when the agent next opens the chest and the count dropped: `{n} {item} missing from my chest {chest}`) · `i found {block}` (first sighting of a resource class this episode).

**Unambiguity.** The grammar must be prefix-safe: `parse(render(c)) == c` for every command over the v1 concept sets. Any phrase collision that breaks this (for example a block phrase equal to an entity phrase after `go to the nearest`) is fixed by a phrase override, and a test enumerates commands to check it.

### 4.4 Python API

```python
from hivemind.lang import Language, Command, Sentence, EntityRef, ParseError

lang = Language.load()                    # data/vocab-1.21.1.json; Language.build(concepts) makes it
lang.words                                # list[str]; lang.word_id(w) -> int; lang.word(i) -> str
lang.phrase(kind, name) -> tuple[str, ...]           # kind: "block" | "item" | "entity" | ...
lang.concept(kind, words) -> str                     # inverse of phrase

Command(verb="mine", args=(("block", "iron_ore"), ("count", 4)))     # frozen; args sorted by slot name
Command(verb="chat", args=(("sentence", Sentence("need", count=3, item="iron_ingot")), ("to", "everyone")))
EntityRef(ordinal=2, entity="zombie")      # entity: entity-class name, or ("dropped", item-class name)

lang.render(cmd) -> str                   # "mine 4 iron ore."
lang.to_words(cmd) -> list[str]           # ["mine", "4", "iron", "ore", "."]
lang.encode(cmd) -> list[int]; lang.decode(ids) -> Command
lang.parse(text_or_words) -> Command      # strict canonical parse; raises ParseError

lang.normalize_human(text) -> list[str]   # lower-case, split, plurals to lemmas, number words to digits,
                                          # synonyms ("pick" -> "pickaxe", "workbench" -> "crafting table"),
                                          # unknown words -> "<unk>"
lang.parse_human(text, speaker) -> Command | Sentence | None   # lenient: drops filler words,
                                          # maps counts to the nearest bin, resolves "me"/"i" to the speaker

lang.result_words(code, cause=None, station=None) -> list[str]
```

### 4.5 Masked, word-by-word generation

The policy writes each command one word at a time. A grammar mask allows only words that continue some valid command, so invalid actions are impossible (DESIGN.md 5, [65]). Words with no alternative are fed in without a decision.

```python
class SlotProvider(Protocol):             # implemented by the simulator (and later the bridge)
    def verbs(self) -> Collection[str]: ...                       # verbs with at least one valid command
    def options(self, verb: str, slot: str, chosen: Mapping[str, Any]) -> Collection[Any]: ...
        # valid values for one slot, given the verb and the slots already chosen (left to right)

dec = lang.decoder(provider)
while not dec.done:
    mask = dec.allowed()                  # np.ndarray[bool] of len(lang.words)
    if dec.forced: dec.feed(int(mask.argmax()))   # one choice only: not a policy decision
    else: dec.feed(policy_pick(mask))
cmd = dec.command()
```

Slot names used by `options`: `block`, `item`, `entity` (returns `EntityRef`s), `player`, `furnace`, `chest`, `dir8`, `dir4`, `explore_dist`, `strip_dist`, `y_band`, `mine_count`, `item_count`, `pattern`, `wait_seconds`, `addressee`, `intent`, plus `template` (which template of a verb; values are template ids). Values are concept names or ints exactly as they appear in `Command.args`.

The network side (built in M5) ties output scores to the same word embeddings that read the observation, sums log-probabilities over the non-forced words, and uses the sum of per-step entropies along the sampled command (an unbiased estimate of the joint entropy).

## 5. Reference simulator (`hivemind.sim.ref`)

Readable Python, the oracle for the C++ core. Works at the class level (section 3) and speaks the language of section 4.

| Module | Provides |
|---|---|
| `rng.py` | `CounterRNG(seed)`: counter-based (SplitMix64-style) draws keyed by `(episode, tick, purpose, entity, draw)`; `u64`, `uniform`, `randint`, `choice`, `poisson`. `Purpose` enum of draw purposes |
| `mechanics.py` | Pure functions and small state classes for section 4.3 of DESIGN.md: `break_ticks(...)` (exact float32 accumulation as vanilla), `can_harvest`, `HungerState` (exhaustion, saturation, regeneration, starvation by difficulty), eating, clock helpers, attack cooldown and damage multiplier, critical hits, armor reduction, fall damage, smelting time, fuel |
| `inventory.py` | `Inventory` over item classes: 36 main slots, armor, off hand, held slot; stack-limited slot accounting; tool durability |
| `crafting.py` | Class-level recipe book: `craftable(inv, near_table)`, `craft(inv, item_class, up_to)`, smelting recipes |
| `loot.py` | Vanilla loot-table sampler (the subset used by v1 blocks and mobs; no Fortune, no Silk Touch), returning item-class counts |
| `world.py` | `World`: uint8 block-class voxels (x, y, z) with y −64..191, block light, heightmap, entity slot arrays, clock, chunk ticking by simulation distance |
| `worldgen.py` | Test worlds until M3: a simple terrain generator with trees and ores placed from the 1.21.1 placed-feature height ranges (labelled [est]) |
| `perception.py` | Exposed faces, the 15×15×7 voxel window with `buried`, line-of-sight resource memory within 24 blocks, the X-ray ablation |
| `nav.py` | Nav grid and A* with the pathfinder rules of DESIGN.md 4.5; `T_goto` duration model with placeholder coefficients [est] |
| `mobs.py` | Spawning (placeholder rates [est] until M3 measures them), chase, attacks, creeper fuse, sunlight burning, despawn, animals |
| `actions.py` | The 17 verbs as interruptible macros with durations, masks (`SlotProvider`) and failure codes |
| `rewards.py` | Achievements (own production only), potential-based shaping, night bonus, death |
| `env.py` | `RefEnv`: reset, `act(agent, command or text)`, `step()` returning ready agents, results, rewards |
| `viewer.py` | Text and top-down ASCII views |

## 6. Observations (`hivemind.features`)

One observation builder, shared by the simulator and the bridge (DESIGN.md 6). Every symbolic field is given as words (token ids from section 4) plus numbers; see DESIGN.md 6.2 for the field list.

### 6.1 Memory (`hivemind/features/memory.py`)

Weights do not change during play, and a GRU is poor at exact recall over a 24,000-tick day. So each agent keeps an explicit **notebook**, written by code and read by the network. The GRU still carries short-term context.

- **Honest:** written only from what the agent itself perceived (section 5 `perception.py`), did, or was told in chat. Never from hidden state.
- **Persistent:** survives death and respawn within an episode; cleared at episode reset.
- **Shared code:** the same `Notebook` runs in the simulator and the bridge.
- **Read as tokens:** every entry is a phrase (words) plus numbers, encoded by a small MLP and read by cross-attention from the trunk (DESIGN.md 6.3), so many entries stay cheap.

| Table | Capacity | Entry |
|---|---|---|
| Places | 48 | kind (`resource`, `station`, `danger`, `home`), phrase (`iron ore`, `my furnace 2`, `my chest 1`, `my bed`, `lava`), position, count seen, first and last seen tick. Eviction: the entry with the lowest value of a fixed score (resource tier, distance, age) [est] |
| Players | 40 | name (`blue 7`), relation from the visible team colour (`teammate`, `enemy`, `human`), last seen position and tick, last message words, counters: damage dealt to me, damage I dealt, kills each way, items given each way, items taken from my chests |
| Containers | 8 own + 8 seen | phrase (`my chest 1`), position, contents (item class → count) as of the last time the agent looked, tick of that look; the agent's own store and take actions update it exactly |
| Events | 16, ring buffer | an event sentence (section 4.3) and its tick |

API sketch: `Notebook(lang, config)`, `notebook.observe(perception_result, tick)`, `notebook.on_event(event)`, `notebook.on_container_view(chest, contents, tick)`, `notebook.entries() -> list[MemoryToken]` (words, numbers), `notebook.reset()`. The simulator's `goto`, `mine`, `store` and `take` masks read the notebook, not the true world.

## 7. Bridge protocol (`bridge/`)

As DESIGN.md 11.1, with the command in words: the conductor sends `act{agent, req, command: "mine 4 iron ore.", deadline_ticks, deadline_ms}`; Node replies `result{agent, req, code, result: "no path", tick0, tick1, wall_ms, n_roundtrips, inv_delta, raw_state}`. The Node side never parses English itself; the conductor sends the parsed command alongside the text (`parsed: {verb, args}`), so both sides share one parser.

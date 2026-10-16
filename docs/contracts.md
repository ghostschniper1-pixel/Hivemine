# Interface contracts (v1)

This file fixes the interfaces between modules so they can be built in parallel. DESIGN.md says *why*; this file says *what each module must provide*. If a contract proves wrong, change it here first, then in every module that uses it.

Owner requests folded in (2026-10-16):
- **Natural language in all actions.** Every action, and every symbol the agent sees (blocks, items, entities, results, chat), is expressed in plain English words from one shared vocabulary, so the network can learn what the words mean.
- **LLM ports.** Spare, inert layers and slots reserved in the network so an LLM can be coupled in later (DESIGN.md 6.5). They do not affect anything built here.
- **Explicit memory** (DESIGN.md 6.3) and a **goal section** shared by the agent and a later LLM (DESIGN.md 6.6). Goals stay off until M11; their words and slots are reserved now.

Added in draft 0.5 (second round of 2026-10-16; DESIGN.md section 1):
- **Surface-only vision.** A cell is seen only through see-through cells on a straight segment from the eye; every other window cell is the reserved class 255 `unseen` (draft 0.4's `buried`). X-ray stays only as a labelled ablation (DESIGN.md 6.1).
- **A bigger network with a low-resolution image.** Network presets S and M (DESIGN.md 6.4); a 21 × 21 × 9 window preset and a semantic panorama for M (section 6 here).
- **F3 debug inputs** in the observation, with biome classes (sections 3 and 6).

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
| PrismarineJS/minecraft-data | commit `201ca802a1b2c968b02e7fa5b31dd74e8bff3d8b`; files from `data/pc/1.21.1/` (blocks, items, foods, materials, recipes, language), `data/pc/1.20.5/` (entities, blockCollisionShapes, biomes), `data/pc/1.20/` (blockLoot, entityLoot: cross-check only, known to be stale) | hardness, harvest tools, states, display names (biome names from `language.json` `biome.minecraft.<name>`), collision shapes, tool speeds |
| misode/mcmeta tag `1.21.1-data` | `aacd2a457333b258d044a642f38270c8cddbc628` | recipes, loot tables, tags (including the biome tag `worldgen/biome/is_overworld`), worldgen placed/configured features |
| misode/mcmeta tag `1.21.1-summary` | `7cffbcdad5fa36f22acef6bd4b869ecad4ca3fd9` | `item_components/data.json` (stack size, durability, food, tool rules and speeds, attack damage and speed), `blocks/data.json` (block states), `registries/data.json` (registries, including `worldgen/biome`: the 64 biomes of 1.21.1) |

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

API: `hivemind.rules.load_rules(path=None) -> Rules` (cached; a thin typed wrapper over the JSON with helpers such as `block(name)`, `item(name)`, `recipes_for(item)`, `tag(kind, name)`, `block_of_state(state)`, `mining_speed(tool, block)`, `can_harvest(tool, block)`, `constant(name)`).

As built, the file also has these additive fields, documented in `hivemind/rules/build_rules.py`: `collision` = `{shapes, by_state}` (`by_state` null when every state uses `shapes[0]`); `tool.damage_per_block`; `food.eat_seconds`, `can_always_eat`, `converts_to`, `effects`; `recipes[].special`. Wall-mounted blocks take the standing variant's loot table. Tool tiers are `wood`, `stone`, `iron`, `gold`, `diamond`, `netherite`. Hard-coded tables keep their labels in `constants` under `armor:<material>`, `fuel:<item or #tag>` and `mob:<entity>:<stat>`. `python -m hivemind.rules.build_rules --check` confirms the committed file is current.

## 3. Concepts (`data/concepts-1.21.1.json`, `data/block_classes-1.21.1.csv`, `hivemind.rules.classes`)

The agent does not see 1,000+ vanilla blocks and items. It sees **classes**: sets of vanilla blocks or items that behave the same in v1. Each class has a natural-language **phrase**. The simulator works at the class level throughout (voxels hold block-class ids, inventories hold item-class counts). The bridge maps real names to classes.

```
{
  "version": "1.21.1", "hash": "...",
  "block_classes":  [{id (0 = air, 255 = unseen), name, phrase: [words], members: [vanilla block names],
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
                      max_health, armor, attack_damage, width, height}],
  "biomes":         [{id (the vanilla registry id, 0..63), name, phrase: [words], members: [the biome],
                      overworld: bool,           # in the biome tag worldgen/biome/is_overworld
                      dimension: "overworld" | "nether" | "end" | null}]      # draft 0.5
}
```

Rules:
- Every 1.21.1 block state, item and entity maps to exactly one class. The CSV has one row per block state: `state_id,block,class_id,class`.
- **Homogeneity.** All members of a class share every property the simulator uses: hardness, mineable tool kinds, harvest requirement, collision, transparency, light, hazard, and drops after mapping items to item classes; for items: stack size, durability, food, tool, armor, fuel and recipe roles. The build checks this and fails, except for explicitly listed and justified exceptions. Consequence: `iron ore` and `deepslate iron ore` are different classes (hardness 3 vs 4.5), while all eight log types form one class `log` (identical in v1).
- Block-class ids fit in one byte. About 100–140 curated classes for natural overworld terrain and early-game placed blocks. Rare or out-of-scope blocks fall back to generic classes keyed by their physical signature (so break times stay exact); fallback classes are never resources.
- Item classes: curated for the v1 tech tree and common overworld drops (about 150–200), with `v1: true`. Everything else falls back to non-v1 classes keyed by stack size.
- Phrases come from English display names (minecraft-data `language.json`), lower-cased and split on spaces and punctuation, with explicit overrides where a class merges members (e.g. `log`, `planks`, `wool`, `bed`). Reserved: air = `air`, and id 255 = `unseen` (draft 0.5; draft 0.4's `buried`): a window cell the agent does not see (DESIGN.md 6.1). `unseen` has no members and is never stored in a world.
- **Biomes** (draft 0.5): every 1.21.1 biome of the `worldgen/biome` registry (64; 53 with `overworld`), one class each, never merged. Phrase: the words of `language.json` `biome.minecraft.<name>`, lower-cased (`birch forest`, `plains`). Biome phrases are unique and equal no curated block, item or entity phrase. The F3 group names the biome at the feet by its phrase (section 6). Test worlds assign one biome per column, and it must be true of the column (DESIGN.md 4.4): consistent with the surface generated there and with the ore-table rows placed there (`ore_table.OreFeature.in_biome`), and never one of the underground biomes (`lush caves`, `dripstone caves`, `deep dark`). Imported worlds (M3) keep the game's 3-D biome cells, which is where the underground biomes appear.
- **Curated phrases are unique** within each kind (curated block classes, v1 item classes, entity classes), so everything a command can name has an invertible phrase. No block phrase equals an entity phrase.
- **Fallback phrases must be true of every member**, because their words are shared with curated classes and the agent learns word meanings from them. A fallback is named by the words all its members' names end with (`carpet`, `slab`), prefixed by `other` if a curated class already has those words, or else generically (`other pickaxe block`). Fallback phrases may repeat (fallbacks never appear in commands), but never equal a curated phrase. Never name a class after one member that does not describe the rest (coal blocks must not be called `bell`).
- Known v1 approximations: class merging hides some real-game details (a bed needs three wool of one colour), and members of one item class share stacks in the simulator but not in the game (DESIGN.md 4.2). Each case is listed in the concepts file's `exceptions`.

API: `hivemind.rules.classes.load_concepts(path=None) -> Concepts` with `block_class(name|id)`, `item_class(name|id)`, `entity_class(name|id)`, `biome(name|id)`, `block_class_of_state(state_id)`, `item_class_of(vanilla_item)`, `entity_class_of(vanilla_entity)`, `biome_of(vanilla_biome)`, `class_by_phrase(kind, words)` (defined for every curated class; a phrase shared by several fallbacks raises `KeyError`), and arrays for the simulator (`hardness[class_id]`, flag arrays, etc.). Additive fields (`state_ranges`, `tool_speeds`, `harvest_tools`, `drops`, `fallback`, `exceptions`, class-level `recipes`) are documented in `hivemind/rules/classes.py`.

## 4. Language (`hivemind.lang`, `data/vocab-1.21.1.json`)

One controlled English vocabulary serves actions, observations, chat and human input. Words mean the same thing everywhere, so the embedding of `iron` is trained by `mine 4 iron ore.`, by `raw iron` in the inventory, by `i need 3 iron ingot` in chat, and by the human typing "got any iron?".

### 4.1 Tokens

- Lower-case words. Numbers are single tokens (`0`–`20`, `32`, `48`, `64`, `-16`, `-32`, `-54`). `.` ends every command and chat sentence.
- Reserved ids: `<pad>` = 0, `<unk>` = 1, `<bos>` = 2 (decoder start), `.` = 3.
- Word list order is stable and saved in `data/vocab-1.21.1.json`. Changing the vocabulary changes the network's input and output layers, so it is versioned with two hashes: `hash` covers the whole file (provenance: record it with runs and replays), and `words_hash` covers only what the network and the grammar masks depend on (the words, concept rows, domains and templates: compare it to decide whether a checkpoint fits). M0's build had 504 words; draft 0.5 drops `buried` and adds 57 (section 4.2), about 560 in all.
- Eight **nonce words** for teaching new meanings in chat (DESIGN.md 9): `dax wug blicket fep toma zib kiki bouba`. They are valid only in chat item slots; their meaning is drawn per episode.
- Canonical text uses each class's display-name form unchanged, whatever the count (`mine 4 iron ore.`, `craft 4 stick.`, `craft 4 planks.`). Some display names are plural (`planks`, `leaves`); that is fine. The human tokenizer maps plural and singular forms to whichever form the vocabulary has.
- Counts that are not number tokens (in event sentences, or typed by a human) are written as the nearest number token (ties round down; above 64 is `64`).

### 4.2 Phrases

Every concept the agent can name has a phrase (a tuple of words), looked up with `lang.phrase(kind, name)`. The phrase kinds are: block, item, entity and biome classes, players, goal setters, addressees, nonce words, numbers, ordinals, directions, place patterns, stations, containers, interrupt causes, relation words, place kinds, equipment slots, and from draft 0.5 time words and face words. Verbs, chat intents, goals, result codes and event kinds are not phrase kinds: they are named by their templates (section 4.3).

Players are the concept names `red_0` … `red_19`, `blue_0` … `blue_19` and `human`, with the phrases `red 0` … `blue 19` and `human` (team words are configurable); a coupled LLM writes goals as `llm`. The single agent of M1–M5 is `red_0` (`red 0`).

The vocabulary must also hold, from the start, the words that appear only in observations (DESIGN.md 6.2), because adding a word later changes the network's input and output layers:
- relation words: `teammate`, `enemy`, `human` (players); `hostile`, `passive` (mobs); `object` (dropped items, arrows, experience orbs, primed TNT);
- the notebook's place kinds: `resource`, `station`, `danger`, `home`;
- the words of event sentences (below) and of goals (DESIGN.md 6.6), and the goal setters `me` and `llm`;
- draft 0.5: `unseen` (block class 255, replacing `buried`); every word of every biome phrase (50 of the 66 are new, e.g. `birch`, `forest`, `plains`); the time words `day`, `dusk`, `night`, `dawn`; the face words `top` and `bottom` (`north`, `east`, `south`, `west` exist as directions). That is 57 new words.

### 4.3 Command grammar

The design's 17 verbs keep their ids; two container verbs and a goal verb are added, making 20 verb ids. Each has one or more English templates. Slots are in braces; everything else is a literal word. Every command ends with `.`.

| Verb id | Templates | Notes |
|---|---|---|
| `goto` | `go to the nearest {block}` · `go to the {ordinal} {entity}` · `go to {player}` · `go to my furnace {furnace}` · `go home` | block from the notebook's places (section 6.1); entity from the visible list |
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
| `plan` | `plan to {goal}` | **off by default** (feature flag `goals`, on in M11); writes the agent's own goal slot; ≤1 per 40 ticks |

`goto` also has `go to my chest {chest}` and `go to my bed` (both masked unless known). The words of `store`, `take` and `plan` are in the vocabulary from the start, so enabling them later (M9, M11) does not change the network's input and output layers.

Goals (DESIGN.md 6.6), used by `plan` and by goals that the owner, an LLM or the curriculum set:

| Goal | Template | Met when [est where not obvious] |
|---|---|---|
| `get` | `get {item_count} {item}` | the inventory holds the count |
| `kill` | `kill {mine_count} {entity}` | that many of the entity class killed by the agent since the goal was set |
| `home` | `go home` | within 3 blocks of home |
| `to_player` | `go to {player}` | within 3 blocks of the player |
| `to_block` | `go to the nearest {block}` | within reach of a block of that class |
| `explore` | `explore {dir8}` | 64 blocks moved in that direction since the goal was set |
| `survive` | `survive until morning` | alive at tick 0 of the next day (when the time word turns from `dawn` to `day`, section 6) |
| `help` | `help {player}` | within 16 blocks of the player for 1,200 ticks |
| `protect` | `protect {player}` | the player takes no damage from others while the agent is within 16 blocks, for 1,200 ticks |

- `ordinal` is `nearest`, `second`, `third`, … `sixteenth`: the rank by distance among visible entities with the same phrase. There is no separate ordinal slot: in commands, `{ordinal} {entity}` is one slot `entity` whose value is an `EntityRef(ordinal, entity)` (section 4.4).
- Entity phrases: the entity class phrase (`zombie`, `cave spider`, `cow`), or `dropped {item}` for an item lying on the ground.
- In goals (`kill {mine_count} {entity}`) and in event sentences, `{entity}` is an entity-class name, not an `EntityRef`. `hivemind.lang.decoder.slot_domain(verb, slot)` gives the domain of every slot of a verb, nested sentences and goals included.
- **Counts mean "up to".** `craft 64 stick.` crafts as many as possible up to 64; `give 8 bread to red 3.` gives at most what is held. A count is valid if at least one unit can be done. This keeps masks simple (the count does not depend on the item chosen after it) and matches how people talk.

Chat sentences (15 sentences for the 14 intents of DESIGN.md 9):

| Intent | Sentence |
|---|---|
| need / have / give | `i need {item_count} {item}` · `i have {item_count} {item}` · `give me {item_count} {item}` |
| where | `where is {item}` · `where is {block}` (draft 0.5; `{block}` is a resource block class, as in `mine`) |
| come, follow, help | `come here` · `follow me` · `help me` |
| attack, flee, stop, wait | `attack` · `flee` · `stop` · `wait` |
| yes, no, thanks | `yes` · `no` · `thanks` |

Example: `tell everyone i need 3 iron ingot.` In chat item slots the nonce words are also allowed. The two `where` sentences are one intent; the slot name tells them apart (`Sentence("where", block="iron_ore")`, `Sentence("where", item="iron_ingot")`). Seventeen resource block classes share their phrase with a v1 item class in the M0 tables (`log`, `stone`, `sand`, `dirt`, `obsidian`, …). Those words are read as the block, the place to go: `where is {item}` does not offer those items, so the grammar stays prefix-safe and `parse(render(s)) == s` holds [design]. A scripted helper treats both readings alike (it gives the item if it holds some, else leads to the block it knows).

**Results** (shown to the agent after each action, and logged; the table in DESIGN.md 5 is authoritative): `done`, `no target`, `no path`, `path too slow`, `stuck`, `dig failed`, `missing ingredient`, `no {station} nearby` (station: crafting table, furnace, bed), `{container} full`, `target lost`, `danger ahead`, `interrupted by {cause}` (cause: `damage`, `message`, or the entity phrase of the hostile that appeared), `died`, `too slow`. These map one-to-one to the failure codes of DESIGN.md 5 (`OK`, `NO_TARGET`, …). The container is `inventory` when a full inventory stops `mine`, `strip_mine`, `craft`, `collect` or `take` partway; `chest` when `store` finds the chest full; and `furnace` (draft 0.5) when the furnace's own slots block `smelt_start`. Of the interruptions in DESIGN.md 4.5, a vanished target ends the action with `target lost` and a full inventory with `inventory full`.

**Event sentences** (written into the notebook's event log, section 6.1; same vocabulary; when each is written is fixed in DESIGN.md 6.3), with their kind names:

| Kind | Sentence |
|---|---|
| `player_attacked_me` | `{player} attacked me` |
| `entity_attacked_me` | `{entity} attacked me` |
| `i_attacked_player` | `i attacked {player}` |
| `player_killed_me` | `{player} killed me` |
| `entity_killed_me` | `{entity} killed me` |
| `i_killed_player` | `i killed {player}` |
| `i_killed_entity` | `i killed {entity}` |
| `i_died` | `i died` |
| `player_gave_me` | `{player} gave me {n} {item}` |
| `i_gave` | `i gave {n} {item} to {player}` |
| `player_opened_chest` | `{player} opened my chest {chest}` (only when exactly one other visible player is near the chest) |
| `someone_opened_chest` | `someone opened my chest {chest}` (otherwise) |
| `missing_from_chest` | `{n} {item} missing from my chest {chest}` (when the agent next opens the chest and a count dropped; no event names a thief) |
| `i_found` | `i found {block}` (first sighting of a resource class this episode) |

In events, `{entity}` is an entity-class name and `{n}` a count snapped to a number token (section 4.1).

**Unambiguity.** The grammar must be prefix-safe: `parse(render(c)) == c` for every command over the v1 concept sets. Any phrase collision that breaks this (for example a block phrase equal to an entity phrase after `go to the nearest`) is fixed by a phrase override, and a test enumerates commands to check it.

### 4.4 Python API

```python
from hivemind.lang import Language, Command, Sentence, Goal, EntityRef, ParseError

lang = Language.load()                    # data/vocab-1.21.1.json; Language.build(concepts) makes it
lang.words                                # list[str]; lang.word_id(w) -> int; lang.word(i) -> str
lang.hash, lang.words_hash                # provenance; network compatibility (section 4.1)
lang.phrase(kind, name) -> tuple[str, ...]           # kind: "block" | "item" | "entity" | "player" | ...
lang.concept(kind, words) -> str                     # inverse of phrase

# Command.args keys are exactly the slot names of section 4.5; frozen; args sorted by slot name
Command(verb="mine", args=(("block", "iron_ore"), ("mine_count", 4)))
Command(verb="attack", args=(("entity", EntityRef(2, "zombie")), ("template", "entity")))
Command(verb="chat", args=(("addressee", "everyone"), ("sentence", Sentence("need", item_count=3, item="iron_ingot"))))
Command(verb="plan", args=(("goal", Goal("get", item_count=3, item="iron_ingot")),))
Command.of("give", item_count=3, item="bread", player="red_7")   # fills "template" when the slots name one
EntityRef(ordinal=2, entity="zombie")      # entity: entity-class name, or ("dropped", item-class name)
# Sentence(intent, **slots) and Goal(kind, **slots) use the same slot names as commands

lang.render(cmd) -> str                   # "mine 4 iron ore."
lang.to_words(cmd) -> list[str]           # ["mine", "4", "iron", "ore", "."]
lang.encode(cmd) -> list[int]; lang.decode(ids) -> Command
lang.parse(text_or_words) -> Command      # strict canonical parse; raises ParseError

lang.normalize_human(text) -> list[str]   # lower-case, split, plurals to lemmas, number words to digits,
                                          # synonyms ("pick" -> "pickaxe", "workbench" -> "crafting table"),
                                          # unknown words -> "<unk>"
lang.parse_human(text, speaker) -> Command | Sentence | None   # lenient: drops filler words,
                                          # maps counts to the nearest bin, resolves "me"/"i" to the speaker

lang.result_words(code, cause=None, station=None, container=None) -> list[str]
lang.event_words(kind, **fields) -> list[str]   # kind: an event kind name of section 4.3
```

As built in M1 (`hivemind/lang/types.py`, `grammar.py`, `decoder.py`):
- `Command.args` holds exactly the template's slot names, plus a `template` arg (the template id: `"entity"`, `"player"`, `"wear"`, …) for the verbs with several templates: `goto`, `attack`, `flee` and `equip`. Some of their templates have the same slots (`go to my bed` and `go home` have none; `hold {item}` and `wear {item}` both have `item`).
- `chat` keeps its sentence under the arg `sentence` and `plan` its goal under `goal`.
- Players are concept names (`red_0`, `human`), never phrases.
- An `entity` slot holds an `EntityRef` in commands and an entity-class name in goals and event sentences; `slot_domain(verb, slot)` tells which.

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

Slot names used by `options`: `block`, `item`, `entity` (returns `EntityRef`s in commands, entity-class names inside a goal), `player`, `furnace`, `chest`, `dir8`, `dir4`, `explore_dist`, `strip_dist`, `y_band`, `mine_count`, `item_count`, `pattern`, `wait_seconds`, `addressee`, `intent` (which chat sentence; the two `where` sentences are told apart by their slot, `item` or `block`), `goal` (which goal template), plus `template` (which template of a verb; values are template ids). There is no `ordinal` slot. Values are concept names or ints exactly as they appear in `Command.args`. A chat sentence or goal is expanded inline: after `intent` or `goal` is chosen, its own slots are asked for with the same names, and they become the fields of the `Sentence` or `Goal`.

The network side (built in M5) ties output scores to the same word embeddings that read the observation, sums log-probabilities over the non-forced words, and uses the sum of per-step entropies along the sampled command (an unbiased estimate of the joint entropy).

## 5. Reference simulator (`hivemind.sim.ref`)

Readable Python, the oracle for the C++ core. Works at the class level (section 3) and speaks the language of section 4.

| Module | Provides |
|---|---|
| `rng.py` | `CounterRNG(seed)`: counter-based (SplitMix64-style) draws keyed by `(episode, tick, purpose, entity, draw)`; `u64`, `uniform`, `randint`, `choice`, `poisson`. `Purpose` enum of draw purposes |
| `mechanics.py` | Pure functions and small state classes for section 4.3 of DESIGN.md: `break_ticks(...)` returning `BreakTime`: the vanilla client's float32 count `ticks`; `design_ticks`, the formula of DESIGN.md 4.3 with the game's tool speeds, which is what Mineflayer waits once the bridge's tool-speed patch is applied (DESIGN.md 5) and what the simulator uses; `mineflayer_ticks`, unpatched Mineflayer's wait, kept for the patch's tests; and `server_accept_ticks`. Also `can_harvest`, `HungerState` (exhaustion, saturation, regeneration, starvation by difficulty), eating, clock helpers, attack cooldown and damage multiplier, critical hits, armor reduction, fall damage, smelting time, fuel |
| `inventory.py` | `Inventory` over item classes: 36 main slots, armor, off hand, held slot; stack-limited slot accounting; tool durability |
| `crafting.py` | Class-level `RecipeBook`: `craftable(inv, near_table)`, `craft(inv, item_class, up_to)`, `smeltable(inv, furnace)` (exactly the `smelt_start` mask), `smelt_start`, `collect`, and `Furnace` (the 1.21.1 furnace tick). A full inventory stops `craft` and `collect` partway with `INV_FULL` (`inventory full`); the furnace's own slots blocking `smelt_start` give `INV_FULL` with container `furnace` |
| `loot.py` | Vanilla loot-table sampler (the subset used by v1 blocks and mobs; no Fortune, no Silk Touch), returning item-class counts |
| `world.py` | `World`: uint8 block-class voxels (x, y, z) with y −64..191 (barrier beside and below the box, air above), block light, heightmap, a biome per column (draft 0.5; a 3-D biome array from M3), change log (`BlockChange`, `changes_since`), entity slot tables (`SlotTable`), clock; `ticking_chunks` by simulation distance. The chest slot table is sized from the agent count: 8 per agent, used from M9 [design] |
| `ore_table.py` | The overworld ore placement of 1.21.1 as a committed table generated from the pinned mcmeta worldgen files (`ORE_TABLE`, `OreFeature`, `OreTarget`, `height_pmf`); `python -m hivemind.sim.ref.ore_table --check` fails if it is stale |
| `worldgen.py` | Test worlds until M3: `generate(seed, config)`, `preset(name)`, `WorldGenConfig`, `spawn_point(world)`; value-noise terrain, caves, ores from `ore_table.py`, trees; biomes per column (draft 0.5), true of each column: consistent with its surface and with the ore rows placed there (`WorldGenConfig.biome` picks them), never an underground biome (section 3). Constants not from the game are labelled [est] |
| `perception.py` | `perceive(...) -> Perception` with the visible-only voxel window (`VoxelWindow`, S or M preset, `unseen` for cells not seen), resource `Sightings` of `Sighting`s, `RaySchedule`, `entity_visible`, the X-ray ablation; draft 0.5 adds the semantic panorama and the F3 fields (DESIGN.md 6.1, 6.2; section 6 here) |
| `nav.py` | Port of mineflayer-pathfinder 2.4.5's moves, costs and A* (DESIGN.md 4.5): `find_path(...) -> NavResult` (`Path` of `Step`s and `Dig`s, or `NO_PATH`/`PATH_TIMEOUT`); goals `GoalBlock`, `GoalNear`, `GoalXZ`, `GoalFlee` and the helpers `near_block`, `near_entity`, `flee_from`; `GotoModel`, the `T_goto` duration model with placeholder coefficients [est]; `bridge_block_lists`, the never-dig and avoid additions the bridge loads into mineflayer-pathfinder |
| `mobs.py` | Spawning (placeholder rates [est] until M3 measures them), chase, attacks, creeper fuse, sunlight burning, despawn, animals |
| `actions.py` | The 20 verb ids as interruptible macros with durations, masks (`SlotProvider`) and failure codes; `store`/`take` behind `containers`, `plan` behind `goals` |
| `resolve.py` | The one resolver (draft 0.5; owned with `actions.py`): parsed command + notebook + inventory (+ recipe book) → plan: which known positions in which order, which frontier point or staircase direction, which vanilla recipe and item variants from the items held. `actions.py`'s macros and the bridge conductor both call it, so the simulator and the real game pick the same targets and recipes (section 7; `bridge/PROTOCOL.md` 6.1–6.2). It reads only what the agent perceived, did or was told (the notebook, the visible entities and the inventory), never hidden world state |
| `goals.py` | The goal slots and the success tests of section 4.3 (needed only when `goals` is on; the slots exist, empty, otherwise) |
| `rewards.py` | Achievements (own production only), potential-based shaping, night bonus, death |
| `env.py` | `RefEnv`: reset, `act(agent, command or text)`, `step()` returning ready agents, results, rewards |
| `viewer.py` | Text and top-down ASCII views |

## 6. Observations (`hivemind.features`)

One observation builder, shared by the simulator and the bridge (DESIGN.md 6). Every symbolic field is given as words (token ids from section 4) plus numbers; see DESIGN.md 6.2 for the field list. It includes the three goal slots (own, given, team: DESIGN.md 6.6), which are always present and masked while `goals` is off.

Draft 0.5 changes (DESIGN.md 6.1, 6.2):
- **Voxel window:** preset S 15 × 15 × 7 (x and z ±7, y −3..+3) or M 21 × 21 × 9 (x and z ±10, y −4..+4), configurable. Only cells seen from the eye show their class; every other cell is class 255 `unseen`.
- **Semantic panorama** (preset M only; off in S, whose resource sightings come from the window alone): 96 × 24 pixels, world-aligned (column 0 starts at north, clockwise, 3.75° each; rows from pitch +30° to −60°, 3.75° each) [est]. Each pixel is the first non-air block cell (water, lava, leaves and glass count) or entity box its ray hits within 64 blocks [est], with fields: block class (`air` when nothing is hit; none when an entity is hit first), entity class or player (or none), distance/64, face hit (`top`, `bottom`, `north`, `south`, `east`, `west`; none, masked, when nothing is hit), and the sky and block light of the cell in front of that face (of the cell where the ray ends when nothing is hit). Any entity box the ray reaches before a block counts, out to 64 blocks; the entity table's `entity_visible` and its 32 blocks do not apply here. The eye's own cell is never a hit, and a ray that starts in water passes water cells until it first leaves the water [design].
- **F3 debug group** (both presets): x and z relative to home, y absolute; yaw and pitch in Minecraft's convention and the direction word (`north`, `east`, `south`, `west`); three targets from one look ray from the eye (DESIGN.md 6.2): the **targeted block**, the first cell that is neither air nor a fluid within 4.5 blocks (class, distance, face word), passing water but stopping at lava, which is not see-through (a stated deviation from F3); the **targeted fluid**, `water` or `lava` only when such a cell comes before the targeted block (else none); the **targeted entity**, the first entity box within 3.0 blocks and nearer than the targeted block's hit point or a lava cell (else none). 4.5 and 3.0 are the reaches [mem]; 4.5 is a design choice, since F3 itself picks blocks out to 20 [mem]. The group is computed from the world (simulator) or the world mirror (bridge) by the shared function, never read from the client's debug screen. Then: tick within the day and its word (`day` 0–11999, `dusk` 12000–12999, `night` 13000–22999, `dawn` 23000–23999); day number; ticks since the episode began; biome at the feet (a biome class, section 3); sky light and block light at the feet. Absolute x and z are never given.
- **Self** keeps health, food, saturation, air, XP and rain; y, the time of day and light moved into the F3 group.
- **Entities:** the entity table lists the visible ones only (`entity_visible`, within 32 blocks).
- **Honesty tests** (DESIGN.md 6.1) change cells that no straight segment from the eye reaches through see-through cells, only into other opaque classes that block light and emit none, and require the window, the panorama, the F3 group and the notebook to stay the same. A zombie behind a wall in the look direction, and separately a lava cell there, must leave the F3 group unchanged.

### 6.1 Memory (`hivemind/features/memory.py`)

Weights do not change during play, and a GRU is poor at exact recall over a 24,000-tick day. So each agent keeps an explicit **notebook**, written by code and read by the network. The GRU still carries short-term context.

- **Honest:** written only from what the agent itself perceived (section 5 `perception.py`), did, or was told in chat. Never from hidden state. Resource places come from seen window cells and, in preset M, panorama pixels up to 64 blocks away [est] (DESIGN.md 6.1).
- **Persistent:** survives death and respawn within an episode; cleared at episode reset.
- **Shared code:** the same `Notebook` runs in the simulator and the bridge.
- **Read as tokens:** every entry is a phrase (words) plus numbers, encoded by a small MLP and read by cross-attention from the trunk (DESIGN.md 6.3), so many entries stay cheap.

| Table | Capacity | Entry |
|---|---|---|
| Places | 48 | kind (`resource`, `station`, `danger`, `home`), phrase (`iron ore`, `my furnace 2`, `my chest 1`, `my bed`, `lava`), position, count seen, first and last seen tick. Eviction: the entry with the lowest value of a fixed score (resource tier, distance, age) [est] |
| Players | 40 | name (`blue 7`), relation from the visible team colour (`teammate`, `enemy`, `human`), last seen position and tick, last message words, counters: damage dealt to me, damage I dealt, kills each way, items given each way, times seen opening my chests |
| Containers | 8 own + 8 seen | phrase (`my chest 1`), position, contents (item class → count) as of the last time the agent looked, tick of that look; the agent's own store and take actions update it exactly |
| Events | 16, ring buffer | an event sentence (section 4.3) and its tick |

Events involving other players are attributed by the fixed, perception-only rules of DESIGN.md 6.3 ("What counts as seen"); the simulator knows the true actor but never writes it. `notebook.entries()` yields one token per place, player and event, and one per (chest, item class) held, up to 32 [est]: about 136 tokens.

API sketch: `Notebook(lang, config)`, `notebook.observe(perception_result, tick)`, `notebook.on_event(event)`, `notebook.on_container_view(chest, contents, tick)`, `notebook.entries() -> list[MemoryToken]` (words, numbers), `notebook.reset()`. The simulator's `goto`, `mine`, `store` and `take` masks read the notebook, not the true world.

## 7. Bridge protocol (`bridge/`)

As DESIGN.md 11.1, with the command in words: the conductor sends `act{agent, req, command: "mine 4 iron ore.", parsed: {verb, args}, plan, deadline_ticks, deadline_ms}`; Node replies `result{agent, req, code, result: "no path", tick0, tick1, wall_ms, n_roundtrips, inv_delta, raw_state}`.
- `command` is the text, for logs. `parsed` is `Command` as a map (`args` keyed by exactly the slot names of section 4.5), and Node uses only `parsed.verb`, to pick the wrapper.
- `plan` is what Node executes: concrete parameters (block positions in order, a frontier point, a vanilla recipe built from the items held) that the conductor gets from `resolve.py` (section 5), the same resolver the simulator's macros use. Node never parses English and never picks targets, so there is one parser and one resolver, both on the Python side, and targets come only from what the agent perceived.

The full protocol, including the hello handshake (protocol version, rules and vocabulary hashes), the state stream, the extra event kinds (`entity_dead`, `pickup`, `lid`, `respawn`) and the `plan` fields per verb, is `bridge/PROTOCOL.md`.

Draft 0.5 needs more of the real world on the Python side, because perception runs there (section 6): in preset M the world mirror must reach the panorama's 64 blocks from anywhere in the bot's chunk (a radius of 4 chunks, ceil(64 / 16), instead of 2 for draft 0.4's 24 blocks; preset S needs only the window's ±7 and the 4.5-block look ray), entities must be reported out to 64 blocks for the panorama (the entity table still lists the 16 nearest visible within 32), and the mirror must carry biomes (the game's 3-D 4 × 4 × 4 biome cells, plus the hashed world seed that F3's biome zoom uses [mem]) and the full day-time counter for the F3 group. The pathfinder settings must include `allowEntityDetection = false` and the lists of `nav.bridge_block_lists` (DESIGN.md 4.5, 5).

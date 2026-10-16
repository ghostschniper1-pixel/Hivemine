'use strict'
/**
 * Dig-time patch for Mineflayer on Java Edition 1.21.1 (DESIGN.md 5, "Tool speeds are patched at
 * startup"; README.md in this folder).
 *
 * prismarine-block's `Block.digTime` takes a tool's speed from minecraft-data's per-block
 * `material`: `registry.materials[block.material][heldItemId]`, else 1. In minecraft-data 3.117.0
 * (pc/1.21.1) every tier-gated block, exactly the 101 members of the block tag
 * `incorrect_for_wooden_tool`, has the material "incorrect_for_wooden_tool", which lists only the
 * four wooden tools (speed 2). Unpatched, Mineflayer therefore mines obsidian with a diamond
 * pickaxe in 1,500 ticks instead of 188 and iron ore with a stone pickaxe in 91 instead of 23, and
 * with a wooden axe, shovel or hoe it sends "finished" before the server accepts the break.
 *
 * In the 1.21.1 tags all 101 blocks are mineable with a pickaxe and with no other tool kind, so
 * giving them the material "mineable/pickaxe" (the speeds of the six pickaxes) makes digTime use
 * the game's tool-rule speeds. Harvest rules are untouched (minecraft-data's `harvestTools` are
 * already right). Afterwards Mineflayer's wait equals DESIGN.md 4.3's formula for every block,
 * tool and condition except moss carpet with a hoe (minecraft-data's material "plant" has no
 * hoes), which the patch leaves alone; test/patches.test.js checks both against the reference
 * mechanics.
 *
 * The patch must run before any Block object exists: a Block copies `material` from the
 * registry when it is constructed. It edits the block definitions in place, and those objects
 * belong to minecraft-data's per-version cache, which every prismarine-registry for 1.21.1 in the
 * process shares (each Mineflayer bot's `bot.registry` is a shallow copy of it). So one call at
 * host startup, before the first `mineflayer.createBot`, patches every bot that process creates.
 *
 * That holds only for the copy of prismarine-registry (and of minecraft-data under it) that
 * Mineflayer itself loads, so the host takes its registry from `mineflayerRegistry()`, which
 * resolves the package from Mineflayer's own folder, never by bare name: a lock that nested a
 * second copy under mineflayer would otherwise patch a cache no bot reads. Each bot must still
 * pass `assertDigTimePatched(bot.registry)` on 'inject_allowed' (README.md): a bot that mined at
 * unpatched speed would show no other sign of it.
 */

const path = require('node:path')

const GAME_VERSION = '1.21.1'
const TIER_GATED_MATERIAL = 'incorrect_for_wooden_tool'
const PATCHED_MATERIAL = 'mineable/pickaxe'
const EXPECTED_COUNT = 101 // DESIGN.md 5: "the 101 blocks of incorrect_for_wooden_tool"

/**
 * The block tag `incorrect_for_wooden_tool` of Java Edition 1.21.1, fully resolved: mcmeta
 * `1.21.1-data` [3] at the commit pinned in data/sources.json, as built into
 * data/rules-1.21.1.json (tags.block). Every member is in `mineable/pickaxe` and in no other
 * `mineable/*` tag there (checked by make_dig_fixture.py; test/patches.test.js checks this list
 * against the fixture that script writes).
 */
const TIER_GATED_PICKAXE_ONLY = Object.freeze([
  'ancient_debris', 'chiseled_copper', 'copper_block', 'copper_bulb', 'copper_door',
  'copper_grate', 'copper_ore', 'copper_trapdoor', 'crafter', 'crying_obsidian', 'cut_copper',
  'cut_copper_slab', 'cut_copper_stairs', 'deepslate_copper_ore', 'deepslate_diamond_ore',
  'deepslate_emerald_ore', 'deepslate_gold_ore', 'deepslate_iron_ore', 'deepslate_lapis_ore',
  'deepslate_redstone_ore', 'diamond_block', 'diamond_ore', 'emerald_block', 'emerald_ore',
  'exposed_chiseled_copper', 'exposed_copper', 'exposed_copper_bulb', 'exposed_copper_door',
  'exposed_copper_grate', 'exposed_copper_trapdoor', 'exposed_cut_copper',
  'exposed_cut_copper_slab', 'exposed_cut_copper_stairs', 'gold_block', 'gold_ore', 'iron_block',
  'iron_ore', 'lapis_block', 'lapis_ore', 'lightning_rod', 'netherite_block', 'obsidian',
  'oxidized_chiseled_copper', 'oxidized_copper', 'oxidized_copper_bulb', 'oxidized_copper_door',
  'oxidized_copper_grate', 'oxidized_copper_trapdoor', 'oxidized_cut_copper',
  'oxidized_cut_copper_slab', 'oxidized_cut_copper_stairs', 'raw_copper_block', 'raw_gold_block',
  'raw_iron_block', 'redstone_ore', 'respawn_anchor', 'waxed_chiseled_copper',
  'waxed_copper_block', 'waxed_copper_bulb', 'waxed_copper_door', 'waxed_copper_grate',
  'waxed_copper_trapdoor', 'waxed_cut_copper', 'waxed_cut_copper_slab', 'waxed_cut_copper_stairs',
  'waxed_exposed_chiseled_copper', 'waxed_exposed_copper', 'waxed_exposed_copper_bulb',
  'waxed_exposed_copper_door', 'waxed_exposed_copper_grate', 'waxed_exposed_copper_trapdoor',
  'waxed_exposed_cut_copper', 'waxed_exposed_cut_copper_slab', 'waxed_exposed_cut_copper_stairs',
  'waxed_oxidized_chiseled_copper', 'waxed_oxidized_copper', 'waxed_oxidized_copper_bulb',
  'waxed_oxidized_copper_door', 'waxed_oxidized_copper_grate', 'waxed_oxidized_copper_trapdoor',
  'waxed_oxidized_cut_copper', 'waxed_oxidized_cut_copper_slab',
  'waxed_oxidized_cut_copper_stairs', 'waxed_weathered_chiseled_copper', 'waxed_weathered_copper',
  'waxed_weathered_copper_bulb', 'waxed_weathered_copper_door', 'waxed_weathered_copper_grate',
  'waxed_weathered_copper_trapdoor', 'waxed_weathered_cut_copper',
  'waxed_weathered_cut_copper_slab', 'waxed_weathered_cut_copper_stairs',
  'weathered_chiseled_copper', 'weathered_copper', 'weathered_copper_bulb',
  'weathered_copper_door', 'weathered_copper_grate', 'weathered_copper_trapdoor',
  'weathered_cut_copper', 'weathered_cut_copper_slab', 'weathered_cut_copper_stairs'
])
const PICKAXE_ONLY = new Set(TIER_GATED_PICKAXE_ONLY)
if (PICKAXE_ONLY.size !== EXPECTED_COUNT) {
  throw new Error(`dig-time patch: the embedded tag has ${PICKAXE_ONLY.size} blocks, expected ${EXPECTED_COUNT}`)
}

class DigTimePatchError extends Error {
  constructor (message) {
    super(`dig-time patch: ${message}`)
    this.name = 'DigTimePatchError'
  }
}

/**
 * Give every block whose minecraft-data material is "incorrect_for_wooden_tool" the material
 * "mineable/pickaxe". Call it once, before any Block object is created (see the module comment).
 *
 * Throws a DigTimePatchError, changing nothing, if the registry is not for 1.21.1, if
 * "mineable/pickaxe" is missing or lists anything but pickaxes, if a block with that material is
 * not one of the 101 pickaxe-only blocks of the embedded 1.21.1 tag, or if one of those 101 has a
 * third material. Calling it again is harmless: blocks that already have "mineable/pickaxe"
 * count as already patched.
 *
 * @param {object} registry a prismarine-registry (or minecraft-data) object for 1.21.1
 * @param {{log?: (message: string) => void}} [options] where to log the count (console.log)
 * @returns {{version: string, patched: string[], alreadyPatched: string[]}} block names
 */
function applyDigTimePatch (registry, { log = console.log } = {}) {
  const version = registry?.version?.minecraftVersion
  if (version !== GAME_VERSION) {
    throw new DigTimePatchError(`the registry is for ${version}, but the embedded tag is for ${GAME_VERSION}`)
  }

  const pickaxeSpeeds = registry.materials?.[PATCHED_MATERIAL]
  const pickaxeIds = Object.keys(pickaxeSpeeds ?? {})
  if (pickaxeIds.length === 0) {
    throw new DigTimePatchError(`minecraft-data has no material "${PATCHED_MATERIAL}"`)
  }
  for (const id of pickaxeIds) {
    const name = registry.items?.[id]?.name
    if (!name?.endsWith('_pickaxe')) {
      throw new DigTimePatchError(`material "${PATCHED_MATERIAL}" lists item ${id} (${name}), not a pickaxe`)
    }
  }

  for (const block of registry.blocksArray) {
    if (block.material === TIER_GATED_MATERIAL && !PICKAXE_ONLY.has(block.name)) {
      throw new DigTimePatchError(`${block.name} has material "${TIER_GATED_MATERIAL}" but is not one of ` +
        `the ${EXPECTED_COUNT} pickaxe-only blocks of the ${GAME_VERSION} tag`)
    }
  }
  const toPatch = []
  const alreadyPatched = []
  for (const name of TIER_GATED_PICKAXE_ONLY) {
    const block = registry.blocksByName[name]
    if (block === undefined) throw new DigTimePatchError(`minecraft-data has no block ${name}`)
    if (block.material === TIER_GATED_MATERIAL) toPatch.push(block)
    else if (block.material === PATCHED_MATERIAL) alreadyPatched.push(name)
    else throw new DigTimePatchError(`${name} has material "${block.material}", expected "${TIER_GATED_MATERIAL}"`)
  }

  for (const block of toPatch) block.material = PATCHED_MATERIAL
  const patched = toPatch.map(b => b.name)
  log(`dig-time patch: ${patched.length} blocks with material "${TIER_GATED_MATERIAL}" now use ` +
    `"${PATCHED_MATERIAL}" (${alreadyPatched.length} already patched; expected ${EXPECTED_COUNT} in all)`)
  return { version, patched, alreadyPatched }
}

/**
 * Throw a DigTimePatchError unless every one of the 101 tier-gated blocks of this registry has the
 * patched material. Changes nothing. The host calls it for every bot on 'inject_allowed', with
 * `bot.registry` (the registry the bot's Blocks will read), and must not let a bot that fails it
 * play: it would mine the whole ore tech tree at unpatched speed with no other sign.
 *
 * @param {object} registry a bot's `bot.registry` (or any prismarine-registry for 1.21.1)
 */
function assertDigTimePatched (registry) {
  const version = registry?.version?.minecraftVersion
  if (version !== GAME_VERSION) {
    throw new DigTimePatchError(`the registry is for ${version}, but the patch is for ${GAME_VERSION}`)
  }
  const unpatched = TIER_GATED_PICKAXE_ONLY.filter(name => registry.blocksByName?.[name]?.material !== PATCHED_MATERIAL)
  if (unpatched.length > 0) {
    throw new DigTimePatchError(`${unpatched.length} of the ${EXPECTED_COUNT} tier-gated blocks of this registry ` +
      `are not patched (${unpatched.slice(0, 3).join(', ')}${unpatched.length > 3 ? ', ...' : ''}); ` +
      'call applyDigTimePatch(mineflayerRegistry()) before creating any bot')
  }
}

/**
 * The folder of a package as `fromDir` would resolve it (Node's lookup through node_modules).
 *
 * @param {string} name package name
 * @param {string} fromDir folder to resolve from
 * @returns {string}
 */
function packageDir (name, fromDir) {
  return path.dirname(require.resolve(`${name}/package.json`, { paths: [fromDir] }))
}

/**
 * A prismarine-registry for `version`, made by the copy of prismarine-registry that Mineflayer
 * itself loads (resolved from Mineflayer's folder), so it shares minecraft-data's block
 * definitions with every `bot.registry` of this process. Patch this one at host startup.
 *
 * @param {string} [version] game version (1.21.1)
 * @param {string} [fromDir] folder from which to resolve mineflayer (this module's folder)
 * @returns {object}
 */
function mineflayerRegistry (version = GAME_VERSION, fromDir = __dirname) {
  const mineflayerDir = packageDir('mineflayer', fromDir)
  return require(require.resolve('prismarine-registry', { paths: [mineflayerDir] }))(version)
}

module.exports = {
  applyDigTimePatch,
  assertDigTimePatched,
  mineflayerRegistry,
  packageDir,
  DigTimePatchError,
  GAME_VERSION,
  TIER_GATED_MATERIAL,
  PATCHED_MATERIAL,
  EXPECTED_COUNT,
  TIER_GATED_PICKAXE_ONLY
}

'use strict'
// Tests for ../patches.js against test/dig_times.json, which make_dig_fixture.py writes from the
// reference mechanics (hivemind.sim.ref.mechanics). Run from bridge/: npm test (Node >= 22).
//
// Everything is measured at load time, in this order, because the patch edits minecraft-data's
// shared block definitions: one Block made before the patch, the unpatched digTime of every
// fixture pair, the patch, then the patched digTime of every pair. The tests only compare.

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const patches = require('../patches')

const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, 'dig_times.json'), 'utf8'))
const registry = require('prismarine-registry')(fixture.version)
const Block = require('prismarine-block')(registry)

const blockNames = Object.keys(fixture.blocks)
const heldIds = fixture.held.map(name => (name === null ? null : registry.itemsByName[name].id))

function makeBlock (reg, BlockClass, name) {
  return BlockClass.fromStateId(reg.blocksByName[name].defaultState, 0)
}

// prismarine-block 1.23.0: digTime(heldItemType, creative, inWater, notOnGround, enchantments,
// effects) in ms; 0 = instant, Infinity = never. In ticks, with null for never.
function digTicks (block, heldId, cond) {
  const ms = block.digTime(heldId, false, cond.in_water, !cond.on_ground, [], {})
  return ms === Infinity ? null : ms / 50
}

// {block: {condition: [ticks per held item]}}, the shape of the fixture's rows
function measureAll () {
  const out = {}
  for (const name of blockNames) {
    const block = makeBlock(registry, Block, name)
    out[name] = {}
    for (const cond of fixture.conditions) out[name][cond.name] = heldIds.map(id => digTicks(block, id, cond))
  }
  return out
}

const GROUND = fixture.conditions.find(c => c.name === 'ground')
const id = name => registry.itemsByName[name].id

const obsidianMadeBeforePatch = makeBlock(registry, Block, 'obsidian')
const unpatched = measureAll()
const patchLog = []
const patchResult = patches.applyDigTimePatch(registry, { log: msg => patchLog.push(msg) })
const patched = measureAll()

function expectedUnpatched (name) {
  return fixture.blocks[name].unpatched ?? fixture.blocks[name].design
}

// Compare a measured table with the expected one; returns readable mismatches (empty when equal).
function mismatches (measured, expectedFor) {
  const out = []
  for (const name of blockNames) {
    for (const cond of fixture.conditions) {
      const want = expectedFor(name, cond.name)
      measured[name][cond.name].forEach((got, i) => {
        if (got !== want[i]) out.push(`${name} with ${fixture.held[i] ?? 'hand'} (${cond.name}): ${got}, expected ${want[i]}`)
      })
    }
  }
  return out
}

test('the embedded block list is the 1.21.1 tag incorrect_for_wooden_tool', () => {
  assert.equal(fixture.tier_gated_tag, patches.TIER_GATED_MATERIAL)
  assert.deepEqual([...patches.TIER_GATED_PICKAXE_ONLY].sort(), fixture.tier_gated)
  assert.equal(fixture.tier_gated.length, patches.EXPECTED_COUNT)
  assert.equal(patches.EXPECTED_COUNT, 101)
  for (const name of fixture.tier_gated) assert.equal(fixture.blocks[name].group, 'tier_gated', name)
})

test('the locked packages are the ones the fixture describes', () => {
  const version = pkg => require(`${pkg}/package.json`).version
  assert.equal(version('prismarine-block'), '1.23.0')
  assert.equal(version('prismarine-registry'), '1.12.0')
  assert.equal(version('minecraft-data'), '3.117.0')
  assert.equal(registry.version.minecraftVersion, '1.21.1')
})

test('unpatched digTime is wrong on tier-gated blocks', () => {
  const dp = fixture.held.indexOf('diamond_pickaxe')
  const sp = fixture.held.indexOf('stone_pickaxe')
  assert.equal(unpatched.obsidian.ground[dp], 1500)
  assert.equal(fixture.blocks.obsidian.design.ground[dp], 188)
  assert.equal(unpatched.iron_ore.ground[sp], 91)
  assert.equal(fixture.blocks.iron_ore.design.ground[sp], 23)
  for (const name of fixture.tier_gated) {
    const wrong = fixture.conditions.some(c => unpatched[name][c.name].some((t, i) => t !== fixture.blocks[name].design[c.name][i]))
    assert.ok(wrong, `${name}: unpatched digTime already equals the design formula`)
  }
})

test('unpatched digTime matches the reference model of unpatched Mineflayer for every pair', () => {
  // mechanics.mineflayer_tool_speed: minecraft-data's material speeds, which the patch replaces
  assert.deepEqual(mismatches(unpatched, (name, cond) => expectedUnpatched(name)[cond]), [])
})

test('the patch gives exactly the 101 tier-gated blocks the pickaxe material and logs the count', () => {
  assert.equal(patchResult.version, '1.21.1')
  assert.deepEqual([...patchResult.patched].sort(), fixture.tier_gated)
  assert.deepEqual(patchResult.alreadyPatched, [])
  assert.equal(patchLog.length, 1)
  assert.match(patchLog[0], /^dig-time patch: 101 blocks with material "incorrect_for_wooden_tool" now use "mineable\/pickaxe"/)
  assert.deepEqual(registry.blocksArray.filter(b => b.material === patches.TIER_GATED_MATERIAL), [])
  for (const name of fixture.tier_gated) assert.equal(registry.blocksByName[name].material, 'mineable/pickaxe')
})

test('patched digTime equals DESIGN.md 4.3 for every pair but the documented moss carpet + hoe', () => {
  // The fixture's exceptions are what patched Mineflayer still waits; the generator allows them
  // only for moss carpet with a hoe (DESIGN.md 5).
  assert.ok(fixture.exceptions.length > 0)
  const exception = new Map()
  for (const e of fixture.exceptions) {
    assert.equal(e.block, 'moss_carpet')
    assert.match(e.held, /_hoe$/)
    assert.notEqual(e.patched, e.design)
    exception.set(`${e.block}|${e.held}|${e.condition}`, e.patched)
  }
  const expected = (name, cond) => fixture.blocks[name].design[cond].map((t, i) => {
    const key = `${name}|${fixture.held[i]}|${cond}`
    return exception.has(key) ? exception.get(key) : t
  })
  assert.deepEqual(mismatches(patched, expected), [])
  const wh = fixture.held.indexOf('wooden_hoe')
  assert.deepEqual([fixture.blocks.moss_carpet.design.ground[wh], patched.moss_carpet.ground[wh]], [2, 3])
})

test('the patch changes nothing outside the tier-gated blocks', () => {
  for (const name of blockNames) {
    if (fixture.blocks[name].group !== 'tier_gated') assert.deepEqual(patched[name], unpatched[name], name)
  }
})

test('harvest rules are untouched: a wooden pickaxe still gets nothing from iron ore', () => {
  const ironOre = makeBlock(registry, Block, 'iron_ore')
  assert.equal(Boolean(ironOre.canHarvest(id('wooden_pickaxe'))), false)
  assert.equal(Boolean(ironOre.canHarvest(id('stone_pickaxe'))), true)
  assert.equal(digTicks(ironOre, id('wooden_pickaxe'), GROUND), 150) // 2 / 3 / 100: the /100 divisor
})

test('a Block made before the patch keeps the old material: the host must patch first', () => {
  assert.equal(obsidianMadeBeforePatch.material, 'incorrect_for_wooden_tool')
  assert.equal(digTicks(obsidianMadeBeforePatch, id('diamond_pickaxe'), GROUND), 1500)
  assert.equal(digTicks(makeBlock(registry, Block, 'obsidian'), id('diamond_pickaxe'), GROUND), 188)
})

test('a registry made after the patch, as each bot makes its own, shares the patched definitions', () => {
  const botRegistry = require('prismarine-registry')('1.21.1')
  assert.notEqual(botRegistry, registry)
  const BotBlock = require('prismarine-block')(botRegistry)
  assert.equal(botRegistry.blocksByName.obsidian.material, 'mineable/pickaxe')
  assert.equal(digTicks(makeBlock(botRegistry, BotBlock, 'obsidian'), id('diamond_pickaxe'), GROUND), 188)
  const again = patches.applyDigTimePatch(botRegistry, { log: () => {} })
  assert.deepEqual([again.patched.length, again.alreadyPatched.length], [0, 101]) // idempotent
})

// A minimal stand-in registry for the refusal tests; never touches the real one.
function fakeRegistry ({ version = '1.21.1', extraBlock = null, pickaxeMaterial = { 1: 2.0 } } = {}) {
  const blocksArray = patches.TIER_GATED_PICKAXE_ONLY.map(name => ({ name, material: 'incorrect_for_wooden_tool' }))
  if (extraBlock) blocksArray.push(extraBlock)
  return {
    version: { minecraftVersion: version },
    materials: { 'mineable/pickaxe': pickaxeMaterial },
    items: { 1: { name: 'wooden_pickaxe' }, 2: { name: 'wooden_axe' } },
    blocksArray,
    blocksByName: Object.fromEntries(blocksArray.map(b => [b.name, b]))
  }
}

test('the patch refuses data it was not written for, and then changes nothing', () => {
  const quiet = { log: () => {} }
  assert.throws(() => patches.applyDigTimePatch(fakeRegistry({ version: '1.20.4' }), quiet), patches.DigTimePatchError)
  assert.throws(() => patches.applyDigTimePatch(fakeRegistry({ pickaxeMaterial: {} }), quiet), /no material/)
  assert.throws(() => patches.applyDigTimePatch(fakeRegistry({ pickaxeMaterial: { 2: 2.0 } }), quiet), /not a pickaxe/)
  const extra = fakeRegistry({ extraBlock: { name: 'stone', material: 'incorrect_for_wooden_tool' } })
  assert.throws(() => patches.applyDigTimePatch(extra, quiet), /stone has material .* not one of the 101 pickaxe-only/)
  assert.ok(extra.blocksArray.every(b => b.material === 'incorrect_for_wooden_tool'))
  const odd = fakeRegistry()
  odd.blocksByName.obsidian.material = 'default'
  assert.throws(() => patches.applyDigTimePatch(odd, quiet), /obsidian has material "default"/)
  assert.equal(odd.blocksByName.iron_ore.material, 'incorrect_for_wooden_tool')
  const ok = fakeRegistry()
  assert.equal(patches.applyDigTimePatch(ok, quiet).patched.length, 101)
})

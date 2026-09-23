import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { buildModels, computeCodIndex, findBridgebench, matchModelToBenchmarks, parseBenchmarks, parseHtml, parsePrice } from '../server/utils/pricing'
import { loadSource } from '../server/utils/source-cache'
import type { CacheStore } from '../server/utils/source-cache'

const fixture = JSON.parse(await readFile(new URL('./fixtures/matching.json', import.meta.url), 'utf8'))
for (const [name, expected] of fixture.cases) {
  test(`model identity: ${name}`, () => {
    assert.equal(matchModelToBenchmarks(name, fixture.benchmarks)?.source_model ?? null, expected)
  })
}

test('pricing table is selected by headers, accepts attributes and decodes entities', async () => {
  const html = await readFile(new URL('./fixtures/pricing.html', import.meta.url), 'utf8')
  const models = parseHtml(html)!
  assert.equal(models.length, 3)
  assert.deepEqual(models[0], { name: 'GPT 6 Sol (≤ 272K tokens)', input_price: '$2.00', output_price: '$10.00', cached_read: '$0.20', cached_write: '-' })
  assert.equal(models[1].name, 'Example Free')
  assert.equal(models[2].name, 'A & B (> 200K tokens)')
  assert.equal(parseHtml('<table><tr><td>Error</td></tr></table>'), null)
  const priced = buildModels(models, fixture.benchmarks)
  assert.ok(Math.abs(priced[0].weighted_price! - 2.1728) < 1e-9)
  assert.equal(priced[1].weighted_price, 0)
  assert.equal(priced[2].weighted_price, null)
  assert.equal(JSON.parse(JSON.stringify(priced))[2].weighted_price, null)
})

test('prices, raw API zero scores, and missing evaluations', () => {
  assert.equal(parsePrice(' Free '), 0)
  assert.equal(parsePrice('$1,234.50'), 1234.5)
  assert.equal(parsePrice('$broken'), Infinity)
  assert.equal(parsePrice('$2.0foo'), Infinity)
  assert.equal(computeCodIndex(48.1), 100)
  assert.ok(computeCodIndex(70)! > 100)
  const data = parseBenchmarks({ data: [{ name: 'Test', evaluations: { gpqa: 0, artificial_analysis_coding_index: 0 } }, { name: 'Empty' }] })
  assert.equal(data.Test.gpqa, 0)
  assert.equal(data.Test.coding, 0)
  assert.equal(data.Empty.coding, null)
  assert.throws(() => parseBenchmarks({ error: 'Unauthorized' }))
  assert.throws(() => parseBenchmarks({ data: [] }))
})

test('BridgeBench snapshot keeps versions and variants separate', () => {
  assert.equal(findBridgebench('GPT 6 Sol (> 272K tokens)'), 643)
  assert.equal(findBridgebench('GLM 5.3 Flash'), 450)
  assert.equal(findBridgebench('Claude Opus 4.6'), null)
  assert.equal(findBridgebench('Qwen3.8 Flash'), null)
})

test('cache expires, refreshes, and preserves original timestamp on failure', async () => {
  const values = new Map<string, unknown>()
  const store: CacheStore = {
    getItem: async <T>(key: string) => values.get(key) as T ?? null,
    setItem: async (key, value) => { values.set(key, value) },
  }
  let calls = 0
  const fetcher = async () => { calls++; return [calls] }
  const first = await loadSource(store, 'prices', 10000, fetcher)
  assert.equal(first?.from_cache, false)
  assert.equal((await loadSource(store, 'prices', 10000, fetcher))?.from_cache, true)
  assert.equal(calls, 1)
  values.set('prices', { fetched_at: Date.now() - 20000, value: [1] })
  assert.deepEqual((await loadSource(store, 'prices', 10000, fetcher))?.value, [2])
  const timestamp = Date.now() - 20000
  values.set('prices', { fetched_at: timestamp, value: [2] })
  const fallback = await loadSource(store, 'prices', 10000, async () => { throw Error('offline') })
  assert.equal(fallback?.stale, true)
  assert.equal(fallback?.fetched_at, timestamp)
  assert.deepEqual(fallback?.value, [2])
  assert.equal((await loadSource(store, 'prices', 10000, null))?.stale, true)
  assert.equal(await loadSource(store, 'missing', 10000, async () => { throw Error('offline') }), null)
})

test('unwritable cache still returns fresh data', async () => {
  const store: CacheStore = {
    getItem: async () => null,
    setItem: async () => { throw Error('read only disk') },
  }
  assert.deepEqual((await loadSource(store, 'prices', 10000, async () => [1]))?.value, [1])
})

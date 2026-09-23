import {
  BENCHMARK_TTL, BENCHMARK_URL, PRICING_URL, bridgebench,
  buildModels, parseBenchmarks, parseHtml,
} from '../utils/pricing'
import { loadSource } from '../utils/source-cache'

async function fetchResponse(url: string, headers?: Record<string, string>) {
  const response = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0', ...headers },
    signal: AbortSignal.timeout(15000),
  })
  if (!response.ok) throw new Error(`Source returned HTTP ${response.status}`)
  return response
}

async function loadModels(apiKey: string) {
  const storage = useStorage('cache')
  const [prices, benchmarks] = await Promise.all([
    loadSource(storage, 'zen:prices:v2', 60 * 60 * 1000, async () => {
      const html = await (await fetchResponse(PRICING_URL)).text()
      const models = parseHtml(html)
      if (!models?.length) throw new Error('Pricing table not found')
      return models
    }),
    loadSource(storage, 'zen:benchmarks:v2', BENCHMARK_TTL, apiKey ? async () => {
      const response = await fetchResponse(BENCHMARK_URL, { 'x-api-key': apiKey })
      return parseBenchmarks(await response.json())
    } : null),
  ])
  if (!prices) {
    throw createError({ statusCode: 503, statusMessage: 'Model pricing unavailable', message: 'Не удалось загрузить цены, сохранённый кэш отсутствует.' })
  }
  const models = buildModels(prices.value, benchmarks?.value ?? null)
  return {
    models,
    has_benchmarks: models.some(model => model.coding !== null || model.gpqa !== null),
    last_update: new Date(prices.fetched_at).toISOString(),
    from_cache: prices.from_cache,
    stale_prices: prices.stale,
    benchmarks_updated_at: benchmarks ? new Date(benchmarks.fetched_at).toISOString() : null,
    benchmarks_stale: benchmarks?.stale ?? false,
    bridgebench_updated_at: bridgebench.checked_at,
    bridgebench_source: bridgebench.source,
  }
}

// Одновременные открытия страницы используют один запрос к источникам.
let pending: ReturnType<typeof loadModels> | null = null

export default defineEventHandler(async (event) => {
  const config = useRuntimeConfig(event)
  const apiKey = config.artificialAnalysisApi || process.env.ARTIFICIAL_ANALYSIS_API || process.env.ARTIFICICAL_ANALYSIS_API || ''
  if (!pending) pending = loadModels(apiKey).finally(() => { pending = null })
  return pending
})

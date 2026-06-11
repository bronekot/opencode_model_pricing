import { URL } from 'node:url'

// Константы для бенчмарков
const COD_INDEX_MIN = 10.7
const COD_INDEX_MAX = 48.1
const COD_INDEX_MIN_VAL = 10
const COD_INDEX_MAX_VAL = 100

const SKIP_BENCHMARK_MODELS = new Set(['big pickle'])

const BRIDGEBENCH_DATA: Record<string, number> = {
  'Claude Sonnet 4.6': 94.9,
  'Claude Opus 4.6': 94.8,
  'GPT-5.3 Codex': 94.6,
  'Qwen3.5 Plus 02-15': 93.6,
  'GPT-5.2 Codex': 92.8,
  'MiniMax M2.5': 92.3,
  'Qwen3.5 397B A17B': 92.1,
  'GLM-5': 89.5,
  'Kimi K2.5': 89.1,
  'Gemini 3.1 Pro Preview': 75.6,
  'Aurora Alpha': 57.6,
}

const MODEL_MAP: Record<string, string | null> = {
  'big pickle': null,
  'minimax m2.5 free': 'MiniMax-M2.5',
  'minimax m2.5': 'MiniMax-M2.5',
  'minimax m2.1': 'MiniMax-M2.1',
  'glm 5 free': 'GLM-5 (Reasoning)',
  'glm 5': 'GLM-5 (Reasoning)',
  'glm 4.7': 'GLM-4.7 (Reasoning)',
  'glm 4.6': 'GLM-4.6 (Reasoning)',
  'kimi k2.5 free': 'Kimi K2.5 (Reasoning)',
  'kimi k2.5': 'Kimi K2.5 (Reasoning)',
  'kimi k2 thinking': 'Kimi K2 Thinking',
  'kimi k2': 'Kimi K2 0905',
  'qwen3 coder 480b': 'Qwen3 Coder 480B A35B Instruct',
  'claude opus 4.6': 'Claude Opus 4.6 (Adaptive Reasoning, Max Effort)',
  'claude opus 4.5': 'Claude Opus 4.5 (Reasoning)',
  'claude opus 4.1': 'Claude 4.1 Opus (Reasoning)',
  'claude sonnet 4.6': 'Claude Sonnet 4.6 (Adaptive Reasoning, Max Effort)',
  'claude sonnet 4.5': 'Claude 4.5 Sonnet (Reasoning)',
  'claude sonnet 4': 'Claude 4 Sonnet (Reasoning)',
  'claude haiku 4.5': 'Claude 4.5 Haiku (Reasoning)',
  'claude haiku 3.5': 'Claude 3.5 Haiku',
  'gemini 3.1 pro': 'Gemini 3.1 Pro Preview',
  'gemini 3 pro': 'Gemini 3 Pro Preview (high)',
  'gemini 3 flash': 'Gemini 3 Flash Preview (Reasoning)',
  'gpt 5.2': 'GPT-5.2 (medium)',
  'gpt 5.2 codex': 'GPT-5.2 Codex (xhigh)',
  'gpt 5.1': 'GPT-5.1 (high)',
  'gpt 5.1 codex': 'GPT-5.1 Codex (high)',
  'gpt 5.1 codex max': 'GPT-5.1 Codex (high)',
  'gpt 5.1 codex mini': 'GPT-5.1 Codex mini (high)',
  'gpt 5': 'GPT-5 (high)',
  'gpt 5 codex': 'GPT-5 Codex (high)',
  'gpt 5 nano': 'GPT-5 nano (high)',
}

interface Benchmark {
  creator?: string
  gpqa: number | null
  coding: number | null
  bridgebench?: number
  _null_date?: string
}

interface Benchmarks {
  [key: string]: Benchmark
}

interface ParsedModel {
  name: string
  input_price: string
  output_price: string
  cached_read?: string
  cached_write?: string
}

interface Model {
  name: string
  input_price: string
  output_price: string
  weighted_price: number
  gpqa: number | null
  coding: number | null
  cod_index: number | null
  bridgebench: number | null
}

// Helper functions
function normalizeName(name: string): string {
  let result = name.toLowerCase().trim()
  result = result.replace(/\(.*?\)/g, '')
  result = result.replace(/[<>]=?\s*\d+k/g, '')
  result = result.replace(/\s+/g, ' ').trim()
  result = result.replace(/-/g, ' ').replace(/_/g, ' ')
  return result
}

function parsePrice(priceStr: string): number {
  if (priceStr === 'Free') return 0
  if (priceStr.startsWith('$')) {
    return parseFloat(priceStr.replace('$', '').replace(',', ''))
  }
  if (priceStr === '-' || !priceStr.trim()) return Infinity
  return Infinity
}

function computeCodIndex(coding: number | null): number | null {
  if (coding === null) return null
  const t = (coding - COD_INDEX_MIN) / (COD_INDEX_MAX - COD_INDEX_MIN)
  const tClamped = Math.max(t, 0)
  return Math.round((COD_INDEX_MIN_VAL + (COD_INDEX_MAX_VAL - COD_INDEX_MIN_VAL) * tClamped) * 10) / 10
}

async function fetchHtml(): Promise<string | null> {
  try {
    const response = await fetch('https://opencode.ai/docs/zen', {
      headers: { 'User-Agent': 'Mozilla/5.0' },
      signal: AbortSignal.timeout(10000)
    })
    return await response.text()
  } catch {
    return null
  }
}

function parseHtml(html: string): ParsedModel[] | null {
  const tableMatch = html.match(/<table>(.*?)<\/table>/s)
  if (!tableMatch) return null

  const tables = html.split(/<table>|<\/table>/).filter(s => s.includes('<tr>'))
  if (tables.length < 2) return null

  const tableHtml = tables[1]
  const rows = tableHtml.match(/<tr>(.*?)<\/tr>/gs) || []

  const models: ParsedModel[] = []

  for (const row of rows) {
    if (row.includes('<th>') || row.includes('<thead')) continue

    const cells = row.match(/<td[^>]*>(.*?)<\/td>/gs) || []
    if (cells.length < 3) continue

    const cleanCells = cells.map(cell =>
      cell.replace(/<[^>]+>/g, '').trim()
    )

    if (cleanCells[0]) {
      models.push({
        name: cleanCells[0],
        input_price: cleanCells[1],
        output_price: cleanCells[2],
        cached_read: cleanCells[3],
        cached_write: cleanCells[4]
      })
    }
  }

  return models
}

async function fetchBenchmarks(apiKey: string): Promise<Benchmarks | null> {
  try {
    const response = await fetch('https://artificialanalysis.ai/api/v2/data/llms/models', {
      headers: {
        'x-api-key': apiKey,
        'User-Agent': 'Mozilla/5.0'
      },
      signal: AbortSignal.timeout(15000)
    })

    const data = await response.json()
    const benchmarks: Benchmarks = {}

    for (const model of data?.data || []) {
      const name = model.name
      const creator = model.model_creator?.name
      const evals = model.evaluations || {}

      benchmarks[name] = {
        creator,
        gpqa: evals.gpqa || null,
        coding: evals.artificial_analysis_coding_index || null
      }
    }

    return benchmarks
  } catch {
    return null
  }
}

function matchModelToBenchmarks(modelName: string, benchmarks: Benchmarks): Benchmark | null {
  const normName = normalizeName(modelName)

  const mappedName = MODEL_MAP[normName]
  if (mappedName === null && normName in MODEL_MAP) return null
  if (mappedName && mappedName in benchmarks) return benchmarks[mappedName]

  let bestMatch: string | null = null
  let bestScore = 0

  for (const apiName of Object.keys(benchmarks)) {
    const normApi = normalizeName(apiName)
    const wordsModel = new Set(normName.split(' '))
    const wordsApi = new Set(normApi.split(' '))

    const common = [...wordsModel].filter(w => wordsApi.has(w)).length
    const total = Math.max(wordsModel.size, 1)
    const score = common / total

    if (score > bestScore && score >= 0.6) {
      bestScore = score
      bestMatch = apiName
    }
  }

  return bestMatch ? benchmarks[bestMatch] : null
}

function findBridgebench(name: string): number | null {
  const normName = normalizeName(name)

  for (const [bbName, score] of Object.entries(BRIDGEBENCH_DATA)) {
    const normBb = normalizeName(bbName)

    if (normName === normBb) return score

    const bbWords = new Set(normBb.split(' '))
    const nameWords = new Set(normName.split(' '))

    const bbHasCodex = bbWords.has('codex')
    const nameHasCodex = nameWords.has('codex')
    if (bbHasCodex !== nameHasCodex) continue

    const excludeWords = new Set(['codex', 'pro', 'preview', 'plus'])
    const bbKeyWords = new Set([...bbWords].filter(w => !excludeWords.has(w)))

    if (bbKeyWords.size > 0 && [...bbKeyWords].every(w => nameWords.has(w))) {
      const bbNums = new Set([...bbWords].filter(w => /\d/.test(w)))
      const nameNums = new Set([...nameWords].filter(w => /\d/.test(w)))

      if (setsEqual(bbNums, nameNums) || bbNums.size === 0) {
        return score
      }
    }
  }

  return null
}

function setsEqual<T>(a: Set<T>, b: Set<T>): boolean {
  if (a.size !== b.size) return false
  for (const item of a) {
    if (!b.has(item)) return false
  }
  return true
}

function getToday(): string {
  return new Date().toISOString().split('T')[0]
}

export default defineEventHandler(async (event) => {
  // Получаем API ключ из runtime config или process.env
  const config = useRuntimeConfig()
  const apiKey = config.artificialAnalysisApi as string || process.env.ARTIFICIAL_ANALYSIS_API || ''

  console.log('[API] API Key present:', !!apiKey, apiKey ? apiKey.substring(0, 10) + '...' : '', 'runtimeConfig:', !!(config.artificialAnalysisApi as string))

  // Fetch HTML data
  const html = await fetchHtml()
  let parsedModels: ParsedModel[] | null = null
  let fromCache = false

  if (html) {
    parsedModels = parseHtml(html)
  }

  if (!parsedModels) {
    // Try cache as fallback
    const cacheFile = await useStorage('assets').getItem('model_pricing_cache')
    if (typeof cacheFile === 'string') {
      const lines = cacheFile.split('\n')
      parsedModels = []
      for (const line of lines) {
        if (line.startsWith('#') || line.startsWith('Model') || !line.trim()) continue
        const parts = line.split('\t')
        if (parts.length >= 3) {
          parsedModels.push({
            name: parts[0],
            input_price: parts[1],
            output_price: parts[2],
            cached_read: parts[3],
            cached_write: parts[4]
          })
        }
      }
      fromCache = true
    }
  }

  if (!parsedModels || parsedModels.length === 0) {
    throw createError({
      statusCode: 503,
      statusMessage: 'Не удалось загрузить данные моделей'
    })
  }

  // Fetch benchmarks if API key available
  let benchmarks: Benchmarks | null = null
  let hasBenchmarks = false

  if (apiKey) {
    benchmarks = await fetchBenchmarks(apiKey)
    hasBenchmarks = benchmarks !== null
  }

  // Process models
  const models: Model[] = []
  const modelNames = parsedModels.map(m => m.name)

  for (const parsed of parsedModels) {
    const inputVal = parsePrice(parsed.input_price)
    const outputVal = parsePrice(parsed.output_price)
    const weightedPrice = (inputVal !== Infinity && outputVal !== Infinity)
      ? 0.9784 * inputVal + 0.0216 * outputVal
      : Infinity

    let bench: Benchmark | null = null
    if (benchmarks) {
      bench = benchmarks[parsed.name] || matchModelToBenchmarks(parsed.name, benchmarks)
    }

    const coding = bench?.coding ?? null
    const gpqa = bench?.gpqa ?? null

    models.push({
      name: parsed.name,
      input_price: parsed.input_price,
      output_price: parsed.output_price,
      weighted_price: weightedPrice,
      gpqa,
      coding,
      cod_index: computeCodIndex(coding),
      bridgebench: findBridgebench(parsed.name)
    })
  }

  return {
    models,
    has_benchmarks: hasBenchmarks,
    last_update: new Date().toLocaleString('ru-RU'),
    from_cache: fromCache
  }
})

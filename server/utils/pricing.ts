import bridgebench from '../../data/bridgebench.json' with { type: 'json' }
import aliases from '../../data/model_aliases.json' with { type: 'json' }

export { bridgebench }
export const PRICING_URL = 'https://opencode.ai/docs/zen/'
export const BENCHMARK_URL = 'https://artificialanalysis.ai/api/v2/data/llms/models'
export const BENCHMARK_TTL = 24 * 60 * 60 * 1000

export interface Benchmark {
  creator?: string
  gpqa: number | null
  coding: number | null
  source_model?: string
}
export type Benchmarks = Record<string, Benchmark>
export interface ParsedModel {
  name: string
  input_price: string
  output_price: string
  cached_read?: string
  cached_write?: string
}

function decodeHtml(text: string): string {
  const entities: Record<string, string> = { amp: '&', lt: '<', gt: '>', le: '≤', ge: '≥', nbsp: ' ', quot: '"', apos: "'", dollar: '$' }
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (original, entity: string) => {
    if (entity.startsWith('#')) {
      const value = entity[1].toLowerCase() === 'x' ? parseInt(entity.slice(2), 16) : parseInt(entity.slice(1), 10)
      return value > 0 && value <= 0x10ffff ? String.fromCodePoint(value) : original
    }
    return entities[entity] ?? original
  })
}

export function normalizeName(name: string): string {
  return decodeHtml(name).toLowerCase().trim()
    .replace(/\(([^()]*)\)/g, (match, value: string) =>
      /\b(?:effort|reasoning|tokens)\b|^(?:max|xhigh|high|medium|low|minimal)$/.test(value)
      || /^[<>≤≥]=?\s*\d+k(?:\s+tokens)?$/.test(value) ? ' ' : match)
    .replace(/\bfree\s*$/, '').replace(/[-_]/g, ' ').replace(/\s+/g, ' ').trim()
}

export function modelIdentity(name: string, useAliases = true): string {
  const normalized = normalizeName(name)
  const canonical = useAliases ? (aliases as Record<string, string>)[normalized] ?? normalized : normalized
  return canonical.split(' ').sort().join(' ')
}

export function effortRank(name: string): number {
  const suffix = [...name.toLowerCase().matchAll(/\(([^()]*)\)/g)].map(match => match[1]).join(' ')
  const efforts = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max']
  for (let rank = efforts.length - 1; rank >= 0; rank--) {
    if (new RegExp(`\\b${efforts[rank]}\\b`).test(suffix)) return rank
  }
  if (suffix.includes('non-reasoning')) return -1
  return suffix.includes('reasoning') ? 2 : 0
}

export function hasUsefulBenchmark(benchmark: Benchmark): boolean {
  return benchmark.gpqa !== null || benchmark.coding !== null
}

export function matchModelToBenchmarks(name: string, benchmarks: Benchmarks): Benchmark | null {
  if (normalizeName(name) === 'big pickle') return null
  const identity = modelIdentity(name)
  const candidates = Object.entries(benchmarks).filter(([key]) => modelIdentity(key) === identity)
  candidates.sort(([nameA, a], [nameB, b]) =>
    Number(hasUsefulBenchmark(b)) - Number(hasUsefulBenchmark(a)) || effortRank(nameB) - effortRank(nameA))
  return candidates.length ? { ...candidates[0][1], source_model: candidates[0][0] } : null
}

export function findBridgebench(name: string): number | null {
  const identity = modelIdentity(name, false)
  return Object.entries(bridgebench.scores).find(([key]) => modelIdentity(key, false) === identity)?.[1] ?? null
}

export function parsePrice(price: string): number {
  const value = decodeHtml(price).trim()
  if (value.toLowerCase() === 'free') return 0
  return /^\$\s*\d[\d,]*(?:\.\d+)?$/.test(value) ? Number(value.slice(1).replaceAll(',', '').trim()) : Infinity
}

export function computeCodIndex(coding: number | null): number | null {
  if (coding === null) return null
  return Math.round((10 + 90 * Math.max((coding - 10.7) / (48.1 - 10.7), 0)) * 10) / 10
}

export function parseHtml(html: string): ParsedModel[] | null {
  for (const table of html.matchAll(/<table\b[^>]*>([\s\S]*?)<\/table\s*>/gi)) {
    const rows = [...table[1].matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr\s*>/gi)].map(row =>
      [...row[1].matchAll(/<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]\s*>/gi)].map(cell =>
        decodeHtml(cell[1].replace(/<br\s*\/?\s*>/gi, ' ').replace(/<[^>]+>/g, '')).replace(/\s+/g, ' ').trim()))
    const headerIndex = rows.findIndex(row => ['model', 'input', 'output'].every(key => row.map(cell => cell.toLowerCase()).includes(key)))
    if (headerIndex === -1) continue
    const headers = rows[headerIndex].map(cell => cell.toLowerCase())
    const models: ParsedModel[] = []
    for (const row of rows.slice(headerIndex + 1)) {
      const cell = (key: string) => row[headers.indexOf(key)] ?? '-'
      if (!cell('model') || cell('model') === '-') continue
      const input = cell('input')
      const output = cell('output')
      if (!Number.isFinite(parsePrice(input)) && !Number.isFinite(parsePrice(output))) continue
      models.push({ name: cell('model'), input_price: input, output_price: output, cached_read: cell('cached read'), cached_write: cell('cached write') })
    }
    if (models.length) return models
  }
  return null
}

export function parseBenchmarks(payload: unknown): Benchmarks {
  if (!payload || typeof payload !== 'object' || !('data' in payload) || !Array.isArray(payload.data)) {
    throw new Error('Invalid Artificial Analysis response')
  }
  const benchmarks: Benchmarks = {}
  for (const model of payload.data) {
    if (!model || typeof model.name !== 'string') continue
    const evaluations = model.evaluations ?? {}
    const gpqa = evaluations.gpqa ?? null
    const coding = evaluations.artificial_analysis_coding_index ?? null
    if ([gpqa, coding].some(value => value !== null && (typeof value !== 'number' || !Number.isFinite(value)))) {
      throw new Error('Invalid benchmark score')
    }
    benchmarks[model.name] = { creator: model.model_creator?.name, gpqa, coding }
  }
  if (!Object.keys(benchmarks).length) throw new Error('Empty Artificial Analysis response')
  return benchmarks
}

export function buildModels(parsed: ParsedModel[], benchmarks: Benchmarks | null) {
  return parsed.map(model => {
    const weighted = 0.9784 * parsePrice(model.input_price) + 0.0216 * parsePrice(model.output_price)
    const benchmark = benchmarks ? matchModelToBenchmarks(model.name, benchmarks) : null
    const coding = benchmark?.coding ?? null
    return {
      ...model,
      // JSON не поддерживает Infinity. Неизвестная цена — null в API и UI.
      weighted_price: Number.isFinite(weighted) ? weighted : null,
      gpqa: benchmark?.gpqa ?? null,
      coding,
      cod_index: computeCodIndex(coding),
      benchmark_model: benchmark?.source_model ?? null,
      bridgebench: findBridgebench(model.name),
    }
  })
}

<template>
  <div class="container">
    <header class="header">
      <h1>OpenCode Zen Pricing</h1>
      <p class="subtitle">Сравнение цен и бенчмарков AI моделей</p>
      <p v-if="lastUpdate" class="update-time">Обновлено: {{ lastUpdate }}</p>
    </header>

    <div v-if="!models || models.length === 0" class="error">
      <p>Не удалось загрузить данные моделей</p>
    </div>

    <main v-else class="main">
      <div class="stats">
        <div class="stat-card">
          <span class="stat-label">Всего моделей</span>
          <span class="stat-value">{{ models.length }}</span>
        </div>
        <div class="stat-card" v-if="hasBenchmarks">
          <span class="stat-label">С бенчмарками</span>
          <span class="stat-value">{{ modelsWithBenchmarks }}</span>
        </div>
      </div>

      <div class="table-wrapper">
        <table class="models-table">
          <thead>
            <tr>
              <th>Модель</th>
              <th class="col-price">Вход</th>
              <th class="col-price">Выход</th>
              <th class="col-price">Взвеш.</th>
              <th v-if="hasBenchmarks" class="col-benchmark">CodIdx</th>
              <th v-if="hasBenchmarks" class="col-benchmark">Coding</th>
              <th v-if="hasBenchmarks" class="col-benchmark">GPQA</th>
              <th v-if="hasBridgebench" class="col-benchmark">Bridge</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="model in sortedModels" :key="model.name" :class="{ 'row-infinite': model.weighted_price === Infinity }">
              <td class="col-name">
                <span class="model-name">{{ model.name }}</span>
              </td>
              <td class="col-price">{{ model.input_price }}</td>
              <td class="col-price">{{ model.output_price }}</td>
              <td class="col-price">
                <span v-if="model.weighted_price !== Infinity" class="price-weighted">
                  {{ formatWeightedPrice(model.weighted_price) }}
                </span>
                <span v-else class="price-na">-</span>
              </td>
              <td v-if="hasBenchmarks" class="col-benchmark">
                {{ formatValue(model.cod_index) }}
              </td>
              <td v-if="hasBenchmarks" class="col-benchmark">
                {{ formatBenchmark(model.coding) }}
              </td>
              <td v-if="hasBenchmarks" class="col-benchmark">
                {{ formatBenchmark(model.gpqa) }}
              </td>
              <td v-if="hasBridgebench" class="col-benchmark">
                {{ formatBenchmark(model.bridgebench) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div v-if="hasBenchmarks" class="legend">
        <p><strong>CodIdx:</strong> линейный индекс кодинга (10=Haiku 3.5, 100=Opus 4.6)</p>
        <p><strong>Coding:</strong> AA Coding Index, <strong>GPQA:</strong> GPQA Diamond</p>
        <p v-if="hasBridgebench"><strong>Bridge:</strong> BridgeBench Overall (bridgemind.ai)</p>
        <p class="source">Источник бенчмарков: artificialanalysis.ai</p>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
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

// Загружаем данные при сборке (статическая генерация)
const { data } = await useFetch('/api/models', {
  key: 'models-data',
  deep: false
})

const models = computed(() => data.value?.models || [])
const hasBenchmarks = computed(() => data.value?.has_benchmarks || false)
const lastUpdate = computed(() => data.value?.last_update || '')

const modelsWithBenchmarks = computed(() =>
  models.value.filter(m => m.coding !== null || m.gpqa !== null).length
)

const hasBridgebench = computed(() =>
  models.value.some(m => m.bridgebench !== null)
)

const sortedModels = computed(() =>
  [...models.value].sort((a, b) => a.weighted_price - b.weighted_price)
)

function formatWeightedPrice(price: number): string {
  return `$${price.toFixed(4)}`
}

function formatValue(value: number | null): string {
  if (value === null) return '-'
  return value.toFixed(1)
}

function formatBenchmark(value: number | null): string {
  if (value === null) return '-'
  if (value < 1) return `${(value * 100).toFixed(1)}%`
  return value.toFixed(1)
}

useHead({
  title: 'OpenCode Zen Pricing',
  meta: [
    { name: 'description', content: 'Сравнение цен и бенчмарков AI моделей' }
  ]
})
</script>

<style scoped>
.container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 2rem;
  min-height: 100vh;
}

.header {
  text-align: center;
  margin-bottom: 2rem;
}

.header h1 {
  font-size: 2.5rem;
  font-weight: 700;
  color: #1a1a1a;
  margin-bottom: 0.5rem;
}

.subtitle {
  color: #666;
  font-size: 1.1rem;
  margin-bottom: 0.5rem;
}

.update-time {
  color: #999;
  font-size: 0.9rem;
}

.error {
  text-align: center;
  padding: 4rem 2rem;
  color: #d32f2f;
}

.stats {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
}

.stat-card {
  flex: 1;
  min-width: 150px;
  padding: 1rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 10px;
  color: white;
  text-align: center;
}

.stat-label {
  display: block;
  font-size: 0.85rem;
  opacity: 0.9;
  margin-bottom: 0.25rem;
}

.stat-value {
  display: block;
  font-size: 1.75rem;
  font-weight: 700;
}

.table-wrapper {
  overflow-x: auto;
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.models-table {
  width: 100%;
  border-collapse: collapse;
  background: white;
}

.models-table th,
.models-table td {
  padding: 0.75rem 1rem;
  text-align: left;
  border-bottom: 1px solid #e0e0e0;
}

.models-table th {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-weight: 600;
  white-space: nowrap;
}

.models-table tbody tr:hover {
  background: #f5f5f5;
}

.models-table tbody tr:last-child td {
  border-bottom: none;
}

.row-infinite {
  opacity: 0.6;
}

.col-name {
  min-width: 200px;
}

.model-name {
  font-weight: 500;
  color: #1a1a1a;
}

.col-price {
  min-width: 80px;
  font-family: 'Monaco', 'Menlo', monospace;
}

.col-benchmark {
  min-width: 80px;
  text-align: center;
  font-family: 'Monaco', 'Menlo', monospace;
}

.price-weighted {
  color: #0066ff;
  font-weight: 600;
}

.price-na {
  color: #999;
}

.legend {
  margin-top: 2rem;
  padding: 1rem 1.5rem;
  background: #f5f5f5;
  border-radius: 8px;
  font-size: 0.9rem;
  color: #666;
}

.legend p {
  margin: 0.25rem 0;
}

.source {
  margin-top: 0.75rem !important;
  font-style: italic;
}

@media (max-width: 768px) {
  .container {
    padding: 1rem;
  }

  .header h1 {
    font-size: 1.75rem;
  }

  .models-table th,
  .models-table td {
    padding: 0.5rem;
    font-size: 0.85rem;
  }

  .stats {
    flex-direction: column;
  }

  .stat-card {
    min-width: auto;
  }
}
</style>

import dotenv from 'dotenv'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

// Load .ENV file
const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
dotenv.config({ path: join(__dirname, '.ENV') })

// Получаем API ключ из переменных окружения (поддерживаем оба варианта написания)
const apiKey = process.env.ARTIFICIAL_ANALYSIS_API || process.env.ARTIFICIAL_ANALYSIS_API || ''

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-04-03',
  devtools: { enabled: true },
  devServer: {
    port: 3000
  },
  nitro: {
    experimental: {
      openAPI: true
    }
  },
  ssr: true,
  css: ['~/assets/css/main.css'],
  runtimeConfig: {
    // Server-side runtime config (доступен только на сервере)
    artificialAnalysisApi: apiKey
  }
})

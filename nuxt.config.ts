import dotenv from 'dotenv'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

// Load .ENV file
const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
dotenv.config({ path: [join(__dirname, '.ENV'), join(__dirname, '.env')], quiet: true })

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-04-03',
  devtools: { enabled: true },
  devServer: {
    port: 3000
  },
  nitro: {
    storage: {
      cache: { driver: 'fs', base: './.data/cache' }
    }
  },
  ssr: true,
  css: ['~/assets/css/main.css'],
  runtimeConfig: {
    // Server-side runtime config (доступен только на сервере)
    artificialAnalysisApi: ''
  }
})

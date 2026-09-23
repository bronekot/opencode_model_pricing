export interface CacheEntry<T> {
  fetched_at: number
  value: T
}
export interface CacheStore {
  getItem<T>(key: string): Promise<T | null>
  setItem<T>(key: string, value: T): Promise<unknown>
}

export async function loadSource<T>(
  storage: CacheStore,
  key: string,
  ttl: number,
  fetcher: (() => Promise<T>) | null,
): Promise<(CacheEntry<T> & { from_cache: boolean; stale: boolean }) | null> {
  const cached = await storage.getItem<CacheEntry<T>>(key).catch(() => null)
  const valid = cached && Number.isFinite(cached.fetched_at) && cached.value != null ? cached : null
  const age = valid ? Date.now() - valid.fetched_at : Infinity
  const stale = age < 0 || age >= ttl
  if (valid && (!stale || !fetcher)) return { ...valid, from_cache: true, stale }
  if (fetcher) {
    try {
      const value = await fetcher()
      const entry = { fetched_at: Date.now(), value }
      // Отказ диска не должен скрывать успешно загруженные данные.
      await storage.setItem(key, entry).catch(() => {})
      return { ...entry, from_cache: false, stale: false }
    } catch {
      // При недоступном источнике сохраняем дату последней успешной загрузки.
    }
  }
  return valid ? { ...valid, from_cache: true, stale: true } : null
}

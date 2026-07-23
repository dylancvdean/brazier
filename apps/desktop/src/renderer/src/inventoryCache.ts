import type { LocalModel } from './api'
import type { RuntimeEntry } from './api'

const MODELS_KEY = 'brazier.models.v1'
const RUNTIMES_KEY = 'brazier.runtimes.v1'

export function readCachedModels(): LocalModel[] {
  try {
    const raw = localStorage.getItem(MODELS_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as LocalModel[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function writeCachedModels(models: LocalModel[]): void {
  try {
    localStorage.setItem(MODELS_KEY, JSON.stringify(models))
  } catch {
    // Ignore quota errors — cache is best-effort.
  }
}

export function readCachedRuntimes(): RuntimeEntry[] {
  try {
    const raw = localStorage.getItem(RUNTIMES_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as RuntimeEntry[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function writeCachedRuntimes(runtimes: RuntimeEntry[]): void {
  try {
    localStorage.setItem(RUNTIMES_KEY, JSON.stringify(runtimes))
  } catch {
    // Ignore quota errors — cache is best-effort.
  }
}

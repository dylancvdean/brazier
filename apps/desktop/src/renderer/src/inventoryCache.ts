import type { LocalModel } from './api'
import type { RuntimeEntry } from './api'
import type { Conversation } from './types'

const MODELS_KEY = 'brazier.models.v1'
const RUNTIMES_KEY = 'brazier.runtimes.v1'
const CONVERSATIONS_KEY = 'brazier.conversations.v1'
/** Enough to fill the visible sidebar; the daemon's answer replaces it. */
const CONVERSATION_CACHE_LIMIT = 50

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

/**
 * Last known conversation list, so the sidebar has content to paint while the
 * daemon is still starting up rather than showing an empty history.
 */
export function readCachedConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(CONVERSATIONS_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as Conversation[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function writeCachedConversations(conversations: Conversation[]): void {
  try {
    localStorage.setItem(
      CONVERSATIONS_KEY,
      JSON.stringify(conversations.slice(0, CONVERSATION_CACHE_LIMIT))
    )
  } catch {
    // Ignore quota errors — cache is best-effort.
  }
}

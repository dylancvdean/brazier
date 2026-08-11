import type { LocalModel } from './api'
import type { RuntimeEntry } from './api'
import type { Conversation } from './types'

const LEGACY_MODELS_KEY = 'brazier.models.v1'
const LEGACY_RUNTIMES_KEY = 'brazier.runtimes.v1'
const LEGACY_CONVERSATIONS_KEY = 'brazier.conversations.v1'
/** Enough to fill the visible sidebar; the daemon's answer replaces it. */
const CONVERSATION_CACHE_LIMIT = 50

function key(kind: string, profileId: string): string {
  return `brazier.${kind}.v2.${encodeURIComponent(profileId)}`
}

function readValue<T>(kind: string, profileId: string, legacyKey: string): T[] {
  try {
    const raw = localStorage.getItem(key(kind, profileId)) ??
      (profileId === 'local' ? localStorage.getItem(legacyKey) : null)
    if (!raw) return []
    const parsed = JSON.parse(raw) as T[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function readCachedModels(profileId: string): LocalModel[] {
  return readValue<LocalModel>('models', profileId, LEGACY_MODELS_KEY)
}

export function writeCachedModels(profileId: string, models: LocalModel[]): void {
  try {
    localStorage.setItem(key('models', profileId), JSON.stringify(models))
  } catch {
    // Ignore quota errors — cache is best-effort.
  }
}

export function readCachedRuntimes(profileId: string): RuntimeEntry[] {
  return readValue<RuntimeEntry>('runtimes', profileId, LEGACY_RUNTIMES_KEY)
}

export function writeCachedRuntimes(profileId: string, runtimes: RuntimeEntry[]): void {
  try {
    localStorage.setItem(key('runtimes', profileId), JSON.stringify(runtimes))
  } catch {
    // Ignore quota errors — cache is best-effort.
  }
}

/**
 * Last known conversation list, so the sidebar has content to paint while the
 * daemon is still starting up rather than showing an empty history.
 */
export function readCachedConversations(profileId: string): Conversation[] {
  return readValue<Conversation>('conversations', profileId, LEGACY_CONVERSATIONS_KEY)
}

export function writeCachedConversations(profileId: string, conversations: Conversation[]): void {
  try {
    localStorage.setItem(
      key('conversations', profileId),
      JSON.stringify(conversations.slice(0, CONVERSATION_CACHE_LIMIT))
    )
  } catch {
    // Ignore quota errors — cache is best-effort.
  }
}

import type { LocalModel } from './api'
import type { RuntimeEntry } from './api'
import type { Conversation } from './types'

/** Enough to fill the visible sidebar; the daemon's answer replaces it. */
const CONVERSATION_CACHE_LIMIT = 50

function key(kind: string, profileId: string): string {
  return `brazier.${kind}.v2.${encodeURIComponent(profileId)}`
}

function readValue<T>(kind: string, profileId: string): T[] {
  try {
    const raw = localStorage.getItem(key(kind, profileId))
    if (!raw) return []
    const parsed = JSON.parse(raw) as T[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function writeValue(kind: string, profileId: string, value: unknown): void {
  try {
    localStorage.setItem(key(kind, profileId), JSON.stringify(value))
  } catch {
    // Ignore quota errors — cache is best-effort.
  }
}

export function readCachedModels(profileId: string): LocalModel[] {
  return readValue<LocalModel>('models', profileId)
}

export function writeCachedModels(profileId: string, models: LocalModel[]): void {
  writeValue('models', profileId, models)
}

export function readCachedRuntimes(profileId: string): RuntimeEntry[] {
  return readValue<RuntimeEntry>('runtimes', profileId)
}

export function writeCachedRuntimes(profileId: string, runtimes: RuntimeEntry[]): void {
  writeValue('runtimes', profileId, runtimes)
}

export function readCachedConversations(profileId: string): Conversation[] {
  return readValue<Conversation>('conversations', profileId)
}

export function writeCachedConversations(profileId: string, conversations: Conversation[]): void {
  writeValue('conversations', profileId, conversations.slice(0, CONVERSATION_CACHE_LIMIT))
}

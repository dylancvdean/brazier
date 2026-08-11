import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  readCachedConversations,
  readCachedModels,
  readCachedRuntimes,
  writeCachedConversations,
  writeCachedModels,
  writeCachedRuntimes
} from './inventoryCache'

beforeEach(() => {
  const values = new Map<string, string>()
  vi.stubGlobal('localStorage', {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value)
  })
})

afterEach(() => vi.unstubAllGlobals())

describe('profile-scoped renderer inventory', () => {
  it('does not expose models, runtimes, or conversations from another daemon profile', () => {
    writeCachedModels('local', [{ id: 'local-model' }] as never)
    writeCachedRuntimes('local', [{ id: 'local-runtime' }] as never)
    writeCachedConversations('local', [{ id: 'local-chat' }] as never)

    expect(readCachedModels('remote-gpu')).toEqual([])
    expect(readCachedRuntimes('remote-gpu')).toEqual([])
    expect(readCachedConversations('remote-gpu')).toEqual([])

    writeCachedModels('remote-gpu', [{ id: 'remote-model' }] as never)
    expect(readCachedModels('local')).toEqual([{ id: 'local-model' }])
    expect(readCachedModels('remote-gpu')).toEqual([{ id: 'remote-model' }])
  })

  it('reads legacy cache keys only for the reserved Local profile', () => {
    localStorage.setItem('brazier.models.v1', JSON.stringify([{ id: 'legacy-local' }]))
    expect(readCachedModels('local')).toEqual([{ id: 'legacy-local' }])
    expect(readCachedModels('remote-gpu')).toEqual([])
  })
})

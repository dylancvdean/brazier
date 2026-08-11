import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  DaemonOfflineError,
  daemonAvailability,
  daemonFetch,
  setDaemonAvailability
} from './daemonAvailability'

afterEach(() => {
  setDaemonAvailability('checking')
  vi.unstubAllGlobals()
})

describe('offline daemon mutation boundary', () => {
  it('fails closed while the selected daemon is still being checked', async () => {
    const fetch_ = vi.fn(async () => new Response(null, { status: 204 }))
    vi.stubGlobal('fetch', fetch_)

    await expect(
      daemonFetch('https://daemon.example/api/v1/agent/sessions', { method: 'POST' })
    ).rejects.toBeInstanceOf(DaemonOfflineError)
    expect(fetch_).not.toHaveBeenCalled()
  })

  it('preserves reads but blocks mutations before they reach the network while offline', async () => {
    const fetch_ = vi.fn(async () => {
      throw new TypeError('still offline')
    })
    vi.stubGlobal('fetch', fetch_)
    setDaemonAvailability('offline')

    await expect(daemonFetch('https://daemon.example/api/v1/models')).rejects.toThrow('still offline')
    await expect(
      daemonFetch('https://daemon.example/api/v1/models', { method: 'DELETE' })
    ).rejects.toBeInstanceOf(DaemonOfflineError)
    expect(fetch_).toHaveBeenCalledTimes(1)
  })

  it('marks transport failures offline and any HTTP response reachable', async () => {
    const fetch_ = vi
      .fn()
      .mockRejectedValueOnce(new TypeError('network down'))
      .mockResolvedValueOnce(new Response('{}', { status: 401 }))
    vi.stubGlobal('fetch', fetch_)

    await expect(daemonFetch('https://daemon.example/health')).rejects.toThrow('network down')
    expect(daemonAvailability()).toBe('offline')
    await daemonFetch('https://daemon.example/health')
    expect(daemonAvailability()).toBe('healthy')
  })
})

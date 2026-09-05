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
  it('checks the method of a Request object and respects explicit overrides', async () => {
    const fetch_ = vi.fn(async () => new Response(null, { status: 204 }))
    vi.stubGlobal('fetch', fetch_)
    const request = new Request('https://daemon.example/models', { method: 'DELETE' })
    await expect(daemonFetch(request)).rejects.toBeInstanceOf(DaemonOfflineError)
    expect(fetch_).not.toHaveBeenCalled()
    await daemonFetch(request, { method: 'GET' })
    expect(fetch_).toHaveBeenCalledOnce()
  })

  it('does not mark the daemon offline when a caller cancels', async () => {
    setDaemonAvailability('healthy')
    const abort = new AbortController()
    vi.stubGlobal('fetch', vi.fn(async () => {
      abort.abort(new Error('Cancelled by user'))
      throw abort.signal.reason
    }))
    await expect(daemonFetch('https://daemon.example/health', { signal: abort.signal }))
      .rejects.toThrow('Cancelled by user')
    expect(daemonAvailability()).toBe('healthy')
  })

  it('aborts old requests on reconnect and ignores their late responses', async () => {
    let resolve!: (response: Response) => void
    let signal!: AbortSignal
    vi.stubGlobal('fetch', vi.fn((_input, init) => {
      signal = init.signal
      return new Promise<Response>((done) => { resolve = done })
    }))
    const pending = daemonFetch('https://old.example/health')
    setDaemonAvailability('checking')
    expect(signal.aborted).toBe(true)
    resolve(new Response('{}'))
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' })
    expect(daemonAvailability()).toBe('checking')
  })

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

import { describe, expect, it, vi } from 'vitest'

import { runPackageSmoke } from './packageSmoke'

describe('runPackageSmoke', () => {
  it('checks the safety helper, loads the worker, opens a no-model session, and awaits every cleanup', async () => {
    const requests: Array<{ url: string; method: string }> = []
    const cleanupOrder: string[] = []
    const fetch_ = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input)
      requests.push({ url, method: init?.method ?? 'GET' })
      if (init?.method === 'POST') {
        return new Response(JSON.stringify({ id: 'session-1' }), {
          status: 200,
          headers: { 'content-type': 'application/json' }
        })
      }
      cleanupOrder.push('delete-session')
      return new Response(null, { status: 204 })
    })
    let clock = 100
    const result = await runPackageSmoke({
      connection: { address: 'http://127.0.0.1:7614', apiKey: 'owner' },
      checkSafetyHelper: async () => {
        cleanupOrder.push('check-safety-helper')
      },
      warmWorker: async () => {
        clock += 30
      },
      openSession: async (id) => {
        expect(id).toBe('session-1')
        clock += 45
      },
      shutdownWorker: vi.fn(async () => {
        cleanupOrder.push('shutdown-worker')
      }),
      shutdownDaemon: vi.fn(async () => {
        cleanupOrder.push('shutdown-daemon')
      }),
      fetch: fetch_ as typeof fetch,
      now: () => clock,
      commit: 'abcdef012345',
      platform: 'macos',
      arch: 'arm64',
      artifact: 'dmg'
    })

    expect(result.passed).toBe(true)
    expect(result.checks).toEqual({
      daemon_started: true,
      safety_helper_present: true,
      windows_sandbox_probe: true,
      worker_loaded: true,
      session_opened: true,
      worker_shutdown: true,
      session_deleted: true,
      daemon_stopped: true,
      clean_shutdown: true
    })
    expect(result.metrics).toEqual({ agent_worker_ready_ms: 30, agent_session_open_ms: 45 })
    expect(requests.map((request) => request.method)).toEqual(['POST', 'DELETE'])
    expect(cleanupOrder).toEqual([
      'check-safety-helper',
      'shutdown-worker',
      'delete-session',
      'shutdown-daemon'
    ])
  })

  it('writes failing evidence and still attempts worker cleanup', async () => {
    const shutdown = vi.fn(async () => undefined)
    const result = await runPackageSmoke({
      connection: { address: 'http://127.0.0.1:7614', apiKey: null },
      checkSafetyHelper: async () => undefined,
      warmWorker: async () => undefined,
      openSession: async () => undefined,
      shutdownWorker: shutdown,
      shutdownDaemon: async () => undefined,
      fetch: vi.fn(async () => new Response('no', { status: 500 })) as typeof fetch,
      commit: 'abcdef012345',
      platform: 'linux',
      arch: 'x64',
      artifact: 'appimage'
    })

    expect(result.passed).toBe(false)
    expect(result.error).toContain('500')
    expect(shutdown).toHaveBeenCalledOnce()
  })

  it('cannot pass until the local daemon exits', async () => {
    let releaseDaemon: (() => void) | undefined
    const shutdownDaemon = vi.fn(
      () => new Promise<void>((resolve) => {
        releaseDaemon = resolve
      })
    )
    let completed = false
    const resultPromise = runPackageSmoke({
      connection: { address: 'http://127.0.0.1:7614', apiKey: null },
      checkSafetyHelper: async () => undefined,
      warmWorker: async () => undefined,
      openSession: async () => undefined,
      shutdownWorker: async () => undefined,
      shutdownDaemon,
      fetch: vi.fn(async (_input: string | URL | Request, init?: RequestInit) =>
        init?.method === 'POST'
          ? new Response(JSON.stringify({ id: 'session-1' }), {
              status: 200,
              headers: { 'content-type': 'application/json' }
            })
          : init?.method === undefined
            ? new Response(JSON.stringify({ sandbox: { sandboxed_execution: true } }), {
                status: 200,
                headers: { 'content-type': 'application/json' }
              })
          : new Response(null, { status: 204 })
      ) as typeof fetch,
      commit: 'abcdef012345',
      platform: 'windows',
      arch: 'x64',
      artifact: 'nsis'
    }).then((result) => {
      completed = true
      return result
    })

    await vi.waitFor(() => expect(shutdownDaemon).toHaveBeenCalledOnce())
    expect(completed).toBe(false)
    releaseDaemon?.()
    const result = await resultPromise
    expect(result.passed).toBe(true)
    expect(result.checks.windows_sandbox_probe).toBe(true)
    expect(result.checks.daemon_stopped).toBe(true)
  })

  it('writes failing evidence when a cleanup prerequisite fails', async () => {
    const result = await runPackageSmoke({
      connection: { address: 'http://127.0.0.1:7614', apiKey: null },
      checkSafetyHelper: async () => {
        throw new Error('brazier-safety is missing')
      },
      warmWorker: async () => undefined,
      openSession: async () => undefined,
      shutdownWorker: async () => undefined,
      shutdownDaemon: async () => {
        throw new Error('did not exit')
      },
      fetch: vi.fn() as typeof fetch,
      commit: 'abcdef012345',
      platform: 'macos',
      arch: 'arm64',
      artifact: 'dmg'
    })

    expect(result.passed).toBe(false)
    expect(result.checks.safety_helper_present).toBe(false)
    expect(result.checks.daemon_stopped).toBe(false)
    expect(result.checks.clean_shutdown).toBe(false)
    expect(result.error).toContain('brazier-safety is missing')
    expect(result.error).toContain('daemon shutdown failed: did not exit')
  })
})

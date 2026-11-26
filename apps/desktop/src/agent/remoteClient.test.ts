import { afterEach, describe, expect, it, vi } from 'vitest'

import { anySignal, timeoutSignal } from './core/abort'
import { BrokerClient } from './core/brokerClient'
import type { WorkerMessage } from './core/protocol'
import { AgentWorkerCore } from './workerCore'

/**
 * A paired device (the mobile app) runs this core against a remote daemon with
 * its own HTTP client and a key that holds `agent` and `inference` but not
 * `management`.
 */

const session = {
  id: 'sess-1',
  title: 'Task',
  model: 'model-a',
  runtime_id: 'simple',
  permission_mode: 'ask',
  permission_settings: { auto_approve_host_actions: false, auto_approve_sandboxed_actions: true },
  enabled_tools: null,
  workspace_path: '/tmp',
  created_at: '2026-01-01',
  updated_at: '2026-01-01'
}

function pairedDaemon(): { http: typeof fetch; paths: string[] } {
  const paths: string[] = []
  const reply = (status: number, body: unknown): Response =>
    new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
  const http = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = new URL(String(input)).pathname
    paths.push(path)
    expect(new Headers(init?.headers).get('authorization')).toBe('Bearer paired-key')
    if (path === '/api/v1/agent/tools') return reply(200, { data: [] })
    if (path === '/api/v1/agent/sessions/sess-1') {
      return reply(200, {
        session,
        messages: [],
        tool_executions: [],
        pending_approvals: [],
        grants: [],
        sandbox: { backend: 'seatbelt', isolated: true, detail: '' }
      })
    }
    if (path === '/api/v1/agent/sessions/sess-1/prompt') {
      return reply(200, { system_prompt: 'prompt', tools: [] })
    }
    // Model and runtime settings need the `management` scope.
    return reply(403, { error: { message: 'Insufficient client scope.' } })
  }) as typeof fetch
  return { http, paths }
}

describe('agent core on a paired remote client', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('uses the injected HTTP client and opens a session without the management scope', async () => {
    const globalFetch = vi.fn(async () => {
      throw new Error('global fetch must not be used')
    })
    vi.stubGlobal('fetch', globalFetch)
    const { http, paths } = pairedDaemon()
    const posts: WorkerMessage[] = []
    const core = new AgentWorkerCore((message) => posts.push(message), undefined, http)

    await core.handle({
      type: 'init',
      requestId: 'a',
      connection: { address: 'https://host.example', apiKey: 'paired-key' }
    })
    await core.handle({ type: 'open-session', requestId: 'b', sessionId: 'sess-1' })

    const results = posts.filter((message) => message.type === 'result')
    expect(results).toEqual([
      expect.objectContaining({ requestId: 'a', ok: true }),
      expect.objectContaining({ requestId: 'b', ok: true })
    ])
    expect(paths).toContain('/api/v1/models/settings')
    expect(paths).toContain('/api/v1/runtime/settings')
    expect(globalFetch).not.toHaveBeenCalled()
  })

  it('gives sessions and subagents a fallback that leaves reasoning to the host', async () => {
    const { http } = pairedDaemon()
    const broker = new BrokerClient({ address: 'https://host.example', apiKey: 'paired-key', fetch: http })
    await expect(broker.runtimeInferenceSettings()).resolves.toEqual({ context_size: 8192 })
  })

  it('still fails to open a session on errors other than a missing scope', async () => {
    const { http } = pairedDaemon()
    const failing = (async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith('/api/v1/runtime/settings')) {
        return new Response(JSON.stringify({ error: { message: 'boom' } }), { status: 500 })
      }
      return http(input, init)
    }) as typeof fetch
    const posts: WorkerMessage[] = []
    const core = new AgentWorkerCore((message) => posts.push(message), undefined, failing)
    await core.handle({
      type: 'init',
      requestId: 'a',
      connection: { address: 'https://host.example', apiKey: 'paired-key' }
    })
    await core.handle({ type: 'open-session', requestId: 'b', sessionId: 'sess-1' })
    expect(posts).toContainEqual({ type: 'result', requestId: 'b', ok: false, error: 'boom' })
  })
})

describe('abort helpers without AbortSignal.any and AbortSignal.timeout', () => {
  afterEach(() => vi.restoreAllMocks())

  function removeStatics(): void {
    const statics = AbortSignal as unknown as { any?: unknown; timeout?: unknown }
    vi.spyOn(statics as { any: unknown }, 'any', 'get').mockReturnValue(undefined)
    vi.spyOn(statics as { timeout: unknown }, 'timeout', 'get').mockReturnValue(undefined)
  }

  it('aborts when any source aborts, including one already aborted', () => {
    removeStatics()
    const first = new AbortController()
    const second = new AbortController()
    const combined = anySignal([first.signal, second.signal])
    expect(combined.aborted).toBe(false)
    second.abort()
    expect(combined.aborted).toBe(true)
    expect(anySignal([first.signal, second.signal]).aborted).toBe(true)
  })

  it('aborts after the timeout', async () => {
    removeStatics()
    vi.useFakeTimers()
    try {
      const signal = timeoutSignal(50)
      expect(signal.aborted).toBe(false)
      await vi.advanceTimersByTimeAsync(50)
      expect(signal.aborted).toBe(true)
    } finally {
      vi.useRealTimers()
    }
  })
})

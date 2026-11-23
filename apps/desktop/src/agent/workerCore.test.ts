import { describe, expect, it, vi } from 'vitest'

import type { WorkerMessage } from './core/protocol'
import type { AgentMessage, AgentRuntime, AgentSession, AgentSessionState } from './core/types'
import { AgentWorkerCore } from './workerCore'

function mockSession(id: string, disposed = false): AgentSession {
  const state: AgentSessionState = {
    id,
    title: 'Task',
    workspacePath: '/tmp',
    model: { id: 'model-a', name: 'model-a' },
    runtimeId: 'simple',
    messages: [],
    toolExecutions: [],
    permissionMode: 'ask',
    permissionSettings: {
      auto_approve_host_actions: false,
      auto_approve_sandboxed_actions: true
    },
    createdAt: '2026-01-01',
    updatedAt: '2026-01-01',
    lastRunStatus: 'idle'
  }
  let messages = [...state.messages]
  return {
    id,
    isBusy: () => false,
    isDisposed: () => disposed,
    getState: () => ({ ...state, messages }),
    rehydrate: (next: AgentMessage[]) => {
      if (disposed) throw new Error('Cannot rehydrate a disposed agent session.')
      messages = next
    },
    refreshInferencePrefs: vi.fn(async () => undefined),
    run: vi.fn(),
    cancel: vi.fn(async () => undefined),
    compact: vi.fn(),
    setModel: vi.fn(),
    setEnabledTools: vi.fn(),
    setPermissionMode: vi.fn(),
    dispose: vi.fn(async () => undefined)
  }
}

function installBrokerAndRuntime(
  core: AgentWorkerCore,
  broker: Record<string, unknown>,
  runtime: AgentRuntime
): void {
  ;(core as unknown as { broker: typeof broker }).broker = broker
  ;(core as unknown as { runtimes: Map<string, AgentRuntime> }).runtimes = new Map([
    ['simple', runtime]
  ])
  ;(core as unknown as { tools: [] }).tools = []
}

describe('AgentWorkerCore.openSession', () => {
  it('does not rehydrate while a run is in flight', async () => {
    const posts: unknown[] = []
    const core = new AgentWorkerCore((message) => {
      posts.push(message)
    })
    const broker = {
      session: vi.fn(async () => ({
        session: {
          id: 'sess-1',
          title: 'Task',
          model: 'model-a',
          runtime_id: 'simple',
          permission_mode: 'ask',
          permission_settings: {
            auto_approve_host_actions: false,
            auto_approve_sandboxed_actions: true
          },
          enabled_tools: null,
          workspace_path: '/tmp',
          created_at: '2026-01-01',
          updated_at: '2026-01-01'
        },
        messages: [{ payload: { role: 'user', text: 'daemon', timestamp: 't' } }],
        tool_executions: [],
        pending_approvals: [],
        grants: [],
        sandbox: {
          backend: 'test',
          isolated: true,
          sandboxed_execution: true,
          filesystem_scoping: true,
          network_isolation: true,
          process_isolation: true,
          profiles: [],
          detail: 'test sandbox'
        }
      })),
      systemPrompt: vi.fn(async () => ({ system_prompt: 'prompt' })),
      tools: vi.fn(async () => []),
      textProfile: vi.fn(async () => ({ context_size: 4096 })),
      runtimeInferenceSettings: vi.fn(async () => ({ context_size: 4096 }))
    }
    const runtime = {
      descriptor: { id: 'simple', name: 'Simple', version: '0', capabilities: {} },
      createSession: vi.fn(),
      restoreSession: vi.fn(),
      dispose: vi.fn()
    } as unknown as AgentRuntime

    await core.handle({
      type: 'init',
      requestId: 'init',
      connection: { address: 'http://127.0.0.1:1', apiKey: null }
    })
    installBrokerAndRuntime(core, broker, runtime)

    const session = mockSession('sess-1')
    const rehydrate = vi.spyOn(session, 'rehydrate')
    ;(core as unknown as { sessions: Map<string, AgentSession> }).sessions.set('sess-1', session)
    ;(core as unknown as { running: Set<string> }).running.add('sess-1')

    await core.handle({ type: 'open-session', requestId: 'open', sessionId: 'sess-1' })

    expect(rehydrate).not.toHaveBeenCalled()
    expect(broker.session).not.toHaveBeenCalled()
  })

  it('drops disposed sessions from the cache before rehydrating', async () => {
    const core = new AgentWorkerCore(() => undefined)
    const broker = {
      session: vi.fn(async () => ({
        session: {
          id: 'sess-2',
          title: 'Task',
          model: 'model-a',
          runtime_id: 'simple',
          permission_mode: 'ask',
          permission_settings: {
            auto_approve_host_actions: false,
            auto_approve_sandboxed_actions: true
          },
          enabled_tools: null,
          workspace_path: '/tmp',
          created_at: '2026-01-01',
          updated_at: '2026-01-01'
        },
        messages: [],
        tool_executions: [],
        pending_approvals: [],
        grants: [],
        sandbox: {
          backend: 'test',
          isolated: true,
          sandboxed_execution: true,
          filesystem_scoping: true,
          network_isolation: true,
          process_isolation: true,
          profiles: [],
          detail: 'test sandbox'
        }
      })),
      systemPrompt: vi.fn(async () => ({ system_prompt: 'prompt' })),
      tools: vi.fn(async () => []),
      textProfile: vi.fn(async () => ({ context_size: 4096 })),
      runtimeInferenceSettings: vi.fn(async () => ({ context_size: 4096 }))
    }
    const created = mockSession('sess-2')
    const createSession = vi.fn(async () => created)
    const runtime = {
      descriptor: { id: 'simple', name: 'Simple', version: '0', capabilities: {} },
      createSession,
      restoreSession: vi.fn(),
      dispose: vi.fn()
    } as unknown as AgentRuntime

    await core.handle({
      type: 'init',
      requestId: 'init',
      connection: { address: 'http://127.0.0.1:1', apiKey: null }
    })
    installBrokerAndRuntime(core, broker, runtime)
    ;(core as unknown as { sessions: Map<string, AgentSession> }).sessions.set(
      'sess-2',
      mockSession('sess-2', true)
    )

    await core.handle({ type: 'open-session', requestId: 'open', sessionId: 'sess-2' })

    expect(createSession).toHaveBeenCalled()
  })
})

describe('AgentWorkerCore scheduling', () => {
  /** A session whose run streams until it is cancelled. */
  function blockingSession(id: string): AgentSession & { stop: () => void } {
    const session = mockSession(id)
    let stop!: () => void
    const stopped = new Promise<void>((resolve) => {
      stop = resolve
    })
    session.run = vi.fn(() => ({
      async *[Symbol.asyncIterator]() {
        yield {
          type: 'run-started' as const,
          sessionId: id,
          runId: 'run-1',
          timestamp: 't',
          sequence: 1
        }
        await stopped
        yield {
          type: 'run-cancelled' as const,
          sessionId: id,
          runId: 'run-1',
          timestamp: 't',
          sequence: 2
        }
      }
    }))
    session.cancel = vi.fn(async () => stop())
    return Object.assign(session, { stop })
  }

  function coreWith(sessions: AgentSession[]): { core: AgentWorkerCore; posts: WorkerMessage[] } {
    const posts: WorkerMessage[] = []
    const core = new AgentWorkerCore((message) => posts.push(message))
    const broker = { cancel: vi.fn(async () => undefined) }
    installBrokerAndRuntime(core, broker, {} as AgentRuntime)
    const map = (core as unknown as { sessions: Map<string, AgentSession> }).sessions
    for (const session of sessions) map.set(session.id, session)
    return { core, posts }
  }

  const resultFor = (posts: WorkerMessage[], requestId: string): WorkerMessage | undefined =>
    posts.find((message) => message.type === 'result' && message.requestId === requestId)

  it('lets Stop reach a session while its run is streaming', async () => {
    const session = blockingSession('sess-a')
    const { core, posts } = coreWith([session])

    const run = core.handle({ type: 'run', requestId: 'run', sessionId: 'sess-a', input: { text: 'go' } })
    await vi.waitFor(() => expect(session.run).toHaveBeenCalled())

    await core.handle({ type: 'cancel', requestId: 'cancel', sessionId: 'sess-a' })
    expect(session.cancel).toHaveBeenCalled()
    expect(resultFor(posts, 'cancel')).toMatchObject({ ok: true })

    await run
    expect(resultFor(posts, 'run')).toMatchObject({ ok: true })
    const types = posts.flatMap((message) => (message.type === 'event' ? [message.event.type] : []))
    expect(types).toEqual(['run-started', 'run-cancelled'])
  })

  it('does not hold other sessions behind a running one', async () => {
    const busy = blockingSession('sess-a')
    const idle = mockSession('sess-b')
    ;(idle.setPermissionMode as ReturnType<typeof vi.fn>).mockResolvedValue(undefined)
    const { core, posts } = coreWith([busy, idle])

    const run = core.handle({ type: 'run', requestId: 'run', sessionId: 'sess-a', input: { text: 'go' } })
    await vi.waitFor(() => expect(busy.run).toHaveBeenCalled())

    await core.handle({
      type: 'set-permission-mode',
      requestId: 'mode',
      sessionId: 'sess-b',
      mode: 'sandbox-only'
    })
    expect(resultFor(posts, 'mode')).toMatchObject({ ok: true })
    expect(resultFor(posts, 'run')).toBeUndefined()

    busy.stop()
    await run
  })

  it('does not start a run that was cancelled while it was opening', async () => {
    const session = mockSession('sess-c')
    const { core, posts } = coreWith([])
    let finishOpen!: () => void
    const opening = new Promise<void>((resolve) => {
      finishOpen = resolve
    })
    ;(core as unknown as { openSession: () => Promise<AgentSession> }).openSession = async () => {
      await opening
      return session
    }

    const run = core.handle({ type: 'run', requestId: 'run', sessionId: 'sess-c', input: { text: 'go' } })
    await core.handle({ type: 'cancel', requestId: 'cancel', sessionId: 'sess-c' })
    finishOpen()
    await run

    expect(session.run).not.toHaveBeenCalled()
    expect(resultFor(posts, 'run')).toMatchObject({ ok: true })
    expect(
      posts.some((message) => message.type === 'event' && message.event.type === 'run-cancelled')
    ).toBe(true)
  })
})

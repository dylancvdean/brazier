import { describe, expect, it, vi, type Mock } from 'vitest'

import type { WorkerMessage } from './core/protocol'
import type { AgentMessage, AgentRuntime, AgentSession, AgentSessionState } from './core/types'
import { AgentWorkerCore } from './workerCore'

function mockSession(id: string, disposed = false): AgentSession {
  const state: AgentSessionState = {
    id,
    title: 'Task',
    workspacePath: '/tmp',
    model: { id: 'model-a', name: 'model-a' },
    runtimeId: 'pi',
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
    ['pi', runtime]
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
          runtime_id: 'pi',
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
      descriptor: { id: 'pi', name: 'Pi', version: '0', capabilities: {} },
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
          runtime_id: 'pi',
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
      descriptor: { id: 'pi', name: 'Pi', version: '0', capabilities: {} },
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

  it('selects the runtime from session.runtime_id', async () => {
    const ompCreate = vi.fn(async () => mockSession('sess-omp'))
    const piCreate = vi.fn(async () => mockSession('sess-pi'))
    const core = new AgentWorkerCore((_message) => undefined, (_broker, id) => {
      if (id === 'omp') {
        return {
          descriptor: { id: 'omp', name: 'Oh My Pi', version: '0', capabilities: {} },
          createSession: ompCreate,
          restoreSession: vi.fn(),
          dispose: vi.fn()
        } as unknown as AgentRuntime
      }
      return {
        descriptor: { id: 'pi', name: 'Pi', version: '0', capabilities: {} },
        createSession: piCreate,
        restoreSession: vi.fn(),
        dispose: vi.fn()
      } as unknown as AgentRuntime
    })
    const broker = {
      session: vi.fn(async () => ({
        session: {
          id: 'sess-omp',
          title: 'Task',
          model: 'model-a',
          runtime_id: 'omp',
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
          isolated: false,
          sandboxed_execution: false,
          filesystem_scoping: false,
          network_isolation: false,
          process_isolation: false,
          profiles: [],
          detail: 'host'
        }
      })),
      systemPrompt: vi.fn(async () => ({ system_prompt: 'prompt' })),
      tools: vi.fn(async () => []),
      textProfile: vi.fn(async () => ({ context_size: 4096 })),
      runtimeInferenceSettings: vi.fn(async () => ({ context_size: 4096 }))
    }

    await core.handle({
      type: 'init',
      requestId: 'init',
      connection: { address: 'http://127.0.0.1:1', apiKey: null }
    })
    ;(core as unknown as { broker: typeof broker }).broker = broker
    ;(core as unknown as { tools: [] }).tools = []

    await core.handle({ type: 'open-session', requestId: 'open', sessionId: 'sess-omp' })

    expect(ompCreate).toHaveBeenCalled()
    expect(piCreate).not.toHaveBeenCalled()
    expect(broker.systemPrompt).not.toHaveBeenCalled()
    expect(ompCreate).toHaveBeenCalledWith(expect.objectContaining({ systemPrompt: '' }))
  })
})

describe('AgentWorkerCore runtime frames and commands', () => {
  /** Core with a session that exposes a runtime frame stream, already open. */
  async function openOmpCore(): Promise<{
    core: AgentWorkerCore
    posts: WorkerMessage[]
    emit: (payload: Record<string, unknown>) => void
    runtimeCommand: Mock
    resolveExtensionUi: Mock
  }> {
    const posts: WorkerMessage[] = []
    const core = new AgentWorkerCore((message) => {
      posts.push(message as WorkerMessage)
    })
    const frameListenerRef: { current?: (payload: Record<string, unknown>) => void } = {}
    const runtimeCommand = vi.fn(async (command: Record<string, unknown>) => ({ echo: command }))
    const resolveExtensionUi = vi.fn(async (response: Record<string, unknown>) => ({ resolved: true }))
    const session: AgentSession = {
      ...mockSession('sess-omp'),
      getState: () => ({
        ...mockSession('sess-omp').getState(),
        runtimeId: 'omp'
      }),
      subscribeRuntimeFrames: (listener) => {
        frameListenerRef.current = listener
        return () => {
          if (frameListenerRef.current === listener) frameListenerRef.current = undefined
        }
      },
      sendRuntimeCommand: runtimeCommand,
      resolveExtensionUi
    }
    const broker = {
      session: vi.fn(async () => ({
        session: {
          id: 'sess-omp',
          title: 'Task',
          model: 'model-a',
          runtime_id: 'omp',
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
          isolated: false,
          sandboxed_execution: false,
          filesystem_scoping: false,
          network_isolation: false,
          process_isolation: false,
          profiles: [],
          detail: 'host'
        }
      })),
      systemPrompt: vi.fn(async () => ({ system_prompt: 'prompt' })),
      tools: vi.fn(async () => []),
      textProfile: vi.fn(async () => ({ context_size: 4096 })),
      runtimeInferenceSettings: vi.fn(async () => ({ context_size: 4096 }))
    }
    await core.handle({
      type: 'init',
      requestId: 'init',
      connection: { address: 'http://127.0.0.1:1', apiKey: null }
    })
    ;(core as unknown as { broker: typeof broker }).broker = broker
    ;(core as unknown as { tools: [] }).tools = []
    ;(core as unknown as { sessions: Map<string, AgentSession> }).sessions.set('sess-omp', session)

    // Open once to arm the frame subscription without rehydrating the injected session.
    await core.handle({ type: 'open-session', requestId: 'open', sessionId: 'sess-omp' })
    return {
      core,
      posts,
      emit: (payload) => frameListenerRef.current?.(payload),
      runtimeCommand,
      resolveExtensionUi
    }
  }

  it('forwards runtime frames to the renderer with session and runtime ids', async () => {
    const { emit, posts } = await openOmpCore()
    posts.length = 0

    emit({ type: 'command_output', text: 'review summary' })

    const forwarded = posts.find((message) => message.type === 'runtime-frame')
    expect(forwarded).toMatchObject({
      type: 'runtime-frame',
      sessionId: 'sess-omp',
      runtimeId: 'omp',
      payload: { type: 'command_output', text: 'review summary' }
    })
  })

  it('does not stack frame listeners when a session is reopened', async () => {
    const { core, emit, posts } = await openOmpCore()

    await core.handle({ type: 'open-session', requestId: 'open2', sessionId: 'sess-omp' })
    posts.length = 0
    emit({ type: 'notice', message: 'once' })

    const forwarded = posts.filter((message) => message.type === 'runtime-frame')
    expect(forwarded).toHaveLength(1)
  })

  it('stops forwarding frames after the session is closed', async () => {
    const { core, emit, posts } = await openOmpCore()

    await core.handle({ type: 'close-session', requestId: 'close', sessionId: 'sess-omp' })
    posts.length = 0
    emit({ type: 'notice', message: 'after close' })

    expect(posts.filter((message) => message.type === 'runtime-frame')).toHaveLength(0)
  })

  it('routes runtime-command through the session and returns its response frame', async () => {
    const { core, posts, runtimeCommand } = await openOmpCore()

    await core.handle({
      type: 'runtime-command',
      requestId: 'cmd',
      sessionId: 'sess-omp',
      runtimeId: 'omp',
      command: { type: 'get_state' }
    })

    expect(runtimeCommand).toHaveBeenCalledWith({ type: 'get_state' })
    const result = posts.find((message) => message.type === 'result' && message.requestId === 'cmd')
    expect(result).toMatchObject({ ok: true, data: { echo: { type: 'get_state' } } })
  })

  it('routes extension-UI resolution through the session', async () => {
    const { core, posts, resolveExtensionUi } = await openOmpCore()

    await core.handle({
      type: 'resolve-extension-ui',
      requestId: 'resolve',
      sessionId: 'sess-omp',
      response: { type: 'extension_ui_response', id: 'ui-1', value: 'yes' }
    })

    expect(resolveExtensionUi).toHaveBeenCalledWith({
      type: 'extension_ui_response',
      id: 'ui-1',
      value: 'yes'
    })
    const result = posts.find((message) => message.type === 'result' && message.requestId === 'resolve')
    expect(result).toMatchObject({ ok: true, data: { resolved: true } })
  })

  it('rejects runtime-command for a mismatched runtime id without calling the session', async () => {
    const { core, posts, runtimeCommand } = await openOmpCore()

    await core.handle({
      type: 'runtime-command',
      requestId: 'cmd',
      sessionId: 'sess-omp',
      runtimeId: 'pi',
      command: { type: 'get_state' }
    })

    expect(runtimeCommand).not.toHaveBeenCalled()
    const result = posts.find((message) => message.type === 'result' && message.requestId === 'cmd')
    expect(result).toMatchObject({ ok: false, error: expect.stringContaining('does not match') })
  })
})

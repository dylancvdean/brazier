/**
 * Integration test across the real adapters.
 *
 * `coordinator.test.ts` covers policy against in-memory fakes. This one drives
 * the actual agent adapter — the translation between the worker's session/run
 * events and the coordinator's correlation ids — plus the chat adapter's
 * normalization, with only the daemon calls and the IPC bridge stubbed.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { AgentEvent } from '../../../agent/core/types'
import type { WorkerMessage } from '../../../agent/core/protocol'
import type { Message } from '../types'
import { setDaemonAvailability } from '../daemonAvailability'

beforeEach(() => {
  setDaemonAvailability('healthy')
})

const daemon = vi.hoisted(() => ({
  conversations: new Map<string, { id: string; agent_session_id: string | null }>(),
  updates: [] as Array<{ id: string; update: Record<string, unknown> }>
}))

vi.mock('../api', () => ({
  getConversation: vi.fn(async (id: string) => {
    const conversation = daemon.conversations.get(id)
    if (!conversation) throw new Error('no such conversation')
    return conversation
  }),
  updateConversation: vi.fn(async (id: string, update: Record<string, unknown>) => {
    daemon.updates.push({ id, update })
    const conversation = daemon.conversations.get(id)
    if (conversation && 'agent_session_id' in update) {
      conversation.agent_session_id = update.agent_session_id as string | null
    }
    return conversation
  }),
  createMessage: vi.fn(),
  updateMessage: vi.fn(),
  createVoiceSession: vi.fn(),
  endVoiceSession: vi.fn(),
  transcribeAudio: vi.fn()
}))

vi.mock('../agentApi', () => ({
  fetchAgentSession: vi.fn(async (id: string) => {
    if (id === 'missing') throw new Error('gone')
    return { session: { id } }
  }),
  updateAgentSession: vi.fn(async () => ({}))
}))

const { WorkerAgentAdapter } = await import('./agentAdapter')
const { InMemoryChatAdapter, toConversationMessage } = await import('./chatAdapter')
const { SessionCoordinator } = await import('./coordinator')
const { DEFAULT_INTEGRATION_CONFIG } = await import('./config')
const { FakeChat, FakeVoice, sequentialIds } = await import('./testFakes')

/** Stub of the preload agent bridge, plus a way to push worker events. */
function installAgentBridge() {
  const listeners = new Set<(message: WorkerMessage) => void>()
  const calls: Array<{ method: string; args: unknown[] }> = []
  let settleRun: (error?: Error) => void = () => {}
  const record = (method: string) =>
    vi.fn(async (...args: unknown[]) => {
      calls.push({ method, args })
    })
  const bridge = {
    openSession: record('openSession'),
    // A real run stays open until the worker finishes, which is after its events.
    run: vi.fn(
      (...args: unknown[]) =>
        new Promise<void>((resolve, reject) => {
          calls.push({ method: 'run', args })
          settleRun = (error?: Error) => {
            if (error) reject(error)
            else resolve()
          }
        })
    ),
    cancel: record('cancel'),
    compact: record('compact'),
    setModel: record('setModel'),
    setTools: record('setTools'),
    closeSession: record('closeSession'),
    status: vi.fn(async () => ({ running: true, crashes: 0 })),
    onMessage: (listener: (message: WorkerMessage) => void) => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    }
  }
  Object.defineProperty(globalThis, 'window', {
    value: { brazier: { agent: bridge } },
    configurable: true,
    writable: true
  })
  return {
    calls,
    finishRun() {
      settleRun()
    },
    failRun(message = 'worker failed') {
      settleRun(new Error(message))
    },
    emit(sessionId: string, event: AgentEvent) {
      for (const listener of [...listeners]) listener({ type: 'event', sessionId, event })
    }
  }
}

function agentEvent<T extends AgentEvent['type']>(
  type: T,
  extra: Record<string, unknown> = {}
): AgentEvent {
  return {
    type,
    sessionId: 'agent-1',
    runId: 'run-1',
    timestamp: new Date().toISOString(),
    sequence: 1,
    ...extra
  } as AgentEvent
}

describe('WorkerAgentAdapter', () => {
  let bridge: ReturnType<typeof installAgentBridge>

  beforeEach(() => {
    daemon.conversations.clear()
    daemon.updates.length = 0
    bridge = installAgentBridge()
  })

  it('adopts the agent session the conversation records', async () => {
    daemon.conversations.set('conv-1', { id: 'conv-1', agent_session_id: 'agent-1' })
    const adapter = new WorkerAgentAdapter()
    expect(await adapter.attachSession('conv-1')).toBe('agent-1')
    expect(adapter.attachedSessionId()).toBe('agent-1')
    expect(bridge.calls.some((call) => call.method === 'openSession')).toBe(true)
  })

  it('reports no session rather than inventing one', async () => {
    daemon.conversations.set('conv-2', { id: 'conv-2', agent_session_id: null })
    const adapter = new WorkerAgentAdapter()
    expect(await adapter.attachSession('conv-2')).toBeNull()

    // A binding that points at a session the daemon no longer has is not honoured.
    daemon.conversations.set('conv-3', { id: 'conv-3', agent_session_id: 'missing' })
    expect(await adapter.attachSession('conv-3')).toBeNull()
  })

  it('binds a session so both surfaces share it', async () => {
    daemon.conversations.set('conv-1', { id: 'conv-1', agent_session_id: null })
    const adapter = new WorkerAgentAdapter()
    await adapter.attachSession('conv-1')
    await adapter.bindSession('agent-9')

    expect(adapter.attachedSessionId()).toBe('agent-9')
    expect(daemon.updates).toEqual([{ id: 'conv-1', update: { agent_session_id: 'agent-9' } }])
  })

  it('refuses a turn with nothing bound', async () => {
    const adapter = new WorkerAgentAdapter()
    await expect(
      adapter.submitTurn({ correlationId: 'turn-1', text: 'hello', source: 'user_voice' })
    ).rejects.toThrow(/No agent session/)
  })

  it('maps worker events onto the turn they belong to', async () => {
    daemon.conversations.set('conv-1', { id: 'conv-1', agent_session_id: 'agent-1' })
    const adapter = new WorkerAgentAdapter()
    await adapter.attachSession('conv-1')
    const seen: string[] = []
    adapter.subscribe((event) => seen.push(`${event.type}:${event.correlationId}`))

    await adapter.submitTurn({ correlationId: 'turn-7', text: 'run the tests', source: 'user_text' })
    bridge.emit('agent-1', agentEvent('run-started'))
    bridge.emit('agent-1', agentEvent('text-delta', { delta: 'thinking', channel: 'reasoning' }))
    bridge.emit('agent-1', agentEvent('text-delta', { delta: 'One test', channel: 'text' }))
    bridge.emit(
      'agent-1',
      agentEvent('tool-completed', {
        toolCallId: 'c1',
        tool: 'shell',
        environment: 'sandbox',
        sandbox: { backend: 'seatbelt', profile: 'p', isolated: true, network: false, detail: '' },
        output: 'FAILED oggOpus.test.ts\nmore detail',
        truncated: false,
        exitCode: 1,
        changedPaths: [],
        durationMs: 12
      })
    )

    expect(seen).toEqual([
      'runStarted:turn-7',
      // Reasoning is not an answer and is not forwarded.
      'responsePartial:turn-7',
      'toolCompleted:turn-7'
    ])
    expect(adapter.getStatus('turn-7')?.status).toBe('running')
  })

  it('summarizes a tool result to something short enough to say', async () => {
    daemon.conversations.set('conv-1', { id: 'conv-1', agent_session_id: 'agent-1' })
    const adapter = new WorkerAgentAdapter()
    await adapter.attachSession('conv-1')
    const outcomes: string[] = []
    adapter.subscribe((event) => {
      if (event.type === 'toolCompleted') outcomes.push(event.outcome)
    })
    await adapter.submitTurn({ correlationId: 'turn-1', text: 'build', source: 'user_text' })
    bridge.emit(
      'agent-1',
      agentEvent('tool-completed', {
        toolCallId: 'c1',
        tool: 'shell',
        environment: 'sandbox',
        sandbox: { backend: 'seatbelt', profile: 'p', isolated: true, network: false, detail: '' },
        output: `${'x'.repeat(400)}\nsecond line`,
        truncated: true,
        exitCode: 0,
        changedPaths: ['src/a.ts', 'src/b.ts'],
        durationMs: 12
      })
    )

    expect(outcomes).toHaveLength(1)
    expect(outcomes[0]).toContain('exit 0')
    expect(outcomes[0]).toContain('changed 2 file(s)')
    // Bounded: a whole build log is never handed to the speech path.
    expect(outcomes[0].length).toBeLessThan(200)
  })

  it('maps an approval the model settled onto the turn', async () => {
    daemon.conversations.set('conv-1', { id: 'conv-1', agent_session_id: 'agent-1' })
    const adapter = new WorkerAgentAdapter()
    await adapter.attachSession('conv-1')
    const seen: string[] = []
    adapter.subscribe((event) => seen.push(event.type))
    await adapter.submitTurn({ correlationId: 'turn-1', text: 'edit it', source: 'user_text' })
    bridge.emit(
      'agent-1',
      agentEvent('approval-reviewed', { toolCallId: 'c1', approvalId: 'ap-1', phase: 'started' })
    )
    bridge.emit(
      'agent-1',
      agentEvent('approval-resolved', {
        toolCallId: 'c1',
        approvalId: 'ap-1',
        status: 'approved',
        decidedBy: 'model-review'
      })
    )
    expect(seen).toEqual(['statusUpdated', 'approvalResolved'])
  })

  it('reports a run that resolved without a terminal event', async () => {
    daemon.conversations.set('conv-1', { id: 'conv-1', agent_session_id: 'agent-1' })
    const adapter = new WorkerAgentAdapter()
    await adapter.attachSession('conv-1')
    const seen: string[] = []
    adapter.subscribe((event) => {
      if (event.type === 'runFailed') seen.push(event.error)
    })
    await adapter.submitTurn({ correlationId: 'turn-1', text: 'hello', source: 'user_text' })
    bridge.finishRun()
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(seen).toEqual(['The run ended before it reported a result.'])
    expect(adapter.getStatus('turn-1')?.status).toBe('failed')
  })

  it('ignores events from another session', async () => {
    daemon.conversations.set('conv-1', { id: 'conv-1', agent_session_id: 'agent-1' })
    const adapter = new WorkerAgentAdapter()
    await adapter.attachSession('conv-1')
    const seen: string[] = []
    adapter.subscribe((event) => seen.push(event.type))
    await adapter.submitTurn({ correlationId: 'turn-1', text: 'x', source: 'user_text' })
    bridge.emit('agent-other', agentEvent('run-started'))
    expect(seen).toEqual([])
  })
})

describe('coordinator over the real agent adapter', () => {
  beforeEach(() => {
    daemon.conversations.clear()
    daemon.conversations.set('conv-1', { id: 'conv-1', agent_session_id: 'agent-1' })
  })

  async function wired() {
    const bridge = installAgentBridge()
    const chat = new FakeChat()
    const voice = new FakeVoice()
    const agent = new WorkerAgentAdapter()
    const coordinator = new SessionCoordinator({
      chat,
      agent,
      voice,
      newId: sequentialIds(),
      config: {
        ...DEFAULT_INTEGRATION_CONFIG,
        voiceEnabled: true,
        voiceSessionTarget: 'agent',
        voiceBackgroundRouting: 'always'
      }
    })
    coordinator.connect()
    await coordinator.attach('conv-1')
    await coordinator.startVoiceSession()
    return { bridge, chat, voice, agent, coordinator }
  }

  it('stores one answer when the worker commits it and then completes the run', async () => {
    const { bridge, chat, voice, coordinator } = await wired()

    voice.emit({ type: 'userTranscriptFinal', utteranceId: 'utt-1', text: 'What failed?' })
    await new Promise((resolve) => setTimeout(resolve, 0))

    bridge.emit('agent-1', agentEvent('run-started'))
    bridge.emit(
      'agent-1',
      agentEvent('message-committed', {
        message: { role: 'assistant', text: 'oggOpus failed.', timestamp: '' }
      })
    )
    // The worker reports the same answer again in its run summary; the second
    // one must not become a second message or a second spoken reply.
    bridge.emit(
      'agent-1',
      agentEvent('run-completed', {
        summary: {
          filesChanged: [],
          commandsRun: [],
          toolCalls: 1,
          failures: [],
          hostActions: [],
          approvalsRequested: 0,
          text: 'oggOpus failed.'
        }
      })
    )
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(chat.assistantMessages()).toHaveLength(1)
    expect(chat.assistantMessages()[0].content).toBe('oggOpus failed.')
    expect(chat.assistantMessages()[0].source).toBe('assistant_agent')
    expect(coordinator.metrics().duplicateEventsIgnored).toBe(1)
  })

  it('does not prematurely complete a multi-step agent turn on intermediate tool-calling messages', async () => {
    const { bridge, voice, chat, coordinator } = await wired()
    voice.emit({ type: 'userTranscriptFinal', utteranceId: 'utt-ms', text: 'Run the test suite.' })
    await new Promise((resolve) => setTimeout(resolve, 0))

    bridge.emit('agent-1', agentEvent('run-started'))
    bridge.emit(
      'agent-1',
      agentEvent('message-committed', {
        message: {
          role: 'assistant',
          text: 'Let me run the tests.',
          toolCalls: [{ id: 'tc-1', name: 'shell_run', arguments: { command: 'pnpm test' } }],
          timestamp: ''
        }
      })
    )
    await new Promise((resolve) => setTimeout(resolve, 0))

    // Turn must still be active after the intermediate tool-calling message.
    expect(coordinator.snapshot().activeCorrelationId).not.toBeNull()
    expect(chat.assistantMessages()).toHaveLength(0)

    bridge.emit(
      'agent-1',
      agentEvent('message-committed', {
        message: { role: 'assistant', text: 'All 42 tests passed.', timestamp: '' }
      })
    )
    bridge.emit(
      'agent-1',
      agentEvent('run-completed', {
        summary: {
          filesChanged: [],
          commandsRun: ['pnpm test'],
          toolCalls: 1,
          failures: [],
          hostActions: [],
          approvalsRequested: 0,
          text: 'All 42 tests passed.'
        }
      })
    )
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(coordinator.snapshot().activeCorrelationId).toBeNull()
    expect(chat.assistantMessages()).toHaveLength(1)
    expect(chat.assistantMessages()[0].content).toBe('All 42 tests passed.')
  })

  it('cancels through the bridge only for the live turn', async () => {
    const { bridge, voice, coordinator } = await wired()
    voice.emit({ type: 'userTranscriptFinal', utteranceId: 'utt-1', text: 'Start the long build.' })
    await new Promise((resolve) => setTimeout(resolve, 0))
    bridge.emit('agent-1', agentEvent('run-started'))

    // Stale id: nothing is cancelled.
    await coordinator.cancelAgentTask('turn-99')
    expect(bridge.calls.filter((call) => call.method === 'cancel')).toHaveLength(0)

    voice.emit({ type: 'userTranscriptFinal', utteranceId: 'utt-2', text: 'Never mind, cancel that.' })
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(bridge.calls.filter((call) => call.method === 'cancel')).toHaveLength(1)
  })
})

describe('chat adapter normalization', () => {
  function stored(overrides: Partial<Message>): Message {
    return {
      id: 'm1',
      conversation_id: 'conv-1',
      parent_id: null,
      role: 'user',
      content: 'hello',
      model: null,
      created_at: '2026-07-25T00:00:00Z',
      ...overrides
    }
  }

  it('keeps recorded attribution', () => {
    const message = toConversationMessage(
      stored({
        role: 'assistant',
        source: 'assistant_agent',
        correlation_id: 'turn-3',
        status: 'final',
        metadata: { queued: false }
      })
    )
    expect(message.source).toBe('assistant_agent')
    expect(message.correlationId).toBe('turn-3')
    expect(message.status).toBe('final')
  })

  it('infers a source for messages written before the integration', () => {
    expect(toConversationMessage(stored({ role: 'user' })).source).toBe('user_text')
    expect(toConversationMessage(stored({ role: 'assistant' })).source).toBe('assistant_chat')
    expect(toConversationMessage(stored({ role: 'tool' })).source).toBe('tool')
    expect(toConversationMessage(stored({ role: 'system' })).source).toBe('system')
    // An unknown label is not trusted through.
    expect(toConversationMessage(stored({ source: 'nonsense' })).source).toBe('user_text')
    expect(toConversationMessage(stored({ status: 'nonsense' })).status).toBe('final')
  })

  it('flattens multi-part content to the text the voice can use', () => {
    const message = toConversationMessage(
      stored({
        content: [
          { type: 'text', text: 'look at this' },
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AA' } },
          { type: 'text', text: 'and this' }
        ]
      })
    )
    expect(message.content).toBe('look at this\nand this')
  })

  it('preserves parent_id, conversation_id, model, and correlation_id in InMemoryChatAdapter.updateMessage', async () => {
    const emitted: Message[] = []
    let tipId: string | null = null
    const adapter = new InMemoryChatAdapter(
      {
        onMessage: (msg) => {
          emitted.push(msg)
          tipId = msg.id
        },
        parentId: () => tipId,
        model: () => 'qwen3-8b'
      },
      'incognito-thread-1'
    )

    const userMsg = await adapter.appendMessage({
      role: 'user',
      source: 'user_text',
      content: 'First incognito question',
      correlationId: 'corr-1',
      status: 'final'
    })
    const assistantPartial = await adapter.appendMessage({
      role: 'assistant',
      source: 'assistant_chat',
      content: 'Working...',
      correlationId: 'corr-1',
      status: 'partial'
    })

    await adapter.updateMessage(assistantPartial.id, {
      content: 'Final incognito answer',
      status: 'final'
    })

    const lastEmitted = emitted.at(-1)!
    expect(lastEmitted.id).toBe(assistantPartial.id)
    expect(lastEmitted.parent_id).toBe(userMsg.id)
    expect(lastEmitted.conversation_id).toBe('incognito-thread-1')
    expect(lastEmitted.model).toBe('qwen3-8b')
    expect(lastEmitted.correlation_id).toBe('corr-1')
    expect(lastEmitted.content).toBe('Final incognito answer')
    expect(lastEmitted.status).toBe('final')
  })
})

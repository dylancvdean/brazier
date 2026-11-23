import { describe, expect, it, vi } from 'vitest'

import {
  ContextOverflowError,
  StreamInterruptedError,
  type BrokerClient,
  type StreamChatCompletionCallbacks,
  type StreamChatCompletionRequest,
  type StreamChatCompletionResult
} from '../core/brokerClient'
import { inferModelCapabilities } from '../core/modelCompat'
import type {
  AgentCompactedEvent,
  AgentEvent,
  AgentMessage,
  AgentSession,
  AgentToolDefinition,
  SandboxDescription
} from '../core/types'
import { NativeAgentRuntime, partitionToolWaves } from './nativeRuntime'

const sandbox: SandboxDescription = {
  backend: 'seatbelt',
  profile: 'workspace',
  isolated: true,
  network: false,
  detail: 'test sandbox'
}

const tools: AgentToolDefinition[] = [
  {
    name: 'fs_read',
    label: 'Read',
    description: 'Read a file.',
    inputSchema: {
      type: 'object',
      properties: { path: { type: 'string' } },
      required: ['path'],
      additionalProperties: false
    },
    risk: 'read',
    executes: false,
    needsWorkspace: true,
    defaultEnvironment: 'sandbox'
  },
  {
    name: 'git_status',
    label: 'Git status',
    description: 'Inspect git status.',
    inputSchema: { type: 'object', properties: {}, required: [] },
    risk: 'read',
    executes: true,
    needsWorkspace: true,
    defaultEnvironment: 'sandbox'
  },
  {
    name: 'fs_patch',
    label: 'Patch',
    description: 'Edit a file.',
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string' },
        old_string: { type: 'string' },
        new_string: { type: 'string' }
      },
      required: ['path', 'old_string', 'new_string'],
      additionalProperties: false
    },
    risk: 'write',
    executes: false,
    needsWorkspace: true,
    defaultEnvironment: 'sandbox'
  },
  {
    name: 'shell_run',
    label: 'Shell',
    description: 'Run a shell command.',
    inputSchema: {
      type: 'object',
      properties: { command: { type: 'string' } },
      required: ['command'],
      additionalProperties: false
    },
    risk: 'execute',
    executes: true,
    needsWorkspace: true,
    defaultEnvironment: 'sandbox'
  }
]

function makeMockBroker(overrides: Record<string, unknown> = {}): BrokerClient {
  return {
    capabilities: vi.fn(async () => ({
      sandbox: { backend: 'seatbelt', isolated: true, detail: 'test sandbox' }
    })),
    session: vi.fn(async (id: string) => ({
      session: {
        id,
        title: 'Task',
        workspace_path: '/tmp',
        model: 'gguf:qwen3-32b.gguf',
        runtime_id: 'simple',
        permission_mode: 'ask',
        permission_settings: {
          auto_approve_sandboxed_actions: true,
          auto_approve_host_actions: false
        },
        enabled_tools: null,
        last_run_status: 'idle',
        runtime_metadata: null,
        created_at: '2026-01-01T00:00:00Z',
        updated_at: '2026-01-01T00:00:00Z'
      },
      messages: [],
      tool_executions: [],
      pending_approvals: [],
      grants: [],
      sandbox: {
        backend: 'seatbelt',
        isolated: true,
        sandboxed_execution: true,
        filesystem_scoping: true,
        network_isolation: true,
        process_isolation: true,
        profiles: ['workspace'],
        detail: 'test sandbox'
      }
    })),
    systemPrompt: vi.fn(async () => ({ system_prompt: 'System prompt.', tools: [] })),
    tools: vi.fn(async () => tools),
    textProfile: vi.fn(async () => ({ context_size: 65536, max_tokens: 8192 })),
    runtimeInferenceSettings: vi.fn(async () => ({
      context_size: 32768,
      max_tokens: 4096,
      enable_reasoning: true,
      drop_reasoning_between_turns: false
    })),
    appendMessages: vi.fn(async () => undefined),
    updateSession: vi.fn(async () => ({})),
    execTool: vi.fn(async (req: { tool: string }) => ({
      status: 'completed',
      tool: req.tool,
      environment: 'sandbox',
      risk: 'read',
      sandbox,
      output: 'ok',
      truncated: false,
      changed_paths: [],
      duration_ms: 1,
      is_error: false
    })),
    execToolStreaming: vi.fn(async (req: { tool: string }) => ({
      status: 'completed',
      tool: req.tool,
      environment: 'sandbox',
      risk: 'execute',
      sandbox,
      output: 'ok',
      truncated: false,
      changed_paths: [],
      duration_ms: 1,
      is_error: false
    })),
    cancel: vi.fn(async () => undefined),
    openAiBaseUrl: () => 'http://127.0.0.1:9/v1',
    apiKey: () => 'brazier-local',
    streamChatCompletion: vi.fn(),
    ...overrides
  } as unknown as BrokerClient
}

describe('partitionToolWaves', () => {
  const defsMap = new Map(tools.map((t) => [t.name, t]))

  it('groups consecutive read-only and spawn_subagent calls into concurrent waves and isolates writes/executes', () => {
    const calls = [
      { id: '1', name: 'fs_read' },
      { id: '2', name: 'git_status' },
      { id: '3', name: 'spawn_subagent' },
      { id: '4', name: 'fs_patch' },
      { id: '5', name: 'fs_patch' },
      { id: '6', name: 'fs_read' },
      { id: '7', name: 'shell_run' }
    ]
    const waves = partitionToolWaves(calls, defsMap, true)
    expect(waves.map((wave) => wave.map((c) => c.id))).toEqual([
      ['1', '2', '3'],
      ['4'],
      ['5'],
      ['6'],
      ['7']
    ])
  })

  it('keeps all calls 1-per-wave when parallelToolCalling is false', () => {
    const calls = [
      { id: '1', name: 'fs_read' },
      { id: '2', name: 'git_status' }
    ]
    const waves = partitionToolWaves(calls, defsMap, false)
    expect(waves.map((wave) => wave.map((c) => c.id))).toEqual([['1'], ['2']])
  })
})

describe('NativeAgentRuntime & NativeAgentSession', () => {
  it('preserves contextWindow in restoreSession instead of reverting to 8,192', async () => {
    const broker = makeMockBroker()
    const runtime = new NativeAgentRuntime(broker)

    // 1. From profile.context_size (65536) when no explicit override is passed
    const restoredFromProfile = await runtime.restoreSession('sess-profile')
    expect(restoredFromProfile.getState().model.contextWindow).toBe(65536)
    expect(restoredFromProfile.getState().model.maxTokens).toBe(8192)

    // 2. Explicit input.contextWindow override
    const restoredExplicit = await runtime.restoreSession({
      sessionId: 'sess-explicit',
      contextWindow: 131072
    })
    expect(restoredExplicit.getState().model.contextWindow).toBe(131072)
  })

  it('runs read tools in parallel waves while serializing fs_patch writes', async () => {
    let inFlightReads = 0
    let maxConcurrentReads = 0
    let inFlightWrites = 0
    let maxConcurrentWrites = 0
    let turn = 0

    const broker = makeMockBroker({
      execTool: vi.fn(async (req: { tool: string; arguments: Record<string, unknown> }) => {
        if (req.tool === 'fs_read' || req.tool === 'git_status') {
          inFlightReads += 1
          maxConcurrentReads = Math.max(maxConcurrentReads, inFlightReads)
          await new Promise((r) => setTimeout(r, 20))
          inFlightReads -= 1
        } else if (req.tool === 'fs_patch') {
          inFlightWrites += 1
          maxConcurrentWrites = Math.max(maxConcurrentWrites, inFlightWrites)
          await new Promise((r) => setTimeout(r, 20))
          inFlightWrites -= 1
        }
        return {
          status: 'completed',
          tool: req.tool,
          environment: 'sandbox',
          risk: req.tool === 'fs_patch' ? 'write' : 'read',
          sandbox,
          output: `ok:${req.tool}`,
          truncated: false,
          changed_paths: req.tool === 'fs_patch' ? [String(req.arguments.path)] : [],
          duration_ms: 20,
          is_error: false
        }
      }),
      streamChatCompletion: vi.fn(async (): Promise<StreamChatCompletionResult> => {
        turn += 1
        if (turn === 1) {
          return {
            content: '',
            reasoning: '',
            toolCalls: [
              { id: 'c1', name: 'fs_read', arguments: '{"path":"a.ts"}' },
              { id: 'c2', name: 'git_status', arguments: '{}' },
              {
                id: 'c3',
                name: 'fs_patch',
                arguments: '{"path":"a.ts","old_string":"1","new_string":"2"}'
              },
              {
                id: 'c4',
                name: 'fs_patch',
                arguments: '{"path":"a.ts","old_string":"2","new_string":"3"}'
              }
            ],
            finishReason: 'tool_calls'
          }
        }
        return { content: 'Done.', reasoning: '', toolCalls: [], finishReason: 'stop' }
      })
    })

    const runtime = new NativeAgentRuntime(broker)
    const session = await runtime.createSession({
      sessionId: 'sess-waves',
      model: { id: 'gguf:qwen3-32b.gguf', contextWindow: 32768 },
      systemPrompt: 'Prompt',
      tools,
      capabilities: inferModelCapabilities('gguf:qwen3-32b.gguf')
    })

    for await (const _ of session.run({ text: 'inspect and patch' })) {
      // consume
    }

    expect(maxConcurrentReads).toBe(2)
    expect(maxConcurrentWrites).toBe(1)
  })

  it('populates keptMessages on compacted event, proactively auto-compacts at >=82% context, and recovers from ContextOverflowError', async () => {
    const priorHistory: AgentMessage[] = Array.from({ length: 10 }, (_, idx) =>
      idx % 2 === 0
        ? { role: 'user', text: `question ${idx}: ${'x'.repeat(800)}`, timestamp: '2026-01-01T00:00:00Z' }
        : {
            role: 'assistant',
            text: `answer ${idx}: ${'y'.repeat(800)}`,
            timestamp: '2026-01-01T00:00:00Z'
          }
    )
    // Add an oversized >16KB tool output inside the tail so overflow recovery truncates it.
    priorHistory.push({
      role: 'assistant',
      text: '',
      toolCalls: [{ id: 'big-1', name: 'fs_read', arguments: { path: 'huge.log' } }],
      timestamp: '2026-01-01T00:00:00Z'
    })
    priorHistory.push({
      role: 'tool',
      toolCallId: 'big-1',
      tool: 'fs_read',
      output: 'Z'.repeat(20_000),
      isError: false,
      timestamp: '2026-01-01T00:00:00Z'
    })

    let calls = 0
    const broker = makeMockBroker({
      streamChatCompletion: vi.fn(
        async (
          _req: StreamChatCompletionRequest,
          callbacks?: StreamChatCompletionCallbacks
        ): Promise<StreamChatCompletionResult> => {
          calls += 1
          if (calls === 1) {
            throw new ContextOverflowError('request exceeds the available context size', 400)
          }
          callbacks?.onContentDelta?.('<think>recovered</think>\nRecovered cleanly.')
          return {
            content: '<think>recovered</think>\nRecovered cleanly.',
            reasoning: '',
            toolCalls: [],
            finishReason: 'stop'
          }
        }
      )
    })

    const runtime = new NativeAgentRuntime(broker)
    const session = await runtime.createSession({
      sessionId: 'sess-compact',
      model: { id: 'gguf:qwen3-32b.gguf', contextWindow: 64_000 },
      systemPrompt: 'Prompt',
      tools,
      messages: priorHistory,
      capabilities: inferModelCapabilities('gguf:qwen3-32b.gguf')
    })

    const events: AgentEvent[] = []
    for await (const event of session.run({ text: 'continue' })) {
      events.push(event)
    }

    expect(calls).toBe(2)
    const compacted = events.find((e): e is AgentCompactedEvent => e.type === 'compacted')
    expect(compacted).toBeDefined()
    expect(compacted?.keptMessages).toBeDefined()
    expect(compacted?.keptMessages?.[0]?.role).toBe('system')

    // Verify oversized tool output in kept tail was truncated below 17KB.
    const toolMsgs = session.getState().messages.filter((m) => m.role === 'tool')
    expect(toolMsgs.length).toBe(1)
    if (toolMsgs[0]?.role === 'tool') {
      expect(toolMsgs[0].output.length).toBeLessThan(17_000)
      expect(toolMsgs[0].output).toContain('[truncated oversized tool output')
    }
    expect(events.at(-1)?.type).toBe('run-completed')
  })

  it('proactively auto-compacts before a model call when estimated prompt tokens >= 82% of contextWindow', async () => {
    const heavyHistory: AgentMessage[] = Array.from({ length: 10 }, (_, idx) => ({
      role: idx % 2 === 0 ? ('user' as const) : ('assistant' as const),
      text: 'w'.repeat(1600), // 400 tokens each -> 4000+ tokens on a 4096 window (>82%)
      timestamp: '2026-01-01T00:00:00Z'
    }))

    let modelSawCompactedContext = false
    const broker = makeMockBroker({
      streamChatCompletion: vi.fn(
        async (req: StreamChatCompletionRequest): Promise<StreamChatCompletionResult> => {
          modelSawCompactedContext = req.messages.some(
            (m) => typeof m.content === 'string' && m.content.startsWith('[context] ')
          )
          return { content: 'Compact ok.', reasoning: '', toolCalls: [], finishReason: 'stop' }
        }
      )
    })

    const runtime = new NativeAgentRuntime(broker)
    const session = await runtime.createSession({
      sessionId: 'sess-proactive',
      model: { id: 'gguf:qwen3-8b.gguf', contextWindow: 4096 },
      systemPrompt: 'Prompt',
      tools,
      messages: heavyHistory,
      capabilities: inferModelCapabilities('gguf:qwen3-8b.gguf')
    })

    const events: AgentEvent[] = []
    for await (const event of session.run({ text: 'next step' })) {
      events.push(event)
    }

    expect(events.some((e) => e.type === 'compacted')).toBe(true)
    expect(modelSawCompactedContext).toBe(true)
  })

  it('validates invalid tool arguments without calling broker.execTool and handles text tool-call fallback', async () => {
    let turn = 0
    const execSpy = vi.fn(async (_req: unknown) => ({
      status: 'completed' as const,
      tool: 'fs_read',
      environment: 'sandbox' as const,
      risk: 'read' as const,
      sandbox,
      output: 'contents of a.ts',
      truncated: false,
      changed_paths: [],
      duration_ms: 1,
      is_error: false
    }))
    const broker = makeMockBroker({
      execTool: execSpy,
      streamChatCompletion: vi.fn(
        async (
          _req: StreamChatCompletionRequest,
          callbacks?: StreamChatCompletionCallbacks
        ): Promise<StreamChatCompletionResult> => {
          turn += 1
          if (turn === 1) {
            // Missing required `path` argument
            return {
              content: '',
              reasoning: '',
              toolCalls: [{ id: 'bad_1', name: 'fs_read', arguments: '{}' }],
              finishReason: 'tool_calls'
            }
          }
          if (turn === 2) {
            // Text fallback with alias `read_file` -> `fs_read`
            const raw = '<tool_call>{"name":"read_file","arguments":{"path":"a.ts"}}</tool_call>'
            callbacks?.onContentDelta?.(raw)
            return { content: raw, reasoning: '', toolCalls: [], finishReason: 'stop' }
          }
          return { content: 'Read succeeded.', reasoning: '', toolCalls: [], finishReason: 'stop' }
        }
      )
    })

    const runtime = new NativeAgentRuntime(broker)
    const session = await runtime.createSession({
      sessionId: 'sess-validate',
      model: { id: 'gguf:qwen3-8b.gguf', contextWindow: 16384 },
      systemPrompt: 'Prompt',
      tools,
      capabilities: inferModelCapabilities('gguf:qwen3-8b.gguf')
    })

    const events: AgentEvent[] = []
    for await (const event of session.run({ text: 'read a.ts' })) {
      events.push(event)
    }

    // Turn 1 failed validation before broker.execTool; Turn 2 executed fs_read once!
    expect(execSpy).toHaveBeenCalledTimes(1)
    expect(execSpy.mock.calls[0]?.[0]).toMatchObject({
      tool: 'fs_read',
      arguments: { path: 'a.ts' }
    })
    // Raw <tool_call> markup must never have streamed onto visible text-delta!
    const streamedVisible = events
      .filter((e): e is Extract<AgentEvent, { type: 'text-delta' }> => e.type === 'text-delta' && e.channel === 'text')
      .map((e) => e.delta)
      .join('')
    expect(streamedVisible).not.toContain('<tool_call>')
  })
})

describe('NativeAgentSession run lifecycle', () => {
  async function sessionFor(broker: BrokerClient): Promise<AgentSession> {
    const runtime = new NativeAgentRuntime(broker)
    return runtime.createSession({
      sessionId: 'sess-life',
      model: { id: 'gguf:qwen3-32b.gguf', contextWindow: 32768 },
      systemPrompt: 'Prompt',
      tools,
      capabilities: inferModelCapabilities('gguf:qwen3-32b.gguf')
    })
  }

  async function collect(events: AsyncIterable<AgentEvent>): Promise<AgentEvent[]> {
    const out: AgentEvent[] = []
    for await (const event of events) out.push(event)
    return out
  }

  const terminal = (events: AgentEvent[]): string[] =>
    events
      .map((event) => event.type)
      .filter((type) => type === 'run-completed' || type === 'run-cancelled' || type === 'run-failed')

  it('reports a cancel exactly once and nothing after it, even while a tool unwinds', async () => {
    let releaseTool!: () => void
    const toolHeld = new Promise<void>((resolve) => {
      releaseTool = resolve
    })
    let toolStarted!: () => void
    const toolRunning = new Promise<void>((resolve) => {
      toolStarted = resolve
    })
    const broker = makeMockBroker({
      execTool: vi.fn(async (req: { tool: string }) => {
        toolStarted()
        // Ignores the abort signal, like a daemon call that is slow to notice.
        await toolHeld
        return {
          status: 'completed',
          tool: req.tool,
          environment: 'sandbox',
          risk: 'read',
          sandbox,
          output: 'late result',
          truncated: false,
          changed_paths: [],
          duration_ms: 1,
          is_error: false
        }
      }),
      streamChatCompletion: vi.fn(async () => ({
        content: '',
        reasoning: '',
        toolCalls: [{ id: 'c1', name: 'fs_read', arguments: '{"path":"a.ts"}' }],
        finishReason: 'tool_calls'
      }))
    })
    const session = await sessionFor(broker)

    const first = collect(session.run({ text: 'read it' }))
    await toolRunning
    await session.cancel()
    const events = await first

    expect(terminal(events)).toEqual(['run-cancelled'])
    expect(events.at(-1)?.type).toBe('run-cancelled')

    // The next run waits for the old loop instead of sharing the transcript with it.
    const second = collect(session.run({ text: 'again' }))
    releaseTool()
    ;(broker.streamChatCompletion as ReturnType<typeof vi.fn>).mockImplementation(async () => ({
      content: 'Done.',
      reasoning: '',
      toolCalls: [],
      finishReason: 'stop'
    }))
    const secondEvents = await second
    expect(terminal(secondEvents)).toEqual(['run-completed'])
    expect(new Set(secondEvents.map((event) => event.runId)).size).toBe(1)
    expect(session.isBusy()).toBe(false)
  })

  it('refuses a second run while the first is still going', async () => {
    let finish!: () => void
    const held = new Promise<void>((resolve) => {
      finish = resolve
    })
    const broker = makeMockBroker({
      streamChatCompletion: vi.fn(async () => {
        await held
        return { content: 'ok', reasoning: '', toolCalls: [], finishReason: 'stop' }
      })
    })
    const session = await sessionFor(broker)
    const first = collect(session.run({ text: 'one' }))
    expect(() => session.run({ text: 'two' })).toThrow(/already running/)
    finish()
    expect(terminal(await first)).toEqual(['run-completed'])
  })

  it('retries a model request that dropped before producing output', async () => {
    let calls = 0
    const broker = makeMockBroker({
      streamChatCompletion: vi.fn(async () => {
        calls += 1
        if (calls === 1) throw new StreamInterruptedError()
        return { content: 'Recovered.', reasoning: '', toolCalls: [], finishReason: 'stop' }
      })
    })
    const session = await sessionFor(broker)
    const events = await collect(session.run({ text: 'hi' }))
    expect(calls).toBe(2)
    expect(terminal(events)).toEqual(['run-completed'])
  })

  it('does not replay a request that already streamed text', async () => {
    const broker = makeMockBroker({
      streamChatCompletion: vi.fn(
        async (_req: StreamChatCompletionRequest, callbacks?: StreamChatCompletionCallbacks) => {
          callbacks?.onContentDelta?.('Partial ans')
          throw new StreamInterruptedError()
        }
      )
    })
    const session = await sessionFor(broker)
    const events = await collect(session.run({ text: 'hi' }))
    expect(broker.streamChatCompletion).toHaveBeenCalledTimes(1)
    expect(events.at(-1)).toMatchObject({ type: 'run-failed' })
  })

  it('nudges once instead of ending the run on an empty reply', async () => {
    const requests: StreamChatCompletionRequest[] = []
    const broker = makeMockBroker({
      streamChatCompletion: vi.fn(async (req: StreamChatCompletionRequest) => {
        requests.push(req)
        if (requests.length === 1) {
          return { content: '', reasoning: '', toolCalls: [], finishReason: 'stop' }
        }
        return { content: 'Here is the answer.', reasoning: '', toolCalls: [], finishReason: 'stop' }
      })
    })
    const session = await sessionFor(broker)
    const events = await collect(session.run({ text: 'hi' }))

    expect(requests).toHaveLength(2)
    const nudge = requests[1]!.messages.at(-1)
    expect(nudge).toMatchObject({ role: 'user' })
    expect(String(nudge?.content)).toContain('Your last reply was empty')
    // The nudge is never persisted into the transcript.
    expect(
      session.getState().messages.some((message) => 'text' in message && message.text.includes('last reply was empty'))
    ).toBe(false)
    expect(terminal(events)).toEqual(['run-completed'])
  })
})

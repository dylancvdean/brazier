import { describe, expect, it, vi } from 'vitest'

import type { BrokerClient, StreamChatCompletionRequest } from './brokerClient'
import { inferModelCapabilities } from './modelCompat'
import { NativeAgentRuntime } from '../native/nativeRuntime'
import {
  buildSubagentMetadata,
  childEnabledTools,
  collectSpawnPrompts,
  DEFAULT_MAX_SUBAGENTS,
  isSubagentSession,
  llamaParallelSlots,
  resolveMaxSubagents,
  resolveSubagentExecutionMode,
  resolveSubagentModel,
  SPAWN_SUBAGENT_TOOL,
  summarizeSubagentResult
} from './subagent'
import type { AgentEvent, AgentToolDefinition, SandboxDescription } from './types'

const sandbox: SandboxDescription = {
  backend: 'seatbelt',
  profile: 'workspace',
  isolated: true,
  network: false,
  detail: 'test sandbox'
}

const toolsCatalog: AgentToolDefinition[] = [
  {
    name: 'fs_read',
    label: 'Read',
    description: 'Read a file.',
    inputSchema: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
    risk: 'read',
    executes: false,
    needsWorkspace: true,
    defaultEnvironment: 'sandbox'
  },
  {
    name: SPAWN_SUBAGENT_TOOL,
    label: 'Subagent',
    description: 'Spawn subagents.',
    inputSchema: {
      type: 'object',
      properties: {
        prompt: { type: 'string' },
        prompts: { type: 'array' },
        model: { type: 'string' }
      },
      required: []
    },
    risk: 'execute',
    executes: false,
    needsWorkspace: false,
    defaultEnvironment: 'sandbox'
  }
]

function makeSubagentTestBroker(options: {
  parentModel: string
  profile: {
    subagent_model?: string | null
    context_size?: number | null
    max_subagents?: number | null
    parallel_subagents?: boolean | null
  } | null
  childDelayMs?: number
}) {
  let childSeq = 0
  const sessions = new Map<string, { model: string; metadata?: Record<string, unknown> }>()
  sessions.set('parent-1', { model: options.parentModel })

  let inFlightChildren = 0
  let maxObservedConcurrentChildren = 0
  const observedChildCalls: Array<{ model: string; slot: number; prompt: string }> = []
  let parentTurn = 0

  const broker = {
    capabilities: vi.fn(async () => ({
      sandbox: { backend: 'seatbelt', isolated: true, detail: 'test sandbox' }
    })),
    session: vi.fn(async (id: string) => ({
      session: {
        id,
        title: id,
        workspace_path: '/tmp',
        model: sessions.get(id)?.model ?? options.parentModel,
        runtime_id: 'simple',
        permission_mode: 'ask' as const,
        permission_settings: {
          auto_approve_sandboxed_actions: true,
          auto_approve_host_actions: false
        },
        enabled_tools: null,
        last_run_status: 'idle',
        runtime_metadata: sessions.get(id)?.metadata ?? null,
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
    createSession: vi.fn(async (input: { model: string }) => {
      childSeq += 1
      const id = `child-${childSeq}`
      sessions.set(id, { model: input.model })
      return {
        id,
        title: id,
        workspace_path: '/tmp',
        model: input.model,
        runtime_id: 'simple',
        permission_mode: 'ask' as const,
        permission_settings: {
          auto_approve_sandboxed_actions: true,
          auto_approve_host_actions: false
        },
        enabled_tools: ['fs_read'],
        last_run_status: 'idle',
        runtime_metadata: null,
        created_at: '2026-01-01T00:00:00Z',
        updated_at: '2026-01-01T00:00:00Z'
      }
    }),
    updateSession: vi.fn(async (id: string, update: Record<string, unknown>) => {
      const entry = sessions.get(id)
      if (entry && update.runtime_metadata) {
        entry.metadata = update.runtime_metadata as Record<string, unknown>
      }
      return { id }
    }),
    deleteSession: vi.fn(async () => undefined),
    systemPrompt: vi.fn(async () => ({ system_prompt: 'You are helpful.', tools: ['fs_read'] })),
    tools: vi.fn(async () => toolsCatalog),
    textProfile: vi.fn(async () => options.profile),
    runtimeInferenceSettings: vi.fn(async () => ({
      context_size: 16384,
      enable_reasoning: false,
      drop_reasoning_between_turns: false
    })),
    appendMessages: vi.fn(async () => undefined),
    openAiBaseUrl: () => 'http://127.0.0.1:1710/v1',
    apiKey: () => 'brazier-local',
    streamChatCompletion: vi.fn(async (req: StreamChatCompletionRequest) => {
      const lastMsg = req.messages.at(-1)
      const isParent = req.tools?.some((t) => t.function.name === SPAWN_SUBAGENT_TOOL) ?? false
      if (isParent) {
        parentTurn += 1
        if (parentTurn === 1) {
          return {
            content: '',
            reasoning: '',
            toolCalls: [
              {
                id: 'call_spawn_1',
                name: SPAWN_SUBAGENT_TOOL,
                arguments: JSON.stringify({ prompts: ['task 1', 'task 2', 'task 3'] })
              }
            ],
            finishReason: 'tool_calls'
          }
        }
        return {
          content: 'Synthesized child findings.',
          reasoning: '',
          toolCalls: [],
          finishReason: 'stop'
        }
      }

      const promptText =
        typeof lastMsg?.content === 'string' ? lastMsg.content : JSON.stringify(lastMsg?.content)
      inFlightChildren += 1
      maxObservedConcurrentChildren = Math.max(maxObservedConcurrentChildren, inFlightChildren)
      observedChildCalls.push({
        model: req.model,
        slot: req.slot ?? 0,
        prompt: promptText
      })
      await new Promise((resolve) => setTimeout(resolve, options.childDelayMs ?? 15))
      inFlightChildren -= 1
      return {
        content: `Completed ${promptText}`,
        reasoning: '',
        toolCalls: [],
        finishReason: 'stop'
      }
    })
  }

  return {
    broker: broker as unknown as BrokerClient,
    observedChildCalls,
    getMaxConcurrent: () => maxObservedConcurrentChildren
  }
}

describe('subagent helpers', () => {
  it('defaults max subagents to 2 and clamps the profile value', () => {
    expect(resolveMaxSubagents(null)).toBe(DEFAULT_MAX_SUBAGENTS)
    expect(resolveMaxSubagents({})).toBe(2)
    expect(resolveMaxSubagents({ max_subagents: 4 })).toBe(4)
    expect(resolveMaxSubagents({ max_subagents: 0 })).toBe(1)
    expect(resolveMaxSubagents({ max_subagents: 99 })).toBe(8)
  })

  it('derives llama parallel slot count from the profile toggle', () => {
    expect(llamaParallelSlots(null)).toBe(1)
    expect(llamaParallelSlots({ parallel_subagents: false, max_subagents: 4 })).toBe(1)
    expect(llamaParallelSlots({ parallel_subagents: true, max_subagents: 2 })).toBe(3)
  })

  it('resolves execution mode explicitly across independent, parallel llama slots, and serialized single-slot', () => {
    expect(
      resolveSubagentExecutionMode({
        parentModelId: 'gguf:qwen3.gguf',
        targetModelId: 'remote:openai/gpt-4o',
        parallelSlots: 1,
        maxSubagents: 3
      })
    ).toEqual({ kind: 'independent_backend', concurrency: 3, slot: 0 })

    expect(
      resolveSubagentExecutionMode({
        parentModelId: 'gguf:parent.gguf',
        targetModelId: 'gguf:small-child.gguf',
        parallelSlots: 1,
        maxSubagents: 2
      })
    ).toEqual({ kind: 'independent_backend', concurrency: 2, slot: 0 })

    expect(
      resolveSubagentExecutionMode({
        parentModelId: 'gguf:qwen3.gguf',
        targetModelId: 'gguf:qwen3.gguf',
        parallelSlots: 3,
        maxSubagents: 4
      })
    ).toEqual({ kind: 'parallel_llama_slots', parallelSlots: 3, childSlotCount: 2, concurrency: 2 })

    expect(
      resolveSubagentExecutionMode({
        parentModelId: 'gguf:qwen3.gguf',
        targetModelId: 'gguf:qwen3.gguf',
        parallelSlots: 1,
        maxSubagents: 2
      })
    ).toEqual({ kind: 'serialized_shared_slot', concurrency: 1, slot: 0 })

    expect(
      resolveSubagentExecutionMode({
        parentModelId: 'mlx:qwen3-8b',
        targetModelId: 'mlx:qwen3-8b',
        parallelSlots: 3,
        maxSubagents: 2
      })
    ).toEqual({ kind: 'serialized_shared_slot', concurrency: 1, slot: 0 })
  })

  it('resolves model as tool arg → profile → parent', () => {
    expect(
      resolveSubagentModel(
        'gguf:child.gguf',
        { subagent_model: 'gguf:profile.gguf' },
        'gguf:parent.gguf'
      )
    ).toBe('gguf:child.gguf')
    expect(
      resolveSubagentModel('  ', { subagent_model: 'gguf:profile.gguf' }, 'gguf:parent.gguf')
    ).toBe('gguf:profile.gguf')
    expect(resolveSubagentModel(undefined, null, 'gguf:parent.gguf')).toBe('gguf:parent.gguf')
  })

  it('strips spawn_subagent from child tools so nesting is impossible', () => {
    expect(childEnabledTools(['fs_read', SPAWN_SUBAGENT_TOOL, 'shell_run'])).toEqual([
      'fs_read',
      'shell_run'
    ])
  })

  it('detects subagent sessions from runtime metadata', () => {
    expect(isSubagentSession({ kind: 'subagent', parent_session_id: 'p1' })).toBe(true)
    expect(isSubagentSession({ worktree: { source_path: '/a', path: '/b', branch: 't' } })).toBe(
      false
    )
  })

  it('copies worktree metadata onto the child', () => {
    const meta = buildSubagentMetadata('parent-1', {
      worktree: { source_path: '/src', path: '/wt', branch: 'agent/task' }
    })
    expect(meta).toEqual({
      kind: 'subagent',
      parent_session_id: 'parent-1',
      worktree: { source_path: '/src', path: '/wt', branch: 'agent/task' }
    })
  })

  it('collects a single prompt or a concurrent prompts list', () => {
    expect(collectSpawnPrompts({ prompt: ' one ' })).toEqual(['one'])
    expect(collectSpawnPrompts({ prompts: ['a', '  ', 'b'] })).toEqual(['a', 'b'])
    expect(collectSpawnPrompts({ prompt: 'ignored', prompts: ['x', 'y'] })).toEqual(['x', 'y'])
    expect(collectSpawnPrompts({})).toEqual([])
  })

  it('summarizes the last assistant reply', () => {
    expect(
      summarizeSubagentResult([
        { role: 'user', text: 'do it' },
        { role: 'assistant', text: '  All done.  ' }
      ])
    ).toBe('All done.')
    expect(summarizeSubagentResult([], { failed: true, error: 'boom' })).toBe(
      'Subagent failed: boom'
    )
  })
})

describe('NativeAgentRuntime spawn_subagent state machine', () => {
  it('runs single-slot local subagents sequentially on slot 0 instead of failing', async () => {
    const { broker, observedChildCalls, getMaxConcurrent } = makeSubagentTestBroker({
      parentModel: 'gguf:qwen3-8b.gguf',
      profile: { parallel_subagents: false, max_subagents: 2 }
    })
    const runtime = new NativeAgentRuntime(broker)
    const session = await runtime.createSession({
      sessionId: 'parent-1',
      model: { id: 'gguf:qwen3-8b.gguf', contextWindow: 16384 },
      systemPrompt: 'Parent prompt',
      tools: toolsCatalog,
      capabilities: inferModelCapabilities('gguf:qwen3-8b.gguf'),
      preloaded: {
        session: {
          id: 'parent-1',
          title: 'Parent',
          permission_mode: 'ask',
          permission_settings: {
            auto_approve_sandboxed_actions: true,
            auto_approve_host_actions: false
          },
          created_at: '2026-01-01T00:00:00Z',
          updated_at: '2026-01-01T00:00:00Z'
        },
        tool_executions: [],
        sandbox
      },
      preloadedInference: { context_size: 16384, enable_reasoning: false }
    })

    const events: AgentEvent[] = []
    for await (const event of session.run({ text: 'fan out' })) {
      events.push(event)
    }

    expect(getMaxConcurrent()).toBe(1)
    expect(observedChildCalls).toHaveLength(3)
    expect(observedChildCalls.every((call) => call.slot === 0)).toBe(true)
    expect(events.filter((e) => e.type === 'tool-failed')).toHaveLength(0)
    expect(events.at(-1)?.type).toBe('run-completed')
  })

  it('batches multi-slot llama.cpp subagents across slots 1..parallelSlots-1 when prompts exceed slots', async () => {
    const { broker, observedChildCalls, getMaxConcurrent } = makeSubagentTestBroker({
      parentModel: 'gguf:qwen3-8b.gguf',
      profile: { parallel_subagents: true, max_subagents: 2 }, // parallelSlots = 3 -> child slots 1, 2
      childDelayMs: 25
    })
    const runtime = new NativeAgentRuntime(broker)
    const session = await runtime.createSession({
      sessionId: 'parent-1',
      model: { id: 'gguf:qwen3-8b.gguf', contextWindow: 16384 },
      systemPrompt: 'Parent prompt',
      tools: toolsCatalog,
      capabilities: inferModelCapabilities('gguf:qwen3-8b.gguf')
    })

    const events: AgentEvent[] = []
    for await (const event of session.run({ text: 'fan out' })) {
      events.push(event)
    }

    expect(getMaxConcurrent()).toBe(2)
    expect(observedChildCalls).toHaveLength(3)
    // Children must only use slots 1 and 2, never parent slot 0.
    expect(observedChildCalls.every((call) => call.slot === 1 || call.slot === 2)).toBe(true)
    expect(events.filter((e) => e.type === 'tool-failed')).toHaveLength(0)
  })

  it('runs remote or distinct-model subagents concurrently up to maxSubagents on slot 0', async () => {
    const { broker, observedChildCalls, getMaxConcurrent } = makeSubagentTestBroker({
      parentModel: 'gguf:qwen3-8b.gguf',
      profile: {
        subagent_model: 'remote:openai/gpt-4o-mini',
        parallel_subagents: false,
        max_subagents: 3
      },
      childDelayMs: 25
    })
    const runtime = new NativeAgentRuntime(broker)
    const session = await runtime.createSession({
      sessionId: 'parent-1',
      model: { id: 'gguf:qwen3-8b.gguf', contextWindow: 16384 },
      systemPrompt: 'Parent prompt',
      tools: toolsCatalog,
      capabilities: inferModelCapabilities('gguf:qwen3-8b.gguf')
    })

    for await (const _ of session.run({ text: 'fan out remote' })) {
      // consume stream
    }

    expect(getMaxConcurrent()).toBe(3)
    expect(observedChildCalls).toHaveLength(3)
    expect(
      observedChildCalls.every(
        (call) => call.model === 'remote:openai/gpt-4o-mini' && call.slot === 0
      )
    ).toBe(true)
  })
})

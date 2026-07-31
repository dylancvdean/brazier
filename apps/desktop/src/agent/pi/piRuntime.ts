/**
 * Pi runtime adapter.
 *
 * This is the ONLY file in the application that imports Pi. Everything outside
 * it speaks the application's own types (`../core/types.ts`), so replacing the
 * runtime means writing a sibling of this file and nothing else. A test
 * (`../boundary.test.ts`) enforces that rule.
 *
 * Division of labour: Pi owns the orchestration loop, tool-call parsing,
 * streaming, and completion detection. The application owns tools, policy,
 * sandboxing, execution, persistence, and the event stream.
 */

import { Agent, type AgentEvent as PiAgentEvent, type AgentTool } from '@earendil-works/pi-agent-core'
import type {
  AssistantMessage,
  Message as PiMessage,
  Model,
  TSchema,
  ToolResultMessage
} from '@earendil-works/pi-ai'
import { streamSimple } from '@earendil-works/pi-ai/api/openai-completions'

import type { BrokerClient } from '../core/brokerClient'
import { mergeSummary, renderTranscript, requestModelSummary } from '../core/compaction'
import { repairToolArguments } from '../core/modelCompat'
import { prefillListener } from './prefillProgress'
import { accumulate, describeSummary, emptySummary } from '../core/runSummary'
import {
  buildSubagentMetadata,
  childEnabledTools,
  collectSpawnPrompts,
  isSubagentSession,
  llamaParallelSlots,
  resolveMaxSubagents,
  resolveSubagentModel,
  SPAWN_SUBAGENT_TOOL,
  summarizeSubagentResult
} from '../core/subagent'
import { AgentToolExecutor, EventSequencer, type ExecuteToolRequest, type ToolExecutionOutcome } from '../core/toolExecutor'
import type {
  AgentCompactionState,
  AgentEvent,
  AgentMessage,
  AgentModelReference,
  AgentRunStatus,
  AgentRuntime,
  AgentRuntimeDescriptor,
  AgentSession,
  AgentSessionState,
  AgentToolCallSummary,
  AgentToolDefinition,
  AgentUserInput,
  CreateAgentSessionOptions,
  SandboxDescription
} from '../core/types'

/** Messages kept verbatim when compaction rewrites the transcript. */
const COMPACTION_KEEP_TAIL = 6

const DESCRIPTOR: AgentRuntimeDescriptor = {
  id: 'pi',
  name: 'Pi',
  version: '0.82.1',
  capabilities: {
    streaming: true,
    toolCalls: true,
    compaction: true,
    cancellation: true,
    sessionRestore: true
  }
}

type ThinkingLevel = 'off' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max'

/** Build a Pi model that points at the daemon's OpenAI-compatible endpoint. */
function piModel(
  model: AgentModelReference,
  baseUrl: string,
  reasoningEnabled: boolean
): Model<'openai-completions'> {
  return {
    id: model.id,
    name: model.name ?? model.id,
    api: 'openai-completions',
    provider: 'brazier',
    baseUrl,
    reasoning: reasoningEnabled,
    input: ['text'],
    // Local inference has no per-token price; the daemon reports no usage cost.
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: model.contextWindow ?? 8_192,
    maxTokens: model.maxTokens ?? 4_096
  }
}

function thinkingLevelFor(reasoningEnabled: boolean): ThinkingLevel {
  return reasoningEnabled ? 'medium' : 'off'
}

/** Convert an application message into the shape Pi hands to a model. */
function toPiMessage(message: AgentMessage, modelId: string): PiMessage | undefined {
  switch (message.role) {
    case 'user':
      return {
        role: 'user',
        content: message.text,
        timestamp: Date.parse(message.timestamp) || Date.now()
      }
    case 'system':
      // Compaction summaries re-enter as user context; Pi has no system role in
      // the transcript (the system prompt is separate).
      return {
        role: 'user',
        content: `[context] ${message.text}`,
        timestamp: Date.parse(message.timestamp) || Date.now()
      }
    case 'assistant': {
      const content: AssistantMessage['content'] = []
      // Always keep thinking in the transcript. Whether it is sent on the next
      // turn is decided at request time from the live drop_reasoning setting.
      // thinkingSignature must be reasoning_content so openai-completions emits
      // it as that field (rather than dropping or inlining it as plain text).
      if (message.reasoning?.trim()) {
        content.push({
          type: 'thinking',
          thinking: message.reasoning,
          thinkingSignature: 'reasoning_content'
        })
      }
      if (message.text) content.push({ type: 'text', text: message.text })
      for (const call of message.toolCalls ?? []) {
        content.push({
          type: 'toolCall',
          id: call.id,
          name: call.name,
          arguments: call.arguments
        })
      }
      const assistant: AssistantMessage = {
        role: 'assistant',
        content,
        api: 'openai-completions',
        provider: 'brazier',
        // Must match the active model id so Pi keeps thinking blocks (same-model
        // path) instead of converting them to plain text.
        model: modelId,
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }
        },
        stopReason: (message.toolCalls?.length ?? 0) > 0 ? 'toolUse' : 'stop',
        timestamp: Date.parse(message.timestamp) || Date.now()
      }
      return assistant
    }
    case 'tool': {
      const result: ToolResultMessage = {
        role: 'toolResult',
        toolCallId: message.toolCallId,
        toolName: message.tool,
        content: [{ type: 'text', text: message.output }],
        isError: message.isError,
        timestamp: Date.parse(message.timestamp) || Date.now()
      }
      return result
    }
    default:
      return undefined
  }
}

/** Drop thinking parts from one message. */
function stripThinkingParts(message: PiMessage): PiMessage {
  if (message.role !== 'assistant' || !Array.isArray(message.content)) return message
  const content = message.content.filter((part) => part.type !== 'thinking')
  return { ...message, content }
}

function assistantHasToolCalls(message: AssistantMessage): boolean {
  return message.content.some((part) => part.type === 'toolCall')
}

/**
 * Omit thinking from turns before the latest user message. Matches the daemon's
 * drop_reasoning_between_turns rule so in-turn tool rounds still keep thinking.
 * Harmony models also keep thinking on prior assistant tool-call turns.
 */
function stripPriorTurnThinking(messages: PiMessage[], harmony: boolean): PiMessage[] {
  let lastUser = -1
  for (let index = 0; index < messages.length; index += 1) {
    if (messages[index]?.role === 'user') lastUser = index
  }
  if (lastUser < 0) return messages
  return messages.map((message, index) => {
    if (index > lastUser) return message
    if (
      harmony &&
      message.role === 'assistant' &&
      assistantHasToolCalls(message)
    ) {
      return message
    }
    return stripThinkingParts(message)
  })
}

/** Text of an assistant message, ignoring thinking and tool calls. */
function assistantText(message: AssistantMessage): string {
  return message.content
    .filter((part) => part.type === 'text')
    .map((part) => {
      // Some runtimes emit a null text part on tool-only turns; coerce safely
      // so join() cannot turn it into the literal string "null".
      const text = (part as { text?: unknown }).text
      return typeof text === 'string' ? text : ''
    })
    .join('')
}

function assistantReasoning(message: AssistantMessage): string | undefined {
  const thinking = message.content
    .filter((part): part is { type: 'thinking'; thinking: string } => part.type === 'thinking')
    .map((part) => part.thinking)
    .join('')
  return thinking.length > 0 ? thinking : undefined
}

function assistantToolCalls(message: AssistantMessage): AgentToolCallSummary[] {
  return message.content
    .filter(
      (part): part is { type: 'toolCall'; id: string; name: string; arguments: Record<string, unknown> } =>
        part.type === 'toolCall'
    )
    .map((part) => ({ id: part.id, name: part.name, arguments: part.arguments ?? {} }))
}

/** Convert a Pi transcript entry into the application's persisted shape. */
function fromPiMessage(message: unknown): AgentMessage | undefined {
  const candidate = message as { role?: string }
  const now = new Date().toISOString()
  if (candidate.role === 'user') {
    const user = message as { content: unknown; timestamp?: number }
    const text =
      typeof user.content === 'string'
        ? user.content
        : Array.isArray(user.content)
          ? user.content
              .filter(
                (part): part is { type: 'text'; text: string } =>
                  typeof part === 'object' && part !== null && (part as { type?: string }).type === 'text'
              )
              .map((part) => part.text)
              .join('')
          : ''
    return { role: 'user', text, timestamp: isoFrom(user.timestamp, now) }
  }
  if (candidate.role === 'assistant') {
    const assistant = message as AssistantMessage
    return {
      role: 'assistant',
      text: assistantText(assistant),
      reasoning: assistantReasoning(assistant),
      toolCalls: assistantToolCalls(assistant),
      error: assistant.errorMessage,
      timestamp: isoFrom(assistant.timestamp, now)
    }
  }
  if (candidate.role === 'toolResult') {
    const result = message as ToolResultMessage
    return {
      role: 'tool',
      toolCallId: result.toolCallId,
      tool: result.toolName,
      output: result.content
        .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
        .map((part) => part.text)
        .join('\n'),
      isError: result.isError,
      timestamp: isoFrom(result.timestamp, now)
    }
  }
  return undefined
}

/** True for the three message roles a model can actually be sent. */
function isLlmMessage(message: unknown): message is PiMessage {
  const role = (message as { role?: unknown } | null)?.role
  return role === 'user' || role === 'assistant' || role === 'toolResult'
}

function isoFrom(timestamp: number | undefined, fallback: string): string {
  if (typeof timestamp !== 'number' || !Number.isFinite(timestamp)) return fallback
  return new Date(timestamp).toISOString()
}

/** A queue that turns pushed events into an async iterable for one run. */
class EventQueue {
  private readonly buffer: AgentEvent[] = []
  private waiting?: (value: IteratorResult<AgentEvent>) => void
  private closed = false

  push(event: AgentEvent): void {
    if (this.closed) return
    if (this.waiting) {
      const resolve = this.waiting
      this.waiting = undefined
      resolve({ value: event, done: false })
      return
    }
    this.buffer.push(event)
  }

  close(): void {
    this.closed = true
    if (this.waiting) {
      const resolve = this.waiting
      this.waiting = undefined
      resolve({ value: undefined as unknown as AgentEvent, done: true })
    }
  }

  iterator(): AsyncIterable<AgentEvent> {
    const queue = this
    return {
      [Symbol.asyncIterator](): AsyncIterator<AgentEvent> {
        return {
          next(): Promise<IteratorResult<AgentEvent>> {
            const next = queue.buffer.shift()
            if (next) return Promise.resolve({ value: next, done: false })
            if (queue.closed) {
              return Promise.resolve({ value: undefined as unknown as AgentEvent, done: true })
            }
            return new Promise((resolve) => {
              queue.waiting = resolve
            })
          }
        }
      }
    }
  }
}

class PiAgentSession implements AgentSession {
  readonly id: string
  private readonly broker: BrokerClient
  private readonly agent: Agent
  private readonly sequencer = new EventSequencer()
  private readonly executor: AgentToolExecutor
  private state: AgentSessionState
  private definitions: AgentToolDefinition[]
  private enabledTools: string[]
  private readonly sandbox: SandboxDescription
  private readonly maxToolsPerTurn?: number
  private readonly harmony: boolean
  private readonly llamaSlot: number
  private readonly spawnSubagent: (
    parent: PiAgentSession,
    request: ExecuteToolRequest
  ) => Promise<ToolExecutionOutcome>
  private readonly cancelChildren: () => Promise<void>
  private readonly onDisposed: () => void
  private reasoningEnabled: boolean
  /** Live flag: refreshed from runtime settings at the start of every run. */
  private dropReasoningBetweenTurns: boolean
  private toolsThisTurn = 0
  private queue?: EventQueue
  private currentRunId?: string
  private disposed = false
  /**
   * Pi can emit several message_end events back-to-back. SQLite append uses a
   * read-then-write transaction to allocate sequence numbers, so overlapping
   * appends from one session can deadlock while upgrading their read locks.
   */
  private persistence = Promise.resolve()

  constructor(options: {
    broker: BrokerClient
    state: AgentSessionState
    definitions: AgentToolDefinition[]
    systemPrompt: string
    sandbox: SandboxDescription
    capabilities: CreateAgentSessionOptions['capabilities']
    reasoningEnabled: boolean
    dropReasoningBetweenTurns: boolean
    spawnSubagent: (
      parent: PiAgentSession,
      request: ExecuteToolRequest
    ) => Promise<ToolExecutionOutcome>
    cancelChildren: () => Promise<void>
    onDisposed: () => void
    llamaSlot?: number
  }) {
    this.id = options.state.id
    this.broker = options.broker
    this.state = options.state
    this.definitions = options.definitions
    this.enabledTools = options.state.enabledTools ?? options.definitions.map((tool) => tool.name)
    this.sandbox = options.sandbox
    this.maxToolsPerTurn = options.capabilities.maxToolsPerTurn
    this.harmony = options.capabilities.harmony
    this.llamaSlot = options.llamaSlot ?? 0
    this.reasoningEnabled = options.reasoningEnabled
    this.dropReasoningBetweenTurns = options.dropReasoningBetweenTurns
    this.spawnSubagent = options.spawnSubagent
    this.cancelChildren = options.cancelChildren
    this.onDisposed = options.onDisposed
    this.executor = new AgentToolExecutor({
      broker: options.broker,
      sessionId: options.state.id,
      emit: (event) => this.queue?.push(event),
      sequencer: this.sequencer,
      definitions: options.definitions,
      sandbox: options.sandbox
    })
    this.executor.setLocalHandler(SPAWN_SUBAGENT_TOOL, (request) =>
      this.spawnSubagent(this, request)
    )

    const modelId = options.state.model.id
    this.agent = new Agent({
      // Every model request goes to the daemon's OpenAI-compatible endpoint,
      // so local and remote models both work with no provider configuration.
      streamFn: (model, context, options) => {
        const progress = prefillListener((event) => {
          this.emit({
            ...this.base('prefill-progress'),
            total: event.total,
            cached: event.cached,
            processed: event.processed,
            elapsedMs: event.elapsed_ms,
            contextTotal: event.context_total
          } as AgentEvent)
        })
        return streamSimple(model as Model<'openai-completions'>, context, {
          ...options,
          headers: {
            ...options?.headers,
            ...progress.headers,
            'x-brazier-mode': 'agent',
            'x-brazier-slot': String(this.llamaSlot)
          }
        })
      },
      // The transcript only ever holds LLM messages, because the application
      // converts its own shapes before they reach the agent. Anything else a
      // runtime version adds for its own bookkeeping is dropped here rather
      // than sent to a model. Drop-reasoning is applied here from the live flag
      // so toggling the setting takes effect on the next model call.
      convertToLlm: (messages) => {
        const llm = messages.filter(isLlmMessage)
        return this.dropReasoningBetweenTurns
          ? stripPriorTurnThinking(llm, this.harmony)
          : llm
      },
      getApiKey: () => options.broker.apiKey(),
      sessionId: options.state.id,
      toolExecution: options.capabilities.parallelToolCalling ? 'parallel' : 'sequential',
      initialState: {
        systemPrompt: options.systemPrompt,
        model: piModel(
          options.state.model,
          options.broker.openAiBaseUrl(),
          options.reasoningEnabled
        ),
        thinkingLevel: thinkingLevelFor(options.reasoningEnabled),
        tools: this.buildTools(),
        messages: options.state.messages
          .map((message) => toPiMessage(message, modelId))
          .filter((message): message is PiMessage => Boolean(message))
      }
    })

    this.agent.subscribe((event) => this.onPiEvent(event))
  }

  /**
   * Replace the in-memory transcript with what the daemon stored. Used when
   * reopening a session so the UI history and the model's context stay aligned.
   * Reasoning stays in the transcript; send-time filtering uses the live setting.
   */
  rehydrate(messages: AgentMessage[], systemPrompt?: string): void {
    if (this.disposed) {
      throw new Error('Cannot rehydrate a disposed agent session.')
    }
    this.state = { ...this.state, messages }
    const modelId = this.state.model.id
    this.agent.state.messages = messages
      .map((message) => toPiMessage(message, modelId))
      .filter((message): message is PiMessage => Boolean(message))
    if (systemPrompt !== undefined) {
      this.agent.state.systemPrompt = systemPrompt
    }
  }

  /** Pull the current drop-reasoning / thinking prefs (next model call uses them). */
  async refreshInferencePrefs(): Promise<void> {
    try {
      const settings = await this.broker.runtimeInferenceSettings()
      this.dropReasoningBetweenTurns = settings.drop_reasoning_between_turns ?? false
      const reasoningEnabled = settings.enable_reasoning ?? this.reasoningEnabled
      if (reasoningEnabled !== this.reasoningEnabled) {
        this.reasoningEnabled = reasoningEnabled
        this.agent.state.model = piModel(
          this.state.model,
          this.broker.openAiBaseUrl(),
          reasoningEnabled
        )
        this.agent.state.thinkingLevel = thinkingLevelFor(reasoningEnabled)
      }
    } catch {
      // Keep the last known prefs if the daemon is briefly unreachable.
    }
  }

  /** Forward a child-session event onto this parent's stream (approvals, etc.). */
  forwardEvent(event: AgentEvent): void {
    this.queue?.push(event)
  }

  /** Application tools, wrapped so Pi can call them but not bypass them. */
  private buildTools(): AgentTool<TSchema>[] {
    return this.definitions
      .filter((definition) => this.enabledTools.includes(definition.name))
      .map((definition) => {
        const tool: AgentTool<TSchema> = {
          name: definition.name,
          label: definition.label,
          description: definition.description,
          // The daemon's JSON Schema is used as-is. Pi validates plain JSON
          // Schema and coerces primitives before calling execute.
          parameters: definition.inputSchema as unknown as TSchema,
          prepareArguments: (args) => repairToolArguments(args, definition),
          // Shell/process tools stay sequential. spawn_subagent must NOT — otherwise
          // concurrent children serialize even when the model emits several calls.
          executionMode:
            definition.name === SPAWN_SUBAGENT_TOOL
              ? undefined
              : definition.executes
                ? 'sequential'
                : undefined,
          execute: async (toolCallId, params, signal) => {
            const runId = this.currentRunId ?? 'run'
            this.toolsThisTurn += 1
            if (this.maxToolsPerTurn !== undefined && this.toolsThisTurn > this.maxToolsPerTurn) {
              // Weak models fare better one call at a time; tell it plainly
              // rather than running a batch it cannot reason about.
              return {
                content: [
                  {
                    type: 'text',
                    text: `Only ${this.maxToolsPerTurn} tool call per turn is allowed for this model. Make the next call after reading this result.`
                  }
                ],
                details: { skipped: true }
              }
            }
            const args = (params ?? {}) as Record<string, unknown>
            const outcome = await this.executor.execute({
              runId,
              toolCallId,
              tool: definition.name,
              args,
              reason: typeof args.reason === 'string' ? args.reason : undefined,
              signal
            })
            return {
              content: [
                { type: 'text' as const, text: outcome.output },
                ...outcome.images.map((image) => ({
                  type: 'image' as const,
                  data: image.data,
                  mimeType: image.mimeType
                }))
              ],
              details: {
                environment: outcome.environment,
                sandbox: outcome.sandbox,
                changedPaths: outcome.changedPaths,
                exitCode: outcome.exitCode,
                truncated: outcome.truncated,
                artifactId: outcome.artifactId,
                denied: outcome.denied
              },
              isError: outcome.isError
            } as never
          }
        }
        return tool
      })
  }

  private emit(event: AgentEvent): void {
    this.queue?.push(event)
  }

  /** Keep this session's transcript and status writes in emission order. */
  private enqueuePersistence<T>(task: () => Promise<T>): Promise<T> {
    const next = this.persistence.then(task, task)
    this.persistence = next.then(
      () => undefined,
      () => undefined
    )
    return next
  }

  private base(type: AgentEvent['type']): {
    type: AgentEvent['type']
    sessionId: string
    runId: string
    timestamp: string
    sequence: number
  } {
    return {
      type,
      sessionId: this.id,
      runId: this.currentRunId ?? 'run',
      timestamp: new Date().toISOString(),
      sequence: this.sequencer.take()
    }
  }

  /** Translate Pi's lifecycle events into application events. */
  private onPiEvent(event: PiAgentEvent): void {
    switch (event.type) {
      case 'turn_start': {
        this.toolsThisTurn = 0
        return
      }
      case 'message_update': {
        const inner = event.assistantMessageEvent
        if (inner.type === 'text_delta') {
          // Tool-only OpenAI messages use `content: null`; adapters and weak
          // models sometimes surface that as visible assistant text.
          const delta = typeof inner.delta === 'string' ? inner.delta : ''
          if (delta && delta.trim() !== 'null') {
            this.emit({ ...this.base('text-delta'), delta, channel: 'text' } as AgentEvent)
          }
        } else if (inner.type === 'thinking_delta') {
          const delta = typeof inner.delta === 'string' ? inner.delta : ''
          if (delta) {
            this.emit({
              ...this.base('text-delta'),
              delta,
              channel: 'reasoning'
            } as AgentEvent)
          }
        }
        return
      }
      case 'message_end': {
        const message = fromPiMessage(event.message)
        if (!message) return
        // Tool results are already reported by the executor's own events; the
        // transcript entry is still persisted below.
        if (message.role !== 'tool') {
          this.emit({ ...this.base('message-committed'), message } as AgentEvent)
        }
        this.state = { ...this.state, messages: [...this.state.messages, message] }
        void this.enqueuePersistence(() => this.broker.appendMessages(this.id, [message])).catch(
          () => {
            // Persistence failures must not abort a run in progress; the run
            // still streams and the UI still shows it.
          }
        )
        return
      }
      default:
        return
    }
  }

  run(input: AgentUserInput): AsyncIterable<AgentEvent> {
    const runId = `run-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
    this.currentRunId = runId
    const queue = new EventQueue()
    this.queue = queue
    this.executor.approvalsRequested = 0
    let summary = emptySummary()
    const tap = (event: AgentEvent): void => {
      summary = accumulate(summary, event)
    }

    // Wrap push so every emitted event also feeds the run summary.
    const originalPush = queue.push.bind(queue)
    queue.push = (event: AgentEvent): void => {
      tap(event)
      originalPush(event)
    }

    this.emit({ ...this.base('run-started') } as AgentEvent)
    void this.setRunStatus('running')

    // Refresh drop-reasoning (and thinking enablement) so Inference menu changes
    // apply on the next turn without reopening the session.
    const promise = this.refreshInferencePrefs()
      .then(() => this.agent.prompt(input.text))
      .then(async () => {
        const error = this.agent.state.errorMessage
        if (error) {
          this.emit({ ...this.base('run-failed'), error } as AgentEvent)
          await this.setRunStatus('failed')
          return
        }
        summary.approvalsRequested = this.executor.approvalsRequested
        summary.text = describeSummary(summary)
        this.emit({ ...this.base('run-completed'), summary } as AgentEvent)
        await this.setRunStatus('completed')
      })
      .catch(async (cause: unknown) => {
        const message = cause instanceof Error ? cause.message : String(cause)
        if (this.agent.signal?.aborted) {
          this.emit({ ...this.base('run-cancelled') } as AgentEvent)
          await this.setRunStatus('cancelled')
          return
        }
        this.emit({ ...this.base('run-failed'), error: message } as AgentEvent)
        await this.setRunStatus('failed')
      })

    void promise.finally(() => {
      queue.close()
      if (this.queue === queue) this.queue = undefined
    })

    return queue.iterator()
  }

  private async setRunStatus(status: AgentRunStatus): Promise<void> {
    this.state = { ...this.state, lastRunStatus: status }
    try {
      await this.enqueuePersistence(() =>
        this.broker.updateSession(this.id, { last_run_status: status })
      )
    } catch {
      // A status write is bookkeeping; losing it must not fail the run.
    }
  }

  async cancel(): Promise<void> {
    // Abort the parent first so in-flight tool AbortSignals fire, then make
    // sure any child sessions are torn down even if a signal was missed.
    this.agent.abort()
    await this.cancelChildren()
    // Also stop anything the tools left running and refuse pending approvals.
    // A broker failure is not a successful cancellation: host processes or
    // approvals may still be live, so propagate it to the caller.
    await this.broker.cancel(this.id)
    this.emit({ ...this.base('run-cancelled') } as AgentEvent)
    await this.setRunStatus('cancelled')
  }

  /**
   * Replace old turns with a structured digest. Goals, decisions, file changes,
   * command outcomes, failures, and granted permissions are preserved; raw tool
   * output is not.
   */
  async compact(): Promise<AgentCompactionState> {
    const messages = this.state.messages
    const keep = messages.slice(-COMPACTION_KEEP_TAIL)
    const dropped = messages.slice(0, Math.max(0, messages.length - COMPACTION_KEEP_TAIL))
    const facts = buildCompactionSummary(dropped, this.state)
    // The model writes what the digest cannot: what was being attempted, and
    // why an approach was abandoned. Its failure is not compaction's failure —
    // the facts are a complete summary on their own.
    const prose =
      dropped.length > 0
        ? await requestModelSummary({
            baseUrl: this.broker.openAiBaseUrl(),
            apiKey: this.broker.apiKey(),
            model: this.state.model.id,
            transcript: renderTranscript(dropped),
            facts
          })
        : null
    const summaryText = mergeSummary(prose, facts)
    const summaryMessage: AgentMessage = {
      role: 'system',
      text: summaryText,
      timestamp: new Date().toISOString()
    }
    const next = dropped.length > 0 ? [summaryMessage, ...keep] : messages
    this.state = { ...this.state, messages: next }
    const modelId = this.state.model.id
    this.agent.state.messages = next
      .map((message) => toPiMessage(message, modelId))
      .filter((message): message is PiMessage => Boolean(message))
    await this.enqueuePersistence(() => this.broker.appendMessages(this.id, next, true))

    const compaction: AgentCompactionState = {
      compactedAt: new Date().toISOString(),
      removedMessages: dropped.length,
      summary: summaryText,
      summarySource: prose ? 'model' : 'deterministic'
    }
    this.state = { ...this.state, compactionState: compaction }
    await this.enqueuePersistence(() =>
      this.broker.updateSession(this.id, {
        compaction: compaction as unknown as Record<string, unknown>
      })
    )
      .catch(() => undefined)
    this.emit({ ...this.base('compacted'), state: compaction } as AgentEvent)
    return compaction
  }

  async setModel(model: AgentModelReference): Promise<void> {
    if (this.agent.state.isStreaming) {
      throw new Error('The model can only be changed between runs.')
    }
    this.state = { ...this.state, model }
    this.agent.state.model = piModel(model, this.broker.openAiBaseUrl(), this.reasoningEnabled)
    this.agent.state.thinkingLevel = thinkingLevelFor(this.reasoningEnabled)
    await this.broker.updateSession(this.id, { model: model.id })
  }

  async setEnabledTools(toolNames: string[]): Promise<void> {
    this.enabledTools = toolNames
    this.state = { ...this.state, enabledTools: toolNames }
    this.executor.setDefinitions(this.definitions)
    this.agent.state.tools = this.buildTools()
    await this.broker.updateSession(this.id, { enabled_tools: toolNames })
  }

  getState(): AgentSessionState {
    return this.state
  }

  isDisposed(): boolean {
    return this.disposed
  }

  async dispose(): Promise<void> {
    if (this.disposed) return
    this.disposed = true
    this.agent.abort()
    this.queue?.close()
    this.queue = undefined
    this.onDisposed()
  }
}

/** Structured digest used when compaction drops old turns. */
export function buildCompactionSummary(
  dropped: AgentMessage[],
  state: Pick<AgentSessionState, 'title' | 'toolExecutions'>
): string {
  const goals = dropped
    .filter((message): message is Extract<AgentMessage, { role: 'user' }> => message.role === 'user')
    .map((message) => message.text.trim())
    .filter((text) => text.length > 0)
  const decisions = dropped
    .filter(
      (message): message is Extract<AgentMessage, { role: 'assistant' }> =>
        message.role === 'assistant' && message.text.trim().length > 0
    )
    .map((message) => message.text.trim())
  const failures = dropped
    .filter((message): message is Extract<AgentMessage, { role: 'tool' }> => message.role === 'tool')
    .filter((message) => message.isError)
    .map((message) => `${message.tool}: ${message.output.split('\n')[0]}`)
  const changed = new Set<string>()
  const commands: string[] = []
  for (const execution of state.toolExecutions) {
    for (const path of execution.changed_paths ?? []) changed.add(path)
    if (execution.tool === 'shell_run' || execution.tool === 'shell_start') {
      const command = execution.arguments?.command
      if (typeof command === 'string') commands.push(command)
    }
  }

  const lines = [`Earlier in this session (${dropped.length} messages compacted):`]
  if (goals.length > 0) {
    lines.push(`Requests: ${goals.slice(-5).map(truncate).join(' | ')}`)
  }
  if (decisions.length > 0) {
    lines.push(`Work so far: ${decisions.slice(-3).map(truncate).join(' | ')}`)
  }
  if (changed.size > 0) {
    lines.push(`Files changed: ${[...changed].slice(0, 20).join(', ')}`)
  }
  if (commands.length > 0) {
    lines.push(`Commands run: ${commands.slice(-8).join(' ; ')}`)
  }
  if (failures.length > 0) {
    lines.push(`Unresolved failures: ${failures.slice(-5).map(truncate).join(' | ')}`)
  }
  lines.push('Full tool output from those turns is no longer in context; re-read files if needed.')
  return lines.join('\n')
}

function truncate(text: string): string {
  return text.length > 240 ? `${text.slice(0, 240)}…` : text
}

/** Pi-backed runtime. Constructed by the worker, never by the UI. */
export class PiAgentRuntime implements AgentRuntime {
  readonly descriptor = DESCRIPTOR
  private readonly broker: BrokerClient
  private readonly sessions = new Map<string, PiAgentSession>()
  /** In-flight child session ids keyed by parent session id. */
  private readonly childrenByParent = new Map<string, Set<string>>()
  /** llama-server KV slot per child session (parent always uses slot 0). */
  private readonly childLlamaSlots = new Map<string, number>()
  private sandbox?: SandboxDescription

  constructor(broker: BrokerClient) {
    this.broker = broker
  }

  private async sandboxDescription(): Promise<SandboxDescription> {
    if (this.sandbox) return this.sandbox
    const capabilities = await this.broker.capabilities()
    this.sandbox = {
      backend: capabilities.sandbox.backend,
      profile: 'workspace',
      isolated: capabilities.sandbox.isolated,
      network: false,
      detail: capabilities.sandbox.detail
    }
    return this.sandbox
  }

  private trackChild(parentId: string, childId: string): void {
    const set = this.childrenByParent.get(parentId) ?? new Set<string>()
    set.add(childId)
    this.childrenByParent.set(parentId, set)
  }

  private untrackChild(parentId: string, childId: string): void {
    const set = this.childrenByParent.get(parentId)
    if (!set) return
    set.delete(childId)
    if (set.size === 0) this.childrenByParent.delete(parentId)
  }

  /** Pick a free llama slot in 1..maxExclusive-1 for a new subagent session. */
  private allocateChildLlamaSlot(maxExclusive: number): number {
    if (maxExclusive <= 1) return 0
    const used = new Set(this.childLlamaSlots.values())
    for (let slot = 1; slot < maxExclusive; slot += 1) {
      if (!used.has(slot)) return slot
    }
    return 0
  }

  private releaseChildLlamaSlot(childId: string): void {
    this.childLlamaSlots.delete(childId)
  }

  async cancelChildren(parentId: string): Promise<void> {
    const children = [...(this.childrenByParent.get(parentId) ?? [])]
    await Promise.all(
      children.map(async (childId) => {
        const child = this.sessions.get(childId)
        if (child) await child.cancel().catch(() => undefined)
        else await this.broker.cancel(childId).catch(() => undefined)
      })
    )
  }

  /**
   * Create depth-1 child Pi session(s), run prompts (concurrently when several),
   * and return their summaries as the parent's tool result. Approvals from
   * children are forwarded onto the parent's event stream.
   */
  async spawnSubagent(
    parent: PiAgentSession,
    request: ExecuteToolRequest
  ): Promise<ToolExecutionOutcome> {
    const started = Date.now()
    const parentState = parent.getState()
    const sandbox = await this.sandboxDescription()
    const fail = (message: string): ToolExecutionOutcome => ({
      output: message,
      isError: true,
      denied: false,
      environment: 'sandbox',
      sandbox,
      changedPaths: [],
      truncated: false,
      durationMs: Date.now() - started,
      images: []
    })

    if (isSubagentSession(parentState.runtimeMetadata)) {
      return fail('Subagents cannot spawn further subagents.')
    }

    const prompts = collectSpawnPrompts(request.args)
    if (prompts.length === 0) {
      return fail('Provide `prompt` or a non-empty `prompts` array.')
    }

    let profile
    try {
      profile = await this.broker.textProfile(parentState.model.id)
    } catch {
      profile = null
    }
    const max = resolveMaxSubagents(profile)
    const inFlight = this.childrenByParent.get(parent.id)?.size ?? 0
    if (inFlight + prompts.length > max) {
      return fail(
        `This would run ${inFlight + prompts.length} subagent(s); max concurrent is ${max}` +
          (inFlight > 0 ? ` (${inFlight} already running)` : '') +
          '. Shrink `prompts` or wait for a child to finish.'
      )
    }

    const modelId = resolveSubagentModel(request.args.model, profile, parentState.model.id)
    const contextWindow = parentState.model.contextWindow
    const parallelSlots = llamaParallelSlots(profile)
    const catalog = await this.broker.tools()
    const parentToolNames =
      parentState.enabledTools ?? catalog.map((tool) => tool.name)
    const enabled = childEnabledTools(parentToolNames)
    const childTools = catalog.filter((tool) => enabled.includes(tool.name))
    const { inferModelCapabilities } = await import('../core/modelCompat')
    const capabilities = inferModelCapabilities(modelId)

    const results = await Promise.all(
      prompts.map((prompt, index) =>
        this.runOneSubagent({
          parent,
          request,
          prompt,
          index,
          total: prompts.length,
          modelId,
          contextWindow,
          enabled,
          childTools,
          capabilities,
          parallelSlots
        })
      )
    )

    const output =
      results.length === 1
        ? results[0]!.summary
        : results
            .map((result, index) => {
              const label = prompts[index]!.length > 60 ? `${prompts[index]!.slice(0, 60)}…` : prompts[index]
              return `### Subagent ${index + 1}: ${label}\nStatus: ${result.status}\n${result.summary}`
            })
            .join('\n\n')

    return {
      output,
      isError: results.some((result) => result.status !== 'completed'),
      denied: false,
      environment: 'sandbox',
      sandbox,
      changedPaths: [],
      truncated: false,
      durationMs: Date.now() - started,
      images: []
    }
  }

  private async runOneSubagent(options: {
    parent: PiAgentSession
    request: ExecuteToolRequest
    prompt: string
    index: number
    total: number
    modelId: string
    contextWindow?: number
    enabled: string[]
    childTools: AgentToolDefinition[]
    capabilities: CreateAgentSessionOptions['capabilities']
    parallelSlots: number
  }): Promise<{ status: 'completed' | 'failed' | 'cancelled'; summary: string }> {
    const { parent, request, prompt, modelId, contextWindow } = options
    const parentState = parent.getState()

    let childRecord
    try {
      childRecord = await this.broker.createSession({
        title: `Subagent · ${prompt.slice(0, 48)}`,
        workspace_path: parentState.workspacePath,
        model: modelId,
        permission_mode: parentState.permissionMode,
        permission_settings: parentState.permissionSettings,
        enabled_tools: options.enabled
      })
      const metadata = buildSubagentMetadata(
        parent.id,
        parentState.runtimeMetadata as Record<string, unknown> | undefined
      )
      childRecord = await this.broker.updateSession(childRecord.id, {
        runtime_metadata: metadata
      })
    } catch (cause) {
      return {
        status: 'failed',
        summary: summarizeSubagentResult([], {
          failed: true,
          error: cause instanceof Error ? cause.message : String(cause)
        })
      }
    }

    const llamaSlot = this.allocateChildLlamaSlot(options.parallelSlots)
    this.childLlamaSlots.set(childRecord.id, llamaSlot)
    this.trackChild(parent.id, childRecord.id)
    parent.forwardEvent({
      type: 'subagent-started',
      sessionId: parent.id,
      runId: request.runId,
      timestamp: new Date().toISOString(),
      sequence: 0,
      toolCallId: request.toolCallId,
      childSessionId: childRecord.id,
      model: modelId,
      prompt
    })

    let status: 'completed' | 'failed' | 'cancelled' = 'completed'
    let summary = ''
    try {
      if (request.signal?.aborted) {
        status = 'cancelled'
        summary = 'Subagent cancelled before it started.'
        await this.broker.cancel(childRecord.id).catch(() => undefined)
      } else {
        const promptPayload = await this.broker.systemPrompt(childRecord.id)
        const childSession = (await this.createSession({
          sessionId: childRecord.id,
          model: { id: modelId, name: modelId, contextWindow },
          systemPrompt: promptPayload.system_prompt,
          tools: options.childTools,
          messages: [],
          capabilities: options.capabilities,
          llamaSlot
        })) as PiAgentSession

        const abortChild = (): void => {
          void childSession.cancel()
        }
        request.signal?.addEventListener('abort', abortChild, { once: true })
        try {
          for await (const event of childSession.run({ text: prompt })) {
            if (event.type === 'approval-required' || event.type === 'elevation-requested') {
              parent.forwardEvent(event)
            }
            const progress = subagentProgressFromChildEvent(
              event,
              request.toolCallId,
              childRecord.id,
              request.runId
            )
            if (progress) parent.forwardEvent(progress)
          }
        } finally {
          request.signal?.removeEventListener('abort', abortChild)
        }

        const childState = childSession.getState()
        if (childState.lastRunStatus === 'cancelled' || request.signal?.aborted) {
          status = 'cancelled'
          summary = 'Subagent was cancelled.'
        } else if (childState.lastRunStatus === 'failed') {
          status = 'failed'
          summary = summarizeSubagentResult(childState.messages, {
            failed: true,
            error: 'the child run failed'
          })
        } else {
          summary = summarizeSubagentResult(childState.messages)
        }
      }
    } catch (cause) {
      status = 'failed'
      summary = summarizeSubagentResult([], {
        failed: true,
        error: cause instanceof Error ? cause.message : String(cause)
      })
    } finally {
      this.untrackChild(parent.id, childRecord.id)
      this.releaseChildLlamaSlot(childRecord.id)
    }

    parent.forwardEvent({
      type: 'subagent-completed',
      sessionId: parent.id,
      runId: request.runId,
      timestamp: new Date().toISOString(),
      sequence: 0,
      toolCallId: request.toolCallId,
      childSessionId: childRecord.id,
      model: modelId,
      status,
      summary
    })

    return { status, summary }
  }

  async createSession(options: CreateAgentSessionOptions): Promise<AgentSession> {
    const messages = options.messages
    const existing = this.sessions.get(options.sessionId)
    if (existing) {
      if (existing.isDisposed()) {
        this.sessions.delete(options.sessionId)
      } else {
        // A cached session may be stale after mode switches or a close that only
        // cleared the worker map. Always re-apply daemon history so the model
        // sees the same transcript the UI just loaded.
        if (messages) {
          existing.rehydrate(messages, options.systemPrompt)
        } else if (options.systemPrompt !== undefined) {
          existing.rehydrate(existing.getState().messages, options.systemPrompt)
        }
        return existing
      }
    }
    const remote = options.preloaded
      ? {
          session: options.preloaded.session,
          tool_executions: options.preloaded.tool_executions,
          sandbox: options.preloaded.sandbox
        }
      : await this.broker.session(options.sessionId).then((payload) => ({
          session: payload.session,
          tool_executions: payload.tool_executions,
          sandbox: {
            backend: payload.sandbox.backend,
            profile: 'workspace',
            isolated: payload.sandbox.isolated,
            network: false,
            detail: payload.sandbox.detail
          } satisfies SandboxDescription
        }))
    const sandbox = options.preloaded?.sandbox ?? (remote.sandbox as SandboxDescription)
    if (options.preloaded) {
      this.sandbox = sandbox
    }
    const defaults =
      options.preloadedInference ?? (await this.broker.runtimeInferenceSettings())
    const state: AgentSessionState = {
      id: remote.session.id,
      title: remote.session.title,
      workspacePath: remote.session.workspace_path ?? null,
      model: options.model,
      runtimeId: DESCRIPTOR.id,
      messages: messages ?? [],
      toolExecutions: remote.tool_executions,
      permissionMode: remote.session.permission_mode,
      permissionSettings: remote.session.permission_settings,
      enabledTools: remote.session.enabled_tools ?? undefined,
      createdAt: remote.session.created_at,
      updatedAt: remote.session.updated_at,
      lastRunStatus: 'idle',
      runtimeMetadata: (remote.session.runtime_metadata as Record<string, unknown> | null) ?? undefined
    }
    const session = new PiAgentSession({
      broker: this.broker,
      state,
      definitions: options.tools,
      systemPrompt: options.systemPrompt,
      sandbox,
      capabilities: options.capabilities,
      reasoningEnabled: defaults.enable_reasoning ?? true,
      dropReasoningBetweenTurns: defaults.drop_reasoning_between_turns ?? false,
      spawnSubagent: (parent, request) => this.spawnSubagent(parent, request),
      cancelChildren: () => this.cancelChildren(options.sessionId),
      onDisposed: () => {
        this.sessions.delete(options.sessionId)
        this.releaseChildLlamaSlot(options.sessionId)
      },
      llamaSlot: options.llamaSlot ?? 0
    })
    this.sessions.set(session.id, session)
    return session
  }

  /**
   * Rebuild a session from what the daemon stored. Nothing is re-executed: the
   * transcript comes back, the workspace is re-checked, and the agent waits.
   */
  async restoreSession(sessionId: string): Promise<AgentSession> {
    const cached = this.sessions.get(sessionId)
    if (cached) return cached
    const remote = await this.broker.session(sessionId)
    const prompt = await this.broker.systemPrompt(sessionId)
    const tools = await this.broker.tools()
    const { inferModelCapabilities } = await import('../core/modelCompat')
    return this.createSession({
      sessionId,
      model: { id: remote.session.model },
      systemPrompt: prompt.system_prompt,
      tools,
      messages: remote.messages.map((record) => record.payload),
      capabilities: inferModelCapabilities(remote.session.model)
    })
  }

  async dispose(): Promise<void> {
    for (const session of this.sessions.values()) {
      await session.dispose()
    }
    this.sessions.clear()
    this.childrenByParent.clear()
  }
}

/** Map a child-session event into a parent-scoped progress update for the UI. */
function subagentProgressFromChildEvent(
  event: AgentEvent,
  parentToolCallId: string,
  childSessionId: string,
  runId: string
): Extract<AgentEvent, { type: 'subagent-progress' }> | null {
  const base = {
    type: 'subagent-progress' as const,
    sessionId: event.sessionId,
    runId,
    timestamp: event.timestamp,
    sequence: event.sequence,
    toolCallId: parentToolCallId,
    childSessionId
  }
  switch (event.type) {
    case 'tool-started':
      return {
        ...base,
        sessionId: event.sessionId,
        activity: {
          id: event.toolCallId,
          tool: event.tool,
          args: event.args,
          status: 'running',
          environment: event.environment
        }
      }
    case 'tool-completed':
      return {
        ...base,
        activity: {
          id: event.toolCallId,
          tool: event.tool,
          status: 'completed',
          detail: event.output.slice(0, 500),
          environment: event.environment
        }
      }
    case 'tool-failed':
      return {
        ...base,
        activity: {
          id: event.toolCallId,
          tool: event.tool,
          status: event.denied ? 'denied' : 'failed',
          detail: event.error.slice(0, 500),
          environment: event.environment
        }
      }
    case 'approval-required':
      return {
        ...base,
        activity: {
          id: event.toolCallId,
          tool: event.approval.tool,
          args: event.approval.arguments,
          status: 'awaiting-approval',
          detail: event.approval.summary,
          environment: event.approval.environment
        }
      }
    case 'message-committed': {
      if (event.message.role !== 'assistant') return null
      const text = event.message.text?.trim()
      if (!text || text === 'null') return null
      return {
        ...base,
        activity: {
          id: `msg-${event.sequence}`,
          status: 'completed',
          detail: text.length > 400 ? `${text.slice(0, 400)}…` : text
        }
      }
    }
    default:
      return null
  }
}

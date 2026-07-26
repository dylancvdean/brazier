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
import { accumulate, describeSummary, emptySummary } from '../core/runSummary'
import { AgentToolExecutor, EventSequencer } from '../core/toolExecutor'
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

/** Build a Pi model that points at the daemon's OpenAI-compatible endpoint. */
function piModel(model: AgentModelReference, baseUrl: string): Model<'openai-completions'> {
  return {
    id: model.id,
    name: model.name ?? model.id,
    api: 'openai-completions',
    provider: 'brazier',
    baseUrl,
    reasoning: false,
    input: ['text'],
    // Local inference has no per-token price; the daemon reports no usage cost.
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: model.contextWindow ?? 8_192,
    maxTokens: model.maxTokens ?? 4_096
  }
}

/** Convert an application message into the shape Pi hands to a model. */
function toPiMessage(message: AgentMessage): PiMessage | undefined {
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
        model: 'restored',
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

/** Text of an assistant message, ignoring thinking and tool calls. */
function assistantText(message: AssistantMessage): string {
  return message.content
    .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
    .map((part) => part.text)
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
  private toolsThisTurn = 0
  private queue?: EventQueue
  private currentRunId?: string

  constructor(options: {
    broker: BrokerClient
    state: AgentSessionState
    definitions: AgentToolDefinition[]
    systemPrompt: string
    sandbox: SandboxDescription
    capabilities: CreateAgentSessionOptions['capabilities']
  }) {
    this.id = options.state.id
    this.broker = options.broker
    this.state = options.state
    this.definitions = options.definitions
    this.enabledTools = options.state.enabledTools ?? options.definitions.map((tool) => tool.name)
    this.sandbox = options.sandbox
    this.maxToolsPerTurn = options.capabilities.maxToolsPerTurn
    this.executor = new AgentToolExecutor({
      broker: options.broker,
      sessionId: options.state.id,
      emit: (event) => this.queue?.push(event),
      sequencer: this.sequencer,
      definitions: options.definitions,
      sandbox: options.sandbox
    })

    this.agent = new Agent({
      // Every model request goes to the daemon's OpenAI-compatible endpoint,
      // so local and remote models both work with no provider configuration.
      streamFn: (model, context, options) =>
        streamSimple(model as Model<'openai-completions'>, context, options),
      // The transcript only ever holds LLM messages, because the application
      // converts its own shapes before they reach the agent. Anything else a
      // runtime version adds for its own bookkeeping is dropped here rather
      // than sent to a model.
      convertToLlm: (messages) => messages.filter(isLlmMessage),
      getApiKey: () => options.broker.apiKey(),
      sessionId: options.state.id,
      toolExecution: options.capabilities.parallelToolCalling ? 'parallel' : 'sequential',
      initialState: {
        systemPrompt: options.systemPrompt,
        model: piModel(options.state.model, options.broker.openAiBaseUrl()),
        thinkingLevel: 'off',
        tools: this.buildTools(),
        messages: options.state.messages.map(toPiMessage).filter((message): message is PiMessage =>
          Boolean(message)
        )
      }
    })

    this.agent.subscribe((event) => this.onPiEvent(event))
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
          executionMode: definition.executes ? 'sequential' : undefined,
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
              content: [{ type: 'text', text: outcome.output }],
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
          this.emit({ ...this.base('text-delta'), delta: inner.delta, channel: 'text' } as AgentEvent)
        } else if (inner.type === 'thinking_delta') {
          this.emit({
            ...this.base('text-delta'),
            delta: inner.delta,
            channel: 'reasoning'
          } as AgentEvent)
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
        void this.broker.appendMessages(this.id, [message]).catch(() => {
          // Persistence failures must not abort a run in progress; the run
          // still streams and the UI still shows it.
        })
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

    const promise = this.agent
      .prompt(input.text)
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
      await this.broker.updateSession(this.id, { last_run_status: status })
    } catch {
      // A status write is bookkeeping; losing it must not fail the run.
    }
  }

  async cancel(): Promise<void> {
    this.agent.abort()
    // Also stop anything the tools left running and refuse pending approvals.
    await this.broker.cancel(this.id).catch(() => undefined)
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
    this.agent.state.messages = next
      .map(toPiMessage)
      .filter((message): message is PiMessage => Boolean(message))
    await this.broker.appendMessages(this.id, next, true)

    const compaction: AgentCompactionState = {
      compactedAt: new Date().toISOString(),
      removedMessages: dropped.length,
      summary: summaryText,
      summarySource: prose ? 'model' : 'deterministic'
    }
    this.state = { ...this.state, compactionState: compaction }
    await this.broker
      .updateSession(this.id, { compaction: compaction as unknown as Record<string, unknown> })
      .catch(() => undefined)
    this.emit({ ...this.base('compacted'), state: compaction } as AgentEvent)
    return compaction
  }

  async setModel(model: AgentModelReference): Promise<void> {
    if (this.agent.state.isStreaming) {
      throw new Error('The model can only be changed between runs.')
    }
    this.state = { ...this.state, model }
    this.agent.state.model = piModel(model, this.broker.openAiBaseUrl())
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

  async dispose(): Promise<void> {
    this.agent.abort()
    this.queue?.close()
    this.queue = undefined
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

  async createSession(options: CreateAgentSessionOptions): Promise<AgentSession> {
    const existing = this.sessions.get(options.sessionId)
    if (existing) return existing
    const remote = await this.broker.session(options.sessionId)
    const sandbox = await this.sandboxDescription()
    const state: AgentSessionState = {
      id: remote.session.id,
      title: remote.session.title,
      workspacePath: remote.session.workspace_path ?? null,
      model: options.model,
      runtimeId: DESCRIPTOR.id,
      messages: options.messages ?? remote.messages.map((record) => record.payload),
      toolExecutions: remote.tool_executions,
      permissionMode: remote.session.permission_mode,
      permissionSettings: remote.session.permission_settings,
      enabledTools: remote.session.enabled_tools ?? undefined,
      createdAt: remote.session.created_at,
      updatedAt: remote.session.updated_at,
      lastRunStatus: 'idle'
    }
    const session = new PiAgentSession({
      broker: this.broker,
      state,
      definitions: options.tools,
      systemPrompt: options.systemPrompt,
      sandbox,
      capabilities: options.capabilities
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
  }
}

/**
 * Native zero-SDK Brazier agent harness.
 *
 * Speaks directly to `BrokerClient.streamChatCompletion` and `AgentToolExecutor`
 * with zero third-party agent framework dependencies.
 */

import {
  ContextOverflowError,
  type BrokerClient,
  type ChatCompletionMessage,
  type ChatCompletionToolSpec,
  type ChatContentPart
} from '../core/brokerClient'
import { mergeSummary, renderTranscript, requestModelSummary } from '../core/compaction'
import {
  inferModelCapabilities,
  parseLooseJson,
  repairToolArguments,
  resolveToolNameAlias,
  validateToolArguments
} from '../core/modelCompat'
import { accumulate, describeSummary, emptySummary } from '../core/runSummary'
import {
  buildSubagentMetadata,
  childEnabledTools,
  collectSpawnPrompts,
  isSubagentSession,
  llamaParallelSlots,
  resolveMaxSubagents,
  resolveSubagentExecutionMode,
  resolveSubagentModel,
  SPAWN_SUBAGENT_TOOL,
  summarizeSubagentResult
} from '../core/subagent'
import {
  AgentToolExecutor,
  EventSequencer,
  type ExecuteToolRequest,
  type ToolExecutionOutcome,
  type ToolImage
} from '../core/toolExecutor'
import type {
  AgentCompactionState,
  AgentEvent,
  AgentMessage,
  AgentModelReference,
  AgentPermissionMode,
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
import { LoopGuard } from './loopGuard'
import { extractTextToolCalls, ThinkStreamSplitter } from './streamParser'

/** Messages kept verbatim when compaction rewrites the transcript. */
export const COMPACTION_KEEP_TAIL = 6

const DEFAULT_CONTEXT_WINDOW = 8_192
const AUTO_COMPACT_THRESHOLD_RATIO = 0.82
const OVERFLOW_TAIL_TOOL_MAX_CHARS = 16_384

const BASE_DESCRIPTOR = {
  version: '1.0.0',
  capabilities: {
    streaming: true,
    toolCalls: true,
    compaction: true,
    cancellation: true,
    sessionRestore: true
  }
} as const

function descriptorFor(runtimeId: string): AgentRuntimeDescriptor {
  if (runtimeId === 'powerful') {
    return { id: 'powerful', name: 'Powerful', ...BASE_DESCRIPTOR }
  }
  return { id: 'simple', name: 'Simple', ...BASE_DESCRIPTOR }
}

/** True when a tool can safely execute concurrently alongside adjacent read-only calls. */
export function isWaveConcurrentTool(
  toolName: string,
  definition?: Pick<AgentToolDefinition, 'risk'>
): boolean {
  if (toolName === SPAWN_SUBAGENT_TOOL) return true
  return definition?.risk === 'safe' || definition?.risk === 'read'
}

/**
 * Partition tool calls into execution waves:
 * - When `parallelToolCalling` is true, consecutive read/safe/`spawn_subagent`
 *   calls form a single concurrent wave, while any write/execute/destructive or
 *   unknown tool runs alone in its own 1-item wave.
 * - When `parallelToolCalling` is false, every call runs in its own 1-item wave.
 */
export function partitionToolWaves<T extends { name: string }>(
  calls: readonly T[],
  definitions: ReadonlyMap<string, Pick<AgentToolDefinition, 'risk'>>,
  parallelToolCalling: boolean
): T[][] {
  if (!parallelToolCalling) return calls.map((call) => [call])
  const waves: T[][] = []
  let currentConcurrent: T[] = []
  for (const call of calls) {
    if (isWaveConcurrentTool(call.name, definitions.get(call.name))) {
      currentConcurrent.push(call)
    } else {
      if (currentConcurrent.length > 0) {
        waves.push(currentConcurrent)
        currentConcurrent = []
      }
      waves.push([call])
    }
  }
  if (currentConcurrent.length > 0) waves.push(currentConcurrent)
  return waves
}

function estimateMessageChars(message: AgentMessage): number {
  switch (message.role) {
    case 'user':
    case 'system':
      return message.text.length + 16
    case 'assistant':
      return (
        message.text.length +
        (message.reasoning?.length ?? 0) +
        (message.toolCalls ? JSON.stringify(message.toolCalls).length : 0) +
        24
      )
    case 'tool':
      return message.output.length + message.tool.length + 24
  }
}

function estimateTokensFromChars(chars: number): number {
  return Math.ceil(chars / 4)
}

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

type PreparedToolCall = {
  id: string
  rawName: string
  resolvedName: string
  rawArgs: unknown
  args: Record<string, unknown>
  definition?: AgentToolDefinition
}

export class NativeAgentSession implements AgentSession {
  readonly id: string
  private readonly broker: BrokerClient
  private readonly sequencer = new EventSequencer()
  private readonly executor: AgentToolExecutor
  private state: AgentSessionState
  private systemPrompt: string
  private definitions: AgentToolDefinition[]
  private enabledTools: string[]
  private readonly sandbox: SandboxDescription
  private parallelToolCalling: boolean
  private maxToolsPerTurn?: number
  private harmony: boolean
  private readonly llamaSlot: number
  private readonly maxTurnsPerRun?: number
  private readonly spawnSubagent: (
    parent: NativeAgentSession,
    request: ExecuteToolRequest
  ) => Promise<ToolExecutionOutcome>
  private readonly cancelChildren: () => Promise<void>
  private readonly onDisposed: () => void
  private reasoningEnabled: boolean
  private dropReasoningBetweenTurns: boolean
  private queue?: EventQueue
  private currentRunId?: string
  private runAbort?: AbortController
  private cancelledCurrentRun = false
  private streaming = false
  private completing = false
  private disposed = false
  private persistence = Promise.resolve()
  private lastKnownPromptTokens = 0
  private lastPromptTokenAnchor?: number
  private latestContextTotal?: number

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
      parent: NativeAgentSession,
      request: ExecuteToolRequest
    ) => Promise<ToolExecutionOutcome>
    cancelChildren: () => Promise<void>
    onDisposed: () => void
    llamaSlot?: number
    maxTurnsPerRun?: number
  }) {
    this.id = options.state.id
    this.broker = options.broker
    this.state = options.state
    this.systemPrompt = options.systemPrompt
    this.definitions = options.definitions
    this.enabledTools = options.state.enabledTools ?? options.definitions.map((tool) => tool.name)
    this.sandbox = options.sandbox
    this.parallelToolCalling = options.capabilities.parallelToolCalling
    this.maxToolsPerTurn = options.capabilities.maxToolsPerTurn
    this.harmony = options.capabilities.harmony
    this.llamaSlot = options.llamaSlot ?? 0
    this.maxTurnsPerRun = options.maxTurnsPerRun
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
  }

  updateModelReference(model: AgentModelReference): void {
    this.state = {
      ...this.state,
      model: {
        ...this.state.model,
        ...model,
        contextWindow: model.contextWindow ?? this.state.model.contextWindow,
        maxTokens: model.maxTokens ?? this.state.model.maxTokens
      }
    }
  }

  rehydrate(messages: AgentMessage[], systemPrompt?: string): void {
    if (this.disposed) {
      throw new Error('Cannot rehydrate a disposed agent session.')
    }
    this.state = { ...this.state, messages }
    if (systemPrompt !== undefined) {
      this.systemPrompt = systemPrompt
    }
    this.lastKnownPromptTokens = 0
    this.lastPromptTokenAnchor = undefined
  }

  async refreshInferencePrefs(): Promise<void> {
    try {
      const settings = await this.broker.runtimeInferenceSettings()
      this.dropReasoningBetweenTurns = settings.drop_reasoning_between_turns ?? false
      this.reasoningEnabled = settings.enable_reasoning ?? this.reasoningEnabled
    } catch {
      // Keep the last known prefs if the daemon is briefly unreachable.
    }
  }

  forwardEvent(event: AgentEvent): void {
    this.queue?.push(event)
  }

  private emit(event: AgentEvent): void {
    this.queue?.push(event)
  }

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

  private commitMessage(message: AgentMessage): void {
    if (message.role !== 'tool') {
      this.emit({ ...this.base('message-committed'), message } as AgentEvent)
    }
    this.state = { ...this.state, messages: [...this.state.messages, message] }
    void this.enqueuePersistence(() => this.broker.appendMessages(this.id, [message])).catch(
      () => undefined
    )
  }

  private activeDefinitions(): AgentToolDefinition[] {
    return this.definitions.filter((def) => this.enabledTools.includes(def.name))
  }

  private effectiveContextWindow(): number {
    return this.latestContextTotal ?? this.state.model.contextWindow ?? DEFAULT_CONTEXT_WINDOW
  }

  private estimatePromptTokens(): number {
    if (
      this.lastKnownPromptTokens > 0 &&
      this.lastPromptTokenAnchor !== undefined &&
      this.lastPromptTokenAnchor <= this.state.messages.length
    ) {
      const addedChars = this.state.messages
        .slice(this.lastPromptTokenAnchor)
        .reduce((sum, msg) => sum + estimateMessageChars(msg), 0)
      return this.lastKnownPromptTokens + estimateTokensFromChars(addedChars)
    }
    const msgChars = this.state.messages.reduce((sum, msg) => sum + estimateMessageChars(msg), 0)
    const toolChars = this.activeDefinitions().reduce(
      (sum, def) => sum + def.name.length + def.description.length + JSON.stringify(def.inputSchema).length,
      0
    )
    return estimateTokensFromChars(this.systemPrompt.length + msgChars + toolChars)
  }

  private buildChatMessages(
    inputImages: readonly string[] | undefined,
    toolImagesByCallId: ReadonlyMap<string, ToolImage[]>
  ): ChatCompletionMessage[] {
    const out: ChatCompletionMessage[] = []
    if (this.systemPrompt.trim().length > 0) {
      out.push({ role: 'system', content: this.systemPrompt })
    }
    const messages = this.state.messages
    let lastUserIndex = -1
    for (let i = 0; i < messages.length; i += 1) {
      if (messages[i]?.role === 'user') lastUserIndex = i
    }

    for (let i = 0; i < messages.length; i += 1) {
      const msg = messages[i]!
      switch (msg.role) {
        case 'user': {
          if (i === lastUserIndex && inputImages && inputImages.length > 0) {
            const parts: ChatContentPart[] = [
              { type: 'text', text: msg.text },
              ...inputImages.map((url) => ({ type: 'image_url' as const, image_url: { url } }))
            ]
            out.push({ role: 'user', content: parts })
          } else {
            out.push({ role: 'user', content: msg.text })
          }
          break
        }
        case 'system': {
          out.push({ role: 'user', content: `[context] ${msg.text}` })
          break
        }
        case 'assistant': {
          const keepReasoning =
            Boolean(msg.reasoning?.trim()) &&
            (!this.dropReasoningBetweenTurns ||
              i > lastUserIndex ||
              (this.harmony && (msg.toolCalls?.length ?? 0) > 0))
          const wireCalls =
            msg.toolCalls && msg.toolCalls.length > 0
              ? msg.toolCalls.map((call) => ({
                  id: call.id,
                  type: 'function' as const,
                  function: {
                    name: call.name,
                    arguments: JSON.stringify(call.arguments ?? {})
                  }
                }))
              : undefined
          out.push({
            role: 'assistant',
            content: msg.text ?? '',
            ...(keepReasoning && msg.reasoning ? { reasoning_content: msg.reasoning } : {}),
            ...(wireCalls ? { tool_calls: wireCalls } : {})
          })
          break
        }
        case 'tool': {
          out.push({
            role: 'tool',
            tool_call_id: msg.toolCallId,
            content: msg.output
          })
          const images = toolImagesByCallId.get(msg.toolCallId)
          if (images && images.length > 0) {
            out.push({
              role: 'user',
              content: [
                { type: 'text', text: `[Tool image output from ${msg.tool}]` },
                ...images.map((img) => ({
                  type: 'image_url' as const,
                  image_url: { url: `data:${img.mimeType};base64,${img.data}` }
                }))
              ]
            })
          }
          break
        }
      }
    }
    return out
  }

  private buildToolSpecs(defs: readonly AgentToolDefinition[]): ChatCompletionToolSpec[] {
    return defs.map((def) => ({
      type: 'function',
      function: {
        name: def.name,
        description: def.description,
        parameters: def.inputSchema
      }
    }))
  }

  private async truncateOversizedKeptToolOutputs(): Promise<void> {
    let changed = false
    const updated = this.state.messages.map((msg) => {
      if (msg.role === 'tool' && msg.output.length > OVERFLOW_TAIL_TOOL_MAX_CHARS) {
        changed = true
        return {
          ...msg,
          output: `${msg.output.slice(
            0,
            OVERFLOW_TAIL_TOOL_MAX_CHARS
          )}\n\n[truncated oversized tool output to recover context window]`
        }
      }
      return msg
    })
    if (changed) {
      await this.enqueuePersistence(() => this.broker.appendMessages(this.id, updated, true))
      this.state = { ...this.state, messages: updated }
      this.lastKnownPromptTokens = 0
      this.lastPromptTokenAnchor = undefined
    }
  }

  private emitSyntheticToolFailure(
    runId: string,
    call: PreparedToolCall,
    error: string
  ): void {
    const environment = call.definition?.defaultEnvironment ?? 'sandbox'
    const risk = call.definition?.risk ?? 'execute'
    this.emit({
      ...this.base('tool-call-proposed'),
      runId,
      toolCallId: call.id,
      tool: call.resolvedName,
      args: call.args,
      environment,
      risk
    } as AgentEvent)
    this.emit({
      ...this.base('tool-failed'),
      runId,
      toolCallId: call.id,
      tool: call.resolvedName,
      environment,
      sandbox: this.sandbox,
      error,
      denied: false,
      durationMs: 0
    } as AgentEvent)
  }

  run(input: AgentUserInput): AsyncIterable<AgentEvent> {
    const runId = `run-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
    this.currentRunId = runId
    this.completing = true
    this.streaming = true
    this.cancelledCurrentRun = false
    const abortController = new AbortController()
    this.runAbort = abortController

    const queue = new EventQueue()
    this.queue = queue
    this.executor.approvalsRequested = 0
    let summary = emptySummary()
    const originalPush = queue.push.bind(queue)
    queue.push = (event: AgentEvent): void => {
      summary = accumulate(summary, event)
      originalPush(event)
    }

    this.emit({ ...this.base('run-started') } as AgentEvent)
    void this.setRunStatus('running')

    const executeRun = async (): Promise<void> => {
      await this.refreshInferencePrefs()

      this.commitMessage({
        role: 'user',
        text: input.text,
        timestamp: new Date().toISOString()
      })

      const loopGuard = new LoopGuard(
        this.maxTurnsPerRun !== undefined ? { maxTurnsPerRun: this.maxTurnsPerRun } : undefined
      )
      const toolImagesByCallId = new Map<string, ToolImage[]>()

      while (!abortController.signal.aborted) {
        const turnGate = loopGuard.startTurn()
        if (!turnGate.allowed) {
          this.commitMessage({
            role: 'assistant',
            text: turnGate.message ?? 'Stopped after reaching the turn limit.',
            timestamp: new Date().toISOString()
          })
          break
        }

        // Proactive auto-compaction before hitting context ceiling.
        if (
          this.state.messages.length > COMPACTION_KEEP_TAIL + 2 &&
          this.estimatePromptTokens() >= AUTO_COMPACT_THRESHOLD_RATIO * this.effectiveContextWindow()
        ) {
          await this.compact()
        }

        const activeDefs = this.activeDefinitions()
        const activeDefsMap = new Map(activeDefs.map((def) => [def.name, def]))
        const toolSpecs = this.buildToolSpecs(activeDefs)

        let overflowRetried = false
        let turnResult
        let splitter = new ThinkStreamSplitter()

        while (true) {
          splitter = new ThinkStreamSplitter()
          const anchorLen = this.state.messages.length
          try {
            turnResult = await this.broker.streamChatCompletion(
              {
                model: this.state.model.id,
                messages: this.buildChatMessages(input.images, toolImagesByCallId),
                tools: toolSpecs.length > 0 ? toolSpecs : undefined,
                maxTokens: this.state.model.maxTokens,
                enableReasoning: this.reasoningEnabled,
                slot: this.llamaSlot
              },
              {
                onPrefill: (prefill) => {
                  this.lastKnownPromptTokens = prefill.total
                  this.lastPromptTokenAnchor = anchorLen
                  if (typeof prefill.context_total === 'number' && prefill.context_total > 0) {
                    this.latestContextTotal = prefill.context_total
                  }
                  this.emit({
                    ...this.base('prefill-progress'),
                    total: prefill.total,
                    cached: prefill.cached,
                    processed: prefill.processed,
                    elapsedMs: prefill.elapsed_ms,
                    contextTotal: prefill.context_total
                  } as AgentEvent)
                },
                onReasoningDelta: (delta) => {
                  splitter.markBackendReasoning()
                  this.emit({
                    ...this.base('text-delta'),
                    delta,
                    channel: 'reasoning'
                  } as AgentEvent)
                },
                onContentDelta: (delta) => {
                  for (const chunk of splitter.push(delta)) {
                    this.emit({
                      ...this.base('text-delta'),
                      delta: chunk.delta,
                      channel: chunk.channel
                    } as AgentEvent)
                  }
                }
              },
              abortController.signal
            )
            if (typeof turnResult.promptTokens === 'number' && turnResult.promptTokens > 0) {
              this.lastKnownPromptTokens = turnResult.promptTokens
              this.lastPromptTokenAnchor = anchorLen
            }
            if (typeof turnResult.contextTotal === 'number' && turnResult.contextTotal > 0) {
              this.latestContextTotal = turnResult.contextTotal
            }
            break
          } catch (cause) {
            if (abortController.signal.aborted) throw cause
            if (cause instanceof ContextOverflowError && !overflowRetried) {
              overflowRetried = true
              await this.compact()
              await this.truncateOversizedKeptToolOutputs()
              continue
            }
            throw cause
          }
        }

        for (const chunk of splitter.flush()) {
          this.emit({
            ...this.base('text-delta'),
            delta: chunk.delta,
            channel: chunk.channel
          } as AgentEvent)
        }

        const combinedReasoning = (turnResult.reasoning + splitter.reasoningText).trim()
        const rawCombinedProse = splitter.visibleText + splitter.heldBackText

        let finalAssistantText = rawCombinedProse
        let rawCalls: Array<{ id: string; name: string; rawArgs: unknown }> = []

        if (turnResult.toolCalls.length > 0) {
          if (splitter.heldBackText) {
            const checkHeld = extractTextToolCalls(splitter.heldBackText, this.enabledTools)
            if (checkHeld.toolCalls.length === 0) {
              this.emit({
                ...this.base('text-delta'),
                delta: splitter.heldBackText,
                channel: 'text'
              } as AgentEvent)
            } else {
              finalAssistantText = (splitter.visibleText + checkHeld.cleanedText).trim()
            }
          }
          rawCalls = turnResult.toolCalls.map((tc) => ({
            id: tc.id,
            name: tc.name,
            rawArgs: parseLooseJson(tc.arguments) ?? tc.arguments
          }))
        } else {
          const fallback = extractTextToolCalls(rawCombinedProse, this.enabledTools)
          if (fallback.toolCalls.length > 0) {
            finalAssistantText = fallback.cleanedText
            rawCalls = fallback.toolCalls.map((tc) => ({
              id: tc.id,
              name: tc.name,
              rawArgs: tc.arguments
            }))
          } else if (splitter.heldBackText) {
            this.emit({
              ...this.base('text-delta'),
              delta: splitter.heldBackText,
              channel: 'text'
            } as AgentEvent)
          }
        }

        const preparedCalls: PreparedToolCall[] = rawCalls.map((call) => {
          const resolvedName = resolveToolNameAlias(call.name, this.enabledTools)
          const definition = activeDefsMap.get(resolvedName)
          const args = definition
            ? repairToolArguments(call.rawArgs, definition)
            : call.rawArgs && typeof call.rawArgs === 'object' && !Array.isArray(call.rawArgs)
              ? (call.rawArgs as Record<string, unknown>)
              : {}
          return {
            id: call.id,
            rawName: call.name,
            resolvedName,
            rawArgs: call.rawArgs,
            args,
            definition
          }
        })

        const callSummaries: AgentToolCallSummary[] = preparedCalls.map((call) => ({
          id: call.id,
          name: call.resolvedName,
          arguments: call.args
        }))

        this.commitMessage({
          role: 'assistant',
          text: finalAssistantText,
          reasoning: combinedReasoning || undefined,
          toolCalls: callSummaries.length > 0 ? callSummaries : undefined,
          timestamp: new Date().toISOString()
        })

        if (preparedCalls.length === 0) {
          break
        }

        const executable =
          this.maxToolsPerTurn !== undefined
            ? preparedCalls.slice(0, this.maxToolsPerTurn)
            : preparedCalls
        const skipped =
          this.maxToolsPerTurn !== undefined ? preparedCalls.slice(this.maxToolsPerTurn) : []

        const waves = partitionToolWaves(
          executable.map((call) => ({ ...call, name: call.resolvedName })),
          activeDefsMap,
          this.parallelToolCalling
        )

        let haltMessage: string | undefined

        for (const wave of waves) {
          if (abortController.signal.aborted) break

          const waveResults = await Promise.all(
            wave.map(async (call): Promise<AgentMessage> => {
              if (!call.definition) {
                const err = `Tool \`${call.rawName}\` is not available or enabled in this session.`
                this.emitSyntheticToolFailure(runId, call, err)
                return {
                  role: 'tool',
                  toolCallId: call.id,
                  tool: call.resolvedName,
                  output: err,
                  isError: true,
                  timestamp: new Date().toISOString()
                }
              }

              const validationError = validateToolArguments(call.args, call.definition)
              if (validationError) {
                this.emitSyntheticToolFailure(runId, call, validationError)
                return {
                  role: 'tool',
                  toolCallId: call.id,
                  tool: call.resolvedName,
                  output: validationError,
                  isError: true,
                  timestamp: new Date().toISOString()
                }
              }

              const loopDecision = loopGuard.beforeCall(call.resolvedName, call.args)
              if (loopDecision.action === 'short-circuit') {
                this.emitSyntheticToolFailure(runId, call, loopDecision.message)
                return {
                  role: 'tool',
                  toolCallId: call.id,
                  tool: call.resolvedName,
                  output: loopDecision.message,
                  isError: true,
                  timestamp: new Date().toISOString()
                }
              }
              if (loopDecision.action === 'halt-run') {
                haltMessage = loopDecision.message
                this.emitSyntheticToolFailure(runId, call, loopDecision.message)
                return {
                  role: 'tool',
                  toolCallId: call.id,
                  tool: call.resolvedName,
                  output: loopDecision.message,
                  isError: true,
                  timestamp: new Date().toISOString()
                }
              }

              const outcome = await this.executor.execute({
                runId,
                toolCallId: call.id,
                tool: call.resolvedName,
                args: call.args,
                reason: typeof call.args.reason === 'string' ? call.args.reason : undefined,
                signal: abortController.signal
              })
              if (outcome.images.length > 0) {
                toolImagesByCallId.set(call.id, outcome.images)
              }
              const guardedOutput = loopGuard.afterCall(
                call.resolvedName,
                call.args,
                outcome.isError,
                outcome.output
              )
              return {
                role: 'tool',
                toolCallId: call.id,
                tool: call.resolvedName,
                output: guardedOutput,
                isError: outcome.isError,
                timestamp: new Date().toISOString()
              }
            })
          )

          for (const toolMessage of waveResults) {
            this.commitMessage(toolMessage)
          }

          if (haltMessage) break
        }

        for (const extra of skipped) {
          this.commitMessage({
            role: 'tool',
            toolCallId: extra.id,
            tool: extra.resolvedName,
            output: `Only ${this.maxToolsPerTurn} tool call per turn is allowed for this model. Make the next call after reading this result.`,
            isError: false,
            timestamp: new Date().toISOString()
          })
        }

        if (haltMessage) {
          this.commitMessage({
            role: 'assistant',
            text: haltMessage,
            timestamp: new Date().toISOString()
          })
          break
        }
      }

      if (abortController.signal.aborted) {
        if (!this.cancelledCurrentRun) {
          this.cancelledCurrentRun = true
          this.emit({ ...this.base('run-cancelled') } as AgentEvent)
          await this.setRunStatus('cancelled')
        }
        return
      }

      summary.approvalsRequested = this.executor.approvalsRequested
      summary.text = describeSummary(summary)
      this.emit({ ...this.base('run-completed'), summary } as AgentEvent)
      await this.setRunStatus('completed')
    }

    void executeRun()
      .catch(async (cause: unknown) => {
        if (abortController.signal.aborted) {
          if (!this.cancelledCurrentRun) {
            this.cancelledCurrentRun = true
            this.emit({ ...this.base('run-cancelled') } as AgentEvent)
            await this.setRunStatus('cancelled')
          }
          return
        }
        const message = cause instanceof Error ? cause.message : String(cause)
        this.emit({ ...this.base('run-failed'), error: message } as AgentEvent)
        await this.setRunStatus('failed')
      })
      .finally(() => {
        this.streaming = false
        if (this.runAbort === abortController) this.runAbort = undefined
        queue.close()
        if (this.queue === queue) this.queue = undefined
        this.completing = false
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
      // Status writes are bookkeeping; losing one must not fail the run.
    }
  }

  async cancel(): Promise<void> {
    this.runAbort?.abort()
    await this.cancelChildren()
    await this.broker.cancel(this.id)
    if (!this.cancelledCurrentRun) {
      this.cancelledCurrentRun = true
      this.emit({ ...this.base('run-cancelled') } as AgentEvent)
    }
    await this.setRunStatus('cancelled')
  }

  async compact(): Promise<AgentCompactionState> {
    const messages = this.state.messages
    let splitIndex = Math.max(0, messages.length - COMPACTION_KEEP_TAIL)
    while (splitIndex < messages.length && messages[splitIndex]?.role === 'tool') {
      splitIndex += 1
    }
    const keep = messages.slice(splitIndex)
    const dropped = messages.slice(0, splitIndex)
    const facts = buildCompactionSummary(dropped, this.state)
    const prose =
      dropped.length > 0
        ? await requestModelSummary({
            baseUrl: this.broker.openAiBaseUrl(),
            apiKey: this.broker.apiKey(),
            model: this.state.model.id,
            transcript: renderTranscript(dropped),
            facts,
            signal: this.runAbort?.signal
          })
        : null
    const summaryText = mergeSummary(prose, facts)
    const summaryMessage: AgentMessage = {
      role: 'system',
      text: summaryText,
      timestamp: new Date().toISOString()
    }
    const next = dropped.length > 0 ? [summaryMessage, ...keep] : messages
    const compaction: AgentCompactionState = {
      compactedAt: new Date().toISOString(),
      removedMessages: dropped.length,
      summary: summaryText,
      summarySource: prose ? 'model' : 'deterministic'
    }

    if (dropped.length > 0) {
      await this.enqueuePersistence(() => this.broker.appendMessages(this.id, next, true))
      this.state = { ...this.state, messages: next }
      this.lastKnownPromptTokens = 0
      this.lastPromptTokenAnchor = undefined
    }

    await this.enqueuePersistence(() =>
      this.broker.updateSession(this.id, {
        compaction: compaction as unknown as Record<string, unknown>
      })
    )
    this.state = { ...this.state, compactionState: compaction }
    this.emit({
      ...this.base('compacted'),
      state: compaction,
      keptMessages: next
    } as AgentEvent)
    return compaction
  }

  isBusy(): boolean {
    return this.streaming || this.completing
  }

  async setModel(model: AgentModelReference): Promise<void> {
    if (this.isBusy()) {
      throw new Error('The model can only be changed between runs.')
    }
    const caps = inferModelCapabilities(model.id)
    this.parallelToolCalling = caps.parallelToolCalling
    this.maxToolsPerTurn = caps.maxToolsPerTurn
    this.harmony = caps.harmony
    this.state = { ...this.state, model }
    this.latestContextTotal = undefined
    await this.broker.updateSession(this.id, { model: model.id })
  }

  async setEnabledTools(toolNames: string[]): Promise<void> {
    if (this.isBusy()) {
      throw new Error('Tools can only be changed between runs.')
    }
    const catalog = await this.broker.tools()
    this.definitions = catalog.filter((tool) => toolNames.includes(tool.name))
    this.enabledTools = toolNames
    this.state = { ...this.state, enabledTools: toolNames }
    this.executor.setDefinitions(this.definitions)
    await this.broker.updateSession(this.id, { enabled_tools: toolNames })
  }

  async setPermissionMode(mode: AgentPermissionMode): Promise<void> {
    if (this.isBusy()) {
      throw new Error('The permission mode can only be changed between runs.')
    }
    this.state = { ...this.state, permissionMode: mode }
    await this.broker.updateSession(this.id, { permission_mode: mode })
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
    this.runAbort?.abort()
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
    lines.push(`Requests: ${goals.slice(-5).map(truncateText).join(' | ')}`)
  }
  if (decisions.length > 0) {
    lines.push(`Work so far: ${decisions.slice(-3).map(truncateText).join(' | ')}`)
  }
  if (changed.size > 0) {
    lines.push(`Files changed: ${[...changed].slice(0, 20).join(', ')}`)
  }
  if (commands.length > 0) {
    lines.push(`Commands run: ${commands.slice(-8).join(' ; ')}`)
  }
  if (failures.length > 0) {
    lines.push(`Unresolved failures: ${failures.slice(-5).map(truncateText).join(' | ')}`)
  }
  lines.push('Full tool output from those turns is no longer in context; re-read files if needed.')
  return lines.join('\n')
}

function truncateText(text: string): string {
  return text.length > 240 ? `${text.slice(0, 240)}…` : text
}

/** Run async tasks with at most `limit` in flight concurrently, preserving result order. */
async function runWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  worker: (item: T, index: number) => Promise<R>
): Promise<PromiseSettledResult<R>[]> {
  const results = new Array<PromiseSettledResult<R>>(items.length)
  let nextIndex = 0
  const runners = Array.from({ length: Math.min(Math.max(1, limit), items.length) }, async () => {
    while (nextIndex < items.length) {
      const idx = nextIndex
      nextIndex += 1
      try {
        const value = await worker(items[idx]!, idx)
        results[idx] = { status: 'fulfilled', value }
      } catch (reason) {
        results[idx] = { status: 'rejected', reason }
      }
    }
  })
  await Promise.all(runners)
  return results
}

export class NativeAgentRuntime implements AgentRuntime {
  readonly descriptor: AgentRuntimeDescriptor
  private readonly broker: BrokerClient
  private readonly sessions = new Map<string, NativeAgentSession>()
  private readonly childrenByParent = new Map<string, Set<string>>()
  private readonly activeChildSlots = new Set<number>()
  private readonly slotWaiters: Array<() => void> = []
  private sharedSlotChain = Promise.resolve()
  private sandbox?: SandboxDescription

  constructor(broker: BrokerClient, runtimeId = 'simple') {
    this.broker = broker
    this.descriptor = descriptorFor(runtimeId)
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

  /** Acquire a child llama slot in `1..parallelSlots-1`, waiting if all are busy. */
  private async acquireChildLlamaSlot(
    parallelSlots: number,
    signal?: AbortSignal
  ): Promise<number> {
    while (!signal?.aborted) {
      for (let slot = 1; slot < parallelSlots; slot += 1) {
        if (!this.activeChildSlots.has(slot)) {
          this.activeChildSlots.add(slot)
          return slot
        }
      }
      await new Promise<void>((resolve) => {
        const wake = (): void => {
          signal?.removeEventListener('abort', wake)
          resolve()
        }
        this.slotWaiters.push(wake)
        signal?.addEventListener('abort', wake, { once: true })
      })
    }
    throw new Error('Subagent cancelled while waiting for a free llama slot.')
  }

  private releaseChildLlamaSlot(slot: number): void {
    if (slot <= 0) return
    this.activeChildSlots.delete(slot)
    const next = this.slotWaiters.shift()
    next?.()
  }

  private runSerializedOnSlotZero<T>(task: () => Promise<T>): Promise<T> {
    const next = this.sharedSlotChain.then(task, task)
    this.sharedSlotChain = next.then(
      () => undefined,
      () => undefined
    )
    return next
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

  async spawnSubagent(
    parent: NativeAgentSession,
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
    const maxSubagents = resolveMaxSubagents(profile)
    const modelId = resolveSubagentModel(request.args.model, profile, parentState.model.id)
    const parallelSlots = llamaParallelSlots(profile)
    const mode = resolveSubagentExecutionMode({
      parentModelId: parentState.model.id,
      targetModelId: modelId,
      parallelSlots,
      maxSubagents
    })

    let childContextWindow = parentState.model.contextWindow
    if (modelId !== parentState.model.id) {
      try {
        const childProfile = await this.broker.textProfile(modelId)
        if (childProfile?.context_size) childContextWindow = childProfile.context_size
      } catch {
        // Keep parent contextWindow fallback when child profile is unavailable.
      }
    }

    const catalog = await this.broker.tools()
    const parentToolNames = parentState.enabledTools ?? catalog.map((tool) => tool.name)
    const enabled = childEnabledTools(parentToolNames)
    const childTools = catalog.filter((tool) => enabled.includes(tool.name))
    const capabilities = inferModelCapabilities(modelId)

    const settled = await runWithConcurrency(prompts, mode.concurrency, async (prompt) => {
      if (mode.kind === 'serialized_shared_slot') {
        return this.runSerializedOnSlotZero(() =>
          this.runOneSubagent({
            parent,
            request,
            prompt,
            modelId,
            contextWindow: childContextWindow,
            enabled,
            childTools,
            capabilities,
            llamaSlot: 0
          })
        )
      }
      if (mode.kind === 'parallel_llama_slots') {
        let slot = 0
        try {
          slot = await this.acquireChildLlamaSlot(mode.parallelSlots, request.signal)
        } catch {
          return {
            status: 'cancelled' as const,
            summary: 'Subagent cancelled before a slot became available.'
          }
        }
        try {
          return await this.runOneSubagent({
            parent,
            request,
            prompt,
            modelId,
            contextWindow: childContextWindow,
            enabled,
            childTools,
            capabilities,
            llamaSlot: slot
          })
        } finally {
          this.releaseChildLlamaSlot(slot)
        }
      }
      return this.runOneSubagent({
        parent,
        request,
        prompt,
        modelId,
        contextWindow: childContextWindow,
        enabled,
        childTools,
        capabilities,
        llamaSlot: 0
      })
    })

    const results = settled.map((outcome, index) => {
      if (outcome.status === 'fulfilled') return outcome.value
      const cause = outcome.reason
      return {
        status: 'failed' as const,
        summary: summarizeSubagentResult([], {
          failed: true,
          error:
            cause instanceof Error ? cause.message : String(cause ?? `Subagent ${index + 1} failed`)
        })
      }
    })

    const output =
      results.length === 1
        ? results[0]!.summary
        : results
            .map((result, index) => {
              const label =
                prompts[index]!.length > 60 ? `${prompts[index]!.slice(0, 60)}…` : prompts[index]
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
    parent: NativeAgentSession
    request: ExecuteToolRequest
    prompt: string
    modelId: string
    contextWindow?: number
    enabled: string[]
    childTools: AgentToolDefinition[]
    capabilities: CreateAgentSessionOptions['capabilities']
    llamaSlot: number
  }): Promise<{ status: 'completed' | 'failed' | 'cancelled'; summary: string }> {
    const { parent, request, prompt, modelId, contextWindow, llamaSlot } = options
    const parentState = parent.getState()

    if (request.signal?.aborted) {
      return { status: 'cancelled', summary: 'Subagent cancelled before it started.' }
    }

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

    let childSession: NativeAgentSession | undefined
    let status: 'completed' | 'failed' | 'cancelled' = 'completed'
    let summary = ''
    let childFinished = false
    try {
      if (request.signal?.aborted) {
        status = 'cancelled'
        summary = 'Subagent cancelled before it started.'
        await this.broker.cancel(childRecord.id).catch(() => undefined)
      } else {
        const promptPayload = await this.broker.systemPrompt(childRecord.id)
        childSession = (await this.createSession({
          sessionId: childRecord.id,
          model: { id: modelId, name: modelId, contextWindow },
          systemPrompt: promptPayload.system_prompt,
          tools: options.childTools,
          messages: [],
          capabilities: options.capabilities,
          llamaSlot
        })) as NativeAgentSession

        const abortChild = (): void => {
          void childSession?.cancel()
        }
        request.signal?.addEventListener('abort', abortChild, { once: true })
        try {
          for await (const event of childSession.run({ text: prompt })) {
            if (childFinished) continue
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
          childFinished = true
        } finally {
          request.signal?.removeEventListener('abort', abortChild)
          childFinished = true
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
      await childSession?.dispose().catch(() => undefined)
      this.untrackChild(parent.id, childRecord.id)
      await this.broker.deleteSession(childRecord.id).catch(() => undefined)
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

  async startSession(options: CreateAgentSessionOptions): Promise<AgentSession> {
    return this.createSession(options)
  }

  getSession(sessionId: string): AgentSession | undefined {
    const existing = this.sessions.get(sessionId)
    return existing && !existing.isDisposed() ? existing : undefined
  }

  async disposeSession(sessionId: string): Promise<void> {
    const existing = this.sessions.get(sessionId)
    if (existing) {
      await existing.dispose()
      this.sessions.delete(sessionId)
    }
  }

  async createSession(options: CreateAgentSessionOptions): Promise<AgentSession> {
    const messages = options.messages
    const existing = this.sessions.get(options.sessionId)
    if (existing) {
      if (existing.isDisposed()) {
        this.sessions.delete(options.sessionId)
      } else {
        existing.updateModelReference(options.model)
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
    const resolvedModel: AgentModelReference = {
      ...options.model,
      contextWindow: options.model.contextWindow ?? defaults.context_size,
      maxTokens: options.model.maxTokens ?? defaults.max_tokens ?? undefined
    }
    const state: AgentSessionState = {
      id: remote.session.id,
      title: remote.session.title,
      workspacePath: remote.session.workspace_path ?? null,
      model: resolvedModel,
      runtimeId: this.descriptor.id,
      messages: messages ?? [],
      toolExecutions: remote.tool_executions,
      permissionMode: remote.session.permission_mode,
      permissionSettings: remote.session.permission_settings,
      enabledTools: remote.session.enabled_tools ?? undefined,
      createdAt: remote.session.created_at,
      updatedAt: remote.session.updated_at,
      lastRunStatus: 'idle',
      runtimeMetadata:
        (remote.session.runtime_metadata as Record<string, unknown> | null) ?? undefined
    }
    const session = new NativeAgentSession({
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
      },
      llamaSlot: options.llamaSlot ?? 0
    })
    this.sessions.set(session.id, session)
    return session
  }

  /**
   * Rebuild a session from daemon state while preserving the configured
   * `contextWindow` (explicit override -> model textProfile -> runtime settings).
   */
  async restoreSession(
    sessionOrInput:
      | string
      | { sessionId: string; contextWindow?: number; maxTokens?: number },
    extra?: { contextWindow?: number; maxTokens?: number }
  ): Promise<AgentSession> {
    const sessionId =
      typeof sessionOrInput === 'string' ? sessionOrInput : sessionOrInput.sessionId
    const explicitContextWindow =
      typeof sessionOrInput === 'object'
        ? sessionOrInput.contextWindow
        : extra?.contextWindow
    const explicitMaxTokens =
      typeof sessionOrInput === 'object' ? sessionOrInput.maxTokens : extra?.maxTokens

    const cached = this.sessions.get(sessionId)
    if (cached && !cached.isDisposed()) {
      if (explicitContextWindow !== undefined || explicitMaxTokens !== undefined) {
        cached.updateModelReference({
          ...cached.getState().model,
          contextWindow: explicitContextWindow ?? cached.getState().model.contextWindow,
          maxTokens: explicitMaxTokens ?? cached.getState().model.maxTokens
        })
      }
      return cached
    }

    const remote = await this.broker.session(sessionId)
    const [prompt, tools, profile, defaults] = await Promise.all([
      this.broker.systemPrompt(sessionId),
      this.broker.tools(),
      this.broker.textProfile(remote.session.model).catch(() => null),
      this.broker.runtimeInferenceSettings().catch(() => ({
        context_size: DEFAULT_CONTEXT_WINDOW,
        max_tokens: undefined,
        enable_reasoning: true,
        drop_reasoning_between_turns: false
      }))
    ])
    const contextWindow =
      explicitContextWindow ?? profile?.context_size ?? defaults.context_size
    const maxTokens =
      explicitMaxTokens ?? profile?.max_tokens ?? defaults.max_tokens ?? undefined

    return this.createSession({
      sessionId,
      model: {
        id: remote.session.model,
        name: remote.session.model,
        contextWindow,
        maxTokens
      },
      systemPrompt: prompt.system_prompt,
      tools,
      messages: remote.messages.map((record) => record.payload),
      capabilities: inferModelCapabilities(remote.session.model),
      preloadedInference: defaults
    })
  }

  async dispose(): Promise<void> {
    for (const session of this.sessions.values()) {
      await session.dispose()
    }
    this.sessions.clear()
    this.childrenByParent.clear()
    this.activeChildSlots.clear()
  }
}

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

/**
 * Oh My Pi (OMP) runtime adapter.
 *
 * This is the ONLY tree that may spawn or speak to `omp`. Everything outside
 * speaks Brazier's own types. OMP owns the fuller coding tool surface
 * (hashline edit, LSP/DAP, embedded shell, task subagents). Brazier maps RPC
 * events into the shared agent timeline and persists the transcript for UI.
 *
 * Trust: unlike Pi, machine effects are not mediated by brazierd's broker.
 * The Manage → Agent section explains that difference.
 */

import { randomUUID } from 'node:crypto'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import type { BrokerClient } from '../core/brokerClient'
import { inferModelCapabilities } from '../core/modelCompat'
import { accumulate, emptySummary } from '../core/runSummary'
import { EventSequencer } from '../core/toolExecutor'
import type {
  AgentCompactionState,
  AgentEvent,
  AgentMessage,
  AgentModelCapabilities,
  AgentModelReference,
  AgentPermissionMode,
  AgentPermissionSettings,
  AgentRunStatus,
  AgentRunSummary,
  AgentRuntime,
  AgentRuntimeDescriptor,
  AgentSession,
  AgentSessionState,
  AgentToolCallSummary,
  AgentToolDefinition,
  AgentUserInput,
  CreateAgentSessionOptions,
  SandboxDescription,
  ToolExecutionRecord
} from '../core/types'
import { detectOmpBinary } from './detect'
import { OmpRpcClient, type OmpRpcFrame } from './rpcClient'

const DESCRIPTOR: AgentRuntimeDescriptor = {
  id: 'omp',
  name: 'Oh My Pi',
  version: 'rpc',
  capabilities: {
    streaming: true,
    toolCalls: true,
    compaction: true,
    cancellation: true,
    sessionRestore: true
  }
}

function approvalModeFor(permission: AgentPermissionMode): 'always-ask' | 'write' | 'yolo' {
  switch (permission) {
    case 'skip-permissions':
      return 'yolo'
    case 'sandbox-only':
      // OMP has no OS sandbox equivalent; degrade to the strictest prompt tier.
      return 'always-ask'
    case 'ask':
    default:
      return 'always-ask'
  }
}

export function ompApprovalModeArg(mode: ReturnType<typeof approvalModeFor>): string {
  return `--approval-mode=${mode}`
}

function hostSandbox(detail: string): SandboxDescription {
  return {
    backend: 'omp',
    profile: 'host',
    isolated: false,
    network: true,
    detail
  }
}

type OmpSidecarOptions = {
  binary: string
  cwd: string
  env: NodeJS.ProcessEnv
  agentDir: string
}

function textFromContent(content: unknown): {
  text: string
  reasoning?: string
  toolCalls: AgentToolCallSummary[]
} {
  if (typeof content === 'string') {
    return { text: content, toolCalls: [] }
  }
  if (!Array.isArray(content)) {
    return { text: '', toolCalls: [] }
  }
  let text = ''
  let reasoning = ''
  const toolCalls: AgentToolCallSummary[] = []
  for (const part of content) {
    if (!part || typeof part !== 'object') continue
    const entry = part as Record<string, unknown>
    if (entry.type === 'text' && typeof entry.text === 'string') text += entry.text
    if (entry.type === 'thinking' && typeof entry.thinking === 'string') reasoning += entry.thinking
    if (entry.type === 'toolCall' || entry.type === 'tool_use') {
      toolCalls.push({
        id: String(entry.id ?? randomUUID()),
        name: String(entry.name ?? 'tool'),
        arguments:
          entry.arguments && typeof entry.arguments === 'object'
            ? (entry.arguments as Record<string, unknown>)
            : {}
      })
    }
  }
  return { text, reasoning: reasoning || undefined, toolCalls }
}

/**
 * OMP emits `message_end` for user, assistant, and tool-result messages.
 * Only assistant messages belong in Brazier's visible response stream.
 */
export function assistantContentFromOmpMessage(
  message: unknown
): ReturnType<typeof textFromContent> | null {
  if (!message || typeof message !== 'object') return null
  const record = message as Record<string, unknown>
  if (record.role !== 'assistant') return null
  return textFromContent(record.content)
}

function isMcpTool(tool: AgentToolDefinition): boolean {
  return tool.name.includes('__') || tool.name.startsWith('mcp_')
}

/** OMP's host-tool transport deliberately uses structured content, not a string. */
export function hostToolResultFrame(
  id: string,
  text: string,
  isError = false
): OmpRpcFrame {
  return {
    type: 'host_tool_result',
    id,
    result: { content: [{ type: 'text', text }] },
    ...(isError ? { isError: true } : {})
  }
}

/**
 * RPC mode has no transcript-import command. Seed only Brazier's prior turns
 * into a fresh sidecar; OMP remains the sole owner of its system prompt.
 */
export function promptWithBrazierHistory(
  history: AgentMessage[],
  userText: string,
  maxChars = 240_000
): string {
  const sections: string[] = []
  if (history.length > 0) {
    const transcript = history
      .map((message) => {
        switch (message.role) {
          case 'assistant':
            return `[assistant]\n${message.text}${message.reasoning ? `\n[reasoning]\n${message.reasoning}` : ''}`
          case 'tool':
            return `[tool ${message.tool}${message.isError ? ' error' : ''}]\n${message.output}`
          default:
            return `[${message.role}]\n${message.text}`
        }
      })
      .join('\n\n')
    // Keep the newest turns, which are the useful part of a restored context.
    sections.push(`## Prior Brazier transcript\n${transcript.slice(-maxChars)}`)
  }
  if (sections.length === 0) return userText
  return `${sections.join('\n\n')}\n\n## Current user request\n${userText}`
}

/** OMP-supported custom provider definition for Brazier's authenticated API. */
export function ompBrazierModelsConfig(
  baseUrl: string,
  model: AgentModelReference,
  capabilities: AgentModelCapabilities
): Record<string, unknown> {
  return {
    providers: {
      brazier: {
        baseUrl,
        apiKey: 'BRAZIER_OPENAI_API_KEY',
        authHeader: true,
        api: 'openai-completions',
        discovery: { type: 'openai-models-list' },
        models: [
          {
            id: model.id,
            name: model.name || model.id,
            reasoning: capabilities.supportsReasoningStream,
            // Brazier's current capability contract does not report vision support.
            // Declaring image input makes OMP send images to text-only local models.
            input: ['text'],
            supportsTools: capabilities.nativeToolCalling,
            ...(model.contextWindow ? { contextWindow: model.contextWindow } : {}),
            ...(model.maxTokens ? { maxTokens: model.maxTokens } : {})
          }
        ]
      }
    }
  }
}

async function createOmpAgentDir(
  baseUrl: string,
  model: AgentModelReference,
  capabilities: AgentModelCapabilities,
  configYaml?: string
): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'brazier-omp-'))
  try {
    await writeFile(
      join(directory, 'models.yml'),
      `${JSON.stringify(ompBrazierModelsConfig(baseUrl, model, capabilities), null, 2)}\n`,
      { mode: 0o600 }
    )
    if (configYaml?.trim()) {
      await writeFile(join(directory, 'config.yml'), configYaml, { mode: 0o600 })
    }
    return directory
  } catch (cause) {
    await rm(directory, { recursive: true, force: true })
    throw cause
  }
}

/** A failed sidecar reconfiguration must leave the existing sidecar's context intact. */
export function contextSeedAfterPermissionModeAttempt(
  previousNeedsSeed: boolean,
  brokerUpdated: boolean
): boolean {
  return brokerUpdated ? true : previousNeedsSeed
}

/** Prevent a late host-tool callback from following a newer run's controller. */
export function isCurrentOmpRun(
  activeRunId: string | undefined,
  requestedRunId: string,
  aborted: boolean
): boolean {
  return activeRunId === requestedRunId && !aborted
}

export class OmpAgentRuntime implements AgentRuntime {
  readonly descriptor = DESCRIPTOR
  private readonly broker: BrokerClient
  private readonly sessions = new Map<string, OmpAgentSession>()

  constructor(broker: BrokerClient) {
    this.broker = broker
  }

  async createSession(options: CreateAgentSessionOptions): Promise<AgentSession> {
    const session = await OmpAgentSession.open(this.broker, options)
    this.sessions.set(session.id, session)
    return session
  }

  async restoreSession(sessionId: string): Promise<AgentSession> {
    const existing = this.sessions.get(sessionId)
    if (existing && !existing.isDisposed()) return existing
    const remote = await this.broker.session(sessionId)
    const tools = await this.broker.tools()
    return this.createSession({
      sessionId,
      model: { id: remote.session.model, name: remote.session.model },
      systemPrompt: '',
      tools,
      messages: remote.messages.map((record) => record.payload),
      capabilities: inferModelCapabilities(remote.session.model),
      preloaded: {
        session: {
          id: remote.session.id,
          title: remote.session.title,
          workspace_path: remote.session.workspace_path,
          permission_mode: remote.session.permission_mode,
          permission_settings: remote.session.permission_settings,
          enabled_tools: remote.session.enabled_tools,
          created_at: remote.session.created_at,
          updated_at: remote.session.updated_at,
          runtime_id: remote.session.runtime_id,
          runtime_metadata: remote.session.runtime_metadata as Record<string, unknown> | null
        },
        tool_executions: remote.tool_executions,
        sandbox: hostSandbox(remote.sandbox.detail)
      }
    })
  }

  async dispose(): Promise<void> {
    for (const session of this.sessions.values()) {
      await session.dispose()
    }
    this.sessions.clear()
  }
}

class OmpAgentSession implements AgentSession {
  readonly id: string
  private readonly broker: BrokerClient
  private client: OmpRpcClient | null = null
  private readonly sidecar: OmpSidecarOptions
  private disposed = false
  private model: AgentModelReference
  private messages: AgentMessage[]
  private toolCatalog: AgentToolDefinition[]
  private toolExecutions: ToolExecutionRecord[]
  private permissionMode: AgentPermissionMode
  private permissionSettings: AgentPermissionSettings
  private enabledTools?: string[]
  private title: string
  private workspacePath?: string | null
  private createdAt: string
  private updatedAt: string
  private lastRunStatus: AgentRunStatus = 'idle'
  private compactionState?: AgentCompactionState
  private runtimeMetadata?: Record<string, unknown>
  private readonly sandbox: SandboxDescription
  private persistence = Promise.resolve()
  private activeRun: { runId: string; abort: boolean; controller: AbortController } | null = null
  private needsContextSeed: boolean
  private readonly sequencer = new EventSequencer()

  private constructor(
    broker: BrokerClient,
    options: CreateAgentSessionOptions,
    client: OmpRpcClient,
    sidecar: OmpSidecarOptions
  ) {
    this.broker = broker
    this.id = options.sessionId
    this.client = client
    this.sidecar = sidecar
    this.model = options.model
    this.messages = [...(options.messages ?? [])]
    this.toolCatalog = [...options.tools]
    this.needsContextSeed = this.messages.length > 0
    const preloaded = options.preloaded
    this.toolExecutions = [...(preloaded?.tool_executions ?? [])]
    this.permissionMode = preloaded?.session.permission_mode ?? 'ask'
    this.permissionSettings = preloaded?.session.permission_settings ?? {
      auto_approve_host_actions: false,
      auto_approve_sandboxed_actions: true
    }
    this.enabledTools = preloaded?.session.enabled_tools ?? undefined
    this.title = preloaded?.session.title ?? 'Agent task'
    this.workspacePath = preloaded?.session.workspace_path
    this.createdAt = preloaded?.session.created_at ?? new Date().toISOString()
    this.updatedAt = preloaded?.session.updated_at ?? this.createdAt
    this.runtimeMetadata = preloaded?.session.runtime_metadata ?? undefined
    this.sandbox =
      preloaded?.sandbox ??
      hostSandbox(
        "Oh My Pi runs as a privileged harness. Tool effects are not mediated by Brazier's OS sandbox."
      )
  }

  static async open(
    broker: BrokerClient,
    options: CreateAgentSessionOptions
  ): Promise<OmpAgentSession> {
    const preference = await broker.agentPreference().catch(() => null)
    const profile = preference?.omp_profile ?? undefined
    const binary = detectOmpBinary(profile?.binary_path)
    if (!binary) {
      throw new Error(
        'Oh My Pi (`omp`) was not found. Install it from https://omp.sh/ or set BRAZIER_OMP_PATH.'
      )
    }
    const permissionMode = options.preloaded?.session.permission_mode ?? 'ask'
    const approvalMode = approvalModeFor(permissionMode)
    const cwd = options.preloaded?.session.workspace_path || process.cwd()
    const agentDir = await createOmpAgentDir(
      broker.openAiBaseUrl(),
      options.model,
      options.capabilities,
      profile?.config_yaml
    )
    const sidecar: OmpSidecarOptions = {
      binary: binary.path,
      cwd,
      agentDir,
      env: {
        ...process.env,
        PI_CODING_AGENT_DIR: agentDir,
        OPENAI_BASE_URL: broker.openAiBaseUrl(),
        OPENAI_API_KEY: broker.apiKey(),
        OPENAI_API_BASE: broker.openAiBaseUrl(),
        BRAZIER_OPENAI_BASE_URL: broker.openAiBaseUrl(),
        BRAZIER_OPENAI_API_KEY: broker.apiKey()
      }
    }
    let client: OmpRpcClient | null = null
    try {
      client = await OmpAgentSession.startSidecar(sidecar, approvalMode)
      const session = new OmpAgentSession(broker, options, client, sidecar)
      await session.configureModel(options.model)
      await session.configureHostTools()
      return session
    } catch (cause) {
      await client?.dispose()
      await rm(agentDir, { recursive: true, force: true })
      throw cause
    }
  }

  isDisposed(): boolean {
    return this.disposed
  }

  getState(): AgentSessionState {
    return {
      id: this.id,
      title: this.title,
      workspacePath: this.workspacePath,
      model: this.model,
      runtimeId: DESCRIPTOR.id,
      messages: [...this.messages],
      toolExecutions: [...this.toolExecutions],
      permissionMode: this.permissionMode,
      permissionSettings: this.permissionSettings,
      enabledTools: this.enabledTools,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt,
      lastRunStatus: this.lastRunStatus,
      compactionState: this.compactionState,
      runtimeMetadata: this.runtimeMetadata
    }
  }

  rehydrate(messages: AgentMessage[], _systemPrompt?: string): void {
    if (this.disposed) throw new Error('Cannot rehydrate a disposed agent session.')
    this.messages = [...messages]
  }

  async refreshInferencePrefs(): Promise<void> {
    // OMP sampling is owned by the sidecar; Brazier inference prefs still apply
    // to the daemon model endpoint the sidecar calls.
  }

  async setModel(model: AgentModelReference): Promise<void> {
    await this.configureModel(model)
    this.model = model
    await this.broker.updateSession(this.id, { model: model.id })
  }

  async setEnabledTools(toolNames: string[]): Promise<void> {
    const tools = await this.broker.tools()
    this.toolCatalog = tools
    await this.configureHostTools(toolNames)
    this.enabledTools = toolNames
    await this.broker.updateSession(this.id, { enabled_tools: toolNames })
  }

  async setPermissionMode(mode: AgentPermissionMode): Promise<void> {
    if (this.activeRun) {
      throw new Error('Cancel the active OMP run before changing its permission mode.')
    }
    if (mode === this.permissionMode) return
    const replacement = await OmpAgentSession.startSidecar(this.sidecar, approvalModeFor(mode))
    const previous = this.requireClient()
    const previousNeedsContextSeed = this.needsContextSeed
    this.client = replacement
    try {
      await this.configureModel(this.model)
      await this.configureHostTools()
      this.permissionMode = mode
      await this.broker.updateSession(this.id, { permission_mode: mode })
      // Changing OMP's startup-only approval mode creates a fresh sidecar.
      // Its model context must be reconstructed before its next prompt.
      this.needsContextSeed = contextSeedAfterPermissionModeAttempt(previousNeedsContextSeed, true)
    } catch (cause) {
      this.client = previous
      this.needsContextSeed = contextSeedAfterPermissionModeAttempt(previousNeedsContextSeed, false)
      await replacement.dispose()
      throw cause
    }
    await previous.dispose()
  }

  async cancel(): Promise<void> {
    if (this.activeRun) {
      this.activeRun.abort = true
      this.activeRun.controller.abort()
    }
    await this.client?.request({ type: 'abort' }).catch(() => undefined)
    await this.broker.cancel(this.id).catch(() => undefined)
    this.lastRunStatus = 'cancelled'
  }

  async compact(): Promise<AgentCompactionState> {
    await this.requireClient().request({ type: 'compact' })
    const state: AgentCompactionState = {
      compactedAt: new Date().toISOString(),
      removedMessages: 0,
      summary: 'Compacted by Oh My Pi.',
      summarySource: 'deterministic'
    }
    this.compactionState = state
    await this.broker.updateSession(this.id, {
      compaction: state,
      last_run_status: this.lastRunStatus
    })
    return state
  }

  async *run(input: AgentUserInput): AsyncIterable<AgentEvent> {
    if (this.disposed) throw new Error('Cannot run a disposed agent session.')
    const client = this.requireClient()
    const runId = randomUUID()
    this.activeRun = { runId, abort: false, controller: new AbortController() }
    this.lastRunStatus = 'running'
    await this.broker.updateSession(this.id, { last_run_status: 'running' })

    const userMessage: AgentMessage = {
      role: 'user',
      text: input.text,
      timestamp: new Date().toISOString()
    }
    this.messages.push(userMessage)
    void this.enqueuePersistence(() => this.broker.appendMessages(this.id, [userMessage]))

    const queue: AgentEvent[] = []
    let resolveWait: (() => void) | null = null
    let done = false
    let failure: string | null = null
    let summary: AgentRunSummary = emptySummary()
    let assistantText = ''
    let assistantReasoning = ''
    const assistantToolCalls: AgentToolCallSummary[] = []

    const push = (next: AgentEvent): void => {
      summary = accumulate(summary, next)
      queue.push(next)
      resolveWait?.()
      resolveWait = null
    }

    const event = (partial: Record<string, unknown> & { type: AgentEvent['type'] }): AgentEvent =>
      ({
        ...partial,
        sessionId: this.id,
        runId,
        timestamp: new Date().toISOString(),
        sequence: this.sequencer.take()
      }) as AgentEvent

    push(event({ type: 'run-started' }))
    push(event({ type: 'message-committed', message: userMessage }))

    const unsubscribe = client.onFrame((frame) => {
      const type = String(frame.type ?? '')
      if (type === 'agent_start') {
        // OMP documents prompt acknowledgement as only acceptance.  The
        // transcript is now known to have entered the agent loop only once an
        // agent-start event arrives; do not discard a recovery seed earlier.
        this.needsContextSeed = false
        return
      }
      if (type === 'host_tool_call') {
        void this.handleHostToolCall(frame, runId, push, event)
        return
      }
      if (type === 'extension_ui_request') {
        void this.handleExtensionUi(frame)
        return
      }
      if (type === 'message_update' || type === 'message_end') {
        const assistantMessage = (frame.assistantMessage ?? frame.message) as
          | Record<string, unknown>
          | undefined
        const inner = frame.assistantMessageEvent as Record<string, unknown> | undefined
        if (inner?.type === 'text_delta' && typeof inner.delta === 'string') {
          assistantText += inner.delta
          push(event({ type: 'text-delta', delta: inner.delta, channel: 'text' }))
        } else if (inner?.type === 'thinking_delta' && typeof inner.delta === 'string') {
          assistantReasoning += inner.delta
          push(event({ type: 'text-delta', delta: inner.delta, channel: 'reasoning' }))
        } else if (assistantMessage) {
          const parsed = assistantContentFromOmpMessage(assistantMessage)
          if (!parsed) return
          if (parsed.text && parsed.text.length > assistantText.length) {
            const delta = parsed.text.slice(assistantText.length)
            assistantText = parsed.text
            if (delta) push(event({ type: 'text-delta', delta, channel: 'text' }))
          }
          if (parsed.reasoning && parsed.reasoning.length > assistantReasoning.length) {
            const delta = parsed.reasoning.slice(assistantReasoning.length)
            assistantReasoning = parsed.reasoning
            if (delta) push(event({ type: 'text-delta', delta, channel: 'reasoning' }))
          }
          for (const call of parsed.toolCalls) {
            if (!assistantToolCalls.some((entry) => entry.id === call.id)) {
              assistantToolCalls.push(call)
            }
          }
        }
        return
      }
      if (type === 'tool_execution_start') {
        const toolCallId = String(frame.toolCallId ?? frame.id ?? randomUUID())
        const tool = String(frame.toolName ?? frame.name ?? 'tool')
        const args =
          frame.args && typeof frame.args === 'object'
            ? (frame.args as Record<string, unknown>)
            : frame.arguments && typeof frame.arguments === 'object'
              ? (frame.arguments as Record<string, unknown>)
              : {}
        push(
          event({
            type: 'tool-call-proposed',
            toolCallId,
            tool,
            args,
            environment: 'host',
            risk: 'execute'
          })
        )
        push(
          event({
            type: 'tool-started',
            toolCallId,
            tool,
            args,
            environment: 'host',
            sandbox: this.sandbox
          })
        )
        return
      }
      if (type === 'tool_execution_update') {
        const toolCallId = String(frame.toolCallId ?? frame.id ?? '')
        const tool = String(frame.toolName ?? frame.name ?? 'tool')
        const chunk =
          typeof frame.chunk === 'string'
            ? frame.chunk
            : typeof frame.output === 'string'
              ? frame.output
              : typeof frame.delta === 'string'
                ? frame.delta
                : ''
        if (chunk) push(event({ type: 'tool-output', toolCallId, tool, chunk }))
        return
      }
      if (type === 'tool_execution_end') {
        const toolCallId = String(frame.toolCallId ?? frame.id ?? randomUUID())
        const tool = String(frame.toolName ?? frame.name ?? 'tool')
        const isError = Boolean(frame.isError ?? frame.error)
        const output =
          typeof frame.result === 'string'
            ? frame.result
            : typeof frame.output === 'string'
              ? frame.output
              : typeof frame.error === 'string'
                ? frame.error
                : JSON.stringify(frame.result ?? frame.output ?? '')
        if (isError) {
          push(
            event({
              type: 'tool-failed',
              toolCallId,
              tool,
              environment: 'host',
              sandbox: this.sandbox,
              error: output || 'Tool failed.',
              denied: false,
              durationMs: Number(frame.durationMs ?? 0)
            })
          )
        } else {
          push(
            event({
              type: 'tool-completed',
              toolCallId,
              tool,
              environment: 'host',
              sandbox: this.sandbox,
              output,
              truncated: Boolean(frame.truncated),
              exitCode: typeof frame.exitCode === 'number' ? frame.exitCode : null,
              changedPaths: Array.isArray(frame.changedPaths)
                ? (frame.changedPaths as string[])
                : [],
              durationMs: Number(frame.durationMs ?? 0)
            })
          )
        }
        return
      }
      if (type === 'agent_end' || type === 'prompt_result') {
        done = true
        resolveWait?.()
        resolveWait = null
        return
      }
      if (type === 'response' && frame.success === false && typeof frame.error === 'string') {
        failure = frame.error
        done = true
        resolveWait?.()
        resolveWait = null
      }
    })

    try {
      const promptImages = (input.images ?? []).map((url) => ({
        type: 'image',
        source: { type: 'data_url', dataUrl: url }
      }))
      const message = this.needsContextSeed
        ? promptWithBrazierHistory(
            this.messages.slice(0, -1),
            input.text,
            Math.max(24_000, Math.min((this.model.contextWindow ?? 80_000) * 3, 240_000))
          )
        : input.text
      const response = await client.request({
        type: 'prompt',
        message,
        ...(promptImages.length > 0 ? { images: promptImages } : {})
      })
      const agentInvoked = (response.data as { agentInvoked?: boolean } | undefined)?.agentInvoked
      if (agentInvoked === false) done = true

      while (!done && !this.activeRun?.abort) {
        if (queue.length === 0) {
          await new Promise<void>((resolve) => {
            resolveWait = resolve
          })
          continue
        }
        yield queue.shift()!
      }
      while (queue.length > 0) yield queue.shift()!

      if (this.activeRun?.abort) {
        this.lastRunStatus = 'cancelled'
        yield event({ type: 'run-cancelled' })
      } else if (failure) {
        this.lastRunStatus = 'failed'
        yield event({ type: 'run-failed', error: failure })
      } else {
        const assistantMessage: AgentMessage = {
          role: 'assistant',
          text: assistantText,
          reasoning: assistantReasoning || undefined,
          toolCalls: assistantToolCalls.length > 0 ? assistantToolCalls : undefined,
          timestamp: new Date().toISOString()
        }
        if (assistantText || assistantReasoning || assistantToolCalls.length > 0) {
          this.messages.push(assistantMessage)
          void this.enqueuePersistence(() => this.broker.appendMessages(this.id, [assistantMessage]))
          yield event({ type: 'message-committed', message: assistantMessage })
        }
        summary = { ...summary, text: assistantText }
        this.lastRunStatus = 'completed'
        yield event({ type: 'run-completed', summary })
      }
    } catch (cause) {
      const error = cause instanceof Error ? cause.message : String(cause)
      this.lastRunStatus = 'failed'
      yield event({ type: 'run-failed', error })
    } finally {
      unsubscribe()
      if (this.activeRun?.runId === runId) {
        // A host-tool callback can still be waiting on a daemon approval when
        // OMP emits agent_end.  Abort it before dropping the run reference so
        // it cannot continue into a subsequent run.
        this.activeRun.controller.abort()
        this.activeRun = null
      }
      await this.broker
        .updateSession(this.id, { last_run_status: this.lastRunStatus })
        .catch(() => undefined)
    }
  }

  async dispose(): Promise<void> {
    if (this.disposed) return
    this.disposed = true
    await this.client?.dispose()
    this.client = null
    await rm(this.sidecar.agentDir, { recursive: true, force: true })
  }

  private async configureModel(model: AgentModelReference): Promise<void> {
    const client = this.requireClient()
    try {
      await client.request({ type: 'set_model', provider: 'brazier', modelId: model.id })
    } catch (cause) {
      const detail = cause instanceof Error ? cause.message : String(cause)
      throw new Error(`OMP could not select Brazier model \`${model.id}\`: ${detail}`)
    }
  }

  private static async startSidecar(
    options: OmpSidecarOptions,
    permissionMode: ReturnType<typeof approvalModeFor>
  ): Promise<OmpRpcClient> {
    const client = new OmpRpcClient({
      binary: options.binary,
      cwd: options.cwd,
      env: options.env,
      // Public OMP CLI flag (v17+). Dotted config keys are not CLI options and
      // make oclif terminate with a usage error before RPC becomes ready.
      args: [ompApprovalModeArg(permissionMode)]
    })
    try {
      await client.waitUntilReady()
      return client
    } catch (cause) {
      await client.dispose()
      throw cause
    }
  }

  private async configureHostTools(enabledTools = this.enabledTools): Promise<void> {
    const tools = this.toolCatalog
      .filter(isMcpTool)
      .filter((tool) => !enabledTools || enabledTools.includes(tool.name))
    await this.requireClient().request({
      type: 'set_host_tools',
      tools: tools.map((tool) => ({
        name: tool.name,
        label: tool.label,
        description: tool.description,
        parameters: tool.inputSchema
      }))
    })
  }

  private async handleHostToolCall(
    frame: OmpRpcFrame,
    runId: string,
    push: (event: AgentEvent) => void,
    event: (partial: Record<string, unknown> & { type: AgentEvent['type'] }) => AgentEvent
  ): Promise<void> {
    const client = this.requireClient()
    const activeRun = this.activeRun
    if (
      !activeRun ||
      !isCurrentOmpRun(activeRun.runId, runId, activeRun.controller.signal.aborted)
    ) {
      client.send(hostToolResultFrame(String(frame.id ?? ''), 'The originating agent run has ended.', true))
      return
    }
    const signal = activeRun.controller.signal
    const requestId = String(frame.id ?? '')
    const toolCallId = String(frame.toolCallId ?? randomUUID())
    const toolName = String(frame.toolName ?? 'tool')
    const args =
      frame.arguments && typeof frame.arguments === 'object'
        ? (frame.arguments as Record<string, unknown>)
        : {}
    push(
      event({
        type: 'tool-call-proposed',
        toolCallId,
        tool: toolName,
        args,
        environment: 'host',
        risk: 'execute'
      })
    )
    try {
      let approvalId: string | undefined
      let result
      while (true) {
        result = await this.broker.execTool({
          sessionId: this.id,
          runId,
          toolCallId,
          tool: toolName,
          arguments: args,
          environment: 'host',
          approvalId
        }, signal)
        if (result.status !== 'approval_required' || !result.approval) break
        this.lastRunStatus = 'awaiting-approval'
        push(event({ type: 'approval-required', toolCallId, approval: result.approval }))
        push(
          event({
            type: 'elevation-requested',
            toolCallId,
            approvalId: result.approval.id,
            request: result.approval.elevation
          })
        )
        const decision = await this.waitForApproval(result.approval, runId, signal)
        if (decision.status !== 'approved') {
          const denied = decision.status === 'denied'
          const reason = denied
            ? `The user denied this action.${decision.note ? ` Note: ${decision.note}` : ''}`
            : 'The approval request expired or the run was cancelled.'
          client.send(hostToolResultFrame(requestId, reason, true))
          push(
            event({
              type: 'tool-failed', toolCallId, tool: toolName, environment: 'host', sandbox: this.sandbox,
              error: reason, denied, durationMs: Number(result.duration_ms ?? 0)
            })
          )
          return
        }
        approvalId = result.approval.id
        this.lastRunStatus = 'running'
      }
      const output = result.output || result.denied_reason || ''
      const isError = Boolean(result.is_error || result.status === 'denied')
      client.send(hostToolResultFrame(requestId, output, isError))
      if (isError) {
        push(
          event({
            type: 'tool-failed',
            toolCallId,
            tool: toolName,
            environment: 'host',
            sandbox: this.sandbox,
            error: output,
            denied: result.status === 'denied' || Boolean(result.denied_reason),
            durationMs: Number(result.duration_ms ?? 0)
          })
        )
      } else {
        push(
          event({
            type: 'tool-completed',
            toolCallId,
            tool: toolName,
            environment: 'host',
            sandbox: this.sandbox,
            output,
            truncated: Boolean(result.truncated),
            exitCode: result.exit_code ?? null,
            changedPaths: result.changed_paths ?? [],
            durationMs: Number(result.duration_ms ?? 0),
            executionId: result.execution_id
          })
        )
      }
    } catch (cause) {
      const error = cause instanceof Error ? cause.message : String(cause)
      client.send(hostToolResultFrame(requestId, error, true))
      push(
        event({
          type: 'tool-failed',
          toolCallId,
          tool: toolName,
          environment: 'host',
          sandbox: this.sandbox,
          error,
          denied: false,
          durationMs: 0
        })
      )
    }
  }

  private async waitForApproval(
    approval: import('../core/types').AgentApproval,
    runId: string,
    signal: AbortSignal
  ) {
    while (
      isCurrentOmpRun(this.activeRun?.runId, runId, signal.aborted) &&
      !this.activeRun?.abort
    ) {
      try {
        const current = await this.broker.waitForApproval(
          approval.id,
          30_000,
          signal
        )
        if (current.status !== 'pending') return current
      } catch {
        return { ...approval, status: 'expired' as const }
      }
    }
    return { ...approval, status: 'expired' as const }
  }

  private async handleExtensionUi(frame: OmpRpcFrame): Promise<void> {
    const client = this.requireClient()
    const id = String(frame.id ?? '')
    const method = String(frame.method ?? '')
    if (method === 'notify' || method === 'setStatus' || method === 'setTitle') {
      client.send({ type: 'extension_ui_response', id, confirmed: true })
      return
    }
    if (method === 'confirm') {
      const auto = this.permissionMode === 'skip-permissions'
      client.send({
        type: 'extension_ui_response',
        id,
        confirmed: auto,
        cancelled: !auto
      })
      return
    }
    client.send({ type: 'extension_ui_response', id, cancelled: true })
  }

  private requireClient(): OmpRpcClient {
    if (!this.client) throw new Error('Oh My Pi session has no RPC client.')
    return this.client
  }

  private enqueuePersistence(task: () => Promise<void>): Promise<void> {
    const next = this.persistence.then(task, task)
    this.persistence = next.then(
      () => undefined,
      () => undefined
    )
    return next
  }
}

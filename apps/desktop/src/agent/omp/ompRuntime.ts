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
  AgentComposerSuggestion,
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
import {
  configYamlWithModelRoles,
  configYamlWithSettings,
  sanitizeOmpSettings
} from './ompSettings'
import type { OmpExtensionUiResponse } from './rpcTypes'
import { OmpRpcClient, type OmpRpcFrame } from './rpcClient'

/** How long the runtime holds a surfaced dialog before unblocking the sidecar. */
function dialogTimeoutMs(): number {
  const raw = process.env.BRAZIER_OMP_DIALOG_TIMEOUT_MS
  const parsed = raw ? Number(raw) : NaN
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 120_000
}

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

/**
 * Append a streamed delta to the running message text.
 *
 * OMP's `message.content` snapshots can run ahead of the `text_delta` stream,
 * so the first token(s) sometimes arrive once via the snapshot diff and again
 * when the delta stream restarts from the beginning. A delta that is a prefix
 * of everything accumulated so far is a restart, not new text; dropping it
 * keeps the first token from appearing twice.
 */
export function appendStreamedDelta(current: string, delta: string): string {
  if (delta.length > 0 && current.length >= delta.length && current.startsWith(delta)) {
    return current
  }
  return current + delta
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

/** One Brazier chat model advertised to OMP, with its routing hints. */
export type OmpBrazierModel = {
  id: string
  name: string
  contextWindow?: number
  maxTokens?: number
  reasoning: boolean
  supportsTools: boolean
  /** Whether the model accepts image input (drives the `vision` role). */
  vision: boolean
}

/**
 * OMP-supported custom provider definition for Brazier's authenticated API.
 * Advertises every chat model the daemon serves so OMP's role routing
 * (`smol`/`slow`/`plan`/`vision`/`task`/`advisor`/…) can pick appropriate local
 * models instead of forcing every role through the single session model.
 */
export function ompBrazierModelsConfig(
  baseUrl: string,
  models: OmpBrazierModel[]
): Record<string, unknown> {
  return {
    providers: {
      brazier: {
        baseUrl,
        apiKey: 'BRAZIER_OPENAI_API_KEY',
        authHeader: true,
        api: 'openai-completions',
        discovery: { type: 'openai-models-list' },
        models: models.map((entry) => ({
          id: entry.id,
          name: entry.name,
          reasoning: entry.reasoning,
          input: entry.vision ? ['text', 'image'] : ['text'],
          supportsTools: entry.supportsTools,
          ...(entry.contextWindow ? { contextWindow: entry.contextWindow } : {}),
          ...(entry.maxTokens ? { maxTokens: entry.maxTokens } : {})
        }))
      }
    }
  }
}

/** Remove a top-level YAML key and its indented block, leaving the rest intact. */
export { stripTopLevelKey } from './ompSettings'

/**
 * Merge `modelRoles` into a config YAML string. The block replaces any existing
 * top-level `modelRoles:` so the profile editor is authoritative without the
 * runtime needing a YAML parser.
 */
export { configYamlWithModelRoles } from './ompSettings'

/**
 * Build the provider catalog from the daemon's chat model list, always ensuring
 * the session's selected model is present (the catalog is what `get_available_models`
 * and role routing see). Capabilities come from the daemon's own report; the
 * selected model falls back to the inference heuristic when the list is empty.
 */
export function buildBrazierModelCatalog(
  daemonModels: Array<{ id?: string; capabilities?: { input_modalities?: string[]; tools?: boolean; reasoning?: boolean; max_context_length?: number | null } | null }> | null,
  selected: AgentModelReference,
  selectedCapabilities: AgentModelCapabilities
): OmpBrazierModel[] {
  const seen = new Set<string>()
  const catalog: OmpBrazierModel[] = []
  for (const entry of daemonModels ?? []) {
    const id = entry?.id
    if (!id || seen.has(id)) continue
    seen.add(id)
    const caps = entry.capabilities
    catalog.push({
      id,
      name: id,
      reasoning: Boolean(caps?.reasoning),
      supportsTools: caps?.tools === true,
      vision: Boolean(caps?.input_modalities?.includes('image')),
      contextWindow: caps?.max_context_length ?? undefined
    })
  }
  if (!seen.has(selected.id)) {
    catalog.unshift({
      id: selected.id,
      name: selected.name || selected.id,
      reasoning: selectedCapabilities.supportsReasoningStream,
      supportsTools: selectedCapabilities.nativeToolCalling,
      vision: false,
      ...(selected.contextWindow ? { contextWindow: selected.contextWindow } : {}),
      ...(selected.maxTokens ? { maxTokens: selected.maxTokens } : {})
    })
  }
  return catalog
}

async function createOmpAgentDir(
  baseUrl: string,
  models: OmpBrazierModel[],
  configYaml?: string
): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'brazier-omp-'))
  try {
    await writeFile(
      join(directory, 'models.yml'),
      `${JSON.stringify(ompBrazierModelsConfig(baseUrl, models), null, 2)}\n`,
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
  private commandSuggestions: AgentComposerSuggestion[] = []
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
  /** Extension-UI dialogs waiting on the user, keyed by the sidecar's dialog id. */
  private readonly pendingDialogs = new Map<string, { timer: NodeJS.Timeout }>()

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
    this.attachExtensionUiHandling()
  }

  static async open(
    broker: BrokerClient,
    options: CreateAgentSessionOptions
  ): Promise<OmpAgentSession> {
    const [preference, daemonModels] = await Promise.all([
      broker.agentPreference().catch(() => null),
      broker.models().catch(() => null)
    ])
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
    const catalog = buildBrazierModelCatalog(daemonModels, options.model, options.capabilities)
    const agentDir = await createOmpAgentDir(
      broker.openAiBaseUrl(),
      catalog,
      configYamlWithSettings(
        configYamlWithModelRoles(profile?.config_yaml, profile?.model_roles ?? {}),
        sanitizeOmpSettings(profile?.settings)
      )
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
      await session.enableSubagentSubscription()
      await session.refreshCommandSuggestions()
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

  async composerSuggestions(): Promise<AgentComposerSuggestion[]> {
    await this.refreshCommandSuggestions()
    return [
      { value: 'ultrathink', description: 'Use the highest supported reasoning effort for this turn.' },
      { value: 'orchestrate', description: 'Plan, delegate independent work, and verify the result.' },
      { value: 'workflowz', description: 'Build a deterministic multi-subagent workflow when task tools are available.' },
      ...this.commandSuggestions
    ]
  }

  /**
   * Mirror the sidecar's raw RPC stdout stream. The worker forwards each frame
   * to the renderer so OMP-specific state (command output, live command lists,
   * notices, config/session updates, subagent frames) can be rendered without
   * squeezing it through the shared event model. Frames are forwarded losslessly
   * — unknown frame types are never dropped before the renderer sees them.
   */
  subscribeRuntimeFrames(listener: (payload: Record<string, unknown>) => void): () => void {
    return this.requireClient().onFrame(listener)
  }

  /**
   * Send an arbitrary typed RPC command to the sidecar and resolve its raw
   * response frame. This is the escape hatch the worker protocol needs so the
   * GUI can drive any OMP surface (get_state, roles, subagents, bash, …)
   * without a new worker command per feature.
   */
  sendRuntimeCommand(command: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.requireClient().request(command)
  }

  /**
   * Answer an extension-UI dialog the sidecar is waiting on. The worker routes
   * this to the session via the `resolve-extension-ui` command; OMP dialog
   * ids are validated here so a stale response cannot leak to the sidecar.
   */
  async resolveExtensionUi(response: Record<string, unknown>): Promise<Record<string, unknown>> {
    if (
      !response ||
      typeof response !== 'object' ||
      typeof (response as OmpExtensionUiResponse).id !== 'string'
    ) {
      throw new Error('Malformed extension-UI resolution.')
    }
    return this.resolveExtensionUiResponse(response as OmpExtensionUiResponse)
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
    // Any dialog the old sidecar held is gone with it.
    this.clearPendingDialogs()
    this.client = replacement
    this.attachExtensionUiHandling()
    try {
      await this.configureModel(this.model)
      await this.configureHostTools()
      await this.enableSubagentSubscription()
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
      if (type === 'message_update' || type === 'message_end') {
        const assistantMessage = (frame.assistantMessage ?? frame.message) as
          | Record<string, unknown>
          | undefined
        const inner = frame.assistantMessageEvent as Record<string, unknown> | undefined
        if (inner?.type === 'text_delta' && typeof inner.delta === 'string') {
          const next = appendStreamedDelta(assistantText, inner.delta)
          if (next !== assistantText) {
            assistantText = next
            push(event({ type: 'text-delta', delta: inner.delta, channel: 'text' }))
          }
        } else if (inner?.type === 'thinking_delta' && typeof inner.delta === 'string') {
          const next = appendStreamedDelta(assistantReasoning, inner.delta)
          if (next !== assistantReasoning) {
            assistantReasoning = next
            push(event({ type: 'text-delta', delta: inner.delta, channel: 'reasoning' }))
          }
        } else if (assistantMessage) {
          const parsed = assistantContentFromOmpMessage(assistantMessage)
          if (!parsed) return
          // The snapshot must be a strict extension of the deltas already
          // streamed, never a replayed prefix, so the tail pushed here cannot
          // duplicate text the delta stream already delivered. When the delta
          // stream diverged from the snapshot, still adopt the snapshot as the
          // authoritative committed text — OMP's message content is what the
          // sidecar actually produced.
          if (
            parsed.text &&
            parsed.text.length > assistantText.length &&
            parsed.text.startsWith(assistantText)
          ) {
            const delta = parsed.text.slice(assistantText.length)
            if (delta) push(event({ type: 'text-delta', delta, channel: 'text' }))
          }
          if (parsed.text && parsed.text.length > assistantText.length) {
            assistantText = parsed.text
          }
          if (
            parsed.reasoning &&
            parsed.reasoning.length > assistantReasoning.length &&
            parsed.reasoning.startsWith(assistantReasoning)
          ) {
            const delta = parsed.reasoning.slice(assistantReasoning.length)
            if (delta) push(event({ type: 'text-delta', delta, channel: 'reasoning' }))
          }
          if (parsed.reasoning && parsed.reasoning.length > assistantReasoning.length) {
            assistantReasoning = parsed.reasoning
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
    this.clearPendingDialogs()
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

  private async refreshCommandSuggestions(): Promise<void> {
    const frame = await this.requireClient()
      .request({ type: 'get_available_commands' })
      .catch(() => null)
    const commands = (frame?.data as { commands?: unknown[] } | undefined)?.commands
    if (!Array.isArray(commands)) return
    this.commandSuggestions = commands.flatMap((command) => {
      if (!command || typeof command !== 'object') return []
      const entry = command as Record<string, unknown>
      const raw = typeof entry.name === 'string' ? entry.name : typeof entry.command === 'string' ? entry.command : ''
      if (!raw) return []
      return [{
        value: raw.startsWith('/') ? raw : `/${raw}`,
        description: typeof entry.description === 'string' ? entry.description : 'OMP command'
      }]
    })
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

  /**
   * Subscribe the GUI to subagent lifecycle/progress frames. Best-effort: an
   * OMP build without a subagent event bus leaves the panel empty rather than
   * failing the session. Re-applied after every fresh sidecar (open, permission
   * change) because the subscription does not survive a restart.
   */
  private async enableSubagentSubscription(): Promise<void> {
    try {
      await this.requireClient().setSubagentSubscription('progress')
    } catch {
      // Recoverable: subagent frames simply do not flow.
    }
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
    // Fire-and-forget status/notification methods resolve immediately.
    if (
      method === 'notify' ||
      method === 'setStatus' ||
      method === 'setTitle' ||
      method === 'set_editor_text' ||
      method === 'open_url'
    ) {
      client.send({ type: 'extension_ui_response', id, confirmed: true })
      return
    }
    if (method === 'cancel') {
      // The sidecar is retracting a dialog it no longer needs.
      this.clearPendingDialog(String(frame.targetId ?? ''))
      return
    }
    // The yolo tier opts out of prompts entirely: confirmations auto-approve.
    if (method === 'confirm' && this.permissionMode === 'skip-permissions') {
      client.send({ type: 'extension_ui_response', id, confirmed: true })
      return
    }
    if (method === 'select' || method === 'confirm' || method === 'input' || method === 'editor') {
      // Hold the dialog open for the GUI. A backstop timer unblocks the sidecar
      // even if the GUI never answers (or the session changes under it).
      this.pendingDialogs.set(id, {
        timer: setTimeout(() => {
          this.pendingDialogs.delete(id)
          client.send({ type: 'extension_ui_response', id, cancelled: true, timedOut: true })
          // Surface the timeout so the GUI does not keep a stale dialog open.
          client.emitLocalFrame({
            type: 'extension_ui_request',
            id: `${id}-timeout`,
            method: 'cancel',
            targetId: id
          })
        }, dialogTimeoutMs())
      })
      return
    }
    // Unknown method: fail closed rather than hang the sidecar.
    client.send({ type: 'extension_ui_response', id, cancelled: true })
  }

  private clearPendingDialog(id: string): void {
    const entry = this.pendingDialogs.get(id)
    if (!entry) return
    clearTimeout(entry.timer)
    this.pendingDialogs.delete(id)
  }

  private clearPendingDialogs(): void {
    for (const [, entry] of this.pendingDialogs) clearTimeout(entry.timer)
    this.pendingDialogs.clear()
  }

  /** Resolve a dialog the GUI answered; returns whether it was still pending. */
  private resolveExtensionUiResponse(response: OmpExtensionUiResponse): { resolved: boolean } {
    const pending = this.pendingDialogs.has(response.id)
    this.clearPendingDialog(response.id)
    if (!pending) return { resolved: false }
    this.requireClient().send(response)
    return { resolved: true }
  }

  /** Route extension-UI requests centrally so they work inside and outside a run. */
  private attachExtensionUiHandling(): void {
    this.client?.onFrame((frame) => {
      if (frame.type === 'extension_ui_request') {
        this.handleExtensionUi(frame).catch(() => undefined)
      }
    })
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

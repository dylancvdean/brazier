/**
 * HTTP client for the daemon's agent endpoints.
 *
 * This is the agent worker's only route to the machine: no filesystem, shell,
 * or host API is reachable from the worker except through these calls, and the
 * daemon applies policy and sandboxing on the far side.
 */

import type {
  AgentApproval,
  AgentEnvironment,
  AgentMessage,
  AgentPermissionMode,
  AgentPermissionSettings,
  AgentToolDefinition,
  ApprovalScope,
  SandboxDescription,
  ToolExecutionRecord,
  ToolRiskLevel
} from './types'

export type BrokerConnection = {
  address: string
  apiKey: string | null
}

export type ToolExecStatus = 'completed' | 'failed' | 'denied' | 'approval_required'

export type ToolImage = {
  mime_type: string
  /** Raw base64, without a data-URL prefix. */
  data: string
}

export type ToolExecResponse = {
  status: ToolExecStatus
  tool: string
  tool_call_id?: string
  environment: AgentEnvironment
  risk: ToolRiskLevel
  sandbox: SandboxDescription
  execution_id?: string
  output: string
  truncated?: boolean
  artifact_id?: string
  exit_code?: number | null
  changed_paths?: string[]
  images?: ToolImage[]
  duration_ms: number
  approval?: AgentApproval
  denied_reason?: string
  is_error: boolean
}

export type AgentSandboxCapabilities = {
  backend: string
  isolated: boolean
  /** Whether this daemon can execute programs in an OS-enforced sandbox. */
  sandboxed_execution: boolean
  filesystem_scoping: boolean
  network_isolation: boolean
  process_isolation: boolean
  profiles: string[]
  detail: string
  program?: string | null
}

export type AgentCapabilitiesResponse = {
  schema_version: number
  sandbox: AgentSandboxCapabilities
  permission_modes: string[]
  runtimes: Array<{
    id: string
    name: string
    /** Version of the adapter contract, not of the runtime package. */
    adapter_api_version: number
    available?: boolean
    trust?: string
    binary_path?: string | null
    unavailable_reason?: string | null
    capabilities: Record<string, boolean>
  }>
  default_runtime_id?: string
  tool_output_limit_chars: number
}

export type AgentPreferenceResponse = {
  default_runtime_id: string
  /** Power tools enabled for Powerful mode, by name. */
  power_tools?: string[]
}

/** A chat model the daemon serves, as advertised by the OpenAI-compatible list. */
export type BrokerModelInfo = {
  id: string
  capabilities?: {
    input_modalities?: string[]
    tools?: boolean
    reasoning?: boolean
    max_context_length?: number | null
  } | null
}

export type DaemonSessionRecord = {
  id: string
  title: string
  workspace_path?: string | null
  model: string
  runtime_id: string
  permission_mode: AgentPermissionMode
  permission_settings: AgentPermissionSettings
  enabled_tools?: string[] | null
  last_run_status: string
  compaction?: Record<string, unknown> | null
  runtime_metadata?: Record<string, unknown> | null
  created_at: string
  updated_at: string
}

export type DaemonMessageRecord = {
  id: string
  session_id: string
  seq: number
  role: string
  payload: AgentMessage
  created_at: string
}

export type PrefillProgress = {
  total: number
  cached: number
  processed: number
  elapsed_ms: number
  context_total?: number | null
}

export type ChatContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } }

export type ChatCompletionMessage =
  | { role: 'system'; content: string }
  | { role: 'user'; content: string | ChatContentPart[] }
  | {
      role: 'assistant'
      content: string
      reasoning_content?: string
      tool_calls?: Array<{
        id: string
        type: 'function'
        function: { name: string; arguments: string }
      }>
    }
  | { role: 'tool'; tool_call_id: string; content: string }

export type ChatCompletionToolSpec = {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: Record<string, unknown>
  }
}

export type StreamChatCompletionRequest = {
  model: string
  messages: ChatCompletionMessage[]
  tools?: ChatCompletionToolSpec[]
  toolChoice?: 'auto' | 'none'
  maxTokens?: number
  enableReasoning?: boolean
  slot?: number
}

export type StreamedToolCall = {
  id: string
  name: string
  arguments: string
}

export type StreamChatCompletionCallbacks = {
  onLoad?: (load: { phase: string; message: string }) => void
  onContentDelta?: (delta: string) => void
  onReasoningDelta?: (delta: string) => void
  onPrefill?: (prefill: PrefillProgress) => void
}

export type StreamChatCompletionResult = {
  content: string
  reasoning: string
  toolCalls: StreamedToolCall[]
  finishReason: string | null
  promptTokens?: number
  completionTokens?: number
  contextTotal?: number | null
}

export class BrokerError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'BrokerError'
    this.status = status
  }
}

export class ContextOverflowError extends BrokerError {
  constructor(message: string, status = 400) {
    super(message, status)
    this.name = 'ContextOverflowError'
  }
}

/** The model produced no stream data (keep-alive comments do not count) for too long. */
export class ModelStallError extends BrokerError {
  constructor(timeoutMs: number) {
    super(
      `The model stopped responding: no output for ${Math.round(timeoutMs / 1000)} seconds.`,
      504
    )
    this.name = 'ModelStallError'
  }
}

/** The connection closed before the daemon's terminal `[DONE]` frame. */
export class StreamInterruptedError extends BrokerError {
  constructor() {
    super('The model stream ended before the response finished.', 502)
    this.name = 'StreamInterruptedError'
  }
}

/**
 * True for failures worth retrying once the same request is replayed: the
 * connection dropped, the gateway hiccuped, or the stream was cut short.
 */
export function isTransientModelError(cause: unknown): boolean {
  if (cause instanceof ContextOverflowError || cause instanceof ModelStallError) return false
  if (cause instanceof StreamInterruptedError) return true
  if (cause instanceof BrokerError) return cause.status === 502 || cause.status === 503
  // undici reports refused / reset connections as `TypeError: fetch failed`.
  return cause instanceof TypeError && /fetch failed|network|socket/i.test(cause.message)
}

function anySignal(signals: Array<AbortSignal | undefined>): AbortSignal | undefined {
  const present = signals.filter((signal): signal is AbortSignal => Boolean(signal))
  if (present.length <= 1) return present[0]
  return AbortSignal.any(present)
}

/**
 * Rejects `stalled` when `reset` has not been called for `timeoutMs`. Racing
 * every network wait against it bounds a wedged engine without bounding a
 * long generation that keeps producing output.
 */
function stallWatchdog(
  timeoutMs: number,
  onStall: () => void
): { stalled: Promise<never>; reset: () => void; stop: () => void } {
  let timer: ReturnType<typeof setTimeout> | undefined
  let reject!: (error: Error) => void
  const stalled = new Promise<never>((_, rejectStall) => {
    reject = rejectStall
  })
  stalled.catch(() => undefined)
  const reset = (): void => {
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => {
      onStall()
      reject(new ModelStallError(timeoutMs))
    }, timeoutMs)
  }
  reset()
  return {
    stalled,
    reset,
    stop: () => {
      if (timer) clearTimeout(timer)
    }
  }
}

/** Detect prompt / KV-cache context overflow errors across local and remote engines. */
export function isContextOverflowMessage(message: string, code?: string | null): boolean {
  const normalizedCode = (code ?? '').toLowerCase()
  if (
    normalizedCode === 'context_length_exceeded' ||
    normalizedCode === 'exceed_context_size_error'
  ) {
    return true
  }
  const lower = message.toLowerCase()
  if (lower.includes('turn off parallel subagents') || lower.includes('failed to load model')) {
    return false
  }
  return (
    lower.includes('maximum context length') ||
    lower.includes('exceeds the available context size') ||
    lower.includes('context_length_exceeded') ||
    lower.includes('context window') ||
    lower.includes('context length') ||
    lower.includes('context size') ||
    lower.includes('n_ctx') ||
    lower.includes('prompt is too long') ||
    lower.includes('input is too long') ||
    lower.includes('too many tokens') ||
    lower.includes('kv cache') ||
    /exceed(?:s|ed)?\b.*\bcontext\b/.test(lower)
  )
}

function normalizeCompleteToolCall(raw: Record<string, unknown>, index: number): StreamedToolCall {
  const fn = raw.function as Record<string, unknown> | undefined
  const id = typeof raw.id === 'string' && raw.id.length > 0 ? raw.id : `call_${index}`
  const name =
    typeof fn?.name === 'string'
      ? fn.name
      : typeof raw.name === 'string'
        ? raw.name
        : ''
  const rawArgs = fn?.arguments ?? raw.arguments
  const args =
    typeof rawArgs === 'string'
      ? rawArgs
      : rawArgs && typeof rawArgs === 'object'
        ? JSON.stringify(rawArgs)
        : '{}'
  return { id, name, arguments: args }
}

const REQUEST_TIMEOUT_MS = 30_000
/**
 * Non-streaming tool calls include project checks and MCP calls that the
 * daemon itself bounds (shell commands at ten minutes). The client limit only
 * has to catch a daemon that never answers, so it sits above those.
 */
const EXEC_TIMEOUT_MS = 12 * 60_000
/** Extra time past the daemon's long-poll window before the client gives up. */
const APPROVAL_WAIT_GRACE_MS = 15_000
/**
 * Longest silence tolerated from the model stream. Prefill progress and token
 * deltas both reset it, so it only fires on a wedged engine or daemon.
 */
export const MODEL_STALL_TIMEOUT_MS = 5 * 60_000

export type BrokerClientOptions = {
  modelStallTimeoutMs?: number
}

/** Read one chunk, rejecting if nothing arrives within `idleMs`. */
export async function readWithIdleTimeout(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  idleMs: number,
  message: string,
  signal?: AbortSignal
): Promise<ReadableStreamReadResult<Uint8Array>> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      void reader.cancel().catch(() => undefined)
      reject(new BrokerError(message, 504))
    }, idleMs)
  })
  try {
    if (signal?.aborted) throw new BrokerError('Request aborted.', 499)
    return await Promise.race([reader.read(), timeout])
  } finally {
    if (timer) clearTimeout(timer)
  }
}

export class BrokerClient {
  private readonly connection: BrokerConnection
  private readonly modelStallTimeoutMs: number

  constructor(connection: BrokerConnection, options: BrokerClientOptions = {}) {
    this.connection = connection
    this.modelStallTimeoutMs = options.modelStallTimeoutMs ?? MODEL_STALL_TIMEOUT_MS
  }

  private async request<T>(
    path: string,
    init?: RequestInit & { signal?: AbortSignal; timeoutMs?: number }
  ): Promise<T> {
    const headers = new Headers(init?.headers)
    headers.set('content-type', 'application/json')
    if (this.connection.apiKey) {
      headers.set('authorization', `Bearer ${this.connection.apiKey}`)
    }
    const timeoutMs = init?.timeoutMs ?? REQUEST_TIMEOUT_MS
    const signal = anySignal([init?.signal ?? undefined, AbortSignal.timeout(timeoutMs)])
    const { timeoutMs: _timeout, ...fetchInit } = init ?? {}
    const response = await fetch(`${this.connection.address}${path}`, {
      ...fetchInit,
      headers,
      signal
    })
    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as {
        error?: { message?: string }
      } | null
      throw new BrokerError(
        payload?.error?.message ?? `Request failed with status ${response.status}.`,
        response.status
      )
    }
    if (response.status === 204 || response.status === 205) return undefined as T
    return (await response.json()) as T
  }

  async capabilities(): Promise<AgentCapabilitiesResponse> {
    return this.request('/api/v1/agent/capabilities')
  }

  /** Chat models the daemon serves. */
  async models(): Promise<BrokerModelInfo[]> {
    const payload = await this.request<{ data: BrokerModelInfo[] }>('/v1/models')
    return payload.data
  }

  async agentPreference(): Promise<AgentPreferenceResponse> {
    return this.request('/api/v1/preferences/agent')
  }

  /** Tool catalog, translated into the application's own tool shape. */
  async tools(): Promise<AgentToolDefinition[]> {
    const payload = await this.request<{
      data: Array<{
        name: string
        label: string
        description: string
        input_schema: Record<string, unknown>
        risk: ToolRiskLevel
        executes: boolean
        needs_workspace: boolean
        default_environment: AgentEnvironment
        power_tool?: boolean
      }>
    }>('/api/v1/agent/tools')
    return payload.data.map((entry) => ({
      name: entry.name,
      label: entry.label,
      description: entry.description,
      inputSchema: entry.input_schema,
      risk: entry.risk,
      executes: entry.executes,
      needsWorkspace: entry.needs_workspace,
      defaultEnvironment: entry.default_environment,
      powerTool: entry.power_tool ?? false
    }))
  }

  async session(id: string): Promise<{
    session: DaemonSessionRecord
    messages: DaemonMessageRecord[]
    tool_executions: ToolExecutionRecord[]
    pending_approvals: AgentApproval[]
    grants: string[]
    sandbox: AgentSandboxCapabilities
  }> {
    return this.request(`/api/v1/agent/sessions/${id}`)
  }

  async createSession(input: {
    title?: string
    workspace_path?: string | null
    model: string
    runtime_id?: string
    permission_mode?: DaemonSessionRecord['permission_mode']
    permission_settings?: DaemonSessionRecord['permission_settings']
    enabled_tools?: string[]
    confine_to_worktree?: boolean
  }): Promise<DaemonSessionRecord> {
    const elevated =
      input.permission_mode === 'skip-permissions' ||
      Boolean(input.permission_settings?.auto_approve_host_actions)
    return this.request('/api/v1/agent/sessions', {
      method: 'POST',
      body: JSON.stringify({
        ...input,
        ...(elevated ? { confirm_elevated_permissions: true } : {})
      })
    })
  }

  /** Delete a session record (transcript, tool executions, sandbox) from the daemon. */
  async deleteSession(id: string): Promise<void> {
    return this.request(`/api/v1/agent/sessions/${id}`, { method: 'DELETE' })
  }

  /** Text-profile overrides for a model, used when resolving subagent defaults. */
  async textProfile(modelId: string): Promise<{
    subagent_model?: string | null
    context_size?: number | null
    max_tokens?: number | null
    max_subagents?: number | null
    parallel_subagents?: boolean | null
  } | null> {
    const payload = await this.request<{
      models: Record<
        string,
        {
          kind?: string
          subagent_model?: string | null
          context_size?: number | null
          max_tokens?: number | null
          max_subagents?: number | null
          parallel_subagents?: boolean | null
        }
      >
    }>('/api/v1/models/settings')
    const profile = payload.models[modelId]
    if (!profile || profile.kind !== 'text') return null
    return {
      subagent_model: profile.subagent_model,
      context_size: profile.context_size,
      max_tokens: profile.max_tokens,
      max_subagents: profile.max_subagents,
      parallel_subagents: profile.parallel_subagents
    }
  }

  async runtimeInferenceSettings(): Promise<{
    context_size: number
    max_tokens?: number | null
    enable_reasoning?: boolean
    drop_reasoning_between_turns?: boolean
  }> {
    return this.request('/api/v1/runtime/settings')
  }

  async systemPrompt(id: string): Promise<{ system_prompt: string; tools: string[] }> {
    return this.request(`/api/v1/agent/sessions/${id}/prompt`)
  }

  async appendMessages(
    sessionId: string,
    messages: AgentMessage[],
    replace = false
  ): Promise<void> {
    if (messages.length === 0 && !replace) return
    await this.request(`/api/v1/agent/sessions/${sessionId}/messages`, {
      method: 'POST',
      body: JSON.stringify({
        replace,
        messages: messages.map((message) => ({ role: message.role, payload: message }))
      })
    })
  }

  async updateSession(
    sessionId: string,
    update: Record<string, unknown>
  ): Promise<DaemonSessionRecord> {
    const settings = update.permission_settings as
      | { auto_approve_host_actions?: boolean }
      | undefined
    const elevated =
      update.permission_mode === 'skip-permissions' ||
      Boolean(settings?.auto_approve_host_actions)
    return this.request(`/api/v1/agent/sessions/${sessionId}`, {
      method: 'PATCH',
      body: JSON.stringify({
        ...update,
        ...(elevated ? { confirm_elevated_permissions: true } : {})
      })
    })
  }

  async execTool(
    request: {
      sessionId: string
      runId?: string
      toolCallId?: string
      tool: string
      arguments: Record<string, unknown>
      environment?: AgentEnvironment
      reason?: string
      approvalId?: string
    },
    signal?: AbortSignal
  ): Promise<ToolExecResponse> {
    return this.request('/api/v1/agent/exec', {
      method: 'POST',
      signal,
      timeoutMs: EXEC_TIMEOUT_MS,
      body: JSON.stringify({
        session_id: request.sessionId,
        run_id: request.runId,
        tool_call_id: request.toolCallId,
        tool: request.tool,
        arguments: request.arguments,
        environment: request.environment,
        reason: request.reason,
        approval_id: request.approvalId
      })
    })
  }

  /** Execute a foreground tool and receive output before the process exits. */
  async execToolStreaming(
    request: {
      sessionId: string
      runId?: string
      toolCallId?: string
      tool: string
      arguments: Record<string, unknown>
      environment?: AgentEnvironment
      reason?: string
      approvalId?: string
    },
    onOutput: (chunk: string) => void,
    signal?: AbortSignal
  ): Promise<ToolExecResponse> {
    const headers = new Headers({ 'content-type': 'application/json' })
    if (this.connection.apiKey) {
      headers.set('authorization', `Bearer ${this.connection.apiKey}`)
    }
    const response = await fetch(`${this.connection.address}/api/v1/agent/exec/stream`, {
      method: 'POST',
      headers,
      signal,
      body: JSON.stringify({
        session_id: request.sessionId,
        run_id: request.runId,
        tool_call_id: request.toolCallId,
        tool: request.tool,
        arguments: request.arguments,
        environment: request.environment,
        reason: request.reason,
        approval_id: request.approvalId
      })
    })
    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as {
        error?: { message?: string }
      } | null
      throw new BrokerError(
        payload?.error?.message ?? `Request failed with status ${response.status}.`,
        response.status
      )
    }
    if (!response.body) throw new BrokerError('The tool output stream had no body.', 502)

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let result: ToolExecResponse | undefined

    const consume = (block: string): void => {
      let event = 'message'
      const data: string[] = []
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
      }
      if (data.length === 0) return
      const payload = JSON.parse(data.join('\n')) as {
        chunk?: string
        message?: string
      }
      if (event === 'output' && typeof payload.chunk === 'string') {
        onOutput(payload.chunk)
      } else if (event === 'result') {
        result = payload as ToolExecResponse
      } else if (event === 'error') {
        throw new BrokerError(payload.message ?? 'Tool execution failed.', 500)
      }
    }

    try {
      while (true) {
        const { value, done } = await readWithIdleTimeout(
          reader,
          EXEC_TIMEOUT_MS,
          'Tool execution stream stalled with no output.',
          signal
        )
        buffer += decoder.decode(value, { stream: !done })
        let boundary = buffer.search(/\r?\n\r?\n/)
        while (boundary >= 0) {
          const separator = buffer.slice(boundary).match(/^\r?\n\r?\n/)?.[0] ?? '\n\n'
          consume(buffer.slice(0, boundary))
          buffer = buffer.slice(boundary + separator.length)
          boundary = buffer.search(/\r?\n\r?\n/)
        }
        if (done) break
      }
      if (buffer.trim()) consume(buffer)
    } finally {
      reader.releaseLock()
    }
    if (!result) throw new BrokerError('The tool output stream ended before its result.', 502)
    return result
  }

  /**
   * Stream a single-pass OpenAI-compatible chat completion directly from the
   * daemon without monkey-patching global `fetch` or cloning the response stream.
   */
  async streamChatCompletion(
    request: StreamChatCompletionRequest,
    callbacks?: StreamChatCompletionCallbacks,
    signal?: AbortSignal
  ): Promise<StreamChatCompletionResult> {
    const slot = request.slot ?? 0
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'x-brazier-mode': 'agent',
      'x-brazier-slot': String(slot)
    }
    if (this.connection.apiKey) {
      headers.Authorization = `Bearer ${this.connection.apiKey}`
    }

    const body: Record<string, unknown> = {
      model: request.model,
      messages: request.messages,
      stream: true
    }
    if (request.tools && request.tools.length > 0) {
      body.tools = request.tools
      body.tool_choice = request.toolChoice ?? 'auto'
    }
    if (request.maxTokens !== undefined) {
      body.max_tokens = request.maxTokens
    }
    if (request.enableReasoning !== undefined) {
      body.enable_reasoning = request.enableReasoning
    }

    const stallAbort = new AbortController()
    const watchdog = stallWatchdog(this.modelStallTimeoutMs, () => stallAbort.abort())
    try {
      return await this.readChatCompletion(
        body,
        headers,
        anySignal([signal, stallAbort.signal])!,
        watchdog,
        callbacks
      )
    } finally {
      watchdog.stop()
    }
  }

  private async readChatCompletion(
    body: Record<string, unknown>,
    headers: Record<string, string>,
    signal: AbortSignal,
    watchdog: ReturnType<typeof stallWatchdog>,
    callbacks?: StreamChatCompletionCallbacks
  ): Promise<StreamChatCompletionResult> {
    const response = await Promise.race([
      fetch(`${this.openAiBaseUrl()}/chat/completions`, {
        method: 'POST',
        headers,
        signal,
        body: JSON.stringify(body)
      }),
      watchdog.stalled
    ])

    if (!response.ok) {
      const rawText = await Promise.race([response.text(), watchdog.stalled]).catch(() => '')
      let message = `Request failed with status ${response.status}.`
      let code: string | undefined
      try {
        const parsed = JSON.parse(rawText) as {
          error?: string | { message?: string; code?: string }
          message?: string
        }
        if (typeof parsed.error === 'string') {
          message = parsed.error
        } else if (parsed.error && typeof parsed.error === 'object') {
          if (typeof parsed.error.message === 'string') message = parsed.error.message
          if (typeof parsed.error.code === 'string') code = parsed.error.code
        } else if (typeof parsed.message === 'string') {
          message = parsed.message
        }
      } catch {
        if (rawText.trim()) message = rawText.trim()
      }
      if (
        (response.status === 400 ||
          response.status === 413 ||
          response.status === 422 ||
          response.status === 500) &&
        isContextOverflowMessage(message, code)
      ) {
        throw new ContextOverflowError(message, response.status)
      }
      throw new BrokerError(message, response.status)
    }

    if (!response.body) {
      throw new BrokerError('The chat completion stream had no body.', 502)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let content = ''
    let pendingNull = ''
    let reasoning = ''
    let finishReason: string | null = null
    let promptTokens: number | undefined
    let completionTokens: number | undefined
    let contextTotal: number | null | undefined
    const toolCallsByIndex = new Map<number, { id: string; name: string; arguments: string }>()
    let completeToolCalls: StreamedToolCall[] | undefined
    let doneFlag = false

    const consumeBlock = (block: string): void => {
      if (doneFlag) return
      let eventType = 'message'
      const dataLines: string[] = []
      for (const line of block.split(/\r?\n/)) {
        if (!line || line.startsWith(':')) continue
        if (line.startsWith('event:')) {
          eventType = line.slice(6).trim()
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).trimStart())
        }
      }
      if (dataLines.length === 0) return
      watchdog.reset()
      const dataStr = dataLines.join('\n')
      if (dataStr === '[DONE]') {
        doneFlag = true
        return
      }

      let payload: Record<string, unknown>
      try {
        payload = JSON.parse(dataStr) as Record<string, unknown>
      } catch {
        if (eventType === 'error') {
          if (isContextOverflowMessage(dataStr)) {
            throw new ContextOverflowError(dataStr, 400)
          }
          throw new BrokerError(dataStr, 500)
        }
        return
      }

      const brazier = payload.brazier as
        | {
            load?: { phase?: string; message?: string }
            prefill?: PrefillProgress
            generation?: { prompt_tokens?: number | null; completion_tokens?: number }
            client_tool_calls?: Array<Record<string, unknown>>
            fork_hints?: unknown
          }
        | undefined

      if (payload.error || eventType === 'error') {
        const rawErr = payload.error
        const errMsg =
          typeof rawErr === 'string'
            ? rawErr
            : rawErr && typeof rawErr === 'object' && typeof (rawErr as { message?: unknown }).message === 'string'
              ? ((rawErr as { message: string }).message)
              : typeof payload.message === 'string'
                ? payload.message
                : 'Model stream failed.'
        const errCode =
          rawErr && typeof rawErr === 'object' && typeof (rawErr as { code?: unknown }).code === 'string'
            ? ((rawErr as { code: string }).code)
            : undefined
        if (!brazier?.fork_hints && isContextOverflowMessage(errMsg, errCode)) {
          throw new ContextOverflowError(errMsg, 400)
        }
        throw new BrokerError(errMsg, 500)
      }

      if (brazier?.load) {
        callbacks?.onLoad?.({
          phase: typeof brazier.load.phase === 'string' ? brazier.load.phase : 'load',
          message:
            typeof brazier.load.message === 'string'
              ? brazier.load.message
              : 'Loading model…'
        })
      }

      if (brazier?.prefill) {
        const prefill = brazier.prefill
        if (typeof prefill.total === 'number') promptTokens = prefill.total
        if (prefill.context_total !== undefined) contextTotal = prefill.context_total
        callbacks?.onPrefill?.(prefill)
      }

      if (brazier?.generation) {
        if (typeof brazier.generation.prompt_tokens === 'number') {
          promptTokens = brazier.generation.prompt_tokens
        }
        if (typeof brazier.generation.completion_tokens === 'number') {
          completionTokens = brazier.generation.completion_tokens
        }
      }

      const usage = payload.usage as
        | { prompt_tokens?: number; completion_tokens?: number }
        | undefined
      if (usage) {
        if (typeof usage.prompt_tokens === 'number' && usage.prompt_tokens > 0) {
          promptTokens = usage.prompt_tokens
        }
        if (typeof usage.completion_tokens === 'number' && usage.completion_tokens > 0) {
          completionTokens = usage.completion_tokens
        }
      }

      if (Array.isArray(brazier?.client_tool_calls)) {
        pendingNull = ''
        completeToolCalls = brazier.client_tool_calls.map((entry, idx) =>
          normalizeCompleteToolCall(entry, idx)
        )
      }

      const choices = payload.choices as Array<Record<string, unknown>> | undefined
      const choice = choices?.[0]
      if (!choice) return

      if (typeof choice.finish_reason === 'string') {
        finishReason = choice.finish_reason
      }

      const delta = choice.delta as Record<string, unknown> | undefined
      if (!delta) return

      const reasoningDelta =
        typeof delta.reasoning_content === 'string'
          ? delta.reasoning_content
          : typeof delta.reasoning === 'string'
            ? delta.reasoning
            : ''
      if (reasoningDelta.length > 0) {
        reasoning += reasoningDelta
        callbacks?.onReasoningDelta?.(reasoningDelta)
      }

      const hasToolDelta = Array.isArray(delta.tool_calls) && delta.tool_calls.length > 0

      if (typeof delta.content === 'string' && delta.content.length > 0) {
        if (content.length === 0 && pendingNull.length === 0 && delta.content.trim() === 'null') {
          // Hold a lone initial "null" artifact; drop if tool-only or standalone,
          // flush only if real prose follows.
          if (!hasToolDelta && finishReason !== 'tool_calls') {
            pendingNull = delta.content
          }
        } else {
          const chunk = pendingNull ? pendingNull + delta.content : delta.content
          pendingNull = ''
          content += chunk
          callbacks?.onContentDelta?.(chunk)
        }
      }

      if (hasToolDelta) {
        pendingNull = ''
        const rawCalls = delta.tool_calls as Array<Record<string, unknown>>
        const allUnindexed = rawCalls.every((tc) => typeof tc?.index !== 'number')
        if (allUnindexed) {
          completeToolCalls = rawCalls.map((tc, idx) => normalizeCompleteToolCall(tc, idx))
        } else {
          for (let i = 0; i < rawCalls.length; i += 1) {
            const tc = rawCalls[i]
            if (!tc || typeof tc !== 'object') continue
            const index = typeof tc.index === 'number' ? tc.index : i
            const acc = toolCallsByIndex.get(index) ?? { id: '', name: '', arguments: '' }
            if (typeof tc.id === 'string' && tc.id.length > 0) {
              acc.id = tc.id
            }
            const fn = tc.function as Record<string, unknown> | undefined
            const rawName =
              typeof fn?.name === 'string'
                ? fn.name
                : typeof tc.name === 'string'
                  ? tc.name
                  : ''
            if (rawName) {
              if (!acc.name) acc.name = rawName
              else if (acc.name !== rawName) acc.name += rawName
            }
            const rawArgs = fn?.arguments ?? tc.arguments
            if (typeof rawArgs === 'string') {
              acc.arguments += rawArgs
            } else if (rawArgs && typeof rawArgs === 'object') {
              acc.arguments = JSON.stringify(rawArgs)
            }
            toolCallsByIndex.set(index, acc)
          }
        }
      }
    }

    let streamEnded = false
    try {
      while (!doneFlag) {
        const { value, done } = await Promise.race([reader.read(), watchdog.stalled])
        streamEnded = done
        buffer += decoder.decode(value, { stream: !done })
        let boundary = buffer.search(/\r?\n\r?\n/)
        while (boundary >= 0) {
          const separator = buffer.slice(boundary).match(/^\r?\n\r?\n/)?.[0] ?? '\n\n'
          consumeBlock(buffer.slice(0, boundary))
          buffer = buffer.slice(boundary + separator.length)
          if (doneFlag) break
          boundary = buffer.search(/\r?\n\r?\n/)
        }
        if (done) break
      }
      if (!doneFlag && buffer.trim()) {
        consumeBlock(buffer)
      }
    } finally {
      if (!streamEnded) await reader.cancel().catch(() => undefined)
      try {
        reader.releaseLock()
      } catch {
        // A read abandoned by the watchdog may still hold the lock.
      }
    }
    // The daemon always closes its stream with `[DONE]`, including after an
    // error frame. Anything else is a dropped connection, and the partial
    // turn must not be mistaken for the model's final answer.
    if (!doneFlag) throw new StreamInterruptedError()

    const toolCalls: StreamedToolCall[] =
      completeToolCalls && completeToolCalls.length > 0
        ? completeToolCalls.filter((tc) => tc.name.length > 0)
        : [...toolCallsByIndex.entries()]
            .sort(([a], [b]) => a - b)
            .map(([index, entry]) => ({
              id: entry.id || `call_${index}`,
              name: entry.name,
              arguments: entry.arguments || '{}'
            }))
            .filter((tc) => tc.name.length > 0)

    if (!doneFlag && finishReason === null && toolCalls.length === 0) {
      throw new BrokerError(
        'The model stream ended unexpectedly before finishing (the model may have crashed or been unloaded).',
        502
      )
    }

    return {
      content,
      reasoning,
      toolCalls,
      finishReason,
      promptTokens,
      completionTokens,
      contextTotal
    }
  }

  /**
   * Block until the user answers, or until `waitMs` elapses. The daemon holds
   * the request open, so no polling loop is needed here.
   */
  async waitForApproval(
    approvalId: string,
    waitMs: number,
    signal?: AbortSignal
  ): Promise<AgentApproval> {
    return this.request(`/api/v1/agent/approvals/${approvalId}?wait_ms=${waitMs}`, {
      signal,
      timeoutMs: waitMs + APPROVAL_WAIT_GRACE_MS
    })
  }

  async decideApproval(
    approvalId: string,
    decision: 'approve' | 'deny',
    scope?: ApprovalScope,
    note?: string
  ): Promise<AgentApproval> {
    return this.request(`/api/v1/agent/approvals/${approvalId}`, {
      method: 'POST',
      body: JSON.stringify({ decision, scope, note })
    })
  }

  async cancel(sessionId: string): Promise<void> {
    await this.request(`/api/v1/agent/sessions/${sessionId}/cancel`, {
      method: 'POST',
      body: JSON.stringify({})
    })
  }

  /** OpenAI-compatible base URL the runtime points its model client at. */
  openAiBaseUrl(): string {
    return `${this.connection.address}/v1`
  }

  apiKey(): string {
    // The daemon requires a bearer token; providers that insist on a non-empty
    // key get a placeholder when auth is disabled.
    return this.connection.apiKey ?? 'brazier-local'
  }
}

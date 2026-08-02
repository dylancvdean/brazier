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
  omp_profile?: { binary_path?: string; config_yaml?: string } | null
}

export type DaemonSessionRecord = {
  id: string
  title: string
  workspace_path?: string | null
  model: string
  runtime_id: string
  permission_mode: 'ask' | 'sandbox-only' | 'skip-permissions'
  permission_settings: {
    auto_approve_sandboxed_actions: boolean
    auto_approve_host_actions: boolean
  }
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

export class BrokerError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'BrokerError'
    this.status = status
  }
}

const REQUEST_TIMEOUT_MS = 30_000

export class BrokerClient {
  private readonly connection: BrokerConnection

  constructor(connection: BrokerConnection) {
    this.connection = connection
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
    const signal =
      init?.signal ??
      AbortSignal.timeout(timeoutMs)
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
      defaultEnvironment: entry.default_environment
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
    return this.request('/api/v1/agent/sessions', {
      method: 'POST',
      body: JSON.stringify(input)
    })
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
    return this.request(`/api/v1/agent/sessions/${sessionId}`, {
      method: 'PATCH',
      body: JSON.stringify(update)
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

    while (true) {
      const { value, done } = await reader.read()
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
    if (!result) throw new BrokerError('The tool output stream ended before its result.', 502)
    return result
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
    return this.request(`/api/v1/agent/approvals/${approvalId}?wait_ms=${waitMs}`, { signal })
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

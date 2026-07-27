/**
 * Agent mode REST client.
 *
 * Session state, approvals, and the tool catalog live in the daemon, so the UI
 * reads them over the same authenticated loopback API the rest of the app uses.
 * Runs go through the worker bridge instead (see `window.brazier.agent`).
 */

import type {
  AgentApproval,
  AgentMessage,
  AgentPermissionMode,
  AgentPermissionSettings,
  ApprovalScope,
  SandboxDescription,
  ToolExecutionRecord
} from '../../agent/core/types'

type Connection = Awaited<ReturnType<typeof window.brazier.getConnection>>
let connectionPromise: Promise<Connection> | undefined

async function connection(): Promise<Connection> {
  connectionPromise ??= window.brazier.getConnection()
  return connectionPromise
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const daemon = await connection()
  const headers = new Headers(init?.headers)
  headers.set('content-type', 'application/json')
  if (daemon.api_key) headers.set('authorization', `Bearer ${daemon.api_key}`)
  const response = await fetch(`${daemon.address}${path}`, { ...init, headers })
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as {
      error?: { message?: string }
    } | null
    throw new Error(payload?.error?.message ?? `Request failed with status ${response.status}.`)
  }
  return response.json() as Promise<T>
}

export type AgentSandboxCapabilities = {
  backend: string
  isolated: boolean
  filesystem_scoping: boolean
  network_isolation: boolean
  process_isolation: boolean
  profiles: string[]
  detail: string
  program?: string | null
}

export type AgentCapabilities = {
  schema_version: number
  sandbox: AgentSandboxCapabilities
  permission_modes: AgentPermissionMode[]
  runtimes: Array<{
    id: string
    name: string
    /** Version of the adapter contract, not of the runtime package. */
    adapter_api_version: number
    capabilities: Record<string, boolean>
  }>
  tool_output_limit_chars: number
}

export type AgentSessionSummary = {
  id: string
  title: string
  workspace_path?: string | null
  model: string
  runtime_id: string
  permission_mode: AgentPermissionMode
  permission_settings: AgentPermissionSettings
  enabled_tools?: string[] | null
  last_run_status: string
  created_at: string
  updated_at: string
  /** Includes worktree confinement metadata when the session is isolated. */
  runtime_metadata?: {
    worktree?: {
      source_path: string
      path: string
      branch: string
    }
  } | null
}

export type AgentWorktreeInfo = {
  source_path: string
  path: string
  branch: string
}

export type AgentSessionDetail = {
  session: AgentSessionSummary
  messages: Array<{
    id: string
    seq: number
    role: string
    payload: AgentMessage
    created_at?: string
  }>
  tool_executions: ToolExecutionRecord[]
  pending_approvals: AgentApproval[]
  grants: string[]
  sandbox: AgentSandboxCapabilities
}

export type AgentToolCatalogEntry = {
  name: string
  label: string
  description: string
  risk: string
  executes: boolean
  needs_workspace: boolean
}

export async function fetchAgentCapabilities(): Promise<AgentCapabilities> {
  return request('/api/v1/agent/capabilities')
}

export async function fetchAgentTools(): Promise<AgentToolCatalogEntry[]> {
  const payload = await request<{ data: AgentToolCatalogEntry[] }>('/api/v1/agent/tools')
  return payload.data
}

export async function listAgentSessions(): Promise<AgentSessionSummary[]> {
  const payload = await request<{ data: AgentSessionSummary[] }>('/api/v1/agent/sessions')
  return payload.data
}

export async function createAgentSession(input: {
  title?: string
  workspace_path?: string | null
  model: string
  permission_mode?: AgentPermissionMode
  permission_settings?: AgentPermissionSettings
  enabled_tools?: string[]
  /** Isolate the session in a fresh git worktree of the workspace. */
  confine_to_worktree?: boolean
}): Promise<AgentSessionSummary> {
  return request('/api/v1/agent/sessions', {
    method: 'POST',
    body: JSON.stringify(input)
  })
}

export async function fetchAgentSession(id: string): Promise<AgentSessionDetail> {
  return request(`/api/v1/agent/sessions/${id}`)
}

export async function updateAgentSession(
  id: string,
  update: {
    title?: string
    workspace_path?: string | null
    model?: string
    permission_mode?: AgentPermissionMode
    permission_settings?: AgentPermissionSettings
    enabled_tools?: string[]
    confine_to_worktree?: boolean
  }
): Promise<AgentSessionSummary> {
  return request(`/api/v1/agent/sessions/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(update)
  })
}

export async function deleteAgentSession(id: string): Promise<void> {
  await request(`/api/v1/agent/sessions/${id}`, { method: 'DELETE' })
}

export async function decideAgentApproval(
  approvalId: string,
  decision: 'approve' | 'deny',
  scope?: ApprovalScope,
  note?: string
): Promise<AgentApproval> {
  return request(`/api/v1/agent/approvals/${approvalId}`, {
    method: 'POST',
    body: JSON.stringify({ decision, scope, note })
  })
}

export async function validateAgentWorkspace(path: string): Promise<{
  path: string
  git_repository: boolean
  sandbox: AgentSandboxCapabilities
}> {
  return request('/api/v1/agent/workspace', {
    method: 'POST',
    body: JSON.stringify({ path })
  })
}

/** Full text of a truncated tool output. */
export async function fetchAgentArtifact(artifactId: string): Promise<string> {
  const daemon = await connection()
  const headers = new Headers()
  if (daemon.api_key) headers.set('authorization', `Bearer ${daemon.api_key}`)
  const response = await fetch(`${daemon.address}/api/v1/agent/artifacts/${artifactId}`, {
    headers
  })
  if (!response.ok) throw new Error(`Could not load artifact ${artifactId}.`)
  return response.text()
}

/**
 * Sandbox badge text derived from what the daemon actually applied. Only the
 * three fields the badge needs are required, so both the capability report and
 * a per-call description can be passed in.
 */
export function sandboxBadge(sandbox: Pick<SandboxDescription, 'backend' | 'isolated' | 'detail'>): {
  label: string
  tone: 'sandboxed' | 'unsandboxed'
  detail: string
} {
  if (sandbox.isolated) {
    return {
      label: `Sandboxed · ${sandbox.backend}`,
      tone: 'sandboxed',
      detail: sandbox.detail
    }
  }
  return {
    label: 'No sandbox',
    tone: 'unsandboxed',
    detail: sandbox.detail
  }
}

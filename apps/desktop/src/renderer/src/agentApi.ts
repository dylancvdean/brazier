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
  ExecutionLocation,
  SandboxDescription,
  ToolExecutionRecord
} from '../../agent/core/types'
import { daemonFetch } from './daemonAvailability'

type Connection = Awaited<ReturnType<typeof window.brazier.getConnection>>
let connectionPromise: Promise<Connection> | undefined

export function invalidateAgentConnectionCache(): void {
  connectionPromise = undefined
}

if (typeof window !== 'undefined' && window.brazier?.onConnectionProfileChanged) {
  window.brazier.onConnectionProfileChanged(invalidateAgentConnectionCache)
}

async function connection(): Promise<Connection> {
  connectionPromise ??= window.brazier.getConnection()
  const pending = connectionPromise
  try {
    return await pending
  } catch (error) {
    if (connectionPromise === pending) connectionPromise = undefined
    throw error
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const daemon = await connection()
  const headers = new Headers(init?.headers)
  headers.set('content-type', 'application/json')
  const response = await daemonFetch(`${daemon.address}${path}`, { ...init, headers })
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
  /** Whether this daemon can execute programs in an OS-enforced sandbox. */
  sandboxed_execution: boolean
  filesystem_scoping: boolean
  network_isolation: boolean
  process_isolation: boolean
  profiles: string[]
  detail: string
  program?: string | null
}

export type AgentRuntimeInfo = {
  id: string
  name: string
  /** Version of the adapter contract, not of the runtime package. */
  adapter_api_version: number
  available?: boolean
  /** `broker` for Pi. */
  trust?: 'broker' | 'privileged_harness' | string
  binary_path?: string | null
  unavailable_reason?: string | null
  capabilities: Record<string, boolean>
}

export type AgentCapabilities = {
  schema_version: number
  sandbox: AgentSandboxCapabilities
  permission_modes: AgentPermissionMode[]
  runtimes: AgentRuntimeInfo[]
  default_runtime_id?: string
  tool_output_limit_chars: number
}

export type AgentPreference = {
  default_runtime_id: string
  /** Power tools enabled for Powerful mode, by name. */
  power_tools?: string[]
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
      last_applied_tree?: string
    }
    kind?: 'subagent'
    parent_session_id?: string
  } | null
}

export type AgentWorktreeInfo = {
  source_path: string
  path: string
  branch: string
}

/** Live inspection used before delete/unconfine confirmation. */
export type AgentWorktreeStatus = {
  source_path: string
  path: string
  branch: string
  exists: boolean
  dirty: boolean
  ahead_of_source: boolean
  has_discardable_changes: boolean
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
  /** Optional "Powerful" mode surface. Simple mode never exposes these. */
  power_tool?: boolean
}

export async function fetchAgentCapabilities(): Promise<AgentCapabilities> {
  return request('/api/v1/agent/capabilities')
}

export async function fetchAgentPreference(): Promise<AgentPreference> {
  return request('/api/v1/preferences/agent')
}

export async function saveAgentPreference(preference: AgentPreference): Promise<AgentPreference> {
  return request('/api/v1/preferences/agent', {
    method: 'PUT',
    body: JSON.stringify(preference)
  })
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
  /** Agent framework adapter (`pi`). Defaults to the saved preference. */
  runtime_id?: string
  permission_mode?: AgentPermissionMode
  permission_settings?: AgentPermissionSettings
  enabled_tools?: string[]
  /** Isolate the session in a fresh git worktree of the workspace. */
  confine_to_worktree?: boolean
}): Promise<AgentSessionSummary> {
  const elevated =
    input.permission_mode === 'skip-permissions' ||
    Boolean(input.permission_settings?.auto_approve_host_actions)
  return request('/api/v1/agent/sessions', {
    method: 'POST',
    body: JSON.stringify({
      ...input,
      ...(elevated ? { confirm_elevated_permissions: true } : {})
    })
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
    /** Only valid when confine_to_worktree is false. */
    discard_unapplied?: boolean
  }
): Promise<AgentSessionSummary> {
  const elevated =
    update.permission_mode === 'skip-permissions' ||
    Boolean(update.permission_settings?.auto_approve_host_actions)
  return request(`/api/v1/agent/sessions/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({
      ...update,
      ...(elevated ? { confirm_elevated_permissions: true } : {})
    })
  })
}

export async function deleteAgentSession(
  id: string,
  options?: { discard_unapplied?: boolean }
): Promise<void> {
  const params = new URLSearchParams()
  if (options?.discard_unapplied) params.set('discard_unapplied', 'true')
  const query = params.toString()
  await request(`/api/v1/agent/sessions/${id}${query ? `?${query}` : ''}`, {
    method: 'DELETE'
  })
}

export async function fetchAgentWorktreeStatus(
  id: string
): Promise<AgentWorktreeStatus | null> {
  const payload = await request<{ worktree: AgentWorktreeStatus | null }>(
    `/api/v1/agent/sessions/${id}/worktree`
  )
  return payload.worktree
}

export async function applyAgentWorktree(id: string): Promise<{
  session: AgentSessionSummary
  changed_paths: string[]
  already_up_to_date: boolean
}> {
  return request(`/api/v1/agent/sessions/${id}/apply-worktree`, {
    method: 'POST',
    body: JSON.stringify({})
  })
}

export async function decideAgentApproval(
  approvalId: string,
  decision: 'approve' | 'deny',
  scope?: ApprovalScope,
  note?: string,
  expectedExecutionLocation?: ExecutionLocation
): Promise<AgentApproval> {
  return request(`/api/v1/agent/approvals/${approvalId}`, {
    method: 'POST',
    body: JSON.stringify({
      decision,
      scope,
      note,
      expected_execution_location: expectedExecutionLocation
    })
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

export type AgentWorkspacePrompt = {
  workspace_path: string
  /** Editable template. Known `{shortcut}` values are expanded at run time. */
  system_prompt: string
  resolved_prompt: string
  components: AgentPromptComponent[]
  customized: boolean
}

export type AgentPromptComponent = {
  name: string
  placeholder: string
  content: string
}

export async function fetchAgentWorkspacePrompt(
  workspacePath: string,
  sessionId?: string
): Promise<AgentWorkspacePrompt> {
  const query = new URLSearchParams({ workspace_path: workspacePath })
  if (sessionId) query.set('session_id', sessionId)
  return request(`/api/v1/agent/workspaces/prompt?${query.toString()}`)
}

export async function saveAgentWorkspacePrompt(
  workspacePath: string,
  systemPrompt: string | null
): Promise<AgentWorkspacePrompt> {
  return request('/api/v1/agent/workspaces/prompt', {
    method: 'PUT',
    body: JSON.stringify({ workspace_path: workspacePath, system_prompt: systemPrompt })
  })
}

/** Full text of a truncated tool output. */
export async function fetchAgentArtifact(artifactId: string): Promise<string> {
  const daemon = await connection()
  const headers = new Headers()
  const response = await daemonFetch(`${daemon.address}/api/v1/agent/artifacts/${artifactId}`, {
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

/**
 * Application-owned Agent mode types.
 *
 * Nothing here references an agent framework. A runtime adapter (see
 * `../pi/piRuntime.ts`) translates its framework's messages, events, and tools
 * into these shapes, so swapping the runtime touches only the adapter.
 */

/** Where a tool call runs. */
export type AgentEnvironment = 'sandbox' | 'host'

/** How much damage a tool call can do. */
export type ToolRiskLevel = 'safe' | 'read' | 'write' | 'execute' | 'destructive'

export type AgentPermissionMode = 'ask' | 'sandbox-only' | 'skip-permissions'

export type AgentPermissionSettings = {
  auto_approve_sandboxed_actions: boolean
  auto_approve_host_actions: boolean
}

export type RequestedPathAccess = {
  path: string
  write?: boolean
}

/** Privileges the agent is asking for and does not currently hold. */
export type AgentElevationRequest = {
  reason: string
  proposed_command?: string
  requested_filesystem_paths?: RequestedPathAccess[]
  requested_network_access?: boolean
  requested_host_execution?: boolean
}

export type ApprovalScope = 'once' | 'session'

export type ApprovalStatus = 'pending' | 'approved' | 'denied' | 'expired' | 'consumed'

/** Honest description of the isolation a call received. Never embellish this. */
export type SandboxDescription = {
  backend: string
  profile: string
  isolated: boolean
  network: boolean
  workspace_path?: string | null
  detail: string
}

export type AgentApproval = {
  id: string
  session_id: string
  tool: string
  arguments: Record<string, unknown>
  arguments_hash: string
  environment: AgentEnvironment
  risk: ToolRiskLevel
  scope_key: string
  /** False for destructive and host actions: those are always one-shot. */
  allow_session_scope: boolean
  elevation: AgentElevationRequest
  summary: string
  sandbox: SandboxDescription
  status: ApprovalStatus
  scope?: ApprovalScope
  note?: string
  decided_at?: string
  created_at: string
}

/** A tool the application can execute, as published by the daemon. */
export type AgentToolDefinition = {
  name: string
  label: string
  description: string
  inputSchema: Record<string, unknown>
  risk: ToolRiskLevel
  executes: boolean
  needsWorkspace: boolean
  defaultEnvironment: AgentEnvironment
}

export type AgentModelReference = {
  /** Brazier model id, e.g. `gguf:owner/repo/file.gguf`. */
  id: string
  /** Display label for the UI. */
  name?: string
  contextWindow?: number
  maxTokens?: number
}

/**
 * What a model can be trusted to do. Weak local models need argument repair and
 * one tool call per turn; capable ones do not.
 */
export type AgentModelCapabilities = {
  nativeToolCalling: boolean
  parallelToolCalling: boolean
  supportsReasoningStream: boolean
  reliableJson: boolean
  maxToolsPerTurn?: number
}

export type AgentRunStatus =
  | 'idle'
  | 'running'
  | 'completed'
  | 'cancelled'
  | 'failed'
  | 'awaiting-approval'

/** A tool call an assistant message asked for. */
export type AgentToolCallSummary = {
  id: string
  name: string
  arguments: Record<string, unknown>
}

/**
 * Runtime-neutral transcript entry. This is what gets persisted, so a session
 * survives a runtime upgrade or replacement.
 */
export type AgentMessage =
  | { role: 'user'; text: string; timestamp: string }
  | {
      role: 'assistant'
      text: string
      reasoning?: string
      toolCalls?: AgentToolCallSummary[]
      timestamp: string
      error?: string
    }
  | {
      role: 'tool'
      toolCallId: string
      tool: string
      output: string
      isError: boolean
      timestamp: string
    }
  | { role: 'system'; text: string; timestamp: string }

/** One recorded tool execution, as the daemon stored it. */
export type ToolExecutionRecord = {
  id: string
  session_id: string
  run_id?: string
  tool_call_id?: string
  tool: string
  arguments: Record<string, unknown>
  environment: string
  risk: string
  status: string
  exit_code?: number | null
  output_preview?: string
  artifact_id?: string
  truncated: boolean
  changed_paths?: string[]
  sandbox?: SandboxDescription
  approval_id?: string
  error?: string
  duration_ms?: number
  created_at: string
}

export type AgentCompactionState = {
  compactedAt: string
  removedMessages: number
  summary: string
  /**
   * Where the narrative half came from. `deterministic` means the model was not
   * asked or did not answer, and the summary is the machine-built digest alone —
   * worth recording, because the two read very differently and a session that
   * silently lost its reasoning should be explicable afterwards.
   */
  summarySource?: 'model' | 'deterministic'
}

/** Canonical session state. The application owns this, not the runtime. */
export type AgentSessionState = {
  id: string
  title: string
  workspacePath?: string | null
  model: AgentModelReference
  runtimeId: string
  messages: AgentMessage[]
  toolExecutions: ToolExecutionRecord[]
  permissionMode: AgentPermissionMode
  permissionSettings: AgentPermissionSettings
  enabledTools?: string[]
  createdAt: string
  updatedAt: string
  lastRunStatus: AgentRunStatus
  compactionState?: AgentCompactionState
  runtimeMetadata?: Record<string, unknown>
}

// --- Events -----------------------------------------------------------------

type EventBase = {
  sessionId: string
  runId: string
  timestamp: string
  sequence: number
}

export type AgentRunStartedEvent = EventBase & { type: 'run-started' }
export type AgentPrefillProgressEvent = EventBase & {
  type: 'prefill-progress'
  total: number
  cached: number
  processed: number
  elapsedMs: number
  contextTotal?: number | null
}
export type AgentTextDeltaEvent = EventBase & {
  type: 'text-delta'
  delta: string
  /** `reasoning` when the model is streaming its thinking. */
  channel: 'text' | 'reasoning'
}
export type AgentToolCallProposedEvent = EventBase & {
  type: 'tool-call-proposed'
  toolCallId: string
  tool: string
  args: Record<string, unknown>
  environment: AgentEnvironment
  risk: ToolRiskLevel
}
export type AgentApprovalRequiredEvent = EventBase & {
  type: 'approval-required'
  toolCallId: string
  approval: AgentApproval
}
export type AgentElevationRequestedEvent = EventBase & {
  type: 'elevation-requested'
  toolCallId: string
  approvalId: string
  request: AgentElevationRequest
}
export type AgentToolStartedEvent = EventBase & {
  type: 'tool-started'
  toolCallId: string
  tool: string
  args: Record<string, unknown>
  environment: AgentEnvironment
  sandbox: SandboxDescription
}
export type AgentToolOutputEvent = EventBase & {
  type: 'tool-output'
  toolCallId: string
  tool: string
  chunk: string
}
export type AgentToolCompletedEvent = EventBase & {
  type: 'tool-completed'
  toolCallId: string
  tool: string
  environment: AgentEnvironment
  sandbox: SandboxDescription
  output: string
  truncated: boolean
  artifactId?: string
  exitCode?: number | null
  changedPaths: string[]
  durationMs: number
  executionId?: string
}
export type AgentToolFailedEvent = EventBase & {
  type: 'tool-failed'
  toolCallId: string
  tool: string
  environment: AgentEnvironment
  sandbox: SandboxDescription
  error: string
  /** True when the user or the policy refused, rather than the tool erroring. */
  denied: boolean
  durationMs: number
}
export type AgentMessageCommittedEvent = EventBase & {
  type: 'message-committed'
  message: AgentMessage
}
export type AgentRunCompletedEvent = EventBase & {
  type: 'run-completed'
  summary: AgentRunSummary
}
export type AgentRunCancelledEvent = EventBase & { type: 'run-cancelled' }
export type AgentRunFailedEvent = EventBase & { type: 'run-failed'; error: string }
export type AgentCompactedEvent = EventBase & {
  type: 'compacted'
  state: AgentCompactionState
}
export type AgentSubagentStartedEvent = EventBase & {
  type: 'subagent-started'
  /** Parent `spawn_subagent` tool call id. */
  toolCallId: string
  childSessionId: string
  model: string
  prompt: string
}
export type AgentSubagentCompletedEvent = EventBase & {
  type: 'subagent-completed'
  toolCallId: string
  childSessionId: string
  model: string
  status: 'completed' | 'failed' | 'cancelled'
  summary: string
}
/** Live nested work from a child session, for the parent's timeline pill. */
export type AgentSubagentProgressEvent = EventBase & {
  type: 'subagent-progress'
  toolCallId: string
  childSessionId: string
  activity: {
    id: string
    tool?: string
    args?: Record<string, unknown>
    status: 'running' | 'completed' | 'failed' | 'denied' | 'awaiting-approval'
    detail?: string
    environment?: AgentEnvironment
  }
}

export type AgentEvent =
  | AgentRunStartedEvent
  | AgentPrefillProgressEvent
  | AgentTextDeltaEvent
  | AgentToolCallProposedEvent
  | AgentApprovalRequiredEvent
  | AgentElevationRequestedEvent
  | AgentToolStartedEvent
  | AgentToolOutputEvent
  | AgentToolCompletedEvent
  | AgentToolFailedEvent
  | AgentMessageCommittedEvent
  | AgentRunCompletedEvent
  | AgentRunCancelledEvent
  | AgentRunFailedEvent
  | AgentCompactedEvent
  | AgentSubagentStartedEvent
  | AgentSubagentCompletedEvent
  | AgentSubagentProgressEvent

/** End-of-run report the UI renders. */
export type AgentRunSummary = {
  filesChanged: string[]
  commandsRun: string[]
  toolCalls: number
  failures: string[]
  hostActions: string[]
  approvalsRequested: number
  text: string
}

// --- Runtime contract -------------------------------------------------------

export type AgentUserInput = {
  text: string
  /** Data URLs for images the model can see, when it supports them. */
  images?: string[]
}

export type AgentRuntimeCapabilities = {
  streaming: boolean
  toolCalls: boolean
  compaction: boolean
  cancellation: boolean
  sessionRestore: boolean
}

export type AgentRuntimeDescriptor = {
  id: string
  name: string
  version: string
  capabilities: AgentRuntimeCapabilities
}

export type CreateAgentSessionOptions = {
  sessionId: string
  model: AgentModelReference
  systemPrompt: string
  tools: AgentToolDefinition[]
  /** Prior transcript, when resuming. */
  messages?: AgentMessage[]
  capabilities: AgentModelCapabilities
}

export interface AgentSession {
  readonly id: string
  run(input: AgentUserInput): AsyncIterable<AgentEvent>
  cancel(): Promise<void>
  compact(): Promise<AgentCompactionState>
  setModel(model: AgentModelReference): Promise<void>
  setEnabledTools(toolNames: string[]): Promise<void>
  getState(): AgentSessionState
  /**
   * Replace the in-memory transcript (and optional system prompt) with what the
   * daemon stored. Required when reopening a session so model context matches
   * the history shown in the UI.
   */
  rehydrate(messages: AgentMessage[], systemPrompt?: string): void
  /** Reload live inference prefs (e.g. drop_reasoning_between_turns) from the daemon. */
  refreshInferencePrefs(): Promise<void>
  /** True after {@link dispose}; the worker must not rehydrate disposed sessions. */
  isDisposed(): boolean
  dispose(): Promise<void>
}

export interface AgentRuntime {
  readonly descriptor: AgentRuntimeDescriptor
  createSession(options: CreateAgentSessionOptions): Promise<AgentSession>
  restoreSession(sessionId: string): Promise<AgentSession>
  dispose(): Promise<void>
}

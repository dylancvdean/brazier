/**
 * Typed wire surface for `omp --mode rpc`.
 *
 * These are structural mirrors of the frames Oh My Pi declares in
 * `packages/coding-agent/src/modes/rpc/rpc-types.ts` plus the session events
 * forwarded from `agent-session.ts`. They are deliberately hand-maintained
 * against the wire (not vendored from `@oh-my-pi/*`) so the rest of Brazier
 * never imports a framework package. Fields are declared for the shapes the
 * GUI renders; unknown fields survive because every frame is transported as a
 * plain object.
 *
 * Release flexibility rule: a new OMP release that adds a frame type must not
 * break the GUI. The renderer routes every frame through a reducer whose
 * default arm handles unknown types generically, so this file only needs
 * updating when a new frame deserves a rich view.
 */

export type OmpRpcFrame = Record<string, unknown>

// --- Thinking / models -----------------------------------------------------

export type OmpThinkingLevel = 'off' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max'
export type OmpEffort = 'none' | 'low' | 'medium' | 'high'

export type OmpModel = {
  provider?: string
  id: string
  name?: string
  input?: string[]
  contextWindow?: number
  maxTokens?: number
  reasoning?: boolean
  supportsTools?: boolean
}

export type OmpContextUsage = {
  tokens?: number
  contextWindow?: number
  percent?: number
}

export type OmpToolSummary = {
  name: string
  description?: string
  parameters?: unknown
  examples?: readonly unknown[]
}

export type OmpTodoStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled'
export type OmpTodoItem = { id: string; content: string; status: OmpTodoStatus }
export type OmpTodoPhase = { id: string; name: string; tasks: OmpTodoItem[] }

// --- Session state (`get_state`) ------------------------------------------

export type OmpSessionState = {
  model?: OmpModel
  thinkingLevel?: OmpThinkingLevel
  isStreaming?: boolean
  isCompacting?: boolean
  steeringMode?: 'all' | 'one-at-a-time'
  followUpMode?: 'all' | 'one-at-a-time'
  interruptMode?: 'immediate' | 'wait'
  sessionFile?: string
  sessionId?: string
  sessionName?: string
  autoCompactionEnabled?: boolean
  fastModeEnabled?: boolean
  fastModeActive?: boolean
  tokensPerSecond?: number | null
  messageCount?: number
  queuedMessageCount?: number
  todoPhases?: OmpTodoPhase[]
  systemPrompt?: string[]
  dumpTools?: OmpToolSummary[]
  contextUsage?: OmpContextUsage
}

// --- Commands (stdin) ------------------------------------------------------

export type OmpRpcCommand =
  // Protocol
  | { id?: string; type: 'negotiate_protocol'; protocolVersion: number }
  // Prompting
  | { id?: string; type: 'prompt'; message: string; images?: unknown[]; streamingBehavior?: 'steer' | 'followUp' }
  | { id?: string; type: 'steer'; message: string; images?: unknown[] }
  | { id?: string; type: 'follow_up'; message: string; images?: unknown[] }
  | { id?: string; type: 'abort' }
  | { id?: string; type: 'abort_and_prompt'; message: string; images?: unknown[] }
  | { id?: string; type: 'new_session'; parentSession?: string }
  // State
  | { id?: string; type: 'get_state' }
  | { id?: string; type: 'set_fast_mode'; enabled: boolean }
  | { id?: string; type: 'get_available_commands' }
  | { id?: string; type: 'set_todos'; phases: OmpTodoPhase[] }
  | { id?: string; type: 'set_host_tools'; tools: unknown[] }
  | { id?: string; type: 'set_host_uri_schemes'; schemes: unknown[] }
  | { id?: string; type: 'set_subagent_subscription'; level: 'off' | 'progress' | 'events' }
  | { id?: string; type: 'get_subagents' }
  | { id?: string; type: 'get_subagent_messages'; subagentId?: string; sessionFile?: string; fromByte?: number }
  // Model
  | { id?: string; type: 'set_model'; provider: string; modelId: string }
  | { id?: string; type: 'cycle_model' }
  | { id?: string; type: 'get_available_models' }
  // Thinking
  | { id?: string; type: 'set_thinking_level'; level: OmpThinkingLevel }
  | { id?: string; type: 'cycle_thinking_level' }
  // Queue modes
  | { id?: string; type: 'set_steering_mode'; mode: 'all' | 'one-at-a-time' }
  | { id?: string; type: 'set_follow_up_mode'; mode: 'all' | 'one-at-a-time' }
  | { id?: string; type: 'set_interrupt_mode'; mode: 'immediate' | 'wait' }
  // Compaction
  | { id?: string; type: 'compact'; customInstructions?: string }
  | { id?: string; type: 'set_auto_compaction'; enabled: boolean }
  // Retry
  | { id?: string; type: 'set_auto_retry'; enabled: boolean }
  | { id?: string; type: 'abort_retry' }
  // Bash
  | { id?: string; type: 'bash'; command: string }
  | { id?: string; type: 'abort_bash' }
  // Session
  | { id?: string; type: 'get_session_stats' }
  | { id?: string; type: 'export_html'; outputPath?: string }
  | { id?: string; type: 'switch_session'; sessionPath: string }
  | { id?: string; type: 'branch'; entryId: string }
  | { id?: string; type: 'get_branch_messages' }
  | { id?: string; type: 'get_last_assistant_text' }
  | { id?: string; type: 'set_session_name'; name: string }
  | { id?: string; type: 'handoff'; customInstructions?: string }
  // Messages
  | { id?: string; type: 'get_messages' }
  | { id?: string; type: 'get_messages_page'; cursor?: string; limit?: number }
  // Login
  | { id?: string; type: 'get_login_providers' }
  | { id?: string; type: 'login'; providerId: string }

export type OmpAvailableCommand = {
  name: string
  aliases?: string[]
  description?: string
  input?: { hint?: string }
  subcommands?: Array<{ name: string; description?: string; usage?: string }>
  source: string
}

export type OmpSubagentSnapshot = {
  id: string
  index: number
  agent?: string
  description?: string
  status?: string
  task?: string
  assignment?: string
  sessionFile?: string
  lastUpdate?: number
  parentToolCallId?: string
  progress?: OmpAgentProgress
}
// --- Session events (stdout) -----------------------------------------------

export type OmpSessionEvent =
  | { type: 'agent_start' } & OmpRpcFrame
  | { type: 'agent_end'; isTerminal?: boolean } & OmpRpcFrame
  | { type: 'turn_start' | 'turn_end' } & OmpRpcFrame
  | { type: 'message_start' | 'message_update' | 'message_end' } & OmpRpcFrame
  | { type: 'tool_execution_start' | 'tool_execution_update' | 'tool_execution_end' } & OmpRpcFrame
  | { type: 'auto_compaction_start'; reason?: string; action?: string } & OmpRpcFrame
  | { type: 'auto_compaction_end'; action?: string; aborted?: boolean; willRetry?: boolean } & OmpRpcFrame
  | { type: 'auto_retry_start'; attempt?: number; maxAttempts?: number; errorMessage?: string } & OmpRpcFrame
  | { type: 'auto_retry_end'; success?: boolean; attempt?: number; finalError?: string } & OmpRpcFrame
  | { type: 'retry_fallback_applied' | 'retry_fallback_succeeded'; from?: string; to?: string; role?: string } & OmpRpcFrame
  | { type: 'model_changed' } & OmpRpcFrame
  | { type: 'ttsr_triggered'; rules?: unknown[] } & OmpRpcFrame
  | { type: 'todo_reminder'; todos?: OmpTodoItem[]; attempt?: number; maxAttempts?: number } & OmpRpcFrame
  | { type: 'todo_auto_clear' } & OmpRpcFrame
  | { type: 'irc_message' } & OmpRpcFrame
  | { type: 'notice'; level?: 'info' | 'warning' | 'error'; message?: string; source?: string } & OmpRpcFrame
  | { type: 'thinking_level_changed'; thinkingLevel?: OmpThinkingLevel } & OmpRpcFrame
  | { type: 'goal_updated' } & OmpRpcFrame

// --- Side channels (slash commands) ----------------------------------------

export type OmpSideChannel =
  | { type: 'command_output'; text: string }
  | { type: 'session_info_update'; title?: string; sessionId?: string }
  | { type: 'config_update'; model?: OmpModel; thinkingLevel?: OmpThinkingLevel }
  | { type: 'available_commands_update'; commands: OmpAvailableCommand[] }
  | { type: 'prompt_result'; id?: string; agentInvoked: boolean }
  | { type: 'extension_error'; extensionPath?: string; event?: string; error?: string }

// --- Extension UI -----------------------------------------------------------

export type OmpExtensionUiRequest =
  | { type: 'extension_ui_request'; id: string; method: 'select'; title: string; options: string[]; timeout?: number }
  | { type: 'extension_ui_request'; id: string; method: 'confirm'; title: string; message: string; timeout?: number }
  | { type: 'extension_ui_request'; id: string; method: 'input'; title: string; placeholder?: string; timeout?: number }
  | { type: 'extension_ui_request'; id: string; method: 'editor'; title: string; prefill?: string; promptStyle?: boolean }
  | { type: 'extension_ui_request'; id: string; method: 'cancel'; targetId: string }
  | { type: 'extension_ui_request'; id: string; method: 'notify'; message: string; notifyType?: string }
  | { type: 'extension_ui_request'; id: string; method: 'setStatus'; statusKey: string; statusText?: string }
  | { type: 'extension_ui_request'; id: string; method: 'setWidget'; widgetKey: string; widgetLines?: string[] | null; widgetPlacement?: string }
  | { type: 'extension_ui_request'; id: string; method: 'setTitle'; title: string }
  | { type: 'extension_ui_request'; id: string; method: 'set_editor_text'; text: string }
  | { type: 'extension_ui_request'; id: string; method: 'open_url'; url: string; launchUrl?: string; instructions?: string }

export type OmpExtensionUiResponse =
  | { type: 'extension_ui_response'; id: string; value: string }
  | { type: 'extension_ui_response'; id: string; confirmed: boolean }
  | { type: 'extension_ui_response'; id: string; cancelled: true; timedOut?: boolean }

// --- Host tool / URI --------------------------------------------------------

export type OmpHostToolCall = {
  type: 'host_tool_call'
  id: string
  toolCallId: string
  toolName: string
  arguments: Record<string, unknown>
}
export type OmpHostToolCancel = { type: 'host_tool_cancel'; id: string; targetId: string }
export type OmpHostUriRequest = {
  type: 'host_uri_request'
  id: string
  operation: 'read' | 'write'
  url: string
  content?: string
}
export type OmpHostUriCancel = { type: 'host_uri_cancel'; id: string; targetId: string }

// --- Subagents --------------------------------------------------------------

export type OmpAgentSource = 'bundled' | 'user' | 'project'
export type OmpSubagentStatus = 'pending' | 'running' | 'completed' | 'failed' | 'aborted'

/** Live progress of one subagent, mirrored from OMP's `AgentProgress`. */
export type OmpAgentProgress = {
  index: number
  id: string
  agent: string
  agentSource: OmpAgentSource
  status: OmpSubagentStatus
  task: string
  assignment?: string
  description?: string
  lastIntent?: string
  currentTool?: string
  currentToolArgs?: string
  currentToolStartMs?: number
  recentTools?: Array<{ tool: string; args: string; endMs: number }>
  recentOutput?: string[]
  toolCount?: number
  requests?: number
  tokens?: number
  contextTokens?: number
  contextWindow?: number
  cost?: number
  durationMs?: number
  resolvedModel?: string
  resolvedModelIsFallback?: boolean
  retryState?: { attempt: number; maxAttempts: number; delayMs: number; errorMessage: string; startedAtMs: number }
  retryFailure?: { attempt: number; errorMessage: string }
}

export type OmpSubagentLifecyclePayload = {
  id: string
  agent: string
  agentSource: OmpAgentSource
  description?: string
  status: 'started' | 'completed' | 'failed' | 'aborted'
  sessionFile?: string
  parentToolCallId?: string
  index: number
  detached?: boolean
}

export type OmpSubagentProgressPayload = {
  index: number
  agent: string
  agentSource: OmpAgentSource
  task: string
  parentToolCallId?: string
  assignment?: string
  progress: OmpAgentProgress
  sessionFile?: string
  detached?: boolean
}

export type OmpSubagentEventPayload = {
  id: string
  /** One session event from inside the subagent's own run. */
  event: Record<string, unknown>
}

export type OmpSubagentFrame =
  | { type: 'subagent_lifecycle'; payload: OmpSubagentLifecyclePayload }
  | { type: 'subagent_progress'; payload: OmpSubagentProgressPayload }
  | { type: 'subagent_event'; payload: OmpSubagentEventPayload }

// --- Response ---------------------------------------------------------------

export type OmpRpcResponse = {
  id?: string
  type: 'response'
  command: string
  success: boolean
  data?: unknown
  error?: string
  code?: string
}

/** Every non-response frame the sidecar can emit on stdout. */
export type OmpEventFrame =
  | OmpSessionEvent
  | OmpSideChannel
  | OmpExtensionUiRequest
  | OmpHostToolCall
  | OmpHostToolCancel
  | OmpHostUriRequest
  | OmpHostUriCancel
  | OmpSubagentFrame
  | { type: 'available_commands_update'; commands: OmpAvailableCommand[] }

/** Every stdout frame except the transport-level `ready`/`rpc_chunk`. */
export type OmpOutboundFrame = OmpEventFrame | OmpRpcResponse

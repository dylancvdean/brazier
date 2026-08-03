/**
 * OMP sidecar state for Agent mode.
 *
 * The worker mirrors the `omp --mode rpc` stdout stream to the renderer as
 * `runtime-frame` messages. This reducer is the renderer's lossless window onto
 * that stream: known frame types get a rich arm, and everything else falls
 * through to a bounded generic record. That default arm is what keeps new OMP
 * releases from breaking the GUI — an unknown frame is surfaced, not dropped.
 *
 * Frames the shared run/transcript path already renders (message updates, tool
 * executions, host tool calls, extension UI requests) are deliberately ignored
 * here so the "recent OMP events" list stays free of duplicates.
 */

import type {
  OmpAgentProgress,
  OmpAvailableCommand,
  OmpRpcFrame,
  OmpSessionState,
  OmpSubagentLifecyclePayload,
  OmpSubagentProgressPayload,
  OmpSubagentSnapshot,
  OmpTodoItem,
  OmpTodoPhase
} from '../../agent/omp/rpcTypes'

export type OmpCommandSuggestion = { value: string; description: string }

export type OmpCommandOutput = {
  id: string
  text: string
  timestamp: string
}

export type OmpRecentFrame = {
  id: string
  type: string
  detail: string
  timestamp: string
}

export type OmpContextUsage = {
  tokens?: number
  contextWindow?: number
  percent?: number
}

export type OmpSessionInfo = {
  title?: string
  sessionId?: string
  modelId?: string
  modelName?: string
  thinkingLevel?: string
  contextUsage?: OmpContextUsage
  fastModeEnabled?: boolean
  fastModeActive?: boolean
  autoCompactionEnabled?: boolean
  isStreaming?: boolean
  isCompacting?: boolean
  tokensPerSecond?: number | null
  todoPhases?: OmpTodoPhase[]
}

/** One live/finished task subagent, reduced from OMP's subagent frames. */
export type OmpSubagentView = {
  id: string
  index: number
  agent: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'aborted'
  description?: string
  task?: string
  assignment?: string
  parentToolCallId?: string
  sessionFile?: string
  detached?: boolean
  currentTool?: string
  currentToolArgs?: string
  recentOutput: string[]
  toolCount?: number
  tokens?: number
  cost?: number
  resolvedModel?: string
  contextTokens?: number
  contextWindow?: number
  lastIntent?: string
  lastUpdate?: number
}

/** An interactive extension-UI request surfaced to the GUI for an answer. */
export type OmpPendingDialog =
  | { kind: 'confirm'; id: string; title: string; message: string }
  | { kind: 'select'; id: string; title: string; options: string[] }
  | { kind: 'input'; id: string; title: string; placeholder?: string }
  | { kind: 'editor'; id: string; title: string; prefill?: string }

export type OmpSidecarState = {
  /** Live slash-command list from `available_commands_update` frames. */
  commands: OmpCommandSuggestion[]
  /** Recent `command_output` blocks, newest last. */
  commandOutputs: OmpCommandOutput[]
  /** Latest session metadata from `get_state` / session/config updates. */
  session: OmpSessionInfo | null
  /** Task subagents, indexed by id, newest activity last. */
  subagents: OmpSubagentView[]
  /** The extension-UI dialog waiting on the user, if any. */
  pendingDialog: OmpPendingDialog | null
  /** Bounded record of frames the GUI does not render richly yet. */
  recentFrames: OmpRecentFrame[]
}

export const EMPTY_OMP_SIDECAR: OmpSidecarState = {
  commands: [],
  commandOutputs: [],
  session: null,
  subagents: [],
  pendingDialog: null,
  recentFrames: []
}

const MAX_COMMAND_OUTPUTS = 12
const MAX_SUBAGENTS = 12
const MAX_RECENT_FRAMES = 30

/** Frames that the shared transcript/timeline path already shows or that are
 *  pure transport noise. Excluded so the generic record stays meaningful. */
const IGNORED_FRAME_TYPES = new Set([
  'ready',
  'rpc_chunk',
  'response',
  'message_start',
  'message_update',
  'message_end',
  'agent_start',
  'agent_end',
  'turn_start',
  'turn_end',
  'tool_execution_start',
  'tool_execution_update',
  'tool_execution_end',
  'host_tool_call',
  'host_tool_cancel'
])

function suggestionValue(raw: string): string {
  return raw.startsWith('/') ? raw : `/${raw}`
}

export function commandSuggestion(command: OmpAvailableCommand): OmpCommandSuggestion {
  return {
    value: suggestionValue(command.name),
    description:
      command.description ??
      command.subcommands?.map((subcommand) => subcommand.name).join(', ') ??
      'OMP command'
  }
}

function dedupeCommands(commands: OmpCommandSuggestion[]): OmpCommandSuggestion[] {
  return [...new Map(commands.map((entry) => [entry.value, entry])).values()]
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

function asBoolean(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined
}

function asNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

/** Fold a `get_state` snapshot into the session metadata without losing fields. */
/** Fold a `get_state` snapshot into the session metadata without losing fields. */
function sessionInfoFromState(prev: OmpSessionInfo | null, data: OmpSessionState): OmpSessionInfo {
  const next: OmpSessionInfo = { ...(prev ?? {}) }
  const model = data.model && typeof data.model === 'object' ? data.model : undefined
  const modelId = asString(model?.id)
  const modelName = asString(model?.name)
  if (modelId) next.modelId = modelId
  if (modelName) next.modelName = modelName
  else if (modelId) next.modelName = modelId
  if (asString(data.thinkingLevel)) next.thinkingLevel = asString(data.thinkingLevel)
  if (typeof data.contextUsage === 'object' && data.contextUsage !== null) {
    next.contextUsage = {
      tokens: asNumber(data.contextUsage.tokens),
      contextWindow: asNumber(data.contextUsage.contextWindow),
      percent: asNumber(data.contextUsage.percent)
    }
  }
  if (asBoolean(data.fastModeEnabled) !== undefined) next.fastModeEnabled = asBoolean(data.fastModeEnabled)
  if (asBoolean(data.fastModeActive) !== undefined) next.fastModeActive = asBoolean(data.fastModeActive)
  if (asBoolean(data.autoCompactionEnabled) !== undefined) {
    next.autoCompactionEnabled = asBoolean(data.autoCompactionEnabled)
  }
  if (asBoolean(data.isStreaming) !== undefined) next.isStreaming = asBoolean(data.isStreaming)
  if (asBoolean(data.isCompacting) !== undefined) next.isCompacting = asBoolean(data.isCompacting)
  if (asNumber(data.tokensPerSecond) !== undefined) next.tokensPerSecond = asNumber(data.tokensPerSecond)
  if (Array.isArray(data.todoPhases)) next.todoPhases = data.todoPhases
  if (asString(data.sessionName)) next.title = asString(data.sessionName)
  if (asString(data.sessionId)) next.sessionId = asString(data.sessionId)
  return next
}

/** Merge a flat `todo_reminder` item list into the phase view by task id. */
function mergeTodosIntoPhases(phases: OmpTodoPhase[] | undefined, todos: OmpTodoItem[]): OmpTodoPhase[] {
  if (todos.length === 0) return phases ?? []
  const byId = new Map(todos.map((item) => [item.id, item]))
  if (!phases || phases.length === 0) {
    return [{ id: 'tasks', name: 'Tasks', tasks: todos }]
  }
  return phases.map((phase) => ({
    ...phase,
    tasks: phase.tasks.map((task) => byId.get(task.id) ?? task)
  }))
}

function upsertSubagent(
  subagents: OmpSubagentView[],
  patch: Partial<OmpSubagentView> & { id: string; index: number }
): OmpSubagentView[] {
  const existing = subagents.find((entry) => entry.id === patch.id)
  const next: OmpSubagentView = existing
    ? { ...existing, ...patch }
    : { agent: 'task', status: 'running', recentOutput: [], ...patch }
  const without = subagents.filter((entry) => entry.id !== patch.id)
  return [...without, next].slice(-MAX_SUBAGENTS)
}

function subagentFromProgress(payload: OmpSubagentProgressPayload): OmpSubagentView {
  const progress = payload.progress
  return {
    id: progress.id,
    index: payload.index,
    agent: progress.agent,
    status: progress.status,
    task: payload.task,
    assignment: payload.assignment,
    parentToolCallId: payload.parentToolCallId,
    sessionFile: payload.sessionFile,
    detached: payload.detached,
    description: progress.description,
    lastIntent: progress.lastIntent,
    currentTool: progress.currentTool,
    currentToolArgs: progress.currentToolArgs,
    recentOutput: Array.isArray(progress.recentOutput) ? progress.recentOutput : [],
    toolCount: progress.toolCount,
    tokens: progress.tokens,
    cost: progress.cost,
    resolvedModel: progress.resolvedModel,
    contextTokens: progress.contextTokens,
    contextWindow: progress.contextWindow,
    lastUpdate: Date.now()
  }
}

function subagentFromLifecycle(payload: OmpSubagentLifecyclePayload): OmpSubagentView {
  return {
    id: payload.id,
    index: payload.index,
    agent: payload.agent,
    status: payload.status === 'started' ? 'running' : payload.status,
    description: payload.description,
    parentToolCallId: payload.parentToolCallId,
    sessionFile: payload.sessionFile,
    detached: payload.detached,
    recentOutput: [],
    lastUpdate: Date.now()
  }
}

function subagentFromSnapshot(snapshot: OmpSubagentSnapshot): OmpSubagentView {
  const progress = snapshot.progress as OmpAgentProgress | undefined
  return {
    id: snapshot.id,
    index: snapshot.index,
    agent: snapshot.agent ?? 'task',
    status: (snapshot.status as OmpSubagentView['status']) ?? 'pending',
    description: snapshot.description,
    task: snapshot.task,
    assignment: snapshot.assignment,
    sessionFile: snapshot.sessionFile,
    parentToolCallId: snapshot.parentToolCallId,
    currentTool: progress?.currentTool,
    recentOutput: Array.isArray(progress?.recentOutput) ? progress.recentOutput : [],
    toolCount: progress?.toolCount,
    tokens: progress?.tokens,
    cost: progress?.cost,
    resolvedModel: progress?.resolvedModel,
    contextTokens: progress?.contextTokens,
    contextWindow: progress?.contextWindow,
    lastIntent: progress?.lastIntent,
    lastUpdate: snapshot.lastUpdate
  }
}

/** Fold an extension-UI request into a surfaced dialog (or an event record). */
function dialogFromFrame(state: OmpSidecarState, frame: OmpRpcFrame): OmpSidecarState {
  const id = asString(frame.id)
  if (!id) return state
  const method = String(frame.method ?? '')
  const title = asString(frame.title) ?? 'Oh My Pi'

  if (method === 'select') {
    const options = Array.isArray(frame.options)
      ? frame.options.filter((option): option is string => typeof option === 'string')
      : []
    if (options.length === 0) return state
    return { ...state, pendingDialog: { kind: 'select', id, title, options } }
  }
  if (method === 'confirm') {
    const message = asString(frame.message) ?? ''
    return { ...state, pendingDialog: { kind: 'confirm', id, title, message } }
  }
  if (method === 'input') {
    return { ...state, pendingDialog: { kind: 'input', id, title, placeholder: asString(frame.placeholder) } }
  }
  if (method === 'editor') {
    return { ...state, pendingDialog: { kind: 'editor', id, title, prefill: asString(frame.prefill) } }
  }
  if (method === 'cancel') {
    const targetId = asString(frame.targetId)
    if (targetId && state.pendingDialog?.id === targetId) {
      return { ...state, pendingDialog: null }
    }
    return state
  }
  if (method === 'notify') {
    const message = asString(frame.message)
    if (!message) return state
    const recentFrame: OmpRecentFrame = {
      id: crypto.randomUUID(),
      type: 'notification',
      detail: message,
      timestamp: new Date().toISOString()
    }
    return { ...state, recentFrames: [...state.recentFrames, recentFrame].slice(-MAX_RECENT_FRAMES) }
  }
  // setStatus / setTitle / set_editor_text / open_url are auto-resolved by the
  // runtime; nothing to render here.
  return state
}

/** Short human-readable summary of a frame for the generic events record. */
export function frameDetail(frame: OmpRpcFrame): string {
  switch (frame.type) {
    case 'notice':
      return asString(frame.message) ?? 'notice'
    case 'thinking_level_changed':
      return asString(frame.thinkingLevel) ? `thinking → ${frame.thinkingLevel}` : 'thinking level changed'
    case 'model_changed':
      return 'model changed'
    case 'goal_updated':
      return 'goal updated'
    case 'todo_reminder':
      return `todo reminder (${Array.isArray(frame.todos) ? frame.todos.length : '?'})`
    case 'todo_auto_clear':
      return 'todos cleared'
    case 'ttsr_triggered':
      return `stream rule injected (${Array.isArray(frame.rules) ? frame.rules.length : '?'})`
    case 'auto_compaction_start':
      return `compaction started${asString(frame.reason) ? ` (${frame.reason})` : ''}`
    case 'auto_compaction_end':
      return `compaction ${frame.aborted ? 'aborted' : frame.skipped ? 'skipped' : 'finished'}`
    case 'auto_retry_start':
      return `retrying (${asString(frame.attempt)}/${asString(frame.maxAttempts)})`
    case 'auto_retry_end':
      return frame.success ? 'retry recovered' : 'retry exhausted'
    case 'retry_fallback_applied':
      return `fallback ${asString(frame.from) ?? '?'} → ${asString(frame.to) ?? '?'}`
    case 'retry_fallback_succeeded':
      return `fallback recovered on ${asString(frame.model) ?? '?'}`
    case 'irc_message':
      return 'IRC message'
    case 'prompt_result':
      return frame.agentInvoked ? 'agent invoked' : 'command completed locally'
    case 'subagent_lifecycle':
      return 'subagent lifecycle'
    case 'subagent_progress':
      return 'subagent progress'
    case 'subagent_event':
      return 'subagent event'
    case 'host_uri_request':
      return `uri ${asString(frame.operation) ?? '?'} ${asString(frame.url) ?? ''}`
    case 'host_uri_cancel':
      return 'uri cancelled'
    case 'session_info_update':
      return 'session info updated'
    case 'config_update':
      return 'config updated'
    case 'extension_error':
      return asString(frame.error) ?? 'extension error'
    default:
      return String(frame.type)
  }
}

export type OmpSidecarAction =
  | { type: 'frame'; frame: OmpRpcFrame }
  | { type: 'reset' }
  | { type: 'dialog-resolved' }

export function ompSidecarReducer(
  state: OmpSidecarState,
  action: OmpSidecarAction
): OmpSidecarState {
  if (action.type === 'reset') return EMPTY_OMP_SIDECAR
  if (action.type === 'dialog-resolved') {
    if (!state.pendingDialog) return state
    return { ...state, pendingDialog: null }
  }
  const frame = action.frame
  const type = String(frame.type ?? '')

  switch (type) {
    case 'command_output': {
      const text = asString(frame.text)
      if (!text) return state
      const output: OmpCommandOutput = {
        id: crypto.randomUUID(),
        text,
        timestamp: new Date().toISOString()
      }
      return {
        ...state,
        commandOutputs: [...state.commandOutputs, output].slice(-MAX_COMMAND_OUTPUTS)
      }
    }
    case 'available_commands_update': {
      // OMP pushes the full command list each time, so replace rather than merge.
      const commands = Array.isArray(frame.commands) ? frame.commands : []
      const suggestions = commands
        .filter((command): command is OmpAvailableCommand => Boolean(command && typeof command === 'object'))
        .map(commandSuggestion)
      return { ...state, commands: dedupeCommands(suggestions) }
    }
    case 'session_info_update': {
      const title = asString(frame.title)
      const sessionId = asString(frame.sessionId)
      if (!title && !sessionId) return state
      return {
        ...state,
        session: {
          ...(state.session ?? {}),
          ...(title ? { title } : {}),
          ...(sessionId ? { sessionId } : {})
        }
      }
    }
    case 'config_update': {
      const model =
        frame.model && typeof frame.model === 'object'
          ? asString((frame.model as Record<string, unknown>).id) ?? asString((frame.model as Record<string, unknown>).name)
          : undefined
      const thinkingLevel = asString(frame.thinkingLevel)
      if (!model && !thinkingLevel) return state
      return {
        ...state,
        session: {
          ...(state.session ?? {}),
          ...(model ? { modelName: model } : {}),
          ...(thinkingLevel ? { thinkingLevel } : {})
        }
      }
    }
    case 'response': {
      // The worker forwards every command response too. The state bar reads the
      // ones that carry session state; the rest are transport noise.
      if (frame.success !== true || typeof frame.command !== 'string') return state
      switch (frame.command) {
        case 'get_state': {
          const data = frame.data
          if (!data || typeof data !== 'object') return state
          return { ...state, session: sessionInfoFromState(state.session, data as OmpSessionState) }
        }
        case 'set_fast_mode': {
          const enabled = asBoolean((frame.data as Record<string, unknown> | undefined)?.enabled)
          const active = asBoolean((frame.data as Record<string, unknown> | undefined)?.active)
          if (enabled === undefined && active === undefined) return state
          return {
            ...state,
            session: {
              ...(state.session ?? {}),
              ...(enabled !== undefined ? { fastModeEnabled: enabled } : {}),
              ...(active !== undefined ? { fastModeActive: active } : {})
            }
          }
        }
        case 'cycle_thinking_level': {
          const level = asString((frame.data as Record<string, unknown> | undefined)?.level)
          if (!level) return state
          return { ...state, session: { ...(state.session ?? {}), thinkingLevel: level } }
        }
        case 'get_subagents': {
          const snapshots = (frame.data as Record<string, unknown> | undefined)?.subagents
          if (!Array.isArray(snapshots)) return state
          const views = snapshots
            .filter((entry): entry is OmpSubagentSnapshot => Boolean(entry && typeof entry === 'object'))
            .map(subagentFromSnapshot)
            .sort((a, b) => a.index - b.index)
            .slice(-MAX_SUBAGENTS)
          return { ...state, subagents: views }
        }
        default:
          return state
      }
    }
    case 'subagent_lifecycle': {
      const payload = frame.payload
      if (!payload || typeof payload !== 'object') return state
      const view = subagentFromLifecycle(payload as OmpSubagentLifecyclePayload)
      return { ...state, subagents: upsertSubagent(state.subagents, view) }
    }
    case 'subagent_progress': {
      const payload = frame.payload
      if (!payload || typeof payload !== 'object') return state
      const progress = (payload as OmpSubagentProgressPayload).progress
      if (!progress || typeof progress !== 'object' || !asString(progress.id)) return state
      const view = subagentFromProgress(payload as OmpSubagentProgressPayload)
      return { ...state, subagents: upsertSubagent(state.subagents, view) }
    }
    case 'subagent_event': {
      // The full event stream is heavy; the panel reads lifecycle + progress.
      return state
    }
    case 'extension_ui_request': {
      return dialogFromFrame(state, frame)
    }
    case 'thinking_level_changed': {
      const level = asString(frame.thinkingLevel)
      if (!level) return state
      return { ...state, session: { ...(state.session ?? {}), thinkingLevel: level } }
    }
    case 'todo_reminder': {
      const todos = Array.isArray(frame.todos) ? (frame.todos as OmpTodoItem[]) : []
      if (todos.length === 0) return state
      return {
        ...state,
        session: {
          ...(state.session ?? {}),
          todoPhases: mergeTodosIntoPhases(state.session?.todoPhases, todos)
        }
      }
    }
    case 'todo_auto_clear': {
      if (!state.session?.todoPhases?.length) return state
      return { ...state, session: { ...state.session, todoPhases: [] } }
    }
    default: {
      if (IGNORED_FRAME_TYPES.has(type)) return state
      const recentFrame: OmpRecentFrame = {
        id: crypto.randomUUID(),
        type,
        detail: frameDetail(frame),
        timestamp: new Date().toISOString()
      }
      return {
        ...state,
        recentFrames: [...state.recentFrames, recentFrame].slice(-MAX_RECENT_FRAMES)
      }
    }
  }
}

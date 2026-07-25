import {
  AlertTriangle,
  Bot,
  Check,
  ChevronDown,
  FolderOpen,
  Layers,
  LoaderCircle,
  Lock,
  ShieldAlert,
  ShieldCheck,
  Square,
  Terminal,
  Trash2,
  Wrench,
  X
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type {
  AgentApproval,
  AgentEvent,
  AgentMessage,
  AgentPermissionMode,
  AgentRunSummary,
  SandboxDescription,
  ToolExecutionRecord
} from '../../../agent/core/types'
import type { WorkerMessage } from '../../../agent/core/protocol'
import {
  createAgentSession,
  decideAgentApproval,
  deleteAgentSession,
  fetchAgentArtifact,
  fetchAgentCapabilities,
  fetchAgentSession,
  fetchAgentTools,
  listAgentSessions,
  sandboxBadge,
  updateAgentSession,
  validateAgentWorkspace,
  type AgentCapabilities,
  type AgentSessionSummary,
  type AgentToolCatalogEntry
} from '../agentApi'
import type { LocalModel } from '../api'
import { modelDisplayName } from '../model-utils'

/**
 * What the shared composer needs to drive a run. Agent mode has no input of its
 * own: the one composer at the bottom of the window serves every mode, so this
 * is how the agent's submit, stop, and readiness reach it.
 */
export type AgentComposerControls = {
  send: (text: string) => Promise<void>
  stop: () => Promise<void>
  running: boolean
  /** Empty when a run can start; otherwise why it cannot. */
  blockedReason: string
  placeholder: string
}

type Props = {
  /** Chat model chosen in the top bar; the agent uses the same picker. */
  modelId: string
  models: LocalModel[]
  /** Publish the controls upward; called with null when Agent mode unmounts. */
  onComposerChange?: (controls: AgentComposerControls | null) => void
  /** Put a suggested task into the shared composer for the user to edit. */
  onSuggestPrompt?: (text: string) => void
  /**
   * Bind the selected agent session to the open conversation, so voice and text
   * turns in that conversation reach this session instead of opening their own.
   * Null unbinds.
   */
  onSessionBound?: (agentSessionId: string | null) => void
  onError: (message: string | null) => void
}

/** One row of the activity timeline. */
type TimelineEntry = {
  toolCallId: string
  tool: string
  args: Record<string, unknown>
  environment: 'sandbox' | 'host'
  status: 'running' | 'awaiting-approval' | 'completed' | 'failed' | 'denied'
  sandbox?: SandboxDescription
  output?: string
  error?: string
  exitCode?: number | null
  changedPaths?: string[]
  truncated?: boolean
  artifactId?: string
  durationMs?: number
}

const PERMISSION_LABELS: Record<AgentPermissionMode, { title: string; detail: string }> = {
  ask: {
    title: 'Ask first',
    detail: 'Approve writes, commands, network use, and anything outside the workspace.'
  },
  'sandbox-only': {
    title: 'Sandbox only',
    detail: 'Sandboxed work runs without prompts. Host access is refused outright.'
  },
  'skip-permissions': {
    title: 'Skip permissions',
    detail: 'No prompts for sandboxed work. Host actions still need the separate opt-in.'
  }
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

function shortPath(path: string | null | undefined): string {
  if (!path) return 'No workspace'
  const parts = path.split('/').filter(Boolean)
  return parts.length <= 2 ? path : `…/${parts.slice(-2).join('/')}`
}

function argsPreview(tool: string, args: Record<string, unknown>): string {
  if (typeof args.command === 'string') return args.command
  if (typeof args.path === 'string') return args.path
  if (typeof args.from === 'string' && typeof args.to === 'string') {
    return `${args.from} → ${args.to}`
  }
  if (typeof args.query === 'string') return `“${args.query}”`
  if (typeof args.process_id === 'string') return args.process_id
  const json = JSON.stringify(args)
  return json === '{}' ? tool : json.slice(0, 120)
}

/** Timeline rows rebuilt from what the daemon persisted, for a restored session. */
function timelineFromRecords(records: ToolExecutionRecord[]): TimelineEntry[] {
  return records.map((record) => ({
    toolCallId: record.tool_call_id ?? record.id,
    tool: record.tool,
    args: record.arguments ?? {},
    environment: record.environment === 'host' ? 'host' : 'sandbox',
    status:
      record.status === 'completed'
        ? 'completed'
        : record.status === 'denied'
          ? 'denied'
          : record.status === 'awaiting-approval'
            ? 'awaiting-approval'
            : 'failed',
    sandbox: record.sandbox,
    output: record.output_preview,
    error: record.error ?? undefined,
    exitCode: record.exit_code ?? null,
    changedPaths: record.changed_paths ?? [],
    truncated: record.truncated,
    artifactId: record.artifact_id,
    durationMs: record.duration_ms
  }))
}

function SandboxBadge({
  sandbox
}: {
  sandbox: { isolated: boolean; backend: string; detail: string }
}): React.JSX.Element {
  const badge = sandboxBadge(sandbox)
  return (
    <span className={`agent-sandbox-badge ${badge.tone}`} title={badge.detail}>
      {badge.tone === 'sandboxed' ? <ShieldCheck size={13} /> : <ShieldAlert size={13} />}
      {badge.label}
    </span>
  )
}

function ApprovalCard({
  approval,
  onDecide,
  busy
}: {
  approval: AgentApproval
  onDecide: (
    approval: AgentApproval,
    decision: 'approve' | 'deny',
    scope?: 'once' | 'session',
    note?: string
  ) => void
  busy: boolean
}): React.JSX.Element {
  const [note, setNote] = useState('')
  const elevation = approval.elevation
  const host = approval.environment === 'host' || elevation.requested_host_execution
  const paths = elevation.requested_filesystem_paths ?? []
  return (
    <div className={`agent-approval ${host ? 'host' : ''}`}>
      <div className="agent-approval-head">
        {host ? <ShieldAlert size={16} /> : <Lock size={16} />}
        <div>
          <strong>{approval.summary}</strong>
          <span>{elevation.reason}</span>
        </div>
      </div>
      <dl className="agent-approval-facts">
        <div>
          <dt>Tool</dt>
          <dd>
            {approval.tool} · {approval.risk}
          </dd>
        </div>
        <div>
          <dt>Environment</dt>
          <dd className={host ? 'warn' : ''}>
            {host ? 'Host — no sandbox, full user privileges' : `Sandbox · ${approval.sandbox.backend}`}
          </dd>
        </div>
        {elevation.proposed_command && (
          <div className="wide">
            <dt>Command</dt>
            <dd>
              <code>{elevation.proposed_command}</code>
            </dd>
          </div>
        )}
        <div className="wide">
          <dt>Filesystem</dt>
          <dd className={paths.length > 0 ? 'warn' : ''}>
            {paths.length === 0
              ? 'Inside the workspace only'
              : paths
                  .map((entry) => `${entry.path}${entry.write ? ' (write)' : ' (read)'}`)
                  .join(', ')}
          </dd>
        </div>
        <div>
          <dt>Network</dt>
          <dd className={elevation.requested_network_access ? 'warn' : ''}>
            {elevation.requested_network_access ? 'Outbound access requested' : 'Blocked'}
          </dd>
        </div>
        <div>
          <dt>Host execution</dt>
          <dd className={elevation.requested_host_execution ? 'warn' : ''}>
            {elevation.requested_host_execution ? 'Yes' : 'No'}
          </dd>
        </div>
      </dl>
      {!approval.sandbox.isolated && (
        <p className="agent-approval-caveat">
          <AlertTriangle size={13} /> {approval.sandbox.detail}
        </p>
      )}
      <input
        className="agent-approval-note"
        placeholder="Optional note for the agent (shown when you deny)…"
        value={note}
        onChange={(event) => setNote(event.target.value)}
      />
      <div className="agent-approval-actions">
        <button
          type="button"
          className="agent-deny"
          disabled={busy}
          onClick={() => onDecide(approval, 'deny', undefined, note.trim() || undefined)}
        >
          <X size={14} /> Deny
        </button>
        {approval.allow_session_scope && (
          <button
            type="button"
            className="agent-approve-session"
            disabled={busy}
            onClick={() => onDecide(approval, 'approve', 'session')}
            title={`Allow ${approval.scope_key} for the rest of this session`}
          >
            Allow for session
          </button>
        )}
        <button
          type="button"
          className="agent-approve"
          disabled={busy}
          onClick={() => onDecide(approval, 'approve', 'once')}
        >
          <Check size={14} /> Allow once
        </button>
      </div>
    </div>
  )
}

function TimelineRow({
  entry,
  onShowFull
}: {
  entry: TimelineEntry
  onShowFull: (artifactId: string) => void
}): React.JSX.Element {
  const statusLabel =
    entry.status === 'running'
      ? 'running'
      : entry.status === 'awaiting-approval'
        ? 'waiting for you'
        : entry.status === 'denied'
          ? 'refused'
          : entry.status
  return (
    <details className={`agent-tool ${entry.status}`}>
      <summary>
        {entry.status === 'running' ? (
          <LoaderCircle className="spin" size={13} />
        ) : entry.status === 'awaiting-approval' ? (
          <Lock size={13} />
        ) : entry.status === 'completed' ? (
          <Check size={13} />
        ) : (
          <X size={13} />
        )}
        <strong>{entry.tool}</strong>
        <span className="agent-tool-args">{argsPreview(entry.tool, entry.args)}</span>
        <span className={`agent-env ${entry.environment}`}>
          {entry.environment === 'host' ? 'host' : 'sandbox'}
        </span>
        <span className="agent-tool-status">{statusLabel}</span>
      </summary>
      <div className="agent-tool-body">
        {entry.sandbox && !entry.sandbox.isolated && entry.status !== 'awaiting-approval' && (
          <p className="agent-tool-caveat">
            <AlertTriangle size={12} /> Ran without OS isolation. {entry.sandbox.detail}
          </p>
        )}
        {entry.changedPaths && entry.changedPaths.length > 0 && (
          <p className="agent-tool-changed">Changed: {entry.changedPaths.join(', ')}</p>
        )}
        <pre>{entry.error ?? entry.output ?? '(no output yet)'}</pre>
        {entry.truncated && entry.artifactId && (
          <button type="button" onClick={() => onShowFull(entry.artifactId!)}>
            Show full output
          </button>
        )}
        <div className="agent-tool-meta">
          {typeof entry.exitCode === 'number' && <span>exit {entry.exitCode}</span>}
          {typeof entry.durationMs === 'number' && <span>{entry.durationMs} ms</span>}
        </div>
      </div>
    </details>
  )
}

export function AgentMode(props: Props): React.JSX.Element {
  const { onError, onSessionBound } = props
  const [capabilities, setCapabilities] = useState<AgentCapabilities | null>(null)
  const [tools, setTools] = useState<AgentToolCatalogEntry[]>([])
  const [sessions, setSessions] = useState<AgentSessionSummary[]>([])
  const [session, setSession] = useState<AgentSessionSummary | null>(null)
  const [messages, setMessages] = useState<AgentMessage[]>([])
  const [timeline, setTimeline] = useState<TimelineEntry[]>([])
  const [approvals, setApprovals] = useState<AgentApproval[]>([])
  const [grants, setGrants] = useState<string[]>([])
  const [streaming, setStreaming] = useState('')
  const [reasoning, setReasoning] = useState('')
  const [running, setRunning] = useState(false)
  const [summary, setSummary] = useState<AgentRunSummary | null>(null)
  const [pendingWorkspace, setPendingWorkspace] = useState<string | null>(null)
  const [deciding, setDeciding] = useState(false)
  const [modeMenuOpen, setModeMenuOpen] = useState(false)
  const [artifact, setArtifact] = useState<{ id: string; text: string } | null>(null)
  const [sessionListOpen, setSessionListOpen] = useState(false)
  const scrollAnchor = useRef<HTMLDivElement>(null)
  const sessionIdRef = useRef<string | null>(null)

  const workspace = session?.workspace_path ?? pendingWorkspace
  const permissionMode: AgentPermissionMode = session?.permission_mode ?? 'ask'
  const modelLabel = useMemo(() => {
    const model = props.models.find((candidate) => candidate.id === props.modelId)
    return props.modelId ? modelDisplayName(props.modelId, model).title : 'No model selected'
  }, [props.modelId, props.models])

  useEffect(() => {
    sessionIdRef.current = session?.id ?? null
  }, [session?.id])

  useEffect(() => {
    void fetchAgentCapabilities()
      .then(setCapabilities)
      .catch((cause: unknown) => onError(errorText(cause)))
    void fetchAgentTools().then(setTools).catch(() => setTools([]))
    void listAgentSessions().then(setSessions).catch(() => setSessions([]))
  }, [onError])

  useEffect(() => {
    scrollAnchor.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, streaming, timeline.length])

  /** Reduce one worker event into the view. */
  const applyEvent = useCallback((event: AgentEvent) => {
    switch (event.type) {
      case 'run-started': {
        setRunning(true)
        setSummary(null)
        setStreaming('')
        setReasoning('')
        return
      }
      case 'text-delta': {
        if (event.channel === 'reasoning') setReasoning((current) => current + event.delta)
        else setStreaming((current) => current + event.delta)
        return
      }
      case 'tool-started': {
        setTimeline((current) => [
          ...current.filter((entry) => entry.toolCallId !== event.toolCallId),
          {
            toolCallId: event.toolCallId,
            tool: event.tool,
            args: event.args,
            environment: event.environment,
            status: 'running',
            sandbox: event.sandbox
          }
        ])
        return
      }
      case 'approval-required': {
        setApprovals((current) =>
          current.some((entry) => entry.id === event.approval.id)
            ? current
            : [...current, event.approval]
        )
        setTimeline((current) =>
          current.map((entry) =>
            entry.toolCallId === event.toolCallId
              ? { ...entry, status: 'awaiting-approval', environment: event.approval.environment }
              : entry
          )
        )
        return
      }
      case 'tool-completed': {
        setTimeline((current) =>
          current.map((entry) =>
            entry.toolCallId === event.toolCallId
              ? {
                  ...entry,
                  status: 'completed',
                  environment: event.environment,
                  sandbox: event.sandbox,
                  output: event.output,
                  exitCode: event.exitCode,
                  changedPaths: event.changedPaths,
                  truncated: event.truncated,
                  artifactId: event.artifactId,
                  durationMs: event.durationMs
                }
              : entry
          )
        )
        return
      }
      case 'tool-failed': {
        setTimeline((current) =>
          current.map((entry) =>
            entry.toolCallId === event.toolCallId
              ? {
                  ...entry,
                  status: event.denied ? 'denied' : 'failed',
                  environment: event.environment,
                  sandbox: event.sandbox,
                  error: event.error,
                  durationMs: event.durationMs
                }
              : entry
          )
        )
        return
      }
      case 'message-committed': {
        // The composer already showed the user's own turn; echoing the
        // committed copy would duplicate it.
        if (event.message.role === 'user') return
        setMessages((current) => [...current, event.message])
        if (event.message.role === 'assistant') {
          setStreaming('')
          setReasoning('')
        }
        return
      }
      case 'run-completed': {
        setRunning(false)
        setSummary(event.summary)
        setStreaming('')
        return
      }
      case 'run-cancelled': {
        setRunning(false)
        setStreaming('')
        setApprovals([])
        return
      }
      case 'run-failed': {
        setRunning(false)
        setStreaming('')
        onError(event.error)
        return
      }
      case 'compacted': {
        setMessages((current) => [
          { role: 'system', text: event.state.summary, timestamp: event.state.compactedAt },
          ...current.slice(-6)
        ])
        return
      }
      default:
        return
    }
  }, [onError])

  useEffect(() => {
    const unsubscribe = window.brazier.agent.onMessage((message: WorkerMessage) => {
      if (message.type === 'event') {
        if (message.sessionId !== sessionIdRef.current) return
        applyEvent(message.event)
        return
      }
      if (message.type === 'log' && message.level === 'error') {
        onError(`Agent worker: ${message.message}`)
      }
    })
    return unsubscribe
  }, [applyEvent, onError])

  const loadSession = useCallback(
    async (id: string): Promise<void> => {
      try {
        const detail = await fetchAgentSession(id)
        setSession(detail.session)
        sessionIdRef.current = detail.session.id
        onSessionBound?.(detail.session.id)
        setMessages(detail.messages.map((record) => record.payload))
        setTimeline(timelineFromRecords(detail.tool_executions))
        setApprovals(detail.pending_approvals)
        setGrants(detail.grants)
        setSummary(null)
        setStreaming('')
        // Restoring never re-runs anything: the worker only rebuilds context.
        await window.brazier.agent.openSession(id)
      } catch (cause) {
        onError(errorText(cause))
      }
    },
    [onError, onSessionBound]
  )

  async function chooseWorkspace(): Promise<void> {
    onError(null)
    const selected = await window.brazier.selectWorkspace()
    if (!selected) return
    try {
      const validated = await validateAgentWorkspace(selected)
      if (session) {
        const updated = await updateAgentSession(session.id, { workspace_path: validated.path })
        setSession(updated)
      } else {
        setPendingWorkspace(validated.path)
      }
    } catch (cause) {
      onError(errorText(cause))
    }
  }

  async function changePermissionMode(mode: AgentPermissionMode): Promise<void> {
    setModeMenuOpen(false)
    if (!session) return
    try {
      const updated = await updateAgentSession(session.id, { permission_mode: mode })
      setSession(updated)
      setSessions((current) =>
        current.map((entry) => (entry.id === updated.id ? updated : entry))
      )
    } catch (cause) {
      onError(errorText(cause))
    }
  }

  async function decide(
    approval: AgentApproval,
    decision: 'approve' | 'deny',
    scope?: 'once' | 'session',
    note?: string
  ): Promise<void> {
    setDeciding(true)
    try {
      await decideAgentApproval(approval.id, decision, scope, note)
      setApprovals((current) => current.filter((entry) => entry.id !== approval.id))
      if (decision === 'approve' && scope === 'session') {
        setGrants((current) => [...current, `${approval.environment}:${approval.scope_key}`])
      }
    } catch (cause) {
      onError(errorText(cause))
    } finally {
      setDeciding(false)
    }
  }

  async function send(input: string): Promise<void> {
    const text = input.trim()
    if (!text || running) return
    onError(null)
    if (!props.modelId) {
      onError('Choose a model in the top bar before starting an agent task.')
      return
    }
    if (!workspace) {
      onError('Choose a workspace folder first. The agent works inside it.')
      return
    }
    try {
      let active = session
      if (!active) {
        active = await createAgentSession({
          title: text.slice(0, 60),
          workspace_path: workspace,
          model: props.modelId
        })
        setSession(active)
        sessionIdRef.current = active.id
        setSessions((current) => [active as AgentSessionSummary, ...current])
        setPendingWorkspace(null)
        onSessionBound?.(active.id)
      } else if (active.model !== props.modelId) {
        // Model changes only take effect between runs.
        await window.brazier.agent.setModel(active.id, { id: props.modelId })
        const updated = await updateAgentSession(active.id, { model: props.modelId })
        setSession(updated)
        active = updated
      }
      setMessages((current) => [
        ...current,
        { role: 'user', text, timestamp: new Date().toISOString() }
      ])
      setRunning(true)
      await window.brazier.agent.run(active.id, { text })
    } catch (cause) {
      setRunning(false)
      onError(errorText(cause))
    } finally {
      // The daemon owns the record of what happened; re-read it so the timeline
      // matches the ledger even if an event was missed.
      if (sessionIdRef.current) {
        void fetchAgentSession(sessionIdRef.current)
          .then((detail) => {
            setTimeline(timelineFromRecords(detail.tool_executions))
            setGrants(detail.grants)
            setSession(detail.session)
          })
          .catch(() => undefined)
      }
    }
  }

  async function stop(): Promise<void> {
    if (!session) return
    try {
      await window.brazier.agent.cancel(session.id)
      setRunning(false)
      setApprovals([])
    } catch (cause) {
      onError(errorText(cause))
    }
  }

  async function compact(): Promise<void> {
    if (!session) return
    try {
      await window.brazier.agent.compact(session.id)
    } catch (cause) {
      onError(errorText(cause))
    }
  }

  async function startNewTask(): Promise<void> {
    if (session) {
      await window.brazier.agent.closeSession(session.id).catch(() => undefined)
    }
    setSession(null)
    sessionIdRef.current = null
    // Nothing is bound until the next task exists, so a voice turn falls back
    // to an ordinary chat answer rather than reaching a closed session.
    onSessionBound?.(null)
    setMessages([])
    setTimeline([])
    setApprovals([])
    setGrants([])
    setSummary(null)
    setStreaming('')
    setReasoning('')
  }

  async function removeSession(id: string): Promise<void> {
    try {
      await deleteAgentSession(id)
      setSessions((current) => current.filter((entry) => entry.id !== id))
      if (session?.id === id) await startNewTask()
    } catch (cause) {
      onError(errorText(cause))
    }
  }

  async function showArtifact(artifactId: string): Promise<void> {
    try {
      const text = await fetchAgentArtifact(artifactId)
      setArtifact({ id: artifactId, text })
    } catch (cause) {
      onError(errorText(cause))
    }
  }

  const sandbox = capabilities?.sandbox
  const executeTools = tools.filter((tool) => tool.executes).length

  // Keep the shared composer in step with what the agent can currently do. The
  // callbacks are re-published on every relevant change rather than held in a
  // ref, so the composer never sends against stale session or model state.
  const { onComposerChange } = props
  const blockedReason = !props.modelId
    ? 'Choose a model in the top bar…'
    : !workspace
      ? 'Choose a workspace folder…'
      : ''
  useEffect(() => {
    onComposerChange?.({
      send,
      stop,
      running,
      blockedReason,
      placeholder: blockedReason
        ? blockedReason
        : running
          ? 'The agent is working. Stop it to send something else…'
          : `Ask ${modelLabel} to do something in ${shortPath(workspace)}…`
    })
    return () => onComposerChange?.(null)
    // `send` and `stop` close over session state, so they are re-created each
    // render; the primitives below are what actually decide a new publication.
  }, [onComposerChange, running, blockedReason, modelLabel, workspace, session?.id, props.modelId])

  return (
    <div className="agent-mode">
      <header className="agent-header">
        <button className="agent-workspace" type="button" onClick={() => void chooseWorkspace()}>
          <FolderOpen size={15} />
          <span>
            <strong>{shortPath(workspace)}</strong>
            <small>{workspace ? 'Workspace' : 'Choose a folder'}</small>
          </span>
        </button>
        {sandbox && <SandboxBadge sandbox={sandbox} />}
        <div className="agent-mode-select">
          <button type="button" onClick={() => setModeMenuOpen((open) => !open)} disabled={!session}>
            <ShieldCheck size={14} />
            {PERMISSION_LABELS[permissionMode].title}
            <ChevronDown size={13} />
          </button>
          {modeMenuOpen && (
            <div className="agent-mode-menu">
              {(Object.keys(PERMISSION_LABELS) as AgentPermissionMode[]).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  className={mode === permissionMode ? 'active' : ''}
                  onClick={() => void changePermissionMode(mode)}
                >
                  <strong>{PERMISSION_LABELS[mode].title}</strong>
                  <span>{PERMISSION_LABELS[mode].detail}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="agent-header-spacer" />
        {grants.length > 0 && (
          <span className="agent-grants" title={grants.join('\n')}>
            {grants.length} standing grant{grants.length === 1 ? '' : 's'}
          </span>
        )}
        <button
          className="chip-button subtle"
          type="button"
          onClick={() => setSessionListOpen((open) => !open)}
        >
          <Layers size={13} /> Tasks
        </button>
        <button className="chip-button subtle" type="button" onClick={() => void startNewTask()}>
          New task
        </button>
        <button
          className="chip-button subtle"
          type="button"
          disabled={!session || running}
          title="Summarize earlier turns to free up context"
          onClick={() => void compact()}
        >
          Compact
        </button>
      </header>

      {sessionListOpen && (
        <div className="agent-session-list">
          {sessions.length === 0 && <p>No agent tasks yet.</p>}
          {sessions.map((entry) => (
            <div className={entry.id === session?.id ? 'agent-session active' : 'agent-session'} key={entry.id}>
              <button type="button" onClick={() => void loadSession(entry.id)}>
                <strong>{entry.title}</strong>
                <span>
                  {shortPath(entry.workspace_path)} · {entry.last_run_status}
                </span>
              </button>
              <button
                type="button"
                className="agent-session-delete"
                title="Delete this task"
                onClick={() => void removeSession(entry.id)}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
      )}

      {sandbox && !sandbox.isolated && (
        <div className="agent-warning">
          <AlertTriangle size={15} />
          <span>
            No sandbox on this host: {sandbox.detail} Commands would run with your full privileges,
            so each one is held for approval and refused in sandbox-only mode.
          </span>
        </div>
      )}

      <div className="agent-transcript">
        {messages.length === 0 && !streaming && (
          <div className="agent-empty">
            <div className="agent-empty-mark">
              <Terminal size={26} />
            </div>
            <h2>Give the agent a task</h2>
            <p>
              It reads and edits files in the workspace and runs commands there. Everything runs
              through Brazier's own policy layer: {executeTools} of {tools.length} tools can execute
              programs, and each needs your approval unless you change the mode above.
            </p>
            <div className="agent-suggestions">
              <button type="button" onClick={() => props.onSuggestPrompt?.('Summarize this repository: layout, build commands, and test entry points.')}>
                Explore the repository
              </button>
              <button type="button" onClick={() => props.onSuggestPrompt?.('Run the test suite and report what fails.')}>
                Run the tests
              </button>
              <button type="button" onClick={() => props.onSuggestPrompt?.('Show me the uncommitted changes and explain them.')}>
                Review my changes
              </button>
            </div>
          </div>
        )}

        {messages.map((message, index) => {
          if (message.role === 'tool') return null
          if (message.role === 'system') {
            return (
              <article className="agent-message system" key={`system-${index}`}>
                <Layers size={14} />
                <div>
                  <strong>Context compacted</strong>
                  <pre>{message.text}</pre>
                </div>
              </article>
            )
          }
          return (
            <article className={`agent-message ${message.role}`} key={`${message.role}-${index}`}>
              <div className="avatar">{message.role === 'assistant' ? <Bot size={16} /> : 'You'}</div>
              <div className="agent-message-body">
                {message.role === 'assistant' && message.reasoning && (
                  <details className="agent-reasoning">
                    <summary>Reasoning</summary>
                    <pre>{message.reasoning}</pre>
                  </details>
                )}
                <p>{message.text}</p>
                {message.role === 'assistant' && message.error && (
                  <p className="agent-message-error">{message.error}</p>
                )}
              </div>
            </article>
          )
        })}

        {timeline.length > 0 && (
          <section className="agent-timeline">
            <div className="section-label">
              <Wrench size={12} /> Activity
            </div>
            {timeline.map((entry) => (
              <TimelineRow key={entry.toolCallId} entry={entry} onShowFull={(id) => void showArtifact(id)} />
            ))}
          </section>
        )}

        {approvals.map((approval) => (
          <ApprovalCard key={approval.id} approval={approval} onDecide={(...args) => void decide(...args)} busy={deciding} />
        ))}

        {(streaming || reasoning) && (
          <article className="agent-message assistant">
            <div className="avatar">
              <Bot size={16} />
            </div>
            <div className="agent-message-body">
              {reasoning && (
                <details className="agent-reasoning" open>
                  <summary>Reasoning</summary>
                  <pre>{reasoning}</pre>
                </details>
              )}
              <p>{streaming}</p>
            </div>
          </article>
        )}

        {summary && (
          <section className="agent-summary">
            <div className="section-label">Run summary</div>
            <ul>
              <li>{summary.toolCalls} tool call{summary.toolCalls === 1 ? '' : 's'}</li>
              {summary.filesChanged.length > 0 && (
                <li>Files changed: {summary.filesChanged.join(', ')}</li>
              )}
              {summary.commandsRun.length > 0 && (
                <li>
                  Commands: {summary.commandsRun.map((command) => <code key={command}>{command}</code>)}
                </li>
              )}
              {summary.approvalsRequested > 0 && (
                <li>{summary.approvalsRequested} approval request(s)</li>
              )}
              {summary.hostActions.length > 0 && (
                <li className="warn">Outside the sandbox: {summary.hostActions.join(', ')}</li>
              )}
              {summary.failures.length > 0 ? (
                <li className="warn">Failures: {summary.failures.join('; ')}</li>
              ) : (
                <li>No failures reported</li>
              )}
            </ul>
          </section>
        )}
        <div ref={scrollAnchor} />
      </div>

      {/* No composer here: the window has one, at the bottom, for every mode. */}

      {artifact && (
        <div className="agent-artifact-overlay" role="dialog">
          <div className="agent-artifact">
            <header>
              <strong>Full tool output</strong>
              <button type="button" onClick={() => setArtifact(null)}>
                <X size={15} />
              </button>
            </header>
            <pre>{artifact.text}</pre>
          </div>
        </div>
      )}
    </div>
  )
}

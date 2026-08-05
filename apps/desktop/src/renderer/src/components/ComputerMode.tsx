import {
  Check,
  LoaderCircle,
  Monitor,
  X
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { parseFaraOutput as parseFaraLocal } from '../../../computer/faraAdapter'
import {
  appendComputerStep,
  computerExec,
  computerScreenshot,
  stopComputerSession,
  setComputerSafetyAuthority,
  createComputerSession,
  decideComputerApproval,
  deleteComputerSession,
  listComputerSessions,
  listComputerSteps,
  parseFaraOutput as parseFaraRemote,
  streamCompletion,
  type ComputerAction,
  type ComputerActionResult,
  type ComputerPermissionMode,
  type ComputerSession,
  type ComputerStep,
  type ComputerTarget,
  type LocalModel
} from '../api'
import { modelDisplayName } from '../model-utils'
import type { AgentComposerControls } from './AgentMode'
import {
  buildComputerHistory,
  computerActionLabel,
  computerModelOutput,
  computerScreenshotDataUrl,
  continuationForResult,
  observationError,
  recoverComputerPause,
  type ComputerContinuation
} from './computerHistory'

/**
 * Session list for the app sidebar while Computer Use is active. Replaces the
 * conversation list so sessions live where chats normally do.
 */
export type ComputerSidebarControls = {
  sessions: ComputerSession[]
  activeId: string | null
  select: (id: string) => void
  remove: (id: string) => void
  newSession: () => void
}

type Props = {
  modelId: string
  models: LocalModel[]
  onError: (message: string | null) => void
  onComposerChange?: (controls: AgentComposerControls | null) => void
  /** Publish the session list for the app sidebar; null on unmount. */
  onSidebarChange?: (controls: ComputerSidebarControls | null) => void
}

const MAX_LOOP_STEPS = 20

const PERMISSION_LABELS: Record<
  ComputerPermissionMode,
  { title: string; detail: string }
> = {
  ask: {
    title: 'Ask first',
    detail: 'Approve navigate, type, click, and other interactive actions.'
  },
  'browser-only': {
    title: 'Browser only',
    detail: 'Host desktop control is refused. Browser actions still prompt as needed.'
  },
  'skip-permissions': {
    title: 'Skip low-risk',
    detail: 'Low-risk browser actions run without prompts. Sensitive ones still ask.'
  },
  'allow-all': {
    title: 'Allow all',
    detail: 'Run all supported actions without approval. The always-visible emergency shortcut remains active.'
  }
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

export function ComputerMode(props: Props): React.JSX.Element {
  const { modelId, models, onError, onComposerChange, onSidebarChange } = props
  const modelLabel = useMemo(() => {
    const model = models.find((entry) => entry.id === modelId)
    return modelDisplayName(modelId, model).title
  }, [modelId, models])

  const [sessions, setSessions] = useState<ComputerSession[]>([])
  const [session, setSession] = useState<ComputerSession | null>(null)
  const [steps, setSteps] = useState<ComputerStep[]>([])
  const [viewportUrl, setViewportUrl] = useState<string | null>(null)
  const [target, setTarget] = useState<ComputerTarget>('browser')
  const [permissionMode, setPermissionMode] = useState<ComputerPermissionMode>('ask')
  const [running, setRunning] = useState(false)
  const [pendingApproval, setPendingApproval] = useState<{
    approvalId: string
    action: ComputerAction
    message?: string | null
  } | null>(null)
  const [pendingUserQuestion, setPendingUserQuestion] = useState<string | null>(null)
  const [liveThought, setLiveThought] = useState('')
  const abortRef = useRef<AbortController | null>(null)
  const sessionRef = useRef<ComputerSession | null>(null)
  sessionRef.current = session

  const refreshSessions = useCallback(async (): Promise<ComputerSession[]> => {
    const list = await listComputerSessions()
    setSessions(list)
    return list
  }, [])

  const loadSession = useCallback(
    async (id: string): Promise<void> => {
      onError(null)
      try {
        const list = await refreshSessions()
        const found = list.find((entry) => entry.id === id) ?? null
        setSession(found)
        sessionRef.current = found
        if (!found) {
          setSteps([])
          setPendingApproval(null)
          setPendingUserQuestion(null)
          return
        }
        setTarget(found.target)
        setPermissionMode(found.permission_mode)
        const nextSteps = await listComputerSteps(id)
        setSteps(nextSteps)
        const lastShot = [...nextSteps]
          .reverse()
          .map((step) => computerScreenshotDataUrl(step.result))
          .find(Boolean)
        setViewportUrl(lastShot ?? null)
        const recovered = recoverComputerPause(nextSteps)
        setPendingApproval(recovered.approval)
        setPendingUserQuestion(recovered.userQuestion)
      } catch (cause) {
        onError(errorText(cause))
      }
    },
    [onError, refreshSessions]
  )

  useEffect(() => {
    void refreshSessions().catch((cause) => onError(errorText(cause)))
  }, [refreshSessions, onError])

  async function createSession(): Promise<ComputerSession> {
    const created = await createComputerSession({
      title: 'Computer task',
      target,
      model_id: modelId || null,
      permission_mode: permissionMode
    })
    setSession(created)
    sessionRef.current = created
    setSteps([])
    setPendingApproval(null)
    setPendingUserQuestion(null)
    setViewportUrl(null)
    await refreshSessions()
    return created
  }

  async function removeSession(id: string): Promise<void> {
    try {
      await deleteComputerSession(id)
      if (sessionRef.current?.id === id) {
        setSession(null)
        sessionRef.current = null
        setSteps([])
        setViewportUrl(null)
        setPendingApproval(null)
        setPendingUserQuestion(null)
      }
      await refreshSessions()
    } catch (cause) {
      onError(errorText(cause))
    }
  }

  async function refreshViewport(sessionId: string): Promise<ComputerActionResult | null> {
    try {
      const shot = await computerScreenshot(sessionId)
      const url = computerScreenshotDataUrl(shot)
      if (url) setViewportUrl(url)
      return shot
    } catch {
      return null
    }
  }

  async function parseModelOutput(text: string): Promise<{
    thought: string | null
    actions: ComputerAction[]
  }> {
    try {
      const remote = await parseFaraRemote(text)
      return {
        thought: remote.thought ?? null,
        actions: remote.actions ?? []
      }
    } catch {
      const local = parseFaraLocal(text)
      return { thought: local.thought, actions: local.actions as ComputerAction[] }
    }
  }

  const stop = useCallback(async (): Promise<void> => {
    abortRef.current?.abort()
    abortRef.current = null
    // Revoke the native overlay/input guard before waiting on the daemon. This
    // remains immediate even if the HTTP service is wedged or has crashed.
    await window.brazier.computer.setActive(false)
    const active = sessionRef.current
    if (active?.target === 'desktop') {
      try {
        await stopComputerSession(active.id)
      } catch {
        // Escape remains a local emergency stop even if the daemon vanished.
      }
    }
    setRunning(false)
  }, [])

  async function syncSteps(sessionId: string): Promise<ComputerStep[]> {
    const next = await listComputerSteps(sessionId)
    setSteps(next)
    return next
  }

  async function runModelLoop(active: ComputerSession, controller: AbortController): Promise<void> {
    // This screenshot is a broker-recorded action, so it survives reload and is
    // visible to the next model turn through buildComputerHistory().
    const observation = await refreshViewport(active.id)
    const observationFailure = observationError(observation)
    if (observationFailure) throw new Error(observationFailure)
    await syncSteps(active.id)
    const activeModelId = active.model_id || modelId

    for (let step = 0; step < MAX_LOOP_STEPS && !controller.signal.aborted; step += 1) {
      const history = buildComputerHistory(active, await syncSteps(active.id))
      let responseText = ''
      setLiveThought('')
      const completion = await streamCompletion(
        history,
        activeModelId,
        controller.signal,
        (token) => {
          responseText += token
        },
        {
          onReasoning: (token) => setLiveThought((current) => current + token),
          toolChoice: 'none',
          // Fara emits its XML action after a think block. With thinking
          // enabled llama.cpp can classify the entire response as reasoning,
          // leaving the action parser an empty completion.
          enableReasoning: false
        }
      )
      responseText = computerModelOutput(completion.responseText || responseText, completion.reasoningText)
      if (controller.signal.aborted) return

      const parsed = await parseModelOutput(responseText)
      if (parsed.thought) setLiveThought(parsed.thought)
      // The model's complete raw reply is narrative context.  Only the broker
      // writes action/result records, preventing duplicate tool timeline rows.
      await appendComputerStep(active.id, {
        role: 'assistant',
        content: responseText,
        thought: parsed.thought
      })

      if (parsed.actions.length === 0) {
        onError('The computer-use model returned no executable action. Try again or choose another model.')
        return
      }
      for (const action of parsed.actions) {
        if (controller.signal.aborted) return
        const result = await computerExec({ session_id: active.id, action })
        const shotUrl = computerScreenshotDataUrl(result)
        if (shotUrl) setViewportUrl(shotUrl)
        await syncSteps(active.id)

        // Fara is browser-trained and can occasionally emit a navigation
        // action even after the desktop-only signature. Preserve that failed
        // action in history, then give it a fresh turn to use the visible GUI
        // rather than treating the whole task as irrecoverably blocked.
        if (active.target === 'desktop' && (action.type === 'visit_url' || action.type === 'web_search')) {
          await appendComputerStep(active.id, {
            role: 'user',
            content: 'That action is unavailable on this desktop. Continue by interacting with the visible desktop GUI using mouse and keyboard only.'
          })
          break
        }

        const continuation: ComputerContinuation = continuationForResult(result)
        if (result.needs_approval || result.status === 'needs_approval') {
          const approvalId = result.approval_id
          if (!approvalId) {
            onError(result.message || 'Action needs approval, but no approval id was returned.')
            return
          }
          setPendingApproval({ approvalId, action, message: result.message })
          return
        }
        if (continuation.kind === 'waiting_for_user') {
          setPendingUserQuestion(continuation.question)
          return
        }
        if (continuation.kind === 'finished') return
        if (continuation.kind === 'blocked') {
          onError(result.message || `Action ${action.type} ${result.status}.`)
          return
        }
      }
    }
  }

  async function send(text: string): Promise<void> {
    const userText = text.trim()
    if (!userText || running || pendingApproval) return
    if (!modelId && !sessionRef.current?.model_id) {
      onError('Choose a computer-use model in the top bar.')
      return
    }

    onError(null)
    setRunning(true)
    setPendingUserQuestion(null)
    setLiveThought('')
    const controller = new AbortController()
    abortRef.current = controller
    try {
      let active = sessionRef.current
      if (!active) {
        active = await createComputerSession({
          title: userText.slice(0, 72) || 'Computer task',
          target,
          model_id: modelId,
          permission_mode: permissionMode
        })
        setSession(active)
        sessionRef.current = active
        await refreshSessions()
      }
      if (active.target === 'desktop') {
        // Fail closed: no screenshot, model turn, approval, or OS action may
        // begin until both the native overlay and Esc watcher report READY.
        await window.brazier.computer.setActive(true)
        try {
          await setComputerSafetyAuthority(active.id, true)
        } catch (cause) {
          await window.brazier.computer.setActive(false)
          throw cause
        }
      }
      // User prompts (including answers to ask_user and ordinary follow-ups)
      // are durable and become part of every resumed/reloaded history.
      await appendComputerStep(active.id, { role: 'user', content: userText })
      await runModelLoop(active, controller)
    } catch (cause) {
      if ((cause as Error).name !== 'AbortError') onError(errorText(cause))
    } finally {
      const active = sessionRef.current
      if (active?.target === 'desktop') {
        void setComputerSafetyAuthority(active.id, false).catch(() => undefined)
      }
      void window.brazier.computer.setActive(false)
      setRunning(false)
      if (abortRef.current === controller) abortRef.current = null
      setLiveThought('')
      const id = sessionRef.current?.id
      if (id) void syncSteps(id).catch(() => undefined)
    }
  }

  async function resolveApproval(approve: boolean): Promise<void> {
    if (!pendingApproval || !session) return
    onError(null)
    let safetyActive = false
    try {
      // Approval executes the pending action inside the broker call, so the
      // safety boundary must be established before sending the decision.
      if (approve && session.target === 'desktop') {
        await window.brazier.computer.setActive(true)
        try {
          await setComputerSafetyAuthority(session.id, true)
        } catch (cause) {
          await window.brazier.computer.setActive(false)
          throw cause
        }
        safetyActive = true
      }
      const { result } = await decideComputerApproval(pendingApproval.approvalId, approve)
      setPendingApproval(null)
      if (result) {
        const shotUrl = computerScreenshotDataUrl(result)
        if (shotUrl) setViewportUrl(shotUrl)
        await syncSteps(session.id)
      }
      if (approve && result && continuationForResult(result).kind === 'model') {
        setRunning(true)
        const controller = new AbortController()
        abortRef.current = controller
        try {
          // The broker result is already in persisted history; continue the
          // same task without inventing a contextless user prompt.
          await runModelLoop(session, controller)
        } finally {
          setRunning(false)
          if (abortRef.current === controller) abortRef.current = null
          setLiveThought('')
          void syncSteps(session.id).catch(() => undefined)
        }
      }
    } catch (cause) {
      onError(errorText(cause))
    } finally {
      if (safetyActive) {
        void setComputerSafetyAuthority(session.id, false).catch(() => undefined)
        void window.brazier.computer.setActive(false)
      }
    }
  }

  const blockedReason = !modelId && !session?.model_id ? 'Choose a computer-use model in the top bar…' : ''
  const composerPlaceholder =
    blockedReason ||
    (running
      ? 'Computer Use is working. Stop it to send something else…'
      : pendingApproval
        ? 'Approve or deny the pending action before continuing…'
        : pendingUserQuestion
          ? `Answer the agent: ${pendingUserQuestion}`
          : `Ask ${modelLabel} to use the ${target}…`)
  const composerActionsRef = useRef({ send, stop })
  composerActionsRef.current = { send, stop }

  useEffect(() => {
    onComposerChange?.({
      send: (text) => composerActionsRef.current.send(text),
      stop: () => composerActionsRef.current.stop(),
      running,
      blockedReason,
      placeholder: composerPlaceholder
    })
    return () => onComposerChange?.(null)
  }, [
    onComposerChange,
    running,
    blockedReason,
    composerPlaceholder,
    modelLabel,
    target,
    pendingApproval,
    pendingUserQuestion
  ])

  useEffect(() => {
    return () => {
      abortRef.current?.abort()
      // setActive(false) clears the overlay marker and revokes daemon desktop
      // authority from main, even when this view unmounts mid-session.
      void window.brazier.computer.setActive(false)
    }
  }, [])

  // Same pattern for the app sidebar: Computer mode replaces conversations
  // with its session list, so the list and its actions live up there.
  const sidebarActionsRef = useRef({ loadSession, removeSession, createSession })
  sidebarActionsRef.current = { loadSession, removeSession, createSession }
  useEffect(() => {
    onSidebarChange?.({
      sessions,
      activeId: session?.id ?? null,
      select: (id) => void sidebarActionsRef.current.loadSession(id),
      remove: (id) => void sidebarActionsRef.current.removeSession(id),
      newSession: () =>
        void sidebarActionsRef.current.createSession().catch((cause) => onError(errorText(cause)))
    })
    return () => onSidebarChange?.(null)
  }, [onSidebarChange, sessions, session?.id, onError])

  useEffect(() => window.brazier.computer.onEscape(() => { void stop() }), [stop])

  return (
    <div className="computer-mode">
      <header className="computer-header">
        <div className="computer-controls">
          <label>
            <span>Target</span>
            <select
              value={target}
              disabled={running || Boolean(session)}
              onChange={(event) => setTarget(event.target.value as ComputerTarget)}
            >
              <option value="browser">Browser</option>
              <option value="desktop">Desktop</option>
            </select>
          </label>
          <label>
            <span>Permissions</span>
            <select
              value={permissionMode}
              disabled={running || Boolean(session)}
              title={PERMISSION_LABELS[permissionMode].detail}
              onChange={(event) =>
                setPermissionMode(event.target.value as ComputerPermissionMode)
              }
            >
              {(Object.keys(PERMISSION_LABELS) as ComputerPermissionMode[]).map((mode) => (
                <option key={mode} value={mode}>
                  {PERMISSION_LABELS[mode].title}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      <div className="computer-body">
        <div className="computer-viewport">
          {viewportUrl ? (
            <img src={viewportUrl} alt="Computer Use viewport" />
          ) : (
            <div className="computer-viewport-empty">
              <Monitor size={28} />
              <p>Viewport preview appears after the first screenshot.</p>
            </div>
          )}
          {running && (
            <div className="computer-viewport-busy">
              <LoaderCircle className="spin" size={16} />
              Working…
            </div>
          )}
        </div>

        <div className="computer-steps">
          {pendingApproval && (
            <div className="computer-approval">
              <div>
                <strong>Approval needed</strong>
                <p>{pendingApproval.message || computerActionLabel(pendingApproval.action)}</p>
              </div>
              <div className="computer-approval-actions">
                <button type="button" onClick={() => void resolveApproval(false)}>
                  <X size={14} />
                  Deny
                </button>
                <button
                  type="button"
                  className="primary"
                  onClick={() => void resolveApproval(true)}
                >
                  <Check size={14} />
                  Approve
                </button>
              </div>
            </div>
          )}

          {pendingUserQuestion && !pendingApproval && (
            <div className="computer-approval">
              <div>
                <strong>Information needed</strong>
                <p>{pendingUserQuestion}</p>
              </div>
            </div>
          )}

          {liveThought && running && (
            <div className="computer-thought live">
              <strong>Thinking</strong>
              <pre>{liveThought}</pre>
            </div>
          )}

          {steps.length === 0 && !running ? (
            <div className="computer-steps-empty">
              <p>
                Describe a task below. Computer Use will screenshot the {target}, ask the model,
                and run the returned actions.
              </p>
            </div>
          ) : (
            steps.map((step) => (
              <article key={step.id} className={`computer-step role-${step.role}`}>
                <header>
                  <strong>{step.role}</strong>
                  {step.action ? <span>{computerActionLabel(step.action)}</span> : null}
                </header>
                {step.thought ? (
                  <div className="computer-thought">
                    <pre>{step.thought}</pre>
                  </div>
                ) : null}
                {step.content && step.content !== computerActionLabel(step.action) ? (
                  <p>{step.content}</p>
                ) : step.content && !step.action ? (
                  <p>{step.content}</p>
                ) : null}
                {step.result?.status ? (
                  <small className={`computer-step-status status-${step.result.status}`}>
                    {step.result.status}
                    {step.result.message ? ` · ${step.result.message}` : ''}
                  </small>
                ) : null}
              </article>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

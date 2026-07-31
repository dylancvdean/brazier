import {
  Check,
  LoaderCircle,
  Monitor,
  Plus,
  Trash2,
  X
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { parseFaraOutput as parseFaraLocal } from '../../../computer/faraAdapter'
import {
  appendComputerStep,
  computerExec,
  computerScreenshot,
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
import type { ContentPart, Message } from '../types'

type Props = {
  modelId: string
  models: LocalModel[]
  onError: (message: string | null) => void
  onComposerChange?: (controls: AgentComposerControls | null) => void
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
  }
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

function actionLabel(action: ComputerAction | null | undefined): string {
  if (!action) return '—'
  switch (action.type) {
    case 'visit_url':
      return `Visit ${action.url}`
    case 'web_search':
      return `Search “${action.query}”`
    case 'type':
      return `Type “${action.text.slice(0, 48)}${action.text.length > 48 ? '…' : ''}”`
    case 'left_click':
    case 'right_click':
    case 'double_click':
    case 'triple_click':
    case 'mouse_move':
      return `${action.type.replaceAll('_', ' ')} (${Math.round(action.x)}, ${Math.round(action.y)})`
    case 'keypress':
      return `Keys ${action.keys.join('+')}`
    case 'ask_user':
      return `Ask: ${action.question}`
    case 'terminate':
      return action.response ? `Done — ${action.response}` : 'Terminate'
    case 'memorize':
      return `Remember: ${action.fact}`
    default:
      return action.type.replaceAll('_', ' ')
  }
}

function screenshotDataUrl(result: ComputerActionResult | null | undefined): string | null {
  if (!result?.screenshot_base64) return null
  const mime = result.mime_type || 'image/png'
  return `data:${mime};base64,${result.screenshot_base64}`
}

function ephemeralMessage(
  role: Message['role'],
  content: string | ContentPart[]
): Message {
  return {
    id: `computer-${role}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    conversation_id: 'computer',
    parent_id: null,
    role,
    content,
    model: null,
    created_at: new Date().toISOString()
  }
}

export function ComputerMode(props: Props): React.JSX.Element {
  const { modelId, models, onError, onComposerChange } = props
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
        if (!found) {
          setSteps([])
          return
        }
        setTarget(found.target)
        setPermissionMode(found.permission_mode)
        const nextSteps = await listComputerSteps(id)
        setSteps(nextSteps)
        const lastShot = [...nextSteps]
          .reverse()
          .map((step) => screenshotDataUrl(step.result))
          .find(Boolean)
        if (lastShot) setViewportUrl(lastShot)
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
    setSteps([])
    setPendingApproval(null)
    setViewportUrl(null)
    await refreshSessions()
    return created
  }

  async function removeSession(id: string): Promise<void> {
    try {
      await deleteComputerSession(id)
      if (sessionRef.current?.id === id) {
        setSession(null)
        setSteps([])
        setViewportUrl(null)
        setPendingApproval(null)
      }
      await refreshSessions()
    } catch (cause) {
      onError(errorText(cause))
    }
  }

  async function refreshViewport(sessionId: string): Promise<ComputerActionResult | null> {
    try {
      const shot = await computerScreenshot(sessionId)
      const url = screenshotDataUrl(shot)
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
    setRunning(false)
  }, [])

  const send = useCallback(
    async (text: string): Promise<void> => {
      const goal = text.trim()
      if (!goal || running) return
      if (!modelId) {
        onError('Choose a computer-use model in the top bar.')
        return
      }

      onError(null)
      setRunning(true)
      setPendingApproval(null)
      setLiveThought('')
      const controller = new AbortController()
      abortRef.current = controller

      try {
        let active = sessionRef.current
        if (!active) {
          active = await createComputerSession({
            title: goal.slice(0, 72) || 'Computer task',
            target,
            model_id: modelId,
            permission_mode: permissionMode
          })
          setSession(active)
          await refreshSessions()
        }

        await appendComputerStep(active.id, { role: 'user', content: goal })
        let nextSteps = await listComputerSteps(active.id)
        setSteps(nextSteps)

        let shot = await refreshViewport(active.id)
        const history: Message[] = [
          ephemeralMessage(
            'system',
            [
              'You are a computer-use agent. Observe the screenshot and act via Fara tool calls.',
              'Emit <tool_call>{"name":"computer_use","arguments":{...}}</tool_call> with one action.',
              'Available actions include screenshot, left_click, type, keypress, scroll, visit_url,',
              'web_search, wait, ask_user, memorize, and terminate.',
              `Target: ${active.target}. Permission mode: ${active.permission_mode}.`
            ].join(' ')
          )
        ]

        for (let step = 0; step < MAX_LOOP_STEPS; step += 1) {
          if (controller.signal.aborted) break

          const imageUrl = screenshotDataUrl(shot)
          const userParts: ContentPart[] = [{ type: 'text', text: step === 0 ? goal : 'Continue.' }]
          if (imageUrl) {
            userParts.push({ type: 'image_url', image_url: { url: imageUrl } })
          }
          history.push(ephemeralMessage('user', userParts))

          let responseText = ''
          setLiveThought('')
          const result = await streamCompletion(
            history,
            modelId,
            controller.signal,
            (token) => {
              responseText += token
            },
            {
              onReasoning: (token) => {
                setLiveThought((current) => current + token)
              },
              toolChoice: 'none'
            }
          )
          responseText = result.responseText || responseText
          history.push(ephemeralMessage('assistant', responseText))

          const parsed = await parseModelOutput(responseText)
          if (parsed.thought) setLiveThought(parsed.thought)

          if (parsed.actions.length === 0) {
            await appendComputerStep(active.id, {
              role: 'assistant',
              content: responseText,
              thought: parsed.thought
            })
            nextSteps = await listComputerSteps(active.id)
            setSteps(nextSteps)
            break
          }

          let stopLoop = false
          for (const action of parsed.actions) {
            if (controller.signal.aborted) {
              stopLoop = true
              break
            }

            const recorded = await appendComputerStep(active.id, {
              role: 'assistant',
              content: actionLabel(action),
              thought: parsed.thought,
              action
            })

            if (action.type === 'ask_user') {
              await appendComputerStep(active.id, {
                role: 'assistant',
                content: action.question,
                thought: parsed.thought,
                action,
                result: {
                  status: 'waiting_for_user',
                  message: action.question
                }
              })
              stopLoop = true
              break
            }

            if (action.type === 'terminate') {
              await appendComputerStep(active.id, {
                role: 'assistant',
                content: action.response || 'Task finished.',
                thought: parsed.thought,
                action,
                result: {
                  status: 'finished',
                  message: action.response || 'finished'
                }
              })
              stopLoop = true
              break
            }

            let execResult = await computerExec({
              session_id: active.id,
              action
            })

            if (execResult.needs_approval || execResult.status === 'needs_approval') {
              const approvalId = execResult.approval_id
              if (!approvalId) {
                onError(execResult.message || 'Action needs approval, but no approval id was returned.')
                stopLoop = true
                break
              }
              setPendingApproval({
                approvalId,
                action,
                message: execResult.message
              })
              await appendComputerStep(active.id, {
                role: 'assistant',
                content: `Waiting for approval: ${actionLabel(action)}`,
                action,
                result: execResult
              })
              // Pause the loop; user decides via the approval buttons.
              setSteps(await listComputerSteps(active.id))
              setRunning(false)
              abortRef.current = null
              return
            }

            if (execResult.status === 'error' || execResult.status === 'refused') {
              await appendComputerStep(active.id, {
                role: 'assistant',
                content: execResult.message || execResult.status,
                action,
                result: execResult
              })
              onError(execResult.message || `Action ${action.type} ${execResult.status}.`)
              stopLoop = true
              break
            }

            const shotUrl = screenshotDataUrl(execResult)
            if (shotUrl) setViewportUrl(shotUrl)
            else shot = (await refreshViewport(active.id)) ?? shot

            await appendComputerStep(active.id, {
              role: 'tool',
              content: execResult.message || actionLabel(action),
              action,
              result: execResult
            }).catch(async () => {
              // If tool role is rejected, keep the assistant step we already wrote.
              void recorded
            })

            if (execResult.screenshot_base64) {
              shot = execResult
            } else {
              shot = (await refreshViewport(active.id)) ?? shot
            }
          }

          nextSteps = await listComputerSteps(active.id)
          setSteps(nextSteps)
          if (stopLoop) break
        }
      } catch (cause) {
        if ((cause as Error).name !== 'AbortError') {
          onError(errorText(cause))
        }
      } finally {
        setRunning(false)
        abortRef.current = null
        setLiveThought('')
        const id = sessionRef.current?.id
        if (id) {
          void listComputerSteps(id).then(setSteps).catch(() => undefined)
        }
      }
    },
    [
      modelId,
      onError,
      permissionMode,
      refreshSessions,
      running,
      target
    ]
  )

  async function resolveApproval(approve: boolean): Promise<void> {
    if (!pendingApproval || !session) return
    onError(null)
    try {
      const { result } = await decideComputerApproval(pendingApproval.approvalId, approve)
      setPendingApproval(null)
      if (result) {
        const shotUrl = screenshotDataUrl(result)
        if (shotUrl) setViewportUrl(shotUrl)
        await appendComputerStep(session.id, {
          role: 'assistant',
          content: approve
            ? `Approved: ${actionLabel(pendingApproval.action)}`
            : `Denied: ${actionLabel(pendingApproval.action)}`,
          action: pendingApproval.action,
          result
        })
        setSteps(await listComputerSteps(session.id))
      }
      if (approve && result && result.status !== 'error' && result.status !== 'refused') {
        // Continue the loop with a nudge.
        await send('Continue after the approved action.')
      }
    } catch (cause) {
      onError(errorText(cause))
    }
  }

  const blockedReason = !modelId ? 'Choose a computer-use model in the top bar…' : ''
  const composerActionsRef = useRef({ send, stop })
  composerActionsRef.current = { send, stop }

  useEffect(() => {
    onComposerChange?.({
      send: (text) => composerActionsRef.current.send(text),
      stop: () => composerActionsRef.current.stop(),
      running,
      blockedReason,
      placeholder: blockedReason
        ? blockedReason
        : running
          ? 'Computer Use is working. Stop it to send something else…'
          : pendingApproval
            ? 'Approve or deny the pending action before continuing…'
            : `Ask ${modelLabel} to use the ${target}…`
    })
    return () => onComposerChange?.(null)
  }, [
    onComposerChange,
    running,
    blockedReason,
    modelLabel,
    target,
    pendingApproval
  ])

  useEffect(() => {
    return () => {
      abortRef.current?.abort()
    }
  }, [])

  return (
    <div className="computer-mode">
      <header className="computer-header">
        <div className="computer-session-bar">
          <button
            type="button"
            className="computer-new-session"
            title="Start a new computer session"
            onClick={() => void createSession().catch((cause) => onError(errorText(cause)))}
          >
            <Plus size={14} />
            New session
          </button>
          <div className="computer-session-list" role="list">
            {sessions.length === 0 ? (
              <span className="computer-session-empty">No sessions yet</span>
            ) : (
              sessions.map((entry) => (
                <button
                  key={entry.id}
                  type="button"
                  role="listitem"
                  className={session?.id === entry.id ? 'active' : ''}
                  onClick={() => void loadSession(entry.id)}
                >
                  <Monitor size={13} />
                  <span>{entry.title || 'Session'}</span>
                  <i
                    role="button"
                    tabIndex={0}
                    title="Delete session"
                    onClick={(event) => {
                      event.stopPropagation()
                      void removeSession(entry.id)
                    }}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault()
                        event.stopPropagation()
                        void removeSession(entry.id)
                      }
                    }}
                  >
                    <Trash2 size={12} />
                  </i>
                </button>
              ))
            )}
          </div>
        </div>
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
                <p>{pendingApproval.message || actionLabel(pendingApproval.action)}</p>
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
                  {step.action ? <span>{actionLabel(step.action)}</span> : null}
                </header>
                {step.thought ? (
                  <div className="computer-thought">
                    <pre>{step.thought}</pre>
                  </div>
                ) : null}
                {step.content && step.content !== actionLabel(step.action) ? (
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

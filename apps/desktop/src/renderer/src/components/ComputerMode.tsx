import {
  Check,
  LoaderCircle,
  Maximize,
  Minimize,
  Monitor,
  Settings2,
  ShieldAlert,
  X
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  buildComputerUseTool,
  normalizeComputerActionCoordinates,
  parseComputerCompletion,
  resolveComputerAdapter
} from '../../../computer/adapters'
import { useFullscreen } from './FullscreenButton'
import {
  appendComputerStep,
  computerExec,
  computerPreview,
  computerScreenshot,
  createComputerSession,
  decideComputerApproval,
  deleteComputerSession,
  fetchComputerPermissions,
  fetchComputerSession,
  fetchComputerUsePreference,
  listComputerSessions,
  listComputerSteps,
  parseFaraOutput as parseFaraRemote,
  setComputerSafetyAuthority,
  stopComputerSession,
  streamCompletion,
  streamComputerPreview,
  updateComputerSession,
  type ComputerAction,
  type ComputerActionResult,
  type ComputerPermissionMode,
  type ComputerSession,
  type ComputerStep,
  type ComputerTarget,
  type LocalModel,
  type OsPermissionStatus,
  type StreamCompletionResult
} from '../api'
import { modelDisplayName } from '../model-utils'
import { useConnectionProfile } from '../connectionProfile'
import type { AgentComposerControls } from './AgentMode'
import {
  buildComputerHistory,
  canInteractComputerViewport,
  canSendComputerPrompt,
  computerActionLabel,
  computerScreenshotDataUrl,
  computerTurnStateFromPause,
  continuationForResult,
  IDLE_COMPUTER_TURN_STATE,
  isComputerTurnBusy,
  observationError,
  pendingApprovalForState,
  pendingUserQuestionForState,
  recoverComputerPause,
  transitionComputerTurnState,
  type ComputerContinuation,
  type ComputerTurnEvent,
  type ComputerTurnState,
  type PendingComputerApproval
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
  /** Open Manage -> Computer use for guided model or permission setup. */
  onOpenManage?: (section: 'computer') => void
}

const MAX_LOOP_STEPS = 20
const MAX_CONSECUTIVE_REPAIRS = 2

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

type LoopOutcome =
  | { kind: 'completed' }
  | { kind: 'aborted' }
  | { kind: 'awaiting_approval'; approval: PendingComputerApproval }
  | { kind: 'awaiting_user'; question: string }
  | { kind: 'failed'; error: string }

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

function hasMissingOsPermissions(status: OsPermissionStatus | null): boolean {
  if (!status) return false
  return (
    status.screen_capture === 'denied' ||
    status.screen_capture === 'missing' ||
    status.input_injection === 'denied' ||
    status.input_injection === 'missing'
  )
}

export function ComputerMode(props: Props): React.JSX.Element {
  const { modelId, models, onError, onComposerChange, onSidebarChange, onOpenManage } = props
  const connectionProfile = useConnectionProfile()
  const [daemonPlatform, setDaemonPlatform] = useState<string | null>(null)
  const [osPermissions, setOsPermissions] = useState<OsPermissionStatus | null>(null)
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
  const [turnState, setTurnState] = useState<ComputerTurnState>(IDLE_COMPUTER_TURN_STATE)
  const [liveThought, setLiveThought] = useState('')
  const [maxScreenshotsKept, setMaxScreenshotsKept] = useState(3)

  const abortRef = useRef<AbortController | null>(null)
  const turnStateRef = useRef<ComputerTurnState>(IDLE_COMPUTER_TURN_STATE)
  const sessionRef = useRef<ComputerSession | null>(null)
  sessionRef.current = session

  const dispatchTurn = useCallback((event: ComputerTurnEvent): ComputerTurnState => {
    const next = transitionComputerTurnState(turnStateRef.current, event)
    turnStateRef.current = next
    setTurnState(next)
    return next
  }, [])

  const running = isComputerTurnBusy(turnState)
  const pendingApproval = pendingApprovalForState(turnState)
  const pendingUserQuestion = pendingUserQuestionForState(turnState)
  const resolvingApproval = turnState.kind === 'resolving_approval'

  useEffect(() => {
    let current = true
    void window.brazier
      .getConnection()
      .then((connection) => {
        if (current) setDaemonPlatform(connection.daemon.daemon?.platform ?? null)
      })
      .catch(() => {
        if (current) setDaemonPlatform(null)
      })
    return () => {
      current = false
    }
  }, [connectionProfile.id])

  useEffect(() => {
    if (
      target !== 'desktop' ||
      connectionProfile.kind === 'remote' ||
      daemonPlatform === null ||
      daemonPlatform === 'windows'
    ) {
      setOsPermissions(null)
      return
    }
    let current = true
    void fetchComputerPermissions()
      .then((status) => {
        if (current) setOsPermissions(status)
      })
      .catch(() => {
        if (current) setOsPermissions(null)
      })
    return () => {
      current = false
    }
  }, [target, connectionProfile.id, connectionProfile.kind, daemonPlatform])

  // Fills the window (or screen) with the browser viewport and the step log
  // side by side, so the user can watch the agent drive the page at full size.
  const { setRef: setFullscreenRef, active: fullscreenActive, toggle: toggleFullscreen } =
    useFullscreen<HTMLDivElement>()

  useEffect(() => {
    void fetchComputerUsePreference()
      .then((preference) => setMaxScreenshotsKept(preference.max_screenshots_kept))
      .catch(() => {})
  }, [])

  const refreshSessions = useCallback(async (): Promise<ComputerSession[]> => {
    const list = await listComputerSessions()
    setSessions(list)
    return list
  }, [])

  const refreshActiveSession = useCallback(
    async (sessionId: string): Promise<ComputerSession | null> => {
      try {
        const latest = await fetchComputerSession(sessionId)
        if (sessionRef.current?.id === sessionId) {
          sessionRef.current = latest
          setSession(latest)
        }
        return latest
      } catch {
        return sessionRef.current?.id === sessionId ? sessionRef.current : null
      }
    },
    []
  )

  const loadSession = useCallback(
    async (id: string): Promise<void> => {
      onError(null)
      if (isComputerTurnBusy(turnStateRef.current) && sessionRef.current && sessionRef.current.id !== id) {
        onError('Stop Computer Use before switching sessions.')
        return
      }
      try {
        const list = await refreshSessions()
        const found = list.find((entry) => entry.id === id) ?? null
        setSession(found)
        sessionRef.current = found
        if (!found) {
          setSteps([])
          dispatchTurn({ type: 'reset' })
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
        const restoredState = computerTurnStateFromPause(recovered)
        turnStateRef.current = restoredState
        setTurnState(restoredState)
      } catch (cause) {
        onError(errorText(cause))
      }
    },
    [dispatchTurn, onError, refreshSessions]
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
    dispatchTurn({ type: 'reset' })
    setViewportUrl(null)
    await refreshSessions()
    return created
  }

  /** Change the permission mode, applying it to the active session if any. */
  async function changePermissionMode(mode: ComputerPermissionMode): Promise<void> {
    setPermissionMode(mode)
    const active = sessionRef.current
    if (!active) return
    try {
      const updated = await updateComputerSession(active.id, mode)
      sessionRef.current = updated
      setSession(updated)
    } catch (cause) {
      onError(errorText(cause))
      setPermissionMode(active.permission_mode)
    }
  }

  function startNewSession(): void {
    if (isComputerTurnBusy(turnStateRef.current)) return
    setSession(null)
    sessionRef.current = null
    setSteps([])
    setViewportUrl(null)
    dispatchTurn({ type: 'reset' })
    setLiveThought('')
  }

  async function removeSession(id: string): Promise<void> {
    try {
      await deleteComputerSession(id)
      if (sessionRef.current?.id === id) {
        setSession(null)
        sessionRef.current = null
        setSteps([])
        setViewportUrl(null)
        dispatchTurn({ type: 'reset' })
      }
      await refreshSessions()
    } catch (cause) {
      onError(errorText(cause))
    }
  }

  function applyResultViewport(sessionId: string, result: ComputerActionResult | null | undefined): void {
    if (!result) return
    const url = computerScreenshotDataUrl(result)
    if (url) setViewportUrl(url)
    if (result.viewport && sessionRef.current?.id === sessionId) {
      const updated: ComputerSession = {
        ...sessionRef.current,
        viewport: result.viewport,
        ...(result.url ? { url: result.url } : {}),
        ...(result.title ? { title_page: result.title } : {})
      }
      sessionRef.current = updated
      setSession(updated)
    }
  }

  async function refreshViewport(sessionId: string): Promise<ComputerActionResult | null> {
    try {
      const shot = await computerScreenshot(sessionId)
      applyResultViewport(sessionId, shot)
      return shot
    } catch {
      return null
    }
  }

  const lastScrollRef = useRef(0)

  async function interact(action: ComputerAction): Promise<void> {
    const active = sessionRef.current
    if (!active || active.target !== 'browser' || !canInteractComputerViewport(turnStateRef.current)) {
      return
    }
    dispatchTurn({ type: 'start_interaction' })
    try {
      const result = await computerExec({ session_id: active.id, action })
      if (result.needs_approval || result.status === 'needs_approval') {
        if (result.approval_id) {
          dispatchTurn({
            type: 'require_approval',
            approval: {
              approvalId: result.approval_id,
              action,
              message: result.message,
              executionLocation: result.execution_location,
              resumeLoop: false
            }
          })
        }
        return
      }
      applyResultViewport(active.id, result)
      await syncSteps(active.id)
    } catch (cause) {
      onError(errorText(cause))
    } finally {
      dispatchTurn({ type: 'end_interaction' })
    }
  }

  /** Pixel position on the screenshot -> Fara's normalized 0..1000 space. */
  function viewportCoords(
    event: { currentTarget: HTMLElement; clientX: number; clientY: number }
  ): { x: number; y: number } {
    const rect = event.currentTarget.getBoundingClientRect()
    return {
      x: Math.min(1000, Math.max(0, ((event.clientX - rect.left) / rect.width) * 1000)),
      y: Math.min(1000, Math.max(0, ((event.clientY - rect.top) / rect.height) * 1000))
    }
  }

  function onViewportClick(event: React.MouseEvent<HTMLImageElement>): void {
    if (event.button !== 0) return
    const { x, y } = viewportCoords(event)
    void interact({ type: 'left_click', x, y })
  }

  function onViewportContextMenu(event: React.MouseEvent<HTMLImageElement>): void {
    event.preventDefault()
    const { x, y } = viewportCoords(event)
    void interact({ type: 'right_click', x, y })
  }

  function onViewportWheel(event: React.WheelEvent<HTMLImageElement>): void {
    const now = Date.now()
    if (now - lastScrollRef.current < 150) return
    lastScrollRef.current = now
    const { x, y } = viewportCoords(event)
    void interact({ type: 'scroll', x, y, delta_x: event.deltaX, delta_y: event.deltaY })
  }

  const VIEWPORT_KEYS: Record<string, string[]> = {
    Enter: ['Enter'],
    Tab: ['Tab'],
    Backspace: ['Backspace'],
    Escape: ['Escape'],
    ' ': ['Space'],
    ArrowUp: ['ArrowUp'],
    ArrowDown: ['ArrowDown'],
    ArrowLeft: ['ArrowLeft'],
    ArrowRight: ['ArrowRight'],
    Home: ['Home'],
    End: ['End'],
    Delete: ['Delete']
  }

  function onViewportKeyDown(event: React.KeyboardEvent<HTMLImageElement>): void {
    const keys = VIEWPORT_KEYS[event.key]
    if (keys) {
      event.preventDefault()
      void interact({ type: 'keypress', keys })
      return
    }
    if (event.key.length === 1 && (event.ctrlKey || event.metaKey) && !event.altKey) {
      event.preventDefault()
      const modifier = event.metaKey ? 'Cmd' : 'Ctrl'
      void interact({ type: 'keypress', keys: [modifier, event.key.toUpperCase()] })
      return
    }
    if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault()
      void interact({ type: 'type', text: event.key })
    }
  }

  const browserInteractive = Boolean(
    session &&
      session.target === 'browser' &&
      viewportUrl &&
      canInteractComputerViewport(turnState)
  )

  useEffect(() => {
    const sessionId = session?.id
    if (!sessionId || session.target !== 'browser') return
    const controller = new AbortController()
    let streaming = true
    void streamComputerPreview(
      sessionId,
      (data) => {
        setViewportUrl((current) => {
          const url = `data:image/jpeg;base64,${data}`
          return url === current ? current : url
        })
      },
      controller.signal
    )
      .catch(() => {})
      .finally(() => {
        streaming = false
      })
    const timer = window.setInterval(() => {
      if (streaming) return
      void computerPreview(sessionId)
        .then((shot) => {
          const url = computerScreenshotDataUrl(shot)
          if (url) {
            setViewportUrl((current) => (url === current ? current : url))
          }
        })
        .catch(() => {})
    }, 50)
    return () => {
      controller.abort()
      window.clearInterval(timer)
    }
  }, [session?.id, session?.target])

  async function parseCompletionWithAdapter(
    completion: StreamCompletionResult,
    activeSession: ComputerSession,
    activeModelId: string
  ): Promise<{ thought: string | null; actions: ComputerAction[]; assistantContent: string }> {
    const model = models.find((entry) => entry.id === activeModelId)
    const adapter = resolveComputerAdapter(activeModelId, model)
    if (
      adapter.dialect === 'fara-xml' &&
      completion.clientToolCalls.length === 0 &&
      completion.responseText.includes('<tool_call>')
    ) {
      try {
        const remote = await parseFaraRemote(completion.responseText)
        if (remote.actions && remote.actions.length > 0) {
          const normalized = remote.actions.map((action) =>
            normalizeComputerActionCoordinates(
              action as Parameters<typeof normalizeComputerActionCoordinates>[0],
              adapter.coordinateSpace,
              activeSession.viewport
            )
          ) as ComputerAction[]
          return {
            thought: completion.reasoningText.trim() || remote.thought || null,
            actions: normalized,
            assistantContent: completion.responseText
          }
        }
      } catch {
        // Fall through to universal local parser.
      }
    }
    const local = parseComputerCompletion(
      {
        responseText: completion.responseText,
        reasoningText: completion.reasoningText,
        clientToolCalls: completion.clientToolCalls
      },
      {
        dialect: adapter.dialect,
        coordinateSpace: adapter.coordinateSpace,
        viewport: activeSession.viewport
      }
    )
    return {
      thought: local.thought,
      actions: local.actions as ComputerAction[],
      assistantContent: local.assistantContent
    }
  }

  const stop = useCallback(async (): Promise<void> => {
    const previousState = turnStateRef.current
    abortRef.current?.abort()
    abortRef.current = null
    dispatchTurn({ type: 'stop' })
    setLiveThought('')
    await window.brazier.computer.setActive(false)
    if (previousState.kind === 'awaiting_approval') {
      void decideComputerApproval(
        previousState.approval.approvalId,
        false,
        previousState.approval.executionLocation ?? undefined
      ).catch(() => undefined)
    }
    const active = sessionRef.current
    if (active) {
      try {
        await stopComputerSession(active.id)
      } catch {
        // Escape remains a local emergency stop even if the daemon vanished.
      }
      void syncSteps(active.id).catch(() => undefined)
    }
  }, [dispatchTurn])

  async function syncSteps(sessionId: string): Promise<ComputerStep[]> {
    const next = await listComputerSteps(sessionId)
    setSteps(next)
    return next
  }

  async function executeActionBatch(
    active: ComputerSession,
    actions: ComputerAction[],
    controller: AbortController,
    repairBudget: { count: number }
  ): Promise<{ outcome: LoopOutcome | null }> {
    for (let index = 0; index < actions.length; index += 1) {
      if (controller.signal.aborted) return { outcome: { kind: 'aborted' } }
      const action = actions[index]
      const result = await computerExec({ session_id: active.id, action })
      applyResultViewport(active.id, result)
      if (action.type === 'memorize' && result.status === 'ok') {
        await refreshActiveSession(active.id)
      }
      await syncSteps(active.id)

      if (
        active.target === 'desktop' &&
        (action.type === 'visit_url' || action.type === 'web_search')
      ) {
        await appendComputerStep(active.id, {
          role: 'user',
          content:
            'That action is unavailable on this desktop. Continue by interacting with the visible desktop GUI using mouse and keyboard only.'
        })
        return { outcome: null }
      }

      if (result.needs_approval || result.status === 'needs_approval') {
        const approvalId = result.approval_id
        if (!approvalId) {
          return {
            outcome: {
              kind: 'failed',
              error: result.message || 'Action needs approval, but no approval id was returned.'
            }
          }
        }
        return {
          outcome: {
            kind: 'awaiting_approval',
            approval: {
              approvalId,
              action,
              message: result.message,
              executionLocation: result.execution_location,
              resumeLoop: true,
              remainingActions: actions.slice(index + 1)
            }
          }
        }
      }

      const continuation: ComputerContinuation = continuationForResult(result)
      if (continuation.kind === 'waiting_for_user') {
        return { outcome: { kind: 'awaiting_user', question: continuation.question } }
      }
      if (continuation.kind === 'finished') {
        return { outcome: { kind: 'completed' } }
      }
      if (continuation.kind === 'blocked') {
        if (result.status === 'error' && repairBudget.count < MAX_CONSECUTIVE_REPAIRS) {
          repairBudget.count += 1
          await appendComputerStep(active.id, {
            role: 'user',
            content: `The previous action (${action.type}) failed: ${result.message || 'execution error'}. Inspect the latest screenshot and choose a valid recovery action.`
          })
          return { outcome: null }
        }
        return {
          outcome: {
            kind: 'failed',
            error: result.message || `Action ${action.type} ${result.status}.`
          }
        }
      }

      repairBudget.count = 0
    }
    return { outcome: null }
  }

  async function runModelLoop(
    initialSession: ComputerSession,
    controller: AbortController,
    options: { skipInitialScreenshot?: boolean; initialActions?: ComputerAction[] } = {}
  ): Promise<LoopOutcome> {
    let active = sessionRef.current?.id === initialSession.id ? sessionRef.current : initialSession
    if (!options.skipInitialScreenshot) {
      const observation = await refreshViewport(active.id)
      const observationFailure = observationError(observation)
      if (observationFailure) {
        return { kind: 'failed', error: observationFailure }
      }
      active = sessionRef.current?.id === active.id ? sessionRef.current : active
      await syncSteps(active.id)
    }

    const repairBudget = { count: 0 }

    if (options.initialActions && options.initialActions.length > 0) {
      const resumed = await executeActionBatch(
        active,
        options.initialActions,
        controller,
        repairBudget
      )
      if (resumed.outcome) return resumed.outcome
    }

    const activeModelId = active.model_id || modelId
    const activeModel = models.find((entry) => entry.id === activeModelId)
    const adapter = resolveComputerAdapter(activeModelId, activeModel)

    for (let step = 0; step < MAX_LOOP_STEPS && !controller.signal.aborted; step += 1) {
      active = (await refreshActiveSession(active.id)) ?? active
      const currentSteps = await syncSteps(active.id)
      const history = buildComputerHistory(active, currentSteps, maxScreenshotsKept, {
        dialect: adapter.dialect,
        coordinateSpace: adapter.coordinateSpace
      })
      let streamedText = ''
      setLiveThought('')
      const useOpenAiTools = adapter.dialect === 'openai-tools'
      const completion = await streamCompletion(
        history,
        activeModelId,
        controller.signal,
        (token) => {
          streamedText += token
        },
        {
          onReasoning: (token) => setLiveThought((current) => current + token),
          ...(useOpenAiTools
            ? {
                extraTools: [
                  buildComputerUseTool(active.target, adapter.coordinateSpace, active.viewport)
                ],
                toolChoice: 'auto' as const
              }
            : {})
        }
      )
      if (controller.signal.aborted) return { kind: 'aborted' }

      const effectiveCompletion: StreamCompletionResult = {
        ...completion,
        responseText: completion.responseText || streamedText
      }
      const parsed = await parseCompletionWithAdapter(effectiveCompletion, active, activeModelId)
      if (parsed.thought) setLiveThought(parsed.thought)

      await appendComputerStep(active.id, {
        role: 'assistant',
        content:
          parsed.assistantContent ||
          effectiveCompletion.responseText ||
          (parsed.thought ? `<think>${parsed.thought}</think>` : '(no action output)'),
        thought: parsed.thought
      })

      if (parsed.actions.length === 0) {
        if (repairBudget.count < MAX_CONSECUTIVE_REPAIRS) {
          repairBudget.count += 1
          await appendComputerStep(active.id, {
            role: 'user',
            content:
              adapter.dialect === 'openai-tools'
                ? 'No executable `computer_use` tool call was returned. Call `computer_use` with one next action (use `terminate` if finished, or `ask_user_question` if you need user input).'
                : adapter.dialect === 'ui-tars'
                  ? "No executable `Action:` line was found. Reply with `Thought: ...` and `Action: click(start_box='(x,y)')` (or `finished(content='...')` if complete)."
                  : 'No executable `<tool_call>` was found. Reply with `<tool_call>{"name":"computer_use","arguments":{"action":"..."}}</tool_call>` (use `terminate` if finished).'
          })
          continue
        }
        return {
          kind: 'failed',
          error:
            'The computer-use model returned no executable action. Try again or choose another model.'
        }
      }

      const parseError = parsed.actions.find((action) => action.type === 'error')
      if (parseError) {
        if (repairBudget.count < MAX_CONSECUTIVE_REPAIRS) {
          repairBudget.count += 1
          await appendComputerStep(active.id, {
            role: 'user',
            content: `Your tool call could not be parsed (${computerActionLabel(parseError)}). Emit valid JSON arguments for a single computer_use action.`
          })
          continue
        }
        return { kind: 'failed', error: computerActionLabel(parseError) }
      }

      const batch = await executeActionBatch(active, parsed.actions, controller, repairBudget)
      if (batch.outcome) return batch.outcome
    }

    return controller.signal.aborted ? { kind: 'aborted' } : { kind: 'completed' }
  }

  function applyLoopOutcome(outcome: LoopOutcome): void {
    switch (outcome.kind) {
      case 'awaiting_approval':
        dispatchTurn({ type: 'require_approval', approval: outcome.approval })
        return
      case 'awaiting_user':
        dispatchTurn({ type: 'require_user', question: outcome.question })
        return
      case 'aborted':
        if (turnStateRef.current.kind !== 'stopped') {
          dispatchTurn({ type: 'stop' })
        }
        return
      case 'failed':
        onError(outcome.error)
        dispatchTurn({ type: 'turn_finished' })
        return
      case 'completed':
        dispatchTurn({ type: 'turn_finished' })
        return
    }
  }

  async function executeApprovedAction(
    activeSession: ComputerSession,
    approval: PendingComputerApproval
  ): Promise<ComputerActionResult | null> {
    const decided = await decideComputerApproval(
      approval.approvalId,
      true,
      approval.executionLocation ?? undefined
    )
    if (decided.result) return decided.result
    return computerExec({
      session_id: activeSession.id,
      action: approval.action,
      approval_id: approval.approvalId
    })
  }

  async function send(text: string): Promise<void> {
    const userText = text.trim()
    if (!userText || !canSendComputerPrompt(turnStateRef.current)) return
    if (!modelId && !sessionRef.current?.model_id) {
      onError('Choose a computer-use model in the top bar.')
      return
    }

    onError(null)
    dispatchTurn({ type: 'start_turn' })
    setLiveThought('')
    const controller = new AbortController()
    abortRef.current = controller
    let desktopSafetyActive = false
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
      if (controller.signal.aborted) {
        applyLoopOutcome({ kind: 'aborted' })
        return
      }
      if (active.target === 'desktop') {
        await window.brazier.computer.setActive(true)
        desktopSafetyActive = true
        try {
          await setComputerSafetyAuthority(active.id, true)
        } catch (cause) {
          await window.brazier.computer.setActive(false)
          desktopSafetyActive = false
          throw cause
        }
      }
      await appendComputerStep(active.id, { role: 'user', content: userText })
      const outcome = await runModelLoop(active, controller)
      applyLoopOutcome(outcome)
    } catch (cause) {
      if ((cause as Error).name !== 'AbortError') onError(errorText(cause))
    } finally {
      const active = sessionRef.current
      if (desktopSafetyActive && active?.target === 'desktop') {
        void setComputerSafetyAuthority(active.id, false).catch(() => undefined)
      }
      void window.brazier.computer.setActive(false)
      // Only transitions `running -> idle`; never clobbers `awaiting_approval`,
      // `awaiting_user`, or `stopped`.
      dispatchTurn({ type: 'turn_finally' })
      if (abortRef.current === controller) abortRef.current = null
      setLiveThought('')
      const id = sessionRef.current?.id
      if (id) void syncSteps(id).catch(() => undefined)
    }
  }

  async function resolveApproval(approve: boolean): Promise<void> {
    const currentState = turnStateRef.current
    if (currentState.kind === 'stopped') {
      if (approve) {
        onError('Computer Use was stopped. Start a new turn to continue.')
      }
      return
    }
    if (currentState.kind !== 'awaiting_approval' || !session) return

    const approval = currentState.approval
    dispatchTurn({ type: 'begin_resolve_approval', approve })
    onError(null)

    if (!approve) {
      try {
        await decideComputerApproval(
          approval.approvalId,
          false,
          approval.executionLocation ?? undefined
        )
        await syncSteps(session.id)
        dispatchTurn({ type: 'approval_denied' })
      } catch (cause) {
        onError(errorText(cause))
        try {
          const nextSteps = await syncSteps(session.id)
          const restored = computerTurnStateFromPause(recoverComputerPause(nextSteps))
          turnStateRef.current = restored
          setTurnState(restored)
        } catch {
          dispatchTurn({ type: 'approval_denied' })
        }
      }
      return
    }

    let safetyActive = false
    const controller = new AbortController()
    abortRef.current = controller
    try {
      if (session.target === 'desktop') {
        await window.brazier.computer.setActive(true)
        safetyActive = true
        try {
          await setComputerSafetyAuthority(session.id, true)
        } catch (cause) {
          await window.brazier.computer.setActive(false)
          safetyActive = false
          throw cause
        }
      }

      const result = await executeApprovedAction(session, approval)
      if (result) {
        applyResultViewport(session.id, result)
        if (approval.action.type === 'memorize' && result.status === 'ok') {
          await refreshActiveSession(session.id)
        }
      }
      const syncedSteps = await syncSteps(session.id)
      const activeSession = (await refreshActiveSession(session.id)) ?? session

      if (controller.signal.aborted) {
        applyLoopOutcome({ kind: 'aborted' })
        return
      }

      if (!result) {
        dispatchTurn({ type: 'turn_finished' })
        return
      }

      if (approval.resumeLoop === false) {
        const continuation = continuationForResult(result)
        if (continuation.kind === 'waiting_for_user') {
          dispatchTurn({ type: 'require_user', question: continuation.question })
        } else {
          dispatchTurn({ type: 'turn_finished' })
        }
        return
      }

      const continuation = continuationForResult(result)
      if (continuation.kind === 'waiting_for_user') {
        dispatchTurn({ type: 'require_user', question: continuation.question })
        return
      }
      if (continuation.kind === 'finished') {
        dispatchTurn({ type: 'turn_finished' })
        return
      }
      if (continuation.kind === 'blocked') {
        onError(result.message || `Action ${approval.action.type} ${result.status}.`)
        dispatchTurn({ type: 'turn_finished' })
        return
      }

      // Transition resolving_approval -> running and resume the model loop.
      dispatchTurn({ type: 'approval_approved_continue' })
      const hasObservation =
        Boolean(result.screenshot_base64) ||
        syncedSteps.some((step) => Boolean(computerScreenshotDataUrl(step.result)))
      const outcome = await runModelLoop(activeSession, controller, {
        skipInitialScreenshot: hasObservation,
        initialActions: approval.remainingActions
      })
      applyLoopOutcome(outcome)
    } catch (cause) {
      if ((cause as Error).name !== 'AbortError') onError(errorText(cause))
      try {
        const nextSteps = await syncSteps(session.id)
        const restored = computerTurnStateFromPause(recoverComputerPause(nextSteps))
        turnStateRef.current = restored
        setTurnState(restored)
      } catch {
        dispatchTurn({ type: 'turn_finished' })
      }
    } finally {
      if (safetyActive) {
        void setComputerSafetyAuthority(session.id, false).catch(() => undefined)
        void window.brazier.computer.setActive(false)
      }
      dispatchTurn({ type: 'turn_finally' })
      if (abortRef.current === controller) abortRef.current = null
      setLiveThought('')
      void syncSteps(session.id).catch(() => undefined)
    }
  }

  const desktopPermissionsMissing =
    target === 'desktop' &&
    connectionProfile.kind !== 'remote' &&
    daemonPlatform !== 'windows' &&
    hasMissingOsPermissions(osPermissions)

  const targetUnavailable =
    daemonPlatform === null
      ? 'Checking whether Computer Use is available on this daemon…'
      : target === 'desktop' && connectionProfile.kind === 'remote'
        ? `Desktop control must be activated on ${connectionProfile.name}; use Browser for this remote daemon.`
        : daemonPlatform === 'windows'
          ? 'Computer Use is unavailable on this Windows daemon; Agent mode remains sandboxed and supported.'
          : ''
  const blockedReason =
    targetUnavailable ||
    (!modelId && !session?.model_id
      ? 'Choose a computer-use model in the top bar…'
      : pendingApproval
        ? 'Approve or deny the pending action before continuing…'
        : '')
  const composerPlaceholder =
    targetUnavailable ||
    (!modelId && !session?.model_id
      ? 'Choose a computer-use model in the top bar…'
      : running
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
      abortRef.current = null
      const id = sessionRef.current?.id
      void window.brazier.computer.setActive(false)
      if (id) void stopComputerSession(id).catch(() => undefined)
    }
  }, [])

  const sidebarActionsRef = useRef({ loadSession, removeSession, createSession, startNewSession })
  sidebarActionsRef.current = { loadSession, removeSession, createSession, startNewSession }
  useEffect(() => {
    onSidebarChange?.({
      sessions,
      activeId: session?.id ?? null,
      select: (id) => void sidebarActionsRef.current.loadSession(id),
      remove: (id) => void sidebarActionsRef.current.removeSession(id),
      newSession: () => sidebarActionsRef.current.startNewSession()
    })
    return () => onSidebarChange?.(null)
  }, [onSidebarChange, sessions, session?.id, onError])

  useEffect(() => {
    const unsub = window.brazier.computer.onEscape(() => {
      void stop()
    })
    return () => {
      unsub?.()
    }
  }, [stop])

  return (
    <div className="computer-mode" ref={setFullscreenRef}>
      <header className="computer-header">
        <div className="computer-controls">
          <label>
            <span>Target</span>
            <select
              value={target}
              disabled={running || Boolean(session)}
              title={
                session
                  ? 'This session target is fixed. Start a new session to pick another.'
                  : 'Target for the next session'
              }
              onChange={(event) => setTarget(event.target.value as ComputerTarget)}
            >
              <option value="browser">Browser</option>
              <option value="desktop" disabled={connectionProfile.kind === 'remote'}>
                Desktop{connectionProfile.kind === 'remote' ? ' (daemon-host activation required)' : ''}
              </option>
            </select>
          </label>
          <label>
            <span>Permissions</span>
            <select
              value={permissionMode}
              disabled={running}
              title={
                session
                  ? PERMISSION_LABELS[permissionMode].detail
                  : `For the next session. ${PERMISSION_LABELS[permissionMode].detail}`
              }
              onChange={(event) =>
                void changePermissionMode(event.target.value as ComputerPermissionMode)
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
        <div className="computer-header-actions">
          <button
            type="button"
            className="chip-button subtle"
            title={
              fullscreenActive
                ? 'Exit fullscreen'
                : 'Fill the window with the browser and steps side by side'
            }
            onClick={toggleFullscreen}
          >
            {fullscreenActive ? <Minimize size={13} /> : <Maximize size={13} />}
            {fullscreenActive ? 'Exit' : 'Fullscreen'}
          </button>
        </div>
      </header>

      <div className="computer-body">
        <div className="computer-viewport">
          {viewportUrl ? (
            <img
              src={viewportUrl}
              alt="Computer Use viewport"
              className={browserInteractive ? 'interactive' : ''}
              tabIndex={browserInteractive ? 0 : undefined}
              onClick={browserInteractive ? onViewportClick : undefined}
              onContextMenu={browserInteractive ? onViewportContextMenu : undefined}
              onKeyDown={browserInteractive ? onViewportKeyDown : undefined}
              onWheel={browserInteractive ? onViewportWheel : undefined}
            />
          ) : (
            <div className="computer-viewport-empty">
              <Monitor size={28} />
              <p>Viewport preview appears after the first screenshot.</p>
            </div>
          )}
          {browserInteractive && !running && (
            <span className="computer-input-hint">Click or type to drive the browser</span>
          )}
          {running && (
            <div className="computer-viewport-busy">
              <LoaderCircle className="spin" size={16} />
              Working…
            </div>
          )}
        </div>

        <div className="computer-steps">
          {desktopPermissionsMissing && (
            <div className="computer-approval">
              <div>
                <strong>
                  <ShieldAlert size={14} style={{ verticalAlign: 'text-bottom', marginRight: 6 }} />
                  Desktop permissions required
                </strong>
                <p>
                  {osPermissions?.detail ||
                    osPermissions?.settings_hint ||
                    'Grant screen capture and input control permissions before driving the desktop.'}
                </p>
              </div>
              {onOpenManage ? (
                <div className="computer-approval-actions">
                  <button
                    type="button"
                    className="primary"
                    onClick={() => onOpenManage('computer')}
                  >
                    <Settings2 size={14} />
                    Configure desktop permissions
                  </button>
                </div>
              ) : null}
            </div>
          )}

          {pendingApproval && (
            <div className="computer-approval">
              <div>
                <strong>Approval needed</strong>
                <p>{pendingApproval.message || computerActionLabel(pendingApproval.action)}</p>
                {pendingApproval.executionLocation ? (
                  <p>
                    Runs on {pendingApproval.executionLocation.daemon_display_name} ·{' '}
                    {pendingApproval.executionLocation.platform}/{pendingApproval.executionLocation.arch} · daemon{' '}
                    <code>{pendingApproval.executionLocation.daemon_instance_id}</code>
                  </p>
                ) : (
                  <p>Daemon host details were not retained for this legacy approval.</p>
                )}
              </div>
              <div className="computer-approval-actions">
                <button
                  type="button"
                  disabled={resolvingApproval}
                  onClick={() => void resolveApproval(false)}
                >
                  {resolvingApproval ? <LoaderCircle className="spin" size={14} /> : <X size={14} />}
                  Deny
                </button>
                <button
                  type="button"
                  className="primary"
                  disabled={resolvingApproval || !pendingApproval.executionLocation}
                  title={
                    pendingApproval.executionLocation
                      ? 'Approve this action on the recorded daemon host.'
                      : 'Legacy approval has no trustworthy execution host and can only be denied.'
                  }
                  onClick={() => void resolveApproval(true)}
                >
                  {resolvingApproval ? <LoaderCircle className="spin" size={14} /> : <Check size={14} />}
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
              {models.length === 0 ? (
                <>
                  <p>
                    Install a computer-use or vision model to screenshot the {target} and run
                    guided GUI actions.
                  </p>
                  {onOpenManage ? (
                    <button
                      type="button"
                      className="primary"
                      onClick={() => onOpenManage('computer')}
                    >
                      <Settings2 size={14} />
                      Set up Computer Use
                    </button>
                  ) : null}
                </>
              ) : (
                <p>
                  Describe a task below. Computer Use will screenshot the {target}, ask the model,
                  and run the returned actions.
                </p>
              )}
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
                {step.result?.execution_location ? (
                  <small className="computer-step-location">
                    Ran on {step.result.execution_location.daemon_display_name} ·{' '}
                    {step.result.execution_location.platform}/{step.result.execution_location.arch} · daemon{' '}
                    <code>{step.result.execution_location.daemon_instance_id}</code>
                    {step.result.decided_by_client_id
                      ? ` · approved by ${step.result.decided_by_client_id}`
                      : ''}
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

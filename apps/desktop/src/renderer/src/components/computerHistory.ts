import type {
  ComputerActionDialect,
  CoordinateSpace
} from '../../../computer/recipes'
import type {
  ComputerAction,
  ComputerActionResult,
  ComputerSession,
  ComputerStep,
  ExecutionLocation
} from '../api'
import type { ContentPart, Message } from '../types'

/** Matches Fara's reference agent: keep only the most recent few screenshots. */
export const MAX_COMPUTER_HISTORY_IMAGES = 3
/** Keep enough trajectory for recovery without letting a long task consume its context. */
export const MAX_COMPUTER_HISTORY_MESSAGES = 80

export type ComputerActionResultExtended = ComputerActionResult & {
  screen_changed?: boolean | null
}

export type PendingComputerApproval = {
  approvalId: string
  action: ComputerAction
  message?: string | null
  executionLocation?: ExecutionLocation | null
  /** True when the approval paused a model loop; false for direct user viewport actions. */
  resumeLoop?: boolean
  /** Remaining actions in the same model turn after the held action. */
  remainingActions?: ComputerAction[]
}

export type ComputerTurnState =
  | { kind: 'idle' }
  | { kind: 'running' }
  | { kind: 'awaiting_approval'; approval: PendingComputerApproval }
  | { kind: 'resolving_approval'; approval: PendingComputerApproval; approve: boolean }
  | { kind: 'awaiting_user'; question: string }
  | { kind: 'interacting'; previousQuestion?: string | null }
  | { kind: 'stopped' }

export type ComputerTurnEvent =
  | { type: 'start_turn' }
  | { type: 'require_approval'; approval: PendingComputerApproval }
  | { type: 'begin_resolve_approval'; approve: boolean }
  | { type: 'approval_denied' }
  | { type: 'approval_approved_continue' }
  | { type: 'require_user'; question: string }
  | { type: 'turn_finished' }
  | { type: 'turn_finally' }
  | { type: 'start_interaction' }
  | { type: 'end_interaction' }
  | { type: 'stop' }
  | { type: 'reset' }

export const IDLE_COMPUTER_TURN_STATE: ComputerTurnState = { kind: 'idle' }

export function isComputerTurnBusy(state: ComputerTurnState): boolean {
  return state.kind === 'running' || state.kind === 'resolving_approval'
}

export function canSendComputerPrompt(state: ComputerTurnState): boolean {
  return (
    state.kind === 'idle' ||
    state.kind === 'awaiting_user' ||
    state.kind === 'stopped'
  )
}

export function canInteractComputerViewport(state: ComputerTurnState): boolean {
  return (
    state.kind === 'idle' ||
    state.kind === 'awaiting_user' ||
    state.kind === 'stopped'
  )
}

export function pendingApprovalForState(
  state: ComputerTurnState
): PendingComputerApproval | null {
  return state.kind === 'awaiting_approval' || state.kind === 'resolving_approval'
    ? state.approval
    : null
}

export function pendingUserQuestionForState(state: ComputerTurnState): string | null {
  return state.kind === 'awaiting_user' ? state.question : null
}

export function computerTurnStateFromPause(
  pause: RecoveredComputerPause
): ComputerTurnState {
  if (pause.approval) {
    return {
      kind: 'awaiting_approval',
      approval: { ...pause.approval, resumeLoop: true }
    }
  }
  if (pause.userQuestion) {
    return { kind: 'awaiting_user', question: pause.userQuestion }
  }
  return IDLE_COMPUTER_TURN_STATE
}

/**
 * Pure state machine reducer for Computer Use turn lifecycle.
 *
 * Critically, `turn_finally` ONLY transitions `running -> idle`, so `send()`'s
 * cleanup block can never overwrite `awaiting_approval`, `awaiting_user`, or `stopped`.
 */
export function transitionComputerTurnState(
  state: ComputerTurnState,
  event: ComputerTurnEvent
): ComputerTurnState {
  switch (event.type) {
    case 'start_turn':
      return canSendComputerPrompt(state) ? { kind: 'running' } : state
    case 'require_approval':
      return { kind: 'awaiting_approval', approval: event.approval }
    case 'begin_resolve_approval':
      return state.kind === 'awaiting_approval'
        ? {
            kind: 'resolving_approval',
            approval: state.approval,
            approve: event.approve
          }
        : state
    case 'approval_denied':
      return IDLE_COMPUTER_TURN_STATE
    case 'approval_approved_continue':
      return { kind: 'running' }
    case 'require_user':
      return { kind: 'awaiting_user', question: event.question }
    case 'turn_finished':
      return IDLE_COMPUTER_TURN_STATE
    case 'turn_finally':
      return state.kind === 'running' ? IDLE_COMPUTER_TURN_STATE : state
    case 'start_interaction':
      return canInteractComputerViewport(state)
        ? {
            kind: 'interacting',
            previousQuestion: state.kind === 'awaiting_user' ? state.question : null
          }
        : state
    case 'end_interaction':
      return state.kind === 'interacting'
        ? state.previousQuestion
          ? { kind: 'awaiting_user', question: state.previousQuestion }
          : IDLE_COMPUTER_TURN_STATE
        : state
    case 'stop':
      return { kind: 'stopped' }
    case 'reset':
      return IDLE_COMPUTER_TURN_STATE
  }
}

export function computerActionLabel(action: ComputerAction | null | undefined): string {
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
    case 'error':
      return `Parse error: ${action.error}${action.raw ? ` (${action.raw.slice(0, 48)}${action.raw.length > 48 ? '…' : ''})` : ''}`
    default:
      return action.type.replaceAll('_', ' ')
  }
}

export function computerScreenshotDataUrl(result: ComputerActionResult | null | undefined): string | null {
  if (!result?.screenshot_base64) return null
  return `data:${result.mime_type || 'image/png'};base64,${result.screenshot_base64}`
}

/** Fail closed: the model must never be invoked without a real current observation. */
export function observationError(result: ComputerActionResult | null | undefined): string | null {
  if (result?.status === 'ok' && result.screenshot_base64) return null
  return result?.message || 'Could not capture the current computer screenshot.'
}

function historyMessage(role: Message['role'], content: string | ContentPart[], id: string): Message {
  return {
    id: `computer-history-${id}`,
    conversation_id: 'computer',
    parent_id: null,
    role,
    content,
    model: null,
    created_at: new Date().toISOString()
  }
}

export type ComputerPromptOptions = {
  dialect?: ComputerActionDialect
  coordinateSpace?: CoordinateSpace
  extraMemories?: string[]
}

function collectMemorizedFacts(
  sessionMemories?: string[] | null,
  steps?: ComputerStep[],
  extraMemories?: string[]
): string[] {
  const seen = new Set<string>()
  const facts: string[] = []
  const push = (raw: string | undefined | null): void => {
    const trimmed = raw?.trim()
    if (!trimmed || seen.has(trimmed)) return
    seen.add(trimmed)
    facts.push(trimmed)
  }
  for (const item of sessionMemories ?? []) push(item)
  for (const item of extraMemories ?? []) push(item)
  for (const step of steps ?? []) {
    if (step.action?.type === 'memorize' && step.result?.status === 'ok') {
      push(step.action.fact)
    }
  }
  return facts
}

function coordinateSpaceInstruction(
  desktop: boolean,
  viewportLabel: string,
  coordinateSpace: CoordinateSpace,
  target: ComputerSession['target'],
  permissionMode: ComputerSession['permission_mode']
): string {
  const surfaceLabel = desktop ? `The visible desktop is ${viewportLabel}.` : `The browser viewport is ${viewportLabel}.`
  if (coordinateSpace === 'pixels') {
    return `${surfaceLabel} Coordinates in computer_use actions are in pixel coordinates matching ${viewportLabel}. Target: ${target}. Permission mode: ${permissionMode}.`
  }
  if (coordinateSpace === 'norm1') {
    return `${surfaceLabel} Coordinates in computer_use actions are unit floats in 0.0..1.0 normalized display space. Target: ${target}. Permission mode: ${permissionMode}.`
  }
  return `${surfaceLabel} Coordinates in computer_use actions are in Fara's normalized 0..1000 model-display space, not screen pixels. Target: ${target}. Permission mode: ${permissionMode}.`
}

export function computerSystemPrompt(
  session: Pick<ComputerSession, 'target' | 'permission_mode'> & {
    viewport?: ComputerSession['viewport']
    memories?: ComputerSession['memories']
  },
  options: ComputerPromptOptions = {}
): string {
  // Microsoft publishes this as the required system prompt for Fara inference.
  // Keep its Critical Point policy intact; only the action signature changes to
  // match the target and dialect that Brazier exposes.
  const microsoftFaraPrompt = [
    'You are a web automation agent that performs actions on websites to fulfill user requests by calling various tools.',
    'You should stop execution at Critical Points. A Critical Point occurs in tasks like checkout, booking, purchase, call, email, or order.',
    'A Critical Point requires the user\'s permission or personal/sensitive information (name, email, credit card, address, payment information, resume, etc.) to complete a transaction or to communicate as a human would.',
    'Guideline: Solve the task as far as possible up until a Critical Point.',
    'For example, find the phone number rather than making a call, and add suitable shoes to a cart rather than placing the order.',
    'Some tasks, like answering questions, may not encounter a Critical Point at all.'
  ]
  const dialect = options.dialect ?? 'fara-xml'
  const coordinateSpace = options.coordinateSpace ?? 'norm1000'
  const desktop = session.target === 'desktop'
  const viewport = session.viewport
    ? `${session.viewport.width}x${session.viewport.height}`
    : 'the latest screenshot dimensions'
  const actions = desktop
    ? 'key, type, mouse_move, left_click, right_click, double_click, triple_click, left_click_drag, scroll, pause_and_memorize_fact, ask_user_question, wait, and terminate'
    : 'key, type, mouse_move, left_click, right_click, double_click, triple_click, left_click_drag, scroll, visit_url, web_search, pause_and_memorize_fact, ask_user_question, wait, and terminate'

  let formatGuidance: string[]
  if (dialect === 'openai-tools') {
    formatGuidance = [
      'Invoke the `computer_use` tool with exactly one next action per turn.',
      `Supported actions are ${actions}.`
    ]
  } else if (dialect === 'ui-tars') {
    const uiTarsActions = desktop
      ? "click(start_box='(x,y)'), left_double(start_box='(x,y)'), right_single(start_box='(x,y)'), drag(start_box='(x1,y1)', end_box='(x2,y2)'), hotkey(key='ctrl c'), type(content='text'), scroll(start_box='(x,y)', direction='down'), wait(), call_user(content='question'), and finished(content='summary')"
      : "click(start_box='(x,y)'), left_double(start_box='(x,y)'), right_single(start_box='(x,y)'), drag(start_box='(x1,y1)', end_box='(x2,y2)'), hotkey(key='ctrl l'), type(content='text'), scroll(start_box='(x,y)', direction='down'), open_url(url='https://example.com'), wait(), call_user(content='question'), and finished(content='summary')"
    formatGuidance = [
      "Respond using `Thought: <reasoning>` on the first line and `Action: <call>` on the second line, for example: `Action: click(start_box='(500,320)')`.",
      `Supported actions are ${uiTarsActions}.`
    ]
  } else {
    const example = desktop
      ? '<tool_call>{"name":"computer_use","arguments":{"action":"left_click","coordinate":[720,450]}}</tool_call>.'
      : '<tool_call>{"name":"computer_use","arguments":{"action":"visit_url","url":"https://example.com"}}</tool_call>.'
    formatGuidance = [
      'Return exactly one next action, for example:',
      example,
      `Supported actions are ${actions}.`
    ]
  }

  const basePrompt = desktop
    ? [
        'You are a computer-use agent that performs actions in a visible desktop GUI to fulfill user requests by calling various tools.',
        ...microsoftFaraPrompt.slice(1),
        'You do not have a terminal, application menu, browser address bar, or any browser-navigation tools. Use only what is visible in the screenshot.',
        'To open an application, use the visible desktop launcher, dock, panel, or icon with mouse and keyboard actions.',
        'Use visual evidence from the latest screenshot and the recorded trajectory; do not assume screen state.',
        ...formatGuidance,
        'Pause with ask_user_question when required personal information is missing or the task is ambiguous.',
        'Before an irreversible action, ask for confirmation unless the user explicitly authorized that exact action.',
        'Never invent personal or payment information.',
        coordinateSpaceInstruction(desktop, viewport, coordinateSpace, session.target, session.permission_mode)
      ].join(' ')
    : [
        ...microsoftFaraPrompt,
        'This is an isolated browser interface.',
        'Use visual evidence from the latest screenshot and the recorded trajectory; do not assume page state.',
        ...formatGuidance,
        'Pause with ask_user_question when required personal information is missing or the task is ambiguous.',
        'Before an irreversible action, ask for confirmation unless the user explicitly authorized that exact action.',
        'Never invent personal or payment information.',
        coordinateSpaceInstruction(desktop, viewport, coordinateSpace, session.target, session.permission_mode)
      ].filter(Boolean).join(' ')

  const memories = collectMemorizedFacts(session.memories, undefined, options.extraMemories)
  if (memories.length === 0) return basePrompt
  return `${basePrompt}\n\nMemorized facts:\n${memories.map((fact) => `- ${fact}`).join('\n')}`
}

function actionRepeatSignature(action: ComputerAction | null | undefined): string | null {
  if (!action) return null
  const bucket = (value: number): number => Math.round(value / 5) * 5
  switch (action.type) {
    case 'left_click':
    case 'right_click':
    case 'double_click':
    case 'triple_click':
    case 'mouse_move':
      return `${action.type}:${bucket(action.x)},${bucket(action.y)}`
    case 'left_click_drag':
      return `drag:${bucket(action.start_x)},${bucket(action.start_y)}->${bucket(action.end_x)},${bucket(action.end_y)}`
    case 'scroll':
      return `scroll:${bucket(action.x)},${bucket(action.y)}:${Math.sign(action.delta_x)},${Math.sign(action.delta_y)}`
    case 'type':
      return `type:${action.text}`
    case 'keypress':
      return `keypress:${action.keys.join('+').toLowerCase()}`
    case 'visit_url':
      return `visit_url:${action.url}`
    case 'web_search':
      return `web_search:${action.query}`
    default:
      return null
  }
}

function isStepUnchanged(
  step: ComputerStep,
  previousScreenshot: string | null,
  previousUrl: string | null
): boolean {
  const result = step.result as ComputerActionResultExtended | null | undefined
  if (!result) return false
  if (result.screen_changed === false) return true
  if (result.screen_changed === true) return false
  if (result.status === 'error') return true
  if (
    previousScreenshot &&
    result.screenshot_base64 &&
    result.screenshot_base64 === previousScreenshot &&
    (result.url ?? null) === previousUrl
  ) {
    return true
  }
  return false
}

function toolResultText(step: ComputerStep, loopWarning: boolean): string {
  const action = computerActionLabel(step.action)
  const result = step.result as ComputerActionResultExtended | null | undefined
  return [
    `Computer action: ${action}.`,
    `Status: ${result?.status || 'unknown'}.`,
    result?.message ? `Result: ${result.message}` : '',
    result?.url ? `URL: ${result.url}` : '',
    result?.screen_changed === false
      ? 'Note: The screen and URL did not visibly change after this action.'
      : '',
    loopWarning
      ? 'Loop warning: Repeating this exact action did not change the screen. Do not repeat the same coordinates or action again—try another visible control, scroll, keyboard navigation (Tab/Enter), or ask_user_question if blocked.'
      : ''
  ]
    .filter(Boolean)
    .join(' ')
}

/**
 * Turn durable ComputerStep records into the prompt for the next model turn.
 * Broker-written tool records are represented as user messages so the model sees
 * the environment observation without pretending it authored the result.
 *
 * `maxImages` bounds how many screenshots are replayed; Fara is trained with
 * only the most recent few, and each one costs roughly 1500 tokens of context.
 */
export function buildComputerHistory(
  session: Pick<ComputerSession, 'target' | 'permission_mode'> & {
    viewport?: ComputerSession['viewport']
    memories?: ComputerSession['memories']
  },
  steps: ComputerStep[],
  maxImages: number = MAX_COMPUTER_HISTORY_IMAGES,
  options: Omit<ComputerPromptOptions, 'extraMemories'> = {}
): Message[] {
  const imageStepIds = new Set(
    steps
      .filter((step) => Boolean(computerScreenshotDataUrl(step.result)))
      .slice(-Math.max(1, maxImages))
      .map((step) => step.id)
  )

  // Detect repeated identical interactive actions that failed to change screen/URL.
  const loopWarningStepIds = new Set<string>()
  let lastScreenshot: string | null = null
  let lastUrl: string | null = null
  let lastSignature: string | null = null
  let unchangedStreak = 0

  for (const step of steps) {
    if (step.role === 'user' || step.role === 'assistant') continue
    const sig = actionRepeatSignature(step.action)
    if (sig) {
      const unchanged = isStepUnchanged(step, lastScreenshot, lastUrl)
      if (sig === lastSignature && unchanged) {
        unchangedStreak += 1
      } else {
        lastSignature = sig
        unchangedStreak = unchanged ? 1 : 0
      }
      if (unchangedStreak >= 2) {
        loopWarningStepIds.add(step.id)
      }
    }
    if (step.result?.screenshot_base64) {
      lastScreenshot = step.result.screenshot_base64
    }
    if (step.result?.url !== undefined) {
      lastUrl = step.result.url ?? null
    }
  }

  const messages: Message[] = []

  for (const step of steps) {
    if (step.role === 'user' || step.role === 'assistant') {
      messages.push(historyMessage(step.role, step.content, step.id))
      continue
    }
    if (!step.action && !step.result) continue
    const parts: ContentPart[] = [
      { type: 'text', text: toolResultText(step, loopWarningStepIds.has(step.id)) }
    ]
    const screenshot = imageStepIds.has(step.id) ? computerScreenshotDataUrl(step.result) : null
    if (screenshot) parts.push({ type: 'image_url', image_url: { url: screenshot } })
    messages.push(historyMessage('user', parts, step.id))
  }

  // Keep the original task intent pinned while retaining the most recent trajectory.
  const originalGoalStep = steps.find((step) => step.role === 'user')
  const originalGoal = originalGoalStep
    ? messages.find((message) => message.id === `computer-history-${originalGoalStep.id}`)
    : undefined
  const recent = messages.slice(-MAX_COMPUTER_HISTORY_MESSAGES)
  const trimmed =
    originalGoal && !recent.some((message) => message.id === originalGoal.id)
      ? [
          historyMessage(
            'user',
            `The task began with this user goal; continue pursuing it: ${textContent(originalGoal.content)}`,
            'original-goal'
          ),
          ...recent
        ]
      : recent

  const extraMemories = collectMemorizedFacts(undefined, steps)
  return [
    historyMessage(
      'system',
      computerSystemPrompt(session, { ...options, extraMemories }),
      'system'
    ),
    ...trimmed
  ]
}

function textContent(content: string | ContentPart[]): string {
  if (typeof content === 'string') return content
  return content
    .filter((part): part is Extract<ContentPart, { type: 'text' }> => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

export type ComputerContinuation =
  | { kind: 'model' }
  | { kind: 'waiting_for_user'; question: string }
  | { kind: 'finished' }
  | { kind: 'blocked' }

/** Decide whether a broker result should continue the model loop or pause it. */
export function continuationForResult(result: ComputerActionResult): ComputerContinuation {
  switch (result.status) {
    case 'waiting_for_user':
      return { kind: 'waiting_for_user', question: result.message || 'Please provide more information.' }
    case 'finished':
      return { kind: 'finished' }
    case 'needs_approval':
    case 'error':
    case 'refused':
      return { kind: 'blocked' }
    case 'ok':
      return { kind: 'model' }
  }
}

export type RecoveredComputerPause = {
  approval: {
    approvalId: string
    action: ComputerAction
    message?: string | null
    executionLocation?: ExecutionLocation | null
  } | null
  userQuestion: string | null
}

/** Reconstruct pauses after a component reload from broker-persisted chronology. */
export function recoverComputerPause(steps: ComputerStep[]): RecoveredComputerPause {
  let approval: RecoveredComputerPause['approval'] = null
  let userQuestion: string | null = null
  for (const step of steps) {
    if (step.role === 'user') {
      // A durable answer always resolves the preceding ask_user pause.
      userQuestion = null
      continue
    }
    const result = step.result
    if (!result || !step.action) continue
    if (result.status === 'needs_approval' && result.approval_id) {
      approval = {
        approvalId: result.approval_id,
        action: step.action,
        message: result.message,
        ...(result.execution_location
          ? { executionLocation: result.execution_location }
          : {})
      }
      continue
    }
    if (approval && result.approval_id === approval.approvalId) {
      approval = null
      continue
    }
    // Older brokers did not echo approval_id on the resolved result. A later
    // terminal record for the same action still proves that this pause was spent.
    if (
      approval &&
      !result.approval_id &&
      result.status !== 'needs_approval' &&
      JSON.stringify(step.action) === JSON.stringify(approval.action)
    ) {
      approval = null
    }
    if (result.status === 'waiting_for_user') {
      userQuestion = result.message || (step.action.type === 'ask_user' ? step.action.question : null)
    }
  }
  return { approval, userQuestion }
}

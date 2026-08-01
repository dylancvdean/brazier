import type { ComputerAction, ComputerActionResult, ComputerSession, ComputerStep } from '../api'
import type { ContentPart, Message } from '../types'

/** Matches the documented vLLM recipes' `--limit-mm-per-prompt image=10`. */
export const MAX_COMPUTER_HISTORY_IMAGES = 10
/** Keep enough trajectory for recovery without letting a long task consume its context. */
export const MAX_COMPUTER_HISTORY_MESSAGES = 80

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

/**
 * Fara actions must be parsed from normal completion content. Older local
 * sessions may have been run with thinking enabled, which caused llama.cpp to
 * classify the whole XML response as reasoning content; retain that text as a
 * compatibility fallback instead of silently ending the task.
 */
export function computerModelOutput(responseText: string, reasoningText: string): string {
  return responseText.trim() ? responseText : reasoningText
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

export function computerSystemPrompt(session: Pick<ComputerSession, 'target' | 'permission_mode'>): string {
  return [
    'You are Fara, a computer-use agent specialized in completing web-browser tasks from screenshots.',
    'Use visual evidence from the latest screenshot and the recorded trajectory; do not assume page state.',
    'Return exactly one next action, for example:',
    '<tool_call>{"name":"computer_use","arguments":{"action":"visit_url","url":"https://example.com"}}</tool_call>.',
    'Supported actions are left_click, right_click, double_click, triple_click, mouse_move,',
    'left_click_drag, type, key, scroll, visit_url, web_search, wait,',
    'pause_and_memorize_fact, ask_user_question, and terminate.',
    'Pause with ask_user_question when required personal information is missing or the task is ambiguous.',
    'Before an irreversible action such as submitting, purchasing, sending, signing in, or deleting,',
    'ask for confirmation unless the user explicitly authorized that exact action.',
    'Never invent personal or payment information.',
    `The viewport is 1440x900. Target: ${session.target}. Permission mode: ${session.permission_mode}.`
  ].join(' ')
}

function toolResultText(step: ComputerStep): string {
  const action = computerActionLabel(step.action)
  const result = step.result
  return [
    `Computer action: ${action}.`,
    `Status: ${result?.status || 'unknown'}.`,
    result?.message ? `Result: ${result.message}` : '',
    result?.url ? `URL: ${result.url}` : ''
  ]
    .filter(Boolean)
    .join(' ')
}

/**
 * Turn durable ComputerStep records into the prompt for the next model turn.
 * Broker-written tool records are represented as user messages so the model sees
 * the environment observation without pretending it authored the result.
 */
export function buildComputerHistory(
  session: Pick<ComputerSession, 'target' | 'permission_mode'>,
  steps: ComputerStep[]
): Message[] {
  const imageStepIds = new Set(
    steps
      .filter((step) => Boolean(computerScreenshotDataUrl(step.result)))
      .slice(-MAX_COMPUTER_HISTORY_IMAGES)
      .map((step) => step.id)
  )
  const messages: Message[] = []

  for (const step of steps) {
    if (step.role === 'user' || step.role === 'assistant') {
      messages.push(historyMessage(step.role, step.content, step.id))
      continue
    }
    if (!step.action && !step.result) continue
    const parts: ContentPart[] = [{ type: 'text', text: toolResultText(step) }]
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
  return [historyMessage('system', computerSystemPrompt(session), 'system'), ...trimmed]
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
  approval: { approvalId: string; action: ComputerAction; message?: string | null } | null
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
      approval = { approvalId: result.approval_id, action: step.action, message: result.message }
      continue
    }
    // Older brokers did not echo approval_id on the resolved result. A later
    // record for the same action still proves that this paused action was spent.
    if (
      approval &&
      (result.approval_id === approval.approvalId ||
        JSON.stringify(step.action) === JSON.stringify(approval.action))
    ) {
      approval = null
    }
    if (result.status === 'waiting_for_user') {
      userQuestion = result.message || (step.action.type === 'ask_user' ? step.action.question : null)
    }
  }
  return { approval, userQuestion }
}

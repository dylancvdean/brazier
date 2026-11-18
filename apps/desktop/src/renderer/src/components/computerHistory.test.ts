import { describe, expect, it } from 'vitest'

import type { ComputerActionResult, ComputerSession, ComputerStep } from '../api'
import {
  buildComputerHistory,
  canInteractComputerViewport,
  canSendComputerPrompt,
  computerActionLabel,
  computerSystemPrompt,
  computerTurnStateFromPause,
  continuationForResult,
  IDLE_COMPUTER_TURN_STATE,
  isComputerTurnBusy,
  MAX_COMPUTER_HISTORY_IMAGES,
  MAX_COMPUTER_HISTORY_MESSAGES,
  observationError,
  pendingApprovalForState,
  pendingUserQuestionForState,
  recoverComputerPause,
  transitionComputerTurnState,
  type PendingComputerApproval
} from './computerHistory'

const session: Pick<ComputerSession, 'target' | 'permission_mode' | 'viewport'> = {
  target: 'browser',
  permission_mode: 'ask',
  viewport: { width: 1920, height: 1080, device_pixel_ratio: 1 }
}

function result(status: ComputerActionResult['status'], index?: number): ComputerActionResult {
  return {
    status,
    message: `${status} result`,
    ...(index === undefined
      ? {}
      : { screenshot_base64: String(index), mime_type: 'image/png' })
  }
}

function step(index: number, overrides: Partial<ComputerStep>): ComputerStep {
  return {
    id: `step-${index}`,
    session_id: 'session',
    role: 'tool',
    content: `step ${index}`,
    created_at: String(index),
    action: { type: 'screenshot' },
    result: result('ok', index),
    ...overrides
  }
}

function imagePayloads(history: ReturnType<typeof buildComputerHistory>): string[] {
  return history.flatMap((message) =>
    typeof message.content === 'string'
      ? []
      : message.content
          .filter((part) => part.type === 'image_url')
          .map((part) => part.image_url.url)
  )
}

describe('computer history', () => {
  it('describes parser diagnostics without treating them as executable actions', () => {
    expect(
      computerActionLabel({ type: 'error', error: 'parse_error', raw: '{ malformed' })
    ).toContain('Parse error: parse_error')
  })

  it('uses Fara grounding dimensions and critical-point safety instructions', () => {
    const prompt = computerSystemPrompt(session)
    expect(prompt).toContain('1920x1080')
    expect(prompt).toContain('normalized 0..1000 model-display space')
    expect(prompt).toContain('exactly one next action')
    expect(prompt).toContain('"action":"visit_url"')
    expect(prompt).not.toContain('{...}')
    expect(prompt).toContain('irreversible action')
    expect(prompt).toContain('Never invent personal or payment information')
  })

  it('uses Microsoft Fara guidance but never offers browser-only actions to desktop sessions', () => {
    const prompt = computerSystemPrompt({ ...session, target: 'desktop' })
    expect(prompt).toContain('You are a computer-use agent')
    expect(prompt).toContain('Critical Point')
    expect(prompt).toContain('desktop GUI')
    expect(prompt).toContain('visible desktop GUI')
    expect(prompt).toContain('browser address bar')
    expect(prompt).not.toContain('"action":"visit_url"')
  })

  it('rebuilds durable conversation and keeps only the most recent screenshots', () => {
    const steps = [
      step(0, { role: 'user', content: 'Book a table for two tomorrow.' }),
      step(1, { role: 'assistant', content: '<tool_call>first action</tool_call>' }),
      ...Array.from({ length: 12 }, (_, index) => step(index + 2, {}))
    ]

    const history = buildComputerHistory(session, steps)
    expect(history[0].role).toBe('system')
    expect(history.some((message) => message.content === 'Book a table for two tomorrow.')).toBe(true)
    expect(history.some((message) => message.content === '<tool_call>first action</tool_call>')).toBe(true)
    const images = imagePayloads(history)
    // Fara is trained with only the most recent few screenshots.
    expect(images).toHaveLength(MAX_COMPUTER_HISTORY_IMAGES)
    expect(images[0]).toContain(`base64,${steps.length - MAX_COMPUTER_HISTORY_IMAGES}`)
    expect(images.at(-1)).toContain('base64,13')
  })

  it('honors a configured screenshot budget', () => {
    const steps = [
      step(0, { role: 'user', content: 'Book a table for two tomorrow.' }),
      ...Array.from({ length: 12 }, (_, index) => step(index + 1, {}))
    ]
    const history = buildComputerHistory(session, steps, 1)
    expect(imagePayloads(history)).toHaveLength(1)
    expect(imagePayloads(history)[0]).toContain('base64,12')
  })

  it('pins the original goal while bounding a very long text trajectory', () => {
    const steps = [
      step(0, { role: 'user', content: 'Find the best train, then buy it.' }),
      ...Array.from({ length: MAX_COMPUTER_HISTORY_MESSAGES + 20 }, (_, index) =>
        step(index + 1, {
          role: 'assistant',
          action: null,
          result: null,
          content: `assistant trajectory ${index}`
        })
      )
    ]
    const history = buildComputerHistory(session, steps)

    expect(history).toHaveLength(MAX_COMPUTER_HISTORY_MESSAGES + 2)
    expect(history[1].content).toContain('Find the best train')
    expect(history.at(-1)?.content).toBe(`assistant trajectory ${MAX_COMPUTER_HISTORY_MESSAGES + 19}`)
  })

  it('represents each broker tool record once, with no renderer-synthesized duplicate', () => {
    const history = buildComputerHistory(session, [
      step(1, {
        content: 'Clicked Search',
        action: { type: 'left_click', x: 12, y: 24 },
        result: result('ok')
      })
    ])

    const toolObservations = history.filter(
      (message) =>
        typeof message.content !== 'string' &&
        message.content.some((part) => part.type === 'text' && part.text.startsWith('Computer action:'))
    )
    expect(toolObservations).toHaveLength(1)
    expect(toolObservations[0].content).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'text', text: expect.stringContaining('left click (12, 24)') })
      ])
    )
  })
})

describe('computer continuation state', () => {
  it('fails closed when the initial browser observation is missing or unsuccessful', () => {
    expect(observationError({ status: 'error', message: 'Chromium failed to launch' })).toBe(
      'Chromium failed to launch'
    )
    expect(observationError({ status: 'ok' })).toBe('Could not capture the current computer screenshot.')
    expect(observationError({ status: 'ok', screenshot_base64: 'image' })).toBeNull()
  })

  it('continues from the broker approval result without adding a synthetic prompt', () => {
    expect(continuationForResult(result('ok'))).toEqual({ kind: 'model' })
  })

  it('pauses for ask_user and permits the next durable user answer to resume', () => {
    expect(continuationForResult({ status: 'waiting_for_user', message: 'Which account?' })).toEqual({
      kind: 'waiting_for_user',
      question: 'Which account?'
    })
  })

  it('does not continue after refused, failed, or terminated actions', () => {
    expect(continuationForResult(result('refused'))).toEqual({ kind: 'blocked' })
    expect(continuationForResult(result('error'))).toEqual({ kind: 'blocked' })
    expect(continuationForResult(result('finished'))).toEqual({ kind: 'finished' })
  })

  it('recovers a persisted approval but clears it once its broker result arrives', () => {
    const requested = step(1, {
      action: { type: 'visit_url', url: 'https://example.com' },
      result: {
        status: 'needs_approval',
        approval_id: 'approval-1',
        message: 'Approve navigation',
        execution_location: {
          kind: 'daemon',
          daemon_instance_id: 'daemon-1',
          daemon_display_name: 'Studio Mac',
          platform: 'macos',
          arch: 'aarch64'
        }
      }
    })
    expect(recoverComputerPause([requested])).toEqual({
      approval: {
        approvalId: 'approval-1',
        action: { type: 'visit_url', url: 'https://example.com' },
        message: 'Approve navigation',
        executionLocation: {
          kind: 'daemon',
          daemon_instance_id: 'daemon-1',
          daemon_display_name: 'Studio Mac',
          platform: 'macos',
          arch: 'aarch64'
        }
      },
      userQuestion: null
    })
    expect(
      recoverComputerPause([
        requested,
        step(2, {
          action: { type: 'visit_url', url: 'https://example.com' },
          // Compatibility with brokers that did not echo the approval id.
          result: { status: 'ok' }
        })
      ]).approval
    ).toBeNull()
  })

  it('does not restore ask_user once a later durable user answer exists', () => {
    expect(
      recoverComputerPause([
        step(1, {
          action: { type: 'ask_user', question: 'Which account?' },
          result: { status: 'waiting_for_user', message: 'Which account?' }
        }),
        step(2, { role: 'user', action: null, result: null, content: 'Use my work account.' })
      ]).userQuestion
    ).toBeNull()
  })

  it('preserves awaiting_approval across send() finally cleanup and transitions cleanly on Approve and Deny', () => {
    const approval: PendingComputerApproval = {
      approvalId: 'approval-42',
      action: { type: 'visit_url', url: 'https://bank.example' },
      message: 'Approve navigation',
      executionLocation: {
        kind: 'daemon',
        daemon_instance_id: 'daemon-1',
        daemon_display_name: 'MacBook',
        platform: 'macos',
        arch: 'aarch64'
      },
      resumeLoop: true
    }

    let state = IDLE_COMPUTER_TURN_STATE
    state = transitionComputerTurnState(state, { type: 'start_turn' })
    expect(state).toEqual({ kind: 'running' })
    expect(isComputerTurnBusy(state)).toBe(true)
    expect(canSendComputerPrompt(state)).toBe(false)

    // Model loop hits approval_required -> transitions to awaiting_approval.
    state = transitionComputerTurnState(state, { type: 'require_approval', approval })
    expect(state.kind).toBe('awaiting_approval')

    // CRITICAL: send()'s finally block dispatches `turn_finally` and MUST NOT clobber awaiting_approval!
    state = transitionComputerTurnState(state, { type: 'turn_finally' })
    expect(state).toEqual({ kind: 'awaiting_approval', approval })
    expect(pendingApprovalForState(state)).toEqual(approval)
    expect(canSendComputerPrompt(state)).toBe(false)

    // Approve branch: awaiting_approval -> resolving_approval -> running -> turn_finished -> idle.
    const resolvingApprove = transitionComputerTurnState(state, {
      type: 'begin_resolve_approval',
      approve: true
    })
    expect(resolvingApprove).toEqual({
      kind: 'resolving_approval',
      approval,
      approve: true
    })
    expect(isComputerTurnBusy(resolvingApprove)).toBe(true)

    const resumed = transitionComputerTurnState(resolvingApprove, {
      type: 'approval_approved_continue'
    })
    expect(resumed).toEqual({ kind: 'running' })
    expect(transitionComputerTurnState(resumed, { type: 'turn_finished' })).toEqual({
      kind: 'idle'
    })

    // Deny branch: awaiting_approval -> resolving_approval -> approval_denied -> idle.
    const resolvingDeny = transitionComputerTurnState(state, {
      type: 'begin_resolve_approval',
      approve: false
    })
    expect(resolvingDeny).toEqual({
      kind: 'resolving_approval',
      approval,
      approve: false
    })
    expect(transitionComputerTurnState(resolvingDeny, { type: 'approval_denied' })).toEqual({
      kind: 'idle'
    })
  })

  it('preserves awaiting_user and stopped across finally and viewport interactions', () => {
    let state = transitionComputerTurnState(IDLE_COMPUTER_TURN_STATE, { type: 'start_turn' })
    state = transitionComputerTurnState(state, {
      type: 'require_user',
      question: 'Which profile should I choose?'
    })
    state = transitionComputerTurnState(state, { type: 'turn_finally' })
    expect(pendingUserQuestionForState(state)).toBe('Which profile should I choose?')
    expect(canSendComputerPrompt(state)).toBe(true)
    expect(canInteractComputerViewport(state)).toBe(true)

    // Viewport interaction while awaiting_user restores the question afterward.
    const interacting = transitionComputerTurnState(state, { type: 'start_interaction' })
    expect(interacting.kind).toBe('interacting')
    const afterInteract = transitionComputerTurnState(interacting, { type: 'end_interaction' })
    expect(pendingUserQuestionForState(afterInteract)).toBe('Which profile should I choose?')

    // Stop transitions to stopped and permits starting a fresh turn later.
    const stopped = transitionComputerTurnState(state, { type: 'stop' })
    expect(stopped).toEqual({ kind: 'stopped' })
    expect(transitionComputerTurnState(stopped, { type: 'turn_finally' })).toEqual({
      kind: 'stopped'
    })
    expect(canSendComputerPrompt(stopped)).toBe(true)
    expect(
      computerTurnStateFromPause({
        approval: {
          approvalId: 'rec-1',
          action: { type: 'left_click', x: 100, y: 200 }
        },
        userQuestion: null
      }).kind
    ).toBe('awaiting_approval')
  })

  it('injects session.memories and memorize step facts into buildComputerHistory', () => {
    const history = buildComputerHistory(
      { ...session, memories: ['Order ID is #99412'] },
      [
        step(0, { role: 'user', action: null, result: null, content: 'Check order total.' }),
        step(1, {
          action: { type: 'memorize', fact: 'Subtotal is $48.00' },
          result: { status: 'ok', message: 'Memorized' }
        })
      ]
    )
    expect(history[0].content).toContain('Memorized facts:')
    expect(history[0].content).toContain('- Order ID is #99412')
    expect(history[0].content).toContain('- Subtotal is $48.00')
  })

  it('notes screen_changed === false and injects a loop-breaker warning after 2+ unchanged repeats', () => {
    const history = buildComputerHistory(session, [
      step(0, { role: 'user', action: null, result: null, content: 'Click Checkout' }),
      step(1, {
        action: { type: 'left_click', x: 500, y: 300 },
        result: {
          status: 'ok',
          screenshot_base64: 'same-frame',
          mime_type: 'image/png',
          url: 'https://shop.example/cart',
          screen_changed: false
        } as ComputerActionResult
      }),
      step(2, {
        action: { type: 'left_click', x: 501, y: 299 },
        result: {
          status: 'ok',
          screenshot_base64: 'same-frame',
          mime_type: 'image/png',
          url: 'https://shop.example/cart',
          screen_changed: false
        } as ComputerActionResult
      })
    ])

    const firstObs = history[2].content
    const secondObs = history[3].content
    expect(JSON.stringify(firstObs)).toContain(
      'Note: The screen and URL did not visibly change after this action.'
    )
    expect(JSON.stringify(firstObs)).not.toContain('Loop warning:')
    expect(JSON.stringify(secondObs)).toContain('Loop warning:')
  })

  it('adapts system prompt instructions for openai-tools, ui-tars, and pixel coordinate spaces', () => {
    const toolsPrompt = computerSystemPrompt(session, {
      dialect: 'openai-tools',
      coordinateSpace: 'pixels'
    })
    expect(toolsPrompt).toContain('Invoke the `computer_use` tool')
    expect(toolsPrompt).toContain('pixel coordinates matching 1920x1080')

    const uiTarsPrompt = computerSystemPrompt(session, {
      dialect: 'ui-tars',
      coordinateSpace: 'norm1000'
    })
    expect(uiTarsPrompt).toContain("Action: click(start_box='(500,320)')")
  })
})

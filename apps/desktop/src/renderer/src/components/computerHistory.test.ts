import { describe, expect, it } from 'vitest'

import type { ComputerActionResult, ComputerSession, ComputerStep } from '../api'
import {
  buildComputerHistory,
  computerModelOutput,
  computerSystemPrompt,
  continuationForResult,
  MAX_COMPUTER_HISTORY_IMAGES,
  MAX_COMPUTER_HISTORY_MESSAGES,
  observationError,
  recoverComputerPause
} from './computerHistory'

const session: Pick<ComputerSession, 'target' | 'permission_mode'> = {
  target: 'browser',
  permission_mode: 'ask'
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
  it('uses reasoning output only as a compatibility fallback for old Fara sessions', () => {
    expect(computerModelOutput('<tool_call>content</tool_call>', '<tool_call>reasoning</tool_call>')).toBe(
      '<tool_call>content</tool_call>'
    )
    expect(computerModelOutput('', '<tool_call>reasoning</tool_call>')).toBe(
      '<tool_call>reasoning</tool_call>'
    )
  })

  it('uses Fara grounding dimensions and critical-point safety instructions', () => {
    const prompt = computerSystemPrompt(session)
    expect(prompt).toContain('1440x900')
    expect(prompt).toContain('exactly one next action')
    expect(prompt).toContain('"action":"visit_url"')
    expect(prompt).not.toContain('{...}')
    expect(prompt).toContain('irreversible action')
    expect(prompt).toContain('Never invent personal or payment information')
  })

  it('rebuilds durable conversation and keeps the newest ten screenshots', () => {
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
    expect(images).toHaveLength(MAX_COMPUTER_HISTORY_IMAGES)
    expect(images[0]).toContain('base64,4')
    expect(images.at(-1)).toContain('base64,13')
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
      result: { status: 'needs_approval', approval_id: 'approval-1', message: 'Approve navigation' }
    })
    expect(recoverComputerPause([requested])).toEqual({
      approval: {
        approvalId: 'approval-1',
        action: { type: 'visit_url', url: 'https://example.com' },
        message: 'Approve navigation'
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
})

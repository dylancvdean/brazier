import { describe, expect, it } from 'vitest'

import { classifyUtterance, isControlIntent } from './interruption'

describe('classifyUtterance', () => {
  it('separates silencing the voice from abandoning the work', () => {
    for (const text of [
      'Stop talking.',
      'Be quiet',
      "That's enough",
      'stop',
      'Hold on',
      'Shut up!'
    ]) {
      expect(classifyUtterance(text, { taskActive: true })).toBe('stop_speaking')
    }
    for (const text of [
      'Never mind, cancel that.',
      'Cancel that',
      'Forget it',
      'Stop the build',
      'Abort the run',
      "Don't bother",
      'Stop working on that'
    ]) {
      expect(classifyUtterance(text, { taskActive: true })).toBe('cancel_task')
    }
  })

  it('recognizes a correction', () => {
    for (const text of [
      'No, I meant the Vulkan backend.',
      'Actually, check the docs instead.',
      "That's not right",
      'Not the Metal one, the Vulkan one'
    ]) {
      expect(classifyUtterance(text, { taskActive: true })).toBe('correction')
    }
  })

  it('treats a question during a task as a follow-up', () => {
    for (const text of [
      'While that runs, what time is the meeting?',
      "How's it going?",
      'Is it done yet?',
      'Any progress?'
    ]) {
      expect(classifyUtterance(text, { taskActive: true })).toBe('follow_up')
    }
  })

  it('does not turn an ordinary question into a control', () => {
    // The point of the classifier: a question that merely mentions stopping
    // must not silently cancel a task.
    expect(classifyUtterance('Why did the build stop working last week?')).toBe('new_request')
    expect(classifyUtterance('Can you explain how cancellation works?')).toBe('new_request')
    expect(classifyUtterance('What cancels a download?')).toBe('new_request')
  })

  it('classifies an unrelated request with nothing in flight', () => {
    expect(classifyUtterance('What time is the standup?')).toBe('new_request')
    expect(classifyUtterance('What time is the standup?', { taskActive: true })).toBe('follow_up')
  })

  it('marks only the two controls as controls', () => {
    expect(isControlIntent('stop_speaking')).toBe(true)
    expect(isControlIntent('cancel_task')).toBe(true)
    expect(isControlIntent('correction')).toBe(false)
    expect(isControlIntent('follow_up')).toBe(false)
    expect(isControlIntent('new_request')).toBe(false)
  })

  it('handles an empty transcript without throwing', () => {
    expect(classifyUtterance('   ')).toBe('follow_up')
  })
})

import { describe, expect, it } from 'vitest'

import {
  buildPersonaPlexHandoffPrompt,
  handoffReplaysAudio,
  handoffRestartsProcess,
  type PersonaPlexHandoffRequest
} from './personaplexHandoff'

const request: PersonaPlexHandoffRequest = {
  correlationId: 'turn-1',
  utteranceId: 'utt-1',
  userText: 'Which test failed?',
  resultText: 'The oggOpus test failed because the final page was not flushed.',
  context: {
    personaInstructions: 'You are warm, concise, and technically precise.',
    behavioralRules: [],
    conversationSummary: '',
    recentTurns: [],
    activeTaskSummary: '',
    currentStatus: 'completed',
    responseDirective: ''
  }
}

describe('PersonaPlex handoff experiments', () => {
  it('formats direct injection as an explicit answer instruction', () => {
    const prompt = buildPersonaPlexHandoffPrompt('reconnect-direct-replay', request)
    expect(prompt).toContain('The user asked: Which test failed?')
    expect(prompt).toContain('background assistant checked')
    expect(prompt).toContain('oggOpus')
  })

  it('formats service injection like PersonaPlex training prompts', () => {
    const prompt = buildPersonaPlexHandoffPrompt('reconnect-service-replay', request)
    expect(prompt).toContain('You work for Brazier')
    expect(prompt).toContain('Information:')
    expect(prompt).toContain('checked and confirmed')
  })

  it('classifies replay and full-process restart independently', () => {
    expect(handoffReplaysAudio('reconnect-service-replay')).toBe(true)
    expect(handoffReplaysAudio('reconnect-service-no-replay')).toBe(false)
    expect(handoffRestartsProcess('reconnect-service-replay')).toBe(false)
    expect(handoffRestartsProcess('restart-service-replay')).toBe(true)
  })

  it('bounds large background answers before placing them in the prompt', () => {
    const prompt = buildPersonaPlexHandoffPrompt('reconnect-service-replay', {
      ...request,
      resultText: 'result '.repeat(1000)
    })
    expect(prompt.length).toBeLessThan(1500)
    expect(prompt).toContain('…')
  })
})

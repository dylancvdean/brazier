import { describe, expect, it } from 'vitest'

import { DEFAULT_INTEGRATION_CONFIG } from './config'
import type { ConversationMessage, MessageSource, TaskState } from './types'
import {
  VOICE_BEHAVIORAL_RULES,
  buildVoiceContext,
  describeTask,
  renderVoicePrompt,
  summarizeForVoice
} from './voiceContext'

let counter = 0
function message(
  role: ConversationMessage['role'],
  content: string,
  overrides: Partial<ConversationMessage> = {}
): ConversationMessage {
  counter += 1
  const source: MessageSource = role === 'user' ? 'user_voice' : 'assistant_agent'
  return {
    id: `msg-${counter}`,
    conversationId: 'conv-1',
    role,
    source,
    content,
    createdAt: new Date(counter * 1000).toISOString(),
    status: 'final',
    ...overrides
  }
}

const task: TaskState = {
  correlationId: 'turn-1',
  label: 'Fix the failing test',
  status: 'running',
  activeTool: 'shell',
  confirmedResults: ['1 test failed', 'oggOpus.test.ts:42'],
  updatedAt: 0
}

const config = DEFAULT_INTEGRATION_CONFIG

describe('buildVoiceContext', () => {
  it('bounds the context instead of sending the whole history', () => {
    const history = Array.from({ length: 40 }, (_, index) =>
      message(index % 2 === 0 ? 'user' : 'assistant', `Turn ${index}`)
    )
    const context = buildVoiceContext({
      personaInstructions: 'You are a terse engineer.',
      conversationSummary: 'x'.repeat(5000),
      messages: history,
      task,
      config
    })

    expect(context.recentTurns).toHaveLength(config.voiceContextRecentTurnLimit)
    expect(context.recentTurns.at(-1)?.content).toBe('Turn 39')
    expect(context.conversationSummary.length).toBeLessThanOrEqual(
      config.voiceContextSummaryLimitChars
    )
    expect(context.behavioralRules[0]).toContain('only audible voice')
    expect(context.behavioralRules[0]).toContain('Lightweight turns')
    expect(context.behavioralRules).toEqual(
      expect.arrayContaining([...VOICE_BEHAVIORAL_RULES])
    )
  })

  it('omits withdrawn turns and the voice’s own renderings', () => {
    const context = buildVoiceContext({
      personaInstructions: 'Persona',
      conversationSummary: '',
      messages: [
        message('user', 'Check the docs', { status: 'superseded' }),
        message('user', 'Check the Vulkan backend'),
        message('assistant', 'Spoken copy', { source: 'assistant_voice' }),
        message('assistant', 'Vulkan builds cleanly')
      ],
      task: null,
      config
    })

    const contents = context.recentTurns.map((turn) => turn.content)
    expect(contents).toEqual(['Check the Vulkan backend', 'Vulkan builds cleanly'])
  })

  it('truncates one very long turn rather than dropping the rest', () => {
    const context = buildVoiceContext({
      personaInstructions: 'Persona',
      conversationSummary: '',
      messages: [message('user', 'word '.repeat(500)), message('assistant', 'Short answer')],
      task: null,
      config
    })
    expect(context.recentTurns).toHaveLength(2)
    expect(context.recentTurns[0].content.length).toBeLessThan(420)
    expect(context.recentTurns[0].content.endsWith('…')).toBe(true)
  })

  it('describes task state from structured fields only', () => {
    expect(describeTask(task)).toBe(
      'Fix the failing test — running · running shell · confirmed: 1 test failed; oggOpus.test.ts:42'
    )
  })
})

describe('renderVoicePrompt', () => {
  it('leads with the directive and always ends with the rules', () => {
    const context = buildVoiceContext({
      personaInstructions: 'You are a terse engineer.',
      conversationSummary: 'Fixing a failing test.',
      messages: [message('user', 'What broke?')],
      task,
      responseDirective: 'One test failed: oggOpus.',
      currentStatus: 'running',
      config
    })
    const prompt = renderVoicePrompt(context)

    expect(prompt.startsWith('Say this now:\nOne test failed: oggOpus.')).toBe(true)
    expect(prompt).toContain('Active task: Fix the failing test')
    expect(prompt).toContain('You are a terse engineer.')
    expect(prompt.indexOf('Rules:')).toBeGreaterThan(prompt.indexOf('You are a terse engineer.'))
    for (const rule of VOICE_BEHAVIORAL_RULES) expect(prompt).toContain(rule)
  })

  it('leaves out sections it has nothing for', () => {
    const prompt = renderVoicePrompt(
      buildVoiceContext({
        personaInstructions: 'Persona',
        conversationSummary: '',
        messages: [],
        task: null,
        config
      })
    )
    expect(prompt).not.toContain('Say this now')
    expect(prompt).not.toContain('Active task')
    expect(prompt).not.toContain('Recent turns')
    expect(prompt).toContain('Rules:')
  })

  it('makes PersonaPlex the only audible voice for integrated turns', () => {
    const prompt = renderVoicePrompt(
      buildVoiceContext({
        personaInstructions: 'Warm and direct.',
        conversationSummary: '',
        messages: [],
        task: null,
        config
      })
    )

    expect(prompt).toContain('only audible voice')
    expect(prompt).toContain('Lightweight turns may stay entirely with you')
    expect(prompt).toContain('need files, tools, or checked facts')
  })

  it('does not constrain PersonaPlex to handoffs when used on its own', () => {
    const prompt = renderVoicePrompt(
      buildVoiceContext({
        personaInstructions: 'Warm and direct.',
        conversationSummary: '',
        messages: [],
        task: null,
        config: { ...config, voiceSessionTarget: 'neither' }
      })
    )

    expect(prompt).not.toContain('only audible voice')
  })
})

describe('summarizeForVoice', () => {
  it('prefers a summary the agent already produced', () => {
    const summary = summarizeForVoice([message('user', 'Anything')], {
      limitChars: 400,
      agentSummary: 'Compacted: replaced the Opus muxer.'
    })
    expect(summary).toBe('Compacted: replaced the Opus muxer.')
  })

  it('keeps the goal, the last answer, and confirmed results', () => {
    const summary = summarizeForVoice(
      [
        message('user', 'Set up the voice mode'),
        message('assistant', 'Started the PersonaPlex server'),
        message('user', 'Now make it talk to the agent'),
        message('assistant', 'Wired the coordinator in')
      ],
      { limitChars: 1200, task }
    )

    expect(summary).toContain('Started with: Set up the voice mode')
    expect(summary).toContain('Current goal: Now make it talk to the agent')
    expect(summary).toContain('Last answer: Wired the coordinator in')
    expect(summary).toContain('1 test failed')
  })

  it('leaves out unverified voice output', () => {
    const summary = summarizeForVoice(
      [
        message('user', 'Did it pass?'),
        message('assistant', 'Yes it passed!', { source: 'assistant_voice' })
      ],
      { limitChars: 400 }
    )
    expect(summary).not.toContain('Yes it passed')
  })

  it('respects its budget', () => {
    const summary = summarizeForVoice(
      [message('user', 'q '.repeat(400)), message('assistant', 'a '.repeat(400))],
      { limitChars: 200 }
    )
    expect(summary.length).toBeLessThanOrEqual(200)
  })
})

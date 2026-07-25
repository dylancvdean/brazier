/**
 * Coordinator behaviour: the basic, interruption, session, and failure flows the
 * integration has to guarantee.
 */

import { describe, expect, it } from 'vitest'

import { SessionCoordinator } from './coordinator'
import { DEFAULT_INTEGRATION_CONFIG, type IntegrationConfig } from './config'
import {
  FakeAgent,
  FakeChat,
  FakeResponder,
  FakeVoice,
  harnessClock,
  sequentialIds
} from './testFakes'

function harness(overrides: Partial<IntegrationConfig> = {}) {
  const clock = harnessClock(1_000)
  const chat = new FakeChat()
  const agent = new FakeAgent()
  const voice = new FakeVoice(clock.now)
  const responder = new FakeResponder()
  const logs: string[] = []
  const coordinator = new SessionCoordinator({
    chat,
    agent,
    voice,
    responder,
    now: clock.now,
    newId: sequentialIds(),
    log: (record) => logs.push(`${record.eventType}:${record.correlationId}`),
    config: { ...DEFAULT_INTEGRATION_CONFIG, voiceEnabled: true, ...overrides }
  })
  return { clock, chat, agent, voice, responder, coordinator, logs }
}

/**
 * Attached, with a live voice session pointed at the agent. Most of what
 * follows is about agent-owned turns; the chat and neither destinations say so
 * explicitly.
 */
async function live(overrides: Partial<IntegrationConfig> = {}) {
  const context = harness({ voiceSessionTarget: 'agent', ...overrides })
  await context.coordinator.attach('conv-1')
  await context.coordinator.startVoiceSession()
  return context
}

function speak(voice: FakeVoice, utteranceId: string, text: string): void {
  voice.emit({ type: 'userTranscriptFinal', utteranceId, text })
}

describe('basic flows', () => {
  it('answers a spoken question through the agent and shows it once', async () => {
    const { chat, agent, voice } = await live()

    speak(voice, 'utt-1', 'Which test is failing?')
    await Promise.resolve()
    expect(agent.submitted).toHaveLength(1)
    const correlationId = agent.submitted[0].correlationId

    agent.completeRun(correlationId, 'The voice adapter test is failing.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    // The transcript and the answer are both in the shared conversation, and the
    // answer appears exactly once even though it was also spoken.
    expect(chat.messages.map((message) => message.source)).toEqual([
      'user_voice',
      'assistant_agent'
    ])
    expect(chat.assistantMessages()).toHaveLength(1)
    expect(voice.authoritative().map((request) => request.text)).toEqual([
      'The voice adapter test is failing.'
    ])
    // The spoken rendering is linked to the stored answer, not stored again.
    expect(voice.authoritative()[0].correlationId).toBe(correlationId)
    expect(chat.assistantMessages()[0].correlationId).toBe(correlationId)
  })

  it('speaks tool-backed results only after the agent supplies them', async () => {
    const { coordinator, agent, voice } = await live()
    speak(voice, 'utt-1', 'Run the tests and tell me what broke.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId

    agent.emit({ type: 'runStarted', correlationId })
    agent.emit({ type: 'toolStarted', correlationId, toolCallId: 'c1', tool: 'shell' })
    await Promise.resolve()
    // Nothing factual has been spoken yet: only an acknowledgment and a cue.
    expect(voice.authoritative()).toHaveLength(0)
    expect(voice.spoken.every((request) => request.kind !== 'authoritative')).toBe(true)

    agent.emit({
      type: 'toolCompleted',
      correlationId,
      toolCallId: 'c1',
      tool: 'shell',
      outcome: '1 test failed'
    })
    expect(coordinator.snapshot().task?.confirmedResults).toEqual(['1 test failed'])
    expect(voice.authoritative()).toHaveLength(0)

    agent.emit({ type: 'responseFinal', correlationId, text: 'One test failed: oggOpus.' })
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(voice.authoritative()[0].text).toBe('One test failed: oggOpus.')
  })

  it('routes typed input to the same agent session without speaking it', async () => {
    const { coordinator, agent, voice, chat } = await live()

    await coordinator.submitText('Check the Vulkan backend.')
    const correlationId = agent.submitted[0].correlationId
    agent.completeRun(correlationId, 'Vulkan builds cleanly.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(agent.submitted).toHaveLength(1)
    expect(chat.messages[0].source).toBe('user_text')
    // Default: text-originated answers are shown, not spoken.
    expect(voice.authoritative()).toHaveLength(0)
    expect(chat.assistantMessages()).toHaveLength(1)
  })

  it('speaks text-originated answers when that is switched on', async () => {
    const { coordinator, agent, voice } = await live({ speakTextOriginatedResponses: true })
    await coordinator.submitText('Summarize the diff.')
    agent.completeRun(agent.submitted[0].correlationId, 'Three files changed.')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(voice.authoritative()).toHaveLength(1)
  })

  it('uses one agent session for voice and text in the same conversation', async () => {
    const { coordinator, agent, voice } = await live()
    await coordinator.submitText('First, by keyboard.')
    agent.completeRun(agent.submitted[0].correlationId, 'Done.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    speak(voice, 'utt-1', 'Now by voice.')
    await Promise.resolve()
    expect(agent.submitted).toHaveLength(2)
    expect(new Set(agent.submitted.map(() => agent.attachedSessionId())).size).toBe(1)
  })

  it('falls back to the chat responder when no agent session is bound', async () => {
    const context = harness()
    context.agent.sessionId = null
    await context.coordinator.attach('conv-1')
    await context.coordinator.submitText('Hello there.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(context.agent.submitted).toHaveLength(0)
    expect(context.chat.assistantMessages()[0].source).toBe('assistant_chat')
    expect(context.coordinator.snapshot().responses[0].owner).toBe('chat')
  })
})

describe('what the voice session is connected to', () => {
  it('sends spoken turns to the agent when that is the destination', async () => {
    const { coordinator, agent, voice, chat } = await live({ voiceSessionTarget: 'agent' })
    speak(voice, 'utt-1', 'With a task bound.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(agent.submitted).toHaveLength(1)
    agent.completeRun(agent.submitted[0].correlationId, 'Agent answered.')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(chat.assistantMessages().at(-1)?.source).toBe('assistant_agent')
    expect(coordinator.snapshot().responses[0].owner).toBe('agent')
  })

  it('keeps spoken turns on the chat model even while a task is bound', async () => {
    const { coordinator, agent, voice, chat } = await live({ voiceSessionTarget: 'chat' })
    speak(voice, 'utt-1', 'Answer this yourself.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(agent.submitted).toHaveLength(0)
    expect(chat.assistantMessages()[0].source).toBe('assistant_chat')
    expect(coordinator.snapshot().responses[0].owner).toBe('chat')
  })

  it('refuses an agent-only turn rather than quietly answering from chat', async () => {
    const { coordinator, agent, voice, chat } = await live({ voiceSessionTarget: 'agent' })
    agent.sessionId = null
    await coordinator.attach('conv-1')
    speak(voice, 'utt-1', 'Run the tests.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    // The turn is kept and marked, not answered by a model without the workspace.
    expect(chat.assistantMessages()).toHaveLength(0)
    expect(chat.messages.at(-1)?.status).toBe('failed')
    expect(chat.statuses.some((status) => status?.includes('no agent session'))).toBe(true)
  })

  it('records and invokes nothing when connected to neither', async () => {
    const { coordinator, agent, voice, chat } = await live({ voiceSessionTarget: 'neither' })
    speak(voice, 'utt-1', 'Just talking.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(chat.messages).toHaveLength(0)
    expect(agent.submitted).toHaveLength(0)
    // PersonaPlex is the only voice there is, so its audio stays audible.
    expect(voice.modelAudioEnabled).toBe(true)
    expect(voice.spoken).toHaveLength(0)
  })

  it('silences PersonaPlex whenever the coordinator delivers answers', async () => {
    const { coordinator, voice } = await live({ voiceSessionTarget: 'neither' })
    const base = { ...DEFAULT_INTEGRATION_CONFIG, voiceEnabled: true }
    expect(voice.modelAudioEnabled).toBe(true)

    coordinator.setConfig({ ...base, voiceSessionTarget: 'chat' })
    expect(voice.modelAudioEnabled).toBe(false)

    coordinator.setConfig({ ...base, voiceSessionTarget: 'neither' })
    expect(voice.modelAudioEnabled).toBe(true)
  })

  it('leaves PersonaPlex audible when this host cannot speak answers', async () => {
    const context = harness({ voiceSessionTarget: 'chat' })
    context.voice.speakable = false
    await context.coordinator.attach('conv-1')
    await context.coordinator.startVoiceSession()
    expect(context.voice.modelAudioEnabled).toBe(true)
  })

  it('still routes typed turns to the agent when voice is chat-only', async () => {
    const { coordinator, agent, voice } = await live({ voiceSessionTarget: 'chat' })
    await coordinator.submitText('Typed, so the agent takes it.')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(agent.submitted).toHaveLength(1)
    void voice
  })
})

describe('interruption flows', () => {
  it('stops audio on barge-in but lets the agent keep working', async () => {
    const { coordinator, agent, voice } = await live()
    speak(voice, 'utt-1', 'Explain the sandbox.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.completeRun(correlationId, 'Seatbelt on macOS, Bubblewrap on Linux.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    voice.emit({ type: 'userSpeechStarted', utteranceId: 'utt-2' })
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(voice.stopped).toContain(correlationId)
    expect(agent.cancelled).toHaveLength(0)
    expect(coordinator.metrics().agentTasksCancelledByInterruption).toBe(0)
    // The answer stays in the chat even though its delivery was cut short.
    expect(coordinator.snapshot().messages.at(-1)?.status).toBe('final')
  })

  it('"stop talking" silences the voice and leaves the task running', async () => {
    const { coordinator, agent, voice } = await live()
    speak(voice, 'utt-1', 'Start the long job.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.emit({ type: 'runStarted', correlationId })
    await new Promise((resolve) => setTimeout(resolve, 0))

    speak(voice, 'utt-2', 'Stop talking.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(agent.cancelled).toHaveLength(0)
    expect(coordinator.snapshot().activeCorrelationId).toBe(correlationId)
    // A control is not a question: it was not submitted as a turn.
    expect(agent.submitted).toHaveLength(1)
  })

  it('"never mind, cancel that" cancels the correlated agent task', async () => {
    const { coordinator, agent, voice, chat } = await live()
    speak(voice, 'utt-1', 'Rebuild everything from source.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.emit({ type: 'runStarted', correlationId })
    await new Promise((resolve) => setTimeout(resolve, 0))

    speak(voice, 'utt-2', 'Never mind, cancel that.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(agent.cancelled).toEqual([correlationId])
    expect(coordinator.snapshot().activeCorrelationId).toBeNull()
    expect(chat.messages.at(-1)?.role).toBe('system')
  })

  it('submits a correction as a follow-up and supersedes queued turns', async () => {
    const { coordinator, agent, voice, chat } = await live()
    speak(voice, 'utt-1', 'Check the Metal backend.')
    await Promise.resolve()
    const first = agent.submitted[0].correlationId
    agent.emit({ type: 'runStarted', correlationId: first })
    await new Promise((resolve) => setTimeout(resolve, 0))

    speak(voice, 'utt-2', 'And also check the docs.')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(coordinator.snapshot().queue).toHaveLength(1)

    speak(voice, 'utt-3', 'No, I meant the Vulkan backend.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    // The running turn is untouched; the queued one is superseded, not dropped.
    expect(agent.cancelled).toHaveLength(0)
    expect(coordinator.snapshot().activeCorrelationId).toBe(first)
    const queuedMessage = chat.messages.find((message) => message.content.includes('the docs'))
    expect(queuedMessage?.status).toBe('superseded')
    expect(chat.messages.some((message) => message.content.includes('Vulkan'))).toBe(true)

    // History is intact: every utterance is still recorded.
    expect(chat.messages.filter((message) => message.role === 'user')).toHaveLength(3)
  })

  it('preserves both messages when text arrives while the voice is speaking', async () => {
    const { coordinator, agent, voice, chat } = await live()
    speak(voice, 'utt-1', 'Describe the policy layer.')
    await Promise.resolve()
    const first = agent.submitted[0].correlationId
    agent.completeRun(first, 'Every call is judged by agent_policy.')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(coordinator.snapshot().speakingCorrelationId).toBe(first)

    await coordinator.submitText('And the approvals?')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(agent.submitted).toHaveLength(2)
    expect(chat.messages.filter((message) => message.role === 'user')).toHaveLength(2)
  })

  it('ignores a stale cancellation once a newer turn is running', async () => {
    const { coordinator, agent, voice } = await live()
    speak(voice, 'utt-1', 'First question.')
    await Promise.resolve()
    const first = agent.submitted[0].correlationId
    agent.completeRun(first, 'First answer.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    speak(voice, 'utt-2', 'Second question.')
    await new Promise((resolve) => setTimeout(resolve, 0))
    const second = agent.submitted[1].correlationId
    agent.emit({ type: 'runStarted', correlationId: second })

    // A cancellation aimed at the finished turn must not touch the live one.
    expect(await coordinator.cancelAgentTask(first)).toBe(false)
    expect(agent.cancelled).toHaveLength(0)
    expect(coordinator.snapshot().activeCorrelationId).toBe(second)

    expect(await coordinator.cancelAgentTask(second)).toBe(true)
    expect(agent.cancelled).toEqual([second])
  })

  it('keeps the three cancellation controls separate', async () => {
    const { coordinator, agent, voice, chat } = await live()
    speak(voice, 'utt-1', 'Tell me about the sandbox.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.completeRun(correlationId, 'Seatbelt and Bubblewrap.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    // Muting the voice must not end anything else.
    await coordinator.cancelVoiceOutput()
    expect(voice.stopped).toContain(correlationId)
    expect(agent.cancelled).toHaveLength(0)
    // Cancelling delivery after the answer was stored keeps it in the chat.
    expect(chat.assistantMessages()).toHaveLength(1)
    expect(chat.assistantMessages()[0].status).toBe('final')
  })

  it('cancels the agent from an interruption only when configured to', async () => {
    const { coordinator, agent, voice } = await live({ interruptCancelsAgent: true })
    speak(voice, 'utt-1', 'Kick off the build.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.emit({ type: 'runStarted', correlationId })
    voice.emit({ type: 'speechStarted', correlationId })
    await new Promise((resolve) => setTimeout(resolve, 0))

    voice.emit({ type: 'userSpeechStarted', utteranceId: 'utt-2' })
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(agent.cancelled).toEqual([correlationId])
    expect(coordinator.metrics().agentTasksCancelledByInterruption).toBe(1)
  })
})

describe('queueing', () => {
  it('runs one agent turn at a time and reports queued state', async () => {
    const { coordinator, agent, voice, chat } = await live()
    speak(voice, 'utt-1', 'One.')
    await Promise.resolve()
    speak(voice, 'utt-2', 'Two.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(agent.submitted).toHaveLength(1)
    expect(coordinator.snapshot().queue).toHaveLength(1)
    expect(chat.queued).toHaveLength(1)

    agent.completeRun(agent.submitted[0].correlationId, 'Answer one.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    // The queued turn starts on its own once the first finishes.
    expect(agent.submitted).toHaveLength(2)
    expect(coordinator.snapshot().queue).toHaveLength(0)
  })
})

describe('session flows', () => {
  it('renews the voice session at its limit without restarting the agent', async () => {
    const { coordinator, agent, voice, clock } = await live()
    speak(voice, 'utt-1', 'Remember this task.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.completeRun(correlationId, 'Noted: the failing oggOpus test.')
    await new Promise((resolve) => setTimeout(resolve, 0))
    voice.emit({ type: 'speechCompleted', correlationId })

    clock.advance(DEFAULT_INTEGRATION_CONFIG.voiceSessionMaxDurationMs + 1)
    await coordinator.tick()

    expect(voice.sessions).toHaveLength(2)
    expect(voice.ended).toBe(1)
    expect(coordinator.metrics().voiceSessionRenewals).toBe(1)
    // The agent session and the conversation are untouched.
    expect(agent.cancelled).toHaveLength(0)
    expect(coordinator.snapshot().agentSessionId).toBe('agent-1')
    expect(coordinator.snapshot().messages).toHaveLength(2)
    expect(coordinator.events.eventsOfType('VOICE_SESSION_RENEWED')).toHaveLength(1)

    // The confirmed result survived into the new session's bounded context.
    const seeded = voice.contexts.at(-1)
    expect(seeded?.conversationSummary).toContain('oggOpus')
    expect(seeded?.recentTurns.length).toBeLessThanOrEqual(
      DEFAULT_INTEGRATION_CONFIG.voiceContextRecentTurnLimit
    )
  })

  it('defers renewal until a safe boundary', async () => {
    const { coordinator, agent, voice } = await live()
    speak(voice, 'utt-1', 'Start something long.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.emit({ type: 'runStarted', correlationId })
    await new Promise((resolve) => setTimeout(resolve, 0))

    await coordinator.requestRenewal('context size')
    expect(voice.sessions).toHaveLength(1)

    agent.emit({ type: 'responseFinal', correlationId, text: 'Finished.' })
    await new Promise((resolve) => setTimeout(resolve, 0))
    voice.emit({ type: 'speechCompleted', correlationId })
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(voice.sessions).toHaveLength(2)
  })

  it('degrades to text when the voice session dies, keeping the agent', async () => {
    const { coordinator, agent, voice, chat } = await live()
    speak(voice, 'utt-1', 'Keep going without me.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.emit({ type: 'runStarted', correlationId })

    voice.emit({ type: 'sessionError', error: 'PersonaPlex process exited.', fatal: true })
    expect(coordinator.snapshot().voiceStatus).toBe('error')
    expect(agent.cancelled).toHaveLength(0)

    agent.emit({ type: 'responseFinal', correlationId, text: 'Still finished the job.' })
    await new Promise((resolve) => setTimeout(resolve, 0))

    // The answer still lands in the chat; nothing was spoken.
    expect(chat.assistantMessages()[0].content).toBe('Still finished the job.')
    expect(voice.authoritative()).toHaveLength(0)
  })

  it('restores a conversation and its agent binding after a reload', async () => {
    const { coordinator, chat, agent, voice } = harness()
    await coordinator.attach('conv-1', {
      messages: [
        {
          id: 'msg-old',
          conversationId: 'conv-1',
          role: 'user',
          source: 'user_voice',
          content: 'Earlier, by voice.',
          createdAt: new Date(0).toISOString(),
          status: 'final'
        }
      ],
      summary: 'Investigating a failing test.'
    })

    expect(coordinator.snapshot().agentSessionId).toBe('agent-1')
    expect(coordinator.snapshot().messages).toHaveLength(1)

    // A fresh voice session is seeded from the summary, not the old KV cache.
    await coordinator.startVoiceSession()
    expect(voice.contexts[0].conversationSummary).toBe('Investigating a failing test.')
    expect(chat.messages).toHaveLength(0)
    expect(agent.submitted).toHaveLength(0)
  })
})

describe('failure flows', () => {
  it('reports a failed tool without speaking a success', async () => {
    const { coordinator, agent, voice } = await live()
    speak(voice, 'utt-1', 'Run the build.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.emit({ type: 'runStarted', correlationId })
    agent.emit({
      type: 'toolFailed',
      correlationId,
      toolCallId: 'c1',
      tool: 'shell',
      error: 'exit 1'
    })
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(voice.authoritative()).toHaveLength(0)
    expect(coordinator.events.eventsOfType('TOOL_FAILED')).toHaveLength(1)
  })

  it('speaks a grounded error when the run fails and invents no answer', async () => {
    const { coordinator, agent, voice, chat } = await live()
    speak(voice, 'utt-1', 'Do the thing.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.emit({ type: 'runStarted', correlationId })
    agent.emit({ type: 'runFailed', correlationId, error: 'The model engine crashed.' })
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(chat.assistantMessages()).toHaveLength(0)
    expect(voice.authoritative()).toHaveLength(0)
    const spokenErrors = voice.spoken.filter((request) => request.kind === 'error')
    expect(spokenErrors).toHaveLength(1)
    expect(spokenErrors[0].text).not.toContain('The model engine crashed')
    expect(coordinator.snapshot().responses[0].status).toBe('failed')
  })

  it('keeps the answer in the chat when speech generation fails', async () => {
    const { coordinator, agent, voice, chat } = await live({ allowVoiceBackchannels: false })
    speak(voice, 'utt-1', 'Anything.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    voice.failSpeak = 'No speech synthesizer on this host.'
    agent.completeRun(correlationId, 'Here is the answer.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(chat.assistantMessages()[0].content).toBe('Here is the answer.')
    expect(coordinator.snapshot().responses[0].spokenStatus).toBe('failed')
    expect(chat.statuses.some((status) => status?.includes('Could not speak'))).toBe(true)
  })

  it('surfaces a failed submission instead of hanging the turn', async () => {
    const { coordinator, agent, voice, chat } = await live()
    agent.failSubmit = 'The agent worker is not running.'
    speak(voice, 'utt-1', 'Try anyway.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(coordinator.snapshot().responses[0].status).toBe('failed')
    expect(coordinator.snapshot().activeCorrelationId).toBeNull()
    expect(chat.statuses).toContain('The agent worker is not running.')
  })

  it('ignores a duplicate final transcript', async () => {
    const { coordinator, agent, voice } = await live()
    speak(voice, 'utt-1', 'Only once, please.')
    speak(voice, 'utt-1', 'Only once, please.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(agent.submitted).toHaveLength(1)
    expect(coordinator.snapshot().queue).toHaveLength(0)
    expect(coordinator.metrics().duplicateEventsIgnored).toBe(1)
  })

  it('ignores a duplicate final agent response', async () => {
    const { coordinator, agent, voice, chat } = await live()
    speak(voice, 'utt-1', 'Answer me.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.emit({ type: 'runStarted', correlationId })
    agent.emit({ type: 'responseFinal', correlationId, text: 'The answer.' })
    agent.emit({ type: 'responseFinal', correlationId, text: 'The answer.' })
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(chat.assistantMessages()).toHaveLength(1)
    expect(voice.authoritative()).toHaveLength(1)
  })

  it('treats a partial transcript as display only', async () => {
    const { coordinator, agent, voice } = await live()
    voice.emit({ type: 'userTranscriptPartial', utteranceId: 'utt-1', text: 'delete every' })
    voice.emit({ type: 'userTranscriptPartial', utteranceId: 'utt-1', text: 'delete everything?' })
    await Promise.resolve()

    expect(agent.submitted).toHaveLength(0)
    expect(coordinator.snapshot().partialTranscript).toBe('delete everything?')

    // The final transcript differs substantially; only it is acted on.
    speak(voice, 'utt-1', 'Tell me about everything in the repository.')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(agent.submitted).toHaveLength(1)
    expect(agent.submitted[0].text).toBe('Tell me about everything in the repository.')
    expect(coordinator.snapshot().partialTranscript).toBe('')
  })
})

describe('trust boundary', () => {
  it('never stores PersonaPlex output as an authoritative answer', async () => {
    const { coordinator, agent, voice, chat } = await live()
    speak(voice, 'utt-1', 'Did the build pass?')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.emit({ type: 'runStarted', correlationId })

    // PersonaPlex volunteers a claim about work it does not own.
    voice.emit({ type: 'modelText', text: 'Yes, the build passed!' })
    await Promise.resolve()

    expect(chat.assistantMessages()).toHaveLength(0)
    expect(coordinator.metrics().voiceClaimsRejected).toBe(1)
    // Visible in the voice pane only, so the user can see what was said.
    expect(coordinator.snapshot().voiceModelText).toContain('the build passed')
  })

  it('does not speak when the host has no speech path', async () => {
    const { coordinator, agent, voice, chat } = await live()
    voice.speakable = false
    speak(voice, 'utt-1', 'Say something.')
    await Promise.resolve()
    agent.completeRun(agent.submitted[0].correlationId, 'Text only.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(voice.spoken).toHaveLength(0)
    expect(chat.assistantMessages()[0].content).toBe('Text only.')
    expect(coordinator.snapshot().responses[0].spokenStatus).toBe('none')
  })
})

describe('observability', () => {
  it('measures the latencies the integration is judged on', async () => {
    const { coordinator, agent, voice, clock } = await live()
    speak(voice, 'utt-1', 'How long did that take?')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId

    clock.advance(120)
    agent.emit({ type: 'runStarted', correlationId })
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(coordinator.metrics().transcriptToAgentStartMs).toEqual([120])

    agent.emit({ type: 'responseFinal', correlationId, text: 'About a second.' })
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(coordinator.metrics().responseToSpeechStartMs).toEqual([0])

    clock.advance(30)
    voice.emit({ type: 'userSpeechStarted', utteranceId: 'utt-2' })
    await new Promise((resolve) => setTimeout(resolve, 0))
    clock.advance(15)
    voice.emit({ type: 'speechInterrupted', correlationId })
    expect(coordinator.metrics().interruptToSpeechStopMs).toEqual([15])
  })
})

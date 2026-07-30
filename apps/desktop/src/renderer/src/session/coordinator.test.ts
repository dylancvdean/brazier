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
    config: {
      ...DEFAULT_INTEGRATION_CONFIG,
      voiceEnabled: true,
      // Most coordinator tests specify the destination, not the new local
      // classifier. Keep their original "every transcript submits" premise.
      voiceBackgroundRouting: 'always',
      ...overrides
    }
  })
  // Adapters are connected explicitly, the way the React binding does from an
  // effect. Subscribing in the constructor is what let a remount detach it.
  const disconnect = coordinator.connect()
  return { clock, chat, agent, voice, responder, coordinator, logs, disconnect }
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

    // The transcript and authoritative answer are both in the shared
    // conversation. PersonaPlex is the only audio source.
    expect(chat.messages.map((message) => message.source)).toEqual([
      'user_voice',
      'assistant_agent'
    ])
    expect(chat.assistantMessages()).toHaveLength(1)
    expect(voice.spoken).toHaveLength(0)
    expect(voice.handoffs).toHaveLength(0)
    expect(chat.assistantMessages()[0].correlationId).toBe(correlationId)
  })

  it('hands a tool-backed result to the selected PersonaPlex experiment', async () => {
    const { coordinator, agent, voice } = await live({
      personaplexHandoffStrategy: 'reconnect-service-replay'
    })
    speak(voice, 'utt-1', 'Run the tests and tell me what broke.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId

    agent.emit({ type: 'runStarted', correlationId })
    agent.emit({ type: 'toolStarted', correlationId, toolCallId: 'c1', tool: 'shell' })
    await Promise.resolve()
    expect(voice.handoffs).toHaveLength(0)

    agent.emit({
      type: 'toolCompleted',
      correlationId,
      toolCallId: 'c1',
      tool: 'shell',
      outcome: '1 test failed'
    })
    expect(coordinator.snapshot().task?.confirmedResults).toEqual(['1 test failed'])
    expect(voice.handoffs).toHaveLength(0)

    agent.emit({ type: 'responseFinal', correlationId, text: 'One test failed: oggOpus.' })
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(voice.spoken).toHaveLength(0)
    expect(voice.handoffs).toHaveLength(1)
    expect(voice.handoffs[0]).toMatchObject({
      strategy: 'reconnect-service-replay',
      request: {
        correlationId,
        utteranceId: 'utt-1',
        userText: 'Run the tests and tell me what broke.',
        resultText: 'One test failed: oggOpus.'
      }
    })
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

  it('never sends typed answers to platform TTS', async () => {
    const { coordinator, agent, voice } = await live()
    await coordinator.submitText('Summarize the diff.')
    agent.completeRun(agent.submitted[0].correlationId, 'Three files changed.')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(voice.spoken).toHaveLength(0)
    expect(voice.handoffs).toHaveLength(0)
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

describe('adapter connection', () => {
  /**
   * The bug this exists for: subscriptions were made in the constructor and torn
   * down from a React effect's cleanup. StrictMode runs setup, cleanup, setup on
   * mount, so the coordinator ended up alive and detached — the microphone ran,
   * the adapter logged frames, and every event went nowhere. No error, no
   * transcript, no reply.
   */
  it('survives the disconnect and reconnect a remount performs', async () => {
    const context = harness({ voiceSessionTarget: 'agent' })
    await context.coordinator.attach('conv-1')

    // What StrictMode does on mount.
    context.disconnect()
    context.coordinator.connect()

    await context.coordinator.startVoiceSession()
    speak(context.voice, 'utt-1', 'Still listening?')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(context.agent.submitted).toHaveLength(1)
    expect(context.chat.messages.map((message) => message.source)).toEqual(['user_voice'])
  })

  it('hears nothing at all once disconnected', async () => {
    const context = harness({ voiceSessionTarget: 'agent' })
    await context.coordinator.attach('conv-1')
    await context.coordinator.startVoiceSession()
    context.disconnect()

    speak(context.voice, 'utt-1', 'Anyone there?')
    await new Promise((resolve) => setTimeout(resolve, 0))

    // Detached is detached — which is why it has to be re-established, not
    // assumed to have been done once in the constructor.
    expect(context.agent.submitted).toHaveLength(0)
    expect(context.chat.messages).toHaveLength(0)
  })
})

describe('what the voice session is connected to', () => {
  it('leaves lightweight auto-routed chat turns with PersonaPlex', async () => {
    const { coordinator, agent, voice, chat } = await live({
      voiceSessionTarget: 'chat',
      voiceBackgroundRouting: 'auto'
    })
    speak(voice, 'utt-1', 'How are you?')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(agent.submitted).toHaveLength(0)
    expect(chat.messages).toHaveLength(0)
    expect(coordinator.snapshot().notice).toContain('background model not called')
  })

  it('always sends Agent-targeted speech to the bound agent', async () => {
    const { agent, voice } = await live({ voiceBackgroundRouting: 'auto' })

    speak(voice, 'utt-1', 'How are you?')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(agent.submitted).toHaveLength(1)
    expect(agent.submitted[0].text).toBe('How are you?')
  })

  it('still sends work cues through the background in auto mode', async () => {
    const { agent, voice, chat } = await live({
      voiceBackgroundRouting: 'auto'
    })
    speak(voice, 'utt-1', 'Run the tests')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(agent.submitted).toHaveLength(1)
    expect(chat.messages.map((message) => message.source)).toEqual(['user_voice'])
  })

  it('only sends explicit work cues in explicit mode', async () => {
    const { agent, responder, voice, chat } = await live({
      voiceSessionTarget: 'chat',
      voiceBackgroundRouting: 'explicit'
    })
    speak(
      voice,
      'utt-1',
      'Could you explain why that approach would be safer than the alternative?'
    )
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(agent.submitted).toHaveLength(0)
    expect(chat.messages).toHaveLength(0)

    speak(voice, 'utt-2', 'Check the repository')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(agent.submitted).toHaveLength(0)
    expect(responder.replies).toHaveLength(1)
  })

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

  it('keeps PersonaPlex as the only audible voice for every destination', async () => {
    const { coordinator, voice } = await live({ voiceSessionTarget: 'neither' })
    const base = { ...DEFAULT_INTEGRATION_CONFIG, voiceEnabled: true }
    expect(voice.modelAudioEnabled).toBe(true)

    coordinator.setConfig({ ...base, voiceSessionTarget: 'chat' })
    expect(voice.modelAudioEnabled).toBe(true)
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(voice.spoken).toHaveLength(0)

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

describe('PersonaPlex before a background handoff', () => {
  const restart = {
    personaplexHandoffStrategy: 'restart-service-replay' as const,
    voiceBackgroundRouting: 'auto' as const
  }

  it('can leave the old stream audible as the control', async () => {
    const { voice } = await live({
      ...restart,
      personaplexPreHandoffMode: 'respond'
    })
    voice.emit({ type: 'userSpeechStarted', utteranceId: 'utt-1' })
    voice.emit({ type: 'userTranscriptPartial', utteranceId: 'utt-1', text: 'Run the tests' })
    expect(voice.modelAudioEnabled).toBe(true)
  })

  it('mutes a speculative transcript that looks background-bound', async () => {
    const { voice } = await live({
      ...restart,
      personaplexPreHandoffMode: 'mute-on-route'
    })
    voice.emit({ type: 'userTranscriptPartial', utteranceId: 'utt-1', text: 'Run the tests' })
    expect(voice.modelAudioEnabled).toBe(false)
  })

  it('reopens a speculative mute when the final turn stays local', async () => {
    const { agent, voice } = await live({
      ...restart,
      voiceSessionTarget: 'chat',
      personaplexPreHandoffMode: 'mute-on-route'
    })
    voice.emit({ type: 'userTranscriptPartial', utteranceId: 'utt-1', text: 'Check the thing' })
    expect(voice.modelAudioEnabled).toBe(false)

    speak(voice, 'utt-1', 'How are you?')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(agent.submitted).toHaveLength(0)
    expect(voice.modelAudioEnabled).toBe(true)
  })

  it('can mute before transcription and reopen on the fresh handoff', async () => {
    const { agent, voice } = await live({
      ...restart,
      personaplexPreHandoffMode: 'mute-on-speech'
    })
    voice.emit({ type: 'userSpeechStarted', utteranceId: 'utt-1' })
    expect(voice.modelAudioEnabled).toBe(false)

    speak(voice, 'utt-1', 'Run the tests')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(voice.modelAudioEnabled).toBe(false)

    agent.completeRun(agent.submitted[0].correlationId, 'All tests passed.')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(voice.handoffs).toHaveLength(1)
    expect(voice.modelAudioEnabled).toBe(true)
  })

  it('does not mute when no replacement handoff is selected', async () => {
    const { voice } = await live({
      personaplexHandoffStrategy: 'continuous',
      personaplexPreHandoffMode: 'mute-on-speech'
    })
    voice.emit({ type: 'userSpeechStarted', utteranceId: 'utt-1' })
    expect(voice.modelAudioEnabled).toBe(true)
  })

  it('reopens an immediate mute when transcription returns no words', async () => {
    const { voice } = await live({
      ...restart,
      personaplexPreHandoffMode: 'mute-on-speech'
    })
    voice.emit({ type: 'userSpeechStarted', utteranceId: 'utt-1' })
    expect(voice.modelAudioEnabled).toBe(false)

    voice.emit({ type: 'transcriptionEmpty', utteranceId: 'utt-1' })
    expect(voice.modelAudioEnabled).toBe(true)
  })
})

describe('interruption flows', () => {
  it('lets PersonaPlex handle barge-in natively while the agent keeps working', async () => {
    const { coordinator, agent, voice } = await live()
    speak(voice, 'utt-1', 'Explain the sandbox.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.completeRun(correlationId, 'Seatbelt on macOS, Bubblewrap on Linux.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    voice.emit({ type: 'userSpeechStarted', utteranceId: 'utt-2' })
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(voice.stopped).toHaveLength(0)
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
    expect(coordinator.snapshot().speakingCorrelationId).toBeNull()

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
    expect(voice.stopped).toContain(undefined)
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
  it('renews at a safe boundary when the PersonaPlex experiment prompt changes', async () => {
    const { coordinator, voice } = await live()
    expect(voice.contexts[0].behavioralRules.join(' ')).toContain('background assistant')

    coordinator.setConfig({
      ...DEFAULT_INTEGRATION_CONFIG,
      voiceEnabled: true,
      voiceSessionTarget: 'agent',
      personaplexHandoffStrategy: 'reconnect-service-replay'
    })
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(voice.sessions).toHaveLength(2)
    expect(voice.contexts.at(-1)?.behavioralRules.join(' ')).toContain('fresh prompt')
  })

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

  it('shows a run failure without synthesizing a second voice', async () => {
    const { coordinator, agent, voice, chat } = await live()
    speak(voice, 'utt-1', 'Do the thing.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.emit({ type: 'runStarted', correlationId })
    agent.emit({ type: 'runFailed', correlationId, error: 'The model engine crashed.' })
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(chat.assistantMessages()).toHaveLength(0)
    expect(voice.authoritative()).toHaveLength(0)
    expect(voice.spoken).toHaveLength(0)
    expect(coordinator.snapshot().notice).toContain('The model engine crashed')
    expect(coordinator.snapshot().responses[0].status).toBe('failed')
  })

  it('keeps the answer in chat and never calls the legacy speech renderer', async () => {
    const { coordinator, agent, voice, chat } = await live()
    speak(voice, 'utt-1', 'Anything.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    voice.failSpeak = 'No speech synthesizer on this host.'
    agent.completeRun(correlationId, 'Here is the answer.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(chat.assistantMessages()[0].content).toBe('Here is the answer.')
    expect(coordinator.snapshot().responses[0].spokenStatus).toBe('none')
    expect(voice.spoken).toHaveLength(0)
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

  /**
   * Voice mode renders no chat transcript, so a failure reported only to the
   * chat adapter left a live session looking like it was listening while every
   * utterance went nowhere.
   */
  it('puts every failure somewhere a voice-only screen can show it', async () => {
    const { coordinator, agent, voice, chat } = await live()

    voice.emit({
      type: 'sessionError',
      error: 'Could not transcribe what you said: no Whisper model.',
      fatal: false
    })
    expect(coordinator.snapshot().notice).toContain('Could not transcribe')

    speak(voice, 'utt-1', 'Then answer this.')
    await new Promise((resolve) => setTimeout(resolve, 0))
    agent.emit({
      type: 'runFailed',
      correlationId: agent.submitted[0].correlationId,
      error: 'The agent worker exited.'
    })
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(coordinator.snapshot().notice).toBe('The agent worker exited.')
    // The turn itself is marked, so a transcript with no reply still shows why.
    expect(chat.messages.at(-1)?.status).toBe('failed')
  })

  /**
   * The transcript arrives on an adapter callback, which cannot await, so a
   * failure storing it was an unhandled rejection: indistinguishable from a
   * transcript that never arrived.
   */
  it('reports a transcript it could not store', async () => {
    const { coordinator, chat, voice } = await live()
    chat.appendMessage = () => Promise.reject(new Error('conversation is gone'))

    speak(voice, 'utt-1', 'Store this somewhere.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(coordinator.snapshot().notice).toContain('conversation is gone')
    expect(coordinator.snapshot().notice).toContain('Submitting what you said')
  })

  it('clears the notice once a turn succeeds', async () => {
    const { coordinator, agent, voice } = await live()
    voice.emit({ type: 'sessionError', error: 'A passing glitch.', fatal: false })
    expect(coordinator.snapshot().notice).not.toBeNull()

    speak(voice, 'utt-1', 'Try again.')
    await new Promise((resolve) => setTimeout(resolve, 0))
    agent.completeRun(agent.submitted[0].correlationId, 'Worked this time.')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(coordinator.snapshot().notice).toBeNull()
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
    const { coordinator, agent, voice, chat } = await live({
      personaplexHandoffStrategy: 'reconnect-direct-replay'
    })
    speak(voice, 'utt-1', 'Answer me.')
    await Promise.resolve()
    const correlationId = agent.submitted[0].correlationId
    agent.emit({ type: 'runStarted', correlationId })
    agent.emit({ type: 'responseFinal', correlationId, text: 'The answer.' })
    agent.emit({ type: 'responseFinal', correlationId, text: 'The answer.' })
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(chat.assistantMessages()).toHaveLength(1)
    expect(voice.handoffs).toHaveLength(1)
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

describe('spoken confirmation', () => {
  /** Hold a call the way the permission broker does, mid-run. */
  async function holdACall(overrides: Partial<IntegrationConfig> = {}) {
    const context = await live(overrides)
    speak(context.voice, 'utt-1', 'Clean up the build directory.')
    await Promise.resolve()
    const correlationId = context.agent.submitted[0].correlationId
    context.agent.emit({ type: 'runStarted', correlationId })
    context.agent.emit({
      type: 'approvalRequired',
      correlationId,
      approvalId: 'apv-1',
      tool: 'shell_run',
      summary: 'Run rm -rf build in /work/brazier',
      risk: 'destructive',
      environment: 'host'
    })
    await new Promise((resolve) => setTimeout(resolve, 0))
    return { ...context, correlationId }
  }

  it('shows the action before it happens without platform TTS', async () => {
    const { coordinator, voice } = await holdACall()
    expect(voice.spoken).toHaveLength(0)
    expect(coordinator.snapshot().notice).toContain('Run rm -rf build')
    expect(coordinator.snapshot().pendingApproval?.approvalId).toBe('apv-1')
  })

  it('allows it on an unmistakable yes, and says so in the conversation', async () => {
    const { coordinator, agent, chat, voice } = await holdACall()
    speak(voice, 'utt-2', 'Yes')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(agent.decisions).toEqual([
      { approvalId: 'apv-1', decision: 'approve', note: 'Spoken answer: “Yes”' }
    ])
    expect(coordinator.snapshot().pendingApproval).toBeNull()
    expect(chat.messages.at(-1)?.content).toContain('Allowed by voice')
    expect(coordinator.metrics().approvalsSpokenApproved).toBe(1)
  })

  it('refuses it on a no', async () => {
    const { agent, chat, voice, coordinator } = await holdACall()
    speak(voice, 'utt-2', 'No, stop.')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(agent.decisions[0].decision).toBe('deny')
    expect(chat.messages.at(-1)?.content).toContain('Refused by voice')
    expect(coordinator.metrics().approvalsSpokenDenied).toBe(1)
  })

  /**
   * The point of the whole feature: the transcript comes from a microphone, an
   * speech detector, and a recogniser, so anything short of a clean yes has to leave
   * the call held rather than guess which way the sentence was leaning.
   */
  it('treats a qualified answer as no answer at all', async () => {
    const { coordinator, agent, voice } = await holdACall()
    speak(voice, 'utt-2', 'yes but only the temp files')
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(agent.decisions).toHaveLength(0)
    expect(coordinator.snapshot().pendingApproval?.approvalId).toBe('apv-1')
    expect(coordinator.snapshot().notice).toContain('not a yes or a no')
    expect(coordinator.metrics().approvalsUnclear).toBe(1)
  })

  it('does not submit the answer as a new request', async () => {
    const { agent, voice } = await holdACall()
    speak(voice, 'utt-2', 'Yes')
    await new Promise((resolve) => setTimeout(resolve, 0))
    // One turn: the original request. "Yes" answered the question, it did not
    // ask a new one of an agent that is stopped mid-action.
    expect(agent.submitted).toHaveLength(1)
  })

  it('keeps the call held when the decision cannot be recorded', async () => {
    const { coordinator, agent, voice } = await holdACall()
    agent.failDecision = 'daemon unreachable'
    speak(voice, 'utt-2', 'Yes')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(coordinator.snapshot().pendingApproval?.approvalId).toBe('apv-1')
    expect(coordinator.snapshot().notice).toContain('daemon unreachable')
  })

  it('can be answered by hand instead', async () => {
    const { coordinator, agent } = await holdACall()
    await coordinator.resolveApproval('deny')
    expect(agent.decisions).toEqual([
      { approvalId: 'apv-1', decision: 'deny', note: undefined }
    ])
    expect(coordinator.snapshot().pendingApproval).toBeNull()
  })

  it('does not speak over a typed request, but still shows what is held', async () => {
    const { coordinator, agent, voice } = await live()
    await coordinator.submitText('Clean up the build directory.')
    const correlationId = agent.submitted[0].correlationId
    const before = voice.spoken.length
    agent.emit({
      type: 'approvalRequired',
      correlationId,
      approvalId: 'apv-2',
      tool: 'shell_run',
      summary: 'Run rm -rf build',
      risk: 'destructive',
      environment: 'host'
    })
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(voice.spoken).toHaveLength(before)
    expect(coordinator.snapshot().pendingApproval?.approvalId).toBe('apv-2')
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
    expect(coordinator.metrics().responseToSpeechStartMs).toEqual([])

    clock.advance(30)
    voice.emit({ type: 'userSpeechStarted', utteranceId: 'utt-2' })
    await new Promise((resolve) => setTimeout(resolve, 0))
    clock.advance(15)
    voice.emit({ type: 'speechInterrupted', correlationId })
    expect(coordinator.metrics().interruptToSpeechStopMs).toEqual([])
  })

  /**
   * Whether batch whisper or the resident streaming worker should transcribe a
   * spoken turn is a question about this machine, and the session is the only
   * thing in a position to answer it.
   */
  it('keeps what each transcription interface costs, separately', async () => {
    const { coordinator, voice } = await live()
    voice.emit({
      type: 'transcriptionMeasured',
      utteranceId: 'utt-1',
      engine: 'whisper.cpp',
      roundTripMs: 400,
      waitedMs: 20,
      engineMs: 380,
      audioSeconds: 2,
      startedAtPause: true
    })
    voice.emit({
      type: 'transcriptionMeasured',
      utteranceId: 'utt-2',
      engine: 'whisper.cpp',
      roundTripMs: 600,
      waitedMs: 600,
      engineMs: 560,
      audioSeconds: 2,
      startedAtPause: false
    })
    voice.emit({
      type: 'transcriptionMeasured',
      utteranceId: 'utt-3',
      engine: 'streaming-asr',
      roundTripMs: 180,
      waitedMs: 180,
      engineMs: 150,
      audioSeconds: 2,
      startedAtPause: false
    })

    const costs = coordinator.snapshot().transcription
    expect(costs).toEqual([
      {
        engine: 'whisper.cpp',
        utterances: 2,
        lastMs: 600,
        averageMs: 500,
        averageWaitMs: 310,
        startedAtPause: 1,
        realTimeFactor: 0.25
      },
      {
        engine: 'streaming-asr',
        utterances: 1,
        lastMs: 180,
        averageMs: 180,
        averageWaitMs: 180,
        startedAtPause: 0,
        realTimeFactor: 0.09
      }
    ])
    // What the turn waited is the number the latency work is judged on.
    expect(coordinator.metrics().transcriptWaitMs).toEqual([20, 600, 180])
  })
})

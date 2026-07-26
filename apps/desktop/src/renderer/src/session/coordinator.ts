/**
 * Session coordinator: one conversation shared by text chat, voice, and the
 * agent.
 *
 * The rules it exists to enforce:
 *
 * - The agent session stays authoritative for tools, task state, and facts.
 *   PersonaPlex acknowledges and renders; it never decides an outcome.
 * - Exactly one subsystem owns the substantive answer to a turn, and ownership
 *   is recorded rather than inferred from timing.
 * - Stopping speech, cancelling a response, and cancelling the agent task are
 *   three different operations and are never conflated.
 * - Voice and text turns go through one submission path into one agent session.
 *
 * It runs in the renderer because that is the only process holding all three
 * edges (audio devices, the agent IPC bridge, the conversation API). The agent
 * run itself lives in the worker process, so a task outlives this object.
 */

import type {
  AgentAdapter,
  AgentAdapterEvent,
  ChatAdapter,
  ChatResponder,
  VoiceAdapter,
  VoiceAdapterEvent
} from './adapters'
import { DEFAULT_INTEGRATION_CONFIG, type IntegrationConfig } from './config'
import { SessionEventLog } from './eventLog'
import { isTooThinToSubmit } from './echoGuard'
import { classifyUtterance, isControlIntent, type UtteranceIntent } from './interruption'
import type {
  ConversationMessage,
  DeliveryTarget,
  DiagnosticRecord,
  EventSource,
  MessageSource,
  ResponseOwner,
  ResponseState,
  SessionEvent,
  SessionEventType,
  SessionMetrics,
  SpeechRequest,
  TaskState,
  VoiceContext
} from './types'
import { buildVoiceContext, summarizeForVoice } from './voiceContext'

export type VoiceStatus = 'off' | 'starting' | 'live' | 'error'

export type QueuedTurn = {
  correlationId: string
  text: string
  source: MessageSource
  deliveryTargets: DeliveryTarget
  userMessageId: string
  queuedAt: number
}

export type CoordinatorSnapshot = {
  conversationId: string | null
  agentSessionId: string | null
  voiceSessionId: string | null
  voiceStatus: VoiceStatus
  voiceError: string | null
  messages: ConversationMessage[]
  responses: ResponseState[]
  activeCorrelationId: string | null
  queue: QueuedTurn[]
  task: TaskState | null
  summary: string
  /** In-flight authoritative text, for the streaming bubble. */
  streamingText: string
  /** Latest unstable user transcript; display only, never submitted. */
  partialTranscript: string
  /** What PersonaPlex said on its own. Shown in the voice pane, never stored. */
  voiceModelText: string
  speakingCorrelationId: string | null
  /**
   * Where the microphone has got to. Voice has more silent steps than any other
   * surface — speech detection, transcription, routing — and each one looks
   * exactly like nobody having spoken.
   */
  hearing: 'idle' | 'speaking' | 'transcribing'
  /**
   * What the microphone has delivered: total frames, and the loudest recent
   * frame on the same scale as the speech gate. Zero frames means the capture
   * graph is not running; frames with a peak under the gate means the room or
   * the gain is too quiet.
   */
  capture: { frames: number; peak: number; status: string; gate: number; noiseFloor: number }
  /**
   * What transcription is costing, per interface that has served an utterance
   * this session. Which one should transcribe a spoken turn is an open
   * question — a whisper.cpp invocation per utterance against a resident
   * Nemotron worker — and this is the number that answers it, on the machine
   * the answer has to hold for.
   */
  transcription: TranscriptionCost[]
  /**
   * The last thing the coordinator wanted to tell the user: agent status, or a
   * failure that did not stop the session. It is in the snapshot as well as on
   * the chat adapter because a host that shows no chat transcript would
   * otherwise drop it, and a turn failing silently looks like nothing happened.
   */
  notice: string | null
}

/** Rolling transcription cost for one ASR interface. */
export type TranscriptionCost = {
  engine: string
  utterances: number
  /** Most recent round trip, in milliseconds. */
  lastMs: number
  /** Mean round trip over the session. */
  averageMs: number
  /**
   * Mean wait after the user stopped talking — the part of the round trip a
   * person actually sits through. Far below `averageMs` means transcription is
   * mostly finishing inside the silence window rather than after it.
   */
  averageWaitMs: number
  /** Utterances whose transcription began at a pause, before the close. */
  startedAtPause: number
  /**
   * Mean round trip divided by the length of the audio. Below 1 the interface
   * transcribes faster than people speak, which is what a conversation needs.
   */
  realTimeFactor: number
}

export type CoordinatorDeps = {
  chat: ChatAdapter
  agent: AgentAdapter
  voice: VoiceAdapter
  responder?: ChatResponder
  config?: IntegrationConfig
  /** Persona text a voice session is launched with. */
  persona?: string
  /** Store the compact summary alongside the conversation. */
  persistSummary?: (conversationId: string, summary: string) => void
  now?: () => number
  newId?: (prefix: string) => string
  log?: (record: DiagnosticRecord) => void
}

/**
 * How many coordinators this page has built.
 *
 * One is expected. More means a remount left an earlier one alive, holding the
 * audio graph and the adapter subscriptions while a newer one renders — which
 * looks like a working session that ignores everything.
 */
let instances = 0

/** Immediate acknowledgments. None of them claims anything happened. */
const BACKCHANNELS = ['Let me check.', "I'm looking at that now.", 'One moment.']

export class SessionCoordinator {
  readonly events = new SessionEventLog()

  /** Identifies this instance in the log; see `instances`. */
  readonly id: string
  private readonly deps: CoordinatorDeps
  private readonly now: () => number
  private readonly newId: (prefix: string) => string
  /** Swapped when the conversation changes; see `setChatAdapter`. */
  private chat: ChatAdapter
  private config: IntegrationConfig
  private persona: string

  private conversationId: string | null = null
  private messages: ConversationMessage[] = []
  private readonly responses = new Map<string, ResponseState>()
  private activeCorrelationId: string | null = null
  private queue: QueuedTurn[] = []
  private task: TaskState | null = null
  private summary = ''
  private streamingText = ''
  private partialTranscript = ''
  private voiceModelText = ''

  private voiceSessionId: string | null = null
  private voiceStartedAt = 0
  private voiceStatus: VoiceStatus = 'off'
  private voiceError: string | null = null
  private speakingCorrelationId: string | null = null
  private notice: string | null = null
  private hearing: CoordinatorSnapshot['hearing'] = 'idle'
  private capture: CoordinatorSnapshot['capture'] = {
    frames: 0,
    peak: 0,
    status: '',
    gate: 0,
    noiseFloor: 0
  }
  /** Per-engine transcription totals; see `CoordinatorSnapshot.transcription`. */
  private readonly transcription = new Map<
    string,
    {
      utterances: number
      lastMs: number
      totalMs: number
      totalWaitedMs: number
      totalAudioSeconds: number
      startedAtPause: number
    }
  >()
  private pendingRenewal: string | null = null
  private backchanneling = new Set<string>()
  private statusCued = new Set<string>()
  private interruptRequestedAt: number | null = null

  private readonly listeners = new Set<(snapshot: CoordinatorSnapshot) => void>()
  private readonly metricsState: SessionMetrics = {
    transcriptWaitMs: [],
    transcriptToAgentStartMs: [],
    responseToSpeechStartMs: [],
    interruptToSpeechStopMs: [],
    duplicateEventsIgnored: 0,
    voiceSessionRenewals: 0,
    agentTasksCancelledByInterruption: 0,
    voiceClaimsRejected: 0
  }
  /** Utterance final timestamps, for the transcript-to-agent-start metric. */
  private readonly turnStartedAt = new Map<string, number>()

  constructor(deps: CoordinatorDeps) {
    this.deps = deps
    this.chat = deps.chat
    this.now = deps.now ?? (() => Date.now())
    let counter = 0
    this.newId =
      deps.newId ??
      ((prefix) => {
        counter += 1
        return `${prefix}-${counter}-${Math.random().toString(36).slice(2, 8)}`
      })
    this.config = deps.config ?? DEFAULT_INTEGRATION_CONFIG
    this.persona = deps.persona ?? 'You are a helpful assistant.'
    instances += 1
    this.id = `coord-${instances}`
    console.debug(`[voice] ${this.id} constructed`)
  }

  /**
   * Subscribe to the adapters, returning a function that detaches again.
   *
   * Subscribing in the constructor while detaching from a React effect's
   * cleanup is asymmetric, and StrictMode exists to expose exactly that: its
   * mount/unmount/mount cycle detached the coordinator and nothing re-attached
   * it, so the microphone kept running and every event it produced went to
   * nobody. Connecting is something that can be undone and redone.
   */
  connect(): () => void {
    const unsubscribes = [
      this.deps.agent.subscribe((event) => this.onAgentEvent(event)),
      this.deps.voice.subscribe((event) => this.onVoiceEvent(event))
    ]
    console.debug(`[voice] ${this.id} connected to its adapters`)
    return () => {
      console.debug(`[voice] ${this.id} disconnected from its adapters`)
      for (const unsubscribe of unsubscribes) unsubscribe()
    }
  }

  dispose(): void {
    this.listeners.clear()
  }

  // --- Lifecycle ------------------------------------------------------------

  /**
   * Bind to a conversation, adopting whatever agent session it already records
   * so text and voice never open one each.
   */
  async attach(
    conversationId: string,
    options: { messages?: ConversationMessage[]; summary?: string } = {}
  ): Promise<void> {
    this.conversationId = conversationId
    this.messages = options.messages ?? []
    this.summary = options.summary ?? ''
    this.responses.clear()
    this.activeCorrelationId = null
    this.queue = []
    this.task = null
    this.streamingText = ''
    this.partialTranscript = ''
    this.voiceModelText = ''
    await this.deps.agent.attachSession(conversationId)
    this.publish()
  }

  /**
   * Point persistence at a different conversation. The chat adapter is bound to
   * one conversation id, but the voice session and the agent binding are not, so
   * switching conversations retargets rather than rebuilds.
   */
  setChatAdapter(chat: ChatAdapter): void {
    this.chat = chat
  }

  setConfig(config: IntegrationConfig): void {
    const previous = this.config
    this.config = config
    if (previous.voiceSessionTarget !== config.voiceSessionTarget) this.applyAudioOwnership()
    this.publish()
  }

  /**
   * Decide whose voice the user hears. When the coordinator delivers answers,
   * PersonaPlex's own audio is silenced so there are never two replies to one
   * question; when connected to nothing, it is the only voice there is.
   */
  private applyAudioOwnership(): void {
    const coordinatorSpeaks =
      this.config.voiceSessionTarget !== 'neither' && this.deps.voice.canSpeak()
    this.deps.voice.setModelAudioEnabled(!coordinatorSpeaks)
  }

  setPersona(persona: string): void {
    this.persona = persona
  }

  snapshot(): CoordinatorSnapshot {
    return {
      conversationId: this.conversationId,
      agentSessionId: this.deps.agent.attachedSessionId(),
      voiceSessionId: this.voiceSessionId,
      voiceStatus: this.voiceStatus,
      voiceError: this.voiceError,
      messages: [...this.messages],
      responses: [...this.responses.values()],
      activeCorrelationId: this.activeCorrelationId,
      queue: [...this.queue],
      task: this.task,
      summary: this.summary,
      streamingText: this.streamingText,
      partialTranscript: this.partialTranscript,
      voiceModelText: this.voiceModelText,
      speakingCorrelationId: this.speakingCorrelationId,
      hearing: this.hearing,
      capture: this.capture,
      transcription: [...this.transcription.entries()].map(([engine, totals]) => ({
        engine,
        utterances: totals.utterances,
        lastMs: totals.lastMs,
        averageMs: Math.round(totals.totalMs / Math.max(1, totals.utterances)),
        averageWaitMs: Math.round(totals.totalWaitedMs / Math.max(1, totals.utterances)),
        startedAtPause: totals.startedAtPause,
        realTimeFactor:
          totals.totalAudioSeconds > 0 ? totals.totalMs / 1000 / totals.totalAudioSeconds : 0
      })),
      notice: this.notice
    }
  }

  /**
   * Tell the user something, through the chat adapter and the snapshot both.
   *
   * Reporting only to the chat adapter meant that in Voice mode — which renders
   * no chat transcript — a failed transcription, a refused turn, or an agent
   * error produced no visible sign at all: the session sat there looking live
   * while every utterance quietly went nowhere.
   */
  /**
   * Run work started from an event handler, reporting a failure rather than
   * leaving it as an unhandled rejection.
   *
   * None of these paths can be awaited by their caller — they hang off adapter
   * callbacks — and a rejection in one was invisible. A transcript that failed
   * to store looked exactly like a transcript that never arrived.
   */
  private track(work: Promise<unknown>, label: string): void {
    void work.catch((cause: unknown) => {
      const message = cause instanceof Error ? cause.message : String(cause)
      console.warn(`[voice] ${label} failed: ${message}`)
      this.report(`${label} failed: ${message}`)
    })
  }

  private report(status: string | null): void {
    this.notice = status
    this.chat.showStatus(status)
    this.publish()
  }

  subscribe(listener: (snapshot: CoordinatorSnapshot) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  metrics(): SessionMetrics {
    return {
      ...this.metricsState,
      duplicateEventsIgnored:
        this.metricsState.duplicateEventsIgnored + this.events.duplicateCount()
    }
  }

  // --- Submission -----------------------------------------------------------

  /** Typed input. Same path as voice, so both share one agent session. */
  async submitText(text: string): Promise<string | null> {
    const trimmed = text.trim()
    if (!trimmed) return null
    const correlationId = this.newId('turn')
    this.emit('USER_TEXT_SUBMITTED', correlationId, 'chat', { text: trimmed })
    return this.submitTurn({ correlationId, text: trimmed, source: 'user_text' })
  }

  /**
   * Route a finalized user turn: record it, then start it or queue it behind the
   * one active run. Returns the correlation id, or null when the turn was
   * consumed as a control rather than a question.
   */
  private async submitTurn(input: {
    correlationId: string
    text: string
    source: MessageSource
    supersedes?: string
  }): Promise<string | null> {
    if (!this.conversationId) {
      // Nothing to write into. Say so rather than dropping what the user said.
      this.report('No conversation is open, so that turn was not recorded.')
      return null
    }
    const deliveryTargets = this.targetsFor(input.source)
    const userMessage = await this.chat.appendMessage({
      role: 'user',
      source: input.source,
      content: input.text,
      correlationId: input.correlationId,
      status: 'final'
    })
    this.recordMessage(userMessage)

    const owner = this.ownerFor(input.source)
    if (!owner) {
      this.report(
        'Voice is set to reach the agent, but no agent session is bound to this conversation. Start a task in Agent mode, or change what voice is connected to.'
      )
      await this.patchMessage(userMessage.id, { status: 'failed' })
      return input.correlationId
    }
    this.responses.set(input.correlationId, {
      correlationId: input.correlationId,
      owner,
      deliveryTargets,
      status: 'pending',
      cancellable: true,
      spokenStatus: 'none',
      originSource: input.source,
      userMessageId: userMessage.id,
      createdAt: this.now()
    })

    if (this.activeCorrelationId) {
      // One active run per conversation: the agent session is not concurrent,
      // and neither message is discarded.
      this.queue.push({
        correlationId: input.correlationId,
        text: input.text,
        source: input.source,
        deliveryTargets,
        userMessageId: userMessage.id,
        queuedAt: this.now()
      })
      this.chat.markQueued(userMessage.id)
      await this.patchMessage(userMessage.id, { metadata: { queued: true } })
      this.publish()
      return input.correlationId
    }

    await this.startTurn({
      correlationId: input.correlationId,
      text: input.text,
      supersedes: input.supersedes
    })
    return input.correlationId
  }

  private async startTurn(input: {
    correlationId: string
    text: string
    supersedes?: string
  }): Promise<void> {
    const response = this.responses.get(input.correlationId)
    if (!response) return
    this.activeCorrelationId = input.correlationId
    this.streamingText = ''
    response.status = 'running'
    response.startedAt = this.now()

    if (response.owner === 'agent') {
      this.emit('AGENT_REQUESTED', input.correlationId, 'coordinator', { text: input.text })
      try {
        await this.deps.agent.submitTurn({
          correlationId: input.correlationId,
          text: input.text,
          source: response.originSource,
          supersedes: input.supersedes
        })
      } catch (cause) {
        this.failResponse(input.correlationId, errorText(cause))
      }
      this.publish()
      return
    }

    // Chat-owned turn: the existing completion path produces the answer.
    const responder = this.deps.responder
    if (!responder) {
      this.failResponse(
        input.correlationId,
        'No model is available to answer. Select a chat model or start an agent task.'
      )
      this.publish()
      return
    }
    this.publish()
    try {
      const result = await responder.respond({
        correlationId: input.correlationId,
        text: input.text,
        onPartial: (delta) => {
          if (this.activeCorrelationId !== input.correlationId) return
          this.streamingText += delta
          this.publish()
        }
      })
      if (this.responses.get(input.correlationId)?.status === 'cancelled') return
      await this.deliverFinal(input.correlationId, result.text)
    } catch (cause) {
      this.failResponse(input.correlationId, errorText(cause))
      this.publish()
    }
  }

  /**
   * Who owns the answer to this turn. Null when the turn asked for the agent and
   * there is none — refused rather than quietly handed to the chat model, which
   * would answer without the workspace, tools, or task state the user expected.
   */
  private ownerFor(source: MessageSource): ResponseOwner | null {
    const bound = this.deps.agent.attachedSessionId() !== null
    // A typed turn goes wherever the conversation is pointed; only speech has a
    // destination the user chose. 'neither' never reaches here — those
    // transcripts are dropped before submission.
    if (source !== 'user_voice') return bound ? 'agent' : 'chat'
    if (this.config.voiceSessionTarget === 'agent') return bound ? 'agent' : null
    return 'chat'
  }

  /** Where a turn's answer should go, given who asked and the configuration. */
  private targetsFor(source: MessageSource): DeliveryTarget {
    const voiceLive = this.config.voiceEnabled && this.voiceSessionId !== null
    if (!voiceLive) return 'text'
    if (source === 'user_voice') {
      return this.config.speakVoiceOriginatedResponses ? 'both' : 'text'
    }
    return this.config.speakTextOriginatedResponses ? 'both' : 'text'
  }

  // --- Agent events ---------------------------------------------------------

  private onAgentEvent(event: AgentAdapterEvent): void {
    const response = this.responses.get(event.correlationId)
    switch (event.type) {
      case 'runStarted': {
        if (response) {
          response.status = 'running'
          response.startedAt ??= this.now()
        }
        const startedAt = this.turnStartedAt.get(event.correlationId)
        if (startedAt !== undefined) {
          this.metricsState.transcriptToAgentStartMs.push(this.now() - startedAt)
          this.turnStartedAt.delete(event.correlationId)
        }
        this.task = {
          correlationId: event.correlationId,
          label: 'Agent task',
          status: 'running',
          confirmedResults: [],
          updatedAt: this.now()
        }
        this.emit('AGENT_STARTED', event.correlationId, 'agent', {})
        this.track(this.acknowledge(event.correlationId), 'Acknowledgement')
        this.publish()
        return
      }
      case 'statusUpdated': {
        if (this.task?.correlationId === event.correlationId) {
          this.task = {
            ...this.task,
            activeTool: event.activeTool,
            updatedAt: this.now()
          }
        }
        this.report(event.status)
        this.emit('AGENT_STATUS_UPDATED', event.correlationId, 'agent', { status: event.status })
        this.publish()
        return
      }
      case 'responsePartial': {
        if (this.activeCorrelationId !== event.correlationId) return
        this.streamingText += event.delta
        this.emit('AGENT_RESPONSE_PARTIAL', event.correlationId, 'agent', { delta: event.delta })
        this.publish()
        return
      }
      case 'responseFinal': {
        // A resent final response must not append a second answer or be spoken
        // twice, so the fact is deduped before anything acts on it.
        const published = this.emit(
          'AGENT_RESPONSE_FINAL',
          event.correlationId,
          'agent',
          { text: event.text },
          `${event.correlationId}:AGENT_RESPONSE_FINAL`
        )
        if (!published) {
          this.diagnose('DUPLICATE_IGNORED', event.correlationId, 'agent')
          return
        }
        this.track(this.deliverFinal(event.correlationId, event.text), 'Storing the answer')
        return
      }
      case 'toolStarted': {
        if (this.task?.correlationId === event.correlationId) {
          this.task = { ...this.task, activeTool: event.tool, updatedAt: this.now() }
        }
        this.emit('TOOL_STARTED', event.correlationId, 'agent', { tool: event.tool })
        this.track(
          this.cueStatus(event.correlationId, `Still working — running ${event.tool}.`),
          'Status cue'
        )
        this.publish()
        return
      }
      case 'toolCompleted': {
        if (this.task?.correlationId === event.correlationId) {
          this.task = {
            ...this.task,
            activeTool: undefined,
            // Only what the agent reported. This is the whole set of facts the
            // voice is permitted to state about the work.
            confirmedResults: [...this.task.confirmedResults, event.outcome].slice(-8),
            updatedAt: this.now()
          }
        }
        this.emit('TOOL_COMPLETED', event.correlationId, 'agent', {
          tool: event.tool,
          outcome: event.outcome
        })
        this.publish()
        return
      }
      case 'toolFailed': {
        if (this.task?.correlationId === event.correlationId) {
          this.task = { ...this.task, activeTool: undefined, updatedAt: this.now() }
        }
        this.emit('TOOL_FAILED', event.correlationId, 'agent', {
          tool: event.tool,
          error: event.error
        })
        this.publish()
        return
      }
      case 'runFailed': {
        this.failResponse(event.correlationId, event.error)
        this.publish()
        return
      }
      case 'runCancelled': {
        const state = this.responses.get(event.correlationId)
        if (state && state.status !== 'delivered') state.status = 'cancelled'
        if (this.task?.correlationId === event.correlationId) {
          this.task = { ...this.task, status: 'cancelled', updatedAt: this.now() }
        }
        void this.stopSpeechFor(event.correlationId)
        this.finishActive(event.correlationId)
        this.publish()
        return
      }
      default:
        return
    }
  }

  /**
   * Store the authoritative answer once, show it, and only then ask for spoken
   * delivery. The spoken rendering shares this message's correlation id instead
   * of becoming a second assistant message.
   */
  private async deliverFinal(correlationId: string, text: string): Promise<void> {
    const response = this.responses.get(correlationId)
    if (!response) return
    if (response.status === 'delivered' || response.status === 'cancelled') return

    const message = await this.chat.appendMessage({
      role: 'assistant',
      source: response.owner === 'agent' ? 'assistant_agent' : 'assistant_chat',
      content: text,
      correlationId,
      status: 'final'
    })
    this.recordMessage(message)
    response.authoritativeMessageId = message.id
    response.status = 'delivered'
    response.cancellable = false
    response.finalizedAt = this.now()
    this.streamingText = ''
    if (this.task?.correlationId === correlationId) {
      this.task = { ...this.task, status: 'completed', activeTool: undefined, updatedAt: this.now() }
    }
    this.report(null)

    if (this.shouldSpeak(response)) await this.requestSpeech(response, text)
    this.finishActive(correlationId)
    this.publish()
  }

  private failResponse(correlationId: string, error: string): void {
    const response = this.responses.get(correlationId)
    if (response && response.status !== 'delivered') {
      response.status = 'failed'
      response.cancellable = false
      if (response.userMessageId) {
        // Marked on the turn itself, not only in metadata, so a transcript that
        // shows no assistant reply still shows that the reply failed.
        void this.patchMessage(response.userMessageId, {
          status: 'failed',
          metadata: { failed: true }
        })
      }
    }
    if (this.task?.correlationId === correlationId) {
      this.task = { ...this.task, status: 'failed', activeTool: undefined, updatedAt: this.now() }
    }
    this.streamingText = ''
    this.report(error)
    this.emit('AGENT_FAILED', correlationId, 'agent', { error })
    this.diagnose('AGENT_FAILED', correlationId, 'agent', { errorCategory: 'agent_run_failed' })
    // Any pending success-oriented speech is stopped; the voice may state the
    // failure briefly, but it never invents a fallback answer.
    void this.stopSpeechFor(correlationId)
    const response2 = this.responses.get(correlationId)
    if (response2 && this.shouldSpeak(response2)) {
      void this.speakRequest({
        correlationId,
        text: 'That run failed. The error is on screen.',
        kind: 'error'
      })
    }
    this.finishActive(correlationId)
  }

  /** Clear the active slot and start whatever was waiting. */
  private finishActive(correlationId: string): void {
    if (this.activeCorrelationId !== correlationId) return
    this.activeCorrelationId = null
    this.track(this.drainQueue(), 'Starting the next turn')
  }

  private async drainQueue(): Promise<void> {
    if (this.activeCorrelationId) return
    const next = this.queue.shift()
    if (!next) {
      await this.runPendingRenewal()
      return
    }
    const response = this.responses.get(next.correlationId)
    if (!response || response.status === 'superseded' || response.status === 'cancelled') {
      await this.drainQueue()
      return
    }
    await this.patchMessage(next.userMessageId, { metadata: { queued: false } })
    await this.startTurn({ correlationId: next.correlationId, text: next.text })
  }

  // --- Voice events ---------------------------------------------------------

  private onVoiceEvent(event: VoiceAdapterEvent): void {
    switch (event.type) {
      case 'userSpeechStarted': {
        this.hearing = 'speaking'
        this.publish()
        this.track(this.onBargeIn(), 'Interrupting speech')
        return
      }
      case 'captureLevel': {
        this.capture = {
          frames: event.frames,
          peak: event.peak,
          status: event.status,
          gate: event.gate,
          noiseFloor: event.noiseFloor
        }
        console.debug(
          `[voice] ${this.id} sees ${event.frames} frames, peak ${event.peak.toFixed(3)}`
        )
        this.publish()
        return
      }
      case 'transcriptionStarted': {
        this.hearing = 'transcribing'
        this.publish()
        return
      }
      case 'transcriptionMeasured': {
        const totals = this.transcription.get(event.engine) ?? {
          utterances: 0,
          lastMs: 0,
          totalMs: 0,
          totalWaitedMs: 0,
          totalAudioSeconds: 0,
          startedAtPause: 0
        }
        totals.utterances += 1
        totals.lastMs = event.roundTripMs
        totals.totalMs += event.roundTripMs
        totals.totalWaitedMs += event.waitedMs
        totals.totalAudioSeconds += event.audioSeconds
        if (event.startedAtPause) totals.startedAtPause += 1
        this.transcription.set(event.engine, totals)
        this.metricsState.transcriptWaitMs.push(event.waitedMs)
        this.emit('USER_VOICE_TRANSCRIBED', event.utteranceId, 'voice', {
          engine: event.engine,
          roundTripMs: event.roundTripMs,
          waitedMs: event.waitedMs,
          engineMs: event.engineMs,
          audioSeconds: Number(event.audioSeconds.toFixed(2)),
          startedAtPause: event.startedAtPause
        })
        this.publish()
        return
      }
      case 'transcriptionEmpty': {
        // Every other step reports itself; without this one an utterance that
        // transcribed to nothing is indistinguishable from one never heard.
        this.hearing = 'idle'
        this.report('That came back with no words in it. Try speaking a little longer.')
        return
      }
      case 'userTranscriptPartial': {
        // Partials are display and barge-in only: they are unstable, and acting
        // on them would run the agent on a half-heard request.
        this.partialTranscript = event.text
        this.emit('USER_VOICE_PARTIAL', this.activeCorrelationId ?? 'none', 'voice', {
          text: event.text
        })
        this.publish()
        return
      }
      case 'userTranscriptFinal': {
        this.track(this.onTranscriptFinal(event.utteranceId, event.text), 'Submitting what you said')
        return
      }
      case 'speechStarted': {
        this.speakingCorrelationId = event.correlationId
        const response = this.responses.get(event.correlationId)
        if (response) response.spokenStatus = 'speaking'
        if (response?.finalizedAt) {
          this.metricsState.responseToSpeechStartMs.push(this.now() - response.finalizedAt)
        }
        this.emit('VOICE_RESPONSE_STARTED', event.correlationId, 'voice', {})
        this.publish()
        return
      }
      case 'speechCompleted': {
        if (this.speakingCorrelationId === event.correlationId) this.speakingCorrelationId = null
        const response = this.responses.get(event.correlationId)
        if (response && response.spokenStatus !== 'interrupted') response.spokenStatus = 'completed'
        this.backchanneling.delete(event.correlationId)
        this.emit('VOICE_RESPONSE_COMPLETED', event.correlationId, 'voice', {})
        this.track(this.runPendingRenewal(), 'Renewing the voice session')
        this.publish()
        return
      }
      case 'speechInterrupted': {
        if (this.speakingCorrelationId === event.correlationId) this.speakingCorrelationId = null
        const response = this.responses.get(event.correlationId)
        if (response) response.spokenStatus = 'interrupted'
        if (this.interruptRequestedAt !== null) {
          this.metricsState.interruptToSpeechStopMs.push(this.now() - this.interruptRequestedAt)
          this.interruptRequestedAt = null
        }
        this.emit('VOICE_RESPONSE_INTERRUPTED', event.correlationId, 'voice', {})
        this.publish()
        return
      }
      case 'modelText': {
        this.onVoiceModelText(event.text)
        return
      }
      case 'sessionError': {
        this.onVoiceSessionError(event.error, event.fatal)
        return
      }
      case 'sessionLimitApproaching': {
        this.track(this.requestRenewal(event.reason), 'Renewing the voice session')
        return
      }
      default:
        return
    }
  }

  /**
   * The user talked over the assistant. Stop the audio; leave the task running.
   * Cancelling here would throw away work the user never asked to abandon.
   */
  private async onBargeIn(): Promise<void> {
    const speaking = this.speakingCorrelationId
    if (!speaking || !this.config.interruptStopsSpeech) return
    this.interruptRequestedAt = this.now()
    await this.deps.voice.stopSpeaking(speaking).catch(() => undefined)
    const response = this.responses.get(speaking)
    if (response) response.spokenStatus = 'interrupted'
    if (this.config.interruptCancelsAgent && this.activeCorrelationId) {
      this.metricsState.agentTasksCancelledByInterruption += 1
      await this.cancelAgentTask(this.activeCorrelationId)
    }
    this.publish()
  }

  private async onTranscriptFinal(utteranceId: string, text: string): Promise<void> {
    const trimmed = text.trim()
    this.hearing = 'idle'
    this.partialTranscript = ''
    if (!trimmed) return
    // One utterance is one turn even if the transcript is delivered twice.
    const published = this.emit(
      'USER_VOICE_FINAL',
      utteranceId,
      'voice',
      { text: trimmed },
      `utterance:${utteranceId}`
    )
    if (!published) {
      this.diagnose('DUPLICATE_IGNORED', utteranceId, 'voice')
      return
    }

    // Connected to nothing: PersonaPlex is answering in its own voice and the
    // conversation is not ours to write to. The transcript is still shown.
    if (this.config.voiceSessionTarget === 'neither') {
      this.partialTranscript = ''
      this.publish()
      return
    }

    // Noise that cleared the gate is not worth a turn. Without this the
    // assistant abandons what it was saying to report that it understood
    // nothing, which is a worse outcome than having ignored the sound.
    if (isTooThinToSubmit(trimmed)) {
      this.report(`Ignored “${trimmed}” — too little to act on.`)
      this.publish()
      return
    }

    const intent = classifyUtterance(trimmed, { taskActive: this.activeCorrelationId !== null })
    if (isControlIntent(intent)) {
      await this.applyControl(intent, trimmed)
      this.publish()
      return
    }

    const correlationId = this.newId('turn')
    this.turnStartedAt.set(correlationId, this.now())
    if (intent === 'correction') await this.supersedeQueued(correlationId)
    await this.submitTurn({
      correlationId,
      text: trimmed,
      source: 'user_voice',
      supersedes: intent === 'correction' ? (this.activeCorrelationId ?? undefined) : undefined
    })
  }

  /** Spoken controls. Recorded, but never submitted as questions. */
  private async applyControl(intent: UtteranceIntent, text: string): Promise<void> {
    if (intent === 'stop_speaking') {
      await this.cancelVoiceOutput()
      return
    }
    // 'cancel_task': the one case where speaking over the assistant also stops
    // the work, because the user said so.
    const target = this.activeCorrelationId
    await this.cancelVoiceOutput()
    if (target) {
      await this.cancelAgentTask(target)
      if (this.conversationId) {
        const note = await this.chat.appendMessage({
          role: 'system',
          source: 'system',
          content: `Cancelled at your request (“${text}”).`,
          correlationId: target,
          status: 'final'
        })
        this.recordMessage(note)
      }
    }
  }

  /** A correction replaces turns still waiting, not the one already running. */
  private async supersedeQueued(bySupersedingId: string): Promise<void> {
    if (this.queue.length === 0) return
    const superseded = this.queue
    this.queue = []
    for (const turn of superseded) {
      const response = this.responses.get(turn.correlationId)
      if (response) response.status = 'superseded'
      await this.patchMessage(turn.userMessageId, {
        status: 'superseded',
        metadata: { queued: false, supersededBy: bySupersedingId }
      })
    }
    this.publish()
  }

  /**
   * PersonaPlex generated text of its own. It is untrusted model output: never
   * an assistant message, never a tool command, and never a claim about a task
   * the agent owns.
   */
  private onVoiceModelText(text: string): void {
    this.voiceModelText = `${this.voiceModelText}${text}`.slice(-2000)
    const agentOwnsTurn = this.activeCorrelationId
      ? this.responses.get(this.activeCorrelationId)?.owner === 'agent'
      : false
    if (agentOwnsTurn) {
      this.metricsState.voiceClaimsRejected += 1
      this.diagnose('VOICE_CLAIM_REJECTED', this.activeCorrelationId ?? 'none', 'voice', {
        errorCategory: 'unbacked_voice_claim'
      })
    }
    this.publish()
  }

  private onVoiceSessionError(error: string, fatal: boolean): void {
    this.voiceError = error
    this.diagnose('VOICE_SESSION_ERROR', this.activeCorrelationId ?? 'none', 'voice', {
      errorCategory: fatal ? 'voice_fatal' : 'voice_recoverable'
    })
    if (fatal) {
      // Voice degrades to text. The agent keeps running and its answers keep
      // arriving in the chat.
      this.voiceStatus = 'error'
      this.voiceSessionId = null
      this.speakingCorrelationId = null
      for (const response of this.responses.values()) {
        if (response.spokenStatus === 'requested' || response.spokenStatus === 'speaking') {
          response.spokenStatus = 'failed'
        }
      }
    }
    this.report(`Voice mode: ${error}`)
    this.publish()
  }

  // --- Speech ---------------------------------------------------------------

  private shouldSpeak(response: ResponseState): boolean {
    return (
      this.config.voiceEnabled &&
      this.config.voiceSessionTarget !== 'neither' &&
      this.voiceSessionId !== null &&
      this.voiceStatus === 'live' &&
      response.deliveryTargets === 'both' &&
      this.deps.voice.canSpeak()
    )
  }

  private async requestSpeech(response: ResponseState, text: string): Promise<void> {
    // A "let me check" still playing would collide with the real answer.
    if (this.backchanneling.delete(response.correlationId)) {
      await this.deps.voice.stopSpeaking(response.correlationId).catch(() => undefined)
    }
    response.spokenStatus = 'requested'
    this.emit('VOICE_RESPONSE_REQUESTED', response.correlationId, 'coordinator', {
      messageId: response.authoritativeMessageId
    })
    await this.speakRequest({
      correlationId: response.correlationId,
      text,
      kind: 'authoritative',
      brevityTargetChars: this.config.spokenBrevityTargetChars
    })
  }

  private async speakRequest(request: SpeechRequest): Promise<void> {
    try {
      await this.deps.voice.speak(request)
    } catch (cause) {
      const response = this.responses.get(request.correlationId)
      if (response && request.kind === 'authoritative') response.spokenStatus = 'failed'
      // Speech failing never loses the answer: it is already in the chat.
      this.report(`Could not speak that answer: ${errorText(cause)}`)
      this.diagnose('VOICE_SESSION_ERROR', request.correlationId, 'voice', {
        errorCategory: 'speech_failed'
      })
      this.publish()
    }
  }

  /** Immediate acknowledgment while the agent works. Not an answer. */
  private async acknowledge(correlationId: string): Promise<void> {
    const response = this.responses.get(correlationId)
    if (!response || !this.config.allowVoiceBackchannels) return
    if (response.originSource !== 'user_voice' || !this.shouldSpeak(response)) return
    this.backchanneling.add(correlationId)
    await this.speakRequest({
      correlationId,
      text: BACKCHANNELS[Math.floor(Math.random() * BACKCHANNELS.length)],
      kind: 'backchannel',
      brevityTargetChars: 40
    })
  }

  /** One short, structured progress cue per turn, at most. */
  private async cueStatus(correlationId: string, cue: string): Promise<void> {
    const response = this.responses.get(correlationId)
    if (!response || !this.config.allowVoiceBackchannels) return
    if (response.originSource !== 'user_voice' || !this.shouldSpeak(response)) return
    if (this.statusCued.has(correlationId)) return
    this.statusCued.add(correlationId)
    await this.speakRequest({ correlationId, text: cue, kind: 'status', brevityTargetChars: 60 })
  }

  private async stopSpeechFor(correlationId: string): Promise<void> {
    this.backchanneling.delete(correlationId)
    if (this.speakingCorrelationId !== correlationId && this.speakingCorrelationId !== null) return
    await this.deps.voice.stopSpeaking(correlationId).catch(() => undefined)
    if (this.speakingCorrelationId === correlationId) this.speakingCorrelationId = null
  }

  // --- Cancellation ---------------------------------------------------------
  //
  // Three separate controls. Muting the voice must not end a task, and ending a
  // task must not delete the answer it already produced.

  /** Silence current audio. The task and the stored answer are untouched. */
  async cancelVoiceOutput(): Promise<void> {
    const speaking = this.speakingCorrelationId
    await this.deps.voice.stopSpeaking(speaking ?? undefined).catch(() => undefined)
    if (speaking) {
      const response = this.responses.get(speaking)
      if (response) response.spokenStatus = 'interrupted'
      this.backchanneling.delete(speaking)
      this.emit('VOICE_RESPONSE_INTERRUPTED', speaking, 'coordinator', { requested: true })
    }
    this.speakingCorrelationId = null
    this.publish()
  }

  /**
   * Abandon the answer to one turn: stop its speech and stop whoever is
   * producing it. A stale id — anything but the active turn — is ignored so a
   * late cancellation cannot kill newer work.
   */
  async cancelCurrentResponse(correlationId?: string): Promise<boolean> {
    const target = correlationId ?? this.activeCorrelationId
    if (!target) return false
    if (correlationId && correlationId !== this.activeCorrelationId) {
      this.diagnose('RESPONSE_CANCEL_REQUESTED', correlationId, 'coordinator', {
        errorCategory: 'stale_cancellation'
      })
      return false
    }
    this.emit('RESPONSE_CANCEL_REQUESTED', target, 'coordinator', {})
    const response = this.responses.get(target)
    if (response && response.status !== 'delivered') {
      response.status = 'cancelled'
      response.cancellable = false
      if (response.userMessageId) {
        this.chat.markCancelled(response.userMessageId)
        await this.patchMessage(response.userMessageId, { metadata: { cancelled: true } })
      }
    }
    await this.stopSpeechFor(target)
    if (response?.owner === 'agent') await this.deps.agent.cancelRun(target).catch(() => undefined)
    else this.deps.responder?.cancel(target)
    this.streamingText = ''
    this.finishActive(target)
    this.publish()
    return true
  }

  /**
   * Cancel the agent task itself. Pending spoken delivery for it stops; the
   * authoritative message, if one was already stored, stays in the chat.
   */
  async cancelAgentTask(correlationId?: string): Promise<boolean> {
    const target = correlationId ?? this.activeCorrelationId
    if (!target) return false
    if (correlationId && this.activeCorrelationId && correlationId !== this.activeCorrelationId) {
      this.diagnose('RESPONSE_CANCEL_REQUESTED', correlationId, 'coordinator', {
        errorCategory: 'stale_cancellation'
      })
      return false
    }
    this.emit('RESPONSE_CANCEL_REQUESTED', target, 'coordinator', { scope: 'agent_task' })
    await this.stopSpeechFor(target)
    await this.deps.agent.cancelRun(target).catch(() => undefined)
    const response = this.responses.get(target)
    if (response && response.status !== 'delivered') {
      response.status = 'cancelled'
      response.cancellable = false
    }
    if (this.task?.correlationId === target) {
      this.task = { ...this.task, status: 'cancelled', activeTool: undefined, updatedAt: this.now() }
    }
    this.streamingText = ''
    this.finishActive(target)
    this.publish()
    return true
  }

  // --- Voice session lifecycle ---------------------------------------------

  /** Start a voice session for this conversation, seeded with bounded context. */
  async startVoiceSession(): Promise<void> {
    if (this.voiceSessionId) return
    this.voiceStatus = 'starting'
    this.voiceError = null
    this.publish()
    try {
      const handle = await this.deps.voice.startSession(this.buildContext())
      this.voiceSessionId = handle.id
      this.voiceStartedAt = handle.startedAt
      this.voiceStatus = 'live'
      this.applyAudioOwnership()
    } catch (cause) {
      this.voiceStatus = 'error'
      this.voiceError = errorText(cause)
      this.report(`Voice mode: ${this.voiceError}`)
    }
    this.publish()
  }

  async endVoiceSession(): Promise<void> {
    if (!this.voiceSessionId) return
    await this.deps.voice.endSession().catch(() => undefined)
    this.voiceSessionId = null
    this.voiceStatus = 'off'
    this.speakingCorrelationId = null
    this.publish()
  }

  /** Push refreshed bounded context at the live session. */
  async refreshVoiceContext(directive?: string): Promise<void> {
    if (!this.voiceSessionId) return
    await this.deps.voice.updateContext(this.buildContext(directive)).catch(() => undefined)
  }

  /**
   * Drive the elapsed-duration threshold. Called by the host rather than by an
   * internal timer, so tests control the clock.
   */
  async tick(): Promise<void> {
    if (!this.voiceSessionId || this.voiceStatus !== 'live') return
    if (this.now() - this.voiceStartedAt >= this.config.voiceSessionMaxDurationMs) {
      await this.requestRenewal('voice session duration limit')
    }
  }

  /** Renew at a safe conversational boundary, deferring if mid-turn. */
  async requestRenewal(reason: string): Promise<void> {
    this.pendingRenewal = reason
    await this.runPendingRenewal()
  }

  private atSafeBoundary(): boolean {
    return (
      this.activeCorrelationId === null &&
      this.speakingCorrelationId === null &&
      this.queue.length === 0
    )
  }

  private async runPendingRenewal(): Promise<void> {
    const reason = this.pendingRenewal
    if (!reason || !this.voiceSessionId || !this.atSafeBoundary()) return
    this.pendingRenewal = null
    await this.renewVoiceSession(reason)
  }

  /**
   * Replace the PersonaPlex session while the conversation and the agent session
   * carry on. The agent is deliberately not touched anywhere in here.
   */
  private async renewVoiceSession(reason: string): Promise<void> {
    const previous = this.voiceSessionId
    this.updateSummary()
    try {
      await this.deps.voice.endSession()
      const handle = await this.deps.voice.startSession(this.buildContext())
      this.voiceSessionId = handle.id
      this.voiceStartedAt = handle.startedAt
      this.voiceStatus = 'live'
      this.metricsState.voiceSessionRenewals += 1
      this.emit('VOICE_SESSION_RENEWED', this.activeCorrelationId ?? 'none', 'coordinator', {
        reason,
        previousVoiceSessionId: previous,
        voiceSessionId: handle.id
      })
    } catch (cause) {
      this.onVoiceSessionError(errorText(cause), true)
    }
    this.publish()
  }

  /** Recompute the compact summary and persist it beside the conversation. */
  updateSummary(agentSummary?: string): string {
    const summary = summarizeForVoice(this.messages, {
      limitChars: this.config.voiceContextSummaryLimitChars,
      agentSummary,
      task: this.task
    })
    if (summary !== this.summary) {
      this.summary = summary
      if (this.conversationId) this.deps.persistSummary?.(this.conversationId, summary)
      this.emit('SESSION_SUMMARY_UPDATED', this.activeCorrelationId ?? 'none', 'coordinator', {
        length: summary.length
      })
    }
    return this.summary
  }

  private buildContext(directive?: string): VoiceContext {
    return buildVoiceContext({
      personaInstructions: this.persona,
      conversationSummary: this.summary,
      messages: this.messages,
      task: this.task,
      responseDirective: directive,
      currentStatus: this.task ? `${this.task.status}` : 'idle',
      config: this.config
    })
  }

  // --- Plumbing -------------------------------------------------------------

  private recordMessage(message: ConversationMessage): void {
    this.messages = [...this.messages.filter((entry) => entry.id !== message.id), message]
  }

  private async patchMessage(
    messageId: string,
    patch: { status?: ConversationMessage['status']; metadata?: Record<string, unknown> }
  ): Promise<void> {
    try {
      const updated = await this.chat.updateMessage(messageId, patch)
      this.recordMessage(updated)
    } catch {
      // A failed relabel is cosmetic; the message itself is already stored.
    }
  }

  private emit(
    type: SessionEventType,
    correlationId: string,
    source: EventSource,
    payload: Record<string, unknown>,
    dedupeKey?: string
  ): boolean {
    const event: SessionEvent = {
      eventId: this.newId('evt'),
      conversationId: this.conversationId ?? 'unbound',
      correlationId,
      timestamp: this.now(),
      source,
      type,
      payload
    }
    const published = this.events.emit(event, dedupeKey)
    if (published) this.diagnose(type, correlationId, source)
    return published
  }

  private diagnose(
    eventType: DiagnosticRecord['eventType'],
    correlationId: string,
    source: EventSource,
    extra: { errorCategory?: string; latencyMs?: number } = {}
  ): void {
    const response = this.responses.get(correlationId)
    this.deps.log?.({
      conversationId: this.conversationId ?? 'unbound',
      correlationId,
      eventType,
      source,
      responseOwner: response?.owner,
      agentRunStatus: response?.status,
      voiceSessionId: this.voiceSessionId ?? undefined,
      timestamp: this.now(),
      ...extra
    })
  }

  private publish(): void {
    const snapshot = this.snapshot()
    for (const listener of [...this.listeners]) listener(snapshot)
  }
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

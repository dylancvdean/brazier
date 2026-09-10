/**
 * Session coordinator: one conversation shared by text chat, voice, and the agent.
 */

import type {
  AgentAdapter,
  AgentAdapterEvent,
  ChatAdapter,
  ChatResponder,
  VoiceAdapter,
  VoiceAdapterEvent
} from './adapters'
import { shouldRouteVoiceToBackground } from './backgroundRouting'
import { DEFAULT_INTEGRATION_CONFIG, type IntegrationConfig } from './config'
import { classifyConfirmation } from './confirmation'
import { SessionEventLog } from './eventLog'
import { isTooThinToSubmit } from './echoGuard'
import { classifyUtterance, isControlIntent, type UtteranceIntent } from './interruption'
import type {
  ConversationMessage,
  DiagnosticRecord,
  EventSource,
  MessageSource,
  ResponseOwner,
  ResponseState,
  SessionEvent,
  SessionEventType,
  SessionMetrics,
  TaskState,
  VoiceContext
} from './types'
import { buildVoiceContext, summarizeForVoice } from './voiceContext'
import type { ExecutionLocation } from '../../../agent/core/types'

export type VoiceStatus = 'off' | 'starting' | 'live' | 'error'

export type QueuedTurn = {
  correlationId: string
  text: string
  source: MessageSource
  userMessageId: string
  queuedAt: number
  supersedes?: string
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
  streamingText: string
  partialTranscript: string
  voiceModelText: string
  hearing: 'idle' | 'speaking' | 'transcribing'
  capture: {
    frames: number
    peak: number
    status: string
    gate: number
    noiseFloor: number
    vad: 'silero-v5' | 'energy-fallback'
    speechProbability: number | null
    vadQueueLagMs: number
    vadInferenceMs: number
    vadInferenceP95Ms: number
    vadQueueLagP95Ms: number
    vadProcessedWindows: number
  }
  transcription: TranscriptionCost[]
  pendingApproval: PendingApproval | null
  notice: string | null
}

export type PendingApproval = {
  approvalId: string
  correlationId: string
  tool: string
  summary: string
  risk: string
  environment: 'sandbox' | 'host'
  executionLocation: ExecutionLocation
  askedAt: number
}

export type TranscriptionCost = {
  engine: string
  utterances: number
  lastMs: number
  averageMs: number
  averageWaitMs: number
  startedAtPause: number
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
  private notice: string | null = null
  private hearing: CoordinatorSnapshot['hearing'] = 'idle'
  private capture: CoordinatorSnapshot['capture'] = {
    frames: 0,
    peak: 0,
    status: '',
    gate: 0,
    noiseFloor: 0,
    vad: 'energy-fallback',
    speechProbability: null,
    vadQueueLagMs: 0,
    vadInferenceMs: 0,
    vadInferenceP95Ms: 0,
    vadQueueLagP95Ms: 0,
    vadProcessedWindows: 0
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
  private pendingApproval: PendingApproval | null = null
  private pendingRenewal: string | null = null
  /** A renewal in flight, serialized against tick()/requestRenewal(). */
  private renewing: Promise<void> | null = null
  /** Queue drain in flight, so cancel + runCancelled cannot start two turns. */
  private draining: Promise<void> | null = null
  /** Invalidates an in-flight startSession/renew when End is pressed. */
  private voiceGeneration = 0
  /** The old PersonaPlex stream is silent while an authoritative turn runs. */
  private personaPlexHeldForRouting = false
  /** Explicit Stop speaking, until the next sustained user utterance. */
  private voiceOutputStopped = false

  private readonly listeners = new Set<(snapshot: CoordinatorSnapshot) => void>()
  private readonly metricsState: SessionMetrics = {
    transcriptTexts: [],
    transcriptWaitMs: [],
    transcriptToAgentStartMs: [],
    responseToSpeechStartMs: [],
    interruptToSpeechStopMs: [],
    duplicateEventsIgnored: 0,
    voiceSessionRenewals: 0,
    agentTasksCancelledByInterruption: 0,
    voiceClaimsRejected: 0,
    approvalsSpokenApproved: 0,
    approvalsSpokenDenied: 0,
    approvalsUnclear: 0
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
    return () => {
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
   *
   * Re-attaching the same conversation only refreshes the agent binding. A
   * different conversation cancels in-flight work so late events cannot land
   * on the new thread, and a live voice session renews against the new summary.
   */
  async attach(
    conversationId: string,
    options: { messages?: ConversationMessage[]; summary?: string } = {}
  ): Promise<void> {
    const same = this.conversationId === conversationId
    if (same) {
      if (options.messages) this.messages = options.messages
      if (options.summary !== undefined) this.summary = options.summary
      await this.deps.agent.attachSession(conversationId)
      this.publish()
      return
    }
    await this.abandonActiveWork()
    this.conversationId = conversationId
    this.messages = options.messages ?? []
    this.summary = options.summary ?? ''
    this.voiceModelText = ''
    this.hearing = 'idle'
    await this.deps.agent.attachSession(conversationId)
    if (this.voiceStatus === 'live' || this.voiceStatus === 'starting') {
      this.track(this.requestRenewal('conversation changed'), 'Refreshing voice for the new conversation')
    }
    this.publish()
  }

  /** Drop the active turn and queue without starting whatever was waiting. */
  private async abandonActiveWork(): Promise<void> {
    const queued = this.queue
    this.queue = []
    for (const turn of queued) {
      const response = this.responses.get(turn.correlationId)
      if (response && response.status !== 'delivered') response.status = 'cancelled'
    }
    const active = this.activeCorrelationId
    this.pendingApproval = null
    this.streamingText = ''
    this.partialTranscript = ''
    this.hearing = 'idle'
    this.releasePersonaPlexRoutingHold()
    if (active) {
      const response = this.responses.get(active)
      if (response && response.status !== 'delivered') {
        response.status = 'cancelled'
        response.cancellable = false
      }
      if (response?.owner === 'agent') await this.deps.agent.cancelRun(active).catch(() => undefined)
      else this.deps.responder?.cancel(active)
      this.activeCorrelationId = null
    }
    this.responses.clear()
    this.task = null
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
    const voicePromptChanged =
      (previous.voiceSessionTarget === 'neither') !== (config.voiceSessionTarget === 'neither') ||
      previous.voiceBackgroundRouting !== config.voiceBackgroundRouting
    if (voicePromptChanged) {
      // These settings are launch-prompt rules as well as audio gates.
      // PersonaPlex cannot update a live prompt, so apply the new role at the
      // next safe boundary instead of leaving prompt and gate in disagreement.
      if (this.voiceSessionId) {
        this.applyAudioOwnership()
        this.track(this.requestRenewal('voice routing settings changed'), 'Refreshing voice role')
      } else {
        this.applyAudioOwnership()
      }
    } else if (previous.voiceSessionTarget !== config.voiceSessionTarget) {
      this.applyAudioOwnership()
    }
    this.publish()
  }

  /**
   * PersonaPlex is the only audible voice. Background chat and agent responses
   * are always text; the live stream stays continuous and is muted while a
   * transcript is routed away from it.
   */
  private applyAudioOwnership(): void {
    if (!this.muteOnRouteEnabled()) this.personaPlexHeldForRouting = false
    if (this.personaPlexHeldForRouting) {
      this.deps.voice.setModelAudioEnabled(false)
      return
    }
    if (this.voiceOutputStopped) return
    this.deps.voice.setModelAudioEnabled(true)
  }

  private muteOnRouteEnabled(): boolean {
    return this.config.voiceSessionTarget !== 'neither'
  }

  private holdPersonaPlexForRouting(): void {
    if (!this.muteOnRouteEnabled()) return
    this.personaPlexHeldForRouting = true
    // Always reapply: the adapter intentionally reopens audio when a new
    // sustained utterance starts, including one that arrives while held.
    this.deps.voice.setModelAudioEnabled(false)
  }

  private releasePersonaPlexRoutingHold(): void {
    if (!this.personaPlexHeldForRouting) return
    this.personaPlexHeldForRouting = false
    if (!this.voiceOutputStopped) this.deps.voice.setModelAudioEnabled(true)
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
      hearing: this.hearing,
      capture: this.capture,
      pendingApproval: this.pendingApproval,
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
   * Run work started from an event handler, reporting a failure rather than
   * leaving it as an unhandled rejection.
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
    utteranceId?: string
    supersedes?: string
  }): Promise<string | null> {
    if (!this.conversationId) {
      // Nothing to write into. Say so rather than dropping what the user said.
      this.report('No conversation is open, so that turn was not recorded.')
      return null
    }
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
      status: 'pending',
      cancellable: true,
      originSource: input.source,
      userText: input.text,
      utteranceId: input.utteranceId,
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
        userMessageId: userMessage.id,
        queuedAt: this.now(),
        supersedes: input.supersedes
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
      if (
        this.responses.get(input.correlationId)?.status === 'cancelled' ||
        this.responses.get(input.correlationId)?.status === 'failed'
      ) {
        return
      }
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
      case 'approvalRequired': {
        this.pendingApproval = {
          approvalId: event.approvalId,
          correlationId: event.correlationId,
          tool: event.tool,
          summary: event.summary,
          risk: event.risk,
          environment: event.environment,
          executionLocation: event.executionLocation,
          askedAt: this.now()
        }
        this.emit('APPROVAL_REQUIRED', event.correlationId, 'agent', {
          approvalId: event.approvalId,
          tool: event.tool,
          risk: event.risk,
          environment: event.environment,
          executionLocation: event.executionLocation
        })
        this.track(this.askForApproval(), 'Reading back what it wants to do')
        this.publish()
        return
      }
      case 'approvalResolved': {
        if (this.pendingApproval?.approvalId === event.approvalId) this.pendingApproval = null
        this.publish()
        return
      }
      case 'toolStarted': {
        if (this.task?.correlationId === event.correlationId) {
          this.task = { ...this.task, activeTool: event.tool, updatedAt: this.now() }
        }
        this.emit('TOOL_STARTED', event.correlationId, 'agent', { tool: event.tool })
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
        if (state && state.status !== 'delivered') {
          state.status = 'cancelled'
          state.cancellable = false
        }
        if (this.task?.correlationId === event.correlationId) {
          this.task = { ...this.task, status: 'cancelled', updatedAt: this.now() }
        }
        this.releasePersonaPlexRoutingHold()
        this.streamingText = ''
        this.finishActive(event.correlationId)
        this.publish()
        return
      }
      default:
        return
    }
  }

  /** Store the authoritative answer once. PersonaPlex is not reseeded with it. */
  private async deliverFinal(correlationId: string, text: string): Promise<void> {
    const response = this.responses.get(correlationId)
    if (!response) return
    if (
      response.status === 'delivered' ||
      response.status === 'cancelled' ||
      response.status === 'failed'
    ) {
      return
    }

    const message = await this.chat.appendMessage({
      role: 'assistant',
      source: response.owner === 'agent' ? 'assistant_agent' : 'assistant_chat',
      content: text,
      correlationId,
      status: 'final'
    })
    const latest = this.responses.get(correlationId)
    if (
      !latest ||
      latest.status === 'cancelled' ||
      latest.status === 'failed' ||
      latest.status === 'delivered'
    ) {
      await this.patchMessage(message.id, { status: 'cancelled' })
      return
    }
    this.recordMessage(message)
    latest.authoritativeMessageId = message.id
    latest.status = 'delivered'
    latest.cancellable = false
    latest.finalizedAt = this.now()
    this.streamingText = ''
    if (this.task?.correlationId === correlationId) {
      this.task = { ...this.task, status: 'completed', activeTool: undefined, updatedAt: this.now() }
    }
    this.report(null)
    this.releasePersonaPlexRoutingHold()
    this.finishActive(correlationId)
    this.publish()
  }

  private failResponse(correlationId: string, error: string): void {
    const response = this.responses.get(correlationId)
    if (
      response &&
      (response.status === 'delivered' ||
        response.status === 'cancelled' ||
        response.status === 'failed')
    ) {
      // A cancelled or failed turn has already had its outcome recorded;
      // a late rejection from the chat responder must not rewrite it or
      // drain the queue a second time.
      return
    }
    if (response) {
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
    this.releasePersonaPlexRoutingHold()
    this.report(error)
    this.emit('AGENT_FAILED', correlationId, 'agent', { error })
    this.diagnose('AGENT_FAILED', correlationId, 'agent', { errorCategory: 'agent_run_failed' })
    this.finishActive(correlationId)
  }

  /** Clear the active slot and start whatever was waiting. */
  private finishActive(correlationId: string): void {
    if (this.activeCorrelationId !== correlationId) return
    this.activeCorrelationId = null
    this.releasePersonaPlexRoutingHold()
    this.track(this.drainQueue(), 'Starting the next turn')
  }

  private async drainQueue(): Promise<void> {
    if (this.draining) {
      try {
        await this.draining
      } catch {
        // The in-flight drain reports its own failures.
      }
      if (!this.activeCorrelationId) await this.drainQueue()
      return
    }
    const run = this.drainQueueBody()
    this.draining = run
    try {
      await run
    } finally {
      if (this.draining === run) this.draining = null
    }
  }

  private async drainQueueBody(): Promise<void> {
    if (this.activeCorrelationId) return
    const next = this.queue.shift()
    if (!next) {
      await this.runPendingRenewal()
      return
    }
    const response = this.responses.get(next.correlationId)
    if (!response || response.status === 'superseded' || response.status === 'cancelled') {
      await this.drainQueueBody()
      return
    }
    await this.patchMessage(next.userMessageId, { metadata: { queued: false } })
    await this.startTurn({
      correlationId: next.correlationId,
      text: next.text,
      supersedes: next.supersedes
    })
  }

  // --- Voice events ---------------------------------------------------------

  private onVoiceEvent(event: VoiceAdapterEvent): void {
    switch (event.type) {
      case 'userSpeechStarted': {
        this.voiceOutputStopped = false
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
          noiseFloor: event.noiseFloor,
          vad: event.vad,
          speechProbability: event.speechProbability,
          vadQueueLagMs: event.vadQueueLagMs,
          vadInferenceMs: event.vadInferenceMs,
          vadInferenceP95Ms: event.vadInferenceP95Ms,
          vadQueueLagP95Ms: event.vadQueueLagP95Ms,
          vadProcessedWindows: event.vadProcessedWindows
        }
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
        this.releasePersonaPlexRoutingHold()
        if (event.reason === 'echo') {
          this.publish()
          return
        }
        this.report(
          'That came back with no words even after short-speech recovery. Try it once more or switch ASR engines.'
        )
        return
      }
      case 'utteranceDiscarded': {
        this.hearing = 'idle'
        this.releasePersonaPlexRoutingHold()
        this.publish()
        return
      }
      case 'userTranscriptPartial': {
        // Partials are display and barge-in only: they are unstable, and acting
        // on them would run the agent on a half-heard request.
        this.partialTranscript = event.text
        if (
          this.muteOnRouteEnabled() &&
          shouldRouteVoiceToBackground(event.text, this.config.voiceBackgroundRouting, {
            taskActive: this.activeCorrelationId !== null
          })
        ) {
          // Display remains speculative and never starts work. Muting is safe
          // and reversible: the final transcript reopens local turns.
          this.holdPersonaPlexForRouting()
        }
        this.emit('USER_VOICE_PARTIAL', this.activeCorrelationId ?? 'none', 'voice', {
          text: event.text
        })
        this.publish()
        return
      }
      case 'userTranscriptFinal': {
        this.metricsState.transcriptTexts.push(event.text)
        this.track(this.onTranscriptFinal(event.utteranceId, event.text), 'Submitting what you said')
        return
      }
      case 'modelText': {
        this.onVoiceModelText(event.text)
        return
      }
      case 'sessionError': {
        this.hearing = 'idle'
        if (!event.fatal) this.releasePersonaPlexRoutingHold()
        this.track(
          this.onVoiceSessionError(event.error, event.fatal),
          event.fatal
            ? 'Tearing down the voice session after a fatal error'
            : 'Reporting a recoverable voice error'
        )
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

  /** PersonaPlex handles duplex interruption; only the opt-in task cancel remains. */
  private async onBargeIn(): Promise<void> {
    if (this.config.interruptCancelsAgent && this.activeCorrelationId) {
      this.metricsState.agentTasksCancelledByInterruption += 1
      await this.cancelAgentTask(this.activeCorrelationId)
    }
    this.publish()
  }

  private async onTranscriptFinal(utteranceId: string, text: string): Promise<void> {
    if (this.voiceStatus !== 'live') return
    const trimmed = text.trim()
    this.hearing = 'idle'
    this.partialTranscript = ''
    if (!trimmed) {
      this.releasePersonaPlexRoutingHold()
      return
    }
    // One utterance is one turn even if the transcript is delivered twice. The
    // dedupe key is emitted before the approval branch as well, so a retry with
    // the same utteranceId cannot re-enter the answerApproval path.
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

    // Noise that cleared the gate is not worth a turn. Without this the
    // assistant abandons what it was saying to report that it understood
    // nothing, which is a worse outcome than having ignored the sound.
    if (isTooThinToSubmit(trimmed)) {
      this.releasePersonaPlexRoutingHold()
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

    // A held tool call takes the next thing said, but only after controls:
    // "cancel that" and "stop talking" must not be swallowed as an unclear yes.
    if (this.pendingApproval) {
      await this.answerApproval(trimmed)
      this.publish()
      return
    }

    // Connected to nothing: PersonaPlex is answering in its own voice and the
    // conversation is not ours to write to. The transcript is still shown.
    if (this.config.voiceSessionTarget === 'neither') {
      this.releasePersonaPlexRoutingHold()
      this.partialTranscript = ''
      this.publish()
      return
    }

    // Agent is an explicit destination, not a hint for the background-routing
    // classifier. Otherwise the screen can say "Agent" while lightweight
    // turns are silently left with PersonaPlex.
    const routesToBackground =
      this.config.voiceSessionTarget === 'agent' ||
      shouldRouteVoiceToBackground(trimmed, this.config.voiceBackgroundRouting, {
        taskActive: this.activeCorrelationId !== null
      })
    if (!routesToBackground) {
      this.releasePersonaPlexRoutingHold()
      // PersonaPlex already heard this turn and is answering it. Do not record a
      // dangling user message with no authoritative assistant message; make the
      // local decision observable in the transient status instead.
      this.report(`PersonaPlex-only: “${trimmed}” — background model not called.`)
      this.publish()
      return
    }
    this.holdPersonaPlexForRouting()

    const correlationId = this.newId('turn')
    this.turnStartedAt.set(correlationId, this.now())
    if (intent === 'correction') await this.supersedeQueued(correlationId)
    await this.submitTurn({
      correlationId,
      text: trimmed,
      source: 'user_voice',
      utteranceId,
      supersedes: intent === 'correction' ? (this.activeCorrelationId ?? undefined) : undefined
    })
  }

  // --- Approvals ------------------------------------------------------------

  /** Surface a held call without introducing a second synthesized voice. */
  private async askForApproval(): Promise<void> {
    const pending = this.pendingApproval
    if (!pending) return
    const response = this.responses.get(pending.correlationId)
    if (!response || response.originSource !== 'user_voice') return
    this.report(
      `Waiting for you to allow on ${pending.executionLocation.daemon_display_name}: ${pending.summary}`
    )
  }

  /**
   * Take a spoken answer to a held call.
   *
   * Only an unmistakable yes allows it. Anything else is not a decision: the
   * call stays held and the question stands, because the cost of mishearing a
   * refusal as consent is a command that has already run.
   */
  private async answerApproval(text: string): Promise<void> {
    const pending = this.pendingApproval
    if (!pending) return
    const verdict = classifyConfirmation(text)
    if (verdict === 'unclear') {
      this.metricsState.approvalsUnclear += 1
      this.emit('APPROVAL_UNCLEAR', pending.correlationId, 'voice', { text })
      const message = `That was not a yes or a no, so nothing has run. On ${pending.executionLocation.daemon_display_name}: ${pending.summary}. Say yes to allow it, or no to stop.`
      this.report(message)
      return
    }
    const decision = verdict === 'affirmative' ? 'approve' : 'deny'
    this.pendingApproval = null
    if (decision === 'approve') this.metricsState.approvalsSpokenApproved += 1
    else this.metricsState.approvalsSpokenDenied += 1
    this.emit('APPROVAL_DECIDED', pending.correlationId, 'voice', {
      approvalId: pending.approvalId,
      decision,
      text
    })
    await this.decideApproval(pending, decision, `Spoken answer: “${text}”`)
    // decideApproval restores the held call on failure, so the transcript must
    // not record an action the broker never received.
    if (this.pendingApproval) return
    await this.recordApprovalNote(pending, decision, `voice (“${text}”)`)
  }

  /**
   * Answer a held call from the UI. Same path as the spoken answer, so the
   * conversation records both the same way.
   */
  async resolveApproval(decision: 'approve' | 'deny'): Promise<void> {
    const pending = this.pendingApproval
    if (!pending) return
    this.pendingApproval = null
    await this.decideApproval(
      pending,
      decision,
      decision === 'approve' ? 'Allowed by the UI' : 'Refused by the UI'
    )
    if (this.pendingApproval) {
      this.publish()
      return
    }
    await this.recordApprovalNote(pending, decision, 'the UI')
    this.publish()
  }

  private async recordApprovalNote(
    pending: PendingApproval,
    decision: 'approve' | 'deny',
    via: string
  ): Promise<void> {
    if (!this.conversationId) return
    const note = await this.chat.appendMessage({
      role: 'system',
      source: 'system',
      content:
        decision === 'approve'
          ? `Allowed by ${via}: ${pending.summary}`
          : `Refused by ${via}: ${pending.summary}`,
      correlationId: pending.correlationId,
      status: 'final'
    })
    this.recordMessage(note)
  }

  private async decideApproval(
    pending: PendingApproval,
    decision: 'approve' | 'deny',
    note?: string
  ): Promise<void> {
    try {
      await this.deps.agent.decideApproval(
        pending.approvalId,
        decision,
        pending.executionLocation,
        note
      )
    } catch (cause) {
      const message = errorText(cause)
      // The broker already committed if this is a double-decide; treat that as
      // success so the card does not stick while the tool runs.
      if (/already (decided|approved|denied|consumed)/i.test(message)) {
        this.pendingApproval = null
        this.publish()
        return
      }
      // The call is still held; say so rather than letting the session look
      // like it went ahead.
      this.pendingApproval = pending
      this.report(`Could not record that decision: ${message}`)
    }
    this.publish()
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
    if (this.pendingApproval) this.pendingApproval = null
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

  private async onVoiceSessionError(error: string, fatal: boolean): Promise<void> {
    this.voiceError = error
    this.diagnose('VOICE_SESSION_ERROR', this.activeCorrelationId ?? 'none', 'voice', {
      errorCategory: fatal ? 'voice_fatal' : 'voice_recoverable'
    })
    if (fatal) {
      // Voice degrades to text. The agent keeps running and its answers keep
      // arriving in the chat. PersonaPlex's endSession is the only path that
      // tears down the capture graph, VAD, segmenter, and daemon-side session,
      // so it must run before the local handle is dropped — otherwise the mic
      // keeps listening and endVoiceSession's early-return leaves it running.
      this.voiceGeneration += 1
      this.voiceStatus = 'error'
      this.hearing = 'idle'
      await this.deps.voice.endSession().catch(() => undefined)
      this.voiceSessionId = null
      this.pendingRenewal = null
    }
    this.report(`Voice mode: ${error}`)
    this.publish()
  }

  // --- Cancellation ---------------------------------------------------------
  //
  // Three separate controls. Muting the voice must not end a task, and ending a
  // task must not delete the answer it already produced.

  /** Silence PersonaPlex. The task and the stored answer are untouched. */
  async cancelVoiceOutput(): Promise<void> {
    this.voiceOutputStopped = true
    await this.deps.voice.stopSpeaking().catch(() => undefined)
    this.publish()
  }

  /**
   * Abandon the answer to one turn: stop whoever is producing it. A stale id —
   * anything but the active turn — is ignored so a late cancellation cannot
   * kill newer work.
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
    this.streamingText = ''
    this.releasePersonaPlexRoutingHold()
    if (response?.owner === 'agent') {
      // finishActive runs on runCancelled so a late event cannot drain twice
      // and start two queued turns.
      await this.deps.agent.cancelRun(target).catch(() => undefined)
      if (this.activeCorrelationId === target) this.finishActive(target)
    } else {
      this.deps.responder?.cancel(target)
      this.finishActive(target)
    }
    this.publish()
    return true
  }

  /**
   * Cancel the agent task itself. The authoritative message, if one was already
   * stored, stays in the chat.
   */
  async cancelAgentTask(correlationId?: string): Promise<boolean> {
    const target = correlationId ?? this.activeCorrelationId
    if (!target) return false
    if (correlationId && correlationId !== this.activeCorrelationId) {
      this.diagnose('RESPONSE_CANCEL_REQUESTED', correlationId, 'coordinator', {
        errorCategory: 'stale_cancellation'
      })
      return false
    }
    this.emit('RESPONSE_CANCEL_REQUESTED', target, 'coordinator', { scope: 'agent_task' })
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
    this.releasePersonaPlexRoutingHold()
    this.finishActive(target)
    this.publish()
    return true
  }

  // --- Voice session lifecycle ---------------------------------------------

  /** Start a voice session for this conversation, seeded with bounded context. */
  async startVoiceSession(): Promise<void> {
    if (this.voiceStatus === 'starting' || this.voiceStatus === 'live') return
    const generation = ++this.voiceGeneration
    this.voiceStatus = 'starting'
    this.personaPlexHeldForRouting = false
    this.voiceOutputStopped = false
    this.voiceError = null
    this.pendingRenewal = null
    this.hearing = 'idle'
    this.publish()
    try {
      const handle = await this.deps.voice.startSession(this.buildContext())
      if (generation !== this.voiceGeneration) {
        await this.deps.voice.endSession().catch(() => undefined)
        return
      }
      this.voiceSessionId = handle.id
      this.voiceStartedAt = handle.startedAt
      this.voiceStatus = 'live'
      this.applyAudioOwnership()
    } catch (cause) {
      if (generation !== this.voiceGeneration) return
      this.voiceStatus = 'error'
      this.voiceSessionId = null
      this.voiceError = errorText(cause)
      this.report(`Voice mode: ${this.voiceError}`)
    }
    this.publish()
  }

  async endVoiceSession(): Promise<void> {
    this.voiceGeneration += 1
    this.pendingRenewal = null
    const hadSession = this.voiceSessionId !== null || this.voiceStatus === 'starting'
    if (!hadSession && this.voiceStatus === 'off') return
    await this.deps.voice.endSession().catch(() => undefined)
    this.voiceSessionId = null
    this.personaPlexHeldForRouting = false
    this.voiceOutputStopped = false
    this.hearing = 'idle'
    this.partialTranscript = ''
    this.voiceStatus = 'off'
    this.voiceError = null
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
    if (this.renewing) return
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
    return this.activeCorrelationId === null && this.queue.length === 0
  }

  private async runPendingRenewal(): Promise<void> {
    // Serialize: a renewal already in flight (between endSession and
    // startSession returning) keeps voiceStatus='live' and the old start time,
    // so tick()/requestRenewal() would otherwise pass atSafeBoundary a second
    // time and start a concurrent renewVoiceSession. Wait for the running one
    // and re-evaluate the pending reason; if one is still set, drain it.
    if (this.renewing) {
      try {
        await this.renewing
      } catch {
        // The in-flight renewal reports its own failures.
      }
      if (this.pendingRenewal) await this.runPendingRenewal()
      return
    }
    const reason = this.pendingRenewal
    if (!reason || !this.voiceSessionId || !this.atSafeBoundary()) return
    this.pendingRenewal = null
    const run = this.renewVoiceSession(reason)
    this.renewing = run
    try {
      await run
    } finally {
      if (this.renewing === run) this.renewing = null
    }
    if (this.pendingRenewal) await this.runPendingRenewal()
  }

  /**
   * Replace the PersonaPlex session while the conversation and the agent session
   * carry on. The agent is deliberately not touched anywhere in here.
   */
  private async renewVoiceSession(reason: string): Promise<void> {
    const previous = this.voiceSessionId
    const generation = ++this.voiceGeneration
    this.updateSummary()
    try {
      this.voiceSessionId = null
      await this.deps.voice.endSession()
      if (generation !== this.voiceGeneration) return
      const handle = await this.deps.voice.startSession(this.buildContext())
      if (generation !== this.voiceGeneration) {
        await this.deps.voice.endSession().catch(() => undefined)
        return
      }
      this.voiceSessionId = handle.id
      this.voiceStartedAt = handle.startedAt
      this.voiceStatus = 'live'
      this.applyAudioOwnership()
      this.metricsState.voiceSessionRenewals += 1
      this.emit('VOICE_SESSION_RENEWED', this.activeCorrelationId ?? 'none', 'coordinator', {
        reason,
        previousVoiceSessionId: previous,
        voiceSessionId: handle.id
      })
    } catch (cause) {
      if (generation !== this.voiceGeneration) return
      await this.onVoiceSessionError(errorText(cause), true)
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

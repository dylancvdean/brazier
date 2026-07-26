/**
 * In-memory adapters for the coordinator tests.
 *
 * Test-only: nothing in the application imports this file. They implement the
 * same interfaces the real adapters do, so a test drives the coordinator
 * through exactly the boundary production code uses.
 */

import type {
  AgentAdapter,
  AgentAdapterEvent,
  AgentRunStatusReport,
  AgentTurnRequest,
  ChatAdapter,
  ChatResponder,
  VoiceAdapter,
  VoiceAdapterEvent,
  VoiceSessionHandle
} from './adapters'
import type {
  ConversationMessage,
  MessagePatch,
  NewMessage,
  SpeechRequest,
  VoiceContext
} from './types'
import type {
  PersonaPlexHandoffRequest,
  PersonaPlexHandoffStrategy
} from './personaplexHandoff'

export class FakeChat implements ChatAdapter {
  messages: ConversationMessage[] = []
  statuses: Array<string | null> = []
  queued: string[] = []
  cancelled: string[] = []
  private next = 0

  constructor(private readonly conversationId = 'conv-1') {}

  async appendMessage(message: NewMessage): Promise<ConversationMessage> {
    this.next += 1
    const stored: ConversationMessage = {
      ...message,
      id: `msg-${this.next}`,
      conversationId: this.conversationId,
      createdAt: message.createdAt ?? new Date(this.next * 1000).toISOString()
    }
    this.messages.push(stored)
    return stored
  }

  async updateMessage(messageId: string, patch: MessagePatch): Promise<ConversationMessage> {
    const index = this.messages.findIndex((entry) => entry.id === messageId)
    if (index < 0) throw new Error(`no such message ${messageId}`)
    const updated: ConversationMessage = {
      ...this.messages[index],
      ...(patch.content === undefined ? {} : { content: patch.content }),
      ...(patch.status === undefined ? {} : { status: patch.status }),
      metadata: { ...this.messages[index].metadata, ...patch.metadata }
    }
    this.messages[index] = updated
    return updated
  }

  showStatus(status: string | null): void {
    this.statuses.push(status)
  }

  markQueued(messageId: string): void {
    this.queued.push(messageId)
  }

  markCancelled(messageId: string): void {
    this.cancelled.push(messageId)
  }

  /** Assistant messages that are the authoritative answer to some turn. */
  assistantMessages(): ConversationMessage[] {
    return this.messages.filter((entry) => entry.role === 'assistant')
  }
}

export class FakeAgent implements AgentAdapter {
  sessionId: string | null = 'agent-1'
  submitted: AgentTurnRequest[] = []
  cancelled: string[] = []
  /** Set to reject the next `submitTurn`. */
  failSubmit: string | null = null
  /** Approval decisions the coordinator passed on. */
  decisions: Array<{ approvalId: string; decision: 'approve' | 'deny'; note?: string }> = []
  /** Set to reject the next `decideApproval`. */
  failDecision: string | null = null
  private readonly listeners = new Set<(event: AgentAdapterEvent) => void>()
  private status = new Map<string, AgentRunStatusReport>()

  async attachSession(): Promise<string | null> {
    return this.sessionId
  }

  attachedSessionId(): string | null {
    return this.sessionId
  }

  async submitTurn(request: AgentTurnRequest): Promise<void> {
    if (this.failSubmit) {
      const message = this.failSubmit
      this.failSubmit = null
      throw new Error(message)
    }
    this.submitted.push(request)
    this.status.set(request.correlationId, {
      correlationId: request.correlationId,
      status: 'running'
    })
  }

  async cancelRun(correlationId: string): Promise<void> {
    this.cancelled.push(correlationId)
    this.status.set(correlationId, { correlationId, status: 'cancelled' })
  }

  async decideApproval(
    approvalId: string,
    decision: 'approve' | 'deny',
    note?: string
  ): Promise<void> {
    if (this.failDecision) {
      const message = this.failDecision
      this.failDecision = null
      throw new Error(message)
    }
    this.decisions.push({ approvalId, decision, note })
  }

  getStatus(correlationId: string): AgentRunStatusReport | null {
    return this.status.get(correlationId) ?? null
  }

  subscribe(listener: (event: AgentAdapterEvent) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  /** Push a normalized agent event, as the real adapter would. */
  emit(event: AgentAdapterEvent): void {
    for (const listener of [...this.listeners]) listener(event)
  }

  /** Shorthand: the whole happy path of a tool-free run. */
  completeRun(correlationId: string, text: string): void {
    this.emit({ type: 'runStarted', correlationId })
    this.emit({ type: 'responseFinal', correlationId, text })
  }
}

export class FakeVoice implements VoiceAdapter {
  spoken: SpeechRequest[] = []
  handoffs: Array<{
    request: PersonaPlexHandoffRequest
    strategy: PersonaPlexHandoffStrategy
  }> = []
  stopped: Array<string | undefined> = []
  contexts: VoiceContext[] = []
  sessions: string[] = []
  ended = 0
  speakable = true
  /** Whether PersonaPlex's own audio is audible. */
  modelAudioEnabled = true
  /** Set to reject the next `speak`. */
  failSpeak: string | null = null
  /** Set to reject the next `startSession`. */
  failStart: string | null = null
  private readonly listeners = new Set<(event: VoiceAdapterEvent) => void>()
  private counter = 0

  constructor(private readonly clock: () => number = () => 0) {}

  async startSession(context: VoiceContext): Promise<VoiceSessionHandle> {
    if (this.failStart) {
      const message = this.failStart
      this.failStart = null
      throw new Error(message)
    }
    this.counter += 1
    const id = `voice-${this.counter}`
    this.sessions.push(id)
    this.contexts.push(context)
    return { id, startedAt: this.clock() }
  }

  async updateContext(context: VoiceContext): Promise<void> {
    this.contexts.push(context)
  }

  async handoffResult(
    request: PersonaPlexHandoffRequest,
    strategy: PersonaPlexHandoffStrategy
  ): Promise<VoiceSessionHandle | null> {
    this.handoffs.push({ request, strategy })
    // The production adapter stops the old stream before reopening output on
    // the replacement that receives the checked result.
    if (strategy !== 'continuous') this.modelAudioEnabled = true
    return null
  }

  async speak(request: SpeechRequest): Promise<void> {
    if (this.failSpeak) {
      const message = this.failSpeak
      this.failSpeak = null
      throw new Error(message)
    }
    this.spoken.push(request)
    this.emit({ type: 'speechStarted', correlationId: request.correlationId })
  }

  async stopSpeaking(correlationId?: string): Promise<void> {
    this.stopped.push(correlationId)
  }

  setModelAudioEnabled(enabled: boolean): void {
    this.modelAudioEnabled = enabled
  }

  async endSession(): Promise<void> {
    this.ended += 1
  }

  canSpeak(): boolean {
    return this.speakable
  }

  subscribe(listener: (event: VoiceAdapterEvent) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  emit(event: VoiceAdapterEvent): void {
    for (const listener of [...this.listeners]) listener(event)
  }

  /** Requests of kind `authoritative`, i.e. real answers. */
  authoritative(): SpeechRequest[] {
    return this.spoken.filter((request) => request.kind === 'authoritative')
  }
}

export class FakeResponder implements ChatResponder {
  replies = new Map<string, string>()
  cancelled: string[] = []
  reply = 'A plain chat answer.'

  async respond(request: {
    correlationId: string
    text: string
    onPartial?: (delta: string) => void
  }): Promise<{ text: string }> {
    request.onPartial?.(this.reply.slice(0, 2))
    this.replies.set(request.correlationId, this.reply)
    return { text: this.reply }
  }

  cancel(correlationId: string): void {
    this.cancelled.push(correlationId)
  }
}

/** Deterministic ids and a clock the test advances by hand. */
export function harnessClock(start = 0): { now: () => number; advance: (ms: number) => void } {
  let value = start
  return {
    now: () => value,
    advance: (ms: number) => {
      value += ms
    }
  }
}

export function sequentialIds(): (prefix: string) => string {
  const counters = new Map<string, number>()
  return (prefix) => {
    const next = (counters.get(prefix) ?? 0) + 1
    counters.set(prefix, next)
    return `${prefix}-${next}`
  }
}

/**
 * Shared conversation model for the chat / voice / agent integration.
 *
 * These types are transport-independent on purpose: the coordinator reasons
 * about them and the three adapters translate to and from the existing daemon
 * API, the PersonaPlex socket, and the agent worker bridge. Nothing here
 * imports from those subsystems.
 */

export type ConversationRole = 'user' | 'assistant' | 'tool' | 'system'

/** Which surface produced a message. */
export type MessageSource =
  | 'user_text'
  | 'user_voice'
  | 'assistant_chat'
  | 'assistant_agent'
  | 'assistant_voice'
  | 'tool'
  | 'system'

/**
 * `superseded` marks a turn a later correction replaced; `cancelled` marks one
 * the user stopped. Neither removes the message — the record of what was said
 * survives.
 */
export type MessageStatus = 'partial' | 'final' | 'cancelled' | 'superseded' | 'failed'

export type ConversationMessage = {
  id: string
  conversationId: string
  role: ConversationRole
  source: MessageSource
  content: string
  createdAt: string
  /** Ties a user turn, its authoritative answer, and its spoken rendering. */
  correlationId?: string
  status: MessageStatus
  metadata?: Record<string, unknown>
}

/** A message the coordinator is asking the chat adapter to persist. */
export type NewMessage = Omit<ConversationMessage, 'id' | 'conversationId' | 'createdAt'> & {
  createdAt?: string
}

export type MessagePatch = {
  content?: string
  status?: MessageStatus
  metadata?: Record<string, unknown>
}

// --- Response ownership -----------------------------------------------------

/** Only these two ever own a substantive answer. Voice is never an owner. */
export type ResponseOwner = 'chat' | 'agent'

export type DeliveryTarget = 'text' | 'voice' | 'both'

export type ResponseStatus =
  | 'pending'
  | 'running'
  | 'delivered'
  | 'cancelled'
  | 'failed'
  | 'superseded'

/** How the spoken rendering of an authoritative answer went. */
export type SpokenStatus = 'none' | 'requested' | 'speaking' | 'completed' | 'interrupted' | 'failed'

export type ResponseState = {
  correlationId: string
  owner: ResponseOwner
  deliveryTargets: DeliveryTarget
  status: ResponseStatus
  cancellable: boolean
  authoritativeMessageId?: string
  spokenStatus: SpokenStatus
  /** Voice-originated turns are the ones spoken back by default. */
  originSource: MessageSource
  userMessageId?: string
  createdAt: number
  startedAt?: number
  finalizedAt?: number
}

// --- Task state -------------------------------------------------------------

export type TaskState = {
  correlationId: string
  /** Short, structured, externally supplied. Never model-authored prose. */
  label: string
  status: 'running' | 'completed' | 'failed' | 'cancelled'
  activeTool?: string
  /** Tool results the agent actually reported. The only facts voice may state. */
  confirmedResults: string[]
  updatedAt: number
}

// --- Events -----------------------------------------------------------------

export type SessionEventType =
  | 'USER_TEXT_SUBMITTED'
  | 'USER_VOICE_PARTIAL'
  /** One utterance transcribed: which interface served it, and what it cost. */
  | 'USER_VOICE_TRANSCRIBED'
  | 'USER_VOICE_FINAL'
  | 'AGENT_REQUESTED'
  | 'AGENT_STARTED'
  | 'AGENT_STATUS_UPDATED'
  | 'AGENT_RESPONSE_PARTIAL'
  | 'AGENT_RESPONSE_FINAL'
  | 'AGENT_FAILED'
  | 'TOOL_STARTED'
  | 'TOOL_COMPLETED'
  | 'TOOL_FAILED'
  | 'VOICE_RESPONSE_REQUESTED'
  | 'VOICE_RESPONSE_STARTED'
  | 'VOICE_RESPONSE_COMPLETED'
  | 'VOICE_RESPONSE_INTERRUPTED'
  | 'RESPONSE_CANCEL_REQUESTED'
  | 'SESSION_SUMMARY_UPDATED'
  | 'VOICE_SESSION_RENEWED'
  /** A tool call is held until someone allows it. */
  | 'APPROVAL_REQUIRED'
  | 'APPROVAL_DECIDED'
  /** Something was said in answer, and it was not a yes or a no. */
  | 'APPROVAL_UNCLEAR'

export type EventSource = 'chat' | 'voice' | 'agent' | 'coordinator'

export type SessionEvent = {
  eventId: string
  conversationId: string
  correlationId: string
  timestamp: number
  source: EventSource
  type: SessionEventType
  payload: Record<string, unknown>
}

// --- Voice context ----------------------------------------------------------

/**
 * The bounded context a PersonaPlex session is given. Never the whole history:
 * the agent session stays authoritative and this is only what the voice needs
 * to speak the current turn well.
 */
export type VoiceContext = {
  personaInstructions: string
  behavioralRules: string[]
  conversationSummary: string
  recentTurns: Array<{ role: ConversationRole; source: MessageSource; content: string }>
  activeTaskSummary: string
  currentStatus: string
  responseDirective: string
}

/** What the coordinator asks the voice adapter to say. */
export type SpeechRequest = {
  correlationId: string
  /** Verbatim authoritative text. The adapter must not alter its facts. */
  text: string
  kind: 'authoritative' | 'backchannel' | 'status' | 'error'
  /** Soft target so a long answer is spoken as a summary-then-detail. */
  brevityTargetChars?: number
  pronunciationHints?: Array<{ written: string; spoken: string }>
  speakingStyle?: string
}

// --- Diagnostics ------------------------------------------------------------

export type DiagnosticRecord = {
  conversationId: string
  correlationId: string
  eventType: SessionEventType | 'DUPLICATE_IGNORED' | 'VOICE_CLAIM_REJECTED' | 'VOICE_SESSION_ERROR'
  source: EventSource
  responseOwner?: ResponseOwner
  agentRunStatus?: ResponseStatus
  voiceSessionId?: string
  timestamp: number
  latencyMs?: number
  errorCategory?: string
}

export type SessionMetrics = {
  /**
   * Utterance close to transcript in hand: the silence between someone
   * finishing a sentence and anything at all happening.
   */
  transcriptWaitMs: number[]
  /** Final transcript to agent start. */
  transcriptToAgentStartMs: number[]
  /** Agent final response to speech start. */
  responseToSpeechStartMs: number[]
  /** Interruption to audio stop. */
  interruptToSpeechStopMs: number[]
  duplicateEventsIgnored: number
  voiceSessionRenewals: number
  /** Must stay zero unless the user asked: interruption is not cancellation. */
  agentTasksCancelledByInterruption: number
  /** Voice output discarded for having no authoritative backing. */
  voiceClaimsRejected: number
  /** Held tool calls allowed by a spoken yes. */
  approvalsSpokenApproved: number
  /** Held tool calls refused by a spoken no. */
  approvalsSpokenDenied: number
  /**
   * Answers to a held call that were neither. Worth counting on its own: a
   * confirmation nobody can give in words is a design problem, not a user error.
   */
  approvalsUnclear: number
}

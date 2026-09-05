/** Shared conversation model for chat, voice, and the agent. */

export type ConversationRole = 'user' | 'assistant' | 'tool' | 'system'

export type MessageSource =
  | 'user_text'
  | 'user_voice'
  | 'assistant_chat'
  | 'assistant_agent'
  | 'assistant_voice'
  | 'tool'
  | 'system'

export type MessageStatus = 'partial' | 'final' | 'cancelled' | 'superseded' | 'failed'

export type ConversationMessage = {
  id: string
  conversationId: string
  role: ConversationRole
  source: MessageSource
  content: string
  createdAt: string
  correlationId?: string
  status: MessageStatus
  metadata?: Record<string, unknown>
}

export type NewMessage = Omit<ConversationMessage, 'id' | 'conversationId' | 'createdAt'> & {
  createdAt?: string
}

export type MessagePatch = {
  content?: string
  status?: MessageStatus
  metadata?: Record<string, unknown>
}

export type ResponseOwner = 'chat' | 'agent'

export type ResponseStatus =
  | 'pending'
  | 'running'
  | 'delivered'
  | 'cancelled'
  | 'failed'
  | 'superseded'

export type ResponseState = {
  correlationId: string
  owner: ResponseOwner
  status: ResponseStatus
  cancellable: boolean
  authoritativeMessageId?: string
  originSource: MessageSource
  userText: string
  utteranceId?: string
  userMessageId?: string
  createdAt: number
  startedAt?: number
  finalizedAt?: number
}

export type TaskState = {
  correlationId: string
  label: string
  status: 'running' | 'completed' | 'failed' | 'cancelled'
  activeTool?: string
  confirmedResults: string[]
  updatedAt: number
}

export type SessionEventType =
  | 'USER_TEXT_SUBMITTED'
  | 'USER_VOICE_PARTIAL'
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
  | 'RESPONSE_CANCEL_REQUESTED'
  | 'SESSION_SUMMARY_UPDATED'
  | 'VOICE_SESSION_RENEWED'
  | 'APPROVAL_REQUIRED'
  | 'APPROVAL_DECIDED'
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

export type VoiceContext = {
  personaInstructions: string
  behavioralRules: string[]
  conversationSummary: string
  recentTurns: Array<{ role: ConversationRole; source: MessageSource; content: string }>
  activeTaskSummary: string
  currentStatus: string
  responseDirective: string
}

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
  transcriptTexts: string[]
  transcriptWaitMs: number[]
  transcriptToAgentStartMs: number[]
  responseToSpeechStartMs: number[]
  interruptToSpeechStopMs: number[]
  duplicateEventsIgnored: number
  voiceSessionRenewals: number
  agentTasksCancelledByInterruption: number
  voiceClaimsRejected: number
  approvalsSpokenApproved: number
  approvalsSpokenDenied: number
  approvalsUnclear: number
}

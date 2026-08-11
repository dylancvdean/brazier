/**
 * The integration boundary.
 *
 * The coordinator talks to the three existing subsystems only through these
 * interfaces, so it holds no knowledge of the Moshi wire protocol, the agent
 * worker IPC, or the daemon's REST shapes. Method names follow the integration
 * plan; the implementations in this directory map them onto the real APIs.
 */

import type {
  ConversationMessage,
  MessagePatch,
  MessageSource,
  NewMessage,
  VoiceContext
} from './types'
import type {
  PersonaPlexHandoffRequest,
  PersonaPlexHandoffStrategy
} from './personaplexHandoff'
import type { ExecutionLocation } from '../../../agent/core/types'

// --- Chat -------------------------------------------------------------------

/** Presentation and persistence of the shared conversation. */
export interface ChatAdapter {
  appendMessage(message: NewMessage): Promise<ConversationMessage>
  updateMessage(messageId: string, patch: MessagePatch): Promise<ConversationMessage>
  /** Transient status line: what the agent is doing, or a voice-mode error. */
  showStatus(status: string | null): void
  markQueued(messageId: string): void
  markCancelled(messageId: string): void
}

/**
 * Produces an ordinary non-agent chat answer. Separate from `ChatAdapter`
 * because generating an answer is not presenting one, and because a
 * conversation with no agent session bound still needs a responder.
 */
export interface ChatResponder {
  respond(request: {
    correlationId: string
    text: string
    onPartial?: (delta: string) => void
  }): Promise<{ text: string }>
  cancel(correlationId: string): void
}

// --- Agent ------------------------------------------------------------------

export type AgentTurnRequest = {
  correlationId: string
  text: string
  /** Attributed so the agent transcript shows where a turn came from. */
  source: MessageSource
  /** Set when the turn corrects a request already submitted. */
  supersedes?: string
}

export type AgentRunStatusReport = {
  correlationId: string
  status: 'idle' | 'running' | 'completed' | 'cancelled' | 'failed' | 'awaiting-approval'
  activeTool?: string
}

/** Normalized agent events. One per plan entry, minus runtime specifics. */
export type AgentAdapterEvent =
  | { type: 'runStarted'; correlationId: string }
  | { type: 'statusUpdated'; correlationId: string; status: string; activeTool?: string }
  | { type: 'responsePartial'; correlationId: string; delta: string }
  | { type: 'responseFinal'; correlationId: string; text: string; runId?: string }
  | { type: 'toolStarted'; correlationId: string; toolCallId: string; tool: string }
  | {
      type: 'toolCompleted'
      correlationId: string
      toolCallId: string
      tool: string
      /** Short, factual outcome the voice may state. Not the raw log. */
      outcome: string
    }
  | { type: 'toolFailed'; correlationId: string; toolCallId: string; tool: string; error: string }
  /**
   * The permission broker is holding a call until someone allows it.
   *
   * Normalized out of the agent's approval record because the coordinator has to
   * read it out loud. The immutable daemon identity travels with the held call:
   * a remote approval must never be described as local or approved after its
   * execution host changes.
   */
  | {
      type: 'approvalRequired'
      correlationId: string
      approvalId: string
      tool: string
      summary: string
      risk: string
      environment: 'sandbox' | 'host'
      executionLocation: ExecutionLocation
    }
  | { type: 'approvalResolved'; correlationId: string; approvalId: string }
  | { type: 'runFailed'; correlationId: string; error: string }
  | { type: 'runCancelled'; correlationId: string }

export interface AgentAdapter {
  /** Bind the conversation to an agent session; null when none is available. */
  attachSession(conversationId: string): Promise<string | null>
  /** The session currently bound, without touching it. */
  attachedSessionId(): string | null
  submitTurn(request: AgentTurnRequest): Promise<void>
  cancelRun(correlationId: string): Promise<void>
  /**
   * Answer a held call. The coordinator only ever passes on a decision someone
   * made — it never decides on their behalf, and there is no timeout that turns
   * silence into consent.
   */
  decideApproval(
    approvalId: string,
    decision: 'approve' | 'deny',
    expectedExecutionLocation: ExecutionLocation,
    note?: string
  ): Promise<void>
  getStatus(correlationId: string): AgentRunStatusReport | null
  subscribe(listener: (event: AgentAdapterEvent) => void): () => void
}

// --- Voice ------------------------------------------------------------------

export type VoiceSessionHandle = {
  id: string
  startedAt: number
}

/** Normalized PersonaPlex events. */
export type VoiceAdapterEvent =
  | { type: 'userTranscriptPartial'; utteranceId: string; text: string }
  | { type: 'userTranscriptFinal'; utteranceId: string; text: string }
  /** The user started talking; the coordinator decides whether to duck audio. */
  | { type: 'userSpeechStarted'; utteranceId: string }
  /**
   * What the microphone is delivering, sampled periodically. Reported because
   * "nothing happened" has two very different causes — no frames arriving at
   * all, and frames too quiet to count as speech — and they are not otherwise
   * distinguishable from outside.
   */
  | {
      type: 'captureLevel'
      frames: number
      peak: number
      status: string
      /** The level a frame currently has to clear, which moves with the room. */
      gate: number
      /** What the room is estimated to sound like when nobody is talking. */
      noiseFloor: number
      /** Which detector is deciding whether captured audio is speech. */
      vad: 'silero-v5' | 'energy-fallback'
      /** Most recent model probability, absent while using RMS fallback. */
      speechProbability: number | null
      /** Current queued audio waiting for VAD inference. */
      vadQueueLagMs: number
      /** Most recent 32 ms Silero inference cost. */
      vadInferenceMs: number
      /** Bounded-session window/queue percentiles and exact processed count. */
      vadInferenceP95Ms: number
      vadQueueLagP95Ms: number
      vadProcessedWindows: number
    }
  /** A finished utterance is being transcribed. */
  | { type: 'transcriptionStarted'; utteranceId: string }
  /**
   * What transcribing one utterance cost, and which interface served it.
   *
   * Reported for every outcome, including an empty transcript, because the
   * choice between batch whisper and a resident streaming worker is an open
   * question that only measurement on real hardware can settle — and because a
   * session that has become slow should be able to say so rather than feeling
   * vaguely sluggish.
   */
  | {
      type: 'transcriptionMeasured'
      utteranceId: string
      engine: string
      /** Wall clock from sending the audio to holding the text. */
      roundTripMs: number
      /**
       * How long the turn waited after the utterance closed. This is the part
       * a person feels, and it is not the same as the round trip: a
       * transcription started at a pause has usually finished by then.
       */
      waitedMs: number
      /** What the daemon reports spending on the audio, when it says. */
      engineMs: number | null
      /** Audio length, so cost per second of speech is recoverable. */
      audioSeconds: number
      /** Whether this transcription began before the user stopped talking. */
      startedAtPause: boolean
    }
  /**
   * Transcription returned nothing. Reported rather than dropped: silence is
   * indistinguishable from a pipeline that stopped working.
   */
  | { type: 'transcriptionEmpty'; utteranceId: string }
  | { type: 'speechStarted'; correlationId: string }
  | { type: 'speechCompleted'; correlationId: string }
  | { type: 'speechInterrupted'; correlationId: string }
  /** Text PersonaPlex generated on its own. Never authoritative. */
  | { type: 'modelText'; text: string }
  | { type: 'sessionError'; error: string; fatal: boolean }
  | { type: 'sessionLimitApproaching'; reason: string }

export interface VoiceAdapter {
  startSession(context: VoiceContext): Promise<VoiceSessionHandle>
  updateContext(context: VoiceContext): Promise<void>
  /**
   * Experimentally feed one background result back to PersonaPlex. A returned
   * handle means the strategy replaced the daemon session/process.
   */
  handoffResult(
    request: PersonaPlexHandoffRequest,
    strategy: PersonaPlexHandoffStrategy
  ): Promise<VoiceSessionHandle | null>
  /** Stop audio for one turn, or all audio when no id is given. */
  stopSpeaking(correlationId?: string): Promise<void>
  /**
   * Let PersonaPlex's own voice be heard, or silence it.
   *
   * Normally always enabled because PersonaPlex is the only audible voice. An
   * explicit stop closes the gate until the next sustained user utterance.
   */
  setModelAudioEnabled(enabled: boolean): void
  endSession(): Promise<void>
  /** Whether realtime PersonaPlex audio can run on this host. */
  canSpeak(): boolean
  subscribe(listener: (event: VoiceAdapterEvent) => void): () => void
}

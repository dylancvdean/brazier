/** Configuration for the chat / voice / agent integration. */
export type IntegrationConfig = {
  voiceEnabled: boolean
  /** Speak the answer to a turn the user spoke. */
  speakVoiceOriginatedResponses: boolean
  /** Speak the answer to a turn the user typed. Off by default: the answer is
   *  already on screen, and speaking it interrupts reading. */
  speakTextOriginatedResponses: boolean
  showVoiceTranscripts: boolean
  allowVoiceBackchannels: boolean
  /** Renew the PersonaPlex session after this long. */
  voiceSessionMaxDurationMs: number
  voiceContextRecentTurnLimit: number
  voiceContextSummaryLimitChars: number
  /** Speaking over PersonaPlex stops the audio. */
  interruptStopsSpeech: boolean
  /** Speaking over PersonaPlex does **not** cancel the agent. Only an explicit
   *  request does, so a long task survives a barge-in. */
  interruptCancelsAgent: boolean
  /** Soft brevity target handed to the speech renderer. */
  spokenBrevityTargetChars: number
}

export const DEFAULT_INTEGRATION_CONFIG: IntegrationConfig = {
  voiceEnabled: false,
  speakVoiceOriginatedResponses: true,
  speakTextOriginatedResponses: false,
  showVoiceTranscripts: true,
  allowVoiceBackchannels: true,
  voiceSessionMaxDurationMs: 20 * 60 * 1000,
  voiceContextRecentTurnLimit: 6,
  voiceContextSummaryLimitChars: 1200,
  interruptStopsSpeech: true,
  interruptCancelsAgent: false,
  spokenBrevityTargetChars: 480
}

const STORAGE_KEY = 'brazier.voiceIntegration'

/** Persisted subset of the configuration, merged over the defaults. */
export function readIntegrationConfig(): IntegrationConfig {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_INTEGRATION_CONFIG
    const parsed = JSON.parse(raw) as Partial<IntegrationConfig>
    return { ...DEFAULT_INTEGRATION_CONFIG, ...parsed }
  } catch {
    return DEFAULT_INTEGRATION_CONFIG
  }
}

export function writeIntegrationConfig(config: IntegrationConfig): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config))
  } catch {
    // Best-effort persistence; the session still honours the in-memory value.
  }
}

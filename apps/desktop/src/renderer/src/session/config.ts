/** Where a live voice session sends what the user says. */
import type { VoiceBackgroundRouting } from './backgroundRouting'

export type VoiceSessionTarget = 'agent' | 'chat' | 'neither'

/**
 * Which ASR interface transcribes spoken turns. `auto` takes whichever is
 * installed, preferring whisper because an utterance is submitted whole and
 * that is what it is best at.
 */
export type AsrPreference = 'auto' | 'whisper.cpp' | 'streaming-asr'

/** Configuration for the chat / voice / agent integration. */
export type IntegrationConfig = {
  voiceEnabled: boolean
  voiceSessionTarget: VoiceSessionTarget
  /**
   * Whether a voice transcript also wakes the chat / agent model. PersonaPlex
   * has already heard every turn and remains the immediate conversational path.
   */
  voiceBackgroundRouting: VoiceBackgroundRouting
  asrPreference: AsrPreference
  showVoiceTranscripts: boolean
  /** Renew the PersonaPlex session after this long. */
  voiceSessionMaxDurationMs: number
  voiceContextRecentTurnLimit: number
  voiceContextSummaryLimitChars: number
  /** Speaking over PersonaPlex does **not** cancel the agent. Only an explicit
   *  request does, so a long task survives a barge-in. */
  interruptCancelsAgent: boolean
}

export const DEFAULT_INTEGRATION_CONFIG: IntegrationConfig = {
  voiceEnabled: false,
  voiceSessionTarget: 'chat',
  voiceBackgroundRouting: 'auto',
  asrPreference: 'auto',
  showVoiceTranscripts: true,
  voiceSessionMaxDurationMs: 20 * 60 * 1000,
  voiceContextRecentTurnLimit: 6,
  voiceContextSummaryLimitChars: 1200,
  interruptCancelsAgent: false
}

/**
 * The engine id to ask the daemon for, or undefined to take its default.
 *
 * `auto` prefers whisper, because an utterance is submitted whole and that is
 * what it is best at, and falls back to the Nemotron worker when that is the
 * interface installed. An explicit choice is honoured even when the capability
 * report disagrees, so the resulting error names the real problem rather than
 * silently transcribing with something the user did not pick.
 */
export function resolveAsrEngine(
  preference: AsrPreference,
  available: { batch: boolean; streaming: boolean }
): string | undefined {
  if (preference === 'whisper.cpp') return undefined
  if (preference === 'streaming-asr') return 'streaming-asr'
  return available.batch || !available.streaming ? undefined : 'streaming-asr'
}

const STORAGE_KEY = 'brazier.voiceIntegration'

const CONFIG_KEYS: readonly (keyof IntegrationConfig)[] = [
  'voiceEnabled',
  'voiceSessionTarget',
  'voiceBackgroundRouting',
  'asrPreference',
  'showVoiceTranscripts',
  'voiceSessionMaxDurationMs',
  'voiceContextRecentTurnLimit',
  'voiceContextSummaryLimitChars',
  'interruptCancelsAgent'
]

/** Persisted subset of the configuration, merged over the defaults. */
export function readIntegrationConfig(): IntegrationConfig {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_INTEGRATION_CONFIG
    const parsed = JSON.parse(raw) as Partial<IntegrationConfig>
    const config = { ...DEFAULT_INTEGRATION_CONFIG }
    for (const key of CONFIG_KEYS) {
      if (parsed[key] !== undefined) {
        Object.assign(config, { [key]: parsed[key] })
      }
    }
    return config
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

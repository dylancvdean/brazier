import type { SessionMetrics } from './types'

export const VOICE_QUALIFICATION_PHRASES = [
  'Brazier keeps private work on the selected machine.',
  'Please summarize the blue note beside the window.',
  'Seven quiet birds crossed the valley at sunrise.',
  'Open the project, but do not change any files.',
  'A short pause should not lose the end of this sentence.',
  'Can you hear numbers four, fifteen, and ninety-two?',
  'The quick brown fox walks past the sleeping dog.',
  'Today I need a careful answer, not a fast guess.',
  'Background music must not become a spoken request.',
  'My workspace is on the daemon host, not this laptop.',
  'Explain why the first attempt failed before retrying.',
  'A keyboard click is noise and should remain noise.',
  'Mixed case names include OpenAI, GitHub, and WebSocket.',
  'Wait until I finish speaking before taking the turn.',
  'The temperature is twenty-one point five degrees.',
  'Create a draft plan with three reversible steps.',
  'I can interrupt the answer without cancelling the task.',
  'Paths on another computer cannot open in this file manager.',
  'Please preserve punctuation, dates, and technical terms.',
  'This is the twentieth and final recall sentence.'
] as const

export const VOICE_QUALIFICATION_BUDGETS = {
  speech_recall_min: 0.95,
  false_sustained_interruptions_per_noise_minute_max: 0.1,
  vad_window_p95_ms_max: 32,
  vad_queue_lag_p95_ms_max: 250,
  transcript_wait_p95_ms_max: 1500,
  interrupt_to_speech_stop_p95_ms_max: 350
} as const

export const VOICE_QUALIFICATION_MINIMUMS = {
  noise_minutes: 5,
  vad_samples: 100,
  interruption_samples: 3,
  expected_speech_utterances: VOICE_QUALIFICATION_PHRASES.length
} as const

function normalizedWords(value: string): Set<string> {
  return new Set(
    value
      .toLocaleLowerCase('en-US')
      .replace(/[^a-z0-9]+/g, ' ')
      .trim()
      .split(/\s+/)
      .filter(Boolean)
  )
}

/**
 * Count fixed phrases actually represented by ASR output, in order. Merely
 * opening the VAD gate is not speech recall: a report must prove that distinct
 * corpus sentences made it through transcription.
 */
export function countRecognizedQualificationPhrases(transcripts: string[]): number {
  const expected = VOICE_QUALIFICATION_PHRASES.map(normalizedWords)
  const actual = transcripts.map(normalizedWords).filter((words) => words.size > 0)
  // Longest ordered fuzzy match: a missed corpus sentence or a stray ASR
  // segment may be skipped without causing every later sentence to disappear.
  const matches = Array.from({ length: expected.length + 1 }, () =>
    Array<number>(actual.length + 1).fill(0)
  )
  for (let phrase = 1; phrase <= expected.length; phrase += 1) {
    for (let transcript = 1; transcript <= actual.length; transcript += 1) {
      const expectedWords = expected[phrase - 1]
      const actualWords = actual[transcript - 1]
      const overlap = [...expectedWords].filter((word) => actualWords.has(word)).length
      const isMatch =
        overlap / expectedWords.size >= 0.7 && overlap / actualWords.size >= 0.5
      matches[phrase][transcript] = Math.max(
        matches[phrase - 1][transcript],
        matches[phrase][transcript - 1],
        isMatch ? matches[phrase - 1][transcript - 1] + 1 : 0
      )
    }
  }
  return matches[expected.length][actual.length]
}

export type VoiceQualificationHost = {
  commit: string
  platform: 'macos' | 'linux' | 'windows'
  arch: string
  memory_gib: number
  gpu_vram_gib: number | null
  gpu_vendor: string | null
}

export type VoiceQualificationInput = {
  host: VoiceQualificationHost
  microphoneClass: 'built-in' | 'usb'
  expectedSpeechUtterances: number
  recognizedSpeechUtterances: number
  noiseMinutes: number
  falseNoiseUtterances: number
  metrics: Pick<
    SessionMetrics,
    'transcriptWaitMs' | 'interruptToSpeechStopMs'
  >
  vadWindowP95Ms: number
  vadQueueLagP95Ms: number
  vadSamples: number
  captureVad: string
  models: { voice: string; background: string }
  createdAt?: string
}

function percentile(values: number[], fraction: number): number {
  if (values.length === 0) return Number.POSITIVE_INFINITY
  const sorted = values.filter(Number.isFinite).sort((left, right) => left - right)
  if (sorted.length === 0) return Number.POSITIVE_INFINITY
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)]
}

function hostId(host: VoiceQualificationHost): string {
  if (host.platform === 'macos' && host.arch === 'arm64') return 'macos-apple-silicon'
  if (host.platform === 'linux' && host.arch === 'x64') return 'linux-nvidia-x64'
  return `${host.platform}-${host.arch}`
}

async function corpusSha256(): Promise<string> {
  const canonical = `${VOICE_QUALIFICATION_PHRASES.join('\n')}\n`
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(canonical))
  return [...new Uint8Array(bytes)].map((value) => value.toString(16).padStart(2, '0')).join('')
}

/** Build the exact evidence schema consumed by the release gate. */
export async function buildVoiceQualificationResult(input: VoiceQualificationInput) {
  const speechRecall =
    input.recognizedSpeechUtterances / Math.max(1, input.expectedSpeechUtterances)
  const falseInterruptions = input.falseNoiseUtterances / Math.max(input.noiseMinutes, Number.EPSILON)
  const metrics = {
    speech_recall: Math.min(1, speechRecall),
    false_sustained_interruptions_per_noise_minute: falseInterruptions,
    vad_window_p95_ms: input.vadWindowP95Ms,
    vad_queue_lag_p95_ms: input.vadQueueLagP95Ms,
    transcript_wait_p95_ms: percentile(input.metrics.transcriptWaitMs, 0.95),
    interrupt_to_speech_stop_p95_ms: percentile(input.metrics.interruptToSpeechStopMs, 0.95)
  }
  const samples = {
    expected_speech_utterances: input.expectedSpeechUtterances,
    recognized_speech_utterances: input.recognizedSpeechUtterances,
    noise_minutes: input.noiseMinutes,
    false_noise_utterances: input.falseNoiseUtterances,
    vad_samples: input.vadSamples,
    transcript_samples: input.metrics.transcriptWaitMs.length,
    interruption_samples: input.metrics.interruptToSpeechStopMs.length
  }
  const passed =
    input.captureVad === 'silero-v5' &&
    input.host.memory_gib >= 16 &&
    (input.host.platform !== 'linux' || (input.host.gpu_vram_gib ?? 0) >= 12) &&
    (input.host.platform !== 'linux' || input.host.gpu_vendor === 'nvidia') &&
    input.recognizedSpeechUtterances <= input.expectedSpeechUtterances &&
    samples.transcript_samples >= input.recognizedSpeechUtterances &&
    input.expectedSpeechUtterances >= VOICE_QUALIFICATION_MINIMUMS.expected_speech_utterances &&
    samples.noise_minutes >= VOICE_QUALIFICATION_MINIMUMS.noise_minutes &&
    samples.vad_samples >= VOICE_QUALIFICATION_MINIMUMS.vad_samples &&
    samples.interruption_samples >= VOICE_QUALIFICATION_MINIMUMS.interruption_samples &&
    metrics.speech_recall >= VOICE_QUALIFICATION_BUDGETS.speech_recall_min &&
    metrics.false_sustained_interruptions_per_noise_minute <=
      VOICE_QUALIFICATION_BUDGETS.false_sustained_interruptions_per_noise_minute_max &&
    metrics.vad_window_p95_ms <= VOICE_QUALIFICATION_BUDGETS.vad_window_p95_ms_max &&
    metrics.vad_queue_lag_p95_ms <= VOICE_QUALIFICATION_BUDGETS.vad_queue_lag_p95_ms_max &&
    metrics.transcript_wait_p95_ms <= VOICE_QUALIFICATION_BUDGETS.transcript_wait_p95_ms_max &&
    metrics.interrupt_to_speech_stop_p95_ms <=
      VOICE_QUALIFICATION_BUDGETS.interrupt_to_speech_stop_p95_ms_max

  return {
    schema_version: 1 as const,
    kind: 'voice-hardware' as const,
    commit: input.host.commit,
    passed,
    host_id: hostId(input.host),
    platform: input.host.platform,
    arch: input.host.arch,
    microphone_class: input.microphoneClass,
    memory_gib: input.host.memory_gib,
    gpu_vram_gib: input.host.gpu_vram_gib,
    gpu_vendor: input.host.gpu_vendor,
    corpus_sha256: await corpusSha256(),
    capture_vad: input.captureVad,
    created_at: input.createdAt ?? new Date().toISOString(),
    models: input.models,
    samples,
    metrics
  }
}

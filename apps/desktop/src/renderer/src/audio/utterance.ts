/**
 * Utterance segmentation for voice input.
 *
 * PersonaPlex sends back its own speech as text, never the user's, so the
 * shared-conversation mode has to produce the user transcript itself. This
 * splits the captured microphone stream into utterances on energy and silence
 * and hands each finished one to the caller, which transcribes it through the
 * daemon's ASR endpoint.
 *
 * Deliberately simple: a fixed threshold with hold-in and hold-out counters.
 * The goal is turn boundaries, not voice activity research, and every parameter
 * is in frames so the behaviour is testable without audio hardware.
 */

export type UtteranceSegmenterOptions = {
  /** RMS above which a frame counts as speech. */
  threshold?: number
  /** Consecutive speech frames before an utterance opens. */
  framesToOpen?: number
  /** Consecutive silent frames that close it. */
  framesToClose?: number
  /** Utterances with less voiced audio than this are noise, not turns. */
  minimumFrames?: number
  /** Hard cap so a stuck-open gate cannot grow without bound. */
  maximumFrames?: number
}

export type UtteranceSegmenterHandlers = {
  /** An utterance began. Used for barge-in, before any transcript exists. */
  onSpeechStart?: (utteranceId: string) => void
  /** A finished utterance, as mono PCM at `sampleRate`. */
  onUtterance?: (utterance: { id: string; samples: Float32Array; sampleRate: number }) => void
  /**
   * An utterance opened and was then thrown away. Reported because it is
   * otherwise indistinguishable from speech never having been detected.
   */
  onDiscarded?: (utteranceId: string, reason: string) => void
}

/** The level speech has to clear, exposed so the UI can say what it is. */
export const SPEECH_THRESHOLD = 0.006

const DEFAULTS: Required<UtteranceSegmenterOptions> = {
  // Low enough for a quiet microphone at modest gain. A fixed 0.02 was above
  // some working setups entirely, and a gate that never opens produces no
  // utterance, no transcript, and no error — the session simply ignores you.
  threshold: SPEECH_THRESHOLD,
  // At 20 ms per frame: 60 ms to open, 700 ms of silence to close, 200 ms of
  // voiced audio to count as a turn, 30 s cap.
  framesToOpen: 3,
  framesToClose: 35,
  minimumFrames: 10,
  maximumFrames: 1500
}

/** Silence kept at the end of an utterance, so ASR sees a clean release. */
const TRAILING_SILENCE_FRAMES = 5

/** Loudness of one captured frame, on the same scale as `threshold`. */
export function frameRms(samples: Float32Array): number {
  return rms(samples)
}

function rms(samples: Float32Array): number {
  let sum = 0
  for (let index = 0; index < samples.length; index += 1) sum += samples[index] * samples[index]
  return Math.sqrt(sum / Math.max(1, samples.length))
}

export class UtteranceSegmenter {
  private readonly options: Required<UtteranceSegmenterOptions>
  private readonly handlers: UtteranceSegmenterHandlers
  private frames: Float32Array[] = []
  private sampleRate = 24000
  private speechRun = 0
  private silenceRun = 0
  private voicedFrames = 0
  private open = false
  private currentId: string | null = null
  private counter = 0

  constructor(handlers: UtteranceSegmenterHandlers, options: UtteranceSegmenterOptions = {}) {
    this.handlers = handlers
    this.options = { ...DEFAULTS, ...options }
  }

  /** Feed one captured frame. */
  push(samples: Float32Array, sampleRate: number): void {
    this.sampleRate = sampleRate
    const loud = rms(samples) >= this.options.threshold

    if (!this.open) {
      this.speechRun = loud ? this.speechRun + 1 : 0
      // Keep a short pre-roll so the utterance does not start clipped.
      this.frames.push(samples.slice())
      if (this.frames.length > this.options.framesToOpen * 2) this.frames.shift()
      if (this.speechRun >= this.options.framesToOpen) {
        this.open = true
        this.silenceRun = 0
        this.voicedFrames = this.speechRun
        this.counter += 1
        this.currentId = `utt-${this.counter}-${Date.now().toString(36)}`
        this.handlers.onSpeechStart?.(this.currentId)
      }
      return
    }

    this.frames.push(samples.slice())
    if (loud) this.voicedFrames += 1
    this.silenceRun = loud ? 0 : this.silenceRun + 1
    if (this.silenceRun >= this.options.framesToClose) {
      this.close()
      return
    }
    if (this.frames.length >= this.options.maximumFrames) this.close()
  }

  /** Force the current utterance closed, e.g. when the microphone is muted. */
  flush(): void {
    if (this.open) this.close()
    this.reset()
  }

  private close(): void {
    const id = this.currentId
    const voiced = this.voicedFrames
    // Long trailing silence is what closed the utterance, not part of it.
    const trailing = Math.max(0, this.silenceRun - TRAILING_SILENCE_FRAMES)
    const frames = this.frames.slice(0, Math.max(0, this.frames.length - trailing))
    this.reset()
    if (!id) return
    if (voiced < this.options.minimumFrames) {
      this.handlers.onDiscarded?.(
        id,
        `only ${voiced} voiced frames, needs ${this.options.minimumFrames}`
      )
      return
    }
    const total = frames.reduce((sum, frame) => sum + frame.length, 0)
    const samples = new Float32Array(total)
    let offset = 0
    for (const frame of frames) {
      samples.set(frame, offset)
      offset += frame.length
    }
    this.handlers.onUtterance?.({ id, samples, sampleRate: this.sampleRate })
  }

  private reset(): void {
    this.frames = []
    this.speechRun = 0
    this.silenceRun = 0
    this.voicedFrames = 0
    this.open = false
    this.currentId = null
  }
}

/**
 * Trailing silence a streaming decoder needs to emit its last words.
 *
 * Nemotron consumes lookahead frames before committing a token, so audio that
 * ends the moment speech does never flushes the tail: "which test is failing"
 * comes back as "which test". Measured against the worker — 300 ms is not
 * enough and 800 ms is, so this leaves margin.
 */
const FLUSH_SILENCE_MS = 1000

/** Append silence, so the tail of an utterance is not left undecoded. */
export function padTrailingSilence(
  samples: Float32Array,
  sampleRate: number,
  milliseconds = FLUSH_SILENCE_MS
): Float32Array {
  const padded = new Float32Array(samples.length + Math.round((sampleRate * milliseconds) / 1000))
  padded.set(samples)
  return padded
}

/** Wrap mono float samples as a 16-bit PCM WAV, which the ASR path accepts. */
export function encodeWav(samples: Float32Array, sampleRate: number): Uint8Array {
  const bytes = new Uint8Array(44 + samples.length * 2)
  const view = new DataView(bytes.buffer)
  const writeAscii = (offset: number, text: string): void => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index))
    }
  }
  writeAscii(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeAscii(8, 'WAVE')
  writeAscii(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  writeAscii(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let index = 0; index < samples.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index]))
    view.setInt16(44 + index * 2, Math.round(clamped * 32767), true)
  }
  return bytes
}

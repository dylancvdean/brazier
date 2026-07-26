/**
 * Streaming Silero VAD v5 over the microphone frames VoiceStream already owns.
 *
 * PersonaPlex wants 24 kHz / 20 ms frames while Silero wants 16 kHz / 512-sample
 * windows. The bridge keeps both streams aligned: each ONNX probability is
 * apportioned to the original capture frames it covered, and only then is that
 * frame handed to the utterance segmenter. This adds roughly one model window
 * of latency without opening a second microphone graph.
 */

import vadModelUrl from '@ricky0123/vad-web/dist/silero_vad_v5.onnx?url'
import * as ort from 'onnxruntime-web/wasm'
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'

const VAD_SAMPLE_RATE = 16_000
const VAD_WINDOW_SAMPLES = 512

export type VadFrame = {
  samples: Float32Array
  sampleRate: number
  speechProbability: number
}

export type VadModel = {
  process(samples: Float32Array): Promise<number>
  release(): Promise<void>
}

/**
 * The small stateful ONNX model. Kept behind an interface so buffering and
 * failure behavior can be tested without loading WebAssembly in Vitest.
 */
export class SileroVadModel implements VadModel {
  private state: ort.Tensor

  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly sampleRate: ort.Tensor
  ) {
    this.state = SileroVadModel.newState()
  }

  static async create(): Promise<SileroVadModel> {
    // A single tiny recurrent model does not benefit from threads, and forcing
    // one avoids SharedArrayBuffer/cross-origin requirements in the renderer.
    ort.env.wasm.numThreads = 1
    ort.env.wasm.proxy = false
    // ONNX Runtime otherwise infers this from its generated module URL. In the
    // Electron dev renderer that URL belongs to Vite's optimized dependency,
    // and the inferred sibling request falls through to index.html — the WASM
    // compiler then sees "<!doctype" instead of the binary magic word. An
    // explicit asset import lets Vite serve the real file in development and
    // rewrite it to the packaged hashed asset in production.
    ort.env.wasm.wasmPaths = { wasm: ortWasmUrl }
    ort.env.logLevel = 'error'
    const session = await ort.InferenceSession.create(vadModelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    })
    return new SileroVadModel(session, new ort.Tensor('int64', [BigInt(VAD_SAMPLE_RATE)]))
  }

  private static newState(): ort.Tensor {
    return new ort.Tensor('float32', new Float32Array(2 * 128), [2, 1, 128])
  }

  async process(samples: Float32Array): Promise<number> {
    if (samples.length !== VAD_WINDOW_SAMPLES) {
      throw new Error(`Silero VAD needs ${VAD_WINDOW_SAMPLES} samples, received ${samples.length}.`)
    }
    const input = new ort.Tensor('float32', samples, [1, samples.length])
    const previousState = this.state
    try {
      const output = await this.session.run({
        input,
        state: previousState,
        sr: this.sampleRate
      })
      const probability = output.output?.data[0]
      const nextState = output.stateN
      if (typeof probability !== 'number' || !nextState) {
        throw new Error('Silero VAD returned an incomplete result.')
      }
      this.state = nextState
      output.output?.dispose()
      return Math.max(0, Math.min(1, probability))
    } finally {
      input.dispose()
      if (this.state !== previousState) previousState.dispose()
    }
  }

  async release(): Promise<void> {
    this.state.dispose()
    this.sampleRate.dispose()
    await this.session.release()
  }
}

type PendingFrame = {
  samples: Float32Array
  sampleRate: number
  vadSamples: number
  consumed: number
  weightedProbability: number
}

/**
 * Serializes stateful inference and maps its 32 ms windows back onto capture
 * frames. `push` never waits on ONNX, so the AudioWorklet message handler stays
 * responsive even on a busy machine.
 */
export class StreamingSileroVad {
  private samples: Float32Array = new Float32Array(0)
  private frames: PendingFrame[] = []
  private drainPromise: Promise<void> | null = null
  private released = false
  private modelReleased = false

  constructor(
    private readonly model: VadModel,
    private readonly onFrame: (frame: VadFrame) => void,
    private readonly onError?: (error: string) => void
  ) {}

  push(samples: Float32Array, sampleRate: number): void {
    if (this.released) return
    const resampled = resampleFrame(samples, sampleRate, VAD_SAMPLE_RATE)
    this.samples = append(this.samples, resampled)
    this.frames.push({
      samples: samples.slice(),
      sampleRate,
      vadSamples: resampled.length,
      consumed: 0,
      weightedProbability: 0
    })
    void this.startDrain()
  }

  private startDrain(): Promise<void> {
    if (this.drainPromise) return this.drainPromise
    if (this.released) return Promise.resolve()
    const running = this.drain()
    this.drainPromise = running
    void running.finally(() => {
      if (this.drainPromise === running) this.drainPromise = null
      if (!this.released && this.samples.length >= VAD_WINDOW_SAMPLES) void this.startDrain()
    })
    return running
  }

  private async drain(): Promise<void> {
    try {
      while (!this.released && this.samples.length >= VAD_WINDOW_SAMPLES) {
        const window = this.samples.slice(0, VAD_WINDOW_SAMPLES)
        this.samples = this.samples.slice(VAD_WINDOW_SAMPLES)
        const probability = await this.model.process(window)
        this.distribute(probability, VAD_WINDOW_SAMPLES)
      }
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause)
      this.onError?.(message)
      // A broken model must not leave captured frames waiting forever. Mark the
      // stream released; the adapter switches subsequent frames to RMS fallback.
      this.released = true
      this.frames = []
      this.samples = new Float32Array(0)
      await this.releaseModel().catch(() => undefined)
    }
  }

  private distribute(probability: number, count: number): void {
    let remaining = count
    while (remaining > 0 && this.frames.length > 0) {
      const frame = this.frames[0]
      const available = frame.vadSamples - frame.consumed
      const used = Math.min(available, remaining)
      frame.consumed += used
      frame.weightedProbability += probability * used
      remaining -= used
      if (frame.consumed < frame.vadSamples) continue
      this.frames.shift()
      this.onFrame({
        samples: frame.samples,
        sampleRate: frame.sampleRate,
        speechProbability: frame.weightedProbability / Math.max(1, frame.vadSamples)
      })
    }
  }

  async release(): Promise<void> {
    this.released = true
    this.frames = []
    this.samples = new Float32Array(0)
    await this.drainPromise?.catch(() => undefined)
    await this.releaseModel()
  }

  private async releaseModel(): Promise<void> {
    if (this.modelReleased) return
    this.modelReleased = true
    await this.model.release()
  }
}

/** Linear resampling is sufficient for a binary speech detector and is cheap. */
export function resampleFrame(
  samples: Float32Array,
  sourceRate: number,
  targetRate: number
): Float32Array {
  if (sourceRate === targetRate) return samples.slice()
  if (samples.length === 0 || sourceRate <= 0 || targetRate <= 0) return new Float32Array()
  const outputLength = Math.max(1, Math.round((samples.length * targetRate) / sourceRate))
  const output = new Float32Array(outputLength)
  const scale = sourceRate / targetRate
  for (let index = 0; index < outputLength; index += 1) {
    const position = index * scale
    const left = Math.min(samples.length - 1, Math.floor(position))
    const right = Math.min(samples.length - 1, left + 1)
    const fraction = position - left
    output[index] = samples[left] * (1 - fraction) + samples[right] * fraction
  }
  return output
}

function append(left: Float32Array, right: Float32Array): Float32Array {
  const joined = new Float32Array(left.length + right.length)
  joined.set(left)
  joined.set(right, left.length)
  return joined
}

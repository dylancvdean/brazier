import { createHash } from 'node:crypto'
import { createRequire } from 'node:module'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'
import { performance } from 'node:perf_hooks'
import { pathToFileURL } from 'node:url'

const root = path.resolve(import.meta.dirname, '..')
const corpusPath = path.join(root, 'qualification', 'voice', 'synthetic-noise-corpus.json')
const desktopRequire = createRequire(path.join(root, 'apps', 'desktop', 'package.json'))
const modelPath = desktopRequire.resolve('@ricky0123/vad-web/dist/silero_vad_v5.onnx')

function percentile(values, fraction) {
  if (values.length === 0) return 0
  const sorted = [...values].sort((left, right) => left - right)
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)]
}

function rng(seed) {
  let state = seed >>> 0
  return () => {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    return (state >>> 0) / 0xffffffff
  }
}

function renderScene(scene, sampleRate, windowSamples) {
  const frames = Math.ceil((scene.duration_seconds * sampleRate) / windowSamples)
  const random = rng(scene.seed ?? 1)
  return Array.from({ length: frames }, (_, frameIndex) => {
    const frame = new Float32Array(windowSamples)
    for (let index = 0; index < frame.length; index += 1) {
      const sampleIndex = frameIndex * windowSamples + index
      const seconds = sampleIndex / sampleRate
      if (scene.kind === 'white-noise') frame[index] = (random() * 2 - 1) * scene.amplitude
      else if (scene.kind === 'hum') {
        frame[index] = Math.sin(seconds * Math.PI * 2 * scene.frequency_hz) * scene.amplitude
      } else if (scene.kind === 'impulses') {
        const interval = Math.max(1, Math.round((scene.interval_ms / 1000) * sampleRate))
        const phase = sampleIndex % interval
        if (phase < 80) frame[index] = (random() * 2 - 1) * scene.amplitude * Math.exp(-phase / 12)
      } else if (scene.kind === 'harmonic-noise') {
        const carrier =
          Math.sin(seconds * Math.PI * 2 * 180) +
          0.5 * Math.sin(seconds * Math.PI * 2 * 360) +
          0.25 * Math.sin(seconds * Math.PI * 2 * 720)
        const envelope = 0.4 + 0.6 * Math.sin(seconds * Math.PI * 3) ** 2
        frame[index] = (carrier * envelope + (random() * 2 - 1) * 0.2) * scene.amplitude
      }
    }
    return frame
  })
}

async function main() {
  const ortModule = await import(pathToFileURL(desktopRequire.resolve('onnxruntime-web/wasm')).href)
  const ort = ortModule.default
  const corpusBytes = await readFile(corpusPath)
  const corpus = JSON.parse(corpusBytes)
  const modelBytes = new Uint8Array(await readFile(modelPath))
  ort.env.wasm.numThreads = 1
  ort.env.wasm.proxy = false
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  })
  const sampleRate = new ort.Tensor('int64', [BigInt(corpus.sample_rate)])
  const inferenceTimes = []
  const queueLags = []
  const sceneResults = []
  let falseSustained = 0
  let noiseSeconds = 0

  try {
    for (const scene of corpus.scenes) {
      let state = new ort.Tensor('float32', new Float32Array(2 * 128), [2, 1, 128])
      let speechRunMs = 0
      let interruptionLatched = false
      let sceneFalseSustained = 0
      let simulatedQueueLagMs = 0
      let maxProbability = 0
      for (const frame of renderScene(scene, corpus.sample_rate, corpus.window_samples)) {
        const input = new ort.Tensor('float32', frame, [1, frame.length])
        const startedAt = performance.now()
        const output = await session.run({ input, state, sr: sampleRate })
        const elapsed = performance.now() - startedAt
        inferenceTimes.push(elapsed)
        simulatedQueueLagMs = Math.max(
          0,
          simulatedQueueLagMs + elapsed - (corpus.window_samples / corpus.sample_rate) * 1_000
        )
        queueLags.push(simulatedQueueLagMs)
        const probability = Number(output.output.data[0])
        maxProbability = Math.max(maxProbability, probability)
        if (probability >= 0.5) speechRunMs += 32
        else {
          speechRunMs = 0
          interruptionLatched = false
        }
        if (speechRunMs >= 300 && !interruptionLatched) {
          sceneFalseSustained += 1
          interruptionLatched = true
        }
        const previousState = state
        state = output.stateN
        input.dispose()
        output.output.dispose()
        previousState.dispose()
      }
      state.dispose()
      falseSustained += sceneFalseSustained
      noiseSeconds += scene.duration_seconds
      sceneResults.push({
        id: scene.id,
        false_sustained_interruptions: sceneFalseSustained,
        max_speech_probability: maxProbability
      })
    }
  } finally {
    sampleRate.dispose()
    await session.release()
  }

  const report = {
    schema_version: 1,
    kind: 'voice-corpus',
    corpus_sha256: createHash('sha256').update(corpusBytes).digest('hex'),
    model: '@ricky0123/vad-web silero_vad_v5.onnx',
    metrics: {
      false_sustained_interruptions_per_noise_minute:
        falseSustained / Math.max(noiseSeconds / 60, Number.EPSILON),
      vad_window_p95_ms: percentile(inferenceTimes, 0.95),
      vad_queue_lag_p95_ms: percentile(queueLags, 0.95)
    },
    scenes: sceneResults
  }
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`)

  const budgets = JSON.parse(
    await readFile(path.join(root, 'qualification', 'beta-manifest.json'), 'utf8')
  ).voice_budgets
  if (
    report.metrics.false_sustained_interruptions_per_noise_minute >
      budgets.false_sustained_interruptions_per_noise_minute_max ||
    report.metrics.vad_window_p95_ms > budgets.vad_window_p95_ms_max ||
    report.metrics.vad_queue_lag_p95_ms > budgets.vad_queue_lag_p95_ms_max
  ) {
    throw new Error('Silero VAD synthetic corpus exceeded a beta qualification budget')
  }
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`)
  process.exitCode = 1
})

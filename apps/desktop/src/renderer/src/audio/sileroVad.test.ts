import { describe, expect, it, vi } from 'vitest'

import { StreamingSileroVad, resampleFrame, type VadModel } from './sileroVad'

class FakeModel implements VadModel {
  windows: Float32Array[] = []
  released = false

  constructor(private readonly probabilities: number[]) {}

  async process(samples: Float32Array): Promise<number> {
    this.windows.push(samples)
    return this.probabilities.shift() ?? 0
  }

  async release(): Promise<void> {
    this.released = true
  }
}

describe('resampleFrame', () => {
  it('converts one 24 kHz capture frame into 16 kHz VAD samples', () => {
    const input = Float32Array.from({ length: 480 }, (_, index) => index / 480)
    const output = resampleFrame(input, 24_000, 16_000)

    expect(output).toHaveLength(320)
    expect(output[0]).toBeCloseTo(0)
    expect(output.at(-1)).toBeGreaterThan(0.99)
  })
})

describe('StreamingSileroVad', () => {
  it('maps overlapping model windows back to capture frames in order', async () => {
    const model = new FakeModel([0.8, 0.4])
    const received: Array<{ marker: number; probability: number }> = []
    const stream = new StreamingSileroVad(model, (frame) => {
      received.push({ marker: frame.samples[0], probability: frame.speechProbability })
    })

    for (const marker of [1, 2, 3, 4]) {
      stream.push(new Float32Array(480).fill(marker), 24_000)
    }
    await vi.waitFor(() => expect(model.windows).toHaveLength(2))

    expect(model.windows.every((window) => window.length === 512)).toBe(true)
    expect(received.map((frame) => frame.marker)).toEqual([1, 2, 3])
    expect(received[0].probability).toBeCloseTo(0.8)
    // Frame two straddles the two windows: 192 samples at .8, 128 at .4.
    expect(received[1].probability).toBeCloseTo(0.64)
    expect(received[2].probability).toBeCloseTo(0.4)
    const diagnostics = stream.diagnostics()
    expect(diagnostics.processedWindows).toBe(2)
    expect(diagnostics.p95InferenceMs).toBeGreaterThanOrEqual(0)
    expect(diagnostics.p95QueueLagMs).toBeGreaterThan(0)
  })

  it('reports inference failure once and releases the model', async () => {
    const model: VadModel = {
      process: async () => {
        throw new Error('bad model')
      },
      release: vi.fn(async () => undefined)
    }
    const error = vi.fn()
    const stream = new StreamingSileroVad(model, vi.fn(), error)

    stream.push(new Float32Array(480), 24_000)
    stream.push(new Float32Array(480), 24_000)
    await vi.waitFor(() => expect(error).toHaveBeenCalledWith('bad model'))
    expect(model.release).toHaveBeenCalledOnce()
  })

  it('falls back instead of accumulating unbounded lag behind realtime', async () => {
    let unblock: (() => void) | undefined
    const model: VadModel = {
      process: () => new Promise<number>((resolve) => (unblock = () => resolve(0.5))),
      release: vi.fn(async () => undefined)
    }
    const error = vi.fn()
    const stream = new StreamingSileroVad(model, vi.fn(), error)

    for (let index = 0; index < 110; index += 1) {
      stream.push(new Float32Array(480), 24_000)
    }

    await vi.waitFor(() => expect(error).toHaveBeenCalledOnce())
    expect(error.mock.calls[0][0]).toContain('behind realtime')
    expect(stream.diagnostics().maxQueueLagMs).toBeGreaterThan(2_000)
    expect(model.release).toHaveBeenCalledOnce()
    unblock?.()
  })
})

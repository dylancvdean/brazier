import { describe, expect, it } from 'vitest'

import { UtteranceSegmenter, encodeWav } from './utterance'

const SAMPLE_RATE = 24000
const FRAME = 480

function frame(amplitude: number): Float32Array {
  const samples = new Float32Array(FRAME)
  for (let index = 0; index < FRAME; index += 1) {
    samples[index] = amplitude * Math.sin((index / FRAME) * Math.PI * 8)
  }
  return samples
}

function feed(segmenter: UtteranceSegmenter, amplitude: number, count: number): void {
  for (let index = 0; index < count; index += 1) segmenter.push(frame(amplitude), SAMPLE_RATE)
}

describe('UtteranceSegmenter', () => {
  it('emits one utterance per spoken stretch', () => {
    const starts: string[] = []
    const utterances: Array<{ id: string; length: number }> = []
    const segmenter = new UtteranceSegmenter({
      onSpeechStart: (id) => starts.push(id),
      onUtterance: ({ id, samples }) => utterances.push({ id, length: samples.length })
    })

    feed(segmenter, 0.001, 10)
    expect(starts).toHaveLength(0)

    feed(segmenter, 0.5, 25)
    expect(starts).toHaveLength(1)
    expect(utterances).toHaveLength(0)

    // Silence closes it.
    feed(segmenter, 0.001, 40)
    expect(utterances).toHaveLength(1)
    expect(utterances[0].id).toBe(starts[0])
    expect(utterances[0].length).toBeGreaterThan(25 * FRAME)

    feed(segmenter, 0.5, 25)
    feed(segmenter, 0.001, 40)
    expect(utterances).toHaveLength(2)
    expect(utterances[1].id).not.toBe(utterances[0].id)
  })

  it('does not break an utterance at a short pause', () => {
    const utterances: number[] = []
    const segmenter = new UtteranceSegmenter({
      onUtterance: ({ samples }) => utterances.push(samples.length)
    })
    feed(segmenter, 0.5, 15)
    feed(segmenter, 0.001, 10) // a breath, well under framesToClose
    feed(segmenter, 0.5, 15)
    feed(segmenter, 0.001, 40)
    expect(utterances).toHaveLength(1)
  })

  it('discards a click too short to be a turn', () => {
    const utterances: unknown[] = []
    const segmenter = new UtteranceSegmenter({ onUtterance: (value) => utterances.push(value) })
    feed(segmenter, 0.6, 4)
    feed(segmenter, 0.001, 40)
    expect(utterances).toHaveLength(0)
  })

  it('caps an utterance that never falls silent', () => {
    const lengths: number[] = []
    const segmenter = new UtteranceSegmenter(
      { onUtterance: ({ samples }) => lengths.push(samples.length) },
      { maximumFrames: 50 }
    )
    feed(segmenter, 0.5, 200)
    expect(lengths.length).toBeGreaterThanOrEqual(1)
    expect(lengths[0]).toBeLessThanOrEqual(50 * FRAME)
  })

  it('flushes a partial utterance on demand', () => {
    const utterances: unknown[] = []
    const segmenter = new UtteranceSegmenter({ onUtterance: (value) => utterances.push(value) })
    feed(segmenter, 0.5, 20)
    segmenter.flush()
    expect(utterances).toHaveLength(1)
    // Flushing twice must not emit the same audio again.
    segmenter.flush()
    expect(utterances).toHaveLength(1)
  })
})

describe('encodeWav', () => {
  it('writes a mono 16-bit PCM header the ASR path can read', () => {
    const samples = new Float32Array([0, 0.5, -0.5, 1, -1])
    const wav = encodeWav(samples, SAMPLE_RATE)
    const view = new DataView(wav.buffer)
    const ascii = (offset: number, length: number): string =>
      String.fromCharCode(...wav.subarray(offset, offset + length))

    expect(ascii(0, 4)).toBe('RIFF')
    expect(ascii(8, 4)).toBe('WAVE')
    expect(ascii(12, 4)).toBe('fmt ')
    expect(view.getUint16(20, true)).toBe(1)
    expect(view.getUint16(22, true)).toBe(1)
    expect(view.getUint32(24, true)).toBe(SAMPLE_RATE)
    expect(view.getUint16(34, true)).toBe(16)
    expect(ascii(36, 4)).toBe('data')
    expect(view.getUint32(40, true)).toBe(samples.length * 2)
    expect(wav.length).toBe(44 + samples.length * 2)

    // Full-scale samples clamp instead of wrapping.
    expect(view.getInt16(44 + 6, true)).toBe(32767)
    expect(view.getInt16(44 + 8, true)).toBe(-32767)
  })
})

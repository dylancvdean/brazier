import { describe, expect, it } from 'vitest'

import { UtteranceSegmenter, encodeWav, padTrailingSilence } from './utterance'

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

  /**
   * The failure this guards: a gate above the microphone's actual speech level
   * never opens, and produces no utterance, no transcript, and no error — the
   * session just ignores you. Microphone gain varies by an order of magnitude
   * across machines, so the floor has to suit a quiet one.
   */
  it('opens for speech that is quiet in absolute terms', () => {
    const utterances: unknown[] = []
    const segmenter = new UtteranceSegmenter({ onUtterance: (value) => utterances.push(value) })
    feed(segmenter, 0.0005, 40) // a quiet room
    feed(segmenter, 0.015, 25) // well under the old fixed 0.02 threshold
    feed(segmenter, 0.0005, 40)
    expect(utterances).toHaveLength(1)
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

  /**
   * The pause snapshot exists so transcription can happen inside the silence
   * window instead of after it. That only works if the audio offered at the
   * pause is what the closed utterance turns out to be — otherwise the
   * transcript describes something the user did not finish saying.
   */
  it('offers the utterance at a pause, identical to what closing delivers', () => {
    const pauses: Array<{ id: string; samples: Float32Array; voicedFrames: number }> = []
    const utterances: Array<{ id: string; samples: Float32Array; voicedFrames: number }> = []
    const segmenter = new UtteranceSegmenter({
      onPause: (snapshot) => pauses.push(snapshot),
      onUtterance: (utterance) => utterances.push(utterance)
    })

    feed(segmenter, 0.5, 25)
    expect(pauses).toHaveLength(0)
    feed(segmenter, 0.001, 15) // framesToPause
    expect(pauses).toHaveLength(1)
    expect(utterances).toHaveLength(0)

    feed(segmenter, 0.001, 20) // on to framesToClose
    expect(utterances).toHaveLength(1)
    expect(utterances[0].id).toBe(pauses[0].id)
    expect(utterances[0].voicedFrames).toBe(pauses[0].voicedFrames)
    expect(Array.from(utterances[0].samples)).toEqual(Array.from(pauses[0].samples))
  })

  it('says how much speech there was, so a stale snapshot is recognisable', () => {
    const pauses: number[] = []
    const utterances: number[] = []
    const segmenter = new UtteranceSegmenter({
      onPause: ({ voicedFrames }) => pauses.push(voicedFrames),
      onUtterance: ({ voicedFrames }) => utterances.push(voicedFrames)
    })

    feed(segmenter, 0.5, 25)
    feed(segmenter, 0.001, 15)
    expect(pauses).toHaveLength(1)
    // Speech resumes: the snapshot no longer covers the whole utterance.
    feed(segmenter, 0.5, 25)
    feed(segmenter, 0.001, 40)
    expect(utterances[0]).toBeGreaterThan(pauses[0])
  })

  it('does not offer a pause for every gap in a hesitant sentence', () => {
    const pauses: number[] = []
    const segmenter = new UtteranceSegmenter({ onPause: () => pauses.push(1) })
    feed(segmenter, 0.5, 25)
    feed(segmenter, 0.001, 15)
    expect(pauses).toHaveLength(1)
    for (let round = 0; round < 2; round += 1) {
      feed(segmenter, 0.5, 4) // a word, under framesBetweenPauses
      feed(segmenter, 0.001, 15)
    }
    expect(pauses).toHaveLength(1)

    // Enough new speech, though, and the snapshot is worth taking again.
    feed(segmenter, 0.5, 12)
    feed(segmenter, 0.001, 15)
    expect(pauses).toHaveLength(2)
  })

  it('offers nothing for a sound too short to be a turn', () => {
    const pauses: unknown[] = []
    const segmenter = new UtteranceSegmenter({ onPause: (value) => pauses.push(value) })
    feed(segmenter, 0.6, 4)
    feed(segmenter, 0.001, 20)
    expect(pauses).toHaveLength(0)
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

describe('padTrailingSilence', () => {
  it('appends silence so a streaming decoder can flush the tail', () => {
    const speech = new Float32Array([0.4, -0.4, 0.4])
    const padded = padTrailingSilence(speech, 24000, 500)
    expect(padded.length).toBe(3 + 12000)
    // Compared with tolerance: these are float32, so 0.4 is not exactly 0.4.
    expect(padded[0]).toBeCloseTo(0.4)
    expect(padded[1]).toBeCloseTo(-0.4)
    expect(padded[2]).toBeCloseTo(0.4)
    expect(padded[3]).toBe(0)
    expect(padded[padded.length - 1]).toBe(0)
  })

  it('defaults to enough silence for the measured worker', () => {
    // 300 ms was not enough against the real worker and 800 ms was.
    const padded = padTrailingSilence(new Float32Array(10), 16000)
    expect(padded.length - 10).toBeGreaterThanOrEqual(16000 * 0.8)
  })
})

describe('barge-in versus capture', () => {
  /**
   * The complaint this fixes: any small sound stopped the assistant mid-sentence
   * and cost it its turn, because opening an utterance and interrupting were the
   * same event. Capturing is cheap and reversible; interrupting is not.
   */
  it('captures a short sound without interrupting', () => {
    const started: string[] = []
    const sustained: string[] = []
    const segmenter = new UtteranceSegmenter({
      onSpeechStart: (id) => started.push(id),
      onSustainedSpeech: (id) => sustained.push(id)
    })
    feed(segmenter, 0.5, 8) // ~160 ms: a cough
    feed(segmenter, 0.001, 40)
    expect(started).toHaveLength(1)
    expect(sustained).toHaveLength(0)
  })

  it('interrupts once speech is sustained', () => {
    const sustained: string[] = []
    const segmenter = new UtteranceSegmenter({ onSustainedSpeech: (id) => sustained.push(id) })
    feed(segmenter, 0.5, 20) // ~400 ms: someone talking
    expect(sustained).toHaveLength(1)
  })

  /**
   * The assistant's own voice reaches the microphone through the speakers. At
   * normal sensitivity that echo was enough to make it interrupt itself.
   */
  it('ignores echo-level audio while the assistant speaks', () => {
    const sustained: string[] = []
    const segmenter = new UtteranceSegmenter({ onSustainedSpeech: (id) => sustained.push(id) })
    segmenter.setGuarded(true)
    feed(segmenter, 0.01, 40) // above the open gate, below the guarded one
    expect(sustained).toHaveLength(0)

    // A person talking over it is still louder than the echo.
    feed(segmenter, 0.5, 20)
    expect(sustained).toHaveLength(1)
  })
})

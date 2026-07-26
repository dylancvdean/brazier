import { describe, expect, it } from 'vitest'

import {
  SPEECH_THRESHOLD,
  UtteranceSegmenter,
  encodeWav,
  padSpeechForAsr,
  padTrailingSilence
} from './utterance'

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

function feedNeural(
  segmenter: UtteranceSegmenter,
  amplitude: number,
  speechProbability: number,
  count: number
): void {
  for (let index = 0; index < count; index += 1) {
    segmenter.push(frame(amplitude), SAMPLE_RATE, speechProbability)
  }
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

  it('keeps a 100 ms spoken burst for short commands', () => {
    const utterances: Array<{ voicedFrames: number }> = []
    const segmenter = new UtteranceSegmenter({
      onUtterance: (value) => utterances.push(value)
    })
    feedNeural(segmenter, 0.2, 0.95, 5)
    feedNeural(segmenter, 0.001, 0.01, 40)
    expect(utterances).toHaveLength(1)
    expect(utterances[0].voicedFrames).toBe(5)
  })

  it('keeps the 200 ms floor when the neural model is unavailable', () => {
    const utterances: unknown[] = []
    const segmenter = new UtteranceSegmenter({
      onUtterance: (value) => utterances.push(value)
    })
    feed(segmenter, 0.2, 5)
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

describe('model-based speech decisions', () => {
  it('ignores loud background noise that Silero says is not speech', () => {
    const started: string[] = []
    const segmenter = new UtteranceSegmenter({ onSpeechStart: (id) => started.push(id) })

    feedNeural(segmenter, 0.5, 0.03, 100)
    expect(started).toHaveLength(0)
  })

  it('hears quiet speech without requiring it to clear the RMS gate', () => {
    const utterances: unknown[] = []
    const segmenter = new UtteranceSegmenter({ onUtterance: (value) => utterances.push(value) })

    feedNeural(segmenter, 0.001, 0.9, 25)
    feedNeural(segmenter, 0.001, 0.01, 40)
    expect(utterances).toHaveLength(1)
  })

  it('keeps the guarded energy check so speaker echo cannot barge in', () => {
    const sustained: string[] = []
    const segmenter = new UtteranceSegmenter({ onSustainedSpeech: (id) => sustained.push(id) })
    segmenter.setGuarded(true)

    feedNeural(segmenter, 0.01, 0.95, 40)
    expect(sustained).toHaveLength(0)

    feedNeural(segmenter, 0.5, 0.95, 20)
    expect(sustained).toHaveLength(1)
  })
})

describe('a room that is not quiet', () => {
  /**
   * The complaint this is for: a fan, a street, or an air conditioner sits above
   * a gate chosen for a quiet microphone, so the gate opens and never closes and
   * the session transcribes the room. The earlier attempt at this learned only
   * from frames below the gate, which is exactly the audio a fan never produces.
   */
  it('stops hearing steady noise as speech', () => {
    const utterances: unknown[] = []
    const discarded: string[] = []
    const segmenter = new UtteranceSegmenter({
      onUtterance: (value) => utterances.push(value),
      onDiscarded: (_id, reason) => discarded.push(reason)
    })

    // Ten seconds of fan, well above the fixed gate.
    feed(segmenter, 0.02, 500)
    expect(utterances).toHaveLength(0)
    expect(discarded.length).toBeGreaterThan(0)
    expect(segmenter.currentGate()).toBeGreaterThan(0.02)
  })

  it('still hears someone talking over it', () => {
    const utterances: unknown[] = []
    const segmenter = new UtteranceSegmenter({ onUtterance: (value) => utterances.push(value) })
    feed(segmenter, 0.02, 500) // learn the room
    feed(segmenter, 0.4, 25) // someone speaks over the fan
    feed(segmenter, 0.02, 40) // and stops
    expect(utterances).toHaveLength(1)
  })

  it('leaves a quiet room exactly as it was', () => {
    const segmenter = new UtteranceSegmenter({})
    feed(segmenter, 0.0004, 300)
    expect(segmenter.currentGate()).toBeCloseTo(SPEECH_THRESHOLD, 4)
  })

  it('does not raise the bar on a long answer to a long question', () => {
    const utterances: unknown[] = []
    const segmenter = new UtteranceSegmenter({ onUtterance: (value) => utterances.push(value) })
    // Fifteen seconds of speech with the ordinary gaps in it.
    for (let round = 0; round < 30; round += 1) {
      feed(segmenter, 0.3, 20)
      feed(segmenter, 0.0005, 5)
    }
    feed(segmenter, 0.0005, 40)
    expect(segmenter.currentGate()).toBeCloseTo(SPEECH_THRESHOLD, 4)
    expect(utterances).toHaveLength(1)
  })

  it('can be switched back to the fixed gate', () => {
    const segmenter = new UtteranceSegmenter({}, { adaptive: false })
    feed(segmenter, 0.02, 500)
    expect(segmenter.currentGate()).toBe(SPEECH_THRESHOLD)
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

describe('padSpeechForAsr', () => {
  it('moves a short word away from the WAV boundary and still flushes its tail', () => {
    const speech = new Float32Array([0.4, -0.4])
    const padded = padSpeechForAsr(speech, 1000, 250, 800)
    expect(padded).toHaveLength(1052)
    expect(padded[249]).toBe(0)
    expect(padded[250]).toBeCloseTo(0.4)
    expect(padded[251]).toBeCloseTo(-0.4)
    expect(padded[252]).toBe(0)
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

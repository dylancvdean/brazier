import { describe, expect, it } from 'vitest'

import { coversUtterance } from './speculativeTranscript'

const snapshot = { utteranceId: 'utt-1', voicedFrames: 25, sampleCount: 14400 }

describe('coversUtterance', () => {
  it('adopts a transcript of exactly this audio', () => {
    expect(coversUtterance(snapshot, { id: 'utt-1', voicedFrames: 25, sampleCount: 14400 })).toBe(
      true
    )
  })

  /** The case that matters: the speaker paused, then finished the sentence. */
  it('refuses one taken before the speaker was done', () => {
    expect(coversUtterance(snapshot, { id: 'utt-1', voicedFrames: 40, sampleCount: 21600 })).toBe(
      false
    )
  })

  it('refuses one from a different utterance', () => {
    expect(coversUtterance(snapshot, { id: 'utt-2', voicedFrames: 25, sampleCount: 14400 })).toBe(
      false
    )
  })

  /**
   * Same voiced count, more audio: silence was appended after the snapshot, so
   * the transcript describes a shorter recording than the one being submitted.
   */
  it('refuses one whose audio no longer matches sample for sample', () => {
    expect(coversUtterance(snapshot, { id: 'utt-1', voicedFrames: 25, sampleCount: 19200 })).toBe(
      false
    )
  })

  it('has nothing to adopt when no pause happened', () => {
    expect(coversUtterance(null, { id: 'utt-1', voicedFrames: 25, sampleCount: 14400 })).toBe(false)
  })
})

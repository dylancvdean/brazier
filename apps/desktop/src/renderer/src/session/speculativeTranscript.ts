/**
 * When a transcript made mid-utterance may be used as the final one.
 *
 * A turn cannot start until the silence gate closes, and then it used to wait
 * again while the audio was decoded — the wait people actually feel is both,
 * one after the other. Transcribing at a pause moves the second wait inside the
 * first, but only if the audio transcribed is the audio the utterance turns out
 * to contain. If someone paused and carried on, the speculative transcript
 * describes half a sentence, and submitting it would run the agent on a request
 * its author never finished making.
 *
 * The rule is deliberately exact rather than a heuristic about prefixes: same
 * utterance, same amount of speech in it, same number of samples. Anything else
 * is a partial, shown and thrown away.
 */

export type SpeculativeTranscript = {
  utteranceId: string
  /** Voiced frames counted when the snapshot was taken. */
  voicedFrames: number
  /** Samples in the snapshot, which the segmenter trims exactly as it does at close. */
  sampleCount: number
}

export type ClosedUtterance = {
  id: string
  voicedFrames: number
  sampleCount: number
}

/** Whether a speculative transcript describes this closed utterance exactly. */
export function coversUtterance(
  speculative: SpeculativeTranscript | null,
  utterance: ClosedUtterance
): boolean {
  if (!speculative) return false
  return (
    speculative.utteranceId === utterance.id &&
    speculative.voicedFrames === utterance.voicedFrames &&
    speculative.sampleCount === utterance.sampleCount
  )
}

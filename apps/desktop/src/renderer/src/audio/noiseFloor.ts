/**
 * What the room sounds like when nobody is talking.
 *
 * The speech gate was a fixed level chosen to suit a quiet microphone, which
 * means a fan, a street, or an air conditioner reads as speech continuously: the
 * gate opens, the utterance never falls silent, and the session transcribes
 * noise. A tracker was written for this before and backed out, because it only
 * learned from frames *below* the gate — so noise above it taught the tracker
 * nothing, and stayed heard as speech forever.
 *
 * The fix is to learn from every frame and decide by shape rather than by level.
 * Room noise is steady; speech is not. Over a second of audio, speech swings
 * between vowels and the gaps between words by tens of decibels, while a fan
 * holds its level. So the floor is allowed to climb only while the recent
 * history looks flat, and to fall at any time — because a level nobody is
 * sustaining is not the floor.
 *
 * The estimate is bounded at both ends: never below a level quiet rooms already
 * work at, and never so high that ordinary speech could not clear it. An energy
 * gate cannot separate quiet speech from equally loud noise, and pretending
 * otherwise would trade a session that hears noise for one that hears nothing.
 */

/** Frames of history the flatness test looks at. At 20 ms a frame, one second. */
const HISTORY_FRAMES = 50

/** Per-frame pull toward a level below the estimate. Fast: rooms go quiet. */
const FALL_RATE = 0.3

/** Per-frame pull toward a level above it. Slow: about a second to converge. */
const RISE_RATE = 0.02

/**
 * Loud-to-quiet ratio within the history above which audio counts as modulated.
 *
 * Speech alternates syllables and gaps, so its spread over a second is large.
 * Three is roughly 10 dB, which no steady source produces and no speech stays
 * under for long.
 */
const MODULATION_LIMIT = 3

/** Never estimate a room quieter than this; below it the gate is the floor. */
const MINIMUM = 0.0005

/**
 * Never estimate a room louder than this. Past it an energy gate is the wrong
 * instrument, and refusing to raise the bar further keeps a noisy room merely
 * unreliable rather than deaf.
 */
const MAXIMUM = 0.05

/** How far above the noise floor a frame has to be to count as speech. */
const GATE_FACTOR = 3

export class NoiseFloorTracker {
  private estimate = MINIMUM
  private readonly history: number[] = []

  /** The current estimate of the room. */
  get level(): number {
    return this.estimate
  }

  /**
   * Learn from one frame.
   *
   * Callers pass every frame, including ones loud enough to have opened an
   * utterance — that is the whole point, and the reason the earlier attempt
   * failed. What they must not pass is audio that is not the room: the
   * assistant's own voice coming back through the speakers is not what the
   * microphone hears when nobody is talking.
   */
  push(rms: number): void {
    this.history.push(rms)
    if (this.history.length > HISTORY_FRAMES) this.history.shift()
    if (rms < this.estimate) {
      this.estimate += (rms - this.estimate) * FALL_RATE
    } else if (this.isSteady()) {
      this.estimate += (rms - this.estimate) * RISE_RATE
    }
    this.estimate = Math.min(MAXIMUM, Math.max(MINIMUM, this.estimate))
  }

  /** The level a frame has to clear, given a fixed floor to stay above. */
  gate(floor: number): number {
    return Math.max(floor, this.estimate * GATE_FACTOR)
  }

  /** Start over, e.g. when the microphone changes. */
  reset(): void {
    this.estimate = MINIMUM
    this.history.length = 0
  }

  /**
   * Whether the last second looks like a sustained source rather than speech.
   *
   * Compared at the tenth and ninetieth percentiles rather than the extremes, so
   * one keyboard click does not make a steady room look like a conversation.
   */
  private isSteady(): boolean {
    if (this.history.length < HISTORY_FRAMES) return false
    const sorted = [...this.history].sort((a, b) => a - b)
    const low = Math.max(sorted[Math.floor(sorted.length * 0.1)], 1e-6)
    const high = sorted[Math.floor(sorted.length * 0.9)]
    return high / low < MODULATION_LIMIT
  }
}

import { describe, expect, it } from 'vitest'

import { NoiseFloorTracker } from './noiseFloor'

/** Feed `count` frames at `level`, with a little jitter so nothing is exactly flat. */
function feed(tracker: NoiseFloorTracker, level: number, count: number): void {
  for (let index = 0; index < count; index += 1) {
    tracker.push(level * (0.9 + 0.2 * ((index % 7) / 7)))
  }
}

/** Speech: loud stretches with the gaps between words in them. */
function speak(tracker: NoiseFloorTracker, level: number, seconds: number): void {
  const frames = Math.round(seconds * 50)
  for (let index = 0; index < frames; index += 1) {
    // ~250 ms of voice, ~100 ms of gap: syllables and breaths.
    tracker.push(index % 18 < 13 ? level : level / 50)
  }
}

describe('NoiseFloorTracker', () => {
  it('stays at the bottom in a quiet room', () => {
    const tracker = new NoiseFloorTracker()
    feed(tracker, 0.0004, 300)
    expect(tracker.gate(0.006)).toBeCloseTo(0.006, 4)
  })

  /**
   * The failure the backed-out version had: noise above the gate taught it
   * nothing, so the fan was heard as speech for the whole session.
   */
  it('learns a fan that is louder than the fixed gate', () => {
    const tracker = new NoiseFloorTracker()
    feed(tracker, 0.02, 300)
    expect(tracker.level).toBeGreaterThan(0.01)
    expect(tracker.gate(0.006)).toBeGreaterThan(0.02)
  })

  it('does not mistake speech for the room', () => {
    const tracker = new NoiseFloorTracker()
    speak(tracker, 0.3, 6)
    // Someone talking for six seconds must not raise the bar on themselves.
    expect(tracker.gate(0.006)).toBeCloseTo(0.006, 4)
  })

  it('keeps hearing speech over a fan it has learned', () => {
    const tracker = new NoiseFloorTracker()
    feed(tracker, 0.02, 300)
    const gate = tracker.gate(0.006)
    expect(0.3).toBeGreaterThan(gate)
  })

  it('comes back down when the room goes quiet', () => {
    const tracker = new NoiseFloorTracker()
    feed(tracker, 0.02, 300)
    const noisy = tracker.level
    feed(tracker, 0.0004, 100)
    expect(tracker.level).toBeLessThan(noisy / 5)
  })

  /**
   * An energy gate cannot separate quiet speech from equally loud noise. The
   * bound keeps a very noisy room unreliable rather than deaf, which is the
   * failure people can at least understand.
   */
  it('refuses to raise the bar past where speech lives', () => {
    const tracker = new NoiseFloorTracker()
    feed(tracker, 0.5, 1000)
    expect(tracker.gate(0.006)).toBeCloseTo(0.15, 4)
  })
})

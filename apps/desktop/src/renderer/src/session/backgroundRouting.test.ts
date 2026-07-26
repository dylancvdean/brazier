import { describe, expect, it } from 'vitest'

import { shouldRouteVoiceToBackground } from './backgroundRouting'

describe('voice background routing', () => {
  it('preserves the original behavior in always mode', () => {
    expect(shouldRouteVoiceToBackground('Thanks', 'always')).toBe(true)
    expect(shouldRouteVoiceToBackground('Hi', 'always')).toBe(true)
  })

  it('keeps lightweight conversation with PersonaPlex in auto mode', () => {
    expect(shouldRouteVoiceToBackground('How are you?', 'auto')).toBe(false)
    expect(shouldRouteVoiceToBackground("What's two plus two?", 'auto')).toBe(false)
    expect(shouldRouteVoiceToBackground('Sick!', 'auto')).toBe(false)
  })

  it('routes stateful work even when the request is short', () => {
    expect(shouldRouteVoiceToBackground('Run the tests', 'auto')).toBe(true)
    expect(shouldRouteVoiceToBackground('Check the current branch', 'auto')).toBe(true)
    expect(shouldRouteVoiceToBackground('What time is it?', 'auto')).toBe(true)
  })

  it('routes nontrivial requests and active-task follow-ups in auto mode', () => {
    expect(
      shouldRouteVoiceToBackground(
        'Could you explain why that approach would be safer than the alternative?',
        'auto'
      )
    ).toBe(true)
    expect(shouldRouteVoiceToBackground('Try the other one', 'auto', { taskActive: true })).toBe(
      true
    )
    expect(shouldRouteVoiceToBackground('Thanks', 'auto', { taskActive: true })).toBe(false)
  })

  it('requires a concrete background cue in explicit mode', () => {
    expect(
      shouldRouteVoiceToBackground(
        'Could you explain why that approach would be safer than the alternative?',
        'explicit'
      )
    ).toBe(false)
    expect(shouldRouteVoiceToBackground('Look up the latest release', 'explicit')).toBe(true)
  })
})

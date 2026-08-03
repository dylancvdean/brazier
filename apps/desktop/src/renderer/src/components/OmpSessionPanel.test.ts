import { describe, expect, it } from 'vitest'

import { contextPercent } from './OmpSessionPanel'

describe('contextPercent', () => {
  it('prefers the reported percentage when present', () => {
    expect(
      contextPercent({
        contextUsage: { tokens: 1000, contextWindow: 2000, percent: 40 }
      })
    ).toBe(40)
  })

  it('derives the percentage from tokens and window when percent is absent', () => {
    expect(contextPercent({ contextUsage: { tokens: 500, contextWindow: 2000 } })).toBe(25)
  })

  it('returns null without a usable measurement', () => {
    expect(contextPercent(null)).toBeNull()
    expect(contextPercent({ contextUsage: {} })).toBeNull()
    expect(contextPercent({ contextUsage: { tokens: 100 } })).toBeNull()
  })
})

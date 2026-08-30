import { describe, expect, it } from 'vitest'

import { normalizeRuntimeId } from './registry'

describe('normalizeRuntimeId', () => {
  it('keeps live catalog ids and falls back for unknown ones', () => {
    expect(normalizeRuntimeId('simple')).toBe('simple')
    expect(normalizeRuntimeId('powerful')).toBe('powerful')
    expect(normalizeRuntimeId('')).toBe('simple')
    expect(normalizeRuntimeId(null)).toBe('simple')
    expect(normalizeRuntimeId('omp')).toBe('simple')
  })
})

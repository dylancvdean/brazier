import { describe, expect, it } from 'vitest'

import { normalizeRuntimeId } from './registry'

describe('normalizeRuntimeId', () => {
  it('maps retired aliases onto Simple', () => {
    expect(normalizeRuntimeId('pi')).toBe('simple')
    expect(normalizeRuntimeId('balanced')).toBe('simple')
    expect(normalizeRuntimeId('')).toBe('simple')
    expect(normalizeRuntimeId(null)).toBe('simple')
  })

  it('keeps live catalog ids and falls back for unknown ones', () => {
    expect(normalizeRuntimeId('simple')).toBe('simple')
    expect(normalizeRuntimeId('powerful')).toBe('powerful')
    expect(normalizeRuntimeId('omp')).toBe('simple')
  })
})

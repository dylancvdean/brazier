import { describe, expect, it } from 'vitest'

import { parseNvidiaSmiMemoryGib } from './qualificationHost'

describe('parseNvidiaSmiMemoryGib', () => {
  it('uses the largest physical GPU without aggregating devices', () => {
    expect(parseNvidiaSmiMemoryGib('8192\n24576\n')).toBe(24)
    expect(parseNvidiaSmiMemoryGib('12288 MiB\r\n')).toBe(12)
  })

  it('rejects empty, malformed, and non-positive output', () => {
    expect(parseNvidiaSmiMemoryGib('')).toBeNull()
    expect(parseNvidiaSmiMemoryGib('N/A\n0\n-1')).toBeNull()
  })
})

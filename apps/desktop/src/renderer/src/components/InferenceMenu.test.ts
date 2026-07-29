import { describe, expect, it } from 'vitest'
import { contextPresets } from './InferenceMenu'

describe('contextPresets', () => {
  it('uses an arbitrary reported model context as the exact maximum', () => {
    expect(contextPresets(98_304)).toEqual([2048, 4096, 8192, 16_384, 32_768, 65_536, 98_304])
  })

  it('does not include presets above the reported model context', () => {
    expect(contextPresets(12_288)).toEqual([2048, 4096, 8192, 12_288])
  })
})

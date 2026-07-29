import { describe, expect, it } from 'vitest'
import { sourceRuntimeId } from './ManagePanel'

describe('sourceRuntimeId', () => {
  it('maps a llama.cpp fork build to its runtime inventory id', () => {
    expect(sourceRuntimeId('llama.cpp', 'main-123')).toBe('source-main-123')
  })

  it('maps Python runtime builds to their engine-qualified inventory ids', () => {
    expect(sourceRuntimeId('mlx-lm', 'main-123')).toBe('mlx-lm-source-main-123')
  })
})

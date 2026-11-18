import { describe, expect, it } from 'vitest'
import {
  SAMPLING_PRESETS,
  contextPresets,
  matchingSamplingPreset,
  summarizeResidency,
  unifiedSaveLabel
} from './InferenceMenu'

describe('contextPresets', () => {
  it('uses an arbitrary reported model context as the exact maximum', () => {
    expect(contextPresets(98_304)).toEqual([2048, 4096, 8192, 16_384, 32_768, 65_536, 98_304])
  })

  it('does not include presets above the reported model context', () => {
    expect(contextPresets(12_288)).toEqual([2048, 4096, 8192, 12_288])
  })
})

describe('sampling presets', () => {
  it('exposes the four canonical sampling presets and detects active preset matches', () => {
    expect(SAMPLING_PRESETS.map((preset) => preset.label)).toEqual([
      'Coding · 0.15',
      'Balanced · 0.7',
      'Creative · 1.0',
      'Greedy · 0.0'
    ])
    expect(matchingSamplingPreset(0.15, 0.9)).toBe('coding')
    expect(matchingSamplingPreset(0.7, 0.95)).toBe('balanced')
    expect(matchingSamplingPreset(1.0, 0.98)).toBe('creative')
    expect(matchingSamplingPreset(0.0, 1.0)).toBe('greedy')
    expect(matchingSamplingPreset(0.42, 0.8)).toBeNull()
  })
})

describe('summarizeResidency', () => {
  const GiB = 1024 * 1024 * 1024

  it('formats GPU weights + KV / capacity, context, and parallel slots', () => {
    const summary = summarizeResidency({
      placement: 'gpu',
      backend: 'metal',
      description: 'Resident on Metal GPU',
      gpu_bytes: 5 * GiB,
      gpu_kv_bytes: 1 * GiB,
      cpu_bytes: 0,
      cpu_kv_bytes: 0,
      gpu_capacity_bytes: 16 * GiB,
      aggregate_context_tokens: 32_768,
      parallel_slots: 3
    })

    expect(summary.headline).toBe('GPU 6.0 GB / 16 GB · 32K ctx · 3 slots')
    expect(summary.detail).toBe('weights 5.0 GB + KV 1.0 GB')
    expect(summary.weightsPercent).toBeCloseTo(31.25)
    expect(summary.kvPercent).toBeCloseTo(6.25)
  })

  it('includes CPU offload bytes and falls back cleanly when byte metrics are absent', () => {
    const offloaded = summarizeResidency({
      placement: 'cpu_offload',
      backend: 'vulkan',
      description: 'Split across GPU and CPU',
      gpu_bytes: 4 * GiB,
      gpu_kv_bytes: 0.5 * GiB,
      cpu_bytes: 2 * GiB,
      cpu_kv_bytes: 0,
      gpu_capacity_bytes: 8 * GiB,
      aggregate_context_tokens: 8192,
      parallel_slots: 1
    })
    expect(offloaded.headline).toBe('GPU 4.5 GB / 8.0 GB · CPU 2.0 GB · 8K ctx · 1 slot')

    const fallback = summarizeResidency({
      placement: 'gpu',
      backend: 'mlx',
      description: 'Resident in Apple unified memory',
      gpu_bytes: null,
      gpu_kv_bytes: null,
      cpu_bytes: null,
      cpu_kv_bytes: null,
      gpu_capacity_bytes: null,
      aggregate_context_tokens: 16_384,
      parallel_slots: null
    })
    expect(fallback.headline).toBe('Resident in Apple unified memory')
    expect(fallback.detail).toBe('16K ctx')
    expect(fallback.weightsPercent).toBeNull()
  })
})

describe('unifiedSaveLabel', () => {
  it('describes whichever combination of global defaults and model overrides is dirty', () => {
    expect(unifiedSaveLabel({ dirty: false, profileDirty: false, saving: false })).toBe(
      'Up to date'
    )
    expect(unifiedSaveLabel({ dirty: true, profileDirty: false, saving: false })).toBe('Apply')
    expect(unifiedSaveLabel({ dirty: false, profileDirty: true, saving: false })).toBe(
      'Save model overrides'
    )
    expect(unifiedSaveLabel({ dirty: true, profileDirty: true, saving: false })).toBe(
      'Save defaults & model overrides'
    )
    expect(unifiedSaveLabel({ dirty: true, profileDirty: true, saving: true })).toBe('Saving…')
  })
})

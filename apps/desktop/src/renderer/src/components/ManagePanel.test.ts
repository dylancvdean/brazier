import { describe, expect, it } from 'vitest'
import type { HardwareInfo, HubFile } from '../api'
import {
  appendBuildDiagnostics,
  groupQuants,
  quantGroupName,
  sortQuantGroups,
  sourceRuntimeId,
  targetSupportedByBuildEngine
} from './ManagePanel'

function file(path: string, size: number | null = 1000): HubFile {
  return { path, size }
}

describe('sourceRuntimeId', () => {
  it('maps a llama.cpp fork build to its runtime inventory id', () => {
    expect(sourceRuntimeId('llama.cpp', 'main-123')).toBe('source-main-123')
  })

  it('maps Python runtime builds to their engine-qualified inventory ids', () => {
    expect(sourceRuntimeId('mlx-lm', 'main-123')).toBe('mlx-lm-source-main-123')
  })
})

describe('appendBuildDiagnostics', () => {
  it('replaces an overlapping streamed tail instead of duplicating it', () => {
    expect(
      appendBuildDiagnostics(['older output', 'line one', 'line three', 'failed'], {
        log_excerpt: 'command\nline one\nline two\nline three\nfailed',
        hints: ['Check the compiler output.']
      })
    ).toEqual([
      'older output',
      'command',
      'line one',
      'line two',
      'line three',
      'failed',
      '',
      'Suggested fixes:',
      '• Check the compiler output.'
    ])
  })

  it('labels an excerpt when no live output overlapped it', () => {
    expect(appendBuildDiagnostics([], { log_excerpt: 'only diagnostic' })).toEqual([
      '',
      '--- last log lines ---',
      'only diagnostic'
    ])
  })
})

describe('targetSupportedByBuildEngine', () => {
  it('disables Vulkan only for vLLM builds', () => {
    expect(targetSupportedByBuildEngine('vllm', 'vulkan')).toBe(false)
    expect(targetSupportedByBuildEngine('vllm', 'rocm')).toBe(true)
    expect(targetSupportedByBuildEngine('vllm', 'cpu')).toBe(true)
    expect(targetSupportedByBuildEngine('llama.cpp', 'vulkan')).toBe(true)
  })
})

describe('groupQuants', () => {
  it('groups the shards of one quant into a single downloadable', () => {
    const groups = groupQuants([
      file('UD-Q4_K_M/Inkling-Small-UD-Q4_K_M-00002-of-00003.gguf', 3000),
      file('UD-Q4_K_M/Inkling-Small-UD-Q4_K_M-00001-of-00003.gguf', 2000),
      file('UD-Q4_K_M/Inkling-Small-UD-Q4_K_M-00003-of-00003.gguf', 1000),
      file('UD-IQ1_S/Inkling-Small-UD-IQ1_S-00001-of-00002.gguf', 500),
      file('UD-IQ1_S/Inkling-Small-UD-IQ1_S-00002-of-00002.gguf', 400)
    ])

    expect(groups).toHaveLength(2)
    const q4 = groups.find((group) => group.key === 'UD-Q4_K_M/Inkling-Small-UD-Q4_K_M.gguf')
    // The size of the quant is the sum of its shards, not one file.
    expect(q4?.size).toBe(6000)
    // Shards stay in order so the first shard is queued first.
    expect(q4?.files.map((part) => part.path)).toEqual([
      'UD-Q4_K_M/Inkling-Small-UD-Q4_K_M-00001-of-00003.gguf',
      'UD-Q4_K_M/Inkling-Small-UD-Q4_K_M-00002-of-00003.gguf',
      'UD-Q4_K_M/Inkling-Small-UD-Q4_K_M-00003-of-00003.gguf'
    ])
    expect(
      groups.find((group) => group.key === 'UD-IQ1_S/Inkling-Small-UD-IQ1_S.gguf')?.size
    ).toBe(900)
  })

  it('keeps single-file quants and companions as their own groups', () => {
    const groups = groupQuants([
      file('model-Q4_K_M.gguf', 4000),
      file('mmproj-model-f16.gguf', 800)
    ])

    expect(groups).toHaveLength(2)
    expect(groups.find((group) => group.key === 'model-Q4_K_M.gguf')?.files).toHaveLength(1)
    expect(groups.find((group) => group.key === 'mmproj-model-f16.gguf')?.size).toBe(800)
  })

  it('reports an unknown size when any shard size is unknown', () => {
    const groups = groupQuants([
      file('model-Q8_0-00001-of-00002.gguf', 1000),
      file('model-Q8_0-00002-of-00002.gguf', null)
    ])

    expect(groups).toHaveLength(1)
    expect(groups[0].size).toBeNull()
  })
})

describe('quantGroupName', () => {
  it('names a sharded group like the single file it stands in for', () => {
    const [group] = groupQuants([
      file('UD-Q4_K_M/Inkling-Small-UD-Q4_K_M-00001-of-00003.gguf', 1)
    ])
    expect(quantGroupName(group)).toBe('Inkling-Small-UD-Q4_K_M.gguf')
  })

  it('keeps an unsharded file name as is', () => {
    const [group] = groupQuants([file('model-Q4_K_M.gguf', 1)])
    expect(quantGroupName(group)).toBe('model-Q4_K_M.gguf')
  })

  it('does not invent a .gguf name for whisper .bin files', () => {
    const [group] = groupQuants([file('ggml-large-v3.bin', 1)])
    expect(quantGroupName(group)).toBe('ggml-large-v3.bin')
  })
})

describe('sortQuantGroups', () => {
  const hardware = {
    os: 'linux',
    architecture: 'x86_64',
    logical_cpus: 8,
    memory_bytes: 10_000,
    vram_bytes: null,
    gpu_offload_memory_bytes: null,
    usable_model_memory_bytes: null,
    gpu: null,
    gpu_arch: null,
    amd_apu: false,
    recommended_target: 'cpu',
    targets: []
  } as unknown as HardwareInfo

  it('ranks groups by fit using the summed size, largest first within a fit', () => {
    const sorted = sortQuantGroups(
      groupQuants([
        // 6000 total: fits the 6000-byte (60% of 10000) system budget.
        file('big-Q8_0-00001-of-00002.gguf', 4000),
        file('big-Q8_0-00002-of-00002.gguf', 2000),
        file('small-Q4_K_M.gguf', 1000),
        // 9000 in one file: no shard grouping hides its real size.
        file('huge-F16.gguf', 9000)
      ]),
      hardware
    )

    expect(sorted.map((group) => group.key)).toEqual([
      'big-Q8_0.gguf',
      'small-Q4_K_M.gguf',
      'huge-F16.gguf'
    ])
  })
})

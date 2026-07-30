import { describe, expect, it } from 'vitest'
import {
  appendBuildDiagnostics,
  sourceRuntimeId,
  targetSupportedByBuildEngine
} from './ManagePanel'

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

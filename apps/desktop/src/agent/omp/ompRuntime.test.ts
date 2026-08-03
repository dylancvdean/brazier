import { describe, expect, it } from 'vitest'

import {
  assistantContentFromOmpMessage,
  buildBrazierModelCatalog,
  configYamlWithModelRoles,
  contextSeedAfterPermissionModeAttempt,
  hostToolResultFrame,
  isCurrentOmpRun,
  ompBrazierModelsConfig,
  ompApprovalModeArg,
  promptWithBrazierHistory,
  stripTopLevelKey
} from './ompRuntime'

describe('OMP adapter protocol helpers', () => {
  it('keeps user echoes and tool results out of the assistant response stream', () => {
    expect(
      assistantContentFromOmpMessage({
        role: 'user',
        content: [{ type: 'text', text: 'Repeat-sensitive prompt' }]
      })
    ).toBeNull()
    expect(
      assistantContentFromOmpMessage({
        role: 'toolResult',
        content: [{ type: 'text', text: '#include <whole-header.h>' }]
      })
    ).toBeNull()
    expect(
      assistantContentFromOmpMessage({
        role: 'assistant',
        content: [{ type: 'text', text: 'The header has one issue.' }]
      })
    ).toEqual({ text: 'The header has one issue.', toolCalls: [] })
  })

  it('uses OMP public approval-mode CLI syntax', () => {
    expect(ompApprovalModeArg('always-ask')).toBe('--approval-mode=always-ask')
    expect(ompApprovalModeArg('yolo')).toBe('--approval-mode=yolo')
  })

  it('uses OMP structured host-tool results and marks errors at the top level', () => {
    expect(hostToolResultFrame('host_1', 'done')).toEqual({
      type: 'host_tool_result',
      id: 'host_1',
      result: { content: [{ type: 'text', text: 'done' }] }
    })
    expect(hostToolResultFrame('host_2', 'denied', true)).toEqual({
      type: 'host_tool_result',
      id: 'host_2',
      result: { content: [{ type: 'text', text: 'denied' }] },
      isError: true
    })
  })

  it('registers the daemon as an OMP provider without rewriting its model ids', () => {
    expect(
      ompBrazierModelsConfig(
        'http://127.0.0.1:7777/v1',
        [
          {
            id: 'gguf-ext:0:owner/model.gguf',
            name: 'Local model',
            contextWindow: 32768,
            reasoning: true,
            supportsTools: true,
            vision: false
          }
        ]
      )
    ).toMatchObject({
      providers: {
        brazier: {
          baseUrl: 'http://127.0.0.1:7777/v1',
          apiKey: 'BRAZIER_OPENAI_API_KEY',
          authHeader: true,
          api: 'openai-completions',
          discovery: { type: 'openai-models-list' },
          models: [
            {
              id: 'gguf-ext:0:owner/model.gguf',
              input: ['text'],
              contextWindow: 32768,
              reasoning: true,
              supportsTools: true
            }
          ]
        }
      }
    })
  })

  it('advertises every model with per-model vision and tool hints', () => {
    const config = ompBrazierModelsConfig('http://127.0.0.1:7777/v1', [
      { id: 'gguf:a/model.gguf', name: 'A', reasoning: false, supportsTools: true, vision: false },
      { id: 'mlx:b/vision', name: 'B', reasoning: true, supportsTools: true, vision: true }
    ])
    const models = (config.providers as { brazier: { models: Array<Record<string, unknown>> } }).brazier.models
    expect(models).toHaveLength(2)
    expect(models[0]).toMatchObject({ id: 'gguf:a/model.gguf', input: ['text'] })
    expect(models[1]).toMatchObject({ id: 'mlx:b/vision', input: ['text', 'image'], reasoning: true })
    // modelRoles belongs in config.yml, not the models file.
    expect(config.modelRoles).toBeUndefined()
  })

  it('builds a catalog from the daemon list and always keeps the selected model', () => {
    const catalog = buildBrazierModelCatalog(
      [
        {
          id: 'gguf:a/model.gguf',
          capabilities: { input_modalities: ['text'], tools: true, reasoning: true, max_context_length: 32768 }
        },
        {
          id: 'mlx:b/vision',
          capabilities: { input_modalities: ['text', 'image'], tools: true, reasoning: false, max_context_length: 8192 }
        }
      ],
      { id: 'gguf:selected/model.gguf', name: 'Selected', contextWindow: 16384 },
      { nativeToolCalling: true, parallelToolCalling: true, supportsReasoningStream: true, harmony: true, reliableJson: true }
    )
    expect(catalog).toEqual([
      { id: 'gguf:selected/model.gguf', name: 'Selected', contextWindow: 16384, reasoning: true, supportsTools: true, vision: false },
      { id: 'gguf:a/model.gguf', name: 'gguf:a/model.gguf', contextWindow: 32768, reasoning: true, supportsTools: true, vision: false },
      { id: 'mlx:b/vision', name: 'mlx:b/vision', contextWindow: 8192, reasoning: false, supportsTools: true, vision: true }
    ])
  })

  it('merges model roles into config YAML and replaces any prior block', () => {
    const base = 'lsp:\n  enabled: true\nmodelRoles:\n  smol: spark/x\ncompaction:\n  enabled: true'
    const merged = configYamlWithModelRoles(base, { smol: 'gguf:smol/model.gguf', plan: 'gguf:big/model.gguf' })
    expect(merged).toContain('lsp:\n  enabled: true')
    expect(merged).toContain('compaction:\n  enabled: true')
    expect(merged).toContain('modelRoles:\n  smol: brazier/gguf:smol/model.gguf\n  plan: brazier/gguf:big/model.gguf')
    expect(merged).not.toContain('spark/x')
  })

  it('returns the base YAML untouched when there are no roles', () => {
    expect(configYamlWithModelRoles('lsp:\n  enabled: true', {})).toBe('lsp:\n  enabled: true')
    expect(configYamlWithModelRoles(undefined, { smol: '  ' })).toBe('')
  })

  it('strips a top-level key block while keeping surrounding keys', () => {
    const yaml = 'a: 1\nmodelRoles:\n  smol: x\n  plan: y\nb: 2'
    expect(stripTopLevelKey(yaml, 'modelRoles')).toBe('a: 1\nb: 2')
  })

  it('seeds prior transcript without overriding OMP system instructions', () => {
    expect(
      promptWithBrazierHistory(
        [
          { role: 'user', text: 'Inspect the release.', timestamp: '2026-01-01T00:00:00Z' },
          { role: 'tool', tool: 'mcp_ci', toolCallId: '1', output: 'green', isError: false, timestamp: '2026-01-01T00:00:01Z' }
        ],
        'What should I do next?'
      )
    ).toContain(
      '## Prior Brazier transcript\n[user]\nInspect the release.\n\n[tool mcp_ci]\ngreen\n\n## Current user request\nWhat should I do next?'
    )
  })

  it('keeps the old sidecar context state when permission persistence rolls back', () => {
    expect(contextSeedAfterPermissionModeAttempt(false, false)).toBe(false)
    expect(contextSeedAfterPermissionModeAttempt(true, false)).toBe(true)
    expect(contextSeedAfterPermissionModeAttempt(false, true)).toBe(true)
  })

  it('never lets a late approval callback attach to a different run', () => {
    expect(isCurrentOmpRun('run-a', 'run-a', false)).toBe(true)
    expect(isCurrentOmpRun(undefined, 'run-a', false)).toBe(false)
    expect(isCurrentOmpRun('run-b', 'run-a', false)).toBe(false)
    expect(isCurrentOmpRun('run-a', 'run-a', true)).toBe(false)
  })
})

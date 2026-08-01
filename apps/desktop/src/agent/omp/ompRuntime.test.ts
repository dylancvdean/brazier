import { describe, expect, it } from 'vitest'

import {
  contextSeedAfterPermissionModeAttempt,
  hostToolResultFrame,
  isCurrentOmpRun,
  promptWithBrazierContext
} from './ompRuntime'

describe('OMP adapter protocol helpers', () => {
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

  it('seeds a fresh sidecar with the Brazier instructions and prior transcript', () => {
    expect(
      promptWithBrazierContext(
        'Never deploy without confirmation.',
        [
          { role: 'user', text: 'Inspect the release.', timestamp: '2026-01-01T00:00:00Z' },
          { role: 'tool', tool: 'mcp_ci', toolCallId: '1', output: 'green', isError: false, timestamp: '2026-01-01T00:00:01Z' }
        ],
        'What should I do next?'
      )
    ).toContain(
      '## Brazier system instructions\nNever deploy without confirmation.\n\n## Prior Brazier transcript\n[user]\nInspect the release.\n\n[tool mcp_ci]\ngreen\n\n## Current user request\nWhat should I do next?'
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

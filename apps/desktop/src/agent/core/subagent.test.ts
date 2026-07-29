import { describe, expect, it } from 'vitest'

import {
  buildSubagentMetadata,
  childEnabledTools,
  collectSpawnPrompts,
  DEFAULT_MAX_SUBAGENTS,
  isSubagentSession,
  resolveMaxSubagents,
  resolveSubagentContext,
  resolveSubagentModel,
  SPAWN_SUBAGENT_TOOL,
  summarizeSubagentResult
} from './subagent'

describe('subagent helpers', () => {
  it('defaults max subagents to 2 and clamps the profile value', () => {
    expect(resolveMaxSubagents(null)).toBe(DEFAULT_MAX_SUBAGENTS)
    expect(resolveMaxSubagents({})).toBe(2)
    expect(resolveMaxSubagents({ max_subagents: 4 })).toBe(4)
    expect(resolveMaxSubagents({ max_subagents: 0 })).toBe(1)
    expect(resolveMaxSubagents({ max_subagents: 99 })).toBe(8)
  })

  it('resolves model as tool arg → profile → parent', () => {
    expect(resolveSubagentModel('gguf:child.gguf', { subagent_model: 'gguf:profile.gguf' }, 'gguf:parent.gguf')).toBe(
      'gguf:child.gguf'
    )
    expect(resolveSubagentModel('  ', { subagent_model: 'gguf:profile.gguf' }, 'gguf:parent.gguf')).toBe(
      'gguf:profile.gguf'
    )
    expect(resolveSubagentModel(undefined, null, 'gguf:parent.gguf')).toBe('gguf:parent.gguf')
  })

  it('defaults subagent context to the parent context', () => {
    expect(resolveSubagentContext(null, 32_768)).toBe(32_768)
    expect(resolveSubagentContext({ subagent_context_size: 16_384 }, 32_768)).toBe(16_384)
  })

  it('strips spawn_subagent from child tools so nesting is impossible', () => {
    expect(
      childEnabledTools(['fs_read', SPAWN_SUBAGENT_TOOL, 'shell_run'])
    ).toEqual(['fs_read', 'shell_run'])
  })

  it('detects subagent sessions from runtime metadata', () => {
    expect(isSubagentSession({ kind: 'subagent', parent_session_id: 'p1' })).toBe(true)
    expect(isSubagentSession({ worktree: { source_path: '/a', path: '/b', branch: 't' } })).toBe(
      false
    )
  })

  it('copies worktree metadata onto the child', () => {
    const meta = buildSubagentMetadata('parent-1', {
      worktree: { source_path: '/src', path: '/wt', branch: 'agent/task' }
    })
    expect(meta).toEqual({
      kind: 'subagent',
      parent_session_id: 'parent-1',
      worktree: { source_path: '/src', path: '/wt', branch: 'agent/task' }
    })
  })

  it('collects a single prompt or a concurrent prompts list', () => {
    expect(collectSpawnPrompts({ prompt: ' one ' })).toEqual(['one'])
    expect(collectSpawnPrompts({ prompts: ['a', '  ', 'b'] })).toEqual(['a', 'b'])
    expect(collectSpawnPrompts({ prompt: 'ignored', prompts: ['x', 'y'] })).toEqual(['x', 'y'])
    expect(collectSpawnPrompts({})).toEqual([])
  })

  it('summarizes the last assistant reply', () => {
    expect(
      summarizeSubagentResult([
        { role: 'user', text: 'do it' },
        { role: 'assistant', text: '  All done.  ' }
      ])
    ).toBe('All done.')
    expect(summarizeSubagentResult([], { failed: true, error: 'boom' })).toBe(
      'Subagent failed: boom'
    )
  })
})

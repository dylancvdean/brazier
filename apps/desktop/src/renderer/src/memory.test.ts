/**
 * Memory helpers: recall rendering, client tool dispatch, and dreaming
 * proposal parsing. The API calls are stubbed; the pure logic under test is
 * the context budget, the proposal extraction, and the safety filters.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { Memory } from './types'

const daemon = vi.hoisted(() => ({
  created: [] as Array<Record<string, unknown>>,
  updated: [] as Array<{ id: string; patch: Record<string, unknown> }>,
  deleted: [] as string[],
  memories: [] as Memory[]
}))

vi.mock('./api', () => ({
  createMemory: vi.fn(async (input: Record<string, unknown>) => {
    daemon.created.push(input)
    return { id: `m-${daemon.created.length}`, text: String(input.text), kind: 'fact', pinned: false, tags: [], source_conversation_id: null, source_message_id: null, created_at: '', updated_at: '' }
  }),
  updateMemory: vi.fn(async (id: string, patch: Record<string, unknown>) => {
    daemon.updated.push({ id, patch })
    return { id, text: String(patch.text), kind: 'fact', pinned: false, tags: [], created_at: '', updated_at: '' }
  }),
  deleteMemory: vi.fn(async (id: string) => {
    daemon.deleted.push(id)
    return { deleted: true }
  }),
  listMemories: vi.fn(async () => daemon.memories),
  streamCompletion: vi.fn()
}))

import {
  buildMemoryContext,
  executeMemoryClientTool,
  extractDreamProposal,
  isMemoryClientTool,
  memoryToolDefinitions,
  MEMORY_RECALL_TOOL,
  MEMORY_SAVE_TOOL,
  normalizeDreamProposal
} from './memory'

function memory(id: string, text: string, pinned = false): Memory {
  return {
    id,
    text,
    kind: 'fact',
    pinned,
    tags: [],
    created_at: '',
    updated_at: ''
  }
}

describe('memoryToolDefinitions', () => {
  it('offers only the save and recall tools', () => {
    const names = memoryToolDefinitions().map((definition) => definition.function.name)
    expect(names).toEqual([MEMORY_SAVE_TOOL, MEMORY_RECALL_TOOL])
  })

  it('isMemoryClientTool accepts exactly those two', () => {
    expect(isMemoryClientTool(MEMORY_SAVE_TOOL)).toBe(true)
    expect(isMemoryClientTool(MEMORY_RECALL_TOOL)).toBe(true)
    expect(isMemoryClientTool('fetch_url')).toBe(false)
  })
})

describe('buildMemoryContext', () => {
  it('returns null for an empty store', () => {
    expect(buildMemoryContext([], 1000)).toBeNull()
  })

  it('stops when the character budget is exhausted', () => {
    const memories = [
      memory('1', 'a'.repeat(100)),
      memory('2', 'b'.repeat(100)),
      memory('3', 'c'.repeat(100))
    ]
    const context = buildMemoryContext(memories, 240)
    expect(context).not.toBeNull()
    expect(context).toContain('aaa')
    expect(context).toContain('bbb')
    expect(context).not.toContain('ccc')
  })
})

describe('executeMemoryClientTool', () => {
  beforeEach(() => {
    daemon.created = []
    daemon.updated = []
    daemon.deleted = []
    daemon.memories = [memory('1', 'User likes dark mode.')]
  })

  it('saves a memory and records its source', async () => {
    const outcome = await executeMemoryClientTool(
      { id: 'call-1', name: MEMORY_SAVE_TOOL, arguments: JSON.stringify({ memory: 'User is left-handed.' }) },
      { conversation_id: 'conv-1', message_id: 'msg-1' }
    )
    expect(outcome.is_error).toBe(false)
    expect(daemon.created[0]).toMatchObject({
      text: 'User is left-handed.',
      source_conversation_id: 'conv-1',
      source_message_id: 'msg-1'
    })
  })

  it('rejects an empty memory body', async () => {
    const outcome = await executeMemoryClientTool(
      { id: 'call-1', name: MEMORY_SAVE_TOOL, arguments: '{}' }
    )
    expect(outcome.is_error).toBe(true)
    expect(daemon.created).toHaveLength(0)
  })

  it('recalls matching memories', async () => {
    const outcome = await executeMemoryClientTool(
      { id: 'call-1', name: MEMORY_RECALL_TOOL, arguments: JSON.stringify({ query: 'dark' }) }
    )
    expect(outcome.is_error).toBe(false)
    expect(outcome.output).toContain('dark mode')
  })
})

describe('dream proposal parsing', () => {
  it('extracts a JSON object from prose and markdown fences', () => {
    const raw = 'Here you go:\n```json\n{"new_memories":[{"text":"User prefers tea."}]}\n```\nDone.'
    expect(extractDreamProposal(raw)).toEqual({ new_memories: [{ text: 'User prefers tea.' }] })
  })

  it('returns null when no JSON is present', () => {
    expect(extractDreamProposal('no json here')).toBeNull()
  })

  it('drops updates and deletes that target unknown or pinned memories', () => {
    const current = [memory('keep', 'Keep this.', true), memory('drop', 'Old fact.')]
    const proposal = {
      new_memories: [{ text: ' New fact. ' }, { text: '' }],
      updates: [
        { id: 'keep', text: 'rewritten pinned' },
        { id: 'drop', text: 'rewritten' },
        { id: 'missing', text: 'nowhere' }
      ],
      deletes: ['keep', 'drop', 'missing', 'drop']
    }
    const normalized = normalizeDreamProposal(proposal, current)
    expect(normalized).not.toBeNull()
    expect(normalized?.new_memories).toEqual([{ text: 'New fact.', tags: undefined }])
    expect(normalized?.updates).toEqual([{ id: 'drop', text: 'rewritten' }])
    expect(normalized?.deletes).toEqual(['drop'])
  })

  it('returns null when nothing survives validation', () => {
    const current = [memory('only', 'Only memory.', true)]
    const proposal = {
      new_memories: [],
      updates: [{ id: 'only', text: 'nope' }],
      deletes: ['only']
    }
    expect(normalizeDreamProposal(proposal, current)).toBeNull()
  })
})

import { describe, expect, it } from 'vitest'
import { messageChain, childCounts } from './graph'
import type { Message } from './types'

const base = {
  conversation_id: 'conversation',
  role: 'user' as const,
  content: 'text',
  model: null,
  created_at: 'now'
}

describe('message graph', () => {
  const messages: Message[] = [
    { ...base, id: 'root', parent_id: null },
    { ...base, id: 'first', parent_id: 'root' },
    { ...base, id: 'fork', parent_id: 'root' }
  ]

  it('selects a branch by its tip', () => {
    expect(messageChain(messages, 'fork').map((message) => message.id)).toEqual(['root', 'fork'])
  })

  it('counts branch points', () => {
    expect(childCounts(messages).get('root')).toBe(2)
  })
})

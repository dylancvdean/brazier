import { describe, expect, it } from 'vitest'
import { messageChain, childCounts, branchSiblings, ROOT_TIP } from './graph'
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

  it('returns an empty chain when rewound to ROOT_TIP so editing the first message branches cleanly', () => {
    expect(messageChain(messages, ROOT_TIP)).toEqual([])
  })

  it('counts branch points', () => {
    expect(childCounts(messages).get('root')).toBe(2)
  })

  it('orders the alternatives for a branch message including root siblings', () => {
    expect(branchSiblings(messages, 'fork').map((message) => message.id)).toEqual(['first', 'fork'])
    const withRootBranch: Message[] = [
      ...messages,
      { ...base, id: 'root-2', parent_id: null }
    ]
    expect(branchSiblings(withRootBranch, 'root').map((message) => message.id)).toEqual([
      'root',
      'root-2'
    ])
  })
})

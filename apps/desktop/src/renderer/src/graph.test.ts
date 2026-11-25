import { describe, expect, it } from 'vitest'
import { messageChain, childCounts, branchSiblings, refreshedMessageTip, ROOT_TIP } from './graph'
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

  it('keeps a first-message edit rooted when foreground refresh finds a newer message', () => {
    const refreshed = [...messages, { ...base, id: 'remote', parent_id: 'fork' }]
    const tip = refreshedMessageTip(refreshed, ROOT_TIP, 'fork')
    expect(tip).toBe(ROOT_TIP)
    expect(messageChain(refreshed, tip).at(-1)?.id ?? null).toBeNull()
    expect(refreshedMessageTip(refreshed, 'fork', 'fork')).toBe('remote')
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

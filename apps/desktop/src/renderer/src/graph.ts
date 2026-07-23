import type { Message } from './types'

export function messageChain(messages: Message[], tipId?: string | null): Message[] {
  if (messages.length === 0) return []
  const byId = new Map(messages.map((message) => [message.id, message]))
  let cursor = (tipId && byId.get(tipId)) || messages.at(-1)
  const result: Message[] = []
  const seen = new Set<string>()
  while (cursor && !seen.has(cursor.id)) {
    seen.add(cursor.id)
    result.push(cursor)
    cursor = cursor.parent_id ? byId.get(cursor.parent_id) : undefined
  }
  return result.reverse()
}

export function childCounts(messages: Message[]): Map<string, number> {
  const counts = new Map<string, number>()
  for (const message of messages) {
    if (message.parent_id) counts.set(message.parent_id, (counts.get(message.parent_id) ?? 0) + 1)
  }
  return counts
}

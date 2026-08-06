/**
 * Chat memory: recall injection, model-facing memory tools, and dreaming.
 *
 * Recall and the memory tools live client-side because memory is a renderer
 * concern: the daemon stores the rows, but deciding which memories a turn sees
 * and which turns may write them depends on the current chat's memory setting
 * and incognito state, which are renderer-owned.
 */

import {
  createMemory,
  deleteMemory,
  listMemories,
  streamCompletion,
  updateMemory,
  type ClientToolCall,
  type OpenAiFunctionTool
} from './api'
import type { Memory, Message } from './types'

/** Names of the client-side tools the model may call. */
export const MEMORY_SAVE_TOOL = 'save_memory'
export const MEMORY_RECALL_TOOL = 'recall_memory'

export function memoryToolDefinitions(): OpenAiFunctionTool[] {
  return [
    {
      type: 'function',
      function: {
        name: MEMORY_SAVE_TOOL,
        description:
          'Save a durable fact, preference, or relationship about the user so it is ' +
          'remembered across future conversations. Use for information the user states ' +
          'about themselves or asks you to remember; do not use for one-off trivia of the ' +
          'current conversation. Prefer a specific, self-contained sentence.',
        parameters: {
          type: 'object',
          properties: {
            memory: {
              type: 'string',
              description:
                'The fact to remember, as a specific self-contained sentence about the user.'
            },
            tags: {
              type: 'array',
              items: { type: 'string' },
              description: 'Optional short tags to group related memories (e.g. work, family).'
            }
          },
          required: ['memory']
        }
      }
    },
    {
      type: 'function',
      function: {
        name: MEMORY_RECALL_TOOL,
        description:
          'Search the user\'s saved long-term memories for something relevant to the current ' +
          'question. Call this when the user refers to something from a past conversation, a ' +
          'preference, or a relationship that is not in the visible conversation.',
        parameters: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: 'Keywords to search the memory store for.'
            }
          },
          required: ['query']
        }
      }
    }
  ]
}

export function isMemoryClientTool(name: string): boolean {
  return name === MEMORY_SAVE_TOOL || name === MEMORY_RECALL_TOOL
}

type MemoryToolArguments = {
  memory?: string
  tags?: string[]
  query?: string
}

function parseArguments(arguments_: string): MemoryToolArguments {
  try {
    const parsed = JSON.parse(arguments_) as Record<string, unknown>
    return {
      memory: typeof parsed.memory === 'string' ? parsed.memory : undefined,
      tags: Array.isArray(parsed.tags)
        ? parsed.tags.filter((tag): tag is string => typeof tag === 'string')
        : undefined,
      query: typeof parsed.query === 'string' ? parsed.query : undefined
    }
  } catch {
    return {}
  }
}

/**
 * Execute a client-side memory tool call, returning the output the model
 * should see as the `tool` result message.
 */
export async function executeMemoryClientTool(
  call: ClientToolCall,
  source: { conversation_id?: string | null; message_id?: string | null } = {}
): Promise<{ output: string; is_error: boolean }> {
  const args = parseArguments(call.arguments)
  if (call.name === MEMORY_SAVE_TOOL) {
    const text = args.memory?.trim()
    if (!text) {
      return { output: 'Error: save_memory requires a non-empty "memory" string.', is_error: true }
    }
    try {
      const memory = await createMemory({
        text,
        tags: args.tags,
        kind: 'fact',
        source_conversation_id: source.conversation_id ?? null,
        source_message_id: source.message_id ?? null
      })
      return {
        output: `Saved memory "${memory.text}".`,
        is_error: false
      }
    } catch (error) {
      return {
        output: `Error saving memory: ${error instanceof Error ? error.message : String(error)}`,
        is_error: true
      }
    }
  }
  if (call.name === MEMORY_RECALL_TOOL) {
    try {
      const memories = await listMemories(args.query)
      if (memories.length === 0) {
        return { output: 'No memories matched that search.', is_error: false }
      }
      const lines = memories
        .slice(0, 10)
        .map((memory) => `- ${memory.text}`)
        .join('\n')
      return {
        output: `Matching memories:\n${lines}`,
        is_error: false
      }
    } catch (error) {
      return {
        output: `Error recalling memories: ${error instanceof Error ? error.message : String(error)}`,
        is_error: true
      }
    }
  }
  return { output: `Error: unknown memory tool ${call.name}.`, is_error: true }
}

/**
 * Load the memories a chat turn should see: keyword search for the user's text
 * (recency-ordered when nothing matches), bounded to `count`.
 */
export async function loadRelevantMemories(
  query: string,
  count: number
): Promise<Memory[]> {
  if (count <= 0) return []
  try {
    return await listMemories(query || undefined)
  } catch {
    return []
  }
}

/** Render a memory store into the system message a turn is prefixed with. */
export function buildMemoryContext(
  memories: Memory[],
  budgetChars: number
): string | null {
  if (memories.length === 0) return null
  const lines: string[] = []
  let used = 0
  for (const memory of memories) {
    const line = `- ${memory.text}`
    if (used + line.length + 1 > budgetChars) break
    lines.push(line)
    used += line.length + 1
  }
  if (lines.length === 0) return null
  return [
    'The user has a long-term memory of the following facts, drawn from past conversations.',
    'Use them when they are relevant, and update them when the user corrects one.',
    ...lines
  ].join('\n')
}

// --- Dreaming ---------------------------------------------------------------

export type DreamingMode = 'off' | 'auto' | 'ask'

export type DreamInputConversation = {
  id: string
  title: string
  summary?: string | null
  updated_at: string
}

export type DreamProposal = {
  new_memories: { text: string; tags?: string[] }[]
  updates: { id: string; text: string }[]
  deletes: string[]
}

export type DreamResult = {
  created: number
  updated: number
  deleted: number
}

const DREAM_SYSTEM_PROMPT = `You are consolidating the user's long-term memory. Below are the current saved
memories and recent conversation summaries. Produce a single JSON object with
exactly these fields:
{
  "new_memories": [{"text": "sentence", "tags": ["tag"]}],
  "updates": [{"id": "existing-id", "text": "rewritten sentence"}],
  "deletes": ["existing-id"]
}

Rules:
- Keep the store small and high-signal. Merge overlapping memories into one,
  delete duplicates and facts the conversations contradict, and add anything new
  the summaries reveal that is not already covered.
- Do not invent facts that are not supported by the input.
- Do not touch memories marked [pinned].
- Every id in "updates" and "deletes" must appear in the CURRENT MEMORIES list.
- Each new memory is one specific, self-contained sentence.
- Return ONLY the JSON object, with no surrounding text or markdown.`

function renderDreamInput(
  memories: Memory[],
  conversations: DreamInputConversation[]
): string {
  const memoryLines = memories.length
    ? memories
        .map((memory) => `[id: ${memory.id}]${memory.pinned ? ' [pinned]' : ''} ${memory.text}`)
        .join('\n')
    : '(none)'
  const conversationLines = conversations.length
    ? conversations
        .map(
          (conversation) =>
            `- ${conversation.title} (${conversation.updated_at}): ${
              conversation.summary?.trim() || 'no summary'
            }`
        )
        .join('\n')
    : '(none)'
  return `CURRENT MEMORIES:\n${memoryLines}\n\nRECENT CONVERSATIONS:\n${conversationLines}`
}

/** Pull the first balanced JSON object out of a model's reply. */
export function extractDreamProposal(raw: string): Record<string, unknown> | null {
  const start = raw.indexOf('{')
  const end = raw.lastIndexOf('}')
  if (start < 0 || end <= start) return null
  const candidate = raw.slice(start, end + 1)
  try {
    const parsed: unknown = JSON.parse(candidate)
    return typeof parsed === 'object' && parsed !== null
      ? (parsed as Record<string, unknown>)
      : null
  } catch {
    return null
  }
}

/** Validate a parsed proposal against the current store, dropping anything
 * that references an unknown id or a pinned memory. */
export function normalizeDreamProposal(
  proposal: Record<string, unknown>,
  current: Memory[]
): DreamProposal | null {
  const byId = new Map(current.map((memory) => [memory.id, memory]))
  const pinned = new Set(current.filter((memory) => memory.pinned).map((memory) => memory.id))
  const newMemories: { text: string; tags?: string[] }[] = []
  if (Array.isArray(proposal.new_memories)) {
    for (const entry of proposal.new_memories) {
      if (typeof entry !== 'object' || entry === null) continue
      const record = entry as Record<string, unknown>
      const text = typeof record.text === 'string' ? record.text.trim() : ''
      if (!text) continue
      const tags = Array.isArray(record.tags)
        ? record.tags.filter((tag): tag is string => typeof tag === 'string' && tag.trim().length > 0)
        : undefined
      newMemories.push({ text, tags: tags?.length ? tags : undefined })
    }
  }
  const updates: { id: string; text: string }[] = []
  if (Array.isArray(proposal.updates)) {
    for (const entry of proposal.updates) {
      if (typeof entry !== 'object' || entry === null) continue
      const record = entry as Record<string, unknown>
      const id = typeof record.id === 'string' ? record.id : ''
      const text = typeof record.text === 'string' ? record.text.trim() : ''
      if (!id || !text || !byId.has(id) || pinned.has(id)) continue
      updates.push({ id, text })
    }
  }
  const deletes: string[] = []
  if (Array.isArray(proposal.deletes)) {
    for (const id of proposal.deletes) {
      if (typeof id === 'string' && byId.has(id) && !pinned.has(id) && !deletes.includes(id)) {
        deletes.push(id)
      }
    }
  }
  if (newMemories.length === 0 && updates.length === 0 && deletes.length === 0) return null
  return { new_memories: newMemories, updates, deletes }
}

/**
 * Run one dreaming pass: ask the model to consolidate the memory store from
 * recent conversation summaries, then apply the proposal.
 */
export async function dream(options: {
  model: string
  signal: AbortSignal
  memories: Memory[]
  conversations: DreamInputConversation[]
  onPartial?: (delta: string) => void
  onLoad?: (event: { phase: string; message: string }) => void
}): Promise<DreamResult> {
  const userText = renderDreamInput(options.memories, options.conversations)
  const now = new Date().toISOString()
  const system: Message = {
    id: `dream-system-${crypto.randomUUID()}`,
    conversation_id: 'dreaming',
    parent_id: null,
    role: 'system',
    content: DREAM_SYSTEM_PROMPT,
    model: null,
    created_at: now
  }
  const user: Message = {
    id: `dream-user-${crypto.randomUUID()}`,
    conversation_id: 'dreaming',
    parent_id: system.id,
    role: 'user',
    content: userText,
    model: null,
    created_at: now
  }
  const result = await streamCompletion(
    [system, user],
    options.model,
    options.signal,
    (token) => options.onPartial?.(token),
    {
      extraTools: [],
      toolChoice: 'none',
      onLoad: options.onLoad
    }
  )
  const proposal = extractDreamProposal(result.responseText)
  if (!proposal) {
    throw new Error('The model did not return a parseable memory consolidation.')
  }
  const normalized = normalizeDreamProposal(proposal, options.memories)
  if (!normalized) {
    return { created: 0, updated: 0, deleted: 0 }
  }
  let created = 0
  let updated = 0
  let deleted = 0
  for (const memory of normalized.new_memories) {
    await createMemory({ text: memory.text, tags: memory.tags, kind: 'summary' })
    created += 1
  }
  for (const update of normalized.updates) {
    await updateMemory(update.id, { text: update.text })
    updated += 1
  }
  for (const id of normalized.deletes) {
    await deleteMemory(id)
    deleted += 1
  }
  return { created, updated, deleted }
}

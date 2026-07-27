import type { ToolCallRecord } from './api'
import type { Message } from './types'

export type DisplayBlob = {
  sha256: string
  mime_type: string
  original_name?: string | null
}

export type AssistantSegment =
  | { kind: 'text'; text: string; key: string }
  | { kind: 'tool'; records: ToolCallRecord[]; key: string }
  | { kind: 'media'; blobs: DisplayBlob[]; key: string }

export type AssistantTurn = {
  kind: 'assistant'
  id: string
  branchId: string
  source?: string | null
  status?: string | null
  reasoning: string
  segments: AssistantSegment[]
}

export type ChatDisplayItem = { kind: 'message'; message: Message } | AssistantTurn

type StoredToolCall = {
  id?: unknown
  function?: {
    name?: unknown
    arguments?: unknown
  }
}

function messageText(message: Message): string {
  if (typeof message.content === 'string') return message.content
  return message.content
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

function messageBlobs(message: Message): DisplayBlob[] {
  if (typeof message.content === 'string') return []
  return message.content.flatMap((part) =>
    part.type === 'brazier_blob'
      ? [
          {
            sha256: part.brazier_blob.sha256,
            mime_type: part.brazier_blob.mime_type,
            original_name: part.brazier_blob.name
          }
        ]
      : []
  )
}

function messageReasoning(message: Message): string {
  const value = message.metadata?.reasoning_content
  return typeof value === 'string' ? value.trim() : ''
}

function mergeReasoning(current: string, next: string): string {
  if (!next) return current
  if (!current) return next
  // Some servers repeat the complete earlier reasoning prefix after a tool
  // result. Keep the longer version instead of showing the same CoT twice.
  if (next.startsWith(current)) return next
  if (current.startsWith(next)) return current
  return `${current}\n\n${next}`
}

function callsFromMessage(message: Message): ToolCallRecord[] {
  if (message.role !== 'assistant' || !Array.isArray(message.tool_calls)) return []
  return message.tool_calls.flatMap((value) => {
    const call = value as StoredToolCall
    const name = typeof call.function?.name === 'string' ? call.function.name : ''
    if (!name) return []
    const id = typeof call.id === 'string' ? call.id : `tool-${message.id}`
    const args = call.function?.arguments
    return [
      {
        call_id: id,
        name,
        arguments: typeof args === 'string' ? args : JSON.stringify(args ?? {}),
        output: '',
        is_error: false
      }
    ]
  })
}

function isGeneratedMediaDisplay(message: Message): boolean {
  return message.metadata?.generated_media_display === true
}

/**
 * Compose protocol-level assistant/tool/system records into the assistant turn
 * a person experienced. Segment order remains faithful to execution, while all
 * reasoning is shown once before the first visible action.
 */
export function buildChatDisplayItems(messages: Message[]): ChatDisplayItem[] {
  const items: ChatDisplayItem[] = []
  let turn: AssistantTurn | null = null
  let lastToolRecord: ToolCallRecord | null = null
  let segmentIndex = 0
  const seenMedia = new Set<string>()

  const flush = (): void => {
    if (turn && (turn.reasoning || turn.segments.length > 0)) items.push(turn)
    turn = null
    lastToolRecord = null
    segmentIndex = 0
    seenMedia.clear()
  }

  const ensureTurn = (message: Message): AssistantTurn => {
    if (!turn) {
      turn = {
        kind: 'assistant',
        id: `assistant-turn-${message.id}`,
        branchId: message.id,
        source: message.source,
        status: message.status,
        reasoning: '',
        segments: []
      }
    }
    return turn
  }

  const addMedia = (message: Message, blobs: DisplayBlob[]): void => {
    const fresh = blobs.filter((blob) => {
      const key = `${blob.sha256}:${blob.mime_type}`
      if (seenMedia.has(key)) return false
      seenMedia.add(key)
      return true
    })
    if (fresh.length === 0) return
    const current = ensureTurn(message)
    if (lastToolRecord) {
      lastToolRecord.media = [...(lastToolRecord.media ?? []), ...fresh]
      return
    }
    current.segments.push({
      kind: 'media',
      blobs: fresh,
      key: `media-${segmentIndex++}`
    })
  }

  for (const message of messages) {
    if (message.role === 'user') {
      flush()
      items.push({ kind: 'message', message })
      continue
    }

    if (message.role === 'assistant') {
      const current = ensureTurn(message)
      current.branchId = message.id
      current.source = message.source ?? current.source
      current.status = message.status ?? current.status
      current.reasoning = mergeReasoning(current.reasoning, messageReasoning(message))

      const text = messageText(message).trim()
      if (text && !isGeneratedMediaDisplay(message)) {
        current.segments.push({
          kind: 'text',
          text,
          key: `text-${segmentIndex++}`
        })
      }

      const calls = callsFromMessage(message)
      if (calls.length > 0) {
        current.segments.push({
          kind: 'tool',
          records: calls,
          key: `tool-${segmentIndex++}`
        })
      }
      for (const call of calls) {
        lastToolRecord = call
      }
      addMedia(message, messageBlobs(message))
      continue
    }

    if (message.role === 'tool') {
      const current = ensureTurn(message)
      const callId = message.tool_call_id ?? `tool-${message.id}`
      let record: ToolCallRecord | undefined
      for (const segment of current.segments) {
        if (segment.kind !== 'tool') continue
        record = segment.records.find((candidate) => candidate.call_id === callId)
        if (record) break
      }
      if (!record) {
        record = {
          call_id: callId,
          name: 'tool',
          arguments: '',
          output: '',
          is_error: false
        }
        current.segments.push({
          kind: 'tool',
          records: [record],
          key: `tool-${segmentIndex++}`
        })
      }
      const output = messageText(message)
      record.output = output
      record.is_error = output.trimStart().toLowerCase().startsWith('error:')
      lastToolRecord = record
      continue
    }

    // Generated-media system context is model plumbing, but its blob belongs
    // visually at this exact point in the preceding tool segment.
    addMedia(message, messageBlobs(message))
  }

  flush()
  return items
}

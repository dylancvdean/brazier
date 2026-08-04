import type { ToolCallRecord } from './api'
import type { Message } from './types'

export type DisplayBlob = {
  sha256: string
  mime_type: string
  original_name?: string | null
  /** The tool call that produced this blob, when the message records it. */
  call_id?: string
}

export type AssistantSegment =
  | { kind: 'reasoning'; text: string; key: string }
  | { kind: 'text'; text: string; key: string }
  | { kind: 'tool'; records: ToolCallRecord[]; key: string }
  | { kind: 'media'; blobs: DisplayBlob[]; key: string }

export type AssistantTurn = {
  kind: 'assistant'
  id: string
  branchId: string
  source?: string | null
  status?: string | null
  /** Merged thinking text, for the collapsed trace preview. */
  reasoning: string
  /** Ordered trace: thinking, tool calls, media, and intermediate text. */
  trace: AssistantSegment[]
  /** The final response text, shown outside the trace. */
  answer: string
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
            original_name: part.brazier_blob.name,
            call_id: part.brazier_blob.call_id
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
 * a person experienced. Reasoning, tool calls, media, and any intermediate
 * text are kept in execution order as the turn's trace; the final response text
 * is separated out as the visible answer. All reasoning is shown once inside
 * the trace rather than repeated in a collapsed header.
 */
export function buildChatDisplayItems(messages: Message[]): ChatDisplayItem[] {
  const items: ChatDisplayItem[] = []
  let turn: AssistantTurn | null = null
  let lastToolRecord: ToolCallRecord | null = null
  let segmentIndex = 0
  const seenMedia = new Set<string>()

  const flush = (): void => {
    if (!turn) return
    // The final text the model produced is the answer; anything earlier is
    // part of the working trace inside the disclosure.
    for (let index = turn.trace.length - 1; index >= 0; index -= 1) {
      const segment = turn.trace[index]
      if (segment.kind === 'text') {
        turn.answer = segment.text
        turn.trace.splice(index, 1)
        break
      }
    }
    if (turn.reasoning || turn.trace.length > 0 || turn.answer) items.push(turn)
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
        trace: [],
        answer: ''
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
    // Attach each blob to the tool call that produced it when the blob records
    // its call_id (the generated-media display message does); otherwise fall
    // back to the most recent tool record.
    const byCallId = new Map<string, ToolCallRecord>()
    for (const segment of current.trace) {
      if (segment.kind !== 'tool') continue
      for (const record of segment.records) byCallId.set(record.call_id, record)
    }
    const unattached: DisplayBlob[] = []
    for (const blob of fresh) {
      const record = blob.call_id ? byCallId.get(blob.call_id) : undefined
      if (record) {
        record.media = [...(record.media ?? []), blob]
      } else {
        unattached.push(blob)
      }
    }
    if (unattached.length === 0) return
    if (lastToolRecord) {
      lastToolRecord.media = [...(lastToolRecord.media ?? []), ...unattached]
      return
    }
    current.trace.push({
      kind: 'media',
      blobs: unattached,
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

      const reasoning = messageReasoning(message)
      if (reasoning) {
        current.trace.push({
          kind: 'reasoning',
          text: reasoning,
          key: `reasoning-${segmentIndex++}`
        })
      }

      const text = messageText(message).trim()
      if (text && !isGeneratedMediaDisplay(message)) {
        current.trace.push({
          kind: 'text',
          text,
          key: `text-${segmentIndex++}`
        })
      }

      const calls = callsFromMessage(message)
      if (calls.length > 0) {
        current.trace.push({
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
      for (const segment of current.trace) {
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
        current.trace.push({
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

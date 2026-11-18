/**
 * Streaming `<think>` splitter and text tool-call fallback parser.
 *
 * Separates inline reasoning tags out of `content` deltas when the engine did
 * not populate `reasoning_content`, holds back raw tool-call markup so XML/JSON
 * control tokens never flash as assistant prose, and extracts text-encoded tool
 * calls across Qwen, Hermes, Harmony, Mistral/Devstral, and fenced JSON dialects.
 */

import { parseLooseJson, resolveToolNameAlias } from '../core/modelCompat'

export type StreamChunk = {
  channel: 'text' | 'reasoning'
  delta: string
}

export type ExtractedTextToolCall = {
  id: string
  name: string
  arguments: Record<string, unknown>
}

const THINK_TAGS = [
  { open: '<think>', close: '</think>' },
  { open: '<thought>', close: '</thought>' },
  { open: '<|thought|>', close: '<|/thought|>' }
] as const

const TOOL_HOLDBACK_MARKERS = [
  '<tool_call>',
  '<function=',
  '[TOOL_CALLS]',
  '<|channel|>commentary'
] as const

function trailingPrefixLength(text: string, markers: readonly string[]): number {
  let maxHold = 0
  for (const marker of markers) {
    const limit = Math.min(text.length, marker.length - 1)
    for (let len = limit; len > maxHold; len -= 1) {
      if (marker.startsWith(text.slice(text.length - len))) {
        maxHold = len
        break
      }
    }
  }
  return maxHold
}

export class ThinkStreamSplitter {
  private backendReasoning = false
  private mode: 'text' | 'thinking' | 'tool_holdback' = 'text'
  private activeCloseTag = ''
  private stripLeadingNewlines = false
  private buffer = ''

  visibleText = ''
  reasoningText = ''
  heldBackText = ''

  /** Notify the splitter that structured `reasoning_content` is active. */
  markBackendReasoning(): void {
    this.backendReasoning = true
  }

  push(delta: string): StreamChunk[] {
    if (!delta) return []
    this.buffer += delta
    const out: StreamChunk[] = []

    const emit = (channel: 'text' | 'reasoning', text: string): void => {
      if (!text) return
      if (channel === 'text') this.visibleText += text
      else this.reasoningText += text
      out.push({ channel, delta: text })
    }

    while (this.buffer.length > 0) {
      if (this.mode === 'tool_holdback') {
        this.heldBackText += this.buffer
        this.buffer = ''
        break
      }

      if (this.mode === 'thinking') {
        const closeIdx = this.buffer.indexOf(this.activeCloseTag)
        if (closeIdx >= 0) {
          emit('reasoning', this.buffer.slice(0, closeIdx))
          this.buffer = this.buffer.slice(closeIdx + this.activeCloseTag.length)
          this.mode = 'text'
          this.activeCloseTag = ''
          this.stripLeadingNewlines = this.visibleText.length === 0
          continue
        }
        const hold = trailingPrefixLength(this.buffer, [this.activeCloseTag])
        const safeLen = this.buffer.length - hold
        if (safeLen > 0) {
          emit('reasoning', this.buffer.slice(0, safeLen))
          this.buffer = this.buffer.slice(safeLen)
        }
        break
      }

      if (this.stripLeadingNewlines) {
        this.buffer = this.buffer.replace(/^[\r\n]+/, '')
        if (!this.buffer) break
        this.stripLeadingNewlines = false
      }

      let earliestIdx = -1
      let matchedThink: (typeof THINK_TAGS)[number] | undefined
      let matchedHoldback: string | undefined

      if (!this.backendReasoning) {
        for (const tag of THINK_TAGS) {
          const idx = this.buffer.indexOf(tag.open)
          if (idx >= 0 && (earliestIdx < 0 || idx < earliestIdx)) {
            earliestIdx = idx
            matchedThink = tag
            matchedHoldback = undefined
          }
        }
      }

      for (const marker of TOOL_HOLDBACK_MARKERS) {
        const idx = this.buffer.indexOf(marker)
        if (idx >= 0 && (earliestIdx < 0 || idx < earliestIdx)) {
          earliestIdx = idx
          matchedHoldback = marker
          matchedThink = undefined
        }
      }

      if (earliestIdx >= 0) {
        if (earliestIdx > 0) {
          emit('text', this.buffer.slice(0, earliestIdx))
        }
        if (matchedThink) {
          this.buffer = this.buffer.slice(earliestIdx + matchedThink.open.length)
          this.mode = 'thinking'
          this.activeCloseTag = matchedThink.close
          continue
        }
        if (matchedHoldback) {
          this.heldBackText += this.buffer.slice(earliestIdx)
          this.buffer = ''
          this.mode = 'tool_holdback'
          break
        }
      }

      const activeMarkers: string[] = [...TOOL_HOLDBACK_MARKERS]
      if (!this.backendReasoning) {
        for (const tag of THINK_TAGS) activeMarkers.push(tag.open)
      }
      const hold = trailingPrefixLength(this.buffer, activeMarkers)
      const safeLen = this.buffer.length - hold
      if (safeLen > 0) {
        emit('text', this.buffer.slice(0, safeLen))
        this.buffer = this.buffer.slice(safeLen)
      }
      break
    }

    return out
  }

  flush(): StreamChunk[] {
    if (!this.buffer) return []
    const rem = this.buffer
    this.buffer = ''
    if (this.mode === 'thinking') {
      this.reasoningText += rem
      return [{ channel: 'reasoning', delta: rem }]
    }
    if (this.mode === 'tool_holdback') {
      this.heldBackText += rem
      return []
    }
    this.visibleText += rem
    return [{ channel: 'text', delta: rem }]
  }
}

function normalizeArgsObject(raw: unknown): Record<string, unknown> {
  if (typeof raw === 'string') {
    const parsed = parseLooseJson(raw)
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
    return {}
  }
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    return raw as Record<string, unknown>
  }
  return {}
}

function parseJsonCallCandidate(
  candidate: unknown,
  enabledTools: readonly string[],
  nextId: () => string
): ExtractedTextToolCall | null {
  if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) return null
  const obj = candidate as Record<string, unknown>
  const fn =
    obj.function && typeof obj.function === 'object' && !Array.isArray(obj.function)
      ? (obj.function as Record<string, unknown>)
      : undefined
  const rawName =
    typeof fn?.name === 'string'
      ? fn.name
      : typeof obj.name === 'string'
        ? obj.name
        : typeof obj.tool === 'string'
          ? obj.tool
          : ''
  if (!rawName.trim()) return null
  const name = resolveToolNameAlias(rawName, enabledTools)
  const rawArgs = fn?.arguments ?? obj.arguments ?? obj.parameters ?? obj.args ?? {}
  const id = typeof obj.id === 'string' && obj.id.trim() ? obj.id.trim() : nextId()
  return { id, name, arguments: normalizeArgsObject(rawArgs) }
}

function parseParameterValue(raw: string): unknown {
  const cleaned = raw.replace(/^\r?\n/, '').replace(/\r?\n$/, '')
  const trimmed = cleaned.trim()
  if (
    (trimmed.startsWith('[') && trimmed.endsWith(']')) ||
    (trimmed.startsWith('{') && trimmed.endsWith('}'))
  ) {
    try {
      return JSON.parse(trimmed)
    } catch {
      return cleaned
    }
  }
  return cleaned
}

function parseFunctionXml(
  segment: string,
  enabledTools: readonly string[],
  nextId: () => string
): ExtractedTextToolCall[] {
  const calls: ExtractedTextToolCall[] = []
  const fnPattern = /<function=\s*["']?([^>"'\s]+)["']?\s*>([\s\S]*?)(?:<\/function>|$)/g
  let match: RegExpExecArray | null
  while ((match = fnPattern.exec(segment)) !== null) {
    const rawName = match[1]?.trim() ?? ''
    if (!rawName) continue
    const body = match[2] ?? ''
    const args: Record<string, unknown> = {}
    const paramPattern = /<parameter=\s*["']?([^>"'\s]+)["']?\s*>([\s\S]*?)<\/parameter>/g
    let paramMatch: RegExpExecArray | null
    while ((paramMatch = paramPattern.exec(body)) !== null) {
      const key = paramMatch[1]?.trim()
      if (key) args[key] = parseParameterValue(paramMatch[2] ?? '')
    }
    calls.push({
      id: nextId(),
      name: resolveToolNameAlias(rawName, enabledTools),
      arguments: args
    })
  }
  return calls
}

function findBalancedJsonEnd(text: string, start: number): number {
  const open = text[start]
  const close = open === '[' ? ']' : open === '{' ? '}' : ''
  if (!close) return -1
  let depth = 0
  let inString = false
  let escaped = false
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i]!
    if (inString) {
      if (escaped) escaped = false
      else if (ch === '\\') escaped = true
      else if (ch === '"') inString = false
      continue
    }
    if (ch === '"') {
      inString = true
    } else if (ch === open) {
      depth += 1
    } else if (ch === close) {
      depth -= 1
      if (depth === 0) return i + 1
    }
  }
  return -1
}

const FENCED_TOOL_KEYS = new Set([
  'id',
  'type',
  'name',
  'tool',
  'function',
  'arguments',
  'parameters',
  'args'
])

/**
 * Fallback parser for models whose backend emitted tool calls inside `content`
 * rather than OpenAI `delta.tool_calls`.
 */
export function extractTextToolCalls(
  text: string,
  enabledTools: readonly string[]
): { cleanedText: string; toolCalls: ExtractedTextToolCall[] } {
  if (!text.trim()) return { cleanedText: text, toolCalls: [] }

  const toolCalls: ExtractedTextToolCall[] = []
  let seq = 0
  const nextId = (): string => {
    seq += 1
    return `call_text_${seq}`
  }

  let working = text

  // 1 & 2a. <tool_call>{...}</tool_call> and <tool_call><function=...>...</function></tool_call>
  working = working.replace(/<tool_call>([\s\S]*?)(?:<\/tool_call>|$)/g, (full, inner: string) => {
    if (inner.includes('<function=')) {
      const parsedXml = parseFunctionXml(inner, enabledTools, nextId)
      if (parsedXml.length > 0) {
        toolCalls.push(...parsedXml)
        return ''
      }
    }
    const parsed = parseLooseJson(inner)
    const items = Array.isArray(parsed) ? parsed : [parsed]
    const extracted = items
      .map((item) => parseJsonCallCandidate(item, enabledTools, nextId))
      .filter((item): item is ExtractedTextToolCall => item !== null)
    if (extracted.length > 0) {
      toolCalls.push(...extracted)
      return ''
    }
    return full
  })

  // 2b. Standalone <function=name><parameter=k>v</parameter></function>
  working = working.replace(
    /<function=\s*["']?([^>"'\s]+)["']?\s*>[\s\S]*?<\/function>/g,
    (full) => {
      const parsedXml = parseFunctionXml(full, enabledTools, nextId)
      if (parsedXml.length > 0) {
        toolCalls.push(...parsedXml)
        return ''
      }
      return full
    }
  )

  // 3. Harmony <|channel|>commentary to=functions.NAME ...<|message|>{...}
  working = working.replace(
    /(?:<\|start\|>assistant)?<\|channel\|>commentary\s+to=([^\s<]+)(?:[^<]|<(?!\|message\|>))*<\|message\|>([\s\S]*?)(?:<\|call\|>|<\|end\|>|<\|return\|>|(?=<\|start\|>|<\|channel\|>)|$)/g,
    (full, rawTarget: string, rawPayload: string) => {
      const target = rawTarget.replace(/:.*$/, '').trim()
      if (!target) return full
      const parsed = parseLooseJson(rawPayload)
      if (parsed === undefined && rawPayload.trim() !== '') return full
      toolCalls.push({
        id: nextId(),
        name: resolveToolNameAlias(target, enabledTools),
        arguments: normalizeArgsObject(parsed ?? {})
      })
      return ''
    }
  )

  // 4. Mistral / Devstral [TOOL_CALLS] [...] and [TOOL_CALLS]name[ARGS]{...}
  let tcMarker = working.indexOf('[TOOL_CALLS]')
  while (tcMarker >= 0) {
    let cursor = tcMarker + '[TOOL_CALLS]'.length
    while (cursor < working.length && /\s/.test(working[cursor]!)) cursor += 1

    if (working[cursor] === '[') {
      const arrayEnd = findBalancedJsonEnd(working, cursor)
      if (arrayEnd > cursor) {
        try {
          const arr = JSON.parse(working.slice(cursor, arrayEnd)) as unknown
          if (Array.isArray(arr)) {
            const extracted = arr
              .map((item) => parseJsonCallCandidate(item, enabledTools, nextId))
              .filter((item): item is ExtractedTextToolCall => item !== null)
            if (extracted.length > 0) {
              toolCalls.push(...extracted)
              working = working.slice(0, tcMarker) + working.slice(arrayEnd)
              tcMarker = working.indexOf('[TOOL_CALLS]', tcMarker)
              continue
            }
          }
        } catch {
          // Leave untouched if invalid JSON.
        }
      }
    } else {
      const compactMatch = working.slice(cursor).match(/^([a-zA-Z0-9_./-]+)\s*\[ARGS\]\s*/)
      if (compactMatch) {
        const braceStart = cursor + compactMatch[0].length
        const braceEnd = findBalancedJsonEnd(working, braceStart)
        if (braceEnd > braceStart) {
          const parsed = parseLooseJson(working.slice(braceStart, braceEnd))
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            toolCalls.push({
              id: nextId(),
              name: resolveToolNameAlias(compactMatch[1]!, enabledTools),
              arguments: parsed as Record<string, unknown>
            })
            working = working.slice(0, tcMarker) + working.slice(braceEnd)
            tcMarker = working.indexOf('[TOOL_CALLS]', tcMarker)
            continue
          }
        }
      }
    }
    tcMarker = working.indexOf('[TOOL_CALLS]', tcMarker + '[TOOL_CALLS]'.length)
  }

  // 5. Standalone fenced ```json {"name": "...", "arguments": {...}} ``` when resolved name is enabled
  if (toolCalls.length === 0 && enabledTools.length > 0) {
    working = working.replace(/```(?:json|tool_call|tool)?\s*\r?\n?([\s\S]*?)```/gi, (full, inner: string) => {
      const trimmed = inner.trim()
      let parsed: unknown
      try {
        parsed = JSON.parse(trimmed)
      } catch {
        return full
      }
      const items = Array.isArray(parsed) ? parsed : [parsed]
      if (items.length === 0) return full
      const extracted: ExtractedTextToolCall[] = []
      for (const item of items) {
        if (!item || typeof item !== 'object' || Array.isArray(item)) return full
        const keys = Object.keys(item as Record<string, unknown>)
        if (keys.length === 0 || !keys.every((k) => FENCED_TOOL_KEYS.has(k))) return full
        const candidate = parseJsonCallCandidate(item, enabledTools, nextId)
        if (!candidate || !enabledTools.includes(candidate.name)) return full
        extracted.push(candidate)
      }
      toolCalls.push(...extracted)
      return ''
    })
  }

  return {
    cleanedText: toolCalls.length > 0 ? working.trim() : text,
    toolCalls
  }
}

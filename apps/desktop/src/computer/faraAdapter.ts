/**
 * Pure TypeScript port of the Fara1.5 XML `<tool_call>` dialect.
 *
 * Used client-side when the daemon parse endpoint is unavailable, and as the
 * reference shape for ComputerAction objects elsewhere in the desktop app.
 */

export type ComputerAction =
  | { type: 'screenshot' }
  | { type: 'left_click'; x: number; y: number }
  | { type: 'right_click'; x: number; y: number }
  | { type: 'double_click'; x: number; y: number }
  | { type: 'triple_click'; x: number; y: number }
  | { type: 'mouse_move'; x: number; y: number }
  | {
      type: 'left_click_drag'
      start_x: number
      start_y: number
      end_x: number
      end_y: number
    }
  | { type: 'type'; text: string }
  | { type: 'keypress'; keys: string[] }
  | { type: 'scroll'; x: number; y: number; delta_x: number; delta_y: number }
  | { type: 'wait'; milliseconds: number }
  | { type: 'visit_url'; url: string }
  | { type: 'web_search'; query: string }
  | { type: 'memorize'; fact: string }
  | { type: 'ask_user'; question: string }
  | { type: 'terminate'; response?: string | null }
  | { type: 'error'; error: string; raw?: string }

export type FaraParseResult = {
  thought: string | null
  actions: ComputerAction[]
  raw_tool_calls: string[]
}

/** Parse model output that may contain a thought block plus Fara tool calls. */
export function parseFaraOutput(text: string): FaraParseResult {
  const thought = extractThought(text)
  const actions: ComputerAction[] = []
  const raw_tool_calls: string[] = []

  for (const block of extractToolCallBlocks(text)) {
    raw_tool_calls.push(block)
    try {
      actions.push(parseToolCallJson(block))
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error)
      console.error(`[fara] failed to parse tool call: ${message}`, block)
      actions.push({ type: 'error', error: 'parse_error', raw: block.slice(0, 500) })
    }
  }

  // Some servers wrap a single JSON object without XML.
  if (actions.length === 0) {
    const trimmed = text.trim()
    if (trimmed.startsWith('{')) {
      try {
        actions.push(parseToolCallJson(trimmed))
        raw_tool_calls.push(trimmed)
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error)
        console.error(`[fara] failed to parse model output: ${message}`, trimmed)
        actions.push({ type: 'error', error: 'parse_error', raw: trimmed.slice(0, 500) })
        raw_tool_calls.push(trimmed)
      }
    }
  }

  return { thought, actions, raw_tool_calls }
}

function extractThought(text: string): string | null {
  for (const [open, close] of [
    ['<think>', '</think>'],
    ['<thought>', '</thought>'],
    ['```thought', '```']
  ] as const) {
    const start = text.indexOf(open)
    if (start < 0) continue
    const after = start + open.length
    const end = text.indexOf(close, after)
    if (end < 0) continue
    const thought = text.slice(after, end).trim()
    if (thought) return thought
  }
  // Text before the first tool call is treated as thought.
  const idx = text.indexOf('<tool_call>')
  if (idx >= 0) {
    const prefix = text.slice(0, idx).trim()
    if (prefix) return prefix
  }
  return null
}

function extractToolCallBlocks(text: string): string[] {
  const blocks: string[] = []
  let rest = text
  while (true) {
    const start = rest.indexOf('<tool_call>')
    if (start < 0) break
    const after = start + '<tool_call>'.length
    const end = rest.indexOf('</tool_call>', after)
    if (end < 0) break
    blocks.push(rest.slice(after, end).trim())
    rest = rest.slice(end + '</tool_call>'.length)
  }
  return blocks
}

function parseToolCallJson(block: string): ComputerAction {
  const value = JSON.parse(block) as Record<string, unknown>
  let args: unknown
  if (value.name === 'computer_use') {
    args = value.arguments ?? value.parameters ?? null
  } else if (value.action != null || value.type != null) {
    args = value
  } else if (value.function && typeof value.function === 'object') {
    const fn = value.function as Record<string, unknown>
    args = fn.arguments ?? null
  } else {
    args = value
  }

  if (typeof args === 'string') {
    args = JSON.parse(args)
  }

  return actionFromArgs((args && typeof args === 'object' ? args : {}) as Record<string, unknown>)
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function coord(
  args: Record<string, unknown>,
  keyX: string,
  keyY: string
): [number, number] {
  const arr = args.coordinate
  if (Array.isArray(arr) && arr.length >= 2) {
    return [asNumber(arr[0]), asNumber(arr[1])]
  }
  return [
    asNumber(args[keyX] ?? args.x),
    asNumber(args[keyY] ?? args.y)
  ]
}

function actionFromArgs(args: Record<string, unknown>): ComputerAction {
  const action =
    (typeof args.action === 'string' && args.action) ||
    (typeof args.type === 'string' && args.type) ||
    'screenshot'

  switch (action) {
    case 'screenshot':
      return { type: 'screenshot' }
    case 'left_click':
    case 'click': {
      const [x, y] = coord(args, 'x', 'y')
      return { type: 'left_click', x, y }
    }
    case 'right_click': {
      const [x, y] = coord(args, 'x', 'y')
      return { type: 'right_click', x, y }
    }
    case 'double_click': {
      const [x, y] = coord(args, 'x', 'y')
      return { type: 'double_click', x, y }
    }
    case 'triple_click': {
      const [x, y] = coord(args, 'x', 'y')
      return { type: 'triple_click', x, y }
    }
    case 'mouse_move': {
      const [x, y] = coord(args, 'x', 'y')
      return { type: 'mouse_move', x, y }
    }
    case 'left_click_drag':
    case 'drag': {
      let start_x = 0
      let start_y = 0
      let end_x = 0
      let end_y = 0
      const startArr = args.start_coordinate
      if (Array.isArray(startArr) && startArr.length >= 2) {
        start_x = asNumber(startArr[0])
        start_y = asNumber(startArr[1])
      } else {
        ;[start_x, start_y] = coord(args, 'start_x', 'start_y')
      }
      const endArr = args.coordinate
      if (Array.isArray(endArr) && endArr.length >= 2) {
        end_x = asNumber(endArr[0])
        end_y = asNumber(endArr[1])
      } else {
        ;[end_x, end_y] = coord(args, 'end_x', 'end_y')
      }
      return { type: 'left_click_drag', start_x, start_y, end_x, end_y }
    }
    case 'type':
    case 'type_text':
      return { type: 'type', text: typeof args.text === 'string' ? args.text : '' }
    case 'keypress':
    case 'key':
    case 'hotkey': {
      let keys: string[] = []
      if (Array.isArray(args.keys)) {
        keys = args.keys.filter((k): k is string => typeof k === 'string')
      } else if (typeof args.key === 'string') {
        keys = [args.key]
      }
      return { type: 'keypress', keys }
    }
    case 'scroll': {
      const [x, y] = coord(args, 'x', 'y')
      return {
        type: 'scroll',
        x,
        y,
        delta_x: asNumber(args.delta_x ?? args.scroll_x),
        delta_y: asNumber(args.delta_y ?? args.scroll_y ?? args.pixels, -400)
      }
    }
    case 'wait':
      return {
        type: 'wait',
        milliseconds: asNumber(args.milliseconds ?? args.ms ?? args.time, 1000)
      }
    case 'visit_url':
    case 'goto':
    case 'navigate':
      return {
        type: 'visit_url',
        url: typeof args.url === 'string' ? args.url : 'about:blank'
      }
    case 'web_search':
    case 'search':
      return {
        type: 'web_search',
        query:
          typeof args.query === 'string'
            ? args.query
            : typeof args.text === 'string'
              ? args.text
              : ''
      }
    case 'pause_and_memorize_fact':
    case 'memorize':
      return {
        type: 'memorize',
        fact:
          typeof args.fact === 'string'
            ? args.fact
            : typeof args.text === 'string'
              ? args.text
              : ''
      }
    case 'ask_user_question':
    case 'ask_user':
      return {
        type: 'ask_user',
        question:
          typeof args.question === 'string'
            ? args.question
            : typeof args.text === 'string'
              ? args.text
              : 'Need your input.'
      }
    case 'terminate':
    case 'done':
    case 'finish':
      return {
        type: 'terminate',
        response:
          typeof args.response === 'string'
            ? args.response
            : typeof args.text === 'string'
              ? args.text
              : null
      }
    default:
      throw new Error(`unsupported Fara action: ${action}`)
  }
}

/** Whether a model id/name looks like a Fara computer-use model. */
export function looksLikeFaraModel(modelId: string): boolean {
  const lower = modelId.toLowerCase()
  return lower.includes('fara') || lower.includes('fara1.5') || lower.includes('fara-1.5')
}

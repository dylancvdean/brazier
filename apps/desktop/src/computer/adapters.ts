import {
  actionFromArgs,
  extractThought,
  extractToolCallBlocks,
  looksLikeFaraModel,
  parseFaraOutput,
  parseToolCallJson,
  parseXmlFunctionCall,
  splitKeyChord,
  type ComputerAction,
  type FaraParseResult
} from './faraAdapter'
import {
  recipeForModel,
  type ComputerActionDialect,
  type CoordinateSpace
} from './recipes'

export type ComputerModelInfo = {
  id: string
  engine?: string
  library_label?: string | null
  repo_id?: string | null
  capabilities?: {
    tools?: boolean
    reasoning?: boolean
    computer_use?: boolean
    input_modalities?: string[]
  }
}

export type ComputerViewportDimensions = {
  width: number
  height: number
}

export type ComputerAdapterSpec = {
  dialect: ComputerActionDialect
  coordinateSpace: CoordinateSpace
  label: string
}

export type ComputerOpenAiTool = {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: Record<string, unknown>
  }
}

export type ComputerCompletionInput = {
  responseText: string
  reasoningText?: string
  clientToolCalls?: Array<{
    id: string
    name: string
    arguments: string
  }>
}

export type ParsedComputerCompletion = {
  thought: string | null
  actions: ComputerAction[]
  /** Durable non-empty assistant content to persist into ComputerStep history. */
  assistantContent: string
}

export function looksLikeUiTarsModel(modelId: string): boolean {
  const lower = modelId.toLowerCase()
  return lower.includes('ui-tars') || lower.includes('uitars')
}

export function looksLikePixelCoordinateModel(modelId: string): boolean {
  const lower = modelId.toLowerCase()
  return (
    lower.includes('qwen2.5-vl') ||
    lower.includes('qwen2_5_vl') ||
    lower.includes('qwen2.5vl') ||
    lower.includes('llama-4') ||
    lower.includes('llama4') ||
    lower.includes('pixtral')
  )
}

export function looksLikeUnitCoordinateModel(modelId: string): boolean {
  const lower = modelId.toLowerCase()
  return lower.includes('norm1') || lower.includes('seeclick')
}

export function resolveComputerAdapter(
  modelId: string,
  model?: ComputerModelInfo
): ComputerAdapterSpec {
  const haystack = [modelId, model?.repo_id ?? '', model?.library_label ?? '']
    .filter(Boolean)
    .join(' ')
  const recipe = recipeForModel(haystack)
  const toolsExplicitlyDisabled =
    model?.capabilities?.tools === false ||
    (model?.engine === 'mlx-vlm' && model.capabilities?.tools !== true)

  if (recipe) {
    if (recipe.actionDialect === 'fara-xml' || recipe.actionDialect === 'ui-tars') {
      return {
        dialect: recipe.actionDialect,
        coordinateSpace: recipe.coordinateSpace,
        label: recipe.label
      }
    }
    return {
      dialect: toolsExplicitlyDisabled ? 'prompted-json' : recipe.actionDialect,
      coordinateSpace: recipe.coordinateSpace,
      label: recipe.label
    }
  }

  if (looksLikeFaraModel(haystack)) {
    return { dialect: 'fara-xml', coordinateSpace: 'norm1000', label: 'Fara' }
  }
  if (looksLikeUiTarsModel(haystack)) {
    return { dialect: 'ui-tars', coordinateSpace: 'norm1000', label: 'UI-TARS' }
  }

  const coordinateSpace: CoordinateSpace = looksLikePixelCoordinateModel(haystack)
    ? 'pixels'
    : looksLikeUnitCoordinateModel(haystack)
      ? 'norm1'
      : 'norm1000'

  if (toolsExplicitlyDisabled) {
    return {
      dialect: 'prompted-json',
      coordinateSpace,
      label: 'Prompted Vision Model'
    }
  }

  return {
    dialect: 'openai-tools',
    coordinateSpace,
    label: 'OpenAI Tool VLM'
  }
}

export function buildComputerUseTool(
  target: 'browser' | 'desktop',
  coordinateSpace: CoordinateSpace = 'norm1000',
  viewport?: ComputerViewportDimensions | null
): ComputerOpenAiTool {
  const supportedActions =
    target === 'desktop'
      ? [
          'key',
          'type',
          'mouse_move',
          'left_click',
          'right_click',
          'double_click',
          'triple_click',
          'left_click_drag',
          'scroll',
          'pause_and_memorize_fact',
          'ask_user_question',
          'wait',
          'terminate'
        ]
      : [
          'key',
          'type',
          'mouse_move',
          'left_click',
          'right_click',
          'double_click',
          'triple_click',
          'left_click_drag',
          'scroll',
          'visit_url',
          'web_search',
          'pause_and_memorize_fact',
          'ask_user_question',
          'wait',
          'terminate'
        ]

  const coordDetail =
    coordinateSpace === 'pixels' && viewport && viewport.width > 0 && viewport.height > 0
      ? `pixel coordinates within [0..${viewport.width}, 0..${viewport.height}]`
      : coordinateSpace === 'norm1'
        ? 'normalized unit coordinates within [0.0..1.0, 0.0..1.0]'
        : 'normalized 0..1000 screen coordinates [x, y] (top-left is [0, 0], bottom-right is [1000, 1000])'

  return {
    type: 'function',
    function: {
      name: 'computer_use',
      description:
        target === 'desktop'
          ? `Perform one GUI action on the visible desktop using ${coordDetail}.`
          : `Perform one action in the isolated browser using ${coordDetail}.`,
      parameters: {
        type: 'object',
        additionalProperties: false,
        required: ['action'],
        properties: {
          action: {
            type: 'string',
            enum: supportedActions,
            description: 'The single next action to execute.'
          },
          coordinate: {
            type: 'array',
            items: { type: 'number' },
            minItems: 2,
            maxItems: 2,
            description: `Target [x, y] coordinate (or drag end point) in ${coordDetail}.`
          },
          start_coordinate: {
            type: 'array',
            items: { type: 'number' },
            minItems: 2,
            maxItems: 2,
            description: `Drag start [x, y] coordinate for left_click_drag in ${coordDetail}.`
          },
          text: {
            type: 'string',
            description: 'Text to enter when action is "type".'
          },
          keys: {
            type: 'array',
            items: { type: 'string' },
            description: 'Key or shortcut chord when action is "key", e.g. ["Enter"] or ["Ctrl", "L"].'
          },
          delta_x: {
            type: 'number',
            description: 'Horizontal wheel scroll delta when action is "scroll".'
          },
          delta_y: {
            type: 'number',
            description:
              'Vertical wheel scroll delta when action is "scroll" (positive scrolls down, negative scrolls up).'
          },
          url: {
            type: 'string',
            description: 'Full URL to open when action is "visit_url" (browser target only).'
          },
          query: {
            type: 'string',
            description: 'Search query when action is "web_search" (browser target only).'
          },
          fact: {
            type: 'string',
            description: 'Important fact from the screen to retain when action is "pause_and_memorize_fact".'
          },
          question: {
            type: 'string',
            description: 'Clarifying or confirmation question for the user when action is "ask_user_question".'
          },
          response: {
            type: 'string',
            description: 'Final answer or completion summary when action is "terminate".'
          },
          milliseconds: {
            type: 'number',
            description: 'Duration to pause in milliseconds when action is "wait".'
          }
        }
      }
    }
  }
}

function clamp1000(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.min(1000, Math.max(0, value))
}

export function normalizePointTo1000(
  x: number,
  y: number,
  coordinateSpace: CoordinateSpace,
  viewport?: ComputerViewportDimensions | null
): [number, number] {
  if (coordinateSpace === 'norm1') {
    return [clamp1000(x * 1000), clamp1000(y * 1000)]
  }
  if (coordinateSpace === 'pixels') {
    const width = viewport?.width && viewport.width > 0 ? viewport.width : 1000
    const height = viewport?.height && viewport.height > 0 ? viewport.height : 1000
    return [clamp1000((x / width) * 1000), clamp1000((y / height) * 1000)]
  }
  return [clamp1000(x), clamp1000(y)]
}

export function normalizeComputerActionCoordinates(
  action: ComputerAction,
  coordinateSpace: CoordinateSpace,
  viewport?: ComputerViewportDimensions | null
): ComputerAction {
  switch (action.type) {
    case 'left_click':
    case 'right_click':
    case 'double_click':
    case 'triple_click':
    case 'mouse_move': {
      const [x, y] = normalizePointTo1000(action.x, action.y, coordinateSpace, viewport)
      return { ...action, x, y }
    }
    case 'scroll': {
      const [x, y] = normalizePointTo1000(action.x, action.y, coordinateSpace, viewport)
      return { ...action, x, y }
    }
    case 'left_click_drag': {
      const [start_x, start_y] = normalizePointTo1000(
        action.start_x,
        action.start_y,
        coordinateSpace,
        viewport
      )
      const [end_x, end_y] = normalizePointTo1000(
        action.end_x,
        action.end_y,
        coordinateSpace,
        viewport
      )
      return { type: 'left_click_drag', start_x, start_y, end_x, end_y }
    }
    default:
      return action
  }
}

export function parseUiTarsOutput(text: string): FaraParseResult {
  let thought: string | null = null
  const thoughtMatch = /(?:^|\n)\s*Thought:\s*([\s\S]*?)(?=(?:^|\n)\s*Action:|$)/i.exec(text)
  if (thoughtMatch?.[1]?.trim()) {
    thought = thoughtMatch[1].trim()
  } else {
    thought = extractThought(text)
  }

  const actions: ComputerAction[] = []
  const raw_tool_calls: string[] = []

  const actionSectionMatch = /(?:^|\n)\s*Action:\s*([\s\S]+)$/i.exec(text)
  const source = actionSectionMatch ? actionSectionMatch[1].trim() : text.trim()

  for (const call of extractUiTarsCalls(source)) {
    raw_tool_calls.push(call.raw)
    try {
      actions.push(uiTarsCallToAction(call.name, call.argsText))
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error)
      console.error(`[ui-tars] failed to parse action: ${message}`, call.raw)
      actions.push({ type: 'error', error: 'parse_error', raw: call.raw.slice(0, 500) })
    }
  }

  return { thought, actions, raw_tool_calls }
}

const KNOWN_UI_TARS_VERBS = new Set([
  'click',
  'left_single',
  'left_click',
  'left_double',
  'double_click',
  'right_single',
  'right_click',
  'hover',
  'mouse_move',
  'move',
  'drag',
  'hotkey',
  'key',
  'press_key',
  'type',
  'scroll',
  'wait',
  'finished',
  'finish',
  'call_user',
  'ask_user',
  'open_url',
  'visit_url',
  'goto'
])

function extractUiTarsCalls(source: string): Array<{ name: string; argsText: string; raw: string }> {
  const results: Array<{ name: string; argsText: string; raw: string }> = []
  const verbPattern = /([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g
  let match: RegExpExecArray | null
  while ((match = verbPattern.exec(source)) !== null) {
    const name = match[1].toLowerCase()
    if (!KNOWN_UI_TARS_VERBS.has(name)) continue
    const openParenIndex = match.index + match[0].length - 1
    const closeParenIndex = findMatchingDelimiter(source, openParenIndex, '(', ')')
    if (closeParenIndex < 0) continue
    const argsText = source.slice(openParenIndex + 1, closeParenIndex)
    const raw = source.slice(match.index, closeParenIndex + 1)
    results.push({ name, argsText, raw })
    verbPattern.lastIndex = closeParenIndex + 1
  }
  return results
}

function findMatchingDelimiter(
  text: string,
  openIndex: number,
  openChar: string,
  closeChar: string
): number {
  let depth = 0
  let quote: '"' | "'" | null = null
  for (let i = openIndex; i < text.length; i += 1) {
    const ch = text[i]
    if (quote) {
      if (ch === '\\') {
        i += 1
        continue
      }
      if (ch === quote) quote = null
      continue
    }
    if (ch === '"' || ch === "'") {
      quote = ch
      continue
    }
    if (ch === openChar) depth += 1
    else if (ch === closeChar) {
      depth -= 1
      if (depth === 0) return i
    }
  }
  return -1
}

function parseUiTarsKwargs(argsText: string): Record<string, string> {
  const kwargs: Record<string, string> = {}
  let i = 0
  while (i < argsText.length) {
    while (i < argsText.length && /[\s,]/.test(argsText[i])) i += 1
    if (i >= argsText.length) break
    const keyMatch = /^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*/.exec(argsText.slice(i))
    if (!keyMatch) break
    const key = keyMatch[1]
    i += keyMatch[0].length
    if (i >= argsText.length) {
      kwargs[key] = ''
      break
    }
    const quote = argsText[i]
    if (quote === "'" || quote === '"') {
      i += 1
      let val = ''
      while (i < argsText.length) {
        const ch = argsText[i]
        if (ch === '\\' && i + 1 < argsText.length) {
          const next = argsText[i + 1]
          if (next === 'n') val += '\n'
          else if (next === 't') val += '\t'
          else if (next === 'r') val += '\r'
          else val += next
          i += 2
          continue
        }
        if (ch === quote) {
          i += 1
          break
        }
        val += ch
        i += 1
      }
      kwargs[key] = val
    } else {
      let end = i
      let depth = 0
      while (end < argsText.length) {
        const ch = argsText[end]
        if (ch === '(' || ch === '[') depth += 1
        else if (ch === ')' || ch === ']') {
          if (depth === 0) break
          depth -= 1
        } else if (ch === ',' && depth === 0) break
        end += 1
      }
      kwargs[key] = argsText.slice(i, end).trim()
      i = end
    }
  }
  return kwargs
}

function parseUiTarsBox(raw: string | undefined): [number, number] | null {
  if (!raw) return null
  const numbers = raw.match(/-?\d+(?:\.\d+)?/g)?.map(Number) ?? []
  if (numbers.length >= 4) {
    let x = (numbers[0] + numbers[2]) / 2
    let y = (numbers[1] + numbers[3]) / 2
    if (raw.includes('.') && x > 0 && x < 1 && y > 0 && y < 1) {
      x *= 1000
      y *= 1000
    }
    return [x, y]
  }
  if (numbers.length >= 2) {
    let [x, y] = numbers
    if (raw.includes('.') && x > 0 && x < 1 && y > 0 && y < 1) {
      x *= 1000
      y *= 1000
    }
    return [x, y]
  }
  return null
}

function uiTarsCallToAction(name: string, argsText: string): ComputerAction {
  const kwargs = parseUiTarsKwargs(argsText)
  const startPoint =
    parseUiTarsBox(kwargs.start_box ?? kwargs.point ?? kwargs.box ?? kwargs.coordinate) ?? [
      500, 500
    ]
  switch (name) {
    case 'click':
    case 'left_single':
    case 'left_click':
      return { type: 'left_click', x: startPoint[0], y: startPoint[1] }
    case 'left_double':
    case 'double_click':
      return { type: 'double_click', x: startPoint[0], y: startPoint[1] }
    case 'right_single':
    case 'right_click':
      return { type: 'right_click', x: startPoint[0], y: startPoint[1] }
    case 'hover':
    case 'mouse_move':
    case 'move':
      return { type: 'mouse_move', x: startPoint[0], y: startPoint[1] }
    case 'drag': {
      const endPoint = parseUiTarsBox(kwargs.end_box ?? kwargs.target_box) ?? startPoint
      return {
        type: 'left_click_drag',
        start_x: startPoint[0],
        start_y: startPoint[1],
        end_x: endPoint[0],
        end_y: endPoint[1]
      }
    }
    case 'hotkey':
    case 'key':
    case 'press_key': {
      const rawKey = kwargs.key ?? kwargs.hotkey ?? kwargs.keys ?? argsText.replace(/['"]/g, '')
      return { type: 'keypress', keys: splitKeyChord(rawKey) }
    }
    case 'type':
      return { type: 'type', text: kwargs.content ?? kwargs.text ?? '' }
    case 'scroll': {
      const direction = (kwargs.direction ?? 'down').toLowerCase()
      const delta_x = direction === 'right' ? 400 : direction === 'left' ? -400 : 0
      const delta_y = direction === 'up' ? -400 : direction === 'down' ? 400 : 0
      return {
        type: 'scroll',
        x: startPoint[0],
        y: startPoint[1],
        delta_x,
        delta_y
      }
    }
    case 'wait': {
      const seconds = kwargs.time ?? kwargs.seconds
      const ms = kwargs.milliseconds ?? kwargs.ms
      const milliseconds = ms
        ? Number(ms) || 1000
        : seconds
          ? (Number(seconds) || 1) * 1000
          : 1000
      return { type: 'wait', milliseconds }
    }
    case 'finished':
    case 'finish':
      return { type: 'terminate', response: kwargs.content ?? kwargs.response ?? null }
    case 'call_user':
    case 'ask_user':
      return {
        type: 'ask_user',
        question: kwargs.content ?? kwargs.question ?? 'Need your input.'
      }
    case 'open_url':
    case 'visit_url':
    case 'goto':
      return { type: 'visit_url', url: kwargs.url ?? 'about:blank' }
    default:
      throw new Error(`unsupported UI-TARS action: ${name}`)
  }
}

function extractMarkdownJsonBlocks(text: string): string[] {
  const blocks: string[] = []
  const fenceRegex = /```(?:json|tool_call|javascript|js)?\s*\n?([\s\S]*?)```/gi
  let match: RegExpExecArray | null
  while ((match = fenceRegex.exec(text)) !== null) {
    const body = match[1].trim()
    if (body.startsWith('{') || body.startsWith('[')) blocks.push(body)
  }
  return blocks
}

function extractBalancedJsonObjects(text: string): string[] {
  const objects: string[] = []
  let i = 0
  while (i < text.length) {
    if (text[i] !== '{') {
      i += 1
      continue
    }
    const end = findMatchingDelimiter(text, i, '{', '}')
    if (end < 0) {
      i += 1
      continue
    }
    objects.push(text.slice(i, end + 1))
    i = end + 1
  }
  return objects
}

function looksLikeActionPayload(candidate: unknown): candidate is Record<string, unknown> {
  if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) return false
  const obj = candidate as Record<string, unknown>
  return (
    obj.name === 'computer_use' ||
    typeof obj.action === 'string' ||
    typeof obj.type === 'string' ||
    (typeof obj.function === 'object' && obj.function !== null)
  )
}

export function parsePromptedModelOutput(text: string): FaraParseResult {
  const xmlBlocks = extractToolCallBlocks(text)
  if (xmlBlocks.length > 0) {
    return parseFaraOutput(text)
  }

  if (text.includes('<function=')) {
    const fnArgs = parseXmlFunctionCall(text)
    if (fnArgs) {
      const thought = extractThought(text)
      try {
        return {
          thought,
          actions: [actionFromArgs(fnArgs)],
          raw_tool_calls: [JSON.stringify(fnArgs)]
        }
      } catch {
        return {
          thought,
          actions: [{ type: 'error', error: 'parse_error', raw: text.slice(0, 500) }],
          raw_tool_calls: [text.slice(0, 500)]
        }
      }
    }
  }

  const fenced = extractMarkdownJsonBlocks(text)
  if (fenced.length > 0) {
    const actions: ComputerAction[] = []
    for (const block of fenced) {
      try {
        const parsed = JSON.parse(block) as unknown
        const items = Array.isArray(parsed) ? parsed : [parsed]
        for (const item of items) {
          actions.push(parseToolCallJson(JSON.stringify(item)))
        }
      } catch {
        actions.push({ type: 'error', error: 'parse_error', raw: block.slice(0, 500) })
      }
    }
    const fenceIdx = text.indexOf('```')
    const prefix = fenceIdx > 0 ? text.slice(0, fenceIdx).replace(/<\/?think>/gi, '').trim() : ''
    return {
      thought: extractThought(text) ?? (prefix || null),
      actions,
      raw_tool_calls: fenced
    }
  }

  if (/(?:^|\n)\s*Action:\s*[a-z_]+\s*\(/i.test(text)) {
    const uiTars = parseUiTarsOutput(text)
    if (uiTars.actions.length > 0) return uiTars
  }

  const trimmed = text.trim()
  if (trimmed.startsWith('{')) {
    return parseFaraOutput(text)
  }

  const embeddedObjects = extractBalancedJsonObjects(text)
  const actions: ComputerAction[] = []
  const raw_tool_calls: string[] = []
  let firstJsonIndex = -1
  for (const candidateText of embeddedObjects) {
    try {
      const parsed = JSON.parse(candidateText) as unknown
      if (!looksLikeActionPayload(parsed)) continue
      if (firstJsonIndex < 0) firstJsonIndex = text.indexOf(candidateText)
      actions.push(parseToolCallJson(candidateText))
      raw_tool_calls.push(candidateText)
    } catch {
      // Ignore non-JSON prose fragments.
    }
  }
  if (actions.length > 0) {
    const prosePrefix =
      firstJsonIndex > 0 ? text.slice(0, firstJsonIndex).replace(/<\/?think>/gi, '').trim() : ''
    return {
      thought: extractThought(text) ?? (prosePrefix || null),
      actions,
      raw_tool_calls
    }
  }

  return { thought: extractThought(text), actions: [], raw_tool_calls: [] }
}

function stripToolArtifactsFromThought(text: string): string {
  return text
    .replace(/<tool_call>[\s\S]*?(?:<\/tool_call>|$)/gi, '')
    .replace(/<function=[a-zA-Z0-9_.-]+>[\s\S]*?(?:<\/function>|$)/gi, '')
    .replace(/(?:^|\n)\s*Action:\s*[\s\S]*$/i, '')
    .replace(/<\/?think>/gi, '')
    .trim()
}

function serializeActionAsToolCall(action: ComputerAction): string {
  switch (action.type) {
    case 'left_click':
    case 'right_click':
    case 'double_click':
    case 'triple_click':
    case 'mouse_move':
      return `<tool_call>${JSON.stringify({
        name: 'computer_use',
        arguments: { action: action.type, coordinate: [Math.round(action.x), Math.round(action.y)] }
      })}</tool_call>`
    case 'left_click_drag':
      return `<tool_call>${JSON.stringify({
        name: 'computer_use',
        arguments: {
          action: 'left_click_drag',
          start_coordinate: [Math.round(action.start_x), Math.round(action.start_y)],
          coordinate: [Math.round(action.end_x), Math.round(action.end_y)]
        }
      })}</tool_call>`
    case 'keypress':
      return `<tool_call>${JSON.stringify({
        name: 'computer_use',
        arguments: { action: 'key', keys: action.keys }
      })}</tool_call>`
    case 'memorize':
      return `<tool_call>${JSON.stringify({
        name: 'computer_use',
        arguments: { action: 'pause_and_memorize_fact', fact: action.fact }
      })}</tool_call>`
    case 'ask_user':
      return `<tool_call>${JSON.stringify({
        name: 'computer_use',
        arguments: { action: 'ask_user_question', question: action.question }
      })}</tool_call>`
    default: {
      const { type, ...rest } = action
      return `<tool_call>${JSON.stringify({
        name: 'computer_use',
        arguments: { action: type, ...rest }
      })}</tool_call>`
    }
  }
}

function parseTextForDialect(text: string, dialect: ComputerActionDialect): FaraParseResult {
  if (!text.trim()) return { thought: null, actions: [], raw_tool_calls: [] }
  if (dialect === 'ui-tars') {
    const uiTars = parseUiTarsOutput(text)
    if (uiTars.actions.length > 0) return uiTars
    return parsePromptedModelOutput(text)
  }
  if (dialect === 'fara-xml') {
    const fara = parseFaraOutput(text)
    if (fara.actions.length > 0) return fara
    return parsePromptedModelOutput(text)
  }
  return parsePromptedModelOutput(text)
}

export function parseComputerCompletion(
  completion: ComputerCompletionInput,
  options: {
    dialect?: ComputerActionDialect
    coordinateSpace?: CoordinateSpace
    viewport?: ComputerViewportDimensions | null
  } = {}
): ParsedComputerCompletion {
  const dialect = options.dialect ?? 'fara-xml'
  const coordinateSpace = options.coordinateSpace ?? 'norm1000'
  const viewport = options.viewport ?? null
  const responseText = completion.responseText ?? ''
  const reasoningText = completion.reasoningText ?? ''
  const clientToolCalls = completion.clientToolCalls ?? []

  if (clientToolCalls.length > 0) {
    const rawActions: ComputerAction[] = []
    for (const call of clientToolCalls) {
      try {
        const parsedArgs = JSON.parse(call.arguments || '{}') as Record<string, unknown>
        const payload =
          call.name === 'computer_use'
            ? { name: 'computer_use', arguments: parsedArgs }
            : { action: call.name, ...parsedArgs }
        rawActions.push(parseToolCallJson(JSON.stringify(payload)))
      } catch {
        rawActions.push({
          type: 'error',
          error: 'parse_error',
          raw: (call.arguments || call.name).slice(0, 500)
        })
      }
    }
    const actions = rawActions.map((action) =>
      normalizeComputerActionCoordinates(action, coordinateSpace, viewport)
    )
    const thought =
      stripToolArtifactsFromThought(reasoningText) ||
      stripToolArtifactsFromThought(responseText) ||
      null
    const assistantContent =
      responseText.trim() ||
      actions
        .filter((action) => action.type !== 'error')
        .map(serializeActionAsToolCall)
        .join('\n') ||
      clientToolCalls
        .map((call) => `<tool_call>${JSON.stringify({ name: call.name, arguments: call.arguments })}</tool_call>`)
        .join('\n')
    return { thought, actions, assistantContent }
  }

  let parsed = parseTextForDialect(responseText, dialect)
  let usedReasoningForActions = false

  if (parsed.actions.length === 0 && reasoningText.trim()) {
    const combined = responseText.trim() ? `${reasoningText}\n${responseText}` : reasoningText
    const fallback = parseTextForDialect(combined, dialect)
    if (fallback.actions.length > 0) {
      parsed = fallback
      usedReasoningForActions = true
    }
  }

  const actions = parsed.actions.map((action) =>
    normalizeComputerActionCoordinates(action, coordinateSpace, viewport)
  )
  const cleanReasoning = stripToolArtifactsFromThought(reasoningText)
  const thought = cleanReasoning || parsed.thought || null
  const assistantContent =
    responseText.trim() ||
    (usedReasoningForActions && parsed.raw_tool_calls.length > 0
      ? parsed.raw_tool_calls.map((raw) => `<tool_call>${raw}</tool_call>`).join('\n')
      : actions
          .filter((action) => action.type !== 'error')
          .map(serializeActionAsToolCall)
          .join('\n')) ||
    responseText

  return { thought, actions, assistantContent }
}

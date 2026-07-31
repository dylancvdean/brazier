/**
 * Compatibility handling for local models with inconsistent tool calling.
 *
 * Small local models emit tool arguments as JSON in a string, wrap them in an
 * extra object, invent parameter names, or fence them in Markdown. Repairing
 * that here keeps the tool schemas honest — the daemon still validates and
 * still refuses anything it does not understand.
 */

import type { AgentModelCapabilities, AgentToolDefinition } from './types'

/** Model families that follow tool schemas reliably enough for parallel calls. */
const STRONG_TOOL_FAMILIES = [
  'gpt-oss',
  'qwen3',
  'qwen2.5-coder',
  'devstral',
  'glm-4',
  'kimi',
  'deepseek-v3',
  'deepseek-r1',
  'mistral-large',
  'codestral'
]

/** Families that stream reasoning as a separate channel. */
const REASONING_FAMILIES = ['gpt-oss', 'deepseek-r1', 'qwen3', 'magistral', 'glm-4']

/**
 * Conservative capability guess from a Brazier model id. Wrong guesses cost a
 * repair pass or an extra turn, never correctness: the daemon validates every
 * call regardless.
 */
export function inferModelCapabilities(modelId: string): AgentModelCapabilities {
  const id = modelId.toLowerCase()
  const strong = STRONG_TOOL_FAMILIES.some((family) => id.includes(family))
  const reasoning = REASONING_FAMILIES.some((family) => id.includes(family))
  const harmony = id.includes('gpt-oss') || id.includes('gpt_oss') || id.includes('gptoss')
  return {
    nativeToolCalling: true,
    parallelToolCalling: strong,
    supportsReasoningStream: reasoning,
    harmony,
    reliableJson: strong,
    // One call per turn keeps a weak model's mistakes cheap to recover from.
    maxToolsPerTurn: strong ? undefined : 1
  }
}

/** Parameter names models commonly reach for instead of the real one. */
const ARGUMENT_ALIASES: Record<string, string[]> = {
  path: ['file_path', 'filepath', 'filename', 'file', 'dir', 'directory', 'target', 'file_name'],
  content: ['contents', 'text', 'body', 'data', 'new_content', 'file_content'],
  command: ['cmd', 'shell', 'script', 'command_line', 'bash'],
  query: ['pattern', 'search', 'term', 'needle', 'text_to_find'],
  old_string: ['old', 'old_text', 'search_string', 'find'],
  new_string: ['new', 'new_text', 'replace_string', 'replacement'],
  from: ['source', 'src', 'old_path'],
  to: ['destination', 'dest', 'new_path'],
  process_id: ['pid', 'id', 'processId'],
  recursive: ['force'],
  reason: ['why', 'justification', 'explanation']
}

/** Wrapper keys models add around the real argument object. */
const WRAPPER_KEYS = ['input', 'arguments', 'args', 'parameters', 'params', 'tool_input', 'kwargs']

/** Strip Markdown fences a model wrapped around JSON. */
function stripFences(text: string): string {
  const trimmed = text.trim()
  if (!trimmed.startsWith('```')) return trimmed
  return trimmed
    .replace(/^```[a-zA-Z]*\s*/, '')
    .replace(/```$/, '')
    .trim()
}

/** Parse a JSON object out of text, tolerating fences and trailing prose. */
export function parseLooseJson(text: string): unknown {
  const cleaned = stripFences(text)
  try {
    return JSON.parse(cleaned)
  } catch {
    // Fall back to the outermost {...} span, which survives trailing commentary.
    const start = cleaned.indexOf('{')
    const end = cleaned.lastIndexOf('}')
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(cleaned.slice(start, end + 1))
      } catch {
        return undefined
      }
    }
    return undefined
  }
}

type SchemaShape = {
  properties: Record<string, { type?: string; enum?: unknown[] }>
  required: string[]
  additionalProperties: boolean
}

function schemaShape(schema: Record<string, unknown>): SchemaShape {
  const properties = (schema.properties as SchemaShape['properties'] | undefined) ?? {}
  const required = Array.isArray(schema.required) ? (schema.required as string[]) : []
  return {
    properties,
    required,
    additionalProperties: schema.additionalProperties !== false
  }
}

/**
 * Coerce one value toward its declared type. Providers hand back strings for
 * numbers and booleans often enough that this is worth doing before validation.
 */
function coerceValue(value: unknown, declared?: { type?: string; enum?: unknown[] }): unknown {
  if (declared?.enum && typeof value === 'string') {
    const match = declared.enum.find(
      (candidate) =>
        typeof candidate === 'string' && candidate.toLowerCase() === value.trim().toLowerCase()
    )
    if (match !== undefined) return match
  }
  switch (declared?.type) {
    case 'boolean': {
      if (typeof value === 'boolean') return value
      if (value === 'true' || value === 'True' || value === 1) return true
      if (value === 'false' || value === 'False' || value === 0) return false
      return value
    }
    case 'integer':
    case 'number': {
      if (typeof value === 'number') return value
      if (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value))) {
        return Number(value)
      }
      return value
    }
    case 'string': {
      if (typeof value === 'string') return value
      if (typeof value === 'number' || typeof value === 'boolean') return String(value)
      return value
    }
    case 'array': {
      if (Array.isArray(value)) return value
      // A single item where a list was asked for is a common near-miss.
      if (value !== undefined && value !== null) return [value]
      return value
    }
    default:
      return value
  }
}

/**
 * Bring model-supplied arguments as close to the schema as can be done without
 * guessing at intent. Unrecognized keys are dropped only when the schema
 * forbids extras, and required keys are never invented.
 */
export function repairToolArguments(
  raw: unknown,
  tool: Pick<AgentToolDefinition, 'inputSchema'>
): Record<string, unknown> {
  let value: unknown = raw

  // A JSON string, possibly double-encoded or fenced.
  for (let attempt = 0; attempt < 2 && typeof value === 'string'; attempt += 1) {
    const parsed = parseLooseJson(value)
    if (parsed === undefined) break
    value = parsed
  }

  const shape = schemaShape(tool.inputSchema)
  const declaredKeys = Object.keys(shape.properties)

  // A wrapper object around the real arguments, e.g. `{ input: { path } }`.
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const entries = Object.entries(value as Record<string, unknown>)
    if (entries.length === 1) {
      const [key, inner] = entries[0]!
      const wrapped = WRAPPER_KEYS.includes(key) && !declaredKeys.includes(key)
      if (wrapped && inner && typeof inner === 'object' && !Array.isArray(inner)) {
        value = inner
      } else if (wrapped && typeof inner === 'string') {
        const parsed = parseLooseJson(inner)
        if (parsed && typeof parsed === 'object') value = parsed
      }
    }
  }

  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    // Nothing usable. A single required string argument is the one case where
    // a bare value has an unambiguous home.
    if (typeof raw === 'string' && shape.required.length === 1) {
      const key = shape.required[0]!
      if (shape.properties[key]?.type === 'string') {
        return { [key]: raw }
      }
    }
    return {}
  }

  const source = value as Record<string, unknown>
  const repaired: Record<string, unknown> = {}

  for (const [key, entry] of Object.entries(source)) {
    if (declaredKeys.includes(key)) {
      repaired[key] = coerceValue(entry, shape.properties[key])
    }
  }

  // Map aliases onto declared names that are still missing.
  for (const [canonical, aliases] of Object.entries(ARGUMENT_ALIASES)) {
    if (!declaredKeys.includes(canonical) || repaired[canonical] !== undefined) continue
    for (const alias of aliases) {
      if (source[alias] !== undefined) {
        repaired[canonical] = coerceValue(source[alias], shape.properties[canonical])
        break
      }
    }
  }

  // Keep unknown keys only when the schema tolerates them.
  if (shape.additionalProperties) {
    for (const [key, entry] of Object.entries(source)) {
      if (repaired[key] === undefined) repaired[key] = entry
    }
  }

  return repaired
}

/** Human-readable note for the timeline when repair changed something. */
export function describeRepair(
  before: unknown,
  after: Record<string, unknown>
): string | undefined {
  const beforeJson = typeof before === 'string' ? before : JSON.stringify(before ?? {})
  const afterJson = JSON.stringify(after)
  if (beforeJson === afterJson) return undefined
  return `Arguments were repaired to match the tool schema.`
}

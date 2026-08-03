/**
 * Structured Oh My Pi settings surface.
 *
 * A curated subset of OMP's `config.yml` keys, declared once here and used by
 * both the Manage → Agent form (renderer) and the sidecar config writer (worker).
 * Only keys the user explicitly set are emitted, so the freeform YAML textarea
 * remains the escape hatch for everything else and OMP defaults are never
 * clobbered. New OMP releases add a field to `OMP_SETTINGS_FIELDS`; nothing else
 * in the app needs to change.
 */

export type OmpSettingType = 'enum' | 'boolean' | 'number'
export type OmpSettingOption = { value: string; label: string }

export type OmpSettingField = {
  /** Dotted config.yml path, e.g. `retry.enabled`. */
  path: string
  /** Form section it renders under. */
  group: string
  label: string
  description: string
  type: OmpSettingType
  options?: OmpSettingOption[]
  /** Number bounds, for the form's input. */
  min?: number
  max?: number
  step?: number
  placeholder?: string
}

const thinkingLevels: OmpSettingOption[] = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'auto'].map(
  (value) => ({ value, label: value })
)

export const OMP_SETTINGS_FIELDS: OmpSettingField[] = [
  // Thinking
  {
    path: 'defaultThinkingLevel',
    group: 'Thinking',
    label: 'Default thinking level',
    description: 'Reasoning effort for ordinary turns.',
    type: 'enum',
    options: thinkingLevels
  },
  // Sampling
  {
    path: 'temperature',
    group: 'Sampling',
    label: 'Temperature',
    description: 'Sampling temperature; -1 uses the provider default.',
    type: 'number',
    min: -1,
    max: 2,
    step: 0.1,
    placeholder: '-1 = provider default'
  },
  {
    path: 'topP',
    group: 'Sampling',
    label: 'Top P',
    description: 'Nucleus sampling cutoff; -1 uses the provider default.',
    type: 'number',
    min: -1,
    max: 1,
    step: 0.05,
    placeholder: '-1 = provider default'
  },
  {
    path: 'topK',
    group: 'Sampling',
    label: 'Top K',
    description: 'Top-K sampling; -1 uses the provider default.',
    type: 'number',
    min: -1,
    max: 100,
    step: 1,
    placeholder: '-1 = provider default'
  },
  {
    path: 'textVerbosity',
    group: 'Sampling',
    label: 'Text verbosity',
    description: 'How verbose assistant text should be.',
    type: 'enum',
    options: [
      { value: 'low', label: 'low' },
      { value: 'medium', label: 'medium' },
      { value: 'high', label: 'high' }
    ]
  },
  // Retry
  {
    path: 'retry.enabled',
    group: 'Retry',
    label: 'Retry transient errors',
    description: 'Retry provider errors (429s, timeouts, outages).',
    type: 'boolean'
  },
  {
    path: 'retry.maxRetries',
    group: 'Retry',
    label: 'Max retries',
    description: 'Retries per request before giving up.',
    type: 'number',
    min: 0,
    max: 50,
    step: 1
  },
  {
    path: 'retry.modelFallback',
    group: 'Retry',
    label: 'Model fallback',
    description: 'Fall back to another model when the active one is unavailable.',
    type: 'boolean'
  },
  // Compaction
  {
    path: 'compaction.enabled',
    group: 'Compaction',
    label: 'Auto-compaction',
    description: 'Automatically compact context when it grows too large.',
    type: 'boolean'
  },
  {
    path: 'compaction.thresholdPercent',
    group: 'Compaction',
    label: 'Trigger threshold (%)',
    description: 'Percent-of-context trigger; -1 uses the reserve-based default.',
    type: 'number',
    min: -1,
    max: 100,
    step: 1,
    placeholder: '-1 = default'
  },
  // Tools
  {
    path: 'tools.approvalMode',
    group: 'Tools',
    label: 'Approval mode',
    description: 'The prompt tier for OMP built-in tools.',
    type: 'enum',
    options: [
      { value: 'always-ask', label: 'always-ask' },
      { value: 'write', label: 'write' },
      { value: 'yolo', label: 'yolo' }
    ]
  },
  {
    path: 'tools.maxTimeout',
    group: 'Tools',
    label: 'Tool timeout (s)',
    description: 'Max tool runtime in seconds; 0 = no cap.',
    type: 'number',
    min: 0,
    max: 3600,
    step: 10,
    placeholder: '0 = no cap'
  },
  {
    path: 'bash.enabled',
    group: 'Tools',
    label: 'Bash tool',
    description: 'Enable the embedded shell.',
    type: 'boolean'
  },
  {
    path: 'browser.enabled',
    group: 'Tools',
    label: 'Browser tool',
    description: 'Enable the browser/Chromium tool.',
    type: 'boolean'
  },
  {
    path: 'web_search.enabled',
    group: 'Tools',
    label: 'Web search tool',
    description: 'Enable the web_search tool.',
    type: 'boolean'
  },
  {
    path: 'lsp.enabled',
    group: 'Tools',
    label: 'LSP integration',
    description: 'Language-server powered edits and diagnostics.',
    type: 'boolean'
  },
  {
    path: 'astEdit.enabled',
    group: 'Tools',
    label: 'AST edit tool',
    description: 'Structural code rewrites via ast-grep.',
    type: 'boolean'
  },
  {
    path: 'astGrep.enabled',
    group: 'Tools',
    label: 'AST grep tool',
    description: 'Structural code search.',
    type: 'boolean'
  },
  {
    path: 'inspect_image.mode',
    group: 'Tools',
    label: 'Image inspection',
    description: 'When the inspect_image tool is exposed.',
    type: 'enum',
    options: [
      { value: 'auto', label: 'auto' },
      { value: 'on', label: 'on' },
      { value: 'off', label: 'off' }
    ]
  },
  // Editing
  {
    path: 'edit.mode',
    group: 'Editing',
    label: 'Edit format',
    description: 'How the edit tool applies changes.',
    type: 'enum',
    options: [
      { value: 'apply_patch', label: 'apply_patch' },
      { value: 'hashline', label: 'hashline' },
      { value: 'patch', label: 'patch' },
      { value: 'replace', label: 'replace' }
    ]
  },
  {
    path: 'edit.fuzzyMatch',
    group: 'Editing',
    label: 'Fuzzy edit anchors',
    description: 'Allow fuzzy matching of edit anchors.',
    type: 'boolean'
  },
  // Advisor
  {
    path: 'advisor.enabled',
    group: 'Advisor',
    label: 'Advisor runtime',
    description: 'A second model reviews turns and injects notes (uses the advisor role).',
    type: 'boolean'
  },
  // Memory
  {
    path: 'memory.backend',
    group: 'Memory',
    label: 'Memory backend',
    description: 'Persistent memory engine for retain/recall/reflect.',
    type: 'enum',
    options: [
      { value: 'off', label: 'off' },
      { value: 'local', label: 'local' },
      { value: 'hindsight', label: 'hindsight' },
      { value: 'mnemopi', label: 'mnemopi' }
    ]
  }
]

/** A partial, flat settings record keyed by dotted config path. */
export type OmpSettings = Record<string, string | number | boolean>

const FIELD_BY_PATH = new Map(OMP_SETTINGS_FIELDS.map((field) => [field.path, field]))

/**
 * Keep only known keys and coerce values to their declared types, so a stale or
 * hand-edited preference can never emit malformed YAML.
 */
export function sanitizeOmpSettings(value: unknown): OmpSettings {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const out: OmpSettings = {}
  for (const [rawPath, raw] of Object.entries(value as Record<string, unknown>)) {
    const field = FIELD_BY_PATH.get(rawPath)
    if (!field) continue
    if (field.type === 'boolean' && typeof raw === 'boolean') out[rawPath] = raw
    else if (field.type === 'number' && typeof raw === 'number' && Number.isFinite(raw)) out[rawPath] = raw
    else if (field.type === 'enum' && typeof raw === 'string') {
      const allowed = field.options?.some((option) => option.value === raw)
      if (allowed) out[rawPath] = raw
    }
  }
  return out
}

function formatYamlValue(value: string | number | boolean): string {
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') return String(value)
  return value
}

/** Emit the explicitly-set settings as YAML, grouped by their top-level key. */
export function ompSettingsYaml(settings: OmpSettings): string {
  const groups = new Map<string, Array<[string, string | number | boolean]>>()
  const scalars = new Map<string, string | number | boolean>()
  for (const field of OMP_SETTINGS_FIELDS) {
    const value = settings[field.path]
    if (value === undefined) continue
    const segments = field.path.split('.')
    if (segments.length === 1) scalars.set(segments[0], value)
    else {
      const list = groups.get(segments[0]) ?? []
      list.push([segments.slice(1).join('.'), value])
      groups.set(segments[0], list)
    }
  }
  const lines: string[] = []
  const emit = (key: string): void => {
    const group = groups.get(key)
    const scalar = scalars.get(key)
    if (group) {
      lines.push(`${key}:`)
      for (const [sub, value] of group) lines.push(`  ${sub}: ${formatYamlValue(value)}`)
    } else if (scalar !== undefined) {
      lines.push(`${key}: ${formatYamlValue(scalar)}`)
    }
  }
  const emitted = new Set<string>()
  for (const field of OMP_SETTINGS_FIELDS) {
    const key = field.path.split('.')[0]
    if (emitted.has(key)) continue
    emitted.add(key)
    emit(key)
  }
  return lines.join('\n')
}

/** Remove a top-level YAML key and its indented block, leaving the rest intact. */
export function stripTopLevelKey(yaml: string, key: string): string {
  const prefix = `${key}:`
  const lines = yaml.split('\n')
  const out: string[] = []
  let inBlock = false
  for (const line of lines) {
    if (!inBlock && !line.startsWith(' ') && line.trim() === prefix) {
      inBlock = true
      continue
    }
    if (inBlock) {
      if (!line.startsWith(' ') && line.trim() !== '') {
        inBlock = false
        out.push(line)
      }
      continue
    }
    out.push(line)
  }
  return out.join('\n')
}

/**
 * Merge `modelRoles` into a config YAML string. The block replaces any existing
 * top-level `modelRoles:` so the profile editor is authoritative without the
 * runtime needing a YAML parser.
 */
export function configYamlWithModelRoles(
  configYaml: string | undefined,
  modelRoles: Record<string, string>
): string {
  const entries = Object.entries(modelRoles).filter(([, modelId]) => modelId.trim().length > 0)
  if (entries.length === 0) return configYaml?.trim() ?? ''
  const base = stripTopLevelKey(configYaml ?? '', 'modelRoles').trim()
  const block = ['modelRoles:', ...entries.map(([role, modelId]) => `  ${role}: brazier/${modelId}`)]
  return base ? `${base}\n${block.join('\n')}\n` : `${block.join('\n')}\n`
}

/**
 * Merge the structured settings into a config YAML string. Top-level keys the
 * settings own are stripped first so the form is authoritative for them, while
 * every other key in the freeform YAML is preserved.
 */
export function configYamlWithSettings(
  configYaml: string | undefined,
  settings: OmpSettings
): string {
  const yaml = ompSettingsYaml(settings)
  if (!yaml) return configYaml?.trim() ?? ''
  const owned = new Set<string>()
  for (const field of OMP_SETTINGS_FIELDS) {
    if (settings[field.path] !== undefined) owned.add(field.path.split('.')[0])
  }
  let base = configYaml?.trim() ?? ''
  for (const key of owned) base = stripTopLevelKey(base, key).trim()
  return base ? `${base}\n${yaml}\n` : `${yaml}\n`
}

/** The top-level keys a settings record currently overrides (for the form's reset). */
export function settingTopLevelKeys(settings: OmpSettings): string[] {
  return [...new Set(Object.keys(settings).map((path) => path.split('.')[0]))]
}

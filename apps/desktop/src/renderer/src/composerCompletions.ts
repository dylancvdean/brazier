/**
 * Composer completions for Chat and shell modes (Agent/Computer).
 *
 * Suggestions are matched against the trailing word of the draft. A leading
 * slash is part of the query so slash-prefixed commands populate and filter
 * as the user types. An empty trailing word never surfaces the popup, so a
 * focused-but-empty composer does not sit an overlay over the transcript.
 */

export type ComposerSuggestion = { value: string; description: string }

const TRAILING_COMMAND = /(?:^|\s)(\/?[a-z0-9_-]*)$/i

export const CHAT_SLASH_SUGGESTIONS: ComposerSuggestion[] = [
  { value: '/new', description: 'Start a new chat' },
  { value: '/incognito', description: 'Toggle incognito chat' },
  { value: '/model', description: 'Choose a model' },
  { value: '/settings', description: 'Open inference settings' },
  { value: '/unload', description: 'Unload resident model' },
  { value: '/export', description: 'Export conversation as JSON' },
  { value: '/dream', description: 'Consolidate memories now' }
]

export const AGENT_SLASH_SUGGESTIONS: ComposerSuggestion[] = [
  { value: '/compact', description: 'Summarize earlier turns to free context' },
  { value: '/new', description: 'Start a new agent task' },
  { value: '/worktree', description: 'Toggle isolated git worktree' },
  { value: '/prompt', description: 'Edit workspace system prompt' },
  { value: '/ask', description: 'Set permissions to Ask first' },
  { value: '/sandbox', description: 'Set permissions to Sandbox only' },
  { value: '/skip', description: 'Set permissions to Skip permissions' }
]

export function isExactSlashCommand(
  input: string,
  commands: readonly ComposerSuggestion[]
): boolean {
  const normalized = input.trim().toLowerCase()
  if (!normalized.startsWith('/')) return false
  return commands.some((entry) => entry.value.toLowerCase() === normalized)
}

/** The lowercased trailing command word of the draft, including a leading `/`. */
export function trailingCommand(draft: string): string {
  const match = draft.match(TRAILING_COMMAND)
  return (match?.[1] ?? '').toLowerCase()
}

/** Suggestions whose value starts with the draft's trailing command word. */
export function composerCompletionsFor(
  draft: string,
  suggestions: ComposerSuggestion[]
): ComposerSuggestion[] {
  const query = trailingCommand(draft)
  if (query.length === 0) return []
  return suggestions.filter((entry) => entry.value.toLowerCase().startsWith(query))
}

/** Replace the trailing command word with a chosen completion, keeping spacing. */
export function replaceTrailingCommand(draft: string, value: string): string {
  return draft.replace(
    TRAILING_COMMAND,
    (word) => `${word.startsWith(' ') ? ' ' : ''}${value} `
  )
}

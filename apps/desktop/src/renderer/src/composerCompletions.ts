/**
 * Composer completions for shell modes (Agent/Computer).
 *
 * Suggestions are matched against the trailing word of the draft. A leading
 * slash is part of the query so slash-prefixed commands populate and filter
 * as the user types. An empty trailing word never surfaces the popup, so a
 * focused-but-empty composer does not sit an overlay over the transcript.
 */

export type ComposerSuggestion = { value: string; description: string }

const TRAILING_COMMAND = /(?:^|\s)(\/?[a-z]*)$/i

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

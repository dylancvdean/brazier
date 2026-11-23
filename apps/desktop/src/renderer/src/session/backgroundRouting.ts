/**
 * Local routing for a finalized voice transcript.
 *
 * PersonaPlex has already heard the audio and can answer immediately. This
 * decides whether the same words also need the slower authoritative background
 * path. It is deliberately lexical and local: routing must not add another
 * model call before the model call it is trying to avoid.
 */

export type VoiceBackgroundRouting = 'always' | 'auto' | 'explicit'

export const VOICE_BACKGROUND_ROUTING_OPTIONS: ReadonlyArray<{
  value: VoiceBackgroundRouting
  label: string
  detail: string
}> = [
  {
    value: 'auto',
    label: 'Auto',
    detail:
      'Small talk stays with the voice model. Requests for work, tools, or current facts go to the background model.'
  },
  {
    value: 'always',
    label: 'Always',
    detail:
      'Every turn goes to the background model. Slowest.'
  },
  {
    value: 'explicit',
    label: 'When asked',
    detail:
      'The background model runs only when you ask it to check, search, run, or edit something.'
  }
]

const LIGHTWEIGHT = [
  /^(?:hi|hello|hey|hiya|yo)(?:\s+there)?[.!?]*$/i,
  /^(?:thanks|thank you|cheers|appreciate it)[.!?]*$/i,
  /^(?:ok(?:ay)?|cool|great|nice|perfect|awesome|sick|got it|sounds good|right|exactly)[.!?]*$/i,
  /^(?:yes|yeah|yep|no|nope|maybe|sure)[.!?]*$/i,
  /^(?:how are you|how's it going|what's up|tell me a joke)[.!?]*$/i
]

/**
 * Signals that the answer depends on state PersonaPlex should not invent.
 *
 * This list errs toward background work. The Auto mode's useful fast path is
 * short conversation without one of these words; Explicit mode exposes the
 * opposite end of the experiment without pretending the heuristic is semantic.
 */
const BACKGROUND_CUE =
  /\b(?:check|look\s*up|search|browse|find|verify|inspect|investigate|research|run|execute|build|test|debug|fix|implement|change|edit|add|remove|delete|create|write|read|open|show|list|summarize|use\s+(?:the\s+)?(?:agent|tool|terminal)|file|folder|directory|repo(?:sitory)?|code|branch|commit|pull request|issue|test suite|terminal|command|workspace|latest|current|today|news|weather|time)\b/i

export function shouldRouteVoiceToBackground(
  text: string,
  policy: VoiceBackgroundRouting,
  options: { taskActive?: boolean } = {}
): boolean {
  if (policy === 'always') return true
  const trimmed = text.trim()
  if (!trimmed) return false
  if (LIGHTWEIGHT.some((pattern) => pattern.test(trimmed))) return false

  const explicitlyNeedsBackground = BACKGROUND_CUE.test(trimmed)
  if (policy === 'explicit') return explicitlyNeedsBackground
  if (explicitlyNeedsBackground) return true

  // A live task gives even a short pronoun-heavy follow-up useful context, but
  // acknowledgements above remain local and cannot accidentally queue work.
  if (options.taskActive) return true

  const words = trimmed.match(/[\p{L}\p{N}'’-]+/gu) ?? []
  return words.length > 8
}

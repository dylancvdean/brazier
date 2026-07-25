/**
 * What the user meant by talking over the assistant.
 *
 * The distinction that matters is "stop talking" versus "stop working": the
 * first must never kill a running agent task, and the second must. Everything
 * else is an ordinary turn, so this classifier deliberately only recognizes
 * explicit phrasings and lets anything ambiguous fall through as a question.
 */

export type UtteranceIntent =
  /** Silence the speech; leave the task alone. */
  | 'stop_speaking'
  /** Abandon the task itself. */
  | 'cancel_task'
  /** The previous request was misunderstood; resubmit corrected. */
  | 'correction'
  /** A question about the thing in progress. */
  | 'follow_up'
  /** Unrelated to the current turn. */
  | 'new_request'

type Rule = { intent: UtteranceIntent; patterns: RegExp[] }

/**
 * Cancellation is checked before stop-speaking, because "stop" appears in both
 * and the explicit object ("cancel that", "stop the build") decides which one
 * the user meant.
 */
const RULES: Rule[] = [
  {
    intent: 'cancel_task',
    patterns: [
      /\b(never\s*mind|nevermind|forget\s+(it|that))\b/,
      /\bcancel\s+(that|it|this|the\s+\w+)\b/,
      /\bcancel\b\s*$/,
      /\b(stop|abort|kill)\s+(the\s+)?(task|run|job|build|command|agent|work)\b/,
      /\bdon'?t\s+(bother|do\s+(it|that))\b/,
      /\bstop\s+working\s+on\s+(that|it|this)\b/
    ]
  },
  {
    intent: 'stop_speaking',
    patterns: [
      /\bstop\s+(talking|speaking|reading)\b/,
      /\b(be\s+quiet|quiet\s+down|shush|hush)\b/,
      /\b(shut\s+up)\b/,
      /\bstop\b\s*$/,
      /\b(hold\s+on|wait|one\s+(second|moment)|hang\s+on)\b\s*$/,
      /\bthat'?s\s+enough\b/,
      /\bskip\s+(that|it|ahead)\b/
    ]
  },
  {
    intent: 'correction',
    patterns: [
      /^\s*(no|nope|not\s+that)\b[,.\s]/,
      /\bi\s+meant\b/,
      /\bactually,?\s+(i|the|it|use|check|look)\b/,
      /\bi\s+said\b/,
      /\bthat'?s\s+(not|the)\s+(right|wrong|correct)\b/,
      // "not the Metal one, the Vulkan one"
      /\bnot\s+the\s+[\w\s]{1,24},\s*the\s+\w+/
    ]
  }
]

/** Words that make an utterance about the work already in flight. */
const FOLLOW_UP_PATTERNS = [
  /\bwhile\s+(that|it|this|you)\b/,
  /\b(how'?s|hows|how\s+is)\s+(that|it|the)\b/,
  /\bis\s+it\s+(done|finished|working|running)\b/,
  /\bany\s+(luck|progress)\b/,
  /\bwhat\s+(about|else)\b/,
  /\band\s+(then|also)\b/
]

/**
 * Classify a final user transcript.
 *
 * `taskActive` only widens the follow-up bucket; it never turns a plain
 * question into a cancellation.
 */
export function classifyUtterance(
  transcript: string,
  { taskActive = false }: { taskActive?: boolean } = {}
): UtteranceIntent {
  const text = transcript.toLowerCase().trim()
  if (!text) return 'follow_up'
  for (const rule of RULES) {
    if (rule.patterns.some((pattern) => pattern.test(text))) return rule.intent
  }
  if (taskActive && FOLLOW_UP_PATTERNS.some((pattern) => pattern.test(text))) return 'follow_up'
  return taskActive ? 'follow_up' : 'new_request'
}

/** Intents that carry no content to submit — they are controls, not turns. */
export function isControlIntent(intent: UtteranceIntent): boolean {
  return intent === 'stop_speaking' || intent === 'cancel_task'
}

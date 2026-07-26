/**
 * Reading a spoken answer as yes, no, or neither.
 *
 * This decides whether an agent is allowed to delete a directory on the
 * strength of a microphone, an energy gate, and a speech recogniser. Every
 * layer of that stack makes mistakes, so the rule is that only an unmistakable
 * answer counts: a short, clean affirmative and nothing else in it. Everything
 * ambiguous is `unclear`, which is not a decision and never becomes one — the
 * approval stays pending, on screen, where a keyboard can settle it.
 *
 * Deliberately not a language model, and deliberately not fuzzy matching. A
 * misheard word must fail closed, and the way to guarantee that is a list short
 * enough to read.
 */

export type Confirmation = 'affirmative' | 'negative' | 'unclear'

/** Whole utterances that mean yes. Not substrings: "yes, but" is not consent. */
const AFFIRMATIVE = new Set([
  'yes',
  'yes please',
  'yeah',
  'yep',
  'yup',
  'sure',
  'ok',
  'okay',
  'go ahead',
  'go for it',
  'do it',
  'please do',
  'approve',
  'approved',
  'allow it',
  'allow',
  'confirm',
  'confirmed',
  'affirmative',
  'that is correct',
  "that's correct",
  'correct'
])

/** Whole utterances that mean no. */
const NEGATIVE = new Set([
  'no',
  'no thanks',
  'no thank you',
  'nope',
  'nah',
  'stop',
  'cancel',
  'deny',
  'denied',
  'refuse',
  'reject',
  'do not',
  "don't",
  'do not do that',
  "don't do that",
  'never mind',
  'nevermind',
  'wait',
  'hold on',
  'negative',
  'abort'
])

/**
 * Strip everything that does not change the meaning: case and sentence
 * punctuation. Commas and semicolons survive, because they separate the parts
 * of an answer given in more than one breath.
 */
function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[.!?:]/g, ' ')
    .replace(/\s*([,;])\s*/g, '$1')
    .replace(/\s+/g, ' ')
    .replace(/[,;]+$/, '')
    .trim()
}

/** Leading politeness that does not qualify the answer. */
const LEADING_FILLER = /^(um|uh|er|well|so|okay so|right|hey|hmm)(?:[,;]\s*|\s+)/

export function classifyConfirmation(text: string): Confirmation {
  let phrase = normalize(text)
  while (LEADING_FILLER.test(phrase)) phrase = phrase.replace(LEADING_FILLER, '')
  // Trailing address — "yes, brazier" — likewise.
  phrase = phrase.replace(/\s+(please|thanks|thank you|brazier)$/, '').trim()
  if (!phrase) return 'unclear'
  // People answer in more than one breath — "no, stop", "yes, go ahead" — and
  // repeating the same answer is still that answer. Every part has to agree;
  // one part that is neither makes the whole thing a sentence again.
  const parts = phrase
    .split(/,|;| and | then /)
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
  if (parts.length === 0) return 'unclear'
  if (parts.every((part) => NEGATIVE.has(part))) return 'negative'
  if (parts.every((part) => AFFIRMATIVE.has(part))) return 'affirmative'
  // Anything else is a sentence, not an answer: it may be a correction, a new
  // request, or a yes with a condition attached, and none of those are consent.
  return 'unclear'
}

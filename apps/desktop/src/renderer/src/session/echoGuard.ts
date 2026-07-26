/**
 * Guard against the assistant hearing itself.
 *
 * PersonaPlex audio comes out of the speakers while the microphone stays open
 * for full-duplex conversation. The browser's echo canceller removes most of
 * it, but anything that leaks through would be transcribed and submitted as a
 * new background turn. A transcript that is mostly the model text we just
 * received is therefore discarded rather than submitted.
 *
 * The comparison is deliberately one-directional: a short fragment of our own
 * speech is echo, while a long utterance that merely quotes it is the user
 * talking and must get through.
 */

function words(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .split(/\s+/)
    .filter(Boolean)
}

/** Above this share of matched words, a transcript counts as echo. */
const ECHO_WORD_OVERLAP = 0.8

/** Transcripts longer than this share of what was spoken are not echo. */
const ECHO_LENGTH_LIMIT = 1.5

export function isEchoOfSpokenText(transcript: string, spoken: string | null): boolean {
  if (!spoken) return false
  const heard = words(transcript)
  const said = words(spoken)
  if (heard.length === 0 || said.length === 0) return false
  // The user saying something longer than the whole answer is not an echo of it.
  if (heard.length > said.length * ECHO_LENGTH_LIMIT) return false

  const remaining = new Map<string, number>()
  for (const word of said) remaining.set(word, (remaining.get(word) ?? 0) + 1)
  let matched = 0
  for (const word of heard) {
    const count = remaining.get(word)
    if (count === undefined || count === 0) continue
    remaining.set(word, count - 1)
    matched += 1
  }
  return matched / heard.length >= ECHO_WORD_OVERLAP
}

/**
 * Whether a transcript is too thin to be a turn.
 *
 * Noise that clears the gate still transcribes to *something* — a syllable, a
 * stray word — and submitting it costs a real turn: the assistant gives up what
 * it was saying to answer that it did not understand. Refusing here is cheaper
 * than answering nothing.
 */
export function isTooThinToSubmit(transcript: string): boolean {
  const cleaned = transcript
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s']/gu, ' ')
    .trim()
  if (cleaned.length < 2) return true
  const words = cleaned.split(/\s+/).filter(Boolean)
  // Length alone cannot be the test: "yes", "no", and "stop" are whole turns,
  // and the segmenter has already required a couple of hundred milliseconds of
  // voiced audio. Only hesitation is refused.
  const filler = new Set(['uh', 'um', 'erm', 'hmm', 'mm', 'mhm', 'ah', 'oh', 'eh', 'huh', 'er'])
  return words.every((word) => filler.has(word))
}

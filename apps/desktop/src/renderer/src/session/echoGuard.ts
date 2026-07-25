/**
 * Guard against the assistant hearing itself.
 *
 * Spoken answers come out of the speakers, and the microphone is still open for
 * barge-in. The browser's echo canceller removes most of it, but anything that
 * leaks through would be transcribed, submitted as a new turn, answered, and
 * spoken again — a loop that runs until the user pulls the plug. So a transcript
 * that is mostly words we just said is discarded rather than submitted.
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

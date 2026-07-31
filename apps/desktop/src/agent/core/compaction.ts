/**
 * Asking the model to summarise the turns compaction is about to drop.
 *
 * V1 built the digest deterministically from the transcript and the tool
 * ledger, which never invents anything and never explains anything either: it
 * lists requests, files, and commands, and the reasoning that connected them is
 * gone. A model can write the part that mattered — what was being attempted and
 * why the last approach was abandoned — but it can also quietly drop a file it
 * did not think was important, or claim a command succeeded.
 *
 * So the two are combined rather than swapped. The model writes the narrative;
 * the facts stay machine-generated and are appended verbatim underneath. If the
 * request fails, times out, or comes back empty, the deterministic digest is the
 * whole summary and compaction proceeds — a summary is not worth failing a
 * session over.
 */

/** How long the summary request may take before compaction gives up on it. */
const SUMMARY_TIMEOUT_MS = 30_000

/** Characters of transcript handed to the model. Enough for a long session. */
const TRANSCRIPT_LIMIT = 24_000

/**
 * Characters of narrative kept.
 *
 * The instruction asks for at most eight sentences, and an instruction is not a
 * guarantee. A model that answers with an essay would trade one context problem
 * for another, so the prose is cut at a sentence boundary near this length.
 */
const PROSE_LIMIT = 2_000

export type SummaryRequest = {
  /** OpenAI-compatible base URL, e.g. `http://127.0.0.1:1710/v1`. */
  baseUrl: string
  apiKey: string
  model: string
  /** The turns being dropped, already rendered as text. */
  transcript: string
  /** The deterministic digest, given as context and kept regardless. */
  facts: string
  fetchImpl?: typeof fetch
  timeoutMs?: number
}

const INSTRUCTIONS = [
  'You are compacting the earlier part of a coding session so work can continue.',
  'Write at most 8 sentences covering: what the user asked for, what was tried,',
  'what was decided and why, and anything left unfinished or unresolved.',
  'Do not list files or commands — those are recorded separately and will be',
  'appended to your summary. Do not claim anything succeeded unless the',
  'transcript says so. Write plainly, in the third person, with no preamble.'
].join(' ')

/**
 * Ask the session's own model for the narrative half of the summary.
 *
 * Returns `null` for every failure, deliberately: the caller has a complete
 * summary without this, and compaction is usually happening because the context
 * is already full — the worst outcome would be to fail there.
 */
export async function requestModelSummary(request: SummaryRequest): Promise<string | null> {
  const fetchImpl = request.fetchImpl ?? fetch
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), request.timeoutMs ?? SUMMARY_TIMEOUT_MS)
  try {
    const response = await fetchImpl(`${request.baseUrl}/chat/completions`, {
      method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${request.apiKey}`,
          'x-brazier-mode': 'agent',
          'x-brazier-slot': '0'
        },
      signal: controller.signal,
      body: JSON.stringify({
        model: request.model,
        stream: false,
        messages: [
          { role: 'system', content: INSTRUCTIONS },
          {
            role: 'user',
            content: `Known facts:\n${request.facts}\n\nTranscript to summarise:\n${clip(
              request.transcript
            )}`
          }
        ]
      })
    })
    if (!response.ok) return null
    const payload = (await response.json()) as {
      choices?: Array<{ message?: { content?: unknown } }>
    }
    const content = payload.choices?.[0]?.message?.content
    if (typeof content !== 'string') return null
    const text = clipProse(content.trim())
    return text.length > 0 ? text : null
  } catch {
    // Aborted, unreachable, or unparseable: all the same outcome here.
    return null
  } finally {
    clearTimeout(timeout)
  }
}

/**
 * The stored summary: the model's narrative first, the facts under it.
 *
 * Facts are never replaced by prose. A model that omits a changed file must not
 * be able to make that file's history disappear from the session.
 */
export function mergeSummary(prose: string | null, facts: string): string {
  if (!prose) return facts
  return `${prose}\n\n${facts}`
}

/** Render dropped turns for the model, oldest first, bounded in size. */
export function renderTranscript(
  messages: Array<{ role: string; text?: string; output?: string; tool?: string }>
): string {
  const lines = messages.map((message) => {
    if (message.role === 'tool') {
      const output = (message.output ?? '').split('\n').slice(0, 3).join(' ').trim()
      return output.length > 0 ? `tool ${message.tool ?? 'unknown'}: ${output}` : ''
    }
    const text = (message.text ?? '').trim()
    // A turn with nothing in it contributes a bare role label and nothing else,
    // which is context spent on saying that a message existed.
    return text.length > 0 ? `${message.role}: ${text}` : ''
  })
  return lines.filter((line) => line.length > 0).join('\n')
}

/** Cut over-long prose at the last sentence that fits. */
export function clipProse(text: string): string {
  if (text.length <= PROSE_LIMIT) return text
  const head = text.slice(0, PROSE_LIMIT)
  const lastStop = Math.max(head.lastIndexOf('. '), head.lastIndexOf('.\n'))
  // Only cut at a sentence if one ends late enough to leave a usable summary;
  // otherwise take the characters and mark the cut.
  if (lastStop > PROSE_LIMIT / 2) return head.slice(0, lastStop + 1)
  return `${head.trimEnd()}…`
}

function clip(text: string): string {
  if (text.length <= TRANSCRIPT_LIMIT) return text
  // Keep the end: the most recent turns are the ones still being acted on.
  return `…${text.slice(text.length - TRANSCRIPT_LIMIT)}`
}

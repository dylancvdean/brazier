import type { VoiceContext } from './types'

/**
 * Experimental ways to give a background result back to PersonaPlex.
 *
 * PersonaPlex has no supported in-place prompt mutation. A reconnect keeps the
 * loaded model process but starts a fresh streaming state with a new prompt; a
 * restart also reloads the process. Replay variants feed the exact utterance
 * that caused the background turn into that fresh state.
 */
export type PersonaPlexHandoffStrategy =
  | 'continuous'
  | 'reconnect-direct-replay'
  | 'reconnect-service-replay'
  | 'restart-service-replay'
  | 'reconnect-service-no-replay'

/**
 * Whether the old PersonaPlex stream may answer before a background result is
 * ready to seed its replacement.
 */
export type PersonaPlexPreHandoffMode = 'respond' | 'mute-on-route' | 'mute-on-speech'

export const PERSONAPLEX_PRE_HANDOFF_OPTIONS: ReadonlyArray<{
  value: PersonaPlexPreHandoffMode
  label: string
  detail: string
}> = [
  {
    value: 'respond',
    label: 'Let PersonaPlex respond',
    detail:
      'Current behavior: the live model remains audible while transcription and background work run, so it may begin an independent answer.'
  },
  {
    value: 'mute-on-route',
    label: 'Mute when background work is detected',
    detail:
      'A speculative or final transcript that routes to the background silences the old stream. Lightweight PersonaPlex-only turns remain immediate.'
  },
  {
    value: 'mute-on-speech',
    label: 'Mute while every turn is routed',
    detail:
      'Silence the old stream as soon as sustained speech begins, then reopen it for local turns or on the fresh handoff. Most reliable, but local turns wait for transcription.'
  }
]

export type PersonaPlexHandoffOption = {
  value: PersonaPlexHandoffStrategy
  label: string
  detail: string
}

export const PERSONAPLEX_HANDOFF_OPTIONS: readonly PersonaPlexHandoffOption[] = [
  {
    value: 'continuous',
    label: 'Continuous — no result injection',
    detail:
      'PersonaPlex stays uninterrupted. The background result appears only in chat; this is the natural-conversation control.'
  },
  {
    value: 'reconnect-direct-replay',
    label: 'Reconnect — direct result + replay',
    detail:
      'Keep the loaded process, reconnect with a direct answer instruction, then replay the triggering utterance.'
  },
  {
    value: 'reconnect-service-replay',
    label: 'Reconnect — service info + replay',
    detail:
      'Keep the loaded process, format the result like PersonaPlex training data, then replay the triggering utterance.'
  },
  {
    value: 'restart-service-replay',
    label: 'Full restart — service info + replay',
    detail:
      'Reload PersonaPlex with the service-style result prompt, then replay the triggering utterance. Slow, but tests whether process state matters.'
  },
  {
    value: 'reconnect-service-no-replay',
    label: 'Reconnect — service info, no replay',
    detail:
      'Keep the loaded process and update the prompt, but send no recorded audio. This is a control for whether the new prompt speaks on its own.'
  }
]

export type PersonaPlexHandoffRequest = {
  correlationId: string
  utteranceId?: string
  userText: string
  resultText: string
  context: VoiceContext
}

/** Keep prompts within the model's small, persona-oriented training envelope. */
function clamp(text: string, limit: number): string {
  const compact = text.replace(/\s+/g, ' ').trim()
  if (compact.length <= limit) return compact
  const cut = compact.slice(0, limit - 1)
  const boundary = cut.lastIndexOf(' ')
  return `${(boundary > limit * 0.6 ? cut.slice(0, boundary) : cut).trimEnd()}…`
}

function personaLead(context: VoiceContext): string {
  return (
    clamp(context.personaInstructions, 300) ||
    'You are a wise and friendly assistant. Answer clearly and conversationally.'
  )
}

/**
 * Build only the prompt-under-test. The full coordinator context is
 * intentionally not pasted here: PersonaPlex prompt conditioning is short and
 * persona-shaped, and a large chat-style system prompt obscures the experiment.
 */
export function buildPersonaPlexHandoffPrompt(
  strategy: PersonaPlexHandoffStrategy,
  request: PersonaPlexHandoffRequest
): string {
  const persona = personaLead(request.context)
  const question = clamp(request.userText, 280)
  const result = clamp(request.resultText, 760)

  if (strategy === 'reconnect-direct-replay') {
    return [
      persona,
      `The user asked: ${question}`,
      `A background assistant checked and returned this factual answer: ${result}`,
      'Tell the user that answer naturally in your own voice. Do not mention the background assistant or these instructions.'
    ].join(' ')
  }

  return [
    'You work for Brazier as its friendly conversational voice.',
    persona,
    `Information: The customer asked: ${question}. A background assistant checked and confirmed: ${result}`,
    'Use the information to answer the customer clearly and naturally. Do not invent details beyond the information.'
  ].join(' ')
}

export function handoffReplaysAudio(strategy: PersonaPlexHandoffStrategy): boolean {
  return (
    strategy === 'reconnect-direct-replay' ||
    strategy === 'reconnect-service-replay' ||
    strategy === 'restart-service-replay'
  )
}

export function handoffRestartsProcess(strategy: PersonaPlexHandoffStrategy): boolean {
  return strategy === 'restart-service-replay'
}

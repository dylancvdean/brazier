/**
 * Exact speech rendering for authoritative text.
 *
 * PersonaPlex cannot be handed a sentence to say: the persona is a process
 * launch flag and the wire protocol carries audio only. So an authoritative
 * answer is spoken through the platform synthesizer instead, verbatim — the
 * plan's exact-rendering fallback. Where the host has no synthesizer this
 * reports unavailable and the answer stays text-only; nothing pretends it was
 * spoken.
 *
 * Which means the voice that answers is not the persona's. It cannot be until
 * the runtime accepts a text frame, but it does not have to be whatever the
 * operating system happened to default to either: the voice and the speaking
 * rate are chosen, remembered, and applied to every authoritative sentence.
 */

export type SpeechHandlers = {
  onStart?: () => void
  onEnd?: () => void
  onError?: (message: string) => void
}

export interface SpeechRenderer {
  available(): boolean
  speak(text: string, handlers?: SpeechHandlers): void
  stop(): void
}

/** Speaking rate when nothing has been chosen: a touch above conversational. */
export const DEFAULT_SPEECH_RATE = 1.05

/**
 * The voice to use, given what the host offers and what was asked for.
 *
 * Matched on `voiceURI` first because that is what a choice is stored as, then
 * on name, so a preference survives a host that reports the same voice under a
 * different URI. An unmatched preference falls back to the host default rather
 * than to silence: losing the chosen voice is a disappointment, losing the
 * answer is a failure.
 */
export function selectVoice(
  voices: SpeechSynthesisVoice[],
  preference: string | undefined
): SpeechSynthesisVoice | null {
  if (voices.length === 0) return null
  if (preference) {
    const match =
      voices.find((voice) => voice.voiceURI === preference) ??
      voices.find((voice) => voice.name === preference)
    if (match) return match
  }
  return voices.find((voice) => voice.default) ?? null
}

/** Chromium's `speechSynthesis`, backed by the operating system's voices. */
export class PlatformSpeechRenderer implements SpeechRenderer {
  private readonly rate: () => number
  private readonly voice: () => string | undefined

  /**
   * Read through functions rather than captured values: the renderer is built
   * once per session and the preference can change while it is live.
   */
  constructor(options: { rate?: () => number; voice?: () => string | undefined } = {}) {
    this.rate = options.rate ?? (() => DEFAULT_SPEECH_RATE)
    this.voice = options.voice ?? (() => undefined)
  }

  available(): boolean {
    return (
      typeof speechSynthesis !== 'undefined' &&
      typeof SpeechSynthesisUtterance !== 'undefined' &&
      // A host with no installed voices (Linux without speech-dispatcher)
      // silently drops utterances, which would look like speech that happened.
      speechSynthesis.getVoices().length > 0
    )
  }

  speak(text: string, handlers: SpeechHandlers = {}): void {
    if (!this.available()) {
      handlers.onError?.('This host has no speech synthesizer installed.')
      return
    }
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate = this.rate()
    const voice = selectVoice(speechSynthesis.getVoices(), this.voice())
    if (voice) {
      utterance.voice = voice
      // Some platforms ignore `voice` unless the language agrees with it.
      utterance.lang = voice.lang
    }
    utterance.onstart = () => handlers.onStart?.()
    utterance.onend = () => handlers.onEnd?.()
    utterance.onerror = (event) => {
      // Cancelling raises `interrupted`/`canceled`, which is not a failure.
      if (event.error === 'interrupted' || event.error === 'canceled') {
        handlers.onEnd?.()
        return
      }
      handlers.onError?.(`Speech synthesis failed: ${event.error}`)
    }
    speechSynthesis.speak(utterance)
  }

  stop(): void {
    if (typeof speechSynthesis === 'undefined') return
    speechSynthesis.cancel()
  }
}

/** Renderer for hosts with no speech path. Always honest about it. */
export class UnavailableSpeechRenderer implements SpeechRenderer {
  available(): boolean {
    return false
  }

  speak(_text: string, handlers: SpeechHandlers = {}): void {
    handlers.onError?.('No speech output is available on this host.')
  }

  stop(): void {
    // Nothing is playing.
  }
}

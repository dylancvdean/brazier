/**
 * Exact speech rendering for authoritative text.
 *
 * PersonaPlex cannot be handed a sentence to say: the persona is a process
 * launch flag and the wire protocol carries audio only. So an authoritative
 * answer is spoken through the platform synthesizer instead, verbatim — the
 * plan's exact-rendering fallback. Where the host has no synthesizer this
 * reports unavailable and the answer stays text-only; nothing pretends it was
 * spoken.
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

/** Chromium's `speechSynthesis`, backed by the operating system's voices. */
export class PlatformSpeechRenderer implements SpeechRenderer {
  private readonly rate: number

  constructor(options: { rate?: number } = {}) {
    this.rate = options.rate ?? 1.05
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
    utterance.rate = this.rate
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

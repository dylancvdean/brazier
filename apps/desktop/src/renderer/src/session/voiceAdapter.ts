/**
 * Voice adapter: the existing PersonaPlex stack, normalized.
 *
 * Two things the plan assumes are not in the Moshi protocol, and this is where
 * they are supplied:
 *
 * - **User transcripts.** The socket's text frames are the *model's* speech.
 *   The user's words come from segmenting the captured microphone stream and
 *   transcribing each finished utterance through the daemon's ASR endpoint.
 * - **Speaking a given sentence.** PersonaPlex takes its persona as a launch
 *   flag and accepts audio only, so an authoritative answer is spoken verbatim
 *   through the exact renderer instead.
 *
 * Because PersonaPlex is a speech-to-speech model it also answers on its own,
 * and it cannot be told to wait. When an exact renderer exists its audio is
 * gated off for the session, so the user never hears two different answers to
 * one question. Its text is still surfaced — and still treated as untrusted.
 */

import {
  createVoiceSession,
  endVoiceSession,
  getVoiceSession,
  transcribeAudio,
  type VoiceSessionInfo
} from '../api'
import { VoiceStream, voiceStreamSupported } from '../audio/voiceStream'
import {
  UtteranceSegmenter,
  encodeWav,
  frameRms,
  padTrailingSilence
} from '../audio/utterance'
import type { VoiceAdapter, VoiceAdapterEvent, VoiceSessionHandle } from './adapters'
import { isEchoOfSpokenText } from './echoGuard'
import { PlatformSpeechRenderer, type SpeechRenderer } from './speechRenderer'
import type { SpeechRequest, VoiceContext } from './types'
import { renderVoicePrompt } from './voiceContext'

/** How often the capture path reports its state, in milliseconds. */
const CAPTURE_REPORT_MS = 1000

/** How long to wait for the first microphone frame before reporting silence. */
const CAPTURE_GRACE_MS = 2000

export type PersonaPlexAdapterOptions = {
  /** PersonaPlex model to run; empty picks the daemon's default. */
  modelId?: () => string
  /**
   * Which ASR interface transcribes utterances: `streaming-asr` for the
   * Nemotron worker, or undefined for the daemon's whisper default.
   */
  asrEngine?: () => string | undefined
  renderer?: SpeechRenderer
  /** Meters for the voice UI. */
  onInputLevel?: (level: number) => void
  onOutputLevel?: (level: number) => void
  /** Reported when a transcription attempt fails. */
  onTranscriptionError?: (message: string) => void
}

export class PersonaPlexVoiceAdapter implements VoiceAdapter {
  private readonly listeners = new Set<(event: VoiceAdapterEvent) => void>()
  private readonly renderer: SpeechRenderer
  private stream: VoiceStream | null = null
  private segmenter: UtteranceSegmenter | null = null
  private sessionId: string | null = null
  private speaking: string | null = null
  private transcribing = 0
  /** Most recent spoken text, so the microphone cannot hear it back in. */
  private lastSpokenText: string | null = null
  /** Whether PersonaPlex's own voice is audible. See `setModelAudioEnabled`. */
  private modelAudioEnabled = true
  private captureFrames = 0
  private capturePeak = 0
  private captureTimer: number | null = null

  constructor(private readonly options: PersonaPlexAdapterOptions = {}) {
    this.renderer = options.renderer ?? new PlatformSpeechRenderer()
  }

  subscribe(listener: (event: VoiceAdapterEvent) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  private publish(event: VoiceAdapterEvent): void {
    // An adapter nobody is listening to behaves exactly like one that is
    // working: the microphone runs, the logs here appear, and every event is
    // dropped on the floor. Worth saying out loud rather than inferring from
    // which downstream lines are missing.
    if (this.listeners.size === 0) {
      console.warn(`[voice] adapter published ${event.type} with no listener attached`)
      return
    }
    for (const listener of [...this.listeners]) listener(event)
  }

  canSpeak(): boolean {
    return this.renderer.available()
  }

  async startSession(context: VoiceContext): Promise<VoiceSessionHandle> {
    if (!voiceStreamSupported()) {
      throw new Error('This build lacks the WebCodecs Opus support realtime voice needs.')
    }
    // A second start would otherwise leave the first graph capturing: both feed
    // the same segmenter and the same counters, while only the newer one is
    // reachable to stop. Two sockets to one PersonaPlex process, and a UI whose
    // session is not the one holding the microphone.
    if (this.stream) {
      console.warn('[voice] replacing a capture graph that was still running')
      await this.endSession()
    }

    // The bounded context becomes the launch persona: it is the only runtime
    // guidance PersonaPlex accepts.
    const persona = renderVoicePrompt(context)
    const session = await this.openSession(persona)

    // Logged at each boundary: an utterance that opens and is then discarded for
    // being too short looks the same from outside as speech never detected.
    this.segmenter = new UtteranceSegmenter({
      onSpeechStart: (utteranceId) => {
        // Capturing, not interrupting. A cough gets recorded and then thrown
        // away without ever taking the assistant's turn from it.
        console.debug(`[voice] speech detected (${utteranceId})`)
      },
      onSustainedSpeech: (utteranceId) => {
        console.debug(`[voice] sustained speech (${utteranceId}) — interrupting`)
        this.publish({ type: 'userSpeechStarted', utteranceId })
      },
      onUtterance: (utterance) => {
        const seconds = (utterance.samples.length / utterance.sampleRate).toFixed(2)
        console.debug(`[voice] utterance ${utterance.id} closed, ${seconds}s — transcribing`)
        void this.transcribe(utterance)
      },
      onDiscarded: (utteranceId, reason) =>
        console.debug(`[voice] utterance ${utteranceId} discarded: ${reason}`)
    })

    const stream = new VoiceStream({
      onText: (text) => this.publish({ type: 'modelText', text }),
      onInputLevel: this.options.onInputLevel,
      onOutputLevel: this.options.onOutputLevel,
      onCaptureFrame: (samples, sampleRate) => this.onCaptureFrame(samples, sampleRate),
      onError: (error) => this.publish({ type: 'sessionError', error, fatal: false }),
      onState: (state) => {
        if (state === 'closed' && this.sessionId) {
          this.publish({
            type: 'sessionError',
            error: 'The PersonaPlex connection closed.',
            fatal: true
          })
        }
      }
    })
    try {
      await stream.start(session.ws_url)
    } catch (cause) {
      await stream.stop()
      await endVoiceSession(session.id).catch(() => undefined)
      throw cause
    }
    stream.setOutputGate(this.modelAudioEnabled)
    this.stream = stream
    this.sessionId = session.id
    this.captureFrames = 0
    this.capturePeak = 0
    this.startCaptureReports(stream)
    // A capture path that produces nothing is silent by nature, so ask it what
    // state it is in rather than waiting for a symptom that never comes.
    setTimeout(() => {
      if (this.stream !== stream) return
      const status = stream.inputStatus()
      // Logged either way: a working capture path is worth confirming, and the
      // console does not depend on the banner's conditions being right.
      if (this.captureFrames > 0) {
        console.debug(`[voice] capture running: ${this.captureFrames} frames, ${status}`)
        return
      }
      const error = `No microphone audio after ${CAPTURE_GRACE_MS / 1000}s — ${status}`
      console.warn(`[voice] ${error}`)
      this.publish({ type: 'sessionError', error, fatal: false })
    }, CAPTURE_GRACE_MS)
    return { id: session.id, startedAt: Date.now() }
  }

  /**
   * Get a session to talk to, adopting one the daemon already has.
   *
   * The daemon allows exactly one realtime session, and it outlives the window:
   * after a reload, or a start that failed partway, one is still registered and
   * creating another is refused. Reusing it costs nothing when its persona
   * already matches; when it does not, it is ended so the new context takes
   * effect, which is worth the model reload.
   */
  private async openSession(persona: string): Promise<VoiceSessionInfo> {
    const existing = await getVoiceSession()
      .then((response) => response.session)
      .catch(() => null)
    if (existing) {
      if (existing.persona_text === persona) return existing
      await endVoiceSession(existing.id).catch(() => undefined)
    }
    return createVoiceSession({
      model_id: this.options.modelId?.() || undefined,
      persona_text: persona
    })
  }

  /**
   * PersonaPlex takes its context only at launch, so a live session cannot be
   * updated. Renewal is how new context is applied; this keeps the interface
   * honest by reporting nothing changed rather than pretending it did.
   */
  async updateContext(_context: VoiceContext): Promise<void> {
    // Intentionally a no-op. See `SessionCoordinator.requestRenewal`.
  }

  async speak(request: SpeechRequest): Promise<void> {
    if (!this.renderer.available()) {
      throw new Error('No speech output is available on this host.')
    }
    // A new utterance replaces whatever was playing; the coordinator has
    // already decided this one should be heard.
    this.renderer.stop()
    this.speaking = request.correlationId
    this.lastSpokenText = request.text
    this.segmenter?.setGuarded(true)
    this.renderer.speak(request.text, {
      onStart: () => this.publish({ type: 'speechStarted', correlationId: request.correlationId }),
      onEnd: () => {
        if (this.speaking !== request.correlationId) return
        this.speaking = null
        this.segmenter?.setGuarded(false)
        this.publish({ type: 'speechCompleted', correlationId: request.correlationId })
      },
      onError: (error) => {
        if (this.speaking === request.correlationId) this.speaking = null
        this.segmenter?.setGuarded(false)
        this.publish({ type: 'sessionError', error, fatal: false })
      }
    })
  }

  setModelAudioEnabled(enabled: boolean): void {
    this.modelAudioEnabled = enabled
    this.stream?.setOutputGate(enabled)
    if (!enabled) return
    // Handing the voice back to PersonaPlex means stopping ours mid-sentence
    // rather than letting the two overlap.
    this.renderer.stop()
    this.speaking = null
  }

  async stopSpeaking(correlationId?: string): Promise<void> {
    if (correlationId && this.speaking && this.speaking !== correlationId) return
    const interrupted = this.speaking
    this.speaking = null
    this.segmenter?.setGuarded(false)
    this.renderer.stop()
    if (interrupted) this.publish({ type: 'speechInterrupted', correlationId: interrupted })
  }

  async endSession(): Promise<void> {
    const sessionId = this.sessionId
    // Cleared first so the socket closing is not reported as a session failure.
    this.sessionId = null
    this.stopCaptureReports()
    this.renderer.stop()
    this.speaking = null
    this.segmenter?.flush()
    this.segmenter = null
    await this.stream?.stop()
    this.stream = null
    if (sessionId) await endVoiceSession(sessionId).catch(() => undefined)
  }

  /** Stop sending microphone audio, ending any utterance in progress. */
  setMuted(muted: boolean): void {
    this.stream?.setMuted(muted)
    if (muted) this.segmenter?.flush()
  }

  /** True while at least one utterance is being transcribed. */
  isTranscribing(): boolean {
    return this.transcribing > 0
  }

  /**
   * Feed the segmenter, and report what the microphone is delivering.
   *
   * The report exists because a session that hears nothing looks the same
   * whether no frames are arriving or every frame is below the speech gate,
   * and the difference decides whether to look at the capture graph or the
   * threshold.
   */
  private onCaptureFrame(samples: Float32Array, sampleRate: number): void {
    this.segmenter?.push(samples, sampleRate)
    this.captureFrames += 1
    this.capturePeak = Math.max(this.capturePeak, frameRms(samples))
  }

  /**
   * Publish the state of the capture path on a timer.
   *
   * Driven by a clock rather than by arriving frames, because zero frames is
   * the case that most needs reporting and a frame callback cannot report it.
   */
  private startCaptureReports(stream: VoiceStream): void {
    this.stopCaptureReports()
    this.captureTimer = window.setInterval(() => {
      if (this.stream !== stream) {
        this.stopCaptureReports()
        return
      }
      this.publish({
        type: 'captureLevel',
        frames: this.captureFrames,
        peak: this.capturePeak,
        status: stream.inputStatus()
      })
      // Peak is per window, so a loud moment cannot mask a later silence.
      this.capturePeak = 0
    }, CAPTURE_REPORT_MS)
  }

  private stopCaptureReports(): void {
    if (this.captureTimer !== null) window.clearInterval(this.captureTimer)
    this.captureTimer = null
  }

  private async transcribe(utterance: {
    id: string
    samples: Float32Array
    sampleRate: number
  }): Promise<void> {
    this.transcribing += 1
    this.publish({ type: 'transcriptionStarted', utteranceId: utterance.id })
    try {
      // Padded so the decoder flushes its last words rather than dropping them.
      const audio = padTrailingSilence(utterance.samples, utterance.sampleRate)
      const text = await transcribeAudio(encodeWav(audio, utterance.sampleRate), {
        engine: this.options.asrEngine?.()
      })
      // Whatever leaked past the echo canceller is not a new question, and is
      // the one case where dropping the utterance without a word is correct.
      if (isEchoOfSpokenText(text, this.lastSpokenText)) {
        this.publish({ type: 'transcriptionEmpty', utteranceId: utterance.id })
        return
      }
      if (!text) {
        this.publish({ type: 'transcriptionEmpty', utteranceId: utterance.id })
        return
      }
      this.publish({ type: 'userTranscriptFinal', utteranceId: utterance.id, text })
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause)
      console.warn(`[voice] transcription failed: ${message}`)
      this.options.onTranscriptionError?.(message)
      // Recoverable: the session stays up and the next utterance may work.
      this.publish({
        type: 'sessionError',
        error: `Could not transcribe what you said: ${message}`,
        fatal: false
      })
    } finally {
      this.transcribing -= 1
    }
  }
}

/**
 * Voice adapter: the existing PersonaPlex stack, normalized.
 *
 * Two things the plan assumes are not in the Moshi protocol, and this is where
 * they are supplied:
 *
 * - **User transcripts.** The socket's text frames are the *model's* speech.
 *   The user's words come from segmenting the captured microphone stream and
 *   transcribing each finished utterance through the daemon's ASR endpoint.
 * - **Background-result experiments.** PersonaPlex has no supported in-place
 *   prompt mutation. We can keep its loaded process and reconnect with a new
 *   per-connection prompt, optionally replaying the exact utterance, or restart
 *   the process as a control. Platform TTS is intentionally not part of this
 *   adapter: PersonaPlex is the only audible voice.
 */

import {
  createVoiceSession,
  endVoiceSession,
  getVoiceSession,
  transcribeAudio,
  type VoiceSessionInfo
} from '../api'
import { VoiceStream, voiceStreamSupported } from '../audio/voiceStream'
import { SileroVadModel, StreamingSileroVad, type VadFrame } from '../audio/sileroVad'
import {
  SPEECH_THRESHOLD,
  UtteranceSegmenter,
  encodeWav,
  frameRms,
  padSpeechForAsr,
  padTrailingSilence
} from '../audio/utterance'
import type { VoiceAdapter, VoiceAdapterEvent, VoiceSessionHandle } from './adapters'
import { isEchoOfSpokenText } from './echoGuard'
import {
  buildPersonaPlexHandoffPrompt,
  handoffReplaysAudio,
  handoffRestartsProcess,
  type PersonaPlexHandoffRequest,
  type PersonaPlexHandoffStrategy
} from './personaplexHandoff'
import { coversUtterance } from './speculativeTranscript'
import type { VoiceContext } from './types'
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
  /**
   * Alternate installed ASR interface used once when a boosted short utterance
   * comes back empty. Null means there is no distinct fallback.
   */
  asrFallbackEngine?: () => { engine?: string } | null
  /** Accept and specially condition one-syllable / clipped turns. */
  shortSpeechBoost?: () => boolean
  /** Meters for the voice UI. */
  onInputLevel?: (level: number) => void
  onOutputLevel?: (level: number) => void
  /** Reported when a transcription attempt fails. */
  onTranscriptionError?: (message: string) => void
}

export class PersonaPlexVoiceAdapter implements VoiceAdapter {
  private readonly listeners = new Set<(event: VoiceAdapterEvent) => void>()
  private stream: VoiceStream | null = null
  private segmenter: UtteranceSegmenter | null = null
  private vad: StreamingSileroVad | null = null
  private vadHealthy = false
  private speechProbability: number | null = null
  private sessionId: string | null = null
  private transcribing = 0
  /**
   * Every transcription AbortController currently outstanding. `endSession`
   * aborts each one so a session that has ended stops consuming the daemon's
   * single ASR worker and cannot publish late transcripts to a listener set
   * the coordinator has already swapped out.
   */
  private readonly activeAbortControllers = new Set<AbortController>()
  /** PersonaPlex output is the only voice; this gate supports an explicit stop. */
  private modelAudioEnabled = true
  private muted = false
  private handoffGeneration = 0
  private sessionInfo: VoiceSessionInfo | null = null
  private lastModelText = ''
  /** Exact utterances retained until their corresponding background result lands. */
  private readonly utteranceAudio = new Map<
    string,
    { samples: Float32Array; sampleRate: number }
  >()
  /**
   * A transcription started at a pause, before the utterance closed. Kept so
   * the close can adopt it instead of paying for the same audio twice.
   */
  private speculative: {
    utteranceId: string
    voicedFrames: number
    sampleCount: number
    audioSeconds: number
    startedAt: number
    abort: AbortController
    done: Promise<{ text: string; engine: string; engineMs: number | null; roundTripMs: number }>
  } | null = null
  private captureFrames = 0
  private capturePeak = 0
  private captureTimer: number | null = null

  constructor(private readonly options: PersonaPlexAdapterOptions = {}) {}

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
    return voiceStreamSupported()
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

    // Capture a generation token so an `endSession` or `handoffResult` that
    // lands while we are awaiting the daemon, the VAD, or the socket can be
    // detected before assigning `this.stream`. Without this check, the stream
    // we build here is orphaned: nobody can stop it, and the Silero model it
    // loaded stays resident.
    const generation = ++this.handoffGeneration
    const bailBeforeStream = async (session: VoiceSessionInfo | null) => {
      await this.stopVad()
      if (session) await endVoiceSession(session.id).catch(() => undefined)
    }

    // The bounded context becomes the launch persona: it is the only runtime
    // guidance PersonaPlex accepts.
    const persona = renderVoicePrompt(context)
    const session = await this.openSession(persona)
    if (generation !== this.handoffGeneration) {
      await bailBeforeStream(session)
      throw new Error('Voice session was cancelled before it started.')
    }
    this.sessionInfo = session

    // Logged at each boundary: an utterance that opens and is then discarded for
    // being too short looks the same from outside as speech never detected.
    this.segmenter = new UtteranceSegmenter(
      {
        onSpeechStart: (utteranceId) => {
          // Capturing, not interrupting. A cough gets recorded and then thrown
          // away without ever taking the assistant's turn from it.
          console.debug(`[voice] speech detected (${utteranceId})`)
        },
        onSustainedSpeech: (utteranceId) => {
          console.debug(`[voice] sustained speech (${utteranceId}) — interrupting`)
          // An explicit stop only lasts until the person starts a new turn.
          this.modelAudioEnabled = true
          this.applyModelAudioGate()
          this.publish({ type: 'userSpeechStarted', utteranceId })
        },
        onPause: (snapshot) => {
          const seconds = (snapshot.samples.length / snapshot.sampleRate).toFixed(2)
          console.debug(`[voice] pause in ${snapshot.id} at ${seconds}s — transcribing ahead`)
          this.onPause(snapshot)
        },
        onUtterance: (utterance) => {
          const seconds = (utterance.samples.length / utterance.sampleRate).toFixed(2)
          console.debug(`[voice] utterance ${utterance.id} closed, ${seconds}s — transcribing`)
          this.rememberUtterance(utterance.id, utterance.samples, utterance.sampleRate)
          void this.transcribe(utterance)
        },
        onDiscarded: (utteranceId, reason) => {
          this.utteranceAudio.delete(utteranceId)
          console.debug(`[voice] utterance ${utteranceId} discarded: ${reason}`)
        }
      },
      {
        // Standard retains the former 200 ms floor as an A/B control.
        minimumNeuralFrames: this.options.shortSpeechBoost?.() === false ? 10 : 5
      }
    )
    await this.startVad()
    if (generation !== this.handoffGeneration) {
      await bailBeforeStream(session)
      throw new Error('Voice session was cancelled before it started.')
    }

    const stream = this.createStream()
    try {
      await stream.start(this.wsUrl(session, persona))
    } catch (cause) {
      await stream.stop()
      await this.stopVad()
      await endVoiceSession(session.id).catch(() => undefined)
      throw cause
    }
    if (generation !== this.handoffGeneration) {
      await stream.stop()
      await this.stopVad()
      await endVoiceSession(session.id).catch(() => undefined)
      throw new Error('Voice session was cancelled before it started.')
    }
    this.stream = stream
    this.applyModelAudioGate()
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

  async handoffResult(
    request: PersonaPlexHandoffRequest,
    strategy: PersonaPlexHandoffStrategy
  ): Promise<VoiceSessionHandle | null> {
    if (strategy === 'continuous') return null
    const generation = ++this.handoffGeneration
    const prompt = buildPersonaPlexHandoffPrompt(strategy, request)
    const recorded = request.utteranceId
      ? this.utteranceAudio.get(request.utteranceId) ?? null
      : null
    if (request.utteranceId) this.utteranceAudio.delete(request.utteranceId)

    let replacement: VoiceSessionHandle | null = null
    let session = this.sessionInfo
    if (handoffRestartsProcess(strategy)) {
      const previousStream = this.stream
      this.stream = null
      await previousStream?.stop()
      const previousId = this.sessionId
      this.sessionId = null
      this.sessionInfo = null
      if (previousId) await endVoiceSession(previousId).catch(() => undefined)
      session = await createVoiceSession({
        model_id: this.options.modelId?.() || undefined,
        persona_text: prompt
      })
      this.sessionId = session.id
      this.sessionInfo = session
      replacement = { id: session.id, startedAt: Date.now() }
    }
    if (!session) throw new Error('No PersonaPlex session is available for the handoff.')

    await this.replaceStream(session, prompt, generation)
    if (
      handoffReplaysAudio(strategy) &&
      recorded &&
      this.stream &&
      generation === this.handoffGeneration
    ) {
      const stream = this.stream
      stream.setMuted(true)
      // The model's downlink audio must stay silent for the wall-clock paced
      // replay: the server is now hearing the user's recorded utterance again,
      // so the user should hear the same question, not the start of the new
      // reply the model is already generating. Reopen only on the same
      // generation check `replayAudio` used to decide whether to continue.
      stream.setOutputGate(false)
      const trailingSilence = new Float32Array(Math.round(recorded.sampleRate * 0.8))
      const replay = new Float32Array(recorded.samples.length + trailingSilence.length)
      replay.set(recorded.samples)
      replay.set(trailingSilence, recorded.samples.length)
      await stream.replayAudio(
        replay,
        recorded.sampleRate,
        () => this.stream === stream && generation === this.handoffGeneration
      )
      if (this.stream === stream && generation === this.handoffGeneration) {
        stream.setMuted(this.muted)
        stream.setOutputGate(this.modelAudioEnabled)
      }
    }
    return replacement
  }

  setModelAudioEnabled(enabled: boolean): void {
    this.modelAudioEnabled = enabled
    this.applyModelAudioGate()
  }

  async stopSpeaking(_correlationId?: string): Promise<void> {
    // Stop PersonaPlex output without ending capture or the background task.
    // The next sustained user turn reopens it.
    this.modelAudioEnabled = false
    this.applyModelAudioGate()
  }

  async endSession(): Promise<void> {
    const sessionId = this.sessionId
    // Cleared first so the socket closing is not reported as a session failure.
    this.sessionId = null
    this.sessionInfo = null
    this.handoffGeneration += 1
    this.stopCaptureReports()
    this.segmenter?.flush()
    this.segmenter = null
    await this.stopVad()
    this.abandonSpeculative()
    for (const controller of this.activeAbortControllers) controller.abort()
    this.activeAbortControllers.clear()
    await this.stream?.stop()
    this.stream = null
    this.utteranceAudio.clear()
    if (sessionId) await endVoiceSession(sessionId).catch(() => undefined)
  }

  /** Stop sending microphone audio, ending any utterance in progress. */
  setMuted(muted: boolean): void {
    this.muted = muted
    this.stream?.setMuted(muted)
    if (muted) {
      this.segmenter?.flush()
    }
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
    if (this.vad && this.vadHealthy) this.vad.push(samples, sampleRate)
    else this.segmenter?.push(samples, sampleRate)
    this.captureFrames += 1
    this.capturePeak = Math.max(this.capturePeak, frameRms(samples))
  }

  private applyModelAudioGate(): void {
    this.stream?.setOutputGate(this.modelAudioEnabled)
  }

  private rememberUtterance(
    utteranceId: string,
    samples: Float32Array,
    sampleRate: number
  ): void {
    this.utteranceAudio.set(utteranceId, { samples: samples.slice(), sampleRate })
    while (this.utteranceAudio.size > 8) {
      const oldest = this.utteranceAudio.keys().next().value
      if (oldest === undefined) break
      this.utteranceAudio.delete(oldest)
    }
  }

  /** Add the per-connection prompt accepted by both PersonaPlex backends. */
  private wsUrl(session: VoiceSessionInfo, prompt: string): string {
    const url = new URL(session.ws_url)
    url.searchParams.set('text_prompt', prompt)
    // NVIDIA's server requires a concrete filename. MLX accepts the same built-
    // in id and resolves it through its voice bundle.
    url.searchParams.set('voice_prompt', session.voice_prompt ?? 'NATF2.pt')
    return url.toString()
  }

  private createStream(): VoiceStream {
    let stream: VoiceStream
    // Once a transport-level fatal has been published (device disconnect, or an
    // uncommanded socket close) the closing `onState('closed')` would otherwise
    // re-publish a fatal with a misleading message. This flag keeps that to
    // one event per stream.
    let fatalPublished = false
    stream = new VoiceStream({
      onText: (text) => {
        this.lastModelText = `${this.lastModelText}${text}`.slice(-2000)
        this.publish({ type: 'modelText', text })
      },
      onInputLevel: this.options.onInputLevel,
      onOutputLevel: this.options.onOutputLevel,
      onCaptureFrame: (samples, sampleRate) => this.onCaptureFrame(samples, sampleRate),
      onError: (error) => {
        if (this.stream === stream) {
          this.publish({ type: 'sessionError', error, fatal: false })
        }
      },
      onFatalError: (error) => {
        if (this.stream === stream && !fatalPublished) {
          fatalPublished = true
          this.publish({ type: 'sessionError', error, fatal: true })
        }
      },
      onState: (state) => {
        if (
          state === 'closed' &&
          this.stream === stream &&
          this.sessionId &&
          !fatalPublished
        ) {
          fatalPublished = true
          this.publish({
            type: 'sessionError',
            error: 'The PersonaPlex connection closed.',
            fatal: true
          })
        }
      }
    })
    return stream
  }

  /**
   * Replace only the browser-side stream. The daemon session keeps the loaded
   * PersonaPlex process unless the selected strategy explicitly restarted it.
   */
  private async replaceStream(
    session: VoiceSessionInfo,
    prompt: string,
    generation: number
  ): Promise<void> {
    const previous = this.stream
    this.stream = null
    await previous?.stop()
    if (generation !== this.handoffGeneration) return
    // The old stream is now gone, so reopening cannot leak its independent
    // answer. The replacement needs to be audible for the checked-result replay.
    this.modelAudioEnabled = true
    const stream = this.createStream()
    await stream.start(this.wsUrl(session, prompt))
    if (generation !== this.handoffGeneration) {
      await stream.stop()
      return
    }
    this.stream = stream
    stream.setMuted(this.muted)
    this.applyModelAudioGate()
    this.startCaptureReports(stream)
  }

  /** Load the bundled model before capture starts, falling back without failing voice. */
  private async startVad(): Promise<void> {
    const startedAt = performance.now()
    try {
      const model = await SileroVadModel.create()
      this.vadHealthy = true
      this.vad = new StreamingSileroVad(
        model,
        (frame) => this.onVadFrame(frame),
        (error) => this.onVadError(error)
      )
      console.debug(`[voice] Silero VAD ready in ${Math.round(performance.now() - startedAt)}ms`)
    } catch (cause) {
      this.onVadError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  private onVadFrame(frame: VadFrame): void {
    if (!this.vadHealthy) return
    this.speechProbability = frame.speechProbability
    this.segmenter?.push(frame.samples, frame.sampleRate, frame.speechProbability)
  }

  private onVadError(error: string): void {
    if (!this.vadHealthy && this.vad === null) {
      console.warn(`[voice] Silero VAD unavailable, using energy fallback: ${error}`)
    } else {
      console.warn(`[voice] Silero VAD stopped, using energy fallback: ${error}`)
    }
    this.vadHealthy = false
    this.speechProbability = null
    this.publish({
      type: 'sessionError',
      error: `Speech detector unavailable; using the less selective energy fallback: ${error}`,
      fatal: false
    })
  }

  private async stopVad(): Promise<void> {
    const vad = this.vad
    this.vad = null
    this.vadHealthy = false
    this.speechProbability = null
    await vad?.release().catch(() => undefined)
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
        status: stream.inputStatus(),
        gate: this.segmenter?.currentGate() ?? SPEECH_THRESHOLD,
        noiseFloor: this.segmenter?.noiseLevel() ?? 0,
        vad: this.vadHealthy ? 'silero-v5' : 'energy-fallback',
        speechProbability: this.speechProbability
      })
      // Peak is per window, so a loud moment cannot mask a later silence.
      this.capturePeak = 0
    }, CAPTURE_REPORT_MS)
  }

  private stopCaptureReports(): void {
    if (this.captureTimer !== null) window.clearInterval(this.captureTimer)
    this.captureTimer = null
  }

  /**
   * Transcribe the utterance so far, while the close window is still running.
   *
   * A turn cannot start until the silence gate closes, and then it used to wait
   * again while the audio was decoded. Most pauses of this length are the end of
   * the sentence, so the transcript is usually already in hand when the gate
   * closes and the second wait disappears. When speech resumes instead, the
   * result is a partial — display only, never a turn — and the work is lost,
   * which is the trade being made.
   */
  private onPause(snapshot: {
    id: string
    samples: Float32Array
    sampleRate: number
    voicedFrames: number
  }): void {
    // A snapshot from earlier in the same sentence is now known to be a
    // fragment. Left running it would hold the daemon's one ASR worker, and the
    // transcription that decides the turn would queue behind work already known
    // to be useless.
    this.abandonSpeculative()
    const pending = this.startTranscription(snapshot.samples, snapshot.sampleRate)
    this.speculative = {
      utteranceId: snapshot.id,
      voicedFrames: snapshot.voicedFrames,
      sampleCount: snapshot.samples.length,
      audioSeconds: snapshot.samples.length / snapshot.sampleRate,
      ...pending
    }
    const claimed = this.speculative
    void pending.done
      .then((result) => {
        // Superseded means the user kept talking and this describes a fragment.
        if (this.speculative !== claimed || !result.text) return
        if (isEchoOfSpokenText(result.text, this.lastModelText)) return
        this.publish({
          type: 'userTranscriptPartial',
          utteranceId: snapshot.id,
          text: result.text
        })
      })
      .catch(() => {
        // A speculative failure is not worth reporting: the utterance is still
        // open, and the transcription that decides the turn has not run yet.
        if (this.speculative === claimed) this.speculative = null
      })
  }

  /** Drop the speculative transcription in flight, and stop paying for it. */
  private abandonSpeculative(): void {
    this.speculative?.abort.abort()
    this.speculative = null
  }

  /** Send audio for transcription, timing the round trip. */
  private startTranscription(
    samples: Float32Array,
    sampleRate: number
  ): {
    startedAt: number
    abort: AbortController
    done: Promise<{ text: string; engine: string; engineMs: number | null; roundTripMs: number }>
  } {
    const startedAt = Date.now()
    const abort = new AbortController()
    this.activeAbortControllers.add(abort)
    const boosted = this.options.shortSpeechBoost?.() !== false
    const short = samples.length / sampleRate <= 2
    // Boosted audio has decoder context before a clipped first syllable as well
    // as after the last token. Standard is retained as the tuning control.
    const audio = boosted && short
      ? padSpeechForAsr(samples, sampleRate)
      : padTrailingSilence(samples, sampleRate)
    const wav = encodeWav(audio, sampleRate)
    const primaryEngine = this.options.asrEngine?.()
    const done = transcribeAudio(wav, {
      engine: primaryEngine,
      signal: abort.signal
    }).then(async (first) => {
      let result = first
      let engineMs = first.durationMs
      // Empty short clips are where the engines differ most. If both are
      // installed, one alternate decode is cheaper than asking the person to
      // repeat themselves verbosely. Successful turns never pay this cost.
      const fallback =
        boosted && short && !first.text
          ? this.options.asrFallbackEngine?.() ?? null
          : null
      if (fallback) {
        const retried = await transcribeAudio(wav, {
          engine: fallback.engine,
          signal: abort.signal
        })
        result = retried
        engineMs =
          first.durationMs === null || retried.durationMs === null
            ? null
            : first.durationMs + retried.durationMs
        console.debug(
          `[voice] empty ${first.engine} short transcript retried with ${retried.engine}`
        )
      }
      return {
        text: result.text,
        engine: fallback ? `${first.engine} → ${result.engine}` : result.engine,
        engineMs,
        roundTripMs: Date.now() - startedAt
      }
    }).finally(() => {
      this.activeTranscriptions.delete(abort)
    })
    // Whichever way the transcription settles, the controller is no longer
    // in flight and must not be aborted by a later `endSession`.
    void done.then(
      () => this.activeAbortControllers.delete(abort),
      () => this.activeAbortControllers.delete(abort)
    )
    return { startedAt, abort, done }
  }

  private async transcribe(utterance: {
    id: string
    samples: Float32Array
    sampleRate: number
    voicedFrames: number
  }): Promise<void> {
    const closedAt = Date.now()
    this.transcribing += 1
    this.publish({ type: 'transcriptionStarted', utteranceId: utterance.id })
    const audioSeconds = utterance.samples.length / utterance.sampleRate
    // Usable only when it covers this exact audio: same utterance, same speech,
    // same samples. Anything else describes a sentence that was still going.
    const speculative = this.speculative
    const reused = coversUtterance(speculative, {
      id: utterance.id,
      voicedFrames: utterance.voicedFrames,
      sampleCount: utterance.samples.length
    })
    if (reused) this.speculative = null
    else this.abandonSpeculative()
    try {
      const pending =
        reused && speculative
          ? speculative
          : this.startTranscription(utterance.samples, utterance.sampleRate)
      const result = await pending.done
      const text = result.text
      // Two different numbers: what the engine cost, and what the turn waited
      // for after the user stopped talking. Only the second is felt.
      const waitedMs = Date.now() - closedAt
      console.debug(
        `[voice] ${result.engine} transcribed ${audioSeconds.toFixed(1)}s in ` +
          `${result.roundTripMs}ms, turn waited ${waitedMs}ms` +
          (reused ? ' (started at the pause)' : '') +
          (result.engineMs === null ? '' : ` (${result.engineMs}ms in the daemon)`)
      )
      this.publish({
        type: 'transcriptionMeasured',
        utteranceId: utterance.id,
        engine: result.engine,
        roundTripMs: result.roundTripMs,
        waitedMs,
        engineMs: result.engineMs,
        audioSeconds,
        startedAtPause: reused
      })
      // Whatever leaked past the echo canceller is not a new question, and is
      // the one case where dropping the utterance without a word is correct.
      if (isEchoOfSpokenText(text, this.lastModelText)) {
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

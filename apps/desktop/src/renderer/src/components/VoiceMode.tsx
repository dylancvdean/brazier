import {
  AlertTriangle,
  AudioLines,
  Bot,
  Download,
  LoaderCircle,
  Mic,
  MicOff,
  PhoneOff,
  ShieldAlert,
  Square,
  Timer,
  Volume2,
  VolumeX
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { BundledTool, LocalModel, RuntimeSettings } from '../api'
import { SPEECH_THRESHOLD } from '../audio/utterance'
import { primeVoiceAudio } from '../audio/voiceStream'
import { modelDisplayName } from '../model-utils'
import { VOICE_BACKGROUND_ROUTING_OPTIONS } from '../session/backgroundRouting'
import type { VoiceSessionTarget } from '../session/config'
import {
  PERSONAPLEX_HANDOFF_OPTIONS,
  PERSONAPLEX_PRE_HANDOFF_OPTIONS
} from '../session/personaplexHandoff'
import type { SessionCoordinatorHandle } from '../session/useSessionCoordinator'
import {
  VOICE_QUALIFICATION_PHRASES,
  buildVoiceQualificationResult,
  countRecognizedQualificationPhrases
} from '../session/voiceQualification'
import { VoiceSessionConfig } from './VoiceSessionConfig'
import { Markdown } from './Markdown'

type Props = {
  models: LocalModel[]
  realtimeAvailable: boolean
  /** Voice model chosen in the top bar; empty when none is installed. */
  modelId: string
  /** Whether the browser can capture and encode audio at all. */
  audioSupported: boolean
  /**
   * Which ASR interfaces the daemon reports as usable. PersonaPlex reports only
   * its own speech, so with none of them a spoken turn never becomes text and
   * nothing reaches the conversation — except with `neither`, which needs no
   * transcript at all.
   */
  asrAvailable: { batch: boolean; streaming: boolean }
  persona: string
  onPersonaChange: (persona: string) => void
  /** Everything the session about to start is configured from. */
  chatModelId: string
  onChatModelChange: (modelId: string) => void
  tools: BundledTool[]
  enabledTools: string[]
  onEnabledToolsChange: (names: string[]) => void
  settings: RuntimeSettings | null
  onSettingsSaved: (settings: RuntimeSettings) => void
  onRuntimeActivated?: () => void
  onAgentSessionBound: (agentSessionId: string) => void
  /** The shared conversation. Voice turns land in it beside typed ones. */
  session: SessionCoordinatorHandle
  onError: (message: string | null) => void
}

/** Engine ids as the daemon reports them, in the words the UI uses elsewhere. */
const ASR_LABELS: Record<string, string> = {
  'whisper.cpp': 'Whisper',
  'streaming-asr': 'Nemotron streaming'
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

/**
 * Where a live voice session sends what you say. Each choice names one
 * destination: a turn that could go to either place gives no way to tell which
 * answered, and no way to aim the next one.
 */
const TARGETS: Array<[VoiceSessionTarget, string, string]> = [
  [
    'chat',
    'Chat',
    'Spoken turns go to the chat model and join this conversation.'
  ],
  [
    'agent',
    'Agent',
    'Spoken turns go to the agent session bound to this conversation. Its results are saved in this shared conversation and tagged Agent. With no task bound, a turn is refused rather than answered without the workspace and tools you expected.'
  ],
  [
    'neither',
    'Neither',
    'Nothing is recorded and nothing is invoked. PersonaPlex answers in its own voice, as it does with voice mode used on its own.'
  ]
]

export function VoiceMode(props: Props): React.JSX.Element {
  const { session } = props
  const { snapshot, config } = session
  const [muted, setMuted] = useState(false)
  const [busy, setBusy] = useState(false)
  const [resolvingApproval, setResolvingApproval] = useState(false)
  const [microphoneClass, setMicrophoneClass] = useState<'built-in' | 'usb'>('built-in')
  const [qualificationTrial, setQualificationTrial] = useState<null | {
    phase: 'speech' | 'noise'
    transcriptBaseline: number
    transcriptTextBaseline: number
    vadProcessedBaseline: number
    interruptBaseline: number
    speechTranscriptEnd?: number
    speechTranscriptTextEnd?: number
    noiseStartedAt?: number
  }>(null)
  const scrollAnchor = useRef<HTMLDivElement>(null)

  const needsTranscripts = config.voiceSessionTarget !== 'neither'
  const anyAsr = props.asrAvailable.batch || props.asrAvailable.streaming
  const live = snapshot.voiceStatus === 'live'
  const starting = snapshot.voiceStatus === 'starting' || busy
  const selected = props.models.find((model) => model.id === props.modelId)
  const task = snapshot.task
  const speaking = snapshot.speakingCorrelationId !== null
  const working = snapshot.activeCorrelationId !== null
  /**
   * Why a conversation cannot start, in the order the user would fix them.
   * Empty when nothing is in the way.
   *
   * "Not connected" on its own read like a fault and sent people hunting
   * through Runtimes for a runtime that was already active, when what was
   * actually missing was ASR.
   */
  const blockedReason = !props.realtimeAvailable
    ? 'No PersonaPlex runtime or model — see below'
    : !props.audioSupported
      ? 'This build cannot capture audio — see below'
      : needsTranscripts && !anyAsr
        ? 'No transcription installed — see below'
        : ''
  const blocked = blockedReason !== ''

  useEffect(() => {
    scrollAnchor.current?.scrollIntoView({ behavior: 'smooth' })
  }, [snapshot.messages.length, snapshot.streamingText])

  useEffect(() => {
    setResolvingApproval(false)
  }, [snapshot.pendingApproval])

  useEffect(() => {
    if (snapshot.voiceStatus === 'off' || snapshot.voiceStatus === 'error') {
      setQualificationTrial(null)
    }
  }, [snapshot.voiceStatus])

  async function guard(action: () => Promise<void>): Promise<void> {
    setBusy(true)
    props.onError(null)
    try {
      await action()
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setBusy(false)
    }
  }

  function toggleMute(): void {
    const next = !muted
    setMuted(next)
    session.setMuted(next)
  }

  function startQualificationTrial(): void {
    const metrics = session.metrics()
    setQualificationTrial({
      phase: 'speech',
      transcriptBaseline: metrics.transcriptWaitMs.length,
      transcriptTextBaseline: metrics.transcriptTexts.length,
      vadProcessedBaseline: snapshot.capture.vadProcessedWindows,
      interruptBaseline: metrics.interruptToSpeechStopMs.length
    })
  }

  function startQualificationNoiseWindow(): void {
    if (!qualificationTrial || qualificationTrial.phase !== 'speech') return
    setQualificationTrial({
      ...qualificationTrial,
      phase: 'noise',
      speechTranscriptEnd: session.metrics().transcriptWaitMs.length,
      speechTranscriptTextEnd: session.metrics().transcriptTexts.length,
      noiseStartedAt: Date.now()
    })
  }

  async function saveTrialReport(): Promise<void> {
    if (
      !qualificationTrial ||
      qualificationTrial.phase !== 'noise' ||
      qualificationTrial.speechTranscriptEnd === undefined ||
      qualificationTrial.speechTranscriptTextEnd === undefined ||
      qualificationTrial.noiseStartedAt === undefined
    ) {
      throw new Error('Finish the speech and noise phases before saving qualification evidence.')
    }
    const allMetrics = session.metrics()
    const noiseMinutes = (Date.now() - qualificationTrial.noiseStartedAt) / 60_000
    const report = await buildVoiceQualificationResult({
      host: await window.brazier.qualificationHost(),
      microphoneClass,
      expectedSpeechUtterances: VOICE_QUALIFICATION_PHRASES.length,
      recognizedSpeechUtterances: countRecognizedQualificationPhrases(
        allMetrics.transcriptTexts.slice(
          qualificationTrial.transcriptTextBaseline,
          qualificationTrial.speechTranscriptTextEnd
        )
      ),
      noiseMinutes,
      falseNoiseUtterances:
        allMetrics.transcriptWaitMs.length - qualificationTrial.speechTranscriptEnd,
      captureVad: snapshot.capture.vad,
      vadWindowP95Ms: snapshot.capture.vadInferenceP95Ms,
      vadQueueLagP95Ms: snapshot.capture.vadQueueLagP95Ms,
      vadSamples: snapshot.capture.vadProcessedWindows - qualificationTrial.vadProcessedBaseline,
      models: { voice: props.modelId, background: props.chatModelId },
      metrics: {
        transcriptWaitMs: allMetrics.transcriptWaitMs.slice(
          qualificationTrial.transcriptBaseline,
          qualificationTrial.speechTranscriptEnd
        ),
        interruptToSpeechStopMs: allMetrics.interruptToSpeechStopMs.slice(
          qualificationTrial.interruptBaseline
        )
      }
    })
    const bytes = new TextEncoder().encode(`${JSON.stringify(report, null, 2)}\n`)
    const saved = await window.brazier.saveFile(
      `brazier-voice-${report.host_id}-${new Date().toISOString().replace(/[:.]/g, '-')}.json`,
      bytes.buffer
    )
    if (!saved) return
    setQualificationTrial(null)
    if (!report.passed) {
      throw new Error('The report was saved, but this trial did not satisfy the beta qualification gate.')
    }
  }

  /**
   * Shown wherever the agent destination is in force. Speech reaching an agent
   * means a misheard word can edit files and run commands. A call the permission
   * broker holds needs a spoken yes or a click — but the broker only holds what
   * its mode says to hold, and
   * the transcript is still the least reliable input in the application.
   */
  const agentWarning = (
    <p className="voice-danger">
      <AlertTriangle size={14} />
      <span>
        Voice control for agents is extremely experimental. Anything the permission broker holds is
        shown here and needs a spoken yes — but only what its mode holds, so a session set to
        skip permissions acts on what it thought it heard. Please don't use it on anything you care
        about, and even then only if you know what you're doing.
      </span>
    </p>
  )

  const statusLabel = starting
    ? 'Loading the voice model — first run takes a minute…'
    : live
      ? muted
        ? 'Microphone muted'
        : speaking
          ? 'Speaking'
          : working
            ? 'Working on your request'
            : // Speech detection and transcription can each stall without a
              // word, so the bar names the step the microphone has reached.
              snapshot.hearing === 'speaking'
              ? 'Hearing you…'
              : snapshot.hearing === 'transcribing'
                ? 'Transcribing…'
                : 'Listening'
      : snapshot.voiceStatus === 'error'
        ? `Stopped: ${snapshot.voiceError ?? 'unknown error'}`
        : blockedReason || 'Ready — press Start conversation'

  return (
    <section className="voice-mode">
      <header className="voice-bar">
        <div className={`voice-status ${live ? 'live' : ''}`}>
          <span className="voice-status-dot" />
          <span>{statusLabel}</span>
        </div>
        {/* Labelled, because two bare bars stacked together gave no way to tell
            the microphone from the model — and "the meter moves" then says
            nothing about whether anything is being heard. */}
        <div className="voice-meters" hidden={!live}>
          <div className="voice-meter-row" title="What the microphone is picking up">
            <Mic size={12} />
            <div className="voice-meter">
              <div
                className="voice-meter-fill"
                style={{ width: `${Math.round(session.inputLevel * 100)}%` }}
              />
            </div>
          </div>
          <div className="voice-meter-row" title="What the model is playing back">
            <Volume2 size={12} />
            <div className="voice-meter">
              <div
                className="voice-meter-fill model"
                style={{ width: `${Math.round(session.outputLevel * 100)}%` }}
              />
            </div>
          </div>
        </div>
        <div className="voice-bar-spacer" />
        {props.modelId ? (
          <span className="voice-status-model">
            {modelDisplayName(props.modelId, selected).title}
          </span>
        ) : null}
        {qualificationTrial?.phase === 'noise' ? (
          <button
            type="button"
            title="Save commit-bound hardware qualification evidence."
            onClick={() => void guard(saveTrialReport)}
          >
            <Download size={15} /> Save qualification
          </button>
        ) : null}
        {live ? (
          <>
            <button type="button" className={muted ? 'toggled' : ''} onClick={toggleMute}>
              {muted ? <MicOff size={15} /> : <Mic size={15} />}
              {muted ? 'Unmute' : 'Mute'}
            </button>
            {/* Three controls, because silencing the voice, dropping this
                answer, and abandoning the task are different decisions. */}
            <button
              type="button"
              title="Silence PersonaPlex until you start speaking again. The task keeps running."
              onClick={() => void guard(() => session.stopSpeaking())}
            >
              <VolumeX size={15} /> Stop speaking
            </button>
            <button
              type="button"
              className="danger"
              disabled={!working}
              title="Cancel the task. Anything already answered stays in the conversation."
              onClick={() => void guard(() => session.cancelAgentTask())}
            >
              <Square size={13} fill="currentColor" /> Cancel task
            </button>
            <button
              type="button"
              className="danger"
              onClick={() => void guard(() => session.endVoice())}
            >
              <PhoneOff size={15} /> End
            </button>
          </>
        ) : null}
      </header>

      {live ? (
        <section className="voice-qualification" aria-label="Voice hardware qualification">
          {!qualificationTrial ? (
            <>
              <span>Beta hardware trial</span>
              <select
                aria-label="Qualification microphone class"
                value={microphoneClass}
                onChange={(event) => setMicrophoneClass(event.target.value as 'built-in' | 'usb')}
              >
                <option value="built-in">Built-in microphone</option>
                <option value="usb">USB microphone</option>
              </select>
              <button
                type="button"
                disabled={!anyAsr || snapshot.capture.vad !== 'silero-v5'}
                title={
                  !anyAsr
                    ? 'Install a transcription runtime first.'
                    : snapshot.capture.vad !== 'silero-v5'
                      ? 'Silero VAD must be active before starting the trial.'
                      : 'Start the fixed speech and noise hardware protocol.'
                }
                onClick={startQualificationTrial}
              >
                Qualify voice
              </button>
            </>
          ) : qualificationTrial.phase === 'speech' ? (
            <details open>
              <summary>
                Read all {VOICE_QUALIFICATION_PHRASES.length} sentences once, and interrupt at
                least three answers
              </summary>
              <ol>
                {VOICE_QUALIFICATION_PHRASES.map((phrase) => <li key={phrase}>{phrase}</li>)}
              </ol>
              <button type="button" onClick={startQualificationNoiseWindow}>
                All read — start noise window
              </button>
            </details>
          ) : (
            <span>
              Noise window: {((Date.now() - (qualificationTrial.noiseStartedAt ?? Date.now())) / 60_000).toFixed(1)} / 5 minutes.
              Do not speak; use the room normally. Save from the top bar when complete.
            </span>
          )}
        </section>
      ) : null}

      {/* A live session that cannot transcribe, or whose turns are refused,
          fails once per utterance and otherwise looks exactly like one that is
          listening. Say so where the user is looking. */}
      {live && (snapshot.notice || snapshot.voiceError) ? (
        <p className="voice-live-notice">
          <AlertTriangle size={13} />
          <span>{snapshot.notice ?? snapshot.voiceError}</span>
        </p>
      ) : null}

      {live && config.voiceSessionTarget === 'agent' ? agentWarning : null}

      {/* A held call is otherwise only visible in the agent panel, which the
          person talking is very likely not looking at. Buttons as well as
          words: a spoken yes is the convenience, not the only way through. */}
      {live && snapshot.pendingApproval ? (
        <div
          className={`voice-approval ${snapshot.pendingApproval.environment === 'host' ? 'host' : ''}`}
        >
          <ShieldAlert size={16} />
          <div className="voice-approval-body">
            <strong>{snapshot.pendingApproval.summary}</strong>
            <span>
              {snapshot.pendingApproval.tool} · {snapshot.pendingApproval.risk} ·{' '}
              {snapshot.pendingApproval.environment === 'host'
                ? `outside the sandbox on ${snapshot.pendingApproval.executionLocation.daemon_display_name}`
                : `in the sandbox on ${snapshot.pendingApproval.executionLocation.daemon_display_name}`}
              {' · '}{snapshot.pendingApproval.executionLocation.platform}/
              {snapshot.pendingApproval.executionLocation.arch}
              {snapshot.pendingApproval.spoken ? ' · read out to you' : ''}
            </span>
            <span className="voice-approval-hint">
              Say <strong>yes</strong> to allow it or <strong>no</strong> to stop. Anything else
              leaves it held.
            </span>
          </div>
          <button
            type="button"
            className="danger"
            disabled={resolvingApproval}
            onClick={() => {
              if (resolvingApproval) return
              setResolvingApproval(true)
              void guard(() => session.resolveApproval('deny')).finally(() =>
                setResolvingApproval(false)
              )
            }}
          >
            {resolvingApproval ? <LoaderCircle className="spin" size={14} /> : null}
            Refuse
          </button>
          <button
            type="button"
            className="primary"
            disabled={resolvingApproval}
            onClick={() => {
              if (resolvingApproval) return
              setResolvingApproval(true)
              void guard(() => session.resolveApproval('approve')).finally(() =>
                setResolvingApproval(false)
              )
            }}
          >
            {resolvingApproval ? <LoaderCircle className="spin" size={14} /> : null}
            Allow once
          </button>
        </div>
      ) : null}

      {live ? (
        <div className="voice-conversation">
          {snapshot.messages.length === 0 && !snapshot.streamingText ? (
            <>
              <p className="voice-hint">
                {config.voiceSessionTarget === 'neither'
                  ? 'Speak whenever you like. Nothing is recorded — this is PersonaPlex on its own.'
                  : config.voiceBackgroundRouting === 'always'
                    ? 'Speak whenever you like. Pause when you are done and the turn is sent to the background.'
                    : config.voiceBackgroundRouting === 'auto'
                      ? 'Speak whenever you like. Lightweight turns stay with PersonaPlex; work and checked facts go to the background.'
                      : 'Speak whenever you like. PersonaPlex handles the turn unless you explicitly ask for background work.'}
              </p>
              {/* Until the first turn lands, say what the microphone is
                  actually delivering. "Nothing happened" has two causes that
                  look identical — no audio arriving, and audio too quiet to
                  count as speech — and only this tells them apart. */}
              <p className="voice-capture">
                {snapshot.capture.frames === 0
                  ? `No audio is reaching the microphone tap yet${
                      snapshot.capture.status ? ` — ${snapshot.capture.status}` : ''
                    }.`
                  : snapshot.capture.vad === 'silero-v5'
                    ? `Microphone: ${snapshot.capture.frames} frames, Silero speech probability ${(
                      snapshot.capture.speechProbability ?? 0
                      ).toFixed(2)}, VAD ${snapshot.capture.vadInferenceMs.toFixed(
                        1
                      )} ms/window with ${snapshot.capture.vadQueueLagMs.toFixed(
                        0
                      )} ms queued, loudest recent ${snapshot.capture.peak.toFixed(3)}. ${
                        snapshot.capture.status
                      }`
                    : `Microphone: ${snapshot.capture.frames} frames, loudest recent ${snapshot.capture.peak.toFixed(
                        3
                      )} — fallback speech has to clear ${(
                        snapshot.capture.gate || SPEECH_THRESHOLD
                      ).toFixed(3)}, the room reads ${snapshot.capture.noiseFloor.toFixed(4)}. ${
                        snapshot.capture.status
                      }`}
              </p>
            </>
          ) : (
            snapshot.messages.map((message) => (
              <article className={`voice-turn ${message.role} ${message.status}`} key={message.id}>
                <div className="voice-turn-who">
                  {message.role === 'user' ? (
                    <>
                      <Mic size={12} /> You
                    </>
                  ) : message.role === 'system' ? (
                    'System'
                  ) : (
                    <>
                      <Bot size={12} /> {message.source === 'assistant_agent' ? 'Agent' : 'Brazier'}
                    </>
                  )}
                  {message.status !== 'final' && (
                    <span className="turn-badge">{message.status}</span>
                  )}
                </div>
                <Markdown>{message.content}</Markdown>
              </article>
            ))
          )}
          {snapshot.streamingText ? (
            <article className="voice-turn assistant">
              <div className="voice-turn-who">
                <Bot size={12} /> Answering
                <LoaderCircle className="spin" size={12} />
              </div>
              <Markdown>{snapshot.streamingText}</Markdown>
            </article>
          ) : null}
          {snapshot.partialTranscript ? (
            <article className="voice-turn user partial">
              <div className="voice-turn-who">
                <AudioLines size={12} /> Hearing
              </div>
              <p>{snapshot.partialTranscript}</p>
            </article>
          ) : null}
          <div ref={scrollAnchor} />
        </div>
      ) : (
        <div className="voice-setup">
          <div className="voice-setup-inner">
            <div className="voice-setup-mark">
              <AudioLines size={26} />
            </div>
            <h2>Talk to it</h2>
            <p className="mode-empty">
              Full-duplex speech with PersonaPlex. Use headphones — the model hears your speakers.
            </p>

            {!props.realtimeAvailable ? (
              <p className="mode-empty">
                Realtime voice needs a PersonaPlex runtime and a downloaded{' '}
                <code>personaplex:</code> model. On Apple Silicon build PersonaPlex MLX from Manage →
                Runtimes (accept the nvidia/personaplex-7b-v1 license and set an HF token); on Linux
                CUDA build PersonaPlex / Moshi.
              </p>
            ) : null}
            {props.realtimeAvailable && !props.audioSupported ? (
              <p className="mode-empty">
                This build has no WebCodecs Opus support, which realtime voice needs for audio in
                and out.
              </p>
            ) : null}
            {props.realtimeAvailable && !anyAsr && needsTranscripts ? (
              <p className="mode-empty">
                Sending what you say to a model needs transcription, which is separate from
                PersonaPlex: the voice model reports only what <em>it</em> says. Any ASR interface
                will do — build WhisperKit under Manage → Runtimes, download a Whisper model from
                Discover if you already built whisper.cpp, or download the Nemotron ASR Streaming
                snapshot if you already built streaming ASR. Until then, use{' '}
                <strong>Neither</strong>, which talks to PersonaPlex directly and needs no
                transcript.
              </p>
            ) : null}

            <label className="voice-field">
              <span className="section-label">Persona</span>
              <textarea
                value={props.persona}
                onChange={(event) => props.onPersonaChange(event.target.value)}
                rows={3}
                placeholder="Describe who the model should be…"
              />
            </label>

            <div className="voice-field">
              <span className="section-label">Send what I say to</span>
              <div className="voice-target-choices" role="radiogroup" aria-label="Send speech to">
                {TARGETS.map(([value, label, detail]) => (
                  <button
                    key={value}
                    type="button"
                    role="radio"
                    aria-checked={config.voiceSessionTarget === value}
                    className={config.voiceSessionTarget === value ? 'active' : ''}
                    title={detail}
                    onClick={() => session.setConfig({ ...config, voiceSessionTarget: value })}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <p className="voice-notice">
                {TARGETS.find(([value]) => value === config.voiceSessionTarget)?.[2]}
              </p>
              {config.voiceSessionTarget === 'agent' ? agentWarning : null}
            </div>

            <VoiceSessionConfig
              target={config.voiceSessionTarget}
              models={props.models}
              voiceModelId={props.modelId}
              chatModelId={props.chatModelId}
              onChatModelChange={props.onChatModelChange}
              tools={props.tools}
              enabledTools={props.enabledTools}
              onEnabledToolsChange={props.onEnabledToolsChange}
              settings={props.settings}
              onSettingsSaved={props.onSettingsSaved}
              onRuntimeActivated={props.onRuntimeActivated}
              asrAvailable={props.asrAvailable}
              asrPreference={config.asrPreference}
              onAsrPreferenceChange={(asrPreference) =>
                session.setConfig({ ...config, asrPreference })
              }
              agentSessionId={snapshot.agentSessionId}
              onAgentSessionBound={props.onAgentSessionBound}
              onError={props.onError}
            />

            <div className="voice-options">
              <label className="voice-field">
                <span>Send transcript to background</span>
                <select
                  value={config.voiceBackgroundRouting}
                  disabled={config.voiceSessionTarget === 'neither'}
                  onChange={(event) =>
                    session.setConfig({
                      ...config,
                      voiceBackgroundRouting: event.target
                        .value as typeof config.voiceBackgroundRouting
                    })
                  }
                >
                  {VOICE_BACKGROUND_ROUTING_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <small>
                  {
                    VOICE_BACKGROUND_ROUTING_OPTIONS.find(
                      (option) => option.value === config.voiceBackgroundRouting
                    )?.detail
                  }
                </small>
              </label>
              <label className="voice-field">
                <span>Background result → PersonaPlex experiment</span>
                <select
                  value={config.personaplexHandoffStrategy}
                  onChange={(event) =>
                    session.setConfig({
                      ...config,
                      personaplexHandoffStrategy: event.target
                        .value as typeof config.personaplexHandoffStrategy
                    })
                  }
                >
                  {PERSONAPLEX_HANDOFF_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <small>
                  {
                    PERSONAPLEX_HANDOFF_OPTIONS.find(
                      (option) => option.value === config.personaplexHandoffStrategy
                    )?.detail
                  }
                </small>
              </label>
              <label className="voice-field">
                <span>PersonaPlex before background handoff</span>
                <select
                  value={config.personaplexPreHandoffMode}
                  disabled={config.personaplexHandoffStrategy === 'continuous'}
                  onChange={(event) =>
                    session.setConfig({
                      ...config,
                      personaplexPreHandoffMode: event.target
                        .value as typeof config.personaplexPreHandoffMode
                    })
                  }
                >
                  {PERSONAPLEX_PRE_HANDOFF_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <small>
                  {config.personaplexHandoffStrategy === 'continuous'
                    ? 'No reconnect or restart is selected, so there is no fresh handoff to wait for.'
                    : PERSONAPLEX_PRE_HANDOFF_OPTIONS.find(
                        (option) => option.value === config.personaplexPreHandoffMode
                      )?.detail}
                </small>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={config.shortSpeechBoost}
                  onChange={(event) =>
                    session.setConfig({ ...config, shortSpeechBoost: event.target.checked })
                  }
                />
                Short speech boost (100 ms floor, ASR padding, alternate retry)
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={config.interruptCancelsAgent}
                  onChange={(event) =>
                    session.setConfig({ ...config, interruptCancelsAgent: event.target.checked })
                  }
                />
                Talking over the assistant also cancels the task
              </label>
            </div>

            <button
              type="button"
              className="primary voice-start"
              disabled={starting || blocked}
              onClick={() => {
                primeVoiceAudio()
                void guard(() => session.startVoice())
              }}
            >
              {starting ? <LoaderCircle className="spin" size={16} /> : <Mic size={16} />}
              Start conversation
            </button>
          </div>
        </div>
      )}

      {/* Which interface should transcribe a spoken turn is an open question,
          and the honest answer is whichever is faster on this machine. Say what
          each one is actually costing rather than leaving it to be felt. */}
      {live && snapshot.transcription.length > 0 ? (
        <p className="voice-asr-cost">
          <Timer size={12} />
          {snapshot.transcription.map((cost) => (
            <span key={cost.engine}>
              {ASR_LABELS[cost.engine] ?? cost.engine}: {(cost.averageMs / 1000).toFixed(2)}s per
              utterance ({cost.realTimeFactor.toFixed(2)}× real time), of which{' '}
              {(cost.averageWaitMs / 1000).toFixed(2)}s waited for — {cost.startedAtPause} of{' '}
              {cost.utterances} started at a pause
            </span>
          ))}
        </p>
      ) : null}

      {live && config.showVoiceTranscripts && snapshot.voiceModelText ? (
        <details className="voice-model-text">
          <summary>
            <Bot size={12} /> What PersonaPlex said on its own
          </summary>
          <p>{snapshot.voiceModelText.slice(-600)}</p>
          <small>
            Not part of the conversation and not checked against anything. Only the turns above are
            authoritative.
          </small>
        </details>
      ) : null}

      {live && task ? (
        <div className="voice-task">
          <strong>
            {task.label} · {task.status}
            {task.activeTool ? ` · ${task.activeTool}` : ''}
          </strong>
          {task.confirmedResults.length > 0 ? (
            <ul>
              {task.confirmedResults.slice(-3).map((result, index) => (
                <li key={`${result}-${index}`}>{result}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </section>
  )
}

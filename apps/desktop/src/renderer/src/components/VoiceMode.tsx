import {
  AlertTriangle,
  AudioLines,
  Bot,
  LoaderCircle,
  Mic,
  MicOff,
  PhoneOff,
  Square,
  Volume2,
  VolumeX
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { BundledTool, LocalModel, RuntimeSettings } from '../api'
import { SPEECH_THRESHOLD } from '../audio/utterance'
import { modelDisplayName } from '../model-utils'
import type { VoiceSessionTarget } from '../session/config'
import type { SessionCoordinatorHandle } from '../session/useSessionCoordinator'
import { VoiceSessionConfig } from './VoiceSessionConfig'

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
    'Spoken turns go to the agent session bound to this conversation. With no task bound, a turn is refused rather than answered without the workspace and tools you expected.'
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
              disabled={!speaking}
              title="Stop the audio. The task keeps running."
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

      {/* A live session that cannot transcribe, or whose turns are refused,
          fails once per utterance and otherwise looks exactly like one that is
          listening. Say so where the user is looking. */}
      {live && (snapshot.notice || snapshot.voiceError) ? (
        <p className="voice-live-notice">
          <AlertTriangle size={13} />
          <span>{snapshot.notice ?? snapshot.voiceError}</span>
        </p>
      ) : null}

      {live ? (
        <div className="voice-conversation">
          {snapshot.messages.length === 0 && !snapshot.streamingText ? (
            <>
              <p className="voice-hint">
                {config.voiceSessionTarget === 'neither'
                  ? 'Speak whenever you like. Nothing is recorded — this is PersonaPlex on its own.'
                  : 'Speak whenever you like. Pause when you are done and the turn is sent.'}
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
                  : `Microphone: ${snapshot.capture.frames} frames, loudest recent ${snapshot.capture.peak.toFixed(
                      3
                    )} — speech has to clear ${SPEECH_THRESHOLD}. ${snapshot.capture.status}`}
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
                <p>{message.content}</p>
              </article>
            ))
          )}
          {snapshot.streamingText ? (
            <article className="voice-turn assistant">
              <div className="voice-turn-who">
                <Bot size={12} /> Answering
                <LoaderCircle className="spin" size={12} />
              </div>
              <p>{snapshot.streamingText}</p>
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
            </div>

            <VoiceSessionConfig
              target={config.voiceSessionTarget}
              models={props.models}
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

            {needsTranscripts && !session.canSpeak ? (
              <p className="voice-notice">
                <AlertTriangle size={13} /> This host has no speech synthesizer, so answers are
                shown rather than spoken.
              </p>
            ) : null}

            <div className="voice-options">
              <label>
                <input
                  type="checkbox"
                  checked={config.allowVoiceBackchannels}
                  onChange={(event) =>
                    session.setConfig({ ...config, allowVoiceBackchannels: event.target.checked })
                  }
                />
                Acknowledge while working (“let me check”)
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
              onClick={() => void guard(() => session.startVoice())}
            >
              {starting ? <LoaderCircle className="spin" size={16} /> : <Mic size={16} />}
              Start conversation
            </button>
          </div>
        </div>
      )}

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

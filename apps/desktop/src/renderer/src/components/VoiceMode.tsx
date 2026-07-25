import {
  AlertTriangle,
  Bot,
  LoaderCircle,
  Mic,
  MicOff,
  PhoneOff,
  Square,
  Volume2,
  VolumeX
} from 'lucide-react'
import { useState } from 'react'
import type { LocalModel } from '../api'
import { modelDisplayName } from '../model-utils'
import type { VoiceSessionTarget } from '../session/config'
import type { SessionCoordinatorHandle } from '../session/useSessionCoordinator'

type Props = {
  models: LocalModel[]
  realtimeAvailable: boolean
  /** Voice model chosen in the top bar; empty when none is installed. */
  modelId: string
  /** Whether the browser can capture and encode audio at all. */
  audioSupported: boolean
  /**
   * Whether the daemon can transcribe. PersonaPlex reports only its own speech,
   * so without ASR a spoken turn never becomes text and nothing reaches the
   * conversation — except with `neither`, which needs no transcript at all.
   */
  asrAvailable: boolean
  persona: string
  onPersonaChange: (persona: string) => void
  /** The shared conversation. Voice turns land in it beside typed ones. */
  session: SessionCoordinatorHandle
  onError: (message: string | null) => void
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

/** What a live voice session can be wired to, and what each choice means. */
const TARGETS: Array<[VoiceSessionTarget, string, string]> = [
  [
    'both',
    'Both',
    'Spoken turns join this conversation and reach the agent when a task is bound to it, or the chat model when none is.'
  ],
  [
    'agent',
    'Agent',
    'Spoken turns always go to the bound agent session. With no task bound, a turn is refused rather than answered without the workspace and tools you expected.'
  ],
  [
    'chat',
    'Chat',
    'Spoken turns always go to the chat model, even while an agent task is running.'
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

  const needsTranscripts = config.voiceSessionTarget !== 'neither'
  const live = snapshot.voiceStatus === 'live'
  const starting = snapshot.voiceStatus === 'starting' || busy
  const selected = props.models.find((model) => model.id === props.modelId)
  const task = snapshot.task
  const speaking = snapshot.speakingCorrelationId !== null
  const working = snapshot.activeCorrelationId !== null

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
        ? 'Live · microphone muted'
        : speaking
          ? 'Live · speaking'
          : working
            ? 'Live · working on your request'
            : 'Live · speak whenever you like'
      : snapshot.voiceStatus === 'error'
        ? `Voice mode stopped: ${snapshot.voiceError ?? 'unknown error'}`
        : 'Not connected'

  return (
    <section className="voice-bar">
      <header className="mode-panel-header">
        <h2>Voice</h2>
        <p>
          Speech and typing share this conversation. What you say is transcribed into it, and the
          agent stays in charge of tools, tasks, and results.
        </p>
      </header>

      {!props.realtimeAvailable ? (
        <p className="mode-empty">
          Realtime voice needs a PersonaPlex runtime and a downloaded <code>personaplex:</code>{' '}
          model. On Apple Silicon build PersonaPlex MLX from Manage → Runtimes (accept the
          nvidia/personaplex-7b-v1 license and set an HF token); on Linux CUDA build PersonaPlex /
          Moshi.
        </p>
      ) : null}
      {props.realtimeAvailable && !props.audioSupported ? (
        <p className="mode-empty">
          This build has no WebCodecs Opus support, which realtime voice needs for audio in and out.
        </p>
      ) : null}
      {props.realtimeAvailable && !props.asrAvailable && needsTranscripts ? (
        <p className="mode-empty">
          Connecting voice to a conversation needs transcription, which is separate from PersonaPlex:
          the voice model reports only what <em>it</em> says, so what <em>you</em> say is transcribed
          by Whisper. Build WhisperKit under Manage → Runtimes, or download a Whisper model from
          Discover if you already built whisper.cpp. Until then, use <strong>Neither</strong>, which
          talks to PersonaPlex directly and needs no transcript.
        </p>
      ) : null}

      <div className="voice-controls">
        <label>
          Persona
          <textarea
            value={props.persona}
            onChange={(event) => props.onPersonaChange(event.target.value)}
            rows={2}
            disabled={live || starting}
            placeholder="Describe who the model should be…"
          />
        </label>

        <div className="voice-actions">
          {!live ? (
            <button
              type="button"
              className="primary"
              disabled={
                starting ||
                !props.realtimeAvailable ||
                !props.audioSupported ||
                (needsTranscripts && !props.asrAvailable)
              }
              onClick={() => void guard(() => session.startVoice())}
            >
              {starting ? <LoaderCircle className="spin" size={16} /> : <Mic size={16} />}
              Start conversation
            </button>
          ) : (
            <button type="button" className={muted ? 'toggled' : ''} onClick={toggleMute}>
              {muted ? <MicOff size={16} /> : <Mic size={16} />}
              {muted ? 'Unmute' : 'Mute'}
            </button>
          )}
          {/* Three separate controls on purpose: silencing the voice, dropping
              this answer, and abandoning the task are different decisions. */}
          <button
            type="button"
            disabled={!speaking}
            title="Stop the audio. The task keeps running."
            onClick={() => void guard(() => session.stopSpeaking())}
          >
            <VolumeX size={16} /> Stop speaking
          </button>
          <button
            type="button"
            className="danger"
            disabled={!working}
            title="Cancel the agent task. Anything it already answered stays in the conversation."
            onClick={() => void guard(() => session.cancelAgentTask())}
          >
            <Square size={14} fill="currentColor" /> Cancel task
          </button>
          {live ? (
            <button
              type="button"
              className="danger"
              disabled={starting}
              title="Turn voice mode off. The conversation and any running task stay."
              onClick={() => void guard(() => session.endVoice())}
            >
              <PhoneOff size={16} /> End voice
            </button>
          ) : null}
        </div>

        <div className={`voice-status ${live ? 'live' : ''}`}>
          <span className="voice-status-dot" />
          {statusLabel}
          {props.modelId ? (
            <span className="voice-status-model">
              {modelDisplayName(props.modelId, selected).title}
            </span>
          ) : null}
        </div>

        {live && !session.canSpeak && config.voiceSessionTarget !== 'neither' ? (
          <p className="voice-notice">
            <AlertTriangle size={13} /> This host has no speech synthesizer, so answers are shown
            rather than spoken. PersonaPlex still replies in its own voice, but only what is written
            in the conversation is authoritative.
          </p>
        ) : null}

        {snapshot.queue.length > 0 ? (
          <p className="voice-notice">
            {snapshot.queue.length} turn{snapshot.queue.length === 1 ? '' : 's'} queued behind the
            one running.
          </p>
        ) : null}

        {task ? (
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

        <div className="voice-meters">
          <div className="voice-meter-row">
            <Mic size={13} />
            <div className="voice-meter">
              <div
                className="voice-meter-fill"
                style={{ width: `${Math.round(session.inputLevel * 100)}%` }}
              />
            </div>
          </div>
          <div className="voice-meter-row">
            <Volume2 size={13} />
            <div className="voice-meter">
              <div
                className="voice-meter-fill model"
                style={{ width: `${Math.round(session.outputLevel * 100)}%` }}
              />
            </div>
          </div>
        </div>

        <div className="voice-target">
          <span className="section-label">Connected to</span>
          <div className="voice-target-choices" role="radiogroup" aria-label="Connected session">
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

        <div className="voice-options">
          <label>
            <input
              type="checkbox"
              checked={config.speakTextOriginatedResponses}
              onChange={(event) =>
                session.setConfig({
                  ...config,
                  speakTextOriginatedResponses: event.target.checked
                })
              }
            />
            Speak answers to typed questions too
          </label>
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

        {config.showVoiceTranscripts && snapshot.voiceModelText ? (
          <details className="voice-model-text">
            <summary>
              <Bot size={12} /> What PersonaPlex said on its own
            </summary>
            <p>{snapshot.voiceModelText.slice(-600)}</p>
            <small>
              Not part of the conversation and not checked against anything. Only the messages above
              are authoritative.
            </small>
          </details>
        ) : null}
      </div>
    </section>
  )
}

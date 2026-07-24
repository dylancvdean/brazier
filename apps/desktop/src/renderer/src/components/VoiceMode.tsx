import { LoaderCircle, Mic, MicOff, PhoneOff, Volume2 } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createVoiceSession,
  endVoiceSession,
  getVoiceSession,
  type LocalModel,
  type RuntimeSettings,
  type VoiceSessionInfo
} from '../api'
import { VoiceStream, voiceStreamSupported } from '../audio/voiceStream'
import { modelDisplayName } from '../model-utils'

type Props = {
  models: LocalModel[]
  settings: RuntimeSettings | null
  realtimeAvailable: boolean
  /** Voice model chosen in the top bar; empty when none is installed. */
  modelId: string
  onError: (message: string | null) => void
}

type Phase = 'idle' | 'starting' | 'connecting' | 'live'

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

export function VoiceMode(props: Props) {
  const [persona, setPersona] = useState(
    props.settings?.default_voice_persona ?? 'You are a helpful assistant.'
  )
  const personaEdited = useRef(false)
  const [session, setSession] = useState<VoiceSessionInfo | null>(null)
  const [phase, setPhase] = useState<Phase>('idle')
  const [muted, setMuted] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [inputLevel, setInputLevel] = useState(0)
  const [outputLevel, setOutputLevel] = useState(0)
  const streamRef = useRef<VoiceStream | null>(null)
  const onError = props.onError
  const selected = props.models.find((model) => model.id === props.modelId)
  const supported = voiceStreamSupported()

  // Engine settings arrive after the first render; adopt the saved persona
  // unless the field has already been typed in.
  useEffect(() => {
    const saved = props.settings?.default_voice_persona
    if (saved && !personaEdited.current) setPersona(saved)
  }, [props.settings?.default_voice_persona])

  useEffect(() => {
    void getVoiceSession()
      .then((response) => setSession(response.session))
      .catch(() => setSession(null))
    return () => {
      void streamRef.current?.stop()
      streamRef.current = null
    }
  }, [])

  const connectAudio = useCallback(
    async (target: VoiceSessionInfo): Promise<void> => {
      setPhase('connecting')
      const stream = new VoiceStream({
        onText: (text) => setTranscript((current) => current + text),
        onInputLevel: setInputLevel,
        onOutputLevel: setOutputLevel,
        onError: (message) => onError(message),
        onState: (state) => {
          if (state === 'live') setPhase('live')
          if (state === 'closed') {
            setPhase((current) => (current === 'idle' ? current : 'idle'))
            setInputLevel(0)
            setOutputLevel(0)
          }
        }
      })
      streamRef.current = stream
      try {
        await stream.start(target.ws_url)
        setMuted(false)
      } catch (cause) {
        await stream.stop()
        streamRef.current = null
        setPhase('idle')
        throw cause
      }
    },
    [onError]
  )

  async function startConversation(): Promise<void> {
    setPhase('starting')
    onError(null)
    setTranscript('')
    try {
      const created =
        session ??
        (await createVoiceSession({
          model_id: props.modelId || undefined,
          persona_text: persona.trim() || undefined
        }))
      setSession(created)
      await connectAudio(created)
    } catch (cause) {
      setPhase('idle')
      onError(errorText(cause))
    }
  }

  async function endConversation(): Promise<void> {
    onError(null)
    const active = session
    await streamRef.current?.stop()
    streamRef.current = null
    setPhase('idle')
    setSession(null)
    if (!active) return
    try {
      await endVoiceSession(active.id)
    } catch (cause) {
      onError(errorText(cause))
    }
  }

  function toggleMute(): void {
    const next = !muted
    setMuted(next)
    streamRef.current?.setMuted(next)
  }

  const live = phase === 'live'
  const busy = phase === 'starting' || phase === 'connecting'
  const statusLabel =
    phase === 'starting'
      ? 'Loading the voice model — first run takes a minute…'
      : phase === 'connecting'
        ? 'Connecting audio…'
        : live
          ? muted
            ? 'Live · microphone muted'
            : 'Live · speak whenever you like'
          : 'Not connected'

  return (
    <section className="mode-panel voice-mode">
      <header className="mode-panel-header">
        <h2>Voice</h2>
        <p>Full-duplex speech with PersonaPlex. Use headphones — the model hears your speakers.</p>
      </header>

      {!props.realtimeAvailable ? (
        <p className="mode-empty">
          Realtime voice needs a PersonaPlex runtime and a downloaded <code>personaplex:</code>{' '}
          model. On Apple Silicon build PersonaPlex MLX from Manage → Runtimes (accept the
          nvidia/personaplex-7b-v1 license and set an HF token); on Linux CUDA build PersonaPlex /
          Moshi.
        </p>
      ) : null}
      {props.realtimeAvailable && !supported ? (
        <p className="mode-empty">
          This build has no WebCodecs Opus support, which realtime voice needs for audio in and out.
        </p>
      ) : null}

      <div className="voice-controls">
        <label>
          Persona
          <textarea
            value={persona}
            onChange={(event) => {
              personaEdited.current = true
              setPersona(event.target.value)
            }}
            rows={3}
            disabled={live || busy}
            placeholder="Describe who the model should be…"
          />
        </label>

        <div className="voice-actions">
          {!live ? (
            <button
              type="button"
              className="primary"
              disabled={busy || !props.realtimeAvailable || !supported}
              onClick={() => void startConversation()}
            >
              {busy ? <LoaderCircle className="spin" size={16} /> : <Mic size={16} />}
              {session && phase === 'idle' ? 'Reconnect audio' : 'Start conversation'}
            </button>
          ) : (
            <button type="button" className={muted ? 'toggled' : ''} onClick={toggleMute}>
              {muted ? <MicOff size={16} /> : <Mic size={16} />}
              {muted ? 'Unmute' : 'Mute'}
            </button>
          )}
          {session || live ? (
            <button type="button" className="danger" disabled={phase === 'starting'} onClick={() => void endConversation()}>
              <PhoneOff size={16} /> End conversation
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

        <div className="voice-meters">
          <div className="voice-meter-row">
            <Mic size={13} />
            <div className="voice-meter">
              <div className="voice-meter-fill" style={{ width: `${Math.round(inputLevel * 100)}%` }} />
            </div>
          </div>
          <div className="voice-meter-row">
            <Volume2 size={13} />
            <div className="voice-meter">
              <div
                className="voice-meter-fill model"
                style={{ width: `${Math.round(outputLevel * 100)}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="voice-transcript">
        {transcript ? (
          <p>{transcript}</p>
        ) : (
          <p className="mode-empty">
            What the model says appears here as it speaks.
          </p>
        )}
      </div>
    </section>
  )
}

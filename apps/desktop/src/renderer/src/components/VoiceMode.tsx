import { LoaderCircle, Mic, MicOff, PhoneOff } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import {
  createVoiceSession,
  endVoiceSession,
  getVoiceSession,
  type LocalModel,
  type RuntimeSettings,
  type VoiceSessionInfo
} from '../api'
import { isVoiceModel, modelDisplayName } from '../model-utils'

type Props = {
  models: LocalModel[]
  settings: RuntimeSettings | null
  realtimeAvailable: boolean
  onError: (message: string | null) => void
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

export function VoiceMode(props: Props) {
  const voiceModels = props.models.filter((model) => isVoiceModel(model))
  const [persona, setPersona] = useState(
    props.settings?.default_voice_persona ?? 'You are a helpful assistant.'
  )
  const [modelId, setModelId] = useState(
    props.settings?.default_voice_model ?? voiceModels[0]?.id ?? ''
  )
  const [session, setSession] = useState<VoiceSessionInfo | null>(null)
  const [busy, setBusy] = useState(false)
  const [listening, setListening] = useState(false)
  const [transcript, setTranscript] = useState<string[]>([])
  const [level, setLevel] = useState(0)
  const wsRef = useRef<WebSocket | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const mediaRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)

  useEffect(() => {
    void getVoiceSession()
      .then((response) => setSession(response.session))
      .catch(() => setSession(null))
    return () => {
      teardownAudio()
      wsRef.current?.close()
    }
  }, [])

  function teardownAudio(): void {
    processorRef.current?.disconnect()
    processorRef.current = null
    mediaRef.current?.getTracks().forEach((track) => track.stop())
    mediaRef.current = null
    void audioCtxRef.current?.close()
    audioCtxRef.current = null
    setListening(false)
    setLevel(0)
  }

  async function startSession(): Promise<void> {
    setBusy(true)
    props.onError(null)
    try {
      const created = await createVoiceSession({
        model_id: modelId || undefined,
        persona_text: persona.trim() || undefined
      })
      setSession(created)
      setTranscript([])
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setBusy(false)
    }
  }

  async function stopSession(): Promise<void> {
    if (!session) return
    setBusy(true)
    props.onError(null)
    try {
      teardownAudio()
      wsRef.current?.close()
      wsRef.current = null
      await endVoiceSession(session.id)
      setSession(null)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setBusy(false)
    }
  }

  async function connectMic(): Promise<void> {
    if (!session?.ws_url) return
    props.onError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaRef.current = stream
      const audioCtx = new AudioContext({ sampleRate: 24000 })
      audioCtxRef.current = audioCtx
      const source = audioCtx.createMediaStreamSource(stream)
      const processor = audioCtx.createScriptProcessor(4096, 1, 1)
      processorRef.current = processor
      const ws = new WebSocket(session.ws_url)
      ws.binaryType = 'arraybuffer'
      wsRef.current = ws

      ws.onopen = () => setListening(true)
      ws.onerror = () => props.onError('Voice WebSocket error')
      ws.onclose = () => {
        teardownAudio()
      }
      ws.onmessage = (event) => {
        if (!(event.data instanceof ArrayBuffer) || event.data.byteLength < 1) return
        const bytes = new Uint8Array(event.data)
        const tag = bytes[0]
        if (tag === 0x02) {
          const text = new TextDecoder().decode(bytes.slice(1))
          if (text.trim()) setTranscript((current) => [...current.slice(-40), text])
        } else if (tag === 0x01) {
          // Opus playback would need a decoder; show activity instead for v1.
          setLevel((current) => Math.min(1, current * 0.6 + 0.4))
        }
      }

      processor.onaudioprocess = (event) => {
        if (ws.readyState !== WebSocket.OPEN) return
        const input = event.inputBuffer.getChannelData(0)
        let sum = 0
        for (let i = 0; i < input.length; i += 1) sum += input[i] * input[i]
        setLevel(Math.min(1, Math.sqrt(sum / input.length) * 4))
        // Raw PCM float32 framed as Moshi audio would need Opus encode.
        // Send a tagged PCM16 payload as a best-effort bridge for local testing.
        const pcm = new Int16Array(input.length)
        for (let i = 0; i < input.length; i += 1) {
          const sample = Math.max(-1, Math.min(1, input[i]))
          pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff
        }
        const frame = new Uint8Array(1 + pcm.byteLength)
        frame[0] = 0x01
        frame.set(new Uint8Array(pcm.buffer), 1)
        ws.send(frame.buffer)
      }
      source.connect(processor)
      processor.connect(audioCtx.destination)
    } catch (cause) {
      props.onError(errorText(cause))
      teardownAudio()
    }
  }

  return (
    <section className="mode-panel voice-mode">
      <header className="mode-panel-header">
        <h2>Voice</h2>
        <p>Full-duplex speech with PersonaPlex over the Moshi protocol.</p>
      </header>

      {!props.realtimeAvailable ? (
        <p className="mode-empty">
          Realtime voice needs a PersonaPlex runtime and usually a downloaded
          `personaplex:` model. On Apple Silicon build PersonaPlex MLX from Manage →
          Runtimes (accept the nvidia/personaplex-7b-v1 license and set an HF token); on
          Linux CUDA build PersonaPlex / Moshi. Then set a default voice model in Manage →
          Engine if you want a local snapshot.
        </p>
      ) : null}

      <div className="voice-controls">
        <label>
          Model
          <select
            value={modelId}
            onChange={(event) => setModelId(event.target.value)}
            disabled={Boolean(session)}
          >
            {voiceModels.length === 0 ? (
              <option value="">No PersonaPlex models installed</option>
            ) : (
              voiceModels.map((model) => {
                const names = modelDisplayName(model.id, model)
                return (
                  <option key={model.id} value={model.id}>
                    {names.title}
                  </option>
                )
              })
            )}
          </select>
        </label>
        <label>
          Persona
          <textarea
            value={persona}
            onChange={(event) => setPersona(event.target.value)}
            rows={3}
            disabled={Boolean(session)}
          />
        </label>

        <div className="voice-actions">
          {!session ? (
            <button
              type="button"
              className="primary"
              disabled={busy || !props.realtimeAvailable}
              onClick={() => void startSession()}
            >
              {busy ? <LoaderCircle className="spin" size={16} /> : <Mic size={16} />}
              Start session
            </button>
          ) : (
            <>
              {!listening ? (
                <button type="button" className="primary" onClick={() => void connectMic()}>
                  <Mic size={16} /> Connect microphone
                </button>
              ) : (
                <button type="button" onClick={() => teardownAudio()}>
                  <MicOff size={16} /> Mute mic
                </button>
              )}
              <button type="button" className="danger" disabled={busy} onClick={() => void stopSession()}>
                <PhoneOff size={16} /> End session
              </button>
            </>
          )}
        </div>

        <div className="voice-meter" aria-hidden>
          <div className="voice-meter-fill" style={{ width: `${Math.round(level * 100)}%` }} />
        </div>
      </div>

      {session ? (
        <p className="voice-session-meta">
          Session {session.id.slice(0, 8)} · {session.ws_url}
        </p>
      ) : null}

      <div className="voice-transcript">
        {transcript.length === 0 ? (
          <p className="mode-empty">Transcript tokens from the model will appear here.</p>
        ) : (
          transcript.map((line, index) => <p key={`${index}-${line.slice(0, 12)}`}>{line}</p>)
        )}
      </div>
    </section>
  )
}

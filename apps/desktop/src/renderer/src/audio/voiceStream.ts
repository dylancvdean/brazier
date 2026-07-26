/**
 * Full-duplex audio transport for a PersonaPlex/Moshi voice session.
 *
 * The server speaks Ogg-encapsulated Opus in both directions over a binary
 * WebSocket (see `oggOpus.ts`), tagging every frame with its kind:
 * `0x00` handshake, `0x01` audio, `0x02` transcript text. WebCodecs does the
 * Opus work, two AudioWorklets move samples on the audio thread, and this
 * class glues them together so the UI only sees levels, text, and state.
 */
import captureWorkletUrl from './captureWorklet.js?url'
import { OggOpusDemuxer, OggOpusMuxer } from './oggOpus'
import playbackWorkletUrl from './playbackWorklet.js?url'

const TAG_HANDSHAKE = 0x00
const TAG_AUDIO = 0x01
const TAG_TEXT = 0x02

/** Server sample rate; frames are 20 ms, matching the reference web client. */
const SAMPLE_RATE = 24000
const FRAME_SAMPLES = 480

export type VoiceStreamState = 'connecting' | 'live' | 'closed'

export type VoiceStreamHandlers = {
  onText?: (text: string) => void
  onInputLevel?: (level: number) => void
  onOutputLevel?: (level: number) => void
  onState?: (state: VoiceStreamState) => void
  onError?: (message: string) => void
  /**
   * Captured microphone frames, before encoding. The shared-conversation mode
   * transcribes these itself: the server's text frames are the model's own
   * speech, not the user's.
   */
  onCaptureFrame?: (samples: Float32Array, sampleRate: number) => void
}

/** Whether this runtime can encode and decode Opus (WebCodecs + worklets). */
export function voiceStreamSupported(): boolean {
  return (
    typeof AudioEncoder !== 'undefined' &&
    typeof AudioDecoder !== 'undefined' &&
    typeof AudioData !== 'undefined' &&
    typeof navigator !== 'undefined' &&
    Boolean(navigator.mediaDevices?.getUserMedia)
  )
}

function rms(samples: Float32Array): number {
  let sum = 0
  for (let index = 0; index < samples.length; index += 1) sum += samples[index] * samples[index]
  return Math.sqrt(sum / Math.max(1, samples.length))
}

export class VoiceStream {
  private readonly handlers: VoiceStreamHandlers
  private socket: WebSocket | null = null
  private captureCtx: AudioContext | null = null
  private playbackCtx: AudioContext | null = null
  private playbackNode: AudioWorkletNode | null = null
  private playbackReady: Promise<void> | null = null
  private media: MediaStream | null = null
  private encoder: AudioEncoder | null = null
  private decoder: AudioDecoder | null = null
  private muxer = new OggOpusMuxer({ sampleRate: SAMPLE_RATE, channelCount: 1 })
  private demuxer = new OggOpusDemuxer()
  private timestamp = 0
  private decodeTimestamp = 0
  private muted = false
  private stopped = false
  private outputOpen = true

  constructor(handlers: VoiceStreamHandlers = {}) {
    this.handlers = handlers
  }

  /** Open the socket, start capture, and begin streaming both ways. */
  async start(wsUrl: string): Promise<void> {
    if (!voiceStreamSupported()) {
      throw new Error('This build lacks the WebCodecs Opus support realtime voice needs.')
    }
    this.handlers.onState?.('connecting')
    // Ask for the browser's echo canceller: the server has none, so without it
    // the model hears itself through the speakers and talks over the user.
    this.media = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1
      }
    })
    await this.openSocket(wsUrl)
    await this.startCapture()
  }

  private openSocket(wsUrl: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(wsUrl)
      socket.binaryType = 'arraybuffer'
      this.socket = socket
      const failed = (message: string) => reject(new Error(message))
      socket.onopen = () => {
        // Opus headers must lead the stream before any audio page.
        socket.send(withTag(TAG_AUDIO, this.muxer.headerPages()))
        this.handlers.onState?.('live')
        resolve()
      }
      socket.onerror = () => {
        if (socket.readyState === WebSocket.CONNECTING) failed('Could not reach the voice server.')
        else this.handlers.onError?.('Voice connection error.')
      }
      socket.onclose = (event) => {
        if (socket.readyState === WebSocket.CONNECTING) failed('The voice server closed the connection.')
        this.handlers.onState?.('closed')
        if (!this.stopped && !event.wasClean) {
          this.handlers.onError?.(
            'The voice server dropped the connection. Check the PersonaPlex process log.'
          )
        }
        void this.stop()
      }
      socket.onmessage = (event) => this.onMessage(event)
    })
  }

  private onMessage(event: MessageEvent): void {
    if (!(event.data instanceof ArrayBuffer) || event.data.byteLength < 1) return
    const bytes = new Uint8Array(event.data)
    switch (bytes[0]) {
      case TAG_TEXT: {
        const text = new TextDecoder().decode(bytes.subarray(1))
        if (text) this.handlers.onText?.(text)
        return
      }
      case TAG_AUDIO:
        for (const packet of this.demuxer.push(bytes.subarray(1))) {
          this.decodePacket(packet)
        }
        return
      // The handshake carries no payload, and unknown tags are ignored.
      case TAG_HANDSHAKE:
      default:
        return
    }
  }

  private decodePacket(packet: Uint8Array): void {
    const decoder = this.ensureDecoder()
    if (!decoder || decoder.state !== 'configured') return
    decoder.decode(
      new EncodedAudioChunk({
        type: 'key',
        timestamp: this.decodeTimestamp,
        data: packet
      })
    )
    // Opus packets from the server are 40 ms; an approximate clock is enough
    // because playback is paced by the ring buffer, not these timestamps.
    this.decodeTimestamp += 40_000
  }

  private ensureDecoder(): AudioDecoder | null {
    if (this.decoder) return this.decoder
    const decoder = new AudioDecoder({
      output: (data) => this.playDecoded(data),
      error: (cause) => this.handlers.onError?.(`Audio decode failed: ${cause.message}`)
    })
    const head = this.demuxer.opusHead
    try {
      decoder.configure({
        codec: 'opus',
        sampleRate: 48000,
        numberOfChannels: 1,
        ...(head ? { description: head } : {})
      })
    } catch (cause) {
      this.handlers.onError?.(`Could not configure the Opus decoder: ${errorText(cause)}`)
      return null
    }
    this.decoder = decoder
    return decoder
  }

  private playDecoded(data: AudioData): void {
    const samples = new Float32Array(data.numberOfFrames)
    // Read the rate before closing: a closed AudioData reports zero.
    const sampleRate = data.sampleRate
    try {
      data.copyTo(samples, { planeIndex: 0, format: 'f32-planar' })
    } finally {
      data.close()
    }
    // Decoding continues while the gate is shut so the stream clock and the
    // text frames stay intact; only the audio is dropped.
    if (!this.outputOpen) {
      this.handlers.onOutputLevel?.(0)
      return
    }
    this.handlers.onOutputLevel?.(Math.min(1, rms(samples) * 4))
    void this.pushPlayback(samples, sampleRate)
  }

  private async pushPlayback(samples: Float32Array, sampleRate: number): Promise<void> {
    if (this.stopped) return
    // Frames arrive faster than the graph can be built, so the first caller
    // owns setup and the rest await the same promise.
    this.playbackReady ??= this.startPlayback(sampleRate)
    await this.playbackReady
    if (this.stopped) return
    this.playbackNode?.port.postMessage(samples)
  }

  private async startPlayback(sampleRate: number): Promise<void> {
    // Match the decoder's output rate so no resampling is needed.
    const context = new AudioContext({ sampleRate })
    this.playbackCtx = context
    await context.audioWorklet.addModule(playbackWorkletUrl)
    if (this.stopped) return
    const node = new AudioWorkletNode(context, 'brazier-playback', {
      numberOfInputs: 0,
      outputChannelCount: [1],
      processorOptions: { capacity: sampleRate * 4 }
    })
    node.connect(context.destination)
    this.playbackNode = node
    await context.resume().catch(() => {})
  }

  private async startCapture(): Promise<void> {
    const context = new AudioContext({ sampleRate: SAMPLE_RATE })
    this.captureCtx = context
    await context.audioWorklet.addModule(captureWorkletUrl)
    if (this.stopped) return

    this.encoder = new AudioEncoder({
      output: (chunk) => this.sendEncoded(chunk),
      error: (cause) => this.handlers.onError?.(`Audio encode failed: ${cause.message}`)
    })
    this.encoder.configure({
      codec: 'opus',
      sampleRate: SAMPLE_RATE,
      numberOfChannels: 1,
      bitrate: 24000,
      opus: { frameDuration: 20_000 }
    })

    const source = context.createMediaStreamSource(this.media!)
    const node = new AudioWorkletNode(context, 'brazier-capture', {
      outputChannelCount: [1],
      processorOptions: { frameSize: FRAME_SAMPLES }
    })
    node.port.onmessage = (event) => this.encodeFrame(event.data as Float32Array<ArrayBuffer>)
    source.connect(node)
    // The processor writes no output; the silent sink is only there to put the
    // node in the graph that reaches the destination, which is what guarantees
    // it is rendered at all. A capture-only node with nowhere to go depends on
    // the implementation choosing to pull it.
    const sink = context.createGain()
    sink.gain.value = 0
    node.connect(sink)
    sink.connect(context.destination)

    // A suspended context renders nothing: no frames, no meter, and no audio
    // reaching the model, which looks exactly like a microphone that works
    // while the model talks to itself. Playback already resumed; capture did
    // not, so it never ran.
    await context.resume().catch(() => {})
    if (context.state !== 'running') {
      this.handlers.onError?.(
        `The audio input did not start (context is ${context.state}). Check the microphone permission.`
      )
    }
  }

  private encodeFrame(samples: Float32Array<ArrayBuffer>): void {
    this.handlers.onInputLevel?.(this.muted ? 0 : Math.min(1, rms(samples) * 4))
    if (!this.muted) this.handlers.onCaptureFrame?.(samples, SAMPLE_RATE)
    if (this.muted || !this.encoder || this.encoder.state !== 'configured') return
    // Muting still advances the clock, so the model hears a gap rather than
    // a jump cut when the microphone comes back.
    const data = new AudioData({
      format: 'f32-planar',
      sampleRate: SAMPLE_RATE,
      numberOfFrames: samples.length,
      numberOfChannels: 1,
      timestamp: this.timestamp,
      data: samples
    })
    this.timestamp += Math.round((samples.length * 1_000_000) / SAMPLE_RATE)
    try {
      this.encoder.encode(data)
    } finally {
      data.close()
    }
  }

  private sendEncoded(chunk: EncodedAudioChunk): void {
    const socket = this.socket
    if (!socket || socket.readyState !== WebSocket.OPEN) return
    const packet = new Uint8Array(chunk.byteLength)
    chunk.copyTo(packet)
    const page = this.muxer.addPacket(packet, FRAME_SAMPLES)
    if (page) socket.send(withTag(TAG_AUDIO, page))
  }

  /**
   * Open or shut the model's audio output.
   *
   * PersonaPlex is a speech-to-speech model: it answers on its own, and it
   * cannot be told to hold a thought. When the agent owns a turn, its audio is
   * shut off and the authoritative answer is spoken instead, so the user never
   * hears two different replies to one question.
   */
  setOutputGate(open: boolean): void {
    this.outputOpen = open
    if (!open) {
      this.playbackNode?.port.postMessage('flush')
      this.handlers.onOutputLevel?.(0)
    }
  }

  /** Stop sending microphone audio without dropping the session. */
  setMuted(muted: boolean): void {
    this.muted = muted
    for (const track of this.media?.getAudioTracks() ?? []) {
      track.enabled = !muted
    }
    if (muted) this.handlers.onInputLevel?.(0)
  }

  /** Tear down the socket, capture graph, and codecs. Safe to call twice. */
  async stop(): Promise<void> {
    if (this.stopped) return
    this.stopped = true
    const socket = this.socket
    this.socket = null
    if (socket && socket.readyState <= WebSocket.OPEN) socket.close()
    for (const track of this.media?.getAudioTracks() ?? []) track.stop()
    this.media = null
    if (this.encoder && this.encoder.state !== 'closed') this.encoder.close()
    this.encoder = null
    if (this.decoder && this.decoder.state !== 'closed') this.decoder.close()
    this.decoder = null
    this.playbackNode?.port.postMessage('flush')
    this.playbackNode = null
    this.playbackReady = null
    await this.captureCtx?.close().catch(() => {})
    this.captureCtx = null
    await this.playbackCtx?.close().catch(() => {})
    this.playbackCtx = null
    this.handlers.onInputLevel?.(0)
    this.handlers.onOutputLevel?.(0)
    this.handlers.onState?.('closed')
  }
}

function withTag(tag: number, payload: Uint8Array): ArrayBuffer {
  const frame = new Uint8Array(payload.length + 1)
  frame[0] = tag
  frame.set(payload, 1)
  return frame.buffer
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

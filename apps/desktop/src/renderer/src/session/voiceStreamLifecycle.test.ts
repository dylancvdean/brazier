import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { VoiceStream } from '../audio/voiceStream'

class FakeMediaStreamTrack {
  enabled = true
  muted = false
  readyState: 'live' | 'ended' = 'live'
  stopped = false
  private readonly listeners = new Map<string, Set<() => void>>()

  stop(): void {
    this.stopped = true
    this.readyState = 'ended'
  }

  addEventListener(type: string, cb: () => void): void {
    let set = this.listeners.get(type)
    if (!set) {
      set = new Set()
      this.listeners.set(type, set)
    }
    set.add(cb)
  }

  removeEventListener(type: string, cb: () => void): void {
    this.listeners.get(type)?.delete(cb)
  }

  emit(type: string): void {
    for (const cb of [...(this.listeners.get(type) ?? [])]) cb()
  }

  listenerCount(type: string): number {
    return this.listeners.get(type)?.size ?? 0
  }
}

class FakeWebSocket {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3
  static instances: FakeWebSocket[] = []

  readyState = FakeWebSocket.CONNECTING
  binaryType = 'blob'
  readonly sent: ArrayBuffer[] = []
  closedCount = 0
  onopen: (() => void) | null = null
  onerror: (() => void) | null = null
  onclose: ((event: { wasClean: boolean }) => void) | null = null
  onmessage: ((event: { data: unknown }) => void) | null = null

  constructor(
    public readonly url: string,
    public readonly protocols?: string[]
  ) {
    FakeWebSocket.instances.push(this)
  }

  open(): void {
    this.readyState = FakeWebSocket.OPEN
    this.onopen?.()
  }

  send(data: ArrayBuffer): void {
    this.sent.push(data)
  }

  close(): void {
    this.closedCount += 1
    this.readyState = FakeWebSocket.CLOSED
  }
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = []
  state: 'suspended' | 'running' | 'closed' = 'suspended'
  closedCount = 0
  readonly destination = {}
  readonly audioWorklet = {
    addModule: vi.fn(async () => undefined)
  }

  constructor() {
    FakeAudioContext.instances.push(this)
  }

  createMediaStreamSource(): { connect: () => void } {
    return { connect: () => undefined }
  }

  createGain(): { gain: { value: number }; connect: () => void } {
    return { gain: { value: 1 }, connect: () => undefined }
  }

  async resume(): Promise<void> {
    if (this.state !== 'closed') this.state = 'running'
  }

  async close(): Promise<void> {
    this.closedCount += 1
    this.state = 'closed'
  }
}

class FakeAudioWorkletNode {
  readonly port = {
    onmessage: null as ((event: { data: unknown }) => void) | null,
    postMessage: vi.fn()
  }
  connect(): void {}
  disconnect(): void {}
}

class FakeAudioEncoder {
  state: 'unconfigured' | 'configured' | 'closed' = 'unconfigured'
  flushedCount = 0
  private readonly onOutput: (chunk: { byteLength: number; copyTo: (dst: Uint8Array) => void }) => void

  constructor(init: {
    output: (chunk: { byteLength: number; copyTo: (dst: Uint8Array) => void }) => void
    error: (err: Error) => void
  }) {
    this.onOutput = init.output
  }

  configure(): void {
    this.state = 'configured'
  }

  encode(): void {
    const payload = new Uint8Array([1, 2, 3, 4])
    this.onOutput({
      byteLength: payload.byteLength,
      copyTo: (dst) => dst.set(payload)
    })
  }

  async flush(): Promise<void> {
    this.flushedCount += 1
  }

  close(): void {
    this.state = 'closed'
  }
}

describe('VoiceStream lifecycle', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
    FakeAudioContext.instances = []
    vi.stubGlobal('WebSocket', FakeWebSocket)
    vi.stubGlobal('AudioContext', FakeAudioContext)
    vi.stubGlobal('AudioWorkletNode', FakeAudioWorkletNode)
    vi.stubGlobal('AudioEncoder', FakeAudioEncoder)
    vi.stubGlobal('AudioDecoder', class {})
    vi.stubGlobal(
      'AudioData',
      class {
        close(): void {}
      }
    )
    vi.stubGlobal('window', {
      setTimeout: globalThis.setTimeout.bind(globalThis),
      clearTimeout: globalThis.clearTimeout.bind(globalThis)
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('cleans up microphone tracks when stop() is called while getUserMedia() is pending', async () => {
    const track = new FakeMediaStreamTrack()
    let resolveMedia!: (media: { getAudioTracks: () => FakeMediaStreamTrack[] }) => void
    vi.stubGlobal('navigator', {
      mediaDevices: {
        getUserMedia: vi.fn(
          () =>
            new Promise((resolve) => {
              resolveMedia = resolve
            })
        )
      }
    })

    const stream = new VoiceStream()
    const starting = stream.start('ws://127.0.0.1:7614/voice')
    await stream.stop()

    resolveMedia({ getAudioTracks: () => [track] })
    await starting

    expect(track.stopped).toBe(true)
    expect(FakeWebSocket.instances).toHaveLength(0)
    expect(FakeAudioContext.instances).toHaveLength(0)
  })

  it('closes socket and stops tracks when stop() is called while openSocket() is pending', async () => {
    const track = new FakeMediaStreamTrack()
    vi.stubGlobal('navigator', {
      mediaDevices: {
        getUserMedia: vi.fn(async () => ({ getAudioTracks: () => [track] }))
      }
    })

    const stream = new VoiceStream()
    const starting = stream.start('ws://127.0.0.1:7614/voice')
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))

    const ws = FakeWebSocket.instances[0]
    await stream.stop()
    ws.open()
    await starting

    expect(track.stopped).toBe(true)
    expect(ws.readyState).toBe(FakeWebSocket.CLOSED)
    expect(FakeAudioContext.instances).toHaveLength(0)
  })

  it('registers capture track listeners once and flushes EOS only once on stop()', async () => {
    const track = new FakeMediaStreamTrack()
    vi.stubGlobal('navigator', {
      mediaDevices: {
        getUserMedia: vi.fn(async () => ({ getAudioTracks: () => [track] }))
      }
    })

    const errors: string[] = []
    const fatals: string[] = []
    const stream = new VoiceStream({
      onError: (msg) => errors.push(msg),
      onFatalError: (msg) => fatals.push(msg)
    })

    const starting = stream.start('ws://127.0.0.1:7614/voice')
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
    const ws = FakeWebSocket.instances[0]
    ws.open()
    await starting

    // Exactly one listener for each event (no duplicate registration).
    expect(track.listenerCount('ended')).toBe(1)
    expect(track.listenerCount('mute')).toBe(1)

    track.emit('mute')
    expect(errors).toEqual(['The microphone was muted by the system.'])

    // Encode one frame so muxer has pending Ogg packet data to flush as EOS.
    const captureNode = (FakeAudioContext.instances[0] as unknown)
    void captureNode
    ;(stream as unknown as { encodeFrame: (s: Float32Array) => void }).encodeFrame(
      new Float32Array(480)
    )

    const sentBeforeStop = ws.sent.length
    track.emit('ended')
    await vi.waitFor(() => expect(ws.readyState).toBe(FakeWebSocket.CLOSED))

    expect(fatals).toEqual(['The microphone was disconnected mid-session.'])
    // At most one final EOS frame sent during stop(), never duplicated.
    expect(ws.sent.length - sentBeforeStop).toBe(1)
    expect(track.listenerCount('ended')).toBe(0)
    expect(track.listenerCount('mute')).toBe(0)
  })
})

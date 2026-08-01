/**
 * Minimal NDJSON client for `omp --mode rpc`.
 *
 * Kept inside the omp adapter directory so the rest of Brazier never imports
 * Oh My Pi types or process details.
 */

import { type ChildProcessWithoutNullStreams, spawn } from 'node:child_process'
import { Buffer } from 'node:buffer'
import { createInterface } from 'node:readline'

export type OmpRpcFrame = Record<string, unknown>
export type OmpFrameListener = (frame: OmpRpcFrame) => void

export type OmpRpcClientOptions = {
  binary: string
  cwd?: string
  env?: NodeJS.ProcessEnv
  args?: string[]
  onExit?: (code: number | null, signal: NodeJS.Signals | null) => void
}

export class OmpRpcError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'OmpRpcError'
  }
}

const MAX_REASSEMBLED_FRAME_BYTES = 64 * 1024 * 1024
const MAX_CHUNK_PAYLOAD_BYTES = 256 * 1024

type PendingChunks = {
  chunkId: string
  count: number
  byteLength: number
  nextIndex: number
  chunks: Buffer[]
  receivedBytes: number
}

function isFrame(value: unknown): value is OmpRpcFrame {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function decodeChunkData(data: unknown): Buffer {
  if (
    typeof data !== 'string' ||
    data.length === 0 ||
    !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(data)
  ) {
    throw new OmpRpcError('Invalid OMP RPC chunk data.')
  }
  const bytes = Buffer.from(data, 'base64')
  if (bytes.toString('base64') !== data) throw new OmpRpcError('Invalid OMP RPC chunk data.')
  return bytes
}

/** Stateful v2 decoder for OMP's chunked, lossless stdout transport. */
export class OmpRpcFrameDecoder {
  private pending?: PendingChunks

  push(value: unknown): OmpRpcFrame | undefined {
    if (!isFrame(value) || value.type !== 'rpc_chunk') {
      if (this.pending) throw new OmpRpcError('OMP RPC chunk sequence was interrupted.')
      if (!isFrame(value)) throw new OmpRpcError('OMP RPC frame must be an object.')
      return value
    }

    const chunkId = value.chunkId
    const index = value.index
    const count = value.count
    const byteLength = value.byteLength
    if (
      typeof chunkId !== 'string' ||
      chunkId.length === 0 ||
      chunkId.length > 128 ||
      typeof index !== 'number' ||
      typeof count !== 'number' ||
      typeof byteLength !== 'number' ||
      !Number.isSafeInteger(index) ||
      !Number.isSafeInteger(count) ||
      !Number.isSafeInteger(byteLength) ||
      index < 0 ||
      count < 2 ||
      count > Math.ceil(MAX_REASSEMBLED_FRAME_BYTES / MAX_CHUNK_PAYLOAD_BYTES) ||
      index >= count ||
      byteLength < 1024 * 1024 ||
      byteLength > MAX_REASSEMBLED_FRAME_BYTES
    ) {
      throw new OmpRpcError('Invalid OMP RPC chunk metadata.')
    }
    const bytes = decodeChunkData(value.data)
    if (bytes.byteLength > MAX_CHUNK_PAYLOAD_BYTES) {
      throw new OmpRpcError('OMP RPC chunk payload exceeds the transport limit.')
    }

    if (!this.pending) {
      if (index !== 0) throw new OmpRpcError('OMP RPC chunk sequence must start at index 0.')
      this.pending = { chunkId, count, byteLength, nextIndex: 0, chunks: [], receivedBytes: 0 }
    }
    const pending = this.pending!
    if (
      pending.chunkId !== chunkId ||
      pending.count !== count ||
      pending.byteLength !== byteLength ||
      pending.nextIndex !== index
    ) {
      throw new OmpRpcError('OMP RPC chunk sequence is inconsistent.')
    }
    pending.chunks.push(bytes)
    pending.receivedBytes += bytes.byteLength
    pending.nextIndex += 1
    if (pending.receivedBytes > pending.byteLength) {
      throw new OmpRpcError('OMP RPC chunk sequence exceeds its declared length.')
    }
    if (pending.nextIndex < pending.count) return undefined
    if (pending.receivedBytes !== pending.byteLength) {
      throw new OmpRpcError('OMP RPC chunk sequence length does not match its declaration.')
    }

    this.pending = undefined
    let frame: unknown
    try {
      frame = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(pending.chunks)))
    } catch {
      throw new OmpRpcError('OMP RPC chunks did not contain valid UTF-8 JSON.')
    }
    if (!isFrame(frame)) throw new OmpRpcError('OMP RPC reassembled frame must be an object.')
    return frame
  }
}

export class OmpRpcClient {
  private readonly child: ChildProcessWithoutNullStreams
  private readonly pending = new Map<
    string,
    { resolve: (frame: OmpRpcFrame) => void; reject: (error: Error) => void }
  >()
  private readonly listeners = new Set<OmpFrameListener>()
  private readonly frameDecoder = new OmpRpcFrameDecoder()
  private nextId = 1
  private ready = false
  private readonly readyWaiters: Array<{
    resolve: () => void
    reject: (error: Error) => void
  }> = []
  private closed = false
  private exited = false
  private protocolVersion: 1 | 2 = 1
  private stderrTail = ''

  constructor(options: OmpRpcClientOptions) {
    const args = ['--mode', 'rpc', '--no-session', ...(options.args ?? [])]
    this.child = spawn(options.binary, args, {
      cwd: options.cwd,
      env: options.env,
      stdio: ['pipe', 'pipe', 'pipe']
    })
    this.child.on('error', (error) => {
      this.closed = true
      this.failAll(error)
    })
    this.child.on('exit', (code, signal) => {
      this.exited = true
      this.closed = true
      const stderr = this.stderrTail.trim()
      this.failAll(
        new OmpRpcError(
          `omp exited${code != null ? ` with code ${code}` : ''}${signal ? ` (${signal})` : ''}.` +
            (stderr ? `\n${stderr}` : '')
        )
      )
      options.onExit?.(code, signal)
    })
    const stdout = createInterface({ input: this.child.stdout })
    stdout.on('line', (line) => this.handleLine(line))
    this.child.stderr.setEncoding('utf8')
    this.child.stderr.on('data', (chunk: string) => {
      // OMP startup diagnostics are essential when a CLI version rejects an
      // option. Keep only a bounded tail so a noisy sidecar cannot grow memory.
      this.stderrTail = (this.stderrTail + chunk).slice(-16_384)
    })
  }

  onFrame(listener: OmpFrameListener): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  async waitUntilReady(timeoutMs = 15_000): Promise<void> {
    if (this.ready) return
    if (this.closed) throw new OmpRpcError('omp RPC client is closed.')
    await new Promise<void>((resolve, reject) => {
      const waiter = {
        resolve: () => {
          clearTimeout(timer)
          resolve()
        },
        reject: (error: Error) => {
          clearTimeout(timer)
          reject(error)
        }
      }
      const timer = setTimeout(() => {
        const index = this.readyWaiters.indexOf(waiter)
        if (index !== -1) this.readyWaiters.splice(index, 1)
        reject(new OmpRpcError('Timed out waiting for omp RPC ready frame.'))
      }, timeoutMs)
      this.readyWaiters.push(waiter)
    })
  }

  async request(command: OmpRpcFrame, timeoutMs = 60_000): Promise<OmpRpcFrame> {
    if (this.closed) throw new OmpRpcError('omp RPC client is closed.')
    await this.waitUntilReady()
    const id = typeof command.id === 'string' ? command.id : `brazier-${this.nextId++}`
    const payload = { ...command, id }
    return await new Promise<OmpRpcFrame>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new OmpRpcError(`omp RPC command \`${String(command.type)}\` timed out.`))
      }, timeoutMs)
      this.pending.set(id, {
        resolve: (frame) => {
          clearTimeout(timer)
          resolve(frame)
        },
        reject: (error) => {
          clearTimeout(timer)
          reject(error)
        }
      })
      this.write(payload)
    })
  }

  /** Fire-and-forget stdin write (host_tool_result, extension_ui_response, …). */
  send(frame: OmpRpcFrame): void {
    this.write(frame)
  }

  async dispose(): Promise<void> {
    if (this.closed) return
    this.closed = true
    this.failAll(new OmpRpcError('omp RPC client disposed.'))
    this.listeners.clear()
    if (this.exited) return

    // `child.killed` merely means that Node successfully sent a signal.  It
    // does not say the sidecar has exited, so it cannot decide escalation.
    await new Promise<void>((resolve) => {
      let settled = false
      const settle = () => {
        if (settled) return
        settled = true
        clearTimeout(termTimer)
        clearTimeout(killTimer)
        resolve()
      }
      const termTimer = setTimeout(() => {
        if (this.exited) return settle()
        try {
          this.child.kill('SIGKILL')
        } finally {
          // Do not let a broken or mocked ChildProcess keep disposal pending
          // indefinitely after the best-effort forced kill.
          killTimer = setTimeout(settle, 1_000)
        }
      }, 2_000)
      let killTimer: ReturnType<typeof setTimeout> | undefined
      this.child.once('exit', settle)
      try {
        this.child.kill('SIGTERM')
      } catch {
        settle()
      }
    })
  }

  private write(frame: OmpRpcFrame): void {
    if (!this.child.stdin.writable) {
      throw new OmpRpcError('omp stdin is not writable.')
    }
    this.child.stdin.write(`${JSON.stringify(frame)}\n`)
  }

  private handleLine(line: string): void {
    const trimmed = line.trim()
    if (!trimmed) return
    let rawFrame: unknown
    try {
      rawFrame = JSON.parse(trimmed)
    } catch {
      return
    }
    let frame: OmpRpcFrame | undefined
    try {
      frame = this.frameDecoder.push(rawFrame)
    } catch (error) {
      this.failProtocol(error)
      return
    }
    if (!frame) return
    if (frame.type === 'ready') {
      this.ready = true
      this.negotiateProtocolV2(frame)
      for (const waiter of this.readyWaiters.splice(0)) waiter.resolve()
      this.emit(frame)
      return
    }
    if (frame.type === 'response') {
      const id = typeof frame.id === 'string' ? frame.id : undefined
      if (id && this.pending.has(id)) {
        const pending = this.pending.get(id)!
        this.pending.delete(id)
        if (frame.success === false) {
          pending.reject(
            new OmpRpcError(
              typeof frame.error === 'string'
                ? frame.error
                : `omp command failed (${String(frame.command)})`
            )
          )
        } else {
          pending.resolve(frame)
        }
        // Still emit so run listeners can observe failures without ids.
        this.emit(frame)
        return
      }
    }
    this.emit(frame)
  }

  private emit(frame: OmpRpcFrame): void {
    for (const listener of this.listeners) listener(frame)
  }

  private failAll(error: Error): void {
    for (const waiter of this.readyWaiters.splice(0)) waiter.reject(error)
    for (const pending of this.pending.values()) pending.reject(error)
    this.pending.clear()
  }

  private negotiateProtocolV2(ready: OmpRpcFrame): void {
    if (!Array.isArray(ready.supportedProtocolVersions) || !ready.supportedProtocolVersions.includes(2)) {
      return
    }
    void this.request({ type: 'negotiate_protocol', protocolVersion: 2 }, 15_000)
      .then(() => {
        this.protocolVersion = 2
      })
      // V1 remains a safe fallback when the sidecar advertises but rejects v2.
      .catch(() => undefined)
  }

  private failProtocol(error: unknown): void {
    const failure =
      error instanceof OmpRpcError ? error : new OmpRpcError(`Invalid OMP RPC frame: ${String(error)}`)
    this.closed = true
    this.failAll(failure)
    try {
      this.child.kill('SIGTERM')
    } catch {
      // The exit listener will perform the remaining cleanup when possible.
    }
  }
}

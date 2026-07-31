/**
 * Minimal NDJSON client for `omp --mode rpc`.
 *
 * Kept inside the omp adapter directory so the rest of Brazier never imports
 * Oh My Pi types or process details.
 */

import { type ChildProcessWithoutNullStreams, spawn } from 'node:child_process'
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

export class OmpRpcClient {
  private readonly child: ChildProcessWithoutNullStreams
  private readonly pending = new Map<
    string,
    { resolve: (frame: OmpRpcFrame) => void; reject: (error: Error) => void }
  >()
  private readonly listeners = new Set<OmpFrameListener>()
  private nextId = 1
  private ready = false
  private readonly readyWaiters: Array<() => void> = []
  private closed = false

  constructor(options: OmpRpcClientOptions) {
    const args = ['--mode', 'rpc', '--no-session', ...(options.args ?? [])]
    this.child = spawn(options.binary, args, {
      cwd: options.cwd,
      env: options.env,
      stdio: ['pipe', 'pipe', 'pipe']
    })
    this.child.on('error', (error) => {
      this.failAll(error)
    })
    this.child.on('exit', (code, signal) => {
      this.closed = true
      this.failAll(
        new OmpRpcError(
          `omp exited${code != null ? ` with code ${code}` : ''}${signal ? ` (${signal})` : ''}.`
        )
      )
      options.onExit?.(code, signal)
    })
    const stdout = createInterface({ input: this.child.stdout })
    stdout.on('line', (line) => this.handleLine(line))
    createInterface({ input: this.child.stderr }).on('line', () => undefined)
  }

  onFrame(listener: OmpFrameListener): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  async waitUntilReady(timeoutMs = 15_000): Promise<void> {
    if (this.ready) return
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new OmpRpcError('Timed out waiting for omp RPC ready frame.'))
      }, timeoutMs)
      this.readyWaiters.push(() => {
        clearTimeout(timer)
        resolve()
      })
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
    if (!this.child.killed) {
      this.child.kill('SIGTERM')
      await new Promise<void>((resolve) => {
        const timer = setTimeout(() => {
          if (!this.child.killed) this.child.kill('SIGKILL')
          resolve()
        }, 2_000)
        this.child.once('exit', () => {
          clearTimeout(timer)
          resolve()
        })
      })
    }
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
    let frame: OmpRpcFrame
    try {
      frame = JSON.parse(trimmed) as OmpRpcFrame
    } catch {
      return
    }
    if (frame.type === 'ready') {
      this.ready = true
      for (const waiter of this.readyWaiters.splice(0)) waiter()
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
    for (const pending of this.pending.values()) pending.reject(error)
    this.pending.clear()
  }
}

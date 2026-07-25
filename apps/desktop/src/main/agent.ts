/**
 * Agent worker supervisor.
 *
 * The agent runtime runs in an Electron `utilityProcess`, not in the main
 * process and not in the renderer: a stuck loop or a crashing runtime must not
 * take the window with it. This file owns that process, forwards commands from
 * the renderer, and streams events back.
 */

import { join } from 'node:path'
import { app, BrowserWindow, ipcMain, utilityProcess, type UtilityProcess } from 'electron'

import {
  AGENT_IPC,
  type WorkerCommand,
  type WorkerCommandInput,
  type WorkerConnection,
  type WorkerMessage
} from '../agent/core/protocol'

type Pending = {
  resolve: (value: unknown) => void
  reject: (error: Error) => void
}

/** Requests that may run for a long time (a run waits on the model and the user). */
const LONG_RUNNING: WorkerCommand['type'][] = ['run']
const SHORT_REQUEST_TIMEOUT_MS = 60_000

export class AgentSupervisor {
  private worker?: UtilityProcess
  private connection?: WorkerConnection
  private readonly pending = new Map<string, Pending>()
  private nextId = 0
  private ready?: Promise<void>
  private crashes = 0

  /**
   * Where the built worker bundle lives. `scripts/build-agent-worker.mjs` emits
   * it beside `out/main`, since electron-vite empties its own output directory.
   */
  private workerPath(): string {
    return join(__dirname, '..', 'agent', 'agent-worker.mjs')
  }

  setConnection(connection: WorkerConnection): void {
    this.connection = connection
  }

  private requestId(): string {
    this.nextId += 1
    return `req-${this.nextId}`
  }

  private broadcast(channel: string, payload: unknown): void {
    for (const window of BrowserWindow.getAllWindows()) {
      if (!window.isDestroyed()) window.webContents.send(channel, payload)
    }
  }

  private onMessage(message: WorkerMessage): void {
    switch (message.type) {
      case 'result': {
        const pending = this.pending.get(message.requestId)
        if (!pending) return
        this.pending.delete(message.requestId)
        if (message.ok) pending.resolve(message.data)
        else pending.reject(new Error(message.error))
        return
      }
      case 'event':
      case 'session-state': {
        this.broadcast(AGENT_IPC.event, message)
        return
      }
      case 'log': {
        const line = `[agent-worker] ${message.message}`
        if (message.level === 'error') console.error(line)
        else console.warn(line)
        return
      }
      case 'ready': {
        console.error(
          `[agent-worker] ready (runtime ${message.runtimeId} ${message.runtimeVersion})`
        )
        return
      }
      default:
        return
    }
  }

  /** Start the worker and initialize it against the daemon connection. */
  private async ensureWorker(): Promise<void> {
    if (this.ready) return this.ready
    this.ready = (async () => {
      const connection = this.connection
      if (!connection) {
        throw new Error('The Brazier daemon connection is not available yet.')
      }
      // The worker inherits no daemon credentials: the bearer token travels in
      // the init message, and BRAZIER_* configuration stays in this process.
      const workerEnv: Record<string, string> = {}
      for (const [key, value] of Object.entries(process.env)) {
        if (value === undefined || key.startsWith('BRAZIER_')) continue
        workerEnv[key] = value
      }
      workerEnv.NODE_ENV =
        process.env.NODE_ENV ?? (app.isPackaged ? 'production' : 'development')

      const worker = utilityProcess.fork(this.workerPath(), [], {
        serviceName: 'brazier-agent',
        stdio: 'inherit',
        env: workerEnv
      })
      this.worker = worker
      worker.on('message', (message: WorkerMessage) => this.onMessage(message))
      worker.on('exit', (code) => {
        this.crashes += 1
        for (const [, pending] of this.pending) {
          pending.reject(new Error(`The agent worker exited (code ${code}).`))
        }
        this.pending.clear()
        this.worker = undefined
        this.ready = undefined
        console.error(`[agent-worker] exited with code ${code}`)
      })
      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(
          () => reject(new Error('The agent worker did not start in time.')),
          20_000
        )
        worker.once('spawn', () => {
          clearTimeout(timer)
          resolve()
        })
      })
      await this.send({ type: 'init', requestId: this.requestId(), connection })
    })()
    try {
      await this.ready
    } catch (error) {
      this.ready = undefined
      throw error
    }
  }

  private send(command: WorkerCommand): Promise<unknown> {
    const worker = this.worker
    if (!worker) return Promise.reject(new Error('The agent worker is not running.'))
    return new Promise((resolve, reject) => {
      this.pending.set(command.requestId, { resolve, reject })
      if (!LONG_RUNNING.includes(command.type)) {
        setTimeout(() => {
          if (!this.pending.has(command.requestId)) return
          this.pending.delete(command.requestId)
          reject(new Error(`The agent worker did not answer \`${command.type}\` in time.`))
        }, SHORT_REQUEST_TIMEOUT_MS)
      }
      worker.postMessage(command)
    })
  }

  /** Renderer entry point: run one command against the worker. */
  async invoke(command: WorkerCommandInput): Promise<unknown> {
    await this.ensureWorker()
    return this.send({ ...command, requestId: this.requestId() } as WorkerCommand)
  }

  async shutdown(): Promise<void> {
    const worker = this.worker
    if (!worker) return
    try {
      await this.send({ type: 'shutdown', requestId: this.requestId() })
    } catch {
      // A worker that cannot answer is killed below.
    }
    worker.kill()
    this.worker = undefined
    this.ready = undefined
  }

  /** Diagnostics for the UI: whether the worker is up and how often it died. */
  status(): { running: boolean; crashes: number } {
    return { running: Boolean(this.worker), crashes: this.crashes }
  }
}

export function registerAgentIpc(supervisor: AgentSupervisor): void {
  ipcMain.handle(AGENT_IPC.invoke, async (_event, payload: WorkerCommandInput) => {
    if (!payload || typeof payload !== 'object' || typeof payload.type !== 'string') {
      throw new Error('Malformed agent command.')
    }
    return supervisor.invoke(payload)
  })
}

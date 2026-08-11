/**
 * Agent worker supervisor.
 *
 * The agent runtime runs in an Electron `utilityProcess`, not in the main
 * process and not in the renderer: a stuck loop or a crashing runtime must not
 * take the window with it. This file owns that process, forwards commands from
 * the renderer, and streams events back.
 */

import { existsSync } from 'node:fs'
import { join, sep } from 'node:path'
import {
  app,
  BrowserWindow,
  ipcMain,
  utilityProcess,
  type IpcMainInvokeEvent,
  type UtilityProcess
} from 'electron'

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
/** A stop is a safety control, so a wedged runtime gets only a short grace period. */
const CANCEL_GRACE_MS = 2_000
const DAEMON_CANCEL_TIMEOUT_MS = 5_000
const DAEMON_PATCH_TIMEOUT_MS = 5_000

const ALLOWED_COMMAND_TYPES = new Set<WorkerCommandInput['type']>([
  'open-session',
  'run',
  'cancel',
  'compact',
  'set-model',
  'set-tools',
  'set-permission-mode',
  'close-session'
])

export class AgentSupervisor {
  private worker?: UtilityProcess
  private connection?: WorkerConnection
  private readonly pending = new Map<string, Pending>()
  private nextId = 0
  private ready?: Promise<void>
  private crashes = 0
  private readonly sessions = new Set<string>()
  private expectedExit?: UtilityProcess

  /**
   * Where the built worker bundle lives. `scripts/build-agent-worker.mjs` emits
   * it beside `out/main`, since electron-vite empties its own output directory.
   *
   * Packaged builds keep the worker in `asarUnpack`; fork must target the
   * unpacked path or Node exits before `parentPort` comes up (code 1).
   */
  private workerPath(): string {
    const relative = join('out', 'agent', 'agent-worker.mjs')
    const candidates = [
      join(__dirname, '..', 'agent', 'agent-worker.mjs'),
      join(app.getAppPath(), relative)
    ]
    if (app.isPackaged) {
      candidates.unshift(
        join(app.getAppPath(), relative).replace(`${sep}app.asar${sep}`, `${sep}app.asar.unpacked${sep}`)
      )
    }
    const resolved = candidates.find((candidate) => existsSync(candidate))
    if (!resolved) {
      throw new Error(
        'The agent worker bundle is missing. Rebuild the desktop app (`pnpm run build:agent` in apps/desktop).'
      )
    }
    return resolved
  }

  setConnection(connection: WorkerConnection): void {
    this.connection = connection
  }

  /** Eager package-smoke hook; normal UI sessions still start lazily. */
  async warmup(): Promise<void> {
    await this.ensureWorker()
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
        const expected = this.expectedExit === worker
        if (!expected) this.crashes += 1
        for (const [, pending] of this.pending) {
          pending.reject(new Error(`The agent worker exited (code ${code}).`))
        }
        this.pending.clear()
        const known = [...this.sessions]
        if (expected) this.sessions.clear()
        this.worker = undefined
        this.ready = undefined
        if (expected) this.expectedExit = undefined
        else void this.markSessionsFailed(known, code)
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

  /**
   * Cancel at the daemon boundary as well as in the runtime.
   *
   * The daemon owns tool processes and pending approvals, while the utility
   * process owns the model loop. Stopping both independently means a wedged
   * renderer or runtime cannot leave one half of a run alive.
   */
  private async cancelDaemon(sessionId: string): Promise<void> {
    const connection = this.connection
    if (!connection) throw new Error('The Brazier daemon connection is not available yet.')
    const headers = new Headers({ 'content-type': 'application/json' })
    if (connection.apiKey) headers.set('authorization', `Bearer ${connection.apiKey}`)
    const response = await fetch(
      `${connection.address}/api/v1/agent/sessions/${encodeURIComponent(sessionId)}/cancel`,
      {
        method: 'POST',
        headers,
        body: '{}',
        signal: AbortSignal.timeout(DAEMON_CANCEL_TIMEOUT_MS)
      }
    )
    if (!response.ok) {
      throw new Error(`The daemon could not cancel the agent run (status ${response.status}).`)
    }
  }

  private async patchSession(
    sessionId: string,
    update: Record<string, unknown>
  ): Promise<void> {
    const connection = this.connection
    if (!connection) return
    const headers = new Headers({ 'content-type': 'application/json' })
    if (connection.apiKey) headers.set('authorization', `Bearer ${connection.apiKey}`)
    const response = await fetch(
      `${connection.address}/api/v1/agent/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: 'PATCH',
        headers,
        body: JSON.stringify(update),
        signal: AbortSignal.timeout(DAEMON_CANCEL_TIMEOUT_MS)
      }
    ).catch(() => null)
    if (!response || !response.ok) {
      console.error(`[agent-worker] could not reconcile session ${sessionId} after exit.`)
    }
  }

  private async markSessionsFailed(sessionIds: string[], code: number): Promise<void> {
    if (sessionIds.length === 0) return
    await Promise.all(
      sessionIds.map((sessionId) =>
        this.patchSession(sessionId, {
          last_run_status: 'failed',
          agent_note: `The agent worker exited (code ${code}).`
        })
      )
    )
  }

  /** Kill a worker that did not acknowledge cancellation and reject its run. */
  private killUnresponsiveWorker(): void {
    const worker = this.worker
    if (!worker) return
    worker.kill()
    const error = new Error('The agent worker was terminated because it did not stop in time.')
    for (const [, pending] of this.pending) pending.reject(error)
    this.pending.clear()
    this.worker = undefined
    this.ready = undefined
  }

  private async cancel(sessionId: string): Promise<{ cancelled: true }> {
    if (!this.sessions.has(sessionId)) {
      throw new Error('Refusing to cancel a session the agent worker does not know about.')
    }
    if (!this.worker) {
      throw new Error('The agent worker is not running; nothing to cancel.')
    }
    // Start both cancellation paths immediately. In particular, do not queue
    // daemon cleanup behind a runtime that may be precisely what is wedged.
    const daemonCancellation = this.cancelDaemon(sessionId)
    const workerWasRunning = Boolean(this.worker)
    const workerCancellation = workerWasRunning
      ? Promise.race([
          this.send({ type: 'cancel', requestId: this.requestId(), sessionId }),
          new Promise<never>((_, reject) => {
            setTimeout(
              () => reject(new Error('The agent worker did not acknowledge cancellation.')),
              CANCEL_GRACE_MS
            )
          })
        ]).catch((error: unknown) => {
          // Do this in the rejection path rather than after waiting for daemon
          // cleanup, so the two-second grace period is a real upper bound.
          this.killUnresponsiveWorker()
          throw error
        })
      : Promise.resolve()

    const [daemonResult, workerResult] = await Promise.allSettled([
      daemonCancellation,
      workerCancellation
    ])

    // Either boundary can conclusively stop the active model loop. Report a
    // failure only if neither could be reached; otherwise Stop remains
    // successful even when its hard-kill fallback was needed.
    if (
      daemonResult.status === 'rejected' &&
      (!workerWasRunning || workerResult.status === 'rejected')
    ) {
      throw new Error(
        `Could not stop the agent safely: ${daemonResult.reason instanceof Error ? daemonResult.reason.message : String(daemonResult.reason)}`
      )
    }
    return { cancelled: true }
  }

  /** Renderer entry point: run one command against the worker. */
  async invoke(command: WorkerCommandInput): Promise<unknown> {
    if (!ALLOWED_COMMAND_TYPES.has(command.type)) {
      throw new Error(`Unknown agent command type \`${command.type}\`.`)
    }
    if ('sessionId' in command && typeof command.sessionId === 'string') {
      this.sessions.add(command.sessionId)
    }
    if (command.type === 'cancel') return this.cancel(command.sessionId)
    await this.ensureWorker()
    return this.send({ ...command, requestId: this.requestId() } as WorkerCommand)
  }

  async shutdown(): Promise<void> {
    const worker = this.worker
    if (!worker) return
    this.expectedExit = worker
    const exited = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error('The agent worker did not exit within 5 seconds.')),
        5_000
      )
      worker.once('exit', () => {
        clearTimeout(timeout)
        resolve()
      })
    })
    try {
      await this.send({ type: 'shutdown', requestId: this.requestId() })
    } catch {
      // A worker that cannot answer is killed below.
    }
    worker.kill()
    await exited
    if (this.worker === worker) {
      this.worker = undefined
      this.ready = undefined
      // Keep expectedExit set so a delayed exit remains intentional.
    }
  }

  /** Diagnostics for the UI: whether the worker is up and how often it died. */
  status(): { running: boolean; crashes: number } {
    return { running: Boolean(this.worker), crashes: this.crashes }
  }
}

export function registerAgentIpc(
  supervisor: AgentSupervisor,
  assertTrustedSender: (event: IpcMainInvokeEvent) => void
): void {
  ipcMain.handle(AGENT_IPC.invoke, async (event, payload: WorkerCommandInput) => {
    assertTrustedSender(event)
    if (!payload || typeof payload !== 'object' || typeof payload.type !== 'string') {
      throw new Error('Malformed agent command.')
    }
    return supervisor.invoke(payload)
  })
  ipcMain.handle('brazil:agent:status', (event) => {
    assertTrustedSender(event)
    return supervisor.status()
  })
}

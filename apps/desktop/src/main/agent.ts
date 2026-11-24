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
const SHUTDOWN_GRACE_MS = 5_000
const SHUTDOWN_KILL_MS = 2_000
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

type ActiveSessionRun = {
  runId: string
  sequence: number
}

export class AgentSupervisor {
  private worker?: UtilityProcess
  private connection?: WorkerConnection
  private workerConnection?: WorkerConnection
  private readonly pending = new Map<string, Pending>()
  private nextId = 0
  private ready?: Promise<void>
  private crashes = 0
  private readonly sessions = new Set<string>()
  private readonly activeSessions = new Map<string, ActiveSessionRun>()
  /** Runs that already ended, so a late event cannot make them look active again. */
  private readonly finishedRuns = new Set<string>()
  private readonly eventListeners = new Set<(message: WorkerMessage) => void>()
  private readonly terminatedWorkers = new WeakSet<UtilityProcess>()

  constructor(
    private readonly options: {
      cancelGraceMs?: number
      forkWorker?: () => UtilityProcess
      fetch?: typeof fetch
    } = {}
  ) {}

  /** Subscribe to worker events (including synthesized terminal `run-failed` events). */
  onEvent(listener: (message: WorkerMessage) => void): () => void {
    this.eventListeners.add(listener)
    return () => {
      this.eventListeners.delete(listener)
    }
  }

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
    try {
      for (const window of BrowserWindow.getAllWindows()) {
        if (!window.isDestroyed()) window.webContents.send(channel, payload)
      }
    } catch {
      // Headless unit tests may run without initialized BrowserWindow state.
    }
  }

  private emitWorkerMessage(message: WorkerMessage): void {
    for (const listener of [...this.eventListeners]) {
      listener(message)
    }
    this.broadcast(AGENT_IPC.event, message)
  }

  private synthesizeTerminalRunFailures(reason: string): void {
    if (this.activeSessions.size === 0) return
    const entries = [...this.activeSessions.entries()]
    this.activeSessions.clear()
    const timestamp = new Date().toISOString()
    for (const [sessionId, active] of entries) {
      this.emitWorkerMessage({
        type: 'event',
        sessionId,
        event: {
          type: 'run-failed',
          sessionId,
          runId: active.runId,
          timestamp,
          sequence: active.sequence + 1,
          error: reason
        }
      })
    }
  }

  private onMessage(worker: UtilityProcess, message: WorkerMessage): void {
    if (this.worker !== worker) return
    switch (message.type) {
      case 'result': {
        const pending = this.pending.get(message.requestId)
        if (!pending) return
        if (message.ok) pending.resolve(message.data)
        else pending.reject(new Error(message.error))
        return
      }
      case 'event': {
        this.trackActiveSession(message.sessionId, message.event)
        this.emitWorkerMessage(message)
        return
      }
      case 'session-state': {
        this.emitWorkerMessage(message)
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

  /**
   * Keep `activeSessions` aligned with the run that is actually in flight.
   * A terminal event drops it, and anything that arrives afterwards for that
   * same run is ignored: otherwise a late tool event would put a finished
   * session back on the list, and the next worker crash would invent a
   * second failure for it.
   */
  private trackActiveSession(
    sessionId: string,
    event: Extract<WorkerMessage, { type: 'event' }>['event']
  ): void {
    if (this.finishedRuns.has(event.runId)) return
    const terminal =
      event.type === 'run-completed' ||
      event.type === 'run-cancelled' ||
      event.type === 'run-failed'
    const tracked = this.activeSessions.get(sessionId)
    // `invoke` records a placeholder before the worker has chosen a run id.
    const placeholder =
      tracked !== undefined &&
      tracked.sequence === 0 &&
      tracked.runId.startsWith(`run-${sessionId}-`)
    const sameRun = tracked?.runId === event.runId
    if (terminal) {
      this.rememberFinishedRun(event.runId)
      if (!tracked || sameRun || placeholder) this.activeSessions.delete(sessionId)
      return
    }
    if (event.type === 'run-started' || sameRun || placeholder) {
      this.activeSessions.set(sessionId, {
        runId: event.runId,
        sequence: event.sequence
      })
    }
  }

  private rememberFinishedRun(runId: string): void {
    this.finishedRuns.add(runId)
    if (this.finishedRuns.size <= 64) return
    const oldest = this.finishedRuns.values().next().value
    if (oldest) this.finishedRuns.delete(oldest)
  }

  /** Start the worker and initialize it against the daemon connection. */
  private async ensureWorker(): Promise<void> {
    if (this.ready) return this.ready
    let spawnedWorker: UtilityProcess | undefined
    const readyPromise = (async () => {
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

      const worker = this.options.forkWorker
        ? this.options.forkWorker()
        : utilityProcess.fork(this.workerPath(), [], {
            serviceName: 'brazier-agent',
            stdio: 'inherit',
            env: workerEnv
          })
      spawnedWorker = worker
      this.worker = worker
      this.workerConnection = connection
      worker.on('message', (message: WorkerMessage) => this.onMessage(worker, message))
      worker.on('exit', (code) => {
        console.error(`[agent-worker] exited with code ${code}`)
        const expected = this.terminatedWorkers.has(worker)
        if (this.worker !== worker) {
          // A stale exit from an already-replaced or already-terminated worker
          // must never clobber the replacement worker or its pending requests.
          return
        }
        if (!expected) this.crashes += 1
        const exitError = `The agent worker exited (code ${code}).`
        for (const [, pending] of this.pending) {
          pending.reject(new Error(exitError))
        }
        this.pending.clear()
        const activeIds = [...this.activeSessions.keys()]
        if (expected) {
          this.sessions.clear()
          this.activeSessions.clear()
        }
        this.worker = undefined
        if (this.ready === readyPromise) this.ready = undefined
        if (!expected) {
          this.synthesizeTerminalRunFailures(exitError)
          void this.markSessionsFailed(connection, activeIds, code)
        }
      })
      await new Promise<void>((resolve, reject) => {
        const onSpawn = (): void => {
          cleanup()
          resolve()
        }
        const onExit = (code: number): void => {
          cleanup()
          reject(new Error(`The agent worker exited before starting (code ${code}).`))
        }
        const timer = setTimeout(() => {
          cleanup()
          reject(new Error('The agent worker did not start in time.'))
        }, 20_000)
        const cleanup = (): void => {
          clearTimeout(timer)
          worker.removeListener('spawn', onSpawn)
          worker.removeListener('exit', onExit)
        }
        worker.once('spawn', onSpawn)
        worker.once('exit', onExit)
      })
      if (this.worker !== worker || this.terminatedWorkers.has(worker)) {
        throw new Error('The agent worker was replaced before it finished starting.')
      }
      await this.send({ type: 'init', requestId: this.requestId(), connection })
    })()
    this.ready = readyPromise
    try {
      await readyPromise
    } catch (error) {
      if (this.ready === readyPromise) this.ready = undefined
      if (spawnedWorker && this.worker === spawnedWorker) {
        this.terminatedWorkers.add(spawnedWorker)
        this.worker = undefined
        for (const [, pending] of this.pending) {
          pending.reject(error instanceof Error ? error : new Error(String(error)))
        }
        this.pending.clear()
        spawnedWorker.kill()
      }
      throw error
    }
  }

  private send(command: WorkerCommand, timeoutMs = SHORT_REQUEST_TIMEOUT_MS): Promise<unknown> {
    const worker = this.worker
    if (!worker) return Promise.reject(new Error('The agent worker is not running.'))
    return new Promise((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | undefined
      const cleanup = (): void => {
        clearTimeout(timer)
        this.pending.delete(command.requestId)
      }
      const pending: Pending = {
        resolve: (value) => {
          cleanup()
          resolve(value)
        },
        reject: (error) => {
          cleanup()
          reject(error)
        }
      }
      this.pending.set(command.requestId, pending)
      if (!LONG_RUNNING.includes(command.type)) {
        timer = setTimeout(() => {
          pending.reject(new Error(`The agent worker did not answer \`${command.type}\` in time.`))
        }, timeoutMs)
      }
      try {
        worker.postMessage(command)
      } catch (error) {
        pending.reject(error instanceof Error ? error : new Error(String(error)))
      }
    })
  }

  /**
   * Cancel at the daemon boundary as well as in the runtime.
   *
   * The daemon owns tool processes and pending approvals, while the utility
   * process owns the model loop. Stopping both independently means a wedged
   * renderer or runtime cannot leave one half of a run alive.
   */
  private async cancelDaemon(connection: WorkerConnection, sessionId: string): Promise<void> {
    const headers = new Headers({ 'content-type': 'application/json' })
    if (connection.apiKey) headers.set('authorization', `Bearer ${connection.apiKey}`)
    const fetch_ = this.options.fetch ?? fetch
    const response = await fetch_(
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
    connection: WorkerConnection,
    sessionId: string,
    update: Record<string, unknown>
  ): Promise<void> {
    const headers = new Headers({ 'content-type': 'application/json' })
    if (connection.apiKey) headers.set('authorization', `Bearer ${connection.apiKey}`)
    const fetch_ = this.options.fetch ?? fetch
    const response = await fetch_(
      `${connection.address}/api/v1/agent/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: 'PATCH',
        headers,
        body: JSON.stringify(update),
        signal: AbortSignal.timeout(DAEMON_PATCH_TIMEOUT_MS)
      }
    ).catch(() => null)
    if (!response || !response.ok) {
      console.error(`[agent-worker] could not reconcile session ${sessionId} after exit.`)
    }
  }

  private async markSessionsFailed(
    connection: WorkerConnection,
    sessionIds: string[],
    code: number
  ): Promise<void> {
    await Promise.all(
      sessionIds.map(async (sessionId) => {
        // The daemon also cancels active children owned by this session's client.
        await this.cancelDaemon(connection, sessionId).catch(() => undefined)
        await this.patchSession(connection, sessionId, {
          last_run_status: 'failed',
          agent_note: `The agent worker exited (code ${code}).`
        })
      })
    )
  }

  /** Kill a worker that did not acknowledge cancellation and reject its run. */
  private killUnresponsiveWorker(worker: UtilityProcess, connection: WorkerConnection): void {
    if (this.worker !== worker || this.terminatedWorkers.has(worker)) return
    const error = new Error('The agent worker was terminated because it did not stop in time.')
    this.terminatedWorkers.add(worker)
    const activeIds = [...this.activeSessions.keys()]
    this.worker = undefined
    this.ready = undefined
    for (const [, pending] of this.pending) pending.reject(error)
    this.pending.clear()
    this.synthesizeTerminalRunFailures(error.message)
    worker.kill()
    if (activeIds.length > 0) {
      void this.markSessionsFailed(connection, activeIds, 137)
    }
  }

  private async cancel(sessionId: string): Promise<{ cancelled: true }> {
    if (!this.sessions.has(sessionId)) {
      throw new Error('Refusing to cancel a session the agent worker does not know about.')
    }
    const worker = this.worker
    const connection = this.workerConnection
    if (!worker || !connection) {
      throw new Error('The agent worker is not running; nothing to cancel.')
    }
    const graceMs = this.options.cancelGraceMs ?? CANCEL_GRACE_MS
    // Start daemon cleanup immediately, even if the runtime is wedged.
    const daemonCancellation = this.cancelDaemon(connection, sessionId)
    const workerCancellation = this.send(
      { type: 'cancel', requestId: this.requestId(), sessionId },
      graceMs
    ).catch((error: unknown) => {
      this.killUnresponsiveWorker(worker, connection)
      throw error
    })

    const [daemonResult, workerResult] = await Promise.allSettled([
      daemonCancellation,
      workerCancellation
    ])

    // Either boundary can conclusively stop the active model loop. Report a
    // failure only if neither could be reached; otherwise Stop remains
    // successful even when its hard-kill fallback was needed.
    if (daemonResult.status === 'rejected' && workerResult.status === 'rejected') {
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
    if (command.type === 'cancel') return this.cancel(command.sessionId)
    if (command.type === 'close-session' && !this.worker) {
      this.sessions.delete(command.sessionId)
      this.activeSessions.delete(command.sessionId)
      return { closed: true }
    }
    if ('sessionId' in command && typeof command.sessionId === 'string') {
      this.sessions.add(command.sessionId)
    }
    await this.ensureWorker()
    let active: ActiveSessionRun | undefined
    if (command.type === 'run') {
      if (this.activeSessions.has(command.sessionId)) {
        throw new Error('That session is already running. Cancel it first.')
      }
      active = {
        runId: `run-${command.sessionId}-${this.nextId + 1}`,
        sequence: 0
      }
      this.activeSessions.set(command.sessionId, active)
    }
    try {
      const result = await this.send({ ...command, requestId: this.requestId() } as WorkerCommand)
      if (command.type === 'close-session') {
        this.sessions.delete(command.sessionId)
        this.activeSessions.delete(command.sessionId)
      }
      return result
    } catch (error) {
      if (command.type === 'run' && this.activeSessions.get(command.sessionId) === active) {
        this.activeSessions.delete(command.sessionId)
      }
      throw error
    }
  }

  async shutdown(): Promise<void> {
    const worker = this.worker
    if (!worker) return
    this.terminatedWorkers.add(worker)
    const exited = new Promise<void>((resolve) => {
      worker.once('exit', () => resolve())
    })
    // `send` normally has a one-minute timeout. Quit must not inherit that:
    // after a bounded opportunity to dispose sessions/runtimes, kill the
    // utility process so daemon cleanup and a subsequent launch can proceed.
    let gracefulTimer: ReturnType<typeof setTimeout> | undefined
    try {
      await Promise.race([
        this.send({ type: 'shutdown', requestId: this.requestId() }).catch(() => undefined),
        new Promise<void>((resolve) => {
          gracefulTimer = setTimeout(resolve, SHUTDOWN_GRACE_MS)
        })
      ])
    } finally {
      if (gracefulTimer) clearTimeout(gracefulTimer)
    }
    if (this.worker === worker) worker.kill()
    let killTimer: ReturnType<typeof setTimeout> | undefined
    try {
      await Promise.race([
        exited,
        new Promise<never>((_, reject) => {
          killTimer = setTimeout(
            () => reject(new Error('The agent worker did not exit after forced shutdown.')),
            SHUTDOWN_KILL_MS
          )
        })
      ])
    } finally {
      if (killTimer) clearTimeout(killTimer)
      if (this.worker === worker) {
        const error = new Error('The agent worker was shut down.')
        for (const pending of this.pending.values()) pending.reject(error)
        this.worker = undefined
        this.ready = undefined
        this.sessions.clear()
        this.activeSessions.clear()
      }
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
  ipcMain.handle('brazier:agent:status', (event) => {
    assertTrustedSender(event)
    return supervisor.status()
  })
}

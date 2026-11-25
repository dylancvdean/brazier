import { EventEmitter } from 'node:events'
import { describe, expect, it, vi } from 'vitest'

import type { UtilityProcess } from 'electron'
import type { WorkerCommand, WorkerMessage } from '../agent/core/protocol'

vi.mock('electron', () => ({
  app: {
    isPackaged: false,
    getAppPath: () => process.cwd()
  },
  BrowserWindow: {
    getAllWindows: () => []
  },
  ipcMain: {
    handle: vi.fn()
  },
  utilityProcess: {
    fork: vi.fn()
  }
}))

const { AgentSupervisor } = await import('./agent')

class FakeUtilityProcess extends EventEmitter {
  readonly sent: WorkerCommand[] = []
  killed = false
  autoRespondInit = true
  ignoreCancel = false

  postMessage(message: WorkerCommand): void {
    this.sent.push(message)
    if (message.type === 'init' && this.autoRespondInit) {
      queueMicrotask(() => {
        this.emit('message', {
          type: 'result',
          requestId: message.requestId,
          ok: true
        } satisfies WorkerMessage)
      })
    } else if (message.type === 'open-session') {
      queueMicrotask(() => {
        this.emit('message', {
          type: 'result',
          requestId: message.requestId,
          ok: true
        } satisfies WorkerMessage)
      })
    } else if (message.type === 'cancel' && !this.ignoreCancel) {
      queueMicrotask(() => {
        this.emit('message', {
          type: 'result',
          requestId: message.requestId,
          ok: true
        } satisfies WorkerMessage)
      })
    }
  }

  kill(): boolean {
    this.killed = true
    queueMicrotask(() => this.emit('exit', 137))
    return true
  }
}

function runEvent(
  sessionId: string,
  type: string,
  runId: string,
  sequence: number
): WorkerMessage {
  return {
    type: 'event',
    sessionId,
    event: {
      type,
      sessionId,
      runId,
      timestamp: new Date().toISOString(),
      sequence
    }
  } as WorkerMessage
}

describe('AgentSupervisor unresponsive worker handling', () => {
  it('synthesizes terminal run-failed events for active sessions when killing an unresponsive worker', async () => {
    const fakeWorker = new FakeUtilityProcess()
    fakeWorker.ignoreCancel = true

    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 })) as unknown as typeof fetch
    const supervisor = new AgentSupervisor({
      cancelGraceMs: 15,
      forkWorker: () => {
        queueMicrotask(() => fakeWorker.emit('spawn'))
        return fakeWorker as unknown as UtilityProcess
      },
      fetch: fetchMock
    })
    supervisor.setConnection({ address: 'http://127.0.0.1:7614', apiKey: null })

    const events: WorkerMessage[] = []
    supervisor.onEvent((message) => events.push(message))

    await supervisor.invoke({ type: 'open-session', sessionId: 'session-a' })
    await supervisor.invoke({ type: 'open-session', sessionId: 'session-b' })

    const runA = supervisor.invoke({
      type: 'run',
      sessionId: 'session-a',
      input: { text: 'hang forever' }
    })
    const runB = supervisor.invoke({
      type: 'run',
      sessionId: 'session-b',
      input: { text: 'parallel run' }
    })

    // Wait until both `run` commands reach the worker.
    await vi.waitFor(() => {
      expect(fakeWorker.sent.filter((cmd) => cmd.type === 'run')).toHaveLength(2)
    })

    fakeWorker.emit('message', {
      type: 'event',
      sessionId: 'session-a',
      event: {
        type: 'run-started',
        sessionId: 'session-a',
        runId: 'worker-run-a',
        timestamp: new Date().toISOString(),
        sequence: 1
      }
    } satisfies WorkerMessage)
    fakeWorker.emit('message', {
      type: 'event',
      sessionId: 'session-b',
      event: {
        type: 'run-started',
        sessionId: 'session-b',
        runId: 'worker-run-b',
        timestamp: new Date().toISOString(),
        sequence: 3
      }
    } satisfies WorkerMessage)

    await expect(
      supervisor.invoke({ type: 'cancel', sessionId: 'session-a' })
    ).resolves.toEqual({ cancelled: true })

    expect(fakeWorker.killed).toBe(true)
    await expect(runA).rejects.toThrow('terminated because it did not stop in time')
    await expect(runB).rejects.toThrow('terminated because it did not stop in time')

    const failedEvents = events.filter(
      (msg): msg is Extract<WorkerMessage, { type: 'event' }> =>
        msg.type === 'event' && msg.event.type === 'run-failed'
    )
    expect(failedEvents).toHaveLength(2)
    expect(failedEvents).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          sessionId: 'session-a',
          event: expect.objectContaining({
            type: 'run-failed',
            runId: 'worker-run-a',
            sequence: 2,
            error: expect.stringContaining('did not stop in time')
          })
        }),
        expect.objectContaining({
          sessionId: 'session-b',
          event: expect.objectContaining({
            type: 'run-failed',
            runId: 'worker-run-b',
            sequence: 4,
            error: expect.stringContaining('did not stop in time')
          })
        })
      ])
    )
  })

  it('does not resurrect a finished run when a late event arrives', async () => {
    const fakeWorker = new FakeUtilityProcess()
    fakeWorker.ignoreCancel = true
    const supervisor = new AgentSupervisor({
      cancelGraceMs: 15,
      forkWorker: () => {
        queueMicrotask(() => fakeWorker.emit('spawn'))
        return fakeWorker as unknown as UtilityProcess
      },
      fetch: vi.fn(async () => new Response('{}', { status: 200 })) as unknown as typeof fetch
    })
    supervisor.setConnection({ address: 'http://127.0.0.1:7614', apiKey: null })
    const events: WorkerMessage[] = []
    supervisor.onEvent((message) => events.push(message))

    await supervisor.invoke({ type: 'open-session', sessionId: 'session-a' })
    const run = supervisor.invoke({
      type: 'run',
      sessionId: 'session-a',
      input: { text: 'done already' }
    })
    await vi.waitFor(() => {
      expect(fakeWorker.sent.filter((cmd) => cmd.type === 'run')).toHaveLength(1)
    })

    fakeWorker.emit('message', runEvent('session-a', 'run-started', 'worker-run-a', 1))
    fakeWorker.emit('message', runEvent('session-a', 'run-completed', 'worker-run-a', 2))
    fakeWorker.emit('message', runEvent('session-a', 'tool-completed', 'worker-run-a', 3))

    await expect(supervisor.invoke({ type: 'cancel', sessionId: 'session-a' })).resolves.toEqual({
      cancelled: true
    })
    await expect(run).rejects.toThrow('did not stop in time')

    const failed = events.filter(
      (message) => message.type === 'event' && message.event.type === 'run-failed'
    )
    expect(failed).toHaveLength(0)
  })

  it('reconciles only interrupted sessions on the daemon that owned the worker', async () => {
    const worker = new FakeUtilityProcess()
    let finishCancellation!: (response: Response) => void
    const cancellation = new Promise<Response>((resolve) => {
      finishCancellation = resolve
    })
    const fetchMock = vi.fn(async (url: string) => {
      if (url.endsWith('/cancel')) return cancellation
      return new Response('{}')
    })
    const supervisor = new AgentSupervisor({
      forkWorker: () => {
        queueMicrotask(() => worker.emit('spawn'))
        return worker as unknown as UtilityProcess
      },
      fetch: fetchMock as unknown as typeof fetch
    })
    const original = 'http://127.0.0.1:7614'
    supervisor.setConnection({ address: original, apiKey: null })
    await supervisor.invoke({ type: 'open-session', sessionId: 'idle' })
    await supervisor.invoke({ type: 'open-session', sessionId: 'active' })
    const run = supervisor.invoke({ type: 'run', sessionId: 'active', input: { text: 'run' } })
    const rejected = expect(run).rejects.toThrow('exited')
    await vi.waitFor(() => expect(worker.sent.some((command) => command.type === 'run')).toBe(true))
    await expect(
      supervisor.invoke({ type: 'run', sessionId: 'active', input: { text: 'duplicate' } })
    ).rejects.toThrow('already running')
    worker.emit('exit', 1)
    supervisor.setConnection({ address: 'http://127.0.0.1:7615', apiKey: null })
    finishCancellation(new Response('{}'))

    await rejected
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      `${original}/api/v1/agent/sessions/active/cancel`,
      `${original}/api/v1/agent/sessions/active`
    ])
  })

  it('refuses to cancel unknown sessions and ignores stale exits from killed workers', async () => {
    const workers: FakeUtilityProcess[] = []
    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 })) as unknown as typeof fetch
    const supervisor = new AgentSupervisor({
      cancelGraceMs: 15,
      forkWorker: () => {
        const w = new FakeUtilityProcess()
        workers.push(w)
        queueMicrotask(() => w.emit('spawn'))
        return w as unknown as UtilityProcess
      },
      fetch: fetchMock
    })
    supervisor.setConnection({ address: 'http://127.0.0.1:7614', apiKey: null })

    await expect(
      supervisor.invoke({ type: 'cancel', sessionId: 'never-opened' })
    ).rejects.toThrow('does not know about')

    await supervisor.invoke({ type: 'open-session', sessionId: 'session-1' })
    const firstWorker = workers[0]!
    // Delay the first worker's OS exit so the replacement spawns before it exits.
    firstWorker.ignoreCancel = true
    firstWorker.kill = () => {
      firstWorker.killed = true
      return true
    }

    const hungRun = supervisor.invoke({
      type: 'run',
      sessionId: 'session-1',
      input: { text: 'hang' }
    })
    await vi.waitFor(() => {
      expect(firstWorker.sent.some((cmd) => cmd.type === 'run')).toBe(true)
    })

    await supervisor.invoke({ type: 'cancel', sessionId: 'session-1' })
    await expect(hungRun).rejects.toThrow('did not stop in time')

    // Spawn a replacement worker, then deliver the stale exit from worker 1.
    await supervisor.invoke({ type: 'open-session', sessionId: 'session-2' })
    expect(workers).toHaveLength(2)
    firstWorker.emit('exit', 137)

    // Replacement worker must remain healthy and running.
    expect(supervisor.status().running).toBe(true)
    expect(supervisor.status().crashes).toBe(0)
  })

  it('restarts after a crash and reattaches open sessions', async () => {
    const workers: FakeUtilityProcess[] = []
    const supervisor = new AgentSupervisor({
      forkWorker: () => {
        const worker = new FakeUtilityProcess()
        workers.push(worker)
        queueMicrotask(() => worker.emit('spawn'))
        return worker as unknown as UtilityProcess
      },
      fetch: (async () => new Response('{}')) as unknown as typeof fetch
    })
    supervisor.setConnection({ address: 'http://127.0.0.1:7614', apiKey: null })
    await supervisor.invoke({ type: 'open-session', sessionId: 's1' })
    expect(supervisor.status().state).toBe('running')

    workers[0].emit('exit', 1)
    expect(supervisor.status()).toMatchObject({ state: 'restarting', crashes: 1, lastExitCode: 1 })
    await vi.waitFor(() => expect(supervisor.status().state).toBe('running'), { timeout: 3000 })
    expect(workers).toHaveLength(2)
    expect(
      workers[1].sent.some(
        (command) => command.type === 'open-session' && command.sessionId === 's1'
      )
    ).toBe(true)
  })

  it('stops restarting after a crash loop until asked to restart', async () => {
    let now = 0
    const workers: FakeUtilityProcess[] = []
    const supervisor = new AgentSupervisor({
      now: () => now,
      forkWorker: () => {
        const worker = new FakeUtilityProcess()
        workers.push(worker)
        queueMicrotask(() => worker.emit('spawn'))
        return worker as unknown as UtilityProcess
      },
      fetch: (async () => new Response('{}')) as unknown as typeof fetch
    })
    supervisor.setConnection({ address: 'http://127.0.0.1:7614', apiKey: null })
    for (let attempt = 0; attempt < 4; attempt += 1) {
      await supervisor.warmup()
      workers[workers.length - 1].emit('exit', 139)
      now += 10_000
    }
    expect(supervisor.status()).toMatchObject({ state: 'crash-loop', crashes: 4 })
    await expect(supervisor.warmup()).rejects.toThrow('stopped restarting')
    expect(workers).toHaveLength(4)

    expect(supervisor.restart().state).toBe('idle')
    await supervisor.warmup()
    expect(supervisor.status().state).toBe('running')
  })
})

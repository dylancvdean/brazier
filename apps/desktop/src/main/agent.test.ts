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
})

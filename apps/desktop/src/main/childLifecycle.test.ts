import { EventEmitter } from 'node:events'
import { describe, expect, it, vi } from 'vitest'

import { terminateChildAndWait, type TerminableChild } from './childLifecycle'

class FakeChild extends EventEmitter {
  exitCode: number | null = null
  signalCode: NodeJS.Signals | null = null
  readonly kill = vi.fn((_signal?: NodeJS.Signals | number) => true)
}

describe('terminateChildAndWait', () => {
  it('resolves immediately when no owned child is running', async () => {
    await expect(terminateChildAndWait(undefined)).resolves.toBeUndefined()
    const exited = new FakeChild()
    exited.exitCode = 0
    await expect(terminateChildAndWait(exited as TerminableChild)).resolves.toBeUndefined()
    expect(exited.kill).not.toHaveBeenCalled()
  })

  it('waits for the exit event after graceful termination', async () => {
    const child = new FakeChild()
    let completed = false
    const shutdown = terminateChildAndWait(child as TerminableChild).then(() => {
      completed = true
    })

    expect(child.kill).toHaveBeenCalledWith('SIGTERM')
    await Promise.resolve()
    expect(completed).toBe(false)
    child.signalCode = 'SIGTERM'
    child.emit('exit', null, 'SIGTERM')
    await shutdown
    expect(completed).toBe(true)
  })

  it('can signal a wrapped daemon while still waiting on its Cargo parent', async () => {
    const child = new FakeChild()
    const sendSignal = vi.fn(() => true)
    const shutdown = terminateChildAndWait(child as TerminableChild, { sendSignal })
    expect(sendSignal).toHaveBeenCalledWith('SIGTERM')
    expect(child.kill).not.toHaveBeenCalled()
    child.exitCode = 0
    child.emit('exit', 0, null)
    await expect(shutdown).resolves.toBeUndefined()
  })

  it('forces a wedged child and still waits for the resulting exit', async () => {
    vi.useFakeTimers()
    try {
      const child = new FakeChild()
      const shutdown = terminateChildAndWait(child as TerminableChild, {
        gracefulTimeoutMs: 10,
        forcedTimeoutMs: 10
      })
      await vi.advanceTimersByTimeAsync(10)
      expect(child.kill).toHaveBeenNthCalledWith(1, 'SIGTERM')
      expect(child.kill).toHaveBeenNthCalledWith(2, 'SIGKILL')
      child.signalCode = 'SIGKILL'
      child.emit('exit', null, 'SIGKILL')
      await expect(shutdown).resolves.toBeUndefined()
    } finally {
      vi.useRealTimers()
    }
  })

  it('rejects when even forced termination produces no exit', async () => {
    vi.useFakeTimers()
    try {
      const child = new FakeChild()
      const shutdown = terminateChildAndWait(child as TerminableChild, {
        gracefulTimeoutMs: 10,
        forcedTimeoutMs: 10
      })
      const assertion = expect(shutdown).rejects.toThrow('did not exit after a forced shutdown')
      await vi.advanceTimersByTimeAsync(20)
      await assertion
    } finally {
      vi.useRealTimers()
    }
  })
})

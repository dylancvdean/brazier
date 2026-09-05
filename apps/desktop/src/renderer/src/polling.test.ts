import { afterEach, describe, expect, it, vi } from 'vitest'
import { startPolling } from './polling'

afterEach(() => vi.useRealTimers())

describe('generation polling', () => {
  it('never overlaps slow requests and ignores results after cleanup', async () => {
    vi.useFakeTimers()
    let resolve!: (value: string) => void
    const load = vi.fn(() => new Promise<string>((done) => { resolve = done }))
    const onValue = vi.fn()
    const stop = startPolling(load, onValue, 1000)
    await vi.advanceTimersByTimeAsync(10_000)
    expect(load).toHaveBeenCalledOnce()
    resolve('running')
    await vi.advanceTimersByTimeAsync(1000)
    expect(load).toHaveBeenCalledTimes(2)
    expect(onValue).toHaveBeenCalledExactlyOnceWith('running')
    stop()
    resolve('late result')
    await vi.advanceTimersByTimeAsync(10_000)
    expect(load).toHaveBeenCalledTimes(2)
    expect(onValue).toHaveBeenCalledOnce()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('recovers from startup failure and keeps checking while idle', async () => {
    vi.useFakeTimers()
    const load = vi.fn().mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce(null).mockResolvedValue('new job')
    const onValue = vi.fn()
    const stop = startPolling(load, onValue, 1000)
    await vi.advanceTimersByTimeAsync(2000)
    expect(onValue.mock.calls).toEqual([[null], ['new job']])
    stop()
    expect(vi.getTimerCount()).toBe(0)
  })
})

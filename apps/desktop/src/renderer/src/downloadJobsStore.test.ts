import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { DownloadJob, RecommendationSetup } from './api'
import { createDownloadJobsStore } from './downloadJobsStore'

function job(id: string, status: string): DownloadJob {
  return {
    id,
    repo_id: 'org/repo',
    filename: `${id}.gguf`,
    revision: 'main',
    status,
    bytes_downloaded: null,
    total_bytes: null,
    sha256: null,
    error: null,
    created_at: '',
    updated_at: ''
  }
}

describe('downloadJobsStore', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('keeps the last good list and marks it stale when a poll fails', async () => {
    const loadJobs = vi
      .fn<() => Promise<DownloadJob[]>>()
      .mockResolvedValueOnce([job('a', 'downloading')])
      .mockRejectedValueOnce(new Error('daemon restarting'))
    const store = createDownloadJobsStore({ loadJobs, loadSetups: async () => [] })
    const stop = store.subscribe(() => undefined)
    await vi.advanceTimersByTimeAsync(0)
    expect(store.getSnapshot()).toMatchObject({ loaded: true, stale: false })
    await vi.advanceTimersByTimeAsync(1200)
    expect(store.getSnapshot().jobs.map((entry) => entry.id)).toEqual(['a'])
    expect(store.getSnapshot()).toMatchObject({ stale: true, error: 'daemon restarting' })
    stop()
  })

  it('backs off after failures, idles when nothing moves, and stops unsubscribed', async () => {
    const loadJobs = vi.fn<() => Promise<DownloadJob[]>>().mockRejectedValue(new Error('down'))
    const store = createDownloadJobsStore({
      loadJobs,
      loadSetups: async () => [],
      activeIntervalMs: 100,
      idleIntervalMs: 1000,
      maxBackoffMs: 400
    })
    const stop = store.subscribe(() => undefined)
    await vi.advanceTimersByTimeAsync(0)
    expect(loadJobs).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(199)
    expect(loadJobs).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(loadJobs).toHaveBeenCalledTimes(2)
    await vi.advanceTimersByTimeAsync(400)
    expect(loadJobs).toHaveBeenCalledTimes(3)
    await vi.advanceTimersByTimeAsync(400)
    expect(loadJobs).toHaveBeenCalledTimes(4)

    loadJobs.mockResolvedValue([job('a', 'completed')])
    await vi.advanceTimersByTimeAsync(400)
    expect(loadJobs).toHaveBeenCalledTimes(5)
    await vi.advanceTimersByTimeAsync(999)
    expect(loadJobs).toHaveBeenCalledTimes(5)
    await vi.advanceTimersByTimeAsync(1)
    expect(loadJobs).toHaveBeenCalledTimes(6)

    stop()
    await vi.advanceTimersByTimeAsync(10_000)
    expect(loadJobs).toHaveBeenCalledTimes(6)
  })

  it('shares one poller and announces jobs that settle', async () => {
    const loadJobs = vi
      .fn<() => Promise<DownloadJob[]>>()
      .mockResolvedValueOnce([job('a', 'downloading')])
      .mockResolvedValue([job('a', 'completed')])
    const store = createDownloadJobsStore({ loadJobs, loadSetups: async () => [] })
    const settled = vi.fn()
    store.onSettled(settled)
    const stopA = store.subscribe(() => undefined)
    const stopB = store.subscribe(() => undefined)
    await vi.advanceTimersByTimeAsync(0)
    expect(loadJobs).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1200)
    expect(loadJobs).toHaveBeenCalledTimes(2)
    expect(settled).toHaveBeenCalledWith(expect.objectContaining({ id: 'a', status: 'completed' }))
    stopA()
    stopB()
  })

  it('keeps reporting job progress while a recommendation setup is still loading', async () => {
    let resolveSetups!: (setups: RecommendationSetup[]) => void
    const loadSetups = vi.fn(() => new Promise<RecommendationSetup[]>((resolve) => {
      resolveSetups = resolve
    }))
    const loadJobs = vi.fn<() => Promise<DownloadJob[]>>()
      .mockResolvedValueOnce([job('a', 'downloading')])
      .mockResolvedValue([job('a', 'completed')])
    const store = createDownloadJobsStore({ loadJobs, loadSetups })
    const settled = vi.fn()
    store.onSettled(settled)
    const stop = store.subscribe(() => undefined)

    await vi.advanceTimersByTimeAsync(0)
    expect(store.getSnapshot().jobs).toEqual([job('a', 'downloading')])
    await vi.advanceTimersByTimeAsync(1200)
    expect(settled).toHaveBeenCalledWith(job('a', 'completed'))
    expect(loadSetups).toHaveBeenCalledOnce()

    const setup: RecommendationSetup = {
      id: 'setup-a', recommendation_id: 'recommended-a', categories: [], steps: [], status: 'completed'
    }
    resolveSetups([setup])
    await vi.advanceTimersByTimeAsync(0)
    expect(store.getSnapshot()).toMatchObject({ setups: [setup], jobs: [job('a', 'completed')] })
    stop()
  })

  it('announces a job that finished between polls but not ones present at first load', async () => {
    const loadJobs = vi
      .fn<() => Promise<DownloadJob[]>>()
      .mockResolvedValueOnce([job('old', 'completed')])
      .mockResolvedValue([job('old', 'completed'), job('fast', 'completed')])
    const store = createDownloadJobsStore({ loadJobs, loadSetups: async () => [] })
    const settled = vi.fn()
    store.onSettled(settled)
    const stop = store.subscribe(() => undefined)
    await vi.advanceTimersByTimeAsync(0)
    expect(settled).not.toHaveBeenCalled()
    await store.refresh()
    expect(settled).toHaveBeenCalledTimes(1)
    expect(settled).toHaveBeenCalledWith(expect.objectContaining({ id: 'fast' }))
    stop()
  })

  it('does not let an in-flight poll undo a local update', async () => {
    let resolve: (jobs: DownloadJob[]) => void = () => undefined
    const loadJobs = vi.fn(
      () => new Promise<DownloadJob[]>((done) => (resolve = done))
    )
    const store = createDownloadJobsStore({ loadJobs, loadSetups: async () => [] })
    const stop = store.subscribe(() => undefined)
    store.update(() => [])
    resolve([job('a', 'downloading')])
    await vi.advanceTimersByTimeAsync(0)
    expect(store.getSnapshot().jobs).toEqual([])
    stop()
  })
})

import { useEffect, useSyncExternalStore } from 'react'
import {
  listDownloadJobs,
  listRecommendationSetups,
  type DownloadJob,
  type RecommendationSetup
} from './api'

/** Statuses that mean a job still wants to finish. */
export const OPEN_DOWNLOAD_STATUSES = new Set(['pending', 'downloading', 'paused'])
const MOVING_DOWNLOAD_STATUSES = new Set(['pending', 'downloading'])
const MOVING_SETUP_STATUSES = new Set(['pending', 'running'])

export type DownloadJobsSnapshot = {
  jobs: DownloadJob[]
  setups: RecommendationSetup[]
  /** At least one poll has succeeded since the last reset. */
  loaded: boolean
  /** The last poll failed; `jobs` and `setups` are the last good values. */
  stale: boolean
  error: string | null
}

export type DownloadJobsStoreOptions = {
  loadJobs?: () => Promise<DownloadJob[]>
  loadSetups?: () => Promise<RecommendationSetup[]>
  /** Poll interval while a transfer or setup is moving. */
  activeIntervalMs?: number
  /** Poll interval while nothing is moving, to notice work queued elsewhere. */
  idleIntervalMs?: number
  /** Ceiling for exponential backoff after failed polls. */
  maxBackoffMs?: number
}

const EMPTY: DownloadJobsSnapshot = {
  jobs: [],
  setups: [],
  loaded: false,
  stale: false,
  error: null
}

/**
 * One poller for the daemon's download queue and recommendation setups,
 * shared by every surface that shows them. Polls only while subscribed, fast
 * only while something is moving, backs off after failures, and keeps the
 * last good list instead of blanking it when a request fails.
 */
export function createDownloadJobsStore(options: DownloadJobsStoreOptions = {}) {
  const loadJobs = options.loadJobs ?? listDownloadJobs
  const loadSetups = options.loadSetups ?? listRecommendationSetups
  const activeMs = options.activeIntervalMs ?? 1200
  const idleMs = options.idleIntervalMs ?? 10_000
  const maxBackoffMs = options.maxBackoffMs ?? 30_000

  let snapshot = EMPTY
  let generation = 0
  let failures = 0
  let timer: ReturnType<typeof setTimeout> | undefined
  let inFlight: Promise<void> | null = null
  let setupsInFlight: Promise<RecommendationSetup[]> | null = null
  const listeners = new Set<() => void>()
  const settledListeners = new Set<(job: DownloadJob) => void>()
  const setupSettledListeners = new Set<(setup: RecommendationSetup) => void>()
  let movingSetupIds = new Set<string>()
  let openIds = new Set<string>()
  // Every job id seen so far, so one that finished between polls still settles.
  let seenIds: Set<string> | null = null

  function emit(next: DownloadJobsSnapshot): void {
    snapshot = next
    for (const listener of [...listeners]) listener()
  }

  function nextDelay(): number {
    if (failures > 0) return Math.min(maxBackoffMs, activeMs * 2 ** failures)
    const moving =
      snapshot.jobs.some((job) => MOVING_DOWNLOAD_STATUSES.has(job.status)) ||
      snapshot.setups.some((setup) => MOVING_SETUP_STATUSES.has(setup.status))
    return moving ? activeMs : idleMs
  }

  function schedule(): void {
    clearTimeout(timer)
    timer = undefined
    if (listeners.size === 0) return
    timer = setTimeout(() => void poll(), nextDelay())
  }

  function announceSettled(jobs: DownloadJob[]): void {
    const nextOpen = new Set<string>()
    for (const job of jobs) {
      if (OPEN_DOWNLOAD_STATUSES.has(job.status)) nextOpen.add(job.id)
      else if (openIds.has(job.id) || (seenIds !== null && !seenIds.has(job.id))) {
        for (const listener of [...settledListeners]) listener(job)
      }
    }
    openIds = nextOpen
    seenIds ??= new Set()
    for (const job of jobs) seenIds.add(job.id)
  }

  // A setup writes its mode defaults when it finishes, which can be a poll
  // after its last download settles, so consumers need this event too.
  function announceSetupsSettled(setups: RecommendationSetup[]): void {
    const nextMoving = new Set<string>()
    for (const setup of setups) {
      if (MOVING_SETUP_STATUSES.has(setup.status)) nextMoving.add(setup.id)
      else if (movingSetupIds.has(setup.id)) {
        for (const listener of [...setupSettledListeners]) listener(setup)
      }
    }
    movingSetupIds = nextMoving
  }

  function pollSetups(): void {
    if (setupsInFlight) return
    // Listing setups also lets the daemon advance them; it must not block jobs.
    const current = loadSetups()
    setupsInFlight = current
    void current.then((setups) => {
      if (setupsInFlight !== current) return
      announceSetupsSettled(setups)
      emit({ ...snapshot, setups })
    }, () => undefined).finally(() => {
      if (setupsInFlight !== current) return
      setupsInFlight = null
      if (!inFlight) schedule()
    })
  }

  async function fetchOnce(): Promise<void> {
    const startedIn = generation
    pollSetups()
    try {
      const jobs = await loadJobs()
      if (startedIn !== generation) return
      failures = 0
      announceSettled(jobs)
      emit({
        ...snapshot,
        jobs,
        loaded: true,
        stale: false,
        error: null
      })
    } catch (cause) {
      if (startedIn !== generation) return
      failures += 1
      emit({
        ...snapshot,
        stale: true,
        error: cause instanceof Error ? cause.message : String(cause)
      })
    }
  }

  function poll(): Promise<void> {
    if (inFlight) return inFlight
    const current: Promise<void> = fetchOnce().finally(() => {
      if (inFlight !== current) return
      inFlight = null
      schedule()
    })
    inFlight = current
    return current
  }

  return {
    getSnapshot(): DownloadJobsSnapshot {
      return snapshot
    },
    subscribe(listener: () => void): () => void {
      listeners.add(listener)
      if (listeners.size === 1 && !inFlight) void poll()
      return () => {
        listeners.delete(listener)
        if (listeners.size === 0) {
          clearTimeout(timer)
          timer = undefined
        }
      }
    },
    /** Called when a job leaves pending/downloading/paused between two polls. */
    onSettled(listener: (job: DownloadJob) => void): () => void {
      settledListeners.add(listener)
      return () => {
        settledListeners.delete(listener)
      }
    },
    /** Called when a recommendation setup leaves pending/running between two polls. */
    onSetupSettled(listener: (setup: RecommendationSetup) => void): () => void {
      setupSettledListeners.add(listener)
      return () => {
        setupSettledListeners.delete(listener)
      }
    },
    /** Poll now (coalesced with any poll already in flight). */
    refresh(): Promise<void> {
      if (inFlight) return inFlight.then(() => poll())
      return poll()
    },
    /** Apply a local change ahead of the next poll, such as a dismissal. */
    update(change: (jobs: DownloadJob[]) => DownloadJob[]): void {
      // A poll that began before this change must not restore the old rows.
      generation += 1
      emit({ ...snapshot, jobs: change(snapshot.jobs) })
    },
    /** Forget everything, e.g. after switching connection profiles. */
    reset(): void {
      generation += 1
      failures = 0
      openIds = new Set()
      seenIds = null
      movingSetupIds = new Set()
      inFlight = null
      setupsInFlight = null
      emit(EMPTY)
      if (listeners.size > 0) void poll()
    }
  }
}

export type DownloadJobsStore = ReturnType<typeof createDownloadJobsStore>

export const downloadJobsStore = createDownloadJobsStore()

let scopedProfileId: string | null = null

const subscribeNowhere = (): (() => void) => () => undefined

/**
 * Subscribe to the shared download queue for the active connection profile.
 * A disabled caller reads the last snapshot without keeping the poller alive.
 */
export function useDownloadJobs(profileId: string, enabled = true): DownloadJobsSnapshot {
  useEffect(() => {
    if (!enabled) return
    if (scopedProfileId !== null && scopedProfileId !== profileId) downloadJobsStore.reset()
    scopedProfileId = profileId
  }, [profileId, enabled])
  return useSyncExternalStore(
    enabled ? downloadJobsStore.subscribe : subscribeNowhere,
    downloadJobsStore.getSnapshot
  )
}

/** Run `listener` whenever a recommendation setup settles. */
export function useRecommendationSetupSettled(
  listener: (setup: RecommendationSetup) => void
): void {
  useEffect(() => downloadJobsStore.onSetupSettled(listener), [listener])
}

/** Run `listener` whenever a download job settles. */
export function useDownloadJobSettled(listener: (job: DownloadJob) => void): void {
  useEffect(() => downloadJobsStore.onSettled(listener), [listener])
}

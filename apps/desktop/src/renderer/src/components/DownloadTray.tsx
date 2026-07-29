import { ChevronDown, ChevronRight, Download, Hammer, Pause, Play, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import {
  cancelDownloadJob,
  cancelBuildJob,
  dismissDownloadJob,
  formatBytes,
  listDownloadJobs,
  listRecommendationSetups,
  pauseDownloadJob,
  resumeDownloadJob,
  type DownloadJob
} from '../api'

/** Statuses that mean the job still wants to finish. */
const OPEN_STATUSES = new Set(['pending', 'downloading', 'paused'])

function jobTitle(job: DownloadJob): string {
  if (job.label) return job.label
  const basename = job.filename.split('/').at(-1) ?? job.filename
  return basename === 'snapshot' ? job.repo_id : basename
}

function jobPercent(job: DownloadJob): number | null {
  if (job.bytes_downloaded == null || !job.total_bytes) return null
  return Math.min(100, (job.bytes_downloaded / job.total_bytes) * 100)
}

function isRuntimeBuild(job: DownloadJob): boolean {
  return job.kind === 'runtime-build'
}

/** Bytes per second, smoothed across polls; see {@link useTransferRates}. */
type RateSample = { bytes: number; at: number; rate: number | null }

/** Round an ETA to something worth reading — nobody needs "4211 seconds". */
function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return ''
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))}s left`
  if (seconds < 3600) return `${Math.round(seconds / 60)} min left`
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.round((seconds % 3600) / 60)
  return minutes > 0 ? `${hours}h ${minutes}m left` : `${hours}h left`
}

function jobDetail(job: DownloadJob, rate: number | null): string {
  if (isRuntimeBuild(job)) {
    if (job.status === 'failed') return job.error ?? 'Build failed'
    if (job.status === 'cancelled') return 'Cancelled'
    if (job.status === 'completed') return 'Build complete'
    return job.error ?? 'Preparing source build…'
  }
  if (job.status === 'paused') return 'Paused — resumes where it left off'
  if (job.status === 'pending') return 'Waiting in queue'
  if (job.status === 'failed') return job.error ?? 'Failed'
  if (job.status === 'cancelled') return 'Cancelled'
  if (job.status === 'completed') return 'Done'
  if (job.bytes_downloaded == null) return 'Starting…'
  const total = job.total_bytes ? ` / ${formatBytes(job.total_bytes)}` : ''
  const size = `${formatBytes(job.bytes_downloaded)}${total}`
  if (!rate || rate <= 0) return size
  const speed = `${formatBytes(rate)}/s`
  if (job.total_bytes == null) return `${size} · ${speed}`
  const eta = formatEta((job.total_bytes - job.bytes_downloaded) / rate)
  return eta ? `${size} · ${speed} · ${eta}` : `${size} · ${speed}`
}

/**
 * Transfer rate per job, derived from successive polls.
 *
 * The daemon records only how many bytes have landed, so speed and ETA are
 * worked out here. The rate is smoothed because a raw delta between two polls
 * swings wildly enough to make the ETA jump around and read as broken.
 */
function useTransferRates(jobs: DownloadJob[]): Record<string, number | null> {
  const samples = useRef<Record<string, RateSample>>({})
  const now = Date.now()
  const rates: Record<string, number | null> = {}
  for (const job of jobs) {
    const bytes = job.bytes_downloaded
    if (job.status !== 'downloading' || bytes == null) {
      delete samples.current[job.id]
      rates[job.id] = null
      continue
    }
    const previous = samples.current[job.id]
    if (!previous) {
      samples.current[job.id] = { bytes, at: now, rate: null }
      rates[job.id] = null
      continue
    }
    const elapsed = (now - previous.at) / 1000
    // Ignore repeat polls that landed on the same reading.
    if (elapsed < 0.5 || bytes === previous.bytes) {
      rates[job.id] = previous.rate
      continue
    }
    const instant = Math.max(0, (bytes - previous.bytes) / elapsed)
    const smoothed = previous.rate == null ? instant : previous.rate * 0.7 + instant * 0.3
    samples.current[job.id] = { bytes, at: now, rate: smoothed }
    rates[job.id] = smoothed
  }
  return rates
}

/**
 * Download queue, pinned bottom-right and visible from anywhere in the app.
 *
 * Downloads run one at a time in the daemon, so this is where anything queued
 * behind the current transfer is seen and reordered by pausing.
 */
export function DownloadTray({ onChanged }: { onChanged?: () => void }): React.JSX.Element | null {
  const [jobs, setJobs] = useState<DownloadJob[]>([])
  const [collapsed, setCollapsed] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [actionError, setActionError] = useState<{ id: string; message: string } | null>(null)
  const rates = useTransferRates(jobs)

  async function refresh(): Promise<void> {
    try {
      const [next] = await Promise.all([listDownloadJobs(), listRecommendationSetups()])
      setJobs(next)
    } catch {
      // Daemon may still be starting; the next tick retries.
    }
  }

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(), 1200)
    return () => window.clearInterval(timer)
  }, [])

  const open = jobs.filter((job) => OPEN_STATUSES.has(job.status))
  // Keep a finished job visible briefly so the outcome is not missed.
  const recentlyFinished = jobs
    .filter((job) => !OPEN_STATUSES.has(job.status))
    .filter((job) => Date.now() - new Date(`${job.updated_at}Z`).getTime() < 30_000)
  const visible = [...open, ...recentlyFinished]
  if (visible.length === 0) return null

  async function act(
    job: DownloadJob,
    action: (id: string) => Promise<void>
  ): Promise<void> {
    setBusy(job.id)
    setActionError(null)
    try {
      await action(job.id)
      await refresh()
      onChanged?.()
    } catch (cause) {
      // A refused resume leaves the row exactly as it was, so without this the
      // button looks like it did nothing at all.
      setActionError({
        id: job.id,
        message: cause instanceof Error ? cause.message : String(cause)
      })
    } finally {
      setBusy(null)
    }
  }

  const active = open.filter((job) => job.status === 'downloading').length
  const waiting = open.length - active

  return (
    <div className="download-tray" role="status" aria-live="polite">
      <button
        className="download-tray-head"
        onClick={() => setCollapsed((value) => !value)}
        title={collapsed ? 'Show downloads' : 'Hide downloads'}
      >
        {collapsed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
        <Download size={13} />
        <strong>Activity</strong>
        <span className="download-tray-count">
          {active > 0 ? `${active} running` : 'idle'}
          {waiting > 0 ? ` · ${waiting} queued` : ''}
        </span>
      </button>

      {!collapsed && (
        <div className="download-tray-list">
          {visible.map((job) => {
            const percent = jobPercent(job)
            const running = job.status === 'downloading'
            const pending = job.status === 'pending'
            const paused = job.status === 'paused'
            const finished = !OPEN_STATUSES.has(job.status)
            const runtimeBuild = isRuntimeBuild(job)
            return (
              <div className={`download-tray-item ${job.status}`} key={job.id}>
                <div className="download-tray-title">
                  <span className="download-tray-name" title={`${job.repo_id} · ${job.filename}`}>
                    {jobTitle(job)}
                  </span>
                  <span className="download-tray-pct">
                    {percent != null ? `${Math.round(percent)}%` : ''}
                  </span>
                </div>
                {!finished && (
                  <div className="progress-track compact">
                    <div
                      className={`progress-fill ${paused ? 'paused' : ''}`}
                      style={{ width: `${Math.min(100, Math.max(4, percent ?? (running ? 8 : 2)))}%` }}
                    />
                  </div>
                )}
                <div className="download-tray-row">
                  <span className="download-tray-detail">{jobDetail(job, rates[job.id] ?? null)}</span>
                  <div className="download-tray-actions">
                    {!runtimeBuild && (running || pending) && (
                      <button
                        title="Pause"
                        disabled={busy === job.id}
                        onClick={() => void act(job, pauseDownloadJob)}
                      >
                        <Pause size={12} />
                      </button>
                    )}
                    {!runtimeBuild && (paused || job.status === 'failed') && (
                      <button
                        title="Resume"
                        disabled={busy === job.id}
                        onClick={() => void act(job, resumeDownloadJob)}
                      >
                        <Play size={12} />
                      </button>
                    )}
                    <button
                      title={finished ? 'Dismiss' : 'Cancel'}
                      disabled={busy === job.id}
                      onClick={() =>
                        void act(
                          job,
                          finished
                            ? dismissDownloadJob
                            : runtimeBuild
                              ? cancelBuildJob
                              : cancelDownloadJob
                        )
                      }
                    >
                      {runtimeBuild && !finished ? <Hammer size={12} /> : <X size={12} />}
                    </button>
                  </div>
                </div>
                {actionError?.id === job.id && (
                  <p className="download-tray-error">{actionError.message}</p>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

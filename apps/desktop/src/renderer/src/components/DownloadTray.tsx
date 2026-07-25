import { ChevronDown, ChevronRight, Download, Pause, Play, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import {
  cancelDownloadJob,
  formatBytes,
  listDownloadJobs,
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

function jobDetail(job: DownloadJob): string {
  if (job.status === 'paused') return 'Paused — resumes where it left off'
  if (job.status === 'pending') return 'Waiting in queue'
  if (job.status === 'failed') return job.error ?? 'Failed'
  if (job.status === 'cancelled') return 'Cancelled'
  if (job.status === 'completed') return 'Done'
  if (job.bytes_downloaded == null) return 'Starting…'
  const total = job.total_bytes ? ` / ${formatBytes(job.total_bytes)}` : ''
  return `${formatBytes(job.bytes_downloaded)}${total}`
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

  async function refresh(): Promise<void> {
    try {
      setJobs(await listDownloadJobs())
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
    try {
      await action(job.id)
      await refresh()
      onChanged?.()
    } catch {
      // Surfaced by the row's own status on the next refresh.
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
        <strong>Downloads</strong>
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
                  <span className="download-tray-detail">{jobDetail(job)}</span>
                  <div className="download-tray-actions">
                    {(running || pending) && (
                      <button
                        title="Pause"
                        disabled={busy === job.id}
                        onClick={() => void act(job, pauseDownloadJob)}
                      >
                        <Pause size={12} />
                      </button>
                    )}
                    {(paused || job.status === 'failed') && (
                      <button
                        title="Resume"
                        disabled={busy === job.id}
                        onClick={() => void act(job, resumeDownloadJob)}
                      >
                        <Play size={12} />
                      </button>
                    )}
                    {!finished && (
                      <button
                        title="Cancel"
                        disabled={busy === job.id}
                        onClick={() => void act(job, cancelDownloadJob)}
                      >
                        <X size={12} />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

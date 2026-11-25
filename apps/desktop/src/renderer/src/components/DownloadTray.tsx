import {
  Check,
  ChevronDown,
  ChevronRight,
  Download,
  Hammer,
  Pause,
  Play,
  RotateCcw,
  X
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  cancelDownloadJob,
  cancelBuildJob,
  dismissDownloadJob,
  formatBytes,
  pauseDownloadJob,
  resumeDownloadJob,
  type DownloadJob,
  type LocalModel
} from '../api'
import { useConnectionProfile } from '../connectionProfile'
import { stripGgufShardSuffix } from '../model-utils'
import { downloadJobsStore, useDownloadJobSettled, useDownloadJobs } from '../downloadJobsStore'

/** Statuses that mean the job still wants to finish. */
const OPEN_STATUSES = new Set(['pending', 'downloading', 'paused'])

/** Parse both SQLite UTC timestamps ("YYYY-MM-DD HH:MM:SS") and RFC3339 ISO strings ("...Z"). */
export function parseDownloadTimestampMs(value: string): number {
  const trimmed = value.trim()
  if (!trimmed) return Number.NaN
  const normalized = /[zZ]$|[+-]\d{2}:?\d{2}$/.test(trimmed) ? trimmed : `${trimmed}Z`
  return Date.parse(normalized)
}

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

function isProjectorOrDraftFile(filename: string): boolean {
  const lower = filename.toLowerCase()
  return lower.includes('mmproj')
}

/**
 * Match a settled download job to its installed `LocalModel` so the user can
 * activate a freshly downloaded model in one click. Returns null for runtime
 * builds, companion projectors, and ASR-only weights.
 */
export function resolveCompletedJobModel(
  job: DownloadJob,
  models: readonly LocalModel[]
): LocalModel | null {
  if (job.status !== 'completed' || isRuntimeBuild(job)) return null
  if (isProjectorOrDraftFile(job.filename)) return null

  const selectable = models.filter(
    (model) => !model.id.startsWith('whisper:') && !model.id.startsWith('streaming-asr:')
  )
  if (selectable.length === 0) return null

  const directCandidates = [
    `gguf:${job.repo_id}/${job.filename}`,
    `mlx:${job.repo_id}`,
    `mlx-vlm:${job.repo_id}`,
    `personaplex:${job.repo_id}`,
    `sdcpp-image:${job.repo_id}`,
    `sdcpp-video:${job.repo_id}`
  ]
  for (const candidateId of directCandidates) {
    const exact = selectable.find((model) => model.id === candidateId)
    if (exact) return exact
  }

  // Sharded GGUF downloads or sd.cpp bundles whose key derives from repo/filename.
  const strippedJobFile = stripGgufShardSuffix(job.filename.split('/').at(-1) ?? job.filename)
    .toLowerCase()
  const byShardOrRepo = selectable.find((model) => {
    if (model.id.startsWith('gguf:')) {
      const key = model.id.slice('gguf:'.length)
      if (!key.toLowerCase().startsWith(`${job.repo_id.toLowerCase()}/`)) return false
      const base = stripGgufShardSuffix(key.split('/').at(-1) ?? key).toLowerCase()
      return base === strippedJobFile
    }
    if (model.id.startsWith('sdcpp-image:') || model.id.startsWith('sdcpp-video:')) {
      return (
        model.repo_id === job.repo_id ||
        model.id.toLowerCase().endsWith(`:${job.repo_id.toLowerCase()}`) ||
        (job.filename !== 'snapshot' && model.id.toLowerCase().endsWith(`:${job.filename.toLowerCase()}`))
      )
    }
    return false
  })
  return byShardOrRepo ?? null
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
  if (job.error) return job.error
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
export function DownloadTray({
  models = [],
  activeModelIds = [],
  onUseModel,
  onChanged
}: {
  models?: LocalModel[]
  activeModelIds?: string[]
  onUseModel?: (model: LocalModel) => void
  onChanged?: () => void
}): React.JSX.Element | null {
  const profile = useConnectionProfile()
  const { jobs, stale } = useDownloadJobs(profile.id)
  const [collapsed, setCollapsed] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [actionError, setActionError] = useState<{ id: string; message: string } | null>(null)
  const rates = useTransferRates(jobs)
  const onChangedRef = useRef(onChanged)
  onChangedRef.current = onChanged
  useDownloadJobSettled(useCallback(() => onChangedRef.current?.(), []))

  useEffect(() => {
    setActionError(null)
  }, [profile.id])

  const open = jobs.filter((job) => OPEN_STATUSES.has(job.status))
  // Keep a finished job visible briefly so the outcome is not missed.
  const recentlyFinished = jobs
    .filter((job) => !OPEN_STATUSES.has(job.status))
    .filter((job) => {
      const updatedMs = parseDownloadTimestampMs(job.updated_at)
      return Number.isFinite(updatedMs) && Date.now() - updatedMs < 30_000
    })
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
      if (action === dismissDownloadJob) {
        downloadJobsStore.update((current) => current.filter((currentJob) => currentJob.id !== job.id))
      } else if (action === cancelDownloadJob || action === cancelBuildJob) {
        downloadJobsStore.update((current) =>
          current.map((currentJob) =>
            currentJob.id === job.id
              ? {
                  ...currentJob,
                  status: 'cancelled',
                  error: 'cancelled by user',
                  updated_at: new Date().toISOString()
                }
              : currentJob
          )
        )
      }
      onChangedRef.current?.()
      void downloadJobsStore.refresh()
    } catch (cause) {
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
        type="button"
        className="download-tray-head"
        onClick={() => setCollapsed((value) => !value)}
        title={collapsed ? 'Show downloads' : 'Hide downloads'}
      >
        {collapsed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
        <Download size={13} />
        {stale ? (
          <span
            className="turn-badge failed"
            title="The last refresh failed; showing the most recent known state."
          >
            offline
          </span>
        ) : null}
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
            const completedModel =
              onUseModel && job.status === 'completed'
                ? resolveCompletedJobModel(job, models)
                : null
            const alreadyActive =
              completedModel != null && activeModelIds.includes(completedModel.id)
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
                    {completedModel && onUseModel && (
                      <button
                        type="button"
                        className="download-tray-use"
                        disabled={alreadyActive}
                        title={
                          alreadyActive ? 'Already selected' : `Use ${jobTitle(job)}`
                        }
                        onClick={() => onUseModel(completedModel)}
                      >
                        {alreadyActive ? (
                          <>
                            <Check size={11} /> In use
                          </>
                        ) : (
                          'Use'
                        )}
                      </button>
                    )}
                    {!runtimeBuild && (running || pending) && (
                      <button
                        type="button"
                        title="Pause"
                        disabled={busy === job.id}
                        onClick={() => void act(job, pauseDownloadJob)}
                      >
                        <Pause size={12} />
                      </button>
                    )}
                    {!runtimeBuild && (paused || job.status === 'failed') && (
                      <button
                        type="button"
                        title={paused ? 'Resume' : 'Retry'}
                        aria-label={paused ? 'Resume' : 'Retry'}
                        disabled={busy === job.id}
                        onClick={() => void act(job, resumeDownloadJob)}
                      >
                        {paused ? <Play size={12} /> : <RotateCcw size={12} />}
                      </button>
                    )}
                    <button
                      type="button"
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

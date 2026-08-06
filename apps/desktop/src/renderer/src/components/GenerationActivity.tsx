import { Image, LoaderCircle, Square, Video } from 'lucide-react'
import { useEffect, useState } from 'react'
import {
  cancelGeneration,
  fetchActiveGeneration,
  fetchBlobObjectUrl,
  type ActiveGeneration
} from '../api'

function elapsedLabel(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`
}

/**
 * The generation running right now, with the prompt behind it and a way to stop.
 *
 * A model can start a render that runs for hours from a prompt the user never
 * saw. Showing that prompt — and the picture it was handed — while the job is
 * still running is what makes stopping it a decision rather than a guess.
 */
export function GenerationActivity({
  onStopped
}: {
  onStopped?: () => void
}): React.JSX.Element | null {
  const [active, setActive] = useState<ActiveGeneration | null>(null)
  const [initImageUrl, setInitImageUrl] = useState<string | null>(null)
  const [stopping, setStopping] = useState(false)

  useEffect(() => {
    let cancelled = false
    void fetchActiveGeneration()
      .then((current) => {
        if (!cancelled) setActive(current)
      })
      .catch(() => {
        // Daemon may be restarting; the next tick retries.
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!active) return
    let cancelled = false
    const timer = window.setInterval(() => {
      void fetchActiveGeneration()
        .then((current) => {
          if (!cancelled) setActive(current)
        })
        .catch(() => {
          // Daemon may be restarting; the next tick retries.
        })
    }, 1000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [active])

  const initBlob = active?.init_image_blob ?? null
  useEffect(() => {
    if (!initBlob) {
      setInitImageUrl(null)
      return
    }
    let cancelled = false
    void fetchBlobObjectUrl(initBlob)
      .then((url) => {
        if (!cancelled) setInitImageUrl(url)
      })
      .catch(() => {
        // The thumbnail is a nicety; the prompt is the part that matters.
      })
    return () => {
      cancelled = true
    }
  }, [initBlob])

  useEffect(() => {
    // Once the job is gone, let the surface refresh whatever it produced.
    if (!active) setStopping(false)
  }, [active])

  if (!active) return null

  async function stop(): Promise<void> {
    setStopping(true)
    try {
      await cancelGeneration()
      onStopped?.()
    } catch {
      setStopping(false)
    }
  }

  const byModel = active.origin === 'model'
  const stepPercent =
    active.total_steps > 0
      ? Math.min(100, Math.round((active.current_step / active.total_steps) * 100))
      : 0
  return (
    <section className="generation-activity" role="status" aria-live="polite">
      <div className="generation-activity-head">
        <LoaderCircle className="spin" size={14} />
        {active.modality === 'video' ? <Video size={14} /> : <Image size={14} />}
        <strong>
          {byModel ? 'The model is generating' : 'Generating'} a {active.modality}
        </strong>
        <span className="generation-activity-timing">
          {active.current_step > 0
            ? `Step ${active.current_step} of ${active.total_steps}`
            : `Preparing · ${active.total_steps} steps`}
          {' · '}
          {elapsedLabel(active.elapsed_secs)} · gives up after{' '}
          {elapsedLabel(active.timeout_secs)}
        </span>
        <button
          type="button"
          className="chip-button subtle"
          disabled={stopping}
          onClick={() => void stop()}
          title="Stop this generation"
        >
          <Square size={12} fill="currentColor" />
          {stopping ? 'Stopping…' : 'Stop'}
        </button>
      </div>
      <div
        className="generation-activity-progress"
        role="progressbar"
        aria-label="Diffusion steps"
        aria-valuemin={0}
        aria-valuemax={active.total_steps}
        aria-valuenow={active.current_step}
      >
        <span style={{ width: `${stepPercent}%` }} />
      </div>
      <div className="generation-activity-body">
        {initImageUrl && (
          <img
            className="generation-activity-init"
            src={initImageUrl}
            alt="Image the generation was given to work from"
          />
        )}
        <div className="generation-activity-prompt">
          <p>{active.prompt}</p>
          {active.negative_prompt ? <p className="negative">Avoiding: {active.negative_prompt}</p> : null}
          <span className="generation-activity-model">{active.model_id}</span>
        </div>
      </div>
    </section>
  )
}

import { useEffect, useState } from 'react'
import type { AgentWorkerStatus } from '../../../agent/core/protocol'

/** Visible worker lifecycle: shown only while the agent worker is restarting or stopped. */
export function AgentWorkerBanner({ onError }: { onError: (message: string | null) => void }): React.JSX.Element | null {
  const [status, setStatus] = useState<AgentWorkerStatus | null>(null)

  useEffect(() => {
    let active = true
    const stop = window.brazier.agent.onStatus((next) => {
      if (active) setStatus(next)
    })
    void window.brazier.agent
      .status()
      .then((next) => {
        if (active) setStatus(next)
      })
      .catch((cause: unknown) => {
        if (active) onError(cause instanceof Error ? cause.message : String(cause))
      })
    return () => {
      active = false
      stop()
    }
  }, [onError])

  if (!status || (status.state !== 'restarting' && status.state !== 'crash-loop')) return null
  const exit = status.lastExitCode === null ? '' : ` (exit code ${status.lastExitCode})`
  if (status.state === 'restarting') {
    return (
      <div className="agent-worker-banner" role="status">
        The agent worker crashed{exit}; restarting and reattaching open sessions…
      </div>
    )
  }
  return (
    <div className="agent-worker-banner stopped" role="alert">
      <span>
        The agent worker crashed repeatedly{exit}, so Brazier stopped restarting it.
      </span>
      <button
        type="button"
        onClick={() =>
          void window.brazier.agent
            .restart()
            .then(setStatus)
            .catch((cause: unknown) =>
              onError(cause instanceof Error ? cause.message : String(cause))
            )
        }
      >
        Restart worker
      </button>
    </div>
  )
}

export interface TerminableChild {
  exitCode: number | null
  signalCode: NodeJS.Signals | null
  kill(signal?: number | NodeJS.Signals): boolean
  once(
    event: 'exit',
    listener: (code: number | null, signal: NodeJS.Signals | null) => void
  ): this
  once(event: 'error', listener: (error: Error) => void): this
  off(event: 'exit', listener: (code: number | null, signal: NodeJS.Signals | null) => void): this
  off(event: 'error', listener: (error: Error) => void): this
}

export type TerminationOptions = {
  gracefulTimeoutMs?: number
  forcedTimeoutMs?: number
  sendSignal?: (signal: NodeJS.Signals) => boolean
}

/**
 * Terminate an owned child and do not claim shutdown until its exit event has
 * actually arrived. A bounded SIGKILL fallback prevents application quit from
 * hanging forever on a wedged inference/runtime cleanup path.
 */
export function terminateChildAndWait(
  child: TerminableChild | undefined,
  options: TerminationOptions = {}
): Promise<void> {
  if (!child || child.exitCode !== null || child.signalCode !== null) return Promise.resolve()

  const gracefulTimeoutMs = options.gracefulTimeoutMs ?? 10_000
  const forcedTimeoutMs = options.forcedTimeoutMs ?? 2_000
  const sendSignal = options.sendSignal ?? ((signal: NodeJS.Signals) => child.kill(signal))
  return new Promise((resolve, reject) => {
    let gracefulTimer: ReturnType<typeof setTimeout> | undefined
    let forcedTimer: ReturnType<typeof setTimeout> | undefined
    let settled = false

    const cleanup = (): void => {
      if (gracefulTimer) clearTimeout(gracefulTimer)
      if (forcedTimer) clearTimeout(forcedTimer)
      child.off('exit', onExit)
      child.off('error', onError)
    }
    const settle = (error?: Error): void => {
      if (settled) return
      settled = true
      cleanup()
      if (error) reject(error)
      else resolve()
    }
    const onExit = (): void => settle()
    const onError = (cause: Error): void => settle(cause)

    child.once('exit', onExit)
    child.once('error', onError)
    if (!sendSignal('SIGTERM')) {
      settle(new Error('the child process did not accept the shutdown signal'))
      return
    }
    gracefulTimer = setTimeout(() => {
      if (child.exitCode !== null || child.signalCode !== null) {
        settle()
        return
      }
      if (!sendSignal('SIGKILL')) {
        settle(new Error('the child process did not accept the forced shutdown signal'))
        return
      }
      forcedTimer = setTimeout(() => {
        settle(new Error('the child process did not exit after a forced shutdown'))
      }, forcedTimeoutMs)
    }, gracefulTimeoutMs)
  })
}

/**
 * Abort-signal helpers that work where `AbortSignal.any` and
 * `AbortSignal.timeout` are missing (React Native's Hermes). Node and Electron
 * use the native implementations.
 */

type AbortSignalStatics = {
  any?: (signals: AbortSignal[]) => AbortSignal
  timeout?: (ms: number) => AbortSignal
}

const statics = AbortSignal as unknown as AbortSignalStatics

/** A signal that aborts after `ms`. */
export function timeoutSignal(ms: number): AbortSignal {
  if (typeof statics.timeout === 'function') return statics.timeout(ms)
  const controller = new AbortController()
  setTimeout(() => controller.abort(), ms)
  return controller.signal
}

/** A signal that aborts as soon as any of `signals` does. */
export function anySignal(signals: AbortSignal[]): AbortSignal {
  if (signals.length === 1) return signals[0] as AbortSignal
  if (typeof statics.any === 'function') return statics.any(signals)
  const controller = new AbortController()
  const abort = (): void => {
    for (const signal of signals) signal.removeEventListener('abort', abort)
    controller.abort()
  }
  for (const signal of signals) {
    if (signal.aborted) {
      abort()
      break
    }
    signal.addEventListener('abort', abort)
  }
  return controller.signal
}

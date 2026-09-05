export type DaemonAvailability = 'checking' | 'healthy' | 'offline'

let availability: DaemonAvailability = 'checking'
let connectionAbort = new AbortController()

export class DaemonOfflineError extends Error {
  constructor() {
    super('The selected Brazier daemon is not ready. Reconnect or switch connection profiles before making changes.')
    this.name = 'DaemonOfflineError'
  }
}

export function setDaemonAvailability(next: DaemonAvailability): void {
  if (next === 'checking') {
    connectionAbort.abort()
    connectionAbort = new AbortController()
  }
  availability = next
}

export function daemonAvailability(): DaemonAvailability {
  return availability
}

export function assertDaemonMutationAllowed(init?: RequestInit): void {
  const method = (init?.method ?? 'GET').toUpperCase()
  if (method === 'GET' || method === 'HEAD' || method === 'OPTIONS') return
  if (availability !== 'healthy') throw new DaemonOfflineError()
}

/** Track reachability and enforce the offline read-only boundary for daemon traffic. */
export async function daemonFetch(
  input: string | URL | Request,
  init?: RequestInit
): Promise<Response> {
  const request = input instanceof Request ? input : undefined
  assertDaemonMutationAllowed({ method: init?.method ?? request?.method })
  const callerSignal = init?.signal ?? request?.signal
  const signal = callerSignal
    ? AbortSignal.any([connectionAbort.signal, callerSignal])
    : connectionAbort.signal
  try {
    signal.throwIfAborted()
    const response = await fetch(input, { ...init, signal })
    signal.throwIfAborted()
    // Any HTTP response proves the host is reachable, including auth and
    // validation failures. Only transport failure means offline.
    availability = 'healthy'
    return response
  } catch (cause) {
    if (!signal.aborted && !(cause instanceof DOMException && cause.name === 'AbortError')) {
      availability = 'offline'
    }
    throw cause
  }
}

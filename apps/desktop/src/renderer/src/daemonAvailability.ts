export type DaemonAvailability = 'checking' | 'healthy' | 'offline'

let availability: DaemonAvailability = 'checking'

export class DaemonOfflineError extends Error {
  constructor() {
    super('The selected Brazier daemon is not ready. Reconnect or switch connection profiles before making changes.')
    this.name = 'DaemonOfflineError'
  }
}

export function setDaemonAvailability(next: DaemonAvailability): void {
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
  assertDaemonMutationAllowed(init)
  try {
    const response = await fetch(input, init)
    // Any HTTP response proves the host is reachable, including auth and
    // validation failures. Only transport failure means offline.
    availability = 'healthy'
    return response
  } catch (cause) {
    if (cause instanceof DaemonOfflineError) throw cause
    availability = 'offline'
    throw cause
  }
}

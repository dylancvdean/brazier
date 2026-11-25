/**
 * Daemon connectivity as one state machine:
 *
 *   checking ─reachable─▶ healthy ─unreachable─▶ offline ─reachable─▶ rehydrating ─rehydrated─▶ healthy
 *
 * A manual retry goes back to `checking` and always rehydrates on success, so a
 * reconnect cannot leave stale models or conversations on screen.
 */
export type DaemonConnection =
  | { phase: 'checking'; rehydrate: boolean; version: string }
  | { phase: 'healthy'; version: string }
  | { phase: 'rehydrating'; version: string }
  | { phase: 'offline'; version: string }

export type DaemonConnectionEvent =
  | { type: 'retry' }
  | { type: 'reachable'; version?: string }
  | { type: 'unreachable' }
  | { type: 'rehydrated' }

export const INITIAL_DAEMON_CONNECTION: DaemonConnection = {
  phase: 'checking',
  rehydrate: false,
  version: ''
}

export function daemonConnectionReducer(
  state: DaemonConnection,
  event: DaemonConnectionEvent
): DaemonConnection {
  switch (event.type) {
    case 'retry':
      return { phase: 'checking', rehydrate: true, version: state.version }
    case 'reachable': {
      const version = event.version ?? state.version
      if (state.phase === 'offline' || (state.phase === 'checking' && state.rehydrate)) {
        return { phase: 'rehydrating', version }
      }
      if (state.phase === 'rehydrating') return { phase: 'rehydrating', version }
      return state.phase === 'healthy' && state.version === version
        ? state
        : { phase: 'healthy', version }
    }
    case 'unreachable':
      return state.phase === 'offline' ? state : { phase: 'offline', version: state.version }
    case 'rehydrated':
      return state.phase === 'rehydrating' ? { phase: 'healthy', version: state.version } : state
  }
}

/** The request-layer gate for an event; `null` leaves it unchanged. */
export function availabilityForEvent(
  event: DaemonConnectionEvent
): 'checking' | 'healthy' | 'offline' | null {
  switch (event.type) {
    case 'retry':
      return 'checking'
    case 'reachable':
      return 'healthy'
    case 'unreachable':
      return 'offline'
    case 'rehydrated':
      return null
  }
}

/** The three-way status the shell shows; rehydration is a healthy connection. */
export function daemonStatusOf(state: DaemonConnection): 'checking' | 'healthy' | 'offline' {
  return state.phase === 'rehydrating' ? 'healthy' : state.phase
}

import { daemonFetch, setDaemonAvailability } from './daemonAvailability'

type Connection = Awaited<ReturnType<typeof window.brazier.getConnection>>
type ConnectionProfile = Connection['profile']
const ACTIVE_CONNECTION_PROFILE_KEY = 'brazier.activeConnectionProfile.v1'
let connectionPromise: Promise<Connection> | undefined

export function invalidateConnectionCache(): void {
  connectionPromise = undefined
}

export function rememberConnectionProfile(profile: ConnectionProfile): void {
  try {
    localStorage.setItem(ACTIVE_CONNECTION_PROFILE_KEY, profile.id)
  } catch {
    // Profile-scoped caches are best-effort.
  }
}

export function rememberedConnectionProfileId(): string {
  try {
    return localStorage.getItem(ACTIVE_CONNECTION_PROFILE_KEY) || 'local'
  } catch {
    return 'local'
  }
}

if (typeof window !== 'undefined' && window.brazier?.onConnectionProfileChanged) {
  window.brazier.onConnectionProfileChanged((profile) => {
    invalidateConnectionCache()
    rememberConnectionProfile(profile)
    setDaemonAvailability('checking')
  })
}

export async function connection(): Promise<Connection> {
  connectionPromise ??= window.brazier.getConnection()
  const pending = connectionPromise
  try {
    const ready = await pending
    if (connectionPromise !== pending) {
      throw new DOMException('The selected connection changed.', 'AbortError')
    }
    rememberConnectionProfile(ready.profile)
    return ready
  } catch (error) {
    if (connectionPromise === pending) {
      connectionPromise = undefined
      setDaemonAvailability('offline')
    }
    throw error
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const daemon = await connection()
  const headers = new Headers(init?.headers)
  headers.set('content-type', 'application/json')
  const response = await daemonFetch(`${daemon.address}${path}`, { ...init, headers })
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as {
      error?: { message?: string }
    } | null
    throw new Error(payload?.error?.message ?? `Request failed with status ${response.status}.`)
  }
  if (response.status === 204 || response.status === 205) return undefined as T
  return response.json() as Promise<T>
}

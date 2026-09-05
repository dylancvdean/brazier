import { useEffect, useState } from 'react'
import { fetchBlob } from './api'
import { useConnectionProfile } from './connectionProfile'
import { ObjectUrlCache } from './objectUrlCache'

const cache = new ObjectUrlCache()
type State = { key: string; urls: Record<string, string>; failed: Record<string, boolean> }

export function useBlobUrls(hashes: string[], retry = 0): Omit<State, 'key'> {
  const profile = useConnectionProfile()
  const key = JSON.stringify([profile.id, [...new Set(hashes)].sort(), retry])
  const [state, setState] = useState<State>({ key, urls: {}, failed: {} })

  useEffect(() => {
    let active = true
    const [profileId, ids] = JSON.parse(key) as [string, string[], number]
    setState({ key, urls: {}, failed: {} })
    const leases = ids.map((id) => {
      const lease = cache.acquire(JSON.stringify([profileId, id, retry]), (signal) => fetchBlob(id, signal))
      void lease.url.then(
        (url) => {
          if (active) setState((current) => ({ ...current, urls: { ...current.urls, [id]: url } }))
        },
        () => {
          if (active) setState((current) => ({ ...current, failed: { ...current.failed, [id]: true } }))
        }
      )
      return lease
    })
    return () => {
      active = false
      leases.forEach((lease) => lease.release())
    }
  }, [key, retry])

  return state.key === key ? state : { urls: {}, failed: {} }
}

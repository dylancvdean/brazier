import { createContext, useContext } from 'react'

export type ActiveConnectionProfile = Awaited<
  ReturnType<Window['brazier']['getCurrentConnectionProfile']>
>

export const LOCAL_CONNECTION_PROFILE: ActiveConnectionProfile = {
  id: 'local',
  name: 'Local',
  kind: 'local',
  baseUrl: null,
  hostLabel: 'This computer'
}

export const ConnectionProfileContext = createContext<ActiveConnectionProfile>(
  LOCAL_CONNECTION_PROFILE
)

export function useConnectionProfile(): ActiveConnectionProfile {
  return useContext(ConnectionProfileContext)
}

export function daemonPathLabel(profile: ActiveConnectionProfile): string {
  return profile.kind === 'local'
    ? 'Path on this computer'
    : `Path on ${profile.name} (${profile.hostLabel})`
}

/** The profile's host, phrased to sit inside a sentence. */
export function hostPhrase(profile: ActiveConnectionProfile): string {
  return profile.kind === 'local' ? 'this computer' : profile.hostLabel
}

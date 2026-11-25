const PREFIX = 'brazier-pairing:'

export const PAIRING_SCOPES = ['inference', 'management', 'agent'] as const
export type PairingScope = typeof PAIRING_SCOPES[number]

export type PairingInvite = {
  version: 1
  origin: string
  pairingId: string
  code: string
  hostName: string
  scopes: PairingScope[]
  expiresAt: number
}

type LegacyPairingInvite = {
  version: 0
  origin?: string
  pairingId: string
  code: string
}

export function normalizeShareableOrigin(input: string): string {
  let url: URL
  try {
    url = new URL(input.trim())
  } catch {
    throw new Error('Enter the HTTPS address that the other device can reach.')
  }
  const host = url.hostname.replace(/\.$/, '')
  if (host === 'localhost' || host.endsWith('.localhost') || host === '[::1]' ||
      host === '[::]' || host.startsWith('[::ffff:') || host.startsWith('127.') ||
      host === '0.0.0.0') {
    throw new Error('This address does not reach this daemon from another device. Enter its reachable HTTPS address.')
  }
  if (url.protocol !== 'https:' || url.username || url.password ||
      url.pathname !== '/' || url.search || url.hash) {
    throw new Error('Use a trusted HTTPS daemon origin without credentials, a path, a query, or a fragment.')
  }
  return url.origin
}

export function encodePairingInvite(invite: PairingInvite): string {
  return PREFIX + JSON.stringify({ ...invite, origin: normalizeShareableOrigin(invite.origin) })
}

export function parsePairingInvite(input: string): PairingInvite | LegacyPairingInvite | null {
  const text = input.trim()
  if (text.length > 4_096) return null
  if (!text.startsWith(PREFIX)) {
    const match = /^(?:Daemon URL: ([^\s]+)\r?\n)?Pairing id: (\S+)\r?\nPairing code: (\S+)$/i.exec(text)
    return match ? { version: 0, origin: match[1], pairingId: match[2], code: match[3] } : null
  }

  try {
    const value = JSON.parse(text.slice(PREFIX.length)) as PairingInvite
    if (value?.version !== 1 || typeof value.origin !== 'string' ||
        typeof value.pairingId !== 'string' || !/^[\w-]{1,128}$/.test(value.pairingId) ||
        typeof value.code !== 'string' || !/^[\w-]{1,128}$/.test(value.code) ||
        typeof value.hostName !== 'string' || !value.hostName.trim() || value.hostName.length > 160 ||
        !Array.isArray(value.scopes) || !value.scopes.length || value.scopes.length > PAIRING_SCOPES.length ||
        !value.scopes.every((scope) => PAIRING_SCOPES.includes(scope)) ||
        !Number.isSafeInteger(value.expiresAt) || value.expiresAt <= 0) {
      return null
    }
    return { ...value, origin: normalizeShareableOrigin(value.origin) }
  } catch {
    return null
  }
}

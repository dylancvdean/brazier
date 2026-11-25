import { describe, expect, it } from 'vitest'
import {
  encodePairingInvite, normalizeShareableOrigin, parsePairingInvite, type PairingInvite
} from '../../../../../packages/pairing/src/invite'

const invite: PairingInvite = {
  version: 1, origin: 'https://host.example', hostName: 'Studio', pairingId: 'id-1',
  code: 'ONE-TIME', scopes: ['inference'], expiresAt: 2000
}

describe('pairing invites', () => {
  it('round-trips the QR and copied payload, and still accepts legacy text with or without an origin', () => {
    expect(parsePairingInvite(encodePairingInvite(invite))).toEqual(invite)
    expect(parsePairingInvite('Daemon URL: https://host.example\r\nPairing id: id-1\r\nPairing code: ONE-TIME'))
      .toEqual({ version: 0, origin: invite.origin, pairingId: invite.pairingId, code: invite.code })
    expect(parsePairingInvite('Pairing id: id-1\nPairing code: ONE-TIME'))
      .toMatchObject({ version: 0, pairingId: 'id-1', code: 'ONE-TIME' })
  })

  it('rejects non-shareable addresses and malformed or unknown versions', () => {
    for (const origin of ['https://localhost.', 'https://127.0.0.2', 'https://[::1]', 'http://host.example', 'https://host.example/path']) {
      expect(() => normalizeShareableOrigin(origin)).toThrow()
    }
    const payload = encodePairingInvite(invite)
    expect(parsePairingInvite(payload.replace('"version":1', '"version":2'))).toBeNull()
    expect(parsePairingInvite(payload.replace('"inference"', '"owner"'))).toBeNull()
    expect(parsePairingInvite(payload.replace('https://host.example', 'https://localhost'))).toBeNull()
  })
})

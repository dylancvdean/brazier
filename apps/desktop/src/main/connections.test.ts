import {
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync
} from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  ConnectionProfileManager,
  ConnectionProfileStore,
  claimPairingCredential,
  fetchDaemonInfo,
  normalizeDaemonBaseUrl,
  normalizePairingDaemonBaseUrl
} from './connections'

const temporaryDirectories: string[] = []

function temporarySettingsPath(): string {
  const directory = mkdtempSync(join(tmpdir(), 'brazier-connections-'))
  temporaryDirectories.push(directory)
  return join(directory, 'connection-profiles.json')
}

function daemonInfo(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    product: 'brazier',
    version: '0.2.13-beta.64',
    management_api: { major: 1, minor: 0 },
    openai_api: { chat_completions: '/v1/chat/completions', responses: '/v1/responses' },
    ...overrides
  }
}

function okFetch(
  calls?: Array<{ url: string; authorization: string | null }>
): typeof fetch {
  return vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
    const headers = new Headers(init?.headers)
    calls?.push({ url: String(input), authorization: headers.get('authorization') })
    return new Response(JSON.stringify(daemonInfo()), {
      status: 200,
      headers: { 'content-type': 'application/json' }
    })
  }) as typeof fetch
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true })
  }
})

describe('connection profile URLs', () => {
  it('normalizes an HTTP(S) origin and rejects ambiguous or credential-bearing URLs', () => {
    expect(normalizeDaemonBaseUrl(' HTTPS://Example.COM:443/ ')).toBe('https://example.com')
    expect(normalizeDaemonBaseUrl('http://[::1]:7614')).toBe('http://[::1]:7614')
    expect(() => normalizeDaemonBaseUrl('ws://example.com')).toThrow('http:// or https://')
    expect(() => normalizeDaemonBaseUrl('https://key@example.com')).toThrow('API key field')
    expect(() => normalizeDaemonBaseUrl('https://example.com/brazier')).toThrow('without a path')
    expect(() => normalizeDaemonBaseUrl('https://example.com?daemon=1')).toThrow('without a path')
  })

  it('requires HTTPS for public pairing hosts but permits explicit private-network HTTP', () => {
    expect(normalizePairingDaemonBaseUrl('https://gpu.example')).toBe('https://gpu.example')
    expect(normalizePairingDaemonBaseUrl('http://192.168.1.12:7614')).toBe(
      'http://192.168.1.12:7614'
    )
    expect(normalizePairingDaemonBaseUrl('http://100.100.12.34:7614')).toBe(
      'http://100.100.12.34:7614'
    )
    expect(() => normalizePairingDaemonBaseUrl('http://labbox.local:7614')).toThrow(
      'requires HTTPS'
    )
    expect(() => normalizePairingDaemonBaseUrl('http://gpu.example:7614')).toThrow(
      'requires HTTPS'
    )
  })
})

describe('one-time remote pairing', () => {
  const client = {
    id: 'client-1',
    name: 'Desktop',
    scopes: ['inference', 'management', 'agent'],
    created_at: '2026-08-10T00:00:00Z',
    last_used_at: null,
    revoked_at: null
  }

  it('claims without an owner bearer and validates the one-time response', async () => {
    const fetch_ = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      expect(String(input)).toBe(
        'https://gpu.example/api/v1/auth/pairings/pairing-1/claim'
      )
      const headers = new Headers(init?.headers)
      expect(headers.get('authorization')).toBeNull()
      expect(JSON.parse(String(init?.body))).toEqual({ code: 'SECRET-CODE' })
      expect(init?.redirect).toBe('manual')
      return new Response(JSON.stringify({ client, api_key: 'issued-client-key' }))
    }) as typeof fetch

    await expect(
      claimPairingCredential('https://gpu.example', 'pairing-1', 'SECRET-CODE', { fetch: fetch_ })
    ).resolves.toEqual({ client, apiKey: 'issued-client-key' })
  })

  it('persists the issued key before handshaking and never returns it to the renderer result', async () => {
    const calls: Array<{ url: string; authorization: string | null }> = []
    const fetch_ = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input)
      const authorization = new Headers(init?.headers).get('authorization')
      calls.push({ url, authorization })
      if (url.endsWith('/claim')) {
        return new Response(JSON.stringify({ client, api_key: 'issued-client-key' }))
      }
      return new Response(JSON.stringify(daemonInfo()))
    }) as typeof fetch
    const store = new ConnectionProfileStore(temporarySettingsPath(), () => 'paired-profile')
    const manager = new ConnectionProfileManager(store, {
      startLocal: vi.fn(),
      stopLocal: vi.fn(),
      fetch: fetch_
    })

    const result = await manager.claimAndSave({
      name: 'Lab GPU',
      baseUrl: 'https://gpu.example',
      pairingId: 'pairing-1',
      code: 'SECRET-CODE'
    })

    expect(calls).toEqual([
      {
        url: 'https://gpu.example/api/v1/auth/pairings/pairing-1/claim',
        authorization: null
      },
      {
        url: 'https://gpu.example/api/v1/daemon/info',
        authorization: 'Bearer issued-client-key'
      }
    ])
    const stored = store.list().find((profile) => profile.kind === 'remote')
    expect(stored).toMatchObject({ apiKey: 'issued-client-key' })
    expect(result).toMatchObject({
      profile: { id: stored?.id, name: 'Lab GPU' },
      daemon: { product: 'brazier' },
      client: { id: 'client-1' }
    })
    expect(JSON.stringify(result)).not.toContain('issued-client-key')
    expect(JSON.stringify(result)).not.toContain('SECRET-CODE')
  })

  it('retains a consumed credential when the compatibility handshake fails', async () => {
    const fetch_ = vi.fn(async (input: string | URL | Request) => {
      if (String(input).endsWith('/claim')) {
        return new Response(JSON.stringify({ client, api_key: 'do-not-lose-me' }))
      }
      return new Response(JSON.stringify(daemonInfo({ management_api: { major: 99, minor: 0 } })))
    }) as typeof fetch
    const store = new ConnectionProfileStore(temporarySettingsPath(), () => 'paired-profile')
    const manager = new ConnectionProfileManager(store, {
      startLocal: vi.fn(),
      stopLocal: vi.fn(),
      fetch: fetch_
    })

    await expect(
      manager.claimAndSave({
        name: 'Lab GPU',
        baseUrl: 'https://gpu.example',
        pairingId: 'pairing-1',
        code: 'SECRET-CODE'
      })
    ).rejects.toThrow('Incompatible daemon management API')
    expect(store.list().find((profile) => profile.kind === 'remote')).toMatchObject({
      apiKey: 'do-not-lose-me'
    })
  })

  it('does not consume a one-time code when secure persistence is unavailable', async () => {
    const fetch_ = vi.fn() as unknown as typeof fetch
    const store = ConnectionProfileStore.localOnly(temporarySettingsPath())
    const manager = new ConnectionProfileManager(store, {
      startLocal: vi.fn(),
      stopLocal: vi.fn(),
      fetch: fetch_
    })

    await expect(
      manager.claimAndSave({
        name: 'Lab GPU',
        baseUrl: 'https://gpu.example',
        pairingId: 'pairing-1',
        code: 'SECRET-CODE'
      })
    ).rejects.toThrow('Unlock secure credential storage')
    expect(fetch_).not.toHaveBeenCalled()
  })
})

describe('connection profile persistence', () => {
  const credentialCodec = {
    encrypt: (plaintext: string): string => Buffer.from(`protected:${plaintext}`).toString('base64'),
    decrypt: (ciphertext: string): string => {
      const decoded = Buffer.from(ciphertext, 'base64').toString()
      if (!decoded.startsWith('protected:')) throw new Error('invalid ciphertext')
      return decoded.slice('protected:'.length)
    }
  }

  it('always reserves Local and repairs malformed or duplicate stored profiles', () => {
    const path = temporarySettingsPath()
    writeFileSync(
      path,
      JSON.stringify({
        version: 999,
        activeId: 'missing',
        profiles: [
          { id: 'gpu', name: ' GPU ', kind: 'remote', baseUrl: 'https://gpu.example/', apiKey: ' key ' },
          { id: 'gpu', name: 'Duplicate', kind: 'remote', baseUrl: 'https://other.example' },
          { id: 'local', name: 'Fake Local', kind: 'remote', baseUrl: 'https://fake.example' },
          { id: 'bad', name: '', kind: 'remote', baseUrl: 'file:///tmp/socket' }
        ]
      })
    )

    const store = new ConnectionProfileStore(path)
    expect(store.current()).toMatchObject({ id: 'local', kind: 'local', name: 'Local' })
    expect(store.list()).toEqual([
      { id: 'local', name: 'Local', kind: 'local', baseUrl: null, apiKey: null },
      { id: 'gpu', name: 'GPU', kind: 'remote', baseUrl: 'https://gpu.example', apiKey: 'key' }
    ])
  })

  it('atomically persists owner-only remote profiles and the active id', () => {
    const path = temporarySettingsPath()
    const store = new ConnectionProfileStore(path, () => 'generated-id')
    const profile = store.upsert({
      name: 'Lab GPU',
      baseUrl: 'http://192.168.1.50:7614/',
      apiKey: 'secret'
    })
    store.select(profile.id)

    expect(statSync(path).mode & 0o777).toBe(0o600)
    expect(readdirSync(join(path, '..')).filter((name) => name.endsWith('.tmp'))).toEqual([])
    expect(JSON.parse(readFileSync(path, 'utf8'))).toMatchObject({
      version: 1,
      activeId: 'generated-id',
      profiles: [
        {
          id: 'generated-id',
          name: 'Lab GPU',
          kind: 'remote',
          baseUrl: 'http://192.168.1.50:7614',
          apiKey: 'secret'
        }
      ]
    })
    expect(new ConnectionProfileStore(path).current()).toMatchObject({
      id: 'generated-id',
      kind: 'remote'
    })
  })

  it('does not permit deleting or replacing the reserved Local profile', () => {
    const store = new ConnectionProfileStore(temporarySettingsPath())
    expect(() => store.delete('local')).toThrow('cannot be deleted')
    expect(() =>
      store.upsert({ id: 'local', name: 'Remote Local', baseUrl: 'https://example.com' })
    ).toThrow('invalid or reserved')
  })

  it('migrates plaintext credentials to injected platform encryption', () => {
    const path = temporarySettingsPath()
    writeFileSync(
      path,
      JSON.stringify({
        version: 1,
        activeId: 'gpu',
        profiles: [
          {
            id: 'gpu',
            name: 'GPU',
            kind: 'remote',
            baseUrl: 'https://gpu.example',
            apiKey: 'legacy-plaintext-key'
          }
        ]
      })
    )

    const store = new ConnectionProfileStore(path, () => 'unused', credentialCodec)
    expect(store.current()).toMatchObject({ id: 'gpu', apiKey: 'legacy-plaintext-key' })
    const persisted = readFileSync(path, 'utf8')
    expect(persisted).not.toContain('legacy-plaintext-key')
    expect(JSON.parse(persisted).profiles[0]).toHaveProperty('encryptedApiKey')
    expect(new ConnectionProfileStore(path, () => 'unused', credentialCodec).current()).toMatchObject({
      id: 'gpu',
      apiKey: 'legacy-plaintext-key'
    })
  })

  it('fails closed when an encrypted credential cannot be decrypted', () => {
    const path = temporarySettingsPath()
    writeFileSync(
      path,
      JSON.stringify({
        version: 1,
        activeId: 'gpu',
        profiles: [
          {
            id: 'gpu',
            name: 'GPU',
            kind: 'remote',
            baseUrl: 'https://gpu.example',
            encryptedApiKey: 'not-valid-for-this-keychain'
          }
        ]
      })
    )
    expect(
      () => new ConnectionProfileStore(path, () => 'unused', credentialCodec)
    ).toThrow('could not be decrypted')
  })

  it('can open Local recovery mode without reading or overwriting a corrupt primary store', () => {
    const primary = temporarySettingsPath()
    const recovery = join(primary, '..', 'connection-profiles-recovery.json')
    writeFileSync(primary, '{broken credentials')

    const store = ConnectionProfileStore.localOnly(recovery, () => 'unused', credentialCodec)

    expect(store.current()).toMatchObject({ id: 'local', kind: 'local' })
    expect(readFileSync(primary, 'utf8')).toBe('{broken credentials')
    expect(store.canPersistRemoteCredentials()).toBe(false)
    expect(() =>
      store.upsert({ name: 'Remote', baseUrl: 'https://remote.example', apiKey: 'key' })
    ).toThrow('secure storage')
  })
})

describe('daemon compatibility handshake', () => {
  it('sends the bearer token to the bounded management-info endpoint', async () => {
    const calls: Array<{ url: string; authorization: string | null }> = []
    const result = await fetchDaemonInfo('https://gpu.example/', 'test-key', {
      fetch: okFetch(calls),
      timeoutMs: 100
    })

    expect(result.product).toBe('brazier')
    expect(calls).toEqual([
      {
        url: 'https://gpu.example/api/v1/daemon/info',
        authorization: 'Bearer test-key'
      }
    ])
  })

  it('does not follow a redirect with a bearer or one-time pairing code', async () => {
    const redirected = vi.fn(async (_input: string | URL | Request, init?: RequestInit) => {
      expect(init?.redirect).toBe('manual')
      return new Response(null, { status: 307, headers: { location: 'http://attacker.example' } })
    }) as typeof fetch

    await expect(
      fetchDaemonInfo('https://gpu.example', 'secret', { fetch: redirected })
    ).rejects.toThrow('status 307')
    await expect(
      claimPairingCredential('https://gpu.example', 'pairing-1', 'SECRET-CODE', {
        fetch: redirected
      })
    ).rejects.toThrow('status 307')
  })

  it('rejects authentication failures, the wrong product, and an incompatible API major', async () => {
    const unauthorized = vi.fn(async () => new Response('{}', { status: 401 })) as typeof fetch
    const wrongProduct = vi.fn(async () =>
      new Response(JSON.stringify(daemonInfo({ product: 'something-else' })))) as typeof fetch
    const wrongApi = vi.fn(async () =>
      new Response(JSON.stringify(daemonInfo({ management_api: { major: 2, minor: 0 } })))) as typeof fetch

    await expect(fetchDaemonInfo('https://gpu.example', 'bad', { fetch: unauthorized })).rejects.toThrow(
      'rejected this API key'
    )
    await expect(fetchDaemonInfo('https://gpu.example', null, { fetch: wrongProduct })).rejects.toThrow(
      'not a Brazier daemon'
    )
    await expect(fetchDaemonInfo('https://gpu.example', null, { fetch: wrongApi })).rejects.toThrow(
      'Incompatible daemon management API'
    )
  })

  it('aborts a handshake that exceeds its deadline', async () => {
    const neverFetch = vi.fn((_input: string | URL | Request, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true })
      })) as typeof fetch

    await expect(
      fetchDaemonInfo('https://gpu.example', null, { fetch: neverFetch, timeoutMs: 5 })
    ).rejects.toThrow('handshake timed out')
  })
})

describe('connection profile lifecycle and switching', () => {
  it('keeps stored credentials out of profile views and preserves an omitted key on edit', async () => {
    const calls: Array<{ url: string; authorization: string | null }> = []
    const store = new ConnectionProfileStore(temporarySettingsPath())
    const remote = store.upsert({
      name: 'Lab',
      baseUrl: 'https://lab.example',
      apiKey: 'renderer-must-not-see-this'
    })
    const manager = new ConnectionProfileManager(store, {
      startLocal: vi.fn(),
      stopLocal: vi.fn(),
      fetch: okFetch(calls)
    })

    const view = manager.list().find((profile) => profile.id === remote.id)
    expect(view).toMatchObject({ kind: 'remote', hasApiKey: true })
    expect(view).not.toHaveProperty('apiKey')

    await manager.test({ id: remote.id, name: 'Lab', baseUrl: 'https://lab.example' })
    expect(calls.at(-1)?.authorization).toBe('Bearer renderer-must-not-see-this')

    await manager.upsert({ id: remote.id, name: 'Renamed', baseUrl: 'https://lab.example' })
    expect(store.get(remote.id)).toMatchObject({
      name: 'Renamed',
      apiKey: 'renderer-must-not-see-this'
    })
  })

  it('tests an unsaved remote candidate without changing the durable profile list', async () => {
    const calls: Array<{ url: string; authorization: string | null }> = []
    const store = new ConnectionProfileStore(temporarySettingsPath())
    const manager = new ConnectionProfileManager(store, {
      startLocal: vi.fn(),
      stopLocal: vi.fn(),
      fetch: okFetch(calls)
    })

    const ready = await manager.test({
      name: 'Candidate',
      baseUrl: 'https://candidate.example/',
      apiKey: 'candidate-key'
    })
    expect(ready.profile).toMatchObject({ name: 'Candidate', kind: 'remote' })
    expect(calls[0]).toEqual({
      url: 'https://candidate.example/api/v1/daemon/info',
      authorization: 'Bearer candidate-key'
    })
    expect(store.list()).toHaveLength(1)
    expect(store.current().id).toBe('local')
  })

  it('never starts or stops Local while a remote-only profile is selected', async () => {
    const store = new ConnectionProfileStore(temporarySettingsPath())
    const remote = store.upsert({ name: 'Lab', baseUrl: 'https://lab.example', apiKey: 'key' })
    store.select(remote.id)
    const startLocal = vi.fn(async () => ({ address: 'http://127.0.0.1:7614', api_key: null }))
    const stopLocal = vi.fn()
    const manager = new ConnectionProfileManager(store, {
      startLocal,
      stopLocal,
      fetch: okFetch()
    })

    const ready = await manager.connection()
    expect(ready.profile).toMatchObject({ id: remote.id, kind: 'remote', hostLabel: 'lab.example' })
    expect(startLocal).not.toHaveBeenCalled()
    manager.shutdown()
    expect(stopLocal).not.toHaveBeenCalled()
  })

  it('starts Local lazily, retains ownership across switches, and stops only its owned child', async () => {
    const store = new ConnectionProfileStore(temporarySettingsPath())
    const remote = store.upsert({ name: 'Lab', baseUrl: 'https://lab.example' })
    store.select(remote.id)
    const startLocal = vi.fn(async () => ({ address: 'http://127.0.0.1:7614', api_key: 'local-key' }))
    const stopLocal = vi.fn()
    const manager = new ConnectionProfileManager(store, {
      startLocal,
      stopLocal,
      fetch: okFetch()
    })

    expect(startLocal).not.toHaveBeenCalled()
    await manager.select('local')
    expect(startLocal).toHaveBeenCalledTimes(1)
    await manager.select(remote.id)
    expect(startLocal).toHaveBeenCalledTimes(1)
    expect(stopLocal).not.toHaveBeenCalled()
    manager.shutdown()
    expect(stopLocal).toHaveBeenCalledTimes(1)
  })

  it('invalidates the resolved connection when the active remote profile changes', async () => {
    const calls: Array<{ url: string; authorization: string | null }> = []
    const store = new ConnectionProfileStore(temporarySettingsPath())
    const remote = store.upsert({ name: 'Lab', baseUrl: 'https://one.example', apiKey: 'one' })
    store.select(remote.id)
    const manager = new ConnectionProfileManager(store, {
      startLocal: vi.fn(),
      stopLocal: vi.fn(),
      fetch: okFetch(calls)
    })

    expect((await manager.connection()).address).toBe('https://one.example')
    await manager.upsert({
      id: remote.id,
      name: 'Lab',
      baseUrl: 'https://two.example',
      apiKey: 'two'
    })
    expect((await manager.connection()).address).toBe('https://two.example')
    expect(calls.map((call) => call.url)).toEqual([
      'https://one.example/api/v1/daemon/info',
      'https://two.example/api/v1/daemon/info'
    ])
  })

  it('does not persist an unusable edit to the active profile', async () => {
    const store = new ConnectionProfileStore(temporarySettingsPath())
    const remote = store.upsert({ name: 'Lab', baseUrl: 'https://working.example' })
    store.select(remote.id)
    const fetch_ = vi.fn(async (input: string | URL | Request) => {
      const status = String(input).startsWith('https://broken.example') ? 401 : 200
      return new Response(JSON.stringify(daemonInfo()), { status })
    }) as typeof fetch
    const manager = new ConnectionProfileManager(store, {
      startLocal: vi.fn(),
      stopLocal: vi.fn(),
      fetch: fetch_
    })

    await manager.connection()
    await expect(
      manager.upsert({ id: remote.id, name: 'Lab', baseUrl: 'https://broken.example' })
    ).rejects.toThrow('rejected this API key')
    expect(store.current()).toMatchObject({ baseUrl: 'https://working.example' })
    expect((await manager.connection()).address).toBe('https://working.example')
  })

  it('does not retain a rejected connection promise', async () => {
    const store = new ConnectionProfileStore(temporarySettingsPath())
    const remote = store.upsert({ name: 'Lab', baseUrl: 'https://lab.example' })
    store.select(remote.id)
    const fetch_ = vi
      .fn()
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce(new Response(JSON.stringify(daemonInfo()))) as typeof fetch
    const manager = new ConnectionProfileManager(store, {
      startLocal: vi.fn(),
      stopLocal: vi.fn(),
      fetch: fetch_
    })

    await expect(manager.connection()).rejects.toThrow('offline')
    await expect(manager.connection()).resolves.toMatchObject({ address: 'https://lab.example' })
    expect(fetch_).toHaveBeenCalledTimes(2)
  })

  it('allows renderer traffic only to the selected endpoint, with WebSocket scheme parity', () => {
    const store = new ConnectionProfileStore(temporarySettingsPath())
    const remote = store.upsert({ name: 'Lab', baseUrl: 'https://lab.example:7614' })
    store.select(remote.id)
    const manager = new ConnectionProfileManager(store, {
      startLocal: vi.fn(),
      stopLocal: vi.fn(),
      fetch: okFetch()
    })

    expect(manager.allowsRendererNetworkUrl('https://lab.example:7614/api/v1/health')).toBe(true)
    expect(manager.allowsRendererNetworkUrl('wss://lab.example:7614/api/v1/voice')).toBe(true)
    expect(manager.allowsRendererNetworkUrl('http://lab.example:7614/api/v1/health')).toBe(false)
    expect(manager.allowsRendererNetworkUrl('https://other.example/api/v1/health')).toBe(false)
    expect(
      manager.allowsRendererNetworkUrl('http://127.0.0.1:5173/src/main.tsx', 'http://127.0.0.1:5173')
    ).toBe(true)

    store.select('local')
    expect(manager.allowsRendererNetworkUrl('ws://127.0.0.1:9000/api/chat')).toBe(false)
    expect(manager.allowsRendererNetworkUrl('http://localhost:7614/api/v1/health')).toBe(false)
    expect(manager.allowsRendererNetworkUrl('https://lab.example:7614/api/v1/health')).toBe(false)
  })

  it('does not widen Local renderer access to unrelated loopback services', async () => {
    const store = new ConnectionProfileStore(temporarySettingsPath())
    const manager = new ConnectionProfileManager(store, {
      startLocal: vi.fn(async () => ({ address: 'http://127.0.0.1:7614', api_key: null })),
      stopLocal: vi.fn(),
      fetch: okFetch()
    })

    expect(manager.allowsRendererNetworkUrl('http://127.0.0.1:7614/api/v1/health')).toBe(false)
    await manager.connection()
    expect(manager.allowsRendererNetworkUrl('http://127.0.0.1:7614/api/v1/health')).toBe(true)
    expect(manager.allowsRendererNetworkUrl('http://127.0.0.1:9000/private')).toBe(false)
    expect(manager.allowsRendererNetworkUrl('http://localhost:7614/private')).toBe(false)
    await expect(manager.rendererApiKeyForUrl('http://127.0.0.1:7614/private')).resolves.toBeNull()
  })

  it('keeps a remote bearer in main while making it available to the exact request interceptor', async () => {
    const store = new ConnectionProfileStore(temporarySettingsPath())
    const remote = store.upsert({
      name: 'Lab',
      baseUrl: 'https://lab.example:7614',
      apiKey: 'main-only-secret'
    })
    store.select(remote.id)
    const manager = new ConnectionProfileManager(store, {
      startLocal: vi.fn(),
      stopLocal: vi.fn(),
      fetch: okFetch()
    })

    expect(manager.list().find((profile) => profile.id === remote.id)).not.toHaveProperty('apiKey')
    await expect(
      manager.rendererApiKeyForUrl('wss://lab.example:7614/api/v1/voice')
    ).resolves.toBe('main-only-secret')
    await expect(
      manager.rendererApiKeyForUrl('https://other.example/api/v1/health')
    ).resolves.toBeNull()
  })

  it('allows the exact wildcard bind address reported by an owned Local daemon', async () => {
    const store = new ConnectionProfileStore(temporarySettingsPath())
    const manager = new ConnectionProfileManager(store, {
      startLocal: vi.fn(async () => ({ address: 'http://0.0.0.0:7614', api_key: null })),
      stopLocal: vi.fn(),
      fetch: okFetch()
    })

    expect(manager.allowsRendererNetworkUrl('http://0.0.0.0:7614/api/v1/health')).toBe(false)
    await manager.connection()
    expect(manager.allowsRendererNetworkUrl('http://0.0.0.0:7614/api/v1/health')).toBe(true)
  })
})

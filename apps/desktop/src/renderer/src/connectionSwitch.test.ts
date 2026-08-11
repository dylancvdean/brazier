import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

beforeEach(async () => {
  const { setDaemonAvailability } = await import('./daemonAvailability')
  setDaemonAvailability('healthy')
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.resetModules()
})

describe('renderer connection switch boundary', () => {
  it('routes pairing and client management to the active daemon without exposing its bearer', async () => {
    const requests: Array<{ url: string; method: string; authorization: string | null; body: unknown }> = []
    vi.stubGlobal('window', {
      brazier: {
        getConnection: vi.fn(async () => ({
          address: 'https://owner.example',
          profile: {
            id: 'owner', name: 'Owner', kind: 'remote' as const,
            baseUrl: 'https://owner.example', hostLabel: 'owner.example'
          },
          daemon: {
            product: 'brazier' as const,
            version: '1.0.0', management_api: { major: 1, minor: 0 }
          }
        })),
        onConnectionProfileChanged: () => () => undefined
      }
    })
    vi.stubGlobal('fetch', vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input)
      requests.push({
        url,
        method: init?.method ?? 'GET',
        authorization: new Headers(init?.headers).get('authorization'),
        body: init?.body ? JSON.parse(String(init.body)) : null
      })
      if (url.endsWith('/auth/pairings') && init?.method === 'POST') {
        return new Response(JSON.stringify({
          pairing: {
            id: 'pair-1', client_name: 'Laptop', scopes: ['inference'],
            expires_at: 2_000, attempts: 0, max_attempts: 5, created_at: 'date'
          },
          code: 'one-time-code'
        }))
      }
      if (url.endsWith('/auth/pairings')) return new Response(JSON.stringify({ data: [] }))
      if (url.endsWith('/auth/clients')) return new Response(JSON.stringify({ data: [] }))
      return new Response(null, { status: 204 })
    }))

    const api = await import('./api')
    const created = await api.createDaemonPairing({
      clientName: 'Laptop',
      scopes: ['inference'],
      ttlSeconds: 300
    })
    await api.listDaemonPairings()
    await api.cancelDaemonPairing('pair/1')
    await api.listDaemonApiClients()
    await api.revokeDaemonApiClient('client/1')

    expect(created.code).toBe('one-time-code')
    expect(api.isPendingDaemonPairing(created.pairing, 1_999)).toBe(true)
    expect(api.isPendingDaemonPairing(created.pairing, 2_000)).toBe(false)
    expect(requests).toEqual([
      {
        url: 'https://owner.example/api/v1/auth/pairings',
        method: 'POST',
        authorization: null,
        body: { client_name: 'Laptop', scopes: ['inference'], ttl_seconds: 300 }
      },
      {
        url: 'https://owner.example/api/v1/auth/pairings',
        method: 'GET', authorization: null, body: null
      },
      {
        url: 'https://owner.example/api/v1/auth/pairings/pair%2F1',
        method: 'DELETE', authorization: null, body: null
      },
      {
        url: 'https://owner.example/api/v1/auth/clients',
        method: 'GET', authorization: null, body: null
      },
      {
        url: 'https://owner.example/api/v1/auth/clients/client%2F1',
        method: 'DELETE', authorization: null, body: null
      }
    ])
  })

  it('resolves daemon-relative voice sockets against the selected remote profile', async () => {
    vi.stubGlobal('window', {
      brazier: {
        getConnection: vi.fn(async () => ({
          address: 'https://voice.example:7443',
          profile: {
            id: 'voice', name: 'Voice host', kind: 'remote' as const,
            baseUrl: 'https://voice.example:7443', hostLabel: 'voice.example:7443'
          },
          daemon: {
            product: 'brazier' as const,
            version: '1.0.0', management_api: { major: 1, minor: 0 }
          }
        })),
        onConnectionProfileChanged: () => () => undefined
      }
    })
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({
      id: 'session-1',
      ws_url: '/api/v1/voice/sessions/session-1/stream',
      ws_protocol: 'brazier.voice.secret',
      persona_text: 'Helpful'
    }))))

    const api = await import('./api')
    const session = await api.createVoiceSession()

    expect(session.ws_url).toBe(
      'wss://voice.example:7443/api/v1/voice/sessions/session-1/stream'
    )
    expect(session.ws_protocol).toBe('brazier.voice.secret')
  })

  it('invalidates both the general and agent API connection promises', async () => {
    const listeners: Array<(profile: unknown) => void> = []
    let selected = {
      address: 'https://one.example',
      profile: {
        id: 'one', name: 'One', kind: 'remote' as const,
        baseUrl: 'https://one.example', hostLabel: 'one.example'
      },
      daemon: {
        product: 'brazier' as const,
        version: '1.0.0', management_api: { major: 1, minor: 0 }
      }
    }
    const getConnection = vi.fn(async () => selected)
    vi.stubGlobal('window', {
      brazier: {
        getConnection,
        onConnectionProfileChanged: (listener: (profile: unknown) => void) => {
          listeners.push(listener)
          return () => undefined
        }
      }
    })
    const requests: string[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: string | URL | Request) => {
      requests.push(String(input))
      const body = String(input).endsWith('/api/v1/preferences/welcome')
        ? { completed: true }
        : {
            schema_version: 1,
            sandbox: {},
            permission_modes: [],
            runtimes: [],
            tool_output_limit_chars: 1_000
          }
      return new Response(JSON.stringify(body))
    }))

    const api = await import('./api')
    const agentApi = await import('./agentApi')
    await api.fetchWelcomePreference()
    await agentApi.fetchAgentCapabilities()

    selected = {
      ...selected,
      address: 'https://two.example',
      profile: {
        id: 'two', name: 'Two', kind: 'remote',
        baseUrl: 'https://two.example', hostLabel: 'two.example'
      }
    }
    for (const listener of listeners) listener(selected.profile)
    await api.fetchWelcomePreference()
    await agentApi.fetchAgentCapabilities()

    expect(listeners).toHaveLength(2)
    expect(getConnection).toHaveBeenCalledTimes(4)
    expect(requests).toEqual([
      'https://one.example/api/v1/preferences/welcome',
      'https://one.example/api/v1/agent/capabilities',
      'https://two.example/api/v1/preferences/welcome',
      'https://two.example/api/v1/agent/capabilities'
    ])
  })
})

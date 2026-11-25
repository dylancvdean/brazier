import { randomBytes, randomUUID } from 'node:crypto'
import {
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync
} from 'node:fs'
import { dirname } from 'node:path'
import { isIP } from 'node:net'

import { normalizeShareableOrigin } from '../../../../packages/pairing/src/invite'
import { isRendererDevelopmentOrigin } from './rendererTrust'

export const LOCAL_CONNECTION_PROFILE_ID = 'local'
export const MANAGEMENT_API_MAJOR = 1
export const DEFAULT_HANDSHAKE_TIMEOUT_MS = 5_000

export type ClientScope = 'inference' | 'management' | 'agent'

export type LocalConnectionProfile = {
  id: typeof LOCAL_CONNECTION_PROFILE_ID
  name: 'Local'
  kind: 'local'
  baseUrl: null
  apiKey: null
}

export type RemoteConnectionProfile = {
  id: string
  name: string
  kind: 'remote'
  baseUrl: string
  apiKey: string | null
  daemonInstanceId?: string
}

export type ConnectionProfile = LocalConnectionProfile | RemoteConnectionProfile

/** Secret-free profile metadata that is safe to expose to the renderer. */
export type ConnectionProfileView =
  | {
      id: typeof LOCAL_CONNECTION_PROFILE_ID
      name: 'Local'
      kind: 'local'
      baseUrl: null
      hostLabel: string
      hasApiKey: false
    }
  | {
      id: string
      name: string
      kind: 'remote'
      baseUrl: string
      hostLabel: string
      hasApiKey: boolean
    }

export type RemoteConnectionProfileInput = {
  id?: string
  name: string
  kind?: 'remote'
  baseUrl: string
  apiKey?: string | null
}

export type PairingClaimInput = {
  id?: string
  name: string
  baseUrl: string
  pairingId: string
  code: string
  expectedDaemonInstanceId?: string
}

export type PairingPreview = {
  pairing: {
    id: string
    client_name: string
    scopes: ClientScope[]
    expires_at: number
  }
  daemon: { instance_id: string; display_name: string }
}

export type ConnectionVerification =
  | ({ status: 'healthy' } & Omit<DaemonConnection, 'api_key'>)
  | { status: 'offline' | 'unauthorized' | 'forbidden' | 'identity-changed'; message: string }

export type PairedClient = {
  id: string
  name: string
  scopes: ClientScope[]
  created_at: string
  last_used_at?: string | null
  revoked_at?: string | null
}

export type ClaimedConnection = {
  profile: ConnectionProfileSummary
  daemon: DaemonInfo
  client: PairedClient
}

export type ConnectionProfileSummary = {
  id: string
  name: string
  kind: 'local' | 'remote'
  baseUrl: string | null
  hostLabel: string
}

export type DaemonInfo = {
  product: 'brazier'
  version: string
  management_api: {
    major: number
    minor: number
  }
  openai_api?: {
    chat_completions?: string
    responses?: string
  }
  daemon?: {
    instance_id: string
    display_name: string
    platform: string
    architecture: string
  }
  client?: {
    id: string
    name: string
    scopes: ClientScope[]
    owner: boolean
  }
}

export type DaemonConnection = {
  address: string
  api_key: string | null
  profile: ConnectionProfileSummary
  daemon: DaemonInfo
}

export type ConnectionTestResult = Pick<DaemonConnection, 'profile' | 'daemon'>

type StoredConnectionProfiles = {
  version: 1
  activeId: string
  profiles: RemoteConnectionProfile[]
}

export type ConnectionProfileManagerDependencies = {
  startLocal: () => Promise<{
    address: string
    api_key: string | null
    local_control_key?: string | null
  }>
  stopLocal: () => void | Promise<void>
  ensureLocalDaemon?: () => Promise<DaemonConnection>
  fetch?: typeof fetch
  handshakeTimeoutMs?: number
}

export type ConnectionCredentialCodec = {
  encrypt: (plaintext: string) => string
  decrypt: (ciphertext: string) => string
}

const LOCAL_PROFILE: LocalConnectionProfile = Object.freeze({
  id: LOCAL_CONNECTION_PROFILE_ID,
  name: 'Local',
  kind: 'local',
  baseUrl: null,
  apiKey: null
})

function cleanName(value: unknown): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error('Connection profile name is required.')
  }
  const name = value.trim()
  if (name.length > 80) throw new Error('Connection profile name must be 80 characters or fewer.')
  return name
}

function validStoredId(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    value !== LOCAL_CONNECTION_PROFILE_ID &&
    /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$/.test(value)
  )
}

/**
 * Normalize a daemon address to an HTTP(S) origin.
 *
 * Renderer API clients append absolute API paths, so accepting a path prefix
 * here would silently discard it and point requests at the wrong service.
 * Credentials also belong in the explicit bearer-token field, never the URL.
 */
export function normalizeDaemonBaseUrl(value: string): string {
  const input = value.trim()
  if (!input) throw new Error('Daemon URL is required.')
  let url: URL
  try {
    url = new URL(input)
  } catch {
    throw new Error('Daemon URL must be a valid http:// or https:// URL.')
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('Daemon URL must use http:// or https://.')
  }
  if (!url.hostname) throw new Error('Daemon URL must include a host.')
  if (url.username || url.password) {
    throw new Error('Put the daemon API key in the API key field, not in the URL.')
  }
  if ((url.pathname && url.pathname !== '/') || url.search || url.hash) {
    throw new Error('Daemon URL must be an origin without a path, query, or fragment.')
  }
  return url.origin
}

function privateIpv4(hostname: string): boolean {
  const parts = hostname.split('.').map(Number)
  if (parts.length !== 4 || parts.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) {
    return false
  }
  return (
    parts[0] === 10 ||
    parts[0] === 127 ||
    (parts[0] === 100 && parts[1] >= 64 && parts[1] <= 127) ||
    (parts[0] === 169 && parts[1] === 254) ||
    (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31) ||
    (parts[0] === 192 && parts[1] === 168)
  )
}

function privateIpv6(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, '')
  return host === '::1' || host.startsWith('fc') || host.startsWith('fd') || /^fe[89ab]/.test(host)
}

function privateNetworkHostname(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, '')
  if (host === 'localhost' || host.endsWith('.localhost')) return true
  if (isIP(host) === 4) return privateIpv4(host)
  if (isIP(host) === 6) return privateIpv6(host)
  // DNS names can be rebound or later resolve publicly. Plaintext credentials
  // are therefore limited to address literals and the special localhost name.
  return false
}

/** Pairing over plaintext is limited to hosts that are explicitly private. */
export function normalizePairingDaemonBaseUrl(value: string): string {
  const address = normalizeDaemonBaseUrl(value)
  const url = new URL(address)
  if (url.protocol === 'http:' && !privateNetworkHostname(url.hostname)) {
    throw new Error('Pairing requires HTTPS unless the daemon URL names a private-network host.')
  }
  return address
}

function normalizeApiKey(value: unknown): string | null {
  if (typeof value !== 'string') return null
  return value.trim() || null
}

function normalizeRemoteProfile(
  input: RemoteConnectionProfileInput & { daemonInstanceId?: string },
  makeId: () => string
): RemoteConnectionProfile {
  const id = input.id === undefined ? makeId() : input.id
  if (!validStoredId(id)) {
    throw new Error('Connection profile id is invalid or reserved.')
  }
  const apiKey = normalizeApiKey(input.apiKey)
  const baseUrl = apiKey
    ? normalizePairingDaemonBaseUrl(input.baseUrl)
    : normalizeDaemonBaseUrl(input.baseUrl)
  return {
    id,
    name: cleanName(typeof input.name === 'string' && !input.name.trim()
      ? new URL(baseUrl).host.slice(0, 80)
      : input.name),
    kind: 'remote',
    baseUrl,
    apiKey,
    ...(input.daemonInstanceId ? { daemonInstanceId: input.daemonInstanceId } : {})
  }
}

class CredentialDecryptionError extends Error {}

function parseStoredRemote(
  value: unknown,
  credentialCodec?: ConnectionCredentialCodec
): { profile: RemoteConnectionProfile; plaintextCredential: boolean } | null {
  if (typeof value !== 'object' || value === null) return null
  const candidate = value as Record<string, unknown>
  if (candidate.kind !== 'remote' || !validStoredId(candidate.id)) return null
  let apiKey: string | null = null
  let plaintextCredential = false
  if (typeof candidate.encryptedApiKey === 'string' && candidate.encryptedApiKey) {
    if (!credentialCodec) {
      throw new CredentialDecryptionError('Encrypted connection credentials are unavailable.')
    }
    try {
      apiKey = normalizeApiKey(credentialCodec.decrypt(candidate.encryptedApiKey))
    } catch {
      throw new CredentialDecryptionError('A saved connection credential could not be decrypted.')
    }
    if (!apiKey) {
      throw new CredentialDecryptionError('A saved connection credential decrypted to an empty value.')
    }
  } else {
    apiKey = normalizeApiKey(candidate.apiKey)
    plaintextCredential = apiKey !== null
  }
  try {
    return {
      profile: normalizeRemoteProfile(
        {
          id: candidate.id,
          name: typeof candidate.name === 'string' ? candidate.name : '',
          baseUrl: typeof candidate.baseUrl === 'string' ? candidate.baseUrl : '',
          apiKey,
          daemonInstanceId: typeof candidate.daemonInstanceId === 'string'
            ? candidate.daemonInstanceId
            : undefined
        },
        randomUUID
      ),
      plaintextCredential
    }
  } catch {
    return null
  }
}

function defaultStoredProfiles(): StoredConnectionProfiles {
  return { version: 1, activeId: LOCAL_CONNECTION_PROFILE_ID, profiles: [] }
}

function loadStoredProfiles(
  path: string,
  credentialCodec?: ConnectionCredentialCodec
): { stored: StoredConnectionProfiles; migrateCredentials: boolean } {
  try {
    const parsed = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>
    const profiles: RemoteConnectionProfile[] = []
    const ids = new Set<string>()
    let migrateCredentials = false
    if (Array.isArray(parsed.profiles)) {
      for (const value of parsed.profiles) {
        const loaded = parseStoredRemote(value, credentialCodec)
        if (!loaded || ids.has(loaded.profile.id)) continue
        ids.add(loaded.profile.id)
        profiles.push(loaded.profile)
        migrateCredentials ||= loaded.plaintextCredential && credentialCodec !== undefined
      }
    }
    const activeId =
      typeof parsed.activeId === 'string' &&
      (parsed.activeId === LOCAL_CONNECTION_PROFILE_ID || ids.has(parsed.activeId))
        ? parsed.activeId
        : LOCAL_CONNECTION_PROFILE_ID
    return { stored: { version: 1, activeId, profiles }, migrateCredentials }
  } catch (cause) {
    if (cause instanceof CredentialDecryptionError) throw cause
    return { stored: defaultStoredProfiles(), migrateCredentials: false }
  }
}

function writeStoredProfiles(
  path: string,
  stored: StoredConnectionProfiles,
  credentialCodec?: ConnectionCredentialCodec
): void {
  const directory = dirname(path)
  mkdirSync(directory, { recursive: true, mode: 0o700 })
  const temporary = `${path}.${randomBytes(6).toString('hex')}.tmp`
  const persisted = {
    version: stored.version,
    activeId: stored.activeId,
    profiles: stored.profiles.map((profile) => ({
      id: profile.id,
      name: profile.name,
      kind: profile.kind,
      baseUrl: profile.baseUrl,
      daemonInstanceId: profile.daemonInstanceId,
      ...(profile.apiKey
        ? credentialCodec
          ? { encryptedApiKey: credentialCodec.encrypt(profile.apiKey) }
          : { apiKey: profile.apiKey }
        : { apiKey: null })
    }))
  }
  try {
    writeFileSync(temporary, `${JSON.stringify(persisted, null, 2)}\n`, { mode: 0o600, flag: 'wx' })
    renameSync(temporary, path)
  } catch (error) {
    try { rmSync(temporary, { force: true }) } catch { /* Preserve the original write error. */ }
    throw error
  }
}

export class ConnectionProfileStore {
  private stored: StoredConnectionProfiles

  constructor(
    private readonly path: string,
    private readonly makeId: () => string = randomUUID,
    private readonly credentialCodec?: ConnectionCredentialCodec,
    startLocalOnly = false,
    private readonly remoteWritesAllowed = true
  ) {
    const loaded = startLocalOnly
      ? { stored: defaultStoredProfiles(), migrateCredentials: false }
      : loadStoredProfiles(path, credentialCodec)
    this.stored = loaded.stored
    if (loaded.migrateCredentials) this.persist()
  }

  /** Open an empty recovery store without reading a broken primary store. */
  static localOnly(
    path: string,
    makeId: () => string = randomUUID,
    credentialCodec?: ConnectionCredentialCodec
  ): ConnectionProfileStore {
    return new ConnectionProfileStore(path, makeId, credentialCodec, true, false)
  }

  canPersistRemoteCredentials(): boolean {
    return this.remoteWritesAllowed
  }

  list(): ConnectionProfile[] {
    return [{ ...LOCAL_PROFILE }, ...this.stored.profiles.map((profile) => ({ ...profile }))]
  }

  current(): ConnectionProfile {
    return this.get(this.stored.activeId) ?? { ...LOCAL_PROFILE }
  }

  get(id: string): ConnectionProfile | undefined {
    if (id === LOCAL_CONNECTION_PROFILE_ID) return { ...LOCAL_PROFILE }
    const profile = this.stored.profiles.find((entry) => entry.id === id)
    return profile ? { ...profile } : undefined
  }

  upsert(input: RemoteConnectionProfileInput & { daemonInstanceId?: string }): RemoteConnectionProfile {
    if (!this.remoteWritesAllowed) {
      throw new Error('Remote credentials cannot be saved while secure storage is unavailable.')
    }
    const profile = normalizeRemoteProfile(input, this.makeId)
    const existing = this.stored.profiles.findIndex((entry) => entry.id === profile.id)
    const profiles = [...this.stored.profiles]
    if (existing === -1) profiles.push(profile)
    else profiles[existing] = profile
    this.persist({ ...this.stored, profiles })
    return { ...profile }
  }

  delete(id: string): boolean {
    if (id === LOCAL_CONNECTION_PROFILE_ID) {
      throw new Error('The reserved Local connection profile cannot be deleted.')
    }
    const profiles = this.stored.profiles.filter((profile) => profile.id !== id)
    if (profiles.length === this.stored.profiles.length) return false
    const activeId = this.stored.activeId === id ? LOCAL_CONNECTION_PROFILE_ID : this.stored.activeId
    this.persist({ ...this.stored, profiles, activeId })
    return true
  }

  select(id: string): ConnectionProfile {
    const profile = this.get(id)
    if (!profile) throw new Error('Connection profile does not exist.')
    if (this.stored.activeId !== id) {
      this.persist({ ...this.stored, activeId: id })
    }
    return profile
  }

  private persist(next = this.stored): void {
    writeStoredProfiles(this.path, next, this.credentialCodec)
    this.stored = next
  }
}

class ConnectionVerificationError extends Error {
  constructor(readonly status: 'unauthorized' | 'forbidden' | 'identity-changed', message: string) {
    super(message)
  }
}

function errorForStatus(status: number): Error {
  if (status === 401) return new ConnectionVerificationError('unauthorized', 'The daemon rejected this API key. Pair again or replace the key.')
  if (status === 403) return new ConnectionVerificationError('forbidden', 'The daemon denied access to this client.')
  return new Error(`The daemon handshake failed with status ${status}.`)
}

const MAX_HANDSHAKE_RESPONSE_BYTES = 1024 * 1024

async function boundedJson(response: Response): Promise<Record<string, unknown> | null> {
  const declared = Number(response.headers.get('content-length'))
  if (Number.isFinite(declared) && declared > MAX_HANDSHAKE_RESPONSE_BYTES) {
    throw new Error('The daemon response is too large.')
  }
  if (!response.body) return null
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let length = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    length += value.byteLength
    if (length > MAX_HANDSHAKE_RESPONSE_BYTES) {
      await reader.cancel()
      throw new Error('The daemon response is too large.')
    }
    chunks.push(value)
  }
  const bytes = new Uint8Array(length)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  try {
    const value: unknown = JSON.parse(new TextDecoder().decode(bytes))
    return typeof value === 'object' && value !== null ? value as Record<string, unknown> : null
  } catch {
    return null
  }
}

function cleanPairingSecret(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} is required.`)
  const secret = value.trim()
  if (secret.length > 512) throw new Error(`${label} is too long.`)
  return secret
}

function pairedClient(value: unknown): PairedClient | null {
  if (typeof value !== 'object' || value === null) return null
  const candidate = value as Record<string, unknown>
  const scopes = Array.isArray(candidate.scopes)
    ? candidate.scopes.filter(
        (scope): scope is ClientScope =>
          scope === 'inference' || scope === 'management' || scope === 'agent'
      )
    : []
  if (
    typeof candidate.id !== 'string' ||
    !candidate.id ||
    typeof candidate.name !== 'string' ||
    !candidate.name ||
    typeof candidate.created_at !== 'string' ||
    scopes.length !== (candidate.scopes as unknown[])?.length
  ) {
    return null
  }
  return {
    id: candidate.id,
    name: candidate.name,
    scopes,
    created_at: candidate.created_at,
    last_used_at: typeof candidate.last_used_at === 'string' ? candidate.last_used_at : null,
    revoked_at: typeof candidate.revoked_at === 'string' ? candidate.revoked_at : null
  }
}

/** Claim a one-time pairing without sending an owner bearer to the prospective host. */
async function requestPairing(
  baseUrl: string,
  pairingId: string,
  code: string,
  action: 'preview' | 'claim',
  options: { fetch?: typeof fetch; timeoutMs?: number; expectedDaemonInstanceId?: string } = {}
): Promise<Record<string, unknown> | null> {
  const address = normalizePairingDaemonBaseUrl(baseUrl)
  const id = cleanPairingSecret(pairingId, 'Pairing id')
  const pairingCode = cleanPairingSecret(code, 'Pairing code')
  const fetch_ = options.fetch ?? fetch
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? DEFAULT_HANDSHAKE_TIMEOUT_MS)
  try {
    let response: Response
    try {
      response = await fetch_(`${address}/api/v1/auth/pairings/${encodeURIComponent(id)}/${action}`, {
        method: 'POST',
        headers: { accept: 'application/json', 'content-type': 'application/json' },
        body: JSON.stringify({ code: pairingCode, expected_daemon_instance_id: options.expectedDaemonInstanceId }),
        redirect: 'manual',
        signal: controller.signal
      })
    } catch (cause) {
      if (controller.signal.aborted) throw new Error('The pairing request timed out.')
      const transportCode = (cause as { cause?: { code?: string } })?.cause?.code
      if (transportCode && /CERT|TLS|SSL/.test(transportCode)) {
        throw new Error('Could not verify the daemon’s HTTPS certificate. Use an HTTPS address with a certificate this device trusts.')
      }
      throw new Error(
        `Could not reach the daemon for pairing: ${cause instanceof Error ? cause.message : String(cause)}`
      )
    }
    if (!response.ok) {
      if (response.status === 400) {
        throw new Error('This invite may have expired, been used or canceled, or have an incorrect code. Ask the host for a fresh invite.')
      }
      if (response.status === 404 && action === 'preview') {
        throw new Error('This daemon does not support reviewing pairing invitations. Update the host Brazier app and try again.')
      }
      if (response.status === 409) throw new Error('The daemon identity changed since review. Check the address and ask for a new invite.')
      if (response.status >= 500) throw new Error('The daemon could not process the pairing request. Check the host and try again.')
      throw new Error(`The pairing request failed with status ${response.status}.`)
    }
    return await boundedJson(response)
  } finally {
    clearTimeout(timeout)
  }
}

export async function claimPairingCredential(
  baseUrl: string,
  pairingId: string,
  code: string,
  options: { fetch?: typeof fetch; timeoutMs?: number; expectedDaemonInstanceId?: string } = {}
): Promise<{ client: PairedClient; apiKey: string }> {
  const value = await requestPairing(baseUrl, pairingId, code, 'claim', options)
  const client = pairedClient(value?.client)
  const apiKey = typeof value?.api_key === 'string' ? value.api_key.trim() : ''
  if (!client || !apiKey || apiKey.length > 4_096) {
    throw new Error('The daemon returned an invalid pairing response. The invite may have been used; ask the host to check its paired devices.')
  }
  return { client, apiKey }
}

export async function previewPairing(
  baseUrl: string,
  pairingId: string,
  code: string
): Promise<PairingPreview> {
  const value = await requestPairing(baseUrl, pairingId, code, 'preview')
  const pairing = value?.pairing as PairingPreview['pairing'] | undefined
  const daemon = value?.daemon as PairingPreview['daemon'] | undefined
  if (pairing?.id !== pairingId.trim() || typeof pairing.client_name !== 'string' ||
      !Array.isArray(pairing.scopes) || !pairing.scopes.length ||
      !pairing.scopes.every((scope) => ['inference', 'management', 'agent'].includes(scope)) ||
      !Number.isSafeInteger(pairing.expires_at) ||
      typeof daemon?.instance_id !== 'string' || !daemon.instance_id ||
      typeof daemon.display_name !== 'string') {
    throw new Error('The daemon returned an invalid pairing preview.')
  }
  return { pairing, daemon }
}

export async function verifyPairingOrigin(
  activeAddress: string,
  advertisedAddress: string,
  fetch_: typeof fetch = fetch
): Promise<void> {
  const origin = normalizeShareableOrigin(advertisedAddress)
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), DEFAULT_HANDSHAKE_TIMEOUT_MS)
  const options = {
    redirect: 'error',
    credentials: 'omit',
    cache: 'no-store',
    signal: controller.signal
  } as const
  const bootId = async (address: string): Promise<string> => {
    const response = await fetch_(`${address}/api/v1/auth/pairings/00000000-0000-0000-0000-000000000000/preview`, {
      ...options,
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ code: '' })
    })
    const useHealth = response.status === 404 || response.status === 405
    const identityResponse = useHealth ? await fetch_(`${address}/health`, options) : response
    if (!identityResponse.ok) {
      throw new Error(useHealth
        ? 'Daemon health request failed.'
        : 'The pairing preview route is not reachable. Check the daemon version and proxy configuration.')
    }
    const identity = await boundedJson(identityResponse)
    if ((useHealth && identity?.status !== 'healthy') || typeof identity?.boot_id !== 'string' || !identity.boot_id) {
      throw new Error('The address did not return a Brazier daemon identity.')
    }
    return identity.boot_id
  }
  try {
    const [active, advertised] = await Promise.all([bootId(normalizeDaemonBaseUrl(activeAddress)), bootId(origin)])
    if (active !== advertised) {
      throw new Error('The reachable address points to a different daemon. Check the address and try again.')
    }
  } finally {
    clearTimeout(timeout)
    controller.abort()
  }
}

/** Perform the bounded, authenticated compatibility handshake. */
export async function fetchDaemonInfo(
  baseUrl: string,
  apiKey: string | null,
  options: { fetch?: typeof fetch; timeoutMs?: number } = {}
): Promise<DaemonInfo> {
  const address = normalizeDaemonBaseUrl(baseUrl)
  const fetch_ = options.fetch ?? fetch
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? DEFAULT_HANDSHAKE_TIMEOUT_MS)
  try {
    const headers = new Headers({ accept: 'application/json' })
    if (apiKey) headers.set('authorization', `Bearer ${apiKey}`)
    let response: Response
    try {
      response = await fetch_(`${address}/api/v1/daemon/info`, {
        method: 'GET',
        headers,
        redirect: 'manual',
        signal: controller.signal
      })
    } catch (cause) {
      if (controller.signal.aborted) {
        throw new Error('The daemon handshake timed out.')
      }
      throw new Error(
        `Could not reach the daemon: ${cause instanceof Error ? cause.message : String(cause)}`
      )
    }
    if (!response.ok) throw errorForStatus(response.status)
    const value = await boundedJson(response)
    const management =
      value && typeof value.management_api === 'object' && value.management_api !== null
        ? (value.management_api as Record<string, unknown>)
        : null
    if (value?.product !== 'brazier') {
      throw new Error('The selected server is not a Brazier daemon.')
    }
    if (management?.major !== MANAGEMENT_API_MAJOR) {
      throw new Error(
        `Incompatible daemon management API (expected major ${MANAGEMENT_API_MAJOR}, received ${String(management?.major ?? 'missing')}).`
      )
    }
    if (
      typeof value.version !== 'string' ||
      !value.version ||
      typeof management.minor !== 'number' ||
      !Number.isInteger(management.minor)
    ) {
      throw new Error('The daemon returned an invalid compatibility response.')
    }
    return value as DaemonInfo
  } finally {
    clearTimeout(timeout)
  }
}

export function summarizeConnectionProfile(profile: ConnectionProfile): ConnectionProfileSummary {
  return {
    id: profile.id,
    name: profile.name,
    kind: profile.kind,
    baseUrl: profile.baseUrl,
    hostLabel: profile.kind === 'local' ? 'This computer' : new URL(profile.baseUrl).host
  }
}

export function viewConnectionProfile(profile: ConnectionProfile): ConnectionProfileView {
  const summary = summarizeConnectionProfile(profile)
  return profile.kind === 'local'
    ? {
        id: LOCAL_CONNECTION_PROFILE_ID,
        name: 'Local',
        kind: 'local',
        baseUrl: null,
        hostLabel: summary.hostLabel,
        hasApiKey: false
      }
    : {
        ...summary,
        kind: 'remote',
        baseUrl: profile.baseUrl,
        hasApiKey: profile.apiKey !== null
      }
}

/**
 * Owns connection resolution while keeping the local child process separate
 * from remote profiles. A remote selection can only perform an HTTP handshake;
 * it never invokes either local lifecycle dependency.
 */
export class ConnectionProfileManager {
  private activeConnection?: Promise<DaemonConnection>
  private localConnection?: Promise<{
    address: string
    api_key: string | null
    local_control_key?: string | null
  }>
  private resolvedLocalAddress?: string
  private resolvedLocalControlKey?: string
  private localStarted = false
  private stopped = false
  private shutdownPromise?: Promise<void>
  private switchGeneration = 0

  constructor(
    readonly store: ConnectionProfileStore,
    private readonly dependencies: ConnectionProfileManagerDependencies
  ) {}

  list(): ConnectionProfileView[] {
    return this.store.list().map(viewConnectionProfile)
  }

  current(): ConnectionProfileSummary {
    return summarizeConnectionProfile(this.store.current())
  }

  async connection(): Promise<DaemonConnection> {
    this.assertRunning()
    if (!this.activeConnection) {
      const generation = this.switchGeneration
      const profile = this.store.current()
      const pending = this.resolve(profile).then(
        (ready) => {
          if (generation !== this.switchGeneration && this.activeConnection !== pending) {
            return this.connection()
          }
          if (this.activeConnection === pending && generation === this.switchGeneration) {
            this.rememberDaemonIdentity(profile, ready.daemon)
          }
          return ready
        }
      ).catch((error: unknown) => {
          if (this.activeConnection === pending) this.activeConnection = undefined
          throw error
      })
      this.activeConnection = pending
    }
    return this.activeConnection
  }

  /**
   * Re-verify the active profile and respawn the owned Local daemon if its
   * cached connection exited or is no longer reachable.
   */
  async reconnect(): Promise<DaemonConnection> {
    this.assertRunning()
    const generation = this.switchGeneration
    const profile = this.store.current()
    let ready: DaemonConnection
    try {
      ready = await this.resolve(profile)
    } catch (error) {
      if (generation === this.switchGeneration && this.store.current().id === profile.id) {
        this.activeConnection = undefined
      }
      throw error
    }
    if (generation !== this.switchGeneration || this.store.current().id !== profile.id) {
      throw new Error('The selected connection changed. Try again.')
    }
    this.rememberDaemonIdentity(profile, ready.daemon)
    this.activeConnection = Promise.resolve(ready)
    return ready
  }

  async verify(onVerified?: (ready: DaemonConnection) => void): Promise<ConnectionVerification> {
    try {
      const ready = await this.reconnect()
      onVerified?.(ready)
      return { status: 'healthy', address: ready.address, profile: ready.profile, daemon: ready.daemon }
    } catch (error) {
      return {
        status: error instanceof ConnectionVerificationError ? error.status : 'offline',
        message: error instanceof Error ? error.message : String(error)
      }
    }
  }

  async activateProfile(id: string): Promise<DaemonConnection> {
    return this.select(id)
  }

  async test(idOrProfile: string | RemoteConnectionProfileInput): Promise<DaemonConnection> {
    this.assertRunning()
    const profile = typeof idOrProfile === 'string'
      ? this.store.get(idOrProfile)
      : this.normalizeRemoteInput(idOrProfile)
    if (!profile) throw new Error('Connection profile does not exist.')
    return this.resolve(profile)
  }

  async upsert(input: RemoteConnectionProfileInput): Promise<RemoteConnectionProfile> {
    this.assertRunning()
    const candidate = this.normalizeRemoteInput(input)
    const isEditingActive = candidate.id === this.store.current().id
    const generation = isEditingActive ? ++this.switchGeneration : this.switchGeneration
    const ready = isEditingActive ? await this.resolve(candidate) : undefined
    if (isEditingActive && generation !== this.switchGeneration) {
      const existing = this.store.get(candidate.id)
      if (existing?.kind === 'remote') return existing
      throw new Error('The connection profile was deleted while it was being updated.')
    }
    const profile = this.store.upsert({
      ...candidate,
      daemonInstanceId: ready?.daemon.daemon?.instance_id ?? candidate.daemonInstanceId
    })
    if (
      ready &&
      generation === this.switchGeneration &&
      this.store.current().id === profile.id
    ) {
      this.activeConnection = Promise.resolve(ready)
    }
    return profile
  }

  async claimAndSave(input: PairingClaimInput): Promise<ClaimedConnection> {
    this.assertRunning()
    if (!this.store.canPersistRemoteCredentials()) {
      throw new Error('Unlock secure credential storage before claiming a one-time pairing code.')
    }
    const candidate = normalizeRemoteProfile(
      {
        ...(input.id ? { id: input.id } : {}),
        name: input.name,
        baseUrl: normalizePairingDaemonBaseUrl(input.baseUrl),
        apiKey: null
      },
      randomUUID
    )
    const issued = await claimPairingCredential(candidate.baseUrl, input.pairingId, input.code, {
      fetch: this.dependencies.fetch,
      timeoutMs: this.dependencies.handshakeTimeoutMs,
      expectedDaemonInstanceId: input.expectedDaemonInstanceId
    })
    // Persist before the authenticated handshake: the server returns this key
    // once, so a transient handshake failure must not discard the credential.
    let profile: RemoteConnectionProfile
    try {
      profile = this.store.upsert({
        ...candidate,
        apiKey: issued.apiKey,
        daemonInstanceId: input.expectedDaemonInstanceId
      })
    } catch {
      throw new Error('Invite claimed, but the credential was not saved. Check secure storage and available disk space. Ask the host to revoke the newly paired client and create a fresh invite.')
    }
    const isCurrent = profile.id === this.store.current().id
    const generation = isCurrent ? ++this.switchGeneration : this.switchGeneration
    if (isCurrent) this.activeConnection = undefined
    const daemon = (await this.resolve(profile)).daemon
    const ready: DaemonConnection = {
      address: profile.baseUrl,
      api_key: issued.apiKey,
      profile: summarizeConnectionProfile(profile),
      daemon
    }
    if (
      generation === this.switchGeneration &&
      profile.id === this.store.current().id
    ) {
      this.activeConnection = Promise.resolve(ready)
    }
    return { profile: ready.profile, daemon, client: issued.client }
  }

  async delete(id: string): Promise<boolean> {
    this.assertRunning()
    const wasActive = this.store.current().id === id
    const generation = wasActive ? ++this.switchGeneration : this.switchGeneration
    let localReplacement: DaemonConnection | undefined
    if (wasActive) {
      // Prove the Local replacement is usable before changing the durable
      // selection. This is the one deletion path that may start Local.
      localReplacement = await this.resolve(this.store.get(LOCAL_CONNECTION_PROFILE_ID)!)
    }
    const deleted = this.store.delete(id)
    if (
      deleted &&
      wasActive &&
      generation === this.switchGeneration &&
      this.store.current().id === LOCAL_CONNECTION_PROFILE_ID
    ) {
      this.activeConnection = localReplacement
        ? Promise.resolve(localReplacement)
        : undefined
    }
    return deleted
  }

  async select(id: string): Promise<DaemonConnection> {
    this.assertRunning()
    const profile = this.store.get(id)
    if (!profile) throw new Error('Connection profile does not exist.')
    const generation = ++this.switchGeneration
    // Validate before persisting. A typo or incompatible endpoint must not
    // strand the next launch on an unusable profile.
    const ready = await this.resolve(profile)
    if (generation !== this.switchGeneration) {
      return this.connection()
    }
    this.rememberDaemonIdentity(profile, ready.daemon)
    this.store.select(id)
    this.activeConnection = Promise.resolve(ready)
    return ready
  }

  invalidate(): void {
    this.activeConnection = undefined
  }

  /**
   * Drop the cached Local daemon handle after the owned `brazierd` process
   * exits so the next connection attempt spawns a fresh child.
   */
  invalidateLocalConnection(): void {
    this.localStarted = false
    this.localConnection = undefined
    this.resolvedLocalAddress = undefined
    this.resolvedLocalControlKey = undefined
    if (this.store.current().kind === 'local') {
      this.activeConnection = undefined
    }
  }

  shutdown(): Promise<void> {
    if (this.shutdownPromise) return this.shutdownPromise
    this.stopped = true
    const stopLocal = this.localStarted
    this.localStarted = false
    this.localConnection = undefined
    this.resolvedLocalAddress = undefined
    this.resolvedLocalControlKey = undefined
    this.activeConnection = undefined
    // Promise.resolve calls a synchronous dependency before returning, which
    // preserves immediate ownership release while allowing normal app quit to
    // wait for the daemon's real exit.
    try {
      this.shutdownPromise = Promise.resolve(stopLocal ? this.dependencies.stopLocal() : undefined)
    } catch (error) {
      this.shutdownPromise = Promise.reject(error)
    }
    return this.shutdownPromise
  }

  /** Main-process network guard for renderer-direct fetch/WebSocket traffic. */
  allowsRendererNetworkUrl(value: string, rendererOrigin?: string): boolean {
    let url: URL
    try {
      url = new URL(value)
    } catch {
      return false
    }
    if (!['http:', 'https:', 'ws:', 'wss:'].includes(url.protocol)) return true
    if (rendererOrigin && isRendererDevelopmentOrigin(value, rendererOrigin)) return true
    const profile = this.store.current()
    if (profile.kind === 'local') {
      return Boolean(
        this.resolvedLocalAddress && sameEndpoint(url, new URL(this.resolvedLocalAddress))
      )
    }
    return sameEndpoint(url, new URL(profile.baseUrl))
  }

  /**
   * Return the active bearer only to the main-process request interceptor.
   * Renderer JavaScript receives connection metadata, never credentials.
   */
  async rendererApiKeyForUrl(value: string): Promise<string | null> {
    let url: URL
    try {
      url = new URL(value)
    } catch {
      return null
    }
    const ready = await this.connection()
    if (!sameEndpoint(url, new URL(ready.address))) return null
    return ready.api_key
  }

  /** Return the independent local elevation credential only to the request interceptor. */
  async rendererLocalControlKeyForUrl(value: string): Promise<string | null> {
    let url: URL
    try {
      url = new URL(value)
    } catch {
      return null
    }
    await this.connection()
    if (!this.resolvedLocalAddress || !sameEndpoint(url, new URL(this.resolvedLocalAddress))) {
      return null
    }
    return this.resolvedLocalControlKey ?? null
  }

  private async resolve(profile: ConnectionProfile): Promise<DaemonConnection> {
    if (profile.kind === 'local') {
      if (this.dependencies.ensureLocalDaemon) {
        const ensured = await this.dependencies.ensureLocalDaemon()
        const normalized = normalizeDaemonBaseUrl(ensured.address)
        this.resolvedLocalAddress = normalized
        return { ...ensured, address: normalized }
      }
      const hadCachedLocal = Boolean(this.localConnection)
      const handshakeLocal = async (): Promise<DaemonConnection> => {
        const raw = await this.resolveLocal()
        const address = normalizeDaemonBaseUrl(raw.address)
        const daemon = await fetchDaemonInfo(address, raw.api_key, {
          fetch: this.dependencies.fetch,
          timeoutMs: this.dependencies.handshakeTimeoutMs
        })
        return {
          address,
          api_key: raw.api_key,
          profile: summarizeConnectionProfile(profile),
          daemon
        }
      }
      try {
        return await handshakeLocal()
      } catch (error) {
        const shouldStop = this.localStarted
        this.invalidateLocalConnection()
        if (shouldStop) {
          await Promise.resolve(this.dependencies.stopLocal()).catch(() => undefined)
        }
        if (!hadCachedLocal || this.stopped) throw error
        return handshakeLocal()
      }
    }

    const address = normalizeDaemonBaseUrl(profile.baseUrl)
    const daemon = await fetchDaemonInfo(address, profile.apiKey, {
      fetch: this.dependencies.fetch,
      timeoutMs: this.dependencies.handshakeTimeoutMs
    })
    if (profile.daemonInstanceId && daemon.daemon?.instance_id !== profile.daemonInstanceId) {
      throw new ConnectionVerificationError('identity-changed', 'The daemon identity changed. Check the host before pairing it again.')
    }
    return {
      address,
      api_key: profile.apiKey,
      profile: summarizeConnectionProfile(profile),
      daemon
    }
  }

  /** An omitted credential preserves an existing secret; explicit null clears it. */
  private normalizeRemoteInput(input: RemoteConnectionProfileInput): RemoteConnectionProfile {
    const existing = input.id ? this.store.get(input.id) : undefined
    if (existing?.kind === 'remote' && existing.apiKey && input.apiKey === undefined &&
        normalizeDaemonBaseUrl(input.baseUrl) !== existing.baseUrl) {
      throw new Error('Enter a key for the new daemon address, remove the saved key, or pair again.')
    }
    return normalizeRemoteProfile(
      {
        ...input,
        apiKey: input.apiKey === undefined && existing?.kind === 'remote' ? existing.apiKey : input.apiKey,
        daemonInstanceId: existing?.kind === 'remote' && normalizeDaemonBaseUrl(input.baseUrl) === existing.baseUrl
          ? existing.daemonInstanceId
          : undefined
      },
      randomUUID
    )
  }

  private rememberDaemonIdentity(profile: ConnectionProfile, daemon: DaemonInfo): void {
    if (profile.kind === 'remote' && !profile.daemonInstanceId && daemon.daemon?.instance_id) {
      this.store.upsert({ ...profile, daemonInstanceId: daemon.daemon.instance_id })
    }
  }

  private async resolveLocal(): Promise<{
    address: string
    api_key: string | null
    local_control_key?: string | null
  }> {
    this.assertRunning()
    if (!this.localConnection) {
      this.localStarted = true
      this.localConnection = this.dependencies
        .startLocal()
        .then((connection) => {
          if (this.stopped) throw new Error('The connection manager is shutting down.')
          this.resolvedLocalAddress = normalizeDaemonBaseUrl(connection.address)
          this.resolvedLocalControlKey = connection.local_control_key ?? undefined
          return connection
        })
        .catch((error: unknown) => {
          this.localStarted = false
          this.localConnection = undefined
          this.resolvedLocalAddress = undefined
          this.resolvedLocalControlKey = undefined
          throw error
        })
    }
    return this.localConnection
  }

  private assertRunning(): void {
    if (this.stopped) throw new Error('The connection manager is shutting down.')
  }
}

function sameEndpoint(candidate: URL, selected: URL): boolean {
  if (candidate.host !== selected.host) return false
  const candidateSecure = candidate.protocol === 'https:' || candidate.protocol === 'wss:'
  const selectedSecure = selected.protocol === 'https:' || selected.protocol === 'wss:'
  return candidateSecure === selectedSecure
}

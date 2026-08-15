import { randomBytes, randomUUID } from 'node:crypto'
import {
  chmodSync,
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync
} from 'node:fs'
import { dirname } from 'node:path'
import { isIP } from 'node:net'

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
}

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
  stopLocal: () => void
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
  input: RemoteConnectionProfileInput,
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
    name: cleanName(input.name),
    kind: 'remote',
    baseUrl,
    apiKey
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
          apiKey
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
      ...(profile.apiKey
        ? credentialCodec
          ? { encryptedApiKey: credentialCodec.encrypt(profile.apiKey) }
          : { apiKey: profile.apiKey }
        : { apiKey: null })
    }))
  }
  writeFileSync(temporary, `${JSON.stringify(persisted, null, 2)}\n`, { mode: 0o600 })
  renameSync(temporary, path)
  // rename(2) preserves the temporary file's mode. chmod also repairs a file
  // created by an older build with broader permissions.
  chmodSync(path, 0o600)
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

  upsert(input: RemoteConnectionProfileInput): RemoteConnectionProfile {
    if (!this.remoteWritesAllowed) {
      throw new Error('Remote credentials cannot be saved while secure storage is unavailable.')
    }
    const profile = normalizeRemoteProfile(input, this.makeId)
    const existing = this.stored.profiles.findIndex((entry) => entry.id === profile.id)
    if (existing === -1) this.stored.profiles.push(profile)
    else this.stored.profiles[existing] = profile
    this.persist()
    return { ...profile }
  }

  delete(id: string): boolean {
    if (id === LOCAL_CONNECTION_PROFILE_ID) {
      throw new Error('The reserved Local connection profile cannot be deleted.')
    }
    const profiles = this.stored.profiles.filter((profile) => profile.id !== id)
    if (profiles.length === this.stored.profiles.length) return false
    this.stored.profiles = profiles
    if (this.stored.activeId === id) this.stored.activeId = LOCAL_CONNECTION_PROFILE_ID
    this.persist()
    return true
  }

  select(id: string): ConnectionProfile {
    const profile = this.get(id)
    if (!profile) throw new Error('Connection profile does not exist.')
    if (this.stored.activeId !== id) {
      this.stored.activeId = id
      this.persist()
    }
    return profile
  }

  private persist(): void {
    writeStoredProfiles(this.path, this.stored, this.credentialCodec)
  }
}

function errorForStatus(status: number): Error {
  if (status === 401 || status === 403) {
    return new Error('The daemon rejected this API key.')
  }
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
export async function claimPairingCredential(
  baseUrl: string,
  pairingId: string,
  code: string,
  options: { fetch?: typeof fetch; timeoutMs?: number } = {}
): Promise<{ client: PairedClient; apiKey: string }> {
  const address = normalizePairingDaemonBaseUrl(baseUrl)
  const id = cleanPairingSecret(pairingId, 'Pairing id')
  const pairingCode = cleanPairingSecret(code, 'Pairing code')
  const fetch_ = options.fetch ?? fetch
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? DEFAULT_HANDSHAKE_TIMEOUT_MS)
  try {
    let response: Response
    try {
      response = await fetch_(`${address}/api/v1/auth/pairings/${encodeURIComponent(id)}/claim`, {
        method: 'POST',
        headers: { accept: 'application/json', 'content-type': 'application/json' },
        body: JSON.stringify({ code: pairingCode }),
        redirect: 'manual',
        signal: controller.signal
      })
    } catch (cause) {
      if (controller.signal.aborted) throw new Error('The pairing request timed out.')
      throw new Error(
        `Could not reach the daemon for pairing: ${cause instanceof Error ? cause.message : String(cause)}`
      )
    }
    if (!response.ok) {
      if (response.status === 400 || response.status === 404) {
        throw new Error('The pairing request is invalid, expired, already used, or the code is incorrect.')
      }
      throw new Error(`The pairing request failed with status ${response.status}.`)
    }
    const value = await boundedJson(response)
    const client = pairedClient(value?.client)
    const apiKey = typeof value?.api_key === 'string' ? value.api_key.trim() : ''
    if (!client || !apiKey || apiKey.length > 4_096) {
      throw new Error('The daemon returned an invalid pairing response.')
    }
    return { client, apiKey }
  } finally {
    clearTimeout(timeout)
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
    if (!this.activeConnection) {
      const pending = this.resolve(this.store.current()).catch((error: unknown) => {
        if (this.activeConnection === pending) this.activeConnection = undefined
        throw error
      })
      this.activeConnection = pending
    }
    return this.activeConnection
  }

  async test(idOrProfile: string | RemoteConnectionProfileInput): Promise<DaemonConnection> {
    const profile = typeof idOrProfile === 'string'
      ? this.store.get(idOrProfile)
      : this.normalizeRemoteInput(idOrProfile)
    if (!profile) throw new Error('Connection profile does not exist.')
    return this.resolve(profile)
  }

  async upsert(input: RemoteConnectionProfileInput): Promise<RemoteConnectionProfile> {
    const activeId = this.store.current().id
    const candidate = this.normalizeRemoteInput(input)
    const ready = candidate.id === activeId ? await this.resolve(candidate) : undefined
    const profile = this.store.upsert(candidate)
    if (ready) this.activeConnection = Promise.resolve(ready)
    return profile
  }

  async claimAndSave(input: PairingClaimInput): Promise<ClaimedConnection> {
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
      timeoutMs: this.dependencies.handshakeTimeoutMs
    })
    // Persist before the authenticated handshake: the server returns this key
    // once, so a transient handshake failure must not discard the credential.
    const profile = this.store.upsert({ ...candidate, apiKey: issued.apiKey })
    if (profile.id === this.store.current().id) this.activeConnection = undefined
    const daemon = await fetchDaemonInfo(profile.baseUrl, issued.apiKey, {
      fetch: this.dependencies.fetch,
      timeoutMs: this.dependencies.handshakeTimeoutMs
    })
    const ready: DaemonConnection = {
      address: profile.baseUrl,
      api_key: issued.apiKey,
      profile: summarizeConnectionProfile(profile),
      daemon
    }
    if (profile.id === this.store.current().id) this.activeConnection = Promise.resolve(ready)
    return { profile: ready.profile, daemon, client: issued.client }
  }

  async delete(id: string): Promise<boolean> {
    const wasActive = this.store.current().id === id
    if (wasActive) {
      // Prove the Local replacement is usable before changing the durable
      // selection. This is the one deletion path that may start Local.
      await this.resolve(this.store.get(LOCAL_CONNECTION_PROFILE_ID)!)
    }
    const deleted = this.store.delete(id)
    if (deleted && wasActive) this.activeConnection = undefined
    return deleted
  }

  async select(id: string): Promise<DaemonConnection> {
    const profile = this.store.get(id)
    if (!profile) throw new Error('Connection profile does not exist.')
    if (profile.id === this.store.current().id) return this.connection()
    // Validate before persisting. A typo or incompatible endpoint must not
    // strand the next launch on an unusable profile.
    const ready = await this.resolve(profile)
    this.store.select(id)
    this.activeConnection = Promise.resolve(ready)
    return ready
  }

  invalidate(): void {
    this.activeConnection = undefined
  }

  shutdown(): void {
    if (!this.localStarted) return
    this.dependencies.stopLocal()
    this.localStarted = false
    this.localConnection = undefined
    this.resolvedLocalAddress = undefined
    this.resolvedLocalControlKey = undefined
    this.activeConnection = undefined
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
    const raw = profile.kind === 'local' ? await this.resolveLocal() : {
      address: profile.baseUrl,
      api_key: profile.apiKey
    }
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

  /** An omitted credential preserves an existing secret; explicit null clears it. */
  private normalizeRemoteInput(input: RemoteConnectionProfileInput): RemoteConnectionProfile {
    const existing = input.id ? this.store.get(input.id) : undefined
    return normalizeRemoteProfile(
      input.apiKey === undefined && existing?.kind === 'remote'
        ? { ...input, apiKey: existing.apiKey }
        : input,
      randomUUID
    )
  }

  private async resolveLocal(): Promise<{
    address: string
    api_key: string | null
    local_control_key?: string | null
  }> {
    if (!this.localConnection) {
      this.localStarted = true
      this.localConnection = this.dependencies
        .startLocal()
        .then((connection) => {
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
}

function sameEndpoint(candidate: URL, selected: URL): boolean {
  if (candidate.host !== selected.host) return false
  const candidateSecure = candidate.protocol === 'https:' || candidate.protocol === 'wss:'
  const selectedSecure = selected.protocol === 'https:' || selected.protocol === 'wss:'
  return candidateSecure === selectedSecure
}

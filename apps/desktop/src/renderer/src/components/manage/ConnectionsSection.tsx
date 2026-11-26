import { Check, Copy, KeyRound, LoaderCircle, RefreshCw, Trash2 } from 'lucide-react'
import { type ClipboardEvent, type FormEvent, useCallback, useEffect, useState } from 'react'
import QRCode from 'react-qr-code'
import {
  encodePairingInvite,
  normalizeShareableOrigin,
  parsePairingInvite,
  type PairingInvite
} from '../../../../../../../packages/pairing/src/invite'
import {
  cancelDaemonPairing,
  claimConnectionProfile,
  createDaemonPairing,
  deleteConnectionProfile,
  fetchCurrentConnectionProfile,
  isPendingDaemonPairing,
  listConnectionProfiles as listDesktopConnectionProfiles,
  listDaemonApiClients,
  listDaemonPairings,
  revokeDaemonApiClient,
  selectConnectionProfile,
  testConnectionProfile as testDesktopConnectionProfile,
  upsertConnectionProfile,
  type ClientScope,
  type ConnectionProfile,
  type ConnectionProfileSummary,
  type DaemonApiClient,
  type DaemonPairingRequest
} from '../../api'

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

type RemoteProfileDraft = {
  id?: string
  name: string
  baseUrl: string
  apiKey: string
  hasSavedApiKey: boolean
  clearApiKey: boolean
  pairingId: string
  pairingCode: string
  pairingExpiresAt?: number
}

const EMPTY_REMOTE_PROFILE: RemoteProfileDraft = {
  name: '',
  baseUrl: '',
  apiKey: '',
  hasSavedApiKey: false,
  clearApiKey: false,
  pairingId: '',
  pairingCode: ''
}

export function ConnectionsSection(props: { onError: (message: string | null) => void }): React.JSX.Element {
  const [profiles, setProfiles] = useState<ConnectionProfile[]>([])
  const [current, setCurrent] = useState<ConnectionProfileSummary | null>(null)
  const [draft, setDraft] = useState<RemoteProfileDraft>(EMPTY_REMOTE_PROFILE)
  const [loading, setLoading] = useState(true)
  const [working, setWorking] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [pairings, setPairings] = useState<DaemonPairingRequest[]>([])
  const [clients, setClients] = useState<DaemonApiClient[]>([])
  const [pairingClientName, setPairingClientName] = useState('Brazier desktop')
  const [pairingScopes, setPairingScopes] = useState<ClientScope[]>(['inference'])
  const [createdPairing, setCreatedPairing] = useState<PairingInvite | null>(null)
  const [pairingPreviewReachable, setPairingPreviewReachable] = useState(true)
  const [inviteOrigin, setInviteOrigin] = useState<string | null>(null)
  const [localOrigin, setLocalOrigin] = useState<string | null>(null)
  const [pairingCopied, setPairingCopied] = useState(false)
  const [review, setReview] = useState<{
    draft: RemoteProfileDraft
    result: Awaited<ReturnType<typeof window.brazier.previewPairing>>
  } | null>(null)
  const [connectNow, setConnectNow] = useState<string | null>(null)
  const [now, setNow] = useState(() => Math.floor(Date.now() / 1_000))
  const [trustError, setTrustError] = useState<string | null>(null)
  const [trustLoading, setTrustLoading] = useState(true)
  const [confirmRevoke, setConfirmRevoke] = useState<string | null>(null)
  const [grantableScopes, setGrantableScopes] = useState<ClientScope[] | null>(null)

  const reload = useCallback(async (): Promise<void> => {
    setLoading(true)
    try {
      const [entries, active] = await Promise.all([
        listDesktopConnectionProfiles(),
        fetchCurrentConnectionProfile()
      ])
      setProfiles(entries)
      setCurrent(active)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setLoading(false)
    }
  }, [props.onError])

  useEffect(() => {
    void reload()
  }, [reload])

  useEffect(() => {
    if (current?.kind !== 'local') {
      setLocalOrigin(null)
      return
    }
    let cancelled = false
    void window.brazier.localPairingOrigin()
      .then((origin) => { if (!cancelled) setLocalOrigin(origin) })
      .catch(() => { if (!cancelled) setLocalOrigin(null) })
    return () => { cancelled = true }
  }, [current?.id, current?.kind])

  const reloadTrust = useCallback(async (): Promise<void> => {
    setTrustLoading(true)
    try {
      const [nextPairings, nextClients] = await Promise.all([
        listDaemonPairings(),
        listDaemonApiClients()
      ])
      setPairings(nextPairings)
      setClients(nextClients)
      setCreatedPairing((invite) => invite &&
        nextPairings.some((pairing) => pairing.id === invite.pairingId)
        && !nextPairings.some((pairing) => pairing.id === invite.pairingId &&
          isPendingDaemonPairing(pairing)) ? null : invite)
      setTrustError(null)
    } catch (cause) {
      setTrustError(errorText(cause))
    } finally {
      setTrustLoading(false)
    }
  }, [])

  useEffect(() => {
    void reloadTrust()
    let active = true
    void window.brazier.getConnection().then(({ daemon }) => {
      if (!active || !daemon.client || daemon.client.owner) return
      const scopes = daemon.client.scopes.includes('management') ? daemon.client.scopes : []
      setGrantableScopes(scopes)
      setPairingScopes((current) => {
        const retained = current.filter((scope) => scopes.includes(scope))
        return retained.length || !scopes.length ? retained : [scopes[0]]
      })
    }).catch(() => undefined)
    return () => { active = false }
  }, [reloadTrust])

  useEffect(() => {
    const receivingExpiry = review?.draft === draft ? review.result.pairing.expires_at : draft.pairingExpiresAt
    if (!createdPairing && !receivingExpiry) return
    const tick = (): void => {
      const seconds = Math.floor(Date.now() / 1_000)
      setNow(seconds)
      if (createdPairing && seconds >= createdPairing.expiresAt) setCreatedPairing(null)
      if (receivingExpiry && seconds >= receivingExpiry) {
        setReview(null)
        setDraft((current) => current === draft ? { ...current, pairingId: '', pairingCode: '', pairingExpiresAt: undefined } : current)
        props.onError('This invite expired. Ask the host for a fresh invite.')
      }
    }
    tick()
    const timer = window.setInterval(tick, 1_000)
    return () => window.clearInterval(timer)
  }, [createdPairing, review, draft, props.onError])

  useEffect(() => {
    if (!createdPairing) return
    const id = createdPairing.pairingId
    let active = true
    const timer = window.setInterval(() => {
      void listDaemonPairings().then((entries) => {
        if (!active) return
        setPairings(entries)
        if (!entries.some((entry) => entry.id === id && isPendingDaemonPairing(entry))) {
          setCreatedPairing((invite) => invite?.pairingId === id ? null : invite)
        }
      }).catch(() => undefined)
    }, 2_000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [createdPairing])

  function edit(profile: Extract<ConnectionProfile, { kind: 'remote' }>): void {
    setDraft({
      id: profile.id,
      name: profile.name,
      baseUrl: profile.baseUrl,
      apiKey: '',
      hasSavedApiKey: profile.hasApiKey,
      clearApiKey: false,
      pairingId: '',
      pairingCode: ''
    })
    setNotice(null)
  }

  async function test(idOrDraft: string | RemoteProfileDraft): Promise<void> {
    setWorking(typeof idOrDraft === 'string' ? `test:${idOrDraft}` : 'test:draft')
    props.onError(null)
    setNotice(null)
    try {
      const ready = await testDesktopConnectionProfile(
        typeof idOrDraft === 'string'
          ? idOrDraft
          : {
              ...(idOrDraft.id ? { id: idOrDraft.id } : {}),
              name: idOrDraft.name,
              baseUrl: idOrDraft.baseUrl,
              ...(idOrDraft.clearApiKey
                ? { apiKey: null }
                : idOrDraft.apiKey
                  ? { apiKey: idOrDraft.apiKey }
                  : idOrDraft.id
                    ? {}
                    : { apiKey: null })
            }
      )
      setNotice(`Connected to ${ready.profile.name} · Brazier ${ready.daemon.version}.`)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setWorking(null)
    }
  }

  async function save(event: FormEvent): Promise<void> {
    event.preventDefault()
    setWorking('save')
    props.onError(null)
    setNotice(null)
    try {
      await upsertConnectionProfile({
        ...(draft.id ? { id: draft.id } : {}),
        name: draft.name,
        baseUrl: draft.baseUrl,
        ...(draft.clearApiKey
          ? { apiKey: null }
          : draft.apiKey
            ? { apiKey: draft.apiKey }
            : draft.id
              ? {}
              : { apiKey: null })
      })
      setDraft(EMPTY_REMOTE_PROFILE)
      setNotice(draft.id ? 'Connection updated.' : 'Connection saved.')
      await reload()
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setWorking(null)
    }
  }

  async function claimPairing(): Promise<void> {
    setWorking('claim:draft')
    props.onError(null)
    setNotice(null)
    try {
      if (!review || review.draft !== draft) {
        const result = await window.brazier.previewPairing(draft.baseUrl, draft.pairingId, draft.pairingCode)
        setReview({ draft, result })
        return
      }
      const claimed = await claimConnectionProfile({
        ...(draft.id ? { id: draft.id } : {}),
        name: draft.name,
        baseUrl: draft.baseUrl,
        pairingId: draft.pairingId,
        code: draft.pairingCode,
        expectedDaemonInstanceId: review.result.daemon.instance_id
      })
      setReview(null)
      setDraft(EMPTY_REMOTE_PROFILE)
      setConnectNow(claimed.profile.id === current?.id ? null : claimed.profile.id)
      setNotice(
        `Paired ${claimed.profile.name} with ${claimed.client.scopes.join(', ')} access · Brazier ${claimed.daemon.version}.`
      )
      await reload()
    } catch (cause) {
      if (review?.draft === draft) {
        setReview(null)
        setDraft((current) => current === draft ? { ...current, pairingId: '', pairingCode: '', pairingExpiresAt: undefined } : current)
      }
      await reload().catch(() => undefined)
      props.onError(errorText(cause))
    } finally {
      setWorking(null)
    }
  }

  function togglePairingScope(scope: ClientScope): void {
    setPairingScopes((currentScopes) =>
      currentScopes.includes(scope)
        ? currentScopes.filter((entry) => entry !== scope)
        : [...currentScopes, scope]
    )
  }

  async function createPairing(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (pairingScopes.length === 0) {
      setTrustError('Select at least one client scope.')
      return
    }
    setWorking('pair:create')
    setTrustError(null)
    setPairingCopied(false)
    setCreatedPairing(null)
    try {
      const origin = normalizeShareableOrigin(originInput.trim() || localOrigin || '')
      const { hostName, previewReachable } = await window.brazier.verifyPairingOrigin(origin)
      const created = await createDaemonPairing({
        clientName: pairingClientName,
        scopes: pairingScopes,
        ttlSeconds: 300
      })
      setNow(Math.floor(Date.now() / 1_000))
      setPairingPreviewReachable(previewReachable)
      setCreatedPairing({
        version: 1,
        origin,
        hostName,
        pairingId: created.pairing.id,
        code: created.code,
        scopes: created.pairing.scopes,
        expiresAt: created.pairing.expires_at
      })
      await reloadTrust()
    } catch (cause) {
      setTrustError(errorText(cause))
    } finally {
      setWorking(null)
    }
  }

  async function copyPairing(): Promise<void> {
    if (!createdPairing) return
    try {
      await window.brazier.copyText(encodePairingInvite(createdPairing))
      setPairingCopied(true)
    } catch (cause) {
      setTrustError(errorText(cause))
    }
  }

  function pastePairing(event: ClipboardEvent<HTMLInputElement>): void {
    const text = event.clipboardData.getData('text/plain')
    const invite = parsePairingInvite(text)
    if (!invite) {
      if (text.trim().startsWith('brazier-pairing:')) {
        event.preventDefault()
        props.onError('This pairing invite is invalid or unsupported. Ask the host for a fresh invite.')
      }
      return
    }
    event.preventDefault()
    if (invite.version === 1 && invite.expiresAt <= Math.floor(Date.now() / 1_000)) {
      props.onError('This invite expired. Ask the host for a fresh invite.')
      return
    }
    setReview(null)
    setDraft({
      ...draft, baseUrl: invite.origin ?? draft.baseUrl, pairingId: invite.pairingId, pairingCode: invite.code,
      pairingExpiresAt: invite.version === 1 ? invite.expiresAt : undefined
    })
  }

  async function cancelPairing(id: string): Promise<void> {
    setWorking(`pair:cancel:${id}`)
    setTrustError(null)
    try {
      await cancelDaemonPairing(id)
      if (createdPairing?.pairingId === id) setCreatedPairing(null)
      await reloadTrust()
    } catch (cause) {
      setTrustError(errorText(cause))
    } finally {
      setWorking(null)
    }
  }

  async function revokeClient(id: string): Promise<void> {
    setWorking(`client:revoke:${id}`)
    setTrustError(null)
    try {
      await revokeDaemonApiClient(id)
      setConfirmRevoke(null)
      await reloadTrust()
    } catch (cause) {
      setTrustError(errorText(cause))
    } finally {
      setWorking(null)
    }
  }

  async function select(id: string): Promise<void> {
    setWorking(`select:${id}`)
    props.onError(null)
    try {
      await selectConnectionProfile(id)
      // Main sends the switch event and reloads after the IPC reply. Keep the
      // control visibly busy during that short reset boundary.
      setNotice('Switching connection…')
    } catch (cause) {
      props.onError(errorText(cause))
      setWorking(null)
    }
  }

  async function remove(id: string): Promise<void> {
    setWorking(`delete:${id}`)
    props.onError(null)
    try {
      await deleteConnectionProfile(id)
      setConfirmDelete(null)
      await reload()
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setWorking(null)
    }
  }

  let suggestedOrigin = ''
  try {
    if (current?.kind === 'remote' && current.baseUrl) {
      suggestedOrigin = normalizeShareableOrigin(current.baseUrl)
    }
  } catch {
    // Leave non-shareable addresses out of the invitation form.
  }
  const originInput = inviteOrigin ?? suggestedOrigin
  const pendingPairings = pairings.filter((pairing) => isPendingDaemonPairing(pairing))
  const activeClients = clients.filter((client) => !client.revoked_at)
  const reviewed = review?.draft === draft ? review.result : null
  const remaining = createdPairing ? Math.max(0, createdPairing.expiresAt - now) : 0

  return (
    <section>
      <header className="manage-heading">
        <h2>Desktop connections</h2>
        <p>
          Choose which Brazier daemon owns conversations, models, runtimes, tools, and filesystem
          paths. Remote profiles never start or stop a daemon on this computer.
        </p>
      </header>

      {loading ? (
        <div className="manage-placeholder">
          <LoaderCircle className="spin" size={16} /> Loading connections…
        </div>
      ) : (
        <div className="runtime-list connection-profile-list">
          {profiles.map((profile) => {
            const selected = current?.id === profile.id
            const testKey = `test:${profile.id}`
            return (
              <article className={`runtime-card${selected ? ' active' : ''}`} key={profile.id}>
                <div className="runtime-card-info">
                  <strong>
                    {profile.name}
                    {selected ? <span className="installed-badge">Current</span> : null}
                  </strong>
                  <span>
                    {profile.kind === 'local' ? 'This computer · managed desktop daemon' : profile.baseUrl}
                  </span>
                </div>
                <div className="library-card-actions">
                  {profile.kind === 'remote' ? (
                    <button className="chip-button subtle" onClick={() => edit(profile)}>
                      Edit
                    </button>
                  ) : null}
                  <button
                    className="chip-button"
                    disabled={working != null}
                    onClick={() => void test(profile.id)}
                  >
                    {working === testKey ? <LoaderCircle className="spin" size={13} /> : 'Test'}
                  </button>
                  <button
                    className="chip-button"
                    disabled={selected || working != null}
                    onClick={() => void select(profile.id)}
                  >
                    {working === `select:${profile.id}` ? 'Switching…' : 'Use'}
                  </button>
                  {profile.kind === 'remote' ? (
                    confirmDelete === profile.id ? (
                      <>
                        <button className="chip-button danger" onClick={() => void remove(profile.id)}>
                          Delete
                        </button>
                        <button className="chip-button subtle" onClick={() => setConfirmDelete(null)}>
                          Keep
                        </button>
                      </>
                    ) : (
                      <button
                        className="chip-button subtle"
                        title="Delete connection profile"
                        onClick={() => setConfirmDelete(profile.id)}
                      >
                        <Trash2 size={13} />
                      </button>
                    )
                  ) : null}
                </div>
              </article>
            )
          })}
        </div>
      )}

      <form className="settings-group" onSubmit={(event) => {
        if (draft.pairingId || draft.pairingCode) {
          event.preventDefault()
          void claimPairing()
        } else void save(event)
      }}>
        <div className="section-label">{draft.id ? 'Edit remote connection' : 'Add remote connection'}</div>
        <div className="settings-grid">
          <label>
            <span>Name (optional)</span>
            <input
              maxLength={80}
              value={draft.name}
              placeholder="Lab GPU"
              onChange={(event) => setDraft({ ...draft, name: event.target.value })}
            />
          </label>
          <label>
            <span>Daemon URL</span>
            <input
              required
              onPaste={pastePairing}
              value={draft.baseUrl}
              placeholder="https://labbox.example:7614"
              onChange={(event) => setDraft({ ...draft, baseUrl: event.target.value })}
            />
          </label>
          <label>
            <span>API key</span>
            <input
              type="password"
              value={draft.apiKey}
              disabled={draft.clearApiKey}
              placeholder={
                draft.hasSavedApiKey
                  ? 'Leave blank to keep the saved key'
                  : 'Optional when authentication is disabled'
              }
              onChange={(event) =>
                setDraft({ ...draft, apiKey: event.target.value, clearApiKey: false })
              }
            />
          </label>
          {draft.id && draft.hasSavedApiKey ? (
            <label className="settings-toggle">
              <input
                type="checkbox"
                checked={draft.clearApiKey}
                onChange={(event) =>
                  setDraft({
                    ...draft,
                    apiKey: '',
                    clearApiKey: event.target.checked
                  })
                }
              />
              <span>Remove the saved API key</span>
            </label>
          ) : null}
          <label>
            <span>Pairing id</span>
            <input
              onPaste={pastePairing}
              required={Boolean(draft.pairingCode)}
              value={draft.pairingId}
              placeholder="Paste pairing details or ID"
              autoComplete="off"
              onChange={(event) => setDraft({ ...draft, pairingId: event.target.value, pairingExpiresAt: undefined })}
            />
          </label>
          <label>
            <span>Pairing code</span>
            <input
              type="password"
              onPaste={pastePairing}
              required={Boolean(draft.pairingId)}
              value={draft.pairingCode}
              placeholder="Paste the one-time pairing code"
              autoComplete="off"
              onChange={(event) => setDraft({ ...draft, pairingCode: event.target.value, pairingExpiresAt: undefined })}
            />
          </label>
        </div>
        <p className="model-help">
          Plain HTTP exposes credentials and conversation data to the network. Prefer HTTPS outside
          a trusted private network. Pairing refuses public HTTP endpoints.
        </p>
        {reviewed ? (
          <div className="runtime-card" role="status">
            <div className="runtime-card-info">
              <strong>Review pairing with {reviewed.daemon.display_name}</strong>
              <span>Daemon: {new URL(draft.baseUrl.trim()).origin}</span>
              <span>Device name: {reviewed.pairing.client_name}</span>
              <span>Access: {reviewed.pairing.scopes.join(', ')}</span>
              <span>Expires in {Math.max(0, reviewed.pairing.expires_at - now)}s</span>
            </div>
          </div>
        ) : null}
        <div className="runtime-actions">
          <button
            type="button"
            className="secondary-action"
            disabled={!draft.baseUrl.trim() || working != null}
            onClick={() => void test(draft)}
          >
            {working === 'test:draft' ? <LoaderCircle className="spin" size={14} /> : null}
            Test
          </button>
          <button className="primary-action" type="submit" disabled={working != null}>
            {working === 'save' || working === 'claim:draft' ? <LoaderCircle className="spin" size={14} /> : null}
            {reviewed ? 'Confirm and save' : draft.pairingId || draft.pairingCode ? 'Review invite' : draft.id ? 'Save changes' : 'Add connection'}
          </button>
          {draft.id ? (
            <button type="button" className="chip-button subtle" onClick={() => setDraft(EMPTY_REMOTE_PROFILE)}>
              Cancel edit
            </button>
          ) : null}
        </div>
        {notice ? <p className="model-help">{notice}</p> : null}
        {connectNow ? (
          <button type="button" className="chip-button" disabled={working != null} onClick={() => void select(connectNow)}>
            Connect now
          </button>
        ) : null}
      </form>

      <div className="settings-group">
        <div className="settings-group-head">
          <div>
            <div className="section-label">Pair clients with {current?.name ?? 'current daemon'}</div>
            <p className="model-help">
              Create a five-minute, one-time invite. Scan its QR code with another device, or copy
              the invite and share it through a trusted channel.
            </p>
          </div>
          <button
            type="button"
            className="chip-button subtle"
            disabled={trustLoading || working != null}
            onClick={() => void reloadTrust()}
          >
            {trustLoading ? <LoaderCircle className="spin" size={13} /> : <RefreshCw size={13} />}
            Refresh
          </button>
        </div>

        {trustError ? <div className="error-banner" role="alert"><span>{trustError}</span></div> : null}

        {grantableScopes?.length === 0 ? (
          <p className="model-help">This device does not have permission to manage pairing invitations.</p>
        ) : (
        <form className="settings-grid" onSubmit={(event) => void createPairing(event)}>
          <label>
            <span>Reachable daemon HTTPS address</span>
            <input
              required={!localOrigin}
              type="url"
              value={originInput}
              placeholder={localOrigin ?? 'https://brazier.example'}
              onChange={(event) => setInviteOrigin(event.target.value)}
            />
            <small>
              Use the address the other device can reach through your HTTPS proxy or private network.
              {localOrigin ? ' Leave blank to use this computer’s local network address.' : null}
            </small>
          </label>
          <label>
            <span>Client name</span>
            <input
              required
              maxLength={80}
              value={pairingClientName}
              placeholder="Dylan's laptop"
              onChange={(event) => setPairingClientName(event.target.value)}
            />
          </label>
          <fieldset>
            <legend>Allowed scopes</legend>
            <small>Start with inference; management can create/revoke clients, and agent can approve machine actions.</small>
            {(grantableScopes ?? ['inference', 'management', 'agent'] as ClientScope[]).map((scope) => (
              <label key={scope}>
                <input
                  type="checkbox"
                  checked={pairingScopes.includes(scope)}
                  onChange={() => togglePairingScope(scope)}
                />
                <span>{scope}</span>
              </label>
            ))}
          </fieldset>
          <div className="runtime-actions">
            <button
              className="primary-action"
              type="submit"
              disabled={!(originInput.trim() || localOrigin) || !pairingClientName.trim() || pairingScopes.length === 0 || working != null}
            >
              {working === 'pair:create' ? <LoaderCircle className="spin" size={14} /> : <KeyRound size={14} />}
              Create pairing
            </button>
          </div>
        </form>
        )}

        {createdPairing && remaining > 0 ? (
          <article className="runtime-card active">
            <div className="runtime-card-info">
              <strong>Pair with {createdPairing.hostName}</strong>
              <span>{createdPairing.origin} · {createdPairing.scopes.join(', ')}</span>
              <span>Expires in {Math.floor(remaining / 60)}:{String(remaining % 60).padStart(2, '0')}</span>
              {!pairingPreviewReachable && (
                <p className="runtime-update-note" role="status">
                  Brazier could not verify the pairing preview route at this address. The other
                  device needs this route to confirm the invite. Before sharing, check that your
                  proxy forwards the route and the host supports pairing preview.
                </p>
              )}
              <div style={{ padding: 12, background: '#fff', width: 'fit-content' }}>
                <QRCode value={encodePairingInvite(createdPairing)} size={172} title="One-time Brazier pairing invite" />
              </div>
            </div>
            <div className="library-card-actions">
              <button type="button" className="chip-button" onClick={() => void copyPairing()}>
                {pairingCopied ? <Check size={13} /> : <Copy size={13} />}
                {pairingCopied ? 'Copied' : 'Copy invite'}
              </button>
              <button type="button" className="chip-button subtle" onClick={() => setCreatedPairing(null)}>
                Hide code
              </button>
            </div>
          </article>
        ) : null}

        <div className="section-label">Pending pairings</div>
        {trustLoading ? (
          <div className="manage-placeholder"><LoaderCircle className="spin" size={16} /> Loading pairings…</div>
        ) : pendingPairings.length === 0 ? (
          <p className="model-help">No pending pairing requests.</p>
        ) : (
          <div className="runtime-list">
            {pendingPairings.map((pairing) => (
              <article className="runtime-card" key={pairing.id}>
                <div className="runtime-card-info">
                  <strong>{pairing.client_name}</strong>
                  <span>{pairing.scopes.join(', ')} · expires {new Date(pairing.expires_at * 1_000).toLocaleString()}</span>
                </div>
                <div className="library-card-actions">
                  <button
                    type="button"
                    className="chip-button danger"
                    disabled={working != null}
                    onClick={() => void cancelPairing(pairing.id)}
                  >
                    {working === `pair:cancel:${pairing.id}` ? <LoaderCircle className="spin" size={13} /> : null}
                    Cancel
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}

        <div className="section-label">Authorized clients</div>
        {!trustLoading && activeClients.length === 0 ? (
          <p className="model-help">No active paired clients.</p>
        ) : (
          <div className="runtime-list">
            {activeClients.map((client) => (
              <article className="runtime-card" key={client.id}>
                <div className="runtime-card-info">
                  <strong>{client.name}</strong>
                  <span>
                    {client.scopes.join(', ')} · {client.last_used_at
                      ? `last used ${new Date(client.last_used_at).toLocaleString()}`
                      : `created ${new Date(client.created_at).toLocaleString()}`}
                  </span>
                </div>
                <div className="library-card-actions">
                  {confirmRevoke === client.id ? (
                    <>
                      <button
                        type="button"
                        className="chip-button danger"
                        disabled={working != null}
                        onClick={() => void revokeClient(client.id)}
                      >
                        {working === `client:revoke:${client.id}` ? <LoaderCircle className="spin" size={13} /> : null}
                        Revoke access
                      </button>
                      <button type="button" className="chip-button subtle" onClick={() => setConfirmRevoke(null)}>Keep</button>
                    </>
                  ) : (
                    <button type="button" className="chip-button subtle" onClick={() => setConfirmRevoke(client.id)}>Revoke</button>
                  )}
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}

import { Check, Copy, KeyRound, LoaderCircle, RefreshCw, Trash2 } from 'lucide-react'
import { type FormEvent, useCallback, useEffect, useState } from 'react'
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
  type CreatedDaemonPairing,
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
  const [createdPairing, setCreatedPairing] = useState<CreatedDaemonPairing | null>(null)
  const [pairingCopied, setPairingCopied] = useState(false)
  const [trustError, setTrustError] = useState<string | null>(null)
  const [trustLoading, setTrustLoading] = useState(true)
  const [confirmRevoke, setConfirmRevoke] = useState<string | null>(null)

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

  const reloadTrust = useCallback(async (): Promise<void> => {
    setTrustLoading(true)
    try {
      const [nextPairings, nextClients] = await Promise.all([
        listDaemonPairings(),
        listDaemonApiClients()
      ])
      setPairings(nextPairings)
      setClients(nextClients)
      setTrustError(null)
    } catch (cause) {
      setTrustError(errorText(cause))
    } finally {
      setTrustLoading(false)
    }
  }, [])

  useEffect(() => {
    void reloadTrust()
  }, [reloadTrust])

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
      const claimed = await claimConnectionProfile({
        ...(draft.id ? { id: draft.id } : {}),
        name: draft.name,
        baseUrl: draft.baseUrl,
        pairingId: draft.pairingId,
        code: draft.pairingCode
      })
      setDraft(EMPTY_REMOTE_PROFILE)
      setNotice(
        `Paired ${claimed.profile.name} with ${claimed.client.scopes.join(', ')} access · Brazier ${claimed.daemon.version}.`
      )
      await reload()
    } catch (cause) {
      // A claim is one-time. Main persists the issued key before compatibility
      // probing, so refresh even when that probe reports an error.
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
    try {
      const created = await createDaemonPairing({
        clientName: pairingClientName,
        scopes: pairingScopes,
        ttlSeconds: 300
      })
      setCreatedPairing(created)
      await reloadTrust()
    } catch (cause) {
      setTrustError(errorText(cause))
    } finally {
      setWorking(null)
    }
  }

  async function copyPairing(): Promise<void> {
    if (!createdPairing) return
    await window.brazier.copyText(
      `Pairing id: ${createdPairing.pairing.id}\nPairing code: ${createdPairing.code}`
    )
    setPairingCopied(true)
  }

  async function cancelPairing(id: string): Promise<void> {
    setWorking(`pair:cancel:${id}`)
    setTrustError(null)
    try {
      await cancelDaemonPairing(id)
      if (createdPairing?.pairing.id === id) setCreatedPairing(null)
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

  const pendingPairings = pairings.filter((pairing) => isPendingDaemonPairing(pairing))
  const activeClients = clients.filter((client) => !client.revoked_at)

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

      <form className="settings-group" onSubmit={(event) => void save(event)}>
        <div className="section-label">{draft.id ? 'Edit remote connection' : 'Add remote connection'}</div>
        <div className="settings-grid">
          <label>
            <span>Name</span>
            <input
              required
              value={draft.name}
              placeholder="Lab GPU"
              onChange={(event) => setDraft({ ...draft, name: event.target.value })}
            />
          </label>
          <label>
            <span>Daemon URL</span>
            <input
              required
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
              value={draft.pairingId}
              placeholder="Paste the one-time pairing id"
              autoComplete="off"
              onChange={(event) => setDraft({ ...draft, pairingId: event.target.value })}
            />
          </label>
          <label>
            <span>Pairing code</span>
            <input
              type="password"
              value={draft.pairingCode}
              placeholder="Paste the one-time pairing code"
              autoComplete="off"
              onChange={(event) => setDraft({ ...draft, pairingCode: event.target.value })}
            />
          </label>
        </div>
        <p className="model-help">
          Plain HTTP exposes credentials and conversation data to the network. Prefer HTTPS outside
          a trusted private network. Pairing refuses public HTTP endpoints.
        </p>
        <div className="runtime-actions">
          <button
            type="button"
            className="secondary-action"
            disabled={!draft.name.trim() || !draft.baseUrl.trim() || working != null}
            onClick={() => void test(draft)}
          >
            {working === 'test:draft' ? <LoaderCircle className="spin" size={14} /> : null}
            Test
          </button>
          <button className="primary-action" type="submit" disabled={working != null}>
            {working === 'save' ? <LoaderCircle className="spin" size={14} /> : null}
            {draft.id ? 'Save changes' : 'Add connection'}
          </button>
          <button
            type="button"
            className="primary-action"
            disabled={
              !draft.name.trim() ||
              !draft.baseUrl.trim() ||
              !draft.pairingId.trim() ||
              !draft.pairingCode.trim() ||
              working != null
            }
            onClick={() => void claimPairing()}
          >
            {working === 'claim:draft' ? <LoaderCircle className="spin" size={14} /> : null}
            Pair and add
          </button>
          {draft.id ? (
            <button type="button" className="chip-button subtle" onClick={() => setDraft(EMPTY_REMOTE_PROFILE)}>
              Cancel edit
            </button>
          ) : null}
        </div>
        {notice ? <p className="model-help">{notice}</p> : null}
      </form>

      <div className="settings-group">
        <div className="settings-group-head">
          <div>
            <div className="section-label">Pair clients with {current?.name ?? 'current daemon'}</div>
            <p className="model-help">
              Create a five-minute, one-time code. The code is shown only here; send it to the
              person adding this daemon through a separate trusted channel.
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

        {trustError ? <div className="error-banner"><span>{trustError}</span></div> : null}

        <form className="settings-grid" onSubmit={(event) => void createPairing(event)}>
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
            {(['inference', 'management', 'agent'] as ClientScope[]).map((scope) => (
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
              disabled={!pairingClientName.trim() || pairingScopes.length === 0 || working != null}
            >
              {working === 'pair:create' ? <LoaderCircle className="spin" size={14} /> : <KeyRound size={14} />}
              Create pairing
            </button>
          </div>
        </form>

        {createdPairing ? (
          <article className="runtime-card active">
            <div className="runtime-card-info">
              <strong>One-time pairing for {createdPairing.pairing.client_name}</strong>
              <span>Pairing id: <code>{createdPairing.pairing.id}</code></span>
              <span>Pairing code: <code>{createdPairing.code}</code></span>
              <span>Expires {new Date(createdPairing.pairing.expires_at * 1_000).toLocaleString()}</span>
            </div>
            <div className="library-card-actions">
              <button type="button" className="chip-button" onClick={() => void copyPairing()}>
                {pairingCopied ? <Check size={13} /> : <Copy size={13} />}
                {pairingCopied ? 'Copied' : 'Copy id and code'}
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

import {
  Box,
  Check,
  ChevronDown,
  ChevronRight,
  Cpu,
  Download,
  Hammer,
  HardDrive,
  LoaderCircle,
  Search,
  Settings2,
  ShieldAlert,
  Trash2,
  X
} from 'lucide-react'
import { type FormEvent, useEffect, useRef, useState } from 'react'
import {
  activateRuntime,
  buildRuntime,
  deleteModel,
  deleteRuntime,
  downloadModel,
  ensureLlamaEngine,
  formatBytes,
  type HardwareInfo,
  type HubFile,
  type LocalModel,
  listHubFiles,
  listRuntimes,
  type ProgressEvent,
  type RuntimeEntry,
  type RuntimeSettings,
  saveRuntimeSettings,
  searchHub
} from '../api'
import type { HubModel } from '../types'

export type ManageSection = 'library' | 'discover' | 'runtimes' | 'engine'

type ManagePanelProps = {
  section: ManageSection
  onSectionChange: (section: ManageSection) => void
  onClose: () => void
  models: LocalModel[]
  modelsLoading: boolean
  refreshModels: () => Promise<void>
  selectedModel: string
  onSelectModel: (modelId: string) => void
  settings: RuntimeSettings | null
  onSettingsSaved: (settings: RuntimeSettings) => void
  hardware: HardwareInfo | null
}

const SECTIONS: Array<{ id: ManageSection; label: string; icon: React.JSX.Element }> = [
  { id: 'library', label: 'Model library', icon: <Box size={15} /> },
  { id: 'discover', label: 'Download models', icon: <Download size={15} /> },
  { id: 'runtimes', label: 'Runtimes', icon: <Cpu size={15} /> },
  { id: 'engine', label: 'Engine configuration', icon: <Settings2 size={15} /> }
]

function progressLabel(event: ProgressEvent | null): string {
  if (!event) return 'Starting…'
  if (event.message) return event.message
  if (event.phase === 'download' && event.bytes != null) {
    const total = event.total != null ? ` / ${formatBytes(event.total)}` : ''
    const percent = event.percent != null ? ` · ${Math.round(event.percent)}%` : ''
    return `Downloading ${formatBytes(event.bytes)}${total}${percent}`
  }
  return event.phase
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

/**
 * Full management surface: model library and downloads, runtime installs and
 * source builds, and engine launch configuration. Everything here is
 * management; day-to-day usage (model choice, sampling) lives in the topbar.
 */
export function ManagePanel(props: ManagePanelProps): React.JSX.Element {
  const [error, setError] = useState<string | null>(null)

  return (
    <div className="drawer-backdrop" onMouseDown={props.onClose}>
      <aside className="manage-panel" onMouseDown={(event) => event.stopPropagation()}>
        <nav className="manage-nav">
          <div className="manage-nav-title">Manage</div>
          {SECTIONS.map((entry) => (
            <button
              key={entry.id}
              className={props.section === entry.id ? 'active' : ''}
              onClick={() => {
                setError(null)
                props.onSectionChange(entry.id)
              }}
            >
              {entry.icon}
              {entry.label}
            </button>
          ))}
          <div className="manage-nav-spacer" />
          <button className="manage-close" onClick={props.onClose}>
            <X size={15} />
            Close
          </button>
        </nav>
        <div className="manage-content">
          {error && (
            <div className="error-banner">
              <span>{error}</span>
              <button onClick={() => setError(null)}>
                <X size={14} />
              </button>
            </div>
          )}
          {props.section === 'library' && <LibrarySection {...props} onError={setError} />}
          {props.section === 'discover' && <DiscoverSection {...props} onError={setError} />}
          {props.section === 'runtimes' && <RuntimesSection {...props} onError={setError} />}
          {props.section === 'engine' && <EngineSection {...props} onError={setError} />}
        </div>
      </aside>
    </div>
  )
}

type SectionProps = ManagePanelProps & { onError: (message: string | null) => void }

function LibrarySection(props: SectionProps): React.JSX.Element {
  const [confirming, setConfirming] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)

  async function removeModel(modelId: string): Promise<void> {
    setDeleting(modelId)
    props.onError(null)
    try {
      await deleteModel(modelId)
      await props.refreshModels()
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setDeleting(null)
      setConfirming(null)
    }
  }

  return (
    <section>
      <header className="manage-heading">
        <h2>Model library</h2>
        <p>GGUF weights installed on this device. Deleting frees disk space immediately.</p>
      </header>
      {props.modelsLoading && (
        <div className="manage-placeholder">
          <LoaderCircle className="spin" size={18} />
          Loading installed models…
        </div>
      )}
      {!props.modelsLoading && props.models.length === 0 && (
        <div className="manage-placeholder">
          <Box size={18} />
          <span>
            Nothing installed yet.{' '}
            <button className="inline-link" onClick={() => props.onSectionChange('discover')}>
              Download a model
            </button>{' '}
            to get started.
          </span>
        </div>
      )}
      <div className="library-list">
        {props.models.map((model) => {
          const key = model.id.startsWith('gguf:') ? model.id.slice('gguf:'.length) : model.id
          const parts = key.split('/')
          const name = parts.at(-1) ?? key
          const repo = parts.slice(0, -1).join('/')
          const caps = model.capabilities
          const isSelected = model.id === props.selectedModel
          return (
            <article className="library-card" key={model.id}>
              <div className="library-card-info">
                <strong>{name}</strong>
                <span>
                  {repo}
                  {model.size_bytes != null ? ` · ${formatBytes(model.size_bytes)}` : ''}
                </span>
                <div className="library-caps">
                  {caps?.input_modalities.includes('image') && <span>vision</span>}
                  {caps?.tools && <span>tools</span>}
                  {caps?.reasoning && <span>reasoning</span>}
                </div>
              </div>
              <div className="library-card-actions">
                <button
                  className={isSelected ? 'chip-button selected' : 'chip-button'}
                  disabled={isSelected}
                  onClick={() => props.onSelectModel(model.id)}
                >
                  {isSelected ? (
                    <>
                      <Check size={13} /> In use
                    </>
                  ) : (
                    'Use'
                  )}
                </button>
                {confirming === model.id ? (
                  <button
                    className="chip-button danger"
                    disabled={deleting === model.id}
                    onClick={() => void removeModel(model.id)}
                  >
                    {deleting === model.id ? (
                      <LoaderCircle className="spin" size={13} />
                    ) : (
                      <Trash2 size={13} />
                    )}
                    Confirm delete
                  </button>
                ) : (
                  <button
                    className="chip-button subtle"
                    title="Delete this model from disk"
                    onClick={() => setConfirming(model.id)}
                  >
                    <Trash2 size={13} />
                  </button>
                )}
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}

function DiscoverSection(props: SectionProps): React.JSX.Element {
  const [query, setQuery] = useState('Qwen')
  const [results, setResults] = useState<HubModel[]>([])
  const [searching, setSearching] = useState(false)
  const [expandedRepo, setExpandedRepo] = useState<string | null>(null)
  const [repoFiles, setRepoFiles] = useState<Record<string, HubFile[]>>({})
  const [preferredFiles, setPreferredFiles] = useState<Record<string, string | null>>({})
  const [loadingFilesFor, setLoadingFilesFor] = useState<string | null>(null)
  const [downloadProgress, setDownloadProgress] = useState<{
    key: string
    event: ProgressEvent | null
  } | null>(null)
  const [enginePhase, setEnginePhase] = useState<string | null>(null)

  async function findModels(event?: FormEvent): Promise<void> {
    event?.preventDefault()
    setSearching(true)
    props.onError(null)
    setExpandedRepo(null)
    try {
      setResults(await searchHub(query.trim(), 'llama.cpp'))
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setSearching(false)
    }
  }

  async function toggleRepo(model: HubModel): Promise<void> {
    if (expandedRepo === model.id) {
      setExpandedRepo(null)
      return
    }
    setExpandedRepo(model.id)
    if (repoFiles[model.id]) return
    setLoadingFilesFor(model.id)
    props.onError(null)
    try {
      const files = await listHubFiles(model.id)
      setRepoFiles((current) => ({ ...current, [model.id]: files.data }))
      setPreferredFiles((current) => ({ ...current, [model.id]: files.preferred_filename }))
    } catch (cause) {
      props.onError(errorText(cause))
      setExpandedRepo(null)
    } finally {
      setLoadingFilesFor(null)
    }
  }

  async function downloadQuant(repoId: string, path: string): Promise<void> {
    const key = `${repoId}::${path}`
    setDownloadProgress({ key, event: null })
    props.onError(null)
    try {
      await downloadModel(repoId, path, (event) => setDownloadProgress({ key, event }))
      await props.refreshModels()
      // Make sure a runtime exists so the model is immediately usable.
      setEnginePhase('Checking the inference runtime…')
      try {
        await ensureLlamaEngine((event) => setEnginePhase(progressLabel(event)))
      } catch (cause) {
        props.onError(
          `Model downloaded, but no runtime is installed yet: ${errorText(cause)}. ` +
            'Open the Runtimes section to install or build one.'
        )
      } finally {
        setEnginePhase(null)
      }
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setDownloadProgress(null)
    }
  }

  return (
    <section>
      <header className="manage-heading">
        <h2>Download models</h2>
        <p>Search Hugging Face for GGUF weights compatible with llama.cpp.</p>
      </header>
      <form className="model-search" onSubmit={(event) => void findModels(event)}>
        <label>
          <Search size={16} />
          <input
            aria-label="Search Hugging Face"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Model name or author"
          />
        </label>
        <button type="submit" disabled={searching || !query.trim()}>
          {searching ? <LoaderCircle className="spin" size={15} /> : 'Search'}
        </button>
      </form>
      {enginePhase && (
        <div className="engine-phase-note">
          <LoaderCircle className="spin" size={14} />
          {enginePhase}
        </div>
      )}
      <div className="model-results">
        {results.map((model) => {
          const expanded = expandedRepo === model.id
          const files = (repoFiles[model.id] ?? []).filter((file) =>
            file.path.toLowerCase().endsWith('.gguf')
          )
          const preferred = preferredFiles[model.id]
          return (
            <article className="model-card expandable" key={model.id}>
              <div className="model-card-main">
                <div>
                  <strong>{model.id.split('/').at(-1)}</strong>
                  <span>{model.author}</span>
                  <div className="model-badges">
                    {model.preferred_quantizer && <span className="unsloth">Unsloth preferred</span>}
                    {model.gated && (
                      <span>
                        <ShieldAlert size={11} /> Gated
                      </span>
                    )}
                    <span>{model.downloads.toLocaleString()} downloads</span>
                  </div>
                </div>
                <button
                  type="button"
                  disabled={loadingFilesFor === model.id}
                  onClick={() => void toggleRepo(model)}
                >
                  {loadingFilesFor === model.id ? (
                    <LoaderCircle className="spin" size={15} />
                  ) : (
                    <>
                      {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                      {expanded ? 'Hide quants' : 'Choose quant'}
                    </>
                  )}
                </button>
              </div>
              {expanded && (
                <div className="quant-list">
                  {files.length === 0 && (
                    <p className="empty-models-inline">No GGUF files found in this repo.</p>
                  )}
                  {files.map((file) => {
                    const key = `${model.id}::${file.path}`
                    const active = downloadProgress?.key === key
                    const basename = file.path.split('/').at(-1) ?? file.path
                    const isProjector = basename.toLowerCase().includes('mmproj')
                    const isPreferred =
                      preferred != null &&
                      (file.path === preferred || file.path.endsWith(`/${preferred}`))
                    return (
                      <div className="quant-row" key={file.path}>
                        <div>
                          <strong>
                            {basename}
                            {isProjector ? ' · multimodal projector' : ''}
                            {isPreferred ? ' · preferred' : ''}
                          </strong>
                          <span>
                            {file.path}
                            {file.size != null ? ` · ${formatBytes(file.size)}` : ''}
                          </span>
                          {active && (
                            <div className="progress-block compact">
                              <div className="progress-track">
                                <div
                                  className="progress-fill"
                                  style={{
                                    width: `${Math.min(100, downloadProgress.event?.percent ?? 8)}%`
                                  }}
                                />
                              </div>
                              <span>{progressLabel(downloadProgress.event)}</span>
                            </div>
                          )}
                        </div>
                        <button
                          type="button"
                          disabled={downloadProgress != null}
                          onClick={() => void downloadQuant(model.id, file.path)}
                        >
                          {active ? (
                            <LoaderCircle className="spin" size={15} />
                          ) : isProjector ? (
                            'Add capability'
                          ) : (
                            'Download'
                          )}
                        </button>
                      </div>
                    )
                  })}
                </div>
              )}
            </article>
          )
        })}
        {!searching && results.length === 0 && (
          <div className="empty-models">
            <Search size={24} />
            <p>Search the Hub, then pick a quant to download into your library.</p>
          </div>
        )}
      </div>
    </section>
  )
}

function RuntimesSection(props: SectionProps): React.JSX.Element {
  const [runtimes, setRuntimes] = useState<RuntimeEntry[] | null>(null)
  const [busyRuntime, setBusyRuntime] = useState<string | null>(null)
  const [confirming, setConfirming] = useState<string | null>(null)
  const [installing, setInstalling] = useState(false)
  const [installPhase, setInstallPhase] = useState<string | null>(null)
  const [savingTarget, setSavingTarget] = useState(false)

  // Build-from-source form.
  const [buildOpen, setBuildOpen] = useState(false)
  const [repository, setRepository] = useState('https://github.com/ggml-org/llama.cpp')
  const [revision, setRevision] = useState('master')
  const [buildTarget, setBuildTarget] = useState('cpu')
  const [building, setBuilding] = useState(false)
  const [buildLog, setBuildLog] = useState<string[]>([])
  const [buildWarning, setBuildWarning] = useState<string | null>(null)
  const logRef = useRef<HTMLPreElement>(null)

  async function refreshRuntimes(): Promise<void> {
    try {
      const response = await listRuntimes()
      setRuntimes(response.data)
    } catch (cause) {
      props.onError(errorText(cause))
      setRuntimes([])
    }
  }

  useEffect(() => {
    void refreshRuntimes()
  }, [])

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight })
  }, [buildLog])

  async function activate(id: string): Promise<void> {
    setBusyRuntime(id)
    props.onError(null)
    try {
      await activateRuntime(id)
      await refreshRuntimes()
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setBusyRuntime(null)
    }
  }

  async function remove(id: string): Promise<void> {
    setBusyRuntime(id)
    props.onError(null)
    try {
      await deleteRuntime(id)
      await refreshRuntimes()
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setBusyRuntime(null)
      setConfirming(null)
    }
  }

  async function selectTarget(target: string): Promise<void> {
    if (!props.settings) return
    setSavingTarget(true)
    props.onError(null)
    try {
      const saved = await saveRuntimeSettings({
        ...props.settings,
        target: target as RuntimeSettings['target']
      })
      props.onSettingsSaved(saved)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setSavingTarget(false)
    }
  }

  async function installLatest(): Promise<void> {
    setInstalling(true)
    setInstallPhase(null)
    props.onError(null)
    try {
      await ensureLlamaEngine((event) => setInstallPhase(progressLabel(event)))
      await refreshRuntimes()
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setInstalling(false)
      setInstallPhase(null)
    }
  }

  async function runBuild(event: FormEvent): Promise<void> {
    event.preventDefault()
    setBuilding(true)
    setBuildLog([])
    setBuildWarning(null)
    props.onError(null)
    try {
      await buildRuntime(repository.trim(), revision.trim(), buildTarget, (progress) => {
        if (progress.phase === 'warning' && progress.message) {
          setBuildWarning(progress.message)
          return
        }
        if (progress.message) {
          setBuildLog((current) => [...current.slice(-400), progress.message ?? ''])
        }
      })
      setBuildLog((current) => [...current, 'Build complete. Activate it below to use it.'])
      await refreshRuntimes()
    } catch (cause) {
      props.onError(`Build failed: ${errorText(cause)}`)
    } finally {
      setBuilding(false)
    }
  }

  const targets = props.hardware?.targets ?? []

  return (
    <section>
      <header className="manage-heading">
        <h2>Runtimes</h2>
        <p>
          Install, build, and choose the llama-server binary that powers local inference. The
          active runtime is used for every generation.
        </p>
      </header>

      <div className="hardware-card">
        <HardDrive size={18} />
        <div>
          <strong>
            {props.hardware?.gpu ?? `${props.hardware?.logical_cpus ?? '—'} CPU threads`}
          </strong>
          <span>
            {props.hardware?.os} {props.hardware?.architecture}
            {props.hardware?.memory_bytes
              ? ` · ${formatBytes(props.hardware.memory_bytes)} RAM`
              : ''}
          </span>
        </div>
      </div>

      <div className="settings-group">
        <div className="section-label">
          Acceleration target {savingTarget && <LoaderCircle className="spin" size={12} />}
        </div>
        <div className="target-grid">
          {targets.map((target) => (
            <button
              key={target.id}
              disabled={!target.available || savingTarget}
              className={props.settings?.target === target.id ? 'active' : ''}
              title={target.detail}
              onClick={() => void selectTarget(target.id)}
            >
              <strong>{target.name}</strong>
              <span>
                {target.recommended
                  ? 'Recommended'
                  : target.available
                    ? 'Available'
                    : 'Not detected'}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="settings-group">
        <div className="section-label">Installed runtimes</div>
        {runtimes == null && (
          <div className="manage-placeholder">
            <LoaderCircle className="spin" size={16} />
            Scanning for installed runtimes…
          </div>
        )}
        {runtimes != null && runtimes.length === 0 && (
          <div className="manage-placeholder">
            <Cpu size={16} />
            No runtime installed yet. Install a release below or build one from source.
          </div>
        )}
        <div className="runtime-list">
          {(runtimes ?? []).map((runtime) => (
            <article
              className={runtime.active ? 'runtime-card active' : 'runtime-card'}
              key={runtime.id}
            >
              <div className="runtime-card-info">
                <strong>
                  {runtime.label}
                  {runtime.active && <span className="active-badge">Active</span>}
                </strong>
                <span>
                  {[runtime.kind, runtime.target, runtime.version].filter(Boolean).join(' · ')}
                </span>
                <code>{runtime.path}</code>
              </div>
              <div className="library-card-actions">
                {!runtime.active && (
                  <button
                    className="chip-button"
                    disabled={busyRuntime != null}
                    onClick={() => void activate(runtime.id)}
                  >
                    {busyRuntime === runtime.id ? (
                      <LoaderCircle className="spin" size={13} />
                    ) : (
                      'Activate'
                    )}
                  </button>
                )}
                {runtime.deletable &&
                  (confirming === runtime.id ? (
                    <button
                      className="chip-button danger"
                      disabled={busyRuntime != null}
                      onClick={() => void remove(runtime.id)}
                    >
                      {busyRuntime === runtime.id ? (
                        <LoaderCircle className="spin" size={13} />
                      ) : (
                        <Trash2 size={13} />
                      )}
                      Confirm
                    </button>
                  ) : (
                    <button
                      className="chip-button subtle"
                      title="Remove this runtime installation"
                      onClick={() => setConfirming(runtime.id)}
                    >
                      <Trash2 size={13} />
                    </button>
                  ))}
              </div>
            </article>
          ))}
        </div>
        <button className="secondary-action" disabled={installing} onClick={() => void installLatest()}>
          {installing ? <LoaderCircle className="spin" size={15} /> : <Download size={15} />}
          {installing ? (installPhase ?? 'Installing…') : 'Install latest release'}
        </button>
      </div>

      <div className="settings-group">
        <button className="build-toggle" onClick={() => setBuildOpen((open) => !open)}>
          {buildOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <Hammer size={14} />
          Build from source
        </button>
        {buildOpen && (
          <form className="build-form" onSubmit={(event) => void runBuild(event)}>
            <label>
              <span>Repository</span>
              <input
                value={repository}
                onChange={(event) => setRepository(event.target.value)}
                placeholder="https://github.com/ggml-org/llama.cpp"
              />
            </label>
            <div className="build-form-row">
              <label>
                <span>Branch, tag, or commit</span>
                <input value={revision} onChange={(event) => setRevision(event.target.value)} />
              </label>
              <label>
                <span>Target</span>
                <select
                  value={buildTarget}
                  onChange={(event) => setBuildTarget(event.target.value)}
                >
                  <option value="cpu">CPU</option>
                  <option value="cuda">CUDA</option>
                  <option value="vulkan">Vulkan</option>
                  <option value="rocm">ROCm</option>
                </select>
              </label>
            </div>
            {buildWarning && (
              <div className="build-warning">
                <ShieldAlert size={14} />
                {buildWarning}
              </div>
            )}
            <p className="model-help">
              Builds run locally with git and cmake into an isolated prefix. Non-upstream forks
              execute untrusted native code — only build sources you trust.
            </p>
            <button
              className="primary-action"
              type="submit"
              disabled={building || !repository.trim() || !revision.trim()}
            >
              {building ? <LoaderCircle className="spin" size={15} /> : <Hammer size={15} />}
              {building ? 'Building…' : 'Start build'}
            </button>
            {(building || buildLog.length > 0) && (
              <pre className="build-log" ref={logRef}>
                {buildLog.join('\n')}
              </pre>
            )}
          </form>
        )}
      </div>
    </section>
  )
}

function EngineSection(props: SectionProps): React.JSX.Element {
  const [draft, setDraft] = useState<RuntimeSettings | null>(props.settings)
  const [saving, setSaving] = useState(false)
  useEffect(() => setDraft(props.settings), [props.settings])

  const dirty =
    draft != null &&
    props.settings != null &&
    JSON.stringify(draft) !== JSON.stringify(props.settings)

  async function apply(): Promise<void> {
    if (!draft) return
    setSaving(true)
    props.onError(null)
    try {
      const saved = await saveRuntimeSettings(draft)
      props.onSettingsSaved(saved)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setSaving(false)
    }
  }

  if (!draft) {
    return (
      <section>
        <header className="manage-heading">
          <h2>Engine configuration</h2>
        </header>
        <div className="manage-placeholder">
          <LoaderCircle className="spin" size={16} />
          Waiting for the daemon…
        </div>
      </section>
    )
  }

  return (
    <section>
      <header className="manage-heading">
        <h2>Engine configuration</h2>
        <p>
          Launch parameters for llama-server. Applying restarts the model server on the next
          generation. Sampling defaults live in the inference menu next to the model picker.
        </p>
      </header>
      <div className="settings-grid">
        <label>
          <span>Context size</span>
          <input
            type="number"
            min={512}
            step={512}
            value={draft.context_size}
            onChange={(event) => setDraft({ ...draft, context_size: Number(event.target.value) })}
          />
        </label>
        <label>
          <span>Batch size</span>
          <input
            type="number"
            min={32}
            step={32}
            value={draft.batch_size}
            onChange={(event) => setDraft({ ...draft, batch_size: Number(event.target.value) })}
          />
        </label>
        <label>
          <span>GPU layers</span>
          <input
            type="number"
            min={-1}
            value={draft.gpu_layers}
            disabled={draft.target === 'cpu'}
            onChange={(event) => setDraft({ ...draft, gpu_layers: Number(event.target.value) })}
          />
        </label>
        <label>
          <span>CPU threads</span>
          <input
            type="number"
            min={1}
            placeholder="Auto"
            value={draft.threads ?? ''}
            onChange={(event) =>
              setDraft({
                ...draft,
                threads: event.target.value ? Number(event.target.value) : null
              })
            }
          />
        </label>
        <label>
          <span>KV cache K</span>
          <select
            value={draft.kv_cache_type_k}
            onChange={(event) => setDraft({ ...draft, kv_cache_type_k: event.target.value })}
          >
            {['f16', 'q8_0', 'q5_1', 'q4_0'].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label>
          <span>KV cache V</span>
          <select
            value={draft.kv_cache_type_v}
            onChange={(event) => setDraft({ ...draft, kv_cache_type_v: event.target.value })}
          >
            {['f16', 'q8_0', 'q5_1', 'q4_0'].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="toggle-list">
        {(
          [
            ['flash_attention', 'Flash attention', 'Faster attention with supported backends'],
            ['jinja', 'Jinja templates', 'Required for modern chat templates and tools']
          ] as const
        ).map(([key, title, detail]) => (
          <label key={key}>
            <div>
              <strong>{title}</strong>
              <span>{detail}</span>
            </div>
            <input
              type="checkbox"
              checked={Boolean(draft[key])}
              onChange={(event) => setDraft({ ...draft, [key]: event.target.checked })}
            />
          </label>
        ))}
      </div>
      <div className="runtime-actions">
        <button className="primary-action" disabled={saving || !dirty} onClick={() => void apply()}>
          {saving ? <LoaderCircle className="spin" size={15} /> : 'Apply & restart'}
        </button>
      </div>
    </section>
  )
}

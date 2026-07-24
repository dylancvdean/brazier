import {
  Box,
  Check,
  ChevronDown,
  ChevronRight,
  Cpu,
  Download,
  FolderOpen,
  Hammer,
  HardDrive,
  LoaderCircle,
  Plug,
  Search,
  Settings2,
  ShieldAlert,
  Trash2,
  X
} from 'lucide-react'
import { type FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import {
  activateRuntime,
  buildRuntime,
  cancelBuild,
  cancelDownloadJob,
  deleteModel,
  createMcpServer,
  deleteMcpServer,
  deleteRuntime,
  downloadModel,
  downloadMlxModel,
  downloadStreamingAsrModel,
  ensureLlamaEngine,
  fetchManagedLlamaStatus,
  fetchModelTrust,
  formatBytes,
  clearHuggingFaceToken,
  huggingFaceTokenStatus,
  listDownloadJobs,
  type DownloadJob,
  type HardwareInfo,
  type ManagedLlamaTargetStatus,
  type HubFile,
  type LocalModel,
  type ModelTrust,
  listHubFiles,
  listMcpServers,
  listRuntimes,
  listTools,
  type BundledTool,
  type McpServer,
  modelLibraryPathSuggestions,
  type ModelLibraryPathSuggestion,
  type ProgressEvent,
  type RuntimeEntry,
  type RuntimeSettings,
  type RuntimeTarget,
  saveRuntimeSettings,
  searchHub,
  setHuggingFaceToken,
  queueModelDownload,
  refreshMcpServer,
  updateMcpServer
} from '../api'
import {
  engineBadgeClass,
  engineLabel,
  modelEngine,
  modelLibraryKey,
  runtimeNoticeForModel,
  runtimesForModel
} from '../model-utils'
import type { HubModel } from '../types'

type DiscoverEngine = 'llama.cpp' | 'mlx-lm' | 'mlx-vlm' | 'whisper.cpp' | 'streaming-asr'
type BuildEngine = 'llama.cpp' | 'mlx-lm' | 'mlx-vlm' | 'whisper.cpp' | 'streaming-asr'

const DISCOVER_ENGINE_HELP: Record<DiscoverEngine, string> = {
  'llama.cpp': 'GGUF weights for llama.cpp on CPU, CUDA, Metal, or Vulkan.',
  'mlx-lm': 'Text-only MLX models for Apple Silicon (chat, tools, reasoning).',
  'mlx-vlm': 'Vision MLX models for Apple Silicon (image + text input).',
  'whisper.cpp': 'Whisper speech-to-text weights (ggml/gguf) for local audio transcription.',
  'streaming-asr':
    'Nemotron ASR Streaming snapshots for low-latency chunked transcription (Transformers).'
}

const DISCOVER_ENGINE_LABELS: Record<DiscoverEngine, string> = {
  'llama.cpp': 'GGUF · llama.cpp',
  'mlx-lm': 'MLX · text',
  'mlx-vlm': 'MLX · vision',
  'whisper.cpp': 'ASR · whisper.cpp',
  'streaming-asr': 'ASR · streaming'
}

const BUILD_ENGINE_DEFAULTS: Record<
  BuildEngine,
  { repository: string; revision: string }
> = {
  'llama.cpp': {
    repository: 'https://github.com/ggml-org/llama.cpp',
    revision: 'master'
  },
  'mlx-lm': {
    repository: 'https://github.com/ml-explore/mlx-lm',
    revision: 'main'
  },
  'mlx-vlm': {
    repository: 'https://github.com/Blaizzy/mlx-vlm',
    revision: 'main'
  },
  'whisper.cpp': {
    repository: 'https://github.com/ggml-org/whisper.cpp',
    revision: 'master'
  },
  'streaming-asr': {
    repository: 'https://github.com/huggingface/transformers',
    revision: 'bundled'
  }
}

function defaultDiscoverEngine(hardware: HardwareInfo | null): DiscoverEngine {
  if (hardware?.os === 'macos' && hardware.architecture === 'aarch64') {
    return 'mlx-lm'
  }
  return 'llama.cpp'
}

export type ManageSection = 'library' | 'discover' | 'runtimes' | 'engine' | 'mcp'

type ManagePanelProps = {
  section: ManageSection
  onSectionChange: (section: ManageSection) => void
  onClose: () => void
  models: LocalModel[]
  modelsLoading: boolean
  refreshModels: () => Promise<void>
  initialRuntimes?: RuntimeEntry[] | null
  selectedModel: string
  onSelectModel: (modelId: string) => void
  modelBindings?: Record<string, string>
  onSetModelBinding?: (modelId: string, runtimeId: string | null) => void
  settings: RuntimeSettings | null
  onSettingsSaved: (settings: RuntimeSettings) => void
  hardware: HardwareInfo | null
  pendingBuild?: { engine: BuildEngine; repository: string } | null
  onPendingBuildConsumed?: () => void
}

const SECTIONS: Array<{ id: ManageSection; label: string; icon: React.JSX.Element }> = [
  { id: 'library', label: 'Model library', icon: <Box size={15} /> },
  { id: 'discover', label: 'Download models', icon: <Download size={15} /> },
  { id: 'runtimes', label: 'Runtimes', icon: <Cpu size={15} /> },
  { id: 'mcp', label: 'MCP servers', icon: <Plug size={15} /> },
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
  if (event.phase === 'discover') return 'Checking for an installed runtime…'
  if (event.phase === 'resolve') return 'Resolving the latest release…'
  if (event.phase === 'extract') return 'Extracting the release archive…'
  if (event.phase === 'hash') return 'Verifying download integrity…'
  if (event.phase === 'build') return 'Building from source…'
  if (event.phase === 'install') return 'Installing built artifacts…'
  if (event.phase === 'log') return 'Compiling…'
  if (event.phase === 'skip') return 'Already present on disk'
  if (event.phase === 'start') return 'Starting download…'
  if (event.phase === 'warning') return event.message ?? 'Notice'
  return event.phase
}

type BuildStep = { current: number; total: number; label: string }

type JobProgressState = {
  headline: string
  step: BuildStep | null
  percent: number | null
  phase: string
  logLines: string[]
  hints: string[]
}

function emptyJobProgress(headline: string): JobProgressState {
  return { headline, step: null, percent: null, phase: 'starting', logLines: [], hints: [] }
}

function stepFromEvent(event: ProgressEvent): BuildStep | null {
  const result = event.result as { step?: number; total?: number; label?: string } | undefined
  if (result?.step && result.total && result.label) {
    return { current: result.step, total: result.total, label: result.label }
  }
  const match = event.message?.match(/^\[(\d+)\/(\d+)\]\s*(.+)$/)
  if (!match) return null
  return {
    current: Number(match[1]),
    total: Number(match[2]),
    label: match[3]
  }
}

function applyJobProgress(current: JobProgressState, event: ProgressEvent): JobProgressState {
  const next = { ...current, phase: event.phase }
  if (event.phase === 'build') {
    next.headline = event.message?.includes('started')
      ? 'Source build started'
      : (stepFromEvent(event)?.label ?? event.message ?? 'Building from source')
    const step = stepFromEvent(event)
    if (step) next.step = step
    if (event.percent != null) next.percent = event.percent
  } else if (event.phase === 'install') {
    next.headline = event.message ?? 'Installing built server'
    next.percent = 95
  } else if (event.phase === 'download') {
    next.headline = progressLabel(event)
    next.percent = event.percent ?? next.percent
  } else if (event.phase === 'discover' || event.phase === 'resolve' || event.phase === 'extract') {
    next.headline = progressLabel(event)
  } else if (event.phase === 'log' && event.message) {
    next.headline = 'Compiling…'
    next.logLines = [...current.logLines.slice(-400), event.message]
  } else if (event.message && event.phase !== 'warning') {
    next.logLines = [...current.logLines.slice(-400), event.message]
  }
  return next
}

function JobProgressPanel({
  progress,
  active
}: {
  progress: JobProgressState
  active: boolean
}): React.JSX.Element | null {
  if (!active && progress.logLines.length === 0) return null
  return (
    <div className="job-progress-panel">
      <div className="job-progress-head">
        {active ? <LoaderCircle className="spin" size={16} /> : <Check size={16} />}
        <div>
          <strong>{progress.headline}</strong>
          {progress.step && (
            <span>
              Step {progress.step.current} of {progress.step.total}: {progress.step.label}
            </span>
          )}
          {!progress.step && active && progress.phase === 'log' && (
            <span>Reading compiler output…</span>
          )}
        </div>
      </div>
      {progress.percent != null && (
        <div className="progress-track">
          <div
            className="progress-fill"
            style={{ width: `${Math.min(100, Math.max(4, progress.percent))}%` }}
          />
        </div>
      )}
      {progress.hints.length > 0 && (
        <ul className="build-hints">
          {progress.hints.map((hint) => (
            <li key={hint}>{hint}</li>
          ))}
        </ul>
      )}
      {progress.logLines.length > 0 && (
        <pre className="build-log compact">{progress.logLines.slice(-12).join('\n')}</pre>
      )}
    </div>
  )
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
          {props.section === 'mcp' && <McpSection {...props} onError={setError} />}
          {props.section === 'engine' && <EngineSection {...props} onError={setError} />}
        </div>
      </aside>
    </div>
  )
}

type SectionProps = ManagePanelProps & { onError: (message: string | null) => void }

function modelCountSummary(ggufCount: number, mlxCount: number): string {
  const parts: string[] = []
  if (ggufCount > 0) {
    parts.push(`${ggufCount} GGUF file${ggufCount === 1 ? '' : 's'}`)
  }
  if (mlxCount > 0) {
    parts.push(`${mlxCount} MLX model${mlxCount === 1 ? '' : 's'}`)
  }
  if (parts.length === 0) {
    return 'No models found'
  }
  return parts.join(' · ')
}

function isExternalModel(model: LocalModel): boolean {
  return Boolean(model.read_only) || model.id.includes('-ext:')
}

function LibrarySection(props: SectionProps): React.JSX.Element {
  const [confirming, setConfirming] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [browseOpen, setBrowseOpen] = useState(false)
  const [suggestions, setSuggestions] = useState<ModelLibraryPathSuggestion[]>([])
  const [configuredPaths, setConfiguredPaths] = useState<string[]>([])
  const [loadingSuggestions, setLoadingSuggestions] = useState(false)
  const [addingPath, setAddingPath] = useState<string | null>(null)
  const [runtimes, setRuntimes] = useState<RuntimeEntry[] | null>(props.initialRuntimes ?? null)

  useEffect(() => {
    void listRuntimes()
      .then((response) => setRuntimes(response.data))
      .catch(() => {
        if (props.initialRuntimes?.length) setRuntimes(props.initialRuntimes)
      })
  }, [props.initialRuntimes])

  useEffect(() => {
    setConfiguredPaths(props.settings?.extra_model_library_paths ?? [])
  }, [props.settings?.extra_model_library_paths])

  async function loadSuggestions(): Promise<void> {
    setLoadingSuggestions(true)
    props.onError(null)
    try {
      const response = await modelLibraryPathSuggestions()
      setSuggestions(response.suggestions)
      setConfiguredPaths(response.configured)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setLoadingSuggestions(false)
    }
  }

  async function openBrowse(): Promise<void> {
    setBrowseOpen(true)
    await loadSuggestions()
  }

  async function addLibraryPath(path: string): Promise<void> {
    if (!props.settings) return
    setAddingPath(path)
    props.onError(null)
    try {
      const nextPaths = [...(props.settings.extra_model_library_paths ?? [])]
      if (!nextPaths.includes(path)) {
        nextPaths.push(path)
      }
      const saved = await saveRuntimeSettings({
        ...props.settings,
        extra_model_library_paths: nextPaths
      })
      props.onSettingsSaved(saved)
      setConfiguredPaths(saved.extra_model_library_paths ?? [])
      setSuggestions((current) =>
        current.map((entry) =>
          entry.path === path ? { ...entry, configured: true } : entry
        )
      )
      await props.refreshModels()
      await loadSuggestions()
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setAddingPath(null)
    }
  }

  async function chooseCustomFolder(): Promise<void> {
    props.onError(null)
    try {
      const path = await window.brazier.selectDirectory()
      if (!path) return
      setBrowseOpen(true)
      await addLibraryPath(path)
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  const visibleSuggestions = suggestions.filter((entry) => entry.exists)

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
        <p>
          Models downloaded by Brazier and models found in folders you add (GGUF and MLX from
          LM Studio, Hugging Face cache, and others). External folders are read-only here.
        </p>
      </header>

      <div className="settings-group">
        <div className="section-label">Other model folders</div>
        <p className="model-help">
          Add directories that already contain GGUF or MLX weights. Brazier scans them alongside
          your local downloads.
        </p>
        {configuredPaths.length > 0 && (
          <div className="library-path-list">
            {configuredPaths.map((path) => (
              <div className="library-path-row" key={path}>
                <code>{path}</code>
              </div>
            ))}
          </div>
        )}
        <div className="library-path-actions">
          <button className="secondary-action" onClick={() => void openBrowse()}>
            <Search size={15} />
            Browse common folders…
          </button>
          <button className="secondary-action" onClick={() => void chooseCustomFolder()}>
            <FolderOpen size={15} />
            Choose folder…
          </button>
        </div>
        {browseOpen && (
          <div className="library-suggestions">
            {loadingSuggestions && (
              <div className="manage-placeholder">
                <LoaderCircle className="spin" size={16} />
                Checking common model locations…
              </div>
            )}
            {!loadingSuggestions && visibleSuggestions.length === 0 && (
              <div className="manage-placeholder">
                <Box size={16} />
                No common model folders were found. Use Choose folder to pick one manually.
              </div>
            )}
            <div className="runtime-offer-list">
              {visibleSuggestions.map((entry) => (
                <article className="runtime-offer" key={entry.id}>
                  <div className="runtime-offer-info">
                    <strong>
                      {entry.label}
                      {entry.configured && <span className="installed-badge">Added</span>}
                    </strong>
                    <span>
                      {entry.path}
                      {` · ${modelCountSummary(entry.gguf_count, entry.mlx_count)}`}
                    </span>
                  </div>
                  <button
                    className="chip-button"
                    disabled={
                      entry.configured ||
                      (entry.gguf_count === 0 && entry.mlx_count === 0) ||
                      addingPath === entry.path
                    }
                    onClick={() => void addLibraryPath(entry.path)}
                  >
                    {addingPath === entry.path ? (
                      <LoaderCircle className="spin" size={13} />
                    ) : entry.configured ? (
                      <>
                        <Check size={13} />
                        Added
                      </>
                    ) : (
                      'Add'
                    )}
                  </button>
                </article>
              ))}
            </div>
            <p className="model-help">
              Use Choose folder for any other directory. Remove added folders in Engine
              configuration → Model library folders.
            </p>
          </div>
        )}
      </div>
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
          const key = modelLibraryKey(model.id)
          const parts = key.split('/')
          const name = parts.at(-1) ?? key
          const repo = parts.slice(0, -1).join('/')
          const caps = model.capabilities
          const isSelected = model.id === props.selectedModel
          const isExternal = isExternalModel(model)
          const engine = modelEngine(model)
          const runtimeNotice = runtimeNoticeForModel(
            model.id,
            props.models,
            runtimes,
            props.modelBindings
          )
          const compatibleRuntimes = runtimesForModel(model, runtimes)
          const boundRuntimeId = props.modelBindings?.[model.id] ?? ''
          return (
            <article className="library-card" key={model.id}>
              <div className="library-card-info">
                <strong>
                  {name}
                  <span className={engineBadgeClass(engine)}>{engineLabel(engine)}</span>
                  {isExternal && model.library_label && (
                    <span className="installed-badge">{model.library_label}</span>
                  )}
                </strong>
                <span>
                  {repo}
                  {model.size_bytes != null ? ` · ${formatBytes(model.size_bytes)}` : ''}
                  {isExternal ? ' · read-only' : ''}
                </span>
                {compatibleRuntimes.length > 0 && props.onSetModelBinding && (
                  <label className="library-runtime-picker">
                    <span>Runtime</span>
                    <select
                      value={boundRuntimeId}
                      onChange={(event) => {
                        const value = event.target.value
                        props.onSetModelBinding?.(model.id, value || null)
                      }}
                    >
                      <option value="">Default (active global runtime)</option>
                      {compatibleRuntimes.map((runtime) => (
                        <option key={runtime.id} value={runtime.id}>
                          {runtime.label}
                          {runtime.active ? ' · active' : ''}
                          {runtime.repository ? ` · fork` : ''}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                {runtimeNotice && (
                  <span className="library-runtime-note">{runtimeNotice}</span>
                )}
                <div className="library-caps">
                  {engine === 'whisper.cpp' && <span>batch ASR</span>}
                  {engine === 'streaming-asr' && <span>streaming ASR</span>}
                  {caps?.audio_input === 'native' && <span>native audio</span>}
                  {caps?.input_modalities.includes('image') && <span>vision</span>}
                  {caps?.input_modalities.includes('audio') &&
                    engine !== 'whisper.cpp' &&
                    engine !== 'streaming-asr' &&
                    caps?.audio_input !== 'native' && <span>audio</span>}
                  {caps?.tools && <span>tools</span>}
                  {caps?.reasoning && <span>reasoning</span>}
                </div>
              </div>
              <div className="library-card-actions">
                {engine === 'whisper.cpp' || engine === 'streaming-asr' ? (
                  <button
                    className="chip-button selected"
                    disabled
                    type="button"
                    title={
                      engine === 'streaming-asr'
                        ? 'Used for streaming transcription via /v1/audio/transcriptions'
                        : 'Used automatically for audio/video transcription when active'
                    }
                  >
                    <Check size={13} /> ASR model
                  </button>
                ) : (
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
                )}
                {!isExternal &&
                  (confirming === model.id ? (
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
                  ))}
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
  const [discoverEngine, setDiscoverEngine] = useState<DiscoverEngine>(() =>
    defaultDiscoverEngine(props.hardware)
  )
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
  const [trustByRepo, setTrustByRepo] = useState<Record<string, ModelTrust>>({})
  const [acknowledgedRepos, setAcknowledgedRepos] = useState<Record<string, boolean>>({})
  const [hfTokenSource, setHfTokenSource] = useState<string>('none')
  const [hfTokenDraft, setHfTokenDraft] = useState('')
  const [savingHfToken, setSavingHfToken] = useState(false)
  const [downloadJobs, setDownloadJobs] = useState<DownloadJob[]>([])

  useEffect(() => {
    void huggingFaceTokenStatus()
      .then((status) => setHfTokenSource(status.source))
      .catch(() => setHfTokenSource('none'))
  }, [])

  useEffect(() => {
    const refresh = (): void => {
      void listDownloadJobs()
        .then(setDownloadJobs)
        .catch(() => setDownloadJobs([]))
    }
    refresh()
    const timer = window.setInterval(refresh, 4000)
    return () => window.clearInterval(timer)
  }, [])

  async function saveHubToken(event: FormEvent): Promise<void> {
    event.preventDefault()
    setSavingHfToken(true)
    props.onError(null)
    try {
      await setHuggingFaceToken(hfTokenDraft.trim())
      setHfTokenDraft('')
      const status = await huggingFaceTokenStatus()
      setHfTokenSource(status.source)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setSavingHfToken(false)
    }
  }

  async function removeHubToken(): Promise<void> {
    setSavingHfToken(true)
    props.onError(null)
    try {
      await clearHuggingFaceToken()
      const status = await huggingFaceTokenStatus()
      setHfTokenSource(status.source)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setSavingHfToken(false)
    }
  }

  useEffect(() => {
    setDiscoverEngine(defaultDiscoverEngine(props.hardware))
  }, [props.hardware?.os, props.hardware?.architecture])

  async function findModels(event?: FormEvent): Promise<void> {
    event?.preventDefault()
    setSearching(true)
    props.onError(null)
    setExpandedRepo(null)
    try {
      setResults(await searchHub(query.trim(), discoverEngine))
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setSearching(false)
    }
  }

  async function downloadSnapshot(repoId: string): Promise<void> {
    if (discoverEngine === 'llama.cpp' || discoverEngine === 'whisper.cpp') return
    const trust = trustByRepo[repoId]
    if (trust?.gated && hfTokenSource === 'none') {
      props.onError('This model is gated on Hugging Face. Save an access token above first.')
      return
    }
    if (trust?.requires_acknowledgement && !acknowledgedRepos[repoId]) {
      props.onError('Review and acknowledge the license / remote-code notice before downloading.')
      return
    }
    const key = `${repoId}::snapshot`
    setDownloadProgress({ key, event: null })
    props.onError(null)
    try {
      const result =
        discoverEngine === 'streaming-asr'
          ? await downloadStreamingAsrModel(repoId, (event) =>
              setDownloadProgress({ key, event })
            )
          : await downloadMlxModel(
              repoId,
              discoverEngine === 'mlx-vlm' ? 'mlx-vlm' : 'mlx-lm',
              (event) => setDownloadProgress({ key, event })
            )
      await props.refreshModels()
      if (result.model_id && discoverEngine !== 'streaming-asr') {
        props.onSelectModel(result.model_id)
      }
      const engine = result.engine ?? discoverEngine
      setEnginePhase(
        discoverEngine === 'streaming-asr'
          ? 'Checking the streaming ASR runtime…'
          : 'Checking the MLX runtime…'
      )
      try {
        const runtimeResponse = await listRuntimes()
        const hasRuntime = runtimeResponse.data.some(
          (entry) => entry.engine === engine && entry.active
        )
        if (result.notice) {
          props.onError(result.notice)
        } else if (!hasRuntime) {
          props.onError(
            `Model downloaded for ${engineLabel(engine)}, but that runtime is not active yet. ` +
              'Build and activate it in the Runtimes section.'
          )
        }
      } finally {
        setEnginePhase(null)
      }
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setDownloadProgress(null)
    }
  }

  async function toggleRepo(model: HubModel): Promise<void> {
    if (expandedRepo === model.id) {
      setExpandedRepo(null)
      return
    }
    setExpandedRepo(model.id)
    if (
      discoverEngine !== 'llama.cpp' &&
      discoverEngine !== 'whisper.cpp'
    ) {
      if (trustByRepo[model.id]) return
      setLoadingFilesFor(model.id)
      props.onError(null)
      try {
        const trust = await fetchModelTrust(model.id)
        setTrustByRepo((current) => ({ ...current, [model.id]: trust }))
      } catch (cause) {
        props.onError(errorText(cause))
        setExpandedRepo(null)
      } finally {
        setLoadingFilesFor(null)
      }
      return
    }
    if (repoFiles[model.id]) return
    setLoadingFilesFor(model.id)
    props.onError(null)
    try {
      const [files, trust] = await Promise.all([
        listHubFiles(model.id),
        fetchModelTrust(model.id)
      ])
      setRepoFiles((current) => ({ ...current, [model.id]: files.data }))
      setPreferredFiles((current) => ({ ...current, [model.id]: files.preferred_filename }))
      setTrustByRepo((current) => ({ ...current, [model.id]: trust }))
    } catch (cause) {
      props.onError(errorText(cause))
      setExpandedRepo(null)
    } finally {
      setLoadingFilesFor(null)
    }
  }

  async function downloadQuant(repoId: string, path: string): Promise<void> {
    const trust = trustByRepo[repoId]
    if (trust?.gated && hfTokenSource === 'none') {
      props.onError('This model is gated on Hugging Face. Save an access token above first.')
      return
    }
    if (trust?.requires_acknowledgement && !acknowledgedRepos[repoId]) {
      props.onError('Review and acknowledge the license / remote-code notice before downloading.')
      return
    }
    const key = `${repoId}::${path}`
    setDownloadProgress({ key, event: null })
    props.onError(null)
    try {
      const engine = discoverEngine === 'whisper.cpp' ? 'whisper.cpp' : 'llama.cpp'
      await downloadModel(
        repoId,
        path,
        (event) => setDownloadProgress({ key, event }),
        'main',
        engine
      )
      await props.refreshModels()
      if (engine === 'whisper.cpp') {
        props.onError(null)
        return
      }
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

  async function queueQuant(repoId: string, path: string): Promise<void> {
    const trust = trustByRepo[repoId]
    if (trust?.gated && hfTokenSource === 'none') {
      props.onError('This model is gated on Hugging Face. Save an access token above first.')
      return
    }
    if (trust?.requires_acknowledgement && !acknowledgedRepos[repoId]) {
      props.onError('Review and acknowledge the license / remote-code notice before downloading.')
      return
    }
    props.onError(null)
    try {
      await queueModelDownload(repoId, path)
      setDownloadJobs(await listDownloadJobs())
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  async function cancelJob(jobId: string): Promise<void> {
    props.onError(null)
    try {
      await cancelDownloadJob(jobId)
      setDownloadJobs(await listDownloadJobs())
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  return (
    <section>
      <header className="manage-heading">
        <h2>Download models</h2>
        <p>
          Search Hugging Face for{' '}
          {discoverEngine === 'llama.cpp'
            ? 'GGUF weights compatible with llama.cpp'
            : discoverEngine === 'whisper.cpp'
              ? 'Whisper speech-to-text weights for whisper.cpp'
              : discoverEngine === 'streaming-asr'
                ? 'Nemotron ASR Streaming snapshots for low-latency transcription'
                : `${engineLabel(discoverEngine)} models for Apple Silicon`}
          .
        </p>
        <p className="manage-subtext">{DISCOVER_ENGINE_HELP[discoverEngine]}</p>
      </header>
      <div className="build-form-row">
        <label>
          <span>Model type</span>
          <select
            value={discoverEngine}
            onChange={(event) => {
              setDiscoverEngine(event.target.value as DiscoverEngine)
              setResults([])
              setExpandedRepo(null)
            }}
          >
            {(
              [
                'llama.cpp',
                'whisper.cpp',
                'streaming-asr',
                ...(props.hardware?.os === 'macos' && props.hardware.architecture === 'aarch64'
                  ? (['mlx-lm', 'mlx-vlm'] as const)
                  : [])
              ] as DiscoverEngine[]
            ).map((engine) => (
              <option key={engine} value={engine}>
                {DISCOVER_ENGINE_LABELS[engine]}
              </option>
            ))}
          </select>
        </label>
      </div>
      <form className="build-form" onSubmit={(event) => void saveHubToken(event)}>
        <label>
          <span>Hugging Face token (for gated models)</span>
          <input
            type="password"
            autoComplete="off"
            value={hfTokenDraft}
            onChange={(event) => setHfTokenDraft(event.target.value)}
            placeholder={
              hfTokenSource === 'environment'
                ? 'Using HF_TOKEN from the environment'
                : hfTokenSource === 'stored'
                  ? 'Token saved — paste to replace'
                  : 'hf_…'
            }
          />
        </label>
        <p className="model-help">
          Stored locally under your Brazier data directory (mode 0600). You can also set{' '}
          <code>HF_TOKEN</code> in the environment.
        </p>
        <div className="build-form-actions">
          <button
            className="secondary-action"
            type="submit"
            disabled={savingHfToken || !hfTokenDraft.trim()}
          >
            {savingHfToken ? <LoaderCircle className="spin" size={14} /> : 'Save token'}
          </button>
          {hfTokenSource === 'stored' && (
            <button
              className="chip-button subtle"
              type="button"
              disabled={savingHfToken}
              onClick={() => void removeHubToken()}
            >
              Remove saved token
            </button>
          )}
        </div>
      </form>
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
      {downloadJobs.length > 0 && (
        <div className="download-jobs-panel">
          <div className="section-label">Download queue</div>
          {downloadJobs.slice(0, 6).map((job) => {
            const active = job.status === 'pending' || job.status === 'downloading'
            const basename = job.filename.split('/').at(-1) ?? job.filename
            const pct =
              job.bytes_downloaded != null && job.total_bytes
                ? Math.min(100, (job.bytes_downloaded / job.total_bytes) * 100)
                : null
            return (
              <div className="download-job-row" key={job.id}>
                <div>
                  <strong>{basename}</strong>
                  <span>
                    {job.repo_id} · {job.status}
                    {job.bytes_downloaded != null && job.total_bytes
                      ? ` · ${formatBytes(job.bytes_downloaded)} / ${formatBytes(job.total_bytes)}`
                      : ''}
                  </span>
                  {pct != null && active && (
                    <div className="progress-track compact">
                      <div className="progress-fill" style={{ width: `${pct}%` }} />
                    </div>
                  )}
                  {job.error && <span className="run-error-text">{job.error}</span>}
                </div>
                {active && (
                  <button type="button" className="chip-button subtle" onClick={() => void cancelJob(job.id)}>
                    Cancel
                  </button>
                )}
              </div>
            )
          })}
        </div>
      )}
      <div className="model-results">
        {results.map((model) => {
          const expanded = expandedRepo === model.id
          const files = (repoFiles[model.id] ?? []).filter((file) => {
            const lower = file.path.toLowerCase()
            if (discoverEngine === 'whisper.cpp') {
              return lower.endsWith('.bin') || lower.endsWith('.gguf')
            }
            return lower.endsWith('.gguf')
          })
          const preferred = preferredFiles[model.id]
          const filePickEngine =
            discoverEngine === 'llama.cpp' || discoverEngine === 'whisper.cpp'
          // streaming-asr / mlx use snapshot download, not per-file picker
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
                      {expanded
                        ? filePickEngine
                          ? 'Hide files'
                          : 'Hide details'
                        : filePickEngine
                          ? 'Choose file'
                          : 'Download'}
                    </>
                  )}
                </button>
              </div>
              {expanded && (
                <div className="quant-list">
                  {trustByRepo[model.id]?.requires_acknowledgement && (
                    <div className="build-warning">
                      <ShieldAlert size={14} />
                      <div>
                        {trustByRepo[model.id]?.license && (
                          <p>License: {trustByRepo[model.id]?.license}</p>
                        )}
                        {trustByRepo[model.id]?.remote_code && (
                          <p>
                            This repository includes non-GGUF artifacts (Transformers / Python).
                            You are downloading weights only; do not execute untrusted code from
                            the Hub.
                          </p>
                        )}
                        {trustByRepo[model.id]?.gated && hfTokenSource === 'none' && (
                          <p>A Hugging Face token is required for this gated model.</p>
                        )}
                        <label className="toggle-row compact">
                          <span>I understand and want to download from this repository.</span>
                          <input
                            type="checkbox"
                            checked={acknowledgedRepos[model.id] ?? false}
                            onChange={(event) =>
                              setAcknowledgedRepos((current) => ({
                                ...current,
                                [model.id]: event.target.checked
                              }))
                            }
                          />
                        </label>
                      </div>
                    </div>
                  )}
                  {discoverEngine !== 'llama.cpp' && discoverEngine !== 'whisper.cpp' ? (
                    <div className="quant-row">
                      <div>
                        <strong>Full MLX snapshot</strong>
                        <span>
                          Downloads config, tokenizer, and weights. Brazier detects whether this
                          is {engineLabel('mlx-lm')} or {engineLabel('mlx-vlm')} from the model
                          config — you do not need to match the dropdown exactly.
                        </span>
                      </div>
                      <div className="quant-actions">
                        <button
                          type="button"
                          disabled={downloadProgress?.key === `${model.id}::snapshot`}
                          onClick={() => void downloadSnapshot(model.id)}
                        >
                          {downloadProgress?.key === `${model.id}::snapshot` ? (
                            <LoaderCircle className="spin" size={14} />
                          ) : (
                            <Download size={14} />
                          )}
                          Download
                        </button>
                      </div>
                    </div>
                  ) : files.length === 0 ? (
                    <p className="empty-models-inline">No GGUF files found in this repo.</p>
                  ) : (
                    files.map((file) => {
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
                        <div className="quant-actions">
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
                          {!active && (
                            <button
                              type="button"
                              className="chip-button subtle"
                              disabled={downloadProgress != null}
                              title="Download in the background queue"
                              onClick={() => void queueQuant(model.id, file.path)}
                            >
                              Queue
                            </button>
                          )}
                        </div>
                      </div>
                    )
                  })
                  )}
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

function sourceBuildTargets(hardware: HardwareInfo | null): RuntimeTarget[] {
  switch (hardware?.os) {
    case 'macos':
      return ['cpu', 'metal']
    case 'windows':
      return ['cpu', 'cuda', 'vulkan']
    default:
      return ['cpu', 'cuda', 'rocm', 'vulkan']
  }
}

const BUILD_TARGET_LABELS: Record<RuntimeTarget, string> = {
  auto: 'Auto',
  cpu: 'CPU',
  cuda: 'NVIDIA CUDA',
  rocm: 'AMD ROCm',
  metal: 'Apple Metal',
  vulkan: 'Vulkan'
}

function llamaRuntimeLabel(target: RuntimeTarget): string {
  return `llama.cpp · ${BUILD_TARGET_LABELS[target]}`
}

function managedRuntimeInstalled(
  runtimes: RuntimeEntry[],
  target: RuntimeTarget
): boolean {
  if (target === 'cpu') {
    return runtimes.some(
      (runtime) =>
        runtime.kind === 'managed' &&
        (runtime.target === 'cpu' || runtime.id === 'managed')
    )
  }
  return runtimes.some(
    (runtime) => runtime.kind === 'managed' && runtime.target === target
  )
}

function managedTargetStatus(
  target: RuntimeTarget,
  statuses: ManagedLlamaTargetStatus[] | null
): ManagedLlamaTargetStatus | null {
  return statuses?.find((entry) => entry.target === target) ?? null
}

function pythonRuntimeInstalled(runtimes: RuntimeEntry[], engine: BuildEngine): boolean {
  return runtimes.some((runtime) => runtime.engine === engine)
}

function RuntimesSection(props: SectionProps): React.JSX.Element {
  const maxBuildJobs = Math.max(1, props.hardware?.logical_cpus ?? 8)
  const initialBuildJobs = Math.max(
    1,
    props.settings?.build_jobs ??
      Math.floor((props.hardware?.logical_cpus ?? maxBuildJobs) / 2)
  )
  const [runtimes, setRuntimes] = useState<RuntimeEntry[] | null>(
    props.initialRuntimes ?? null
  )
  const [busyRuntime, setBusyRuntime] = useState<string | null>(null)
  const [confirming, setConfirming] = useState<string | null>(null)
  const [installingTarget, setInstallingTarget] = useState<RuntimeTarget | null>(null)
  const [installProgress, setInstallProgress] = useState<JobProgressState | null>(null)
  const [savingTarget, setSavingTarget] = useState(false)

  // Build-from-source form.
  const [buildOpen, setBuildOpen] = useState(false)
  const isAppleSilicon =
    props.hardware?.os === 'macos' && props.hardware.architecture === 'aarch64'
  const [buildEngine, setBuildEngine] = useState<BuildEngine>(() =>
    isAppleSilicon ? 'mlx-lm' : 'llama.cpp'
  )
  const [repository, setRepository] = useState(
    BUILD_ENGINE_DEFAULTS[isAppleSilicon ? 'mlx-lm' : 'llama.cpp'].repository
  )
  const [revision, setRevision] = useState(
    BUILD_ENGINE_DEFAULTS[isAppleSilicon ? 'mlx-lm' : 'llama.cpp'].revision
  )
  const [buildTarget, setBuildTarget] = useState<RuntimeTarget>('cpu')
  const buildTargets = useMemo(
    () => sourceBuildTargets(props.hardware),
    [props.hardware?.os]
  )
  const managedTargets = useMemo(
    () => (props.hardware?.targets ?? []).filter((target) => target.managed_install),
    [props.hardware?.targets]
  )
  const buildEngineOptions = useMemo((): BuildEngine[] => {
    const engines: BuildEngine[] = ['streaming-asr']
    if (props.hardware?.os === 'macos' && props.hardware.architecture === 'aarch64') {
      engines.unshift('mlx-lm', 'mlx-vlm')
    }
    return engines
  }, [props.hardware?.os, props.hardware?.architecture])
  const [buildJobs, setBuildJobs] = useState(initialBuildJobs)
  const [building, setBuilding] = useState(false)
  const [activeBuildId, setActiveBuildId] = useState<string | null>(null)
  const [buildProgress, setBuildProgress] = useState<JobProgressState>(() =>
    emptyJobProgress('Preparing source build')
  )
  const [buildWarning, setBuildWarning] = useState<string | null>(null)
  const [managedStatuses, setManagedStatuses] = useState<ManagedLlamaTargetStatus[] | null>(
    null
  )
  const logRef = useRef<HTMLPreElement>(null)

  function applyBuildEngine(engine: BuildEngine, repositoryOverride?: string): void {
    const defaults = BUILD_ENGINE_DEFAULTS[engine]
    setBuildEngine(engine)
    setRepository(repositoryOverride ?? defaults.repository)
    setRevision(defaults.revision)
  }

  useEffect(() => {
    if (!props.pendingBuild) return
    applyBuildEngine(props.pendingBuild.engine, props.pendingBuild.repository)
    setBuildOpen(true)
    props.onPendingBuildConsumed?.()
  }, [props.pendingBuild, props.onPendingBuildConsumed])

  async function refreshManagedStatuses(): Promise<void> {
    try {
      const response = await fetchManagedLlamaStatus()
      setManagedStatuses(response.targets)
    } catch {
      setManagedStatuses(null)
    }
  }

  async function refreshRuntimes(): Promise<void> {
    try {
      const response = await listRuntimes()
      setRuntimes(response.data)
    } catch (cause) {
      if (props.initialRuntimes?.length) {
        setRuntimes(props.initialRuntimes)
      } else {
        props.onError(errorText(cause))
        setRuntimes([])
      }
    }
  }

  useEffect(() => {
    void refreshRuntimes()
    void refreshManagedStatuses()
  }, [])

  useEffect(() => {
    setBuildJobs(
      Math.max(
        1,
        props.settings?.build_jobs ??
          Math.floor((props.hardware?.logical_cpus ?? maxBuildJobs) / 2)
      )
    )
  }, [props.settings?.build_jobs, props.hardware?.logical_cpus, maxBuildJobs])

  useEffect(() => {
    const recommended = props.hardware?.recommended_target
    if (recommended && recommended !== 'auto' && buildTargets.includes(recommended)) {
      setBuildTarget(recommended)
    }
  }, [props.hardware?.recommended_target, buildTargets])

  useEffect(() => {
    if (!buildTargets.includes(buildTarget)) {
      setBuildTarget('cpu')
    }
  }, [buildTarget, buildTargets])

  async function persistBuildJobs(jobs: number): Promise<void> {
    if (!props.settings || props.settings.build_jobs === jobs) return
    try {
      const saved = await saveRuntimeSettings({ ...props.settings, build_jobs: jobs })
      props.onSettingsSaved(saved)
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight })
  }, [buildProgress.logLines])

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

  async function installManaged(target: RuntimeTarget, force = false): Promise<void> {
    const label = llamaRuntimeLabel(target)
    setInstallingTarget(target)
    setInstallProgress(
      emptyJobProgress(force ? `Updating ${label}` : `Installing ${label}`)
    )
    props.onError(null)
    try {
      await ensureLlamaEngine(
        (event) => {
          setInstallProgress((current) =>
            applyJobProgress(
              current ?? emptyJobProgress(force ? `Updating ${label}` : `Installing ${label}`),
              event
            )
          )
        },
        { target, force }
      )
      await refreshRuntimes()
      await refreshManagedStatuses()
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setInstallingTarget(null)
    }
  }

  function openBuildForEngine(engine: BuildEngine): void {
    applyBuildEngine(engine)
    setBuildOpen(true)
  }

  function appendBuildDiagnostics(
    lines: string[],
    diagnostics: Record<string, unknown> | undefined
  ): string[] {
    if (!diagnostics) return lines
    const hints = diagnostics.hints
    const excerpt = diagnostics.log_excerpt
    const next = [...lines]
    if (typeof excerpt === 'string' && excerpt.trim()) {
      next.push('', '--- last log lines ---', excerpt)
    }
    if (Array.isArray(hints) && hints.length > 0) {
      next.push('', 'Suggested fixes:')
      for (const hint of hints) {
        if (typeof hint === 'string') next.push(`• ${hint}`)
      }
    }
    return next
  }

  const isPythonBuild =
    buildEngine === 'mlx-lm' || buildEngine === 'mlx-vlm' || buildEngine === 'streaming-asr'
  const isWhisperBuild = buildEngine === 'whisper.cpp'
  const isStreamingAsrBuild = buildEngine === 'streaming-asr'

  async function runBuild(event: FormEvent): Promise<void> {
    event.preventDefault()
    setBuilding(true)
    setActiveBuildId(null)
    setBuildProgress(emptyJobProgress('Preparing source build'))
    setBuildWarning(null)
    props.onError(null)
    try {
      await buildRuntime(
        buildEngine,
        repository.trim(),
        revision.trim(),
        buildTarget,
        buildJobs,
        (progress) => {
          if (progress.phase === 'warning' && progress.message) {
            setBuildWarning(progress.message)
            return
          }
          if (progress.result && typeof progress.result === 'object') {
            const buildId = (progress.result as { build_id?: string }).build_id
            if (buildId) setActiveBuildId(buildId)
          }
          setBuildProgress((current) => applyJobProgress(current, progress))
        },
        { onBuildId: setActiveBuildId }
      )
      setBuildProgress((current) => ({
        ...current,
        headline: 'Build complete — activate it below to use it.',
        percent: 100,
        phase: 'done'
      }))
      await refreshRuntimes()
    } catch (cause) {
      const diagnostics =
        cause instanceof Error
          ? (cause as Error & { diagnostics?: Record<string, unknown> }).diagnostics
          : undefined
      setBuildProgress((current) => ({
        ...current,
        logLines: appendBuildDiagnostics(current.logLines, diagnostics),
        hints: Array.isArray(diagnostics?.hints)
          ? diagnostics.hints.filter((hint): hint is string => typeof hint === 'string')
          : current.hints,
        headline: 'Build failed',
        phase: 'error'
      }))
      props.onError(`Build failed: ${errorText(cause)}`)
    } finally {
      setBuilding(false)
      setActiveBuildId(null)
    }
  }

  async function cancelActiveBuild(): Promise<void> {
    if (!activeBuildId) return
    props.onError(null)
    try {
      await cancelBuild(activeBuildId)
      setBuildProgress((current) => ({
        ...current,
        headline: 'Cancellation requested…',
        logLines: [...current.logLines, 'Cancellation requested…']
      }))
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  const targets = props.hardware?.targets ?? []
  const runtimeList = runtimes ?? []
  const customRuntimes = runtimeList.filter((runtime) => runtime.kind !== 'managed')

  return (
    <section>
      <header className="manage-heading">
        <h2>Runtimes</h2>
        <p>
          Install or build an inference engine for your hardware, then activate it below. GGUF models
          use llama.cpp; MLX models on Apple Silicon use MLX-LM or MLX-VLM.
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
        <div className="section-label">Available for your hardware</div>
        {managedTargets.length === 0 && buildEngineOptions.length === 0 && (
          <div className="manage-placeholder">
            <Cpu size={16} />
            Detecting supported runtimes…
          </div>
        )}
        {installProgress && (
          <JobProgressPanel progress={installProgress} active={installingTarget != null} />
        )}
        <div className="runtime-offer-list">
          {managedTargets.map((target) => {
            const installed = managedRuntimeInstalled(runtimeList, target.id)
            const status = managedTargetStatus(target.id, managedStatuses)
            const updateAvailable = status?.update_available ?? false
            const installedVersion = status?.installed_version
            const latestVersion = status?.latest_version
            const installing = installingTarget === target.id
            const versionLine = [
              target.detail,
              installed && installedVersion ? `Installed · ${installedVersion}` : null,
              updateAvailable && latestVersion ? `Latest · ${latestVersion}` : null
            ]
              .filter(Boolean)
              .join(' · ')
            return (
              <article
                className={`runtime-offer ${target.available ? '' : 'unavailable'}`}
                key={target.id}
              >
                <div className="runtime-offer-info">
                  <strong>
                    {llamaRuntimeLabel(target.id)}
                    {target.recommended && <span className="active-badge">Recommended</span>}
                    {installed && !updateAvailable && (
                      <span className="installed-badge">Up to date</span>
                    )}
                    {updateAvailable && <span className="installed-badge">Update</span>}
                  </strong>
                  <span>{versionLine}</span>
                </div>
                <button
                  className="chip-button"
                  disabled={!target.available || installing || (installed && !updateAvailable)}
                  title={
                    !target.available
                      ? target.detail
                      : updateAvailable
                        ? `Update to ${latestVersion ?? 'latest'}`
                        : installed
                          ? `Installed (${installedVersion ?? 'unknown version'})`
                          : `Download ${llamaRuntimeLabel(target.id)}`
                  }
                  onClick={() => void installManaged(target.id, updateAvailable)}
                >
                  {installing ? (
                    <LoaderCircle className="spin" size={13} />
                  ) : updateAvailable ? (
                    <>
                      <Download size={13} />
                      Update
                    </>
                  ) : installed ? (
                    <>
                      <Check size={13} />
                      Installed
                    </>
                  ) : (
                    <>
                      <Download size={13} />
                      Download
                    </>
                  )}
                </button>
              </article>
            )
          })}
          {buildEngineOptions.map((engine) => {
            const installed = pythonRuntimeInstalled(runtimeList, engine)
            return (
              <article className="runtime-offer" key={engine}>
                <div className="runtime-offer-info">
                  <strong>
                    {DISCOVER_ENGINE_LABELS[engine]}
                    {engine === 'mlx-lm' && isAppleSilicon && (
                      <span className="active-badge">Recommended</span>
                    )}
                    {installed && <span className="installed-badge">Built</span>}
                  </strong>
                  <span>
                    {engine === 'streaming-asr'
                      ? 'Build an isolated Python environment with Transformers for Nemotron streaming ASR. Requires uv.'
                      : 'Build a local Python environment with uv. Required for MLX models on Apple Silicon.'}
                  </span>
                </div>
                <button
                  className="chip-button"
                  title={`Build ${DISCOVER_ENGINE_LABELS[engine]} from source`}
                  onClick={() => openBuildForEngine(engine)}
                >
                  <Hammer size={13} />
                  {installed ? 'Build again' : 'Build'}
                </button>
              </article>
            )
          })}
        </div>
        {managedTargets.some((target) => !target.available) && (
          <p className="model-help">
            Grayed-out options need hardware or drivers that were not detected on this machine. You
            can still build llama.cpp for them from source below.
          </p>
        )}
      </div>

      <div className="settings-group">
        <div className="section-label">Custom runtimes</div>
        {runtimes == null && !props.initialRuntimes?.length && (
          <div className="manage-placeholder">
            <LoaderCircle className="spin" size={16} />
            Scanning for custom runtimes…
          </div>
        )}
        {runtimes != null && customRuntimes.length === 0 && (
          <div className="manage-placeholder compact">
            <Cpu size={16} />
            Source builds and forks appear here after you build them.
          </div>
        )}
        <div className="runtime-list">
          {customRuntimes.map((runtime) => (
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
                  {[runtime.version, runtime.target].filter(Boolean).join(' · ')}
                </span>
                <code title={runtime.path}>{runtime.path}</code>
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
      </div>

      <div className="settings-group">
        <button className="build-toggle" onClick={() => setBuildOpen((open) => !open)}>
          {buildOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <Hammer size={14} />
          Build from source
        </button>
        {buildOpen && (
          <form className="build-form" onSubmit={(event) => void runBuild(event)}>
            {isAppleSilicon ||
            buildEngine === 'whisper.cpp' ||
            buildEngine === 'llama.cpp' ||
            buildEngine === 'streaming-asr' ? (
              <label>
                <span>Engine</span>
                <select
                  value={buildEngine}
                  onChange={(event) =>
                    applyBuildEngine(event.target.value as BuildEngine)
                  }
                >
                  {(
                    [
                      'llama.cpp',
                      'whisper.cpp',
                      'streaming-asr',
                      ...(isAppleSilicon ? (['mlx-lm', 'mlx-vlm'] as const) : [])
                    ] as BuildEngine[]
                  ).map((engine) => (
                    <option key={engine} value={engine}>
                      {DISCOVER_ENGINE_LABELS[engine]}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            {!isStreamingAsrBuild && (
              <label>
                <span>Repository</span>
                <input
                  value={repository}
                  onChange={(event) => setRepository(event.target.value)}
                  placeholder={BUILD_ENGINE_DEFAULTS[buildEngine].repository}
                />
              </label>
            )}
            <div className="build-form-row">
              {!isStreamingAsrBuild && (
                <label>
                  <span>Branch, tag, or commit</span>
                  <input value={revision} onChange={(event) => setRevision(event.target.value)} />
                </label>
              )}
              {!isPythonBuild && (
                <label>
                  <span>Target</span>
                  <select
                    value={buildTarget}
                    onChange={(event) => setBuildTarget(event.target.value as RuntimeTarget)}
                  >
                    {buildTargets.map((target) => (
                      <option key={target} value={target}>
                        {BUILD_TARGET_LABELS[target]}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>
            <p className="model-help">
              {isStreamingAsrBuild
                ? 'Installs a bundled Transformers worker plus pinned deps with uv (no Git checkout). Then download a Nemotron ASR Streaming snapshot from Discover.'
                : isPythonBuild
                  ? 'MLX builds create an isolated Python environment with uv. Install uv (`brew install uv`) before starting the build.'
                  : isWhisperBuild
                    ? 'whisper.cpp builds produce the whisper-cli binary used to transcribe audio and video soundtracks before chat.'
                    : props.hardware?.os === 'macos'
                      ? 'macOS builds use Xcode Command Line Tools. Metal is the recommended GPU target.'
                      : props.hardware?.os === 'windows'
                        ? 'Windows builds need Git, CMake, and Visual Studio 2022 Build Tools with the C++ workload.'
                        : 'Linux builds need git, cmake, and a distro C++ toolchain. GPU targets also need the matching SDK or driver stack.'}
            </p>
            {!isPythonBuild && (
              <label className="slider-row">
                <span>
                  Parallel jobs (-j) <em>{buildJobs}</em>
                </span>
                <input
                  type="range"
                  min={1}
                  max={maxBuildJobs}
                  step={1}
                  value={buildJobs}
                  onChange={(event) => setBuildJobs(Number(event.target.value))}
                  onMouseUp={(event) => void persistBuildJobs(Number(event.currentTarget.value))}
                  onTouchEnd={(event) => void persistBuildJobs(Number(event.currentTarget.value))}
                  onKeyUp={(event) => void persistBuildJobs(Number(event.currentTarget.value))}
                />
              </label>
            )}
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
            <div className="build-form-actions">
              <button
                className="primary-action"
                type="submit"
                disabled={building || !repository.trim() || !revision.trim()}
              >
                {building ? <LoaderCircle className="spin" size={15} /> : <Hammer size={15} />}
                {building ? 'Building…' : 'Start build'}
              </button>
              {building && activeBuildId && (
                <button
                  className="chip-button danger"
                  type="button"
                  onClick={() => void cancelActiveBuild()}
                >
                  Cancel build
                </button>
              )}
            </div>
            <JobProgressPanel progress={buildProgress} active={building} />
            {(building || buildProgress.logLines.length > 0) && (
              <pre className="build-log" ref={logRef}>
                {buildProgress.logLines.join('\n')}
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
    const pathsChanged =
      JSON.stringify(props.settings?.extra_model_library_paths ?? []) !==
      JSON.stringify(draft.extra_model_library_paths ?? [])
    try {
      const saved = await saveRuntimeSettings(draft)
      props.onSettingsSaved(saved)
      if (pathsChanged) {
        await props.refreshModels()
      }
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
      <div className="settings-group">
        <div className="section-label">Model library folders</div>
        <p className="model-help">
          Extra directories scanned for GGUF models. Removing a folder hides those models in Brazier
          but does not delete files on disk.
        </p>
        {draft.extra_model_library_paths.length === 0 ? (
          <div className="manage-placeholder">
            <Box size={16} />
            No external folders added. Browse common locations from the Model library section.
          </div>
        ) : (
          <div className="library-path-list">
            {draft.extra_model_library_paths.map((path) => (
              <div className="library-path-row" key={path}>
                <code>{path}</code>
                <button
                  className="chip-button subtle"
                  onClick={() =>
                    setDraft({
                      ...draft,
                      extra_model_library_paths: draft.extra_model_library_paths.filter(
                        (entry) => entry !== path
                      )
                    })
                  }
                >
                  <X size={13} />
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
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

function McpSection(props: SectionProps): React.JSX.Element {
  const [servers, setServers] = useState<McpServer[]>([])
  const [catalog, setCatalog] = useState<BundledTool[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [draft, setDraft] = useState({
    id: '',
    name: '',
    command: '',
    args: ''
  })

  async function reload(): Promise<void> {
    setLoading(true)
    props.onError(null)
    try {
      const [serverList, tools] = await Promise.all([listMcpServers(), listTools()])
      setServers(serverList)
      setCatalog(tools)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void reload()
  }, [])

  async function addServer(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (!draft.id.trim() || !draft.command.trim()) return
    setSaving(true)
    props.onError(null)
    try {
      await createMcpServer({
        id: draft.id.trim(),
        name: draft.name.trim() || draft.id.trim(),
        command: draft.command.trim(),
        args: draft.args
          .split(',')
          .map((part) => part.trim())
          .filter(Boolean)
      })
      setDraft({ id: '', name: '', command: '', args: '' })
      await reload()
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setSaving(false)
    }
  }

  async function toggleEnabled(server: McpServer): Promise<void> {
    props.onError(null)
    try {
      await updateMcpServer(server.id, {
        id: server.id,
        name: server.name,
        command: server.command,
        args: server.args,
        enabled: !server.enabled
      })
      await reload()
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  async function removeServer(id: string): Promise<void> {
    props.onError(null)
    try {
      await deleteMcpServer(id)
      await reload()
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  async function refreshTools(id: string): Promise<void> {
    setRefreshing(id)
    props.onError(null)
    try {
      await refreshMcpServer(id)
      await reload()
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setRefreshing(null)
    }
  }

  const builtinTools = catalog.filter((tool) => tool.source !== 'mcp')

  return (
    <section>
      <header className="manage-heading">
        <h2>MCP servers</h2>
        <p>
          Connect Model Context Protocol servers over stdio. Their tools are merged with bundled
          tools when tools are enabled in chat.
        </p>
      </header>

      {loading ? (
        <div className="manage-placeholder">
          <LoaderCircle className="spin" size={16} />
          Loading…
        </div>
      ) : (
        <>
          <div className="runtime-list">
            {servers.length === 0 ? (
              <p className="model-help">No MCP servers configured yet.</p>
            ) : (
              servers.map((server) => (
                <article className="runtime-card" key={server.id}>
                  <div className="runtime-card-info">
                    <strong>{server.name}</strong>
                    <span>
                      {server.command} {server.args.join(' ')}
                    </span>
                    {server.tools.length > 0 && (
                      <span>
                        {server.tools.length} tool{server.tools.length === 1 ? '' : 's'} cached
                      </span>
                    )}
                  </div>
                  <div className="library-card-actions">
                    <label className="chip-button subtle" title="Enable server">
                      <input
                        type="checkbox"
                        checked={server.enabled}
                        onChange={() => void toggleEnabled(server)}
                      />
                      Enabled
                    </label>
                    <button
                      className="chip-button"
                      disabled={refreshing === server.id}
                      onClick={() => void refreshTools(server.id)}
                    >
                      {refreshing === server.id ? (
                        <LoaderCircle className="spin" size={13} />
                      ) : (
                        'Refresh tools'
                      )}
                    </button>
                    <button
                      className="chip-button danger"
                      onClick={() => void removeServer(server.id)}
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </article>
              ))
            )}
          </div>

          <form className="settings-group" onSubmit={(event) => void addServer(event)}>
            <div className="section-label">Add server</div>
            <div className="settings-grid">
              <label>
                <span>ID</span>
                <input
                  value={draft.id}
                  onChange={(event) => setDraft({ ...draft, id: event.target.value })}
                  placeholder="filesystem"
                  required
                />
              </label>
              <label>
                <span>Display name</span>
                <input
                  value={draft.name}
                  onChange={(event) => setDraft({ ...draft, name: event.target.value })}
                  placeholder="Filesystem tools"
                />
              </label>
              <label>
                <span>Command</span>
                <input
                  value={draft.command}
                  onChange={(event) => setDraft({ ...draft, command: event.target.value })}
                  placeholder="npx"
                  required
                />
              </label>
              <label>
                <span>Args (comma-separated)</span>
                <input
                  value={draft.args}
                  onChange={(event) => setDraft({ ...draft, args: event.target.value })}
                  placeholder="-y,@modelcontextprotocol/server-filesystem,/Users/me"
                />
              </label>
            </div>
            <div className="runtime-actions">
              <button className="primary-action" type="submit" disabled={saving}>
                {saving ? <LoaderCircle className="spin" size={15} /> : 'Add server'}
              </button>
            </div>
          </form>

          {builtinTools.length > 0 && (
            <div className="settings-group">
              <div className="section-label">Bundled tools</div>
              <p className="model-help">
                {builtinTools.map((tool) => tool.title).join(' · ')}
              </p>
            </div>
          )}
        </>
      )}
    </section>
  )
}

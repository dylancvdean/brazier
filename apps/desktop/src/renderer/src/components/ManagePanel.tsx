import {
  Bot,
  Box,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Cpu,
  Download,
  FolderOpen,
  Globe,
  Hammer,
  HardDrive,
  KeyRound,
  LayoutDashboard,
  LoaderCircle,
  MessageSquare,
  Pin,
  Plug,
  RefreshCw,
  Search,
  Settings2,
  ShieldAlert,
  Sparkles,
  SlidersHorizontal,
  Trash2,
  X
} from 'lucide-react'
import { type FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  activateRuntime,
  deactivateRuntime,
  buildRuntime,
  cancelBuild,
  cancelBuildJob,
  cancelDownloadJob,
  dismissDownloadJob,
  dismissFinishedDownloadJobs,
  resumeDownloadJob,
  checkRuntimeUpdates,
  type SourceRuntimeUpdate,
  deleteModel,
  createMcpServer,
  deleteMcpServer,
  deleteRemoteConnection,
  deleteRuntime,
  ensureLlamaEngine,
  ensureSdcppEngine,
  ensureWhisperEngine,
  fetchModelDescription,
  fetchRecommendations,
  fetchManagedSdcppStatus,
  fetchManagedWhisperStatus,
  fetchManagedLlamaStatus,
  fetchModelTrust,
  assembleSdcppBundle,
  acceptSdcppLicense,
  deleteSdcppBundle,
  formatBytes,
  listSdcppBundles,
  queueSdcppInstall,
  resolveBundleVariants,
  queueSnapshotDownload,
  saveSdcppBundle,
  clearHuggingFaceToken,
  huggingFaceTokenStatus,
  fetchToolchainStatus,
  type ToolchainTool,
  listDownloadJobs,
  listRemoteConnections,
  saveRemoteConnection,
  testRemoteConnection,
  type RemoteConnection,
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
  type Recommendations,
  type RuntimeEntry,
  type RuntimeSettings,
  type RuntimeTarget,
  type VllmModelSettings,
  saveRuntimeSettings,
  saveSupportBundle,
  saveWorkspacePreference,
  searchHub,
  setHuggingFaceToken,
  updateRecommendationState,
  queueModelDownload,
  refreshMcpServer,
  updateMcpServer,
  fetchComputerPermissions,
  fetchComputerUsePreference,
  requestComputerPermissions,
  saveComputerUsePreference,
  fetchWorkspacePreference,
  type ComputerUsePreference,
  type OsPermissionStatus,
  type SdcppBundle,
  type SdcppProposal,
  type WorkspaceModesPreference,
  deleteMemory,
  fetchMemoryPreference,
  listMemories,
  saveMemoryPreference,
  updateMemory,
  type DreamingMode,
  type MemoryPreference
} from '../api'
import type { Memory } from '../types'
import {
  fetchAgentCapabilities,
  fetchAgentPreference,
  fetchAgentTools,
  saveAgentPreference,
  type AgentRuntimeInfo,
  type AgentToolCatalogEntry
} from '../agentApi'
import { RecommendedModels } from './RecommendedModels'
import { CapabilityIcons, capabilityFlags, hubCapabilityFlags } from './CapabilityIcons'
import {
  engineBadgeClass,
  engineLabel,
  modelEngine,
  modelLibraryKey,
  runtimeNoticeForModel,
  modelDisplayName,
  runtimesForModel
} from '../model-utils'
import type { HubModel } from '../types'
import {
  MANAGED_FARA_BUNDLES,
  modelIdForManagedFara,
  type ManagedFaraBundle
} from '../../../computer/recipes'

type DiscoverEngine =
  | 'llama.cpp'
  | 'mlx-lm'
  | 'mlx-vlm'
  | 'whisper.cpp'
  | 'streaming-asr'
  | 'stable-diffusion.cpp'
  | 'personaplex'
type InputGuardStatus = Awaited<ReturnType<Window['brazier']['computer']['inputGuardStatus']>>
type BuildEngine =
  | 'llama.cpp'
  | 'mlx-lm'
  | 'mlx-vlm'
  | 'vllm'
  | 'whisper.cpp'
  | 'streaming-asr'
  | 'stable-diffusion.cpp'
  | 'personaplex'
  | 'personaplex-mlx'
  | 'whisperkit'

const DISCOVER_ENGINE_HELP: Record<DiscoverEngine, string> = {
  'llama.cpp': 'GGUF weights for llama.cpp on CPU, CUDA, Metal, or Vulkan.',
  'mlx-lm': 'Text-only MLX models for Apple Silicon (chat, tools, reasoning).',
  'mlx-vlm': 'Vision MLX models for Apple Silicon (image + text input).',
  'whisper.cpp': 'Whisper speech-to-text weights (ggml/gguf) for local audio transcription.',
  'streaming-asr':
    'Nemotron ASR Streaming snapshots for low-latency chunked transcription (Transformers).',
  'stable-diffusion.cpp':
    'Image (SD/Flux/Qwen) and video (Wan/LTX) checkpoints for sd-cli generation.',
  personaplex: 'PersonaPlex / Moshi speech-to-speech snapshots for realtime Voice mode.'
}

const DISCOVER_ENGINE_LABELS: Record<DiscoverEngine, string> = {
  'llama.cpp': 'GGUF · llama.cpp',
  'mlx-lm': 'MLX · text',
  'mlx-vlm': 'MLX · vision',
  'whisper.cpp': 'ASR · whisper.cpp',
  'streaming-asr': 'ASR · streaming',
  'stable-diffusion.cpp': 'Gen · sd.cpp',
  personaplex: 'Voice · PersonaPlex'
}

/** Labels for build offers (includes engines that are not download categories). */
const BUILD_ENGINE_LABELS: Record<BuildEngine, string> = {
  ...DISCOVER_ENGINE_LABELS,
  vllm: 'Language · vLLM (experimental)',
  'personaplex-mlx': 'Voice · PersonaPlex MLX',
  whisperkit: 'ASR · WhisperKit'
}

// stable-diffusion.cpp intentionally has an unstable CLI. Keep this in sync
// with the managed release pin in crates/brazier-runtime/src/sdcpp.rs; the
// build form remains editable for users who deliberately choose another revision.
const SDCPP_SOURCE_REVISION = 'ea7f0c87cfe4c673263b4c201c596c7f1cbe2528'

const BUILD_ENGINE_DEFAULTS: Record<
  BuildEngine,
  { repository: string; revision: string }
> = {
  'llama.cpp': {
    repository: 'https://github.com/ggml-org/llama.cpp',
    revision: ''
  },
  'mlx-lm': {
    repository: 'https://github.com/ml-explore/mlx-lm',
    revision: 'main'
  },
  'mlx-vlm': {
    repository: 'https://github.com/Blaizzy/mlx-vlm',
    revision: 'main'
  },
  vllm: {
    repository: 'https://github.com/vllm-project/vllm',
    revision: 'main'
  },
  'whisper.cpp': {
    repository: 'https://github.com/ggml-org/whisper.cpp',
    revision: 'master'
  },
  'streaming-asr': {
    repository: 'https://github.com/huggingface/transformers',
    revision: 'bundled'
  },
  'stable-diffusion.cpp': {
    repository: 'https://github.com/leejet/stable-diffusion.cpp',
    revision: SDCPP_SOURCE_REVISION
  },
  personaplex: {
    repository: 'https://github.com/NVIDIA/personaplex',
    revision: 'main'
  },
  'personaplex-mlx': {
    repository: 'https://github.com/mu-hashmi/personaplex-mlx',
    revision: 'main'
  },
  whisperkit: {
    repository: 'https://github.com/argmaxinc/argmax-oss-swift',
    revision: 'main'
  }
}

export function sourceRuntimeId(engine: BuildEngine, buildId: string): string {
  switch (engine) {
    case 'llama.cpp':
      return `source-${buildId}`
    case 'stable-diffusion.cpp':
      return `sdcpp-source-${buildId}`
    case 'whisper.cpp':
      return `whisper-source-${buildId}`
    case 'whisperkit':
      return `whisperkit-source-${buildId}`
    default:
      return `${engine}-source-${buildId}`
  }
}

function defaultDiscoverEngine(hardware: HardwareInfo | null): DiscoverEngine {
  if (hardware?.os === 'macos' && hardware.architecture === 'aarch64') {
    return 'mlx-lm'
  }
  return 'llama.cpp'
}

type QuantFit = 'gpu' | 'offload' | 'system' | 'none' | 'unknown'

/**
 * A search result is a weight file, not a promise that it can be fully
 * offloaded. Keep the labels conservative: reserve headroom for the KV cache,
 * runtime allocations, and the desktop rather than comparing raw bytes.
 *
 * Multi-component bundles (stable-diffusion.cpp) stage their encoders,
 * denoiser, and VAE as separate phases, streaming each component's weights
 * through VRAM, so the whole bundle never needs to be resident in GPU memory
 * at once. For those, the GPU fit is judged against the diffusion checkpoint —
 * the one component used on every step — with the rest counting as staged
 * offload. Single weight files pass `diffusionBytes` equal to `bytes`, which
 * keeps the original all-resident behaviour.
 */
function generationFit(
  bytes: number | null | undefined,
  hardware: HardwareInfo | null,
  diffusionBytes = bytes
): QuantFit {
  if (bytes == null || !hardware) return 'unknown'
  // `gpu_offload_memory_bytes` is the placement budget used by the runtime.
  // Prefer it so AMD systems are not accidentally assessed against all RAM.
  const gpu = hardware.gpu_offload_memory_bytes ?? hardware.vram_bytes
  const system = hardware.memory_bytes
  if (gpu != null) {
    if (bytes <= gpu * 0.7) return 'gpu'
    // The denoiser must be resident (it runs every step); encoders and VAE
    // stream through VRAM for their own phases, so only it constrains a GPU
    // fit. Its activation buffers vary sharply with resolution and video
    // frames, so this is still a staged-offload fit, not a guaranteed green
    // resident-GPU fit.
    if (diffusionBytes != null && diffusionBytes <= gpu * 0.7 && system != null) {
      return 'offload'
    }
    if (system != null && bytes <= system * 0.6) return 'offload'
    return 'none'
  }
  // A detected discrete GPU with no readable VRAM must not be reported as a
  // green system-memory fit. That would hide the actual GPU constraint.
  // Apple Silicon and integrated GPUs (AMD APU, Intel iGPU) intentionally have
  // no separate VRAM figure: their GPU uses unified memory, which is the
  // correct budget to assess here.
  if (
    hardware.os !== 'macos' &&
    hardware.gpu &&
    !hardware.amd_apu &&
    !hardware.intel_igpu
  )
    return 'unknown'
  if (system != null && bytes <= system * 0.6) return 'system'
  return system == null ? 'unknown' : 'none'
}

/**
 * One downloadable quantisation of a model. Split GGUFs publish one quant as
 * several `-00001-of-0000N` shards; they are a single thing to pick and their
 * size is the sum of the shards.
 */
export type QuantGroup = {
  /** File path with any shard suffix stripped — unique per quantisation. */
  key: string
  /** Every file of the quant, in shard order. */
  files: HubFile[]
  /** Summed size across all files, or null when any part's size is unknown. */
  size: number | null
}

function generationFitLabel(fit: QuantFit): string {
  switch (fit) {
    case 'gpu': return 'Fits in GPU memory'
    case 'offload': return 'Fits with staged offload'
    case 'system': return 'Fits in system memory'
    case 'none': return 'Likely too large for this machine'
    default: return 'Memory estimate unavailable'
  }
}

function quantGroup(path: string): string {
  return path.replace(/-\d{1,5}-of-\d{1,5}(?=\.gguf$)/i, '')
}

/** Group a repository's quant files into downloadable quantisations. */
export function groupQuants(files: HubFile[]): QuantGroup[] {
  const byGroup = new Map<string, HubFile[]>()
  for (const file of files) {
    const key = quantGroup(file.path)
    const group = byGroup.get(key)
    if (group) {
      group.push(file)
    } else {
      byGroup.set(key, [file])
    }
  }
  return [...byGroup.entries()].map(([key, groupFiles]) => {
    // Shard order matters for the download: llama.cpp discovers siblings
    // from the first shard's name.
    const sorted = [...groupFiles].sort((left, right) => left.path.localeCompare(right.path))
    let size: number | null = 0
    for (const file of sorted) {
      if (file.size == null) {
        size = null
        break
      }
      size += file.size
    }
    return { key, files: sorted, size }
  })
}

/** Display name for a group: the file name a single-file quant would have. */
export function quantGroupName(group: QuantGroup): string {
  // The key keeps the original extension, so a sharded quant reads like the
  // single file it stands in for and a whisper `.bin` keeps its own name.
  return group.key.split('/').at(-1) ?? group.key
}

export function sortQuantGroups(
  groups: QuantGroup[],
  hardware: HardwareInfo | null
): QuantGroup[] {
  const rank: Record<QuantFit, number> = { gpu: 0, system: 0, offload: 1, unknown: 2, none: 3 }
  return [...groups].sort((left, right) => {
    const fit =
      rank[generationFit(left.size, hardware)] - rank[generationFit(right.size, hardware)]
    return fit || (right.size ?? 0) - (left.size ?? 0) || left.key.localeCompare(right.key)
  })
}

export type ManageSection =
  | 'library'
  | 'recommended'
  | 'discover'
  | 'runtimes'
  | 'websearch'
  | 'engine'
  | 'server'
  | 'mcp'
  | 'agent'
  | 'computer'
  | 'remote'
  | 'chat'
  | 'customization'
  | 'support'

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
  /** Open a model's advanced configuration. */
  onConfigureModel?: (modelId: string) => void
  /** How many settings each model already carries. */
  profileCounts?: Record<string, number>
  pendingBuild?: { engine: BuildEngine; repository: string } | null
  onPendingBuildConsumed?: () => void
  /** Fired after workspace mode toggles are saved. */
  onWorkspaceModesChange?: (modes: WorkspaceModesPreference) => void
  /** The user asked Settings to run a dreaming pass with the current model. */
  onDreamRequest?: () => void
}

const SECTIONS: Array<{ id: ManageSection; label: string; icon: React.JSX.Element }> = [
  { id: 'library', label: 'Model library', icon: <Box size={15} /> },
  { id: 'recommended', label: 'Recommended models', icon: <Sparkles size={15} /> },
  { id: 'discover', label: 'Download models', icon: <Download size={15} /> },
  { id: 'runtimes', label: 'Runtimes', icon: <Cpu size={15} /> },
  { id: 'websearch', label: 'Web search', icon: <Search size={15} /> },
  { id: 'mcp', label: 'MCP servers', icon: <Plug size={15} /> },
  { id: 'agent', label: 'Agent', icon: <Bot size={15} /> },
  { id: 'computer', label: 'Computer use', icon: <LayoutDashboard size={15} /> },
  { id: 'remote', label: 'Remote servers', icon: <Globe size={15} /> },
  { id: 'engine', label: 'Engine configuration', icon: <Settings2 size={15} /> },
  { id: 'server', label: 'OpenAI server', icon: <KeyRound size={15} /> },
  { id: 'chat', label: 'Chat', icon: <MessageSquare size={15} /> },
  { id: 'customization', label: 'Customization', icon: <LayoutDashboard size={15} /> },
  { id: 'support', label: 'Support', icon: <ShieldAlert size={15} /> }
]

type McpRecipe = {
  id: string
  name: string
  command: string
  args: string[]
  description: string
  setup: string
}

/**
 * Well-known, local stdio server configurations. Applying a recipe only fills
 * the editable form; it never installs or starts third-party code on its own.
 */
const MCP_RECIPES: McpRecipe[] = [
  {
    id: 'blender',
    name: 'Blender bridge',
    command: 'blender-mcp-server',
    args: [],
    description: 'Control an open Blender scene: create objects, edit materials, render, and export.',
    setup: 'Install blender-mcp-server and enable its Blender MCP Bridge add-on; the bridge must be listening locally.'
  }
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
  if (event.phase === 'resolve') return 'Resolving release…'
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

export function appendBuildDiagnostics(
  lines: string[],
  diagnostics: Record<string, unknown> | undefined
): string[] {
  if (!diagnostics) return lines
  const hints = diagnostics.hints
  const excerpt = diagnostics.log_excerpt
  let next = [...lines]
  if (typeof excerpt === 'string' && excerpt.trim()) {
    const excerptLines = excerpt.trimEnd().split('\n')
    // Build output is streamed live before the terminal failure event includes
    // its full tail. Replace the overlapping streamed tail with that complete
    // excerpt instead of displaying the same output a second time.
    const recentStart = Math.max(0, next.length - 100)
    const overlap = excerptLines
      .filter((line) => line.length > 0)
      .map((line) => next.lastIndexOf(line))
      .find((index) => index >= recentStart)
    if (overlap != null && overlap >= recentStart) {
      next = [...next.slice(0, overlap), ...excerptLines]
    } else {
      next.push('', '--- last log lines ---', ...excerptLines)
    }
  }
  if (Array.isArray(hints) && hints.length > 0) {
    next.push('', 'Suggested fixes:')
    for (const hint of hints) {
      if (typeof hint === 'string') next.push(`• ${hint}`)
    }
  }
  return next
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
  active,
  showLog = true
}: {
  progress: JobProgressState
  active: boolean
  showLog?: boolean
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
      {showLog && progress.logLines.length > 0 && (
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
        <header className="manage-header">
          <span className="manage-header-title">Manage</span>
          <button
            className="manage-close"
            onClick={props.onClose}
            title="Close"
            aria-label="Close"
          >
            <X size={16} />
          </button>
        </header>
        <div className="manage-body">
          <nav className="manage-nav">
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
            {props.section === 'recommended' && (
              <RecommendedSection {...props} onError={setError} />
            )}
            {props.section === 'discover' && <DiscoverSection {...props} onError={setError} />}
            {props.section === 'runtimes' && <RuntimesSection {...props} onError={setError} />}
            {props.section === 'websearch' && <WebSearchSection {...props} onError={setError} />}
            {props.section === 'mcp' && <McpSection {...props} onError={setError} />}
            {props.section === 'agent' && <AgentSection {...props} onError={setError} />}
            {props.section === 'computer' && (
              <ComputerUseSection {...props} onError={setError} />
            )}
            {props.section === 'remote' && <RemoteSection {...props} onError={setError} />}
            {props.section === 'engine' && <EngineSection {...props} onError={setError} />}
            {props.section === 'server' && <ServerSection {...props} onError={setError} />}
            {props.section === 'chat' && <ChatSection {...props} onError={setError} />}
            {props.section === 'customization' && (
              <CustomizationSection {...props} onError={setError} />
            )}
            {props.section === 'support' && <SupportSection {...props} onError={setError} />}
          </div>
        </div>
      </aside>
    </div>
  )
}

type SectionProps = ManagePanelProps & { onError: (message: string | null) => void }

const DEFAULT_WORKSPACE_MODES: WorkspaceModesPreference = {
  chat: true,
  agent: true,
  generate: true,
  voice: true,
  computer: false
}

const MODE_DESCRIPTIONS: Record<
  keyof WorkspaceModesPreference,
  { label: string; detail: string }
> = {
  chat: { label: 'Chat', detail: 'Private conversations with local chat models.' },
  agent: { label: 'Agent', detail: 'Workspace tasks that edit files and run commands.' },
  generate: { label: 'Generate', detail: 'Image and video generation with sd.cpp models.' },
  voice: { label: 'Voice (alpha)', detail: 'Realtime speech with PersonaPlex voice models.' },
  computer: {
    label: 'Computer (beta)',
    detail: 'Screenshot-driven computer use with Fara-style action models.'
  }
}

function permissionStateLabel(state: OsPermissionStatus['screen_capture']): string {
  switch (state) {
    case 'granted':
      return 'Granted'
    case 'missing':
      return 'Missing'
    case 'unsupported':
      return 'Unsupported'
    default:
      return 'Unknown'
  }
}

function InputGuardSetup(props: { onError: (message: string | null) => void }): React.JSX.Element {
  const [status, setStatus] = useState<InputGuardStatus | null>(null)
  const [installing, setInstalling] = useState(false)

  useEffect(() => {
    let cancelled = false
    void window.brazier.computer.inputGuardStatus()
      .then((next) => {
        if (!cancelled) setStatus(next)
      })
      .catch((cause) => {
        if (!cancelled) {
          setStatus({
            supported: true,
            installed: false,
            secure: false,
            ready: false,
            current: false,
            version: null,
            detail: errorText(cause)
          })
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  async function install(): Promise<void> {
    setInstalling(true)
    props.onError(null)
    try {
      setStatus(await window.brazier.computer.setupInputGuard())
    } catch (cause) {
      props.onError(errorText(cause))
      try {
        setStatus(await window.brazier.computer.inputGuardStatus())
      } catch {
        // Keep the prior status when the local probe itself is unavailable.
      }
    } finally {
      setInstalling(false)
    }
  }

  return (
    <div>
      <p className="model-help">
        The desktop portal is preferred. This small privileged watcher is used only when the
        compositor cannot activate the global shortcut. It runs only during Computer Use,
        recognizes Ctrl+Shift+Esc, and never reports individual keys.
      </p>
      <dl className="customization-permissions">
        <div>
          <dt>Status</dt>
          <dd>
            {status?.ready && status.current
              ? 'Ready'
              : status?.ready
                ? 'Update available'
                : status?.installed
                  ? 'Needs repair'
                  : 'Not installed'}
          </dd>
        </div>
        {status?.version ? (
          <div>
            <dt>Version</dt>
            <dd>{status.version}</dd>
          </div>
        ) : null}
        <div>
          <dt>Detail</dt>
          <dd>{status?.detail ?? 'Checking the local safety fallback…'}</dd>
        </div>
      </dl>
      <button
        type="button"
        disabled={installing}
        onClick={() => void install()}
        style={{ marginTop: 12 }}
      >
        {installing ? (
          <LoaderCircle className="spin" size={14} />
        ) : (
          <ShieldAlert size={14} />
        )}
        {installing
          ? 'Waiting for administrator approval…'
          : status?.ready && status.current
            ? 'Reinstall safety fallback'
            : status?.installed
              ? 'Repair safety fallback'
              : 'Install safety fallback'}
      </button>
    </div>
  )
}

function ChatSection(props: SectionProps): React.JSX.Element {
  const [preference, setPreference] = useState<MemoryPreference | null>(null)
  const [memories, setMemories] = useState<Memory[]>([])
  const [filter, setFilter] = useState('')
  const [loadingMemories, setLoadingMemories] = useState(true)
  const [savingPreference, setSavingPreference] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editText, setEditText] = useState('')
  const [busyId, setBusyId] = useState<string | null>(null)

  const loadMemories = useCallback(async (): Promise<void> => {
    setLoadingMemories(true)
    try {
      setMemories(await listMemories())
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setLoadingMemories(false)
    }
  }, [props])

  useEffect(() => {
    let cancelled = false
    void fetchMemoryPreference()
      .then((preference) => {
        if (!cancelled) setPreference(preference)
      })
      .catch((cause) => {
        if (!cancelled) props.onError(errorText(cause))
      })
    void listMemories()
      .then((memories) => {
        if (!cancelled) setMemories(memories)
      })
      .catch((cause) => {
        if (!cancelled) props.onError(errorText(cause))
      })
      .finally(() => {
        if (!cancelled) setLoadingMemories(false)
      })
    return () => {
      cancelled = true
    }
    // Load once when the section mounts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function savePreference(patch: Partial<MemoryPreference>): Promise<void> {
    if (!preference) return
    props.onError(null)
    setSavingPreference(true)
    try {
      const saved = await saveMemoryPreference({ ...preference, ...patch })
      setPreference(saved)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setSavingPreference(false)
    }
  }

  async function applyMemoryMutation(
    action: () => Promise<void>
  ): Promise<void> {
    props.onError(null)
    try {
      await action()
      await loadMemories()
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  const visibleMemories = useMemo(() => {
    const needle = filter.trim().toLowerCase()
    if (!needle) return memories
    return memories.filter(
      (memory) =>
        memory.text.toLowerCase().includes(needle) || memory.tags.some((tag) => tag.toLowerCase().includes(needle))
    )
  }, [memories, filter])

  return (
    <section>
      <header className="manage-heading">
        <h2>Chat</h2>
        <p>
          Long-term memory the chat model can draw on and write to, plus how
          often it reflects on what it has learned.
        </p>
      </header>

      <div className="settings-group">
        <div className="section-label">Chat memory</div>
        <p className="model-help">
          When on, relevant memories are injected into each chat turn and the
          model can save and recall them with <code>save_memory</code> and{' '}
          <code>recall_memory</code>. Incognito conversations never read or
          write memories.
        </p>
        <label className="settings-toggle">
          <input
            type="checkbox"
            checked={preference?.enabled ?? true}
            disabled={!preference || savingPreference}
            onChange={(event) => void savePreference({ enabled: event.target.checked })}
          />
          <span>
            <strong>Remember across conversations</strong>
            <small>Let the model save and use long-term memory in chat.</small>
          </span>
        </label>
        {preference?.enabled && (
          <div className="settings-grid">
            <label>
              <span>Memories recalled per turn</span>
              <input
                type="number"
                min={0}
                max={100}
                value={preference.recall_count}
                disabled={savingPreference}
                onChange={(event) =>
                  void savePreference({ recall_count: Math.max(0, Number(event.target.value) || 0) })
                }
              />
            </label>
            <label>
              <span>Recalled characters per turn</span>
              <input
                type="number"
                min={0}
                max={40000}
                value={preference.recall_chars}
                disabled={savingPreference}
                onChange={(event) =>
                  void savePreference({ recall_chars: Math.max(0, Number(event.target.value) || 0) })
                }
              />
            </label>
          </div>
        )}
      </div>

      <div className="settings-group">
        <div className="section-label">Dreaming</div>
        <p className="model-help">
          While the app is idle, the model reviews recent conversations and the
          memory store — merging duplicates, pruning stale facts, and adding
          what the conversations revealed. Auto runs it silently; ask prompts
          first; off never runs it.
        </p>
        <div className="toggle-list">
          {(
            [
              ['off', 'Off', 'Never run dreaming.'],
              ['auto', 'Auto', 'Run silently while idle after a conversation.'],
              ['ask', 'Ask', 'Ask before running a dreaming pass.']
            ] as Array<[DreamingMode, string, string]>
          ).map(([mode, label, detail]) => (
            <label key={mode}>
              <div>
                <strong>{label}</strong>
                <span>{detail}</span>
              </div>
              <input
                type="radio"
                name="dreaming-mode"
                checked={preference?.dreaming === mode}
                disabled={!preference || savingPreference}
                onChange={() => void savePreference({ dreaming: mode })}
              />
            </label>
          ))}
        </div>
        <div className="runtime-actions">
          <button
            className="primary-action"
            disabled={!props.selectedModel}
            onClick={() => props.onDreamRequest?.()}
          >
            Run dreaming now
          </button>
        </div>
      </div>

      <div className="settings-group">
        <div className="settings-group-head">
          <div className="section-label">Memories</div>
          <input
            className="memory-search"
            type="search"
            placeholder="Search memories…"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          />
        </div>
        <p className="model-help">
          Everything the model has saved. Edit or delete any of it; pinned
          memories are protected from dreaming's pruning.
        </p>
        {loadingMemories ? (
          <div className="manage-placeholder">Loading memories…</div>
        ) : visibleMemories.length === 0 ? (
          <div className="manage-placeholder">
            {memories.length === 0 ? 'No memories yet.' : 'No memories match that search.'}
          </div>
        ) : (
          <ul className="memory-list">
            {visibleMemories.map((memory) =>
              editingId === memory.id ? (
                <li className="memory-row memory-row-editing" key={memory.id}>
                  <textarea
                    value={editText}
                    autoFocus
                    rows={2}
                    onChange={(event) => setEditText(event.target.value)}
                  />
                  <div className="memory-row-actions">
                    <button
                      className="secondary-action"
                      disabled={busyId === memory.id}
                      onClick={() => {
                        setBusyId(memory.id)
                        void applyMemoryMutation(async () => {
                          await updateMemory(memory.id, { text: editText.trim() })
                        }).finally(() => {
                          setEditingId(null)
                          setBusyId(null)
                        })
                      }}
                    >
                      Save
                    </button>
                    <button
                      className="secondary-action"
                      disabled={busyId === memory.id}
                      onClick={() => {
                        setEditingId(null)
                        setEditText('')
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                </li>
              ) : (
                <li className="memory-row" key={memory.id}>
                  <div className="memory-row-text">
                    {memory.pinned && <Pin className="memory-pin" size={12} />}
                    <span>{memory.text}</span>
                    <small>
                      {memory.kind}
                      {memory.tags.length > 0 && ` · ${memory.tags.join(', ')}`}
                      {` · ${new Date(memory.updated_at).toLocaleDateString()}`}
                    </small>
                  </div>
                  <div className="memory-row-actions">
                    <button
                      className="secondary-action"
                      disabled={busyId === memory.id}
                      title={memory.pinned ? 'Unpin' : 'Pin'}
                      onClick={() => {
                        setBusyId(memory.id)
                        void applyMemoryMutation(() =>
                          updateMemory(memory.id, { pinned: !memory.pinned }).then(() => undefined)
                        ).finally(() => setBusyId(null))
                      }}
                    >
                      <Pin size={13} />
                    </button>
                    <button
                      className="secondary-action"
                      disabled={busyId === memory.id}
                      title="Edit"
                      onClick={() => {
                        setEditingId(memory.id)
                        setEditText(memory.text)
                      }}
                    >
                      Edit
                    </button>
                    <button
                      className="secondary-action danger-action"
                      disabled={busyId === memory.id}
                      title="Delete"
                      onClick={() => {
                        setBusyId(memory.id)
                        void applyMemoryMutation(() => deleteMemory(memory.id).then(() => undefined)).finally(
                          () => setBusyId(null)
                        )
                      }}
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </li>
              )
            )}
          </ul>
        )}
      </div>
    </section>
  )
}

function CustomizationSection(props: SectionProps): React.JSX.Element {
  const [modes, setModes] = useState<WorkspaceModesPreference>(DEFAULT_WORKSPACE_MODES)
  const [modesLoading, setModesLoading] = useState(true)
  const [savingModes, setSavingModes] = useState(false)
  const [updateSettings, setUpdateSettings] = useState<{
    supported: boolean
    checkOnStartup: boolean
    autoDownload: boolean
  } | null>(null)
  const [checkingUpdates, setCheckingUpdates] = useState(false)
  const [permissions, setPermissions] = useState<OsPermissionStatus | null>(null)
  const [requestingComputerPermissions, setRequestingComputerPermissions] = useState(false)
  const [computerPreference, setComputerPreference] = useState<ComputerUsePreference | null>(null)
  const [savingComputerPreference, setSavingComputerPreference] = useState(false)

  useEffect(() => {
    let cancelled = false
    setModesLoading(true)
    void fetchWorkspacePreference()
      .then((result) => {
        if (!cancelled) setModes(result.modes)
      })
      .catch((cause) => {
        if (!cancelled) props.onError(errorText(cause))
      })
      .finally(() => {
        if (!cancelled) setModesLoading(false)
      })

    const brazier = window.brazier as Window['brazier'] & {
      getUpdateSettings?: () => Promise<{
        supported: boolean
        checkOnStartup: boolean
        autoDownload: boolean
      }>
    }
    void brazier.getUpdateSettings?.()
      .then((settings) => {
        if (!cancelled) setUpdateSettings(settings)
      })
      .catch(() => {
        if (!cancelled) {
          setUpdateSettings({ supported: false, checkOnStartup: true, autoDownload: false })
        }
      })

    void fetchComputerPermissions()
      .then((status) => {
        if (!cancelled) setPermissions(status)
      })
      .catch(() => {
        if (!cancelled) setPermissions(null)
      })

    void fetchComputerUsePreference()
      .then((preference) => {
        if (!cancelled) setComputerPreference(preference)
      })
      .catch((cause) => {
        if (!cancelled) props.onError(errorText(cause))
      })

    return () => {
      cancelled = true
    }
    // Load once when the section mounts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function toggleMode(key: keyof WorkspaceModesPreference, on: boolean): Promise<void> {
    const next = { ...modes, [key]: on }
    const enabled = (Object.keys(next) as Array<keyof WorkspaceModesPreference>).filter(
      (mode) => next[mode]
    )
    if (enabled.length === 0) {
      props.onError('Keep at least one workspace mode enabled.')
      return
    }
    props.onError(null)
    setModes(next)
    setSavingModes(true)
    try {
      const saved = await saveWorkspacePreference(next)
      setModes(saved.modes)
      props.onWorkspaceModesChange?.(saved.modes)
    } catch (cause) {
      setModes(modes)
      props.onError(errorText(cause))
    } finally {
      setSavingModes(false)
    }
  }


  async function saveUpdates(patch: {
    checkOnStartup?: boolean
    autoDownload?: boolean
  }): Promise<void> {
    props.onError(null)
    try {
      const brazier = window.brazier as Window['brazier'] & {
        saveUpdateSettings?: (settings: {
          checkOnStartup?: boolean
          autoDownload?: boolean
        }) => Promise<{
          supported: boolean
          checkOnStartup: boolean
          autoDownload: boolean
        }>
      }
      if (!brazier.saveUpdateSettings) {
        props.onError('Update settings are not available in this build.')
        return
      }
      const saved = await brazier.saveUpdateSettings(patch)
      setUpdateSettings(saved)
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  async function checkUpdates(): Promise<void> {
    setCheckingUpdates(true)
    props.onError(null)
    try {
      const result = await window.brazier.checkForUpdates()
      if (!result.supported) {
        props.onError('App updates are not available for this installation.')
      }
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setCheckingUpdates(false)
    }
  }

  async function requestComputerAccess(): Promise<void> {
    setRequestingComputerPermissions(true)
    props.onError(null)
    try {
      const [nextPermissions] = await Promise.all([
        requestComputerPermissions(),
        window.brazier.computer.prepareSafety()
      ])
      setPermissions(nextPermissions)
    } catch (cause) {
      props.onError(errorText(cause))
      try {
        setPermissions(await fetchComputerPermissions())
      } catch {
        // Keep the last known status when the portal itself could not reply.
      }
    } finally {
      setRequestingComputerPermissions(false)
    }
  }

  async function saveActionSettleDelay(milliseconds: number): Promise<void> {
    const action_settle_delay_ms = Math.max(0, Math.min(10_000, Math.round(milliseconds)))
    const previous = computerPreference
    const next = { action_settle_delay_ms, max_screenshots_kept: computerPreference?.max_screenshots_kept ?? 3 }
    setComputerPreference(next)
    setSavingComputerPreference(true)
    props.onError(null)
    try {
      setComputerPreference(await saveComputerUsePreference(next))
    } catch (cause) {
      setComputerPreference(previous)
      props.onError(errorText(cause))
    } finally {
      setSavingComputerPreference(false)
    }
  }

  async function saveScreenshotsKept(count: number): Promise<void> {
    const max_screenshots_kept = Math.max(1, Math.min(20, Math.round(count)))
    const previous = computerPreference
    const next = {
      action_settle_delay_ms: computerPreference?.action_settle_delay_ms ?? 750,
      max_screenshots_kept
    }
    setComputerPreference(next)
    setSavingComputerPreference(true)
    props.onError(null)
    try {
      setComputerPreference(await saveComputerUsePreference(next))
    } catch (cause) {
      setComputerPreference(previous)
      props.onError(errorText(cause))
    } finally {
      setSavingComputerPreference(false)
    }
  }

  return (
    <section>
      <header className="manage-heading">
        <h2>Customization</h2>
        <p>Choose which workspace modes appear, how updates behave, and review OS permissions.</p>
      </header>

      <div className="settings-group">
        <div className="section-label">Workspace modes</div>
        <p className="model-help">
          Hide modes you do not use. At least one mode must stay on.
        </p>
        {modesLoading ? (
          <p className="model-help">Loading preferences…</p>
        ) : (
          <div className="toggle-list">
            {(Object.keys(MODE_DESCRIPTIONS) as Array<keyof WorkspaceModesPreference>).map(
              (key) => (
                <label key={key}>
                  <div>
                    <strong>{MODE_DESCRIPTIONS[key].label}</strong>
                    <span>{MODE_DESCRIPTIONS[key].detail}</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={modes[key]}
                    disabled={savingModes}
                    onChange={(event) => void toggleMode(key, event.target.checked)}
                  />
                </label>
              )
            )}
          </div>
        )}
      </div>

      <div className="settings-group">
        <div className="section-label">App updates</div>
        <p className="model-help">
          Checks use the signed GitHub release feed. Downloads still ask before installing unless
          you enable auto-download.
        </p>
        <div className="toggle-list">
          <label>
            <div>
              <strong>Check on startup</strong>
              <span>Look for a newer Brazier build when the app launches.</span>
            </div>
            <input
              type="checkbox"
              checked={updateSettings?.checkOnStartup ?? true}
              disabled={!updateSettings}
              onChange={(event) => void saveUpdates({ checkOnStartup: event.target.checked })}
            />
          </label>
          <label>
            <div>
              <strong>Auto-download updates</strong>
              <span>Download signed updates without an extra confirmation prompt.</span>
            </div>
            <input
              type="checkbox"
              checked={updateSettings?.autoDownload ?? false}
              disabled={!updateSettings}
              onChange={(event) => void saveUpdates({ autoDownload: event.target.checked })}
            />
          </label>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 12 }}>
          <button
            type="button"
            disabled={checkingUpdates || updateSettings?.supported === false}
            onClick={() => void checkUpdates()}
          >
            {checkingUpdates ? (
              <LoaderCircle className="spin" size={14} />
            ) : (
              <RefreshCw size={14} />
            )}
            {checkingUpdates ? 'Checking…' : 'Check for updates'}
          </button>
          {updateSettings && !updateSettings.supported ? (
            <span className="model-help">Updates are disabled for this installation.</span>
          ) : null}
        </div>
      </div>

      <div className="settings-group">
        <div className="section-label">Computer Use permissions</div>
        <p className="model-help">
          Desktop capture, input injection, the always-visible safety overlay, and the global
          emergency shortcut are all required. Browser target does not need these OS grants.
        </p>
        <label className="model-field" style={{ maxWidth: 360, marginBottom: 16 }}>
          <span>Action settle delay (milliseconds)</span>
          <input
            type="number"
            min={0}
            max={10_000}
            step={50}
            value={computerPreference?.action_settle_delay_ms ?? 750}
            disabled={!computerPreference || savingComputerPreference}
            onChange={(event) =>
              setComputerPreference({
                action_settle_delay_ms: Number(event.target.value),
                max_screenshots_kept: computerPreference?.max_screenshots_kept ?? 3
              })
            }
            onBlur={(event) => void saveActionSettleDelay(Number(event.target.value))}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.currentTarget.blur()
              }
            }}
          />
          <small>
            Wait after an input action before capturing the next screenshot. Explicit wait actions
            are not delayed again.
          </small>
        </label>
        <label className="model-field" style={{ maxWidth: 360, marginBottom: 16 }}>
          <span>Screenshots kept in history</span>
          <input
            type="number"
            min={1}
            max={20}
            step={1}
            value={computerPreference?.max_screenshots_kept ?? 3}
            disabled={!computerPreference || savingComputerPreference}
            onChange={(event) =>
              setComputerPreference({
                action_settle_delay_ms: computerPreference?.action_settle_delay_ms ?? 750,
                max_screenshots_kept: Number(event.target.value)
              })
            }
            onBlur={(event) => void saveScreenshotsKept(Number(event.target.value))}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.currentTarget.blur()
              }
            }}
          />
          <small>
            How many recent screenshots the model can see on each step. Each costs roughly 1500
            tokens of context; Fara is trained with 3.
          </small>
        </label>
        {permissions ? (
          <dl className="customization-permissions">
            <div>
              <dt>Platform</dt>
              <dd>
                {permissions.platform}
                {permissions.display_server ? ` · ${permissions.display_server}` : ''}
              </dd>
            </div>
            <div>
              <dt>Screen capture</dt>
              <dd>{permissionStateLabel(permissions.screen_capture)}</dd>
            </div>
            <div>
              <dt>Input injection</dt>
              <dd>{permissionStateLabel(permissions.input_injection)}</dd>
            </div>
            {permissions.detail ? (
              <div>
                <dt>Detail</dt>
                <dd>{permissions.detail}</dd>
              </div>
            ) : null}
            {permissions.settings_hint ? (
              <div>
                <dt>Hint</dt>
                <dd>{permissions.settings_hint}</dd>
              </div>
            ) : null}
          </dl>
        ) : (
          <p className="model-help">Could not read OS permission status from the daemon.</p>
        )}
        {window.brazier.platform === 'linux' ? (
          <div style={{ marginTop: 14 }}>
            <div className="section-label">Wayland emergency fallback</div>
            <InputGuardSetup onError={props.onError} />
          </div>
        ) : null}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 12 }}>
          <button
            type="button"
            disabled={requestingComputerPermissions || permissions?.platform === 'windows'}
            onClick={() => void requestComputerAccess()}
          >
            {requestingComputerPermissions ? <LoaderCircle className="spin" size={14} /> : null}
            {requestingComputerPermissions ? 'Waiting for OS approval…' : 'Request desktop access'}
          </button>
          <span className="model-help">
            On Wayland this requests Screen Share, Remote Desktop, and Ctrl+Shift+Esc. X11 and macOS
            use Esc. If Wayland cannot activate the shortcut, install the privileged emergency
            fallback above. Computer Use stays disabled unless one guard is verified.
          </span>
        </div>
      </div>
    </section>
  )
}

function SupportSection(props: SectionProps): React.JSX.Element {
  const [saving, setSaving] = useState(false)
  const [savedPath, setSavedPath] = useState<string | null>(null)

  async function download(): Promise<void> {
    setSaving(true)
    setSavedPath(null)
    props.onError(null)
    try {
      const path = await saveSupportBundle()
      if (path) setSavedPath(path)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setSaving(false)
    }
  }

  return (
    <section>
      <header className="manage-heading">
        <h2>Support</h2>
        <p>Create a diagnostic archive you can inspect before sharing.</p>
      </header>

      <div className="settings-group">
        <div className="section-label">Redacted support bundle</div>
        <p className="model-help">
          Includes engine, runtime, hardware, and toolchain information. Conversations, prompts,
          responses, attachments, credentials, and logs are not included. User-home and Brazier
          data-directory prefixes, URL credentials, and secret-bearing fields are removed.
        </p>
        <div className="runtime-actions">
          <button className="primary-action" disabled={saving} onClick={() => void download()}>
            {saving ? <LoaderCircle className="spin" size={15} /> : <Download size={15} />}
            {saving ? 'Creating bundle…' : 'Download support bundle'}
          </button>
        </div>
        {savedPath && <p className="model-help">Saved to {savedPath}</p>}
      </div>
    </section>
  )
}

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

/**
 * The first-run recommendations, revisited.
 *
 * The welcome flow only offers the categories you picked at the time, and a
 * machine's memory does not change but its owner's mind does — so the whole set
 * is available here, permanently, rather than only in the ten seconds after
 * installing.
 */
function RecommendedSection(props: SectionProps): React.JSX.Element {
  const [recommendations, setRecommendations] = useState<Recommendations | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      setRecommendations(await fetchRecommendations())
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setLoading(false)
    }
  }, [props])

  useEffect(() => {
    void refresh()
    // Refreshed only on mount: sizing these costs a Hub request per model.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <section>
      <header className="manage-heading">
        <h2>Recommended models</h2>
        <p>
          One model per thing you might want to do, chosen for how much memory this machine has
          and at the largest quantisation it can hold comfortably.
        </p>
      </header>

      {loading && !recommendations ? (
        <div className="manage-placeholder">
          <LoaderCircle className="spin" size={18} />
          Sizing recommendations against your memory…
        </div>
      ) : recommendations ? (
        <>
          <RecommendedModels
            recommendations={recommendations}
            onInstalled={() => void props.refreshModels()}
            onError={props.onError}
            onOpenRuntimes={() => props.onSectionChange('runtimes')}
          />
          <div className="settings-group">
            <label className="engine-toggle">
              <input
                type="checkbox"
                checked={recommendations.state.suppressed}
                onChange={(event) => {
                  const suppressed = event.target.checked
                  void updateRecommendationState({ suppressed })
                    .then((state) =>
                      setRecommendations((current) =>
                        current ? { ...current, state } : current
                      )
                    )
                    .catch((cause: unknown) => props.onError(errorText(cause)))
                }}
              />
              <span>
                Don&apos;t tell me when a recommendation changes
                <small>
                  Brazier otherwise mentions it on startup when a category you set up here has a
                  newer suggestion.
                </small>
              </span>
            </label>
          </div>
        </>
      ) : null}
    </section>
  )
}

function ComputerUseSection(props: SectionProps): React.JSX.Element {
  const [jobs, setJobs] = useState<DownloadJob[]>([])
  const [installing, setInstalling] = useState<string | null>(null)
  const [runtimePhase, setRuntimePhase] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const settledSignature = useRef('')

  useEffect(() => {
    let cancelled = false
    async function refresh(): Promise<void> {
      try {
        const next = await listDownloadJobs()
        if (cancelled) return
        setJobs(next)
        const managed = next.filter((job) =>
          MANAGED_FARA_BUNDLES.some((bundle) => bundle.quantRepo === job.repo_id)
        )
        const signature = managed
          .filter((job) => ['completed', 'failed', 'cancelled'].includes(job.status))
          .map((job) => `${job.id}:${job.status}:${job.updated_at}`)
          .join('|')
        if (signature && signature !== settledSignature.current) {
          settledSignature.current = signature
          await props.refreshModels()
        }
      } catch {
        if (!cancelled) setJobs([])
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 2000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [props.refreshModels])

  async function queueBundleFile(bundle: ManagedFaraBundle, filename: string): Promise<void> {
    // The daemon returns newest jobs first. Prefer the latest attempt so an
    // older failed row cannot be resumed alongside a newer active download.
    const existing = jobs.find(
      (job) => job.repo_id === bundle.quantRepo && job.filename === filename
    )
    if (existing && ['pending', 'downloading'].includes(existing.status)) return
    if (existing && ['paused', 'failed', 'cancelled'].includes(existing.status)) {
      await resumeDownloadJob(existing.id)
      return
    }
    await queueModelDownload(bundle.quantRepo, filename)
  }

  async function enableComputerMode(): Promise<void> {
    const preference = await fetchWorkspacePreference()
    if (preference.modes.computer) return
    const saved = await saveWorkspacePreference({ ...preference.modes, computer: true })
    props.onWorkspaceModesChange?.(saved.modes)
  }

  async function installBundle(bundle: ManagedFaraBundle): Promise<void> {
    setInstalling(bundle.id)
    setRuntimePhase('Checking the llama.cpp runtime…')
    setNotice(null)
    props.onError(null)
    try {
      await Promise.all([
        queueBundleFile(bundle, bundle.modelFile),
        queueBundleFile(bundle, bundle.projectorFile),
        ensureLlamaEngine((event) => setRuntimePhase(progressLabel(event))),
        enableComputerMode()
      ])
      setJobs(await listDownloadJobs())
      setNotice(
        `${bundle.label} is in the download queue. Brazier has also prepared its vision projector, inference runtime, and Computer Use workspace mode.`
      )
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setInstalling(null)
      setRuntimePhase(null)
    }
  }

  return (
    <section>
      <header className="manage-heading">
        <h2>Computer use</h2>
        <p>
          Install a ready-to-run model stack. Brazier downloads the model and matching vision
          projector, installs the best managed llama.cpp runtime for this machine, and enables the
          Computer workspace mode.
        </p>
      </header>

      <div className="settings-group">
        <div className="section-label">Fara1.5 models</div>
        <p className="model-help">
          These are balanced Q4_K_M GGUF conversions of Microsoft&apos;s MIT-licensed Fara1.5
          releases. The 4B model is the recommended starting point; larger variants improve
          multi-step reliability but need substantially more memory and compute.
        </p>
        <div className="runtime-offer-list computer-model-offers">
          {MANAGED_FARA_BUNDLES.map((bundle) => {
            const fit = generationFit(bundle.downloadBytes, props.hardware)
            const model = props.models.find((entry) => entry.id === modelIdForManagedFara(bundle))
            const ready = Boolean(model?.capabilities?.computer_use)
            const filenames = new Set([bundle.modelFile, bundle.projectorFile])
            const bundleJobs = jobs.filter(
              (job) => job.repo_id === bundle.quantRepo && filenames.has(job.filename)
            )
            const activeJobs = bundleJobs.filter((job) =>
              ['pending', 'downloading'].includes(job.status)
            )
            const paused = bundleJobs.some((job) => job.status === 'paused')
            const failed = bundleJobs.some((job) => job.status === 'failed')
            const downloaded = activeJobs.reduce(
              (total, job) => total + (job.bytes_downloaded ?? 0),
              0
            )
            const activeTotal = activeJobs.reduce(
              (total, job) => total + (job.total_bytes ?? 0),
              0
            )
            const percent = activeTotal > 0 ? Math.min(100, (downloaded / activeTotal) * 100) : 0
            const busy = installing === bundle.id || activeJobs.length > 0
            return (
              <article className="runtime-offer computer-model-offer" key={bundle.id}>
                <div className="runtime-offer-info">
                  <strong>
                    {bundle.label}
                    {bundle.recommended && <span className="installed-badge">Recommended</span>}
                    {ready && <span className="installed-badge">Ready</span>}
                  </strong>
                  <span>{bundle.summary}</span>
                  <span>
                    {formatBytes(bundle.downloadBytes)} download ·{' '}
                    <span className={`generation-fit ${fit}`}>{generationFitLabel(fit)}</span>
                  </span>
                  <span>
                    Base:{' '}
                    <a
                      href={`https://huggingface.co/${bundle.sourceRepo}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {bundle.sourceRepo}
                    </a>{' '}
                    · Quantization:{' '}
                    <a
                      href={`https://huggingface.co/${bundle.quantRepo}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {bundle.quantRepo}
                    </a>
                  </span>
                  {activeJobs.length > 0 && (
                    <>
                      <span>
                        Downloading {formatBytes(downloaded)}
                        {activeTotal > 0 ? ` / ${formatBytes(activeTotal)}` : ''}
                      </span>
                      <div className="progress-track compact">
                        <div className="progress-fill" style={{ width: `${percent}%` }} />
                      </div>
                    </>
                  )}
                  {!ready && model && activeJobs.length === 0 && (
                    <span>The model is present, but its vision projector still needs setup.</span>
                  )}
                  {!ready && failed && activeJobs.length === 0 && (
                    <span className="run-error-text">A download failed; setup will resume it.</span>
                  )}
                </div>
                <button
                  type="button"
                  className="secondary-action"
                  disabled={ready || busy}
                  onClick={() => void installBundle(bundle)}
                >
                  {installing === bundle.id ? (
                    <LoaderCircle className="spin" size={13} />
                  ) : ready ? (
                    <Check size={13} />
                  ) : (
                    <Download size={13} />
                  )}
                  {ready
                    ? 'Ready'
                    : installing === bundle.id
                      ? 'Preparing…'
                      : activeJobs.length > 0
                        ? 'Downloading…'
                        : paused
                          ? 'Resume setup'
                          : failed
                            ? 'Retry setup'
                            : model
                              ? 'Finish setup'
                              : 'Install all'}
                </button>
              </article>
            )
          })}
        </div>
        {runtimePhase && <p className="model-help">{runtimePhase}</p>}
        {notice && <p className="model-help">{notice}</p>}
      </div>

      {window.brazier.platform === 'linux' ? (
        <div className="settings-group">
          <div className="section-label">Wayland emergency fallback</div>
          <InputGuardSetup onError={props.onError} />
          <button
            type="button"
            className="secondary-action"
            onClick={() => props.onSectionChange('customization')}
            style={{ marginTop: 10 }}
          >
            <Settings2 size={14} />
            Review all desktop permissions
          </button>
        </div>
      ) : null}

      <div className="settings-group">
        <div className="section-label">What Brazier manages</div>
        <p className="model-help">
          Downloads are resumable and remain visible in the global activity tray. The matching
          projector is loaded automatically when the model starts, and the runtime uses the
          hardware target selected under Runtimes. Browser tasks use an installed Chromium-family
          browser in a fresh dedicated profile; desktop control remains opt-in and reports its OS
          permission requirements in Computer Use settings.
        </p>
      </div>
    </section>
  )
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
                <button
                  className="chip-button subtle"
                  title="Advanced settings for this model"
                  onClick={() => props.onConfigureModel?.(model.id)}
                >
                  <SlidersHorizontal size={13} />
                  {(props.profileCounts?.[model.id] ?? 0) > 0 ? 'Configured' : 'Configure'}
                </button>
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
                      title={
                        model.id.startsWith('gguf:')
                          ? 'Delete this model from disk (all shards and an orphaned mmproj)'
                          : model.id.startsWith('mlx:') ||
                              model.id.startsWith('mlx-vlm:') ||
                              model.id.startsWith('streaming-asr:') ||
                              model.id.startsWith('sdcpp-') ||
                              model.id.startsWith('personaplex:')
                            ? 'Delete this model directory from disk'
                            : 'Delete this model from disk'
                      }
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

function VllmServedModels({ settings, onSaved, onError }: { settings: RuntimeSettings; onSaved: (settings: RuntimeSettings) => void; onError: (message: string | null) => void }): React.JSX.Element {
  const [models, setModels] = useState<VllmModelSettings[]>(settings.vllm_models ?? [])
  const [draft, setDraft] = useState<VllmModelSettings>({ repository: '', revision: null, context_size: null, dtype: null, gpu_memory_utilization: null, tensor_parallel_size: null, trust_remote_code: false, prefix_caching: true, extra_args: [] })
  const [saving, setSaving] = useState(false)
  useEffect(() => setModels(settings.vllm_models ?? []), [settings.vllm_models])
  async function persist(next: VllmModelSettings[], active = settings.vllm_model): Promise<void> {
    setSaving(true); onError(null)
    try { const saved = await saveRuntimeSettings({ ...settings, vllm_models: next, vllm_model: active }); setModels(next); onSaved(saved) }
    catch (cause) { onError(errorText(cause)) } finally { setSaving(false) }
  }
  return <div className="settings-group vllm-served-models">
    <div className="section-label">Served models</div>
    <p className="model-help">Each model keeps its own vLLM launch options. Activating or saving the active model restarts only vLLM.</p>
    {models.map((model, index) => <details key={model.repository} className="runtime-card" open={model.repository === settings.vllm_model}>
      <summary><strong>{model.repository}</strong>{model.repository === settings.vllm_model && <span className="active-badge">Active</span>}</summary>
      <div className="settings-grid">
        <label><span>Revision</span><input value={model.revision ?? ''} onChange={(e) => { const next=[...models]; next[index]={...model,revision:e.target.value||null}; setModels(next) }} placeholder="main" /></label>
        <label><span>Context length</span><input type="number" min={512} value={model.context_size ?? ''} onChange={(e)=>{const next=[...models];next[index]={...model,context_size:e.target.value?Number(e.target.value):null};setModels(next)}} /></label>
        <label><span>Precision</span><select value={model.dtype ?? ''} onChange={(e)=>{const next=[...models];next[index]={...model,dtype:e.target.value||null};setModels(next)}}><option value="">Auto</option><option value="auto">Auto</option><option value="bfloat16">bfloat16</option><option value="float16">float16</option><option value="float32">float32</option></select></label>
        <label><span>GPU memory limit</span><input type="number" min={0.1} max={1} step={0.05} value={model.gpu_memory_utilization ?? ''} onChange={(e)=>{const next=[...models];next[index]={...model,gpu_memory_utilization:e.target.value?Number(e.target.value):null};setModels(next)}} placeholder="0.90" /></label>
        <label><span>Tensor parallel GPUs</span><input type="number" min={1} value={model.tensor_parallel_size ?? ''} onChange={(e)=>{const next=[...models];next[index]={...model,tensor_parallel_size:e.target.value?Number(e.target.value):null};setModels(next)}} /></label>
        <label className="settings-toggle"><input type="checkbox" checked={model.trust_remote_code} onChange={(e)=>{const next=[...models];next[index]={...model,trust_remote_code:e.target.checked};setModels(next)}} /><span>Trust remote code</span></label>
        <label className="settings-toggle"><input type="checkbox" checked={model.prefix_caching ?? true} onChange={(e)=>{const next=[...models];next[index]={...model,prefix_caching:e.target.checked};setModels(next)}} /><span>Prefix caching</span></label>
        <label className="span-2"><span>Additional arguments (one token per line)</span><textarea value={model.extra_args.join('\n')} onChange={(e)=>{const next=[...models];next[index]={...model,extra_args:e.target.value.split('\n').map(x=>x.trim()).filter(Boolean)};setModels(next)}} /></label>
      </div>
      <div className="runtime-actions"><button className="chip-button" disabled={saving} onClick={()=>void persist(models,model.repository)}>Make active</button><button className="chip-button danger" disabled={saving} onClick={()=>void persist(models.filter((_,i)=>i!==index), settings.vllm_model===model.repository?null:settings.vllm_model)}>Remove</button><button className="primary-action" disabled={saving} onClick={()=>void persist(models)}>Save launch options</button></div>
    </details>)}
    <div className="settings-grid"><label><span>Add Hugging Face repository</span><input value={draft.repository} onChange={(e)=>setDraft({...draft,repository:e.target.value})} placeholder="org/model" /></label><div className="runtime-actions"><button className="chip-button" disabled={!draft.repository.trim()||saving||models.some(m=>m.repository===draft.repository.trim())} onClick={()=>{const next=[...models,{...draft,repository:draft.repository.trim()}];setDraft({...draft,repository:''});void persist(next)}}>Add model</button></div></div>
  </div>
}

function DiscoverSection(props: SectionProps): React.JSX.Element {
  const [query, setQuery] = useState('')
  const [bundles, setBundles] = useState<SdcppBundle[]>([])
  const [bundlesLoading, setBundlesLoading] = useState(false)
  const [showAllBundles, setShowAllBundles] = useState(false)
  /** A licensed bundle awaiting explicit acceptance before it can install. */
  const [consentBundle, setConsentBundle] = useState<SdcppBundle | null>(null)
  const [acceptingConsent, setAcceptingConsent] = useState(false)
  /** Variant choices pending the consent dialog, so agreeing keeps the picks. */
  const [pendingChoices, setPendingChoices] = useState<Record<number, string>>({})
  /** Chosen size per bundle, keyed by the component's position. */
  const [variantChoices, setVariantChoices] = useState<
    Record<string, Record<number, string>>
  >({})
  const [queuedRepos, setQueuedRepos] = useState<Record<string, boolean>>({})
  const [assembleRepo, setAssembleRepo] = useState('')
  const [assemblePath, setAssemblePath] = useState('')
  const [assembling, setAssembling] = useState(false)
  const [proposal, setProposal] = useState<SdcppProposal | null>(null)
  const [discoverEngine, setDiscoverEngine] = useState<DiscoverEngine>(() =>
    defaultDiscoverEngine(props.hardware)
  )
  const [results, setResults] = useState<HubModel[]>([])
  const [hasSearched, setHasSearched] = useState(false)
  const [suggested, setSuggested] = useState<HubModel[]>([])
  const [suggestLoading, setSuggestLoading] = useState(false)
  const [searching, setSearching] = useState(false)
  const [expandedRepo, setExpandedRepo] = useState<string | null>(null)
  const [repoFiles, setRepoFiles] = useState<Record<string, HubFile[]>>({})
  const [fitPreviews, setFitPreviews] = useState<Record<string, QuantFit>>({})
  const [preferredFiles, setPreferredFiles] = useState<Record<string, string | null>>({})
  const [loadingFilesFor, setLoadingFilesFor] = useState<string | null>(null)
  const [enginePhase, setEnginePhase] = useState<string | null>(null)
  const [trustByRepo, setTrustByRepo] = useState<Record<string, ModelTrust>>({})
  const [acknowledgedRepos, setAcknowledgedRepos] = useState<Record<string, boolean>>({})
  const [hfTokenSource, setHfTokenSource] = useState<string>('none')
  const [hfTokenDraft, setHfTokenDraft] = useState('')
  const [savingHfToken, setSavingHfToken] = useState(false)
  const [downloadJobs, setDownloadJobs] = useState<DownloadJob[]>([])
  const downloadJobsRefreshRef = useRef(0)
  const [openDescription, setOpenDescription] = useState<string | null>(null)
  const [descriptions, setDescriptions] = useState<Record<string, string>>({})
  const [descriptionLoading, setDescriptionLoading] = useState<string | null>(null)

  async function toggleDescription(repoId: string): Promise<void> {
    if (openDescription === repoId) {
      setOpenDescription(null)
      return
    }
    setOpenDescription(repoId)
    if (descriptions[repoId] !== undefined) return
    setDescriptionLoading(repoId)
    try {
      const text = await fetchModelDescription(repoId)
      setDescriptions((current) => ({ ...current, [repoId]: text }))
    } catch (cause) {
      setDescriptions((current) => ({ ...current, [repoId]: errorText(cause) }))
    } finally {
      setDescriptionLoading((current) => (current === repoId ? null : current))
    }
  }

  useEffect(() => {
    void huggingFaceTokenStatus()
      .then((status) => setHfTokenSource(status.source))
      .catch(() => setHfTokenSource('none'))
  }, [])

  async function refreshDownloadJobs(): Promise<void> {
    const refreshId = ++downloadJobsRefreshRef.current
    try {
      const jobs = await listDownloadJobs()
      if (refreshId === downloadJobsRefreshRef.current) setDownloadJobs(jobs)
    } catch {
      if (refreshId === downloadJobsRefreshRef.current) setDownloadJobs([])
    }
  }

  useEffect(() => {
    void refreshDownloadJobs()
    const timer = window.setInterval(() => void refreshDownloadJobs(), 2000)
    return () => window.clearInterval(timer)
  }, [])

  // The instant "Queued" flag is never cleared on its own, which permanently
  // disabled an in-place bundle (MiniMax-H3's FL2VA/Ref2VA and quant switches
  // share one id) from being queued again. The download queue is the source of
  // truth: once its job for the bundle settles, forget the flag.
  useEffect(() => {
    if (Object.keys(queuedRepos).length === 0) return
    setQueuedRepos((current) => {
      let next = current
      for (const id of Object.keys(current)) {
        const bundle = bundles.find((candidate) => candidate.id === id)
        if (!bundle) continue
        const repos = new Set(bundle.components.map((component) => component.repo_id))
        const bundleJobs = downloadJobs.filter(
          (job) => job.kind === 'sdcpp-bundle' && job.repo_id != null && repos.has(job.repo_id)
        )
        const stillActive = bundleJobs.some(
          (job) => job.status === 'pending' || job.status === 'downloading'
        )
        if (bundleJobs.length > 0 && !stillActive) {
          if (next === current) next = { ...current }
          delete next[id]
        }
      }
      return next
    })
  }, [downloadJobs])

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

  // Load fresh suggestions for the selected engine and reset any prior search,
  // so opening (or switching category) shows curated picks, not stale results.
  useEffect(() => {
    let cancelled = false
    setResults([])
    setHasSearched(false)
    setExpandedRepo(null)
    setSuggestLoading(true)
    void searchHub('', discoverEngine)
      .then((models) => {
        if (!cancelled) setSuggested(models.slice(0, 9))
      })
      .catch(() => {
        if (!cancelled) setSuggested([])
      })
      .finally(() => {
        if (!cancelled) setSuggestLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [discoverEngine])

  async function findModels(event?: FormEvent): Promise<void> {
    event?.preventDefault()
    if (!query.trim()) return
    setSearching(true)
    setHasSearched(true)
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

  // sd.cpp models are installed as curated bundles rather than single files:
  // Flux and Wan need their VAE and text encoders fetched alongside the
  // diffusion weights, and a manifest written so sd-cli can find them.
  const isSdcpp = discoverEngine === 'stable-diffusion.cpp'

  // Surface the best available tier before a card is expanded. For GGUF
  // repositories this is the best quant; snapshot repositories use the total
  // published weight files as a conservative estimate.
  useEffect(() => {
    if (isSdcpp) return
    const models = hasSearched ? results : suggested
    if (models.length === 0) return
    let cancelled = false
    void Promise.all(
      models.map(async (model) => {
        try {
          const response = await listHubFiles(model.id)
          const files = response.data
          const ggufs = files.filter((file) => {
            const lower = file.path.toLowerCase()
            return lower.endsWith('.gguf') || lower.endsWith('.bin')
          })
          if (ggufs.length > 0) {
            const best = sortQuantGroups(groupQuants(ggufs), props.hardware).find(
              (group) => !group.key.toLowerCase().includes('mmproj')
            )
            return [model.id, generationFit(best?.size, props.hardware)] as const
          }
          const snapshotBytes = files
            .filter((file) => /\.(safetensors|onnx|bin)$/i.test(file.path))
            .reduce((total, file) => total + (file.size ?? 0), 0)
          return [model.id, generationFit(snapshotBytes || null, props.hardware)] as const
        } catch {
          return [model.id, 'unknown' as QuantFit] as const
        }
      })
    ).then((entries) => {
      if (!cancelled) setFitPreviews((current) => ({ ...current, ...Object.fromEntries(entries) }))
    })
    return () => {
      cancelled = true
    }
  }, [isSdcpp, hasSearched, results, suggested, props.hardware])

  useEffect(() => {
    if (!isSdcpp || bundles.length > 0) return
    setBundlesLoading(true)
    void listSdcppBundles()
      .then(setBundles)
      .catch((cause: unknown) => props.onError(errorText(cause)))
      .finally(() => setBundlesLoading(false))
  }, [isSdcpp])

  async function refreshBundles(): Promise<void> {
    try {
      setBundles(await listSdcppBundles())
    } catch {
      // Non-fatal: the list keeps showing what it last loaded.
    }
  }

  // A shortlist by default. The full catalog covers gated, very large, and
  // niche models, which is a poor first thing to meet — but everything stays
  // one click away rather than being hidden.
  const featuredBundles = bundles.filter(
    (bundle) => bundle.featured || bundle.origin === 'custom' || bundle.installed
  )
  const visibleBundles = showAllBundles || featuredBundles.length === 0 ? bundles : featuredBundles
  const hiddenBundleCount = bundles.length - featuredBundles.length
  // A chosen quant installs under its own key, so `installed` on the shipped
  // bundle does not answer whether *this* size is already on disk.
  const installedModelIds = new Set(props.models.map((model) => model.id))

  async function installBundle(
    bundle: SdcppBundle,
    choices: Record<number, string>
  ): Promise<void> {
    if (bundle.gated && hfTokenSource === 'none') {
      props.onError(
        `${bundle.label} includes a file from a gated repository. Accept its terms on Hugging Face and save an access token above first.`
      )
      return
    }
    // A licensed model may not be installed until its agreement is accepted.
    // The check mirrors the daemon's own gate, so a stale catalog entry cannot
    // send the user to a download the server would refuse.
    if (bundle.requires_license_acceptance && !bundle.consent?.accepted) {
      // Keep the chosen sizes so agreeing installs what the person picked,
      // not the bundle's defaults.
      setPendingChoices(choices)
      setConsentBundle(bundle)
      return
    }
    props.onError(null)
    const resolved = resolveBundleVariants(bundle, choices)
    try {
      // A bundle whose sizes were chosen is no longer the shipped one, so it
      // travels in full rather than by id. That includes in-place choices
      // (e.g. MiniMax-H3's FL2VA/Ref2VA and quant switches), which keep the
      // same id but must still install the files the person picked — sending
      // `{ id }` here would silently install the default variant instead.
      const choicesMade = Object.keys(choices).length > 0
      await queueSdcppInstall(
        choicesMade ? { bundle: resolved } : { id: bundle.id }
      )
      setQueuedRepos((current) => ({ ...current, [resolved.id]: true }))
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  /** Record acceptance of a licensed bundle's terms, then install it. */
  async function agreeAndInstall(bundle: SdcppBundle): Promise<void> {
    setAcceptingConsent(true)
    props.onError(null)
    try {
      await acceptSdcppLicense(bundle.id)
      // The bundle handed to install must carry the fresh acceptance, or
      // installBundle would re-open the consent dialog on its own gate.
      const acceptedBundle = bundle.consent
        ? { ...bundle, consent: { ...bundle.consent, accepted: true } }
        : bundle
      setBundles((current) =>
        current.map((candidate) =>
          candidate.id === bundle.id && candidate.consent
            ? { ...candidate, consent: { ...candidate.consent, accepted: true } }
            : candidate
        )
      )
      setConsentBundle(null)
      const choices = pendingChoices
      setPendingChoices({})
      await installBundle(acceptedBundle, choices)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setAcceptingConsent(false)
    }
  }

  async function assembleBundle(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (!assembleRepo.trim() || !assemblePath.trim()) return
    setAssembling(true)
    setProposal(null)
    props.onError(null)
    try {
      setProposal(
        await assembleSdcppBundle({
          repo_id: assembleRepo.trim(),
          path: assemblePath.trim()
        })
      )
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setAssembling(false)
    }
  }

  /** Edit one component of the proposal before it is saved or installed. */
  function updateProposalComponent(
    index: number,
    patch: { repo_id?: string; path?: string }
  ): void {
    setProposal((current) => {
      if (!current) return current
      const components = current.bundle.components.map((component, position) =>
        position === index ? { ...component, ...patch } : component
      )
      return { ...current, bundle: { ...current.bundle, components } }
    })
  }

  async function saveProposal(install: boolean): Promise<void> {
    if (!proposal) return
    props.onError(null)
    try {
      const saved = await saveSdcppBundle(proposal.bundle)
      await refreshBundles()
      setProposal(null)
      setAssembleRepo('')
      setAssemblePath('')
      // A hand-assembled bundle names exact files already; nothing to choose.
      if (install) await installBundle(saved, {})
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  async function removeBundle(bundle: SdcppBundle): Promise<void> {
    props.onError(null)
    try {
      await deleteSdcppBundle(bundle.id)
      await refreshBundles()
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  async function downloadSnapshot(repoId: string): Promise<void> {
    if (
      discoverEngine === 'llama.cpp' ||
      discoverEngine === 'whisper.cpp' ||
      discoverEngine === 'stable-diffusion.cpp'
    ) {
      return
    }
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
    // Queued rather than awaited, so more models can be added while this runs.
    try {
      const kind =
        discoverEngine === 'streaming-asr'
          ? 'streaming-asr'
          : discoverEngine === 'personaplex'
            ? 'personaplex'
            : 'mlx'
      const engine =
        kind === 'mlx' ? (discoverEngine === 'mlx-vlm' ? 'mlx-vlm' : 'mlx-lm') : undefined
      await queueSnapshotDownload(kind, repoId, engine)
      setQueuedRepos((current) => ({ ...current, [repoId]: true }))
    } catch (cause) {
      props.onError(errorText(cause))
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

  /**
   * Queue a quant for download. Every file takes this route, so each one has
   * a job row that survives losing this panel — or the window — and can be
   * paused and resumed from the tray. A split GGUF is several files but one
   * quant: all of its shards are queued, first shard first.
   */
  async function downloadQuant(repoId: string, paths: string[]): Promise<void> {
    if (paths.length === 0) return
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
    const engine = discoverEngine === 'whisper.cpp' ? 'whisper.cpp' : 'llama.cpp'
    try {
      for (const path of paths) {
        await queueModelDownload(repoId, path, 'main', engine)
      }
      setQueuedRepos((current) => ({ ...current, [repoId]: true }))
      setDownloadJobs(await listDownloadJobs())
    } catch (cause) {
      props.onError(errorText(cause))
      return
    }
    if (engine === 'whisper.cpp') return
    // Install the runtime alongside the transfer rather than after it, so the
    // model is usable the moment it lands.
    setEnginePhase('Checking the inference runtime…')
    try {
      await ensureLlamaEngine((event) => setEnginePhase(progressLabel(event)))
    } catch (cause) {
      props.onError(
        `The download is queued, but no runtime is installed yet: ${errorText(cause)}. ` +
          'Open the Runtimes section to install or build one.'
      )
    } finally {
      setEnginePhase(null)
    }
  }

  async function cancelJob(job: DownloadJob): Promise<void> {
    props.onError(null)
    // Prevent a poll begun before this click from restoring its old queued
    // snapshot after the cancellation response arrives.
    downloadJobsRefreshRef.current += 1
    try {
      if (job.kind === 'runtime-build') {
        await cancelBuildJob(job.id)
      } else {
        await cancelDownloadJob(job.id)
      }
      // Reflect cancellation before the next polling round. The daemon still
      // owns the durable state and the regular refresh reconciles it.
      setDownloadJobs((current) =>
        current.map((currentJob) =>
          currentJob.id === job.id
            ? {
                ...currentJob,
                status: 'cancelled',
                error: 'cancelled by user',
                updated_at: new Date().toISOString()
              }
            : currentJob
        )
      )
      void refreshDownloadJobs()
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  async function dismissJob(jobId: string): Promise<void> {
    props.onError(null)
    try {
      await dismissDownloadJob(jobId)
      setDownloadJobs(await listDownloadJobs())
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  async function clearFinishedJobs(): Promise<void> {
    props.onError(null)
    try {
      await dismissFinishedDownloadJobs()
      setDownloadJobs(await listDownloadJobs())
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  async function retryJob(jobId: string): Promise<void> {
    props.onError(null)
    try {
      await resumeDownloadJob(jobId)
      setDownloadJobs(await listDownloadJobs())
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  /** Settled jobs, which are the ones "clear finished" would remove. */
  const finishedJobCount = downloadJobs.filter(
    (job) => job.status !== 'pending' && job.status !== 'downloading' && job.status !== 'paused'
  ).length

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
                : discoverEngine === 'stable-diffusion.cpp'
                  ? 'image and video generation checkpoints for stable-diffusion.cpp'
                  : discoverEngine === 'personaplex'
                    ? 'PersonaPlex / Moshi speech-to-speech snapshots'
                    : `${engineLabel(discoverEngine)} models for Apple Silicon`}
          .
        </p>
        <p className="manage-subtext">{DISCOVER_ENGINE_HELP[discoverEngine]}</p>
      </header>
      <div className="build-form discover-engine-form">
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
                'stable-diffusion.cpp',
                'personaplex',
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
          <span className="label-with-link">
            Hugging Face token (for gated models)
            <a
              className="inline-link"
              href="https://huggingface.co/settings/tokens"
              target="_blank"
              rel="noreferrer"
            >
              Create a token
            </a>
          </span>
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
      {!isSdcpp && (
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
      )}
      {enginePhase && (
        <div className="engine-phase-note">
          <LoaderCircle className="spin" size={14} />
          {enginePhase}
        </div>
      )}
      {downloadJobs.length > 0 && (
        <div className="download-jobs-panel">
          <div className="download-jobs-head">
            <div className="section-label">Download queue</div>
            {finishedJobCount > 0 && (
              <button
                type="button"
                className="chip-button subtle"
                title="Remove finished, failed, and cancelled downloads from this list"
                onClick={() => void clearFinishedJobs()}
              >
                <Trash2 size={12} />
                Clear {finishedJobCount} finished
              </button>
            )}
          </div>
          {downloadJobs.slice(0, 12).map((job) => {
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
                    {pct != null ? ` · ${Math.round(pct)}%` : ''}
                  </span>
                  {pct != null && active && (
                    <div className="progress-track compact">
                      <div className="progress-fill" style={{ width: `${pct}%` }} />
                    </div>
                  )}
                  {job.error && <span className="run-error-text">{job.error}</span>}
                </div>
                <div className="download-job-actions">
                  {job.status === 'failed' && job.kind !== 'runtime-build' && (
                    <button
                      type="button"
                      className="chip-button subtle"
                      title="Try this download again, resuming from what it already fetched"
                      onClick={() => void retryJob(job.id)}
                    >
                      Retry
                    </button>
                  )}
                  {active ? (
                    <button
                      type="button"
                      className="chip-button subtle"
                      onClick={() => void cancelJob(job)}
                    >
                      Cancel
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="icon-button subtle"
                      title="Dismiss"
                      aria-label={`Dismiss ${basename}`}
                      onClick={() => void dismissJob(job.id)}
                    >
                      <X size={13} />
                    </button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
      {isSdcpp && (
        <div className="bundle-assemble">
          <div className="section-label">Add any model from Hugging Face</div>
          <p className="manage-subtext">
            Brazier reads the checkpoint's header to work out its architecture, then fills in the
            VAE and text encoders sd-cli will need. Review the file list before installing.
          </p>
          <form className="build-form" onSubmit={(event) => void assembleBundle(event)}>
            <label>
              <span>Repository</span>
              <input
                value={assembleRepo}
                onChange={(event) => setAssembleRepo(event.target.value)}
                placeholder="city96/FLUX.1-dev-gguf"
              />
            </label>
            <label>
              <span>File path in the repo</span>
              <input
                value={assemblePath}
                onChange={(event) => setAssemblePath(event.target.value)}
                placeholder="flux1-dev-Q4_K_S.gguf"
              />
            </label>
            <button
              className="chip-button"
              type="submit"
              disabled={assembling || !assembleRepo.trim() || !assemblePath.trim()}
            >
              {assembling ? <LoaderCircle className="spin" size={13} /> : <Search size={13} />}
              Identify model
            </button>
          </form>

          {proposal && (
            <article className="bundle-card proposal">
              <div className="bundle-head">
                <strong>{proposal.bundle.label}</strong>
                <span className="bundle-meta">
                  {proposal.architecture_label ?? 'Unrecognized architecture'}
                  {proposal.variant ? ` · ${proposal.variant}` : ''}
                  {` · detected by ${proposal.detected_by}`}
                </span>
              </div>
              {proposal.warnings.map((warning) => (
                <p className="bundle-warning" key={warning}>
                  <ShieldAlert size={13} /> {warning}
                </p>
              ))}
              <div className="proposal-components">
                {proposal.bundle.components.map((component, index) => (
                  <div className="proposal-row" key={`${component.role}-${index}`}>
                    <span className="bundle-role">
                      {component.role}
                      {component.flag ? <code>--{component.flag}</code> : <code>-m</code>}
                    </span>
                    <input
                      aria-label={`${component.role} repository`}
                      value={component.repo_id}
                      onChange={(event) =>
                        updateProposalComponent(index, { repo_id: event.target.value })
                      }
                    />
                    <input
                      aria-label={`${component.role} file`}
                      value={component.path}
                      onChange={(event) =>
                        updateProposalComponent(index, { path: event.target.value })
                      }
                    />
                  </div>
                ))}
              </div>
              <div className="proposal-actions">
                <button className="chip-button" onClick={() => void saveProposal(true)}>
                  <Download size={13} /> Save and install
                </button>
                <button className="chip-button subtle" onClick={() => void saveProposal(false)}>
                  Save for later
                </button>
                <button className="chip-button subtle" onClick={() => setProposal(null)}>
                  Discard
                </button>
              </div>
            </article>
          )}
        </div>
      )}
      {isSdcpp && (
        <div className="bundle-list">
          <div className="section-label">
            {showAllBundles
              ? 'Every preconfigured model · installs each required file'
              : 'Recommended models · installs every required file'}
          </div>
          {bundlesLoading && bundles.length === 0 && (
            <p className="empty-models-inline">
              <LoaderCircle className="spin" size={14} /> Loading the model catalog…
            </p>
          )}
          {visibleBundles.map((bundle) => {
            const chosen = variantChoices[bundle.id] ?? {}
            const resolved = resolveBundleVariants(bundle, chosen)
            const queued = Boolean(queuedRepos[resolved.id])
            const chosenOptions = Object.entries(chosen)
              .map(([index, label]) =>
                bundle.components[Number(index)]?.variants?.find((option) => option.label === label)
              )
              .filter((option): option is NonNullable<typeof option> => option != null)
            // Decoder alternatives replace just that file and rewrite the same
            // manifest, so always leave their action available as “switch”.
            // Other variants remain distinct model installs.
            const hasInPlaceChoice = chosenOptions.some((option) => option.in_place)
            const installed = hasInPlaceChoice
              ? false
              : Object.keys(chosen).length === 0
                ? bundle.installed
                : installedModelIds.has(resolved.model_id)
            const totalBytes = resolved.components.reduce(
              (sum, component) => sum + (component.approx_bytes ?? 0),
              0
            )
            const diffusionBytes = resolved.components
              .filter((component) => component.flag === 'diffusion-model' || !component.flag)
              .reduce((sum, component) => sum + (component.approx_bytes ?? 0), 0)
            const fit = generationFit(totalBytes || null, props.hardware, diffusionBytes || null)
            return (
              <article className="bundle-card" key={bundle.id}>
                <div className="bundle-head">
                  <div className="bundle-title">
                  <strong>
                    {bundle.label}
                    <CapabilityIcons
                      flags={{
                        imageOut: bundle.modality === 'image',
                        videoOut: bundle.modality === 'video',
                        audioOut: bundle.components.some(
                          (component) => component.flag === 'audio-vae'
                        ),
                        imageIn: bundle.supports_init_image
                      }}
                    />
                    {bundle.modality === 'video' && (
                      // The one distinction that decides whether a photo can be
                      // handed to this model at all.
                      <span
                        className={`kind-badge ${bundle.supports_init_image ? 'i2v' : 't2v'}`}
                        title={
                          bundle.supports_init_image
                            ? 'Image-to-video: animates a picture you supply, and can also work from text alone.'
                            : 'Text-to-video: builds the clip from the prompt only. It cannot start from a picture.'
                        }
                      >
                        {bundle.supports_init_image ? 'Image → video' : 'Text → video'}
                      </span>
                    )}
                    {installed && <span className="installed-badge">Installed</span>}
                    {bundle.gated && !installed && (
                      <span className="installed-badge gated">Token required</span>
                    )}
                    {bundle.requires_license_acceptance && !installed && (
                      <span
                        className={`installed-badge ${bundle.consent?.accepted ? '' : 'gated'}`}
                        title={
                          bundle.consent?.accepted
                            ? 'You have accepted this model\u2019s license agreement.'
                            : 'This model is licensed and must be accepted before it can be installed.'
                        }
                      >
                        {bundle.consent?.accepted ? 'License accepted' : 'License required'}
                      </span>
                    )}
                  </strong>
                  <span className={`generation-fit ${fit}`}>{generationFitLabel(fit)}</span>
                  </div>
                  <span className="bundle-meta">
                    {bundle.modality === 'video' ? 'Video' : 'Image'}
                    {bundle.origin === 'custom' ? ' · Yours' : ''}
                    {bundle.license ? ` · ${bundle.license}` : ''}
                    {totalBytes > 0 ? ` · ~${formatBytes(totalBytes)}` : ''}
                  </span>
                </div>
                <p className="bundle-summary">{bundle.summary}</p>
                <ul className="bundle-components">
                  {bundle.components.map((component, index) => {
                    const options = component.variants ?? []
                    const selected =
                      chosen[index] ??
                      options.find((option) => option.path === component.path)?.label ??
                      ''
                    const active = options.find((option) => option.label === selected)
                    return (
                      <li key={`${component.repo_id}/${component.path}`}>
                        <span className="bundle-role">{component.role}</span>
                        <code>{component.repo_id}</code>
                        {options.length > 0 ? (
                          <select
                            className="bundle-quant"
                            value={selected}
                            title={active?.note ?? 'Choose a size'}
                            onChange={(event) =>
                              setVariantChoices((current) => ({
                                ...current,
                                [bundle.id]: { ...chosen, [index]: event.target.value }
                              }))
                            }
                          >
                            {options.map((option) => (
                              <option key={option.label} value={option.label}>
                                {option.label}
                                {option.approx_bytes
                                  ? ` · ${formatBytes(option.approx_bytes)}`
                                  : ''}
                              </option>
                            ))}
                          </select>
                        ) : component.approx_bytes ? (
                          <span className="bundle-size">{formatBytes(component.approx_bytes)}</span>
                        ) : null}
                      </li>
                    )
                  })}
                </ul>
                <div className="bundle-actions">
                <button
                  className="chip-button"
                  disabled={queued || installed}
                  onClick={() => void installBundle(bundle, chosen)}
                >
                  {queued ? (
                    <>
                      <LoaderCircle className="spin" size={13} /> Queued
                    </>
                  ) : installed ? (
                    <>
                      <Check size={13} /> Installed
                    </>
                  ) : (
                    <>
                      <Download size={13} /> Install all {bundle.components.length} files
                    </>
                  )}
                </button>
                {bundle.origin === 'custom' && (
                  <button
                    className="chip-button subtle"
                    title="Remove this bundle from the list. Downloaded files stay on disk."
                    onClick={() => void removeBundle(bundle)}
                  >
                    <Trash2 size={13} /> Forget
                  </button>
                )}
                </div>
              </article>
            )
          })}
          {hiddenBundleCount > 0 && (
            <button
              type="button"
              className="chip-button subtle bundle-show-all"
              onClick={() => setShowAllBundles((shown) => !shown)}
            >
              {showAllBundles
                ? 'Show only the recommended models'
                : `Show all ${bundles.length} preconfigured models`}
            </button>
          )}
        </div>
      )}
      {!isSdcpp && !hasSearched && (
        <div className="section-label suggested-label">
          {suggestLoading
            ? 'Loading suggestions…'
            : suggested.length > 0
              ? 'Suggested · popular & recent'
              : 'Type a model name or author to search'}
        </div>
      )}
      {!isSdcpp && hasSearched && !searching && results.length === 0 && (
        <p className="empty-models-inline">
          No models matched “{query.trim()}”. Try a different name or author.
        </p>
      )}
      <div className="model-results">
        {(isSdcpp ? [] : hasSearched ? results : suggested).map((model) => {
          const expanded = expandedRepo === model.id
          const previewFit = fitPreviews[model.id] ?? 'unknown'
          const groups = sortQuantGroups(groupQuants((repoFiles[model.id] ?? []).filter((file) => {
            const lower = file.path.toLowerCase()
            if (discoverEngine === 'whisper.cpp') {
              return lower.endsWith('.bin') || lower.endsWith('.gguf')
            }
            return lower.endsWith('.gguf')
          })), props.hardware)
          const preferred = preferredFiles[model.id]
          const filePickEngine =
            discoverEngine === 'llama.cpp' || discoverEngine === 'whisper.cpp'
          // streaming-asr / mlx use snapshot download, not per-file picker
          return (
            <article className="model-card expandable" key={model.id}>
              <div className="model-card-main">
                <div className="model-card-heading">
                  <button
                    type="button"
                    className="model-name-button"
                    title="Show model description"
                    onClick={() => void toggleDescription(model.id)}
                  >
                    <span className="model-name-text">{model.id.split('/').at(-1)}</span>
                    {discoverEngine !== 'whisper.cpp' &&
                      discoverEngine !== 'streaming-asr' &&
                      discoverEngine !== 'personaplex' && (
                        <CapabilityIcons flags={hubCapabilityFlags(model.tags)} inferred />
                      )}
                    {openDescription === model.id ? (
                      <ChevronDown size={13} />
                    ) : (
                      <ChevronRight size={13} />
                    )}
                  </button>
                  <span className={`generation-fit model-fit ${previewFit}`}>
                    {generationFitLabel(previewFit)}
                  </span>
                  <span className="model-card-author">{model.author}</span>
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
              {openDescription === model.id && (
                <p className="model-description">
                  {descriptionLoading === model.id
                    ? 'Loading description…'
                    : (descriptions[model.id] ?? '')}
                </p>
              )}
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
                          <p>
                            A Hugging Face token is required for this gated model. Accept the
                            license on the{' '}
                            <a
                              className="inline-link"
                              href={`https://huggingface.co/${model.id}`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              model page
                            </a>
                            , then{' '}
                            <a
                              className="inline-link"
                              href="https://huggingface.co/settings/tokens"
                              target="_blank"
                              rel="noreferrer"
                            >
                              create a token
                            </a>{' '}
                            and save it above.
                          </p>
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
                          title="Downloads in the background; track it in the download tray"
                          onClick={() => void downloadSnapshot(model.id)}
                        >
                          <Download size={14} />
                          Download
                        </button>
                      </div>
                    </div>
                  ) : groups.length === 0 ? (
                    <p className="empty-models-inline">No GGUF files found in this repo.</p>
                  ) : (
                    groups.map((group) => {
                    const basename = quantGroupName(group)
                    const fit = generationFit(group.size, props.hardware)
                    const lowerName = basename.toLowerCase()
                    const isProjector = lowerName.includes('mmproj')
                    const isDraft =
                      !isProjector &&
                      (lowerName.includes('dspark') ||
                        lowerName.includes('dflash') ||
                        lowerName.includes('draft'))
                    const isPreferred =
                      preferred != null &&
                      group.files.some(
                        (file) => file.path === preferred || file.path.endsWith(`/${preferred}`)
                      )
                    return (
                      <div className="quant-row" key={group.key}>
                        <div>
                          <strong>
                            {basename}
                            {isProjector ? ' · multimodal projector' : ''}
                            {isDraft ? ' · speculative draft' : ''}
                            {isPreferred ? ' · preferred' : ''}
                          </strong>
                          {!isProjector && !isDraft && (
                            <span className={`generation-fit quant-fit ${fit}`}>
                              {generationFitLabel(fit)}
                            </span>
                          )}
                          <span>
                            {group.key}
                            {group.size != null ? ` · ${formatBytes(group.size)}` : ''}
                            {group.files.length > 1 ? ` · ${group.files.length} parts` : ''}
                          </span>
                        </div>
                        <div className="quant-actions">
                          <button
                            type="button"
                            title="Downloads in the background; track it in the download tray"
                            onClick={() => void downloadQuant(model.id, group.files.map((file) => file.path))}
                          >
                            <Download size={14} />
                            {isProjector || isDraft ? 'Add capability' : 'Download'}
                          </button>
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

      {consentBundle?.consent && (
        <div
          className="menu-backdrop model-settings-backdrop"
          onMouseDown={() => {
            setPendingChoices({})
            setConsentBundle(null)
          }}
        >
          <div
            className="model-settings-modal license-consent-modal"
            role="dialog"
            aria-label={`${consentBundle.label} license agreement`}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header className="model-settings-head">
              <div>
                <strong>License required</strong>
                <span>{consentBundle.label} must be accepted before it can be installed</span>
              </div>
              <button
                type="button"
                className="icon-button"
                onClick={() => {
                  setPendingChoices({})
                  setConsentBundle(null)
                }}
                aria-label="Close"
              >
                <X size={17} />
              </button>
            </header>
            <div className="model-settings-body">
              <div className="build-warning license-consent-body">
                <ShieldAlert size={15} />
                <div>
                  <p>
                    This model is released under the <strong>{consentBundle.consent.id}</strong>.
                  </p>
                  <p className="license-consent-summary">{consentBundle.consent.summary}</p>
                  <p>
                    You are responsible for making sure you are allowed to use it. In particular,
                    the agreement limits where it may be used — a separate license from MiniMax may
                    be required in some countries, including the United States. Read the full terms
                    before agreeing.
                  </p>
                  <p>
                    <a
                      href={consentBundle.consent.url}
                      target="_blank"
                      rel="noreferrer"
                      className="license-consent-link"
                    >
                      Read the full {consentBundle.consent.id}
                    </a>
                  </p>
                </div>
              </div>
            </div>
            <footer className="model-settings-foot">
              <button
                type="button"
                className="chip-button subtle"
                disabled={acceptingConsent}
                onClick={() => {
                  setPendingChoices({})
                  setConsentBundle(null)
                }}
              >
                Not now
              </button>
              <div className="model-settings-foot-actions">
                <button
                  type="button"
                  className="chip-button"
                  disabled={acceptingConsent}
                  onClick={() => void agreeAndInstall(consentBundle)}
                >
                  {acceptingConsent ? (
                    <>
                      <LoaderCircle className="spin" size={13} /> Saving…
                    </>
                  ) : (
                    'I agree — install the model'
                  )}
                </button>
              </div>
            </footer>
          </div>
        </div>
      )}
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

export function targetSupportedByBuildEngine(
  engine: BuildEngine,
  target: RuntimeTarget
): boolean {
  return engine !== 'vllm' || target !== 'vulkan'
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
  // Scoped to llama.cpp: managed whisper/sd.cpp installs also use target `cpu`
  // and must not mark the llama CPU runtime as installed.
  return runtimes.some(
    (runtime) =>
      runtime.kind === 'managed' &&
      runtime.engine === 'llama.cpp' &&
      (runtime.target === target || (target === 'cpu' && runtime.id === 'managed'))
  )
}

function managedTargetStatus(
  target: RuntimeTarget,
  statuses: ManagedLlamaTargetStatus[] | null
): ManagedLlamaTargetStatus | null {
  return statuses?.find((entry) => entry.target === target) ?? null
}

/** Whether a *source build* exists for an engine — managed prebuilts (which
 * share the same engine string, e.g. stable-diffusion.cpp) must not count. */
function sourceRuntimeInstalled(runtimes: RuntimeEntry[], engine: BuildEngine): boolean {
  return runtimes.some((runtime) => runtime.kind === 'source' && runtime.engine === engine)
}

/** Whether a managed prebuilt is installed for an engine. */
function managedEngineInstalled(runtimes: RuntimeEntry[], engine: string): boolean {
  return runtimes.some((runtime) => runtime.kind === 'managed' && runtime.engine === engine)
}

type RuntimeTab = 'language' | 'speech' | 'media'

/** Engines grouped by modality, mirroring the main UI's Chat/Voice/Generate split. */
const RUNTIME_TAB_ENGINES: Record<RuntimeTab, string[]> = {
  language: ['llama.cpp', 'mlx-lm', 'mlx-vlm', 'vllm'],
  speech: ['whisper.cpp', 'whisperkit', 'streaming-asr', 'personaplex', 'personaplex-mlx'],
  media: ['stable-diffusion.cpp']
}

const RUNTIME_TABS: ReadonlyArray<readonly [RuntimeTab, string]> = [
  ['language', 'Language'],
  ['speech', 'Speech'],
  ['media', 'Media']
]

/** Recipe `supported_platforms`, mirrored so the UI only offers engines that can
 * actually build on this host (e.g. PersonaPlex is Linux x64 only). */
const BUILD_ENGINE_PLATFORMS: Partial<Record<BuildEngine, string[]>> = {
  'mlx-lm': ['macos-arm64'],
  'mlx-vlm': ['macos-arm64'],
  vllm: ['linux-x64', 'linux-arm64', 'macos-arm64'],
  'streaming-asr': ['macos-arm64', 'macos-x64', 'linux-x64', 'linux-arm64'],
  'stable-diffusion.cpp': [
    'linux-x64',
    'linux-arm64',
    'macos-x64',
    'macos-arm64',
    'windows-x64',
    'windows-arm64'
  ],
  personaplex: ['linux-x64'],
  'personaplex-mlx': ['macos-arm64'],
  whisperkit: ['macos-arm64']
}

/** Platform tag matching the daemon's `builds::current_platform()`. */
function platformTag(hardware: HardwareInfo | null): string | null {
  if (!hardware) return null
  const arch =
    hardware.architecture === 'aarch64'
      ? 'arm64'
      : hardware.architecture === 'x86_64'
        ? 'x64'
        : null
  if (!arch) return null
  if (hardware.os === 'macos' || hardware.os === 'linux' || hardware.os === 'windows') {
    return `${hardware.os}-${arch}`
  }
  return null
}

function engineBuildable(engine: BuildEngine, hardware: HardwareInfo | null): boolean {
  const platforms = BUILD_ENGINE_PLATFORMS[engine]
  if (!platforms) return true
  const tag = platformTag(hardware)
  return tag ? platforms.includes(tag) : false
}

/**
 * The architecture a managed runtime is built for: the accelerator's own
 * architecture (gfx1101, …) when one is detected, Apple Silicon on macOS, and
 * the platform architecture otherwise. Shown on every managed runtime offer so
 * a download can be checked against this machine at a glance.
 */
function managedRuntimeArch(hardware: HardwareInfo | null): string | null {
  if (!hardware) return null
  if (hardware.gpu_arch) return hardware.gpu_arch
  if (hardware.os === 'macos') return 'Apple Silicon'
  return hardware.architecture
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
  // What this machine has of what a build needs. Read once per visit: it
  // changes when someone installs something, not while they look at it.
  const [toolchainTools, setToolchainTools] = useState<ToolchainTool[]>([])

  useEffect(() => {
    void fetchToolchainStatus()
      .then((status) => setToolchainTools(status.tools))
      .catch(() => setToolchainTools([]))
  }, [])

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
  const [buildName, setBuildName] = useState('')
  const [revision, setRevision] = useState(
    BUILD_ENGINE_DEFAULTS[isAppleSilicon ? 'mlx-lm' : 'llama.cpp'].revision
  )
  const [buildTarget, setBuildTarget] = useState<RuntimeTarget>('cpu')
  const buildTargets = useMemo(
    () =>
      sourceBuildTargets(props.hardware).filter(
        (target) =>
          targetSupportedByBuildEngine(buildEngine, target) &&
          !(buildEngine === 'vllm' && isAppleSilicon && target !== 'metal')
      ),
    [props.hardware?.os, buildEngine, isAppleSilicon]
  )
  const managedTargets = useMemo(
    () => (props.hardware?.targets ?? []).filter((target) => target.managed_install),
    [props.hardware?.targets]
  )
  const buildEngineOptions = useMemo((): BuildEngine[] => {
    const candidates: BuildEngine[] = [
      'mlx-lm',
      'mlx-vlm',
      'vllm',
      'streaming-asr',
      'stable-diffusion.cpp',
      'personaplex',
      'personaplex-mlx',
      'whisperkit'
    ]
    return candidates.filter((engine) => engineBuildable(engine, props.hardware))
  }, [props.hardware?.os, props.hardware?.architecture])
  const [buildJobs, setBuildJobs] = useState(initialBuildJobs)
  const [building, setBuilding] = useState(false)
  const [activateCompletedBuild, setActivateCompletedBuild] = useState(false)
  const [activeBuildId, setActiveBuildId] = useState<string | null>(null)
  const [buildProgress, setBuildProgress] = useState<JobProgressState>(() =>
    emptyJobProgress('Preparing source build')
  )
  const [buildWarning, setBuildWarning] = useState<string | null>(null)
  const [managedStatuses, setManagedStatuses] = useState<ManagedLlamaTargetStatus[] | null>(
    null
  )
  /** Set while the daemon is still checking upstream for newer releases. */
  const [updateCheckPending, setUpdateCheckPending] = useState(false)
  const logRef = useRef<HTMLPreElement>(null)
  const [updates, setUpdates] = useState<Record<string, SourceRuntimeUpdate>>({})
  const [checkingUpdates, setCheckingUpdates] = useState(false)
  const [updatesChecked, setUpdatesChecked] = useState(false)
  const [runtimeTab, setRuntimeTab] = useState<RuntimeTab>('language')

  async function checkUpdates(): Promise<void> {
    setCheckingUpdates(true)
    props.onError(null)
    try {
      const results = await checkRuntimeUpdates()
      const map: Record<string, SourceRuntimeUpdate> = {}
      for (const entry of results) map[entry.id] = entry
      setUpdates(map)
      setUpdatesChecked(true)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setCheckingUpdates(false)
    }
  }

  function rebuildFromUpdate(update: SourceRuntimeUpdate): void {
    applyBuildEngine(update.engine as BuildEngine, update.repository)
    setRevision(update.revision)
    setBuildOpen(true)
  }

  function applyBuildEngine(engine: BuildEngine, repositoryOverride?: string): void {
    const defaults = BUILD_ENGINE_DEFAULTS[engine]
    setBuildEngine(engine)
    // vLLM's macOS CPU backend is intentionally not offered: Apple Silicon
    // builds use the MLX-backed vLLM-Metal plugin, which installs vLLM core and
    // its matching dependencies into the same isolated environment.
    const platformDefault =
      engine === 'vllm' && isAppleSilicon
        ? 'https://github.com/vllm-project/vllm-metal'
        : defaults.repository
    setRepository(repositoryOverride ?? platformDefault)
    setRevision(defaults.revision)
  }

  useEffect(() => {
    if (!props.pendingBuild) return
    applyBuildEngine(props.pendingBuild.engine, props.pendingBuild.repository)
    // A fork offered after a failed model load is a recovery action. Once it
    // builds, make it the engine's default so retrying the model uses it.
    setActivateCompletedBuild(true)
    setBuildOpen(true)
    props.onPendingBuildConsumed?.()
  }, [props.pendingBuild, props.onPendingBuildConsumed])

  async function refreshManagedStatuses(force = false): Promise<void> {
    try {
      const [llama, whisper, sdcpp] = await Promise.all([
        fetchManagedLlamaStatus(force),
        fetchManagedWhisperStatus(force).catch(() => null),
        fetchManagedSdcppStatus(force).catch(() => null)
      ])
      setManagedStatuses(llama.targets)
      setUpdateCheckPending(
        Boolean(llama.latest_pending || whisper?.latest_pending || sdcpp?.latest_pending)
      )
      // Surface whisper/sd.cpp availability in the install helper copy via statuses.
      void whisper
      void sdcpp
    } catch {
      setManagedStatuses(null)
      setUpdateCheckPending(false)
    }
  }

  // Installed versions arrive immediately; the upstream check runs in the
  // background, so poll briefly until it reports a result.
  useEffect(() => {
    if (!updateCheckPending) return
    const timer = window.setTimeout(() => void refreshManagedStatuses(), 3000)
    return () => window.clearTimeout(timer)
  }, [updateCheckPending, managedStatuses])

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
      setBuildTarget(buildTargets[0] ?? 'cpu')
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

  async function deactivate(id: string): Promise<void> {
    setBusyRuntime(id)
    try {
      await deactivateRuntime(id)
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

  async function installManaged(
    target: RuntimeTarget,
    force = false,
    engine: 'llama.cpp' | 'whisper.cpp' | 'stable-diffusion.cpp' = 'llama.cpp'
  ): Promise<void> {
    const label =
      engine === 'whisper.cpp'
        ? `whisper.cpp · ${target}`
        : engine === 'stable-diffusion.cpp'
          ? `stable-diffusion.cpp · ${target}`
          : llamaRuntimeLabel(target)
    setInstallingTarget(target)
    setInstallProgress(
      emptyJobProgress(force ? `Updating ${label}` : `Installing ${label}`)
    )
    props.onError(null)
    try {
      const ensure =
        engine === 'whisper.cpp'
          ? ensureWhisperEngine
          : engine === 'stable-diffusion.cpp'
            ? ensureSdcppEngine
            : ensureLlamaEngine
      await ensure(
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

  const isPythonBuild =
    buildEngine === 'mlx-lm' ||
    buildEngine === 'mlx-vlm' ||
    buildEngine === 'vllm' ||
    buildEngine === 'streaming-asr' ||
    buildEngine === 'personaplex' ||
    buildEngine === 'personaplex-mlx'
  const isSwiftBuild = buildEngine === 'whisperkit'
  const isWhisperBuild = buildEngine === 'whisper.cpp'
  const isStreamingAsrBuild = buildEngine === 'streaming-asr'

  async function runBuild(event: FormEvent): Promise<void> {
    event.preventDefault()
    setBuilding(true)
    setActiveBuildId(null)
    setBuildProgress(emptyJobProgress('Preparing source build'))
    setBuildWarning(null)
    props.onError(null)
    let buildCompleted = false
    try {
      const built = await buildRuntime(
        buildEngine,
        repository.trim(),
        revision.trim(),
        buildTarget,
        buildJobs,
        buildName,
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
      buildCompleted = true
      if (activateCompletedBuild) {
        await activateRuntime(sourceRuntimeId(buildEngine, built.build_id))
        setActivateCompletedBuild(false)
      }
      setBuildProgress((current) => ({
        ...current,
        headline: activateCompletedBuild
          ? 'Build complete — activated as the default runtime.'
          : 'Build complete — activate it below to use it.',
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
        headline: buildCompleted ? 'Build complete, but activation failed' : 'Build failed',
        phase: 'error'
      }))
      props.onError(
        buildCompleted
          ? `The runtime was built but could not be activated: ${errorText(cause)}`
          : `Build failed: ${errorText(cause)}`
      )
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
  const selectedTargetDetail = targets.find(
    (target) => target.id === props.settings?.target
  )?.detail
  const runtimeList = runtimes ?? []
  // Managed downloads are real runtime choices too. Keeping them out of this
  // list left the only Activate button reachable only for source builds, so a
  // downloaded Vulkan/CPU runtime could never be made the default.
  const installedRuntimes = runtimeList
  const tabEngines = RUNTIME_TAB_ENGINES[runtimeTab]
  const tabBuildOptions = buildEngineOptions.filter((engine) => tabEngines.includes(engine))
  const tabInstalledRuntimes = installedRuntimes.filter((runtime) => tabEngines.includes(runtime.engine))
  const runtimeArch = managedRuntimeArch(props.hardware)

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
            {runtimeArch ? ` · ${runtimeArch}` : ''}
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
        {/* Beneath the grid rather than in a tooltip: the ROCm caveat about
            integrated graphics is the difference between a working GPU and a
            hung one, and nobody hovers a button they have already chosen. */}
        {selectedTargetDetail && <p className="target-detail">{selectedTargetDetail}</p>}
      </div>

      <div className="mode-switch runtime-tabs" role="tablist" aria-label="Runtime type">
        {RUNTIME_TABS.map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            className={runtimeTab === id ? 'active' : ''}
            aria-selected={runtimeTab === id}
            onClick={() => setRuntimeTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="settings-group">
        <div className="settings-group-head">
          <div className="section-label">Available for your hardware</div>
          <button
            className="chip-button subtle"
            disabled={updateCheckPending}
            title="Query upstream releases for managed prebuilts now, bypassing the update-check cache"
            onClick={() => void refreshManagedStatuses(true)}
          >
            {updateCheckPending ? (
              <LoaderCircle className="spin" size={13} />
            ) : (
              <RefreshCw size={13} />
            )}
            Check for updates
          </button>
        </div>
        {installProgress && (
          <JobProgressPanel progress={installProgress} active={installingTarget != null} />
        )}
        <div className="runtime-offer-list">
          {runtimeTab === 'language' &&
            managedTargets.map((target) => {
            const installed = managedRuntimeInstalled(runtimeList, target.id)
            const status = managedTargetStatus(target.id, managedStatuses)
            const updateAvailable = status?.update_available ?? false
            const installedVersion = status?.installed_version
            const latestVersion = status?.latest_version
            const installing = installingTarget === target.id
            const archLabel = target.recommended && runtimeArch ? runtimeArch : null
            const versionLine = [
              target.detail,
              archLabel ? `Built for ${archLabel}` : null,
              installed && installedVersion ? `Installed · ${installedVersion}` : null,
              updateAvailable && latestVersion ? `Latest · ${latestVersion}` : null,
              installed && !latestVersion && updateCheckPending ? 'Checking for updates…' : null
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
                      // Without an upstream tag we only know it is installed.
                      <span className="installed-badge">
                        {latestVersion ? 'Up to date' : 'Installed'}
                      </span>
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
          {runtimeTab === 'speech' &&
            (() => {
              const installed = managedEngineInstalled(runtimeList, 'whisper.cpp')
              const installing = installingTarget === 'cpu'
              return (
                <article className="runtime-offer">
                  <div className="runtime-offer-info">
                    <strong>
                      whisper.cpp · managed
                      {installed && <span className="installed-badge">Installed</span>}
                    </strong>
                    <span>
                      Official CLI prebuilts on Linux/Windows. macOS releases are XCFramework-only —
                      build from source there.
                    </span>
                    {runtimeArch && <span className="runtime-offer-arch">Built for {runtimeArch}</span>}
                  </div>
                  <button
                    className="chip-button"
                    disabled={installed || installing}
                    title={installed ? 'Managed whisper-cli installed' : 'Download managed whisper-cli'}
                    onClick={() => void installManaged('cpu', false, 'whisper.cpp')}
                  >
                    {installing ? (
                      <LoaderCircle className="spin" size={13} />
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
            })()}
          {runtimeTab === 'media' &&
            (() => {
              const installed = managedEngineInstalled(runtimeList, 'stable-diffusion.cpp')
              const installing = installingTarget === 'cpu'
              return (
                <article className="runtime-offer">
                  <div className="runtime-offer-info">
                    <strong>
                      stable-diffusion.cpp · managed
                      {installed && <span className="installed-badge">Installed</span>}
                    </strong>
                    <span>
                      Prebuilt sd-cli pinned to Brazier's supported release; build from source to
                      opt into a newer version.
                    </span>
                    {runtimeArch && <span className="runtime-offer-arch">Built for {runtimeArch}</span>}
                  </div>
                  <button
                    className="chip-button"
                    disabled={installed || installing}
                    title={installed ? 'Managed sd-cli installed' : 'Download managed sd-cli'}
                    onClick={() => void installManaged('cpu', false, 'stable-diffusion.cpp')}
                  >
                    {installing ? (
                      <LoaderCircle className="spin" size={13} />
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
            })()}
          {tabBuildOptions.map((engine) => {
            const installed = sourceRuntimeInstalled(runtimeList, engine)
            return (
              <article className="runtime-offer" key={engine}>
                <div className="runtime-offer-info">
                  <strong>
                    {BUILD_ENGINE_LABELS[engine]}
                    {engine === 'mlx-lm' && isAppleSilicon && (
                      <span className="active-badge">Recommended</span>
                    )}
                    {installed && <span className="installed-badge">Built</span>}
                  </strong>
                  <span>
                    {engine === 'streaming-asr'
                      ? 'Build an isolated Python environment with Transformers for Nemotron streaming ASR. Requires uv.'
                      : engine === 'personaplex'
                        ? 'Build PersonaPlex / Moshi (Linux CUDA) for realtime Voice mode. Requires uv.'
                        : engine === 'personaplex-mlx'
                          ? 'Build PersonaPlex-MLX for realtime Voice mode on Apple Silicon. Requires uv and HF access to nvidia/personaplex-7b-v1.'
                          : engine === 'whisperkit'
                            ? 'Build WhisperKit (Argmax) for on-device CoreML ASR on Apple Silicon. Requires Xcode/Swift. Models download on first use.'
                            : engine === 'stable-diffusion.cpp'
                              ? 'Build sd-cli from source or a trusted fork.'
                              : 'Build a local Python environment with uv. Required for MLX models on Apple Silicon.'}
                  </span>
                </div>
                <button
                  className="chip-button"
                  title={`Build ${BUILD_ENGINE_LABELS[engine]} from source`}
                  onClick={() => openBuildForEngine(engine)}
                >
                  <Hammer size={13} />
                  {installed ? 'Build again' : 'Build'}
                </button>
              </article>
            )
          })}
        </div>
        {runtimeTab === 'language' && managedTargets.some((target) => !target.available) && (
          <p className="model-help">
            Grayed-out options need hardware or drivers that were not detected on this machine. You
            can still build llama.cpp for them from source below.
          </p>
        )}
      </div>

      <div className="settings-group">
        <div className="settings-group-head">
          <div className="section-label">Installed runtimes</div>
          {tabInstalledRuntimes.some((runtime) => runtime.kind === 'source') && (
            <button
              className="chip-button subtle"
              disabled={checkingUpdates}
              title="Query upstream git refs for source builds"
              onClick={() => void checkUpdates()}
            >
              {checkingUpdates ? (
                <LoaderCircle className="spin" size={13} />
              ) : (
                <RefreshCw size={13} />
              )}
              Check for updates
            </button>
          )}
        </div>
        {runtimes == null && !props.initialRuntimes?.length && (
          <div className="manage-placeholder">
            <LoaderCircle className="spin" size={16} />
            Scanning installed runtimes…
          </div>
        )}
        {runtimes != null && tabInstalledRuntimes.length === 0 && (
          <div className="manage-placeholder compact">
            <Cpu size={16} />
            Downloaded and source-built runtimes appear here.
          </div>
        )}
        <div className="runtime-list">
          {tabInstalledRuntimes.map((runtime) => {
            const update = updates[runtime.id]
            const canRebuild =
              runtime.kind === 'source' &&
              update != null &&
              !update.pinned &&
              !update.error &&
              (update.update_available || (!update.current_commit && update.upstream_commit != null))
            return (
            <article
              className={runtime.active ? 'runtime-card active' : 'runtime-card'}
              key={runtime.id}
            >
              <div className="runtime-card-info">
                <strong>
                  {runtime.label}
                  {runtime.active && <span className="active-badge">Active</span>}
                  {update?.update_available && (
                    <span className="installed-badge update">Update</span>
                  )}
                  {update != null &&
                    !update.update_available &&
                    !update.pinned &&
                    !update.error &&
                    update.current_commit != null && (
                      <span className="active-badge">Up to date</span>
                    )}
                  {update?.pinned && <span className="pinned-badge">Pinned</span>}
                </strong>
                <span>
                  {[runtime.version, runtime.target].filter(Boolean).join(' · ')}
                </span>
                {update != null && (
                  <span className="runtime-update-note">
                    {update.error
                      ? `Update check failed: ${update.error}`
                      : update.pinned
                        ? `Pinned to ${update.revision}`
                        : update.update_available
                          ? `Upstream ${update.upstream_commit ?? '?'} · built ${update.current_commit ?? 'unknown'}`
                          : update.current_commit != null
                            ? `Up to date at ${update.current_commit}`
                            : `Upstream ${update.upstream_commit ?? '?'} · rebuild to track the commit`}
                  </span>
                )}
                <code title={runtime.path}>{runtime.path}</code>
              </div>
              <div className="library-card-actions">
                {canRebuild && (
                  <button
                    className="chip-button"
                    title={`Rebuild from ${update.repository} @ ${update.revision}`}
                    onClick={() => rebuildFromUpdate(update)}
                  >
                    <Hammer size={13} />
                    Rebuild
                  </button>
                )}
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
                {runtime.active && (
                  <button
                    className="chip-button subtle"
                    disabled={busyRuntime != null}
                    onClick={() => void deactivate(runtime.id)}
                  >
                    {busyRuntime === runtime.id ? <LoaderCircle className="spin" size={13} /> : 'Deactivate'}
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
              {runtime.engine === 'vllm' && runtime.active && props.settings && (
                <VllmServedModels settings={props.settings} onSaved={props.onSettingsSaved} onError={props.onError} />
              )}
            </article>
            )
          })}
        </div>
        {updatesChecked && tabInstalledRuntimes.some((runtime) => runtime.kind === 'source') && (
          <p className="model-help">
            Update checks compare each source build against the current upstream ref via git.
            Builds made before commit tracking show the latest upstream commit and offer a rebuild.
          </p>
        )}
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
            buildEngine === 'streaming-asr' ||
            buildEngine === 'stable-diffusion.cpp' ||
            buildEngine === 'personaplex' ||
            buildEngine === 'personaplex-mlx' ||
            buildEngine === 'whisperkit' ? (
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
                      'whisperkit',
                      'mlx-lm',
                      'mlx-vlm',
                      'vllm',
                      'streaming-asr',
                      'stable-diffusion.cpp',
                      'personaplex',
                      'personaplex-mlx'
                    ] as BuildEngine[]
                  )
                    .filter((engine) => engineBuildable(engine, props.hardware))
                    .map((engine) => (
                      <option key={engine} value={engine}>
                        {BUILD_ENGINE_LABELS[engine]}
                      </option>
                    ))}
                </select>
              </label>
            ) : null}
            {!isStreamingAsrBuild && (
              <label>
                <span>Build name</span>
                <input value={buildName} onChange={(event) => setBuildName(event.target.value)} placeholder="Optional label" />
              </label>
            )}
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
                  <input
                    value={revision}
                    placeholder="Default branch"
                    onChange={(event) => setRevision(event.target.value)}
                  />
                </label>
              )}
              {/* The prose above says what a build needs; this says what this
                machine has, with the command to fix it. A build that fails
                twenty minutes in because cmake is missing is a worse way to
                learn it, and nothing here elevates or installs on its own. */}
              <ToolchainChecklist tools={toolchainTools} />
              {(buildEngine === 'vllm' || (!isPythonBuild && !isSwiftBuild)) && (
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
                : isSwiftBuild
                  ? 'WhisperKit builds require Xcode or the Swift toolchain. CoreML models download on first transcription (or install via `brew install whisperkit-cli`).'
                  : isPythonBuild
                    ? 'MLX builds create an isolated Python environment with uv. Install uv (`brew install uv`) before starting the build.'
                    : isWhisperBuild
                      ? 'whisper.cpp builds produce the whisper-cli binary used to transcribe audio and video soundtracks before chat.'
                      : buildEngine === 'stable-diffusion.cpp'
                        ? "stable-diffusion.cpp defaults to Brazier's reviewed commit; edit the revision above to opt into another upstream version."
                      : props.hardware?.os === 'macos'
                        ? 'macOS builds use Xcode Command Line Tools. Metal is the recommended GPU target.'
                        : props.hardware?.os === 'windows'
                          ? 'Windows builds need Git, CMake, and Visual Studio 2022 Build Tools with the C++ workload.'
                          : 'Linux builds need git, cmake, and a distro C++ toolchain. GPU targets also need the matching SDK or driver stack.'}
            </p>
            {!isPythonBuild && !isSwiftBuild && (
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
                disabled={building || !repository.trim()}
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
            <JobProgressPanel progress={buildProgress} active={building} showLog={false} />
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
      <div className="settings-group">
        <div className="section-label">Generated media in chat</div>
        <p className="model-help">
          After a chat tool generates something, show it back to the model when the model can see
          images. It can then check its own output against what was asked for.
        </p>
        <label className="settings-toggle">
          <input
            type="checkbox"
            checked={draft.show_generated_images_to_model ?? true}
            onChange={(event) =>
              setDraft({ ...draft, show_generated_images_to_model: event.target.checked })
            }
          />
          <span>
            Show generated images to the model
            <small>Adds one image per generation to the conversation.</small>
          </span>
        </label>
        <label className="settings-toggle">
          <input
            type="checkbox"
            checked={draft.show_generated_video_to_model ?? false}
            onChange={(event) =>
              setDraft({ ...draft, show_generated_video_to_model: event.target.checked })
            }
          />
          <span>
            Show generated video to the model
            <small className="settings-warning">
              More expensive: a clip is sampled into several frames, so each one costs many times
              an image in context and time. Needs ffmpeg.
            </small>
          </span>
        </label>
      </div>
      <div className="settings-group">
        <div className="section-label">Generation & voice defaults</div>
        <p className="model-help">
          These models are used by chat tools (`generate_image` / `generate_video`) and as defaults
          in Generate / Voice modes.
        </p>
        <div className="settings-grid">
          <label>
            <span>Default image model</span>
            <select
              value={draft.default_image_gen_model ?? ''}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  default_image_gen_model: event.target.value || null
                })
              }
            >
              <option value="">None</option>
              {props.models
                .filter((model) => model.id.startsWith('sdcpp-image:'))
                .map((model) => (
                  <option key={model.id} value={model.id}>
                    {modelDisplayName(model.id, model).title}
                  </option>
                ))}
            </select>
          </label>
          <label>
            <span>Default video model</span>
            <select
              value={draft.default_video_gen_model ?? ''}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  default_video_gen_model: event.target.value || null
                })
              }
            >
              <option value="">None</option>
              {props.models
                .filter((model) => model.id.startsWith('sdcpp-video:'))
                .map((model) => (
                  <option key={model.id} value={model.id}>
                    {modelDisplayName(model.id, model).title}
                  </option>
                ))}
            </select>
          </label>
          <label>
            <span>Generation timeout (seconds)</span>
            <input
              type="number"
              min={0}
              max={86400}
              step={60}
              value={draft.generation_timeout_secs ?? 0}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  generation_timeout_secs: Math.max(0, Number(event.target.value) || 0)
                })
              }
            />
            <small className="model-help">
              0 lets Brazier work it out from the frames and steps asked for, which suits most
              machines. Raise it if a slow, CPU-only host is being cut off while still rendering;
              a running job can always be stopped by hand.
            </small>
          </label>
          <label>
            <span>Default voice model</span>
            <select
              value={draft.default_voice_model ?? ''}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  default_voice_model: event.target.value || null
                })
              }
            >
              <option value="">None</option>
              {props.models
                .filter((model) => model.id.startsWith('personaplex:'))
                .map((model) => (
                  <option key={model.id} value={model.id}>
                    {modelDisplayName(model.id, model).title}
                  </option>
                ))}
            </select>
          </label>
          <label className="span-2">
            <span>Default voice persona</span>
            <input
              value={draft.default_voice_persona ?? ''}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  default_voice_persona: event.target.value || null
                })
              }
              placeholder="You are a helpful assistant."
            />
          </label>
        </div>
      </div>
      <div className="settings-group">
        <div className="section-label">Generation memory</div>
        <p className="model-help">
          Image and video models load their own weights. On shared-memory machines they can exceed
          RAM alongside a resident chat model. Auto evicts the chat model only when it will not fit,
          then reloads it when generation finishes.
        </p>
        <div className="settings-grid">
          <label>
            <span>When generating media</span>
            <select
              value={draft.generation_memory_policy}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  generation_memory_policy: event.target
                    .value as RuntimeSettings['generation_memory_policy']
                })
              }
            >
              <option value="auto">Auto — evict chat model only if needed</option>
              <option value="coresident">Keep both models loaded</option>
              <option value="exclusive">Always evict chat model</option>
            </select>
          </label>
          <label>
            <span>RAM headroom (MiB)</span>
            <input
              type="number"
              min={0}
              step={256}
              value={draft.generation_memory_headroom_mb}
              disabled={draft.generation_memory_policy !== 'auto'}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  generation_memory_headroom_mb: Number(event.target.value)
                })
              }
            />
          </label>
        </div>
        <div className="toggle-list">
          <label>
            <div>
              <strong>Reload chat model after generation</strong>
              <span>Bring the evicted chat model back once media generation completes.</span>
            </div>
            <input
              type="checkbox"
              checked={draft.reload_llm_after_generation}
              disabled={draft.generation_memory_policy === 'coresident'}
              onChange={(event) =>
                setDraft({ ...draft, reload_llm_after_generation: event.target.checked })
              }
            />
          </label>
        </div>
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
          <small className="model-help">
            -1 automatically sizes GPU offload from the model and accelerator-memory budget.
          </small>
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
            [
              'jinja',
              'Jinja templates',
              'Always enabled for llama-server so chat templates can parse tools'
            ]
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

/** Pick the web search backend and how keyless searches behave when blocked. */
function WebSearchSection(props: SectionProps): React.JSX.Element {
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
          <h2>Web search</h2>
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
        <h2>Web search</h2>
        <p>
          Choose the search engine behind the <code>web_search</code> tool. Both chat and agent
          search use it. DuckDuckGo needs no key; Brave needs a paid API key and is the reliable
          option when DuckDuckGo rate-limits this machine.
        </p>
      </header>
      <div className="settings-group">
        <div className="section-label">Backend</div>
        <p className="model-help">
          DuckDuckGo is keyless and works for most queries, but it sometimes serves a bot-check
          challenge to honest clients. Brave (1000 free searches/month, then $5 per extra 1000)
          has no such limit — if DuckDuckGo starts failing, add a Brave key and switch.
        </p>
        <div className="settings-grid">
          <label>
            <span>Provider</span>
            <select
              value={draft.web_search_provider ?? 'duckduckgo'}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  web_search_provider: event.target.value as 'duckduckgo' | 'brave'
                })
              }
            >
              <option value="duckduckgo">DuckDuckGo (no key)</option>
              <option value="brave">Brave Search API</option>
            </select>
          </label>
          <label>
            <span>SafeSearch</span>
            <select
              value={draft.web_safesearch ?? 'moderate'}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  web_safesearch: event.target.value as 'moderate' | 'strict' | 'off'
                })
              }
            >
              <option value="moderate">Moderate</option>
              <option value="strict">Strict</option>
              <option value="off">Off</option>
            </select>
          </label>
          <label>
            <span>Region (DuckDuckGo)</span>
            <input
              type="text"
              placeholder="us-en, de-de, wt-wt"
              value={draft.web_search_region ?? ''}
              onChange={(event) =>
                setDraft({ ...draft, web_search_region: event.target.value || null })
              }
            />
          </label>
        </div>
        {draft.web_search_provider === 'brave' && (
          <label className="setting-row">
            <span>Brave API key</span>
            <input
              type="password"
              autoComplete="off"
              placeholder={draft.brave_api_key ? 'Stored — enter a replacement to rotate' : 'Required'}
              value={draft.brave_api_key ?? ''}
              onChange={(event) =>
                setDraft({ ...draft, brave_api_key: event.target.value || null })
              }
            />
          </label>
        )}
        {draft.web_search_provider === 'brave' && !draft.brave_api_key && (
          <p className="settings-warning">
            Brave is selected but no API key is saved; web search will fail until one is set.
          </p>
        )}
      </div>
      <div className="runtime-actions">
        <button className="primary-action" disabled={saving || !dirty} onClick={() => void apply()}>
          {saving ? <LoaderCircle className="spin" size={15} /> : 'Apply & restart'}
        </button>
      </div>
    </section>
  )
}

/** Configure Brazier's own OpenAI-compatible daemon without exposing secrets to React state. */
function ServerSection(props: SectionProps): React.JSX.Element {
  const [settings, setSettings] = useState<Awaited<ReturnType<typeof window.brazier.getServerSettings>> | null>(null)
  const [keyName, setKeyName] = useState('')
  const [revealed, setRevealed] = useState<{ id: string; name: string; value: string } | null>(null)
  const [copied, setCopied] = useState(false)
  const [saving, setSaving] = useState(false)
  const [adding, setAdding] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const revealTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    void window.brazier.getServerSettings().then(setSettings).catch((cause) => props.onError(errorText(cause)))
    return () => {
      if (revealTimer.current) clearTimeout(revealTimer.current)
    }
  }, [])

  function reveal(key: { id: string; name: string; value: string }): void {
    if (revealTimer.current) clearTimeout(revealTimer.current)
    setRevealed(key)
    setCopied(false)
    // The full key is never stored in React state beyond this brief window.
    // Auto-dismiss keeps it from lingering on screen for shoulder surfers.
    revealTimer.current = setTimeout(() => setRevealed(null), 30_000)
  }

  async function copyKey(value: string): Promise<void> {
    await window.brazier.copyText(value)
    setCopied(true)
  }

  async function addKey(): Promise<void> {
    setAdding(true)
    props.onError(null)
    try {
      const key = await window.brazier.addServerApiKey(keyName)
      setKeyName('')
      if (settings) setSettings({ ...settings, hasApiKeys: true, keys: [...settings.keys, { id: key.id, name: key.name, createdAt: key.createdAt }] })
      reveal(key)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setAdding(false)
    }
  }

  async function removeKey(id: string): Promise<void> {
    props.onError(null)
    try {
      setSettings(await window.brazier.removeServerApiKey(id))
      if (revealed?.id === id) setRevealed(null)
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  async function save(): Promise<void> {
    if (!settings) return
    setSaving(true)
    props.onError(null)
    try {
      const saved = await window.brazier.saveServerSettings({
        enabled: settings.enabled,
        port: settings.port,
        apiKeyEnabled: settings.apiKeyEnabled,
        localhostOnly: settings.localhostOnly,
        jitLoading: settings.jitLoading
      })
      setSettings(saved)
      setNotice('Saved. Restart Brazier to apply server exposure, port, authentication, or JIT changes.')
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setSaving(false)
    }
  }

  if (!settings) return <section><div className="manage-placeholder"><LoaderCircle className="spin" size={16} />Loading server settings…</div></section>

  const baseUrl = settings.enabled
    ? `http://${settings.localhostOnly ? '127.0.0.1' : '<this-machine>'}:${settings.port}/v1`
    : 'Private to Brazier desktop'
  const keysMissing = settings.enabled && settings.apiKeyEnabled && settings.keys.length === 0
  const saveDisabled = saving || keysMissing

  return (
    <section>
      <header className="manage-heading">
        <h2>OpenAI-compatible server</h2>
        <p>Expose Brazier’s Chat Completions, Responses, Models, transcription, and generation APIs to clients that speak the OpenAI protocol.</p>
      </header>
      <div className="settings-group">
        <div className="section-label">Network access</div>
        <label className="settings-toggle">
          <input type="checkbox" checked={settings.enabled} onChange={(event) => setSettings({ ...settings, enabled: event.target.checked })} />
          <span>Enable network server<small>Off keeps the daemon private to this desktop app. On serves the OpenAI-compatible API on the configured port.</small></span>
        </label>
        <label className="settings-toggle">
          <input type="checkbox" checked={settings.localhostOnly} disabled={!settings.enabled} onChange={(event) => setSettings({ ...settings, localhostOnly: event.target.checked })} />
          <span>Listen on localhost only<small>On accepts connections only from this machine (127.0.0.1) — nothing else on your network can reach the server. Off listens on every local network interface.</small></span>
        </label>
        <div className="settings-grid">
          <label><span>Port</span><input type="number" min={1} max={65535} disabled={!settings.enabled} value={settings.port} onChange={(event) => setSettings({ ...settings, port: Number(event.target.value) })} /></label>
          <label><span>Base URL</span><input readOnly value={baseUrl} /></label>
        </div>
      </div>
      <div className="settings-group">
        <div className="section-label">Authentication</div>
        <label className="settings-toggle">
          <input type="checkbox" checked={settings.apiKeyEnabled} disabled={!settings.enabled} onChange={(event) => setSettings({ ...settings, apiKeyEnabled: event.target.checked })} />
          <span>Require API key<small>Every named key below is accepted. Keys are stored securely and shown in full only once, right after you create them.</small></span>
        </label>
        {settings.keys.length > 0 && (
          <div className="server-key-list">
            {settings.keys.map((key) => (
              <div className="server-key-row" key={key.id}>
                <span className="server-key-name">{key.name || 'Unnamed key'}</span>
                <span className="server-key-meta">Created {new Date(key.createdAt).toLocaleDateString()}</span>
                <button className="chip-button subtle" onClick={() => void removeKey(key.id)} title="Revoke this key" disabled={!settings.apiKeyEnabled}>
                  <Trash2 size={13} />
                </button>
              </div>
            ))}
          </div>
        )}
        <div className="server-key-add">
          <label className="setting-row"><span>New key name</span><input type="text" value={keyName} placeholder="e.g. VS Code extension" maxLength={60} disabled={!settings.enabled || !settings.apiKeyEnabled} onChange={(event) => setKeyName(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') void addKey() }} /></label>
          <div className="runtime-actions"><button className="chip-button" onClick={() => void addKey()} disabled={!settings.enabled || !settings.apiKeyEnabled || adding}>{adding ? <LoaderCircle className="spin" size={13} /> : <KeyRound size={13} />}Generate key</button></div>
        </div>
        {revealed && (
          <div className="server-key-reveal">
            <div className="server-key-reveal-head">
              <strong>Key “{revealed.name}” — shown only once</strong>
              <span>It disappears automatically; copy it before then.</span>
            </div>
            <code>{revealed.value}</code>
            <div className="runtime-actions">
              <button className="chip-button" onClick={() => void copyKey(revealed.value)}>{copied ? <Check size={13} /> : <Copy size={13} />}{copied ? 'Copied' : 'Copy key'}</button>
              <button className="chip-button subtle" onClick={() => setRevealed(null)}>Hide</button>
            </div>
          </div>
        )}
        {!settings.apiKeyEnabled && settings.enabled && <p className="settings-warning">Anyone who can reach this server can call Brazier, including agent and management APIs. Keep this off only on an isolated, trusted network.</p>}
        {keysMissing && <p className="settings-warning">Add at least one API key before enabling the server with authentication.</p>}
      </div>
      <div className="settings-group">
        <div className="section-label">Model loading</div>
        <label className="settings-toggle">
          <input type="checkbox" checked={settings.jitLoading} onChange={(event) => setSettings({ ...settings, jitLoading: event.target.checked })} />
          <span>Enable JIT model loading<small>Allow API requests to start the selected local model when it is not already resident.</small></span>
        </label>
      </div>
      {notice && <p className="model-help">{notice}</p>}
      <div className="runtime-actions"><button className="primary-action" disabled={saveDisabled} onClick={() => void save()}>{saving ? <LoaderCircle className="spin" size={15} /> : 'Save server settings'}</button></div>
    </section>
  )
}

/**
 * Servers someone else is running, speaking the same OpenAI-compatible protocol
 * the local engines do.
 *
 * Kept deliberately plain: a URL, a label, an optional key. Nothing is
 * discovered on the network — a local application that scans for open ports is
 * doing something its user did not ask for — and a connection is contacted only
 * when it is saved, tested, or listed.
 */
function RemoteSection(props: SectionProps): React.JSX.Element {
  const [connections, setConnections] = useState<RemoteConnection[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState<string | null>(null)
  const [results, setResults] = useState<
    Record<string, { reachable: boolean; models: string[]; error?: string }>
  >({})
  const [draft, setDraft] = useState({ id: '', label: '', base_url: '', api_key: '', llama_cpp_compatible: false })

  async function reload(): Promise<void> {
    setLoading(true)
    props.onError(null)
    try {
      setConnections(await listRemoteConnections())
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void reload()
  }, [])

  async function addConnection(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (!draft.id.trim() || !draft.base_url.trim()) return
    setSaving(true)
    props.onError(null)
    try {
      const next = await saveRemoteConnection({
        id: draft.id.trim(),
        label: draft.label.trim() || draft.id.trim(),
        base_url: draft.base_url.trim(),
        // Sent only when typed: an empty field means "leave it alone", which is
        // what editing an existing connection needs.
        ...(draft.api_key.trim() ? { api_key: draft.api_key.trim() } : {}),
        enabled: true,
        llama_cpp_compatible: draft.llama_cpp_compatible
      })
      setConnections(next)
      setDraft({ id: '', label: '', base_url: '', api_key: '', llama_cpp_compatible: false })
      // The model list has different contents now.
      void props.refreshModels()
      await test(draft.id.trim())
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setSaving(false)
    }
  }

  async function toggleEnabled(connection: RemoteConnection): Promise<void> {
    props.onError(null)
    try {
      setConnections(
        await saveRemoteConnection({
          id: connection.id,
          label: connection.label,
          base_url: connection.base_url,
          enabled: !connection.enabled,
          llama_cpp_compatible: connection.llama_cpp_compatible
        })
      )
      void props.refreshModels()
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  async function setLlamaCppCompatible(
    connection: RemoteConnection,
    llama_cpp_compatible: boolean
  ): Promise<void> {
    props.onError(null)
    try {
      setConnections(
        await saveRemoteConnection({
          id: connection.id,
          label: connection.label,
          base_url: connection.base_url,
          enabled: connection.enabled,
          llama_cpp_compatible
        })
      )
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  async function remove(id: string): Promise<void> {
    props.onError(null)
    try {
      setConnections(await deleteRemoteConnection(id))
      void props.refreshModels()
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  async function test(id: string): Promise<void> {
    setTesting(id)
    try {
      const result = await testRemoteConnection(id)
      setResults((current) => ({ ...current, [id]: result }))
    } catch (cause) {
      setResults((current) => ({
        ...current,
        [id]: { reachable: false, models: [], error: errorText(cause) }
      }))
    } finally {
      setTesting(null)
    }
  }

  return (
    <section>
      <header className="manage-heading">
        <h2>Remote servers</h2>
        <p>
          Talk to an OpenAI-compatible server you already run — vLLM, llama-server, or anything
          else speaking the same protocol. Its models appear in the model list, marked with the
          connection they came from, so a conversation records where its answers were produced.
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
            {connections.length === 0 ? (
              <p className="model-help">No remote servers configured.</p>
            ) : (
              connections.map((connection) => {
                const result = results[connection.id]
                return (
                  <article className="runtime-card" key={connection.id}>
                    <div className="runtime-card-info">
                      <strong>{connection.label}</strong>
                      <span>{connection.base_url}</span>
                      <span>
                        {connection.has_api_key ? 'API key stored' : 'No API key'}
                        {connection.llama_cpp_compatible ? ' · llama.cpp KV hints' : ''}
                        {result
                          ? result.reachable
                            ? ` · ${result.models.length} model${
                                result.models.length === 1 ? '' : 's'
                              }`
                            : ` · unreachable: ${result.error ?? 'no reason given'}`
                          : ''}
                      </span>
                    </div>
                    <div className="library-card-actions">
                      <label className="chip-button subtle" title="Send llama.cpp cache_prompt and id_slot on requests">
                        <input
                          type="checkbox"
                          checked={connection.llama_cpp_compatible}
                          onChange={(event) =>
                            void setLlamaCppCompatible(connection, event.target.checked)
                          }
                        />
                        llama.cpp
                      </label>
                      <label className="chip-button subtle" title="Use this server">
                        <input
                          type="checkbox"
                          checked={connection.enabled}
                          onChange={() => void toggleEnabled(connection)}
                        />
                        Enabled
                      </label>
                      <button
                        className="chip-button"
                        disabled={testing === connection.id}
                        onClick={() => void test(connection.id)}
                      >
                        {testing === connection.id ? (
                          <LoaderCircle className="spin" size={13} />
                        ) : (
                          'Test'
                        )}
                      </button>
                      <button
                        className="chip-button danger"
                        onClick={() => void remove(connection.id)}
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </article>
                )
              })
            )}
          </div>

          <form className="settings-group" onSubmit={(event) => void addConnection(event)}>
            <div className="section-label">Add a server</div>
            <label className="setting-row">
              <span>Name</span>
              <input
                value={draft.id}
                onChange={(event) => setDraft({ ...draft, id: event.target.value })}
                placeholder="workstation"
              />
            </label>
            <label className="setting-row">
              <span>Label</span>
              <input
                value={draft.label}
                onChange={(event) => setDraft({ ...draft, label: event.target.value })}
                placeholder="Workstation (vLLM)"
              />
            </label>
            <label className="setting-row">
              <span>Base URL</span>
              <input
                value={draft.base_url}
                onChange={(event) => setDraft({ ...draft, base_url: event.target.value })}
                placeholder="http://10.0.0.4:8000"
              />
            </label>
            <label className="setting-row">
              <span>API key</span>
              <input
                type="password"
                value={draft.api_key}
                onChange={(event) => setDraft({ ...draft, api_key: event.target.value })}
                placeholder="Optional"
              />
            </label>
            <label className="settings-toggle">
              <input
                type="checkbox"
                checked={draft.llama_cpp_compatible}
                onChange={(event) =>
                  setDraft({ ...draft, llama_cpp_compatible: event.target.checked })
                }
              />
              <span>llama.cpp server (send KV cache slot hints)</span>
            </label>
            <p className="model-help">
              Requests leave this machine. A server reached over plain HTTP carries your
              conversation in the clear, which is fine on a network you trust and not otherwise.
            </p>
            <button className="primary" type="submit" disabled={saving}>
              {saving ? <LoaderCircle className="spin" size={14} /> : 'Add server'}
            </button>
          </form>
        </>
      )}
    </section>
  )
}

/**
 * What this machine has of what a source build needs.
 *
 * Install commands are shown, never run: elevation belongs to the user's own
 * shell, where they can see what it is about to do.
 */
function ToolchainChecklist({ tools }: { tools: ToolchainTool[] }): React.JSX.Element | null {
  if (tools.length === 0) return null
  return (
    <div className="toolchain-checklist">
      {tools.map((tool) => (
        <div className={`toolchain-tool ${tool.available ? 'ok' : 'missing'}`} key={tool.id}>
          <span className="toolchain-tool-name">
            {tool.available ? <Check size={13} /> : <ShieldAlert size={13} />}
            {tool.label}
          </span>
          {tool.available ? (
            <span className="toolchain-tool-detail" title={tool.required_for}>
              {tool.path ?? 'found'}
            </span>
          ) : (
            <code className="toolchain-tool-detail" title={tool.required_for}>
              {tool.install_hint ?? 'not found'}
            </code>
          )}
        </div>
      ))}
    </div>
  )
}

function AgentSection(props: SectionProps): React.JSX.Element {
  const [runtimes, setRuntimes] = useState<AgentRuntimeInfo[]>([])
  const [selected, setSelected] = useState('simple')
  const [powerToolsCatalog, setPowerToolsCatalog] = useState<AgentToolCatalogEntry[]>([])
  const [powerTools, setPowerTools] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [savedNotice, setSavedNotice] = useState<string | null>(null)

  async function reload(): Promise<void> {
    setLoading(true)
    props.onError(null)
    try {
      const [capabilities, preference, tools] = await Promise.all([
        fetchAgentCapabilities(),
        fetchAgentPreference(),
        fetchAgentTools()
      ])
      const catalog = tools.filter((tool) => tool.power_tool === true)
      setRuntimes(capabilities.runtimes)
      setSelected(preference.default_runtime_id || capabilities.default_runtime_id || 'simple')
      setPowerTools(preference.power_tools ?? catalog.map((tool) => tool.name))
      setPowerToolsCatalog(catalog)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void reload()
  }, [])

  async function persist(defaultRuntimeId: string, nextPowerTools: string[]): Promise<void> {
    setSaving(true)
    props.onError(null)
    setSavedNotice(null)
    try {
      const saved = await saveAgentPreference({
        default_runtime_id: defaultRuntimeId,
        power_tools: nextPowerTools
      })
      setSelected(saved.default_runtime_id)
      setPowerTools(saved.power_tools ?? [])
      setSavedNotice(
        saved.default_runtime_id === 'powerful'
          ? 'New agent tasks will use Powerful mode.'
          : 'New agent tasks will use Simple mode.'
      )
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setSaving(false)
    }
  }

  async function chooseRuntime(runtimeId: string): Promise<void> {
    const entry = runtimes.find((runtime) => runtime.id === runtimeId)
    if (!entry || entry.available === false) return
    await persist(runtimeId, powerTools)
  }

  function togglePowerTool(name: string, on: boolean): void {
    const next = on
      ? [...powerTools.filter((entry) => entry !== name), name]
      : powerTools.filter((entry) => entry !== name)
    setPowerTools(next)
    void persist(selected, next)
  }

  return (
    <section>
      <header className="manage-heading">
        <h2>Agent</h2>
        <p>
          Agent tasks run through two modes. Simple exposes the standard broker-sandboxed tool set.
          Powerful adds the extra tools you enable below; both keep approvals, sandboxing, and
          execution under one trust boundary.
        </p>
      </header>

      {loading ? (
        <div className="manage-placeholder">
          <LoaderCircle className="spin" size={16} />
          Loading…
        </div>
      ) : (
        <>
          <div className="settings-group">
            <div className="section-label">Agent mode</div>
            <div className="agent-runtime-choice">
              {runtimes.map((runtime) => {
                const unavailable = runtime.available === false
                return (
                  <label
                    key={runtime.id}
                    className={[
                      selected === runtime.id ? 'active' : '',
                      unavailable ? 'disabled' : ''
                    ]
                      .filter(Boolean)
                      .join(' ')}
                  >
                    <input
                      type="radio"
                      name="agent-runtime"
                      value={runtime.id}
                      checked={selected === runtime.id}
                      disabled={unavailable || saving}
                      onChange={() => void chooseRuntime(runtime.id)}
                    />
                    <span>
                      <strong>{runtime.name}</strong>
                      <small>
                        {runtime.id === 'simple'
                          ? 'A low bloat harness for lightweight work or low-context models'
                          : 'A feature-rich harness for more in depth work'}
                      </small>
                      {unavailable && runtime.unavailable_reason && (
                        <small>{runtime.unavailable_reason}</small>
                      )}
                    </span>
                  </label>
                )
              })}
            </div>
            {savedNotice && <p className="model-help">{savedNotice}</p>}
          </div>

          <div className="settings-group">
            <div className="section-label">Powerful mode tools</div>
            <p className="model-help">
              These extra tools are only exposed to Powerful mode sessions. They default to on;
              toggle the ones you want new Powerful tasks to start with. Each task can still trim
              its own set in the Agent header&apos;s Tools menu.
            </p>
            {selected !== 'powerful' && (
              <p className="model-help">
                Powerful mode is not the default yet. Switch the mode above to use these tools.
              </p>
            )}
            {powerToolsCatalog.length === 0 ? (
              <p className="model-help">No power tools yet — they land in a later build.</p>
            ) : (
              <div className="power-tools-list">
                {powerToolsCatalog.map((tool) => {
                  const on = powerTools.includes(tool.name)
                  return (
                    <label
                      key={tool.name}
                      className={`power-tool${on ? ' active' : ''}`}
                    >
                      <input
                        type="checkbox"
                        checked={on}
                        disabled={saving}
                        onChange={(event) => togglePowerTool(tool.name, event.target.checked)}
                      />
                      <span className="power-tool-label">
                        <strong>{tool.label}</strong>
                        <small>{tool.description}</small>
                      </span>
                    </label>
                  )
                })}
              </div>
            )}
          </div>

          <div className="settings-group">
            <div className="section-label">Trust boundary</div>
            <p className="model-help">
              The agent worker reaches the machine only through <code>POST /api/v1/agent/exec</code>.
              The daemon applies Seatbelt or Bubblewrap when available, binds approvals to tool
              arguments, and can refuse host escape in sandbox-only mode.
            </p>
          </div>
        </>
      )}
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

  function applyRecipe(recipe: McpRecipe): void {
    setDraft({
      id: recipe.id,
      name: recipe.name,
      command: recipe.command,
      args: recipe.args.join(',')
    })
  }

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

          <div className="settings-group">
            <div className="section-label">Popular recipes</div>
            <div className="runtime-offer-list">
              {MCP_RECIPES.map((recipe) => {
                const configured = servers.some((server) => server.id === recipe.id)
                return (
                  <article className="runtime-offer" key={recipe.id}>
                    <div className="runtime-offer-info">
                      <strong>{recipe.name}</strong>
                      <span>{recipe.description}</span>
                      <span>{recipe.setup}</span>
                    </div>
                    <button
                      className="secondary-action"
                      disabled={configured}
                      onClick={() => applyRecipe(recipe)}
                      title={configured ? 'This recipe is already configured' : 'Fill the editable server form'}
                    >
                      {configured ? 'Configured' : 'Use recipe'}
                    </button>
                  </article>
                )
              })}
            </div>
            <p className="model-help">
              Recipes fill the form below. Review or change the command and arguments before adding a
              server.
            </p>
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

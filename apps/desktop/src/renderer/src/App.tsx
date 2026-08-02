import {
  AudioLines,
  Bot,
  Box,
  Brain,
  Cpu,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Ellipsis,
  Gauge,
  Image,
  LoaderCircle,
  Menu,
  MessageSquarePlus,
  Mic,
  Paperclip,
  Download,
  RefreshCw,
  Search,
  Send,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  Square,
  Pencil,
  Trash2,
  Upload,
  Video,
  Wrench,
  X
} from 'lucide-react'
import { type ChangeEvent, type FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  createConversation,
  createMessage,
  deleteConversation,
  engineStatus,
  fetchCapabilities,
  fetchModelBindings,
  fetchRecommendations,
  fetchModelSettings,
  health,
  exportConversation,
  importConversation,
  listConversations,
  listMessages,
  listModels,
  listRuntimes,
  listRunSnapshots,
  listTools,
  type BundledTool,
  GenerationFailure,
  prepareModel,
  prefillProgressLabel,
  setModelBinding,
  type ConversationExport,
  type HardwareInfo,
  type LocalModel,
  type ModelProfile,
  type ModelLoadMode,
  type ModelResidency,
  type PipelineFeatures,
  type PendingSwap,
  type RunSnapshot,
  type RuntimeEntry,
  type RuntimeForkHint,
  type RuntimeSettings,
  type ToolCallRecord,
  fetchWorkspacePreference,
  recordRun,
  saveRuntimeSettings,
  streamCompletion,
  unloadModel,
  updateConversation,
  updateRecommendationState,
  uploadAttachmentBlob,
  type WorkspaceModesPreference
} from './api'
import { AgentMode, type AgentComposerControls, type AgentSidebarControls } from './components/AgentMode'
import { AgentSessionSidebar } from './components/AgentSessionSidebar'
import { ComputerMode } from './components/ComputerMode'
import { DownloadTray } from './components/DownloadTray'
import { GenerationActivity } from './components/GenerationActivity'
import {
  GenerateHistorySidebar,
  type GenerateHistoryEntry
} from './components/GenerateHistorySidebar'
import { MessageMedia } from './components/MessageMedia'
import { Markdown } from './components/Markdown'
import { ReasoningDisclosure } from './components/ReasoningDisclosure'
import { GenerateMode } from './components/GenerateMode'
import { InferenceMenu } from './components/InferenceMenu'
import { ManagePanel, type ManageSection } from './components/ManagePanel'
import { ModelMenu } from './components/ModelMenu'
import { profileCount } from './components/ModelSettingsFields'
import { ModelSettingsModal } from './components/ModelSettingsModal'
import { ToolsMenu } from './components/ToolsMenu'
import { VoiceMode } from './components/VoiceMode'
import { WelcomeScreen } from './components/WelcomeScreen'
import { hasCompletedWelcome, markWelcomeCompleted } from './welcomePrefs'
import { voiceStreamSupported } from './audio/voiceStream'
import { useSessionCoordinator } from './session/useSessionCoordinator'
import {
  isChatModel,
  isComputerUseModel,
  isImageGenModel,
  isVideoGenModel,
  isVoiceModel,
  modelDisplayName,
  modelKindFor,
  runtimeNoticeForModel,
  visionCapabilityTitle
} from './model-utils'
import { branchSiblings, messageChain } from './graph'
import { buildChatDisplayItems } from './chatDisplay'
import {
  readCachedConversations,
  readCachedModels,
  readCachedRuntimes,
  writeCachedConversations,
  writeCachedModels,
  writeCachedRuntimes
} from './inventoryCache'
import type { Attachment, ContentPart, Conversation, Message, Role } from './types'

const ENABLED_TOOLS_KEY = 'brazier.enabledTools'
const CHAT_TITLE_MODE_KEY = 'brazier.chatTitleMode.v1'
const GENERATE_HISTORY_KEY = 'brazier.generateHistory.v1'

type AppMode = 'chat' | 'agent' | 'generate' | 'voice' | 'computer'

const DEFAULT_WORKSPACE_MODES: WorkspaceModesPreference = {
  chat: true,
  agent: true,
  generate: true,
  voice: true,
  computer: false
}

type ChatTitleMode = 'never' | 'always' | 'over-20-tokens'

function readGenerateHistory(): GenerateHistoryEntry[] {
  try {
    const value = localStorage.getItem(GENERATE_HISTORY_KEY)
    const parsed = value ? (JSON.parse(value) as unknown) : []
    return Array.isArray(parsed) ? (parsed as GenerateHistoryEntry[]) : []
  } catch {
    return []
  }
}

function writeGenerateHistory(entries: GenerateHistoryEntry[]): void {
  try {
    localStorage.setItem(GENERATE_HISTORY_KEY, JSON.stringify(entries.slice(0, 100)))
  } catch {
    // Best-effort local history.
  }
}

function readChatTitleMode(): ChatTitleMode {
  try {
    const value = localStorage.getItem(CHAT_TITLE_MODE_KEY)
    return value === 'never' || value === 'over-20-tokens' || value === 'always' ? value : 'always'
  } catch {
    return 'always'
  }
}

function titleFromCompletion(text: string): string | null {
  const title = text
    .replace(/[\r\n]+/g, ' ')
    .replace(/^\s*["'`]+|["'`]+\s*$/g, '')
    .trim()
  return title ? title.slice(0, 80) : null
}

function readEnabledTools(): string[] {
  try {
    const raw = localStorage.getItem(ENABLED_TOOLS_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as unknown
    return Array.isArray(parsed) ? parsed.filter((value): value is string => typeof value === 'string') : []
  } catch {
    return []
  }
}

function writeEnabledTools(names: string[]): void {
  try {
    localStorage.setItem(ENABLED_TOOLS_KEY, JSON.stringify(names))
  } catch {
    // Best-effort persistence.
  }
}

function BranchNavigator({
  messages,
  messageId,
  onSelect
}: {
  messages: Message[]
  messageId: string
  onSelect: (id: string) => void
}): React.JSX.Element | null {
  const siblings = branchSiblings(messages, messageId)
  const index = siblings.findIndex((message) => message.id === messageId)
  if (siblings.length < 2 || index < 0) return null
  return (
    <div className="branch-navigator" aria-label={`Branch ${index + 1} of ${siblings.length}`}>
      <button
        type="button"
        title="Previous branch"
        aria-label="Previous branch"
        disabled={index === 0}
        onClick={() => onSelect(siblings[index - 1].id)}
      >
        <ChevronLeft size={13} />
      </button>
      <span>{index + 1}/{siblings.length}</span>
      <button
        type="button"
        title="Next branch"
        aria-label="Next branch"
        disabled={index === siblings.length - 1}
        onClick={() => onSelect(siblings[index + 1].id)}
      >
        <ChevronRight size={13} />
      </button>
    </div>
  )
}

function contentText(message: Message): string {
  if (typeof message.content === 'string') return message.content
  return message.content
    .filter((part): part is Extract<ContentPart, { type: 'text' }> => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

function contentMedia(message: Message): Array<'image' | 'audio' | 'video'> {
  if (typeof message.content === 'string') return []
  return message.content.flatMap((part) => {
    if (part.type === 'brazier_blob') {
      if (part.brazier_blob.mime_type.startsWith('image/')) return ['image'] as const
      if (part.brazier_blob.mime_type.startsWith('audio/')) return ['audio'] as const
      if (part.brazier_blob.mime_type.startsWith('video/')) return ['video'] as const
      return []
    }
    if (part.type === 'image_url') return ['image'] as const
    if (part.type === 'input_audio') return ['audio'] as const
    if (part.type === 'input_video') return ['video'] as const
    return []
  })
}

/** Stored blobs attached to a message, which are the ones that can be saved. */
function contentBlobs(
  message: Message
): Array<{ sha256: string; mime_type: string; original_name?: string | null }> {
  if (typeof message.content === 'string') return []
  return message.content.flatMap((part) =>
    part.type === 'brazier_blob'
      ? [
          {
            sha256: part.brazier_blob.sha256,
            mime_type: part.brazier_blob.mime_type,
            original_name: part.brazier_blob.name
          }
        ]
      : []
  )
}

/**
 * Badge for a turn that did not run normally, so queued, cancelled, and
 * superseded messages are visibly different from live ones.
 */
function turnLabel(message: Message): 'queued' | 'cancelled' | 'superseded' | 'failed' | null {
  if (message.status === 'cancelled') return 'cancelled'
  if (message.status === 'superseded') return 'superseded'
  if (message.status === 'failed') return 'failed'
  const metadata = message.metadata ?? {}
  if (metadata.queued === true) return 'queued'
  if (metadata.cancelled === true) return 'cancelled'
  if (metadata.failed === true) return 'failed'
  return null
}

async function fileToAttachment(file: File): Promise<Attachment> {
  const stored = await uploadAttachmentBlob(file)
  return {
    id: crypto.randomUUID(),
    name: file.name,
    type: file.type || stored.mime_type,
    sha256: stored.sha256
  }
}

function isDocumentFile(file: File): boolean {
  if (
    [
      'application/pdf',
      'application/json',
      'application/xml',
      'application/rtf',
      'text/rtf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/plain',
      'text/markdown',
      'text/csv',
      'text/html'
    ].includes(file.type) ||
    file.type.startsWith('text/')
  ) {
    return true
  }
  const extension = file.name.split('.').pop()?.toLowerCase()
  return ['pdf', 'rtf', 'doc', 'docx', 'txt', 'md', 'csv', 'json', 'xml', 'html', 'htm'].includes(
    extension ?? ''
  )
}

function attachmentPart(attachment: Attachment): ContentPart {
  return {
    type: 'brazier_blob',
    brazier_blob: {
      sha256: attachment.sha256,
      mime_type: attachment.type,
      name: attachment.name
    }
  }
}

function ToolChips({
  records,
  onError
}: {
  records: ToolCallRecord[]
  onError?: (message: string) => void
}): React.JSX.Element {
  const media = records.flatMap((record) =>
    (record.media ?? []).map((blob) => ({
      sha256: blob.sha256,
      mime_type: blob.mime_type,
      original_name: blob.name
    }))
  )
  return (
    <>
      <div className="tool-chip-row">
        {records.map((record, index) => (
          <details className={record.is_error ? 'tool-chip error' : 'tool-chip'} key={index}>
            <summary>
              <Wrench size={12} />
              {record.name}
            </summary>
            <div className="tool-chip-body">
              <div>
                <span>Arguments</span>
                <code>{record.arguments}</code>
              </div>
              <div>
                <span>Result</span>
                <code>{record.output}</code>
              </div>
            </div>
          </details>
        ))}
      </div>
      <MessageMedia blobs={media} onError={onError} />
    </>
  )
}

function StreamingTurnSegments({
  text,
  records,
  offsets,
  onError
}: {
  text: string
  records: ToolCallRecord[]
  offsets: number[]
  onError?: (message: string) => void
}): React.JSX.Element {
  const segments: React.JSX.Element[] = []
  let cursor = 0
  records.forEach((record, index) => {
    const offset = Math.max(cursor, Math.min(offsets[index] ?? cursor, text.length))
    if (offset > cursor) {
      segments.push(<Markdown key={`text-before-tool-${index}`}>{text.slice(cursor, offset)}</Markdown>)
    }
    segments.push(<ToolChips key={`stream-tool-${index}`} records={[record]} onError={onError} />)
    cursor = offset
  })
  if (cursor < text.length) {
    segments.push(<Markdown key="text-after-tools">{text.slice(cursor)}</Markdown>)
  }
  return <>{segments}</>
}

function RunHistory({
  runs,
  expandedId,
  onToggle
}: {
  runs: RunSnapshot[]
  expandedId: string | null
  onToggle: (id: string) => void
}): React.JSX.Element | null {
  if (runs.length === 0) return null
  return (
    <div className="run-history">
      <div className="section-label">Run history</div>
      {runs.slice(0, 8).map((run) => {
        const expanded = expandedId === run.id
        const label = run.created_at.replace('T', ' ').slice(0, 16)
        return (
          <div className="run-entry" key={run.id}>
            <button type="button" className="run-entry-head" onClick={() => onToggle(run.id)}>
              <span>{label}</span>
              <span className="run-model">{run.model.split('/').at(-1) ?? run.model}</span>
              {run.error && <span className="run-error">failed</span>}
            </button>
            {expanded && (
              <div className="run-entry-body">
                {run.error && <p className="run-error-text">{run.error}</p>}
                {run.response_text && <p>{run.response_text.slice(0, 400)}</p>}
                {run.tool_calls && run.tool_calls.length > 0 && (
                  <ToolChips records={run.tool_calls} />
                )}
                <details>
                  <summary>Settings snapshot</summary>
                  <pre>{JSON.stringify(run.settings, null, 2)}</pre>
                </details>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function App(): React.JSX.Element {
  const [conversations, setConversations] = useState<Conversation[]>(() => readCachedConversations())
  const [conversationSearch, setConversationSearch] = useState('')
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [chatTitleMode, setChatTitleMode] = useState<ChatTitleMode>(() => readChatTitleMode())
  const [conversationMenuId, setConversationMenuId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [tipId, setTipId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [streamingText, setStreamingText] = useState('')
  const [streamingReasoning, setStreamingReasoning] = useState('')
  const [streamingTools, setStreamingTools] = useState<ToolCallRecord[]>([])
  const [streamingToolOffsets, setStreamingToolOffsets] = useState<number[]>([])
  const [generationRate, setGenerationRate] = useState<number | null>(null)
  const [generationTokens, setGenerationTokens] = useState<{
    prompt: number | null
    completion: number
  } | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [forkHints, setForkHints] = useState<RuntimeForkHint[]>([])
  const [modelLoadStatus, setModelLoadStatus] = useState<string | null>(null)
  const [pendingBuild, setPendingBuild] = useState<{
    engine: 'llama.cpp' | 'mlx-lm' | 'mlx-vlm'
    repository: string
  } | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [checkingForUpdates, setCheckingForUpdates] = useState(false)
  const [localModels, setLocalModels] = useState<LocalModel[]>(() => readCachedModels())
  const [modelsLoading, setModelsLoading] = useState(() => readCachedModels().length === 0)
  const [modelsLoadFailed, setModelsLoadFailed] = useState(false)
  const [selectedModel, setSelectedModel] = useState('')
  const [modelPrepareState, setModelPrepareState] = useState<
    'idle' | 'loading' | 'ready' | 'error'
  >('idle')
  const [modelResidency, setModelResidency] = useState<ModelResidency | null>(null)
  const [modelUnloading, setModelUnloading] = useState(false)
  const [modelBindings, setModelBindings] = useState<Record<string, string>>({})
  const [prefetchedRuntimes, setPrefetchedRuntimes] = useState<RuntimeEntry[] | null>(() => {
    const cached = readCachedRuntimes()
    return cached.length > 0 ? cached : null
  })
  const [enabledTools, setEnabledTools] = useState<string[]>(() => readEnabledTools())
  const [availableTools, setAvailableTools] = useState<BundledTool[]>([])
  const [toolsMenuOpen, setToolsMenuOpen] = useState(false)
  const toolsEnabled = enabledTools.length > 0
  const [daemonStatus, setDaemonStatus] = useState<'checking' | 'healthy' | 'offline'>('checking')
  const [daemonVersion, setDaemonVersion] = useState('')
  const [hardware, setHardware] = useState<HardwareInfo | null>(null)
  const [runtime, setRuntime] = useState<RuntimeSettings | null>(null)
  const [savingInference, setSavingInference] = useState(false)

  const [modelMenuOpen, setModelMenuOpen] = useState(false)
  const [inferenceMenuOpen, setInferenceMenuOpen] = useState(false)
  // Advanced per-model configuration, held here because three surfaces open it:
  // the model picker, the library, and the inference menu.
  const [modelProfiles, setModelProfiles] = useState<Record<string, ModelProfile>>({})
  const [configuringModel, setConfiguringModel] = useState<string | null>(null)
  const [manageOpen, setManageOpen] = useState(false)
  const [manageSection, setManageSection] = useState<ManageSection>('library')
  const [runSnapshots, setRunSnapshots] = useState<RunSnapshot[]>([])
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null)
  const [showWelcome, setShowWelcome] = useState<boolean | null>(null)
  const [recommendationSwaps, setRecommendationSwaps] = useState<PendingSwap[]>([])
  const [appMode, setAppMode] = useState<AppMode>('chat')
  const [workspaceModes, setWorkspaceModes] =
    useState<WorkspaceModesPreference>(DEFAULT_WORKSPACE_MODES)
  const [realtimeVoiceAvailable, setRealtimeVoiceAvailable] = useState(false)
  // Generate, Voice, and Computer pick from their own model families; the top
  // bar shows whichever belongs to the mode on screen.
  const [voiceModel, setVoiceModel] = useState('')
  const [generateModel, setGenerateModel] = useState('')
  const [computerModel, setComputerModel] = useState('')
  const [generateModality, setGenerateModality] = useState<'image' | 'video'>('image')
  const [generateHistory, setGenerateHistory] = useState<GenerateHistoryEntry[]>(() =>
    readGenerateHistory()
  )
  const [activeGenerateHistoryId, setActiveGenerateHistoryId] = useState<string | null>(null)
  const [persona, setPersona] = useState('You are a helpful assistant.')
  const personaEdited = useRef(false)
  // Agent / Computer modes have no composer of their own; they publish these
  // so the one at the bottom of the window can drive them.
  const [agentComposer, setAgentComposer] = useState<AgentComposerControls | null>(null)
  const [computerComposer, setComputerComposer] = useState<AgentComposerControls | null>(null)
  const [agentSidebar, setAgentSidebar] = useState<AgentSidebarControls | null>(null)

  useEffect(() => {
    if (showWelcome !== false) return
    void fetchRecommendations()
      .then((result) => setRecommendationSwaps(result.swaps))
      .catch(() => {
        // Recommendations are optional and may require the Hub; startup is not.
      })
  }, [showWelcome])

  useEffect(() => {
    void fetchWorkspacePreference()
      .then((result) => setWorkspaceModes(result.modes))
      .catch(() => {
        // Defaults keep the familiar mode strip until preferences load.
      })
  }, [])

  useEffect(() => {
    const order: AppMode[] = ['chat', 'agent', 'generate', 'voice', 'computer']
    if (workspaceModes[appMode]) return
    const fallback = order.find((mode) => workspaceModes[mode])
    if (fallback) setAppMode(fallback)
  }, [workspaceModes, appMode])

  const abortRef = useRef<AbortController | undefined>(undefined)
  const prepareAbortRef = useRef<AbortController | undefined>(undefined)
  const conversationRefreshRef = useRef(0)
  // Message requests can outlive a cancelled generation or a newly created
  // chat. Keep a generation counter separate from the conversation list so a
  // late response for the old chat cannot repaint the new one.
  const messageRefreshRef = useRef(0)
  const fileInput = useRef<HTMLInputElement>(null)
  const importInput = useRef<HTMLInputElement>(null)
  const scrollAnchor = useRef<HTMLDivElement>(null)

  const chain = useMemo(() => messageChain(messages, tipId), [messages, tipId])
  const chatDisplayItems = useMemo(() => buildChatDisplayItems(chain), [chain])
  const canChat = Boolean(selectedModel) && modelPrepareState === 'ready' && !modelUnloading
  const selectedCapabilities = localModels.find((model) => model.id === selectedModel)?.capabilities
  const chatModels = useMemo(() => localModels.filter((model) => isChatModel(model)), [localModels])
  const voiceModels = useMemo(() => localModels.filter((model) => isVoiceModel(model)), [localModels])
  const computerModels = useMemo(
    () => localModels.filter((model) => isComputerUseModel(model)),
    [localModels]
  )
  const generateModels = useMemo(
    () =>
      localModels.filter((model) =>
        generateModality === 'image' ? isImageGenModel(model) : isVideoGenModel(model)
      ),
    [localModels, generateModality]
  )
  const [pipelineFeatures, setPipelineFeatures] = useState<PipelineFeatures>({
    asr: false,
    video_preprocess: false
  })
  const runtimeWarning = useMemo(
    () => runtimeNoticeForModel(selectedModel, localModels, prefetchedRuntimes, modelBindings),
    [selectedModel, localModels, prefetchedRuntimes, modelBindings]
  )
  const canAttachImage = Boolean(selectedCapabilities?.input_modalities.includes('image'))
  const selectedNativeAudio = selectedCapabilities?.audio_input === 'native'
  const canAttachAudio = pipelineFeatures.asr || selectedNativeAudio
  const canAttachVideo =
    pipelineFeatures.video_preprocess && canAttachImage
  // Documents are converted to text before inference, so they do not need a
  // vision-capable model. Keeping this separate from media capabilities also
  // makes the picker usable with ordinary text-only chat models.
  const canAttachDocuments = true
  const canAttach = canAttachDocuments || canAttachImage || canAttachAudio || canAttachVideo
  const canUseTools = selectedCapabilities?.tools !== false
  const audioBadgeTitle = selectedNativeAudio
    ? 'Native audio: this chat model can consume audio directly; falls back to batch ASR if the engine rejects input_audio'
    : pipelineFeatures.asr
      ? 'Batch ASR: attachments are transcribed with whisper.cpp, then sent as text'
      : pipelineFeatures.streaming_asr
        ? 'Streaming ASR is available via /v1/audio/transcriptions?stream=true; chat attachments still use batch ASR or native audio'
        : 'No audio path yet — build whisper.cpp + download a Whisper model, install streaming ASR, or select a native-audio chat model.'

  const updateEnabledTools = useCallback((next: string[]): void => {
    setEnabledTools(next)
    writeEnabledTools(next)
  }, [])

  /**
   * Re-read what the host can do. Building a runtime or downloading a model
   * changes these while the window is open, and a stale answer decides which
   * transcription engine a spoken turn is sent to — or hides voice mode behind
   * a requirement that has already been met.
   */
  const refreshCapabilities = useCallback(async (): Promise<void> => {
    try {
      const payload = await fetchCapabilities()
      const audio = payload.features.audio_interfaces
      setPipelineFeatures({
        asr: Boolean(payload.features.asr ?? audio?.batch_asr?.available),
        video_preprocess: Boolean(payload.features.video_preprocess),
        whisper_cpp_engine: Boolean(payload.features.whisper_cpp_engine),
        native_model_audio: Boolean(audio?.native_model_audio?.available),
        streaming_asr: Boolean(audio?.streaming_asr?.available),
        realtime_voice: Boolean(audio?.realtime_voice?.available)
      })
      setRealtimeVoiceAvailable(Boolean(audio?.realtime_voice?.available))
    } catch {
      // Keep the last known answer; the picker still shows what it had.
    }
  }, [])

  async function refreshTools(): Promise<void> {
    try {
      const tools = await listTools()
      setAvailableTools(tools)
      // Drop selections for tools that no longer exist (e.g. removed MCP server).
      setEnabledTools((current) => {
        const names = new Set(tools.map((tool) => tool.name))
        const filtered = current.filter((name) => names.has(name))
        if (filtered.length !== current.length) writeEnabledTools(filtered)
        return filtered
      })
    } catch {
      // Non-fatal: the tools popover simply shows what it last loaded.
    }
  }

  async function refreshLocalModels(): Promise<void> {
    const models = await listModels()
    setLocalModels(models)
    writeCachedModels(models)
    setSelectedModel((current) => {
      if (current && models.some((model) => model.id === current)) return current
      return ''
    })
  }

  const selectModel = useCallback((modelId: string, mode?: ModelLoadMode): void => {
    if (modelId.startsWith('whisper:') || modelId.startsWith('streaming-asr:')) {
      setError('ASR models are used for transcription, not chat. Pick a chat model instead.')
      return
    }
    prepareAbortRef.current?.abort()
    setSelectedModel(modelId)
    setModelResidency(null)
    setForkHints([])
    if (!modelId) {
      setModelPrepareState('idle')
      setModelLoadStatus(null)
      setError(null)
      return
    }
    setModelPrepareState('loading')
    setModelLoadStatus('Preparing model…')
    setError(null)
    const controller = new AbortController()
    prepareAbortRef.current = controller
    void prepareModel(modelId, {
      signal: controller.signal,
      mode: mode ?? (appMode === 'agent' ? 'agent' : 'chat'),
      onLoad: (event) => {
        if (!controller.signal.aborted) setModelLoadStatus(event.message)
      }
    })
      .then((residency) => {
        if (controller.signal.aborted) return
        setModelResidency(residency)
        setModelPrepareState('ready')
        setModelLoadStatus(null)
        void prefetchRuntimes()
        void fetchModelBindings().then(setModelBindings).catch(() => {})
        void refreshCapabilities()
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted || (cause as Error).name === 'AbortError') return
        setModelPrepareState('error')
        setModelLoadStatus(null)
        if (cause instanceof GenerationFailure) {
          setError(cause.message)
          setForkHints(cause.forkHints)
        } else {
          setError(cause instanceof Error ? cause.message : String(cause))
        }
      })
  }, [appMode])

  const switchAppMode = useCallback(
    (next: AppMode): void => {
      if (next === appMode) return
      setAppMode(next)
      if ((next === 'chat' || next === 'agent') && selectedModel) {
        selectModel(selectedModel, next)
      }
      // Chat stays mounted while other modes are active; re-read the open
      // conversation when returning so the next completion uses the same
      // history shown on screen (mirrors agent session rehydrate).
      if (next === 'chat' && conversationId) {
        void refreshMessages(conversationId).catch(() => undefined)
      }
    },
    [appMode, selectedModel, selectModel, conversationId]
  )

  const unloadSelectedModel = useCallback(async (): Promise<void> => {
    if (!selectedModel || modelUnloading) return
    prepareAbortRef.current?.abort()
    setModelUnloading(true)
    setModelPrepareState('loading')
    setModelLoadStatus('Unloading model…')
    setError(null)
    try {
      await unloadModel()
      setSelectedModel('')
      setModelResidency(null)
      setModelPrepareState('idle')
      setForkHints([])
      void prefetchRuntimes()
    } catch (cause) {
      setModelPrepareState('ready')
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setModelUnloading(false)
      setModelLoadStatus(null)
    }
  }, [selectedModel, modelUnloading])

  /** Model list, selection, and setter for whichever mode is on screen. */
  const modeModel = useMemo(() => {
    if (appMode === 'voice') {
      return {
        models: voiceModels,
        selected: voiceModel,
        select: setVoiceModel,
        emptyTitle: 'No voice model',
        emptySubtitle: 'Download a PersonaPlex model'
      }
    }
    if (appMode === 'generate') {
      return {
        models: generateModels,
        selected: generateModel,
        select: setGenerateModel,
        emptyTitle: `No ${generateModality} model`,
        emptySubtitle: 'Download one from the library'
      }
    }
    if (appMode === 'computer') {
      return {
        models: computerModels,
        selected: computerModel,
        select: setComputerModel,
        emptyTitle: 'No computer-use model',
        emptySubtitle: 'Install a Fara1.5 or similar vision agent model'
      }
    }
    return {
      models: chatModels,
      selected: selectedModel,
      select: selectModel,
      emptyTitle: 'Select a model',
      emptySubtitle: 'No model loaded yet'
    }
  }, [
    appMode,
    voiceModels,
    voiceModel,
    generateModels,
    generateModel,
    generateModality,
    computerModels,
    computerModel,
    chatModels,
    selectedModel,
    selectModel
  ])

  const configuredModelEntry = useMemo(
    () => localModels.find((model) => model.id === configuringModel) ?? null,
    [localModels, configuringModel]
  )
  const profileCounts = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(modelProfiles).map(([modelId, profile]) => [
          modelId,
          profileCount(profile)
        ])
      ),
    [modelProfiles]
  )

  const selectedMeta = useMemo(() => {
    if (modelsLoading && localModels.length === 0) {
      return { title: 'Loading models…', subtitle: 'Scanning local library' }
    }
    if (!modeModel.selected) {
      return { title: modeModel.emptyTitle, subtitle: modeModel.emptySubtitle }
    }
    return modelDisplayName(
      modeModel.selected,
      localModels.find((model) => model.id === modeModel.selected)
    )
  }, [modeModel, localModels, modelsLoading])

  // Engine settings arrive after the first render; adopt the saved persona
  // unless the field has already been typed in.
  useEffect(() => {
    const saved = runtime?.default_voice_persona
    if (saved && !personaEdited.current) setPersona(saved)
  }, [runtime?.default_voice_persona])

  /**
   * Ordinary chat answers for the shared conversation, used when no agent
   * session is bound to it. The same completion path the composer uses, so a
   * spoken question is answered exactly like a typed one.
   */
  const chainRef = useRef<Message[]>([])
  chainRef.current = chain
  const voiceResponder = useMemo(() => {
    const controllers = new Map<string, AbortController>()
    return {
      async respond({
        correlationId,
        text,
        onPartial
      }: {
        correlationId: string
        text: string
        onPartial?: (delta: string) => void
      }): Promise<{ text: string }> {
        if (!selectedModel) {
          throw new Error('Select a chat model before asking a question by voice.')
        }
        const controller = new AbortController()
        controllers.set(correlationId, controller)
        // The coordinator has already stored the user message; React has not
        // re-rendered yet, so the request carries it explicitly.
        const asked: Message = {
          id: `pending-${correlationId}`,
          conversation_id: conversationId ?? '',
          parent_id: chainRef.current.at(-1)?.id ?? null,
          role: 'user',
          content: text,
          model: null,
          created_at: new Date().toISOString()
        }
        try {
          const result = await streamCompletion(
            [...chainRef.current, asked],
            selectedModel,
            controller.signal,
            (token) => onPartial?.(token),
            {
              builtinTools: toolsEnabled,
              builtinToolNames: toolsEnabled ? enabledTools : undefined,
              toolChoice: toolsEnabled ? 'auto' : undefined
            }
          )
          return { text: result.responseText }
        } finally {
          controllers.delete(correlationId)
        }
      },
      cancel(correlationId: string): void {
        controllers.get(correlationId)?.abort()
      }
    }
  }, [conversationId, selectedModel, toolsEnabled, enabledTools])

  const session = useSessionCoordinator({
    conversationId,
    messages,
    summary: conversations.find((entry) => entry.id === conversationId)?.summary ?? null,
    chatModelId: selectedModel,
    voiceModelId: voiceModel,
    asrAvailable: {
      batch: Boolean(pipelineFeatures.asr),
      streaming: Boolean(pipelineFeatures.streaming_asr)
    },
    persona,
    responder: voiceResponder,
    onMessage: (message) => {
      setMessages((current) =>
        current.some((entry) => entry.id === message.id)
          ? current.map((entry) => (entry.id === message.id ? message : entry))
          : [...current, message]
      )
      setTipId(message.id)
    },
    onStatus: setModelLoadStatus,
    parentId: () => chainRef.current.at(-1)?.id ?? null
  })
  const agentMode = appMode === 'agent'
  const computerMode = appMode === 'computer'
  const shellComposer = agentMode ? agentComposer : computerMode ? computerComposer : null
  const shellComposerMode = agentMode || computerMode
  const voiceLive = session.snapshot.voiceStatus === 'live'
  const audioSupported = useMemo(() => voiceStreamSupported(), [])
  /** Whichever answer is streaming: the composer's own, or a coordinated turn. */
  const liveText = streamingText || session.snapshot.streamingText

  // Seed the Voice and Generate selections from saved defaults, falling back to
  // the first installed model of the right family.
  useEffect(() => {
    setVoiceModel((current) => {
      if (current && voiceModels.some((model) => model.id === current)) return current
      const preferred = runtime?.default_voice_model
      if (preferred && voiceModels.some((model) => model.id === preferred)) return preferred
      return voiceModels[0]?.id ?? ''
    })
  }, [voiceModels, runtime?.default_voice_model])

  useEffect(() => {
    setGenerateModel((current) => {
      if (current && generateModels.some((model) => model.id === current)) return current
      const preferred =
        generateModality === 'image'
          ? runtime?.default_image_gen_model
          : runtime?.default_video_gen_model
      if (preferred && generateModels.some((model) => model.id === preferred)) return preferred
      return generateModels[0]?.id ?? ''
    })
  }, [
    generateModels,
    generateModality,
    runtime?.default_image_gen_model,
    runtime?.default_video_gen_model
  ])

  useEffect(() => {
    setComputerModel((current) => {
      if (current && computerModels.some((model) => model.id === current)) return current
      return computerModels[0]?.id ?? ''
    })
  }, [computerModels])

  async function updateModelBinding(modelId: string, runtimeId: string | null): Promise<void> {
    setError(null)
    try {
      const bindings = await setModelBinding(modelId, runtimeId)
      setModelBindings(bindings)
      await prefetchRuntimes()
      if (modelId === selectedModel) {
        selectModel(modelId)
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  async function loadLocalModels(): Promise<void> {
    const hadCache = localModels.length > 0
    if (!hadCache) {
      setModelsLoading(true)
    }
    setModelsLoadFailed(false)
    try {
      await refreshLocalModels()
    } catch (cause) {
      if (!hadCache) setModelsLoadFailed(true)
      throw cause
    } finally {
      setModelsLoading(false)
    }
  }

  async function prefetchRuntimes(): Promise<void> {
    try {
      const response = await listRuntimes()
      setPrefetchedRuntimes(response.data)
      writeCachedRuntimes(response.data)
    } catch {
      // Best-effort prefetch — Manage panel falls back to its own fetch.
    }
  }

  async function refreshConversations(query = conversationSearch): Promise<void> {
    const refreshId = ++conversationRefreshRef.current
    const data = await listConversations(query)
    // Search requests can resolve out of order while the user types. Only the
    // newest query is allowed to replace the visible results or selection.
    if (refreshId !== conversationRefreshRef.current) return
    setConversations(data)
    // Only an unfiltered list is worth caching for the next cold start.
    if (!query.trim()) writeCachedConversations(data)
    if (!conversationId && data[0]) setConversationId(data[0].id)
  }

  async function exportCurrentConversation(): Promise<void> {
    if (!conversationId) return
    setError(null)
    try {
      const bundle = await exportConversation(conversationId)
      const blob = new Blob([JSON.stringify(bundle, null, 2)], {
        type: 'application/json'
      })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `brazier-${bundle.conversation.title.replace(/\s+/g, '-').slice(0, 48)}.json`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  async function importConversationFromFile(file: File): Promise<void> {
    setError(null)
    try {
      const bundle = JSON.parse(await file.text()) as ConversationExport
      const conversation = await importConversation(bundle)
      await refreshConversations('')
      setConversationSearch('')
      setConversationId(conversation.id)
      await refreshMessages(conversation.id)
      await refreshConversations('')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  async function refreshMessages(id: string, preferredTip?: string): Promise<void> {
    const refreshId = ++messageRefreshRef.current
    const data = await listMessages(id)
    if (refreshId !== messageRefreshRef.current || id !== conversationId) return
    setMessages(data)
    setTipId(preferredTip ?? data.at(-1)?.id ?? null)
  }

  async function refreshRuntime(): Promise<void> {
    const status = await engineStatus()
    setRuntime(status.settings)
    setHardware(status.hardware)
  }

  useEffect(() => {
    let cancelled = false
    void Promise.all([window.brazier.getFlags(), hasCompletedWelcome()])
      .then(([flags, completed]) => {
        if (cancelled) return
        setShowWelcome(Boolean(flags.forceWelcome) || !completed)
      })
      .catch(async () => {
        const completed = await hasCompletedWelcome()
        if (!cancelled) setShowWelcome(!completed)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    void refreshConversations().catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
    void loadLocalModels().catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
    void prefetchRuntimes()
    void refreshTools()
    void fetchModelBindings().then(setModelBindings).catch(() => {})
    // Advanced per-model settings, so the picker can say which models carry any
    // before one of them is opened.
    void fetchModelSettings()
      .then((response) => setModelProfiles(response.models))
      .catch(() => {})
    void refreshRuntime().catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
    void refreshCapabilities()
  }, [])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refreshConversations(conversationSearch).catch((cause: unknown) =>
        setError(cause instanceof Error ? cause.message : String(cause))
      )
    }, 250)
    return () => window.clearTimeout(timer)
  }, [conversationSearch])

  useEffect(() => {
    let active = true
    const check = (): void => {
      void health()
        .then((result) => {
          if (!active) return
          setDaemonStatus('healthy')
          setDaemonVersion(result.version)
        })
        .catch(() => active && setDaemonStatus('offline'))
    }
    check()
    const timer = window.setInterval(check, 10_000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    // A selected chat changed; any fetch started for the previous one is now
    // stale, even if it completes after this effect has begun loading the new
    // conversation.
    messageRefreshRef.current += 1
    if (!conversationId) {
      setMessages([])
      setTipId(null)
      setRunSnapshots([])
      setExpandedRunId(null)
      return
    }
    setError(null)
    refreshMessages(conversationId).catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
    void listRunSnapshots(conversationId)
      .then(setRunSnapshots)
      .catch(() => setRunSnapshots([]))
  }, [conversationId])

  useEffect(() => {
    scrollAnchor.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chain.length, liveText])

  // Entering Voice mode is exactly when a runtime or model installed since
  // launch would otherwise still look missing.
  useEffect(() => {
    if (appMode !== 'voice') return
    void refreshCapabilities()
  }, [appMode, refreshCapabilities])

  // Voice writes into a conversation, so there has to be one before the user can
  // start speaking.
  useEffect(() => {
    if (appMode !== 'voice' || conversationId) return
    void newConversation().catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
  }, [appMode, conversationId])

  async function newConversation(): Promise<void> {
    const conversation = await createConversation()
    await refreshConversations()
    // Invalidate in-flight loads for the previous chat before clearing its
    // display. This matters most after Stop, when a final fetch may still be
    // resolving in the background.
    messageRefreshRef.current += 1
    setConversationId(conversation.id)
    setMessages([])
    setTipId(null)
    setDraft('')
  }

  function maybeGenerateConversationTitle(
    id: string,
    prompt: string,
    answer: string,
    outputTokensPerSecond: number | null,
    mode: ChatTitleMode
  ): void {
    if (
      mode === 'never' ||
      (mode === 'over-20-tokens' && outputTokensPerSecond != null && outputTokensPerSecond <= 20)
    ) {
      return
    }
    const titleMessages: Message[] = [
      {
        id: 'title-instruction',
        conversation_id: id,
        parent_id: null,
        role: 'system',
        content:
          'Write a short, specific chat title for this exchange. Return only the title, with no quotation marks or punctuation at the end.',
        model: null,
        created_at: new Date().toISOString()
      },
      {
        id: 'title-user',
        conversation_id: id,
        parent_id: 'title-instruction',
        role: 'user',
        content: `User: ${prompt}\n\nAssistant: ${answer}`,
        model: null,
        created_at: new Date().toISOString()
      }
    ]
    void streamCompletion(titleMessages, selectedModel, new AbortController().signal, () => {}, {
      toolChoice: 'none'
    })
      .then(({ responseText }) => {
        const title = titleFromCompletion(responseText)
        if (!title) return
        return updateConversation(id, { title }).then(() => refreshConversations(''))
      })
      .catch(() => {
        // Naming is a convenience. A completed chat must not report an error
        // because its optional title request failed.
      })
  }

  function changeChatTitleMode(mode: ChatTitleMode): void {
    setChatTitleMode(mode)
    try {
      localStorage.setItem(CHAT_TITLE_MODE_KEY, mode)
    } catch {
      // Best-effort persistence.
    }
  }

  async function renameConversation(conversation: Conversation): Promise<void> {
    setConversationMenuId(null)
    const title = window.prompt('Rename conversation', conversation.title)?.trim()
    if (!title || title === conversation.title) return
    try {
      await updateConversation(conversation.id, { title })
      await refreshConversations('')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  async function removeConversation(conversation: Conversation): Promise<void> {
    setConversationMenuId(null)
    if (!window.confirm(`Delete “${conversation.title}”? This cannot be undone.`)) return
    try {
      await deleteConversation(conversation.id)
      if (conversation.id === conversationId) {
        setConversationId(null)
        setMessages([])
        setTipId(null)
      }
      const remaining = await listConversations('')
      setConversations(remaining)
      writeCachedConversations(remaining)
      setConversationSearch('')
      if (conversation.id === conversationId) setConversationId(remaining[0]?.id ?? null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  async function applyInferenceSettings(next: RuntimeSettings): Promise<void> {
    setSavingInference(true)
    setError(null)
    try {
      const model = localModels.find((entry) => entry.id === selectedModel)
      const caps = model?.capabilities
      let adjusted = { ...next }
      const maxContext = caps?.max_context_length
      if (maxContext && adjusted.context_size > maxContext) {
        adjusted = { ...adjusted, context_size: maxContext }
      }
      const reasoningModes =
        caps?.reasoning_modes ?? (caps?.reasoning ? ['off', 'on'] : [])
      if (reasoningModes.length === 0) {
        adjusted = { ...adjusted, enable_reasoning: false, reasoning_budget_tokens: null }
      } else if (!reasoningModes.includes('budget')) {
        adjusted = { ...adjusted, reasoning_budget_tokens: null }
      }
      const saved = await saveRuntimeSettings(adjusted)
      setRuntime(saved)
      if (selectedModel && modelPrepareState === 'ready') {
        selectModel(selectedModel)
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSavingInference(false)
    }
  }

  function openManage(section: ManageSection): void {
    setManageSection(section)
    setManageOpen(true)
  }

  async function checkForUpdates(): Promise<void> {
    setCheckingForUpdates(true)
    try {
      await window.brazier.checkForUpdates()
    } finally {
      setCheckingForUpdates(false)
    }
  }

  function startForkBuild(hint: RuntimeForkHint): void {
    setPendingBuild({
      engine: hint.engine as 'llama.cpp' | 'mlx-lm' | 'mlx-vlm',
      repository: hint.repository
    })
    setManageSection('runtimes')
    setManageOpen(true)
  }

  async function selectFiles(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    try {
      const files = Array.from(event.target.files ?? [])
      const accepted = files.filter((file) => {
        if (file.type.startsWith('image/')) return canAttachImage
        if (file.type.startsWith('audio/')) return canAttachAudio
        if (file.type.startsWith('video/')) return canAttachVideo
        return isDocumentFile(file)
      })
      if (accepted.length === 0 && files.length > 0) {
        const reasons: string[] = []
        if (!canAttachImage) reasons.push('vision model for images')
        if (!canAttachAudio)
          reasons.push(
            selectedNativeAudio
              ? 'native-audio model path'
              : 'batch ASR (whisper.cpp + Whisper model) or a native-audio chat model'
          )
        if (!canAttachVideo) reasons.push('ffmpeg + vision model for video')
        setError(`Cannot attach that file yet. Need: ${reasons.join('; ')}.`)
        event.target.value = ''
        return
      }
      const loaded = await Promise.all(accepted.map(fileToAttachment))
      setAttachments((current) => [...current, ...loaded])
      event.target.value = ''
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  async function submit(event: FormEvent): Promise<void> {
    event.preventDefault()
    const text = draft.trim()
    // Agent mode: the same box, pointed at the agent. Its own transcript and
    // approval cards render above; only the input is shared.
    if (appMode === 'agent' || appMode === 'computer') {
      const composer = appMode === 'agent' ? agentComposer : computerComposer
      if (!text || !composer || composer.running) return
      setDraft('')
      setError(null)
      await composer.send(text)
      return
    }
    if ((!text && attachments.length === 0) || busy) return
    // With voice live, typing goes through the coordinator so both surfaces
    // share one conversation and one agent session. Attachments keep the
    // ordinary path, which knows how to hydrate them.
    if (voiceLive && text && attachments.length === 0) {
      setDraft('')
      setError(null)
      await session.submitText(text)
      return
    }
    if (!selectedModel) {
      setError('Select or download a local model first.')
      setModelMenuOpen(true)
      return
    }
    if (runtimeWarning) {
      setError(runtimeWarning)
      openManage('runtimes')
      return
    }
    setBusy(true)
    setError(null)
    setForkHints([])
    setModelLoadStatus('Preparing model…')
    setStreamingText('')
    setStreamingReasoning('')
    setStreamingTools([])
    setStreamingToolOffsets([])
    setGenerationRate(null)
    setGenerationTokens(null)
    const controller = new AbortController()
    abortRef.current = controller
    const shouldGenerateTitle =
      !conversationId ||
      (chain.length === 0 &&
        conversations.find((conversation) => conversation.id === conversationId)?.title ===
          'New conversation')
    let latestGenerationRate: number | null = null

    try {
      let activeConversationId = conversationId
      if (!activeConversationId) {
        const created = await createConversation()
        activeConversationId = created.id
        setConversationId(created.id)
      }
      const parts: ContentPart[] = [
        ...(text ? [{ type: 'text' as const, text }] : []),
        ...attachments.map(attachmentPart)
      ]
      const content = attachments.length === 0 ? text : parts
      const userMessage = await createMessage(activeConversationId, {
        parent_id: chain.at(-1)?.id ?? null,
        role: 'user',
        content
      })
      let requestMessages = [...chain, userMessage]
      setMessages((current) => [...current, userMessage])
      setTipId(userMessage.id)
      setDraft('')
      setAttachments([])

      let parentId = userMessage.id
      let responseText = ''
      let responseReasoning = ''
      const toolRecords: ToolCallRecord[] = []
      const toolOffsets: number[] = []
      const maxClientRounds = 4
      for (let round = 0; round < maxClientRounds; round += 1) {
        setStreamingReasoning('')
        let roundReasoning = ''
        const result = await streamCompletion(
          requestMessages,
          selectedModel,
          controller.signal,
          (token) => {
            setModelLoadStatus(null)
            responseText += token
            setStreamingText(responseText)
          },
          {
            builtinTools: toolsEnabled,
            builtinToolNames: toolsEnabled ? enabledTools : undefined,
            toolChoice: toolsEnabled ? 'auto' : undefined,
            onLoad: (event) => setModelLoadStatus(event.message),
            onPrefill: (event) => setModelLoadStatus(prefillProgressLabel(event)),
            onReasoning: (token) => {
              setModelLoadStatus(null)
              roundReasoning += token
              setStreamingReasoning(roundReasoning)
            },
            onToolCall: (record) => {
              toolRecords.push(record)
              toolOffsets.push(responseText.length)
              setStreamingTools([...toolRecords])
              setStreamingToolOffsets([...toolOffsets])
            }
          }
        )
        responseText = result.responseText
        responseReasoning = result.reasoningText
        if (result.generationStats) {
          latestGenerationRate =
            (result.generationStats.completion_tokens * 1000) /
            Math.max(1, result.generationStats.decode_duration_ms)
          setGenerationRate(latestGenerationRate)
          setGenerationTokens({
            prompt: result.generationStats.prompt_tokens ?? null,
            completion: result.generationStats.completion_tokens
          })
        }
        for (const entry of result.transcript) {
          const role = entry.role as Role
          const entryReasoning =
            typeof entry.reasoning_content === 'string' ? entry.reasoning_content.trim() : ''
          const created = await createMessage(activeConversationId, {
            parent_id: parentId,
            role,
            content: entry.content ?? '',
            tool_calls: entry.tool_calls ?? null,
            tool_call_id: entry.tool_call_id ?? null,
            model: role === 'assistant' ? selectedModel : undefined,
            ...(role === 'assistant' && entryReasoning
              ? { metadata: { reasoning_content: entryReasoning } }
              : {})
          })
          requestMessages = [...requestMessages, created]
          parentId = created.id
        }
        if (result.clientToolCalls.length === 0) {
          break
        }
        throw new GenerationFailure(
          `The model requested client-side tools that Brazier cannot execute: ${result.clientToolCalls.map((call) => call.name).join(', ')}`
        )
      }
      const assistant = await createMessage(activeConversationId, {
        parent_id: parentId,
        role: 'assistant',
        content: responseText,
        model: selectedModel,
        ...(responseReasoning.trim()
          ? { metadata: { reasoning_content: responseReasoning } }
          : {})
      })
      if (shouldGenerateTitle && responseText.trim()) {
        maybeGenerateConversationTitle(
          activeConversationId,
          text,
          responseText,
          latestGenerationRate,
          chatTitleMode
        )
      }
      let finalTipId = assistant.id
      const generatedMedia = [
        ...new Map(
          toolRecords
            .flatMap((record) => record.media ?? [])
            .map((media) => [`${media.sha256}:${media.mime_type}`, media])
        ).values()
      ]
      if (generatedMedia.length > 0) {
        const generated = await createMessage(activeConversationId, {
          parent_id: assistant.id,
          role: 'assistant',
          content: generatedMedia.map((media, index) => ({
            type: 'brazier_blob' as const,
            brazier_blob: {
              sha256: media.sha256,
              mime_type: media.mime_type,
              name:
                media.name ??
                (media.mime_type === 'application/pdf'
                  ? `document-${index + 1}.pdf`
                  : media.mime_type.startsWith('video/')
                    ? `generated-video-${index + 1}.mp4`
                    : `generated-image-${index + 1}.png`)
            }
          })),
          model: selectedModel,
          source: 'assistant_chat',
          metadata: { generated_media_display: true }
        })
        finalTipId = generated.id
      }
      if (runtime) {
        await recordRun(activeConversationId, {
          parent_message_id: userMessage.id,
          assistant_message_id: assistant.id,
          model: selectedModel,
          settings: runtime,
          tool_calls: toolRecords,
          response_text: responseText
        })
      }
      setStreamingText('')
      setStreamingReasoning('')
      setStreamingTools([])
      setStreamingToolOffsets([])
      setModelLoadStatus(null)
      await refreshMessages(activeConversationId, finalTipId)
      await refreshConversations()
      void listRunSnapshots(activeConversationId).then(setRunSnapshots).catch(() => {})
    } catch (cause) {
      if ((cause as Error).name !== 'AbortError') {
        if (cause instanceof GenerationFailure) {
          setError(cause.message)
          setForkHints(cause.forkHints)
        } else {
          setError(cause instanceof Error ? cause.message : String(cause))
          setForkHints([])
        }
      }
    } finally {
      setBusy(false)
      setModelLoadStatus(null)
      abortRef.current = undefined
    }
  }

  if (showWelcome === null) {
    return <main className="app-shell first-run-boot" aria-busy="true" />
  }

  if (showWelcome) {
    return (
      <main className="app-shell">
        <WelcomeScreen
          onContinue={() => {
            void markWelcomeCompleted()
            setShowWelcome(false)
          }}
          onOpenRuntimes={() => {
            // Opening Runtimes leaves the walkthrough just like choosing
            // Continue: do not show it again after an update.
            void markWelcomeCompleted()
            setManageSection('runtimes')
            setManageOpen(true)
            setShowWelcome(false)
          }}
          onModelsChanged={() => void refreshLocalModels().catch(() => {})}
        />
      </main>
    )
  }

  return (
    <main className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        {/* Narrow windows only (hidden by CSS otherwise): the history panel
            docks to the bottom and this handle rides its top edge, keeping
            the toggle next to what it opens. The topbar hamburger takes over
            in the wide layout. */}
        <button
          type="button"
          className="sidebar-handle"
          aria-expanded={sidebarOpen}
          onClick={() => setSidebarOpen((open) => !open)}
        >
          <Menu size={16} />
          <span>
            {appMode === 'agent'
              ? 'Tasks'
              : appMode === 'generate'
                ? 'History'
                : appMode === 'computer'
                  ? 'Computer'
                  : 'Conversations'}
          </span>
          <ChevronUp className="handle-chevron" size={15} />
        </button>
        {appMode === 'agent' ? (
          <AgentSessionSidebar controls={agentSidebar} />
        ) : appMode === 'generate' ? (
          <GenerateHistorySidebar
            entries={generateHistory}
            activeId={activeGenerateHistoryId}
            onSelect={setActiveGenerateHistoryId}
          />
        ) : (
          <>
            <button className="new-chat" onClick={() => void newConversation()}>
              <MessageSquarePlus size={17} />
              New conversation
            </button>
            <label className="conversation-search">
              <Search size={14} />
              <input
                aria-label="Search conversations"
                value={conversationSearch}
                onChange={(event) => setConversationSearch(event.target.value)}
                placeholder="Search conversations…"
              />
            </label>
            <div className="sidebar-actions">
              <button
                className="chip-button subtle"
                disabled={!conversationId}
                title="Export this conversation as JSON"
                onClick={() => void exportCurrentConversation()}
              >
                <Download size={13} />
                Export
              </button>
              <button
                className="chip-button subtle"
                title="Import a conversation JSON export"
                onClick={() => importInput.current?.click()}
              >
                <Upload size={13} />
                Import
              </button>
              <input
                ref={importInput}
                type="file"
                accept="application/json,.json"
                hidden
                onChange={(event) => {
                  const file = event.target.files?.[0]
                  if (file) void importConversationFromFile(file)
                  event.target.value = ''
                }}
              />
            </div>
            <label className="chat-title-mode">
              <span>Generated names</span>
              <select
                aria-label="Generated chat names"
                value={chatTitleMode}
                onChange={(event) => changeChatTitleMode(event.target.value as ChatTitleMode)}
              >
                <option value="never">Never</option>
                <option value="always">Always</option>
                <option value="over-20-tokens">Over 20 tok/s</option>
              </select>
            </label>
            <div className="conversation-list">
              <div className="section-label">Recent</div>
              {conversations.map((conversation) => (
                <div
                  className={conversation.id === conversationId ? 'conversation active' : 'conversation'}
                  key={conversation.id}
                >
                  <button className="conversation-select" onClick={() => setConversationId(conversation.id)}>
                    <span>{conversation.title}</span>
                    <time>{conversation.updated_at.slice(0, 10)}</time>
                  </button>
                  <button
                    className="conversation-menu-button"
                    aria-label={`Actions for ${conversation.title}`}
                    title="Conversation actions"
                    onClick={() =>
                      setConversationMenuId((current) =>
                        current === conversation.id ? null : conversation.id
                      )
                    }
                  >
                    <Ellipsis size={16} />
                  </button>
                  {conversationMenuId === conversation.id ? (
                    <div className="conversation-menu">
                      <button onClick={() => void renameConversation(conversation)}>
                        <Pencil size={13} /> Rename
                      </button>
                      <button className="danger" onClick={() => void removeConversation(conversation)}>
                        <Trash2 size={13} /> Delete
                      </button>
                    </div>
                  ) : null}
                </div>
              ))}
              {conversations.length === 0 && (
                <p className="empty-sidebar">Your conversations stay on this device.</p>
              )}
            </div>
            <RunHistory
              runs={runSnapshots}
              expandedId={expandedRunId}
              onToggle={(id) => setExpandedRunId((current) => (current === id ? null : id))}
            />
          </>
        )}
        <div className="privacy-note">
          <span className={`status-dot ${daemonStatus}`} />
          {daemonStatus === 'healthy'
            ? `Daemon ${daemonVersion}`
            : daemonStatus === 'offline'
              ? 'Daemon unavailable'
              : 'Checking daemon…'}
        </div>
        <button
          type="button"
          className="sidebar-update-button"
          disabled={checkingForUpdates}
          title="Check for a Brazier app update"
          onClick={() => void checkForUpdates()}
        >
          <RefreshCw className={checkingForUpdates ? 'spin' : ''} size={13} />
          {checkingForUpdates ? 'Checking for updates…' : 'Check for updates'}
        </button>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <button className="icon-button" onClick={() => setSidebarOpen((open) => !open)}>
            <Menu size={19} />
          </button>
          <div className="mode-switch" role="tablist" aria-label="Workspace mode">
            {(
              [
                ['chat', 'Chat'],
                ['agent', 'Agent'],
                ['generate', 'Generate'],
                ['voice', 'Voice'],
                ['computer', 'Computer']
              ] as const
            )
              .filter(([id]) => workspaceModes[id])
              .map(([id, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                className={appMode === id ? 'active' : ''}
                aria-selected={appMode === id}
                onClick={() => switchAppMode(id)}
              >
                {label}
              </button>
            ))}
          </div>
          {(appMode === 'chat' || appMode === 'agent') &&
            selectedModel &&
            (modelPrepareState === 'ready' || modelUnloading) && (
              <button
                type="button"
                className="model-unload-button"
                disabled={modelUnloading || busy || Boolean(shellComposer?.running)}
                title={
                  busy || shellComposer?.running
                    ? 'Stop the active response before unloading the model'
                    : 'Unload model from memory'
                }
                aria-label="Unload model from memory"
                onClick={() => void unloadSelectedModel()}
              >
                {modelUnloading ? (
                  <LoaderCircle className="spin" size={14} />
                ) : (
                  <X size={14} />
                )}
              </button>
            )}
          <button
            className="model-picker"
            disabled={modelUnloading}
            title={
              appMode === 'chat'
                ? 'Choose which installed model to chat with'
                : appMode === 'agent'
                  ? 'Choose which installed model drives the agent'
                  : appMode === 'voice'
                    ? 'Choose which PersonaPlex model to speak with'
                    : appMode === 'computer'
                      ? 'Choose which computer-use model drives the session'
                      : `Choose which ${generateModality} model to generate with`
            }
            onClick={() => setModelMenuOpen(true)}
          >
            <div className="model-icon">
              <Box size={16} />
            </div>
            <div>
              <div className="model-title-row">
                <strong>{selectedMeta.title}</strong>
                {(appMode === 'chat' || appMode === 'agent') &&
                  modelPrepareState === 'ready' &&
                  modelResidency && (
                    <i
                      className={`model-residency-dot ${modelResidency.placement}`}
                      role="img"
                      aria-label={modelResidency.description}
                      title={modelResidency.description}
                    />
                  )}
              </div>
              <span>{selectedMeta.subtitle}</span>
            </div>
          </button>
          <button
            className="icon-button"
            title="Inference settings (sampling, reasoning)"
            onClick={() => setInferenceMenuOpen(true)}
          >
            <SlidersHorizontal size={17} />
          </button>
          <button
            className="icon-button manage-button"
            title="Manage models, runtimes, and engine configuration"
            onClick={() => openManage('library')}
          >
            <Settings2 size={18} />
          </button>
        </header>

        {runtimeWarning && (
          <div className="runtime-notice">
            <span>{runtimeWarning}</span>
            <button type="button" onClick={() => openManage('runtimes')}>
              Open Runtimes
            </button>
          </div>
        )}

        {modelPrepareState === 'loading' && modelLoadStatus && (
          <div className="runtime-notice model-prepare-notice">
            <LoaderCircle className="spin" size={16} />
            <span>{modelLoadStatus}</span>
          </div>
        )}

        {appMode === 'generate' ? (
          <GenerateMode
            models={generateModels}
            modality={generateModality}
            onModalityChange={setGenerateModality}
            modelId={generateModel}
            settings={runtime}
            hardware={hardware}
            onError={setError}
            history={generateHistory}
            activeHistoryId={activeGenerateHistoryId}
            onGenerated={(entry) => {
              setGenerateHistory((current) => {
                const next = [entry, ...current]
                writeGenerateHistory(next)
                return next
              })
              setActiveGenerateHistoryId(entry.id)
            }}
          />
        ) : null}
        {appMode === 'voice' ? (
          <VoiceMode
            models={localModels}
            realtimeAvailable={realtimeVoiceAvailable}
            modelId={voiceModel}
            audioSupported={audioSupported}
            asrAvailable={{
              batch: Boolean(pipelineFeatures.asr),
              streaming: Boolean(pipelineFeatures.streaming_asr)
            }}
            chatModelId={selectedModel}
            onChatModelChange={selectModel}
            tools={availableTools}
            enabledTools={enabledTools}
            onEnabledToolsChange={updateEnabledTools}
            settings={runtime}
            onSettingsSaved={(saved) => {
              setRuntime(saved)
              // Choosing a Whisper or Nemotron model can be what makes
              // transcription available at all.
              void refreshCapabilities()
            }}
            onRuntimeActivated={() => void refreshCapabilities()}
            onAgentSessionBound={(agentSessionId) =>
              void session.bindAgentSession(agentSessionId)
            }
            persona={persona}
            onPersonaChange={(next) => {
              personaEdited.current = true
              setPersona(next)
            }}
            session={session}
            onError={setError}
          />
        ) : null}
        {appMode === 'agent' ? (
          <AgentMode
            modelId={selectedModel}
            models={localModels}
            onComposerChange={setAgentComposer}
            onSidebarChange={setAgentSidebar}
            onSuggestPrompt={setDraft}
            onSessionBound={session.bindAgentSession}
            onError={setError}
          />
        ) : null}
        {appMode === 'computer' ? (
          <ComputerMode
            modelId={computerModel}
            models={computerModels}
            onComposerChange={setComputerComposer}
            onError={setError}
          />
        ) : null}

        <div className="chat" hidden={appMode !== 'chat'}>
          {modelLoadStatus && (modelPrepareState === 'loading' || (busy && !streamingText)) && (
            <div className="model-load-status">
              <LoaderCircle className="spin" size={18} />
              <span>{modelLoadStatus}</span>
            </div>
          )}
          {chain.length === 0 && !liveText ? (
            <div className="welcome">
              <div className="welcome-mark">
                <Bot size={30} />
              </div>
              <h1>What are we exploring?</h1>
              <p>
                {modelsLoading
                  ? 'Starting the local runtime and loading your model library…'
                  : !selectedModel
                    ? 'Choose a model to load it locally. Nothing is selected yet.'
                    : modelPrepareState === 'loading'
                      ? 'Loading the model and runtime…'
                      : modelPrepareState === 'error'
                        ? 'Model load failed. Check the error below or open Manage to adjust the runtime pairing.'
                        : canChat
                          ? 'Chat privately with local models. Attach media or start with a question.'
                          : 'Download a model from Hugging Face to start chatting locally.'}
              </p>
              <div className="starter-grid">
                {canChat ? (
                  <button onClick={() => setDraft('Explain how speculative decoding works.')}>
                    <Brain size={18} />
                    <span>
                      <strong>Explore a concept</strong>
                      Speculative decoding
                    </span>
                  </button>
                ) : (
                  <button onClick={() => openManage('discover')}>
                    <Box size={18} />
                    <span>
                      <strong>Browse models</strong>
                      Find models on Hugging Face
                    </span>
                  </button>
                )}
                <button onClick={() => fileInput.current?.click()}>
                  <Paperclip size={18} />
                  <span>
                    <strong>Analyze media</strong>
                    Image, audio, or video
                  </span>
                </button>
              </div>
            </div>
          ) : (
            <div className="messages">
              {chatDisplayItems.map((item) => {
                if (item.kind === 'assistant') {
                  const status =
                    item.status === 'cancelled' ||
                    item.status === 'superseded' ||
                    item.status === 'failed'
                      ? item.status
                      : null
                  return (
                    <article className="message assistant" key={item.id}>
                      <div className="avatar">
                        <Bot />
                      </div>
                      <div className="message-body">
                        <div className="message-meta">
                          <strong>{item.source === 'assistant_agent' ? 'Agent' : 'Brazier'}</strong>
                          {status ? (
                            <span className={`turn-badge ${status}`}>{status}</span>
                          ) : null}
                        </div>
                        <ReasoningDisclosure text={item.reasoning} />
                        {item.segments.map((segment) => {
                          if (segment.kind === 'tool') {
                            return (
                              <ToolChips
                                key={segment.key}
                                records={segment.records}
                                onError={setError}
                              />
                            )
                          }
                          if (segment.kind === 'media') {
                            return (
                              <MessageMedia
                                key={segment.key}
                                blobs={segment.blobs}
                                onError={setError}
                              />
                            )
                          }
                          return (
                            <Markdown key={segment.key}>{segment.text}</Markdown>
                          )
                        })}
                        <BranchNavigator
                          messages={messages}
                          messageId={item.branchId}
                          onSelect={setTipId}
                        />
                      </div>
                    </article>
                  )
                }

                const message = item.message
                const media = contentMedia(message)
                const spoken = message.source === 'user_voice'
                const turn = turnLabel(message)
                return (
                  <article className={`message ${message.role}`} key={message.id}>
                    <div className="avatar">You</div>
                    <div className="message-body">
                      <div className="message-meta">
                        <strong>You</strong>
                        {spoken ? (
                          <span className="message-source" title="Spoken, transcribed into chat">
                            <Mic size={11} /> voice
                          </span>
                        ) : null}
                        {turn ? <span className={`turn-badge ${turn}`}>{turn}</span> : null}
                      </div>
                      {media.length > 0 && (
                        <div className="media-row">
                          {media.map((kind, index) => (
                            <span key={`${kind}-${index}`}>
                              {kind === 'image' && <Image size={14} />}
                              {kind === 'audio' && <AudioLines size={14} />}
                              {kind === 'video' && <Video size={14} />}
                              {kind}
                            </span>
                          ))}
                        </div>
                      )}
                      <MessageMedia blobs={contentBlobs(message)} onError={setError} />
                      <Markdown>{contentText(message)}</Markdown>
                      <button
                        className="fork-button"
                        title="Edit this message and send an alternate branch"
                        onClick={() => {
                          setDraft(contentText(message))
                          setAttachments([])
                          setTipId(message.parent_id)
                        }}
                      >
                        <Pencil size={13} /> Edit and branch
                      </button>
                      <BranchNavigator
                        messages={messages}
                        messageId={message.id}
                        onSelect={setTipId}
                      />
                    </div>
                  </article>
                )
              })}
              {(liveText || streamingReasoning || streamingTools.length > 0) && (
                <article className="message assistant">
                  <div className="avatar">
                    <Bot />
                  </div>
                  <div className="message-body">
                    <div className="message-meta">
                      <strong>{session.snapshot.streamingText ? 'Agent' : 'Brazier'}</strong>
                      {busy ? <LoaderCircle className="spin" size={14} /> : null}
                    </div>
                    <ReasoningDisclosure text={streamingReasoning} defaultOpen />
                    <StreamingTurnSegments
                      text={liveText}
                      records={streamingTools}
                      offsets={streamingToolOffsets}
                      onError={setError}
                    />
                  </div>
                </article>
              )}
            </div>
          )}
          <div ref={scrollAnchor} />
        </div>

        <div className="composer-area">
          {/* Sits above the composer in every mode: a generation a model
              started is otherwise invisible until it finishes. */}
          <GenerationActivity onStopped={() => void refreshLocalModels().catch(() => {})} />
          {error && (
            <div className="error-banner">
              <div className="error-banner-body">
                <span>{error}</span>
                {forkHints.length > 0 && (
                  <div className="fork-hints">
                    <div className="section-label">Runtime forks linked in model README</div>
                    {forkHints.map((hint) => (
                      <div className="fork-hint-row" key={`${hint.engine}:${hint.repository}`}>
                        <div>
                          <strong>{hint.display_name}</strong>
                          <span>{hint.repository}</span>
                        </div>
                        <button type="button" onClick={() => startForkBuild(hint)}>
                          Build fork
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <button
                onClick={() => {
                  setError(null)
                  setForkHints([])
                }}
              >
                <X size={14} />
              </button>
            </div>
          )}
          {attachments.length > 0 && (
            <div className="attachment-tray">
              {attachments.map((attachment) => (
                <span key={attachment.id}>
                  {attachment.type.startsWith('image/') && <Image size={14} />}
                  {attachment.type.startsWith('audio/') && <AudioLines size={14} />}
                  {attachment.type.startsWith('video/') && <Video size={14} />}
                  {attachment.name}
                  <button
                    onClick={() =>
                      setAttachments((current) =>
                        current.filter((candidate) => candidate.id !== attachment.id)
                      )
                    }
                  >
                    <X size={13} />
                  </button>
                </span>
              ))}
            </div>
          )}
          {/* One composer, for the modes that take typed input. Voice owns its
              whole screen and has no text box, and Generate carries its own
              prompt form: in both cases a second input would be a second,
              unlabelled one. */}
          <form
            className="composer"
            hidden={appMode === 'voice' || appMode === 'generate'}
            onSubmit={(event) => void submit(event)}
          >
            <textarea
              aria-label={
                agentMode ? 'Agent task' : computerMode ? 'Computer Use task' : 'Message'
              }
              placeholder={
                shellComposerMode
                  ? (shellComposer?.placeholder ??
                    (agentMode ? 'Loading agent…' : 'Loading computer use…'))
                  : !selectedModel
                    ? 'Select a model to start chatting…'
                    : modelPrepareState === 'loading'
                      ? 'Loading model…'
                      : modelPrepareState === 'error'
                        ? 'Fix model load to chat…'
                        : tipId
                          ? 'Continue this branch…'
                          : 'Message a local model…'
              }
              rows={shellComposerMode ? 2 : 1}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  event.currentTarget.form?.requestSubmit()
                }
              }}
            />
            {shellComposerMode && shellComposer?.suggestions && (() => {
              const match = draft.match(/(?:^|\s)([a-z]*)$/)
              const query = match?.[1] ?? ''
              const suggestions = shellComposer.suggestions.filter((entry) =>
                entry.value.startsWith(query.toLowerCase())
              )
              if (suggestions.length === 0 || (query && suggestions[0]?.value === query)) return null
              return (
                <div className="composer-suggestions" role="listbox" aria-label="OMP prompt suggestions">
                  {suggestions.map((entry) => (
                    <button
                      type="button"
                      key={entry.value}
                      onClick={() => setDraft((current) => current.replace(/(?:^|\s)[a-z]*$/, (word) => `${word.startsWith(' ') ? ' ' : ''}${entry.value} `))}
                    >
                      <strong>{entry.value}</strong><span>{entry.description}</span>
                    </button>
                  ))}
                </div>
              )
            })()}
            <div className="composer-actions">
              <input
                ref={fileInput}
                type="file"
                accept={[
                  canAttachDocuments
                    ? [
                        '.pdf',
                        '.rtf',
                        '.doc',
                        '.docx',
                        '.txt',
                        '.md',
                        '.csv',
                        '.json',
                        '.xml',
                        '.html',
                        '.htm',
                        'text/*',
                        'application/pdf',
                        'application/rtf',
                        'text/rtf',
                        'application/msword',
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        'application/json',
                        'application/xml'
                      ].join(',')
                    : null,
                  canAttachImage ? 'image/*' : null,
                  canAttachAudio ? 'audio/*' : null,
                  canAttachVideo ? 'video/*' : null
                ]
                  .filter(Boolean)
                  .join(',') || 'image/*'}
                multiple
                hidden
                onChange={(event) => void selectFiles(event)}
              />
              {/* Chat tools and attachments do not apply to agent/computer: their
                  tool sets and permissions are their own, and they take no media. */}
              <div className="tool-menu-anchor" hidden={shellComposerMode}>
                <button
                  className={toolsEnabled ? 'attach-button tools-on' : 'attach-button'}
                  type="button"
                  disabled={!canUseTools}
                  title={
                    !canUseTools
                      ? 'This model does not advertise tool support'
                      : toolsEnabled
                        ? `Tools: ${enabledTools.length} enabled`
                        : 'Choose tools (bundled + MCP)'
                  }
                  onClick={() => {
                    void refreshTools()
                    setToolsMenuOpen((open) => !open)
                  }}
                >
                  <Wrench size={17} />
                  {toolsEnabled && <span className="tool-count">{enabledTools.length}</span>}
                </button>
                {toolsMenuOpen && (
                  <ToolsMenu
                    tools={availableTools}
                    enabled={enabledTools}
                    disabled={!canUseTools}
                    onToggle={(name, on) =>
                      updateEnabledTools(
                        on
                          ? Array.from(new Set([...enabledTools, name]))
                          : enabledTools.filter((entry) => entry !== name)
                      )
                    }
                    onSetAll={(names, on) =>
                      updateEnabledTools(
                        on
                          ? Array.from(new Set([...enabledTools, ...names]))
                          : enabledTools.filter((entry) => !names.includes(entry))
                      )
                    }
                    onClose={() => setToolsMenuOpen(false)}
                  />
                )}
              </div>
              <button
                className="attach-button"
                type="button"
                hidden={shellComposerMode}
                title={
                  canAttach
                    ? [
                        canAttachDocuments ? 'documents' : null,
                        canAttachImage ? 'images' : null,
                        canAttachAudio ? 'audio' : null,
                        canAttachVideo ? 'video' : null
                      ]
                        .filter(Boolean)
                        .join(', ')
                    : 'Attach a document, or configure vision and/or ASR for media'
                }
                onClick={() => fileInput.current?.click()}
              >
                <Paperclip size={18} />
              </button>
              {(shellComposerMode ? shellComposer?.running : busy) ? (
                <button
                  className="send-button stop"
                  type="button"
                  title={
                    agentMode
                      ? 'Stop the run, terminate its processes, and refuse pending approvals'
                      : computerMode
                        ? 'Stop the computer-use loop'
                        : 'Stop generation'
                  }
                  onClick={() => {
                    if (shellComposerMode) void shellComposer?.stop()
                    else abortRef.current?.abort()
                  }}
                >
                  <Square size={15} fill="currentColor" />
                </button>
              ) : (
                <button
                  className="send-button"
                  type="submit"
                  title={
                    agentMode ? 'Start the task' : computerMode ? 'Start computer use' : 'Send'
                  }
                  disabled={
                    shellComposerMode
                      ? !draft.trim() || !shellComposer || shellComposer.blockedReason !== ''
                      : (!draft.trim() && attachments.length === 0) || !canChat
                  }
                >
                  <Send size={17} />
                </button>
              )}
            </div>
          </form>
          <p className="composer-hint" hidden={appMode === 'voice' || appMode === 'generate'}>
            {agentMode ? (
              'The agent edits files and runs commands in the workspace above. Each action is judged by its permission mode.'
            ) : computerMode ? (
              'Computer Use screenshots the target, asks the model, and runs the returned actions under the permission mode above.'
            ) : (
              <>
                {generationRate != null ? (
                  <span
                    className="generation-rate"
                    title="Local engine decode rate, measured from its token stream."
                  >
                    {generationRate.toFixed(1)} tok/s
                  </span>
                ) : null}
                {generationTokens ? (
                  <span className="generation-tokens" title="Token usage reported by the inference server.">
                    {generationTokens.prompt != null
                      ? `${generationTokens.prompt.toLocaleString()} prompt · `
                      : ''}
                    {generationTokens.completion.toLocaleString()} output tokens
                  </span>
                ) : null}
                Local models can be inaccurate. Verify important information.
                <span className="capabilities" aria-label="Model capabilities">
                  <span
                    title={`Acceleration target: ${runtime?.target ?? 'auto'}`}
                    aria-label={`Acceleration target: ${runtime?.target ?? 'auto'}`}
                  >
                    <Cpu size={14} />
                  </span>
                  <span
                    title={`Context window: ${runtime?.context_size?.toLocaleString() ?? '4,096'} tokens${
                      selectedCapabilities?.max_context_length
                        ? ` · ${selectedCapabilities.max_context_length.toLocaleString()} supported by this model`
                        : ''
                    }`}
                    aria-label="Context window"
                  >
                    <Gauge size={14} />
                  </span>
                  <span
                    className={
                      (selectedCapabilities?.reasoning_modes?.length ??
                        (selectedCapabilities?.reasoning ? 1 : 0)) > 0
                        ? ''
                        : 'unavailable'
                    }
                    title={
                      (selectedCapabilities?.reasoning_modes?.length ??
                        (selectedCapabilities?.reasoning ? 1 : 0)) > 0
                        ? 'Reasoning: this model can think before answering'
                        : 'Reasoning: not advertised by this model'
                    }
                    aria-label="Reasoning"
                  >
                    <Brain size={14} />
                  </span>
                  <span
                    className={canAttachImage ? '' : 'unavailable'}
                    title={`Vision — ${visionCapabilityTitle(selectedModel, localModels, canAttachImage)}`}
                    aria-label="Vision"
                  >
                    <Image size={14} />
                  </span>
                  <span
                    className={canAttachAudio ? '' : 'unavailable'}
                    title={`${
                      selectedNativeAudio ? 'Native audio' : pipelineFeatures.asr ? 'ASR' : 'Audio'
                    } — ${audioBadgeTitle}`}
                    aria-label={
                      selectedNativeAudio ? 'Native audio' : pipelineFeatures.asr ? 'ASR' : 'Audio'
                    }
                  >
                    <AudioLines size={14} />
                  </span>
                  <span
                    className={canAttachVideo ? '' : 'unavailable'}
                    title={
                      canAttachVideo
                        ? 'Video: sampled with ffmpeg and transcribed when ASR is available'
                        : 'Video: needs ffmpeg plus a vision model (and whisper.cpp for the soundtrack)'
                    }
                    aria-label="Video"
                  >
                    <Video size={14} />
                  </span>
                  <span
                    className={selectedCapabilities?.tools ? '' : 'unavailable'}
                    title={
                      selectedCapabilities?.tools
                        ? 'Tools: this model advertises tool calling'
                        : 'Tools: not advertised by this model'
                    }
                    aria-label="Tools"
                  >
                    <Wrench size={14} />
                  </span>
                </span>
                {toolsEnabled && canUseTools ? ' Tools are enabled (bundled + MCP).' : ''}
                {selectedCapabilities?.harmony
                  ? ' This model uses OpenAI Harmony (gpt-oss); reasoning is routed automatically.'
                  : ''}
              </>
            )}
          </p>
        </div>
      </section>

      {modelMenuOpen && (
        <ModelMenu
          models={modeModel.models}
          title={
            appMode === 'chat'
              ? 'Choose a model'
              : appMode === 'agent'
                ? 'Choose a model for the agent'
                : appMode === 'voice'
                  ? 'Choose a voice model'
                  : appMode === 'computer'
                    ? 'Choose a computer-use model'
                    : `Choose a ${generateModality} model`
          }
          selectedModel={modeModel.selected}
          loading={modelsLoading}
          videoPipeline={Boolean(pipelineFeatures.video_preprocess)}
          onSelect={modeModel.select}
          onConfigure={setConfiguringModel}
          configuredCounts={profileCounts}
          onManage={() =>
            openManage(
              appMode === 'computer'
                ? 'computer'
                : modelsLoadFailed || localModels.length === 0
                  ? 'discover'
                  : 'library'
            )
          }
          onClose={() => setModelMenuOpen(false)}
        />
      )}
      {inferenceMenuOpen && (
        <InferenceMenu
          settings={runtime}
          hardware={hardware}
          selectedModel={selectedModel}
          models={localModels}
          saving={savingInference}
          advancedModelId={modeModel.selected}
          profile={modelProfiles[modeModel.selected]}
          onApply={(next) => void applyInferenceSettings(next)}
          onProfileSaved={(models) => {
            setModelProfiles(models)
            if (modeModel.selected === selectedModel && modelPrepareState === 'ready') {
              selectModel(selectedModel)
            }
          }}
          onClose={() => setInferenceMenuOpen(false)}
        />
      )}
      {configuringModel && configuredModelEntry && (
        <ModelSettingsModal
          model={configuredModelEntry}
          kind={modelKindFor(configuringModel)}
          profile={modelProfiles[configuringModel]}
          settings={runtime}
          hardware={hardware}
          onSaved={(models) => {
            setModelProfiles(models)
            if (configuringModel === selectedModel && modelPrepareState === 'ready') {
              selectModel(selectedModel)
            }
          }}
          onClose={() => setConfiguringModel(null)}
        />
      )}
      {recommendationSwaps.length > 0 && (
        <aside className="recommendation-swap-notice" aria-live="polite">
          <Sparkles size={16} />
          <div>
            <strong>A recommended model has changed</strong>
            <span>
              {recommendationSwaps
                .map((swap) => `${swap.recommended_label} for ${swap.category}`)
                .join(' · ')}
            </span>
          </div>
          <button
            type="button"
            className="chip-button"
            onClick={() => {
              setManageSection('recommended')
              setManageOpen(true)
              setRecommendationSwaps([])
            }}
          >
            Review
          </button>
          <button
            type="button"
            className="icon-button"
            title="Dismiss these suggestions"
            onClick={() => {
              const swaps = recommendationSwaps
              setRecommendationSwaps([])
              void Promise.all(
                swaps.map((swap) =>
                  updateRecommendationState({ dismiss: swap.recommended_id })
                )
              ).catch(() => {})
            }}
          >
            <X size={14} />
          </button>
        </aside>
      )}
      <DownloadTray onChanged={() => void refreshLocalModels().catch(() => {})} />

      {manageOpen && (
        <ManagePanel
          section={manageSection}
          onSectionChange={setManageSection}
          onClose={() => {
            setManageOpen(false)
            void prefetchRuntimes()
            void refreshTools()
          }}
          pendingBuild={pendingBuild}
          onPendingBuildConsumed={() => setPendingBuild(null)}
          models={localModels}
          modelsLoading={modelsLoading}
          refreshModels={refreshLocalModels}
          initialRuntimes={prefetchedRuntimes}
          selectedModel={selectedModel}
          onSelectModel={selectModel}
          modelBindings={modelBindings}
          onSetModelBinding={(modelId, runtimeId) => void updateModelBinding(modelId, runtimeId)}
          settings={runtime}
          onSettingsSaved={setRuntime}
          hardware={hardware}
          onConfigureModel={setConfiguringModel}
          profileCounts={profileCounts}
          onWorkspaceModesChange={setWorkspaceModes}
        />
      )}
    </main>
  )
}

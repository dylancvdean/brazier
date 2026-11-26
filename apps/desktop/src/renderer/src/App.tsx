import {
  AudioLines,
  Bot,
  Box,
  Brain,
  Check,
  CircleAlert,
  Copy,
  Cpu,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Ellipsis,
  EyeOff,
  Gauge,
  Image,
  LoaderCircle,
  Menu,
  MessageSquarePlus,
  Mic,
  Paperclip,
  Download,
  RefreshCw,
  RotateCcw,
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
  X,
  Zap,
  ListPlus
} from 'lucide-react'
import {
  type ChangeEvent,
  type ClipboardEvent,
  type DragEvent,
  type FormEvent,
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState
} from 'react'
import {
  createConversation,
  createMessage,
  deleteConversation,
  engineStatus,
  fetchCapabilities,
  fetchModelBindings,
  fetchRecommendations,
  fetchModelSettings,
  verifyDaemonConnection,
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
  type WorkspaceModesPreference,
  fetchMemoryPreference,
  saveMemoryPreference,
  DEFAULT_MEMORY_PREFERENCE,
  listMemories,
  type ClientToolCall,
  type MemoryPreference,
  fetchCurrentConnectionProfile,
  rememberedConnectionProfileId,
  type ConnectionProfileSummary
} from './api'
import { AgentMode, type AgentComposerControls, type AgentSidebarControls } from './components/AgentMode'
import { AgentSessionSidebar } from './components/AgentSessionSidebar'
import { ComputerMode, type ComputerSidebarControls } from './components/ComputerMode'
import { ComputerSessionSidebar } from './components/ComputerSessionSidebar'
import { DownloadTray } from './components/DownloadTray'
import { GenerationActivity } from './components/GenerationActivity'
import {
  GenerateHistorySidebar,
  type GenerateHistoryEntry
} from './components/GenerateHistorySidebar'
import { MessageMedia } from './components/MessageMedia'
import { Markdown } from './components/Markdown'
import { ErrorBoundary, RenderSlot } from './components/ErrorBoundary'
import { ReasoningDisclosure, TurnTrace } from './components/ReasoningDisclosure'
import { CommandPalette } from './components/CommandPalette'
import { GenerateMode } from './components/GenerateMode'
import { InferenceMenu } from './components/InferenceMenu'
import { ManagePanel, buildServerCurlSnippet, type ManageSection } from './components/ManagePanel'
import { ModelMenu } from './components/ModelMenu'
import { profileCount } from './components/ModelSettingsFields'
import { ModelSettingsModal } from './components/ModelSettingsModal'
import { CATEGORY_LABELS } from './components/RecommendedModels'
import { ToolsMenu } from './components/ToolsMenu'
import { VoiceMode } from './components/VoiceMode'
import { WelcomeScreen } from './components/WelcomeScreen'
import brazierLogo from './assets/brazier-logo.png'
import { hasCompletedWelcome, markWelcomeCompleted } from './welcomePrefs'
import { readChatTitleMode, type ChatTitleMode } from './chatTitleMode'
import { voiceStreamSupported } from './audio/voiceStream'
import { useSessionCoordinator } from './session/useSessionCoordinator'
import {
  buildMemoryContext,
  dream,
  executeMemoryClientTool,
  isMemoryClientTool,
  loadRelevantMemories,
  memoryToolDefinitions,
  type DreamInputConversation
} from './memory'
import {
  dataDirFromManagedBinaryPath,
  isChatModel,
  isComputerUseModel,
  isImageGenModel,
  isVideoGenModel,
  isVoiceModel,
  modelDisplayName,
  modelKindFor,
  runtimeNoticeForModel,
  sortComputerUseModels,
  visionCapabilityTitle,
  withMergedCapabilities
} from './model-utils'
import { branchSiblings, messageChain, refreshedMessageTip, ROOT_TIP } from './graph'
import { buildChatDisplayItems } from './chatDisplay'
import {
  CHAT_SLASH_SUGGESTIONS,
  composerCompletionsFor,
  isExactSlashCommand,
  replaceTrailingCommand,
  trailingCommand
} from './composerCompletions'
import {
  readCachedConversations,
  readCachedModels,
  readCachedRuntimes,
  writeCachedConversations,
  writeCachedModels,
  writeCachedRuntimes
} from './inventoryCache'
import type { Attachment, ContentPart, Conversation, Message, Role } from './types'
import { ConnectionProfileContext, LOCAL_CONNECTION_PROFILE } from './connectionProfile'
import {
  DaemonOfflineError,
  daemonAvailability,
  setDaemonAvailability
} from './daemonAvailability'
import {
  INITIAL_DAEMON_CONNECTION,
  availabilityForEvent,
  daemonConnectionReducer,
  daemonStatusOf,
  type DaemonConnectionEvent
} from './daemonConnection'
import { cacheVerifiedConnection, connection } from './daemonClient'
import { selectedConversationWasRemoved, startConversationSync } from './conversationSync'

const ENABLED_TOOLS_KEY = 'brazier.enabledTools'
const GENERATE_HISTORY_KEY = 'brazier.generateHistory.v1'
/** How long the completed-dream notice stays before clearing itself. */
const DREAM_STATUS_DISMISS_MS = 8_000

type AppMode = 'chat' | 'agent' | 'generate' | 'voice' | 'computer'

const DEFAULT_WORKSPACE_MODES: WorkspaceModesPreference = {
  chat: true,
  agent: true,
  generate: true,
  voice: true,
  computer: false
}

function shortcutLabel(key: string, shift = false): string {
  return window.brazier.platform === 'darwin'
    ? `⌘${shift ? '⇧' : ''}${key}`
    : `Ctrl+${shift ? 'Shift+' : ''}${key}`
}

function profileStorageKey(base: string, profileId: string): string {
  return `${base}.${encodeURIComponent(profileId)}`
}

function readGenerateHistory(profileId: string): GenerateHistoryEntry[] {
  try {
    const value = localStorage.getItem(profileStorageKey(GENERATE_HISTORY_KEY, profileId))
    const parsed = value ? (JSON.parse(value) as unknown) : []
    return Array.isArray(parsed) ? (parsed as GenerateHistoryEntry[]) : []
  } catch {
    return []
  }
}

function writeGenerateHistory(profileId: string, entries: GenerateHistoryEntry[]): void {
  try {
    localStorage.setItem(
      profileStorageKey(GENERATE_HISTORY_KEY, profileId),
      JSON.stringify(entries.slice(0, 100))
    )
  } catch {
    // Best-effort local history.
  }
}

/** "14:05" today, "Yesterday", a weekday within the week, otherwise "Sep 3". */
/** Sidebar bucket for a chat's last activity. */
function historyGroup(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return 'Older'
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const days = Math.ceil((today.getTime() - date.getTime()) / 86_400_000)
  if (days <= 0) return 'Today'
  if (days === 1) return 'Yesterday'
  if (days < 7) return 'Previous 7 days'
  if (days < 30) return 'Previous 30 days'
  return 'Older'
}

function shortDate(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  const now = new Date()
  const startOfDay = (value: Date): number =>
    new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime()
  const days = Math.round((startOfDay(now) - startOfDay(date)) / 86_400_000)
  if (days <= 0) return date.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
  if (days === 1) return 'Yesterday'
  if (days < 7) return date.toLocaleDateString(undefined, { weekday: 'short' })
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    ...(date.getFullYear() !== now.getFullYear() ? { year: 'numeric' } : {})
  })
}

function titleFromCompletion(text: string): string | null {
  const title = text
    .replace(/[\r\n]+/g, ' ')
    .replace(/^\s*["'`]+|["'`]+\s*$/g, '')
    .trim()
  return title ? title.slice(0, 80) : null
}

/**
 * Turn a model-load progress event into a status line that says what is really
 * happening. The daemon's own messages describe a normal chat turn ("Model
 * ready — generating…"), which is misleading mid-dream, so they are relabeled.
 */
function dreamStatusForLoad(event: { phase: string; message: string }): string {
  if (event.phase === 'ready') return 'Dreaming…'
  return `Dreaming: ${event.message}`
}

function readEnabledTools(profileId: string): string[] {
  try {
    const raw = localStorage.getItem(profileStorageKey(ENABLED_TOOLS_KEY, profileId))
    if (!raw) return []
    const parsed = JSON.parse(raw) as unknown
    return Array.isArray(parsed) ? parsed.filter((value): value is string => typeof value === 'string') : []
  } catch {
    return []
  }
}

function writeEnabledTools(profileId: string, names: string[]): void {
  try {
    localStorage.setItem(profileStorageKey(ENABLED_TOOLS_KEY, profileId), JSON.stringify(names))
  } catch {
    // Best-effort persistence.
  }
}

/**
 * Walk upward from an assistant turn node until reaching the direct child of
 * the preceding user message, so multi-step tool turns share branch siblings.
 */
function assistantTurnRootId(messages: Message[], branchId: string): string {
  const byId = new Map(messages.map((message) => [message.id, message]))
  let current = byId.get(branchId)
  while (current?.parent_id) {
    const parent = byId.get(current.parent_id)
    if (!parent || parent.role === 'user') break
    current = parent
  }
  return current?.id ?? branchId
}

function parentUserMessageForTurn(messages: Message[], branchId: string): Message | null {
  const byId = new Map(messages.map((message) => [message.id, message]))
  const root = byId.get(assistantTurnRootId(messages, branchId))
  if (!root?.parent_id) return null
  const parent = byId.get(root.parent_id)
  return parent?.role === 'user' ? parent : null
}

/**
 * Resolve the latest descendant leaf for a selected branch node so switching
 * branches shows the complete conversation subtree rather than stopping early.
 */
function latestBranchTipId(messages: Message[], startId: string): string {
  const childrenByParent = new Map<string, Message[]>()
  for (const message of messages) {
    if (!message.parent_id) continue
    const list = childrenByParent.get(message.parent_id)
    if (list) list.push(message)
    else childrenByParent.set(message.parent_id, [message])
  }
  let currentId = startId
  const visited = new Set<string>()
  while (!visited.has(currentId)) {
    visited.add(currentId)
    const children = childrenByParent.get(currentId)
    const next = children?.at(-1)
    if (!next) break
    currentId = next.id
  }
  return currentId
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
  if (typeof message.content === 'string') return message.content.trim()
  return message.content
    .filter((part): part is Extract<ContentPart, { type: 'text' }> => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
    .trim()
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
      // Streaming tool events carry `name`; chatDisplay re-attached blobs carry
      // `original_name`. Either way, never fall back to showing the MIME type
      // as the file's title.
      original_name: blob.original_name ?? blob.name
    }))
  )
  return (
    <>
      <div className="tool-chip-row">
        {records.map((record, index) => (
          <details
            className={record.is_error ? 'tool-chip error' : 'tool-chip'}
            key={record.call_id ?? record.name ?? index}
          >
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
    <details className="run-history">
      <summary className="section-label run-history-summary">
        <span>Run history</span>
        <span className="run-history-count">{runs.length}</span>
      </summary>
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
                {run.response_text && <p>{run.response_text.trim().slice(0, 400)}</p>}
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
    </details>
  )
}

export function App(): React.JSX.Element {
  const initialProfileId = useRef(rememberedConnectionProfileId()).current
  const [connectionProfile, setConnectionProfile] = useState<ConnectionProfileSummary | null>(null)
  const [conversations, setConversations] = useState<Conversation[]>(() =>
    readCachedConversations(initialProfileId)
  )
  const [conversationSearch, setConversationSearch] = useState('')
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [conversationMenuId, setConversationMenuId] = useState<string | null>(null)
  const [renaming, setRenaming] = useState<{ id: string; draft: string } | null>(null)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [tipId, setTipId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const draftsRef = useRef({ chat: '', agent: '', computer: '' })
  const chatTurnRef = useRef(0)
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
  const [localModels, setLocalModels] = useState<LocalModel[]>(() =>
    readCachedModels(initialProfileId)
  )
  const [modelsLoading, setModelsLoading] = useState(
    () => readCachedModels(initialProfileId).length === 0
  )
  const [modelsLoadFailed, setModelsLoadFailed] = useState(false)
  const [selectedModel, setSelectedModel] = useState('')
  const [modelPrepareState, setModelPrepareState] = useState<
    'idle' | 'loading' | 'ready' | 'error'
  >('idle')
  const [modelResidency, setModelResidency] = useState<ModelResidency | null>(null)
  const [modelUnloading, setModelUnloading] = useState(false)
  const [modelBindings, setModelBindings] = useState<Record<string, string>>({})
  const [prefetchedRuntimes, setPrefetchedRuntimes] = useState<RuntimeEntry[] | null>(() => {
    const cached = readCachedRuntimes(initialProfileId)
    return cached.length > 0 ? cached : null
  })
  const [enabledTools, setEnabledTools] = useState<string[]>(() =>
    readEnabledTools(initialProfileId)
  )
  const [availableTools, setAvailableTools] = useState<BundledTool[]>([])
  const [toolsMenuOpen, setToolsMenuOpen] = useState(false)
  const toolsEnabled = enabledTools.length > 0
  const [daemonConnection, dispatchDaemonConnection] = useReducer(
    daemonConnectionReducer,
    INITIAL_DAEMON_CONNECTION
  )
  const daemonStatus = daemonStatusOf(daemonConnection)
  const daemonVersion = daemonConnection.version
  const [connectionProblem, setConnectionProblem] = useState<string | null>(null)
  const connectionCheckRef = useRef(0)
  const updateDaemonConnection = useCallback((event: DaemonConnectionEvent): void => {
    // The request gate changes synchronously so no mutation slips through
    // between this event and the next render.
    const availability = availabilityForEvent(event)
    if (availability) setDaemonAvailability(availability)
    dispatchDaemonConnection(event)
  }, [])
  // Background loads and saves report failures instead of leaving stale
  // controls; while offline the connection banner already explains them.
  const reportBackgroundFailure = useCallback((cause: unknown): void => {
    if (daemonAvailability() === 'offline' || cause instanceof DaemonOfflineError) return
    if (cause instanceof DOMException && cause.name === 'AbortError') return
    setError(cause instanceof Error ? cause.message : String(cause))
  }, [])
  const [hardware, setHardware] = useState<HardwareInfo | null>(null)
  const [runtime, setRuntime] = useState<RuntimeSettings | null>(null)
  const [daemonDataDir, setDaemonDataDir] = useState<string | null>(null)
  const [savingInference, setSavingInference] = useState(false)

  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false)
  const [modelMenuOpen, setModelMenuOpen] = useState(false)
  const [inferenceMenuOpen, setInferenceMenuOpen] = useState(false)
  const [composerDragOver, setComposerDragOver] = useState(false)
  const dragDepthRef = useRef(0)
  const [suggestionIndex, setSuggestionIndex] = useState(0)
  const [dismissedSuggestionQuery, setDismissedSuggestionQuery] = useState<string | null>(null)
  const [copiedTurnId, setCopiedTurnId] = useState<string | null>(null)
  const copiedTurnTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
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
  const [preferredWorkspaceModes, setWorkspaceModes] =
    useState<WorkspaceModesPreference>(DEFAULT_WORKSPACE_MODES)
  const [daemonClient, setDaemonClient] =
    useState<Extract<Awaited<ReturnType<typeof verifyDaemonConnection>>, { status: 'healthy' }>['daemon']['client']>()
  const workspaceModes = useMemo(() => {
    // Computer Use is unavailable on Windows in this beta, whatever the
    // daemon-wide preference says (it may have been enabled from another host).
    const preferred =
      window.brazier.platform === 'win32'
        ? { ...preferredWorkspaceModes, computer: false }
        : preferredWorkspaceModes
    if (!daemonClient || daemonClient.owner) return preferred
    const inference = daemonClient.scopes.includes('inference')
    const agent = daemonClient.scopes.includes('agent')
    return {
      chat: preferred.chat && inference,
      generate: preferred.generate && inference,
      voice: preferred.voice && inference,
      agent: preferred.agent && agent,
      computer: preferred.computer && agent
    }
  }, [daemonClient, preferredWorkspaceModes])
  const [realtimeVoiceAvailable, setRealtimeVoiceAvailable] = useState(false)
  // Generate, Voice, and Computer pick from their own model families; the top
  // bar shows whichever belongs to the mode on screen.
  const [voiceModel, setVoiceModel] = useState('')
  const [generateModel, setGenerateModel] = useState('')
  const [computerModel, setComputerModel] = useState('')
  const [generateModality, setGenerateModality] = useState<'image' | 'video'>('image')
  const [generateHistory, setGenerateHistory] = useState<GenerateHistoryEntry[]>(() =>
    readGenerateHistory(initialProfileId)
  )
  const [activeGenerateHistoryId, setActiveGenerateHistoryId] = useState<string | null>(null)
  const [persona, setPersona] = useState('You are a helpful assistant.')
  const personaEdited = useRef(false)
  // Chat memory: what the model may recall and write, and how dreaming runs.
  const [memoryPreference, setMemoryPreference] = useState<MemoryPreference | null>(null)
  // Incognito chat sessions are ephemeral and memory-free. They keep a
  // synthetic conversation id so the message-loading effects treat them like a
  // conversation, but every write is intercepted before it reaches the daemon.
  const [incognito, setIncognito] = useState(false)
  const incognitoRef = useRef(incognito)
  incognitoRef.current = incognito
  const INCOGNITO_ID = 'incognito'
  // In-app confirmation before discarding a non-empty incognito session. A
  // native `window.confirm` blocks the renderer main loop and trips Chromium's
  // GLib signal handling on Linux, so destructive actions use an inline banner
  // instead.
  const [incognitoDiscardOpen, setIncognitoDiscardOpen] = useState(false)
  // Agent / Computer modes have no composer of their own; they publish these
  // so the one at the bottom of the window can drive them.
  const [agentComposer, setAgentComposer] = useState<AgentComposerControls | null>(null)
  /** Set by Ctrl/Cmd+Enter or the Interrupt button for the next agent submit. */
  const interruptNextSubmitRef = useRef(false)
  const composerFormRef = useRef<HTMLFormElement>(null)
  const [computerComposer, setComputerComposer] = useState<AgentComposerControls | null>(null)
  const [agentSidebar, setAgentSidebar] = useState<AgentSidebarControls | null>(null)
  const [computerSidebar, setComputerSidebar] = useState<ComputerSidebarControls | null>(null)
  const profileId = connectionProfile?.id ?? initialProfileId
  const profileIdRef = useRef(profileId)
  profileIdRef.current = profileId
  const conversationIdRef = useRef(conversationId)
  conversationIdRef.current = conversationId
  const conversationsRef = useRef(conversations)
  conversationsRef.current = conversations
  const conversationSearchRef = useRef(conversationSearch)
  conversationSearchRef.current = conversationSearch
  const selectedModelRef = useRef(selectedModel)
  selectedModelRef.current = selectedModel
  const busyRef = useRef(busy)
  busyRef.current = busy
  const lastMessageIdRef = useRef<string | undefined>(undefined)
  lastMessageIdRef.current = messages.at(-1)?.id
  const memoryPreferenceRef = useRef(memoryPreference)
  memoryPreferenceRef.current = memoryPreference
  const skipNextConversationLoadRef = useRef<string | null>(null)
  const creatingConversationRef = useRef(false)
  const pendingIncognitoActionRef = useRef<(() => void) | null>(null)
  const renameCancelledRef = useRef(false)
  const titleAbortControllersRef = useRef<Map<string, AbortController>>(new Map())
  const chatScrollContainerRef = useRef<HTMLDivElement>(null)
  const stickToBottomRef = useRef(true)
  const modelOpIdRef = useRef(0)
  const userUnloadedRef = useRef(false)

  useEffect(() => {
    return () => {
      if (copiedTurnTimerRef.current) clearTimeout(copiedTurnTimerRef.current)
    }
  }, [])

  useEffect(() => {
    let active = true
    const apply = (profile: ConnectionProfileSummary): void => {
      if (!active) return
      if (profile.id !== profileIdRef.current) {
        connectionCheckRef.current += 1
        setConnectionProblem(null)
        setDaemonClient(undefined)
        profileIdRef.current = profile.id
        userUnloadedRef.current = false
        prepareAbortRef.current?.abort()
        abortRef.current?.abort()
        abortRef.current = undefined
        chatTurnRef.current += 1
        messageRefreshRef.current += 1
        conversationRefreshRef.current += 1
        for (const controller of titleAbortControllersRef.current.values()) {
          controller.abort()
        }
        titleAbortControllersRef.current.clear()
        const cachedModels = readCachedModels(profile.id)
        setConversations(readCachedConversations(profile.id))
        setLocalModels(cachedModels)
        setModelsLoading(cachedModels.length === 0)
        const cachedRuntimes = readCachedRuntimes(profile.id)
        setPrefetchedRuntimes(cachedRuntimes.length > 0 ? cachedRuntimes : null)
        setEnabledTools(readEnabledTools(profile.id))
        setGenerateHistory(readGenerateHistory(profile.id))
        setSelectedModel('')
        setModelResidency(null)
        setModelPrepareState('idle')
        setModelProfiles({})
        setRuntime(null)
        setAttachments([])
        setBusy(false)
        setStreamingText('')
        setStreamingReasoning('')
        setStreamingTools([])
        setStreamingToolOffsets([])
        conversationIdRef.current = null
        setConversationId(null)
        setMessages([])
        setTipId(null)
      }
      setConnectionProfile(profile)
    }
    void fetchCurrentConnectionProfile().then(apply).catch(() => {
      updateDaemonConnection({ type: 'unreachable' })
    })
    const removeListener = window.brazier.onConnectionProfileChanged(apply)
    return () => {
      active = false
      removeListener()
    }
    // The main process reloads after a switch; this effect owns only this
    // mount's initial profile resolution and pre-reload cache isolation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

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
  const abortChatGeneration = useCallback((): void => {
    chatTurnRef.current += 1
    abortRef.current?.abort()
    abortRef.current = undefined
    setBusy(false)
    setStreamingText('')
    setStreamingReasoning('')
    setStreamingTools([])
    setStreamingToolOffsets([])
    setGenerationRate(null)
    setGenerationTokens(null)
    setModelLoadStatus(null)
  }, [])
  const prepareAbortRef = useRef<AbortController | undefined>(undefined)
  const preparingModelRef = useRef('')
  const conversationRefreshRef = useRef(0)
  // Message requests can outlive a cancelled generation or a newly created
  // chat. Keep a generation counter separate from the conversation list so a
  // late response for the old chat cannot repaint the new one.
  const messageRefreshRef = useRef(0)
  const fileInput = useRef<HTMLInputElement>(null)
  const importInput = useRef<HTMLInputElement>(null)
  const scrollAnchor = useRef<HTMLDivElement>(null)
  const dreamAbortRef = useRef<AbortController | undefined>(undefined)
  const dreamTimerRef = useRef<number | undefined>(undefined)
  const dreamDismissRef = useRef<number | undefined>(undefined)
  const lastDreamAtRef = useRef(0)
  const [dreamPromptOpen, setDreamPromptOpen] = useState(false)
  const [dreamStatus, setDreamStatus] = useState<string | null>(null)

  const chain = useMemo(() => messageChain(messages, tipId), [messages, tipId])
  const chatDisplayItems = useMemo(() => buildChatDisplayItems(chain), [chain])
  // `modelPrepareState === 'idle'` with a seeded `selectedModel` represents
  // `selected_lazy`: the selection is ready to chat immediately, and the daemon
  // JIT-loads weights on the first message without blocking launch VRAM.
  const canChat =
    Boolean(selectedModel) &&
    (modelPrepareState === 'ready' || modelPrepareState === 'idle') &&
    !modelUnloading
  const effectiveLocalModels = useMemo(
    () => localModels.map((model) => withMergedCapabilities(model, modelProfiles[model.id])),
    [localModels, modelProfiles]
  )
  const selectedCapabilities = effectiveLocalModels.find(
    (model) => model.id === selectedModel
  )?.capabilities
  const chatModels = useMemo(
    () => effectiveLocalModels.filter((model) => isChatModel(model)),
    [effectiveLocalModels]
  )
  const voiceModels = useMemo(
    () => effectiveLocalModels.filter((model) => isVoiceModel(model)),
    [effectiveLocalModels]
  )
  const computerModels = useMemo(
    () =>
      sortComputerUseModels(
        effectiveLocalModels.filter((model) => isComputerUseModel(model))
      ),
    [effectiveLocalModels]
  )
  const generateModels = useMemo(
    () =>
      effectiveLocalModels.filter((model) =>
        generateModality === 'image' ? isImageGenModel(model) : isVideoGenModel(model)
      ),
    [effectiveLocalModels, generateModality]
  )
  const [pipelineFeatures, setPipelineFeatures] = useState<PipelineFeatures>({
    asr: false,
    video_preprocess: false
  })
  const runtimeWarning = useMemo(
    () =>
      runtimeNoticeForModel(
        selectedModel,
        effectiveLocalModels,
        prefetchedRuntimes,
        modelBindings
      ),
    [selectedModel, effectiveLocalModels, prefetchedRuntimes, modelBindings]
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

  const updateEnabledTools = useCallback(
    (next: string[]): void => {
      setEnabledTools(next)
      writeEnabledTools(profileId, next)
    },
    [profileId]
  )

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
        asr: Boolean(audio?.batch_asr?.available),
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
        if (filtered.length !== current.length) writeEnabledTools(profileId, filtered)
        return filtered
      })
    } catch {
      // Non-fatal: the tools popover simply shows what it last loaded.
    }
  }

  const refreshLocalModels = useCallback(async (): Promise<void> => {
    const activeProfile = profileIdRef.current
    const models = await listModels()
    if (profileIdRef.current !== activeProfile) return
    setLocalModels(models)
    writeCachedModels(activeProfile, models)
    setSelectedModel((current) => {
      if (current && models.some((model) => model.id === current)) return current
      return ''
    })
  }, [])

  const selectModel = useCallback((modelId: string, mode?: ModelLoadMode): void => {
    if (modelId.startsWith('whisper:') || modelId.startsWith('streaming-asr:')) {
      setError('ASR models are used for transcription, not chat. Pick a chat model instead.')
      return
    }
    const opId = ++modelOpIdRef.current
    prepareAbortRef.current?.abort()
    preparingModelRef.current = modelId
    setModelUnloading(false)
    if (modelId) userUnloadedRef.current = false
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
        if (!controller.signal.aborted && opId === modelOpIdRef.current) {
          setModelLoadStatus(event.message)
        }
      }
    })
      .then((residency) => {
        if (
          controller.signal.aborted ||
          opId !== modelOpIdRef.current ||
          preparingModelRef.current !== modelId
        ) {
          return
        }
        setModelResidency(residency)
        setModelPrepareState('ready')
        setModelLoadStatus(null)
        void prefetchRuntimes()
        void fetchModelBindings().then(setModelBindings).catch(reportBackgroundFailure)
        void refreshCapabilities()
      })
      .catch((cause: unknown) => {
        if (
          controller.signal.aborted ||
          opId !== modelOpIdRef.current ||
          preparingModelRef.current !== modelId ||
          (cause as Error).name === 'AbortError'
        ) {
          return
        }
        // Selection happens before the daemon can load the weights. If that
        // load fails, do not leave a model highlighted in the top bar as if it
        // were active. Keep the error visible so the user can diagnose it.
        preparingModelRef.current = ''
        setSelectedModel('')
        setModelResidency(null)
        setModelPrepareState('error')
        setModelLoadStatus(null)
        if (cause instanceof GenerationFailure) {
          setError(cause.message)
          setForkHints(cause.forkHints)
        } else {
          setError(cause instanceof Error ? cause.message : String(cause))
        }
      })
  }, [appMode, refreshCapabilities])

  const switchAppMode = useCallback(
    (next: AppMode, options?: { skipModelPrepare?: boolean }): void => {
      if (next === appMode) return
      const draftKey = (mode: AppMode): 'chat' | 'agent' | 'computer' =>
        mode === 'agent' || mode === 'computer' ? mode : 'chat'
      draftsRef.current[draftKey(appMode)] = draft
      setDraft(draftsRef.current[draftKey(next)] ?? '')
      if (next !== 'chat') setAttachments([])
      if (appMode === 'chat' && busy && next !== 'chat') abortChatGeneration()
      setAppMode(next)
      const currentModel = selectedModelRef.current
      if (!options?.skipModelPrepare && (next === 'chat' || next === 'agent') && currentModel) {
        if (busy) abortChatGeneration()
        selectModel(currentModel, next)
      }
      // Chat stays mounted while other modes are active; re-read the open
      // conversation when returning so the next completion uses the same
      // history shown on screen (mirrors agent session rehydrate).
      if (next === 'chat' && conversationIdRef.current) {
        void refreshMessages(conversationIdRef.current).catch(() => undefined)
      }
    },
    [appMode, selectModel, draft, busy, abortChatGeneration]
  )

  const unloadSelectedModel = useCallback(async (): Promise<void> => {
    if (!selectedModel || modelUnloading) return
    const opId = ++modelOpIdRef.current
    userUnloadedRef.current = true
    prepareAbortRef.current?.abort()
    preparingModelRef.current = ''
    setModelUnloading(true)
    setModelPrepareState('loading')
    setModelLoadStatus('Unloading model…')
    setError(null)
    try {
      await unloadModel()
      if (opId !== modelOpIdRef.current) return
      setSelectedModel('')
      setModelResidency(null)
      setModelPrepareState('idle')
      setForkHints([])
      void prefetchRuntimes()
    } catch (cause) {
      if (opId !== modelOpIdRef.current) return
      setModelPrepareState('ready')
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      if (opId === modelOpIdRef.current) {
        setModelUnloading(false)
        setModelLoadStatus(null)
      }
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
        emptySubtitle: 'None installed'
      }
    }
    if (appMode === 'generate') {
      return {
        models: generateModels,
        selected: generateModel,
        select: setGenerateModel,
        emptyTitle: `No ${generateModality} model`,
        emptySubtitle: 'None installed'
      }
    }
    if (appMode === 'computer') {
      return {
        models: computerModels,
        selected: computerModel,
        select: setComputerModel,
        emptyTitle: 'No computer-use model',
        emptySubtitle: 'None installed'
      }
    }
    return {
      models: chatModels,
      selected: selectedModel,
      select: selectModel,
      emptyTitle: 'Choose a model',
      emptySubtitle: 'None loaded'
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
    if (modelsLoading && effectiveLocalModels.length === 0) {
      return { title: 'Loading models…', subtitle: 'Reading your library' }
    }
    if (!modeModel.selected) {
      return { title: modeModel.emptyTitle, subtitle: modeModel.emptySubtitle }
    }
    return modelDisplayName(
      modeModel.selected,
      effectiveLocalModels.find((model) => model.id === modeModel.selected)
    )
  }, [modeModel, effectiveLocalModels, modelsLoading])

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
      async respond(turn: {
        correlationId: string
        text: string
        userMessageId?: string
        onPartial?: (delta: string) => void
      }): Promise<{ text: string }> {
        const { correlationId, text, userMessageId, onPartial } = turn
        if (!selectedModel) {
          throw new Error('Select a chat model before asking a question by voice.')
        }
        const controller = new AbortController()
        controllers.set(correlationId, controller)
        // Fresh turns run before React re-renders the stored user message into
        // `chainRef.current`; queued turns run later when `chainRef.current`
        // already contains the stored user message (`userMessageId` or
        // `correlation_id`). Deduplicate so queued turns never send the user
        // prompt twice.
        const existingIdx = chainRef.current.findIndex(
          (entry) =>
            (userMessageId != null && entry.id === userMessageId) ||
            (entry.role === 'user' && entry.correlation_id === correlationId)
        )
        const asked: Message =
          existingIdx >= 0
            ? chainRef.current[existingIdx]
            : {
                id: userMessageId ?? `pending-${correlationId}`,
                conversation_id: conversationId ?? '',
                parent_id: chainRef.current.at(-1)?.id ?? null,
                role: 'user',
                content: text,
                model: null,
                created_at: new Date().toISOString()
              }
        try {
          const memoryEnabled = (memoryPreference?.enabled ?? true) && !incognito
          let requestMessages = await (async (): Promise<Message[]> => {
            const base =
              existingIdx >= 0
                ? chainRef.current.slice(0, existingIdx + 1)
                : [...chainRef.current, asked]
            if (!memoryEnabled || !memoryPreference || !text) return base
            const memories = await loadRelevantMemories(text, memoryPreference.recall_count)
            const context = buildMemoryContext(memories, memoryPreference.recall_chars)
            if (!context) return base
            return [
              {
                id: `memory-${crypto.randomUUID()}`,
                conversation_id: conversationId ?? INCOGNITO_ID,
                parent_id: null,
                role: 'system',
                content: context,
                model: null,
                created_at: new Date().toISOString()
              },
              ...base
            ]
          })()
          let responseText = ''
          const maxClientRounds = 4
          for (let round = 0; round < maxClientRounds; round += 1) {
            const result = await streamCompletion(
              requestMessages,
              selectedModel,
              controller.signal,
              (token) => onPartial?.(token),
              {
                builtinTools: toolsEnabled,
                builtinToolNames: toolsEnabled ? enabledTools : undefined,
                extraTools: memoryEnabled ? memoryToolDefinitions() : undefined,
                toolChoice: toolsEnabled || memoryEnabled ? 'auto' : undefined
              }
            )
            setModelPrepareState((current) => (current === 'idle' ? 'ready' : current))
            responseText = result.responseText
            for (const entry of result.transcript) {
              requestMessages = [
                ...requestMessages,
                {
                  id: `voice-step-${crypto.randomUUID()}`,
                  conversation_id: conversationId ?? INCOGNITO_ID,
                  parent_id: requestMessages.at(-1)?.id ?? null,
                  role: entry.role as Role,
                  content: entry.content ?? '',
                  tool_calls: entry.tool_calls ?? null,
                  tool_call_id: entry.tool_call_id ?? null,
                  model: entry.role === 'assistant' ? selectedModel : null,
                  created_at: new Date().toISOString()
                }
              ]
            }
            if (result.clientToolCalls.length === 0) break
            let allClientHandled = true
            for (const call of result.clientToolCalls) {
              if (isMemoryClientTool(call.name)) {
                const outcome = incognito
                  ? {
                      output: 'Memory is disabled in incognito chats; nothing was saved or read.',
                      is_error: true
                    }
                  : await executeMemoryClientTool(call, {
                      conversation_id: conversationId ?? undefined,
                      message_id: asked.id
                    })
                requestMessages = [
                  ...requestMessages,
                  {
                    id: `voice-tool-${crypto.randomUUID()}`,
                    conversation_id: conversationId ?? INCOGNITO_ID,
                    parent_id: requestMessages.at(-1)?.id ?? null,
                    role: 'tool',
                    content: outcome.output,
                    tool_call_id: call.id,
                    source: 'tool',
                    model: null,
                    created_at: new Date().toISOString()
                  }
                ]
              } else {
                allClientHandled = false
              }
            }
            if (allClientHandled) continue
            throw new GenerationFailure(
              `The model requested client-side tools that Brazier cannot execute: ${result.clientToolCalls.map((call: ClientToolCall) => call.name).join(', ')}`
            )
          }
          return { text: responseText }
        } finally {
          controllers.delete(correlationId)
        }
      },
      cancel(correlationId: string): void {
        controllers.get(correlationId)?.abort()
      }
    }
  }, [conversationId, selectedModel, toolsEnabled, enabledTools, memoryPreference, incognito])

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
    incognito,
    responder: voiceResponder,
    onMessage: (message) => {
      if (message.conversation_id !== conversationIdRef.current) return
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
  const coordinatorWorking = session.snapshot.activeCorrelationId !== null
  const streamingOwner = session.snapshot.responses.find(
    (entry) => entry.correlationId === session.snapshot.activeCorrelationId
  )?.owner
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

  // Chat and agent share one model, so the chat default seeds both. Only the
  // selection changes; the model itself loads on the next message.
  useEffect(() => {
    setSelectedModel((current) => {
      if (current && chatModels.some((model) => model.id === current)) return current
      if (userUnloadedRef.current) return ''
      const preferred = runtime?.default_chat_model
      if (preferred && chatModels.some((model) => model.id === preferred)) return preferred
      return ''
    })
  }, [chatModels, runtime?.default_chat_model])

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
    const activeProfile = profileIdRef.current
    try {
      const response = await listRuntimes()
      if (profileIdRef.current !== activeProfile) return
      setPrefetchedRuntimes(response.data)
      writeCachedRuntimes(activeProfile, response.data)
    } catch {
      // Best-effort prefetch — Manage panel falls back to its own fetch.
    }
  }

  async function refreshConversations(query = conversationSearchRef.current, background = false): Promise<void> {
    const activeProfile = profileIdRef.current
    const refreshId = ++conversationRefreshRef.current
    const selected = conversationIdRef.current
    const turn = chatTurnRef.current
    const data = await listConversations(query)
    // Search requests can resolve out of order while the user types. Only the
    // newest query for the current profile is allowed to replace results.
    if (refreshId !== conversationRefreshRef.current || profileIdRef.current !== activeProfile) {
      return
    }
    setConversations(data)
    // Only an unfiltered list is worth caching for the next cold start.
    if (!query.trim()) writeCachedConversations(activeProfile, data)
    if (background && selected && !incognitoRef.current &&
        await selectedConversationWasRemoved(selected, data, query, () => listConversations(''))) {
      if (refreshId !== conversationRefreshRef.current || profileIdRef.current !== activeProfile ||
          conversationIdRef.current !== selected || turn !== chatTurnRef.current ||
          busyRef.current || incognitoRef.current) return
      messageRefreshRef.current += 1
      conversationIdRef.current = null
      setConversationId(null)
      setMessages([])
      setTipId(null)
    } else if (!background && !conversationIdRef.current && !incognitoRef.current && data[0]) {
      conversationIdRef.current = data[0].id
      setConversationId(data[0].id)
    }
  }

  async function exportConversationFile(id: string): Promise<void> {
    setConversationMenuId(null)
    setError(null)
    try {
      const bundle = await exportConversation(id)
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
      setConversationSearch('')
      if (incognito) setIncognito(false)
      setConversationId(conversation.id)
      await refreshConversations('')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  async function refreshMessages(id: string, preferredTip?: string, background = false): Promise<void> {
    if (id === INCOGNITO_ID) return
    const refreshId = ++messageRefreshRef.current
    const turn = chatTurnRef.current
    const data = await listMessages(id)
    if (refreshId !== messageRefreshRef.current || id !== conversationIdRef.current) return
    if (background && (busyRef.current || incognitoRef.current || turn !== chatTurnRef.current)) return
    const previousLast = lastMessageIdRef.current
    setMessages(data)
    setTipId((current) => background
      ? refreshedMessageTip(data, current, previousLast)
      : preferredTip ?? data.at(-1)?.id ?? null)
  }

  async function refreshRuntime(): Promise<void> {
    const { daemon } = await connection()
    if (daemon.client && !daemon.client.owner && !daemon.client.scopes.includes('management')) return
    const status = await engineStatus()
    setRuntime(status.settings)
    setHardware(status.hardware)
    setDaemonDataDir(dataDirFromManagedBinaryPath(status.managed_binary_path))
  }

  async function rehydrateDaemonState(): Promise<void> {
    await Promise.allSettled([
      refreshConversations(''),
      loadLocalModels(),
      prefetchRuntimes(),
      refreshTools(),
      fetchModelBindings().then(setModelBindings),
      fetchModelSettings().then((response) => setModelProfiles(response.models)),
      refreshRuntime(),
      refreshCapabilities(),
      fetchMemoryPreference().then(setMemoryPreference)
    ])
  }

  useEffect(() => {
    if (daemonConnection.phase !== 'rehydrating') return
    setError(null)
    void rehydrateDaemonState().finally(() =>
      updateDaemonConnection({ type: 'rehydrated' })
    )
    // Runs once per entry into `rehydrating`; the loaders read current state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [daemonConnection.phase])

  useEffect(() => {
    let cancelled = false
    void Promise.all([window.brazier.getFlags(), hasCompletedWelcome()])
      .then(([flags, completed]) => {
        if (cancelled) return
        setShowWelcome(Boolean(flags.forceWelcome) || !completed)
      })
      .catch(() => {
        // An offline selected profile must still reach the shell so the user
        // can retry or open Connections. Cached renderer data remains visible.
        if (!cancelled) setShowWelcome(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!connectionProfile) return
    void refreshConversations().catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
    void loadLocalModels().catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
    void prefetchRuntimes()
    void refreshTools()
    void fetchModelBindings().then(setModelBindings).catch(reportBackgroundFailure)
    // Advanced per-model settings, so the picker can say which models carry any
    // before one of them is opened.
    void fetchModelSettings()
      .then((response) => setModelProfiles(response.models))
      .catch(reportBackgroundFailure)
    void refreshRuntime().catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
    void refreshCapabilities()
    void fetchMemoryPreference()
      .then(setMemoryPreference)
      .catch(() => {
        // Memory defaults apply until the preference loads.
      })
  }, [connectionProfile?.id])

  useEffect(() => {
    if (!connectionProfile || daemonStatus === 'offline') return
    const timer = window.setTimeout(() => {
      void refreshConversations(conversationSearch).catch((cause: unknown) =>
        setError(cause instanceof Error ? cause.message : String(cause))
      )
    }, 250)
    return () => window.clearTimeout(timer)
  }, [conversationSearch, connectionProfile?.id, daemonStatus])

  async function reconnectDaemon(manual = true): Promise<void> {
    const generation = ++connectionCheckRef.current
    if (manual) updateDaemonConnection({ type: 'retry' })
    const result = await verifyDaemonConnection().catch(() => ({
      status: 'offline' as const,
      message: 'Could not reach the selected daemon.'
    }))
    if (generation !== connectionCheckRef.current) return
    if (result.status !== 'healthy') {
      updateDaemonConnection({ type: 'unreachable', blocked: result.status !== 'offline' })
      setConnectionProblem(result.status === 'offline' ? null : result.message)
      return
    }
    cacheVerifiedConnection(result)
    setDaemonClient(result.daemon.client)
    setConnectionProblem(null)
    // The reducer decides whether this reconnect needs a rehydrate.
    updateDaemonConnection({ type: 'reachable', version: result.daemon.version })
  }


  useEffect(() => {
    if (!connectionProfile) return
    const check = (): void => { void reconnectDaemon(false) }
    check()
    const timer = window.setInterval(check, 10_000)
    return () => {
      connectionCheckRef.current += 1
      window.clearInterval(timer)
    }
  }, [connectionProfile?.id])

  useEffect(() => {
    if (!connectionProfile || daemonStatus !== 'healthy') return
    return startConversationSync(window, document, () => !busyRef.current && !incognitoRef.current, () => {
      const id = conversationIdRef.current
      return Promise.allSettled([
        refreshConversations(conversationSearchRef.current, true),
        id ? refreshMessages(id, undefined, true) : Promise.resolve()
      ])
    })
  }, [connectionProfile?.id, daemonStatus])

  useEffect(() => {
    // A selected chat changed; any fetch started for the previous one is now
    // stale, even if it completes after this effect has begun loading the new
    // conversation.
    messageRefreshRef.current += 1
    const loadId = messageRefreshRef.current
    stickToBottomRef.current = true
    if (incognito) {
      // The ephemeral session lives entirely in renderer state: no daemon
      // conversation to load, no runs, and any stale fetch is abandoned.
      return
    }
    if (!conversationId) {
      setMessages([])
      setTipId(null)
      setRunSnapshots([])
      setExpandedRunId(null)
      return
    }
    if (skipNextConversationLoadRef.current === conversationId) {
      // `submit()` just created this conversation and owns its first-turn load;
      // avoid racing with `persistChatMessage` and wiping optimistic state.
      skipNextConversationLoadRef.current = null
      return
    }
    setError(null)
    refreshMessages(conversationId).catch((cause: unknown) => {
      if (messageRefreshRef.current === loadId) {
        setError(cause instanceof Error ? cause.message : String(cause))
      }
    })
    void listRunSnapshots(conversationId)
      .then((snapshots) => {
        if (messageRefreshRef.current === loadId && conversationIdRef.current === conversationId) {
          setRunSnapshots(snapshots)
        }
      })
      .catch(() => {
        if (messageRefreshRef.current === loadId && conversationIdRef.current === conversationId) {
          setRunSnapshots([])
        }
      })
  }, [conversationId, incognito])

  useEffect(() => {
    if (!stickToBottomRef.current) return
    scrollAnchor.current?.scrollIntoView({ behavior: liveText ? 'auto' : 'smooth' })
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
    if ((appMode !== 'voice' && appMode !== 'agent') || conversationId) return
    void openUntouchedConversation().catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
  }, [appMode, conversationId])

  /** Reuses a never-used chat, so visiting Voice or Agent doesn't pile up empty ones. */
  async function openUntouchedConversation(): Promise<void> {
    const candidate = conversations.find(
      (conversation) =>
        conversation.title === 'New conversation' &&
        conversation.created_at === conversation.updated_at &&
        !conversation.agent_session_id &&
        !conversation.incognito
    )
    if (candidate && (await listMessages(candidate.id)).length === 0) {
      if (incognito) setIncognito(false)
      setConversationId(candidate.id)
      return
    }
    await newConversation()
  }

  // Unmounting VoiceMode must actually stop the microphone. The coordinator
  // lives on App, so leaving the tab is the teardown, not App unmount.
  useEffect(() => {
    if (appMode === 'voice') return
    if (session.snapshot.voiceStatus === 'off' || session.snapshot.voiceStatus === 'error') return
    void session.endVoice()
  }, [appMode, session.endVoice, session.snapshot.voiceStatus])

  function confirmOrLeaveIncognito(proceed: () => void): void {
    if (incognitoRef.current && messages.length > 0) {
      pendingIncognitoActionRef.current = proceed
      setIncognitoDiscardOpen(true)
      return
    }
    proceed()
  }

  async function newConversation(options?: { bypassIncognitoConfirm?: boolean }): Promise<void> {
    if (!options?.bypassIncognitoConfirm && incognitoRef.current && messages.length > 0) {
      pendingIncognitoActionRef.current = () => {
        void newConversation({ bypassIncognitoConfirm: true })
      }
      setIncognitoDiscardOpen(true)
      return
    }
    if (creatingConversationRef.current) return
    creatingConversationRef.current = true
    try {
      abortChatGeneration()
      if (incognitoRef.current) {
        incognitoRef.current = false
        setIncognito(false)
      }
      setIncognitoDiscardOpen(false)
      pendingIncognitoActionRef.current = null
      const conversation = await createConversation()
      // Invalidate in-flight loads for the previous chat before clearing its
      // display. This matters most after Stop, when a final fetch may still be
      // resolving in the background.
      messageRefreshRef.current += 1
      conversationIdRef.current = conversation.id
      setConversationId(conversation.id)
      setMessages([])
      setTipId(null)
      setDraft('')
      setAttachments([])
      await refreshConversations()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      creatingConversationRef.current = false
    }
  }

  /** Begin a private session: no persistence, no memory, discarded on leave. */
  function startIncognito(): void {
    if (busy || dreamAbortRef.current) return
    incognitoRef.current = true
    setIncognito(true)
    setIncognitoDiscardOpen(false)
    pendingIncognitoActionRef.current = null
    messageRefreshRef.current += 1
    conversationIdRef.current = INCOGNITO_ID
    setConversationId(INCOGNITO_ID)
    setMessages([])
    setTipId(null)
    setRunSnapshots([])
    setExpandedRunId(null)
    setDraft('')
    setAttachments([])
    setError(null)
  }

  /** Leave incognito, discarding the ephemeral conversation. */
  function leaveIncognito(): void {
    if (busy) return
    incognitoRef.current = false
    setIncognito(false)
    setIncognitoDiscardOpen(false)
    pendingIncognitoActionRef.current = null
    messageRefreshRef.current += 1
    conversationIdRef.current = null
    setConversationId(null)
    setMessages([])
    setTipId(null)
    setRunSnapshots([])
    setExpandedRunId(null)
    setDraft('')
    setAttachments([])
    void refreshConversations().catch(() => undefined)
  }

  function toggleIncognito(): void {
    if (incognito) {
      // Discarding a non-empty session is destructive; arm the inline confirm
      // rather than popping a native dialog.
      if (messages.length > 0) {
        pendingIncognitoActionRef.current = null
        setIncognitoDiscardOpen(true)
        return
      }
      leaveIncognito()
      return
    }
    // Starting incognito loses nothing: the current conversation stays in the
    // sidebar untouched, so no confirmation is needed.
    startIncognito()
  }

  async function runDream(): Promise<void> {
    setDreamPromptOpen(false)
    if (dreamAbortRef.current) return
    const dreamModel = selectedModelRef.current
    if (!dreamModel) {
      setError('Select a chat model before running dreaming.')
      return
    }
    // A stale dismiss timer from a previous pass must not clear this one.
    if (dreamDismissRef.current !== undefined) {
      window.clearTimeout(dreamDismissRef.current)
      dreamDismissRef.current = undefined
    }
    setDreamStatus('Dreaming…')
    const controller = new AbortController()
    dreamAbortRef.current = controller
    try {
      const conversations: DreamInputConversation[] = (await listConversations(''))
        .slice(0, 10)
        .map((conversation) => ({
          id: conversation.id,
          title: conversation.title,
          summary: conversation.summary ?? null,
          updated_at: conversation.updated_at
        }))
      const memories = await listMemories()
      const result = await dream({
        model: dreamModel,
        signal: controller.signal,
        memories,
        conversations,
        prompt: memoryPreferenceRef.current?.dream_prompt ?? undefined,
        onLoad: (event) => setDreamStatus(dreamStatusForLoad(event))
      })
      markDreamed()
      const changed = result.created + result.updated + result.deleted
      setDreamStatus(
        changed === 0
          ? 'Dreaming found nothing to change.'
          : `Dreaming done: ${result.created} added, ${result.updated} updated, ${result.deleted} removed.`
      )
      // The status is transient: a finished pass should not sit in the notice
      // bar until dismissed, especially one that ran silently in auto mode.
      scheduleDreamStatusDismiss()
    } catch (cause) {
      if ((cause as Error).name !== 'AbortError') {
        setError(cause instanceof Error ? cause.message : String(cause))
      }
      scheduleDreamStatusDismiss()
    } finally {
      dreamAbortRef.current = undefined
    }
  }
  const runDreamRef = useRef(runDream)
  runDreamRef.current = runDream

  /** Clear the dreaming notice a few seconds after a pass ends. */
  function scheduleDreamStatusDismiss(): void {
    if (dreamDismissRef.current !== undefined) window.clearTimeout(dreamDismissRef.current)
    dreamDismissRef.current = window.setTimeout(() => {
      dreamDismissRef.current = undefined
      setDreamStatus(null)
    }, DREAM_STATUS_DISMISS_MS)
  }

  /** Record that a dreaming pass ran (or was declined) so the configured
   * interval is honored even after the app restarts. */
  function markDreamed(): void {
    const now = Date.now()
    lastDreamAtRef.current = now
    const latest = memoryPreferenceRef.current
    setMemoryPreference((current) => (current ? { ...current, last_dream_at: now } : current))
    if (latest) {
      const next = { ...latest, last_dream_at: now }
      memoryPreferenceRef.current = next
      void saveMemoryPreference(next).catch(reportBackgroundFailure)
    }
  }

  /**
   * After a chat turn, schedule a dreaming pass when the app stays idle. Auto
   * runs it silently; ask surfaces a prompt first. Incognito and conversations
   * that are still active never dream, and a pass runs at most once per the
   * configured interval (default weekly).
   */
  function scheduleDream(): void {
    const pref = memoryPreferenceRef.current
    const mode = pref?.dreaming
    if (!mode || mode === 'off' || incognitoRef.current || !selectedModelRef.current) return
    const intervalMs =
      (pref?.dream_interval_days ?? DEFAULT_MEMORY_PREFERENCE.dream_interval_days) *
      24 *
      60 *
      60 *
      1000
    const lastDreamAt = pref?.last_dream_at ?? lastDreamAtRef.current
    // Don't re-dream within the interval, or while the user is actively using
    // the app (a new submit clears this timer).
    if (Date.now() - lastDreamAt < intervalMs) return
    if (dreamTimerRef.current !== undefined) window.clearTimeout(dreamTimerRef.current)
    dreamTimerRef.current = window.setTimeout(() => {
      dreamTimerRef.current = undefined
      const livePref = memoryPreferenceRef.current
      const liveMode = livePref?.dreaming
      if (
        busyRef.current ||
        dreamAbortRef.current ||
        incognitoRef.current ||
        !selectedModelRef.current ||
        !liveMode ||
        liveMode === 'off'
      ) {
        return
      }
      if (liveMode === 'auto') {
        void runDreamRef.current()
      } else if (liveMode === 'ask') {
        setDreamPromptOpen(true)
      }
    }, 2 * 60 * 1000)
  }

  function clearDreamTimer(): void {
    if (dreamTimerRef.current !== undefined) {
      window.clearTimeout(dreamTimerRef.current)
      dreamTimerRef.current = undefined
    }
  }

  useEffect(() => () => clearDreamTimer(), [])

  useEffect(() => {
    return () => {
      if (dreamDismissRef.current !== undefined) window.clearTimeout(dreamDismissRef.current)
    }
  }, [])

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
    const modelForTitle = selectedModelRef.current
    if (!modelForTitle) return
    titleAbortControllersRef.current.get(id)?.abort()
    const controller = new AbortController()
    titleAbortControllersRef.current.set(id, controller)
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
    void streamCompletion(titleMessages, modelForTitle, controller.signal, () => {}, {
      toolChoice: 'none'
    })
      .then(({ responseText }) => {
        if (controller.signal.aborted) return
        const title = titleFromCompletion(responseText)
        if (!title) return
        // Do not overwrite if the user manually renamed this conversation while
        // the background title request was in flight.
        const current = conversationsRef.current.find((entry) => entry.id === id)
        if (current && current.title !== 'New conversation') return
        return updateConversation(id, { title }).then(() =>
          refreshConversations(conversationSearchRef.current)
        )
      })
      .catch(() => {
        // Naming is a convenience. A completed chat must not report an error
        // because its optional title request failed.
      })
      .finally(() => {
        if (titleAbortControllersRef.current.get(id) === controller) {
          titleAbortControllersRef.current.delete(id)
        }
      })
  }

  function startRename(conversation: Conversation): void {
    setConversationMenuId(null)
    renameCancelledRef.current = false
    setRenaming({ id: conversation.id, draft: conversation.title })
  }

  async function commitRename(conversation: Conversation, draftTitle: string): Promise<void> {
    if (renameCancelledRef.current) {
      renameCancelledRef.current = false
      return
    }
    setRenaming(null)
    const title = draftTitle.trim()
    if (!title || title === conversation.title) return
    titleAbortControllersRef.current.get(conversation.id)?.abort()
    setConversations((current) =>
      current.map((entry) => (entry.id === conversation.id ? { ...entry, title } : entry))
    )
    try {
      await updateConversation(conversation.id, { title })
      await refreshConversations(conversationSearchRef.current)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
      void refreshConversations(conversationSearchRef.current).catch(() => undefined)
    }
  }

  async function removeConversation(conversation: Conversation): Promise<void> {
    setConversationMenuId(null)
    setConfirmDeleteId(null)
    titleAbortControllersRef.current.get(conversation.id)?.abort()
    titleAbortControllersRef.current.delete(conversation.id)
    const wasSelected = conversation.id === conversationIdRef.current
    if (wasSelected) {
      abortChatGeneration()
      messageRefreshRef.current += 1
      conversationIdRef.current = null
      setConversationId(null)
      setMessages([])
      setTipId(null)
      setAttachments([])
    }
    try {
      await deleteConversation(conversation.id)
      const refreshId = ++conversationRefreshRef.current
      const activeProfile = profileIdRef.current
      const remaining = await listConversations('')
      if (refreshId !== conversationRefreshRef.current || profileIdRef.current !== activeProfile) {
        return
      }
      setConversations(remaining)
      writeCachedConversations(activeProfile, remaining)
      setConversationSearch('')
      if (wasSelected && conversationIdRef.current === null && !incognitoRef.current) {
        const nextId = remaining[0]?.id ?? null
        conversationIdRef.current = nextId
        setConversationId(nextId)
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  async function applyInferenceSettings(
    next: RuntimeSettings,
    options?: { skipReload?: boolean }
  ): Promise<void> {
    setSavingInference(true)
    setError(null)
    try {
      const model = effectiveLocalModels.find((entry) => entry.id === selectedModel)
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
      if (!options?.skipReload && selectedModel && modelPrepareState === 'ready') {
        selectModel(selectedModel)
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
      throw cause
    } finally {
      setSavingInference(false)
    }
  }

  function openManage(section: ManageSection): void {
    setManageSection(section)
    setManageOpen(true)
  }

  function startForkBuild(hint: RuntimeForkHint): void {
    setPendingBuild({
      engine: hint.engine as 'llama.cpp' | 'mlx-lm' | 'mlx-vlm',
      repository: hint.repository
    })
    setManageSection('runtimes')
    setManageOpen(true)
  }

  const selectModelFromAnySurface = useCallback(
    (model: LocalModel): void => {
      if (isImageGenModel(model)) {
        setGenerateModality('image')
        setGenerateModel(model.id)
        if (workspaceModes.generate && appMode !== 'generate') switchAppMode('generate')
        return
      }
      if (isVideoGenModel(model)) {
        setGenerateModality('video')
        setGenerateModel(model.id)
        if (workspaceModes.generate && appMode !== 'generate') switchAppMode('generate')
        return
      }
      if (isVoiceModel(model)) {
        setVoiceModel(model.id)
        if (workspaceModes.voice && appMode !== 'voice') switchAppMode('voice')
        return
      }
      if (appMode === 'computer' && isComputerUseModel(model)) {
        setComputerModel(model.id)
        return
      }
      const targetMode: ModelLoadMode = appMode === 'agent' ? 'agent' : 'chat'
      selectModel(model.id, targetMode)
      if (appMode !== 'chat' && appMode !== 'agent' && workspaceModes.chat) {
        switchAppMode('chat', { skipModelPrepare: true })
      }
    },
    [appMode, selectModel, switchAppMode, workspaceModes]
  )

  const triggerNewForActiveMode = useCallback((): void => {
    if (appMode === 'agent') {
      agentSidebar?.newTask()
      return
    }
    if (appMode === 'computer') {
      computerSidebar?.newSession()
      return
    }
    if (appMode === 'generate') {
      setActiveGenerateHistoryId(null)
      return
    }
    void newConversation()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appMode, agentSidebar, computerSidebar, incognito])

  const openActiveModelSettings = useCallback((): void => {
    if (appMode === 'chat' || appMode === 'agent') {
      setInferenceMenuOpen((open) => !open)
      return
    }
    if (modeModel.selected) {
      setConfiguringModel(modeModel.selected)
      return
    }
    setModelMenuOpen(true)
  }, [appMode, modeModel.selected])

  const copyApiCurl = useCallback(async (): Promise<void> => {
    try {
      const server = await window.brazier
        .getServerSettings()
        .catch(() => ({ port: 11435, apiKeyEnabled: false }))
      const snippet = buildServerCurlSnippet({
        port: server.port,
        modelId: selectedModel || effectiveLocalModels[0]?.id,
        apiKeyEnabled: server.apiKeyEnabled
      })
      await window.brazier.copyText(snippet)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }, [selectedModel, effectiveLocalModels])

  function copyTurnAnswer(turnId: string, answer: string): void {
    if (!answer) return
    void window.brazier.copyText(answer).then(() => {
      setCopiedTurnId(turnId)
      if (copiedTurnTimerRef.current) clearTimeout(copiedTurnTimerRef.current)
      copiedTurnTimerRef.current = setTimeout(() => {
        setCopiedTurnId((current) => (current === turnId ? null : current))
      }, 1600)
    })
  }

  // Global power-user keyboard shortcuts across workspace modes.
  useEffect(() => {
    if (showWelcome !== false) return
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.defaultPrevented) return
      const mod = event.metaKey || event.ctrlKey
      const key = event.key.toLowerCase()

      if (mod && !event.altKey && !event.shiftKey && key === 'k') {
        event.preventDefault()
        setCommandPaletteOpen((open) => !open)
        return
      }
      if (mod && !event.altKey && !event.shiftKey && key === 'n') {
        event.preventDefault()
        triggerNewForActiveMode()
        return
      }
      if (mod && !event.altKey && !event.shiftKey && key === 'b') {
        event.preventDefault()
        setSidebarOpen((open) => !open)
        return
      }
      if (mod && !event.altKey && !event.shiftKey && event.key === ',') {
        event.preventDefault()
        setManageOpen((open) => !open)
        return
      }
      if (mod && !event.altKey && event.shiftKey && key === 'm') {
        event.preventDefault()
        setModelMenuOpen((open) => !open)
        return
      }
      if (mod && !event.altKey && event.shiftKey && key === 'i') {
        if (!workspaceModes.chat) return
        event.preventDefault()
        if (appMode !== 'chat') switchAppMode('chat')
        toggleIncognito()
        return
      }
      if (mod && !event.altKey && !event.shiftKey && event.key === '.') {
        event.preventDefault()
        if (shellComposerMode && shellComposer?.running) {
          void shellComposer.stop()
        } else if (coordinatorWorking) {
          void session.cancelResponse()
        } else if (busy) {
          abortChatGeneration()
        }
        return
      }
      if (mod && !event.altKey && !event.shiftKey && /^[1-5]$/.test(event.key)) {
        const ordered: AppMode[] = ['chat', 'agent', 'generate', 'voice', 'computer']
        const enabled = ordered.filter((mode) => workspaceModes[mode])
        const target = enabled[Number(event.key) - 1]
        if (target) {
          event.preventDefault()
          switchAppMode(target)
        }
        return
      }
      if (event.key === 'Escape' && !mod && !event.altKey && !event.shiftKey) {
        if (commandPaletteOpen) {
          event.preventDefault()
          setCommandPaletteOpen(false)
        } else if (configuringModel) {
          event.preventDefault()
          setConfiguringModel(null)
        } else if (inferenceMenuOpen) {
          event.preventDefault()
          setInferenceMenuOpen(false)
        } else if (modelMenuOpen) {
          event.preventDefault()
          setModelMenuOpen(false)
        } else if (toolsMenuOpen) {
          event.preventDefault()
          setToolsMenuOpen(false)
        } else if (conversationMenuId) {
          event.preventDefault()
          setConversationMenuId(null)
          setConfirmDeleteId(null)
        } else if (manageOpen) {
          event.preventDefault()
          setManageOpen(false)
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    showWelcome,
    appMode,
    workspaceModes,
    shellComposerMode,
    shellComposer,
    coordinatorWorking,
    busy,
    commandPaletteOpen,
    configuringModel,
    inferenceMenuOpen,
    modelMenuOpen,
    toolsMenuOpen,
    conversationMenuId,
    manageOpen,
    incognito,
    messages.length,
    triggerNewForActiveMode,
    switchAppMode,
    abortChatGeneration
  ])

  const suggestionSource = shellComposerMode
    ? shellComposer?.suggestions
    : appMode === 'chat'
      ? CHAT_SLASH_SUGGESTIONS
      : undefined
  const activeSuggestionQuery = suggestionSource ? trailingCommand(draft) : ''
  const visibleSuggestions = useMemo(() => {
    if (!suggestionSource || !activeSuggestionQuery) return []
    if (dismissedSuggestionQuery === activeSuggestionQuery) return []
    const matched = composerCompletionsFor(draft, suggestionSource)
    if (
      matched.length === 0 ||
      (matched.length === 1 && matched[0]?.value.toLowerCase() === activeSuggestionQuery)
    ) {
      return []
    }
    return matched
  }, [suggestionSource, activeSuggestionQuery, dismissedSuggestionQuery, draft])
  const clampedSuggestionIndex = Math.min(
    suggestionIndex,
    Math.max(0, visibleSuggestions.length - 1)
  )

  useEffect(() => {
    setSuggestionIndex(0)
    if (dismissedSuggestionQuery && dismissedSuggestionQuery !== activeSuggestionQuery) {
      setDismissedSuggestionQuery(null)
    }
  }, [activeSuggestionQuery, dismissedSuggestionQuery])

  async function attachFiles(files: File[]): Promise<void> {
    if (files.length === 0) return
    try {
      const accepted = files.filter((file) => {
        if (file.type.startsWith('image/')) return canAttachImage
        if (file.type.startsWith('audio/')) return canAttachAudio
        if (file.type.startsWith('video/')) return canAttachVideo
        return isDocumentFile(file)
      })
      if (accepted.length === 0) {
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
        return
      }
      const loaded = await Promise.all(accepted.map(fileToAttachment))
      setAttachments((current) => [...current, ...loaded])
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  async function selectFiles(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    const files = Array.from(event.target.files ?? [])
    event.target.value = ''
    await attachFiles(files)
  }

  function handleComposerPaste(event: ClipboardEvent<HTMLTextAreaElement>): void {
    if (appMode !== 'chat') return
    const files = Array.from(event.clipboardData?.items ?? [])
      .filter((item) => item.kind === 'file')
      .map((item) => item.getAsFile())
      .filter((file): file is File => file !== null)
    if (files.length === 0) return
    if (!event.clipboardData.getData('text/plain')) {
      event.preventDefault()
    }
    void attachFiles(files)
  }

  function handleWorkspaceDragEnter(event: DragEvent<HTMLElement>): void {
    if (appMode !== 'chat' || !event.dataTransfer.types.includes('Files')) return
    event.preventDefault()
    dragDepthRef.current += 1
    setComposerDragOver(true)
  }

  function handleWorkspaceDragOver(event: DragEvent<HTMLElement>): void {
    if (appMode !== 'chat' || !event.dataTransfer.types.includes('Files')) return
    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
  }

  function handleWorkspaceDragLeave(event: DragEvent<HTMLElement>): void {
    if (appMode !== 'chat' || !event.dataTransfer.types.includes('Files')) return
    event.preventDefault()
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)
    if (dragDepthRef.current === 0) setComposerDragOver(false)
  }

  function handleWorkspaceDrop(event: DragEvent<HTMLElement>): void {
    if (appMode !== 'chat' || !event.dataTransfer.types.includes('Files')) return
    event.preventDefault()
    dragDepthRef.current = 0
    setComposerDragOver(false)
    const files = Array.from(event.dataTransfer.files ?? [])
    if (files.length > 0) void attachFiles(files)
  }

  /**
   * Persist a chat message, or keep it in renderer memory for incognito
   * sessions where nothing may reach the daemon.
   */
  async function persistChatMessage(
    targetConversationId: string,
    input: {
      parent_id: string | null
      role: Role
      content: string | ContentPart[]
      model?: string
      tool_calls?: unknown[] | null
      tool_call_id?: string | null
      source?: string
      status?: string
      correlation_id?: string
      metadata?: Record<string, unknown>
    },
    expectedTurn?: number
  ): Promise<Message> {
    if (incognitoRef.current) {
      const message: Message = {
        id: `ephemeral-${crypto.randomUUID()}`,
        conversation_id: INCOGNITO_ID,
        parent_id: input.parent_id,
        role: input.role,
        content: input.content,
        model: input.model ?? null,
        ...(input.tool_calls !== undefined ? { tool_calls: input.tool_calls } : {}),
        ...(input.tool_call_id !== undefined ? { tool_call_id: input.tool_call_id } : {}),
        ...(input.source !== undefined ? { source: input.source } : {}),
        ...(input.status !== undefined ? { status: input.status } : {}),
        ...(input.correlation_id !== undefined ? { correlation_id: input.correlation_id } : {}),
        ...(input.metadata !== undefined ? { metadata: input.metadata } : {}),
        created_at: new Date().toISOString()
      }
      if (expectedTurn === undefined || expectedTurn === chatTurnRef.current) {
        setMessages((current) => (incognitoRef.current ? [...current, message] : current))
      }
      return message
    }
    return createMessage(targetConversationId, input)
  }

  async function executeAssistantTurn(params: {
    turn: number
    controller: AbortController
    activeConversationId: string
    userMessage: Message
    baseChain: Message[]
    promptText: string
    shouldGenerateTitle: boolean
  }): Promise<void> {
    const {
      turn,
      controller,
      activeConversationId,
      userMessage,
      baseChain,
      promptText,
      shouldGenerateTitle
    } = params
    const wasIncognito = incognitoRef.current
    const memoryEnabled = (memoryPreference?.enabled ?? true) && !wasIncognito
    let requestMessages = [...baseChain]
    let latestGenerationRate: number | null = null

    // Recall relevant memories into this turn before the first request. The
    // system message is ephemeral — it is never persisted to the chain.
    if (memoryEnabled && memoryPreference && promptText) {
      const memories = await loadRelevantMemories(promptText, memoryPreference.recall_count)
      if (turn !== chatTurnRef.current) return
      const context = buildMemoryContext(memories, memoryPreference.recall_chars)
      if (context) {
        const memorySystemMessage: Message = {
          id: `memory-${crypto.randomUUID()}`,
          conversation_id: activeConversationId,
          parent_id: null,
          role: 'system',
          content: context,
          model: null,
          created_at: new Date().toISOString()
        }
        requestMessages = [memorySystemMessage, ...requestMessages]
      }
    }

    let parentId = userMessage.id
    let responseText = ''
    let responseReasoning = ''
    const toolRecords: ToolCallRecord[] = []
    const toolOffsets: number[] = []
    const maxClientRounds = 4
    for (let round = 0; round < maxClientRounds; round += 1) {
      if (turn !== chatTurnRef.current) return
      setStreamingReasoning('')
      let roundReasoning = ''
      const result = await streamCompletion(
        requestMessages,
        selectedModel,
        controller.signal,
        (token) => {
          if (turn !== chatTurnRef.current) return
          setModelLoadStatus(null)
          responseText += token
          setStreamingText(responseText)
        },
        {
          builtinTools: toolsEnabled,
          builtinToolNames: toolsEnabled ? enabledTools : undefined,
          extraTools: memoryEnabled ? memoryToolDefinitions() : undefined,
          toolChoice: toolsEnabled || memoryEnabled ? 'auto' : undefined,
          onLoad: (event) => {
            if (turn === chatTurnRef.current) setModelLoadStatus(event.message)
          },
          onPrefill: (event) => {
            if (turn === chatTurnRef.current) setModelLoadStatus(prefillProgressLabel(event))
          },
          onReasoning: (token) => {
            if (turn !== chatTurnRef.current) return
            setModelLoadStatus(null)
            roundReasoning += token
            setStreamingReasoning(roundReasoning)
          },
          onToolCall: (record) => {
            if (turn !== chatTurnRef.current) return
            toolRecords.push(record)
            toolOffsets.push(responseText.length)
            setStreamingTools([...toolRecords])
            setStreamingToolOffsets([...toolOffsets])
          }
        }
      )
      if (turn !== chatTurnRef.current) return
      setModelPrepareState((current) => (current === 'idle' ? 'ready' : current))
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
        const created = await persistChatMessage(
          activeConversationId,
          {
            parent_id: parentId,
            role,
            content: entry.content ?? '',
            tool_calls: entry.tool_calls ?? null,
            tool_call_id: entry.tool_call_id ?? null,
            model: role === 'assistant' ? selectedModel : undefined,
            ...(role === 'assistant' && entryReasoning
              ? { metadata: { reasoning_content: entryReasoning } }
              : {})
          },
          turn
        )
        if (turn !== chatTurnRef.current) return
        requestMessages = [...requestMessages, created]
        parentId = created.id
      }
      if (result.clientToolCalls.length === 0) {
        break
      }
      // Memory tools have no server-side handler; the renderer executes them
      // against the memory API and feeds the result back to the model.
      let allClientHandled = true
      for (const call of result.clientToolCalls) {
        if (isMemoryClientTool(call.name)) {
          const outcome = wasIncognito
            ? {
                output: 'Memory is disabled in incognito chats; nothing was saved or read.',
                is_error: true
              }
            : await executeMemoryClientTool(call, {
                conversation_id: activeConversationId,
                message_id: userMessage.id
              })
          if (turn !== chatTurnRef.current) return
          const toolMessage = await persistChatMessage(
            activeConversationId,
            {
              parent_id: parentId,
              role: 'tool',
              content: outcome.output,
              tool_call_id: call.id,
              source: 'tool'
            },
            turn
          )
          if (turn !== chatTurnRef.current) return
          requestMessages = [...requestMessages, toolMessage]
          parentId = toolMessage.id
        } else {
          allClientHandled = false
        }
      }
      if (allClientHandled) continue
      throw new GenerationFailure(
        `The model requested client-side tools that Brazier cannot execute: ${result.clientToolCalls.map((call) => call.name).join(', ')}`
      )
    }
    const assistant = await persistChatMessage(
      activeConversationId,
      {
        parent_id: parentId,
        role: 'assistant',
        content: responseText,
        model: selectedModel,
        ...(responseReasoning.trim()
          ? { metadata: { reasoning_content: responseReasoning } }
          : {})
      },
      turn
    )
    if (turn !== chatTurnRef.current) return
    if (!wasIncognito && shouldGenerateTitle && responseText.trim()) {
      maybeGenerateConversationTitle(
        activeConversationId,
        promptText,
        responseText,
        latestGenerationRate,
        readChatTitleMode()
      )
    }
    let finalTipId = assistant.id
    const generatedMedia = [
      ...new Map(
        toolRecords
          .flatMap((record) =>
            (record.media ?? []).map((media) => ({ ...media, call_id: record.call_id }))
          )
          .map((media) => [`${media.sha256}:${media.mime_type}`, media])
      ).values()
    ]
    if (generatedMedia.length > 0) {
      const generated = await persistChatMessage(
        activeConversationId,
        {
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
                    : `generated-image-${index + 1}.png`),
              call_id: media.call_id
            }
          })),
          model: selectedModel,
          source: 'assistant_chat',
          metadata: { generated_media_display: true }
        },
        turn
      )
      if (turn !== chatTurnRef.current) return
      finalTipId = generated.id
    }
    if (runtime && !wasIncognito) {
      await recordRun(activeConversationId, {
        parent_message_id: userMessage.id,
        assistant_message_id: assistant.id,
        model: selectedModel,
        settings: runtime,
        tool_calls: toolRecords,
        response_text: responseText
      })
      if (turn !== chatTurnRef.current) return
    }
    setStreamingText('')
    setStreamingReasoning('')
    setStreamingTools([])
    setStreamingToolOffsets([])
    setModelLoadStatus(null)
    if (wasIncognito) {
      setTipId(finalTipId)
    } else {
      await refreshMessages(activeConversationId, finalTipId)
      if (turn !== chatTurnRef.current) return
      await refreshConversations()
      if (turn !== chatTurnRef.current) return
      void listRunSnapshots(activeConversationId)
        .then((snapshots) => {
          if (turn === chatTurnRef.current) setRunSnapshots(snapshots)
        })
        .catch(reportBackgroundFailure)
    }
    scheduleDream()
  }

  async function retryAssistantTurn(userMessage: Message): Promise<void> {
    if (busy || coordinatorWorking) return
    if (!selectedModel) {
      setError('Select or download a local model first.')
      setModelMenuOpen(true)
      return
    }
    if (!canChat) return
    if (runtimeWarning) {
      setError(runtimeWarning)
      openManage('runtimes')
      return
    }
    stickToBottomRef.current = true
    const activeConversationId = incognito ? INCOGNITO_ID : conversationId
    if (!activeConversationId) return
    const turn = ++chatTurnRef.current
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
    setTipId(userMessage.id)
    clearDreamTimer()
    setDreamPromptOpen(false)
    setIncognitoDiscardOpen(false)
    const controller = new AbortController()
    abortRef.current = controller
    const baseChain = messageChain(messages, userMessage.id)
    try {
      await executeAssistantTurn({
        turn,
        controller,
        activeConversationId,
        userMessage,
        baseChain,
        promptText: contentText(userMessage),
        shouldGenerateTitle: false
      })
    } catch (cause) {
      if (turn === chatTurnRef.current && (cause as Error).name !== 'AbortError') {
        if (cause instanceof GenerationFailure) {
          setError(cause.message)
          setForkHints(cause.forkHints)
        } else {
          setError(cause instanceof Error ? cause.message : String(cause))
          setForkHints([])
        }
      }
    } finally {
      if (turn === chatTurnRef.current) {
        setBusy(false)
        setStreamingText('')
        setStreamingReasoning('')
        setStreamingTools([])
        setStreamingToolOffsets([])
        setModelLoadStatus(null)
        abortRef.current = undefined
      }
    }
  }

  async function submit(event: FormEvent): Promise<void> {
    event.preventDefault()
    const text = draft.trim()
    // Agent mode: the same box, pointed at the agent. Its own transcript and
    // approval cards render above; only the input is shared.
    if (appMode === 'agent' || appMode === 'computer') {
      const composer = appMode === 'agent' ? agentComposer : computerComposer
      if (!text || !composer) return
      // Agent mode takes follow-ups mid-run: queued by default, or sent as an
      // interruption. Computer mode still waits for its run to stop.
      const interrupt = interruptNextSubmitRef.current
      interruptNextSubmitRef.current = false
      if (composer.running && appMode !== 'agent') return
      const isAllowedSlash =
        Boolean(composer.suggestions) && isExactSlashCommand(draft, composer.suggestions!)
      if (composer.blockedReason !== '' && !isAllowedSlash) return
      setDraft('')
      setError(null)
      await composer.send(text, interrupt ? 'interrupt' : 'queue')
      return
    }
    if ((!text && attachments.length === 0) || busy || coordinatorWorking) return
    if (attachments.length === 0 && text.startsWith('/')) {
      const cmd = text.toLowerCase()
      if (cmd === '/new') {
        setDraft('')
        await newConversation()
        return
      }
      if (cmd === '/incognito') {
        setDraft('')
        toggleIncognito()
        return
      }
      if (cmd === '/model') {
        setDraft('')
        setModelMenuOpen(true)
        return
      }
      if (cmd === '/settings') {
        setDraft('')
        setInferenceMenuOpen(true)
        return
      }
      if (cmd === '/unload') {
        setDraft('')
        await unloadSelectedModel()
        return
      }
      if (cmd === '/export') {
        setDraft('')
        if (conversationId && !incognito) {
          await exportConversationFile(conversationId)
        } else {
          setError('Open a saved conversation before exporting.')
        }
        return
      }
      if (cmd === '/dream') {
        setDraft('')
        await runDream()
        return
      }
    }
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
    if (!canChat) return
    if (runtimeWarning) {
      setError(runtimeWarning)
      openManage('runtimes')
      return
    }
    stickToBottomRef.current = true
    const turn = ++chatTurnRef.current
    const submittedAttachments = attachments
    const baseChainSnapshot = chain
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
    // New activity defers dreaming: the idle timer restarts on completion, and
    // a pending "ask" prompt is withdrawn rather than nagging mid-conversation.
    clearDreamTimer()
    setDreamPromptOpen(false)
    setIncognitoDiscardOpen(false)
    const controller = new AbortController()
    abortRef.current = controller
    const shouldGenerateTitle =
      !conversationId ||
      (baseChainSnapshot.length === 0 &&
        conversations.find((conversation) => conversation.id === conversationId)?.title ===
          'New conversation')

    try {
      let activeConversationId = conversationId
      if (incognito) {
        activeConversationId = INCOGNITO_ID
      } else if (!activeConversationId) {
        const created = await createConversation()
        if (turn !== chatTurnRef.current) return
        activeConversationId = created.id
        conversationIdRef.current = created.id
        skipNextConversationLoadRef.current = created.id
        setConversationId(created.id)
      }
      const parts: ContentPart[] = [
        ...(text ? [{ type: 'text' as const, text }] : []),
        ...submittedAttachments.map(attachmentPart)
      ]
      const content = submittedAttachments.length === 0 ? text : parts
      const userMessage = await persistChatMessage(
        activeConversationId,
        {
          parent_id: baseChainSnapshot.at(-1)?.id ?? null,
          role: 'user',
          content
        },
        turn
      )
      if (turn !== chatTurnRef.current) return
      if (!incognitoRef.current) setMessages((current) => [...current, userMessage])
      setTipId(userMessage.id)
      setDraft('')
      setAttachments([])

      await executeAssistantTurn({
        turn,
        controller,
        activeConversationId,
        userMessage,
        baseChain: [...baseChainSnapshot, userMessage],
        promptText: text,
        shouldGenerateTitle
      })
    } catch (cause) {
      if (turn === chatTurnRef.current && (cause as Error).name !== 'AbortError') {
        if (cause instanceof GenerationFailure) {
          setError(cause.message)
          setForkHints(cause.forkHints)
        } else {
          setError(cause instanceof Error ? cause.message : String(cause))
          setForkHints([])
        }
      }
    } finally {
      if (turn === chatTurnRef.current) {
        setBusy(false)
        setStreamingText('')
        setStreamingReasoning('')
        setStreamingTools([])
        setStreamingToolOffsets([])
        setModelLoadStatus(null)
        abortRef.current = undefined
      }
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
            void fetchWorkspacePreference()
              .then((result) => setWorkspaceModes(result.modes))
              .catch(reportBackgroundFailure)
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
          onModelsChanged={() => {
            // Installs write their mode defaults daemon-side; pull the fresh
            // settings so the seeded selections pick them up.
            void refreshRuntime().catch(reportBackgroundFailure)
            void refreshLocalModels().catch(reportBackgroundFailure)
          }}
        />
      </main>
    )
  }

  return (
    <ConnectionProfileContext.Provider value={connectionProfile ?? LOCAL_CONNECTION_PROFILE}>
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
                  ? 'Sessions'
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
            onSelect={(id) => {
              const entry = id ? generateHistory.find((item) => item.id === id) : undefined
              if (entry) setGenerateModality(entry.modality)
              setActiveGenerateHistoryId(id)
            }}
          />
        ) : appMode === 'computer' ? (
          <ComputerSessionSidebar controls={computerSidebar} />
        ) : (
          <>
            <button
              type="button"
              className="new-chat"
              onClick={() => void newConversation()}
            >
              <MessageSquarePlus size={17} />
              New chat
            </button>
            <label className="conversation-search">
              <Search size={14} />
              <input
                aria-label="Search chats"
                value={conversationSearch}
                onChange={(event) => setConversationSearch(event.target.value)}
                placeholder="Search"
              />
            </label>
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
            <div className="conversation-list">
              <div className="section-label conversation-list-head">
                Chats
                <button
                  type="button"
                  className="conversation-import"
                  title="Import a chat from a JSON export"
                  aria-label="Import chat"
                  onClick={() => importInput.current?.click()}
                >
                  <Upload size={12} />
                </button>
              </div>
              {conversations.map((conversation, index) => (
                <Fragment key={conversation.id}>
                  {index === 0 ||
                  historyGroup(conversations[index - 1].updated_at) !==
                    historyGroup(conversation.updated_at) ? (
                    <div className="conversation-group-label">
                      {historyGroup(conversation.updated_at)}
                    </div>
                  ) : null}
                <div
                  className={conversation.id === conversationId ? 'conversation active' : 'conversation'}
                >
                  {renaming?.id === conversation.id ? (
                    <input
                      className="conversation-rename"
                      aria-label="Chat name"
                      autoFocus
                      value={renaming.draft}
                      onFocus={(event) => event.currentTarget.select()}
                      onChange={(event) =>
                        setRenaming({ id: conversation.id, draft: event.target.value })
                      }
                      onBlur={(event) => void commitRename(conversation, event.currentTarget.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          event.preventDefault()
                          event.currentTarget.blur()
                        } else if (event.key === 'Escape') {
                          event.preventDefault()
                          event.stopPropagation()
                          renameCancelledRef.current = true
                          setRenaming(null)
                        }
                      }}
                    />
                  ) : (
                    <button
                      className="conversation-select"
                      onClick={() => {
                        if (conversation.id === conversationId && !incognito) return
                        confirmOrLeaveIncognito(() => {
                          abortChatGeneration()
                          if (incognitoRef.current) {
                            incognitoRef.current = false
                            setIncognito(false)
                          }
                          setIncognitoDiscardOpen(false)
                          pendingIncognitoActionRef.current = null
                          conversationIdRef.current = conversation.id
                          setMessages([])
                          setTipId(null)
                          setAttachments([])
                          setConversationId(conversation.id)
                        })
                      }}
                      onDoubleClick={() => startRename(conversation)}
                    >
                      <span>{conversation.title}</span>
                      <time dateTime={conversation.updated_at}>
                        {shortDate(conversation.updated_at)}
                      </time>
                    </button>
                  )}
                  <button
                    className="conversation-menu-button"
                    aria-label={`Actions for ${conversation.title}`}
                    title="More"
                    onClick={() => {
                      setConfirmDeleteId(null)
                      setConversationMenuId((current) =>
                        current === conversation.id ? null : conversation.id
                      )
                    }}
                  >
                    <Ellipsis size={16} />
                  </button>
                  {conversationMenuId === conversation.id ? (
                    <>
                      <div
                        className="conversation-menu-backdrop"
                        onMouseDown={() => {
                          setConversationMenuId(null)
                          setConfirmDeleteId(null)
                        }}
                      />
                      <div className="conversation-menu">
                        {confirmDeleteId === conversation.id ? (
                          <>
                            <p>Delete this chat? This can’t be undone.</p>
                            <button
                              className="danger"
                              onClick={() => void removeConversation(conversation)}
                            >
                              <Trash2 size={13} /> Delete
                            </button>
                            <button onClick={() => setConfirmDeleteId(null)}>Cancel</button>
                          </>
                        ) : (
                          <>
                            <button onClick={() => startRename(conversation)}>
                              <Pencil size={13} /> Rename
                            </button>
                            <button onClick={() => void exportConversationFile(conversation.id)}>
                              <Download size={13} /> Export
                            </button>
                            <button
                              className="danger"
                              onClick={() => setConfirmDeleteId(conversation.id)}
                            >
                              <Trash2 size={13} /> Delete
                            </button>
                          </>
                        )}
                      </div>
                    </>
                  ) : null}
                </div>
                </Fragment>
              ))}
              {conversations.length === 0 && (
                <p className="empty-sidebar">
                  {conversationSearch.trim() ? 'No matching chats.' : 'No chats yet.'}
                </p>
              )}
            </div>
            <RunHistory
              runs={runSnapshots}
              expandedId={expandedRunId}
              onToggle={(id) => setExpandedRunId((current) => (current === id ? null : id))}
            />
          </>
        )}
        <button
          type="button"
          className="privacy-note"
          title="Connections"
          onClick={() => openManage('connections')}
        >
          <span className={`status-dot ${daemonStatus}`} />
          <span>
            {connectionProfile?.kind === 'remote'
              ? `${connectionProfile.name} · ${connectionProfile.hostLabel}`
              : 'This computer'}
            {daemonStatus === 'healthy'
              ? daemonVersion
                ? ` · v${daemonVersion}`
                : ''
              : daemonStatus === 'offline'
                ? connectionProblem ? ' · access unavailable' : ' · offline'
                : ' · connecting…'}
          </span>
        </button>
      </aside>

      <section
        className="workspace"
        onDragEnter={handleWorkspaceDragEnter}
        onDragOver={handleWorkspaceDragOver}
        onDragLeave={handleWorkspaceDragLeave}
        onDrop={handleWorkspaceDrop}
      >
        <header className="topbar">
          <button
            className="icon-button"
            title={`${sidebarOpen ? 'Hide' : 'Show'} sidebar (${shortcutLabel('B')})`}
            aria-label={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
            aria-expanded={sidebarOpen}
            onClick={() => setSidebarOpen((open) => !open)}
          >
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
            (modelPrepareState === 'ready' ||
              modelPrepareState === 'idle' ||
              modelUnloading) && (
              <button
                type="button"
                className="model-unload-button"
                disabled={modelUnloading || busy || Boolean(shellComposer?.running)}
                title={
                  busy || shellComposer?.running ? 'Stop the response to unload' : 'Unload model'
                }
                aria-label="Unload model"
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
            title={`Choose model (${shortcutLabel('M', true)})`}
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
            disabled={appMode !== 'chat' && appMode !== 'agent' && !modeModel.selected}
            title={
              appMode === 'chat' || appMode === 'agent'
                ? 'Generation settings'
                : modeModel.selected
                  ? 'Model settings'
                  : 'Select a model to configure settings'
            }
            aria-label={
              appMode === 'chat' || appMode === 'agent' ? 'Generation settings' : 'Model settings'
            }
            onClick={openActiveModelSettings}
          >
            <SlidersHorizontal size={17} />
          </button>
          {appMode === 'chat' && (
            <button
              className={`icon-button incognito-toggle${incognito ? ' active' : ''}`}
              title={`${incognito ? 'Leave incognito' : 'Incognito chat'} (${shortcutLabel('I', true)})`}
              aria-label={incognito ? 'Leave incognito' : 'Incognito chat'}
              aria-pressed={incognito}
              onClick={toggleIncognito}
            >
              <EyeOff size={17} />
            </button>
          )}
          <button
            className="icon-button manage-button"
            title={`Manage (${shortcutLabel(',')})`}
            aria-label="Manage"
            onClick={() => openManage('library')}
          >
            <Settings2 size={18} />
          </button>
        </header>

        {daemonStatus === 'offline' && (
          <div className="runtime-notice connection-offline-notice" role="status">
            <CircleAlert size={15} />
            <span>
              {connectionProblem ?? `Can’t reach ${connectionProfile?.kind === 'remote' ? connectionProfile.name : 'Brazier’s engine'}.`}
              {' '}You can browse what’s cached, but changes are paused.
            </span>
            <button type="button" onClick={() => void reconnectDaemon()}>
              Retry
            </button>
            <button type="button" onClick={() => openManage('connections')}>
              Connections
            </button>
          </div>
        )}

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

        {incognito && appMode === 'chat' && (
          <div className="runtime-notice incognito-notice">
            <EyeOff size={14} />
            <span>Incognito. Nothing is saved, and memory is off.</span>
            {incognitoDiscardOpen && (
              <>
                <span className="incognito-discard-label">Discard this chat?</span>
                <button
                  type="button"
                  onClick={() => {
                    const pending = pendingIncognitoActionRef.current
                    pendingIncognitoActionRef.current = null
                    setIncognitoDiscardOpen(false)
                    if (pending) {
                      pending()
                    } else {
                      leaveIncognito()
                    }
                  }}
                >
                  Discard
                </button>
                <button
                  type="button"
                  onClick={() => {
                    pendingIncognitoActionRef.current = null
                    setIncognitoDiscardOpen(false)
                  }}
                  className="incognito-keep"
                >
                  Keep
                </button>
              </>
            )}
          </div>
        )}

        {dreamPromptOpen && (
          <div className="runtime-notice dream-notice">
            <Brain size={14} />
            <span>
              Dream with {selectedMeta.title}? It merges and prunes memories from recent chats.
            </span>
            <button type="button" onClick={() => void runDream()}>
              Dream
            </button>
            <button
              type="button"
              onClick={() => {
                setDreamPromptOpen(false)
                markDreamed()
              }}
            >
              Not now
            </button>
          </div>
        )}

        {dreamStatus && !dreamPromptOpen && (
          <div className="runtime-notice dream-notice">
            <Brain size={14} />
            <span>{dreamStatus}</span>
            <button type="button" onClick={() => setDreamStatus(null)}>
              <X size={14} />
            </button>
          </div>
        )}

        <div hidden={appMode !== 'generate'}>
          <ErrorBoundary label="Generate workspace" resetKey={profileId}>
          <GenerateMode
            models={generateModels}
            modality={generateModality}
            onModalityChange={setGenerateModality}
            modelId={generateModel}
            settings={runtime}
            hardware={hardware}
            modelProfiles={modelProfiles}
            onOpenManage={openManage}
            onError={setError}
            history={generateHistory}
            activeHistoryId={activeGenerateHistoryId}
            onGenerated={(entry) => {
              setGenerateHistory((current) => {
                const next = [entry, ...current]
                writeGenerateHistory(profileId, next)
                return next
              })
              setActiveGenerateHistoryId(entry.id)
            }}
          />
          </ErrorBoundary>
        </div>
        {appMode === 'voice' ? (
          <ErrorBoundary label="Voice workspace" resetKey={profileId}>
          <VoiceMode
            models={effectiveLocalModels}
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
            onOpenManage={openManage}
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
          </ErrorBoundary>
        ) : null}
        {appMode === 'agent' ? (
          <ErrorBoundary label="Agent workspace" resetKey={profileId}>
          <AgentMode
            modelId={selectedModel}
            models={effectiveLocalModels}
            onComposerChange={setAgentComposer}
            onSidebarChange={setAgentSidebar}
            onSuggestPrompt={setDraft}
            onSessionBound={session.bindAgentSession}
            onSubmitTurn={session.submitText}
            onCancelTurn={
              coordinatorWorking && session.snapshot.agentSessionId
                ? session.cancelResponse
                : undefined
            }
            onClearQueue={session.clearQueuedTurns}
            onOpenModelMenu={() => setModelMenuOpen(true)}
            onError={setError}
          />
          </ErrorBoundary>
        ) : null}
        {appMode === 'computer' ? (
          <ErrorBoundary label="Computer Use workspace" resetKey={profileId}>
          <ComputerMode
            modelId={computerModel}
            models={computerModels}
            onComposerChange={setComputerComposer}
            onSidebarChange={setComputerSidebar}
            onOpenManage={openManage}
            onError={setError}
          />
          </ErrorBoundary>
        ) : null}

        <div
          ref={chatScrollContainerRef}
          className="chat"
          hidden={appMode !== 'chat'}
          onScroll={(event) => {
            const el = event.currentTarget
            stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 96
          }}
        >
          {modelLoadStatus &&
            modelPrepareState !== 'loading' &&
            busy &&
            !streamingText && (
              <div className="model-load-status">
                <LoaderCircle className="spin" size={18} />
                <span>{modelLoadStatus}</span>
              </div>
            )}
          {chain.length === 0 && !liveText ? (
            <div className="welcome">
              <img className="welcome-logo" src={brazierLogo} alt="" />
              <h1>
                {modelsLoading
                  ? 'Starting up…'
                  : chatModels.length === 0 && daemonStatus === 'offline'
                    ? 'Can’t reach the engine'
                    : chatModels.length === 0
                    ? 'No chat models yet'
                    : !selectedModel && modelPrepareState !== 'error'
                      ? 'Choose a model'
                      : modelPrepareState === 'loading'
                        ? `Loading ${selectedMeta.title}…`
                        : modelPrepareState === 'error'
                          ? 'The model didn’t load'
                          : incognito
                            ? 'Incognito chat'
                            : 'What are we exploring?'}
              </h1>
              <p>
                {modelsLoading
                  ? 'Loading your model library.'
                  : chatModels.length === 0 && daemonStatus === 'offline'
                    ? 'Brazier reconnects on its own once the engine is back.'
                    : chatModels.length === 0
                    ? 'Download one to start chatting.'
                    : !selectedModel && modelPrepareState !== 'error'
                      ? `You have ${chatModels.length} installed.`
                      : modelPrepareState === 'loading'
                        ? 'This can take a moment for large models.'
                        : modelPrepareState === 'error'
                          ? 'See the error below, or try another model.'
                          : incognito
                            ? 'Nothing here is saved, and memory is off.'
                            : 'Ask a question, or attach a file.'}
              </p>
              {!modelsLoading &&
              !selectedModel &&
              modelPrepareState !== 'loading' &&
              !(chatModels.length === 0 && daemonStatus === 'offline') ? (
                <div className="welcome-actions">
                  {chatModels.length === 0 ? (
                    <button
                      type="button"
                      className="primary-button"
                      onClick={() => openManage('recommended')}
                    >
                      Browse models
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="primary-button"
                      onClick={() => setModelMenuOpen(true)}
                    >
                      Choose a model
                    </button>
                  )}
                </div>
              ) : null}
            </div>
          ) : (
            <div className="messages">
              {chatDisplayItems.map((item) => (
                <ErrorBoundary
                  key={item.kind === 'message' ? item.message.id : item.id}
                  label="message"
                >
                  <RenderSlot
                    render={() => {
                if (item.kind === 'assistant') {
                  const status =
                    item.status === 'cancelled' ||
                    item.status === 'superseded' ||
                    item.status === 'failed'
                      ? item.status
                      : null
                  const turnMessage = messages.find((entry) => entry.id === item.branchId)
                  const turnModelTitle =
                    turnMessage?.model && item.source !== 'assistant_agent'
                      ? modelDisplayName(
                          turnMessage.model,
                          effectiveLocalModels.find((entry) => entry.id === turnMessage.model)
                        ).title
                      : null
                  const turnRootId = assistantTurnRootId(messages, item.branchId)
                  const parentUser = parentUserMessageForTurn(messages, item.branchId)
                  return (
                    <article className="message assistant" key={item.id}>
                      <div className="avatar">
                        <Bot />
                      </div>
                      <div className="message-body">
                        <div className="message-meta">
                          <strong>{item.source === 'assistant_agent' ? 'Agent' : 'Brazier'}</strong>
                          {turnModelTitle ? (
                            <span className="message-model-label">{turnModelTitle}</span>
                          ) : null}
                          {status ? (
                            <span className={`turn-badge ${status}`}>{status}</span>
                          ) : null}
                        </div>
                        {item.trace.length > 0 && (
                          <TurnTrace
                            reasoning={item.reasoning}
                            defaultOpen={item.trace.some(
                              (segment) => segment.kind === 'tool' || segment.kind === 'media'
                            )}
                          >
                            {item.trace.map((segment) => {
                              if (segment.kind === 'reasoning') {
                                return (
                                  <div
                                    key={segment.key}
                                    className="reasoning-chunk"
                                  >
                                    {segment.text}
                                  </div>
                                )
                              }
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
                          </TurnTrace>
                        )}
                        {item.answer && <Markdown>{item.answer}</Markdown>}
                        <div className="message-actions">
                          {item.answer ? (
                            <button
                              type="button"
                              className="message-action-button"
                              title="Copy response"
                              onClick={() => copyTurnAnswer(item.id, item.answer)}
                            >
                              {copiedTurnId === item.id ? (
                                <>
                                  <Check size={12} /> Copied
                                </>
                              ) : (
                                <>
                                  <Copy size={12} /> Copy
                                </>
                              )}
                            </button>
                          ) : null}
                          {parentUser && item.source !== 'assistant_agent' ? (
                            <button
                              type="button"
                              className="message-action-button"
                              disabled={busy || coordinatorWorking}
                              title="Regenerate as a new branch"
                              onClick={() => void retryAssistantTurn(parentUser)}
                            >
                              <RotateCcw size={12} /> Retry
                            </button>
                          ) : null}
                          <BranchNavigator
                            messages={messages}
                            messageId={turnRootId}
                            onSelect={(id) => setTipId(latestBranchTipId(messages, id))}
                          />
                        </div>
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
                      <div className="message-actions">
                        <button
                          type="button"
                          className="fork-button"
                          disabled={busy || coordinatorWorking}
                          title="Edit and resend as a new branch"
                          onClick={() => {
                            const restoredAttachments: Attachment[] = contentBlobs(message).map(
                              (blob, index) => ({
                                id: `${blob.sha256}-${index}-${crypto.randomUUID()}`,
                                name:
                                  blob.original_name ||
                                  (blob.mime_type.startsWith('image/')
                                    ? 'image'
                                    : blob.mime_type.startsWith('audio/')
                                      ? 'audio'
                                      : blob.mime_type.startsWith('video/')
                                        ? 'video'
                                        : 'attachment'),
                                type: blob.mime_type,
                                sha256: blob.sha256
                              })
                            )
                            setDraft(contentText(message))
                            setAttachments(restoredAttachments)
                            setTipId(message.parent_id ?? ROOT_TIP)
                          }}
                        >
                          <Pencil size={13} /> Edit
                        </button>
                        <BranchNavigator
                          messages={messages}
                          messageId={message.id}
                          onSelect={(id) => setTipId(latestBranchTipId(messages, id))}
                        />
                      </div>
                    </div>
                  </article>
                )
                    }}
                  />
                </ErrorBoundary>
              ))}
              {(liveText || streamingReasoning || streamingTools.length > 0) && (
                <article className="message assistant">
                  <div className="avatar">
                    <Bot />
                  </div>
                  <div className="message-body">
                    <div className="message-meta">
                      <strong>
                        {session.snapshot.streamingText && streamingOwner === 'agent'
                          ? 'Agent'
                          : 'Brazier'}
                      </strong>
                      {busy || coordinatorWorking ? <LoaderCircle className="spin" size={14} /> : null}
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
          <GenerationActivity onStopped={() => void refreshLocalModels().catch(reportBackgroundFailure)} />
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
            ref={composerFormRef}
            className={composerDragOver ? 'composer drag-over' : 'composer'}
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
                    ? 'Choose a model to start chatting…'
                    : modelPrepareState === 'loading'
                      ? 'Loading model…'
                      : modelPrepareState === 'error'
                        ? 'Choose a model to chat…'
                        : tipId
                          ? 'Continue this branch…'
                          : 'Message…'
              }
              rows={shellComposerMode ? 2 : 1}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onPaste={handleComposerPaste}
              onKeyDown={(event) => {
                if (visibleSuggestions.length > 0) {
                  if (event.key === 'ArrowDown') {
                    event.preventDefault()
                    setSuggestionIndex((index) => (index + 1) % visibleSuggestions.length)
                    return
                  }
                  if (event.key === 'ArrowUp') {
                    event.preventDefault()
                    setSuggestionIndex(
                      (index) => (index - 1 + visibleSuggestions.length) % visibleSuggestions.length
                    )
                    return
                  }
                  if (event.key === 'Tab' || (event.key === 'Enter' && !event.shiftKey)) {
                    const chosen = visibleSuggestions[clampedSuggestionIndex]
                    if (chosen) {
                      event.preventDefault()
                      setDraft((current) => replaceTrailingCommand(current, chosen.value))
                      return
                    }
                  }
                  if (event.key === 'Escape') {
                    event.preventDefault()
                    event.stopPropagation()
                    setDismissedSuggestionQuery(activeSuggestionQuery)
                    return
                  }
                }
                if (event.nativeEvent.isComposing || event.keyCode === 229) return
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  interruptNextSubmitRef.current =
                    agentMode && Boolean(agentComposer?.running) && (event.ctrlKey || event.metaKey)
                  event.currentTarget.form?.requestSubmit()
                }
              }}
            />
            {visibleSuggestions.length > 0 && (
              <div className="composer-suggestions" role="listbox" aria-label="Prompt suggestions">
                {visibleSuggestions.map((entry, index) => (
                  <button
                    type="button"
                    role="option"
                    aria-selected={index === clampedSuggestionIndex}
                    className={index === clampedSuggestionIndex ? 'active' : ''}
                    key={entry.value}
                    onMouseEnter={() => setSuggestionIndex(index)}
                    onClick={() =>
                      setDraft((current) => replaceTrailingCommand(current, entry.value))
                    }
                  >
                    <strong>{entry.value}</strong>
                    <span>{entry.description}</span>
                  </button>
                ))}
              </div>
            )}
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
                      ? 'This model doesn’t support tools'
                      : toolsEnabled
                        ? `Tools (${enabledTools.length} on)`
                        : 'Tools'
                  }
                  aria-label="Tools"
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
                title="Attach files"
                aria-label="Attach files"
                onClick={() => fileInput.current?.click()}
              >
                <Paperclip size={18} />
              </button>
              {agentMode && agentComposer?.running && draft.trim() ? (
                <>
                  <button
                    className="send-button"
                    type="button"
                    title="Stop the current turn and send this now (Ctrl+Enter)"
                    aria-label="Interrupt"
                    onClick={() => {
                      interruptNextSubmitRef.current = true
                      composerFormRef.current?.requestSubmit()
                    }}
                  >
                    <Zap size={16} />
                  </button>
                  <button
                    className="send-button"
                    type="submit"
                    title="Queue: send after the current turn (Enter)"
                    aria-label="Queue"
                  >
                    <ListPlus size={17} />
                  </button>
                </>
              ) : (shellComposerMode ? shellComposer?.running : busy || coordinatorWorking) ? (
                <button
                  className="send-button stop"
                  type="button"
                  title={`Stop (${shortcutLabel('.')})`}
                  aria-label="Stop"
                  onClick={() => {
                    if (shellComposerMode) void shellComposer?.stop()
                    else if (coordinatorWorking) void session.cancelResponse()
                    else abortChatGeneration()
                  }}
                >
                  <Square size={15} fill="currentColor" />
                </button>
              ) : (
                <button
                  className="send-button"
                  type="submit"
                  title="Send"
                  aria-label="Send"
                  disabled={
                    shellComposerMode
                      ? !draft.trim() ||
                        !shellComposer ||
                        (shellComposer.blockedReason !== '' &&
                          !(
                            shellComposer.suggestions &&
                            isExactSlashCommand(draft, shellComposer.suggestions)
                          ))
                      : ((!draft.trim() && attachments.length === 0) || !canChat) &&
                        !(
                          attachments.length === 0 &&
                          isExactSlashCommand(draft, CHAT_SLASH_SUGGESTIONS)
                        )
                  }
                >
                  <Send size={17} />
                </button>
              )}
            </div>
          </form>
          <p className="composer-hint" hidden={appMode !== 'chat' || !selectedModel}>
            {
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
                    title={`Vision — ${visionCapabilityTitle(selectedModel, effectiveLocalModels, canAttachImage)}`}
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
              </>
            }
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
          hardware={hardware}
          onSelect={modeModel.select}
          onConfigure={setConfiguringModel}
          configuredCounts={profileCounts}
          onManage={() =>
            openManage(
              appMode === 'computer'
                ? 'computer'
                : modelsLoadFailed || effectiveLocalModels.length === 0
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
          models={effectiveLocalModels}
          rawModels={localModels}
          residency={modelResidency}
          saving={savingInference}
          advancedModelId={modeModel.selected}
          profile={modelProfiles[modeModel.selected]}
          onApply={(next, options) => applyInferenceSettings(next, options)}
          onProfileSaved={(models) => {
            setModelProfiles(models)
            if (modeModel.selected === selectedModel && modelPrepareState === 'ready') {
              selectModel(selectedModel)
            }
          }}
          onClose={() => setInferenceMenuOpen(false)}
        />
      )}
      <CommandPalette
        open={commandPaletteOpen}
        appMode={appMode}
        workspaceModes={workspaceModes}
        models={effectiveLocalModels}
        selectedModel={selectedModel}
        voiceModel={voiceModel}
        generateModel={generateModel}
        computerModel={computerModel}
        conversations={conversations}
        activeConversationId={incognito ? null : conversationId}
        incognito={incognito}
        canUnload={Boolean(selectedModel) && !modelUnloading && !busy}
        isMac={window.brazier.platform === 'darwin'}
        onClose={() => setCommandPaletteOpen(false)}
        onSwitchMode={switchAppMode}
        onSelectModelForApp={selectModelFromAnySurface}
        onConfigureModel={setConfiguringModel}
        onSelectConversation={(id) => {
          if (appMode !== 'chat') switchAppMode('chat')
          if (id === conversationId && !incognito) return
          confirmOrLeaveIncognito(() => {
            abortChatGeneration()
            if (incognitoRef.current) {
              incognitoRef.current = false
              setIncognito(false)
            }
            setIncognitoDiscardOpen(false)
            pendingIncognitoActionRef.current = null
            conversationIdRef.current = id
            setMessages([])
            setTipId(null)
            setAttachments([])
            setConversationId(id)
          })
        }}
        onNewAction={triggerNewForActiveMode}
        onToggleIncognito={() => {
          if (appMode !== 'chat' && workspaceModes.chat) switchAppMode('chat')
          toggleIncognito()
        }}
        onUnloadModel={() => void unloadSelectedModel()}
        onExportConversation={() => {
          if (conversationId && !incognito) void exportConversationFile(conversationId)
        }}
        onCopyApiCurl={() => void copyApiCurl()}
        onOpenModelMenu={() => setModelMenuOpen(true)}
        onOpenInferenceSettings={openActiveModelSettings}
        onToggleSidebar={() => setSidebarOpen((open) => !open)}
        onRunDream={() => void runDream()}
        onOpenManage={openManage}
      />
      {configuringModel && configuredModelEntry && (
        <ModelSettingsModal
          key={configuringModel}
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
                .map((swap) => `${swap.recommended_label} for ${CATEGORY_LABELS[swap.category] ?? swap.category}`)
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
              ).catch(reportBackgroundFailure)
            }}
          >
            <X size={14} />
          </button>
        </aside>
      )}
      <DownloadTray
        models={effectiveLocalModels}
        activeModelIds={[selectedModel, generateModel, voiceModel, computerModel].filter(Boolean)}
        onUseModel={selectModelFromAnySurface}
        onChanged={() => {
          void refreshLocalModels().catch(reportBackgroundFailure)
          // A background install can settle after the welcome screen is left;
          // it writes its mode defaults daemon-side, so pull fresh settings for
          // the seeded selections to pick them up.
          void refreshRuntime().catch(reportBackgroundFailure)
        }}
      />

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
          models={effectiveLocalModels}
          modelsLoading={modelsLoading}
          refreshModels={refreshLocalModels}
          initialRuntimes={prefetchedRuntimes}
          selectedModel={selectedModel}
          onSelectModel={(modelId) => {
            const found = effectiveLocalModels.find((entry) => entry.id === modelId)
            if (found) selectModelFromAnySurface(found)
            else selectModel(modelId)
          }}
          modelBindings={modelBindings}
          onSetModelBinding={(modelId, runtimeId) => void updateModelBinding(modelId, runtimeId)}
          onRuntimesChanged={() => {
            void prefetchRuntimes()
            void fetchModelBindings().then(setModelBindings).catch(reportBackgroundFailure)
          }}
          settings={runtime}
          onSettingsSaved={setRuntime}
          hardware={hardware}
          dataDir={daemonDataDir}
          onConfigureModel={setConfiguringModel}
          profileCounts={profileCounts}
          onWorkspaceModesChange={setWorkspaceModes}
          onMemoryPreferenceSaved={setMemoryPreference}
          onDreamRequest={() => void runDream()}
        />
      )}
    </main>
    </ConnectionProfileContext.Provider>
  )
}

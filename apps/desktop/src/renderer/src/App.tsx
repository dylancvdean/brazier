import {
  AudioLines,
  Bot,
  Box,
  Brain,
  Cpu,
  Gauge,
  GitBranch,
  Image,
  LoaderCircle,
  Menu,
  MessageSquarePlus,
  Paperclip,
  Download,
  Search,
  Send,
  Settings2,
  SlidersHorizontal,
  Square,
  Upload,
  Video,
  Wrench,
  X
} from 'lucide-react'
import { type ChangeEvent, type FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  createConversation,
  createMessage,
  engineStatus,
  fetchCapabilities,
  fetchModelBindings,
  health,
  exportConversation,
  importConversation,
  listConversations,
  listMessages,
  listModels,
  listRuntimes,
  listRunSnapshots,
  GenerationFailure,
  prepareModel,
  setModelBinding,
  type ConversationExport,
  type HardwareInfo,
  type LocalModel,
  type PipelineFeatures,
  type RunSnapshot,
  type RuntimeEntry,
  type RuntimeForkHint,
  type RuntimeSettings,
  type ToolCallRecord,
  recordRun,
  saveRuntimeSettings,
  streamCompletion,
  uploadAttachmentBlob
} from './api'
import { InferenceMenu } from './components/InferenceMenu'
import { ManagePanel, type ManageSection } from './components/ManagePanel'
import { ModelMenu } from './components/ModelMenu'
import {
  isChatModel,
  modelDisplayName,
  runtimeNoticeForModel,
  visionCapabilityTitle
} from './model-utils'
import { childCounts, messageChain } from './graph'
import {
  readCachedModels,
  readCachedRuntimes,
  writeCachedModels,
  writeCachedRuntimes
} from './inventoryCache'
import type { Attachment, ContentPart, Conversation, Message, Role } from './types'

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

/** Tool records for UI display from native or legacy tool messages. */
function toolRecordsFromMessage(message: Message): ToolCallRecord[] | null {
  if (message.role !== 'tool') return null
  if (message.tool_call_id && typeof message.content === 'string') {
    return [
      {
        call_id: message.tool_call_id,
        name: 'tool',
        arguments: '',
        output: message.content,
        is_error: false
      }
    ]
  }
  if (typeof message.content !== 'string') return null
  try {
    const parsed = JSON.parse(message.content) as { brazier_tool_calls?: ToolCallRecord[] }
    return Array.isArray(parsed.brazier_tool_calls) ? parsed.brazier_tool_calls : null
  } catch {
    return null
  }
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

function ToolChips({ records }: { records: ToolCallRecord[] }): React.JSX.Element {
  return (
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
  )
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
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [conversationSearch, setConversationSearch] = useState('')
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [tipId, setTipId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [streamingText, setStreamingText] = useState('')
  const [streamingTools, setStreamingTools] = useState<ToolCallRecord[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [forkHints, setForkHints] = useState<RuntimeForkHint[]>([])
  const [modelLoadStatus, setModelLoadStatus] = useState<string | null>(null)
  const [pendingBuild, setPendingBuild] = useState<{
    engine: 'llama.cpp' | 'mlx-lm' | 'mlx-vlm'
    repository: string
  } | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [localModels, setLocalModels] = useState<LocalModel[]>(() => readCachedModels())
  const [modelsLoading, setModelsLoading] = useState(() => readCachedModels().length === 0)
  const [modelsLoadFailed, setModelsLoadFailed] = useState(false)
  const [selectedModel, setSelectedModel] = useState('')
  const [modelPrepareState, setModelPrepareState] = useState<
    'idle' | 'loading' | 'ready' | 'error'
  >('idle')
  const [modelBindings, setModelBindings] = useState<Record<string, string>>({})
  const [prefetchedRuntimes, setPrefetchedRuntimes] = useState<RuntimeEntry[] | null>(() => {
    const cached = readCachedRuntimes()
    return cached.length > 0 ? cached : null
  })
  const [toolsEnabled, setToolsEnabled] = useState(false)
  const [daemonStatus, setDaemonStatus] = useState<'checking' | 'healthy' | 'offline'>('checking')
  const [daemonVersion, setDaemonVersion] = useState('')
  const [hardware, setHardware] = useState<HardwareInfo | null>(null)
  const [runtime, setRuntime] = useState<RuntimeSettings | null>(null)
  const [savingInference, setSavingInference] = useState(false)

  const [modelMenuOpen, setModelMenuOpen] = useState(false)
  const [inferenceMenuOpen, setInferenceMenuOpen] = useState(false)
  const [manageOpen, setManageOpen] = useState(false)
  const [manageSection, setManageSection] = useState<ManageSection>('library')
  const [runSnapshots, setRunSnapshots] = useState<RunSnapshot[]>([])
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null)

  const abortRef = useRef<AbortController | undefined>(undefined)
  const prepareAbortRef = useRef<AbortController | undefined>(undefined)
  const fileInput = useRef<HTMLInputElement>(null)
  const importInput = useRef<HTMLInputElement>(null)
  const scrollAnchor = useRef<HTMLDivElement>(null)

  const chain = useMemo(() => messageChain(messages, tipId), [messages, tipId])
  const branches = useMemo(() => childCounts(messages), [messages])
  const selectedMeta = useMemo(() => {
    if (modelsLoading && localModels.length === 0) {
      return { title: 'Loading models…', subtitle: 'Scanning local library' }
    }
    return modelDisplayName(selectedModel, localModels.find((m) => m.id === selectedModel))
  }, [selectedModel, localModels, modelsLoading])
  const canChat = Boolean(selectedModel) && modelPrepareState === 'ready'
  const selectedCapabilities = localModels.find((model) => model.id === selectedModel)?.capabilities
  const chatModels = useMemo(() => localModels.filter((model) => isChatModel(model)), [localModels])
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
  const canAttach = canAttachImage || canAttachAudio || canAttachVideo
  const canUseTools = selectedCapabilities?.tools !== false
  const audioBadgeTitle = selectedNativeAudio
    ? 'Native audio: this chat model can consume audio directly (not Whisper ASR)'
    : pipelineFeatures.asr
      ? 'Batch ASR: attachments are transcribed with whisper.cpp, then sent as text'
      : 'No audio path yet — build whisper.cpp + download a Whisper model, or select a native-audio chat model. Realtime PersonaPlex voice is not available yet.'

  async function refreshLocalModels(): Promise<void> {
    const models = await listModels()
    setLocalModels(models)
    writeCachedModels(models)
    setSelectedModel((current) => {
      if (current && models.some((model) => model.id === current)) return current
      return ''
    })
  }

  const selectModel = useCallback((modelId: string): void => {
    if (modelId.startsWith('whisper:')) {
      setError('Whisper models are used for audio transcription, not chat. Pick a chat model instead.')
      return
    }
    prepareAbortRef.current?.abort()
    setSelectedModel(modelId)
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
      onLoad: (event) => setModelLoadStatus(event.message)
    })
      .then(() => {
        if (controller.signal.aborted) return
        setModelPrepareState('ready')
        setModelLoadStatus(null)
        void prefetchRuntimes()
        void fetchModelBindings().then(setModelBindings).catch(() => {})
        void fetchCapabilities()
          .then((payload) => {
            const audio = payload.features.audio_interfaces
            setPipelineFeatures({
              asr: Boolean(payload.features.asr ?? audio?.batch_asr?.available),
              video_preprocess: Boolean(payload.features.video_preprocess),
              whisper_cpp_engine: Boolean(payload.features.whisper_cpp_engine),
              native_model_audio: Boolean(audio?.native_model_audio?.available),
              streaming_asr: Boolean(audio?.streaming_asr?.available),
              realtime_voice: Boolean(audio?.realtime_voice?.available)
            })
          })
          .catch(() => {})
      })
      .catch((cause: unknown) => {
        if ((cause as Error).name === 'AbortError') return
        setModelPrepareState('error')
        setModelLoadStatus(null)
        if (cause instanceof GenerationFailure) {
          setError(cause.message)
          setForkHints(cause.forkHints)
        } else {
          setError(cause instanceof Error ? cause.message : String(cause))
        }
      })
  }, [])

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
    const data = await listConversations(query)
    setConversations(data)
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
    const data = await listMessages(id)
    setMessages(data)
    setTipId(preferredTip ?? data.at(-1)?.id ?? null)
  }

  async function refreshRuntime(): Promise<void> {
    const status = await engineStatus()
    setRuntime(status.settings)
    setHardware(status.hardware)
  }

  useEffect(() => {
    void refreshConversations().catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
    void loadLocalModels().catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
    void prefetchRuntimes()
    void fetchModelBindings().then(setModelBindings).catch(() => {})
    void refreshRuntime().catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
    void fetchCapabilities()
      .then((payload) => {
        const audio = payload.features.audio_interfaces
        setPipelineFeatures({
          asr: Boolean(payload.features.asr ?? audio?.batch_asr?.available),
          video_preprocess: Boolean(payload.features.video_preprocess),
          whisper_cpp_engine: Boolean(payload.features.whisper_cpp_engine),
          native_model_audio: Boolean(audio?.native_model_audio?.available),
          streaming_asr: Boolean(audio?.streaming_asr?.available),
          realtime_voice: Boolean(audio?.realtime_voice?.available)
        })
      })
      .catch(() => {})
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
  }, [chain.length, streamingText])

  async function newConversation(): Promise<void> {
    const conversation = await createConversation()
    await refreshConversations()
    setConversationId(conversation.id)
    setMessages([])
    setTipId(null)
    setDraft('')
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
        return false
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
        setError(`Cannot attach that media yet. Need: ${reasons.join('; ')}.`)
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
    if ((!text && attachments.length === 0) || busy) return
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
    setStreamingTools([])
    const controller = new AbortController()
    abortRef.current = controller

    try {
      let activeConversationId = conversationId
      if (!activeConversationId) {
        const created = await createConversation(text.slice(0, 48) || 'Multimodal conversation')
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
      const toolRecords: ToolCallRecord[] = []
      const maxClientRounds = 4
      for (let round = 0; round < maxClientRounds; round += 1) {
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
            toolChoice: toolsEnabled ? 'auto' : undefined,
            onLoad: (event) => setModelLoadStatus(event.message),
            onToolCall: (record) => {
              toolRecords.push(record)
              setStreamingTools([...toolRecords])
            }
          }
        )
        responseText = result.responseText
        for (const entry of result.transcript) {
          const role = entry.role as Role
          const created = await createMessage(activeConversationId, {
            parent_id: parentId,
            role,
            content: entry.content ?? '',
            tool_calls: entry.tool_calls ?? null,
            tool_call_id: entry.tool_call_id ?? null,
            model: role === 'assistant' ? selectedModel : undefined
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
        model: selectedModel
      })
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
      setStreamingTools([])
      setModelLoadStatus(null)
      await refreshMessages(activeConversationId, assistant.id)
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

  return (
    <main className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="brand">
          <div className="brand-mark">B</div>
          <div>
            <strong>Brazier</strong>
            <span>Local AI workspace</span>
          </div>
        </div>
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
        <div className="conversation-list">
          <div className="section-label">Recent</div>
          {conversations.map((conversation) => (
            <button
              className={conversation.id === conversationId ? 'conversation active' : 'conversation'}
              key={conversation.id}
              onClick={() => setConversationId(conversation.id)}
            >
              <span>{conversation.title}</span>
              <time>{conversation.updated_at.slice(0, 10)}</time>
            </button>
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
        <div className="privacy-note">
          <span className={`status-dot ${daemonStatus}`} />
          {daemonStatus === 'healthy'
            ? `Daemon ${daemonVersion}`
            : daemonStatus === 'offline'
              ? 'Daemon unavailable'
              : 'Checking daemon…'}
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <button className="icon-button" onClick={() => setSidebarOpen((open) => !open)}>
            <Menu size={19} />
          </button>
          <button
            className="model-picker"
            title="Choose which installed model to chat with"
            onClick={() => setModelMenuOpen(true)}
          >
            <div className="model-icon">
              <Box size={16} />
            </div>
            <div>
              <strong>{selectedMeta.title}</strong>
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
          <div className="capabilities">
            <span title="Active acceleration target">
              <Cpu size={14} /> {runtime?.target ?? 'auto'}
            </span>
            <span title="Context window">
              <Gauge size={14} /> {runtime?.context_size?.toLocaleString() ?? '4,096'} ctx
              {selectedCapabilities?.max_context_length
                ? ` · ${selectedCapabilities.max_context_length.toLocaleString()} max`
                : ''}
            </span>
            <span
              className={
                (selectedCapabilities?.reasoning_modes?.length ??
                  (selectedCapabilities?.reasoning ? 1 : 0)) > 0
                  ? ''
                  : 'unavailable'
              }
            >
              <Brain size={14} /> Reasoning
            </span>
            <span
              className={canAttachImage ? '' : 'unavailable'}
              title={visionCapabilityTitle(selectedModel, localModels, canAttachImage)}
            >
              <Image size={14} /> Vision
            </span>
            <span
              className={canAttachAudio ? '' : 'unavailable'}
              title={audioBadgeTitle}
            >
              <AudioLines size={14} />{' '}
              {selectedNativeAudio ? 'Native audio' : pipelineFeatures.asr ? 'ASR' : 'Audio'}
            </span>
            <span
              className={canAttachVideo ? '' : 'unavailable'}
              title={
                canAttachVideo
                  ? 'Video is sampled with ffmpeg and transcribed when ASR is available'
                  : 'Need ffmpeg plus a vision model (and whisper.cpp for soundtrack)'
              }
            >
              <Video size={14} /> Video
            </span>
            <span className={selectedCapabilities?.tools ? '' : 'unavailable'}>
              <Wrench size={14} /> Tools
            </span>
          </div>
          <button
            className="icon-button"
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

        <div className="chat">
          {modelLoadStatus && (modelPrepareState === 'loading' || (busy && !streamingText)) && (
            <div className="model-load-status">
              <LoaderCircle className="spin" size={18} />
              <span>{modelLoadStatus}</span>
            </div>
          )}
          {chain.length === 0 && !streamingText ? (
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
              {chain.map((message) => {
                const toolRecords = toolRecordsFromMessage(message)
                if (toolRecords) {
                  return (
                    <article className="message tool" key={message.id}>
                      <div className="avatar tool-avatar">
                        <Wrench size={15} />
                      </div>
                      <div className="message-body">
                        <div className="message-meta">
                          <strong>Tools</strong>
                        </div>
                        <ToolChips records={toolRecords} />
                      </div>
                    </article>
                  )
                }
                if (message.role === 'tool') return null
                const media = contentMedia(message)
                return (
                  <article className={`message ${message.role}`} key={message.id}>
                    <div className="avatar">{message.role === 'assistant' ? <Bot /> : 'You'}</div>
                    <div className="message-body">
                      <div className="message-meta">
                        <strong>{message.role === 'assistant' ? 'Brazier' : 'You'}</strong>
                        {branches.get(message.id) && branches.get(message.id)! > 1 ? (
                          <span className="branch-count">
                            <GitBranch size={12} /> {branches.get(message.id)} branches
                          </span>
                        ) : null}
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
                      <p>{contentText(message)}</p>
                      <button
                        className="fork-button"
                        title="Continue a new branch from this message"
                        onClick={() => setTipId(message.id)}
                      >
                        <GitBranch size={13} /> Branch here
                      </button>
                    </div>
                  </article>
                )
              })}
              {(streamingText || streamingTools.length > 0) && (
                <article className="message assistant">
                  <div className="avatar">
                    <Bot />
                  </div>
                  <div className="message-body">
                    <div className="message-meta">
                      <strong>Brazier</strong>
                      <LoaderCircle className="spin" size={14} />
                    </div>
                    {streamingTools.length > 0 && <ToolChips records={streamingTools} />}
                    <p>{streamingText}</p>
                  </div>
                </article>
              )}
            </div>
          )}
          <div ref={scrollAnchor} />
        </div>

        <div className="composer-area">
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
          <form className="composer" onSubmit={(event) => void submit(event)}>
            <textarea
              aria-label="Message"
              placeholder={
                !selectedModel
                  ? 'Select a model to start chatting…'
                  : modelPrepareState === 'loading'
                    ? 'Loading model…'
                    : modelPrepareState === 'error'
                      ? 'Fix model load to chat…'
                      : tipId
                        ? 'Continue this branch…'
                        : 'Message a local model…'
              }
              rows={1}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  event.currentTarget.form?.requestSubmit()
                }
              }}
            />
            <div className="composer-actions">
              <input
                ref={fileInput}
                type="file"
                accept={[
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
              <button
                className={toolsEnabled ? 'attach-button tools-on' : 'attach-button'}
                type="button"
                disabled={!canUseTools}
                title={
                  !canUseTools
                    ? 'This model does not advertise tool support'
                    : toolsEnabled
                      ? 'Tools enabled: bundled tools and MCP servers'
                      : 'Enable tools (bundled + MCP)'
                }
                onClick={() => setToolsEnabled((enabled) => !enabled)}
              >
                <Wrench size={17} />
              </button>
              <button
                className="attach-button"
                type="button"
                title={
                  canAttach
                    ? [
                        canAttachImage ? 'images' : null,
                        canAttachAudio ? 'audio' : null,
                        canAttachVideo ? 'video' : null
                      ]
                        .filter(Boolean)
                        .join(', ')
                    : 'Attach media once vision and/or ASR pipelines are available'
                }
                onClick={() => fileInput.current?.click()}
              >
                <Paperclip size={18} />
              </button>
              {busy ? (
                <button
                  className="send-button stop"
                  type="button"
                  title="Stop generation"
                  onClick={() => abortRef.current?.abort()}
                >
                  <Square size={15} fill="currentColor" />
                </button>
              ) : (
                <button
                  className="send-button"
                  type="submit"
                  title="Send"
                  disabled={(!draft.trim() && attachments.length === 0) || !canChat}
                >
                  <Send size={17} />
                </button>
              )}
            </div>
          </form>
          <p className="composer-hint">
            Local models can be inaccurate. Verify important information.
            {toolsEnabled && canUseTools ? ' Tools are enabled (bundled + MCP).' : ''}
            {toolsEnabled && canUseTools && runtime && !runtime.jinja
              ? ' Enable Jinja templates in Engine configuration for reliable tool calling.'
              : ''}
            {selectedCapabilities?.harmony
              ? ' This model uses OpenAI Harmony (gpt-oss); reasoning is routed automatically.'
              : ''}
          </p>
        </div>
      </section>

      {modelMenuOpen && (
        <ModelMenu
          models={chatModels}
          selectedModel={selectedModel}
          loading={modelsLoading}
          onSelect={selectModel}
          onManage={() => openManage(modelsLoadFailed || localModels.length === 0 ? 'discover' : 'library')}
          onClose={() => setModelMenuOpen(false)}
        />
      )}
      {inferenceMenuOpen && (
        <InferenceMenu
          settings={runtime}
          selectedModel={selectedModel}
          models={localModels}
          saving={savingInference}
          onApply={(next) => void applyInferenceSettings(next)}
          onClose={() => setInferenceMenuOpen(false)}
        />
      )}
      {manageOpen && (
        <ManagePanel
          section={manageSection}
          onSectionChange={setManageSection}
          onClose={() => {
            setManageOpen(false)
            void prefetchRuntimes()
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
        />
      )}
    </main>
  )
}

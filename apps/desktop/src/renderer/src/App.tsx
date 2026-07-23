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
import { type ChangeEvent, type FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import {
  createConversation,
  createMessage,
  engineStatus,
  health,
  exportConversation,
  importConversation,
  listConversations,
  listMessages,
  listModels,
  type ConversationExport,
  type HardwareInfo,
  type LocalModel,
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
import { childCounts, messageChain } from './graph'
import type { Attachment, ContentPart, Conversation, Message } from './types'

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

/** Tool records are persisted as a `tool` role message with JSON content. */
function toolRecordsFromMessage(message: Message): ToolCallRecord[] | null {
  if (message.role !== 'tool' || typeof message.content !== 'string') return null
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

function modelLabel(modelId: string, models: LocalModel[]): { title: string; subtitle: string } {
  const match = models.find((model) => model.id === modelId)
  if (modelId.startsWith('gguf:')) {
    const file = modelId.slice('gguf:'.length).split('/').at(-1) ?? modelId
    return {
      title: file,
      subtitle: match ? `${match.owned_by.replace('brazier:', '')} · Local GGUF` : 'Local GGUF'
    }
  }
  if (!modelId) {
    return { title: 'Select a model', subtitle: 'Download a GGUF to get started' }
  }
  return { title: modelId, subtitle: match?.owned_by ?? 'Local' }
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
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [localModels, setLocalModels] = useState<LocalModel[]>([])
  const [modelsLoading, setModelsLoading] = useState(true)
  const [modelsLoadFailed, setModelsLoadFailed] = useState(false)
  const [selectedModel, setSelectedModel] = useState('')
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

  const abortRef = useRef<AbortController | undefined>(undefined)
  const fileInput = useRef<HTMLInputElement>(null)
  const importInput = useRef<HTMLInputElement>(null)
  const scrollAnchor = useRef<HTMLDivElement>(null)

  const chain = useMemo(() => messageChain(messages, tipId), [messages, tipId])
  const branches = useMemo(() => childCounts(messages), [messages])
  const selectedMeta = useMemo(() => {
    if (modelsLoading) return { title: 'Loading models…', subtitle: 'Starting local daemon' }
    return modelLabel(selectedModel, localModels)
  }, [selectedModel, localModels, modelsLoading])
  const canChat = Boolean(selectedModel)
  const selectedCapabilities = localModels.find((model) => model.id === selectedModel)?.capabilities
  const canAttach = Boolean(
    selectedCapabilities?.input_modalities.some((modality) =>
      ['image', 'audio', 'video'].includes(modality)
    )
  )

  async function refreshLocalModels(): Promise<void> {
    const models = await listModels()
    setLocalModels(models)
    setSelectedModel((current) => {
      if (current && models.some((model) => model.id === current)) return current
      return models[0]?.id ?? ''
    })
  }

  async function loadLocalModels(): Promise<void> {
    setModelsLoading(true)
    setModelsLoadFailed(false)
    try {
      await refreshLocalModels()
    } catch (cause) {
      setModelsLoadFailed(true)
      throw cause
    } finally {
      setModelsLoading(false)
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
    void refreshRuntime().catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
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
      return
    }
    setError(null)
    refreshMessages(conversationId).catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
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
      const saved = await saveRuntimeSettings(next)
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

  async function selectFiles(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    try {
      const files = Array.from(event.target.files ?? [])
      const accepted = files.filter(
        (file) =>
          file.type.startsWith('image/') ||
          file.type.startsWith('audio/') ||
          file.type.startsWith('video/')
      )
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
      setError('Select or download a local GGUF model first.')
      setModelMenuOpen(true)
      return
    }
    setBusy(true)
    setError(null)
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
      const requestMessages = [...chain, userMessage]
      setMessages((current) => [...current, userMessage])
      setTipId(userMessage.id)
      setDraft('')
      setAttachments([])

      let responseText = ''
      const toolRecords: ToolCallRecord[] = []
      await streamCompletion(
        requestMessages,
        selectedModel,
        controller.signal,
        (token) => {
          responseText += token
          setStreamingText(responseText)
        },
        {
          builtinTools: toolsEnabled,
          onToolCall: (record) => {
            toolRecords.push(record)
            setStreamingTools([...toolRecords])
          }
        }
      )
      let parentId = userMessage.id
      if (toolRecords.length > 0) {
        const toolMessage = await createMessage(activeConversationId, {
          parent_id: parentId,
          role: 'tool',
          content: JSON.stringify({ brazier_tool_calls: toolRecords })
        })
        parentId = toolMessage.id
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
      await refreshMessages(activeConversationId, assistant.id)
      await refreshConversations()
    } catch (cause) {
      if ((cause as Error).name !== 'AbortError') {
        setError(cause instanceof Error ? cause.message : String(cause))
      }
    } finally {
      setBusy(false)
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
              <Gauge size={14} /> {runtime?.context_size ?? 4096} ctx
            </span>
            <span className={selectedCapabilities?.reasoning ? '' : 'unavailable'}>
              <Brain size={14} /> Reasoning
            </span>
            <span
              className={selectedCapabilities?.input_modalities.includes('image') ? '' : 'unavailable'}
              title={canAttach ? 'Multimodal projector installed' : 'Install the model mmproj GGUF'}
            >
              <Image size={14} /> Vision
            </span>
            <span
              className={selectedCapabilities?.input_modalities.includes('audio') ? '' : 'unavailable'}
            >
              <AudioLines size={14} /> Audio
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

        <div className="chat">
          {chain.length === 0 && !streamingText ? (
            <div className="welcome">
              <div className="welcome-mark">
                <Bot size={30} />
              </div>
              <h1>What are we exploring?</h1>
              <p>
                {modelsLoading
                  ? 'Starting the local runtime and loading your model library…'
                  : canChat
                    ? 'Chat privately with local models. Attach media or start with a question.'
                    : 'Download a GGUF from Hugging Face to start chatting with a local model.'}
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
                      Find a GGUF on Hugging Face
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
              <span>{error}</span>
              <button onClick={() => setError(null)}>
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
                !canChat
                  ? 'Select a model to start chatting…'
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
                accept="image/*,audio/*,video/*"
                multiple
                hidden
                onChange={(event) => void selectFiles(event)}
              />
              <button
                className={toolsEnabled ? 'attach-button tools-on' : 'attach-button'}
                type="button"
                title={
                  toolsEnabled
                    ? 'Bundled tools enabled: time, calculator, web fetch, JS sandbox'
                    : 'Enable bundled tools (time, calculator, web fetch, JS sandbox)'
                }
                onClick={() => setToolsEnabled((enabled) => !enabled)}
              >
                <Wrench size={17} />
              </button>
              <button
                className="attach-button"
                type="button"
                title={
                  canAttach ? 'Attach media' : 'Attach media (the model may require an mmproj GGUF)'
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
            {toolsEnabled ? ' Bundled tools are enabled for this chat.' : ''}
          </p>
        </div>
      </section>

      {modelMenuOpen && (
        <ModelMenu
          models={localModels}
          selectedModel={selectedModel}
          loading={modelsLoading}
          onSelect={setSelectedModel}
          onManage={() => openManage(modelsLoadFailed || localModels.length === 0 ? 'discover' : 'library')}
          onClose={() => setModelMenuOpen(false)}
        />
      )}
      {inferenceMenuOpen && (
        <InferenceMenu
          settings={runtime}
          saving={savingInference}
          onApply={(next) => void applyInferenceSettings(next)}
          onClose={() => setInferenceMenuOpen(false)}
        />
      )}
      {manageOpen && (
        <ManagePanel
          section={manageSection}
          onSectionChange={setManageSection}
          onClose={() => setManageOpen(false)}
          models={localModels}
          modelsLoading={modelsLoading}
          refreshModels={refreshLocalModels}
          selectedModel={selectedModel}
          onSelectModel={setSelectedModel}
          settings={runtime}
          onSettingsSaved={setRuntime}
          hardware={hardware}
        />
      )}
    </main>
  )
}

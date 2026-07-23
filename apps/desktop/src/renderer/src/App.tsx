import {
  AudioLines,
  Bot,
  Box,
  Brain,
  GitBranch,
  Image,
  LoaderCircle,
  Menu,
  MessageSquarePlus,
  Paperclip,
  Search,
  Send,
  ShieldAlert,
  Square,
  Video,
  X
} from 'lucide-react'
import { type ChangeEvent, type FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import {
  createConversation,
  createMessage,
  listConversations,
  listMessages,
  searchHub,
  streamCompletion
} from './api'
import { childCounts, messageChain } from './graph'
import type { Attachment, ContentPart, Conversation, HubModel, Message } from './types'

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
    if (part.type === 'image_url') return ['image'] as const
    if (part.type === 'input_audio') return ['audio'] as const
    if (part.type === 'input_video') return ['video'] as const
    return []
  })
}

function fileToAttachment(file: File): Promise<Attachment> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error ?? new Error('Could not read attachment.'))
    reader.onload = () =>
      resolve({
        id: crypto.randomUUID(),
        name: file.name,
        type: file.type,
        dataUrl: String(reader.result)
      })
    reader.readAsDataURL(file)
  })
}

function attachmentPart(attachment: Attachment): ContentPart {
  if (attachment.type.startsWith('image/')) {
    return { type: 'image_url', image_url: { url: attachment.dataUrl } }
  }
  if (attachment.type.startsWith('audio/')) {
    return {
      type: 'input_audio',
      input_audio: {
        data: attachment.dataUrl.split(',')[1] ?? attachment.dataUrl,
        format: attachment.type.split('/')[1] ?? 'wav'
      }
    }
  }
  return { type: 'input_video', video_url: { url: attachment.dataUrl } }
}

export function App(): React.JSX.Element {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [tipId, setTipId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [streamingText, setStreamingText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [modelBrowserOpen, setModelBrowserOpen] = useState(false)
  const [modelQuery, setModelQuery] = useState('Qwen')
  const [modelEngine, setModelEngine] = useState('llama.cpp')
  const [hubModels, setHubModels] = useState<HubModel[]>([])
  const [searchingModels, setSearchingModels] = useState(false)
  const abortRef = useRef<AbortController | undefined>(undefined)
  const fileInput = useRef<HTMLInputElement>(null)
  const scrollAnchor = useRef<HTMLDivElement>(null)

  const chain = useMemo(() => messageChain(messages, tipId), [messages, tipId])
  const branches = useMemo(() => childCounts(messages), [messages])

  async function refreshConversations(): Promise<void> {
    const data = await listConversations()
    setConversations(data)
    if (!conversationId && data[0]) setConversationId(data[0].id)
  }

  async function refreshMessages(id: string, preferredTip?: string): Promise<void> {
    const data = await listMessages(id)
    setMessages(data)
    setTipId(preferredTip ?? data.at(-1)?.id ?? null)
  }

  useEffect(() => {
    refreshConversations().catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause))
    )
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

  async function findModels(): Promise<void> {
    setSearchingModels(true)
    setError(null)
    try {
      setHubModels(await searchHub(modelQuery.trim(), modelEngine))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSearchingModels(false)
    }
  }

  async function selectFiles(event: ChangeEvent<HTMLInputElement>): Promise<void> {
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
  }

  async function submit(event: FormEvent): Promise<void> {
    event.preventDefault()
    const text = draft.trim()
    if ((!text && attachments.length === 0) || busy) return
    setBusy(true)
    setError(null)
    setStreamingText('')
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
      await streamCompletion(requestMessages, controller.signal, (token) => {
        responseText += token
        setStreamingText(responseText)
      })
      const assistant = await createMessage(activeConversationId, {
        parent_id: userMessage.id,
        role: 'assistant',
        content: responseText,
        model: 'brazier/mock'
      })
      setStreamingText('')
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
          <div className="brand-mark">H</div>
          <div>
            <strong>Brazier</strong>
            <span>Local AI workspace</span>
          </div>
        </div>
        <button className="new-chat" onClick={() => void newConversation()}>
          <MessageSquarePlus size={17} />
          New conversation
        </button>
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
          <span className="status-dot" />
          Local daemon connected
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <button className="icon-button" onClick={() => setSidebarOpen((open) => !open)}>
            <Menu size={19} />
          </button>
          <button className="model-picker" onClick={() => setModelBrowserOpen(true)}>
            <div className="model-icon">
              <Box size={16} />
            </div>
            <div>
              <strong>Brazier Development Model</strong>
              <span>Mock engine · Local</span>
            </div>
          </button>
          <div className="capabilities">
            <span>
              <Brain size={14} /> Reasoning
            </span>
            <span>
              <Image size={14} /> Vision
            </span>
            <span>
              <AudioLines size={14} /> Audio
            </span>
            <span>
              <Video size={14} /> Video
            </span>
          </div>
        </header>

        <div className="chat">
          {chain.length === 0 && !streamingText ? (
            <div className="welcome">
              <div className="welcome-mark">
                <Bot size={30} />
              </div>
              <h1>What are we exploring?</h1>
              <p>
                Chat privately with local models. Attach an image, recording, or video—or start with
                a simple question.
              </p>
              <div className="starter-grid">
                <button onClick={() => setDraft('Explain how speculative decoding works.')}>
                  <Brain size={18} />
                  <span>
                    <strong>Explore a concept</strong>
                    Speculative decoding
                  </span>
                </button>
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
              {streamingText && (
                <article className="message assistant">
                  <div className="avatar">
                    <Bot />
                  </div>
                  <div className="message-body">
                    <div className="message-meta">
                      <strong>Brazier</strong>
                      <LoaderCircle className="spin" size={14} />
                    </div>
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
              placeholder={tipId ? 'Continue this branch…' : 'Message a local model…'}
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
                className="attach-button"
                type="button"
                title="Attach media"
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
                  disabled={!draft.trim() && attachments.length === 0}
                >
                  <Send size={17} />
                </button>
              )}
            </div>
          </form>
          <p className="composer-hint">
            Local models can be inaccurate. Verify important information.
          </p>
        </div>
      </section>
      {modelBrowserOpen && (
        <div className="drawer-backdrop" onMouseDown={() => setModelBrowserOpen(false)}>
          <aside className="model-drawer" onMouseDown={(event) => event.stopPropagation()}>
            <div className="drawer-heading">
              <div>
                <span className="eyebrow">Hugging Face</span>
                <h2>Discover models</h2>
              </div>
              <button className="icon-button" onClick={() => setModelBrowserOpen(false)}>
                <X size={18} />
              </button>
            </div>
            <form
              className="model-search"
              onSubmit={(event) => {
                event.preventDefault()
                void findModels()
              }}
            >
              <label>
                <Search size={16} />
                <input
                  aria-label="Search Hugging Face"
                  value={modelQuery}
                  onChange={(event) => setModelQuery(event.target.value)}
                  placeholder="Model name or author"
                />
              </label>
              <select value={modelEngine} onChange={(event) => setModelEngine(event.target.value)}>
                <option value="llama.cpp">llama.cpp</option>
                <option value="mlx-lm">MLX-LM</option>
                <option value="mlx-vlm">MLX-VLM</option>
                <option value="vllm">vLLM</option>
              </select>
              <button type="submit" disabled={searchingModels || !modelQuery.trim()}>
                {searchingModels ? <LoaderCircle className="spin" size={15} /> : 'Search'}
              </button>
            </form>
            <p className="model-help">
              Results are compatibility-filtered before Unsloth quantizations receive a preference
              boost.
            </p>
            <div className="model-results">
              {hubModels.map((model) => (
                <article className="model-card" key={model.id}>
                  <div>
                    <strong>{model.id.split('/').at(-1)}</strong>
                    <span>{model.author}</span>
                  </div>
                  <div className="model-badges">
                    {model.preferred_quantizer && <span className="unsloth">Unsloth preferred</span>}
                    {model.gated && (
                      <span>
                        <ShieldAlert size={11} /> Gated
                      </span>
                    )}
                    <span>{model.downloads.toLocaleString()} downloads</span>
                  </div>
                  <button disabled title="Managed downloads are the next implementation milestone">
                    Download
                  </button>
                </article>
              ))}
              {!searchingModels && hubModels.length === 0 && (
                <div className="empty-models">
                  <Search size={24} />
                  <p>Search the Hub for artifacts compatible with your selected engine.</p>
                </div>
              )}
            </div>
          </aside>
        </div>
      )}
    </main>
  )
}

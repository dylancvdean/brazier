import {
  Box,
  Compass,
  MessageSquare,
  Search,
  Settings2,
  SlidersHorizontal,
  Terminal,
  X
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { LocalModel, WorkspaceModesPreference } from '../api'
import { useDialogOverlay } from '../hooks/useDialogOverlay'
import {
  isChatModel,
  isComputerUseModel,
  isImageGenModel,
  isVideoGenModel,
  isVoiceModel,
  modelDisplayName
} from '../model-utils'
import type { Conversation } from '../types'
import type { ManageSection } from './ManagePanel'

export type CommandPaletteCategory = 'actions' | 'models' | 'conversations' | 'manage'

export type CommandPaletteItem = {
  id: string
  category: CommandPaletteCategory
  title: string
  subtitle?: string
  keywords?: string[]
  shortcut?: string
  active?: boolean
  onSelect: () => void
  secondaryAction?: {
    label: string
    onSelect: () => void
  }
}

export type ManagePaletteDescriptor = {
  section: ManageSection
  title: string
  subtitle: string
  keywords: string[]
}

export const MANAGE_PALETTE_ENTRIES: readonly ManagePaletteDescriptor[] = [
  {
    section: 'library',
    title: 'Manage · Model library',
    subtitle: 'Installed weights, external folders, projectors, disk usage',
    keywords: [
      'models',
      'library',
      'disk',
      'space',
      'delete',
      'gguf',
      'mlx',
      'mmproj',
      'projector',
      'lora',
      'ollama weights',
      'lm studio',
      'external folder'
    ]
  },
  {
    section: 'recommended',
    title: 'Manage · Recommended models',
    subtitle: 'Curated models sized to your VRAM and system memory',
    keywords: ['recommended', 'starter', 'tier', 'vram', 'ram', 'memory', 'setup']
  },
  {
    section: 'discover',
    title: 'Manage · Download models',
    subtitle: 'Search Hugging Face for GGUF, MLX, sd.cpp, Whisper, and PersonaPlex',
    keywords: ['download', 'discover', 'huggingface', 'hf', 'hub', 'flux', 'wan', 'whisper']
  },
  {
    section: 'runtimes',
    title: 'Manage · Runtimes',
    subtitle: 'Install or compile llama.cpp, MLX, vLLM, sd.cpp, Whisper, PersonaPlex',
    keywords: [
      'runtimes',
      'llama.cpp',
      'mlx',
      'vllm',
      'sd.cpp',
      'whisper',
      'cuda',
      'metal',
      'vulkan',
      'rocm',
      'build',
      'compile',
      'fork',
      'system path'
    ]
  },
  {
    section: 'engine',
    title: 'Manage · Engine configuration',
    subtitle: 'GPU layers, KV cache quantisation, flash attention, JS sandbox, memory policy',
    keywords: [
      'engine',
      'vram',
      'kv cache',
      'flash attention',
      'gpu layers',
      'offload',
      'threads',
      'batch',
      'jinja',
      'sandbox',
      'javascript',
      'headroom'
    ]
  },
  {
    section: 'chat',
    title: 'Manage · Chat & memory',
    subtitle: 'Default chat model, long-term memory recall, and dreaming schedule',
    keywords: ['chat', 'memory', 'memories', 'dreaming', 'dream', 'recall', 'pinned']
  },
  {
    section: 'agent',
    title: 'Manage · Agent',
    subtitle: 'Simple vs Powerful harness, power tools, and OS sandbox trust boundary',
    keywords: ['agent', 'sandbox', 'seatbelt', 'bubblewrap', 'powerful', 'simple', 'subagent']
  },
  {
    section: 'computer',
    title: 'Manage · Computer use',
    subtitle: 'Fara1.5 bundles, desktop screen/input permissions, settle delay',
    keywords: ['computer', 'fara', 'cua', 'browser', 'desktop', 'screenshot', 'wayland', 'permissions']
  },
  {
    section: 'generate',
    title: 'Manage · Generate',
    subtitle: 'Default image and video models and generation timeout',
    keywords: ['generate', 'image', 'video', 'diffusion', 'sd.cpp', 'timeout']
  },
  {
    section: 'voice',
    title: 'Manage · Voice',
    subtitle: 'Default PersonaPlex voice model and standing persona',
    keywords: ['voice', 'personaplex', 'moshi', 'persona', 'speech', 'microphone']
  },
  {
    section: 'websearch',
    title: 'Manage · Web search',
    subtitle: 'DuckDuckGo or Brave Search provider, SafeSearch, and API key',
    keywords: ['web search', 'duckduckgo', 'brave', 'api key', 'safesearch', 'internet']
  },
  {
    section: 'mcp',
    title: 'Manage · MCP servers',
    subtitle: 'Local Model Context Protocol stdio servers and tool refresh',
    keywords: ['mcp', 'model context protocol', 'tools', 'stdio', 'servers', 'integrations']
  },
  {
    section: 'remote',
    title: 'Manage · Remote servers',
    subtitle: 'Connect Ollama, vLLM, llama-server, LM Studio, or OpenAI-compatible endpoints',
    keywords: [
      'remote',
      'ollama',
      'vllm',
      'lm studio',
      'openai',
      'llama-server',
      'endpoint',
      'base url',
      'api key'
    ]
  },
  {
    section: 'server',
    title: 'Manage · OpenAI server',
    subtitle: 'Expose Brazier on localhost/LAN, manage API keys, and JIT model loading',
    keywords: ['server', 'openai', 'api key', 'port', 'localhost', 'curl', 'jit', 'expose']
  },
  {
    section: 'connections',
    title: 'Manage · Connections',
    subtitle: 'Switch between this computer and paired remote Brazier daemons',
    keywords: ['connections', 'daemon', 'pair', 'pairing', 'profile', 'host']
  },
  {
    section: 'customization',
    title: 'Manage · Customization',
    subtitle: 'Visible workspace modes, automatic chat titles, and app updates',
    keywords: ['customization', 'modes', 'tabs', 'updates', 'auto title', 'preferences']
  },
  {
    section: 'support',
    title: 'Manage · Support',
    subtitle: 'Export a privacy-filtered diagnostic bundle and logs',
    keywords: ['support', 'logs', 'diagnostics', 'bundle', 'debug', 'troubleshoot']
  }
]

const CATEGORY_LABELS: Record<CommandPaletteCategory, string> = {
  actions: 'Modes & actions',
  models: 'Installed models',
  conversations: 'Recent conversations',
  manage: 'Manage & settings'
}

const CATEGORY_PRIORITY: Record<CommandPaletteCategory, number> = {
  actions: 4,
  models: 3,
  manage: 2,
  conversations: 1
}

function subsequenceScore(needle: string, haystack: string): number {
  if (needle.length < 2 || needle.length > haystack.length) return 0
  let nIdx = 0
  let firstMatch = -1
  let lastMatch = -1
  for (let hIdx = 0; hIdx < haystack.length && nIdx < needle.length; hIdx += 1) {
    if (haystack[hIdx] === needle[nIdx]) {
      if (firstMatch === -1) firstMatch = hIdx
      lastMatch = hIdx
      nIdx += 1
    }
  }
  if (nIdx < needle.length) return 0
  const span = lastMatch - firstMatch + 1
  // Reject overly scattered character matches so unrelated long titles don't match.
  if (span > needle.length * 3) return 0
  return Math.max(8, 26 - (span - needle.length) * 2)
}

function tokenMatchScore(
  token: string,
  title: string,
  subtitle: string,
  keywords: string[]
): number {
  if (title === token) return 140
  if (title.startsWith(token)) return 105
  const titleWords = title.split(/[\s·/_:.-]+/)
  if (titleWords.some((word) => word === token)) return 92
  if (titleWords.some((word) => word.startsWith(token))) return 80
  if (title.includes(token)) return 65

  for (const kw of keywords) {
    if (kw === token) return 78
    if (kw.startsWith(token)) return 68
    if (kw.split(/[\s/_:.-]+/).some((part) => part.startsWith(token))) return 60
    if (kw.includes(token)) return 52
  }

  if (subtitle.includes(token)) return 42

  const fuzzyTitle = subsequenceScore(token, title)
  if (fuzzyTitle > 0) return fuzzyTitle
  for (const kw of keywords) {
    const fuzzyKw = subsequenceScore(token, kw)
    if (fuzzyKw > 0) return Math.max(6, fuzzyKw - 4)
  }
  return 0
}

/**
 * Score a palette item against a search query. Returns `0` when the item does
 * not match every query token.
 */
export function scorePaletteItem(item: CommandPaletteItem, rawQuery: string): number {
  const normalized = rawQuery.trim().toLowerCase()
  if (!normalized) return 1
  const title = item.title.toLowerCase()
  const subtitle = (item.subtitle ?? '').toLowerCase()
  const keywords = (item.keywords ?? []).map((kw) => kw.toLowerCase())

  // Multi-word phrase bonus when the full query appears verbatim in a keyword or title.
  let phraseBonus = 0
  if (normalized.includes(' ')) {
    if (title.includes(normalized)) phraseBonus = 45
    else if (keywords.some((kw) => kw === normalized)) phraseBonus = 50
    else if (keywords.some((kw) => kw.includes(normalized))) phraseBonus = 35
    else if (subtitle.includes(normalized)) phraseBonus = 25
  }

  const tokens = normalized.split(/\s+/).filter(Boolean)
  let total = phraseBonus
  for (const token of tokens) {
    const score = tokenMatchScore(token, title, subtitle, keywords)
    if (score <= 0) return 0
    total += score
  }
  if (item.active) total += 6
  total += CATEGORY_PRIORITY[item.category]
  return total
}

export function filterAndRankPaletteItems(
  items: readonly CommandPaletteItem[],
  query: string
): CommandPaletteItem[] {
  if (!query.trim()) return [...items]
  const scored: Array<{ item: CommandPaletteItem; score: number; index: number }> = []
  items.forEach((item, index) => {
    const score = scorePaletteItem(item, query)
    if (score > 0) scored.push({ item, score, index })
  })
  scored.sort((left, right) => right.score - left.score || left.index - right.index)
  return scored.map((entry) => entry.item)
}

type AppMode = 'chat' | 'agent' | 'generate' | 'voice' | 'computer'

type CommandPaletteProps = {
  open: boolean
  appMode: AppMode
  workspaceModes: WorkspaceModesPreference
  models: LocalModel[]
  selectedModel: string
  voiceModel: string
  generateModel: string
  computerModel: string
  conversations: Conversation[]
  activeConversationId: string | null
  incognito: boolean
  canUnload: boolean
  isMac?: boolean
  onClose: () => void
  onSwitchMode: (mode: AppMode) => void
  onSelectModelForApp: (model: LocalModel) => void
  onConfigureModel: (modelId: string) => void
  onSelectConversation: (conversationId: string) => void
  onNewAction: () => void
  onToggleIncognito: () => void
  onUnloadModel: () => void
  onExportConversation: () => void
  onCopyApiCurl: () => void
  onOpenModelMenu: () => void
  onOpenInferenceSettings: () => void
  onToggleSidebar: () => void
  onRunDream: () => void
  onOpenManage: (section: ManageSection) => void
}

export function CommandPalette(props: CommandPaletteProps): React.JSX.Element | null {
  const [query, setQuery] = useState('')
  const [activeIndex, setActiveIndex] = useState(0)
  const listRef = useRef<HTMLDivElement>(null)
  const overlayRef = useDialogOverlay(props.open, props.onClose)
  const mod =
    (props.isMac ?? (typeof navigator !== 'undefined' && /mac/i.test(navigator.platform)))
      ? '⌘'
      : 'Ctrl+'

  useEffect(() => {
    if (props.open) {
      setQuery('')
      setActiveIndex(0)
    }
  }, [props.open])

  const allItems = useMemo((): CommandPaletteItem[] => {
    const run = (fn: () => void) => (): void => {
      props.onClose()
      fn()
    }

    const actions: CommandPaletteItem[] = [
      {
        id: 'action:new',
        category: 'actions',
        title:
          props.appMode === 'agent'
            ? 'New agent task'
            : props.appMode === 'computer'
              ? 'New computer session'
              : props.appMode === 'generate'
                ? 'New generation'
                : 'New chat',
        subtitle: 'Start fresh in the active mode',
        shortcut: `${mod}N`,
        keywords: ['new', 'create', 'clear', 'reset', 'chat', 'task'],
        onSelect: run(props.onNewAction)
      },
      {
        id: 'action:choose-model',
        category: 'actions',
        title: 'Choose model…',
        subtitle: 'Open the top-bar model picker',
        shortcut: `${mod}${mod === '⌘' ? '⇧M' : 'Shift+M'}`,
        keywords: ['model', 'picker', 'switch model'],
        onSelect: run(props.onOpenModelMenu)
      },
      {
        id: 'action:inference-settings',
        category: 'actions',
        title: 'Inference settings…',
        subtitle: 'Context window, sampling presets, reasoning, and residency',
        keywords: ['sampling', 'temperature', 'top_p', 'context', 'reasoning', 'residency', 'vram'],
        onSelect: run(props.onOpenInferenceSettings)
      },
      {
        id: 'action:incognito',
        category: 'actions',
        title: props.incognito ? 'Leave incognito chat' : 'Toggle incognito chat',
        subtitle: 'Private ephemeral chat with memory off',
        shortcut: `${mod}${mod === '⌘' ? '⇧I' : 'Shift+I'}`,
        keywords: ['incognito', 'private', 'ephemeral'],
        onSelect: run(props.onToggleIncognito)
      }
    ]

    if (props.canUnload) {
      actions.push({
        id: 'action:unload',
        category: 'actions',
        title: 'Unload model',
        subtitle: 'Stop the resident local chat model and free VRAM',
        keywords: ['unload', 'eject', 'free', 'vram', 'memory', 'stop server'],
        onSelect: run(props.onUnloadModel)
      })
    }

    if (props.activeConversationId && !props.incognito) {
      actions.push({
        id: 'action:export',
        category: 'actions',
        title: 'Export conversation',
        subtitle: 'Save the current chat transcript as JSON',
        keywords: ['export', 'download', 'json', 'save chat', 'backup'],
        onSelect: run(props.onExportConversation)
      })
    }

    actions.push(
      {
        id: 'action:copy-curl',
        category: 'actions',
        title: 'Copy API curl',
        subtitle: 'Copy an OpenAI-compatible /v1/chat/completions curl command',
        keywords: ['curl', 'api', 'openai', 'http', 'snippet', 'endpoint'],
        onSelect: run(props.onCopyApiCurl)
      },
      {
        id: 'action:sidebar',
        category: 'actions',
        title: 'Toggle sidebar',
        subtitle: 'Show or hide the left history panel',
        shortcut: `${mod}B`,
        keywords: ['sidebar', 'panel', 'history'],
        onSelect: run(props.onToggleSidebar)
      }
    )

    if (props.selectedModel) {
      actions.push({
        id: 'action:dream',
        category: 'actions',
        title: 'Run memory dreaming',
        subtitle: 'Consolidate and prune memories from recent conversations',
        keywords: ['dream', 'dreaming', 'memory', 'consolidate'],
        onSelect: run(props.onRunDream)
      })
    }

    const orderedModes: Array<[AppMode, string]> = [
      ['chat', 'Chat'],
      ['agent', 'Agent'],
      ['generate', 'Generate'],
      ['voice', 'Voice'],
      ['computer', 'Computer']
    ]
    const enabledModes = orderedModes.filter(([id]) => props.workspaceModes[id])
    enabledModes.forEach(([modeId, label], idx) => {
      actions.push({
        id: `mode:${modeId}`,
        category: 'actions',
        title: `Switch to ${label}`,
        subtitle: props.appMode === modeId ? 'Active workspace mode' : 'Workspace mode',
        shortcut: `${mod}${idx + 1}`,
        active: props.appMode === modeId,
        keywords: ['mode', 'tab', 'workspace', label.toLowerCase()],
        onSelect: run(() => props.onSwitchMode(modeId))
      })
    })

    const selectableModels = props.models.filter(
      (model) => !model.id.startsWith('whisper:') && !model.id.startsWith('streaming-asr:')
    )
    const modelItems: CommandPaletteItem[] = selectableModels.map((model) => {
      const meta = modelDisplayName(model.id, model)
      const isCurrent =
        (props.appMode === 'voice' && model.id === props.voiceModel) ||
        (props.appMode === 'generate' && model.id === props.generateModel) ||
        (props.appMode === 'computer' && model.id === props.computerModel) ||
        ((props.appMode === 'chat' || props.appMode === 'agent') &&
          model.id === props.selectedModel)
      const modeHint = isImageGenModel(model)
        ? 'Image'
        : isVideoGenModel(model)
          ? 'Video'
          : isVoiceModel(model)
            ? 'Voice'
            : isComputerUseModel(model) && props.appMode === 'computer'
              ? 'Computer'
              : isChatModel(model)
                ? 'Chat / Agent'
                : ''
      return {
        id: `model:${model.id}`,
        category: 'models',
        title: meta.title,
        subtitle: [meta.subtitle, modeHint].filter(Boolean).join(' · '),
        active: isCurrent,
        keywords: [
          model.id,
          model.engine ?? '',
          model.repo_id ?? '',
          model.library_label ?? '',
          'model',
          'use',
          'select',
          'configure',
          'lora'
        ].filter(Boolean),
        onSelect: run(() => props.onSelectModelForApp(model)),
        secondaryAction: {
          label: 'Configure',
          onSelect: run(() => props.onConfigureModel(model.id))
        }
      }
    })

    const conversationItems: CommandPaletteItem[] = props.conversations.slice(0, 25).map((conv) => ({
      id: `conv:${conv.id}`,
      category: 'conversations',
      title: conv.title || 'Untitled chat',
      subtitle: 'Conversation',
      active: !props.incognito && conv.id === props.activeConversationId,
      keywords: ['chat', 'conversation', 'history'],
      onSelect: run(() => props.onSelectConversation(conv.id))
    }))

    const manageItems: CommandPaletteItem[] = MANAGE_PALETTE_ENTRIES.map((entry) => ({
      id: `manage:${entry.section}`,
      category: 'manage',
      title: entry.title,
      subtitle: entry.subtitle,
      shortcut: entry.section === 'library' ? `${mod},` : undefined,
      keywords: ['manage', 'settings', 'preferences', ...entry.keywords],
      onSelect: run(() => props.onOpenManage(entry.section))
    }))

    return [...actions, ...modelItems, ...conversationItems, ...manageItems]
  }, [props, mod])

  const visibleItems = useMemo(() => {
    if (!query.trim()) {
      // Curated compact default overview when opened with an empty query.
      const actions = allItems.filter((item) => item.category === 'actions').slice(0, 8)
      const models = allItems.filter((item) => item.category === 'models').slice(0, 6)
      const convs = allItems.filter((item) => item.category === 'conversations').slice(0, 5)
      const manage = allItems.filter((item) => item.category === 'manage').slice(0, 6)
      return [...actions, ...models, ...convs, ...manage]
    }
    return filterAndRankPaletteItems(allItems, query).slice(0, 40)
  }, [allItems, query])

  useEffect(() => {
    setActiveIndex((current) =>
      visibleItems.length === 0 ? 0 : Math.min(current, visibleItems.length - 1)
    )
  }, [visibleItems])

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(
      `[data-palette-index="${activeIndex}"]`
    )
    el?.scrollIntoView({ block: 'nearest' })
  }, [activeIndex])

  if (!props.open) return null

  const isSearching = query.trim().length > 0

  function categoryIcon(category: CommandPaletteCategory): React.JSX.Element {
    switch (category) {
      case 'models':
        return <Box size={14} />
      case 'conversations':
        return <MessageSquare size={14} />
      case 'manage':
        return <Settings2 size={14} />
      default:
        return <Terminal size={14} />
    }
  }

  function handleInputKeyDown(event: React.KeyboardEvent<HTMLInputElement>): void {
    if (event.key === 'ArrowDown') {
      if (visibleItems.length === 0) return
      event.preventDefault()
      setActiveIndex((current) => (current + 1) % visibleItems.length)
      return
    }
    if (event.key === 'ArrowUp') {
      if (visibleItems.length === 0) return
      event.preventDefault()
      setActiveIndex((current) => (current - 1 + visibleItems.length) % visibleItems.length)
      return
    }
    if (event.key === 'Enter') {
      const target = visibleItems[activeIndex]
      if (!target) return
      event.preventDefault()
      if (event.shiftKey && target.secondaryAction) {
        target.secondaryAction.onSelect()
      } else {
        target.onSelect()
      }
    }
  }

  return (
    <div
      className="command-palette-backdrop"
      onMouseDown={props.onClose}
      ref={overlayRef}
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      <div className="command-palette" onMouseDown={(event) => event.stopPropagation()}>
        <div className="command-palette-input-row">
          <Search size={15} />
          <input
            data-autofocus
            value={query}
            onChange={(event) => {
              setQuery(event.target.value)
              setActiveIndex(0)
            }}
            onKeyDown={handleInputKeyDown}
            placeholder="Search models, chats, actions, or settings (vram, lora, mcp, ollama…)"
            aria-label="Command palette search"
            spellCheck={false}
          />
          {query ? (
            <button
              type="button"
              className="icon-button subtle"
              aria-label="Clear query"
              onClick={() => {
                setQuery('')
                setActiveIndex(0)
              }}
            >
              <X size={13} />
            </button>
          ) : null}
        </div>

        <div className="command-palette-list" ref={listRef} role="listbox">
          {visibleItems.length === 0 ? (
            <div className="command-palette-empty">
              <Compass size={16} />
              <span>No matches for “{query.trim()}”.</span>
            </div>
          ) : (
            visibleItems.map((item, index) => {
              const previous = index > 0 ? visibleItems[index - 1] : null
              const showGroupHeader = !isSearching && previous?.category !== item.category
              const selected = index === activeIndex
              return (
                <div key={item.id}>
                  {showGroupHeader && (
                    <div className="command-palette-group">{CATEGORY_LABELS[item.category]}</div>
                  )}
                  <div
                    data-palette-index={index}
                    role="option"
                    aria-selected={selected}
                    onMouseEnter={() => setActiveIndex(index)}
                    className={[
                      'command-palette-item',
                      selected ? 'selected' : '',
                      item.active ? 'current' : ''
                    ]
                      .filter(Boolean)
                      .join(' ')}
                  >
                    <button
                      type="button"
                      className="command-palette-item-main"
                      onClick={item.onSelect}
                    >
                      <span className="command-palette-icon">{categoryIcon(item.category)}</span>
                      <span className="command-palette-copy">
                        <strong>
                          {item.title}
                          {item.active ? (
                            <span className="command-palette-badge">Active</span>
                          ) : null}
                        </strong>
                        {item.subtitle ? <small>{item.subtitle}</small> : null}
                      </span>
                      {item.shortcut ? (
                        <kbd className="command-palette-shortcut">{item.shortcut}</kbd>
                      ) : null}
                    </button>
                    {item.secondaryAction ? (
                      <button
                        type="button"
                        className="command-palette-secondary"
                        title={`${item.secondaryAction.label} (Shift+Enter)`}
                        onClick={item.secondaryAction.onSelect}
                      >
                        <SlidersHorizontal size={13} />
                        <span>{item.secondaryAction.label}</span>
                      </button>
                    ) : null}
                  </div>
                </div>
              )
            })
          )}
        </div>

        <footer className="command-palette-footer">
          <span>↑↓ navigate</span>
          <span>↵ run</span>
          <span>⇧↵ configure model</span>
          <span>esc close</span>
        </footer>
      </div>
    </div>
  )
}

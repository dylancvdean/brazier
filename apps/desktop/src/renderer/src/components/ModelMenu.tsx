import { Box, Check, HardDrive, LoaderCircle, Search, SlidersHorizontal, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { formatBytes, type HardwareInfo, type LocalModel } from '../api'
import { modelDisplayName } from '../model-utils'
import { CapabilityIcons, capabilityFlags } from './CapabilityIcons'
import { generationFit, generationFitLabel, type QuantFit } from './manage/quant'

type ModelMenuProps = {
  models: LocalModel[]
  selectedModel: string
  loading: boolean
  /** Heading for the popover; names the family the list is filtered to. */
  title?: string
  /** Whether ffmpeg is available, which is what makes video input possible. */
  videoPipeline?: boolean
  /** Host hardware profile for subtle VRAM/RAM fit badges on local models. */
  hardware?: HardwareInfo | null
  onSelect: (modelId: string) => void
  /** Open this model's advanced configuration. */
  onConfigure: (modelId: string) => void
  /** How many settings each model carries, so a configured one says so. */
  configuredCounts?: Record<string, number>
  onManage: () => void
  onClose: () => void
}

function fitShortLabel(fit: QuantFit): string | null {
  switch (fit) {
    case 'gpu':
      return 'GPU'
    case 'system':
      return 'Fits'
    case 'offload':
      return 'Offload'
    case 'none':
      return 'Tight'
    default:
      return null
  }
}

export function filterModelsForMenu(models: LocalModel[], query: string): LocalModel[] {
  const needle = query.trim().toLowerCase()
  if (!needle) return models
  const tokens = needle.split(/\s+/).filter(Boolean)
  return models.filter((model) => {
    const meta = modelDisplayName(model.id, model)
    const haystack = [
      meta.title,
      meta.subtitle,
      model.id,
      model.engine ?? '',
      model.repo_id ?? '',
      model.library_label ?? ''
    ]
      .join(' ')
      .toLowerCase()
    return tokens.every((token) => haystack.includes(token))
  })
}

/**
 * Searchable, keyboard-navigable model *selection* popover. Model management
 * (downloading, deleting) lives in the Manage panel instead.
 */
export function ModelMenu({
  models,
  selectedModel,
  loading,
  title = 'Choose a model',
  videoPipeline = false,
  hardware = null,
  onSelect,
  onConfigure,
  configuredCounts,
  onManage,
  onClose
}: ModelMenuProps): React.JSX.Element {
  const [query, setQuery] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  const filtered = useMemo(() => filterModelsForMenu(models, query), [models, query])
  const [activeIndex, setActiveIndex] = useState(() => {
    const selectedIdx = models.findIndex((model) => model.id === selectedModel)
    return selectedIdx >= 0 ? selectedIdx : 0
  })

  useEffect(() => {
    if (models.length > 0) {
      inputRef.current?.focus()
    }
  }, [models.length])

  useEffect(() => {
    setActiveIndex((current) => {
      if (filtered.length === 0) return 0
      if (!query.trim()) {
        const selectedIdx = filtered.findIndex((model) => model.id === selectedModel)
        if (selectedIdx >= 0) return selectedIdx
      }
      return Math.min(current, filtered.length - 1)
    })
  }, [filtered, query, selectedModel])

  useEffect(() => {
    const row = listRef.current?.querySelector<HTMLElement>(`[data-model-index="${activeIndex}"]`)
    row?.scrollIntoView({ block: 'nearest' })
  }, [activeIndex])

  function handleKeyDown(event: React.KeyboardEvent): void {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      onClose()
      return
    }
    if (event.key === 'ArrowDown') {
      if (filtered.length === 0) return
      event.preventDefault()
      setActiveIndex((current) => (current + 1) % filtered.length)
      return
    }
    if (event.key === 'ArrowUp') {
      if (filtered.length === 0) return
      event.preventDefault()
      setActiveIndex((current) => (current - 1 + filtered.length) % filtered.length)
      return
    }
    if (event.key === 'Enter' && document.activeElement === inputRef.current) {
      const picked = filtered[activeIndex]
      if (!picked) return
      event.preventDefault()
      onSelect(picked.id)
      onClose()
    }
  }

  return (
    <div className="menu-backdrop" onMouseDown={onClose}>
      <div
        className="popover model-menu"
        role="dialog"
        aria-label={title}
        onMouseDown={(event) => event.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <div className="popover-title">{title}</div>
        {models.length > 0 && (
          <label className="model-menu-search">
            <Search size={13} />
            <input
              ref={inputRef}
              value={query}
              onChange={(event) => {
                setQuery(event.target.value)
                setActiveIndex(0)
              }}
              placeholder={models.length > 4 ? `Filter ${models.length} models…` : 'Filter models…'}
              aria-label="Filter models"
              spellCheck={false}
            />
            {query ? (
              <button
                type="button"
                className="model-menu-search-clear"
                aria-label="Clear filter"
                onClick={() => {
                  setQuery('')
                  inputRef.current?.focus()
                }}
              >
                <X size={12} />
              </button>
            ) : null}
          </label>
        )}
        {loading && (
          <div className="popover-empty">
            <LoaderCircle className="spin" size={16} />
            Loading your library…
          </div>
        )}
        {!loading && models.length === 0 && (
          <div className="popover-empty">
            <Box size={16} />
            No local models yet. Download one from the library.
          </div>
        )}
        {!loading && models.length > 0 && filtered.length === 0 && (
          <div className="popover-empty">
            <Search size={15} />
            No models match “{query.trim()}”.
          </div>
        )}
        <div className="model-menu-list" ref={listRef} role="listbox" aria-label={title}>
          {filtered.map((model, index) => {
            const meta = modelDisplayName(model.id, model)
            const active = model.id === selectedModel
            const focused = index === activeIndex
            const configured = configuredCounts?.[model.id] ?? 0
            const isRemote = model.id.startsWith('remote:') || model.engine === 'remote'
            const fit: QuantFit =
              !isRemote && model.size_bytes != null
                ? generationFit(model.size_bytes, hardware)
                : 'unknown'
            const fitBadge = fitShortLabel(fit)
            return (
              <div
                key={model.id}
                data-model-index={index}
                role="option"
                aria-selected={active}
                onMouseEnter={() => setActiveIndex(index)}
                className={[
                  'model-menu-item',
                  active ? 'active' : '',
                  focused ? 'focused' : ''
                ]
                  .filter(Boolean)
                  .join(' ')}
              >
                <button
                  type="button"
                  className="model-menu-item-select"
                  onClick={() => {
                    onSelect(model.id)
                    onClose()
                  }}
                >
                  <div className="model-menu-item-name">
                    <div className="model-menu-item-title">
                      <strong>{meta.title}</strong>
                      {fitBadge ? (
                        <span
                          className={`model-fit-badge ${fit}`}
                          title={generationFitLabel(fit)}
                        >
                          {fitBadge}
                        </span>
                      ) : null}
                      <CapabilityIcons flags={capabilityFlags(model, videoPipeline)} />
                    </div>
                    <span>
                      {meta.subtitle}
                      {model.size_bytes != null ? ` · ${formatBytes(model.size_bytes)}` : ''}
                      {configured > 0 ? ` · ${configured} setting${configured === 1 ? '' : 's'}` : ''}
                    </span>
                  </div>
                  {active && <Check size={15} />}
                </button>
                <button
                  type="button"
                  className={
                    configured > 0 ? 'model-menu-item-configure set' : 'model-menu-item-configure'
                  }
                  title={`Configure ${meta.title}`}
                  aria-label={`Configure ${meta.title}`}
                  onClick={() => {
                    onClose()
                    onConfigure(model.id)
                  }}
                >
                  <SlidersHorizontal size={14} />
                </button>
              </div>
            )
          })}
        </div>
        <button
          type="button"
          className="popover-footer-action"
          onClick={() => {
            onClose()
            onManage()
          }}
        >
          <HardDrive size={14} />
          Manage model library…
        </button>
      </div>
    </div>
  )
}

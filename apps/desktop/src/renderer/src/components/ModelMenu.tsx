import { Box, Check, HardDrive, LoaderCircle, SlidersHorizontal } from 'lucide-react'
import { formatBytes, type LocalModel } from '../api'
import { modelDisplayName } from '../model-utils'
import { CapabilityIcons, capabilityFlags } from './CapabilityIcons'

type ModelMenuProps = {
  models: LocalModel[]
  selectedModel: string
  loading: boolean
  /** Heading for the popover; names the family the list is filtered to. */
  title?: string
  /** Whether ffmpeg is available, which is what makes video input possible. */
  videoPipeline?: boolean
  onSelect: (modelId: string) => void
  /** Open this model's advanced configuration. */
  onConfigure: (modelId: string) => void
  /** How many settings each model carries, so a configured one says so. */
  configuredCounts?: Record<string, number>
  onManage: () => void
  onClose: () => void
}

/**
 * Lightweight model *selection* popover. Model management (downloading,
 * deleting) lives in the Manage panel instead.
 */
export function ModelMenu({
  models,
  selectedModel,
  loading,
  title = 'Choose a model',
  videoPipeline = false,
  onSelect,
  onConfigure,
  configuredCounts,
  onManage,
  onClose
}: ModelMenuProps): React.JSX.Element {
  return (
    <div className="menu-backdrop" onMouseDown={onClose}>
      <div className="popover model-menu" onMouseDown={(event) => event.stopPropagation()}>
        <div className="popover-title">{title}</div>
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
        <div className="model-menu-list">
          {models.map((model) => {
            const meta = modelDisplayName(model.id, model)
            const active = model.id === selectedModel
            const configured = configuredCounts?.[model.id] ?? 0
            return (
              <div
                key={model.id}
                className={active ? 'model-menu-item active' : 'model-menu-item'}
              >
                <button
                  className="model-menu-item-select"
                  onClick={() => {
                    onSelect(model.id)
                    onClose()
                  }}
                >
                  <div className="model-menu-item-name">
                    <div className="model-menu-item-title">
                      <strong>{meta.title}</strong>
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
                  className={configured > 0 ? 'model-menu-item-configure set' : 'model-menu-item-configure'}
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

import { Box, Check, HardDrive, LoaderCircle } from 'lucide-react'
import { formatBytes, type LocalModel } from '../api'
import { modelDisplayName } from '../model-utils'

type ModelMenuProps = {
  models: LocalModel[]
  selectedModel: string
  loading: boolean
  /** Heading for the popover; names the family the list is filtered to. */
  title?: string
  onSelect: (modelId: string) => void
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
  onSelect,
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
            return (
              <button
                key={model.id}
                className={active ? 'model-menu-item active' : 'model-menu-item'}
                onClick={() => {
                  onSelect(model.id)
                  onClose()
                }}
              >
                <div className="model-menu-item-name">
                  <strong>{meta.title}</strong>
                  <span>
                    {meta.subtitle}
                    {model.size_bytes != null ? ` · ${formatBytes(model.size_bytes)}` : ''}
                  </span>
                </div>
                {active && <Check size={15} />}
              </button>
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

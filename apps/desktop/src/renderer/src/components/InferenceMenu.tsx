import { LoaderCircle } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { RuntimeSettings } from '../api'

type InferenceMenuProps = {
  settings: RuntimeSettings | null
  saving: boolean
  onApply: (settings: RuntimeSettings) => void
  onClose: () => void
}

/**
 * Per-generation inference defaults (sampling and reasoning). These are
 * usage-side settings, deliberately separated from engine launch
 * configuration and runtime management.
 */
export function InferenceMenu({
  settings,
  saving,
  onApply,
  onClose
}: InferenceMenuProps): React.JSX.Element {
  const [draft, setDraft] = useState<RuntimeSettings | null>(settings)
  useEffect(() => setDraft(settings), [settings])

  const dirty =
    draft != null && settings != null && JSON.stringify(draft) !== JSON.stringify(settings)

  return (
    <div className="menu-backdrop" onMouseDown={onClose}>
      <div className="popover inference-menu" onMouseDown={(event) => event.stopPropagation()}>
        <div className="popover-title">Inference settings</div>
        {!draft ? (
          <div className="popover-empty">
            <LoaderCircle className="spin" size={16} />
            Waiting for the daemon…
          </div>
        ) : (
          <>
            <label className="slider-row">
              <span>
                Temperature <em>{draft.temperature.toFixed(2)}</em>
              </span>
              <input
                type="range"
                min={0}
                max={2}
                step={0.05}
                value={draft.temperature}
                onChange={(event) =>
                  setDraft({ ...draft, temperature: Number(event.target.value) })
                }
              />
            </label>
            <label className="slider-row">
              <span>
                Top P <em>{draft.top_p.toFixed(2)}</em>
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={draft.top_p}
                onChange={(event) => setDraft({ ...draft, top_p: Number(event.target.value) })}
              />
            </label>
            <label className="field-row">
              <span>Max tokens</span>
              <input
                type="number"
                min={1}
                placeholder="Model default"
                value={draft.max_tokens ?? ''}
                onChange={(event) =>
                  setDraft({
                    ...draft,
                    max_tokens: event.target.value ? Number(event.target.value) : null
                  })
                }
              />
            </label>
            <label className="toggle-row">
              <div>
                <strong>Reasoning</strong>
                <span>Let thinking models deliberate before answering</span>
              </div>
              <input
                type="checkbox"
                checked={draft.enable_reasoning}
                onChange={(event) =>
                  setDraft({ ...draft, enable_reasoning: event.target.checked })
                }
              />
            </label>
            <button
              className="popover-apply"
              disabled={!dirty || saving}
              onClick={() => onApply(draft)}
            >
              {saving ? <LoaderCircle className="spin" size={14} /> : null}
              {dirty ? 'Apply' : 'Up to date'}
            </button>
          </>
        )}
      </div>
    </div>
  )
}

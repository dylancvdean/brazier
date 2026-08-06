import { LoaderCircle, X } from 'lucide-react'
import { useEffect, useState } from 'react'

import { saveRuntimeSettings, type LocalModel, type RuntimeSettings } from '../api'
import { modelDisplayName } from '../model-utils'

type Props = {
  settings: RuntimeSettings | null
  models: LocalModel[]
  onSaved: (settings: RuntimeSettings) => void
  onError: (message: string | null) => void
  onClose: () => void
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

/**
 * Generate-mode defaults: the models Image and Video use when the user has not
 * picked one, plus the job timeout. These are the same defaults the chat
 * `generate_image` / `generate_video` tools run with, configured here instead
 * of under Manage → Engine.
 */
export function GenerateSettings(props: Props): React.JSX.Element {
  const [draft, setDraft] = useState<RuntimeSettings | null>(props.settings)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => setDraft(props.settings), [props.settings])

  const dirty =
    draft != null &&
    props.settings != null &&
    JSON.stringify(draft) !== JSON.stringify(props.settings)

  const imageModels = props.models.filter((model) => model.id.startsWith('sdcpp-image:'))
  const videoModels = props.models.filter((model) => model.id.startsWith('sdcpp-video:'))

  async function save(): Promise<void> {
    if (!draft) return
    setSaving(true)
    props.onError(null)
    setError(null)
    try {
      props.onSaved(await saveRuntimeSettings(draft))
      props.onClose()
    } catch (cause) {
      setError(errorText(cause))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="menu-backdrop model-settings-backdrop" onMouseDown={props.onClose}>
      <div
        className="model-settings-modal"
        role="dialog"
        aria-label="Generate settings"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="model-settings-head">
          <div>
            <strong>Generate settings</strong>
            <span>
              Which models Image and Video use by default, and how long a job is allowed to run.
            </span>
          </div>
          <button type="button" className="icon-button" onClick={props.onClose} aria-label="Close">
            <X size={17} />
          </button>
        </header>

        {error ? <div className="error-banner">{error}</div> : null}

        <div className="model-settings-body">
          {!draft ? (
            <div className="manage-placeholder">
              <LoaderCircle className="spin" size={16} />
              Waiting for the daemon…
            </div>
          ) : (
            <div className="settings-group">
              <div className="section-label">Default models</div>
              <p className="model-help">
                Picked when no model is selected, and what the chat `generate_image` /
                `generate_video` tools run with. Choose <em>None</em> to leave the tab empty until a
                model is picked.
              </p>
              <div className="settings-grid">
                <label>
                  <span>Default image model</span>
                  <select
                    value={draft.default_image_gen_model ?? ''}
                    onChange={(event) =>
                      setDraft({
                        ...draft,
                        default_image_gen_model: event.target.value || null
                      })
                    }
                  >
                    <option value="">None</option>
                    {imageModels.map((model) => (
                      <option key={model.id} value={model.id}>
                        {modelDisplayName(model.id, model).title}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Default video model</span>
                  <select
                    value={draft.default_video_gen_model ?? ''}
                    onChange={(event) =>
                      setDraft({
                        ...draft,
                        default_video_gen_model: event.target.value || null
                      })
                    }
                  >
                    <option value="">None</option>
                    {videoModels.map((model) => (
                      <option key={model.id} value={model.id}>
                        {modelDisplayName(model.id, model).title}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="settings-grid">
                <label>
                  <span>Generation timeout (seconds)</span>
                  <input
                    type="number"
                    min={0}
                    max={86400}
                    step={60}
                    value={draft.generation_timeout_secs ?? 0}
                    onChange={(event) =>
                      setDraft({
                        ...draft,
                        generation_timeout_secs: Math.max(0, Number(event.target.value) || 0)
                      })
                    }
                  />
                </label>
              </div>
              <p className="model-help">
                0 lets Brazier work it out from the frames and steps asked for, which suits most
                machines. Raise it if a slow, CPU-only host is being cut off while still rendering;
                a running job can always be stopped by hand.
              </p>
            </div>
          )}
        </div>

        <footer className="model-settings-foot">
          <div className="model-settings-foot-actions">
            <button type="button" className="chip-button subtle" onClick={props.onClose}>
              Cancel
            </button>
            <button
              type="button"
              className="chip-button"
              disabled={!dirty || saving}
              onClick={() => void save()}
            >
              {saving ? <LoaderCircle className="spin" size={13} /> : null}
              {dirty ? 'Save' : 'Saved'}
            </button>
          </div>
        </footer>
      </div>
    </div>
  )
}

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
 * Voice-mode defaults: which PersonaPlex model talks to you by default and the
 * persona it speaks as. Configuring them here, rather than under Manage →
 * Engine, keeps the mode's own settings where the mode is.
 */
export function VoiceSettings(props: Props): React.JSX.Element {
  const [draft, setDraft] = useState<RuntimeSettings | null>(props.settings)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => setDraft(props.settings), [props.settings])

  const dirty =
    draft != null &&
    props.settings != null &&
    JSON.stringify(draft) !== JSON.stringify(props.settings)

  const voiceModels = props.models.filter((model) => model.id.startsWith('personaplex:'))

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
        aria-label="Voice settings"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="model-settings-head">
          <div>
            <strong>Voice settings</strong>
            <span>
              Which PersonaPlex model speaks by default, and how it introduces itself.
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
              <div className="section-label">Default model & persona</div>
              <p className="model-help">
                The model is picked when none is selected, and is also what a voice session starts
                with when the top bar has nothing chosen.
              </p>
              <div className="settings-grid">
                <label>
                  <span>Default voice model</span>
                  <select
                    value={draft.default_voice_model ?? ''}
                    onChange={(event) =>
                      setDraft({
                        ...draft,
                        default_voice_model: event.target.value || null
                      })
                    }
                  >
                    <option value="">None</option>
                    {voiceModels.map((model) => (
                      <option key={model.id} value={model.id}>
                        {modelDisplayName(model.id, model).title}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="span-2">
                  <span>Default voice persona</span>
                  <input
                    value={draft.default_voice_persona ?? ''}
                    onChange={(event) =>
                      setDraft({
                        ...draft,
                        default_voice_persona: event.target.value || null
                      })
                    }
                    placeholder="You are a helpful assistant."
                  />
                </label>
              </div>
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

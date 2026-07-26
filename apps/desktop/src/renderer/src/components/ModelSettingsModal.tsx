/**
 * Per-model configuration.
 *
 * One model, all of the settings its engine will take, and nothing that belongs
 * to another kind of model. What is stored here are that model's defaults: they
 * apply whenever it is used, from chat, from Generate, from a tool call a model
 * made itself, or from the API.
 */

import { LoaderCircle, RotateCcw, X } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'

import {
  listAdapters,
  resetModelProfile,
  saveModelProfile,
  type Adapter,
  type LocalModel,
  type ModelKind,
  type ModelProfile,
  type RuntimeSettings
} from '../api'
import { modelDisplayName, modelEngine } from '../model-utils'
import {
  ModelSettingsFields,
  emptyProfile,
  profileIsEmpty,
  type InheritedDefaults
} from './ModelSettingsFields'

type Props = {
  model: LocalModel
  kind: ModelKind
  /** The stored profile, or undefined when this model has never been configured. */
  profile: ModelProfile | undefined
  /** Global defaults, shown as the placeholder behind each unset field. */
  settings: RuntimeSettings | null
  onSaved: (models: Record<string, ModelProfile>) => void
  onClose: () => void
}

const KIND_BLURB: Record<ModelKind, string> = {
  text: 'Sampling applies to the next message. Loading settings restart the model.',
  image: 'Defaults for every image this model renders, here and from tool calls.',
  video: 'Defaults for every clip this model renders, here and from tool calls.',
  transcription: 'Applied whenever this model transcribes speech.',
  voice: 'Applied when a realtime voice session starts with this model.'
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

export function ModelSettingsModal(props: Props): React.JSX.Element {
  const [draft, setDraft] = useState<ModelProfile>(props.profile ?? emptyProfile(props.kind))
  const [adapters, setAdapters] = useState<Adapter[]>([])
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const refreshAdapters = useCallback(() => {
    void listAdapters()
      .then(setAdapters)
      .catch((cause: unknown) => setError(errorText(cause)))
  }, [])

  useEffect(refreshAdapters, [refreshAdapters])

  const meta = modelDisplayName(props.model.id, props.model)
  const engine = modelEngine(props.model)
  const inherited: InheritedDefaults = {
    contextSize: props.settings?.context_size,
    batchSize: props.settings?.batch_size,
    temperature: props.settings?.temperature,
    topP: props.settings?.top_p,
    flashAttention: props.settings?.flash_attention,
    kvCacheTypeK: props.settings?.kv_cache_type_k,
    kvCacheTypeV: props.settings?.kv_cache_type_v,
    maxTokens: props.settings?.max_tokens ?? null
  }
  const dirty = JSON.stringify(draft) !== JSON.stringify(props.profile ?? emptyProfile(props.kind))

  async function save(): Promise<void> {
    setSaving(true)
    setError(null)
    try {
      const models = await saveModelProfile(props.model.id, draft)
      props.onSaved(models)
      props.onClose()
    } catch (cause) {
      setError(errorText(cause))
    } finally {
      setSaving(false)
    }
  }

  async function reset(): Promise<void> {
    setSaving(true)
    setError(null)
    try {
      const models = await resetModelProfile(props.model.id)
      props.onSaved(models)
      setDraft(emptyProfile(props.kind))
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
        aria-label={`Configure ${meta.title}`}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="model-settings-head">
          <div>
            <strong>{meta.title}</strong>
            <span>
              {meta.subtitle} · {KIND_BLURB[props.kind]}
            </span>
          </div>
          <button type="button" className="icon-button" onClick={props.onClose} aria-label="Close">
            <X size={17} />
          </button>
        </header>

        {error ? <div className="error-banner">{error}</div> : null}

        <div className="model-settings-body">
          <ModelSettingsFields
            modelId={props.model.id}
            kind={props.kind}
            engine={engine}
            profile={draft}
            adapters={adapters}
            inherited={inherited}
            onChange={setDraft}
            onAdapterAdded={refreshAdapters}
            onError={setError}
          />
        </div>

        <footer className="model-settings-foot">
          <button
            type="button"
            className="chip-button subtle"
            disabled={saving || profileIsEmpty(draft)}
            title="Forget every override and follow the global defaults again"
            onClick={() => void reset()}
          >
            <RotateCcw size={13} /> Reset to defaults
          </button>
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

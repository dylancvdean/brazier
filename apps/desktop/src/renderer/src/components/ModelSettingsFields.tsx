/**
 * The advanced settings one model can be given, laid out by the kind of model
 * it is.
 *
 * These fields are shared rather than duplicated: the same set appears in the
 * per-model configuration modal, behind "show advanced" in the inference menu,
 * and at the foot of the voice setup screen. Wherever they appear they edit the
 * same thing — the model's own profile — so a value set in one place is the
 * value seen in the others.
 *
 * Every control is tri-state on purpose. An empty number field, or a boolean
 * left on "Default", means *inherit*: the global inference settings decide, and
 * failing those the engine does. Only what has been deliberately set is stored,
 * which is what lets a model be configured without freezing it against changes
 * to the defaults later.
 */

import { AlertTriangle, FolderOpen, X } from 'lucide-react'
import { useEffect, useState } from 'react'

import {
  fetchModelChatTemplate,
  listModels,
  registerAdapter,
  type Adapter,
  type ControlNetBinding,
  type DiffusionProfile,
  type LocalModel,
  type LoraBinding,
  type ModelKind,
  type ModelProfile,
  type TextProfile,
  type TranscriptionProfile,
  type VoiceProfile
} from '../api'
import { modelDisplayName, modelKindFor } from '../model-utils'

/** Extensions a LoRA or ControlNet is published as. */
const ADAPTER_FILTERS = [
  { name: 'Adapter weights', extensions: ['safetensors', 'gguf', 'ckpt', 'pt', 'pth', 'bin'] }
]

const IMAGE_FILTERS = [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp'] }]

const KV_CACHE_TYPES = ['f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1']

const SAMPLING_METHODS = [
  'euler',
  'euler_a',
  'heun',
  'dpm2',
  'dpm++2s_a',
  'dpm++2m',
  'dpm++2mv2',
  'ipndm',
  'ipndm_v',
  'lcm',
  'ddim_trailing',
  'tcd'
]

const SCHEDULES = [
  'default',
  'discrete',
  'karras',
  'exponential',
  'ays',
  'gits',
  'smoothstep',
  'sgm_uniform',
  'simple'
]

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

// ---------------------------------------------------------------------------
// Field primitives
// ---------------------------------------------------------------------------

type NumberFieldProps = {
  label: string
  /** What happens when this is left empty, shown as the placeholder. */
  inherited?: string | number
  hint?: string
  value: number | null | undefined
  min?: number
  max?: number
  step?: number
  onChange: (value: number | null) => void
}

function NumberField(props: NumberFieldProps): React.JSX.Element {
  return (
    <label className="model-field" title={props.hint}>
      <span>{props.label}</span>
      <input
        type="number"
        min={props.min}
        max={props.max}
        step={props.step}
        placeholder={props.inherited != null ? String(props.inherited) : 'Default'}
        value={props.value ?? ''}
        onChange={(event) =>
          props.onChange(event.target.value === '' ? null : Number(event.target.value))
        }
      />
    </label>
  )
}

type TextFieldProps = {
  label: string
  placeholder?: string
  hint?: string
  value: string | null | undefined
  multiline?: boolean
  onChange: (value: string | null) => void
}

function TextField(props: TextFieldProps): React.JSX.Element {
  const commit = (value: string): void => props.onChange(value === '' ? null : value)
  return (
    <label className="model-field" title={props.hint}>
      <span>{props.label}</span>
      {props.multiline ? (
        <textarea
          rows={3}
          placeholder={props.placeholder ?? 'Default'}
          value={props.value ?? ''}
          onChange={(event) => commit(event.target.value)}
        />
      ) : (
        <input
          placeholder={props.placeholder ?? 'Default'}
          value={props.value ?? ''}
          onChange={(event) => commit(event.target.value)}
        />
      )}
    </label>
  )
}

type SelectFieldProps = {
  label: string
  hint?: string
  options: string[]
  /** Label for the unset choice, which inherits whatever decides otherwise. */
  defaultLabel?: string
  value: string | null | undefined
  onChange: (value: string | null) => void
}

function SelectField(props: SelectFieldProps): React.JSX.Element {
  return (
    <label className="model-field" title={props.hint}>
      <span>{props.label}</span>
      <select
        value={props.value ?? ''}
        onChange={(event) => props.onChange(event.target.value === '' ? null : event.target.value)}
      >
        <option value="">{props.defaultLabel ?? 'Default'}</option>
        {props.options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  )
}

type ToggleFieldProps = {
  label: string
  hint?: string
  /** What "Default" resolves to right now, when it is known. */
  inherited?: boolean
  value: boolean | null | undefined
  onChange: (value: boolean | null) => void
}

/**
 * A three-way switch, because "off" and "not set" are different answers: one
 * overrides the global setting, the other follows it.
 */
function ToggleField(props: ToggleFieldProps): React.JSX.Element {
  const current = props.value == null ? '' : props.value ? 'on' : 'off'
  return (
    <label className="model-field" title={props.hint}>
      <span>{props.label}</span>
      <select
        value={current}
        onChange={(event) => {
          const next = event.target.value
          props.onChange(next === '' ? null : next === 'on')
        }}
      >
        <option value="">
          {props.inherited == null
            ? 'Default'
            : `Default · ${props.inherited ? 'on' : 'off'}`}
        </option>
        <option value="on">On</option>
        <option value="off">Off</option>
      </select>
    </label>
  )
}

/** A named block of fields, collapsed until it is wanted. */
function FieldGroup(props: {
  title: string
  summary?: string
  open?: boolean
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <details className="model-field-group" open={props.open}>
      <summary>
        <span>{props.title}</span>
        {props.summary ? <small>{props.summary}</small> : null}
      </summary>
      <div className="model-field-grid">{props.children}</div>
    </details>
  )
}

/** Free-form engine arguments, kept as one line each. */
function ExtraArgsField(props: {
  engine: string
  value: string[] | undefined
  onChange: (value: string[]) => void
}): React.JSX.Element {
  return (
    <label className="model-field wide">
      <span>Extra {props.engine} arguments</span>
      <textarea
        rows={2}
        spellCheck={false}
        placeholder="--some-flag value"
        value={(props.value ?? []).join(' ')}
        onChange={(event) =>
          props.onChange(event.target.value.split(/\s+/u).filter((part) => part.length > 0))
        }
      />
      <small>
        Passed to {props.engine} as written, after everything above. Brazier does not check them —
        an argument this build does not know stops the engine from starting, and the error appears
        where it is launched.
      </small>
    </label>
  )
}

/**
 * Editable Jinja chat template. Defaults to the GGUF-bundled
 * `tokenizer.chat_template` so the box always shows what the model will use.
 * Clearing the override (Reset) goes back to that bundled source.
 */
function ChatTemplateField(props: {
  modelId: string
  value: string | null | undefined
  onChange: (value: string | null) => void
}): React.JSX.Element {
  const [bundled, setBundled] = useState<string | null>(null)
  const [source, setSource] = useState<'gguf' | 'missing' | 'unsupported' | 'loading'>('loading')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setSource('loading')
    setError(null)
    void fetchModelChatTemplate(props.modelId)
      .then((response) => {
        if (cancelled) return
        setBundled(response.chat_template)
        setSource(response.source)
      })
      .catch((cause: unknown) => {
        if (cancelled) return
        setBundled(null)
        setSource('missing')
        setError(cause instanceof Error ? cause.message : String(cause))
      })
    return () => {
      cancelled = true
    }
  }, [props.modelId])

  const display = props.value ?? bundled ?? ''
  const customized = props.value != null
  const loading = source === 'loading'

  return (
    <label className="model-field wide">
      <span>Chat template (Jinja)</span>
      <textarea
        rows={12}
        spellCheck={false}
        className="chat-template-editor"
        disabled={loading}
        placeholder={
          loading
            ? 'Loading GGUF template…'
            : source === 'unsupported'
              ? 'Not available for this model format'
              : source === 'missing'
                ? 'No tokenizer.chat_template in this GGUF'
                : 'GGUF-bundled template'
        }
        value={display}
        onChange={(event) => {
          const next = event.target.value
          if (bundled != null && next === bundled) {
            props.onChange(null)
            return
          }
          props.onChange(next === '' ? null : next)
        }}
      />
      <small>
        {error
          ? error
          : customized
            ? 'Custom override — restarts the model when saved.'
            : source === 'gguf'
              ? 'Using the template bundled in the GGUF.'
              : source === 'unsupported'
                ? 'Chat templates are only read from GGUF files.'
                : source === 'missing'
                  ? 'This GGUF has no embedded chat template.'
                  : 'Loading…'}
        {bundled != null && customized ? (
          <>
            {' '}
            <button type="button" className="text-button" onClick={() => props.onChange(null)}>
              Reset to bundled
            </button>
          </>
        ) : null}
      </small>
    </label>
  )
}

// ---------------------------------------------------------------------------
// Adapters
// ---------------------------------------------------------------------------

function adapterName(adapters: Adapter[], binding: { adapter_id: string }): string {
  return adapters.find((entry) => entry.id === binding.adapter_id)?.name ?? binding.adapter_id
}

/**
 * LoRAs chosen for this model, from the adapters an engine can actually load.
 *
 * The catalogue is filtered by engine rather than shown in full: llama.cpp only
 * reads a GGUF LoRA and stable-diffusion.cpp only the safetensors kind, so an
 * adapter offered to the wrong model would fail minutes into a job.
 */
function LoraSection(props: {
  engine: string
  adapters: Adapter[]
  loras: LoraBinding[]
  singleOnly?: boolean
  onChange: (loras: LoraBinding[]) => void
  onAdapterAdded: () => void
  onError: (message: string | null) => void
}): React.JSX.Element {
  const compatible = props.adapters.filter(
    (entry) => entry.kind === 'lora' && entry.engines.includes(props.engine)
  )
  const chosen = props.loras
  const unused = compatible.filter(
    (entry) => !chosen.some((binding) => binding.adapter_id === entry.id)
  )

  async function addFromDisk(): Promise<void> {
    props.onError(null)
    try {
      const path = await window.brazier.selectFile('Choose a LoRA', ADAPTER_FILTERS)
      if (!path) return
      const adapter = await registerAdapter('lora', path)
      props.onAdapterAdded()
      if (adapter.engines.includes(props.engine)) {
        props.onChange([...chosen, { adapter_id: adapter.id, scale: 1, enabled: true }])
      } else {
        props.onError(
          `${adapter.name} is a ${adapter.engines.join('/') || 'unrecognised'} adapter, which ${props.engine} cannot load. It has been added to the library for a model that can.`
        )
      }
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  return (
    <div className="adapter-section">
      <div className="adapter-section-head">
        <span className="section-label">LoRA adapters</span>
        <button type="button" className="chip-button subtle" onClick={() => void addFromDisk()}>
          <FolderOpen size={12} /> Add from disk…
        </button>
      </div>

      {chosen.length === 0 ? (
        <p className="model-help">None applied. A LoRA adjusts this model without replacing it.</p>
      ) : (
        <div className="adapter-list">
          {chosen.map((binding, index) => (
            <div className="adapter-row" key={binding.adapter_id}>
              <label className="adapter-row-name">
                <input
                  type="checkbox"
                  checked={binding.enabled}
                  onChange={(event) =>
                    props.onChange(
                      chosen.map((entry, position) =>
                        position === index
                          ? { ...entry, enabled: event.target.checked }
                          : entry
                      )
                    )
                  }
                />
                <strong>{adapterName(props.adapters, binding)}</strong>
              </label>
              <label className="adapter-row-scale">
                <span>Strength {binding.scale.toFixed(2)}</span>
                <input
                  type="range"
                  min={-2}
                  max={2}
                  step={0.05}
                  value={binding.scale}
                  onChange={(event) =>
                    props.onChange(
                      chosen.map((entry, position) =>
                        position === index
                          ? { ...entry, scale: Number(event.target.value) }
                          : entry
                      )
                    )
                  }
                />
              </label>
              <button
                type="button"
                className="chip-button subtle"
                title="Remove from this model"
                onClick={() =>
                  props.onChange(chosen.filter((_, position) => position !== index))
                }
              >
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      )}

      {props.singleOnly && chosen.filter((binding) => binding.enabled).length > 1 ? (
        <p className="model-help warn">
          <AlertTriangle size={12} /> MLX loads one adapter at a time — the first enabled LoRA is
          the one applied.
        </p>
      ) : null}

      {unused.length > 0 ? (
        <label className="model-field">
          <span>Add an installed LoRA</span>
          <select
            value=""
            onChange={(event) => {
              if (!event.target.value) return
              props.onChange([
                ...chosen,
                { adapter_id: event.target.value, scale: 1, enabled: true }
              ])
            }}
          >
            <option value="">Choose…</option>
            {unused.map((adapter) => (
              <option key={adapter.id} value={adapter.id}>
                {adapter.name}
                {adapter.external ? ' · external' : ''}
              </option>
            ))}
          </select>
        </label>
      ) : compatible.length === 0 ? (
        <p className="model-help">
          No LoRAs installed that {props.engine} can load
          {props.engine === 'llama.cpp'
            ? ' — llama.cpp needs a GGUF LoRA, not safetensors.'
            : props.engine.startsWith('mlx')
              ? ' — MLX needs a directory of adapter weights.'
              : '.'}
        </p>
      ) : null}
    </div>
  )
}

/** ControlNets chosen for a diffusion model. */
function ControlNetSection(props: {
  adapters: Adapter[]
  controlNets: ControlNetBinding[]
  onChange: (controlNets: ControlNetBinding[]) => void
  onAdapterAdded: () => void
  onError: (message: string | null) => void
}): React.JSX.Element {
  const compatible = props.adapters.filter((entry) => entry.kind === 'controlnet')
  const chosen = props.controlNets
  const unused = compatible.filter(
    (entry) => !chosen.some((binding) => binding.adapter_id === entry.id)
  )
  const enabledCount = chosen.filter((binding) => binding.enabled).length

  function update(index: number, patch: Partial<ControlNetBinding>): void {
    props.onChange(
      chosen.map((entry, position) => (position === index ? { ...entry, ...patch } : entry))
    )
  }

  async function addFromDisk(): Promise<void> {
    props.onError(null)
    try {
      const path = await window.brazier.selectFile('Choose a ControlNet', ADAPTER_FILTERS)
      if (!path) return
      const adapter = await registerAdapter('controlnet', path)
      props.onAdapterAdded()
      props.onChange([
        ...chosen,
        { adapter_id: adapter.id, strength: 1, cpu: false, enabled: chosen.length === 0 }
      ])
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  async function chooseImage(index: number): Promise<void> {
    props.onError(null)
    try {
      const path = await window.brazier.selectFile('Choose a control image', IMAGE_FILTERS)
      if (path) update(index, { image_path: path })
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  return (
    <div className="adapter-section">
      <div className="adapter-section-head">
        <span className="section-label">ControlNet</span>
        <button type="button" className="chip-button subtle" onClick={() => void addFromDisk()}>
          <FolderOpen size={12} /> Add from disk…
        </button>
      </div>

      {chosen.length === 0 ? (
        <p className="model-help">
          None applied. A ControlNet steers composition from a reference image — a pose, a depth
          map, an edge trace.
        </p>
      ) : (
        <div className="adapter-list">
          {chosen.map((binding, index) => (
            <div className="adapter-row control-net-row" key={binding.adapter_id}>
              <label className="adapter-row-name">
                <input
                  type="checkbox"
                  checked={binding.enabled}
                  onChange={(event) => update(index, { enabled: event.target.checked })}
                />
                <strong>{adapterName(props.adapters, binding)}</strong>
              </label>
              <label className="adapter-row-scale">
                <span>Strength {binding.strength.toFixed(2)}</span>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  value={binding.strength}
                  onChange={(event) => update(index, { strength: Number(event.target.value) })}
                />
              </label>
              <button
                type="button"
                className="chip-button subtle"
                onClick={() => void chooseImage(index)}
                title={binding.image_path ?? 'Pick the image this ControlNet reads'}
              >
                {binding.image_path ? 'Change image' : 'Control image…'}
              </button>
              <label className="adapter-row-toggle" title="Slower, but leaves VRAM for the model">
                <input
                  type="checkbox"
                  checked={binding.cpu}
                  onChange={(event) => update(index, { cpu: event.target.checked })}
                />
                On CPU
              </label>
              <button
                type="button"
                className="chip-button subtle"
                title="Remove from this model"
                onClick={() =>
                  props.onChange(chosen.filter((_, position) => position !== index))
                }
              >
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      )}

      {enabledCount > 1 ? (
        <p className="model-help warn">
          <AlertTriangle size={12} /> stable-diffusion.cpp applies one ControlNet per job — the
          first enabled one is used.
        </p>
      ) : null}

      {unused.length > 0 ? (
        <label className="model-field">
          <span>Add an installed ControlNet</span>
          <select
            value=""
            onChange={(event) => {
              if (!event.target.value) return
              props.onChange([
                ...chosen,
                {
                  adapter_id: event.target.value,
                  strength: 1,
                  cpu: false,
                  enabled: enabledCount === 0
                }
              ])
            }}
          >
            <option value="">Choose…</option>
            {unused.map((adapter) => (
              <option key={adapter.id} value={adapter.id}>
                {adapter.name}
                {adapter.external ? ' · external' : ''}
              </option>
            ))}
          </select>
        </label>
      ) : null}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Per-kind field sets
// ---------------------------------------------------------------------------

/** What the global settings would supply for a field left unset. */
export type InheritedDefaults = {
  contextSize?: number
  batchSize?: number
  temperature?: number
  topP?: number
  flashAttention?: boolean
  kvCacheTypeK?: string
  kvCacheTypeV?: string
  maxTokens?: number | null
  diffusionWidth?: number
  diffusionHeight?: number
  videoFrames?: number
  vaeTiling?: boolean
  clipOnCpu?: boolean
  diffusionFa?: boolean
  autoFit?: boolean
  maxVram?: number
  paramsBackend?: string
  streamLayers?: boolean
  offloadToCpu?: boolean
}

type SectionProps<T> = {
  modelId: string
  profile: T
  onChange: (profile: T) => void
  engine: string
  adapters: Adapter[]
  inherited: InheritedDefaults
  onAdapterAdded: () => void
  onError: (message: string | null) => void
}

/** Agent-mode defaults for chat models: which model subagents use, and how many. */
function AgentFields(props: {
  profile: TextProfile
  onChange: <K extends keyof TextProfile>(key: K, value: TextProfile[K]) => void
  excludeModelId: string
  parentContext?: number
  /** Parallel slots only apply to llama.cpp launches. */
  isLlama: boolean
}): React.JSX.Element {
  const [chatModels, setChatModels] = useState<LocalModel[]>([])
  const maxSubagents = props.profile.max_subagents ?? 2
  const parallelSlots = 1 + maxSubagents

  useEffect(() => {
    let cancelled = false
    void listModels()
      .then((models) => {
        if (cancelled) return
        setChatModels(
          models
            .filter((model) => modelKindFor(model.id) === 'text')
            .sort((a, b) =>
              modelDisplayName(a.id, a).title.localeCompare(modelDisplayName(b.id, b).title)
            )
        )
      })
      .catch(() => {
        if (!cancelled) setChatModels([])
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <FieldGroup title="Agent" summary="Subagents spawned from Agent mode">
      <label
        className="model-field"
        title="Model used when this agent spawns a subagent. Leave on This model to reuse the parent."
      >
        <span>Subagent model</span>
        <select
          value={props.profile.subagent_model ?? ''}
          onChange={(event) =>
            props.onChange('subagent_model', event.target.value === '' ? null : event.target.value)
          }
        >
          <option value="">This model</option>
          {chatModels
            .filter((model) => model.id !== props.excludeModelId)
            .map((model) => (
              <option key={model.id} value={model.id}>
                {modelDisplayName(model.id, model).title}
              </option>
            ))}
        </select>
      </label>
      <NumberField
        label="Subagent context"
        hint="Context available to each child. Unset uses the same context as the parent."
        inherited={props.profile.context_size ?? props.parentContext}
        min={512}
        value={props.profile.subagent_context_size}
        onChange={(value) => props.onChange('subagent_context_size', value)}
      />
      <NumberField
        label="Max subagents"
        hint="How many child agents may run at once. Default 2."
        inherited={2}
        min={1}
        max={8}
        value={props.profile.max_subagents}
        onChange={(value) => props.onChange('max_subagents', value)}
      />
      {props.isLlama ? (
        <ToggleField
          label="Parallel subagents"
          hint={`Off keeps one llama.cpp slot (safest on memory). On starts the server with --parallel ${parallelSlots} (1 + max subagents) so concurrent children can continuous-batch. Each slot gets the configured per-agent context, so KV memory grows with the slot count and a reload is required. Turn off or lower either context setting if launch runs out of memory.`}
          inherited={false}
          value={props.profile.parallel_subagents}
          onChange={(value) => props.onChange('parallel_subagents', value)}
        />
      ) : null}
    </FieldGroup>
  )
}

function TextFields(props: SectionProps<TextProfile>): React.JSX.Element {
  const { profile, onChange } = props
  const set = <K extends keyof TextProfile>(key: K, value: TextProfile[K]): void =>
    onChange({ ...profile, [key]: value })
  const isLlama = props.engine === 'llama.cpp'
  const isMlx = props.engine.startsWith('mlx')

  return (
    <>
      <FieldGroup title="Sampling" summary="Applied per request" open>
        <NumberField
          label="Temperature"
          inherited={props.inherited.temperature}
          min={0}
          max={2}
          step={0.05}
          value={profile.temperature}
          onChange={(value) => set('temperature', value)}
        />
        <NumberField
          label="Top P"
          inherited={props.inherited.topP}
          min={0}
          max={1}
          step={0.05}
          value={profile.top_p}
          onChange={(value) => set('top_p', value)}
        />
        <NumberField
          label="Top K"
          hint="Keep only the K most likely tokens. 0 disables it."
          min={0}
          value={profile.top_k}
          onChange={(value) => set('top_k', value)}
        />
        <NumberField
          label="Min P"
          hint="Drop tokens below this share of the most likely one."
          min={0}
          max={1}
          step={0.01}
          value={profile.min_p}
          onChange={(value) => set('min_p', value)}
        />
        <NumberField
          label="Max tokens"
          inherited={props.inherited.maxTokens ?? 'Model default'}
          min={1}
          value={profile.max_tokens}
          onChange={(value) => set('max_tokens', value)}
        />
        <NumberField
          label="Seed"
          hint="Fix the seed to make a generation repeatable."
          value={profile.seed}
          onChange={(value) => set('seed', value)}
        />
        {isLlama ? (
          <>
            <NumberField
              label="Typical P"
              min={0}
              max={1}
              step={0.05}
              value={profile.typical_p}
              onChange={(value) => set('typical_p', value)}
            />
            <NumberField
              label="Repetition penalty"
              min={0}
              max={4}
              step={0.01}
              value={profile.repeat_penalty}
              onChange={(value) => set('repeat_penalty', value)}
            />
            <NumberField
              label="Repetition window"
              hint="How many recent tokens the penalty looks back over."
              value={profile.repeat_last_n}
              onChange={(value) => set('repeat_last_n', value)}
            />
          </>
        ) : null}
        {isMlx ? (
          <>
            <NumberField
              label="Repetition penalty"
              min={0}
              max={4}
              step={0.01}
              value={profile.repeat_penalty}
              onChange={(value) => set('repeat_penalty', value)}
            />
            <NumberField
              label="Repetition window"
              value={profile.repeat_last_n}
              onChange={(value) => set('repeat_last_n', value)}
            />
          </>
        ) : null}
        <NumberField
          label="Presence penalty"
          min={-2}
          max={2}
          step={0.05}
          value={profile.presence_penalty}
          onChange={(value) => set('presence_penalty', value)}
        />
        <NumberField
          label="Frequency penalty"
          min={-2}
          max={2}
          step={0.05}
          value={profile.frequency_penalty}
          onChange={(value) => set('frequency_penalty', value)}
        />
        <TextField
          label="Stop sequences"
          placeholder="Comma separated"
          hint="Generation stops when the model produces one of these."
          value={(profile.stop ?? []).join(', ') || null}
          onChange={(value) =>
            set(
              'stop',
              (value ?? '')
                .split(',')
                .map((part) => part.trim())
                .filter((part) => part.length > 0)
            )
          }
        />
        <TextField
          label="System prompt"
          multiline
          hint="Prepended to every conversation with this model."
          value={profile.system_prompt}
          onChange={(value) => set('system_prompt', value)}
        />
      </FieldGroup>

      {isLlama ? (
        <FieldGroup title="Repetition control" summary="DRY and Mirostat">
          <NumberField
            label="DRY multiplier"
            hint="0 disables DRY, which suppresses repeated phrases rather than repeated tokens."
            min={0}
            step={0.1}
            value={profile.dry_multiplier}
            onChange={(value) => set('dry_multiplier', value)}
          />
          <NumberField
            label="DRY base"
            min={0}
            step={0.05}
            value={profile.dry_base}
            onChange={(value) => set('dry_base', value)}
          />
          <NumberField
            label="DRY allowed length"
            min={0}
            value={profile.dry_allowed_length}
            onChange={(value) => set('dry_allowed_length', value)}
          />
          <SelectField
            label="Mirostat"
            hint="Targets a perplexity instead of truncating the distribution."
            options={['0', '1', '2']}
            defaultLabel="Default · off"
            value={profile.mirostat == null ? null : String(profile.mirostat)}
            onChange={(value) => set('mirostat', value == null ? null : Number(value))}
          />
          <NumberField
            label="Mirostat tau"
            step={0.1}
            value={profile.mirostat_tau}
            onChange={(value) => set('mirostat_tau', value)}
          />
          <NumberField
            label="Mirostat eta"
            step={0.01}
            value={profile.mirostat_eta}
            onChange={(value) => set('mirostat_eta', value)}
          />
        </FieldGroup>
      ) : null}

      <FieldGroup title="Reasoning">
        <ToggleField
          label="Thinking"
          hint="Let this model deliberate before answering, when it can."
          value={profile.enable_reasoning}
          onChange={(value) => set('enable_reasoning', value)}
        />
        <NumberField
          label="Thinking budget"
          hint="Token cap on deliberation."
          min={1}
          value={profile.reasoning_budget_tokens}
          onChange={(value) => set('reasoning_budget_tokens', value)}
        />
      </FieldGroup>

      <AgentFields
        profile={profile}
        onChange={set}
        excludeModelId={props.modelId}
        parentContext={props.inherited.contextSize}
        isLlama={isLlama}
      />

      {isLlama ? (
        <FieldGroup title="Loading" summary="Restarts the model when changed">
          <NumberField
            label="Context length"
            inherited={props.inherited.contextSize}
            min={512}
            value={profile.context_size}
            onChange={(value) => set('context_size', value)}
          />
          <NumberField
            label="Batch size"
            inherited={props.inherited.batchSize}
            min={32}
            max={8192}
            value={profile.batch_size}
            onChange={(value) => set('batch_size', value)}
          />
          <NumberField
            label="Physical batch"
            hint="--ubatch-size: how much of a batch is computed at once."
            min={1}
            max={8192}
            value={profile.ubatch_size}
            onChange={(value) => set('ubatch_size', value)}
          />
          <NumberField
            label="GPU layers"
            hint="-1 offloads every layer it can."
            min={-1}
            max={999}
            value={profile.gpu_layers}
            onChange={(value) => set('gpu_layers', value)}
          />
          <NumberField
            label="Threads"
            min={1}
            value={profile.threads}
            onChange={(value) => set('threads', value)}
          />
          <ToggleField
            label="Flash attention"
            inherited={props.inherited.flashAttention}
            value={profile.flash_attention}
            onChange={(value) => set('flash_attention', value)}
          />
          <ToggleField
            label="MTP speculative decoding"
            hint="Auto-detected for MTP GGUFs. It uses one llama.cpp slot and cannot combine with image projectors."
            value={profile.mtp}
            onChange={(value) => set('mtp', value)}
          />
          <NumberField
            label="MTP draft tokens"
            hint="How many tokens the model predicts ahead. Default 2; higher is not always faster."
            min={1}
            max={6}
            value={profile.mtp_draft_tokens}
            onChange={(value) => set('mtp_draft_tokens', value)}
          />
          <SelectField
            label="KV cache · keys"
            hint="Quantising the cache buys context length with a little quality."
            options={KV_CACHE_TYPES}
            defaultLabel={`Default · ${props.inherited.kvCacheTypeK ?? 'f16'}`}
            value={profile.kv_cache_type_k}
            onChange={(value) => set('kv_cache_type_k', value)}
          />
          <SelectField
            label="KV cache · values"
            options={KV_CACHE_TYPES}
            defaultLabel={`Default · ${props.inherited.kvCacheTypeV ?? 'f16'}`}
            value={profile.kv_cache_type_v}
            onChange={(value) => set('kv_cache_type_v', value)}
          />
          <ToggleField
            label="Jinja templates"
            value={profile.jinja}
            onChange={(value) => set('jinja', value)}
          />
          <ChatTemplateField
            modelId={props.modelId}
            value={profile.chat_template}
            onChange={(value) => set('chat_template', value)}
          />
          <ToggleField
            label="Lock in memory"
            hint="--mlock: stops the weights being paged out."
            value={profile.mlock}
            onChange={(value) => set('mlock', value)}
          />
          <ToggleField
            label="Disable mmap"
            hint="--no-mmap: read the weights instead of mapping them."
            value={profile.no_mmap}
            onChange={(value) => set('no_mmap', value)}
          />
          <NumberField
            label="MoE layers on CPU"
            hint="--n-cpu-moe: how a large mixture-of-experts model fits on a small GPU."
            min={0}
            value={profile.n_cpu_moe}
            onChange={(value) => set('n_cpu_moe', value)}
          />
          <NumberField
            label="Cache reuse"
            hint="Tokens of prefix llama.cpp may reuse between requests."
            min={0}
            value={profile.cache_reuse}
            onChange={(value) => set('cache_reuse', value)}
          />
          <NumberField
            label="Defrag threshold"
            step={0.05}
            value={profile.defrag_threshold}
            onChange={(value) => set('defrag_threshold', value)}
          />
        </FieldGroup>
      ) : null}

      {isLlama ? (
        <FieldGroup title="Context extension and multi-GPU">
          <SelectField
            label="RoPE scaling"
            hint="Stretches the position encoding past the trained window."
            options={['none', 'linear', 'yarn']}
            value={profile.rope_scaling}
            onChange={(value) => set('rope_scaling', value)}
          />
          <NumberField
            label="RoPE frequency base"
            step={1000}
            value={profile.rope_freq_base}
            onChange={(value) => set('rope_freq_base', value)}
          />
          <NumberField
            label="RoPE frequency scale"
            step={0.05}
            value={profile.rope_freq_scale}
            onChange={(value) => set('rope_freq_scale', value)}
          />
          <NumberField
            label="YaRN original context"
            hint="The window the model was actually trained at."
            min={0}
            value={profile.yarn_orig_ctx}
            onChange={(value) => set('yarn_orig_ctx', value)}
          />
          <SelectField
            label="Split mode"
            options={['none', 'layer', 'row']}
            value={profile.split_mode}
            onChange={(value) => set('split_mode', value)}
          />
          <NumberField
            label="Main GPU"
            min={0}
            value={profile.main_gpu}
            onChange={(value) => set('main_gpu', value)}
          />
          <TextField
            label="Tensor split"
            placeholder="e.g. 0.6,0.4"
            hint="How the layers are divided across GPUs."
            value={profile.tensor_split}
            onChange={(value) => set('tensor_split', value)}
          />
        </FieldGroup>
      ) : null}

      {isLlama || isMlx ? (
        <LoraSection
          engine={props.engine}
          adapters={props.adapters}
          loras={profile.loras ?? []}
          singleOnly={isMlx}
          onChange={(loras) => set('loras', loras)}
          onAdapterAdded={props.onAdapterAdded}
          onError={props.onError}
        />
      ) : (
        <p className="model-help">
          Adapters apply to models Brazier runs itself. This one is served elsewhere, so its LoRAs
          are configured wherever it runs.
        </p>
      )}

      {isLlama || isMlx ? (
        <FieldGroup title="Engine arguments" summary="Escape hatch">
          <ExtraArgsField
            engine={isLlama ? 'llama-server' : 'the MLX server'}
            value={profile.extra_args}
            onChange={(value) => set('extra_args', value)}
          />
        </FieldGroup>
      ) : null}
    </>
  )
}

function DiffusionFields(
  props: SectionProps<DiffusionProfile> & { video: boolean }
): React.JSX.Element {
  const { profile, onChange } = props
  const set = <K extends keyof DiffusionProfile>(
    key: K,
    value: DiffusionProfile[K]
  ): void => onChange({ ...profile, [key]: value })

  return (
    <>
      <FieldGroup title="Output" open>
        <NumberField
          label="Width"
          inherited={props.inherited.diffusionWidth ?? 512}
          min={64}
          max={4096}
          step={64}
          value={profile.width}
          onChange={(value) => set('width', value)}
        />
        <NumberField
          label="Height"
          inherited={props.inherited.diffusionHeight ?? 512}
          min={64}
          max={4096}
          step={64}
          value={profile.height}
          onChange={(value) => set('height', value)}
        />
        <NumberField
          label="Steps"
          inherited={20}
          min={1}
          max={1000}
          value={profile.steps}
          onChange={(value) => set('steps', value)}
        />
        {props.video ? (
          <>
            <NumberField
              label="Frames"
              inherited={props.inherited.videoFrames ?? 16}
              min={1}
              max={1024}
              value={profile.video_frames}
              onChange={(value) => set('video_frames', value)}
            />
            <NumberField
              label="FPS"
              inherited={24}
              min={1}
              max={120}
              value={profile.fps}
              onChange={(value) => set('fps', value)}
            />
          </>
        ) : null}
        <NumberField
          label="Batch count"
          hint="How many outputs one job renders."
          min={1}
          max={64}
          value={profile.batch_count}
          onChange={(value) => set('batch_count', value)}
        />
        <NumberField
          label="Seed"
          value={profile.seed}
          onChange={(value) => set('seed', value)}
        />
      </FieldGroup>

      <FieldGroup title="Guidance" summary="What the prompt is worth" open>
        <NumberField
          label="CFG scale"
          hint="Distilled models such as Flux schnell need 1.0; SDXL likes 5–8."
          min={0}
          max={30}
          step={0.5}
          value={profile.cfg_scale}
          onChange={(value) => set('cfg_scale', value)}
        />
        <NumberField
          label="Guidance"
          hint="Distilled guidance, used by the Flux family instead of CFG."
          min={0}
          max={30}
          step={0.1}
          value={profile.guidance}
          onChange={(value) => set('guidance', value)}
        />
        <NumberField
          label="Image CFG scale"
          hint="Applied to the starting image in image-to-image."
          min={0}
          max={30}
          step={0.5}
          value={profile.img_cfg_scale}
          onChange={(value) => set('img_cfg_scale', value)}
        />
        <NumberField
          label="Denoise strength"
          hint="How far a starting image is departed from. 0 keeps it, 1 ignores it."
          min={0}
          max={1}
          step={0.05}
          value={profile.strength}
          onChange={(value) => set('strength', value)}
        />
        <TextField
          label="Standing negative prompt"
          multiline
          hint="Added to whatever a job asks to avoid, rather than replacing it."
          value={profile.negative_prompt}
          onChange={(value) => set('negative_prompt', value)}
        />
      </FieldGroup>

      <FieldGroup title="Sampler">
        <SelectField
          label="Sampling method"
          options={SAMPLING_METHODS}
          value={profile.sampling_method}
          onChange={(value) => set('sampling_method', value)}
        />
        <SelectField
          label="Schedule"
          options={SCHEDULES}
          value={profile.schedule}
          onChange={(value) => set('schedule', value)}
        />
        <NumberField
          label="CLIP skip"
          hint="-1 leaves it to the model."
          min={-1}
          max={12}
          value={profile.clip_skip}
          onChange={(value) => set('clip_skip', value)}
        />
        <NumberField
          label="Eta"
          min={0}
          max={1}
          step={0.05}
          value={profile.eta}
          onChange={(value) => set('eta', value)}
        />
        <NumberField
          label="Flow shift"
          hint="Timestep shift for flow-matching models (Flux, Wan)."
          min={0}
          step={0.1}
          value={profile.flow_shift}
          onChange={(value) => set('flow_shift', value)}
        />
        <SelectField
          label="RNG"
          options={['std_default', 'cuda']}
          value={profile.rng}
          onChange={(value) => set('rng', value)}
        />
        <NumberField
          label="Skip-layer guidance"
          min={0}
          max={30}
          step={0.1}
          value={profile.slg_scale}
          onChange={(value) => set('slg_scale', value)}
        />
        <TextField
          label="Skip layers"
          placeholder="e.g. 7,8,9"
          value={profile.skip_layers}
          onChange={(value) => set('skip_layers', value)}
        />
        <NumberField
          label="Skip-layer start"
          min={0}
          max={1}
          step={0.05}
          value={profile.skip_layer_start}
          onChange={(value) => set('skip_layer_start', value)}
        />
        <NumberField
          label="Skip-layer end"
          min={0}
          max={1}
          step={0.05}
          value={profile.skip_layer_end}
          onChange={(value) => set('skip_layer_end', value)}
        />
      </FieldGroup>

      <FieldGroup title="Memory and placement" summary="Fitting a large model on this machine">
        <ToggleField
          label="VAE tiling"
          hint="Decodes in tiles, which is how a large image fits in VRAM."
          inherited={props.inherited.vaeTiling}
          value={profile.vae_tiling}
          onChange={(value) => set('vae_tiling', value)}
        />
        <ToggleField
          label="VAE on CPU"
          value={profile.vae_on_cpu}
          onChange={(value) => set('vae_on_cpu', value)}
        />
        <ToggleField
          label="CLIP on CPU"
          inherited={props.inherited.clipOnCpu}
          value={profile.clip_on_cpu}
          onChange={(value) => set('clip_on_cpu', value)}
        />
        <ToggleField
          label="Diffusion flash attention"
          inherited={props.inherited.diffusionFa}
          value={profile.diffusion_fa}
          onChange={(value) => set('diffusion_fa', value)}
        />
        <ToggleField
          label="Automatic placement"
          hint="Lets sd.cpp choose execution devices. Disabled by default on integrated Vulkan GPUs, which upstream auto-fit currently skips."
          inherited={props.inherited.autoFit}
          value={profile.auto_fit}
          onChange={(value) => set('auto_fit', value)}
        />
        <NumberField
          label="GPU graph budget (GiB)"
          hint="Maximum GiB used by one device before sd.cpp splits the execution graph."
          inherited={props.inherited.maxVram}
          min={0}
          max={256}
          step={0.5}
          value={profile.max_vram}
          onChange={(value) => set('max_vram', value)}
        />
        <TextField
          label="Parameter residency"
          hint="Where weights wait between executions. CPU residency is required for layer streaming; disk residency only loads whole phases lazily."
          placeholder={props.inherited.paramsBackend ?? 'Default'}
          value={profile.params_backend}
          onChange={(value) => set('params_backend', value)}
        />
        <ToggleField
          label="Stream layers"
          hint="Loads and prefetches layers within the GPU graph budget instead of keeping the full model resident."
          inherited={props.inherited.streamLayers}
          value={profile.stream_layers}
          onChange={(value) => set('stream_layers', value)}
        />
        <ToggleField
          label="Offload to CPU"
          hint="Keeps weights in RAM and moves them per step. Slower, but it fits."
          inherited={props.inherited.offloadToCpu}
          value={profile.offload_to_cpu}
          onChange={(value) => set('offload_to_cpu', value)}
        />
        <NumberField
          label="Threads"
          min={1}
          value={profile.threads}
          onChange={(value) => set('threads', value)}
        />
      </FieldGroup>

      <LoraSection
        engine="stable-diffusion.cpp"
        adapters={props.adapters}
        loras={profile.loras ?? []}
        onChange={(loras) => set('loras', loras)}
        onAdapterAdded={props.onAdapterAdded}
        onError={props.onError}
      />

      <ControlNetSection
        adapters={props.adapters}
        controlNets={profile.control_nets ?? []}
        onChange={(controlNets) => set('control_nets', controlNets)}
        onAdapterAdded={props.onAdapterAdded}
        onError={props.onError}
      />

      <FieldGroup title="Engine arguments" summary="Escape hatch">
        <ExtraArgsField
          engine="sd-cli"
          value={profile.extra_args}
          onChange={(value) => set('extra_args', value)}
        />
      </FieldGroup>
    </>
  )
}

function TranscriptionFields(
  props: SectionProps<TranscriptionProfile> & { streaming: boolean }
): React.JSX.Element {
  const { profile, onChange } = props
  const set = <K extends keyof TranscriptionProfile>(
    key: K,
    value: TranscriptionProfile[K]
  ): void => onChange({ ...profile, [key]: value })

  if (props.streaming) {
    return (
      <>
        <FieldGroup title="Streaming" open>
          <NumberField
            label="Lookahead"
            hint="Frames of audio the model may wait for. Higher is more accurate and less immediate."
            min={0}
            max={1000}
            value={profile.lookahead}
            onChange={(value) => set('lookahead', value)}
          />
        </FieldGroup>
        <p className="model-help">
          The streaming recogniser decodes as you speak, so the batch decoding options below it —
          beam search, fallbacks, thresholds — do not apply.
        </p>
      </>
    )
  }

  return (
    <>
      <FieldGroup title="Language" open>
        <TextField
          label="Language"
          placeholder="auto"
          hint="ISO code, or `auto` to detect."
          value={profile.language}
          onChange={(value) => set('language', value)}
        />
        <ToggleField
          label="Translate to English"
          value={profile.translate}
          onChange={(value) => set('translate', value)}
        />
        <TextField
          label="Initial prompt"
          multiline
          hint="Biases the first window — useful for names and jargon it keeps mishearing."
          value={profile.initial_prompt}
          onChange={(value) => set('initial_prompt', value)}
        />
      </FieldGroup>

      <FieldGroup title="Decoding">
        <NumberField
          label="Beam size"
          min={1}
          max={64}
          value={profile.beam_size}
          onChange={(value) => set('beam_size', value)}
        />
        <NumberField
          label="Best of"
          min={1}
          max={64}
          value={profile.best_of}
          onChange={(value) => set('best_of', value)}
        />
        <NumberField
          label="Temperature"
          min={0}
          max={2}
          step={0.05}
          value={profile.temperature}
          onChange={(value) => set('temperature', value)}
        />
        <NumberField
          label="Context carried over"
          hint="Tokens of previous text the next window sees. 0 stops earlier mistakes propagating."
          value={profile.max_context}
          onChange={(value) => set('max_context', value)}
        />
        <ToggleField
          label="No fallback"
          hint="Do not retry a window at a higher temperature."
          value={profile.no_fallback}
          onChange={(value) => set('no_fallback', value)}
        />
        <NumberField
          label="Threads"
          min={1}
          value={profile.threads}
          onChange={(value) => set('threads', value)}
        />
        <ToggleField
          label="Flash attention"
          value={profile.flash_attention}
          onChange={(value) => set('flash_attention', value)}
        />
      </FieldGroup>

      <FieldGroup title="Segmentation and thresholds">
        <NumberField
          label="Maximum segment length"
          hint="Characters per segment. 0 leaves it to the model."
          min={0}
          value={profile.max_len}
          onChange={(value) => set('max_len', value)}
        />
        <ToggleField
          label="Split on word"
          value={profile.split_on_word}
          onChange={(value) => set('split_on_word', value)}
        />
        <NumberField
          label="Word threshold"
          step={0.01}
          value={profile.word_threshold}
          onChange={(value) => set('word_threshold', value)}
        />
        <NumberField
          label="Entropy threshold"
          step={0.05}
          value={profile.entropy_threshold}
          onChange={(value) => set('entropy_threshold', value)}
        />
        <NumberField
          label="Log-probability threshold"
          step={0.1}
          value={profile.logprob_threshold}
          onChange={(value) => set('logprob_threshold', value)}
        />
        <NumberField
          label="No-speech threshold"
          hint="How confident it must be that a window is silence."
          step={0.05}
          value={profile.no_speech_threshold}
          onChange={(value) => set('no_speech_threshold', value)}
        />
        <ToggleField
          label="Suppress non-speech"
          hint="Stops `[music]` and its like appearing in the transcript."
          value={profile.suppress_nst}
          onChange={(value) => set('suppress_nst', value)}
        />
      </FieldGroup>

      <FieldGroup title="Engine arguments" summary="Escape hatch">
        <ExtraArgsField
          engine="whisper-cli"
          value={profile.extra_args}
          onChange={(value) => set('extra_args', value)}
        />
      </FieldGroup>
    </>
  )
}

function VoiceFields(props: SectionProps<VoiceProfile>): React.JSX.Element {
  const { profile, onChange } = props
  const set = <K extends keyof VoiceProfile>(key: K, value: VoiceProfile[K]): void =>
    onChange({ ...profile, [key]: value })

  async function chooseClip(): Promise<void> {
    props.onError(null)
    try {
      const path = await window.brazier.selectFile('Choose a reference voice clip', [
        { name: 'Audio', extensions: ['wav', 'mp3', 'flac', 'ogg', 'm4a'] }
      ])
      if (path) set('voice_prompt_path', path)
    } catch (cause) {
      props.onError(errorText(cause))
    }
  }

  return (
    <>
      <FieldGroup title="Persona and voice" open>
        <TextField
          label="Persona"
          multiline
          hint="How this model is told to behave when a session starts."
          value={profile.persona_text}
          onChange={(value) => set('persona_text', value)}
        />
        <TextField
          label="Built-in voice"
          placeholder="NATF2"
          hint="The voice id the model speaks in when no clip is given."
          value={profile.voice_id}
          onChange={(value) => set('voice_id', value)}
        />
        <label className="model-field">
          <span>Reference clip</span>
          <div className="model-field-actions">
            <button type="button" className="chip-button subtle" onClick={() => void chooseClip()}>
              <FolderOpen size={12} />
              {profile.voice_prompt_path ? 'Change clip…' : 'Choose clip…'}
            </button>
            {profile.voice_prompt_path ? (
              <button
                type="button"
                className="chip-button subtle"
                onClick={() => set('voice_prompt_path', null)}
              >
                <X size={12} /> Clear
              </button>
            ) : null}
          </div>
          <small>
            {profile.voice_prompt_path
              ? profile.voice_prompt_path
              : 'A clip clones its voice, and takes precedence over the id above.'}
          </small>
        </label>
      </FieldGroup>

      <FieldGroup title="Loading">
        <SelectField
          label="Quantisation"
          hint="Weight precision in bits. 4 is the on-device default; 8 costs memory for fidelity."
          options={['4', '8', '16']}
          defaultLabel="Default · 4-bit"
          value={profile.quantization == null ? null : String(profile.quantization)}
          onChange={(value) => set('quantization', value == null ? null : Number(value))}
        />
      </FieldGroup>

      <FieldGroup title="Engine arguments" summary="Escape hatch">
        <ExtraArgsField
          engine="the PersonaPlex server"
          value={profile.extra_args}
          onChange={(value) => set('extra_args', value)}
        />
      </FieldGroup>
    </>
  )
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

type ModelSettingsFieldsProps = {
  modelId: string
  kind: ModelKind
  /** Engine id, which decides which flags exist at all. */
  engine: string
  profile: ModelProfile
  adapters: Adapter[]
  inherited: InheritedDefaults
  onChange: (profile: ModelProfile) => void
  onAdapterAdded: () => void
  onError: (message: string | null) => void
}

/** The whole field set for one model, chosen by its kind. */
export function ModelSettingsFields(props: ModelSettingsFieldsProps): React.JSX.Element {
  const shared = {
    modelId: props.modelId,
    engine: props.engine,
    adapters: props.adapters,
    inherited: props.inherited,
    onAdapterAdded: props.onAdapterAdded,
    onError: props.onError
  }

  switch (props.profile.kind) {
    case 'text':
      return (
        <TextFields
          {...shared}
          profile={props.profile}
          onChange={(profile) => props.onChange({ ...profile, kind: 'text' })}
        />
      )
    case 'image':
    case 'video': {
      const kind = props.profile.kind
      return (
        <DiffusionFields
          {...shared}
          video={kind === 'video'}
          profile={props.profile}
          onChange={(profile) => props.onChange({ ...profile, kind })}
        />
      )
    }
    case 'transcription':
      return (
        <TranscriptionFields
          {...shared}
          streaming={props.modelId.startsWith('streaming-asr:')}
          profile={props.profile}
          onChange={(profile) => props.onChange({ ...profile, kind: 'transcription' })}
        />
      )
    case 'voice':
      return (
        <VoiceFields
          {...shared}
          profile={props.profile}
          onChange={(profile) => props.onChange({ ...profile, kind: 'voice' })}
        />
      )
  }
}

/** An empty profile of the kind a model takes. */
export function emptyProfile(kind: ModelKind): ModelProfile {
  return { kind } as ModelProfile
}

/** Whether a profile carries any decision at all. */
export function profileIsEmpty(profile: ModelProfile): boolean {
  return Object.entries(profile).every(([key, value]) => {
    if (key === 'kind') return true
    if (value == null) return true
    return Array.isArray(value) && value.length === 0
  })
}

/** How many settings a profile carries, for a badge on the button that opens it. */
export function profileCount(profile: ModelProfile | undefined): number {
  if (!profile) return 0
  return Object.entries(profile).filter(([key, value]) => {
    if (key === 'kind' || value == null) return false
    return !Array.isArray(value) || value.length > 0
  }).length
}

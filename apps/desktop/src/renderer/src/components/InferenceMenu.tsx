import { ChevronDown, ChevronRight, LoaderCircle } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  formatBytes,
  listAdapters,
  saveModelProfile,
  type Adapter,
  type HardwareInfo,
  type LocalModel,
  type ModelProfile,
  type ModelResidency,
  type RuntimeSettings
} from '../api'
import { modelEngine, modelKindFor } from '../model-utils'
import {
  INTEGRATED_GPU_VIDEO_DEFAULTS,
  usesIntegratedGpuVulkanDefaults
} from '../runtime-defaults'
import { ModelSettingsFields, emptyProfile } from './ModelSettingsFields'

type InferenceMenuProps = {
  settings: RuntimeSettings | null
  hardware: HardwareInfo | null
  selectedModel: string
  models: LocalModel[]
  /** Raw un-overridden model catalog so capability toggles show their true base default. */
  rawModels?: LocalModel[]
  residency?: ModelResidency | null
  saving: boolean
  /**
   * Model the advanced section configures. The bar shows a different model per
   * mode — a diffusion model in Generate, a voice model in Voice — and advanced
   * settings should follow what is on screen rather than the chat model behind
   * it. The sampling controls above stay global either way.
   */
  advancedModelId?: string
  /** That model's stored overrides, when it has any. */
  profile?: ModelProfile
  onApply: (settings: RuntimeSettings, options?: { skipReload?: boolean }) => void | Promise<void>
  /** Persist the model's own overrides, which are separate from the defaults. */
  onProfileSaved: (models: Record<string, ModelProfile>) => void
  onClose: () => void
}

type ReasoningMode = 'off' | 'on' | 'budget'

export type SamplingPreset = {
  id: 'coding' | 'balanced' | 'creative' | 'greedy'
  label: string
  temperature: number
  top_p: number
}

export const SAMPLING_PRESETS: readonly SamplingPreset[] = [
  { id: 'coding', label: 'Coding · 0.15', temperature: 0.15, top_p: 0.9 },
  { id: 'balanced', label: 'Balanced · 0.7', temperature: 0.7, top_p: 0.95 },
  { id: 'creative', label: 'Creative · 1.0', temperature: 1.0, top_p: 0.98 },
  { id: 'greedy', label: 'Greedy · 0.0', temperature: 0.0, top_p: 1.0 }
]

export function matchingSamplingPreset(
  temperature: number,
  topP: number
): SamplingPreset['id'] | null {
  const match = SAMPLING_PRESETS.find(
    (preset) =>
      Math.abs(preset.temperature - temperature) < 0.005 &&
      Math.abs(preset.top_p - topP) < 0.005
  )
  return match?.id ?? null
}

export type ResidencySummary = {
  placement: ModelResidency['placement']
  headline: string
  detail: string | null
  weightsPercent: number | null
  kvPercent: number | null
}

export function summarizeResidency(residency: ModelResidency): ResidencySummary {
  const hasGpuNumbers = residency.gpu_bytes != null || residency.gpu_kv_bytes != null
  const gpuWeights = residency.gpu_bytes ?? 0
  const gpuKv = residency.gpu_kv_bytes ?? 0
  const gpuTotal = gpuWeights + gpuKv
  const cpuWeights = residency.cpu_bytes ?? 0
  const cpuKv = residency.cpu_kv_bytes ?? 0
  const cpuTotal = cpuWeights + cpuKv

  const ctxPart = residency.aggregate_context_tokens
    ? `${formatTokens(residency.aggregate_context_tokens)} ctx`
    : null
  const slotsPart =
    residency.parallel_slots != null
      ? `${residency.parallel_slots} slot${residency.parallel_slots === 1 ? '' : 's'}`
      : null

  if (!hasGpuNumbers && cpuTotal === 0) {
    const extras = [ctxPart, slotsPart].filter(Boolean).join(' · ')
    return {
      placement: residency.placement,
      headline: residency.description,
      detail: extras || null,
      weightsPercent: null,
      kvPercent: null
    }
  }

  const capacity = residency.gpu_capacity_bytes
  const gpuLabel = capacity
    ? `GPU ${formatBytes(gpuTotal)} / ${formatBytes(capacity)}`
    : `GPU ${formatBytes(gpuTotal)}`
  const cpuLabel = cpuTotal > 0 ? `CPU ${formatBytes(cpuTotal)}` : null
  const headline = [gpuLabel, cpuLabel, ctxPart, slotsPart].filter(Boolean).join(' · ')

  const breakdownParts: string[] = []
  if (residency.gpu_bytes != null) breakdownParts.push(`weights ${formatBytes(residency.gpu_bytes)}`)
  if (residency.gpu_kv_bytes != null) breakdownParts.push(`KV ${formatBytes(residency.gpu_kv_bytes)}`)
  if (residency.cpu_kv_bytes != null && residency.cpu_kv_bytes > 0) {
    breakdownParts.push(`CPU KV ${formatBytes(residency.cpu_kv_bytes)}`)
  }

  let weightsPercent: number | null = null
  let kvPercent: number | null = null
  if (capacity && capacity > 0) {
    weightsPercent = Math.min(100, Math.max(0, (gpuWeights / capacity) * 100))
    kvPercent = Math.min(100 - weightsPercent, Math.max(0, (gpuKv / capacity) * 100))
  }

  return {
    placement: residency.placement,
    headline,
    detail: breakdownParts.length > 0 ? breakdownParts.join(' + ') : residency.description,
    weightsPercent,
    kvPercent
  }
}

export function unifiedSaveLabel(state: {
  dirty: boolean
  profileDirty: boolean
  saving: boolean
}): string {
  if (state.saving) return 'Saving…'
  if (state.dirty && state.profileDirty) return 'Save defaults & model overrides'
  if (state.profileDirty) return 'Save model overrides'
  if (state.dirty) return 'Apply'
  return 'Up to date'
}

export function contextPresets(maxContext: number | null | undefined): number[] {
  const candidates = [2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072]
  const capped = maxContext
    ? candidates.filter((value) => value <= maxContext)
    : candidates
  if (maxContext && !capped.includes(maxContext)) {
    capped.push(maxContext)
  }
  return capped.sort((a, b) => a - b)
}

/** Compact token label, e.g. 4096 -> "4K", 131072 -> "128K". */
function formatTokens(value: number): string {
  if (value >= 1024 && value % 1024 === 0) return `${value / 1024}K`
  if (value >= 1000) return `${Math.round(value / 1000)}K`
  return String(value)
}

/** Index of the preset closest to `value`, so the slider snaps sensibly. */
function nearestPresetIndex(value: number, options: number[]): number {
  let best = 0
  let bestDelta = Number.POSITIVE_INFINITY
  options.forEach((option, index) => {
    const delta = Math.abs(option - value)
    if (delta < bestDelta) {
      bestDelta = delta
      best = index
    }
  })
  return best
}

function reasoningModeFromSettings(settings: RuntimeSettings): ReasoningMode {
  if (!settings.enable_reasoning) return 'off'
  if (settings.reasoning_budget_tokens != null) return 'budget'
  return 'on'
}

function applyReasoningMode(
  settings: RuntimeSettings,
  mode: ReasoningMode,
  budget: number | null
): RuntimeSettings {
  switch (mode) {
    case 'off':
      return {
        ...settings,
        enable_reasoning: false,
        reasoning_budget_tokens: null
      }
    case 'budget':
      return {
        ...settings,
        enable_reasoning: true,
        reasoning_budget_tokens: budget ?? settings.reasoning_budget_tokens ?? 1024
      }
    default:
      return {
        ...settings,
        enable_reasoning: true,
        reasoning_budget_tokens: null
      }
  }
}

/**
 * Per-generation inference defaults (sampling, context, reasoning) + live
 * memory residency and unified per-model overrides.
 */
export function InferenceMenu({
  settings,
  hardware,
  selectedModel,
  models,
  rawModels,
  residency,
  saving,
  advancedModelId,
  profile,
  onApply,
  onProfileSaved,
  onClose
}: InferenceMenuProps): React.JSX.Element {
  const [draft, setDraft] = useState<RuntimeSettings | null>(settings)
  const [reasoningMode, setReasoningMode] = useState<ReasoningMode>('on')
  const [reasoningBudget, setReasoningBudget] = useState(1024)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [adapters, setAdapters] = useState<Adapter[]>([])
  const [profileDraft, setProfileDraft] = useState<ModelProfile | null>(null)
  const [profileError, setProfileError] = useState<string | null>(null)
  const [savingProfile, setSavingProfile] = useState(false)

  const model = useMemo(
    () => models.find((entry) => entry.id === selectedModel),
    [models, selectedModel]
  )
  const caps = model?.capabilities
  const maxContext = caps?.max_context_length ?? null
  const reasoningModes = caps?.reasoning_modes ?? (caps?.reasoning ? ['off', 'on'] : [])
  const contextOptions = useMemo(() => contextPresets(maxContext), [maxContext])

  useEffect(() => {
    if (!settings) return
    let next = settings
    if (maxContext && next.context_size > maxContext) {
      next = { ...next, context_size: maxContext }
    }
    setDraft(next)
    setReasoningMode(reasoningModeFromSettings(next))
    setReasoningBudget(next.reasoning_budget_tokens ?? 1024)
  }, [settings, maxContext])

  const dirty =
    draft != null && settings != null && JSON.stringify(draft) !== JSON.stringify(settings)

  const advancedModel = advancedModelId ?? selectedModel
  const advancedEntry = models.find((entry) => entry.id === advancedModel)
  const rawAdvancedEntry = (rawModels ?? models).find((entry) => entry.id === advancedModel)
  const modelKind = advancedModel ? modelKindFor(advancedModel) : null
  const useIntegratedDefaults = usesIntegratedGpuVulkanDefaults(draft, hardware)
  const storedProfile = useMemo(
    () => profile ?? (modelKind ? emptyProfile(modelKind) : null),
    [profile, modelKind]
  )
  const effectiveProfile = profileDraft ?? storedProfile
  const profileDirty =
    profileDraft != null &&
    storedProfile != null &&
    JSON.stringify(profileDraft) !== JSON.stringify(storedProfile)

  useEffect(() => {
    setProfileDraft(null)
    setProfileError(null)
  }, [advancedModel])

  const refreshAdapters = useCallback(() => {
    void listAdapters()
      .then(setAdapters)
      .catch(() => {
        // Non-fatal: the adapter pickers just show an empty library.
      })
  }, [])

  useEffect(() => {
    if (advancedOpen) refreshAdapters()
  }, [advancedOpen, refreshAdapters])

  function updateReasoningMode(mode: ReasoningMode): void {
    if (!draft) return
    setReasoningMode(mode)
    setDraft(applyReasoningMode(draft, mode, reasoningBudget))
  }

  const anyDirty = dirty || profileDirty
  const anySaving = saving || savingProfile

  async function saveAll(): Promise<void> {
    if (!draft || !anyDirty || anySaving) return
    setProfileError(null)
    try {
      if (dirty) {
        await onApply(draft, { skipReload: profileDirty })
      }
      if (profileDirty && profileDraft && advancedModel) {
        setSavingProfile(true)
        try {
          const savedModels = await saveModelProfile(advancedModel, profileDraft)
          onProfileSaved(savedModels)
          setProfileDraft(null)
        } finally {
          setSavingProfile(false)
        }
      }
    } catch (cause) {
      setProfileError(cause instanceof Error ? cause.message : String(cause))
    }
  }

  const residencyInfo = useMemo(
    () => (residency ? summarizeResidency(residency) : null),
    [residency]
  )
  const activePreset = draft ? matchingSamplingPreset(draft.temperature, draft.top_p) : null

  return (
    <div className="menu-backdrop" onMouseDown={onClose}>
      <div
        className="popover inference-menu"
        onMouseDown={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            event.preventDefault()
            event.stopPropagation()
            onClose()
          }
        }}
      >
        <div className="popover-title">Inference settings</div>
        {!draft ? (
          <div className="popover-empty">
            <LoaderCircle className="spin" size={16} />
            Waiting for the daemon…
          </div>
        ) : (
          <>
            {residencyInfo && (
              <div className="inference-residency" aria-label="Memory residency">
                <div className="inference-residency-head">
                  <span className="inference-residency-label">
                    <i className={`model-residency-dot ${residencyInfo.placement}`} />
                    Memory residency
                  </span>
                  <strong>{residencyInfo.headline}</strong>
                </div>
                {residencyInfo.weightsPercent != null && (
                  <div className="inference-residency-bar" aria-hidden="true">
                    <div
                      className="inference-residency-weights"
                      style={{ width: `${residencyInfo.weightsPercent}%` }}
                      title="Model weights in GPU memory"
                    />
                    {residencyInfo.kvPercent != null && residencyInfo.kvPercent > 0 ? (
                      <div
                        className="inference-residency-kv"
                        style={{ width: `${residencyInfo.kvPercent}%` }}
                        title="KV cache in GPU memory"
                      />
                    ) : null}
                  </div>
                )}
                {residencyInfo.detail ? (
                  <span className="inference-residency-detail">{residencyInfo.detail}</span>
                ) : null}
              </div>
            )}

            <label className="slider-row context-slider">
              <span>
                <span>Context length</span>
                <em>{draft.context_size.toLocaleString()} tokens</em>
              </span>
              <input
                type="range"
                min={0}
                max={Math.max(0, contextOptions.length - 1)}
                step={1}
                value={nearestPresetIndex(draft.context_size, contextOptions)}
                onChange={(event) =>
                  setDraft({
                    ...draft,
                    context_size: contextOptions[Number(event.target.value)] ?? draft.context_size
                  })
                }
              />
              <div className="slider-ticks" aria-hidden="true">
                {contextOptions.map((value, index) => (
                  <button
                    type="button"
                    key={value}
                    className={
                      index === nearestPresetIndex(draft.context_size, contextOptions)
                        ? 'active'
                        : ''
                    }
                    onClick={() => setDraft({ ...draft, context_size: value })}
                  >
                    {formatTokens(value)}
                  </button>
                ))}
              </div>
            </label>
            <p className="inference-help">
              Snaps to common windows{maxContext ? ` · max ${formatTokens(maxContext)}` : ''}. Larger
              context uses more memory; stay at or below the model&apos;s trained window.
            </p>

            <div className="sampling-presets" role="group" aria-label="Sampling presets">
              {SAMPLING_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  className={
                    activePreset === preset.id ? 'chip-button selected' : 'chip-button subtle'
                  }
                  aria-pressed={activePreset === preset.id}
                  onClick={() =>
                    setDraft({
                      ...draft,
                      temperature: preset.temperature,
                      top_p: preset.top_p
                    })
                  }
                >
                  {preset.label}
                </button>
              ))}
            </div>

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
            {reasoningModes.length > 0 && (
              <>
                <label className="field-row">
                  <span>Reasoning</span>
                  <select
                    value={reasoningMode}
                    onChange={(event) =>
                      updateReasoningMode(event.target.value as ReasoningMode)
                    }
                  >
                    {reasoningModes.includes('off') && <option value="off">Off</option>}
                    {reasoningModes.includes('on') && (
                      <option value="on">On · full thinking</option>
                    )}
                    {reasoningModes.includes('budget') && (
                      <option value="budget">Limited budget</option>
                    )}
                  </select>
                </label>
                {reasoningMode === 'budget' && (
                  <label className="field-row">
                    <span>Thinking budget (tokens)</span>
                    <input
                      type="number"
                      min={128}
                      max={131072}
                      step={128}
                      value={reasoningBudget}
                      onChange={(event) => {
                        const parsed = Number(event.target.value)
                        const next = Number.isFinite(parsed)
                          ? Math.max(128, Math.round(parsed))
                          : (draft.reasoning_budget_tokens ?? 1024)
                        setReasoningBudget(next)
                        setDraft(applyReasoningMode(draft, 'budget', next))
                      }}
                    />
                  </label>
                )}
                <p className="inference-help">
                  Budget caps how many tokens the model can spend thinking.
                </p>
              </>
            )}
            <label className="settings-toggle inference-toggle">
              <input
                type="checkbox"
                checked={draft.drop_reasoning_between_turns ?? false}
                onChange={(event) =>
                  setDraft({
                    ...draft,
                    drop_reasoning_between_turns: event.target.checked
                  })
                }
              />
              <span>
                Drop reasoning between turns
                <small>
                  Saves context by not resending earlier thinking. It still shows here.
                </small>
              </span>
            </label>

            {/* Everything above is the default for every model. Everything
                below belongs to the one selected, and outranks it. */}
            {advancedModel && !advancedEntry ? (
              <div className="inference-advanced">
                <p className="inference-help">Waiting for model library…</p>
              </div>
            ) : modelKind && effectiveProfile && advancedEntry ? (
              <div className="inference-advanced">
                <button
                  type="button"
                  className="inference-advanced-toggle"
                  aria-expanded={advancedOpen}
                  onClick={() => setAdvancedOpen((open) => !open)}
                >
                  {advancedOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  Show advanced
                </button>
                {advancedOpen ? (
                  <>
                    <p className="inference-help">
                      These apply to <strong>{advancedEntry.id}</strong> alone and override the defaults
                      above. Anything left blank follows them.
                    </p>
                    <ModelSettingsFields
                      modelId={advancedModel}
                      kind={modelKind}
                      engine={modelEngine(advancedEntry)}
                      profile={effectiveProfile}
                      adapters={adapters}
                      inherited={{
                        contextSize: draft.context_size,
                        batchSize: draft.batch_size,
                        temperature: draft.temperature,
                        topP: draft.top_p,
                        flashAttention: draft.flash_attention,
                        kvCacheTypeK: draft.kv_cache_type_k,
                        kvCacheTypeV: draft.kv_cache_type_v,
                        maxTokens: draft.max_tokens,
                        capabilityVision: Boolean(
                          rawAdvancedEntry?.capabilities?.input_modalities.includes('image')
                        ),
                        capabilityReasoning: Boolean(rawAdvancedEntry?.capabilities?.reasoning),
                        capabilityTools: Boolean(rawAdvancedEntry?.capabilities?.tools),
                        capabilityComputerUse: Boolean(
                          rawAdvancedEntry?.capabilities?.computer_use
                        ),
                        diffusionWidth:
                          useIntegratedDefaults && modelKind === 'video'
                            ? INTEGRATED_GPU_VIDEO_DEFAULTS.width
                            : 512,
                        diffusionHeight:
                          useIntegratedDefaults && modelKind === 'video'
                            ? INTEGRATED_GPU_VIDEO_DEFAULTS.height
                            : 512,
                        videoFrames: useIntegratedDefaults
                          ? INTEGRATED_GPU_VIDEO_DEFAULTS.frames
                          : 16,
                        vaeTiling: useIntegratedDefaults,
                        clipOnCpu: useIntegratedDefaults && modelKind === 'image',
                        diffusionFa: true,
                        autoFit: !useIntegratedDefaults,
                        maxVram:
                          useIntegratedDefaults && modelKind === 'video'
                            ? INTEGRATED_GPU_VIDEO_DEFAULTS.maxVram
                            : undefined,
                        paramsBackend:
                          useIntegratedDefaults && modelKind === 'video' ? 'cpu' : undefined,
                        streamLayers: useIntegratedDefaults && modelKind === 'video',
                        offloadToCpu: false
                      }}
                      onChange={setProfileDraft}
                      onAdapterAdded={refreshAdapters}
                      onError={setProfileError}
                    />
                  </>
                ) : null}
              </div>
            ) : null}

            {profileError ? <div className="error-banner">{profileError}</div> : null}

            <div className="inference-footer">
              <button
                type="button"
                className="popover-apply"
                disabled={!anyDirty || anySaving}
                onClick={() => void saveAll()}
              >
                {anySaving ? <LoaderCircle className="spin" size={14} /> : null}
                {unifiedSaveLabel({ dirty, profileDirty, saving: anySaving })}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

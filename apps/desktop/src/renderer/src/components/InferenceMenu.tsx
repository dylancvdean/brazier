import { ChevronDown, ChevronRight, LoaderCircle } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  listAdapters,
  saveModelProfile,
  type Adapter,
  type HardwareInfo,
  type LocalModel,
  type ModelProfile,
  type RuntimeSettings
} from '../api'
import { modelEngine, modelKindFor } from '../model-utils'
import {
  AMD_APU_VIDEO_DEFAULTS,
  usesAmdApuVulkanDefaults
} from '../runtime-defaults'
import { ModelSettingsFields, emptyProfile } from './ModelSettingsFields'

type InferenceMenuProps = {
  settings: RuntimeSettings | null
  hardware: HardwareInfo | null
  selectedModel: string
  models: LocalModel[]
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
  onApply: (settings: RuntimeSettings) => void
  /** Persist the model's own overrides, which are separate from the defaults. */
  onProfileSaved: (models: Record<string, ModelProfile>) => void
  onClose: () => void
}

type ReasoningMode = 'off' | 'on' | 'budget'

function contextPresets(maxContext: number | null | undefined): number[] {
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
 * Per-generation inference defaults (sampling, context, reasoning). These are
 * usage-side settings, deliberately separated from engine launch
 * configuration and runtime management.
 */
export function InferenceMenu({
  settings,
  hardware,
  selectedModel,
  models,
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
  const modelKind = advancedModel ? modelKindFor(advancedModel) : null
  const useApuDefaults = usesAmdApuVulkanDefaults(draft, hardware)
  const storedProfile = useMemo(
    () => profile ?? (modelKind ? emptyProfile(modelKind) : null),
    [profile, modelKind]
  )
  const effectiveProfile = profileDraft ?? storedProfile
  const profileDirty =
    profileDraft != null &&
    storedProfile != null &&
    JSON.stringify(profileDraft) !== JSON.stringify(storedProfile)

  // A different model has a different profile; whatever was half-edited for the
  // last one is not an override for this one.
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

  async function saveProfile(): Promise<void> {
    if (!profileDraft || !advancedModel) return
    setSavingProfile(true)
    setProfileError(null)
    try {
      onProfileSaved(await saveModelProfile(advancedModel, profileDraft))
      setProfileDraft(null)
    } catch (cause) {
      setProfileError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSavingProfile(false)
    }
  }

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
                      step={128}
                      value={reasoningBudget}
                      onChange={(event) => {
                        const next = Number(event.target.value)
                        setReasoningBudget(next)
                        setDraft(applyReasoningMode(draft, 'budget', next))
                      }}
                    />
                  </label>
                )}
                <p className="inference-help">
                  Thinking models can deliberate before answering. Budget mode maps to
                  llama.cpp&apos;s per-request thinking token cap.
                </p>
              </>
            )}
            <button
              className="popover-apply"
              disabled={!dirty || saving}
              onClick={() => onApply(draft)}
            >
              {saving ? <LoaderCircle className="spin" size={14} /> : null}
              {dirty ? 'Apply' : 'Up to date'}
            </button>

            {/* Everything above is the default for every model. Everything
                below belongs to the one selected, and outranks it. */}
            {modelKind && effectiveProfile && advancedEntry ? (
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
                    {profileError ? <div className="error-banner">{profileError}</div> : null}
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
                        diffusionWidth:
                          useApuDefaults && modelKind === 'video'
                            ? AMD_APU_VIDEO_DEFAULTS.width
                            : 512,
                        diffusionHeight:
                          useApuDefaults && modelKind === 'video'
                            ? AMD_APU_VIDEO_DEFAULTS.height
                            : 512,
                        videoFrames: useApuDefaults
                          ? AMD_APU_VIDEO_DEFAULTS.frames
                          : 16,
                        vaeTiling: useApuDefaults,
                        clipOnCpu: useApuDefaults && modelKind === 'image',
                        diffusionFa: useApuDefaults,
                        autoFit: false,
                        maxVram:
                          useApuDefaults && modelKind === 'video'
                            ? AMD_APU_VIDEO_DEFAULTS.maxVram
                            : undefined,
                        paramsBackend:
                          useApuDefaults && modelKind === 'video' ? 'cpu' : undefined,
                        streamLayers: useApuDefaults && modelKind === 'video',
                        offloadToCpu: false
                      }}
                      onChange={setProfileDraft}
                      onAdapterAdded={refreshAdapters}
                      onError={setProfileError}
                    />
                    <button
                      className="popover-apply"
                      disabled={!profileDirty || savingProfile}
                      onClick={() => void saveProfile()}
                    >
                      {savingProfile ? <LoaderCircle className="spin" size={14} /> : null}
                      {profileDirty ? 'Save model settings' : 'Model settings saved'}
                    </button>
                  </>
                ) : null}
              </div>
            ) : null}
          </>
        )}
      </div>
    </div>
  )
}

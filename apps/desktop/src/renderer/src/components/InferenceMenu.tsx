import { LoaderCircle } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { LocalModel, RuntimeSettings } from '../api'

type InferenceMenuProps = {
  settings: RuntimeSettings | null
  selectedModel: string
  models: LocalModel[]
  saving: boolean
  onApply: (settings: RuntimeSettings) => void
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
  selectedModel,
  models,
  saving,
  onApply,
  onClose
}: InferenceMenuProps): React.JSX.Element {
  const [draft, setDraft] = useState<RuntimeSettings | null>(settings)
  const [reasoningMode, setReasoningMode] = useState<ReasoningMode>('on')
  const [reasoningBudget, setReasoningBudget] = useState(1024)

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

  function updateReasoningMode(mode: ReasoningMode): void {
    if (!draft) return
    setReasoningMode(mode)
    setDraft(applyReasoningMode(draft, mode, reasoningBudget))
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
            <label className="field-row">
              <span>
                Context length
                {maxContext ? ` · max ${maxContext.toLocaleString()}` : ''}
              </span>
              <select
                value={draft.context_size}
                onChange={(event) =>
                  setDraft({ ...draft, context_size: Number(event.target.value) })
                }
              >
                {contextOptions.map((value) => (
                  <option key={value} value={value}>
                    {value.toLocaleString()} tokens
                  </option>
                ))}
              </select>
            </label>
            <p className="inference-help">
              Applies on the next generation. Larger context uses more memory; stay at or below
              the model&apos;s trained window when known.
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
          </>
        )}
      </div>
    </div>
  )
}

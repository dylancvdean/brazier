import {
  ArrowLeft,
  ArrowRight,
  Check,
  CircleAlert,
  LoaderCircle,
  RefreshCw,
  Wrench
} from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import brazierLogo from '../assets/brazier-logo.png'
import {
  fetchRecommendations,
  fetchToolchainStatus,
  fetchWorkspacePreference,
  hardwareInfo,
  huggingFaceTokenStatus,
  saveWorkspacePreference,
  setHuggingFaceToken,
  setupToolchain,
  type HardwareInfo,
  type RecommendationCategory,
  type RuntimeTarget,
  type Recommendations,
  type ToolchainNeeds,
  type ToolchainStatus,
  type ToolchainTool
} from '../api'
import { CATEGORY_LABELS, RecommendedModels } from './RecommendedModels'

type WelcomeScreenProps = {
  onContinue: () => void
  onOpenRuntimes: () => void
  /** Refresh the model list after the flow installs anything. */
  onModelsChanged?: () => void
}

/** The order features are offered and then walked through. */
const FEATURES: RecommendationCategory[] = ['text', 'agent', 'image', 'video', 'voice']

const FEATURE_BLURBS: Record<RecommendationCategory, string> = {
  text: 'Talk with a model running on this machine.',
  agent: 'Let a model edit files and run commands in a folder you choose.',
  image: 'Make images from a prompt.',
  video: 'Make short clips from a prompt.',
  voice: 'Speak with a model in real time.',
  computer_use: ''
}

const FEATURE_LABELS: Record<RecommendationCategory, string> = {
  ...CATEGORY_LABELS,
  voice: 'Voice (alpha)'
}

const TARGET_LABELS: Record<RuntimeTarget, string> = {
  auto: 'Auto',
  cpu: 'CPU',
  cuda: 'CUDA',
  rocm: 'ROCm',
  metal: 'Metal',
  vulkan: 'Vulkan',
  sycl: 'SYCL'
}

function platformLines(
  hardware: HardwareInfo | null,
  toolchain: ToolchainStatus | null
): string[] {
  const lines: string[] = []
  const mlx = toolchain?.platforms.mlx ?? false
  const streaming = toolchain?.platforms.streaming_asr ?? true
  if (mlx) {
    lines.push('Apple Silicon: llama.cpp, MLX, whisper.cpp, streaming speech, and video.')
  } else if (hardware?.os === 'macos') {
    lines.push('Intel Mac: llama.cpp, whisper.cpp, streaming speech, and video. MLX needs Apple Silicon.')
  } else if (hardware?.os === 'linux' || toolchain?.os.family === 'linux') {
    lines.push('Linux: llama.cpp, whisper.cpp, streaming speech, and video.')
  } else if (hardware?.os === 'windows' || toolchain?.os.family === 'windows') {
    lines.push('Windows: llama.cpp and whisper.cpp. Streaming speech and MLX aren’t available yet.')
  } else {
    lines.push('llama.cpp and whisper.cpp everywhere, MLX on Apple Silicon, streaming speech on macOS and Linux.')
  }
  if (streaming && !mlx && toolchain?.tools.some((tool) => tool.id === 'uv')) {
    lines.push('Python engines need uv on your PATH.')
  }
  return lines
}

function ToolRow({ tool }: { tool: ToolchainTool }) {
  return (
    <li className={`welcome-check ${tool.available ? 'ok' : 'missing'}`}>
      <div className="welcome-check-icon">
        {tool.available ? <Check size={14} /> : <CircleAlert size={14} />}
      </div>
      <div className="welcome-check-body">
        <div className="welcome-check-title">
          <strong>{tool.label}</strong>
          <span>{tool.available ? 'Found' : 'Missing'}</span>
        </div>
        <p>{tool.required_for}</p>
        {!tool.available && tool.install_hint && (
          <code className="welcome-hint">{tool.install_hint}</code>
        )}
      </div>
    </li>
  )
}

/**
 * The first-launch walkthrough.
 *
 * Three stages: ask what someone actually wants to do, check only the host
 * tools that choice needs, then show one recommended model per chosen feature.
 * Someone who has never run a local model should not need to know that image
 * generation, chat, voice, and source builds have different prerequisites.
 */
export function WelcomeScreen(props: WelcomeScreenProps) {
  const [toolchain, setToolchain] = useState<ToolchainStatus | null>(null)
  const [hardware, setHardware] = useState<HardwareInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [stage, setStage] = useState<'features' | 'token' | 'checklist' | 'models'>('features')
  const [wanted, setWanted] = useState<RecommendationCategory[]>(['text'])
  const [wantsComputerUse, setWantsComputerUse] = useState(false)
  const computerUseUnavailable = hardware?.os === 'windows'
  const [customRuntimes, setCustomRuntimes] = useState(false)
  const [recommendations, setRecommendations] = useState<Recommendations | null>(null)
  const [loadingRecommendations, setLoadingRecommendations] = useState(false)
  const [settingUp, setSettingUp] = useState(false)
  const [setupOutput, setSetupOutput] = useState<string | null>(null)
  const [hfTokenSource, setHfTokenSource] = useState('none')
  const [hfTokenDraft, setHfTokenDraft] = useState('')
  const [savingHfToken, setSavingHfToken] = useState(false)

  useEffect(() => {
    void huggingFaceTokenStatus()
      .then((status) => setHfTokenSource(status.source))
      .catch(() => setHfTokenSource('none'))
    void hardwareInfo()
      .then(setHardware)
      .catch(() => setHardware(null))
  }, [])

  const needs: ToolchainNeeds = {
    customRuntimes,
    voice: wanted.includes('voice'),
    computerUse: wantsComputerUse,
    video: wanted.includes('video')
  }

  // Fetched when the model stage is reached rather than on mount: it costs a
  // round trip to Hugging Face per recommended model, to size the download.
  useEffect(() => {
    if (stage !== 'models' || recommendations) return
    setLoadingRecommendations(true)
    void fetchRecommendations()
      .then(setRecommendations)
      .catch((cause: unknown) =>
        setError(cause instanceof Error ? cause.message : String(cause))
      )
      .finally(() => setLoadingRecommendations(false))
  }, [stage, recommendations])

  function toggleFeature(feature: RecommendationCategory): void {
    setWanted((current) =>
      current.includes(feature)
        ? current.filter((entry) => entry !== feature)
        : [...current, feature]
    )
  }

  async function saveHubToken(): Promise<void> {
    setSavingHfToken(true)
    setError(null)
    try {
      await setHuggingFaceToken(hfTokenDraft.trim())
      setHfTokenDraft('')
      const status = await huggingFaceTokenStatus()
      setHfTokenSource(status.source)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSavingHfToken(false)
    }
  }

  const refresh = useCallback(async (selectedNeeds: ToolchainNeeds) => {
    setLoading(true)
    setError(null)
    try {
      const [nextToolchain, nextHardware] = await Promise.all([
        fetchToolchainStatus(selectedNeeds),
        hardwareInfo().catch(() => null)
      ])
      setToolchain(nextToolchain)
      setHardware(nextHardware)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (stage === 'checklist') void refresh(needs)
  }, [refresh, stage, customRuntimes, wantsComputerUse, wanted])

  const missing = toolchain?.tools.filter((tool) => !tool.available) ?? []
  const readyCount = toolchain?.tools.filter((tool) => tool.available).length ?? 0
  const total = toolchain?.tools.length ?? 0

  if (stage === 'features') {
    return (
      <div className="first-run">
        <div className="first-run-card">
          <img className="welcome-logo" src={brazierLogo} alt="Brazier" />
          <p className="first-run-eyebrow">
            Step 1 of 4
          </p>
          <h1>Welcome to Brazier</h1>
          <p className="first-run-lede">What do you want to use it for? You can change this later.</p>

          <div className="welcome-feature-list" role="group" aria-label="Features to set up">
            {FEATURES.map((feature) => {
              const on = wanted.includes(feature)
              return (
                <button
                  key={feature}
                  type="button"
                  role="checkbox"
                  aria-checked={on}
                  className={on ? 'welcome-feature active' : 'welcome-feature'}
                  onClick={() => toggleFeature(feature)}
                >
                  <span className="welcome-feature-check">{on ? <Check size={13} /> : null}</span>
                  <span>
                    <strong>{FEATURE_LABELS[feature]}</strong>
                    <small>{FEATURE_BLURBS[feature]}</small>
                  </span>
                </button>
              )
            })}
            <button
              type="button"
              role="checkbox"
              aria-checked={wantsComputerUse}
              className={wantsComputerUse ? 'welcome-feature active' : 'welcome-feature'}
              disabled={computerUseUnavailable}
              title={
                computerUseUnavailable ? 'Not available on Windows yet' : undefined
              }
              onClick={() => {
                if (computerUseUnavailable) return
                setWantsComputerUse((current) => !current)
              }}
            >
              <span className="welcome-feature-check">{wantsComputerUse ? <Check size={13} /> : null}</span>
              <span>
                <strong>Computer use (beta)</strong>
                <small>
                  {computerUseUnavailable
                    ? 'Not available on Windows yet.'
                    : 'Let a model operate a browser or desktop.'}
                </small>
              </span>
            </button>
            <button
              type="button"
              role="checkbox"
              aria-checked={customRuntimes}
              className={customRuntimes ? 'welcome-feature active' : 'welcome-feature'}
              onClick={() => setCustomRuntimes((current) => !current)}
            >
              <span className="welcome-feature-check">{customRuntimes ? <Check size={13} /> : null}</span>
              <span>
                <strong>Custom runtimes (advanced)</strong>
                <small>Build llama.cpp, MLX, whisper.cpp, or vLLM from source.</small>
              </span>
            </button>
          </div>

          <div className="first-run-actions">
            <button
              type="button"
              className="primary-button"
              disabled={wanted.length === 0}
              onClick={() => setStage('token')}
            >
              Continue <ArrowRight size={15} />
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (stage === 'token') {
    return (
      <div className="first-run">
        <div className="first-run-card">
          <img className="welcome-logo" src={brazierLogo} alt="Brazier" />
          <p className="first-run-eyebrow">
            Step 2 of 4
          </p>
          <h1>Connect Hugging Face</h1>
          <p className="first-run-lede">
            Optional. A free token unlocks license-gated models, including the voice model and the
            best image models.
          </p>

          {error && <div className="runtime-notice">{error}</div>}

          <form
            className="build-form"
            onSubmit={(event) => {
              event.preventDefault()
              void saveHubToken()
            }}
          >
            <label>
              <span className="label-with-link">
                Access token
                <a
                  className="inline-link"
                  href="https://huggingface.co/settings/tokens"
                  target="_blank"
                  rel="noreferrer"
                >
                  Get a token
                </a>
              </span>
              <input
                type="password"
                autoComplete="off"
                value={hfTokenDraft}
                onChange={(event) => setHfTokenDraft(event.target.value)}
                placeholder={
                  hfTokenSource === 'environment'
                    ? 'Using HF_TOKEN from your environment'
                    : hfTokenSource === 'stored'
                      ? 'Saved. Paste a new one to replace it'
                      : 'hf_…'
                }
              />
            </label>
            <p className="model-help">Kept on this device and sent only to Hugging Face.</p>
            <div className="build-form-actions">
              <button
                className="secondary-action"
                type="submit"
                disabled={savingHfToken || !hfTokenDraft.trim()}
              >
                {savingHfToken ? <LoaderCircle className="spin" size={14} /> : 'Save'}
              </button>
            </div>
          </form>

          {hfTokenSource === 'stored' ? (
            <p className="welcome-token-status ok">
              <Check size={13} /> Token saved
            </p>
          ) : hfTokenSource === 'environment' ? (
            <p className="welcome-token-status ok">
              <Check size={13} /> Using HF_TOKEN from your environment
            </p>
          ) : null}

          <div className="first-run-actions">
            <button
              type="button"
              className="chip-button subtle"
              onClick={() => setStage('features')}
            >
              <ArrowLeft size={15} /> Back
            </button>
            <button
              type="button"
              className="primary-button"
              onClick={() => {
                setError(null)
                setStage('checklist')
              }}
            >
              {hfTokenSource === 'none' ? 'Skip for now' : 'Continue'} <ArrowRight size={15} />
            </button>
          </div>
          <p className="first-run-footnote">You can change this later in Manage → Discover.</p>
        </div>
      </div>
    )
  }

  if (stage === 'models') {
    return (
      <div className="first-run">
        <div className="first-run-card wide">
          <img className="welcome-logo" src={brazierLogo} alt="Brazier" />
          <p className="first-run-eyebrow">
            Step 4 of 4
          </p>
          <h1>Recommended for this machine</h1>
          <p className="first-run-lede">Downloads run in the background.</p>

          {error && <div className="runtime-notice">{error}</div>}

          {loadingRecommendations || !recommendations ? (
            <div className="manage-placeholder compact">
              <LoaderCircle className="spin" size={16} />
              Finding models that fit…
            </div>
          ) : (
            <RecommendedModels
              recommendations={recommendations}
              categories={[...wanted, ...(wantsComputerUse ? (['computer_use'] as const) : [])]}
              onInstalled={props.onModelsChanged}
              onError={setError}
              onOpenRuntimes={props.onOpenRuntimes}
            />
          )}

          {wanted.includes('voice') && recommendations && !recommendations.voice ? (
            <p className="recommendation-note warn">
              <CircleAlert size={12} />
              Voice needs a Hugging Face token. Go back to add one.
            </p>
          ) : null}

          <div className="first-run-actions">
            <button
              type="button"
              className="chip-button subtle"
              onClick={() => setStage('checklist')}
            >
              <ArrowLeft size={15} /> Back
            </button>
            <button
              type="button"
              className="primary-button"
              onClick={() => {
                void (async () => {
                  if (wantsComputerUse && !computerUseUnavailable) {
                    try {
                      const preference = await fetchWorkspacePreference()
                      if (!preference.modes.computer) {
                        await saveWorkspacePreference({ ...preference.modes, computer: true })
                      }
                    } catch {
                      // The Manage → Customization toggle remains available.
                    }
                  }
                  props.onContinue()
                })()
              }}
            >
              Start using Brazier
            </button>
          </div>
          <p className="first-run-footnote">Find more in Manage → Recommended.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="first-run">
      <div className="first-run-card">
        <img className="welcome-logo" src={brazierLogo} alt="Brazier" />
        <p className="first-run-eyebrow">
          Step 3 of 4
        </p>
        <h1>Check this machine</h1>
        <p className="first-run-lede">Only the tools your choices need.</p>

        <div className="first-run-platform">
          {platformLines(hardware, toolchain).map((line) => (
            <p key={line}>{line}</p>
          ))}
          {hardware && (
            <p className="first-run-meta">
              {[
                hardware.architecture,
                hardware.gpu,
                hardware.recommended_target && hardware.recommended_target !== 'auto'
                  ? TARGET_LABELS[hardware.recommended_target]
                  : null
              ]
                .filter(Boolean)
                .join(' · ')}
            </p>
          )}
        </div>

        <div className="first-run-section-head">
          <h2>
            <Wrench size={15} /> Tools
          </h2>
          <button
            type="button"
            className="chip-button subtle"
            onClick={() => void refresh(needs)}
            disabled={loading}
          >
            {loading ? <LoaderCircle className="spin" size={13} /> : <RefreshCw size={13} />}
            Recheck
          </button>
        </div>

        {error && <div className="runtime-notice">{error}</div>}

        {loading && !toolchain ? (
          <div className="manage-placeholder compact">
            <LoaderCircle className="spin" size={16} />
            Checking…
          </div>
        ) : (
          <>
            <p className="first-run-score">
              {total === 0
                ? 'Nothing extra to install.'
                : missing.length > 0
                  ? `${readyCount} of ${total} found`
                  : `All ${total} found`}
            </p>
            {(toolchain?.tools ?? []).length > 0 ? (
              <ul className="welcome-checklist">
                {(toolchain?.tools ?? []).map((tool) => (
                  <ToolRow key={tool.id} tool={tool} />
                ))}
              </ul>
            ) : null}
          </>
        )}

        {setupOutput && <div className="runtime-notice">{setupOutput}</div>}

        <div className="first-run-actions">
          <button type="button" className="chip-button subtle" onClick={() => setStage('token')}>
            <ArrowLeft size={15} /> Back
          </button>
          {toolchain?.os.family === 'macos' && missing.length > 0 && (
            <button
              type="button"
              className="chip-button"
              disabled={settingUp || loading}
              onClick={() => {
                setSettingUp(true)
                setError(null)
                setSetupOutput(null)
                void setupToolchain(needs)
                  .then((result) => {
                    setToolchain(result.status)
                    setSetupOutput(
                      result.output ||
                        'Homebrew setup finished. If macOS is still installing Command Line Tools, recheck when it’s done.'
                    )
                  })
                  .catch((cause: unknown) =>
                    setError(cause instanceof Error ? cause.message : String(cause))
                  )
                  .finally(() => setSettingUp(false))
              }}
            >
              {settingUp ? <LoaderCircle className="spin" size={13} /> : <Wrench size={13} />}
              {settingUp ? 'Setting up…' : 'Set up for me'}
            </button>
          )}
          <button type="button" className="chip-button" onClick={props.onOpenRuntimes}>
            Open Runtimes
          </button>
          <button type="button" className="primary-button" onClick={() => setStage('models')}>
            Choose models <ArrowRight size={15} />
          </button>
        </div>
      </div>
    </div>
  )
}

import {
  ArrowLeft,
  ArrowRight,
  Check,
  CircleAlert,
  LoaderCircle,
  RefreshCw,
  Sparkles,
  Wrench
} from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import brazierLogo from '../assets/brazier-logo.png'
import {
  fetchRecommendations,
  fetchToolchainStatus,
  hardwareInfo,
  setupToolchain,
  type HardwareInfo,
  type RecommendationCategory,
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
  text: 'Ask questions, write, and think out loud with a model running on this machine.',
  agent: 'Let a model edit files and run commands inside a folder you choose.',
  image: 'Generate pictures from a description.',
  video: 'Generate short clips from a description.',
  voice: 'Talk to a model and be answered out loud, in real time.'
}

const FEATURE_LABELS: Record<RecommendationCategory, string> = {
  ...CATEGORY_LABELS,
  voice: 'Voice (alpha)'
}

function platformLines(
  hardware: HardwareInfo | null,
  toolchain: ToolchainStatus | null
): string[] {
  const lines: string[] = []
  const mlx = toolchain?.platforms.mlx ?? false
  const streaming = toolchain?.platforms.streaming_asr ?? true
  if (mlx) {
    lines.push('Apple Silicon — llama.cpp, MLX, whisper.cpp, streaming ASR, and video are supported.')
  } else if (hardware?.os === 'macos') {
    lines.push('macOS Intel — llama.cpp, whisper.cpp, streaming ASR, and video. MLX requires Apple Silicon.')
  } else if (hardware?.os === 'linux' || toolchain?.os.family === 'linux') {
    lines.push('Linux — llama.cpp, whisper.cpp, streaming ASR, and video. MLX is macOS Apple Silicon only.')
  } else if (hardware?.os === 'windows' || toolchain?.os.family === 'windows') {
    lines.push('Windows — llama.cpp and whisper.cpp. Streaming ASR and MLX are not available here yet.')
  } else {
    lines.push('Local engines: llama.cpp and whisper.cpp everywhere; MLX on Apple Silicon; streaming ASR on macOS/Linux.')
  }
  if (streaming && !mlx && toolchain?.tools.some((tool) => tool.id === 'uv')) {
    lines.push('Python engines need uv on your PATH before you build them under Runtimes.')
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
  const [stage, setStage] = useState<'features' | 'checklist' | 'models'>('features')
  const [wanted, setWanted] = useState<RecommendationCategory[]>(['text'])
  const [wantsComputerUse, setWantsComputerUse] = useState(false)
  const [customRuntimes, setCustomRuntimes] = useState(false)
  const [recommendations, setRecommendations] = useState<Recommendations | null>(null)
  const [loadingRecommendations, setLoadingRecommendations] = useState(false)
  const [settingUp, setSettingUp] = useState(false)
  const [setupOutput, setSetupOutput] = useState<string | null>(null)

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
            <Sparkles size={13} /> Step 1 of 3
          </p>
          <h1>What do you want to do?</h1>
          <p className="first-run-lede">
            Tell Brazier what you want first. We’ll choose the right model and only check for the
            host tools those choices actually need.
          </p>

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
              onClick={() => setWantsComputerUse((current) => !current)}
            >
              <span className="welcome-feature-check">{wantsComputerUse ? <Check size={13} /> : null}</span>
              <span>
                <strong>Computer use (beta)</strong>
                <small>Let a model use a browser or desktop to complete tasks.</small>
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
                <strong>Build custom runtimes (advanced)</strong>
                <small>Build engines such as llama.cpp, MLX, whisper.cpp, or vLLM from source.</small>
              </span>
            </button>
          </div>

          <div className="first-run-actions">
            <button
              type="button"
              className="primary-button"
              disabled={wanted.length === 0}
              onClick={() => setStage('checklist')}
            >
              Check this machine <ArrowRight size={15} />
            </button>
          </div>
          <p className="first-run-footnote">
            You can change these choices later. Advanced builds are optional; managed runtimes are
            the easiest place to start.
          </p>
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
            <Sparkles size={13} /> Step 3 of 3
          </p>
          <h1>Recommended for this machine</h1>
          <p className="first-run-lede">
            One model per thing you chose, at the largest quantisation this machine can hold
            comfortably. Downloads continue in the background — you do not have to wait here.
          </p>

          {error && <div className="runtime-notice">{error}</div>}

          {loadingRecommendations || !recommendations ? (
            <div className="manage-placeholder compact">
              <LoaderCircle className="spin" size={16} />
              Sizing these against your memory…
            </div>
          ) : (
            <RecommendedModels
              recommendations={recommendations}
              categories={wanted}
              onInstalled={props.onModelsChanged}
              onError={setError}
              onOpenRuntimes={props.onOpenRuntimes}
            />
          )}

          <div className="first-run-actions">
            <button
              type="button"
              className="chip-button subtle"
              onClick={() => setStage('checklist')}
            >
              <ArrowLeft size={15} /> Back
            </button>
            <button type="button" className="primary-button" onClick={props.onContinue}>
              Continue to Brazier
            </button>
          </div>
          <p className="first-run-footnote">
            Reopen this from Manage → Recommended models whenever you want the others.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="first-run">
      <div className="first-run-card">
        <img className="welcome-logo" src={brazierLogo} alt="Brazier" />
        <p className="first-run-eyebrow">
          <Sparkles size={13} /> Step 2 of 3
        </p>
        <h1>Welcome to Brazier</h1>
        <p className="first-run-lede">
          A private workbench for local models. This check is based on what you selected, so a
          managed-runtime setup won’t ask you to install source-build tools you do not need.
        </p>

        <div className="first-run-platform">
          {platformLines(hardware, toolchain).map((line) => (
            <p key={line}>{line}</p>
          ))}
          {hardware && (
            <p className="first-run-meta">
              Detected {hardware.os}/{hardware.architecture}
              {hardware.gpu ? ` · ${hardware.gpu}` : ''}
              {hardware.recommended_target ? ` · prefer ${hardware.recommended_target}` : ''}
            </p>
          )}
        </div>

        <div className="first-run-section-head">
          <h2>
            <Wrench size={15} /> Tools this setup needs
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
            Scanning this machine…
          </div>
        ) : (
          <>
            <p className="first-run-score">
              {total === 0
                ? 'No extra host tools are needed for these choices'
                : `${readyCount}/${total} needed tools available${
                    missing.length > 0 ? ` · ${missing.length} still needed` : ' · ready to go'
                  }`}
            </p>
            {(toolchain?.tools ?? []).length > 0 ? (
              <ul className="welcome-checklist">
                {(toolchain?.tools ?? []).map((tool) => (
                  <ToolRow key={tool.id} tool={tool} />
                ))}
              </ul>
            ) : (
              <div className="first-run-platform">Managed runtimes are ready to handle the rest.</div>
            )}
          </>
        )}

        {setupOutput && <div className="runtime-notice">{setupOutput}</div>}

        <div className="first-run-actions">
          <button type="button" className="chip-button subtle" onClick={() => setStage('features')}>
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
                    setSetupOutput(result.output || 'Homebrew setup finished. Recheck if macOS is still installing Command Line Tools.')
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
        <p className="first-run-footnote">
          You can reopen this screen anytime with{' '}
          <code>pnpm dev:welcome</code> or <code>BRAZIER_FORCE_WELCOME=1</code>.
        </p>
      </div>
    </div>
  )
}

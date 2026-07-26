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
  type HardwareInfo,
  type RecommendationCategory,
  type Recommendations,
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

function platformLines(hardware: HardwareInfo | null, toolchain: ToolchainStatus | null): string[] {
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
  if (streaming && !mlx) {
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
 * Three stages: check this machine has the tools each engine needs, ask what
 * you actually want to do with it, then show one recommended model per thing
 * you chose. The middle stage is the point — someone who has never run a local
 * model does not know that image generation and chat are different downloads,
 * or that voice needs two of them.
 */
export function WelcomeScreen(props: WelcomeScreenProps) {
  const [toolchain, setToolchain] = useState<ToolchainStatus | null>(null)
  const [hardware, setHardware] = useState<HardwareInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [stage, setStage] = useState<'checklist' | 'features' | 'models'>('checklist')
  const [wanted, setWanted] = useState<RecommendationCategory[]>(['text'])
  const [recommendations, setRecommendations] = useState<Recommendations | null>(null)
  const [loadingRecommendations, setLoadingRecommendations] = useState(false)

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

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [nextToolchain, nextHardware] = await Promise.all([
        fetchToolchainStatus(),
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
    void refresh()
  }, [refresh])

  const missing = toolchain?.tools.filter((tool) => !tool.available) ?? []
  const readyCount = toolchain?.tools.filter((tool) => tool.available).length ?? 0
  const total = toolchain?.tools.length ?? 0

  if (stage === 'features') {
    return (
      <div className="first-run">
        <div className="first-run-card">
          <img className="welcome-logo" src={brazierLogo} alt="Brazier" />
          <p className="first-run-eyebrow">
            <Sparkles size={13} /> Step 2 of 3
          </p>
          <h1>What do you want to do?</h1>
          <p className="first-run-lede">
            Each has different model or runtime requirements, so Brazier only downloads what you
            will use. Chat and Agent can share a model when the same one is the best fit.
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
                    <strong>{CATEGORY_LABELS[feature]}</strong>
                    <small>{FEATURE_BLURBS[feature]}</small>
                  </span>
                </button>
              )
            })}
          </div>

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
              disabled={wanted.length === 0}
              onClick={() => setStage('models')}
            >
              See what fits this machine <ArrowRight size={15} />
            </button>
          </div>
          <p className="first-run-footnote">
            Choosing nothing is allowed — skip ahead and pick models yourself from Discover.
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
              onClick={() => setStage('features')}
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
          <Sparkles size={13} /> Step 1 of 3
        </p>
        <h1>Welcome to Brazier</h1>
        <p className="first-run-lede">
          A private workbench for local models. Before you download weights, make sure this machine
          has the tools each engine needs.
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
            <Wrench size={15} /> System checklist
          </h2>
          <button
            type="button"
            className="chip-button subtle"
            onClick={() => void refresh()}
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
              {readyCount}/{total} ready
              {missing.length > 0
                ? ` · ${missing.length} to install for full coverage`
                : ' · you can use every current engine path'}
            </p>
            <ul className="welcome-checklist">
              {(toolchain?.tools ?? []).map((tool) => (
                <ToolRow key={tool.id} tool={tool} />
              ))}
            </ul>
          </>
        )}

        <div className="first-run-actions">
          <button type="button" className="chip-button" onClick={props.onOpenRuntimes}>
            Open Runtimes
          </button>
          <button type="button" className="primary-button" onClick={() => setStage('features')}>
            Choose what to set up <ArrowRight size={15} />
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

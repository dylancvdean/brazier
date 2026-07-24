import {
  Check,
  CircleAlert,
  LoaderCircle,
  RefreshCw,
  Sparkles,
  Wrench
} from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import {
  fetchToolchainStatus,
  hardwareInfo,
  type HardwareInfo,
  type ToolchainStatus,
  type ToolchainTool
} from '../api'

type WelcomeScreenProps = {
  onContinue: () => void
  onOpenRuntimes: () => void
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

export function WelcomeScreen(props: WelcomeScreenProps) {
  const [toolchain, setToolchain] = useState<ToolchainStatus | null>(null)
  const [hardware, setHardware] = useState<HardwareInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

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

  return (
    <div className="first-run">
      <div className="first-run-card">
        <div className="welcome-mark">B</div>
        <p className="first-run-eyebrow">
          <Sparkles size={13} /> First launch
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

        <div className="first-run-next">
          <h2>What to do next</h2>
          <ol>
            <li>Install any missing tools above, then hit Recheck.</li>
            <li>
              Open <strong>Runtimes</strong> to install llama.cpp (and whisper.cpp / streaming ASR /
              MLX as you need them).
            </li>
            <li>
              Open <strong>Discover</strong> to download a model, then start chatting.
            </li>
          </ol>
        </div>

        <div className="first-run-actions">
          <button type="button" className="chip-button" onClick={props.onOpenRuntimes}>
            Open Runtimes
          </button>
          <button type="button" className="primary-button" onClick={props.onContinue}>
            Continue to Brazier
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

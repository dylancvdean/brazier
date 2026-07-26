import { Download, Image, LoaderCircle, Square, Video } from 'lucide-react'
import { type FormEvent, useEffect, useState } from 'react'
import {
  cancelGeneration,
  fetchBlobObjectUrl,
  fetchModelSettings,
  generateImage,
  generateVideo,
  listSdcppBundles,
  saveBlobToDisk,
  type DiffusionProfile,
  type GenerateBlobResult,
  type HardwareInfo,
  type LocalModel,
  type RuntimeSettings,
  type SdcppDefaults
} from '../api'
import {
  AMD_APU_VIDEO_DEFAULTS,
  usesAmdApuVulkanDefaults
} from '../runtime-defaults'

type Modality = 'image' | 'video'

type Props = {
  /** Installed models for the active modality. */
  models: LocalModel[]
  modality: Modality
  onModalityChange: (modality: Modality) => void
  /** Model chosen in the top bar; empty when none is installed. */
  modelId: string
  settings: RuntimeSettings | null
  hardware: HardwareInfo | null
  onError: (message: string | null) => void
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

/** Whether a failed generation was simply stopped by the user. */
function isCancellation(cause: unknown): boolean {
  return errorText(cause).toLowerCase().includes('stopped by the user')
}

export function GenerateMode(props: Props) {
  const modality = props.modality
  const [prompt, setPrompt] = useState('')
  const [negative, setNegative] = useState('')
  const [width, setWidth] = useState(512)
  const [height, setHeight] = useState(512)
  const [steps, setSteps] = useState(20)
  const [frames, setFrames] = useState(16)
  const [seed, setSeed] = useState('')
  const [cfgScale, setCfgScale] = useState(7)
  const [guidance, setGuidance] = useState('')
  const [fps, setFps] = useState(24)
  const [busy, setBusy] = useState(false)
  const [stopping, setStopping] = useState(false)
  const [saving, setSaving] = useState<string | null>(null)
  const [saved, setSaved] = useState<Record<string, string>>({})
  const [results, setResults] = useState<GenerateBlobResult[]>([])
  const [urls, setUrls] = useState<Record<string, string>>({})
  const [defaultsByModel, setDefaultsByModel] = useState<Record<string, SdcppDefaults>>({})
  const [configuredByModel, setConfiguredByModel] = useState<Record<string, DiffusionProfile>>({})

  useEffect(() => {
    let cancelled = false
    void (async () => {
      const next: Record<string, string> = {}
      for (const result of results) {
        if (urls[result.blob.sha256]) {
          next[result.blob.sha256] = urls[result.blob.sha256]
          continue
        }
        try {
          next[result.blob.sha256] = await fetchBlobObjectUrl(result.blob.sha256)
        } catch {
          // ignore individual load failures
        }
      }
      if (!cancelled) setUrls((current) => ({ ...current, ...next }))
    })()
    return () => {
      cancelled = true
    }
  }, [results])

  // What this model has been configured with, which outranks the curated
  // defaults below: a size or step count chosen for a model is a decision, and
  // the panel should open showing it.
  useEffect(() => {
    void fetchModelSettings()
      .then((response) =>
        setConfiguredByModel(
          Object.fromEntries(
            Object.entries(response.models).flatMap(([modelId, profile]) =>
              profile.kind === 'image' || profile.kind === 'video'
                ? [[modelId, profile as DiffusionProfile]]
                : []
            )
          )
        )
      )
      .catch(() => {
        // Non-fatal: the panel keeps the curated defaults.
      })
  }, [])

  // Curated models carry the settings they expect — most importantly CFG,
  // which has to be 1.0 for distilled models like Flux schnell.
  useEffect(() => {
    void listSdcppBundles()
      .then((bundles) =>
        setDefaultsByModel(
          Object.fromEntries(bundles.map((bundle) => [bundle.model_id, bundle.defaults]))
        )
      )
      .catch(() => {
        // Non-fatal: the panel just keeps whatever settings are on screen.
      })
  }, [])

  const available = props.models
  const selected = props.modelId
  const useApuDefaults = usesAmdApuVulkanDefaults(props.settings, props.hardware)

  useEffect(() => {
    const curated = defaultsByModel[selected]
    const configured = configuredByModel[selected]
    if (!curated && !configured) return
    const apuWidth = modality === 'video' ? AMD_APU_VIDEO_DEFAULTS.width : 512
    const apuHeight = modality === 'video' ? AMD_APU_VIDEO_DEFAULTS.height : 512
    const width =
      configured?.width ??
      (useApuDefaults ? Math.min(curated?.width ?? apuWidth, apuWidth) : curated?.width)
    const height =
      configured?.height ??
      (useApuDefaults ? Math.min(curated?.height ?? apuHeight, apuHeight) : curated?.height)
    const steps = configured?.steps ?? curated?.steps
    const cfg = configured?.cfg_scale ?? curated?.cfg_scale
    const guidance = configured?.guidance ?? curated?.guidance
    const frames =
      configured?.video_frames ??
      (useApuDefaults && modality === 'video'
        ? Math.min(
            curated?.video_frames ?? AMD_APU_VIDEO_DEFAULTS.frames,
            AMD_APU_VIDEO_DEFAULTS.frames
          )
        : curated?.video_frames)
    const fps = configured?.fps ?? curated?.fps
    if (width) setWidth(width)
    if (height) setHeight(height)
    if (steps) setSteps(steps)
    if (cfg != null) setCfgScale(cfg)
    setGuidance(guidance != null ? String(guidance) : '')
    if (frames) setFrames(frames)
    if (fps) setFps(fps)
  }, [selected, modality, defaultsByModel, configuredByModel, useApuDefaults])

  async function onSubmit(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (!prompt.trim() || !selected) return
    setBusy(true)
    props.onError(null)
    try {
      const body = {
        prompt: prompt.trim(),
        model_id: selected,
        negative_prompt: negative.trim() || undefined,
        width,
        height,
        steps,
        seed: seed.trim() ? Number(seed) : undefined,
        cfg_scale: cfgScale,
        guidance: guidance.trim() ? Number(guidance) : undefined,
        video_frames: frames,
        fps
      }
      const result =
        modality === 'image' ? await generateImage(body) : await generateVideo(body)
      setResults((current) => [result, ...current])
    } catch (cause) {
      // Stopping it yourself is not an error worth a red banner.
      props.onError(isCancellation(cause) ? null : errorText(cause))
    } finally {
      setBusy(false)
    }
  }

  async function save(result: GenerateBlobResult): Promise<void> {
    setSaving(result.blob.sha256)
    try {
      const path = await saveBlobToDisk(
        result.blob.sha256,
        result.blob.mime_type,
        result.blob.original_name
      )
      // Dismissing the dialog is a decision, not a failure.
      if (path) setSaved((current) => ({ ...current, [result.blob.sha256]: path }))
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setSaving(null)
    }
  }

  async function stop(): Promise<void> {
    setStopping(true)
    try {
      await cancelGeneration()
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setStopping(false)
    }
  }

  return (
    <section className="mode-panel generate-mode">
      <header className="mode-panel-header">
        <h2>Generate</h2>
        <p>Run local image and video models through stable-diffusion.cpp.</p>
      </header>

      <div className="mode-toggle" role="tablist">
        <button
          type="button"
          role="tab"
          className={modality === 'image' ? 'active' : ''}
          aria-selected={modality === 'image'}
          onClick={() => props.onModalityChange('image')}
        >
          <Image size={16} /> Image
        </button>
        <button
          type="button"
          role="tab"
          className={modality === 'video' ? 'active' : ''}
          aria-selected={modality === 'video'}
          onClick={() => props.onModalityChange('video')}
        >
          <Video size={16} /> Video
        </button>
      </div>

      {available.length === 0 ? (
        <p className="mode-empty">
          No {modality} generation models installed yet. Download one from Manage → Discover
          (stable-diffusion.cpp) and set a default in Manage → Engine.
        </p>
      ) : (
        <form className="generate-form" onSubmit={(event) => void onSubmit(event)}>
          <label>
            Prompt
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              rows={4}
              required
              placeholder={
                modality === 'image'
                  ? 'A watercolor fox in a birch forest at dawn'
                  : 'A drone shot flying over coastal cliffs at sunset'
              }
            />
          </label>
          <label>
            Negative prompt
            <input
              value={negative}
              onChange={(event) => setNegative(event.target.value)}
              placeholder="Optional"
            />
          </label>
          <div className="generate-params">
            <label>
              Width
              <input
                type="number"
                min={64}
                max={2048}
                value={width}
                onChange={(event) => setWidth(Number(event.target.value))}
              />
            </label>
            <label>
              Height
              <input
                type="number"
                min={64}
                max={2048}
                value={height}
                onChange={(event) => setHeight(Number(event.target.value))}
              />
            </label>
            <label>
              Steps
              <input
                type="number"
                min={1}
                max={150}
                value={steps}
                onChange={(event) => setSteps(Number(event.target.value))}
              />
            </label>
            <label title="Classifier-free guidance. Distilled models (Flux schnell, 4-step Wan) need 1.0; SDXL likes 5–8.">
              CFG scale
              <input
                type="number"
                min={0}
                max={30}
                step={0.5}
                value={cfgScale}
                onChange={(event) => setCfgScale(Number(event.target.value))}
              />
            </label>
            <label title="Distilled guidance, used by Flux-family models instead of CFG. Leave blank for other architectures.">
              Guidance
              <input
                value={guidance}
                onChange={(event) => setGuidance(event.target.value)}
                placeholder="Model default"
              />
            </label>
            {modality === 'video' ? (
              <>
                <label>
                  Frames
                  <input
                    type="number"
                    min={1}
                    max={241}
                    value={frames}
                    onChange={(event) => setFrames(Number(event.target.value))}
                  />
                </label>
                <label title="Playback rate written into the clip.">
                  FPS
                  <input
                    type="number"
                    min={1}
                    max={60}
                    value={fps}
                    onChange={(event) => setFps(Number(event.target.value))}
                  />
                </label>
              </>
            ) : null}
            <label>
              Seed
              <input
                value={seed}
                onChange={(event) => setSeed(event.target.value)}
                placeholder="Random"
              />
            </label>
          </div>
          <div className="generate-submit">
            <button type="submit" className="primary" disabled={busy || !prompt.trim()}>
              {busy ? (
                <>
                  <LoaderCircle className="spin" size={16} /> Generating…
                </>
              ) : (
                `Generate ${modality}`
              )}
            </button>
            {busy && (
              <button type="button" onClick={() => void stop()} disabled={stopping}>
                <Square size={14} fill="currentColor" />
                {stopping ? 'Stopping…' : 'Stop'}
              </button>
            )}
          </div>
        </form>
      )}

      {results.length > 0 ? (
        <div className="generate-gallery">
          {results.map((result) => {
            const url = urls[result.blob.sha256]
            return (
              <figure key={result.blob.sha256} className="generate-card">
                {!url ? (
                  <div className="manage-placeholder">Loading…</div>
                ) : result.blob.mime_type.startsWith('video/') ? (
                  <video src={url} controls playsInline />
                ) : (
                  <img src={url} alt="Generated output" />
                )}
                <figcaption>
                  <span>
                    {result.blob.mime_type} · {(result.blob.size_bytes / 1024).toFixed(0)} KB
                  </span>
                  <button
                    type="button"
                    className="chip-button subtle"
                    disabled={saving === result.blob.sha256}
                    onClick={() => void save(result)}
                  >
                    <Download size={12} />
                    {saved[result.blob.sha256] ? 'Saved' : 'Save'}
                  </button>
                </figcaption>
              </figure>
            )
          })}
        </div>
      ) : null}
    </section>
  )
}

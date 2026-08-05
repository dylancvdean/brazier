import { Download, Image, LoaderCircle, Music, Paperclip, Square, Video, X } from 'lucide-react'
import { type FormEvent, useEffect, useRef, useState } from 'react'
import { MediaFullscreenExit, MediaFullscreenIcon, useFullscreen } from './FullscreenButton'
import {
  cancelGeneration,
  fetchBlobObjectUrl,
  fetchModelSettings,
  generateImage,
  generateVideo,
  listSdcppBundles,
  saveBlobToDisk,
  uploadAttachmentBlob,
  type DiffusionProfile,
  type GenerateBlobResult,
  type HardwareInfo,
  type LocalModel,
  type RuntimeSettings,
  type SdcppBundle,
  type SdcppDefaults,
  type StoredBlob
} from '../api'
import {
  AMD_APU_VIDEO_DEFAULTS,
  usesAmdApuVulkanDefaults
} from '../runtime-defaults'
import type { GenerateHistoryEntry } from './GenerateHistorySidebar'

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
  history: GenerateHistoryEntry[]
  activeHistoryId: string | null
  onGenerated: (entry: GenerateHistoryEntry) => void
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

function GenerateCard({
  result,
  url,
  saving,
  saved,
  onSave
}: {
  result: GenerateBlobResult
  url?: string
  saving: boolean
  saved: boolean
  onSave: (result: GenerateBlobResult) => void
}): React.JSX.Element {
  const { setRef, active, toggle } = useFullscreen<HTMLElement>()
  const isVideo = result.blob.mime_type.startsWith('video/')
  return (
    <figure className="generate-card">
      {!url ? (
        <div className="manage-placeholder">Loading…</div>
      ) : isVideo ? (
        <video ref={setRef} src={url} controls playsInline />
      ) : (
        <div
          className={`generate-card-preview${active ? ' media-fullscreen' : ''}`}
          ref={setRef}
        >
          <img src={url} alt="Generated output" />
          {!active && <MediaFullscreenIcon active={active} toggle={toggle} />}
          {active && <MediaFullscreenExit toggle={toggle} />}
        </div>
      )}
      <figcaption>
        <span>
          {result.blob.mime_type} · {(result.blob.size_bytes / 1024).toFixed(0)} KB
        </span>
        <button
          type="button"
          className="chip-button subtle"
          disabled={saving}
          onClick={() => onSave(result)}
        >
          <Download size={12} />
          {saved ? 'Saved' : 'Save'}
        </button>
      </figcaption>
    </figure>
  )
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
  const [bundlesByModel, setBundlesByModel] = useState<Record<string, SdcppBundle>>({})
  const [configuredByModel, setConfiguredByModel] = useState<Record<string, DiffusionProfile>>({})
  const initImageInput = useRef<HTMLInputElement>(null)
  const endImageInput = useRef<HTMLInputElement>(null)
  const refImageInput = useRef<HTMLInputElement>(null)
  const refVideoInput = useRef<HTMLInputElement>(null)
  const refAudioInput = useRef<HTMLInputElement>(null)
  /** First-frame image for image-to-video conditioning. */
  const [initImage, setInitImage] = useState<StoredBlob | null>(null)
  /** Last-frame image for first/last-frame conditioning. */
  const [endImage, setEndImage] = useState<StoredBlob | null>(null)
  /** Reference images for Ref2VA conditioning. */
  const [refImages, setRefImages] = useState<StoredBlob[]>([])
  /** Reference videos for Ref2VA, each with an optional paired soundtrack. */
  const [refVideos, setRefVideos] = useState<{ blob: StoredBlob; soundtrack: StoredBlob | null }[]>([])
  /** Standalone audio references for Ref2VA conditioning. */
  const [refAudios, setRefAudios] = useState<StoredBlob[]>([])
  const [uploading, setUploading] = useState<string | null>(null)

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
      .then((bundles) => {
        setDefaultsByModel(
          Object.fromEntries(bundles.map((bundle) => [bundle.model_id, bundle.defaults]))
        )
        setBundlesByModel(
          Object.fromEntries(bundles.map((bundle) => [bundle.model_id, bundle]))
        )
      })
      .catch(() => {
        // Non-fatal: the panel just keeps whatever settings are on screen.
      })
  }, [])

  const available = props.models
  const selected = props.modelId
  const useApuDefaults = usesAmdApuVulkanDefaults(props.settings, props.hardware)
  const selectedBundle = bundlesByModel[selected]
  const conditioning = selectedBundle?.conditioning ?? null

  // Conditioning inputs belong to the previous model; clear them on change.
  useEffect(() => {
    setInitImage(null)
    setEndImage(null)
    setRefImages([])
    setRefVideos([])
    setRefAudios([])
  }, [selected])

  async function uploadFile(
    file: File,
    onUploaded: (blob: StoredBlob) => void
  ): Promise<void> {
    setUploading(file.name)
    try {
      onUploaded(await uploadAttachmentBlob(file))
    } finally {
      setUploading(null)
    }
  }

  function pickFiles(
    input: HTMLInputElement | null,
    accept: string[],
    onFiles: (files: File[]) => void
  ): void {
    if (!input) return
    input.accept = accept.join(',')
    input.onchange = () => {
      const files = Array.from(input.files ?? [])
      input.value = ''
      if (files.length > 0) onFiles(files)
    }
    input.click()
  }

  function attachImage(
    input: HTMLInputElement | null,
    setter: (blob: StoredBlob) => void
  ): void {
    pickFiles(input, ['image/png', 'image/jpeg', 'image/webp'], (files) => {
      for (const file of files) void uploadFile(file, setter)
    })
  }

  function attachRefVideos(files: File[]): void {
    for (const file of files) {
      // Upload first, then append — an entry must never be rendered before
      // its blob exists.
      void uploadFile(file, (blob) => {
        setRefVideos((current) =>
          current.length >= 3 ? current : [...current, { blob, soundtrack: null }]
        )
      })
    }
  }

  function attachSoundtrack(entry: { blob: StoredBlob; soundtrack: StoredBlob | null }): void {
    pickFiles(refAudioInput.current, ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/flac'], (files) => {
      for (const file of files) {
        void uploadFile(file, (blob) => {
          setRefVideos((latest) =>
            latest.map((candidate) =>
              candidate === entry ? { ...candidate, soundtrack: blob } : candidate
            )
          )
        })
      }
    })
  }

  useEffect(() => {
    if (!props.activeHistoryId) {
      setResults([])
      return
    }
    const entry = props.history.find((candidate) => candidate.id === props.activeHistoryId)
    if (!entry) return
    setPrompt(entry.prompt)
    setNegative(entry.negativePrompt)
    setResults([entry.result])
  }, [props.activeHistoryId, props.history])

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
        fps,
        ...(initImage ? { init_image_blob: initImage.sha256 } : {}),
        ...(endImage ? { end_image_blob: endImage.sha256 } : {}),
        ...(refImages.length > 0 ? { ref_image_blobs: refImages.map((blob) => blob.sha256) } : {}),
        ...(refVideos.length > 0
          ? { ref_video_blobs: refVideos.map((entry) => entry.blob.sha256) }
          : {}),
        ...(refVideos.some((entry) => entry.soundtrack)
          ? {
              ref_video_audio_blobs: refVideos.flatMap((entry) =>
                entry.soundtrack ? [entry.soundtrack.sha256] : []
              )
            }
          : {}),
        ...(refAudios.length > 0 ? { ref_audio_blobs: refAudios.map((blob) => blob.sha256) } : {})
      }
      const result =
        modality === 'image' ? await generateImage(body) : await generateVideo(body)
      setResults((current) => [result, ...current])
      props.onGenerated({
        id: crypto.randomUUID(),
        prompt: body.prompt,
        negativePrompt: body.negative_prompt ?? '',
        modality,
        result,
        createdAt: new Date().toISOString()
      })
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

          {modality === 'video' && conditioning !== null && conditioning !== 'text' && (
            <div className="generate-conditioning">
              <div className="generate-conditioning-head">
                <Paperclip size={13} />
                <strong>
                  {conditioning === 'first_last_frame'
                    ? 'First / last frame'
                    : conditioning === 'references'
                      ? 'Reference inputs'
                      : 'Starting image'}
                </strong>
                <span>
                  {conditioning === 'first_last_frame'
                    ? 'Start from an image, and optionally pin the final frame too.'
                    : conditioning === 'references'
                      ? 'Images, videos, and audio that steer the result (max 9 images, 3 videos, 3 audio).'
                      : 'Animate a photo you supply.'}
                </span>
              </div>

              {conditioning === 'init_image' ? (
                <div className="conditioning-row">
                  <button
                    type="button"
                    className="chip-button subtle"
                    onClick={() => attachImage(initImageInput.current, setInitImage)}
                  >
                    {uploading ? (
                      <LoaderCircle className="spin" size={13} />
                    ) : (
                      <Image size={13} />
                    )}
                    {initImage ? 'Change starting image' : 'Starting image'}
                  </button>
                  {initImage && (
                    <span className="conditioning-file">
                      {initImage.original_name ?? 'starting image'}
                      <button
                        type="button"
                        aria-label="Remove starting image"
                        onClick={() => setInitImage(null)}
                      >
                        <X size={13} />
                      </button>
                    </span>
                  )}
                </div>
              ) : conditioning === 'first_last_frame' ? (
                <div className="conditioning-row">
                  <button
                    type="button"
                    className="chip-button subtle"
                    onClick={() => attachImage(initImageInput.current, setInitImage)}
                  >
                    {uploading ? (
                      <LoaderCircle className="spin" size={13} />
                    ) : (
                      <Image size={13} />
                    )}
                    {initImage ? 'Change first frame' : 'First frame'}
                  </button>
                  {initImage && (
                    <span className="conditioning-file">
                      {initImage.original_name ?? 'first frame'}
                      <button
                        type="button"
                        aria-label="Remove first frame"
                        onClick={() => setInitImage(null)}
                      >
                        <X size={13} />
                      </button>
                    </span>
                  )}
                  <button
                    type="button"
                    className="chip-button subtle"
                    onClick={() => attachImage(endImageInput.current, setEndImage)}
                  >
                    <Image size={13} />
                    {endImage ? 'Change last frame' : 'Last frame (optional)'}
                  </button>
                  {endImage && (
                    <span className="conditioning-file">
                      {endImage.original_name ?? 'last frame'}
                      <button
                        type="button"
                        aria-label="Remove last frame"
                        onClick={() => setEndImage(null)}
                      >
                        <X size={13} />
                      </button>
                    </span>
                  )}
                </div>
              ) : conditioning === 'references' ? (
                <div className="conditioning-stacks">
                  <div className="conditioning-row">
                    <button
                      type="button"
                      className="chip-button subtle"
                      onClick={() =>
                        pickFiles(refImageInput.current, ['image/png', 'image/jpeg', 'image/webp'], (files) => {
                          for (const file of files) {
                            if (refImages.length >= 9) break
                            void uploadFile(file, (blob) =>
                              setRefImages((current) => [...current, blob])
                            )
                          }
                        })
                      }
                      disabled={refImages.length >= 9}
                    >
                      <Image size={13} /> Reference images ({refImages.length}/9)
                    </button>
                    {refImages.map((blob) => (
                      <span className="conditioning-file" key={blob.sha256}>
                        {blob.original_name ?? 'image'}
                        <button
                          type="button"
                          aria-label="Remove reference image"
                          onClick={() =>
                            setRefImages((current) => current.filter((candidate) => candidate !== blob))
                          }
                        >
                          <X size={13} />
                        </button>
                      </span>
                    ))}
                  </div>
                  <div className="conditioning-row">
                    <button
                      type="button"
                      className="chip-button subtle"
                      onClick={() =>
                        pickFiles(
                          refVideoInput.current,
                          ['video/mp4', 'video/webm', 'video/quicktime', 'video/x-matroska'],
                          (files) => attachRefVideos(files.slice(0, 3 - refVideos.length))
                        )
                      }
                      disabled={refVideos.length >= 3}
                    >
                      <Video size={13} /> Reference videos ({refVideos.length}/3)
                    </button>
                    {refVideos.map((entry) => (
                      <span className="conditioning-file" key={entry.blob.sha256}>
                        {entry.blob.original_name ?? 'video'}
                        <button
                          type="button"
                          className="chip-button subtle"
                          onClick={() => attachSoundtrack(entry)}
                        >
                          <Music size={12} />
                          {entry.soundtrack ? 'Change soundtrack' : 'Soundtrack'}
                        </button>
                        <button
                          type="button"
                          aria-label="Remove reference video"
                          onClick={() =>
                            setRefVideos((current) => current.filter((candidate) => candidate !== entry))
                          }
                        >
                          <X size={13} />
                        </button>
                      </span>
                    ))}
                  </div>
                  <div className="conditioning-row">
                    <button
                      type="button"
                      className="chip-button subtle"
                      onClick={() =>
                        pickFiles(
                          refAudioInput.current,
                          ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/flac', 'audio/ogg'],
                          (files) => {
                            for (const file of files) {
                              if (refAudios.length >= 3) break
                              void uploadFile(file, (blob) =>
                                setRefAudios((current) => [...current, blob])
                              )
                            }
                          }
                        )
                      }
                      disabled={refAudios.length >= 3}
                    >
                      <Music size={13} /> Reference audio ({refAudios.length}/3)
                    </button>
                    {refAudios.map((blob) => (
                      <span className="conditioning-file" key={blob.sha256}>
                        {blob.original_name ?? 'audio'}
                        <button
                          type="button"
                          aria-label="Remove reference audio"
                          onClick={() =>
                            setRefAudios((current) => current.filter((candidate) => candidate !== blob))
                          }
                        >
                          <X size={13} />
                        </button>
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          )}

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
          <input ref={initImageInput} type="file" style={{ display: 'none' }} />
          <input ref={endImageInput} type="file" style={{ display: 'none' }} />
          <input ref={refImageInput} type="file" style={{ display: 'none' }} multiple />
          <input ref={refVideoInput} type="file" style={{ display: 'none' }} multiple />
          <input ref={refAudioInput} type="file" style={{ display: 'none' }} multiple />
        </form>
      )}

      {results.length > 0 ? (
        <div className="generate-gallery">
          {results.map((result) => (
            <GenerateCard
              key={result.blob.sha256}
              result={result}
              url={urls[result.blob.sha256]}
              saving={saving === result.blob.sha256}
              saved={Boolean(saved[result.blob.sha256])}
              onSave={(target) => void save(target)}
            />
          ))}
        </div>
      ) : null}
    </section>
  )
}

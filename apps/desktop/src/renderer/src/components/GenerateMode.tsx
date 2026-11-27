import {
  Download,
  Image,
  LoaderCircle,
  Lock,
  LockOpen,
  Music,
  Paperclip,
  Shuffle,
  Square,
  Video,
  X
} from 'lucide-react'
import { type FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import { MediaFullscreenExit, MediaFullscreenIcon, useFullscreen } from './FullscreenButton'
import {
  cancelGeneration,
  generateImage,
  generateVideo,
  listSdcppBundles,
  saveBlobToDisk,
  uploadAttachmentBlob,
  type DiffusionProfile,
  type GenerateBlobResult,
  type HardwareInfo,
  type LocalModel,
  type ModelProfile,
  type RuntimeSettings,
  type SdcppBundle,
  type SdcppDefaults,
  type StoredBlob
} from '../api'
import {
  INTEGRATED_GPU_VIDEO_DEFAULTS,
  usesIntegratedGpuVulkanDefaults
} from '../runtime-defaults'
import type { GenerateHistoryEntry } from './GenerateHistorySidebar'
import type { ManageSection } from './ManagePanel'
import { useBlobUrls } from '../useBlobUrls'

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
  /** Live per-model overrides from App state (replaces stale one-shot fetch). */
  modelProfiles?: Record<string, ModelProfile>
  onOpenManage?: (section: ManageSection) => void
  onError: (message: string | null) => void
  history: GenerateHistoryEntry[]
  activeHistoryId: string | null
  onGenerated: (entry: GenerateHistoryEntry) => void
}

const IMAGE_RESOLUTION_PRESETS: ReadonlyArray<readonly [number, number, string]> = [
  [512, 512, '512×512'],
  [768, 768, '768×768'],
  [1024, 1024, '1024×1024'],
  [1024, 768, '1024×768 · 4:3'],
  [768, 1024, '768×1024 · 3:4'],
  [1280, 720, '1280×720 · 16:9']
]

const VIDEO_RESOLUTION_PRESETS: ReadonlyArray<readonly [number, number, string]> = [
  [512, 512, '512×512'],
  [640, 360, '640×360 · 16:9'],
  [832, 480, '832×480 · 16:9'],
  [480, 832, '480×832 · 9:16'],
  [1280, 720, '1280×720 · 720p']
]

type ParsedGenerationMeta = {
  width?: number
  height?: number
  steps?: number
  seed?: number
}

function parseGenerationMetadata(raw: unknown): ParsedGenerationMeta {
  if (!raw || typeof raw !== 'object') return {}
  const record = raw as Record<string, unknown>
  const num = (key: string): number | undefined => {
    const value = record[key]
    return typeof value === 'number' && Number.isFinite(value) ? value : undefined
  }
  return {
    width: num('width'),
    height: num('height'),
    steps: num('steps'),
    seed: num('seed')
  }
}

function formatGenerationMetadata(meta: ParsedGenerationMeta): string | null {
  const parts: string[] = []
  if (meta.width && meta.height) parts.push(`${meta.width}×${meta.height}`)
  if (meta.steps != null) parts.push(`${meta.steps} steps`)
  if (meta.seed != null) parts.push(`seed ${meta.seed}`)
  return parts.length > 0 ? parts.join(' · ') : null
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

function GenerateCard({
  result,
  url,
  failed,
  saving,
  saved,
  onSave,
  onUseSeed
}: {
  result: GenerateBlobResult
  url?: string
  failed: boolean
  saving: boolean
  saved: boolean
  onSave: (result: GenerateBlobResult) => void
  onUseSeed?: (seed: number) => void
}): React.JSX.Element {
  const { setRef, active, toggle } = useFullscreen<HTMLElement>()
  const isVideo = result.blob.mime_type.startsWith('video/')
  const meta = parseGenerationMetadata(result.metadata)
  const metaText = formatGenerationMetadata(meta)
  return (
    <figure className="generate-card">
      {!url ? (
        <div className="manage-placeholder">
          {failed ? 'Preview unavailable. You can still save this file.' : 'Loading…'}
        </div>
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
        <div className="generate-card-meta">
          <span>
            {metaText ? `${metaText} · ` : ''}
            {(result.blob.size_bytes / 1024).toFixed(0)} KB
          </span>
        </div>
        <div className="generate-card-actions">
          {meta.seed != null && onUseSeed ? (
            <button
              type="button"
              className="chip-button subtle"
              title="Lock and reuse this seed"
              onClick={() => onUseSeed(meta.seed!)}
            >
              <Lock size={11} />
              Seed
            </button>
          ) : null}
          <button
            type="button"
            className="chip-button subtle"
            disabled={saving}
            onClick={() => onSave(result)}
          >
            <Download size={12} />
            {saved ? 'Saved' : 'Save'}
          </button>
        </div>
      </figcaption>
    </figure>
  )
}

/** Whether a failed generation was simply stopped by the user. */
function isCancellation(cause: unknown): boolean {
  return errorText(cause).toLowerCase().includes('stopped by the user')
}

export function GenerateMode(props: Props): React.JSX.Element {
  const modality = props.modality
  const [prompt, setPrompt] = useState('')
  const [negative, setNegative] = useState('')
  const [width, setWidth] = useState(512)
  const [height, setHeight] = useState(512)
  const [steps, setSteps] = useState(20)
  const [frames, setFrames] = useState(16)
  const [seed, setSeed] = useState('')
  const [seedLocked, setSeedLocked] = useState(false)
  const [cfgScale, setCfgScale] = useState(7)
  const [guidance, setGuidance] = useState('')
  const [fps, setFps] = useState(24)
  const [busy, setBusy] = useState(false)
  const [stopping, setStopping] = useState(false)
  const [saving, setSaving] = useState<string | null>(null)
  const [saved, setSaved] = useState<Record<string, string>>({})
  const [results, setResults] = useState<GenerateBlobResult[]>([])
  const [defaultsByModel, setDefaultsByModel] = useState<Record<string, SdcppDefaults>>({})
  const [bundlesByModel, setBundlesByModel] = useState<Record<string, SdcppBundle>>({})
  const initImageInput = useRef<HTMLInputElement>(null)
  const endImageInput = useRef<HTMLInputElement>(null)
  const refImageInput = useRef<HTMLInputElement>(null)
  const refVideoInput = useRef<HTMLInputElement>(null)
  const refAudioInput = useRef<HTMLInputElement>(null)
  const [initImage, setInitImage] = useState<StoredBlob | null>(null)
  const [endImage, setEndImage] = useState<StoredBlob | null>(null)
  const [refImages, setRefImages] = useState<StoredBlob[]>([])
  const [refVideos, setRefVideos] = useState<{ blob: StoredBlob; soundtrack: StoredBlob | null }[]>([])
  const [refAudios, setRefAudios] = useState<StoredBlob[]>([])
  const [uploadingCount, setUploadingCount] = useState(0)
  const uploading = uploadingCount > 0
  const selectedRef = useRef(props.modelId)
  selectedRef.current = props.modelId
  const seedLockedRef = useRef(seedLocked)
  seedLockedRef.current = seedLocked
  const lastGeneratedHistoryIdRef = useRef<string | null>(null)

  const { urls, failed } = useBlobUrls(results.map((result) => result.blob.sha256))

  const configuredByModel = useMemo((): Record<string, DiffusionProfile> => {
    if (!props.modelProfiles) return {}
    return Object.fromEntries(
      Object.entries(props.modelProfiles).flatMap(([modelId, profile]) =>
        profile.kind === 'image' || profile.kind === 'video'
          ? [[modelId, profile as DiffusionProfile]]
          : []
      )
    )
  }, [props.modelProfiles])

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
  const useIntegratedDefaults = usesIntegratedGpuVulkanDefaults(props.settings, props.hardware)
  const selectedBundle = bundlesByModel[selected]
  const conditioning = selectedBundle?.conditioning ?? null
  const resolutionPresets =
    modality === 'video' ? VIDEO_RESOLUTION_PRESETS : IMAGE_RESOLUTION_PRESETS

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
    const modelAtStart = selectedRef.current
    setUploadingCount((count) => count + 1)
    try {
      const uploaded = await uploadAttachmentBlob(file)
      if (selectedRef.current === modelAtStart) {
        onUploaded(uploaded)
      }
    } catch (cause) {
      if (selectedRef.current === modelAtStart) {
        props.onError(errorText(cause))
      }
    } finally {
      setUploadingCount((count) => Math.max(0, count - 1))
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
      void uploadFile(file, (blob) => {
        setRefVideos((current) =>
          current.length >= 3 ? current : [...current, { blob, soundtrack: null }]
        )
      })
    }
  }

  function attachSoundtrack(entry: { blob: StoredBlob; soundtrack: StoredBlob | null }): void {
    const targetSha = entry.blob.sha256
    pickFiles(refAudioInput.current, ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/flac'], (files) => {
      const first = files[0]
      if (!first) return
      void uploadFile(first, (blob) => {
        setRefVideos((latest) =>
          latest.map((candidate) =>
            candidate.blob.sha256 === targetSha ? { ...candidate, soundtrack: blob } : candidate
          )
        )
      })
    })
  }

  useEffect(() => {
    if (!props.activeHistoryId) {
      lastGeneratedHistoryIdRef.current = null
      setResults([])
      return
    }
    if (props.activeHistoryId === lastGeneratedHistoryIdRef.current) {
      // Keep the multi-item session gallery intact after generating a new item;
      // only replace the gallery when the user explicitly selects a past entry.
      return
    }
    const entry = props.history.find((candidate) => candidate.id === props.activeHistoryId)
    if (!entry) return
    setPrompt(entry.prompt)
    setNegative(entry.negativePrompt)
    setResults([entry.result])
    // The session gallery is gone, so reselecting its newest item must load it.
    lastGeneratedHistoryIdRef.current = null
  }, [props.activeHistoryId, props.history])

  useEffect(() => {
    const curated = defaultsByModel[selected]
    const configured = configuredByModel[selected]
    if (!curated && !configured) return
    const apuWidth = modality === 'video' ? INTEGRATED_GPU_VIDEO_DEFAULTS.width : 512
    const apuHeight = modality === 'video' ? INTEGRATED_GPU_VIDEO_DEFAULTS.height : 512
    const nextWidth =
      configured?.width ??
      (useIntegratedDefaults ? Math.min(curated?.width ?? apuWidth, apuWidth) : curated?.width)
    const nextHeight =
      configured?.height ??
      (useIntegratedDefaults ? Math.min(curated?.height ?? apuHeight, apuHeight) : curated?.height)
    const nextSteps = configured?.steps ?? curated?.steps
    const cfg = configured?.cfg_scale ?? curated?.cfg_scale
    const nextGuidance = configured?.guidance ?? curated?.guidance
    const nextFrames =
      configured?.video_frames ??
      (useIntegratedDefaults && modality === 'video'
        ? Math.min(
            curated?.video_frames ?? INTEGRATED_GPU_VIDEO_DEFAULTS.frames,
            INTEGRATED_GPU_VIDEO_DEFAULTS.frames
          )
        : curated?.video_frames)
    const nextFps = configured?.fps ?? curated?.fps
    if (nextWidth) setWidth(nextWidth)
    if (nextHeight) setHeight(nextHeight)
    if (nextSteps) setSteps(nextSteps)
    if (cfg != null) setCfgScale(cfg)
    setGuidance(nextGuidance != null ? String(nextGuidance) : '')
    if (nextFrames) setFrames(nextFrames)
    if (nextFps) setFps(nextFps)
    if (configured?.seed != null && !seedLockedRef.current) {
      setSeed(String(configured.seed))
    }
  }, [selected, modality, defaultsByModel, configuredByModel, useIntegratedDefaults])

  function toggleSeedLock(): void {
    const next = !seedLocked
    setSeedLocked(next)
    if (next && !seed.trim()) {
      const lastSeed = results.length > 0 ? parseGenerationMetadata(results[0].metadata).seed : undefined
      const chosen = lastSeed ?? Math.floor(Math.random() * 2_147_483_647)
      setSeed(String(chosen))
    } else if (!next) {
      setSeed('')
    }
  }

  async function onSubmit(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (busy) return
    const modelId =
      selected ||
      (modality === 'image'
        ? props.settings?.default_image_gen_model
        : props.settings?.default_video_gen_model) ||
      ''
    if (!prompt.trim() || !modelId) return
    setBusy(true)
    props.onError(null)
    const effectiveSeed =
      seedLocked && !seed.trim()
        ? Math.floor(Math.random() * 2_147_483_647)
        : seed.trim()
          ? Number(seed)
          : undefined
    if (seedLocked && !seed.trim() && effectiveSeed != null) {
      setSeed(String(effectiveSeed))
    }
    try {
      const body = {
        prompt: prompt.trim(),
        model_id: modelId,
        negative_prompt: negative.trim() || undefined,
        width,
        height,
        steps,
        seed: effectiveSeed,
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
      const rawResult =
        modality === 'image' ? await generateImage(body) : await generateVideo(body)
      // Ensure width/height/steps/seed remain readable on the card even if the
      // engine returned minimal metadata.
      const existingMeta =
        rawResult.metadata && typeof rawResult.metadata === 'object'
          ? (rawResult.metadata as Record<string, unknown>)
          : {}
      const enriched: GenerateBlobResult = {
        ...rawResult,
        metadata: {
          width: body.width,
          height: body.height,
          steps: body.steps,
          ...(body.seed != null ? { seed: body.seed } : {}),
          ...existingMeta
        }
      }
      const historyId = crypto.randomUUID()
      lastGeneratedHistoryIdRef.current = historyId
      setResults((current) => [enriched, ...current])
      props.onGenerated({
        id: historyId,
        prompt: body.prompt,
        negativePrompt: body.negative_prompt ?? '',
        modality,
        result: enriched,
        createdAt: new Date().toISOString()
      })
    } catch (cause) {
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

  function handleShortcutSubmit(
    event: React.KeyboardEvent<HTMLTextAreaElement | HTMLInputElement>
  ): void {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
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
          disabled={busy}
          onClick={() => props.onModalityChange('image')}
        >
          <Image size={16} /> Image
        </button>
        <button
          type="button"
          role="tab"
          className={modality === 'video' ? 'active' : ''}
          aria-selected={modality === 'video'}
          disabled={busy}
          onClick={() => props.onModalityChange('video')}
        >
          <Video size={16} /> Video
        </button>
      </div>

      {available.length === 0 ? (
        <div className="mode-empty-state">
          <p className="mode-empty">
            No {modality} generation models installed yet. Install a recommended bundle or download
            one from stable-diffusion.cpp.
          </p>
          {props.onOpenManage ? (
            <div className="mode-empty-actions">
              <button
                type="button"
                className="primary-action"
                onClick={() => props.onOpenManage?.('recommended')}
              >
                Recommended models
              </button>
              <button
                type="button"
                className="secondary-action"
                onClick={() => props.onOpenManage?.('discover')}
              >
                Download models
              </button>
              <button
                type="button"
                className="secondary-action"
                onClick={() => props.onOpenManage?.('runtimes')}
              >
                Runtimes
              </button>
            </div>
          ) : null}
        </div>
      ) : (
        <form className="generate-form" onSubmit={(event) => void onSubmit(event)}>
          <label>
            Prompt
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={handleShortcutSubmit}
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
              onKeyDown={handleShortcutSubmit}
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
                          const remaining = Math.max(0, 9 - refImages.length)
                          for (const file of files.slice(0, remaining)) {
                            void uploadFile(file, (blob) =>
                              setRefImages((current) =>
                                current.length >= 9 ? current : [...current, blob]
                              )
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
                            const remaining = Math.max(0, 3 - refAudios.length)
                            for (const file of files.slice(0, remaining)) {
                              void uploadFile(file, (blob) =>
                                setRefAudios((current) =>
                                  current.length >= 3 ? current : [...current, blob]
                                )
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

          <div className="generate-presets" role="group" aria-label="Resolution presets">
            {resolutionPresets.map(([presetW, presetH, label]) => {
              const active = width === presetW && height === presetH
              return (
                <button
                  key={label}
                  type="button"
                  className={active ? 'chip-button selected' : 'chip-button subtle'}
                  aria-pressed={active}
                  onClick={() => {
                    setWidth(presetW)
                    setHeight(presetH)
                  }}
                >
                  {label}
                </button>
              )
            })}
          </div>

          <div className="generate-params">
            <label>
              Width
              <input
                type="number"
                min={64}
                max={2048}
                step={64}
                value={width}
                onChange={(event) => {
                  const v = Number(event.target.value)
                  if (Number.isFinite(v)) setWidth(v)
                }}
              />
            </label>
            <label>
              Height
              <input
                type="number"
                min={64}
                max={2048}
                step={64}
                value={height}
                onChange={(event) => {
                  const v = Number(event.target.value)
                  if (Number.isFinite(v)) setHeight(v)
                }}
              />
            </label>
            <label>
              Steps
              <input
                type="number"
                min={1}
                max={150}
                value={steps}
                onChange={(event) => {
                  const v = Number(event.target.value)
                  if (Number.isFinite(v)) setSteps(v)
                }}
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
                onChange={(event) => {
                  const v = Number(event.target.value)
                  if (Number.isFinite(v)) setCfgScale(v)
                }}
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
                    onChange={(event) => {
                      const v = Number(event.target.value)
                      if (Number.isFinite(v)) setFrames(v)
                    }}
                  />
                </label>
                <label title="Playback rate written into the clip.">
                  FPS
                  <input
                    type="number"
                    min={1}
                    max={60}
                    value={fps}
                    onChange={(event) => {
                      const v = Number(event.target.value)
                      if (Number.isFinite(v)) setFps(v)
                    }}
                  />
                </label>
              </>
            ) : null}
            <label className="generate-seed-field">
              Seed
              <div className="generate-seed-row">
                <input
                  value={seed}
                  onChange={(event) => {
                    setSeed(event.target.value)
                    if (!event.target.value.trim()) setSeedLocked(false)
                  }}
                  placeholder="Random"
                />
                <button
                  type="button"
                  className={seedLocked ? 'chip-button selected' : 'chip-button subtle'}
                  title={seedLocked ? 'Unlock seed (return to random)' : 'Lock seed across runs'}
                  aria-label={seedLocked ? 'Unlock seed' : 'Lock seed'}
                  aria-pressed={seedLocked}
                  onClick={toggleSeedLock}
                >
                  {seedLocked ? <Lock size={12} /> : <LockOpen size={12} />}
                </button>
                <button
                  type="button"
                  className="chip-button subtle"
                  title="Pick a new random seed"
                  aria-label="New random seed"
                  onClick={() => {
                    const nextSeed = Math.floor(Math.random() * 2_147_483_647)
                    setSeed(String(nextSeed))
                    setSeedLocked(true)
                  }}
                >
                  <Shuffle size={12} />
                </button>
              </div>
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
            <span className="generate-submit-hint">⌘/Ctrl+Enter</span>
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
              failed={Boolean(failed[result.blob.sha256])}
              saving={saving === result.blob.sha256}
              saved={Boolean(saved[result.blob.sha256])}
              onSave={(target) => void save(target)}
              onUseSeed={(chosenSeed) => {
                setSeed(String(chosenSeed))
                setSeedLocked(true)
              }}
            />
          ))}
        </div>
      ) : null}
    </section>
  )
}

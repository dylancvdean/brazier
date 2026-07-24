import { Image, LoaderCircle, Video } from 'lucide-react'
import { type FormEvent, useEffect, useState } from 'react'
import {
  fetchBlobObjectUrl,
  generateImage,
  generateVideo,
  type GenerateBlobResult,
  type LocalModel,
  type RuntimeSettings
} from '../api'

type Modality = 'image' | 'video'

type Props = {
  /** Installed models for the active modality. */
  models: LocalModel[]
  modality: Modality
  onModalityChange: (modality: Modality) => void
  /** Model chosen in the top bar; empty when none is installed. */
  modelId: string
  settings: RuntimeSettings | null
  onError: (message: string | null) => void
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
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
  const [busy, setBusy] = useState(false)
  const [results, setResults] = useState<GenerateBlobResult[]>([])
  const [urls, setUrls] = useState<Record<string, string>>({})

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

  const available = props.models
  const selected = props.modelId

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
        video_frames: frames
      }
      const result =
        modality === 'image' ? await generateImage(body) : await generateVideo(body)
      setResults((current) => [result, ...current])
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setBusy(false)
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
            {modality === 'video' ? (
              <label>
                Frames
                <input
                  type="number"
                  min={1}
                  max={120}
                  value={frames}
                  onChange={(event) => setFrames(Number(event.target.value))}
                />
              </label>
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
          <button type="submit" className="primary" disabled={busy || !prompt.trim()}>
            {busy ? (
              <>
                <LoaderCircle className="spin" size={16} /> Generating…
              </>
            ) : (
              `Generate ${modality}`
            )}
          </button>
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
                  {result.blob.mime_type} · {(result.blob.size_bytes / 1024).toFixed(0)} KB
                </figcaption>
              </figure>
            )
          })}
        </div>
      ) : null}
    </section>
  )
}

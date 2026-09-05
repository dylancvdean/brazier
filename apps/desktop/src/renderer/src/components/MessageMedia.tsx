import { Download, RefreshCw } from 'lucide-react'
import { useState } from 'react'

import { saveBlobToDisk } from '../api'
import { useBlobUrls } from '../useBlobUrls'
import { MediaFullscreenExit, MediaFullscreenIcon, useFullscreen } from './FullscreenButton'

export type MessageBlob = {
  sha256: string
  mime_type: string
  original_name?: string | null
}

function MessageMediaItem({
  blob,
  url,
  failed,
  saving,
  saved,
  onSave,
  onRetry
}: {
  blob: MessageBlob
  url?: string
  failed: boolean
  saving: boolean
  saved: boolean
  onSave: (blob: MessageBlob) => void
  onRetry: (blob: MessageBlob) => void
}): React.JSX.Element {
  const { setRef, active, toggle } = useFullscreen<HTMLElement>()
  const isVideo = blob.mime_type.startsWith('video/')
  const isImage = blob.mime_type.startsWith('image/')
  const isAudio = blob.mime_type.startsWith('audio/')
  return (
    <figure className="message-media-item">
      {!isImage && !isVideo && !isAudio ? (
        <div className="message-media-placeholder">Save this file to open it.</div>
      ) : url && isImage ? (
        <div
          className={`message-media-preview${active ? ' media-fullscreen' : ''}`}
          ref={setRef}
        >
          <img src={url} alt="Attached or generated image" />
          {!active && <MediaFullscreenIcon active={active} toggle={toggle} />}
          {active && <MediaFullscreenExit toggle={toggle} />}
        </div>
      ) : url && isVideo ? (
        <video ref={setRef} src={url} controls playsInline />
      ) : url && isAudio ? (
        <audio src={url} controls />
      ) : failed ? (
        <div className="message-media-placeholder failed">
          <span>Failed to load</span>
          <button
            type="button"
            className="chip-button subtle"
            onClick={() => onRetry(blob)}
          >
            <RefreshCw size={12} />
            Retry
          </button>
        </div>
      ) : (
        <div className="message-media-placeholder">Loading…</div>
      )}
      <figcaption>
        <span className="message-media-name">{blob.original_name ?? blob.mime_type}</span>
        <button
          type="button"
          className="chip-button subtle"
          disabled={saving}
          onClick={() => onSave(blob)}
        >
          <Download size={12} />
          {saving ? 'Saving…' : saved ? 'Saved' : 'Save'}
        </button>
      </figcaption>
    </figure>
  )
}

/**
 * Media attached to a chat message, shown rather than merely named.
 *
 * A picture or clip a model generated used to appear in the transcript as the
 * word "image", leaving the result itself reachable only through the blob
 * store. Rendering it here is also what gives it somewhere to hang a save
 * action.
 */
export function MessageMedia({
  blobs,
  onError
}: {
  blobs: MessageBlob[]
  onError?: (message: string) => void
}): React.JSX.Element | null {
  const [saving, setSaving] = useState<Record<string, boolean>>({})
  const [saved, setSaved] = useState<Record<string, boolean>>({})
  const [retries, setRetries] = useState(0)

  const { urls, failed } = useBlobUrls(
    blobs.filter((blob) => /^(image|video|audio)\//.test(blob.mime_type)).map((blob) => blob.sha256),
    retries
  )

  if (blobs.length === 0) return null

  async function save(blob: MessageBlob): Promise<void> {
    setSaving((current) => ({ ...current, [blob.sha256]: true }))
    try {
      const path = await saveBlobToDisk(blob.sha256, blob.mime_type, blob.original_name)
      if (path) setSaved((current) => ({ ...current, [blob.sha256]: true }))
    } catch (cause) {
      onError?.(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSaving((current) => ({ ...current, [blob.sha256]: false }))
    }
  }

  function retry(): void {
    setRetries((value) => value + 1)
  }

  return (
    <div className="message-media">
      {blobs.map((blob, index) => (
        <MessageMediaItem
          key={`${blob.sha256}:${blob.mime_type}:${index}`}
          blob={blob}
          url={urls[blob.sha256]}
          failed={Boolean(failed[blob.sha256])}
          saving={Boolean(saving[blob.sha256])}
          saved={Boolean(saved[blob.sha256])}
          onSave={(target) => void save(target)}
          onRetry={retry}
        />
      ))}
    </div>
  )
}

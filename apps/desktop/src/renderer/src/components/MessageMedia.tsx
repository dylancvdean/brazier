import { Download } from 'lucide-react'
import { useEffect, useState } from 'react'

import { fetchBlobObjectUrl, saveBlobToDisk } from '../api'

export type MessageBlob = {
  sha256: string
  mime_type: string
  original_name?: string | null
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
  const [urls, setUrls] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState<string | null>(null)
  const [saved, setSaved] = useState<Record<string, boolean>>({})

  const keys = blobs.map((blob) => blob.sha256).join(',')
  useEffect(() => {
    let cancelled = false
    void (async () => {
      for (const blob of blobs) {
        if (urls[blob.sha256]) continue
        try {
          const url = await fetchBlobObjectUrl(blob.sha256)
          if (cancelled) return
          setUrls((current) => ({ ...current, [blob.sha256]: url }))
        } catch {
          // A blob that no longer loads still lists its save button, which
          // reports the same failure with a message attached.
        }
      }
    })()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keys])

  if (blobs.length === 0) return null

  async function save(blob: MessageBlob): Promise<void> {
    setSaving(blob.sha256)
    try {
      const path = await saveBlobToDisk(blob.sha256, blob.mime_type, blob.original_name)
      if (path) setSaved((current) => ({ ...current, [blob.sha256]: true }))
    } catch (cause) {
      onError?.(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSaving(null)
    }
  }

  return (
    <div className="message-media">
      {blobs.map((blob, index) => {
        const url = urls[blob.sha256]
        const isVideo = blob.mime_type.startsWith('video/')
        const isImage = blob.mime_type.startsWith('image/')
        const isAudio = blob.mime_type.startsWith('audio/')
        return (
          <figure className="message-media-item" key={`${blob.sha256}:${blob.mime_type}:${index}`}>
            {url && isImage && <img src={url} alt="Attached or generated image" />}
            {url && isVideo && <video src={url} controls playsInline />}
            {url && isAudio && <audio src={url} controls />}
            {!url && <div className="message-media-placeholder">Loading…</div>}
            <figcaption>
              <span>{blob.original_name ?? blob.mime_type}</span>
              <button
                type="button"
                className="chip-button subtle"
                disabled={saving === blob.sha256}
                onClick={() => void save(blob)}
              >
                <Download size={12} />
                {saved[blob.sha256] ? 'Saved' : 'Save'}
              </button>
            </figcaption>
          </figure>
        )
      })}
    </div>
  )
}

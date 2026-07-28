import { Image, Search, Video } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { fetchBlobObjectUrl, type GenerateBlobResult } from '../api'

export type GenerateHistoryEntry = {
  id: string
  prompt: string
  negativePrompt: string
  modality: 'image' | 'video'
  result: GenerateBlobResult
  createdAt: string
}

type Props = {
  entries: GenerateHistoryEntry[]
  activeId: string | null
  onSelect: (id: string | null) => void
}

export function GenerateHistorySidebar({ entries, activeId, onSelect }: Props): React.JSX.Element {
  const [search, setSearch] = useState('')
  const [urls, setUrls] = useState<Record<string, string>>({})
  const query = search.trim().toLowerCase()
  const visible = useMemo(
    () =>
      query
        ? entries.filter((entry) => `${entry.prompt} ${entry.negativePrompt}`.toLowerCase().includes(query))
        : entries,
    [entries, query]
  )

  useEffect(() => {
    const missing = visible.filter((entry) => !urls[entry.result.blob.sha256])
    if (missing.length === 0) return
    let cancelled = false
    void Promise.all(
      missing.map(
        async (entry) =>
          [entry.result.blob.sha256, await fetchBlobObjectUrl(entry.result.blob.sha256)] as const
      )
    )
      .then((loaded) => {
        if (!cancelled) setUrls((current) => ({ ...current, ...Object.fromEntries(loaded) }))
      })
      .catch(() => {
        // A missing historical blob should not prevent the rest of the list.
      })
    return () => {
      cancelled = true
    }
  }, [visible, urls])

  return (
    <>
      <button className="new-chat" type="button" onClick={() => onSelect(null)}>
        <Image size={17} />
        New generation
      </button>
      <label className="conversation-search">
        <Search size={14} />
        <input
          aria-label="Search generations"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Search generations…"
        />
      </label>
      <div className="conversation-list generate-history-list">
        <div className="section-label">History</div>
        {visible.map((entry) => {
          const url = urls[entry.result.blob.sha256]
          const label = [entry.prompt, entry.negativePrompt].filter(Boolean).join(' · ')
          return (
            <button
              className={entry.id === activeId ? 'generate-history-entry active' : 'generate-history-entry'}
              key={entry.id}
              type="button"
              onClick={() => onSelect(entry.id)}
            >
              <span className="generate-history-thumbnail">
                {url ? (
                  entry.modality === 'video' ? (
                    <video src={url} muted preload="metadata" />
                  ) : (
                    <img src={url} alt="" />
                  )
                ) : entry.modality === 'video' ? (
                  <Video size={16} />
                ) : (
                  <Image size={16} />
                )}
              </span>
              <span className="generate-history-copy">
                <strong title={label}>{label}</strong>
                <small>{entry.createdAt.slice(0, 10)}</small>
              </span>
            </button>
          )
        })}
        {visible.length === 0 ? (
          <p className="empty-sidebar">
            {entries.length === 0 ? 'Completed images and videos will appear here.' : 'No generations match.'}
          </p>
        ) : null}
      </div>
    </>
  )
}

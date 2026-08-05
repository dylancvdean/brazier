import { Monitor, Search, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'

import type { ComputerSidebarControls } from './ComputerMode'

type Props = {
  controls: ComputerSidebarControls | null
}

/**
 * Sidebar body for Computer mode: new-session control plus the session list,
 * so sessions live where chats normally do instead of in their own section.
 */
export function ComputerSessionSidebar({ controls }: Props): React.JSX.Element {
  const [search, setSearch] = useState('')
  const query = search.trim().toLowerCase()

  const sessions = useMemo(() => {
    if (!controls) return []
    return query
      ? controls.sessions.filter((entry) =>
          `${entry.title} ${entry.target}`.toLowerCase().includes(query)
        )
      : controls.sessions
  }, [controls, query])

  if (!controls) {
    return (
      <>
        <button className="new-chat" type="button" disabled>
          <Monitor size={17} />
          New session
        </button>
        <p className="empty-sidebar">Loading computer sessions…</p>
      </>
    )
  }

  return (
    <>
      <button className="new-chat" type="button" onClick={() => controls.newSession()}>
        <Monitor size={17} />
        New session
      </button>
      <label className="conversation-search">
        <Search size={14} />
        <input
          aria-label="Search computer sessions"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Search sessions…"
        />
      </label>
      <div className="conversation-list computer-sidebar-sessions">
        {sessions.length === 0 && (
          <p className="empty-sidebar">
            {controls.sessions.length === 0
              ? 'No computer sessions yet. Describe a task in the composer.'
              : 'No sessions match that search.'}
          </p>
        )}
        {sessions.map((entry) => (
          <div
            className={entry.id === controls.activeId ? 'computer-session active' : 'computer-session'}
            key={entry.id}
          >
            <button type="button" onClick={() => controls.select(entry.id)}>
              <strong>{entry.title || 'Session'}</strong>
              <span>
                {entry.target === 'desktop' ? 'Desktop' : 'Browser'}
                {entry.running ? ' · Running' : ''} · {entry.updated_at.slice(0, 10)}
              </span>
            </button>
            <button
              type="button"
              className="computer-session-delete"
              title="Delete this session"
              onClick={() => controls.remove(entry.id)}
            >
              <Trash2 size={13} />
            </button>
          </div>
        ))}
      </div>
    </>
  )
}

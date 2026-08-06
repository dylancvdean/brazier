import { Bot, FolderOpen, Search, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'

import {
  groupAgentSessionsByDirectory,
  type AgentSidebarControls
} from './AgentMode'

const RUNTIME_LABELS: Record<string, string> = {
  powerful: 'Powerful',
  balanced: 'Balanced'
}

function runtimeLabel(runtimeId: string | null | undefined): string {
  if (!runtimeId) return 'Simple'
  return RUNTIME_LABELS[runtimeId] ?? 'Simple'
}

type Props = {
  controls: AgentSidebarControls | null
}

const RUNTIME_LABELS: Record<string, string> = {
  powerful: 'Powerful',
  balanced: 'Balanced'
}

function runtimeLabel(id: string | undefined): string {
  return RUNTIME_LABELS[id ?? ''] ?? 'Simple'
}

/**
 * Sidebar body for Agent mode: new-task control plus sessions grouped by the
 * project directory they belong to (worktrees nest under their source repo).
 */
export function AgentSessionSidebar({ controls }: Props): React.JSX.Element {
  const [search, setSearch] = useState('')
  const query = search.trim().toLowerCase()

  const groups = useMemo(() => {
    if (!controls) return []
    const filtered = query
      ? controls.sessions.filter((session) => {
          const haystack = [
            session.title,
            session.workspace_path ?? '',
            session.runtime_id ?? '',
            session.runtime_metadata?.worktree?.source_path ?? '',
            session.runtime_metadata?.worktree?.branch ?? '',
            session.last_run_status ?? ''
          ]
            .join(' ')
            .toLowerCase()
          return haystack.includes(query)
        })
      : controls.sessions
    return groupAgentSessionsByDirectory(filtered)
  }, [controls, query])

  if (!controls) {
    return (
      <>
        <button className="new-chat" type="button" disabled>
          <Bot size={17} />
          New task
        </button>
        <p className="empty-sidebar">Loading agent tasks…</p>
      </>
    )
  }

  return (
    <>
      <button className="new-chat" type="button" onClick={() => controls.newTask()}>
        <Bot size={17} />
        New task
      </button>
      <label className="conversation-search">
        <Search size={14} />
        <input
          aria-label="Search agent tasks"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Search tasks…"
        />
      </label>
      <div className="conversation-list agent-sidebar-list">
        {groups.length === 0 && (
          <p className="empty-sidebar">
            {controls.sessions.length === 0
              ? 'No agent tasks yet. Start one from the composer.'
              : 'No tasks match that search.'}
          </p>
        )}
        {groups.map((group) => (
          <div className="agent-sidebar-group" key={group.path || 'none'}>
            <div className="agent-sidebar-group-label" title={group.path || undefined}>
              <FolderOpen size={12} />
              <span>{group.label}</span>
            </div>
            {group.sessions.map((entry) => {
              const worktree = entry.runtime_metadata?.worktree
              return (
                <div
                  className={
                    entry.id === controls.activeId ? 'agent-session active' : 'agent-session'
                  }
                  key={entry.id}
                >
                  <button type="button" onClick={() => controls.select(entry.id)}>
                    <strong>{entry.title}</strong>
                    <span>
                      {runtimeLabel(entry.runtime_id)} ·{' '}
                      {worktree
                        ? `Worktree · ${worktree.branch} · ${entry.last_run_status ?? '—'}`
                        : entry.last_run_status ?? '—'}
                    </span>
                  </button>
                  <button
                    type="button"
                    className="agent-session-delete"
                    title={
                      worktree
                        ? 'Delete this task. You will be asked about its worktree first.'
                        : 'Delete this task'
                    }
                    onClick={() => controls.remove(entry.id)}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </>
  )
}

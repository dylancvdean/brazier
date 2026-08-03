import { Activity, Cpu, FastForward, ListTodo, LoaderCircle, Zap } from 'lucide-react'

import type { OmpTodoPhase } from '../../../agent/omp/rpcTypes'
import type { OmpRecentFrame, OmpSessionInfo, OmpSubagentView } from '../ompSidecar'

/**
 * Live Oh My Pi session state, surfaced from the sidecar's `get_state` stream:
 * context usage, active model and thinking level, fast mode / auto-compaction
 * toggles, the agent's todo list, and a bounded record of recent sidecar events.
 *
 * The context usage bar is the at-a-glance parity win; the rest is the same
 * state OMP's own TUI shows, mirrored as native controls.
 */

export function contextPercent(info: OmpSessionInfo | null | undefined): number | null {
  const usage = info?.contextUsage
  if (usage?.percent != null && Number.isFinite(usage.percent)) return usage.percent
  if (usage?.tokens != null && usage.contextWindow != null && usage.contextWindow > 0) {
    return (usage.tokens / usage.contextWindow) * 100
  }
  return null
}

function todoCount(phases: OmpTodoPhase[] | undefined): number {
  return phases?.reduce((total, phase) => total + phase.tasks.length, 0) ?? 0
}

function TodoList({ phases }: { phases: OmpTodoPhase[] | undefined }): React.JSX.Element | null {
  if (!phases || phases.length === 0) return null
  return (
    <div className="omp-session-todos">
      <strong>Todos</strong>
      {phases.map((phase) => (
        <div className="omp-todo-phase" key={phase.id}>
          {phases.length > 1 && <span className="omp-todo-phase-name">{phase.name}</span>}
          <ul>
            {phase.tasks.map((task) => (
              <li className={`omp-todo-item ${task.status}`} key={task.id}>
                <span className="omp-todo-dot" />
                <span>{task.content}</span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  )
}

function SubagentList({
  subagents
}: {
  subagents: OmpSubagentView[]
}): React.JSX.Element | null {
  if (subagents.length === 0) return null
  const ordered = [...subagents].sort((a, b) => a.index - b.index)
  return (
    <div className="omp-session-subagents">
      <strong>Subagents</strong>
      {ordered.map((subagent) => {
        const running = subagent.status === 'running' || subagent.status === 'pending'
        const output = subagent.recentOutput.at(-1)
        return (
          <div className={`omp-subagent ${subagent.status}`} key={subagent.id}>
            <div className="omp-subagent-head">
              <span className="omp-subagent-status">
                {running ? <LoaderCircle className="spin" size={12} /> : <span className="omp-subagent-dot" />}
              </span>
              <strong>{subagent.agent}</strong>
              {subagent.resolvedModel && <span className="omp-subagent-model">{subagent.resolvedModel}</span>}
            </div>
            <div className="omp-subagent-task">
              {subagent.task
                ? subagent.task.length > 120
                  ? `${subagent.task.slice(0, 120)}…`
                  : subagent.task
                : subagent.assignment
                  ? subagent.assignment
                  : subagent.description ?? 'Task subagent'}
            </div>
            {running && subagent.currentTool && (
              <span className="omp-subagent-tool">
                {subagent.currentTool}
                {subagent.currentToolArgs ? ` · ${subagent.currentToolArgs.slice(0, 60)}` : ''}
              </span>
            )}
            {output && <span className="omp-subagent-output">{output.slice(0, 120)}</span>}
            {(subagent.toolCount != null || subagent.tokens != null || subagent.cost != null) && (
              <span className="omp-subagent-meta">
                {subagent.toolCount != null && `${subagent.toolCount} tool${subagent.toolCount === 1 ? '' : 's'}`}
                {subagent.tokens != null && `${subagent.tokens >= 1000 ? `${(subagent.tokens / 1000).toFixed(1)}k` : subagent.tokens} tok`}
                {subagent.cost != null && subagent.cost > 0 && `· $${subagent.cost.toFixed(4)}`}
              </span>
            )}
          </div>
        )
      })}
    </div>
  )
}

type OmpSessionPanelProps = {
  info: OmpSessionInfo | null
  subagents: OmpSubagentView[]
  recentFrames: OmpRecentFrame[]
  busy: boolean
  onSetFastMode: (enabled: boolean) => void
  onSetAutoCompaction: (enabled: boolean) => void
  onCycleThinking: () => void
}

export function OmpSessionPanel({
  info,
  subagents,
  recentFrames,
  busy,
  onSetFastMode,
  onSetAutoCompaction,
  onCycleThinking
}: OmpSessionPanelProps): React.JSX.Element {
  const percent = contextPercent(info)
  const todos = todoCount(info?.todoPhases)
  const runningSubagents = subagents.filter(
    (subagent) => subagent.status === 'running' || subagent.status === 'pending'
  ).length
  return (
    <details className="omp-session" title="Live Oh My Pi session state">
      <summary>
        {percent != null ? (
          <>
            <Activity size={13} />
            <span className="omp-session-summary-context">
              <span className="omp-session-summary-bar">
                <span style={{ width: `${Math.min(100, Math.max(2, percent))}%` }} />
              </span>
              <span>{Math.round(percent)}%</span>
            </span>
          </>
        ) : (
          <>
            <Cpu size={13} />
            <span>OMP session</span>
          </>
        )}
        {runningSubagents > 0 && (
          <span className="omp-session-summary-todos" title={`${runningSubagents} subagent${runningSubagents === 1 ? '' : 's'} active`}>
            <LoaderCircle className="spin" size={12} />
            {runningSubagents}
          </span>
        )}
        {todos > 0 && (
          <span className="omp-session-summary-todos" title={`${todos} todo${todos === 1 ? '' : 's'}`}>
            <ListTodo size={12} />
            {todos}
          </span>
        )}
        {recentFrames.length > 0 && (
          <span className="omp-session-summary-events" title={`${recentFrames.length} recent sidecar events`}>
            {recentFrames.length}
          </span>
        )}
      </summary>
      <div className="omp-session-panel">
        <section className="omp-session-section">
          <strong>Context</strong>
          {info?.contextUsage?.contextWindow ? (
            <>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{ width: `${Math.min(100, Math.max(0, percent ?? 0))}%` }}
                />
              </div>
              <span className="omp-session-context-facts">
                {info.contextUsage.tokens?.toLocaleString() ?? 0} /{' '}
                {info.contextUsage.contextWindow.toLocaleString()} tokens
                {info.tokensPerSecond != null
                  ? ` · ${info.tokensPerSecond} tok/s`
                  : ''}
                {info.isStreaming ? ' · streaming' : ''}
                {info.isCompacting ? ' · compacting' : ''}
              </span>
            </>
          ) : (
            <span className="omp-session-muted">No context measurement yet.</span>
          )}
        </section>

        <section className="omp-session-section">
          <strong>Model &amp; reasoning</strong>
          <dl className="omp-session-facts">
            <div>
              <dt>Model</dt>
              <dd>{info?.modelName ?? '—'}</dd>
            </div>
            <div>
              <dt>Thinking</dt>
              <dd>
                <span className="omp-thinking-level">{info?.thinkingLevel ?? '—'}</span>
                <button
                  type="button"
                  className="chip-button subtle"
                  disabled={busy}
                  onClick={onCycleThinking}
                  title="Cycle the reasoning effort level"
                >
                  Cycle
                </button>
              </dd>
            </div>
          </dl>
        </section>

        <section className="omp-session-section">
          <strong>Behavior</strong>
          <div className="omp-session-toggles">
            <label>
              <input
                type="checkbox"
                disabled={busy}
                checked={info?.fastModeEnabled ?? false}
                onChange={(event) => onSetFastMode(event.target.checked)}
              />
              <FastForward size={13} />
              <span>
                Fast mode
                {info?.fastModeActive && !info.fastModeEnabled ? <small> (active via provider)</small> : null}
              </span>
            </label>
            <label>
              <input
                type="checkbox"
                disabled={busy}
                checked={info?.autoCompactionEnabled ?? false}
                onChange={(event) => onSetAutoCompaction(event.target.checked)}
              />
              <Zap size={13} />
              <span>Auto-compaction</span>
            </label>
          </div>
        </section>

        <TodoList phases={info?.todoPhases} />

        <SubagentList subagents={subagents} />

        {recentFrames.length > 0 && (
          <section className="omp-session-section omp-session-events">
            <strong>Events</strong>
            <div className="omp-session-events-list">
              {[...recentFrames].reverse().map((frame) => (
                <div key={frame.id} title={frame.type}>
                  <span className="omp-events-type">{frame.type}</span>
                  <span className="omp-events-detail">{frame.detail}</span>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </details>
  )
}

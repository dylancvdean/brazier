/**
 * End-of-run report: what the agent actually did, assembled from the event
 * stream rather than from the model's own account of itself.
 */

import type { AgentEvent, AgentRunSummary } from './types'

export function emptySummary(): AgentRunSummary {
  return {
    filesChanged: [],
    commandsRun: [],
    toolCalls: 0,
    failures: [],
    hostActions: [],
    approvalsRequested: 0,
    text: ''
  }
}

/** Fold one event into a summary. Call for every event of the run. */
export function accumulate(summary: AgentRunSummary, event: AgentEvent): AgentRunSummary {
  switch (event.type) {
    case 'tool-call-proposed': {
      return { ...summary, toolCalls: summary.toolCalls + 1 }
    }
    case 'approval-required': {
      return { ...summary, approvalsRequested: summary.approvalsRequested + 1 }
    }
    case 'tool-started': {
      const command =
        event.tool === 'shell_run' || event.tool === 'shell_start'
          ? String(event.args.command ?? '')
          : undefined
      const commandsRun =
        command && !summary.commandsRun.includes(command)
          ? [...summary.commandsRun, command]
          : summary.commandsRun
      return { ...summary, commandsRun }
    }
    case 'tool-completed': {
      const filesChanged = [...summary.filesChanged]
      for (const path of event.changedPaths) {
        if (!filesChanged.includes(path)) filesChanged.push(path)
      }
      const hostActions =
        event.environment === 'host'
          ? [...summary.hostActions, `${event.tool} (host)`]
          : summary.hostActions
      return { ...summary, filesChanged, hostActions }
    }
    case 'tool-failed': {
      const label = event.denied ? `${event.tool}: refused` : `${event.tool}: ${firstLine(event.error)}`
      return { ...summary, failures: [...summary.failures, label] }
    }
    case 'run-failed': {
      return { ...summary, failures: [...summary.failures, firstLine(event.error)] }
    }
    default:
      return summary
  }
}

function firstLine(text: string): string {
  const line = text.split('\n').find((candidate) => candidate.trim().length > 0) ?? text
  return line.trim().slice(0, 200)
}

/** Human-readable closing note. Empty when the run did nothing worth listing. */
export function describeSummary(summary: AgentRunSummary): string {
  const parts: string[] = []
  if (summary.filesChanged.length > 0) {
    parts.push(
      `Changed ${summary.filesChanged.length} file${summary.filesChanged.length === 1 ? '' : 's'}: ${summary.filesChanged
        .slice(0, 8)
        .join(', ')}${summary.filesChanged.length > 8 ? ', …' : ''}`
    )
  }
  if (summary.commandsRun.length > 0) {
    parts.push(
      `Ran ${summary.commandsRun.length} command${summary.commandsRun.length === 1 ? '' : 's'}`
    )
  }
  if (summary.hostActions.length > 0) {
    parts.push(`${summary.hostActions.length} action(s) ran outside the sandbox`)
  }
  if (summary.failures.length > 0) {
    parts.push(`${summary.failures.length} failure(s): ${summary.failures.slice(0, 3).join('; ')}`)
  }
  return parts.join(' · ')
}

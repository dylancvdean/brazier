/**
 * Agent adapter: the existing agent session, normalized.
 *
 * The agent subsystem is treated as an external component. This translates
 * between the coordinator's correlation ids and the worker's session/run events,
 * and nothing else — it never reinterprets a tool result, only transports it.
 *
 * The agent session is bound to the conversation in the daemon, so text and
 * voice turns reach the same session and the binding survives a restart.
 */

import { decideAgentApproval, fetchAgentSession, updateAgentSession } from '../agentApi'
import { getConversation, updateConversation } from '../api'
import type { AgentEvent } from '../../../agent/core/types'
import type { WorkerMessage } from '../../../agent/core/protocol'
import type { AgentAdapter, AgentAdapterEvent, AgentRunStatusReport, AgentTurnRequest } from './adapters'

/** How a tool result is reduced to something short enough to say out loud. */
function toolOutcome(event: Extract<AgentEvent, { type: 'tool-completed' }>): string {
  const parts: string[] = []
  if (typeof event.exitCode === 'number') parts.push(`exit ${event.exitCode}`)
  if (event.changedPaths.length > 0) parts.push(`changed ${event.changedPaths.length} file(s)`)
  const firstLine = event.output.split('\n').find((line) => line.trim().length > 0)
  if (firstLine) parts.push(firstLine.trim().slice(0, 120))
  return `${event.tool}: ${parts.join(', ') || 'completed'}`
}

export class WorkerAgentAdapter implements AgentAdapter {
  private sessionId: string | null = null
  private conversationId: string | null = null
  /**
   * The correlation id of the run in flight. The worker reports events per
   * session and run, not per turn, and a session runs one turn at a time, so
   * this mapping is exact rather than a guess.
   */
  private activeCorrelationId: string | null = null
  private activeRunId: string | null = null
  private lastRunId: string | null = null
  private readonly statuses = new Map<string, AgentRunStatusReport>()
  private readonly listeners = new Set<(event: AgentAdapterEvent) => void>()
  /** Attached while anyone is listening; see `subscribe`. */
  private bridge: (() => void) | undefined
  /** Text accumulated for the current run, to synthesize a final if needed. */
  private streamed = ''

  constructor(private readonly model?: () => string | undefined) {}

  dispose(): void {
    this.bridge?.()
    this.bridge = undefined
    this.listeners.clear()
  }

  /** Adopt whichever agent session the conversation already records. */
  async attachSession(conversationId: string): Promise<string | null> {
    this.conversationId = conversationId
    try {
      const conversation = await getConversation(conversationId)
      const sessionId = conversation.agent_session_id ?? null
      if (!sessionId) {
        this.sessionId = null
        return null
      }
      // Confirm the session still exists before claiming it is bound.
      await fetchAgentSession(sessionId)
      this.sessionId = sessionId
      await window.brazier.agent.openSession(sessionId)
      return sessionId
    } catch {
      this.sessionId = null
      return null
    }
  }

  attachedSessionId(): string | null {
    return this.sessionId
  }

  /** Bind a session to this conversation, so both surfaces share it. */
  async bindSession(sessionId: string | null): Promise<void> {
    this.sessionId = sessionId
    if (this.conversationId) {
      await updateConversation(this.conversationId, { agent_session_id: sessionId })
    }
    if (sessionId) await window.brazier.agent.openSession(sessionId)
  }

  async submitTurn(request: AgentTurnRequest): Promise<void> {
    const sessionId = this.sessionId
    if (!sessionId) throw new Error('No agent session is bound to this conversation.')
    const model = this.model?.()
    if (model) {
      // Model changes only take effect between runs, which is where we are.
      await window.brazier.agent.setModel(sessionId, { id: model }).catch(() => undefined)
      await updateAgentSession(sessionId, { model }).catch(() => undefined)
    }
    this.activeCorrelationId = request.correlationId
    this.activeRunId = null
    this.streamed = ''
    this.statuses.set(request.correlationId, {
      correlationId: request.correlationId,
      status: 'running'
    })
    // The worker resolves this when the run ends; the coordinator does not wait
    // on it, because the response arrives as events.
    void window.brazier.agent
      .run(sessionId, { text: request.text })
      .catch((cause: unknown) =>
        this.publish({
          type: 'runFailed',
          correlationId: request.correlationId,
          error: cause instanceof Error ? cause.message : String(cause)
        })
      )
  }

  async decideApproval(
    approvalId: string,
    decision: 'approve' | 'deny',
    note?: string
  ): Promise<void> {
    // One-shot on purpose: a decision made by voice covers the call that was
    // read out, never the rest of the session. Session scope is a deliberate
    // choice made where the arguments are on screen.
    await decideAgentApproval(approvalId, decision, 'once', note)
    this.publish({
      type: 'approvalResolved',
      correlationId: this.activeCorrelationId ?? '',
      approvalId
    })
  }

  async cancelRun(correlationId: string): Promise<void> {
    if (!this.sessionId) return
    if (this.activeCorrelationId && this.activeCorrelationId !== correlationId) return
    if (this.activeRunId !== null) this.lastRunId = this.activeRunId
    await window.brazier.agent.cancel(this.sessionId)
  }

  getStatus(correlationId: string): AgentRunStatusReport | null {
    return this.statuses.get(correlationId) ?? null
  }

  /**
   * The worker bridge is attached while there is someone to hand events to, so
   * subscribing and unsubscribing stay symmetric. Attaching in the constructor
   * and detaching on dispose did not: one React remount left the adapter
   * permanently deaf to the worker.
   */
  subscribe(listener: (event: AgentAdapterEvent) => void): () => void {
    this.listeners.add(listener)
    this.bridge ??= window.brazier.agent.onMessage((message: WorkerMessage) => {
      if (message.type !== 'event') return
      if (message.sessionId !== this.sessionId) return
      this.onWorkerEvent(message.event)
    })
    return () => {
      this.listeners.delete(listener)
      if (this.listeners.size === 0) {
        this.bridge?.()
        this.bridge = undefined
      }
    }
  }

  private publish(event: AgentAdapterEvent): void {
    for (const listener of [...this.listeners]) listener(event)
  }

  private setStatus(correlationId: string, status: AgentRunStatusReport['status']): void {
    this.statuses.set(correlationId, { correlationId, status })
  }

  private onWorkerEvent(event: AgentEvent): void {
    if (this.lastRunId !== null && event.runId === this.lastRunId) return
    if (this.activeRunId === null) this.activeRunId = event.runId
    const correlationId = this.activeCorrelationId
    if (!correlationId) return
    switch (event.type) {
      case 'run-started':
        this.setStatus(correlationId, 'running')
        this.publish({ type: 'runStarted', correlationId })
        return
      case 'prefill-progress':
        this.publish({
          type: 'statusUpdated',
          correlationId,
          status: `Prefilling ${Math.min(event.processed, event.total).toLocaleString()} / ${event.total.toLocaleString()} tokens${
            event.contextTotal
              ? ` · context ${event.total.toLocaleString()} / ${event.contextTotal.toLocaleString()}`
              : ''
          }`
        })
        return
      case 'text-delta':
        // Reasoning is not an answer and is never spoken.
        if (event.channel === 'reasoning') return
        this.streamed += event.delta
        this.publish({ type: 'responsePartial', correlationId, delta: event.delta })
        return
      case 'tool-call-proposed':
        this.publish({
          type: 'statusUpdated',
          correlationId,
          status: `Preparing ${event.tool}…`,
          activeTool: event.tool
        })
        return
      case 'approval-required':
        this.setStatus(correlationId, 'awaiting-approval')
        this.publish({
          type: 'statusUpdated',
          correlationId,
          status: `Waiting for your approval: ${event.approval.summary}`
        })
        this.publish({
          type: 'approvalRequired',
          correlationId,
          approvalId: event.approval.id,
          tool: event.approval.tool,
          summary: event.approval.summary,
          risk: event.approval.risk,
          environment: event.approval.environment
        })
        return
      case 'tool-started':
        this.publish({
          type: 'toolStarted',
          correlationId,
          toolCallId: event.toolCallId,
          tool: event.tool
        })
        return
      case 'tool-completed':
        this.publish({
          type: 'toolCompleted',
          correlationId,
          toolCallId: event.toolCallId,
          tool: event.tool,
          outcome: toolOutcome(event)
        })
        return
      case 'tool-failed':
        this.publish({
          type: 'toolFailed',
          correlationId,
          toolCallId: event.toolCallId,
          tool: event.tool,
          error: event.error
        })
        return
      case 'message-committed': {
        // The committed assistant message is the authoritative answer.
        if (event.message.role !== 'assistant') return
        const text = event.message.text.trim()
        if (!text) return
        this.publish({ type: 'responseFinal', correlationId, text, runId: event.runId })
        return
      }
      case 'run-completed': {
        this.setStatus(correlationId, 'completed')
        // Duplicate finals are deduped by the coordinator, so publishing the
        // summary text as a fallback cannot produce a second answer.
        const text = event.summary.text.trim() || this.streamed.trim()
        if (text) this.publish({ type: 'responseFinal', correlationId, text, runId: event.runId })
        this.lastRunId = event.runId
        this.activeRunId = null
        this.activeCorrelationId = null
        return
      }
      case 'run-cancelled':
        this.setStatus(correlationId, 'cancelled')
        this.publish({ type: 'runCancelled', correlationId })
        this.lastRunId = event.runId
        this.activeRunId = null
        this.activeCorrelationId = null
        return
      case 'run-failed':
        this.setStatus(correlationId, 'failed')
        this.publish({ type: 'runFailed', correlationId, error: event.error })
        this.lastRunId = event.runId
        this.activeRunId = null
        this.activeCorrelationId = null
        return
      default:
        return
    }
  }
}

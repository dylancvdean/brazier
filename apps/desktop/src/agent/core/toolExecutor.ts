/**
 * Application-owned tool execution: the approval round trip and the event
 * stream the UI renders.
 *
 * A runtime adapter calls [`AgentToolExecutor.execute`] and gets back text for
 * the model. It never decides whether a call is allowed, never sees the
 * filesystem, and cannot skip the approval step: the daemon refuses a call
 * whose approval is missing, spent, or issued for different arguments.
 */

import { reviewEligibility, type ApprovalReviewVerdict } from './approvalReview'
import type { BrokerClient, ToolExecResponse } from './brokerClient'
import type {
  AgentApproval,
  AgentEnvironment,
  AgentEvent,
  AgentToolDefinition,
  ApprovalStatus,
  SandboxDescription
} from './types'

/** How long each wait for a user decision blocks before re-checking. */
const APPROVAL_WAIT_MS = 60_000
/** Back-to-back failed long-polls before the wait is treated as expired. */
const APPROVAL_WAIT_MAX_FAILURES = 5
const APPROVAL_RETRY_DELAY_MS = 1_000

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(done, ms)
    function done(): void {
      clearTimeout(timer)
      signal?.removeEventListener('abort', done)
      resolve()
    }
    signal?.addEventListener('abort', done, { once: true })
  })
}

export type ToolImage = {
  mimeType: string
  data: string
}

export type ToolExecutionOutcome = {
  output: string
  isError: boolean
  /** True when the user or the policy refused, rather than the tool failing. */
  denied: boolean
  environment: AgentEnvironment
  sandbox: SandboxDescription
  changedPaths: string[]
  exitCode?: number | null
  truncated: boolean
  artifactId?: string
  executionId?: string
  durationMs: number
  /** Page renders and similar images for vision models. */
  images: ToolImage[]
}

export type ExecuteToolRequest = {
  runId: string
  toolCallId: string
  tool: string
  args: Record<string, unknown>
  /** Model-supplied justification, shown in the approval dialog. */
  reason?: string
  signal?: AbortSignal
}

/** Hook for "Approve for me". The session supplies the model and the instructions. */
export type ApprovalReviewer = {
  /** Read live, so toggling the setting mid-run applies to the next approval. */
  enabled(): Promise<boolean>
  review(request: ExecuteToolRequest, approval: AgentApproval): Promise<ApprovalReviewVerdict>
}

type DecidedApproval = Omit<AgentApproval, 'status'> & { status: ApprovalStatus }

/** Monotonic sequence for the event stream of one session. */
export class EventSequencer {
  private next = 0

  take(): number {
    this.next += 1
    return this.next
  }
}

export class AgentToolExecutor {
  private readonly broker: BrokerClient
  private readonly sessionId: string
  private readonly emit: (event: AgentEvent) => void
  private readonly sequencer: EventSequencer
  private definitions: Map<string, AgentToolDefinition>
  private fallbackSandbox: SandboxDescription
  /** Approvals put to the user during the current run, for the run summary. */
  approvalsRequested = 0
  /** Approvals "Approve for me" granted during the current run. */
  approvalsAutoApproved = 0
  private reviewer?: ApprovalReviewer
  /** Worker-handled tools that never reach the daemon exec endpoint. */
  private readonly localHandlers = new Map<
    string,
    (request: ExecuteToolRequest) => Promise<ToolExecutionOutcome>
  >()

  constructor(options: {
    broker: BrokerClient
    sessionId: string
    emit: (event: AgentEvent) => void
    sequencer: EventSequencer
    definitions: AgentToolDefinition[]
    sandbox: SandboxDescription
  }) {
    this.broker = options.broker
    this.sessionId = options.sessionId
    this.emit = options.emit
    this.sequencer = options.sequencer
    this.definitions = new Map(options.definitions.map((tool) => [tool.name, tool]))
    this.fallbackSandbox = options.sandbox
  }

  setDefinitions(definitions: AgentToolDefinition[]): void {
    this.definitions = new Map(definitions.map((tool) => [tool.name, tool]))
  }

  setReviewer(reviewer: ApprovalReviewer | undefined): void {
    this.reviewer = reviewer
  }

  /** Register a tool that the worker executes itself (e.g. `spawn_subagent`). */
  setLocalHandler(
    tool: string,
    handler: (request: ExecuteToolRequest) => Promise<ToolExecutionOutcome>
  ): void {
    this.localHandlers.set(tool, handler)
  }

  private event<T extends AgentEvent['type']>(
    runId: string,
    type: T,
    rest: Omit<Extract<AgentEvent, { type: T }>, keyof BaseFields | 'type'>
  ): void {
    this.emit({
      type,
      sessionId: this.sessionId,
      runId,
      timestamp: new Date().toISOString(),
      sequence: this.sequencer.take(),
      ...rest
    } as AgentEvent)
  }

  async execute(request: ExecuteToolRequest): Promise<ToolExecutionOutcome> {
    const definition = this.definitions.get(request.tool)
    const requestedEnvironment: AgentEnvironment = definition?.defaultEnvironment ?? 'sandbox'

    this.event(request.runId, 'tool-call-proposed', {
      toolCallId: request.toolCallId,
      tool: request.tool,
      args: request.args,
      environment: requestedEnvironment,
      risk: definition?.risk ?? 'execute'
    })
    this.event(request.runId, 'tool-started', {
      toolCallId: request.toolCallId,
      tool: request.tool,
      args: request.args,
      environment: requestedEnvironment,
      sandbox: this.fallbackSandbox
    })

    const local = this.localHandlers.get(request.tool)
    if (local) {
      try {
        const outcome = await local(request)
        if (outcome.isError) {
          this.event(request.runId, 'tool-failed', {
            toolCallId: request.toolCallId,
            tool: request.tool,
            environment: outcome.environment,
            sandbox: outcome.sandbox,
            error: outcome.output,
            denied: outcome.denied,
            durationMs: outcome.durationMs
          })
        } else {
          this.event(request.runId, 'tool-completed', {
            toolCallId: request.toolCallId,
            tool: request.tool,
            environment: outcome.environment,
            sandbox: outcome.sandbox,
            output: outcome.output,
            truncated: outcome.truncated,
            artifactId: outcome.artifactId,
            exitCode: outcome.exitCode,
            changedPaths: outcome.changedPaths,
            durationMs: outcome.durationMs,
            executionId: outcome.executionId
          })
        }
        return outcome
      } catch (cause) {
        const message = cause instanceof Error ? cause.message : String(cause)
        return this.fail(request, message, false, 0)
      }
    }

    let approvalId: string | undefined
    // At most one approval round trip per call: the daemon issues an approval
    // bound to these exact arguments, and a second refusal is final.
    for (let attempt = 0; attempt < 2; attempt += 1) {
      if (request.signal?.aborted) {
        return this.fail(request, 'The run was cancelled before this tool ran.', true, 0)
      }
      let response: ToolExecResponse
      try {
        const brokerRequest = {
          sessionId: this.sessionId,
          runId: request.runId,
          toolCallId: request.toolCallId,
          tool: request.tool,
          arguments: request.args,
          environment: requestedEnvironment,
          reason: request.reason,
          approvalId
        }
        response =
          request.tool === 'shell_run'
            ? await this.broker.execToolStreaming(
                brokerRequest,
                (chunk) =>
                  this.event(request.runId, 'tool-output', {
                    toolCallId: request.toolCallId,
                    tool: request.tool,
                    chunk
                  }),
                request.signal
              )
            : await this.broker.execTool(brokerRequest, request.signal)
      } catch (cause) {
        const message = cause instanceof Error ? cause.message : String(cause)
        return this.fail(request, message, false, 0)
      }

      if (response.status === 'approval_required' && response.approval) {
        const decided =
          (await this.reviewForUser(request, response.approval)) ??
          (await this.awaitDecision(request, response.approval))
        if (decided.status === 'approved') {
          approvalId = response.approval.id
          // Transition timeline pills back from awaiting-approval to running.
          this.event(request.runId, 'tool-started', {
            toolCallId: request.toolCallId,
            tool: request.tool,
            args: request.args,
            environment: response.environment,
            sandbox: response.sandbox
          })
          continue
        }
        return this.fail(
          request,
          decided.status === 'denied'
            ? `The user denied this action.${decided.note ? ` Note: ${decided.note}` : ''}`
            : decided.status === 'cancelled'
              ? 'The run was cancelled while this action waited for approval.'
              : 'The approval request expired before anyone answered it.',
          true,
          response.duration_ms,
          response.sandbox,
          response.environment
        )
      }

      const outcome: ToolExecutionOutcome = {
        output: response.output,
        isError: response.is_error,
        denied: response.status === 'denied',
        environment: response.environment,
        sandbox: response.sandbox,
        changedPaths: response.changed_paths ?? [],
        exitCode: response.exit_code ?? null,
        truncated: Boolean(response.truncated),
        artifactId: response.artifact_id,
        executionId: response.execution_id,
        durationMs: response.duration_ms,
        images: (response.images ?? []).map((image) => ({
          mimeType: image.mime_type,
          data: image.data
        }))
      }

      if (outcome.isError) {
        this.event(request.runId, 'tool-failed', {
          toolCallId: request.toolCallId,
          tool: request.tool,
          environment: outcome.environment,
          sandbox: outcome.sandbox,
          error: outcome.output,
          denied: outcome.denied,
          durationMs: outcome.durationMs
        })
      } else {
        this.event(request.runId, 'tool-completed', {
          toolCallId: request.toolCallId,
          tool: request.tool,
          environment: outcome.environment,
          sandbox: outcome.sandbox,
          output: outcome.output,
          truncated: outcome.truncated,
          artifactId: outcome.artifactId,
          exitCode: outcome.exitCode,
          changedPaths: outcome.changedPaths,
          durationMs: outcome.durationMs,
          executionId: outcome.executionId
        })
      }
      return outcome
    }

    return this.fail(request, 'This action was not approved.', true, 0)
  }

  /**
   * Surface the request to the UI and wait for the user's answer. Returns the
   * decided approval, including any note the user attached; a wait that cannot
   * be completed resolves as expired, never as approved.
   */
  private async awaitDecision(
    request: ExecuteToolRequest,
    approval: AgentApproval
  ): Promise<DecidedApproval> {
    this.approvalsRequested += 1
    this.event(request.runId, 'approval-required', {
      toolCallId: request.toolCallId,
      approval
    })
    this.event(request.runId, 'elevation-requested', {
      toolCallId: request.toolCallId,
      approvalId: approval.id,
      request: approval.elevation
    })
    const decided = await this.waitForUser(request, approval)
    this.event(request.runId, 'approval-resolved', {
      toolCallId: request.toolCallId,
      approvalId: approval.id,
      status: decided.status,
      decidedBy: decided.status === 'approved' || decided.status === 'denied' ? 'user' : 'system'
    })
    return decided
  }

  private async waitForUser(
    request: ExecuteToolRequest,
    approval: AgentApproval
  ): Promise<DecidedApproval> {
    // Transport failures are retried for as long as the approval itself can
    // live: one dropped long-poll must not turn into a refused tool call.
    const deadlineMs = Date.now() + 16 * 60_000
    let consecutiveFailures = 0
    while (!request.signal?.aborted) {
      const remainingMs = deadlineMs - Date.now()
      if (remainingMs <= 0) {
        return { ...approval, status: 'expired' }
      }
      const waitMs = Math.min(APPROVAL_WAIT_MS, remainingMs)
      let current: AgentApproval
      try {
        current = await this.broker.waitForApproval(
          approval.id,
          waitMs,
          request.signal
        )
        consecutiveFailures = 0
      } catch {
        if (request.signal?.aborted) break
        consecutiveFailures += 1
        // A transport failure must never silently authorize anything.
        if (consecutiveFailures >= APPROVAL_WAIT_MAX_FAILURES) {
          return { ...approval, status: 'expired' }
        }
        await sleep(APPROVAL_RETRY_DELAY_MS * consecutiveFailures, request.signal)
        continue
      }
      if (current.status !== 'pending') return current
    }
    return { ...approval, status: 'cancelled' }
  }

  /**
   * "Approve for me": let the session's model approve an in-scope request.
   * Returns the decided approval when the model approved it, or `undefined`
   * to put the request to the user as usual.
   */
  private async reviewForUser(
    request: ExecuteToolRequest,
    approval: AgentApproval
  ): Promise<DecidedApproval | undefined> {
    const reviewer = this.reviewer
    if (!reviewer) return undefined
    let enabled = false
    try {
      enabled = await reviewer.enabled()
    } catch {
      enabled = false
    }
    if (!enabled) return undefined

    const escalate = (reason: string): undefined => {
      this.event(request.runId, 'approval-reviewed', {
        toolCallId: request.toolCallId,
        approvalId: approval.id,
        phase: 'escalated',
        reason
      })
      return undefined
    }

    const eligibility = reviewEligibility(approval)
    if (!eligibility.eligible) return escalate(eligibility.reason)

    this.event(request.runId, 'approval-reviewed', {
      toolCallId: request.toolCallId,
      approvalId: approval.id,
      phase: 'started'
    })
    const verdict = await reviewer.review(request, approval)
    if (request.signal?.aborted) return undefined
    if (verdict.decision !== 'approve') return escalate(verdict.reason)

    let decided: AgentApproval
    try {
      decided = await this.broker.decideApproval(
        approval.id,
        'approve',
        'once',
        `Approved for you by model review: ${verdict.reason}`
      )
    } catch (cause) {
      // Usually the user answered first or the request expired; the normal
      // wait reads whatever the daemon actually recorded.
      const detail = cause instanceof Error ? cause.message : String(cause)
      return escalate(`The review approved this, but the approval could not be recorded (${detail}).`)
    }
    if (decided.status !== 'approved') {
      return escalate('The approval was already decided elsewhere.')
    }
    this.approvalsAutoApproved += 1
    this.event(request.runId, 'approval-reviewed', {
      toolCallId: request.toolCallId,
      approvalId: approval.id,
      phase: 'approved',
      reason: verdict.reason
    })
    this.event(request.runId, 'approval-resolved', {
      toolCallId: request.toolCallId,
      approvalId: approval.id,
      status: 'approved',
      decidedBy: 'model-review'
    })
    return decided
  }

  private fail(
    request: ExecuteToolRequest,
    error: string,
    denied: boolean,
    durationMs: number,
    sandbox: SandboxDescription = this.fallbackSandbox,
    environment: AgentEnvironment = 'sandbox'
  ): ToolExecutionOutcome {
    this.event(request.runId, 'tool-failed', {
      toolCallId: request.toolCallId,
      tool: request.tool,
      environment,
      sandbox,
      error,
      denied,
      durationMs
    })
    return {
      output: error,
      isError: true,
      denied,
      environment,
      sandbox,
      changedPaths: [],
      truncated: false,
      durationMs,
      images: []
    }
  }
}

type BaseFields = {
  sessionId: string
  runId: string
  timestamp: string
  sequence: number
}

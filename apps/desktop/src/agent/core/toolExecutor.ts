/**
 * Application-owned tool execution: the approval round trip and the event
 * stream the UI renders.
 *
 * A runtime adapter calls [`AgentToolExecutor.execute`] and gets back text for
 * the model. It never decides whether a call is allowed, never sees the
 * filesystem, and cannot skip the approval step: the daemon refuses a call
 * whose approval is missing, spent, or issued for different arguments.
 */

import type { BrokerClient, ToolExecResponse } from './brokerClient'
import type {
  AgentApproval,
  AgentEnvironment,
  AgentEvent,
  AgentToolDefinition,
  SandboxDescription
} from './types'

/** How long each wait for a user decision blocks before re-checking. */
const APPROVAL_WAIT_MS = 60_000

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
  /** Approvals requested during the current run, for the run summary. */
  approvalsRequested = 0

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

    let approvalId: string | undefined
    // At most one approval round trip per call: the daemon issues an approval
    // bound to these exact arguments, and a second refusal is final.
    for (let attempt = 0; attempt < 2; attempt += 1) {
      if (request.signal?.aborted) {
        return this.fail(request, 'The run was cancelled before this tool ran.', true, 0)
      }
      let response: ToolExecResponse
      try {
        response = await this.broker.execTool(
          {
            sessionId: this.sessionId,
            runId: request.runId,
            toolCallId: request.toolCallId,
            tool: request.tool,
            arguments: request.args,
            environment: requestedEnvironment,
            reason: request.reason,
            approvalId
          },
          request.signal
        )
      } catch (cause) {
        const message = cause instanceof Error ? cause.message : String(cause)
        return this.fail(request, message, false, 0)
      }

      if (response.status === 'approval_required' && response.approval) {
        this.approvalsRequested += 1
        const decided = await this.awaitDecision(request, response.approval)
        if (decided.status === 'approved') {
          approvalId = response.approval.id
          continue
        }
        return this.fail(
          request,
          decided.status === 'denied'
            ? `The user denied this action.${decided.note ? ` Note: ${decided.note}` : ''}`
            : 'The approval request expired or the run was cancelled.',
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
        durationMs: response.duration_ms
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
  ): Promise<AgentApproval> {
    this.event(request.runId, 'approval-required', {
      toolCallId: request.toolCallId,
      approval
    })
    this.event(request.runId, 'elevation-requested', {
      toolCallId: request.toolCallId,
      approvalId: approval.id,
      request: approval.elevation
    })

    while (!request.signal?.aborted) {
      let current: AgentApproval
      try {
        current = await this.broker.waitForApproval(
          approval.id,
          APPROVAL_WAIT_MS,
          request.signal
        )
      } catch {
        // A transport hiccup must never silently authorize anything.
        return { ...approval, status: 'expired' }
      }
      if (current.status !== 'pending') return current
    }
    return { ...approval, status: 'expired' }
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
      durationMs
    }
  }
}

type BaseFields = {
  sessionId: string
  runId: string
  timestamp: string
  sequence: number
}

import { describe, expect, it, vi } from 'vitest'

import type { BrokerClient, ToolExecResponse } from './brokerClient'
import { AgentToolExecutor, EventSequencer } from './toolExecutor'
import type { AgentApproval, AgentEvent, AgentToolDefinition, SandboxDescription } from './types'

const sandbox: SandboxDescription = {
  backend: 'seatbelt',
  profile: 'workspace',
  isolated: true,
  network: false,
  detail: 'Seatbelt confines writes to the workspace.'
}

const definitions: AgentToolDefinition[] = [
  {
    name: 'fs_write',
    label: 'Write',
    description: 'Write a file.',
    inputSchema: { type: 'object', properties: {}, required: [] },
    risk: 'write',
    executes: false,
    needsWorkspace: true,
    defaultEnvironment: 'sandbox'
  },
  {
    name: 'shell_run',
    label: 'Run',
    description: 'Run a command.',
    inputSchema: { type: 'object', properties: {}, required: [] },
    risk: 'execute',
    executes: true,
    needsWorkspace: true,
    defaultEnvironment: 'sandbox'
  }
]

function completed(overrides: Partial<ToolExecResponse> = {}): ToolExecResponse {
  return {
    status: 'completed',
    tool: 'fs_write',
    environment: 'sandbox',
    risk: 'write',
    sandbox,
    output: 'Wrote 2 bytes to a.txt.',
    truncated: false,
    changed_paths: ['a.txt'],
    duration_ms: 4,
    is_error: false,
    execution_id: 'exec-1',
    ...overrides
  }
}

function approvalRequired(approval: AgentApproval): ToolExecResponse {
  return {
    status: 'approval_required',
    tool: approval.tool,
    environment: approval.environment,
    risk: approval.risk,
    sandbox: approval.sandbox,
    output: '',
    duration_ms: 1,
    is_error: false,
    approval
  }
}

function pendingApproval(overrides: Partial<AgentApproval> = {}): AgentApproval {
  return {
    id: 'approval-1',
    session_id: 'session-1',
    tool: 'fs_write',
    arguments: { path: 'a.txt', content: 'hi' },
    arguments_hash: 'hash',
    environment: 'sandbox',
    risk: 'write',
    scope_key: 'fs-write:workspace',
    allow_session_scope: true,
    elevation: { reason: 'Needs to write the file' },
    summary: 'Write a.txt in the sandbox',
    sandbox,
    execution_location: {
      kind: 'daemon',
      daemon_instance_id: 'daemon-1',
      daemon_display_name: 'Studio Mac',
      platform: 'macos',
      arch: 'aarch64'
    },
    status: 'pending',
    created_at: '2026-07-24T00:00:00Z',
    ...overrides
  }
}

type FakeBroker = {
  client: BrokerClient
  execTool: ReturnType<typeof vi.fn>
  execToolStreaming: ReturnType<typeof vi.fn>
  waitForApproval: ReturnType<typeof vi.fn>
}

function fakeBroker(
  execResponses: ToolExecResponse[],
  approvalStates: AgentApproval[] = []
): FakeBroker {
  const execTool = vi.fn(async () => {
    const next = execResponses.shift()
    if (!next) throw new Error('unexpected extra exec call')
    return next
  })
  const execToolStreaming = vi.fn(
    async (_request: unknown, onOutput: (chunk: string) => void) => {
      onOutput('live output\n')
      const next = execResponses.shift()
      if (!next) throw new Error('unexpected extra exec call')
      return next
    }
  )
  const waitForApproval = vi.fn(async () => {
    const next = approvalStates.shift()
    if (!next) throw new Error('unexpected extra approval wait')
    return next
  })
  return {
    client: { execTool, execToolStreaming, waitForApproval } as unknown as BrokerClient,
    execTool,
    execToolStreaming,
    waitForApproval
  }
}

function makeExecutor(broker: BrokerClient): { executor: AgentToolExecutor; events: AgentEvent[] } {
  const events: AgentEvent[] = []
  const executor = new AgentToolExecutor({
    broker,
    sessionId: 'session-1',
    emit: (event) => events.push(event),
    sequencer: new EventSequencer(),
    definitions,
    sandbox
  })
  return { executor, events }
}

const request = {
  runId: 'run-1',
  toolCallId: 'call-1',
  tool: 'fs_write',
  args: { path: 'a.txt', content: 'hi' }
}

describe('AgentToolExecutor', () => {
  it('reports an allowed call as started then completed', async () => {
    const broker = fakeBroker([completed()])
    const { executor, events } = makeExecutor(broker.client)

    const outcome = await executor.execute(request)

    expect(outcome.isError).toBe(false)
    expect(outcome.changedPaths).toEqual(['a.txt'])
    expect(events.map((event) => event.type)).toEqual([
      'tool-call-proposed',
      'tool-started',
      'tool-completed'
    ])
    // Events carry a monotonic sequence so the UI can order them.
    expect(events.map((event) => event.sequence)).toEqual([1, 2, 3])
    expect(events.every((event) => event.runId === 'run-1')).toBe(true)
  })

  it('waits for approval, then retries the identical call with the grant', async () => {
    const approval = pendingApproval()
    const broker = fakeBroker(
      [approvalRequired(approval), completed()],
      [{ ...approval, status: 'approved', scope: 'once' }]
    )
    const { executor, events } = makeExecutor(broker.client)

    const outcome = await executor.execute(request)

    expect(outcome.isError).toBe(false)
    expect(events.map((event) => event.type)).toEqual([
      'tool-call-proposed',
      'tool-started',
      'approval-required',
      'elevation-requested',
      'approval-resolved',
      'tool-started',
      'tool-completed'
    ])
    expect(events.find((event) => event.type === 'approval-resolved')).toMatchObject({
      approvalId: 'approval-1',
      status: 'approved',
      decidedBy: 'user'
    })
    // The retry must carry the approval and the same arguments; the daemon
    // rejects a grant spent on anything else.
    expect(broker.execTool).toHaveBeenCalledTimes(2)
    const retry = broker.execTool.mock.calls[1]?.[0] as { approvalId?: string; arguments: unknown }
    expect(retry.approvalId).toBe('approval-1')
    expect(retry.arguments).toEqual(request.args)
    expect(executor.approvalsRequested).toBe(1)
  })

  it('does not run the tool when the user denies it', async () => {
    const approval = pendingApproval()
    const broker = fakeBroker(
      [approvalRequired(approval)],
      [{ ...approval, status: 'denied', note: 'not that file' }]
    )
    const { executor, events } = makeExecutor(broker.client)

    const outcome = await executor.execute(request)

    expect(outcome.isError).toBe(true)
    expect(outcome.denied).toBe(true)
    expect(outcome.output).toContain('denied')
    expect(outcome.output).toContain('not that file')
    expect(broker.execTool).toHaveBeenCalledTimes(1)
    expect(events.at(-1)?.type).toBe('tool-failed')
  })

  it('keeps waiting while the approval is still pending', async () => {
    const approval = pendingApproval()
    const broker = fakeBroker(
      [approvalRequired(approval), completed()],
      [approval, approval, { ...approval, status: 'approved' }]
    )
    const { executor } = makeExecutor(broker.client)

    const outcome = await executor.execute(request)

    expect(broker.waitForApproval).toHaveBeenCalledTimes(3)
    expect(outcome.isError).toBe(false)
  })

  it('treats an expired approval as a refusal', async () => {
    const approval = pendingApproval()
    const broker = fakeBroker([approvalRequired(approval)], [{ ...approval, status: 'expired' }])
    const { executor } = makeExecutor(broker.client)

    const outcome = await executor.execute(request)

    expect(outcome.isError).toBe(true)
    expect(outcome.denied).toBe(true)
    expect(outcome.output).toContain('expired')
  })

  it('does not authorize anything when the approval wait keeps failing', async () => {
    vi.useFakeTimers()
    try {
      const approval = pendingApproval()
      const execTool = vi.fn(async () => approvalRequired(approval))
      const waitForApproval = vi.fn(async () => {
        throw new Error('connection reset')
      })
      const { executor, events } = makeExecutor({
        execTool,
        waitForApproval
      } as unknown as BrokerClient)

      const pending = executor.execute(request)
      await vi.runAllTimersAsync()
      const outcome = await pending

      expect(outcome.isError).toBe(true)
      expect(outcome.denied).toBe(true)
      expect(execTool).toHaveBeenCalledTimes(1)
      expect(waitForApproval.mock.calls.length).toBeGreaterThan(1)
      expect(events.find((event) => event.type === 'approval-resolved')).toMatchObject({
        status: 'expired',
        decidedBy: 'system'
      })
    } finally {
      vi.useRealTimers()
    }
  })

  it('survives a dropped approval long-poll and still honors the decision', async () => {
    vi.useFakeTimers()
    try {
      const approval = pendingApproval()
      const execTool = vi
        .fn()
        .mockResolvedValueOnce(approvalRequired(approval))
        .mockResolvedValueOnce(completed())
      const waitForApproval = vi
        .fn()
        .mockRejectedValueOnce(new Error('The operation was aborted due to timeout'))
        .mockResolvedValueOnce({ ...approval, status: 'approved' })
      const { executor } = makeExecutor({ execTool, waitForApproval } as unknown as BrokerClient)

      const pending = executor.execute(request)
      await vi.runAllTimersAsync()
      const outcome = await pending

      expect(outcome.isError).toBe(false)
      expect(waitForApproval).toHaveBeenCalledTimes(2)
      expect(execTool).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })

  it('stops after a single approval round trip', async () => {
    const approval = pendingApproval()
    // A daemon that asks again after a grant must not loop forever.
    const broker = fakeBroker(
      [approvalRequired(approval), approvalRequired(approval)],
      [{ ...approval, status: 'approved' }, { ...approval, status: 'approved' }]
    )
    const { executor } = makeExecutor(broker.client)

    const outcome = await executor.execute(request)

    expect(outcome.isError).toBe(true)
    expect(broker.execTool).toHaveBeenCalledTimes(2)
  })

  it('surfaces a refusal from the policy without asking the user', async () => {
    const broker = fakeBroker([
      completed({
        status: 'denied',
        is_error: true,
        output: 'Refused: sandbox-only mode refuses host access.',
        changed_paths: []
      })
    ])
    const { executor, events } = makeExecutor(broker.client)

    const outcome = await executor.execute(request)

    expect(outcome.denied).toBe(true)
    expect(events.map((event) => event.type)).not.toContain('approval-required')
    expect(events.at(-1)?.type).toBe('tool-failed')
  })

  it('reports a transport failure as a tool failure, not a denial', async () => {
    const execTool = vi.fn(async () => {
      throw new Error('daemon unavailable')
    })
    const { executor, events } = makeExecutor({ execTool } as unknown as BrokerClient)

    const outcome = await executor.execute(request)

    expect(outcome.isError).toBe(true)
    expect(outcome.denied).toBe(false)
    expect(outcome.output).toContain('daemon unavailable')
    expect(events.at(-1)?.type).toBe('tool-failed')
  })

  it('refuses to run once the run has been cancelled', async () => {
    const controller = new AbortController()
    controller.abort()
    const execTool = vi.fn()
    const { executor } = makeExecutor({ execTool } as unknown as BrokerClient)

    const outcome = await executor.execute({ ...request, signal: controller.signal })

    expect(execTool).not.toHaveBeenCalled()
    expect(outcome.isError).toBe(true)
    expect(outcome.output).toContain('cancelled')
  })

  it('passes host results through with their real environment', async () => {
    const broker = fakeBroker([
      completed({
        tool: 'shell_run',
        environment: 'host',
        sandbox: {
          backend: 'none',
          profile: 'host',
          isolated: false,
          network: true,
          detail: 'Host execution: no sandbox, full user privileges.'
        },
        output: 'done\n\n[exit code 0]',
        exit_code: 0,
        changed_paths: []
      })
    ])
    const { executor, events } = makeExecutor(broker.client)

    const outcome = await executor.execute({ ...request, tool: 'shell_run' })

    expect(outcome.environment).toBe('host')
    expect(outcome.sandbox.isolated).toBe(false)
    expect(events.find((event) => event.type === 'tool-output')).toMatchObject({
      type: 'tool-output',
      chunk: 'live output\n'
    })
    const completedEvent = events.find((event) => event.type === 'tool-completed')
    expect(completedEvent && 'sandbox' in completedEvent && completedEvent.sandbox.isolated).toBe(
      false
    )
  })

  it('runs local handlers without calling the daemon exec endpoint', async () => {
    const broker = fakeBroker([])
    const { executor, events } = makeExecutor(broker.client)
    executor.setLocalHandler('spawn_subagent', async () => ({
      output: 'child done',
      isError: false,
      denied: false,
      environment: 'sandbox',
      sandbox,
      changedPaths: [],
      truncated: false,
      durationMs: 12,
      images: []
    }))

    const outcome = await executor.execute({
      runId: 'run-1',
      toolCallId: 'call-sub',
      tool: 'spawn_subagent',
      args: { prompt: 'investigate' }
    })

    expect(outcome.output).toBe('child done')
    expect(broker.execTool).not.toHaveBeenCalled()
    expect(events.map((event) => event.type)).toEqual([
      'tool-call-proposed',
      'tool-started',
      'tool-completed'
    ])
  })
})

describe('AgentToolExecutor approve for me', () => {
  function reviewBroker(approval: AgentApproval): {
    client: BrokerClient
    execTool: ReturnType<typeof vi.fn>
    waitForApproval: ReturnType<typeof vi.fn>
    decideApproval: ReturnType<typeof vi.fn>
  } {
    const execTool = vi
      .fn()
      .mockResolvedValueOnce(approvalRequired(approval))
      .mockResolvedValueOnce(completed())
    const waitForApproval = vi.fn(async () => ({ ...approval, status: 'approved' as const }))
    const decideApproval = vi.fn(async () => ({ ...approval, status: 'approved' as const }))
    return {
      client: { execTool, waitForApproval, decideApproval } as unknown as BrokerClient,
      execTool,
      waitForApproval,
      decideApproval
    }
  }

  it('approves an in-scope request without asking the user', async () => {
    const approval = pendingApproval()
    const broker = reviewBroker(approval)
    const { executor, events } = makeExecutor(broker.client)
    const review = vi.fn(async () => ({ decision: 'approve' as const, reason: 'User asked for a.txt.' }))
    executor.setReviewer({ enabled: async () => true, review })

    const outcome = await executor.execute(request)

    expect(outcome.isError).toBe(false)
    expect(review).toHaveBeenCalledOnce()
    expect(broker.decideApproval).toHaveBeenCalledWith(
      'approval-1',
      'approve',
      'once',
      expect.stringContaining('User asked for a.txt.')
    )
    expect(broker.waitForApproval).not.toHaveBeenCalled()
    const retry = broker.execTool.mock.calls[1]?.[0] as { approvalId?: string }
    expect(retry.approvalId).toBe('approval-1')
    const types = events.map((event) => event.type)
    expect(types).not.toContain('approval-required')
    expect(events.filter((event) => event.type === 'approval-reviewed')).toMatchObject([
      { phase: 'started' },
      { phase: 'approved', reason: 'User asked for a.txt.' }
    ])
    expect(events.find((event) => event.type === 'approval-resolved')).toMatchObject({
      decidedBy: 'model-review',
      status: 'approved'
    })
    expect(executor.approvalsAutoApproved).toBe(1)
    expect(executor.approvalsRequested).toBe(0)
  })

  it('hands the request to the user when the review is unsure', async () => {
    const approval = pendingApproval()
    const broker = reviewBroker(approval)
    const { executor, events } = makeExecutor(broker.client)
    executor.setReviewer({
      enabled: async () => true,
      review: async () => ({ decision: 'ask-user', reason: 'Not mentioned by the user.' })
    })

    const outcome = await executor.execute(request)

    expect(outcome.isError).toBe(false)
    expect(broker.decideApproval).not.toHaveBeenCalled()
    expect(broker.waitForApproval).toHaveBeenCalledOnce()
    const types = events.map((event) => event.type)
    expect(types.indexOf('approval-reviewed')).toBeLessThan(types.indexOf('approval-required'))
    expect(events.filter((event) => event.type === 'approval-reviewed').at(-1)).toMatchObject({
      phase: 'escalated',
      reason: 'Not mentioned by the user.'
    })
    expect(executor.approvalsRequested).toBe(1)
  })

  it.each([
    ['host', pendingApproval({ environment: 'host' })],
    ['destructive', pendingApproval({ risk: 'destructive' })],
    [
      'host-elevating',
      pendingApproval({ elevation: { reason: 'x', requested_host_execution: true } })
    ]
  ])('never lets the model decide a %s action', async (_label, approval) => {
    const broker = reviewBroker(approval)
    const { executor, events } = makeExecutor(broker.client)
    const review = vi.fn(async () => ({ decision: 'approve' as const, reason: 'fine' }))
    executor.setReviewer({ enabled: async () => true, review })

    await executor.execute(request)

    expect(review).not.toHaveBeenCalled()
    expect(broker.decideApproval).not.toHaveBeenCalled()
    expect(events.map((event) => event.type)).toContain('approval-required')
  })

  it('stays out of the way when the setting is off', async () => {
    const approval = pendingApproval()
    const broker = reviewBroker(approval)
    const { executor, events } = makeExecutor(broker.client)
    const review = vi.fn()
    executor.setReviewer({ enabled: async () => false, review })

    await executor.execute(request)

    expect(review).not.toHaveBeenCalled()
    expect(events.map((event) => event.type)).not.toContain('approval-reviewed')
  })

  it('falls back to the user when the approval cannot be recorded', async () => {
    const approval = pendingApproval()
    const broker = reviewBroker(approval)
    broker.decideApproval.mockRejectedValueOnce(new Error('approval is no longer pending'))
    const { executor, events } = makeExecutor(broker.client)
    executor.setReviewer({
      enabled: async () => true,
      review: async () => ({ decision: 'approve', reason: 'ok' })
    })

    const outcome = await executor.execute(request)

    expect(outcome.isError).toBe(false)
    expect(broker.waitForApproval).toHaveBeenCalledOnce()
    expect(events.map((event) => event.type)).toContain('approval-required')
    expect(executor.approvalsAutoApproved).toBe(0)
  })
})

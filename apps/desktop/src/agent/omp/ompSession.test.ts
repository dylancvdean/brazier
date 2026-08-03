import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { AgentEvent, AgentSession } from '../core/types'
import { OmpAgentRuntime } from './ompRuntime'

/**
 * Drives a real `OmpAgentSession` against a scripted `omp` fixture binary, so
 * the stateful core (run loop, extension-UI dialogs, subagent subscription,
 * sidecar restart) is exercised the way the app actually runs it. Each received
 * stdin frame is appended to a log file the test reads back.
 */

function fixtureScript(): string {
  return `#!/usr/bin/env node
const fs = require('fs')
const LOG = process.env.OMP_TEST_LOG
const log = (obj) => { if (LOG) fs.appendFileSync(LOG, JSON.stringify(obj) + '\\n') }
const send = (obj) => process.stdout.write(JSON.stringify(obj) + '\\n')
send({ type: 'ready', protocolVersion: 1, supportedProtocolVersions: [1, 2] })
let buffered = ''
process.stdin.setEncoding('utf8')
process.stdin.on('data', (chunk) => {
  buffered += chunk
  let idx
  while ((idx = buffered.indexOf('\\n')) !== -1) {
    const line = buffered.slice(0, idx)
    buffered = buffered.slice(idx + 1)
    if (!line.trim()) continue
    let cmd
    try { cmd = JSON.parse(line) } catch { continue }
    log(cmd)
    handle(cmd)
  }
})
function handle(cmd) {
  const id = cmd.id
  switch (cmd.type) {
    case 'set_model':
      send({ id, type: 'response', command: 'set_model', success: true, data: { provider: 'brazier', id: cmd.modelId } })
      return
    case 'set_host_tools':
      send({ id, type: 'response', command: 'set_host_tools', success: true, data: { toolNames: [] } })
      return
    case 'set_subagent_subscription':
      send({ id, type: 'response', command: 'set_subagent_subscription', success: true, data: { level: cmd.level } })
      return
    case 'get_available_commands':
      send({ id, type: 'response', command: 'get_available_commands', success: true, data: { commands: [{ name: '/review', source: 'builtin' }] } })
      return
    case 'get_state':
      send({ id, type: 'response', command: 'get_state', success: true, data: { sessionId: 'fixture', thinkingLevel: 'high', isStreaming: false, isCompacting: false, autoCompactionEnabled: true, messageCount: 0, queuedMessageCount: 0, todoPhases: [] } })
      return
    case 'compact':
      send({ id, type: 'response', command: 'compact', success: true, data: { summary: 'compacted', removedMessages: 0 } })
      return
    case 'prompt':
      send({ id, type: 'response', command: 'prompt', success: true, data: { agentInvoked: true } })
      send({ type: 'agent_start' })
      if (process.env.OMP_TEST_DIALOG === 'input') {
        send({ type: 'extension_ui_request', id: 'ui-1', method: 'input', title: 'Name' })
        return
      }
      if (process.env.OMP_TEST_TIMEOUT_DIALOG) {
        send({ type: 'extension_ui_request', id: 'ui-1', method: 'confirm', title: 'Proceed?', message: 'Continue?' })
        return
      }
      send({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', delta: 'hello' }, message: { role: 'assistant', content: [] } })
      send({ type: 'message_end', message: { role: 'assistant', content: [{ type: 'text', text: 'hello' }] } })
      send({ type: 'agent_end' })
      return
    case 'extension_ui_response':
      send({ type: 'message_end', message: { role: 'assistant', content: [{ type: 'text', text: 'resolved:' + JSON.stringify(cmd) }] } })
      send({ type: 'agent_end' })
      return
    default:
      send({ id, type: 'response', command: cmd.type, success: true })
  }
}
process.on('SIGTERM', () => process.exit(0))
`
}

async function collectRun(iter: AsyncIterable<AgentEvent>): Promise<AgentEvent[]> {
  const out: AgentEvent[] = []
  for await (const event of iter) out.push(event)
  return out
}

async function waitFor(predicate: () => boolean, timeoutMs: number): Promise<void> {
  const start = Date.now()
  while (!predicate()) {
    if (Date.now() - start > timeoutMs) throw new Error('Timed out waiting for a fixture condition.')
    await new Promise((resolve) => setTimeout(resolve, 20))
  }
}

function readLog(path: string): Array<Record<string, unknown>> {
  return readFileSync(path, 'utf8')
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Record<string, unknown>)
}

let directory: string | undefined

afterEach(() => {
  delete process.env.OMP_TEST_LOG
  delete process.env.OMP_TEST_DIALOG
  delete process.env.OMP_TEST_TIMEOUT_DIALOG
  delete process.env.BRAZIER_OMP_DIALOG_TIMEOUT_MS
  if (directory) rmSync(directory, { recursive: true, force: true })
  directory = undefined
})

describe.skipIf(process.platform === 'win32')('OmpAgentSession against a fixture sidecar', () => {
  async function openSession(): Promise<{ session: AgentSession; logPath: string; broker: Record<string, unknown> }> {
    directory = mkdtempSync(join(tmpdir(), 'brazier-omp-session-'))
    const binary = join(directory, 'omp-fixture.js')
    writeFileSync(binary, fixtureScript(), { mode: 0o755 })
    chmodSync(binary, 0o755)
    const logPath = join(directory, 'log.jsonl')
    process.env.OMP_TEST_LOG = logPath
    const workspace = join(directory, 'workspace')
    mkdirSync(workspace)

    const broker: Record<string, unknown> = {
      agentPreference: vi.fn(async () => ({
        default_runtime_id: 'omp',
        omp_profile: { binary_path: binary }
      })),
      models: vi.fn(async () => [
        { id: 'gguf:main/model.gguf', capabilities: { input_modalities: ['text'], tools: true, reasoning: true, max_context_length: 32768 } }
      ]),
      openAiBaseUrl: () => 'http://127.0.0.1:1/v1',
      apiKey: () => 'test-key',
      updateSession: vi.fn(async () => ({ id: 'sess-1' })),
      appendMessages: vi.fn(async () => undefined),
      cancel: vi.fn(async () => undefined)
    }
    const runtime = new OmpAgentRuntime(broker as never)
    const session = await runtime.createSession({
      sessionId: 'sess-1',
      model: { id: 'gguf:main/model.gguf', name: 'Main' },
      systemPrompt: '',
      tools: [],
      messages: [],
      capabilities: {
        nativeToolCalling: true,
        parallelToolCalling: true,
        supportsReasoningStream: true,
        harmony: true,
        reliableJson: true
      },
      preloaded: {
        session: {
          id: 'sess-1',
          title: 'Task',
          workspace_path: workspace,
          permission_mode: 'ask',
          permission_settings: {
            auto_approve_host_actions: false,
            auto_approve_sandboxed_actions: true
          },
          enabled_tools: null,
          created_at: '2026-01-01',
          updated_at: '2026-01-01',
          runtime_id: 'omp',
          runtime_metadata: null
        },
        tool_executions: [],
        sandbox: { backend: 'omp', profile: 'host', isolated: false, network: true, detail: 'host' }
      }
    })
    return { session, logPath, broker }
  }

  it('registers the model, host tools, and subagent subscription at open', async () => {
    const { session, logPath } = await openSession()
    const log = readLog(logPath)
    expect(log.some((entry) => entry.type === 'set_model')).toBe(true)
    expect(log.some((entry) => entry.type === 'set_host_tools')).toBe(true)
    expect(
      log.some((entry) => entry.type === 'set_subagent_subscription' && entry.level === 'progress')
    ).toBe(true)
    await session.dispose()
  })

  it('runs a prompt to completion and commits the assistant text', async () => {
    const { session, broker } = await openSession()
    const events = await collectRun(session.run({ text: 'hello' }))
    expect(events.some((event) => event.type === 'run-completed')).toBe(true)
    const completed = events.find((event) => event.type === 'run-completed')
    expect(completed && completed.type === 'run-completed' ? completed.summary.text : '').toBe('hello')
    expect(broker.appendMessages).toHaveBeenCalled()
    await session.dispose()
  })

  it('surfaces an extension-UI dialog and forwards the resolution to the sidecar', async () => {
    process.env.OMP_TEST_DIALOG = 'input'
    const { session, logPath } = await openSession()
    const frames: Array<Record<string, unknown>> = []
    const unsubscribe = session.subscribeRuntimeFrames!((frame) => frames.push(frame))

    const eventsPromise = collectRun(session.run({ text: 'do it' }))
    await waitFor(
      () => frames.some((frame) => frame.type === 'extension_ui_request' && frame.id === 'ui-1'),
      5_000
    )
    const result = await session.resolveExtensionUi!({
      type: 'extension_ui_response',
      id: 'ui-1',
      value: 'world'
    })
    expect(result).toEqual({ resolved: true })

    const events = await eventsPromise
    expect(events.some((event) => event.type === 'run-completed')).toBe(true)
    const log = readLog(logPath)
    expect(
      log.some(
        (entry) => entry.type === 'extension_ui_response' && entry.id === 'ui-1' && entry.value === 'world'
      )
    ).toBe(true)
    unsubscribe()
    await session.dispose()
  })

  it('unblocks the sidecar when an unanswered dialog times out', async () => {
    process.env.BRAZIER_OMP_DIALOG_TIMEOUT_MS = '100'
    process.env.OMP_TEST_TIMEOUT_DIALOG = '1'
    const { session, logPath } = await openSession()

    const events = await collectRun(session.run({ text: 'hi' }))
    expect(events.some((event) => event.type === 'run-completed')).toBe(true)

    const log = readLog(logPath)
    expect(
      log.some(
        (entry) =>
          entry.type === 'extension_ui_response' &&
          entry.id === 'ui-1' &&
          entry.cancelled === true &&
          entry.timedOut === true
      )
    ).toBe(true)
    await session.dispose()
  })

  it('passes arbitrary runtime commands straight through to the sidecar', async () => {
    const { session, logPath } = await openSession()
    const response = await session.sendRuntimeCommand!({ type: 'get_state' })
    expect(response).toMatchObject({ type: 'response', command: 'get_state', success: true })
    expect(readLog(logPath).some((entry) => entry.type === 'get_state')).toBe(true)
    await session.dispose()
  })
})

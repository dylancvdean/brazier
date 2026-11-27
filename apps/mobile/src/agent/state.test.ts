import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  agentErrorText, applyEvent, argsPreview, emptyView, permissionModesFor, selectableTools, summaryLine,
  transcriptRows, viewFromDetail, type AgentApproval, type AgentEvent, type AgentView,
} from './state.ts';

let sequence = 0;
const base = () => ({ sessionId: 's', runId: 'r', sequence: sequence += 1, timestamp: `2026-01-01T00:00:${String(sequence).padStart(2, '0')}Z` });
const sandbox = { backend: 'seatbelt', profile: 'workspace', isolated: true, network: false, detail: '' };
const reduce = (events: AgentEvent[], from: AgentView = emptyView) => events.reduce(applyEvent, from);

const approval = {
  id: 'a1', session_id: 's', tool: 'run_command', arguments: { command: 'touch x' }, arguments_hash: 'h',
  environment: 'sandbox', risk: 'execute', scope_key: 'run_command', allow_session_scope: true,
  elevation: { reason: 'Runs a command.' }, summary: 'Run touch x', sandbox,
  execution_location: { kind: 'daemon', daemon_instance_id: 'i', daemon_display_name: 'Host', platform: 'macos', arch: 'arm64' },
  status: 'pending', created_at: '2026-01-01T00:00:00Z',
} satisfies AgentApproval;

test('streams reasoning and text, then replaces them with the committed message', () => {
  const streaming = reduce([
    { ...base(), type: 'run-started' },
    { ...base(), type: 'model-loading', phase: 'load', message: 'Loading model…' },
    { ...base(), type: 'text-delta', channel: 'reasoning', delta: 'think' },
    { ...base(), type: 'text-delta', channel: 'text', delta: 'Hel' },
    { ...base(), type: 'text-delta', channel: 'text', delta: 'lo' },
  ]);
  assert.deepEqual([streaming.running, streaming.progress, streaming.reasoning, streaming.streaming],
    [true, '', 'think', 'Hello']);
  const committed = applyEvent(streaming, {
    ...base(), type: 'message-committed',
    message: { role: 'assistant', text: 'Hello', reasoning: 'think', timestamp: '2026-01-01T00:01:00Z' },
  });
  assert.deepEqual([committed.streaming, committed.reasoning, committed.messages.length], ['', '', 1]);
});

test('does not echo the user turn or show tool-only assistant turns and tool messages', () => {
  const view = reduce([
    { ...base(), type: 'message-committed', message: { role: 'user', text: 'hi', timestamp: 't' } },
    { ...base(), type: 'message-committed', message: { role: 'assistant', text: 'null', timestamp: 't' } },
    { ...base(), type: 'message-committed', message: { role: 'tool', toolCallId: 'c', tool: 'x', output: 'o', isError: false, timestamp: 't' } },
  ]);
  assert.equal(view.messages.length, 0);
});

test('follows a tool call through approval to completion', () => {
  const started: AgentEvent = { ...base(), type: 'tool-started', toolCallId: 'c1', tool: 'run_command', args: { command: 'touch x' }, environment: 'sandbox', sandbox };
  const waiting = reduce([
    { ...base(), type: 'run-started' }, started,
    { ...base(), type: 'approval-required', toolCallId: 'c1', approval },
    { ...base(), type: 'approval-required', toolCallId: 'c1', approval },
  ]);
  assert.equal(waiting.approvals.length, 1);
  assert.equal(waiting.timeline[0]?.status, 'awaiting-approval');

  const done = reduce([
    { ...base(), type: 'approval-resolved', toolCallId: 'c1', approvalId: 'a1', status: 'approved', decidedBy: 'user' },
    { ...base(), type: 'tool-output', toolCallId: 'c1', tool: 'run_command', chunk: 'par' },
    { ...base(), type: 'tool-output', toolCallId: 'c1', tool: 'run_command', chunk: 'tial' },
  ], waiting);
  assert.equal(done.approvals.length, 0);
  assert.equal(done.timeline[0]?.output, 'partial');

  const completed = applyEvent(done, {
    ...base(), type: 'tool-completed', toolCallId: 'c1', tool: 'run_command', environment: 'sandbox', sandbox,
    output: 'final', truncated: false, exitCode: 0, changedPaths: ['x'], durationMs: 5,
  });
  assert.deepEqual([completed.timeline[0]?.status, completed.timeline[0]?.output, completed.timeline[0]?.changedPaths],
    ['completed', 'final', ['x']]);
});

test('marks a refused call as denied and an errored one as failed', () => {
  const start = (id: string): AgentEvent => ({ ...base(), type: 'tool-started', toolCallId: id, tool: 't', args: {}, environment: 'sandbox', sandbox });
  const fail = (id: string, denied: boolean): AgentEvent => ({
    ...base(), type: 'tool-failed', toolCallId: id, tool: 't', environment: 'sandbox', sandbox, error: 'no', denied, durationMs: 1,
  });
  const view = reduce([start('a'), start('b'), fail('a', true), fail('b', false)]);
  assert.deepEqual(view.timeline.map((entry) => entry.status), ['denied', 'failed']);
});

test('shows "Approve for me" reviewing, approving and escalating', () => {
  const start: AgentEvent = { ...base(), type: 'tool-started', toolCallId: 'c', tool: 't', args: {}, environment: 'sandbox', sandbox };
  const review = (phase: 'started' | 'approved' | 'escalated'): AgentEvent =>
    ({ ...base(), type: 'approval-reviewed', toolCallId: 'c', approvalId: 'a', phase, reason: 'in scope' });
  assert.equal(reduce([start, review('started')]).timeline[0]?.status, 'reviewing');
  assert.equal(reduce([start, review('approved')]).timeline[0]?.reviewNote, 'Approved for you: in scope');
  assert.equal(reduce([start, review('escalated')]).timeline[0]?.status, 'awaiting-approval');
});

test('ending a run clears approvals and leaves nothing running', () => {
  const running = reduce([
    { ...base(), type: 'run-started' },
    { ...base(), type: 'tool-started', toolCallId: 'c', tool: 't', args: {}, environment: 'sandbox', sandbox },
    { ...base(), type: 'approval-required', toolCallId: 'c', approval },
    { ...base(), type: 'text-delta', channel: 'text', delta: 'half' },
  ]);
  const cancelled = applyEvent(running, { ...base(), type: 'run-cancelled' });
  assert.deepEqual([cancelled.running, cancelled.approvals.length, cancelled.streaming, cancelled.timeline[0]?.status],
    [false, 0, '', 'failed']);

  const failed = applyEvent(running, { ...base(), type: 'run-failed', error: 'the owner must authorize this workspace before a paired client can use it' });
  assert.match(failed.error, /has not been opened on the host/);

  const summary = { filesChanged: ['x'], commandsRun: [], toolCalls: 2, failures: [], hostActions: [], approvalsRequested: 1, text: '' };
  const completed = applyEvent(running, { ...base(), type: 'run-completed', summary });
  assert.equal(completed.running, false);
  assert.equal(summaryLine(completed.summary!), '2 tool calls · 1 file changed · 1 approval');
});

test('restores a stored session and interleaves messages with tool calls by time', () => {
  const view = viewFromDetail({
    session: {} as never,
    messages: [
      { payload: { role: 'user', text: 'go', timestamp: '2026-01-01T00:00:01Z' } },
      { payload: { role: 'assistant', text: '', toolCalls: [{ id: 'c', name: 'read_file', arguments: {} }], timestamp: '2026-01-01T00:00:02Z' } },
      { payload: { role: 'tool', toolCallId: 'c', tool: 'read_file', output: 'o', isError: false, timestamp: '2026-01-01T00:00:03Z' } },
      { payload: { role: 'assistant', text: 'Done', timestamp: '2026-01-01T00:00:04Z' } },
    ],
    tool_executions: [{
      id: 'e', session_id: 's', tool_call_id: 'c', tool: 'read_file', arguments: { path: 'a.txt' }, environment: 'sandbox',
      risk: 'read', status: 'completed', output_preview: 'o', truncated: false, created_at: '2026-01-01 00:00:02',
    }, {
      id: 'e2', session_id: 's', tool: 'run_command', arguments: {}, environment: 'host', risk: 'execute',
      status: 'error', truncated: false, created_at: '2026-01-01 00:00:05',
    }],
    pending_approvals: [approval],
  });
  assert.deepEqual(transcriptRows(view).map((row) => row.kind === 'tool' ? `${row.entry.tool}:${row.entry.status}` : row.message.role),
    ['user', 'read_file:completed', 'assistant', 'run_command:failed']);
  assert.equal(view.timeline[1]?.environment, 'host');
  assert.equal(view.approvals.length, 1);
});

test('a stored tool call is listed after the message that made it, even when its time was cut to the second', () => {
  const view = viewFromDetail({
    session: {} as never,
    messages: [
      { payload: { role: 'user', text: 'go', timestamp: '2026-01-01T00:00:02.400Z' } },
      { payload: { role: 'assistant', text: 'Reading.', toolCalls: [{ id: 'c', name: 'read_file', arguments: {} }], timestamp: '2026-01-01T00:00:02.700Z' } },
      { payload: { role: 'tool', toolCallId: 'c', tool: 'read_file', output: 'o', isError: false, timestamp: '2026-01-01T00:00:02.800Z' } },
      { payload: { role: 'assistant', text: 'Done', timestamp: '2026-01-01T00:00:02.900Z' } },
    ],
    tool_executions: [{
      id: 'e', session_id: 's', tool_call_id: 'c', tool: 'read_file', arguments: {}, environment: 'sandbox',
      risk: 'read', status: 'completed', output_preview: 'o', truncated: false, created_at: '2026-01-01 00:00:02',
    }],
    pending_approvals: [],
  });
  assert.deepEqual(transcriptRows(view).map((row) => row.kind === 'tool' ? row.entry.tool : row.message.role === 'tool' ? '' : row.message.text),
    ['go', 'Reading.', 'read_file', 'Done']);
});

test('offers a paired device only the modes it may use', () => {
  const sandboxOf = (sandboxed_execution: boolean) => ({ backend: 'b', isolated: sandboxed_execution, sandboxed_execution, detail: '' });
  const all = ['ask', 'approve-for-me', 'sandbox-only', 'skip-permissions'] as const;
  assert.deepEqual(permissionModesFor({ sandbox: sandboxOf(true), permission_modes: [...all] }),
    ['ask', 'approve-for-me', 'sandbox-only']);
  assert.deepEqual(permissionModesFor({ sandbox: sandboxOf(false), permission_modes: [...all] }), ['ask', 'approve-for-me']);
  assert.deepEqual(permissionModesFor({ sandbox: sandboxOf(true), permission_modes: ['ask'] }), ['ask']);
  assert.ok(!permissionModesFor(null).includes('skip-permissions'));
});

test('lists power tools only when the host enabled them for the session', () => {
  const tool = (name: string, power_tool = false) => ({ name, label: name, description: '', risk: 'read', needs_workspace: true, power_tool });
  const catalog = [tool('read_file'), tool('browser', true)];
  assert.deepEqual(selectableTools(catalog, null).map(({ name }) => name), ['read_file']);
  assert.deepEqual(selectableTools(catalog, ['browser']).map(({ name }) => name), ['read_file', 'browser']);
});

test('explains host-only refusals and previews tool arguments', () => {
  assert.match(agentErrorText('paired clients cannot exercise local desktop authority'), /from the host itself/);
  assert.equal(agentErrorText('Something else.'), 'Something else.');
  assert.equal(argsPreview('run_command', { command: 'ls' }), 'ls');
  assert.equal(argsPreview('move', { from: 'a', to: 'b' }), 'a → b');
  assert.equal(argsPreview('list', {}), 'list');
});

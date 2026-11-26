import assert from 'node:assert/strict';
import { test } from 'node:test';
import { DaemonError, UnreachableError } from '../api.ts';
import { agentApi, type AgentFetch } from './rest.ts';
import type { AgentApproval } from './state.ts';

const credential = { origin: 'https://brazier.example', instanceId: 'host', apiKey: 'paired-key' };
type Call = { url: string; init: Parameters<AgentFetch>[1] };

function host(status: number, body: unknown, calls: Call[] = []): AgentFetch {
  return async (url, init) => {
    calls.push({ url, init });
    return { ok: status < 400, status, json: async () => body };
  };
}

test('creates a session without ever claiming the host-only elevated permissions', async () => {
  const calls: Call[] = [];
  const api = agentApi(host(200, { id: 's' }, calls), credential);
  await api.createSession({ title: 'T', workspace_path: '/w', model: 'm', permission_mode: 'ask', enabled_tools: ['read_file'] });
  const [{ url, init }] = calls as [Call];
  assert.equal(url, 'https://brazier.example/api/v1/agent/sessions');
  assert.equal(init.method, 'POST');
  assert.equal(init.headers?.authorization, 'Bearer paired-key');
  assert.deepEqual([init.credentials, init.redirect], ['omit', 'error']);
  assert.deepEqual(JSON.parse(init.body!), {
    title: 'T', workspace_path: '/w', model: 'm', permission_mode: 'ask', enabled_tools: ['read_file'],
    permission_settings: { auto_approve_sandboxed_actions: true, auto_approve_host_actions: false },
  });
});

test('decides an approval for the host that showed it', async () => {
  const calls: Call[] = [];
  const location = { kind: 'daemon', daemon_instance_id: 'i', daemon_display_name: 'Host', platform: 'macos', arch: 'arm64' };
  await agentApi(host(200, {}, calls), credential)
    .decide({ id: 'a/1', execution_location: location } as AgentApproval, 'approve', 'session');
  assert.equal(calls[0]?.url, 'https://brazier.example/api/v1/agent/approvals/a%2F1');
  assert.deepEqual(JSON.parse(calls[0]!.init.body!), { decision: 'approve', scope: 'session', expected_execution_location: location });
});

test('unwraps lists and sends no body on reads and deletes', async () => {
  const calls: Call[] = [];
  assert.deepEqual(await agentApi(host(200, { data: [{ id: 's' }] }, calls), credential).sessions(), [{ id: 's' }]);
  assert.equal(await agentApi(host(204, null, calls), credential).deleteSession('s'), undefined);
  assert.deepEqual(calls.map(({ init }) => [init.method, init.body]), [['GET', undefined], ['DELETE', undefined]]);
});

test('says what to do when the host refuses a folder, and keeps other refusals verbatim', async () => {
  const refused = agentApi(host(403, { error: { message: 'the owner must authorize this workspace before a paired client can use it' } }), credential);
  await assert.rejects(refused.validateWorkspace('/w'), (error: DaemonError) =>
    error.status === 403 && /Open it once in Agent mode in Brazier on the host/.test(error.message));
  await assert.rejects(agentApi(host(400, { error: { message: 'workspace does not exist' } }), credential).validateWorkspace('/w'),
    /workspace does not exist/);
  await assert.rejects(agentApi(host(401, {}), credential).sessions(), (error: DaemonError) => error.status === 401);
  await assert.rejects(agentApi(async () => { throw new Error('offline'); }, credential).sessions(), UnreachableError);
});

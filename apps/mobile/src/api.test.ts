import assert from 'node:assert/strict';
import { test } from 'node:test';
import { encodePairingInvite, parsePairingInvite } from '../../../packages/pairing/src/invite.ts';
import {
  chatBranch, chatContext, claimInvite, consumeChatStream, daemonClient, DaemonError,
  pairingOrigin, parseCredential, previewInvite,
} from './api.ts';

const origin = 'https://brazier.example';
const until = Math.floor(Date.now() / 1000) + 300;
const credential = { origin, instanceId: 'known-host', apiKey: 'paired-key' };

test('previews both shared invite formats before claiming against the reviewed identity', async () => {
  const payload = encodePairingInvite({
    version: 1, origin, pairingId: 'invite', code: 'code', hostName: 'Host',
    expiresAt: until, scopes: ['inference'],
  });
  const invite = parsePairingInvite(payload);
  assert.ok(invite?.origin);
  assert.equal(parsePairingInvite(`Daemon URL: ${origin}\nPairing id: invite\nPairing code: code`)?.pairingId, 'invite');
  const input = { origin: invite.origin, pairingId: invite.pairingId, code: invite.code };
  const calls: string[] = [];
  const fetch: Parameters<typeof claimInvite>[0] = async (url, init) => {
    calls.push(url);
    assert.equal(new Headers(init.headers).has('authorization'), false);
    assert.equal(init.redirect, 'error');
    if (url.endsWith('/preview')) {
      assert.deepEqual(JSON.parse(init.body!), { code: 'code' });
      return Response.json({
        pairing: { id: 'invite', client_name: 'Phone', scopes: ['inference'], expires_at: until },
        daemon: { instance_id: 'known-host', display_name: 'Host' },
      });
    }
    assert.deepEqual(JSON.parse(init.body!), { code: 'code', expected_daemon_instance_id: 'known-host' });
    return Response.json({ api_key: credential.apiKey });
  };
  const preview = await previewInvite(fetch, input);
  assert.equal(preview.daemon.display_name, 'Host');
  assert.deepEqual(calls, [origin + '/api/v1/auth/pairings/invite/preview']);
  assert.deepEqual(await claimInvite(fetch, input, preview), credential);
});

test('checks a saved host with authenticated daemon info and distinguishes revoked access from a changed identity', async () => {
  let status = 200;
  let instance = credential.instanceId;
  const client = daemonClient(async (url, request) => {
    assert.equal(url, origin + '/api/v1/daemon/info');
    assert.equal(new Headers(request.headers).get('authorization'), 'Bearer paired-key');
    return Response.json({
      product: 'brazier', management_api: { major: 1 },
      daemon: { instance_id: instance, display_name: 'Host' },
      client: { scopes: ['inference'] },
    }, { status });
  }, credential);
  assert.deepEqual(await client.verify(), { displayName: 'Host', scopes: ['inference'] });
  status = 401;
  await assert.rejects(client.verify(), { status: 401, message: /Pair again/ });
  status = 200;
  instance = 'other-host';
  await assert.rejects(client.verify(), { status: 409, message: /identity changed/ });
});

test('rejects plaintext in production and only allows the exact Android emulator alias in debug mode', () => {
  const androidOrigin = 'http://10.0.2.2:7614';
  assert.throws(() => pairingOrigin(androidOrigin));
  assert.equal(pairingOrigin(androidOrigin, true), androidOrigin);
  assert.throws(() => pairingOrigin('http://10.0.2.2:7614/api', true));
  assert.throws(() => pairingOrigin('http://10.0.2.3:7614', true));
  assert.throws(() => parseCredential(JSON.stringify({ ...credential, origin: androidOrigin })));
  assert.deepEqual(parseCredential(JSON.stringify(credential)), credential);
});

test('prepares a selected text model for chat without management access', async () => {
  const client = daemonClient(async (url, init) => {
    if (url.endsWith('/v1/models')) return Response.json({ data: [
      { id: 'image', capabilities: { output_modalities: ['image'] } },
      { id: 'text', capabilities: { output_modalities: ['text'] } },
    ] });
    assert.equal(url, origin + '/api/v1/models/prepare');
    assert.deepEqual(JSON.parse(init.body!), { model_id: 'text', mode: 'chat' });
    return Response.json({ status: 'ready' });
  }, credential);
  assert.deepEqual((await client.models()).map(({ id }) => id), ['text']);
  assert.deepEqual(await client.prepare('text'), { status: 'ready' });
});

test('chat follows the current history branch and omits non-text content from completion context', () => {
  const history = chatBranch([
    { id: 'root', parent_id: null, role: 'user', content: 'question' },
    { id: 'earlier', parent_id: 'root', role: 'assistant', content: 'old branch' },
    { id: 'latest', parent_id: 'root', role: 'assistant', content: [{ type: 'text', text: 'answer' }, { type: 'image' }] },
    { id: 'cancelled', parent_id: 'latest', role: 'assistant', content: 'partial', status: 'cancelled' },
  ]);
  assert.deepEqual(chatContext(history), [
    { role: 'user', content: 'question' },
    { role: 'assistant', content: 'answer' },
  ]);
});

test('streams POST chat with split SSE frames and UTF-8 bytes', async () => {
  const bytes = new TextEncoder().encode(
    'event: load\r\ndata: {"phase":"start"}\r\n\r\ndata: {"choices":[{"delta":{"content":"hé"}}]}\n\n' +
    'data: {"choices":[{"delta":{"content":"llo"}}]}\n\ndata: [DONE]\n\n',
  );
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
      controller.close();
    },
  });
  const tokens: string[] = [];
  const controller = new AbortController();
  const messages = [{ role: 'user', content: 'say hello' }];
  const client = daemonClient(async (url, init) => {
    assert.equal(url, origin + '/v1/chat/completions');
    assert.equal(init.method, 'POST');
    assert.equal(init.signal, controller.signal);
    assert.deepEqual(JSON.parse(init.body!), { model: 'small', messages, stream: true });
    return new Response(body);
  }, credential);
  assert.equal(await client.chat('small', messages, controller.signal, (token) => tokens.push(token)), 'héllo');
  assert.deepEqual(tokens, ['hé', 'llo']);
});

test('aborting releases a waiting response stream without yielding a finished answer', async () => {
  let cancelled = false;
  const stream = new ReadableStream<Uint8Array>({ cancel() { cancelled = true; } });
  const controller = new AbortController();
  const pending = consumeChatStream(stream, controller.signal, () => assert.fail('Unexpected token'));
  controller.abort();
  await assert.rejects(pending, { message: /Chat stopped/ });
  assert.ok(cancelled);
  assert.equal(stream.locked, false);
});

test('releases a completed native stream without cancelling its pending close', async () => {
  let cancelled = false;
  let source!: ReadableStreamDefaultController<Uint8Array>;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      source = controller;
      controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
    },
    cancel() { cancelled = true; },
  });
  await consumeChatStream(stream, new AbortController().signal, () => {});
  assert.equal(stream.locked, false);
  assert.equal(cancelled, false);
  source.close();
});

test('an SSE error is not silently treated as a completed answer', async () => {
  const stream = new Response('data: {"error":{"message":"runtime failed"}}\n\ndata: [DONE]\n\n').body!;
  await assert.rejects(consumeChatStream(stream, new AbortController().signal, () => {}),
    (error) => error instanceof DaemonError && error.message === 'runtime failed');
});

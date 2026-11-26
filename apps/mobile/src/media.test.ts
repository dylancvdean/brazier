import assert from 'node:assert/strict';
import test from 'node:test';
import type { Message, ToolCall } from './api.ts';
import {
  addHistory, blobFileName, entryDetails, generateBody, generatedMediaMessage, generationModels, generationProgress,
  historyEntry, isCancellation, maxHistory, mediaByMessage, mediaClient, messageBlobs, parseHistory, toolCallMedia,
  type GenerateForm, type HistoryEntry,
} from './media.ts';

const sha = (digit: string) => digit.repeat(64);
const form: GenerateForm = {
  modality: 'image', modelId: 'sdcpp-image:flux', prompt: ' a fox ', negative: ' ', width: 512, height: 768,
  steps: '', seed: '', frames: '',
};
const part = (sha256: string, call_id?: string, mime_type = 'image/png') =>
  ({ type: 'brazier_blob', brazier_blob: { sha256, mime_type, name: 'out.png', ...(call_id ? { call_id } : {}) } });
const message = (id: string, role: string, content: unknown, extra: Partial<Message> = {}): Message =>
  ({ id, parent_id: null, role, content, ...extra });

test('a form becomes a request that leaves blank settings to the host', () => {
  assert.deepEqual(generateBody(form), { prompt: 'a fox', model_id: 'sdcpp-image:flux', width: 512, height: 768 });
  assert.deepEqual(generateBody({ ...form, negative: 'blur', steps: '20', seed: '0', frames: '33' }),
    { prompt: 'a fox', model_id: 'sdcpp-image:flux', width: 512, height: 768, negative_prompt: 'blur', steps: 20, seed: 0 });
  assert.equal((generateBody({ ...form, modality: 'video', frames: '33' }) as { video_frames: number }).video_frames, 33);
});

test('a form that cannot be sent says why', () => {
  assert.equal(generateBody({ ...form, modelId: '' }), 'Choose an image model.');
  assert.equal(generateBody({ ...form, prompt: '  ' }), 'Enter a prompt.');
  assert.match(generateBody({ ...form, steps: '0' }) as string, /^Steps must be/);
  assert.match(generateBody({ ...form, steps: '2.5' }) as string, /^Steps must be/);
  assert.match(generateBody({ ...form, seed: '-1' }) as string, /^Seed must be/);
  assert.match(generateBody({ ...form, modality: 'video', frames: 'many' }) as string, /^Frames must be/);
});

test('generation models are split by modality', () => {
  const models = [{ id: 'sdcpp-image:a' }, { id: 'sdcpp-video:b' }, { id: 'llama:c' }];
  assert.deepEqual(generationModels(models, 'image'), [{ id: 'sdcpp-image:a' }]);
  assert.deepEqual(generationModels(models, 'video'), [{ id: 'sdcpp-video:b' }]);
});

const result = (sha256: string, metadata: unknown = {}) =>
  ({ blob: { sha256, mime_type: 'image/png', size_bytes: 2048 }, metadata, engine: 'stable-diffusion.cpp' });

test('a history entry records the settings the host reports using', () => {
  const body = generateBody(form) as Exclude<ReturnType<typeof generateBody>, string>;
  const entry = historyEntry('1', 'image', body, result(sha('a'), { seed: 42, steps: 8, width: 'x' }), '2026-01-01T00:00:00Z');
  assert.deepEqual([entry.seed, entry.steps, entry.width, entry.height], [42, 8, 512, 768]);
  assert.equal(entryDetails(entry), '512×768 · 8 steps · seed 42 · 2 KB');
  assert.equal(entry.frames, undefined);
  const video = generateBody({ ...form, modality: 'video', frames: '33' }) as typeof body;
  assert.equal(historyEntry('2', 'video', video, result(sha('b')), '').frames, 33);
});

test('history is newest first, capped, and survives a round trip but not corrupt data', () => {
  const body = generateBody(form) as Exclude<ReturnType<typeof generateBody>, string>;
  let history: HistoryEntry[] = [];
  for (let index = 0; index < maxHistory + 5; index += 1) {
    history = addHistory(history, historyEntry(String(index), 'image', body, result(sha('a')), ''));
  }
  assert.equal(history.length, maxHistory);
  assert.equal(history[0]!.id, String(maxHistory + 4));
  assert.deepEqual(parseHistory(JSON.stringify(history)), JSON.parse(JSON.stringify(history)));
  assert.deepEqual(parseHistory('{'), []);
  assert.deepEqual(parseHistory(null), []);
  assert.deepEqual(parseHistory(JSON.stringify([{ ...history[0], blob: { sha256: '../x', mime_type: 'image/png' } }, 3])), []);
});

test('progress shows steps once the host reports them', () => {
  const active = {
    id: '1', modality: 'image' as const, model_id: 'm', prompt: 'p', origin: 'user' as const,
    elapsed_secs: 75, timeout_secs: 600, current_step: 0, total_steps: 0,
  };
  assert.deepEqual(generationProgress(null), { fraction: null, label: 'Starting…' });
  assert.deepEqual(generationProgress(active), { fraction: null, label: 'Loading model · 1:15' });
  assert.deepEqual(generationProgress({ ...active, current_step: 5, total_steps: 20 }),
    { fraction: 0.25, label: 'Step 5 of 20 · 1:15' });
  assert.equal(generationProgress({ ...active, current_step: 30, total_steps: 20 }).fraction, 1);
});

test('a cancelled generation is told apart from a failure', () => {
  assert.equal(isCancellation(new Error('generation was stopped by the user')), true);
  assert.equal(isCancellation(new Error('out of memory')), false);
});

test('blob file names keep the extension and refuse anything but a hash', () => {
  assert.equal(blobFileName({ sha256: sha('a'), mime_type: 'video/mp4' }), `${sha('a')}.mp4`);
  assert.equal(blobFileName({ sha256: sha('a'), mime_type: 'IMAGE/PNG' }), `${sha('a')}.png`);
  assert.equal(blobFileName({ sha256: sha('a'), mime_type: 'application/x-thing' }), `${sha('a')}.bin`);
  assert.throws(() => blobFileName({ sha256: '../../etc/passwd', mime_type: 'image/png' }));
});

test('media parts are read from saved content and malformed ones dropped', () => {
  assert.deepEqual(messageBlobs('text'), []);
  assert.deepEqual(messageBlobs([
    { type: 'text', text: 'hi' }, part(sha('a'), 'call-1'), { type: 'brazier_blob', brazier_blob: { sha256: 'nope', mime_type: 'image/png' } },
    { type: 'brazier_blob' }, null,
  ]), [{ sha256: sha('a'), mime_type: 'image/png', name: 'out.png', call_id: 'call-1' }]);
});

const call = (call_id: string, media: ToolCall['media']): ToolCall =>
  ({ call_id, name: 'generate_image', arguments: '{}', output: '', is_error: false, media });

test('streamed tool calls yield each blob once, tied to its call', () => {
  const calls = [call('1', [{ sha256: sha('a'), mime_type: 'image/png' }]), call('2', [{ sha256: sha('a'), mime_type: 'image/png' }]),
    call('3', undefined), call('4', [{ sha256: sha('b'), mime_type: 'video/mp4', name: 'clip.mp4' }])];
  assert.deepEqual(toolCallMedia(calls), [
    { sha256: sha('a'), mime_type: 'image/png', call_id: '1' },
    { sha256: sha('b'), mime_type: 'video/mp4', name: 'clip.mp4', call_id: '4' },
  ]);
  assert.equal(generatedMediaMessage([call('3', [])]), null);
  const saved = generatedMediaMessage(calls)!;
  assert.deepEqual(saved.metadata, { generated_media_display: true });
  assert.deepEqual(saved.content.map(({ brazier_blob }) => brazier_blob.name), ['generated-image-1.png', 'clip.mp4']);
  // What is saved reads back as the same media.
  assert.deepEqual(messageBlobs(saved.content).map(({ sha256, call_id }) => [sha256, call_id]), [[sha('a'), '1'], [sha('b'), '4']]);
});

test('saved media is shown once, under the message whose tool call made it', () => {
  // The shape the desktop saves: tool-calling assistant, tool result, answer, then a media-only message.
  const messages = [
    message('u1', 'user', 'draw a fox'),
    message('a1', 'assistant', '', { tool_calls: [{ id: 'call-1' }] }),
    message('t1', 'tool', [{ type: 'text', text: 'done' }, part(sha('a'))], { tool_call_id: 'call-1' }),
    message('a2', 'assistant', 'Here it is.'),
    message('m1', 'assistant', [part(sha('a'), 'call-1')]),
    message('u2', 'user', [{ type: 'text', text: 'like this' }, part(sha('c'))]),
    message('a3', 'assistant', [part(sha('a'))]),
  ];
  const { media, moved } = mediaByMessage(messages);
  assert.deepEqual([...media.keys()], ['a1', 'u2', 'a3']);
  assert.deepEqual(media.get('a1')!.map(({ sha256 }) => sha256), [sha('a')]);
  assert.deepEqual([...moved], ['t1', 'm1']);
});

test('media the host hands back to a vision model is not shown as an upload', () => {
  const { media, moved } = mediaByMessage([
    message('u1', 'user', 'draw a fox'),
    message('a1', 'assistant', '', { tool_calls: [{ id: 'call-1' }] }),
    message('t1', 'tool', 'done', { tool_call_id: 'call-1' }),
    message('c1', 'user', [{ type: 'text', text: 'The requested media was generated.' }, part(sha('a'))]),
    message('a2', 'assistant', 'Here is your fox.'),
    message('m1', 'assistant', [part(sha('a'), 'call-1')]),
  ]);
  assert.deepEqual([...media], [['a1', [{ sha256: sha('a'), mime_type: 'image/png', name: 'out.png' }]]]);
  assert.deepEqual([...moved], ['c1', 'm1']);
});

test('an upload sent right after a tool result stays an upload', () => {
  const { media, moved } = mediaByMessage([
    message('u1', 'user', 'draw a fox'),
    message('a1', 'assistant', '', { tool_calls: [{ id: 'call-1' }] }),
    message('t1', 'tool', 'done', { tool_call_id: 'call-1' }),
    message('u2', 'user', [{ type: 'text', text: 'What is this?' }, part(sha('b'))]),
  ]);
  assert.deepEqual([...media.keys()], ['u2']);
  assert.equal(moved.size, 0);
});

test('a media-only message with no tool call shows its own media', () => {
  const { media, moved } = mediaByMessage([message('u', 'user', 'hi'), message('m', 'assistant', [part(sha('a'), 'gone')])]);
  assert.deepEqual([...media.keys()], ['m']);
  assert.equal(moved.size, 0);
});

test('the media client sends the key, validates results and refuses odd blob ids', async () => {
  const seen: { url: string; method?: string; auth: string | null; body?: string }[] = [];
  let reply: unknown = {};
  const fetch = async (url: string, init: { method?: string; headers?: HeadersInit; body?: string }) => {
    seen.push({ url, method: init.method, auth: new Headers(init.headers).get('authorization'), body: init.body });
    return new Response(JSON.stringify(reply));
  };
  const client = mediaClient(fetch, { origin: 'https://host:1', instanceId: 'i', apiKey: 'key' });
  reply = result(sha('a'));
  assert.equal((await client.generate('video', { prompt: 'p', model_id: 'm', width: 1, height: 1 })).blob.sha256, sha('a'));
  assert.deepEqual(seen[0], {
    url: 'https://host:1/api/v1/generate/video', method: 'POST', auth: 'Bearer key',
    body: '{"prompt":"p","model_id":"m","width":1,"height":1}',
  });
  reply = { blob: { sha256: 'x' } };
  await assert.rejects(client.generate('image', { prompt: 'p', model_id: 'm', width: 1, height: 1 }), /did not return/);
  reply = { active: null };
  assert.equal(await client.active(), null);
  reply = { cancelled: true };
  assert.equal(await client.cancel(), true);
  assert.equal(seen.at(-1)!.url, 'https://host:1/api/v1/generate/cancel');
  reply = { data: [{ id: 'sdcpp-image:a' }, { id: 'chat' }] };
  assert.deepEqual(await client.models(), [{ id: 'sdcpp-image:a' }]);
  await assert.rejects(client.blob('../secrets'), /Invalid media/);
  await client.blob(sha('a'));
  assert.equal(seen.at(-1)!.url, `https://host:1/api/v1/blobs/${sha('a')}`);
});

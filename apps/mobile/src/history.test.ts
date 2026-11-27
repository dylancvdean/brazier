import assert from 'node:assert/strict';
import { test } from 'node:test';
import { conversationDay } from './conversationDates.ts';
import {
  dayLabel, historyClient, historyDays, historyStore, hostHistoryEntry, mergeHistory, searchHistory, type HistoryRemote,
} from './history.ts';
import { entryDetails, historyEntry, maxHistory, parseHistory, type HistoryEntry } from './media.ts';

const sha = (letter: string) => letter.repeat(64);
const entry = (id: string, change: Partial<HistoryEntry> = {}): HistoryEntry => ({
  id, prompt: `prompt ${id}`, negativePrompt: '', modality: 'image', modelId: 'sdcpp-image:flux', width: 512, height: 512,
  createdAt: '2026-03-10T12:00:00.000Z', blob: { sha256: sha('a'), mime_type: 'image/png', size_bytes: 2048 }, ...change,
});
const hostEntry = (id: string, change: Partial<HistoryEntry> = {}) => entry(id, { hostId: id, ...change });

function memoryFile(initial: string | null = null) {
  const state = { text: initial, writes: 0 };
  return {
    state,
    file: {
      read: async () => state.text,
      write: async (text: string) => { state.text = text; state.writes += 1; },
    },
  };
}

test('a host history entry carries the prompt, model, settings and date the host recorded', () => {
  const parsed = hostHistoryEntry({
    id: 'h1', modality: 'video', created_at: 1_773_144_000,
    blob: { sha256: sha('b'), mime_type: 'video/mp4', size_bytes: 3_500_000 },
    metadata: {
      model_id: 'sdcpp-video:wan', prompt: 'a tide pool', negative_prompt: 'blur', width: 832, height: 480,
      steps: 20, seed: 9, cfg_scale: 6, video_frames: 33,
    },
  });
  assert.deepEqual(parsed, {
    id: 'h1', hostId: 'h1', modality: 'video', prompt: 'a tide pool', negativePrompt: 'blur', modelId: 'sdcpp-video:wan',
    width: 832, height: 480, steps: 20, seed: 9, frames: 33, createdAt: '2026-03-10T12:00:00.000Z',
    blob: { sha256: sha('b'), mime_type: 'video/mp4', size_bytes: 3_500_000 },
  });
  assert.equal(entryDetails(parsed!), '832×480 · 20 steps · seed 9 · 33 frames · 3.3 MB');
  // What it is saved as on the phone reads back the same.
  assert.deepEqual(parseHistory(JSON.stringify([parsed])), [parsed]);
});

test('a host history entry without usable media is skipped, and missing metadata is left blank', () => {
  for (const raw of [null, 'x', {}, { id: 'h', modality: 'audio', blob: { sha256: sha('b'), mime_type: 'audio/wav' } },
    { id: 'h', modality: 'image', blob: { sha256: 'short', mime_type: 'image/png' } },
    { id: '', modality: 'image', blob: { sha256: sha('b'), mime_type: 'image/png' } }]) {
    assert.equal(hostHistoryEntry(raw), null);
  }
  const bare = hostHistoryEntry({ id: 'h', modality: 'image', blob: { sha256: sha('b'), mime_type: 'image/png' }, metadata: null })!;
  assert.deepEqual([bare.prompt, bare.modelId, bare.width, bare.steps, bare.blob.size_bytes], ['', '', 0, undefined, 0]);
  assert.equal(entryDetails(bare), '');
});

test('a result from a host that keeps a history is listed under the host’s id', () => {
  const body = { prompt: 'p', model_id: 'sdcpp-image:flux', width: 512, height: 512 };
  const blob = { sha256: sha('c'), mime_type: 'image/png', size_bytes: 1 };
  const kept = historyEntry('fallback', 'image', body, { blob, metadata: {}, engine: 'e', history_id: 'h9' }, 'now');
  assert.deepEqual([kept.id, kept.hostId], ['h9', 'h9']);
  const old = historyEntry('fallback', 'image', body, { blob, metadata: {}, engine: 'e' }, 'now');
  assert.deepEqual([old.id, 'hostId' in old], ['fallback', false]);
});

test('the host’s list wins; the phone adds only what the host never listed', () => {
  const host = [
    hostEntry('h2', { createdAt: '2026-03-12T00:00:00.000Z', blob: { sha256: sha('b'), mime_type: 'image/png', size_bytes: 1 } }),
    hostEntry('h1', { createdAt: '2026-03-10T00:00:00.000Z' }),
  ];
  const phone = [
    hostEntry('h2', { prompt: 'stale copy' }),
    // Deleted on the host from another device.
    hostEntry('h0', { blob: { sha256: sha('d'), mime_type: 'image/png', size_bytes: 1 } }),
    // Made before the host kept a history, and still only known here.
    entry('p1', { createdAt: '2026-03-11T00:00:00.000Z', blob: { sha256: sha('e'), mime_type: 'image/png', size_bytes: 1 } }),
    // An earlier run that made the same file as h1: another generation, so it stays.
    entry('p2', { createdAt: '2026-03-09T00:00:00.000Z' }),
  ];
  assert.deepEqual(mergeHistory(host, phone).map(({ id }) => id), ['h2', 'p1', 'h1', 'p2']);
  assert.equal(mergeHistory(host, phone)[0]!.prompt, 'prompt h2');
  assert.deepEqual(mergeHistory([], phone).map(({ id }) => id), ['p1', 'p2']);
});

test('history is searched by every word, across prompt, negative prompt and model, within one kind', () => {
  const history = [
    entry('1', { prompt: 'A red Lighthouse at dusk' }),
    entry('2', { prompt: 'harbour', negativePrompt: 'lighthouse', modality: 'video', modelId: 'sdcpp-video:wan' }),
    entry('3', { prompt: 'mountain', modelId: 'dir-1' }),
  ];
  const ids = (...args: Parameters<typeof searchHistory>) => searchHistory(...args).map(({ id }) => id);
  assert.deepEqual(ids(history, ''), ['1', '2', '3']);
  assert.deepEqual(ids(history, '  LIGHTHOUSE '), ['1', '2']);
  assert.deepEqual(ids(history, 'dusk red'), ['1']);
  assert.deepEqual(ids(history, 'lighthouse', 'video'), ['2']);
  assert.deepEqual(ids(history, '', 'image'), ['1', '3']);
  assert.deepEqual(ids(history, 'wan'), ['2']);
  // This device names models by a label, not by their id.
  assert.deepEqual(ids(history, 'dreamshaper', 'all', (id) => id === 'dir-1' ? 'DreamShaper 8' : id), ['3']);
  assert.deepEqual(ids(history, 'nothing'), []);
});

test('entries are grouped by local day, newest first', () => {
  const now = new Date(2026, 2, 12, 9, 0);
  const at = (day: number, hour: number) => new Date(2026, 2, day, hour).toISOString();
  assert.equal(dayLabel(at(12, 1), now), 'Today');
  assert.equal(dayLabel(at(11, 23), now), 'Yesterday');
  assert.equal(dayLabel('nonsense', now), 'Undated');
  assert.match(dayLabel(at(3, 12), now), /3/);
  assert.doesNotMatch(dayLabel(at(3, 12), now), /2026/);
  assert.match(dayLabel(new Date(2025, 11, 30).toISOString(), now), /2025/);
  const days = historyDays([entry('a', { createdAt: at(12, 8) }), entry('b', { createdAt: at(12, 1) }),
    entry('c', { createdAt: at(11, 20) })], now);
  assert.deepEqual(days.map(({ day, entries }) => [day, entries.map(({ id }) => id)]),
    [['Today', ['a', 'b']], ['Yesterday', ['c']]]);
});

test('a conversation’s day reads a host’s UTC timestamp and this device’s ISO one', () => {
  const now = new Date('2026-03-12T12:00:00Z');
  assert.equal(conversationDay('2026-03-12 11:00:00', now), dayLabel('2026-03-12T11:00:00Z', now));
  assert.equal(conversationDay('2026-03-12T11:00:00.000Z', now), dayLabel('2026-03-12T11:00:00Z', now));
  assert.equal(conversationDay(undefined, now), '');
  assert.equal(conversationDay('soon', now), '');
});

test('this device’s history survives a restart, and removing an entry is saved', async () => {
  const { file, state } = memoryFile();
  const first = historyStore(file);
  assert.deepEqual(await first.list(), []);
  await first.add(entry('1'));
  assert.deepEqual((await first.add(entry('2'))).map(({ id }) => id), ['2', '1']);

  const reopened = historyStore(file);
  assert.deepEqual((await reopened.list()).map(({ id }) => id), ['2', '1']);
  assert.deepEqual((await reopened.remove(entry('2'))).map(({ id }) => id), ['1']);
  assert.deepEqual(parseHistory(state.text).map(({ id }) => id), ['1']);
});

test('changes made at once are all kept', async () => {
  const { file } = memoryFile();
  const store = historyStore(file);
  await Promise.all(['1', '2', '3', '4'].map((id) => store.add(entry(id))));
  assert.deepEqual((await store.list()).map(({ id }) => id).sort(), ['1', '2', '3', '4']);
});

test('the history holds the newest entries up to its limit', async () => {
  const { file } = memoryFile(JSON.stringify(Array.from({ length: maxHistory }, (_, index) => entry(`old-${index}`))));
  const saved = await historyStore(file).add(entry('new'));
  assert.equal(saved.length, maxHistory);
  assert.equal(saved[0]!.id, 'new');
  assert.equal(saved.at(-1)!.id, `old-${maxHistory - 2}`);
});

test('a result is still listed when the history cannot be written', async () => {
  const store = historyStore({ read: async () => null, write: async () => { throw new Error('disk full'); } });
  assert.deepEqual((await store.add(entry('1'))).map(({ id }) => id), ['1']);
});

test('a paired host’s history shows what any client made, and the saved copy when the host is away', async () => {
  const { file, state } = memoryFile(JSON.stringify([entry('p1', { blob: { sha256: sha('e'), mime_type: 'image/png', size_bytes: 1 } })]));
  let host: HistoryEntry[] | null | Error = [hostEntry('h1', { createdAt: '2026-03-11T00:00:00.000Z' })];
  const removed: string[] = [];
  const remote: HistoryRemote = {
    list: async () => { if (host instanceof Error) throw host; return host; },
    remove: async (id) => { if (host instanceof Error) throw host; removed.push(id); },
  };
  const store = historyStore(file, remote);
  assert.deepEqual((await store.list()).map(({ id }) => id), ['h1', 'p1']);
  assert.equal(state.writes, 1);
  // Nothing new: nothing written.
  await store.list();
  assert.equal(state.writes, 1);

  host = new Error('Network request failed');
  assert.deepEqual((await historyStore(file, remote).list()).map(({ id }) => id), ['h1', 'p1']);
  // The host must agree before its entry goes; the phone's own entry needs no host.
  await assert.rejects(store.remove(hostEntry('h1')), /Network request failed/);
  assert.deepEqual((await store.remove(entry('p1'))).map(({ id }) => id), ['h1']);

  host = [];
  assert.deepEqual((await store.remove(hostEntry('h1'))).map(({ id }) => id), []);
  assert.deepEqual(removed, ['h1']);
  assert.equal(state.text, '[]');
});

test('a host too old to keep a history leaves the phone’s list as it is', async () => {
  const { file, state } = memoryFile(JSON.stringify([entry('p1')]));
  const store = historyStore(file, { list: async () => null, remove: async () => undefined });
  assert.deepEqual((await store.list()).map(({ id }) => id), ['p1']);
  assert.equal(state.writes, 0);
});

test('the history client lists and deletes on the host with the pairing key', async () => {
  const calls: string[] = [];
  const credential = { origin: 'http://host:1', instanceId: 'i', apiKey: 'key' };
  const client = historyClient(async (url, init) => {
    calls.push(`${init?.method ?? 'GET'} ${url} ${new Headers(init?.headers).get('authorization')}`);
    if (init?.method === 'DELETE') return Response.json({ error: { message: 'generation not found' } }, { status: 404 });
    return Response.json({ data: [
      { id: 'h1', modality: 'image', created_at: 5, blob: { sha256: sha('a'), mime_type: 'image/png', size_bytes: 3 }, metadata: { prompt: 'p' } },
      { id: 'broken' },
    ] });
  }, credential);
  assert.deepEqual((await client.list())!.map(({ id, prompt }) => [id, prompt]), [['h1', 'p']]);
  // Already gone on the host: the deletion is done.
  await client.remove('h 1');
  assert.deepEqual(calls, [
    'GET http://host:1/api/v1/generate/history Bearer key',
    'DELETE http://host:1/api/v1/generate/history/h%201 Bearer key',
  ]);

  const old = historyClient(async () => Response.json({ error: { message: 'Not found' } }, { status: 404 }), credential);
  assert.equal(await old.list(), null);
  const refused = historyClient(async () => Response.json({ error: { message: 'no' } }, { status: 401 }), credential);
  await assert.rejects(refused.list());
});

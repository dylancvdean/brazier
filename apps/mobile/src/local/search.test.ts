import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { Fetch } from './huggingface.ts';
import {
  debouncedSearch, formatAge, formatCount, mergeResults, namedRepository, parseResults, searchFormats, searchModels, searchUrl,
  type SearchResult,
} from './search.ts';

const reply = (body: unknown, link: string | null = null, status = 200) => ({
  ok: status >= 200 && status < 300, status, json: async () => body, headers: { get: (name: string) => name === 'link' ? link : null },
});
const gguf = (id: string, extra: object = {}) => ({
  id, downloads: 10, likes: 2, lastModified: '2026-01-01T00:00:00.000Z', pipeline_tag: 'text-generation', tags: ['gguf'],
  siblings: [{ rfilename: 'README.md' }, { rfilename: 'm-Q4_K_M.gguf' }, { rfilename: 'm-Q8_0.gguf' }], ...extra,
});
const result = (slug: string): SearchResult => ({ slug, format: 'gguf', downloads: 0, likes: 0, updated: null, files: 1 });

test('the search formats are the ones this device can run', () => {
  assert.deepEqual(searchFormats(['llama.cpp', 'sd.cpp']), ['gguf', 'image']);
  assert.deepEqual(searchFormats(['apple', 'mlx', 'llama.cpp']), ['gguf', 'mlx']);
  assert.deepEqual(searchFormats(['apple']), []);
});

test('a search link carries the query, format, sort and page size', () => {
  const url = new URL(searchUrl({ query: ' qwen 3 & more ', format: 'gguf', sort: 'trendingScore' }));
  assert.equal(url.origin + url.pathname, 'https://huggingface.co/api/models');
  assert.equal(url.searchParams.get('search'), 'qwen 3 & more');
  assert.equal(url.searchParams.get('filter'), 'gguf');
  assert.equal(url.searchParams.get('sort'), 'trendingScore');
  assert.equal(url.searchParams.get('direction'), '-1');
  assert.equal(url.searchParams.get('limit'), '20');
  assert.ok(url.searchParams.getAll('expand[]').includes('siblings'));
  assert.equal(new URL(searchUrl({ query: '', format: 'image', sort: 'downloads' })).searchParams.get('filter'), 'text-to-image');
  assert.equal(new URL(searchUrl({ query: '', format: 'mlx', sort: 'likes' })).searchParams.has('search'), false);
});

test('a query that names a repository is recognised', () => {
  assert.equal(namedRepository(' unsloth/Qwen3-4B-GGUF '), 'unsloth/Qwen3-4B-GGUF');
  assert.equal(namedRepository('https://huggingface.co/unsloth/Qwen3-4B-GGUF/tree/main'), 'unsloth/Qwen3-4B-GGUF');
  assert.equal(namedRepository('qwen 4b'), null);
  assert.equal(namedRepository(''), null);
});

test('results keep only public repositories with a file the format can load', () => {
  const results = parseResults([
    gguf('a/chat'),
    gguf('a/gated', { gated: 'auto' }),
    gguf('a/private', { private: true }),
    gguf('a/embedding', { pipeline_tag: 'feature-extraction' }),
    gguf('a/speech', { pipeline_tag: undefined, tags: ['gguf', 'text-to-speech'] }),
    gguf('a/empty', { siblings: [{ rfilename: 'README.md' }] }),
    gguf('a/projector-only', { siblings: [{ rfilename: 'mmproj-model-f16.gguf' }] }),
    gguf('not a slug'),
    { downloads: 1 },
    null,
  ], 'gguf');
  assert.deepEqual(results, [{
    slug: 'a/chat', format: 'gguf', downloads: 10, likes: 2, updated: '2026-01-01T00:00:00.000Z', files: 2,
  }]);
  assert.throws(() => parseResults({ error: 'x' }, 'gguf'), /unexpected/);
});

test('image results leave out families that need separate encoder files', () => {
  const image = (id: string, tags: string[] = []) =>
    ({ id, pipeline_tag: 'text-to-image', tags, siblings: [{ rfilename: 'model-Q8_0.gguf' }] });
  const slugs = parseResults([
    image('a/dreamshaper-8'), image('city96/FLUX.1-dev-gguf'), image('a/sd3.5-large'), image('a/model', ['base_model:x/Z-Image-Turbo']),
    image('a/sdxl-turbo'),
  ], 'image').map(({ slug }) => slug);
  assert.deepEqual(slugs, ['a/dreamshaper-8', 'a/sdxl-turbo']);
});

test('MLX results are whole repositories', () => {
  const files = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'].map((rfilename) => ({ rfilename }));
  const results = parseResults([
    { id: 'mlx-community/Qwen3-4B-4bit', siblings: files, tags: ['mlx'] },
    { id: 'mlx-community/whisper', siblings: files, pipeline_tag: 'automatic-speech-recognition' },
    { id: 'mlx-community/no-weights', siblings: [{ rfilename: 'config.json' }] },
  ], 'mlx');
  assert.deepEqual(results.map(({ slug, files: count }) => [slug, count]), [['mlx-community/Qwen3-4B-4bit', 1]]);
});

test('a search follows the next link and skips pages with nothing usable', async () => {
  const second = 'https://huggingface.co/api/models?cursor=2';
  const third = 'https://huggingface.co/api/models?cursor=3';
  const pages: Record<string, ReturnType<typeof reply>> = {
    [searchUrl({ query: 'q', format: 'gguf', sort: 'downloads' })]: reply([gguf('a/one')], `<${second}>; rel="next"`),
    [second]: reply([gguf('a/gated', { gated: true })], `<${third}>; rel="next"`),
    [third]: reply([gguf('a/three')], '<https://elsewhere.example/api/models?x=1>; rel="next"'),
  };
  const seen: string[] = [];
  const fetch: Fetch = async (url, init) => {
    seen.push(url);
    assert.equal(init?.headers?.accept, 'application/json');
    return pages[url];
  };
  const request = { query: 'q', format: 'gguf', sort: 'downloads' } as const;
  const first = await searchModels(fetch, request);
  assert.deepEqual([first.results.map(({ slug }) => slug), first.next], [['a/one'], second]);
  const rest = await searchModels(fetch, request, first.next);
  assert.deepEqual([rest.results.map(({ slug }) => slug), rest.next], [['a/three'], null]);
  assert.equal(seen.length, 3);
});

test('a search stops skipping empty pages after a few', async () => {
  let calls = 0;
  const fetch: Fetch = async () => { calls += 1; return reply([], `<https://huggingface.co/api/models?cursor=${calls}>; rel="next"`); };
  const page = await searchModels(fetch, { query: 'q', format: 'gguf', sort: 'downloads' });
  assert.deepEqual([page.results, calls], [[], 4]);
  assert.equal(page.next, 'https://huggingface.co/api/models?cursor=4');
});

test('search failures are explained', async () => {
  const request = { query: 'q', format: 'gguf', sort: 'downloads' } as const;
  await assert.rejects(searchModels(async () => { throw new Error('offline'); }, request), /Could not reach Hugging Face/);
  await assert.rejects(searchModels(async () => reply(null, null, 429), request), /limiting/);
  await assert.rejects(searchModels(async () => reply(null, null, 500), request), /HTTP 500/);
});

test('pages merge without repeating a repository', () => {
  assert.deepEqual(mergeResults([result('a/1'), result('a/2')], [result('a/2'), result('a/3'), result('a/3')]).map(({ slug }) => slug),
    ['a/1', 'a/2', 'a/3']);
});

test('counts and ages are short', () => {
  assert.deepEqual([0, 999, 1000, 1234, 15_600, 123_456, 1_234_567, 250_000_000, -1, NaN].map(formatCount),
    ['0', '999', '1k', '1.2k', '15.6k', '123k', '1.2M', '250M', '0', '0']);
  const now = Date.parse('2026-06-01T00:00:00Z');
  assert.deepEqual(['2026-06-01T00:00:00Z', '2026-05-20T00:00:00Z', '2026-03-01T00:00:00Z', '2024-05-01T00:00:00Z', 'x', null]
    .map((date) => formatAge(date, now)), ['today', '12 d ago', '3 mo ago', '2 y ago', '', '']);
});

function harness() {
  const pending: { run: () => void; ms: number }[] = [];
  const answers = new Map<string, { resolve(value: string): void; reject(error: unknown): void }>();
  const reports: { request: string; result?: string; error?: unknown }[] = [];
  const searched: string[] = [];
  const search = debouncedSearch<string, string>(
    (request) => new Promise((resolve, reject) => { searched.push(request); answers.set(request, { resolve, reject }); }),
    (outcome) => reports.push(outcome), 300,
    { set: (run, ms) => { const timer = { run, ms }; pending.push(timer); return timer; },
      clear: (timer) => { pending.splice(pending.indexOf(timer as typeof pending[number]), 1); } });
  const elapse = () => pending.splice(0).forEach(({ run }) => run());
  const settle = () => new Promise((resolve) => setImmediate(resolve));
  return { pending, answers, reports, searched, search, elapse, settle };
}

test('typing searches once, after the pause', async () => {
  const { pending, answers, reports, searched, search, elapse, settle } = harness();
  for (const text of ['q', 'qw', 'qwe']) search.request(text);
  assert.deepEqual([searched, pending.length, pending[0].ms], [[], 1, 300]);
  elapse();
  assert.deepEqual(searched, ['qwe']);
  answers.get('qwe')!.resolve('found');
  await settle();
  assert.deepEqual(reports, [{ request: 'qwe', result: 'found' }]);
});

test('an answer to an older search is dropped', async () => {
  const { answers, reports, searched, search, elapse, settle } = harness();
  search.request('old', true);
  search.request('new');
  elapse();
  assert.deepEqual(searched, ['old', 'new']);
  answers.get('new')!.resolve('newer');
  answers.get('old')!.resolve('stale');
  await settle();
  assert.deepEqual(reports, [{ request: 'new', result: 'newer' }]);
});

test('a failure is reported, and cancel drops what is pending or in flight', async () => {
  const { pending, answers, reports, searched, search, settle } = harness();
  const failure = new Error('offline');
  search.request('a', true);
  answers.get('a')!.reject(failure);
  await settle();
  assert.deepEqual(reports, [{ request: 'a', error: failure }]);
  search.request('b', true);
  search.request('c');
  search.cancel();
  answers.get('b')!.resolve('late');
  await settle();
  assert.deepEqual([pending.length, searched, reports.length], [0, ['a', 'b'], 1]);
});

test('speech repositories without a pipeline tag and image repositories are not chat results', () => {
  const results = parseResults([
    gguf('a/chat'),
    gguf('a/asr', { pipeline_tag: undefined, tags: ['mlx-audio', 'qwen3_asr', 'speech-to-text', 'asr'] }),
    gguf('a/sd-turbo', { pipeline_tag: undefined, tags: ['gguf', 'sd.cpp', 'stable-diffusion.cpp'] }),
  ], 'gguf');
  assert.deepEqual(results.map(({ slug }) => slug), ['a/chat']);
});

test('image results leave out add-ons and files that are only part of a model', () => {
  const image = (id: string, files: string[], tags: string[] = []) => ({
    id, pipeline_tag: 'text-to-image', tags, siblings: files.map((rfilename) => ({ rfilename })),
  });
  const results = parseResults([
    image('a/checkpoint', ['model.safetensors', 'model_unet.safetensors', 'model_lora.safetensors']),
    image('a/controlnet-depth', ['diffusion_pytorch_model.safetensors']),
    image('a/pose', ['OpenPoseXL2.safetensors'], ['ControlNet']),
    image('a/style-lora', ['style.safetensors'], ['lora']),
    image('a/unet', ['diffusion_pytorch_model.safetensors', 'model_unet.safetensors']),
    image('a/explorations', ['model.safetensors']),
  ], 'image');
  assert.deepEqual(results.map(({ slug, files }) => [slug, files]), [['a/checkpoint', 1], ['a/explorations', 1]]);
});

test('an image search leads with repositories made for stable-diffusion.cpp', async () => {
  const image = (id: string, extra: object = {}) => ({ id, pipeline_tag: 'text-to-image', siblings: [{ rfilename: 'm.gguf' }], ...extra });
  const request = { query: 'turbo', format: 'image', sort: 'downloads' } as const;
  const next = 'https://huggingface.co/api/models?cursor=2';
  const seen: string[] = [];
  const fetch: Fetch = async (url) => {
    seen.push(url);
    if (url.includes('filter=stable-diffusion.cpp')) return reply([image('a/made', { pipeline_tag: undefined }), image('a/both')]);
    return url === next ? reply([image('a/later')]) : reply([image('a/both'), image('a/tagged')], `<${next}>; rel="next"`);
  };
  const first = await searchModels(fetch, request);
  assert.deepEqual([first.results.map(({ slug }) => slug), first.next], [['a/made', 'a/both', 'a/tagged'], next]);
  assert.deepEqual((await searchModels(fetch, request, first.next)).results.map(({ slug }) => slug), ['a/later']);
  assert.equal(seen.length, 3);
});

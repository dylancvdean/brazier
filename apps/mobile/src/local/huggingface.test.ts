import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileUrl, formatBytes, isImageRepo, listRepoFiles, modelChoices, parseSlug, parseTree } from './huggingface.ts';

test('a slug is owner/name, typed or pasted as a link', () => {
  assert.equal(parseSlug(' unsloth/Qwen3-0.6B-GGUF '), 'unsloth/Qwen3-0.6B-GGUF');
  assert.equal(parseSlug('https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/tree/main?x=1'), 'unsloth/Qwen3-0.6B-GGUF');
  assert.equal(parseSlug('hf.co/mlx-community/SmolLM-135M-Instruct-4bit'), 'mlx-community/SmolLM-135M-Instruct-4bit');
  for (const bad of ['', 'model', 'a/b/c', '../x/y', 'a/..', 'a b/c', 'https://example.com/a/b']) {
    assert.throws(() => parseSlug(bad), /owner\/name/, bad);
  }
});

test('file links escape each path segment', () => {
  assert.equal(fileUrl('a/b', 'dir/my file#1.gguf'), 'https://huggingface.co/a/b/resolve/main/dir/my%20file%231.gguf');
});

test('the tree keeps files, with the real size of large files', () => {
  assert.deepEqual(parseTree([
    { type: 'directory', path: 'onnx', size: 0 },
    { type: 'file', path: 'config.json', size: 700 },
    { type: 'file', path: 'model.gguf', size: 135, lfs: { size: 4_000_000_000 } },
    { type: 'file', path: '../escape.gguf', size: 1 },
    { type: 'file', path: 'no-size.txt' },
  ]), [
    { path: 'config.json', size: 700 }, { path: 'model.gguf', size: 4_000_000_000 }, { path: 'no-size.txt', size: 0 },
  ]);
  assert.throws(() => parseTree({ error: 'x' }), /unexpected/);
});

const page = (status: number, body: unknown, link: string | null = null) => ({
  ok: status >= 200 && status < 300, status, json: async () => body, headers: { get: () => link },
});

test('listing follows pages on Hugging Face only', async () => {
  const seen: string[] = [];
  const files = await listRepoFiles(async (url) => {
    seen.push(url);
    return seen.length === 1
      ? page(200, [{ type: 'file', path: 'a.gguf', size: 1 }], '<https://huggingface.co/api/models/o/n/tree/main?cursor=2>; rel="next"')
      : page(200, [{ type: 'file', path: 'b.gguf', size: 2 }], '<https://elsewhere.example/next>; rel="next"');
  }, 'o/n');
  assert.deepEqual(files.map(({ path }) => path), ['a.gguf', 'b.gguf']);
  assert.deepEqual(seen, [
    'https://huggingface.co/api/models/o/n/tree/main?recursive=true',
    'https://huggingface.co/api/models/o/n/tree/main?cursor=2',
  ]);
});

test('listing explains a missing, gated or unreachable repository', async () => {
  await assert.rejects(listRepoFiles(async () => page(404, {}), 'o/n'), /No model named o\/n/);
  await assert.rejects(listRepoFiles(async () => page(401, {}), 'o/n'), /private or gated/);
  await assert.rejects(listRepoFiles(async () => page(500, {}), 'o/n'), /HTTP 500/);
  await assert.rejects(listRepoFiles(async () => { throw new Error('offline'); }, 'o/n'), /Could not reach Hugging Face/);
});

const gguf = [
  { path: 'README.md', size: 10 },
  { path: 'model-Q8_0.gguf', size: 800 },
  { path: 'model-Q4_K_M.gguf', size: 400 },
  { path: 'mmproj-model-f16.gguf', size: 50 },
  { path: 'MTP/mtp-model-Q8_0.gguf', size: 40 },
  { path: 'mtp-model.gguf', size: 40 },
  { path: 'big/model-F16-00001-of-00002.gguf', size: 900 },
  { path: 'big/model-F16-00002-of-00002.gguf', size: 300 },
  { path: 'partial-00001-of-00003.gguf', size: 5 },
];

test('GGUF files are choices for llama.cpp, smallest first, with split files as one', () => {
  const choices = modelChoices(gguf, ['llama.cpp', 'mlx']);
  assert.deepEqual(choices.map(({ name, size, runtime }) => [name, size, runtime]), [
    ['model-Q4_K_M.gguf', 400, 'llama.cpp'], ['model-Q8_0.gguf', 800, 'llama.cpp'], ['model-F16.gguf', 1200, 'llama.cpp'],
  ]);
  assert.equal(choices[2].entry, 'big/model-F16-00001-of-00002.gguf');
  assert.equal(choices[2].files.length, 2);
});

const mlx = [
  { path: 'config.json', size: 1 }, { path: 'model.safetensors', size: 100 }, { path: 'tokenizer.json', size: 5 },
  { path: 'chat_template.jinja', size: 2 }, { path: 'README.md', size: 9 }, { path: 'onnx/model.onnx', size: 500 },
];

test('an MLX repository is one choice, only where MLX can run', () => {
  const [choice] = modelChoices(mlx, ['llama.cpp', 'mlx']);
  assert.equal(choice.runtime, 'mlx');
  assert.equal(choice.size, 108);
  assert.deepEqual(choice.files.map(({ path }) => path), ['config.json', 'model.safetensors', 'tokenizer.json', 'chat_template.jinja']);
  assert.deepEqual(modelChoices(mlx, ['llama.cpp']), []);
  assert.deepEqual(modelChoices(gguf, ['mlx']), []);
});

test('sizes read in decimal units', () => {
  assert.deepEqual([0, 999, 1500, 484_000_000, 4_920_000_000].map(formatBytes), ['0 B', '999 B', '1.5 KB', '484 MB', '4.9 GB']);
});

test('a repository is an image repository by its Hugging Face tags', async () => {
  const info = (body: unknown, status = 200) => async (url: string) => {
    assert.equal(url, 'https://huggingface.co/api/models/o/n');
    return { ok: status === 200, status, json: async () => body, headers: { get: () => null } };
  };
  assert.equal(await isImageRepo(info({ pipeline_tag: 'text-to-image', tags: ['gguf'] }), 'o/n'), true);
  assert.equal(await isImageRepo(info({ tags: ['gguf', 'sd.cpp'] }), 'o/n'), true);
  assert.equal(await isImageRepo(info({ pipeline_tag: 'text-generation', tags: ['gguf', 'llama'] }), 'o/n'), false);
  assert.equal(await isImageRepo(info({}), 'o/n'), false);
  assert.equal(await isImageRepo(info({}, 500), 'o/n'), false);
  assert.equal(await isImageRepo(async () => { throw new Error('offline'); }, 'o/n'), false);
});

test('an image repository offers its single-file checkpoints to stable-diffusion.cpp only', () => {
  const files = [
    { path: 'sd_turbo-f16.gguf', size: 2_500 },
    { path: 'old/sd_turbo-q8_0.gguf', size: 2_000 },
    { path: 'v1-5-pruned-emaonly.safetensors', size: 4_000 },
    { path: 'unet/diffusion_pytorch_model.safetensors', size: 3_000 },
    { path: 'vae-ft-mse.safetensors', size: 300 },
    { path: 'big-00001-of-00002.gguf', size: 10 },
    { path: 'model_index.json', size: 1 },
  ];
  const all = ['llama.cpp', 'mlx', 'sd.cpp'] as const;
  assert.deepEqual(modelChoices(files, all, true).map(({ runtime, name, entry, size }) => [runtime, name, entry, size]), [
    ['sd.cpp', 'sd_turbo-q8_0.gguf', 'old/sd_turbo-q8_0.gguf', 2_000],
    ['sd.cpp', 'sd_turbo-f16.gguf', 'sd_turbo-f16.gguf', 2_500],
    ['sd.cpp', 'v1-5-pruned-emaonly.safetensors', 'v1-5-pruned-emaonly.safetensors', 4_000],
  ]);
  // A build without the image runtime offers nothing from it, and never loads its GGUFs as chat models.
  assert.deepEqual(modelChoices(files, ['llama.cpp', 'mlx'], true), []);
  // Without the tags the same GGUFs are chat models, as before.
  assert.deepEqual(modelChoices(files, all).map(({ runtime }) => runtime), ['llama.cpp', 'llama.cpp']);
});

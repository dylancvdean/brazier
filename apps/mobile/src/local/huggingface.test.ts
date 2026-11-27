import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileUrl, formatBytes, listRepoFiles, modelChoices, parseSlug, parseTree } from './huggingface.ts';

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

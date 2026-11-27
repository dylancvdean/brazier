import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  catalogue, catalogueChoice, fitNote, memoryFit, quantisation, recommend, suggestedChoice, type CatalogueEntry,
} from './catalogue.ts';
import type { ModelChoice } from './huggingface.ts';
import { IMAGE_WORKING, memoryBudget, memoryNeeded } from './memory.ts';
import { modelId } from './models.ts';

const GB = 1e9;
const all = ['llama.cpp', 'mlx', 'sd.cpp'] as const;
const phone = (total: number, free = 64 * GB, runtimes: readonly ('llama.cpp' | 'mlx' | 'sd.cpp')[] = ['llama.cpp', 'sd.cpp']) =>
  ({ runtimes, memory: { total: total * GB }, free });
const best = (kind: 'chat' | 'image', device: Parameters<typeof recommend>[1]) =>
  recommend(catalogue, device).find((item) => item.best && item.entry.kind === kind)?.entry.id;

test('catalogue entries are well formed and unique', () => {
  const ids = new Set<string>();
  const downloads = new Set<string>();
  for (const entry of catalogue) {
    assert.ok(!ids.has(entry.id), entry.id);
    ids.add(entry.id);
    const download = `${entry.slug}/${entry.entry}`;
    assert.ok(!downloads.has(download), download);
    downloads.add(download);
    assert.match(entry.slug, /^[\w.-]+\/[\w.-]+$/, entry.id);
    assert.ok(entry.size > 100e6 && entry.size < 8 * GB, entry.id);
    assert.ok(entry.description.length > 20 && entry.description.length <= 120 && !entry.description.includes('\n'), entry.id);
    assert.equal(entry.runtime === 'mlx', entry.entry === '', entry.id);
    if (entry.runtime !== 'mlx') assert.match(entry.entry, /\.gguf$/, entry.id);
    assert.equal(entry.kind === 'image', entry.runtime === 'sd.cpp', entry.id);
  }
  for (const runtime of all) assert.ok(catalogue.some((entry) => entry.runtime === runtime), runtime);
});

test('fit uses the load warning\'s estimate, with a tight band below the budget', () => {
  const memory = { total: 8 * GB };
  const budget = memoryBudget(memory)!;
  const spare = budget - memoryNeeded(0);
  assert.equal(memoryFit(spare * 0.5, memory), 'fits');
  assert.equal(memoryFit(spare - 0.1 * GB, memory), 'tight');
  assert.equal(memoryFit(spare + 1, memory), 'too-large');
  assert.equal(memoryFit(1 * GB, {}), 'unknown');
  assert.equal(memoryFit(2 * GB, { limit: 4 * GB, total: 64 * GB }, IMAGE_WORKING), 'fits');
  assert.equal(memoryFit(2 * GB, { limit: 2.5 * GB, total: 64 * GB }, IMAGE_WORKING), 'too-large');
});

test('only models a usable runtime can load are recommended', () => {
  assert.ok(recommend(catalogue, phone(8)).every(({ entry }) => entry.runtime !== 'mlx'));
  assert.ok(recommend(catalogue, phone(8, 64 * GB, ['llama.cpp'])).every(({ entry }) => entry.kind === 'chat'));
  assert.ok(recommend(catalogue, phone(8, 64 * GB, all)).some(({ entry }) => entry.runtime === 'mlx'));
  assert.deepEqual(recommend(catalogue, phone(8, 64 * GB, [])), []);
});

test('the best pick grows with the device and always fits', () => {
  const sizes = [3, 4, 6, 8, 12, 16].map((total) => {
    const picks = recommend(catalogue, phone(total)).filter((item) => item.best && item.entry.kind === 'chat');
    assert.equal(picks.length, 1, `${total} GB`);
    assert.equal(picks[0].fit, 'fits', `${total} GB`);
    return picks[0].entry.size;
  });
  assert.deepEqual(sizes, [...sizes].sort((a, b) => a - b));
  assert.ok(sizes[0] < sizes.at(-1)!);
  assert.equal(best('chat', phone(16)), 'qwen3.5-9b');
  assert.equal(best('image', phone(8)), 'sd-turbo');
});

test('the best pick comes first, then what fits by rank, then what is too large by size', () => {
  const list = recommend(catalogue, phone(6, 64 * GB, ['llama.cpp']));
  assert.ok(list[0].best);
  assert.equal(list.filter((item) => item.best).length, 1);
  const usable = list.slice(1).filter((item) => item.fit !== 'too-large');
  const large = list.filter((item) => item.fit === 'too-large');
  assert.ok(large.length > 0);
  assert.deepEqual(list.slice(1), [...usable, ...large]);
  assert.deepEqual(usable.map(({ entry }) => entry.rank), usable.map(({ entry }) => entry.rank).sort((a, b) => b - a));
  assert.deepEqual(large.map(({ entry }) => entry.size), large.map(({ entry }) => entry.size).sort((a, b) => a - b));
});

test('the best pick needs storage, and stays small when memory is unknown', () => {
  const full = recommend(catalogue, phone(16, 1 * GB, ['llama.cpp']));
  const pick = full.find((item) => item.best)!;
  assert.ok(pick.entry.size <= 1 * GB && pick.space);
  assert.ok(full.some((item) => !item.space));
  assert.equal(recommend(catalogue, phone(16, 1e6)).some((item) => item.best), false);
  const unknown = recommend(catalogue, { runtimes: ['llama.cpp'], memory: {}, free: 64 * GB }).find((item) => item.best)!;
  assert.equal(unknown.fit, 'unknown');
  assert.ok(unknown.entry.size <= 1 * GB);
});

test('an installed model is marked and needs no more storage', () => {
  const entry = catalogue.find(({ id }) => id === 'qwen3.5-4b')!;
  const [item] = recommend([entry], { ...phone(12, 0), installed: [modelId(entry.slug, entry)] });
  assert.deepEqual([item.installed, item.space, item.best], [true, true, true]);
});

test('a catalogue entry resolves to the repository\'s file with its current size', () => {
  const entry: CatalogueEntry = { ...catalogue[0], slug: 'a/b', entry: 'b-Q4_K_M.gguf' };
  const files = [{ path: 'b-Q4_K_M.gguf', size: 42 }, { path: 'b-Q8_0.gguf', size: 99 }, { path: 'README.md', size: 1 }];
  assert.equal(catalogueChoice(entry, files)?.size, 42);
  assert.equal(catalogueChoice(entry, files.slice(1)), null);
  const mlx = catalogue.find(({ runtime }) => runtime === 'mlx')!;
  const repository = [{ path: 'config.json', size: 10 }, { path: 'model.safetensors', size: 1000 }, { path: 'tokenizer.json', size: 5 }];
  assert.equal(catalogueChoice(mlx, repository)?.runtime, 'mlx');
  assert.equal(catalogueChoice(mlx, [{ path: 'model.gguf', size: 1 }]), null);
});

test('a quantisation is read from a file name', () => {
  for (const [name, expected] of [
    ['Qwen3-4B-Q4_K_M.gguf', 'Q4_K_M'], ['model.Q8_0.gguf', 'Q8_0'], ['gemma-3-4b-it-IQ4_XS.gguf', 'IQ4_XS'],
    ['sd_turbo-f16-q8_0.gguf', 'Q8_0'], ['Llama-3.2-3B-Instruct-f16.gguf', 'F16'], ['Qwen3-4B-4bit', '4-bit'],
    ['Phi-4-mini-instruct-Q4_K_M', 'Q4_K_M'], ['Qwen3-0.6B-UD-Q4_K_XL.gguf', 'Q4_K_XL'],
    ['v1-5-pruned-emaonly.safetensors', null], ['Qwen3-4B.gguf', null],
  ] as const) assert.equal(quantisation(name), expected, name);
});

test('the suggested file is Q4_K_M when it fits, else the largest smaller one that does', () => {
  const choice = (name: string, size: number): ModelChoice =>
    ({ runtime: 'llama.cpp', name, entry: `${name}.gguf`, files: [{ path: `${name}.gguf`, size }], size });
  const choices = [choice('m-Q2_K', 1 * GB), choice('m-Q3_K_M', 1.5 * GB), choice('m-Q4_K_M', 2 * GB), choice('m-Q8_0', 4 * GB)];
  const roomy = { total: 16 * GB };
  assert.equal(suggestedChoice(choices, roomy, 64 * GB)?.name, 'm-Q4_K_M');
  assert.equal(suggestedChoice(choices, roomy, 1.6 * GB)?.name, 'm-Q3_K_M');
  assert.equal(suggestedChoice(choices, { total: 5.5 * GB }, 64 * GB)?.name, 'm-Q3_K_M');
  assert.equal(suggestedChoice(choices, { total: 2 * GB }, 64 * GB), null);
  assert.equal(suggestedChoice(choices, {}, 64 * GB)?.name, 'm-Q4_K_M');
  assert.equal(suggestedChoice([choices[0], choices[3]], roomy, 64 * GB), null);
  assert.equal(suggestedChoice([choices[3]], roomy, 64 * GB)?.name, 'm-Q8_0');
  assert.equal(suggestedChoice([choices[3]], roomy, 1 * GB), null);
  assert.equal(suggestedChoice([], roomy, 64 * GB), null);
});

test('a fit note puts storage before memory', () => {
  assert.deepEqual(fitNote('fits', false), { text: 'Not enough free space', problem: true });
  assert.equal(fitNote('too-large', true)?.problem, true);
  assert.equal(fitNote('tight', true)?.problem, false);
  assert.equal(fitNote('fits', true)?.text, 'Fits this device');
  assert.equal(fitNote('unknown', true), null);
});

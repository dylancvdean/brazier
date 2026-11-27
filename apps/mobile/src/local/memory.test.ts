import assert from 'node:assert/strict';
import test from 'node:test';
import { contextBytes, memoryBudget, memoryNeeded, memoryWarning } from './memory.ts';

const GB = 1e9;

test('the context cache follows layers and key-value heads, not file size', () => {
  // Qwen3 8B: 36 layers, 8 of 32 heads hold keys and values, 4096 wide.
  const qwen = { 'general.architecture': 'qwen3', 'qwen3.block_count': '36', 'qwen3.attention.head_count': '32',
    'qwen3.attention.head_count_kv': '8', 'qwen3.embedding_length': '4096' };
  assert.equal(contextBytes(qwen), 2 * 2 * 4096 * 36 * 8 * 128);
  // Phi-3 mini: every head holds keys and values, so a smaller model has a larger cache.
  const phi = { 'general.architecture': 'phi3', 'phi3.block_count': 32, 'phi3.attention.head_count': 32,
    'phi3.attention.head_count_kv': 32, 'phi3.embedding_length': 3072 };
  assert.equal(contextBytes(phi), 2 * 2 * 4096 * 32 * 32 * 96);
  assert.ok(contextBytes(phi)! > contextBytes(qwen)!);
});

test('the context cache uses the stated head size and falls back to all heads', () => {
  const gemma = { 'general.architecture': 'gemma3', 'gemma3.block_count': 34, 'gemma3.attention.head_count': 8,
    'gemma3.attention.head_count_kv': 4, 'gemma3.embedding_length': 2560, 'gemma3.attention.key_length': 256 };
  assert.equal(contextBytes(gemma), 2 * 2 * 4096 * 34 * 4 * 256);
  const plain = { 'general.architecture': 'llama', 'llama.block_count': 22, 'llama.attention.head_count': 32,
    'llama.embedding_length': 2048 };
  assert.equal(contextBytes(plain, 2048), 2 * 2 * 2048 * 22 * 32 * 64);
});

test('metadata without the figures gives no context size', () => {
  assert.equal(contextBytes(null), null);
  assert.equal(contextBytes({}), null);
  assert.equal(contextBytes({ 'general.architecture': 'llama', 'llama.block_count': 22 }), null);
});

test('a model needs its file size, its context cache and working memory', () => {
  assert.equal(memoryNeeded(5 * GB, 1.6 * GB), 6.85 * GB);
  assert.equal(memoryNeeded(5 * GB), 5.85 * GB);
  assert.equal(memoryNeeded(5 * GB, null), 5.85 * GB);
});

test('the budget is what the system allows, or half of memory when it does not say', () => {
  assert.equal(memoryBudget({ limit: 3.4 * GB, total: 12 * GB }), 3.4 * GB);
  assert.equal(memoryBudget({ limit: null, total: 8 * GB }), 4 * GB);
  assert.equal(memoryBudget({ limit: 0, total: 8 * GB }), 4 * GB);
  assert.equal(memoryBudget({}), null);
  assert.equal(memoryBudget({ limit: Number.NaN, total: 0 }), null);
});

test('warns against the limit the system reports', () => {
  // A 12 GB iPhone that allows this app 3.4 GB: a 2.5 GB file fits, a 3 GB file does not.
  const phone = { limit: 3.4 * GB, total: 12 * GB };
  assert.equal(memoryWarning(2.5 * GB, phone), null);
  assert.equal(memoryWarning(3 * GB, phone),
    "May be too large for this device's memory (3.0 GB model, 3.4 GB available)");
});

test('the model’s own context size decides near the limit', () => {
  const phone = { limit: 4 * GB };
  assert.equal(memoryWarning(2.4 * GB, phone), null);
  assert.equal(memoryWarning(2.4 * GB, phone, 0.5 * GB), null);
  assert.ok(memoryWarning(2.4 * GB, phone, 1.6 * GB));
  assert.ok(memoryWarning(3.2 * GB, phone));
  assert.equal(memoryWarning(3.2 * GB, phone, 0.1 * GB), null);
});

test('without a reported limit, half of physical memory is the budget', () => {
  assert.equal(memoryWarning(3 * GB, { total: 8 * GB }), null);
  assert.equal(memoryWarning(3.2 * GB, { total: 8 * GB }),
    "May be too large for this device's memory (3.2 GB model, 4.0 GB available)");
});

test('unknown memory or size never warns', () => {
  assert.equal(memoryWarning(50 * GB, {}), null);
  assert.equal(memoryWarning(50 * GB, { limit: null, total: null }), null);
  assert.equal(memoryWarning(50 * GB, { total: Number.NaN }), null);
  assert.equal(memoryWarning(0, { total: 8 * GB }), null);
  assert.equal(memoryWarning(Number.NaN, { total: 8 * GB }), null);
});

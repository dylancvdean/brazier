import assert from 'node:assert/strict';
import test from 'node:test';
import { memoryBudget, memoryNeeded, memoryWarning } from './memory.ts';

const GB = 1e9;

test('a model needs more memory than its file size', () => {
  assert.equal(memoryNeeded(5 * GB), 6 * GB);
});

test('an app can count on half of physical memory', () => {
  assert.equal(memoryBudget(8 * GB), 4 * GB);
});

test('small models on ordinary phones are not warned about', () => {
  assert.equal(memoryWarning(0.4 * GB, 4 * GB), null);
  assert.equal(memoryWarning(2.5 * GB, 8 * GB), null);
});

test('warns once the loaded model passes half of memory', () => {
  // 8 GB phone: 4 GB budget, so files above 4 / 1.2 = 3.33 GB warn.
  assert.equal(memoryWarning(3.3 * GB, 8 * GB), null);
  assert.equal(memoryWarning(3.4 * GB, 8 * GB),
    "May be too large for this device's memory (3.4 GB model, 8.0 GB memory)");
  assert.match(memoryWarning(16.4 * GB, 6 * GB) ?? '', /16\.4 GB model, 6\.0 GB memory/);
});

test('a model that fits on disk but is larger than memory warns', () => {
  assert.ok(memoryWarning(9 * GB, 8 * GB));
});

test('unknown memory or size never warns', () => {
  assert.equal(memoryWarning(50 * GB, null), null);
  assert.equal(memoryWarning(50 * GB, undefined), null);
  assert.equal(memoryWarning(50 * GB, 0), null);
  assert.equal(memoryWarning(50 * GB, Number.NaN), null);
  assert.equal(memoryWarning(0, 8 * GB), null);
  assert.equal(memoryWarning(Number.NaN, 8 * GB), null);
});

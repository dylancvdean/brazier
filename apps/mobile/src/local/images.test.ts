import assert from 'node:assert/strict';
import { test } from 'node:test';
import { generateBody, historyEntry, isCancellation, type GenerateBody } from '../media.ts';
import { ImageCancelled, imageDefaults, imageEngine, imageModelId, type ImageRequest, type ImageRuntime } from './images.ts';
import type { InstalledModel } from './models.ts';

const model = (id: string, name = id): InstalledModel =>
  ({ id, slug: 'owner/repo', runtime: 'sd.cpp', name, entry: `${name}.gguf`, files: [`${name}.gguf`], size: 2e9 });
const chat: InstalledModel = { ...model('gguf--chat'), runtime: 'llama.cpp' };
const sha256 = 'a'.repeat(64);
const body = (id: string, extra: Partial<GenerateBody> = {}): GenerateBody =>
  ({ prompt: 'a lighthouse', model_id: imageModelId({ id }), width: 512, height: 512, ...extra });

function setup(installed: InstalledModel[] = [model('sd--turbo', 'sd_turbo-q8_0'), model('sd--v15', 'v1-5-Q4_0'), chat]) {
  const log: string[] = [];
  const requests: ImageRequest[] = [];
  let finish: (() => void) | null = null;
  let hold = false;
  let clock = 1000;
  const runtime: ImageRuntime = {
    id: 'sd.cpp',
    available: async () => true,
    load: async ({ id }) => { log.push(`load ${id}`); },
    async generate(request, signal, onProgress) {
      requests.push(request);
      log.push('generate');
      onProgress({ stage: 'loading', completed: 3, total: 900 });
      onProgress({ stage: 'sampling', completed: 2, total: request.steps });
      if (hold) await new Promise<void>((resolve) => { finish = resolve; signal.addEventListener('abort', () => resolve()); });
      if (signal.aborted) throw new ImageCancelled();
      return { sha256, size: 4096, width: request.width, height: request.height };
    },
    unload: async () => { log.push('unload'); },
  };
  const engine = imageEngine(runtime, {
    installed: async () => installed,
    output: () => '/images/new.part',
    keep: (output, file) => { log.push(`keep ${output} ${file.sha256.slice(0, 4)}`); },
    beforeLoad: async () => { log.push('free chat model'); },
    now: () => clock,
    random: () => 0.5,
  });
  return {
    engine, log, requests, runtime,
    holdJobs: (on = true) => { hold = on; },
    finish: () => finish?.(),
    advance: (ms: number) => { clock += ms; },
  };
}
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

test('only image models are offered, under ids the Generate screen lists as image models', async () => {
  const { engine, runtime } = setup();
  assert.deepEqual(await engine.models(), [
    { id: 'sdcpp-image:sd--turbo', name: 'sd_turbo-q8_0', engine: 'sd.cpp' },
    { id: 'sdcpp-image:sd--v15', name: 'v1-5-Q4_0', engine: 'sd.cpp' },
  ]);
  runtime.available = async () => false;
  assert.deepEqual(await engine.models(), []);
});

test('a job frees the chat model, loads the image model once, and returns what the host returns', async () => {
  const { engine, log, requests } = setup();
  const result = await engine.generate(body('sd--v15', { negative_prompt: 'blurry', steps: 8, seed: 7 }));
  assert.deepEqual(log, ['free chat model', 'unload', 'load sd--v15', 'generate', `keep /images/new.part aaaa`]);
  assert.deepEqual(requests[0], {
    prompt: 'a lighthouse', negative: 'blurry', width: 512, height: 512, steps: 8, cfg: 7, seed: 7, output: '/images/new.part',
  });
  assert.deepEqual(result, {
    blob: { sha256, mime_type: 'image/png', size_bytes: 4096 },
    metadata: { width: 512, height: 512, steps: 8, seed: 7 }, engine: 'sd.cpp',
  });
  // The history entry is built the same way as for a host.
  const entry = historyEntry('1', 'image', body('sd--v15'), result, '2026-01-01T00:00:00Z');
  assert.equal(entry.steps, 8);
  assert.equal(entry.seed, 7);

  await engine.generate(body('sd--v15'));
  assert.equal(log.filter((line) => line.startsWith('load')).length, 1);
  await engine.generate(body('sd--turbo'));
  assert.deepEqual(log.slice(-5), ['free chat model', 'unload', 'load sd--turbo', 'generate', 'keep /images/new.part aaaa']);
});

test('blank steps and seed get defaults that suit the model, and the seed used is reported', async () => {
  assert.deepEqual(imageDefaults('sd_turbo-f16-q8_0'), { steps: 4, cfg: 1 });
  assert.deepEqual(imageDefaults('dreamshaper-LCM'), { steps: 4, cfg: 1 });
  assert.deepEqual(imageDefaults('stable-diffusion-v1-5-pruned-emaonly-Q4_0'), { steps: 20, cfg: 7 });
  const { engine, requests } = setup();
  const result = await engine.generate(body('sd--turbo'));
  assert.equal(requests[0].steps, 4);
  assert.equal(requests[0].cfg, 1);
  assert.equal(requests[0].seed, 1_073_741_823);
  assert.deepEqual(result.metadata, { width: 512, height: 512, steps: 4, seed: 1_073_741_823 });
});

test('progress is reported in sampling steps with the time since the job started', async () => {
  const { engine, holdJobs, finish, advance } = setup();
  assert.equal(engine.active(), null);
  holdJobs();
  const job = engine.generate(body('sd--v15', { steps: 8 }));
  assert.deepEqual(engine.active(), {
    id: '1000', modality: 'image', model_id: 'sdcpp-image:sd--v15', prompt: 'a lighthouse', origin: 'user',
    timeout_secs: 0, current_step: 0, total_steps: 0, elapsed_secs: 0,
  });
  await tick();
  advance(2500);
  const active = engine.active()!;
  assert.deepEqual([active.current_step, active.total_steps, active.elapsed_secs], [2, 8, 2.5]);
  finish();
  await job;
  assert.equal(engine.active(), null);
});

test('cancelling stops the job with the error the Generate screen treats as a stop', async () => {
  const { engine, holdJobs, log } = setup();
  assert.equal(engine.cancel(), false);
  holdJobs();
  const job = engine.generate(body('sd--v15'));
  await tick();
  assert.equal(engine.cancel(), true);
  const error = await job.then(() => null, (cause: unknown) => cause);
  assert.ok(error instanceof ImageCancelled);
  assert.equal(isCancellation(error), true);
  assert.equal(log.some((line) => line.startsWith('keep')), false);
  assert.equal(engine.active(), null);
  // The model stays loaded for the next job.
  holdJobs(false);
  await engine.generate(body('sd--v15'));
  assert.equal(log.filter((line) => line.startsWith('load')).length, 1);
});

test('one job at a time, and unknown or chat models are refused', async () => {
  const { engine, holdJobs, finish, log } = setup();
  const none: string[] = [];
  await assert.rejects(engine.generate(body('missing')), /not on this device/);
  await assert.rejects(engine.generate(body('gguf--chat')), /not on this device/);
  await assert.rejects(engine.generate({ ...body('sd--v15'), model_id: 'sd--v15' }), /not on this device/);
  assert.deepEqual(log, none);
  holdJobs();
  const job = engine.generate(body('sd--v15'));
  await assert.rejects(engine.generate(body('sd--v15')), /still being generated/);
  await tick();
  finish();
  await job;
});

test('unloading waits for the running job and can be limited to one model', async () => {
  const { engine, holdJobs, finish, log } = setup();
  await engine.unload();
  assert.equal(log.length, 0);
  await engine.generate(body('sd--v15'));
  await engine.unload('sd--turbo');
  assert.equal(log.at(-1), 'keep /images/new.part aaaa');
  holdJobs();
  const job = engine.generate(body('sd--v15'));
  await tick();
  const unloaded = engine.unload('sd--v15');
  await tick();
  assert.equal(log.at(-1), 'generate');
  finish();
  await Promise.all([job, unloaded]);
  assert.equal(log.at(-1), 'unload');
  holdJobs(false);
  await engine.generate(body('sd--v15'));
  assert.equal(log.filter((line) => line.startsWith('load')).length, 2);
});

test('a model that fails to load is not treated as loaded', async () => {
  const { engine, runtime, log } = setup();
  const load = runtime.load;
  runtime.load = async () => { throw new Error('not an image model'); };
  await assert.rejects(engine.generate(body('sd--v15')), /not an image model/);
  runtime.load = load;
  await engine.generate(body('sd--v15'));
  assert.equal(log.filter((line) => line.startsWith('load')).length, 1);
});

test('the form for this device makes the same request as for a host', () => {
  assert.deepEqual(generateBody({
    modality: 'image', modelId: 'sdcpp-image:sd--v15', prompt: ' a lighthouse ', negative: '', width: 512, height: 512,
    steps: '', seed: '', frames: '',
  }), { prompt: 'a lighthouse', model_id: 'sdcpp-image:sd--v15', width: 512, height: 512 });
});

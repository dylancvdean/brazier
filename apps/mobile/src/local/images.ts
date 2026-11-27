import type { Model } from '../api.ts';
import type { ActiveGeneration, GenerateBody, GenerateResult } from '../media.ts';
import type { RuntimeId } from './huggingface.ts';
import type { InstalledModel } from './models.ts';

export type ImageRequest = {
  prompt: string; negative: string; width: number; height: number; steps: number; cfg: number; seed: number;
  /** Where the PNG is written. */
  output: string;
};
/** `stage` is the runtime's name for what it is doing; `total` is 0 when it cannot say how much there is. */
export type ImageProgress = { stage: string; completed: number; total: number };
export type ImageFile = { sha256: string; size: number; width: number; height: number };

/** One way of making images on this device. */
export type ImageRuntime = {
  id: RuntimeId;
  /** False when this build cannot run it; its models are then not offered. */
  available(): Promise<boolean>;
  load(model: InstalledModel): Promise<void>;
  /** Aborting `signal` stops the job, which then rejects with `ImageCancelled`. */
  generate(request: ImageRequest, signal: AbortSignal, onProgress: (progress: ImageProgress) => void): Promise<ImageFile>;
  unload(): Promise<void>;
};

export class ImageCancelled extends Error {
  // The wording the Generate screen recognises as a stop rather than a failure.
  constructor() { super('Generation stopped by the user.'); }
}

const prefix = 'sdcpp-image:';
export const imageModelId = (model: Pick<InstalledModel, 'id'>) => `${prefix}${model.id}`;

/**
 * Steps and guidance when the form leaves them blank. Distilled models (Turbo, LCM, Lightning, Hyper) are made
 * for a few steps without guidance; the name is the only sign of one.
 */
export function imageDefaults(name: string): { steps: number; cfg: number } {
  return /turbo|lcm|lightning|hyper/i.test(name) ? { steps: 4, cfg: 1 } : { steps: 20, cfg: 7 };
}

/** Sizes offered on a phone: memory and time grow with the area. */
export const imageSizes: readonly (readonly [number, number])[] = [[512, 512], [384, 384], [512, 768], [768, 512]];

export type ImageEngine = {
  models(): Promise<Model[]>;
  generate(body: GenerateBody): Promise<GenerateResult>;
  active(): ActiveGeneration | null;
  /** True when a job was running and has been told to stop. */
  cancel(): boolean;
  /** Frees the loaded model; with an id, only if that model is the one loaded. Waits for a running job. */
  unload(modelId?: string): Promise<void>;
};

/**
 * Image generation on this device: one model loaded, one job at a time. `output` names the file for a new
 * image, `keep` files it under its hash, and `beforeLoad` makes room (the chat model is unloaded).
 */
export function imageEngine(runtime: ImageRuntime, deps: {
  installed(): Promise<InstalledModel[]>;
  output(): string;
  keep(output: string, file: ImageFile): void;
  beforeLoad(): Promise<void>;
  now(): number;
  random(): number;
}): ImageEngine {
  let loaded: string | null = null;
  type Job = { state: Omit<ActiveGeneration, 'elapsed_secs'>; started: number; abort: AbortController };
  let job: Job | null = null;
  let last: Promise<unknown> = Promise.resolve();
  const settled = () => last.catch(() => undefined);

  async function run(body: GenerateBody, current: Job): Promise<GenerateResult> {
    const id = body.model_id.startsWith(prefix) ? body.model_id.slice(prefix.length) : '';
    const model = (await deps.installed()).find((candidate) => candidate.id === id && candidate.runtime === runtime.id);
    if (!model) throw new Error('This image model is not on this device. Add it in Models.');
    if (!await runtime.available()) throw new Error('This build of the app cannot generate images on this device.');
    if (loaded !== model.id) {
      await deps.beforeLoad();
      loaded = null;
      await runtime.unload().catch(() => undefined);
      if (current.abort.signal.aborted) throw new ImageCancelled();
      await runtime.load(model);
      loaded = model.id;
    }
    if (current.abort.signal.aborted) throw new ImageCancelled();
    const defaults = imageDefaults(model.name);
    const steps = body.steps ?? defaults.steps;
    const seed = body.seed ?? Math.floor(deps.random() * 2_147_483_647);
    const output = deps.output();
    current.state = { ...current.state, total_steps: steps };
    const file = await runtime.generate({
      prompt: body.prompt, negative: body.negative_prompt ?? '', width: body.width, height: body.height,
      steps, cfg: defaults.cfg, seed, output,
    }, current.abort.signal, ({ stage, completed, total }) => {
      if (stage !== 'sampling' || total <= 0) return;
      current.state = { ...current.state, current_step: Math.min(completed, total), total_steps: total };
    });
    deps.keep(output, file);
    return {
      blob: { sha256: file.sha256, mime_type: 'image/png', size_bytes: file.size },
      metadata: { width: file.width, height: file.height, steps, seed }, engine: runtime.id,
    };
  }

  return {
    async models() {
      if (!await runtime.available()) return [];
      return (await deps.installed()).filter((model) => model.runtime === runtime.id)
        .map((model): Model => ({ id: imageModelId(model), name: model.name, engine: runtime.id }));
    },
    generate(body) {
      if (job) return Promise.reject(new Error('Another image is still being generated.'));
      const current: Job = {
        started: deps.now(), abort: new AbortController(),
        state: {
          id: String(deps.now()), modality: 'image', model_id: body.model_id, prompt: body.prompt, origin: 'user',
          timeout_secs: 0, current_step: 0, total_steps: 0,
        },
      };
      job = current;
      const result = settled().then(() => run(body, current)).finally(() => { if (job === current) job = null; });
      last = result;
      return result;
    },
    active: () => job ? { ...job.state, elapsed_secs: Math.max(0, (deps.now() - job.started) / 1000) } : null,
    cancel() {
      if (!job) return false;
      job.abort.abort();
      return true;
    },
    unload(modelId) {
      const next = settled().then(async () => {
        if (!loaded || (modelId && loaded !== modelId)) return;
        loaded = null;
        await runtime.unload();
      });
      last = next;
      return next;
    },
  };
}

import type { ModelChoice, RuntimeId } from './huggingface.ts';

/** A model whose files are complete on this device. `entry` is the file to load, or '' for the whole directory. */
export type InstalledModel = {
  id: string; slug: string; runtime: RuntimeId; name: string; entry: string; files: string[]; size: number;
};

const safe = (text: string) => text.replace(/[^A-Za-z0-9._-]/g, '_');

/** Also the model's directory name, so it is unique per repository, runtime and file. */
export function modelId(slug: string, choice: Pick<ModelChoice, 'runtime' | 'entry'>): string {
  const runtime = choice.runtime === 'llama.cpp' ? 'gguf' : choice.runtime;
  return [runtime, safe(slug.replace('/', '--')), ...(choice.entry ? [safe(choice.entry)] : [])].join('--');
}

export function installedFrom(slug: string, choice: ModelChoice): InstalledModel {
  return {
    id: modelId(slug, choice), slug, runtime: choice.runtime, entry: choice.entry, size: choice.size,
    // Short enough for the model chip; the Models screen shows the repository beside it.
    name: choice.runtime === 'mlx' ? slug.slice(slug.indexOf('/') + 1) : choice.name.replace(/\.gguf$/i, ''),
    files: choice.files.map(({ path }) => path),
  };
}

export function parseInstalled(raw: string | null): InstalledModel[] {
  if (!raw) return [];
  let value: unknown;
  try { value = JSON.parse(raw); } catch { return []; }
  if (!Array.isArray(value)) return [];
  const seen = new Set<string>();
  return value.flatMap((entry): InstalledModel[] => {
    const model = entry as Partial<InstalledModel>;
    if (typeof model?.id !== 'string' || model.id !== safe(model.id) || !model.id || seen.has(model.id) ||
        typeof model.slug !== 'string' || typeof model.name !== 'string' || typeof model.entry !== 'string' ||
        !['llama.cpp', 'mlx'].includes(model.runtime ?? '') || !Array.isArray(model.files) ||
        !model.files.every((file) => typeof file === 'string' && !file.split('/').includes('..'))) return [];
    seen.add(model.id);
    return [{
      id: model.id, slug: model.slug, runtime: model.runtime!, name: model.name, entry: model.entry,
      files: model.files, size: typeof model.size === 'number' ? model.size : 0,
    }];
  });
}

export const addInstalled = (models: readonly InstalledModel[], model: InstalledModel): InstalledModel[] =>
  [...models.filter(({ id }) => id !== model.id), model];

export const removeInstalled = (models: readonly InstalledModel[], id: string): InstalledModel[] =>
  models.filter((model) => model.id !== id);

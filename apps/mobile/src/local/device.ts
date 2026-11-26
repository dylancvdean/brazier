import { fetch } from 'expo/fetch';
import { Directory, File, Paths } from 'expo-file-system';
import { createDownloadResumable } from 'expo-file-system/legacy';
import { conversationStore } from './conversations.ts';
import { availableRuntimes, localEngine, type ChatBackend, type Runtime } from './engine.ts';
import { fileUrl, listRepoFiles, modelChoices, parseSlug, type ModelChoice, type RuntimeId } from './huggingface.ts';
import { addInstalled, installedFrom, modelId, parseInstalled, removeInstalled, type InstalledModel } from './models.ts';
import { appleRuntime, llamaRuntime } from './runtimes.ts';

// Everything "This device" keeps lives under one directory in the app's documents.
const root = () => {
  const directory = new Directory(Paths.document, 'local');
  if (!directory.exists) directory.create({ intermediates: true, idempotent: true });
  return directory;
};
const modelsRoot = () => new Directory(root(), 'models');
const modelDirectory = (id: string) => new Directory(modelsRoot(), id);
const filePath = (file: File | Directory) => decodeURIComponent(file.uri.replace(/^file:\/\//, '')).replace(/\/$/, '');

const textFiles = {
  async read(name: string) {
    const file = new File(root(), name);
    if (file.exists) return file.text();
    // A write that was interrupted between removing the old file and moving the new one into place.
    const next = new File(root(), `${name}.next`);
    return next.exists ? next.text() : null;
  },
  // Written beside the file and then moved into place, so a write that fails (a full disk) keeps the old contents.
  async write(name: string, text: string) {
    const next = new File(root(), `${name}.next`);
    if (next.exists) next.delete();
    next.create({ intermediates: true });
    next.write(text);
    const file = new File(root(), name);
    if (file.exists) file.delete();
    next.move(file);
  },
  async remove(name: string) {
    const file = new File(root(), name);
    if (file.exists) file.delete();
  },
};

const installedFile = 'models.json';
export const installedModels = async (): Promise<InstalledModel[]> => parseInstalled(await textFiles.read(installedFile));

// A download finishing and a delete both rewrite the list, so changes to it run one at a time.
let listChange: Promise<unknown> = Promise.resolve();
function changeInstalled(change: (models: InstalledModel[]) => InstalledModel[]): Promise<void> {
  const next = listChange.catch(() => undefined)
    .then(async () => textFiles.write(installedFile, JSON.stringify(change(await installedModels()))));
  listChange = next;
  return next;
}

const newId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

export const runtimes: readonly Runtime[] = [
  llamaRuntime((model) => filePath(new File(modelDirectory(model.id), model.entry))),
  appleRuntime(),
];

export const deviceEngine = localEngine(runtimes, installedModels);

/** Chat on this device: local runtimes, conversations kept on the phone, and no tools. */
export const deviceBackend: ChatBackend = {
  ...deviceEngine, ...conversationStore(textFiles, newId), tools: async () => [],
};

export const freeSpace = (): number => Paths.availableDiskSpace;

export async function usableRuntimes(): Promise<RuntimeId[]> {
  return (await availableRuntimes(runtimes)).map(({ id }) => id);
}

export async function builtInModels() {
  return (await Promise.all((await availableRuntimes(runtimes)).map((runtime) => runtime.builtIn?.() ?? []))).flat();
}

export async function repositoryChoices(input: string): Promise<{ slug: string; choices: ModelChoice[] }> {
  const slug = parseSlug(input);
  return { slug, choices: modelChoices(await listRepoFiles(fetch, slug), await usableRuntimes()) };
}

export type ActiveDownload = { id: string; name: string; size: number; bytes: number };

// One download at a time, kept here rather than in the Models screen so it carries on, and is still shown,
// when the screen is closed and reopened.
let active: (ActiveDownload & { cancel(): void }) | null = null;
const watchers = new Set<() => void>();
const notify = () => watchers.forEach((watcher) => watcher());

export const activeDownload = (): ActiveDownload | null => active;
export const cancelDownload = () => active?.cancel();
/** Called when the download starts, makes progress, and ends. */
export function watchDownload(watcher: () => void): () => void {
  watchers.add(watcher);
  return () => { watchers.delete(watcher); };
}

/** Files land in a `.part` directory that becomes the model's directory only once all of them are complete. */
export async function downloadModel(slug: string, choice: ModelChoice): Promise<InstalledModel> {
  if (active) throw new Error('Another model is still downloading.');
  const id = modelId(slug, choice);
  const partial = new Directory(modelsRoot(), `${id}.part`);
  let cancelled = false;
  let current: ReturnType<typeof createDownloadResumable> | null = null;
  let shown = 0;
  const progress = (bytes: number) => {
    if (!active) return;
    active = { ...active, bytes };
    // Progress arrives far more often than a screen needs to redraw.
    if (Date.now() - shown < 250 && bytes < choice.size) return;
    shown = Date.now();
    notify();
  };
  active = {
    id, name: choice.name, size: choice.size, bytes: 0,
    cancel: () => { cancelled = true; void current?.cancelAsync().catch(() => undefined); },
  };
  notify();
  try {
    if (partial.exists) partial.delete();
    partial.create({ intermediates: true, idempotent: true });
    let finished = 0;
    for (const file of choice.files) {
      if (cancelled) break;
      const target = new File(partial, file.path);
      if (!target.parentDirectory.exists) target.parentDirectory.create({ intermediates: true, idempotent: true });
      current = createDownloadResumable(fileUrl(slug, file.path), target.uri, {},
        ({ totalBytesWritten }) => progress(finished + totalBytesWritten));
      const result = await current.downloadAsync();
      if (cancelled) break;
      if (!result || result.status !== 200) {
        throw new Error(`Hugging Face returned HTTP ${result?.status ?? 'no response'} for ${file.path}.`);
      }
      finished += file.size;
      progress(finished);
    }
    if (cancelled) throw new DownloadCancelled();
    await deviceEngine.unload(id);
    const directory = modelDirectory(id);
    if (directory.exists) directory.delete();
    partial.move(directory);
    const model = installedFrom(slug, choice);
    await changeInstalled((models) => addInstalled(models, model));
    return model;
  } catch (cause) {
    try { if (partial.exists) partial.delete(); } catch { /* removed on the next attempt */ }
    throw cause;
  } finally {
    active = null;
    notify();
  }
}

export class DownloadCancelled extends Error {
  constructor() { super('Download cancelled.'); }
}

export async function deleteModel(id: string): Promise<void> {
  await deviceEngine.unload(id);
  await changeInstalled((models) => removeInstalled(models, id));
  const directory = modelDirectory(id);
  if (directory.exists) directory.delete();
}

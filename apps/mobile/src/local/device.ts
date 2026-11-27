import * as Device from 'expo-device';
import { fetch } from 'expo/fetch';
import { Directory, File, Paths } from 'expo-file-system';
import { createDownloadResumable } from 'expo-file-system/legacy';
import { conversationStore } from './conversations.ts';
import { blobFileName, parseHistory, type Generator, type HistoryEntry } from '../media.ts';
import { availableRuntimes, localEngine, type ChatBackend, type LocalEngine, type Runtime } from './engine.ts';
import { imageEngine } from './images.ts';
import { fileUrl, isImageRepo, listRepoFiles, modelChoices, parseSlug, type ModelChoice, type RuntimeId } from './huggingface.ts';
import { IMAGE_WORKING, memoryWarning, type DeviceMemory } from './memory.ts';
import { addInstalled, installedFrom, modelId, parseInstalled, removeInstalled, type InstalledModel } from './models.ts';
import { appMemoryLimit, appleRuntime, llamaContextBytes, llamaRuntime, mlxRuntime, sdcppRuntime } from './runtimes.ts';

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
  mlxRuntime((model) => filePath(modelDirectory(model.id))),
];

const chatEngine = localEngine(runtimes, installedModels);
const imageRuntime = sdcppRuntime((model) => filePath(new File(modelDirectory(model.id), model.entry)));

// Generated images are kept under their hash, like a host's blobs, so history entries have the same shape.
const imagesRoot = () => {
  const directory = new Directory(root(), 'images');
  if (!directory.exists) directory.create({ intermediates: true, idempotent: true });
  return directory;
};
const imageFile = (sha256: string) => new File(imagesRoot(), blobFileName({ sha256, mime_type: 'image/png' }));

let replying = 0;
const images = imageEngine(imageRuntime, {
  installed: installedModels,
  output: () => filePath(new File(imagesRoot(), `${newId()}.part`)),
  keep(output, { sha256 }) {
    const written = new File(imagesRoot(), output.slice(output.lastIndexOf('/') + 1));
    const target = imageFile(sha256);
    if (target.exists) written.delete();
    else written.move(target);
  },
  // Phones rarely have memory for a chat model and an image model together.
  async beforeLoad() {
    // Unloading under a reply would end it early and save the partial text as the answer.
    if (replying > 0) throw new Error('Wait for the chat reply to finish, or stop it, before generating an image.');
    await chatEngine.unload();
  },
  now: Date.now,
  random: Math.random,
});

/** Chat models on this device. Loading one frees the image model, and the other way round. */
export const deviceEngine: LocalEngine = {
  ...chatEngine,
  async prepare(modelId, signal) { await images.unload(); return chatEngine.prepare(modelId, signal); },
  async chat(...request) {
    replying += 1;
    try {
      await images.unload();
      return await chatEngine.chat(...request);
    } finally {
      replying -= 1;
    }
  },
  async unload(modelId) { await Promise.all([chatEngine.unload(modelId), images.unload(modelId)]); },
};

const historyName = 'generate-history.json';
const listed = new Set<string>();
let saving: Promise<void> = Promise.resolve();

/** The Generate screen for this device: images only, made and kept on the phone. */
export const deviceGenerator: Generator = {
  local: true,
  models: images.models,
  generate: (_modality, body) => images.generate(body),
  active: async () => images.active(),
  cancel: async () => images.cancel(),
  async file(blob) {
    const file = imageFile(blob.sha256);
    if (!file.exists) throw new Error('This image is no longer on this device.');
    return file.uri;
  },
  async history(): Promise<HistoryEntry[]> {
    const history = parseHistory(await textFiles.read(historyName));
    for (const { blob } of history) listed.add(blobFileName(blob));
    return history;
  },
  saveHistory(history) {
    const kept = new Set(history.map(({ blob }) => blobFileName(blob)));
    saving = saving.then(() => textFiles.write(historyName, JSON.stringify(history))).then(() => {
      // An image that left the saved history (removed, or pushed out by newer ones) is deleted with it. Only
      // files the history has listed are candidates, so a new image whose entry is not saved yet survives.
      for (const name of listed) {
        if (kept.has(name)) continue;
        listed.delete(name);
        const file = new File(imagesRoot(), name);
        if (file.exists) file.delete();
      }
      for (const name of kept) listed.add(name);
    }).catch(() => undefined);
  },
};

/** Chat on this device: local runtimes, conversations kept on the phone, and no tools. */
/** What this device has for a model: the app's limit where iOS reports it, and physical memory. */
export const deviceMemory = async (): Promise<DeviceMemory> => ({ limit: await appMemoryLimit(), total: Device.totalMemory });

const contexts = new Map<string, Promise<number | null>>();

/** The context cache an installed model needs, read from its file once; null when unknown (MLX, unreadable). */
export function contextMemory(model: Pick<InstalledModel, 'id' | 'runtime' | 'entry'>): Promise<number | null> {
  if (model.runtime === 'sd.cpp') return Promise.resolve(IMAGE_WORKING);
  if (model.runtime !== 'llama.cpp') return Promise.resolve(null);
  let known = contexts.get(model.id);
  if (!known) {
    known = llamaContextBytes(filePath(new File(modelDirectory(model.id), model.entry)));
    contexts.set(model.id, known);
  }
  return known;
}

export const deviceBackend: ChatBackend = {
  ...deviceEngine, ...conversationStore(textFiles, newId), tools: async () => [],
  async loadWarning(modelId) {
    const model = (await installedModels()).find(({ id }) => id === modelId);
    return model ? memoryWarning(model.size, await deviceMemory(), await contextMemory(model)) : null;
  },
};

export const freeSpace = (): number => Paths.availableDiskSpace;

export async function usableRuntimes(): Promise<RuntimeId[]> {
  return [...(await availableRuntimes(runtimes)).map(({ id }) => id), ...(await imageRuntime.available() ? [imageRuntime.id] : [])];
}

export async function builtInModels() {
  return (await Promise.all((await availableRuntimes(runtimes)).map((runtime) => runtime.builtIn?.() ?? []))).flat();
}

export async function repositoryChoices(input: string): Promise<{ slug: string; choices: ModelChoice[] }> {
  const slug = parseSlug(input);
  const [files, image] = await Promise.all([listRepoFiles(fetch, slug), isImageRepo(fetch, slug)]);
  return { slug, choices: modelChoices(files, await usableRuntimes(), image) };
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

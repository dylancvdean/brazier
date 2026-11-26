import { fetch } from 'expo/fetch';
import { Directory, File, Paths } from 'expo-file-system';
import { createDownloadResumable } from 'expo-file-system/legacy';
import { conversationStore } from './conversations.ts';
import { availableRuntimes, localEngine, type ChatBackend, type Runtime } from './engine.ts';
import { fileUrl, listRepoFiles, modelChoices, parseSlug, type ModelChoice, type RuntimeId } from './huggingface.ts';
import { addInstalled, installedFrom, modelId, parseInstalled, removeInstalled, type InstalledModel } from './models.ts';
import { llamaRuntime } from './runtimes.ts';

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
    return file.exists ? file.text() : null;
  },
  async write(name: string, text: string) {
    const file = new File(root(), name);
    if (!file.exists) file.create({ intermediates: true });
    file.write(text);
  },
  async remove(name: string) {
    const file = new File(root(), name);
    if (file.exists) file.delete();
  },
};

const installedFile = 'models.json';
export const installedModels = async (): Promise<InstalledModel[]> => parseInstalled(await textFiles.read(installedFile));

const newId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

export const runtimes: readonly Runtime[] = [
  llamaRuntime((model) => filePath(new File(modelDirectory(model.id), model.entry))),
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

export type Download = { done: Promise<InstalledModel>; cancel(): void };

/** Files land in a `.part` directory that becomes the model's directory only once all of them are complete. */
export function downloadModel(slug: string, choice: ModelChoice, onProgress: (bytes: number) => void): Download {
  const id = modelId(slug, choice);
  const partial = new Directory(modelsRoot(), `${id}.part`);
  let cancelled = false;
  let current: ReturnType<typeof createDownloadResumable> | null = null;
  const discard = () => { if (partial.exists) partial.delete(); };
  const done = (async () => {
    discard();
    partial.create({ intermediates: true, idempotent: true });
    let finished = 0;
    for (const file of choice.files) {
      if (cancelled) break;
      const target = new File(partial, file.path);
      if (!target.parentDirectory.exists) target.parentDirectory.create({ intermediates: true, idempotent: true });
      current = createDownloadResumable(fileUrl(slug, file.path), target.uri, {},
        ({ totalBytesWritten }) => onProgress(finished + totalBytesWritten));
      const result = await current.downloadAsync();
      if (cancelled) break;
      if (!result || result.status !== 200) {
        throw new Error(`Hugging Face returned HTTP ${result?.status ?? 'no response'} for ${file.path}.`);
      }
      finished += file.size;
      onProgress(finished);
    }
    if (cancelled) throw new Error('Download cancelled.');
    await deviceEngine.unload(id);
    const directory = modelDirectory(id);
    if (directory.exists) directory.delete();
    partial.move(directory);
    const model = installedFrom(slug, choice);
    await textFiles.write(installedFile, JSON.stringify(addInstalled(await installedModels(), model)));
    return model;
  })();
  done.catch(() => { try { discard(); } catch { /* removed on the next attempt */ } });
  return { done, cancel: () => { cancelled = true; void current?.cancelAsync().catch(() => undefined); } };
}

export async function deleteModel(id: string): Promise<void> {
  await deviceEngine.unload(id);
  await textFiles.write(installedFile, JSON.stringify(removeInstalled(await installedModels(), id)));
  const directory = modelDirectory(id);
  if (directory.exists) directory.delete();
}

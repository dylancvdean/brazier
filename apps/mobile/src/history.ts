import { DaemonError, request, type Credential } from './api.ts';
import { addHistory, modelLabel, parseHistory, type HistoryEntry, type Modality } from './media.ts';

const isSha256 = (value: unknown): value is string => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);

/** An entry of a host's generation history (`GET /api/v1/generate/history`) as the Generate screen lists it. */
export function hostHistoryEntry(raw: unknown): HistoryEntry | null {
  const entry = raw as {
    id?: unknown; modality?: unknown; created_at?: unknown; metadata?: unknown;
    blob?: { sha256?: unknown; mime_type?: unknown; size_bytes?: unknown };
  } | null;
  if (typeof entry?.id !== 'string' || !entry.id || (entry.modality !== 'image' && entry.modality !== 'video') ||
      !isSha256(entry.blob?.sha256) || typeof entry.blob.mime_type !== 'string') return null;
  const reported = entry.metadata && typeof entry.metadata === 'object' ? entry.metadata as Record<string, unknown> : {};
  const number = (key: string) => {
    const value = reported[key];
    return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
  };
  const text = (key: string) => typeof reported[key] === 'string' ? reported[key] as string : '';
  const seconds = typeof entry.created_at === 'number' && Number.isFinite(entry.created_at) ? entry.created_at : 0;
  const [steps, seed, frames] = [number('steps'), number('seed'), number('video_frames')];
  return {
    id: entry.id, hostId: entry.id, modality: entry.modality, prompt: text('prompt'), negativePrompt: text('negative_prompt'),
    modelId: text('model_id'), width: number('width') ?? 0, height: number('height') ?? 0,
    ...(steps !== undefined ? { steps } : {}), ...(seed !== undefined ? { seed } : {}),
    ...(frames !== undefined ? { frames } : {}),
    createdAt: new Date(seconds * 1000).toISOString(),
    blob: {
      sha256: entry.blob.sha256, mime_type: entry.blob.mime_type,
      size_bytes: typeof entry.blob.size_bytes === 'number' ? entry.blob.size_bytes : 0,
    },
  };
}

const time = (entry: HistoryEntry) => Date.parse(entry.createdAt) || 0;

/**
 * What a host's Generate history shows on this phone: the host's own list, plus what the phone recorded that the
 * host does not list (made on a host too old to keep a history). An entry the host once listed and no longer does
 * was deleted there, so it is dropped here too. Newest first.
 */
export function mergeHistory(host: readonly HistoryEntry[], phone: readonly HistoryEntry[]): HistoryEntry[] {
  const ids = new Set(host.map(({ id }) => id));
  const own = phone.filter((entry) => !entry.hostId && !ids.has(entry.id));
  return [...host, ...own].sort((a, b) => time(b) - time(a));
}

export type HistoryFilter = 'all' | Modality;

/** Entries of one kind whose prompt, negative prompt or model contain every word of the query. */
export function searchHistory(
  history: readonly HistoryEntry[], query: string, filter: HistoryFilter = 'all',
  label: (modelId: string) => string = modelLabel,
): HistoryEntry[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  return history.filter((entry) => {
    if (filter !== 'all' && entry.modality !== filter) return false;
    if (!words.length) return true;
    const text = `${entry.prompt}\n${entry.negativePrompt}\n${entry.modelId}\n${label(entry.modelId)}`.toLowerCase();
    return words.every((word) => text.includes(word));
  });
}

const dayStart = (date: Date) => new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();

/** "Today", "Yesterday", or the date; the year only when it is not this one. */
export function dayLabel(iso: string, now: Date = new Date()): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return 'Undated';
  const days = Math.round((dayStart(now) - dayStart(date)) / 86_400_000);
  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  return date.toLocaleDateString(undefined, {
    day: 'numeric', month: 'short', ...(date.getFullYear() === now.getFullYear() ? {} : { year: 'numeric' }),
  });
}

/** The day and time of an entry, for its details. */
export function whenLabel(iso: string, now: Date = new Date()): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return 'Undated';
  return `${dayLabel(iso, now)}, ${date.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })}`;
}

/** Consecutive entries of the same day, in the order given. */
export function historyDays(history: readonly HistoryEntry[], now: Date = new Date()): { day: string; entries: HistoryEntry[] }[] {
  const days: { day: string; entries: HistoryEntry[] }[] = [];
  for (const entry of history) {
    const day = dayLabel(entry.createdAt, now);
    const last = days.at(-1);
    if (last?.day === day) last.entries.push(entry);
    else days.push({ day, entries: [entry] });
  }
  return days;
}

/** A host's generation history. A host too old to keep one answers 404, which reads as null. */
export function historyClient(fetch: Parameters<typeof request>[0], credential: Credential) {
  const call = (path: string, init: Parameters<typeof request>[3] = {}) =>
    request(fetch, credential.origin, path, init, credential.apiKey);
  const unlessMissing = async <T>(work: () => Promise<T>): Promise<T | null> => {
    try { return await work(); } catch (cause) {
      if (cause instanceof DaemonError && (cause.status === 404 || cause.status === 405)) return null;
      throw cause;
    }
  };
  return {
    list: () => unlessMissing(async () => {
      const { data } = await (await call('/api/v1/generate/history')).json() as { data?: unknown };
      return (Array.isArray(data) ? data : []).flatMap((raw) => hostHistoryEntry(raw) ?? []);
    }),
    /** Resolves also when the host no longer lists the entry. */
    async remove(id: string): Promise<void> {
      await unlessMissing(() => call(`/api/v1/generate/history/${encodeURIComponent(id)}`, { method: 'DELETE' }));
    },
  };
}

export type HistoryRemote = { list(): Promise<HistoryEntry[] | null>; remove(id: string): Promise<void> };

/**
 * The Generate history of one place, kept in one file on the phone so it is there after a restart and without the
 * host. With `remote` (a paired host) the host's list is the truth whenever it can be reached, and the file is the
 * copy shown when it cannot.
 */
export function historyStore(
  file: { read(): Promise<string | null>; write(text: string): Promise<void> }, remote?: HistoryRemote,
) {
  // Every change rewrites the whole file, so they run one at a time.
  let queue: Promise<unknown> = Promise.resolve();
  const serial = <T>(work: () => Promise<T>): Promise<T> => {
    const result = queue.then(work, work);
    queue = result.catch(() => undefined);
    return result;
  };
  const saved = async () => parseHistory(await file.read());
  const save = async (history: HistoryEntry[]) => {
    // History is a convenience: a full disk must not fail the generation or the deletion that led here.
    await file.write(JSON.stringify(history)).catch(() => undefined);
    return history;
  };
  return {
    list: (): Promise<HistoryEntry[]> => serial(async () => {
      const phone = await saved();
      if (!remote) return phone;
      let host: HistoryEntry[] | null;
      try { host = await remote.list(); } catch { return phone; }
      if (!host) return phone;
      const merged = mergeHistory(host, phone);
      return JSON.stringify(merged) === JSON.stringify(phone) ? phone : save(merged);
    }),
    add: (entry: HistoryEntry): Promise<HistoryEntry[]> => serial(async () => save(addHistory(await saved(), entry))),
    /** Fails, leaving the entry, when the host that lists it cannot be told. */
    remove: (entry: HistoryEntry): Promise<HistoryEntry[]> => serial(async () => {
      if (remote && entry.hostId) await remote.remove(entry.hostId);
      return save((await saved()).filter(({ id }) => id !== entry.id));
    }),
  };
}

import { request, type Credential, type Message, type Model, type ToolCall } from './api.ts';

export type Modality = 'image' | 'video';
export type MediaBlob = { sha256: string; mime_type: string; name?: string | null; call_id?: string };
export type GenerateResult = {
  blob: { sha256: string; mime_type: string; size_bytes: number; original_name?: string | null };
  metadata: unknown; engine: string;
  /** The entry in the host's generation history, from hosts that keep one. */
  history_id?: string | null;
};
/** A generation in flight on the host, including one a chat model started. */
export type ActiveGeneration = {
  id: string; modality: Modality; model_id: string; prompt: string; origin: 'user' | 'model';
  elapsed_secs: number; timeout_secs: number; current_step: number; total_steps: number;
};
export type GenerateForm = {
  modality: Modality; modelId: string; prompt: string; negative: string;
  width: number; height: number; steps: string; seed: string; frames: string;
};
export type GenerateBody = {
  prompt: string; model_id: string; negative_prompt?: string; width: number; height: number;
  steps?: number; seed?: number; video_frames?: number;
};
export type HistoryEntry = {
  id: string; prompt: string; negativePrompt: string; modality: Modality; modelId: string;
  width: number; height: number; steps?: number; seed?: number; frames?: number; createdAt: string;
  blob: { sha256: string; mime_type: string; size_bytes: number };
  /** Set when the paired host lists this entry in its own history; deleting it here deletes it there. */
  hostId?: string;
};

/**
 * Where the Generate screen's models, jobs, files and history come from: a paired host, or this device
 * (`local`: images only, with its own sizes and model names).
 */
export type Generator = {
  local: boolean;
  models(): Promise<Model[]>;
  /** Resolves when the job finishes; it can run for minutes. */
  generate(modality: Modality, body: GenerateBody): Promise<GenerateResult>;
  active(): Promise<ActiveGeneration | null>;
  cancel(): Promise<boolean>;
  /** URI of a local file holding the blob. */
  file(blob: MediaBlob): Promise<string>;
  /** Past generations, newest first: this device's, or the host's whichever client started them. */
  history(): Promise<HistoryEntry[]>;
  /** Both resolve to the history after the change. */
  addHistory(entry: HistoryEntry): Promise<HistoryEntry[]>;
  removeHistory(entry: HistoryEntry): Promise<HistoryEntry[]>;
};

// The desktop app's presets.
export const sizePresets: Record<Modality, readonly (readonly [number, number])[]> = {
  image: [[512, 512], [768, 768], [1024, 1024], [1024, 768], [768, 1024], [1280, 720]],
  video: [[512, 512], [640, 360], [832, 480], [480, 832], [1280, 720]],
};

// As many as a host keeps.
export const maxHistory = 500;

export function generationModels(models: readonly Model[], modality: Modality): Model[] {
  return models.filter(({ id }) => id.startsWith(`sdcpp-${modality}:`));
}

export const modelLabel = (id: string) => id.replace(/^sdcpp-(image|video):/, '');

const wholeNumber = (text: string, min: number, max: number): number | undefined => {
  if (!/^\d+$/.test(text.trim())) return undefined;
  const value = Number(text.trim());
  return value >= min && value <= max ? value : undefined;
};

/** The request for a form, or the reason it cannot be sent. Blank steps, seed and frames use the host's defaults. */
export function generateBody(form: GenerateForm): GenerateBody | string {
  const prompt = form.prompt.trim();
  if (!form.modelId) return `Choose ${form.modality === 'image' ? 'an image' : 'a video'} model.`;
  if (!prompt) return 'Enter a prompt.';
  const optional = (text: string, min: number, max: number, name: string) => {
    if (!text.trim()) return undefined;
    return wholeNumber(text, min, max) ?? `${name} must be a whole number from ${min} to ${max}.`;
  };
  const steps = optional(form.steps, 1, 200, 'Steps');
  const seed = optional(form.seed, 0, 2_147_483_647, 'Seed');
  const frames = form.modality === 'video' ? optional(form.frames, 1, 1000, 'Frames') : undefined;
  for (const value of [steps, seed, frames]) if (typeof value === 'string') return value;
  const negative = form.negative.trim();
  return {
    prompt, model_id: form.modelId, width: form.width, height: form.height,
    ...(negative ? { negative_prompt: negative } : {}),
    ...(steps !== undefined ? { steps: steps as number } : {}),
    ...(seed !== undefined ? { seed: seed as number } : {}),
    ...(frames !== undefined ? { video_frames: frames as number } : {}),
  };
}

export function historyEntry(
  fallbackId: string, modality: Modality, body: GenerateBody, result: GenerateResult, createdAt: string,
): HistoryEntry {
  const hostId = typeof result.history_id === 'string' && result.history_id ? result.history_id : undefined;
  const id = hostId ?? fallbackId;
  const reported = result.metadata && typeof result.metadata === 'object' ? result.metadata as Record<string, unknown> : {};
  const number = (key: string, fallback?: number) => {
    const value = reported[key];
    return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
  };
  return {
    id, modality, prompt: body.prompt, negativePrompt: body.negative_prompt ?? '', modelId: body.model_id,
    width: number('width', body.width)!, height: number('height', body.height)!,
    steps: number('steps', body.steps), seed: number('seed', body.seed), createdAt,
    ...(body.video_frames !== undefined ? { frames: body.video_frames } : {}),
    blob: { sha256: result.blob.sha256, mime_type: result.blob.mime_type, size_bytes: result.blob.size_bytes },
    ...(hostId ? { hostId } : {}),
  };
}

export function addHistory(history: readonly HistoryEntry[], entry: HistoryEntry): HistoryEntry[] {
  return [entry, ...history.filter(({ id }) => id !== entry.id)].slice(0, maxHistory);
}

const isSha256 = (value: unknown): value is string => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);

export function parseHistory(raw: string | null): HistoryEntry[] {
  if (!raw) return [];
  try {
    const value = JSON.parse(raw) as unknown;
    if (!Array.isArray(value)) return [];
    return (value as HistoryEntry[]).filter((entry) => typeof entry?.id === 'string' && typeof entry.prompt === 'string' &&
      (entry.modality === 'image' || entry.modality === 'video') && isSha256(entry.blob?.sha256) &&
      typeof entry.blob.mime_type === 'string').slice(0, maxHistory);
  } catch {
    return [];
  }
}

export function entryDetails(entry: HistoryEntry): string {
  const kb = entry.blob.size_bytes / 1024;
  return [
    entry.width > 0 && entry.height > 0 ? `${entry.width}×${entry.height}` : '',
    entry.steps != null ? `${entry.steps} steps` : '', entry.seed != null ? `seed ${entry.seed}` : '',
    entry.frames != null ? `${entry.frames} frames` : '',
    entry.blob.size_bytes > 0 ? kb >= 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(kb))} KB` : '',
  ].filter(Boolean).join(' · ');
}

const clock = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;

/** Fraction done (null until the first step is reported) and a one-line status. */
export function generationProgress(active: ActiveGeneration | null): { fraction: number | null; label: string } {
  if (!active) return { fraction: null, label: 'Starting…' };
  const elapsed = clock(Math.max(0, active.elapsed_secs));
  if (active.total_steps > 0 && active.current_step > 0) {
    const step = Math.min(active.current_step, active.total_steps);
    return { fraction: step / active.total_steps, label: `Step ${step} of ${active.total_steps} · ${elapsed}` };
  }
  return { fraction: null, label: `Loading model · ${elapsed}` };
}

export const isCancellation = (error: unknown) =>
  error instanceof Error && error.message.toLowerCase().includes('stopped by the user');

const extensions: Record<string, string> = {
  'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp', 'image/gif': 'gif',
  'video/mp4': 'mp4', 'video/quicktime': 'mov', 'video/webm': 'webm', 'video/x-msvideo': 'avi',
};

export const mediaKind = (mime: string): 'image' | 'video' | 'other' =>
  mime.startsWith('image/') ? 'image' : mime.startsWith('video/') ? 'video' : 'other';

/** Local file name for a blob; Photos and the share sheet need the extension. */
export function blobFileName(blob: Pick<MediaBlob, 'sha256' | 'mime_type'>): string {
  if (!isSha256(blob.sha256)) throw new Error('Invalid media reference.');
  return `${blob.sha256}.${extensions[blob.mime_type.toLowerCase()] ?? 'bin'}`;
}

/** The media parts of a saved message's content. */
export function messageBlobs(content: unknown): MediaBlob[] {
  if (!Array.isArray(content)) return [];
  return content.flatMap((part): MediaBlob[] => {
    const blob = part?.type === 'brazier_blob' ? part.brazier_blob : null;
    if (!isSha256(blob?.sha256) || typeof blob.mime_type !== 'string') return [];
    return [{
      sha256: blob.sha256, mime_type: blob.mime_type,
      ...(typeof blob.name === 'string' ? { name: blob.name } : {}),
      ...(typeof blob.call_id === 'string' ? { call_id: blob.call_id } : {}),
    }];
  });
}

/** Media of tool calls as they arrive in the chat stream, each blob once. */
export function toolCallMedia(calls: readonly ToolCall[]): MediaBlob[] {
  const seen = new Map<string, MediaBlob>();
  for (const call of calls) {
    for (const media of call.media ?? []) {
      if (!isSha256(media?.sha256) || typeof media.mime_type !== 'string') continue;
      const key = `${media.sha256}:${media.mime_type}`;
      if (!seen.has(key)) seen.set(key, { ...media, call_id: call.call_id });
    }
  }
  return [...seen.values()];
}

/**
 * The message that keeps a turn's generated media in the conversation, in the
 * shape the desktop app saves so both show it. Null when the turn made none.
 */
export function generatedMediaMessage(calls: readonly ToolCall[]) {
  const media = toolCallMedia(calls);
  if (!media.length) return null;
  return {
    role: 'assistant', source: 'assistant_chat', metadata: { generated_media_display: true },
    content: media.map((blob, index) => ({
      type: 'brazier_blob',
      brazier_blob: {
        sha256: blob.sha256, mime_type: blob.mime_type, call_id: blob.call_id,
        name: blob.name ?? (blob.mime_type === 'application/pdf' ? `document-${index + 1}.pdf`
          : blob.mime_type.startsWith('video/') ? `generated-video-${index + 1}.mp4` : `generated-image-${index + 1}.png`),
      },
    })),
  };
}

/** The host opens the media it hands back with a notice; a person's own message does not. */
function isHostNotice(content: unknown): boolean {
  const first = Array.isArray(content) ? content[0] : null;
  return first?.type === 'text' && typeof first.text === 'string' && first.text.startsWith('The requested ');
}

/**
 * Where each saved message's media is shown. Within an assistant turn a blob
 * appears once, under the message whose tool call produced it; `moved` holds
 * the messages that only carried media shown elsewhere.
 */
export function mediaByMessage(messages: readonly Message[]): { media: Map<string, MediaBlob[]>; moved: Set<string> } {
  const media = new Map<string, MediaBlob[]>();
  const moved = new Set<string>();
  let seen = new Set<string>();
  let callOwner = new Map<string, string>();
  let lastCaller: string | null = null;
  let lastAssistant: string | null = null;
  const attach = (id: string, blob: MediaBlob) => media.set(id, [...(media.get(id) ?? []), blob]);
  let afterTool = false;
  for (const message of messages) {
    const blobs = messageBlobs(message.content);
    // For a vision model the host follows a tool result with a user-role message
    // that hands the model the media it made. It is part of the turn, not an upload.
    const hostContext = message.role === 'user' && afterTool && blobs.length > 0 && isHostNotice(message.content);
    afterTool = message.role === 'tool';
    if (hostContext) moved.add(message.id);
    else if (message.role === 'user') {
      seen = new Set(); callOwner = new Map(); lastCaller = null; lastAssistant = null;
      if (blobs.length) media.set(message.id, blobs);
      continue;
    }
    if (message.role === 'assistant') {
      lastAssistant = message.id;
      for (const call of message.tool_calls ?? []) {
        const id = (call as { id?: unknown })?.id;
        if (typeof id === 'string') callOwner.set(id, message.id);
      }
      if (message.tool_calls?.length) lastCaller = message.id;
    }
    let elsewhere = false;
    for (const blob of blobs) {
      const key = `${blob.sha256}:${blob.mime_type}`;
      if (seen.has(key)) { elsewhere = true; continue; }
      seen.add(key);
      const target = (blob.call_id ? callOwner.get(blob.call_id) : undefined) ?? lastCaller ?? lastAssistant;
      if (!target) continue;
      if (target !== message.id) elsewhere = true;
      attach(target, blob);
    }
    if (elsewhere && !media.has(message.id)) moved.add(message.id);
  }
  for (const id of media.keys()) moved.delete(id);
  return { media, moved };
}

export function mediaClient(fetch: Parameters<typeof request>[0], credential: Credential) {
  const call = (path: string, init: Parameters<typeof request>[3] = {}) =>
    request(fetch, credential.origin, path, init, credential.apiKey);
  return {
    async models(): Promise<Model[]> {
      return (await (await call('/v1/models')).json() as { data: Model[] }).data
        .filter(({ id }) => /^sdcpp-(image|video):/.test(id));
    },
    /** Resolves when the job finishes; it can run for minutes. */
    async generate(modality: Modality, body: GenerateBody): Promise<GenerateResult> {
      const result = await (await call(`/api/v1/generate/${modality}`, { method: 'POST', body: JSON.stringify(body) }))
        .json() as GenerateResult;
      if (!isSha256(result?.blob?.sha256) || typeof result.blob.mime_type !== 'string') {
        throw new Error('The host did not return the generated file.');
      }
      return result;
    },
    async active(): Promise<ActiveGeneration | null> {
      return (await (await call('/api/v1/generate/active')).json() as { active: ActiveGeneration | null }).active ?? null;
    },
    async cancel(): Promise<boolean> {
      return (await (await call('/api/v1/generate/cancel', { method: 'POST' })).json() as { cancelled: boolean }).cancelled;
    },
    async blob(sha256: string, signal?: AbortSignal): Promise<ReadableStream<Uint8Array>> {
      if (!isSha256(sha256)) throw new Error('Invalid media reference.');
      const response = await call(`/api/v1/blobs/${sha256}`, { signal });
      if (!response.body) throw new Error('The host did not return the file.');
      return response.body;
    },
  };
}

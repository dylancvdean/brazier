import { modelChoices, parseSlug, type Fetch, type RuntimeId } from './huggingface.ts';

/** What to look for: GGUF chat models (llama.cpp), MLX repositories, or image checkpoints (stable-diffusion.cpp). */
export type SearchFormat = 'gguf' | 'mlx' | 'image';
export type SearchSort = 'downloads' | 'trendingScore' | 'likes' | 'lastModified';
export type SearchRequest = { query: string; format: SearchFormat; sort: SearchSort };
/** `files` is how many downloads the repository offers this device, such as its quantisations. */
export type SearchResult = { slug: string; format: SearchFormat; downloads: number; likes: number; updated: string | null; files: number };
export type SearchPage = { results: SearchResult[]; next: string | null };

export const sortNames: Record<SearchSort, string> = {
  downloads: 'Most downloaded', trendingScore: 'Trending', likes: 'Most liked', lastModified: 'Recently updated',
};
export const formatNames: Record<SearchFormat, string> = { gguf: 'Chat · GGUF', mlx: 'Chat · MLX', image: 'Images' };

const formatRuntime: Record<SearchFormat, RuntimeId> = { gguf: 'llama.cpp', mlx: 'mlx', image: 'sd.cpp' };

/** The searches this device can use the results of. */
export const searchFormats = (runtimes: readonly RuntimeId[]): SearchFormat[] =>
  (['gguf', 'mlx', 'image'] as const).filter((format) => runtimes.includes(formatRuntime[format]));

/** The repository a query names outright, as `owner/name` or a huggingface.co link, so it can be opened without a search. */
export function namedRepository(query: string): string | null {
  try {
    return parseSlug(query);
  } catch {
    return null;
  }
}

const origin = 'https://huggingface.co';
const pageSize = 20;
const fields = ['siblings', 'downloads', 'likes', 'pipeline_tag', 'tags', 'lastModified', 'gated'];

export function searchUrl({ query, format, sort }: SearchRequest, tag = format === 'image' ? 'text-to-image' : format): string {
  const parameters = [
    ...(query.trim() ? [`search=${encodeURIComponent(query.trim())}`] : []),
    `filter=${tag}`,
    `sort=${sort}`, 'direction=-1', `limit=${pageSize}`,
    ...fields.map((field) => `expand[]=${field}`),
  ];
  return `${origin}/api/models?${parameters.join('&')}`;
}

// Repositories made for stable-diffusion.cpp are tagged with it, and often not as text-to-image.
const sdcpp = 'stable-diffusion.cpp';
// A GGUF or MLX repository with one of these is not a chat model.
const notChat = new Set([
  'text-to-image', 'image-to-image', 'text-to-video', 'image-to-video', 'text-to-speech', 'text-to-audio',
  'automatic-speech-recognition', 'audio-to-audio', 'feature-extraction', 'sentence-similarity', 'text-ranking',
  'image-text-to-image', 'image-text-to-video', 'image-classification', 'object-detection', 'image-segmentation',
  'stable-diffusion', sdcpp, 'sd.cpp',
]);
// Speech and embedding repositories often carry no pipeline tag, only these.
const otherKind = /^(mlx-audio|asr|stt|tts|speech|speech-to-text|speech-recognition|transcription|whisper|audio|embeddings?|reranker)$/i;
// Add-ons to an image model, which generate nothing on their own.
const addOn = /controlnet|(^|[^a-z])lora([^a-z]|$)|adapter/i;
// Image families whose checkpoints need separate text-encoder or VAE files, which this app cannot load.
const needsMoreFiles = /flux|sd3|stable-diffusion-3|qwen-image|z-image|hunyuan|chroma|kontext|lumina|pixart|auraflow|cascade|wan\d|hidream|omnigen|sana/i;

/**
 * The repositories of one page that this device can download from. Gated and private repositories, ones of
 * another kind (speech, embeddings, video) and ones with no loadable file are left out.
 */
export function parseResults(value: unknown, format: SearchFormat): SearchResult[] {
  if (!Array.isArray(value)) throw new Error('Hugging Face returned an unexpected search result.');
  return value.flatMap((entry): SearchResult[] => {
    const model = entry as {
      id?: unknown; gated?: unknown; private?: unknown; downloads?: unknown; likes?: unknown; lastModified?: unknown;
      pipeline_tag?: unknown; tags?: unknown; siblings?: unknown;
    };
    if (typeof model?.id !== 'string' || !/^[^/\s]+\/[^/\s]+$/.test(model.id) || model.private === true ||
        (model.gated !== undefined && model.gated !== false)) return [];
    const tags = [model.pipeline_tag, ...(Array.isArray(model.tags) ? model.tags : [])].filter((tag) => typeof tag === 'string');
    const named = `${model.id} ${tags.join(' ')}`;
    if (format === 'image' ? needsMoreFiles.test(named) || addOn.test(named)
      : tags.some((tag) => notChat.has(tag) || otherKind.test(tag))) return [];
    const files = (Array.isArray(model.siblings) ? model.siblings : []).flatMap((file) => {
      const path = (file as { rfilename?: unknown })?.rfilename;
      return typeof path === 'string' ? [{ path, size: 0 }] : [];
    });
    const count = modelChoices(files, [formatRuntime[format]], format === 'image').length;
    if (count === 0) return [];
    const number = (field: unknown) => typeof field === 'number' && field >= 0 ? field : 0;
    return [{
      slug: model.id, format, downloads: number(model.downloads), likes: number(model.likes), files: count,
      updated: typeof model.lastModified === 'string' ? model.lastModified : null,
    }];
  });
}

const nextPage = (link: string | null) => {
  const next = /<([^>]+)>;\s*rel="next"/.exec(link ?? '')?.[1];
  return next?.startsWith(`${origin}/api/models?`) ? next : null;
};

/**
 * One page of results: the first for a request, or the page at `next` from the page before. Pages whose
 * repositories are all filtered out are skipped, a few at a time.
 */
export async function searchModels(fetch: Fetch, request: SearchRequest, next?: string | null): Promise<SearchPage> {
  const read = async (url: string) => {
    let response: Awaited<ReturnType<Fetch>>;
    try {
      response = await fetch(url, { headers: { accept: 'application/json' } });
    } catch {
      throw new Error('Could not reach Hugging Face. Check the connection and try again.');
    }
    if (response.status === 429) throw new Error('Hugging Face is limiting searches. Wait a moment and try again.');
    if (!response.ok) throw new Error(`Hugging Face returned HTTP ${response.status}.`);
    return { results: parseResults(await response.json(), request.format), next: nextPage(response.headers.get('link')) };
  };
  // The first page of an image search leads with the repositories made for stable-diffusion.cpp.
  const made = !next && request.format === 'image' ? (await read(searchUrl(request, sdcpp))).results : [];
  let url: string | null = next ?? searchUrl(request);
  let results: SearchResult[] = [];
  for (let page = 0; url && page < 4 && results.length === 0; page += 1) {
    const found = await read(url);
    results = found.results;
    url = found.next;
  }
  return { results: mergeResults(made, results), next: url };
}

/** Adds a page to the results shown, without repeating a repository. */
export function mergeResults(shown: readonly SearchResult[], page: readonly SearchResult[]): SearchResult[] {
  const seen = new Set(shown.map(({ slug }) => slug));
  return [...shown, ...page.filter(({ slug }) => !seen.has(slug) && seen.add(slug))];
}

/** 1234567 as 1.2M. */
export function formatCount(count: number): string {
  if (!Number.isFinite(count) || count < 0) return '0';
  for (const [unit, size] of [['M', 1e6], ['k', 1e3]] as const) {
    if (count >= size) return `${(count / size).toFixed(count >= size * 100 ? 0 : 1).replace(/\.0$/, '')}${unit}`;
  }
  return String(Math.round(count));
}

/** How long ago an ISO date was, in the largest whole unit. */
export function formatAge(iso: string | null, now: number): string {
  const then = iso ? Date.parse(iso) : NaN;
  if (!Number.isFinite(then)) return '';
  const days = Math.max(0, Math.floor((now - then) / 86_400_000));
  if (days < 1) return 'today';
  if (days < 30) return `${days} d ago`;
  if (days < 365) return `${Math.floor(days / 30)} mo ago`;
  return `${Math.floor(days / 365)} y ago`;
}

/**
 * Runs `search` only once typing has paused for `delay`, and reports only the newest search: an answer that
 * arrives after a later request, or after `cancel`, is dropped.
 */
export function debouncedSearch<Request, Result>(
  search: (request: Request) => Promise<Result>,
  report: (outcome: { request: Request; result?: Result; error?: unknown }) => void,
  delay: number,
  timers: { set(run: () => void, ms: number): unknown; clear(timer: unknown): void } = {
    set: (run, ms) => setTimeout(run, ms), clear: (timer) => clearTimeout(timer as ReturnType<typeof setTimeout>),
  },
) {
  let latest = 0;
  let timer: unknown = null;
  const run = (request: Request, turn: number) => {
    search(request).then(
      (result) => { if (turn === latest) report({ request, result }); },
      (error) => { if (turn === latest) report({ request, error }); });
  };
  return {
    /** `now` skips the pause, for a changed filter or a submitted query. */
    request(request: Request, now = false) {
      const turn = latest += 1;
      if (timer !== null) timers.clear(timer);
      timer = null;
      if (now) run(request, turn);
      else timer = timers.set(() => { timer = null; run(request, turn); }, delay);
    },
    cancel() {
      latest += 1;
      if (timer !== null) timers.clear(timer);
      timer = null;
    },
  };
}

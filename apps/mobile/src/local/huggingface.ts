export type RuntimeId = 'llama.cpp' | 'mlx' | 'apple' | 'sd.cpp';
export type RepoFile = { path: string; size: number };
/** Something in a repository that one runtime can load: a GGUF (with its parts), a whole MLX repository or an image checkpoint. */
export type ModelChoice = { runtime: RuntimeId; name: string; entry: string; files: RepoFile[]; size: number };

export type Fetch = (url: string, init?: { headers?: Record<string, string> }) => Promise<{
  ok: boolean; status: number; json(): Promise<unknown>; headers: { get(name: string): string | null };
}>;

const origin = 'https://huggingface.co';
const segment = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

/** Accepts `owner/name` or a huggingface.co model link. */
export function parseSlug(input: string): string {
  let text = input.trim();
  const link = /^(?:https?:\/\/)?(?:www\.)?(?:huggingface\.co|hf\.co)\/(.+)$/i.exec(text);
  if (link) text = link[1].split(/[?#]/)[0];
  const parts = text.split('/').filter(Boolean);
  if (parts.length < 2 || !segment.test(parts[0]) || !segment.test(parts[1]) || (!link && parts.length !== 2)) {
    throw new Error('Enter a Hugging Face model as owner/name.');
  }
  return `${parts[0]}/${parts[1]}`;
}

const encodePath = (path: string) => path.split('/').map(encodeURIComponent).join('/');

export const fileUrl = (slug: string, path: string) => `${origin}/${slug}/resolve/main/${encodePath(path)}`;

export function parseTree(value: unknown): RepoFile[] {
  if (!Array.isArray(value)) throw new Error('Hugging Face returned an unexpected file list.');
  return value.flatMap((entry): RepoFile[] => {
    const { type, path, size, lfs } = entry as { type?: unknown; path?: unknown; size?: unknown; lfs?: { size?: unknown } };
    if (type !== 'file' || typeof path !== 'string' || !path || path.split('/').some((part) => part === '..' || part === '')) return [];
    const bytes = typeof lfs?.size === 'number' ? lfs.size : size;
    return [{ path, size: typeof bytes === 'number' && bytes >= 0 ? bytes : 0 }];
  });
}

const nextPage = (link: string | null) => /<([^>]+)>;\s*rel="next"/.exec(link ?? '')?.[1];

export async function listRepoFiles(fetch: Fetch, slug: string): Promise<RepoFile[]> {
  const files: RepoFile[] = [];
  let url: string | undefined = `${origin}/api/models/${slug}/tree/main?recursive=true`;
  for (let page = 0; url && page < 20; page += 1) {
    let response: Awaited<ReturnType<Fetch>>;
    try {
      response = await fetch(url, { headers: { accept: 'application/json' } });
    } catch {
      throw new Error('Could not reach Hugging Face. Check the connection and try again.');
    }
    if (response.status === 404) throw new Error(`No model named ${slug} on Hugging Face.`);
    if (response.status === 401 || response.status === 403) {
      throw new Error(`${slug} is private or gated. Only public models can be added.`);
    }
    if (!response.ok) throw new Error(`Hugging Face returned HTTP ${response.status}.`);
    files.push(...parseTree(await response.json()));
    const next = nextPage(response.headers.get('link'));
    url = next?.startsWith(`${origin}/`) ? next : undefined;
  }
  return files;
}

const imageTags = ['text-to-image', 'image-to-image', 'stable-diffusion', 'stable-diffusion.cpp', 'sd.cpp', 'diffusers'];

/**
 * Whether the repository holds image models, from its Hugging Face tags. A GGUF does not say in its name whether
 * it is a chat model or an image model. False when the tags cannot be read.
 */
export async function isImageRepo(fetch: Fetch, slug: string): Promise<boolean> {
  try {
    const response = await fetch(`${origin}/api/models/${slug}`, { headers: { accept: 'application/json' } });
    if (!response.ok) return false;
    const { pipeline_tag: pipeline, tags } = await response.json() as { pipeline_tag?: unknown; tags?: unknown };
    return [pipeline, ...(Array.isArray(tags) ? tags : [])].some((tag) => imageTags.includes(tag as string));
  } catch {
    return false;
  }
}

const baseName = (path: string) => path.slice(path.lastIndexOf('/') + 1);
const split = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/i;
const mlxFiles = /(\.safetensors|\.json|\.jinja|\.model|\.txt)$/i;

// Parts of a model that stable-diffusion.cpp cannot run on their own.
const imagePart = /(^|[-_.])(vae|taesd|lora|clip|t5|t5xxl|text[-_]encoders?|control|controlnet|unet|adapter|mmproj)([-_.]|$)|^diffusion_pytorch_model/i;
// A prediction head for speculative decoding: a small file that only works beside its base model. A full model
// that embeds one can carry the same word in its name, so only a file led by it or under 1 GB is taken for a head.
const head = /(^|[-_.])mtp([-_.]|$)/i;
const isHead = (name: string, size: number) => head.test(name) && (/^mtp/i.test(name) || (size > 0 && size < 1e9));

/**
 * What the listed files offer each runtime. A split GGUF is one choice; projector and prediction-head files are not chat models.
 * In an image repository (`image`) the single-file checkpoints, a GGUF or a `.safetensors` at the top level,
 * are image models and nothing in it is a chat model.
 */
export function modelChoices(files: readonly RepoFile[], runtimes: readonly RuntimeId[], image = false): ModelChoice[] {
  const choices: ModelChoice[] = [];
  if (image) {
    if (!runtimes.includes('sd.cpp')) return [];
    return files
      .filter(({ path }) => (/\.gguf$/i.test(path) || (/\.safetensors$/i.test(path) && !path.includes('/'))) &&
        !split.test(path) && !imagePart.test(baseName(path)))
      .map((file): ModelChoice => ({ runtime: 'sd.cpp', entry: file.path, files: [file], name: baseName(file.path), size: file.size }))
      .sort((a, b) => a.size - b.size || a.name.localeCompare(b.name));
  }
  if (runtimes.includes('llama.cpp')) {
    const parts = new Map<string, RepoFile[]>();
    for (const file of files) {
      const name = baseName(file.path);
      if (!/\.gguf$/i.test(name) || /mmproj/i.test(name) || isHead(name, file.size)) continue;
      const match = split.exec(file.path);
      const key = match ? `${match[1]}|${match[3]}` : file.path;
      parts.set(key, [...parts.get(key) ?? [], file]);
    }
    for (const group of parts.values()) {
      const sorted = [...group].sort((a, b) => a.path.localeCompare(b.path));
      const match = split.exec(sorted[0].path);
      if (match && sorted.length !== Number(match[3])) continue;
      choices.push({
        runtime: 'llama.cpp', entry: sorted[0].path, files: sorted,
        name: match ? `${baseName(match[1])}.gguf` : baseName(sorted[0].path),
        size: sorted.reduce((total, file) => total + file.size, 0),
      });
    }
    choices.sort((a, b) => a.size - b.size || a.name.localeCompare(b.name));
  }
  const root = files.filter((file) => !file.path.includes('/'));
  if (runtimes.includes('mlx') && root.some((file) => file.path === 'config.json') &&
      root.some((file) => /\.safetensors$/i.test(file.path))) {
    const needed = root.filter((file) => mlxFiles.test(file.path));
    choices.push({
      runtime: 'mlx', name: 'MLX model', entry: '', files: needed,
      size: needed.reduce((total, file) => total + file.size, 0),
    });
  }
  return choices;
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) { value /= 1000; unit += 1; }
  return `${unit === 0 || value >= 100 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

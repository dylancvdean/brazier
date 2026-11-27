import { modelChoices, type ModelChoice, type RepoFile, type RuntimeId } from './huggingface.ts';
import { IMAGE_WORKING, memoryBudget, memoryNeeded, type DeviceMemory } from './memory.ts';
import { modelId } from './models.ts';

/**
 * A model the app suggests. `entry` is the file to download, or '' for a whole MLX repository. `size` is the
 * download in bytes. `rank` orders models of one kind by how good their answers or images are; it is a
 * judgement, higher is better. `scripts/verify-catalogue.ts` checks every entry against Hugging Face.
 */
export type CatalogueEntry = {
  id: string; kind: 'chat' | 'image'; runtime: Exclude<RuntimeId, 'apple'>; slug: string; entry: string;
  name: string; detail: string; description: string; size: number; rank: number;
};

const gguf = (id: string, slug: string, entry: string, name: string, parameters: string, size: number, rank: number,
  description: string): CatalogueEntry => ({
  id, kind: 'chat', runtime: 'llama.cpp', slug, entry, name, size, rank, description,
  detail: `${parameters}  ·  ${/-(Q\d\w*)\.gguf$/.exec(entry)?.[1] ?? 'GGUF'}`,
});
// An MLX build ranks just under the GGUF of the same model, which is the path this app exercises most.
const mlx = (id: string, slug: string, name: string, parameters: string, size: number, rank: number,
  description: string): CatalogueEntry => ({
  id, kind: 'chat', runtime: 'mlx', slug, entry: '', name, size, rank: rank - 0.5, description, detail: `${parameters}  ·  4-bit`,
});
const image = (id: string, slug: string, entry: string, name: string, detail: string, size: number, rank: number,
  description: string): CatalogueEntry => ({ id, kind: 'image', runtime: 'sd.cpp', slug, entry, name, detail, size, rank, description });

/** To add a model: add a line here, then run `pnpm --filter @brazier/mobile catalogue:verify`. */
export const catalogue: readonly CatalogueEntry[] = [
  gguf('qwen3.5-9b', 'unsloth/Qwen3.5-9B-GGUF', 'Qwen3.5-9B-Q4_K_M.gguf', 'Qwen3.5 9B', '9B', 5_680_522_464, 90,
    'The strongest pick: better reasoning and writing, for devices with plenty of memory. Slower to reply.'),
  gguf('qwen3.5-4b', 'unsloth/Qwen3.5-4B-GGUF', 'Qwen3.5-4B-Q4_K_M.gguf', 'Qwen3.5 4B', '4B', 2_740_937_888, 80,
    'Capable all-rounder with step-by-step thinking, in about 200 languages.'),
  gguf('qwen3-4b', 'unsloth/Qwen3-4B-GGUF', 'Qwen3-4B-Q4_K_M.gguf', 'Qwen3 4B', '4B', 2_497_281_312, 74,
    'Well-tested thinking model; good at reasoning, code and multilingual chat.'),
  gguf('gemma-3-4b', 'unsloth/gemma-3-4b-it-GGUF', 'gemma-3-4b-it-Q4_K_M.gguf', 'Gemma 3 4B', '4B', 2_489_894_016, 72,
    'Google\'s model: natural, friendly writing and quick direct answers.'),
  gguf('phi-4-mini', 'unsloth/Phi-4-mini-instruct-GGUF', 'Phi-4-mini-instruct-Q4_K_M.gguf', 'Phi-4 mini', '3.8B', 2_491_874_272, 70,
    'Microsoft\'s compact model: strong at maths, logic and structured answers.'),
  gguf('llama-3.2-3b', 'bartowski/Llama-3.2-3B-Instruct-GGUF', 'Llama-3.2-3B-Instruct-Q4_K_M.gguf', 'Llama 3.2 3B', '3B', 2_019_377_696, 62,
    'Meta\'s model: dependable everyday chat and summaries at a moderate size.'),
  gguf('qwen3.5-2b', 'unsloth/Qwen3.5-2B-GGUF', 'Qwen3.5-2B-Q4_K_M.gguf', 'Qwen3.5 2B', '2B', 1_280_835_840, 56,
    'Small thinking model that stays quick; a good fit for most phones.'),
  gguf('qwen3-1.7b', 'unsloth/Qwen3-1.7B-GGUF', 'Qwen3-1.7B-Q4_K_M.gguf', 'Qwen3 1.7B', '1.7B', 1_107_409_472, 50,
    'Small thinking model; fine for questions, rewriting and short code.'),
  gguf('lfm2.5-1.2b', 'LiquidAI/LFM2.5-1.2B-Instruct-GGUF', 'LFM2.5-1.2B-Instruct-Q4_K_M.gguf', 'LFM2.5 1.2B', '1.2B', 730_895_168, 46,
    'Liquid AI\'s model built for phones: very fast replies for its quality.'),
  gguf('gemma-3-1b', 'unsloth/gemma-3-1b-it-GGUF', 'gemma-3-1b-it-Q4_K_M.gguf', 'Gemma 3 1B', '1B', 806_058_272, 42,
    'Tiny Google model for short questions and simple rewriting.'),
  gguf('llama-3.2-1b', 'bartowski/Llama-3.2-1B-Instruct-GGUF', 'Llama-3.2-1B-Instruct-Q4_K_M.gguf', 'Llama 3.2 1B', '1B', 807_694_464, 40,
    'Tiny Meta model for quick, simple chat.'),
  gguf('qwen3.5-0.8b', 'unsloth/Qwen3.5-0.8B-GGUF', 'Qwen3.5-0.8B-Q4_K_M.gguf', 'Qwen3.5 0.8B', '0.8B', 532_517_120, 36,
    'Very small thinking model for phones short on memory or storage.'),
  gguf('qwen3-0.6b', 'unsloth/Qwen3-0.6B-GGUF', 'Qwen3-0.6B-Q4_K_M.gguf', 'Qwen3 0.6B', '0.6B', 396_705_472, 30,
    'Very small and fast; simple questions only.'),
  gguf('lfm2.5-350m', 'LiquidAI/LFM2.5-350M-GGUF', 'LFM2.5-350M-Q4_K_M.gguf', 'LFM2.5 350M', '350M', 229_312_224, 20,
    'The smallest download; runs anywhere, for the simplest tasks.'),

  mlx('mlx-qwen3-4b', 'mlx-community/Qwen3-4B-4bit', 'Qwen3 4B (MLX)', '4B', 2_278_969_756, 74,
    'Well-tested thinking model on Apple\'s MLX; good at reasoning, code and multilingual chat.'),
  mlx('mlx-llama-3.2-3b', 'mlx-community/Llama-3.2-3B-Instruct-4bit', 'Llama 3.2 3B (MLX)', '3B', 1_824_807_894, 62,
    'Meta\'s model on Apple\'s MLX: dependable everyday chat and summaries.'),
  mlx('mlx-qwen3-1.7b', 'mlx-community/Qwen3-1.7B-4bit', 'Qwen3 1.7B (MLX)', '1.7B', 984_013_244, 50,
    'Small thinking model on Apple\'s MLX; fine for questions, rewriting and short code.'),
  mlx('mlx-gemma-3-1b', 'mlx-community/gemma-3-1b-it-qat-4bit', 'Gemma 3 1B (MLX)', '1B', 771_860_249, 42,
    'Tiny Google model on Apple\'s MLX for short questions and simple rewriting.'),
  mlx('mlx-llama-3.2-1b', 'mlx-community/Llama-3.2-1B-Instruct-4bit', 'Llama 3.2 1B (MLX)', '1B', 712_575_975, 40,
    'Tiny Meta model on Apple\'s MLX for quick, simple chat.'),
  mlx('mlx-qwen3-0.6b', 'mlx-community/Qwen3-0.6B-4bit', 'Qwen3 0.6B (MLX)', '0.6B', 351_383_618, 30,
    'Very small and fast on Apple\'s MLX; simple questions only.'),

  image('sd-turbo', 'Green-Sky/SD-Turbo-GGUF', 'sd_turbo-f16-q8_0.gguf', 'SD-Turbo', '512×512  ·  4 steps', 2_023_745_376, 60,
    'Fastest: a usable image in a few steps. Best first choice on a phone.'),
  image('sd-1.5', 'second-state/stable-diffusion-v1-5-GGUF', 'stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf', 'Stable Diffusion 1.5',
    '512×512  ·  20 steps', 1_763_578_176, 40, 'The classic general-purpose model and the smaller download; follows negative prompts, but takes several times longer.'),
];

/** `tight` loads but leaves little room; `unknown` when the device does not say how much memory it has. */
export type Fit = 'fits' | 'tight' | 'too-large' | 'unknown';

// A model that needs more than this share of the budget is offered, but not as the best pick.
const HEADROOM = 0.85;
// Without a memory figure the best pick stays small enough for a phone with 4 GB.
const UNKNOWN_BEST = 1.0e9;

/** How a model of `size` bytes on disk sits in this device's memory, by the same estimate as the load warning. */
export function memoryFit(size: number, memory: DeviceMemory, context?: number | null): Fit {
  const budget = memoryBudget(memory);
  if (budget === null) return 'unknown';
  const needed = memoryNeeded(size, context);
  return needed > budget ? 'too-large' : needed > budget * HEADROOM ? 'tight' : 'fits';
}

/** The working memory to assume for a download that is not on the device yet. */
export const choiceContext = (runtime: RuntimeId): number | undefined => runtime === 'sd.cpp' ? IMAGE_WORKING : undefined;

export type DeviceFacts = { runtimes: readonly RuntimeId[]; memory: DeviceMemory; free: number; installed?: readonly string[] };
export type Recommendation = { entry: CatalogueEntry; modelId: string; fit: Fit; space: boolean; installed: boolean; best: boolean };

/**
 * The catalogue for this device: only models a usable runtime can load, the ones that fit first and the
 * better ones before the smaller ones, then the ones that are too large, smallest first. One model of each
 * kind is the best pick: the highest ranked that fits with room to spare and that there is storage for.
 */
export function recommend(entries: readonly CatalogueEntry[], device: DeviceFacts): Recommendation[] {
  const all = entries.filter(({ runtime }) => device.runtimes.includes(runtime)).map((entry): Recommendation => {
    const id = modelId(entry.slug, entry);
    const installed = device.installed?.includes(id) ?? false;
    return {
      entry, modelId: id, installed, best: false,
      fit: memoryFit(entry.size, device.memory, choiceContext(entry.runtime)), space: installed || entry.size <= device.free,
    };
  });
  const usable = (item: Recommendation) => item.fit !== 'too-large';
  const order = (a: Recommendation, b: Recommendation) => Number(usable(b)) - Number(usable(a)) ||
    (usable(a) ? b.entry.rank - a.entry.rank : a.entry.size - b.entry.size);
  for (const kind of ['chat', 'image'] as const) {
    const candidates = all.filter((item) => item.entry.kind === kind && usable(item) && item.space).sort(order);
    const best = candidates.find((item) => item.fit === 'fits') ??
      candidates.find((item) => item.fit === 'unknown' && item.entry.size <= UNKNOWN_BEST) ??
      candidates.find((item) => item.fit === 'tight') ?? candidates.at(-1);
    if (best) best.best = true;
  }
  return all.sort((a, b) => Number(b.best) - Number(a.best) || order(a, b));
}

/** The catalogue entry among a repository's files, with the sizes Hugging Face reports now. */
export function catalogueChoice(entry: CatalogueEntry, files: readonly RepoFile[]): ModelChoice | null {
  return modelChoices(files, [entry.runtime], entry.kind === 'image')
    .find((choice) => choice.runtime === entry.runtime && choice.entry === entry.entry) ?? null;
}

const quantisationPattern = /(?:^|[-._])((?:i?q\d(?:_[a-z0-9]+)*)|f16|f32|bf16|fp16|fp32|\d(?:\.\d+)?bpw|\dbit)(?=[-._]|$)/gi;

/** The quantisation a file name states, such as Q4_K_M, or null. */
export function quantisation(name: string): string | null {
  const found = [...name.replace(/\.(gguf|safetensors)$/i, '').matchAll(quantisationPattern)].at(-1)?.[1];
  return found ? found.toUpperCase().replace('BPW', ' bpw').replace('BIT', '-bit') : null;
}

/**
 * The file to suggest in a repository: Q4_K_M, the usual balance of size and quality, when it fits; otherwise
 * the largest smaller file that does. Null when nothing fits or the repository has no Q4_K_M to measure against.
 */
export function suggestedChoice(choices: readonly ModelChoice[], memory: DeviceMemory, free: number): ModelChoice | null {
  const fits = (choice: ModelChoice) => choice.size <= free &&
    memoryFit(choice.size, memory, choiceContext(choice.runtime)) !== 'too-large';
  const sized = choices.filter((choice) => choice.size > 0);
  if (sized.length === 1) return fits(sized[0]) ? sized[0] : null;
  const usual = sized.filter((choice) => quantisation(choice.name) === 'Q4_K_M').sort((a, b) => a.size - b.size)[0];
  if (!usual) return null;
  if (fits(usual)) return usual;
  return sized.filter((choice) => choice.size < usual.size && fits(choice)).sort((a, b) => b.size - a.size)[0] ?? null;
}

/** What to say about a download's fit, and whether it is a reason not to download: storage first, then memory. */
export function fitNote(fit: Fit, space: boolean): { text: string; problem: boolean } | null {
  if (!space) return { text: 'Not enough free space', problem: true };
  if (fit === 'too-large') return { text: 'Too large for this device\'s memory', problem: true };
  if (fit === 'tight') return { text: 'Fits, with little memory to spare', problem: false };
  return fit === 'fits' ? { text: 'Fits this device', problem: false } : null;
}

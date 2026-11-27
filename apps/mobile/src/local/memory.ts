import { formatBytes } from './huggingface.ts';

/** The context length every on-device model is loaded with. */
export const CONTEXT_TOKENS = 4096;

// The context cache of a typical grouped-query model at CONTEXT_TOKENS; used until the model's own
// figures can be read from its file.
const TYPICAL_CONTEXT = 0.6e9;
// Working buffers of the runtime plus the app itself.
const WORKING = 0.25e9;
// Where the system does not say how much one app may use, half of physical memory.
const FALLBACK_SHARE = 0.5;

const positive = (value: unknown): value is number => typeof value === 'number' && value > 0 && Number.isFinite(value);

/**
 * Bytes of the context (KV) cache from a GGUF file's metadata: keys and values, 16-bit, for every
 * layer and key-value head. Null when the metadata does not have the figures.
 */
export function contextBytes(metadata: Record<string, unknown> | null | undefined, tokens = CONTEXT_TOKENS): number | null {
  if (!metadata) return null;
  const architecture = metadata['general.architecture'];
  if (typeof architecture !== 'string') return null;
  const number = (key: string) => Number(metadata[`${architecture}.${key}`]);
  const layers = number('block_count');
  const heads = number('attention.head_count');
  const keyValueHeads = number('attention.head_count_kv') || heads;
  const headSize = number('attention.key_length') || number('embedding_length') / heads;
  const bytes = 2 * 2 * tokens * layers * keyValueHeads * headSize;
  return positive(bytes) ? bytes : null;
}

/** Memory a model of `size` bytes on disk needs once loaded: its weights, its context cache and working buffers. */
export const memoryNeeded = (size: number, context?: number | null): number =>
  size + (positive(context) ? context : TYPICAL_CONTEXT) + WORKING;

/** `limit`: what the system lets this app use, where it says. `total`: physical memory. Both in bytes. */
export type DeviceMemory = { limit?: number | null; total?: number | null };

/** Memory this app can expect to use, or null when nothing is known. */
export const memoryBudget = ({ limit, total }: DeviceMemory): number | null =>
  positive(limit) ? limit : positive(total) ? total * FALLBACK_SHARE : null;

/**
 * A warning when loading a model of `size` bytes may get the app ended, or null. It is an estimate, so it
 * only warns; unknown memory never warns.
 */
export function memoryWarning(size: number, memory: DeviceMemory, context?: number | null): string | null {
  const budget = memoryBudget(memory);
  if (budget === null || !positive(size) || memoryNeeded(size, context) <= budget) return null;
  return `May be too large for this device's memory (${formatBytes(size)} model, ${formatBytes(budget)} available)`;
}

import { formatBytes } from './huggingface.ts';

// A phone ends an app that uses too much memory long before the app holds all of it: on iOS the limit for
// one app is roughly half of physical memory, and Android behaves similarly under pressure.
const APP_SHARE = 0.5;
// The weights are held in memory whole, and the context (KV cache) and working buffers come on top.
const LOAD_FACTOR = 1.2;

/** Memory a model of `size` bytes on disk needs once loaded. */
export const memoryNeeded = (size: number): number => size * LOAD_FACTOR;

/** Memory one app can expect to use on a device with `total` bytes of physical memory. */
export const memoryBudget = (total: number): number => total * APP_SHARE;

/**
 * A warning when loading a model of `size` bytes may get the app ended, or null. It is an estimate, so it
 * only warns; an unknown memory size (`total` null or 0) never warns.
 */
export function memoryWarning(size: number, total: number | null | undefined): string | null {
  if (!total || !(total > 0) || !(size > 0)) return null;
  if (memoryNeeded(size) <= memoryBudget(total)) return null;
  return `May be too large for this device's memory (${formatBytes(size)} model, ${formatBytes(total)} memory)`;
}

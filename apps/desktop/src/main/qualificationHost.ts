/** Parse `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`. */
export function parseNvidiaSmiMemoryGib(output: string): number | null {
  const memoryMiB = output
    .split(/\r?\n/)
    .map((line) => Number.parseFloat(line.trim().replace(/\s*MiB$/i, '')))
    .filter((value) => Number.isFinite(value) && value > 0)
  if (memoryMiB.length === 0) return null
  // Qualification is per usable device, not aggregate memory across cards.
  return Math.max(...memoryMiB) / 1024
}

import type { HardwareInfo, HubFile } from '../../api'

export type QuantFit = 'gpu' | 'offload' | 'system' | 'none' | 'unknown'

/**
 * A search result is a weight file, not a promise that it can be fully
 * offloaded. Keep the labels conservative: reserve headroom for the KV cache,
 * runtime allocations, and the desktop rather than comparing raw bytes.
 *
 * Multi-component bundles (stable-diffusion.cpp) stage their encoders,
 * denoiser, and VAE as separate phases, streaming each component's weights
 * through VRAM, so the whole bundle never needs to be resident in GPU memory
 * at once. For those, the GPU fit is judged against the diffusion checkpoint —
 * the one component used on every step — with the rest counting as staged
 * offload. Single weight files pass `diffusionBytes` equal to `bytes`, which
 * keeps the original all-resident behaviour.
 */
export function generationFit(
  bytes: number | null | undefined,
  hardware: HardwareInfo | null,
  diffusionBytes = bytes
): QuantFit {
  if (bytes == null || !hardware) return 'unknown'
  // `gpu_offload_memory_bytes` is the placement budget used by the runtime.
  // Prefer it so AMD systems are not accidentally assessed against all RAM.
  const gpu = hardware.gpu_offload_memory_bytes ?? hardware.vram_bytes
  const system = hardware.memory_bytes
  if (gpu != null) {
    if (bytes <= gpu * 0.7) return 'gpu'
    // The denoiser must be resident (it runs every step); encoders and VAE
    // stream through VRAM for their own phases, so only it constrains a GPU
    // fit. Its activation buffers vary sharply with resolution and video
    // frames, so this is still a staged-offload fit, not a guaranteed green
    // resident-GPU fit.
    if (diffusionBytes != null && diffusionBytes <= gpu * 0.7 && system != null) {
      return 'offload'
    }
    if (system != null && bytes <= system * 0.6) return 'offload'
    return 'none'
  }
  // A detected discrete GPU with no readable VRAM must not be reported as a
  // green system-memory fit. That would hide the actual GPU constraint.
  // Apple Silicon and integrated GPUs (AMD APU, Intel iGPU) intentionally have
  // no separate VRAM figure: their GPU uses unified memory, which is the
  // correct budget to assess here.
  if (
    hardware.os !== 'macos' &&
    hardware.gpu &&
    !hardware.amd_apu &&
    !hardware.intel_igpu
  )
    return 'unknown'
  if (system != null && bytes <= system * 0.6) return 'system'
  return system == null ? 'unknown' : 'none'
}

/** One downloadable quantisation of a model. Split GGUFs are one pick. */
export type QuantGroup = {
  /** File path with any shard suffix stripped — unique per quantisation. */
  key: string
  /** Every file of the quant, in shard order. */
  files: HubFile[]
  /** Summed size across all files, or null when any part's size is unknown. */
  size: number | null
}

export function generationFitLabel(fit: QuantFit): string {
  switch (fit) {
    case 'gpu': return 'Fits in GPU memory'
    case 'offload': return 'Fits with staged offload'
    case 'system': return 'Fits in system memory'
    case 'none': return 'Likely too large for this machine'
    default: return 'Memory estimate unavailable'
  }
}

function quantGroup(path: string): string {
  return path.replace(/-\d{1,5}-of-\d{1,5}(?=\.gguf$)/i, '')
}

/** Group a repository's quant files into downloadable quantisations. */
export function groupQuants(files: HubFile[]): QuantGroup[] {
  const byGroup = new Map<string, HubFile[]>()
  for (const file of files) {
    const key = quantGroup(file.path)
    const group = byGroup.get(key)
    if (group) {
      group.push(file)
    } else {
      byGroup.set(key, [file])
    }
  }
  return [...byGroup.entries()].map(([key, groupFiles]) => {
    // Shard order matters for the download: llama.cpp discovers siblings
    // from the first shard's name.
    const sorted = [...groupFiles].sort((left, right) => left.path.localeCompare(right.path))
    let size: number | null = 0
    for (const file of sorted) {
      if (file.size == null) {
        size = null
        break
      }
      size += file.size
    }
    return { key, files: sorted, size }
  })
}

/** Display name for a group: the file name a single-file quant would have. */
export function quantGroupName(group: QuantGroup): string {
  return group.key.split('/').at(-1) ?? group.key
}

export function sortQuantGroups(
  groups: QuantGroup[],
  hardware: HardwareInfo | null
): QuantGroup[] {
  const rank: Record<QuantFit, number> = { gpu: 0, system: 0, offload: 1, unknown: 2, none: 3 }
  return [...groups].sort((left, right) => {
    const fit =
      rank[generationFit(left.size, hardware)] - rank[generationFit(right.size, hardware)]
    return fit || (right.size ?? 0) - (left.size ?? 0) || left.key.localeCompare(right.key)
  })
}

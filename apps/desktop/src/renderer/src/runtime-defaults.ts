import type { HardwareInfo, RuntimeSettings } from './api'

export const INTEGRATED_GPU_VIDEO_DEFAULTS = {
  width: 512,
  height: 320,
  frames: 17,
  maxVram: 2
} as const

/** Whether sd.cpp should use its unified-memory Vulkan safety defaults. */
export function usesIntegratedGpuVulkanDefaults(
  settings: RuntimeSettings | null,
  hardware: HardwareInfo | null
): boolean {
  if (!hardware || !settings) return false
  const integrated = hardware.amd_apu || hardware.intel_igpu
  if (!integrated) return false
  const target =
    settings.target === 'auto' ? hardware.recommended_target : settings.target
  return target === 'vulkan'
}

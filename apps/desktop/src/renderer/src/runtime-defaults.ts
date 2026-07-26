import type { HardwareInfo, RuntimeSettings } from './api'

export const AMD_APU_VIDEO_DEFAULTS = {
  width: 512,
  height: 320,
  frames: 17,
  maxVram: 4
} as const

export const AMD_APU_IMAGE_MAX_VRAM = 8

/** Whether sd.cpp should use its unified-memory Vulkan safety defaults. */
export function usesAmdApuVulkanDefaults(
  settings: RuntimeSettings | null,
  hardware: HardwareInfo | null
): boolean {
  if (!hardware?.amd_apu || !settings) return false
  const target =
    settings.target === 'auto' ? hardware.recommended_target : settings.target
  return target === 'vulkan'
}

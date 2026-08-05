import { describe, expect, it } from 'vitest'

import type { HardwareInfo, RuntimeSettings } from './api'
import { usesIntegratedGpuVulkanDefaults } from './runtime-defaults'

const hardware = {
  amd_apu: true,
  intel_igpu: false,
  recommended_target: 'vulkan'
} as HardwareInfo

describe('usesIntegratedGpuVulkanDefaults', () => {
  it('enables the policy for explicit and automatically selected Vulkan', () => {
    expect(
      usesIntegratedGpuVulkanDefaults({ target: 'vulkan' } as RuntimeSettings, hardware)
    ).toBe(true)
    expect(
      usesIntegratedGpuVulkanDefaults({ target: 'auto' } as RuntimeSettings, hardware)
    ).toBe(true)
  })

  it('covers an Intel iGPU as well as an AMD APU', () => {
    expect(
      usesIntegratedGpuVulkanDefaults(
        { target: 'auto' } as RuntimeSettings,
        { ...hardware, amd_apu: false, intel_igpu: true }
      )
    ).toBe(true)
  })

  it('does not affect discrete GPUs or non-Vulkan runtimes', () => {
    expect(
      usesIntegratedGpuVulkanDefaults(
        { target: 'vulkan' } as RuntimeSettings,
        { ...hardware, amd_apu: false, intel_igpu: false }
      )
    ).toBe(false)
    expect(
      usesIntegratedGpuVulkanDefaults({ target: 'cpu' } as RuntimeSettings, hardware)
    ).toBe(false)
  })
})

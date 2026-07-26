import { describe, expect, it } from 'vitest'

import type { HardwareInfo, RuntimeSettings } from './api'
import { usesAmdApuVulkanDefaults } from './runtime-defaults'

const hardware = {
  amd_apu: true,
  recommended_target: 'vulkan'
} as HardwareInfo

describe('usesAmdApuVulkanDefaults', () => {
  it('enables the policy for explicit and automatically selected Vulkan', () => {
    expect(
      usesAmdApuVulkanDefaults({ target: 'vulkan' } as RuntimeSettings, hardware)
    ).toBe(true)
    expect(
      usesAmdApuVulkanDefaults({ target: 'auto' } as RuntimeSettings, hardware)
    ).toBe(true)
  })

  it('does not affect discrete GPUs or non-Vulkan runtimes', () => {
    expect(
      usesAmdApuVulkanDefaults(
        { target: 'vulkan' } as RuntimeSettings,
        { ...hardware, amd_apu: false }
      )
    ).toBe(false)
    expect(
      usesAmdApuVulkanDefaults({ target: 'cpu' } as RuntimeSettings, hardware)
    ).toBe(false)
  })
})

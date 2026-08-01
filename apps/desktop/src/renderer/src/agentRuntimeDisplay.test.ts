import { describe, expect, it } from 'vitest'

import { showsBrazierSandboxStatus } from './agentRuntimeDisplay'

describe('agent runtime status display', () => {
  it('never labels OMP with Brazier host sandbox capabilities', () => {
    expect(showsBrazierSandboxStatus('omp')).toBe(false)
    expect(showsBrazierSandboxStatus('pi')).toBe(true)
  })
})

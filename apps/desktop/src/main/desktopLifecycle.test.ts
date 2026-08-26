import { describe, expect, it } from 'vitest'

import { DesktopLifecycle } from './desktopLifecycle'

describe('DesktopLifecycle', () => {
  it('shows repeated launch requests while the primary instance is healthy', () => {
    const lifecycle = new DesktopLifecycle()
    expect(lifecycle.canShowWindow).toBe(true)
    expect(lifecycle.requestActivation()).toBe('show-window')
    expect(lifecycle.requestQuit()).toBe('start-cleanup')
    expect(lifecycle.cleanupFinished()).toEqual({ relaunch: false })
  })

  it('coalesces quit requests into one cleanup and then permits exit', () => {
    const lifecycle = new DesktopLifecycle()
    expect(lifecycle.requestQuit()).toBe('start-cleanup')
    expect(lifecycle.requestQuit()).toBe('wait-for-cleanup')
    expect(lifecycle.canShowWindow).toBe(false)
    expect(lifecycle.cleanupFinished()).toEqual({ relaunch: false })
    expect(lifecycle.requestQuit()).toBe('allow-exit')
  })

  it('turns an activation received during cleanup into exactly one relaunch', () => {
    const lifecycle = new DesktopLifecycle()
    lifecycle.requestQuit()
    expect(lifecycle.requestActivation()).toBe('relaunch-after-cleanup')
    expect(lifecycle.requestActivation()).toBe('relaunch-after-cleanup')
    expect(lifecycle.cleanupFinished()).toEqual({ relaunch: true })
  })
})

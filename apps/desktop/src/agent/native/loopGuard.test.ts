import { describe, expect, it } from 'vitest'

import { canonicalizeArgs, DEFAULT_MAX_TURNS_PER_RUN, LoopGuard } from './loopGuard'

describe('LoopGuard', () => {
  it('canonicalizes argument key order deterministically', () => {
    expect(canonicalizeArgs({ b: 2, a: { y: true, x: 1 } })).toBe(
      canonicalizeArgs({ a: { x: 1, y: true }, b: 2 })
    )
  })

  it('enforces maxTurnsPerRun = 40 by default', () => {
    const guard = new LoopGuard()
    expect(guard.maxTurnsPerRun).toBe(DEFAULT_MAX_TURNS_PER_RUN)
    for (let i = 1; i <= 40; i += 1) {
      expect(guard.startTurn().allowed).toBe(true)
    }
    const overflow = guard.startTurn()
    expect(overflow.allowed).toBe(false)
    expect(overflow.message).toContain('40 turns')
  })

  it('warns at 3 identical calls, short-circuits at 5, and halts if repeated once more', () => {
    const guard = new LoopGuard()
    const args = { path: 'src/index.ts' }

    // Calls 1 & 2: allowed, unmodified output
    expect(guard.beforeCall('fs_read', args)).toEqual({ action: 'allow' })
    expect(guard.afterCall('fs_read', args, false, 'file body')).toBe('file body')

    expect(guard.beforeCall('fs_read', { path: 'src/index.ts' })).toEqual({ action: 'allow' })
    expect(guard.afterCall('fs_read', args, false, 'file body')).toBe('file body')

    // Call 3: appends [harness notice: ...]
    expect(guard.beforeCall('fs_read', args)).toEqual({ action: 'allow' })
    const thirdOut = guard.afterCall('fs_read', args, false, 'file body')
    expect(thirdOut).toContain('[harness notice:')
    expect(thirdOut).toContain('Do not repeat this exact call')

    // Call 4: still executes once more
    expect(guard.beforeCall('fs_read', args)).toEqual({ action: 'allow' })
    expect(guard.afterCall('fs_read', args, false, 'file body')).toContain('[harness notice:')

    // Call 5: short-circuits broker execution!
    const fifth = guard.beforeCall('fs_read', args)
    expect(fifth.action).toBe('short-circuit')
    if (fifth.action === 'short-circuit') {
      expect(fifth.message).toContain('5th identical call')
    }

    // Call 6 (repeats once more): stops the turn loop cleanly!
    const sixth = guard.beforeCall('fs_read', args)
    expect(sixth.action).toBe('halt-run')
  })

  it('short-circuits at 4 consecutive identical tool errors and halts on the 5th', () => {
    const guard = new LoopGuard()
    const args = { command: 'nonexistent_cmd' }

    for (let i = 1; i <= 3; i += 1) {
      expect(guard.beforeCall('shell_run', args)).toEqual({ action: 'allow' })
      const out = guard.afterCall('shell_run', args, true, 'command not found')
      if (i === 3) expect(out).toContain('[harness notice:')
    }

    // 4th consecutive identical error -> short-circuits!
    const fourth = guard.beforeCall('shell_run', args)
    expect(fourth.action).toBe('short-circuit')
    if (fourth.action === 'short-circuit') {
      expect(fourth.message).toContain('4th consecutive identical failing call')
    }

    // Repeating once more -> halts!
    expect(guard.beforeCall('shell_run', args).action).toBe('halt-run')
  })

  it('does not falsely block repeated reads when file content changes between edits', () => {
    const guard = new LoopGuard()
    const args = { path: 'src/app.ts' }

    for (let version = 1; version <= 6; version += 1) {
      expect(guard.beforeCall('fs_read', args)).toEqual({ action: 'allow' })
      const out = guard.afterCall('fs_read', args, false, `content v${version}`)
      expect(out).toBe(`content v${version}`)
    }
  })
})

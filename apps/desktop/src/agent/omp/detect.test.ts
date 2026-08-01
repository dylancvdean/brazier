import { chmodSync, mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { detectOmpBinary } from './detect'

let directory: string | undefined

afterEach(() => {
  vi.unstubAllEnvs()
  if (directory) rmSync(directory, { recursive: true, force: true })
  directory = undefined
})

describe.skipIf(process.platform === 'win32')('detectOmpBinary', () => {
  it('only accepts a regular executable, never a merely existing path', () => {
    directory = mkdtempSync(join(tmpdir(), 'brazier-omp-detect-'))
    const candidate = join(directory, 'omp')
    writeFileSync(candidate, '#!/bin/sh\nexit 0\n')
    chmodSync(candidate, 0o644)
    vi.stubEnv('BRAZIER_OMP_PATH', candidate)
    vi.stubEnv('OMP_PATH', '')
    vi.stubEnv('PATH', directory)

    expect(detectOmpBinary()).toBeNull()

    chmodSync(candidate, 0o755)
    expect(detectOmpBinary()).toEqual({ path: candidate, source: 'env' })
  })

  it('does not identify a directory in PATH as the omp executable', () => {
    directory = mkdtempSync(join(tmpdir(), 'brazier-omp-detect-'))
    mkdirSync(join(directory, 'omp'))
    vi.stubEnv('BRAZIER_OMP_PATH', '')
    vi.stubEnv('OMP_PATH', '')
    vi.stubEnv('PATH', directory)

    expect(detectOmpBinary()).toBeNull()
  })
})

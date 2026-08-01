/**
 * Locate a system Oh My Pi (`omp`) binary for the RPC sidecar.
 *
 * OMP is an optional stock runtime: Brazier does not ship its ~300MB native
 * closure by default. Operators install `omp` themselves (or point
 * BRAZIER_OMP_PATH at a binary).
 */

import { accessSync, constants, statSync } from 'node:fs'
import { delimiter, join } from 'node:path'

export type OmpBinary = {
  path: string
  source: 'env' | 'path'
}

function isExecutable(path: string): boolean {
  try {
    if (!statSync(path).isFile()) return false
    accessSync(path, constants.X_OK)
    return true
  } catch {
    return false
  }
}

/** Resolve the omp CLI, or null when the fuller runtime cannot start. */
export function detectOmpBinary(configuredPath?: string): OmpBinary | null {
  const override = configuredPath?.trim() || process.env.BRAZIER_OMP_PATH || process.env.OMP_PATH
  if (override && isExecutable(override)) {
    return { path: override, source: 'env' }
  }
  const pathVar = process.env.PATH ?? ''
  for (const directory of pathVar.split(delimiter)) {
    if (!directory) continue
    for (const name of ['omp', 'omp.exe']) {
      const candidate = join(directory, name)
      if (isExecutable(candidate)) {
        return { path: candidate, source: 'path' }
      }
    }
  }
  return null
}

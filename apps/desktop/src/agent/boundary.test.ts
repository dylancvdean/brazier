import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * The framework-independence rule from the Agent mode plan, enforced instead of
 * documented: framework packages may only be imported inside their adapter
 * directory (`src/agent/pi`, …).
 */

const AGENT_ROOT = join(__dirname)
const ADAPTER_DIRECTORIES = ['pi']
const PI_PACKAGE_PATTERN = /@earendil-works\/pi-|@mariozechner\/pi-/

function sourceFiles(directory: string): string[] {
  const entries = readdirSync(directory)
  return entries.flatMap((entry) => {
    const path = join(directory, entry)
    if (statSync(path).isDirectory()) return sourceFiles(path)
    return path.endsWith('.ts') ? [path] : []
  })
}

describe('runtime adapter boundary', () => {
  const files = sourceFiles(AGENT_ROOT)

  it('finds the agent sources', () => {
    expect(files.length).toBeGreaterThan(5)
  })

  it('imports Pi only inside an adapter directory', () => {
    const offenders = files.filter((file) => {
      const relativePath = relative(AGENT_ROOT, file).split('\\').join('/')
      const topLevel = relativePath.split('/')[0]
      if (ADAPTER_DIRECTORIES.includes(topLevel ?? '')) return false
      // Two tests name the packages as data rather than importing them: this
      // one guards against them, and the packaging test walks their dependency
      // closure. Naming both explicitly keeps the exemption from widening into
      // "any test file".
      if (relativePath === 'boundary.test.ts' || relativePath === 'packaging.test.ts') return false
      return PI_PACKAGE_PATTERN.test(readFileSync(file, 'utf8'))
    })
    expect(
      offenders.map((file) => relative(AGENT_ROOT, file).split('\\').join('/'))
    ).toEqual([])
  })

  it('keeps the application types self-contained', () => {
    const types = readFileSync(join(AGENT_ROOT, 'core/types.ts'), 'utf8')
    expect(types).not.toMatch(PI_PACKAGE_PATTERN)
    // A pure type module: no imports at all, so no framework shape can leak
    // into the application's own vocabulary.
    expect(types).not.toMatch(/^\s*import\s/m)
  })

  it('routes brokered tool calls through the daemon client', () => {
    // The executor is the only place that may call the exec endpoint, and it
    // must go through the broker client rather than raw fetch.
    const executor = readFileSync(join(AGENT_ROOT, 'core/toolExecutor.ts'), 'utf8')
    expect(executor).toMatch(/broker\.execTool/)
    expect(executor).not.toMatch(/\bfetch\(/)

    for (const file of files) {
      const relativePath = relative(AGENT_ROOT, file).split('\\').join('/')
      if (relativePath === 'core/brokerClient.ts') continue
      if (relativePath.endsWith('.test.ts')) continue
      const source = readFileSync(file, 'utf8')
      expect(source, `${relativePath} must not call fetch directly`).not.toMatch(/\bfetch\(/)
    }
  })
})

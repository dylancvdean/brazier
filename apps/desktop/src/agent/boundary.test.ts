import { builtinModules } from 'node:module'
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * Architectural boundary invariants for the native zero-SDK agent harness:
 * - Only `src/agent/native` implements the runtime adapter (`src/agent/pi` is gone).
 * - Zero third-party npm packages are imported anywhere in non-test agent sources.
 * - `core/types.ts` has zero `import` statements.
 * - Only `core/brokerClient.ts` calls `fetch(`.
 */

const AGENT_ROOT = join(__dirname)
const ADAPTER_DIRECTORIES = ['native']
const THIRD_PARTY_FRAMEWORK_PATTERN = /@earendil-works\/pi-|@mariozechner\/pi-/
const NODE_BUILTINS = new Set([
  ...builtinModules,
  ...builtinModules.map((mod) => `node:${mod}`)
])

function sourceFiles(directory: string): string[] {
  const entries = readdirSync(directory)
  return entries.flatMap((entry) => {
    const path = join(directory, entry)
    if (statSync(path).isDirectory()) return sourceFiles(path)
    return path.endsWith('.ts') ? [path] : []
  })
}

function moduleSpecifiers(source: string): string[] {
  const specifiers: string[] = []
  const staticPattern = /^\s*(?:import|export)\s+(?:[\s\S]*?\s+from\s+)?['"]([^'"]+)['"]/gm
  const dynamicPattern = /\bimport\(\s*['"]([^'"]+)['"]\s*\)/g
  for (const pattern of [staticPattern, dynamicPattern]) {
    let match: RegExpExecArray | null
    while ((match = pattern.exec(source)) !== null) {
      if (match[1]) specifiers.push(match[1])
    }
  }
  return specifiers
}

describe('runtime adapter boundary', () => {
  const files = sourceFiles(AGENT_ROOT)

  it('finds the agent sources and keeps only the native adapter directory', () => {
    expect(files.length).toBeGreaterThan(5)
    expect(ADAPTER_DIRECTORIES).toEqual(['native'])
    expect(existsSync(join(AGENT_ROOT, 'native'))).toBe(true)
    expect(existsSync(join(AGENT_ROOT, 'pi'))).toBe(false)
  })

  it('has zero third-party framework leaks and zero external npm imports in production sources', () => {
    const frameworkOffenders = files.filter((file) => {
      const relativePath = relative(AGENT_ROOT, file).split('\\').join('/')
      if (relativePath === 'boundary.test.ts' || relativePath === 'packaging.test.ts') return false
      return THIRD_PARTY_FRAMEWORK_PATTERN.test(readFileSync(file, 'utf8'))
    })
    expect(
      frameworkOffenders.map((file) => relative(AGENT_ROOT, file).split('\\').join('/'))
    ).toEqual([])

    for (const file of files) {
      const relativePath = relative(AGENT_ROOT, file).split('\\').join('/')
      if (relativePath.endsWith('.test.ts')) continue
      const source = readFileSync(file, 'utf8')
      const externalSpecs = moduleSpecifiers(source).filter(
        (spec) => !spec.startsWith('.') && !NODE_BUILTINS.has(spec)
      )
      expect(externalSpecs, `${relativePath} must not import external npm packages`).toEqual([])

      if (relativePath.startsWith('core/')) {
        const adapterLeaks = moduleSpecifiers(source).filter((spec) => spec.includes('native/'))
        expect(adapterLeaks, `${relativePath} must not depend on native/`).toEqual([])
      }
    }
  })

  it('keeps the application types self-contained', () => {
    const types = readFileSync(join(AGENT_ROOT, 'core/types.ts'), 'utf8')
    expect(types).not.toMatch(THIRD_PARTY_FRAMEWORK_PATTERN)
    // A pure type module: no imports at all, so no framework shape can leak
    // into the application's own vocabulary.
    expect(types).not.toMatch(/^\s*import\s/m)
  })

  it('routes brokered tool calls and model requests through the daemon client', () => {
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

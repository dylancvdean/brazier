/**
 * Stage a self-contained node_modules tree for the agent worker.
 *
 * pnpm keeps transitive deps (e.g. `openai` for `@earendil-works/pi-ai`) in the
 * virtual store, not as top-level symlinks under apps/desktop/node_modules.
 * `cp -aL` on that directory therefore omits them and the utilityProcess worker
 * dies on import with exit code 1. This walks the same closure as
 * packaging.test.ts and copies every package the worker resolves at run time.
 *
 * Usage: node scripts/stage-packaged-node-modules.mjs <dest/node_modules>
 */

import { cpSync, existsSync, mkdirSync, readFileSync, realpathSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const destRoot = process.argv[2]
if (!destRoot) {
  console.error('Usage: node scripts/stage-packaged-node-modules.mjs <dest/node_modules>')
  process.exit(2)
}

const appRoot = dirname(dirname(fileURLToPath(import.meta.url)))
const manifest = JSON.parse(readFileSync(join(appRoot, 'package.json'), 'utf8'))

const WORKER_ROOTS = Object.keys(manifest.dependencies).filter((name) =>
  name.startsWith('@earendil-works/')
)

function resolveManifest(name, from) {
  let directory = dirname(realpathSync(from))
  for (let depth = 0; depth < 12; depth += 1) {
    const candidate = join(directory, 'node_modules', name, 'package.json')
    if (existsSync(candidate)) return candidate
    const parent = dirname(directory)
    if (parent === directory) break
    directory = parent
  }
  return null
}

function runtimeClosure(roots) {
  const manifests = new Map()
  const queue = [...roots.map((name) => ({ name, from: join(appRoot, 'package.json') }))]
  while (queue.length > 0) {
    const { name, from } = queue.shift()
    if (manifests.has(name)) continue
    const manifestPath = resolveManifest(name, from)
    if (!manifestPath) continue
    manifests.set(name, manifestPath)
    const packageJson = JSON.parse(readFileSync(manifestPath, 'utf8'))
    for (const dependency of Object.keys(packageJson.dependencies ?? {})) {
      queue.push({ name: dependency, from: manifestPath })
    }
  }
  return manifests
}

function copyPackage(name, manifestPath) {
  const sourceRoot = dirname(realpathSync(manifestPath))
  const segments = name.startsWith('@') ? name.split('/') : [name]
  const targetRoot = join(destRoot, ...segments)
  mkdirSync(dirname(targetRoot), { recursive: true })
  cpSync(sourceRoot, targetRoot, { recursive: true, dereference: true })
}

mkdirSync(destRoot, { recursive: true })
const closure = runtimeClosure(WORKER_ROOTS)
for (const [name, manifestPath] of [...closure.entries()].sort(([a], [b]) => a.localeCompare(b))) {
  copyPackage(name, manifestPath)
}
console.error(`[stage-packaged-node-modules] copied ${closure.size} packages to ${destRoot}`)

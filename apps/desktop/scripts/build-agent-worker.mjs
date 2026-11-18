/**
 * Build the self-contained agent worker bundle.
 *
 * The worker runs as an Electron `utilityProcess` (`out/agent/agent-worker.mjs`)
 * and uses Brazier's zero-dependency native agent harness. Bundling everything
 * except Node built-ins (`node:*`) means packaged builds need no `node_modules`
 * runtime closure at all.
 *
 * Usage: node scripts/build-agent-worker.mjs [--watch]
 */

import { build, context } from 'esbuild'
import { realpathSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptPath = fileURLToPath(import.meta.url)
const root = dirname(dirname(scriptPath))

export function workerBuildOptions(appRoot = root) {
  return {
    entryPoints: [join(appRoot, 'src/agent/worker.ts')],
    // Deliberately outside out/main: electron-vite empties that directory on
    // every build, so a sibling directory keeps the two builds order-independent.
    outfile: join(appRoot, 'out/agent/agent-worker.mjs'),
    bundle: true,
    platform: 'node',
    // Electron 39 ships a Node 22 runtime.
    target: 'node22',
    format: 'esm',
    external: ['node:*'],
    sourcemap: true,
    logLevel: 'info'
  }
}

const isMain =
  Boolean(process.argv[1]) &&
  (() => {
    try {
      return realpathSync(process.argv[1]) === realpathSync(scriptPath)
    } catch {
      return false
    }
  })()

if (isMain) {
  const options = workerBuildOptions(root)
  if (process.argv.includes('--watch')) {
    const ctx = await context(options)
    await ctx.watch()
    console.error('[agent-worker] watching for changes')
  } else {
    await build(options)
  }
}

/**
 * Build the agent worker bundle.
 *
 * The worker runs as an Electron `utilityProcess` and imports the Pi runtime,
 * which is ESM-only. electron-vite emits the main process as CommonJS, so the
 * worker gets its own tiny ESM build here instead. Dependencies stay external:
 * Pi is a normal installed package, never vendored or inlined.
 *
 * Usage: node scripts/build-agent-worker.mjs [--watch]
 */

import { context, build } from 'esbuild'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const root = dirname(dirname(fileURLToPath(import.meta.url)))

const options = {
  entryPoints: [join(root, 'src/agent/worker.ts')],
  // Deliberately outside out/main: electron-vite empties that directory on
  // every build, so a sibling directory keeps the two builds order-independent.
  outfile: join(root, 'out/agent/agent-worker.mjs'),
  bundle: true,
  platform: 'node',
  // Electron 39 ships a Node 22 runtime.
  target: 'node22',
  format: 'esm',
  // Every bare import resolves from node_modules at run time. Keeping the Pi
  // packages external means an upgrade is an install, not a rebuild of a
  // vendored copy.
  packages: 'external',
  sourcemap: true,
  logLevel: 'info'
}

if (process.argv.includes('--watch')) {
  const ctx = await context(options)
  await ctx.watch()
  console.error('[agent-worker] watching for changes')
} else {
  await build(options)
}

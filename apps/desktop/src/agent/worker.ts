/**
 * Agent worker entry point.
 *
 * Runs as an Electron `utilityProcess`: a Node process with no window, no
 * renderer access, and no privileged helpers. It talks to the main process over
 * `parentPort` and to the daemon over authenticated loopback HTTP. The agent
 * runtime lives here so a crash or a runaway loop cannot take the UI down.
 */

import type { WorkerCommand, WorkerMessage } from './core/protocol'
import { AgentWorkerCore } from './workerCore'

type ParentPort = {
  on(channel: 'message', listener: (event: { data: unknown }) => void): void
  postMessage(message: unknown): void
}

const parentPort = (process as unknown as { parentPort?: ParentPort }).parentPort

if (!parentPort) {
  throw new Error('The agent worker must be started as an Electron utilityProcess.')
}

const post = (message: WorkerMessage): void => {
  parentPort.postMessage(message)
}

const core = new AgentWorkerCore(post)

/** Handle one command at a time so open/rehydrate cannot race a run. */
let commandChain = Promise.resolve()
parentPort.on('message', (event) => {
  const command = event.data as WorkerCommand
  if (!command || typeof command !== 'object' || typeof command.type !== 'string') {
    return
  }
  commandChain = commandChain.then(() => core.handle(command))
})

process.on('uncaughtException', (error: Error) => {
  post({ type: 'log', level: 'error', message: `uncaught: ${error.message}` })
})

process.on('unhandledRejection', (reason: unknown) => {
  post({
    type: 'log',
    level: 'error',
    message: `unhandled rejection: ${reason instanceof Error ? reason.message : String(reason)}`
  })
})

/**
 * Message protocol between the Electron main process and the agent worker.
 *
 * The worker is a separate process with no window, no renderer access, and no
 * host privileges of its own. Everything it can do is either a message on this
 * channel or an authenticated call to the daemon.
 */

import type {
  AgentEvent,
  AgentModelReference,
  AgentPermissionMode,
  AgentSessionState,
  AgentUserInput
} from './types'

export type WorkerConnection = {
  address: string
  apiKey: string | null
}

export type WorkerCommand =
  | { type: 'init'; requestId: string; connection: WorkerConnection }
  | {
      type: 'open-session'
      requestId: string
      sessionId: string
    }
  | { type: 'run'; requestId: string; sessionId: string; input: AgentUserInput }
  | { type: 'cancel'; requestId: string; sessionId: string }
  | { type: 'compact'; requestId: string; sessionId: string }
  | {
      type: 'set-model'
      requestId: string
      sessionId: string
      model: AgentModelReference
    }
  | { type: 'set-tools'; requestId: string; sessionId: string; tools: string[] }
  | { type: 'composer-suggestions'; requestId: string; sessionId: string }
  | { type: 'set-permission-mode'; requestId: string; sessionId: string; mode: AgentPermissionMode }
  | {
      type: 'runtime-command'
      requestId: string
      sessionId: string
      runtimeId?: string
      command: Record<string, unknown>
    }
  | { type: 'close-session'; requestId: string; sessionId: string }
  | { type: 'shutdown'; requestId: string }

/**
 * A command before the supervisor stamps a request id. Distributive so each
 * union member keeps its own fields; a plain `Omit` over a union would collapse
 * to the shared keys only.
 */
export type WorkerCommandInput = WorkerCommand extends infer Member
  ? Member extends { requestId: string }
    ? Omit<Member, 'requestId'>
    : never
  : never

export type WorkerMessage =
  | { type: 'ready'; runtimeId: string; runtimeVersion: string }
  | { type: 'result'; requestId: string; ok: true; data?: unknown }
  | { type: 'result'; requestId: string; ok: false; error: string }
  | { type: 'event'; sessionId: string; event: AgentEvent }
  | { type: 'session-state'; sessionId: string; state: AgentSessionState }
  | {
      type: 'runtime-frame'
      sessionId: string
      runtimeId: string
      /** One frame from the runtime's own stream (e.g. an OMP RPC stdout frame). */
      payload: Record<string, unknown>
    }
  | { type: 'log'; level: 'info' | 'warn' | 'error'; message: string }

/** IPC channel names the renderer sees through the preload bridge. */
export const AGENT_IPC = {
  invoke: 'brazier:agent:invoke',
  event: 'brazier:agent:event'
} as const

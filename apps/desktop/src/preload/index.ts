import { contextBridge, ipcRenderer } from 'electron'

import { AGENT_IPC, type WorkerCommandInput, type WorkerMessage } from '../agent/core/protocol'
import type { AgentModelReference, AgentUserInput } from '../agent/core/types'

export type BrazierConnection = {
  address: string
  api_key: string | null
}

export type BrazierFlags = {
  forceWelcome: boolean
}

const invokeAgent = (command: WorkerCommandInput): Promise<unknown> =>
  ipcRenderer.invoke(AGENT_IPC.invoke, command)

/**
 * Agent bridge. The renderer stays sandboxed with no Node integration: it can
 * only send these named commands and listen for events. It cannot fork the
 * worker, read files, or execute anything itself.
 */
const agent = {
  openSession: (sessionId: string): Promise<unknown> =>
    invokeAgent({ type: 'open-session', sessionId }),
  run: (sessionId: string, input: AgentUserInput): Promise<unknown> =>
    invokeAgent({ type: 'run', sessionId, input }),
  cancel: (sessionId: string): Promise<unknown> => invokeAgent({ type: 'cancel', sessionId }),
  compact: (sessionId: string): Promise<unknown> => invokeAgent({ type: 'compact', sessionId }),
  setModel: (sessionId: string, model: AgentModelReference): Promise<unknown> =>
    invokeAgent({ type: 'set-model', sessionId, model }),
  setTools: (sessionId: string, tools: string[]): Promise<unknown> =>
    invokeAgent({ type: 'set-tools', sessionId, tools }),
  closeSession: (sessionId: string): Promise<unknown> =>
    invokeAgent({ type: 'close-session', sessionId }),
  status: (): Promise<{ running: boolean; crashes: number }> =>
    ipcRenderer.invoke('brazier:agent:status'),
  /** Subscribe to worker events. Returns an unsubscribe function. */
  onMessage: (listener: (message: WorkerMessage) => void): (() => void) => {
    const handler = (_event: unknown, message: WorkerMessage): void => listener(message)
    ipcRenderer.on(AGENT_IPC.event, handler)
    return () => ipcRenderer.removeListener(AGENT_IPC.event, handler)
  }
}

contextBridge.exposeInMainWorld('brazier', {
  getConnection: (): Promise<BrazierConnection> => ipcRenderer.invoke('brazier:connection'),
  getFlags: (): Promise<BrazierFlags> => ipcRenderer.invoke('brazier:flags'),
  platform: process.platform,
  selectDirectory: (): Promise<string | null> => ipcRenderer.invoke('brazier:select-directory'),
  selectWorkspace: (): Promise<string | null> => ipcRenderer.invoke('brazier:select-workspace'),
  /** Pick a file (or an adapter directory) already on disk. Null when dismissed. */
  selectFile: (
    title: string,
    filters: Array<{ name: string; extensions: string[] }>
  ): Promise<string | null> => ipcRenderer.invoke('brazier:select-file', title, filters),
  /** Ask where to put a file and write it. Resolves null when dismissed. */
  saveFile: (suggestedName: string, data: ArrayBuffer): Promise<string | null> =>
    ipcRenderer.invoke('brazier:save-file', suggestedName, data),
  revealFile: (path: string): Promise<void> => ipcRenderer.invoke('brazier:reveal-file', path),
  agent
})

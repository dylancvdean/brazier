import { contextBridge, ipcRenderer } from 'electron'

import { AGENT_IPC, type WorkerCommandInput, type WorkerMessage } from '../agent/core/protocol'
import type { AgentModelReference, AgentPermissionMode, AgentUserInput } from '../agent/core/types'

export type BrazierConnection = {
  address: string
  api_key: string | null
}

export type BrazierFlags = {
  forceWelcome: boolean
}

export type BrazierServerSettings = {
  enabled: boolean
  port: number
  apiKeyEnabled: boolean
  hasApiKey: boolean
  jitLoading: boolean
}

export type BrazierInputGuardStatus = {
  supported: boolean
  installed: boolean
  secure: boolean
  ready: boolean
  current: boolean
  version: string | null
  detail: string
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
  composerSuggestions: (sessionId: string): Promise<Array<{ value: string; description: string }>> =>
    invokeAgent({ type: 'composer-suggestions', sessionId }) as Promise<Array<{ value: string; description: string }>>,
  setPermissionMode: (sessionId: string, mode: AgentPermissionMode): Promise<unknown> =>
    invokeAgent({ type: 'set-permission-mode', sessionId, mode }),
  /**
   * Drive an arbitrary runtime command (OMP RPC) and resolve its response
   * frame. Optional `runtimeId` guards against routing to the wrong session.
   */
  runtimeCommand: (
    sessionId: string,
    runtimeId: string | undefined,
    command: Record<string, unknown>
  ): Promise<unknown> => invokeAgent({ type: 'runtime-command', sessionId, runtimeId, command }),
  /** Answer an extension-UI dialog the runtime is holding open for the user. */
  resolveExtensionUi: (
    sessionId: string,
    response: Record<string, unknown>
  ): Promise<unknown> => invokeAgent({ type: 'resolve-extension-ui', sessionId, response }),
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
  getServerSettings: (): Promise<BrazierServerSettings> => ipcRenderer.invoke('brazier:server-settings'),
  saveServerSettings: (
    settings: Omit<BrazierServerSettings, 'hasApiKey'> & { apiKey?: string | null }
  ): Promise<BrazierServerSettings> => ipcRenderer.invoke('brazier:save-server-settings', settings),
  generateServerApiKey: (): Promise<string> => ipcRenderer.invoke('brazier:generate-server-api-key'),
  getFlags: (): Promise<BrazierFlags> => ipcRenderer.invoke('brazier:flags'),
  checkForUpdates: (): Promise<{ supported: boolean }> => ipcRenderer.invoke('brazier:check-for-updates'),
  getUpdateSettings: (): Promise<{
    supported: boolean
    checkOnStartup: boolean
    autoDownload: boolean
  }> => ipcRenderer.invoke('brazier:get-update-settings'),
  saveUpdateSettings: (settings: {
    checkOnStartup?: boolean
    autoDownload?: boolean
  }): Promise<{
    supported: boolean
    checkOnStartup: boolean
    autoDownload: boolean
  }> => ipcRenderer.invoke('brazier:save-update-settings', settings),
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
  computer: {
    setActive: (active: boolean): Promise<void> => ipcRenderer.invoke('brazier:computer:set-active', active),
    prepareSafety: (): Promise<void> => ipcRenderer.invoke('brazier:computer:prepare-safety'),
    inputGuardStatus: (): Promise<BrazierInputGuardStatus> =>
      ipcRenderer.invoke('brazier:computer:input-guard-status'),
    setupInputGuard: (): Promise<BrazierInputGuardStatus> =>
      ipcRenderer.invoke('brazier:computer:setup-input-guard'),
    onEscape: (listener: () => void): (() => void) => {
      const handler = (): void => listener()
      ipcRenderer.on('brazier:computer:escape', handler)
      return () => ipcRenderer.removeListener('brazier:computer:escape', handler)
    }
  },
  agent
})

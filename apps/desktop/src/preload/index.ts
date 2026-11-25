import { contextBridge, ipcRenderer } from 'electron'

import {
  AGENT_IPC,
  type AgentWorkerStatus,
  type WorkerCommandInput,
  type WorkerMessage
} from '../agent/core/protocol'
import type { AgentModelReference, AgentPermissionMode, AgentUserInput } from '../agent/core/types'
import type {
  ConnectionProfileSummary,
  ConnectionProfileView,
  ConnectionTestResult,
  ClaimedConnection,
  ConnectionVerification,
  DaemonInfo,
  PairingPreview,
  PairingClaimInput,
  RemoteConnectionProfileInput
} from '../main/connections'

export type BrazierConnection = {
  address: string
  profile: ConnectionProfileSummary
  daemon: DaemonInfo
}

export type BrazierConnectionProfile = ConnectionProfileView
export type BrazierConnectionProfileSummary = ConnectionProfileSummary
export type BrazierRemoteConnectionProfile = Extract<ConnectionProfileView, { kind: 'remote' }>
export type BrazierConnectionTestResult = ConnectionTestResult
export type BrazierRemoteConnectionProfileInput = RemoteConnectionProfileInput
export type BrazierPairingClaimInput = PairingClaimInput
export type BrazierClaimedConnection = ClaimedConnection

export type BrazierFlags = {
  forceWelcome: boolean
}

export type BrazierServerKey = {
  id: string
  name: string
  createdAt: number
}

export type BrazierServerSettings = {
  enabled: boolean
  port: number
  apiKeyEnabled: boolean
  hasApiKeys: boolean
  localhostOnly: boolean
  allowInsecureRemote: boolean
  jitLoading: boolean
  keys: BrazierServerKey[]
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

// Electron rewrites handler failures as "Error invoking remote method
// '<channel>': Error: <message>"; the renderer shows the message to people.
const REMOTE_ERROR_PREFIX = /^Error invoking remote method '[^']*': (?:[A-Za-z]*Error: )?/

async function invoke(channel: string, ...args: unknown[]): Promise<any> {
  try {
    return await ipcRenderer.invoke(channel, ...args)
  } catch (error) {
    if (error instanceof Error) error.message = error.message.replace(REMOTE_ERROR_PREFIX, '')
    throw error
  }
}

const invokeAgent = async (command: WorkerCommandInput): Promise<unknown> => {
  // Agent mode can be the first daemon consumer after launch. Resolve the
  // selected profile first so the main-process supervisor receives the same
  // connection that renderer REST calls use.
  await invoke('brazier:connection')
  return invoke(AGENT_IPC.invoke, command)
}

const agentMessageListeners = new Set<(message: WorkerMessage) => void>()
let agentMessageDispatcher:
  | ((_event: unknown, message: WorkerMessage) => void)
  | null = null

function ensureAgentMessageDispatcher(): void {
  if (agentMessageDispatcher) return
  const handler = (_event: unknown, message: WorkerMessage): void => {
    for (const listener of [...agentMessageListeners]) listener(message)
  }
  agentMessageDispatcher = handler
  ipcRenderer.on(AGENT_IPC.event, handler)
}

function removeAgentMessageListener(
  listener: (message: WorkerMessage) => void
): void {
  agentMessageListeners.delete(listener)
  if (agentMessageListeners.size === 0 && agentMessageDispatcher) {
    ipcRenderer.removeListener(AGENT_IPC.event, agentMessageDispatcher)
    agentMessageDispatcher = null
  }
}

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
  cancel: (sessionId: string): Promise<unknown> =>
    ipcRenderer.invoke(AGENT_IPC.invoke, { type: 'cancel', sessionId }),
  compact: (sessionId: string): Promise<unknown> => invokeAgent({ type: 'compact', sessionId }),
  setModel: (sessionId: string, model: AgentModelReference): Promise<unknown> =>
    invokeAgent({ type: 'set-model', sessionId, model }),
  setTools: (sessionId: string, tools: string[]): Promise<unknown> =>
    invokeAgent({ type: 'set-tools', sessionId, tools }),
  setPermissionMode: (sessionId: string, mode: AgentPermissionMode): Promise<unknown> =>
    invokeAgent({ type: 'set-permission-mode', sessionId, mode }),
  closeSession: (sessionId: string): Promise<unknown> =>
    ipcRenderer.invoke(AGENT_IPC.invoke, { type: 'close-session', sessionId }),
  status: (): Promise<AgentWorkerStatus> => invoke(AGENT_IPC.status),
  restart: (): Promise<AgentWorkerStatus> => invoke(AGENT_IPC.restart),
  onStatus: (listener: (status: AgentWorkerStatus) => void): (() => void) => {
    const handler = (_event: unknown, status: AgentWorkerStatus): void =>
      listener(status)
    ipcRenderer.on(AGENT_IPC.statusChanged, handler)
    return () => {
      ipcRenderer.removeListener(AGENT_IPC.statusChanged, handler)
    }
  },
  onMessage: (listener: (message: WorkerMessage) => void): (() => void) => {
    agentMessageListeners.add(listener)
    ensureAgentMessageDispatcher()
    let removed = false
    return () => {
      if (removed) return
      removed = true
      removeAgentMessageListener(listener)
    }
  }
}

contextBridge.exposeInMainWorld('brazier', {
  getConnection: (): Promise<BrazierConnection> => invoke('brazier:connection'),
  verifyConnection: (): Promise<ConnectionVerification> =>
    invoke('brazier:connection-profiles:verify'),
  previewPairing: (origin: string, id: string, code: string): Promise<PairingPreview> =>
    invoke('brazier:connection-profiles:preview-pairing', origin, id, code),
  verifyPairingOrigin: (origin: string): Promise<string> =>
    invoke('brazier:connection-profiles:verify-pairing-origin', origin),
  listConnectionProfiles: (): Promise<BrazierConnectionProfile[]> =>
    invoke('brazier:connection-profiles:list'),
  getCurrentConnectionProfile: (): Promise<BrazierConnectionProfileSummary> =>
    invoke('brazier:connection-profiles:current'),
  upsertConnectionProfile: (
    profile: BrazierRemoteConnectionProfileInput
  ): Promise<BrazierRemoteConnectionProfile> =>
    invoke('brazier:connection-profiles:upsert', profile),
  testConnectionProfile: (
    idOrProfile: string | BrazierRemoteConnectionProfileInput
  ): Promise<BrazierConnectionTestResult> =>
    invoke('brazier:connection-profiles:test', idOrProfile),
  claimConnectionProfile: (
    input: BrazierPairingClaimInput
  ): Promise<BrazierClaimedConnection> =>
    invoke('brazier:connection-profiles:claim-pairing', input),
  selectConnectionProfile: (id: string): Promise<BrazierConnectionProfileSummary> =>
    invoke('brazier:connection-profiles:select', id),
  deleteConnectionProfile: (id: string): Promise<boolean> =>
    invoke('brazier:connection-profiles:delete', id),
  onConnectionProfileChanged: (
    listener: (profile: BrazierConnectionProfileSummary) => void
  ): (() => void) => {
    const handler = (_event: unknown, profile: BrazierConnectionProfileSummary): void =>
      listener(profile)
    ipcRenderer.on('brazier:connection-changed', handler)
    return () => ipcRenderer.removeListener('brazier:connection-changed', handler)
  },
  getServerSettings: (): Promise<BrazierServerSettings> => invoke('brazier:server-settings'),
  saveServerSettings: (
    settings: Omit<BrazierServerSettings, 'hasApiKeys' | 'keys'>
  ): Promise<BrazierServerSettings> => invoke('brazier:save-server-settings', settings),
  addServerApiKey: (
    name: string
  ): Promise<BrazierServerKey & { value: string }> => invoke('brazier:add-server-api-key', name),
  removeServerApiKey: (id: string): Promise<BrazierServerSettings> =>
    invoke('brazier:remove-server-api-key', id),
  copyText: (text: string): Promise<void> => invoke('brazier:copy-text', text),
  getFlags: (): Promise<BrazierFlags> => invoke('brazier:flags'),
  qualificationHost: (): Promise<{
    commit: string
    platform: 'macos' | 'linux' | 'windows'
    arch: string
    memory_gib: number
    gpu_vram_gib: number | null
    gpu_vendor: string | null
  }> => invoke('brazier:qualification-host'),
  checkForUpdates: (): Promise<{ supported: boolean }> => invoke('brazier:check-for-updates'),
  getUpdateSettings: (): Promise<{
    supported: boolean
    checkOnStartup: boolean
    autoDownload: boolean
  }> => invoke('brazier:get-update-settings'),
  saveUpdateSettings: (settings: {
    checkOnStartup?: boolean
    autoDownload?: boolean
  }): Promise<{
    supported: boolean
    checkOnStartup: boolean
    autoDownload: boolean
  }> => invoke('brazier:save-update-settings', settings),
  platform: process.platform,
  selectDirectory: (): Promise<string | null> => invoke('brazier:select-directory'),
  selectWorkspace: (): Promise<string | null> => invoke('brazier:select-workspace'),
  /** Pick a file (or an adapter directory) already on disk. Null when dismissed. */
  selectFile: (
    title: string,
    filters: Array<{ name: string; extensions: string[] }>
  ): Promise<string | null> => invoke('brazier:select-file', title, filters),
  /** Ask where to put a file and write it. Resolves null when dismissed. */
  saveFile: (suggestedName: string, data: ArrayBuffer): Promise<string | null> =>
    invoke('brazier:save-file', suggestedName, data),
  revealFile: (path: string): Promise<void> => invoke('brazier:reveal-file', path),
  computer: {
    setActive: (active: boolean): Promise<void> => invoke('brazier:computer:set-active', active),
    prepareSafety: (): Promise<void> => invoke('brazier:computer:prepare-safety'),
    inputGuardStatus: (): Promise<BrazierInputGuardStatus> =>
      invoke('brazier:computer:input-guard-status'),
    setupInputGuard: (): Promise<BrazierInputGuardStatus> =>
      invoke('brazier:computer:setup-input-guard'),
    onEscape: (listener: () => void): (() => void) => {
      const handler = (): void => listener()
      ipcRenderer.on('brazier:computer:escape', handler)
      return () => ipcRenderer.removeListener('brazier:computer:escape', handler)
    }
  },
  agent
})

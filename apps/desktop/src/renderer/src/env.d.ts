export {}

/** Vite emits the asset and resolves the import to its URL. */
declare module '*?url' {
  const url: string
  export default url
}

type BrazierConnectionProfile =
  | {
      id: 'local'
      name: 'Local'
      kind: 'local'
      baseUrl: null
      hostLabel: string
      hasApiKey: false
    }
  | {
      id: string
      name: string
      kind: 'remote'
      baseUrl: string
      hostLabel: string
      hasApiKey: boolean
    }

type BrazierConnectionProfileSummary = {
  id: string
  name: string
  kind: 'local' | 'remote'
  baseUrl: string | null
  hostLabel: string
}

type BrazierDaemonInfo = {
  product: 'brazier'
  version: string
  management_api: { major: number; minor: number }
  openai_api?: { chat_completions?: string; responses?: string }
  daemon?: {
    instance_id: string
    display_name: string
    platform: string
    architecture: string
  }
  client?: {
    id: string
    name: string
    scopes: Array<'inference' | 'management' | 'agent'>
    owner: boolean
  }
}

type BrazierConnection = {
  address: string
  profile: BrazierConnectionProfileSummary
  daemon: BrazierDaemonInfo
}

declare global {
  interface Window {
    brazier: {
      getConnection(): Promise<{
        address: string
        profile: BrazierConnectionProfileSummary
        daemon: BrazierDaemonInfo
      }>
      listConnectionProfiles(): Promise<BrazierConnectionProfile[]>
      getCurrentConnectionProfile(): Promise<BrazierConnectionProfileSummary>
      upsertConnectionProfile(profile: {
        id?: string
        name: string
        kind?: 'remote'
        baseUrl: string
        apiKey?: string | null
      }): Promise<Extract<BrazierConnectionProfile, { kind: 'remote' }>>
      testConnectionProfile(idOrProfile: string | {
        id?: string
        name: string
        kind?: 'remote'
        baseUrl: string
        apiKey?: string | null
      }): Promise<{
        profile: BrazierConnectionProfileSummary
        daemon: BrazierDaemonInfo
      }>
      claimConnectionProfile(input: {
        id?: string
        name: string
        baseUrl: string
        pairingId: string
        code: string
      }): Promise<{
        profile: BrazierConnectionProfileSummary
        daemon: BrazierDaemonInfo
        client: {
          id: string
          name: string
          scopes: Array<'inference' | 'management' | 'agent'>
          created_at: string
          last_used_at?: string | null
          revoked_at?: string | null
        }
      }>
      selectConnectionProfile(id: string): Promise<BrazierConnectionProfileSummary>
      deleteConnectionProfile(id: string): Promise<boolean>
      onConnectionProfileChanged(
        listener: (profile: BrazierConnectionProfileSummary) => void
      ): () => void
      getServerSettings(): Promise<{
        enabled: boolean
        port: number
        apiKeyEnabled: boolean
        hasApiKeys: boolean
        localhostOnly: boolean
        allowInsecureRemote: boolean
        jitLoading: boolean
        keys: Array<{ id: string; name: string; createdAt: number }>
      }>
      saveServerSettings(settings: {
        enabled: boolean
        port: number
        apiKeyEnabled: boolean
        localhostOnly: boolean
        allowInsecureRemote: boolean
        jitLoading: boolean
      }): Promise<{
        enabled: boolean
        port: number
        apiKeyEnabled: boolean
        hasApiKeys: boolean
        localhostOnly: boolean
        allowInsecureRemote: boolean
        jitLoading: boolean
        keys: Array<{ id: string; name: string; createdAt: number }>
      }>
      addServerApiKey(name: string): Promise<{
        id: string
        name: string
        value: string
        createdAt: number
      }>
      removeServerApiKey(id: string): Promise<{
        enabled: boolean
        port: number
        apiKeyEnabled: boolean
        hasApiKeys: boolean
        localhostOnly: boolean
        allowInsecureRemote: boolean
        jitLoading: boolean
        keys: Array<{ id: string; name: string; createdAt: number }>
      }>
      copyText(text: string): Promise<void>
      getFlags(): Promise<{
        forceWelcome: boolean
      }>
      qualificationHost(): Promise<{
        commit: string
        platform: 'macos' | 'linux' | 'windows'
        arch: string
        memory_gib: number
        gpu_vram_gib: number | null
        gpu_vendor: string | null
      }>
      /** Check the signed release feed; download still requires confirmation. */
      checkForUpdates(): Promise<{ supported: boolean }>
      getUpdateSettings(): Promise<{
        supported: boolean
        checkOnStartup: boolean
        autoDownload: boolean
      }>
      saveUpdateSettings(settings: {
        checkOnStartup?: boolean
        autoDownload?: boolean
      }): Promise<{
        supported: boolean
        checkOnStartup: boolean
        autoDownload: boolean
      }>
      platform: NodeJS.Platform
      selectDirectory(): Promise<string | null>
      /** Folder picker for an agent workspace. */
      selectWorkspace(): Promise<string | null>
      /** Pick a file — or an adapter directory — already on disk. */
      selectFile(
        title: string,
        filters: Array<{ name: string; extensions: string[] }>
      ): Promise<string | null>
      /** Save bytes somewhere the user chooses; null when they dismiss it. */
      saveFile(suggestedName: string, data: ArrayBuffer): Promise<string | null>
      revealFile(path: string): Promise<void>
      computer: {
        setActive(active: boolean): Promise<void>
        prepareSafety(): Promise<void>
        inputGuardStatus(): Promise<{
          supported: boolean
          installed: boolean
          secure: boolean
          ready: boolean
          current: boolean
          version: string | null
          detail: string
        }>
        setupInputGuard(): Promise<{
          supported: boolean
          installed: boolean
          secure: boolean
          ready: boolean
          current: boolean
          version: string | null
          detail: string
        }>
        onEscape(listener: () => void): () => void
      }
      /**
       * Agent mode bridge. Commands go to the agent worker process; events come
       * back on one channel. The renderer never touches the worker directly.
       */
      agent: {
        openSession(sessionId: string): Promise<unknown>
        run(sessionId: string, input: { text: string; images?: string[] }): Promise<unknown>
        cancel(sessionId: string): Promise<unknown>
        compact(sessionId: string): Promise<unknown>
        setModel(
          sessionId: string,
          model: { id: string; name?: string; contextWindow?: number; maxTokens?: number }
        ): Promise<unknown>
        setTools(sessionId: string, tools: string[]): Promise<unknown>
        setPermissionMode(
          sessionId: string,
          mode: import('../../agent/core/types').AgentPermissionMode
        ): Promise<unknown>
        closeSession(sessionId: string): Promise<unknown>
        status(): Promise<{ running: boolean; crashes: number }>
        onMessage(listener: (message: import('../../agent/core/protocol').WorkerMessage) => void): () => void
      }
    }
  }
}

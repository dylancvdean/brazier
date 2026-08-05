export {}

/** Vite emits the asset and resolves the import to its URL. */
declare module '*?url' {
  const url: string
  export default url
}

declare global {
  interface Window {
    brazier: {
      getConnection(): Promise<{
        address: string
        api_key: string | null
      }>
      getServerSettings(): Promise<{
        enabled: boolean
        port: number
        apiKeyEnabled: boolean
        hasApiKeys: boolean
        localhostOnly: boolean
        jitLoading: boolean
        keys: Array<{ id: string; name: string; createdAt: number }>
      }>
      saveServerSettings(settings: {
        enabled: boolean
        port: number
        apiKeyEnabled: boolean
        localhostOnly: boolean
        jitLoading: boolean
      }): Promise<{
        enabled: boolean
        port: number
        apiKeyEnabled: boolean
        hasApiKeys: boolean
        localhostOnly: boolean
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
        jitLoading: boolean
        keys: Array<{ id: string; name: string; createdAt: number }>
      }>
      copyText(text: string): Promise<void>
      getFlags(): Promise<{
        forceWelcome: boolean
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

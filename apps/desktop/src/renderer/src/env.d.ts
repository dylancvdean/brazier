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
      getFlags(): Promise<{
        forceWelcome: boolean
      }>
      platform: NodeJS.Platform
      selectDirectory(): Promise<string | null>
      /** Folder picker for an agent workspace. */
      selectWorkspace(): Promise<string | null>
      /** Save bytes somewhere the user chooses; null when they dismiss it. */
      saveFile(suggestedName: string, data: ArrayBuffer): Promise<string | null>
      revealFile(path: string): Promise<void>
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
        closeSession(sessionId: string): Promise<unknown>
        status(): Promise<{ running: boolean; crashes: number }>
        onMessage(listener: (message: import('../../agent/core/protocol').WorkerMessage) => void): () => void
      }
    }
  }
}

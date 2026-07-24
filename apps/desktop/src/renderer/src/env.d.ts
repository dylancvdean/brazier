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
    }
  }
}

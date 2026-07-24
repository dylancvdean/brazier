export {}

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

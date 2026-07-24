export {}

declare global {
  interface Window {
    brazier: {
      getConnection(): Promise<{
        address: string
        api_key: string | null
      }>
      platform: NodeJS.Platform
      selectDirectory(): Promise<string | null>
    }
  }
}

/** Poll immediately, then wait between requests. Cleanup ignores any late result. */
export function startPolling<T>(
  load: () => Promise<T>,
  onValue: (value: T) => void,
  intervalMs: number
): () => void {
  let stopped = false
  let timer: ReturnType<typeof setTimeout> | undefined
  async function poll(): Promise<void> {
    try {
      const value = await load()
      if (!stopped) onValue(value)
    } catch {
      // Transient failures retry on the next tick.
    } finally {
      if (!stopped) timer = setTimeout(() => void poll(), intervalMs)
    }
  }
  void poll()
  return () => {
    stopped = true
    clearTimeout(timer)
  }
}

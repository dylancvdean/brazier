/** Longest silence, measured in `data:` events, tolerated from a model stream. */
export const MODEL_STREAM_STALL_MS = 5 * 60_000

/** The model produced no `data:` event for too long. Comments do not count. */
export class SseStallError extends Error {
  constructor(timeoutMs: number) {
    super(`The model stopped responding: no output for ${Math.round(timeoutMs / 1000)} seconds.`)
    this.name = 'SseStallError'
  }
}

export type ReadSseOptions = {
  /**
   * Cancel the read when this long passes without a `data:` event.
   * Keep-alive comments keep the socket up but do not reset the timer.
   */
  stallTimeoutMs?: number
  signal?: AbortSignal
}

/** Read SSE data across arbitrary byte boundaries, releasing the stream on exit. */
export async function* readSseData(
  body: ReadableStream<Uint8Array>,
  options?: ReadSseOptions
): AsyncGenerator<string> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let line = ''
  let data: string[] = []
  let afterCr = false

  function* consume(text: string): Generator<string> {
    if (!text) return
    if (afterCr && text.startsWith('\n')) text = text.slice(1)
    afterCr = text.endsWith('\r')
    let start = 0
    for (const ending of text.matchAll(/\r\n|\r|\n/g)) {
      line += text.slice(start, ending.index)
      start = ending.index + ending[0].length
      if (line === '') {
        if (data.length) yield data.join('\n')
        data = []
      } else if (line === 'data' || line.startsWith('data:')) {
        const value = line.slice(5)
        data.push(value.startsWith(' ') ? value.slice(1) : value)
      }
      line = ''
    }
    line += text.slice(start)
  }

  const stallTimeoutMs = options?.stallTimeoutMs
  let stallTimer: ReturnType<typeof setTimeout> | undefined
  let stalled = false
  const armStall = (): void => {
    if (!stallTimeoutMs || stalled) return
    if (stallTimer) clearTimeout(stallTimer)
    stallTimer = setTimeout(() => {
      stalled = true
      void reader.cancel().catch(() => undefined)
    }, stallTimeoutMs)
  }
  const onAbort = (): void => {
    void reader.cancel().catch(() => undefined)
  }
  options?.signal?.addEventListener('abort', onAbort)
  armStall()

  /** Yield one chunk and report whether it contained a data event. */
  function* take(text: string): Generator<string> {
    let produced = false
    try {
      for (const item of consume(text)) {
        produced = true
        yield item
      }
    } finally {
      if (produced) armStall()
    }
  }

  try {
    if (options?.signal?.aborted) {
      const reason = options.signal.reason
      throw reason instanceof Error
        ? reason
        : new DOMException('The operation was aborted.', 'AbortError')
    }
    while (true) {
      let next: ReadableStreamReadResult<Uint8Array>
      try {
        next = await reader.read()
      } catch (error) {
        if (stalled && stallTimeoutMs) throw new SseStallError(stallTimeoutMs)
        throw error
      }
      if (stalled && stallTimeoutMs) throw new SseStallError(stallTimeoutMs)
      yield* take(decoder.decode(next.value, { stream: !next.done }))
      if (stalled && stallTimeoutMs) throw new SseStallError(stallTimeoutMs)
      if (next.done) break
    }
    // Some local runtimes close immediately after the final data line.
    yield* take('\n\n')
  } finally {
    if (stallTimer) clearTimeout(stallTimer)
    options?.signal?.removeEventListener('abort', onAbort)
    try {
      await reader.cancel()
    } catch {
      // Keep the original stream or consumer error.
    }
    reader.releaseLock()
  }
}

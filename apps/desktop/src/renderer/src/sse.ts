/** Read SSE data across arbitrary byte boundaries, releasing the stream on exit. */
export async function* readSseData(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
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

  try {
    while (true) {
      const { done, value } = await reader.read()
      yield* consume(decoder.decode(value, { stream: !done }))
      if (done) break
    }
    // Some local runtimes close immediately after the final data line.
    yield* consume('\n\n')
  } finally {
    try {
      await reader.cancel()
    } catch {
      // Keep the original stream or consumer error.
    }
    reader.releaseLock()
  }
}

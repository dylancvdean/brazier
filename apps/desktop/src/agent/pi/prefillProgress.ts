/**
 * Observe Brazier's provider-extension chunks without forking Pi's OpenAI
 * provider. The OpenAI SDK intentionally discards unknown chunk fields, so a
 * narrowly scoped fetch wrapper reads a cloned response while Pi consumes the
 * original byte-for-byte.
 */

export type PrefillProgress = {
  total: number
  cached: number
  processed: number
  elapsed_ms: number
  context_total?: number | null
}

type Listener = (progress: PrefillProgress) => void

const LISTENER_HEADER = 'x-brazier-prefill-listener'
const listeners = new Map<string, Listener>()
let installed = false

function listenerId(input: RequestInfo | URL, init?: RequestInit): string | null {
  const headers = new Headers(input instanceof Request ? input.headers : undefined)
  new Headers(init?.headers).forEach((value, key) => headers.set(key, value))
  return headers.get(LISTENER_HEADER)
}

async function inspect(response: Response, id: string): Promise<void> {
  if (!response.body) {
    listeners.delete(id)
    return
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const frames = buffer.split('\n\n')
      buffer = frames.pop() ?? ''
      for (const frame of frames) {
        const data = frame
          .split('\n')
          .find((line) => line.startsWith('data:'))
          ?.slice(5)
          .trim()
        if (!data || data === '[DONE]') continue
        const payload = JSON.parse(data) as { brazier?: { prefill?: PrefillProgress } }
        if (payload.brazier?.prefill) listeners.get(id)?.(payload.brazier.prefill)
      }
    }
  } catch {
    // Progress is presentational; Pi's original response remains authoritative.
  } finally {
    listeners.delete(id)
    reader.releaseLock()
  }
}

function install(): void {
  if (installed) return
  installed = true
  const original = globalThis.fetch.bind(globalThis)
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const id = listenerId(input, init)
    try {
      const response = await original(input, init)
      if (id && listeners.has(id)) void inspect(response.clone(), id)
      return response
    } catch (cause) {
      if (id) listeners.delete(id)
      throw cause
    }
  }
}

export function prefillListener(
  listener: Listener
): { id: string; headers: Record<string, string> } {
  install()
  const id = `prefill-${crypto.randomUUID()}`
  listeners.set(id, listener)
  return { id, headers: { [LISTENER_HEADER]: id } }
}

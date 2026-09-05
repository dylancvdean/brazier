import { describe, expect, it, vi } from 'vitest'
import { readSseData } from './sse'

function stream(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(chunk))
      controller.close()
    }
  })
}

async function collect(body: ReadableStream<Uint8Array>): Promise<string[]> {
  const values: string[] = []
  for await (const data of readSseData(body)) values.push(data)
  expect(body.locked).toBe(false)
  return values
}

describe('SSE decoding', () => {
  it.each(['\n', '\r\n', '\r'])('handles %j line endings at every byte boundary', async (newline) => {
    const encoded = new TextEncoder().encode([
      ': keepalive', 'event: delta', 'data: {"text":', 'data: "🔥"}', '',
      'data: [DONE]', '', ''
    ].join(newline))
    const expected = ['{"text":\n"🔥"}', '[DONE]']
    for (let index = 0; index <= encoded.length; index++) {
      expect(await collect(stream([encoded.slice(0, index), encoded.slice(index)]))).toEqual(expected)
    }
    expect(await collect(stream(Array.from(encoded, (byte) => Uint8Array.of(byte))))).toEqual(expected)
  })

  it('preserves data whitespace and accepts the final data line at EOF', async () => {
    expect(await collect(stream([new TextEncoder().encode('data:  hello  \n\ndata: final')])))
      .toEqual([' hello  ', 'final'])
  })

  it('cancels and unlocks an open response when the consumer stops', async () => {
    const cancel = vi.fn()
    const body = new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(new TextEncoder().encode('data: done\n\n')) },
      cancel
    })
    for await (const data of readSseData(body)) {
      expect(data).toBe('done')
      break
    }
    expect(cancel).toHaveBeenCalledOnce()
    expect(body.locked).toBe(false)
  })

  it('unlocks a failed response without hiding its error', async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) { controller.error(new Error('connection lost')) }
    })
    await expect(collect(body)).rejects.toThrow('connection lost')
    expect(body.locked).toBe(false)
  })
})

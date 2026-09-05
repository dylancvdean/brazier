import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  encodePcm16,
  filenameWithExtension,
  downloadModel,
  invalidateConnectionCache,
  messagesForCompletion,
  prefillProgressLabel,
  prepareModel,
  reasoningAfterTranscriptBoundary,
  streamCompletion,
  transcribeAudioIncrementally
} from './api'
import { setDaemonAvailability } from './daemonAvailability'
import type { Message } from './types'

function message(overrides: Partial<Message>): Message {
  return {
    id: 'message-1',
    conversation_id: 'conversation-1',
    parent_id: null,
    role: 'assistant',
    content: '',
    model: null,
    created_at: '2026-07-26T00:00:00Z',
    ...overrides
  }
}

describe('encodePcm16', () => {
  it('clamps normalized float audio into little-endian signed PCM', () => {
    const encoded = encodePcm16(new Float32Array([-2, -1, -0.5, 0, 0.5, 1, 2]))
    const view = new DataView(encoded.buffer)
    expect(Array.from({ length: 7 }, (_, index) => view.getInt16(index * 2, true))).toEqual([
      -32768,
      -32768,
      -16384,
      0,
      16383,
      32767,
      32767
    ])
  })
})

describe('transcribeAudioIncrementally', () => {
  afterEach(() => {
    invalidateConnectionCache()
    vi.unstubAllGlobals()
  })

  it('receives a partial before closing the independent PCM upload', async () => {
    const encoder = new TextEncoder()
    let eventController: ReadableStreamDefaultController<Uint8Array> | null = null
    const uploaded: number[] = []
    const fetchMock = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/v1/audio/transcriptions/sessions')) {
        return new Response(JSON.stringify({ id: 'stream-1' }), {
          status: 200,
          headers: { 'content-type': 'application/json' }
        })
      }
      if (url.endsWith('/stream-1/events')) {
        return new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              eventController = controller
              controller.enqueue(
                encoder.encode(
                  'event: transcription.delta\ndata: {"type":"transcription.delta","text":"hello "}\n\n'
                )
              )
            }
          }),
          { status: 200, headers: { 'content-type': 'text/event-stream' } }
        )
      }
      if (url.endsWith('/stream-1/audio')) {
        const reader = (init?.body as ReadableStream<Uint8Array>).getReader()
        while (true) {
          const chunk = await reader.read()
          if (chunk.done) break
          uploaded.push(...chunk.value)
        }
        eventController?.enqueue(
          encoder.encode(
            'event: transcription.done\ndata: {"type":"transcription.done","text":"hello world","engine":"streaming-asr","duration_ms":42}\n\n'
          )
        )
        eventController?.close()
        return new Response(null, { status: 204 })
      }
      throw new Error(`unexpected request: ${url}`)
    })
    vi.stubGlobal('window', {
      brazier: {
        getConnection: vi.fn().mockResolvedValue({
          address: 'http://127.0.0.1:9999',
          profile: { id: 'local' }
        })
      }
    })
    vi.stubGlobal('fetch', fetchMock)
    setDaemonAvailability('healthy')
    invalidateConnectionCache()

    let resolvePartial: ((text: string) => void) | null = null
    const partial = new Promise<string>((resolve) => {
      resolvePartial = resolve
    })
    const transcription = transcribeAudioIncrementally(16000, {
      onPartial: (text) => resolvePartial?.(text)
    })
    transcription.push(new Float32Array([0.5, -0.5]))

    expect(await partial).toBe('hello')
    // Seeing a delta cannot depend on finish; the upload is still open here.
    expect(uploaded).toHaveLength(4)
    const result = await transcription.finish()
    expect(result).toEqual({
      text: 'hello world',
      engine: 'streaming-asr',
      durationMs: 42
    })
  })

  it.each(['network', 'http'])('fails promptly when the PCM upload fails (%s)', async (failure) => {
    vi.stubGlobal('window', {
      brazier: { getConnection: vi.fn().mockResolvedValue({
        address: 'http://localhost:9999', profile: { id: 'local' }
      }) }
    })
    vi.stubGlobal('fetch', vi.fn(async (input, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/sessions')) return new Response(JSON.stringify({ id: 'broken' }))
      if (url.endsWith('/events')) {
        return new Response(new ReadableStream<Uint8Array>({
          start(controller) {
            init?.signal?.addEventListener('abort', () => controller.error(init.signal?.reason))
          }
        }))
      }
      if (failure === 'network') throw new TypeError('upload disconnected')
      return new Response(JSON.stringify({ error: { message: 'upload rejected' } }), { status: 500 })
    }))
    setDaemonAvailability('healthy')
    const transcription = transcribeAudioIncrementally(16000)
    await expect(transcription.done).rejects.toThrow(
      failure === 'network' ? 'upload disconnected' : 'upload rejected'
    )
  })
})

describe('stream completion boundaries', () => {
  afterEach(() => {
    invalidateConnectionCache()
    vi.unstubAllGlobals()
  })

  function respond(data: string, close = true): ReturnType<typeof vi.fn> {
    vi.stubGlobal('window', { brazier: { getConnection: vi.fn().mockResolvedValue({
      address: 'http://localhost:9999', profile: { id: 'local' }
    }) } })
    const cancel = vi.fn()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(data))
        if (close) controller.close()
      },
      cancel
    }))))
    setDaemonAvailability('healthy')
    return cancel
  }

  it('returns chat output and closes the reader at DONE even if the server stays open', async () => {
    const cancel = respond('data: {"choices":[{"delta":{"content":"hello"}}]}\r\n\r\ndata: [DONE]\r\n\r\n', false)
    const token = vi.fn()
    const result = await streamCompletion([], 'model', new AbortController().signal, token, { dropReasoningBetweenTurns: false })
    expect(result.responseText).toBe('hello')
    expect(token).toHaveBeenCalledExactlyOnceWith('hello')
    expect(cancel).toHaveBeenCalledOnce()
  })

  it('reports truncated model preparation instead of claiming success', async () => {
    respond('data: {"phase":"loading","message":"Loading model"}\n\n')
    await expect(prepareModel('model')).rejects.toThrow('before the model was ready')
  })

  it('closes model preparation as soon as the model is ready', async () => {
    const cancel = respond('data: {"status":"ready","residency":null}\r\n\r\n', false)
    await expect(prepareModel('model')).resolves.toBeNull()
    expect(cancel).toHaveBeenCalledOnce()
  })

  it('accepts a download completion without a trailing blank line', async () => {
    respond('data: {"phase":"done","done":true,"result":{"model_id":"test","path":"/models/test.gguf"}}')
    const progress = vi.fn()
    await expect(downloadModel('owner/repo', 'test.gguf', progress)).resolves.toMatchObject({ path: '/models/test.gguf' })
    expect(progress).toHaveBeenCalledOnce()
  })
})

describe('messagesForCompletion', () => {
  it('does not send the human-only generated-media display back to the model', () => {
    const payload = messagesForCompletion([
      message({
        content: [
          {
            type: 'brazier_blob',
            brazier_blob: {
              sha256: 'visible-image',
              mime_type: 'image/png',
              name: 'generated-image'
            }
          }
        ],
        metadata: { generated_media_display: true }
      }),
      message({ id: 'user-2', role: 'user', content: 'Now make it warmer.' })
    ])

    expect(payload).toEqual([{ role: 'user', content: 'Now make it warmer.' }])
  })

  it('keeps deferred generated media as system context for the next user turn', () => {
    const content = [
      { type: 'text' as const, text: 'Generated media context.' },
      {
        type: 'brazier_blob' as const,
        brazier_blob: {
          sha256: 'context-image',
          mime_type: 'image/png',
          name: 'generated-image'
        }
      }
    ]
    const payload = messagesForCompletion([
      message({ role: 'system', content }),
      message({ id: 'user-2', role: 'user', content: 'What should change?' })
    ])

    expect(payload[0]).toEqual({ role: 'system', content })
    expect(payload[1]).toEqual({ role: 'user', content: 'What should change?' })
  })

  it('round-trips assistant reasoning_content so Jinja can keep parsing tools', () => {
    const payload = messagesForCompletion([
      message({
        content: '',
        tool_calls: [
          {
            id: 'call_1',
            type: 'function',
            function: { name: 'run_javascript', arguments: '{"code":"1"}' }
          }
        ],
        metadata: { reasoning_content: 'Need a calculator.' }
      }),
      message({
        id: 'tool-1',
        role: 'tool',
        tool_call_id: 'call_1',
        content: '1'
      })
    ])

    expect(payload[0]).toMatchObject({
      role: 'assistant',
      reasoning_content: 'Need a calculator.'
    })
    expect(payload[1]).toMatchObject({ role: 'tool', tool_call_id: 'call_1' })
  })

  it('drops prior-turn reasoning when asked, but keeps current-turn tool reasoning', () => {
    const payload = messagesForCompletion(
      [
        message({
          id: 'a1',
          content: 'first answer',
          metadata: { reasoning_content: 'old thought' }
        }),
        message({ id: 'u2', role: 'user', content: 'next' }),
        message({
          id: 'a2',
          content: '',
          tool_calls: [
            {
              id: 'call_1',
              type: 'function',
              function: { name: 'run_javascript', arguments: '{}' }
            }
          ],
          metadata: { reasoning_content: 'current thought' }
        }),
        message({
          id: 'tool-1',
          role: 'tool',
          tool_call_id: 'call_1',
          content: '1'
        })
      ],
      { dropReasoningBetweenTurns: true }
    )

    expect(payload[0]).toEqual({ role: 'assistant', content: 'first answer' })
    expect(payload[1]).toEqual({ role: 'user', content: 'next' })
    expect(payload[2]).toMatchObject({
      role: 'assistant',
      reasoning_content: 'current thought'
    })
  })
})

describe('reasoningAfterTranscriptBoundary', () => {
  it('commits reasoning at an assistant tool-round boundary without clearing it for tool plumbing', () => {
    expect(
      reasoningAfterTranscriptBoundary('first-round reasoning', {
        role: 'assistant',
        content: '',
        tool_calls: []
      })
    ).toBe('')
    expect(
      reasoningAfterTranscriptBoundary('final-round reasoning', {
        role: 'system',
        content: 'generated media context'
      })
    ).toBe('final-round reasoning')
  })
})

describe('prefillProgressLabel', () => {
  it('shows both prompt progress and configured context usage', () => {
    expect(
      prefillProgressLabel({
        total: 2_048,
        cached: 1_024,
        processed: 1_536,
        elapsed_ms: 87,
        context_total: 32_768
      })
    ).toBe('Prefilling 1,536 / 2,048 tokens · context 2,048 / 32,768')
  })

  it('clamps a server overrun and tolerates an unknown context limit', () => {
    expect(
      prefillProgressLabel({
        total: 128,
        cached: 0,
        processed: 129,
        elapsed_ms: 10
      })
    ).toBe('Prefilling 128 / 128 tokens')
  })
})

describe('filenameWithExtension', () => {
  it('adds a PDF extension when a generated or rendered name has none', () => {
    expect(filenameWithExtension('generated-image', 'application/pdf', 'fallback.pdf')).toBe(
      'generated-image.pdf'
    )
    expect(filenameWithExtension('report.pdf', 'application/pdf', 'fallback.pdf')).toBe(
      'report.pdf'
    )
  })
})

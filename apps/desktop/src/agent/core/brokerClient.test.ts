import { afterEach, describe, expect, it, vi } from 'vitest'

import { BrokerClient, BrokerError, ContextOverflowError } from './brokerClient'

function sseResponse(frames: Array<Record<string, unknown> | '[DONE]'>): Response {
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const frame of frames) {
        const data = frame === '[DONE]' ? '[DONE]' : JSON.stringify(frame)
        controller.enqueue(encoder.encode(`data: ${data}\n\n`))
      }
      controller.close()
    }
  })
  return new Response(stream, {
    status: 200,
    headers: { 'content-type': 'text/event-stream' }
  })
}

describe('BrokerClient.streamChatCompletion', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('streams content, reasoning, prefill, usage, and incremental + daemon tool calls in one pass', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      sseResponse([
        {
          choices: [{ index: 0, delta: {}, finish_reason: null }],
          brazier: {
            prefill: {
              total: 512,
              cached: 256,
              processed: 256,
              elapsed_ms: 42,
              context_total: 32768
            }
          }
        },
        {
          choices: [{ index: 0, delta: { reasoning_content: 'Checking file...' }, finish_reason: null }]
        },
        {
          choices: [{ index: 0, delta: { content: 'null' }, finish_reason: null }]
        },
        {
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  { index: 0, id: 'call_1', type: 'function', function: { name: 'fs_read', arguments: '{"pa' } }
                ]
              },
              finish_reason: null
            }
          ]
        },
        {
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [{ index: 0, type: 'function', function: { arguments: 'th":"src/a.ts"}' } }]
              },
              finish_reason: null
            }
          ]
        },
        {
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    id: 'call_1',
                    type: 'function',
                    function: { name: 'fs_read', arguments: '{"path":"src/a.ts"}' }
                  }
                ]
              },
              finish_reason: 'tool_calls'
            }
          ]
        },
        {
          choices: [{ index: 0, delta: {}, finish_reason: null }],
          brazier: { generation: { prompt_tokens: 512, completion_tokens: 24 } }
        },
        '[DONE]'
      ])
    )

    const client = new BrokerClient({ address: 'http://127.0.0.1:1710', apiKey: 'secret-token' })
    const prefills: unknown[] = []
    const contentDeltas: string[] = []
    const reasoningDeltas: string[] = []

    const result = await client.streamChatCompletion(
      {
        model: 'gguf:qwen3.gguf',
        messages: [{ role: 'user', content: 'read src/a.ts' }],
        slot: 2
      },
      {
        onPrefill: (p) => prefills.push(p),
        onContentDelta: (d) => contentDeltas.push(d),
        onReasoningDelta: (d) => reasoningDeltas.push(d)
      }
    )

    expect(fetchSpy).toHaveBeenCalledTimes(1)
    const [url, init] = fetchSpy.mock.calls[0]!
    expect(url).toBe('http://127.0.0.1:1710/v1/chat/completions')
    const headers = new Headers(init?.headers)
    expect(headers.get('authorization')).toBe('Bearer secret-token')
    expect(headers.get('x-brazier-mode')).toBe('agent')
    expect(headers.get('x-brazier-slot')).toBe('2')
    expect(headers.get('content-type')).toBe('application/json')

    expect(prefills).toHaveLength(1)
    expect(contentDeltas).toEqual([])
    expect(reasoningDeltas).toEqual(['Checking file...'])
    expect(result.content).toBe('')
    expect(result.reasoning).toBe('Checking file...')
    expect(result.promptTokens).toBe(512)
    expect(result.completionTokens).toBe(24)
    expect(result.contextTotal).toBe(32768)
    expect(result.finishReason).toBe('tool_calls')
    expect(result.toolCalls).toEqual([
      { id: 'call_1', name: 'fs_read', arguments: '{"path":"src/a.ts"}' }
    ])
  })

  it('throws ContextOverflowError on HTTP 400/500 overflow responses and streamed error frames', async () => {
    const client = new BrokerClient({ address: 'http://127.0.0.1:1710', apiKey: null })

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          error: { message: 'request (9000 tokens) exceeds the available context size (8192 tokens)' }
        }),
        { status: 400 }
      )
    )
    await expect(
      client.streamChatCompletion({
        model: 'gguf:demo.gguf',
        messages: [{ role: 'user', content: 'hi' }]
      })
    ).rejects.toBeInstanceOf(ContextOverflowError)

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      sseResponse([
        {
          choices: [{ index: 0, delta: {}, finish_reason: 'error' }],
          error: { message: 'prompt is too long for n_ctx=4096' }
        }
      ])
    )
    await expect(
      client.streamChatCompletion({
        model: 'gguf:demo.gguf',
        messages: [{ role: 'user', content: 'hi' }]
      })
    ).rejects.toBeInstanceOf(ContextOverflowError)

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(JSON.stringify({ error: { message: 'upstream gateway timeout' } }), {
        status: 502
      })
    )
    await expect(
      client.streamChatCompletion({
        model: 'gguf:demo.gguf',
        messages: [{ role: 'user', content: 'hi' }]
      })
    ).rejects.toThrow(BrokerError)
  })
})

import { describe, expect, it } from 'vitest'

import { buildChatDisplayItems } from './chatDisplay'
import type { Message } from './types'

function message(id: string, role: Message['role'], overrides: Partial<Message> = {}): Message {
  return {
    id,
    conversation_id: 'conversation-1',
    parent_id: null,
    role,
    content: '',
    model: null,
    created_at: '2026-07-27T00:00:00Z',
    ...overrides
  }
}

describe('buildChatDisplayItems', () => {
  it('renders a generated image as one ordered assistant turn without repeated reasoning', () => {
    const items = buildChatDisplayItems([
      message('user', 'user', { content: 'Make a picture.' }),
      message('call', 'assistant', {
        tool_calls: [
          {
            id: 'call-1',
            type: 'function',
            function: { name: 'generate_image', arguments: '{"prompt":"A cottage"}' }
          }
        ],
        metadata: { reasoning_content: 'I should create a cottage.' }
      }),
      message('result', 'tool', {
        tool_call_id: 'call-1',
        content: 'Image generation succeeded.'
      }),
      message('context', 'system', {
        content: [
          { type: 'text', text: 'The image was displayed.' },
          {
            type: 'brazier_blob',
            brazier_blob: {
              sha256: 'image-1',
              mime_type: 'image/png',
              name: 'generated-image'
            }
          }
        ]
      }),
      message('answer', 'assistant', {
        content: 'Here is your picture.',
        metadata: {
          reasoning_content:
            'I should create a cottage.\nThe image is complete, so I should confirm completion.'
        }
      }),
      message('display', 'assistant', {
        source: 'assistant_chat',
        metadata: { generated_media_display: true },
        content: [
          {
            type: 'brazier_blob',
            brazier_blob: {
              sha256: 'image-1',
              mime_type: 'image/png',
              name: 'generated-image-1'
            }
          }
        ]
      })
    ])

    expect(items).toHaveLength(2)
    expect(items[1].kind).toBe('assistant')
    if (items[1].kind !== 'assistant') throw new Error('expected assistant turn')
    expect(items[1].reasoning).toBe(
      'I should create a cottage.\nThe image is complete, so I should confirm completion.'
    )
    expect(items[1].answer).toBe('Here is your picture.')
    expect(items[1].trace.map((segment) => segment.kind)).toEqual([
      'reasoning',
      'tool',
      'reasoning'
    ])
    const tool = items[1].trace[1]
    if (tool.kind !== 'tool') throw new Error('expected tool segment')
    expect(tool.records[0]).toMatchObject({
      name: 'generate_image',
      output: 'Image generation succeeded.',
      media: [{ sha256: 'image-1', mime_type: 'image/png' }]
    })
  })

  it('keeps text on either side of a tool call in execution order', () => {
    const items = buildChatDisplayItems([
      message('user', 'user', { content: 'Show me.' }),
      message('before', 'assistant', {
        content: 'I will make that now.',
        tool_calls: [
          {
            id: 'call-1',
            type: 'function',
            function: { name: 'generate_image', arguments: '{"prompt":"A lake"}' }
          }
        ],
        metadata: { reasoning_content: 'I need an image.' }
      }),
      message('result', 'tool', {
        tool_call_id: 'call-1',
        content: 'Image generation succeeded.'
      }),
      message('after', 'assistant', { content: 'It is ready.' })
    ])

    const turn = items[1]
    if (turn.kind !== 'assistant') throw new Error('expected assistant turn')
    expect(turn.trace.map((segment) => segment.kind)).toEqual(['reasoning', 'text', 'tool'])
    expect(turn.answer).toBe('It is ready.')
    expect(turn.reasoning).toBe('I need an image.')
  })

  it('attaches a fetched document to the tool call that produced it', () => {
    const items = buildChatDisplayItems([
      message('user', 'user', { content: 'Read this.' }),
      message('fetch', 'assistant', {
        tool_calls: [
          {
            id: 'call-fetch',
            type: 'function',
            function: {
              name: 'fetch_url',
              arguments: '{"url":"https://example.com/report.pdf"}'
            }
          }
        ],
        metadata: { reasoning_content: 'Let me fetch the report.' }
      }),
      message('result', 'tool', {
        tool_call_id: 'call-fetch',
        content: 'Fetched and attached PDF report.pdf. It is 3 pages long.'
      }),
      message('read', 'assistant', {
        tool_calls: [
          {
            id: 'call-read',
            type: 'function',
            function: { name: 'doc_read', arguments: '{"document":"abc123","start_page":1}' }
          }
        ],
        metadata: { reasoning_content: 'Now read the first pages.' }
      }),
      message('read-result', 'tool', {
        tool_call_id: 'call-read',
        content: '[pages 1-3 of 3]\ntext'
      }),
      message('answer', 'assistant', { content: 'Here is the report.' }),
      message('display', 'assistant', {
        source: 'assistant_chat',
        metadata: { generated_media_display: true },
        content: [
          {
            type: 'brazier_blob',
            brazier_blob: {
              sha256: 'pdf-blob-1',
              mime_type: 'application/pdf',
              name: 'report.pdf',
              call_id: 'call-fetch'
            }
          }
        ]
      })
    ])

    const turn = items[1]
    if (turn.kind !== 'assistant') throw new Error('expected assistant turn')
    const fetch = turn.trace.find((segment) => segment.kind === 'tool')?.records[0]
    expect(fetch).toBeDefined()
    expect(fetch?.name).toBe('fetch_url')
    // The PDF lands on the fetch_url pill, not the later doc_read pill.
    expect(fetch?.media).toEqual([
      { sha256: 'pdf-blob-1', mime_type: 'application/pdf', original_name: 'report.pdf', call_id: 'call-fetch' }
    ])
  })
})

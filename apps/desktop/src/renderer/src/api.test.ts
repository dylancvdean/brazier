import { describe, expect, it } from 'vitest'

import { messagesForCompletion } from './api'
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
})

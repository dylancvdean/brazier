import { describe, expect, it, vi } from 'vitest'

import { clipProse, mergeSummary, renderTranscript, requestModelSummary } from './compaction'

const base = {
  baseUrl: 'http://127.0.0.1:1710/v1',
  apiKey: 'brazier-local',
  model: 'gguf:acme/demo.gguf',
  transcript: 'user: fix the build\nassistant: the linker flag was wrong',
  facts: 'Files changed: src/main.rs'
}

function replyWith(content: unknown, ok = true): typeof fetch {
  return vi.fn(async () =>
    new Response(JSON.stringify({ choices: [{ message: { content } }] }), {
      status: ok ? 200 : 500
    })
  ) as unknown as typeof fetch
}

describe('requestModelSummary', () => {
  it('asks the session model and returns its prose', async () => {
    const fetchImpl = replyWith('The build failed on a linker flag, which was corrected.')
    const summary = await requestModelSummary({ ...base, fetchImpl })
    expect(summary).toBe('The build failed on a linker flag, which was corrected.')

    const [url, init] = (fetchImpl as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('http://127.0.0.1:1710/v1/chat/completions')
    expect(new Headers((init as RequestInit).headers).get('x-brazier-mode')).toBe('agent')
    const body = JSON.parse((init as RequestInit).body as string)
    expect(body.model).toBe(base.model)
    expect(body.stream).toBe(false)
    // The facts go in as context so the narrative can refer to them.
    expect(body.messages[1].content).toContain('src/main.rs')
    expect(body.messages[1].content).toContain('linker flag')
  })

  /**
   * Compaction usually runs because the context is already full. Failing there
   * would take the session down at its least recoverable moment, so every
   * failure has the same outcome: no prose, and the digest stands alone.
   */
  it('gives up quietly on anything that goes wrong', async () => {
    expect(await requestModelSummary({ ...base, fetchImpl: replyWith('text', false) })).toBeNull()
    expect(await requestModelSummary({ ...base, fetchImpl: replyWith('   ') })).toBeNull()
    expect(await requestModelSummary({ ...base, fetchImpl: replyWith(null) })).toBeNull()
    const throws = vi.fn(async () => {
      throw new Error('connection refused')
    }) as unknown as typeof fetch
    expect(await requestModelSummary({ ...base, fetchImpl: throws })).toBeNull()
  })

  it('stops waiting rather than blocking compaction', async () => {
    const hangs = vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => reject(new Error('aborted')))
        })
    ) as unknown as typeof fetch
    expect(await requestModelSummary({ ...base, fetchImpl: hangs, timeoutMs: 5 })).toBeNull()
  })
})

describe('clipProse', () => {
  it('leaves an ordinary summary alone', () => {
    expect(clipProse('Two sentences. That is all.')).toBe('Two sentences. That is all.')
  })

  /** The instruction asks for eight sentences; nothing enforced it. */
  it('cuts an essay at a sentence boundary', () => {
    const essay = 'A sentence about the work. '.repeat(200)
    const clipped = clipProse(essay)
    expect(clipped.length).toBeLessThan(essay.length)
    expect(clipped.endsWith('.')).toBe(true)
  })

  it('marks a cut it could not make cleanly', () => {
    const unbroken = 'x'.repeat(5000)
    expect(clipProse(unbroken).endsWith('…')).toBe(true)
  })
})

describe('mergeSummary', () => {
  /** A model that forgets a changed file must not erase it from the session. */
  it('keeps the machine-built facts under the narrative', () => {
    expect(mergeSummary('Prose about the work.', 'Files changed: a.rs')).toBe(
      'Prose about the work.\n\nFiles changed: a.rs'
    )
  })

  it('is the facts alone when there is no narrative', () => {
    expect(mergeSummary(null, 'Files changed: a.rs')).toBe('Files changed: a.rs')
  })
})

describe('renderTranscript', () => {
  it('renders turns oldest first and truncates tool output', () => {
    const rendered = renderTranscript([
      { role: 'user', text: 'run the tests' },
      { role: 'tool', tool: 'shell_run', output: 'line one\nline two\nline three\nline four' },
      { role: 'assistant', text: 'one test failed' },
      { role: 'assistant', text: '' }
    ])
    expect(rendered).toBe(
      [
        'user: run the tests',
        'tool shell_run: line one line two line three',
        'assistant: one test failed'
      ].join('\n')
    )
  })
})

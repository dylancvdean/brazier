import { describe, expect, it } from 'vitest'

import { parseFaraOutput } from './faraAdapter'

describe('parseFaraOutput', () => {
  it('parses an XML tool_call with thought prefix', () => {
    const text = `I should open the site.
<tool_call>
{"name":"computer_use","arguments":{"action":"visit_url","url":"https://example.com"}}
</tool_call>`
    const parsed = parseFaraOutput(text)
    expect(parsed.thought).toContain('open the site')
    expect(parsed.actions).toEqual([{ type: 'visit_url', url: 'https://example.com' }])
  })

  it('parses click coordinates', () => {
    const text = `<tool_call>
{"name":"computer_use","arguments":{"action":"left_click","coordinate":[120,240]}}
</tool_call>`
    const parsed = parseFaraOutput(text)
    expect(parsed.actions[0]).toEqual({ type: 'left_click', x: 120, y: 240 })
  })

  it('reads <think> blocks', () => {
    const text = `<think>plan next</think>
<tool_call>
{"name":"computer_use","arguments":{"action":"wait","milliseconds":500}}
</tool_call>`
    const parsed = parseFaraOutput(text)
    expect(parsed.thought).toBe('plan next')
    expect(parsed.actions[0]).toEqual({ type: 'wait', milliseconds: 500 })
  })

  it('accepts bare JSON without XML wrappers', () => {
    const parsed = parseFaraOutput(
      '{"name":"computer_use","arguments":{"action":"terminate","response":"done"}}'
    )
    expect(parsed.actions).toEqual([{ type: 'terminate', response: 'done' }])
  })

  it('parses pause_and_memorize_fact into a memorize action', () => {
    const parsed = parseFaraOutput(`<tool_call>
{"name":"computer_use","arguments":{"action":"pause_and_memorize_fact","fact":"The account is #88321"}}
</tool_call>`)
    expect(parsed.actions).toEqual([
      { type: 'memorize', fact: 'The account is #88321' }
    ])
  })

  it('surfaces an error action when an XML tool call block is malformed', () => {
    const parsed = parseFaraOutput(`<tool_call>
{"not valid json"}
</tool_call>`)
    expect(parsed.actions).toHaveLength(1)
    expect(parsed.actions[0].type).toBe('error')
  })

  it('surfaces an error action when bare JSON is malformed', () => {
    const parsed = parseFaraOutput('{ this is not json')
    expect(parsed.actions).toHaveLength(1)
    expect(parsed.actions[0].type).toBe('error')
    if (parsed.actions[0].type === 'error') {
      expect(parsed.actions[0].error).toBe('parse_error')
      expect(typeof parsed.actions[0].raw).toBe('string')
    }
  })
})

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
})

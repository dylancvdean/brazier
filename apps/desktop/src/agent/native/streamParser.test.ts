import { describe, expect, it } from 'vitest'

import { extractTextToolCalls, ThinkStreamSplitter } from './streamParser'

describe('ThinkStreamSplitter', () => {
  it('splits <think>, <thought>, and <|thought|> into reasoning even across 1-char chunks', () => {
    for (const [open, close] of [
      ['<think>', '</think>'],
      ['<thought>', '</thought>'],
      ['<|thought|>', '<|/thought|>']
    ] as const) {
      const splitter = new ThinkStreamSplitter()
      const input = `${open}plan step 1${close}\n\nVisible answer.`
      const chunks = [...input].flatMap((ch) => splitter.push(ch))
      chunks.push(...splitter.flush())

      const reasoning = chunks
        .filter((c) => c.channel === 'reasoning')
        .map((c) => c.delta)
        .join('')
      const visible = chunks
        .filter((c) => c.channel === 'text')
        .map((c) => c.delta)
        .join('')

      expect(reasoning).toBe('plan step 1')
      expect(visible).toBe('Visible answer.')
    }
  })

  it('leaves inline tags alone when backend reasoning_content is already active', () => {
    const splitter = new ThinkStreamSplitter()
    splitter.markBackendReasoning()
    const chunks = [...splitter.push('<think>literal</think> ok'), ...splitter.flush()]
    expect(chunks.every((c) => c.channel === 'text')).toBe(true)
    expect(splitter.visibleText).toBe('<think>literal</think> ok')
  })

  it('holds back <tool_call>, <function=, [TOOL_CALLS], and <|channel|>commentary prefixes', () => {
    for (const raw of [
      'Checking file.\n<tool_call>{"name":"fs_read","arguments":{"path":"a.ts"}}</tool_call>',
      'Patching.\n<function=fs_patch><parameter=path>a.ts</parameter></function>',
      'Listing.\n[TOOL_CALLS] [{"name":"fs_list","arguments":{"path":"."}}]',
      'Reading.\n<|channel|>commentary to=functions.fs_read <|constrain|>json<|message|>{"path":"a.ts"}<|call|>'
    ]) {
      const splitter = new ThinkStreamSplitter()
      // Stream 3 characters at a time to exercise prefix boundary buffering.
      const streamed: string[] = []
      for (let i = 0; i < raw.length; i += 3) {
        for (const out of splitter.push(raw.slice(i, i + 3))) {
          if (out.channel === 'text') streamed.push(out.delta)
        }
      }
      for (const out of splitter.flush()) {
        if (out.channel === 'text') streamed.push(out.delta)
      }
      const visible = streamed.join('')
      expect(visible).not.toMatch(/<tool_call>|<function=|\[TOOL_CALLS\]|<\|channel\|>commentary/)
      expect(splitter.heldBackText.length).toBeGreaterThan(10)
    }
  })
})

describe('extractTextToolCalls', () => {
  const enabled = ['fs_read', 'fs_patch', 'fs_list', 'shell_run', 'spawn_subagent']

  it('extracts Qwen/Hermes <tool_call>{...}</tool_call> with alias mapping', () => {
    const input =
      'Let me read the file.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "src/main.rs"}}\n</tool_call>'
    const { cleanedText, toolCalls } = extractTextToolCalls(input, enabled)
    expect(cleanedText).toBe('Let me read the file.')
    expect(toolCalls).toEqual([
      { id: 'call_text_1', name: 'fs_read', arguments: { path: 'src/main.rs' } }
    ])
  })

  it('extracts Qwen3-Coder <function=name><parameter=k>v</parameter></function> both wrapped and bare', () => {
    const wrapped = [
      'Applying edit.',
      '<tool_call>',
      '<function=edit_file>',
      '<parameter=path>',
      'src/app.ts',
      '</parameter>',
      '<parameter=old_string>',
      'const x = 1',
      '</parameter>',
      '<parameter=new_string>',
      'const x = 2',
      '</parameter>',
      '</function>',
      '</tool_call>'
    ].join('\n')

    const res1 = extractTextToolCalls(wrapped, enabled)
    expect(res1.cleanedText).toBe('Applying edit.')
    expect(res1.toolCalls).toEqual([
      {
        id: 'call_text_1',
        name: 'fs_patch',
        arguments: { path: 'src/app.ts', old_string: 'const x = 1', new_string: 'const x = 2' }
      }
    ])

    const bare = '<function=spawn_subagent><parameter=prompts>["task a","task b"]</parameter></function>'
    const res2 = extractTextToolCalls(bare, enabled)
    expect(res2.cleanedText).toBe('')
    expect(res2.toolCalls).toEqual([
      {
        id: 'call_text_1',
        name: 'spawn_subagent',
        arguments: { prompts: ['task a', 'task b'] }
      }
    ])
  })

  it('extracts Harmony <|channel|>commentary to=functions.NAME ...<|message|>{...}', () => {
    const input =
      '<|channel|>commentary to=functions.fs_read <|constrain|>json<|message|>{"path":"Cargo.toml"}<|call|>'
    const { cleanedText, toolCalls } = extractTextToolCalls(input, enabled)
    expect(cleanedText).toBe('')
    expect(toolCalls).toEqual([
      { id: 'call_text_1', name: 'fs_read', arguments: { path: 'Cargo.toml' } }
    ])
  })

  it('extracts Mistral/Devstral [TOOL_CALLS] [...] including nested arrays in arguments', () => {
    const input =
      'Delegating tasks.\n[TOOL_CALLS] [{"name": "spawn_subagent", "arguments": {"prompts": ["one", "two"]}}]'
    const { cleanedText, toolCalls } = extractTextToolCalls(input, enabled)
    expect(cleanedText).toBe('Delegating tasks.')
    expect(toolCalls).toEqual([
      { id: 'call_text_1', name: 'spawn_subagent', arguments: { prompts: ['one', 'two'] } }
    ])
  })

  it('extracts standalone fenced json only when name resolves to an enabled tool', () => {
    const validFence = '```json\n{"name": "bash", "arguments": {"command": "pwd"}}\n```'
    const resValid = extractTextToolCalls(validFence, enabled)
    expect(resValid.cleanedText).toBe('')
    expect(resValid.toolCalls).toEqual([
      { id: 'call_text_1', name: 'shell_run', arguments: { command: 'pwd' } }
    ])

    // Ordinary package.json prose must NOT be hijacked as a tool call.
    const packageJsonFence = '```json\n{"name": "@brazier/desktop", "version": "0.3.1"}\n```'
    const resIgnored = extractTextToolCalls(packageJsonFence, enabled)
    expect(resIgnored.toolCalls).toEqual([])
    expect(resIgnored.cleanedText).toBe(packageJsonFence)
  })
})

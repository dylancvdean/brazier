import { describe, expect, it } from 'vitest'

import { inferModelCapabilities, parseLooseJson, repairToolArguments } from './modelCompat'
import type { AgentToolDefinition } from './types'

const fsWrite: Pick<AgentToolDefinition, 'inputSchema'> = {
  inputSchema: {
    type: 'object',
    properties: {
      path: { type: 'string' },
      content: { type: 'string' }
    },
    required: ['path', 'content'],
    additionalProperties: false
  }
}

const shellRun: Pick<AgentToolDefinition, 'inputSchema'> = {
  inputSchema: {
    type: 'object',
    properties: {
      command: { type: 'string' },
      timeout_ms: { type: 'integer' },
      network: { type: 'boolean' }
    },
    required: ['command'],
    additionalProperties: false
  }
}

const fsRead: Pick<AgentToolDefinition, 'inputSchema'> = {
  inputSchema: {
    type: 'object',
    properties: { path: { type: 'string' } },
    required: ['path'],
    additionalProperties: false
  }
}

describe('repairToolArguments', () => {
  it('passes well-formed arguments through unchanged', () => {
    const args = { path: 'src/main.rs', content: 'fn main() {}' }
    expect(repairToolArguments(args, fsWrite)).toEqual(args)
  })

  it('parses arguments delivered as a JSON string', () => {
    const raw = '{"path":"a.txt","content":"hi"}'
    expect(repairToolArguments(raw, fsWrite)).toEqual({ path: 'a.txt', content: 'hi' })
  })

  it('parses double-encoded JSON', () => {
    const raw = JSON.stringify(JSON.stringify({ path: 'a.txt', content: 'hi' }))
    expect(repairToolArguments(raw, fsWrite)).toEqual({ path: 'a.txt', content: 'hi' })
  })

  it('strips Markdown fences a model wrapped around the JSON', () => {
    const raw = '```json\n{"command":"cargo test"}\n```'
    expect(repairToolArguments(raw, shellRun)).toEqual({ command: 'cargo test' })
  })

  it('tolerates trailing prose after the object', () => {
    const raw = '{"command":"ls"} — this lists the directory'
    expect(repairToolArguments(raw, shellRun)).toEqual({ command: 'ls' })
  })

  it('unwraps a wrapper object around the real arguments', () => {
    expect(repairToolArguments({ input: { command: 'ls' } }, shellRun)).toEqual({ command: 'ls' })
    expect(repairToolArguments({ parameters: { command: 'ls' } }, shellRun)).toEqual({
      command: 'ls'
    })
    expect(repairToolArguments({ arguments: '{"command":"ls"}' }, shellRun)).toEqual({
      command: 'ls'
    })
  })

  it('maps common wrong parameter names onto the real ones', () => {
    expect(repairToolArguments({ file_path: 'a.txt', contents: 'hi' }, fsWrite)).toEqual({
      path: 'a.txt',
      content: 'hi'
    })
    expect(repairToolArguments({ cmd: 'cargo build' }, shellRun)).toEqual({
      command: 'cargo build'
    })
  })

  it('prefers the declared name over an alias when both are present', () => {
    expect(repairToolArguments({ path: 'real.txt', file_path: 'alias.txt' }, fsRead)).toEqual({
      path: 'real.txt'
    })
  })

  it('coerces primitives that arrived as strings', () => {
    expect(repairToolArguments({ command: 'ls', timeout_ms: '5000', network: 'true' }, shellRun)).toEqual(
      { command: 'ls', timeout_ms: 5000, network: true }
    )
  })

  it('drops unknown keys when the schema forbids extras', () => {
    expect(
      repairToolArguments({ command: 'ls', hallucinated: 'nonsense' }, shellRun)
    ).toEqual({ command: 'ls' })
  })

  it('keeps unknown keys when the schema allows extras', () => {
    const permissive: Pick<AgentToolDefinition, 'inputSchema'> = {
      inputSchema: {
        type: 'object',
        properties: { a: { type: 'string' } },
        required: [],
        additionalProperties: true
      }
    }
    expect(repairToolArguments({ a: 'x', extra: 1 }, permissive)).toEqual({ a: 'x', extra: 1 })
  })

  it('places a bare string into the single required string argument', () => {
    expect(repairToolArguments('src/main.rs', fsRead)).toEqual({ path: 'src/main.rs' })
  })

  it('does not guess when the target is ambiguous', () => {
    // Two required arguments: a bare string has no unambiguous home, so the
    // daemon should receive nothing and report the validation failure.
    expect(repairToolArguments('mystery', fsWrite)).toEqual({})
  })

  it('never invents a required argument', () => {
    expect(repairToolArguments({ content: 'hi' }, fsWrite)).toEqual({ content: 'hi' })
  })

  it('normalizes enum casing', () => {
    const withEnum: Pick<AgentToolDefinition, 'inputSchema'> = {
      inputSchema: {
        type: 'object',
        properties: { mode: { type: 'string', enum: ['ask', 'sandbox-only'] } },
        required: ['mode'],
        additionalProperties: false
      }
    }
    expect(repairToolArguments({ mode: 'Ask' }, withEnum)).toEqual({ mode: 'ask' })
  })

  it('wraps a lone value where a list was declared', () => {
    const withList: Pick<AgentToolDefinition, 'inputSchema'> = {
      inputSchema: {
        type: 'object',
        properties: { paths: { type: 'array' } },
        required: [],
        additionalProperties: false
      }
    }
    expect(repairToolArguments({ paths: '/etc/hosts' }, withList)).toEqual({
      paths: ['/etc/hosts']
    })
  })

  it('returns an empty object for unusable input', () => {
    expect(repairToolArguments(null, shellRun)).toEqual({})
    expect(repairToolArguments(42, shellRun)).toEqual({})
  })
})

describe('parseLooseJson', () => {
  it('returns undefined when there is no JSON to find', () => {
    expect(parseLooseJson('not json at all')).toBeUndefined()
  })
})

describe('inferModelCapabilities', () => {
  it('limits weak models to one tool call per turn', () => {
    const capabilities = inferModelCapabilities('gguf:acme/tinyllama-1b/model.gguf')
    expect(capabilities.maxToolsPerTurn).toBe(1)
    expect(capabilities.parallelToolCalling).toBe(false)
    expect(capabilities.reliableJson).toBe(false)
  })

  it('lets known-strong families call tools in parallel', () => {
    const capabilities = inferModelCapabilities('mlx:openai/gpt-oss-20b')
    expect(capabilities.parallelToolCalling).toBe(true)
    expect(capabilities.maxToolsPerTurn).toBeUndefined()
    expect(capabilities.supportsReasoningStream).toBe(true)
  })

  it('always assumes native tool calling, since the daemon validates anyway', () => {
    expect(inferModelCapabilities('anything').nativeToolCalling).toBe(true)
  })
})

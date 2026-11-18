import { describe, expect, it } from 'vitest'

import {
  inferModelCapabilities,
  parseLooseJson,
  repairToolArguments,
  resolveToolNameAlias,
  validateToolArguments
} from './modelCompat'
import type { AgentToolDefinition } from './types'

const fsWrite: Pick<AgentToolDefinition, 'name' | 'inputSchema'> = {
  name: 'fs_write',
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

const shellRun: Pick<AgentToolDefinition, 'name' | 'inputSchema'> = {
  name: 'shell_run',
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

const fsRead: Pick<AgentToolDefinition, 'name' | 'inputSchema'> = {
  name: 'fs_read',
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
    expect(
      repairToolArguments({ command: 'ls', timeout_ms: '5000', network: 'true' }, shellRun)
    ).toEqual({ command: 'ls', timeout_ms: 5000, network: true })
  })

  it('drops unknown keys when the schema forbids extras', () => {
    expect(repairToolArguments({ command: 'ls', hallucinated: 'nonsense' }, shellRun)).toEqual({
      command: 'ls'
    })
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
    expect(capabilities.harmony).toBe(true)
  })

  it('recognizes expanded local and remote strong families', () => {
    for (const modelId of [
      'gguf:google/gemma-4-27b-it.gguf',
      'mlx:google/gemma4-12b',
      'gguf:acme/bonsai-8b.gguf',
      'gguf:meta/llama-3.3-70b.gguf',
      'gguf:qwen/qwq-32b.gguf',
      'gguf:mistral/ministral-8b.gguf',
      'remote:anthropic/claude-sonnet-4',
      'remote:openai/gpt-5',
      'remote:openai/o3-mini',
      'remote:google/gemini-2.5-pro',
      'remote:xai/grok-3'
    ]) {
      const caps = inferModelCapabilities(modelId)
      expect(caps.parallelToolCalling, modelId).toBe(true)
      expect(caps.supportsReasoningStream, modelId).toBe(true)
      expect(caps.maxToolsPerTurn, modelId).toBeUndefined()
    }
  })

  it('always assumes native tool calling, since the daemon validates anyway', () => {
    expect(inferModelCapabilities('anything').nativeToolCalling).toBe(true)
  })
})

describe('resolveToolNameAlias', () => {
  const enabled = [
    'fs_patch',
    'fs_read',
    'fs_write',
    'fs_list',
    'shell_run',
    'fs_search',
    'fs_find'
  ]

  it('maps common hallucinated tool names onto canonical enabled tools', () => {
    expect(resolveToolNameAlias('edit_file', enabled)).toBe('fs_patch')
    expect(resolveToolNameAlias('fs_edit', enabled)).toBe('fs_patch')
    expect(resolveToolNameAlias('str_replace_editor', enabled)).toBe('fs_patch')
    expect(resolveToolNameAlias('replace_in_file', enabled)).toBe('fs_patch')
    expect(resolveToolNameAlias('read_file', enabled)).toBe('fs_read')
    expect(resolveToolNameAlias('view_file', enabled)).toBe('fs_read')
    expect(resolveToolNameAlias('write_file', enabled)).toBe('fs_write')
    expect(resolveToolNameAlias('create_file', enabled)).toBe('fs_write')
    expect(resolveToolNameAlias('list_dir', enabled)).toBe('fs_list')
    expect(resolveToolNameAlias('ls', enabled)).toBe('fs_list')
    expect(resolveToolNameAlias('bash', enabled)).toBe('shell_run')
    expect(resolveToolNameAlias('sh', enabled)).toBe('shell_run')
    expect(resolveToolNameAlias('run_command', enabled)).toBe('shell_run')
    expect(resolveToolNameAlias('execute_command', enabled)).toBe('shell_run')
    expect(resolveToolNameAlias('grep', enabled)).toBe('fs_search')
    expect(resolveToolNameAlias('ripgrep', enabled)).toBe('fs_search')
    expect(resolveToolNameAlias('search_files', enabled)).toBe('fs_search')
    expect(resolveToolNameAlias('find_files', enabled)).toBe('fs_find')
    expect(resolveToolNameAlias('glob', enabled)).toBe('fs_find')
  })

  it('strips Harmony functions.* prefix when target is enabled', () => {
    expect(resolveToolNameAlias('functions.fs_read', enabled)).toBe('fs_read')
    expect(resolveToolNameAlias('functions.edit_file', enabled)).toBe('fs_patch')
  })

  it('never rewrites a tool name that is already directly enabled', () => {
    expect(resolveToolNameAlias('bash', ['bash', 'shell_run'])).toBe('bash')
    expect(resolveToolNameAlias('fs_read', enabled)).toBe('fs_read')
  })

  it('leaves rawName untouched when canonical target is not enabled', () => {
    expect(resolveToolNameAlias('edit_file', ['fs_read'])).toBe('edit_file')
    expect(resolveToolNameAlias('functions.shell_run', ['fs_read'])).toBe('functions.shell_run')
  })
})

describe('validateToolArguments', () => {
  it('returns null for valid repaired arguments', () => {
    expect(validateToolArguments({ path: 'a.txt', content: 'ok' }, fsWrite)).toBeNull()
    expect(
      validateToolArguments({ command: 'ls', timeout_ms: 1000, network: false }, shellRun)
    ).toBeNull()
  })

  it('reports missing required fields clearly', () => {
    const err = validateToolArguments({ path: 'a.txt' }, fsWrite)
    expect(err).toContain('Missing required argument `content`')
    expect(err).toContain('fs_write')
  })

  it('checks integer, number, boolean, array, and object types', () => {
    const complexTool: Pick<AgentToolDefinition, 'name' | 'inputSchema'> = {
      name: 'complex',
      inputSchema: {
        type: 'object',
        properties: {
          count: { type: 'integer' },
          ratio: { type: 'number' },
          flag: { type: 'boolean' },
          items: { type: 'array' },
          meta: { type: 'object' }
        },
        required: []
      }
    }
    expect(validateToolArguments({ count: 1.5 }, complexTool)).toContain('expected integer')
    expect(validateToolArguments({ ratio: 'nope' }, complexTool)).toContain('expected number')
    expect(validateToolArguments({ flag: 'maybe' }, complexTool)).toContain('expected boolean')
    expect(validateToolArguments({ items: null }, complexTool)).toContain('expected array')
    expect(validateToolArguments({ meta: ['not-obj'] }, complexTool)).toContain('expected object')
    expect(
      validateToolArguments(
        { count: 3, ratio: 0.5, flag: true, items: [1], meta: { ok: true } },
        complexTool
      )
    ).toBeNull()
  })

  it('rejects out-of-range enum values', () => {
    const enumTool: Pick<AgentToolDefinition, 'name' | 'inputSchema'> = {
      name: 'mode_tool',
      inputSchema: {
        type: 'object',
        properties: {
          mode: { type: 'string', enum: ['ask', 'sandbox-only'] }
        },
        required: ['mode']
      }
    }
    const err = validateToolArguments({ mode: 'yolo' }, enumTool)
    expect(err).toContain('expected one of ["ask", "sandbox-only"]')
  })
})

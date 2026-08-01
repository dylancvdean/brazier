import { chmodSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { OmpRpcClient, OmpRpcFrameDecoder } from './rpcClient'

let directory: string | undefined

function rpcFixture(ignoreTerminate = false): string {
  directory = mkdtempSync(join(tmpdir(), 'brazier-omp-rpc-'))
  const binary = join(directory, 'omp-fixture.mjs')
  writeFileSync(
    binary,
    `#!/usr/bin/env node
${ignoreTerminate ? "process.on('SIGTERM', () => undefined)" : ''}
process.stdout.write(JSON.stringify({ type: 'ready' }) + '\\n')
let buffered = ''
process.stdin.on('data', (chunk) => {
  buffered += chunk
  while (true) {
    const newline = buffered.indexOf('\\n')
    if (newline === -1) break
    const line = buffered.slice(0, newline)
    buffered = buffered.slice(newline + 1)
    if (!line) continue
    const request = JSON.parse(line)
    process.stdout.write(JSON.stringify({
      type: 'response', id: request.id, success: true, command: request.type, received: request
    }) + '\\n')
  }
})
`
  )
  chmodSync(binary, 0o755)
  return binary
}

afterEach(() => {
  if (directory) rmSync(directory, { recursive: true, force: true })
  directory = undefined
  vi.useRealTimers()
})

describe.skipIf(process.platform === 'win32')('OmpRpcClient', () => {
  it('reassembles validated protocol-v2 chunk frames losslessly', () => {
    const logical = { type: 'message_end', text: 'x'.repeat(1_100_000) }
    const bytes = Buffer.from(JSON.stringify(logical), 'utf8')
    const decoder = new OmpRpcFrameDecoder()
    const chunkSize = 256 * 1024
    const count = Math.ceil(bytes.length / chunkSize)
    let completed: unknown
    for (let index = 0; index < count; index++) {
      completed = decoder.push({
        type: 'rpc_chunk',
        chunkId: 'large-message',
        index,
        count,
        byteLength: bytes.length,
        data: bytes.subarray(index * chunkSize, (index + 1) * chunkSize).toString('base64')
      })
      if (index < count - 1) expect(completed).toBeUndefined()
    }

    expect(completed).toEqual(logical)
  })

  it('rejects a non-chunk frame that interrupts reassembly', () => {
    const decoder = new OmpRpcFrameDecoder()
    const bytes = Buffer.alloc(1024 * 1024, 0)
    decoder.push({
      type: 'rpc_chunk',
      chunkId: 'interrupted',
      index: 0,
      count: 2,
      byteLength: bytes.length,
      data: bytes.subarray(0, 256 * 1024).toString('base64')
    })

    expect(() => decoder.push({ type: 'response', success: true })).toThrow('interrupted')
  })

  it('sends commands as NDJSON frames and correlates the response id', async () => {
    const client = new OmpRpcClient({ binary: rpcFixture() })

    await expect(client.request({ type: 'ping', payload: { value: 1 } })).resolves.toMatchObject({
      type: 'response',
      command: 'ping',
      received: {
        type: 'ping',
        payload: { value: 1 },
        id: expect.stringMatching(/^brazier-\d+$/)
      }
    })

    await client.dispose()
  })

  it('escalates to SIGKILL when the sidecar ignores SIGTERM', async () => {
    const onExit = vi.fn()
    const client = new OmpRpcClient({ binary: rpcFixture(true), onExit })
    await client.waitUntilReady()

    await client.dispose()

    expect(onExit).toHaveBeenCalledWith(null, 'SIGKILL')
  }, 5_000)

  it('includes bounded sidecar stderr in startup failures', async () => {
    const binary = rpcFixture()
    // Replace the ready fixture with a process that emits the diagnostic and
    // exits before the RPC ready frame.
    writeFileSync(
      binary,
      `#!/usr/bin/env node\nprocess.stderr.write('Error: Nonexistent flag --bad-option\\n')\nprocess.exit(2)\n`
    )
    chmodSync(binary, 0o755)
    const client = new OmpRpcClient({ binary })

    await expect(client.waitUntilReady()).rejects.toThrow(
      'omp exited with code 2.\nError: Nonexistent flag --bad-option'
    )
  })
})

import { afterEach, describe, expect, it, vi } from 'vitest'
import { ObjectUrlCache } from './objectUrlCache'

afterEach(() => vi.restoreAllMocks())

describe('attachment memory ownership', () => {
  it('shares concurrent downloads and revokes only after the last view releases', async () => {
    const revoke = vi.spyOn(URL, 'revokeObjectURL')
    const cache = new ObjectUrlCache()
    const load = vi.fn(async () => new Blob(['image']))
    const first = cache.acquire('local/hash', load)
    const second = cache.acquire('local/hash', load)
    const url = await first.url
    expect(await second.url).toBe(url)
    expect(load).toHaveBeenCalledOnce()
    first.release()
    first.release()
    await Promise.resolve()
    expect(revoke).not.toHaveBeenCalled()
    second.release()
    await Promise.resolve()
    expect(revoke).toHaveBeenCalledExactlyOnceWith(url)
  })

  it('does not revoke assets reacquired during effect cleanup/setup', async () => {
    const revoke = vi.spyOn(URL, 'revokeObjectURL')
    const cache = new ObjectUrlCache()
    const load = vi.fn(async () => new Blob(['image']))
    const first = cache.acquire('local/hash', load)
    const url = await first.url
    first.release()
    const next = cache.acquire('local/hash', load)
    expect(await next.url).toBe(url)
    expect(revoke).not.toHaveBeenCalled()
    next.release()
    await Promise.resolve()
    expect(revoke).toHaveBeenCalledExactlyOnceWith(url)
  })

  it('aborts unused downloads and never creates a URL from their late results', async () => {
    const create = vi.spyOn(URL, 'createObjectURL')
    let resolve!: (blob: Blob) => void
    let signal!: AbortSignal
    const cache = new ObjectUrlCache()
    const lease = cache.acquire('local/hash', (abort) => {
      signal = abort
      return new Promise<Blob>((done) => { resolve = done })
    })
    const rejected = expect(lease.url).rejects.toMatchObject({ name: 'AbortError' })
    await Promise.resolve()
    lease.release()
    await Promise.resolve()
    expect(signal.aborted).toBe(true)
    resolve(new Blob(['late']))
    await rejected
    expect(create).not.toHaveBeenCalled()
  })

  it('isolates profiles and retries failed downloads after release', async () => {
    const cache = new ObjectUrlCache()
    const failed = cache.acquire('old/hash', async () => { throw new Error('missing') })
    const other = cache.acquire('new/hash', async () => new Blob(['available']))
    await expect(failed.url).rejects.toThrow('missing')
    await expect(other.url).resolves.toMatch(/^blob:/)
    failed.release()
    other.release()
    await Promise.resolve()
    const retry = cache.acquire('old/hash', async () => new Blob(['recovered']))
    await expect(retry.url).resolves.toMatch(/^blob:/)
    retry.release()
  })
})

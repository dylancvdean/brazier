type Entry = {
  references: number
  abort: AbortController
  promise: Promise<string>
  url?: string
}

/** Share downloads while displayed, and release their memory when the last view leaves. */
export class ObjectUrlCache {
  private entries = new Map<string, Entry>()

  acquire(key: string, load: (signal: AbortSignal) => Promise<Blob>): {
    url: Promise<string>
    release: () => void
  } {
    let entry = this.entries.get(key)
    if (!entry) {
      const abort = new AbortController()
      const created: Entry = {
        references: 0,
        abort,
        promise: Promise.resolve().then(async () => {
          abort.signal.throwIfAborted()
          const blob = await load(abort.signal)
          abort.signal.throwIfAborted()
          created.url = URL.createObjectURL(blob)
          return created.url
        })
      }
      entry = created
      this.entries.set(key, entry)
    }
    entry.references++
    let released = false
    return {
      url: entry.promise,
      release: () => {
        if (released) return
        released = true
        entry.references--
        // Effect cleanup/setup can reacquire the same asset in this turn.
        queueMicrotask(() => {
          if (entry.references || this.entries.get(key) !== entry) return
          this.entries.delete(key)
          entry.abort.abort()
          if (entry.url) URL.revokeObjectURL(entry.url)
        })
      }
    }
  }
}

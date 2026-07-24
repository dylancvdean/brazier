/**
 * Minimal Ogg encapsulation for Opus, in both directions.
 *
 * PersonaPlex/Moshi servers exchange Ogg-encapsulated Opus over the chat
 * WebSocket (`sphn::OpusStreamReader` / `OpusStreamWriter` on the Rust side).
 * WebCodecs gives us bare Opus packets, so this module is the missing layer:
 * [`OggOpusMuxer`] wraps outgoing packets into pages and [`OggOpusDemuxer`]
 * recovers packets from the incoming page stream.
 *
 * Only what a live mono voice stream needs is implemented: no seeking, no
 * chained logical bitstreams, no multi-stream channel mapping families.
 */

/** Opus decodes at 48 kHz internally, so granule positions always count 48 kHz samples. */
const GRANULE_RATE = 48000
/** Encoder delay reported in `OpusHead`; a small overestimate only trims a few ms. */
const DEFAULT_PRE_SKIP = 312
const OGG_MAGIC = 0x4f676753 // "OggS"
const HEADER_BYTES = 27

const CRC_TABLE = (() => {
  const table = new Uint32Array(256)
  for (let index = 0; index < 256; index += 1) {
    let value = index << 24
    for (let bit = 0; bit < 8; bit += 1) {
      value = value & 0x80000000 ? ((value << 1) ^ 0x04c11db7) >>> 0 : (value << 1) >>> 0
    }
    table[index] = value >>> 0
  }
  return table
})()

/** Ogg's CRC-32 variant: no input/output reflection, zero init, no final xor. */
export function oggCrc32(bytes: Uint8Array): number {
  let crc = 0
  for (let index = 0; index < bytes.length; index += 1) {
    crc = ((crc << 8) ^ CRC_TABLE[((crc >>> 24) ^ bytes[index]) & 0xff]) >>> 0
  }
  return crc >>> 0
}

function lacing(packetLengths: number[]): number[] {
  const segments: number[] = []
  for (const length of packetLengths) {
    let remaining = length
    while (remaining >= 255) {
      segments.push(255)
      remaining -= 255
    }
    segments.push(remaining)
  }
  return segments
}

export type OpusHeadInfo = {
  channelCount: number
  preSkip: number
  inputSampleRate: number
}

/** Parse the 19-byte `OpusHead` identification packet. */
export function parseOpusHead(packet: Uint8Array): OpusHeadInfo | null {
  if (packet.length < 19 || !isMagic(packet, 'OpusHead')) return null
  const view = new DataView(packet.buffer, packet.byteOffset, packet.byteLength)
  return {
    channelCount: packet[9],
    preSkip: view.getUint16(10, true),
    inputSampleRate: view.getUint32(12, true)
  }
}

function isMagic(packet: Uint8Array, magic: string): boolean {
  if (packet.length < magic.length) return false
  for (let index = 0; index < magic.length; index += 1) {
    if (packet[index] !== magic.charCodeAt(index)) return false
  }
  return true
}

export function buildOpusHead(
  channelCount: number,
  sampleRate: number,
  preSkip: number
): Uint8Array<ArrayBuffer> {
  const packet = new Uint8Array(19)
  packet.set(new TextEncoder().encode('OpusHead'), 0)
  const view = new DataView(packet.buffer)
  packet[8] = 1 // version
  packet[9] = channelCount
  view.setUint16(10, preSkip, true)
  view.setUint32(12, sampleRate, true)
  view.setUint16(16, 0, true) // output gain
  packet[18] = 0 // channel mapping family
  return packet
}

function buildOpusTags(vendor: string): Uint8Array<ArrayBuffer> {
  const encoded = new TextEncoder().encode(vendor)
  const packet = new Uint8Array(8 + 4 + encoded.length + 4)
  packet.set(new TextEncoder().encode('OpusTags'), 0)
  const view = new DataView(packet.buffer)
  view.setUint32(8, encoded.length, true)
  packet.set(encoded, 12)
  view.setUint32(12 + encoded.length, 0, true) // zero user comments
  return packet
}

/**
 * Packs Opus packets into Ogg pages for a single logical bitstream.
 *
 * Audio packets are batched `framesPerPage` at a time, matching the cadence
 * the reference Moshi web client uses (20 ms frames, two frames per page), so
 * the server sees the packet rate it is tuned for.
 */
export class OggOpusMuxer {
  private readonly serial: number
  private readonly channelCount: number
  private readonly sampleRate: number
  private readonly framesPerPage: number
  private sequence = 0
  private granule = 0
  private pending: Uint8Array[] = []
  private pendingSamples = 0

  constructor(options: {
    sampleRate: number
    channelCount?: number
    framesPerPage?: number
    serial?: number
  }) {
    this.sampleRate = options.sampleRate
    this.channelCount = options.channelCount ?? 1
    this.framesPerPage = Math.max(1, options.framesPerPage ?? 2)
    this.serial = options.serial ?? (Math.floor(Math.random() * 0xffffffff) >>> 0)
  }

  /** The `OpusHead` and `OpusTags` pages that must precede any audio. */
  headerPages(): Uint8Array<ArrayBuffer> {
    const head = this.page([buildOpusHead(this.channelCount, this.sampleRate, DEFAULT_PRE_SKIP)], 0, {
      first: true
    })
    const tags = this.page([buildOpusTags('brazier')], 0)
    return concat([head, tags])
  }

  /**
   * Queue one encoded packet, returning a complete page once enough frames
   * have accumulated (`null` until then).
   *
   * `durationSamples` is the packet's decoded length in `sampleRate` samples.
   */
  addPacket(packet: Uint8Array, durationSamples: number): Uint8Array<ArrayBuffer> | null {
    this.pending.push(packet)
    this.pendingSamples += durationSamples
    if (this.pending.length < this.framesPerPage) return null
    return this.flush()
  }

  /** Emit whatever is queued as one page, or `null` when nothing is queued. */
  flush(options: { last?: boolean } = {}): Uint8Array<ArrayBuffer> | null {
    if (this.pending.length === 0) return null
    this.granule += Math.round((this.pendingSamples * GRANULE_RATE) / this.sampleRate)
    const page = this.page(this.pending, this.granule, { last: options.last })
    this.pending = []
    this.pendingSamples = 0
    return page
  }

  private page(
    packets: Uint8Array[],
    granule: number,
    flags: { first?: boolean; last?: boolean } = {}
  ): Uint8Array<ArrayBuffer> {
    const segments = lacing(packets.map((packet) => packet.length))
    if (segments.length > 255) {
      throw new Error('Ogg page overflow: too many segments for one page')
    }
    const body = concat(packets)
    const page = new Uint8Array(HEADER_BYTES + segments.length + body.length)
    const view = new DataView(page.buffer)
    view.setUint32(0, OGG_MAGIC, false)
    page[4] = 0 // stream structure version
    page[5] = (flags.first ? 0x02 : 0) | (flags.last ? 0x04 : 0)
    // Granule positions are 64-bit; a voice session never reaches 2^32 samples,
    // but write both halves so long sessions stay well-formed.
    view.setUint32(6, granule >>> 0, true)
    view.setUint32(10, Math.floor(granule / 0x100000000), true)
    view.setUint32(14, this.serial, true)
    view.setUint32(18, this.sequence, true)
    view.setUint32(22, 0, true) // checksum placeholder
    page[26] = segments.length
    page.set(segments, HEADER_BYTES)
    page.set(body, HEADER_BYTES + segments.length)
    view.setUint32(22, oggCrc32(page), true)
    this.sequence += 1
    return page
  }
}

/**
 * Incremental Ogg page reader: feed it arbitrary byte chunks, get back whole
 * Opus packets. Packets split across pages are rejoined, and the `OpusHead`
 * identification packet is surfaced separately for decoder configuration.
 */
export class OggOpusDemuxer {
  private buffer: Uint8Array<ArrayBuffer> = new Uint8Array(0)
  private continuation: Uint8Array<ArrayBuffer>[] = []
  private head: Uint8Array<ArrayBuffer> | null = null

  /** `OpusHead` bytes once seen, for use as an `AudioDecoder` description. */
  get opusHead(): Uint8Array<ArrayBuffer> | null {
    return this.head
  }

  /** Append received bytes and drain every audio packet they complete. */
  push(chunk: Uint8Array): Uint8Array<ArrayBuffer>[] {
    this.buffer = concat([this.buffer, chunk])
    const packets: Uint8Array<ArrayBuffer>[] = []
    let offset = 0
    while (true) {
      const page = this.readPage(offset)
      if (!page) break
      offset = page.end
      for (const packet of page.packets) {
        if (isMagic(packet, 'OpusHead')) {
          this.head = packet
        } else if (!isMagic(packet, 'OpusTags')) {
          packets.push(packet)
        }
      }
    }
    if (offset > 0) this.buffer = this.buffer.slice(offset)
    return packets
  }

  /**
   * Parse the page starting at `offset`, or return `null` when the buffer
   * holds no complete page there (resyncing past leading garbage if needed).
   */
  private readPage(offset: number): { end: number; packets: Uint8Array<ArrayBuffer>[] } | null {
    let start = offset
    while (start + HEADER_BYTES <= this.buffer.length) {
      if (
        this.buffer[start] === 0x4f &&
        this.buffer[start + 1] === 0x67 &&
        this.buffer[start + 2] === 0x67 &&
        this.buffer[start + 3] === 0x53
      ) {
        break
      }
      start += 1
    }
    if (start + HEADER_BYTES > this.buffer.length) return null

    const segmentCount = this.buffer[start + 26]
    const tableEnd = start + HEADER_BYTES + segmentCount
    if (tableEnd > this.buffer.length) return null
    const segments = this.buffer.subarray(start + HEADER_BYTES, tableEnd)
    let bodyLength = 0
    for (const segment of segments) bodyLength += segment
    const end = tableEnd + bodyLength
    if (end > this.buffer.length) return null

    const continued = (this.buffer[start + 5] & 0x01) !== 0
    if (!continued) this.continuation = []
    const packets: Uint8Array<ArrayBuffer>[] = []
    let cursor = tableEnd
    let current: Uint8Array<ArrayBuffer>[] = this.continuation
    for (const segment of segments) {
      current.push(this.buffer.slice(cursor, cursor + segment))
      cursor += segment
      if (segment < 255) {
        packets.push(concat(current))
        current = []
      }
    }
    // A trailing run of 255s means the last packet spills into the next page.
    this.continuation = current
    return { end, packets }
  }
}

function concat(chunks: Uint8Array[]): Uint8Array<ArrayBuffer> {
  let length = 0
  for (const chunk of chunks) length += chunk.length
  const out = new Uint8Array(length)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.length
  }
  return out
}

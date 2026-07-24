import { describe, expect, it } from 'vitest'
import { OggOpusDemuxer, OggOpusMuxer, oggCrc32, parseOpusHead } from './oggOpus'

function packet(length: number, seed: number): Uint8Array {
  const bytes = new Uint8Array(length)
  for (let index = 0; index < length; index += 1) bytes[index] = (index * 31 + seed) & 0xff
  return bytes
}

function mux(packets: Uint8Array[], framesPerPage = 2): Uint8Array {
  const muxer = new OggOpusMuxer({ sampleRate: 24000, framesPerPage })
  const pages: Uint8Array[] = [muxer.headerPages()]
  for (const entry of packets) {
    const page = muxer.addPacket(entry, 480)
    if (page) pages.push(page)
  }
  const tail = muxer.flush({ last: true })
  if (tail) pages.push(tail)
  const total = pages.reduce((sum, page) => sum + page.length, 0)
  const merged = new Uint8Array(total)
  let offset = 0
  for (const page of pages) {
    merged.set(page, offset)
    offset += page.length
  }
  return merged
}

describe('ogg opus framing', () => {
  it('matches the reference CRC-32 for the standard check vector', () => {
    // Ogg's CRC-32 is CRC-32/CKSUM without the final xor, so the usual
    // "123456789" vector lands on 0x765e7680 ^ 0xffffffff.
    expect(oggCrc32(new TextEncoder().encode('123456789'))).toBe(0x89a1897f)
  })

  it('round-trips packets whose lengths straddle the 255-byte lacing boundary', () => {
    const packets = [1, 254, 255, 256, 600, 40].map((length, index) => packet(length, index))
    const demuxer = new OggOpusDemuxer()
    const decoded = demuxer.push(mux(packets))
    expect(decoded.map((entry) => entry.length)).toEqual(packets.map((entry) => entry.length))
    decoded.forEach((entry, index) => expect(Array.from(entry)).toEqual(Array.from(packets[index])))
  })

  it('reassembles a stream delivered in arbitrary chunk sizes', () => {
    const packets = Array.from({ length: 25 }, (_, index) => packet(120 + index * 7, index))
    const stream = mux(packets)
    const demuxer = new OggOpusDemuxer()
    const decoded: Uint8Array[] = []
    for (let offset = 0; offset < stream.length; offset += 13) {
      decoded.push(...demuxer.push(stream.subarray(offset, offset + 13)))
    }
    expect(decoded.length).toBe(packets.length)
    expect(Array.from(decoded[24])).toEqual(Array.from(packets[24]))
  })

  it('exposes OpusHead and drops the header packets from the audio stream', () => {
    const demuxer = new OggOpusDemuxer()
    const decoded = demuxer.push(mux([packet(60, 1)]))
    expect(decoded.length).toBe(1)
    expect(parseOpusHead(demuxer.opusHead!)).toEqual({
      channelCount: 1,
      preSkip: 312,
      inputSampleRate: 24000
    })
  })

  it('marks the first page as beginning-of-stream and numbers pages in order', () => {
    const stream = mux([packet(50, 0), packet(50, 1)])
    expect(stream[5] & 0x02).toBe(0x02)
    const view = new DataView(stream.buffer)
    expect(view.getUint32(18, true)).toBe(0)
  })
})

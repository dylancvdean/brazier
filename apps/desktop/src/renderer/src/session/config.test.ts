import { describe, expect, it } from 'vitest'

import { DEFAULT_INTEGRATION_CONFIG, resolveAsrEngine } from './config'

const both = { batch: true, streaming: true }
const whisperOnly = { batch: true, streaming: false }
const streamingOnly = { batch: false, streaming: true }
const neither = { batch: false, streaming: false }

describe('resolveAsrEngine', () => {
  it('prefers whisper when it is installed, since utterances arrive whole', () => {
    expect(resolveAsrEngine('auto', both)).toBeUndefined()
    expect(resolveAsrEngine('auto', whisperOnly)).toBeUndefined()
  })

  it('falls back to the Nemotron worker when that is what is installed', () => {
    // The case that made this necessary: engine built, no Whisper model.
    expect(resolveAsrEngine('auto', streamingOnly)).toBe('streaming-asr')
  })

  it('honours an explicit choice even against the capability report', () => {
    // Naming the engine makes the daemon's error the real one — "no Nemotron
    // snapshot" — rather than silently transcribing with something else.
    expect(resolveAsrEngine('streaming-asr', whisperOnly)).toBe('streaming-asr')
    expect(resolveAsrEngine('whisper.cpp', streamingOnly)).toBeUndefined()
  })

  it('takes the daemon default when nothing is installed', () => {
    expect(resolveAsrEngine('auto', neither)).toBeUndefined()
  })

  it('defaults to automatic', () => {
    expect(DEFAULT_INTEGRATION_CONFIG.asrPreference).toBe('auto')
    expect(DEFAULT_INTEGRATION_CONFIG.voiceBackgroundRouting).toBe('auto')
    expect(DEFAULT_INTEGRATION_CONFIG.shortSpeechBoost).toBe(true)
    expect(DEFAULT_INTEGRATION_CONFIG.personaplexPreHandoffMode).toBe('mute-on-route')
  })
})

import { describe, expect, it } from 'vitest'

import { selectVoice } from './speechRenderer'

function voice(name: string, uri: string, isDefault = false): SpeechSynthesisVoice {
  return {
    name,
    voiceURI: uri,
    lang: 'en-GB',
    localService: true,
    default: isDefault
  } as SpeechSynthesisVoice
}

const voices = [voice('Daniel', 'urn:daniel', true), voice('Serena', 'urn:serena')]

describe('selectVoice', () => {
  it('takes the chosen voice', () => {
    expect(selectVoice(voices, 'urn:serena')?.name).toBe('Serena')
  })

  /** Hosts have been seen to report the same voice under a different URI. */
  it('falls back to matching by name', () => {
    expect(selectVoice(voices, 'Serena')?.voiceURI).toBe('urn:serena')
  })

  /**
   * A voice that has been uninstalled must not cost the answer. Losing the
   * chosen voice is a disappointment; losing the sentence is a failure.
   */
  it('falls back to the host default when the choice is gone', () => {
    expect(selectVoice(voices, 'urn:removed')?.name).toBe('Daniel')
  })

  it('takes the host default when nothing was chosen', () => {
    expect(selectVoice(voices, undefined)?.name).toBe('Daniel')
  })

  it('has nothing to pick on a host with no voices', () => {
    expect(selectVoice([], 'urn:serena')).toBeNull()
  })
})

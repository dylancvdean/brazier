import { describe, expect, it } from 'vitest'

import { isEchoOfSpokenText, isTooThinToSubmit } from './echoGuard'

describe('isEchoOfSpokenText', () => {
  const spoken = 'One test failed: the oggOpus muxer test in the audio suite.'

  it('discards the assistant hearing its own answer', () => {
    expect(isEchoOfSpokenText('one test failed the oggopus muxer test', spoken)).toBe(true)
    // Transcripts arrive without punctuation and with inconsistent case.
    expect(isEchoOfSpokenText('ONE TEST FAILED THE OGGOPUS MUXER TEST!', spoken)).toBe(true)
  })

  it('lets the user speak, including about what was just said', () => {
    expect(isEchoOfSpokenText('Why did it fail?', spoken)).toBe(false)
    expect(isEchoOfSpokenText('Show me the oggOpus muxer test', spoken)).toBe(false)
    expect(
      isEchoOfSpokenText(
        'You said one test failed in the oggOpus muxer test, so please open that file and explain what the audio suite is checking there.',
        spoken
      )
    ).toBe(false)
  })

  it('lets an interruption through', () => {
    expect(isEchoOfSpokenText('Stop talking.', spoken)).toBe(false)
    expect(isEchoOfSpokenText('No, I meant the Vulkan backend.', spoken)).toBe(false)
  })

  it('is inert when nothing was spoken', () => {
    expect(isEchoOfSpokenText('anything at all', null)).toBe(false)
    expect(isEchoOfSpokenText('anything at all', '')).toBe(false)
    expect(isEchoOfSpokenText('', spoken)).toBe(false)
  })

  it('does not treat a repeated word as a whole echo', () => {
    // "test" appears twice in the answer; three times is not accounted for.
    expect(isEchoOfSpokenText('test test test test', spoken)).toBe(false)
  })
})

describe('isTooThinToSubmit', () => {
  it('refuses what noise transcribes to', () => {
    for (const text of ['', ' ', '.', 'a', 'uh', 'um', 'hmm', 'uh um', 'Mm-hmm', '...']) {
      expect(isTooThinToSubmit(text)).toBe(true)
    }
  })

  it('keeps short answers, which are whole turns', () => {
    // The reason length cannot be the test. Dropping these would make the
    // assistant ignore an answer to its own question.
    for (const text of ['yes', 'No.', 'stop', 'One.', 'Vulkan', 'the docs']) {
      expect(isTooThinToSubmit(text)).toBe(false)
    }
  })
})

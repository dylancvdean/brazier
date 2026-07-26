import { describe, expect, it } from 'vitest'

import { classifyConfirmation } from './confirmation'

describe('classifyConfirmation', () => {
  it('accepts a plain yes, however it is said', () => {
    for (const text of ['Yes', 'yeah', 'Yep.', 'go ahead', 'Do it!', 'Approve', 'okay']) {
      expect(classifyConfirmation(text)).toBe('affirmative')
    }
  })

  it('accepts a plain no', () => {
    for (const text of ['No', 'nope', 'Stop.', 'cancel', "don't", 'never mind', 'wait']) {
      expect(classifyConfirmation(text)).toBe('negative')
    }
  })

  it('accepts an answer given in more than one breath', () => {
    expect(classifyConfirmation('No, stop.')).toBe('negative')
    expect(classifyConfirmation('yes, go ahead')).toBe('affirmative')
    expect(classifyConfirmation('no, cancel that')).toBe('unclear')
  })

  it('ignores filler around a one-word answer', () => {
    expect(classifyConfirmation('Um, yes please')).toBe('affirmative')
    expect(classifyConfirmation('uh no thanks')).toBe('negative')
  })

  /**
   * The case that decides the design: a qualified yes is not consent. Nothing
   * here is clever enough to work out which part of the sentence the condition
   * attaches to, so it must not try.
   */
  it('refuses to read a sentence as consent', () => {
    for (const text of [
      'yes but not the second one',
      'yes if it is only the build directory',
      'well I think so',
      'sure, after you show me the diff',
      'delete the temp files instead',
      'no wait yes'
    ]) {
      expect(classifyConfirmation(text)).toBe('unclear')
    }
  })

  it('treats nothing at all as no answer', () => {
    expect(classifyConfirmation('')).toBe('unclear')
    expect(classifyConfirmation('   ')).toBe('unclear')
  })
})

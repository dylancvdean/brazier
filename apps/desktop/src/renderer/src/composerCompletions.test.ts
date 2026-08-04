import { describe, expect, it } from 'vitest'

import {
  composerCompletionsFor,
  replaceTrailingCommand,
  trailingCommand
} from './composerCompletions'

const SUGGESTIONS = [
  { value: 'ultrathink', description: 'Max reasoning' },
  { value: '/review', description: 'Review changes' },
  { value: '/todo', description: 'Show todos' },
  { value: '/todo-add', description: 'Add a todo' }
]

describe('composerCompletions', () => {
  it('never surfaces suggestions for an empty draft', () => {
    expect(trailingCommand('')).toBe('')
    expect(composerCompletionsFor('', SUGGESTIONS)).toEqual([])
    expect(composerCompletionsFor('   ', SUGGESTIONS)).toEqual([])
  })

  it('surfaces every slash command when the user types a bare slash', () => {
    expect(trailingCommand('/')).toBe('/')
    expect(composerCompletionsFor('/', SUGGESTIONS).map((entry) => entry.value)).toEqual([
      '/review',
      '/todo',
      '/todo-add'
    ])
  })

  it('matches slash commands by their slash-prefixed query', () => {
    expect(composerCompletionsFor('/re', SUGGESTIONS).map((entry) => entry.value)).toEqual(['/review'])
    expect(composerCompletionsFor('run /todo', SUGGESTIONS).map((entry) => entry.value)).toEqual([
      '/todo',
      '/todo-add'
    ])
  })

  it('still matches bare magic words', () => {
    expect(composerCompletionsFor('ultra', SUGGESTIONS).map((entry) => entry.value)).toEqual([
      'ultrathink'
    ])
  })

  it('is case-insensitive', () => {
    expect(trailingCommand('/REV')).toBe('/rev')
    expect(composerCompletionsFor('/REV', SUGGESTIONS).map((entry) => entry.value)).toEqual([
      '/review'
    ])
  })

  it('returns nothing when the trailing word is not a command prefix', () => {
    expect(trailingCommand('fix the parser')).toBe('parser')
    expect(composerCompletionsFor('fix the parser', SUGGESTIONS)).toEqual([])
  })

  it('replaces the trailing word with the chosen completion', () => {
    expect(replaceTrailingCommand('run /re', '/review')).toBe('run /review ')
    expect(replaceTrailingCommand('/to', '/todo')).toBe('/todo ')
    expect(replaceTrailingCommand('ultra', 'ultrathink')).toBe('ultrathink ')
    expect(replaceTrailingCommand('', '/todo')).toBe('/todo ')
  })
})

import { describe, expect, it } from 'vitest'

import {
  AGENT_SLASH_SUGGESTIONS,
  CHAT_SLASH_SUGGESTIONS,
  composerCompletionsFor,
  isExactSlashCommand,
  replaceTrailingCommand,
  trailingCommand
} from './composerCompletions'

const SUGGESTIONS = [
  { value: 'ultrathink', description: 'Max reasoning' },
  { value: '/review', description: 'Review changes' },
  { value: '/todo', description: 'Show todos' },
  { value: '/todo-add', description: 'Add a todo' },
  { value: '/gpt_4_mode', description: 'Underscore and digit command' }
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
      '/todo-add',
      '/gpt_4_mode'
    ])
  })

  it('matches slash commands with hyphens, underscores, and digits', () => {
    expect(trailingCommand('/todo-a')).toBe('/todo-a')
    expect(composerCompletionsFor('/todo-', SUGGESTIONS).map((entry) => entry.value)).toEqual([
      '/todo-add'
    ])
    expect(trailingCommand('/gpt_4')).toBe('/gpt_4')
    expect(composerCompletionsFor('/gpt_4', SUGGESTIONS).map((entry) => entry.value)).toEqual([
      '/gpt_4_mode'
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

  it('replaces the trailing word with the chosen completion, including hyphenated queries', () => {
    expect(replaceTrailingCommand('run /re', '/review')).toBe('run /review ')
    expect(replaceTrailingCommand('/todo-a', '/todo-add')).toBe('/todo-add ')
    expect(replaceTrailingCommand('/to', '/todo')).toBe('/todo ')
    expect(replaceTrailingCommand('ultra', 'ultrathink')).toBe('ultrathink ')
    expect(replaceTrailingCommand('', '/todo')).toBe('/todo ')
  })

  it('recognizes built-in Chat and Agent slash commands', () => {
    expect(isExactSlashCommand('/incognito', CHAT_SLASH_SUGGESTIONS)).toBe(true)
    expect(isExactSlashCommand(' /dream ', CHAT_SLASH_SUGGESTIONS)).toBe(true)
    expect(isExactSlashCommand('/unknown', CHAT_SLASH_SUGGESTIONS)).toBe(false)
    expect(isExactSlashCommand('/worktree', AGENT_SLASH_SUGGESTIONS)).toBe(true)
    expect(isExactSlashCommand('/sandbox', AGENT_SLASH_SUGGESTIONS)).toBe(true)
  })
})

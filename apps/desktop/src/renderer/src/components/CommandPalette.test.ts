import { describe, expect, it } from 'vitest'
import {
  MANAGE_PALETTE_ENTRIES,
  filterAndRankPaletteItems,
  scorePaletteItem,
  type CommandPaletteItem
} from './CommandPalette'

const noop = (): void => {}

function buildManageItems(): CommandPaletteItem[] {
  return MANAGE_PALETTE_ENTRIES.map((entry) => ({
    id: `manage:${entry.section}`,
    category: 'manage',
    title: entry.title,
    subtitle: entry.subtitle,
    keywords: entry.keywords,
    onSelect: noop
  }))
}

describe('CommandPalette search and ranking', () => {
  it('deep-links Manage sections by domain keywords (vram, kv cache, lora, mcp, ollama, api key, sandbox)', () => {
    const items = buildManageItems()

    expect(filterAndRankPaletteItems(items, 'kv cache')[0]?.id).toBe('manage:engine')
    expect(filterAndRankPaletteItems(items, 'vram')[0]?.id).toBe('manage:recommended')
    expect(filterAndRankPaletteItems(items, 'lora')[0]?.id).toBe('manage:library')
    expect(filterAndRankPaletteItems(items, 'mcp')[0]?.id).toBe('manage:mcp')
    expect(filterAndRankPaletteItems(items, 'remote')[0]?.id).toBe('manage:remote')
    expect(filterAndRankPaletteItems(items, 'ollama')[0]?.id).toBe('manage:remote')
    expect(
      filterAndRankPaletteItems(items, 'api key')
        .slice(0, 3)
        .map((entry) => entry.id)
    ).toContain('manage:server')
    expect(
      filterAndRankPaletteItems(items, 'sandbox')
        .slice(0, 2)
        .map((entry) => entry.id)
    ).toContain('manage:agent')
  })

  it('prefers exact and prefix title matches over distant substring or fuzzy matches', () => {
    const items: CommandPaletteItem[] = [
      {
        id: 'conv',
        category: 'conversations',
        title: 'Debugging unload behavior in scripts',
        onSelect: noop
      },
      {
        id: 'unload',
        category: 'actions',
        title: 'Unload model',
        subtitle: 'Free VRAM',
        onSelect: noop
      }
    ]
    const ranked = filterAndRankPaletteItems(items, 'unload')
    expect(ranked[0]?.id).toBe('unload')
  })

  it('requires every whitespace-separated query token to match', () => {
    const item: CommandPaletteItem = {
      id: 'model:qwen',
      category: 'models',
      title: 'Qwen3-VL-8B-Instruct',
      subtitle: 'Remote · Workstation',
      keywords: ['qwen', 'remote'],
      onSelect: noop
    }
    expect(scorePaletteItem(item, 'qwen workstation')).toBeGreaterThan(0)
    expect(scorePaletteItem(item, 'qwen whisper')).toBe(0)
  })

  it('supports compact subsequence fuzzy queries on titles', () => {
    const item: CommandPaletteItem = {
      id: 'action:incognito',
      category: 'actions',
      title: 'Toggle incognito chat',
      onSelect: noop
    }
    expect(scorePaletteItem(item, 'incog')).toBeGreaterThan(0)
    expect(scorePaletteItem(item, 'tglinc')).toBeGreaterThan(0)
  })
})

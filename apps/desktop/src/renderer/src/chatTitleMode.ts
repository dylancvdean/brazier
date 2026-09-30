export type ChatTitleMode = 'never' | 'always' | 'over-20-tokens'

const CHAT_TITLE_MODE_KEY = 'brazier.chatTitleMode.v1'

export function readChatTitleMode(): ChatTitleMode {
  try {
    const value = localStorage.getItem(CHAT_TITLE_MODE_KEY)
    return value === 'never' || value === 'over-20-tokens' || value === 'always' ? value : 'always'
  } catch {
    return 'always'
  }
}

export function writeChatTitleMode(mode: ChatTitleMode): void {
  try {
    localStorage.setItem(CHAT_TITLE_MODE_KEY, mode)
  } catch {
    // Best-effort persistence.
  }
}

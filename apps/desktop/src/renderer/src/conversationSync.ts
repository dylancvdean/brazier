export async function selectedConversationWasRemoved(
  id: string,
  visible: ReadonlyArray<{ id: string }>,
  query: string,
  loadAll: () => Promise<ReadonlyArray<{ id: string }>>
): Promise<boolean> {
  if (visible.some((conversation) => conversation.id === id)) return false
  return !query.trim() || !(await loadAll()).some((conversation) => conversation.id === id)
}

export function startConversationSync(
  browser: Pick<Window, 'addEventListener' | 'removeEventListener' | 'setInterval' | 'clearInterval'>,
  page: Pick<Document, 'addEventListener' | 'removeEventListener' | 'visibilityState'>,
  idle: () => boolean,
  refresh: () => Promise<unknown>
): () => void {
  let pending = false
  const revalidate = (): void => {
    if (pending || page.visibilityState !== 'visible' || !idle()) return
    pending = true
    void refresh().catch(() => undefined).finally(() => { pending = false })
  }
  const timer = browser.setInterval(revalidate, 15_000)
  browser.addEventListener('focus', revalidate)
  page.addEventListener('visibilitychange', revalidate)
  return () => {
    browser.clearInterval(timer)
    browser.removeEventListener('focus', revalidate)
    page.removeEventListener('visibilitychange', revalidate)
  }
}

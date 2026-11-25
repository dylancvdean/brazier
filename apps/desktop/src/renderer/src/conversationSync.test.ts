import { afterEach, expect, it, vi } from 'vitest'
import { selectedConversationWasRemoved, startConversationSync } from './conversationSync'

afterEach(() => vi.useRealTimers())

it('confirms the selected chat was deleted without mistaking a search miss for deletion', async () => {
  const selected = { id: 'selected' }
  const loadAll = vi.fn(async () => [selected])
  await expect(selectedConversationWasRemoved(selected.id, [selected], 'search', loadAll)).resolves.toBe(false)
  await expect(selectedConversationWasRemoved(selected.id, [], '', loadAll)).resolves.toBe(true)
  expect(loadAll).not.toHaveBeenCalled()
  await expect(selectedConversationWasRemoved(selected.id, [], 'search', loadAll)).resolves.toBe(false)
  loadAll.mockResolvedValueOnce([])
  await expect(selectedConversationWasRemoved(selected.id, [], 'search', loadAll)).resolves.toBe(true)
})

it('refreshes on focus, resume, and the foreground interval without interrupting a turn', async () => {
  vi.useFakeTimers()
  const browser = Object.assign(new EventTarget(), { setInterval, clearInterval }) as unknown as Window
  const page = Object.assign(new EventTarget(), { visibilityState: 'hidden' })
  let idle = true
  const refresh = vi.fn(async () => undefined)
  const stop = startConversationSync(browser, page as unknown as Document, () => idle, refresh)

  await vi.advanceTimersByTimeAsync(15_000)
  expect(refresh).not.toHaveBeenCalled()
  page.visibilityState = 'visible'
  page.dispatchEvent(new Event('visibilitychange'))
  browser.dispatchEvent(new Event('focus'))
  expect(refresh).toHaveBeenCalledTimes(1)
  await vi.advanceTimersByTimeAsync(15_000)
  expect(refresh).toHaveBeenCalledTimes(2)

  idle = false
  browser.dispatchEvent(new Event('focus'))
  await vi.advanceTimersByTimeAsync(15_000)
  expect(refresh).toHaveBeenCalledTimes(2)
  idle = true
  browser.dispatchEvent(new Event('focus'))
  expect(refresh).toHaveBeenCalledTimes(3)
  stop()
  browser.dispatchEvent(new Event('focus'))
  await vi.advanceTimersByTimeAsync(15_000)
  expect(refresh).toHaveBeenCalledTimes(3)
})

import { useEffect, useRef } from 'react'

export function focusableElements(root: HTMLElement): HTMLElement[] {
  return Array.from(
    root.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )
  ).filter(
    (el) =>
      !el.hasAttribute('disabled') &&
      el.getAttribute('aria-hidden') !== 'true' &&
      !el.hasAttribute('hidden')
  )
}

/**
 * Shared accessible modal/dialog focus-trap + Escape dismissal hook.
 *
 * Focuses `[data-autofocus]` (or the first focusable control) when opened,
 * wraps Tab/Shift+Tab inside the overlay, stops Escape propagation so parent
 * Escape handlers do not close underlying panels, and restores prior focus.
 */
export function useDialogOverlay(
  open: boolean,
  onClose: () => void
): React.RefObject<HTMLDivElement | null> {
  const overlayRef = useRef<HTMLDivElement | null>(null)
  const onCloseRef = useRef(onClose)
  onCloseRef.current = onClose

  useEffect(() => {
    if (!open) return
    const root = overlayRef.current
    if (!root) return
    const previouslyFocused = document.activeElement as HTMLElement | null
    const preferred = root.querySelector<HTMLElement>('[data-autofocus]:not([disabled])')
    const items = focusableElements(root)
    ;(preferred ?? items[0])?.focus()

    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        onCloseRef.current()
        return
      }
      if (event.key !== 'Tab') return
      const current = focusableElements(root)
      if (current.length === 0) return
      const first = current[0]
      const last = current[current.length - 1]
      const active = document.activeElement
      if (event.shiftKey && (active === first || !root.contains(active))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && active === last) {
        event.preventDefault()
        first.focus()
      }
    }

    root.addEventListener('keydown', onKeyDown)
    return () => {
      root.removeEventListener('keydown', onKeyDown)
      previouslyFocused?.focus?.()
    }
  }, [open])

  return overlayRef
}

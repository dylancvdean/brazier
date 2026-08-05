import { Maximize, Minimize, X } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'

/**
 * Track the native Chromium fullscreen state of a single element.
 *
 * `setRef` is a callback ref: attach it to the element that should go
 * fullscreen (an image, a video, or a whole panel). `active` is true only when
 * that specific element is the one Chromium has fullscreened, so several of
 * these can live in the same view without fighting over the state.
 */
export function useFullscreen<T extends HTMLElement>(): {
  setRef: (element: T | null) => void
  active: boolean
  toggle: () => void
} {
  const [element, setElement] = useState<T | null>(null)
  const [active, setActive] = useState(false)

  useEffect(() => {
    const sync = (): void =>
      setActive(Boolean(element && document.fullscreenElement === element))
    document.addEventListener('fullscreenchange', sync)
    sync()
    return () => document.removeEventListener('fullscreenchange', sync)
  }, [element])

  const toggle = useCallback(() => {
    if (document.fullscreenElement) {
      void document.exitFullscreen()
    } else {
      void element?.requestFullscreen?.()
    }
  }, [element])

  return { setRef: setElement, active, toggle }
}

/**
 * Compact icon button that overlays the corner of a previewed image. Videos
 * already expose fullscreen through their native controls, so this is only
 * rendered for stills.
 */
export function MediaFullscreenIcon({
  active,
  toggle
}: {
  active: boolean
  toggle: () => void
}): React.JSX.Element {
  return (
    <button
      type="button"
      className="media-fullscreen-corner"
      title={active ? 'Exit fullscreen' : 'View fullscreen'}
      aria-label={active ? 'Exit fullscreen' : 'View fullscreen'}
      onClick={toggle}
    >
      {active ? <Minimize size={13} /> : <Maximize size={13} />}
    </button>
  )
}

/**
 * Visible close button shown while an image fills the screen. The fullscreen
 * element is the preview wrapper, so this button lives inside it and is only
 * rendered when Chromium reports that wrapper as the fullscreen target.
 */
export function MediaFullscreenExit({ toggle }: { toggle: () => void }): React.JSX.Element {
  return (
    <button
      type="button"
      className="media-fullscreen-exit"
      title="Exit fullscreen"
      aria-label="Exit fullscreen"
      onClick={toggle}
    >
      <X size={16} />
    </button>
  )
}

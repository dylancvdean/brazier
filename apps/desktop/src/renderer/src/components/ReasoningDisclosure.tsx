import { useState } from 'react'

const PREVIEW_CHARS = 160

/** Collapsed preview of a thinking trace; expand for the full text. */
export function ReasoningDisclosure({
  text,
  defaultOpen = false
}: {
  text: string
  defaultOpen?: boolean
}): React.JSX.Element | null {
  const trimmed = text.trim()
  if (!trimmed) return null

  const [open, setOpen] = useState(defaultOpen)
  const needsCollapse = trimmed.length > PREVIEW_CHARS || trimmed.includes('\n')
  const preview = needsCollapse
    ? trimmed.replace(/\s+/g, ' ').slice(0, PREVIEW_CHARS).trimEnd() +
      (trimmed.length > PREVIEW_CHARS ? '…' : '')
    : trimmed

  return (
    <details
      className="reasoning-disclosure"
      open={open}
      onToggle={(event) => setOpen((event.target as HTMLDetailsElement).open)}
    >
      <summary>
        <span className="reasoning-disclosure-label">Reasoning</span>
        {!open && needsCollapse ? (
          <span className="reasoning-disclosure-preview">{preview}</span>
        ) : null}
      </summary>
      <pre>{trimmed}</pre>
    </details>
  )
}

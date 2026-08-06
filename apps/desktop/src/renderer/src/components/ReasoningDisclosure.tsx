import { type ReactNode, useState } from 'react'

const PREVIEW_CHARS = 160

/** Collapsed preview of a thinking trace; expand for the full text. */
export function ReasoningDisclosure({
  text,
  defaultOpen = false
}: {
  text: string
  defaultOpen?: boolean
}): React.JSX.Element | null {
  const [open, setOpen] = useState(defaultOpen)
  const trimmed = text.trim()
  if (!trimmed) return null

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

/**
 * A multi-step assistant turn's working trace: the thinking text, tool calls
 * (as pills), media, and any intermediate prose, shown in execution order
 * inside one disclosure. The final response text is rendered separately, not
 * here. Defaults to open when the turn actually used tools, so tool activity
 * stays visible without being scattered across the message body.
 */
export function TurnTrace({
  reasoning,
  defaultOpen = false,
  children
}: {
  reasoning: string
  defaultOpen?: boolean
  children: ReactNode
}): React.JSX.Element | null {
  const trimmed = reasoning.trim()
  const [open, setOpen] = useState(defaultOpen)
  const preview =
    trimmed.length > PREVIEW_CHARS
      ? trimmed.replace(/\s+/g, ' ').slice(0, PREVIEW_CHARS).trimEnd() + '…'
      : trimmed

  return (
    <details
      className="reasoning-disclosure"
      open={open}
      onToggle={(event) => setOpen((event.target as HTMLDetailsElement).open)}
    >
      <summary>
        <span className="reasoning-disclosure-label">Reasoning & tool calls</span>
        {!open && preview ? (
          <span className="reasoning-disclosure-preview">{preview}</span>
        ) : null}
      </summary>
      <div className="reasoning-trace">{children}</div>
    </details>
  )
}

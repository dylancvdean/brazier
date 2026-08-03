import { AlertTriangle, Check, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import type { OmpExtensionUiResponse } from '../../../agent/omp/rpcTypes'
import type { OmpPendingDialog } from '../ompSidecar'

/**
 * Native re-render of an OMP extension-UI dialog (`ask`/select/confirm/input/
 * editor). The runtime holds the request until this answers with an
 * `extension_ui_response`; until then the sidecar's tool call is parked.
 */

type OmpDialogProps = {
  dialog: OmpPendingDialog
  onResolve: (response: OmpExtensionUiResponse) => void
}

export function OmpDialog({ dialog, onResolve }: OmpDialogProps): React.JSX.Element {
  const [value, setValue] = useState(dialog.kind === 'editor' ? (dialog.prefill ?? '') : '')
  const inputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (dialog.kind === 'input') inputRef.current?.focus()
    if (dialog.kind === 'editor') textareaRef.current?.focus()
  }, [dialog.kind])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        onResolve({ type: 'extension_ui_response', id: dialog.id, cancelled: true })
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [dialog.id, onResolve])

  const ok = (): void => {
    if (dialog.kind === 'confirm') {
      onResolve({ type: 'extension_ui_response', id: dialog.id, confirmed: true })
    } else {
      onResolve({ type: 'extension_ui_response', id: dialog.id, value })
    }
  }

  const cancel = (): void =>
    onResolve({ type: 'extension_ui_response', id: dialog.id, cancelled: true })

  return (
    <div className="omp-dialog-backdrop" role="presentation">
      <div className="omp-dialog" role="dialog" aria-modal="true" aria-label={dialog.title}>
        <header className="omp-dialog-head">
          <strong>{dialog.title}</strong>
        </header>

        {dialog.kind === 'confirm' && (
          <p className="omp-dialog-message">{dialog.message || 'Confirm?'}</p>
        )}

        {dialog.kind === 'select' && (
          <div className="omp-dialog-options">
            {dialog.options.map((option) => (
              <button
                key={option}
                type="button"
                onClick={() =>
                  onResolve({ type: 'extension_ui_response', id: dialog.id, value: option })
                }
              >
                {option}
              </button>
            ))}
          </div>
        )}

        {dialog.kind === 'input' && (
          <input
            ref={inputRef}
            value={value}
            placeholder={dialog.placeholder}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') ok()
            }}
          />
        )}

        {dialog.kind === 'editor' && (
          <textarea
            ref={textareaRef}
            value={value}
            spellCheck={false}
            onChange={(event) => setValue(event.target.value)}
          />
        )}

        {dialog.kind === 'select' && (
          <p className="omp-dialog-hint">
            <AlertTriangle size={12} /> Pick an option, or press Escape to cancel.
          </p>
        )}

        {(dialog.kind === 'confirm' || dialog.kind === 'input' || dialog.kind === 'editor') && (
          <div className="omp-dialog-actions">
            <button type="button" className="omp-dialog-cancel" onClick={cancel}>
              <X size={14} /> Cancel
            </button>
            <button
              type="button"
              className="omp-dialog-ok"
              disabled={dialog.kind !== 'confirm' && !value.trim()}
              onClick={ok}
            >
              <Check size={14} /> {dialog.kind === 'confirm' ? 'Confirm' : 'OK'}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { ErrorBoundary, RenderSlot } from './ErrorBoundary'

describe('ErrorBoundary', () => {
  it('derives an error state and clears it when the reset key changes', () => {
    const error = new Error('bad payload')
    expect(ErrorBoundary.getDerivedStateFromError(error)).toEqual({ error })
    expect(
      ErrorBoundary.getDerivedStateFromProps(
        { label: 'message', resetKey: 'chat', children: null },
        { error, resetKey: 'chat' }
      )
    ).toBeNull()
    expect(
      ErrorBoundary.getDerivedStateFromProps(
        { label: 'message', resetKey: 'agent', children: null },
        { error, resetKey: 'chat' }
      )
    ).toEqual({ error: null, resetKey: 'agent' })
  })

  it('renders children through a slot when nothing throws', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    expect(
      renderToStaticMarkup(
        <ErrorBoundary label="message">
          <RenderSlot render={() => <p>ok</p>} />
        </ErrorBoundary>
      )
    ).toBe('<p>ok</p>')
  })
})

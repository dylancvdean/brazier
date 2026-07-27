import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { ReasoningDisclosure } from './ReasoningDisclosure'

describe('ReasoningDisclosure', () => {
  it('renders nothing for empty text', () => {
    expect(renderToStaticMarkup(<ReasoningDisclosure text="   " />)).toBe('')
  })

  it('shows a collapsed preview for long traces', () => {
    const long =
      'First I consider the polynomial ring, then I reduce the expression step by step until the Jacobian identity is clear enough to verify with code that exercises every branch of the proposed mapping.'
    const html = renderToStaticMarkup(<ReasoningDisclosure text={long} />)
    expect(html).toContain('Reasoning')
    expect(html).toContain('reasoning-disclosure-preview')
    expect(html).toContain('First I consider the polynomial')
    expect(html).toContain('<pre>')
  })

  it('can start expanded while streaming', () => {
    const html = renderToStaticMarkup(
      <ReasoningDisclosure text={'short thought'} defaultOpen />
    )
    expect(html).toContain('open')
    expect(html).toContain('short thought')
  })
})

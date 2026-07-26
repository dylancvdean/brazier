import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Markdown } from './Markdown'

describe('Markdown', () => {
  it('renders conversational Markdown and GFM structures', () => {
    const html = renderToStaticMarkup(
      <Markdown>{'# Result\n\n- one\n- two\n\n| A | B |\n| - | - |\n| 1 | 2 |'}</Markdown>
    )
    expect(html).toContain('<h1>Result</h1>')
    expect(html).toContain('<ul>')
    expect(html).toContain('<table>')
  })

  it('does not render raw model-authored HTML', () => {
    const html = renderToStaticMarkup(<Markdown>{'<script>alert(1)</script>'}</Markdown>)
    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;')
  })
})

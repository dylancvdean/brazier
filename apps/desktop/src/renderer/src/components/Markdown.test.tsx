import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Markdown } from './Markdown'
import { normalizeMathDelimiters } from './mathDelimiters'

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

  it('renders inline and display LaTeX via KaTeX', () => {
    const inline = renderToStaticMarkup(<Markdown>{'Euler: $e^{i\\pi}+1=0$'}</Markdown>)
    expect(inline).toContain('class="katex"')
    expect(inline).toContain('e^{i\\pi}+1=0')

    const paren = renderToStaticMarkup(<Markdown>{'Inline \\(a+b\\) here'}</Markdown>)
    expect(paren).toContain('class="katex"')

    const singleLineDisplay = renderToStaticMarkup(
      <Markdown>{'Block:\n\n$$\\int_0^1 x\\,dx$$'}</Markdown>
    )
    expect(singleLineDisplay).toContain('katex-display')

    const bracketDisplay = renderToStaticMarkup(
      <Markdown>{'Block:\n\n\\[a^2+b^2=c^2\\]'}</Markdown>
    )
    expect(bracketDisplay).toContain('katex-display')
  })
})

describe('normalizeMathDelimiters', () => {
  it('rewrites LaTeX parenthesis and bracket delimiters', () => {
    expect(normalizeMathDelimiters('see \\(x^2\\) now')).toBe('see $x^2$ now')
    expect(normalizeMathDelimiters('block\n\\[E=mc^2\\]\ndone')).toBe(
      'block\n\n$$\nE=mc^2\n$$\n\ndone'
    )
  })

  it('promotes same-line $$ to display fences', () => {
    expect(normalizeMathDelimiters('intro\n\n$$a+b$$\n\nout')).toBe('intro\n\n$$\na+b\n$$\n\nout')
  })

  it('leaves fenced and inline code alone', () => {
    const fenced = '```\n\\(x\\)\n$$\na$$\n```'
    expect(normalizeMathDelimiters(fenced)).toBe(fenced)
    expect(normalizeMathDelimiters('use `\\(x\\)` and `$y$`')).toBe('use `\\(x\\)` and `$y$`')
  })
})

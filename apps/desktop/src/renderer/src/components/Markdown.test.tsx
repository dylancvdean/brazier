import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Markdown } from './Markdown'
import { sanitizeMarkdownUrl } from './markdownParser'
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
    expect(inline).toContain('markdown-math-inline')
    expect(inline).toContain('e^{i\\pi}+1=0')

    const paren = renderToStaticMarkup(<Markdown>{'Inline \\(a+b\\) here'}</Markdown>)
    expect(paren).toContain('class="katex"')

    const singleLineDisplay = renderToStaticMarkup(
      <Markdown>{'Block:\n\n$$\\int_0^1 x\\,dx$$'}</Markdown>
    )
    expect(singleLineDisplay).toContain('markdown-math-display')
    expect(singleLineDisplay).toContain('katex-display')

    const multiLineDisplay = renderToStaticMarkup(
      <Markdown>{'$$\n\\sum_{k=1}^n k = \\frac{n(n+1)}{2}\n$$'}</Markdown>
    )
    expect(multiLineDisplay).toContain('katex-display')

    const bracketDisplay = renderToStaticMarkup(
      <Markdown>{'Block:\n\n\\[a^2+b^2=c^2\\]'}</Markdown>
    )
    expect(bracketDisplay).toContain('katex-display')
  })

  it('renders GFM tables with alignment, escaped pipes, inline code pipes, and inline math pipes', () => {
    const md = [
      '| Expr | Code | Escaped | Price |',
      '| :--- | :---: | ---: | --- |',
      '| $|x| + |y|$ | `a|b` | foo \\| bar | $100 |'
    ].join('\n')

    const html = renderToStaticMarkup(<Markdown>{md}</Markdown>)
    expect(html).toContain('<th style="text-align:left">Expr</th>')
    expect(html).toContain('<th style="text-align:center">Code</th>')
    expect(html).toContain('<th style="text-align:right">Escaped</th>')
    expect(html).toContain('<th>Price</th>')
    expect(html).toContain('<code>a|b</code>')
    expect(html).toContain('foo | bar')
    expect(html).toContain('class="katex"')
    expect(html).toContain('|x| + |y|')
    expect(html).toContain('<td')
    expect(html).toContain('>$100</td>')
  })

  it('preserves snake_case_identifiers while supporting emphasis and strikethrough', () => {
    const html = renderToStaticMarkup(
      <Markdown>
        {
          'Keep snake_case_identifier and foo_bar_baz intact, but render _italic_, **bold**, __strong__, ***both***, and ~~gone~~ (~10ms stays plain).'
        }
      </Markdown>
    )
    expect(html).toContain('snake_case_identifier')
    expect(html).toContain('foo_bar_baz')
    expect(html).not.toContain('<em>case</em>')
    expect(html).not.toContain('<em>bar</em>')
    expect(html).toContain('<em>italic</em>')
    expect(html).toContain('<strong>bold</strong>')
    expect(html).toContain('<strong>strong</strong>')
    expect(html).toContain('<em><strong>both</strong></em>')
    expect(html).toContain('<del>gone</del>')
    expect(html).toContain('~10ms stays plain')
  })

  it('renders nested lists, ordered start offsets, and GFM task list checkboxes', () => {
    const md = [
      '3. Third item',
      '   - Nested bullet A',
      '   - Nested bullet B',
      '4. Fourth item',
      '',
      '- [ ] Pending task',
      '- [x] Completed lowercase',
      '- [X] Completed uppercase'
    ].join('\n')

    const html = renderToStaticMarkup(<Markdown>{md}</Markdown>)
    expect(html).toContain('<ol start="3">')
    expect(html).toContain('<li>Third item<ul><li>Nested bullet A</li><li>Nested bullet B</li></ul></li>')
    expect(html).toContain('<input type="checkbox" disabled="" readOnly=""/>Pending task')
    expect(html).toContain(
      '<input type="checkbox" disabled="" readOnly="" checked=""/>Completed lowercase'
    )
    expect(html).toContain(
      '<input type="checkbox" disabled="" readOnly="" checked=""/>Completed uppercase'
    )
  })

  it('supports balanced parentheses in link/image URLs and GFM autolinks with punctuation trimming', () => {
    const md = [
      'Read [Rust](https://en.wikipedia.org/wiki/Rust_(programming_language) "Wiki") now.',
      '',
      'Bare link (see https://en.wikipedia.org/wiki/Rust_(programming_language)). Also <https://example.com/docs>.'
    ].join('\n')

    const html = renderToStaticMarkup(<Markdown>{md}</Markdown>)
    expect(html).toContain(
      '<a href="https://en.wikipedia.org/wiki/Rust_(programming_language)" title="Wiki" target="_blank" rel="noreferrer noopener">Rust</a>'
    )
    expect(html).toContain(
      '(see <a href="https://en.wikipedia.org/wiki/Rust_(programming_language)" target="_blank" rel="noreferrer noopener">https://en.wikipedia.org/wiki/Rust_(programming_language)</a>).'
    )
    expect(html).toContain(
      '<a href="https://example.com/docs" target="_blank" rel="noreferrer noopener">https://example.com/docs</a>'
    )
  })

  it('rejects javascript:, vbscript:, file:, and unsafe data: URLs while keeping safe URLs', () => {
    expect(sanitizeMarkdownUrl('https://example.com/a', 'link')).toBe('https://example.com/a')
    expect(sanitizeMarkdownUrl('mailto:dev@example.com', 'link')).toBe('mailto:dev@example.com')
    expect(sanitizeMarkdownUrl('mailto:dev@example.com', 'image')).toBeNull()
    expect(sanitizeMarkdownUrl('#section-2', 'link')).toBe('#section-2')
    expect(sanitizeMarkdownUrl('/assets/pic.png', 'image')).toBe('/assets/pic.png')
    expect(sanitizeMarkdownUrl('./local/doc.md', 'link')).toBe('./local/doc.md')
    expect(sanitizeMarkdownUrl('blob:https://example.com/123', 'image')).toBe(
      'blob:https://example.com/123'
    )
    expect(sanitizeMarkdownUrl('blob:https://example.com/123', 'link')).toBeNull()
    expect(sanitizeMarkdownUrl('data:image/png;base64,iVBORw0KGgo=', 'image')).toBe(
      'data:image/png;base64,iVBORw0KGgo='
    )
    expect(sanitizeMarkdownUrl('data:image/svg+xml;base64,PHN2Zz4=', 'image')).toBeNull()
    expect(sanitizeMarkdownUrl('data:text/html,<script>alert(1)</script>', 'link')).toBeNull()
    expect(sanitizeMarkdownUrl('javascript:alert(1)', 'link')).toBeNull()
    expect(sanitizeMarkdownUrl('JAVASCRIPT:alert(1)', 'link')).toBeNull()
    expect(sanitizeMarkdownUrl('java\nscript:alert(1)', 'link')).toBeNull()
    expect(sanitizeMarkdownUrl('java\tscript:alert(1)', 'image')).toBeNull()
    expect(sanitizeMarkdownUrl('java script:alert(1)', 'link')).toBeNull()
    expect(sanitizeMarkdownUrl('vbscript:msgbox(1)', 'link')).toBeNull()
    expect(sanitizeMarkdownUrl('file:///etc/passwd', 'link')).toBeNull()
    expect(sanitizeMarkdownUrl('//evil.example.com/x', 'link')).toBeNull()

    const html = renderToStaticMarkup(
      <Markdown>
        {'[Click me](javascript:alert(1)) and ![Bad](javascript:alert(2)) and [Jump](#ok)'}
      </Markdown>
    )
    expect(html).not.toContain('javascript:')
    expect(html).not.toContain('<img')
    expect(html).toContain('Click me and Bad and <a href="#ok">Jump</a>')
  })

  it('distinguishes $100 currency amounts from $x + y$ inline math', () => {
    const currency = renderToStaticMarkup(
      <Markdown>
        {
          'Plans cost $100 or $200, ranges $10-$20, escaped \\$x + y\\$, and $100 while $x + y = z$ is math.'
        }
      </Markdown>
    )
    expect(currency).toContain('Plans cost $100 or $200, ranges $10-$20, escaped $x + y$, and $100 while ')
    expect(currency).toContain('class="katex"')
    expect(currency).toContain('x + y = z')
  })

  it('renders fenced code blocks with toolbar, language badge, copy button, and handles unclosed streaming blocks', () => {
    const closed = renderToStaticMarkup(
      <Markdown>{'```tsx\nconst view = <App />\n```'}</Markdown>
    )
    expect(closed).toContain('class="markdown-code-block"')
    expect(closed).toContain('class="markdown-code-toolbar"')
    expect(closed).toContain('<span class="markdown-code-lang">tsx</span>')
    expect(closed).toContain('class="markdown-code-copy"')
    expect(closed).toContain('aria-label="Copy code"')
    expect(closed).toContain('<code class="language-tsx">const view = &lt;App /&gt;</code>')

    const streamingFence = renderToStaticMarkup(
      <Markdown>{'```python\ndef stream_chunk():\n    return 42'}</Markdown>
    )
    expect(streamingFence).toContain('<span class="markdown-code-lang">python</span>')
    expect(streamingFence).toContain('<code class="language-python">def stream_chunk():\n    return 42</code>')

    const streamingMath = renderToStaticMarkup(<Markdown>{'$$\n\\frac{a + b}{c}'}</Markdown>)
    expect(streamingMath).toContain('markdown-math-display')
    expect(streamingMath).toContain('katex-display')
  })

  it('renders Setext headings, recursive blockquotes, horizontal rules, and hard line breaks', () => {
    const md = [
      'Main Title',
      '===',
      '',
      'Subtitle',
      '---',
      '',
      '> ### Quoted Heading',
      '> - Quoted item 1',
      '> - Quoted item 2',
      '',
      'Line one  ',
      'Line two\\',
      'Line three',
      '',
      '***'
    ].join('\n')

    const html = renderToStaticMarkup(<Markdown>{md}</Markdown>)
    expect(html).toContain('<h1>Main Title</h1>')
    expect(html).toContain('<h2>Subtitle</h2>')
    expect(html).toContain('<blockquote><h3>Quoted Heading</h3><ul><li>Quoted item 1</li><li>Quoted item 2</li></ul></blockquote>')
    expect(html).toContain('Line one<br/>Line two<br/>Line three')
    expect(html).toContain('<hr/>')
  })

  it('respects CommonMark fence run lengths and multi-backtick inline code spans', () => {
    const md = [
      '````md',
      '```js',
      'const inside = 1',
      '```',
      '````',
      '',
      'Inline `` `literal` `` backticks.'
    ].join('\n')

    const html = renderToStaticMarkup(<Markdown>{md}</Markdown>)
    expect(html).toContain('<span class="markdown-code-lang">md</span>')
    expect(html).toContain('```js\nconst inside = 1\n```')
    expect(html).toContain('<code>`literal`</code>')
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

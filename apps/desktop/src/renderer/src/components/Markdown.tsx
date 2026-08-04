import 'katex/dist/katex.min.css'

import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import { normalizeMathDelimiters } from './mathDelimiters'

/**
 * Shared, safe Markdown presentation for model-authored conversational text.
 *
 * Raw HTML stays disabled by react-markdown's default. GFM adds the forms local
 * models commonly emit—tables, task lists, strikethrough, and autolinks. Math
 * uses remark-math + KaTeX (`$…$` / `$$…$$`), after normalizing `\(...\)` /
 * `\[...\]` and same-line display delimiters.
 */
export function Markdown({
  children,
  className = ''
}: {
  children: string
  className?: string
}): React.JSX.Element {
  // Model output frequently ends with newlines. Trimming trailing whitespace
  // keeps them out of the DOM, so selecting and copying a message does not
  // pick up invisible trailing newlines.
  const source = normalizeMathDelimiters(children).trimEnd()

  return (
    <div className={`markdown-body ${className}`.trim()}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: 'ignore' }]]}
        components={{
          a: ({ children: label, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer">
              {label}
            </a>
          )
        }}
      >
        {source}
      </ReactMarkdown>
    </div>
  )
}

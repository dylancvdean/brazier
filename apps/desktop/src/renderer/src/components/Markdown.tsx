import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

/**
 * Shared, safe Markdown presentation for model-authored conversational text.
 *
 * Raw HTML stays disabled by react-markdown's default. GFM adds the forms local
 * models commonly emit—tables, task lists, strikethrough, and autolinks.
 */
export function Markdown({
  children,
  className = ''
}: {
  children: string
  className?: string
}): React.JSX.Element {
  return (
    <div className={`markdown-body ${className}`.trim()}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ children: label, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer">
              {label}
            </a>
          )
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  )
}

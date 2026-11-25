import { ErrorBoundary } from './ErrorBoundary'
import 'katex/dist/katex.min.css'

import katex from 'katex'
import { Check, Copy } from 'lucide-react'
import {
  Fragment,
  type JSX,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'

import {
  type BlockNode,
  type InlineNode,
  type ListItemNode,
  parseMarkdown,
  sanitizeMarkdownUrl
} from './markdownParser'

const KatexMath = memo(function KatexMath({
  tex,
  displayMode
}: {
  tex: string
  displayMode: boolean
}): JSX.Element {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, {
        displayMode,
        throwOnError: false,
        strict: 'ignore'
      })
    } catch {
      return ''
    }
  }, [tex, displayMode])

  if (!html) {
    return displayMode ? (
      <div className="markdown-math-display">{tex}</div>
    ) : (
      <span className="markdown-math-inline">{tex}</span>
    )
  }

  return displayMode ? (
    <div className="markdown-math-display" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span className="markdown-math-inline" dangerouslySetInnerHTML={{ __html: html }} />
  )
})

const CodeBlock = memo(function CodeBlock({
  lang,
  code
}: {
  lang: string
  code: string
}): JSX.Element {
  const [copied, setCopied] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current)
    }
  }, [])

  const handleCopy = useCallback(() => {
    const markCopied = (): void => {
      setCopied(true)
      if (timerRef.current !== null) clearTimeout(timerRef.current)
      timerRef.current = setTimeout(() => setCopied(false), 1800)
    }

    try {
      const brazier =
        typeof window !== 'undefined'
          ? (window as unknown as { brazier?: { copyText?: (t: string) => unknown } }).brazier
          : undefined
      if (brazier?.copyText) {
        void Promise.resolve(brazier.copyText(code)).then(markCopied).catch(() => undefined)
      } else if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
        void Promise.resolve(navigator.clipboard.writeText(code))
          .then(markCopied)
          .catch(() => undefined)
      }
    } catch {
      // Ignore clipboard errors in restricted environments
    }
  }, [code])

  return (
    <div className="markdown-code-block">
      <div className="markdown-code-toolbar">
        {lang ? <span className="markdown-code-lang">{lang}</span> : <span />}
        <button
          type="button"
          className="markdown-code-copy"
          onClick={handleCopy}
          aria-label="Copy code"
          title="Copy code"
        >
          {copied ? <Check size={13} /> : <Copy size={13} />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <pre>
        <code className={lang ? `language-${lang}` : undefined}>{code}</code>
      </pre>
    </div>
  )
})

function renderInlines(nodes: InlineNode[]): JSX.Element[] {
  return nodes.map((node, idx) => {
    switch (node.type) {
      case 'text':
        return <Fragment key={idx}>{node.value}</Fragment>
      case 'code':
        return <code key={idx}>{node.value}</code>
      case 'math_inline':
        return <KatexMath key={idx} tex={node.tex} displayMode={false} />
      case 'br':
        return <br key={idx} />
      case 'strong':
        return <strong key={idx}>{renderInlines(node.children)}</strong>
      case 'em':
        return <em key={idx}>{renderInlines(node.children)}</em>
      case 'del':
        return <del key={idx}>{renderInlines(node.children)}</del>
      case 'link': {
        const safeHref = sanitizeMarkdownUrl(node.href, 'link')
        if (!safeHref) return <Fragment key={idx}>{renderInlines(node.children)}</Fragment>
        const external = !safeHref.startsWith('#')
        return (
          <a
            key={idx}
            href={safeHref}
            title={node.title}
            target={external ? '_blank' : undefined}
            rel={external ? 'noreferrer noopener' : undefined}
          >
            {renderInlines(node.children)}
          </a>
        )
      }
      case 'image': {
        const safeSrc = sanitizeMarkdownUrl(node.src, 'image')
        if (!safeSrc) return <Fragment key={idx}>{node.alt}</Fragment>
        return <img key={idx} src={safeSrc} alt={node.alt} title={node.title} loading="lazy" />
      }
    }
  })
}

function renderListItem(item: ListItemNode, key: number): JSX.Element {
  const checkbox =
    item.checked !== null ? (
      <input type="checkbox" checked={item.checked} disabled readOnly />
    ) : null

  if (item.tight) {
    if (item.children.length === 0) {
      return <li key={key}>{checkbox}</li>
    }
    return (
      <li key={key}>
        {item.children.map((child, idx) => {
          const lead = idx === 0 ? checkbox : null
          if (child.type === 'paragraph') {
            return (
              <Fragment key={idx}>
                {lead}
                {renderInlines(child.children)}
              </Fragment>
            )
          }
          return (
            <Fragment key={idx}>
              {lead}
              {renderBlock(child, idx)}
            </Fragment>
          )
        })}
      </li>
    )
  }

  if (checkbox && item.children[0]?.type === 'paragraph') {
    const [first, ...rest] = item.children
    return (
      <li key={key}>
        <p>
          {checkbox}
          {renderInlines(first.children)}
        </p>
        {renderBlocks(rest)}
      </li>
    )
  }

  return (
    <li key={key}>
      {checkbox}
      {renderBlocks(item.children)}
    </li>
  )
}

function renderBlock(block: BlockNode, key: number): JSX.Element {
  switch (block.type) {
    case 'heading': {
      const Tag = `h${block.depth}` as 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6'
      return <Tag key={key}>{renderInlines(block.children)}</Tag>
    }
    case 'paragraph':
      return <p key={key}>{renderInlines(block.children)}</p>
    case 'code_block':
      return <CodeBlock key={key} lang={block.lang} code={block.code} />
    case 'math_block':
      return <KatexMath key={key} tex={block.tex} displayMode={true} />
    case 'hr':
      return <hr key={key} />
    case 'blockquote':
      return <blockquote key={key}>{renderBlocks(block.children)}</blockquote>
    case 'list':
      return block.ordered ? (
        <ol key={key} start={block.start !== 1 ? block.start : undefined}>
          {block.items.map((item, idx) => renderListItem(item, idx))}
        </ol>
      ) : (
        <ul key={key}>{block.items.map((item, idx) => renderListItem(item, idx))}</ul>
      )
    case 'table':
      return (
        <table key={key}>
          <thead>
            <tr>
              {block.headers.map((cell, colIdx) => {
                const textAlign = block.align[colIdx] ?? undefined
                return (
                  <th key={colIdx} style={textAlign ? { textAlign } : undefined}>
                    {renderInlines(cell)}
                  </th>
                )
              })}
            </tr>
          </thead>
          {block.rows.length > 0 ? (
            <tbody>
              {block.rows.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  {row.map((cell, colIdx) => {
                    const textAlign = block.align[colIdx] ?? undefined
                    return (
                      <td key={colIdx} style={textAlign ? { textAlign } : undefined}>
                        {renderInlines(cell)}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          ) : null}
        </table>
      )
  }
}

function renderBlocks(blocks: BlockNode[]): JSX.Element[] {
  return blocks.map((block, idx) => renderBlock(block, idx))
}

/**
 * Shared, safe Markdown presentation for model-authored conversational text.
 *
 * Uses the in-house CommonMark + GFM + KaTeX parser with strict URL allowlisting
 * and zero raw HTML injection.
 */
export function Markdown(props: { children: string; className?: string }): JSX.Element {
  const rootClass = props.className
    ? `markdown-body ${props.className}`.trim()
    : 'markdown-body'
  return (
    <ErrorBoundary
      label="formatted text"
      resetKey={props.children}
      fallback={() => (
        <div className={rootClass}>
          <p className="markdown-plain-fallback">{props.children}</p>
        </div>
      )}
    >
      <MarkdownContent {...props} />
    </ErrorBoundary>
  )
}

function MarkdownContent({
  children,
  className = ''
}: {
  children: string
  className?: string
}): JSX.Element {
  const blocks = useMemo(() => parseMarkdown(children), [children])
  const rootClass = className ? `markdown-body ${className}`.trim() : 'markdown-body'

  return <div className={rootClass}>{renderBlocks(blocks)}</div>
}

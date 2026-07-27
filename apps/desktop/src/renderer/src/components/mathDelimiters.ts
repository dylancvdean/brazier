/**
 * Normalize model-authored math delimiters into the `$` / `$$` forms
 * remark-math understands.
 *
 * Local models often emit `\(...\)` / `\[...\]`, and single-line `$$...$$`
 * for display math. micromark-extension-math only parses dollar delimiters,
 * and treats same-line `$$...$$` as inline — so rewrite those outside code.
 */
export function normalizeMathDelimiters(source: string): string {
  return splitOutsideFencedCode(source)
    .map((part) => (part.code ? part.text : normalizeMathInProse(part.text)))
    .join('')
}

function normalizeMathInProse(text: string): string {
  return splitOutsideInlineCode(text)
    .map((part) => {
      if (part.code) return part.text
      return (
        part.text
          // Display: \[ ... ]
          .replace(/\\\[([\s\S]*?)\\\]/g, (_match, body: string) => `\n$$\n${body.trim()}\n$$\n`)
          // Inline: \( ... )
          .replace(/\\\(([\s\S]*?)\\\)/g, (_match, body: string) => `$${body}$`)
          // Same-line $$...$$ → display fences (multiline $$ is already display)
          .replace(/(^|\n)[^\S\n]*\$\$([^\n]+?)\$\$[^\S\n]*(?=\n|$)/g, (_match, lead: string, body: string) => {
            return `${lead}$$\n${body.trim()}\n$$`
          })
      )
    })
    .join('')
}

function splitOutsideFencedCode(source: string): Array<{ text: string; code: boolean }> {
  const parts: Array<{ text: string; code: boolean }> = []
  const pattern = /(```[\s\S]*?```|~~~[\s\S]*?~~~)/g
  let last = 0
  for (const match of source.matchAll(pattern)) {
    const index = match.index ?? 0
    if (index > last) parts.push({ text: source.slice(last, index), code: false })
    parts.push({ text: match[0], code: true })
    last = index + match[0].length
  }
  if (last < source.length) parts.push({ text: source.slice(last), code: false })
  return parts
}

function splitOutsideInlineCode(source: string): Array<{ text: string; code: boolean }> {
  const parts: Array<{ text: string; code: boolean }> = []
  const pattern = /(`+)([\s\S]*?)\1/g
  let last = 0
  for (const match of source.matchAll(pattern)) {
    const index = match.index ?? 0
    if (index > last) parts.push({ text: source.slice(last, index), code: false })
    parts.push({ text: match[0], code: true })
    last = index + match[0].length
  }
  if (last < source.length) parts.push({ text: source.slice(last), code: false })
  return parts
}

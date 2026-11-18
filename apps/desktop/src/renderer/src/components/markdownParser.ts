export type TableAlign = 'left' | 'center' | 'right' | null

export interface ListItemNode {
  checked: boolean | null
  children: BlockNode[]
  tight: boolean
}

export type BlockNode =
  | { type: 'heading'; depth: 1 | 2 | 3 | 4 | 5 | 6; children: InlineNode[] }
  | { type: 'paragraph'; children: InlineNode[] }
  | { type: 'code_block'; lang: string; code: string }
  | { type: 'math_block'; tex: string }
  | { type: 'hr' }
  | { type: 'blockquote'; children: BlockNode[] }
  | { type: 'list'; ordered: boolean; start: number; items: ListItemNode[] }
  | { type: 'table'; align: TableAlign[]; headers: InlineNode[][]; rows: InlineNode[][][] }

export type InlineNode =
  | { type: 'text'; value: string }
  | { type: 'code'; value: string }
  | { type: 'math_inline'; tex: string }
  | { type: 'br' }
  | { type: 'strong'; children: InlineNode[] }
  | { type: 'em'; children: InlineNode[] }
  | { type: 'del'; children: InlineNode[] }
  | { type: 'link'; href: string; title?: string; children: InlineNode[] }
  | { type: 'image'; src: string; alt: string; title?: string }

const ASCII_PUNCT = new Set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
const SAFE_DATA_IMAGE_RE = /^data:image\/(?:png|jpeg|gif|webp);base64,[a-z0-9+/]+=*$/i

export function sanitizeMarkdownUrl(url: string, kind: 'link' | 'image'): string | null {
  const trimmed = url.trim()
  if (!trimmed || /[\u0000-\u001f\u007f]/.test(trimmed)) return null

  const compact = trimmed.replace(/\s+/g, '')
  if (compact.startsWith('//') || compact.startsWith('\\\\')) return null
  if (compact.startsWith('#')) return trimmed

  const colonIdx = compact.indexOf(':')
  const slashIdx = compact.search(/[/?#]/)
  if (colonIdx === -1 || (slashIdx !== -1 && colonIdx > slashIdx)) return trimmed

  const scheme = compact.slice(0, colonIdx + 1).toLowerCase()
  if (scheme === 'http:' || scheme === 'https:') {
    return /^https?:\/\/\S/i.test(trimmed) ? trimmed : null
  }
  if (kind === 'link' && scheme === 'mailto:') return trimmed.length > 7 ? trimmed : null
  if (kind === 'image') {
    if (scheme === 'blob:') return trimmed.length > 5 ? trimmed : null
    if (scheme === 'data:') return SAFE_DATA_IMAGE_RE.test(compact) ? compact : null
  }
  return null
}

export function parseMarkdown(source: string): BlockNode[] {
  const cleaned = source.replace(/\r\n?/g, '\n').replace(/\n+$/, '')
  if (!cleaned.trim()) return []
  return parseBlocks(cleaned.split('\n'))
}

function unescapeBackslashes(input: string): string {
  let out = ''
  for (let i = 0; i < input.length; i++) {
    if (input[i] === '\\' && i + 1 < input.length && ASCII_PUNCT.has(input[i + 1])) {
      out += input[++i]
    } else {
      out += input[i]
    }
  }
  return out
}

function measureIndent(line: string): { columns: number; chars: number } {
  let columns = 0
  let chars = 0
  while (chars < line.length) {
    if (line[chars] === ' ') {
      columns++
      chars++
    } else if (line[chars] === '\t') {
      columns += 4 - (columns % 4)
      chars++
    } else break
  }
  return { columns, chars }
}

function stripColumns(line: string, maxColumns: number): string {
  let columns = 0
  let i = 0
  while (i < line.length && columns < maxColumns) {
    if (line[i] === ' ') {
      columns++
      i++
    } else if (line[i] === '\t') {
      const step = 4 - (columns % 4)
      if (columns + step > maxColumns) {
        return ' '.repeat(columns + step - maxColumns) + line.slice(i + 1)
      }
      columns += step
      i++
    } else break
  }
  return line.slice(i)
}

function isBlank(line: string): boolean {
  return /^[ \t]*$/.test(line)
}

function isHorizontalRule(line: string): boolean {
  return /^ {0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$/.test(line)
}

interface FenceOpen {
  char: '`' | '~'
  length: number
  indent: number
  lang: string
}

function matchFenceOpen(line: string): FenceOpen | null {
  const match = /^( {0,3})(`{3,}|~{3,})(.*)$/.exec(line)
  if (!match) return null
  const [, spaces, fence, rest] = match
  if (fence[0] === '`' && rest.includes('`')) return null
  const rawLang = rest.trim().split(/\s+/)[0] ?? ''
  return {
    char: fence[0] as '`' | '~',
    length: fence.length,
    indent: spaces.length,
    lang: unescapeBackslashes(rawLang)
  }
}

function matchAtxHeading(line: string): BlockNode | null {
  const match = /^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$/.exec(line)
  if (!match) return null
  const depth = match[1].length as 1 | 2 | 3 | 4 | 5 | 6
  const raw = (match[2] ?? '').replace(/^(#+)$/, '').replace(/[ \t]+#+$/, '')
  return { type: 'heading', depth, children: parseInlines(raw.trim()) }
}

function scanDelimitedMath(
  lines: string[],
  start: number,
  openToken: string,
  closeToken: string
): { node: BlockNode; next: number } | null {
  const trimmedStart = lines[start].replace(/^ {0,3}/, '')
  if (lines[start].length - trimmedStart.length > 3 || !trimmedStart.startsWith(openToken)) {
    return null
  }
  const rest = trimmedStart.slice(openToken.length)
  const trimmedRest = rest.trimEnd()
  if (trimmedRest.endsWith(closeToken)) {
    const inner = trimmedRest.slice(0, -closeToken.length)
    if (inner.trim() && !inner.includes(closeToken)) {
      return { node: { type: 'math_block', tex: inner.trim() }, next: start + 1 }
    }
    if (trimmedRest.length > closeToken.length) return null
  }
  if (rest.includes(closeToken)) return null

  const buf: string[] = rest.trim() ? [rest] : []
  for (let j = start + 1; j < lines.length; j++) {
    const t = lines[j].trimEnd()
    if (t.replace(/^ {0,3}/, '') === closeToken) {
      return { node: { type: 'math_block', tex: buf.join('\n').trim() }, next: j + 1 }
    }
    if (t.endsWith(closeToken)) {
      const head = t.slice(0, -closeToken.length)
      if (!head.includes(closeToken)) {
        if (head.trim()) buf.push(head)
        return { node: { type: 'math_block', tex: buf.join('\n').trim() }, next: j + 1 }
      }
    }
    buf.push(lines[j])
  }
  return { node: { type: 'math_block', tex: buf.join('\n').trim() }, next: lines.length }
}

function parseMathBlockAt(lines: string[], start: number): { node: BlockNode; next: number } | null {
  return scanDelimitedMath(lines, start, '$$', '$$') ?? scanDelimitedMath(lines, start, '\\[', '\\]')
}

interface ListMarker {
  ordered: boolean
  markerKey: string
  start: number
  baseIndent: number
  contentIndent: number
  rest: string
}

function matchListMarker(line: string, maxIndent = 3): ListMarker | null {
  if (isHorizontalRule(line)) return null
  const { columns: baseIndent, chars } = measureIndent(line)
  if (baseIndent > maxIndent) return null
  const tail = line.slice(chars)

  const ul = /^([-*+])(?:([ \t]+)(.*)|$)/.exec(tail)
  if (ul) {
    const [, bullet, ws = '', rest = ''] = ul
    const wsCols = ws ? measureIndent(ws).columns : 1
    return {
      ordered: false,
      markerKey: bullet,
      start: 1,
      baseIndent,
      contentIndent: baseIndent + 1 + (rest && wsCols <= 4 ? wsCols : 1),
      rest
    }
  }

  const ol = /^(\d{1,9})([.)])(?:([ \t]+)(.*)|$)/.exec(tail)
  if (ol) {
    const [, digits, delim, ws = '', rest = ''] = ol
    const wsCols = ws ? measureIndent(ws).columns : 1
    return {
      ordered: true,
      markerKey: delim,
      start: Number.parseInt(digits, 10),
      baseIndent,
      contentIndent: baseIndent + digits.length + 1 + (rest && wsCols <= 4 ? wsCols : 1),
      rest
    }
  }
  return null
}

function findBacktickSpan(text: string, openIdx: number): { ticks: number; closeIdx: number } {
  let ticks = 1
  while (text[openIdx + ticks] === '`') ticks++
  let pos = openIdx + ticks
  while (pos < text.length) {
    if (text[pos] === '`') {
      let run = 1
      while (text[pos + run] === '`') run++
      if (run === ticks) return { ticks, closeIdx: pos }
      pos += run
    } else {
      pos++
    }
  }
  return { ticks, closeIdx: -1 }
}

function findInlineDollarClose(text: string, openIdx: number): number {
  if (openIdx + 1 >= text.length || text[openIdx + 1] === '$') return -1
  const prev = openIdx > 0 ? text[openIdx - 1] : ''
  if ((prev && /[\p{L}\p{N}$]/u.test(prev)) || /\s/.test(text[openIdx + 1])) return -1

  for (let j = openIdx + 1; j < text.length; j++) {
    const ch = text[j]
    if (ch === '\\') {
      j++
      continue
    }
    if (ch === '`') return -1
    if (ch === '$') {
      if (text[j + 1] === '$' || /\s/.test(text[j - 1])) return -1
      const afterClose = text[j + 1] ?? ''
      const inner = text.slice(openIdx + 1, j)
      if (/\d/.test(afterClose) && !/[\\^_{}=]/.test(inner)) return -1
      if (/[\p{L}]/u.test(afterClose) && /^\d/.test(inner) && !/[\\^_{}=+\-*<>]/.test(inner)) {
        return -1
      }
      return j
    }
  }
  return -1
}

function parseTableRowCells(rawLine: string): string[] | null {
  let line = rawLine.trim()
  let pipes = 0
  if (line.startsWith('|')) {
    line = line.slice(1)
    pipes++
  }
  if (line.endsWith('|')) {
    let slashes = 0
    for (let k = line.length - 2; k >= 0 && line[k] === '\\'; k--) slashes++
    if (slashes % 2 === 0) {
      line = line.slice(0, -1)
      pipes++
    }
  }

  const cells: string[] = []
  let cur = ''
  let i = 0
  while (i < line.length) {
    const ch = line[i]
    if (ch === '\\' && line[i + 1] === '(') {
      const close = line.indexOf('\\)', i + 2)
      if (close !== -1) {
        cur += line.slice(i, close + 2)
        i = close + 2
        continue
      }
    }
    if (ch === '\\' && i + 1 < line.length) {
      cur += ch + line[i + 1]
      i += 2
      continue
    }
    if (ch === '`') {
      const { ticks, closeIdx } = findBacktickSpan(line, i)
      const end = closeIdx !== -1 ? closeIdx + ticks : i + ticks
      cur += line.slice(i, end)
      i = end
      continue
    }
    if (ch === '$') {
      const close =
        line[i + 1] === '$'
          ? line.indexOf('$$', i + 2) !== -1
            ? line.indexOf('$$', i + 2) + 1
            : -1
          : findInlineDollarClose(line, i)
      if (close !== -1) {
        cur += line.slice(i, close + 1)
        i = close + 1
        continue
      }
    }
    if (ch === '|') {
      pipes++
      cells.push(cur.trim())
      cur = ''
      i++
      continue
    }
    cur += ch
    i++
  }
  if (pipes === 0) return null
  cells.push(cur.trim())
  return cells
}

function parseTableAt(lines: string[], start: number): { node: BlockNode; next: number } | null {
  if (start + 1 >= lines.length) return null
  const headerRaw = parseTableRowCells(lines[start])
  const delimRaw = parseTableRowCells(lines[start + 1])
  if (!headerRaw || !delimRaw || headerRaw.length === 0 || headerRaw.length !== delimRaw.length) {
    return null
  }
  const align: TableAlign[] = []
  for (const cell of delimRaw) {
    if (!/^:?-+:?$/.test(cell)) return null
    const l = cell.startsWith(':')
    const r = cell.endsWith(':')
    align.push(l && r ? 'center' : l ? 'left' : r ? 'right' : null)
  }

  const colCount = align.length
  const headers = headerRaw.map((c) => parseInlines(c))
  const rows: InlineNode[][][] = []
  let j = start + 2
  while (j < lines.length) {
    const rowLine = lines[j]
    if (
      isBlank(rowLine) ||
      matchFenceOpen(rowLine) ||
      parseMathBlockAt(lines, j) ||
      matchAtxHeading(rowLine) ||
      isHorizontalRule(rowLine) ||
      /^ {0,3}>/.test(rowLine) ||
      matchListMarker(rowLine)
    ) {
      break
    }
    const rawCells = parseTableRowCells(rowLine)
    if (!rawCells) break
    rows.push(
      Array.from({ length: colCount }, (_, c) => (c < rawCells.length ? parseInlines(rawCells[c]) : []))
    )
    j++
  }
  return { node: { type: 'table', align, headers, rows }, next: j }
}

function canInterruptParagraph(lines: string[], idx: number): boolean {
  const line = lines[idx]
  if (
    isBlank(line) ||
    matchFenceOpen(line) ||
    parseMathBlockAt(lines, idx) ||
    matchAtxHeading(line) ||
    isHorizontalRule(line) ||
    /^ {0,3}>/.test(line) ||
    parseTableAt(lines, idx)
  ) {
    return true
  }
  const marker = matchListMarker(line)
  return Boolean(marker && marker.rest.trim() && (!marker.ordered || marker.start === 1))
}

function parseListAt(lines: string[], start: number): { node: BlockNode; next: number } {
  const first = matchListMarker(lines[start])!
  const itemsRaw: string[][] = []
  let loose = false
  let i = start

  while (i < lines.length) {
    const marker = matchListMarker(lines[i])
    if (
      !marker ||
      marker.ordered !== first.ordered ||
      marker.markerKey !== first.markerKey ||
      marker.baseIndent > first.baseIndent + 1
    ) {
      break
    }

    const itemLines: string[] = [marker.rest]
    let hadBlankInItem = false
    let inFence: FenceOpen | null = matchFenceOpen(marker.rest)
    i++

    while (i < lines.length) {
      const cur = lines[i]
      if (isBlank(cur)) {
        let k = i + 1
        while (k < lines.length && isBlank(lines[k])) k++
        if (k >= lines.length) {
          i = k
          break
        }
        const { columns: nextIndent } = measureIndent(lines[k])
        const subMarker = matchListMarker(lines[k], 99)
        if (nextIndent >= marker.contentIndent || (subMarker && nextIndent >= marker.baseIndent + 2)) {
          while (i < k) {
            itemLines.push('')
            i++
          }
          hadBlankInItem = true
          continue
        }
        const sib = matchListMarker(lines[k])
        if (
          sib &&
          sib.ordered === first.ordered &&
          sib.markerKey === first.markerKey &&
          sib.baseIndent <= first.baseIndent + 1
        ) {
          loose = true
        }
        i = k
        break
      }

      const { columns: curIndent } = measureIndent(cur)
      const subMarker = matchListMarker(cur, 99)
      if (curIndent >= marker.contentIndent) {
        if (hadBlankInItem) loose = true
        const stripped = stripColumns(cur, marker.contentIndent)
        if (!inFence) inFence = matchFenceOpen(stripped)
        else if (new RegExp(`^ {0,3}${inFence.char}{${inFence.length},}[ \\t]*$`).test(stripped)) {
          inFence = null
        }
        itemLines.push(stripped)
        i++
        continue
      }
      if (!inFence && subMarker && curIndent >= marker.baseIndent + 2) {
        if (hadBlankInItem) loose = true
        itemLines.push(stripColumns(cur, curIndent))
        i++
        continue
      }
      if (!inFence && !hadBlankInItem && !canInterruptParagraph(lines, i) && !matchListMarker(cur)) {
        itemLines.push(cur.trimStart())
        i++
        continue
      }
      break
    }
    itemsRaw.push(itemLines)
  }

  const items: ListItemNode[] = itemsRaw.map((rawLines) => {
    const copy = [...rawLines]
    const taskMatch = /^\[([ xX])\](?:[ \t]+(.*)|$)/.exec(copy[0] ?? '')
    let checked: boolean | null = null
    if (taskMatch) {
      checked = taskMatch[1].toLowerCase() === 'x'
      copy[0] = taskMatch[2] ?? ''
    }
    return { checked, children: parseBlocks(copy), tight: !loose }
  })

  return { node: { type: 'list', ordered: first.ordered, start: first.start, items }, next: i }
}

function parseBlocks(lines: string[]): BlockNode[] {
  const blocks: BlockNode[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]
    if (isBlank(line)) {
      i++
      continue
    }

    const fence = matchFenceOpen(line)
    if (fence) {
      const codeLines: string[] = []
      const closeRe = new RegExp(`^ {0,3}${fence.char}{${fence.length},}[ \\t]*$`)
      i++
      while (i < lines.length && !closeRe.test(lines[i])) {
        codeLines.push(fence.indent > 0 ? stripColumns(lines[i], fence.indent) : lines[i])
        i++
      }
      if (i < lines.length) i++
      const body = codeLines.join('\n')
      blocks.push(
        fence.lang.toLowerCase() === 'math'
          ? { type: 'math_block', tex: body.trim() }
          : { type: 'code_block', lang: fence.lang, code: body }
      )
      continue
    }

    const mathBlock = parseMathBlockAt(lines, i)
    if (mathBlock) {
      blocks.push(mathBlock.node)
      i = mathBlock.next
      continue
    }

    const heading = matchAtxHeading(line)
    if (heading) {
      blocks.push(heading)
      i++
      continue
    }

    if (isHorizontalRule(line)) {
      blocks.push({ type: 'hr' })
      i++
      continue
    }

    if (/^ {0,3}>/.test(line)) {
      const quoteLines: string[] = []
      while (i < lines.length && /^ {0,3}>/.test(lines[i])) {
        quoteLines.push(lines[i++].replace(/^ {0,3}> ?/, ''))
      }
      blocks.push({ type: 'blockquote', children: parseBlocks(quoteLines) })
      continue
    }

    const table = parseTableAt(lines, i)
    if (table) {
      blocks.push(table.node)
      i = table.next
      continue
    }

    if (matchListMarker(line)) {
      const list = parseListAt(lines, i)
      blocks.push(list.node)
      i = list.next
      continue
    }

    const paraLines: string[] = [line]
    i++
    let setextDepth: 1 | 2 | null = null
    while (i < lines.length) {
      const setext = /^ {0,3}(={2,}|-{2,})[ \t]*$/.exec(lines[i])
      if (setext) {
        setextDepth = setext[1][0] === '=' ? 1 : 2
        i++
        break
      }
      if (canInterruptParagraph(lines, i)) break
      paraLines.push(lines[i++])
    }

    const rawPara = paraLines.join('\n').trim()
    if (setextDepth) {
      blocks.push({ type: 'heading', depth: setextDepth, children: parseInlines(rawPara) })
    } else if (rawPara) {
      blocks.push({ type: 'paragraph', children: parseInlines(rawPara) })
    }
  }
  return blocks
}

type Token =
  | { kind: 'node'; node: InlineNode }
  | {
      kind: 'delim'
      char: '*' | '_' | '~'
      origCount: number
      count: number
      canOpen: boolean
      canClose: boolean
    }

function computeFlanking(
  text: string,
  start: number,
  end: number,
  ch: '*' | '_' | '~'
): { canOpen: boolean; canClose: boolean } {
  const before = start > 0 ? text[start - 1] : '\n'
  const after = end < text.length ? text[end] : '\n'
  const beforeWs = /\s/.test(before)
  const afterWs = /\s/.test(after)
  const beforePunct = /[\p{P}\p{S}]/u.test(before)
  const afterPunct = /[\p{P}\p{S}]/u.test(after)
  const left = !afterWs && (!afterPunct || beforeWs || beforePunct)
  const right = !beforeWs && (!beforePunct || afterWs || afterPunct)
  if (ch === '_') {
    return { canOpen: left && (!right || beforePunct), canClose: right && (!left || afterPunct) }
  }
  return { canOpen: left, canClose: right }
}

function findClosingBracket(text: string, openIdx: number): number {
  let depth = 1
  for (let i = openIdx + 1; i < text.length; i++) {
    if (text[i] === '\\') {
      i++
      continue
    }
    if (text[i] === '`') {
      const { ticks, closeIdx } = findBacktickSpan(text, i)
      i = (closeIdx !== -1 ? closeIdx + ticks : i + ticks) - 1
      continue
    }
    if (text[i] === '[') depth++
    else if (text[i] === ']' && --depth === 0) return i
  }
  return -1
}

function parseLinkTarget(
  text: string,
  openParenIdx: number
): { url: string; title?: string; end: number } | null {
  if (text[openParenIdx] !== '(') return null
  let i = openParenIdx + 1
  while (i < text.length && /[ \t\n]/.test(text[i])) i++
  if (i >= text.length) return null

  let url = ''
  if (text[i] === '<') {
    i++
    while (i < text.length && text[i] !== '>' && text[i] !== '\n') {
      url += text[i] === '\\' && i + 1 < text.length ? text[++i] : text[i]
      i++
    }
    if (text[i] !== '>') return null
    i++
  } else {
    let parenDepth = 0
    while (i < text.length && !/\s/.test(text[i])) {
      const ch = text[i]
      if (ch === '\\' && i + 1 < text.length && ASCII_PUNCT.has(text[i + 1])) {
        url += text[++i]
        i++
        continue
      }
      if (ch === '(') parenDepth++
      else if (ch === ')') {
        if (parenDepth === 0) break
        parenDepth--
      }
      url += ch
      i++
    }
    if (parenDepth !== 0) return null
  }

  const wsStart = i
  while (i < text.length && /[ \t\n]/.test(text[i])) i++
  if (text[i] === ')') return { url, end: i + 1 }
  if (i === wsStart || i >= text.length) return null

  const quote = text[i]
  if (quote !== '"' && quote !== "'" && quote !== '(') return null
  const closeQuote = quote === '(' ? ')' : quote
  i++
  let title = ''
  while (i < text.length && text[i] !== closeQuote) {
    title += text[i] === '\\' && i + 1 < text.length && ASCII_PUNCT.has(text[i + 1]) ? text[++i] : text[i]
    i++
  }
  if (text[i] !== closeQuote) return null
  i++
  while (i < text.length && /[ \t\n]/.test(text[i])) i++
  return text[i] === ')' ? { url, title, end: i + 1 } : null
}

function extractPlainText(nodes: InlineNode[]): string {
  return nodes
    .map((n) =>
      n.type === 'text' || n.type === 'code'
        ? n.value
        : n.type === 'math_inline'
          ? n.tex
          : n.type === 'br'
            ? ' '
            : n.type === 'image'
              ? n.alt
              : extractPlainText(n.children)
    )
    .join('')
}

function containsLink(nodes: InlineNode[]): boolean {
  return nodes.some((n) => n.type === 'link' || ('children' in n && containsLink(n.children)))
}

export function parseInlines(text: string): InlineNode[] {
  const tokens: Token[] = []
  const pushText = (value: string): void => {
    if (value) tokens.push({ kind: 'node', node: { type: 'text', value } })
  }

  let i = 0
  while (i < text.length) {
    const ch = text[i]

    if (ch === ' ' && text[i + 1] === ' ') {
      let sp = i
      while (text[sp] === ' ') sp++
      if (text[sp] === '\n') {
        tokens.push({ kind: 'node', node: { type: 'br' } })
        i = sp + 1
        while (text[i] === ' ' || text[i] === '\t') i++
        continue
      }
    }
    if (ch === '\n') {
      const prev = tokens[tokens.length - 1]
      if (prev?.kind === 'node' && prev.node.type === 'text') {
        prev.node.value = prev.node.value.replace(/ +$/, '')
      }
      pushText('\n')
      i++
      while (text[i] === ' ' || text[i] === '\t') i++
      continue
    }

    if (ch === '\\') {
      const next = text[i + 1]
      if (next === '\n') {
        tokens.push({ kind: 'node', node: { type: 'br' } })
        i += 2
        while (text[i] === ' ' || text[i] === '\t') i++
        continue
      }
      if (next === '(' || next === '[') {
        const closeMark = next === '(' ? '\\)' : '\\]'
        const close = text.indexOf(closeMark, i + 2)
        if (close !== -1) {
          const body = text.slice(i + 2, close).trim()
          tokens.push({
            kind: 'node',
            node: { type: 'math_inline', tex: next === '[' ? `\\displaystyle ${body}` : body }
          })
          i = close + 2
          continue
        }
      }
      if (next && ASCII_PUNCT.has(next)) {
        pushText(next)
        i += 2
      } else {
        pushText('\\')
        i++
      }
      continue
    }

    if (ch === '`') {
      const { ticks, closeIdx } = findBacktickSpan(text, i)
      if (closeIdx !== -1) {
        let code = text.slice(i + ticks, closeIdx).replace(/\n/g, ' ')
        if (code.length >= 2 && code.startsWith(' ') && code.endsWith(' ') && /[^ ]/.test(code)) {
          code = code.slice(1, -1)
        }
        tokens.push({ kind: 'node', node: { type: 'code', value: code } })
        i = closeIdx + ticks
      } else {
        pushText('`'.repeat(ticks))
        i += ticks
      }
      continue
    }

    if (ch === '$') {
      if (text[i + 1] === '$' && text[i + 2] !== '$') {
        const close = text.indexOf('$$', i + 2)
        if (close > i + 2) {
          tokens.push({
            kind: 'node',
            node: { type: 'math_inline', tex: text.slice(i + 2, close).trim() }
          })
          i = close + 2
          continue
        }
      } else {
        const close = findInlineDollarClose(text, i)
        if (close !== -1) {
          tokens.push({ kind: 'node', node: { type: 'math_inline', tex: text.slice(i + 1, close) } })
          i = close + 1
          continue
        }
      }
      let run = 1
      while (text[i + run] === '$') run++
      pushText('$'.repeat(run))
      i += run
      continue
    }

    if ((ch === '!' && text[i + 1] === '[') || ch === '[') {
      const isImage = ch === '!'
      const openBracket = isImage ? i + 1 : i
      const closeBracket = findClosingBracket(text, openBracket)
      if (closeBracket !== -1 && text[closeBracket + 1] === '(') {
        const target = parseLinkTarget(text, closeBracket + 1)
        if (target) {
          const innerNodes = parseInlines(text.slice(openBracket + 1, closeBracket))
          if (isImage) {
            const alt = extractPlainText(innerNodes)
            const safeSrc = sanitizeMarkdownUrl(target.url, 'image')
            if (safeSrc) {
              tokens.push({
                kind: 'node',
                node: { type: 'image', src: safeSrc, alt, title: target.title }
              })
            } else if (alt) pushText(alt)
            i = target.end
            continue
          } else if (!containsLink(innerNodes)) {
            const safeHref = sanitizeMarkdownUrl(target.url, 'link')
            if (safeHref) {
              tokens.push({
                kind: 'node',
                node: { type: 'link', href: safeHref, title: target.title, children: innerNodes }
              })
            } else {
              for (const child of innerNodes) tokens.push({ kind: 'node', node: child })
            }
            i = target.end
            continue
          }
        }
      }
      pushText(isImage ? '![' : '[')
      i += isImage ? 2 : 1
      continue
    }

    if (ch === '<') {
      const autoUrl = /^<((?:https?:\/\/|mailto:)[^\s<>]+)>/i.exec(text.slice(i))
      if (autoUrl) {
        const safeHref = sanitizeMarkdownUrl(autoUrl[1], 'link')
        if (safeHref) {
          const label = autoUrl[1].replace(/^mailto:/i, '')
          tokens.push({
            kind: 'node',
            node: { type: 'link', href: safeHref, children: [{ type: 'text', value: label }] }
          })
          i += autoUrl[0].length
          continue
        }
      }
      const autoEmail =
        /^<([a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+)>/.exec(
          text.slice(i)
        )
      if (autoEmail) {
        tokens.push({
          kind: 'node',
          node: {
            type: 'link',
            href: `mailto:${autoEmail[1]}`,
            children: [{ type: 'text', value: autoEmail[1] }]
          }
        })
        i += autoEmail[0].length
        continue
      }
      pushText('<')
      i++
      continue
    }

    if (
      (ch === 'h' || ch === 'H') &&
      /^https?:\/\/[^\s<]/i.test(text.slice(i)) &&
      (i === 0 || !/[\p{L}\p{N}_/.:]/u.test(text[i - 1]))
    ) {
      const match = /^https?:\/\/[^\s<]+/i.exec(text.slice(i))
      if (match) {
        let raw = match[0]
        const openChars = new Set(
          tokens.filter((t): t is Extract<Token, { kind: 'delim' }> => t.kind === 'delim' && t.canOpen).map((t) => t.char)
        )
        while (raw.length > 8) {
          const last = raw[raw.length - 1]
          if (
            '.,;:!?\'"'.includes(last) ||
            ((last === '*' || last === '_' || last === '~') && openChars.has(last))
          ) {
            raw = raw.slice(0, -1)
            continue
          }
          if (last === ')') {
            let balance = 0
            for (const c of raw) balance += c === '(' ? 1 : c === ')' ? -1 : 0
            if (balance < 0) {
              raw = raw.slice(0, -1)
              continue
            }
          }
          break
        }
        const safeHref = sanitizeMarkdownUrl(raw, 'link')
        if (safeHref) {
          tokens.push({
            kind: 'node',
            node: { type: 'link', href: safeHref, children: [{ type: 'text', value: raw }] }
          })
          i += raw.length
          continue
        }
      }
    }

    if (ch === '*' || ch === '_' || ch === '~') {
      let run = 1
      while (text[i + run] === ch) run++
      if (ch === '~' && run !== 2) {
        pushText('~'.repeat(run))
      } else {
        const { canOpen, canClose } = computeFlanking(text, i, i + run, ch)
        if (!canOpen && !canClose) pushText(ch.repeat(run))
        else tokens.push({ kind: 'delim', char: ch, origCount: run, count: run, canOpen, canClose })
      }
      i += run
      continue
    }

    const start = i++
    while (i < text.length) {
      const c = text[i]
      if (
        '\\`$![<*_~\n'.includes(c) ||
        (c === ' ' && text[i + 1] === ' ') ||
        ((c === 'h' || c === 'H') && (text[i + 1] === 't' || text[i + 1] === 'T'))
      ) {
        break
      }
      i++
    }
    pushText(text.slice(start, i))
  }

  return resolveEmphasisAndMerge(tokens)
}

function resolveEmphasisAndMerge(tokens: Token[]): InlineNode[] {
  let cIdx = 0
  while (cIdx < tokens.length) {
    const closer = tokens[cIdx]
    if (closer.kind !== 'delim' || !closer.canClose || closer.count === 0) {
      cIdx++
      continue
    }
    let oIdx = -1
    for (let k = cIdx - 1; k >= 0; k--) {
      const cand = tokens[k]
      if (cand.kind !== 'delim' || !cand.canOpen || cand.count === 0 || cand.char !== closer.char) {
        continue
      }
      if (
        cand.char !== '~' &&
        ((cand.canOpen && cand.canClose) || (closer.canOpen && closer.canClose)) &&
        (cand.origCount + closer.origCount) % 3 === 0 &&
        !(cand.origCount % 3 === 0 && closer.origCount % 3 === 0)
      ) {
        continue
      }
      oIdx = k
      break
    }
    if (oIdx === -1) {
      cIdx++
      continue
    }
    const opener = tokens[oIdx] as Extract<Token, { kind: 'delim' }>
    const use = opener.char === '~' ? 2 : opener.count >= 2 && closer.count >= 2 ? 2 : 1
    opener.count -= use
    closer.count -= use

    const innerNodes = tokensToNodes(tokens.splice(oIdx + 1, cIdx - oIdx - 1))
    const wrapped: InlineNode =
      opener.char === '~'
        ? { type: 'del', children: innerNodes }
        : use === 2
          ? { type: 'strong', children: innerNodes }
          : { type: 'em', children: innerNodes }

    tokens.splice(oIdx + 1, 0, { kind: 'node', node: wrapped })
    cIdx = oIdx + 2 + (closer.count === 0 ? 1 : 0)
  }
  return tokensToNodes(tokens)
}

function tokensToNodes(tokens: Token[]): InlineNode[] {
  const out: InlineNode[] = []
  const appendText = (val: string): void => {
    if (!val) return
    const prev = out[out.length - 1]
    if (prev?.type === 'text') prev.value += val
    else out.push({ type: 'text', value: val })
  }
  for (const tok of tokens) {
    if (tok.kind === 'delim') {
      if (tok.count > 0) appendText(tok.char.repeat(tok.count))
    } else if (tok.node.type === 'text') {
      appendText(tok.node.value)
    } else {
      out.push(tok.node)
    }
  }
  return out
}

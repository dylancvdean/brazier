export type Inline =
  | { type: 'text'; text: string }
  | { type: 'code'; text: string }
  | { type: 'strong' | 'em'; children: Inline[] }
  | { type: 'link'; href: string; children: Inline[] };

export type Block =
  | { type: 'paragraph' | 'quote'; children: Inline[] }
  | { type: 'heading'; level: number; children: Inline[] }
  | { type: 'item'; marker: string; depth: number; children: Inline[] }
  | { type: 'code'; language: string; text: string }
  | { type: 'rule' };

const inlinePattern = /(`+)([\s\S]+?)\1|\*\*([\s\S]+?)\*\*|__([\s\S]+?)__|\*([^*\s][^*]*?)\*|\b_([^_\s][^_]*?)_\b|\[([^\]]+)\]\(((?:[^()\s]|\([^()\s]*\))+)\)/;

export function parseInline(source: string): Inline[] {
  const out: Inline[] = [];
  let rest = source;
  while (rest) {
    const match = inlinePattern.exec(rest);
    if (!match) break;
    if (match.index) out.push({ type: 'text', text: rest.slice(0, match.index) });
    if (match[1]) out.push({ type: 'code', text: match[2].trim() });
    else if (match[3] ?? match[4]) out.push({ type: 'strong', children: parseInline(match[3] ?? match[4]) });
    else if (match[5] ?? match[6]) out.push({ type: 'em', children: parseInline(match[5] ?? match[6]) });
    else if (/^(https?:|mailto:)/i.test(match[8])) out.push({ type: 'link', href: match[8], children: parseInline(match[7]) });
    else out.push({ type: 'text', text: match[0] });
    rest = rest.slice(match.index + match[0].length);
  }
  if (rest) out.push({ type: 'text', text: rest });
  return out;
}

/** The subset of Markdown that chat models write. An unclosed code fence, as in a reply still streaming, runs to the end. */
export function parseMarkdown(source: string): Block[] {
  const blocks: Block[] = [];
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  let paragraph: string[] = [];
  const flush = () => {
    if (paragraph.length) blocks.push({ type: 'paragraph', children: parseInline(paragraph.join('\n')) });
    paragraph = [];
  };
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const fence = /^\s*(`{3,}|~{3,})\s*([^\s`]*)/.exec(line);
    if (fence) {
      flush();
      // Only a bare fence at least as long as the opener closes the block.
      const closes = (text: string) => text.trim().startsWith(fence[1]) && /^(`+|~+)$/.test(text.trim());
      const body: string[] = [];
      for (i += 1; i < lines.length && !closes(lines[i]); i += 1) body.push(lines[i]);
      blocks.push({ type: 'code', language: fence[2], text: body.join('\n') });
      continue;
    }
    const heading = /^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$/.exec(line);
    const item = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/.exec(line);
    const quote = /^\s*>\s?(.*)$/.exec(line);
    if (!line.trim()) flush();
    else if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); blocks.push({ type: 'rule' }); }
    else if (heading) { flush(); blocks.push({ type: 'heading', level: heading[1].length, children: parseInline(heading[2]) }); }
    else if (item) {
      flush();
      blocks.push({
        type: 'item', marker: /\d/.test(item[2]) ? item[2].replace(')', '.') : '•',
        depth: Math.floor(item[1].replace(/\t/g, '  ').length / 2), children: parseInline(item[3]),
      });
    } else if (quote) { flush(); blocks.push({ type: 'quote', children: parseInline(quote[1]) }); }
    else paragraph.push(line);
  }
  flush();
  return blocks;
}

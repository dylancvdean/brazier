/**
 * Tools that run on the phone for chats on this device: no host is involved. They carry the names of the
 * host's own tools, so a conversation shows them the same way.
 */
import type { Tool } from '../api.ts';

export const FETCH_TIMEOUT_MS = 15_000;
export const FETCH_MAX_BYTES = 512 * 1024;
/** A phone model has a small context, so a page is cut much shorter than a host would cut it. */
export const FETCH_MAX_CHARS = 3_000;
const SEARCH_RESULTS = 5;
const SNIPPET_CHARS = 240;
// Like the host's web tools, requests say who they are from instead of posing as a browser.
const USER_AGENT = 'Brazier-Mobile (+https://github.com/dylancvdean/brazier; FOSS mobile client; web tools)';

export const LOCAL_TOOLS: readonly Tool[] = [
  {
    name: 'web_search', title: 'Web search', network: true, source: 'device',
    description: 'Searches DuckDuckGo from this phone. The query leaves the device; no account or key is used.',
  },
  {
    name: 'fetch_url', title: 'Web fetch', network: true, source: 'device',
    description: `Reads the text of a public web page from this phone (max ${FETCH_MAX_BYTES / 1024} KB, ${FETCH_TIMEOUT_MS / 1000}s). Local and private addresses are blocked.`,
  },
  {
    name: 'get_current_time', title: 'Current time', network: false, source: 'device',
    description: 'The date and time on this phone.',
  },
];

const DEFINITIONS: Record<string, { description: string; parameters: object }> = {
  web_search: {
    description: 'Search the web and get a short ranked list of results with titles, URLs and snippets. Use it for recent facts or anything you are unsure of.',
    parameters: { type: 'object', properties: { query: { type: 'string', description: 'What to search for.' } }, required: ['query'] },
  },
  fetch_url: {
    description: 'Fetch a public web page and return its readable text. Use it to read a search result.',
    parameters: { type: 'object', properties: { url: { type: 'string', description: 'An http or https URL.' } }, required: ['url'] },
  },
  get_current_time: {
    description: 'Get the current date and time.',
    parameters: { type: 'object', properties: {} },
  },
};

/** OpenAI-style function definitions for the named tools, in catalog order. */
export function toolDefinitions(names: readonly string[]): object[] {
  return LOCAL_TOOLS.filter(({ name }) => names.includes(name) && DEFINITIONS[name])
    .map(({ name }) => ({ type: 'function', function: { name, ...DEFINITIONS[name] } }));
}

/** A failure whose message is meant for the model and the user as written. */
class ToolError extends Error {}

function ipv4(host: string): number[] | null {
  const parts = host.split('.');
  if (parts.length !== 4 || !parts.every((part) => /^\d{1,3}$/.test(part))) return null;
  const bytes = parts.map(Number);
  return bytes.every((byte) => byte <= 255) ? bytes : null;
}

function privateIpv4([a, b]: number[]): boolean {
  return a === 0 || a === 10 || a === 127 || (a === 100 && b >= 64 && b <= 127) || (a === 169 && b === 254) ||
    (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 192 && b === 0) ||
    (a === 198 && (b === 18 || b === 19)) || a >= 224;
}

/**
 * A URL a model may fetch: http(s) to a public name. Loopback, private, link-local and other reserved
 * addresses, names that only resolve on a local network, and URLs carrying credentials are refused.
 * A public name that resolves to a private address cannot be seen from JavaScript and is not caught.
 */
export function safeUrl(input: string): URL {
  let url: URL;
  try { url = new URL(input.trim()); } catch { throw new ToolError('That is not a valid URL.'); }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') throw new ToolError('Only http and https URLs can be fetched.');
  if (url.username || url.password) throw new ToolError('URLs with a user name or password are not fetched.');
  const host = url.hostname.toLowerCase().replace(/\.$/, '');
  const blocked = new ToolError('Local and private addresses are not fetched.');
  if (host.startsWith('[') || host.includes(':')) {
    const address = host.replace(/^\[|\]$/g, '');
    const mapped = /^::ffff:(\d+\.\d+\.\d+\.\d+)$/.exec(address)?.[1] ??
      (/^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/.exec(address) ?? []).slice(1)
        .map((group) => parseInt(group, 16)).flatMap((group) => [group >> 8, group & 255]).join('.');
    const bytes = mapped ? ipv4(mapped) : null;
    if (bytes ? privateIpv4(bytes) : !/^[23][0-9a-f]{3}:/.test(address)) throw blocked;
    return url;
  }
  const bytes = ipv4(host);
  if (bytes) {
    if (privateIpv4(bytes)) throw blocked;
    return url;
  }
  // Decimal, hex and short forms of an IPv4 address (`2130706433`, `0x7f.1`) are normalised by URL parsing
  // on most engines; anything still made only of digits, dots and hex markers is not a public name.
  if (!host.includes('.') || /^[0-9a-fx.]+$/.test(host) || host === 'localhost' ||
      /\.(local|localhost|internal|lan|home|intranet|corp|home\.arpa)$/.test(host)) throw blocked;
  return url;
}

const ENTITIES: Record<string, string> = {
  amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', ndash: '–', mdash: '—', hellip: '…', copy: '©',
  deg: '°', rsquo: '’', lsquo: '‘', rdquo: '”', ldquo: '“',
};

export function decodeEntities(text: string): string {
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (whole, name: string) => {
    if (name[0] !== '#') return ENTITIES[name.toLowerCase()] ?? whole;
    const code = name[1].toLowerCase() === 'x' ? parseInt(name.slice(2), 16) : parseInt(name.slice(1), 10);
    return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : whole;
  });
}

const plain = (html: string) => decodeEntities(html.replace(/<[^>]*>/g, '')).replace(/\s+/g, ' ').trim();

/** The title and readable text of an HTML page: the article or main part when the page marks one. */
export function readableText(html: string): { title: string; text: string } {
  const title = plain(/<title[^>]*>([\s\S]*?)<\/title>/i.exec(html)?.[1] ?? '');
  let body = html.replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<(script|style|noscript|template|svg|iframe|head|title|form|button|select)\b[\s\S]*?<\/\1\s*>/gi, ' ');
  const main = /<(article|main)\b[^>]*>([\s\S]*)<\/\1\s*>/i.exec(body)?.[2];
  // A page whose marked part is tiny (a teaser card) is read whole.
  if (main && plain(main).length >= 200) body = main;
  else body = body.replace(/<(nav|footer|aside|header)\b[\s\S]*?<\/\1\s*>/gi, ' ');
  const text = decodeEntities(body
    .replace(/<(br|hr)\b[^>]*>/gi, '\n')
    .replace(/<li\b[^>]*>/gi, '\n- ')
    .replace(/<\/?(p|div|section|article|main|h[1-6]|ul|ol|table|tr|blockquote|pre|figure|figcaption|dd|dt)\b[^>]*>/gi, '\n')
    .replace(/<[^>]*>/g, ''))
    .replace(/[ \t\r\f\v\u00a0]+/g, ' ').replace(/ ?\n ?/g, '\n')
    // A list item whose text sits in a nested block starts on the line after its marker.
    .replace(/\n-\n+(?!-)/g, '\n- ').replace(/\n{3,}/g, '\n\n').trim();
  return { title, text };
}

export type SearchResult = { title: string; url: string; snippet: string };

/** The page a DuckDuckGo result link leads to; its links go through `/l/?uddg=<encoded url>`. */
function resultUrl(href: string): string | null {
  const raw = decodeEntities(href);
  let url: URL;
  try { url = new URL(raw.startsWith('//') ? `https:${raw}` : raw, 'https://duckduckgo.com'); } catch { return null; }
  if (/(^|\.)duckduckgo\.com$/.test(url.hostname)) {
    const target = url.pathname === '/l/' ? url.searchParams.get('uddg') : null;
    if (!target) return null; // An advert or an internal link.
    try { url = new URL(target); } catch { return null; }
  }
  return url.protocol === 'http:' || url.protocol === 'https:' ? url.href : null;
}

/**
 * Results from a DuckDuckGo results page, in rank order. The HTML endpoint marks a result's title link
 * `result__a` and its snippet `result__snippet`; the lite endpoint uses `rel="nofollow"` and `result-snippet`.
 */
export function parseSearchResults(html: string, limit = SEARCH_RESULTS): SearchResult[] {
  const results: SearchResult[] = [];
  const anchors = html.includes('result__a')
    ? /<a\b[^>]*class="[^"]*\bresult__a\b[^"]*"[^>]*>[\s\S]*?<\/a>/gi
    : /<a\b[^>]*rel="nofollow"[^>]*>[\s\S]*?<\/a>/gi;
  const found = [...html.matchAll(anchors)];
  found.forEach((match, index) => {
    if (results.length >= limit) return;
    const url = resultUrl(/href="([^"]+)"/i.exec(match[0])?.[1] ?? '');
    const title = plain(match[0]);
    if (!url || !title || results.some((other) => other.url === url)) return;
    // The snippet sits between this result's title and the next one.
    const until = found[index + 1]?.index ?? html.length;
    const rest = html.slice(match.index + match[0].length, until);
    const snippet = plain(/class="[^"]*\bresult(?:__|-)snippet\b[^"]*"[^>]*>([\s\S]*?)<\/(a|div|td)>/i.exec(rest)?.[1] ?? '');
    results.push({ title, url, snippet: snippet.slice(0, SNIPPET_CHARS) });
  });
  return results;
}

type Body = { getReader(): { read(): Promise<{ done: boolean; value?: Uint8Array }>; cancel(): Promise<unknown> } };
type FetchResponse = {
  ok: boolean; status: number; url?: string; headers: { get(name: string): string | null };
  body?: Body | null; text(): Promise<string>;
};
export type ToolFetch = (url: string, init: {
  method?: string; body?: string; headers?: Record<string, string>; signal?: AbortSignal; credentials?: 'omit';
}) => Promise<FetchResponse>;
export type ToolEnvironment = { fetch: ToolFetch; now?: () => Date; timeoutMs?: number };
export type ToolResult = { output: string; is_error: boolean };

/** The response text, read no further than `limit` bytes. */
async function limitedText(response: FetchResponse, limit: number): Promise<{ text: string; truncated: boolean }> {
  if (!response.body) {
    const text = await response.text();
    return { text: text.slice(0, limit), truncated: text.length > limit };
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let text = '';
  let read = 0;
  for (;;) {
    const chunk = await reader.read();
    if (chunk.done || !chunk.value) return { text: text + decoder.decode(), truncated: false };
    const room = limit - read;
    read += chunk.value.length;
    text += decoder.decode(chunk.value.subarray(0, room), { stream: true });
    if (read >= limit) {
      void reader.cancel().catch(() => undefined);
      return { text, truncated: true };
    }
  }
}

/** One request with the tool's time limit; stopping the reply stops it too. */
async function request<T>(
  environment: ToolEnvironment, signal: AbortSignal, url: string, init: { method?: string; body?: string; headers?: Record<string, string> },
  read: (response: FetchResponse) => Promise<T>,
): Promise<T> {
  const limit = new AbortController();
  const stop = () => limit.abort();
  signal.addEventListener('abort', stop, { once: true });
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; limit.abort(); }, environment.timeoutMs ?? FETCH_TIMEOUT_MS);
  try {
    if (signal.aborted) throw new ToolError('Stopped.');
    return await read(await environment.fetch(url, {
      ...init, signal: limit.signal, credentials: 'omit',
      headers: { 'user-agent': USER_AGENT, 'accept-language': 'en-US,en;q=0.8', ...init.headers },
    }));
  } catch (cause) {
    if (timedOut) throw new ToolError(`No answer within ${(environment.timeoutMs ?? FETCH_TIMEOUT_MS) / 1000} seconds.`);
    if (signal.aborted) throw new ToolError('Stopped.');
    if (cause instanceof ToolError) throw cause;
    throw new ToolError('Could not reach the site. Check the address and this phone’s connection.');
  } finally {
    clearTimeout(timer);
    signal.removeEventListener('abort', stop);
  }
}

async function fetchUrl(args: Record<string, unknown>, environment: ToolEnvironment, signal: AbortSignal): Promise<string> {
  if (typeof args.url !== 'string' || !args.url.trim()) throw new ToolError('fetch_url needs a "url".');
  const url = safeUrl(/^[a-z][a-z0-9+.-]*:/i.test(args.url.trim()) ? args.url : `https://${args.url.trim()}`);
  return request(environment, signal, url.href, { headers: { accept: 'text/html,text/plain,application/json;q=0.9,*/*;q=0.1' } },
    async (response) => {
      // Redirects are followed by the platform, so where they ended is checked before anything is read.
      if (response.url) safeUrl(response.url);
      if (!response.ok) throw new ToolError(`HTTP ${response.status} from ${url.hostname}.`);
      const type = (response.headers.get('content-type') ?? '').split(';')[0].trim().toLowerCase();
      const html = !type || type.includes('html');
      if (!html && !type.startsWith('text/') && !/(json|xml)$/.test(type)) {
        throw new ToolError(`This is a ${type} file, not a page of text.`);
      }
      const { text: raw, truncated } = await limitedText(response, FETCH_MAX_BYTES);
      const page = html && /<[a-z!][\s\S]*>/i.test(raw) ? readableText(raw) : { title: '', text: raw.trim() };
      if (!page.text) throw new ToolError('The page has no readable text. It may need JavaScript to show its content.');
      const cut = page.text.length > FETCH_MAX_CHARS;
      return [
        page.title ? `Title: ${page.title}` : '', `URL: ${response.url || url.href}`, '',
        cut ? page.text.slice(0, FETCH_MAX_CHARS) : page.text,
        cut || truncated ? '\n[The page continues; only its beginning is shown.]' : '',
      ].filter((line, index) => line || index === 2).join('\n');
    });
}

async function webSearch(args: Record<string, unknown>, environment: ToolEnvironment, signal: AbortSignal): Promise<string> {
  const query = typeof args.query === 'string' ? args.query.trim().slice(0, 400) : '';
  if (!query) throw new ToolError('web_search needs a "query".');
  // DuckDuckGo answers a client it is rate limiting with 202 or 403 and a bot check instead of results.
  const limited = new ToolError('DuckDuckGo is limiting searches from this network. Try again in a minute.');
  const ask = (endpoint: string) => request(environment, signal, endpoint, {
    method: 'POST', body: `q=${encodeURIComponent(query)}&b=&kp=-1`,
    headers: { 'content-type': 'application/x-www-form-urlencoded', accept: 'text/html,application/xhtml+xml' },
  }, async (response) => {
    if (response.status === 202 || response.status === 403) throw limited;
    if (!response.ok) throw new ToolError(`HTTP ${response.status} from DuckDuckGo.`);
    const { text } = await limitedText(response, FETCH_MAX_BYTES);
    const parsed = parseSearchResults(text);
    if (!parsed.length && /anomaly|challenge|captcha|Just a moment/i.test(text)) throw limited;
    return parsed;
  });
  // The lite endpoint is limited separately, so it is tried when the first one gives nothing.
  let results = await ask('https://html.duckduckgo.com/html/').catch((cause: unknown) => {
    if (signal.aborted) throw cause;
    return null;
  });
  if (!results?.length) results = await ask('https://lite.duckduckgo.com/lite/');
  if (!results.length) return `No results for "${query}".`;
  return results.map((result, index) =>
    `${index + 1}. ${result.title}\n${result.url}${result.snippet ? `\n${result.snippet}` : ''}`).join('\n\n');
}

/** Arguments as an object. Small models wrap the JSON in a code fence or encode it twice. */
export function parseArguments(raw: string): Record<string, unknown> {
  let text = raw.trim().replace(/^```(?:json)?\s*|\s*```$/g, '');
  for (let depth = 0; depth < 2; depth += 1) {
    if (!text) return {};
    let value: unknown;
    try { value = JSON.parse(text); } catch { break; }
    if (typeof value === 'string') { text = value.trim(); continue; }
    if (value && typeof value === 'object' && !Array.isArray(value)) return value as Record<string, unknown>;
    break;
  }
  throw new ToolError('The arguments are not a JSON object.');
}

/** Run one tool. A failure is returned as text the model can react to, as the host's tools do. */
export async function runLocalTool(
  name: string, rawArguments: string, environment: ToolEnvironment, signal: AbortSignal,
): Promise<ToolResult> {
  try {
    const args = parseArguments(rawArguments);
    if (name === 'web_search') return { output: await webSearch(args, environment, signal), is_error: false };
    if (name === 'fetch_url') return { output: await fetchUrl(args, environment, signal), is_error: false };
    if (name === 'get_current_time') {
      const now = environment.now?.() ?? new Date();
      return { output: `${now.toString()} (UTC ${now.toISOString()})`, is_error: false };
    }
    throw new ToolError(`There is no tool named ${name} on this device.`);
  } catch (cause) {
    return { output: `Error: ${cause instanceof Error ? cause.message : String(cause)}`, is_error: true };
  }
}

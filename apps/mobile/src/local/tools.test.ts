import assert from 'node:assert/strict';
import test from 'node:test';
import {
  FETCH_MAX_BYTES, FETCH_MAX_CHARS, parseArguments, parseSearchResults, readableText, runLocalTool, safeUrl, toolDefinitions,
  type ToolFetch,
} from './tools.ts';

const never = new AbortController().signal;
const page = (body: string, init: { status?: number; type?: string; url?: string } = {}) => ({
  ok: (init.status ?? 200) < 400, status: init.status ?? 200, url: init.url,
  headers: { get: (name: string) => name === 'content-type' ? init.type ?? 'text/html; charset=utf-8' : null },
  body: new Response(body).body, text: async () => body,
});
const serving = (body: string, init?: Parameters<typeof page>[1]): ToolFetch => async () => page(body, init);

test('only public http and https addresses are safe to fetch', () => {
  for (const url of ['https://example.com/a?b=1', 'http://8.8.8.8/', 'https://[2606:4700::1111]/', 'https://sub.example.co.uk']) {
    assert.equal(safeUrl(url).href, new URL(url).href);
  }
  for (const url of [
    'ftp://example.com', 'file:///etc/passwd', 'javascript:alert(1)', 'not a url', 'https://user:pw@example.com',
    'http://localhost:8080', 'http://127.0.0.1', 'http://10.0.2.2:3000', 'http://192.168.1.10', 'http://172.16.0.1',
    'http://169.254.169.254/latest/meta-data', 'http://100.64.0.1', 'http://0.0.0.0', 'http://[::1]/', 'http://[fe80::1]/',
    'http://[fd00::1]/', 'http://[::ffff:127.0.0.1]/', 'http://2130706433', 'http://0x7f.0.0.1', 'http://printer',
    'http://nas.local', 'http://db.internal', 'http://router.lan.',
  ]) assert.throws(() => safeUrl(url), Error, url);
});

test('readable text keeps the article and drops scripts, styles and navigation', () => {
  const { title, text } = readableText(`<html><head><title>Tea &amp; Coffee</title><style>p{color:red}</style></head>
    <body><nav><a href="/">Home</a></nav><script>track()</script>
    <article><h1>Brewing</h1><p>Water at 90&deg;C &mdash; not boiling. ${'Steep for three minutes. '.repeat(12)}</p>
    <ul><li>Green</li><li>Black &#38; white</li></ul></article><footer>Copyright</footer></body></html>`);
  assert.equal(title, 'Tea & Coffee');
  assert.match(text, /^Brewing\n\nWater at 90°C — not boiling\./);
  assert.match(text, /\n\n- Green\n- Black & white$/);
  assert.doesNotMatch(text, /Home|track|color|Copyright/);
});

test('search results are read from a DuckDuckGo page with their real addresses', () => {
  const html = `
    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs%3Fa%3D1&amp;rut=x">Example <b>Docs</b></a>
    <a class="result__snippet" href="#">The <b>docs</b> &amp; guides.</a>
    <a class="result__a" href="https://duckduckgo.com/y.js?ad_provider=x">An advert</a>
    <a class="result__snippet">Buy now</a>
    <a class="result__a" href="https://second.example/">Second</a>
    <a class="result__a" href="https://second.example/">Second again</a>`;
  assert.deepEqual(parseSearchResults(html), [
    { title: 'Example Docs', url: 'https://example.com/docs?a=1', snippet: 'The docs & guides.' },
    { title: 'Second', url: 'https://second.example/', snippet: '' },
  ]);
});

test('web_search posts the query to DuckDuckGo and lists the results', async () => {
  let asked: { url: string; body?: string } | null = null;
  const fetch: ToolFetch = async (url, init) => {
    asked = { url, body: init.body };
    return page('<a class="result__a" href="https://example.com/">Example</a><a class="result__snippet">A site.</a>');
  };
  const result = await runLocalTool('web_search', '{"query":"tea & coffee"}', { fetch }, never);
  assert.deepEqual(asked, { url: 'https://html.duckduckgo.com/html/', body: 'q=tea%20%26%20coffee&b=&kp=-1' });
  assert.deepEqual(result, { output: '1. Example\nhttps://example.com/\nA site.', is_error: false });
  assert.match((await runLocalTool('web_search', '{"query":"x"}', { fetch: serving('<p>anomaly detected</p>') }, never)).output,
    /^Error: DuckDuckGo is limiting/);
  // A limited first endpoint falls back to the lite one, whose page is laid out differently.
  const asked2: string[] = [];
  const fallback: ToolFetch = async (url) => {
    asked2.push(new URL(url).hostname);
    return url.includes('lite.') ? page(`<table><tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2F&amp;rut=a" class='result-link'>Lite hit</a></td></tr>
      <tr><td class="result-snippet">From the lite page.</td></tr></table>`) : page('', { status: 202 });
  };
  assert.equal((await runLocalTool('web_search', '{"query":"x"}', { fetch: fallback }, never)).output,
    '1. Lite hit\nhttps://example.org/\nFrom the lite page.');
  assert.deepEqual(asked2, ['html.duckduckgo.com', 'lite.duckduckgo.com']);
  assert.equal((await runLocalTool('web_search', '{"query":"x"}', { fetch: serving('<p>nothing</p>') }, never)).output,
    'No results for "x".');
  assert.deepEqual(await runLocalTool('web_search', '{}', { fetch }, never),
    { output: 'Error: web_search needs a "query".', is_error: true });
});

test('fetch_url returns the page text, cut to what a phone model can read', async () => {
  const result = await runLocalTool('fetch_url', '{"url":"example.com/a"}',
    { fetch: serving(`<title>Long</title><p>${'word '.repeat(2000)}</p>`, { url: 'https://example.com/a' }) }, never);
  assert.equal(result.is_error, false);
  assert.match(result.output, /^Title: Long\nURL: https:\/\/example\.com\/a\n\nword /);
  assert.match(result.output, /\[The page continues; only its beginning is shown\.\]$/);
  assert.ok(result.output.length < FETCH_MAX_CHARS + 200);
  const plain = await runLocalTool('fetch_url', '{"url":"https://example.com/a.txt"}',
    { fetch: serving('a < b', { type: 'text/plain' }) }, never);
  assert.equal(plain.output, 'URL: https://example.com/a.txt\n\na < b');
});

test('fetch_url refuses unsafe addresses, redirects into a private network, files and failures', async () => {
  let called = 0;
  const counting: ToolFetch = async () => { called += 1; return page('x'); };
  assert.match((await runLocalTool('fetch_url', '{"url":"http://192.168.1.1/admin"}', { fetch: counting }, never)).output,
    /^Error: Local and private addresses/);
  assert.equal(called, 0);
  const run = (fetch: ToolFetch) => runLocalTool('fetch_url', '{"url":"https://example.com"}', { fetch, timeoutMs: 20 }, never);
  assert.match((await run(serving('secret', { url: 'http://127.0.0.1:8080/' }))).output, /^Error: Local and private addresses/);
  assert.match((await run(serving('', { status: 404 }))).output, /^Error: HTTP 404 from example\.com/);
  assert.match((await run(serving('%PDF', { type: 'application/pdf' }))).output, /^Error: This is a application\/pdf file/);
  assert.match((await run(async () => { throw new TypeError('Network request failed'); })).output, /^Error: Could not reach the site/);
  const hanging: ToolFetch = (_url, init) => new Promise((_resolve, reject) => {
    init.signal?.addEventListener('abort', () => reject(new Error('aborted')));
  });
  assert.equal((await run(hanging)).output, 'Error: No answer within 0.02 seconds.');
  const stop = new AbortController();
  const pending = runLocalTool('fetch_url', '{"url":"https://example.com"}', { fetch: hanging }, stop.signal);
  stop.abort();
  assert.deepEqual(await pending, { output: 'Error: Stopped.', is_error: true });
});

test('fetch_url stops reading a page at the size limit', async () => {
  let pulled = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) { pulled += 1; controller.enqueue(new TextEncoder().encode('a'.repeat(64 * 1024))); },
  });
  const fetch: ToolFetch = async () => ({ ...page('', { type: 'text/plain' }), body });
  const result = await runLocalTool('fetch_url', '{"url":"https://example.com/big"}', { fetch }, never);
  assert.equal(result.is_error, false);
  assert.ok(pulled <= FETCH_MAX_BYTES / (64 * 1024) + 2, `read ${pulled} chunks`);
});

test('arguments from small models are repaired, and unknown tools fail as text', async () => {
  assert.deepEqual(parseArguments('```json\n{"query":"x"}\n```'), { query: 'x' });
  assert.deepEqual(parseArguments('"{\\"query\\":\\"x\\"}"'), { query: 'x' });
  assert.deepEqual(parseArguments(''), {});
  assert.throws(() => parseArguments('[1]'));
  assert.deepEqual(await runLocalTool('shell', '{}', { fetch: serving('') }, never),
    { output: 'Error: There is no tool named shell on this device.', is_error: true });
  const time = await runLocalTool('get_current_time', '', { fetch: serving(''), now: () => new Date(0) }, never);
  assert.match(time.output, /UTC 1970-01-01T00:00:00\.000Z/);
});

test('only the chosen tools are defined for the model', () => {
  const names = (toolDefinitions(['fetch_url', 'web_search', 'nope']) as { function: { name: string } }[])
    .map((definition) => definition.function.name);
  assert.deepEqual(names, ['web_search', 'fetch_url']);
  assert.deepEqual(toolDefinitions([]), []);
});

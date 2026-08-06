//! Shared web search and retrieval for the chat built-in tools and the agent
//! power tools.
//!
//! One guarded downloader, one search-backend selection, and one rate budget
//! serve the whole daemon: a private-network fetch is impossible from any path,
//! and a single search engine's rate limit is not burned from several callers
//! at once.
//!
//! Search runs keyless against DuckDuckGo by default. DuckDuckGo sometimes
//! serves a bot-check challenge to honest clients; the error message points the
//! user at the Brave Search API (Manage → Web search), which is the reliable
//! higher-limit path.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::Context;
use regex::Regex;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::runtime_settings::RuntimeSettings;

/// Honest, identifying user agent. Search engines and sites can still choose
/// to serve us, but we do not pretend to be a web browser: the request states
/// who it is and what it is for.
pub const WEB_USER_AGENT: &str = concat!(
    "Brazier/",
    env!("CARGO_PKG_VERSION"),
    " (+https://github.com/dylancvdean/brazier; FOSS desktop client; web tools)"
);

/// Bounds for the guarded downloader shared by chat `fetch_url` and agent
/// `web_fetch`.
pub const FETCH_TIMEOUT: Duration = Duration::from_secs(15);
pub const FETCH_MAX_BYTES: usize = 256 * 1024;
pub const FETCH_MAX_OUTPUT_CHARS: usize = 8_000;

/// Keyless search budget, applied per outbound provider request rather than
/// per logical search: one `web_search` that tries DuckDuckGo's HTML page and
/// then its lite page consumes two budget slots. Brave's paid API is the
/// higher-limit path and is never gated here.
const SEARCH_RATE_LIMIT: usize = 30;
const SEARCH_RATE_WINDOW: Duration = Duration::from_secs(60);

/// Page fetches share a gentler budget so doc-heavy workflows are not
/// throttled.
const FETCH_RATE_LIMIT: usize = 40;
const FETCH_RATE_WINDOW: Duration = Duration::from_secs(60);

/// Which backend answers `web_search`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchProvider {
    /// Keyless, rate-limited scraping of DuckDuckGo. The default. DDG serves a
    /// bot-check challenge to many headless clients, so when it blocks, the
    /// error points the user at Brave's paid API.
    DuckDuckGo,
    /// Brave's paid Search API. Requires `brave_api_key`.
    Brave,
}

impl SearchProvider {
    pub fn from_settings(settings: &RuntimeSettings) -> Self {
        match settings.web_search_provider.as_str() {
            "brave" => Self::Brave,
            _ => Self::DuckDuckGo,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::DuckDuckGo => "duckduckgo",
            Self::Brave => "brave",
        }
    }
}

/// One formatted search hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

// --- Rate budget -------------------------------------------------------------

struct RateGate {
    max: usize,
    window: Duration,
    hits: VecDeque<Instant>,
}

impl RateGate {
    fn new(max: usize, window: Duration) -> Self {
        Self {
            max,
            window,
            hits: VecDeque::new(),
        }
    }

    /// Record one hit, or refuse when the window is already spent. The caller
    /// surfaces the wait so the model can retry instead of hanging the loop.
    async fn acquire(&mut self) -> anyhow::Result<()> {
        let now = Instant::now();
        while self
            .hits
            .front()
            .is_some_and(|hit| now.duration_since(*hit) >= self.window)
        {
            self.hits.pop_front();
        }
        if self.hits.len() >= self.max {
            let oldest = self.hits.front().copied().unwrap_or(now);
            let retry = self
                .window
                .saturating_sub(now.duration_since(oldest))
                .as_secs()
                .max(1);
            anyhow::bail!(
                "web tool rate limit reached ({} per {}s); wait ~{}s and try again",
                self.max,
                self.window.as_secs(),
                retry
            );
        }
        self.hits.push_back(now);
        Ok(())
    }
}

static SEARCH_GATE: Mutex<Option<RateGate>> = Mutex::const_new(None);
static FETCH_GATE: Mutex<Option<RateGate>> = Mutex::const_new(None);

async fn gate(cell: &Mutex<Option<RateGate>>, max: usize, window: Duration) -> anyhow::Result<()> {
    let mut guard = cell.lock().await;
    let inner = guard.get_or_insert_with(|| RateGate::new(max, window));
    inner.acquire().await
}

// --- Web search --------------------------------------------------------------

/// Search the web with the configured backend and return ranked hits.
///
/// `region` and `safesearch` override the daemon's configured defaults when
/// given. `safesearch` is one of `moderate`, `strict`, or `off`; `region` is a
/// DuckDuckGo region code such as `us-en`, `de-de`, or `wt-wt` (Brave ignores
/// it).
///
/// DuckDuckGo is rate-gated per outbound request inside the provider function,
/// so each HTTP request it makes counts against the shared budget; the paid
/// Brave API is never gated. When DuckDuckGo blocks the machine, the error
/// recommends configuring Brave instead.
pub async fn search(
    client: &reqwest::Client,
    query: &str,
    max_results: usize,
    region: Option<&str>,
    safesearch: Option<&str>,
    settings: &RuntimeSettings,
) -> anyhow::Result<Vec<WebResult>> {
    let query = query.trim();
    anyhow::ensure!(!query.is_empty(), "query must not be empty");
    let max_results = max_results.clamp(1, 10);
    let region = region
        .filter(|region| !region.trim().is_empty())
        .or(settings.web_search_region.as_deref());
    let safesearch = safesearch.unwrap_or(settings.web_safesearch.as_str());
    match SearchProvider::from_settings(settings) {
        SearchProvider::DuckDuckGo => {
            ddg_search(client, query, max_results, region, safesearch).await
        }
        SearchProvider::Brave => {
            let api_key = settings.brave_api_key.as_deref().context(
                "web search provider is set to Brave but no Brave API key is configured. Add one \
                 in Manage → Web search, or switch back to DuckDuckGo.",
            )?;
            brave_search(client, api_key, query, max_results, safesearch).await
        }
    }
}

async fn ddg_search(
    client: &reqwest::Client,
    query: &str,
    max_results: usize,
    region: Option<&str>,
    safesearch: &str,
) -> anyhow::Result<Vec<WebResult>> {
    // HTML endpoint first. When it blocks the client (HTTP 202/403 or an empty
    // page), fall back to the lite page before giving up.
    let html = ddg_html(client, query, max_results, region, safesearch).await;
    if let Ok(results) = &html
        && !results.is_empty()
    {
        return Ok(results.clone());
    }
    match ddg_lite(client, query, max_results, region, safesearch).await {
        Ok(results) if !results.is_empty() => Ok(results),
        Ok(_) => match html {
            Ok(_) => Ok(Vec::new()),
            Err(html_error) => Err(html_error),
        },
        Err(lite_error) => match html {
            Err(html_error) => Err(anyhow::anyhow!(
                "DuckDuckGo search failed for `{query}` (html: {html_error:#}; lite: {lite_error:#})"
            )),
            Ok(_) => Err(lite_error),
        },
    }
}

async fn ddg_lite(
    client: &reqwest::Client,
    query: &str,
    max_results: usize,
    region: Option<&str>,
    safesearch: &str,
) -> anyhow::Result<Vec<WebResult>> {
    let (status, html) = ddg_get(
        client,
        "https://lite.duckduckgo.com/lite/",
        query,
        region,
        safesearch,
    )
    .await?;
    if is_ddg_block(status, &html) {
        return Err(blocked_error(query));
    }
    Ok(parse_lite_results(&html, max_results))
}

async fn ddg_html(
    client: &reqwest::Client,
    query: &str,
    max_results: usize,
    region: Option<&str>,
    safesearch: &str,
) -> anyhow::Result<Vec<WebResult>> {
    let (status, html) = ddg_get(
        client,
        "https://html.duckduckgo.com/html",
        query,
        region,
        safesearch,
    )
    .await?;
    if is_ddg_block(status, &html) {
        return Err(blocked_error(query));
    }
    Ok(parse_html_results(&html, max_results))
}

fn blocked_error(query: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "DuckDuckGo is serving a bot-check challenge for `{query}` (rate limit or anti-bot \
         detection). Try again later, rephrase the query, or set a Brave Search API key in \
         Manage → Web search to lift the keyless ceiling."
    )
}

/// A response that DuckDuckGo's anti-bot layer returns instead of results: the
/// HTML endpoint answers a plain client with HTTP 202 and an empty page, a 403
/// is the other classic block signal, and a Cloudflare-style interstitial can
/// ride along on a 200.
fn is_ddg_block(status: reqwest::StatusCode, html: &str) -> bool {
    if status == reqwest::StatusCode::ACCEPTED || status == reqwest::StatusCode::FORBIDDEN {
        return true;
    }
    if status.is_success() && html.trim().is_empty() {
        return true;
    }
    if status.is_success() && status != reqwest::StatusCode::OK {
        return false;
    }
    const CLOUDFLARE_SIGNALS: &[&str] = &[
        "cf-mitigated",
        "Just a moment...",
        "Enable JavaScript and cookies to continue",
        "Checking your browser before accessing",
    ];
    let sample = &html[..html.len().min(4096)];
    CLOUDFLARE_SIGNALS
        .iter()
        .any(|signal| sample.contains(signal))
}

async fn ddg_get(
    client: &reqwest::Client,
    base: &str,
    query: &str,
    region: Option<&str>,
    safesearch: &str,
) -> anyhow::Result<(reqwest::StatusCode, String)> {
    gate(&SEARCH_GATE, SEARCH_RATE_LIMIT, SEARCH_RATE_WINDOW).await?;
    // The HTML endpoint accepts a form POST (q, b, kl, kp); the lite endpoint
    // takes the same shape. Only honest headers are sent: our identifying user
    // agent, a normal Accept, and no navigation-fetch hints.
    let mut form: Vec<(&str, String)> = vec![("q", query.to_owned()), ("b", String::new())];
    if let Some(region) = region.filter(|region| !region.trim().is_empty()) {
        form.push(("kl", region.trim().to_owned()));
    }
    form.push(("kp", ddg_kp(safesearch).to_string()));
    let response = client
        .post(base)
        .form(&form)
        .header("user-agent", WEB_USER_AGENT)
        .header("accept", "text/html,application/xhtml+xml")
        .header("accept-language", "en-US,en;q=0.9")
        .timeout(FETCH_TIMEOUT)
        .send()
        .await
        .with_context(|| format!("DuckDuckGo search request failed for `{query}`"))?;
    let status = response.status();
    let body = response.text().await.context("read DuckDuckGo response")?;
    Ok((status, body))
}

/// DuckDuckGo `kp` parameter: 1 strict, -1 moderate, -2 off.
fn ddg_kp(safesearch: &str) -> i8 {
    match safesearch.trim().to_ascii_lowercase().as_str() {
        "strict" | "on" => 1,
        "off" => -2,
        _ => -1,
    }
}

/// Format hits the way the DuckDuckGo MCP server does: a count line then
/// ranked `title / URL / Summary` blocks, which search-tuned models are used to
/// reading.
pub fn format_results(results: &[WebResult]) -> String {
    if results.is_empty() {
        return "No results were found for your search query. This could be due to the search \
                engine blocking the request or the query returning no matches. Please try \
                rephrasing your search or try again in a few minutes."
            .to_owned();
    }
    let mut output = format!("Found {} search results:\n\n", results.len());
    for (index, result) in results.iter().enumerate() {
        output.push_str(&format!(
            "{}. {}\n   URL: {}\n   Summary: {}\n\n",
            index + 1,
            result.title,
            result.url,
            result.snippet
        ));
    }
    output.trim_end().to_owned()
}

async fn brave_search(
    client: &reqwest::Client,
    api_key: &str,
    query: &str,
    max_results: usize,
    safesearch: &str,
) -> anyhow::Result<Vec<WebResult>> {
    let safesearch = match safesearch.trim().to_ascii_lowercase().as_str() {
        "strict" | "on" => "strict",
        "off" => "off",
        _ => "moderate",
    };
    let url = reqwest::Url::parse_with_params(
        "https://api.search.brave.com/res/v1/web/search",
        &[
            ("q", query.to_owned()),
            ("count", max_results.min(20).to_string()),
            ("safesearch", safesearch.to_owned()),
        ],
    )
    .context("build Brave search URL")?;
    let response = client
        .get(url)
        .header("X-Subscription-Token", api_key)
        .header("Accept", "application/json")
        .timeout(FETCH_TIMEOUT)
        .send()
        .await
        .with_context(|| format!("Brave search request failed for `{query}`"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("read Brave search response")?;
    anyhow::ensure!(
        status.is_success(),
        "Brave search returned HTTP {status}: {}",
        brave_error_summary(&body)
    );
    let parsed: Value = serde_json::from_str(&body).context("Brave returned invalid JSON")?;
    let mut results = Vec::new();
    let mut seen = std::collections::HashSet::new();
    if let Some(items) = parsed.pointer("/web/results").and_then(Value::as_array) {
        for item in items {
            let title = item
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let url = item.get("url").and_then(Value::as_str).unwrap_or("").trim();
            let snippet = item
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if title.is_empty() || url.is_empty() || !seen.insert(url.to_owned()) {
                continue;
            }
            results.push(WebResult {
                title: title.to_owned(),
                url: url.to_owned(),
                snippet: snippet.to_owned(),
            });
            if results.len() >= max_results {
                break;
            }
        }
    }
    Ok(results)
}

/// Keep an error response readable without dumping the whole body.
fn brave_error_summary(body: &str) -> String {
    if let Ok(parsed) = serde_json::from_str::<Value>(body)
        && let Some(message) = parsed
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| parsed.get("error").and_then(Value::as_str))
    {
        return message.to_owned();
    }
    let compact: String = body.chars().take(200).collect();
    if compact.is_empty() {
        "empty error body".to_owned()
    } else {
        compact
    }
}

/// Parse DuckDuckGo's lite result page into `(title, url, snippet)` tuples.
pub fn parse_lite_results(html: &str, max: usize) -> Vec<WebResult> {
    // Lite packs each hit into one cell: the first `rel=nofollow` link is the
    // title, and the snippet follows in the same cell. Drop the title again so
    // it is not repeated inside the snippet.
    let title_re = Regex::new(r#"(?is)<a[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*>(.*?)</a>"#)
        .expect("valid regex");
    let snippet_re =
        Regex::new(r#"(?is)<td[^>]*class="result-snippet"[^>]*>(.*?)</td>"#).expect("valid regex");
    let links: Vec<(String, String)> = title_re
        .captures_iter(html)
        .map(|captures| {
            let href = decode_html_entities(&captures[1]);
            let url = redirect_url(&href).unwrap_or(href);
            (url, strip_inline(&captures[2]))
        })
        .collect();
    let cells: Vec<String> = snippet_re
        .captures_iter(html)
        .map(|captures| strip_inline(&captures[1]))
        .collect();
    links
        .into_iter()
        .zip(cells.into_iter().chain(std::iter::repeat_with(String::new)))
        .filter_map(|((url, title), mut snippet)| {
            if url.is_empty() || title.is_empty() || url.contains("y.js") {
                return None;
            }
            if !snippet.is_empty() {
                let trimmed = snippet.trim_start_matches(&title).trim();
                if !trimmed.is_empty() {
                    snippet = trimmed.to_owned();
                }
            }
            Some(WebResult {
                title,
                url,
                snippet,
            })
        })
        .take(max)
        .collect()
}

/// Parse DuckDuckGo's HTML result page into `(title, url, snippet)` tuples.
pub fn parse_html_results(html: &str, max: usize) -> Vec<WebResult> {
    let title_re = Regex::new(r#"(?is)<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>"#)
        .expect("valid regex");
    let snippet_re =
        Regex::new(r#"(?is)<a[^>]*class="result__snippet"[^>]*>(.*?)</a>"#).expect("valid regex");
    let titles: Vec<(String, String)> = title_re
        .captures_iter(html)
        .map(|captures| {
            let href = decode_html_entities(&captures[1]);
            let url = redirect_url(&href).unwrap_or(href);
            (url, strip_inline(&captures[2]))
        })
        .collect();
    let snippets: Vec<String> = snippet_re
        .captures_iter(html)
        .map(|captures| strip_inline(&captures[1]))
        .collect();
    titles
        .into_iter()
        .zip(
            snippets
                .into_iter()
                .chain(std::iter::repeat_with(String::new)),
        )
        .filter_map(|((url, title), snippet)| {
            if url.is_empty() || title.is_empty() || url.contains("y.js") {
                return None;
            }
            Some(WebResult {
                title,
                url,
                snippet,
            })
        })
        .take(max)
        .collect()
}

/// DuckDuckGo wraps real URLs in `//duckduckgo.com/l/?uddg=<encoded>&rut=…`.
///
/// Decodes like Python's `urllib.parse.unquote`: only `%XX` escapes, never a
/// `+` to space (a literal plus is legal inside a URL).
fn redirect_url(href: &str) -> Option<String> {
    let query = href.split('?').nth(1)?;
    let uddg = query
        .split('&')
        .find_map(|pair| pair.strip_prefix("uddg="))?;
    Some(percent_decode_url(uddg))
}

fn percent_decode_url(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' if index + 2 < bytes.len() => {
                match u8::from_str_radix(&input[index + 1..index + 3], 16) {
                    Ok(value) => {
                        out.push(value);
                        index += 3;
                    }
                    Err(_) => {
                        out.push(bytes[index]);
                        index += 1;
                    }
                }
            }
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Decode the handful of HTML entities search results actually use.
fn decode_html_entities(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        let Some(end) = tail.find(';') else {
            out.push_str(tail);
            return out;
        };
        let entity = &tail[..=end];
        let replacement = match entity {
            "&amp;" => "&",
            "&lt;" => "<",
            "&gt;" => ">",
            "&quot;" => "\"",
            "&#39;" | "&apos;" => "'",
            "&nbsp;" => " ",
            _ => {
                if let Some(code) = entity
                    .strip_prefix("&#")
                    .and_then(|rest| rest.strip_suffix(';'))
                    && let Ok(code) = code.parse::<u32>()
                    && let Some(character) = char::from_u32(code)
                {
                    out.push(character);
                    rest = &tail[entity.len()..];
                    continue;
                }
                entity
            }
        };
        out.push_str(replacement);
        rest = &tail[entity.len()..];
    }
    out.push_str(rest);
    out
}

/// Strip tags and decode entities from a title/snippet fragment.
fn strip_inline(fragment: &str) -> String {
    let tags = Regex::new(r"(?is)<[^>]+>").expect("valid regex");
    let text = decode_html_entities(&tags.replace_all(fragment, ""));
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Percent-encode a search query for a URL query string (`+` for space).
pub fn percent_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// Percent-decode a query-string fragment (`+` becomes space).
pub fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' if index + 2 < bytes.len() => {
                match u8::from_str_radix(&input[index + 1..index + 3], 16) {
                    Ok(value) => {
                        out.push(value);
                        index += 3;
                    }
                    Err(_) => {
                        out.push(bytes[index]);
                        index += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                index += 1;
            }
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// True when an address is safe for outbound tool fetches (not loopback,
/// private, link-local, documentation, or similarly non-routable).
pub fn ip_is_public(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            !(v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_documentation()
                // Carrier-grade NAT 100.64.0.0/10.
                || (v4.octets()[0] == 100 && (64..=127).contains(&v4.octets()[1])))
        }
        std::net::IpAddr::V6(v6) => {
            // IPv4-mapped addresses must use the IPv4 policy; otherwise
            // ::ffff:127.0.0.1 and ::ffff:169.254.169.254 look "public".
            if let Some(v4) = v6.to_ipv4_mapped() {
                return ip_is_public(std::net::IpAddr::V4(v4));
            }
            let segments = v6.segments();
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (segments[0] & 0xfe00) == 0xfc00 // unique local fc00::/7
                || (segments[0] & 0xffc0) == 0xfe80 // link local fe80::/10
                || (segments[0] == 0x2001 && segments[1] == 0xdb8)) // documentation
        }
    }
}

/// Resolve `host` and ensure every address is publicly routable. Returns the
/// public addresses so callers can pin connects and avoid DNS rebinding.
pub async fn resolve_public_host(
    host: &str,
    port: u16,
) -> anyhow::Result<Vec<std::net::SocketAddr>> {
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        anyhow::ensure!(ip_is_public(ip), "address {ip} is not publicly routable");
        return Ok(vec![std::net::SocketAddr::new(ip, port)]);
    }
    anyhow::ensure!(
        !host.eq_ignore_ascii_case("localhost") && !host.ends_with(".local"),
        "local hostnames are not allowed"
    );
    let addresses = tokio::net::lookup_host((host, port))
        .await
        .with_context(|| format!("resolve host {host}"))?;
    let mut public = Vec::new();
    for address in addresses {
        anyhow::ensure!(
            ip_is_public(address.ip()),
            "host {host} resolves to non-public address {}",
            address.ip()
        );
        public.push(address);
    }
    anyhow::ensure!(!public.is_empty(), "host {host} resolved to no addresses");
    Ok(public)
}

pub async fn guard_host(host: &str) -> anyhow::Result<()> {
    resolve_public_host(host, 443).await.map(|_| ())
}

/// Strip tags from HTML and collapse whitespace. Deliberately simple.
///
/// Script/style/nav/header/footer blocks are dropped entirely (matching the
/// DuckDuckGo MCP server's `fetch_content`), and the rest is flattened.
pub fn html_to_text(html: &str) -> String {
    let mut output = String::with_capacity(html.len() / 2);
    let mut chars = html.char_indices().peekable();
    let mut skip_until: Option<&str> = None;
    let lower = html.to_ascii_lowercase();
    while let Some((index, character)) = chars.next() {
        if let Some(end_tag) = skip_until {
            if character == '<' && lower[index..].starts_with(end_tag) {
                skip_until = None;
                // Consume through the closing '>'.
                for (_, inner) in chars.by_ref() {
                    if inner == '>' {
                        break;
                    }
                }
            }
            continue;
        }
        if character == '<' {
            let skip = [
                ("<script", "</script"),
                ("<style", "</style"),
                ("<nav", "</nav"),
                ("<header", "</header"),
                ("<footer", "</footer"),
            ]
            .into_iter()
            .find(|(open, _)| lower[index..].starts_with(open));
            if let Some((_, end_tag)) = skip {
                skip_until = Some(end_tag);
                continue;
            }
            for (_, inner) in chars.by_ref() {
                if inner == '>' {
                    break;
                }
            }
            output.push(' ');
            continue;
        }
        output.push(character);
    }
    // Collapse runs of whitespace but keep paragraph-ish newlines.
    let mut collapsed = String::with_capacity(output.len());
    let mut last_was_space = true;
    for character in output.chars() {
        if character.is_whitespace() {
            if !last_was_space {
                collapsed.push(' ');
                last_was_space = true;
            }
        } else {
            collapsed.push(character);
            last_was_space = false;
        }
    }
    collapsed.trim().to_owned()
}

// --- Guarded page download ---------------------------------------------------

/// A fetched page: the final URL after redirects, its content type, and bytes.
pub struct DownloadedUrl {
    pub final_url: reqwest::Url,
    pub content_type: String,
    pub bytes: Vec<u8>,
}

pub fn downloaded_is_pdf(download: &DownloadedUrl) -> bool {
    download.content_type.contains("application/pdf") || download.bytes.starts_with(b"%PDF-")
}

/// Download a public URL with the shared SSRF guard.
///
/// Redirects are followed by hand and every hop is DNS-pinned to addresses that
/// were already vetted as public, so a public URL cannot be steered at a
/// private target mid-flight. Uses the shared fetch rate budget.
pub async fn download_url(url: &str) -> anyhow::Result<DownloadedUrl> {
    gate(&FETCH_GATE, FETCH_RATE_LIMIT, FETCH_RATE_WINDOW).await?;
    let mut parsed = reqwest::Url::parse(url).context("invalid URL")?;
    anyhow::ensure!(
        matches!(parsed.scheme(), "http" | "https"),
        "only http and https URLs are supported"
    );
    let mut redirects = 0_u8;
    let response = loop {
        let host = parsed
            .host_str()
            .context("URL has no host")?
            .trim_start_matches('[')
            .trim_end_matches(']')
            .to_owned();
        let port = parsed.port_or_known_default().context("URL has no port")?;
        let addresses = resolve_public_host(&host, port).await?;
        let mut builder = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(FETCH_TIMEOUT);
        for address in &addresses {
            builder = builder.resolve(&host, *address);
        }
        let client = builder.build().context("build bounded fetch client")?;
        let response = client
            .get(parsed.clone())
            .header("user-agent", WEB_USER_AGENT)
            .send()
            .await
            .context("request failed")?;
        if !response.status().is_redirection() {
            break response;
        }
        redirects += 1;
        anyhow::ensure!(redirects <= 10, "too many redirects");
        let next = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|value| value.to_str().ok())
            .context("redirect response has no valid Location header")?;
        parsed = parsed.join(next).context("invalid redirect URL")?;
        anyhow::ensure!(
            matches!(parsed.scheme(), "http" | "https"),
            "redirected to unsupported URL scheme"
        );
    };
    let status = response.status();
    anyhow::ensure!(status.is_success(), "server returned {status}");
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    use futures::StreamExt;
    let mut stream = response.bytes_stream();
    let mut bytes: Vec<u8> = Vec::new();
    let pdf_content = content_type.contains("application/pdf");
    let byte_limit = if pdf_content || content_type.contains("octet-stream") {
        crate::blob_store::MAX_DOCUMENT_BYTES as usize
    } else {
        FETCH_MAX_BYTES
    };
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read response body")?;
        bytes.extend_from_slice(&chunk);
        anyhow::ensure!(
            bytes.len() <= byte_limit,
            "response exceeds the {} limit",
            if pdf_content { "document" } else { "web fetch" }
        );
    }
    Ok(DownloadedUrl {
        final_url: parsed,
        content_type,
        bytes,
    })
}

/// Convert a downloaded body into model-facing text, mirroring the DuckDuckGo
/// MCP server's `fetch_content`:
///
/// - script/style/nav/header/footer boilerplate is dropped,
/// - JSON is pretty-printed,
/// - `start`/`max_chars` paginate by character and the tail describes what was
///   shown so the model can fetch the rest with `start=<next>`.
pub fn fetch_content_text(
    download: &DownloadedUrl,
    start: usize,
    max_chars: usize,
) -> anyhow::Result<String> {
    let content_type = download.content_type.trim().to_ascii_lowercase();
    anyhow::ensure!(
        content_type.is_empty()
            || content_type.contains("text/")
            || content_type.contains("json")
            || content_type.contains("xml"),
        "unsupported content type `{content_type}`"
    );
    let body = String::from_utf8_lossy(&download.bytes);
    let text = if content_type.contains("html") {
        html_to_text(&body)
    } else if content_type.contains("json") {
        serde_json::from_slice::<Value>(&download.bytes)
            .ok()
            .and_then(|value| serde_json::to_string_pretty(&value).ok())
            .unwrap_or_else(|| body.into_owned())
    } else {
        body.into_owned()
    };
    let total = text.chars().count();
    let mut shown: String = text.chars().skip(start).take(max_chars).collect();
    if shown.is_empty() && total > 0 && start < total {
        shown = "(no extractable text in that range)".to_owned();
    }
    let end = start.saturating_add(shown.chars().count());
    let mut out = shown;
    out.push_str(&format!(
        "\n\n---\n[Content info: Showing characters {start}-{end} of {total} total"
    ));
    if end < total {
        out.push_str(&format!(". Use start={end} to see more"));
    }
    out.push(']');
    anyhow::ensure!(total > 0, "response contained no text");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lite_results_are_parsed_and_unwrapped() {
        let html = r#"
<html><body>
<table class="result">
  <tr><td class="result-snippet">
    <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs&amp;rut=abc">Example Docs</a>
    — Read the manual here.
  </td></tr>
  <tr><td class="result-snippet">
    <a rel="nofollow" href="https://other.example.org/guide">Other Guide</a>
    A second snippet with <b>formatting</b>.
  </td></tr>
</table>
</body></html>"#;
        let results = parse_lite_results(html, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Example Docs");
        assert_eq!(results[0].url, "https://example.com/docs");
        assert!(results[0].snippet.contains("Read the manual here."));
        assert_eq!(results[1].url, "https://other.example.org/guide");
    }

    #[test]
    fn html_results_are_parsed_with_redirect_unwrapping() {
        let html = r#"
<div class="result results_links_deep web_result">
  <h2 class="result__title"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpaper.pdf&amp;rut=abc">A Paper</a></h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpaper.pdf">Read the full paper here.</a>
</div>"#;
        let results = parse_html_results(html, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "A Paper");
        assert_eq!(results[0].url, "https://example.com/paper.pdf");
        assert_eq!(results[0].snippet, "Read the full paper here.");
    }

    #[test]
    fn html_parser_respects_max_results() {
        let html = r#"<a class="result__a" href="https://a.example/1">One</a>
                      <a class="result__a" href="https://a.example/2">Two</a>
                      <a class="result__a" href="https://a.example/3">Three</a>"#;
        assert_eq!(parse_html_results(html, 2).len(), 2);
    }

    #[test]
    fn html_parser_filters_ad_links() {
        let html = r#"<a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Ad</a>
                      <a class="result__a" href="https://real.example/">Real</a>"#;
        let results = parse_html_results(html, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Real");
    }

    #[test]
    fn redirect_urls_keep_literal_plus_signs() {
        // urllib.parse.unquote decodes only %XX; a literal '+' in the target
        // URL is not a space.
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%2Bb%3Fx%3D1%2B2&rut=abc";
        assert_eq!(
            redirect_url(href).as_deref(),
            Some("https://example.com/a+b?x=1+2")
        );
    }

    #[test]
    fn block_detection_flags_the_anti_bot_signals() {
        let ok = reqwest::StatusCode::OK;
        assert!(is_ddg_block(reqwest::StatusCode::ACCEPTED, ""));
        assert!(is_ddg_block(reqwest::StatusCode::FORBIDDEN, ""));
        assert!(is_ddg_block(ok, "   \n  "));
        assert!(!is_ddg_block(ok, "<html><body>results</body></html>"));
        assert!(is_ddg_block(
            ok,
            "<html>Just a moment... enable javascript</html>"
        ));
        assert!(is_ddg_block(ok, "cf-mitigated challenge"));
    }

    #[test]
    fn format_results_matches_the_mcp_shape() {
        let results = vec![
            WebResult {
                title: "Rust docs".into(),
                url: "https://example.com/docs".into(),
                snippet: "Learn Rust.".into(),
            },
            WebResult {
                title: "crates.io".into(),
                url: "https://crates.io".into(),
                snippet: "".into(),
            },
        ];
        let formatted = format_results(&results);
        assert!(formatted.starts_with("Found 2 search results:"));
        assert!(
            formatted.contains(
                "1. Rust docs\n   URL: https://example.com/docs\n   Summary: Learn Rust."
            )
        );
        assert!(formatted.contains("2. crates.io"));
        assert!(format_results(&[]).starts_with("No results were found"));
    }

    #[test]
    fn fetch_content_text_strips_boilerplate_and_paginates() {
        let html = "<html><nav>Nav links</nav><header>Banner</header><main>\
                    <p>Hello brave new world</p></main><footer>Copyright</footer></html>";
        let download = DownloadedUrl {
            final_url: "https://example.com/".parse().unwrap(),
            content_type: "text/html; charset=utf-8".into(),
            bytes: html.as_bytes().to_vec(),
        };
        let first = fetch_content_text(&download, 0, 7).unwrap();
        assert!(first.starts_with("Hello b"));
        assert!(!first.contains("Nav links"));
        assert!(!first.contains("Copyright"));
        assert!(first.contains("Showing characters 0-7 of 21 total"));
        assert!(first.contains("Use start=7 to see more"));

        let rest = fetch_content_text(&download, 7, 50).unwrap();
        assert!(rest.starts_with("rave new world"));
        assert!(!rest.contains("Use start="));
    }

    #[test]
    fn fetch_content_text_rejects_binary_content_types() {
        let download = DownloadedUrl {
            final_url: "https://example.com/a.png".parse().unwrap(),
            content_type: "image/png".into(),
            bytes: vec![0x89, 0x50, 0x4e, 0x47],
        };
        assert!(fetch_content_text(&download, 0, 100).is_err());
    }

    #[test]
    fn percent_encoding_round_trips() {
        let encoded = percent_encode("what is brazier? 100% done");
        assert_eq!(encoded, "what+is+brazier%3F+100%25+done");
        assert_eq!(percent_decode(&encoded), "what is brazier? 100% done");
        assert_eq!(percent_decode("hello%20world"), "hello world");
    }

    #[test]
    fn html_reduction_strips_scripts_and_tags() {
        let html = "<html><head><script>alert(1)</script><style>p{}</style></head>\
                    <body><h1>Title</h1><p>Hello <b>world</b></p></body></html>";
        assert_eq!(html_to_text(html), "Title Hello world");
    }

    #[test]
    fn private_addresses_are_blocked() {
        assert!(!ip_is_public("127.0.0.1".parse().unwrap()));
        assert!(!ip_is_public("10.1.2.3".parse().unwrap()));
        assert!(!ip_is_public("192.168.1.1".parse().unwrap()));
        assert!(!ip_is_public("169.254.169.254".parse().unwrap()));
        assert!(!ip_is_public("100.100.0.1".parse().unwrap()));
        assert!(!ip_is_public("fe80::1".parse().unwrap()));
        assert!(!ip_is_public("::1".parse().unwrap()));
        assert!(!ip_is_public("::ffff:127.0.0.1".parse().unwrap()));
        assert!(!ip_is_public("::ffff:169.254.169.254".parse().unwrap()));
        assert!(!ip_is_public("::ffff:10.0.0.1".parse().unwrap()));
        assert!(!ip_is_public("2001:db8::1".parse().unwrap()));
        assert!(!ip_is_public("ff02::1".parse().unwrap()));
        assert!(ip_is_public("93.184.216.34".parse().unwrap()));
        assert!(ip_is_public(
            "2606:2800:220:1:248:1893:25c8:1946".parse().unwrap()
        ));
    }

    #[test]
    fn ddg_kp_maps_safesearch_levels() {
        assert_eq!(ddg_kp("strict"), 1);
        assert_eq!(ddg_kp("on"), 1);
        assert_eq!(ddg_kp("moderate"), -1);
        assert_eq!(ddg_kp("off"), -2);
        assert_eq!(ddg_kp("bogus"), -1);
    }

    #[test]
    fn rate_gate_limits_bursts_and_expires() {
        let window = Duration::from_millis(200);
        let mut gate = RateGate::new(3, window);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        runtime.block_on(async {
            for _ in 0..3 {
                gate.acquire().await.unwrap();
            }
            assert!(gate.acquire().await.is_err());
            tokio::time::sleep(window).await;
            gate.acquire().await.unwrap();
        });
    }
}

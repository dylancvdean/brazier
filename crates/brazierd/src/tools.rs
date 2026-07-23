//! Bundled, safe built-in tools the daemon can execute on behalf of a model.
//!
//! Tools are intentionally conservative: no filesystem access, no shell, and
//! web retrieval is bounded (size, time, and private-network guard).

use std::time::Duration;

use anyhow::Context;
use serde_json::{Value, json};

const FETCH_MAX_BYTES: usize = 256 * 1024;
const FETCH_MAX_OUTPUT_CHARS: usize = 8_000;
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);

/// A completed built-in tool invocation, suitable for UI display and for the
/// `tool` role message returned to the model.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolInvocation {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub output: String,
    pub is_error: bool,
}

/// OpenAI-style tool definitions for every bundled tool.
pub fn definitions() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current date and time in UTC, plus the Unix timestamp.",
                "parameters": { "type": "object", "properties": {}, "required": [] }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate an arithmetic expression. Supports + - * / % ^, parentheses, and unary minus. Numbers are 64-bit floats.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The expression to evaluate, e.g. `(2 + 3) * 4 ^ 2`."
                        }
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch a public http(s) URL and return its text content. Responses are truncated; HTML is reduced to text. Private and local addresses are blocked.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": { "type": "string", "description": "Absolute http:// or https:// URL." }
                    },
                    "required": ["url"]
                }
            }
        }
    ])
}

/// Human-facing catalog used by the `/api/v1/tools` endpoint.
pub fn catalog() -> Value {
    json!({
        "data": [
            {
                "name": "get_current_time",
                "title": "Current time",
                "description": "Current UTC date, time, and Unix timestamp.",
                "network": false
            },
            {
                "name": "calculator",
                "title": "Calculator",
                "description": "Arithmetic expressions with + - * / % ^ and parentheses.",
                "network": false
            },
            {
                "name": "fetch_url",
                "title": "Web fetch",
                "description": format!(
                    "Bounded retrieval of public web pages (max {} KB, {}s timeout). Local and private addresses are blocked.",
                    FETCH_MAX_BYTES / 1024,
                    FETCH_TIMEOUT.as_secs()
                ),
                "network": true
            }
        ]
    })
}

pub fn is_builtin(name: &str) -> bool {
    matches!(name, "get_current_time" | "calculator" | "fetch_url")
}

/// Execute one bundled tool call. Errors are folded into the output string so
/// the model can react to them; `is_error` marks them for the UI.
pub async fn execute(
    client: &reqwest::Client,
    call_id: &str,
    name: &str,
    arguments: &str,
) -> ToolInvocation {
    let parsed: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
    let result = match name {
        "get_current_time" => Ok(current_time()),
        "calculator" => parsed
            .get("expression")
            .and_then(Value::as_str)
            .context("calculator requires an `expression` string argument")
            .and_then(|expression| {
                evaluate(expression).map(|value| {
                    if value == value.trunc() && value.abs() < 1e15 {
                        format!("{}", value as i64)
                    } else {
                        format!("{value}")
                    }
                })
            }),
        "fetch_url" => match parsed.get("url").and_then(Value::as_str) {
            Some(url) => fetch_url(client, url).await,
            None => Err(anyhow::anyhow!(
                "fetch_url requires a `url` string argument"
            )),
        },
        other => Err(anyhow::anyhow!("unknown built-in tool `{other}`")),
    };
    match result {
        Ok(output) => ToolInvocation {
            call_id: call_id.to_owned(),
            name: name.to_owned(),
            arguments: arguments.to_owned(),
            output,
            is_error: false,
        },
        Err(error) => ToolInvocation {
            call_id: call_id.to_owned(),
            name: name.to_owned(),
            arguments: arguments.to_owned(),
            output: format!("Error: {error:#}"),
            is_error: true,
        },
    }
}

fn current_time() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = now.as_secs() as i64;
    let (year, month, day, hour, minute, second) = civil_from_unix(seconds);
    format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z (Unix timestamp {seconds})"
    )
}

/// Convert a Unix timestamp to civil UTC date-time (Howard Hinnant's algorithm).
fn civil_from_unix(seconds: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let hour = (seconds_of_day / 3_600) as u32;
    let minute = ((seconds_of_day % 3_600) / 60) as u32;
    let second = (seconds_of_day % 60) as u32;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if month <= 2 { year + 1 } else { year };
    (year, month, day, hour, minute, second)
}

// --- Calculator: recursive-descent parser over f64 ---------------------------

struct Parser<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            position: 0,
        }
    }

    fn skip_whitespace(&mut self) {
        while self.position < self.input.len() && self.input[self.position].is_ascii_whitespace() {
            self.position += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_whitespace();
        self.input.get(self.position).copied()
    }

    fn expression(&mut self) -> anyhow::Result<f64> {
        let mut value = self.term()?;
        while let Some(op) = self.peek() {
            match op {
                b'+' => {
                    self.position += 1;
                    value += self.term()?;
                }
                b'-' => {
                    self.position += 1;
                    value -= self.term()?;
                }
                _ => break,
            }
        }
        Ok(value)
    }

    fn term(&mut self) -> anyhow::Result<f64> {
        let mut value = self.power()?;
        while let Some(op) = self.peek() {
            match op {
                b'*' => {
                    self.position += 1;
                    value *= self.power()?;
                }
                b'/' => {
                    self.position += 1;
                    value /= self.power()?;
                }
                b'%' => {
                    self.position += 1;
                    value %= self.power()?;
                }
                _ => break,
            }
        }
        Ok(value)
    }

    fn power(&mut self) -> anyhow::Result<f64> {
        let base = self.unary()?;
        if self.peek() == Some(b'^') {
            self.position += 1;
            // Right-associative.
            let exponent = self.power()?;
            return Ok(base.powf(exponent));
        }
        Ok(base)
    }

    fn unary(&mut self) -> anyhow::Result<f64> {
        match self.peek() {
            Some(b'-') => {
                self.position += 1;
                Ok(-self.unary()?)
            }
            Some(b'+') => {
                self.position += 1;
                self.unary()
            }
            _ => self.atom(),
        }
    }

    fn atom(&mut self) -> anyhow::Result<f64> {
        match self.peek() {
            Some(b'(') => {
                self.position += 1;
                let value = self.expression()?;
                anyhow::ensure!(self.peek() == Some(b')'), "expected closing parenthesis");
                self.position += 1;
                Ok(value)
            }
            Some(byte) if byte.is_ascii_digit() || byte == b'.' => {
                let start = self.position;
                while self
                    .input
                    .get(self.position)
                    .is_some_and(|b| b.is_ascii_digit() || *b == b'.' || *b == b'e' || *b == b'E')
                {
                    // Allow exponent sign directly after e/E.
                    if matches!(self.input[self.position], b'e' | b'E')
                        && matches!(self.input.get(self.position + 1), Some(b'+') | Some(b'-'))
                    {
                        self.position += 1;
                    }
                    self.position += 1;
                }
                let text = std::str::from_utf8(&self.input[start..self.position])?;
                text.parse::<f64>()
                    .map_err(|_| anyhow::anyhow!("invalid number `{text}`"))
            }
            Some(byte) => anyhow::bail!("unexpected character `{}`", byte as char),
            None => anyhow::bail!("unexpected end of expression"),
        }
    }
}

/// Evaluate an arithmetic expression.
pub fn evaluate(expression: &str) -> anyhow::Result<f64> {
    anyhow::ensure!(expression.len() <= 1_000, "expression is too long");
    let mut parser = Parser::new(expression);
    let value = parser.expression()?;
    parser.skip_whitespace();
    anyhow::ensure!(
        parser.position == parser.input.len(),
        "unexpected trailing input in expression"
    );
    anyhow::ensure!(
        value.is_finite(),
        "expression did not evaluate to a finite number"
    );
    Ok(value)
}

// --- Bounded web retrieval ----------------------------------------------------

fn ip_is_public(ip: std::net::IpAddr) -> bool {
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
            !(v6.is_loopback()
                || v6.is_unspecified()
                || (v6.segments()[0] & 0xfe00) == 0xfc00 // unique local fc00::/7
                || (v6.segments()[0] & 0xffc0) == 0xfe80) // link local fe80::/10
        }
    }
}

async fn guard_host(host: &str) -> anyhow::Result<()> {
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        anyhow::ensure!(ip_is_public(ip), "address {ip} is not publicly routable");
        return Ok(());
    }
    anyhow::ensure!(
        !host.eq_ignore_ascii_case("localhost") && !host.ends_with(".local"),
        "local hostnames are not allowed"
    );
    let addresses = tokio::net::lookup_host((host, 443))
        .await
        .with_context(|| format!("resolve host {host}"))?;
    for address in addresses {
        anyhow::ensure!(
            ip_is_public(address.ip()),
            "host {host} resolves to non-public address {}",
            address.ip()
        );
    }
    Ok(())
}

/// Strip tags from HTML and collapse whitespace. Deliberately simple.
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
            if lower[index..].starts_with("<script") {
                skip_until = Some("</script");
                continue;
            }
            if lower[index..].starts_with("<style") {
                skip_until = Some("</style");
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

async fn fetch_url(client: &reqwest::Client, url: &str) -> anyhow::Result<String> {
    let parsed = reqwest::Url::parse(url).context("invalid URL")?;
    anyhow::ensure!(
        matches!(parsed.scheme(), "http" | "https"),
        "only http and https URLs are supported"
    );
    let host = parsed.host_str().context("URL has no host")?;
    guard_host(host).await?;

    let response = client
        .get(parsed)
        .header("user-agent", "brazier-tools/0.1 (+bounded-fetch)")
        .timeout(FETCH_TIMEOUT)
        .send()
        .await
        .context("request failed")?;
    let status = response.status();
    anyhow::ensure!(status.is_success(), "server returned {status}");
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    anyhow::ensure!(
        content_type.is_empty()
            || content_type.contains("text/")
            || content_type.contains("json")
            || content_type.contains("xml"),
        "unsupported content type `{content_type}`"
    );

    use futures::StreamExt;
    let mut stream = response.bytes_stream();
    let mut bytes: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read response body")?;
        bytes.extend_from_slice(&chunk);
        if bytes.len() >= FETCH_MAX_BYTES {
            bytes.truncate(FETCH_MAX_BYTES);
            break;
        }
    }
    let body = String::from_utf8_lossy(&bytes);
    let mut text = if content_type.contains("html") {
        html_to_text(&body)
    } else {
        body.into_owned()
    };
    if text.len() > FETCH_MAX_OUTPUT_CHARS {
        let mut cut = FETCH_MAX_OUTPUT_CHARS;
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        text.truncate(cut);
        text.push_str("… [truncated]");
    }
    anyhow::ensure!(!text.is_empty(), "response contained no text");
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculator_handles_precedence_and_parens() {
        assert_eq!(evaluate("2 + 3 * 4").unwrap(), 14.0);
        assert_eq!(evaluate("(2 + 3) * 4").unwrap(), 20.0);
        assert_eq!(evaluate("2 ^ 3 ^ 2").unwrap(), 512.0); // right-assoc
        assert_eq!(evaluate("-4 + 10 % 3").unwrap(), -3.0);
        assert_eq!(evaluate("1.5e2 / 3").unwrap(), 50.0);
    }

    #[test]
    fn calculator_rejects_garbage() {
        assert!(evaluate("2 +").is_err());
        assert!(evaluate("(1").is_err());
        assert!(evaluate("2; rm -rf /").is_err());
        assert!(evaluate("1/0").is_err()); // infinity is rejected
    }

    #[test]
    fn time_tool_formats_a_known_timestamp() {
        assert_eq!(civil_from_unix(0), (1970, 1, 1, 0, 0, 0));
        assert_eq!(civil_from_unix(1_753_000_000), (2025, 7, 20, 8, 26, 40));
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
        assert!(ip_is_public("93.184.216.34".parse().unwrap()));
        assert!(ip_is_public(
            "2606:2800:220:1:248:1893:25c8:1946".parse().unwrap()
        ));
    }

    #[tokio::test]
    async fn execute_reports_unknown_tool_as_error() {
        let client = reqwest::Client::new();
        let result = execute(&client, "call_1", "launch_missiles", "{}").await;
        assert!(result.is_error);
        assert!(result.output.contains("unknown"));
    }

    #[tokio::test]
    async fn execute_calculator_round_trip() {
        let client = reqwest::Client::new();
        let result = execute(&client, "call_1", "calculator", "{\"expression\": \"6*7\"}").await;
        assert!(!result.is_error);
        assert_eq!(result.output, "42");
    }
}

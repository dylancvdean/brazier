//! Model Context Protocol client and server configuration.
//!
//! Brazier connects to MCP servers over stdio, caches their tool schemas, and
//! exposes them to models using OpenAI-style function definitions.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
};

use crate::{toolchain_hints::resolve_command, tools::ToolInvocation};

const PROTOCOL_VERSION: &str = "2024-11-05";
const TOOL_PREFIX: &str = "mcp/";
/// Handshake bound. A server that never answers initialize must not pin the tool.
const INITIALIZE_TIMEOUT: Duration = Duration::from_secs(30);
/// Bound for `tools/list` and `tools/call`. Longer work belongs in the tool itself.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Substrings that mark an environment variable as credential-bearing.
const SECRET_ENV_PATTERNS: &[&str] = &[
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "APIKEY",
    "API_KEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
    "SESSION_KEY",
    "AUTH",
    "COOKIE",
    "LICENSE_KEY",
];

const DENIED_ENV_NAMES: &[&str] = &[
    "AWS_ACCESS_KEY_ID",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "SSH_AUTH_SOCK",
    "SSH_AGENT_PID",
    "GPG_AGENT_INFO",
    "NETRC",
    "npm_config_registry",
    "PIP_INDEX_URL",
];

fn is_secret_env(name: &str) -> bool {
    if DENIED_ENV_NAMES.iter().any(|denied| denied == &name) {
        return true;
    }
    let upper = name.to_ascii_uppercase();
    SECRET_ENV_PATTERNS
        .iter()
        .any(|pattern| upper.contains(pattern))
}

/// Environment inherited by an MCP child: host vars minus secrets and daemon
/// config, then the server's explicit `env` map (which may re-add granted
/// credentials).
fn mcp_child_env(extra: &HashMap<String, String>) -> HashMap<String, String> {
    let mut env = HashMap::new();
    for (key, value) in std::env::vars() {
        if is_secret_env(&key) || key.starts_with("BRAZIER_") {
            continue;
        }
        env.insert(key, value);
    }
    for (key, value) in extra {
        env.insert(key.clone(), value.clone());
    }
    env
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpToolEntry {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpServerConfig {
    pub id: String,
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<McpToolEntry>,
}

fn default_enabled() -> bool {
    true
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct McpConfig {
    pub servers: Vec<McpServerConfig>,
}

impl McpConfig {
    pub fn find_mut(&mut self, id: &str) -> Option<&mut McpServerConfig> {
        self.servers.iter_mut().find(|server| server.id == id)
    }

    pub fn find(&self, id: &str) -> Option<&McpServerConfig> {
        self.servers.iter().find(|server| server.id == id)
    }
}

pub fn config_path(data_dir: &Path) -> PathBuf {
    data_dir.join("mcp-servers.json")
}

pub fn load(data_dir: &Path) -> McpConfig {
    let path = config_path(data_dir);
    let Ok(bytes) = std::fs::read(&path) else {
        return McpConfig::default();
    };
    serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        tracing::warn!(%error, path = %path.display(), "ignoring invalid MCP config");
        McpConfig::default()
    })
}

pub async fn save(data_dir: &Path, config: &McpConfig) -> anyhow::Result<()> {
    let path = config_path(data_dir);
    crate::persistence::write_json(&path, config, "MCP config").await
}

pub fn enabled_servers(data_dir: &Path) -> Vec<McpServerConfig> {
    load(data_dir)
        .servers
        .into_iter()
        .filter(|server| server.enabled)
        .collect()
}

pub fn openai_tool_name(server_id: &str, tool_name: &str) -> String {
    format!("{TOOL_PREFIX}{server_id}/{tool_name}")
}

pub fn parse_tool_name(full_name: &str) -> Option<(String, String)> {
    let rest = full_name.strip_prefix(TOOL_PREFIX)?;
    let (server_id, tool_name) = rest.split_once('/')?;
    if server_id.is_empty() || tool_name.is_empty() {
        return None;
    }
    Some((server_id.to_owned(), tool_name.to_owned()))
}

pub fn tool_to_openai(server_id: &str, tool: &McpToolEntry) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": openai_tool_name(server_id, &tool.name),
            "description": tool.description.clone().unwrap_or_else(|| tool.name.clone()),
            "parameters": if tool.input_schema.is_null() {
                json!({ "type": "object", "properties": {} })
            } else {
                tool.input_schema.clone()
            }
        }
    })
}

pub fn catalog(data_dir: &Path) -> Value {
    let servers = load(data_dir).servers;
    json!({
        "data": servers.iter().map(|server| {
            json!({
                "id": server.id,
                "name": server.name,
                "command": server.command,
                "args": server.args,
                "enabled": server.enabled,
                "tools": server.tools.iter().map(|tool| {
                    json!({
                        "name": openai_tool_name(&server.id, &tool.name),
                        "title": tool.name,
                        "description": tool.description,
                        "network": true,
                        "source": "mcp",
                        "server_name": server.name
                    })
                }).collect::<Vec<_>>()
            })
        }).collect::<Vec<_>>()
    })
}

struct JsonRpcClient {
    /// Retain ownership for the full RPC exchange. `kill_on_drop` then cleans
    /// up the server when this client is done instead of killing it at the end
    /// of `connect`.
    _child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    reader: BufReader<tokio::process::ChildStdout>,
    next_id: u64,
    initialize_timeout: Duration,
    request_timeout: Duration,
}

/// A stdout line from an MCP server. Logs and blank lines are not messages.
fn decode_stdout_line(line: &str) -> Option<Value> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    serde_json::from_str(trimmed).ok()
}

enum Inbound {
    /// The response to the request we just sent.
    Response,
    /// A request the server started. `id` is answered with method-not-found.
    ServerRequest(Value),
    /// A notification, or a response for some other id.
    Ignore,
}

fn classify_inbound(message: &Value, expected_id: u64) -> Inbound {
    if message.get("method").is_some() {
        return match message.get("id") {
            Some(id) if !id.is_null() => Inbound::ServerRequest(id.clone()),
            _ => Inbound::Ignore,
        };
    }
    if message.get("id") == Some(&json!(expected_id)) {
        Inbound::Response
    } else {
        Inbound::Ignore
    }
}

impl JsonRpcClient {
    async fn connect(config: &McpServerConfig) -> anyhow::Result<Self> {
        Self::connect_timeouts(config, INITIALIZE_TIMEOUT, REQUEST_TIMEOUT).await
    }

    async fn connect_timeouts(
        config: &McpServerConfig,
        initialize_timeout: Duration,
        request_timeout: Duration,
    ) -> anyhow::Result<Self> {
        // GUI-launched Electron apps do not inherit the user's shell PATH.
        // Resolve named commands through the runtime's user-scoped locations
        // so Homebrew-installed MCP launchers such as `uvx` work on macOS.
        let executable = if Path::new(&config.command).is_absolute() {
            PathBuf::from(&config.command)
        } else {
            resolve_command(&config.command).unwrap_or_else(|| PathBuf::from(&config.command))
        };
        let mut command = Command::new(executable);
        command
            .args(&config.args)
            .env_clear()
            .envs(mcp_child_env(&config.env))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let mut child = command
            .spawn()
            .with_context(|| format!("spawn MCP server `{}` ({})", config.name, config.command))?;
        let stdin = child
            .stdin
            .take()
            .context("MCP server process has no stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("MCP server process has no stdout")?;
        Self::from_child(child, stdin, stdout, initialize_timeout, request_timeout).await
    }

    async fn from_child(
        child: tokio::process::Child,
        stdin: tokio::process::ChildStdin,
        stdout: tokio::process::ChildStdout,
        initialize_timeout: Duration,
        request_timeout: Duration,
    ) -> anyhow::Result<Self> {
        let mut client = Self {
            _child: child,
            stdin,
            reader: BufReader::new(stdout),
            next_id: 1,
            initialize_timeout,
            request_timeout,
        };
        client.initialize().await?;
        Ok(client)
    }

    async fn write_line(&mut self, value: &Value) -> anyhow::Result<()> {
        let mut line = serde_json::to_string(value).context("encode JSON-RPC message")?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn read_message(&mut self) -> anyhow::Result<Value> {
        loop {
            let mut line = String::new();
            let read = self.reader.read_line(&mut line).await?;
            if read == 0 {
                anyhow::bail!("MCP server closed stdout");
            }
            // Servers sometimes log on stdout. A non-JSON line is not a frame.
            if let Some(message) = decode_stdout_line(&line) {
                return Ok(message);
            }
        }
    }

    async fn request(&mut self, method: &str, params: Value) -> anyhow::Result<Value> {
        let timeout = if method == "initialize" {
            self.initialize_timeout
        } else {
            self.request_timeout
        };
        match tokio::time::timeout(timeout, self.exchange(method, params)).await {
            Ok(result) => result,
            Err(_) => anyhow::bail!("MCP `{method}` timed out after {}ms", timeout.as_millis()),
        }
    }

    async fn exchange(&mut self, method: &str, params: Value) -> anyhow::Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        self.write_line(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        }))
        .await?;
        loop {
            let message = self.read_message().await?;
            match classify_inbound(&message, id) {
                Inbound::ServerRequest(request_id) => {
                    // Leaving this unanswered makes servers that wait for a
                    // reply stall the whole session. We do not implement their
                    // methods, so the honest answer is that we have none.
                    self.write_line(&json!({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": { "code": -32601, "message": "Method not found" }
                    }))
                    .await?;
                }
                Inbound::Ignore => {}
                Inbound::Response => {
                    if let Some(error) = message.get("error") {
                        anyhow::bail!("MCP error: {error}");
                    }
                    return Ok(message.get("result").cloned().unwrap_or(Value::Null));
                }
            }
        }
    }

    async fn initialize(&mut self) -> anyhow::Result<()> {
        self.request(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "brazier",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        )
        .await?;
        self.write_line(&json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }))
        .await?;
        Ok(())
    }

    async fn list_tools(&mut self) -> anyhow::Result<Vec<McpToolEntry>> {
        let result = self.request("tools/list", json!({})).await?;
        let tools = result
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut entries = Vec::new();
        for tool in tools {
            let Some(name) = tool.get("name").and_then(Value::as_str) else {
                continue;
            };
            entries.push(McpToolEntry {
                name: name.to_owned(),
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                input_schema: tool
                    .get("inputSchema")
                    .cloned()
                    .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
            });
        }
        Ok(entries)
    }

    async fn call_tool(&mut self, name: &str, arguments: Value) -> anyhow::Result<Value> {
        let result = self
            .request(
                "tools/call",
                json!({ "name": name, "arguments": arguments }),
            )
            .await?;
        if result.get("isError").and_then(Value::as_bool) == Some(true) {
            anyhow::bail!("{}", content_to_text(&result));
        }
        Ok(result)
    }
}

fn content_to_text(result: &Value) -> String {
    let Some(content) = result.get("content").and_then(Value::as_array) else {
        return result.to_string();
    };
    let mut parts = Vec::new();
    for item in content {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    parts.push(text.to_owned());
                }
            }
            Some(other) => {
                if other == "resource_link" {
                    let name = item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("resource");
                    let uri = item.get("uri").and_then(Value::as_str).unwrap_or("");
                    parts.push(format!("[{name}] {uri}"));
                } else if other == "resource" {
                    let uri = item
                        .pointer("/resource/uri")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    parts.push(format!("[resource] {uri}"));
                } else {
                    parts.push(format!("[{other} content omitted]"));
                }
            }
            None => {}
        }
    }
    if parts.is_empty() {
        result.to_string()
    } else {
        parts.join("\n")
    }
}

const MAX_AUTOMATIC_PDF_CANDIDATES: usize = 3;

fn pdf_url_candidate(value: &str) -> Option<String> {
    let Ok(url) = reqwest::Url::parse(value) else {
        return None;
    };
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }
    let path = url.path().to_ascii_lowercase();
    if path.ends_with(".pdf") {
        return Some(value.to_owned());
    }
    // Search providers sometimes wrap the real result in a redirect URL such
    // as `...?uddg=https%3A%2F%2Fhost%2Fpaper.pdf`.
    url.query_pairs()
        .map(|(_, value)| value.into_owned())
        .find_map(|value| pdf_url_candidate(&value))
        .or_else(|| {
            url.query()
                .is_some_and(|query| query.to_ascii_lowercase().contains(".pdf"))
                .then_some(value.to_owned())
        })
}

fn candidate_urls(text: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    for token in text.split_whitespace() {
        let mut rest = token;
        while let Some(index) = rest.find("http") {
            let candidate = rest[index..].trim_matches(|character: char| {
                matches!(
                    character,
                    '"' | '\'' | '<' | '>' | '[' | ']' | '(' | ')' | '{' | '}' | ',' | ';' | '.'
                )
            });
            if let Some(candidate) = pdf_url_candidate(candidate)
                && !candidates.iter().any(|url| url == &candidate)
            {
                candidates.push(candidate);
            }
            if candidates.len() == MAX_AUTOMATIC_PDF_CANDIDATES {
                break;
            }
            rest = &rest[index + 4..];
        }
        if candidates.len() == MAX_AUTOMATIC_PDF_CANDIDATES {
            break;
        }
    }
    candidates
}

fn embedded_pdf_resources(result: &Value) -> Vec<(String, Vec<u8>)> {
    let Some(content) = result.get("content").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut resources = Vec::new();
    for item in content {
        let Some(resource) = item.get("resource") else {
            continue;
        };
        if resource
            .get("mimeType")
            .and_then(Value::as_str)
            .is_none_or(|mime| mime != "application/pdf")
        {
            continue;
        }
        let Some(blob) = resource.get("blob").and_then(Value::as_str) else {
            continue;
        };
        let Ok(bytes) = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, blob)
        else {
            continue;
        };
        let name = resource
            .get("uri")
            .and_then(Value::as_str)
            .and_then(|uri| uri.rsplit('/').next())
            .filter(|name| !name.is_empty())
            .unwrap_or("mcp-resource.pdf")
            .to_owned();
        resources.push((name, bytes));
    }
    resources
}

pub async fn refresh_tools(data_dir: &Path, server_id: &str) -> anyhow::Result<Vec<McpToolEntry>> {
    let mut config = load(data_dir);
    let snapshot = config
        .find(server_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("unknown MCP server `{server_id}`"))?;
    let mut client = JsonRpcClient::connect(&snapshot).await?;
    let tools = client.list_tools().await?;
    if let Some(server) = config.find_mut(server_id) {
        server.tools = tools.clone();
    }
    save(data_dir, &config).await?;
    Ok(tools)
}

pub async fn call_tool(
    data_dir: &Path,
    server_id: &str,
    tool_name: &str,
    arguments: &str,
) -> ToolInvocation {
    let call_id = uuid::Uuid::new_v4().simple().to_string();
    let config = load(data_dir);
    let Some(server) = config.find(server_id) else {
        return ToolInvocation {
            call_id,
            name: openai_tool_name(server_id, tool_name),
            arguments: arguments.to_owned(),
            output: format!("Error: unknown MCP server `{server_id}`"),
            is_error: true,
            media: Vec::new(),
        };
    };
    if !server.enabled {
        return ToolInvocation {
            call_id,
            name: openai_tool_name(server_id, tool_name),
            arguments: arguments.to_owned(),
            output: format!("Error: MCP server `{server_id}` is disabled"),
            is_error: true,
            media: Vec::new(),
        };
    }
    if !server.tools.iter().any(|tool| tool.name == tool_name) {
        return ToolInvocation {
            call_id,
            name: openai_tool_name(server_id, tool_name),
            arguments: arguments.to_owned(),
            output: format!(
                "Error: MCP server `{server_id}` does not advertise tool `{tool_name}`"
            ),
            is_error: true,
            media: Vec::new(),
        };
    }
    let parsed_args: Value = match crate::tool_registry::parse_json_arguments(arguments) {
        Ok(value) => value,
        Err(error) => {
            let looks_like_json =
                arguments.trim_start().starts_with('{') || arguments.trim_start().starts_with('[');
            if looks_like_json {
                return ToolInvocation {
                    call_id,
                    name: openai_tool_name(server_id, tool_name),
                    arguments: arguments.to_owned(),
                    output: format!("Error: {error:#}"),
                    is_error: true,
                    media: Vec::new(),
                };
            }
            json!({ "input": arguments })
        }
    };
    match JsonRpcClient::connect(server).await {
        Ok(mut client) => match client.call_tool(tool_name, parsed_args).await {
            Ok(result) => {
                let mut output = content_to_text(&result);
                let mut media = Vec::new();
                let serialized_result = serde_json::to_string(&result).unwrap_or_default();
                for url in candidate_urls(&format!("{output} {serialized_result}")) {
                    match crate::tools::fetch_pdf_candidate(data_dir, &url).await {
                        Ok(Some((document_output, document_media))) => {
                            output.push_str(&format!("\n\n[MCP PDF handoff]\n{document_output}"));
                            media.extend(document_media);
                        }
                        Ok(None) => {}
                        Err(error) => {
                            output.push_str(&format!(
                                "\n\n[PDF handoff failed for {url}: {error:#}]"
                            ));
                        }
                    }
                }
                for (name, bytes) in embedded_pdf_resources(&result) {
                    match crate::tools::ingest_pdf_bytes(data_dir, &bytes, &name, &json!({})).await
                    {
                        Ok(document) => {
                            output.push_str(&format!("\n\n[MCP PDF resource]\n{}", document.text));
                            media.extend(document.media);
                        }
                        Err(error) => output.push_str(&format!(
                            "\n\n[PDF resource handoff failed for {name}: {error:#}]"
                        )),
                    }
                }
                ToolInvocation {
                    call_id,
                    name: openai_tool_name(server_id, tool_name),
                    arguments: arguments.to_owned(),
                    output,
                    is_error: false,
                    media,
                }
            }
            Err(error) => ToolInvocation {
                call_id,
                name: openai_tool_name(server_id, tool_name),
                arguments: arguments.to_owned(),
                output: format!("Error: {error:#}"),
                is_error: true,
                media: Vec::new(),
            },
        },
        Err(error) => ToolInvocation {
            call_id,
            name: openai_tool_name(server_id, tool_name),
            arguments: arguments.to_owned(),
            output: format!("Error: {error:#}"),
            is_error: true,
            media: Vec::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn round_trips_namespaced_tool_names() {
        let full = openai_tool_name("filesystem", "read_file");
        assert_eq!(full, "mcp/filesystem/read_file");
        assert_eq!(
            parse_tool_name(&full),
            Some(("filesystem".into(), "read_file".into()))
        );
    }

    #[test]
    fn extracts_pdf_urls_from_plain_and_markdown_text() {
        let urls = candidate_urls(
            "See https://example.com/paper.pdf, [the report](https://example.com/report.PDF).",
        );
        assert_eq!(
            urls,
            vec![
                "https://example.com/paper.pdf".to_owned(),
                "https://example.com/report.PDF".to_owned()
            ]
        );
    }

    #[test]
    fn extracts_pdf_url_from_a_search_redirect() {
        let urls =
            candidate_urls("https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpaper.pdf");
        assert_eq!(urls, vec!["https://example.com/paper.pdf"]);
    }

    #[test]
    fn preserves_mcp_resource_links_as_text() {
        let result = json!({
            "content": [{
                "type": "resource_link",
                "name": "paper",
                "uri": "https://example.com/paper.pdf"
            }]
        });
        let text = content_to_text(&result);
        assert!(text.contains("paper"));
        assert!(text.contains("paper.pdf"));
    }

    #[test]
    fn finds_embedded_pdf_resources() {
        let result = json!({
            "content": [{
                "type": "resource",
                "resource": {
                    "uri": "https://example.com/paper.pdf",
                    "mimeType": "application/pdf",
                    "blob": "JVBERi0xLjQ="
                }
            }]
        });
        let resources = embedded_pdf_resources(&result);
        assert_eq!(
            resources,
            vec![("paper.pdf".to_owned(), b"%PDF-1.4".to_vec())]
        );
    }

    #[tokio::test]
    async fn calls_only_enabled_advertised_tools() {
        let dir = tempdir().unwrap();
        let mut config = McpConfig {
            servers: vec![McpServerConfig {
                id: "demo".into(),
                name: "Demo".into(),
                command: "/definitely/not/a/program".into(),
                args: Vec::new(),
                env: HashMap::new(),
                enabled: false,
                tools: vec![McpToolEntry {
                    name: "ping".into(),
                    description: None,
                    input_schema: json!({ "type": "object" }),
                }],
            }],
        };
        save(dir.path(), &config).await.unwrap();
        let disabled = call_tool(dir.path(), "demo", "ping", "{}").await;
        assert!(disabled.is_error);
        assert!(disabled.output.contains("disabled"), "{}", disabled.output);

        config.servers[0].enabled = true;
        save(dir.path(), &config).await.unwrap();
        let unknown = call_tool(dir.path(), "demo", "other", "{}").await;
        assert!(unknown.is_error);
        assert!(
            unknown.output.contains("does not advertise"),
            "{}",
            unknown.output
        );
    }

    #[test]
    fn stdout_logs_are_not_messages_and_server_requests_are_rejected() {
        assert!(decode_stdout_line("").is_none());
        assert!(decode_stdout_line("listening on stdio").is_none());
        assert!(decode_stdout_line(r#"{"jsonrpc":"2.0","id":1,"result":{}}"#).is_some());

        let request = json!({"jsonrpc": "2.0", "id": 7, "method": "roots/list"});
        assert!(matches!(
            classify_inbound(&request, 1),
            Inbound::ServerRequest(_)
        ));
        let notification = json!({"jsonrpc": "2.0", "method": "notifications/progress"});
        assert!(matches!(
            classify_inbound(&notification, 1),
            Inbound::Ignore
        ));
        let response = json!({"jsonrpc": "2.0", "id": 1, "result": {"ok": true}});
        assert!(matches!(classify_inbound(&response, 1), Inbound::Response));
    }

    fn server_config(
        command: &str,
        args: Vec<String>,
        env: HashMap<String, String>,
    ) -> McpServerConfig {
        McpServerConfig {
            id: "demo".into(),
            name: "Demo".into(),
            command: command.into(),
            args,
            env,
            enabled: true,
            tools: vec![McpToolEntry {
                name: "ping".into(),
                description: None,
                input_schema: json!({ "type": "object" }),
            }],
        }
    }

    #[tokio::test]
    async fn initialize_times_out_when_the_server_never_answers() {
        let dir = tempdir().unwrap();
        let script = dir.path().join("hang.py");
        std::fs::write(&script, "import time\ntime.sleep(30)\n").unwrap();
        let config = server_config(
            "python3",
            vec![script.display().to_string()],
            HashMap::new(),
        );
        match JsonRpcClient::connect_timeouts(
            &config,
            Duration::from_millis(200),
            Duration::from_millis(200),
        )
        .await
        {
            Ok(_) => panic!("a silent server must not pin the client"),
            Err(error) => assert!(error.to_string().contains("timed out"), "{error:#}"),
        }
    }

    #[tokio::test]
    async fn skips_stdout_logs_and_answers_server_requests() {
        let dir = tempdir().unwrap();
        let script = dir.path().join("server.py");
        std::fs::write(
            &script,
            r#"
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    print("log: " + str(method), flush=True)
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "roots/list"}), flush=True)
        reply = json.loads(sys.stdin.readline())
        assert reply["id"] == 7, reply
        assert reply["error"]["code"] == -32601, reply
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": msg["id"],
            "result": {"protocolVersion": "2024-11-05", "capabilities": {}}
        }), flush=True)
    elif method == "tools/call":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": msg["id"],
            "result": {"content": [{"type": "text", "text": "pong"}]}
        }), flush=True)
"#,
        )
        .unwrap();
        let config = server_config(
            "python3",
            vec![script.display().to_string()],
            HashMap::new(),
        );
        let mut client = JsonRpcClient::connect(&config)
            .await
            .expect("server should complete the handshake");
        let result = client
            .call_tool("ping", json!({}))
            .await
            .expect("tool call");
        assert_eq!(content_to_text(&result), "pong");
    }
}

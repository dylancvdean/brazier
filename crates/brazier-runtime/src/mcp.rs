//! Model Context Protocol client and server configuration.
//!
//! Brazier connects to MCP servers over stdio, caches their tool schemas, and
//! exposes them to models using OpenAI-style function definitions.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Stdio,
};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
};

use crate::tools::ToolInvocation;

const PROTOCOL_VERSION: &str = "2024-11-05";
const TOOL_PREFIX: &str = "mcp/";

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
}

impl JsonRpcClient {
    async fn connect(config: &McpServerConfig) -> anyhow::Result<Self> {
        let mut command = Command::new(&config.command);
        command
            .args(&config.args)
            .envs(&config.env)
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
        let mut client = Self {
            _child: child,
            stdin,
            reader: BufReader::new(stdout),
            next_id: 1,
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
        let mut line = String::new();
        self.reader.read_line(&mut line).await?;
        anyhow::ensure!(!line.trim().is_empty(), "MCP server closed stdout");
        serde_json::from_str(&line).context("decode JSON-RPC message")
    }

    async fn request(&mut self, method: &str, params: Value) -> anyhow::Result<Value> {
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
            if message.get("id") == Some(&json!(id)) {
                if let Some(error) = message.get("error") {
                    anyhow::bail!("MCP error: {error}");
                }
                return Ok(message.get("result").cloned().unwrap_or(Value::Null));
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

    async fn call_tool(&mut self, name: &str, arguments: Value) -> anyhow::Result<String> {
        let result = self
            .request(
                "tools/call",
                json!({ "name": name, "arguments": arguments }),
            )
            .await?;
        if result.get("isError").and_then(Value::as_bool) == Some(true) {
            anyhow::bail!("{}", content_to_text(&result));
        }
        Ok(content_to_text(&result))
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
                parts.push(format!("[{other} content omitted]"));
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
    let parsed_args: Value =
        serde_json::from_str(arguments).unwrap_or_else(|_| json!({ "input": arguments }));
    match JsonRpcClient::connect(server).await {
        Ok(mut client) => match client.call_tool(tool_name, parsed_args).await {
            Ok(output) => ToolInvocation {
                call_id,
                name: openai_tool_name(server_id, tool_name),
                arguments: arguments.to_owned(),
                output,
                is_error: false,
                media: Vec::new(),
            },
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
}

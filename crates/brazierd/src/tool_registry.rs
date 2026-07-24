//! Unified tool definitions and execution for bundled and MCP tools.

use std::path::Path;

use reqwest::Client;
use serde_json::{Value, json};

use crate::{harmony, mcp, tools::ToolInvocation};

pub struct ToolContext<'a> {
    pub data_dir: &'a Path,
    pub http: &'a Client,
}

/// Merge bundled, request, and MCP tool definitions into one OpenAI-style array.
pub fn merge_definitions(
    data_dir: &Path,
    request: &crate::types::ChatCompletionRequest,
    harmony: bool,
) -> Option<Value> {
    if !request.tools_active() {
        return None;
    }

    let mut defs: Vec<Value> = Vec::new();
    let tools_on = request.builtin_tools.unwrap_or(false);

    if tools_on {
        if let Some(items) = crate::tools::definitions().as_array() {
            defs.extend(items.iter().cloned());
        }
        for server in mcp::enabled_servers(data_dir) {
            for tool in &server.tools {
                defs.push(mcp::tool_to_openai(&server.id, tool));
            }
        }
    }

    if let Some(custom) = &request.tools {
        match custom {
            Value::Array(items) => defs.extend(items.iter().cloned()),
            other => defs.push(other.clone()),
        }
    }

    if defs.is_empty() {
        None
    } else {
        Some(Value::Array(
            defs.iter()
                .map(|def| harmony::adapt_tool_definition(def, harmony))
                .collect(),
        ))
    }
}

pub fn tools_enabled(
    data_dir: &Path,
    request: &crate::types::ChatCompletionRequest,
    harmony: bool,
) -> bool {
    merge_definitions(data_dir, request, harmony).is_some()
}

pub fn can_execute(name: &str) -> bool {
    let logical = harmony::logical_tool_name(name);
    crate::tools::is_builtin(&logical) || mcp::parse_tool_name(&logical).is_some()
}

pub async fn execute(
    ctx: &ToolContext<'_>,
    call_id: &str,
    name: &str,
    arguments: &str,
) -> ToolInvocation {
    let logical = harmony::logical_tool_name(name);
    if crate::tools::is_builtin(&logical) {
        return crate::tools::execute(ctx.http, call_id, &logical, arguments).await;
    }
    if let Some((server_id, tool_name)) = mcp::parse_tool_name(&logical) {
        let mut invocation = mcp::call_tool(ctx.data_dir, &server_id, &tool_name, arguments).await;
        invocation.call_id = call_id.to_owned();
        invocation.name = name.to_owned();
        return invocation;
    }
    ToolInvocation {
        call_id: call_id.to_owned(),
        name: name.to_owned(),
        arguments: arguments.to_owned(),
        output: format!("Error: no server-side handler for tool `{name}`"),
        is_error: true,
    }
}

pub fn combined_catalog(data_dir: &Path) -> Value {
    let mut entries: Vec<Value> = Vec::new();
    if let Some(builtin) = crate::tools::catalog().get("data").and_then(Value::as_array) {
        entries.extend(builtin.iter().cloned());
    }
    if let Some(mcp_items) = mcp::catalog(data_dir).get("data").and_then(Value::as_array) {
        for server in mcp_items {
            if let Some(tools) = server.get("tools").and_then(Value::as_array) {
                entries.extend(tools.iter().cloned());
            }
        }
    }
    json!({ "data": entries })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn merge_requires_tools_active() {
        let dir = tempdir().unwrap();
        let request = crate::types::ChatCompletionRequest {
            model: "gguf:test".into(),
            messages: Vec::new(),
            stream: false,
            tools: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            seed: None,
            enable_reasoning: None,
            reasoning_budget_tokens: None,
            tool_choice: None,
            builtin_tools: None,
        };
        assert!(merge_definitions(dir.path(), &request, false).is_none());
    }

    #[test]
    fn merge_includes_builtin_and_mcp_when_enabled() {
        let dir = tempdir().unwrap();
        let config = crate::mcp::McpConfig {
            servers: vec![crate::mcp::McpServerConfig {
                id: "demo".into(),
                name: "Demo".into(),
                command: "echo".into(),
                args: Vec::new(),
                env: Default::default(),
                enabled: true,
                tools: vec![crate::mcp::McpToolEntry {
                    name: "ping".into(),
                    description: Some("Ping".into()),
                    input_schema: json!({ "type": "object" }),
                }],
            }],
        };
        std::fs::write(
            crate::mcp::config_path(dir.path()),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let request = crate::types::ChatCompletionRequest {
            model: "gguf:test".into(),
            messages: Vec::new(),
            stream: false,
            tools: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            seed: None,
            enable_reasoning: None,
            reasoning_budget_tokens: None,
            tool_choice: None,
            builtin_tools: Some(true),
        };
        let merged = merge_definitions(dir.path(), &request, false).unwrap();
        let names: Vec<_> = merged
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|entry| entry.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert!(names.contains(&"get_current_time"));
        assert!(names.contains(&"mcp/demo/ping"));
    }

    #[test]
    fn mcp_excluded_when_tools_toggle_off() {
        let dir = tempdir().unwrap();
        let config = crate::mcp::McpConfig {
            servers: vec![crate::mcp::McpServerConfig {
                id: "demo".into(),
                name: "Demo".into(),
                command: "echo".into(),
                args: Vec::new(),
                env: Default::default(),
                enabled: true,
                tools: vec![crate::mcp::McpToolEntry {
                    name: "ping".into(),
                    description: None,
                    input_schema: json!({ "type": "object" }),
                }],
            }],
        };
        std::fs::write(
            crate::mcp::config_path(dir.path()),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let request = crate::types::ChatCompletionRequest {
            model: "gguf:test".into(),
            messages: Vec::new(),
            stream: false,
            tools: Some(json!([{
                "type": "function",
                "function": { "name": "custom_tool", "parameters": { "type": "object" } }
            }])),
            temperature: None,
            top_p: None,
            max_tokens: None,
            seed: None,
            enable_reasoning: None,
            reasoning_budget_tokens: None,
            tool_choice: None,
            builtin_tools: None,
        };
        let merged = merge_definitions(dir.path(), &request, false).unwrap();
        let names: Vec<_> = merged
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|entry| entry.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert!(names.contains(&"custom_tool"));
        assert!(!names.iter().any(|name| name.contains("mcp/")));
    }
}

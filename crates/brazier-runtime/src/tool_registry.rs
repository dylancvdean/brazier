//! Unified tool definitions and execution for bundled and MCP tools.

use std::path::Path;

use reqwest::Client;
use serde_json::{Value, json};

use crate::{harmony, mcp, tools::ToolInvocation};

pub struct ToolContext<'a> {
    pub data_dir: &'a Path,
    pub http: &'a Client,
    /// Image blobs already in this conversation, newest last. Generation tools
    /// use these for image-to-video and image-to-image, so a user can attach a
    /// photo and ask the model to animate it.
    pub images: Vec<ConversationImage>,
    /// Document blobs the model may pass to `doc_read`, oldest first.
    pub documents: Vec<ConversationDocument>,
}

/// An image the model can refer to when calling a generation tool.
#[derive(Debug, Clone)]
pub struct ConversationImage {
    pub sha256: String,
    pub mime_type: String,
}

/// A document attachment available to `doc_read`.
#[derive(Debug, Clone)]
pub struct ConversationDocument {
    pub sha256: String,
    pub mime_type: String,
    pub name: String,
}

/// Collect image attachments from a request's messages, oldest first.
pub fn conversation_images(
    request: &crate::types::ChatCompletionRequest,
) -> Vec<ConversationImage> {
    let mut images = Vec::new();
    for message in &request.messages {
        let serde_json::Value::Array(parts) = &message.content else {
            continue;
        };
        for part in parts {
            let Some(blob) = part.get("brazier_blob") else {
                continue;
            };
            let mime = blob
                .get("mime_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            if !mime.starts_with("image/") {
                continue;
            }
            if let Some(sha256) = blob.get("sha256").and_then(serde_json::Value::as_str) {
                images.push(ConversationImage {
                    sha256: sha256.to_owned(),
                    mime_type: mime.to_owned(),
                });
            }
        }
    }
    images
}

/// Collect PDF / Office attachments the model can read with `doc_read`.
pub fn conversation_documents(
    request: &crate::types::ChatCompletionRequest,
) -> Vec<ConversationDocument> {
    let mut documents = Vec::new();
    for message in &request.messages {
        let serde_json::Value::Array(parts) = &message.content else {
            continue;
        };
        for part in parts {
            let Some(blob) = part.get("brazier_blob") else {
                continue;
            };
            let mime = blob
                .get("mime_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            let name = blob
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("document");
            if !crate::documents::is_supported_document(mime, name) {
                continue;
            }
            if let Some(sha256) = blob.get("sha256").and_then(serde_json::Value::as_str) {
                documents.push(ConversationDocument {
                    sha256: sha256.to_owned(),
                    mime_type: mime.to_owned(),
                    name: name.to_owned(),
                });
            }
        }
    }
    documents
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
        let def_name = |def: &Value| {
            def.pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned()
        };
        if let Some(items) = crate::tools::definitions_for(data_dir).as_array() {
            for item in items {
                if request.tool_name_allowed(&def_name(item)) {
                    defs.push(item.clone());
                }
            }
        }
        for server in mcp::enabled_servers(data_dir) {
            for tool in &server.tools {
                let def = mcp::tool_to_openai(&server.id, tool);
                if request.tool_name_allowed(&def_name(&def)) {
                    defs.push(def);
                }
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
        return crate::tools::execute_with_context(
            ctx.http,
            Some(ctx.data_dir),
            &ctx.images,
            &ctx.documents,
            call_id,
            &logical,
            arguments,
        )
        .await;
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
        media: Vec::new(),
    }
}

pub fn combined_catalog(data_dir: &Path) -> Value {
    let mut entries: Vec<Value> = Vec::new();
    if let Some(builtin) = crate::tools::catalog_for(data_dir)
        .get("data")
        .and_then(Value::as_array)
    {
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
mod context_tests {
    use super::*;
    use crate::types::ChatCompletionRequest;

    fn request_with(parts: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{ "role": "user", "content": parts }]
        }))
        .expect("valid request")
    }

    #[test]
    fn conversation_images_are_collected_in_order_and_filtered_by_type() {
        let request = request_with(serde_json::json!([
            { "type": "text", "text": "animate this" },
            { "type": "brazier_blob", "brazier_blob": { "sha256": "aaa", "mime_type": "image/png" } },
            { "type": "brazier_blob", "brazier_blob": { "sha256": "bbb", "mime_type": "audio/wav" } },
            { "type": "brazier_blob", "brazier_blob": { "sha256": "ccc", "mime_type": "image/jpeg" } }
        ]));
        let images = conversation_images(&request);
        assert_eq!(
            images.iter().map(|i| i.sha256.as_str()).collect::<Vec<_>>(),
            ["aaa", "ccc"],
            "only images, oldest first"
        );
        // `latest` resolves to the most recent image, which is what a user
        // means by "animate the photo I just sent".
        assert_eq!(images.last().unwrap().sha256, "ccc");
    }

    #[test]
    fn a_text_only_conversation_offers_no_images() {
        let request = request_with(serde_json::json!([{ "type": "text", "text": "hello" }]));
        assert!(conversation_images(&request).is_empty());
    }

    #[test]
    fn conversation_documents_collect_supported_formats_only() {
        let request = request_with(serde_json::json!([
            { "type": "text", "text": "summarize" },
            { "type": "brazier_blob", "brazier_blob": {
                "sha256": "aaa", "mime_type": "application/pdf", "name": "a.pdf"
            }},
            { "type": "brazier_blob", "brazier_blob": {
                "sha256": "bbb", "mime_type": "text/plain", "name": "notes.txt"
            }},
            { "type": "brazier_blob", "brazier_blob": {
                "sha256": "ccc", "mime_type": "application/octet-stream", "name": "letter.docx"
            }}
        ]));
        let documents = conversation_documents(&request);
        assert_eq!(
            documents
                .iter()
                .map(|doc| doc.sha256.as_str())
                .collect::<Vec<_>>(),
            ["aaa", "ccc"]
        );
        assert_eq!(documents[1].name, "letter.docx");
    }
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
            builtin_tool_names: None,
            llama_slot: None,
            brazier_mode: None,
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
            builtin_tool_names: None,
            llama_slot: None,
            brazier_mode: None,
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
    fn whitelist_restricts_builtin_tools() {
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
            builtin_tools: Some(true),
            builtin_tool_names: Some(vec!["calculator".into()]),
            llama_slot: None,
            brazier_mode: None,
        };
        let merged = merge_definitions(dir.path(), &request, false).unwrap();
        let names: Vec<_> = merged
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|entry| entry.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert_eq!(names, vec!["calculator"]);
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
            builtin_tool_names: None,
            llama_slot: None,
            brazier_mode: None,
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

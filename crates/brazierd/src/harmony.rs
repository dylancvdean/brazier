//! OpenAI Harmony format helpers for gpt-oss models.
//!
//! llama.cpp renders Harmony prompts via Jinja chat templates; Brazier maps tool
//! names to the `functions.*` namespace and enables harmony-specific server flags.

use serde_json::{Value, json};

pub const FUNCTIONS_PREFIX: &str = "functions.";

/// True when the model id or path indicates a gpt-oss / Harmony-trained checkpoint.
pub fn is_harmony_model(model_id: &str) -> bool {
    let lower = model_id.to_ascii_lowercase();
    lower.contains("gpt-oss") || lower.contains("gpt_oss") || lower.contains("gptoss")
}

/// Tool name sent to the model (Harmony expects `functions.{name}` for builtins).
pub fn wire_tool_name(logical: &str, harmony: bool) -> String {
    if !harmony || logical.starts_with(FUNCTIONS_PREFIX) || logical.starts_with("mcp/") {
        return logical.to_owned();
    }
    format!("{FUNCTIONS_PREFIX}{logical}")
}

/// Canonical name used for server-side dispatch (`tools.rs`, MCP).
pub fn logical_tool_name(wire: &str) -> String {
    wire.strip_prefix(FUNCTIONS_PREFIX)
        .unwrap_or(wire)
        .to_owned()
}

/// Rewrite one OpenAI function definition for Harmony models.
pub fn adapt_tool_definition(def: &Value, harmony: bool) -> Value {
    if !harmony {
        return def.clone();
    }
    let Some(name) = def
        .pointer("/function/name")
        .and_then(Value::as_str)
        .map(logical_tool_name)
        .map(|logical| wire_tool_name(&logical, true))
    else {
        return def.clone();
    };
    let mut adapted = def.clone();
    if let Some(object) = adapted.as_object_mut() {
        if let Some(function) = object.get_mut("function").and_then(Value::as_object_mut) {
            function.insert("name".into(), Value::String(name));
        }
    }
    adapted
}

/// Rewrite tool names inside an assistant `tool_calls` array from wire → logical form.
pub fn normalize_tool_calls(value: &Value) -> Value {
    let Some(items) = value.as_array() else {
        return value.clone();
    };
    Value::Array(
        items
            .iter()
            .map(|call| {
                let mut normalized = call.clone();
                if let Some(name) = normalized
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .map(logical_tool_name)
                {
                    if let Some(function) = normalized
                        .pointer_mut("/function")
                        .and_then(Value::as_object_mut)
                    {
                        function.insert("name".into(), Value::String(name));
                    }
                }
                normalized
            })
            .collect(),
    )
}

/// llama-server flag value when launching Harmony models.
pub fn llama_reasoning_format() -> &'static str {
    "auto"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_gpt_oss_models() {
        assert!(is_harmony_model("gguf:openai/gpt-oss-20b/model.gguf"));
        assert!(!is_harmony_model("gguf:meta/llama-3/model.gguf"));
    }

    #[test]
    fn round_trips_tool_names() {
        assert_eq!(wire_tool_name("calculator", true), "functions.calculator");
        assert_eq!(logical_tool_name("functions.calculator"), "calculator");
        assert_eq!(wire_tool_name("mcp/demo/ping", true), "mcp/demo/ping");
    }
}

//! OpenAI Harmony format helpers for gpt-oss models.
//!
//! llama.cpp renders Harmony prompts via Jinja chat templates; Brazier maps tool
//! names to the `functions.*` namespace and enables harmony-specific server flags.

use serde_json::Value;

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
    if let Some(object) = adapted.as_object_mut()
        && let Some(function) = object.get_mut("function").and_then(Value::as_object_mut)
    {
        function.insert("name".into(), Value::String(name));
    }
    adapted
}

/// Rewrite tool names inside an assistant `tool_calls` array from logical → wire form.
pub fn wire_tool_calls(value: &Value, harmony: bool) -> Value {
    if !harmony {
        return value.clone();
    }
    let Some(items) = value.as_array() else {
        return value.clone();
    };
    Value::Array(
        items
            .iter()
            .map(|call| {
                let mut wired = call.clone();
                if let Some(name) = wired
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .map(|logical| wire_tool_name(logical, true))
                    && let Some(function) = wired
                        .pointer_mut("/function")
                        .and_then(Value::as_object_mut)
                {
                    function.insert("name".into(), Value::String(name));
                }
                wired
            })
            .collect(),
    )
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
                    && let Some(function) = normalized
                        .pointer_mut("/function")
                        .and_then(Value::as_object_mut)
                {
                    function.insert("name".into(), Value::String(name));
                }
                normalized
            })
            .collect(),
    )
}

/// True when an assistant message carries one or more tool calls.
pub fn has_tool_calls(tool_calls: &Option<Value>) -> bool {
    tool_calls
        .as_ref()
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty())
}

/// Whether `reasoning_content` should travel on this message.
///
/// Harmony models need reasoning preserved on every assistant tool-call turn —
/// llama.cpp's parser assumes the client resends it — even when prior-turn
/// reasoning is otherwise dropped to save context.
pub fn include_reasoning_on_message(
    has_reasoning: bool,
    has_tool_calls: bool,
    after_last_user: bool,
    drop_prior_reasoning: bool,
    harmony: bool,
) -> bool {
    if !has_reasoning {
        return false;
    }
    if !drop_prior_reasoning || after_last_user {
        return true;
    }
    harmony && has_tool_calls
}

/// Rewrite `tool_choice` so named functions use the Harmony wire namespace.
pub fn adapt_tool_choice(value: &Value, harmony: bool) -> Value {
    if !harmony {
        return value.clone();
    }
    let Some(name) = value.pointer("/function/name").and_then(Value::as_str) else {
        return value.clone();
    };
    let mut adapted = value.clone();
    if let Some(function) = adapted
        .pointer_mut("/function")
        .and_then(Value::as_object_mut)
    {
        function.insert(
            "name".into(),
            Value::String(wire_tool_name(name, true)),
        );
    }
    adapted
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

    #[test]
    fn wire_tool_calls_rewrites_assistant_history() {
        let logical = serde_json::json!([{
            "id": "call_0",
            "type": "function",
            "function": { "name": "calculator", "arguments": "{}" }
        }]);
        let wired = wire_tool_calls(&logical, true);
        assert_eq!(
            wired[0]["function"]["name"],
            "functions.calculator"
        );
        assert_eq!(wire_tool_calls(&logical, false), logical);
    }

    #[test]
    fn harmony_keeps_reasoning_on_prior_tool_turns_when_dropping() {
        assert!(include_reasoning_on_message(
            true,
            true,
            false,
            true,
            true
        ));
        assert!(!include_reasoning_on_message(
            true,
            false,
            false,
            true,
            true
        ));
    }

    #[test]
    fn adapt_tool_choice_wires_function_name() {
        let choice = serde_json::json!({
            "type": "function",
            "function": { "name": "calculator" }
        });
        let wired = adapt_tool_choice(&choice, true);
        assert_eq!(wired["function"]["name"], "functions.calculator");
    }
}

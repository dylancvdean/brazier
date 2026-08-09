//! Conversion between persisted messages and OpenAI chat payloads.

use serde_json::Value;

use crate::types::{Message, OpenAiMessage};

/// Build OpenAI chat messages from stored conversation rows.
pub fn messages_to_openai(messages: &[Message]) -> Vec<OpenAiMessage> {
    messages
        .iter()
        .map(|message| {
            let mut openai = OpenAiMessage {
                role: message.role.as_str().to_owned(),
                content: message.content.clone(),
                tool_calls: message.tool_calls.clone(),
                tool_call_id: message.tool_call_id.clone(),
                reasoning_content: reasoning_from_metadata(message),
            };
            if openai.tool_calls.is_some() && matches!(openai.content, Value::Null) {
                openai.content = Value::String(String::new());
            }
            openai
        })
        .collect()
}

fn reasoning_from_metadata(message: &Message) -> Option<String> {
    message
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("reasoning_content"))
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
}

/// Extract display text from assistant/user content (string or `{ "text": "..." }`).
pub fn text_from_message_content(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Object(map) => map
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        other => other.to_string(),
    }
}

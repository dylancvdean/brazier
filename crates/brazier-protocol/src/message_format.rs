//! Conversion between persisted messages and OpenAI chat payloads.

use serde_json::{Value, json};

use crate::types::{Message, OpenAiMessage};

/// Build OpenAI chat messages from stored conversation rows (legacy + native).
pub fn messages_to_openai(messages: &[Message]) -> Vec<OpenAiMessage> {
    let mut payload = Vec::with_capacity(messages.len());
    for message in messages {
        payload.extend(message_to_openai_pair(message));
    }
    payload
}

fn message_to_openai_pair(message: &Message) -> Vec<OpenAiMessage> {
    if message.role == crate::types::Role::Tool {
        if let Some(tool_call_id) = &message.tool_call_id {
            return vec![OpenAiMessage {
                role: "tool".to_owned(),
                content: message.content.clone(),
                tool_calls: None,
                tool_call_id: Some(tool_call_id.clone()),
                reasoning_content: None,
            }];
        }
        if let Some(records) = legacy_tool_records(&message.content) {
            let mut out = Vec::new();
            out.push(OpenAiMessage {
                role: "assistant".to_owned(),
                content: Value::String(String::new()),
                tool_calls: Some(Value::Array(
                    records
                        .iter()
                        .map(|record| {
                            json!({
                                "id": record.call_id,
                                "type": "function",
                                "function": {
                                    "name": record.name,
                                    "arguments": record.arguments
                                }
                            })
                        })
                        .collect(),
                )),
                tool_call_id: None,
                reasoning_content: reasoning_from_metadata(message),
            });
            for record in records {
                out.push(OpenAiMessage {
                    role: "tool".to_owned(),
                    content: Value::String(record.output),
                    tool_calls: None,
                    tool_call_id: Some(record.call_id),
                    reasoning_content: None,
                });
            }
            return out;
        }
    }

    let role = message.role.as_str().to_owned();
    let mut openai = OpenAiMessage {
        role,
        content: message.content.clone(),
        tool_calls: message.tool_calls.clone(),
        tool_call_id: message.tool_call_id.clone(),
        reasoning_content: reasoning_from_metadata(message),
    };
    if openai.tool_calls.is_some() && matches!(openai.content, Value::Null) {
        openai.content = Value::String(String::new());
    }
    vec![openai]
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

#[derive(Debug, Clone, serde::Deserialize)]
struct LegacyToolRecord {
    call_id: String,
    name: String,
    arguments: String,
    output: String,
}

fn legacy_tool_records(content: &Value) -> Option<Vec<LegacyToolRecord>> {
    let parsed = match content {
        Value::String(text) => serde_json::from_str(text).ok()?,
        other => other.clone(),
    };
    let items = parsed.get("brazier_tool_calls")?.as_array()?;
    let mut records = Vec::new();
    for item in items {
        records.push(serde_json::from_value(item.clone()).ok()?);
    }
    if records.is_empty() {
        None
    } else {
        Some(records)
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Role;

    #[test]
    fn expands_legacy_tool_blob() {
        let message = Message {
            id: "1".into(),
            conversation_id: "c".into(),
            parent_id: None,
            role: Role::Tool,
            content: json!(
                serde_json::to_string(&json!({
                    "brazier_tool_calls": [{
                        "call_id": "call_1",
                        "name": "calculator",
                        "arguments": "{\"expression\":\"1+1\"}",
                        "output": "2",
                        "is_error": false
                    }]
                }))
                .unwrap()
            ),
            model: None,
            tool_calls: None,
            tool_call_id: None,
            source: None,
            correlation_id: None,
            status: None,
            metadata: None,
            created_at: String::new(),
        };
        let openai = messages_to_openai(&[message]);
        assert_eq!(openai.len(), 2);
        assert_eq!(openai[0].role, "assistant");
        assert_eq!(openai[1].role, "tool");
    }
}

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub conversation_id: String,
    pub parent_id: Option<String>,
    pub role: Role,
    pub content: Value,
    pub model: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateConversation {
    pub title: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateMessage {
    pub parent_id: Option<String>,
    pub role: Role,
    pub content: Value,
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub streaming: bool,
    pub tools: bool,
    pub reasoning: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub id: String,
    pub name: String,
    pub engine: String,
    pub capabilities: ModelCapabilities,
    #[serde(default)]
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiMessage {
    pub role: String,
    pub content: Value,
    #[serde(default)]
    pub tool_calls: Option<Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub enable_reasoning: Option<bool>,
    /// When true, the daemon offers its bundled tools to the model and
    /// executes returned tool calls server-side.
    #[serde(default)]
    pub builtin_tools: Option<bool>,
}

fn default_model() -> String {
    String::new()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub input: Value,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub enable_reasoning: Option<bool>,
    #[serde(default)]
    pub builtin_tools: Option<bool>,
}

pub fn text_from_content(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                let kind = part.get("type")?.as_str()?;
                match kind {
                    "text" | "input_text" | "output_text" => {
                        part.get("text")?.as_str().map(ToOwned::to_owned)
                    }
                    "image_url" | "input_image" => Some("[image]".to_owned()),
                    "input_audio" | "audio" => Some("[audio]".to_owned()),
                    "input_video" | "video" => Some("[video]".to_owned()),
                    "brazier_blob" => {
                        let mime = part
                            .pointer("/brazier_blob/mime_type")
                            .and_then(Value::as_str)
                            .unwrap_or("file");
                        Some(format!("[attachment: {mime}]"))
                    }
                    _ => None,
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        other => other.to_string(),
    }
}

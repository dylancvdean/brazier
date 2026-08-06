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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "sqlx", derive(sqlx::FromRow))]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    /// Agent session this conversation's turns are submitted to, when one is
    /// bound. Text and voice share it rather than opening one each.
    #[serde(default)]
    pub agent_session_id: Option<String>,
    /// Compact summary a fresh voice session is seeded with.
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub summary_updated_at: Option<String>,
    /// Incognito conversations are ephemeral and memory-free: the daemon
    /// refuses to persist messages for them or to source memories from them.
    #[serde(default)]
    pub incognito: bool,
}

/// Which surface produced a message. Voice renderings of an agent answer are
/// `assistant_voice` and share the answer's correlation id; they are never a
/// second authoritative reply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub conversation_id: String,
    pub parent_id: Option<String>,
    pub role: Role,
    pub content: Value,
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    pub created_at: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateConversation {
    pub title: Option<String>,
}

/// Fields a conversation update may set. `None` leaves a field alone; clearing
/// the agent binding is an explicit `Some(None)`.
#[derive(Debug, Default, Deserialize)]
pub struct UpdateConversation {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(
        default,
        deserialize_with = "crate::serde_util::deserialize_double_option"
    )]
    pub agent_session_id: Option<Option<String>>,
    #[serde(default)]
    pub summary: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateMessage {
    pub parent_id: Option<String>,
    pub role: Role,
    pub content: Value,
    pub model: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub correlation_id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

/// In-place edit of a stored message. Used to finalize a streamed answer and to
/// mark a turn cancelled, superseded, or failed.
#[derive(Debug, Default, Deserialize)]
pub struct UpdateMessage {
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

/// A durable user memory the model saved across conversations. The store is
/// global (not per-conversation); `source_*` records where a memory came from
/// so the Settings editor can trace and manage it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub text: String,
    /// `fact`, `preference`, or `summary`.
    #[serde(default = "default_memory_kind")]
    pub kind: String,
    /// Pinned memories are exempt from dreaming's prune/merge passes.
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_conversation_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_message_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

fn default_memory_kind() -> String {
    "fact".to_owned()
}

#[derive(Debug, Deserialize)]
pub struct CreateMemory {
    pub text: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub pinned: Option<bool>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub source_conversation_id: Option<String>,
    #[serde(default)]
    pub source_message_id: Option<String>,
}

/// Partial edit of a memory row. Absent fields are left alone.
#[derive(Debug, Default, Deserialize)]
pub struct UpdateMemory {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub pinned: Option<bool>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub streaming: bool,
    pub tools: bool,
    pub reasoning: bool,
    /// Model-native context window when known from config/metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_length: Option<u32>,
    /// Supported reasoning controls: `off`, `on`, and optionally `budget`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_modes: Vec<String>,
    /// Uses OpenAI Harmony wire format (gpt-oss family).
    #[serde(default)]
    pub harmony: bool,
    /// How the chat model consumes audio attachments, when advertised.
    ///
    /// - omitted / null: no native audio; Brazier may still run **batch ASR**
    ///   (whisper.cpp) and inject a transcript.
    /// - `"native"`: model accepts audio features/tokens (OpenAI `input_audio`
    ///   style). Prefer this path over ASR when both are available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_input: Option<String>,
    /// When true, the model is advertised for Computer Use (screenshot → action).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub computer_use: bool,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            input_modalities: vec!["text".into()],
            output_modalities: vec!["text".into()],
            streaming: true,
            tools: false,
            reasoning: false,
            max_context_length: None,
            reasoning_modes: Vec::new(),
            harmony: false,
            audio_input: None,
            computer_use: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub id: String,
    pub name: String,
    pub engine: String,
    pub capabilities: ModelCapabilities,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub read_only: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library_label: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiMessage {
    pub role: String,
    pub content: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Interleaved thinking trace (Qwen 3.5+, DeepSeek, …). Must be passed back
    /// on later turns so Jinja/llama.cpp keep parsing native tool-call dialects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
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
    #[serde(default)]
    pub reasoning_budget_tokens: Option<u32>,
    /// How the model should use tools (`auto`, `none`, or a specific function).
    #[serde(default)]
    pub tool_choice: Option<Value>,
    /// When true, the daemon offers its bundled tools to the model and
    /// executes returned tool calls server-side.
    #[serde(default)]
    pub builtin_tools: Option<bool>,
    /// Optional whitelist of tool names (bundled + `mcp/<server>/<tool>`) to
    /// offer when `builtin_tools` is on. `None` offers every enabled tool.
    #[serde(default)]
    pub builtin_tool_names: Option<Vec<String>>,
    /// Brazier extension selecting the resident-server shape for this request.
    ///
    /// OpenAI-compatible clients omit it and get the single-stream chat shape.
    /// The desktop agent marks its requests as `agent` so llama.cpp reserves
    /// the configured continuous-batching slots.
    #[serde(default)]
    pub brazier_mode: Option<String>,
    /// llama-server KV slot (`id_slot`). Parent agent sessions use 0; subagents
    /// use 1..N when parallel subagents are enabled on the model profile.
    #[serde(default)]
    pub llama_slot: Option<u32>,
}

impl ChatCompletionRequest {
    /// Whether any server-side tools (bundled, MCP, or request-supplied schemas) are active.
    pub fn tools_active(&self) -> bool {
        self.builtin_tools.unwrap_or(false) || self.tools.is_some()
    }

    /// Whether a tool name passes the request's whitelist (all when unset).
    pub fn tool_name_allowed(&self, name: &str) -> bool {
        self.builtin_tool_names
            .as_ref()
            .is_none_or(|names| names.iter().any(|allowed| allowed == name))
    }
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
    pub reasoning_budget_tokens: Option<u32>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
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

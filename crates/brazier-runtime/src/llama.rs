//! llama.cpp binary discovery, managed installation, and server lifecycle.

use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use anyhow::Context;
use flate2::read::GzDecoder;
use futures::StreamExt;
use tar::Archive;
use tokio::{
    io::AsyncWriteExt,
    process::{Child, Command},
};

use crate::{
    model_settings::TextProfile,
    progress::{ProgressCallback, ProgressEvent},
    runtime_settings::{RuntimeSettings, RuntimeTarget},
    types::{ChatCompletionRequest, OpenAiMessage},
};

const GITHUB_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";
const USER_AGENT: &str = "brazier-llama-manager";

/// Cap stderr we surface so an OOM dump does not flood the UI.
const STARTUP_STDERR_LIMIT: usize = 4_000;

/// Build a user-facing error when a local inference server dies during launch.
///
/// OOMs and allocation failures get an explicit remediation hint; other exits
/// keep a truncated stderr excerpt so the failure is diagnosable without
/// burying the daemon in a generic 500.
pub fn describe_server_startup_failure(
    server: &str,
    status: impl std::fmt::Display,
    stderr: &str,
) -> String {
    let trimmed = stderr.trim();
    let excerpt = if trimmed.is_empty() {
        "(no stderr)".to_owned()
    } else if trimmed.len() <= STARTUP_STDERR_LIMIT {
        trimmed.to_owned()
    } else {
        format!("{}…", &trimmed[..STARTUP_STDERR_LIMIT])
    };
    if startup_looks_like_oom(trimmed) {
        format!(
            "{server} ran out of memory while starting ({status}). \
             Lower context size, turn off Parallel subagents, reduce GPU layers, \
             or close other apps, then try again.\n\n{excerpt}"
        )
    } else {
        format!("{server} exited during startup with {status}:\n{excerpt}")
    }
}

/// Whether stderr / status text points at an out-of-memory launch failure.
pub fn startup_looks_like_oom(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("out of memory")
        || lower.contains("out-of-memory")
        || lower.contains("cuda_error_out_of_memory")
        || lower.contains("hip_error_out_of_memory")
        || lower.contains("cannot allocate")
        || lower.contains("failed to allocate")
        || lower.contains("std::bad_alloc")
        || lower.contains("metal: failed to create buffer")
        || lower.contains("insufficient memory")
        || (lower.contains("killed") && lower.contains("memory"))
        || lower
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|token| token == "oom")
}

/// Managed install prefix under the application data directory.
pub fn managed_engine_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("engines").join("llama.cpp")
}

pub fn managed_binary_path(data_dir: &Path) -> PathBuf {
    managed_engine_dir(data_dir).join("bin").join(binary_name())
}

pub fn managed_binary_path_for_target(data_dir: &Path, target: RuntimeTarget) -> PathBuf {
    if matches!(
        target,
        RuntimeTarget::Auto | RuntimeTarget::Cpu | RuntimeTarget::Metal
    ) {
        return managed_binary_path(data_dir);
    }
    managed_engine_dir(data_dir)
        .join(target.as_str())
        .join("bin")
        .join(binary_name())
}

/// Root directory where managed install metadata (VERSION) lives for a target.
pub fn managed_install_root(data_dir: &Path, target: RuntimeTarget) -> PathBuf {
    let engine_dir = managed_engine_dir(data_dir);
    match target {
        RuntimeTarget::Auto | RuntimeTarget::Cpu | RuntimeTarget::Metal => engine_dir,
        _ => engine_dir.join(target.as_str()),
    }
}

pub fn managed_is_installed(data_dir: &Path, target: RuntimeTarget) -> bool {
    managed_binary_path_for_target(data_dir, target).is_file()
}

pub fn managed_installed_version(data_dir: &Path, target: RuntimeTarget) -> Option<String> {
    std::fs::read_to_string(managed_install_root(data_dir, target).join("VERSION"))
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// Directory that must appear on `LD_LIBRARY_PATH` / `PATH` for managed builds.
pub fn managed_lib_dir(data_dir: &Path) -> PathBuf {
    managed_engine_dir(data_dir).join("bin")
}

pub fn binary_name() -> &'static str {
    if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

/// Whether a path names a `llama-server` executable.
///
/// The smoke test in [`binary_appears_runnable`] only proves that *something*
/// starts and tolerates `--version`, which a Python interpreter does. Pinning
/// the wrong program produces a failure one step later, at the next chat
/// request, phrased as though the model were at fault — so the name is checked
/// before anything is pinned.
pub fn is_llama_server_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            let lower = name.to_ascii_lowercase();
            lower == "llama-server" || lower == "llama-server.exe"
        })
}

/// Platform tag used to select a GitHub release asset.
pub fn platform_asset_tag() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Some("ubuntu-x64"),
        ("linux", "aarch64") => Some("ubuntu-arm64"),
        ("macos", "x86_64") => Some("macos-x64"),
        ("macos", "aarch64") => Some("macos-arm64"),
        ("windows", "x86_64") => Some("win-cpu-x64"),
        ("windows", "aarch64") => Some("win-cpu-arm64"),
        _ => None,
    }
}

/// Choose the best prebuilt release asset for this host from a release asset list.
pub fn select_release_asset<'a>(
    asset_names: impl IntoIterator<Item = &'a str>,
    platform_tag: &str,
) -> Option<&'a str> {
    let mut candidates: Vec<&str> = asset_names
        .into_iter()
        .filter(|name| {
            let lower = name.to_ascii_lowercase();
            lower.contains(platform_tag)
                && (lower.ends_with(".tar.gz") || lower.ends_with(".zip"))
                && !lower.contains("vulkan")
                && !lower.contains("cuda")
                && !lower.contains("rocm")
                && !lower.contains("sycl")
                && !lower.contains("openvino")
                && !lower.contains("cudart")
        })
        .collect();
    // Prefer the plain CPU ubuntu/macos package (shortest name among filtered).
    candidates.sort_by_key(|name| name.len());
    candidates.first().copied()
}

pub fn select_release_asset_for_target<'a>(
    asset_names: impl IntoIterator<Item = &'a str>,
    platform_tag: &str,
    target: RuntimeTarget,
) -> Option<&'a str> {
    if matches!(
        target,
        RuntimeTarget::Auto | RuntimeTarget::Cpu | RuntimeTarget::Metal
    ) {
        return select_release_asset(asset_names, platform_tag);
    }
    let arch = if platform_tag.contains("arm64") {
        "arm64"
    } else {
        "x64"
    };
    let platform = if platform_tag.starts_with("ubuntu") {
        "ubuntu"
    } else if platform_tag.starts_with("win") {
        "win"
    } else {
        platform_tag
    };
    let flavor = target.as_str();
    let mut candidates: Vec<&str> = asset_names
        .into_iter()
        .filter(|name| {
            let lower = name.to_ascii_lowercase();
            lower.contains(platform)
                && lower.contains(arch)
                && lower.contains(flavor)
                && (lower.ends_with(".tar.gz") || lower.ends_with(".zip"))
                && !lower.contains("cudart")
        })
        .collect();
    candidates.sort_by_key(|name| name.len());
    candidates.first().copied()
}

/// Candidate paths where a user- or app-installed llama-server might live.
pub fn discovery_candidates(data_dir: &Path, path_env: Option<&str>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    candidates.push(managed_binary_path(data_dir));
    if let Some(path_env) = path_env {
        for dir in std::env::split_paths(path_env) {
            candidates.push(dir.join(binary_name()));
        }
    }
    for dir in [
        "/usr/local/bin",
        "/usr/bin",
        "/opt/homebrew/bin",
        "/opt/local/bin",
    ] {
        candidates.push(PathBuf::from(dir).join(binary_name()));
    }
    candidates
}

/// Return the first existing executable among discovery candidates.
pub fn discover_binary(data_dir: &Path, path_env: Option<&str>) -> Option<PathBuf> {
    discovery_candidates(data_dir, path_env)
        .into_iter()
        .find(|path| path.is_file())
}

/// Which sampler names the endpoint on the other end understands.
///
/// llama.cpp exposes a much larger sampler set than OpenAI describes, and MLX a
/// different subset again. Sending a key a server does not know is not always
/// ignored — some reject the request outright — so the extended samplers are
/// only written for a server known to read them, and a remote endpoint gets the
/// standard fields alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerDialect {
    LlamaCpp,
    Mlx,
    OpenAi,
}

/// Everything outside the request itself that shapes the body sent to a server.
#[derive(Clone, Copy)]
pub struct ChatContext<'a> {
    pub settings: &'a RuntimeSettings,
    /// The selected model's overrides, when it has any.
    pub profile: Option<&'a TextProfile>,
    pub dialect: SamplerDialect,
    pub model_alias: &'a str,
    pub stream: bool,
}

impl<'a> ChatContext<'a> {
    /// A context for a local llama-server with no per-model overrides.
    pub fn local(settings: &'a RuntimeSettings, model_alias: &'a str, stream: bool) -> Self {
        Self {
            settings,
            profile: None,
            dialect: SamplerDialect::LlamaCpp,
            model_alias,
            stream,
        }
    }
}

/// Write `value` under `key` when it is set.
fn put<T: Into<serde_json::Value>>(body: &mut serde_json::Value, key: &str, value: Option<T>) {
    if let Some(value) = value {
        body[key] = value.into();
    }
}

/// Translate a Brazier chat request into the JSON body expected by llama-server.
pub fn translate_chat_request(
    request: &ChatCompletionRequest,
    context: ChatContext<'_>,
) -> serde_json::Value {
    let ChatContext {
        settings,
        profile,
        dialect,
        model_alias,
        stream,
    } = context;
    let mut messages: Vec<serde_json::Value> = request
        .messages
        .iter()
        .map(message_to_openai_json)
        .collect();
    // A model's own system prompt leads the conversation rather than replacing
    // whatever the caller sent, which may be a tool preamble it still needs.
    if let Some(prompt) = profile
        .and_then(|profile| profile.system_prompt.as_deref())
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
    {
        messages.insert(
            0,
            serde_json::json!({ "role": "system", "content": prompt }),
        );
    }

    let temperature = request
        .temperature
        .or(profile.and_then(|profile| profile.temperature))
        .unwrap_or(settings.temperature);
    let top_p = request
        .top_p
        .or(profile.and_then(|profile| profile.top_p))
        .unwrap_or(settings.top_p);
    let reasoning = request
        .enable_reasoning
        .or(profile.and_then(|profile| profile.enable_reasoning))
        .unwrap_or(settings.enable_reasoning);
    let mut body = serde_json::json!({
        "model": model_alias,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": { "enable_thinking": reasoning }
    });
    if let Some(budget) = request
        .reasoning_budget_tokens
        .or(profile.and_then(|profile| profile.reasoning_budget_tokens))
        .or(settings.reasoning_budget_tokens)
    {
        body["thinking_budget_tokens"] = serde_json::json!(budget);
    }
    put(
        &mut body,
        "max_tokens",
        request
            .max_tokens
            .or(profile.and_then(|profile| profile.max_tokens))
            .or(settings.max_tokens),
    );
    put(
        &mut body,
        "seed",
        request.seed.or(profile.and_then(|profile| profile.seed)),
    );

    if let Some(profile) = profile {
        if !profile.stop.is_empty() {
            body["stop"] = serde_json::json!(profile.stop);
        }
        // Penalties OpenAI itself defines travel everywhere.
        put(&mut body, "presence_penalty", profile.presence_penalty);
        put(&mut body, "frequency_penalty", profile.frequency_penalty);
        match dialect {
            SamplerDialect::LlamaCpp => {
                put(&mut body, "top_k", profile.top_k);
                put(&mut body, "min_p", profile.min_p);
                put(&mut body, "typical_p", profile.typical_p);
                put(&mut body, "repeat_penalty", profile.repeat_penalty);
                put(&mut body, "repeat_last_n", profile.repeat_last_n);
                put(&mut body, "dry_multiplier", profile.dry_multiplier);
                put(&mut body, "dry_base", profile.dry_base);
                put(&mut body, "dry_allowed_length", profile.dry_allowed_length);
                put(&mut body, "mirostat", profile.mirostat);
                put(&mut body, "mirostat_tau", profile.mirostat_tau);
                put(&mut body, "mirostat_eta", profile.mirostat_eta);
            }
            SamplerDialect::Mlx => {
                put(&mut body, "top_k", profile.top_k);
                put(&mut body, "min_p", profile.min_p);
                put(&mut body, "repetition_penalty", profile.repeat_penalty);
                put(
                    &mut body,
                    "repetition_context_size",
                    profile.repeat_last_n.filter(|value| *value > 0),
                );
            }
            SamplerDialect::OpenAi => {}
        }
    }

    if let Some(value) = &request.tools {
        body["tools"] = value.clone();
    }
    if let Some(value) = &request.tool_choice {
        body["tool_choice"] = value.clone();
    }
    body
}

/// Extract assistant content delta from an OpenAI-compatible stream chunk JSON body.
pub fn extract_stream_delta_content(body: &serde_json::Value) -> Option<String> {
    body.pointer("/choices/0/delta/content")
        .and_then(serde_json::Value::as_str)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamParse {
    Skip,
    Content(String),
    Done,
}

/// Parse one SSE `data:` payload from llama-server / OpenAI streaming.
pub fn parse_stream_data_payload(data: &str) -> StreamParse {
    let data = data.trim();
    if data.is_empty() {
        return StreamParse::Skip;
    }
    if data == "[DONE]" {
        return StreamParse::Done;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
        return StreamParse::Skip;
    };
    match extract_stream_delta_content(&value) {
        Some(text) => StreamParse::Content(text),
        None => StreamParse::Skip,
    }
}

/// One partial tool call carried by a streamed delta.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCallFragment {
    pub index: usize,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// Decoded content of one streamed completion chunk.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamChunk {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCallFragment>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkParse {
    Skip,
    Chunk(StreamChunk),
    Done,
}

/// Parse one SSE `data:` payload including tool-call deltas and finish reason.
pub fn parse_stream_chunk(data: &str) -> ChunkParse {
    let data = data.trim();
    if data.is_empty() {
        return ChunkParse::Skip;
    }
    if data == "[DONE]" {
        return ChunkParse::Done;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
        return ChunkParse::Skip;
    };
    let Some(choice) = value.pointer("/choices/0") else {
        return ChunkParse::Skip;
    };
    let content = choice
        .pointer("/delta/content")
        .and_then(serde_json::Value::as_str)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned);
    let reasoning = choice
        .pointer("/delta/reasoning_content")
        .and_then(serde_json::Value::as_str)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            choice
                .pointer("/delta/reasoning")
                .and_then(serde_json::Value::as_str)
                .filter(|text| !text.is_empty())
                .map(ToOwned::to_owned)
        });
    let tool_calls: Vec<ToolCallFragment> = choice
        .pointer("/delta/tool_calls")
        .and_then(serde_json::Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .map(|call| ToolCallFragment {
                    index: call
                        .get("index")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0) as usize,
                    id: call
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .map(ToOwned::to_owned),
                    name: call
                        .pointer("/function/name")
                        .and_then(serde_json::Value::as_str)
                        .map(ToOwned::to_owned),
                    arguments: call
                        .pointer("/function/arguments")
                        .and_then(serde_json::Value::as_str)
                        .map(ToOwned::to_owned),
                })
                .collect()
        })
        .unwrap_or_default();
    let finish_reason = choice
        .get("finish_reason")
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned);
    if content.is_none() && reasoning.is_none() && tool_calls.is_empty() && finish_reason.is_none()
    {
        return ChunkParse::Skip;
    }
    ChunkParse::Chunk(StreamChunk {
        content,
        reasoning,
        tool_calls,
        finish_reason,
    })
}

/// A fully accumulated tool call, ready for execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccumulatedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Accumulates streamed tool-call fragments by index.
#[derive(Debug, Default)]
pub struct ToolCallAccumulator {
    calls: Vec<AccumulatedToolCall>,
}

impl ToolCallAccumulator {
    pub fn absorb(&mut self, fragments: &[ToolCallFragment]) {
        for fragment in fragments {
            while self.calls.len() <= fragment.index {
                self.calls.push(AccumulatedToolCall {
                    id: String::new(),
                    name: String::new(),
                    arguments: String::new(),
                });
            }
            let call = &mut self.calls[fragment.index];
            if let Some(id) = &fragment.id {
                call.id = id.clone();
            }
            if let Some(name) = &fragment.name {
                call.name.push_str(name);
            }
            if let Some(arguments) = &fragment.arguments {
                call.arguments.push_str(arguments);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    pub fn into_calls(self) -> Vec<AccumulatedToolCall> {
        self.calls
            .into_iter()
            .enumerate()
            .map(|(index, mut call)| {
                if call.id.is_empty() {
                    call.id = format!("call_{index}");
                }
                if call.arguments.is_empty() {
                    call.arguments = "{}".to_owned();
                }
                call.name = crate::harmony::logical_tool_name(&call.name);
                call
            })
            .filter(|call| !call.name.is_empty())
            .collect()
    }
}

/// Extract complete tool calls from a non-streamed chat completion body.
pub fn extract_tool_calls(body: &serde_json::Value) -> Vec<AccumulatedToolCall> {
    body.pointer("/choices/0/message/tool_calls")
        .and_then(serde_json::Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .enumerate()
                .filter_map(|(index, call)| {
                    let name = call
                        .pointer("/function/name")
                        .and_then(serde_json::Value::as_str)?;
                    Some(AccumulatedToolCall {
                        id: call
                            .get("id")
                            .and_then(serde_json::Value::as_str)
                            .map(ToOwned::to_owned)
                            .unwrap_or_else(|| format!("call_{index}")),
                        name: crate::harmony::logical_tool_name(name),
                        arguments: call
                            .pointer("/function/arguments")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("{}")
                            .to_owned(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Serialize accumulated tool calls back into OpenAI assistant-message form.
pub fn tool_calls_to_json(calls: &[AccumulatedToolCall]) -> serde_json::Value {
    serde_json::Value::Array(
        calls
            .iter()
            .map(|call| {
                serde_json::json!({
                    "id": call.id,
                    "type": "function",
                    "function": { "name": call.name, "arguments": call.arguments }
                })
            })
            .collect(),
    )
}

/// Serialize one streamed tool-call fragment into an OpenAI `delta.tool_calls` entry.
pub fn tool_call_fragment_to_delta(fragment: &ToolCallFragment) -> serde_json::Value {
    let mut entry = serde_json::json!({ "index": fragment.index, "type": "function" });
    if let Some(id) = &fragment.id {
        entry["id"] = serde_json::Value::String(id.clone());
    }
    let mut function = serde_json::Map::new();
    if let Some(name) = &fragment.name {
        function.insert("name".into(), serde_json::Value::String(name.clone()));
    }
    if let Some(arguments) = &fragment.arguments {
        function.insert(
            "arguments".into(),
            serde_json::Value::String(arguments.clone()),
        );
    }
    if !function.is_empty() {
        entry["function"] = serde_json::Value::Object(function);
    }
    entry
}

fn message_to_openai_json(message: &OpenAiMessage) -> serde_json::Value {
    let content = match &message.content {
        serde_json::Value::String(text) => serde_json::Value::String(text.clone()),
        serde_json::Value::Array(parts) => serde_json::Value::Array(
            parts
                .iter()
                .map(|part| {
                    if part.get("type").and_then(|value| value.as_str()) == Some("brazier_blob") {
                        let mime = part
                            .pointer("/brazier_blob/mime_type")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("application/octet-stream");
                        let name = part
                            .pointer("/brazier_blob/name")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("attachment");
                        serde_json::json!({
                            "type": "text",
                            "text": format!("[attachment: {name} ({mime})]")
                        })
                    } else {
                        part.clone()
                    }
                })
                .collect(),
        ),
        other => serde_json::Value::String(other.to_string()),
    };
    let mut json = serde_json::json!({
        "role": message.role,
        "content": content
    });
    if let Some(tool_calls) = &message.tool_calls {
        json["tool_calls"] = tool_calls.clone();
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        json["tool_call_id"] = serde_json::json!(tool_call_id);
    }
    if let Some(reasoning) = &message.reasoning_content
        && !reasoning.is_empty()
    {
        json["reasoning_content"] = serde_json::Value::String(reasoning.clone());
    }
    json
}

/// Extract assistant text from a llama-server / OpenAI chat completion body.
pub fn extract_assistant_text(body: &serde_json::Value) -> anyhow::Result<String> {
    if let Some(text) = body
        .pointer("/choices/0/message/content")
        .and_then(serde_json::Value::as_str)
    {
        return Ok(text.to_owned());
    }
    if let Some(text) = body
        .pointer("/choices/0/text")
        .and_then(serde_json::Value::as_str)
    {
        return Ok(text.to_owned());
    }
    anyhow::bail!("llama-server response did not include assistant content")
}

/// Extract interleaved thinking from a non-streamed chat completion body.
pub fn extract_reasoning(body: &serde_json::Value) -> Option<String> {
    body.pointer("/choices/0/message/reasoning_content")
        .and_then(serde_json::Value::as_str)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            body.pointer("/choices/0/message/reasoning")
                .and_then(serde_json::Value::as_str)
                .filter(|text| !text.is_empty())
                .map(ToOwned::to_owned)
        })
}

#[derive(Debug, Clone)]
pub struct ReleaseAsset {
    pub name: String,
    pub browser_download_url: String,
}

/// Newest release tag from cache, without waiting on GitHub.
///
/// Status views call this on every open, so a stale-but-instant answer beats
/// a blocking lookup; the refresh it triggers lands in time for the next one.
pub fn cached_release_tag(client: &reqwest::Client) -> crate::github_releases::CachedRelease {
    crate::github_releases::cached_or_refresh(client, GITHUB_API, USER_AGENT)
}

/// Resolve the preferred managed binary download for this platform.
pub async fn resolve_managed_release(
    client: &reqwest::Client,
    target: RuntimeTarget,
) -> anyhow::Result<(String, ReleaseAsset)> {
    let platform = platform_asset_tag()
        .context("managed llama.cpp binaries are not available for this platform")?;
    let release = crate::github_releases::latest_release(client, GITHUB_API, USER_AGENT).await?;
    let names: Vec<String> = release.asset_names().map(str::to_owned).collect();
    let selected =
        select_release_asset_for_target(names.iter().map(String::as_str), platform, target)
            .context("no matching llama.cpp release asset for this platform")?
            .to_owned();
    let asset = release
        .asset(&selected)
        .context("selected asset missing from release")?;
    Ok((
        release.tag_name.clone(),
        ReleaseAsset {
            name: asset.name.clone(),
            browser_download_url: asset.browser_download_url.clone(),
        },
    ))
}

/// Download and extract a managed llama-server binary into the data directory.
pub async fn install_managed_binary(
    client: &reqwest::Client,
    data_dir: &Path,
) -> anyhow::Result<PathBuf> {
    install_managed_binary_with_progress(client, data_dir, RuntimeTarget::Cpu, Box::new(|_| {}))
        .await
}

pub async fn install_managed_binary_with_progress(
    client: &reqwest::Client,
    data_dir: &Path,
    target: RuntimeTarget,
    mut progress: ProgressCallback,
) -> anyhow::Result<PathBuf> {
    progress(ProgressEvent::phase(
        "resolve",
        "Looking up the latest llama.cpp release",
    ));
    let (tag, asset) = resolve_managed_release(client, target).await?;
    tracing::info!(%tag, asset = %asset.name, "downloading managed llama.cpp binary");
    progress(ProgressEvent::phase(
        "download",
        format!("Downloading {tag} ({})", asset.name),
    ));

    let response = client
        .get(&asset.browser_download_url)
        .header("user-agent", USER_AGENT)
        .send()
        .await
        .context("download llama.cpp release")?
        .error_for_status()
        .context("llama.cpp release download failed")?;
    let total = response.content_length();
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    let mut written = 0_u64;
    let mut last_emit = 0_u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read llama.cpp release body")?;
        written += chunk.len() as u64;
        bytes.extend_from_slice(&chunk);
        if written.saturating_sub(last_emit) >= 256 * 1024 || total == Some(written) {
            progress(ProgressEvent::download(written, total));
            last_emit = written;
        }
    }
    progress(ProgressEvent::download(written, total.or(Some(written))));

    let binary = managed_binary_path_for_target(data_dir, target);
    let bin_dir = binary
        .parent()
        .context("managed binary path has no parent")?
        .to_path_buf();
    let engine_dir = bin_dir
        .parent()
        .context("managed binary directory has no parent")?
        .to_path_buf();
    // Replace any prior install so shared libraries stay consistent with the binary.
    if bin_dir.exists() {
        tokio::fs::remove_dir_all(&bin_dir)
            .await
            .context("clear previous managed engine install")?;
    }
    tokio::fs::create_dir_all(&bin_dir)
        .await
        .context("create engine bin directory")?;
    let archive_path = engine_dir.join(&asset.name);
    {
        let mut file = tokio::fs::File::create(&archive_path)
            .await
            .context("write release archive")?;
        file.write_all(&bytes).await?;
        file.flush().await?;
    }

    progress(ProgressEvent::phase(
        "extract",
        "Extracting llama-server and shared libraries",
    ));
    extract_release_archive(&archive_path, &bin_dir).context("extract llama.cpp release")?;
    anyhow::ensure!(
        binary.is_file(),
        "archive did not contain {}; extracted into {}",
        binary_name(),
        bin_dir.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for entry in std::fs::read_dir(&bin_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let mut permissions = std::fs::metadata(&path)?.permissions();
                permissions.set_mode(0o755);
                std::fs::set_permissions(&path, permissions)?;
            }
        }
    }
    let _ = tokio::fs::remove_file(&archive_path).await;
    tokio::fs::write(engine_dir.join("VERSION"), format!("{tag}\n")).await?;
    progress(ProgressEvent::done(serde_json::json!({
        "binary": binary.display().to_string(),
        "tag": tag,
        "status": "ready"
    })));
    Ok(binary)
}

/// Extract release members into `bin_dir`, flattening a single top-level prefix directory.
fn extract_release_archive(archive_path: &Path, bin_dir: &Path) -> anyhow::Result<()> {
    let file = std::fs::File::open(archive_path).context("open archive")?;
    let name = archive_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);
        let mut found_server = false;
        for entry in archive.entries().context("read tar entries")? {
            let mut entry = entry.context("tar entry")?;
            let path = entry.path().context("tar entry path")?.into_owned();
            if entry.header().entry_type().is_dir() {
                continue;
            }
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if file_name.is_empty() || file_name == "LICENSE" {
                continue;
            }
            // Keep binaries and shared libraries; skip bulky unrelated tools optional but keep all libs.
            let is_lib = file_name.contains(".so")
                || file_name.ends_with(".dll")
                || file_name.ends_with(".dylib");
            let is_server = file_name == "llama-server" || file_name == "llama-server.exe";
            if !(is_lib || is_server || file_name.starts_with("llama")) {
                continue;
            }
            let dest = bin_dir.join(file_name);
            entry
                .unpack(&dest)
                .with_context(|| format!("unpack {file_name}"))?;
            if is_server {
                found_server = true;
            }
        }
        anyhow::ensure!(found_server, "llama-server binary not found in archive");
        return Ok(());
    }
    if name.ends_with(".zip") {
        let mut archive = zip::ZipArchive::new(file).context("read zip archive")?;
        let mut found_server = false;
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index).context("read zip entry")?;
            if entry.is_dir() {
                continue;
            }
            let Some(path) = entry.enclosed_name() else {
                continue;
            };
            let file_name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            let is_lib = file_name.contains(".so")
                || file_name.ends_with(".dll")
                || file_name.ends_with(".dylib");
            let is_server = file_name == "llama-server" || file_name == "llama-server.exe";
            if !(is_lib || is_server || file_name.starts_with("llama")) {
                continue;
            }
            let mut destination =
                std::fs::File::create(bin_dir.join(file_name)).context("create zip output")?;
            std::io::copy(&mut entry, &mut destination).context("extract zip entry")?;
            found_server |= is_server;
        }
        anyhow::ensure!(found_server, "llama-server binary not found in archive");
        return Ok(());
    }
    anyhow::bail!("unsupported archive format: {name}");
}

/// Best-effort check that a binary can start (shared libraries resolve).
pub fn binary_appears_runnable(binary: &Path) -> bool {
    let mut command = std::process::Command::new(binary);
    command
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(dir) = binary.parent() {
        #[cfg(unix)]
        {
            let key = if cfg!(target_os = "macos") {
                "DYLD_LIBRARY_PATH"
            } else {
                "LD_LIBRARY_PATH"
            };
            let mut paths = vec![dir.to_path_buf()];
            if let Some(existing) = std::env::var_os(key) {
                paths.extend(std::env::split_paths(&existing));
            }
            if let Ok(joined) = std::env::join_paths(paths) {
                command.env(key, joined);
            }
        }
    }
    matches!(command.status(), Ok(status) if status.success() || status.code() == Some(0))
}

/// Ensure a llama-server binary is available, installing a managed build if needed.
pub async fn ensure_binary(client: &reqwest::Client, data_dir: &Path) -> anyhow::Result<PathBuf> {
    ensure_binary_with_progress(
        client,
        data_dir,
        RuntimeTarget::Auto,
        false,
        Box::new(|_| {}),
    )
    .await
}

pub async fn ensure_binary_with_progress(
    client: &reqwest::Client,
    data_dir: &Path,
    target: RuntimeTarget,
    force: bool,
    mut progress: ProgressCallback,
) -> anyhow::Result<PathBuf> {
    let target = if target == RuntimeTarget::Auto {
        crate::hardware::detect().recommended_target
    } else {
        target
    };
    if force {
        return install_managed_binary_with_progress(client, data_dir, target, progress).await;
    }
    let managed = managed_binary_path_for_target(data_dir, target);
    progress(ProgressEvent::phase(
        "discover",
        "Looking for an existing llama-server binary",
    ));
    let discovered = if managed.is_file() {
        Some(managed.clone())
    } else {
        discovery_candidates(
            data_dir,
            std::env::var_os("PATH").as_deref().and_then(|p| p.to_str()),
        )
        .into_iter()
        .skip(1)
        .find(|path| path.is_file())
    };
    if let Some(path) = discovered {
        if binary_appears_runnable(&path) {
            progress(ProgressEvent::done(serde_json::json!({
                "binary": path.display().to_string(),
                "status": "ready",
                "source": "discovered"
            })));
            return Ok(path);
        }
        // Broken managed install (missing .so): reinstall. Leave other PATH hits alone.
        if path != managed {
            tracing::warn!(
                binary = %path.display(),
                "discovered llama-server failed a smoke test; trying managed install"
            );
        }
    }
    install_managed_binary_with_progress(client, data_dir, target, progress).await
}

/// Running llama-server child process bound to a loopback port.
pub struct LlamaServer {
    child: Child,
    pub base_url: String,
    pub model_path: PathBuf,
    pub projector_path: Option<PathBuf>,
    pub binary: PathBuf,
    /// The launch settings this process was started with.
    ///
    /// Everything below is fixed at spawn time — a context size or a LoRA
    /// cannot be changed on a running server — so a request whose model is
    /// already loaded still has to check that it is loaded the way the model is
    /// now configured, and restart it when it is not.
    pub launch_key: String,
}

/// What llama-server is actually started with, once the model's overrides have
/// been laid over the global settings.
struct LaunchPlan {
    context_size: u32,
    batch_size: u32,
    ubatch_size: Option<u32>,
    threads: Option<u16>,
    gpu_layers: i32,
    flash_attention: bool,
    kv_cache_type_k: String,
    kv_cache_type_v: String,
    jinja: bool,
    mlock: bool,
    no_mmap: bool,
    rope_scaling: Option<String>,
    rope_freq_base: Option<f32>,
    rope_freq_scale: Option<f32>,
    yarn_orig_ctx: Option<u32>,
    n_cpu_moe: Option<u32>,
    main_gpu: Option<u32>,
    tensor_split: Option<String>,
    split_mode: Option<String>,
    cache_reuse: Option<u32>,
    defrag_threshold: Option<f32>,
    /// llama-server `--parallel` slot count.
    parallel: u32,
    loras: Vec<(PathBuf, f32)>,
    extra_args: Vec<String>,
    /// Custom Jinja chat template; written to a temp file at spawn.
    chat_template: Option<String>,
}

impl LaunchPlan {
    fn resolve(
        settings: &RuntimeSettings,
        profile: Option<&TextProfile>,
        loras: Vec<(PathBuf, f32)>,
        effective_target: RuntimeTarget,
    ) -> Self {
        let field = |get: &dyn Fn(&TextProfile) -> Option<u32>, fallback: u32| {
            profile.and_then(get).unwrap_or(fallback)
        };
        Self {
            context_size: field(&|profile| profile.context_size, settings.context_size),
            batch_size: field(&|profile| profile.batch_size, settings.batch_size),
            ubatch_size: profile.and_then(|profile| profile.ubatch_size),
            threads: profile
                .and_then(|profile| profile.threads)
                .or(settings.threads),
            gpu_layers: if effective_target == RuntimeTarget::Cpu {
                0
            } else {
                profile
                    .and_then(|profile| profile.gpu_layers)
                    .unwrap_or(settings.gpu_layers)
            },
            flash_attention: profile
                .and_then(|profile| profile.flash_attention)
                .unwrap_or(settings.flash_attention),
            kv_cache_type_k: profile
                .and_then(|profile| profile.kv_cache_type_k.clone())
                .unwrap_or_else(|| settings.kv_cache_type_k.clone()),
            kv_cache_type_v: profile
                .and_then(|profile| profile.kv_cache_type_v.clone())
                .unwrap_or_else(|| settings.kv_cache_type_v.clone()),
            jinja: profile
                .and_then(|profile| profile.jinja)
                .unwrap_or(settings.jinja),
            mlock: profile.and_then(|profile| profile.mlock).unwrap_or(false),
            no_mmap: profile.and_then(|profile| profile.no_mmap).unwrap_or(false),
            rope_scaling: profile.and_then(|profile| profile.rope_scaling.clone()),
            rope_freq_base: profile.and_then(|profile| profile.rope_freq_base),
            rope_freq_scale: profile.and_then(|profile| profile.rope_freq_scale),
            yarn_orig_ctx: profile.and_then(|profile| profile.yarn_orig_ctx),
            n_cpu_moe: profile.and_then(|profile| profile.n_cpu_moe),
            main_gpu: profile.and_then(|profile| profile.main_gpu),
            tensor_split: profile.and_then(|profile| profile.tensor_split.clone()),
            split_mode: profile.and_then(|profile| profile.split_mode.clone()),
            cache_reuse: profile.and_then(|profile| profile.cache_reuse),
            defrag_threshold: profile.and_then(|profile| profile.defrag_threshold),
            parallel: crate::model_settings::llama_parallel_slots(profile),
            loras,
            extra_args: profile
                .map(|profile| profile.extra_args.clone())
                .unwrap_or_default(),
            chat_template: profile.and_then(|profile| {
                profile
                    .chat_template
                    .as_ref()
                    .map(|text| text.trim())
                    .filter(|text| !text.is_empty())
                    .map(str::to_owned)
            }),
        }
    }

    /// A fingerprint of everything that can only be applied at spawn time.
    fn key(&self, harmony: bool) -> String {
        let reasoning_format = if harmony { "auto" } else { "deepseek" };
        let template_fp = self
            .chat_template
            .as_ref()
            .map(|text| {
                use sha2::{Digest, Sha256};
                hex::encode(Sha256::digest(text.as_bytes()))
            })
            .unwrap_or_default();
        format!(
            "{}|{}|{:?}|{:?}|{}|{}|{}|{}|{}|{}|{}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|p={}|{}|jinja|rf={}|tpl={}|{}",
            self.context_size,
            self.batch_size,
            self.ubatch_size,
            self.threads,
            self.gpu_layers,
            self.flash_attention,
            self.kv_cache_type_k,
            self.kv_cache_type_v,
            self.jinja,
            self.mlock,
            self.no_mmap,
            self.rope_scaling,
            self.rope_freq_base,
            self.rope_freq_scale,
            self.yarn_orig_ctx,
            self.n_cpu_moe,
            self.main_gpu,
            self.tensor_split,
            self.split_mode,
            self.cache_reuse,
            self.defrag_threshold,
            self.loras,
            self.parallel,
            harmony,
            reasoning_format,
            template_fp,
            self.extra_args.join(" "),
        )
    }

    fn apply(&self, command: &mut Command, _harmony: bool) {
        command
            .arg("--ctx-size")
            .arg(self.context_size.to_string())
            .arg("--batch-size")
            .arg(self.batch_size.to_string())
            .arg("--parallel")
            .arg(self.parallel.to_string())
            .arg("--n-gpu-layers")
            .arg(self.gpu_layers.to_string())
            .arg("--flash-attn")
            .arg(if self.flash_attention { "on" } else { "off" })
            .arg("--cache-type-k")
            .arg(&self.kv_cache_type_k)
            .arg("--cache-type-v")
            .arg(&self.kv_cache_type_v);
        if let Some(value) = self.ubatch_size {
            command.arg("--ubatch-size").arg(value.to_string());
        }
        if let Some(threads) = self.threads {
            command.arg("--threads").arg(threads.to_string());
        }
        // Always enable Jinja so GGUF chat templates can parse native tool-call
        // dialects (Qwen XML, Hermes JSON, …) into OpenAI `tool_calls`.
        let _ = self.jinja;
        command.arg("--jinja");
        if self.mlock {
            command.arg("--mlock");
        }
        if self.no_mmap {
            command.arg("--no-mmap");
        }
        if let Some(value) = &self.rope_scaling {
            command.arg("--rope-scaling").arg(value);
        }
        if let Some(value) = self.rope_freq_base {
            command.arg("--rope-freq-base").arg(value.to_string());
        }
        if let Some(value) = self.rope_freq_scale {
            command.arg("--rope-freq-scale").arg(value.to_string());
        }
        if let Some(value) = self.yarn_orig_ctx {
            command.arg("--yarn-orig-ctx").arg(value.to_string());
        }
        if let Some(value) = self.n_cpu_moe {
            command.arg("--n-cpu-moe").arg(value.to_string());
        }
        if let Some(value) = self.main_gpu {
            command.arg("--main-gpu").arg(value.to_string());
        }
        if let Some(value) = &self.tensor_split {
            command.arg("--tensor-split").arg(value);
        }
        if let Some(value) = &self.split_mode {
            command.arg("--split-mode").arg(value);
        }
        if let Some(value) = self.cache_reuse {
            command.arg("--cache-reuse").arg(value.to_string());
        }
        if let Some(value) = self.defrag_threshold {
            command.arg("--defrag-thold").arg(value.to_string());
        }
        for (path, scale) in &self.loras {
            command
                .arg("--lora-scaled")
                .arg(path)
                .arg(scale.to_string());
        }
        for arg in &self.extra_args {
            command.arg(arg);
        }
    }
}

/// Persist a chat template under a content-addressed temp path for `--chat-template-file`.
fn materialize_chat_template(template: &str) -> anyhow::Result<PathBuf> {
    use sha2::{Digest, Sha256};
    let hash = hex::encode(Sha256::digest(template.as_bytes()));
    let dir = std::env::temp_dir().join("brazier-chat-templates");
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let path = dir.join(format!("{}.jinja", &hash[..16.min(hash.len())]));
    if !path.is_file() {
        std::fs::write(&path, template).with_context(|| format!("write {}", path.display()))?;
    }
    Ok(path)
}

/// The fingerprint a server started with these inputs would carry.
///
/// Asked before a request is served, so a model whose configuration changed
/// since it was loaded is reloaded rather than answered from a process that no
/// longer reflects it.
pub fn launch_key(
    settings: &RuntimeSettings,
    profile: Option<&TextProfile>,
    loras: Vec<(PathBuf, f32)>,
    harmony: bool,
) -> String {
    let effective_target = if settings.target == RuntimeTarget::Auto {
        crate::hardware::detect().recommended_target
    } else {
        settings.target
    };
    LaunchPlan::resolve(settings, profile, loras, effective_target).key(harmony)
}

impl LlamaServer {
    /// Spawn llama-server for a single GGUF model on an ephemeral loopback port.
    pub async fn start(
        binary: &Path,
        model_path: &Path,
        settings: &RuntimeSettings,
        harmony: bool,
    ) -> anyhow::Result<Self> {
        Self::start_with_profile(binary, model_path, settings, harmony, None, Vec::new()).await
    }

    /// Spawn llama-server with a model's own launch overrides and LoRAs.
    pub async fn start_with_profile(
        binary: &Path,
        model_path: &Path,
        settings: &RuntimeSettings,
        harmony: bool,
        profile: Option<&TextProfile>,
        loras: Vec<(PathBuf, f32)>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            binary.is_file(),
            "llama-server binary missing: {}",
            binary.display()
        );
        anyhow::ensure!(
            model_path.is_file(),
            "model file missing: {}",
            model_path.display()
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .context("reserve port for llama-server")?;
        let port = listener.local_addr()?.port();
        drop(listener);

        let mut command = Command::new(binary);
        command
            .arg("-m")
            .arg(model_path)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let projector_path = crate::models_store::projector_for_model(model_path);
        if let Some(projector) = &projector_path {
            command.arg("--mmproj").arg(projector);
        }
        let effective_target = if settings.target == RuntimeTarget::Auto {
            crate::hardware::detect().recommended_target
        } else {
            settings.target
        };
        let plan = LaunchPlan::resolve(settings, profile, loras, effective_target);
        let launch_key = plan.key(harmony);
        plan.apply(&mut command, harmony);
        if let Some(template) = &plan.chat_template {
            let path = materialize_chat_template(template)?;
            command.arg("--chat-template-file").arg(path);
        }
        // Separate think tags into `reasoning_content` so Jinja can parse tool
        // calls from the remaining content. Harmony uses its own format value.
        command.arg("--reasoning-format").arg(if harmony {
            crate::harmony::llama_reasoning_format()
        } else {
            "deepseek"
        });
        // Managed releases ship companion .so files next to llama-server.
        if let Some(dir) = binary.parent() {
            prepend_library_path(&mut command, dir);
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("spawn {}", binary.display()))?;

        let base_url = format!("http://127.0.0.1:{port}");
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()?;
        let health_url = format!("{base_url}/health");
        let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
        loop {
            if let Some(status) = child.try_wait().context("poll llama-server")? {
                let mut stderr = String::new();
                if let Some(mut pipe) = child.stderr.take() {
                    use tokio::io::AsyncReadExt;
                    let mut buf = Vec::new();
                    let _ = pipe.read_to_end(&mut buf).await;
                    stderr = String::from_utf8_lossy(&buf).into_owned();
                }
                anyhow::bail!(describe_server_startup_failure(
                    "llama-server",
                    status,
                    &stderr
                ));
            }
            match client.get(&health_url).send().await {
                Ok(response) if response.status().is_success() => break,
                _ => {
                    if tokio::time::Instant::now() > deadline {
                        let _ = child.start_kill();
                        anyhow::bail!("llama-server health check timed out at {base_url}");
                    }
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        }

        Ok(Self {
            child,
            base_url,
            model_path: model_path.to_path_buf(),
            projector_path,
            binary: binary.to_path_buf(),
            launch_key,
        })
    }

    pub fn is_running(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(None) => true,
            Ok(Some(_)) | Err(_) => false,
        }
    }

    pub async fn stop(&mut self) -> anyhow::Result<()> {
        if self.child.try_wait()?.is_none() {
            self.child.start_kill().context("kill llama-server")?;
            let _ = self.child.wait().await;
        }
        Ok(())
    }
}

/// Query a running llama-server for health and loaded models.
pub async fn probe_server(client: &reqwest::Client, base_url: &str) -> serde_json::Value {
    use serde_json::json;
    let mut probe = json!({
        "base_url": base_url,
        "health_ok": false,
        "model_count": 0,
    });
    if let Ok(response) = client
        .get(format!("{base_url}/health"))
        .timeout(Duration::from_secs(3))
        .send()
        .await
    {
        probe["health_ok"] = json!(response.status().is_success());
        probe["health_status"] = json!(response.status().as_u16());
    }
    if let Ok(response) = client
        .get(format!("{base_url}/v1/models"))
        .timeout(Duration::from_secs(5))
        .send()
        .await
        && response.status().is_success()
        && let Ok(body) = response.json::<serde_json::Value>().await
    {
        let count = body
            .get("data")
            .and_then(serde_json::Value::as_array)
            .map(|models| models.len())
            .unwrap_or(0);
        probe["model_count"] = json!(count);
        probe["models"] = body.get("data").cloned().unwrap_or(json!([]));
    }
    probe
}

/// Send one non-streamed chat completion to a running llama-server and return
/// the full response body.
pub async fn chat_once(
    client: &reqwest::Client,
    base_url: &str,
    api_key: Option<&str>,
    body: &serde_json::Value,
) -> anyhow::Result<serde_json::Value> {
    let mut request = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(body)
        .timeout(Duration::from_secs(300));
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    let response = request.send().await.context("llama-server chat request")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("llama-server returned {status}: {text}");
    }
    response.json().await.context("decode llama-server chat")
}

/// Open a streamed chat completion against a running llama-server.
///
/// Each received item is a decoded chunk; the channel closes at end-of-stream.
pub async fn open_chat_stream(
    client: &reqwest::Client,
    base_url: &str,
    api_key: Option<&str>,
    body: &serde_json::Value,
) -> anyhow::Result<tokio::sync::mpsc::Receiver<anyhow::Result<StreamChunk>>> {
    let mut request = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(body)
        .timeout(Duration::from_secs(600));
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    let response = request
        .send()
        .await
        .context("llama-server stream request")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("llama-server stream returned {status}: {text}");
    }

    let (tx, rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(async move {
        use futures::StreamExt;
        let mut byte_stream = response.bytes_stream();
        let mut buffer = String::new();
        while let Some(chunk) = byte_stream.next().await {
            let chunk = match chunk {
                Ok(bytes) => bytes,
                Err(error) => {
                    let _ = tx
                        .send(Err(anyhow::anyhow!(
                            "llama-server stream read failed: {error}"
                        )))
                        .await;
                    return;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(frame_end) = buffer.find("\n\n") {
                let frame = buffer[..frame_end].to_owned();
                buffer = buffer[frame_end + 2..].to_owned();
                let data = frame
                    .lines()
                    .find_map(|line| line.strip_prefix("data:").map(str::trim));
                let Some(data) = data else {
                    continue;
                };
                match parse_stream_chunk(data) {
                    ChunkParse::Done => return,
                    ChunkParse::Chunk(decoded) => {
                        if tx.send(Ok(decoded)).await.is_err() {
                            return;
                        }
                    }
                    ChunkParse::Skip => {}
                }
            }
        }
    });
    Ok(rx)
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

/// Bind address helper kept for tests that spin up stub engines.
pub fn loopback_url(addr: SocketAddr) -> String {
    format!("http://{addr}")
}

fn prepend_library_path(command: &mut Command, dir: &Path) {
    #[cfg(unix)]
    {
        let key = if cfg!(target_os = "macos") {
            "DYLD_LIBRARY_PATH"
        } else {
            "LD_LIBRARY_PATH"
        };
        let mut paths = vec![dir.to_path_buf()];
        if let Some(existing) = std::env::var_os(key) {
            paths.extend(std::env::split_paths(&existing));
        }
        if let Ok(joined) = std::env::join_paths(paths) {
            command.env(key, joined);
        }
    }
    #[cfg(windows)]
    {
        let mut paths = vec![dir.to_path_buf()];
        if let Some(existing) = std::env::var_os("PATH") {
            paths.extend(std::env::split_paths(&existing));
        }
        if let Ok(joined) = std::env::join_paths(paths) {
            command.env("PATH", joined);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatCompletionRequest, OpenAiMessage};
    use serde_json::json;

    #[test]
    fn recognizes_only_llama_server_executables() {
        assert!(is_llama_server_path(Path::new(
            "/usr/local/bin/llama-server"
        )));
        // Written without a separator: `\` is not one off Windows, so a literal
        // Windows path would compare as a single file name here.
        assert!(is_llama_server_path(Path::new("llama-server.exe")));
        // The path that got pinned in place of llama-server.
        assert!(!is_llama_server_path(Path::new(
            "/data/engines/personaplex-mlx/builds/main-1/venv/bin/python"
        )));
        assert!(!is_llama_server_path(Path::new("/usr/bin/python3")));
        assert!(!is_llama_server_path(Path::new("/opt/bin/llama-cli")));
        assert!(!is_llama_server_path(Path::new("/opt/bin")));
    }

    #[test]
    fn selects_plain_ubuntu_cpu_asset() {
        let assets = [
            "llama-b10092-bin-ubuntu-vulkan-x64.tar.gz",
            "llama-b10092-bin-ubuntu-x64.tar.gz",
            "llama-b10092-bin-ubuntu-cuda-x64.tar.gz",
            "llama-b10092-bin-macos-arm64.tar.gz",
        ];
        assert_eq!(
            select_release_asset(assets, "ubuntu-x64"),
            Some("llama-b10092-bin-ubuntu-x64.tar.gz")
        );
    }

    #[test]
    fn discovery_prefers_managed_path_first() {
        let data = PathBuf::from("/tmp/brazier-data");
        let candidates = discovery_candidates(&data, Some("/usr/bin:/opt/bin"));
        assert_eq!(candidates[0], managed_binary_path(&data));
        assert!(candidates.iter().any(|p| p.ends_with("llama-server")));
    }

    #[test]
    fn preserves_multimodal_messages_for_llama_server() {
        let request = ChatCompletionRequest {
            model: "gguf:demo.gguf".into(),
            messages: vec![OpenAiMessage {
                role: "user".into(),
                content: json!([
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "AAAA", "format": "wav"},
                        "brazier_sha256": "deadbeef"
                    }
                ]),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
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
        };
        let settings = RuntimeSettings::default();
        let body = translate_chat_request(&request, ChatContext::local(&settings, "local", false));
        assert_eq!(body["model"], "local");
        assert!(!body["stream"].as_bool().unwrap());
        assert_eq!(body["messages"][0]["content"][0]["text"], "Describe");
        assert_eq!(body["messages"][0]["content"][1]["type"], "image_url");
        assert_eq!(body["messages"][0]["content"][2]["type"], "input_audio");
        assert_eq!(
            body["messages"][0]["content"][2]["input_audio"]["format"],
            "wav"
        );
        let streamed =
            translate_chat_request(&request, ChatContext::local(&settings, "local", true));
        assert!(streamed["stream"].as_bool().unwrap());
    }

    /// Floats make the round trip through JSON as doubles, so they are read
    /// back at the precision they were written with.
    fn as_f32(value: &serde_json::Value) -> f32 {
        value.as_f64().expect("a number") as f32
    }

    fn plain_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "gguf:acme/model.gguf".into(),
            messages: vec![OpenAiMessage {
                role: "user".into(),
                content: json!("Hello"),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
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
        }
    }

    /// A model's own sampling settings stand in for the global ones, and the
    /// extended llama.cpp samplers travel with them.
    #[test]
    fn a_model_profile_overrides_the_global_sampling() {
        let settings = RuntimeSettings::default();
        let profile = TextProfile {
            temperature: Some(0.2),
            top_k: Some(40),
            min_p: Some(0.05),
            repeat_penalty: Some(1.1),
            stop: vec!["<|end|>".into()],
            ..TextProfile::default()
        };
        let body = translate_chat_request(
            &plain_request(),
            ChatContext {
                settings: &settings,
                profile: Some(&profile),
                dialect: SamplerDialect::LlamaCpp,
                model_alias: "local",
                stream: false,
            },
        );
        assert_eq!(as_f32(&body["temperature"]), 0.2);
        assert_eq!(body["top_k"], 40);
        assert_eq!(as_f32(&body["min_p"]), 0.05);
        assert_eq!(as_f32(&body["repeat_penalty"]), 1.1);
        assert_eq!(body["stop"][0], "<|end|>");
        // Untouched fields still come from the global settings.
        assert_eq!(as_f32(&body["top_p"]), settings.top_p);
    }

    /// Someone else's server is not llama.cpp: sending it llama.cpp's sampler
    /// names risks a rejected request over a setting it was never going to
    /// honour anyway.
    #[test]
    fn llama_only_samplers_are_withheld_from_a_remote_server() {
        let settings = RuntimeSettings::default();
        let profile = TextProfile {
            top_k: Some(40),
            mirostat: Some(2),
            presence_penalty: Some(0.5),
            ..TextProfile::default()
        };
        let body = translate_chat_request(
            &plain_request(),
            ChatContext {
                settings: &settings,
                profile: Some(&profile),
                dialect: SamplerDialect::OpenAi,
                model_alias: "gpt-oss",
                stream: false,
            },
        );
        assert!(body.get("top_k").is_none());
        assert!(body.get("mirostat").is_none());
        // The penalties OpenAI itself defines still go.
        assert_eq!(as_f32(&body["presence_penalty"]), 0.5);
    }

    /// MLX takes a few of the same ideas under different names.
    #[test]
    fn mlx_gets_its_own_names_for_the_penalties() {
        let settings = RuntimeSettings::default();
        let profile = TextProfile {
            repeat_penalty: Some(1.15),
            repeat_last_n: Some(64),
            ..TextProfile::default()
        };
        let body = translate_chat_request(
            &plain_request(),
            ChatContext {
                settings: &settings,
                profile: Some(&profile),
                dialect: SamplerDialect::Mlx,
                model_alias: "local",
                stream: false,
            },
        );
        assert_eq!(as_f32(&body["repetition_penalty"]), 1.15);
        assert_eq!(body["repetition_context_size"], 64);
        assert!(body.get("repeat_penalty").is_none());
    }

    /// A request that names a temperature outranks the model's default, which
    /// is what makes a per-request override still mean something.
    #[test]
    fn the_request_still_wins_over_the_profile() {
        let settings = RuntimeSettings::default();
        let profile = TextProfile {
            temperature: Some(0.2),
            ..TextProfile::default()
        };
        let mut request = plain_request();
        request.temperature = Some(1.4);
        let body = translate_chat_request(
            &request,
            ChatContext {
                settings: &settings,
                profile: Some(&profile),
                dialect: SamplerDialect::LlamaCpp,
                model_alias: "local",
                stream: false,
            },
        );
        assert_eq!(as_f32(&body["temperature"]), 1.4);
    }

    #[test]
    fn a_configured_system_prompt_leads_the_conversation() {
        let settings = RuntimeSettings::default();
        let profile = TextProfile {
            system_prompt: Some("  Answer in French.  ".into()),
            ..TextProfile::default()
        };
        let body = translate_chat_request(
            &plain_request(),
            ChatContext {
                settings: &settings,
                profile: Some(&profile),
                dialect: SamplerDialect::LlamaCpp,
                model_alias: "local",
                stream: false,
            },
        );
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "Answer in French.");
        assert_eq!(body["messages"][1]["content"], "Hello");
    }

    /// Every launch setting is fixed when the process starts, so a change to one
    /// has to be visible as a different fingerprint or the running server is
    /// mistaken for a current one.
    #[test]
    fn the_launch_key_changes_with_anything_applied_at_spawn() {
        let settings = RuntimeSettings::default();
        let base = launch_key(&settings, None, Vec::new(), false);
        let resized = launch_key(
            &settings,
            Some(&TextProfile {
                context_size: Some(16_384),
                ..TextProfile::default()
            }),
            Vec::new(),
            false,
        );
        let with_lora = launch_key(
            &settings,
            None,
            vec![(PathBuf::from("/adapters/style.gguf"), 0.8)],
            false,
        );
        let with_template = launch_key(
            &settings,
            Some(&TextProfile {
                chat_template: Some("{% raw %}custom{% endraw %}".into()),
                ..TextProfile::default()
            }),
            Vec::new(),
            false,
        );
        assert_ne!(base, resized);
        assert_ne!(base, with_lora);
        assert_ne!(base, with_template);
        assert_eq!(base, launch_key(&settings, None, Vec::new(), false));

        let with_parallel = launch_key(
            &settings,
            Some(&TextProfile {
                parallel_subagents: Some(true),
                max_subagents: Some(2),
                ..TextProfile::default()
            }),
            Vec::new(),
            false,
        );
        assert_ne!(base, with_parallel);
    }

    #[test]
    fn startup_oom_detection_covers_common_backends() {
        assert!(startup_looks_like_oom("CUDA error: out of memory"));
        assert!(startup_looks_like_oom("ggml_metal: failed to allocate"));
        assert!(startup_looks_like_oom("std::bad_alloc"));
        assert!(!startup_looks_like_oom("model file not found"));
        let message = describe_server_startup_failure(
            "llama-server",
            "exit status: 1",
            "CUDA_ERROR_OUT_OF_MEMORY\nmore detail",
        );
        assert!(message.contains("ran out of memory"));
        assert!(message.contains("Parallel subagents"));
    }

    #[test]
    fn extracts_openai_style_content() {
        let body = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Hello from weights"}
            }]
        });
        assert_eq!(extract_assistant_text(&body).unwrap(), "Hello from weights");
    }

    #[test]
    fn parses_stream_payloads() {
        assert_eq!(parse_stream_data_payload("[DONE]"), StreamParse::Done);
        assert_eq!(parse_stream_data_payload(""), StreamParse::Skip);
        let delta = json!({
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]
        });
        assert_eq!(
            parse_stream_data_payload(&delta.to_string()),
            StreamParse::Content("Hello".into())
        );
        let role_only = json!({
            "choices": [{"delta": {"role": "assistant"}, "finish_reason": null}]
        });
        assert_eq!(
            parse_stream_data_payload(&role_only.to_string()),
            StreamParse::Skip
        );
    }

    #[test]
    fn extracts_release_archive_flattens_prefix() {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use tar::Builder;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let archive_path = dir.path().join("llama-test.tar.gz");
        {
            let file = std::fs::File::create(&archive_path).unwrap();
            let enc = GzEncoder::new(file, Compression::default());
            let mut builder = Builder::new(enc);
            let mut header = tar::Header::new_gnu();
            let payload = b"#!/bin/sh\necho ok\n";
            header.set_size(payload.len() as u64);
            header.set_mode(0o755);
            header.set_cksum();
            builder
                .append_data(&mut header, "llama-b1/llama-server", payload.as_slice())
                .unwrap();
            let mut header = tar::Header::new_gnu();
            let lib = b"fake-so";
            header.set_size(lib.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(
                    &mut header,
                    "llama-b1/libllama-server-impl.so",
                    lib.as_slice(),
                )
                .unwrap();
            builder.finish().unwrap();
        }
        let bin = dir.path().join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        extract_release_archive(&archive_path, &bin).unwrap();
        assert!(bin.join("llama-server").is_file());
        assert!(bin.join("libllama-server-impl.so").is_file());
    }

    #[test]
    fn parse_stream_chunk_captures_reasoning_content() {
        let data = r#"{"choices":[{"index":0,"delta":{"reasoning_content":"Let me think"},"finish_reason":null}]}"#;
        match parse_stream_chunk(data) {
            ChunkParse::Chunk(chunk) => {
                assert_eq!(chunk.reasoning.as_deref(), Some("Let me think"));
                assert!(chunk.content.is_none());
                assert!(chunk.tool_calls.is_empty());
            }
            other => panic!("expected chunk, got {other:?}"),
        }
    }

    #[test]
    fn parse_stream_chunk_skips_empty_reasoning_only_when_truly_empty() {
        let data = r#"{"choices":[{"index":0,"delta":{},"finish_reason":null}]}"#;
        assert!(matches!(parse_stream_chunk(data), ChunkParse::Skip));
    }

    #[test]
    fn message_json_round_trips_reasoning_for_jinja() {
        let message = OpenAiMessage {
            role: "assistant".into(),
            content: json!(""),
            tool_calls: Some(json!([{
                "id": "call_1",
                "type": "function",
                "function": { "name": "run_javascript", "arguments": "{\"code\":\"1+1\"}" }
            }])),
            tool_call_id: None,
            reasoning_content: Some("Need to compute.".into()),
        };
        let encoded = message_to_openai_json(&message);
        assert_eq!(encoded["reasoning_content"], "Need to compute.");
        assert_eq!(
            encoded["tool_calls"][0]["function"]["name"],
            "run_javascript"
        );
    }

    #[test]
    fn extract_reasoning_reads_message_field() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "done",
                    "reasoning_content": "step by step"
                }
            }]
        });
        assert_eq!(extract_reasoning(&body).as_deref(), Some("step by step"));
    }
}

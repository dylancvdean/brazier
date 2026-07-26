//! Bundled, safe built-in tools the daemon can execute on behalf of a model.
//!
//! Tools are intentionally conservative: no filesystem access, no shell, and
//! web retrieval is bounded (size, time, and private-network guard). JavaScript
//! runs in an isolated QuickJS runtime with memory, stack, and time limits.

use std::time::Duration;

use anyhow::Context;
use serde_json::{Value, json};

const FETCH_MAX_BYTES: usize = 256 * 1024;
const FETCH_MAX_OUTPUT_CHARS: usize = 8_000;
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);
const JS_SANDBOX_TIMEOUT: Duration = Duration::from_secs(2);

/// A completed built-in tool invocation, suitable for UI display and for the
/// `tool` role message returned to the model.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolInvocation {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub output: String,
    pub is_error: bool,
    /// Blobs this call produced, so the caller can offer them back to a model
    /// that can see them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<ToolMedia>,
}

/// A blob produced by a tool call.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolMedia {
    pub sha256: String,
    pub mime_type: String,
}

/// Tool definitions with the generation tools described for this machine.
///
/// Whether a photo can be handed to `generate_video` is not a property of the
/// tool but of the model installed behind it: a text-to-video model rejects an
/// init image outright. Saying which one is configured — and what it accepts —
/// is what stops a model from confidently passing a picture to a model that
/// cannot take one.
pub fn definitions_for(data_dir: &std::path::Path) -> Value {
    let mut defs = definitions();
    let settings = crate::runtime_settings::load(data_dir);
    let Some(items) = defs.as_array_mut() else {
        return defs;
    };
    for item in items {
        let Some(name) = item.pointer("/function/name").and_then(Value::as_str) else {
            continue;
        };
        let note = match name {
            "generate_video" => Some(describe_video_model(data_dir, &settings)),
            "generate_image" => Some(describe_image_model(&settings)),
            _ => None,
        };
        if let Some(note) = note
            && let Some(description) = item.pointer_mut("/function/description")
            && let Some(text) = description.as_str()
        {
            *description = Value::String(format!("{text} {note}"));
        }
    }
    defs
}

/// What the configured video model is, and whether it can animate a picture.
fn describe_video_model(
    data_dir: &std::path::Path,
    settings: &crate::runtime_settings::RuntimeSettings,
) -> String {
    let Some(model_id) = settings.default_video_gen_model.as_deref() else {
        return "No default video model is configured yet, so this tool cannot run until the \
                user installs one."
            .to_owned();
    };
    if crate::sdcpp::supports_init_image(data_dir, model_id) {
        format!(
            "The configured model (`{model_id}`) is an image-to-video model: it accepts \
             `init_image`, so a photo the user shared can be animated."
        )
    } else {
        format!(
            "The configured model (`{model_id}`) is text-to-video only: it has no `init_image` \
             support and will refuse one. To animate a picture the user shared, tell them they \
             need an image-to-video model such as Wan 2.2 TI2V or LTX-2.3 instead of guessing."
        )
    }
}

fn describe_image_model(settings: &crate::runtime_settings::RuntimeSettings) -> String {
    match settings.default_image_gen_model.as_deref() {
        Some(model_id) => format!("The configured model is `{model_id}`."),
        None => "No default image model is configured yet, so this tool cannot run until the \
                 user installs one."
            .to_owned(),
    }
}

/// OpenAI-style tool definitions for every bundled tool.
pub fn definitions() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current date and time in UTC, plus the Unix timestamp.",
                "parameters": { "type": "object", "properties": {}, "required": [] }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate an arithmetic expression. Supports + - * / % ^, parentheses, and unary minus. Numbers are 64-bit floats.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The expression to evaluate, e.g. `(2 + 3) * 4 ^ 2`."
                        }
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch a public http(s) URL and return its text content. Responses are truncated; HTML is reduced to text. Private and local addresses are blocked.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": { "type": "string", "description": "Absolute http:// or https:// URL." }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_javascript",
                "description": "Run JavaScript in a sandboxed QuickJS environment. No network, filesystem, or host APIs. Use for data transforms, date math, or small algorithms. Use `return` to produce a JSON-serializable result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "JavaScript source executed inside a strict-mode function body."
                        }
                    },
                    "required": ["code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate an image with the configured local stable-diffusion.cpp model. Returns a brazier_blob reference the user can view. Requires an installed sd-cli runtime and a default image generation model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text description of the image to generate."
                        },
                        "negative_prompt": {
                            "type": "string",
                            "description": "Optional negative prompt."
                        },
                        "width": { "type": "integer", "description": "Image width in pixels (default 512)." },
                        "height": { "type": "integer", "description": "Image height in pixels (default 512)." },
                        "steps": { "type": "integer", "description": "Diffusion steps (default 20)." },
                        "seed": { "type": "integer", "description": "Optional RNG seed." },
                        "init_image": {
                            "type": "string",
                            "description": "Start from an image the user shared instead of noise: pass `latest` for the most recent one, or its brazier_blob hash."
                        }
                    },
                    "required": ["prompt"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_video",
                "description": "Generate a short video with the configured local stable-diffusion.cpp Wan/LTX model. Returns a brazier_blob reference. Requires an installed sd-cli runtime and a default video generation model. Video models come in two kinds: text-to-video builds the clip from the prompt alone, while image-to-video starts from a picture passed as `init_image`. Only pass `init_image` when the configured model supports it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text description of the video to generate."
                        },
                        "negative_prompt": {
                            "type": "string",
                            "description": "Optional negative prompt."
                        },
                        "width": { "type": "integer", "description": "Frame width in pixels (default 512)." },
                        "height": { "type": "integer", "description": "Frame height in pixels (default 512)." },
                        "steps": { "type": "integer", "description": "Diffusion steps (default 20)." },
                        "video_frames": { "type": "integer", "description": "Number of frames (default 16)." },
                        "seed": { "type": "integer", "description": "Optional RNG seed." },
                        "init_image": {
                            "type": "string",
                            "description": "Animate an image the user shared: pass `latest` for the most recent one, or its brazier_blob hash. Only works with image-to-video models."
                        }
                    },
                    "required": ["prompt"]
                }
            }
        }
    ])
}

/// Human-facing catalog used by the `/api/v1/tools` endpoint.
pub fn catalog() -> Value {
    json!({
        "data": [
            {
                "name": "get_current_time",
                "title": "Current time",
                "description": "Current UTC date, time, and Unix timestamp.",
                "network": false,
                "source": "builtin"
            },
            {
                "name": "calculator",
                "title": "Calculator",
                "description": "Arithmetic expressions with + - * / % ^ and parentheses.",
                "network": false,
                "source": "builtin"
            },
            {
                "name": "fetch_url",
                "title": "Web fetch",
                "description": format!(
                    "Bounded retrieval of public web pages (max {} KB, {}s timeout). Local and private addresses are blocked.",
                    FETCH_MAX_BYTES / 1024,
                    FETCH_TIMEOUT.as_secs()
                ),
                "network": true,
                "source": "builtin"
            },
            {
                "name": "run_javascript",
                "title": "JavaScript sandbox",
                "description": format!(
                    "QuickJS sandbox ({} KB code limit, {}s timeout, no I/O).",
                    crate::js_sandbox::MAX_CODE_BYTES / 1024,
                    JS_SANDBOX_TIMEOUT.as_secs()
                ),
                "network": false,
                "source": "builtin"
            },
            {
                "name": "generate_image",
                "title": "Generate image",
                "description": "Local image generation via stable-diffusion.cpp (requires runtime + default model).",
                "network": false,
                "source": "builtin"
            },
            {
                "name": "generate_video",
                "title": "Generate video",
                "description": "Local video generation via stable-diffusion.cpp (requires runtime + default model).",
                "network": false,
                "source": "builtin"
            }
        ]
    })
}

pub fn is_builtin(name: &str) -> bool {
    matches!(
        name,
        "get_current_time"
            | "calculator"
            | "fetch_url"
            | "run_javascript"
            | "generate_image"
            | "generate_video"
    )
}

/// Execute one bundled tool call. Errors are folded into the output string so
/// the model can react to them; `is_error` marks them for the UI.
pub async fn execute(
    client: &reqwest::Client,
    call_id: &str,
    name: &str,
    arguments: &str,
) -> ToolInvocation {
    execute_with_context(client, None, &[], call_id, name, arguments).await
}

pub async fn execute_with_data_dir(
    client: &reqwest::Client,
    data_dir: Option<&std::path::Path>,
    call_id: &str,
    name: &str,
    arguments: &str,
) -> ToolInvocation {
    execute_with_context(client, data_dir, &[], call_id, name, arguments).await
}

/// Run a built-in tool with the conversation's images available, so generation
/// tools can start from a photo the user already attached.
pub async fn execute_with_context(
    client: &reqwest::Client,
    data_dir: Option<&std::path::Path>,
    images: &[crate::tool_registry::ConversationImage],
    call_id: &str,
    name: &str,
    arguments: &str,
) -> ToolInvocation {
    let parsed: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
    let result: anyhow::Result<ToolOutput> = match name {
        "generate_image" => match data_dir {
            Some(dir) => generate_image_tool(dir, &parsed, images).await,
            None => Err(anyhow::anyhow!(
                "generate_image requires daemon data directory context"
            )),
        },
        "generate_video" => match data_dir {
            Some(dir) => generate_video_tool(dir, &parsed, images).await,
            None => Err(anyhow::anyhow!(
                "generate_video requires daemon data directory context"
            )),
        },
        other => simple_tool(client, other, &parsed)
            .await
            .map(ToolOutput::from),
    };
    match result {
        Ok(output) => ToolInvocation {
            call_id: call_id.to_owned(),
            name: name.to_owned(),
            arguments: arguments.to_owned(),
            output: output.text,
            is_error: false,
            media: output.media,
        },
        Err(error) => ToolInvocation {
            call_id: call_id.to_owned(),
            name: name.to_owned(),
            arguments: arguments.to_owned(),
            output: format!("Error: {error:#}"),
            is_error: true,
            media: Vec::new(),
        },
    }
}

/// Built-in tools that return plain text.
async fn simple_tool(
    client: &reqwest::Client,
    name: &str,
    parsed: &Value,
) -> anyhow::Result<String> {
    match name {
        "get_current_time" => Ok(current_time()),
        "calculator" => parsed
            .get("expression")
            .and_then(Value::as_str)
            .context("calculator requires an `expression` string argument")
            .and_then(|expression| {
                evaluate(expression).map(|value| {
                    if value == value.trunc() && value.abs() < 1e15 {
                        format!("{}", value as i64)
                    } else {
                        format!("{value}")
                    }
                })
            }),
        "fetch_url" => match parsed.get("url").and_then(Value::as_str) {
            Some(url) => fetch_url(client, url).await,
            None => Err(anyhow::anyhow!(
                "fetch_url requires a `url` string argument"
            )),
        },
        "run_javascript" => match parsed.get("code").and_then(Value::as_str) {
            Some(code) => {
                let code = code.to_owned();
                match tokio::task::spawn_blocking(move || {
                    crate::js_sandbox::run_javascript(&code, JS_SANDBOX_TIMEOUT)
                })
                .await
                {
                    Ok(Ok(output)) => Ok(output),
                    Ok(Err(error)) => Err(error),
                    Err(join_error) => Err(anyhow::anyhow!(
                        "javascript sandbox task failed: {join_error}"
                    )),
                }
            }
            None => Err(anyhow::anyhow!(
                "run_javascript requires a `code` string argument"
            )),
        },
        other => Err(anyhow::anyhow!("unknown built-in tool `{other}`")),
    }
}

/// What a built-in tool produced: text for the model, plus any blobs.
struct ToolOutput {
    text: String,
    media: Vec<ToolMedia>,
}

impl From<String> for ToolOutput {
    fn from(text: String) -> Self {
        Self {
            text,
            media: Vec::new(),
        }
    }
}

/// Restate a generation failure in terms the calling model can act on.
///
/// A stop is not an engine fault: the person watched the prompt, decided it was
/// not what they wanted, and stopped it. Saying so plainly is what lets the
/// model wait for direction instead of dutifully generating the same thing
/// again.
fn describe_generation_failure(error: anyhow::Error) -> anyhow::Error {
    if error
        .downcast_ref::<crate::sdcpp::CancelledError>()
        .is_some()
    {
        return anyhow::anyhow!(
            "The user stopped this generation before it finished. Do not start it again on your \
             own — acknowledge the interruption and ask what they would like changed."
        );
    }
    error
}

/// An init image resolved to both its file and the blob it came from.
struct InitImage {
    path: std::path::PathBuf,
    sha256: String,
}

/// Resolve an `init_image` argument to a stored blob.
///
/// The model can pass `latest` to mean the most recent image in the
/// conversation, which is what it has when a user attaches a photo and asks
/// for it to be animated or restyled, or an explicit blob hash.
fn resolve_init_image(
    data_dir: &std::path::Path,
    args: &Value,
    images: &[crate::tool_registry::ConversationImage],
) -> anyhow::Result<Option<InitImage>> {
    let Some(requested) = args.get("init_image").and_then(Value::as_str) else {
        return Ok(None);
    };
    let requested = requested.trim();
    if requested.is_empty() || requested.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    let sha256 = if requested.eq_ignore_ascii_case("latest") {
        images
            .last()
            .map(|image| image.sha256.clone())
            .context("no image has been shared in this conversation yet")?
    } else {
        requested
            .strip_prefix("brazier_blob:")
            .unwrap_or(requested)
            .to_owned()
    };
    let path = crate::blob_store::blob_path(data_dir, &sha256)
        .context("that image is not in this conversation")?;
    anyhow::ensure!(path.is_file(), "that image is no longer stored locally");
    Ok(Some(InitImage { path, sha256 }))
}

async fn generate_image_tool(
    data_dir: &std::path::Path,
    args: &Value,
    images: &[crate::tool_registry::ConversationImage],
) -> anyhow::Result<ToolOutput> {
    let prompt = args
        .get("prompt")
        .and_then(Value::as_str)
        .context("generate_image requires a `prompt` string")?;
    let settings = crate::runtime_settings::load(data_dir);
    let model_id = settings
        .default_image_gen_model
        .clone()
        .context("no default image generation model configured (set one in Manage → Engine)")?;
    let init_image = resolve_init_image(data_dir, args, images)?;
    let request = crate::sdcpp::GenerateImageRequest {
        prompt: prompt.to_owned(),
        model_id,
        negative_prompt: args
            .get("negative_prompt")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        width: args
            .get("width")
            .and_then(Value::as_u64)
            .map(|v| v as u32)
            .unwrap_or(512),
        height: args
            .get("height")
            .and_then(Value::as_u64)
            .map(|v| v as u32)
            .unwrap_or(512),
        steps: args
            .get("steps")
            .and_then(Value::as_u64)
            .map(|v| v as u32)
            .unwrap_or(20),
        seed: args.get("seed").and_then(Value::as_i64),
        cfg_scale: args
            .get("cfg_scale")
            .and_then(Value::as_f64)
            .map(|v| v as f32),
        guidance: args
            .get("guidance")
            .and_then(Value::as_f64)
            .map(|v| v as f32),
        init_image: init_image.as_ref().map(|image| image.path.clone()),
        init_image_blob: init_image.as_ref().map(|image| image.sha256.clone()),
        origin: crate::sdcpp::GenerationOrigin::Model,
        timeout_secs: Some(settings.generation_timeout_secs),
    };
    let result = crate::sdcpp::generate_image(data_dir, settings.sdcpp_binary.as_deref(), &request)
        .await
        .map_err(describe_generation_failure)?;
    let bytes = tokio::fs::read(&result.output_path)
        .await
        .context("read generated image")?;
    let blob = crate::blob_store::store_bytes(data_dir, &bytes, "image/png", Some("generated.png"))
        .await?;
    let _ = tokio::fs::remove_file(&result.output_path).await;
    Ok(ToolOutput {
        text: format!(
            "Generated image stored as brazier_blob:{} ({} bytes).",
            blob.sha256, blob.size_bytes
        ),
        media: vec![ToolMedia {
            sha256: blob.sha256.clone(),
            mime_type: "image/png".to_owned(),
        }],
    })
}

async fn generate_video_tool(
    data_dir: &std::path::Path,
    args: &Value,
    images: &[crate::tool_registry::ConversationImage],
) -> anyhow::Result<ToolOutput> {
    let prompt = args
        .get("prompt")
        .and_then(Value::as_str)
        .context("generate_video requires a `prompt` string")?;
    let settings = crate::runtime_settings::load(data_dir);
    let model_id = settings
        .default_video_gen_model
        .clone()
        .context("no default video generation model configured (set one in Manage → Engine)")?;
    let init_image = resolve_init_image(data_dir, args, images)?;
    if init_image.is_some() {
        anyhow::ensure!(
            crate::sdcpp::supports_init_image(data_dir, &model_id),
            "`{model_id}` is text-to-video only. Install an image-to-video model (for example Wan 2.2 TI2V) to animate an attached photo."
        );
    }
    let request = crate::sdcpp::GenerateVideoRequest {
        prompt: prompt.to_owned(),
        model_id,
        negative_prompt: args
            .get("negative_prompt")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        width: args
            .get("width")
            .and_then(Value::as_u64)
            .map(|v| v as u32)
            .unwrap_or(512),
        height: args
            .get("height")
            .and_then(Value::as_u64)
            .map(|v| v as u32)
            .unwrap_or(512),
        steps: args
            .get("steps")
            .and_then(Value::as_u64)
            .map(|v| v as u32)
            .unwrap_or(20),
        seed: args.get("seed").and_then(Value::as_i64),
        cfg_scale: args
            .get("cfg_scale")
            .and_then(Value::as_f64)
            .map(|v| v as f32),
        guidance: args
            .get("guidance")
            .and_then(Value::as_f64)
            .map(|v| v as f32),
        init_image: init_image.as_ref().map(|image| image.path.clone()),
        init_image_blob: init_image.as_ref().map(|image| image.sha256.clone()),
        origin: crate::sdcpp::GenerationOrigin::Model,
        timeout_secs: Some(settings.generation_timeout_secs),
        video_frames: args
            .get("video_frames")
            .and_then(Value::as_u64)
            .map(|v| v as u32)
            .unwrap_or(16),
        fps: args.get("fps").and_then(Value::as_u64).map(|v| v as u32),
    };
    let result = crate::sdcpp::generate_video(data_dir, settings.sdcpp_binary.as_deref(), &request)
        .await
        .map_err(describe_generation_failure)?;
    let bytes = tokio::fs::read(&result.output_path)
        .await
        .context("read generated video")?;
    let mime = if result
        .output_path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("webm"))
    {
        "video/webm"
    } else {
        "video/mp4"
    };
    let blob = crate::blob_store::store_bytes(
        data_dir,
        &bytes,
        mime,
        Some(
            result
                .output_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("generated.mp4"),
        ),
    )
    .await?;
    let _ = tokio::fs::remove_file(&result.output_path).await;
    Ok(ToolOutput {
        text: format!(
            "Generated video stored as brazier_blob:{} ({} bytes).",
            blob.sha256, blob.size_bytes
        ),
        media: vec![ToolMedia {
            sha256: blob.sha256.clone(),
            mime_type: mime.to_owned(),
        }],
    })
}

fn current_time() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = now.as_secs() as i64;
    let (year, month, day, hour, minute, second) = civil_from_unix(seconds);
    format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z (Unix timestamp {seconds})"
    )
}

/// Convert a Unix timestamp to civil UTC date-time (Howard Hinnant's algorithm).
fn civil_from_unix(seconds: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let hour = (seconds_of_day / 3_600) as u32;
    let minute = ((seconds_of_day % 3_600) / 60) as u32;
    let second = (seconds_of_day % 60) as u32;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if month <= 2 { year + 1 } else { year };
    (year, month, day, hour, minute, second)
}

// --- Calculator: recursive-descent parser over f64 ---------------------------

struct Parser<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            position: 0,
        }
    }

    fn skip_whitespace(&mut self) {
        while self.position < self.input.len() && self.input[self.position].is_ascii_whitespace() {
            self.position += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_whitespace();
        self.input.get(self.position).copied()
    }

    fn expression(&mut self) -> anyhow::Result<f64> {
        let mut value = self.term()?;
        while let Some(op) = self.peek() {
            match op {
                b'+' => {
                    self.position += 1;
                    value += self.term()?;
                }
                b'-' => {
                    self.position += 1;
                    value -= self.term()?;
                }
                _ => break,
            }
        }
        Ok(value)
    }

    fn term(&mut self) -> anyhow::Result<f64> {
        let mut value = self.power()?;
        while let Some(op) = self.peek() {
            match op {
                b'*' => {
                    self.position += 1;
                    value *= self.power()?;
                }
                b'/' => {
                    self.position += 1;
                    value /= self.power()?;
                }
                b'%' => {
                    self.position += 1;
                    value %= self.power()?;
                }
                _ => break,
            }
        }
        Ok(value)
    }

    fn power(&mut self) -> anyhow::Result<f64> {
        let base = self.unary()?;
        if self.peek() == Some(b'^') {
            self.position += 1;
            // Right-associative.
            let exponent = self.power()?;
            return Ok(base.powf(exponent));
        }
        Ok(base)
    }

    fn unary(&mut self) -> anyhow::Result<f64> {
        match self.peek() {
            Some(b'-') => {
                self.position += 1;
                Ok(-self.unary()?)
            }
            Some(b'+') => {
                self.position += 1;
                self.unary()
            }
            _ => self.atom(),
        }
    }

    fn atom(&mut self) -> anyhow::Result<f64> {
        match self.peek() {
            Some(b'(') => {
                self.position += 1;
                let value = self.expression()?;
                anyhow::ensure!(self.peek() == Some(b')'), "expected closing parenthesis");
                self.position += 1;
                Ok(value)
            }
            Some(byte) if byte.is_ascii_digit() || byte == b'.' => {
                let start = self.position;
                while self
                    .input
                    .get(self.position)
                    .is_some_and(|b| b.is_ascii_digit() || *b == b'.' || *b == b'e' || *b == b'E')
                {
                    // Allow exponent sign directly after e/E.
                    if matches!(self.input[self.position], b'e' | b'E')
                        && matches!(self.input.get(self.position + 1), Some(b'+') | Some(b'-'))
                    {
                        self.position += 1;
                    }
                    self.position += 1;
                }
                let text = std::str::from_utf8(&self.input[start..self.position])?;
                text.parse::<f64>()
                    .map_err(|_| anyhow::anyhow!("invalid number `{text}`"))
            }
            Some(byte) => anyhow::bail!("unexpected character `{}`", byte as char),
            None => anyhow::bail!("unexpected end of expression"),
        }
    }
}

/// Evaluate an arithmetic expression.
pub fn evaluate(expression: &str) -> anyhow::Result<f64> {
    anyhow::ensure!(expression.len() <= 1_000, "expression is too long");
    let mut parser = Parser::new(expression);
    let value = parser.expression()?;
    parser.skip_whitespace();
    anyhow::ensure!(
        parser.position == parser.input.len(),
        "unexpected trailing input in expression"
    );
    anyhow::ensure!(
        value.is_finite(),
        "expression did not evaluate to a finite number"
    );
    Ok(value)
}

// --- Bounded web retrieval ----------------------------------------------------

fn ip_is_public(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            !(v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_documentation()
                // Carrier-grade NAT 100.64.0.0/10.
                || (v4.octets()[0] == 100 && (64..=127).contains(&v4.octets()[1])))
        }
        std::net::IpAddr::V6(v6) => {
            !(v6.is_loopback()
                || v6.is_unspecified()
                || (v6.segments()[0] & 0xfe00) == 0xfc00 // unique local fc00::/7
                || (v6.segments()[0] & 0xffc0) == 0xfe80) // link local fe80::/10
        }
    }
}

async fn guard_host(host: &str) -> anyhow::Result<()> {
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        anyhow::ensure!(ip_is_public(ip), "address {ip} is not publicly routable");
        return Ok(());
    }
    anyhow::ensure!(
        !host.eq_ignore_ascii_case("localhost") && !host.ends_with(".local"),
        "local hostnames are not allowed"
    );
    let addresses = tokio::net::lookup_host((host, 443))
        .await
        .with_context(|| format!("resolve host {host}"))?;
    for address in addresses {
        anyhow::ensure!(
            ip_is_public(address.ip()),
            "host {host} resolves to non-public address {}",
            address.ip()
        );
    }
    Ok(())
}

/// Strip tags from HTML and collapse whitespace. Deliberately simple.
pub fn html_to_text(html: &str) -> String {
    let mut output = String::with_capacity(html.len() / 2);
    let mut chars = html.char_indices().peekable();
    let mut skip_until: Option<&str> = None;
    let lower = html.to_ascii_lowercase();
    while let Some((index, character)) = chars.next() {
        if let Some(end_tag) = skip_until {
            if character == '<' && lower[index..].starts_with(end_tag) {
                skip_until = None;
                // Consume through the closing '>'.
                for (_, inner) in chars.by_ref() {
                    if inner == '>' {
                        break;
                    }
                }
            }
            continue;
        }
        if character == '<' {
            if lower[index..].starts_with("<script") {
                skip_until = Some("</script");
                continue;
            }
            if lower[index..].starts_with("<style") {
                skip_until = Some("</style");
                continue;
            }
            for (_, inner) in chars.by_ref() {
                if inner == '>' {
                    break;
                }
            }
            output.push(' ');
            continue;
        }
        output.push(character);
    }
    // Collapse runs of whitespace but keep paragraph-ish newlines.
    let mut collapsed = String::with_capacity(output.len());
    let mut last_was_space = true;
    for character in output.chars() {
        if character.is_whitespace() {
            if !last_was_space {
                collapsed.push(' ');
                last_was_space = true;
            }
        } else {
            collapsed.push(character);
            last_was_space = false;
        }
    }
    collapsed.trim().to_owned()
}

async fn fetch_url(client: &reqwest::Client, url: &str) -> anyhow::Result<String> {
    let parsed = reqwest::Url::parse(url).context("invalid URL")?;
    anyhow::ensure!(
        matches!(parsed.scheme(), "http" | "https"),
        "only http and https URLs are supported"
    );
    let host = parsed.host_str().context("URL has no host")?;
    guard_host(host).await?;

    let response = client
        .get(parsed)
        .header("user-agent", "brazier-tools/0.1 (+bounded-fetch)")
        .timeout(FETCH_TIMEOUT)
        .send()
        .await
        .context("request failed")?;
    let status = response.status();
    anyhow::ensure!(status.is_success(), "server returned {status}");
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    anyhow::ensure!(
        content_type.is_empty()
            || content_type.contains("text/")
            || content_type.contains("json")
            || content_type.contains("xml"),
        "unsupported content type `{content_type}`"
    );

    use futures::StreamExt;
    let mut stream = response.bytes_stream();
    let mut bytes: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read response body")?;
        bytes.extend_from_slice(&chunk);
        if bytes.len() >= FETCH_MAX_BYTES {
            bytes.truncate(FETCH_MAX_BYTES);
            break;
        }
    }
    let body = String::from_utf8_lossy(&bytes);
    let mut text = if content_type.contains("html") {
        html_to_text(&body)
    } else {
        body.into_owned()
    };
    if text.len() > FETCH_MAX_OUTPUT_CHARS {
        let mut cut = FETCH_MAX_OUTPUT_CHARS;
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        text.truncate(cut);
        text.push_str("… [truncated]");
    }
    anyhow::ensure!(!text.is_empty(), "response contained no text");
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculator_handles_precedence_and_parens() {
        assert_eq!(evaluate("2 + 3 * 4").unwrap(), 14.0);
        assert_eq!(evaluate("(2 + 3) * 4").unwrap(), 20.0);
        assert_eq!(evaluate("2 ^ 3 ^ 2").unwrap(), 512.0); // right-assoc
        assert_eq!(evaluate("-4 + 10 % 3").unwrap(), -3.0);
        assert_eq!(evaluate("1.5e2 / 3").unwrap(), 50.0);
    }

    #[test]
    fn calculator_rejects_garbage() {
        assert!(evaluate("2 +").is_err());
        assert!(evaluate("(1").is_err());
        assert!(evaluate("2; rm -rf /").is_err());
        assert!(evaluate("1/0").is_err()); // infinity is rejected
    }

    #[test]
    fn time_tool_formats_a_known_timestamp() {
        assert_eq!(civil_from_unix(0), (1970, 1, 1, 0, 0, 0));
        assert_eq!(civil_from_unix(1_753_000_000), (2025, 7, 20, 8, 26, 40));
    }

    #[test]
    fn html_reduction_strips_scripts_and_tags() {
        let html = "<html><head><script>alert(1)</script><style>p{}</style></head>\
                    <body><h1>Title</h1><p>Hello <b>world</b></p></body></html>";
        assert_eq!(html_to_text(html), "Title Hello world");
    }

    #[test]
    fn private_addresses_are_blocked() {
        assert!(!ip_is_public("127.0.0.1".parse().unwrap()));
        assert!(!ip_is_public("10.1.2.3".parse().unwrap()));
        assert!(!ip_is_public("192.168.1.1".parse().unwrap()));
        assert!(!ip_is_public("169.254.169.254".parse().unwrap()));
        assert!(!ip_is_public("100.100.0.1".parse().unwrap()));
        assert!(!ip_is_public("fe80::1".parse().unwrap()));
        assert!(!ip_is_public("::1".parse().unwrap()));
        assert!(ip_is_public("93.184.216.34".parse().unwrap()));
        assert!(ip_is_public(
            "2606:2800:220:1:248:1893:25c8:1946".parse().unwrap()
        ));
    }

    #[tokio::test]
    async fn execute_reports_unknown_tool_as_error() {
        let client = reqwest::Client::new();
        let result = execute(&client, "call_1", "launch_missiles", "{}").await;
        assert!(result.is_error);
        assert!(result.output.contains("unknown"));
    }

    #[tokio::test]
    async fn execute_calculator_round_trip() {
        let client = reqwest::Client::new();
        let result = execute(&client, "call_1", "calculator", "{\"expression\": \"6*7\"}").await;
        assert!(!result.is_error);
        assert_eq!(result.output, "42");
    }

    #[tokio::test]
    async fn execute_javascript_round_trip() {
        let client = reqwest::Client::new();
        let result = execute(
            &client,
            "call_1",
            "run_javascript",
            "{\"code\": \"return 6*7;\"}",
        )
        .await;
        assert!(!result.is_error);
        assert_eq!(result.output, "42");
    }
}

#[cfg(all(test, unix))]
mod generation_tests {
    use super::*;
    use crate::tool_registry::ConversationImage;

    /// A stand-in for sd-cli that records its argv and writes an output file.
    fn stub_sd_cli(dir: &std::path::Path) -> std::path::PathBuf {
        let path = dir.join("fake-sd-cli");
        // Records next to itself rather than via an env var, so tests running
        // in parallel cannot clobber each other's log.
        std::fs::write(
            &path,
            r#"#!/bin/sh
printf '%s\n' "$@" > "$0.argv"
prev=""; out=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
[ -n "$out" ] && printf 'x' > "$out"
exit 0
"#,
        )
        .unwrap();
        std::fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o755))
            .unwrap();
        path
    }

    fn install_video_model(data_dir: &std::path::Path, key: &str, supports_init_image: bool) {
        let dir = crate::sdcpp::video_root(data_dir).join(key);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), b"weights").unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec(&serde_json::json!({
                "modality": "video",
                "args": {},
                "single_file": "model.safetensors",
                "supports_init_image": supports_init_image
            }))
            .unwrap(),
        )
        .unwrap();
    }

    async fn setup(supports_init_image: bool) -> (tempfile::TempDir, String, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().to_path_buf();
        let binary = stub_sd_cli(&data_dir);
        let argv_log = data_dir.join("fake-sd-cli.argv");
        install_video_model(&data_dir, "test/model", supports_init_image);
        let mut settings = crate::runtime_settings::RuntimeSettings::default();
        settings.sdcpp_binary = Some(binary.display().to_string());
        settings.default_video_gen_model = Some("sdcpp-video:test/model".into());
        std::fs::write(
            crate::runtime_settings::settings_path(&data_dir),
            serde_json::to_vec(&settings).unwrap(),
        )
        .unwrap();
        let blob =
            crate::blob_store::store_bytes(&data_dir, b"fake png", "image/png", Some("p.png"))
                .await
                .unwrap();
        (dir, blob.sha256, argv_log)
    }

    #[tokio::test]
    async fn an_attached_photo_is_passed_to_an_image_to_video_model() {
        let (dir, sha256, argv_log) = setup(true).await;
        let images = vec![ConversationImage {
            sha256: sha256.clone(),
            mime_type: "image/png".into(),
        }];
        let invocation = execute_with_context(
            &reqwest::Client::new(),
            Some(dir.path()),
            &images,
            "call-1",
            "generate_video",
            r#"{"prompt":"make it move","init_image":"latest"}"#,
        )
        .await;

        assert!(!invocation.is_error, "{}", invocation.output);
        let argv = std::fs::read_to_string(&argv_log).unwrap();
        assert!(argv.contains("-i\n"), "init image flag missing:\n{argv}");
        assert!(
            argv.contains(&sha256[..16]),
            "the attached photo should be the init image:\n{argv}"
        );
        // The result advertises its blob so the chat loop can show it back.
        assert_eq!(invocation.media.len(), 1);
        assert!(invocation.media[0].mime_type.starts_with("video/"));
    }

    #[tokio::test]
    async fn text_to_video_models_refuse_an_init_image_with_a_useful_message() {
        let (dir, sha256, _) = setup(false).await;
        let images = vec![ConversationImage {
            sha256,
            mime_type: "image/png".into(),
        }];
        let invocation = execute_with_context(
            &reqwest::Client::new(),
            Some(dir.path()),
            &images,
            "call-2",
            "generate_video",
            r#"{"prompt":"make it move","init_image":"latest"}"#,
        )
        .await;
        assert!(invocation.is_error);
        assert!(
            invocation.output.contains("image-to-video"),
            "{}",
            invocation.output
        );
    }

    #[tokio::test]
    async fn latest_needs_an_image_to_have_been_shared() {
        let (dir, _, _) = setup(true).await;
        let invocation = execute_with_context(
            &reqwest::Client::new(),
            Some(dir.path()),
            &[],
            "call-3",
            "generate_video",
            r#"{"prompt":"move","init_image":"latest"}"#,
        )
        .await;
        assert!(invocation.is_error);
        assert!(
            invocation.output.contains("no image"),
            "{}",
            invocation.output
        );
    }

    fn video_tool_description(defs: &Value) -> String {
        defs.as_array()
            .expect("array")
            .iter()
            .find(|item| {
                item.pointer("/function/name").and_then(Value::as_str) == Some("generate_video")
            })
            .and_then(|item| item.pointer("/function/description"))
            .and_then(Value::as_str)
            .expect("generate_video description")
            .to_owned()
    }

    /// Refusing a bad call is a poor substitute for not making it: the model
    /// should be able to read which kind of video model is installed.
    #[tokio::test]
    async fn the_video_tool_says_whether_it_can_take_a_picture() {
        let (i2v, _, _) = setup(true).await;
        let described = video_tool_description(&definitions_for(i2v.path()));
        assert!(
            described.contains("image-to-video model: it accepts"),
            "{described}"
        );

        let (t2v, _, _) = setup(false).await;
        let described = video_tool_description(&definitions_for(t2v.path()));
        assert!(described.contains("text-to-video only"), "{described}");
        assert!(
            described.contains("sdcpp-video:test/model"),
            "it should name the model actually configured: {described}"
        );
    }

    #[test]
    fn an_unconfigured_machine_says_so_rather_than_promising_a_model() {
        let dir = tempfile::tempdir().unwrap();
        let described = video_tool_description(&definitions_for(dir.path()));
        assert!(described.contains("No default video model"), "{described}");
    }
}

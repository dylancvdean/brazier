//! Bundled tools the runtime can execute on behalf of a model.
//!
//! Tools are intentionally conservative: no filesystem access, no shell, and
//! web retrieval is bounded (size, time, rate, and private-network guard).
//! JavaScript runs in an isolated QuickJS runtime with memory, stack, and time
//! limits.

use anyhow::Context;
use serde_json::{Value, json};

use crate::web;

const FETCH_MAX_OUTPUT_CHARS: usize = web::FETCH_MAX_OUTPUT_CHARS;

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
    /// Original display/save name when the media came from a named attachment
    /// or a rendered/generated file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
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
            "run_javascript" => Some(
                crate::js_sandbox::JsSandboxConfig::from_runtime_settings(&settings)
                    .describe_for_model(),
            ),
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
                "description": "Fetch a public http(s) URL and return text content. A PDF must be fetched here, never passed to doc_read: fetching a PDF stores it and returns a document id plus its page count for doc_read to page through — its contents are not included in this result. PDF links returned by MCP web tools use the same path. Private and local addresses are blocked. Long pages are truncated with a character count; pass start to fetch the next chunk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": { "type": "string", "description": "Absolute http:// or https:// URL." },
                        "start": { "type": "integer", "minimum": 0, "description": "Character offset to start from; use the count in the previous result to page through long pages. Default 0." },
                        "max_chars": { "type": "integer", "minimum": 500, "maximum": 50000, "description": "Maximum characters to return. Default 8000." }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for up-to-date information and return a ranked list of results with titles, URLs, and short snippets. Use it for facts that changed recently or are outside your knowledge. DuckDuckGo (no key) or Brave (API key) can be chosen in Manage → Web search; if DuckDuckGo rate-limits this machine, the user can add a Brave key there.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query, phrased like you would type into a search engine."
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Maximum results to return. Default 5."
                        },
                        "region": {
                            "type": "string",
                            "description": "Optional region/locale code such as `us-en`, `de-de`, or `wt-wt`. DuckDuckGo only."
                        },
                        "safesearch": {
                            "type": "string",
                            "enum": ["moderate", "strict", "off"],
                            "description": "Filtering level. Default moderate."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_javascript",
                "description": "Run JavaScript in a sandboxed QuickJS environment. No network, filesystem, or host APIs. Use for data transforms, date math, or small algorithms.",
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
                "name": "doc_read",
                "description": "Read a PDF, RTF, DOC, or DOCX the user attached. Pass the short document id from the attachment notice (about 12 hex characters). It does not accept URLs — to read a PDF from the web, call fetch_url first and use the document id it returns. For PDFs, choose a page range (default: first 3 pages) or set render_pages to true to receive page images — use that for scanned PDFs with no text layer, or when layout matters. For RTF/DOC/DOCX, choose a line range instead. Output is truncated when too long; narrow the range and call again.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document": {
                            "type": "string",
                            "description": "Short document id from the attachment notice (unique prefix of the blob sha256)."
                        },
                        "start_page": {
                            "type": "integer",
                            "description": "First PDF page to read, 1-based. Default 1.",
                            "minimum": 1
                        },
                        "end_page": {
                            "type": "integer",
                            "description": "Last PDF page to read, inclusive. Defaults to start_page + 2 (three pages). Max window is 25 pages for text, 4 for render_pages.",
                            "minimum": 1
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "First line for non-PDF documents, 1-based. Default 1.",
                            "minimum": 1
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "Last line for non-PDF documents, inclusive.",
                            "minimum": 1
                        },
                        "render_pages": {
                            "type": "boolean",
                            "description": "If true, render PDF pages as images instead of extracting text. Max 4 pages per call."
                        }
                    },
                    "required": ["document"]
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
    catalog_for_config(&crate::js_sandbox::JsSandboxConfig::default())
}

/// Catalog entries reflecting the live JavaScript sandbox configuration.
pub fn catalog_for(data_dir: &std::path::Path) -> Value {
    let settings = crate::runtime_settings::load(data_dir);
    catalog_for_config(&crate::js_sandbox::JsSandboxConfig::from_runtime_settings(
        &settings,
    ))
}

fn catalog_for_config(js_config: &crate::js_sandbox::JsSandboxConfig) -> Value {
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
                    "Bounded retrieval of public web pages (max {} KB, {}s timeout); MCP PDF links use the same guarded downloader. Local and private addresses are blocked.",
                    web::FETCH_MAX_BYTES / 1024,
                    web::FETCH_TIMEOUT.as_secs()
                ),
                "network": true,
                "source": "builtin"
            },
            {
                "name": "web_search",
                "title": "Web search",
                "description": "Ranked web search results (DuckDuckGo keyless, Brave with an API key). Rate-limited to stay under the engine's block threshold.",
                "network": true,
                "source": "builtin"
            },
            {
                "name": "run_javascript",
                "title": "JavaScript sandbox",
                "description": js_config.describe_for_catalog(),
                "network": false,
                "source": "builtin"
            },
            {
                "name": "doc_read",
                "title": "Read document",
                "description": "Read attached PDF, RTF, DOC, or DOCX by page or line range; render PDF pages as images when needed (Poppler for PDFs).",
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
            | "web_search"
            | "run_javascript"
            | "doc_read"
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
    execute_with_context(client, None, &[], &[], call_id, name, arguments).await
}

pub async fn execute_with_data_dir(
    client: &reqwest::Client,
    data_dir: Option<&std::path::Path>,
    call_id: &str,
    name: &str,
    arguments: &str,
) -> ToolInvocation {
    execute_with_context(client, data_dir, &[], &[], call_id, name, arguments).await
}

/// Run a built-in tool with the conversation's images and documents available.
pub async fn execute_with_context(
    client: &reqwest::Client,
    data_dir: Option<&std::path::Path>,
    images: &[crate::tool_registry::ConversationImage],
    documents: &[crate::tool_registry::ConversationDocument],
    call_id: &str,
    name: &str,
    arguments: &str,
) -> ToolInvocation {
    let parsed: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
    let result: anyhow::Result<ToolOutput> = match name {
        "doc_read" => match data_dir {
            Some(dir) => doc_read_tool(dir, &parsed, documents).await,
            None => Err(anyhow::anyhow!(
                "doc_read requires daemon data directory context"
            )),
        },
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
        "fetch_url" => match (data_dir, parsed.get("url").and_then(Value::as_str)) {
            (Some(dir), Some(url)) => fetch_url(client, dir, url, &parsed).await,
            (None, _) => Err(anyhow::anyhow!(
                "fetch_url requires daemon data directory context"
            )),
            (_, None) => Err(anyhow::anyhow!(
                "fetch_url requires a `url` string argument"
            )),
        },
        "web_search" => match data_dir {
            Some(dir) => web_search_tool(client, dir, &parsed).await,
            None => Err(anyhow::anyhow!(
                "web_search requires daemon data directory context"
            )),
        },
        other => simple_tool(client, data_dir, other, &parsed)
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
    _client: &reqwest::Client,
    data_dir: Option<&std::path::Path>,
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
        "run_javascript" => match parsed.get("code").and_then(Value::as_str) {
            Some(code) => {
                let code = code.to_owned();
                let config = match data_dir {
                    Some(dir) => {
                        let settings = crate::runtime_settings::load(dir);
                        crate::js_sandbox::JsSandboxConfig::from_runtime_settings(&settings)
                    }
                    None => crate::js_sandbox::JsSandboxConfig::default(),
                };
                match tokio::task::spawn_blocking(move || {
                    crate::js_sandbox::run_javascript_with_config(&code, &config)
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
pub(crate) struct ToolOutput {
    pub(crate) text: String,
    pub(crate) media: Vec<ToolMedia>,
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

/// Resolve a `doc_read` document argument against conversation attachments.
///
/// Accepts a full sha256, an optional `brazier_blob:` prefix, or the short
/// unique prefix shown in attachment notices.
fn resolve_conversation_document<'a>(
    requested: &str,
    documents: &'a [crate::tool_registry::ConversationDocument],
) -> anyhow::Result<&'a crate::tool_registry::ConversationDocument> {
    let sha256 = requested
        .strip_prefix("brazier_blob:")
        .unwrap_or(requested)
        .trim();
    anyhow::ensure!(
        !sha256.is_empty(),
        "doc_read requires a `document` string (the id from the attachment notice)"
    );
    if let Some(document) = documents
        .iter()
        .find(|doc| doc.sha256.eq_ignore_ascii_case(sha256))
    {
        return Ok(document);
    }
    let matches: Vec<_> = documents
        .iter()
        .filter(|doc| {
            doc.sha256.len() >= sha256.len()
                && doc.sha256[..sha256.len()].eq_ignore_ascii_case(sha256)
        })
        .collect();
    match matches.as_slice() {
        [document] => Ok(document),
        [] => anyhow::bail!(
            "that document is not in this conversation — use the document id from an attachment notice"
        ),
        _ => anyhow::bail!(
            "that document id is ambiguous — pass more characters from the attachment notice"
        ),
    }
}

async fn doc_read_tool(
    data_dir: &std::path::Path,
    args: &Value,
    documents: &[crate::tool_registry::ConversationDocument],
) -> anyhow::Result<ToolOutput> {
    let requested = args
        .get("document")
        .and_then(Value::as_str)
        .context("doc_read requires a `document` string (the id from the attachment notice)")?
        .trim();
    let document = resolve_conversation_document(requested, documents)?;
    let kind = crate::documents::kind_for_mime(&document.mime_type, &document.name)
        .context("that attachment is not a PDF, RTF, DOC, or DOCX document")?;
    let path = crate::blob_store::blob_path(data_dir, &document.sha256)
        .context("document blob is missing")?;
    anyhow::ensure!(path.is_file(), "that document is no longer stored locally");
    if kind == crate::documents::DocumentKind::Pdf {
        crate::documents::ensure_poppler_available()?;
    }

    let render = args
        .get("render_pages")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if render {
        anyhow::ensure!(
            kind == crate::documents::DocumentKind::Pdf,
            "render_pages only applies to PDFs"
        );
        let start = args
            .get("start_page")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .unwrap_or(1)
            .max(1);
        let end = args
            .get("end_page")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .unwrap_or(start + crate::documents::DEFAULT_PAGE_COUNT - 1)
            .max(start);
        let count = (end - start + 1).min(crate::documents::MAX_RENDER_PAGES);
        let rendered = crate::documents::render_pages(data_dir, &path, start, count).await?;
        let pages: Vec<String> = rendered
            .iter()
            .map(|page| format!("page {}", page.page))
            .collect();
        return Ok(ToolOutput {
            text: format!(
                "Rendered {} of {} ({}) as images. The pages are included below for a vision model.",
                document.name,
                kind.label(),
                pages.join(", ")
            ),
            media: rendered
                .into_iter()
                .map(|page| ToolMedia {
                    sha256: page.sha256,
                    mime_type: page.mime_type,
                    name: Some(crate::documents::rendered_page_name(
                        &document.name,
                        page.page,
                    )),
                })
                .collect(),
        });
    }

    let pages = if kind == crate::documents::DocumentKind::Pdf {
        let start = args
            .get("start_page")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .unwrap_or(1)
            .max(1);
        let end = args
            .get("end_page")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .unwrap_or(start + crate::documents::DEFAULT_PAGE_COUNT - 1)
            .max(start);
        anyhow::ensure!(
            end - start < crate::documents::MAX_TEXT_PAGES,
            "PDF text window is limited to {} pages; narrow start_page/end_page",
            crate::documents::MAX_TEXT_PAGES
        );
        Some((start, end))
    } else {
        None
    };
    let lines = if kind == crate::documents::DocumentKind::Pdf {
        None
    } else {
        match (
            args.get("start_line").and_then(Value::as_u64),
            args.get("end_line").and_then(Value::as_u64),
        ) {
            (None, None) => None,
            (start, end) => {
                let start = start.unwrap_or(1).max(1) as usize;
                let end = end
                    .map(|value| value as usize)
                    .unwrap_or(start.saturating_add(199))
                    .max(start);
                Some((start, end))
            }
        }
    };
    let extraction = crate::documents::extract_text(
        &path,
        kind,
        pages,
        lines,
        crate::documents::MAX_EXTRACTION_CHARS,
    )
    .await?;
    Ok(ToolOutput::from(extraction.describe()))
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
        .context("no default image generation model configured (set one in Generate settings)")?;
    let init_image = resolve_init_image(data_dir, args, images)?;
    let request = crate::sdcpp::GenerateImageRequest {
        prompt: prompt.to_owned(),
        model_id,
        negative_prompt: args
            .get("negative_prompt")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        // Left unset when the model says nothing, so the size and step count
        // this generation model was configured for are what it runs at.
        width: args.get("width").and_then(Value::as_u64).map(|v| v as u32),
        height: args.get("height").and_then(Value::as_u64).map(|v| v as u32),
        steps: args.get("steps").and_then(Value::as_u64).map(|v| v as u32),
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
    let profiles = crate::model_settings::load(data_dir);
    let result = crate::sdcpp::generate_image(
        data_dir,
        settings.sdcpp_binary.as_deref(),
        &request,
        profiles.diffusion(&request.model_id),
    )
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
            "Image generation succeeded. The image has already been displayed to the user \
             (brazier_blob:{}, {} bytes). Do not generate another image unless the user asks \
             for another version or requests a change.",
            blob.sha256, blob.size_bytes
        ),
        media: vec![ToolMedia {
            sha256: blob.sha256.clone(),
            mime_type: "image/png".to_owned(),
            name: Some("generated.png".to_owned()),
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
        .context("no default video generation model configured (set one in Generate settings)")?;
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
        width: args.get("width").and_then(Value::as_u64).map(|v| v as u32),
        height: args.get("height").and_then(Value::as_u64).map(|v| v as u32),
        steps: args.get("steps").and_then(Value::as_u64).map(|v| v as u32),
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
            .map(|v| v as u32),
        fps: args.get("fps").and_then(Value::as_u64).map(|v| v as u32),
        end_image: None,
        end_image_blob: None,
        ref_images: Vec::new(),
        ref_image_blobs: Vec::new(),
        ref_videos: Vec::new(),
        ref_video_blobs: Vec::new(),
        ref_video_audios: Vec::new(),
        ref_audios: Vec::new(),
        ref_audio_blobs: Vec::new(),
    };
    let profiles = crate::model_settings::load(data_dir);
    let result = crate::sdcpp::generate_video(
        data_dir,
        settings.sdcpp_binary.as_deref(),
        &request,
        profiles.diffusion(&request.model_id),
    )
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
    let output_name = result
        .output_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("generated.mp4")
        .to_owned();
    let blob = crate::blob_store::store_bytes(data_dir, &bytes, mime, Some(&output_name)).await?;
    let _ = tokio::fs::remove_file(&result.output_path).await;
    Ok(ToolOutput {
        text: format!(
            "Video generation succeeded. The video has already been displayed to the user \
             (brazier_blob:{}, {} bytes). Do not generate another video unless the user asks \
             for another version or requests a change.",
            blob.sha256, blob.size_bytes
        ),
        media: vec![ToolMedia {
            sha256: blob.sha256.clone(),
            mime_type: mime.to_owned(),
            name: Some(output_name),
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

// --- Web search and bounded retrieval ----------------------------------------

/// Run the shared `web_search` against the daemon's configured backend.
async fn web_search_tool(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    parsed: &Value,
) -> anyhow::Result<ToolOutput> {
    let query = parsed
        .get("query")
        .and_then(Value::as_str)
        .context("web_search requires a `query` string argument")?
        .trim();
    anyhow::ensure!(!query.is_empty(), "`query` must not be empty");
    let max_results = parsed
        .get("max_results")
        .and_then(Value::as_u64)
        .unwrap_or(5) as usize;
    let region = parsed.get("region").and_then(Value::as_str);
    let safesearch = parsed.get("safesearch").and_then(Value::as_str);
    let settings = crate::runtime_settings::load(data_dir);
    let results = web::search(client, query, max_results, region, safesearch, &settings).await?;
    Ok(ToolOutput::from(web::format_results(&results)))
}

async fn ingest_pdf(
    data_dir: &std::path::Path,
    download: web::DownloadedUrl,
    args: &Value,
) -> anyhow::Result<ToolOutput> {
    anyhow::ensure!(web::downloaded_is_pdf(&download), "response is not a PDF");
    let name = download
        .final_url
        .path_segments()
        .and_then(|mut parts| parts.next_back())
        .filter(|part| !part.is_empty())
        .unwrap_or("download.pdf");
    ingest_pdf_bytes(data_dir, &download.bytes, name, args).await
}

pub(crate) async fn ingest_pdf_bytes(
    data_dir: &std::path::Path,
    bytes: &[u8],
    name: &str,
    _args: &Value,
) -> anyhow::Result<ToolOutput> {
    anyhow::ensure!(bytes.starts_with(b"%PDF-"), "resource is not a PDF");
    let blob =
        crate::blob_store::store_bytes(data_dir, bytes, "application/pdf", Some(name)).await?;
    // Deliberately do not read the PDF here. The point of fetching a PDF is to
    // store it and hand the model a document id plus its page count, so it can
    // choose a page range with doc_read instead of dumping the first pages into
    // context. Reading happens only when the model asks for it.
    let path = crate::blob_store::blob_path(data_dir, &blob.sha256)?;
    let pages = if crate::documents::missing_poppler_tools().is_empty() {
        crate::documents::page_count(&path).await.unwrap_or(None)
    } else {
        None
    };
    let document_id = crate::documents::short_document_id(&blob.sha256);
    let mut text = format!("Fetched and attached PDF {name}.");
    if let Some(count) = pages {
        text.push_str(&format!(" It is {count} pages long."));
    }
    text.push_str(&format!(
        " Its contents are not included here. Use the doc_read tool with document \
         \"{document_id}\" to read it, choosing a page range{}. If it is a scan with no text \
         layer, set render_pages to receive page images.",
        if pages.is_some() {
            " from the page count"
        } else {
            " (page count could not be determined)"
        }
    ));
    Ok(ToolOutput {
        text,
        media: vec![ToolMedia {
            sha256: blob.sha256,
            mime_type: "application/pdf".into(),
            name: Some(name.to_owned()),
        }],
    })
}

/// Fetch a URL surfaced by another tool, attaching it only when it is a PDF.
/// The caller can keep the original tool's text when this returns `None`.
pub(crate) async fn fetch_pdf_candidate(
    data_dir: &std::path::Path,
    url: &str,
) -> anyhow::Result<Option<(String, Vec<ToolMedia>)>> {
    let download = web::download_url(url).await?;
    if !web::downloaded_is_pdf(&download) {
        return Ok(None);
    }
    let output = ingest_pdf(data_dir, download, &json!({})).await?;
    Ok(Some((output.text, output.media)))
}

async fn fetch_url(
    _client: &reqwest::Client,
    data_dir: &std::path::Path,
    url: &str,
    args: &Value,
) -> anyhow::Result<ToolOutput> {
    let download = web::download_url(url).await?;
    if web::downloaded_is_pdf(&download) {
        return ingest_pdf(data_dir, download, args).await;
    }
    let start = args.get("start").and_then(Value::as_u64).unwrap_or(0) as usize;
    let max_chars = args
        .get("max_chars")
        .and_then(Value::as_u64)
        .unwrap_or(FETCH_MAX_OUTPUT_CHARS as u64)
        .clamp(500, 50_000) as usize;
    Ok(ToolOutput::from(web::fetch_content_text(
        &download, start, max_chars,
    )?))
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
        let parsed: Value = serde_json::from_str(&result.output).expect("json envelope");
        assert_eq!(parsed["return"], 42);
        assert_eq!(parsed["logs"], json!([]));
    }

    #[tokio::test]
    async fn execute_javascript_captures_console() {
        let client = reqwest::Client::new();
        let result = execute(
            &client,
            "call_1",
            "run_javascript",
            r#"{"code": "console.log('hi'); return {ok: true};"}"#,
        )
        .await;
        assert!(!result.is_error, "{}", result.output);
        let parsed: Value = serde_json::from_str(&result.output).expect("json envelope");
        assert_eq!(parsed["return"]["ok"], true);
        assert_eq!(parsed["logs"][0], "hi");
    }

    #[test]
    fn javascript_tool_description_reflects_sandbox_config() {
        let dir = tempfile::tempdir().unwrap();
        let described = definitions_for(dir.path())
            .as_array()
            .unwrap()
            .iter()
            .find(|item| {
                item.pointer("/function/name").and_then(Value::as_str) == Some("run_javascript")
            })
            .and_then(|item| item.pointer("/function/description"))
            .and_then(Value::as_str)
            .unwrap()
            .to_owned();
        assert!(
            described.contains("Not Node"),
            "should be honest about the environment: {described}"
        );
        assert!(
            described.contains("console.log"),
            "default profile should mention console: {described}"
        );
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
        let settings = crate::runtime_settings::RuntimeSettings {
            sdcpp_binary: Some(binary.display().to_string()),
            default_video_gen_model: Some("sdcpp-video:test/model".into()),
            ..crate::runtime_settings::RuntimeSettings::default()
        };
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
            &[],
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
            &[],
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

    #[test]
    fn doc_read_resolves_short_document_ids() {
        let documents = vec![
            crate::tool_registry::ConversationDocument {
                sha256: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789".into(),
                mime_type: "application/pdf".into(),
                name: "a.pdf".into(),
            },
            crate::tool_registry::ConversationDocument {
                sha256: "ffffff0123456789abcdef0123456789abcdef0123456789abcdef0123456789".into(),
                mime_type: "application/pdf".into(),
                name: "b.pdf".into(),
            },
        ];
        assert_eq!(
            resolve_conversation_document("abcdef012345", &documents)
                .unwrap()
                .name,
            "a.pdf"
        );
        assert_eq!(
            resolve_conversation_document(
                "brazier_blob:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
                &documents
            )
            .unwrap()
            .name,
            "a.pdf"
        );
        assert!(resolve_conversation_document("deadbeef", &documents).is_err());
    }
}

//! Engine adapters: llama.cpp runtime over on-disk GGUF models.

use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::{
    llama::{self, LlamaServer},
    models_store,
    progress::ProgressCallback,
    runtime_settings::{self, RuntimeSettings},
    tools,
    types::{ChatCompletionRequest, ModelDescriptor, OpenAiMessage},
};

#[derive(Debug, Clone)]
pub struct Generation {
    pub text: String,
    pub reasoning: Option<String>,
    pub tool_invocations: Vec<tools::ToolInvocation>,
}

/// One item in a streamed generation.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Assistant content delta.
    Content(String),
    /// A bundled tool was executed server-side.
    Tool(tools::ToolInvocation),
    /// Generation finished.
    End,
}

/// Maximum model round-trips when the model keeps requesting tools.
const MAX_TOOL_ROUNDS: usize = 4;

#[async_trait]
pub trait Engine: Send + Sync {
    fn id(&self) -> &'static str;
    async fn models(&self) -> anyhow::Result<Vec<ModelDescriptor>>;
    async fn generate(&self, request: &ChatCompletionRequest) -> anyhow::Result<Generation>;
}

struct LlamaState {
    binary: Option<PathBuf>,
    server: Option<LlamaServer>,
}

/// Runtime that lists on-disk GGUF models and serves them through llama-server.
pub struct Runtime {
    data_dir: PathBuf,
    http: reqwest::Client,
    llama: Mutex<LlamaState>,
    settings: Mutex<RuntimeSettings>,
}

impl Runtime {
    pub fn new(data_dir: PathBuf, http: reqwest::Client) -> Arc<Self> {
        let settings = runtime_settings::load(&data_dir);
        let path_env = std::env::var_os("PATH");
        let effective_target = if settings.target == crate::runtime_settings::RuntimeTarget::Auto {
            crate::hardware::detect().recommended_target
        } else {
            settings.target
        };
        let pinned = settings
            .binary_override
            .as_ref()
            .map(PathBuf::from)
            .filter(|path| path.is_file());
        let managed = llama::managed_binary_path_for_target(&data_dir, effective_target);
        let discovered = pinned
            .or_else(|| managed.is_file().then_some(managed))
            .or_else(|| {
                llama::discovery_candidates(
                    &data_dir,
                    path_env.as_deref().and_then(|value| value.to_str()),
                )
                .into_iter()
                .skip(1)
                .find(|path| path.is_file())
            });
        Arc::new(Self {
            data_dir,
            http,
            llama: Mutex::new(LlamaState {
                binary: discovered,
                server: None,
            }),
            settings: Mutex::new(settings),
        })
    }

    pub fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    pub async fn engine_status(&self) -> serde_json::Value {
        let settings = self.settings.lock().await.clone();
        let guard = self.llama.lock().await;
        let binary = guard.binary.as_ref().map(|path| path.display().to_string());
        let running = guard.server.as_ref().map(|server| {
            serde_json::json!({
                "base_url": server.base_url,
                "model_path": server.model_path.display().to_string(),
                "projector_path": server.projector_path.as_ref().map(|path| path.display().to_string()),
            })
        });
        serde_json::json!({
            "id": self.id(),
            "llama_binary": binary,
            "llama_server": running,
            "llama_probe": self.llama_diagnostics().await,
            "managed_binary_path": llama::managed_binary_path(&self.data_dir).display().to_string(),
            "platform_asset_tag": llama::platform_asset_tag(),
            "settings": settings,
            "hardware": crate::hardware::detect(),
        })
    }

    /// Live capability probe against a running llama-server, if any.
    pub async fn llama_diagnostics(&self) -> Option<serde_json::Value> {
        let guard = self.llama.lock().await;
        let server = guard.server.as_ref()?;
        let base_url = server.base_url.clone();
        drop(guard);
        Some(llama::probe_server(&self.http, &base_url).await)
    }

    pub async fn settings(&self) -> RuntimeSettings {
        self.settings.lock().await.clone()
    }

    pub async fn update_settings(
        &self,
        settings: RuntimeSettings,
    ) -> anyhow::Result<RuntimeSettings> {
        settings.validate()?;
        runtime_settings::save(&self.data_dir, &settings).await?;
        let mut current = self.settings.lock().await;
        let target_changed = current.target != settings.target;
        if *current != settings {
            let mut llama = self.llama.lock().await;
            if let Some(mut server) = llama.server.take() {
                let _ = server.stop().await;
            }
            // A pinned binary survives target changes; otherwise re-resolve.
            if target_changed && settings.binary_override.is_none() {
                llama.binary = None;
            }
        }
        *current = settings.clone();
        Ok(settings)
    }

    /// Currently selected llama-server binary, if any.
    pub async fn active_binary(&self) -> Option<PathBuf> {
        self.llama.lock().await.binary.clone()
    }

    /// Pin a specific llama-server binary and persist the choice. Stops any
    /// running server so the next generation uses the new binary.
    pub async fn activate_binary(&self, path: PathBuf) -> anyhow::Result<PathBuf> {
        anyhow::ensure!(
            path.is_file(),
            "runtime binary not found: {}",
            path.display()
        );
        let runnable = {
            let candidate = path.clone();
            tokio::task::spawn_blocking(move || llama::binary_appears_runnable(&candidate))
                .await
                .unwrap_or(false)
        };
        anyhow::ensure!(
            runnable,
            "{} failed a smoke test (missing shared libraries or incompatible build)",
            path.display()
        );
        let mut settings = self.settings.lock().await;
        settings.binary_override = Some(path.display().to_string());
        runtime_settings::save(&self.data_dir, &settings).await?;
        drop(settings);
        let mut guard = self.llama.lock().await;
        if let Some(mut server) = guard.server.take() {
            let _ = server.stop().await;
        }
        guard.binary = Some(path.clone());
        Ok(path)
    }

    /// Forget a binary that was removed from disk. Clears the pin if it pointed
    /// at the deleted path and stops any server running from it.
    pub async fn release_binary(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let mut guard = self.llama.lock().await;
        let served_from_deleted = guard
            .server
            .as_ref()
            .is_some_and(|server| server.binary == path);
        if served_from_deleted && let Some(mut server) = guard.server.take() {
            let _ = server.stop().await;
        }
        if guard.binary.as_deref() == Some(path) {
            guard.binary = None;
        }
        drop(guard);
        let mut settings = self.settings.lock().await;
        if settings.binary_override.as_deref() == Some(&path.display().to_string()) {
            settings.binary_override = None;
            runtime_settings::save(&self.data_dir, &settings).await?;
        }
        Ok(())
    }

    /// Stop the server if it is serving the given model file (used before the
    /// model is deleted from the library).
    pub async fn release_model(&self, model_path: &std::path::Path) {
        let mut guard = self.llama.lock().await;
        if guard
            .server
            .as_ref()
            .is_some_and(|server| server.model_path == model_path)
            && let Some(mut server) = guard.server.take()
        {
            let _ = server.stop().await;
        }
    }

    /// Discover an existing binary or download a managed release.
    pub async fn ensure_llama_binary(&self) -> anyhow::Result<PathBuf> {
        self.ensure_llama_binary_with_progress(Box::new(|_| {}))
            .await
    }

    pub async fn ensure_llama_binary_with_progress(
        &self,
        progress: ProgressCallback,
    ) -> anyhow::Result<PathBuf> {
        let target = self.settings.lock().await.target;
        {
            let guard = self.llama.lock().await;
            if let Some(path) = &guard.binary
                && path.is_file()
                && llama::binary_appears_runnable(path)
            {
                return Ok(path.clone());
            }
        }
        let path = llama::ensure_binary_with_progress(&self.http, &self.data_dir, target, progress)
            .await?;
        let mut guard = self.llama.lock().await;
        guard.binary = Some(path.clone());
        Ok(path)
    }

    async fn ensure_server_for_model(&self, model_path: &std::path::Path) -> anyhow::Result<()> {
        let settings = self.settings.lock().await.clone();
        let binary = {
            let guard = self.llama.lock().await;
            if let Some(path) = &guard.binary
                && path.is_file()
            {
                path.clone()
            } else {
                drop(guard);
                self.ensure_llama_binary().await?
            }
        };

        let mut guard = self.llama.lock().await;
        if let Some(server) = guard.server.as_mut() {
            if server.model_path == model_path
                && server.projector_path == models_store::projector_for_model(model_path)
                && server.is_running()
            {
                return Ok(());
            }
            let _ = server.stop().await;
            guard.server = None;
        }

        let server = LlamaServer::start(&binary, model_path, &settings).await?;
        guard.server = Some(server);
        guard.binary = Some(binary);
        Ok(())
    }

    fn resolve_model_path(&self, model: &str) -> anyhow::Result<std::path::PathBuf> {
        if model.is_empty() {
            anyhow::bail!("a model id is required; download a GGUF and select it first");
        }
        if !model.starts_with("gguf:") {
            anyhow::bail!(
                "unknown model `{model}`; download a GGUF model (ids look like `gguf:…`)"
            );
        }
        let model_path = models_store::path_for_model_id(&self.data_dir, model)?;
        anyhow::ensure!(model_path.is_file(), "model file not found for {model}");
        Ok(model_path)
    }

    /// Base URL of the running llama-server for the requested model.
    async fn prepare_generation(
        &self,
        request: &ChatCompletionRequest,
    ) -> anyhow::Result<(String, RuntimeSettings, ChatCompletionRequest)> {
        let model_path = self.resolve_model_path(&request.model)?;
        if let Err(error) = self.ensure_server_for_model(&model_path).await {
            // Engine crash / startup failure must not poison the daemon.
            let mut guard = self.llama.lock().await;
            guard.server = None;
            return Err(error);
        }
        let settings = self.settings.lock().await.clone();
        let guard = self.llama.lock().await;
        let Some(server) = guard.server.as_ref() else {
            anyhow::bail!("llama-server is not running");
        };
        let base_url = server.base_url.clone();
        let mut request = request.clone();
        if request.builtin_tools.unwrap_or(false) {
            request.tools = Some(tools::definitions());
        }
        Ok((base_url, settings, request))
    }

    /// Drop the llama-server handle if the child process died.
    async fn reap_dead_server(&self) {
        let mut guard = self.llama.lock().await;
        if let Some(server) = guard.server.as_mut()
            && !server.is_running()
        {
            guard.server = None;
        }
    }

    /// Stream content deltas from llama-server (true token streaming, not fake
    /// chunking). When bundled tools are enabled, tool calls are executed
    /// server-side and generation continues in additional rounds.
    pub async fn generate_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> anyhow::Result<tokio::sync::mpsc::Receiver<anyhow::Result<StreamEvent>>> {
        let (base_url, settings, request) = self.prepare_generation(request).await?;
        let builtin = request.builtin_tools.unwrap_or(false);
        let http = self.http.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            if let Err(error) =
                stream_tool_rounds(&http, &base_url, request, settings, builtin, &tx).await
            {
                let _ = tx.send(Err(error)).await;
            }
        });
        Ok(rx)
    }

    /// Stop any child llama-server (called on daemon shutdown).
    pub async fn shutdown(&self) {
        let mut guard = self.llama.lock().await;
        if let Some(mut server) = guard.server.take() {
            let _ = server.stop().await;
        }
    }
}

#[async_trait]
impl Engine for Runtime {
    fn id(&self) -> &'static str {
        "brazier"
    }

    async fn models(&self) -> anyhow::Result<Vec<ModelDescriptor>> {
        Ok(models_store::list_gguf_models(&self.data_dir)?)
    }

    async fn generate(&self, request: &ChatCompletionRequest) -> anyhow::Result<Generation> {
        let (base_url, settings, mut request) = self.prepare_generation(request).await?;
        let builtin = request.builtin_tools.unwrap_or(false);
        let mut invocations = Vec::new();
        for round in 0..MAX_TOOL_ROUNDS {
            let last_round = round + 1 == MAX_TOOL_ROUNDS;
            let mut body = llama::translate_chat_request(&request, &settings, "local", false);
            if last_round && let Some(object) = body.as_object_mut() {
                object.remove("tools");
            }
            let response = match llama::chat_once(&self.http, &base_url, &body).await {
                Ok(response) => response,
                Err(error) => {
                    self.reap_dead_server().await;
                    return Err(error);
                }
            };
            let calls = llama::extract_tool_calls(&response);
            if !builtin || calls.is_empty() {
                return Ok(Generation {
                    text: llama::extract_assistant_text(&response).unwrap_or_default(),
                    reasoning: None,
                    tool_invocations: invocations,
                });
            }
            let round_text = llama::extract_assistant_text(&response).unwrap_or_default();
            append_tool_round(
                &mut request.messages,
                round_text,
                &calls,
                &self.http,
                &mut invocations,
                None,
            )
            .await;
        }
        anyhow::bail!("generation exceeded the bundled tool round limit");
    }
}

/// Execute one round of tool calls: append the assistant tool-call message and
/// one `tool` message per executed call. Streams invocations to `events` when
/// provided.
async fn append_tool_round(
    messages: &mut Vec<OpenAiMessage>,
    round_text: String,
    calls: &[llama::AccumulatedToolCall],
    http: &reqwest::Client,
    invocations: &mut Vec<tools::ToolInvocation>,
    events: Option<&tokio::sync::mpsc::Sender<anyhow::Result<StreamEvent>>>,
) -> bool {
    messages.push(OpenAiMessage {
        role: "assistant".to_owned(),
        content: serde_json::Value::String(round_text),
        tool_calls: Some(llama::tool_calls_to_json(calls)),
        tool_call_id: None,
    });
    for call in calls {
        let invocation = tools::execute(http, &call.id, &call.name, &call.arguments).await;
        messages.push(OpenAiMessage {
            role: "tool".to_owned(),
            content: serde_json::Value::String(invocation.output.clone()),
            tool_calls: None,
            tool_call_id: Some(invocation.call_id.clone()),
        });
        invocations.push(invocation.clone());
        if let Some(tx) = events
            && tx.send(Ok(StreamEvent::Tool(invocation))).await.is_err()
        {
            return false;
        }
    }
    true
}

/// Streaming generation loop with server-side bundled tool execution.
async fn stream_tool_rounds(
    http: &reqwest::Client,
    base_url: &str,
    mut request: ChatCompletionRequest,
    settings: RuntimeSettings,
    builtin: bool,
    tx: &tokio::sync::mpsc::Sender<anyhow::Result<StreamEvent>>,
) -> anyhow::Result<()> {
    for round in 0..MAX_TOOL_ROUNDS {
        let last_round = round + 1 == MAX_TOOL_ROUNDS;
        let mut body = llama::translate_chat_request(&request, &settings, "local", true);
        if last_round && let Some(object) = body.as_object_mut() {
            object.remove("tools");
        }
        let mut chunks = llama::open_chat_stream(http, base_url, &body).await?;
        let mut accumulator = llama::ToolCallAccumulator::default();
        let mut round_text = String::new();
        while let Some(item) = chunks.recv().await {
            let chunk = item?;
            if let Some(content) = chunk.content {
                round_text.push_str(&content);
                if tx.send(Ok(StreamEvent::Content(content))).await.is_err() {
                    return Ok(());
                }
            }
            accumulator.absorb(&chunk.tool_calls);
        }
        let calls = accumulator.into_calls();
        if !builtin || calls.is_empty() {
            let _ = tx.send(Ok(StreamEvent::End)).await;
            return Ok(());
        }
        let mut invocations = Vec::new();
        if !append_tool_round(
            &mut request.messages,
            round_text,
            &calls,
            http,
            &mut invocations,
            Some(tx),
        )
        .await
        {
            return Ok(());
        }
    }
    let _ = tx.send(Ok(StreamEvent::End)).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[tokio::test]
    async fn runtime_lists_only_disk_gguf() {
        let dir = tempdir().unwrap();
        let file = models_store::download_destination(dir.path(), "acme/demo", "m.gguf").unwrap();
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"gguf").unwrap();
        let runtime = Runtime::new(dir.path().to_path_buf(), reqwest::Client::new());
        let models = runtime.models().await.unwrap();
        assert!(models.iter().all(|model| model.id.starts_with("gguf:")));
        assert!(
            models
                .iter()
                .any(|model| model.id == "gguf:acme/demo/m.gguf" && model.engine == "llama.cpp")
        );
        assert!(!models.iter().any(|model| model.id == "brazier/mock"));
    }

    #[tokio::test]
    async fn empty_model_list_without_downloads() {
        let dir = tempdir().unwrap();
        let runtime = Runtime::new(dir.path().to_path_buf(), reqwest::Client::new());
        let models = runtime.models().await.unwrap();
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn missing_gguf_returns_error_without_panic() {
        let dir = tempdir().unwrap();
        let runtime = Runtime::new(dir.path().to_path_buf(), reqwest::Client::new());
        let error = runtime
            .generate(&ChatCompletionRequest {
                model: "gguf:missing/repo/file.gguf".into(),
                messages: vec![crate::types::OpenAiMessage {
                    role: "user".into(),
                    content: json!("hi"),
                    tool_calls: None,
                    tool_call_id: None,
                }],
                stream: false,
                tools: None,
                temperature: None,
                top_p: None,
                max_tokens: None,
                seed: None,
                enable_reasoning: None,
                builtin_tools: None,
            })
            .await
            .unwrap_err();
        assert!(error.to_string().contains("not found") || error.to_string().contains("missing"));
    }

    #[tokio::test]
    async fn rejects_non_gguf_model_ids() {
        let dir = tempdir().unwrap();
        let runtime = Runtime::new(dir.path().to_path_buf(), reqwest::Client::new());
        let error = runtime
            .generate(&ChatCompletionRequest {
                model: "brazier/mock".into(),
                messages: vec![crate::types::OpenAiMessage {
                    role: "user".into(),
                    content: json!("hi"),
                    tool_calls: None,
                    tool_call_id: None,
                }],
                stream: false,
                tools: None,
                temperature: None,
                top_p: None,
                max_tokens: None,
                seed: None,
                enable_reasoning: None,
                builtin_tools: None,
            })
            .await
            .unwrap_err();
        assert!(error.to_string().contains("unknown model") || error.to_string().contains("gguf"));
    }
}

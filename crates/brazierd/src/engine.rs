//! Engine adapters: llama.cpp runtime over on-disk GGUF models.

use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::{
    fork_hints::{self, ModelLoadError},
    llama::{self, LlamaServer},
    media::{self, MediaContext},
    mlx::{self, MlxKind, MlxServer},
    model_bindings, models_store,
    progress::{ProgressCallback, ProgressEvent},
    runtime_settings::{self, RuntimeSettings},
    runtimes, sdcpp, streaming_asr,
    tool_registry::{self, ToolContext},
    tools,
    types::{ChatCompletionRequest, ModelCapabilities, ModelDescriptor, OpenAiMessage},
    voice, whisper,
};

#[derive(Debug, Clone)]
pub struct Generation {
    pub text: String,
    pub reasoning: Option<String>,
    pub tool_invocations: Vec<tools::ToolInvocation>,
    /// Tool calls the client should execute (no server-side handler).
    pub client_tool_calls: Vec<llama::AccumulatedToolCall>,
    /// Assistant/tool messages produced during server-side tool rounds.
    pub transcript: Vec<OpenAiMessage>,
}

/// One item in a streamed generation.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Model/server preparation progress.
    Load { phase: String, message: String },
    /// Assistant content delta.
    Content(String),
    /// A bundled or MCP tool was executed server-side.
    Tool(tools::ToolInvocation),
    /// Tool calls returned to the client for execution.
    ClientToolCalls(Vec<llama::AccumulatedToolCall>),
    /// Partial tool-call delta while the model is streaming.
    ToolCallDelta(llama::ToolCallFragment),
    /// A message added to context during a tool round (for faithful persistence).
    TranscriptMessage(OpenAiMessage),
    /// Generation finished.
    End,
}

type LoadNotifier = Option<tokio::sync::mpsc::Sender<anyhow::Result<StreamEvent>>>;

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

struct MlxState {
    lm_python: Option<PathBuf>,
    vlm_python: Option<PathBuf>,
    server: Option<MlxServer>,
}

struct WhisperState {
    binary: Option<PathBuf>,
}

struct StreamingAsrState {
    python: Option<PathBuf>,
}

struct SdCppState {
    binary: Option<PathBuf>,
}

pub struct VoiceState {
    pub python: Option<PathBuf>,
    pub server: Option<voice::VoiceServer>,
    pub sessions: voice::SessionManager,
}

enum ActiveBackend {
    Llama(String),
    Mlx(String),
}

/// Fraction of total RAM treated as usable for model residency in `auto`
/// memory arbitration. The remainder absorbs the OS, other apps, and runtime
/// overhead beyond raw weight size.
const USABLE_MEMORY_FRACTION: f64 = 0.85;

/// Outcome of pre-generation memory arbitration, returned so the caller can
/// restore the evicted chat model after the generation finishes.
pub struct GenerationMemoryPlan {
    /// Whether a chat model was evicted to make room.
    pub ejected: bool,
    /// Human-readable explanation of the decision (for logs/telemetry).
    pub reason: String,
    reload_llama_path: Option<PathBuf>,
}

impl GenerationMemoryPlan {
    fn noop(reason: impl Into<String>) -> Self {
        Self {
            ejected: false,
            reason: reason.into(),
            reload_llama_path: None,
        }
    }
}

fn file_size(path: &std::path::Path) -> u64 {
    std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0)
}

/// Runtime that lists on-disk GGUF models and serves them through llama-server.
pub struct Runtime {
    data_dir: PathBuf,
    http: reqwest::Client,
    llama: Mutex<LlamaState>,
    mlx: Mutex<MlxState>,
    whisper: Mutex<WhisperState>,
    streaming_asr: Mutex<StreamingAsrState>,
    sdcpp: Mutex<SdCppState>,
    voice: Mutex<VoiceState>,
    settings: Mutex<RuntimeSettings>,
    models_cache: Mutex<Option<Vec<ModelDescriptor>>>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EngineStatusOptions {
    pub probe: bool,
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
        let whisper_binary = whisper::resolve_binary(&data_dir, settings.whisper_binary.as_deref());
        let streaming_asr_python =
            streaming_asr::resolve_python(&data_dir, settings.streaming_asr_python.as_deref());
        let sdcpp_binary = sdcpp::resolve_binary(&data_dir, settings.sdcpp_binary.as_deref());
        let voice_python = voice::resolve_python(&data_dir, settings.voice_python.as_deref());
        Arc::new(Self {
            data_dir,
            http,
            llama: Mutex::new(LlamaState {
                binary: discovered,
                server: None,
            }),
            mlx: Mutex::new(MlxState {
                lm_python: settings.mlx_lm_python.as_ref().map(PathBuf::from),
                vlm_python: settings.mlx_vlm_python.as_ref().map(PathBuf::from),
                server: None,
            }),
            whisper: Mutex::new(WhisperState {
                binary: whisper_binary,
            }),
            streaming_asr: Mutex::new(StreamingAsrState {
                python: streaming_asr_python,
            }),
            sdcpp: Mutex::new(SdCppState {
                binary: sdcpp_binary,
            }),
            voice: Mutex::new(VoiceState {
                python: voice_python,
                server: None,
                sessions: voice::SessionManager::new(),
            }),
            settings: Mutex::new(settings),
            models_cache: Mutex::new(None),
        })
    }

    pub async fn invalidate_models_cache(&self) {
        *self.models_cache.lock().await = None;
    }

    pub async fn cached_models(&self) -> anyhow::Result<Vec<ModelDescriptor>> {
        if let Some(models) = self.models_cache.lock().await.clone() {
            return Ok(models);
        }
        let settings = self.settings.lock().await.clone();
        let extra_paths: Vec<PathBuf> = settings
            .extra_model_library_paths
            .iter()
            .map(PathBuf::from)
            .collect();
        let data_dir = self.data_dir.clone();
        let models = tokio::task::spawn_blocking(move || {
            models_store::list_local_models(&data_dir, &extra_paths)
        })
        .await??;
        *self.models_cache.lock().await = Some(models.clone());
        Ok(models)
    }

    async fn extra_library_paths(&self) -> Vec<PathBuf> {
        self.settings
            .lock()
            .await
            .extra_model_library_paths
            .iter()
            .map(PathBuf::from)
            .collect()
    }

    pub fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    pub async fn engine_status(&self, options: EngineStatusOptions) -> serde_json::Value {
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
        drop(guard);
        let llama_probe = if options.probe {
            self.llama_diagnostics().await
        } else {
            None
        };
        let mlx_probe = if options.probe {
            self.mlx_diagnostics().await
        } else {
            None
        };
        serde_json::json!({
            "id": self.id(),
            "llama_binary": binary,
            "llama_server": running,
            "llama_probe": llama_probe,
            "mlx_lm_python": self.mlx.lock().await.lm_python.as_ref().map(|path| path.display().to_string()),
            "mlx_vlm_python": self.mlx.lock().await.vlm_python.as_ref().map(|path| path.display().to_string()),
            "mlx_server": self.mlx_server_summary().await,
            "mlx_probe": mlx_probe,
            "managed_binary_path": llama::managed_binary_path(&self.data_dir).display().to_string(),
            "platform_asset_tag": llama::platform_asset_tag(),
            "settings": settings,
            "hardware": crate::hardware::detect(),
        })
    }

    /// Lightweight server summary without an HTTP probe (for frequent health checks).
    pub async fn llama_server_summary(&self) -> Option<serde_json::Value> {
        let guard = self.llama.lock().await;
        guard.server.as_ref().map(|server| {
            serde_json::json!({
                "base_url": server.base_url,
                "model_path": server.model_path.display().to_string(),
                "projector_path": server.projector_path.as_ref().map(|path| path.display().to_string()),
            })
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

    pub async fn mlx_server_summary(&self) -> Option<serde_json::Value> {
        let guard = self.mlx.lock().await;
        guard.server.as_ref().map(|server| {
            serde_json::json!({
                "base_url": server.base_url,
                "model_ref": server.model_ref,
                "engine": server.kind.engine_id(),
                "python": server.python.display().to_string(),
            })
        })
    }

    pub async fn mlx_diagnostics(&self) -> Option<serde_json::Value> {
        let guard = self.mlx.lock().await;
        let server = guard.server.as_ref()?;
        let base_url = server.base_url.clone();
        drop(guard);
        Some(mlx::probe_server(&self.http, &base_url).await)
    }

    pub async fn active_runtimes(&self) -> runtimes::ActiveRuntimes {
        let llama = self.llama.lock().await.binary.clone();
        let mlx = self.mlx.lock().await;
        let whisper = self.whisper.lock().await.binary.clone();
        let streaming_asr = self.streaming_asr.lock().await.python.clone();
        let sdcpp = self.sdcpp.lock().await.binary.clone();
        let voice = self.voice.lock().await.python.clone();
        runtimes::ActiveRuntimes {
            llama,
            mlx_lm: mlx.lm_python.clone(),
            mlx_vlm: mlx.vlm_python.clone(),
            whisper,
            streaming_asr,
            sdcpp,
            voice,
        }
    }

    pub async fn settings(&self) -> RuntimeSettings {
        self.settings.lock().await.clone()
    }

    /// Lock the voice runtime state (PersonaPlex sessions).
    pub async fn voice_state(&self) -> tokio::sync::MutexGuard<'_, VoiceState> {
        self.voice.lock().await
    }

    pub async fn update_settings(
        &self,
        mut settings: RuntimeSettings,
    ) -> anyhow::Result<RuntimeSettings> {
        settings.extra_model_library_paths =
            crate::model_library::normalize_library_paths(&settings.extra_model_library_paths)?;
        settings.validate()?;
        runtime_settings::save(&self.data_dir, &settings).await?;
        let mut current = self.settings.lock().await;
        let target_changed = current.target != settings.target;
        let library_paths_changed =
            current.extra_model_library_paths != settings.extra_model_library_paths;
        if *current != settings {
            let mut llama = self.llama.lock().await;
            if let Some(mut server) = llama.server.take() {
                let _ = server.stop().await;
            }
            let mut mlx = self.mlx.lock().await;
            if let Some(mut server) = mlx.server.take() {
                let _ = server.stop().await;
            }
            // A pinned binary survives target changes; otherwise re-resolve.
            if target_changed && settings.binary_override.is_none() {
                llama.binary = None;
            }
            mlx.lm_python = settings.mlx_lm_python.as_ref().map(PathBuf::from);
            mlx.vlm_python = settings.mlx_vlm_python.as_ref().map(PathBuf::from);
            let mut whisper = self.whisper.lock().await;
            whisper.binary = settings
                .whisper_binary
                .as_ref()
                .map(PathBuf::from)
                .or_else(|| whisper::resolve_binary(&self.data_dir, None));
            let mut streaming_asr = self.streaming_asr.lock().await;
            streaming_asr.python = settings
                .streaming_asr_python
                .as_ref()
                .map(PathBuf::from)
                .or_else(|| streaming_asr::resolve_python(&self.data_dir, None));
            let mut sdcpp_state = self.sdcpp.lock().await;
            sdcpp_state.binary = settings
                .sdcpp_binary
                .as_ref()
                .map(PathBuf::from)
                .or_else(|| sdcpp::resolve_binary(&self.data_dir, None));
            let mut voice_state = self.voice.lock().await;
            voice_state.python = settings
                .voice_python
                .as_ref()
                .map(PathBuf::from)
                .or_else(|| voice::resolve_python(&self.data_dir, None));
        }
        *current = settings.clone();
        drop(current);
        if library_paths_changed {
            self.invalidate_models_cache().await;
        }
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

    /// Pin a MLX Python interpreter and persist the choice.
    pub async fn activate_python(&self, kind: MlxKind, path: PathBuf) -> anyhow::Result<PathBuf> {
        anyhow::ensure!(
            path.is_file(),
            "runtime Python interpreter not found: {}",
            path.display()
        );
        anyhow::ensure!(
            mlx::python_appears_runnable(&path, kind),
            "{} failed an import check for {}",
            path.display(),
            kind.engine_id()
        );
        let mut settings = self.settings.lock().await;
        match kind {
            MlxKind::Lm => settings.mlx_lm_python = Some(path.display().to_string()),
            MlxKind::Vlm => settings.mlx_vlm_python = Some(path.display().to_string()),
        }
        runtime_settings::save(&self.data_dir, &settings).await?;
        drop(settings);
        let mut mlx = self.mlx.lock().await;
        if let Some(mut server) = mlx.server.take() {
            let _ = server.stop().await;
        }
        match kind {
            MlxKind::Lm => mlx.lm_python = Some(path.clone()),
            MlxKind::Vlm => mlx.vlm_python = Some(path.clone()),
        }
        Ok(path)
    }

    /// Pin a whisper-cli / whisperkit-cli binary and persist the choice.
    pub async fn activate_whisper(&self, path: PathBuf) -> anyhow::Result<PathBuf> {
        anyhow::ensure!(
            path.is_file(),
            "whisper binary not found: {}",
            path.display()
        );
        let runnable = {
            let candidate = path.clone();
            tokio::task::spawn_blocking(move || {
                if crate::whisperkit::is_whisperkit_binary(&candidate) {
                    crate::whisperkit::binary_appears_runnable(&candidate)
                } else {
                    whisper::binary_appears_runnable(&candidate)
                }
            })
            .await
            .unwrap_or(false)
        };
        anyhow::ensure!(
            runnable,
            "{} failed a smoke test (missing shared libraries or incompatible build)",
            path.display()
        );
        let mut settings = self.settings.lock().await;
        settings.whisper_binary = Some(path.display().to_string());
        runtime_settings::save(&self.data_dir, &settings).await?;
        drop(settings);
        self.whisper.lock().await.binary = Some(path.clone());
        Ok(path)
    }

    /// Pin an sd-cli binary and persist the choice.
    pub async fn activate_sdcpp(&self, path: PathBuf) -> anyhow::Result<PathBuf> {
        anyhow::ensure!(
            path.is_file(),
            "sd-cli binary not found: {}",
            path.display()
        );
        let runnable = {
            let candidate = path.clone();
            tokio::task::spawn_blocking(move || sdcpp::binary_appears_runnable(&candidate))
                .await
                .unwrap_or(false)
        };
        anyhow::ensure!(
            runnable,
            "{} failed a smoke test (missing shared libraries or incompatible build)",
            path.display()
        );
        let mut settings = self.settings.lock().await;
        settings.sdcpp_binary = Some(path.display().to_string());
        runtime_settings::save(&self.data_dir, &settings).await?;
        drop(settings);
        self.sdcpp.lock().await.binary = Some(path.clone());
        Ok(path)
    }

    /// Pin a PersonaPlex / Moshi Python interpreter and persist the choice.
    pub async fn activate_voice(&self, path: PathBuf) -> anyhow::Result<PathBuf> {
        anyhow::ensure!(
            path.is_file(),
            "PersonaPlex Python not found: {}",
            path.display()
        );
        let runnable = {
            let candidate = path.clone();
            tokio::task::spawn_blocking(move || voice::python_appears_runnable(&candidate))
                .await
                .unwrap_or(false)
        };
        anyhow::ensure!(
            runnable,
            "{} failed an import check for PersonaPlex (Moshi or MLX)",
            path.display()
        );
        let mut settings = self.settings.lock().await;
        settings.voice_python = Some(path.display().to_string());
        runtime_settings::save(&self.data_dir, &settings).await?;
        drop(settings);
        let mut voice_state = self.voice.lock().await;
        if let Some(mut server) = voice_state.server.take() {
            let _ = server.stop().await;
        }
        voice_state.python = Some(path.clone());
        Ok(path)
    }

    /// Pin a streaming ASR Python interpreter and persist the choice.
    pub async fn activate_streaming_asr(&self, path: PathBuf) -> anyhow::Result<PathBuf> {
        anyhow::ensure!(
            path.is_file(),
            "streaming ASR Python not found: {}",
            path.display()
        );
        let runnable = {
            let candidate = path.clone();
            tokio::task::spawn_blocking(move || streaming_asr::python_appears_runnable(&candidate))
                .await
                .unwrap_or(false)
        };
        anyhow::ensure!(
            runnable,
            "{} failed an import check for streaming ASR",
            path.display()
        );
        let mut settings = self.settings.lock().await;
        settings.streaming_asr_python = Some(path.display().to_string());
        runtime_settings::save(&self.data_dir, &settings).await?;
        drop(settings);
        self.streaming_asr.lock().await.python = Some(path.clone());
        Ok(path)
    }

    /// Forget a runtime that was removed from disk.
    pub async fn release_runtime(&self, path: &std::path::Path) -> anyhow::Result<()> {
        self.release_binary(path).await?;
        {
            let mut whisper = self.whisper.lock().await;
            if whisper.binary.as_deref() == Some(path) {
                whisper.binary = None;
            }
        }
        {
            let mut streaming_asr = self.streaming_asr.lock().await;
            if streaming_asr.python.as_deref() == Some(path) {
                streaming_asr.python = None;
            }
        }
        let mut mlx = self.mlx.lock().await;
        let served_from_deleted = mlx
            .server
            .as_ref()
            .is_some_and(|server| server.python == path);
        if served_from_deleted && let Some(mut server) = mlx.server.take() {
            let _ = server.stop().await;
        }
        if mlx.lm_python.as_deref() == Some(path) {
            mlx.lm_python = None;
        }
        if mlx.vlm_python.as_deref() == Some(path) {
            mlx.vlm_python = None;
        }
        drop(mlx);
        let mut settings = self.settings.lock().await;
        let mut changed = false;
        if settings.mlx_lm_python.as_deref() == Some(&path.display().to_string()) {
            settings.mlx_lm_python = None;
            changed = true;
        }
        if settings.mlx_vlm_python.as_deref() == Some(&path.display().to_string()) {
            settings.mlx_vlm_python = None;
            changed = true;
        }
        if settings.whisper_binary.as_deref() == Some(&path.display().to_string()) {
            settings.whisper_binary = None;
            changed = true;
        }
        if settings.streaming_asr_python.as_deref() == Some(&path.display().to_string()) {
            settings.streaming_asr_python = None;
            changed = true;
        }
        if changed {
            runtime_settings::save(&self.data_dir, &settings).await?;
        }
        Ok(())
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

    /// Stop the server if it is serving the given model (used before deletion).
    pub async fn release_model(&self, model_id: &str) {
        if model_id.starts_with("gguf:") || model_id.starts_with("gguf-ext:") {
            let extra = self.extra_library_paths().await;
            if let Ok(model_path) =
                models_store::path_for_model_id(&self.data_dir, model_id, &extra)
            {
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
            return;
        }
        if model_id.starts_with("mlx:")
            || model_id.starts_with("mlx-vlm:")
            || model_id.starts_with("mlx-ext:")
            || model_id.starts_with("mlx-vlm-ext:")
        {
            if let Ok(model_ref) = models_store::mlx_server_model_ref(
                &self.data_dir,
                model_id,
                &self.extra_library_paths().await,
            ) {
                let mut guard = self.mlx.lock().await;
                if guard
                    .server
                    .as_ref()
                    .is_some_and(|server| server.model_ref == model_ref)
                    && let Some(mut server) = guard.server.take()
                {
                    let _ = server.stop().await;
                }
            }
        }
    }

    /// Decide whether resident chat servers must be evicted to fit a generation
    /// model of `gen_bytes`, act on the configured policy, and return a plan the
    /// caller passes to [`Runtime::restore_after_generation`] once done.
    pub async fn prepare_generation_memory(&self, gen_bytes: u64) -> GenerationMemoryPlan {
        use crate::runtime_settings::GenerationMemoryPolicy;
        let settings = self.settings.lock().await.clone();
        let policy = settings.generation_memory_policy;
        if matches!(policy, GenerationMemoryPolicy::Coresident) {
            return GenerationMemoryPlan::noop("policy: keep chat and generation models resident");
        }

        let (llama_path, resident_bytes) = {
            let guard = self.llama.lock().await;
            match guard.server.as_ref() {
                Some(server) => {
                    let path = server.model_path.clone();
                    let bytes = file_size(&path);
                    (Some(path), bytes)
                }
                None => (None, 0),
            }
        };
        let mlx_resident = self.mlx.lock().await.server.is_some();
        if llama_path.is_none() && !mlx_resident {
            return GenerationMemoryPlan::noop("no chat model resident");
        }

        let must_evict = match policy {
            GenerationMemoryPolicy::Exclusive => true,
            GenerationMemoryPolicy::Coresident => false,
            GenerationMemoryPolicy::Auto => {
                let total = crate::hardware::detect().memory_bytes.unwrap_or(0);
                if total == 0 {
                    // Unknown system memory: keep the chat model to avoid churn.
                    false
                } else {
                    let headroom = u64::from(settings.generation_memory_headroom_mb) * 1024 * 1024;
                    let budget = (total as f64 * USABLE_MEMORY_FRACTION) as u64;
                    resident_bytes
                        .saturating_add(gen_bytes)
                        .saturating_add(headroom)
                        > budget
                }
            }
        };
        if !must_evict {
            return GenerationMemoryPlan::noop("chat and generation models fit together");
        }

        let reason = if matches!(policy, GenerationMemoryPolicy::Exclusive) {
            "policy: exclusive generation — evicted chat model".to_owned()
        } else {
            "evicted chat model so the generation model fits in memory".to_owned()
        };
        let reload_llama_path = if settings.reload_llm_after_generation {
            llama_path.clone()
        } else {
            None
        };
        if let Some(mut server) = self.llama.lock().await.server.take() {
            let _ = server.stop().await;
        }
        if let Some(mut server) = self.mlx.lock().await.server.take() {
            let _ = server.stop().await;
        }
        tracing::info!(reason = %reason, "pre-generation memory arbitration evicted chat model");
        GenerationMemoryPlan {
            ejected: true,
            reason,
            reload_llama_path,
        }
    }

    /// Reload a chat model evicted by [`Runtime::prepare_generation_memory`].
    /// MLX servers are left to reload lazily on the next chat request.
    pub async fn restore_after_generation(&self, plan: GenerationMemoryPlan) {
        if let Some(path) = plan.reload_llama_path
            && let Err(error) = self.ensure_server_for_model(&path).await
        {
            tracing::warn!(%error, "failed to reload chat model after generation");
        }
    }

    async fn activate_runtime_entry(&self, entry: &runtimes::RuntimeEntry) -> anyhow::Result<()> {
        match entry.engine.as_str() {
            "mlx-lm" => {
                self.activate_python(MlxKind::Lm, PathBuf::from(&entry.path))
                    .await?;
            }
            "mlx-vlm" => {
                self.activate_python(MlxKind::Vlm, PathBuf::from(&entry.path))
                    .await?;
            }
            "whisper.cpp" | "whisperkit" => {
                self.activate_whisper(PathBuf::from(&entry.path)).await?;
            }
            "streaming-asr" => {
                self.activate_streaming_asr(PathBuf::from(&entry.path))
                    .await?;
            }
            "stable-diffusion.cpp" => {
                self.activate_sdcpp(PathBuf::from(&entry.path)).await?;
            }
            "personaplex" | "personaplex-mlx" => {
                self.activate_voice(PathBuf::from(&entry.path)).await?;
            }
            _ => {
                self.activate_binary(PathBuf::from(&entry.path)).await?;
            }
        }
        Ok(())
    }

    /// Activate a runtime by inventory id (used for per-model bindings).
    pub async fn activate_runtime_by_id(&self, runtime_id: &str) -> anyhow::Result<()> {
        let path_env = std::env::var("PATH").ok();
        let active = self.active_runtimes().await;
        let entry = runtimes::find(
            &self.data_dir,
            path_env.as_deref(),
            runtime_id,
            false,
            &active,
        )
        .ok_or_else(|| anyhow::anyhow!("unknown runtime `{runtime_id}`"))?;
        self.activate_runtime_entry(&entry).await
    }

    async fn apply_model_binding(&self, model_id: &str) -> anyhow::Result<()> {
        let bindings = model_bindings::load(&self.data_dir);
        let Some(runtime_id) = bindings.get(model_id) else {
            return Ok(());
        };
        self.activate_runtime_by_id(runtime_id).await
    }

    async fn prepare_with_recovery(
        &self,
        request: &ChatCompletionRequest,
        load_tx: LoadNotifier,
    ) -> anyhow::Result<(ActiveBackend, RuntimeSettings, ChatCompletionRequest)> {
        self.apply_model_binding(&request.model).await?;
        match self.prepare_generation(request, load_tx.clone()).await {
            Ok(value) => Ok(value),
            Err(error) => {
                let Some(load_err) = error.downcast_ref::<ModelLoadError>() else {
                    return Err(error);
                };
                for hint in &load_err.fork_hints {
                    let active = self.active_runtimes().await;
                    let Some(entry) = runtimes::find_for_fork(&self.data_dir, &active, hint) else {
                        continue;
                    };
                    model_bindings::set_binding(&self.data_dir, &request.model, &entry.id).await?;
                    self.activate_runtime_entry(&entry).await?;
                    if let Some(tx) = &load_tx {
                        let _ = tx
                            .send(Ok(StreamEvent::Load {
                                phase: "fork".to_owned(),
                                message: format!("Paired with {} — retrying load…", entry.label),
                            }))
                            .await;
                    }
                    return self.prepare_generation(request, load_tx).await;
                }
                Err(ModelLoadError {
                    cause: load_err.cause.clone(),
                    fork_hints: load_err.fork_hints.clone(),
                }
                .into())
            }
        }
    }

    /// Discover an existing binary or download a managed release.
    pub async fn ensure_llama_binary(&self) -> anyhow::Result<PathBuf> {
        self.ensure_llama_binary_with_progress(None, false, Box::new(|_| {}))
            .await
    }

    pub async fn ensure_whisper_binary_with_progress(
        &self,
        target_override: Option<crate::runtime_settings::RuntimeTarget>,
        force: bool,
        mut progress: ProgressCallback,
    ) -> anyhow::Result<PathBuf> {
        let target = if let Some(target) = target_override {
            target
        } else {
            self.settings.lock().await.target
        };
        if target_override.is_none() && !force {
            let guard = self.whisper.lock().await;
            if let Some(path) = &guard.binary
                && path.is_file()
                && whisper::binary_appears_runnable(path)
            {
                progress(ProgressEvent::phase(
                    "skip",
                    "Using the active whisper-cli binary",
                ));
                progress(ProgressEvent::done(serde_json::json!({
                    "binary": path.display().to_string(),
                    "status": "ready",
                    "source": "active"
                })));
                return Ok(path.clone());
            }
        }
        let path = whisper::ensure_binary_with_progress(
            &self.http,
            &self.data_dir,
            target,
            force,
            progress,
        )
        .await?;
        if target_override.is_none() {
            let mut guard = self.whisper.lock().await;
            guard.binary = Some(path.clone());
        }
        Ok(path)
    }

    pub async fn ensure_sdcpp_binary_with_progress(
        &self,
        target_override: Option<crate::runtime_settings::RuntimeTarget>,
        force: bool,
        mut progress: ProgressCallback,
    ) -> anyhow::Result<PathBuf> {
        let target = if let Some(target) = target_override {
            target
        } else {
            self.settings.lock().await.target
        };
        if target_override.is_none() && !force {
            let guard = self.sdcpp.lock().await;
            if let Some(path) = &guard.binary
                && path.is_file()
                && sdcpp::binary_appears_runnable(path)
            {
                progress(ProgressEvent::phase(
                    "skip",
                    "Using the active sd-cli binary",
                ));
                progress(ProgressEvent::done(serde_json::json!({
                    "binary": path.display().to_string(),
                    "status": "ready",
                    "source": "active"
                })));
                return Ok(path.clone());
            }
        }
        let path =
            sdcpp::ensure_binary_with_progress(&self.http, &self.data_dir, target, force, progress)
                .await?;
        if target_override.is_none() {
            let mut guard = self.sdcpp.lock().await;
            guard.binary = Some(path.clone());
        }
        Ok(path)
    }

    pub async fn ensure_llama_binary_with_progress(
        &self,
        target_override: Option<crate::runtime_settings::RuntimeTarget>,
        force: bool,
        mut progress: ProgressCallback,
    ) -> anyhow::Result<PathBuf> {
        let target = if let Some(target) = target_override {
            target
        } else {
            self.settings.lock().await.target
        };
        if target_override.is_none() && !force {
            let guard = self.llama.lock().await;
            if let Some(path) = &guard.binary
                && path.is_file()
                && llama::binary_appears_runnable(path)
            {
                progress(ProgressEvent::phase(
                    "skip",
                    "Using the active llama-server binary",
                ));
                progress(ProgressEvent::done(serde_json::json!({
                    "binary": path.display().to_string(),
                    "status": "ready",
                    "source": "active"
                })));
                return Ok(path.clone());
            }
        }
        let path =
            llama::ensure_binary_with_progress(&self.http, &self.data_dir, target, force, progress)
                .await?;
        if target_override.is_none() {
            let mut guard = self.llama.lock().await;
            guard.binary = Some(path.clone());
        }
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

        let server = LlamaServer::start(
            &binary,
            model_path,
            &settings,
            crate::harmony::is_harmony_model(&model_path.to_string_lossy()),
        )
        .await?;
        guard.server = Some(server);
        guard.binary = Some(binary);
        Ok(())
    }

    async fn ensure_mlx_server_for_model(
        &self,
        model_id: &str,
        kind: MlxKind,
    ) -> anyhow::Result<()> {
        let settings = self.settings.lock().await.clone();
        let python = {
            let guard = self.mlx.lock().await;
            let selected = match kind {
                MlxKind::Lm => guard.lm_python.clone(),
                MlxKind::Vlm => guard.vlm_python.clone(),
            };
            if let Some(path) = selected.filter(|path| path.is_file()) {
                path
            } else {
                drop(guard);
                anyhow::bail!(
                    "no active {} runtime; build and activate one in Runtimes first",
                    kind.engine_id()
                );
            }
        };
        let extra = self.extra_library_paths().await;
        let model_ref = models_store::mlx_server_model_ref(&self.data_dir, model_id, &extra)?;

        let mut guard = self.mlx.lock().await;
        if let Some(server) = guard.server.as_mut() {
            if server.kind == kind
                && server.model_ref == model_ref
                && server.python == python
                && server.is_running()
            {
                return Ok(());
            }
            let _ = server.stop().await;
            guard.server = None;
        }

        let server = MlxServer::start(&python, kind, &model_ref, &settings).await?;
        guard.server = Some(server);
        match kind {
            MlxKind::Lm => guard.lm_python = Some(python),
            MlxKind::Vlm => guard.vlm_python = Some(python),
        }
        Ok(())
    }

    fn resolve_model(
        &self,
        model: &str,
        extra_library_paths: &[PathBuf],
    ) -> anyhow::Result<(ActiveBackend, String)> {
        if model.is_empty() {
            anyhow::bail!("a model id is required; download a model and select it first");
        }
        if model.starts_with("gguf:") || model.starts_with("gguf-ext:") {
            let model_path =
                models_store::path_for_model_id(&self.data_dir, model, extra_library_paths)?;
            anyhow::ensure!(model_path.is_file(), "model file not found for {model}");
            return Ok((
                ActiveBackend::Llama(model_path.display().to_string()),
                model.to_owned(),
            ));
        }
        if MlxKind::from_model_id(model).is_some() {
            if model.starts_with("mlx-ext:") || model.starts_with("mlx-vlm-ext:") {
                let model_path =
                    models_store::path_for_model_id(&self.data_dir, model, extra_library_paths)?;
                anyhow::ensure!(
                    model_path.is_dir(),
                    "MLX model directory not found for {model}"
                );
            } else {
                models_store::mlx_repo_id(model)?;
            }
            return Ok((ActiveBackend::Mlx(model.to_owned()), model.to_owned()));
        }
        anyhow::bail!(
            "unknown model `{model}`; download a GGUF (`gguf:…`) or MLX (`mlx:…`) model first"
        );
    }

    async fn prepare_generation(
        &self,
        request: &ChatCompletionRequest,
        load_tx: LoadNotifier,
    ) -> anyhow::Result<(ActiveBackend, RuntimeSettings, ChatCompletionRequest)> {
        async fn emit(load_tx: &LoadNotifier, phase: &str, message: &str) {
            if let Some(tx) = load_tx {
                let _ = tx
                    .send(Ok(StreamEvent::Load {
                        phase: phase.to_owned(),
                        message: message.to_owned(),
                    }))
                    .await;
            }
        }

        emit(&load_tx, "resolve", "Locating model files…").await;
        let extra = self.extra_library_paths().await;
        let (backend, model_id) = self.resolve_model(&request.model, &extra)?;
        match &backend {
            ActiveBackend::Llama(model_path) => {
                emit(&load_tx, "server", "Starting llama.cpp server…").await;
                emit(&load_tx, "load", "Loading GGUF weights into memory…").await;
                if let Err(error) = self
                    .ensure_server_for_model(std::path::Path::new(model_path))
                    .await
                {
                    let mut guard = self.llama.lock().await;
                    guard.server = None;
                    return Err(fork_hints::load_error_with_hints(
                        &self.http,
                        &self.data_dir,
                        &model_id,
                        error,
                    )
                    .await
                    .into());
                }
            }
            ActiveBackend::Mlx(_) => {
                emit(&load_tx, "server", "Starting MLX server…").await;
                emit(&load_tx, "load", "Loading MLX model weights…").await;
                let (kind, mismatch) =
                    models_store::resolve_mlx_launch_kind(&self.data_dir, &model_id, &extra)?;
                if let Some(notice) = mismatch {
                    tracing::warn!("{notice}");
                }
                if let Err(error) = self.ensure_mlx_server_for_model(&model_id, kind).await {
                    let mut guard = self.mlx.lock().await;
                    guard.server = None;
                    return Err(fork_hints::load_error_with_hints(
                        &self.http,
                        &self.data_dir,
                        &model_id,
                        error,
                    )
                    .await
                    .into());
                }
            }
        }
        emit(&load_tx, "ready", "Model ready — preparing media…").await;
        let settings = self.settings.lock().await.clone();
        let mut request = request.clone();
        let model_caps = self
            .cached_models()
            .await
            .ok()
            .and_then(|models| {
                models
                    .into_iter()
                    .find(|model| model.id == model_id)
                    .map(|model| model.capabilities)
            })
            .unwrap_or_else(|| ModelCapabilities {
                input_modalities: vec!["text".into()],
                output_modalities: vec!["text".into()],
                streaming: true,
                tools: true,
                reasoning: false,
                max_context_length: None,
                reasoning_modes: Vec::new(),
                harmony: false,
                audio_input: None,
            });
        let whisper_binary = self.whisper.lock().await.binary.clone().or_else(|| {
            whisper::resolve_binary(&self.data_dir, settings.whisper_binary.as_deref())
        });
        let whisper_model =
            whisper::resolve_model_path(&self.data_dir, settings.whisper_model.as_deref());
        let features = media::detect_pipeline_features(
            &self.data_dir,
            settings.whisper_binary.as_deref(),
            settings.whisper_model.as_deref(),
        );
        let media_ctx = MediaContext {
            data_dir: &self.data_dir,
            model_caps: &model_caps,
            features,
            whisper_binary,
            whisper_model,
            whisper_model_pref: settings.whisper_model.as_deref(),
        };
        let progress = if load_tx.is_some() {
            let tx = load_tx.clone();
            Some(Box::new(move |phase: String, message: String| {
                if let Some(tx) = &tx {
                    let _ = tx.try_send(Ok(StreamEvent::Load { phase, message }));
                }
            }) as media::ProgressFn)
        } else {
            None
        };
        media::prepare_messages(&media_ctx, &mut request.messages, progress).await?;
        emit(&load_tx, "ready", "Model ready — generating…").await;
        let harmony = crate::harmony::is_harmony_model(&model_id);
        if let Some(merged) = tool_registry::merge_definitions(&self.data_dir, &request, harmony) {
            request.tools = Some(merged);
        }
        Ok((backend, settings, request))
    }

    /// When the chat engine rejects native `input_audio`, rewrite to Whisper
    /// transcripts and return true so the caller can retry once.
    async fn fallback_native_audio_with_asr(
        &self,
        request: &mut ChatCompletionRequest,
        load_tx: LoadNotifier,
    ) -> anyhow::Result<bool> {
        let settings = self.settings.lock().await.clone();
        let model_caps = self
            .cached_models()
            .await
            .ok()
            .and_then(|models| {
                models
                    .into_iter()
                    .find(|model| model.id == request.model)
                    .map(|model| model.capabilities)
            })
            .unwrap_or_else(|| ModelCapabilities {
                input_modalities: vec!["text".into()],
                output_modalities: vec!["text".into()],
                streaming: true,
                tools: true,
                reasoning: false,
                max_context_length: None,
                reasoning_modes: Vec::new(),
                harmony: false,
                audio_input: None,
            });
        let whisper_binary = self.whisper.lock().await.binary.clone().or_else(|| {
            whisper::resolve_binary(&self.data_dir, settings.whisper_binary.as_deref())
        });
        let whisper_model =
            whisper::resolve_model_path(&self.data_dir, settings.whisper_model.as_deref());
        let features = media::detect_pipeline_features(
            &self.data_dir,
            settings.whisper_binary.as_deref(),
            settings.whisper_model.as_deref(),
        );
        let media_ctx = MediaContext {
            data_dir: &self.data_dir,
            model_caps: &model_caps,
            features,
            whisper_binary,
            whisper_model,
            whisper_model_pref: settings.whisper_model.as_deref(),
        };
        let progress = if load_tx.is_some() {
            let tx = load_tx.clone();
            Some(Box::new(move |phase: String, message: String| {
                if let Some(tx) = &tx {
                    let _ = tx.try_send(Ok(StreamEvent::Load { phase, message }));
                }
            }) as media::ProgressFn)
        } else {
            None
        };
        let converted = media::fallback_native_audio_to_asr(
            &media_ctx,
            &mut request.messages,
            progress.as_ref(),
        )
        .await?;
        Ok(converted > 0)
    }

    async fn backend_base_url(&self, backend: &ActiveBackend) -> anyhow::Result<String> {
        match backend {
            ActiveBackend::Llama(_) => {
                let guard = self.llama.lock().await;
                let Some(server) = guard.server.as_ref() else {
                    anyhow::bail!("llama-server is not running");
                };
                Ok(server.base_url.clone())
            }
            ActiveBackend::Mlx(_) => {
                let guard = self.mlx.lock().await;
                let Some(server) = guard.server.as_ref() else {
                    anyhow::bail!("MLX server is not running");
                };
                Ok(server.base_url.clone())
            }
        }
    }

    async fn reap_dead_server(&self) {
        let mut guard = self.llama.lock().await;
        if let Some(server) = guard.server.as_mut()
            && !server.is_running()
        {
            guard.server = None;
        }
        drop(guard);
        let mut guard = self.mlx.lock().await;
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
        self: &Arc<Self>,
        request: &ChatCompletionRequest,
    ) -> anyhow::Result<tokio::sync::mpsc::Receiver<anyhow::Result<StreamEvent>>> {
        let runtime = Arc::clone(self);
        let request = request.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            let prepared = runtime
                .prepare_with_recovery(&request, Some(tx.clone()))
                .await;
            let (backend, settings, request) = match prepared {
                Ok(value) => value,
                Err(error) => {
                    let _ = tx.send(Err(error)).await;
                    return;
                }
            };
            let base_url = match runtime.backend_base_url(&backend).await {
                Ok(url) => url,
                Err(error) => {
                    let _ = tx.send(Err(error)).await;
                    return;
                }
            };
            let tools_active = tool_registry::tools_enabled(
                &runtime.data_dir,
                &request,
                crate::harmony::is_harmony_model(&request.model),
            );
            if let Err(error) =
                stream_tool_rounds(&runtime, &base_url, request, settings, tools_active, &tx).await
            {
                let _ = tx.send(Err(error)).await;
            }
        });
        Ok(rx)
    }

    /// Load a model and its runtime without generating tokens (warmup on select).
    pub async fn prepare_model_stream(
        self: &Arc<Self>,
        model_id: &str,
    ) -> anyhow::Result<tokio::sync::mpsc::Receiver<anyhow::Result<StreamEvent>>> {
        let runtime = Arc::clone(self);
        let request = ChatCompletionRequest {
            model: model_id.to_owned(),
            messages: vec![OpenAiMessage {
                role: "user".to_owned(),
                content: serde_json::json!(""),
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
            reasoning_budget_tokens: None,
            tool_choice: None,
            builtin_tools: None,
            builtin_tool_names: None,
        };
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            if let Err(error) = runtime
                .prepare_with_recovery(&request, Some(tx.clone()))
                .await
            {
                let _ = tx.send(Err(error)).await;
            }
        });
        Ok(rx)
    }

    /// Stop any child inference servers (called on daemon shutdown).
    pub async fn shutdown(&self) {
        let mut guard = self.llama.lock().await;
        if let Some(mut server) = guard.server.take() {
            let _ = server.stop().await;
        }
        drop(guard);
        let mut guard = self.mlx.lock().await;
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
        self.cached_models().await
    }

    async fn generate(&self, request: &ChatCompletionRequest) -> anyhow::Result<Generation> {
        let (backend, settings, mut request) = self.prepare_generation(request, None).await?;
        let base_url = self.backend_base_url(&backend).await?;
        let tools_active = tool_registry::tools_enabled(
            &self.data_dir,
            &request,
            crate::harmony::is_harmony_model(&request.model),
        );
        let mut invocations = Vec::new();
        let mut transcript = Vec::new();
        let ctx = ToolContext {
            data_dir: &self.data_dir,
            http: &self.http,
        };
        let mut audio_fallback_attempted = false;
        for round in 0..MAX_TOOL_ROUNDS {
            let last_round = round + 1 == MAX_TOOL_ROUNDS;
            let mut body = llama::translate_chat_request(&request, &settings, "local", false);
            if last_round && let Some(object) = body.as_object_mut() {
                object.remove("tools");
            }
            let response = match llama::chat_once(&self.http, &base_url, &body).await {
                Ok(response) => response,
                Err(error) => {
                    if !audio_fallback_attempted
                        && media::messages_contain_input_audio(&request.messages)
                        && media::looks_like_audio_rejection(&error.to_string())
                    {
                        audio_fallback_attempted = true;
                        if self
                            .fallback_native_audio_with_asr(&mut request, None)
                            .await
                            .unwrap_or(false)
                        {
                            continue;
                        }
                    }
                    self.reap_dead_server().await;
                    return Err(error);
                }
            };
            let calls = llama::extract_tool_calls(&response);
            if !tools_active || calls.is_empty() {
                return Ok(Generation {
                    text: llama::extract_assistant_text(&response).unwrap_or_default(),
                    reasoning: None,
                    tool_invocations: invocations,
                    client_tool_calls: Vec::new(),
                    transcript,
                });
            }
            let round_text = llama::extract_assistant_text(&response).unwrap_or_default();
            match append_tool_round(
                &mut request.messages,
                round_text,
                &calls,
                &ctx,
                &mut invocations,
                &mut transcript,
                None,
            )
            .await
            {
                AppendRoundOutcome::Continue => {}
                AppendRoundOutcome::Stop { client_tool_calls } => {
                    return Ok(Generation {
                        text: String::new(),
                        reasoning: None,
                        tool_invocations: invocations,
                        client_tool_calls,
                        transcript,
                    });
                }
                AppendRoundOutcome::ChannelClosed => {
                    return Ok(Generation {
                        text: String::new(),
                        reasoning: None,
                        tool_invocations: invocations,
                        client_tool_calls: Vec::new(),
                        transcript,
                    });
                }
            }
        }
        anyhow::bail!("generation exceeded the tool round limit");
    }
}

/// Execute one round of tool calls: append the assistant tool-call message and
/// one `tool` message per executed call. Streams invocations to `events` when
/// provided.
enum AppendRoundOutcome {
    Continue,
    Stop {
        client_tool_calls: Vec<llama::AccumulatedToolCall>,
    },
    ChannelClosed,
}

async fn append_tool_round(
    messages: &mut Vec<OpenAiMessage>,
    round_text: String,
    calls: &[llama::AccumulatedToolCall],
    ctx: &ToolContext<'_>,
    invocations: &mut Vec<tools::ToolInvocation>,
    transcript: &mut Vec<OpenAiMessage>,
    events: Option<&tokio::sync::mpsc::Sender<anyhow::Result<StreamEvent>>>,
) -> AppendRoundOutcome {
    let assistant = OpenAiMessage {
        role: "assistant".to_owned(),
        content: serde_json::Value::String(round_text),
        tool_calls: Some(llama::tool_calls_to_json(calls)),
        tool_call_id: None,
    };
    messages.push(assistant.clone());
    transcript.push(assistant.clone());
    if let Some(tx) = events
        && tx
            .send(Ok(StreamEvent::TranscriptMessage(assistant)))
            .await
            .is_err()
    {
        return AppendRoundOutcome::ChannelClosed;
    }
    let mut client_calls = Vec::new();
    for call in calls {
        if !tool_registry::can_execute(&call.name) {
            client_calls.push(call.clone());
            continue;
        }
        let invocation = tool_registry::execute(ctx, &call.id, &call.name, &call.arguments).await;
        let tool_message = OpenAiMessage {
            role: "tool".to_owned(),
            content: serde_json::Value::String(invocation.output.clone()),
            tool_calls: None,
            tool_call_id: Some(invocation.call_id.clone()),
        };
        messages.push(tool_message.clone());
        transcript.push(tool_message.clone());
        invocations.push(invocation.clone());
        if let Some(tx) = events {
            if tx.send(Ok(StreamEvent::Tool(invocation))).await.is_err() {
                return AppendRoundOutcome::ChannelClosed;
            }
            if tx
                .send(Ok(StreamEvent::TranscriptMessage(tool_message)))
                .await
                .is_err()
            {
                return AppendRoundOutcome::ChannelClosed;
            }
        }
    }
    if !client_calls.is_empty() {
        return AppendRoundOutcome::Stop {
            client_tool_calls: client_calls,
        };
    }
    AppendRoundOutcome::Continue
}

/// Streaming generation loop with server-side tool execution.
async fn stream_tool_rounds(
    runtime: &Runtime,
    base_url: &str,
    mut request: ChatCompletionRequest,
    settings: RuntimeSettings,
    tools_active: bool,
    tx: &tokio::sync::mpsc::Sender<anyhow::Result<StreamEvent>>,
) -> anyhow::Result<()> {
    let ctx = ToolContext {
        data_dir: &runtime.data_dir,
        http: &runtime.http,
    };
    let mut audio_fallback_attempted = false;
    for round in 0..MAX_TOOL_ROUNDS {
        let last_round = round + 1 == MAX_TOOL_ROUNDS;
        let mut body = llama::translate_chat_request(&request, &settings, "local", true);
        if last_round && let Some(object) = body.as_object_mut() {
            object.remove("tools");
        }
        let mut chunks = match llama::open_chat_stream(&runtime.http, base_url, &body).await {
            Ok(chunks) => chunks,
            Err(error) => {
                if !audio_fallback_attempted
                    && media::messages_contain_input_audio(&request.messages)
                    && media::looks_like_audio_rejection(&error.to_string())
                {
                    audio_fallback_attempted = true;
                    if runtime
                        .fallback_native_audio_with_asr(&mut request, Some(tx.clone()))
                        .await
                        .unwrap_or(false)
                    {
                        continue;
                    }
                }
                return Err(error);
            }
        };
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
            for fragment in &chunk.tool_calls {
                if tx
                    .send(Ok(StreamEvent::ToolCallDelta(fragment.clone())))
                    .await
                    .is_err()
                {
                    return Ok(());
                }
            }
            accumulator.absorb(&chunk.tool_calls);
        }
        let calls = accumulator.into_calls();
        if !tools_active || calls.is_empty() {
            let _ = tx.send(Ok(StreamEvent::End)).await;
            return Ok(());
        }
        let mut invocations = Vec::new();
        let mut transcript = Vec::new();
        match append_tool_round(
            &mut request.messages,
            round_text,
            &calls,
            &ctx,
            &mut invocations,
            &mut transcript,
            Some(tx),
        )
        .await
        {
            AppendRoundOutcome::Continue => {}
            AppendRoundOutcome::Stop { client_tool_calls } => {
                let _ = tx
                    .send(Ok(StreamEvent::ClientToolCalls(client_tool_calls)))
                    .await;
                let _ = tx.send(Ok(StreamEvent::End)).await;
                return Ok(());
            }
            AppendRoundOutcome::ChannelClosed => return Ok(()),
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
                reasoning_budget_tokens: None,
                tool_choice: None,
                builtin_tools: None,
                builtin_tool_names: None,
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
                reasoning_budget_tokens: None,
                tool_choice: None,
                builtin_tools: None,
                builtin_tool_names: None,
            })
            .await
            .unwrap_err();
        assert!(error.to_string().contains("unknown model") || error.to_string().contains("gguf"));
    }
}

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeTarget {
    Auto,
    Cpu,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
}

impl RuntimeTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
        }
    }
}

/// How to manage memory when image/video generation runs while a chat model is
/// resident. Generation engines load their own weights; on shared-memory
/// machines both models can exceed RAM at once.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GenerationMemoryPolicy {
    /// Evict the chat model only when it will not fit alongside the gen model.
    #[default]
    Auto,
    /// Always keep both models resident (never evict for generation).
    Coresident,
    /// Always evict chat models before generation, reload after.
    Exclusive,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RuntimeSettings {
    pub target: RuntimeTarget,
    pub context_size: u32,
    pub batch_size: u32,
    pub threads: Option<u16>,
    /// `-1` asks Auto target selection to choose safe placement from the
    /// model size and accelerator-memory budget. Explicit targets preserve
    /// llama.cpp's `-1` meaning: offload every layer it can.
    pub gpu_layers: i32,
    pub flash_attention: bool,
    pub kv_cache_type_k: String,
    pub kv_cache_type_v: String,
    pub jinja: bool,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: Option<u32>,
    pub enable_reasoning: bool,
    /// Whether requests may start or replace a local model that is not resident.
    #[serde(default = "default_true")]
    pub jit_loading: bool,
    /// Token cap for thinking models when reasoning mode is `budget`.
    #[serde(default)]
    pub reasoning_budget_tokens: Option<u32>,
    /// When true, prior-turn reasoning/thinking is omitted from the next model
    /// request (it still appears in the UI). Default keeps reasoning in context.
    #[serde(default)]
    pub drop_reasoning_between_turns: bool,
    /// Absolute path of an explicitly activated llama-server binary. When set,
    /// it takes precedence over discovery and managed installs.
    pub binary_override: Option<String>,
    /// Absolute path of an activated mlx-lm Python interpreter.
    #[serde(default)]
    pub mlx_lm_python: Option<String>,
    /// Absolute path of an activated mlx-vlm Python interpreter.
    #[serde(default)]
    pub mlx_vlm_python: Option<String>,
    /// Absolute path of an activated Linux vLLM Python interpreter.
    #[serde(default)]
    pub vllm_python: Option<String>,
    /// Hugging Face repository served by the active vLLM runtime.
    #[serde(default)]
    pub vllm_model: Option<String>,
    /// Registered vLLM repositories and their launch configuration. vLLM owns
    /// their snapshot cache; Brazier owns only the explicit served-model list.
    #[serde(default)]
    pub vllm_models: Vec<VllmModelSettings>,
    /// Absolute path of an activated whisper-cli binary.
    #[serde(default)]
    pub whisper_binary: Option<String>,
    /// Preferred whisper model id (`whisper:…`) or absolute path.
    #[serde(default)]
    pub whisper_model: Option<String>,
    /// Absolute path of an activated streaming ASR Python interpreter.
    #[serde(default)]
    pub streaming_asr_python: Option<String>,
    /// Preferred streaming ASR model id (`streaming-asr:…`) or absolute path.
    #[serde(default)]
    pub streaming_asr_model: Option<String>,
    /// Absolute path of an activated sd-cli binary.
    #[serde(default)]
    pub sdcpp_binary: Option<String>,
    /// Default image generation model id (`sdcpp-image:…`) for tools and Generate mode.
    /// Show generated images back to the chat model when it can see images,
    /// so it can critique or iterate on its own output.
    #[serde(default = "default_true")]
    pub show_generated_images_to_model: bool,
    /// Same for video, by sampling frames. Off by default: a clip becomes
    /// several images, which costs far more context than a single picture.
    #[serde(default)]
    pub show_generated_video_to_model: bool,
    #[serde(default)]
    pub default_image_gen_model: Option<String>,
    /// Default video generation model id (`sdcpp-video:…`) for tools and Generate mode.
    #[serde(default)]
    pub default_video_gen_model: Option<String>,
    /// Flat ceiling in seconds for one generation job. Zero keeps the budget
    /// Brazier derives from the frames and steps asked for, which suits most
    /// machines; a slow CPU-only host may want hours instead.
    #[serde(default)]
    pub generation_timeout_secs: u32,
    /// Absolute path of an activated PersonaPlex / Moshi Python interpreter.
    #[serde(default)]
    pub voice_python: Option<String>,
    /// Preferred PersonaPlex model id (`personaplex:…`) or absolute path.
    #[serde(default)]
    pub default_voice_model: Option<String>,
    /// Default persona text prompt for realtime voice sessions.
    #[serde(default)]
    pub default_voice_persona: Option<String>,
    /// Default chat model id for chat and agent (they share one model). Seeded
    /// on install from the welcome recommendations.
    #[serde(default)]
    pub default_chat_model: Option<String>,
    /// Parallel compile jobs for source builds (`cmake --build … --parallel`).
    #[serde(default = "default_build_jobs")]
    pub build_jobs: u16,
    /// Additional directories to scan for GGUF models (read-only; not used for downloads).
    #[serde(default)]
    pub extra_model_library_paths: Vec<String>,
    /// Memory arbitration between chat and image/video generation models.
    #[serde(default)]
    pub generation_memory_policy: GenerationMemoryPolicy,
    /// RAM headroom (MiB) to preserve when deciding co-residency in `auto`.
    #[serde(default = "default_generation_headroom_mb")]
    pub generation_memory_headroom_mb: u32,
    /// Reload the evicted chat model after generation completes.
    #[serde(default = "default_true")]
    pub reload_llm_after_generation: bool,
    /// Chat `run_javascript` sandbox profile and optional limit overrides.
    #[serde(default)]
    pub javascript_sandbox: crate::js_sandbox::JavascriptSandboxSettings,
    /// Which backend answers `web_search`: `duckduckgo` (keyless, rate-limited)
    /// or `brave` (paid API, needs `brave_api_key`). Keyless search is
    /// intentionally limited to DuckDuckGo — it is the only keyless engine we
    /// are comfortable querying — so machines it blocks should switch to Brave.
    #[serde(default = "default_web_search_provider")]
    pub web_search_provider: String,
    /// Brave Search API key. Setting one and choosing `brave` as the provider
    /// raises the search rate limit well above the keyless budget.
    #[serde(default)]
    pub brave_api_key: Option<String>,
    /// SafeSearch level for web search: `moderate` (default), `strict`, `off`.
    #[serde(default = "default_web_safesearch")]
    pub web_safesearch: String,
    /// Default region/locale for web search (e.g. `us-en`, `de-de`, `wt-wt`).
    #[serde(default)]
    pub web_search_region: Option<String>,
}

fn default_web_search_provider() -> String {
    "duckduckgo".to_owned()
}

fn default_web_safesearch() -> String {
    "moderate".to_owned()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VllmModelSettings {
    pub repository: String,
    pub revision: Option<String>,
    pub context_size: Option<u32>,
    pub dtype: Option<String>,
    pub gpu_memory_utilization: Option<f32>,
    pub tensor_parallel_size: Option<u32>,
    pub trust_remote_code: bool,
    /// When true (default), pass `--enable-prefix-caching` at launch.
    #[serde(default = "default_true")]
    pub prefix_caching: bool,
    pub extra_args: Vec<String>,
}

impl Default for VllmModelSettings {
    fn default() -> Self {
        Self {
            repository: String::new(),
            revision: None,
            context_size: None,
            dtype: None,
            gpu_memory_utilization: None,
            tensor_parallel_size: None,
            trust_remote_code: false,
            prefix_caching: true,
            extra_args: Vec::new(),
        }
    }
}

pub fn default_build_jobs() -> u16 {
    std::thread::available_parallelism()
        .map(|count| (count.get() / 2).max(1) as u16)
        .unwrap_or(4)
}

pub fn default_generation_headroom_mb() -> u32 {
    1024
}

fn default_true() -> bool {
    true
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            target: RuntimeTarget::Auto,
            context_size: 4096,
            batch_size: 512,
            threads: None,
            gpu_layers: -1,
            flash_attention: true,
            kv_cache_type_k: "f16".to_owned(),
            kv_cache_type_v: "f16".to_owned(),
            jinja: true,
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: None,
            enable_reasoning: true,
            jit_loading: true,
            reasoning_budget_tokens: None,
            drop_reasoning_between_turns: false,
            binary_override: None,
            mlx_lm_python: None,
            mlx_vlm_python: None,
            vllm_python: None,
            vllm_model: None,
            vllm_models: Vec::new(),
            whisper_binary: None,
            whisper_model: None,
            streaming_asr_python: None,
            streaming_asr_model: None,
            sdcpp_binary: None,
            show_generated_images_to_model: true,
            show_generated_video_to_model: false,
            default_image_gen_model: None,
            default_video_gen_model: None,
            generation_timeout_secs: 0,
            voice_python: None,
            default_voice_model: None,
            default_voice_persona: None,
            default_chat_model: None,
            build_jobs: default_build_jobs(),
            extra_model_library_paths: Vec::new(),
            generation_memory_policy: GenerationMemoryPolicy::Auto,
            generation_memory_headroom_mb: default_generation_headroom_mb(),
            reload_llm_after_generation: true,
            javascript_sandbox: crate::js_sandbox::JavascriptSandboxSettings::default(),
            web_search_provider: default_web_search_provider(),
            brave_api_key: None,
            web_safesearch: default_web_safesearch(),
            web_search_region: None,
        }
    }
}

impl RuntimeSettings {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.context_size >= 512,
            "context_size must be at least 512"
        );
        anyhow::ensure!(
            (32..=8192).contains(&self.batch_size),
            "batch_size must be between 32 and 8192"
        );
        anyhow::ensure!(
            self.threads.is_none_or(|value| value > 0),
            "threads must be greater than zero"
        );
        anyhow::ensure!(
            (-1..=999).contains(&self.gpu_layers),
            "gpu_layers must be between -1 and 999"
        );
        anyhow::ensure!(
            self.temperature.is_finite() && (0.0..=2.0).contains(&self.temperature),
            "temperature must be between 0 and 2"
        );
        anyhow::ensure!(
            self.top_p.is_finite() && (0.0..=1.0).contains(&self.top_p),
            "top_p must be between 0 and 1"
        );
        anyhow::ensure!(
            self.max_tokens.is_none_or(|value| value > 0),
            "max_tokens must be greater than zero"
        );
        anyhow::ensure!(
            self.reasoning_budget_tokens.is_none_or(|value| value > 0),
            "reasoning_budget_tokens must be greater than zero"
        );
        for value in [&self.kv_cache_type_k, &self.kv_cache_type_v] {
            anyhow::ensure!(
                matches!(
                    value.as_str(),
                    "f32" | "f16" | "bf16" | "q8_0" | "q4_0" | "q4_1" | "iq4_nl" | "q5_0" | "q5_1"
                ),
                "unsupported KV cache type `{value}`"
            );
        }
        let max_jobs = std::thread::available_parallelism()
            .map(|count| count.get() as u16)
            .unwrap_or(128);
        anyhow::ensure!(
            (1..=max_jobs.max(1)).contains(&self.build_jobs),
            "build_jobs must be between 1 and {max_jobs}"
        );
        anyhow::ensure!(
            self.generation_memory_headroom_mb <= 1_048_576,
            "generation_memory_headroom_mb must be at most 1048576"
        );
        for path in &self.extra_model_library_paths {
            let path = PathBuf::from(path);
            anyhow::ensure!(
                path.is_absolute(),
                "library path must be absolute: {}",
                path.display()
            );
            anyhow::ensure!(
                path.is_dir(),
                "library path must be an existing directory: {}",
                path.display()
            );
        }
        anyhow::ensure!(
            matches!(self.web_search_provider.as_str(), "duckduckgo" | "brave"),
            "web_search_provider must be `duckduckgo` or `brave`"
        );
        anyhow::ensure!(
            matches!(self.web_safesearch.as_str(), "moderate" | "strict" | "off"),
            "web_safesearch must be `moderate`, `strict`, or `off`"
        );
        self.javascript_sandbox.validate()?;
        Ok(())
    }

    /// Point a mode default at a recommendation install, so the welcome flow
    /// leaves the app using the models it downloaded.
    ///
    /// `category` is a recommendation category (`text`, `agent`, `image`,
    /// `video`, `voice`). Returns whether anything changed.
    pub fn set_recommended_default(&mut self, category: &str, model_id: &str) -> bool {
        fn set(slot: &mut Option<String>, model_id: &str) -> bool {
            if slot.as_deref() == Some(model_id) {
                return false;
            }
            *slot = Some(model_id.to_owned());
            true
        }
        match category {
            "image" => set(&mut self.default_image_gen_model, model_id),
            "video" => set(&mut self.default_video_gen_model, model_id),
            // The recognizer (whisper) is not the speaking model; only a
            // PersonaPlex snapshot can be the default voice model.
            "voice" => {
                if model_id.starts_with("personaplex:") {
                    set(&mut self.default_voice_model, model_id)
                } else {
                    false
                }
            }
            "text" => set(&mut self.default_chat_model, model_id),
            // Chat and agent share one model in the app. When the agent model
            // is the same as the chat model this is a no-op; a different agent
            // model must not hijack the chat default.
            "agent" => {
                if self.default_chat_model.is_none()
                    || self.default_chat_model.as_deref() == Some(model_id)
                {
                    set(&mut self.default_chat_model, model_id)
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

pub fn settings_path(data_dir: &Path) -> PathBuf {
    data_dir.join("runtime-settings.json")
}

pub fn load(data_dir: &Path) -> RuntimeSettings {
    let path = settings_path(data_dir);
    let Ok(bytes) = std::fs::read(&path) else {
        return RuntimeSettings::default();
    };
    match serde_json::from_slice::<RuntimeSettings>(&bytes).and_then(|settings| {
        settings
            .validate()
            .map_err(serde::de::Error::custom)
            .map(|_| settings)
    }) {
        Ok(mut settings) => {
            // An override that does not name llama-server cannot be a
            // llama-server: earlier builds could store a voice interpreter here,
            // which broke every chat request and, while it stayed set, also
            // suppressed re-resolution when the acceleration target changed.
            if let Some(binary) = settings.binary_override.as_deref()
                && !crate::llama::is_llama_server_path(Path::new(binary))
            {
                tracing::warn!(
                    binary,
                    "discarding a binary_override that is not llama-server"
                );
                settings.binary_override = None;
            }
            settings
        }
        Err(error) => {
            tracing::warn!(%error, path = %path.display(), "ignoring invalid runtime settings");
            RuntimeSettings::default()
        }
    }
}

pub async fn save(data_dir: &Path, settings: &RuntimeSettings) -> anyhow::Result<()> {
    settings.validate()?;
    let path = settings_path(data_dir);
    crate::persistence::write_json(&path, settings, "runtime settings").await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_model_reported_context_windows_above_one_million_tokens() {
        let settings = RuntimeSettings {
            context_size: 2_000_000,
            ..RuntimeSettings::default()
        };
        settings.validate().unwrap();
    }

    #[test]
    fn web_search_settings_default_to_keyless_duckduckgo_and_validate() {
        let settings = RuntimeSettings::default();
        assert_eq!(settings.web_search_provider, "duckduckgo");
        assert_eq!(settings.web_safesearch, "moderate");
        assert!(settings.brave_api_key.is_none());
        settings.validate().unwrap();

        let brave = RuntimeSettings {
            web_search_provider: "brave".into(),
            brave_api_key: Some("key".into()),
            ..RuntimeSettings::default()
        };
        brave.validate().unwrap();

        RuntimeSettings {
            web_search_provider: "brave".into(),
            ..RuntimeSettings::default()
        }
        .validate()
        .unwrap(); // a missing key is a runtime error, not a config failure

        let bogus = RuntimeSettings {
            web_search_provider: "startpage".into(),
            ..RuntimeSettings::default()
        };
        assert!(bogus.validate().is_err());
    }

    /// Reproduces a real poisoned settings file: activating the PersonaPlex MLX
    /// runtime stored its virtualenv interpreter as the llama-server override,
    /// and every chat request then ran `python -m <model>.gguf`.
    #[tokio::test]
    async fn load_discards_an_override_that_is_not_llama_server() {
        let dir = tempfile::tempdir().unwrap();
        let settings = RuntimeSettings {
            binary_override: Some(
                "/data/engines/personaplex-mlx/builds/main-1/venv/bin/python".into(),
            ),
            ..RuntimeSettings::default()
        };
        save(dir.path(), &settings).await.unwrap();

        assert!(load(dir.path()).binary_override.is_none());
    }

    #[tokio::test]
    async fn load_keeps_a_real_llama_server_override() {
        let dir = tempfile::tempdir().unwrap();
        let settings = RuntimeSettings {
            binary_override: Some("/opt/homebrew/bin/llama-server".into()),
            ..RuntimeSettings::default()
        };
        save(dir.path(), &settings).await.unwrap();

        assert_eq!(
            load(dir.path()).binary_override.as_deref(),
            Some("/opt/homebrew/bin/llama-server")
        );
    }

    #[test]
    fn javascript_sandbox_settings_round_trip_in_runtime_settings() {
        let settings = RuntimeSettings {
            javascript_sandbox: crate::js_sandbox::JavascriptSandboxSettings {
                profile: crate::js_sandbox::JsSandboxProfile::Roomy,
                capture_console: Some(true),
                timeout_ms: Some(8_000),
                memory_mb: None,
                max_code_bytes: None,
                max_output_chars: None,
                max_stack_kb: None,
            },
            ..RuntimeSettings::default()
        };
        let encoded = serde_json::to_value(&settings).unwrap();
        let decoded: RuntimeSettings = serde_json::from_value(encoded).unwrap();
        assert_eq!(
            decoded.javascript_sandbox.profile,
            crate::js_sandbox::JsSandboxProfile::Roomy
        );
        assert_eq!(decoded.javascript_sandbox.timeout_ms, Some(8_000));
        let config = crate::js_sandbox::JsSandboxConfig::from_runtime_settings(&decoded);
        assert_eq!(config.timeout, std::time::Duration::from_millis(8_000));
        assert!(config.capture_console);
    }

    #[test]
    fn recommended_defaults_map_categories_to_mode_defaults() {
        let mut settings = RuntimeSettings::default();

        assert!(settings.set_recommended_default("image", "sdcpp-image:acme/flux"));
        assert_eq!(
            settings.default_image_gen_model.as_deref(),
            Some("sdcpp-image:acme/flux")
        );
        assert!(!settings.set_recommended_default("image", "sdcpp-image:acme/flux"));

        assert!(settings.set_recommended_default("video", "sdcpp-video:acme/wan"));
        assert_eq!(
            settings.default_video_gen_model.as_deref(),
            Some("sdcpp-video:acme/wan")
        );

        assert!(settings.set_recommended_default("voice", "personaplex:kyutai/moshi"));
        assert_eq!(
            settings.default_voice_model.as_deref(),
            Some("personaplex:kyutai/moshi")
        );
        // A whisper recognizer never becomes the default voice model.
        assert!(!settings.set_recommended_default("voice", "whisper:base"));
        assert_eq!(
            settings.default_voice_model.as_deref(),
            Some("personaplex:kyutai/moshi")
        );

        // Chat and agent share one model: the same install is a no-op, and a
        // different agent model does not hijack the chat default.
        assert!(settings.set_recommended_default("text", "gguf:acme/fara1.5/model.gguf"));
        assert!(!settings.set_recommended_default("agent", "gguf:acme/fara1.5/model.gguf"));
        assert_eq!(
            settings.default_chat_model.as_deref(),
            Some("gguf:acme/fara1.5/model.gguf")
        );
        assert!(!settings.set_recommended_default("agent", "gguf:acme/other/model.gguf"));
        assert_eq!(
            settings.default_chat_model.as_deref(),
            Some("gguf:acme/fara1.5/model.gguf")
        );

        // With no chat default yet, an agent install seeds it.
        let mut fresh = RuntimeSettings::default();
        assert!(fresh.set_recommended_default("agent", "gguf:acme/agent/model.gguf"));
        assert_eq!(
            fresh.default_chat_model.as_deref(),
            Some("gguf:acme/agent/model.gguf")
        );
    }
}

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
    /// Token cap for thinking models when reasoning mode is `budget`.
    #[serde(default)]
    pub reasoning_budget_tokens: Option<u32>,
    /// Absolute path of an explicitly activated llama-server binary. When set,
    /// it takes precedence over discovery and managed installs.
    pub binary_override: Option<String>,
    /// Absolute path of an activated mlx-lm Python interpreter.
    #[serde(default)]
    pub mlx_lm_python: Option<String>,
    /// Absolute path of an activated mlx-vlm Python interpreter.
    #[serde(default)]
    pub mlx_vlm_python: Option<String>,
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
            reasoning_budget_tokens: None,
            binary_override: None,
            mlx_lm_python: None,
            mlx_vlm_python: None,
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
            build_jobs: default_build_jobs(),
            extra_model_library_paths: Vec::new(),
            generation_memory_policy: GenerationMemoryPolicy::Auto,
            generation_memory_headroom_mb: default_generation_headroom_mb(),
            reload_llm_after_generation: true,
            javascript_sandbox: crate::js_sandbox::JavascriptSandboxSettings::default(),
        }
    }
}

impl RuntimeSettings {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            (512..=1_048_576).contains(&self.context_size),
            "context_size must be between 512 and 1048576"
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
        self.javascript_sandbox.validate()?;
        Ok(())
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
}

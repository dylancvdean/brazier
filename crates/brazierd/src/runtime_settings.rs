use std::path::{Path, PathBuf};

use anyhow::Context;
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RuntimeSettings {
    pub target: RuntimeTarget,
    pub context_size: u32,
    pub batch_size: u32,
    pub threads: Option<u16>,
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
    #[serde(default)]
    pub default_image_gen_model: Option<String>,
    /// Default video generation model id (`sdcpp-video:…`) for tools and Generate mode.
    #[serde(default)]
    pub default_video_gen_model: Option<String>,
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
}

pub fn default_build_jobs() -> u16 {
    std::thread::available_parallelism()
        .map(|count| (count.get() / 2).max(1) as u16)
        .unwrap_or(4)
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
            default_image_gen_model: None,
            default_video_gen_model: None,
            voice_python: None,
            default_voice_model: None,
            default_voice_persona: None,
            build_jobs: default_build_jobs(),
            extra_model_library_paths: Vec::new(),
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
        Ok(settings) => settings,
        Err(error) => {
            tracing::warn!(%error, path = %path.display(), "ignoring invalid runtime settings");
            RuntimeSettings::default()
        }
    }
}

pub async fn save(data_dir: &Path, settings: &RuntimeSettings) -> anyhow::Result<()> {
    settings.validate()?;
    let path = settings_path(data_dir);
    let temporary = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(settings).context("encode runtime settings")?;
    tokio::fs::write(&temporary, bytes)
        .await
        .context("write runtime settings")?;
    tokio::fs::rename(&temporary, &path)
        .await
        .context("commit runtime settings")?;
    Ok(())
}

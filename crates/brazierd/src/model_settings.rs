//! Per-model configuration: the advanced settings one model should be run
//! with, rather than the ones every model shares.
//!
//! [`crate::runtime_settings::RuntimeSettings`] holds what the whole
//! installation does — which acceleration target, how large a context, which
//! sampling defaults. That is the right shape for a preference and the wrong
//! shape for a model: a 4-step distilled diffusion model needs CFG 1.0, a
//! reasoning GGUF wants a different repetition penalty from a chat one, and a
//! LoRA belongs to the model it was trained against and to nothing else.
//!
//! So every field here is optional and means *override*: unset falls through to
//! the global settings and then to the engine's own default. What can be set
//! depends on the kind of model, because the flags are the engine's, and an
//! image model has no context window any more than a chat model has a scheduler.
//!
//! Nothing here is validated against a particular engine build. Options move
//! between llama.cpp releases, so an unrecognised flag is reported by the engine
//! when it starts rather than guessed at here.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::adapters::{self, AdapterKind};

/// Which family of settings a model takes, decided by its id and engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    /// Chat and completion models: llama.cpp, MLX, remote servers.
    Text,
    Image,
    Video,
    /// Speech to text: whisper.cpp, WhisperKit, streaming ASR.
    Transcription,
    /// Realtime speech to speech: PersonaPlex / Moshi.
    Voice,
}

impl ModelKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Video => "video",
            Self::Transcription => "transcription",
            Self::Voice => "voice",
        }
    }
}

/// The kind of settings a model id takes.
pub fn kind_for(model_id: &str) -> ModelKind {
    if model_id.starts_with("sdcpp-image:") {
        return ModelKind::Image;
    }
    if model_id.starts_with("sdcpp-video:") {
        return ModelKind::Video;
    }
    if model_id.starts_with("whisper:") || model_id.starts_with("streaming-asr:") {
        return ModelKind::Transcription;
    }
    if model_id.starts_with("personaplex:") {
        return ModelKind::Voice;
    }
    ModelKind::Text
}

/// A LoRA chosen for a model, at the strength it should be applied.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoraBinding {
    /// Catalogue id, so the binding survives the file moving.
    pub adapter_id: String,
    /// Last known path, so it still resolves when the catalogue has not been
    /// rebuilt yet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default = "default_scale")]
    pub scale: f32,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

/// A ControlNet chosen for an image or video model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlNetBinding {
    pub adapter_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// `--control-strength`; 1.0 applies the guidance fully.
    #[serde(default = "default_scale")]
    pub strength: f32,
    /// Default control image, used when a request supplies none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_path: Option<String>,
    /// Run the ControlNet on the CPU (`--control-net-cpu`), which trades speed
    /// for the VRAM the main model then keeps.
    #[serde(default)]
    pub cpu: bool,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_scale() -> f32 {
    1.0
}

fn default_true() -> bool {
    true
}

/// Everything a chat model can be told, across llama.cpp and MLX.
///
/// The launch fields are only read when Brazier starts the server, so changing
/// one restarts it; the sampling fields ride on each request and take effect
/// immediately.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TextProfile {
    // --- llama-server launch ---
    pub context_size: Option<u32>,
    pub batch_size: Option<u32>,
    /// Physical batch (`--ubatch-size`).
    pub ubatch_size: Option<u32>,
    pub threads: Option<u16>,
    pub gpu_layers: Option<i32>,
    pub flash_attention: Option<bool>,
    pub kv_cache_type_k: Option<String>,
    pub kv_cache_type_v: Option<String>,
    pub jinja: Option<bool>,
    /// Override the GGUF-bundled Jinja chat template (`--chat-template-file`).
    /// `None` keeps whatever `tokenizer.chat_template` the GGUF embeds.
    pub chat_template: Option<String>,
    /// Keep the weights resident (`--mlock`).
    pub mlock: Option<bool>,
    /// Read the weights instead of mapping them (`--no-mmap`).
    pub no_mmap: Option<bool>,
    /// `none`, `linear`, or `yarn`.
    pub rope_scaling: Option<String>,
    pub rope_freq_base: Option<f32>,
    pub rope_freq_scale: Option<f32>,
    pub yarn_orig_ctx: Option<u32>,
    /// Layers of a mixture-of-experts model to keep on the CPU
    /// (`--n-cpu-moe`), which is how a large MoE fits on a small GPU.
    pub n_cpu_moe: Option<u32>,
    pub main_gpu: Option<u32>,
    /// Comma-separated split across GPUs (`--tensor-split`).
    pub tensor_split: Option<String>,
    /// `none`, `layer`, or `row`.
    pub split_mode: Option<String>,
    pub cache_reuse: Option<u32>,
    pub defrag_threshold: Option<f32>,

    // --- sampling, sent per request ---
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub typical_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub dry_multiplier: Option<f32>,
    pub dry_base: Option<f32>,
    pub dry_allowed_length: Option<u32>,
    /// 0 off, 1 Mirostat, 2 Mirostat 2.0.
    pub mirostat: Option<u8>,
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    pub seed: Option<i64>,
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    pub enable_reasoning: Option<bool>,
    pub reasoning_budget_tokens: Option<u32>,
    /// Prepended to every conversation with this model.
    pub system_prompt: Option<String>,

    // --- agent ---
    /// Model id for `spawn_subagent` children. `None` means the parent's model.
    pub subagent_model: Option<String>,
    /// Max concurrent subagents a parent may run. Default 2 when unset.
    pub max_subagents: Option<u32>,
    /// When true, llama-server starts with `--parallel = 1 + max_subagents` so
    /// concurrent subagent generations can continuous-batch. Off by default
    /// because the context budget is shared across slots.
    pub parallel_subagents: Option<bool>,

    // --- adapters ---
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<LoraBinding>,
    /// Arguments appended to the server's command line verbatim.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub extra_args: Vec<String>,
}

/// Diffusion settings, shared by image and video models since sd-cli takes the
/// same flags for both and only the frame count separates them.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct DiffusionProfile {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub cfg_scale: Option<f32>,
    /// Distilled guidance, used by the Flux family instead of CFG.
    pub guidance: Option<f32>,
    /// Guidance applied to the init image in img2img (`--img-cfg-scale`).
    pub img_cfg_scale: Option<f32>,
    /// `euler`, `euler_a`, `heun`, `dpm2`, `dpm++2s_a`, `dpm++2m`, `dpm++2mv2`,
    /// `ipndm`, `ipndm_v`, `lcm`, `ddim_trailing`, `tcd`.
    pub sampling_method: Option<String>,
    /// `discrete`, `karras`, `exponential`, `ays`, `gits`, `smoothstep`,
    /// `sgm_uniform`, `simple`.
    pub schedule: Option<String>,
    pub clip_skip: Option<i32>,
    pub seed: Option<i64>,
    pub batch_count: Option<u32>,
    /// How far an init image is departed from, 0–1 (`--strength`).
    pub strength: Option<f32>,
    pub eta: Option<f32>,
    /// Skip-layer guidance scale and the layers it applies to.
    pub slg_scale: Option<f32>,
    pub skip_layers: Option<String>,
    pub skip_layer_start: Option<f32>,
    pub skip_layer_end: Option<f32>,
    /// Timestep shift for flow-matching models (Flux, Wan).
    pub flow_shift: Option<f32>,
    pub threads: Option<u16>,
    pub vae_tiling: Option<bool>,
    pub vae_on_cpu: Option<bool>,
    pub clip_on_cpu: Option<bool>,
    /// Flash attention in the diffusion model (`--diffusion-fa`).
    pub diffusion_fa: Option<bool>,
    /// Let sd.cpp choose phase-aware runtime and parameter placement.
    pub auto_fit: Option<bool>,
    /// Per-device graph execution budget in GiB (`--max-vram`).
    pub max_vram: Option<f32>,
    /// Where sd.cpp keeps weights between executions (`--params-backend`).
    pub params_backend: Option<String>,
    /// Stream and prefetch layers within the graph budget.
    pub stream_layers: Option<bool>,
    /// Keep weights in RAM and move them per step (`--offload-to-cpu`).
    pub offload_to_cpu: Option<bool>,
    /// `std_default` or `cuda`.
    pub rng: Option<String>,
    /// Appended to every prompt sent to this model.
    pub negative_prompt: Option<String>,
    /// Video only.
    pub video_frames: Option<u32>,
    pub fps: Option<u32>,

    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<LoraBinding>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub control_nets: Vec<ControlNetBinding>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub extra_args: Vec<String>,
}

/// whisper.cpp and streaming ASR decoding options.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TranscriptionProfile {
    /// ISO code, or `auto` to detect.
    pub language: Option<String>,
    /// Translate to English rather than transcribing in place.
    pub translate: Option<bool>,
    pub beam_size: Option<u32>,
    pub best_of: Option<u32>,
    pub temperature: Option<f32>,
    /// Tokens of previous text carried into the next window (`--max-context`).
    pub max_context: Option<i32>,
    pub max_len: Option<u32>,
    pub split_on_word: Option<bool>,
    pub word_threshold: Option<f32>,
    pub entropy_threshold: Option<f32>,
    pub logprob_threshold: Option<f32>,
    pub no_speech_threshold: Option<f32>,
    pub no_fallback: Option<bool>,
    /// Suppress non-speech tokens, which is what stops `[music]` appearing.
    pub suppress_nst: Option<bool>,
    pub threads: Option<u16>,
    pub flash_attention: Option<bool>,
    /// Text biasing the first window.
    pub initial_prompt: Option<String>,
    /// Streaming ASR: frames of audio the model may look ahead by.
    pub lookahead: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub extra_args: Vec<String>,
}

/// Realtime voice settings for a PersonaPlex / Moshi model.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VoiceProfile {
    /// Persona prompt the model is started with.
    pub persona_text: Option<String>,
    /// Built-in voice id, e.g. `NATF2`.
    pub voice_id: Option<String>,
    /// Reference clip to clone a voice from, which takes precedence over the id.
    pub voice_prompt_path: Option<String>,
    /// Weight quantisation in bits; 4 is the Apple Silicon default.
    pub quantization: Option<u8>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub extra_args: Vec<String>,
}

/// One model's overrides, in the shape its kind takes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelProfile {
    Text(TextProfile),
    Image(DiffusionProfile),
    Video(DiffusionProfile),
    Transcription(TranscriptionProfile),
    Voice(VoiceProfile),
}

impl ModelProfile {
    pub fn kind(&self) -> ModelKind {
        match self {
            Self::Text(_) => ModelKind::Text,
            Self::Image(_) => ModelKind::Image,
            Self::Video(_) => ModelKind::Video,
            Self::Transcription(_) => ModelKind::Transcription,
            Self::Voice(_) => ModelKind::Voice,
        }
    }

    /// An empty profile of the kind a model id takes.
    pub fn empty_for(model_id: &str) -> Self {
        match kind_for(model_id) {
            ModelKind::Text => Self::Text(TextProfile::default()),
            ModelKind::Image => Self::Image(DiffusionProfile::default()),
            ModelKind::Video => Self::Video(DiffusionProfile::default()),
            ModelKind::Transcription => Self::Transcription(TranscriptionProfile::default()),
            ModelKind::Voice => Self::Voice(VoiceProfile::default()),
        }
    }

    pub fn as_text(&self) -> Option<&TextProfile> {
        match self {
            Self::Text(profile) => Some(profile),
            _ => None,
        }
    }

    pub fn as_diffusion(&self) -> Option<&DiffusionProfile> {
        match self {
            Self::Image(profile) | Self::Video(profile) => Some(profile),
            _ => None,
        }
    }

    pub fn as_transcription(&self) -> Option<&TranscriptionProfile> {
        match self {
            Self::Transcription(profile) => Some(profile),
            _ => None,
        }
    }

    pub fn as_voice(&self) -> Option<&VoiceProfile> {
        match self {
            Self::Voice(profile) => Some(profile),
            _ => None,
        }
    }

    /// Whether anything is actually set. An empty profile is dropped rather
    /// than stored, so the file stays a record of decisions taken.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(profile) => profile == &TextProfile::default(),
            Self::Image(profile) | Self::Video(profile) => profile == &DiffusionProfile::default(),
            Self::Transcription(profile) => profile == &TranscriptionProfile::default(),
            Self::Voice(profile) => profile == &VoiceProfile::default(),
        }
    }

    pub fn validate(&self, model_id: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.kind() == kind_for(model_id),
            "`{model_id}` does not take {} settings",
            self.kind().as_str()
        );
        match self {
            Self::Text(profile) => validate_text(profile),
            Self::Image(profile) | Self::Video(profile) => validate_diffusion(profile),
            Self::Transcription(profile) => validate_transcription(profile),
            Self::Voice(profile) => validate_voice(profile),
        }
    }
}

fn ensure_range<T: PartialOrd + std::fmt::Display + Copy>(
    value: Option<T>,
    low: T,
    high: T,
    what: &str,
) -> anyhow::Result<()> {
    if let Some(value) = value {
        anyhow::ensure!(
            value >= low && value <= high,
            "{what} must be between {low} and {high}"
        );
    }
    Ok(())
}

fn ensure_one_of(value: Option<&String>, allowed: &[&str], what: &str) -> anyhow::Result<()> {
    if let Some(value) = value {
        anyhow::ensure!(
            allowed.contains(&value.as_str()),
            "unsupported {what} `{value}` (expected one of {})",
            allowed.join(", ")
        );
    }
    Ok(())
}

/// Command-line arguments are handed to a child process, so they are checked
/// for the shapes a shell would treat as more than an argument even though no
/// shell is involved — a NUL cannot be passed at all, and a newline in a flag
/// is a mistake worth catching here rather than in the engine's parser.
fn validate_extra_args(args: &[String]) -> anyhow::Result<()> {
    anyhow::ensure!(args.len() <= 64, "too many extra arguments");
    for arg in args {
        anyhow::ensure!(!arg.is_empty(), "an extra argument cannot be empty");
        anyhow::ensure!(arg.len() <= 512, "extra argument is too long: {arg}");
        anyhow::ensure!(
            !arg.contains('\0') && !arg.contains('\n'),
            "extra argument contains a control character: {arg}"
        );
    }
    Ok(())
}

fn validate_loras(loras: &[LoraBinding]) -> anyhow::Result<()> {
    anyhow::ensure!(loras.len() <= 16, "too many LoRAs");
    for lora in loras {
        anyhow::ensure!(!lora.adapter_id.is_empty(), "a LoRA needs an adapter id");
        anyhow::ensure!(
            lora.scale.is_finite() && (-4.0..=4.0).contains(&lora.scale),
            "LoRA scale must be between -4 and 4"
        );
    }
    Ok(())
}

fn validate_text(profile: &TextProfile) -> anyhow::Result<()> {
    ensure_range(profile.context_size, 512, 1_048_576, "context size")?;
    ensure_range(profile.batch_size, 32, 8192, "batch size")?;
    ensure_range(profile.ubatch_size, 1, 8192, "physical batch size")?;
    ensure_range(profile.gpu_layers, -1, 999, "GPU layers")?;
    ensure_range(profile.temperature, 0.0, 2.0, "temperature")?;
    ensure_range(profile.top_p, 0.0, 1.0, "top P")?;
    ensure_range(profile.min_p, 0.0, 1.0, "min P")?;
    ensure_range(profile.typical_p, 0.0, 1.0, "typical P")?;
    ensure_range(profile.top_k, 0, 100_000, "top K")?;
    ensure_range(profile.repeat_penalty, 0.0, 4.0, "repetition penalty")?;
    ensure_range(profile.presence_penalty, -2.0, 2.0, "presence penalty")?;
    ensure_range(profile.frequency_penalty, -2.0, 2.0, "frequency penalty")?;
    ensure_range(profile.dry_multiplier, 0.0, 100.0, "DRY multiplier")?;
    ensure_range(profile.dry_base, 0.0, 100.0, "DRY base")?;
    ensure_range(profile.mirostat, 0, 2, "Mirostat mode")?;
    anyhow::ensure!(
        profile.threads.is_none_or(|value| value > 0),
        "threads must be greater than zero"
    );
    anyhow::ensure!(
        profile.max_tokens.is_none_or(|value| value > 0),
        "max tokens must be greater than zero"
    );
    anyhow::ensure!(
        profile
            .reasoning_budget_tokens
            .is_none_or(|value| value > 0),
        "thinking budget must be greater than zero"
    );
    if let Some(template) = &profile.chat_template {
        anyhow::ensure!(
            template.len() <= 2 * 1024 * 1024,
            "chat template must be at most 2 MiB"
        );
        anyhow::ensure!(
            !template.contains('\0'),
            "chat template must not contain NUL bytes"
        );
    }
    for value in [&profile.kv_cache_type_k, &profile.kv_cache_type_v] {
        ensure_one_of(
            value.as_ref(),
            &[
                "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
            ],
            "KV cache type",
        )?;
    }
    ensure_one_of(
        profile.rope_scaling.as_ref(),
        &["none", "linear", "yarn"],
        "RoPE scaling",
    )?;
    ensure_one_of(
        profile.split_mode.as_ref(),
        &["none", "layer", "row"],
        "split mode",
    )?;
    if let Some(split) = &profile.tensor_split {
        anyhow::ensure!(
            split
                .chars()
                .all(|c| c.is_ascii_digit() || matches!(c, '.' | ',' | ' ')),
            "tensor split must be a comma-separated list of numbers"
        );
    }
    anyhow::ensure!(profile.stop.len() <= 8, "at most eight stop sequences");
    if let Some(model) = &profile.subagent_model {
        anyhow::ensure!(
            !model.trim().is_empty(),
            "subagent model must not be empty when set"
        );
        anyhow::ensure!(model.len() <= 512, "subagent model id is too long");
    }
    ensure_range(profile.max_subagents, 1, 8, "max subagents")?;
    validate_loras(&profile.loras)?;
    validate_extra_args(&profile.extra_args)
}

/// Default `max_subagents` when the profile leaves it unset.
pub const DEFAULT_MAX_SUBAGENTS: u32 = 2;

/// llama-server `--parallel` slots for a text profile.
///
/// Off (default): 1. On: `1 + max_subagents` so the parent plus concurrent
/// children can share continuous batching.
pub fn llama_parallel_slots(profile: Option<&TextProfile>) -> u32 {
    let enabled = profile
        .and_then(|profile| profile.parallel_subagents)
        .unwrap_or(false);
    if !enabled {
        return 1;
    }
    let max = profile
        .and_then(|profile| profile.max_subagents)
        .unwrap_or(DEFAULT_MAX_SUBAGENTS)
        .clamp(1, 8);
    1 + max
}

fn validate_diffusion(profile: &DiffusionProfile) -> anyhow::Result<()> {
    ensure_range(profile.width, 64, 4096, "width")?;
    ensure_range(profile.height, 64, 4096, "height")?;
    ensure_range(profile.steps, 1, 1000, "steps")?;
    ensure_range(profile.cfg_scale, 0.0, 30.0, "CFG scale")?;
    ensure_range(profile.guidance, 0.0, 30.0, "guidance")?;
    ensure_range(profile.img_cfg_scale, 0.0, 30.0, "image CFG scale")?;
    ensure_range(profile.strength, 0.0, 1.0, "strength")?;
    ensure_range(profile.eta, 0.0, 1.0, "eta")?;
    ensure_range(profile.clip_skip, -1, 12, "CLIP skip")?;
    ensure_range(profile.batch_count, 1, 64, "batch count")?;
    ensure_range(profile.slg_scale, 0.0, 30.0, "skip-layer guidance")?;
    ensure_range(profile.skip_layer_start, 0.0, 1.0, "skip-layer start")?;
    ensure_range(profile.skip_layer_end, 0.0, 1.0, "skip-layer end")?;
    ensure_range(profile.flow_shift, 0.0, 100.0, "flow shift")?;
    ensure_range(profile.max_vram, 0.0, 256.0, "maximum VRAM")?;
    if let Some(value) = &profile.params_backend {
        anyhow::ensure!(
            !value.is_empty()
                && value.len() <= 512
                && !value.contains('\0')
                && !value.contains('\n'),
            "parameter backend assignment is invalid"
        );
    }
    ensure_range(profile.video_frames, 1, 1024, "frames")?;
    ensure_range(profile.fps, 1, 120, "FPS")?;
    ensure_one_of(
        profile.sampling_method.as_ref(),
        &[
            "euler",
            "euler_a",
            "heun",
            "dpm2",
            "dpm++2s_a",
            "dpm++2m",
            "dpm++2mv2",
            "ipndm",
            "ipndm_v",
            "lcm",
            "ddim_trailing",
            "tcd",
        ],
        "sampling method",
    )?;
    ensure_one_of(
        profile.schedule.as_ref(),
        &[
            "default",
            "discrete",
            "karras",
            "exponential",
            "ays",
            "gits",
            "smoothstep",
            "sgm_uniform",
            "simple",
        ],
        "schedule",
    )?;
    ensure_one_of(profile.rng.as_ref(), &["std_default", "cuda"], "RNG")?;
    if let Some(layers) = &profile.skip_layers {
        anyhow::ensure!(
            layers
                .chars()
                .all(|c| c.is_ascii_digit() || matches!(c, ',' | ' ')),
            "skip layers must be a comma-separated list of layer numbers"
        );
    }
    anyhow::ensure!(profile.control_nets.len() <= 8, "too many ControlNets");
    for control in &profile.control_nets {
        anyhow::ensure!(
            !control.adapter_id.is_empty(),
            "a ControlNet needs an adapter id"
        );
        anyhow::ensure!(
            control.strength.is_finite() && (0.0..=2.0).contains(&control.strength),
            "ControlNet strength must be between 0 and 2"
        );
    }
    validate_loras(&profile.loras)?;
    validate_extra_args(&profile.extra_args)
}

fn validate_transcription(profile: &TranscriptionProfile) -> anyhow::Result<()> {
    ensure_range(profile.beam_size, 1, 64, "beam size")?;
    ensure_range(profile.best_of, 1, 64, "best of")?;
    ensure_range(profile.temperature, 0.0, 2.0, "temperature")?;
    ensure_range(profile.max_len, 0, 4096, "maximum segment length")?;
    ensure_range(profile.lookahead, 0, 1000, "lookahead")?;
    if let Some(language) = &profile.language {
        anyhow::ensure!(
            language.len() <= 16
                && language
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-'),
            "language must be an ISO code or `auto`"
        );
    }
    anyhow::ensure!(
        profile.threads.is_none_or(|value| value > 0),
        "threads must be greater than zero"
    );
    validate_extra_args(&profile.extra_args)
}

fn validate_voice(profile: &VoiceProfile) -> anyhow::Result<()> {
    if let Some(bits) = profile.quantization {
        anyhow::ensure!(
            matches!(bits, 4 | 8 | 16),
            "quantisation must be 4, 8, or 16 bits"
        );
    }
    if let Some(voice) = &profile.voice_id {
        anyhow::ensure!(
            voice.len() <= 64
                && voice
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '/' | '.')),
            "voice id contains unexpected characters"
        );
    }
    if let Some(path) = &profile.voice_prompt_path {
        anyhow::ensure!(
            Path::new(path).is_absolute(),
            "voice clip path must be absolute"
        );
    }
    validate_extra_args(&profile.extra_args)
}

/// Every model that has been configured, keyed by model id.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelSettingsStore {
    pub models: BTreeMap<String, ModelProfile>,
}

impl ModelSettingsStore {
    pub fn get(&self, model_id: &str) -> Option<&ModelProfile> {
        self.models.get(model_id)
    }

    pub fn text(&self, model_id: &str) -> Option<&TextProfile> {
        self.get(model_id).and_then(ModelProfile::as_text)
    }

    pub fn diffusion(&self, model_id: &str) -> Option<&DiffusionProfile> {
        self.get(model_id).and_then(ModelProfile::as_diffusion)
    }

    pub fn transcription(&self, model_id: &str) -> Option<&TranscriptionProfile> {
        self.get(model_id).and_then(ModelProfile::as_transcription)
    }

    pub fn voice(&self, model_id: &str) -> Option<&VoiceProfile> {
        self.get(model_id).and_then(ModelProfile::as_voice)
    }
}

pub fn settings_path(data_dir: &Path) -> PathBuf {
    data_dir.join("model-settings.json")
}

/// Read the store, treating an unreadable or invalid file as empty.
///
/// Losing an override is recoverable; refusing to start because one field of
/// one model no longer parses is not.
pub fn load(data_dir: &Path) -> ModelSettingsStore {
    let path = settings_path(data_dir);
    let Ok(bytes) = std::fs::read(&path) else {
        return ModelSettingsStore::default();
    };
    match serde_json::from_slice::<ModelSettingsStore>(&bytes) {
        Ok(store) => {
            let mut store = store;
            store
                .models
                .retain(|model_id, profile| match profile.validate(model_id) {
                    Ok(()) => true,
                    Err(error) => {
                        tracing::warn!(%error, model_id, "ignoring invalid model settings");
                        false
                    }
                });
            store
        }
        Err(error) => {
            tracing::warn!(%error, path = %path.display(), "ignoring invalid model settings");
            ModelSettingsStore::default()
        }
    }
}

pub async fn save(data_dir: &Path, store: &ModelSettingsStore) -> anyhow::Result<()> {
    let path = settings_path(data_dir);
    let temporary = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(store).context("encode model settings")?;
    tokio::fs::write(&temporary, bytes)
        .await
        .context("write model settings")?;
    tokio::fs::rename(&temporary, &path)
        .await
        .context("commit model settings")?;
    Ok(())
}

/// Store one model's profile, dropping it when nothing is set.
pub async fn set_profile(
    data_dir: &Path,
    model_id: &str,
    profile: ModelProfile,
) -> anyhow::Result<ModelSettingsStore> {
    anyhow::ensure!(!model_id.is_empty(), "a model id is required");
    profile.validate(model_id)?;
    let mut store = load(data_dir);
    if profile.is_empty() {
        store.models.remove(model_id);
    } else {
        store.models.insert(model_id.to_owned(), profile);
    }
    save(data_dir, &store).await?;
    Ok(store)
}

/// Forget a model's overrides, returning it to the global defaults.
pub async fn clear_profile(data_dir: &Path, model_id: &str) -> anyhow::Result<ModelSettingsStore> {
    let mut store = load(data_dir);
    store.models.remove(model_id);
    save(data_dir, &store).await?;
    Ok(store)
}

/// A LoRA binding resolved to a file that exists.
pub struct ResolvedLora {
    pub path: PathBuf,
    pub scale: f32,
}

/// Resolve enabled LoRA bindings to files, keeping only those an engine can
/// load. A binding whose file has gone is skipped with a warning rather than
/// failing the request: the model still runs, just without that adapter.
pub fn resolve_loras(data_dir: &Path, bindings: &[LoraBinding], engine: &str) -> Vec<ResolvedLora> {
    let mut resolved = Vec::new();
    for binding in bindings.iter().filter(|binding| binding.enabled) {
        let Some(path) =
            adapters::resolve_path(data_dir, Some(&binding.adapter_id), binding.path.as_deref())
        else {
            tracing::warn!(
                adapter = binding.adapter_id,
                "skipping a LoRA whose file is missing"
            );
            continue;
        };
        if !adapters::engines_for(AdapterKind::Lora, &path)
            .iter()
            .any(|candidate| candidate == engine)
        {
            tracing::warn!(
                adapter = binding.adapter_id,
                engine,
                "skipping a LoRA this engine cannot load"
            );
            continue;
        }
        resolved.push(ResolvedLora {
            path,
            scale: binding.scale,
        });
    }
    resolved
}

/// The ControlNet a diffusion job should run with.
///
/// stable-diffusion.cpp takes one per invocation, so the first enabled binding
/// wins and the interface says as much rather than silently dropping the rest.
pub fn resolve_control_net(
    data_dir: &Path,
    bindings: &[ControlNetBinding],
) -> Option<(PathBuf, ControlNetBinding)> {
    bindings
        .iter()
        .filter(|binding| binding.enabled)
        .find_map(|binding| {
            adapters::resolve_path(data_dir, Some(&binding.adapter_id), binding.path.as_deref())
                .map(|path| (path, binding.clone()))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_follow_the_model_id() {
        assert_eq!(kind_for("gguf:acme/model.gguf"), ModelKind::Text);
        assert_eq!(kind_for("mlx:acme/model"), ModelKind::Text);
        assert_eq!(kind_for("sdcpp-image:flux"), ModelKind::Image);
        assert_eq!(kind_for("sdcpp-video:wan"), ModelKind::Video);
        assert_eq!(kind_for("whisper:base.en"), ModelKind::Transcription);
        assert_eq!(kind_for("personaplex:kyutai/moshi"), ModelKind::Voice);
    }

    /// The kinds are not interchangeable: a scheduler is meaningless to a chat
    /// model, and storing one would silently do nothing.
    #[test]
    fn a_profile_of_the_wrong_kind_is_refused() {
        let profile = ModelProfile::Image(DiffusionProfile::default());
        assert!(profile.validate("gguf:acme/model.gguf").is_err());
        assert!(profile.validate("sdcpp-image:flux").is_ok());
    }

    #[tokio::test]
    async fn round_trips_a_text_profile() {
        let dir = tempfile::tempdir().unwrap();
        let profile = ModelProfile::Text(TextProfile {
            top_k: Some(40),
            min_p: Some(0.05),
            n_cpu_moe: Some(12),
            ..TextProfile::default()
        });
        set_profile(dir.path(), "gguf:acme/model.gguf", profile)
            .await
            .unwrap();

        let store = load(dir.path());
        let text = store.text("gguf:acme/model.gguf").unwrap();
        assert_eq!(text.top_k, Some(40));
        assert_eq!(text.n_cpu_moe, Some(12));
    }

    /// Clearing every field is how the interface says "back to defaults", so it
    /// has to remove the entry rather than store an empty one.
    #[tokio::test]
    async fn an_emptied_profile_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let model = "gguf:acme/model.gguf";
        set_profile(
            dir.path(),
            model,
            ModelProfile::Text(TextProfile {
                top_k: Some(40),
                ..TextProfile::default()
            }),
        )
        .await
        .unwrap();
        let store = set_profile(
            dir.path(),
            model,
            ModelProfile::Text(TextProfile::default()),
        )
        .await
        .unwrap();
        assert!(store.models.is_empty());
        assert!(load(dir.path()).models.is_empty());
    }

    #[test]
    fn out_of_range_values_are_refused() {
        let profile = ModelProfile::Text(TextProfile {
            temperature: Some(9.0),
            ..TextProfile::default()
        });
        assert!(profile.validate("gguf:acme/model.gguf").is_err());

        let diffusion = ModelProfile::Image(DiffusionProfile {
            sampling_method: Some("magic".into()),
            ..DiffusionProfile::default()
        });
        assert!(diffusion.validate("sdcpp-image:flux").is_err());

        let diffusion = ModelProfile::Video(DiffusionProfile {
            max_vram: Some(512.0),
            ..DiffusionProfile::default()
        });
        assert!(diffusion.validate("sdcpp-video:wan").is_err());

        let diffusion = ModelProfile::Video(DiffusionProfile {
            params_backend: Some("disk\n--unexpected".to_owned()),
            ..DiffusionProfile::default()
        });
        assert!(diffusion.validate("sdcpp-video:wan").is_err());
    }

    #[test]
    fn extra_arguments_reject_control_characters() {
        let profile = ModelProfile::Text(TextProfile {
            extra_args: vec!["--foo\nbar".into()],
            ..TextProfile::default()
        });
        assert!(profile.validate("gguf:acme/model.gguf").is_err());
    }

    #[test]
    fn subagent_settings_validate_defaults_and_bounds() {
        assert!(
            ModelProfile::Text(TextProfile::default())
                .validate("gguf:acme/model.gguf")
                .is_ok()
        );
        assert!(
            ModelProfile::Text(TextProfile {
                max_subagents: Some(2),
                subagent_model: Some("gguf:other.gguf".into()),
                ..TextProfile::default()
            })
            .validate("gguf:acme/model.gguf")
            .is_ok()
        );
        assert!(
            ModelProfile::Text(TextProfile {
                max_subagents: Some(0),
                ..TextProfile::default()
            })
            .validate("gguf:acme/model.gguf")
            .is_err()
        );
        assert!(
            ModelProfile::Text(TextProfile {
                max_subagents: Some(9),
                ..TextProfile::default()
            })
            .validate("gguf:acme/model.gguf")
            .is_err()
        );
        assert!(
            ModelProfile::Text(TextProfile {
                subagent_model: Some("  ".into()),
                ..TextProfile::default()
            })
            .validate("gguf:acme/model.gguf")
            .is_err()
        );
    }

    #[test]
    fn llama_parallel_slots_follow_the_toggle_and_max() {
        assert_eq!(llama_parallel_slots(None), 1);
        assert_eq!(
            llama_parallel_slots(Some(&TextProfile {
                parallel_subagents: Some(false),
                max_subagents: Some(4),
                ..TextProfile::default()
            })),
            1
        );
        assert_eq!(
            llama_parallel_slots(Some(&TextProfile {
                parallel_subagents: Some(true),
                ..TextProfile::default()
            })),
            1 + DEFAULT_MAX_SUBAGENTS
        );
        assert_eq!(
            llama_parallel_slots(Some(&TextProfile {
                parallel_subagents: Some(true),
                max_subagents: Some(4),
                ..TextProfile::default()
            })),
            5
        );
    }

    /// A settings file that has drifted out of range must not stop the daemon
    /// from starting; the offending model loses its overrides and says so.
    #[test]
    fn load_drops_an_entry_that_no_longer_validates() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            settings_path(dir.path()),
            r#"{"models":{"gguf:a.gguf":{"kind":"text","temperature":9.0},
                          "gguf:b.gguf":{"kind":"text","top_k":40}}}"#,
        )
        .unwrap();
        let store = load(dir.path());
        assert!(store.text("gguf:a.gguf").is_none());
        assert_eq!(store.text("gguf:b.gguf").unwrap().top_k, Some(40));
    }

    #[test]
    fn loras_are_filtered_to_engines_that_can_load_them() {
        let dir = tempfile::tempdir().unwrap();
        let root = adapters::root_for(dir.path(), AdapterKind::Lora);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("chat.gguf"), b"x").unwrap();
        std::fs::write(root.join("art.safetensors"), b"x").unwrap();

        let bindings = vec![
            LoraBinding {
                adapter_id: "lora:chat.gguf".into(),
                path: None,
                scale: 0.8,
                enabled: true,
            },
            LoraBinding {
                adapter_id: "lora:art.safetensors".into(),
                path: None,
                scale: 1.0,
                enabled: true,
            },
        ];
        let text = resolve_loras(dir.path(), &bindings, crate::runtimes::ENGINE);
        assert_eq!(text.len(), 1);
        assert_eq!(text[0].scale, 0.8);
        let image = resolve_loras(dir.path(), &bindings, crate::sdcpp::ENGINE);
        assert_eq!(image.len(), 1);
        assert!(image[0].path.ends_with("art.safetensors"));
    }

    #[test]
    fn a_disabled_lora_is_left_out() {
        let dir = tempfile::tempdir().unwrap();
        let root = adapters::root_for(dir.path(), AdapterKind::Lora);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("chat.gguf"), b"x").unwrap();
        let bindings = vec![LoraBinding {
            adapter_id: "lora:chat.gguf".into(),
            path: None,
            scale: 1.0,
            enabled: false,
        }];
        assert!(resolve_loras(dir.path(), &bindings, crate::runtimes::ENGINE).is_empty());
    }
}

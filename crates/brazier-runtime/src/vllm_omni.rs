//! vLLM-Omni image/video generation backend.
//!
//! Separate Python engine from chat vLLM. Serves one Diffusers HF model at a
//! time via `vllm serve <model> --omni`, then drives generation over HTTP:
//! - Image: `POST /v1/images/generations`
//! - Video: `POST /v1/videos` → poll → `GET /v1/videos/{id}/content`
//!
//! Server residency is origin-aware: tool-call (`Model`) jobs stop the server
//! after the request so chat can reclaim VRAM; Generate-mode (`User`) jobs keep
//! the process warm across successive clicks.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    sync::OnceLock,
    time::Duration,
};

use anyhow::Context;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use tokio::{io::AsyncWriteExt, process::{Child, Command}, sync::Mutex};

use crate::{
    generation::{
        self, ActiveGeneration, CancelledError, GenerationOrigin, JobRegistration, Modality,
    },
    model_settings::DiffusionProfile,
    runtime_settings::{RuntimeSettings, VllmOmniModelSettings},
    types::{ModelCapabilities, ModelDescriptor},
};

pub const ENGINE: &str = "vllm-omni";

const IMAGE_PREFIX: &str = "vllm-omni-image:";
const VIDEO_PREFIX: &str = "vllm-omni-video:";

/// Diffusion model load can take many minutes; chat vLLM uses 180s.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(900);
const IMAGE_TIMEOUT: Duration = Duration::from_secs(3600);
const VIDEO_TIMEOUT_BASE: Duration = Duration::from_secs(1800);
const VIDEO_TIMEOUT_PER_FRAME_STEP: Duration = Duration::from_secs(30);
const VIDEO_TIMEOUT_MAX: Duration = Duration::from_secs(24 * 3600);

const FALLBACK_WIDTH: u32 = 1024;
const FALLBACK_HEIGHT: u32 = 1024;
const FALLBACK_STEPS: u32 = 30;
const FALLBACK_VIDEO_FRAMES: u32 = 33;
const FALLBACK_FPS: u32 = 16;

// ---------------------------------------------------------------------------
// Model ids
// ---------------------------------------------------------------------------

pub fn image_model_id(repo: &str) -> anyhow::Result<String> {
    validate_model_ref(repo)?;
    Ok(format!("{IMAGE_PREFIX}{repo}"))
}

pub fn video_model_id(repo: &str) -> anyhow::Result<String> {
    validate_model_ref(repo)?;
    Ok(format!("{VIDEO_PREFIX}{repo}"))
}

pub fn parse_model_id(model_id: &str) -> anyhow::Result<(Modality, &str)> {
    if let Some(repo) = model_id.strip_prefix(IMAGE_PREFIX) {
        validate_model_ref(repo)?;
        return Ok((Modality::Image, repo));
    }
    if let Some(repo) = model_id.strip_prefix(VIDEO_PREFIX) {
        validate_model_ref(repo)?;
        return Ok((Modality::Video, repo));
    }
    anyhow::bail!("not a vLLM-Omni model id: {model_id}");
}

pub fn is_omni_model_id(model_id: &str) -> bool {
    model_id.starts_with(IMAGE_PREFIX) || model_id.starts_with(VIDEO_PREFIX)
}

fn validate_model_ref(value: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value.is_empty() && value.len() <= 300,
        "invalid vLLM-Omni model reference"
    );
    anyhow::ensure!(
        value
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != ".."),
        "invalid vLLM-Omni model reference"
    );
    anyhow::ensure!(
        value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '-' | '_' | '.')),
        "invalid vLLM-Omni model reference"
    );
    Ok(())
}

pub fn python_appears_runnable(python: &Path) -> bool {
    python.is_file()
        && std::process::Command::new(python)
            .args(["-c", "import vllm_omni"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
}

// ---------------------------------------------------------------------------
// Settings helpers
// ---------------------------------------------------------------------------

pub fn find_model_settings<'a>(
    settings: &'a RuntimeSettings,
    model_id: &str,
) -> Option<&'a VllmOmniModelSettings> {
    let Ok((modality, repo)) = parse_model_id(model_id) else {
        return None;
    };
    let modality_str = match modality {
        Modality::Image => "image",
        Modality::Video => "video",
    };
    settings.vllm_omni_models.iter().find(|entry| {
        entry.repository == repo
            && (entry.modality.is_empty() || entry.modality.eq_ignore_ascii_case(modality_str))
    })
}

pub fn supports_init_image(settings: &RuntimeSettings, model_id: &str) -> bool {
    find_model_settings(settings, model_id)
        .map(|entry| entry.supports_init_image)
        .unwrap_or(false)
}

pub fn list_model_descriptors(settings: &RuntimeSettings) -> Vec<ModelDescriptor> {
    let mut models = Vec::new();
    for entry in &settings.vllm_omni_models {
        if entry.repository.is_empty() {
            continue;
        }
        let modality = if entry.modality.eq_ignore_ascii_case("video") {
            Modality::Video
        } else {
            Modality::Image
        };
        let Ok(id) = (match modality {
            Modality::Image => image_model_id(&entry.repository),
            Modality::Video => video_model_id(&entry.repository),
        }) else {
            continue;
        };
        let (input, output) = match modality {
            Modality::Image => (vec!["text".into()], vec!["image".into()]),
            Modality::Video => {
                let mut inputs = vec!["text".into()];
                if entry.supports_init_image {
                    inputs.push("image".into());
                }
                (inputs, vec!["video".into()])
            }
        };
        models.push(ModelDescriptor {
            id,
            name: entry.repository.clone(),
            engine: ENGINE.to_owned(),
            capabilities: ModelCapabilities {
                input_modalities: input,
                output_modalities: output,
                streaming: false,
                tools: false,
                reasoning: false,
                max_context_length: None,
                reasoning_modes: Vec::new(),
                harmony: false,
                audio_input: None,
                computer_use: false,
            },
            size_bytes: None,
            read_only: false,
            library_label: None,
        });
    }
    models
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

pub struct Server {
    child: Child,
    pub base_url: String,
    pub model_ref: String,
    pub modality: Modality,
    pub python: PathBuf,
    pub launch_key: String,
}

pub fn launch_key(settings: &RuntimeSettings, model_ref: &str, modality: Modality) -> String {
    let configured = settings.vllm_omni_models.iter().find(|entry| {
        entry.repository == model_ref
            && match modality {
                Modality::Image => {
                    entry.modality.is_empty() || entry.modality.eq_ignore_ascii_case("image")
                }
                Modality::Video => entry.modality.eq_ignore_ascii_case("video"),
            }
    });
    let dtype = configured
        .and_then(|e| e.dtype.as_deref())
        .unwrap_or_default();
    let mem = configured
        .and_then(|e| e.gpu_memory_utilization)
        .map(|v| v.to_string())
        .unwrap_or_default();
    let tp = configured
        .and_then(|e| e.tensor_parallel_size)
        .map(|v| v.to_string())
        .unwrap_or_default();
    let trust = configured.map(|e| e.trust_remote_code).unwrap_or(false);
    let extra = configured
        .map(|e| e.extra_args.join("\u{1f}"))
        .unwrap_or_default();
    let rev = configured
        .and_then(|e| e.revision.as_deref())
        .unwrap_or_default();
    format!("{model_ref}|{modality:?}|{dtype}|{mem}|{tp}|{trust}|{rev}|{extra}")
}

impl Server {
    pub async fn start(
        python: &Path,
        model_ref: &str,
        modality: Modality,
        settings: &RuntimeSettings,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            python.is_file(),
            "vLLM-Omni Python interpreter missing: {}",
            python.display()
        );
        anyhow::ensure!(
            python_appears_runnable(python),
            "{} does not provide vLLM-Omni",
            python.display()
        );
        validate_model_ref(model_ref)?;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .context("reserve port for vLLM-Omni server")?;
        let port = listener.local_addr()?.port();
        drop(listener);

        let mut command = Command::new(python);
        // `vllm serve` is installed into the venv; invoke via python -m so PATH
        // does not need to include the venv bin (desktop apps often have a bare PATH).
        command
            .args([
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                model_ref,
                "--omni",
                "--host",
                "127.0.0.1",
                "--port",
            ])
            .arg(port.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        if let Some(configured) = settings.vllm_omni_models.iter().find(|entry| {
            entry.repository == model_ref
                && match modality {
                    Modality::Image => {
                        entry.modality.is_empty() || entry.modality.eq_ignore_ascii_case("image")
                    }
                    Modality::Video => entry.modality.eq_ignore_ascii_case("video"),
                }
        }) {
            if let Some(revision) = configured
                .revision
                .as_deref()
                .filter(|value| !value.is_empty())
            {
                command.args(["--revision", revision]);
            }
            if let Some(dtype) = configured
                .dtype
                .as_deref()
                .filter(|value| !value.is_empty())
            {
                command.args(["--dtype", dtype]);
            }
            if let Some(memory) = configured.gpu_memory_utilization {
                command.args(["--gpu-memory-utilization", &memory.to_string()]);
            }
            if let Some(parallel) = configured.tensor_parallel_size {
                command.args(["--tensor-parallel-size", &parallel.to_string()]);
            }
            if configured.trust_remote_code {
                command.arg("--trust-remote-code");
            }
            for arg in &configured.extra_args {
                command.arg(arg);
            }
        }

        let mut child = command
            .spawn()
            .with_context(|| format!("spawn {} vLLM-Omni", python.display()))?;
        let base_url = format!("http://127.0.0.1:{port}");
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;
        let deadline = tokio::time::Instant::now() + HEALTH_TIMEOUT;
        loop {
            if let Some(status) = child.try_wait().context("poll vLLM-Omni server")? {
                let mut stderr = String::new();
                if let Some(mut pipe) = child.stderr.take() {
                    use tokio::io::AsyncReadExt;
                    let mut bytes = Vec::new();
                    let _ = pipe.read_to_end(&mut bytes).await;
                    stderr = String::from_utf8_lossy(&bytes).into_owned();
                }
                anyhow::bail!(crate::llama::describe_server_startup_failure(
                    "vLLM-Omni server",
                    status,
                    &stderr
                ));
            }
            if client
                .get(format!("{base_url}/health"))
                .send()
                .await
                .is_ok_and(|response| response.status().is_success())
            {
                break;
            }
            if tokio::time::Instant::now() > deadline {
                let _ = child.start_kill();
                anyhow::bail!("vLLM-Omni server health check timed out at {base_url}");
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        Ok(Self {
            child,
            base_url,
            model_ref: model_ref.into(),
            modality,
            python: python.into(),
            launch_key: launch_key(settings, model_ref, modality),
        })
    }

    pub fn is_running(&mut self) -> bool {
        self.child.try_wait().is_ok_and(|status| status.is_none())
    }

    pub async fn stop(&mut self) -> anyhow::Result<()> {
        if self.child.try_wait()?.is_none() {
            self.child.start_kill()?;
            let _ = self.child.wait().await;
        }
        Ok(())
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

// ---------------------------------------------------------------------------
// Request / result types (aligned with sdcpp for dispatch)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct GenerateImageRequest {
    pub prompt: String,
    pub model_id: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub cfg_scale: Option<f32>,
    #[serde(default)]
    pub guidance: Option<f32>,
    #[serde(default)]
    pub init_image: Option<PathBuf>,
    #[serde(default)]
    pub init_image_blob: Option<String>,
    #[serde(default)]
    pub origin: GenerationOrigin,
    #[serde(default)]
    pub timeout_secs: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerateVideoRequest {
    pub prompt: String,
    pub model_id: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub cfg_scale: Option<f32>,
    #[serde(default)]
    pub guidance: Option<f32>,
    #[serde(default)]
    pub init_image: Option<PathBuf>,
    #[serde(default)]
    pub init_image_blob: Option<String>,
    #[serde(default)]
    pub origin: GenerationOrigin,
    #[serde(default)]
    pub timeout_secs: Option<u32>,
    #[serde(default)]
    pub video_frames: Option<u32>,
    #[serde(default)]
    pub fps: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GenerateResult {
    pub output_path: PathBuf,
    pub metadata: serde_json::Value,
}

fn effective_timeout(computed: Duration, override_secs: Option<u32>) -> Duration {
    match override_secs {
        Some(secs) if secs > 0 => Duration::from_secs(u64::from(secs)),
        _ => computed,
    }
}

fn video_timeout(steps: u32, frames: u32) -> Duration {
    let work = u64::from(steps.max(1)) * u64::from(frames.max(1));
    VIDEO_TIMEOUT_BASE
        .saturating_add(
            VIDEO_TIMEOUT_PER_FRAME_STEP.saturating_mul(work.min(u32::MAX as u64) as u32),
        )
        .min(VIDEO_TIMEOUT_MAX)
}

async fn job_output_dir(data_dir: &Path) -> anyhow::Result<PathBuf> {
    let dir = data_dir
        .join("tmp")
        .join("vllm-omni")
        .join(uuid::Uuid::new_v4().simple().to_string());
    tokio::fs::create_dir_all(&dir)
        .await
        .context("create vllm-omni job output directory")?;
    Ok(dir)
}

fn merge_negative_prompt(
    request: Option<&str>,
    profile: Option<&DiffusionProfile>,
) -> Option<String> {
    let from_request = request.map(str::trim).filter(|s| !s.is_empty());
    let from_profile = profile
        .and_then(|p| p.negative_prompt.as_deref())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    match (from_request, from_profile) {
        (Some(a), Some(b)) if a != b => Some(format!("{a}, {b}")),
        (Some(a), _) => Some(a.to_owned()),
        (None, Some(b)) => Some(b.to_owned()),
        (None, None) => None,
    }
}

// ---------------------------------------------------------------------------
// Runtime-facing state + generate entry points
// ---------------------------------------------------------------------------

/// Mutable omni runtime slot shared by Runtime and generation tools.
pub struct OmniState {
    pub python: Option<PathBuf>,
    pub server: Option<Server>,
}

impl Default for OmniState {
    fn default() -> Self {
        Self {
            python: None,
            server: None,
        }
    }
}

/// Process-global omni state so chat tools can generate without a Runtime handle.
static SHARED_STATE: OnceLock<Mutex<OmniState>> = OnceLock::new();

pub fn shared_state() -> &'static Mutex<OmniState> {
    SHARED_STATE.get_or_init(|| Mutex::new(OmniState::default()))
}

/// Ensure a matching omni server is running; restart when model or launch config changes.
pub async fn ensure_server(
    settings: &RuntimeSettings,
    model_ref: &str,
    modality: Modality,
) -> anyhow::Result<String> {
    let wanted_key = launch_key(settings, model_ref, modality);
    let mut guard = shared_state().lock().await;
    let python = guard
        .python
        .clone()
        .or_else(|| settings.vllm_omni_python.as_ref().map(PathBuf::from))
        .filter(|path| path.is_file())
        .ok_or_else(|| anyhow::anyhow!("no vLLM-Omni Python interpreter activated"))?;

    if let Some(server) = guard.server.as_mut() {
        if server.model_ref == model_ref
            && server.modality == modality
            && server.launch_key == wanted_key
            && server.python == python
            && server.is_running()
        {
            return Ok(server.base_url.clone());
        }
        let _ = server.stop().await;
        guard.server = None;
    }

    let server = Server::start(&python, model_ref, modality, settings).await?;
    let base_url = server.base_url.clone();
    guard.server = Some(server);
    Ok(base_url)
}

pub async fn stop_server() -> anyhow::Result<()> {
    let mut guard = shared_state().lock().await;
    if let Some(mut server) = guard.server.take() {
        server.stop().await?;
    }
    Ok(())
}

/// Generate an image. Caller must hold no generation lock; this acquires it.
pub async fn generate_image(
    data_dir: &Path,
    settings: &RuntimeSettings,
    request: &GenerateImageRequest,
    profile: Option<&DiffusionProfile>,
) -> anyhow::Result<GenerateResult> {
    let _permit = generation::try_acquire_job().await?;
    let (modality, model_ref) = parse_model_id(&request.model_id)?;
    anyhow::ensure!(
        modality == Modality::Image,
        "model `{}` is not an image model",
        request.model_id
    );

    let width = request
        .width
        .or(profile.and_then(|p| p.width))
        .unwrap_or(FALLBACK_WIDTH);
    let height = request
        .height
        .or(profile.and_then(|p| p.height))
        .unwrap_or(FALLBACK_HEIGHT);
    let steps = request
        .steps
        .or(profile.and_then(|p| p.steps))
        .unwrap_or(FALLBACK_STEPS);
    let seed = request.seed.or(profile.and_then(|p| p.seed));
    let cfg_scale = request
        .cfg_scale
        .or(profile.and_then(|p| p.cfg_scale));
    let guidance = request.guidance.or(profile.and_then(|p| p.guidance));
    let negative = merge_negative_prompt(request.negative_prompt.as_deref(), profile);
    let timeout = effective_timeout(IMAGE_TIMEOUT, request.timeout_secs);

    let job = JobRegistration::open(ActiveGeneration {
        id: uuid::Uuid::new_v4().simple().to_string(),
        modality: Modality::Image,
        model_id: request.model_id.clone(),
        prompt: request.prompt.clone(),
        negative_prompt: negative.clone(),
        init_image_blob: request.init_image_blob.clone(),
        origin: request.origin,
        elapsed_secs: 0,
        timeout_secs: timeout.as_secs(),
        current_step: 0,
        total_steps: steps,
    });

    let result = async {
        let base_url = ensure_server(settings, model_ref, modality).await?;
        if job.cancelled() {
            return Err(CancelledError.into());
        }

        let mut body = serde_json::json!({
            "prompt": request.prompt,
            "size": format!("{width}x{height}"),
            "n": 1,
            "response_format": "b64_json",
            "num_inference_steps": steps,
            "model": model_ref,
        });
        if let Some(seed) = seed {
            body["seed"] = serde_json::json!(seed);
        }
        if let Some(negative) = &negative {
            body["negative_prompt"] = serde_json::json!(negative);
        }
        if let Some(cfg) = cfg_scale {
            body["guidance_scale"] = serde_json::json!(cfg);
        }
        if let Some(g) = guidance {
            body["true_cfg_scale"] = serde_json::json!(g);
        }

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .context("build HTTP client for vLLM-Omni image")?;
        let response = tokio::select! {
            response = client
                .post(format!("{base_url}/v1/images/generations"))
                .json(&body)
                .send() => response.context("POST /v1/images/generations")?,
            _ = job.notify.notified() => {
                if job.cancelled() {
                    return Err(CancelledError.into());
                }
                anyhow::bail!("generation interrupted");
            }
        };
        if job.cancelled() {
            return Err(CancelledError.into());
        }
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("vLLM-Omni image generation failed ({status}): {text}");
        }
        let payload: serde_json::Value = response
            .json()
            .await
            .context("decode image generation response")?;
        let b64 = payload
            .pointer("/data/0/b64_json")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("image response missing data[0].b64_json"))?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .context("decode base64 image")?;

        let job_dir = job_output_dir(data_dir).await?;
        let output_path = job_dir.join("output.png");
        let mut file = tokio::fs::File::create(&output_path)
            .await
            .context("create image output file")?;
        file.write_all(&bytes)
            .await
            .context("write image output file")?;
        file.flush().await.ok();

        Ok(GenerateResult {
            output_path,
            metadata: serde_json::json!({
                "model_id": request.model_id,
                "prompt": request.prompt,
                "negative_prompt": negative,
                "width": width,
                "height": height,
                "steps": steps,
                "seed": seed,
                "cfg_scale": cfg_scale,
                "engine": ENGINE,
            }),
        })
    }
    .await;

    // Ephemeral residency for tool calls; Generate mode keeps the server warm.
    if request.origin == GenerationOrigin::Model {
        let _ = stop_server().await;
    }

    if job.cancelled() {
        return Err(CancelledError.into());
    }
    result
}

/// Generate a video via async job API.
pub async fn generate_video(
    data_dir: &Path,
    settings: &RuntimeSettings,
    request: &GenerateVideoRequest,
    profile: Option<&DiffusionProfile>,
) -> anyhow::Result<GenerateResult> {
    let _permit = generation::try_acquire_job().await?;
    let (modality, model_ref) = parse_model_id(&request.model_id)?;
    anyhow::ensure!(
        modality == Modality::Video,
        "model `{}` is not a video model",
        request.model_id
    );
    if let Some(init) = &request.init_image {
        anyhow::ensure!(init.is_file(), "init image not found: {}", init.display());
        anyhow::ensure!(
            supports_init_image(settings, &request.model_id),
            "`{}` is text-to-video only. Register it with supports_init_image for image-to-video.",
            request.model_id
        );
    }

    let width = request
        .width
        .or(profile.and_then(|p| p.width))
        .unwrap_or(FALLBACK_WIDTH);
    let height = request
        .height
        .or(profile.and_then(|p| p.height))
        .unwrap_or(FALLBACK_HEIGHT);
    let steps = request
        .steps
        .or(profile.and_then(|p| p.steps))
        .unwrap_or(FALLBACK_STEPS);
    let video_frames = request
        .video_frames
        .or(profile.and_then(|p| p.video_frames))
        .unwrap_or(FALLBACK_VIDEO_FRAMES);
    let fps = request
        .fps
        .or(profile.and_then(|p| p.fps))
        .unwrap_or(FALLBACK_FPS);
    let seed = request.seed.or(profile.and_then(|p| p.seed));
    let cfg_scale = request
        .cfg_scale
        .or(profile.and_then(|p| p.cfg_scale));
    let guidance = request.guidance.or(profile.and_then(|p| p.guidance));
    let negative = merge_negative_prompt(request.negative_prompt.as_deref(), profile);
    let timeout = effective_timeout(video_timeout(steps, video_frames), request.timeout_secs);

    let job = JobRegistration::open(ActiveGeneration {
        id: uuid::Uuid::new_v4().simple().to_string(),
        modality: Modality::Video,
        model_id: request.model_id.clone(),
        prompt: request.prompt.clone(),
        negative_prompt: negative.clone(),
        init_image_blob: request.init_image_blob.clone(),
        origin: request.origin,
        elapsed_secs: 0,
        timeout_secs: timeout.as_secs(),
        current_step: 0,
        total_steps: steps,
    });

    let result = async {
        let base_url = ensure_server(settings, model_ref, modality).await?;
        if job.cancelled() {
            return Err(CancelledError.into());
        }

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .context("build HTTP client for vLLM-Omni video")?;

        let mut form = reqwest::multipart::Form::new()
            .text("prompt", request.prompt.clone())
            .text("size", format!("{width}x{height}"))
            .text("num_frames", video_frames.to_string())
            .text("fps", fps.to_string())
            .text("num_inference_steps", steps.to_string());
        if let Some(seed) = seed {
            form = form.text("seed", seed.to_string());
        }
        if let Some(negative) = &negative {
            form = form.text("negative_prompt", negative.clone());
        }
        if let Some(cfg) = cfg_scale {
            form = form.text("guidance_scale", cfg.to_string());
        }
        if let Some(g) = guidance {
            form = form.text("true_cfg_scale", g.to_string());
        }
        if let Some(init) = &request.init_image {
            let bytes = tokio::fs::read(init)
                .await
                .context("read init image for i2v")?;
            let file_name = init
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("init.png")
                .to_owned();
            let part = reqwest::multipart::Part::bytes(bytes)
                .file_name(file_name)
                .mime_str("application/octet-stream")
                .unwrap_or_else(|_| {
                    reqwest::multipart::Part::bytes(Vec::new()).file_name("init.png")
                });
            form = form.part("input_reference", part);
        }

        let create = tokio::select! {
            response = client
                .post(format!("{base_url}/v1/videos"))
                .multipart(form)
                .send() => response.context("POST /v1/videos")?,
            _ = job.notify.notified() => {
                if job.cancelled() {
                    return Err(CancelledError.into());
                }
                anyhow::bail!("generation interrupted");
            }
        };
        if !create.status().is_success() {
            let status = create.status();
            let text = create.text().await.unwrap_or_default();
            anyhow::bail!("vLLM-Omni video create failed ({status}): {text}");
        }
        let created: serde_json::Value = create.json().await.context("decode video create")?;
        let video_id = created
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("video create response missing id"))?
            .to_owned();

        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if job.cancelled() {
                let _ = client
                    .delete(format!("{base_url}/v1/videos/{video_id}"))
                    .send()
                    .await;
                return Err(CancelledError.into());
            }
            if tokio::time::Instant::now() > deadline {
                let _ = client
                    .delete(format!("{base_url}/v1/videos/{video_id}"))
                    .send()
                    .await;
                anyhow::bail!("vLLM-Omni video job timed out after {}", timeout.as_secs());
            }

            let status_resp = client
                .get(format!("{base_url}/v1/videos/{video_id}"))
                .send()
                .await
                .context("poll video job")?;
            if !status_resp.status().is_success() {
                let status = status_resp.status();
                let text = status_resp.text().await.unwrap_or_default();
                anyhow::bail!("video job poll failed ({status}): {text}");
            }
            let status_json: serde_json::Value =
                status_resp.json().await.context("decode video status")?;
            let status = status_json
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            match status {
                "completed" => break,
                "failed" => {
                    let detail = status_json
                        .get("error")
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "unknown error".into());
                    anyhow::bail!("vLLM-Omni video job failed: {detail}");
                }
                "queued" | "in_progress" | "processing" => {
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_millis(750)) => {}
                        _ = job.notify.notified() => {
                            if job.cancelled() {
                                let _ = client
                                    .delete(format!("{base_url}/v1/videos/{video_id}"))
                                    .send()
                                    .await;
                                return Err(CancelledError.into());
                            }
                        }
                    }
                }
                other => {
                    // Unknown statuses: keep polling until timeout.
                    tracing::debug!(status = other, "vLLM-Omni video job status");
                    tokio::time::sleep(Duration::from_millis(750)).await;
                }
            }
        }

        let content = client
            .get(format!("{base_url}/v1/videos/{video_id}/content"))
            .send()
            .await
            .context("download video content")?;
        if !content.status().is_success() {
            let status = content.status();
            let text = content.text().await.unwrap_or_default();
            anyhow::bail!("video content download failed ({status}): {text}");
        }
        let bytes = content
            .bytes()
            .await
            .context("read video content body")?;

        let job_dir = job_output_dir(data_dir).await?;
        let output_path = job_dir.join("output.mp4");
        let mut file = tokio::fs::File::create(&output_path)
            .await
            .context("create video output file")?;
        file.write_all(&bytes)
            .await
            .context("write video output file")?;
        file.flush().await.ok();

        // Best-effort cleanup of server-side storage.
        let _ = client
            .delete(format!("{base_url}/v1/videos/{video_id}"))
            .send()
            .await;

        Ok(GenerateResult {
            output_path,
            metadata: serde_json::json!({
                "model_id": request.model_id,
                "prompt": request.prompt,
                "negative_prompt": negative,
                "width": width,
                "height": height,
                "steps": steps,
                "video_frames": video_frames,
                "fps": fps,
                "seed": seed,
                "cfg_scale": cfg_scale,
                "engine": ENGINE,
            }),
        })
    }
    .await;

    if request.origin == GenerationOrigin::Model {
        let _ = stop_server().await;
    }

    if job.cancelled() {
        return Err(CancelledError.into());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_ids_round_trip() {
        assert_eq!(
            image_model_id("Qwen/Qwen-Image").unwrap(),
            "vllm-omni-image:Qwen/Qwen-Image"
        );
        assert_eq!(
            video_model_id("Wan-AI/Wan2.2-T2V-A14B-Diffusers").unwrap(),
            "vllm-omni-video:Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        );
        let (m, r) = parse_model_id("vllm-omni-image:Qwen/Qwen-Image").unwrap();
        assert_eq!(m, Modality::Image);
        assert_eq!(r, "Qwen/Qwen-Image");
        assert!(parse_model_id("sdcpp-image:x").is_err());
        assert!(image_model_id("../etc").is_err());
    }

    #[test]
    fn launch_key_changes_with_settings() {
        let base = RuntimeSettings::default();
        let a = launch_key(&base, "Qwen/Qwen-Image", Modality::Image);
        let mut tuned = base.clone();
        tuned.vllm_omni_models = vec![VllmOmniModelSettings {
            repository: "Qwen/Qwen-Image".into(),
            modality: "image".into(),
            dtype: Some("float16".into()),
            ..Default::default()
        }];
        let b = launch_key(&tuned, "Qwen/Qwen-Image", Modality::Image);
        assert_ne!(a, b);
    }

    #[test]
    fn list_descriptors_skips_empty_repos() {
        let settings = RuntimeSettings {
            vllm_omni_models: vec![
                VllmOmniModelSettings {
                    repository: String::new(),
                    modality: "image".into(),
                    ..Default::default()
                },
                VllmOmniModelSettings {
                    repository: "Qwen/Qwen-Image".into(),
                    modality: "image".into(),
                    ..Default::default()
                },
                VllmOmniModelSettings {
                    repository: "Wan-AI/Wan2.2-I2V".into(),
                    modality: "video".into(),
                    supports_init_image: true,
                    ..Default::default()
                },
            ],
            ..RuntimeSettings::default()
        };
        let models = list_model_descriptors(&settings);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "vllm-omni-image:Qwen/Qwen-Image");
        assert_eq!(models[1].id, "vllm-omni-video:Wan-AI/Wan2.2-I2V");
        assert!(
            models[1]
                .capabilities
                .input_modalities
                .iter()
                .any(|m| m == "image")
        );
    }
}

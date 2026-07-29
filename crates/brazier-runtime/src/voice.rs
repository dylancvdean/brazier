//! Moshi-protocol and PersonaPlex full-duplex realtime voice engine.
//!
//! PersonaPlex (NVIDIA) and the open-source Moshi runtime it builds on
//! (Kyutai Labs) implement full-duplex speech-to-speech conversation: audio
//! streamed in and audio (plus an inner-monologue text transcript) streamed
//! back out over a single WebSocket, in real time. This module manages the
//! isolated Python environment and the `moshi.server` process lifecycle,
//! plus a minimal single-session layer. It intentionally does not proxy the
//! WebSocket itself — that belongs in the HTTP/WS layer alongside axum.
//!
//! ## Moshi WebSocket wire protocol
//!
//! Once connected to `/api/chat`, every message is a binary WebSocket frame
//! whose first byte is a tag identifying the frame kind:
//!
//! | tag    | direction         | payload                                        |
//! |--------|-------------------|-------------------------------------------------|
//! | `0x00` | server -> client  | handshake, sent once as the first frame         |
//! | `0x01` | bidirectional     | Opus-encoded audio                              |
//! | `0x02` | server -> client  | UTF-8 text (inner monologue / transcript delta) |
//!
//! See [`protocol`] for the tag constants.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};

use anyhow::Context;
use tokio::{
    process::{Child, Command},
    sync::Mutex,
};
use uuid::Uuid;

use crate::{
    builds, models_store,
    types::{ModelCapabilities, ModelDescriptor},
};

pub const ENGINE: &str = "personaplex";
/// Apple Silicon MLX port (`mu-hashmi/personaplex-mlx`); same Moshi wire protocol.
pub const ENGINE_MLX: &str = "personaplex-mlx";
/// Weight quantization the MLX backend runs at; 4-bit is the practical
/// default for on-device Apple Silicon.
const MLX_QUANTIZATION: u8 = 4;

/// Which Python package backs a given voice interpreter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceBackend {
    /// NVIDIA PersonaPlex / Kyutai `moshi.server` (Linux CUDA).
    Moshi,
    /// `personaplex_mlx.local_web` (macOS Apple Silicon).
    Mlx,
}

/// Moshi WebSocket wire protocol frame type tags (first byte of every frame).
pub mod protocol {
    /// Server -> client handshake, sent once as the first frame after connect.
    pub const HANDSHAKE: u8 = 0x00;
    /// Bidirectional Opus-encoded audio frame.
    pub const AUDIO: u8 = 0x01;
    /// Server -> client UTF-8 text frame (inner monologue / transcript delta).
    pub const TEXT: u8 = 0x02;
}

// ---------------------------------------------------------------------------
// Model store: `personaplex:{owner}/{name}` snapshots under
// `data_dir/models/personaplex/`.
// ---------------------------------------------------------------------------

pub fn models_root(data_dir: &Path) -> PathBuf {
    data_dir.join("models").join("personaplex")
}

/// Stable model id for a downloaded PersonaPlex/Moshi snapshot.
///
/// Format: `personaplex:{owner}/{name}`.
pub fn model_id_for_repo(repo_id: &str) -> anyhow::Result<String> {
    models_store::validate_repo_id(repo_id)?;
    Ok(format!("{ENGINE}:{repo_id}"))
}

/// Resolve a `personaplex:...` model id to an on-disk snapshot directory.
pub fn path_for_model_id(data_dir: &Path, model_id: &str) -> anyhow::Result<PathBuf> {
    let repo_id = model_id
        .strip_prefix(&format!("{ENGINE}:"))
        .ok_or_else(|| anyhow::anyhow!("not a PersonaPlex model id: {model_id}"))?;
    models_store::validate_repo_id(repo_id)?;
    let path = models_root(data_dir).join(repo_id);
    anyhow::ensure!(
        path.is_dir() && directory_is_personaplex_model(&path),
        "PersonaPlex model not found: {model_id}"
    );
    Ok(path)
}

pub fn download_root(data_dir: &Path, repo_id: &str) -> anyhow::Result<PathBuf> {
    models_store::validate_repo_id(repo_id)?;
    Ok(models_root(data_dir).join(repo_id))
}

/// Destination path for one file inside a PersonaPlex/Moshi snapshot.
///
/// Unlike the GGUF store, Moshi checkpoints ship `config.json`, tokenizer,
/// and `.safetensors`/`.pt` weight files, so filenames are validated as
/// plain relative paths rather than requiring a specific extension.
pub fn download_destination(
    data_dir: &Path,
    repo_id: &str,
    filename: &str,
) -> anyhow::Result<PathBuf> {
    models_store::validate_repo_id(repo_id)?;
    models_store::validate_relative_path(filename)?;
    Ok(models_root(data_dir).join(repo_id).join(filename))
}

/// Heuristic check that a directory holds a usable Moshi/PersonaPlex checkpoint.
pub fn directory_is_personaplex_model(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    dir.read_dir()
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            name.ends_with(".safetensors") || name.ends_with(".pt") || name.ends_with(".bin")
        })
}

/// List on-disk PersonaPlex/Moshi snapshots.
pub fn list_models(data_dir: &Path) -> anyhow::Result<Vec<ModelDescriptor>> {
    let root = models_root(data_dir);
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut models = Vec::new();
    for org in std::fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let org = org?;
        if !org.path().is_dir() {
            continue;
        }
        let org_name = org.file_name().to_string_lossy().into_owned();
        for model_dir in std::fs::read_dir(org.path())
            .with_context(|| format!("read {}", org.path().display()))?
        {
            let model_dir = model_dir?;
            if !model_dir.path().is_dir() {
                continue;
            }
            if !directory_is_personaplex_model(&model_dir.path()) {
                continue;
            }
            let name = model_dir.file_name().to_string_lossy().into_owned();
            let repo_id = format!("{org_name}/{name}");
            let id = model_id_for_repo(&repo_id)?;
            let size = dir_size(&model_dir.path()).unwrap_or(0);
            models.push(ModelDescriptor {
                id,
                name: repo_id,
                engine: ENGINE.to_owned(),
                capabilities: ModelCapabilities {
                    input_modalities: vec!["audio".into()],
                    output_modalities: vec!["audio".into(), "text".into()],
                    streaming: true,
                    tools: false,
                    reasoning: false,
                    max_context_length: None,
                    reasoning_modes: Vec::new(),
                    harmony: false,
                    audio_input: Some("native".into()),
                },
                size_bytes: Some(size),
                read_only: false,
                library_label: None,
            });
        }
    }
    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

fn dir_size(path: &Path) -> anyhow::Result<u64> {
    let mut total = 0_u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_file() {
            total += meta.len();
        } else if meta.is_dir() {
            total += dir_size(&entry.path())?;
        }
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Local snapshot -> PersonaPlex-MLX launch flags.
//
// `personaplex_mlx.local_web` treats `--hf-repo` strictly as a Hugging Face
// repo id (it is handed to `huggingface_hub.hf_hub_download`), so a local
// snapshot directory cannot be passed there. Instead the repo id the snapshot
// came from is passed as `--hf-repo` and every asset already on disk is wired
// up through the explicit per-file flags.
// ---------------------------------------------------------------------------

/// Hugging Face repo id implied by a snapshot directory laid out as
/// `.../{owner}/{name}` (how [`download_root`] stores them).
fn repo_id_from_snapshot_dir(dir: &Path) -> Option<String> {
    let name = dir.file_name()?.to_str()?;
    let owner = dir.parent()?.file_name()?.to_str()?;
    let repo_id = format!("{owner}/{name}");
    models_store::validate_repo_id(&repo_id).ok()?;
    Some(repo_id)
}

/// Language-model weight file to load, preferring a pre-quantized variant
/// matching `quantized` so MLX skips the quantize-after-load pass.
fn snapshot_lm_weight(dir: &Path, quantized: u8) -> Option<PathBuf> {
    let prequantized = dir.join(format!("model.q{quantized}.safetensors"));
    if prequantized.is_file() {
        return Some(prequantized);
    }
    let full = dir.join("model.safetensors");
    full.is_file().then_some(full)
}

/// Mimi audio-tokenizer checkpoint (`tokenizer-{hash}-checkpoint{n}.safetensors`).
fn snapshot_mimi_weight(dir: &Path) -> Option<PathBuf> {
    dir.read_dir()
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            name.starts_with("tokenizer-") && name.ends_with(".safetensors")
        })
}

/// Directory of built-in voice prompts, extracting `voices.tgz` on first use.
///
/// Returns `None` when the snapshot has no voices archive, in which case the
/// server falls back to downloading it from `--hf-repo`.
fn snapshot_voice_prompt_dir(dir: &Path) -> Option<PathBuf> {
    let voices = dir.join("voices");
    if voices.is_dir() {
        return Some(voices);
    }
    let archive = dir.join("voices.tgz");
    if !archive.is_file() {
        return None;
    }
    let file = std::fs::File::open(&archive).ok()?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    // The archive already contains a single top-level `voices/` directory.
    tar.unpack(dir).ok()?;
    voices.is_dir().then_some(voices)
}

/// Resolve a preferred model id/path, falling back to the first on-disk snapshot.
pub fn resolve_model_path(data_dir: &Path, preferred: Option<&str>) -> Option<PathBuf> {
    if let Some(id) = preferred {
        if let Ok(path) = path_for_model_id(data_dir, id) {
            return Some(path);
        }
        let as_path = PathBuf::from(id);
        if as_path.is_dir() && directory_is_personaplex_model(&as_path) {
            return Some(as_path);
        }
    }
    list_models(data_dir)
        .ok()?
        .into_iter()
        .next()
        .and_then(|model| path_for_model_id(data_dir, &model.id).ok())
}

// ---------------------------------------------------------------------------
// Python resolution (mirrors `streaming_asr::resolve_python`).
// ---------------------------------------------------------------------------

/// Resolve the Python interpreter to use for realtime voice.
///
/// Prefers an explicit override, then the most recent completed source build
/// for either `personaplex` (Moshi) or `personaplex-mlx`.
pub fn resolve_python(data_dir: &Path, override_path: Option<&str>) -> Option<PathBuf> {
    if let Some(path) = override_path
        .map(PathBuf::from)
        .filter(|path| path.is_file())
    {
        return Some(path);
    }
    for engine in [ENGINE, ENGINE_MLX] {
        for (_, record) in builds::list_builds(data_dir, engine) {
            let path = PathBuf::from(&record.binary);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn python_has_module(python: &Path, module: &str) -> bool {
    if !python.is_file() {
        return false;
    }
    std::process::Command::new(python)
        .args(["-c", &format!("import {module}")])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Detect whether a Python env provides Moshi or PersonaPlex-MLX.
pub fn detect_backend(python: &Path) -> Option<VoiceBackend> {
    // Prefer MLX when both are somehow present (Apple Silicon builds).
    if python_has_module(python, "personaplex_mlx") {
        Some(VoiceBackend::Mlx)
    } else if python_has_module(python, "moshi") {
        Some(VoiceBackend::Moshi)
    } else {
        None
    }
}

/// Verify that a Python interpreter can import a supported voice package.
pub fn python_appears_runnable(python: &Path) -> bool {
    detect_backend(python).is_some()
}

/// Whether realtime speech-to-speech looks usable with the given (already
/// resolved) Python interpreter and, optionally, a local model snapshot.
///
/// A model is not strictly required: both Moshi and PersonaPlex-MLX can fetch
/// a default checkpoint via Hugging Face when no local snapshot is selected.
pub fn realtime_voice_available(python: Option<&Path>, model: Option<&Path>) -> bool {
    let Some(python) = python else {
        return false;
    };
    if !python_appears_runnable(python) {
        return false;
    }
    model.is_none_or(Path::is_dir)
}

// ---------------------------------------------------------------------------
// Server lifecycle.
// ---------------------------------------------------------------------------

/// Build the client-facing WebSocket proxy URL for a running server on `port`.
pub fn proxy_ws_url(port: u16) -> String {
    format!("ws://127.0.0.1:{port}/api/chat")
}

/// Options for launching a realtime voice server process.
#[derive(Debug, Clone, Default)]
pub struct VoiceLaunchOptions {
    /// Persona / system-style text prompt (MLX `--text-prompt`; ignored by Moshi CLI).
    pub persona_text: Option<String>,
    /// Optional reference voice clip path (MLX `--voice-prompt`).
    pub voice_prompt: Option<PathBuf>,
    /// Built-in voice id when no clip is provided (MLX `--voice`, default NATF2).
    pub voice_id: Option<String>,
    /// Hugging Face token for gated model downloads on first run.
    pub hf_token: Option<String>,
    /// Weight quantisation in bits (MLX `-q`); 4 is the on-device default.
    pub quantization: Option<u8>,
    /// Arguments appended to the server's command line verbatim.
    pub extra_args: Vec<String>,
}

impl VoiceLaunchOptions {
    /// Lay a model's configured voice settings under whatever the session asked
    /// for, so a session that names nothing still starts the model the way it
    /// was configured.
    pub fn with_profile(mut self, profile: Option<&crate::model_settings::VoiceProfile>) -> Self {
        let Some(profile) = profile else { return self };
        if self
            .persona_text
            .as_deref()
            .map(str::trim)
            .is_none_or(str::is_empty)
        {
            self.persona_text = profile.persona_text.clone();
        }
        if self.voice_prompt.is_none() {
            self.voice_prompt = profile
                .voice_prompt_path
                .as_deref()
                .map(PathBuf::from)
                .filter(|path| path.is_file());
        }
        if self.voice_id.is_none() {
            self.voice_id = profile.voice_id.clone();
        }
        self.quantization = self.quantization.or(profile.quantization);
        self.extra_args.extend(profile.extra_args.iter().cloned());
        self
    }
}

/// A running Moshi or PersonaPlex-MLX process bound to loopback.
///
/// Both backends serve the realtime chat WebSocket (`/api/chat`) on the same
/// HTTP port, so `base_url` doubles as the health-check target and the prefix
/// for the WS proxy URL.
pub struct VoiceServer {
    child: Child,
    pub base_url: String,
    pub python: PathBuf,
    pub model_path: Option<PathBuf>,
    pub backend: VoiceBackend,
}

impl VoiceServer {
    /// Spawn the appropriate voice server on an ephemeral loopback port and
    /// wait for it to start accepting connections.
    ///
    /// Moshi: `python -m moshi.server --host --port [--hf-repo]`.
    /// MLX: `python -m personaplex_mlx.local_web --host --port --no-browser
    ///       --text-prompt --voice|-voice-prompt [-q 4] [--static none]
    ///       [--hf-repo] [--lm-config --tokenizer --moshi-weight
    ///       --mimi-weight --voice-prompt-dir]`.
    ///
    /// When `model_path` is provided (a local snapshot directory), Moshi loads
    /// it directly via `--hf-repo`; MLX instead takes the repo id in
    /// `--hf-repo` and each local asset through its own flag, since it only
    /// ever resolves `--hf-repo` through the Hugging Face hub.
    pub async fn start(
        python: &Path,
        model_path: Option<&Path>,
        options: VoiceLaunchOptions,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            python.is_file(),
            "PersonaPlex Python interpreter missing: {}",
            python.display()
        );
        let backend = detect_backend(python).ok_or_else(|| {
            anyhow::anyhow!(
                "{} does not provide `moshi` or `personaplex_mlx`",
                python.display()
            )
        })?;
        if let Some(path) = model_path {
            anyhow::ensure!(
                path.is_dir(),
                "PersonaPlex model directory missing: {}",
                path.display()
            );
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .context("reserve port for PersonaPlex server")?;
        let port = listener.local_addr()?.port();
        drop(listener);

        let mut command = Command::new(python);
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(token) = options.hf_token.as_deref() {
            command.env("HF_TOKEN", token);
            command.env("HUGGING_FACE_HUB_TOKEN", token);
        }

        match backend {
            VoiceBackend::Moshi => {
                command
                    .arg("-m")
                    .arg("moshi.server")
                    .arg("--host")
                    .arg("127.0.0.1")
                    .arg("--port")
                    .arg(port.to_string());
                if let Some(path) = model_path {
                    command.arg("--hf-repo").arg(path);
                }
            }
            VoiceBackend::Mlx => {
                let persona = options
                    .persona_text
                    .as_deref()
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or("You are a helpful assistant.");
                command
                    .arg("-m")
                    .arg("personaplex_mlx.local_web")
                    .arg("--host")
                    .arg("127.0.0.1")
                    .arg("--port")
                    .arg(port.to_string())
                    .arg("--no-browser")
                    .arg("--text-prompt")
                    .arg(persona)
                    // 4-bit is the practical default for on-device Apple Silicon.
                    .arg("-q")
                    .arg(options.quantization.unwrap_or(MLX_QUANTIZATION).to_string())
                    // Clients talk to the WS proxy, never the bundled web UI;
                    // skipping it also avoids a `dist.tgz` download on startup.
                    .arg("--static")
                    .arg("none");
                if let Some(path) = options.voice_prompt.as_ref() {
                    command.arg("--voice-prompt").arg(path);
                } else {
                    let voice = options
                        .voice_id
                        .as_deref()
                        .filter(|value| !value.trim().is_empty())
                        .unwrap_or("NATF2");
                    command.arg("--voice").arg(voice);
                }
                if let Some(dir) = model_path {
                    // `--hf-repo` must stay a repo id (the server downloads
                    // through it, and the 7B config fallback keys off its
                    // exact value); local files are passed file by file.
                    if let Some(repo_id) = repo_id_from_snapshot_dir(dir) {
                        command.arg("--hf-repo").arg(repo_id);
                    }
                    let config = dir.join("config.json");
                    if config.is_file() {
                        command.arg("--lm-config").arg(config);
                    }
                    let tokenizer = dir.join("tokenizer_spm_32k_3.model");
                    if tokenizer.is_file() {
                        command.arg("--tokenizer").arg(tokenizer);
                    }
                    // The weight file has to match the quantisation asked for,
                    // or the server loads one precision and is told another.
                    if let Some(weight) =
                        snapshot_lm_weight(dir, options.quantization.unwrap_or(MLX_QUANTIZATION))
                    {
                        command.arg("--moshi-weight").arg(weight);
                    }
                    if let Some(weight) = snapshot_mimi_weight(dir) {
                        command.arg("--mimi-weight").arg(weight);
                    }
                    if let Some(voices) = snapshot_voice_prompt_dir(dir) {
                        command.arg("--voice-prompt-dir").arg(voices);
                    }
                }
            }
        }

        for arg in &options.extra_args {
            command.arg(arg);
        }

        let module = match backend {
            VoiceBackend::Moshi => "moshi.server",
            VoiceBackend::Mlx => "personaplex_mlx.local_web",
        };
        let mut child = command
            .spawn()
            .with_context(|| format!("spawn {} -m {module}", python.display()))?;

        let base_url = format!("http://127.0.0.1:{port}");
        // MLX first-run model download can exceed Moshi's typical warm-up.
        let timeout_secs = match backend {
            VoiceBackend::Moshi => 180,
            VoiceBackend::Mlx => 600,
        };
        let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
        wait_until_listening(&mut child, port, &base_url, deadline).await?;

        Ok(Self {
            child,
            base_url,
            python: python.to_path_buf(),
            model_path: model_path.map(Path::to_path_buf),
            backend,
        })
    }

    fn port(&self) -> u16 {
        self.base_url
            .rsplit(':')
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or_default()
    }

    /// WebSocket URL clients should be proxied to for realtime chat.
    pub fn proxy_url(&self) -> String {
        proxy_ws_url(self.port())
    }

    pub fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    pub async fn stop(&mut self) -> anyhow::Result<()> {
        if self.child.try_wait()?.is_none() {
            self.child.start_kill().context("kill PersonaPlex server")?;
            let _ = self.child.wait().await;
        }
        Ok(())
    }
}

impl Drop for VoiceServer {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

/// Poll until the server accepts a loopback TCP connection (or exits/times out).
///
/// A bare TCP connect is used rather than an HTTP health endpoint because
/// Moshi does not define one; the same port also answers plain HTTP for the
/// bundled web UI, so a GET to `base_url` would work equally well once the
/// process is listening.
async fn wait_until_listening(
    child: &mut Child,
    port: u16,
    base_url: &str,
    deadline: tokio::time::Instant,
) -> anyhow::Result<()> {
    loop {
        if let Some(status) = child.try_wait().context("poll PersonaPlex server")? {
            let mut stderr = String::new();
            if let Some(mut pipe) = child.stderr.take() {
                use tokio::io::AsyncReadExt;
                let mut buf = Vec::new();
                let _ = pipe.read_to_end(&mut buf).await;
                stderr = String::from_utf8_lossy(&buf).into_owned();
            }
            anyhow::bail!("PersonaPlex server exited during startup with {status}: {stderr}");
        }
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return Ok(());
        }
        if tokio::time::Instant::now() > deadline {
            let _ = child.start_kill();
            anyhow::bail!("PersonaPlex server health check timed out at {base_url}");
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

// ---------------------------------------------------------------------------
// Session layer: a single active full-duplex voice session (v1).
// ---------------------------------------------------------------------------

/// One realtime voice conversation bound to a running [`VoiceServer`].
#[derive(Clone)]
pub struct VoiceSession {
    pub id: String,
    /// System-prompt-like description of the persona/character to speak as.
    pub persona_text: String,
    /// Optional reference audio clip used to condition the voice.
    pub voice_prompt: Option<PathBuf>,
    /// Handle to the underlying `moshi.server` process backing this session.
    pub server: Arc<Mutex<VoiceServer>>,
}

impl VoiceSession {
    /// Base HTTP URL of the backing server (`http://127.0.0.1:{port}`).
    pub async fn base_url(&self) -> String {
        self.server.lock().await.base_url.clone()
    }

    /// WebSocket URL clients should be proxied to for this session.
    pub async fn proxy_url(&self) -> String {
        self.server.lock().await.proxy_url()
    }
}

/// Tracks at most one active realtime voice session (v1 limitation).
pub struct SessionManager {
    session: Mutex<Option<VoiceSession>>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            session: Mutex::new(None),
        }
    }

    /// Start a new PersonaPlex server and session. Fails if a session is
    /// already active — end it first.
    pub async fn create_session(
        &self,
        python: &Path,
        model_path: Option<&Path>,
        persona_text: String,
        voice_prompt: Option<PathBuf>,
        hf_token: Option<String>,
        profile: Option<&crate::model_settings::VoiceProfile>,
    ) -> anyhow::Result<VoiceSession> {
        let mut guard = self.session.lock().await;
        anyhow::ensure!(
            guard.is_none(),
            "a realtime voice session is already active; end it before starting another"
        );
        let server = VoiceServer::start(
            python,
            model_path,
            VoiceLaunchOptions {
                persona_text: Some(persona_text.clone()),
                voice_prompt: voice_prompt.clone(),
                voice_id: None,
                hf_token,
                quantization: None,
                extra_args: Vec::new(),
            }
            .with_profile(profile),
        )
        .await?;
        let session = VoiceSession {
            id: Uuid::new_v4().to_string(),
            persona_text,
            voice_prompt,
            server: Arc::new(Mutex::new(server)),
        };
        *guard = Some(session.clone());
        Ok(session)
    }

    pub async fn get_session(&self, id: &str) -> Option<VoiceSession> {
        self.session
            .lock()
            .await
            .as_ref()
            .filter(|session| session.id == id)
            .cloned()
    }

    pub async fn active_session(&self) -> Option<VoiceSession> {
        self.session.lock().await.clone()
    }

    /// Stop the backing server and clear the active session.
    pub async fn end_session(&self, id: &str) -> anyhow::Result<()> {
        let mut guard = self.session.lock().await;
        let session = match guard.as_ref() {
            Some(session) if session.id == id => session.clone(),
            Some(_) => anyhow::bail!("session id does not match the active realtime voice session"),
            None => anyhow::bail!("no active realtime voice session"),
        };
        // Keep the session reachable until shutdown succeeds. Clearing it
        // first would orphan a server when stop reports an error, leaving no
        // handle for status reporting or a retry.
        session.server.lock().await.stop().await?;
        guard.take();
        Ok(())
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_snapshot(root: &Path, repo_id: &str) -> PathBuf {
        let dir = root.join(repo_id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"model_type":"moshi"}"#).unwrap();
        std::fs::write(dir.join("model.safetensors"), b"weights").unwrap();
        dir
    }

    #[test]
    fn model_ids_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let root = models_root(dir.path());
        write_snapshot(&root, "kyutai/moshika-pytorch-bf16");

        let id = model_id_for_repo("kyutai/moshika-pytorch-bf16").unwrap();
        assert_eq!(id, "personaplex:kyutai/moshika-pytorch-bf16");

        let resolved = path_for_model_id(dir.path(), &id).unwrap();
        assert_eq!(resolved, root.join("kyutai/moshika-pytorch-bf16"));

        let models = list_models(dir.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].engine, ENGINE);
        assert!(models[0].capabilities.streaming);
        assert_eq!(
            models[0].capabilities.audio_input.as_deref(),
            Some("native")
        );
    }

    #[test]
    fn rejects_ids_from_other_engines() {
        let dir = tempfile::tempdir().unwrap();
        assert!(path_for_model_id(dir.path(), "streaming-asr:kyutai/moshika").is_err());
        assert!(path_for_model_id(dir.path(), "kyutai/moshika").is_err());
    }

    #[test]
    fn rejects_path_traversal_in_model_id() {
        let dir = tempfile::tempdir().unwrap();
        assert!(path_for_model_id(dir.path(), "personaplex:/abs/path").is_err());
        assert!(path_for_model_id(dir.path(), "personaplex:owner/name/extra").is_err());
        assert!(model_id_for_repo("just-a-name").is_err());
        assert!(model_id_for_repo("owner/name/extra").is_err());
    }

    #[test]
    fn rejects_invalid_download_filenames() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            download_destination(dir.path(), "kyutai/moshika", "../escape.safetensors").is_err()
        );
        assert!(download_destination(dir.path(), "kyutai/moshika", "/abs.safetensors").is_err());
        assert!(download_destination(dir.path(), "kyutai/moshika", "config.json").is_ok());
        assert!(download_destination(dir.path(), "kyutai/moshika", "model.safetensors").is_ok());
    }

    #[test]
    fn missing_model_id_does_not_resolve() {
        let dir = tempfile::tempdir().unwrap();
        assert!(path_for_model_id(dir.path(), "personaplex:kyutai/does-not-exist").is_err());
        assert!(
            resolve_model_path(dir.path(), Some("personaplex:kyutai/does-not-exist")).is_none()
        );
    }

    #[test]
    fn resolve_model_path_falls_back_to_first_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let root = models_root(dir.path());
        let snapshot = write_snapshot(&root, "kyutai/moshiko-pytorch-bf16");
        assert_eq!(resolve_model_path(dir.path(), None), Some(snapshot));
    }

    #[test]
    fn snapshot_dir_yields_its_repo_id() {
        let dir = tempfile::tempdir().unwrap();
        let root = models_root(dir.path());
        let snapshot = write_snapshot(&root, "nvidia/personaplex-7b-v1");
        assert_eq!(
            repo_id_from_snapshot_dir(&snapshot).as_deref(),
            Some("nvidia/personaplex-7b-v1")
        );
        assert!(repo_id_from_snapshot_dir(Path::new("/")).is_none());
    }

    #[test]
    fn snapshot_weights_prefer_prequantized() {
        let dir = tempfile::tempdir().unwrap();
        let root = models_root(dir.path());
        let snapshot = write_snapshot(&root, "nvidia/personaplex-7b-v1");
        assert_eq!(
            snapshot_lm_weight(&snapshot, 4),
            Some(snapshot.join("model.safetensors"))
        );

        std::fs::write(snapshot.join("model.q4.safetensors"), b"quantized").unwrap();
        assert_eq!(
            snapshot_lm_weight(&snapshot, 4),
            Some(snapshot.join("model.q4.safetensors"))
        );
        // A different bit width must not pick up the q4 file.
        assert_eq!(
            snapshot_lm_weight(&snapshot, 8),
            Some(snapshot.join("model.safetensors"))
        );

        assert!(snapshot_mimi_weight(&snapshot).is_none());
        let mimi = snapshot.join("tokenizer-e351c8d8-checkpoint125.safetensors");
        std::fs::write(&mimi, b"mimi").unwrap();
        assert_eq!(snapshot_mimi_weight(&snapshot), Some(mimi));
    }

    #[test]
    fn voice_prompt_dir_is_absent_without_an_archive() {
        let dir = tempfile::tempdir().unwrap();
        let root = models_root(dir.path());
        let snapshot = write_snapshot(&root, "nvidia/personaplex-7b-v1");
        assert!(snapshot_voice_prompt_dir(&snapshot).is_none());

        let voices = snapshot.join("voices");
        std::fs::create_dir_all(&voices).unwrap();
        assert_eq!(snapshot_voice_prompt_dir(&snapshot), Some(voices));
    }

    #[test]
    fn python_appears_runnable_rejects_missing_interpreter() {
        assert!(!python_appears_runnable(Path::new("/nonexistent/python")));
    }

    #[test]
    fn realtime_voice_available_requires_python() {
        assert!(!realtime_voice_available(None, None));
        assert!(!realtime_voice_available(
            Some(Path::new("/nonexistent/python")),
            None
        ));
    }

    #[test]
    fn builds_moshi_proxy_ws_url() {
        assert_eq!(proxy_ws_url(8998), "ws://127.0.0.1:8998/api/chat");
    }

    #[test]
    fn protocol_tags_match_moshi_wire_format() {
        assert_eq!(protocol::HANDSHAKE, 0x00);
        assert_eq!(protocol::AUDIO, 0x01);
        assert_eq!(protocol::TEXT, 0x02);
    }
}

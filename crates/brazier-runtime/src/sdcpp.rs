//! stable-diffusion.cpp binary discovery, managed installation, model store, and
//! spawn-per-request image/video generation jobs.
//!
//! Mirrors the managed-install conventions from `llama.rs` and the CLI
//! spawn/timeout conventions from `whisper.rs`. Unlike llama-server (a
//! long-lived HTTP server) and whisper-cli (a quick batch transcription),
//! `sd-cli` runs one full diffusion job per invocation, so jobs are
//! serialized behind a process-global lock to protect the GPU.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU32, Ordering as AtomicOrdering},
    },
    time::{Duration, Instant},
};

use anyhow::Context;
use flate2::read::GzDecoder;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tar::Archive;
use tokio::{io::AsyncWriteExt, process::Command, sync::Mutex as AsyncMutex, sync::Notify};

use crate::{
    model_settings::DiffusionProfile,
    models_store,
    progress::{ProgressCallback, ProgressEvent},
    runtime_settings::RuntimeTarget,
    types::{ModelCapabilities, ModelDescriptor},
};

pub const ENGINE: &str = "stable-diffusion.cpp";

/// Reviewed sd.cpp compatibility boundary.
///
/// Upstream intentionally ships an unstable CLI, so managed installs stay on
/// this release until Brazier is updated and tested against a newer one. Users
/// can still build another revision explicitly under Manage → Runtimes.
const PINNED_RELEASE_TAG: &str = "master-796-2d0385b";
pub const PINNED_SOURCE_REVISION: &str = "2d0385ba85af358f7115dda608a63eafd9de7ffd";
const GITHUB_API: &str =
    "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/tags/master-796-2d0385b";
const USER_AGENT: &str = "brazier-sdcpp-manager";

const IMAGE_TIMEOUT: Duration = Duration::from_secs(3600);
const AMD_APU_VIDEO_WIDTH: u32 = 512;
const AMD_APU_VIDEO_HEIGHT: u32 = 320;
const AMD_APU_VIDEO_FRAMES: u32 = 17;
/// Floor for a video job, covering model load plus a short clip.
const VIDEO_TIMEOUT_BASE: Duration = Duration::from_secs(1800);
/// Added per frame-step, so long clips are not cut off mid-render.
///
/// Sized for a machine with no usable GPU: an integrated APU renders a frame
/// step in seconds, not milliseconds, and the previous allowance killed those
/// jobs hours before they would have finished. Being generous costs nothing now
/// that a job can be stopped from the interface — waiting too long is a click
/// away from over, while cutting one off throws away everything it rendered.
const VIDEO_TIMEOUT_PER_FRAME_STEP: Duration = Duration::from_secs(30);
/// Ceiling, so a wedged sd-cli is still eventually reclaimed.
const VIDEO_TIMEOUT_MAX: Duration = Duration::from_secs(24 * 3600);

/// Budget for one video job.
///
/// Render time scales with frames × steps, and a Wan clip at a useful length
/// runs well past the half hour a fixed cap allowed, so the deadline grows
/// with the work requested rather than failing a job that was progressing.
fn video_timeout(steps: u32, frames: u32) -> Duration {
    let work = u64::from(steps.max(1)) * u64::from(frames.max(1));
    VIDEO_TIMEOUT_BASE
        .saturating_add(
            VIDEO_TIMEOUT_PER_FRAME_STEP.saturating_mul(work.min(u32::MAX as u64) as u32),
        )
        .min(VIDEO_TIMEOUT_MAX)
}

/// Apply a configured override to a computed budget.
///
/// No amount of tuning suits every machine, so Engine configuration can name a
/// flat limit; zero or absent keeps the size-derived one.
fn effective_timeout(computed: Duration, override_secs: Option<u32>) -> Duration {
    match override_secs {
        Some(secs) if secs > 0 => Duration::from_secs(u64::from(secs)),
        _ => computed,
    }
}

// ---------------------------------------------------------------------------
// Managed install
// ---------------------------------------------------------------------------

pub fn managed_engine_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("engines").join("stable-diffusion.cpp")
}

pub fn binary_name() -> &'static str {
    if cfg!(windows) {
        "sd-cli.exe"
    } else {
        "sd-cli"
    }
}

pub fn managed_binary_path(data_dir: &Path) -> PathBuf {
    managed_engine_dir(data_dir).join("bin").join(binary_name())
}

pub fn managed_binary_path_for_target(data_dir: &Path, target: RuntimeTarget) -> PathBuf {
    if matches!(
        target,
        RuntimeTarget::Auto | RuntimeTarget::Cpu | RuntimeTarget::Metal
    ) {
        return managed_binary_path(data_dir);
    }
    managed_engine_dir(data_dir)
        .join(target.as_str())
        .join("bin")
        .join(binary_name())
}

/// Root directory where managed install metadata (VERSION) lives for a target.
pub fn managed_install_root(data_dir: &Path, target: RuntimeTarget) -> PathBuf {
    let engine_dir = managed_engine_dir(data_dir);
    match target {
        RuntimeTarget::Auto | RuntimeTarget::Cpu | RuntimeTarget::Metal => engine_dir,
        _ => engine_dir.join(target.as_str()),
    }
}

pub fn managed_is_installed(data_dir: &Path, target: RuntimeTarget) -> bool {
    managed_binary_path_for_target(data_dir, target).is_file()
}

pub fn managed_installed_version(data_dir: &Path, target: RuntimeTarget) -> Option<String> {
    std::fs::read_to_string(managed_install_root(data_dir, target).join("VERSION"))
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// Coarse OS/arch tag used to select a GitHub release asset. Only platforms
/// with an upstream prebuilt CLI package are supported here; everything else
/// falls back to a source build.
pub fn platform_asset_tag() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Some("macos-arm64"),
        ("linux", "x86_64") => Some("linux-x64"),
        ("windows", "x86_64") => Some("windows-x64"),
        _ => None,
    }
}

/// Restrict an acceleration target to what this platform's managed releases
/// actually publish, falling back to the platform's baseline flavor.
fn constrain_target_to_platform(target: RuntimeTarget) -> RuntimeTarget {
    match (std::env::consts::OS, target) {
        ("macos", RuntimeTarget::Cuda | RuntimeTarget::Rocm | RuntimeTarget::Vulkan) => {
            RuntimeTarget::Metal
        }
        ("linux", RuntimeTarget::Cuda | RuntimeTarget::Metal) => RuntimeTarget::Cpu,
        ("windows", RuntimeTarget::Rocm | RuntimeTarget::Metal) => RuntimeTarget::Cpu,
        _ => target,
    }
}

/// Choose the best prebuilt release asset for this host/target combination.
///
/// - macOS arm64: asset name contains `Darwin` and `arm64`.
/// - Linux x64 CPU: contains `Linux` + `x86_64`, without `vulkan`/`rocm`/`cuda`.
/// - Linux x64 Vulkan: contains `vulkan`.
/// - Linux x64 ROCm: contains `rocm`.
/// - Windows CPU: `win-cpu-x64`, or `win` + `cpu` + `x64`.
/// - Windows CUDA: `win-cuda`, or `cuda12` + `win`.
/// - Windows Vulkan: `win-vulkan`, or `win` + `vulkan`.
///
/// `cudart` redistributable packages are always skipped.
pub fn select_release_asset_for_target<'a>(
    asset_names: impl IntoIterator<Item = &'a str>,
    platform_tag: &str,
    target: RuntimeTarget,
) -> Option<&'a str> {
    let mut candidates: Vec<&str> = asset_names
        .into_iter()
        .filter(|name| {
            let lower = name.to_ascii_lowercase();
            if lower.contains("cudart") {
                return false;
            }
            if !(lower.ends_with(".zip") || lower.ends_with(".tar.gz") || lower.ends_with(".tgz")) {
                return false;
            }
            match platform_tag {
                "macos-arm64" => lower.contains("darwin") && lower.contains("arm64"),
                "linux-x64" => {
                    let base = lower.contains("linux") && lower.contains("x86_64");
                    if !base {
                        return false;
                    }
                    match target {
                        RuntimeTarget::Vulkan => lower.contains("vulkan"),
                        RuntimeTarget::Rocm => lower.contains("rocm"),
                        _ => {
                            !lower.contains("vulkan")
                                && !lower.contains("rocm")
                                && !lower.contains("cuda")
                        }
                    }
                }
                "windows-x64" => match target {
                    RuntimeTarget::Cuda => {
                        lower.contains("win-cuda")
                            || (lower.contains("cuda12") && lower.contains("win"))
                    }
                    RuntimeTarget::Vulkan => {
                        lower.contains("win-vulkan")
                            || (lower.contains("win") && lower.contains("vulkan"))
                    }
                    _ => {
                        lower.contains("win-cpu-x64")
                            || (lower.contains("win")
                                && lower.contains("cpu")
                                && lower.contains("x64"))
                    }
                },
                _ => false,
            }
        })
        .collect();
    // Prefer the shortest matching name (fewest extra qualifiers).
    candidates.sort_by_key(|name| name.len());
    candidates.first().copied()
}

#[derive(Debug, Clone)]
pub struct ReleaseAsset {
    pub name: String,
    pub browser_download_url: String,
}

/// Supported release tag from cache, without waiting on GitHub.
///
/// Status views call this on every open, so a stale-but-instant answer beats
/// a blocking lookup; the refresh it triggers lands in time for the next one.
pub fn cached_release_tag(client: &reqwest::Client) -> crate::github_releases::CachedRelease {
    crate::github_releases::cached_or_refresh(client, GITHUB_API, USER_AGENT)
}

/// Resolve the preferred managed binary download for this platform/target.
pub async fn resolve_managed_release(
    client: &reqwest::Client,
    target: RuntimeTarget,
) -> anyhow::Result<(String, ReleaseAsset)> {
    let platform = platform_asset_tag()
        .context("managed stable-diffusion.cpp binaries are not available for this platform")?;
    let release = crate::github_releases::latest_release(client, GITHUB_API, USER_AGENT).await?;
    let names: Vec<String> = release.asset_names().map(str::to_owned).collect();
    let selected =
        select_release_asset_for_target(names.iter().map(String::as_str), platform, target)
            .context("no matching stable-diffusion.cpp release asset for this platform/target")?
            .to_owned();
    let asset = release
        .asset(&selected)
        .context("selected asset missing from release")?;
    Ok((
        release.tag_name.clone(),
        ReleaseAsset {
            name: asset.name.clone(),
            browser_download_url: asset.browser_download_url.clone(),
        },
    ))
}

fn should_keep_extracted_file(file_name: &str) -> bool {
    let lower = file_name.to_ascii_lowercase();
    let is_lib = lower.contains(".so") || lower.ends_with(".dll") || lower.ends_with(".dylib");
    let is_cli = file_name == binary_name();
    is_lib
        || is_cli
        || lower.starts_with("sd")
        || lower.starts_with("ggml")
        || lower.starts_with("stable-diffusion")
}

/// Extract release members into `bin_dir`, flattening any top-level prefix directory.
fn extract_release_archive(archive_path: &Path, bin_dir: &Path) -> anyhow::Result<()> {
    let file = std::fs::File::open(archive_path).context("open archive")?;
    let name = archive_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);
        let mut found_cli = false;
        for entry in archive.entries().context("read tar entries")? {
            let mut entry = entry.context("tar entry")?;
            let path = entry.path().context("tar entry path")?.into_owned();
            if entry.header().entry_type().is_dir() {
                continue;
            }
            let file_name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            if file_name.is_empty() || !should_keep_extracted_file(file_name) {
                continue;
            }
            let dest = bin_dir.join(file_name);
            entry
                .unpack(&dest)
                .with_context(|| format!("unpack {file_name}"))?;
            if file_name == binary_name() {
                found_cli = true;
            }
        }
        anyhow::ensure!(found_cli, "sd-cli binary not found in archive");
        return Ok(());
    }
    if name.ends_with(".zip") {
        let mut archive = zip::ZipArchive::new(file).context("read zip archive")?;
        let mut found_cli = false;
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index).context("read zip entry")?;
            if entry.is_dir() {
                continue;
            }
            let Some(path) = entry.enclosed_name() else {
                continue;
            };
            let file_name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            if !should_keep_extracted_file(file_name) {
                continue;
            }
            let mut destination =
                std::fs::File::create(bin_dir.join(file_name)).context("create zip output")?;
            std::io::copy(&mut entry, &mut destination).context("extract zip entry")?;
            found_cli |= file_name == binary_name();
        }
        anyhow::ensure!(found_cli, "sd-cli binary not found in archive");
        return Ok(());
    }
    anyhow::bail!("unsupported archive format: {name}");
}

/// Download and extract a managed sd-cli binary into the data directory.
pub async fn install_managed_binary_with_progress(
    client: &reqwest::Client,
    data_dir: &Path,
    target: RuntimeTarget,
    mut progress: ProgressCallback,
) -> anyhow::Result<PathBuf> {
    progress(ProgressEvent::phase(
        "resolve",
        format!("Looking up supported stable-diffusion.cpp release {PINNED_RELEASE_TAG}"),
    ));
    let (tag, asset) = resolve_managed_release(client, target).await?;
    tracing::info!(%tag, asset = %asset.name, "downloading managed stable-diffusion.cpp binary");
    progress(ProgressEvent::phase(
        "download",
        format!("Downloading {tag} ({})", asset.name),
    ));

    let response = client
        .get(&asset.browser_download_url)
        .header("user-agent", USER_AGENT)
        .send()
        .await
        .context("download stable-diffusion.cpp release")?
        .error_for_status()
        .context("stable-diffusion.cpp release download failed")?;
    let total = response.content_length();
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    let mut written = 0_u64;
    let mut last_emit = 0_u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read stable-diffusion.cpp release body")?;
        written += chunk.len() as u64;
        bytes.extend_from_slice(&chunk);
        if written.saturating_sub(last_emit) >= 256 * 1024 || total == Some(written) {
            progress(ProgressEvent::download(written, total));
            last_emit = written;
        }
    }
    progress(ProgressEvent::download(written, total.or(Some(written))));

    let binary = managed_binary_path_for_target(data_dir, target);
    let bin_dir = binary
        .parent()
        .context("managed binary path has no parent")?
        .to_path_buf();
    let engine_dir = bin_dir
        .parent()
        .context("managed binary directory has no parent")?
        .to_path_buf();
    if bin_dir.exists() {
        tokio::fs::remove_dir_all(&bin_dir)
            .await
            .context("clear previous managed sdcpp install")?;
    }
    tokio::fs::create_dir_all(&bin_dir)
        .await
        .context("create sdcpp engine bin directory")?;
    let archive_path = engine_dir.join(&asset.name);
    {
        let mut file = tokio::fs::File::create(&archive_path)
            .await
            .context("write release archive")?;
        file.write_all(&bytes).await?;
        file.flush().await?;
    }

    progress(ProgressEvent::phase(
        "extract",
        "Extracting sd-cli and shared libraries",
    ));
    extract_release_archive(&archive_path, &bin_dir)
        .context("extract stable-diffusion.cpp release")?;
    anyhow::ensure!(
        binary.is_file(),
        "archive did not contain {}; extracted into {}",
        binary_name(),
        bin_dir.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for entry in std::fs::read_dir(&bin_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let mut permissions = std::fs::metadata(&path)?.permissions();
                permissions.set_mode(0o755);
                std::fs::set_permissions(&path, permissions)?;
            }
        }
    }
    let _ = tokio::fs::remove_file(&archive_path).await;
    tokio::fs::write(engine_dir.join("VERSION"), format!("{tag}\n")).await?;
    progress(ProgressEvent::done(serde_json::json!({
        "binary": binary.display().to_string(),
        "tag": tag,
        "status": "ready"
    })));
    Ok(binary)
}

/// Ensure an sd-cli binary is available, installing a managed build if needed.
pub async fn ensure_binary_with_progress(
    client: &reqwest::Client,
    data_dir: &Path,
    target: RuntimeTarget,
    force: bool,
    mut progress: ProgressCallback,
) -> anyhow::Result<PathBuf> {
    let target = if target == RuntimeTarget::Auto {
        constrain_target_to_platform(crate::hardware::detect().recommended_target)
    } else {
        target
    };
    if force {
        return install_managed_binary_with_progress(client, data_dir, target, progress).await;
    }
    let managed = managed_binary_path_for_target(data_dir, target);
    progress(ProgressEvent::phase(
        "discover",
        "Looking for an existing sd-cli binary",
    ));
    let discovered = if managed.is_file() {
        Some(managed.clone())
    } else {
        resolve_binary(data_dir, None)
    };
    if let Some(path) = discovered {
        if binary_appears_runnable(&path) {
            progress(ProgressEvent::done(serde_json::json!({
                "binary": path.display().to_string(),
                "status": "ready",
                "source": "discovered"
            })));
            return Ok(path);
        }
        if path != managed {
            tracing::warn!(
                binary = %path.display(),
                "discovered sd-cli failed a smoke test; trying managed install"
            );
        }
    }
    install_managed_binary_with_progress(client, data_dir, target, progress).await
}

/// Candidate paths where a user- or app-installed sd-cli might live.
pub fn discovery_candidates(data_dir: &Path, path_env: Option<&str>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for target in [
        RuntimeTarget::Cpu,
        RuntimeTarget::Cuda,
        RuntimeTarget::Rocm,
        RuntimeTarget::Vulkan,
        RuntimeTarget::Metal,
    ] {
        let managed = managed_binary_path_for_target(data_dir, target);
        if !candidates.contains(&managed) {
            candidates.push(managed);
        }
    }
    for (_, record) in crate::builds::list_builds(data_dir, ENGINE) {
        candidates.push(PathBuf::from(record.binary));
    }
    if let Some(path_env) = path_env {
        for dir in std::env::split_paths(path_env) {
            candidates.push(dir.join(binary_name()));
        }
    }
    for dir in [
        "/usr/local/bin",
        "/usr/bin",
        "/opt/homebrew/bin",
        "/opt/local/bin",
    ] {
        candidates.push(PathBuf::from(dir).join(binary_name()));
    }
    candidates
}

/// Resolve an activated, discovered, or PATH-installed sd-cli binary.
pub fn resolve_binary(data_dir: &Path, override_path: Option<&str>) -> Option<PathBuf> {
    if let Some(path) = override_path
        .map(PathBuf::from)
        .filter(|path| path.is_file())
    {
        return Some(path);
    }
    discovery_candidates(
        data_dir,
        std::env::var_os("PATH").as_deref().and_then(|p| p.to_str()),
    )
    .into_iter()
    .find(|path| path.is_file())
}

fn prepend_library_path(command: &mut Command, dir: &Path) {
    #[cfg(unix)]
    {
        let key = if cfg!(target_os = "macos") {
            "DYLD_LIBRARY_PATH"
        } else {
            "LD_LIBRARY_PATH"
        };
        let mut paths = vec![dir.to_path_buf()];
        if let Some(existing) = std::env::var_os(key) {
            paths.extend(std::env::split_paths(&existing));
        }
        if let Ok(joined) = std::env::join_paths(paths) {
            command.env(key, joined);
        }
    }
    #[cfg(windows)]
    {
        let mut paths = vec![dir.to_path_buf()];
        if let Some(existing) = std::env::var_os("PATH") {
            paths.extend(std::env::split_paths(&existing));
        }
        if let Ok(joined) = std::env::join_paths(paths) {
            command.env("PATH", joined);
        }
    }
}

/// Best-effort check that a binary can start (shared libraries resolve).
pub fn binary_appears_runnable(binary: &Path) -> bool {
    let mut command = std::process::Command::new(binary);
    command
        .arg("-h")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(dir) = binary.parent() {
        #[cfg(unix)]
        {
            let key = if cfg!(target_os = "macos") {
                "DYLD_LIBRARY_PATH"
            } else {
                "LD_LIBRARY_PATH"
            };
            let mut paths = vec![dir.to_path_buf()];
            if let Some(existing) = std::env::var_os(key) {
                paths.extend(std::env::split_paths(&existing));
            }
            if let Ok(joined) = std::env::join_paths(paths) {
                command.env(key, joined);
            }
        }
    }
    matches!(command.status(), Ok(status) if status.success() || status.code().is_some())
}

// ---------------------------------------------------------------------------
// Model store
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Image,
    Video,
}

impl Modality {
    fn prefix(self) -> &'static str {
        match self {
            Self::Image => "sdcpp-image",
            Self::Video => "sdcpp-video",
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
        }
    }

    fn root(self, data_dir: &Path) -> PathBuf {
        match self {
            Self::Image => image_root(data_dir),
            Self::Video => video_root(data_dir),
        }
    }
}

/// Root for image models: `<data>/models/sdcpp/image`.
pub fn image_root(data_dir: &Path) -> PathBuf {
    data_dir.join("models").join("sdcpp").join("image")
}

/// Root for video models: `<data>/models/sdcpp/video`.
pub fn video_root(data_dir: &Path) -> PathBuf {
    data_dir.join("models").join("sdcpp").join("video")
}

/// Sidecar manifest describing how to invoke sd-cli for one model.
///
/// Placed as `manifest.json` inside the model's directory, either next to a
/// multi-component model's constituent weight files (`args`) or alongside a
/// single checkpoint file (`single_file`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdcppManifest {
    pub modality: Modality,
    /// Maps an sd-cli flag name (without leading `--`, e.g. `diffusion-model`,
    /// `vae`, `t5xxl`, `clip_l`, `llm`, `clip_vision`) to a path relative to
    /// the manifest's directory.
    #[serde(default)]
    pub args: BTreeMap<String, String>,
    /// Pinned hashes for direct-source components. This makes a corrected
    /// curated download distinguishable from an older mirror with the same
    /// filename and command-line flag.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub component_sources: BTreeMap<String, String>,
    /// Relative path (from the manifest's directory) to a single
    /// `.safetensors`/`.gguf` checkpoint, used with sd-cli's `-m` flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub single_file: Option<String>,
    /// Whether this model can start from a supplied frame (`-i`), which is
    /// what image-to-video and image-to-image need. Text-to-video models
    /// ignore the flag or produce nonsense, so it is opt-in per model.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub supports_init_image: bool,
}

/// Whether a model accepts an init image, from its installed manifest.
pub fn supports_init_image(data_dir: &Path, model_id: &str) -> bool {
    resolve_manifest(data_dir, model_id)
        .map(|(_, manifest)| manifest.supports_init_image)
        .unwrap_or(false)
}

fn manifest_file_path(dir: &Path) -> PathBuf {
    dir.join("manifest.json")
}

/// Load and parse the manifest for a model directory.
pub fn load_manifest(dir: &Path) -> anyhow::Result<SdcppManifest> {
    let path = manifest_file_path(dir);
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn validate_key(key: &str) -> anyhow::Result<()> {
    anyhow::ensure!(!key.is_empty(), "empty stable-diffusion.cpp model key");
    anyhow::ensure!(
        !key.split('/')
            .any(|part| part.is_empty() || part == "." || part == ".."),
        "invalid stable-diffusion.cpp model key"
    );
    Ok(())
}

fn validate_component_filename(filename: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !filename.is_empty() && filename.len() <= 260,
        "invalid filename"
    );
    anyhow::ensure!(
        !filename.starts_with('/') && !filename.contains('\\'),
        "filename must be a relative path"
    );
    anyhow::ensure!(
        !filename
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == ".."),
        "filename must not contain empty or parent path segments"
    );
    Ok(())
}

/// Stable model id: `sdcpp-image:{key}` or `sdcpp-video:{key}`.
pub fn model_id(modality: Modality, key: &str) -> String {
    format!("{}:{key}", modality.prefix())
}

/// Directory a model with this key installs into, after validating the key.
pub fn model_dir_for_key(
    data_dir: &Path,
    modality: Modality,
    key: &str,
) -> anyhow::Result<PathBuf> {
    validate_key(key)?;
    Ok(modality.root(data_dir).join(key))
}

/// Destination for one component file inside a model directory.
pub fn component_destination(
    data_dir: &Path,
    modality: Modality,
    key: &str,
    file_name: &str,
) -> anyhow::Result<PathBuf> {
    validate_component_filename(file_name)?;
    Ok(model_dir_for_key(data_dir, modality, key)?.join(file_name))
}

/// Write the sidecar manifest that binds downloaded files to sd-cli flags.
pub async fn write_manifest(dir: &Path, manifest: &SdcppManifest) -> anyhow::Result<()> {
    tokio::fs::create_dir_all(dir)
        .await
        .with_context(|| format!("create {}", dir.display()))?;
    let payload = serde_json::to_vec_pretty(manifest).context("serialize sdcpp manifest")?;
    let path = manifest_file_path(dir);
    tokio::fs::write(&path, payload)
        .await
        .with_context(|| format!("write {}", path.display()))
}

/// Alias of [`model_id`] that reads naturally from the catalog.
pub fn model_id_for_key(modality: Modality, key: &str) -> String {
    model_id(modality, key)
}

/// Split a stable-diffusion.cpp model id into its modality and key.
pub fn parse_model_id(model_id: &str) -> anyhow::Result<(Modality, &str)> {
    if let Some(key) = model_id.strip_prefix("sdcpp-image:") {
        return Ok((Modality::Image, key));
    }
    if let Some(key) = model_id.strip_prefix("sdcpp-video:") {
        return Ok((Modality::Video, key));
    }
    anyhow::bail!("not a stable-diffusion.cpp model id: {model_id}")
}

/// Resolve a model directory and its parsed manifest from a model id.
fn resolve_manifest(data_dir: &Path, model_id: &str) -> anyhow::Result<(PathBuf, SdcppManifest)> {
    let (modality, key) = parse_model_id(model_id)?;
    validate_key(key)?;
    let dir = modality.root(data_dir).join(key);
    anyhow::ensure!(dir.is_dir(), "model not found: {model_id}");
    let manifest = load_manifest(&dir)?;
    anyhow::ensure!(
        manifest.modality == modality,
        "manifest modality does not match model id {model_id}"
    );
    Ok((dir, manifest))
}

/// Resolve a `sdcpp-image:…`/`sdcpp-video:…` model id to an on-disk path.
///
/// Returns the model's manifest directory for multi-component models, or the
/// single checkpoint file when the manifest declares `single_file`.
pub fn path_for_model_id(data_dir: &Path, model_id: &str) -> anyhow::Result<PathBuf> {
    let (dir, manifest) = resolve_manifest(data_dir, model_id)?;
    if let Some(single_file) = &manifest.single_file {
        validate_component_filename(single_file)?;
        let file = dir.join(single_file);
        anyhow::ensure!(
            file.is_file(),
            "manifest single_file missing: {}",
            file.display()
        );
        return Ok(file);
    }
    Ok(dir)
}

/// Destination path for one downloaded weight/component file.
pub fn download_destination(
    data_dir: &Path,
    modality: Modality,
    repo_id: &str,
    filename: &str,
) -> anyhow::Result<PathBuf> {
    models_store::validate_repo_id(repo_id)?;
    validate_component_filename(filename)?;
    Ok(modality.root(data_dir).join(repo_id).join(filename))
}

fn directory_size_bytes(dir: &Path) -> u64 {
    let mut total = 0_u64;
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            total += std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
        } else if path.is_dir() {
            total += directory_size_bytes(&path);
        }
    }
    total
}

fn capabilities_for(modality: Modality) -> ModelCapabilities {
    ModelCapabilities {
        input_modalities: vec!["text".into(), "image".into()],
        output_modalities: vec![modality.as_str().to_owned()],
        streaming: false,
        tools: false,
        reasoning: false,
        max_context_length: None,
        reasoning_modes: Vec::new(),
        harmony: false,
        audio_input: None,
    }
}

fn collect_manifests(
    modality: Modality,
    root: &Path,
    dir: &Path,
    models: &mut Vec<ModelDescriptor>,
) -> anyhow::Result<()> {
    let entries = std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if manifest_file_path(&path).is_file() {
            if let Ok(relative) = path.strip_prefix(root) {
                let key = relative
                    .components()
                    .map(|component| component.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                if key.is_empty() {
                    continue;
                }
                match load_manifest(&path) {
                    Ok(manifest) if manifest.modality == modality => {
                        models.push(ModelDescriptor {
                            id: model_id(modality, &key),
                            name: key,
                            engine: ENGINE.to_owned(),
                            capabilities: capabilities_for(modality),
                            size_bytes: Some(directory_size_bytes(&path)),
                            read_only: false,
                            library_label: None,
                        });
                    }
                    Ok(_) => {
                        tracing::warn!(
                            path = %path.display(),
                            "skipping stable-diffusion.cpp manifest with mismatched modality"
                        );
                    }
                    Err(error) => {
                        tracing::warn!(%error, path = %path.display(), "skipping invalid stable-diffusion.cpp manifest");
                    }
                }
            }
            // A model directory's own subdirectories are manifest-relative
            // component storage, not further models.
            continue;
        }
        collect_manifests(modality, root, &path, models)?;
    }
    Ok(())
}

/// List on-disk stable-diffusion.cpp image and video models.
pub fn list_models(data_dir: &Path) -> anyhow::Result<Vec<ModelDescriptor>> {
    let mut models = Vec::new();
    for modality in [Modality::Image, Modality::Video] {
        let root = modality.root(data_dir);
        if !root.is_dir() {
            continue;
        }
        collect_manifests(modality, &root, &root, &mut models)?;
    }
    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

// ---------------------------------------------------------------------------
// Job runners
// ---------------------------------------------------------------------------

/// Single-flight lock protecting the GPU: only one sd-cli job may run at a time.
static JOB_LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();

fn job_lock() -> &'static AsyncMutex<()> {
    JOB_LOCK.get_or_init(|| AsyncMutex::new(()))
}

/// Returned when another generation job is already running.
#[derive(Debug)]
pub struct BusyError;

impl std::fmt::Display for BusyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "stable-diffusion.cpp is busy running another generation job"
        )
    }
}

impl std::error::Error for BusyError {}

/// Returned when the user stopped a generation from the interface.
///
/// A distinct type because this is not a failure: a model that asked for the
/// picture needs to hear that the person decided against it, not that the
/// engine broke.
#[derive(Debug)]
pub struct CancelledError;

impl std::fmt::Display for CancelledError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "generation was stopped by the user")
    }
}

impl std::error::Error for CancelledError {}

/// Who asked for a generation, so the interface can say whose prompt it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GenerationOrigin {
    /// Typed by the person, in Generate mode.
    #[default]
    User,
    /// Requested by a model through the generate tools.
    Model,
}

/// What a running generation is doing, for the interface to show and stop.
///
/// A model-driven generation is otherwise opaque: it can run for hours on the
/// strength of a prompt the user never saw, so everything needed to judge it —
/// and to decide to stop it — is published here while it runs.
#[derive(Debug, Clone, Serialize)]
pub struct ActiveGeneration {
    pub id: String,
    pub modality: Modality,
    pub model_id: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    /// Blob the conditioning image came from, so the interface can show it.
    pub init_image_blob: Option<String>,
    pub origin: GenerationOrigin,
    /// How long it has been running, refreshed on every read.
    pub elapsed_secs: u64,
    /// When this job will be given up on, so a long render is not a mystery.
    pub timeout_secs: u64,
    /// Diffusion sampling progress reported by sd-cli.
    pub current_step: u32,
    pub total_steps: u32,
}

struct RunningJob {
    info: ActiveGeneration,
    started: Instant,
    cancel: Arc<AtomicBool>,
    notify: Arc<Notify>,
    current_step: Arc<AtomicU32>,
}

static RUNNING: OnceLock<Mutex<Option<RunningJob>>> = OnceLock::new();

fn running() -> &'static Mutex<Option<RunningJob>> {
    RUNNING.get_or_init(|| Mutex::new(None))
}

/// The generation in flight, if any, with its elapsed time brought up to date.
pub fn active_generation() -> Option<ActiveGeneration> {
    let guard = running().lock().expect("generation lock");
    guard.as_ref().map(|job| {
        let mut info = job.info.clone();
        info.elapsed_secs = job.started.elapsed().as_secs();
        info.current_step = job.current_step.load(AtomicOrdering::Relaxed);
        info
    })
}

/// Ask the running generation to stop. False when nothing is running.
pub fn cancel_active_generation() -> bool {
    let guard = running().lock().expect("generation lock");
    match guard.as_ref() {
        Some(job) => {
            job.cancel.store(true, AtomicOrdering::SeqCst);
            job.notify.notify_waiters();
            true
        }
        None => false,
    }
}

/// Registers a generation for the lifetime of the job and clears it on drop,
/// so a panic cannot leave the interface showing a job that is not running.
struct JobRegistration {
    cancel: Arc<AtomicBool>,
    notify: Arc<Notify>,
    current_step: Arc<AtomicU32>,
    total_steps: u32,
}

impl JobRegistration {
    fn open(info: ActiveGeneration) -> Self {
        let cancel = Arc::new(AtomicBool::new(false));
        let notify = Arc::new(Notify::new());
        let current_step = Arc::new(AtomicU32::new(info.current_step));
        let total_steps = info.total_steps;
        *running().lock().expect("generation lock") = Some(RunningJob {
            info,
            started: Instant::now(),
            cancel: Arc::clone(&cancel),
            notify: Arc::clone(&notify),
            current_step: Arc::clone(&current_step),
        });
        Self {
            cancel,
            notify,
            current_step,
            total_steps,
        }
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(AtomicOrdering::SeqCst)
    }
}

impl Drop for JobRegistration {
    fn drop(&mut self) {
        *running().lock().expect("generation lock") = None;
    }
}

/// Fallbacks for a job that names no size and belongs to a model that has not
/// been configured with one.
const FALLBACK_WIDTH: u32 = 512;
const FALLBACK_HEIGHT: u32 = 512;
const FALLBACK_STEPS: u32 = 20;
const FALLBACK_VIDEO_FRAMES: u32 = 16;

#[derive(Debug, Clone, Deserialize)]
pub struct GenerateImageRequest {
    pub prompt: String,
    pub model_id: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// Absent means "whatever this model is configured for", which is how a
    /// model that only works at one resolution gets used at it.
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
    /// Distilled guidance scale, used by Flux-family models instead of CFG.
    #[serde(default)]
    pub guidance: Option<f32>,
    /// Optional path to a local image to condition an img2img generation.
    #[serde(default)]
    pub init_image: Option<PathBuf>,
    /// Blob the init image came from, carried only so the interface can show
    /// what a running job was given.
    #[serde(default)]
    pub init_image_blob: Option<String>,
    /// Whether the person or a model asked for this.
    #[serde(default)]
    pub origin: GenerationOrigin,
    /// Flat timeout in seconds from Engine configuration; 0 or absent uses the
    /// size-derived budget.
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
    /// Distilled guidance scale, used by Flux-family models instead of CFG.
    #[serde(default)]
    pub guidance: Option<f32>,
    /// Optional path to a local image to condition an img2vid generation.
    #[serde(default)]
    pub init_image: Option<PathBuf>,
    /// Blob the init image came from, carried only so the interface can show
    /// what a running job was given.
    #[serde(default)]
    pub init_image_blob: Option<String>,
    /// Whether the person or a model asked for this.
    #[serde(default)]
    pub origin: GenerationOrigin,
    /// Flat timeout in seconds from Engine configuration; 0 or absent uses the
    /// size-derived budget.
    #[serde(default)]
    pub timeout_secs: Option<u32>,
    #[serde(default)]
    pub video_frames: Option<u32>,
    /// Playback rate written into the clip; sd-cli defaults to 24.
    #[serde(default)]
    pub fps: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GenerateResult {
    pub output_path: PathBuf,
    pub metadata: serde_json::Value,
}

async fn job_output_dir(data_dir: &Path) -> anyhow::Result<PathBuf> {
    let dir = data_dir
        .join("tmp")
        .join("sdcpp")
        .join(uuid::Uuid::new_v4().simple().to_string());
    tokio::fs::create_dir_all(&dir)
        .await
        .context("create sdcpp job output directory")?;
    Ok(dir)
}

fn apply_manifest_args(
    command: &mut Command,
    model_dir: &Path,
    manifest: &SdcppManifest,
) -> anyhow::Result<()> {
    if let Some(single_file) = &manifest.single_file {
        validate_component_filename(single_file)?;
        let path = model_dir.join(single_file);
        anyhow::ensure!(
            path.is_file(),
            "manifest single_file missing: {}",
            path.display()
        );
        command.arg("-m").arg(path);
    }
    for (flag, relative) in &manifest.args {
        validate_component_filename(relative)?;
        let path = model_dir.join(relative);
        anyhow::ensure!(
            path.exists(),
            "manifest arg `--{flag}` missing file: {}",
            path.display()
        );
        command.arg(format!("--{flag}")).arg(path);
    }
    Ok(())
}

/// Combine a request's negative prompt with the model's standing one.
///
/// A model configured to always avoid something should keep avoiding it when a
/// job names something else to avoid, so the two are joined rather than one
/// replacing the other.
fn merge_negative_prompt(
    request: Option<&str>,
    profile: Option<&DiffusionProfile>,
) -> Option<String> {
    let configured = profile
        .and_then(|profile| profile.negative_prompt.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let requested = request.map(str::trim).filter(|value| !value.is_empty());
    match (configured, requested) {
        (Some(configured), Some(requested)) => Some(format!("{requested}, {configured}")),
        (Some(only), None) | (None, Some(only)) => Some(only.to_owned()),
        (None, None) => None,
    }
}

/// Conservative sd.cpp defaults for a Vulkan AMD APU.
///
/// RADV exposes an APU as a unified-memory Vulkan device, but sd.cpp otherwise
/// places every model component on it. Large text encoders can then make the
/// compute context reset with `VK_ERROR_DEVICE_LOST`. Keep those encoders in
/// host memory and reduce peak attention/VAE allocations. A model profile can
/// explicitly override every value here.
fn with_amd_apu_vulkan_defaults(
    profile: Option<&DiffusionProfile>,
    enabled: bool,
    modality: Modality,
) -> Option<DiffusionProfile> {
    if !enabled {
        return profile.cloned();
    }

    let mut profile = profile.cloned().unwrap_or_default();
    let (width, height) = match modality {
        Modality::Image => (FALLBACK_WIDTH, FALLBACK_HEIGHT),
        Modality::Video => (AMD_APU_VIDEO_WIDTH, AMD_APU_VIDEO_HEIGHT),
    };
    profile.width.get_or_insert(width);
    profile.height.get_or_insert(height);
    if modality == Modality::Video {
        profile.video_frames.get_or_insert(AMD_APU_VIDEO_FRAMES);
    }
    profile.vae_tiling.get_or_insert(true);
    profile
        .clip_on_cpu
        .get_or_insert(modality == Modality::Image);
    profile.diffusion_fa.get_or_insert(true);
    // Upstream auto-fit currently enumerates only discrete GPU device types,
    // so RADV integrated devices are skipped even though Vulkan can run them.
    profile.auto_fit.get_or_insert(false);
    if modality == Modality::Video {
        profile.stream_layers.get_or_insert(true);
        // In the pinned sd.cpp API, layer streaming is enabled only when the
        // diffusion parameter backend is CPU. The previous disk default caused
        // sd.cpp to ignore --stream-layers and submit whole graph-cut segments,
        // which still reset RADV on an APU. Also migrate that invalid pair if it
        // was saved from the old Brazier default.
        if profile.stream_layers == Some(true)
            && matches!(profile.params_backend.as_deref(), None | Some("disk"))
        {
            profile.params_backend = Some("cpu".to_owned());
        }
    }
    // This flag streams weights to the GPU every step. Unified memory does not
    // need that extra churn, and an explicit per-model `true` still wins.
    profile.offload_to_cpu.get_or_insert(false);
    Some(profile)
}

/// Apply the generation defaults shipped with a curated or locally defined
/// bundle. The Generate panel also uses these values to prefill its fields, but
/// engine-only settings such as Qwen-Image's flow shift must be applied here so
/// chat tools and direct API callers get the same correct invocation.
fn with_bundle_defaults(
    data_dir: &Path,
    model_id: &str,
    profile: Option<&DiffusionProfile>,
) -> Option<DiffusionProfile> {
    let bundle = crate::sdcpp_catalog::catalog(data_dir)
        .into_iter()
        .map(|entry| entry.bundle)
        .find(|bundle| bundle.model_id() == model_id)?;
    let defaults = bundle.defaults;
    let mut effective = profile.cloned().unwrap_or_default();
    effective.sampling_method = effective.sampling_method.or(defaults.sampling_method);
    effective.schedule = effective.schedule.or(defaults.schedule);
    effective.width = effective.width.or(defaults.width);
    effective.height = effective.height.or(defaults.height);
    effective.steps = effective.steps.or(defaults.steps);
    effective.cfg_scale = effective.cfg_scale.or(defaults.cfg_scale);
    effective.guidance = effective.guidance.or(defaults.guidance);
    effective.flow_shift = effective.flow_shift.or(defaults.flow_shift);
    effective.video_frames = effective.video_frames.or(defaults.video_frames);
    effective.fps = effective.fps.or(defaults.fps);
    // Video VAE/TAE decoding can require a large temporary GPU buffer even
    // after a successful denoise. Curated video bundles may make CPU decode a
    // safety default; it deliberately takes precedence over stale false
    // values saved before that default existed.
    effective.vae_on_cpu = defaults.vae_on_cpu.or(effective.vae_on_cpu);
    Some(effective)
}

#[derive(Debug, Clone, Copy, Default)]
struct InstalledMemoryPlan {
    total_bytes: u64,
    diffusion_bytes: u64,
}

/// Real installed component sizes, separated into the diffusion checkpoint
/// and everything that can be staged around it (text encoders and VAE).
fn installed_memory_plan(data_dir: &Path, model_id: &str) -> InstalledMemoryPlan {
    let Ok((dir, manifest)) = resolve_manifest(data_dir, model_id) else {
        return InstalledMemoryPlan::default();
    };
    let mut plan = InstalledMemoryPlan::default();
    let mut add = |flag: Option<&str>, name: &str| {
        let Ok(bytes) = std::fs::metadata(dir.join(name)).map(|meta| meta.len()) else {
            return;
        };
        plan.total_bytes = plan.total_bytes.saturating_add(bytes);
        if flag == Some("diffusion-model") || flag.is_none() {
            plan.diffusion_bytes = plan.diffusion_bytes.max(bytes);
        }
    };
    if let Some(file) = manifest.single_file.as_deref() {
        add(None, file);
    }
    for (flag, file) in &manifest.args {
        add(Some(flag), file);
    }
    plan
}

/// Apply sd.cpp's component-placement controls when its defaults need help to
/// stay within the accelerator budget. This is deliberately derived from the
/// installed files, not catalogue estimates, and never overwrites a person's
/// explicit model profile.
fn with_component_placement_defaults(
    profile: Option<&DiffusionProfile>,
    accelerator_memory_bytes: Option<u64>,
    plan: InstalledMemoryPlan,
) -> Option<DiffusionProfile> {
    let Some(memory) = accelerator_memory_bytes else {
        return profile.cloned();
    };
    if plan.total_bytes == 0 {
        return profile.cloned();
    }
    let budget = memory.saturating_mul(3) / 4;
    let mut effective = profile.cloned().unwrap_or_default();
    // Let sd.cpp use the full safe budget even when every component fits;
    // this also gives graph execution a hard ceiling for activations.
    effective
        .max_vram
        .get_or_insert(budget as f32 / (1024_f32 * 1024_f32 * 1024_f32));
    if plan.total_bytes > budget {
        // Encoders and VAE run in separate phases from denoising. Tell sd.cpp
        // to stage them and stream graph weights rather than requiring every
        // bundle component to remain resident in VRAM together.
        effective.auto_fit.get_or_insert(true);
        effective.diffusion_fa.get_or_insert(true);
        effective.offload_to_cpu.get_or_insert(true);
        effective
            .params_backend
            .get_or_insert_with(|| "cpu".to_owned());
        effective.stream_layers.get_or_insert(true);
        // If the denoiser itself fits, sd.cpp can stage the encoder and VAE
        // into VRAM for their own phases. Only force CPU computation when the
        // denoiser already needs the whole graph budget.
        if plan.diffusion_bytes > budget {
            effective.clip_on_cpu.get_or_insert(true);
            effective.vae_on_cpu.get_or_insert(true);
            effective.vae_tiling.get_or_insert(true);
        }
    }
    Some(effective)
}

/// Resolve the platform policy once for both image and video launch paths.
fn effective_diffusion_profile(
    data_dir: &Path,
    model_id: &str,
    profile: Option<&DiffusionProfile>,
    modality: Modality,
    model_bytes: Option<u64>,
) -> Option<DiffusionProfile> {
    let hardware = crate::hardware::detect();
    let configured_target = crate::runtime_settings::load(data_dir).target;
    let effective_target = if configured_target == RuntimeTarget::Auto {
        hardware.recommended_target
    } else {
        configured_target
    };
    let enabled = hardware.amd_apu && effective_target == RuntimeTarget::Vulkan;
    if enabled {
        tracing::info!("applying Vulkan AMD APU defaults to stable-diffusion.cpp generation");
    }
    let profile = with_bundle_defaults(data_dir, model_id, profile).or_else(|| profile.cloned());
    let profile = with_component_placement_defaults(
        profile.as_ref(),
        hardware.gpu_offload_memory_bytes,
        installed_memory_plan(data_dir, model_id),
    );
    let profile = with_amd_apu_vulkan_defaults(profile.as_ref(), enabled, modality);
    with_accelerator_memory_budget(
        profile.as_ref(),
        enabled,
        hardware.gpu_offload_memory_bytes,
        model_bytes,
    )
}

/// Apply a model-size-aware sd.cpp memory cap on Vulkan AMD APUs. Unlike
/// llama.cpp, sd.cpp has no layer-count control: `--max-vram` is its placement
/// budget and it streams graph parameters as needed. The user can always
/// override this value in the model settings.
fn with_accelerator_memory_budget(
    profile: Option<&DiffusionProfile>,
    enabled: bool,
    accelerator_memory_bytes: Option<u64>,
    model_bytes: Option<u64>,
) -> Option<DiffusionProfile> {
    if !enabled {
        return profile.cloned();
    }
    let Some(accelerator_memory_bytes) = accelerator_memory_bytes else {
        return profile.cloned();
    };
    let budget = accelerator_memory_bytes.saturating_mul(3) / 4;
    let allocation = model_bytes.map(|bytes| bytes.min(budget)).unwrap_or(budget);
    let gib = allocation as f64 / (1024_f64 * 1024_f64 * 1024_f64);
    if gib <= 0.0 {
        return profile.cloned();
    }
    let mut profile = profile.cloned().unwrap_or_default();
    profile.max_vram.get_or_insert(gib as f32);
    Some(profile)
}

/// The `<lora:name:scale>` tags sd-cli reads out of the prompt itself.
///
/// stable-diffusion.cpp has no flag naming a LoRA. It is given one directory to
/// search and the prompt decides which files in it are applied and how
/// strongly, so applying a LoRA means writing into the text the model is given.
fn lora_tags(loras: &[crate::model_settings::ResolvedLora]) -> String {
    loras
        .iter()
        .filter_map(|lora| {
            let name = lora.path.file_stem()?.to_str()?;
            Some(format!(" <lora:{name}:{}>", lora.scale))
        })
        .collect()
}

/// Apply a model's diffusion settings to an sd-cli command line.
///
/// Returns the prompt to use, which is the given one plus any LoRA tags. Every
/// flag here is opt-in: a model with no profile produces the same command line
/// it did before this existed.
async fn apply_diffusion_profile(
    command: &mut Command,
    data_dir: &Path,
    profile: Option<&DiffusionProfile>,
    prompt: &str,
) -> anyhow::Result<String> {
    let Some(profile) = profile else {
        return Ok(prompt.to_owned());
    };

    if let Some(value) = &profile.sampling_method {
        command.arg("--sampling-method").arg(value);
    }
    if let Some(value) = &profile.schedule {
        // `default` is Brazier's word for "say nothing and let sd-cli choose".
        if value != "default" {
            command.arg("--schedule").arg(value);
        }
    }
    if let Some(value) = profile.clip_skip {
        command.arg("--clip-skip").arg(value.to_string());
    }
    if let Some(value) = profile.batch_count {
        command.arg("--batch-count").arg(value.to_string());
    }
    if let Some(value) = profile.strength {
        command.arg("--strength").arg(value.to_string());
    }
    if let Some(value) = profile.img_cfg_scale {
        command.arg("--img-cfg-scale").arg(value.to_string());
    }
    if let Some(value) = profile.eta {
        command.arg("--eta").arg(value.to_string());
    }
    if let Some(value) = profile.slg_scale {
        command.arg("--slg-scale").arg(value.to_string());
    }
    if let Some(value) = &profile.skip_layers {
        command.arg("--skip-layers").arg(value);
    }
    if let Some(value) = profile.skip_layer_start {
        command.arg("--skip-layer-start").arg(value.to_string());
    }
    if let Some(value) = profile.skip_layer_end {
        command.arg("--skip-layer-end").arg(value.to_string());
    }
    if let Some(value) = profile.flow_shift {
        command.arg("--flow-shift").arg(value.to_string());
    }
    if let Some(value) = profile.threads {
        command.arg("--threads").arg(value.to_string());
    }
    if let Some(value) = &profile.rng {
        command.arg("--rng").arg(value);
    }
    if profile.vae_tiling.unwrap_or(false) {
        command.arg("--vae-tiling");
    }
    if profile.vae_on_cpu.unwrap_or(false) {
        command.arg("--vae-on-cpu");
    }
    if profile.clip_on_cpu.unwrap_or(false) {
        command.arg("--clip-on-cpu");
    }
    if profile.diffusion_fa.unwrap_or(false) {
        command.arg("--diffusion-fa");
    }
    if profile.auto_fit.unwrap_or(false) {
        command.arg("--auto-fit");
    }
    if let Some(value) = profile.max_vram {
        command.arg("--max-vram").arg(value.to_string());
    }
    if profile.offload_to_cpu.unwrap_or(false) {
        command.arg("--offload-to-cpu");
    }
    if let Some(value) = &profile.params_backend {
        command.arg("--params-backend").arg(value);
    }
    if profile.stream_layers.unwrap_or(false) {
        command.arg("--stream-layers");
    }

    // sd-cli takes one ControlNet per invocation, so the first enabled binding
    // is the one applied and the interface says as much.
    if let Some((path, binding)) =
        crate::model_settings::resolve_control_net(data_dir, &profile.control_nets)
    {
        command.arg("--control-net").arg(&path);
        command
            .arg("--control-strength")
            .arg(binding.strength.to_string());
        if let Some(image) = binding.image_path.as_deref().filter(|value| {
            let exists = Path::new(value).is_file();
            if !exists {
                tracing::warn!(image = value, "ControlNet reference image is missing");
            }
            exists
        }) {
            command.arg("--control-image").arg(image);
        }
        if binding.cpu {
            command.arg("--control-net-cpu");
        }
    }

    let loras = crate::model_settings::resolve_loras(data_dir, &profile.loras, ENGINE);
    let mut prompt = prompt.to_owned();
    if !loras.is_empty() {
        let dir = crate::adapters::stage_lora_dir(
            data_dir,
            &loras
                .iter()
                .map(|lora| lora.path.clone())
                .collect::<Vec<_>>(),
        )
        .await?;
        command.arg("--lora-model-dir").arg(dir);
        prompt.push_str(&lora_tags(&loras));
    }

    for arg in &profile.extra_args {
        command.arg(arg);
    }
    Ok(prompt)
}

/// How many trailing output lines to keep for diagnosis.
const OUTPUT_TAIL_LINES: usize = 40;

/// Last lines sd-cli wrote, newest last.
#[derive(Clone, Default)]
struct OutputTail(Arc<Mutex<std::collections::VecDeque<String>>>);

impl OutputTail {
    fn push(&self, line: String) {
        let mut lines = self.0.lock().expect("sd-cli output lock");
        if lines.len() == OUTPUT_TAIL_LINES {
            lines.pop_front();
        }
        lines.push_back(line);
    }

    fn text(&self) -> String {
        self.0
            .lock()
            .expect("sd-cli output lock")
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn parse_step_progress(line: &str, expected_total: u32) -> Option<u32> {
    let bytes = line.as_bytes();
    for slash in bytes
        .iter()
        .enumerate()
        .filter_map(|(index, byte)| (*byte == b'/').then_some(index))
    {
        let left = bytes[..slash]
            .iter()
            .rposition(|byte| !byte.is_ascii_digit())
            .map_or(0, |index| index + 1);
        let right = slash
            + 1
            + bytes[slash + 1..]
                .iter()
                .position(|byte| !byte.is_ascii_digit())
                .unwrap_or(bytes.len() - slash - 1);
        let current = std::str::from_utf8(&bytes[left..slash])
            .ok()
            .and_then(|value| value.parse::<u32>().ok());
        let total = std::str::from_utf8(&bytes[slash + 1..right])
            .ok()
            .and_then(|value| value.parse::<u32>().ok());
        if let (Some(current), Some(total)) = (current, total)
            && total == expected_total
            && current <= total
        {
            return Some(current);
        }
    }
    None
}

fn record_output_segment(
    bytes: &[u8],
    tail: &OutputTail,
    current_step: &AtomicU32,
    total_steps: u32,
) {
    let line = String::from_utf8_lossy(bytes).trim().to_owned();
    if line.is_empty() {
        return;
    }
    if let Some(step) = parse_step_progress(&line, total_steps) {
        current_step.fetch_max(step, AtomicOrdering::Relaxed);
    }
    tracing::debug!(target: "sdcpp", "{line}");
    tail.push(line);
}

/// Drain a child pipe into the tail buffer, also echoing it to the daemon log.
///
/// Progress bars redraw with carriage returns rather than newlines, so the
/// reader handles both separators to publish each diffusion step promptly.
fn collect_output<R>(
    reader: R,
    tail: OutputTail,
    current_step: Arc<AtomicU32>,
    total_steps: u32,
) -> tokio::task::JoinHandle<()>
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        use tokio::io::AsyncReadExt;
        let mut reader = tokio::io::BufReader::new(reader);
        let mut chunk = [0_u8; 8192];
        let mut pending = Vec::new();
        loop {
            let Ok(read) = reader.read(&mut chunk).await else {
                break;
            };
            if read == 0 {
                break;
            }
            for byte in &chunk[..read] {
                if matches!(*byte, b'\r' | b'\n') {
                    record_output_segment(&pending, &tail, &current_step, total_steps);
                    pending.clear();
                } else {
                    pending.push(*byte);
                }
            }
        }
        if !pending.is_empty() {
            record_output_segment(&pending, &tail, &current_step, total_steps);
        }
    })
}

/// Spawn `sd-cli`, waiting up to `timeout` for it to finish.
///
/// Its output is captured as it runs rather than read at the end, so a job that
/// times out or is stopped can still say what the engine was doing — previously
/// a timeout surfaced as a bare failure with the detail left in the terminal.
async fn run_sd_cli(
    mut command: Command,
    timeout: Duration,
    job: &JobRegistration,
) -> anyhow::Result<()> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let started = Instant::now();
    let mut child = command.spawn().context("spawn sd-cli")?;

    let tail = OutputTail::default();
    let mut readers = Vec::new();
    if let Some(pipe) = child.stdout.take() {
        readers.push(collect_output(
            pipe,
            tail.clone(),
            Arc::clone(&job.current_step),
            job.total_steps,
        ));
    }
    if let Some(pipe) = child.stderr.take() {
        readers.push(collect_output(
            pipe,
            tail.clone(),
            Arc::clone(&job.current_step),
            job.total_steps,
        ));
    }

    let status = loop {
        tokio::select! {
            finished = child.wait() => break finished.context("wait for sd-cli")?,
            _ = tokio::time::sleep_until((started + timeout).into()) => {
                let _ = child.kill().await;
                for reader in readers {
                    let _ = reader.await;
                }
                anyhow::bail!(
                    "sd-cli ran for {} without finishing and was stopped (limit {}). \
                     Try fewer frames or steps, or raise the generation timeout in \
                     Engine configuration.{}",
                    format_duration(started.elapsed()),
                    format_duration(timeout),
                    format_tail(&tail.text()),
                );
            }
            _ = job.notify.notified() => {
                if job.cancelled() {
                    let _ = child.kill().await;
                    for reader in readers {
                        let _ = reader.await;
                    }
                    return Err(CancelledError.into());
                }
            }
        }
    };

    for reader in readers {
        let _ = reader.await;
    }
    // A stop that lands as the process exits still counts as a stop.
    if job.cancelled() {
        return Err(CancelledError.into());
    }
    if !status.success() {
        anyhow::bail!(
            "sd-cli failed with {status} after {}.{}",
            format_duration(started.elapsed()),
            format_tail(&tail.text()),
        );
    }
    Ok(())
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds < 60 {
        return format!("{seconds}s");
    }
    let minutes = seconds / 60;
    if minutes < 60 {
        return format!("{minutes}m {}s", seconds % 60);
    }
    format!("{}h {}m", minutes / 60, minutes % 60)
}

fn format_tail(tail: &str) -> String {
    let trimmed = tail.trim();
    if trimmed.is_empty() {
        String::new()
    } else {
        format!("\n\nLast output from sd-cli:\n{trimmed}")
    }
}

/// Resolve the file produced by `sd-cli` for a video request.
///
/// The pinned stable-diffusion.cpp release always writes its video encoder's
/// AVI container, appending `.avi` even when the requested output name ends in
/// `.mp4`. Convert that AVI here so callers consistently receive a browser-
/// playable MP4 instead of treating a successfully rendered clip as missing.
async fn finalize_video_output(requested_output: &Path) -> anyhow::Result<PathBuf> {
    if requested_output.is_file() {
        return Ok(requested_output.to_path_buf());
    }

    let avi_output = PathBuf::from(format!("{}.avi", requested_output.display()));
    anyhow::ensure!(
        avi_output.is_file(),
        "sd-cli did not produce an output video at {} (also checked {})",
        requested_output.display(),
        avi_output.display(),
    );

    let ffmpeg = crate::toolchain_hints::resolve_command("ffmpeg").ok_or_else(|| {
        anyhow::anyhow!(
            "sd-cli produced {} as AVI, but ffmpeg is required to convert it to a playable MP4. {}",
            avi_output.display(),
            crate::media::ffmpeg_missing_message(),
        )
    })?;
    let output = Command::new(ffmpeg)
        .arg("-y")
        .arg("-i")
        .arg(&avi_output)
        .args([
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(requested_output)
        .output()
        .await
        .context("convert sd-cli AVI output to MP4 with ffmpeg")?;
    anyhow::ensure!(
        output.status.success() && requested_output.is_file(),
        "sd-cli produced {}, but ffmpeg could not convert it to MP4: {}",
        avi_output.display(),
        String::from_utf8_lossy(&output.stderr).trim(),
    );
    let _ = tokio::fs::remove_file(&avi_output).await;
    Ok(requested_output.to_path_buf())
}

/// Generate a single image with sd-cli. Resolves the binary and the model's
/// manifest, writes output under a fresh temp directory, and serializes GPU
/// access via a process-global lock.
pub async fn generate_image(
    data_dir: &Path,
    binary_override: Option<&str>,
    request: &GenerateImageRequest,
    profile: Option<&DiffusionProfile>,
) -> anyhow::Result<GenerateResult> {
    let _permit = job_lock().try_lock().map_err(|_| BusyError)?;

    let binary = resolve_binary(data_dir, binary_override)
        .filter(|path| path.is_file())
        .ok_or_else(|| anyhow::anyhow!("no stable-diffusion.cpp (sd-cli) binary available"))?;
    let (model_dir, manifest) = resolve_manifest(data_dir, &request.model_id)?;
    anyhow::ensure!(
        manifest.modality == Modality::Image,
        "model `{}` is not an image model",
        request.model_id
    );
    if let Some(init_image) = &request.init_image {
        anyhow::ensure!(
            init_image.is_file(),
            "init image not found: {}",
            init_image.display()
        );
    }

    let job_dir = job_output_dir(data_dir).await?;
    let output_path = job_dir.join("output.png");
    let profile = effective_diffusion_profile(
        data_dir,
        &request.model_id,
        profile,
        Modality::Image,
        Some(directory_size_bytes(&model_dir)),
    );
    let profile = profile.as_ref();

    let width = request
        .width
        .or(profile.and_then(|profile| profile.width))
        .unwrap_or(FALLBACK_WIDTH);
    let height = request
        .height
        .or(profile.and_then(|profile| profile.height))
        .unwrap_or(FALLBACK_HEIGHT);
    let steps = request
        .steps
        .or(profile.and_then(|profile| profile.steps))
        .unwrap_or(FALLBACK_STEPS);
    let seed = request.seed.or(profile.and_then(|profile| profile.seed));
    let cfg_scale = request
        .cfg_scale
        .or(profile.and_then(|profile| profile.cfg_scale));
    let guidance = request
        .guidance
        .or(profile.and_then(|profile| profile.guidance));
    let negative = merge_negative_prompt(request.negative_prompt.as_deref(), profile);

    let mut command = Command::new(&binary);
    command.arg("-M").arg("img_gen");
    if let Some(negative) = &negative {
        command.arg("-n").arg(negative);
    }
    command
        .arg("-W")
        .arg(width.to_string())
        .arg("-H")
        .arg(height.to_string())
        .arg("--steps")
        .arg(steps.to_string())
        .arg("-o")
        .arg(&output_path);
    if let Some(seed) = seed {
        command.arg("-s").arg(seed.to_string());
    }
    if let Some(cfg_scale) = cfg_scale {
        command.arg("--cfg-scale").arg(cfg_scale.to_string());
    }
    if let Some(guidance) = guidance {
        command.arg("--guidance").arg(guidance.to_string());
    }
    if let Some(init_image) = &request.init_image {
        command.arg("-i").arg(init_image);
    }
    // Last, because the prompt it returns carries the LoRA tags sd-cli reads.
    let prompt = apply_diffusion_profile(&mut command, data_dir, profile, &request.prompt).await?;
    command.arg("-p").arg(&prompt);
    apply_manifest_args(&mut command, &model_dir, &manifest)?;
    if let Some(dir) = binary.parent() {
        prepend_library_path(&mut command, dir);
    }

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
    run_sd_cli(command, timeout, &job).await?;
    anyhow::ensure!(
        output_path.is_file(),
        "sd-cli did not produce an output image at {}",
        output_path.display()
    );

    Ok(GenerateResult {
        output_path,
        metadata: serde_json::json!({
            "model_id": request.model_id,
            "prompt": prompt,
            "negative_prompt": negative,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "cfg_scale": cfg_scale,
        }),
    })
}

/// Generate a short video clip with sd-cli. Same lifecycle as
/// [`generate_image`] but with a longer timeout and video-specific argv.
pub async fn generate_video(
    data_dir: &Path,
    binary_override: Option<&str>,
    request: &GenerateVideoRequest,
    profile: Option<&DiffusionProfile>,
) -> anyhow::Result<GenerateResult> {
    let _permit = job_lock().try_lock().map_err(|_| BusyError)?;

    let binary = resolve_binary(data_dir, binary_override)
        .filter(|path| path.is_file())
        .ok_or_else(|| anyhow::anyhow!("no stable-diffusion.cpp (sd-cli) binary available"))?;
    let (model_dir, manifest) = resolve_manifest(data_dir, &request.model_id)?;
    anyhow::ensure!(
        manifest.modality == Modality::Video,
        "model `{}` is not a video model",
        request.model_id
    );
    if let Some(init_image) = &request.init_image {
        anyhow::ensure!(
            init_image.is_file(),
            "init image not found: {}",
            init_image.display()
        );
    }

    let job_dir = job_output_dir(data_dir).await?;
    let output_path = job_dir.join("output.mp4");
    let profile = effective_diffusion_profile(
        data_dir,
        &request.model_id,
        profile,
        Modality::Video,
        Some(directory_size_bytes(&model_dir)),
    );
    let profile = profile.as_ref();

    let width = request
        .width
        .or(profile.and_then(|profile| profile.width))
        .unwrap_or(FALLBACK_WIDTH);
    let height = request
        .height
        .or(profile.and_then(|profile| profile.height))
        .unwrap_or(FALLBACK_HEIGHT);
    let steps = request
        .steps
        .or(profile.and_then(|profile| profile.steps))
        .unwrap_or(FALLBACK_STEPS);
    let video_frames = request
        .video_frames
        .or(profile.and_then(|profile| profile.video_frames))
        .unwrap_or(FALLBACK_VIDEO_FRAMES);
    let fps = request.fps.or(profile.and_then(|profile| profile.fps));
    let seed = request.seed.or(profile.and_then(|profile| profile.seed));
    let cfg_scale = request
        .cfg_scale
        .or(profile.and_then(|profile| profile.cfg_scale));
    let guidance = request
        .guidance
        .or(profile.and_then(|profile| profile.guidance));
    let negative = merge_negative_prompt(request.negative_prompt.as_deref(), profile);

    let mut command = Command::new(&binary);
    command.arg("-M").arg("vid_gen");
    if let Some(negative) = &negative {
        command.arg("-n").arg(negative);
    }
    command
        .arg("-W")
        .arg(width.to_string())
        .arg("-H")
        .arg(height.to_string())
        .arg("--steps")
        .arg(steps.to_string())
        .arg("--video-frames")
        .arg(video_frames.to_string())
        .arg("-o")
        .arg(&output_path);
    if let Some(seed) = seed {
        command.arg("-s").arg(seed.to_string());
    }
    if let Some(cfg_scale) = cfg_scale {
        command.arg("--cfg-scale").arg(cfg_scale.to_string());
    }
    if let Some(guidance) = guidance {
        command.arg("--guidance").arg(guidance.to_string());
    }
    if let Some(fps) = fps {
        command.arg("--fps").arg(fps.to_string());
    }
    if let Some(init_image) = &request.init_image {
        command.arg("-i").arg(init_image);
    }
    let prompt = apply_diffusion_profile(&mut command, data_dir, profile, &request.prompt).await?;
    command.arg("-p").arg(&prompt);
    apply_manifest_args(&mut command, &model_dir, &manifest)?;
    if let Some(dir) = binary.parent() {
        prepend_library_path(&mut command, dir);
    }

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
    run_sd_cli(command, timeout, &job).await?;
    let output_path = finalize_video_output(&output_path).await?;

    Ok(GenerateResult {
        output_path,
        metadata: serde_json::json!({
            "model_id": request.model_id,
            "prompt": prompt,
            "negative_prompt": negative,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "cfg_scale": cfg_scale,
            "video_frames": video_frames,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn video_timeout_grows_with_the_work_requested() {
        // A long clip must not be killed while it is still rendering.
        let short = video_timeout(20, 16);
        let long = video_timeout(30, 81);
        assert!(short >= VIDEO_TIMEOUT_BASE);
        assert!(long > short, "{long:?} should exceed {short:?}");
        assert!(long <= VIDEO_TIMEOUT_MAX);
        // Absurd requests still hit a ceiling rather than hanging forever.
        assert_eq!(video_timeout(150, 241), VIDEO_TIMEOUT_MAX);
        // Zero-ish inputs must not underflow into an instant timeout.
        assert!(video_timeout(0, 0) >= VIDEO_TIMEOUT_BASE);
    }

    #[test]
    fn a_configured_timeout_replaces_the_derived_one() {
        let derived = video_timeout(20, 49);
        // Absent or zero keeps whatever the frames and steps imply.
        assert_eq!(effective_timeout(derived, None), derived);
        assert_eq!(effective_timeout(derived, Some(0)), derived);
        // A slow host can be given hours instead.
        assert_eq!(
            effective_timeout(derived, Some(6 * 3600)),
            Duration::from_secs(6 * 3600)
        );
        // Including a shorter one, for someone who would rather fail fast.
        assert_eq!(
            effective_timeout(derived, Some(60)),
            Duration::from_secs(60)
        );
    }

    #[tokio::test]
    async fn qwen_image_bundle_supplies_flow_shift_and_allows_an_override() {
        let dir = tempdir().unwrap();
        let model_id = "sdcpp-image:qwen/qwen-image";

        let defaults = with_bundle_defaults(dir.path(), model_id, None).unwrap();
        assert_eq!(defaults.width, Some(1024));
        assert_eq!(defaults.height, Some(1024));
        assert_eq!(defaults.steps, Some(20));
        assert_eq!(defaults.cfg_scale, Some(2.5));
        assert_eq!(defaults.flow_shift, Some(3.0));
        let mut command = Command::new("sd-cli");
        apply_diffusion_profile(&mut command, dir.path(), Some(&defaults), "test")
            .await
            .unwrap();
        let args = command
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(args.windows(2).any(|args| args == ["--flow-shift", "3"]));

        let configured = DiffusionProfile {
            flow_shift: Some(4.0),
            ..DiffusionProfile::default()
        };
        let overridden = with_bundle_defaults(dir.path(), model_id, Some(&configured)).unwrap();
        assert_eq!(overridden.flow_shift, Some(4.0));
    }

    #[test]
    fn vulkan_amd_apu_defaults_cover_image_and_video_flags() {
        let image = with_amd_apu_vulkan_defaults(None, true, Modality::Image).unwrap();
        assert_eq!(image.width, Some(512));
        assert_eq!(image.height, Some(512));
        assert_eq!(image.vae_tiling, Some(true));
        assert_eq!(image.clip_on_cpu, Some(true));
        assert_eq!(image.diffusion_fa, Some(true));
        assert_eq!(image.auto_fit, Some(false));
        assert_eq!(image.max_vram, None);
        assert_eq!(image.params_backend, None);
        assert_eq!(image.stream_layers, None);
        assert_eq!(image.offload_to_cpu, Some(false));

        let video = with_amd_apu_vulkan_defaults(None, true, Modality::Video).unwrap();
        assert_eq!(video.width, Some(AMD_APU_VIDEO_WIDTH));
        assert_eq!(video.height, Some(AMD_APU_VIDEO_HEIGHT));
        assert_eq!(video.video_frames, Some(AMD_APU_VIDEO_FRAMES));
        assert_eq!(video.auto_fit, Some(false));
        assert_eq!(video.max_vram, None);
        assert_eq!(video.params_backend.as_deref(), Some("cpu"));
        assert_eq!(video.stream_layers, Some(true));
    }

    #[test]
    fn explicit_model_settings_override_vulkan_amd_apu_defaults() {
        let configured = DiffusionProfile {
            width: Some(768),
            vae_tiling: Some(false),
            clip_on_cpu: Some(false),
            diffusion_fa: Some(false),
            auto_fit: Some(false),
            max_vram: Some(6.0),
            params_backend: Some("cpu".to_owned()),
            stream_layers: Some(false),
            offload_to_cpu: Some(true),
            ..DiffusionProfile::default()
        };
        let profile =
            with_amd_apu_vulkan_defaults(Some(&configured), true, Modality::Image).unwrap();
        assert_eq!(profile.width, Some(768));
        assert_eq!(profile.height, Some(512));
        assert_eq!(profile.vae_tiling, Some(false));
        assert_eq!(profile.clip_on_cpu, Some(false));
        assert_eq!(profile.diffusion_fa, Some(false));
        assert_eq!(profile.auto_fit, Some(false));
        assert_eq!(profile.max_vram, Some(6.0));
        assert_eq!(profile.params_backend.as_deref(), Some("cpu"));
        assert_eq!(profile.stream_layers, Some(false));
        assert_eq!(profile.offload_to_cpu, Some(true));

        assert_eq!(
            with_amd_apu_vulkan_defaults(None, false, Modality::Video),
            None
        );
    }

    #[test]
    fn saved_legacy_placement_still_inherits_new_residency_defaults() {
        let configured = DiffusionProfile {
            clip_on_cpu: Some(true),
            offload_to_cpu: Some(false),
            ..DiffusionProfile::default()
        };
        let profile =
            with_amd_apu_vulkan_defaults(Some(&configured), true, Modality::Video).unwrap();
        assert_eq!(profile.clip_on_cpu, Some(true));
        assert_eq!(profile.offload_to_cpu, Some(false));
        assert_eq!(profile.auto_fit, Some(false));
        assert_eq!(profile.max_vram, None);
        assert_eq!(profile.params_backend.as_deref(), Some("cpu"));
        assert_eq!(profile.stream_layers, Some(true));
    }

    #[test]
    fn accelerator_budget_uses_model_size_and_preserves_an_override() {
        let gib = 1024_u64 * 1024 * 1024;
        let profile =
            with_accelerator_memory_budget(None, true, Some(23 * gib), Some(28 * gib)).unwrap();
        assert_eq!(profile.max_vram, Some(17.25));

        let explicit = DiffusionProfile {
            max_vram: Some(4.0),
            ..DiffusionProfile::default()
        };
        let preserved =
            with_accelerator_memory_budget(Some(&explicit), true, Some(23 * gib), Some(28 * gib))
                .unwrap();
        assert_eq!(preserved.max_vram, Some(4.0));
    }

    #[test]
    fn old_disk_streaming_default_is_migrated_but_an_explicit_disk_mode_is_preserved() {
        let old_default = DiffusionProfile {
            params_backend: Some("disk".to_owned()),
            stream_layers: Some(true),
            ..DiffusionProfile::default()
        };
        let migrated =
            with_amd_apu_vulkan_defaults(Some(&old_default), true, Modality::Video).unwrap();
        assert_eq!(migrated.params_backend.as_deref(), Some("cpu"));
        assert_eq!(migrated.stream_layers, Some(true));

        let explicit_disk = DiffusionProfile {
            params_backend: Some("disk".to_owned()),
            stream_layers: Some(false),
            ..DiffusionProfile::default()
        };
        let preserved =
            with_amd_apu_vulkan_defaults(Some(&explicit_disk), true, Modality::Video).unwrap();
        assert_eq!(preserved.params_backend.as_deref(), Some("disk"));
        assert_eq!(preserved.stream_layers, Some(false));
    }

    #[tokio::test]
    async fn modern_residency_settings_reach_sd_cli() {
        let dir = tempdir().unwrap();
        let profile = DiffusionProfile {
            max_vram: Some(4.0),
            params_backend: Some("cpu".to_owned()),
            stream_layers: Some(true),
            ..DiffusionProfile::default()
        };
        let mut command = Command::new("sd-cli");
        apply_diffusion_profile(&mut command, dir.path(), Some(&profile), "test")
            .await
            .unwrap();
        let args = command
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(!args.iter().any(|arg| arg == "--auto-fit"));
        assert!(args.windows(2).any(|args| args == ["--max-vram", "4"]));
        assert!(
            args.windows(2)
                .any(|args| args == ["--params-backend", "cpu"])
        );
        assert!(args.iter().any(|arg| arg == "--stream-layers"));
    }

    #[test]
    fn managed_sdcpp_release_and_source_revision_are_pinned_together() {
        assert!(GITHUB_API.ends_with(&format!("/releases/tags/{PINNED_RELEASE_TAG}")));
        assert!(PINNED_RELEASE_TAG.ends_with(&PINNED_SOURCE_REVISION[..7]));
    }

    #[test]
    fn a_cpu_only_host_gets_hours_rather_than_minutes() {
        // The reported failure: a small model rendering a short clip on an
        // integrated GPU was cut off while it was still making progress.
        let budget = video_timeout(20, 16);
        assert!(
            budget >= Duration::from_secs(2 * 3600),
            "{budget:?} is too tight for a machine without a discrete GPU"
        );
    }

    #[test]
    fn stopping_nothing_is_harmless() {
        assert!(active_generation().is_none());
        assert!(!cancel_active_generation(), "nothing was running");
    }

    #[test]
    fn parses_carriage_return_style_diffusion_progress() {
        assert_eq!(
            parse_step_progress("sampling:  35%|████ | 7/20 [00:03<00:06]", 20),
            Some(7)
        );
        assert_eq!(
            parse_step_progress("sampling: 100%|████|20/20 [00:09<00:00]", 20),
            Some(20)
        );
        assert_eq!(parse_step_progress("loaded 7/12 tensors", 20), None);
    }

    #[test]
    fn manifests_round_trip_through_the_install_helpers() {
        let dir = tempdir().unwrap();
        let target =
            model_dir_for_key(dir.path(), Modality::Image, "acme/flux").expect("valid key");
        let manifest = SdcppManifest {
            modality: Modality::Image,
            args: BTreeMap::from([
                ("diffusion-model".to_owned(), "model.gguf".to_owned()),
                ("t5xxl".to_owned(), "t5xxl_fp16.safetensors".to_owned()),
            ]),
            component_sources: BTreeMap::new(),
            single_file: None,
            supports_init_image: false,
        };
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(write_manifest(&target, &manifest))
            .unwrap();
        let loaded = load_manifest(&target).unwrap();
        assert_eq!(loaded.modality, Modality::Image);
        assert_eq!(
            loaded.args.get("t5xxl").map(String::as_str),
            Some("t5xxl_fp16.safetensors")
        );
    }

    #[test]
    fn component_destinations_reject_traversal() {
        let dir = tempdir().unwrap();
        assert!(
            component_destination(
                dir.path(),
                Modality::Video,
                "acme/wan",
                "../escape.safetensors"
            )
            .is_err()
        );
        assert!(
            component_destination(dir.path(), Modality::Video, "../..", "vae.safetensors").is_err()
        );
        assert!(
            component_destination(dir.path(), Modality::Video, "acme/wan", "vae.safetensors")
                .is_ok()
        );
    }

    #[test]
    fn selects_macos_arm64_asset() {
        let assets = [
            "sd-master-Darwin-arm64.tar.gz",
            "sd-master-Darwin-x86_64.tar.gz",
            "sd-master-Linux-x86_64.tar.gz",
        ];
        assert_eq!(
            select_release_asset_for_target(assets, "macos-arm64", RuntimeTarget::Cpu),
            Some("sd-master-Darwin-arm64.tar.gz")
        );
        assert_eq!(
            select_release_asset_for_target(assets, "macos-arm64", RuntimeTarget::Metal),
            Some("sd-master-Darwin-arm64.tar.gz")
        );
    }

    #[test]
    fn selects_linux_cpu_vulkan_rocm_assets() {
        let assets = [
            "sd-master-Linux-x86_64.tar.gz",
            "sd-master-Linux-x86_64-vulkan.tar.gz",
            "sd-master-Linux-x86_64-rocm.tar.gz",
            "sd-master-Linux-x86_64-cuda12.tar.gz",
        ];
        assert_eq!(
            select_release_asset_for_target(assets, "linux-x64", RuntimeTarget::Cpu),
            Some("sd-master-Linux-x86_64.tar.gz")
        );
        assert_eq!(
            select_release_asset_for_target(assets, "linux-x64", RuntimeTarget::Vulkan),
            Some("sd-master-Linux-x86_64-vulkan.tar.gz")
        );
        assert_eq!(
            select_release_asset_for_target(assets, "linux-x64", RuntimeTarget::Rocm),
            Some("sd-master-Linux-x86_64-rocm.tar.gz")
        );
    }

    #[test]
    fn selects_windows_cpu_cuda_vulkan_assets() {
        let assets = [
            "sd-master-win-cpu-x64.zip",
            "sd-master-win-cuda12-x64.zip",
            "sd-master-win-vulkan-x64.zip",
        ];
        assert_eq!(
            select_release_asset_for_target(assets, "windows-x64", RuntimeTarget::Cpu),
            Some("sd-master-win-cpu-x64.zip")
        );
        assert_eq!(
            select_release_asset_for_target(assets, "windows-x64", RuntimeTarget::Cuda),
            Some("sd-master-win-cuda12-x64.zip")
        );
        assert_eq!(
            select_release_asset_for_target(assets, "windows-x64", RuntimeTarget::Vulkan),
            Some("sd-master-win-vulkan-x64.zip")
        );
    }

    #[test]
    fn skips_cudart_redistributable_packages() {
        let assets = [
            "sd-master-win-cuda12-x64.zip",
            "sd-master-win-cudart-x64.zip",
        ];
        assert_eq!(
            select_release_asset_for_target(assets, "windows-x64", RuntimeTarget::Cuda),
            Some("sd-master-win-cuda12-x64.zip")
        );
    }

    #[test]
    fn no_match_returns_none() {
        let assets = ["sd-master-Linux-x86_64.tar.gz"];
        assert_eq!(
            select_release_asset_for_target(assets, "windows-x64", RuntimeTarget::Cpu),
            None
        );
    }

    #[test]
    fn model_id_round_trips() {
        assert_eq!(
            model_id(Modality::Image, "acme/flux-schnell"),
            "sdcpp-image:acme/flux-schnell"
        );
        assert_eq!(
            model_id(Modality::Video, "acme/wan2.2"),
            "sdcpp-video:acme/wan2.2"
        );
        let (modality, key) = parse_model_id("sdcpp-image:acme/flux-schnell").unwrap();
        assert_eq!(modality, Modality::Image);
        assert_eq!(key, "acme/flux-schnell");
        let (modality, key) = parse_model_id("sdcpp-video:acme/wan2.2").unwrap();
        assert_eq!(modality, Modality::Video);
        assert_eq!(key, "acme/wan2.2");
    }

    #[test]
    fn parse_model_id_rejects_other_prefixes() {
        assert!(parse_model_id("gguf:acme/model.gguf").is_err());
        assert!(parse_model_id("sdcpp-image").is_err());
    }

    #[test]
    fn discovery_candidates_cover_all_targets() {
        let data = PathBuf::from("/tmp/brazier-data");
        let candidates = discovery_candidates(&data, Some("/usr/bin:/opt/bin"));
        assert!(candidates.contains(&managed_binary_path(&data)));
        assert!(
            candidates
                .iter()
                .any(|path| path.ends_with(format!("cuda/bin/{}", binary_name())))
        );
        assert!(candidates.iter().any(|path| path.ends_with(binary_name())));
    }

    #[test]
    fn lists_multi_component_and_single_file_models() {
        let dir = tempdir().unwrap();
        let image_model = image_root(dir.path()).join("acme/flux-schnell");
        std::fs::create_dir_all(&image_model).unwrap();
        std::fs::write(image_model.join("diffusion_model.safetensors"), b"a").unwrap();
        std::fs::write(image_model.join("vae.safetensors"), b"b").unwrap();
        std::fs::write(
            manifest_file_path(&image_model),
            serde_json::to_vec(&SdcppManifest {
                modality: Modality::Image,
                args: BTreeMap::from([
                    (
                        "diffusion-model".to_owned(),
                        "diffusion_model.safetensors".to_owned(),
                    ),
                    ("vae".to_owned(), "vae.safetensors".to_owned()),
                ]),
                component_sources: BTreeMap::new(),
                single_file: None,
                supports_init_image: false,
            })
            .unwrap(),
        )
        .unwrap();

        let video_model = video_root(dir.path()).join("acme/wan2.2");
        std::fs::create_dir_all(&video_model).unwrap();
        std::fs::write(video_model.join("model.gguf"), b"c").unwrap();
        std::fs::write(
            manifest_file_path(&video_model),
            serde_json::to_vec(&SdcppManifest {
                modality: Modality::Video,
                args: BTreeMap::new(),
                component_sources: BTreeMap::new(),
                single_file: Some("model.gguf".to_owned()),
                supports_init_image: false,
            })
            .unwrap(),
        )
        .unwrap();

        let models = list_models(dir.path()).unwrap();
        assert_eq!(models.len(), 2);
        let image = models
            .iter()
            .find(|model| model.id == "sdcpp-image:acme/flux-schnell")
            .unwrap();
        assert_eq!(image.engine, ENGINE);
        assert_eq!(image.capabilities.output_modalities, vec!["image"]);
        let video = models
            .iter()
            .find(|model| model.id == "sdcpp-video:acme/wan2.2")
            .unwrap();
        assert_eq!(video.capabilities.output_modalities, vec!["video"]);

        assert_eq!(
            path_for_model_id(dir.path(), &image.id).unwrap(),
            image_model
        );
        assert_eq!(
            path_for_model_id(dir.path(), &video.id).unwrap(),
            video_model.join("model.gguf")
        );
    }

    #[test]
    fn download_destination_validates_repo_id_and_filename() {
        let dir = tempdir().unwrap();
        let path = download_destination(
            dir.path(),
            Modality::Image,
            "acme/flux-schnell",
            "vae.safetensors",
        )
        .unwrap();
        assert_eq!(
            path,
            image_root(dir.path()).join("acme/flux-schnell/vae.safetensors")
        );
        assert!(
            download_destination(
                dir.path(),
                Modality::Image,
                "not-a-repo-id",
                "vae.safetensors"
            )
            .is_err()
        );
        assert!(
            download_destination(dir.path(), Modality::Image, "acme/flux-schnell", "../evil")
                .is_err()
        );
    }
}

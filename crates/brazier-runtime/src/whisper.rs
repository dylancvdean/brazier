//! whisper.cpp discovery, managed installation, activation, and transcription.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use anyhow::Context;
use flate2::read::GzDecoder;
use futures::StreamExt;
use tar::Archive;
use tokio::{io::AsyncWriteExt, process::Command};

use crate::{
    models_store,
    progress::{ProgressCallback, ProgressEvent},
    runtime_settings::RuntimeTarget,
};

pub const ENGINE: &str = "whisper.cpp";

const GITHUB_API: &str = "https://api.github.com/repos/ggml-org/whisper.cpp/releases/latest";
const USER_AGENT: &str = "brazier-whisper-manager";

pub fn managed_engine_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("engines").join("whisper.cpp")
}

pub fn binary_name() -> &'static str {
    if cfg!(windows) {
        "whisper-cli.exe"
    } else {
        "whisper-cli"
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

/// Official whisper.cpp releases ship CLI tarballs for Linux/Windows only.
/// macOS assets are XCFramework packages, not whisper-cli binaries.
pub fn managed_prebuilts_supported() -> bool {
    platform_asset_tag().is_some()
}

/// Platform tag substring used to select a GitHub release asset.
pub fn platform_asset_tag() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Some("ubuntu-x64"),
        ("linux", "aarch64") => Some("ubuntu-arm64"),
        ("windows", "x86_64") => Some("x64"),
        ("windows", "aarch64") => None,
        ("macos", _) => None,
        _ => None,
    }
}

/// Choose the best prebuilt release asset for this host.
pub fn select_release_asset<'a>(
    asset_names: impl IntoIterator<Item = &'a str>,
    platform_tag: &str,
) -> Option<&'a str> {
    let mut candidates: Vec<&str> = asset_names
        .into_iter()
        .filter(|name| {
            let lower = name.to_ascii_lowercase();
            if !(lower.ends_with(".tar.gz") || lower.ends_with(".zip")) {
                return false;
            }
            if lower.contains("xcframework") || lower.contains("cublas") || lower.contains("blas") {
                return false;
            }
            if platform_tag.starts_with("ubuntu") {
                return lower.contains("whisper-bin") && lower.contains(platform_tag);
            }
            // Windows CPU: whisper-bin-x64.zip / whisper-bin-Win32.zip — prefer x64.
            lower.contains("whisper-bin")
                && lower.contains(platform_tag)
                && !lower.contains("win32")
        })
        .collect();
    candidates.sort_by_key(|name| name.len());
    candidates.first().copied()
}

pub fn select_release_asset_for_target<'a>(
    asset_names: impl IntoIterator<Item = &'a str>,
    platform_tag: &str,
    target: RuntimeTarget,
) -> Option<&'a str> {
    let names: Vec<&str> = asset_names.into_iter().collect();
    if matches!(
        target,
        RuntimeTarget::Auto | RuntimeTarget::Cpu | RuntimeTarget::Metal
    ) {
        return select_release_asset(names, platform_tag);
    }
    if target == RuntimeTarget::Cuda && platform_tag == "x64" {
        let mut candidates: Vec<&str> = names
            .into_iter()
            .filter(|name| {
                let lower = name.to_ascii_lowercase();
                lower.contains("cublas")
                    && lower.contains("x64")
                    && (lower.ends_with(".zip") || lower.ends_with(".tar.gz"))
            })
            .collect();
        // Prefer newer CUDA 12.x packages.
        candidates.sort_by(|a, b| {
            let score = |name: &str| {
                let lower = name.to_ascii_lowercase();
                if lower.contains("12.4") {
                    0
                } else if lower.contains("12.") {
                    1
                } else {
                    2
                }
            };
            score(a).cmp(&score(b)).then_with(|| a.len().cmp(&b.len()))
        });
        return candidates.first().copied();
    }
    None
}

#[derive(Debug, Clone)]
pub struct ReleaseAsset {
    pub name: String,
    pub browser_download_url: String,
}

/// Newest release tag from cache, without waiting on GitHub.
///
/// Status views call this on every open, so a stale-but-instant answer beats
/// a blocking lookup; the refresh it triggers lands in time for the next one.
pub fn cached_release_tag(client: &reqwest::Client) -> crate::github_releases::CachedRelease {
    crate::github_releases::cached_or_refresh(client, GITHUB_API, USER_AGENT)
}

pub async fn resolve_managed_release(
    client: &reqwest::Client,
    target: RuntimeTarget,
) -> anyhow::Result<(String, ReleaseAsset)> {
    let platform = platform_asset_tag()
        .context("managed whisper.cpp CLI binaries are not available for this platform (macOS uses source builds)")?;
    let release = crate::github_releases::latest_release(client, GITHUB_API, USER_AGENT).await?;
    let names: Vec<String> = release.asset_names().map(str::to_owned).collect();
    let selected =
        select_release_asset_for_target(names.iter().map(String::as_str), platform, target)
            .context("no matching whisper.cpp release asset for this platform/target")?
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

pub async fn install_managed_binary_with_progress(
    client: &reqwest::Client,
    data_dir: &Path,
    target: RuntimeTarget,
    mut progress: ProgressCallback,
) -> anyhow::Result<PathBuf> {
    progress(ProgressEvent::phase(
        "resolve",
        "Looking up the latest whisper.cpp release",
    ));
    let (tag, asset) = resolve_managed_release(client, target).await?;
    tracing::info!(%tag, asset = %asset.name, "downloading managed whisper.cpp binary");
    progress(ProgressEvent::phase(
        "download",
        format!("Downloading {tag} ({})", asset.name),
    ));

    let response = client
        .get(&asset.browser_download_url)
        .header("user-agent", USER_AGENT)
        .send()
        .await
        .context("download whisper.cpp release")?
        .error_for_status()
        .context("whisper.cpp release download failed")?;
    let total = response.content_length();
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    let mut written = 0_u64;
    let mut last_emit = 0_u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read whisper.cpp release body")?;
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
            .context("clear previous managed whisper install")?;
    }
    tokio::fs::create_dir_all(&bin_dir)
        .await
        .context("create whisper engine bin directory")?;
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
        "Extracting whisper-cli and shared libraries",
    ));
    extract_release_archive(&archive_path, &bin_dir).context("extract whisper.cpp release")?;
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

fn extract_release_archive(archive_path: &Path, bin_dir: &Path) -> anyhow::Result<()> {
    let file = std::fs::File::open(archive_path).context("open archive")?;
    let name = archive_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let keep = |file_name: &str| -> bool {
        let is_lib = file_name.contains(".so")
            || file_name.ends_with(".dll")
            || file_name.ends_with(".dylib");
        let is_cli = file_name == "whisper-cli"
            || file_name == "whisper-cli.exe"
            || file_name == "main"
            || file_name == "main.exe";
        is_lib || is_cli || file_name.starts_with("whisper") || file_name.starts_with("ggml")
    };
    let rename_cli = |file_name: &str| -> String {
        if file_name == "main" || file_name == "main.exe" {
            binary_name().to_owned()
        } else {
            file_name.to_owned()
        }
    };

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
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if file_name.is_empty() || file_name == "LICENSE" || !keep(file_name) {
                continue;
            }
            let dest_name = rename_cli(file_name);
            let dest = bin_dir.join(&dest_name);
            entry
                .unpack(&dest)
                .with_context(|| format!("unpack {file_name}"))?;
            if dest_name == binary_name() {
                found_cli = true;
            }
        }
        anyhow::ensure!(found_cli, "whisper-cli binary not found in archive");
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
            if !keep(file_name) {
                continue;
            }
            let dest_name = rename_cli(file_name);
            let mut destination =
                std::fs::File::create(bin_dir.join(&dest_name)).context("create zip output")?;
            std::io::copy(&mut entry, &mut destination).context("extract zip entry")?;
            found_cli |= dest_name == binary_name();
        }
        anyhow::ensure!(found_cli, "whisper-cli binary not found in archive");
        return Ok(());
    }
    anyhow::bail!("unsupported archive format: {name}");
}

pub async fn ensure_binary_with_progress(
    client: &reqwest::Client,
    data_dir: &Path,
    target: RuntimeTarget,
    force: bool,
    mut progress: ProgressCallback,
) -> anyhow::Result<PathBuf> {
    let target = if target == RuntimeTarget::Auto {
        let detected = crate::hardware::detect().recommended_target;
        // Metal/macOS has no managed CLI; fall back to discovery only.
        if !managed_prebuilts_supported() {
            RuntimeTarget::Cpu
        } else if detected == RuntimeTarget::Cuda {
            RuntimeTarget::Cuda
        } else {
            RuntimeTarget::Cpu
        }
    } else {
        target
    };
    if force {
        anyhow::ensure!(
            managed_prebuilts_supported(),
            "managed whisper.cpp binaries are not available on this platform; build from source instead"
        );
        return install_managed_binary_with_progress(client, data_dir, target, progress).await;
    }
    let managed = managed_binary_path_for_target(data_dir, target);
    progress(ProgressEvent::phase(
        "discover",
        "Looking for an existing whisper-cli binary",
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
                "discovered whisper-cli failed a smoke test; trying managed install"
            );
        }
    }
    anyhow::ensure!(
        managed_prebuilts_supported(),
        "no whisper-cli found and managed binaries are not available on this platform; build whisper.cpp from source"
    );
    install_managed_binary_with_progress(client, data_dir, target, progress).await
}

pub fn whisper_root(data_dir: &Path) -> PathBuf {
    data_dir.join("models").join("whisper")
}

/// Resolve an activated or discovered ASR binary (whisper-cli or WhisperKit).
pub fn resolve_binary(data_dir: &Path, override_path: Option<&str>) -> Option<PathBuf> {
    if let Some(path) = override_path
        .map(PathBuf::from)
        .filter(|path| path.is_file())
    {
        return Some(path);
    }
    for target in [
        RuntimeTarget::Cpu,
        RuntimeTarget::Cuda,
        RuntimeTarget::Rocm,
        RuntimeTarget::Vulkan,
    ] {
        let managed = managed_binary_path_for_target(data_dir, target);
        if managed.is_file() {
            return Some(managed);
        }
    }
    for (build_id, record) in crate::builds::list_builds(data_dir, ENGINE) {
        let _ = build_id;
        let path = PathBuf::from(&record.binary);
        if path.is_file() {
            return Some(path);
        }
    }
    // Apple Silicon: WhisperKit source builds or brew-installed CLI.
    if let Some(path) = crate::whisperkit::resolve_binary(data_dir, None) {
        return Some(path);
    }
    which_binary(binary_name())
}

fn which_binary(name: &str) -> Option<PathBuf> {
    let path_env = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_env) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

pub fn discovery_candidates(data_dir: &Path, path_env: Option<&str>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for target in [
        RuntimeTarget::Cpu,
        RuntimeTarget::Cuda,
        RuntimeTarget::Rocm,
        RuntimeTarget::Vulkan,
    ] {
        let managed = managed_binary_path_for_target(data_dir, target);
        if managed.is_file() {
            candidates.push(managed);
        }
    }
    for (_, record) in crate::builds::list_builds(data_dir, ENGINE) {
        let path = PathBuf::from(record.binary);
        if path.is_file() {
            candidates.push(path);
        }
    }
    if let Some(path_env) = path_env {
        for dir in std::env::split_paths(path_env) {
            let candidate = dir.join(binary_name());
            if candidate.is_file() {
                candidates.push(candidate);
            }
        }
    }
    candidates
}

pub fn binary_appears_runnable(path: &Path) -> bool {
    std::process::Command::new(path)
        .arg("-h")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success() || status.code().is_some())
        .unwrap_or(false)
}

/// Stable model id for a whisper weight file under `models/whisper`.
pub fn model_id_for_path(whisper_root: &Path, file: &Path) -> anyhow::Result<String> {
    let relative = file
        .strip_prefix(whisper_root)
        .map_err(|_| anyhow::anyhow!("model path is outside the whisper store"))?;
    let key = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    anyhow::ensure!(!key.is_empty(), "empty whisper model key");
    Ok(format!("whisper:{key}"))
}

fn validate_whisper_key(key: &str) -> anyhow::Result<()> {
    anyhow::ensure!(!key.is_empty(), "empty whisper model key");
    anyhow::ensure!(
        !key.split('/')
            .any(|part| part.is_empty() || part == "." || part == ".."),
        "invalid whisper model key"
    );
    Ok(())
}

pub fn path_for_model_id(data_dir: &Path, model_id: &str) -> anyhow::Result<PathBuf> {
    let key = model_id
        .strip_prefix("whisper:")
        .ok_or_else(|| anyhow::anyhow!("not a whisper model id: {model_id}"))?;
    validate_whisper_key(key)?;
    let path = whisper_root(data_dir).join(key);
    anyhow::ensure!(path.is_file(), "whisper model not found: {model_id}");
    Ok(path)
}

pub fn download_destination(
    data_dir: &Path,
    repo_id: &str,
    filename: &str,
) -> anyhow::Result<PathBuf> {
    models_store::validate_repo_id(repo_id)?;
    models_store::validate_filename(filename)?;
    Ok(whisper_root(data_dir).join(repo_id).join(filename))
}

fn is_whisper_weight(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    (lower.ends_with(".bin") || lower.ends_with(".gguf"))
        && (lower.contains("whisper")
            || lower.contains("ggml-")
            || lower.starts_with("ggml")
            || lower.contains("tiny")
            || lower.contains("base")
            || lower.contains("small")
            || lower.contains("medium")
            || lower.contains("large"))
}

/// List on-disk whisper weight files.
pub fn list_models(data_dir: &Path) -> anyhow::Result<Vec<crate::types::ModelDescriptor>> {
    let root = whisper_root(data_dir);
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut models = Vec::new();
    collect_whisper(&root, &root, &mut models)?;
    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

fn collect_whisper(
    root: &Path,
    dir: &Path,
    models: &mut Vec<crate::types::ModelDescriptor>,
) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_whisper(root, &path, models)?;
            continue;
        }
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if !is_whisper_weight(name) {
            continue;
        }
        let id = model_id_for_path(root, &path)?;
        let size = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
        models.push(crate::types::ModelDescriptor {
            id,
            name: name.to_owned(),
            engine: ENGINE.to_owned(),
            capabilities: crate::types::ModelCapabilities {
                input_modalities: vec!["audio".into()],
                output_modalities: vec!["text".into()],
                streaming: false,
                tools: false,
                reasoning: false,
                max_context_length: None,
                reasoning_modes: Vec::new(),
                harmony: false,
                // Whisper weights are batch ASR engines, not native audio chat models.
                audio_input: None,
            },
            size_bytes: Some(size),
            read_only: false,
            library_label: None,
        });
    }
    Ok(())
}

/// Pick the first available whisper model, or a preferred override.
pub fn resolve_model_path(data_dir: &Path, preferred: Option<&str>) -> Option<PathBuf> {
    if let Some(id) = preferred {
        if let Ok(path) = path_for_model_id(data_dir, id) {
            return Some(path);
        }
        let as_path = PathBuf::from(id);
        if as_path.is_file() {
            return Some(as_path);
        }
    }
    list_models(data_dir)
        .ok()?
        .into_iter()
        .next()
        .and_then(|model| path_for_model_id(data_dir, &model.id).ok())
}

pub struct TranscribeRequest<'a> {
    pub binary: &'a Path,
    pub model: &'a Path,
    pub audio: &'a Path,
    /// Decoding options configured for this ASR model, when it has any.
    pub profile: Option<&'a crate::model_settings::TranscriptionProfile>,
}

/// Add a model's decoding options to a whisper-cli command line.
///
/// Every flag is opt-in, so a model with no profile is invoked exactly as it
/// was before. The names are whisper.cpp's own rather than its short forms,
/// which reads better against the settings that produced them.
fn apply_transcription_profile(
    command: &mut Command,
    profile: Option<&crate::model_settings::TranscriptionProfile>,
) {
    let Some(profile) = profile else { return };
    if let Some(language) = &profile.language {
        command.arg("--language").arg(language);
    }
    if profile.translate.unwrap_or(false) {
        command.arg("--translate");
    }
    if let Some(value) = profile.beam_size {
        command.arg("--beam-size").arg(value.to_string());
    }
    if let Some(value) = profile.best_of {
        command.arg("--best-of").arg(value.to_string());
    }
    if let Some(value) = profile.temperature {
        command.arg("--temperature").arg(value.to_string());
    }
    if let Some(value) = profile.max_context {
        command.arg("--max-context").arg(value.to_string());
    }
    if let Some(value) = profile.max_len {
        command.arg("--max-len").arg(value.to_string());
    }
    if profile.split_on_word.unwrap_or(false) {
        command.arg("--split-on-word");
    }
    if let Some(value) = profile.word_threshold {
        command.arg("--word-thold").arg(value.to_string());
    }
    if let Some(value) = profile.entropy_threshold {
        command.arg("--entropy-thold").arg(value.to_string());
    }
    if let Some(value) = profile.logprob_threshold {
        command.arg("--logprob-thold").arg(value.to_string());
    }
    if let Some(value) = profile.no_speech_threshold {
        command.arg("--no-speech-thold").arg(value.to_string());
    }
    if profile.no_fallback.unwrap_or(false) {
        command.arg("--no-fallback");
    }
    if profile.suppress_nst.unwrap_or(false) {
        command.arg("--suppress-nst");
    }
    if let Some(value) = profile.threads {
        command.arg("--threads").arg(value.to_string());
    }
    if profile.flash_attention.unwrap_or(false) {
        command.arg("--flash-attn");
    }
    if let Some(prompt) = profile
        .initial_prompt
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        command.arg("--prompt").arg(prompt);
    }
    for arg in &profile.extra_args {
        command.arg(arg);
    }
}

/// Optional context for backends that download models themselves (WhisperKit).
pub struct TranscribeContext<'a> {
    pub data_dir: &'a Path,
    /// Preferred model id / path / WhisperKit variant name.
    pub model_pref: Option<&'a str>,
}

/// Run whisper-cli (or WhisperKit when the binary is `whisperkit-cli`) on audio.
pub async fn transcribe(request: TranscribeRequest<'_>) -> anyhow::Result<String> {
    transcribe_with_context(request, None).await
}

/// Like [`transcribe`], with optional data-dir context for WhisperKit model cache.
pub async fn transcribe_with_context(
    request: TranscribeRequest<'_>,
    context: Option<TranscribeContext<'_>>,
) -> anyhow::Result<String> {
    anyhow::ensure!(request.binary.is_file(), "ASR binary missing");
    anyhow::ensure!(request.audio.is_file(), "audio file missing");

    if crate::whisperkit::is_whisperkit_binary(request.binary) {
        let data_dir = context
            .as_ref()
            .map(|ctx| ctx.data_dir)
            .ok_or_else(|| anyhow::anyhow!("WhisperKit transcription requires data_dir context"))?;
        let model_pref = context.as_ref().and_then(|ctx| ctx.model_pref);
        return crate::whisperkit::transcribe(crate::whisperkit::TranscribeRequest {
            binary: request.binary,
            data_dir,
            model: model_pref,
            audio: request.audio,
            // WhisperKit takes a language and nothing else this profile offers,
            // so the rest of the decoding options do not reach it.
            language: request
                .profile
                .and_then(|profile| profile.language.as_deref())
                .filter(|value| *value != "auto"),
        })
        .await;
    }

    anyhow::ensure!(request.model.is_file(), "whisper model missing");

    let output_base = request.audio.with_extension("");
    let output_txt = PathBuf::from(format!("{}.txt", output_base.display()));
    let _ = tokio::fs::remove_file(&output_txt).await;

    let mut command = Command::new(request.binary);
    command
        .arg("-m")
        .arg(request.model)
        .arg("-f")
        .arg(request.audio)
        .arg("-otxt")
        .arg("-of")
        .arg(&output_base)
        .arg("-np");
    apply_transcription_profile(&mut command, request.profile);
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("spawn whisper-cli")?;

    let status = tokio::time::timeout(Duration::from_secs(600), child.wait())
        .await
        .context("whisper-cli timed out")?
        .context("wait for whisper-cli")?;
    anyhow::ensure!(status.success(), "whisper-cli failed with {status}");

    if output_txt.is_file() {
        let text = tokio::fs::read_to_string(&output_txt)
            .await
            .context("read whisper transcript")?;
        let _ = tokio::fs::remove_file(&output_txt).await;
        return Ok(text.trim().to_owned());
    }

    // Fallback: some builds print the transcript to stdout.
    let mut command = Command::new(request.binary);
    command
        .arg("-m")
        .arg(request.model)
        .arg("-f")
        .arg(request.audio)
        .arg("-nt")
        .arg("-np");
    apply_transcription_profile(&mut command, request.profile);
    let output = command
        .output()
        .await
        .context("re-run whisper-cli for stdout")?;
    anyhow::ensure!(
        output.status.success(),
        "whisper-cli failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_ids_are_stable() {
        let root = Path::new("/data/models/whisper");
        let file = root.join("ggerganov/whisper.cpp/ggml-base.en.bin");
        assert_eq!(
            model_id_for_path(root, &file).unwrap(),
            "whisper:ggerganov/whisper.cpp/ggml-base.en.bin"
        );
    }

    #[test]
    fn selects_linux_cpu_asset() {
        let assets = [
            "whisper-bin-ubuntu-x64.tar.gz",
            "whisper-bin-ubuntu-arm64.tar.gz",
            "whisper-cublas-12.4.0-bin-x64.zip",
            "whisper-v1.9.0-xcframework.zip",
        ];
        assert_eq!(
            select_release_asset(assets, "ubuntu-x64"),
            Some("whisper-bin-ubuntu-x64.tar.gz")
        );
        assert_eq!(
            select_release_asset_for_target(assets, "x64", RuntimeTarget::Cuda),
            Some("whisper-cublas-12.4.0-bin-x64.zip")
        );
    }

    #[test]
    fn selects_windows_cpu_asset() {
        let assets = [
            "whisper-bin-x64.zip",
            "whisper-bin-Win32.zip",
            "whisper-blas-bin-x64.zip",
            "whisper-cublas-12.4.0-bin-x64.zip",
        ];
        assert_eq!(
            select_release_asset(assets, "x64"),
            Some("whisper-bin-x64.zip")
        );
    }
}

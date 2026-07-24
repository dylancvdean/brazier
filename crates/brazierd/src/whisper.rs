//! whisper.cpp CLI discovery, activation, and one-shot transcription.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use anyhow::Context;
use tokio::process::Command;

use crate::models_store;

pub const ENGINE: &str = "whisper.cpp";

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

pub fn whisper_root(data_dir: &Path) -> PathBuf {
    data_dir.join("models").join("whisper")
}

/// Resolve an activated or discovered whisper-cli binary.
pub fn resolve_binary(data_dir: &Path, override_path: Option<&str>) -> Option<PathBuf> {
    if let Some(path) = override_path.map(PathBuf::from).filter(|path| path.is_file()) {
        return Some(path);
    }
    let managed = managed_engine_dir(data_dir).join("bin").join(binary_name());
    if managed.is_file() {
        return Some(managed);
    }
    for (build_id, record) in crate::builds::list_builds(data_dir, ENGINE) {
        let _ = build_id;
        let path = PathBuf::from(&record.binary);
        if path.is_file() {
            return Some(path);
        }
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
    let managed = managed_engine_dir(data_dir).join("bin").join(binary_name());
    if managed.is_file() {
        candidates.push(managed);
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
}

/// Run whisper-cli on a WAV (or ffmpeg-converted) audio file and return transcript text.
pub async fn transcribe(request: TranscribeRequest<'_>) -> anyhow::Result<String> {
    anyhow::ensure!(request.binary.is_file(), "whisper-cli binary missing");
    anyhow::ensure!(request.model.is_file(), "whisper model missing");
    anyhow::ensure!(request.audio.is_file(), "audio file missing");

    let output_base = request.audio.with_extension("");
    let output_txt = PathBuf::from(format!("{}.txt", output_base.display()));
    let _ = tokio::fs::remove_file(&output_txt).await;

    let mut child = Command::new(request.binary)
        .arg("-m")
        .arg(request.model)
        .arg("-f")
        .arg(request.audio)
        .arg("-otxt")
        .arg("-of")
        .arg(&output_base)
        .arg("-np")
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
    let output = Command::new(request.binary)
        .arg("-m")
        .arg(request.model)
        .arg("-f")
        .arg(request.audio)
        .arg("-nt")
        .arg("-np")
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
}

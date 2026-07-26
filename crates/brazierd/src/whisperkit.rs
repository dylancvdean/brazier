//! WhisperKit / Argmax CLI for on-device speech recognition on Apple Silicon.
//!
//! Builds produce `whisperkit-cli` from [argmax-oss-swift](https://github.com/argmaxinc/argmax-oss-swift).
//! The CLI downloads CoreML models on demand (Hugging Face `argmaxinc/whisperkit-coreml`)
//! and is also available via Homebrew (`brew install whisperkit-cli`).

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use anyhow::Context;
use tokio::process::Command;

use crate::builds;

pub const ENGINE: &str = "whisperkit";
pub const BINARY_NAME: &str = "whisperkit-cli";

/// Default model variant when none is configured (downloaded on first use).
pub const DEFAULT_MODEL: &str = "base";

pub fn models_root(data_dir: &Path) -> PathBuf {
    data_dir.join("models").join("whisperkit")
}

pub fn download_model_dir(data_dir: &Path) -> PathBuf {
    models_root(data_dir).join("models")
}

pub fn download_tokenizer_dir(data_dir: &Path) -> PathBuf {
    models_root(data_dir).join("tokenizers")
}

/// Whether `path` looks like a WhisperKit / Argmax CLI binary.
pub fn is_whisperkit_binary(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            let lower = name.to_ascii_lowercase();
            lower.contains("whisperkit") || lower == "argmax-cli"
        })
}

pub fn binary_appears_runnable(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    std::process::Command::new(path)
        .arg("--help")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success() || status.code().is_some())
        .unwrap_or(false)
}

/// Resolve a WhisperKit CLI: settings override, then source builds, then PATH.
pub fn resolve_binary(data_dir: &Path, override_path: Option<&str>) -> Option<PathBuf> {
    if let Some(path) = override_path
        .map(PathBuf::from)
        .filter(|path| path.is_file() && is_whisperkit_binary(path))
    {
        return Some(path);
    }
    for (_, record) in builds::list_builds(data_dir, ENGINE) {
        let path = PathBuf::from(&record.binary);
        if path.is_file() {
            return Some(path);
        }
    }
    which_binary(BINARY_NAME).filter(|path| path.is_file())
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
    for (_, record) in builds::list_builds(data_dir, ENGINE) {
        let path = PathBuf::from(record.binary);
        if path.is_file() {
            candidates.push(path);
        }
    }
    if let Some(path_env) = path_env {
        for dir in std::env::split_paths(path_env) {
            let candidate = dir.join(BINARY_NAME);
            if candidate.is_file() {
                candidates.push(candidate);
            }
        }
    }
    candidates
}

/// Model specifier for the CLI: a CoreML directory, a `whisperkit:name` id, or a bare name.
#[derive(Debug, Clone)]
pub enum ModelSpec {
    /// Local CoreML model directory (`--model-path`).
    Path(PathBuf),
    /// Named variant downloaded on demand (`--model tiny|base|…`).
    Name(String),
}

/// Resolve how to pass the model to `whisperkit-cli`.
pub fn resolve_model_spec(data_dir: &Path, preferred: Option<&str>) -> ModelSpec {
    if let Some(value) = preferred.map(str::trim).filter(|value| !value.is_empty()) {
        if let Some(name) = value.strip_prefix("whisperkit:") {
            let name = name.trim();
            if !name.is_empty() {
                let as_path = models_root(data_dir).join(name);
                if as_path.is_dir() {
                    return ModelSpec::Path(as_path);
                }
                return ModelSpec::Name(name.to_owned());
            }
        }
        let as_path = PathBuf::from(value);
        if as_path.is_dir() {
            return ModelSpec::Path(as_path);
        }
        // Bare model names (tiny, base, small, large-v3, …) — not ggml paths.
        if !value.contains('/') && !value.ends_with(".bin") && !value.ends_with(".gguf") {
            return ModelSpec::Name(value.to_owned());
        }
    }
    ModelSpec::Name(DEFAULT_MODEL.to_owned())
}

pub struct TranscribeRequest<'a> {
    pub binary: &'a Path,
    pub data_dir: &'a Path,
    pub model: Option<&'a str>,
    pub audio: &'a Path,
    pub language: Option<&'a str>,
}

/// Run `whisperkit-cli transcribe` and return transcript text from stdout.
pub async fn transcribe(request: TranscribeRequest<'_>) -> anyhow::Result<String> {
    anyhow::ensure!(
        request.binary.is_file(),
        "whisperkit-cli binary missing: {}",
        request.binary.display()
    );
    anyhow::ensure!(request.audio.is_file(), "audio file missing");

    let model_dir = download_model_dir(request.data_dir);
    let tokenizer_dir = download_tokenizer_dir(request.data_dir);
    tokio::fs::create_dir_all(&model_dir)
        .await
        .context("create whisperkit model cache")?;
    tokio::fs::create_dir_all(&tokenizer_dir)
        .await
        .context("create whisperkit tokenizer cache")?;

    let mut command = Command::new(request.binary);
    command
        .arg("transcribe")
        .arg("--audio-path")
        .arg(request.audio)
        .arg("--download-model-path")
        .arg(&model_dir)
        .arg("--download-tokenizer-path")
        .arg(&tokenizer_dir)
        .arg("--skip-special-tokens")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    match resolve_model_spec(request.data_dir, request.model) {
        ModelSpec::Path(path) => {
            command.arg("--model-path").arg(path);
        }
        ModelSpec::Name(name) => {
            command.arg("--model").arg(name);
        }
    }
    if let Some(language) = request.language {
        command.arg("--language").arg(language);
    }

    let output = tokio::time::timeout(Duration::from_secs(900), command.output())
        .await
        .context("whisperkit-cli timed out")?
        .context("spawn whisperkit-cli")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("whisperkit-cli failed with {}: {stderr}", output.status);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let text = extract_transcript(&stdout);
    if text.is_empty() {
        // Some builds only print on stderr with --verbose; treat empty success as soft error.
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("whisperkit-cli produced no transcript text. stderr: {stderr}");
    }
    Ok(text)
}

/// Pull the spoken text out of CLI stdout (skip progress / status lines when present).
fn extract_transcript(stdout: &str) -> String {
    let lines: Vec<&str> = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    if lines.is_empty() {
        return String::new();
    }
    // Prefer lines after a common header if present.
    if let Some(index) = lines.iter().position(|line| {
        line.starts_with("Transcription") || line.eq_ignore_ascii_case("transcript:")
    }) {
        let rest: Vec<&str> = lines[index + 1..]
            .iter()
            .copied()
            .filter(|line| !line.starts_with('[') && !line.starts_with("Loading"))
            .collect();
        if !rest.is_empty() {
            return rest.join("\n").trim().to_owned();
        }
    }
    // Otherwise take the last non-status block (CLI often prints the text last).
    let filtered: Vec<&str> = lines
        .into_iter()
        .filter(|line| {
            let lower = line.to_ascii_lowercase();
            !lower.starts_with("loading")
                && !lower.starts_with("downloading")
                && !lower.starts_with("resolved")
                && !lower.starts_with("using ")
                && !lower.starts_with("starting")
                && !lower.starts_with("model:")
        })
        .collect();
    filtered.join("\n").trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_whisperkit_binary_names() {
        assert!(is_whisperkit_binary(Path::new("/opt/bin/whisperkit-cli")));
        assert!(is_whisperkit_binary(Path::new("argmax-cli")));
        assert!(!is_whisperkit_binary(Path::new("whisper-cli")));
    }

    #[test]
    fn model_spec_prefers_named_variants() {
        let dir = tempfile::tempdir().unwrap();
        match resolve_model_spec(dir.path(), Some("whisperkit:tiny")) {
            ModelSpec::Name(name) => assert_eq!(name, "tiny"),
            other => panic!("expected name, got {other:?}"),
        }
        match resolve_model_spec(dir.path(), None) {
            ModelSpec::Name(name) => assert_eq!(name, DEFAULT_MODEL),
            other => panic!("expected default name, got {other:?}"),
        }
    }

    #[test]
    fn model_spec_accepts_directory_paths() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("coreml-model");
        std::fs::create_dir_all(&model).unwrap();
        match resolve_model_spec(dir.path(), Some(model.to_str().unwrap())) {
            ModelSpec::Path(path) => assert_eq!(path, model),
            other => panic!("expected path, got {other:?}"),
        }
    }

    #[test]
    fn extract_transcript_skips_status_noise() {
        let out = "Loading model...\nDownloading weights\nHello from the mic.\n";
        assert_eq!(extract_transcript(out), "Hello from the mic.");
    }
}

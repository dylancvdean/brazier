//! Streaming ASR via a managed Python environment (Nemotron ASR Streaming).

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use anyhow::Context;
use serde::Deserialize;
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
    sync::mpsc,
};

use crate::{builds, models_store, types::ModelDescriptor};

pub const ENGINE: &str = "streaming-asr";

pub fn models_root(data_dir: &Path) -> PathBuf {
    data_dir.join("models").join("streaming-asr")
}

pub fn resolve_python(data_dir: &Path, override_path: Option<&str>) -> Option<PathBuf> {
    if let Some(path) = override_path
        .map(PathBuf::from)
        .filter(|path| path.is_file())
    {
        return Some(path);
    }
    for (_, record) in builds::list_builds(data_dir, ENGINE) {
        let path = PathBuf::from(&record.binary);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

pub fn python_appears_runnable(python: &Path) -> bool {
    if !python.is_file() {
        return false;
    }
    std::process::Command::new(python)
        .args(["-c", "import brazier_streaming_asr, transformers"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub fn model_id_for_repo(repo_id: &str) -> anyhow::Result<String> {
    models_store::validate_repo_id(repo_id)?;
    Ok(format!("{ENGINE}:{repo_id}"))
}

pub fn path_for_model_id(data_dir: &Path, model_id: &str) -> anyhow::Result<PathBuf> {
    let repo_id = model_id
        .strip_prefix(&format!("{ENGINE}:"))
        .ok_or_else(|| anyhow::anyhow!("not a streaming ASR model id: {model_id}"))?;
    models_store::validate_repo_id(repo_id)?;
    let path = models_root(data_dir).join(repo_id);
    anyhow::ensure!(
        path.is_dir() && directory_is_streaming_asr_model(&path),
        "streaming ASR model not found: {model_id}"
    );
    Ok(path)
}

pub fn download_root(data_dir: &Path, repo_id: &str) -> anyhow::Result<PathBuf> {
    models_store::validate_repo_id(repo_id)?;
    Ok(models_root(data_dir).join(repo_id))
}

pub fn download_destination(
    data_dir: &Path,
    repo_id: &str,
    filename: &str,
) -> anyhow::Result<PathBuf> {
    models_store::validate_repo_id(repo_id)?;
    // A Nemotron snapshot is `config.json`, a tokenizer, and `.safetensors`;
    // the GGUF filename rule rejected all of them and failed every download.
    models_store::validate_relative_path(filename)?;
    Ok(models_root(data_dir).join(repo_id).join(filename))
}

pub fn directory_is_streaming_asr_model(dir: &Path) -> bool {
    if !dir.is_dir() || !dir.join("config.json").is_file() {
        return false;
    }
    let has_weights = dir
        .read_dir()
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .to_ascii_lowercase()
                .ends_with(".safetensors")
        });
    if !has_weights {
        return false;
    }
    let Ok(config) = std::fs::read_to_string(dir.join("config.json")) else {
        return true;
    };
    let lower = config.to_ascii_lowercase();
    lower.contains("nemotron_asr")
        || lower.contains("rnnt")
        || looks_like_streaming_asr_repo(
            dir.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default(),
        )
}

pub fn looks_like_streaming_asr_repo(repo_or_name: &str) -> bool {
    let lower = repo_or_name.to_ascii_lowercase();
    lower.contains("nemotron-speech")
        || lower.contains("nemotron_speech")
        || lower.contains("nemotron-3.5-asr")
        || lower.contains("nemotron_3.5_asr")
        || lower.contains("asr-streaming")
        || lower.contains("asr_streaming")
        || (lower.contains("streaming") && lower.contains("asr"))
}

/// List on-disk streaming ASR snapshots.
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
            if !directory_is_streaming_asr_model(&model_dir.path()) {
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
                capabilities: crate::types::ModelCapabilities {
                    input_modalities: vec!["audio".into()],
                    output_modalities: vec!["text".into()],
                    streaming: true,
                    tools: false,
                    reasoning: false,
                    max_context_length: None,
                    reasoning_modes: Vec::new(),
                    harmony: false,
                    audio_input: None,
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

pub fn resolve_model_path(data_dir: &Path, preferred: Option<&str>) -> Option<PathBuf> {
    if let Some(id) = preferred {
        if let Ok(path) = path_for_model_id(data_dir, id) {
            return Some(path);
        }
        let as_path = PathBuf::from(id);
        if as_path.is_dir() && directory_is_streaming_asr_model(&as_path) {
            return Some(as_path);
        }
    }
    list_models(data_dir)
        .ok()?
        .into_iter()
        .next()
        .and_then(|model| path_for_model_id(data_dir, &model.id).ok())
}

pub fn detect_available(data_dir: &Path, python: Option<&str>, model: Option<&str>) -> bool {
    let Some(python) = resolve_python(data_dir, python) else {
        return false;
    };
    if !python_appears_runnable(&python) {
        return false;
    }
    resolve_model_path(data_dir, model).is_some()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerEvent {
    Status {
        phase: Option<String>,
        message: Option<String>,
        latency_ms: Option<u64>,
    },
    Delta {
        text: String,
    },
    Done {
        text: String,
    },
    Error {
        message: String,
    },
}

pub struct StreamTranscribeRequest<'a> {
    pub python: &'a Path,
    pub model: &'a Path,
    pub audio: &'a Path,
    pub lookahead: Option<u32>,
}

/// Spawn the Python worker and stream NDJSON events.
pub async fn transcribe_stream(
    request: StreamTranscribeRequest<'_>,
) -> anyhow::Result<mpsc::Receiver<anyhow::Result<WorkerEvent>>> {
    anyhow::ensure!(request.python.is_file(), "streaming ASR Python missing");
    anyhow::ensure!(
        request.model.is_dir(),
        "streaming ASR model directory missing"
    );
    anyhow::ensure!(request.audio.is_file(), "audio file missing");

    let mut command = Command::new(request.python);
    command
        .arg("-m")
        .arg("brazier_streaming_asr")
        .arg("--model")
        .arg(request.model)
        .arg("--audio")
        .arg(request.audio)
        .arg("--lookahead")
        .arg(request.lookahead.unwrap_or(6).to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = command.spawn().context("spawn streaming ASR worker")?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("worker stdout missing"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("worker stderr missing"))?;

    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        let mut stderr_lines = BufReader::new(stderr).lines();
        let mut stderr_buf = String::new();
        loop {
            tokio::select! {
                line = lines.next_line() => {
                    match line {
                        Ok(Some(line)) => {
                            if line.trim().is_empty() {
                                continue;
                            }
                            match serde_json::from_str::<WorkerEvent>(&line) {
                                Ok(event) => {
                                    if tx.send(Ok(event)).await.is_err() {
                                        let _ = child.kill().await;
                                        return;
                                    }
                                }
                                Err(error) => {
                                    let _ = tx
                                        .send(Err(anyhow::anyhow!(
                                            "invalid worker event: {error}; line={line}"
                                        )))
                                        .await;
                                    let _ = child.kill().await;
                                    return;
                                }
                            }
                        }
                        Ok(None) => break,
                        Err(error) => {
                            let _ = tx.send(Err(error.into())).await;
                            let _ = child.kill().await;
                            return;
                        }
                    }
                }
                line = stderr_lines.next_line() => {
                    if let Ok(Some(line)) = line {
                        if stderr_buf.len() < 8_192 {
                            if !stderr_buf.is_empty() {
                                stderr_buf.push('\n');
                            }
                            stderr_buf.push_str(&line);
                        }
                    }
                }
            }
        }
        match tokio::time::timeout(Duration::from_secs(30), child.wait()).await {
            Ok(Ok(status)) if status.success() => {}
            Ok(Ok(status)) => {
                let detail = if stderr_buf.is_empty() {
                    format!("streaming ASR worker exited with {status}")
                } else {
                    format!("streaming ASR worker exited with {status}: {stderr_buf}")
                };
                let _ = tx.send(Err(anyhow::anyhow!(detail))).await;
            }
            Ok(Err(error)) => {
                let _ = tx.send(Err(error.into())).await;
            }
            Err(_) => {
                let _ = child.kill().await;
                let _ = tx
                    .send(Err(anyhow::anyhow!(
                        "streaming ASR worker timed out after stdout closed"
                    )))
                    .await;
            }
        }
    });
    Ok(rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every file a Nemotron snapshot is made of, none of them GGUF. Applying
    /// the GGUF store's filename rule here failed the download on its first
    /// file, so the whole engine could never be given a model.
    #[test]
    fn accepts_the_files_a_snapshot_is_made_of() {
        let dir = tempfile::tempdir().unwrap();
        for file in [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "preprocessor_config.json",
            "nested/weights.safetensors",
        ] {
            assert!(
                download_destination(dir.path(), "nvidia/nemotron-3.5-asr-streaming-0.6b", file)
                    .is_ok(),
                "{file} must be downloadable"
            );
        }
    }

    #[test]
    fn still_refuses_paths_that_escape_the_store() {
        let dir = tempfile::tempdir().unwrap();
        for file in ["../escape.safetensors", "/abs.safetensors", "a/../../b", ""] {
            assert!(
                download_destination(dir.path(), "nvidia/nemotron", file).is_err(),
                "{file} must be refused"
            );
        }
        assert!(download_destination(dir.path(), "../evil", "config.json").is_err());
    }

    #[test]
    fn detects_nemotron_repo_names() {
        assert!(looks_like_streaming_asr_repo(
            "nvidia/nemotron-speech-streaming-en-0.6b"
        ));
        assert!(looks_like_streaming_asr_repo(
            "nemotron-3.5-asr-streaming-0.6b"
        ));
        assert!(!looks_like_streaming_asr_repo("mlx-community/Qwen2.5-0.5B"));
    }

    #[test]
    fn model_ids_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let root = models_root(dir.path()).join("nvidia/nemotron-speech-streaming-en-0.6b");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("config.json"),
            r#"{"model_type":"nemotron_asr_streaming"}"#,
        )
        .unwrap();
        std::fs::write(root.join("model.safetensors"), b"weights").unwrap();
        let id = model_id_for_repo("nvidia/nemotron-speech-streaming-en-0.6b").unwrap();
        assert_eq!(id, "streaming-asr:nvidia/nemotron-speech-streaming-en-0.6b");
        assert_eq!(path_for_model_id(dir.path(), &id).unwrap(), root);
        let models = list_models(dir.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert!(models[0].capabilities.streaming);
        assert_eq!(models[0].engine, ENGINE);
    }
}

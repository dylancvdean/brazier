//! Streaming ASR via a managed Python environment.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use anyhow::Context;
use serde::Deserialize;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::mpsc,
};

use crate::{builds, models_store, types::ModelDescriptor};

pub const ENGINE: &str = "streaming-asr";

/// `num_lookahead_tokens`: how much audio the decoder waits for before
/// committing a token. Six is roughly 560 ms.
pub const DEFAULT_LOOKAHEAD: u32 = 6;

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

/// A worker process kept alive across requests.
///
/// Loading Nemotron costs seconds, and it was being paid per utterance: every
/// spoken turn waited on a model that had been in memory moments earlier. The
/// process now stays resident and takes one request per line on stdin.
///
/// Requests are serialised by the mutex the caller holds it behind, which is
/// what the protocol needs anyway — one set of events per request, read until
/// `done` or `error`.
pub struct Worker {
    child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
    /// The model this process loaded. A different one needs a different process.
    model: PathBuf,
    python: PathBuf,
}

impl Worker {
    /// Start a worker and wait for it to report that the model is loaded.
    ///
    /// `package_dir` is put on `PYTHONPATH` so the worker source always matches
    /// the daemon that ships it, rather than whatever copy was installed into
    /// the virtualenv when the runtime was last built.
    pub async fn start(
        python: &Path,
        model: &Path,
        package_dir: &Path,
        lookahead: u32,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(python.is_file(), "streaming ASR Python missing");
        anyhow::ensure!(model.is_dir(), "streaming ASR model directory missing");

        let mut child = Command::new(python)
            .arg("-m")
            .arg("brazier_streaming_asr")
            .arg("--model")
            .arg(model)
            .arg("--serve")
            .arg("--lookahead")
            .arg(lookahead.to_string())
            .env("PYTHONPATH", package_dir)
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .context("spawn streaming ASR worker")?;

        let stdin = child.stdin.take().context("worker stdin missing")?;
        let stdout = child.stdout.take().context("worker stdout missing")?;
        let stderr = child.stderr.take().context("worker stderr missing")?;
        // Drained continuously: a full stderr pipe would otherwise block the
        // worker mid-request, and the lines are worth having in the log.
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(target: "streaming_asr", "{line}");
            }
        });

        let mut worker = Self {
            child,
            stdin,
            lines: BufReader::new(stdout).lines(),
            model: model.to_path_buf(),
            python: python.to_path_buf(),
        };
        worker.wait_until_ready().await?;
        Ok(worker)
    }

    /// Whether this worker can serve a request for `python` and `model`.
    pub fn serves(&mut self, python: &Path, model: &Path) -> bool {
        self.model == model && self.python == python && matches!(self.child.try_wait(), Ok(None))
    }

    async fn wait_until_ready(&mut self) -> anyhow::Result<()> {
        // Model load, so generous: a cold snapshot read is slow on any disk.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(300);
        loop {
            let line = tokio::time::timeout_at(deadline, self.lines.next_line())
                .await
                .context("streaming ASR worker did not become ready in time")?
                .context("read from streaming ASR worker")?
                .context("streaming ASR worker exited before becoming ready")?;
            match parse_event(&line)? {
                Some(WorkerEvent::Status { phase, .. }) if phase.as_deref() == Some("ready") => {
                    return Ok(());
                }
                Some(WorkerEvent::Error { message }) => anyhow::bail!(message),
                _ => continue,
            }
        }
    }

    /// Transcribe one file, forwarding events until the request completes.
    ///
    /// Returns the final text. Errors reported by the worker are request
    /// failures rather than worker failures, so the process is left running.
    pub async fn transcribe(
        &mut self,
        audio: &Path,
        lookahead: Option<u32>,
        events: &mpsc::Sender<anyhow::Result<WorkerEvent>>,
    ) -> anyhow::Result<String> {
        anyhow::ensure!(audio.is_file(), "audio file missing");
        let request = serde_json::json!({
            "audio": audio.display().to_string(),
            "lookahead": lookahead.unwrap_or(DEFAULT_LOOKAHEAD),
        });
        self.stdin
            .write_all(format!("{request}\n").as_bytes())
            .await
            .context("write request to streaming ASR worker")?;
        self.stdin
            .flush()
            .await
            .context("flush request to streaming ASR worker")?;

        loop {
            let line = self
                .lines
                .next_line()
                .await
                .context("read from streaming ASR worker")?
                .context("streaming ASR worker closed mid-request")?;
            let Some(event) = parse_event(&line)? else {
                continue;
            };
            let finished = match &event {
                WorkerEvent::Done { text } => Some(Ok(text.clone())),
                WorkerEvent::Error { message } => Some(Err(anyhow::anyhow!(message.clone()))),
                _ => None,
            };
            let _ = events.send(Ok(event)).await;
            match finished {
                Some(result) => return result,
                None => continue,
            }
        }
    }
}

/// Parse one NDJSON line, treating blank lines as nothing to report.
fn parse_event(line: &str) -> anyhow::Result<Option<WorkerEvent>> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    serde_json::from_str(line)
        .map(Some)
        .with_context(|| format!("invalid worker event: {line}"))
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

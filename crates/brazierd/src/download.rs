//! Hugging Face artifact download with resume, integrity hashing, and progress.

use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use anyhow::Context;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    fs::OpenOptions,
    io::{AsyncSeekExt, AsyncWriteExt},
};

use crate::{
    db::Database,
    hf_auth,
    mlx::MlxKind,
    models_store::{
        self, download_destination, downloads_dir, mlx_download_destination, mlx_model_id,
        model_id_for_path, validate_filename, validate_repo_id,
    },
    progress::{ProgressCallback, ProgressEvent},
    streaming_asr,
};

#[derive(Debug, Clone, Deserialize)]
pub struct DownloadRequest {
    pub repo_id: String,
    pub filename: String,
    #[serde(default = "default_revision")]
    pub revision: String,
    /// `llama.cpp` (default) or `whisper.cpp`.
    #[serde(default = "default_download_engine")]
    pub engine: String,
}

fn default_revision() -> String {
    "main".to_owned()
}

fn default_download_engine() -> String {
    "llama.cpp".to_owned()
}

#[derive(Debug, Clone, Deserialize)]
pub struct MlxDownloadRequest {
    pub repo_id: String,
    pub engine: String,
    #[serde(default = "default_revision")]
    pub revision: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadResult {
    pub model_id: String,
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub resumed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notice: Option<String>,
}

/// Content-addressed partial file for resumable downloads.
pub fn partial_path(data_dir: &Path, repo_id: &str, filename: &str) -> PathBuf {
    let safe_repo = repo_id.replace('/', "__");
    let safe_file = filename.replace('/', "__");
    downloads_dir(data_dir).join(format!("{safe_repo}__{safe_file}.partial"))
}

/// Build the Hugging Face resolve URL for a repository file.
pub fn resolve_url(repo_id: &str, revision: &str, filename: &str) -> String {
    format!("https://huggingface.co/{repo_id}/resolve/{revision}/{filename}")
}

fn noop_progress() -> ProgressCallback {
    Box::new(|_| {})
}

/// Download a GGUF file into the models store, resuming partial files when present.
pub async fn download_gguf(
    client: &reqwest::Client,
    data_dir: &Path,
    request: DownloadRequest,
) -> anyhow::Result<DownloadResult> {
    download_gguf_with_progress(client, data_dir, request, noop_progress(), None, None).await
}

pub async fn download_gguf_with_progress(
    client: &reqwest::Client,
    data_dir: &Path,
    request: DownloadRequest,
    mut progress: ProgressCallback,
    job: Option<(Database, String)>,
    cancel: Option<Arc<AtomicBool>>,
) -> anyhow::Result<DownloadResult> {
    validate_repo_id(&request.repo_id)?;
    validate_filename(&request.filename)?;
    anyhow::ensure!(
        !request.revision.is_empty()
            && request.revision.len() <= 200
            && request
                .revision
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '-' | '_' | '.')),
        "invalid revision"
    );

    let whisper = request.engine == "whisper.cpp";
    let destination = if whisper {
        crate::whisper::download_destination(data_dir, &request.repo_id, &request.filename)?
    } else {
        download_destination(data_dir, &request.repo_id, &request.filename)?
    };
    let engine_label = if whisper {
        "whisper.cpp"
    } else {
        "llama.cpp"
    };
    if destination.is_file() {
        progress(ProgressEvent::phase(
            "skip",
            "Model already present on disk",
        ));
        let bytes = tokio::fs::metadata(&destination).await?.len();
        progress(ProgressEvent::download(bytes, Some(bytes)));
        let sha256 = hash_file(&destination).await?;
        let model_id = if whisper {
            crate::whisper::model_id_for_path(&crate::whisper::whisper_root(data_dir), &destination)?
        } else {
            model_id_for_path(&crate::models_store::gguf_root(data_dir), &destination)?
        };
        let result = DownloadResult {
            model_id,
            path: destination.display().to_string(),
            bytes,
            sha256,
            resumed: false,
            engine: Some(engine_label.to_owned()),
            notice: None,
        };
        progress(ProgressEvent::done(serde_json::to_value(&result)?));
        if let Some((db, job_id)) = &job {
            let _ = db
                .complete_download_job(job_id, &result.sha256, result.bytes)
                .await;
        }
        return Ok(result);
    }

    if let Some(parent) = destination.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("create model directory")?;
    }
    tokio::fs::create_dir_all(downloads_dir(data_dir))
        .await
        .context("create downloads directory")?;

    let partial = partial_path(data_dir, &request.repo_id, &request.filename);
    let existing = if partial.is_file() {
        tokio::fs::metadata(&partial).await?.len()
    } else {
        0
    };
    let resumed = existing > 0;
    progress(ProgressEvent::phase(
        "start",
        if resumed {
            format!("Resuming download of {}", request.filename)
        } else {
            format!("Downloading {}", request.filename)
        },
    ));
    if let Some((db, job_id)) = &job {
        let _ = db.start_download_job(job_id).await;
    }
    let url = resolve_url(&request.repo_id, &request.revision, &request.filename);

    let mut builder = hf_auth::apply_auth(
        client
            .get(&url)
            .header(
                "user-agent",
                format!("brazier/{}", env!("CARGO_PKG_VERSION")),
            )
            .timeout(Duration::from_secs(600)),
        data_dir,
    );
    if existing > 0 {
        builder = builder.header("range", format!("bytes={existing}-"));
    }

    let response = builder
        .send()
        .await
        .context("start Hugging Face download")?;
    let status = response.status();
    if status.as_u16() == 416 {
        anyhow::bail!("server rejected resume range; delete the partial and retry");
    }
    if !(status.is_success() || status.as_u16() == 206) {
        let body = response.text().await.unwrap_or_default();
        if status.as_u16() == 401 || status.as_u16() == 403 {
            if hf_auth::load_token(data_dir).is_none() {
                anyhow::bail!(
                    "Hugging Face rejected the download ({status}). This model may be gated — add a Hugging Face token in Manage → Download models."
                );
            }
        }
        anyhow::bail!("Hugging Face download failed ({status}): {body}");
    }

    let total = content_length_total(&response, existing, status.as_u16() == 206);
    let append = status.as_u16() == 206 && existing > 0;
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(&partial)
        .await
        .context("open partial download")?;
    if append {
        file.seek(std::io::SeekFrom::End(0)).await?;
    }

    let mut stream = response.bytes_stream();
    let mut written = existing;
    let mut last_emit = 0_u64;
    progress(ProgressEvent::download(written, total));
    while let Some(chunk) = stream.next().await {
        if cancel
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            drop(file);
            let _ = tokio::fs::remove_file(&partial).await;
            anyhow::bail!("download cancelled");
        }
        let chunk = chunk.context("read download chunk")?;
        file.write_all(&chunk)
            .await
            .context("write download chunk")?;
        written += chunk.len() as u64;
        if written.saturating_sub(last_emit) >= 256 * 1024 || total == Some(written) {
            progress(ProgressEvent::download(written, total));
            if let Some((db, job_id)) = &job {
                let _ = db
                    .update_download_job_progress(job_id, written, total)
                    .await;
            }
            last_emit = written;
        }
    }
    progress(ProgressEvent::download(written, total.or(Some(written))));
    file.flush().await?;
    drop(file);

    progress(ProgressEvent::phase("hash", "Verifying SHA-256"));
    tokio::fs::rename(&partial, &destination)
        .await
        .context("promote partial download")?;
    let sha256 = hash_file(&destination).await?;
    let model_id = if whisper {
        crate::whisper::model_id_for_path(&crate::whisper::whisper_root(data_dir), &destination)?
    } else {
        model_id_for_path(&crate::models_store::gguf_root(data_dir), &destination)?
    };
    let result = DownloadResult {
        model_id,
        path: destination.display().to_string(),
        bytes: written,
        sha256,
        resumed,
        engine: Some(engine_label.to_owned()),
        notice: None,
    };
    progress(ProgressEvent::done(serde_json::to_value(&result)?));
    if let Some((db, job_id)) = &job {
        let _ = db
            .complete_download_job(job_id, &result.sha256, result.bytes)
            .await;
    }
    Ok(result)
}

fn content_length_total(
    response: &reqwest::Response,
    existing: u64,
    is_partial: bool,
) -> Option<u64> {
    if is_partial
        && let Some(range) = response.headers().get(reqwest::header::CONTENT_RANGE)
        && let Ok(value) = range.to_str()
        && let Some(total) = value.rsplit('/').next()
        && let Ok(total) = total.parse::<u64>()
    {
        return Some(total);
    }
    response.content_length().map(|length| {
        if is_partial {
            existing + length
        } else {
            length
        }
    })
}

async fn hash_file(path: &Path) -> anyhow::Result<String> {
    use tokio::io::AsyncReadExt;
    let mut file = tokio::fs::File::open(path)
        .await
        .context("open file for hashing")?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .await
            .context("read file for hashing")?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Hash bytes (used by tests and small fixture writes).
pub fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

async fn download_file_to(
    client: &reqwest::Client,
    data_dir: &Path,
    repo_id: &str,
    revision: &str,
    filename: &str,
    destination: &Path,
    progress: &mut ProgressCallback,
    job: Option<(&Database, &str)>,
    cancel: Option<&Arc<AtomicBool>>,
) -> anyhow::Result<u64> {
    if destination.is_file() {
        return Ok(tokio::fs::metadata(destination).await?.len());
    }
    if let Some(parent) = destination.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("create model directory")?;
    }
    tokio::fs::create_dir_all(downloads_dir(data_dir))
        .await
        .context("create downloads directory")?;

    let partial = partial_path(data_dir, repo_id, filename);
    let existing = if partial.is_file() {
        tokio::fs::metadata(&partial).await?.len()
    } else {
        0
    };
    progress(ProgressEvent::phase(
        "start",
        if existing > 0 {
            format!("Resuming download of {filename}")
        } else {
            format!("Downloading {filename}")
        },
    ));
    let url = resolve_url(repo_id, revision, filename);

    let mut builder = hf_auth::apply_auth(
        client
            .get(&url)
            .header(
                "user-agent",
                format!("brazier/{}", env!("CARGO_PKG_VERSION")),
            )
            .timeout(Duration::from_secs(600)),
        data_dir,
    );
    if existing > 0 {
        builder = builder.header("range", format!("bytes={existing}-"));
    }

    let response = builder
        .send()
        .await
        .context("start Hugging Face download")?;
    let status = response.status();
    if status.as_u16() == 416 {
        anyhow::bail!("server rejected resume range; delete the partial and retry");
    }
    if !(status.is_success() || status.as_u16() == 206) {
        let body = response.text().await.unwrap_or_default();
        if status.as_u16() == 401 || status.as_u16() == 403 {
            if hf_auth::load_token(data_dir).is_none() {
                anyhow::bail!(
                    "Hugging Face rejected the download ({status}). This model may be gated — add a Hugging Face token in Manage → Download models."
                );
            }
        }
        anyhow::bail!("Hugging Face download failed ({status}): {body}");
    }

    let total = content_length_total(&response, existing, status.as_u16() == 206);
    let append = status.as_u16() == 206 && existing > 0;
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(&partial)
        .await
        .context("open partial download")?;
    if append {
        file.seek(std::io::SeekFrom::End(0)).await?;
    }

    let mut stream = response.bytes_stream();
    let mut written = existing;
    let mut last_emit = 0_u64;
    progress(ProgressEvent::download(written, total));
    while let Some(chunk) = stream.next().await {
        if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            drop(file);
            let _ = tokio::fs::remove_file(&partial).await;
            anyhow::bail!("download cancelled");
        }
        let chunk = chunk.context("read download chunk")?;
        file.write_all(&chunk)
            .await
            .context("write download chunk")?;
        written += chunk.len() as u64;
        if written.saturating_sub(last_emit) >= 256 * 1024 || total == Some(written) {
            progress(ProgressEvent::download(written, total));
            if let Some((db, job_id)) = job {
                let _ = db
                    .update_download_job_progress(job_id, written, total)
                    .await;
            }
            last_emit = written;
        }
    }
    progress(ProgressEvent::download(written, total.or(Some(written))));
    file.flush().await?;
    drop(file);

    tokio::fs::rename(&partial, destination)
        .await
        .context("promote partial download")?;
    Ok(written)
}

/// Download the files required for a local MLX model snapshot.
pub async fn download_mlx_snapshot_with_progress(
    client: &reqwest::Client,
    data_dir: &Path,
    request: MlxDownloadRequest,
    mut progress: ProgressCallback,
    job: Option<(Database, String)>,
    cancel: Option<Arc<AtomicBool>>,
) -> anyhow::Result<DownloadResult> {
    validate_repo_id(&request.repo_id)?;
    anyhow::ensure!(
        !request.revision.is_empty()
            && request.revision.len() <= 200
            && request
                .revision
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '-' | '_' | '.')),
        "invalid revision"
    );
    let requested = MlxKind::from_engine_id(&request.engine)
        .ok_or_else(|| anyhow::anyhow!("unsupported MLX engine `{}`", request.engine))?;
    let files = crate::hf::list_mlx_snapshot_files(client, data_dir, &request.repo_id, &request.revision)
        .await?;
    anyhow::ensure!(
        !files.is_empty(),
        "no MLX snapshot files were found for {}",
        request.repo_id
    );
    progress(ProgressEvent::phase(
        "start",
        format!("Downloading {} MLX files for {}", files.len(), request.repo_id),
    ));
    if let Some((db, job_id)) = &job {
        let _ = db.start_download_job(job_id).await;
    }
    let mut total_bytes = 0_u64;
    for (index, file) in files.iter().enumerate() {
        if cancel.as_ref().is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            anyhow::bail!("download cancelled");
        }
        progress(ProgressEvent::phase(
            "download",
            format!("Fetching {} ({}/{})", file.path, index + 1, files.len()),
        ));
        let destination =
            mlx_download_destination(data_dir, &request.repo_id, &file.path)?;
        let bytes = download_file_to(
            client,
            data_dir,
            &request.repo_id,
            &request.revision,
            &file.path,
            &destination,
            &mut progress,
            job.as_ref().map(|(db, id)| (db, id.as_str())),
            cancel.as_ref(),
        )
        .await?;
        total_bytes += bytes;
    }
    let root = models_store::mlx_download_root(data_dir, &request.repo_id)?;
    let detected = models_store::detect_mlx_kind(&root);
    let notice = if detected != requested {
        Some(format!(
            "This model is {} (you selected {}). It was saved for {}.",
            detected.engine_id(),
            requested.engine_id(),
            detected.engine_id()
        ))
    } else {
        None
    };
    if let Some(message) = &notice {
        progress(ProgressEvent::phase("warning", message.clone()));
    }
    let model_id = mlx_model_id(detected, &request.repo_id)?;
    let sha256 = hash_directory(&root).await?;
    let result = DownloadResult {
        model_id: model_id.clone(),
        path: root.display().to_string(),
        bytes: total_bytes,
        sha256,
        resumed: false,
        engine: Some(detected.engine_id().to_owned()),
        notice,
    };
    progress(ProgressEvent::done(serde_json::to_value(&result)?));
    if let Some((db, job_id)) = &job {
        let _ = db
            .complete_download_job(job_id, &result.sha256, result.bytes)
            .await;
    }
    Ok(result)
}

/// Download a Hugging Face transformers snapshot for streaming ASR.
pub async fn download_streaming_asr_snapshot_with_progress(
    client: &reqwest::Client,
    data_dir: &Path,
    request: MlxDownloadRequest,
    mut progress: ProgressCallback,
    job: Option<(Database, String)>,
    cancel: Option<Arc<AtomicBool>>,
) -> anyhow::Result<DownloadResult> {
    validate_repo_id(&request.repo_id)?;
    anyhow::ensure!(
        request.engine == streaming_asr::ENGINE,
        "unsupported streaming ASR engine `{}`",
        request.engine
    );
    anyhow::ensure!(
        !request.revision.is_empty()
            && request.revision.len() <= 200
            && request
                .revision
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '-' | '_' | '.')),
        "invalid revision"
    );
    let files = crate::hf::list_mlx_snapshot_files(client, data_dir, &request.repo_id, &request.revision)
        .await?;
    anyhow::ensure!(
        !files.is_empty(),
        "no streaming ASR snapshot files were found for {}",
        request.repo_id
    );
    progress(ProgressEvent::phase(
        "start",
        format!(
            "Downloading {} streaming ASR files for {}",
            files.len(),
            request.repo_id
        ),
    ));
    if let Some((db, job_id)) = &job {
        let _ = db.start_download_job(job_id).await;
    }
    let mut total_bytes = 0_u64;
    for (index, file) in files.iter().enumerate() {
        if cancel.as_ref().is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            anyhow::bail!("download cancelled");
        }
        progress(ProgressEvent::phase(
            "download",
            format!("Fetching {} ({}/{})", file.path, index + 1, files.len()),
        ));
        let destination =
            streaming_asr::download_destination(data_dir, &request.repo_id, &file.path)?;
        let bytes = download_file_to(
            client,
            data_dir,
            &request.repo_id,
            &request.revision,
            &file.path,
            &destination,
            &mut progress,
            job.as_ref().map(|(db, id)| (db, id.as_str())),
            cancel.as_ref(),
        )
        .await?;
        total_bytes += bytes;
    }
    let root = streaming_asr::download_root(data_dir, &request.repo_id)?;
    anyhow::ensure!(
        streaming_asr::directory_is_streaming_asr_model(&root),
        "downloaded snapshot does not look like a streaming ASR model"
    );
    let model_id = streaming_asr::model_id_for_repo(&request.repo_id)?;
    let sha256 = hash_directory(&root).await?;
    let result = DownloadResult {
        model_id: model_id.clone(),
        path: root.display().to_string(),
        bytes: total_bytes,
        sha256,
        resumed: false,
        engine: Some(streaming_asr::ENGINE.to_owned()),
        notice: None,
    };
    progress(ProgressEvent::done(serde_json::to_value(&result)?));
    if let Some((db, job_id)) = &job {
        let _ = db
            .complete_download_job(job_id, &result.sha256, result.bytes)
            .await;
    }
    Ok(result)
}

async fn hash_directory(dir: &Path) -> anyhow::Result<String> {
    let mut entries = Vec::new();
    collect_files(dir, dir, &mut entries)?;
    entries.sort();
    let mut digest = Sha256::new();
    for path in entries {
        let relative = path.strip_prefix(dir).unwrap_or(&path);
        digest.update(relative.to_string_lossy().as_bytes());
        digest.update([0]);
        let bytes = tokio::fs::read(&path).await?;
        digest.update(bytes);
    }
    Ok(hex::encode(digest.finalize()))
}

fn collect_files(root: &Path, dir: &Path, files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, files)?;
        } else {
            files.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn resolve_url_is_stable() {
        assert_eq!(
            resolve_url("unsloth/Tiny", "main", "model.gguf"),
            "https://huggingface.co/unsloth/Tiny/resolve/main/model.gguf"
        );
    }

    #[test]
    fn partial_path_is_flat_and_safe() {
        let dir = tempdir().unwrap();
        let path = partial_path(dir.path(), "acme/demo", "x.gguf");
        assert!(path.starts_with(downloads_dir(dir.path())));
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .contains("acme__demo")
        );
    }

    #[tokio::test]
    async fn skips_redownload_when_file_exists() {
        let dir = tempdir().unwrap();
        let dest = download_destination(dir.path(), "acme/demo", "model.gguf").unwrap();
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        std::fs::write(&dest, b"fixture-weights").unwrap();
        let client = reqwest::Client::new();
        let result = download_gguf(
            &client,
            dir.path(),
            DownloadRequest {
                repo_id: "acme/demo".into(),
                filename: "model.gguf".into(),
                revision: "main".into(),
                engine: "llama.cpp".into(),
            },
        )
        .await
        .unwrap();
        assert_eq!(result.model_id, "gguf:acme/demo/model.gguf");
        assert_eq!(result.bytes, 15);
        assert_eq!(result.sha256, sha256_hex(b"fixture-weights"));
        assert!(!result.resumed);
    }
}

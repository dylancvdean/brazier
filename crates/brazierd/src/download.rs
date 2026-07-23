//! Hugging Face artifact download with resume, integrity hashing, and progress.

use std::{
    path::{Path, PathBuf},
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
    models_store::{
        download_destination, downloads_dir, model_id_for_path, validate_filename, validate_repo_id,
    },
    progress::{ProgressCallback, ProgressEvent},
};

#[derive(Debug, Clone, Deserialize)]
pub struct DownloadRequest {
    pub repo_id: String,
    pub filename: String,
    #[serde(default = "default_revision")]
    pub revision: String,
}

fn default_revision() -> String {
    "main".to_owned()
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadResult {
    pub model_id: String,
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub resumed: bool,
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
    download_gguf_with_progress(client, data_dir, request, noop_progress()).await
}

pub async fn download_gguf_with_progress(
    client: &reqwest::Client,
    data_dir: &Path,
    request: DownloadRequest,
    mut progress: ProgressCallback,
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

    let destination = download_destination(data_dir, &request.repo_id, &request.filename)?;
    if destination.is_file() {
        progress(ProgressEvent::phase(
            "skip",
            "Model already present on disk",
        ));
        let bytes = tokio::fs::metadata(&destination).await?.len();
        progress(ProgressEvent::download(bytes, Some(bytes)));
        let sha256 = hash_file(&destination).await?;
        let model_id = model_id_for_path(&crate::models_store::gguf_root(data_dir), &destination)?;
        let result = DownloadResult {
            model_id,
            path: destination.display().to_string(),
            bytes,
            sha256,
            resumed: false,
        };
        progress(ProgressEvent::done(serde_json::to_value(&result)?));
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
    let url = resolve_url(&request.repo_id, &request.revision, &request.filename);

    let mut builder = client
        .get(&url)
        .header(
            "user-agent",
            format!("brazier/{}", env!("CARGO_PKG_VERSION")),
        )
        .timeout(Duration::from_secs(600));
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
        let chunk = chunk.context("read download chunk")?;
        file.write_all(&chunk)
            .await
            .context("write download chunk")?;
        written += chunk.len() as u64;
        // Throttle UI updates to roughly every 256 KiB.
        if written.saturating_sub(last_emit) >= 256 * 1024 || total == Some(written) {
            progress(ProgressEvent::download(written, total));
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
    let model_id = model_id_for_path(&crate::models_store::gguf_root(data_dir), &destination)?;
    let result = DownloadResult {
        model_id,
        path: destination.display().to_string(),
        bytes: written,
        sha256,
        resumed,
    };
    progress(ProgressEvent::done(serde_json::to_value(&result)?));
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

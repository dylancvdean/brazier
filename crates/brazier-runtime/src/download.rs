//! Hugging Face artifact download with resume, integrity hashing, and progress.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
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
    active_downloads::{StopFlag, StopReason},
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// How long to wait for the response headers before giving up on a transfer.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(120);

/// How long a transfer may receive nothing before it is called stalled.
///
/// This is deliberately an *idle* timeout rather than a deadline on the whole
/// request. A multi-gigabyte model legitimately takes hours on a slow link, and
/// a total timeout kills it mid-stream — which surfaced as a download that
/// stalled with "read download chunk" and could only be nudged along by
/// starting it again.
const IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// Await the response headers, failing with a clear message on a hung connect.
async fn send_with_connect_timeout(
    builder: reqwest::RequestBuilder,
    what: &str,
) -> anyhow::Result<reqwest::Response> {
    match tokio::time::timeout(CONNECT_TIMEOUT, builder.send()).await {
        Ok(result) => result.with_context(|| what.to_owned()),
        Err(_) => anyhow::bail!(
            "{what}: no response after {}s. Check your network connection and retry.",
            CONNECT_TIMEOUT.as_secs()
        ),
    }
}

/// Pull the next chunk of a body, failing when the connection goes quiet.
///
/// Returning `Ok(None)` means the body ended normally.
async fn next_chunk<S, B>(stream: &mut S, downloaded: u64) -> anyhow::Result<Option<B>>
where
    S: futures::Stream<Item = reqwest::Result<B>> + Unpin,
{
    match tokio::time::timeout(IDLE_TIMEOUT, stream.next()).await {
        Ok(Some(chunk)) => Ok(Some(chunk.context("read download chunk")?)),
        Ok(None) => Ok(None),
        Err(_) => anyhow::bail!(
            "download stalled: nothing received for {}s after {}. It can be resumed from here.",
            IDLE_TIMEOUT.as_secs(),
            format_bytes_short(downloaded)
        ),
    }
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
    cancel: Option<Arc<StopFlag>>,
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
    let engine_label = if whisper { "whisper.cpp" } else { "llama.cpp" };
    if destination.is_file() {
        let bytes = tokio::fs::metadata(&destination).await?.len();
        if looks_like_lfs_pointer(&destination).await {
            let _ = tokio::fs::remove_file(&destination).await;
        } else {
            progress(ProgressEvent::phase(
                "skip",
                "Model already present on disk",
            ));
            progress(ProgressEvent::download(bytes, Some(bytes)));
            let sha256 = hash_file(&destination).await?;
            let model_id = if whisper {
                crate::whisper::model_id_for_path(
                    &crate::whisper::whisper_root(data_dir),
                    &destination,
                )?
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
    if existing > 0 && looks_like_lfs_pointer(&partial).await {
        let _ = tokio::fs::remove_file(&partial).await;
    }
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
        client.get(&url).header(
            "user-agent",
            format!("brazier/{}", env!("CARGO_PKG_VERSION")),
        ),
        data_dir,
    );
    if existing > 0 {
        builder = builder.header("range", format!("bytes={existing}-"));
    }

    let response = send_with_connect_timeout(builder, "start Hugging Face download").await?;
    let status = response.status();
    if status.as_u16() == 416 {
        anyhow::bail!("server rejected resume range; delete the partial and retry");
    }
    if !(status.is_success() || status.as_u16() == 206) {
        let body = response.text().await.unwrap_or_default();
        if status.as_u16() == 401 || status.as_u16() == 403 {
            if hf_auth::load_token(data_dir).is_none() {
                anyhow::bail!(
                    "Hugging Face rejected the download ({status}). This model may be gated — add a Hugging Face token in Manage → Download models, accept the model license on the Hub, and retry."
                );
            }
            anyhow::bail!(
                "Hugging Face rejected the download ({status}). Confirm your token can access {} and that you accepted the model license on the Hub.",
                request.repo_id
            );
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
    while let Some(chunk) = next_chunk(&mut stream, written).await? {
        if let Some(reason) = cancel.as_ref().and_then(|flag| flag.reason()) {
            // Flush before letting go so a paused transfer resumes from every
            // byte it actually received; only a cancel discards the partial.
            let _ = file.flush().await;
            drop(file);
            if reason == StopReason::Cancel {
                let _ = tokio::fs::remove_file(&partial).await;
                anyhow::bail!("download cancelled");
            }
            anyhow::bail!("download paused");
        }
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

    if looks_like_lfs_pointer(&partial).await {
        let _ = tokio::fs::remove_file(&partial).await;
        anyhow::bail!(
            "downloaded an LFS pointer for {} instead of the real file. Authenticate with a Hugging Face token that has access to {}.",
            request.filename,
            request.repo_id
        );
    }

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

async fn sha256_matches(path: &Path, expected: Option<&str>) -> anyhow::Result<bool> {
    match expected {
        Some(expected) => Ok(hash_file(path).await? == expected),
        None => Ok(true),
    }
}

/// Hash bytes (used by tests and small fixture writes).
pub fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

/// True when a file looks like a Git LFS pointer (tiny text) rather than real weights.
async fn looks_like_lfs_pointer(path: &Path) -> bool {
    use tokio::io::AsyncReadExt;
    let Ok(mut file) = tokio::fs::File::open(path).await else {
        return false;
    };
    let mut buf = vec![0_u8; 80];
    let Ok(read) = file.read(&mut buf).await else {
        return false;
    };
    std::str::from_utf8(&buf[..read])
        .map(|text| text.starts_with("version https://git-lfs.github.com/spec/v1"))
        .unwrap_or(false)
}

/// Whether an on-disk file matches the Hub's expected size, when available.
fn size_matches_expected(actual: u64, expected: Option<u64>) -> bool {
    let Some(expected) = expected.filter(|value| *value > 0) else {
        return actual > 0;
    };
    // The Hub tree API reports the exact byte count. Do not accept a partially
    // downloaded file as complete just because it is close in size.
    actual == expected
}

/// Optional multi-file progress context so UI shows overall snapshot progress,
/// not 100% of the current small file only.
#[derive(Clone, Copy)]
struct SnapshotProgressCtx<'a> {
    completed_before: u64,
    overall_total: Option<u64>,
    file_index: usize,
    file_count: usize,
    file_path: &'a str,
}

struct FileDownload<'a> {
    client: &'a reqwest::Client,
    data_dir: &'a Path,
    repo_id: &'a str,
    revision: &'a str,
    filename: &'a str,
    source_url: Option<&'a str>,
    destination: &'a Path,
    job: Option<(&'a Database, &'a str)>,
    cancel: Option<&'a Arc<StopFlag>>,
    expected_size: Option<u64>,
    expected_sha256: Option<&'a str>,
    snapshot: Option<SnapshotProgressCtx<'a>>,
}

fn emit_file_download_progress(
    progress: &mut ProgressCallback,
    snapshot: Option<SnapshotProgressCtx<'_>>,
    file_bytes: u64,
    file_total: Option<u64>,
    job: Option<(&Database, &str)>,
) {
    if let Some(ctx) = snapshot {
        let overall_bytes = ctx.completed_before.saturating_add(file_bytes);
        let mut event = ProgressEvent::download(overall_bytes, ctx.overall_total.or(file_total));
        let file_pct = file_total
            .filter(|total| *total > 0)
            .map(|total| ((file_bytes as f64 / total as f64) * 100.0).round() as u64)
            .unwrap_or(0);
        event.message = Some(format!(
            "File {}/{} · {} · {}{}",
            ctx.file_index + 1,
            ctx.file_count,
            ctx.file_path,
            format_bytes_short(file_bytes),
            file_total
                .map(|total| format!(" / {} (file {file_pct}%)", format_bytes_short(total)))
                .unwrap_or_default()
        ));
        progress(event);
        if let Some((db, job_id)) = job {
            // Fire-and-forget progress for queue jobs (best-effort).
            let db = db.clone();
            let job_id = job_id.to_owned();
            tokio::spawn(async move {
                let _ = db
                    .update_download_job_progress(&job_id, overall_bytes, ctx.overall_total)
                    .await;
            });
        }
    } else {
        progress(ProgressEvent::download(file_bytes, file_total));
        if let Some((db, job_id)) = job {
            let db = db.clone();
            let job_id = job_id.to_owned();
            tokio::spawn(async move {
                let _ = db
                    .update_download_job_progress(&job_id, file_bytes, file_total)
                    .await;
            });
        }
    }
}

fn format_bytes_short(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let value = bytes as f64;
    if value >= GB {
        format!("{:.2} GB", value / GB)
    } else if value >= MB {
        format!("{:.1} MB", value / MB)
    } else if value >= KB {
        format!("{:.0} KB", value / KB)
    } else {
        format!("{bytes} B")
    }
}

async fn download_file_to_with_opts(
    request: FileDownload<'_>,
    progress: &mut ProgressCallback,
) -> anyhow::Result<u64> {
    let FileDownload {
        client,
        data_dir,
        repo_id,
        revision,
        filename,
        source_url,
        destination,
        job,
        cancel,
        expected_size,
        expected_sha256,
        snapshot,
    } = request;
    if destination.is_file() {
        let len = tokio::fs::metadata(destination).await?.len();
        let pointer = looks_like_lfs_pointer(destination).await;
        if !pointer && size_matches_expected(len, expected_size) {
            let checksum_matches = if expected_sha256.is_some() {
                progress(ProgressEvent::phase(
                    "hash",
                    format!("Verifying {filename}"),
                ));
                sha256_matches(destination, expected_sha256).await?
            } else {
                true
            };
            if checksum_matches {
                if let Some(ctx) = snapshot {
                    emit_file_download_progress(
                        progress,
                        Some(ctx),
                        len,
                        expected_size.or(Some(len)),
                        job,
                    );
                }
                return Ok(len);
            }
        }
        // Stale LFS pointer, truncated copy, or complete-but-corrupt file.
        let _ = tokio::fs::remove_file(destination).await;
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
    // Partial that is already "complete" for expected size may still be an LFS pointer.
    if existing > 0 && looks_like_lfs_pointer(&partial).await {
        let _ = tokio::fs::remove_file(&partial).await;
    }
    let existing = if partial.is_file() {
        tokio::fs::metadata(&partial).await?.len()
    } else {
        0
    };
    if let Some(expected) = expected_size.filter(|size| *size > 0) {
        if existing == expected && !looks_like_lfs_pointer(&partial).await {
            let checksum_matches = if expected_sha256.is_some() {
                progress(ProgressEvent::phase(
                    "hash",
                    format!("Verifying {filename}"),
                ));
                sha256_matches(&partial, expected_sha256).await?
            } else {
                true
            };
            if checksum_matches {
                // A previous attempt may have finished writing this file but
                // been interrupted before the final rename.
                tokio::fs::rename(&partial, destination)
                    .await
                    .context("promote completed partial download")?;
                emit_file_download_progress(progress, snapshot, existing, Some(expected), job);
                return Ok(existing);
            }
            let _ = tokio::fs::remove_file(&partial).await;
        }
        if existing > expected {
            // A partial from a different revision cannot be safely resumed.
            let _ = tokio::fs::remove_file(&partial).await;
        }
    }
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
    let url = source_url
        .map(str::to_owned)
        .unwrap_or_else(|| resolve_url(repo_id, revision, filename));

    let request = client.get(&url).header(
        "user-agent",
        format!("brazier/{}", env!("CARGO_PKG_VERSION")),
    );
    // Hugging Face credentials must never accompany an explicitly configured
    // third-party source (such as an upstream GitHub release).
    let mut builder = if source_url.is_some() {
        request
    } else {
        hf_auth::apply_auth(request, data_dir)
    };
    if existing > 0 {
        builder = builder.header("range", format!("bytes={existing}-"));
    }

    let response = send_with_connect_timeout(builder, "start Hugging Face download").await?;
    let status = response.status();
    if status.as_u16() == 416 {
        anyhow::bail!("server rejected resume range; delete the partial and retry");
    }
    if !(status.is_success() || status.as_u16() == 206) {
        let body = response.text().await.unwrap_or_default();
        if status.as_u16() == 401 || status.as_u16() == 403 {
            if hf_auth::load_token(data_dir).is_none() {
                anyhow::bail!(
                    "Hugging Face rejected the download ({status}). This model may be gated — add a Hugging Face token in Manage → Download models, accept the model license on the Hub, and retry."
                );
            }
            anyhow::bail!(
                "Hugging Face rejected the download ({status}). Confirm your token can access {repo_id} and that you accepted the model license on the Hub."
            );
        }
        anyhow::bail!("Hugging Face download failed ({status}): {body}");
    }

    let total = content_length_total(&response, existing, status.as_u16() == 206).or(expected_size);
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
    emit_file_download_progress(progress, snapshot, written, total, job);
    while let Some(chunk) = next_chunk(&mut stream, written).await? {
        if let Some(reason) = cancel.and_then(|flag| flag.reason()) {
            let _ = file.flush().await;
            drop(file);
            if reason == StopReason::Cancel {
                let _ = tokio::fs::remove_file(&partial).await;
                anyhow::bail!("download cancelled");
            }
            anyhow::bail!("download paused");
        }
        file.write_all(&chunk)
            .await
            .context("write download chunk")?;
        written += chunk.len() as u64;
        if written.saturating_sub(last_emit) >= 256 * 1024 || total == Some(written) {
            emit_file_download_progress(progress, snapshot, written, total, job);
            last_emit = written;
        }
    }
    emit_file_download_progress(progress, snapshot, written, total.or(Some(written)), job);
    file.flush().await?;
    drop(file);

    if looks_like_lfs_pointer(&partial).await {
        let _ = tokio::fs::remove_file(&partial).await;
        anyhow::bail!(
            "downloaded an LFS pointer for {filename} instead of the real file. \
             Authenticate with a Hugging Face token that has access to {repo_id}."
        );
    }
    if !size_matches_expected(written, expected_size.or(total)) {
        let _ = tokio::fs::remove_file(&partial).await;
        anyhow::bail!(
            "download of {filename} finished at {} bytes but ~{} were expected — retry after checking Hub access",
            written,
            expected_size.or(total).unwrap_or(0)
        );
    }
    if let Some(expected) = expected_sha256 {
        progress(ProgressEvent::phase(
            "hash",
            format!("Verifying {filename}"),
        ));
        let actual = hash_file(&partial).await?;
        if actual != expected {
            let _ = tokio::fs::remove_file(&partial).await;
            anyhow::bail!(
                "download of {filename} failed SHA-256 verification (expected {expected}, got {actual})"
            );
        }
    }

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
    cancel: Option<Arc<StopFlag>>,
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
    let files =
        crate::hf::list_mlx_snapshot_files(client, data_dir, &request.repo_id, &request.revision)
            .await?;
    anyhow::ensure!(
        !files.is_empty(),
        "no MLX snapshot files were found for {}",
        request.repo_id
    );
    progress(ProgressEvent::phase(
        "start",
        format!(
            "Downloading {} MLX files for {}",
            files.len(),
            request.repo_id
        ),
    ));
    if let Some((db, job_id)) = &job {
        let _ = db.start_download_job(job_id).await;
    }
    let overall_total = files.iter().try_fold(0_u64, |total, file| {
        file.size.and_then(|size| total.checked_add(size))
    });
    let mut total_bytes = 0_u64;
    for (index, file) in files.iter().enumerate() {
        if cancel.as_ref().is_some_and(|flag| flag.should_stop()) {
            anyhow::bail!("download cancelled");
        }
        progress(ProgressEvent::phase(
            "download",
            format!("Fetching {} ({}/{})", file.path, index + 1, files.len()),
        ));
        let destination = mlx_download_destination(data_dir, &request.repo_id, &file.path)?;
        let bytes = download_file_to_with_opts(
            FileDownload {
                client,
                data_dir,
                repo_id: &request.repo_id,
                revision: &request.revision,
                filename: &file.path,
                source_url: None,
                destination: &destination,
                job: job.as_ref().map(|(db, id)| (db, id.as_str())),
                cancel: cancel.as_ref(),
                expected_size: file.size,
                expected_sha256: file.sha256.as_deref(),
                snapshot: Some(SnapshotProgressCtx {
                    completed_before: total_bytes,
                    overall_total,
                    file_index: index,
                    file_count: files.len(),
                    file_path: &file.path,
                }),
            },
            &mut progress,
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
    let sha256 = snapshot_manifest_sha256(&files);
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
    cancel: Option<Arc<StopFlag>>,
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
    let files =
        crate::hf::list_mlx_snapshot_files(client, data_dir, &request.repo_id, &request.revision)
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
    let overall_total = files.iter().try_fold(0_u64, |total, file| {
        file.size.and_then(|size| total.checked_add(size))
    });
    let mut total_bytes = 0_u64;
    for (index, file) in files.iter().enumerate() {
        if cancel.as_ref().is_some_and(|flag| flag.should_stop()) {
            anyhow::bail!("download cancelled");
        }
        progress(ProgressEvent::phase(
            "download",
            format!("Fetching {} ({}/{})", file.path, index + 1, files.len()),
        ));
        let destination =
            streaming_asr::download_destination(data_dir, &request.repo_id, &file.path)?;
        let bytes = download_file_to_with_opts(
            FileDownload {
                client,
                data_dir,
                repo_id: &request.repo_id,
                revision: &request.revision,
                filename: &file.path,
                source_url: None,
                destination: &destination,
                job: job.as_ref().map(|(db, id)| (db, id.as_str())),
                cancel: cancel.as_ref(),
                expected_size: file.size,
                expected_sha256: file.sha256.as_deref(),
                snapshot: Some(SnapshotProgressCtx {
                    completed_before: total_bytes,
                    overall_total,
                    file_index: index,
                    file_count: files.len(),
                    file_path: &file.path,
                }),
            },
            &mut progress,
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
    let sha256 = snapshot_manifest_sha256(&files);
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

/// Install a curated stable-diffusion.cpp bundle: every component file plus
/// the manifest that binds them to sd-cli flags.
///
/// Components come from several repositories — a Flux install pulls its VAE
/// and both text encoders from two other repos — so files are fetched
/// individually rather than as one snapshot, and the manifest is written last
/// so a cancelled install leaves nothing the model list would pick up.
/// Download one LoRA or ControlNet file into the adapter library.
///
/// A single file rather than a snapshot: adapters are published one weight file
/// per variant, and pulling the whole repository would fetch every variant of
/// something meant to be a few hundred megabytes.
pub struct AdapterDownload<'a> {
    pub data_dir: &'a Path,
    pub kind: crate::adapters::AdapterKind,
    pub repo_id: &'a str,
    pub revision: &'a str,
    pub filename: &'a str,
    pub cancel: Option<Arc<StopFlag>>,
}

pub async fn download_adapter_with_progress(
    client: &reqwest::Client,
    request: AdapterDownload<'_>,
    mut progress: ProgressCallback,
) -> anyhow::Result<DownloadResult> {
    let AdapterDownload {
        data_dir,
        kind,
        repo_id,
        revision,
        filename,
        cancel,
    } = request;
    validate_repo_id(repo_id)?;
    anyhow::ensure!(
        !revision.is_empty()
            && revision.len() <= 200
            && revision
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '-' | '_' | '.')),
        "invalid revision"
    );
    let destination = crate::adapters::download_destination(data_dir, kind, repo_id, filename)?;
    progress(ProgressEvent::phase(
        "start",
        format!("Downloading {filename} from {repo_id}"),
    ));
    let expected =
        crate::hf::paths_info(client, data_dir, repo_id, revision, &[filename.to_owned()])
            .await
            .ok()
            .and_then(|infos| infos.into_iter().next());
    let bytes = download_file_to_with_opts(
        FileDownload {
            client,
            data_dir,
            repo_id,
            revision,
            filename,
            source_url: None,
            destination: &destination,
            job: None,
            cancel: cancel.as_ref(),
            expected_size: expected.as_ref().and_then(|info| info.size),
            expected_sha256: expected.as_ref().and_then(|info| info.sha256.as_deref()),
            snapshot: None,
        },
        &mut progress,
    )
    .await?;
    let sha256 = hash_file(&destination).await?;
    let result = DownloadResult {
        model_id: format!(
            "{}:{repo_id}/{}",
            kind.as_str(),
            destination
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        ),
        path: destination.display().to_string(),
        bytes,
        sha256,
        resumed: false,
        engine: None,
        notice: None,
    };
    progress(ProgressEvent::done(serde_json::to_value(&result)?));
    Ok(result)
}

pub async fn install_sdcpp_bundle_with_progress(
    client: &reqwest::Client,
    data_dir: &Path,
    bundle: &crate::sdcpp_catalog::Bundle,
    mut progress: ProgressCallback,
    job: Option<(Database, String)>,
    cancel: Option<Arc<StopFlag>>,
) -> anyhow::Result<DownloadResult> {
    let dir = bundle.install_dir(data_dir)?;
    progress(ProgressEvent::phase(
        "start",
        format!(
            "Installing {} — {} files including text encoders",
            bundle.label,
            bundle.components.len()
        ),
    ));
    if let Some((db, job_id)) = &job {
        let _ = db.start_download_job(job_id).await;
    }

    // Resolve real sizes from the Hub: the catalog's figures are estimates for
    // the install summary. LFS checksums catch complete files whose bytes are
    // wrong even when their length happens to match.
    let mut sizes: HashMap<(String, String), Option<u64>> = HashMap::new();
    let mut hashes: HashMap<(String, String), Option<String>> = HashMap::new();
    for component in &bundle.components {
        if let Some(source_url) = &component.source_url {
            let url = reqwest::Url::parse(source_url)
                .with_context(|| format!("invalid direct component URL for {}", component.role))?;
            anyhow::ensure!(url.scheme() == "https", "direct component URL must use HTTPS");
            anyhow::ensure!(
                component.source_size.is_some() && component.source_sha256.is_some(),
                "direct component {} must pin its size and SHA-256",
                component.role
            );
            let key = (component.repo_id.clone(), component.path.clone());
            sizes.insert(key.clone(), component.source_size);
            hashes.insert(key, component.source_sha256.clone());
            continue;
        }
        validate_repo_id(&component.repo_id)?;
        if sizes.contains_key(&(component.repo_id.clone(), component.path.clone())) {
            continue;
        }
        let paths: Vec<String> = bundle
            .components
            .iter()
            .filter(|other| other.repo_id == component.repo_id)
            .map(|other| other.path.clone())
            .collect();
        let infos = crate::hf::paths_info(client, data_dir, &component.repo_id, "main", &paths)
            .await
            .with_context(|| format!("look up files in {}", component.repo_id))?;
        for path in &paths {
            let info = infos
                .iter()
                .find(|info| &info.path == path)
                .with_context(|| format!("{path} is missing from {}", component.repo_id))?;
            let key = (component.repo_id.clone(), path.clone());
            sizes.insert(key.clone(), info.size);
            hashes.insert(key, info.sha256.clone());
        }
    }

    let overall_total = bundle
        .components
        .iter()
        .try_fold(0_u64, |total, component| {
            sizes
                .get(&(component.repo_id.clone(), component.path.clone()))
                .copied()
                .flatten()
                .and_then(|size| total.checked_add(size))
        });
    let mut total_bytes = 0_u64;
    let count = bundle.components.len();
    for (index, component) in bundle.components.iter().enumerate() {
        if cancel.as_ref().is_some_and(|flag| flag.should_stop()) {
            anyhow::bail!("install cancelled");
        }
        progress(ProgressEvent::phase(
            "download",
            format!("{} ({}/{})", component.role, index + 1, count),
        ));
        let destination = crate::sdcpp::component_destination(
            data_dir,
            bundle.modality,
            &bundle.key,
            component.file_name(),
        )?;
        let bytes = download_file_to_with_opts(
            FileDownload {
                client,
                data_dir,
                repo_id: &component.repo_id,
                revision: "main",
                filename: &component.path,
                source_url: component.source_url.as_deref(),
                destination: &destination,
                job: job.as_ref().map(|(db, id)| (db, id.as_str())),
                cancel: cancel.as_ref(),
                expected_size: sizes
                    .get(&(component.repo_id.clone(), component.path.clone()))
                    .copied()
                    .flatten(),
                expected_sha256: hashes
                    .get(&(component.repo_id.clone(), component.path.clone()))
                    .and_then(|hash| hash.as_deref()),
                snapshot: Some(SnapshotProgressCtx {
                    completed_before: total_bytes,
                    overall_total,
                    file_index: index,
                    file_count: count,
                    file_path: &component.path,
                }),
            },
            &mut progress,
        )
        .await?;
        total_bytes += bytes;
    }

    progress(ProgressEvent::phase(
        "verify",
        "Writing the sd-cli manifest",
    ));
    crate::sdcpp::write_manifest(&dir, &bundle.manifest()).await?;

    let result = DownloadResult {
        model_id: bundle.model_id(),
        path: dir.display().to_string(),
        bytes: total_bytes,
        sha256: bundle_manifest_sha256(bundle),
        resumed: false,
        engine: Some("stable-diffusion.cpp".to_owned()),
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

/// Stable identity for an installed bundle, derived from the exact component
/// list it was built from rather than from the downloaded bytes.
fn bundle_manifest_sha256(bundle: &crate::sdcpp_catalog::Bundle) -> String {
    let mut digest = Sha256::new();
    digest.update(b"brazier-sdcpp-bundle-v1\0");
    digest.update(bundle.id.as_bytes());
    for component in &bundle.components {
        digest.update(b"\0");
        digest.update(component.repo_id.as_bytes());
        digest.update(b"\0");
        digest.update(component.path.as_bytes());
    }
    hex::encode(digest.finalize())
}

/// Download a Hugging Face snapshot for PersonaPlex / Moshi.
pub async fn download_personaplex_snapshot_with_progress(
    client: &reqwest::Client,
    data_dir: &Path,
    request: MlxDownloadRequest,
    mut progress: ProgressCallback,
    job: Option<(Database, String)>,
    cancel: Option<Arc<StopFlag>>,
) -> anyhow::Result<DownloadResult> {
    validate_repo_id(&request.repo_id)?;
    anyhow::ensure!(
        request.engine == crate::voice::ENGINE,
        "unsupported PersonaPlex engine `{}`",
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
    let files = crate::hf::list_personaplex_snapshot_files(
        client,
        data_dir,
        &request.repo_id,
        &request.revision,
    )
    .await?;
    anyhow::ensure!(
        !files.is_empty(),
        "no PersonaPlex snapshot files were found for {}",
        request.repo_id
    );
    progress(ProgressEvent::phase(
        "start",
        format!(
            "Downloading {} PersonaPlex files for {}",
            files.len(),
            request.repo_id
        ),
    ));
    if let Some((db, job_id)) = &job {
        let _ = db.start_download_job(job_id).await;
    }
    let overall_total = files.iter().try_fold(0_u64, |total, file| {
        file.size.and_then(|size| total.checked_add(size))
    });
    let mut total_bytes = 0_u64;
    for (index, file) in files.iter().enumerate() {
        if cancel.as_ref().is_some_and(|flag| flag.should_stop()) {
            anyhow::bail!("download cancelled");
        }
        progress(ProgressEvent::phase(
            "download",
            format!("Fetching {} ({}/{})", file.path, index + 1, files.len()),
        ));
        let destination =
            crate::voice::download_destination(data_dir, &request.repo_id, &file.path)?;
        let bytes = download_file_to_with_opts(
            FileDownload {
                client,
                data_dir,
                repo_id: &request.repo_id,
                revision: &request.revision,
                filename: &file.path,
                source_url: None,
                destination: &destination,
                job: job.as_ref().map(|(db, id)| (db, id.as_str())),
                cancel: cancel.as_ref(),
                expected_size: file.size,
                expected_sha256: file.sha256.as_deref(),
                snapshot: Some(SnapshotProgressCtx {
                    completed_before: total_bytes,
                    overall_total,
                    file_index: index,
                    file_count: files.len(),
                    file_path: &file.path,
                }),
            },
            &mut progress,
        )
        .await?;
        total_bytes += bytes;
    }
    let root = crate::voice::download_root(data_dir, &request.repo_id)?;
    let model_id = crate::voice::model_id_for_repo(&request.repo_id)?;
    progress(ProgressEvent::phase(
        "verify",
        "Finalizing verified PersonaPlex snapshot",
    ));
    let sha256 = snapshot_manifest_sha256(&files);
    let result = DownloadResult {
        model_id: model_id.clone(),
        path: root.display().to_string(),
        bytes: total_bytes,
        sha256,
        resumed: false,
        engine: Some(crate::voice::ENGINE.to_owned()),
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

/// Stable identity for a verified snapshot without reading every model byte a
/// second time. Each LFS-backed file was checked against its Hub SHA-256 during
/// download, and the manifest binds the record to those exact objects.
fn snapshot_manifest_sha256(files: &[crate::hf::RepoFile]) -> String {
    let mut ordered: Vec<_> = files.iter().collect();
    ordered.sort_unstable_by(|left, right| left.path.cmp(&right.path));

    let mut digest = Sha256::new();
    digest.update(b"brazier-snapshot-manifest-v2\0");
    for file in ordered {
        digest.update(file.path.as_bytes());
        digest.update([0]);
        match file.size {
            Some(size) => {
                digest.update([1]);
                digest.update(size.to_le_bytes());
            }
            None => digest.update([0]),
        }
        match &file.sha256 {
            Some(sha256) => {
                digest.update([1]);
                digest.update(sha256.as_bytes());
            }
            None => digest.update([0]),
        }
    }
    hex::encode(digest.finalize())
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

    #[test]
    fn snapshot_manifest_is_order_independent() {
        let files = vec![
            crate::hf::RepoFile {
                path: "weights/model.safetensors".into(),
                size: Some(42),
                sha256: Some("a".repeat(64)),
            },
            crate::hf::RepoFile {
                path: "config.json".into(),
                size: Some(7),
                sha256: None,
            },
        ];
        let reversed = vec![files[1].clone(), files[0].clone()];
        assert_eq!(
            snapshot_manifest_sha256(&files),
            snapshot_manifest_sha256(&reversed)
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

    #[tokio::test]
    async fn checksum_rejects_complete_file_with_wrong_bytes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"same-length-bad").unwrap();
        assert!(
            !sha256_matches(&path, Some(&sha256_hex(b"same-length-ok!")))
                .await
                .unwrap()
        );
        assert!(
            sha256_matches(&path, Some(&sha256_hex(b"same-length-bad")))
                .await
                .unwrap()
        );
    }
}

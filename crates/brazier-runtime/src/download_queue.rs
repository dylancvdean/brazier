//! Background queue for model downloads.
//!
//! A few transfers run at once and the rest wait their turn. Transfers are
//! bandwidth- and disk-bound, so running everything in parallel makes each one
//! slower without finishing the set any sooner — but a strict single file makes
//! a small download wait behind a multi-gigabyte one, which is worse to sit in
//! front of. Work is described by [`QueuedWork`], which is persisted on the job
//! row so a paused download can be resumed later, including after a restart.

use std::{path::Path, sync::Arc};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::{
    active_downloads::{ActiveDownloads, StopReason},
    db::Database,
    download::{self, DownloadRequest, MlxDownloadRequest},
    engine::Runtime,
    model_settings, models_store,
};

/// What a queued job should download. Serialized onto the job row.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum QueuedWork {
    /// A single GGUF (or whisper `.bin`) file.
    Gguf(DownloadRequest),
    /// An MLX snapshot for `mlx-lm` / `mlx-vlm`.
    Mlx(MlxDownloadRequest),
    /// A PersonaPlex / Moshi snapshot.
    Personaplex(MlxDownloadRequest),
    /// A streaming-ASR snapshot.
    StreamingAsr(MlxDownloadRequest),
    /// A curated or custom stable-diffusion.cpp bundle.
    SdcppBundle(Box<crate::sdcpp_catalog::Bundle>),
}

impl QueuedWork {
    /// Short tag stored in the `kind` column.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Gguf(_) => "gguf",
            Self::Mlx(_) => "mlx",
            Self::Personaplex(_) => "personaplex",
            Self::StreamingAsr(_) => "streaming-asr",
            Self::SdcppBundle(_) => "sdcpp-bundle",
        }
    }

    /// Repository the work comes from, for the job row and the queue UI.
    pub fn repo_id(&self) -> String {
        match self {
            Self::Gguf(request) => request.repo_id.clone(),
            Self::Mlx(request) | Self::Personaplex(request) | Self::StreamingAsr(request) => {
                request.repo_id.clone()
            }
            Self::SdcppBundle(bundle) => bundle
                .components
                .first()
                .map(|component| component.repo_id.clone())
                .unwrap_or_else(|| bundle.key.clone()),
        }
    }

    /// Specific file, where the work names one.
    pub fn filename(&self) -> String {
        match self {
            Self::Gguf(request) => request.filename.clone(),
            Self::Mlx(_) | Self::Personaplex(_) | Self::StreamingAsr(_) => "snapshot".to_owned(),
            Self::SdcppBundle(bundle) => format!("{} files", bundle.components.len()),
        }
    }

    /// Name shown in the queue.
    pub fn label(&self) -> String {
        match self {
            Self::Gguf(request) => request
                .filename
                .rsplit('/')
                .next()
                .unwrap_or(&request.filename)
                .to_owned(),
            Self::Mlx(request) | Self::Personaplex(request) | Self::StreamingAsr(request) => {
                request.repo_id.clone()
            }
            Self::SdcppBundle(bundle) => bundle.label.clone(),
        }
    }

    pub fn revision(&self) -> String {
        match self {
            Self::Gguf(request) => request.revision.clone(),
            Self::Mlx(request) | Self::Personaplex(request) | Self::StreamingAsr(request) => {
                request.revision.clone()
            }
            Self::SdcppBundle(_) => "main".to_owned(),
        }
    }
}

pub struct QueuedDownload {
    pub job_id: String,
    pub work: QueuedWork,
}

#[derive(Clone)]
pub struct DownloadQueue {
    tx: mpsc::Sender<QueuedDownload>,
}

/// How many transfers may be in flight at once.
///
/// Enough that a small model is not stuck behind a huge one, few enough that
/// they are not all crawling.
const CONCURRENT_TRANSFERS: usize = 3;

impl DownloadQueue {
    pub fn spawn(
        http: reqwest::Client,
        data_dir: std::path::PathBuf,
        db: Database,
        active: Arc<ActiveDownloads>,
        runtime: Arc<Runtime>,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<QueuedDownload>(256);
        tokio::spawn(async move {
            let slots = Arc::new(tokio::sync::Semaphore::new(CONCURRENT_TRANSFERS));
            while let Some(work) = rx.recv().await {
                // Taking the slot before accepting the next job keeps jobs
                // starting in the order they were queued.
                let Ok(slot) = Arc::clone(&slots).acquire_owned().await else {
                    break;
                };
                let http = http.clone();
                let data_dir = data_dir.clone();
                let db = db.clone();
                let active = Arc::clone(&active);
                let runtime = Arc::clone(&runtime);
                tokio::spawn(async move {
                    if run_one(&http, &data_dir, &db, &active, work).await {
                        // Model discovery is cached because it includes remote
                        // probes. A queued install must invalidate that cache
                        // just like the streaming download endpoints do.
                        runtime.invalidate_models_cache().await;
                    }
                    drop(slot);
                });
            }
        });
        Self { tx }
    }

    pub async fn enqueue(&self, work: QueuedDownload) -> anyhow::Result<()> {
        self.tx.send(work).await.context("download queue is closed")
    }

    /// Put a paused, failed, or cancelled job back in line from its stored request.
    pub async fn resume_job(&self, db: &Database, job_id: &str) -> anyhow::Result<()> {
        let job = db.get_download_job_public(job_id).await?;
        let payload = job
            .payload
            .as_deref()
            .context("this job predates the queue and cannot be resumed")?;
        let work: QueuedWork = serde_json::from_str(payload).context("unreadable job payload")?;
        db.requeue_download_job(job_id).await?;
        self.enqueue(QueuedDownload {
            job_id: job.id,
            work,
        })
        .await
    }
}

/// Automatic resume attempts after a dropped connection, before the job is
/// marked failed. Each attempt continues from the partial file.
const TRANSIENT_RETRIES: u32 = 3;

/// Network hiccups worth retrying, as opposed to auth, missing files, or a
/// checksum mismatch that would fail the same way again.
fn is_transient(error: &anyhow::Error) -> bool {
    if error.chain().any(|cause| {
        cause.downcast_ref::<reqwest::Error>().is_some_and(|error| {
            error.is_timeout() || error.is_connect() || error.is_body() || error.is_request()
        })
    }) {
        return true;
    }
    let message = format!("{error:#}");
    [
        "download stalled",
        "no response after",
        "resume to finish",
        "failed (429",
        "failed (500",
        "failed (502",
        "failed (503",
        "failed (504",
    ]
    .iter()
    .any(|marker| message.contains(marker))
}

async fn run_one(
    http: &reqwest::Client,
    data_dir: &Path,
    db: &Database,
    active: &ActiveDownloads,
    work: QueuedDownload,
) -> bool {
    // Register before checking durable state so cancellation cannot slip
    // between "still pending" and registration. Once the flag is visible,
    // every later cancel reaches this worker directly.
    let cancel = active.register(&work.job_id);
    // A job cancelled or paused while still waiting in line should not start.
    match db.get_download_job_public(&work.job_id).await {
        Ok(job) if matches!(job.status.as_str(), "pending" | "downloading") => {}
        Ok(_) => {
            active.finish_if_matches(&work.job_id, &cancel);
            return false;
        }
        Err(error) => {
            tracing::error!(
                job_id = %work.job_id,
                error = %error,
                "refusing to start download without durable job state"
            );
            active.finish_if_matches(&work.job_id, &cancel);
            return false;
        }
    }

    let mut attempt = 0;
    let result = loop {
        let result = run_work(http, data_dir, db, &work, &cancel).await;
        let Err(error) = &result else { break result };
        if attempt >= TRANSIENT_RETRIES || cancel.reason().is_some() || !is_transient(error) {
            break result;
        }
        attempt += 1;
        let delay = std::time::Duration::from_secs(5 * u64::from(attempt));
        tracing::warn!(job_id = %work.job_id, %error, attempt, "retrying interrupted download");
        let _ = db
            .update_download_job_message(
                &work.job_id,
                &format!(
                    "Connection dropped — retrying in {}s ({attempt}/{TRANSIENT_RETRIES})",
                    delay.as_secs()
                ),
            )
            .await;
        tokio::select! {
            _ = cancel.wait_stopped() => break result,
            () = tokio::time::sleep(delay) => {}
        }
    };

    // Read our own worker's flag and persist the terminal DB state BEFORE
    // unregistering from `active`, so a concurrent `resume_model_download`
    // cannot observe `is_running() == false`, requeue to `pending`, and then
    // get overwritten back to `paused` by this exiting worker.
    let stop = cancel.reason();
    if let Err(error) = result {
        match stop {
            // A paused job keeps its partial file and its place in the list.
            Some(StopReason::Pause) => {
                let _ = db.pause_download_job(&work.job_id).await;
            }
            Some(StopReason::Cancel) => {
                let _ = db.cancel_download_job(&work.job_id).await;
            }
            None => {
                let _ = db.fail_download_job(&work.job_id, &error.to_string()).await;
            }
        }
        active.finish_if_matches(&work.job_id, &cancel);
        return false;
    }
    active.finish_if_matches(&work.job_id, &cancel);
    true
}

async fn run_work(
    http: &reqwest::Client,
    data_dir: &Path,
    db: &Database,
    work: &QueuedDownload,
    cancel: &Arc<crate::active_downloads::StopFlag>,
) -> anyhow::Result<()> {
    let job_handle = Some((db.clone(), work.job_id.clone()));
    let progress = Box::new(|_| {});
    match work.work.clone() {
        QueuedWork::Gguf(request) => {
            let result = download::download_gguf_with_progress(
                http,
                data_dir,
                request.clone(),
                progress,
                job_handle,
                Some(cancel.clone()),
            )
            .await
            .map(|_| ());
            if result.is_ok() && request.engine == "llama.cpp" {
                apply_installed_draft_defaults(data_dir, &request.repo_id, &request.filename).await;
            }
            result
        }
        QueuedWork::Mlx(request) => download::download_mlx_snapshot_with_progress(
            http,
            data_dir,
            request,
            progress,
            job_handle,
            Some(cancel.clone()),
        )
        .await
        .map(|_| ()),
        QueuedWork::Personaplex(request) => download::download_personaplex_snapshot_with_progress(
            http,
            data_dir,
            request,
            progress,
            job_handle,
            Some(cancel.clone()),
        )
        .await
        .map(|_| ()),
        QueuedWork::StreamingAsr(request) => {
            download::download_streaming_asr_snapshot_with_progress(
                http,
                data_dir,
                request,
                progress,
                job_handle,
                Some(cancel.clone()),
            )
            .await
            .map(|_| ())
        }
        QueuedWork::SdcppBundle(bundle) => download::install_sdcpp_bundle_with_progress(
            http,
            data_dir,
            &bundle,
            progress,
            job_handle,
            Some(cancel.clone()),
        )
        .await
        .map(|_| ()),
    }
}

/// After a speculative-draft companion (`dspark`, `dflash`, `draft`) installs,
/// make it the default draft for every primary model in its repository — but
/// only when that model has not been configured otherwise, and never for a
/// drafter mainline llama.cpp cannot load. This is what turns "I installed the
/// drafter" into "the model now uses it", until the person changes the choice
/// in the model library.
async fn apply_installed_draft_defaults(data_dir: &Path, repo_id: &str, filename: &str) {
    if !models_store::is_speculative_draft_file(Path::new(filename)) {
        return;
    }
    if models_store::is_mainline_incompatible_draft(Path::new(filename)) {
        return;
    }
    let root = models_store::gguf_root(data_dir);
    let directory = root.join(repo_id);
    let Ok(entries) = std::fs::read_dir(&directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.to_ascii_lowercase().ends_with(".gguf") {
            continue;
        }
        if models_store::is_projector_file(&path) || models_store::is_speculative_draft_file(&path)
        {
            continue;
        }
        let Ok(model_id) = models_store::model_id_for_path(&root, &path) else {
            continue;
        };
        if let Err(error) =
            model_settings::set_draft_default_if_unset(data_dir, &model_id, filename).await
        {
            tracing::warn!(%error, model_id, "could not set installed drafter as the default");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_network_hiccups_are_retried() {
        for message in [
            "download stalled: nothing received for 120s after 1.2 GB. It can be resumed from here.",
            "start Hugging Face download: no response after 30s. Check your network connection and retry.",
            "download of model.gguf ended prematurely at 1 of 2 bytes — resume to finish",
            "Hugging Face download failed (503 Service Unavailable): busy",
        ] {
            assert!(
                is_transient(&anyhow::anyhow!(message.to_owned())),
                "{message}"
            );
        }
        for message in [
            "Hugging Face rejected the download (401 Unauthorized). This model may be gated",
            "Hugging Face download failed (404 Not Found): missing",
            "model.gguf failed its checksum (expected SHA-256 aa, got bb).",
        ] {
            assert!(
                !is_transient(&anyhow::anyhow!(message.to_owned())),
                "{message}"
            );
        }
    }

    /// Resuming reads the work back off the job row, so a payload that does not
    /// survive the round trip is a job that can only be started from scratch.
    #[test]
    fn queued_work_survives_the_job_row_it_is_stored_on() {
        let work = QueuedWork::Gguf(DownloadRequest {
            repo_id: "acme/models".into(),
            filename: "nested/model-Q4_K_M.gguf".into(),
            revision: "main".into(),
            engine: "whisper.cpp".into(),
            beside_model_id: None,
        });
        let payload = serde_json::to_string(&work).expect("serialize");
        let restored: QueuedWork = serde_json::from_str(&payload).expect("deserialize");

        assert_eq!(restored.kind(), "gguf");
        assert_eq!(restored.repo_id(), "acme/models");
        assert_eq!(restored.filename(), "nested/model-Q4_K_M.gguf");
        // The label is what the tray shows, so it stays the bare file name.
        assert_eq!(restored.label(), "model-Q4_K_M.gguf");
        let QueuedWork::Gguf(request) = restored else {
            panic!("kind changed across the round trip");
        };
        // The engine decides where the file lands; losing it would resume a
        // whisper model into the llama.cpp store.
        assert_eq!(request.engine, "whisper.cpp");
    }
}

//! Serial background queue for every kind of model download.
//!
//! One transfer runs at a time — they are bandwidth- and disk-bound, and
//! interleaving them only makes each slower — but anything can be queued while
//! one is in flight. Work is described by [`QueuedWork`], which is persisted on
//! the job row so a paused download can be resumed later, including after a
//! restart.

use std::{path::Path, sync::Arc};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::{
    active_downloads::{ActiveDownloads, StopReason},
    db::Database,
    download::{self, DownloadRequest, MlxDownloadRequest},
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
    SdcppBundle(crate::sdcpp_catalog::Bundle),
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

impl DownloadQueue {
    pub fn spawn(
        http: reqwest::Client,
        data_dir: std::path::PathBuf,
        db: Database,
        active: Arc<ActiveDownloads>,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<QueuedDownload>(256);
        tokio::spawn(async move {
            while let Some(work) = rx.recv().await {
                run_one(&http, &data_dir, &db, &active, work).await;
            }
        });
        Self { tx }
    }

    pub async fn enqueue(&self, work: QueuedDownload) -> anyhow::Result<()> {
        self.tx.send(work).await.context("download queue is closed")
    }
}

async fn run_one(
    http: &reqwest::Client,
    data_dir: &Path,
    db: &Database,
    active: &ActiveDownloads,
    work: QueuedDownload,
) {
    // A job cancelled or paused while still waiting in line should not start.
    if let Ok(job) = db.get_download_job_public(&work.job_id).await {
        if matches!(job.status.as_str(), "paused" | "cancelled") {
            return;
        }
    }

    let cancel = active.register(&work.job_id);
    let job_handle = Some((db.clone(), work.job_id.clone()));
    let progress = Box::new(|_| {});
    let result = match work.work {
        QueuedWork::Gguf(request) => download::download_gguf_with_progress(
            http,
            data_dir,
            request,
            progress,
            job_handle,
            Some(cancel.clone()),
        )
        .await
        .map(|_| ()),
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
    };

    let stop = active.stop_reason(&work.job_id);
    active.finish(&work.job_id);
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
    }
}

//! Serial background queue for Hugging Face GGUF downloads.

use std::{path::Path, sync::Arc};

use anyhow::Context;
use tokio::sync::mpsc;

use crate::{
    active_downloads::ActiveDownloads,
    db::Database,
    download::{self, DownloadRequest},
};

pub struct QueuedDownload {
    pub job_id: String,
    pub request: DownloadRequest,
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
        let (tx, mut rx) = mpsc::channel::<QueuedDownload>(64);
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
    let cancel = active.register(&work.job_id);
    let job_handle = Some((db.clone(), work.job_id.clone()));
    let result = download::download_gguf_with_progress(
        http,
        data_dir,
        work.request,
        Box::new(|_| {}),
        job_handle,
        Some(cancel.clone()),
    )
    .await;
    active.finish(&work.job_id);
    if let Err(error) = result {
        let message = error.to_string();
        if message.contains("cancelled") {
            let _ = db.cancel_download_job(&work.job_id).await;
        } else {
            let _ = db.fail_download_job(&work.job_id, &message).await;
        }
    }
}

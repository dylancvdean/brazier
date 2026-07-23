pub mod active_downloads;
pub mod api;
pub mod blob_store;
pub mod build_recipe;
pub mod builds;
pub mod db;
pub mod download;
pub mod download_queue;
pub mod engine;
pub mod hardware;
pub mod hf;
pub mod hf_auth;
pub mod js_sandbox;
pub mod llama;
pub mod models_store;
pub mod progress;
pub mod runtime_settings;
pub mod runtimes;
pub mod tools;
pub mod types;

use std::{path::PathBuf, sync::Arc};

use db::Database;
use download_queue::DownloadQueue;
use engine::Runtime;
use active_downloads::ActiveDownloads;

#[derive(Clone)]
pub struct AppState {
    pub db: Database,
    pub runtime: Arc<Runtime>,
    pub api_key: Option<String>,
    pub http: reqwest::Client,
    pub data_dir: PathBuf,
    pub active_builds: Arc<builds::ActiveBuilds>,
    pub active_downloads: Arc<ActiveDownloads>,
    pub download_queue: DownloadQueue,
}

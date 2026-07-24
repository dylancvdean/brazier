pub mod active_downloads;
pub mod api;
pub mod blob_store;
pub mod build_recipe;
pub mod builds;
pub mod db;
pub mod download;
pub mod download_queue;
pub mod engine;
pub mod fork_hints;
pub mod github_releases;
pub mod hardware;
pub mod harmony;
pub mod hf;
pub mod hf_auth;
pub mod js_sandbox;
pub mod llama;
pub mod mcp;
pub mod media;
pub mod message_format;
pub mod mlx;
pub mod model_bindings;
pub mod model_library;
pub mod models_store;
pub mod progress;
pub mod runtime_settings;
pub mod runtimes;
pub mod sdcpp;
pub mod streaming_asr;
pub mod tool_registry;
pub mod toolchain_hints;
pub mod tools;
pub mod types;
pub mod voice;
pub mod whisper;
pub mod whisperkit;

use std::{path::PathBuf, sync::Arc};

use active_downloads::ActiveDownloads;
use db::Database;
use download_queue::DownloadQueue;
use engine::Runtime;
use runtimes::RuntimeEntry;
use tokio::sync::Mutex;

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
    pub runtimes_cache: Arc<Mutex<Option<Vec<RuntimeEntry>>>>,
}

impl AppState {
    pub async fn invalidate_runtimes_cache(&self) {
        *self.runtimes_cache.lock().await = None;
    }

    pub async fn invalidate_models_cache(&self) {
        self.runtime.invalidate_models_cache().await;
    }
}

pub mod active_downloads;
pub mod api;
pub mod blob_store;
pub mod build_recipe;
pub mod builds;
pub mod db;
pub mod download;
pub mod download_queue;
pub mod fork_hints;
pub mod engine;
pub mod harmony;
pub mod hardware;
pub mod hf;
pub mod hf_auth;
pub mod js_sandbox;
pub mod llama;
pub mod media;
pub mod message_format;
pub mod mlx;
pub mod mcp;
pub mod model_bindings;
pub mod model_library;
pub mod models_store;
pub mod progress;
pub mod runtime_settings;
pub mod runtimes;
pub mod tool_registry;
pub mod toolchain_hints;
pub mod tools;
pub mod sdcpp;
pub mod streaming_asr;
pub mod types;
pub mod voice;
pub mod whisper;

use std::{path::PathBuf, sync::Arc};

use db::Database;
use download_queue::DownloadQueue;
use engine::Runtime;
use active_downloads::ActiveDownloads;
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

pub mod api;
pub mod build_recipe;
pub mod builds;
pub mod db;
pub mod download;
pub mod engine;
pub mod hardware;
pub mod hf;
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
use engine::Runtime;

#[derive(Clone)]
pub struct AppState {
    pub db: Database,
    pub runtime: Arc<Runtime>,
    pub api_key: Option<String>,
    pub http: reqwest::Client,
    pub data_dir: PathBuf,
}

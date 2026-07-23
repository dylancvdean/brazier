pub mod api;
pub mod build_recipe;
pub mod db;
pub mod engine;
pub mod hf;
pub mod types;

use std::sync::Arc;

use db::Database;
use engine::Engine;

#[derive(Clone)]
pub struct AppState {
    pub db: Database,
    pub engine: Arc<dyn Engine>,
    pub api_key: Option<String>,
    pub http: reqwest::Client,
}

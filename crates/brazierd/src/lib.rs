pub mod api;
pub mod service;
pub mod support;

pub use brazier_agent::{
    agent_exec, agent_policy, agent_sandbox, agent_tools, agent_worktree, computer_browser,
    computer_desktop, computer_exec, computer_fara, computer_policy,
};
pub use brazier_formats::{gguf_meta, wav};
pub use brazier_protocol::{agent_types, computer_types, message_format, progress, types};
pub use brazier_runtime::{
    active_downloads, adapters, build_recipe, builds, download, download_queue, engine, fork_hints,
    github_releases, hardware, harmony, hf, hf_auth, js_sandbox, llama, mcp, media, mlx,
    model_bindings, model_library, model_settings, models_store, recommendations, remote, rocm,
    runtime_settings, runtimes, sdcpp, sdcpp_arch, sdcpp_catalog, streaming_asr, tool_registry,
    toolchain_hints, tools, voice, whisper, whisperkit,
};
pub use brazier_storage::{agent_store, blob_store, db};

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
    /// Source compilation is deliberately serialized: concurrent C++ builds
    /// compete for every scarce local resource at once.
    pub build_slots: Arc<tokio::sync::Semaphore>,
    pub active_downloads: Arc<ActiveDownloads>,
    pub download_queue: DownloadQueue,
    pub runtimes_cache: Arc<Mutex<Option<Vec<RuntimeEntry>>>>,
    /// Policy, sandbox, and execution for Agent mode. The agent runtime reaches
    /// the host only through this broker.
    pub agent_broker: Arc<agent_exec::AgentBroker>,
    /// Computer Use observe–act broker (browser/desktop drivers + approvals).
    pub computer_broker: Arc<computer_exec::ComputerBroker>,
}

impl AppState {
    pub async fn invalidate_runtimes_cache(&self) {
        *self.runtimes_cache.lock().await = None;
    }

    pub async fn invalidate_models_cache(&self) {
        self.runtime.invalidate_models_cache().await;
    }
}

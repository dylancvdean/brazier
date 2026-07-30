//! Local model discovery, installation, execution, and tool orchestration.

pub mod active_downloads;
pub mod adapters;
pub mod build_recipe;
pub mod builds;
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
pub mod mlx;
pub mod model_bindings;
pub mod model_library;
pub mod model_settings;
pub mod models_store;
pub mod recommendations;
pub mod remote;
pub mod rocm;
pub mod runtime_settings;
pub mod runtimes;
pub mod sdcpp;
pub mod sdcpp_arch;
pub mod sdcpp_catalog;
pub mod streaming_asr;
pub mod tool_registry;
pub mod toolchain_hints;
pub mod tools;
pub mod vllm;
pub mod voice;
pub mod whisper;
pub mod whisperkit;

mod persistence;

pub use brazier_formats::{gguf_meta, wav};
pub use brazier_protocol::{message_format, progress, types};
pub use brazier_storage::{blob_store, db};

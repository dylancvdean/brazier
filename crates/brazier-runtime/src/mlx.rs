//! Apple MLX inference via isolated Python virtual environments.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use anyhow::Context;
use tokio::process::{Child, Command};

use crate::{llama, runtime_settings::RuntimeSettings};

/// Supported MLX Python engine packages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxKind {
    Lm,
    Vlm,
}

impl MlxKind {
    pub fn engine_id(self) -> &'static str {
        match self {
            Self::Lm => "mlx-lm",
            Self::Vlm => "mlx-vlm",
        }
    }

    pub fn module(self) -> &'static str {
        match self {
            Self::Lm => "mlx_lm.server",
            Self::Vlm => "mlx_vlm.server",
        }
    }

    pub fn import_check(self) -> &'static str {
        match self {
            Self::Lm => "import mlx_lm",
            Self::Vlm => "import mlx_vlm",
        }
    }

    pub fn from_engine_id(engine: &str) -> Option<Self> {
        match engine {
            "mlx-lm" => Some(Self::Lm),
            "mlx-vlm" => Some(Self::Vlm),
            _ => None,
        }
    }

    pub fn from_model_id(model_id: &str) -> Option<Self> {
        if model_id.starts_with("mlx-vlm-ext:") || model_id.starts_with("mlx-vlm:") {
            Some(Self::Vlm)
        } else if model_id.starts_with("mlx-ext:") || model_id.starts_with("mlx:") {
            Some(Self::Lm)
        } else {
            None
        }
    }
}

pub fn python_name() -> &'static str {
    if cfg!(windows) {
        "python.exe"
    } else {
        "python"
    }
}

pub fn venv_python(venv: &Path) -> PathBuf {
    venv.join("bin").join(python_name())
}

/// Verify that a venv Python can import the expected MLX package.
pub fn python_appears_runnable(python: &Path, kind: MlxKind) -> bool {
    if !python.is_file() {
        return false;
    }
    std::process::Command::new(python)
        .args(["-c", kind.import_check()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Running MLX HTTP server child process.
pub struct MlxServer {
    child: Child,
    pub base_url: String,
    pub model_ref: String,
    pub python: PathBuf,
    pub kind: MlxKind,
    /// What this process was launched with, so a model reconfigured since it
    /// loaded is restarted rather than served stale.
    pub launch_key: String,
}

/// mlx-lm prompt-cache and batching limits when parallel subagents are on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlxLmAgentCacheArgs {
    pub prompt_cache_size: u32,
    pub decode_concurrency: u32,
    pub prompt_concurrency: u32,
}

/// Match llama-server slot count: one LRU cache entry and one batch lane per
/// concurrent session (parent + subagents). Off when parallel subagents is off.
pub fn mlx_lm_agent_cache_args(
    kind: MlxKind,
    profile: Option<&crate::model_settings::TextProfile>,
) -> Option<MlxLmAgentCacheArgs> {
    if kind != MlxKind::Lm {
        return None;
    }
    let slots = crate::model_settings::llama_parallel_slots(profile);
    if slots <= 1 {
        return None;
    }
    Some(MlxLmAgentCacheArgs {
        prompt_cache_size: slots,
        decode_concurrency: slots,
        prompt_concurrency: slots,
    })
}

/// The fingerprint a server started with these inputs would carry.
pub fn launch_key(
    kind: MlxKind,
    settings: &RuntimeSettings,
    profile: Option<&crate::model_settings::TextProfile>,
    adapter: Option<&Path>,
) -> String {
    let max_tokens = profile
        .and_then(|profile| profile.max_tokens)
        .or(settings.max_tokens);
    let extra = profile
        .map(|profile| profile.extra_args.join(" "))
        .unwrap_or_default();
    let agent = mlx_lm_agent_cache_args(kind, profile)
        .map(|args| {
            format!(
                "{}|{}|{}",
                args.prompt_cache_size, args.decode_concurrency, args.prompt_concurrency
            )
        })
        .unwrap_or_default();
    format!(
        "{max_tokens:?}|{:?}|{adapter:?}|{agent}|{extra}",
        max_kv_size(kind, settings, profile)
    )
}

/// MLX-VLM's server accepts an explicit token cap for its KV cache. Apply
/// Brazier's selected per-model/global context so its capability label and the
/// running server agree. MLX-LM's pinned HTTP server does not accept this
/// option; it follows the model's native configuration instead.
fn max_kv_size(
    kind: MlxKind,
    settings: &RuntimeSettings,
    profile: Option<&crate::model_settings::TextProfile>,
) -> Option<u32> {
    (kind == MlxKind::Vlm).then(|| {
        profile
            .and_then(|profile| profile.context_size)
            .unwrap_or(settings.context_size)
    })
}

impl MlxServer {
    pub async fn start(
        python: &Path,
        kind: MlxKind,
        model_ref: &str,
        settings: &RuntimeSettings,
    ) -> anyhow::Result<Self> {
        Self::start_with_profile(python, kind, model_ref, settings, None, None).await
    }

    /// Start an MLX server with a model's own launch overrides.
    ///
    /// `adapter` is a directory of LoRA weights fine-tuned against this model.
    /// mlx-lm loads one at a time (`--adapter-path`), so a model configured with
    /// several is served by the first its engine can read.
    pub async fn start_with_profile(
        python: &Path,
        kind: MlxKind,
        model_ref: &str,
        settings: &RuntimeSettings,
        profile: Option<&crate::model_settings::TextProfile>,
        adapter: Option<&Path>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            python.is_file(),
            "MLX Python interpreter missing: {}",
            python.display()
        );
        anyhow::ensure!(
            python_appears_runnable(python, kind),
            "{} does not provide `{}`",
            python.display(),
            kind.engine_id()
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .context("reserve port for MLX server")?;
        let port = listener.local_addr()?.port();
        drop(listener);

        let mut command = Command::new(python);
        command
            .arg("-m")
            .arg(kind.module())
            .arg("--model")
            .arg(model_ref)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--log-level")
            .arg("WARNING")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        // mlx-lm uses --max-tokens as a server default when provided.
        if let Some(max_tokens) = profile
            .and_then(|profile| profile.max_tokens)
            .or(settings.max_tokens)
        {
            command.arg("--max-tokens").arg(max_tokens.to_string());
        }
        if let Some(max_kv_size) = max_kv_size(kind, settings, profile) {
            command.arg("--max-kv-size").arg(max_kv_size.to_string());
        }
        if let Some(args) = mlx_lm_agent_cache_args(kind, profile) {
            command
                .arg("--prompt-cache-size")
                .arg(args.prompt_cache_size.to_string())
                .arg("--decode-concurrency")
                .arg(args.decode_concurrency.to_string())
                .arg("--prompt-concurrency")
                .arg(args.prompt_concurrency.to_string());
        }
        if let Some(adapter) = adapter {
            command.arg("--adapter-path").arg(adapter);
        }
        for arg in profile
            .map(|profile| profile.extra_args.as_slice())
            .unwrap_or_default()
        {
            command.arg(arg);
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("spawn {} {}", python.display(), kind.module()))?;

        let base_url = format!("http://127.0.0.1:{port}");
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()?;
        let health_url = format!("{base_url}/health");
        let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
        loop {
            if let Some(status) = child.try_wait().context("poll MLX server")? {
                let mut stderr = String::new();
                if let Some(mut pipe) = child.stderr.take() {
                    use tokio::io::AsyncReadExt;
                    let mut buf = Vec::new();
                    let _ = pipe.read_to_end(&mut buf).await;
                    stderr = String::from_utf8_lossy(&buf).into_owned();
                }
                anyhow::bail!(crate::llama::describe_server_startup_failure(
                    "MLX server",
                    status,
                    &stderr
                ));
            }
            match client.get(&health_url).send().await {
                Ok(response) if response.status().is_success() => break,
                _ => {
                    if tokio::time::Instant::now() > deadline {
                        let _ = child.start_kill();
                        anyhow::bail!("MLX server health check timed out at {base_url}");
                    }
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
            }
        }

        Ok(Self {
            child,
            base_url,
            model_ref: model_ref.to_owned(),
            python: python.to_path_buf(),
            kind,
            launch_key: launch_key(kind, settings, profile, adapter),
        })
    }

    pub fn is_running(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(None) => true,
            Ok(Some(_)) | Err(_) => false,
        }
    }

    pub async fn stop(&mut self) -> anyhow::Result<()> {
        if self.child.try_wait()?.is_none() {
            self.child.start_kill().context("kill MLX server")?;
            let _ = self.child.wait().await;
        }
        Ok(())
    }
}

impl Drop for MlxServer {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

/// Query a running MLX server for health and loaded models.
pub async fn probe_server(client: &reqwest::Client, base_url: &str) -> serde_json::Value {
    llama::probe_server(client, base_url).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_engine_and_model_ids() {
        assert_eq!(MlxKind::from_engine_id("mlx-lm"), Some(MlxKind::Lm));
        assert_eq!(MlxKind::from_engine_id("mlx-vlm"), Some(MlxKind::Vlm));
        assert_eq!(MlxKind::from_model_id("mlx:acme/demo"), Some(MlxKind::Lm));
        assert_eq!(
            MlxKind::from_model_id("mlx-vlm:acme/vision"),
            Some(MlxKind::Vlm)
        );
        assert_eq!(MlxKind::from_model_id("gguf:demo.gguf"), None);
    }

    #[test]
    fn only_vlm_server_uses_the_selected_context_as_its_kv_limit() {
        let settings = RuntimeSettings {
            context_size: 1_048_576,
            ..RuntimeSettings::default()
        };
        assert_eq!(max_kv_size(MlxKind::Vlm, &settings, None), Some(1_048_576));
        assert_eq!(max_kv_size(MlxKind::Lm, &settings, None), None);
    }

    #[test]
    fn mlx_lm_agent_cache_follows_parallel_subagent_slots() {
        use crate::model_settings::TextProfile;

        assert_eq!(mlx_lm_agent_cache_args(MlxKind::Vlm, None), None);
        assert_eq!(mlx_lm_agent_cache_args(MlxKind::Lm, None), None);
        assert_eq!(
            mlx_lm_agent_cache_args(
                MlxKind::Lm,
                Some(&TextProfile {
                    parallel_subagents: Some(false),
                    max_subagents: Some(4),
                    ..TextProfile::default()
                })
            ),
            None
        );
        assert_eq!(
            mlx_lm_agent_cache_args(
                MlxKind::Lm,
                Some(&TextProfile {
                    parallel_subagents: Some(true),
                    max_subagents: Some(2),
                    ..TextProfile::default()
                })
            ),
            Some(MlxLmAgentCacheArgs {
                prompt_cache_size: 3,
                decode_concurrency: 3,
                prompt_concurrency: 3,
            })
        );
    }
}

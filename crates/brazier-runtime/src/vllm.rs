//! Linux vLLM OpenAI-compatible server lifecycle.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use anyhow::Context;
use tokio::process::{Child, Command};

use crate::runtime_settings::RuntimeSettings;

pub const ENGINE: &str = "vllm";

pub fn model_id(repo_id: &str) -> anyhow::Result<String> {
    validate_model_ref(repo_id)?;
    Ok(format!("{ENGINE}:{repo_id}"))
}

pub fn model_ref(model_id: &str) -> anyhow::Result<&str> {
    let value = model_id
        .strip_prefix("vllm:")
        .ok_or_else(|| anyhow::anyhow!("not a vLLM model id: {model_id}"))?;
    validate_model_ref(value)?;
    Ok(value)
}

fn validate_model_ref(value: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value.is_empty() && value.len() <= 300,
        "invalid vLLM model reference"
    );
    anyhow::ensure!(
        value
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != ".."),
        "invalid vLLM model reference"
    );
    anyhow::ensure!(
        value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '-' | '_' | '.')),
        "invalid vLLM model reference"
    );
    Ok(())
}

pub fn python_appears_runnable(python: &Path) -> bool {
    python.is_file()
        && std::process::Command::new(python)
            .args(["-c", "import vllm"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
}

pub struct Server {
    child: Child,
    pub base_url: String,
    pub model_ref: String,
    pub python: PathBuf,
    pub launch_key: String,
}

pub fn launch_key(
    settings: &RuntimeSettings,
    profile: Option<&crate::model_settings::TextProfile>,
) -> String {
    let context = profile
        .and_then(|p| p.context_size)
        .unwrap_or(settings.context_size);
    let extra = profile
        .map(|p| p.extra_args.join("\u{1f}"))
        .unwrap_or_default();
    format!("{context}|{extra}")
}

impl Server {
    pub async fn start(
        python: &Path,
        model_ref: &str,
        settings: &RuntimeSettings,
        profile: Option<&crate::model_settings::TextProfile>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            python.is_file(),
            "vLLM Python interpreter missing: {}",
            python.display()
        );
        anyhow::ensure!(
            python_appears_runnable(python),
            "{} does not provide vLLM",
            python.display()
        );
        validate_model_ref(model_ref)?;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .context("reserve port for vLLM server")?;
        let port = listener.local_addr()?.port();
        drop(listener);
        let context = profile
            .and_then(|p| p.context_size)
            .unwrap_or(settings.context_size);
        let mut command = Command::new(python);
        command
            .args([
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model_ref,
                "--host",
                "127.0.0.1",
                "--port",
            ])
            .arg(port.to_string())
            .arg("--max-model-len")
            .arg(context.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(configured) = settings
            .vllm_models
            .iter()
            .find(|entry| entry.repository == model_ref)
        {
            if let Some(revision) = configured
                .revision
                .as_deref()
                .filter(|value| !value.is_empty())
            {
                command.args(["--revision", revision]);
            }
            if let Some(dtype) = configured
                .dtype
                .as_deref()
                .filter(|value| !value.is_empty())
            {
                command.args(["--dtype", dtype]);
            }
            if let Some(memory) = configured.gpu_memory_utilization {
                command.args(["--gpu-memory-utilization", &memory.to_string()]);
            }
            if let Some(parallel) = configured.tensor_parallel_size {
                command.args(["--tensor-parallel-size", &parallel.to_string()]);
            }
            if configured.trust_remote_code {
                command.arg("--trust-remote-code");
            }
            for arg in &configured.extra_args {
                command.arg(arg);
            }
        }
        for arg in profile.map(|p| p.extra_args.as_slice()).unwrap_or_default() {
            command.arg(arg);
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("spawn {} vLLM", python.display()))?;
        let base_url = format!("http://127.0.0.1:{port}");
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(180);
        loop {
            if let Some(status) = child.try_wait().context("poll vLLM server")? {
                let mut stderr = String::new();
                if let Some(mut pipe) = child.stderr.take() {
                    use tokio::io::AsyncReadExt;
                    let mut bytes = Vec::new();
                    let _ = pipe.read_to_end(&mut bytes).await;
                    stderr = String::from_utf8_lossy(&bytes).into_owned();
                }
                anyhow::bail!(crate::llama::describe_server_startup_failure(
                    "vLLM server",
                    status,
                    &stderr
                ));
            }
            if client
                .get(format!("{base_url}/health"))
                .send()
                .await
                .is_ok_and(|response| response.status().is_success())
            {
                break;
            }
            if tokio::time::Instant::now() > deadline {
                let _ = child.start_kill();
                anyhow::bail!("vLLM server health check timed out at {base_url}");
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        Ok(Self {
            child,
            base_url,
            model_ref: model_ref.into(),
            python: python.into(),
            launch_key: launch_key(settings, profile),
        })
    }
    pub fn is_running(&mut self) -> bool {
        self.child.try_wait().is_ok_and(|status| status.is_none())
    }
    pub async fn stop(&mut self) -> anyhow::Result<()> {
        if self.child.try_wait()?.is_none() {
            self.child.start_kill()?;
            let _ = self.child.wait().await;
        }
        Ok(())
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_ids_are_scoped_and_safe() {
        assert_eq!(model_id("Qwen/Qwen3-8B").unwrap(), "vllm:Qwen/Qwen3-8B");
        assert!(model_ref("vllm:../../etc").is_err());
        assert!(model_ref("mlx:Qwen/Qwen3").is_err());
    }
}

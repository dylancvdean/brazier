use std::{os::unix::fs::MetadataExt as _, process::Stdio, time::Duration};

use anyhow::{Context as _, Result, bail};
use tokio::{
    io::{AsyncBufReadExt as _, AsyncReadExt as _, BufReader, Lines},
    process::{Child, ChildStderr, ChildStdout, Command},
};

pub const INSTALLED_PATH: &str = "/usr/lib/brazier-input-guard";

pub struct InputGuard {
    child: Child,
    lines: Lines<BufReader<ChildStdout>>,
    stderr: ChildStderr,
}

impl InputGuard {
    pub async fn open() -> Result<Self> {
        validate_install()?;
        let mut child = Command::new(INSTALLED_PATH)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .context("launch the privileged keyboard safety fallback")?;
        let stdout = child
            .stdout
            .take()
            .context("capture keyboard safety fallback output")?;
        let stderr = child
            .stderr
            .take()
            .context("capture keyboard safety fallback diagnostics")?;
        let mut lines = BufReader::new(stdout).lines();
        let first = tokio::time::timeout(Duration::from_secs(5), lines.next_line())
            .await
            .context("timed out starting the privileged keyboard safety fallback")?
            .context("read keyboard safety fallback readiness")?;
        if !first
            .as_deref()
            .is_some_and(|line| line.starts_with("READY "))
        {
            let detail = collect_failure(&mut child, stderr).await;
            bail!("the privileged keyboard safety fallback did not become ready{detail}")
        }
        Ok(Self {
            child,
            lines,
            stderr,
        })
    }

    pub async fn wait(mut self) -> Result<()> {
        loop {
            match self
                .lines
                .next_line()
                .await
                .context("read privileged keyboard safety fallback")?
                .as_deref()
            {
                Some("ESC") => {
                    let status = self
                        .child
                        .wait()
                        .await
                        .context("finish privileged keyboard safety fallback")?;
                    if !status.success() {
                        bail!("privileged keyboard safety fallback failed after emergency stop")
                    }
                    return Ok(());
                }
                Some(_) => {}
                None => {
                    let mut detail = String::new();
                    self.stderr.read_to_string(&mut detail).await.ok();
                    let status = self.child.wait().await.ok();
                    let detail = detail.trim();
                    if detail.is_empty() {
                        bail!(
                            "privileged keyboard safety fallback exited unexpectedly ({status:?})"
                        )
                    }
                    bail!("privileged keyboard safety fallback exited unexpectedly: {detail}")
                }
            }
        }
    }
}

fn validate_install() -> Result<()> {
    let metadata = std::fs::symlink_metadata(INSTALLED_PATH)
        .context("privileged keyboard safety fallback is not installed")?;
    let mode = metadata.mode();
    if !metadata.file_type().is_file()
        || metadata.uid() != 0
        || mode & 0o2000 == 0
        || mode & 0o022 != 0
    {
        bail!(
            "{INSTALLED_PATH} must be a root-owned, non-writable setgid executable; repair it from Settings > Computer Use permissions"
        )
    }
    Ok(())
}

async fn collect_failure(child: &mut Child, mut stderr: ChildStderr) -> String {
    let _ = child.wait().await;
    let mut detail = String::new();
    let _ = stderr.read_to_string(&mut detail).await;
    let detail = detail.trim();
    if detail.is_empty() {
        String::new()
    } else {
        format!(": {detail}")
    }
}

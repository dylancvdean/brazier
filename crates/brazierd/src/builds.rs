//! Execute approved engine build plans from source.
//!
//! Each build runs in an isolated prefix under
//! `<data>/engines/<engine>/builds/<id>/` with `source/`, `build/`, and
//! `install/` subdirectories, so existing installations are never touched
//! until the user activates the new binary.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::{
    build_recipe::{self, BuildPlanRequest, PlannedCommand},
    progress::{ProgressCallback, ProgressEvent},
    runtime_settings::RuntimeTarget,
};

#[derive(Debug, Clone, Deserialize)]
pub struct BuildRequest {
    pub engine: String,
    pub repository: String,
    pub revision: String,
    /// Acceleration flavor to configure (defaults to CPU-only).
    #[serde(default)]
    pub target: Option<RuntimeTarget>,
}

/// Metadata persisted next to every completed source build.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildRecord {
    pub engine: String,
    pub repository: String,
    pub revision: String,
    pub target: String,
    pub created_at: String,
    pub binary: String,
}

pub fn builds_root(data_dir: &Path, engine: &str) -> PathBuf {
    data_dir.join("engines").join(engine).join("builds")
}

/// Platform string matching recipe `supported_platforms`.
pub fn current_platform() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Some("linux-x64"),
        ("linux", "aarch64") => Some("linux-arm64"),
        ("macos", "x86_64") => Some("macos-x64"),
        ("macos", "aarch64") => Some("macos-arm64"),
        ("windows", "x86_64") => Some("windows-x64"),
        ("windows", "aarch64") => Some("windows-arm64"),
        _ => None,
    }
}

/// CMake flags enabling the requested acceleration backend.
pub fn target_flags(target: RuntimeTarget) -> Vec<String> {
    match target {
        RuntimeTarget::Cuda => vec!["-DGGML_CUDA=ON".into()],
        RuntimeTarget::Vulkan => vec!["-DGGML_VULKAN=ON".into()],
        RuntimeTarget::Rocm => vec!["-DGGML_HIP=ON".into()],
        RuntimeTarget::Metal => vec!["-DGGML_METAL=ON".into()],
        RuntimeTarget::Auto | RuntimeTarget::Cpu => Vec::new(),
    }
}

fn sanitize_id_segment(value: &str) -> String {
    let cleaned: String = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '-'
            }
        })
        .collect();
    cleaned.trim_matches('-').chars().take(48).collect()
}

/// Substitute `{source}` / `{build}` / `{install}` placeholders and expand the
/// `{target_flags}` pseudo-argument.
pub fn resolve_command_args(
    args: &[String],
    source: &Path,
    build: &Path,
    install: &Path,
    flags: &[String],
) -> Vec<String> {
    let mut resolved = Vec::with_capacity(args.len());
    for arg in args {
        if arg == "{target_flags}" {
            resolved.extend(flags.iter().cloned());
            continue;
        }
        resolved.push(
            arg.replace("{source}", &source.display().to_string())
                .replace("{build}", &build.display().to_string())
                .replace("{install}", &install.display().to_string()),
        );
    }
    resolved
}

fn command_available(program: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|directory| {
            directory.join(program).is_file()
                || (cfg!(windows) && directory.join(format!("{program}.exe")).is_file())
        })
    })
}

async fn run_step(
    label: &str,
    program: &str,
    args: &[String],
    workdir: &Path,
    log: &mut String,
    progress: &mut ProgressCallback,
) -> anyhow::Result<()> {
    progress(ProgressEvent::phase("build", format!("{label}…")));
    log.push_str(&format!("$ {program} {}\n", args.join(" ")));
    let mut child = tokio::process::Command::new(program)
        .args(args)
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn {program}"))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let (line_tx, mut line_rx) = tokio::sync::mpsc::channel::<String>(256);
    let mut readers = Vec::new();
    if let Some(stdout) = stdout {
        let tx = line_tx.clone();
        readers.push(tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if tx.send(line).await.is_err() {
                    break;
                }
            }
        }));
    }
    if let Some(stderr) = stderr {
        let tx = line_tx.clone();
        readers.push(tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if tx.send(line).await.is_err() {
                    break;
                }
            }
        }));
    }
    drop(line_tx);

    let mut emitted = 0_usize;
    while let Some(line) = line_rx.recv().await {
        log.push_str(&line);
        log.push('\n');
        emitted += 1;
        // Forward a bounded stream of log lines so the UI stays responsive.
        if emitted.is_multiple_of(5) || emitted < 40 {
            progress(ProgressEvent::phase("log", line));
        }
    }
    for reader in readers {
        let _ = reader.await;
    }
    let status = child.wait().await.context("wait for build step")?;
    anyhow::ensure!(status.success(), "{label} failed with {status}");
    Ok(())
}

/// Copy the built server binary and any shared libraries into `install/bin`.
fn install_artifacts(build_dir: &Path, install_bin: &Path) -> anyhow::Result<PathBuf> {
    let server_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let built_bin = build_dir.join("bin");
    let server = built_bin.join(server_name);
    anyhow::ensure!(
        server.is_file(),
        "build completed but {server_name} was not produced at {}",
        server.display()
    );
    std::fs::create_dir_all(install_bin).context("create install directory")?;
    let mut installed_server = None;
    for entry in std::fs::read_dir(&built_bin).context("read build output")? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let is_lib = name.contains(".so") || name.ends_with(".dll") || name.ends_with(".dylib");
        if !(is_lib || name == server_name) {
            continue;
        }
        let destination = install_bin.join(name);
        std::fs::copy(&path, &destination).with_context(|| format!("install {name}"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = std::fs::metadata(&destination)?.permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(&destination, permissions)?;
        }
        if name == server_name {
            installed_server = Some(destination);
        }
    }
    installed_server.context("server binary missing after install")
}

/// Execute a full source build with streamed progress. Returns the installed
/// binary path. The build directory is removed on failure (the log is kept in
/// the progress stream and daemon log).
pub async fn run_build_with_progress(
    data_dir: &Path,
    request: BuildRequest,
    mut progress: ProgressCallback,
) -> anyhow::Result<PathBuf> {
    let platform =
        current_platform().context("source builds are not supported on this platform")?;
    let plan = build_recipe::plan(BuildPlanRequest {
        engine: request.engine.clone(),
        repository: request.repository.clone(),
        revision: request.revision.clone(),
        platform: platform.to_owned(),
    })?;
    anyhow::ensure!(
        plan.engine == "llama.cpp",
        "only llama.cpp source builds are currently executable"
    );
    for program in ["git", "cmake"] {
        anyhow::ensure!(
            command_available(program),
            "`{program}` is required to build from source; install it and try again"
        );
    }
    if let Some(warning) = &plan.warning {
        progress(ProgressEvent::phase("warning", warning.clone()));
    }

    let target = request.target.unwrap_or(RuntimeTarget::Cpu);
    let build_id = format!(
        "{}-{}",
        sanitize_id_segment(&request.revision),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );
    let root = builds_root(data_dir, &plan.engine).join(&build_id);
    tokio::fs::create_dir_all(&root)
        .await
        .context("create build directory")?;
    // Steps run with the build root as their working directory, so every
    // substituted path must be absolute.
    let root = std::path::absolute(&root).context("resolve build directory")?;
    let source = root.join("source");
    let build = root.join("build");
    let install = root.join("install");
    let install_bin = install.join("bin");

    let flags = target_flags(target);
    let mut log = String::new();
    let result: anyhow::Result<PathBuf> = async {
        let steps: Vec<&PlannedCommand> = plan.checkout.iter().chain(plan.build.iter()).collect();
        let total = steps.len();
        for (index, step) in steps.into_iter().enumerate() {
            let args = resolve_command_args(&step.args, &source, &build, &install, &flags);
            progress(ProgressEvent::phase(
                "build",
                format!("[{}/{}] {}", index + 1, total, step.label),
            ));
            run_step(
                &step.label,
                &step.program,
                &args,
                &root,
                &mut log,
                &mut progress,
            )
            .await?;
        }
        progress(ProgressEvent::phase(
            "install",
            "Installing the built server into an isolated prefix",
        ));
        let binary = install_artifacts(&build, &install_bin)?;
        // The source and intermediate build trees are large; keep only the install.
        let _ = tokio::fs::remove_dir_all(&source).await;
        let _ = tokio::fs::remove_dir_all(&build).await;
        Ok(binary)
    }
    .await;

    match result {
        Ok(binary) => {
            let record = BuildRecord {
                engine: plan.engine.clone(),
                repository: request.repository,
                revision: request.revision,
                target: target.as_str().to_owned(),
                created_at: format!(
                    "{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()
                ),
                binary: binary.display().to_string(),
            };
            tokio::fs::write(
                root.join("build.json"),
                serde_json::to_vec_pretty(&record).context("encode build record")?,
            )
            .await
            .context("write build record")?;
            tokio::fs::write(root.join("build.log"), &log).await.ok();
            progress(ProgressEvent::done(serde_json::json!({
                "binary": binary.display().to_string(),
                "build_id": build_id,
                "status": "ready"
            })));
            Ok(binary)
        }
        Err(error) => {
            tracing::error!(error = %error, log = %log, "source build failed");
            let _ = tokio::fs::remove_dir_all(&root).await;
            Err(error)
        }
    }
}

/// List completed source builds for an engine.
pub fn list_builds(data_dir: &Path, engine: &str) -> Vec<(String, BuildRecord)> {
    let root = builds_root(data_dir, engine);
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut builds = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(bytes) = std::fs::read(path.join("build.json")) else {
            continue;
        };
        let Ok(record) = serde_json::from_slice::<BuildRecord>(&bytes) else {
            continue;
        };
        if !Path::new(&record.binary).is_file() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().into_owned();
        builds.push((id, record));
    }
    builds.sort_by(|a, b| b.1.created_at.cmp(&a.1.created_at));
    builds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_placeholders_and_flag_expansion() {
        let args = vec![
            "-S".to_owned(),
            "{source}".to_owned(),
            "-B".to_owned(),
            "{build}".to_owned(),
            "{target_flags}".to_owned(),
            "--prefix".to_owned(),
            "{install}".to_owned(),
        ];
        let resolved = resolve_command_args(
            &args,
            Path::new("/tmp/s"),
            Path::new("/tmp/b"),
            Path::new("/tmp/i"),
            &["-DGGML_CUDA=ON".to_owned()],
        );
        assert_eq!(
            resolved,
            vec![
                "-S",
                "/tmp/s",
                "-B",
                "/tmp/b",
                "-DGGML_CUDA=ON",
                "--prefix",
                "/tmp/i"
            ]
        );
    }

    #[test]
    fn empty_target_flags_disappear() {
        let args = vec!["{target_flags}".to_owned(), "ok".to_owned()];
        let resolved = resolve_command_args(
            &args,
            Path::new("/s"),
            Path::new("/b"),
            Path::new("/i"),
            &[],
        );
        assert_eq!(resolved, vec!["ok"]);
    }

    #[test]
    fn build_ids_are_filesystem_safe() {
        assert_eq!(
            sanitize_id_segment("feature/new model"),
            "feature-new-model"
        );
        assert_eq!(sanitize_id_segment("--evil"), "evil");
    }

    #[test]
    fn lists_only_completed_builds() {
        let dir = tempfile::tempdir().unwrap();
        let root = builds_root(dir.path(), "llama.cpp");
        // Complete build with a real binary file.
        let done = root.join("main-1");
        let bin = done.join("install").join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let binary = bin.join("llama-server");
        std::fs::write(&binary, b"#!/bin/sh\n").unwrap();
        std::fs::write(
            done.join("build.json"),
            serde_json::to_vec(&BuildRecord {
                engine: "llama.cpp".into(),
                repository: "https://github.com/ggml-org/llama.cpp".into(),
                revision: "main".into(),
                target: "cpu".into(),
                created_at: "1".into(),
                binary: binary.display().to_string(),
            })
            .unwrap(),
        )
        .unwrap();
        // Incomplete build without metadata.
        std::fs::create_dir_all(root.join("broken")).unwrap();

        let builds = list_builds(dir.path(), "llama.cpp");
        assert_eq!(builds.len(), 1);
        assert_eq!(builds[0].0, "main-1");
        assert_eq!(builds[0].1.revision, "main");
    }
}

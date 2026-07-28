//! Execute approved engine build plans from source.
//!
//! Each build runs in an isolated prefix under
//! `<data>/engines/<engine>/builds/<id>/` with `source/`, `build/`, and
//! `install/` subdirectories, so existing installations are never touched
//! until the user activates the new binary.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::{
    build_recipe::{self, BuildPlanRequest, PlannedCommand},
    progress::{ProgressCallback, ProgressEvent},
    runtime_settings::{self, RuntimeTarget},
    toolchain_hints::{self, ToolchainPackage},
};

#[derive(Debug, Clone, Deserialize)]
pub struct BuildRequest {
    pub engine: String,
    pub repository: String,
    pub revision: String,
    /// Acceleration flavor to configure (defaults to CPU-only).
    #[serde(default)]
    pub target: Option<RuntimeTarget>,
    /// Parallel compile jobs (`cmake --build … --parallel N`). Defaults to half
    /// of available CPU threads when omitted.
    #[serde(default)]
    pub jobs: Option<u16>,
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
    /// Resolved upstream commit at build time (`git rev-parse HEAD`). Absent for
    /// builds made before commit capture, and for checkout-less Python recipes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
}

/// Structured failure report streamed to clients and written under the build prefix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildFailureReport {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<String>,
    pub hints: Vec<String>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub log_excerpt: String,
}

/// Tracks in-flight source builds so clients can request cancellation.
#[derive(Default)]
pub struct ActiveBuilds {
    builds: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

impl ActiveBuilds {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, build_id: &str) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        self.builds
            .lock()
            .expect("active builds lock")
            .insert(build_id.to_owned(), flag.clone());
        flag
    }

    pub fn cancel(&self, build_id: &str) -> bool {
        let flag = self
            .builds
            .lock()
            .expect("active builds lock")
            .get(build_id)
            .cloned();
        if let Some(flag) = flag {
            flag.store(true, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    pub fn finish(&self, build_id: &str) {
        self.builds
            .lock()
            .expect("active builds lock")
            .remove(build_id);
    }
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

/// Paths and target settings available to a build command template.
pub struct CommandArgsContext<'a> {
    pub source: &'a Path,
    pub build: &'a Path,
    pub install: &'a Path,
    pub venv: &'a Path,
    pub python: &'a Path,
    pub recipe: &'a Path,
    pub flags: &'a [String],
    pub parallel_jobs: u16,
}

/// Substitute build placeholders and expand the `{target_flags}` pseudo-argument.
pub fn resolve_command_args(args: &[String], context: &CommandArgsContext<'_>) -> Vec<String> {
    let parallel = context.parallel_jobs.to_string();
    let mut resolved = Vec::with_capacity(args.len());
    for arg in args {
        if arg == "{target_flags}" {
            resolved.extend(context.flags.iter().cloned());
            continue;
        }
        resolved.push(
            arg.replace("{source}", &context.source.display().to_string())
                .replace("{build}", &context.build.display().to_string())
                .replace("{install}", &context.install.display().to_string())
                .replace("{venv}", &context.venv.display().to_string())
                .replace("{python}", &context.python.display().to_string())
                .replace("{recipe}", &context.recipe.display().to_string())
                .replace("{parallel}", &parallel),
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

/// Check the local prerequisites for a llama.cpp source build without creating
/// a checkout. Recommendation selection uses this to avoid offering models
/// that need a fork when this machine cannot build that fork yet.
pub fn llama_cpp_build_preflight(target: RuntimeTarget) -> Result<(), String> {
    for program in ["git", "cmake"] {
        if !command_available(program) {
            return Err(format!("`{program}` is required to build from source"));
        }
    }
    if let Some(message) = toolchain_hints::validate_build_target(target) {
        return Err(message);
    }
    if !cfg!(target_os = "windows")
        && let Some(message) = toolchain_hints::cpp_compiler_preflight_message()
    {
        return Err(message);
    }
    if matches!(target, RuntimeTarget::Rocm)
        && let Some(message) = toolchain_hints::rocm_preflight_message()
    {
        return Err(message);
    }
    Ok(())
}

async fn run_step(
    label: &str,
    program: &str,
    args: &[String],
    workdir: &Path,
    log: &mut String,
    progress: &mut ProgressCallback,
    cancel: &Arc<AtomicBool>,
) -> anyhow::Result<()> {
    if cancel.load(Ordering::Relaxed) {
        anyhow::bail!("build cancelled");
    }
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
    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill().await;
            anyhow::bail!("build cancelled");
        }
        let line = tokio::time::timeout(Duration::from_millis(250), line_rx.recv()).await;
        match line {
            Ok(Some(line)) => {
                log.push_str(&line);
                log.push('\n');
                emitted += 1;
                if emitted.is_multiple_of(5) || emitted < 40 {
                    progress(ProgressEvent::log_line(line));
                }
            }
            Ok(None) => break,
            Err(_) => continue,
        }
    }
    for reader in readers {
        let _ = reader.await;
    }
    if cancel.load(Ordering::Relaxed) {
        let _ = child.kill().await;
        anyhow::bail!("build cancelled");
    }
    let status = child.wait().await.context("wait for build step")?;
    if cancel.load(Ordering::Relaxed) {
        anyhow::bail!("build cancelled");
    }
    anyhow::ensure!(status.success(), "{label} failed with {status}");
    Ok(())
}

fn primary_binary_name(engine: &str) -> &'static str {
    match engine {
        "whisper.cpp" => {
            if cfg!(windows) {
                "whisper-cli.exe"
            } else {
                "whisper-cli"
            }
        }
        "whisperkit" => crate::whisperkit::BINARY_NAME,
        "stable-diffusion.cpp" => {
            if cfg!(windows) {
                "sd-cli.exe"
            } else {
                "sd-cli"
            }
        }
        _ => {
            if cfg!(windows) {
                "llama-server.exe"
            } else {
                "llama-server"
            }
        }
    }
}

/// Copy a Swift PM product from `{build}/release/<name>` into `install/bin`.
fn install_swift_product(
    build_dir: &Path,
    install_bin: &Path,
    product: &str,
) -> anyhow::Result<PathBuf> {
    let candidates = [
        build_dir.join("release").join(product),
        build_dir
            .join("arm64-apple-macosx")
            .join("release")
            .join(product),
        build_dir.join("bin").join(product),
    ];
    let built = candidates
        .into_iter()
        .find(|path| path.is_file())
        .with_context(|| {
            format!(
                "build completed but {product} was not produced under {}",
                build_dir.display()
            )
        })?;
    std::fs::create_dir_all(install_bin).context("create install directory")?;
    let destination = install_bin.join(product);
    std::fs::copy(&built, &destination).with_context(|| format!("install {product}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(&destination)?.permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&destination, permissions)?;
    }
    Ok(destination)
}

/// Copy the built binary and any shared libraries into `install/bin`.
fn install_artifacts(
    build_dir: &Path,
    install_bin: &Path,
    engine: &str,
) -> anyhow::Result<PathBuf> {
    if build_recipe::is_swift_engine(engine) {
        return install_swift_product(build_dir, install_bin, primary_binary_name(engine));
    }
    let server_name = primary_binary_name(engine);
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

fn install_python_env(venv: &Path, engine: &str) -> anyhow::Result<PathBuf> {
    let python = crate::mlx::venv_python(venv);
    anyhow::ensure!(
        python.is_file(),
        "build completed but the virtual environment Python was not created at {}",
        python.display()
    );
    if engine == crate::streaming_asr::ENGINE {
        anyhow::ensure!(
            crate::streaming_asr::python_appears_runnable(&python),
            "Python environment at {} failed an import check for streaming ASR",
            python.display()
        );
        return Ok(python);
    }
    if engine == crate::voice::ENGINE || engine == crate::voice::ENGINE_MLX {
        anyhow::ensure!(
            crate::voice::python_appears_runnable(&python),
            "Python environment at {} failed an import check for PersonaPlex (Moshi or MLX)",
            python.display()
        );
        return Ok(python);
    }
    let kind = crate::mlx::MlxKind::from_engine_id(engine)
        .ok_or_else(|| anyhow::anyhow!("unsupported Python engine `{engine}`"))?;
    anyhow::ensure!(
        crate::mlx::python_appears_runnable(&python, kind),
        "Python environment at {} failed an import check for {}",
        python.display(),
        kind.engine_id()
    );
    Ok(python)
}

fn log_excerpt(log: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = log.lines().collect();
    if lines.len() <= max_lines {
        return log.trim_end().to_owned();
    }
    lines[lines.len() - max_lines..].join("\n")
}

pub fn diagnose_failure(
    message: &str,
    step: Option<&str>,
    log: &str,
    target: RuntimeTarget,
) -> BuildFailureReport {
    let mut hints = Vec::new();
    let log_lower = log.to_ascii_lowercase();
    let message_lower = message.to_ascii_lowercase();

    if message_lower.contains("cancelled") {
        hints.push(
            "The build was stopped before it finished. You can start a new build when ready."
                .into(),
        );
    }
    if message_lower.contains("is not supported on") {
        hints.push(
            "This engine has no build recipe for your operating system and CPU architecture, so it cannot be built on this machine.".into(),
        );
    }
    if message_lower.contains("`git` is required") {
        hints.push(toolchain_hints::install_hint(ToolchainPackage::Git));
    }
    if message_lower.contains("`uv` is required") {
        hints.push(toolchain_hints::install_hint(ToolchainPackage::Uv));
    }
    if message_lower.contains("`cmake` is required") {
        hints.push(toolchain_hints::install_hint(ToolchainPackage::Cmake));
    }
    if toolchain_hints::missing_cpp_compiler(&log_lower, &message_lower) {
        hints.push(toolchain_hints::install_hint(ToolchainPackage::CppBuild));
        if let Some(hint) = toolchain_hints::windows_vs_environment_hint() {
            hints.push(hint);
        }
    }
    if toolchain_hints::missing_cmake_or_vs_generator(&log_lower) {
        hints.push(toolchain_hints::install_hint(ToolchainPackage::CppBuild));
        if let Some(hint) = toolchain_hints::windows_vs_environment_hint() {
            hints.push(hint);
        }
    }
    if log_lower.contains("could not find git")
        || log_lower.contains("not a git repository")
        || log_lower.contains("fatal: unable to access")
        || log_lower.contains("could not read from remote")
    {
        hints.push(
            "Git could not fetch the repository. Check the URL, your network connection, and any authentication required for private forks.".into(),
        );
    }
    if toolchain_hints::missing_cuda(&log_lower) {
        hints.push(toolchain_hints::install_hint(ToolchainPackage::Cuda));
    }
    if toolchain_hints::missing_vulkan(&log_lower, target) {
        hints.push(toolchain_hints::install_hint(ToolchainPackage::Vulkan));
    }
    if toolchain_hints::missing_rocm_hip(&log_lower, &message_lower, target) {
        hints.push(toolchain_hints::install_hint(ToolchainPackage::RocmHip));
        if let Some(hint) = toolchain_hints::rocm_path_hint() {
            hints.push(hint);
        }
    }
    if log_lower.contains("ninja: error")
        || log_lower.contains("make: ***")
        || log_lower.contains("msbuild : error")
        || log_lower.contains("error: ")
    {
        hints.push(
            "The compiler reported errors. Search the log above for the first `error:` line — it usually points to the root cause.".into(),
        );
    }
    if log_lower.contains("killed") || log_lower.contains("out of memory") {
        hints.push(
            "The build may have run out of memory. Close other apps or reduce parallel jobs (for example `cmake --build … --parallel 2`).".into(),
        );
    }
    if message_lower.contains("llama-server was not produced")
        || message_lower.contains("whisper-cli was not produced")
        || message_lower.contains("server binary missing")
    {
        hints.push(
            "The expected binary was not produced. Confirm the recipe still builds the target (`llama-server` or `whisper-cli`) and that the checkout revision is compatible.".into(),
        );
    }
    if message_lower.contains("virtual environment python was not created")
        || message_lower.contains("failed an import check")
    {
        hints.push(
            "The Python environment did not finish correctly. Confirm `uv` is installed and retry the build; the log usually shows the first pip or import error.".into(),
        );
    }
    if step.is_some_and(|label| label.starts_with("Clone ")) && hints.is_empty() {
        hints.push(
            "Clone failed. Verify the repository URL and that the revision (branch, tag, or commit) exists.".into(),
        );
    }
    if step == Some("Checkout selected revision") && hints.is_empty() {
        hints.push(
            "Checkout failed. Verify that the requested revision exists in the repository.".into(),
        );
    }
    if step == Some("Initialize source submodules") {
        hints.push(
            "A required Git submodule could not be checked out. Verify its URL and your access to it, especially when building a private fork.".into(),
        );
    }

    if hints.is_empty() {
        hints.push(
            "Review the build log above, starting from the first error near the failed step."
                .into(),
        );
    }

    BuildFailureReport {
        message: message.to_owned(),
        step: step.map(str::to_owned),
        hints,
        log_excerpt: log_excerpt(log, 40),
    }
}

async fn persist_failure_artifacts(
    root: &Path,
    source: &Path,
    build: &Path,
    log: &str,
    report: &BuildFailureReport,
) {
    let _ = tokio::fs::write(root.join("build.log"), log).await;
    if let Ok(bytes) = serde_json::to_vec_pretty(report) {
        let _ = tokio::fs::write(root.join("failure.json"), bytes).await;
    }
    let _ = tokio::fs::remove_dir_all(source).await;
    let _ = tokio::fs::remove_dir_all(build).await;
}

/// Execute a full source build with streamed progress. Returns the installed
/// binary path. Failed builds keep `build.log` and `failure.json` under the
/// build prefix (large intermediate trees are removed).
pub async fn run_build_with_progress(
    data_dir: &Path,
    request: BuildRequest,
    active_builds: &ActiveBuilds,
    mut progress: ProgressCallback,
) -> Result<PathBuf, BuildFailureReport> {
    let target = request.target.unwrap_or(RuntimeTarget::Cpu);
    let parallel_jobs = request
        .jobs
        .filter(|jobs| *jobs > 0)
        .unwrap_or_else(runtime_settings::default_build_jobs);
    let fail = |message: String, step: Option<&str>, log: &str| -> BuildFailureReport {
        diagnose_failure(&message, step, log, target)
    };

    let platform = match current_platform() {
        Some(platform) => platform,
        None => {
            return Err(fail(
                "source builds are not supported on this platform".into(),
                None,
                "",
            ));
        }
    };
    let plan = match build_recipe::plan(BuildPlanRequest {
        engine: request.engine.clone(),
        repository: request.repository.clone(),
        revision: request.revision.clone(),
        platform: platform.to_owned(),
    }) {
        Ok(plan) => plan,
        Err(error) => return Err(fail(error.to_string(), None, "")),
    };
    let python_engine = build_recipe::is_python_engine(&plan.engine);
    let swift_engine = build_recipe::is_swift_engine(&plan.engine);
    if python_engine {
        if !command_available("uv") {
            return Err(fail(
                "`uv` is required to build Python engine environments; install it and try again"
                    .into(),
                None,
                "",
            ));
        }
    } else if swift_engine {
        // WhisperKit / Argmax: SwiftPM product only (macOS arm64).
        if !command_available("swift") {
            return Err(fail(
                "`swift` is required to build WhisperKit; install Xcode or the Swift toolchain and try again"
                    .into(),
                None,
                "",
            ));
        }
        if !command_available("git") {
            return Err(fail(
                "`git` is required to build from source; install it and try again".into(),
                None,
                "",
            ));
        }
    } else if plan.engine != "llama.cpp"
        && plan.engine != "whisper.cpp"
        && plan.engine != "stable-diffusion.cpp"
    {
        return Err(fail(
            format!("source builds for `{}` are not executable yet", plan.engine),
            None,
            "",
        ));
    }
    if !python_engine && !swift_engine {
        for program in ["git", "cmake"] {
            if !command_available(program) {
                return Err(fail(
                    format!(
                        "`{program}` is required to build from source; install it and try again"
                    ),
                    None,
                    "",
                ));
            }
        }
        if let Some(message) = toolchain_hints::validate_build_target(target) {
            return Err(fail(message, Some("Preflight"), ""));
        }
        if !cfg!(target_os = "windows")
            && let Some(message) = toolchain_hints::cpp_compiler_preflight_message()
        {
            return Err(fail(message, Some("Preflight"), ""));
        }
        if matches!(target, RuntimeTarget::Rocm)
            && let Some(message) = toolchain_hints::rocm_preflight_message()
        {
            return Err(fail(message, Some("Preflight"), ""));
        }
    } else if !plan.skip_checkout && !command_available("git") {
        return Err(fail(
            "`git` is required to build from source; install it and try again".into(),
            None,
            "",
        ));
    }
    if let Some(warning) = &plan.warning {
        progress(ProgressEvent::phase("warning", warning.clone()));
    }

    let build_id = format!(
        "{}-{}",
        sanitize_id_segment(&request.revision),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );
    let cancel = active_builds.register(&build_id);
    progress(ProgressEvent::build_started(&build_id));

    let root = builds_root(data_dir, &plan.engine).join(&build_id);
    if let Err(error) = tokio::fs::create_dir_all(&root).await {
        active_builds.finish(&build_id);
        return Err(fail(error.to_string(), None, ""));
    }
    let root = match std::path::absolute(&root) {
        Ok(root) => root,
        Err(error) => {
            active_builds.finish(&build_id);
            return Err(fail(error.to_string(), None, ""));
        }
    };
    let source = root.join("source");
    let build = root.join("build");
    let install = root.join("install");
    let install_bin = install.join("bin");
    let venv = root.join("venv");
    let python = crate::mlx::venv_python(&venv);
    let recipe_dir = match build_recipe::ensure_recipe_files(data_dir) {
        Ok(dir) => dir,
        Err(error) => {
            active_builds.finish(&build_id);
            return Err(fail(error.to_string(), None, ""));
        }
    };

    let flags = target_flags(target);
    let mut log = String::new();
    let mut failed_step: Option<String> = None;
    let mut built_commit: Option<String> = None;
    let result: Result<PathBuf, anyhow::Error> = async {
        let steps: Vec<&PlannedCommand> = plan.checkout.iter().chain(plan.build.iter()).collect();
        let total = steps.len();
        for (index, step) in steps.into_iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("build cancelled");
            }
            failed_step = Some(step.label.clone());
            let args = resolve_command_args(
                &step.args,
                &CommandArgsContext {
                    source: &source,
                    build: &build,
                    install: &install,
                    venv: &venv,
                    python: &python,
                    recipe: &recipe_dir,
                    flags: &flags,
                    parallel_jobs,
                },
            );
            progress(ProgressEvent::build_step(
                index + 1,
                total,
                step.label.clone(),
            ));
            run_step(
                &step.label,
                &step.program,
                &args,
                &root,
                &mut log,
                &mut progress,
                &cancel,
            )
            .await?;
        }
        progress(ProgressEvent::phase(
            "install",
            if python_engine {
                "Verifying the Python virtual environment"
            } else if swift_engine {
                "Installing the Swift CLI into an isolated prefix"
            } else {
                "Installing the built server into an isolated prefix"
            },
        ));
        failed_step = Some("install".into());
        let binary = if python_engine {
            install_python_env(&venv, &plan.engine)?
        } else {
            install_artifacts(&build, &install_bin, &plan.engine)?
        };
        if swift_engine {
            anyhow::ensure!(
                crate::whisperkit::binary_appears_runnable(&binary),
                "Swift CLI at {} failed a smoke test",
                binary.display()
            );
        }
        // Record the exact commit that was built so update checks can compare it
        // against the upstream ref later, before the checkout is discarded.
        if source.join(".git").exists()
            && let Ok(output) = tokio::process::Command::new("git")
                .arg("-C")
                .arg(&source)
                .arg("rev-parse")
                .arg("HEAD")
                .output()
                .await
            && output.status.success()
        {
            let sha = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if !sha.is_empty() {
                built_commit = Some(sha);
            }
        }
        let _ = tokio::fs::remove_dir_all(&source).await;
        if !python_engine {
            let _ = tokio::fs::remove_dir_all(&build).await;
        }
        Ok(binary)
    }
    .await;

    active_builds.finish(&build_id);

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
                commit: built_commit,
            };
            if let Ok(bytes) = serde_json::to_vec_pretty(&record) {
                let _ = tokio::fs::write(root.join("build.json"), bytes).await;
            }
            let _ = tokio::fs::write(root.join("build.log"), &log).await;
            progress(ProgressEvent::done(serde_json::json!({
                "binary": binary.display().to_string(),
                "build_id": build_id,
                "status": "ready"
            })));
            Ok(binary)
        }
        Err(error) => {
            tracing::error!(error = %error, log = %log, "source build failed");
            let report = fail(error.to_string(), failed_step.as_deref(), &log);
            persist_failure_artifacts(&root, &source, &build, &log, &report).await;
            progress(ProgressEvent::build_failed(
                &serde_json::to_value(&report)
                    .unwrap_or_else(|_| serde_json::json!({ "message": report.message.clone() })),
            ));
            Err(report)
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
            &CommandArgsContext {
                source: Path::new("/tmp/s"),
                build: Path::new("/tmp/b"),
                install: Path::new("/tmp/i"),
                venv: Path::new("/tmp/v"),
                python: Path::new("/tmp/v/bin/python"),
                recipe: Path::new("/tmp/recipes"),
                flags: &["-DGGML_CUDA=ON".to_owned()],
                parallel_jobs: 4,
            },
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
            &CommandArgsContext {
                source: Path::new("/s"),
                build: Path::new("/b"),
                install: Path::new("/i"),
                venv: Path::new("/v"),
                python: Path::new("/v/bin/python"),
                recipe: Path::new("/recipes"),
                flags: &[],
                parallel_jobs: 2,
            },
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
    fn checkout_failures_receive_step_specific_hints() {
        let clone = diagnose_failure(
            "Clone source without running hooks failed with exit status: 128",
            Some("Clone source without running hooks"),
            "",
            RuntimeTarget::Cpu,
        );
        assert!(
            clone
                .hints
                .iter()
                .any(|hint| hint.starts_with("Clone failed."))
        );

        let checkout = diagnose_failure(
            "Checkout selected revision failed with exit status: 128",
            Some("Checkout selected revision"),
            "",
            RuntimeTarget::Cpu,
        );
        assert!(
            checkout
                .hints
                .iter()
                .any(|hint| hint.starts_with("Checkout failed."))
        );

        let submodule = diagnose_failure(
            "Initialize source submodules failed with exit status: 128",
            Some("Initialize source submodules"),
            "fatal: could not read from remote repository",
            RuntimeTarget::Cpu,
        );
        assert!(
            submodule
                .hints
                .iter()
                .any(|hint| hint.contains("required Git submodule"))
        );
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
                commit: None,
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

    #[test]
    fn cancel_flag_is_observed() {
        let active = ActiveBuilds::new();
        let flag = active.register("demo-1");
        assert!(!flag.load(Ordering::Relaxed));
        assert!(active.cancel("demo-1"));
        assert!(flag.load(Ordering::Relaxed));
        assert!(!active.cancel("missing"));
    }

    #[test]
    fn diagnostics_surface_missing_msvc() {
        let report = diagnose_failure(
            "Configure failed with 1",
            Some("Configure"),
            "CMake Error: Could not find any instance of Visual Studio",
            RuntimeTarget::Cpu,
        );
        assert!(report.hints.iter().any(|hint| {
            let lower = hint.to_ascii_lowercase();
            lower.contains("c++")
                || lower.contains("compiler")
                || lower.contains("visual studio")
                || lower.contains("build-essential")
                || lower.contains("base-devel")
        }));
    }

    #[test]
    fn diagnostics_surface_missing_cuda() {
        let report = diagnose_failure(
            "Configure failed with 1",
            Some("Configure"),
            "CMake Error: Could NOT find CUDA (missing: CUDA_TOOLKIT_ROOT_DIR)",
            RuntimeTarget::Cuda,
        );
        assert!(
            report
                .hints
                .iter()
                .any(|hint| hint.to_ascii_lowercase().contains("cuda"))
        );
    }

    #[test]
    fn diagnostics_surface_missing_rocm_hip() {
        let report = diagnose_failure(
            "Configure failed with 1",
            Some("Configure"),
            "CMake Error: Could NOT find HIP (missing: HIP_LIBRARY HIP_INCLUDE_DIR)",
            RuntimeTarget::Rocm,
        );
        assert!(
            report
                .hints
                .iter()
                .any(|hint| hint.to_ascii_lowercase().contains("pacman") || hint.contains("ROCm"))
        );
    }

    #[test]
    fn diagnostics_mark_cancelled_builds() {
        let report = diagnose_failure("build cancelled", None, "", RuntimeTarget::Cpu);
        assert!(report.message.contains("cancelled"));
        assert!(
            report
                .hints
                .iter()
                .any(|hint| hint.to_ascii_lowercase().contains("stopped"))
        );
    }
}

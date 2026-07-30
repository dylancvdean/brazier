//! Inventory and lifecycle of installed inference runtimes.
//!
//! Tracks llama-server binaries and MLX Python virtual environments built from
//! approved source recipes.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Serialize;

use crate::{builds, llama};

pub const ENGINE: &str = "llama.cpp";
const MANAGED_FLAVORS: &[&str] = &["cuda", "rocm", "vulkan"];
const PYTHON_ENGINES: &[&str] = &[
    "mlx-lm",
    "mlx-vlm",
    "vllm",
    "streaming-asr",
    "personaplex",
    "personaplex-mlx",
];
const SDCPP_MANAGED_FLAVORS: &[&str] = &["cuda", "rocm", "vulkan"];

fn llama_target_label(target: &str) -> &str {
    match target {
        "cpu" => "CPU",
        "cuda" => "CUDA",
        "rocm" => "ROCm",
        "vulkan" => "Vulkan",
        "metal" => "Metal",
        _ => target,
    }
}

fn llama_managed_label(target: &str) -> String {
    format!("llama.cpp · {}", llama_target_label(target))
}

fn default_llama_managed_label() -> String {
    // The macOS release bundle contains Metal support in the same install
    // location as the CPU fallback. Calling it "CPU" made a correctly
    // installed Apple-Silicon runtime look like the wrong download.
    #[cfg(target_os = "macos")]
    {
        llama_managed_label("metal")
    }
    #[cfg(not(target_os = "macos"))]
    {
        llama_managed_label("cpu")
    }
}

fn source_label(record: &builds::BuildRecord) -> String {
    if let Some(name) = record.name.as_deref() {
        return name.to_owned();
    }
    let fork = crate::fork_hints::normalize_github_repo_url(&record.repository)
        .and_then(|url| {
            let mut parts = url.rsplit('/');
            let repository = parts.next()?;
            let owner = parts.next()?;
            Some(if repository.eq_ignore_ascii_case("llama.cpp") {
                format!("{owner} llama.cpp")
            } else {
                repository.to_owned()
            })
        })
        .filter(|name| !name.eq_ignore_ascii_case("ggml-org llama.cpp"));
    match fork {
        Some(fork) => format!(
            "llama.cpp · {} · Source · {} · {}",
            fork,
            record.revision,
            build_stamp(record)
        ),
        None => format!(
            "llama.cpp · Source · {} · {}",
            record.revision,
            build_stamp(record)
        ),
    }
}

#[derive(Debug, Clone, Default)]
pub struct ActiveRuntimes {
    pub llama: Option<PathBuf>,
    pub mlx_lm: Option<PathBuf>,
    pub mlx_vlm: Option<PathBuf>,
    pub vllm: Option<PathBuf>,
    pub whisper: Option<PathBuf>,
    pub streaming_asr: Option<PathBuf>,
    pub sdcpp: Option<PathBuf>,
    pub voice: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeEntry {
    pub id: String,
    pub engine: String,
    pub kind: String,
    pub label: String,
    pub target: Option<String>,
    pub version: Option<String>,
    /// Source repository URL when this runtime was built from a fork.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    pub path: String,
    pub active: bool,
    pub deletable: bool,
}

fn read_version(dir: &Path) -> Option<String> {
    std::fs::read_to_string(dir.join("VERSION"))
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// Whether a path is the `bin/python` of a virtual environment.
///
/// `pyvenv.cfg` beside the `bin` directory is what makes a virtualenv one, so
/// it is the honest test.
fn is_venv_interpreter(path: &Path) -> bool {
    path.parent()
        .and_then(Path::parent)
        .is_some_and(|root| root.join("pyvenv.cfg").is_file())
}

/// Calendar date and time (UTC) for a Unix timestamp, as `YYYY-MM-DD HH:MM`.
///
/// The date alone is not enough: rebuilding a branch twice in an afternoon is
/// the normal case, and that is exactly when the rows need telling apart.
fn civil_timestamp(seconds: i64) -> String {
    let seconds_of_day = seconds.rem_euclid(86_400);
    format!(
        "{} {:02}:{:02}",
        civil_date(seconds),
        seconds_of_day / 3_600,
        (seconds_of_day % 3_600) / 60
    )
}

/// Calendar date (UTC) for a Unix timestamp, as `YYYY-MM-DD`.
///
/// Days-to-civil, from Howard Hinnant's `chrono`-compatible algorithm. One
/// label does not justify a date dependency.
fn civil_date(seconds: i64) -> String {
    let days = seconds.div_euclid(86_400);
    // Shift the epoch to 0000-03-01 so leap days land at the end of the cycle.
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_position = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_position + 2) / 5 + 1;
    let month = if month_position < 10 {
        month_position + 3
    } else {
        month_position - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Build date, so rebuilds of one revision are distinguishable.
///
/// Every build of `main` is otherwise labelled identically, and the list then
/// gives no way to tell the row just activated from the two beside it. Records
/// store Unix seconds; older ones may hold an ISO timestamp.
fn build_stamp(record: &builds::BuildRecord) -> String {
    let raw = record.created_at.trim();
    if let Ok(seconds) = raw.parse::<i64>() {
        return civil_timestamp(seconds);
    }
    match raw.split('T').next().unwrap_or_default() {
        "" => "undated".to_owned(),
        date => date.to_owned(),
    }
}

fn same_file(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    // Two virtualenvs are two runtimes even when their interpreters are the
    // same file: `venv/bin/python` is a symlink to a shared base interpreter,
    // which uv guarantees, so resolving it throws away the only thing telling
    // the builds apart and every one of them reports as the active runtime.
    if is_venv_interpreter(a) && is_venv_interpreter(b) {
        return false;
    }
    // Elsewhere resolving is what makes a discovered binary and the managed
    // install it points at recognisable as one runtime.
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

/// Enumerate every known runtime on this machine.
pub fn list(
    data_dir: &Path,
    active: &ActiveRuntimes,
    path_env: Option<&str>,
    include_system: bool,
) -> Vec<RuntimeEntry> {
    let mut entries = Vec::new();
    let engine_dir = llama::managed_engine_dir(data_dir);
    let is_active = |path: &Path, selected: &Option<PathBuf>| {
        selected
            .as_deref()
            .is_some_and(|active_path| same_file(path, active_path))
    };

    // Managed default (CPU / Metal) install.
    let default_binary = llama::managed_binary_path(data_dir);
    if default_binary.is_file() {
        entries.push(RuntimeEntry {
            id: "managed".to_owned(),
            engine: ENGINE.to_owned(),
            kind: "managed".to_owned(),
            label: default_llama_managed_label(),
            target: Some(
                if cfg!(target_os = "macos") {
                    "metal"
                } else {
                    "cpu"
                }
                .to_owned(),
            ),
            version: read_version(&engine_dir),
            repository: None,
            path: default_binary.display().to_string(),
            active: is_active(&default_binary, &active.llama),
            deletable: true,
        });
    }

    // Managed acceleration flavors.
    for flavor in MANAGED_FLAVORS {
        let flavor_dir = engine_dir.join(flavor);
        let binary = flavor_dir.join("bin").join(if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        });
        if binary.is_file() {
            entries.push(RuntimeEntry {
                id: format!("managed-{flavor}"),
                engine: ENGINE.to_owned(),
                kind: "managed".to_owned(),
                label: llama_managed_label(flavor),
                target: Some((*flavor).to_owned()),
                version: read_version(&flavor_dir),
                repository: None,
                path: binary.display().to_string(),
                active: is_active(&binary, &active.llama),
                deletable: true,
            });
        }
    }

    // Source builds.
    for (build_id, record) in builds::list_builds(data_dir, ENGINE) {
        let path = PathBuf::from(&record.binary);
        entries.push(RuntimeEntry {
            id: format!("source-{build_id}"),
            engine: ENGINE.to_owned(),
            kind: "source".to_owned(),
            label: source_label(&record),
            target: Some(record.target.clone()),
            version: Some(record.revision.clone()),
            repository: Some(record.repository.clone()),
            path: record.binary.clone(),
            active: is_active(&path, &active.llama),
            deletable: true,
        });
    }

    for engine in PYTHON_ENGINES {
        for (build_id, record) in builds::list_builds(data_dir, engine) {
            let path = PathBuf::from(&record.binary);
            let display = match *engine {
                "mlx-lm" => "MLX-LM",
                "mlx-vlm" => "MLX-VLM",
                "vllm" => "vLLM",
                "streaming-asr" => "Streaming ASR",
                "personaplex" => "PersonaPlex",
                "personaplex-mlx" => "PersonaPlex MLX",
                other => other,
            };
            let selected = match *engine {
                "mlx-lm" => &active.mlx_lm,
                "mlx-vlm" => &active.mlx_vlm,
                "vllm" => &active.vllm,
                "streaming-asr" => &active.streaming_asr,
                "personaplex" | "personaplex-mlx" => &active.voice,
                _ => &None,
            };
            entries.push(RuntimeEntry {
                id: format!("{engine}-source-{build_id}"),
                engine: (*engine).to_owned(),
                kind: "source".to_owned(),
                label: format!("{display} · {} · {}", record.revision, build_stamp(&record)),
                target: Some(record.target.clone()),
                version: Some(record.revision.clone()),
                repository: Some(record.repository.clone()),
                path: record.binary.clone(),
                active: is_active(&path, selected),
                deletable: true,
            });
        }
    }

    // Managed stable-diffusion.cpp installs.
    let sdcpp_engine_dir = crate::sdcpp::managed_engine_dir(data_dir);
    let sdcpp_default = crate::sdcpp::managed_binary_path(data_dir);
    if sdcpp_default.is_file() {
        entries.push(RuntimeEntry {
            id: "sdcpp-managed".to_owned(),
            engine: crate::sdcpp::ENGINE.to_owned(),
            kind: "managed".to_owned(),
            // Like llama.cpp, the macOS release has Metal enabled in the
            // default install location alongside its CPU fallback.
            label: format!(
                "stable-diffusion.cpp · {}",
                if cfg!(target_os = "macos") {
                    "Metal"
                } else {
                    "CPU"
                }
            ),
            target: Some(
                if cfg!(target_os = "macos") {
                    "metal"
                } else {
                    "cpu"
                }
                .to_owned(),
            ),
            version: read_version(&sdcpp_engine_dir),
            repository: None,
            path: sdcpp_default.display().to_string(),
            active: is_active(&sdcpp_default, &active.sdcpp),
            deletable: true,
        });
    }
    for flavor in SDCPP_MANAGED_FLAVORS {
        let flavor_dir = sdcpp_engine_dir.join(flavor);
        let binary = flavor_dir.join("bin").join(crate::sdcpp::binary_name());
        if binary.is_file() {
            entries.push(RuntimeEntry {
                id: format!("sdcpp-managed-{flavor}"),
                engine: crate::sdcpp::ENGINE.to_owned(),
                kind: "managed".to_owned(),
                label: format!("stable-diffusion.cpp · {}", llama_target_label(flavor)),
                target: Some((*flavor).to_owned()),
                version: read_version(&flavor_dir),
                repository: None,
                path: binary.display().to_string(),
                active: is_active(&binary, &active.sdcpp),
                deletable: true,
            });
        }
    }
    for (build_id, record) in builds::list_builds(data_dir, crate::sdcpp::ENGINE) {
        let path = PathBuf::from(&record.binary);
        entries.push(RuntimeEntry {
            id: format!("sdcpp-source-{build_id}"),
            engine: crate::sdcpp::ENGINE.to_owned(),
            kind: "source".to_owned(),
            label: format!("stable-diffusion.cpp · Source · {}", record.revision),
            target: Some(record.target.clone()),
            version: Some(record.revision.clone()),
            repository: Some(record.repository.clone()),
            path: record.binary.clone(),
            active: is_active(&path, &active.sdcpp),
            deletable: true,
        });
    }

    // Managed whisper.cpp installs (Linux/Windows CLI prebuilts).
    let whisper_engine_dir = crate::whisper::managed_engine_dir(data_dir);
    let whisper_default = crate::whisper::managed_binary_path(data_dir);
    if whisper_default.is_file() {
        entries.push(RuntimeEntry {
            id: "whisper-managed".to_owned(),
            engine: crate::whisper::ENGINE.to_owned(),
            kind: "managed".to_owned(),
            label: "whisper.cpp · CPU".to_owned(),
            target: Some("cpu".to_owned()),
            version: read_version(&whisper_engine_dir),
            repository: None,
            path: whisper_default.display().to_string(),
            active: is_active(&whisper_default, &active.whisper),
            deletable: true,
        });
    }
    {
        let flavor = "cuda";
        let flavor_dir = whisper_engine_dir.join(flavor);
        let binary = flavor_dir.join("bin").join(crate::whisper::binary_name());
        if binary.is_file() {
            entries.push(RuntimeEntry {
                id: format!("whisper-managed-{flavor}"),
                engine: crate::whisper::ENGINE.to_owned(),
                kind: "managed".to_owned(),
                label: format!("whisper.cpp · {}", llama_target_label(flavor)),
                target: Some((*flavor).to_owned()),
                version: read_version(&flavor_dir),
                repository: None,
                path: binary.display().to_string(),
                active: is_active(&binary, &active.whisper),
                deletable: true,
            });
        }
    }

    for (build_id, record) in builds::list_builds(data_dir, crate::whisper::ENGINE) {
        let path = PathBuf::from(&record.binary);
        entries.push(RuntimeEntry {
            id: format!("whisper-source-{build_id}"),
            engine: crate::whisper::ENGINE.to_owned(),
            kind: "source".to_owned(),
            label: format!("whisper.cpp · Source · {}", record.revision),
            target: Some(record.target.clone()),
            version: Some(record.revision.clone()),
            repository: Some(record.repository.clone()),
            path: record.binary.clone(),
            active: is_active(&path, &active.whisper),
            deletable: true,
        });
    }

    // WhisperKit (Argmax) source builds — Apple Silicon CoreML ASR.
    for (build_id, record) in builds::list_builds(data_dir, crate::whisperkit::ENGINE) {
        let path = PathBuf::from(&record.binary);
        entries.push(RuntimeEntry {
            id: format!("whisperkit-source-{build_id}"),
            engine: crate::whisperkit::ENGINE.to_owned(),
            kind: "source".to_owned(),
            label: format!("WhisperKit · {}", record.revision),
            target: Some(record.target.clone()),
            version: Some(record.revision.clone()),
            repository: Some(record.repository.clone()),
            path: record.binary.clone(),
            active: is_active(&path, &active.whisper),
            deletable: true,
        });
    }

    // System binaries on PATH or well-known prefixes (optional — can be slow).
    if !include_system {
        return entries;
    }
    let data_prefix = data_dir
        .canonicalize()
        .unwrap_or_else(|_| data_dir.to_path_buf());
    for candidate in llama::discovery_candidates(data_dir, path_env)
        .into_iter()
        .skip(1)
        .filter(|path| path.is_file())
    {
        let canonical = candidate
            .canonicalize()
            .unwrap_or_else(|_| candidate.clone());
        if canonical.starts_with(&data_prefix) {
            continue;
        }
        if entries
            .iter()
            .any(|entry| same_file(Path::new(&entry.path), &candidate))
        {
            continue;
        }
        entries.push(RuntimeEntry {
            id: format!("system-{}", canonical.display()),
            engine: ENGINE.to_owned(),
            kind: "system".to_owned(),
            label: "llama.cpp · System".to_owned(),
            target: None,
            version: None,
            repository: None,
            path: candidate.display().to_string(),
            active: is_active(&candidate, &active.llama),
            deletable: false,
        });
    }
    for candidate in crate::whisper::discovery_candidates(data_dir, path_env)
        .into_iter()
        .filter(|path| path.is_file())
        .filter(|path| !crate::whisperkit::is_whisperkit_binary(path))
    {
        let canonical = candidate
            .canonicalize()
            .unwrap_or_else(|_| candidate.clone());
        if canonical.starts_with(&data_prefix) {
            continue;
        }
        if entries
            .iter()
            .any(|entry| same_file(Path::new(&entry.path), &candidate))
        {
            continue;
        }
        entries.push(RuntimeEntry {
            id: format!("whisper-system-{}", canonical.display()),
            engine: crate::whisper::ENGINE.to_owned(),
            kind: "system".to_owned(),
            label: "whisper.cpp · System".to_owned(),
            target: None,
            version: None,
            repository: None,
            path: candidate.display().to_string(),
            active: is_active(&candidate, &active.whisper),
            deletable: false,
        });
    }
    for candidate in crate::whisperkit::discovery_candidates(data_dir, path_env)
        .into_iter()
        .filter(|path| path.is_file())
    {
        let canonical = candidate
            .canonicalize()
            .unwrap_or_else(|_| candidate.clone());
        if canonical.starts_with(&data_prefix) {
            continue;
        }
        if entries
            .iter()
            .any(|entry| same_file(Path::new(&entry.path), &candidate))
        {
            continue;
        }
        entries.push(RuntimeEntry {
            id: format!("whisperkit-system-{}", canonical.display()),
            engine: crate::whisperkit::ENGINE.to_owned(),
            kind: "system".to_owned(),
            label: "WhisperKit · System".to_owned(),
            target: None,
            version: None,
            repository: None,
            path: candidate.display().to_string(),
            active: is_active(&candidate, &active.whisper),
            deletable: false,
        });
    }
    entries
}

/// Upstream update status for a single source-built runtime.
#[derive(Debug, Clone, Serialize)]
pub struct SourceUpdate {
    pub id: String,
    pub engine: String,
    pub label: String,
    pub repository: String,
    pub revision: String,
    /// Short commit that was built, when it was recorded.
    pub current_commit: Option<String>,
    /// Short commit the upstream ref currently points at.
    pub upstream_commit: Option<String>,
    pub update_available: bool,
    /// Revision is a pinned commit — there is no moving ref to track.
    pub pinned: bool,
    pub error: Option<String>,
}

fn short_commit(sha: &str) -> String {
    sha.chars().take(12).collect()
}

fn looks_like_commit_sha(revision: &str) -> bool {
    let len = revision.len();
    (7..=40).contains(&len) && revision.chars().all(|c| c.is_ascii_hexdigit())
}

/// Resolve the commit an upstream ref currently points at via `git ls-remote`.
/// Returns `Ok(None)` when the ref is absent on the remote.
fn ls_remote_commit(repository: &str, revision: &str) -> anyhow::Result<Option<String>> {
    let output = std::process::Command::new("git")
        .arg("ls-remote")
        .arg(repository)
        .arg(revision)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .context("run git ls-remote")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let trimmed = stderr.trim();
        if trimmed.is_empty() {
            anyhow::bail!("git ls-remote failed");
        }
        anyhow::bail!("git ls-remote failed: {trimmed}");
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let head_ref = format!("refs/heads/{revision}");
    let tag_ref = format!("refs/tags/{revision}");
    let mut first: Option<String> = None;
    for line in stdout.lines() {
        let mut parts = line.split_whitespace();
        let sha = parts.next().unwrap_or("");
        let refname = parts.next().unwrap_or("");
        if sha.is_empty() {
            continue;
        }
        if first.is_none() {
            first = Some(sha.to_owned());
        }
        if refname == head_ref || refname == tag_ref {
            return Ok(Some(sha.to_owned()));
        }
    }
    Ok(first)
}

fn source_id_prefix(engine: &str) -> &'static str {
    match engine {
        "mlx-lm" => "mlx-lm-source-",
        "mlx-vlm" => "mlx-vlm-source-",
        "streaming-asr" => "streaming-asr-source-",
        "personaplex" => "personaplex-source-",
        "personaplex-mlx" => "personaplex-mlx-source-",
        "stable-diffusion.cpp" => "sdcpp-source-",
        "whisper.cpp" => "whisper-source-",
        "whisperkit" => "whisperkit-source-",
        _ => "source-",
    }
}

fn source_engine_label(engine: &str) -> &str {
    match engine {
        "mlx-lm" => "MLX-LM",
        "mlx-vlm" => "MLX-VLM",
        "streaming-asr" => "Streaming ASR",
        "personaplex" => "PersonaPlex",
        "personaplex-mlx" => "PersonaPlex MLX",
        "whisperkit" => "WhisperKit",
        other => other,
    }
}

/// Check every source-built runtime against its upstream ref. Performs blocking
/// network I/O (`git ls-remote`); callers should run it off the async runtime.
pub fn check_source_updates(data_dir: &Path) -> Vec<SourceUpdate> {
    let engines = [
        ENGINE,
        "mlx-lm",
        "mlx-vlm",
        "streaming-asr",
        "personaplex",
        "personaplex-mlx",
        crate::sdcpp::ENGINE,
        crate::whisper::ENGINE,
        crate::whisperkit::ENGINE,
    ];
    // Dedupe remote queries so multiple builds of the same ref hit the network once.
    let mut remote_cache: std::collections::HashMap<
        (String, String),
        Result<Option<String>, String>,
    > = std::collections::HashMap::new();
    let mut out = Vec::new();
    for engine in engines {
        for (build_id, record) in builds::list_builds(data_dir, engine) {
            let id = format!("{}{build_id}", source_id_prefix(engine));
            let label = format!("{} · {}", source_engine_label(engine), record.revision);
            let current_commit = record.commit.as_deref().map(short_commit);
            if looks_like_commit_sha(&record.revision) {
                out.push(SourceUpdate {
                    id,
                    engine: engine.to_owned(),
                    label,
                    repository: record.repository.clone(),
                    revision: record.revision.clone(),
                    current_commit: current_commit.clone(),
                    upstream_commit: current_commit,
                    update_available: false,
                    pinned: true,
                    error: None,
                });
                continue;
            }
            let resolved = remote_cache
                .entry((record.repository.clone(), record.revision.clone()))
                .or_insert_with(|| {
                    ls_remote_commit(&record.repository, &record.revision)
                        .map_err(|error| error.to_string())
                })
                .clone();
            let (upstream_commit, update_available, error) = match resolved {
                Ok(Some(upstream)) => {
                    let available = record
                        .commit
                        .as_deref()
                        .is_some_and(|built| built != upstream);
                    (Some(short_commit(&upstream)), available, None)
                }
                Ok(None) => (None, false, Some("Upstream ref not found".to_owned())),
                Err(message) => (None, false, Some(message)),
            };
            out.push(SourceUpdate {
                id,
                engine: engine.to_owned(),
                label,
                repository: record.repository.clone(),
                revision: record.revision.clone(),
                current_commit,
                upstream_commit,
                update_available,
                pinned: false,
                error,
            });
        }
    }
    out
}

/// Resolve a runtime id from `list` back to its entry.
pub fn find(
    data_dir: &Path,
    path_env: Option<&str>,
    id: &str,
    include_system: bool,
    active: &ActiveRuntimes,
) -> Option<RuntimeEntry> {
    list(data_dir, active, path_env, include_system)
        .into_iter()
        .find(|entry| entry.id == id)
}

/// Find an installed runtime that matches a README fork hint.
pub fn find_for_fork(
    data_dir: &Path,
    active: &ActiveRuntimes,
    hint: &crate::fork_hints::RuntimeForkHint,
) -> Option<RuntimeEntry> {
    let normalized = crate::fork_hints::normalize_github_repo_url(&hint.repository)?;
    list(data_dir, active, None, false)
        .into_iter()
        .find(|entry| {
            if entry.engine != hint.engine {
                return false;
            }
            entry
                .repository
                .as_deref()
                .and_then(crate::fork_hints::normalize_github_repo_url)
                .as_deref()
                == Some(normalized.as_str())
        })
}

/// Delete a managed or source runtime installation. Returns the binary path
/// that was removed so callers can release it if it was active.
pub fn delete(data_dir: &Path, id: &str) -> anyhow::Result<PathBuf> {
    let engine_dir = llama::managed_engine_dir(data_dir);
    if id == "managed" {
        let binary = llama::managed_binary_path(data_dir);
        anyhow::ensure!(binary.is_file(), "managed runtime is not installed");
        let bin_dir = binary.parent().context("managed binary has no parent")?;
        std::fs::remove_dir_all(bin_dir).context("remove managed runtime")?;
        let _ = std::fs::remove_file(engine_dir.join("VERSION"));
        return Ok(binary);
    }
    if let Some(flavor) = id.strip_prefix("managed-") {
        anyhow::ensure!(
            MANAGED_FLAVORS.contains(&flavor),
            "unknown managed runtime `{id}`"
        );
        let flavor_dir = engine_dir.join(flavor);
        let binary = flavor_dir.join("bin").join(if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        });
        anyhow::ensure!(
            flavor_dir.is_dir(),
            "managed runtime `{id}` is not installed"
        );
        std::fs::remove_dir_all(&flavor_dir).context("remove managed runtime")?;
        return Ok(binary);
    }
    if let Some(build_id) = id.strip_prefix("source-") {
        // Guard against traversal: the build id must resolve to a direct child.
        anyhow::ensure!(
            !build_id.is_empty()
                && !build_id.contains('/')
                && !build_id.contains('\\')
                && build_id != "."
                && build_id != "..",
            "invalid build id"
        );
        let root = builds::builds_root(data_dir, ENGINE).join(build_id);
        anyhow::ensure!(root.is_dir(), "source build `{build_id}` does not exist");
        let binary = root.join("install").join("bin").join(if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        });
        std::fs::remove_dir_all(&root).context("remove source build")?;
        return Ok(binary);
    }
    for engine in PYTHON_ENGINES {
        let prefix = format!("{engine}-source-");
        if let Some(build_id) = id.strip_prefix(&prefix) {
            anyhow::ensure!(
                !build_id.is_empty()
                    && !build_id.contains('/')
                    && !build_id.contains('\\')
                    && build_id != "."
                    && build_id != "..",
                "invalid build id"
            );
            let root = builds::builds_root(data_dir, engine).join(build_id);
            anyhow::ensure!(root.is_dir(), "source build `{build_id}` does not exist");
            let python = crate::mlx::venv_python(&root.join("venv"));
            std::fs::remove_dir_all(&root).context("remove source build")?;
            return Ok(python);
        }
    }
    if id == "sdcpp-managed" {
        let binary = crate::sdcpp::managed_binary_path(data_dir);
        anyhow::ensure!(binary.is_file(), "managed sd.cpp runtime is not installed");
        let bin_dir = binary.parent().context("managed binary has no parent")?;
        std::fs::remove_dir_all(bin_dir).context("remove managed sd.cpp runtime")?;
        let _ = std::fs::remove_file(crate::sdcpp::managed_engine_dir(data_dir).join("VERSION"));
        return Ok(binary);
    }
    if let Some(flavor) = id.strip_prefix("sdcpp-managed-") {
        anyhow::ensure!(
            SDCPP_MANAGED_FLAVORS.contains(&flavor),
            "unknown managed sd.cpp runtime `{id}`"
        );
        let flavor_dir = crate::sdcpp::managed_engine_dir(data_dir).join(flavor);
        let binary = flavor_dir.join("bin").join(crate::sdcpp::binary_name());
        anyhow::ensure!(
            flavor_dir.is_dir(),
            "managed sd.cpp runtime `{id}` is not installed"
        );
        std::fs::remove_dir_all(&flavor_dir).context("remove managed sd.cpp runtime")?;
        return Ok(binary);
    }
    if let Some(build_id) = id.strip_prefix("sdcpp-source-") {
        anyhow::ensure!(
            !build_id.is_empty()
                && !build_id.contains('/')
                && !build_id.contains('\\')
                && build_id != "."
                && build_id != "..",
            "invalid build id"
        );
        let root = builds::builds_root(data_dir, crate::sdcpp::ENGINE).join(build_id);
        anyhow::ensure!(root.is_dir(), "source build `{build_id}` does not exist");
        let binary = root
            .join("install")
            .join("bin")
            .join(crate::sdcpp::binary_name());
        std::fs::remove_dir_all(&root).context("remove source build")?;
        return Ok(binary);
    }
    if id == "whisper-managed" {
        let binary = crate::whisper::managed_binary_path(data_dir);
        anyhow::ensure!(binary.is_file(), "managed whisper runtime is not installed");
        let bin_dir = binary.parent().context("managed binary has no parent")?;
        std::fs::remove_dir_all(bin_dir).context("remove managed whisper runtime")?;
        let _ = std::fs::remove_file(crate::whisper::managed_engine_dir(data_dir).join("VERSION"));
        return Ok(binary);
    }
    if let Some(flavor) = id.strip_prefix("whisper-managed-") {
        anyhow::ensure!(flavor == "cuda", "unknown managed whisper runtime `{id}`");
        let flavor_dir = crate::whisper::managed_engine_dir(data_dir).join(flavor);
        let binary = flavor_dir.join("bin").join(crate::whisper::binary_name());
        anyhow::ensure!(
            flavor_dir.is_dir(),
            "managed whisper runtime `{id}` is not installed"
        );
        std::fs::remove_dir_all(&flavor_dir).context("remove managed whisper runtime")?;
        return Ok(binary);
    }
    if let Some(build_id) = id.strip_prefix("whisper-source-") {
        anyhow::ensure!(
            !build_id.is_empty()
                && !build_id.contains('/')
                && !build_id.contains('\\')
                && build_id != "."
                && build_id != "..",
            "invalid build id"
        );
        let root = builds::builds_root(data_dir, crate::whisper::ENGINE).join(build_id);
        anyhow::ensure!(root.is_dir(), "source build `{build_id}` does not exist");
        let binary = root
            .join("install")
            .join("bin")
            .join(crate::whisper::binary_name());
        std::fs::remove_dir_all(&root).context("remove source build")?;
        return Ok(binary);
    }
    if let Some(build_id) = id.strip_prefix("whisperkit-source-") {
        anyhow::ensure!(
            !build_id.is_empty()
                && !build_id.contains('/')
                && !build_id.contains('\\')
                && build_id != "."
                && build_id != "..",
            "invalid build id"
        );
        let root = builds::builds_root(data_dir, crate::whisperkit::ENGINE).join(build_id);
        anyhow::ensure!(root.is_dir(), "source build `{build_id}` does not exist");
        let binary = root
            .join("install")
            .join("bin")
            .join(crate::whisperkit::BINARY_NAME);
        std::fs::remove_dir_all(&root).context("remove source build")?;
        return Ok(binary);
    }
    anyhow::bail!("runtime `{id}` cannot be deleted");
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"bin").unwrap();
    }

    #[test]
    fn lists_managed_flavors_and_source_builds() {
        let dir = tempdir().unwrap();
        let engine_dir = llama::managed_engine_dir(dir.path());
        touch(&engine_dir.join("bin").join("llama-server"));
        std::fs::write(engine_dir.join("VERSION"), "b100\n").unwrap();
        touch(&engine_dir.join("cuda").join("bin").join("llama-server"));

        let build_root = builds::builds_root(dir.path(), ENGINE).join("main-1");
        let build_binary = build_root.join("install").join("bin").join("llama-server");
        touch(&build_binary);
        std::fs::write(
            build_root.join("build.json"),
            serde_json::to_vec(&builds::BuildRecord {
                engine: ENGINE.into(),
                repository: "https://github.com/ggml-org/llama.cpp".into(),
                revision: "main".into(),
                target: "cpu".into(),
                created_at: "1".into(),
                binary: build_binary.display().to_string(),
                name: None,
                commit: None,
            })
            .unwrap(),
        )
        .unwrap();

        let entries = list(
            dir.path(),
            &ActiveRuntimes {
                llama: Some(build_binary),
                ..ActiveRuntimes::default()
            },
            None,
            false,
        );
        let ids: Vec<&str> = entries.iter().map(|entry| entry.id.as_str()).collect();
        assert!(ids.contains(&"managed"));
        assert!(ids.contains(&"managed-cuda"));
        assert!(ids.contains(&"source-main-1"));
        let managed = entries.iter().find(|entry| entry.id == "managed").unwrap();
        assert_eq!(managed.version.as_deref(), Some("b100"));
        let source = entries
            .iter()
            .find(|entry| entry.id == "source-main-1")
            .unwrap();
        assert!(source.active);
        assert!(!managed.active);
    }

    #[test]
    fn delete_removes_source_build_directory() {
        let dir = tempdir().unwrap();
        let build_root = builds::builds_root(dir.path(), ENGINE).join("main-1");
        let build_binary = build_root.join("install").join("bin").join("llama-server");
        touch(&build_binary);
        std::fs::write(build_root.join("build.json"), b"{}").unwrap();

        let removed = delete(dir.path(), "source-main-1").unwrap();
        assert_eq!(removed, build_binary);
        assert!(!build_root.exists());
        assert!(delete(dir.path(), "source-../evil").is_err());
        assert!(delete(dir.path(), "system-/usr/bin/llama-server").is_err());
    }

    #[test]
    fn detects_pinned_commit_revisions() {
        assert!(looks_like_commit_sha("a1b2c3d"));
        assert!(looks_like_commit_sha(
            "0123456789abcdef0123456789abcdef01234567"
        ));
        assert!(!looks_like_commit_sha("main"));
        assert!(!looks_like_commit_sha("v1.2.3"));
    }

    #[test]
    fn check_updates_reports_pinned_builds_without_network() {
        let dir = tempdir().unwrap();
        let sha = "0123456789abcdef0123456789abcdef01234567";
        let build_root = builds::builds_root(dir.path(), ENGINE).join("pinned-1");
        let build_binary = build_root.join("install").join("bin").join("llama-server");
        touch(&build_binary);
        std::fs::write(
            build_root.join("build.json"),
            serde_json::to_vec(&builds::BuildRecord {
                engine: ENGINE.into(),
                repository: "https://github.com/ggml-org/llama.cpp".into(),
                revision: sha.into(),
                target: "cpu".into(),
                created_at: "1".into(),
                binary: build_binary.display().to_string(),
                name: None,
                commit: Some(sha.into()),
            })
            .unwrap(),
        )
        .unwrap();

        let updates = check_source_updates(dir.path());
        let entry = updates
            .iter()
            .find(|update| update.id == "source-pinned-1")
            .expect("pinned build reported");
        assert!(entry.pinned);
        assert!(!entry.update_available);
        assert!(entry.error.is_none());
        assert_eq!(entry.current_commit.as_deref(), Some("0123456789ab"));
    }

    /// Two virtualenvs whose `bin/python` symlinks resolve to the same base
    /// interpreter — which is what uv produces for every build — are two
    /// runtimes. Resolving the symlink reported all of them as active at once.
    #[test]
    fn separate_virtualenvs_are_separate_runtimes() {
        let dir = tempdir().unwrap();
        let base = dir.path().join("shared/bin");
        std::fs::create_dir_all(&base).unwrap();
        let interpreter = base.join("python3.12");
        touch(&interpreter);

        let mut venvs = Vec::new();
        for name in ["one", "two"] {
            let venv = dir.path().join(name).join("venv");
            std::fs::create_dir_all(venv.join("bin")).unwrap();
            std::fs::write(venv.join("pyvenv.cfg"), "home = /shared\n").unwrap();
            let python = venv.join("bin").join("python");
            std::os::unix::fs::symlink(&interpreter, &python).unwrap();
            venvs.push(python);
        }

        assert!(same_file(&venvs[0], &venvs[0]));
        assert!(
            !same_file(&venvs[0], &venvs[1]),
            "distinct venvs, one active"
        );

        // Outside a virtualenv, resolving is still what pairs a discovered
        // binary with the managed install it points at.
        let link = dir.path().join("llama-server");
        std::os::unix::fs::symlink(&interpreter, &link).unwrap();
        assert!(same_file(&link, &interpreter));
        assert!(!same_file(&link, &dir.path().join("missing")));
    }

    #[test]
    fn renders_unix_seconds_as_a_calendar_date() {
        assert_eq!(civil_date(0), "1970-01-01");
        assert_eq!(civil_date(86_399), "1970-01-01");
        assert_eq!(civil_date(86_400), "1970-01-02");
        // Leap day, and the day after it.
        assert_eq!(civil_date(1_709_164_800), "2024-02-29");
        assert_eq!(civil_date(1_709_251_200), "2024-03-01");
        // Century that is not a leap year.
        assert_eq!(civil_date(4_107_542_400), "2100-03-01");
        // The builds on this machine.
        assert_eq!(civil_date(1_785_015_875), "2026-07-25");
        // Before the epoch, so the flooring division is exercised both ways.
        assert_eq!(civil_date(-1), "1969-12-31");
        assert_eq!(civil_timestamp(1_785_015_875), "2026-07-25 21:44");
    }

    #[test]
    fn source_builds_of_one_revision_are_distinguishable() {
        let record = builds::BuildRecord {
            engine: "personaplex-mlx".into(),
            repository: "https://github.com/example/personaplex-mlx".into(),
            revision: "main".into(),
            target: "metal".into(),
            created_at: "1785015875".into(),
            binary: "/venv/bin/python".into(),
            name: None,
            commit: None,
        };
        assert_eq!(build_stamp(&record), "2026-07-25 21:44");
        // Two builds of one branch on one afternoon must not read the same.
        assert_ne!(
            build_stamp(&builds::BuildRecord {
                created_at: "1785013062".into(),
                ..record.clone()
            }),
            build_stamp(&record)
        );
        // Records written before the switch to Unix seconds.
        assert_eq!(
            build_stamp(&builds::BuildRecord {
                created_at: "2026-07-25T14:44:32Z".into(),
                ..record.clone()
            }),
            "2026-07-25"
        );
        assert_eq!(
            build_stamp(&builds::BuildRecord {
                created_at: String::new(),
                ..record
            }),
            "undated"
        );
    }

    #[test]
    fn source_labels_name_custom_llama_forks() {
        let record = builds::BuildRecord {
            engine: ENGINE.into(),
            repository: "https://github.com/PrismML-Eng/llama.cpp".into(),
            revision: "main".into(),
            target: "metal".into(),
            created_at: "0".into(),
            binary: "/bin/llama-server".into(),
            name: None,
            commit: None,
        };
        assert!(source_label(&record).contains("PrismML-Eng"));
    }

    #[test]
    fn find_for_fork_matches_source_repository() {
        let dir = tempdir().unwrap();
        let build_root = builds::builds_root(dir.path(), ENGINE).join("fork-1");
        let build_binary = build_root.join("install").join("bin").join("llama-server");
        touch(&build_binary);
        std::fs::write(
            build_root.join("build.json"),
            serde_json::to_vec(&builds::BuildRecord {
                engine: ENGINE.into(),
                repository: "https://github.com/example/llama.cpp".into(),
                revision: "main".into(),
                target: "cpu".into(),
                created_at: "1".into(),
                binary: build_binary.display().to_string(),
                name: None,
                commit: None,
            })
            .unwrap(),
        )
        .unwrap();
        let hint = crate::fork_hints::RuntimeForkHint {
            engine: ENGINE.into(),
            display_name: "llama.cpp".into(),
            repository: "https://github.com/example/llama.cpp".into(),
            trusted: false,
            summary: "test".into(),
        };
        let entry = find_for_fork(dir.path(), &ActiveRuntimes::default(), &hint).unwrap();
        assert_eq!(entry.id, "source-fork-1");
    }
}

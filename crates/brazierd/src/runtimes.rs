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
const PYTHON_ENGINES: &[&str] = &["mlx-lm", "mlx-vlm"];

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

#[derive(Debug, Clone, Default)]
pub struct ActiveRuntimes {
    pub llama: Option<PathBuf>,
    pub mlx_lm: Option<PathBuf>,
    pub mlx_vlm: Option<PathBuf>,
    pub whisper: Option<PathBuf>,
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

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => a == b,
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
            label: llama_managed_label("cpu"),
            target: Some("cpu".to_owned()),
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
            label: format!("llama.cpp · Source · {}", record.revision),
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
            let display = if *engine == "mlx-lm" {
                "MLX-LM"
            } else {
                "MLX-VLM"
            };
            entries.push(RuntimeEntry {
                id: format!("{engine}-source-{build_id}"),
                engine: (*engine).to_owned(),
                kind: "source".to_owned(),
                label: format!("{display} · {}", record.revision),
                target: Some(record.target.clone()),
                version: Some(record.revision.clone()),
                repository: Some(record.repository.clone()),
                path: record.binary.clone(),
                active: is_active(
                    &path,
                    if *engine == "mlx-lm" {
                        &active.mlx_lm
                    } else {
                        &active.mlx_vlm
                    },
                ),
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
    entries
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

//! Inventory and lifecycle of installed llama-server runtimes.
//!
//! Three kinds are tracked: `managed` (prebuilt releases installed by the
//! daemon, one per acceleration flavor), `source` (user-approved builds from
//! source), and `system` (binaries discovered on PATH, never managed here).

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Serialize;

use crate::{builds, llama};

pub const ENGINE: &str = "llama.cpp";
const MANAGED_FLAVORS: &[&str] = &["cuda", "rocm", "vulkan"];

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeEntry {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub target: Option<String>,
    pub version: Option<String>,
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

/// Enumerate every known llama-server runtime on this machine.
pub fn list(
    data_dir: &Path,
    active: Option<&Path>,
    path_env: Option<&str>,
    include_system: bool,
) -> Vec<RuntimeEntry> {
    let mut entries = Vec::new();
    let engine_dir = llama::managed_engine_dir(data_dir);
    let is_active = |path: &Path| active.is_some_and(|active_path| same_file(path, active_path));

    // Managed default (CPU / Metal) install.
    let default_binary = llama::managed_binary_path(data_dir);
    if default_binary.is_file() {
        entries.push(RuntimeEntry {
            id: "managed".to_owned(),
            kind: "managed".to_owned(),
            label: "Managed release".to_owned(),
            target: Some("cpu".to_owned()),
            version: read_version(&engine_dir),
            path: default_binary.display().to_string(),
            active: is_active(&default_binary),
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
                kind: "managed".to_owned(),
                label: format!("Managed release ({flavor})"),
                target: Some((*flavor).to_owned()),
                version: read_version(&flavor_dir),
                path: binary.display().to_string(),
                active: is_active(&binary),
                deletable: true,
            });
        }
    }

    // Source builds.
    for (build_id, record) in builds::list_builds(data_dir, ENGINE) {
        let path = PathBuf::from(&record.binary);
        entries.push(RuntimeEntry {
            id: format!("source-{build_id}"),
            kind: "source".to_owned(),
            label: format!("Source build · {}", record.revision),
            target: Some(record.target.clone()),
            version: Some(record.revision.clone()),
            path: record.binary.clone(),
            active: is_active(&path),
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
            kind: "system".to_owned(),
            label: "System binary".to_owned(),
            target: None,
            version: None,
            path: candidate.display().to_string(),
            active: is_active(&candidate),
            deletable: false,
        });
    }
    entries
}

/// Resolve a runtime id from `list` back to its binary path.
pub fn find(
    data_dir: &Path,
    path_env: Option<&str>,
    id: &str,
    include_system: bool,
) -> Option<RuntimeEntry> {
    list(data_dir, None, path_env, include_system)
        .into_iter()
        .find(|entry| entry.id == id)
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

        let entries = list(dir.path(), Some(&build_binary), None, false);
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
}

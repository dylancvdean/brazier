//! Git worktree confinement for agent sessions.
//!
//! When enabled, the session's workspace is a detached worktree beside the
//! source repository so the agent can edit and run commands without dirtying
//! the user's current checkout. The worktree lives outside Brazier's data
//! directory (which cannot be an agent workspace) as a sibling folder:
//! `{repo_parent}/.brazier-worktrees/{repo_name}/{session_id}`.

use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use tokio::process::Command;

/// Metadata stored on the session under `runtime_metadata.worktree`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorktreeInfo {
    /// Original repository the user selected.
    pub source_path: String,
    /// Absolute path of the worktree checkout (the session workspace).
    pub path: String,
    /// Branch created for this session.
    pub branch: String,
}

/// True when `path` is inside a git working tree (primary checkout or worktree).
pub async fn is_git_repository(path: &Path) -> bool {
    let output = Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .current_dir(path)
        .output()
        .await;
    match output {
        Ok(result) if result.status.success() => {
            String::from_utf8_lossy(&result.stdout).trim() == "true"
        }
        _ => path.join(".git").exists(),
    }
}

/// Directory that will hold the worktree for this session.
pub fn worktree_path(source: &Path, session_id: &str) -> anyhow::Result<PathBuf> {
    validate_path_component(session_id, "session id")?;
    let parent = source
        .parent()
        .with_context(|| format!("{} has no parent directory", source.display()))?;
    let name = source
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("repo");
    Ok(parent
        .join(".brazier-worktrees")
        .join(name)
        .join(session_id))
}

fn validate_path_component(value: &str, label: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value.is_empty()
            && value.len() <= 128
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
        "{label} contains unsafe characters"
    );
    Ok(())
}

fn branch_name(session_id: &str) -> String {
    let short = session_id.chars().take(8).collect::<String>();
    format!("brazier/agent-{short}")
}

async fn git_output(cwd: &Path, args: &[&str]) -> anyhow::Result<std::process::Output> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .await
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    Ok(output)
}

fn git_stderr(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        format!("git exited with status {}", output.status)
    }
}

/// Create a new branch worktree for `session_id` rooted at `source`.
pub async fn create_worktree(source: &Path, session_id: &str) -> anyhow::Result<WorktreeInfo> {
    if !is_git_repository(source).await {
        bail!("{} is not a git repository", source.display());
    }
    let path = worktree_path(source, session_id)?;
    if path.exists() {
        bail!("worktree path {} already exists", path.display());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
    }
    let branch = branch_name(session_id);
    let output = git_output(
        source,
        &[
            "worktree",
            "add",
            "-b",
            &branch,
            &path.display().to_string(),
            "HEAD",
        ],
    )
    .await?;
    if !output.status.success() {
        bail!("could not create worktree: {}", git_stderr(&output));
    }
    let resolved = std::fs::canonicalize(&path)
        .with_context(|| format!("worktree at {} was not created", path.display()))?;
    Ok(WorktreeInfo {
        source_path: std::fs::canonicalize(source)
            .unwrap_or_else(|_| source.to_path_buf())
            .display()
            .to_string(),
        path: resolved.display().to_string(),
        branch,
    })
}

/// Remove a session worktree and its branch. Best-effort: missing trees are fine.
pub async fn remove_worktree(info: &WorktreeInfo) -> anyhow::Result<()> {
    let source = PathBuf::from(&info.source_path);
    let path = PathBuf::from(&info.path);
    validate_worktree_info(&source, &path, &info.branch)?;
    if source.is_dir() {
        let output = git_output(&source, &["worktree", "remove", "--force", &info.path]).await?;
        if !output.status.success() && path.exists() {
            // Fall back to deleting the directory and pruning the registry.
            let _ = std::fs::remove_dir_all(&path);
            let _ = git_output(&source, &["worktree", "prune"]).await?;
        }
        let _ = git_output(&source, &["branch", "-D", &info.branch]).await?;
    } else if path.exists() {
        std::fs::remove_dir_all(&path)
            .with_context(|| format!("could not remove {}", path.display()))?;
    }
    Ok(())
}

fn validate_worktree_info(source: &Path, path: &Path, branch: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        source.is_absolute() && path.is_absolute(),
        "worktree metadata paths must be absolute"
    );
    let parent = source
        .parent()
        .with_context(|| format!("{} has no parent directory", source.display()))?;
    let repository = source
        .file_name()
        .and_then(|value| value.to_str())
        .context("repository path has no UTF-8 name")?;
    let session_id = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("worktree path has no UTF-8 session id")?;
    validate_path_component(session_id, "worktree session id")?;
    let expected_parent = parent.join(".brazier-worktrees").join(repository);
    anyhow::ensure!(
        path.parent() == Some(expected_parent.as_path()),
        "worktree path is outside the managed worktree directory"
    );
    let branch_suffix = branch
        .strip_prefix("brazier/agent-")
        .context("worktree branch is not managed by Brazier")?;
    validate_path_component(branch_suffix, "worktree branch suffix")?;
    Ok(())
}

/// Read worktree metadata from a session's `runtime_metadata` object.
pub fn worktree_from_metadata(metadata: Option<&serde_json::Value>) -> Option<WorktreeInfo> {
    let value = metadata?.get("worktree")?;
    serde_json::from_value(value.clone()).ok()
}

/// Merge worktree info into (or clear it from) runtime metadata.
pub fn metadata_with_worktree(
    existing: Option<serde_json::Value>,
    worktree: Option<WorktreeInfo>,
) -> serde_json::Value {
    let mut root = match existing {
        Some(serde_json::Value::Object(map)) => serde_json::Value::Object(map),
        _ => serde_json::json!({}),
    };
    if let Some(object) = root.as_object_mut() {
        match worktree {
            Some(info) => {
                object.insert(
                    "worktree".to_owned(),
                    serde_json::to_value(info).unwrap_or(serde_json::Value::Null),
                );
            }
            None => {
                object.remove("worktree");
            }
        }
    }
    root
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn creates_and_removes_a_worktree() {
        let root = tempdir().unwrap();
        let repo = root.path().join("project");
        std::fs::create_dir_all(&repo).unwrap();
        for args in [
            ["init"][..].as_ref(),
            ["config", "user.email", "test@example.com"].as_ref(),
            ["config", "user.name", "Test"].as_ref(),
        ] {
            let status = Command::new("git")
                .args(args)
                .current_dir(&repo)
                .status()
                .await
                .unwrap();
            assert!(status.success());
        }
        std::fs::write(repo.join("README.md"), "hi\n").unwrap();
        for args in [["add", "."].as_ref(), ["commit", "-m", "init"].as_ref()] {
            let status = Command::new("git")
                .args(args)
                .current_dir(&repo)
                .status()
                .await
                .unwrap();
            assert!(status.success(), "git {args:?}");
        }

        let info = create_worktree(&repo, "session-abcdef01")
            .await
            .expect("create");
        assert!(PathBuf::from(&info.path).join("README.md").is_file());
        assert!(info.branch.starts_with("brazier/agent-"));
        let expected = std::fs::canonicalize(
            root.path()
                .join(".brazier-worktrees")
                .join("project")
                .join("session-abcdef01"),
        )
        .unwrap();
        assert_eq!(PathBuf::from(&info.path), expected);

        remove_worktree(&info).await.expect("remove");
        assert!(!PathBuf::from(&info.path).exists());
    }

    #[test]
    fn metadata_round_trips() {
        let info = WorktreeInfo {
            source_path: "/src".into(),
            path: "/wt".into(),
            branch: "brazier/agent-abc".into(),
        };
        let meta = metadata_with_worktree(None, Some(info.clone()));
        assert_eq!(worktree_from_metadata(Some(&meta)), Some(info));
        let cleared = metadata_with_worktree(Some(meta), None);
        assert_eq!(worktree_from_metadata(Some(&cleared)), None);
    }

    #[test]
    fn rejects_traversal_in_session_ids_and_persisted_metadata() {
        let source = Path::new("/work/project");
        assert!(worktree_path(source, "../../escape").is_err());

        let info = WorktreeInfo {
            source_path: source.display().to_string(),
            path: "/work/other/session".to_owned(),
            branch: "brazier/agent-session".to_owned(),
        };
        assert!(
            validate_worktree_info(
                Path::new(&info.source_path),
                Path::new(&info.path),
                &info.branch
            )
            .is_err()
        );
    }
}

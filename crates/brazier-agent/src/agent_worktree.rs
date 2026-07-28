//! Git worktree confinement for agent sessions.
//!
//! When enabled, the session's workspace is a task-branch worktree beside the
//! source repository so the agent can edit and run commands without dirtying
//! the user's current checkout. The worktree lives outside Brazier's data
//! directory (which cannot be an agent workspace) as a sibling folder:
//! `{repo_parent}/.brazier-worktrees/{repo_name}/{session_id}`.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use tokio::{io::AsyncWriteExt, process::Command};
use uuid::Uuid;

/// Metadata stored on the session under `runtime_metadata.worktree`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorktreeInfo {
    /// Original repository the user selected.
    pub source_path: String,
    /// Absolute path of the worktree checkout (the session workspace).
    pub path: String,
    /// Branch created for this session.
    pub branch: String,
    /// Git tree last copied into the source checkout, for repeatable incremental applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_applied_tree: Option<String>,
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

fn branch_name(session_id: &str, title: &str) -> String {
    let short = session_id.chars().take(8).collect::<String>();
    let slug = title
        .chars()
        .flat_map(char::to_lowercase)
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .take(6)
        .collect::<Vec<_>>()
        .join("-");
    let slug = if slug.is_empty() {
        "task".to_owned()
    } else {
        slug.chars().take(40).collect()
    };
    format!("brazier/{slug}-{short}")
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

async fn git_output_with_env(
    cwd: &Path,
    args: &[&str],
    key: &str,
    value: &Path,
) -> anyhow::Result<std::process::Output> {
    Command::new("git")
        .args(args)
        .env(key, value)
        .current_dir(cwd)
        .output()
        .await
        .with_context(|| format!("failed to run git {}", args.join(" ")))
}

async fn git_with_input(
    cwd: &Path,
    args: &[&str],
    input: &[u8],
) -> anyhow::Result<std::process::Output> {
    let mut child = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(input).await.context("write git patch")?;
    }
    child
        .wait_with_output()
        .await
        .with_context(|| format!("wait for git {}", args.join(" ")))
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

/// Create a new, task-named branch worktree for `session_id` rooted at `source`.
pub async fn create_worktree(
    source: &Path,
    session_id: &str,
    title: &str,
) -> anyhow::Result<WorktreeInfo> {
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
    let branch = branch_name(session_id, title);
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
    let base_tree = git_output(source, &["rev-parse", "HEAD^{tree}"]).await?;
    if !base_tree.status.success() {
        bail!("could not identify source tree: {}", git_stderr(&base_tree));
    }
    Ok(WorktreeInfo {
        source_path: std::fs::canonicalize(source)
            .unwrap_or_else(|_| source.to_path_buf())
            .display()
            .to_string(),
        path: resolved.display().to_string(),
        branch,
        last_applied_tree: Some(String::from_utf8_lossy(&base_tree.stdout).trim().to_owned()),
    })
}

/// Materialize the worktree's current tracked and untracked files as a Git tree.
///
/// A temporary index keeps the user's real index untouched. Ignored files are
/// intentionally excluded: build products and dependency trees are not source.
async fn snapshot_tree(path: &Path) -> anyhow::Result<String> {
    let index_path =
        std::env::temp_dir().join(format!("brazier-worktree-index-{}", Uuid::new_v4()));
    let result = async {
        let read = git_output_with_env(path, &["read-tree", "HEAD"], "GIT_INDEX_FILE", &index_path)
            .await?;
        if !read.status.success() {
            bail!("could not prepare worktree snapshot: {}", git_stderr(&read));
        }
        let add = git_output_with_env(path, &["add", "-A"], "GIT_INDEX_FILE", &index_path).await?;
        if !add.status.success() {
            bail!("could not snapshot worktree changes: {}", git_stderr(&add));
        }
        let tree =
            git_output_with_env(path, &["write-tree"], "GIT_INDEX_FILE", &index_path).await?;
        if !tree.status.success() {
            bail!("could not write worktree snapshot: {}", git_stderr(&tree));
        }
        Ok(String::from_utf8_lossy(&tree.stdout).trim().to_owned())
    }
    .await;
    let _ = std::fs::remove_file(index_path);
    result
}

#[derive(Debug, Clone, Serialize)]
pub struct ApplyWorktreeResult {
    pub worktree: WorktreeInfo,
    pub changed_paths: Vec<String>,
    pub already_up_to_date: bool,
}

/// Apply the worktree delta to its source checkout without committing or staging it.
///
/// Repeated calls are incremental: `last_applied_tree` is advanced only after
/// the complete patch passes preflight and applies successfully.
pub async fn apply_to_source(info: &WorktreeInfo) -> anyhow::Result<ApplyWorktreeResult> {
    let source = PathBuf::from(&info.source_path);
    let path = PathBuf::from(&info.path);
    validate_worktree_info(&source, &path, &info.branch)?;
    anyhow::ensure!(
        source.is_dir(),
        "source checkout {} is unavailable",
        source.display()
    );
    anyhow::ensure!(path.is_dir(), "worktree {} is unavailable", path.display());
    let source_status = git_output(
        &source,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )
    .await?;
    if !source_status.status.success() {
        bail!(
            "could not inspect source checkout: {}",
            git_stderr(&source_status)
        );
    }
    anyhow::ensure!(
        source_status.stdout.is_empty(),
        "source checkout must be clean before applying worktree changes"
    );

    let old_tree = match &info.last_applied_tree {
        Some(tree) => tree.clone(),
        None => {
            let merge_base = git_output(&source, &["merge-base", "HEAD", &info.branch]).await?;
            if !merge_base.status.success() {
                bail!(
                    "could not find the worktree base: {}",
                    git_stderr(&merge_base)
                );
            }
            let commit = String::from_utf8_lossy(&merge_base.stdout)
                .trim()
                .to_owned();
            let tree = git_output(&source, &["rev-parse", &format!("{commit}^{{tree}}")]).await?;
            if !tree.status.success() {
                bail!(
                    "could not identify the worktree base tree: {}",
                    git_stderr(&tree)
                );
            }
            String::from_utf8_lossy(&tree.stdout).trim().to_owned()
        }
    };
    let new_tree = snapshot_tree(&path).await?;
    let names = git_output(&path, &["diff", "--name-only", "-z", &old_tree, &new_tree]).await?;
    if !names.status.success() {
        bail!("could not list worktree changes: {}", git_stderr(&names));
    }
    let changed_paths = names
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| String::from_utf8_lossy(path).into_owned())
        .collect::<Vec<_>>();
    if old_tree == new_tree {
        return Ok(ApplyWorktreeResult {
            worktree: info.clone(),
            changed_paths,
            already_up_to_date: true,
        });
    }

    let patch = git_output(&path, &["diff", "--binary", &old_tree, &new_tree, "--"]).await?;
    if !patch.status.success() {
        bail!("could not prepare worktree patch: {}", git_stderr(&patch));
    }
    let check = git_with_input(
        &source,
        &["apply", "--check", "--binary", "-"],
        &patch.stdout,
    )
    .await?;
    if !check.status.success() {
        bail!(
            "source checkout has conflicting changes; no files were applied: {}",
            git_stderr(&check)
        );
    }
    let applied = git_with_input(&source, &["apply", "--binary", "-"], &patch.stdout).await?;
    if !applied.status.success() {
        bail!("could not apply worktree changes: {}", git_stderr(&applied));
    }

    let mut updated = info.clone();
    updated.last_applied_tree = Some(new_tree);
    Ok(ApplyWorktreeResult {
        worktree: updated,
        changed_paths,
        already_up_to_date: false,
    })
}

/// True when the managed worktree contains staged, unstaged, or untracked work.
pub async fn worktree_is_dirty(info: &WorktreeInfo) -> anyhow::Result<bool> {
    let source = PathBuf::from(&info.source_path);
    let path = PathBuf::from(&info.path);
    validate_worktree_info(&source, &path, &info.branch)?;
    if !path.exists() {
        return Ok(false);
    }
    let output = git_output(
        &path,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )
    .await?;
    if !output.status.success() {
        bail!("could not inspect worktree: {}", git_stderr(&output));
    }
    Ok(!output.stdout.is_empty())
}

/// Remove a clean session worktree while preserving its branch.
///
/// Committed work remains recoverable on the task branch. Uncommitted work is
/// never discarded implicitly; callers must ask the user to resolve it first.
pub async fn remove_worktree(info: &WorktreeInfo) -> anyhow::Result<()> {
    let source = PathBuf::from(&info.source_path);
    let path = PathBuf::from(&info.path);
    validate_worktree_info(&source, &path, &info.branch)?;
    let dirty = worktree_is_dirty(info).await?;
    let fully_applied = if dirty {
        match &info.last_applied_tree {
            Some(applied) => snapshot_tree(&path).await? == *applied,
            None => false,
        }
    } else {
        true
    };
    if !fully_applied {
        bail!(
            "worktree {} has unapplied changes; apply, commit, or discard them before cleanup",
            path.display()
        );
    }
    if source.is_dir() {
        let output = if dirty {
            git_output(&source, &["worktree", "remove", "--force", &info.path]).await?
        } else {
            git_output(&source, &["worktree", "remove", &info.path]).await?
        };
        if !output.status.success() && path.exists() {
            bail!("could not remove worktree: {}", git_stderr(&output));
        }
    } else if path.exists() {
        bail!(
            "source repository {} is unavailable; refusing to remove {}",
            source.display(),
            path.display()
        );
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
        .strip_prefix("brazier/")
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

        let info = create_worktree(&repo, "abcdef01-task", "Fix the frobnicator")
            .await
            .expect("create");
        assert!(PathBuf::from(&info.path).join("README.md").is_file());
        assert_eq!(info.branch, "brazier/fix-the-frobnicator-abcdef01");
        let expected = std::fs::canonicalize(
            root.path()
                .join(".brazier-worktrees")
                .join("project")
                .join("abcdef01-task"),
        )
        .unwrap();
        assert_eq!(PathBuf::from(&info.path), expected);

        remove_worktree(&info).await.expect("remove");
        assert!(!PathBuf::from(&info.path).exists());
        let branch = git_output(&repo, &["branch", "--list", &info.branch])
            .await
            .unwrap();
        assert!(!branch.stdout.is_empty(), "task branch should be preserved");
    }

    #[tokio::test]
    async fn refuses_to_remove_a_dirty_worktree() {
        let root = tempdir().unwrap();
        let repo = root.path().join("project");
        std::fs::create_dir_all(&repo).unwrap();
        for args in [
            ["init"][..].as_ref(),
            ["config", "user.email", "test@example.com"].as_ref(),
            ["config", "user.name", "Test"].as_ref(),
        ] {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(&repo)
                    .status()
                    .await
                    .unwrap()
                    .success()
            );
        }
        std::fs::write(repo.join("README.md"), "hi\n").unwrap();
        for args in [["add", "."].as_ref(), ["commit", "-m", "init"].as_ref()] {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(&repo)
                    .status()
                    .await
                    .unwrap()
                    .success()
            );
        }
        let info = create_worktree(&repo, "session-dirty123", "Dirty task")
            .await
            .unwrap();
        std::fs::write(PathBuf::from(&info.path).join("README.md"), "changed\n").unwrap();

        assert!(worktree_is_dirty(&info).await.unwrap());
        assert!(remove_worktree(&info).await.is_err());
        assert!(PathBuf::from(&info.path).exists());
    }

    #[tokio::test]
    async fn applies_committed_uncommitted_and_untracked_changes_incrementally() {
        let root = tempdir().unwrap();
        let repo = root.path().join("project");
        std::fs::create_dir_all(&repo).unwrap();
        for args in [
            ["init"][..].as_ref(),
            ["config", "user.email", "test@example.com"].as_ref(),
            ["config", "user.name", "Test"].as_ref(),
        ] {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(&repo)
                    .status()
                    .await
                    .unwrap()
                    .success()
            );
        }
        std::fs::write(repo.join("tracked.txt"), "base\n").unwrap();
        for args in [["add", "."].as_ref(), ["commit", "-m", "base"].as_ref()] {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(&repo)
                    .status()
                    .await
                    .unwrap()
                    .success()
            );
        }

        let info = create_worktree(&repo, "apply123-task", "Apply changes")
            .await
            .unwrap();
        let worktree = PathBuf::from(&info.path);
        std::fs::write(worktree.join("tracked.txt"), "committed\n").unwrap();
        for args in [
            ["add", "tracked.txt"][..].as_ref(),
            ["commit", "-m", "agent commit"].as_ref(),
        ] {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(&worktree)
                    .status()
                    .await
                    .unwrap()
                    .success()
            );
        }
        std::fs::write(worktree.join("new.txt"), "untracked\n").unwrap();

        std::fs::write(repo.join("local-only.txt"), "do not overwrite\n").unwrap();
        let dirty_error = apply_to_source(&info).await.unwrap_err();
        assert!(
            dirty_error
                .to_string()
                .contains("source checkout must be clean")
        );
        assert_eq!(
            std::fs::read_to_string(repo.join("tracked.txt")).unwrap(),
            "base\n"
        );
        std::fs::remove_file(repo.join("local-only.txt")).unwrap();

        let first = apply_to_source(&info).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(repo.join("tracked.txt")).unwrap(),
            "committed\n"
        );
        assert_eq!(
            std::fs::read_to_string(repo.join("new.txt")).unwrap(),
            "untracked\n"
        );
        assert_eq!(first.changed_paths, vec!["new.txt", "tracked.txt"]);

        // Applying intentionally leaves checkout changes for testing. Commit
        // them before pulling the next incremental snapshot across.
        for args in [
            ["add", "."].as_ref(),
            ["commit", "-m", "apply first"].as_ref(),
        ] {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(&repo)
                    .status()
                    .await
                    .unwrap()
                    .success()
            );
        }
        std::fs::write(worktree.join("tracked.txt"), "second\n").unwrap();
        let second = apply_to_source(&first.worktree).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(repo.join("tracked.txt")).unwrap(),
            "second\n"
        );

        std::fs::write(worktree.join("tracked.txt"), "third\n").unwrap();
        let error = apply_to_source(&second.worktree).await.unwrap_err();
        assert!(error.to_string().contains("source checkout must be clean"));
        assert_eq!(
            std::fs::read_to_string(repo.join("tracked.txt")).unwrap(),
            "second\n"
        );

        // Returning the worktree to its last-applied snapshot makes cleanup
        // safe even though that snapshot contains uncommitted files.
        std::fs::write(worktree.join("tracked.txt"), "second\n").unwrap();
        remove_worktree(&second.worktree).await.unwrap();
        assert!(!worktree.exists());
    }

    #[test]
    fn metadata_round_trips() {
        let info = WorktreeInfo {
            source_path: "/src".into(),
            path: "/wt".into(),
            branch: "brazier/agent-abc".into(),
            last_applied_tree: None,
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
            last_applied_tree: None,
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

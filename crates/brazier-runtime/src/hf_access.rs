//! Gated-model access requests.
//!
//! A gated repository on Hugging Face needs its terms accepted and the account
//! granted access before files can be fetched. Acceptance happens on the Hub,
//! outside Brazier, so a request to install a gated model records the
//! repository here and is re-checked — with the saved token — until access
//! lands. The queue lives in settings, not the global download tray, because a
//! download cannot start yet; there is nothing to show there.
//!
//! Checking is deliberately bounded: at most once every five minutes, and at
//! most [`MAX_CHECKS`] times per request. That allows two hours for the grant
//! to land without pinging the Hub forever.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Maximum number of times a repository is checked for access.
pub const MAX_CHECKS: u32 = 24;

/// Minimum gap between checks of the same repository (five minutes).
pub const CHECK_INTERVAL_SECS: u64 = 5 * 60;

/// A repository Brazier is waiting to gain access to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AccessRequest {
    pub repo_id: String,
    /// Unix seconds when the request was recorded.
    pub requested_at: u64,
    /// How many times access has been checked for.
    #[serde(default)]
    pub checks: u32,
    /// Unix seconds of the last check; 0 before the first.
    #[serde(default)]
    pub last_checked_at: u64,
    /// Whether the last check found access granted.
    #[serde(default)]
    pub granted: bool,
}

pub fn state_path(data_dir: &Path) -> PathBuf {
    data_dir.join("hf-access-requests.json")
}

pub fn load(data_dir: &Path) -> Vec<AccessRequest> {
    let path = state_path(data_dir);
    let Ok(bytes) = std::fs::read(&path) else {
        return Vec::new();
    };
    serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        tracing::warn!(%error, path = %path.display(), "ignoring invalid HF access requests");
        Vec::new()
    })
}

pub async fn save(data_dir: &Path, requests: &[AccessRequest]) -> anyhow::Result<()> {
    let path = state_path(data_dir);
    crate::persistence::write_json(&path, requests, "HF access requests").await
}

pub fn now_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// Record a repository access request, re-arming its check window if it is
/// already present (a fresh request deserves a fresh two hours).
pub async fn add(data_dir: &Path, repo_id: &str) -> anyhow::Result<Vec<AccessRequest>> {
    let mut requests = load(data_dir);
    let now = now_seconds();
    match requests.iter_mut().find(|request| request.repo_id == repo_id) {
        Some(request) => {
            request.requested_at = now;
            request.checks = 0;
            request.last_checked_at = 0;
            request.granted = false;
        }
        None => requests.push(AccessRequest {
            repo_id: repo_id.to_owned(),
            requested_at: now,
            checks: 0,
            last_checked_at: 0,
            granted: false,
        }),
    }
    save(data_dir, &requests).await?;
    Ok(requests)
}

/// Forget a repository access request.
pub async fn remove(data_dir: &Path, repo_id: &str) -> anyhow::Result<Vec<AccessRequest>> {
    let requests = load(data_dir);
    let kept: Vec<AccessRequest> = requests
        .into_iter()
        .filter(|request| request.repo_id != repo_id)
        .collect();
    save(data_dir, &kept).await?;
    Ok(kept)
}

/// Whether a request has used up its check budget.
pub fn expired(request: &AccessRequest) -> bool {
    request.checks >= MAX_CHECKS
}

/// Whether enough time has passed since the last check to check again.
pub fn due(request: &AccessRequest, now: u64) -> bool {
    request.last_checked_at == 0
        || now.saturating_sub(request.last_checked_at) >= CHECK_INTERVAL_SECS
}

/// Whether the saved token can reach this repository right now.
///
/// The Hub's model API is itself gated: it returns 401 for a gated repository
/// the authenticated account has not been granted, and 200 once access lands.
/// A 200 therefore means granted (or the repository was never gated).
pub async fn access_granted(
    client: &reqwest::Client,
    data_dir: &Path,
    repo_id: &str,
) -> bool {
    crate::hf::model_trust(client, data_dir, repo_id)
        .await
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn add_is_idempotent_and_persists() {
        let dir = tempdir().unwrap();
        let first = add(dir.path(), "org/gated-model").await.unwrap();
        assert_eq!(first.len(), 1);
        let again = add(dir.path(), "org/gated-model").await.unwrap();
        assert_eq!(again.len(), 1);
        assert_eq!(load(dir.path()).len(), 1);
    }

    #[tokio::test]
    async fn re_adding_re_arms_the_check_window() {
        let dir = tempdir().unwrap();
        add(dir.path(), "org/gated-model").await.unwrap();
        let mut requests = load(dir.path());
        requests[0].checks = MAX_CHECKS;
        requests[0].granted = false;
        save(dir.path(), &requests).await.unwrap();

        add(dir.path(), "org/gated-model").await.unwrap();
        let reloaded = load(dir.path());
        assert_eq!(reloaded[0].checks, 0);
        assert_eq!(reloaded[0].last_checked_at, 0);
        assert!(!expired(&reloaded[0]));
    }

    #[test]
    fn check_budget_is_bounded_and_spaced() {
        let mut request = AccessRequest {
            repo_id: "org/gated-model".into(),
            requested_at: 1_000,
            checks: MAX_CHECKS - 1,
            last_checked_at: 100,
            granted: false,
        };
        assert!(!expired(&request));
        // Last check a second ago — not due yet.
        assert!(!due(&request, 101));
        // Five minutes later — due.
        assert!(due(&request, 100 + CHECK_INTERVAL_SECS));
        request.checks = MAX_CHECKS;
        assert!(expired(&request));
    }

    #[tokio::test]
    async fn remove_forgets_only_the_named_repo() {
        let dir = tempdir().unwrap();
        add(dir.path(), "org/one").await.unwrap();
        add(dir.path(), "org/two").await.unwrap();
        let kept = remove(dir.path(), "org/one").await.unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].repo_id, "org/two");
    }
}


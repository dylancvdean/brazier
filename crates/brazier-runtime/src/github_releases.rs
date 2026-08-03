//! Cached release lookups for GitHub-hosted managed engines.
//!
//! Manage → Runtimes needs the newest release tag for llama.cpp, whisper.cpp,
//! and stable-diffusion.cpp every time it opens, and each upstream call costs
//! a few hundred milliseconds. Releases land every few days at most, so they
//! are cached on disk (surviving daemon restarts) and served stale while a
//! refresh runs in the background — status views never wait on the network.
//! Installs and updates always re-fetch: a release can appear on `/latest`
//! before every asset has finished uploading, so a within-TTL cache may be
//! missing the binary the user just asked to install.

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use serde::{Deserialize, Serialize};

/// How long a cached release is considered current for status views.
/// Install/update paths always re-fetch regardless of this TTL.
const CACHE_TTL: Duration = Duration::from_secs(2 * 24 * 60 * 60);
const CACHE_FILE: &str = "github-releases.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseAsset {
    pub name: String,
    pub browser_download_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Release {
    pub tag_name: String,
    pub assets: Vec<ReleaseAsset>,
}

impl Release {
    /// Asset matching an exact asset name.
    pub fn asset(&self, name: &str) -> Option<&ReleaseAsset> {
        self.assets.iter().find(|asset| asset.name == name)
    }

    pub fn asset_names(&self) -> impl Iterator<Item = &str> {
        self.assets.iter().map(|asset| asset.name.as_str())
    }
}

/// A cache read: the release we can show now, plus whether a refresh is running.
pub struct CachedRelease {
    pub release: Option<Release>,
    /// True while a background lookup is in flight, so callers can report the
    /// difference between "no update" and "not checked yet".
    pub refreshing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entry {
    /// Unix seconds; wall-clock so the age survives a restart.
    fetched_at: u64,
    release: Release,
}

impl Entry {
    fn is_stale(&self) -> bool {
        now_unix().saturating_sub(self.fetched_at) > CACHE_TTL.as_secs()
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or_default()
}

static CACHE_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Point the cache at a directory under the data dir. Without this the cache
/// still works, but only for the lifetime of the process.
pub fn set_cache_dir(dir: PathBuf) {
    let _ = CACHE_DIR.set(dir);
}

fn cache_path() -> Option<PathBuf> {
    Some(CACHE_DIR.get()?.join(CACHE_FILE))
}

fn memory() -> &'static Mutex<HashMap<String, Entry>> {
    static MEMORY: OnceLock<Mutex<HashMap<String, Entry>>> = OnceLock::new();
    MEMORY.get_or_init(|| Mutex::new(load_from_disk()))
}

fn load_from_disk() -> HashMap<String, Entry> {
    let Some(path) = cache_path() else {
        return HashMap::new();
    };
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn save_to_disk() {
    let Some(path) = cache_path() else { return };
    let Ok(entries) = memory().lock() else { return };
    let Ok(payload) = serde_json::to_vec_pretty(&*entries) else {
        return;
    };
    drop(entries);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, payload);
}

fn entry(url: &str) -> Option<Entry> {
    memory().lock().ok()?.get(url).cloned()
}

fn store(url: &str, release: Release) {
    if let Ok(mut entries) = memory().lock() {
        entries.insert(
            url.to_owned(),
            Entry {
                fetched_at: now_unix(),
                release,
            },
        );
    }
    save_to_disk();
}

fn inflight() -> &'static Mutex<HashSet<String>> {
    static INFLIGHT: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    INFLIGHT.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Claim the right to refresh `url`, or `false` if another task already has it.
fn claim_refresh(url: &str) -> bool {
    inflight()
        .lock()
        .map(|mut urls| urls.insert(url.to_owned()))
        .unwrap_or(false)
}

fn release_refresh(url: &str) {
    if let Ok(mut urls) = inflight().lock() {
        urls.remove(url);
    }
}

fn is_refreshing(url: &str) -> bool {
    inflight()
        .lock()
        .map(|urls| urls.contains(url))
        .unwrap_or(false)
}

async fn fetch(client: &reqwest::Client, url: &str, user_agent: &str) -> anyhow::Result<Release> {
    let release: Release = client
        .get(url)
        .header("user-agent", user_agent)
        .send()
        .await
        .context("contact GitHub releases")?
        .error_for_status()
        .context("GitHub releases request failed")?
        .json()
        .await
        .context("decode GitHub release")?;
    store(url, release.clone());
    Ok(release)
}

/// Latest release for a repository, always contacting GitHub.
///
/// Use this on install/update paths that need the current asset list. Status
/// views should keep using [`cached_or_refresh`] so opening Manage stays
/// instant. Falls back to a cached copy only when the network request fails.
pub async fn latest_release(
    client: &reqwest::Client,
    url: &str,
    user_agent: &str,
) -> anyhow::Result<Release> {
    match fetch(client, url, user_agent).await {
        Ok(release) => Ok(release),
        Err(error) => {
            if let Some(entry) = entry(url) {
                tracing::warn!(
                    %url,
                    %error,
                    "using cached GitHub release after fetch failure"
                );
                Ok(entry.release)
            } else {
                Err(error)
            }
        }
    }
}

/// Whatever is cached right now, kicking off a background refresh when the
/// entry is missing or stale — or unconditionally when `force` is set, so a
/// manual "check for updates" can notice a release published within the cache
/// window. Never contacts GitHub on the calling task.
pub fn cached_or_refresh(
    client: &reqwest::Client,
    url: &str,
    user_agent: &str,
    force: bool,
) -> CachedRelease {
    let cached = entry(url);
    let needs_refresh = force || cached.as_ref().is_none_or(Entry::is_stale);
    let mut refreshing = is_refreshing(url);
    if needs_refresh && claim_refresh(url) {
        refreshing = true;
        let client = client.clone();
        let url = url.to_owned();
        let user_agent = user_agent.to_owned();
        tokio::spawn(async move {
            if let Err(error) = fetch(&client, &url, &user_agent).await {
                tracing::debug!(%url, %error, "background release refresh failed");
            }
            release_refresh(&url);
        });
    }
    CachedRelease {
        release: cached.map(|entry| entry.release),
        refreshing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_expire_after_the_cache_window() {
        let release = Release {
            tag_name: "b1".into(),
            assets: Vec::new(),
        };
        let fresh = Entry {
            fetched_at: now_unix(),
            release: release.clone(),
        };
        let day_old = Entry {
            fetched_at: now_unix() - 24 * 60 * 60,
            release: release.clone(),
        };
        let ancient = Entry {
            fetched_at: now_unix() - 3 * 24 * 60 * 60,
            release,
        };
        assert!(!fresh.is_stale());
        assert!(!day_old.is_stale());
        assert!(ancient.is_stale());
    }

    #[test]
    fn assets_are_looked_up_by_exact_name() {
        let release = Release {
            tag_name: "b6100".into(),
            assets: vec![ReleaseAsset {
                name: "llama-b6100-bin-macos-arm64.zip".into(),
                browser_download_url: "https://example.invalid/a.zip".into(),
            }],
        };
        assert!(release.asset("llama-b6100-bin-macos-arm64.zip").is_some());
        assert!(release.asset("llama-b6100-bin-ubuntu-x64.zip").is_none());
        assert_eq!(
            release.asset_names().collect::<Vec<_>>(),
            vec!["llama-b6100-bin-macos-arm64.zip"]
        );
    }

    #[test]
    fn a_refresh_is_claimed_by_only_one_task() {
        let url = "https://example.invalid/claim-once";
        assert!(claim_refresh(url));
        assert!(!claim_refresh(url));
        assert!(is_refreshing(url));
        release_refresh(url);
        assert!(!is_refreshing(url));
    }

    #[tokio::test]
    async fn force_bypasses_the_cache_window() {
        let client = reqwest::Client::new();
        let url = "https://127.0.0.1:1/force-refresh";
        store(
            url,
            Release {
                tag_name: "b1".into(),
                assets: Vec::new(),
            },
        );

        let served = cached_or_refresh(&client, url, "brazier-test", false);
        assert!(
            !served.refreshing,
            "a fresh cache entry stays quiet without force"
        );
        assert_eq!(
            served
                .release
                .as_ref()
                .map(|release| release.tag_name.as_str()),
            Some("b1")
        );

        let forced = cached_or_refresh(&client, url, "brazier-test", true);
        assert!(
            forced.refreshing,
            "force triggers a refresh even within the cache window"
        );
        assert_eq!(
            forced
                .release
                .as_ref()
                .map(|release| release.tag_name.as_str()),
            Some("b1"),
            "the cached tag is still served while the refresh runs"
        );
    }
}

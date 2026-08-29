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
/// llama.cpp's stable GitHub release ships this pointer instead of binaries.
const BUILD_POINTER_ASSET: &str = "nightly-tag.txt";
const BUILD_POINTER_MAX_BYTES: usize = 128;

fn parse_build_pointer_tag(value: &str) -> Option<String> {
    let tag = value.trim();
    if tag.len() >= 2
        && tag.len() <= 32
        && tag.starts_with('b')
        && tag[1..].bytes().all(|byte| byte.is_ascii_digit())
    {
        Some(tag.to_owned())
    } else {
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseAsset {
    pub name: String,
    pub browser_download_url: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Release {
    pub tag_name: String,
    #[serde(default)]
    pub assets: Vec<ReleaseAsset>,
    /// Build tag named by `nightly-tag.txt`, when the GitHub `/latest` release
    /// is a pointer rather than the binaries. Status views compare this to the
    /// installed VERSION so llama.cpp updates do not wait on a stable tag bump.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_build_tag: Option<String>,
}

impl Release {
    /// Asset matching an exact asset name.
    pub fn asset(&self, name: &str) -> Option<&ReleaseAsset> {
        self.assets.iter().find(|asset| asset.name == name)
    }

    pub fn asset_names(&self) -> impl Iterator<Item = &str> {
        self.assets.iter().map(|asset| asset.name.as_str())
    }

    /// Tag that identifies the bits on disk: the nightly pointer when present,
    /// otherwise the GitHub release tag.
    pub fn effective_tag(&self) -> &str {
        self.resolved_build_tag
            .as_deref()
            .map(str::trim)
            .filter(|tag| !tag.is_empty())
            .unwrap_or(&self.tag_name)
    }

    /// True when `/latest` is a pointer release we have not resolved yet.
    /// Status views must not claim "up to date" in this state.
    pub fn needs_build_pointer_resolution(&self) -> bool {
        self.resolved_build_tag.is_none() && self.asset(BUILD_POINTER_ASSET).is_some()
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

async fn fetch_small_text(
    client: &reqwest::Client,
    url: &str,
    user_agent: &str,
) -> anyhow::Result<String> {
    let response = client
        .get(url)
        .header("user-agent", user_agent)
        .send()
        .await
        .context("download release pointer")?
        .error_for_status()
        .context("release pointer download failed")?;
    if response
        .content_length()
        .is_some_and(|length| length > BUILD_POINTER_MAX_BYTES as u64)
    {
        anyhow::bail!("release pointer response was unexpectedly large");
    }
    let bytes = response.bytes().await.context("read release pointer")?;
    anyhow::ensure!(
        bytes.len() <= BUILD_POINTER_MAX_BYTES,
        "release pointer response was unexpectedly large"
    );
    let text = std::str::from_utf8(&bytes).context("release pointer was not UTF-8")?;
    Ok(text.trim().to_owned())
}

async fn fetch(client: &reqwest::Client, url: &str, user_agent: &str) -> anyhow::Result<Release> {
    let mut release: Release = client
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
    let pointer_url = release
        .asset(BUILD_POINTER_ASSET)
        .map(|asset| asset.browser_download_url.clone());
    if let Some(pointer_url) = pointer_url {
        match fetch_small_text(client, &pointer_url, user_agent).await {
            Ok(tag) => match parse_build_pointer_tag(&tag) {
                Some(tag) => release.resolved_build_tag = Some(tag),
                None if !tag.is_empty() => {
                    tracing::debug!(%url, tag, "build pointer had an unexpected format");
                }
                None => {}
            },
            Err(error) => {
                tracing::debug!(%url, %error, "build pointer lookup failed");
            }
        }
    }
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

/// A release from an arbitrary GitHub releases API URL, always contacting
/// GitHub and falling back to the cached response on transient failure.
///
/// This is also used for release-by-tag lookups when an upstream `latest`
/// release points at a separate build release containing the binaries.
pub async fn release_at(
    client: &reqwest::Client,
    url: &str,
    user_agent: &str,
) -> anyhow::Result<Release> {
    latest_release(client, url, user_agent).await
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
            ..Release::default()
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
            ..Release::default()
        };
        assert!(release.asset("llama-b6100-bin-macos-arm64.zip").is_some());
        assert!(release.asset("llama-b6100-bin-ubuntu-x64.zip").is_none());
        assert_eq!(
            release.asset_names().collect::<Vec<_>>(),
            vec!["llama-b6100-bin-macos-arm64.zip"]
        );
    }

    #[test]
    fn effective_tag_follows_a_resolved_nightly_pointer() {
        let unresolved = Release {
            tag_name: "v0.3.0".into(),
            assets: vec![ReleaseAsset {
                name: BUILD_POINTER_ASSET.into(),
                browser_download_url: "https://example.invalid/nightly-tag.txt".into(),
            }],
            ..Release::default()
        };
        assert_eq!(unresolved.effective_tag(), "v0.3.0");
        assert!(unresolved.needs_build_pointer_resolution());

        let resolved = Release {
            resolved_build_tag: Some("b10621".into()),
            ..unresolved
        };
        assert_eq!(resolved.effective_tag(), "b10621");
        assert!(!resolved.needs_build_pointer_resolution());
    }

    #[test]
    fn build_pointer_tags_must_be_llama_build_ids() {
        assert_eq!(
            parse_build_pointer_tag("b10621\n").as_deref(),
            Some("b10621")
        );
        assert_eq!(parse_build_pointer_tag("  b1  ").as_deref(), Some("b1"));
        for invalid in ["", "v0.3.0", "b", "b12x", "nightly"] {
            assert_eq!(parse_build_pointer_tag(invalid), None, "{invalid:?}");
        }
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
                ..Release::default()
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

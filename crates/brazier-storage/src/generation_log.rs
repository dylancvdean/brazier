//! A record of what the Generate endpoints produced, so any client of this
//! host can list past generations rather than only the ones it started.
//!
//! The media itself stays in the blob store; an entry holds the blob reference
//! and the settings the job ran with. Deleting an entry leaves the blob, which
//! a conversation may also reference.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::blob_store::StoredBlob;

/// Oldest entries are dropped beyond this.
pub const MAX_ENTRIES: usize = 500;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationBlob {
    pub sha256: String,
    pub mime_type: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationEntry {
    pub id: String,
    /// `image` or `video`.
    pub modality: String,
    /// Seconds since the Unix epoch.
    pub created_at: u64,
    pub blob: GenerationBlob,
    /// What the engine reported for the job: prompt, model and settings.
    pub metadata: Value,
}

// One process writes the file; this keeps its read-modify-write cycles apart.
static LOCK: Mutex<()> = Mutex::new(());

fn log_path(data_dir: &Path) -> PathBuf {
    data_dir.join("generations.json")
}

fn read(data_dir: &Path) -> Vec<GenerationEntry> {
    let Ok(raw) = std::fs::read(log_path(data_dir)) else {
        return Vec::new();
    };
    // Entries are read one by one so a damaged one does not hide the rest.
    serde_json::from_slice::<Vec<Value>>(&raw)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|entry| serde_json::from_value(entry).ok())
        .collect()
}

fn write(data_dir: &Path, entries: &[GenerationEntry]) -> anyhow::Result<()> {
    let path = log_path(data_dir);
    let partial = path.with_extension("json.part");
    std::fs::write(&partial, serde_json::to_vec(entries)?).context("write generation log")?;
    std::fs::rename(&partial, &path).context("replace generation log")
}

/// Add a finished generation, newest first.
pub fn record(
    data_dir: &Path,
    modality: &str,
    blob: &StoredBlob,
    metadata: &Value,
) -> anyhow::Result<GenerationEntry> {
    let entry = GenerationEntry {
        id: uuid::Uuid::new_v4().to_string(),
        modality: modality.to_owned(),
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0),
        blob: GenerationBlob {
            sha256: blob.sha256.clone(),
            mime_type: blob.mime_type.clone(),
            size_bytes: blob.size_bytes,
        },
        metadata: metadata.clone(),
    };
    let _guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut entries = read(data_dir);
    entries.insert(0, entry.clone());
    entries.truncate(MAX_ENTRIES);
    write(data_dir, &entries)?;
    Ok(entry)
}

/// Newest first.
pub fn list(data_dir: &Path) -> Vec<GenerationEntry> {
    let _guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    read(data_dir)
}

/// False when no entry has this id.
pub fn delete(data_dir: &Path, id: &str) -> anyhow::Result<bool> {
    let _guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut entries = read(data_dir);
    let before = entries.len();
    entries.retain(|entry| entry.id != id);
    if entries.len() == before {
        return Ok(false);
    }
    write(data_dir, &entries)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn blob(sha: char) -> StoredBlob {
        StoredBlob {
            sha256: sha.to_string().repeat(64),
            mime_type: "image/png".to_owned(),
            size_bytes: 12,
            original_name: Some("generated.png".to_owned()),
        }
    }

    #[test]
    fn records_newest_first_and_deletes_by_id() {
        let dir = tempfile::tempdir().unwrap();
        assert!(list(dir.path()).is_empty());
        let first = record(dir.path(), "image", &blob('a'), &json!({ "prompt": "one" })).unwrap();
        let second = record(dir.path(), "video", &blob('b'), &json!({ "prompt": "two" })).unwrap();
        let listed = list(dir.path());
        assert_eq!(listed, vec![second.clone(), first.clone()]);
        assert_eq!(listed[0].metadata["prompt"], "two");

        assert!(delete(dir.path(), &first.id).unwrap());
        assert!(!delete(dir.path(), &first.id).unwrap());
        assert_eq!(list(dir.path()), vec![second]);
    }

    #[test]
    fn keeps_readable_entries_from_a_damaged_log() {
        let dir = tempfile::tempdir().unwrap();
        let kept = record(dir.path(), "image", &blob('c'), &json!({})).unwrap();
        let mut raw: Vec<Value> =
            serde_json::from_slice(&std::fs::read(log_path(dir.path())).unwrap()).unwrap();
        raw.push(json!({ "id": 7 }));
        std::fs::write(log_path(dir.path()), serde_json::to_vec(&raw).unwrap()).unwrap();
        assert_eq!(list(dir.path()), vec![kept]);

        std::fs::write(log_path(dir.path()), b"not json").unwrap();
        assert!(list(dir.path()).is_empty());
    }

    #[test]
    fn drops_the_oldest_beyond_the_limit() {
        let dir = tempfile::tempdir().unwrap();
        let entry = |index: usize| GenerationEntry {
            id: index.to_string(),
            modality: "image".to_owned(),
            created_at: index as u64,
            blob: GenerationBlob {
                sha256: "d".repeat(64),
                mime_type: "image/png".to_owned(),
                size_bytes: 1,
            },
            metadata: json!({}),
        };
        let full: Vec<_> = (0..MAX_ENTRIES).map(entry).collect();
        write(dir.path(), &full).unwrap();
        let newest = record(dir.path(), "image", &blob('e'), &json!({})).unwrap();
        let listed = list(dir.path());
        assert_eq!(listed.len(), MAX_ENTRIES);
        assert_eq!(listed[0], newest);
        assert_eq!(listed.last().unwrap().id, (MAX_ENTRIES - 2).to_string());
    }
}

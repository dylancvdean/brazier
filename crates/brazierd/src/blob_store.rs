//! Content-addressed media blobs for chat attachments.

use std::path::{Path, PathBuf};

use anyhow::Context;
use sha2::{Digest, Sha256};

pub const MAX_IMAGE_BYTES: u64 = 10 * 1024 * 1024;
pub const MAX_AUDIO_BYTES: u64 = 20 * 1024 * 1024;
pub const MAX_VIDEO_BYTES: u64 = 50 * 1024 * 1024;

#[derive(Debug, Clone, serde::Serialize)]
pub struct StoredBlob {
    pub sha256: String,
    pub mime_type: String,
    pub size_bytes: u64,
    pub original_name: Option<String>,
}

pub fn blobs_root(data_dir: &Path) -> PathBuf {
    data_dir.join("blobs")
}

pub fn blob_path(data_dir: &Path, sha256: &str) -> anyhow::Result<PathBuf> {
    validate_sha256(sha256)?;
    Ok(blobs_root(data_dir).join(&sha256[..2]).join(sha256))
}

pub fn validate_sha256(value: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        value.len() == 64 && value.chars().all(|c| c.is_ascii_hexdigit()),
        "invalid blob id"
    );
    Ok(())
}

fn max_bytes_for_mime(mime_type: &str) -> anyhow::Result<u64> {
    if mime_type.starts_with("image/") {
        Ok(MAX_IMAGE_BYTES)
    } else if mime_type.starts_with("audio/") {
        Ok(MAX_AUDIO_BYTES)
    } else if mime_type.starts_with("video/") {
        Ok(MAX_VIDEO_BYTES)
    } else {
        anyhow::bail!("unsupported attachment type `{mime_type}` (images, audio, and video only)")
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Store bytes under their SHA-256 digest (deduplicated on disk).
pub async fn store_bytes(
    data_dir: &Path,
    bytes: &[u8],
    mime_type: &str,
    original_name: Option<&str>,
) -> anyhow::Result<StoredBlob> {
    let mime_type = mime_type.trim();
    anyhow::ensure!(!mime_type.is_empty(), "mime_type is required");
    let max = max_bytes_for_mime(mime_type)?;
    anyhow::ensure!(
        bytes.len() as u64 <= max,
        "attachment exceeds the {} limit for this media type",
        format_bytes(max)
    );

    let sha256 = sha256_hex(bytes);
    let path = blob_path(data_dir, &sha256)?;
    if !path.is_file() {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .context("create blob directory")?;
        }
        tokio::fs::write(&path, bytes).await.context("write blob")?;
    }

    Ok(StoredBlob {
        sha256,
        mime_type: mime_type.to_owned(),
        size_bytes: bytes.len() as u64,
        original_name: original_name.map(str::to_owned),
    })
}

pub async fn read_blob(data_dir: &Path, sha256: &str) -> anyhow::Result<(Vec<u8>, String)> {
    let path = blob_path(data_dir, sha256)?;
    anyhow::ensure!(path.is_file(), "blob not found");
    let bytes = tokio::fs::read(&path).await.context("read blob")?;
    Ok((bytes, mime_type_from_path(&path)))
}

fn mime_type_from_path(path: &Path) -> String {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("jpg" | "jpeg") => "image/jpeg".into(),
        Some("png") => "image/png".into(),
        Some("gif") => "image/gif".into(),
        Some("webp") => "image/webp".into(),
        Some("wav") => "audio/wav".into(),
        Some("mp3") => "audio/mpeg".into(),
        Some("mp4") => "video/mp4".into(),
        Some("webm") => "video/webm".into(),
        _ => "application/octet-stream".into(),
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{} MB", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        format!("{} KB", bytes / 1024)
    } else {
        format!("{bytes} B")
    }
}

/// Collect blob digests referenced in a message content value.
pub fn blob_refs_in_content(content: &serde_json::Value) -> Vec<String> {
    let mut refs = Vec::new();
    match content {
        serde_json::Value::Array(parts) => {
            for part in parts {
                if let Some(sha256) = part
                    .get("brazier_blob")
                    .and_then(|blob| blob.get("sha256"))
                    .and_then(serde_json::Value::as_str)
                {
                    refs.push(sha256.to_owned());
                }
            }
        }
        _ => {}
    }
    refs.sort();
    refs.dedup();
    refs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_oversized_images() {
        let err = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async {
                store_bytes(
                    std::path::Path::new("/tmp"),
                    &vec![0_u8; (MAX_IMAGE_BYTES + 1) as usize],
                    "image/png",
                    None,
                )
                .await
            })
            .unwrap_err();
        assert!(err.to_string().contains("limit"));
    }

    #[tokio::test]
    async fn deduplicates_identical_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let first = store_bytes(dir.path(), b"hello", "image/png", Some("a.png"))
            .await
            .unwrap();
        let second = store_bytes(dir.path(), b"hello", "image/png", Some("b.png"))
            .await
            .unwrap();
        assert_eq!(first.sha256, second.sha256);
        assert!(blob_path(dir.path(), &first.sha256).unwrap().is_file());
    }
}

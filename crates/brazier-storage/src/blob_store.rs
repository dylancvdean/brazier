//! Content-addressed media blobs for chat attachments.

use std::path::{Path, PathBuf};

use anyhow::Context;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

pub const MAX_IMAGE_BYTES: u64 = 10 * 1024 * 1024;
pub const MAX_AUDIO_BYTES: u64 = 20 * 1024 * 1024;
pub const MAX_VIDEO_BYTES: u64 = 50 * 1024 * 1024;
pub const MAX_DOCUMENT_BYTES: u64 = 25 * 1024 * 1024;

/// Files we can retain as chat documents. Text-based formats are provided to
/// the model directly; PDFs and Office documents are handled by the
/// document-preparation pipeline.
pub fn is_document_mime(mime_type: &str) -> bool {
    matches!(
        mime_type,
        "application/pdf"
            | "application/json"
            | "application/xml"
            | "application/rtf"
            | "text/rtf"
            | "application/msword"
            | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            | "text/plain"
            | "text/markdown"
            | "text/csv"
            | "text/html"
    ) || mime_type.starts_with("text/")
}

/// Browsers do not always report a type for Office documents; when the upload
/// arrives as a generic stream, trust the file extension instead.
fn document_mime_for_name(name: &str) -> Option<&'static str> {
    let extension = name.rsplit('.').next()?.to_ascii_lowercase();
    Some(match extension.as_str() {
        "pdf" => "application/pdf",
        "rtf" => "application/rtf",
        "doc" => "application/msword",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        _ => return None,
    })
}

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
    } else if is_document_mime(mime_type) {
        Ok(MAX_DOCUMENT_BYTES)
    } else {
        anyhow::bail!(
            "unsupported attachment type `{mime_type}` (images, audio, video, PDFs, Office documents, and text are supported)"
        )
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
    let mime_type = if mime_type == "application/octet-stream" {
        original_name
            .and_then(document_mime_for_name)
            .unwrap_or(mime_type)
    } else {
        mime_type
    };
    let max = max_bytes_for_mime(mime_type)?;
    anyhow::ensure!(
        bytes.len() as u64 <= max,
        "attachment exceeds the {} limit for this media type",
        format_bytes(max)
    );

    let sha256 = sha256_hex(bytes);
    let path = blob_path(data_dir, &sha256)?;
    if !path.is_file() {
        publish_blob(&path, bytes).await?;
    }

    Ok(StoredBlob {
        sha256,
        mime_type: mime_type.to_owned(),
        size_bytes: bytes.len() as u64,
        original_name: original_name.map(str::to_owned),
    })
}

async fn publish_blob(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("blob path has no parent"))?;
    #[cfg(unix)]
    {
        let mut builder = tokio::fs::DirBuilder::new();
        builder.recursive(true);
        builder.mode(0o700);
        builder
            .create(parent)
            .await
            .context("create blob directory")?;
    }
    #[cfg(not(unix))]
    {
        tokio::fs::create_dir_all(parent)
            .await
            .context("create blob directory")?;
    }
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("blob"),
        uuid::Uuid::new_v4()
    ));
    let mut options = tokio::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        options.mode(0o600);
    }
    let mut file = options
        .open(&temporary)
        .await
        .context("write temporary blob")?;
    file.write_all(bytes)
        .await
        .context("write temporary blob")?;
    file.flush().await.context("flush temporary blob")?;
    file.sync_all().await.context("sync temporary blob")?;
    drop(file);
    if let Err(error) = tokio::fs::rename(&temporary, path).await {
        if !path.is_file() {
            let _ = tokio::fs::remove_file(&temporary).await;
            return Err(error).context("publish blob");
        }
        let _ = tokio::fs::remove_file(&temporary).await;
    }
    #[cfg(unix)]
    {
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

pub async fn read_blob(data_dir: &Path, sha256: &str) -> anyhow::Result<(Vec<u8>, String)> {
    let path = blob_path(data_dir, sha256)?;
    anyhow::ensure!(path.is_file(), "blob not found");
    let bytes = tokio::fs::read(&path).await.context("read blob")?;
    let mime_type = mime_type_from_bytes(&bytes).unwrap_or_else(|| mime_type_from_path(&path));
    Ok((bytes, mime_type))
}

fn mime_type_from_bytes(bytes: &[u8]) -> Option<String> {
    let mime = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        "image/png"
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        "image/jpeg"
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        "image/gif"
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        "image/webp"
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        "audio/wav"
    } else if bytes.get(4..8) == Some(b"ftyp") {
        "video/mp4"
    } else if bytes.starts_with(b"\x1a\x45\xdf\xa3") {
        "video/webm"
    } else {
        return None;
    };
    Some(mime.to_owned())
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
    if let serde_json::Value::Array(parts) = content {
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

    #[test]
    fn detects_content_type_for_extensionless_blobs() {
        assert_eq!(
            mime_type_from_bytes(b"\x89PNG\r\n\x1a\nrest").as_deref(),
            Some("image/png")
        );
        assert_eq!(
            mime_type_from_bytes(b"\xff\xd8\xffrest").as_deref(),
            Some("image/jpeg")
        );
        assert_eq!(
            mime_type_from_bytes(b"\0\0\0\x18ftypisom").as_deref(),
            Some("video/mp4")
        );
    }

    #[tokio::test]
    async fn stores_pdf_and_text_documents() {
        let dir = tempfile::tempdir().unwrap();
        let pdf = store_bytes(
            dir.path(),
            b"%PDF-1.7",
            "application/pdf",
            Some("notes.pdf"),
        )
        .await
        .unwrap();
        let text = store_bytes(dir.path(), b"hello", "text/markdown", Some("notes.md"))
            .await
            .unwrap();
        assert_eq!(pdf.mime_type, "application/pdf");
        assert_eq!(text.mime_type, "text/markdown");
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

    #[tokio::test]
    async fn concurrent_writes_publish_one_complete_blob() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = vec![42_u8; 128 * 1024];
        let (first, second) = tokio::join!(
            store_bytes(dir.path(), &bytes, "image/png", None),
            store_bytes(dir.path(), &bytes, "image/png", None)
        );
        let first = first.unwrap();
        assert_eq!(first.sha256, second.unwrap().sha256);
        assert_eq!(
            tokio::fs::read(blob_path(dir.path(), &first.sha256).unwrap())
                .await
                .unwrap(),
            bytes
        );
        let entries = std::fs::read_dir(blobs_root(dir.path()).join(&first.sha256[..2]))
            .unwrap()
            .count();
        assert_eq!(entries, 1);
    }
}

use std::path::Path;

use anyhow::Context;
use serde::Serialize;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

pub(crate) async fn write_json(
    path: &Path,
    value: &(impl Serialize + ?Sized),
    label: &str,
) -> anyhow::Result<()> {
    let bytes = serde_json::to_vec_pretty(value).with_context(|| format!("encode {label}"))?;
    write(path, &bytes, label).await
}

pub(crate) async fn write(path: &Path, bytes: &[u8], label: &str) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("{label} path has no parent"))?;
    tokio::fs::create_dir_all(parent)
        .await
        .with_context(|| format!("create {label} directory"))?;
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("state");
    let temporary = parent.join(format!(".{name}.{}.tmp", Uuid::new_v4()));

    let mut options = tokio::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        options.mode(0o600);
    }
    let mut file = options
        .open(&temporary)
        .await
        .with_context(|| format!("create temporary {label}"))?;
    if let Err(error) = file.write_all(bytes).await {
        drop(file);
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(error).with_context(|| format!("write temporary {label}"));
    }
    file.flush()
        .await
        .with_context(|| format!("flush temporary {label}"))?;
    drop(file);

    #[cfg(windows)]
    if path.is_file() {
        tokio::fs::remove_file(path)
            .await
            .with_context(|| format!("replace {label}"))?;
    }
    if let Err(error) = tokio::fs::rename(&temporary, path).await {
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(error).with_context(|| format!("commit {label}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn concurrent_writers_do_not_share_a_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let writer_one = json!({ "writer": 1 });
        let writer_two = json!({ "writer": 2 });
        let (first, second) = tokio::join!(
            write_json(&path, &writer_one, "settings"),
            write_json(&path, &writer_two, "settings")
        );
        first.unwrap();
        second.unwrap();

        let value: serde_json::Value =
            serde_json::from_slice(&tokio::fs::read(&path).await.unwrap()).unwrap();
        assert!(value["writer"] == 1 || value["writer"] == 2);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
}

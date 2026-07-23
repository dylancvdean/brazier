//! Hugging Face access token storage (env override + optional persisted token).

use std::path::{Path, PathBuf};

use anyhow::Context;

pub fn token_file(data_dir: &Path) -> PathBuf {
    data_dir.join("huggingface").join("token")
}

/// Resolved token for Hub API calls, if any.
pub fn load_token(data_dir: &Path) -> Option<String> {
    if let Ok(token) = std::env::var("HF_TOKEN") {
        let token = token.trim();
        if !token.is_empty() {
            return Some(token.to_owned());
        }
    }
    if let Ok(token) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        let token = token.trim();
        if !token.is_empty() {
            return Some(token.to_owned());
        }
    }
    let path = token_file(data_dir);
    std::fs::read_to_string(path)
        .ok()
        .map(|token| token.trim().to_owned())
        .filter(|token| !token.is_empty())
}

pub fn token_configured(data_dir: &Path) -> bool {
    load_token(data_dir).is_some()
}

pub async fn save_token(data_dir: &Path, token: &str) -> anyhow::Result<()> {
    let token = token.trim();
    anyhow::ensure!(!token.is_empty(), "token must not be empty");
    let path = token_file(data_dir);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("create huggingface config directory")?;
    }
    tokio::fs::write(&path, format!("{token}\n"))
        .await
        .context("write hugging face token")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(&path)?.permissions();
        permissions.set_mode(0o600);
        std::fs::set_permissions(&path, permissions)?;
    }
    Ok(())
}

pub async fn clear_token(data_dir: &Path) -> anyhow::Result<()> {
    let path = token_file(data_dir);
    if path.is_file() {
        tokio::fs::remove_file(path)
            .await
            .context("remove hugging face token")?;
    }
    Ok(())
}

pub fn apply_auth(builder: reqwest::RequestBuilder, data_dir: &Path) -> reqwest::RequestBuilder {
    if let Some(token) = load_token(data_dir) {
        builder.header("authorization", format!("Bearer {token}"))
    } else {
        builder
    }
}

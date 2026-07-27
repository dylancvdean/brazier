//! Per-model runtime bindings.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ModelRuntimeBindings {
    pub bindings: HashMap<String, String>,
}

impl ModelRuntimeBindings {
    pub fn get(&self, model_id: &str) -> Option<&str> {
        self.bindings.get(model_id).map(String::as_str)
    }

    pub fn set(&mut self, model_id: impl Into<String>, runtime_id: impl Into<String>) {
        self.bindings.insert(model_id.into(), runtime_id.into());
    }

    pub fn remove(&mut self, model_id: &str) -> Option<String> {
        self.bindings.remove(model_id)
    }
}

pub fn bindings_path(data_dir: &Path) -> PathBuf {
    data_dir.join("model-runtime-bindings.json")
}

pub fn load(data_dir: &Path) -> ModelRuntimeBindings {
    let path = bindings_path(data_dir);
    let Ok(bytes) = std::fs::read(&path) else {
        return ModelRuntimeBindings::default();
    };
    serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        tracing::warn!(%error, path = %path.display(), "ignoring invalid model runtime bindings");
        ModelRuntimeBindings::default()
    })
}

pub async fn save(data_dir: &Path, bindings: &ModelRuntimeBindings) -> anyhow::Result<()> {
    let path = bindings_path(data_dir);
    crate::persistence::write_json(&path, bindings, "model runtime bindings").await
}

pub async fn set_binding(
    data_dir: &Path,
    model_id: &str,
    runtime_id: &str,
) -> anyhow::Result<ModelRuntimeBindings> {
    let mut bindings = load(data_dir);
    bindings.set(model_id, runtime_id);
    save(data_dir, &bindings).await?;
    Ok(bindings)
}

pub async fn clear_binding(
    data_dir: &Path,
    model_id: &str,
) -> anyhow::Result<ModelRuntimeBindings> {
    let mut bindings = load(data_dir);
    bindings.remove(model_id);
    save(data_dir, &bindings).await?;
    Ok(bindings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn persists_bindings_round_trip() {
        let dir = tempdir().unwrap();
        let mut bindings = ModelRuntimeBindings::default();
        bindings.set("gguf:acme/model/file.gguf", "source-abc123");
        save(dir.path(), &bindings).await.unwrap();
        let loaded = load(dir.path());
        assert_eq!(
            loaded.get("gguf:acme/model/file.gguf"),
            Some("source-abc123")
        );
    }
}

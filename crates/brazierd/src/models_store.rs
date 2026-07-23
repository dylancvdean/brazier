//! On-disk GGUF model layout and listing.

use std::path::{Path, PathBuf};

use crate::types::{ModelCapabilities, ModelDescriptor};

/// Content-keyed root for downloaded GGUF weights.
pub fn gguf_root(data_dir: &Path) -> PathBuf {
    data_dir.join("models").join("gguf")
}

/// Temporary directory for partial downloads.
pub fn downloads_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("downloads")
}

/// Stable model id for a GGUF file under the models root.
///
/// Format: `gguf:{repo_relative_path}` where path uses `/` separators.
pub fn model_id_for_path(gguf_root: &Path, file: &Path) -> anyhow::Result<String> {
    let relative = file
        .strip_prefix(gguf_root)
        .map_err(|_| anyhow::anyhow!("model path is outside the GGUF store"))?;
    let key = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    anyhow::ensure!(!key.is_empty(), "empty model key");
    anyhow::ensure!(!key.contains(".."), "model key must not contain '..'");
    Ok(format!("gguf:{key}"))
}

/// Resolve a `gguf:...` model id to an absolute path under the data directory.
pub fn path_for_model_id(data_dir: &Path, model_id: &str) -> anyhow::Result<PathBuf> {
    let Some(key) = model_id.strip_prefix("gguf:") else {
        anyhow::bail!("not a local GGUF model id: {model_id}");
    };
    anyhow::ensure!(!key.is_empty(), "empty GGUF model key");
    anyhow::ensure!(
        !key.split('/')
            .any(|part| part.is_empty() || part == "." || part == ".."),
        "invalid GGUF model key"
    );
    let path = gguf_root(data_dir).join(key);
    anyhow::ensure!(
        path.extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf")),
        "model path must end in .gguf"
    );
    Ok(path)
}

/// Destination path for a Hugging Face GGUF artifact.
pub fn download_destination(
    data_dir: &Path,
    repo_id: &str,
    filename: &str,
) -> anyhow::Result<PathBuf> {
    validate_repo_id(repo_id)?;
    validate_filename(filename)?;
    Ok(gguf_root(data_dir).join(repo_id).join(filename))
}

pub fn validate_repo_id(repo_id: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !repo_id.is_empty() && repo_id.len() <= 200,
        "invalid repository id"
    );
    let mut parts = repo_id.split('/');
    let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        anyhow::bail!("repository id must be owner/name");
    };
    for part in [owner, name] {
        anyhow::ensure!(
            !part.is_empty()
                && part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')),
            "invalid repository id segment"
        );
    }
    Ok(())
}

pub fn validate_filename(filename: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !filename.is_empty() && filename.len() <= 260,
        "invalid filename"
    );
    anyhow::ensure!(
        !filename.starts_with('/') && !filename.contains('\\'),
        "filename must be a relative path"
    );
    anyhow::ensure!(
        !filename
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == ".."),
        "filename must not contain empty or parent path segments"
    );
    anyhow::ensure!(
        filename.to_ascii_lowercase().ends_with(".gguf"),
        "only GGUF artifacts are supported"
    );
    Ok(())
}

fn is_projector(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.to_ascii_lowercase().contains("mmproj"))
}

pub fn projector_for_model(model_path: &Path) -> Option<PathBuf> {
    let directory = model_path.parent()?;
    std::fs::read_dir(directory)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
                && is_projector(path)
        })
}

fn gguf_capabilities(path: &Path) -> ModelCapabilities {
    let multimodal = projector_for_model(path).is_some();
    let mut input_modalities = vec!["text".into()];
    if multimodal {
        input_modalities.extend(["image".into(), "audio".into(), "video".into()]);
    }
    ModelCapabilities {
        input_modalities,
        output_modalities: vec!["text".into()],
        streaming: true,
        tools: true,
        reasoning: true,
    }
}

/// Scan the GGUF store and return OpenAI-style model descriptors.
pub fn list_gguf_models(data_dir: &Path) -> anyhow::Result<Vec<ModelDescriptor>> {
    let root = gguf_root(data_dir);
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut models = Vec::new();
    collect_gguf(&root, &root, &mut models)?;
    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

fn collect_gguf(root: &Path, dir: &Path, models: &mut Vec<ModelDescriptor>) -> anyhow::Result<()> {
    let entries = std::fs::read_dir(dir)
        .map_err(|error| anyhow::anyhow!("read model directory {}: {error}", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_gguf(root, &path, models)?;
            continue;
        }
        if !path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            continue;
        }
        if is_projector(&path) {
            continue;
        }
        let id = model_id_for_path(root, &path)?;
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| id.clone());
        models.push(ModelDescriptor {
            id,
            name,
            engine: "llama.cpp".to_owned(),
            capabilities: gguf_capabilities(&path),
            size_bytes: std::fs::metadata(&path).ok().map(|meta| meta.len()),
        });
    }
    Ok(())
}

/// Delete a downloaded model file and prune empty parent directories.
pub fn delete_model(data_dir: &Path, model_id: &str) -> anyhow::Result<PathBuf> {
    let path = path_for_model_id(data_dir, model_id)?;
    anyhow::ensure!(path.is_file(), "model file not found for {model_id}");
    std::fs::remove_file(&path)
        .map_err(|error| anyhow::anyhow!("delete {}: {error}", path.display()))?;
    let root = gguf_root(data_dir);
    let mut directory = path.parent().map(Path::to_path_buf);
    while let Some(current) = directory {
        if current == root || !current.starts_with(&root) {
            break;
        }
        let empty = std::fs::read_dir(&current)
            .map(|mut entries| entries.next().is_none())
            .unwrap_or(false);
        if !empty {
            break;
        }
        let _ = std::fs::remove_dir(&current);
        directory = current.parent().map(Path::to_path_buf);
    }
    Ok(path)
}

/// Prefer a practical default quant from a list of GGUF filenames.
pub fn prefer_gguf_filename(filenames: &[String]) -> Option<String> {
    let ggufs: Vec<&String> = filenames
        .iter()
        .filter(|name| name.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    if ggufs.is_empty() {
        return None;
    }
    const PREFERRED: &[&str] = &[
        "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q4_0", "q5_0", "q3_k_m", "q6_k", "q8_0",
    ];
    for quant in PREFERRED {
        if let Some(name) = ggufs
            .iter()
            .find(|name| name.to_ascii_lowercase().contains(quant))
        {
            return Some((*name).clone());
        }
    }
    ggufs.into_iter().min_by_key(|name| name.len()).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn model_id_round_trips_through_path() {
        let dir = tempdir().unwrap();
        let root = gguf_root(dir.path());
        let file = root
            .join("unsloth")
            .join("Tiny-GGUF")
            .join("model-Q4_K_M.gguf");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"gguf").unwrap();
        let id = model_id_for_path(&root, &file).unwrap();
        assert_eq!(id, "gguf:unsloth/Tiny-GGUF/model-Q4_K_M.gguf");
        let resolved = path_for_model_id(dir.path(), &id).unwrap();
        assert_eq!(resolved, file);
    }

    #[test]
    fn lists_nested_gguf_files() {
        let dir = tempdir().unwrap();
        let file = download_destination(dir.path(), "acme/demo", "demo-q4_k_m.gguf").unwrap();
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"gguf").unwrap();
        let models = list_gguf_models(dir.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].engine, "llama.cpp");
        assert!(models[0].capabilities.streaming);
        assert_eq!(models[0].id, "gguf:acme/demo/demo-q4_k_m.gguf");
    }

    #[test]
    fn projector_enables_multimodal_capabilities_without_becoming_a_model() {
        let dir = tempdir().unwrap();
        let model = download_destination(dir.path(), "acme/vision", "model-q4.gguf").unwrap();
        let projector = download_destination(dir.path(), "acme/vision", "mmproj-f16.gguf").unwrap();
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&projector, b"projector").unwrap();
        let models = list_gguf_models(dir.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(projector_for_model(&model), Some(projector));
        assert!(
            models[0]
                .capabilities
                .input_modalities
                .contains(&"image".to_owned())
        );
    }

    #[test]
    fn rejects_path_traversal_in_model_id() {
        let dir = tempdir().unwrap();
        assert!(path_for_model_id(dir.path(), "gguf:../escape.gguf").is_err());
        assert!(path_for_model_id(dir.path(), "gguf:/abs.gguf").is_err());
        assert!(validate_filename("../x.gguf").is_err());
        assert!(validate_filename("ok/nested-q4_k_m.gguf").is_ok());
        assert!(validate_repo_id("../evil/name").is_err());
    }

    #[test]
    fn prefers_balanced_quants() {
        let names = vec![
            "model-f16.gguf".into(),
            "model-q8_0.gguf".into(),
            "model-q4_k_m.gguf".into(),
            "readme.md".into(),
        ];
        assert_eq!(
            prefer_gguf_filename(&names).as_deref(),
            Some("model-q4_k_m.gguf")
        );
    }
}

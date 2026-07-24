//! External model library folders and common-path suggestions.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::models_store;

#[derive(Debug, Clone, Serialize)]
pub struct LibraryPathSuggestion {
    pub id: &'static str,
    pub label: &'static str,
    pub path: String,
    pub exists: bool,
    pub gguf_count: u32,
    pub mlx_count: u32,
    pub configured: bool,
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn candidate_paths() -> Vec<(&'static str, &'static str, PathBuf)> {
    let Some(home) = home_dir() else {
        return Vec::new();
    };
    let mut candidates = Vec::new();
    candidates.push(("lmstudio-home", "LM Studio", home.join(".lmstudio/models")));
    if cfg!(target_os = "macos") {
        candidates.push((
            "lm-studio-app-support",
            "LM Studio (App Support)",
            home.join("Library/Application Support/LM Studio/models"),
        ));
        candidates.push((
            "lm-studio-cache",
            "LM Studio (cache)",
            home.join(".cache/lm-studio/models"),
        ));
    }
    if cfg!(target_os = "windows") {
        candidates.push((
            "lm-studio-cache",
            "LM Studio (cache)",
            home.join(".cache/lm-studio/models"),
        ));
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            candidates.push((
                "lm-studio-local",
                "LM Studio (AppData)",
                PathBuf::from(local).join("LM Studio/models"),
            ));
        }
    }
    if cfg!(target_os = "linux") {
        candidates.push((
            "lm-studio-cache",
            "LM Studio (cache)",
            home.join(".cache/lm-studio/models"),
        ));
        candidates.push((
            "lm-studio-local",
            "LM Studio (local share)",
            home.join(".local/share/lm-studio/models"),
        ));
    }
    candidates.push((
        "huggingface-hub",
        "Hugging Face cache",
        home.join(".cache/huggingface/hub"),
    ));
    candidates
}

pub fn count_mlx_model_dirs(root: &Path) -> u32 {
    if !root.is_dir() {
        return 0;
    }
    let mut count = 0u32;
    count_mlx_model_dirs_recursive(root, &mut count);
    count
}

fn count_mlx_model_dirs_recursive(dir: &Path, count: &mut u32) {
    if directory_is_mlx_model(dir) {
        *count = count.saturating_add(1);
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            count_mlx_model_dirs_recursive(&path, count);
        }
    }
}

fn directory_is_mlx_model(dir: &Path) -> bool {
    models_store::directory_is_mlx_model(dir)
}

pub fn count_gguf_files(root: &Path) -> u32 {
    if !root.is_dir() {
        return 0;
    }
    let mut count = 0u32;
    count_gguf_recursive(root, &mut count);
    count
}

fn count_gguf_recursive(dir: &Path, count: &mut u32) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            count_gguf_recursive(&path, count);
            continue;
        }
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
            && !models_store::is_projector_file(&path)
        {
            *count = count.saturating_add(1);
        }
    }
}

pub fn library_path_suggestions(configured: &[String]) -> Vec<LibraryPathSuggestion> {
    let configured_canonical: Vec<PathBuf> = configured
        .iter()
        .filter_map(|path| std::fs::canonicalize(path).ok())
        .collect();
    let mut seen_paths = std::collections::HashSet::new();
    let mut suggestions = Vec::new();
    for (id, label, path) in candidate_paths() {
        if !path.is_dir() {
            continue;
        }
        let canonical = path.canonicalize().ok();
        if let Some(canonical) = &canonical
            && !seen_paths.insert(canonical.clone())
        {
            continue;
        }
        let configured = canonical.as_ref().is_some_and(|canonical| {
            configured_canonical
                .iter()
                .any(|existing| existing == canonical)
        });
        suggestions.push(LibraryPathSuggestion {
            id,
            label,
            path: path.display().to_string(),
            exists: true,
            gguf_count: count_gguf_files(&path),
            mlx_count: count_mlx_model_dirs(&path),
            configured,
        });
    }
    suggestions
}

pub fn normalize_library_paths(paths: &[String]) -> anyhow::Result<Vec<String>> {
    let mut normalized = Vec::new();
    for path in paths {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            continue;
        }
        let path = PathBuf::from(trimmed);
        anyhow::ensure!(
            path.is_absolute(),
            "library path must be absolute: {}",
            path.display()
        );
        let metadata = std::fs::metadata(&path)
            .map_err(|error| anyhow::anyhow!("library path {}: {error}", path.display()))?;
        anyhow::ensure!(
            metadata.is_dir(),
            "library path must be a directory: {}",
            path.display()
        );
        let canonical = std::fs::canonicalize(&path)
            .map_err(|error| anyhow::anyhow!("library path {}: {error}", path.display()))?;
        if normalized
            .iter()
            .any(|existing: &String| PathBuf::from(existing) == canonical)
        {
            continue;
        }
        normalized.push(canonical.display().to_string());
    }
    Ok(normalized)
}

pub fn label_for_library_path(path: &str) -> String {
    let path_buf = PathBuf::from(path);
    let canonical = std::fs::canonicalize(path).ok();
    for (_, label, candidate) in candidate_paths() {
        if candidate.display().to_string() == path {
            return label.to_owned();
        }
        if let (Ok(left), Some(right)) = (candidate.canonicalize(), &canonical)
            && left == *right
        {
            return label.to_owned();
        }
    }
    path_buf
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggestions_skip_missing_directories() {
        let suggestions = library_path_suggestions(&[]);
        assert!(suggestions.iter().all(|entry| entry.exists));
        for entry in &suggestions {
            assert!(PathBuf::from(&entry.path).is_dir());
        }
    }
}

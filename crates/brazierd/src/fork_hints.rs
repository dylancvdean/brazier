//! Detect custom runtime forks linked from Hugging Face model documentation.

use std::collections::HashSet;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{build_recipe, hf_auth, models_store};

const SUPPORTED_ENGINES: &[&str] = &["llama.cpp", "mlx-lm", "mlx-vlm"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeForkHint {
    pub engine: String,
    pub display_name: String,
    pub repository: String,
    pub trusted: bool,
    pub summary: String,
}

#[derive(Debug, Clone)]
pub struct ModelLoadError {
    pub cause: String,
    pub fork_hints: Vec<RuntimeForkHint>,
}

impl std::fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.cause)?;
        if !self.fork_hints.is_empty() {
            write!(f, "\n\n{}", format_fork_hints(&self.fork_hints))?;
        }
        Ok(())
    }
}

impl std::error::Error for ModelLoadError {}

pub fn format_fork_hints(hints: &[RuntimeForkHint]) -> String {
    let mut lines = vec!["The model README links custom runtime forks:".to_owned()];
    for hint in hints {
        lines.push(format!(
            "• {} — {} ({})",
            hint.display_name, hint.repository, hint.summary
        ));
    }
    lines.push("Build the fork in Manage → Runtimes.".to_owned());
    lines.join("\n")
}

/// Normalize a GitHub URL to `https://github.com/{owner}/{repo}`.
pub fn normalize_github_repo_url(url: &str) -> Option<String> {
    let trimmed = url
        .trim()
        .trim_end_matches([')', ']', '.', ',', ';', '"', '\'']);
    let lower = trimmed.to_ascii_lowercase();
    let marker = "github.com/";
    let index = lower.find(marker)?;
    let tail = &trimmed[index + marker.len()..];
    let path = tail.split('#').next()?.split('?').next()?;
    let mut parts = path.trim_start_matches('/').split('/');
    let owner = parts.next()?.trim();
    let repo = parts.next()?.trim().trim_end_matches(".git");
    if owner.is_empty() || repo.is_empty() {
        return None;
    }
    Some(format!("https://github.com/{owner}/{repo}"))
}

fn repo_matches_engine(repository: &str, engine: &str) -> bool {
    let lower = repository.to_ascii_lowercase();
    match engine {
        "llama.cpp" => lower.contains("llama.cpp") || lower.contains("llama-cpp"),
        "mlx-lm" => lower.contains("mlx-lm") || lower.contains("mlx_lm"),
        "mlx-vlm" => lower.contains("mlx-vlm") || lower.contains("mlx_vlm"),
        _ => false,
    }
}

fn is_trusted_origin(repository: &str, engine: &str) -> bool {
    let Ok(recipe) = build_recipe::recipe(engine) else {
        return false;
    };
    recipe.upstream_origins.iter().any(|origin| {
        normalize_github_repo_url(origin).as_deref() == Some(repository)
    })
}

/// Extract GitHub repository URLs referenced in markdown or plain text.
pub fn extract_github_urls(text: &str) -> Vec<String> {
    let mut urls = Vec::new();
    let mut search = text;
    while let Some(index) = search.find("http") {
        let rest = &search[index..];
        let end = rest
            .char_indices()
            .find(|(_, ch)| {
                ch.is_whitespace() || matches!(ch, ')' | ']' | '"' | '\'' | '<' | '>')
            })
            .map(|(offset, _)| offset)
            .unwrap_or(rest.len());
        let candidate = rest[..end].trim_end_matches(['.', ',', ';']);
        if candidate.contains("github.com/") {
            urls.push(candidate.to_owned());
        }
        search = &search[index + 1..];
    }
    urls
}

/// Scan documentation text for links to non-official forks of supported runtimes.
pub fn runtime_fork_hints_from_text(text: &str) -> Vec<RuntimeForkHint> {
    let mut hints = Vec::new();
    let mut seen = HashSet::new();
    for url in extract_github_urls(text) {
        let Some(repository) = normalize_github_repo_url(&url) else {
            continue;
        };
        if !seen.insert(repository.clone()) {
            continue;
        }
        for engine in SUPPORTED_ENGINES {
            if !repo_matches_engine(&repository, engine) {
                continue;
            }
            let Ok(recipe) = build_recipe::recipe(engine) else {
                continue;
            };
            let trusted = is_trusted_origin(&repository, engine);
            if trusted {
                continue;
            }
            hints.push(RuntimeForkHint {
                engine: (*engine).to_owned(),
                display_name: recipe.display_name.clone(),
                repository: repository.clone(),
                trusted: false,
                summary: format!(
                    "README links a custom {} fork",
                    recipe.display_name
                ),
            });
        }
    }
    hints
}

async fn fetch_readme_file(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    repo_id: &str,
) -> anyhow::Result<String> {
    for candidate in ["README.md", "Readme.md", "readme.md"] {
        let url = format!("https://huggingface.co/{repo_id}/resolve/main/{candidate}");
        let response = hf_auth::apply_auth(client.get(url), data_dir)
            .send()
            .await
            .with_context(|| format!("fetch {candidate} for {repo_id}"))?;
        if response.status().is_success() {
            return response.text().await.context("decode README body");
        }
    }
    anyhow::bail!("README not found for {repo_id}")
}

/// Load model card content or README markdown from Hugging Face.
pub async fn model_documentation(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    repo_id: &str,
) -> anyhow::Result<String> {
    models_store::validate_repo_id(repo_id)?;
    let url = format!("https://huggingface.co/api/models/{repo_id}");
    let value: Value = hf_auth::apply_auth(client.get(url), data_dir)
        .send()
        .await
        .context("contact Hugging Face model API")?
        .error_for_status()
        .context("Hugging Face model request failed")?
        .json()
        .await
        .context("decode Hugging Face model response")?;
    if let Some(content) = value
        .pointer("/cardData/content")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|content| !content.is_empty())
    {
        return Ok(content.to_owned());
    }
    fetch_readme_file(client, data_dir, repo_id).await
}

/// Fetch Hugging Face documentation and return runtime fork hints, if any.
pub async fn hints_for_repo(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    repo_id: &str,
) -> anyhow::Result<Vec<RuntimeForkHint>> {
    match model_documentation(client, data_dir, repo_id).await {
        Ok(text) => Ok(runtime_fork_hints_from_text(&text)),
        Err(error) => {
            tracing::debug!(repo_id, error = %error, "model documentation unavailable for fork hints");
            Ok(Vec::new())
        }
    }
}

pub async fn load_error_with_hints(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    model_id: &str,
    error: anyhow::Error,
) -> ModelLoadError {
    let cause = error.to_string();
    let fork_hints = if let Some(repo_id) = models_store::managed_repo_id(model_id) {
        hints_for_repo(client, data_dir, &repo_id)
            .await
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    ModelLoadError { cause, fork_hints }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_github_urls() {
        assert_eq!(
            normalize_github_repo_url("https://github.com/city96/llama.cpp.git"),
            Some("https://github.com/city96/llama.cpp".into())
        );
        assert_eq!(
            normalize_github_repo_url("https://github.com/city96/llama.cpp/tree/feature-x)"),
            Some("https://github.com/city96/llama.cpp".into())
        );
    }

    #[test]
    fn detects_custom_llama_fork_in_readme() {
        let text = r#"
        This model requires a patched llama.cpp build:
        https://github.com/example/llama.cpp
        "#;
        let hints = runtime_fork_hints_from_text(text);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].engine, "llama.cpp");
        assert_eq!(hints[0].repository, "https://github.com/example/llama.cpp");
        assert!(!hints[0].trusted);
    }

    #[test]
    fn ignores_official_upstream_links() {
        let text = "Use https://github.com/ggml-org/llama.cpp for inference.";
        assert!(runtime_fork_hints_from_text(text).is_empty());
    }

    #[test]
    fn detects_mlx_lm_fork() {
        let text = "Build https://github.com/acme/mlx-lm first.";
        let hints = runtime_fork_hints_from_text(text);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].engine, "mlx-lm");
    }
}

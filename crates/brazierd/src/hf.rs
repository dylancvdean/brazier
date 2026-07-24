use std::cmp::Ordering;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::models_store::prefer_gguf_filename;
use crate::hf_auth;

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub q: Option<String>,
    pub engine: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HubModel {
    pub id: String,
    pub author: String,
    pub downloads: u64,
    pub likes: u64,
    pub last_modified: Option<String>,
    pub tags: Vec<String>,
    pub gated: bool,
    pub score: f64,
    pub preferred_quantizer: bool,
}

fn tag_matches(tag: &str, needle: &str) -> bool {
    tag.to_ascii_lowercase().contains(needle)
}

fn tags_contain_any<'a>(tags: impl IntoIterator<Item = &'a String>, needles: &[&str]) -> bool {
    tags.into_iter()
        .any(|tag| needles.iter().any(|needle| tag_matches(tag, needle)))
}

fn mlx_vlm_tags(tags: &[String]) -> bool {
    tags_contain_any(
        tags,
        &[
            "image-text-to-text",
            "image-to-text",
            "multimodal",
            "vision-language",
            "vision_language",
            "vlm",
            "llava",
            "paligemma",
        ],
    ) || tags.iter().any(|tag| {
        let tag = tag.to_ascii_lowercase();
        tag == "vision" || tag.ends_with("-vision")
    })
}

fn mlx_lm_tags(tags: &[String]) -> bool {
    tags.iter().any(|tag| {
        let tag = tag.to_ascii_lowercase();
        tag.contains("mlx") || tag.contains("mlx-community")
    })
}

fn compatible(tags: &[String], engine: &str) -> bool {
    match engine {
        "llama.cpp" => tags.iter().any(|tag| {
            let tag = tag.to_ascii_lowercase();
            tag.contains("gguf") || tag.contains("llama.cpp")
        }),
        "mlx-lm" => mlx_lm_tags(tags) && !mlx_vlm_tags(tags),
        "mlx-vlm" => mlx_lm_tags(tags) && mlx_vlm_tags(tags),
        "whisper.cpp" => {
            tags_contain_any(
                tags,
                &[
                    "whisper",
                    "automatic-speech-recognition",
                    "speech-to-text",
                    "asr",
                ],
            ) || tags.iter().any(|tag| {
                let tag = tag.to_ascii_lowercase();
                tag.contains("whisper") || tag.contains("whisper.cpp")
            })
        }
        "streaming-asr" => {
            tags_contain_any(
                tags,
                &[
                    "automatic-speech-recognition",
                    "speech-to-text",
                    "asr",
                    "nemotron",
                ],
            ) || tags.iter().any(|tag| {
                let tag = tag.to_ascii_lowercase();
                tag.contains("nemotron")
                    || tag.contains("streaming")
                    || tag.contains("asr")
            })
        }
        "vllm" => tags.iter().any(|tag| {
            matches!(
                tag.as_str(),
                "safetensors" | "transformers" | "text-generation"
            )
        }),
        _ => true,
    }
}

pub async fn search(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    query: SearchQuery,
) -> anyhow::Result<Vec<HubModel>> {
    let limit = query.limit.unwrap_or(30).clamp(1, 100);
    let mut request = hf_auth::apply_auth(
        client
            .get("https://huggingface.co/api/models")
            .query(&[("limit", limit.to_string()), ("full", "true".to_owned())]),
        data_dir,
    );
    if let Some(q) = &query.q {
        request = request.query(&[("search", q)]);
    }
    let values: Vec<Value> = request
        .send()
        .await
        .context("contact Hugging Face")?
        .error_for_status()
        .context("Hugging Face search failed")?
        .json()
        .await
        .context("decode Hugging Face response")?;

    let mut models = values
        .into_iter()
        .filter_map(|value| {
            let id = value.get("id")?.as_str()?.to_owned();
            let author = value
                .get("author")
                .and_then(Value::as_str)
                .unwrap_or_else(|| id.split('/').next().unwrap_or(""))
                .to_owned();
            let tags = value
                .get("tags")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            if let Some(engine) = &query.engine
                && !compatible(&tags, engine)
            {
                return None;
            }
            let downloads = value.get("downloads").and_then(Value::as_u64).unwrap_or(0);
            let likes = value.get("likes").and_then(Value::as_u64).unwrap_or(0);
            let preferred_quantizer = author.eq_ignore_ascii_case("unsloth")
                || id.to_ascii_lowercase().contains("unsloth");
            let score = (downloads as f64 + 1.0).ln()
                + (likes as f64 + 1.0).ln()
                + if preferred_quantizer { 8.0 } else { 0.0 };
            Some(HubModel {
                id,
                author,
                downloads,
                likes,
                last_modified: value
                    .get("lastModified")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                tags,
                gated: value
                    .get("gated")
                    .is_some_and(|gated| gated == true || gated.as_str().is_some()),
                score,
                preferred_quantizer,
            })
        })
        .collect::<Vec<_>>();
    models.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    Ok(models)
}

#[derive(Debug, Clone, Serialize)]
pub struct RepoFile {
    pub path: String,
    pub size: Option<u64>,
}

/// List files in a Hugging Face model repository (tree API).
pub async fn list_repo_files(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<Vec<RepoFile>> {
    crate::models_store::validate_repo_id(repo_id)?;
    anyhow::ensure!(!revision.is_empty(), "revision is required");
    let url = format!("https://huggingface.co/api/models/{repo_id}/tree/{revision}");
    let values: Vec<Value> = hf_auth::apply_auth(client.get(url), data_dir)
        .send()
        .await
        .context("contact Hugging Face tree API")?
        .error_for_status()
        .context("Hugging Face tree request failed")?
        .json()
        .await
        .context("decode Hugging Face tree response")?;
    Ok(values
        .into_iter()
        .filter_map(|value| {
            let path = value.get("path")?.as_str()?.to_owned();
            let kind = value.get("type").and_then(Value::as_str).unwrap_or("file");
            if kind != "file" {
                return None;
            }
            let size = value.get("size").and_then(Value::as_u64);
            Some(RepoFile { path, size })
        })
        .collect())
}

/// List GGUF filenames for a repository and suggest a default quant.
pub async fn list_gguf_files(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    repo_id: &str,
    revision: Option<&str>,
) -> anyhow::Result<(Vec<RepoFile>, Option<String>)> {
    let revision = revision.unwrap_or("main");
    let files = list_repo_files(client, data_dir, repo_id, revision).await?;
    let ggufs: Vec<RepoFile> = files
        .into_iter()
        .filter(|file| file.path.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    let names: Vec<String> = ggufs
        .iter()
        .filter_map(|file| file.path.rsplit('/').next().map(ToOwned::to_owned))
        .collect();
    let preferred = prefer_gguf_filename(&names);
    Ok((ggufs, preferred))
}

fn is_mlx_snapshot_file(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".md")
        || lower.ends_with(".onnx")
        || lower.ends_with(".gguf")
        || lower.ends_with(".bin")
        || lower.ends_with(".pt")
        || lower.ends_with(".pth")
    {
        return false;
    }
    lower.ends_with(".safetensors")
        || lower.ends_with(".json")
        || lower.ends_with(".txt")
        || lower.ends_with(".model")
        || lower.ends_with(".tiktoken")
        || lower.ends_with("tokenizer.model")
}

/// List files needed for a local MLX snapshot download.
pub async fn list_mlx_snapshot_files(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<Vec<RepoFile>> {
    let files = list_repo_files(client, data_dir, repo_id, revision).await?;
    Ok(files
        .into_iter()
        .filter(|file| is_mlx_snapshot_file(&file.path))
        .collect())
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelTrust {
    pub repo_id: String,
    pub gated: bool,
    pub license: Option<String>,
    pub remote_code: bool,
    pub requires_acknowledgement: bool,
}

/// Hub metadata used for license and remote-code acknowledgement before download.
pub async fn model_trust(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    repo_id: &str,
) -> anyhow::Result<ModelTrust> {
    crate::models_store::validate_repo_id(repo_id)?;
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

    let gated = value
        .get("gated")
        .is_some_and(|gated| gated == &Value::Bool(true) || gated.as_str().is_some());
    let license = value
        .get("license")
        .or_else(|| value.pointer("/cardData/license"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            value
                .get("tags")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .find(|tag| tag.starts_with("license:"))
                .map(|tag| tag.trim_start_matches("license:").to_owned())
        });
    let tags = value
        .get("tags")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    let remote_code = tags.iter().any(|tag| {
        matches!(
            tag.as_str(),
            "transformers" | "pytorch" | "safetensors" | "custom_code"
        )
    });
    let requires_acknowledgement = gated || license.is_some() || remote_code;

    Ok(ModelTrust {
        repo_id: repo_id.to_owned(),
        gated,
        license,
        remote_code,
        requires_acknowledgement,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_compatibility_is_a_hard_filter() {
        assert!(compatible(&["gguf".into()], "llama.cpp"));
        assert!(!compatible(&["mlx".into()], "llama.cpp"));
        assert!(compatible(&["mlx".into(), "text-generation".into()], "mlx-lm"));
        assert!(!compatible(
            &["mlx".into(), "image-text-to-text".into()],
            "mlx-lm"
        ));
        assert!(compatible(
            &["mlx".into(), "image-text-to-text".into()],
            "mlx-vlm"
        ));
        assert!(!compatible(&["mlx".into(), "text-generation".into()], "mlx-vlm"));
    }
}

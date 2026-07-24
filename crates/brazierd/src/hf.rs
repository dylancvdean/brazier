use std::cmp::Ordering;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::hf_auth;
use crate::models_store::prefer_gguf_filename;

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
                tag.contains("nemotron") || tag.contains("streaming") || tag.contains("asr")
            })
        }
        "vllm" => tags.iter().any(|tag| {
            matches!(
                tag.as_str(),
                "safetensors" | "transformers" | "text-generation"
            )
        }),
        "stable-diffusion.cpp" => {
            tags_contain_any(
                tags,
                &[
                    "text-to-image",
                    "image-to-image",
                    "text-to-video",
                    "image-to-video",
                    "diffusers",
                ],
            ) || tags.iter().any(|tag| {
                let tag = tag.to_ascii_lowercase();
                tag.contains("stable-diffusion")
                    || tag.contains("flux")
                    || tag.contains("wan")
                    || tag.contains("ltx")
                    || tag.contains("sdxl")
                    || tag.contains("qwen-image")
            })
        }
        "personaplex" => {
            tags_contain_any(tags, &["audio-to-audio", "text-to-speech", "speech"])
                || tags.iter().any(|tag| {
                    let tag = tag.to_ascii_lowercase();
                    tag.contains("personaplex")
                        || tag.contains("moshi")
                        || tag.contains("speech-to-speech")
                })
        }
        _ => true,
    }
}

pub async fn search(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    query: SearchQuery,
) -> anyhow::Result<Vec<HubModel>> {
    let limit = query.limit.unwrap_or(30).clamp(1, 100);
    let trimmed_q = query
        .q
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let mut request = hf_auth::apply_auth(
        client
            .get("https://huggingface.co/api/models")
            .query(&[("limit", limit.to_string()), ("full", "true".to_owned())]),
        data_dir,
    );
    if let Some(q) = trimmed_q {
        request = request.query(&[("search", q)]);
    } else {
        // No query: surface trending models — HF's trendingScore blends
        // popularity and recency — pre-narrowed by a representative tag so the
        // engine filter below has compatible candidates to keep.
        request = request.query(&[("sort", "trendingScore"), ("direction", "-1")]);
        if let Some(engine) = &query.engine
            && let Some(tag) = engine_filter_tag(engine)
        {
            request = request.query(&[("filter", tag)]);
        }
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
    if trimmed_q.is_some() {
        // Explicit query: rank by our relevance/popularity score and return
        // everything the user might be looking for, quant repos included.
        models.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    } else {
        // Suggestions: keep Hugging Face's trendingScore order, but drop
        // near-zero-download noise and, for GGUF, single-quant repos so we
        // recommend model families the user can pick a quant from.
        let gguf = query.engine.as_deref() == Some("llama.cpp");
        models.retain(|model| {
            model.downloads >= MIN_SUGGESTED_DOWNLOADS
                && !(gguf && looks_like_single_quant(&model.id))
        });
    }
    Ok(models)
}

/// Minimum downloads before a model is trusted enough to suggest unprompted.
const MIN_SUGGESTED_DOWNLOADS: u64 = 100;

/// A single representative Hub tag used to pre-narrow trending results per engine.
fn engine_filter_tag(engine: &str) -> Option<&'static str> {
    match engine {
        "llama.cpp" => Some("gguf"),
        "mlx-lm" | "mlx-vlm" => Some("mlx"),
        "whisper.cpp" | "streaming-asr" => Some("automatic-speech-recognition"),
        "stable-diffusion.cpp" => Some("text-to-image"),
        "personaplex" => Some("text-to-speech"),
        _ => None,
    }
}

/// Whether a repo id names one specific quantization (e.g. `…-Q4_K_M-GGUF`)
/// rather than a family repo that holds many quant files to choose from.
fn looks_like_single_quant(repo_id: &str) -> bool {
    let name = repo_id
        .rsplit('/')
        .next()
        .unwrap_or(repo_id)
        .to_ascii_lowercase();
    const MARKERS: &[&str] = &[
        "q2_k", "q3_k", "q4_0", "q4_1", "q4_k", "q5_0", "q5_1", "q5_k", "q6_k", "q8_0", "iq1_",
        "iq2_", "iq3_", "iq4_", "-f16", "_f16", "-bf16", "_bf16", "-f32", "_f32",
    ];
    MARKERS.iter().any(|marker| name.contains(marker))
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

/// Exact sizes for specific files in a repository.
///
/// The tree API only lists one directory level, so nested component paths
/// (`split_files/vae/…`) need the paths-info endpoint. Resolving real sizes
/// up front also confirms every path exists before any bytes are fetched.
pub async fn paths_info(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    repo_id: &str,
    revision: &str,
    paths: &[String],
) -> anyhow::Result<Vec<RepoFile>> {
    crate::models_store::validate_repo_id(repo_id)?;
    anyhow::ensure!(!revision.is_empty(), "revision is required");
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    let url = format!("https://huggingface.co/api/models/{repo_id}/paths-info/{revision}");
    let values: Vec<Value> = hf_auth::apply_auth(client.post(url), data_dir)
        .json(&serde_json::json!({ "paths": paths }))
        .send()
        .await
        .context("contact Hugging Face paths-info API")?
        .error_for_status()
        .context("Hugging Face paths-info request failed")?
        .json()
        .await
        .context("decode Hugging Face paths-info response")?;
    Ok(values
        .into_iter()
        .filter_map(|value| {
            let path = value.get("path")?.as_str()?.to_owned();
            if value.get("type").and_then(Value::as_str).unwrap_or("file") != "file" {
                return None;
            }
            // LFS-backed files report the real size under `lfs`.
            let size = value
                .get("lfs")
                .and_then(|lfs| lfs.get("size"))
                .and_then(Value::as_u64)
                .or_else(|| value.get("size").and_then(Value::as_u64));
            Some(RepoFile { path, size })
        })
        .collect())
}

/// Fetch a short plain-text description for a model from its README.
pub async fn model_description(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    repo_id: &str,
) -> anyhow::Result<String> {
    crate::models_store::validate_repo_id(repo_id)?;
    let url = format!("https://huggingface.co/{repo_id}/raw/main/README.md");
    let response = hf_auth::apply_auth(client.get(url), data_dir)
        .send()
        .await
        .context("contact Hugging Face")?;
    if !response.status().is_success() {
        return Ok("This model has no README description on Hugging Face.".to_owned());
    }
    let body = response.text().await.context("read README")?;
    Ok(readme_summary(&body))
}

/// Reduce a model README to a short plain-text summary: drop YAML frontmatter,
/// headings, badges, tables, and code fences, then take the first paragraph.
fn readme_summary(readme: &str) -> String {
    let body = readme
        .strip_prefix("---")
        .and_then(|rest| rest.split_once("\n---").map(|(_, after)| after))
        .unwrap_or(readme);
    let mut out = String::new();
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !out.is_empty() {
                break;
            }
            continue;
        }
        if trimmed.starts_with('#')
            || trimmed.starts_with('!')
            || trimmed.starts_with('<')
            || trimmed.starts_with("[![")
            || trimmed.starts_with('|')
            || trimmed.starts_with("---")
            || trimmed.starts_with("```")
        {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(trimmed);
        if out.chars().count() >= 600 {
            break;
        }
    }
    let summary = out.trim();
    if summary.is_empty() {
        return "This model's README has no short description.".to_owned();
    }
    let mut result: String = summary.chars().take(600).collect();
    if summary.chars().count() > 600 {
        result.push('…');
    }
    result
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

/// Weights, configs, tokenizers, and voice bundles for PersonaPlex / Moshi snapshots.
///
/// Unlike MLX snapshots, these repos ship large `.safetensors` weights, SentencePiece
/// models, and compressed voice packs (`.tgz`) — not a pure JSON+safetensors layout.
fn is_personaplex_snapshot_file(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    if name.starts_with('.') || name == "license" {
        return false;
    }
    if lower.contains("/figures/")
        || lower.starts_with("figures/")
        || lower.ends_with(".md")
        || lower.ends_with(".png")
        || lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".gif")
        || lower.ends_with(".svg")
        || lower.ends_with(".ds_store")
    {
        return false;
    }
    lower.ends_with(".safetensors")
        || lower.ends_with(".pt")
        || lower.ends_with(".pth")
        || lower.ends_with(".bin")
        || lower.ends_with(".json")
        || lower.ends_with(".model")
        || lower.ends_with(".tiktoken")
        || lower.ends_with(".tgz")
        || lower.ends_with(".tar.gz")
        || lower.ends_with(".wav")
        || lower.ends_with(".npz")
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

/// List files needed for a PersonaPlex / Moshi snapshot download.
pub async fn list_personaplex_snapshot_files(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<Vec<RepoFile>> {
    let files = list_repo_files(client, data_dir, repo_id, revision).await?;
    Ok(files
        .into_iter()
        .filter(|file| is_personaplex_snapshot_file(&file.path))
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
    fn readme_summary_strips_frontmatter_and_headings() {
        let readme = "---\nlicense: apache-2.0\ntags:\n- text-generation\n---\n\n# My Model\n\n![badge](x.png)\n\nThis model is a fine-tune for helpful chat. It handles tools well.\n\nMore details below.";
        let summary = readme_summary(readme);
        assert!(summary.starts_with("This model is a fine-tune"));
        assert!(!summary.contains('#'));
        assert!(!summary.contains("license"));
    }

    #[test]
    fn single_quant_repos_are_detected() {
        assert!(looks_like_single_quant("bartowski/Qwen2.5-7B-Q4_K_M-GGUF"));
        assert!(looks_like_single_quant("someone/Model-IQ4_XS-GGUF"));
        assert!(looks_like_single_quant("author/Model-f16-GGUF"));
        // Family repos that hold many quants are kept.
        assert!(!looks_like_single_quant(
            "bartowski/Qwen2.5-7B-Instruct-GGUF"
        ));
        assert!(!looks_like_single_quant(
            "unsloth/Llama-3.3-70B-Instruct-GGUF"
        ));
    }

    #[test]
    fn personaplex_snapshot_keeps_weights_voices_and_tokenizers() {
        assert!(is_personaplex_snapshot_file("model.safetensors"));
        assert!(is_personaplex_snapshot_file("config.json"));
        assert!(is_personaplex_snapshot_file("tokenizer_spm_32k_3.model"));
        assert!(is_personaplex_snapshot_file("voices.tgz"));
        assert!(is_personaplex_snapshot_file("dist.tgz"));
        assert!(!is_personaplex_snapshot_file("README.md"));
        assert!(!is_personaplex_snapshot_file("figures/results.png"));
        // MLX filter must not be used for PersonaPlex: it drops .tgz voice packs.
        assert!(!is_mlx_snapshot_file("voices.tgz"));
    }

    #[test]
    fn engine_compatibility_is_a_hard_filter() {
        assert!(compatible(&["gguf".into()], "llama.cpp"));
        assert!(!compatible(&["mlx".into()], "llama.cpp"));
        assert!(compatible(
            &["mlx".into(), "text-generation".into()],
            "mlx-lm"
        ));
        assert!(!compatible(
            &["mlx".into(), "image-text-to-text".into()],
            "mlx-lm"
        ));
        assert!(compatible(
            &["mlx".into(), "image-text-to-text".into()],
            "mlx-vlm"
        ));
        assert!(!compatible(
            &["mlx".into(), "text-generation".into()],
            "mlx-vlm"
        ));
    }
}

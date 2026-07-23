use std::cmp::Ordering;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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

fn compatible(tags: &[String], engine: &str) -> bool {
    match engine {
        "llama.cpp" => tags.iter().any(|tag| {
            let tag = tag.to_ascii_lowercase();
            tag.contains("gguf") || tag.contains("llama.cpp")
        }),
        "mlx-lm" | "mlx-vlm" => tags.iter().any(|tag| {
            let tag = tag.to_ascii_lowercase();
            tag.contains("mlx") || tag.contains("mlx-community")
        }),
        "vllm" => tags.iter().any(|tag| {
            matches!(
                tag.as_str(),
                "safetensors" | "transformers" | "text-generation"
            )
        }),
        _ => true,
    }
}

pub async fn search(client: &reqwest::Client, query: SearchQuery) -> anyhow::Result<Vec<HubModel>> {
    let limit = query.limit.unwrap_or(30).clamp(1, 100);
    let mut request = client
        .get("https://huggingface.co/api/models")
        .query(&[("limit", limit.to_string()), ("full", "true".to_owned())]);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_compatibility_is_a_hard_filter() {
        assert!(compatible(&["gguf".into()], "llama.cpp"));
        assert!(!compatible(&["mlx".into()], "llama.cpp"));
        assert!(compatible(&["mlx".into()], "mlx-lm"));
    }
}

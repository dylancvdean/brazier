use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

const LLAMA_CPP: &str = include_str!("../../../engine-recipes/llama.cpp.json");
const MLX_LM: &str = include_str!("../../../engine-recipes/mlx-lm.json");
const MLX_VLM: &str = include_str!("../../../engine-recipes/mlx-vlm.json");
const VLLM: &str = include_str!("../../../engine-recipes/vllm.json");
const WHISPER_CPP: &str = include_str!("../../../engine-recipes/whisper.cpp.json");
const MLX_LM_LOCK: &str = include_str!("../../../engine-recipes/mlx-lm.lock");
const MLX_VLM_LOCK: &str = include_str!("../../../engine-recipes/mlx-vlm.lock");

pub fn is_python_engine(engine: &str) -> bool {
    matches!(engine, "mlx-lm" | "mlx-vlm" | "vllm")
}

/// Directory containing shipped lock files for Python engine builds.
pub fn recipe_root(data_dir: &Path) -> PathBuf {
    data_dir.join("engine-recipes")
}

/// Write bundled recipe lock files into the application data directory.
pub fn ensure_recipe_files(data_dir: &Path) -> anyhow::Result<PathBuf> {
    let dir = recipe_root(data_dir);
    std::fs::create_dir_all(&dir).context("create engine-recipes directory")?;
    std::fs::write(dir.join("mlx-lm.lock"), MLX_LM_LOCK).context("write mlx-lm.lock")?;
    std::fs::write(dir.join("mlx-vlm.lock"), MLX_VLM_LOCK).context("write mlx-vlm.lock")?;
    Ok(dir)
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuildRecipe {
    pub id: String,
    pub display_name: String,
    pub upstream_origins: Vec<String>,
    pub supported_platforms: Vec<String>,
    pub steps: Vec<RecipeStep>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RecipeStep {
    pub label: String,
    pub program: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuildPlanRequest {
    pub engine: String,
    pub repository: String,
    pub revision: String,
    pub platform: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuildPlan {
    pub engine: String,
    pub display_name: String,
    pub repository: String,
    pub revision: String,
    pub platform: String,
    pub trusted_origin: bool,
    pub warning: Option<String>,
    pub checkout: Vec<PlannedCommand>,
    pub build: Vec<PlannedCommand>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlannedCommand {
    pub label: String,
    pub program: String,
    pub args: Vec<String>,
}

pub fn recipe(engine: &str) -> anyhow::Result<BuildRecipe> {
    let source = match engine {
        "llama.cpp" => LLAMA_CPP,
        "mlx-lm" => MLX_LM,
        "mlx-vlm" => MLX_VLM,
        "vllm" => VLLM,
        "whisper.cpp" => WHISPER_CPP,
        _ => anyhow::bail!("unsupported engine recipe: {engine}"),
    };
    Ok(serde_json::from_str(source)?)
}

fn valid_repository(repository: &str) -> bool {
    (repository.starts_with("https://")
        || repository.starts_with("ssh://")
        || repository.starts_with("git@"))
        && !repository.chars().any(char::is_control)
}

fn valid_revision(revision: &str) -> bool {
    !revision.is_empty()
        && revision.len() <= 200
        && !revision.starts_with('-')
        && revision
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "/-_.+".contains(character))
}

pub fn plan(request: BuildPlanRequest) -> anyhow::Result<BuildPlan> {
    anyhow::ensure!(
        valid_repository(&request.repository),
        "repository must be an HTTPS or SSH Git URL"
    );
    anyhow::ensure!(
        valid_revision(&request.revision),
        "revision contains unsupported characters"
    );
    let recipe = recipe(&request.engine)?;
    anyhow::ensure!(
        recipe
            .supported_platforms
            .iter()
            .any(|platform| platform == &request.platform),
        "{} is not supported on {}",
        recipe.display_name,
        request.platform
    );
    let normalized_repository = request.repository.trim_end_matches('/');
    let trusted_origin = recipe
        .upstream_origins
        .iter()
        .any(|origin| origin.trim_end_matches('/') == normalized_repository);
    let warning = (!trusted_origin).then(|| {
        format!(
            "{} is not a whitelisted upstream. Building and running this fork executes untrusted native code.",
            request.repository
        )
    });
    let checkout = vec![
        PlannedCommand {
            label: "Clone source without running hooks".to_owned(),
            program: "git".to_owned(),
            args: vec![
                "-c".into(),
                "core.hooksPath=".into(),
                "clone".into(),
                "--no-checkout".into(),
                "--filter=blob:none".into(),
                request.repository.clone(),
                "{source}".into(),
            ],
        },
        PlannedCommand {
            label: "Checkout selected revision".to_owned(),
            program: "git".to_owned(),
            args: vec![
                "-c".into(),
                "core.hooksPath=".into(),
                "-C".into(),
                "{source}".into(),
                "checkout".into(),
                "--detach".into(),
                request.revision.clone(),
                "--".into(),
            ],
        },
    ];
    let build = recipe
        .steps
        .into_iter()
        .map(|step| PlannedCommand {
            label: step.label,
            program: step.program,
            args: step.args,
        })
        .collect();
    Ok(BuildPlan {
        engine: recipe.id,
        display_name: recipe.display_name,
        repository: request.repository,
        revision: request.revision,
        platform: request.platform,
        trusted_origin,
        warning,
        checkout,
        build,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_any_fork_but_warns_when_it_is_not_whitelisted() {
        let plan = plan(BuildPlanRequest {
            engine: "llama.cpp".into(),
            repository: "https://github.com/example/llama.cpp".into(),
            revision: "feature/new-model".into(),
            platform: "linux-x64".into(),
        })
        .unwrap();
        assert!(!plan.trusted_origin);
        assert!(plan.warning.is_some());
        assert!(plan.build.iter().all(|step| step.program != "sh"));
    }

    #[test]
    fn recognizes_the_official_upstream() {
        let plan = plan(BuildPlanRequest {
            engine: "llama.cpp".into(),
            repository: "https://github.com/ggml-org/llama.cpp.git".into(),
            revision: "main".into(),
            platform: "macos-arm64".into(),
        })
        .unwrap();
        assert!(plan.trusted_origin);
        assert!(plan.warning.is_none());
    }

    #[test]
    fn rejects_option_injection_in_revision() {
        let result = plan(BuildPlanRequest {
            engine: "llama.cpp".into(),
            repository: "https://github.com/example/llama.cpp".into(),
            revision: "--upload-pack=bad".into(),
            platform: "linux-x64".into(),
        });
        assert!(result.is_err());
    }
}

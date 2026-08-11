use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::Context;
use serde::{Deserialize, Serialize};

const LLAMA_CPP: &str = include_str!("../../../engine-recipes/llama.cpp.json");
const MLX_LM: &str = include_str!("../../../engine-recipes/mlx-lm.json");
const MLX_VLM: &str = include_str!("../../../engine-recipes/mlx-vlm.json");
const VLLM: &str = include_str!("../../../engine-recipes/vllm.json");
const WHISPER_CPP: &str = include_str!("../../../engine-recipes/whisper.cpp.json");
const WHISPERKIT: &str = include_str!("../../../engine-recipes/whisperkit.json");
const STREAMING_ASR: &str = include_str!("../../../engine-recipes/streaming-asr.json");
const SDCPP: &str = include_str!("../../../engine-recipes/stable-diffusion.cpp.json");
const PERSONAPLEX: &str = include_str!("../../../engine-recipes/personaplex.json");
const PERSONAPLEX_MLX: &str = include_str!("../../../engine-recipes/personaplex-mlx.json");
const STREAMING_ASR_PYPROJECT: &str = include_str!("../python/streaming_asr_pkg/pyproject.toml");
const STREAMING_ASR_INIT: &str =
    include_str!("../python/streaming_asr_pkg/brazier_streaming_asr/__init__.py");
const STREAMING_ASR_MAIN: &str =
    include_str!("../python/streaming_asr_pkg/brazier_streaming_asr/__main__.py");

pub fn is_python_engine(engine: &str) -> bool {
    matches!(
        engine,
        "mlx-lm" | "mlx-vlm" | "vllm" | "streaming-asr" | "personaplex" | "personaplex-mlx"
    )
}

/// Swift Package Manager engines (Apple Silicon CLI tools).
pub fn is_swift_engine(engine: &str) -> bool {
    matches!(engine, "whisperkit")
}

/// Directory containing the bundled streaming ASR worker package.
pub fn recipe_root(data_dir: &Path) -> PathBuf {
    data_dir.join("engine-recipes")
}

/// Write bundled Python packages into the data directory.
pub fn ensure_recipe_files(data_dir: &Path) -> anyhow::Result<PathBuf> {
    let dir = recipe_root(data_dir);
    std::fs::create_dir_all(&dir).context("create engine-recipes directory")?;

    let pkg = dir.join("streaming_asr_pkg");
    let module = pkg.join("brazier_streaming_asr");
    std::fs::create_dir_all(&module).context("create streaming_asr_pkg")?;
    std::fs::write(pkg.join("pyproject.toml"), STREAMING_ASR_PYPROJECT)
        .context("write streaming_asr pyproject")?;
    std::fs::write(module.join("__init__.py"), STREAMING_ASR_INIT)
        .context("write streaming_asr __init__")?;
    std::fs::write(module.join("__main__.py"), STREAMING_ASR_MAIN)
        .context("write streaming_asr __main__")?;
    Ok(dir)
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuildRecipe {
    pub id: String,
    pub display_name: String,
    pub upstream_origins: Vec<String>,
    pub supported_platforms: Vec<String>,
    #[serde(default)]
    pub skip_checkout: bool,
    /// Reviewed commands required only on a particular platform, run after the
    /// environment exists and before the selected source is installed.
    #[serde(default)]
    pub platform_pre_steps: HashMap<String, Vec<RecipeStep>>,
    /// Build steps selected by acceleration target. Recipes which define this
    /// map must provide an entry for every target they accept.
    #[serde(default)]
    pub target_steps: HashMap<String, Vec<RecipeStep>>,
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
    pub target: String,
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
    #[serde(skip)]
    pub skip_checkout: bool,
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
        "whisperkit" => WHISPERKIT,
        "streaming-asr" => STREAMING_ASR,
        "stable-diffusion.cpp" => SDCPP,
        "personaplex" => PERSONAPLEX,
        "personaplex-mlx" => PERSONAPLEX_MLX,
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
    revision.is_empty()
        || (revision.len() <= 200
            && !revision.starts_with('-')
            && revision
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || "/-_.+".contains(character)))
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
    let mut recipe = recipe(&request.engine)?;
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
    let checkout = if recipe.skip_checkout {
        Vec::new()
    } else {
        let mut checkout = vec![PlannedCommand {
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
        }];
        // `--no-checkout` keeps clone hooks from touching the source tree, but
        // also leaves the worktree empty. Populate it with either the requested
        // revision or the clone's default HEAD before configuring a build.
        checkout.push(PlannedCommand {
            label: if request.revision.is_empty() {
                "Checkout repository default revision".to_owned()
            } else {
                "Checkout selected revision".to_owned()
            },
            program: "git".to_owned(),
            args: vec![
                "-c".into(),
                "core.hooksPath=".into(),
                "-C".into(),
                "{source}".into(),
                "checkout".into(),
                "--detach".into(),
                if request.revision.is_empty() {
                    "HEAD".into()
                } else {
                    request.revision.clone()
                },
                "--".into(),
            ],
        });
        checkout.push(PlannedCommand {
            label: "Initialize source submodules".to_owned(),
            program: "git".to_owned(),
            args: vec![
                "-c".into(),
                "core.hooksPath=".into(),
                "-C".into(),
                "{source}".into(),
                "submodule".into(),
                "update".into(),
                "--init".into(),
                "--recursive".into(),
            ],
        });
        checkout
    };
    let skip_checkout = recipe.skip_checkout;
    let target_steps = if recipe.target_steps.is_empty() {
        Vec::new()
    } else {
        recipe
            .target_steps
            .remove(&request.target)
            .with_context(|| {
                format!(
                    "{} has no build steps for target {}",
                    recipe.display_name, request.target
                )
            })?
    };
    let build = recipe
        .steps
        .iter()
        .take(1)
        .cloned()
        .chain(
            recipe
                .platform_pre_steps
                .remove(&request.platform)
                .unwrap_or_default(),
        )
        .chain(recipe.steps.iter().skip(1).cloned())
        .chain(target_steps)
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
        skip_checkout,
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
            target: "cpu".into(),
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
            target: "cpu".into(),
        })
        .unwrap();
        assert!(plan.trusted_origin);
        assert!(plan.warning.is_none());
    }

    #[test]
    fn initializes_submodules_after_checking_out_the_requested_revision() {
        let plan = plan(BuildPlanRequest {
            engine: "stable-diffusion.cpp".into(),
            repository: "https://github.com/leejet/stable-diffusion.cpp.git".into(),
            revision: "master".into(),
            platform: "linux-x64".into(),
            target: "cpu".into(),
        })
        .unwrap();

        assert_eq!(
            plan.checkout
                .iter()
                .map(|step| step.label.as_str())
                .collect::<Vec<_>>(),
            vec![
                "Clone source without running hooks",
                "Checkout selected revision",
                "Initialize source submodules",
            ]
        );
        let submodules = &plan.checkout[2];
        assert_eq!(submodules.program, "git");
        assert_eq!(
            submodules.args,
            [
                "-c",
                "core.hooksPath=",
                "-C",
                "{source}",
                "submodule",
                "update",
                "--init",
                "--recursive",
            ]
        );
    }

    #[test]
    fn rejects_option_injection_in_revision() {
        let result = plan(BuildPlanRequest {
            engine: "llama.cpp".into(),
            repository: "https://github.com/example/llama.cpp".into(),
            revision: "--upload-pack=bad".into(),
            platform: "linux-x64".into(),
            target: "cpu".into(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn empty_revision_uses_the_repository_default_branch() {
        let plan = plan(BuildPlanRequest {
            engine: "llama.cpp".into(),
            repository: "https://github.com/example/llama.cpp".into(),
            revision: String::new(),
            platform: "linux-x64".into(),
            target: "cpu".into(),
        })
        .unwrap();
        assert_eq!(
            plan.checkout
                .iter()
                .map(|step| step.label.as_str())
                .collect::<Vec<_>>(),
            vec![
                "Clone source without running hooks",
                "Checkout repository default revision",
                "Initialize source submodules"
            ]
        );
        assert_eq!(plan.checkout[1].args[6], "HEAD");
    }

    #[test]
    fn streaming_asr_skips_git_checkout() {
        let plan = plan(BuildPlanRequest {
            engine: "streaming-asr".into(),
            repository: "https://github.com/huggingface/transformers.git".into(),
            revision: "bundled".into(),
            platform: "macos-arm64".into(),
            target: "cpu".into(),
        })
        .unwrap();
        assert!(plan.skip_checkout);
        assert!(plan.checkout.is_empty());
        assert!(plan.trusted_origin);
        assert!(
            plan.build
                .iter()
                .any(|step| step.label.contains("streaming ASR"))
        );
    }

    #[test]
    fn vllm_uses_target_specific_requirements_and_non_isolated_builds() {
        for target in ["cpu", "cuda", "rocm"] {
            let plan = plan(BuildPlanRequest {
                engine: "vllm".into(),
                repository: "https://github.com/vllm-project/vllm".into(),
                revision: "main".into(),
                platform: "linux-x64".into(),
                target: target.into(),
            })
            .unwrap();
            let args = plan
                .build
                .iter()
                .flat_map(|step| step.args.iter())
                .cloned()
                .collect::<Vec<_>>();
            assert!(args.contains(&format!("{{source}}/requirements/build/{target}.txt")));
            assert!(args.contains(&format!("{{source}}/requirements/{target}.txt")));
            assert!(args.contains(&"--no-build-isolation".into()));
            assert!(args.contains(&"--no-deps".into()));
        }
    }

    #[test]
    fn vllm_rocm_considers_pypi_when_the_pytorch_index_has_an_old_package() {
        let plan = plan(BuildPlanRequest {
            engine: "vllm".into(),
            repository: "https://github.com/vllm-project/vllm".into(),
            revision: "main".into(),
            platform: "linux-x64".into(),
            target: "rocm".into(),
        })
        .unwrap();
        let dependency_step = plan
            .build
            .iter()
            .find(|step| step.label.contains("ROCm build and runtime dependencies"))
            .unwrap();
        assert!(
            dependency_step
                .args
                .windows(2)
                .any(|args| { args == ["--index-strategy", "unsafe-best-match"] })
        );
    }

    #[test]
    fn vllm_metal_builds_plugin_artifacts_from_the_selected_source() {
        let plan = plan(BuildPlanRequest {
            engine: "vllm".into(),
            repository: "https://github.com/vllm-project/vllm-metal".into(),
            revision: "main".into(),
            platform: "macos-arm64".into(),
            target: "metal".into(),
        })
        .unwrap();
        let labels = plan
            .build
            .iter()
            .map(|step| step.label.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            vec![
                "Create isolated environment",
                "Build and install selected vLLM-Metal source",
                "Compile vLLM-Metal native artifacts",
            ]
        );
        assert_eq!(plan.build[2].program, "{python}");
    }

    #[test]
    fn ensure_recipe_files_writes_streaming_asr_package() {
        let dir = tempfile::tempdir().unwrap();
        let root = ensure_recipe_files(dir.path()).unwrap();
        assert!(
            root.join("streaming_asr_pkg/brazier_streaming_asr/__main__.py")
                .is_file()
        );
    }
}

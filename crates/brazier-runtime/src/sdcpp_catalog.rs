//! Curated stable-diffusion.cpp model bundles.
//!
//! sd-cli needs more than a checkpoint: Flux and Wan expect their VAE and text
//! encoders as separate files, passed through their own flags. Downloading a
//! diffusion model on its own therefore produces something that cannot
//! generate anything, so installs are described as *bundles* — every file the
//! model needs, where it lives on the Hub, and the sd-cli flag it binds to.
//! Installing one writes the `manifest.json` that [`crate::sdcpp`] reads.

use std::{collections::BTreeMap, path::Path, sync::OnceLock};

use anyhow::Context;

use serde::{Deserialize, Serialize};

use crate::sdcpp::{self, Modality};

const BUNDLES: &str = include_str!("../../../model-recipes/sdcpp-bundles.json");

/// One interchangeable file for a component: a quantisation of the same model.
///
/// The trade-off is the familiar one from GGUF language models — a smaller
/// quant fits a smaller machine and looks slightly worse — so it is offered the
/// same way, as a choice made at download time rather than a separate recipe
/// per size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    /// Short label for the picker, e.g. `Q4_K_M`.
    pub label: String,
    /// Path inside the repository. The component's `repo_id` is used unless
    /// this names its own.
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo_id: Option<String>,
    /// Override the sd-cli argument used for this alternative. This lets a
    /// bundle offer, for example, a compact TAE decoder (`--tae`) alongside
    /// the full-quality VAE (`--vae`) without duplicating its large model
    /// weights.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flag: Option<String>,
    /// Switch this file within the same model directory instead of creating a
    /// separate variant install. Decoders use this because their large
    /// diffusion model and text encoder are shared.
    #[serde(default)]
    pub in_place: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approx_bytes: Option<u64>,
    /// One-line note on what this size costs or buys.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// One file within a bundle, and the sd-cli flag it is passed as.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Component {
    pub repo_id: String,
    /// Path inside the Hub repository. When `variants` is non-empty this is the
    /// default choice, and the interface may substitute another.
    pub path: String,
    /// sd-cli flag without the leading `--` (`diffusion-model`, `vae`,
    /// `clip_l`, `t5xxl`, …). Absent for a self-contained checkpoint, which is
    /// passed to `-m` instead.
    #[serde(default)]
    pub flag: Option<String>,
    /// Human label for the install list, e.g. "Text encoder (T5-XXL)".
    pub role: String,
    /// Whether the source repo requires accepting terms plus a Hub token.
    #[serde(default)]
    pub gated: bool,
    /// Rough size for the pre-download summary; not used for verification.
    #[serde(default)]
    pub approx_bytes: Option<u64>,
    /// Sizes this component is offered in. Empty means the one file above.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variants: Vec<Variant>,
}

impl Component {
    /// Filename this component is stored under inside the model directory.
    pub fn file_name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }
}

/// Generation settings that suit a bundle, used to prefill the Generate panel.
///
/// These matter: distilled models like Flux schnell need CFG 1.0, and running
/// them at sd-cli's default of 7.0 doubles the work and degrades the image.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationDefaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flow_shift: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_frames: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bundle {
    pub id: String,
    pub label: String,
    pub modality: Modality,
    /// Install directory under `models/sdcpp/{modality}/`, `owner/name` style.
    pub key: String,
    pub summary: String,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub defaults: GenerationDefaults,
    /// Whether the model can animate or restyle a supplied image.
    ///
    /// For video this is the text-to-video / image-to-video split, which
    /// decides whether a photo can be handed to it at all.
    #[serde(default)]
    pub supports_init_image: bool,
    /// Shown in the short list rather than behind "show every model". A few
    /// good defaults are more useful than a wall of near-identical choices.
    #[serde(default)]
    pub featured: bool,
    pub components: Vec<Component>,
}

impl Bundle {
    /// Model id this bundle installs as, e.g. `sdcpp-image:owner/name`.
    pub fn model_id(&self) -> String {
        sdcpp::model_id_for_key(self.modality, &self.key)
    }

    /// Whether any component needs a Hugging Face token.
    pub fn gated(&self) -> bool {
        self.components.iter().any(|component| component.gated)
    }

    pub fn approx_bytes(&self) -> Option<u64> {
        self.components.iter().try_fold(0_u64, |total, component| {
            component
                .approx_bytes
                .and_then(|size| total.checked_add(size))
        })
    }

    /// The manifest that lets sd-cli find every downloaded component.
    pub fn manifest(&self) -> sdcpp::SdcppManifest {
        let mut args = BTreeMap::new();
        let mut single_file = None;
        for component in &self.components {
            match &component.flag {
                Some(flag) => {
                    args.insert(flag.clone(), component.file_name().to_owned());
                }
                None => single_file = Some(component.file_name().to_owned()),
            }
        }
        sdcpp::SdcppManifest {
            modality: self.modality,
            args,
            single_file,
            supports_init_image: self.supports_init_image,
        }
    }

    /// Directory this bundle installs into.
    pub fn install_dir(&self, data_dir: &Path) -> anyhow::Result<std::path::PathBuf> {
        sdcpp::model_dir_for_key(data_dir, self.modality, &self.key)
    }

    /// Whether a usable install already exists on disk.
    pub fn installed(&self, data_dir: &Path) -> bool {
        self.install_dir(data_dir)
            .ok()
            .and_then(|dir| sdcpp::load_manifest(&dir).ok())
            .is_some_and(|manifest| {
                manifest.modality == self.modality
                    && manifest.args == self.manifest().args
                    && manifest.single_file == self.manifest().single_file
            })
    }
}

#[derive(Debug, Deserialize)]
struct Catalog {
    bundles: Vec<Bundle>,
}

/// Where a bundle came from, so the UI can distinguish shipped from user-made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Origin {
    /// Shipped with the app and not editable.
    Builtin,
    /// Assembled or hand-written locally, stored under the data directory.
    Custom,
}

/// A bundle together with where it came from.
#[derive(Debug, Clone)]
pub struct CatalogEntry {
    pub bundle: Bundle,
    pub origin: Origin,
}

/// Bundles shipped with the app, parsed once.
pub fn builtin_bundles() -> &'static [Bundle] {
    static PARSED: OnceLock<Vec<Bundle>> = OnceLock::new();
    PARSED
        .get_or_init(|| {
            serde_json::from_str::<Catalog>(BUNDLES)
                .expect("bundled sdcpp catalog is valid JSON")
                .bundles
        })
        .as_slice()
}

/// File holding locally defined bundles.
pub fn custom_catalog_path(data_dir: &Path) -> std::path::PathBuf {
    data_dir.join("model-recipes").join("sdcpp-custom.json")
}

/// Locally defined bundles. A malformed file is ignored rather than fatal, so
/// a bad hand edit cannot take the model list down.
pub fn custom_bundles(data_dir: &Path) -> Vec<Bundle> {
    std::fs::read(custom_catalog_path(data_dir))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Catalog>(&bytes).ok())
        .map(|catalog| catalog.bundles)
        .unwrap_or_default()
}

fn write_custom_bundles(data_dir: &Path, bundles: &[Bundle]) -> anyhow::Result<()> {
    let path = custom_catalog_path(data_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let payload = serde_json::to_vec_pretty(&serde_json::json!({ "bundles": bundles }))
        .context("serialize custom bundles")?;
    std::fs::write(&path, payload).with_context(|| format!("write {}", path.display()))
}

/// Every bundle on offer: shipped ones first, then local ones. A custom bundle
/// may shadow a builtin with the same id, which is how a shipped definition
/// gets corrected without waiting for a release.
pub fn catalog(data_dir: &Path) -> Vec<CatalogEntry> {
    let custom = custom_bundles(data_dir);
    let shadowed: std::collections::HashSet<&str> =
        custom.iter().map(|bundle| bundle.id.as_str()).collect();
    let mut entries: Vec<CatalogEntry> = builtin_bundles()
        .iter()
        .filter(|bundle| !shadowed.contains(bundle.id.as_str()))
        .map(|bundle| CatalogEntry {
            bundle: bundle.clone(),
            origin: Origin::Builtin,
        })
        .collect();
    entries.extend(custom.into_iter().map(|bundle| CatalogEntry {
        bundle,
        origin: Origin::Custom,
    }));
    entries
}

pub fn find(data_dir: &Path, id: &str) -> Option<Bundle> {
    catalog(data_dir)
        .into_iter()
        .find(|entry| entry.bundle.id == id)
        .map(|entry| entry.bundle)
}

/// Validate a bundle before it is stored or installed.
///
/// Paths become filenames on disk and flags become command-line arguments, so
/// both are checked here rather than at spawn time.
pub fn validate(bundle: &Bundle) -> anyhow::Result<()> {
    anyhow::ensure!(
        !bundle.id.is_empty()
            && bundle.id.len() <= 120
            && bundle
                .id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')),
        "bundle id must be alphanumeric with dashes, dots, or underscores"
    );
    anyhow::ensure!(!bundle.label.trim().is_empty(), "bundle needs a label");
    sdcpp::model_dir_for_key(Path::new("/"), bundle.modality, &bundle.key)
        .context("invalid bundle key")?;
    anyhow::ensure!(
        !bundle.components.is_empty(),
        "bundle needs at least one file"
    );
    let checkpoints = bundle
        .components
        .iter()
        .filter(|component| component.flag.is_none())
        .count();
    anyhow::ensure!(
        checkpoints <= 1,
        "a bundle can only have one self-contained checkpoint"
    );
    let models = checkpoints
        + bundle
            .components
            .iter()
            .filter(|component| component.flag.as_deref() == Some("diffusion-model"))
            .count();
    anyhow::ensure!(
        models == 1,
        "a bundle needs exactly one checkpoint or diffusion model"
    );
    /// Paths become filenames on disk, so every one is checked — including the
    /// alternatives a quant picker can substitute for the default.
    fn valid_path(path: &str) -> bool {
        !path.is_empty()
            && !path.starts_with('/')
            && !path.contains('\\')
            && !path
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
    }

    let mut seen_files = std::collections::HashSet::new();
    for component in &bundle.components {
        crate::models_store::validate_repo_id(&component.repo_id)
            .with_context(|| format!("invalid repository `{}`", component.repo_id))?;
        anyhow::ensure!(
            valid_path(&component.path),
            "invalid file path `{}`",
            component.path
        );
        for variant in &component.variants {
            anyhow::ensure!(
                !variant.label.trim().is_empty(),
                "a size option needs a label"
            );
            anyhow::ensure!(
                valid_path(&variant.path),
                "invalid file path `{}`",
                variant.path
            );
            if let Some(repo_id) = &variant.repo_id {
                crate::models_store::validate_repo_id(repo_id)
                    .with_context(|| format!("invalid repository `{repo_id}`"))?;
            }
            if let Some(flag) = &variant.flag {
                anyhow::ensure!(
                    !flag.is_empty()
                        && flag.len() <= 40
                        && flag
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_')),
                    "invalid sd-cli flag `{flag}`"
                );
            }
            sdcpp::component_destination(
                Path::new("/"),
                bundle.modality,
                &bundle.key,
                variant.path.rsplit('/').next().unwrap_or(&variant.path),
            )
            .with_context(|| format!("invalid file name in `{}`", variant.path))?;
        }
        sdcpp::component_destination(
            Path::new("/"),
            bundle.modality,
            &bundle.key,
            component.file_name(),
        )
        .with_context(|| format!("invalid file name in `{}`", component.path))?;
        if let Some(flag) = &component.flag {
            anyhow::ensure!(
                !flag.is_empty()
                    && flag.len() <= 40
                    && flag
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_')),
                "invalid sd-cli flag `{flag}`"
            );
        }
        anyhow::ensure!(
            seen_files.insert(component.file_name().to_owned()),
            "two files would be saved as `{}`; rename one",
            component.file_name()
        );
    }
    Ok(())
}

/// Store a locally defined bundle, replacing any earlier one with the same id.
pub fn save_custom(data_dir: &Path, bundle: Bundle) -> anyhow::Result<Bundle> {
    validate(&bundle)?;
    let mut bundles = custom_bundles(data_dir);
    match bundles.iter_mut().find(|existing| existing.id == bundle.id) {
        Some(existing) => *existing = bundle.clone(),
        None => bundles.push(bundle.clone()),
    }
    write_custom_bundles(data_dir, &bundles)?;
    Ok(bundle)
}

/// Forget a locally defined bundle. Downloaded files are left in place.
pub fn delete_custom(data_dir: &Path, id: &str) -> anyhow::Result<()> {
    let mut bundles = custom_bundles(data_dir);
    let before = bundles.len();
    bundles.retain(|bundle| bundle.id != id);
    anyhow::ensure!(bundles.len() < before, "no custom bundle `{id}`");
    write_custom_bundles(data_dir, &bundles)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builtin(id: &str) -> Option<&'static Bundle> {
        builtin_bundles().iter().find(|bundle| bundle.id == id)
    }

    #[test]
    fn catalog_parses_and_is_internally_consistent() {
        let bundles = builtin_bundles();
        assert!(!bundles.is_empty());
        for bundle in bundles {
            assert!(!bundle.components.is_empty(), "{} has no files", bundle.id);
            // Every bundle needs exactly one thing to pass as the model: either
            // a self-contained checkpoint or a standalone diffusion model.
            let checkpoints = bundle
                .components
                .iter()
                .filter(|component| component.flag.is_none())
                .count();
            let diffusion = bundle
                .components
                .iter()
                .filter(|component| component.flag.as_deref() == Some("diffusion-model"))
                .count();
            assert_eq!(
                checkpoints + diffusion,
                1,
                "{} must name exactly one model file",
                bundle.id
            );
            assert!(
                bundle.model_id().starts_with(match bundle.modality {
                    Modality::Image => "sdcpp-image:",
                    Modality::Video => "sdcpp-video:",
                }),
                "{} has a mismatched model id",
                bundle.id
            );
        }
    }

    #[test]
    fn multi_component_bundles_produce_flag_manifests() {
        let flux = builtin("flux1-schnell").expect("flux bundle");
        let manifest = flux.manifest();
        assert_eq!(manifest.single_file, None);
        assert!(
            manifest
                .args
                .get("diffusion-model")
                .is_some_and(|name| name.starts_with("flux1-schnell-") && name.ends_with(".gguf")),
            "the manifest names whichever quant was chosen"
        );
        // The whole point of bundles: encoders arrive with the model.
        assert_eq!(
            manifest.args.get("clip_l").map(String::as_str),
            Some("clip_l.safetensors")
        );
        assert_eq!(
            manifest.args.get("t5xxl").map(String::as_str),
            Some("t5xxl_fp16.safetensors")
        );
        assert!(flux.gated(), "flux VAE comes from a gated repo");
    }

    #[test]
    fn qwen_image_uses_the_supported_gguf_text_encoder() {
        let qwen = builtin("qwen-image").expect("Qwen Image bundle");
        let llm = qwen
            .components
            .iter()
            .find(|component| component.flag.as_deref() == Some("llm"))
            .expect("Qwen2.5-VL text encoder");

        assert_eq!(llm.repo_id, "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF");
        assert_eq!(llm.path, "Qwen2.5-VL-7B-Instruct-Q8_0.gguf");
        assert_eq!(
            qwen.manifest().args.get("llm").map(String::as_str),
            Some("Qwen2.5-VL-7B-Instruct-Q8_0.gguf")
        );
    }

    /// Every offered size must be a real, safe path, since picking one
    /// substitutes it into the component that gets downloaded.
    #[test]
    fn quant_options_are_coherent() {
        let mut with_variants = 0;
        for bundle in builtin_bundles() {
            validate(bundle).unwrap_or_else(|error| panic!("{}: {error}", bundle.id));
            for component in &bundle.components {
                if component.variants.is_empty() {
                    continue;
                }
                with_variants += 1;
                // The default has to be one of the choices, or the picker opens
                // showing a size the bundle would not actually download.
                assert!(
                    component
                        .variants
                        .iter()
                        .any(|variant| variant.path == component.path),
                    "{}: default {} is not among its own size options",
                    bundle.id,
                    component.path
                );
            }
        }
        assert!(with_variants > 0, "the catalog offers sizes to choose from");
    }

    /// The text-to-video / image-to-video split is what decides whether a photo
    /// can be handed to a model at all, so the catalog has to offer both.
    #[test]
    fn the_catalog_covers_both_kinds_of_video_model() {
        let video: Vec<_> = builtin_bundles()
            .iter()
            .filter(|bundle| bundle.modality == Modality::Video)
            .collect();
        assert!(
            video.iter().any(|bundle| bundle.supports_init_image),
            "no image-to-video model on offer"
        );
        assert!(
            video.iter().any(|bundle| !bundle.supports_init_image),
            "no text-to-video model on offer"
        );
        assert!(
            builtin_bundles().iter().any(|bundle| bundle.featured),
            "a shortlist needs something on it"
        );
    }

    #[test]
    fn single_file_bundles_use_the_model_flag() {
        let sdxl = builtin("sdxl-base-1.0").expect("sdxl bundle");
        let manifest = sdxl.manifest();
        assert_eq!(
            manifest.single_file.as_deref(),
            Some("sd_xl_base_1.0.safetensors")
        );
        assert!(manifest.args.is_empty());
        assert!(!sdxl.gated());
    }

    #[test]
    fn nested_repo_paths_flatten_to_file_names() {
        let wan = builtin("wan2.2-ti2v-5b").expect("wan bundle");
        let decoder = wan
            .components
            .iter()
            .find(|component| component.flag.as_deref() == Some("tae"))
            .expect("TAEHV decoder component");
        assert_eq!(decoder.path, "safetensors/taew2_2.safetensors");
        assert_eq!(decoder.file_name(), "taew2_2.safetensors");
        assert!(decoder.variants.iter().any(|variant| {
            variant.flag.as_deref() == Some("vae")
                && variant.path == "split_files/vae/wan2.2_vae.safetensors"
        }));
    }

    fn sample_bundle(id: &str) -> Bundle {
        Bundle {
            id: id.to_owned(),
            label: "Test".into(),
            modality: Modality::Image,
            key: "acme/test".into(),
            summary: "test".into(),
            license: None,
            defaults: GenerationDefaults::default(),
            supports_init_image: false,
            featured: false,
            components: vec![Component {
                repo_id: "acme/repo".into(),
                path: "model.safetensors".into(),
                flag: None,
                role: "Checkpoint".into(),
                gated: false,
                approx_bytes: None,
                variants: Vec::new(),
            }],
        }
    }

    #[test]
    fn custom_bundles_survive_a_restart_and_can_shadow_builtins() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path();
        assert!(custom_bundles(data).is_empty());

        save_custom(data, sample_bundle("my-model")).unwrap();
        // Reading goes back to disk, so this is what a fresh process would see.
        let entries = catalog(data);
        let mine = entries
            .iter()
            .find(|entry| entry.bundle.id == "my-model")
            .expect("saved bundle");
        assert_eq!(mine.origin, Origin::Custom);
        assert!(entries.iter().any(|entry| entry.origin == Origin::Builtin));

        // A custom bundle with a builtin's id replaces it rather than duplicating.
        let mut shadow = sample_bundle("sdxl-base-1.0");
        shadow.label = "My corrected SDXL".into();
        save_custom(data, shadow).unwrap();
        let entries = catalog(data);
        let matches: Vec<_> = entries
            .iter()
            .filter(|entry| entry.bundle.id == "sdxl-base-1.0")
            .collect();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].bundle.label, "My corrected SDXL");
        assert_eq!(matches[0].origin, Origin::Custom);

        delete_custom(data, "my-model").unwrap();
        assert!(find(data, "my-model").is_none());
        assert!(delete_custom(data, "my-model").is_err());
    }

    #[test]
    fn a_corrupt_custom_file_does_not_take_out_the_catalog() {
        let dir = tempfile::tempdir().unwrap();
        let path = custom_catalog_path(dir.path());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{ this is not json").unwrap();
        assert!(custom_bundles(dir.path()).is_empty());
        assert!(!catalog(dir.path()).is_empty(), "builtins still load");
    }

    #[test]
    fn validation_rejects_unsafe_or_incoherent_bundles() {
        let mut escape = sample_bundle("escape");
        escape.components[0].path = "../../etc/passwd".into();
        assert!(validate(&escape).is_err());

        let mut bad_key = sample_bundle("bad-key");
        bad_key.key = "../../escape".into();
        assert!(validate(&bad_key).is_err());

        let mut bad_flag = sample_bundle("bad-flag");
        bad_flag.components[0].flag = Some("vae; rm -rf /".into());
        assert!(validate(&bad_flag).is_err());

        let mut two_models = sample_bundle("two-models");
        two_models.components.push(Component {
            repo_id: "acme/repo".into(),
            path: "other.safetensors".into(),
            flag: Some("diffusion-model".into()),
            role: "Diffusion".into(),
            gated: false,
            approx_bytes: None,
            variants: Vec::new(),
        });
        assert!(validate(&two_models).is_err());

        // Two components whose paths collapse to the same filename on disk.
        let mut collision = sample_bundle("collision");
        collision.components[0].flag = Some("diffusion-model".into());
        collision.components.push(Component {
            repo_id: "acme/other".into(),
            path: "nested/model.safetensors".into(),
            flag: Some("vae".into()),
            role: "VAE".into(),
            gated: false,
            approx_bytes: None,
            variants: Vec::new(),
        });
        assert!(validate(&collision).is_err());

        assert!(validate(&sample_bundle("fine")).is_ok());
    }

    #[test]
    fn unknown_ids_are_rejected() {
        assert!(builtin("no-such-bundle").is_none());
    }
}

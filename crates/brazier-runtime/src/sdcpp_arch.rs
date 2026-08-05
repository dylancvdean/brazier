//! Architecture detection and bundle assembly for stable-diffusion.cpp models.
//!
//! Rather than keeping a per-model list, this reads the checkpoint's own header
//! and matches it against architecture rules: GGUF files carry
//! `general.architecture`, and safetensors files carry tensor names and shapes
//! in a JSON header. Both sit at the start of the file, so a range request for
//! the first few hundred kilobytes is enough to identify a model without
//! downloading it.
//!
//! The upshot is that a *new model of a known architecture* assembles itself —
//! only a genuinely new architecture needs a rule added.

use std::collections::BTreeMap;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::{
    sdcpp::Modality,
    sdcpp_catalog::{Bundle, Component, GenerationDefaults},
};

const RULES: &str = include_str!("../../../model-recipes/sdcpp-architectures.json");
/// Header bytes to fetch. Safetensors headers for large checkpoints run to a
/// few hundred KB; GGUF puts its metadata in the first few.
const HEADER_BYTES: u64 = 512 * 1024;

// ---------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct Requirement {
    pub flag: String,
    pub role: String,
    pub repo_id: String,
    pub path: String,
    #[serde(default)]
    pub gated: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Detect {
    #[serde(default)]
    pub gguf_arch: Vec<String>,
    #[serde(default)]
    pub tensor_patterns: Vec<String>,
    #[serde(default)]
    pub name_patterns: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorChannels {
    pub tensor: String,
    pub axis: usize,
    pub equals: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct VariantWhen {
    #[serde(default)]
    pub name_patterns: Vec<String>,
    #[serde(default)]
    pub tensor_channels: Option<TensorChannels>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Variant {
    pub id: String,
    pub label: String,
    pub when: VariantWhen,
    /// Replaces the architecture's requirements when present.
    #[serde(default)]
    pub requires: Option<Vec<Requirement>>,
    #[serde(default)]
    pub defaults: Option<GenerationDefaults>,
    #[serde(default)]
    pub supports_init_image: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Architecture {
    pub id: String,
    pub label: String,
    pub modality: Modality,
    pub detect: Detect,
    #[serde(default)]
    pub requires: Vec<Requirement>,
    #[serde(default)]
    pub defaults: GenerationDefaults,
    /// Whether models of this architecture can start from a supplied frame.
    #[serde(default)]
    pub supports_init_image: bool,
    #[serde(default)]
    pub variants: Vec<Variant>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct SelfContainedMarkers {
    #[serde(default)]
    vae: Vec<String>,
    #[serde(default)]
    text_encoder: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Rules {
    #[serde(default)]
    self_contained_markers: SelfContainedMarkers,
    architectures: Vec<Architecture>,
}

fn rules() -> &'static Rules {
    static PARSED: std::sync::OnceLock<Rules> = std::sync::OnceLock::new();
    PARSED.get_or_init(|| {
        serde_json::from_str(RULES).expect("bundled sdcpp architecture rules are valid JSON")
    })
}

pub fn architectures() -> &'static [Architecture] {
    &rules().architectures
}

// ---------------------------------------------------------------------------
// Header probing
// ---------------------------------------------------------------------------

/// What a checkpoint's header tells us about itself.
#[derive(Debug, Clone, Default)]
pub struct ModelProbe {
    /// `general.architecture` for GGUF files.
    pub gguf_arch: Option<String>,
    /// Tensor names seen in the header (possibly truncated for huge headers).
    pub tensor_names: Vec<String>,
    /// Shapes for the tensors that were fully parsed.
    pub shapes: BTreeMap<String, Vec<u64>>,
}

impl ModelProbe {
    fn has_tensor_like(&self, pattern: &str) -> bool {
        self.tensor_names.iter().any(|name| name.contains(pattern))
    }

    /// Look up a shape by exact name or by suffix, since checkpoints vary in
    /// whether tensors carry a `model.diffusion_model.` prefix.
    fn shape_for(&self, tensor: &str) -> Option<&Vec<u64>> {
        self.shapes.get(tensor).or_else(|| {
            self.shapes
                .iter()
                .find(|(name, _)| name.ends_with(tensor))
                .map(|(_, shape)| shape)
        })
    }
}

/// Parse a checkpoint header. Accepts a truncated prefix of the file.
pub fn probe_header(bytes: &[u8]) -> ModelProbe {
    if bytes.starts_with(b"GGUF") {
        return probe_gguf(bytes);
    }
    probe_safetensors(bytes)
}

/// Read `general.architecture` out of a GGUF metadata block.
///
/// The key/value section starts after a 24-byte header; each key is a
/// length-prefixed string followed by a typed value. Only the one string key
/// is needed, so parsing stops as soon as it is found.
fn probe_gguf(bytes: &[u8]) -> ModelProbe {
    let mut probe = ModelProbe::default();
    let needle = b"general.architecture";
    let Some(position) = bytes
        .windows(needle.len())
        .position(|window| window == needle)
    else {
        return probe;
    };
    // Key bytes are followed by a u32 value type and, for strings, a u64 length.
    let value_start = position + needle.len();
    let Some(type_bytes) = bytes.get(value_start..value_start + 4) else {
        return probe;
    };
    // Type 8 is a string in every GGUF version that ships today.
    if u32::from_le_bytes(type_bytes.try_into().unwrap_or_default()) != 8 {
        return probe;
    }
    let length_start = value_start + 4;
    let Some(length_bytes) = bytes.get(length_start..length_start + 8) else {
        return probe;
    };
    let length = u64::from_le_bytes(length_bytes.try_into().unwrap_or_default()) as usize;
    if length == 0 || length > 128 {
        return probe;
    }
    let text_start = length_start + 8;
    if let Some(text) = bytes.get(text_start..text_start + length) {
        probe.gguf_arch = std::str::from_utf8(text).ok().map(str::to_owned);
    }
    // Tensor names follow the metadata and are useful for variant checks.
    probe.tensor_names = scan_ascii_tensor_names(bytes);
    probe
}

/// Parse a safetensors JSON header, tolerating a truncated prefix.
fn probe_safetensors(bytes: &[u8]) -> ModelProbe {
    let mut probe = ModelProbe::default();
    let Some(length_bytes) = bytes.get(0..8) else {
        return probe;
    };
    let header_len = u64::from_le_bytes(length_bytes.try_into().unwrap_or_default()) as usize;
    if header_len == 0 || header_len > 1 << 30 {
        return probe;
    }
    let available = &bytes[8..];
    let header = &available[..header_len.min(available.len())];
    let text = String::from_utf8_lossy(header);

    // Whole header present: parse it properly and keep every shape.
    if header_len <= available.len()
        && let Ok(map) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&text)
    {
        for (name, value) in map {
            if name == "__metadata__" {
                continue;
            }
            probe.tensor_names.push(name.clone());
            if let Some(shape) = value.get("shape").and_then(|shape| shape.as_array()) {
                probe.shapes.insert(
                    name,
                    shape.iter().filter_map(serde_json::Value::as_u64).collect(),
                );
            }
        }
        return probe;
    }

    // Truncated: recover what the prefix contains.
    probe.tensor_names = scan_ascii_tensor_names(header);
    probe.shapes = scan_shapes(&text);
    probe
}

/// Pull plausible tensor names out of raw bytes.
fn scan_ascii_tensor_names(bytes: &[u8]) -> Vec<String> {
    let mut names = Vec::new();
    let mut current = String::new();
    for &byte in bytes {
        let ch = byte as char;
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_') {
            current.push(ch);
            continue;
        }
        if current.len() >= 8 && current.contains('.') {
            names.push(std::mem::take(&mut current));
        } else {
            current.clear();
        }
        if names.len() >= 4096 {
            break;
        }
    }
    names
}

/// Recover `"name":{...,"shape":[…]}` pairs from a truncated JSON header.
fn scan_shapes(text: &str) -> BTreeMap<String, Vec<u64>> {
    let mut shapes = BTreeMap::new();
    for (index, _) in text.match_indices("\"shape\":[") {
        let Some(open) = text[index..].find('[').map(|offset| index + offset + 1) else {
            continue;
        };
        let Some(close) = text[open..].find(']').map(|offset| open + offset) else {
            continue;
        };
        let dims: Vec<u64> = text[open..close]
            .split(',')
            .filter_map(|part| part.trim().parse().ok())
            .collect();
        // Walk back to the tensor name that owns this entry.
        let Some(name) = text[..index].rfind("\":{").and_then(|brace| {
            let before = &text[..brace];
            before.rfind('"').map(|start| &before[start + 1..])
        }) else {
            continue;
        };
        if !dims.is_empty() {
            shapes.insert(name.to_owned(), dims);
        }
    }
    shapes
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

/// A bundle proposed for an arbitrary checkpoint, for review before install.
#[derive(Debug, Clone, Serialize)]
pub struct Proposal {
    pub bundle: Bundle,
    /// Architecture rule that matched, when one did.
    pub architecture: Option<String>,
    pub architecture_label: Option<String>,
    pub variant: Option<String>,
    /// How the architecture was identified, for the UI to show its working.
    pub detected_by: String,
    /// Whether the checkpoint carries its own VAE and text encoder.
    pub self_contained: bool,
    /// Anything the user should check before installing.
    pub warnings: Vec<String>,
}

fn matches_name(patterns: &[String], haystack: &str) -> bool {
    let lower = haystack.to_ascii_lowercase();
    patterns
        .iter()
        .any(|pattern| lower.contains(&pattern.to_ascii_lowercase()))
}

/// Whether a checkpoint already contains a VAE and a text encoder.
fn is_self_contained(probe: &ModelProbe) -> bool {
    let markers = &rules().self_contained_markers;
    let has_vae = markers
        .vae
        .iter()
        .any(|marker| probe.has_tensor_like(marker));
    let has_text = markers
        .text_encoder
        .iter()
        .any(|marker| probe.has_tensor_like(marker));
    has_vae && has_text
}

/// Match a probe (and the file's name) against the architecture rules.
pub fn detect(probe: &ModelProbe, name: &str) -> Option<(&'static Architecture, &'static str)> {
    if let Some(arch) = probe.gguf_arch.as_deref()
        && let Some(found) = architectures().iter().find(|candidate| {
            candidate
                .detect
                .gguf_arch
                .iter()
                .any(|value| value.eq_ignore_ascii_case(arch))
        })
    {
        return Some((found, "GGUF metadata"));
    }
    if let Some(found) = architectures().iter().find(|candidate| {
        candidate
            .detect
            .tensor_patterns
            .iter()
            .any(|pattern| probe.has_tensor_like(pattern))
    }) {
        return Some((found, "tensor names"));
    }
    architectures()
        .iter()
        .find(|candidate| matches_name(&candidate.detect.name_patterns, name))
        .map(|found| (found, "file name"))
}

fn variant_matches(variant: &Variant, probe: &ModelProbe, name: &str) -> bool {
    if let Some(channels) = &variant.when.tensor_channels {
        return probe
            .shape_for(&channels.tensor)
            .and_then(|shape| shape.get(channels.axis).copied())
            .is_some_and(|value| value == channels.equals);
    }
    !variant.when.name_patterns.is_empty() && matches_name(&variant.when.name_patterns, name)
}

/// Build a bundle proposal for one checkpoint file in a Hub repository.
pub fn assemble(
    repo_id: &str,
    path: &str,
    probe: &ModelProbe,
    modality_override: Option<Modality>,
) -> Proposal {
    let file_name = path.rsplit('/').next().unwrap_or(path);
    let haystack = format!("{repo_id}/{path}");
    let detected = detect(probe, &haystack);
    let self_contained = is_self_contained(probe);
    let mut warnings = Vec::new();

    let (architecture, architecture_label, detected_by) = match detected {
        Some((arch, how)) => (Some(arch), Some(arch.label.clone()), how.to_owned()),
        None => {
            warnings.push(
                "Could not identify this architecture from the file header. Add the VAE and text encoders it needs by hand before installing."
                    .to_owned(),
            );
            (None, None, "unrecognized".to_owned())
        }
    };

    // A variant can swap both the companion files and the defaults.
    let variant = architecture.and_then(|arch| {
        arch.variants
            .iter()
            .find(|variant| variant_matches(variant, probe, &haystack))
    });

    let requirements: Vec<Requirement> = if self_contained {
        Vec::new()
    } else {
        variant
            .and_then(|variant| variant.requires.clone())
            .or_else(|| architecture.map(|arch| arch.requires.clone()))
            .unwrap_or_default()
    };

    let mut defaults = architecture
        .map(|arch| arch.defaults.clone())
        .unwrap_or_default();
    if let Some(overrides) = variant.and_then(|variant| variant.defaults.clone()) {
        defaults = merge_defaults(defaults, overrides);
    }

    let modality = modality_override
        .or_else(|| architecture.map(|arch| arch.modality))
        .unwrap_or(Modality::Image);

    let mut components = Vec::new();
    components.push(Component {
        repo_id: repo_id.to_owned(),
        path: path.to_owned(),
        source_url: None,
        source_sha256: None,
        source_size: None,
        // A self-contained checkpoint is passed to `-m`; everything else is a
        // standalone diffusion model.
        flag: (!self_contained).then(|| "diffusion-model".to_owned()),
        role: if self_contained {
            "Checkpoint (UNet + VAE + text encoders)".to_owned()
        } else {
            "Diffusion model".to_owned()
        },
        gated: false,
        approx_bytes: None,
        variants: Vec::new(),
    });
    for requirement in requirements {
        if requirement.gated {
            warnings.push(format!(
                "{} comes from {}, which requires accepting its terms on Hugging Face and saving an access token.",
                requirement.role, requirement.repo_id
            ));
        }
        components.push(Component {
            repo_id: requirement.repo_id,
            path: requirement.path,
            source_url: None,
            source_sha256: None,
            source_size: None,
            flag: Some(requirement.flag),
            role: requirement.role,
            gated: requirement.gated,
            approx_bytes: None,
            variants: Vec::new(),
        });
    }

    if !self_contained && components.len() == 1 && architecture.is_some() {
        warnings.push(
            "This architecture is self-contained in the rules, but the file does not appear to include a VAE or text encoder. Check the result before generating."
                .to_owned(),
        );
    }

    let label = architecture
        .map(|arch| format!("{} · {}", arch.label, file_name))
        .unwrap_or_else(|| file_name.to_owned());

    Proposal {
        bundle: Bundle {
            supports_init_image: variant
                .and_then(|variant| variant.supports_init_image)
                .or_else(|| architecture.map(|arch| arch.supports_init_image))
                .unwrap_or(false),
            id: bundle_id(repo_id, file_name),
            label,
            modality,
            key: bundle_key(repo_id, file_name),
            summary: format!("Assembled from {repo_id}/{path}"),
            license: None,
            requires_license_acceptance: false,
            license_url: None,
            license_version: None,
            license_summary: None,
            defaults,
            // Hand-assembled bundles are the user's own, not a shortlist pick.
            featured: false,
            components,
        },
        architecture: architecture.map(|arch| arch.id.clone()),
        architecture_label,
        variant: variant.map(|variant| variant.label.clone()),
        detected_by,
        self_contained,
        warnings,
    }
}

fn merge_defaults(base: GenerationDefaults, overrides: GenerationDefaults) -> GenerationDefaults {
    GenerationDefaults {
        sampling_method: overrides.sampling_method.or(base.sampling_method),
        schedule: overrides.schedule.or(base.schedule),
        width: overrides.width.or(base.width),
        height: overrides.height.or(base.height),
        steps: overrides.steps.or(base.steps),
        cfg_scale: overrides.cfg_scale.or(base.cfg_scale),
        guidance: overrides.guidance.or(base.guidance),
        flow_shift: overrides.flow_shift.or(base.flow_shift),
        video_frames: overrides.video_frames.or(base.video_frames),
        fps: overrides.fps.or(base.fps),
        vae_on_cpu: overrides.vae_on_cpu.or(base.vae_on_cpu),
    }
}

/// Slug safe for use as a bundle id and directory name.
fn slug(value: &str) -> String {
    let mut out = String::new();
    let mut previous_dash = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' {
            out.push(ch.to_ascii_lowercase());
            previous_dash = false;
        } else if !previous_dash && !out.is_empty() {
            out.push('-');
            previous_dash = true;
        }
    }
    out.trim_matches('-').to_owned()
}

fn bundle_id(repo_id: &str, file_name: &str) -> String {
    let stem = file_name
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(file_name);
    format!("custom-{}", slug(&format!("{repo_id}-{stem}")))
}

fn bundle_key(repo_id: &str, file_name: &str) -> String {
    let owner = repo_id.split('/').next().unwrap_or("custom");
    let stem = file_name
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(file_name);
    format!("{}/{}", slug(owner), slug(stem))
}

/// Fetch enough of a Hub file to identify it, without downloading the weights.
pub async fn probe_hub_file(
    client: &reqwest::Client,
    data_dir: &std::path::Path,
    repo_id: &str,
    path: &str,
) -> anyhow::Result<ModelProbe> {
    crate::models_store::validate_repo_id(repo_id)?;
    let url = crate::download::resolve_url(repo_id, "main", path);
    let response = crate::hf_auth::apply_auth(
        client
            .get(url)
            .header("range", format!("bytes=0-{}", HEADER_BYTES - 1))
            .header(
                "user-agent",
                format!("brazier/{}", env!("CARGO_PKG_VERSION")),
            ),
        data_dir,
    )
    .send()
    .await
    .context("fetch checkpoint header")?
    .error_for_status()
    .context("checkpoint header request failed")?;
    let bytes = response.bytes().await.context("read checkpoint header")?;
    Ok(probe_header(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn safetensors_header(entries: &[(&str, Vec<u64>)]) -> Vec<u8> {
        let mut map = serde_json::Map::new();
        for (name, shape) in entries {
            map.insert(
                (*name).to_owned(),
                serde_json::json!({ "dtype": "F16", "shape": shape, "data_offsets": [0, 2] }),
            );
        }
        let json = serde_json::to_vec(&map).unwrap();
        let mut out = (json.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&json);
        out
    }

    #[test]
    fn reads_the_architecture_out_of_a_gguf_header() {
        // Mirrors the real layout: key string, u32 type tag, u64 length, value.
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&[0; 20]);
        bytes.extend_from_slice(b"general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&4_u64.to_le_bytes());
        bytes.extend_from_slice(b"flux");
        let probe = probe_header(&bytes);
        assert_eq!(probe.gguf_arch.as_deref(), Some("flux"));

        let (arch, how) = detect(&probe, "flux1-schnell-Q8_0.gguf").expect("flux rule");
        assert_eq!(arch.id, "flux");
        assert_eq!(how, "GGUF metadata");
    }

    #[test]
    fn identifies_flux_from_tensor_names_without_gguf_metadata() {
        let header = safetensors_header(&[
            ("double_blocks.0.img_attn.qkv.weight", vec![9216, 3072]),
            ("single_blocks.0.linear1.weight", vec![21504, 3072]),
        ]);
        let probe = probe_header(&header);
        let (arch, how) = detect(&probe, "some-unknown-finetune.safetensors").expect("flux rule");
        assert_eq!(arch.id, "flux");
        assert_eq!(how, "tensor names");
    }

    #[test]
    fn a_new_flux_finetune_assembles_with_its_encoders() {
        let header =
            safetensors_header(&[("double_blocks.0.img_attn.qkv.weight", vec![9216, 3072])]);
        let probe = probe_header(&header);
        let proposal = assemble(
            "someone/brand-new-flux-tune",
            "model.safetensors",
            &probe,
            None,
        );
        assert_eq!(proposal.architecture.as_deref(), Some("flux"));
        assert!(!proposal.self_contained);
        let flags: Vec<&str> = proposal
            .bundle
            .components
            .iter()
            .filter_map(|component| component.flag.as_deref())
            .collect();
        assert_eq!(flags, ["diffusion-model", "vae", "clip_l", "t5xxl"]);
        // The gated Flux VAE has to be called out before anyone starts a download.
        assert!(proposal.warnings.iter().any(|w| w.contains("access token")));
    }

    #[test]
    fn wan_variant_picks_its_vae_from_the_latent_channel_count() {
        let wan21 = probe_header(&safetensors_header(&[
            ("blocks.0.cross_attn.norm_q.weight", vec![1536]),
            ("patch_embedding.weight", vec![1536, 16, 1, 2, 2]),
        ]));
        let proposal = assemble("acme/wan", "wan2.1_t2v_1.3B_bf16.safetensors", &wan21, None);
        assert_eq!(proposal.architecture.as_deref(), Some("wan"));
        assert_eq!(proposal.bundle.modality, Modality::Video);
        assert!(proposal.bundle.components.iter().any(|component| {
            component.flag.as_deref() == Some("vae") && component.path.contains("wan_2.1_vae")
        }));

        let wan22 = probe_header(&safetensors_header(&[
            ("blocks.0.cross_attn.norm_q.weight", vec![3072]),
            ("patch_embedding.weight", vec![3072, 48, 1, 2, 2]),
        ]));
        let proposal = assemble("acme/wan", "wan2.2_ti2v_5B_fp16.safetensors", &wan22, None);
        assert_eq!(
            proposal.variant.as_deref(),
            Some("Wan 2.2 (48-channel VAE)")
        );
        assert!(proposal.bundle.components.iter().any(|component| {
            component.flag.as_deref() == Some("vae") && component.path.contains("wan2.2_vae")
        }));
    }

    #[test]
    fn self_contained_checkpoints_need_no_companions() {
        let header = safetensors_header(&[
            (
                "model.diffusion_model.input_blocks.0.0.weight",
                vec![320, 4, 3, 3],
            ),
            (
                "first_stage_model.encoder.conv_in.weight",
                vec![128, 3, 3, 3],
            ),
            (
                "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.weight",
                vec![768],
            ),
        ]);
        let probe = probe_header(&header);
        let proposal = assemble("acme/some-xl-merge", "merge_v3.safetensors", &probe, None);
        assert!(proposal.self_contained);
        assert_eq!(proposal.bundle.components.len(), 1);
        assert_eq!(proposal.bundle.components[0].flag, None);
        assert!(proposal.warnings.is_empty());
    }

    #[test]
    fn unknown_architectures_are_reported_rather_than_guessed() {
        let header = safetensors_header(&[("mystery.layer.0.weight", vec![16, 16])]);
        let probe = probe_header(&header);
        let proposal = assemble("acme/mystery", "mystery.safetensors", &probe, None);
        assert!(proposal.architecture.is_none());
        assert_eq!(proposal.detected_by, "unrecognized");
        assert_eq!(proposal.bundle.components.len(), 1);
        assert!(proposal.warnings.iter().any(|w| w.contains("by hand")));
    }

    #[test]
    fn truncated_headers_still_yield_tensor_names() {
        let full = safetensors_header(&[
            ("double_blocks.0.img_attn.qkv.weight", vec![9216, 3072]),
            ("double_blocks.1.img_attn.qkv.weight", vec![9216, 3072]),
        ]);
        let truncated = &full[..full.len() - 40];
        let probe = probe_header(truncated);
        assert!(probe.has_tensor_like("double_blocks.0.img_attn"));
    }

    #[test]
    fn ids_and_keys_are_filesystem_safe() {
        let id = bundle_id(
            "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "wan2.2_ti2v_5B_fp16.safetensors",
        );
        assert!(!id.contains('/'));
        assert!(!id.contains(".."));
        let key = bundle_key("Comfy-Org/Wan_2.2", "wan2.2_ti2v_5B_fp16.safetensors");
        assert_eq!(key.split('/').count(), 2);
        assert!(!key.contains(".."));
    }
}

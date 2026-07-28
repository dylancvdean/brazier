//! What to install on this machine, and whether that answer has changed.
//!
//! A new installation is a list of engines and an empty model library, which is
//! the point at which someone who has never run a local model has to decide
//! which of several thousand to download and at which of a dozen quantisations.
//! That is the question this module answers: given how much memory a machine
//! has, one named model per thing you might want to do.
//!
//! Recommendations are data, not code — see `model-recipes/recommendations.json`
//! — so they can be corrected as models are released without touching the
//! application, and a copy in the data directory overrides the shipped one
//! entirely.
//!
//! Two things are deliberately resolved late rather than written down:
//!
//! - **Which quantisation.** A file that fits one machine does not fit another,
//!   and quant ladders differ per repository, so the choice is made against the
//!   repository's real file sizes at the moment it is shown.
//! - **Whether it changed.** Each entry carries an id; installing through the
//!   flow records the id that was installed, and a later id for the same
//!   category is what makes an update worth mentioning.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const CATALOG: &str = include_str!("../../../model-recipes/recommendations.json");

/// Share of usable memory a model's weights may occupy.
///
/// The rest is not slack: the KV cache grows with context, the engine has its
/// own overhead, and on a unified-memory machine the OS and everything else on
/// screen are drawing from the same pool. Filling memory with weights produces a
/// model that loads and then stalls the machine on the first long conversation.
const WEIGHT_MEMORY_FRACTION: f64 = 0.60;

/// Quantisations in the order they would be preferred if memory were free,
/// each with the filename fragments that mean the same rung.
///
/// Unsloth's dynamic builds are named for the bits they average rather than for
/// the uniform quant they replace — `UD-Q4_K_XL` is a dynamic build of Q4, not a
/// rung of its own — so matching on the plain name alone would miss exactly the
/// files most worth preferring.
///
/// Stopping at Q2_K is deliberate: below it a model is usually worse than a
/// smaller model at a higher quant, so a machine that cannot hold Q2_K is better
/// served by a different recommendation than by a one-bit build of this one.
const QUANT_LADDER: [(&str, &[&str]); 6] = [
    ("Q8_0", &["q8_0", "q8_k_xl"]),
    ("Q6_K", &["q6_k"]),
    ("Q5_K_M", &["q5_k_m", "q5_k_xl"]),
    ("Q4_K_M", &["q4_k_m", "q4_k_xl"]),
    ("Q3_K_M", &["q3_k_m", "q3_k_xl"]),
    ("Q2_K", &["q2_k"]),
];

/// Filename markers for GGUF files in a model repository that are not the model.
///
/// Repositories ship companions next to the weights — a vision projector, a
/// draft model for speculative decoding — and they are much smaller than any
/// real quant. Without this, "the smallest file that fits" reliably picks a
/// 500MB projector and calls it a 26B model.
///
/// `dspark` covers PrismML's DSpark speculative-decoding drafter layer, shipped
/// alongside the Bonsai weights — at ~7 GB the bf16 reference is large enough
/// to fool the size-based fallback otherwise, and would be picked as "the
/// model" on a 16 GB machine.
const COMPANION_MARKERS: [&str; 5] = ["mmproj", "mtp-", "-draft", "projector", "dspark"];

// ---------------------------------------------------------------------------
// Catalogue
// ---------------------------------------------------------------------------

/// A chat, agent, or transcription model named by repository.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RepoRecommendation {
    pub id: String,
    pub label: String,
    pub repo_id: String,
    /// `by_memory`, or a literal quant name to pin one.
    #[serde(default)]
    pub quant: Option<String>,
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    /// `os-arch` pairs this model cannot run on, e.g. `macos-aarch64`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unavailable_on: Vec<String>,
    #[serde(default)]
    pub unavailable_note: Option<String>,
    /// Optional companion files to fetch into the same model directory after
    /// the main weights, e.g. a vision projector (`mmproj-…gguf`) that
    /// `llama.cpp` auto-attaches when it sits next to the model. Filters the
    /// repository's real file list at resolve time so an absent companion is
    /// reported instead of failing the install.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub companion_files: Vec<String>,
    /// A mainline-compatible recommendation to use when this model requires a
    /// source-built runtime fork whose local toolchain is unavailable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback: Option<Box<RepoRecommendation>>,
}

impl RepoRecommendation {
    fn runs_here(&self) -> bool {
        let here = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
        !self.unavailable_on.iter().any(|entry| entry == &here)
    }
}

/// One installable half of a split video recommendation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BundlePart {
    pub bundle_id: String,
    /// What this half is for, e.g. `text-to-video`.
    pub role: String,
    pub label: String,
    #[serde(default)]
    pub variant: Option<String>,
}

/// An image or video model, named by stable-diffusion.cpp bundle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BundleRecommendation {
    pub id: String,
    pub label: String,
    /// The single bundle to install. Absent when `parts` names several.
    #[serde(default)]
    pub bundle_id: Option<String>,
    /// Component size to substitute, matched against the bundle's variants.
    #[serde(default)]
    pub variant: Option<String>,
    /// Separate text-to-video and image-to-video models, when the
    /// recommendation is split across two.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<BundlePart>,
    #[serde(default)]
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tier {
    pub min_gb: u32,
    #[serde(default)]
    pub text: Option<RepoRecommendation>,
    /// Absent means agent work uses the same model as chat.
    #[serde(default)]
    pub agent: Option<RepoRecommendation>,
    /// Additional agent models that fit this tier. The first `agent` remains
    /// the default so existing installs and recommendation state stay stable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agent_options: Vec<RepoRecommendation>,
    #[serde(default)]
    pub image: Option<BundleRecommendation>,
    #[serde(default)]
    pub video: Option<BundleRecommendation>,
}

/// A model a voice session needs, by the engine that loads it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoiceModel {
    pub id: String,
    pub label: String,
    /// `personaplex` or `whisper`.
    pub kind: String,
    pub repo_id: String,
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
}

/// Voice is not tiered.
///
/// PersonaPlex is the only realtime voice family Brazier runs and the
/// recogniser beside it is small, so there is no choice to make by memory. What
/// a new installation is missing is not which model but the fact that voice
/// needs two of them plus a runtime, which is what this describes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoiceRecommendation {
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub models: Vec<VoiceModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    #[serde(default)]
    pub schema_version: u32,
    pub tiers: Vec<Tier>,
    #[serde(default)]
    pub voice: Option<VoiceRecommendation>,
}

impl Catalog {
    /// The tier a machine with this much usable memory falls into.
    ///
    /// The largest tier at or below the machine's memory, so a size between two
    /// tiers is treated as the smaller one rather than being promoted into
    /// recommendations it cannot hold.
    pub fn tier_for(&self, memory_bytes: u64) -> Option<&Tier> {
        let gb = memory_bytes / (1024 * 1024 * 1024);
        self.tiers
            .iter()
            .filter(|tier| u64::from(tier.min_gb) <= gb)
            .max_by_key(|tier| tier.min_gb)
    }
}

/// A data-directory copy replaces the shipped catalogue entirely.
pub fn override_path(data_dir: &Path) -> PathBuf {
    data_dir.join("model-recipes").join("recommendations.json")
}

/// The catalogue in force: the data-directory copy when there is a readable
/// one, and the shipped file otherwise.
pub fn catalog(data_dir: &Path) -> Catalog {
    if let Ok(bytes) = std::fs::read(override_path(data_dir)) {
        match serde_json::from_slice::<Catalog>(&bytes) {
            Ok(catalog) => return catalog,
            Err(error) => {
                tracing::warn!(%error, "ignoring an invalid recommendations override");
            }
        }
    }
    serde_json::from_str(CATALOG).expect("the shipped recommendations catalog must parse")
}

// ---------------------------------------------------------------------------
// Quantisation choice
// ---------------------------------------------------------------------------

/// One quantisation of a model, as a thing that can be downloaded.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct QuantChoice {
    /// Quant name when it was recognised, e.g. `Q4_K_M`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    /// Files to fetch, in order. More than one when the quant is sharded.
    pub files: Vec<String>,
    pub bytes: u64,
    /// True when nothing fitted the budget and this is simply the smallest
    /// build there is.
    pub tight: bool,
}

/// Strip a shard suffix, so the parts of one quantisation group together.
fn shard_group(path: &str) -> String {
    crate::models_store::shard_group(path)
}

fn is_companion(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower).to_owned();
    COMPANION_MARKERS
        .iter()
        .any(|marker| name.starts_with(marker) || name.contains(marker))
}

/// One candidate build of a model: every shard of one quantisation.
#[derive(Debug, Clone)]
struct Candidate {
    group: String,
    files: Vec<String>,
    bytes: u64,
}

/// Group a repository's GGUF listing into candidate builds.
///
/// Companion files are dropped, and shards of one quantisation are summed, so
/// what comes back is the set of things that could actually be run.
fn candidates(files: &[(String, Option<u64>)]) -> Vec<Candidate> {
    let mut grouped: std::collections::BTreeMap<String, Candidate> =
        std::collections::BTreeMap::new();
    for (path, size) in files {
        if !path.to_ascii_lowercase().ends_with(".gguf") || is_companion(path) {
            continue;
        }
        let group = shard_group(path);
        let entry = grouped.entry(group.clone()).or_insert_with(|| Candidate {
            group,
            files: Vec::new(),
            bytes: 0,
        });
        entry.files.push(path.clone());
        entry.bytes = entry.bytes.saturating_add(size.unwrap_or(0));
    }
    for candidate in grouped.values_mut() {
        candidate.files.sort();
    }
    grouped.into_values().collect()
}

/// Whether a candidate is a build of a rung, by any of its names.
fn matches_rung(candidate: &Candidate, aliases: &[&str]) -> bool {
    let lower = candidate.group.to_ascii_lowercase();
    aliases.iter().any(|alias| lower.contains(alias))
}

/// Prefer Unsloth's dynamic build of a rung over the plain one.
///
/// `UD-…_XL` quants spend their bits where the model is most sensitive, so at
/// the same nominal rung they are meaningfully better than a uniform quant and
/// only slightly larger.
fn dynamic_first(left: &Candidate, right: &Candidate) -> std::cmp::Ordering {
    let score = |candidate: &Candidate| {
        let lower = candidate.group.to_ascii_lowercase();
        u8::from(lower.contains("ud-")) + u8::from(lower.ends_with("_xl"))
    };
    score(right)
        .cmp(&score(left))
        .then(left.bytes.cmp(&right.bytes))
}

/// Choose the build of a model to install on a machine with this much memory.
///
/// Walks the quality ladder from the top and takes the first rung that fits the
/// weight budget. A repository whose quants are named in some other scheme —
/// and there are several — still resolves: the ladder simply matches nothing
/// and the largest build that fits is taken instead.
pub fn choose_quant(files: &[(String, Option<u64>)], memory_bytes: u64) -> Option<QuantChoice> {
    let budget = (memory_bytes as f64 * WEIGHT_MEMORY_FRACTION) as u64;
    let candidates = candidates(files);
    if candidates.is_empty() {
        return None;
    }

    for (rung, aliases) in QUANT_LADDER {
        let mut matching: Vec<&Candidate> = candidates
            .iter()
            .filter(|candidate| matches_rung(candidate, aliases))
            .collect();
        matching.sort_by(|left, right| dynamic_first(left, right));
        if let Some(choice) = matching
            .into_iter()
            .find(|candidate| candidate.bytes <= budget && candidate.bytes > 0)
        {
            return Some(QuantChoice {
                quant: Some(rung.to_owned()),
                files: choice.files.clone(),
                bytes: choice.bytes,
                tight: false,
            });
        }
    }

    // Nothing on the ladder fits, or this repository names its quants
    // differently. Take the largest build inside the budget.
    if let Some(choice) = candidates
        .iter()
        .filter(|candidate| candidate.bytes <= budget && candidate.bytes > 0)
        .max_by_key(|candidate| candidate.bytes)
    {
        return Some(QuantChoice {
            quant: None,
            files: choice.files.clone(),
            bytes: choice.bytes,
            tight: false,
        });
    }

    // Nothing fits at all. Offer the smallest build there is and say so, rather
    // than showing a recommendation with no way to act on it.
    candidates
        .iter()
        .filter(|candidate| candidate.bytes > 0)
        .min_by_key(|candidate| candidate.bytes)
        .map(|choice| QuantChoice {
            quant: None,
            files: choice.files.clone(),
            bytes: choice.bytes,
            tight: true,
        })
}

/// Pin a named quantisation rather than choosing by memory.
pub fn find_quant(files: &[(String, Option<u64>)], quant: &str) -> Option<QuantChoice> {
    let aliases = [quant.to_ascii_lowercase()];
    let aliases: Vec<&str> = aliases.iter().map(String::as_str).collect();
    let mut matching: Vec<Candidate> = candidates(files)
        .into_iter()
        .filter(|candidate| matches_rung(candidate, &aliases))
        .collect();
    matching.sort_by(dynamic_first);
    matching.into_iter().next().map(|choice| QuantChoice {
        quant: Some(quant.to_owned()),
        files: choice.files,
        bytes: choice.bytes,
        tight: false,
    })
}

// ---------------------------------------------------------------------------
// What has been installed through the flow
// ---------------------------------------------------------------------------

/// A category that was installed from a recommendation, and which one.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InstalledRecommendation {
    /// The recommendation `id` in force when it was installed.
    pub recommendation_id: String,
    /// What it resolved to, for the record.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Seconds since the Unix epoch, as a string.
    pub installed_at: String,
}

/// Which categories were set up through the recommendation flow.
///
/// Only these are worth telling anyone about when a recommendation changes: a
/// model chosen deliberately from Discover is nobody's business to second-guess.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RecommendationState {
    /// Whether the person asked not to be told about any of this again.
    pub suppressed: bool,
    /// Category (`text`, `agent`, `image`, `video`, `voice`) → what was installed.
    pub installed: std::collections::BTreeMap<String, InstalledRecommendation>,
    /// Recommendation ids that were offered as a swap and declined, so a
    /// declined offer is not made again for the same model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dismissed: Vec<String>,
}

pub fn state_path(data_dir: &Path) -> PathBuf {
    data_dir.join("recommendation-state.json")
}

pub fn load_state(data_dir: &Path) -> RecommendationState {
    std::fs::read(state_path(data_dir))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

pub async fn save_state(data_dir: &Path, state: &RecommendationState) -> anyhow::Result<()> {
    let path = state_path(data_dir);
    crate::persistence::write_json(&path, state, "recommendation state").await
}

/// Record one category without disturbing any other category that happens to
/// point at the same installed model.
///
/// Chat and agent commonly begin by sharing one model. If only the agent
/// recommendation changes later, accepting it must leave chat bound to the
/// original recommendation. This state operation deliberately has no model
/// deletion side effect; removing weights is a separate, explicit library
/// action.
pub fn record_install(
    state: &mut RecommendationState,
    category: String,
    recommendation_id: String,
    model_id: Option<String>,
    installed_at: String,
) {
    state.installed.insert(
        category,
        InstalledRecommendation {
            recommendation_id: recommendation_id.clone(),
            model_id,
            installed_at,
        },
    );
    state.dismissed.retain(|entry| entry != &recommendation_id);
}

/// A category whose recommendation has moved on since it was installed.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PendingSwap {
    pub category: String,
    pub installed_id: String,
    pub recommended_id: String,
    pub recommended_label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// The categories worth offering a swap for.
///
/// Nothing is offered when the person opted out, for a category they did not
/// install through the flow, or for a swap they have already declined.
pub fn pending_swaps(
    catalog: &Catalog,
    state: &RecommendationState,
    tier: &Tier,
) -> Vec<PendingSwap> {
    if state.suppressed {
        return Vec::new();
    }
    let _ = catalog;
    let mut swaps = Vec::new();
    let mut consider =
        |category: &str, id: Option<&str>, label: Option<&str>, summary: Option<&str>| {
            let (Some(id), Some(label)) = (id, label) else {
                return;
            };
            let Some(installed) = state.installed.get(category) else {
                return;
            };
            if installed.recommendation_id == id || state.dismissed.iter().any(|entry| entry == id)
            {
                return;
            }
            swaps.push(PendingSwap {
                category: category.to_owned(),
                installed_id: installed.recommendation_id.clone(),
                recommended_id: id.to_owned(),
                recommended_label: label.to_owned(),
                summary: summary.map(ToOwned::to_owned),
            });
        };

    let agent = resolved_agent(tier);
    consider(
        "text",
        tier.text.as_ref().map(|entry| entry.id.as_str()),
        tier.text.as_ref().map(|entry| entry.label.as_str()),
        tier.text
            .as_ref()
            .and_then(|entry| entry.summary.as_deref()),
    );
    consider(
        "agent",
        agent.map(|entry| entry.id.as_str()),
        agent.map(|entry| entry.label.as_str()),
        agent.and_then(|entry| entry.summary.as_deref()),
    );
    consider(
        "image",
        tier.image.as_ref().map(|entry| entry.id.as_str()),
        tier.image.as_ref().map(|entry| entry.label.as_str()),
        tier.image
            .as_ref()
            .and_then(|entry| entry.summary.as_deref()),
    );
    consider(
        "video",
        tier.video.as_ref().map(|entry| entry.id.as_str()),
        tier.video.as_ref().map(|entry| entry.label.as_str()),
        tier.video
            .as_ref()
            .and_then(|entry| entry.summary.as_deref()),
    );
    swaps
}

/// The model agent work should use at a tier.
///
/// A tier that names no agent model, or one that cannot run on this machine,
/// uses the chat model — the agent still needs something to think with, and the
/// chat recommendation is already known to fit.
pub fn resolved_agent(tier: &Tier) -> Option<&RepoRecommendation> {
    match tier.agent.as_ref() {
        Some(agent) if agent.runs_here() => Some(agent),
        _ => tier.text.as_ref(),
    }
}

/// Every agent choice that is available on this machine, default first.
pub fn resolved_agent_options(tier: &Tier) -> Vec<&RepoRecommendation> {
    let mut options = Vec::new();
    if let Some(agent) = resolved_agent(tier) {
        options.push(agent);
    }
    for agent in &tier.agent_options {
        if agent.runs_here() && !options.iter().any(|existing| existing.id == agent.id) {
            options.push(agent);
        }
    }
    options
}

/// Why the tier's own agent model was not used, when it was not.
pub fn agent_substitution_note(tier: &Tier) -> Option<&str> {
    let agent = tier.agent.as_ref()?;
    if agent.runs_here() {
        return None;
    }
    agent.unavailable_note.as_deref()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gb(value: u64) -> u64 {
        value * 1024 * 1024 * 1024
    }

    #[test]
    fn the_shipped_catalog_parses_and_covers_the_documented_tiers() {
        let catalog: Catalog = serde_json::from_str(CATALOG).unwrap();
        let sizes: Vec<u32> = catalog.tiers.iter().map(|tier| tier.min_gb).collect();
        assert_eq!(sizes, vec![8, 16, 24, 32, 48, 64, 96, 192]);
    }

    /// A machine between two tiers gets the smaller one; it cannot hold what
    /// the larger tier assumes.
    #[test]
    fn a_tier_is_the_largest_that_fits() {
        let catalog: Catalog = serde_json::from_str(CATALOG).unwrap();
        assert_eq!(catalog.tier_for(gb(8)).unwrap().min_gb, 8);
        assert_eq!(catalog.tier_for(gb(20)).unwrap().min_gb, 16);
        assert_eq!(catalog.tier_for(gb(128)).unwrap().min_gb, 96);
        assert_eq!(catalog.tier_for(gb(512)).unwrap().min_gb, 192);
        // Below the smallest tier there is nothing honest to suggest.
        assert!(catalog.tier_for(gb(4)).is_none());
    }

    #[test]
    fn a_tier_without_an_agent_model_uses_its_chat_model() {
        let catalog: Catalog = serde_json::from_str(CATALOG).unwrap();
        let tier = catalog.tier_for(gb(16)).unwrap();
        assert!(tier.agent.is_none());
        assert_eq!(
            resolved_agent(tier).unwrap().id,
            tier.text.as_ref().unwrap().id
        );
    }

    /// A projector or draft model is much smaller than any real quant, so
    /// without excluding them "the smallest build" is reliably wrong.
    #[test]
    fn companion_files_are_not_mistaken_for_the_model() {
        let files = vec![
            ("mmproj-F16.gguf".to_owned(), Some(1_200_000_000)),
            (
                "mtp-gemma-4-26B-A4B-it-Q8_0.gguf".to_owned(),
                Some(500_000_000),
            ),
            (
                "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf".to_owned(),
                Some(16_900_000_000),
            ),
        ];
        let choice = choose_quant(&files, gb(32)).unwrap();
        assert_eq!(choice.quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(choice.files, vec!["gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"]);
    }

    /// The real ladder from `unsloth/gemma-4-26B-A4B-it-GGUF`, so the rule is
    /// tested against the sizes it will actually meet.
    #[test]
    fn picks_the_best_rung_that_fits_the_budget() {
        let files = vec![
            (
                "gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf".to_owned(),
                Some(10_500_000_000),
            ),
            (
                "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf".to_owned(),
                Some(12_700_000_000),
            ),
            (
                "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf".to_owned(),
                Some(16_900_000_000),
            ),
            (
                "gemma-4-26B-A4B-it-UD-Q5_K_M.gguf".to_owned(),
                Some(21_200_000_000),
            ),
            (
                "gemma-4-26B-A4B-it-UD-Q6_K.gguf".to_owned(),
                Some(23_200_000_000),
            ),
            (
                "gemma-4-26B-A4B-it-Q8_0.gguf".to_owned(),
                Some(26_900_000_000),
            ),
        ];
        // 24GB machine: 14.4GB budget.
        assert_eq!(
            choose_quant(&files, gb(24)).unwrap().quant.as_deref(),
            Some("Q3_K_M")
        );
        // 32GB machine: 19.2GB budget.
        assert_eq!(
            choose_quant(&files, gb(32)).unwrap().quant.as_deref(),
            Some("Q4_K_M")
        );
        // 48GB machine: 28.8GB budget, so the whole ladder is available.
        assert_eq!(
            choose_quant(&files, gb(48)).unwrap().quant.as_deref(),
            Some("Q8_0")
        );
    }

    /// Unsloth publishes both a uniform and a dynamic build of a rung; the
    /// dynamic one is better at nearly the same size.
    #[test]
    fn a_dynamic_build_of_a_rung_wins_over_the_uniform_one() {
        let files = vec![
            ("model-Q4_K_M.gguf".to_owned(), Some(16_000_000_000)),
            ("model-UD-Q4_K_XL.gguf".to_owned(), Some(17_000_000_000)),
        ];
        let choice = choose_quant(&files, gb(48)).unwrap();
        assert_eq!(choice.files, vec!["model-UD-Q4_K_XL.gguf"]);
    }

    /// A quant too large for one file is published as shards, which are one
    /// choice and have to be downloaded together.
    #[test]
    fn shards_of_one_quant_are_one_choice() {
        let files = vec![
            (
                "Laguna-S-2.1-UD-Q4_K_M-00001-of-00002.gguf".to_owned(),
                Some(40_000_000_000),
            ),
            (
                "Laguna-S-2.1-UD-Q4_K_M-00002-of-00002.gguf".to_owned(),
                Some(33_100_000_000),
            ),
        ];
        let choice = choose_quant(&files, gb(128)).unwrap();
        assert_eq!(choice.quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(choice.files.len(), 2);
        assert_eq!(choice.bytes, 73_100_000_000);
    }

    /// `prism-ml/Bonsai-27B-gguf` names its builds in a scheme the ladder does
    /// not recognise, and ships a speculative-decoding drafter (`dspark`)
    /// alongside the weights that is large enough to fool a size-based fallback
    /// otherwise. The recommendation still has to resolve to the real model.
    #[test]
    fn an_unfamiliar_quant_scheme_falls_back_to_the_largest_that_fits() {
        let files = vec![
            ("Bonsai-27B-mmproj-Q8_0.gguf".to_owned(), Some(600_000_000)),
            (
                "Bonsai-27B-dspark-Q4_1.gguf".to_owned(),
                Some(1_800_000_000),
            ),
            ("Bonsai-27B-Q1_0.gguf".to_owned(), Some(3_800_000_000)),
            (
                "Bonsai-27B-dspark-bf16.gguf".to_owned(),
                Some(7_300_000_000),
            ),
            ("Bonsai-27B-F16.gguf".to_owned(), Some(53_800_000_000)),
        ];
        // 8GB machine: 4.8GB budget. The projector and drafter must not be
        // chosen; the only real build that fits is the 1-bit pack.
        let small = choose_quant(&files, gb(8)).unwrap();
        assert_eq!(small.files, vec!["Bonsai-27B-Q1_0.gguf"]);
        assert!(!small.tight);
        // 16GB machine: 9.6GB budget. The 7.3GB dspark-bf16 drafter would have
        // been picked by the size fallback before `dspark` was a companion
        // marker; it must now still resolve to the 1-bit model, since the
        // 53.8GB full-precision build does not fit.
        let larger = choose_quant(&files, gb(16)).unwrap();
        assert_eq!(larger.files, vec!["Bonsai-27B-Q1_0.gguf"]);
    }

    /// Nothing fitting is a fact worth stating, not a reason to show no model.
    #[test]
    fn a_model_that_cannot_fit_is_offered_with_a_warning() {
        let files = vec![("huge-Q8_0.gguf".to_owned(), Some(300_000_000_000))];
        let choice = choose_quant(&files, gb(16)).unwrap();
        assert!(choice.tight);
        assert_eq!(choice.bytes, 300_000_000_000);
    }

    #[test]
    fn a_repository_with_no_weights_resolves_to_nothing() {
        let files = vec![("mmproj-F16.gguf".to_owned(), Some(1_000_000))];
        assert!(choose_quant(&files, gb(64)).is_none());
    }

    #[tokio::test]
    async fn swaps_are_offered_only_for_what_the_flow_installed() {
        let dir = tempfile::tempdir().unwrap();
        let catalog: Catalog = serde_json::from_str(CATALOG).unwrap();
        let tier = catalog.tier_for(gb(32)).unwrap();

        let mut state = RecommendationState::default();
        // Installed a model that is no longer the recommendation.
        state.installed.insert(
            "text".into(),
            InstalledRecommendation {
                recommendation_id: "something-older".into(),
                model_id: None,
                installed_at: "1767225600".into(),
            },
        );
        save_state(dir.path(), &state).await.unwrap();
        let loaded = load_state(dir.path());

        let swaps = pending_swaps(&catalog, &loaded, tier);
        assert_eq!(swaps.len(), 1);
        assert_eq!(swaps[0].category, "text");
        assert_eq!(swaps[0].recommended_id, tier.text.as_ref().unwrap().id);
    }

    #[test]
    fn opting_out_silences_every_category() {
        let catalog: Catalog = serde_json::from_str(CATALOG).unwrap();
        let tier = catalog.tier_for(gb(32)).unwrap();
        let mut state = RecommendationState {
            suppressed: true,
            ..RecommendationState::default()
        };
        state.installed.insert(
            "text".into(),
            InstalledRecommendation {
                recommendation_id: "something-older".into(),
                model_id: None,
                installed_at: "1767225600".into(),
            },
        );
        assert!(pending_swaps(&catalog, &state, tier).is_empty());
    }

    #[test]
    fn a_declined_swap_is_not_offered_again() {
        let catalog: Catalog = serde_json::from_str(CATALOG).unwrap();
        let tier = catalog.tier_for(gb(32)).unwrap();
        let recommended = tier.text.as_ref().unwrap().id.clone();
        let mut state = RecommendationState {
            dismissed: vec![recommended],
            ..RecommendationState::default()
        };
        state.installed.insert(
            "text".into(),
            InstalledRecommendation {
                recommendation_id: "something-older".into(),
                model_id: None,
                installed_at: "1767225600".into(),
            },
        );
        assert!(pending_swaps(&catalog, &state, tier).is_empty());
    }

    #[test]
    fn upgrading_agent_preserves_a_chat_model_the_categories_used_to_share() {
        let catalog: Catalog = serde_json::from_str(CATALOG).unwrap();
        let tier = catalog.tier_for(gb(128)).unwrap();
        let text = tier.text.as_ref().unwrap();
        let agent = resolved_agent(tier).unwrap();
        assert_ne!(text.id, agent.id);

        // Both categories originally shared the current chat recommendation.
        let shared = InstalledRecommendation {
            recommendation_id: text.id.clone(),
            model_id: Some("gguf:shared-chat-model".into()),
            installed_at: "1767225600".into(),
        };
        let mut state = RecommendationState::default();
        state.installed.insert("text".into(), shared.clone());
        state.installed.insert("agent".into(), shared);

        // Only agent moved, so only agent should be offered a swap.
        let swaps = pending_swaps(&catalog, &state, tier);
        assert_eq!(swaps.len(), 1);
        assert_eq!(swaps[0].category, "agent");
        assert_eq!(swaps[0].recommended_id, agent.id);

        record_install(
            &mut state,
            "agent".into(),
            agent.id.clone(),
            Some("gguf:new-agent-model".into()),
            "1767225700".into(),
        );

        let chat = state.installed.get("text").unwrap();
        assert_eq!(chat.recommendation_id, text.id);
        assert_eq!(chat.model_id.as_deref(), Some("gguf:shared-chat-model"));
        assert_eq!(
            state.installed.get("agent").unwrap().recommendation_id,
            agent.id
        );
    }
}

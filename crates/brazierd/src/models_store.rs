//! On-disk model layout and listing for GGUF and MLX weights.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::{
    mlx::MlxKind,
    model_library,
    types::{ModelCapabilities, ModelDescriptor},
};

/// Content-keyed root for downloaded GGUF weights.
pub fn gguf_root(data_dir: &Path) -> PathBuf {
    data_dir.join("models").join("gguf")
}

/// Root for downloaded MLX model snapshots (`owner/repo` directories).
pub fn mlx_root(data_dir: &Path) -> PathBuf {
    data_dir.join("models").join("mlx")
}

/// Temporary directory for partial downloads.
pub fn downloads_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("downloads")
}

/// Stable model id for a GGUF file under the models root.
///
/// Format: `gguf:{repo_relative_path}` where path uses `/` separators.
pub fn model_id_for_path(gguf_root: &Path, file: &Path) -> anyhow::Result<String> {
    let relative = file
        .strip_prefix(gguf_root)
        .map_err(|_| anyhow::anyhow!("model path is outside the GGUF store"))?;
    let key = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    anyhow::ensure!(!key.is_empty(), "empty model key");
    anyhow::ensure!(!key.contains(".."), "model key must not contain '..'");
    Ok(format!("gguf:{key}"))
}

/// Resolve a local model id to an on-disk path.
pub fn path_for_model_id(
    data_dir: &Path,
    model_id: &str,
    extra_library_paths: &[PathBuf],
) -> anyhow::Result<PathBuf> {
    if let Some(path) = model_id.strip_prefix("gguf-ext:") {
        return path_for_external_gguf_id(extra_library_paths, path);
    }
    if let Some(key) = model_id.strip_prefix("gguf:") {
        return path_for_gguf_id(data_dir, key);
    }
    if let Some(payload) = model_id.strip_prefix("mlx-vlm-ext:") {
        return path_for_external_mlx_id(extra_library_paths, payload);
    }
    if let Some(payload) = model_id.strip_prefix("mlx-ext:") {
        return path_for_external_mlx_id(extra_library_paths, payload);
    }
    if let Some(key) = model_id.strip_prefix("mlx-vlm:") {
        return path_for_mlx_id(data_dir, key);
    }
    if let Some(key) = model_id.strip_prefix("mlx:") {
        return path_for_mlx_id(data_dir, key);
    }
    if model_id.starts_with("streaming-asr:") {
        return crate::streaming_asr::path_for_model_id(data_dir, model_id);
    }
    if model_id.starts_with("sdcpp-image:") || model_id.starts_with("sdcpp-video:") {
        return crate::sdcpp::path_for_model_id(data_dir, model_id);
    }
    if model_id.starts_with("personaplex:") {
        return crate::voice::path_for_model_id(data_dir, model_id);
    }
    anyhow::bail!("unknown local model id: {model_id}");
}

fn validate_library_relative_key(key: &str) -> anyhow::Result<()> {
    anyhow::ensure!(!key.is_empty(), "empty model key");
    anyhow::ensure!(
        !key.split('/')
            .any(|part| part.is_empty() || part == "." || part == ".."),
        "invalid model key"
    );
    Ok(())
}

fn validate_gguf_key(key: &str) -> anyhow::Result<()> {
    validate_library_relative_key(key)
}

fn model_id_for_external_path(index: usize, root: &Path, file: &Path) -> anyhow::Result<String> {
    let root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let file = std::fs::canonicalize(file).unwrap_or_else(|_| file.to_path_buf());
    let relative = file
        .strip_prefix(&root)
        .map_err(|_| anyhow::anyhow!("model path is outside the library root"))?;
    let key = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    validate_gguf_key(&key)?;
    Ok(format!("gguf-ext:{index}:{key}"))
}

pub fn path_for_external_gguf_id(
    extra_library_paths: &[PathBuf],
    payload: &str,
) -> anyhow::Result<PathBuf> {
    let (index_str, key) = payload
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("invalid external GGUF model id"))?;
    let index: usize = index_str
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid external library index"))?;
    validate_gguf_key(key)?;
    let root = extra_library_paths
        .get(index)
        .ok_or_else(|| anyhow::anyhow!("unknown external library index {index}"))?;
    let path = root.join(key);
    anyhow::ensure!(
        path.extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf")),
        "model path must end in .gguf"
    );
    let canonical_root = std::fs::canonicalize(root)
        .map_err(|error| anyhow::anyhow!("library root {}: {error}", root.display()))?;
    let canonical_path = std::fs::canonicalize(&path)
        .map_err(|_| anyhow::anyhow!("external model file not found for gguf-ext:{index}:{key}"))?;
    anyhow::ensure!(
        canonical_path.starts_with(&canonical_root),
        "external model path escapes its library root"
    );
    Ok(canonical_path)
}

fn model_id_for_external_mlx(
    kind: MlxKind,
    index: usize,
    root: &Path,
    dir: &Path,
) -> anyhow::Result<String> {
    let root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let dir = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    let relative = dir
        .strip_prefix(&root)
        .map_err(|_| anyhow::anyhow!("model path is outside the library root"))?;
    let key = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    validate_library_relative_key(&key)?;
    Ok(format!("{}-ext:{index}:{key}", engine_prefix(kind)))
}

pub fn path_for_external_mlx_id(
    extra_library_paths: &[PathBuf],
    payload: &str,
) -> anyhow::Result<PathBuf> {
    let (index_str, key) = payload
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("invalid external MLX model id"))?;
    let index: usize = index_str
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid external library index"))?;
    validate_library_relative_key(key)?;
    let root = extra_library_paths
        .get(index)
        .ok_or_else(|| anyhow::anyhow!("unknown external library index {index}"))?;
    let path = root.join(key);
    let canonical_root = std::fs::canonicalize(root)
        .map_err(|error| anyhow::anyhow!("library root {}: {error}", root.display()))?;
    let canonical_path = std::fs::canonicalize(&path)
        .map_err(|_| anyhow::anyhow!("external MLX model directory not found for {payload}"))?;
    anyhow::ensure!(
        canonical_path.starts_with(&canonical_root),
        "external model path escapes its library root"
    );
    anyhow::ensure!(
        directory_is_mlx_model(&canonical_path),
        "external MLX model directory is missing config/weights"
    );
    Ok(canonical_path)
}

/// Resolve a `gguf:...` model id to an absolute path under the data directory.
pub fn path_for_gguf_id(data_dir: &Path, key: &str) -> anyhow::Result<PathBuf> {
    validate_gguf_key(key)?;
    let path = gguf_root(data_dir).join(key);
    anyhow::ensure!(
        path.extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf")),
        "model path must end in .gguf"
    );
    Ok(path)
}

fn path_for_mlx_id(data_dir: &Path, key: &str) -> anyhow::Result<PathBuf> {
    validate_repo_id(key)?;
    Ok(mlx_root(data_dir).join(key))
}

pub fn mlx_model_id(engine: MlxKind, repo_id: &str) -> anyhow::Result<String> {
    validate_repo_id(repo_id)?;
    Ok(format!("{}:{repo_id}", engine_prefix(engine)))
}

fn engine_prefix(engine: MlxKind) -> &'static str {
    match engine {
        MlxKind::Lm => "mlx",
        MlxKind::Vlm => "mlx-vlm",
    }
}

pub fn mlx_kind_for_model_id(model_id: &str) -> Option<MlxKind> {
    MlxKind::from_model_id(model_id)
}

/// Best-effort Hugging Face repo id for a local model id.
pub fn managed_repo_id(model_id: &str) -> Option<String> {
    if model_id.starts_with("mlx-ext:")
        || model_id.starts_with("mlx-vlm-ext:")
        || model_id.starts_with("gguf-ext:")
    {
        let key = model_id
            .split_once(':')
            .and_then(|(_, rest)| rest.split_once(':').map(|(_, path)| path))?;
        let parts: Vec<&str> = key.split('/').collect();
        if parts.len() >= 2 && !parts[0].is_empty() && !parts[1].is_empty() {
            return Some(format!("{}/{}", parts[0], parts[1]));
        }
        return None;
    }
    if let Ok(repo_id) = mlx_repo_id(model_id) {
        return Some(repo_id.to_owned());
    }
    if let Some(key) = model_id.strip_prefix("gguf:") {
        let parts: Vec<&str> = key.split('/').collect();
        if parts.len() >= 2 && !parts[0].is_empty() && !parts[1].is_empty() {
            return Some(format!("{}/{}", parts[0], parts[1]));
        }
    }
    None
}

pub fn mlx_repo_id(model_id: &str) -> anyhow::Result<&str> {
    if model_id.starts_with("mlx-ext:") || model_id.starts_with("mlx-vlm-ext:") {
        anyhow::bail!("not a managed MLX repo id: {model_id}");
    }
    let key = model_id
        .strip_prefix("mlx-vlm:")
        .or_else(|| model_id.strip_prefix("mlx:"))
        .ok_or_else(|| anyhow::anyhow!("not an MLX model id: {model_id}"))?;
    validate_repo_id(key)?;
    Ok(key)
}

/// Server argument for an MLX model (local directory or Hugging Face repo id).
pub fn mlx_server_model_ref(
    data_dir: &Path,
    model_id: &str,
    extra_library_paths: &[PathBuf],
) -> anyhow::Result<String> {
    if model_id.starts_with("mlx-ext:") || model_id.starts_with("mlx-vlm-ext:") {
        let local = path_for_model_id(data_dir, model_id, extra_library_paths)?;
        return Ok(local.display().to_string());
    }
    let repo_id = mlx_repo_id(model_id)?;
    let local = mlx_root(data_dir).join(repo_id);
    if local.is_dir() && directory_is_mlx_model(&local) {
        Ok(local.display().to_string())
    } else {
        Ok(repo_id.to_owned())
    }
}

/// Destination path for a Hugging Face GGUF artifact.
pub fn download_destination(
    data_dir: &Path,
    repo_id: &str,
    filename: &str,
) -> anyhow::Result<PathBuf> {
    validate_repo_id(repo_id)?;
    validate_filename(filename)?;
    Ok(gguf_root(data_dir).join(repo_id).join(filename))
}

/// Destination directory for an MLX model snapshot.
pub fn mlx_download_root(data_dir: &Path, repo_id: &str) -> anyhow::Result<PathBuf> {
    validate_repo_id(repo_id)?;
    Ok(mlx_root(data_dir).join(repo_id))
}

/// Destination path for one file inside an MLX snapshot.
pub fn mlx_download_destination(
    data_dir: &Path,
    repo_id: &str,
    filename: &str,
) -> anyhow::Result<PathBuf> {
    validate_repo_id(repo_id)?;
    validate_relative_path(filename)?;
    Ok(mlx_root(data_dir).join(repo_id).join(filename))
}

/// Path safety for a file inside a snapshot: relative, no traversal, no
/// extension rule.
///
/// Snapshot stores (MLX, PersonaPlex, streaming ASR) keep configs, tokenizers,
/// and `.safetensors`, so this is the check they need. [`validate_filename`] is
/// the GGUF store's, and using it on a snapshot rejects every file in it.
pub fn validate_relative_path(path: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !path.is_empty() && path.len() <= 260,
        "invalid relative path"
    );
    anyhow::ensure!(
        !path.starts_with('/') && !path.contains('\\'),
        "path must be relative"
    );
    anyhow::ensure!(
        !path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == ".."),
        "path must not contain empty or parent segments"
    );
    Ok(())
}

pub fn validate_repo_id(repo_id: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !repo_id.is_empty() && repo_id.len() <= 200,
        "invalid repository id"
    );
    let mut parts = repo_id.split('/');
    let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        anyhow::bail!("repository id must be owner/name");
    };
    for part in [owner, name] {
        anyhow::ensure!(
            !part.is_empty()
                && part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')),
            "invalid repository id segment"
        );
        // Dots are legal inside a name (`model.v2`), but a segment of nothing
        // but dots is `.` or `..`, which walks out of the store when the id is
        // joined to it. No Hugging Face owner or name looks like that.
        anyhow::ensure!(
            part.chars().any(|c| c != '.'),
            "repository id segment must not be a path segment"
        );
    }
    Ok(())
}

pub fn validate_filename(filename: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !filename.is_empty() && filename.len() <= 260,
        "invalid filename"
    );
    anyhow::ensure!(
        !filename.starts_with('/') && !filename.contains('\\'),
        "filename must be a relative path"
    );
    anyhow::ensure!(
        !filename
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == ".."),
        "filename must not contain empty or parent path segments"
    );
    anyhow::ensure!(
        filename.to_ascii_lowercase().ends_with(".gguf"),
        "only GGUF artifacts are supported"
    );
    Ok(())
}

fn is_projector(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.to_ascii_lowercase().contains("mmproj"))
}

/// Length of a `-00001-of-00003` shard suffix (llama.cpp split naming).
const SHARD_SUFFIX_LEN: usize = 15;

/// Strip a `-NNNNN-of-NNNNN` suffix from a GGUF filename stem so shards of one
/// quant group together.
pub fn shard_group(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    if let Some(base) = strip_shard_suffix(stem) {
        return base.to_owned();
    }
    stem.to_owned()
}

fn strip_shard_suffix(stem: &str) -> Option<&str> {
    if stem.len() <= SHARD_SUFFIX_LEN {
        return None;
    }
    let tail = &stem[stem.len() - SHARD_SUFFIX_LEN..];
    if tail.starts_with('-')
        && tail[1..6].bytes().all(|byte| byte.is_ascii_digit())
        && &tail[6..10] == "-of-"
        && tail[10..].bytes().all(|byte| byte.is_ascii_digit())
    {
        Some(&stem[..stem.len() - SHARD_SUFFIX_LEN])
    } else {
        None
    }
}

fn is_gguf_shard_name(name: &str) -> bool {
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    strip_shard_suffix(stem).is_some()
}

/// Prefer the first shard (`-00001-of-…`) and sum sizes so split GGUFs appear
/// once in the library.
fn coalesce_gguf_shards(files: Vec<(PathBuf, u64)>) -> Vec<(PathBuf, u64)> {
    let mut groups: std::collections::BTreeMap<String, Vec<(PathBuf, u64)>> =
        std::collections::BTreeMap::new();
    for (path, size) in files {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let key = if is_gguf_shard_name(&name) {
            format!(
                "{}::{}",
                path.parent()
                    .map(|parent| parent.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                shard_group(&name)
            )
        } else {
            // Unique key per unsharded file so they never merge.
            path.to_string_lossy().into_owned()
        };
        groups.entry(key).or_default().push((path, size));
    }
    let mut out = Vec::with_capacity(groups.len());
    for mut shards in groups.into_values() {
        shards.sort_by(|left, right| left.0.cmp(&right.0));
        let total = shards.iter().fold(0u64, |sum, (_, size)| sum.saturating_add(*size));
        let (path, _) = shards.remove(0);
        out.push((path, total));
    }
    out.sort_by(|left, right| left.0.cmp(&right.0));
    out
}

/// Sibling shard files for a first-shard GGUF path (including itself).
fn sibling_gguf_shards(path: &Path) -> Vec<PathBuf> {
    let Some(directory) = path.parent() else {
        return vec![path.to_path_buf()];
    };
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return vec![path.to_path_buf()];
    };
    if !is_gguf_shard_name(name) {
        return vec![path.to_path_buf()];
    }
    let group = shard_group(name);
    let Ok(entries) = std::fs::read_dir(directory) else {
        return vec![path.to_path_buf()];
    };
    let mut siblings: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|candidate| {
            candidate
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
                && candidate
                    .file_name()
                    .and_then(|file| file.to_str())
                    .is_some_and(|file| shard_group(file) == group)
        })
        .collect();
    if siblings.is_empty() {
        siblings.push(path.to_path_buf());
    }
    siblings.sort();
    siblings
}

pub fn is_projector_file(path: &Path) -> bool {
    is_projector(path)
}

pub fn projector_for_model(model_path: &Path) -> Option<PathBuf> {
    let directory = model_path.parent()?;
    std::fs::read_dir(directory)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
                && is_projector(path)
        })
}

fn dir_has_projector(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let path = entry.path();
        path.extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
            && is_projector(&path)
    })
}

fn gguf_capabilities(has_projector: bool, model_key: &str) -> ModelCapabilities {
    let mut input_modalities = vec!["text".into()];
    if has_projector {
        // mmproj enables vision only unless the checkpoint is a known audio LLM.
        input_modalities.push("image".into());
    }
    let native_audio = looks_like_native_audio_model(model_key, None);
    if native_audio {
        if !input_modalities.iter().any(|m| m == "audio") {
            input_modalities.push("audio".into());
        }
    }
    let (reasoning, reasoning_modes) = infer_reasoning_profile(model_key, None);
    ModelCapabilities {
        input_modalities,
        output_modalities: vec!["text".into()],
        streaming: true,
        tools: true,
        reasoning,
        max_context_length: infer_gguf_context_hint(model_key),
        reasoning_modes,
        harmony: crate::harmony::is_harmony_model(model_key),
        audio_input: native_audio.then(|| "native".to_owned()),
    }
}

/// True when the checkpoint is likely a chat model that consumes audio tokens
/// directly (not a Whisper-class ASR weight and not vision-only mmproj).
pub fn looks_like_native_audio_model(model_key: &str, config_text: Option<&str>) -> bool {
    let lower = model_key.to_ascii_lowercase();
    // Exclude dedicated ASR / Whisper weights — those are batch ASR engines.
    if lower.contains("whisper")
        || lower.contains("parakeet")
        || lower.contains("nemotron-speech")
        || lower.contains("nemotron-3.5-asr")
        || lower.contains("nemotron_3.5_asr")
        || lower.contains("canary-") && !lower.contains("canary-qwen")
        || (lower.contains("asr") && !lower.contains("audio"))
    {
        return false;
    }
    const NEEDLES: &[&str] = &[
        "qwen2-audio",
        "qwen2_audio",
        "qwen-audio",
        "qwen_audio",
        "ultravox",
        "mini-omni",
        "mini_omni",
        "salmonn",
        "speech_llm",
        "speech-llm",
        "audio-llm",
        "audio_llm",
        "audioflamingo",
        "audio-flamingo",
        "gamaudio",
        "vita-audio",
        "moshi",
        "kyutai",
    ];
    if NEEDLES.iter().any(|needle| lower.contains(needle)) {
        return true;
    }
    if let Some(config) = config_text {
        let config_lower = config.to_ascii_lowercase();
        if config_lower.contains("\"audio_config\"")
            || config_lower.contains("\"audio_encoder\"")
            || config_lower.contains("qwen2_audio")
            || config_lower.contains("qwen2audio")
            || config_lower.contains("ultravox")
        {
            return true;
        }
    }
    false
}

fn infer_gguf_context_hint(model_key: &str) -> Option<u32> {
    let lower = model_key.to_ascii_lowercase();
    for (needle, context) in [
        ("128k", 131_072),
        ("64k", 65_536),
        ("32k", 32_768),
        ("16k", 16_384),
        ("8k", 8_192),
        ("4k", 4_096),
    ] {
        if lower.contains(needle) {
            return Some(context);
        }
    }
    None
}

fn max_context_from_config(value: &serde_json::Value) -> Option<u32> {
    for pointer in [
        "/max_position_embeddings",
        "/text_config/max_position_embeddings",
        "/sliding_window",
        "/max_seq_len",
    ] {
        if let Some(number) = value.pointer(pointer).and_then(|entry| entry.as_u64()) {
            if number >= 512 {
                return Some(number.min(u32::MAX as u64) as u32);
            }
        }
    }
    None
}

fn infer_reasoning_profile(
    model_key: &str,
    config: Option<&serde_json::Value>,
) -> (bool, Vec<String>) {
    let lower = model_key.to_ascii_lowercase();
    let model_type = config
        .and_then(|value| value.get("model_type"))
        .and_then(|entry| entry.as_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let thinking_model = [
        "qwen3",
        "qwq",
        "deepseek-r1",
        "deepseek_r1",
        "command-r",
        "command_r",
        "phi-4-reasoning",
        "magistral",
    ]
    .iter()
    .any(|needle| lower.contains(needle) || model_type.contains(needle));
    if !thinking_model {
        return (false, Vec::new());
    }
    let budget_supported = lower.contains("qwen3")
        || lower.contains("qwq")
        || model_type.contains("qwen3")
        || model_type.contains("qwq");
    if budget_supported {
        (true, vec!["off".into(), "on".into(), "budget".into()])
    } else {
        (true, vec!["off".into(), "on".into()])
    }
}

/// Scan the GGUF store and return OpenAI-style model descriptors.
pub fn list_gguf_models(data_dir: &Path) -> anyhow::Result<Vec<ModelDescriptor>> {
    let root = gguf_root(data_dir);
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut models = Vec::new();
    collect_gguf(&root, &root, &mut models)?;
    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

pub fn directory_is_mlx_model(dir: &Path) -> bool {
    if !dir.join("config.json").is_file() {
        return false;
    }
    std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .any(|entry| {
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            name.ends_with(".safetensors")
                || name.ends_with(".npz")
                || name == "model.safetensors.index.json"
        })
}

/// Classify a local MLX snapshot as text (mlx-lm) or vision (mlx-vlm).
pub fn detect_mlx_kind(dir: &Path) -> MlxKind {
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return MlxKind::Lm;
    };
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
        if config_value_indicates_vlm(&value) {
            return MlxKind::Vlm;
        }
    }
    let lower = text.to_ascii_lowercase();
    if lower.contains("\"vision_config\"")
        || lower.contains("\"mm_projector\"")
        || lower.contains("\"multi_modal_projector\"")
        || lower.contains("\"image_processor_config\"")
        || lower.contains("\"model_type\": \"llava")
        || lower.contains("\"model_type\": \"paligemma")
        || lower.contains("\"model_type\": \"qwen2_vl")
        || lower.contains("\"model_type\": \"idefics")
        || lower.contains("\"model_type\": \"mllama")
    {
        MlxKind::Vlm
    } else {
        MlxKind::Lm
    }
}

/// Resolve which MLX server to launch for a model, preferring on-disk config over id prefix.
pub fn resolve_mlx_launch_kind(
    data_dir: &Path,
    model_id: &str,
    extra_library_paths: &[PathBuf],
) -> anyhow::Result<(MlxKind, Option<String>)> {
    let id_kind = MlxKind::from_model_id(model_id)
        .ok_or_else(|| anyhow::anyhow!("not an MLX model id: {model_id}"))?;
    let detected = match path_for_model_id(data_dir, model_id, extra_library_paths) {
        Ok(path) if path.is_dir() && directory_is_mlx_model(&path) => detect_mlx_kind(&path),
        _ => id_kind,
    };
    let notice = if detected != id_kind {
        Some(format!(
            "Model id says {} but config indicates {}; using {}.",
            id_kind.engine_id(),
            detected.engine_id(),
            detected.engine_id()
        ))
    } else {
        None
    };
    Ok((detected, notice))
}

fn config_value_indicates_vlm(value: &serde_json::Value) -> bool {
    if value.get("vision_config").is_some()
        || value.get("mm_projector").is_some()
        || value.get("multi_modal_projector").is_some()
        || value.get("image_processor_config").is_some()
    {
        return true;
    }
    if let Some(model_type) = value.get("model_type").and_then(|entry| entry.as_str()) {
        let model_type = model_type.to_ascii_lowercase();
        if model_type.contains("vl")
            || model_type.contains("vision")
            || model_type.contains("llava")
            || model_type.contains("paligemma")
            || model_type.contains("idefics")
            || model_type.contains("mllama")
        {
            return true;
        }
    }
    if let Some(architectures) = value
        .get("architectures")
        .and_then(|entry| entry.as_array())
    {
        if architectures
            .iter()
            .filter_map(|entry| entry.as_str())
            .any(|name| {
                let name = name.to_ascii_lowercase();
                name.contains("vision") || name.contains("vl") || name.contains("llava")
            })
        {
            return true;
        }
    }
    false
}

fn mlx_capabilities(kind: MlxKind, dir: &Path, model_key: &str) -> ModelCapabilities {
    let mut input_modalities = vec!["text".into()];
    if matches!(kind, MlxKind::Vlm) {
        input_modalities.push("image".into());
    }
    let config_text = std::fs::read_to_string(dir.join("config.json")).ok();
    let config_value = config_text
        .as_deref()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok());
    if config_text.as_deref().is_some_and(|text| {
        let lower = text.to_ascii_lowercase();
        lower.contains("vision") || lower.contains("image") || lower.contains("vl")
    }) {
        if !input_modalities.iter().any(|value| value == "image") {
            input_modalities.push("image".into());
        }
    }
    let native_audio = looks_like_native_audio_model(model_key, config_text.as_deref());
    if native_audio && !input_modalities.iter().any(|value| value == "audio") {
        input_modalities.push("audio".into());
    }
    let (reasoning, reasoning_modes) = infer_reasoning_profile(model_key, config_value.as_ref());
    ModelCapabilities {
        input_modalities,
        output_modalities: vec!["text".into()],
        streaming: true,
        tools: true,
        reasoning,
        max_context_length: config_value.as_ref().and_then(max_context_from_config),
        reasoning_modes,
        harmony: crate::harmony::is_harmony_model(model_key),
        audio_input: native_audio.then(|| "native".to_owned()),
    }
}

fn directory_size_bytes(dir: &Path) -> u64 {
    let mut total = 0;
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            total += std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
        }
    }
    total
}

pub fn list_mlx_models(data_dir: &Path) -> anyhow::Result<Vec<ModelDescriptor>> {
    let root = mlx_root(data_dir);
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut models = Vec::new();
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Ok(models);
    };
    for owner_entry in entries.flatten() {
        let owner_path = owner_entry.path();
        if !owner_path.is_dir() {
            continue;
        }
        let owner = owner_entry.file_name().to_string_lossy().into_owned();
        let Ok(name_entries) = std::fs::read_dir(&owner_path) else {
            continue;
        };
        for name_entry in name_entries.flatten() {
            let model_dir = name_entry.path();
            if !model_dir.is_dir() || !directory_is_mlx_model(&model_dir) {
                continue;
            }
            let repo_id = format!("{}/{}", owner, name_entry.file_name().to_string_lossy());
            validate_repo_id(&repo_id).ok();
            let kind = detect_mlx_kind(&model_dir);
            let id = mlx_model_id(kind, &repo_id)
                .unwrap_or_else(|_| format!("{}:{repo_id}", engine_prefix(kind)));
            models.push(ModelDescriptor {
                id,
                name: repo_id.clone(),
                engine: kind.engine_id().to_owned(),
                capabilities: mlx_capabilities(kind, &model_dir, &repo_id),
                size_bytes: Some(directory_size_bytes(&model_dir)),
                read_only: false,
                library_label: None,
            });
        }
    }
    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

/// Scan all on-disk model stores.
pub fn list_local_models(
    data_dir: &Path,
    extra_library_paths: &[PathBuf],
) -> anyhow::Result<Vec<ModelDescriptor>> {
    let mut models = list_gguf_models(data_dir)?;
    let mut seen_paths = models
        .iter()
        .filter_map(|model| path_for_model_id(data_dir, &model.id, extra_library_paths).ok())
        .filter_map(|path| std::fs::canonicalize(path).ok())
        .collect::<HashSet<_>>();
    for model in list_mlx_models(data_dir)? {
        if let Ok(path) = path_for_model_id(data_dir, &model.id, extra_library_paths) {
            if let Ok(canonical) = std::fs::canonicalize(path) {
                seen_paths.insert(canonical);
            }
        }
        models.push(model);
    }
    for model in crate::whisper::list_models(data_dir)? {
        if let Ok(path) = crate::whisper::path_for_model_id(data_dir, &model.id) {
            if let Ok(canonical) = std::fs::canonicalize(path) {
                seen_paths.insert(canonical);
            }
        }
        models.push(model);
    }
    for model in crate::streaming_asr::list_models(data_dir)? {
        if let Ok(path) = crate::streaming_asr::path_for_model_id(data_dir, &model.id) {
            if let Ok(canonical) = std::fs::canonicalize(path) {
                seen_paths.insert(canonical);
            }
        }
        models.push(model);
    }
    for model in crate::sdcpp::list_models(data_dir)? {
        if let Ok(path) = crate::sdcpp::path_for_model_id(data_dir, &model.id) {
            if let Ok(canonical) = std::fs::canonicalize(path) {
                seen_paths.insert(canonical);
            }
        }
        models.push(model);
    }
    for model in crate::voice::list_models(data_dir)? {
        if let Ok(path) = crate::voice::path_for_model_id(data_dir, &model.id) {
            if let Ok(canonical) = std::fs::canonicalize(path) {
                seen_paths.insert(canonical);
            }
        }
        models.push(model);
    }
    for (index, root) in extra_library_paths.iter().enumerate() {
        if !root.is_dir() {
            continue;
        }
        let label = model_library::label_for_library_path(&root.display().to_string());
        list_external_models_from_root(root, index, &label, &mut models, &mut seen_paths)?;
    }
    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

fn list_external_models_from_root(
    root: &Path,
    index: usize,
    label: &str,
    models: &mut Vec<ModelDescriptor>,
    seen_paths: &mut HashSet<PathBuf>,
) -> anyhow::Result<()> {
    collect_external_library(root, root, index, label, models, seen_paths)
}

fn collect_external_library(
    root: &Path,
    dir: &Path,
    index: usize,
    label: &str,
    models: &mut Vec<ModelDescriptor>,
    seen_paths: &mut HashSet<PathBuf>,
) -> anyhow::Result<()> {
    let has_projector = dir_has_projector(dir);
    let entries = std::fs::read_dir(dir)
        .map_err(|error| anyhow::anyhow!("read model directory {}: {error}", dir.display()))?;
    let mut ggufs = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if directory_is_mlx_model(&path) {
                let canonical = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                if !seen_paths.insert(canonical.clone()) {
                    continue;
                }
                let kind = detect_mlx_kind(&path);
                let id = model_id_for_external_mlx(kind, index, root, &canonical)?;
                let name = path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| id.clone());
                models.push(ModelDescriptor {
                    id,
                    name: name.clone(),
                    engine: kind.engine_id().to_owned(),
                    capabilities: mlx_capabilities(kind, &path, &name),
                    size_bytes: Some(directory_size_bytes(&path)),
                    read_only: true,
                    library_label: Some(label.to_owned()),
                });
                continue;
            }
            collect_external_library(root, &path, index, label, models, seen_paths)?;
            continue;
        }
        if !path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            continue;
        }
        if is_projector(&path) {
            continue;
        }
        let size = std::fs::metadata(&path).ok().map(|meta| meta.len()).unwrap_or(0);
        ggufs.push((path, size));
    }
    for (path, size) in coalesce_gguf_shards(ggufs) {
        let canonical = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if !seen_paths.insert(canonical.clone()) {
            continue;
        }
        let id = model_id_for_external_path(index, root, &canonical)?;
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| id.clone());
        let key = id
            .strip_prefix("gguf-ext:")
            .and_then(|payload| payload.split_once(':').map(|(_, path)| path.to_owned()))
            .unwrap_or_else(|| name.clone());
        models.push(ModelDescriptor {
            id,
            name,
            engine: "llama.cpp".to_owned(),
            capabilities: gguf_capabilities(has_projector, &key),
            size_bytes: Some(size),
            read_only: true,
            library_label: Some(label.to_owned()),
        });
    }
    Ok(())
}

fn collect_gguf(root: &Path, dir: &Path, models: &mut Vec<ModelDescriptor>) -> anyhow::Result<()> {
    let has_projector = dir_has_projector(dir);
    let entries = std::fs::read_dir(dir)
        .map_err(|error| anyhow::anyhow!("read model directory {}: {error}", dir.display()))?;
    let mut ggufs = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_gguf(root, &path, models)?;
            continue;
        }
        if !path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            continue;
        }
        if is_projector(&path) {
            continue;
        }
        let size = std::fs::metadata(&path).ok().map(|meta| meta.len()).unwrap_or(0);
        ggufs.push((path, size));
    }
    for (path, size) in coalesce_gguf_shards(ggufs) {
        let id = model_id_for_path(root, &path)?;
        let key = id
            .strip_prefix("gguf:")
            .map(|value| value.to_owned())
            .unwrap_or_else(|| id.clone());
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| id.clone());
        models.push(ModelDescriptor {
            id,
            name,
            engine: "llama.cpp".to_owned(),
            capabilities: gguf_capabilities(has_projector, &key),
            size_bytes: Some(size),
            read_only: false,
            library_label: None,
        });
    }
    Ok(())
}

/// Delete a downloaded model and prune empty parent directories.
pub fn delete_model(
    data_dir: &Path,
    model_id: &str,
    extra_library_paths: &[PathBuf],
) -> anyhow::Result<PathBuf> {
    if model_id.starts_with("gguf-ext:") {
        anyhow::bail!("external library models cannot be deleted from Brazier");
    }
    if model_id.starts_with("mlx-ext:") || model_id.starts_with("mlx-vlm-ext:") {
        anyhow::bail!("external library models cannot be deleted from Brazier");
    }
    if model_id.starts_with("gguf:") {
        let path = path_for_gguf_id(data_dir, model_id.strip_prefix("gguf:").unwrap())?;
        anyhow::ensure!(path.is_file(), "model file not found for {model_id}");
        let siblings = sibling_gguf_shards(&path);
        for sibling in &siblings {
            std::fs::remove_file(sibling)
                .map_err(|error| anyhow::anyhow!("delete {}: {error}", sibling.display()))?;
        }
        prune_empty_parents(path.parent(), &gguf_root(data_dir));
        return Ok(path);
    }
    if model_id.starts_with("mlx:") || model_id.starts_with("mlx-vlm:") {
        let path = path_for_model_id(data_dir, model_id, extra_library_paths)?;
        anyhow::ensure!(path.is_dir(), "model directory not found for {model_id}");
        std::fs::remove_dir_all(&path)
            .map_err(|error| anyhow::anyhow!("delete {}: {error}", path.display()))?;
        prune_empty_parents(path.parent(), &mlx_root(data_dir));
        return Ok(path);
    }
    if model_id.starts_with("streaming-asr:") {
        let path = crate::streaming_asr::path_for_model_id(data_dir, model_id)?;
        anyhow::ensure!(path.is_dir(), "model directory not found for {model_id}");
        std::fs::remove_dir_all(&path)
            .map_err(|error| anyhow::anyhow!("delete {}: {error}", path.display()))?;
        prune_empty_parents(path.parent(), &crate::streaming_asr::models_root(data_dir));
        return Ok(path);
    }
    anyhow::bail!("unknown local model id: {model_id}");
}

fn prune_empty_parents(mut directory: Option<&Path>, root: &Path) {
    while let Some(current) = directory {
        if current == root || !current.starts_with(root) {
            break;
        }
        let empty = std::fs::read_dir(current)
            .map(|mut entries| entries.next().is_none())
            .unwrap_or(false);
        if !empty {
            break;
        }
        let _ = std::fs::remove_dir(current);
        directory = current.parent();
    }
}

/// Prefer a practical default quant from a list of GGUF filenames.
pub fn prefer_gguf_filename(filenames: &[String]) -> Option<String> {
    let ggufs: Vec<&String> = filenames
        .iter()
        .filter(|name| name.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    if ggufs.is_empty() {
        return None;
    }
    const PREFERRED: &[&str] = &[
        "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q4_0", "q5_0", "q3_k_m", "q6_k", "q8_0",
    ];
    for quant in PREFERRED {
        if let Some(name) = ggufs
            .iter()
            .find(|name| name.to_ascii_lowercase().contains(quant))
        {
            return Some((*name).clone());
        }
    }
    ggufs.into_iter().min_by_key(|name| name.len()).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn model_id_round_trips_through_path() {
        let dir = tempdir().unwrap();
        let root = gguf_root(dir.path());
        let file = root
            .join("unsloth")
            .join("Tiny-GGUF")
            .join("model-Q4_K_M.gguf");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"gguf").unwrap();
        let id = model_id_for_path(&root, &file).unwrap();
        assert_eq!(id, "gguf:unsloth/Tiny-GGUF/model-Q4_K_M.gguf");
        let resolved = path_for_model_id(dir.path(), &id, &[]).unwrap();
        assert_eq!(resolved, file);
    }

    #[test]
    fn lists_nested_gguf_files() {
        let dir = tempdir().unwrap();
        let file = download_destination(dir.path(), "acme/demo", "demo-q4_k_m.gguf").unwrap();
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"gguf").unwrap();
        let models = list_gguf_models(dir.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].engine, "llama.cpp");
        assert!(models[0].capabilities.streaming);
        assert_eq!(models[0].id, "gguf:acme/demo/demo-q4_k_m.gguf");
    }

    #[test]
    fn coalesces_split_gguf_shards_into_one_model() {
        let dir = tempdir().unwrap();
        let first =
            download_destination(dir.path(), "acme/big", "model-Q4_K_M-00001-of-00002.gguf")
                .unwrap();
        let second =
            download_destination(dir.path(), "acme/big", "model-Q4_K_M-00002-of-00002.gguf")
                .unwrap();
        std::fs::create_dir_all(first.parent().unwrap()).unwrap();
        std::fs::write(&first, vec![0u8; 40]).unwrap();
        std::fs::write(&second, vec![0u8; 33]).unwrap();
        let models = list_gguf_models(dir.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(
            models[0].id,
            "gguf:acme/big/model-Q4_K_M-00001-of-00002.gguf"
        );
        assert_eq!(models[0].size_bytes, Some(73));
    }

    #[test]
    fn deleting_first_shard_removes_the_whole_split_set() {
        let dir = tempdir().unwrap();
        let first =
            download_destination(dir.path(), "acme/big", "model-Q4_K_M-00001-of-00002.gguf")
                .unwrap();
        let second =
            download_destination(dir.path(), "acme/big", "model-Q4_K_M-00002-of-00002.gguf")
                .unwrap();
        std::fs::create_dir_all(first.parent().unwrap()).unwrap();
        std::fs::write(&first, b"a").unwrap();
        std::fs::write(&second, b"b").unwrap();
        delete_model(
            dir.path(),
            "gguf:acme/big/model-Q4_K_M-00001-of-00002.gguf",
            &[],
        )
        .unwrap();
        assert!(!first.exists());
        assert!(!second.exists());
    }

    #[test]
    fn projector_enables_multimodal_capabilities_without_becoming_a_model() {
        let dir = tempdir().unwrap();
        let model = download_destination(dir.path(), "acme/vision", "model-q4.gguf").unwrap();
        let projector = download_destination(dir.path(), "acme/vision", "mmproj-f16.gguf").unwrap();
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&projector, b"projector").unwrap();
        let models = list_gguf_models(dir.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(projector_for_model(&model), Some(projector));
        assert!(
            models[0]
                .capabilities
                .input_modalities
                .contains(&"image".to_owned())
        );
        assert!(
            !models[0]
                .capabilities
                .input_modalities
                .contains(&"audio".to_owned())
        );
        assert!(models[0].capabilities.audio_input.is_none());
    }

    #[test]
    fn detects_native_audio_llms_not_whisper_asr() {
        assert!(looks_like_native_audio_model(
            "Qwen2-Audio-7B-Instruct",
            None
        ));
        assert!(looks_like_native_audio_model("ultravox-v0.5", None));
        assert!(!looks_like_native_audio_model(
            "ggml-whisper-large-v3",
            None
        ));
        assert!(!looks_like_native_audio_model(
            "nemotron-3.5-asr-streaming-0.6b",
            None
        ));
        assert!(!looks_like_native_audio_model("ordinary-chat-q4_k_m", None));
    }

    #[test]
    fn rejects_path_traversal_in_model_id() {
        let dir = tempdir().unwrap();
        assert!(path_for_model_id(dir.path(), "gguf:../escape.gguf", &[]).is_err());
        assert!(path_for_model_id(dir.path(), "gguf:/abs.gguf", &[]).is_err());
        assert!(validate_filename("../x.gguf").is_err());
        assert!(validate_filename("ok/nested-q4_k_m.gguf").is_ok());
        assert!(validate_repo_id("../evil/name").is_err());
        // Two segments passes the owner/name shape, and `.` matched the allowed
        // character set, so these walked out of the store when joined to it.
        assert!(validate_repo_id("../evil").is_err());
        assert!(validate_repo_id("../..").is_err());
        assert!(validate_repo_id("./x").is_err());
        // Dots inside a name are ordinary and stay allowed.
        assert!(validate_repo_id("owner/model.v2").is_ok());
        assert!(validate_repo_id("mlx-community/nemotron-3.5-asr-streaming-0.6b").is_ok());
    }

    #[test]
    fn snapshot_files_are_not_held_to_the_gguf_rule() {
        // The GGUF store's filename rule and the snapshot stores' path rule are
        // different checks; using the former on a snapshot rejects every file.
        assert!(validate_filename("config.json").is_err());
        assert!(validate_relative_path("config.json").is_ok());
        assert!(validate_relative_path("model.safetensors").is_ok());
        assert!(validate_relative_path("../escape").is_err());
        assert!(validate_relative_path("/abs").is_err());
    }

    #[test]
    fn prefers_balanced_quants() {
        let names = vec![
            "model-f16.gguf".into(),
            "model-q8_0.gguf".into(),
            "model-q4_k_m.gguf".into(),
            "readme.md".into(),
        ];
        assert_eq!(
            prefer_gguf_filename(&names).as_deref(),
            Some("model-q4_k_m.gguf")
        );
    }

    #[test]
    fn infers_thinking_model_reasoning_modes() {
        let (_, modes) = infer_reasoning_profile("mlx-community/Qwen3-8B", None);
        assert!(modes.contains(&"budget".into()));
        let (_, modes) = infer_reasoning_profile("acme/Llama-3", None);
        assert!(modes.is_empty());
    }

    #[test]
    fn managed_repo_id_extracts_hf_repo() {
        assert_eq!(
            managed_repo_id("mlx:mlx-community/Qwen2.5-0.5B-Instruct"),
            Some("mlx-community/Qwen2.5-0.5B-Instruct".into())
        );
        assert_eq!(
            managed_repo_id("gguf:unsloth/Tiny-GGUF/model-Q4_K_M.gguf"),
            Some("unsloth/Tiny-GGUF".into())
        );
        assert_eq!(
            managed_repo_id("gguf-ext:0:owner/repo/file.gguf"),
            Some("owner/repo".into())
        );
    }

    #[test]
    fn resolve_mlx_launch_kind_prefers_config() {
        let dir = tempdir().unwrap();
        let root = mlx_root(dir.path());
        let model = root.join("acme/vision");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(
            model.join("config.json"),
            r#"{"model_type":"llava","vision_config":{"hidden_size":1024}}"#,
        )
        .unwrap();
        std::fs::write(model.join("weights.safetensors"), b"mlx").unwrap();
        let (kind, notice) = resolve_mlx_launch_kind(dir.path(), "mlx:acme/vision", &[]).unwrap();
        assert_eq!(kind, MlxKind::Vlm);
        assert!(notice.is_some());
    }

    #[test]
    fn detect_mlx_kind_reads_config() {
        let dir = tempdir().unwrap();
        let text = dir.path().join("text");
        std::fs::create_dir_all(&text).unwrap();
        std::fs::write(text.join("config.json"), r#"{"model_type":"qwen2"}"#).unwrap();
        std::fs::write(text.join("weights.safetensors"), b"mlx").unwrap();
        assert_eq!(detect_mlx_kind(&text), MlxKind::Lm);

        let vision = dir.path().join("vision");
        std::fs::create_dir_all(&vision).unwrap();
        std::fs::write(
            vision.join("config.json"),
            r#"{"model_type":"llava","vision_config":{"hidden_size":1024}}"#,
        )
        .unwrap();
        std::fs::write(vision.join("weights.safetensors"), b"mlx").unwrap();
        assert_eq!(detect_mlx_kind(&vision), MlxKind::Vlm);
    }

    #[test]
    fn lists_mlx_models_with_correct_engine() {
        let dir = tempdir().unwrap();
        let root = mlx_root(dir.path());
        let text = root.join("acme/text");
        std::fs::create_dir_all(&text).unwrap();
        std::fs::write(text.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        std::fs::write(text.join("model.safetensors"), b"mlx").unwrap();
        let models = list_mlx_models(dir.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "mlx:acme/text");
        assert_eq!(models[0].engine, "mlx-lm");
    }

    #[test]
    fn lists_external_mlx_model_dirs() {
        let dir = tempdir().unwrap();
        let external = dir.path().join("external");
        let model_dir = external.join("vendor/my-mlx");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        std::fs::write(model_dir.join("weights.safetensors"), b"mlx").unwrap();
        let extra = vec![external.clone()];
        let models = list_local_models(dir.path(), &extra).unwrap();
        assert_eq!(models.len(), 1);
        assert!(models[0].id.starts_with("mlx-ext:0:"));
        assert_eq!(models[0].engine, "mlx-lm");
        assert!(models[0].read_only);
    }

    #[test]
    fn lists_external_library_gguf_files() {
        let dir = tempdir().unwrap();
        let external = dir.path().join("external");
        let file = external.join("vendor/model-q4_k_m.gguf");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"gguf").unwrap();
        let extra = vec![external.clone()];
        let models = list_local_models(dir.path(), &extra).unwrap();
        assert_eq!(models.len(), 1);
        assert!(models[0].id.starts_with("gguf-ext:0:"));
        assert!(models[0].read_only);
        assert_eq!(
            path_for_model_id(dir.path(), &models[0].id, &extra).unwrap(),
            std::fs::canonicalize(&file).unwrap()
        );
    }

    #[test]
    fn dedupes_when_library_path_overlaps_primary_store() {
        let dir = tempdir().unwrap();
        let root = gguf_root(dir.path());
        let file = root.join("shared/model.gguf");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"gguf").unwrap();
        let models = list_local_models(dir.path(), &[root.clone()]).unwrap();
        assert_eq!(models.len(), 1);
        assert!(models[0].id.starts_with("gguf:"));
        assert!(!models[0].read_only);
    }
}

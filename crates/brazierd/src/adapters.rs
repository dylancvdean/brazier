//! LoRA adapters and ControlNets: where they live, what can load them, and
//! how one is handed to an engine.
//!
//! Adapters are not models. They are small files that modify a model already
//! installed, and every engine takes them differently: llama.cpp wants a GGUF
//! LoRA passed with a scale, MLX wants a directory of adapter weights, and
//! stable-diffusion.cpp wants a directory to search plus a tag written into the
//! prompt. What they share is the question the interface has to answer — which
//! of the installed adapters can this model actually use — so they are catalogued
//! together and each one carries the engines that can load it.
//!
//! Managed adapters live under `<data>/models/adapters/{lora,controlnet}`.
//! Anything already on disk elsewhere can be registered by path instead of
//! copied, because a LoRA collection is usually shared with another tool.

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use anyhow::Context;
use serde::{Deserialize, Serialize};

/// Extensions a single-file adapter is recognised by.
const WEIGHT_EXTENSIONS: [&str; 6] = ["safetensors", "gguf", "ckpt", "pt", "pth", "bin"];

/// How deep a scan descends. Collections are organised by author or base model,
/// rarely deeper, and an unbounded walk over a shared folder is a stall.
const MAX_SCAN_DEPTH: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterKind {
    Lora,
    ControlNet,
}

impl AdapterKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lora => "lora",
            Self::ControlNet => "controlnet",
        }
    }

    fn id_prefix(self) -> &'static str {
        self.as_str()
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "lora" => Some(Self::Lora),
            "controlnet" | "control_net" => Some(Self::ControlNet),
            _ => None,
        }
    }
}

/// One installed adapter, with everything the interface needs to offer it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterDescriptor {
    /// `lora:<relative path>` for managed files, `lora-ext:<digest>` for
    /// registered ones. Stable across restarts either way.
    pub id: String,
    pub kind: AdapterKind,
    /// What to call it in a list — the file stem, or the directory name.
    pub name: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Engines that can load this file, derived from its shape. An engine that
    /// is not listed here would fail at load time, so it is not offered.
    pub engines: Vec<String>,
    /// Registered from outside the managed root, so deleting it is not ours.
    pub external: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_repo: Option<String>,
}

/// An adapter registered from a path Brazier does not own.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegisteredAdapter {
    kind: AdapterKind,
    path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_repo: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct RegistrationFile {
    entries: Vec<RegisteredAdapter>,
}

pub fn adapters_root(data_dir: &Path) -> PathBuf {
    data_dir.join("models").join("adapters")
}

pub fn root_for(data_dir: &Path, kind: AdapterKind) -> PathBuf {
    adapters_root(data_dir).join(kind.as_str())
}

fn registrations_path(data_dir: &Path) -> PathBuf {
    data_dir.join("adapters.json")
}

fn load_registrations(data_dir: &Path) -> RegistrationFile {
    let path = registrations_path(data_dir);
    let Ok(bytes) = std::fs::read(&path) else {
        return RegistrationFile::default();
    };
    serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        tracing::warn!(%error, path = %path.display(), "ignoring invalid adapter registrations");
        RegistrationFile::default()
    })
}

async fn save_registrations(data_dir: &Path, file: &RegistrationFile) -> anyhow::Result<()> {
    let path = registrations_path(data_dir);
    let temporary = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(file).context("encode adapter registrations")?;
    tokio::fs::write(&temporary, bytes)
        .await
        .context("write adapter registrations")?;
    tokio::fs::rename(&temporary, &path)
        .await
        .context("commit adapter registrations")?;
    Ok(())
}

/// Whether a directory holds MLX adapter weights rather than being a folder of
/// separate adapters.
fn is_mlx_adapter_dir(path: &Path) -> bool {
    path.join("adapter_config.json").is_file() || path.join("adapters.safetensors").is_file()
}

fn has_weight_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .is_some_and(|value| WEIGHT_EXTENSIONS.contains(&value.as_str()))
}

/// Engines that can load this adapter, from the shape of what is on disk.
///
/// The distinction is not cosmetic: llama.cpp only accepts a GGUF LoRA, MLX
/// only a directory of adapter weights, and stable-diffusion.cpp only the
/// safetensors/ckpt files the diffusion ecosystem publishes. Offering the wrong
/// one produces a load failure several minutes into a job.
pub fn engines_for(kind: AdapterKind, path: &Path) -> Vec<String> {
    if kind == AdapterKind::ControlNet {
        return vec![crate::sdcpp::ENGINE.to_owned()];
    }
    if path.is_dir() {
        return vec!["mlx-lm".to_owned(), "mlx-vlm".to_owned()];
    }
    match path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("gguf") => vec![crate::runtimes::ENGINE.to_owned()],
        Some(_) => vec![crate::sdcpp::ENGINE.to_owned()],
        None => Vec::new(),
    }
}

fn display_name(path: &Path) -> String {
    if path.is_dir() {
        return path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("adapter")
            .to_owned();
    }
    path.file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("adapter")
        .to_owned()
}

fn managed_id(kind: AdapterKind, root: &Path, path: &Path) -> String {
    let relative = path.strip_prefix(root).unwrap_or(path);
    format!(
        "{}:{}",
        kind.id_prefix(),
        relative.to_string_lossy().replace('\\', "/")
    )
}

fn external_id(kind: AdapterKind, path: &Path) -> String {
    let digest = crate::download::sha256_hex(path.to_string_lossy().as_bytes());
    format!("{}-ext:{}", kind.id_prefix(), &digest[..12])
}

fn size_of(path: &Path) -> Option<u64> {
    if path.is_dir() {
        let mut total = 0_u64;
        for entry in std::fs::read_dir(path).ok()?.flatten() {
            if let Ok(meta) = entry.metadata()
                && meta.is_file()
            {
                total = total.saturating_add(meta.len());
            }
        }
        return Some(total);
    }
    std::fs::metadata(path).ok().map(|meta| meta.len())
}

fn describe(kind: AdapterKind, path: &Path, id: String, external: bool) -> AdapterDescriptor {
    AdapterDescriptor {
        id,
        kind,
        name: display_name(path),
        path: path.display().to_string(),
        size_bytes: size_of(path),
        engines: engines_for(kind, path),
        external,
        source_repo: None,
    }
}

/// Walk one managed root, collecting single-file adapters and MLX adapter
/// directories. An MLX directory is one adapter, not a folder to descend into.
fn scan_root(
    kind: AdapterKind,
    root: &Path,
    dir: &Path,
    depth: usize,
    out: &mut Vec<AdapterDescriptor>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if is_mlx_adapter_dir(&path) {
                out.push(describe(kind, &path, managed_id(kind, root, &path), false));
            } else if depth + 1 < MAX_SCAN_DEPTH {
                scan_root(kind, root, &path, depth + 1, out);
            }
            continue;
        }
        if has_weight_extension(&path) {
            out.push(describe(kind, &path, managed_id(kind, root, &path), false));
        }
    }
}

/// Every adapter Brazier can offer, managed and registered, sorted by name.
///
/// Entries whose files have gone are dropped rather than listed as broken: a
/// registration outliving the folder it pointed at is the common case when a
/// collection is shared with another tool.
pub fn list(data_dir: &Path) -> Vec<AdapterDescriptor> {
    let mut out = Vec::new();
    for kind in [AdapterKind::Lora, AdapterKind::ControlNet] {
        let root = root_for(data_dir, kind);
        if root.is_dir() {
            scan_root(kind, &root, &root, 0, &mut out);
        }
    }
    let mut seen: BTreeSet<String> = out.iter().map(|entry| entry.path.clone()).collect();
    for registered in load_registrations(data_dir).entries {
        let path = PathBuf::from(&registered.path);
        if !path.exists() || !seen.insert(registered.path.clone()) {
            continue;
        }
        let mut descriptor = describe(
            registered.kind,
            &path,
            external_id(registered.kind, &path),
            true,
        );
        if let Some(name) = registered.name {
            descriptor.name = name;
        }
        descriptor.source_repo = registered.source_repo;
        out.push(descriptor);
    }
    out.sort_by(|left, right| {
        (left.kind, left.name.to_lowercase()).cmp(&(right.kind, right.name.to_lowercase()))
    });
    out
}

pub fn find(data_dir: &Path, id: &str) -> Option<AdapterDescriptor> {
    list(data_dir).into_iter().find(|entry| entry.id == id)
}

/// Resolve an id or an absolute path to a usable file.
///
/// Bindings store the path, so a settings file written before an adapter moved
/// still resolves through the catalogue by id when the path no longer exists.
pub fn resolve_path(data_dir: &Path, id: Option<&str>, path: Option<&str>) -> Option<PathBuf> {
    if let Some(path) = path {
        let candidate = PathBuf::from(path);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    let id = id?;
    find(data_dir, id).map(|entry| PathBuf::from(entry.path))
}

/// Record an adapter that lives outside the managed root.
pub async fn register(
    data_dir: &Path,
    kind: AdapterKind,
    path: &Path,
    name: Option<String>,
) -> anyhow::Result<AdapterDescriptor> {
    anyhow::ensure!(path.is_absolute(), "adapter path must be absolute");
    anyhow::ensure!(path.exists(), "adapter not found: {}", path.display());
    anyhow::ensure!(
        path.is_dir() || has_weight_extension(path),
        "{} is not a recognised adapter file (expected {})",
        path.display(),
        WEIGHT_EXTENSIONS.join(", ")
    );
    let engines = engines_for(kind, path);
    anyhow::ensure!(!engines.is_empty(), "no engine can load {}", path.display());

    let mut file = load_registrations(data_dir);
    let stored = path.display().to_string();
    if let Some(existing) = file.entries.iter_mut().find(|entry| entry.path == stored) {
        existing.kind = kind;
        existing.name = name.clone();
    } else {
        file.entries.push(RegisteredAdapter {
            kind,
            path: stored,
            name: name.clone(),
            source_repo: None,
        });
    }
    save_registrations(data_dir, &file).await?;

    let mut descriptor = describe(kind, path, external_id(kind, path), true);
    if let Some(name) = name {
        descriptor.name = name;
    }
    Ok(descriptor)
}

/// Drop a registration. The file itself is left alone — it was never ours.
pub async fn forget(data_dir: &Path, id: &str) -> anyhow::Result<()> {
    let Some(entry) = find(data_dir, id) else {
        anyhow::bail!("unknown adapter `{id}`");
    };
    anyhow::ensure!(
        entry.external,
        "`{id}` is installed in the adapter library; delete it instead of forgetting it"
    );
    let mut file = load_registrations(data_dir);
    file.entries
        .retain(|registered| registered.path != entry.path);
    save_registrations(data_dir, &file).await
}

/// Delete a managed adapter from disk.
pub async fn delete(data_dir: &Path, id: &str) -> anyhow::Result<()> {
    let Some(entry) = find(data_dir, id) else {
        anyhow::bail!("unknown adapter `{id}`");
    };
    anyhow::ensure!(
        !entry.external,
        "`{id}` lives outside the adapter library; remove it there, or forget it here"
    );
    let path = PathBuf::from(&entry.path);
    // Confined to the managed root, so a crafted id cannot reach past it.
    anyhow::ensure!(
        path.starts_with(adapters_root(data_dir)),
        "refusing to delete outside the adapter library"
    );
    if path.is_dir() {
        tokio::fs::remove_dir_all(&path).await
    } else {
        tokio::fs::remove_file(&path).await
    }
    .with_context(|| format!("delete {}", path.display()))
}

fn validate_component(value: &str, what: &str) -> anyhow::Result<()> {
    anyhow::ensure!(!value.is_empty() && value.len() <= 260, "invalid {what}");
    anyhow::ensure!(
        !value.starts_with('/') && !value.contains('\\'),
        "{what} must be a relative path"
    );
    anyhow::ensure!(
        !value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == ".."),
        "{what} must not contain empty or parent path segments"
    );
    Ok(())
}

/// Where a downloaded adapter lands: `<root>/<owner>/<repo>/<file name>`.
pub fn download_destination(
    data_dir: &Path,
    kind: AdapterKind,
    repo_id: &str,
    filename: &str,
) -> anyhow::Result<PathBuf> {
    validate_component(repo_id, "repository id")?;
    validate_component(filename, "file name")?;
    let file_name = filename.rsplit('/').next().unwrap_or(filename);
    Ok(root_for(data_dir, kind).join(repo_id).join(file_name))
}

/// Gather LoRA files into one directory for stable-diffusion.cpp.
///
/// sd-cli searches a single `--lora-model-dir` and matches `<lora:name:scale>`
/// tags against the file names it finds there, so adapters chosen from
/// different folders have to be brought together first. Links are used where
/// the platform allows them and the bytes are copied where it does not, which
/// keeps a staged set cheap enough to rebuild per job.
pub async fn stage_lora_dir(data_dir: &Path, paths: &[PathBuf]) -> anyhow::Result<PathBuf> {
    let mut digest_input = String::new();
    for path in paths {
        digest_input.push_str(&path.to_string_lossy());
        digest_input.push('\n');
    }
    let digest = crate::download::sha256_hex(digest_input.as_bytes());
    let dir = data_dir.join("tmp").join("sdcpp-loras").join(&digest[..16]);
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("create {}", dir.display()))?;
    for path in paths {
        let Some(name) = path.file_name() else {
            continue;
        };
        let link = dir.join(name);
        if link.exists() {
            continue;
        }
        if link_file(path, &link).is_err() {
            tokio::fs::copy(path, &link)
                .await
                .with_context(|| format!("stage {}", path.display()))?;
        }
    }
    Ok(dir)
}

#[cfg(unix)]
fn link_file(source: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(source, link)
}

#[cfg(windows)]
fn link_file(source: &Path, link: &Path) -> std::io::Result<()> {
    std::fs::hard_link(source, link)
}

#[cfg(not(any(unix, windows)))]
fn link_file(_source: &Path, _link: &Path) -> std::io::Result<()> {
    Err(std::io::Error::other("links unsupported on this platform"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"weights").unwrap();
    }

    #[test]
    fn lists_managed_adapters_by_shape() {
        let dir = tempfile::tempdir().unwrap();
        let lora = root_for(dir.path(), AdapterKind::Lora);
        touch(&lora.join("acme/detail.safetensors"));
        touch(&lora.join("acme/style.gguf"));
        touch(&lora.join("mlx-tune/adapters.safetensors"));
        touch(&root_for(dir.path(), AdapterKind::ControlNet).join("canny.safetensors"));

        let entries = list(dir.path());
        let by_name = |name: &str| {
            entries
                .iter()
                .find(|entry| entry.name == name)
                .unwrap_or_else(|| panic!("missing {name}"))
                .clone()
        };
        assert_eq!(by_name("detail").engines, vec!["stable-diffusion.cpp"]);
        assert_eq!(by_name("style").engines, vec!["llama.cpp"]);
        assert_eq!(by_name("mlx-tune").engines, vec!["mlx-lm", "mlx-vlm"]);
        assert_eq!(by_name("canny").kind, AdapterKind::ControlNet);
        assert!(entries.iter().all(|entry| !entry.external));
    }

    /// A collection shared with another tool stays where it is; Brazier only
    /// remembers where to look.
    #[tokio::test]
    async fn registers_and_forgets_an_external_adapter() {
        let dir = tempfile::tempdir().unwrap();
        let elsewhere = dir.path().join("elsewhere/film-grain.safetensors");
        touch(&elsewhere);

        let registered = register(dir.path(), AdapterKind::Lora, &elsewhere, None)
            .await
            .unwrap();
        assert!(registered.external);
        assert_eq!(list(dir.path()).len(), 1);

        forget(dir.path(), &registered.id).await.unwrap();
        assert!(list(dir.path()).is_empty());
        assert!(elsewhere.is_file(), "forgetting must not delete the file");
    }

    #[tokio::test]
    async fn delete_refuses_an_external_adapter() {
        let dir = tempfile::tempdir().unwrap();
        let elsewhere = dir.path().join("elsewhere/film-grain.safetensors");
        touch(&elsewhere);
        let registered = register(dir.path(), AdapterKind::Lora, &elsewhere, None)
            .await
            .unwrap();

        assert!(delete(dir.path(), &registered.id).await.is_err());
        assert!(elsewhere.is_file());
    }

    /// sd-cli reads one directory, so a set chosen from several has to be
    /// collected before the job can name them in the prompt.
    #[tokio::test]
    async fn stages_loras_from_several_folders_into_one() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("a/one.safetensors");
        let second = dir.path().join("b/two.safetensors");
        touch(&first);
        touch(&second);

        let staged = stage_lora_dir(dir.path(), &[first, second]).await.unwrap();
        assert!(staged.join("one.safetensors").exists());
        assert!(staged.join("two.safetensors").exists());
    }

    #[test]
    fn download_destination_rejects_traversal() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            download_destination(dir.path(), AdapterKind::Lora, "acme/../..", "x.safetensors")
                .is_err()
        );
    }
}

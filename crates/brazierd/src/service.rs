//! Durable state for a deliberately long-lived daemon.
//!
//! The desktop app starts a disposable loopback daemon, whose random key can
//! disappear with the process. A headless service is different: a restart
//! must not strand its clients with a new endpoint or credential. This module
//! owns only that small service contract; pairing and scoped client credentials
//! remain a later remote-access layer.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use serde::Serialize;
use uuid::Uuid;

/// The predictable port used by `brazierd --service` when `--port` is omitted.
pub const DEFAULT_SERVICE_PORT: u16 = 7614;

const KEY_FILE: &str = "service/api-key";
const READY_FILE: &str = "service/ready.json";

#[derive(Debug, Serialize)]
pub struct ReadyDescriptor<'a> {
    pub address: &'a str,
    pub pid: u32,
    pub product: &'static str,
    pub version: &'static str,
    pub management_api: ManagementApiVersion,
}

#[derive(Debug, Serialize)]
pub struct ManagementApiVersion {
    pub major: u8,
    pub minor: u8,
}

pub fn api_key_path(data_dir: &Path) -> PathBuf {
    data_dir.join(KEY_FILE)
}

pub fn ready_path(data_dir: &Path) -> PathBuf {
    data_dir.join(READY_FILE)
}

/// Return the service key already assigned to this data directory, or create
/// it with owner-only permissions. An explicitly supplied key is intentionally
/// not copied into this file: deployment tooling remains its source of truth.
pub fn service_api_key(data_dir: &Path, configured: Option<String>) -> anyhow::Result<String> {
    if let Some(key) = configured {
        anyhow::ensure!(!key.trim().is_empty(), "the API key must not be empty");
        return Ok(key);
    }

    let path = api_key_path(data_dir);
    if path.exists() {
        ensure_private(&path)?;
        let key = fs::read_to_string(&path)
            .with_context(|| format!("read service API key at {}", path.display()))?;
        let key = key.trim().to_owned();
        anyhow::ensure!(
            !key.is_empty(),
            "service API key at {} is empty",
            path.display()
        );
        return Ok(key);
    }

    let parent = path.parent().expect("service key has a parent");
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let key = format!("brazier_{}", Uuid::new_v4().simple());
    write_new_private(&path, key.as_bytes())?;
    // `create_new` may have lost a concurrent-start race. Re-read the winner
    // so every process uses the same durable credential.
    let stored = fs::read_to_string(&path)
        .with_context(|| format!("read service API key at {}", path.display()))?;
    let stored = stored.trim().to_owned();
    anyhow::ensure!(
        !stored.is_empty(),
        "service API key at {} is empty",
        path.display()
    );
    Ok(stored)
}

fn write_new_private(path: &Path, contents: &[u8]) -> anyhow::Result<()> {
    use std::io::Write as _;
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    match options.open(path) {
        Ok(mut file) => {
            file.write_all(contents)
                .with_context(|| format!("write service API key at {}", path.display()))?;
            Ok(())
        }
        // A concurrent service startup created it first. Treat its key as the
        // authoritative one instead of giving the two processes different keys.
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            ensure_private(path)?;
            Ok(())
        }
        Err(error) => {
            Err(error).with_context(|| format!("create service API key at {}", path.display()))
        }
    }
}

fn ensure_private(path: &Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = fs::metadata(path)
            .with_context(|| format!("inspect service API key at {}", path.display()))?
            .permissions()
            .mode()
            & 0o777;
        anyhow::ensure!(
            mode & 0o077 == 0,
            "refusing service API key at {}: it is readable by group or others",
            path.display()
        );
    }
    Ok(())
}

/// Atomically publish the endpoint only after the listener has bound. It has
/// no bearer credential, but remains owner-only operational metadata on Unix.
pub fn write_ready_descriptor(path: &Path, address: &str) -> anyhow::Result<()> {
    let descriptor = ReadyDescriptor {
        address,
        pid: std::process::id(),
        product: "brazier",
        version: env!("CARGO_PKG_VERSION"),
        management_api: ManagementApiVersion { major: 1, minor: 0 },
    };
    let bytes =
        serde_json::to_vec_pretty(&descriptor).context("encode service ready descriptor")?;
    let parent = path
        .parent()
        .context("service ready descriptor path has no parent")?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let temporary = parent.join(format!(".ready-{}.tmp", Uuid::new_v4()));
    write_new_private(&temporary, &bytes)?;
    #[cfg(windows)]
    if path.exists() {
        fs::remove_file(path).with_context(|| format!("replace {}", path.display()))?;
    }
    fs::rename(&temporary, path).with_context(|| format!("publish {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_key_survives_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let first = service_api_key(dir.path(), None).unwrap();
        let second = service_api_key(dir.path(), None).unwrap();
        assert_eq!(first, second);
        assert!(first.starts_with("brazier_"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                fs::metadata(api_key_path(dir.path()))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn explicit_key_is_not_persisted() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            service_api_key(dir.path(), Some("from-secret-store".into())).unwrap(),
            "from-secret-store"
        );
        assert!(!api_key_path(dir.path()).exists());
    }

    #[test]
    fn ready_descriptor_has_no_credential_and_is_private() {
        let dir = tempfile::tempdir().unwrap();
        let path = ready_path(dir.path());
        write_ready_descriptor(&path, "http://127.0.0.1:7614").unwrap();
        let json: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(json["address"], "http://127.0.0.1:7614");
        assert_eq!(json["management_api"]["major"], 1);
        assert!(json.get("api_key").is_none());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn refuses_an_exposed_existing_key() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = tempfile::tempdir().unwrap();
        let path = api_key_path(dir.path());
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "secret").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(service_api_key(dir.path(), None).is_err());
    }
}

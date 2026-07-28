//! Privacy-conscious diagnostics that can be attached to a support request.
//!
//! The bundle is deliberately assembled from a small allow-list. It never
//! reads conversations, messages, model output, attachment blobs, credentials,
//! or daemon logs. A second recursive pass redacts secret-shaped fields and
//! removes identifying prefixes from paths before anything is serialized.

use std::{
    io::{Cursor, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use serde_json::{Map, Value, json};
use zip::{CompressionMethod, ZipWriter, write::SimpleFileOptions};

use crate::{AppState, hardware, remote, runtimes, toolchain_hints};

const README: &str = "\
Brazier support bundle
======================

This archive contains a redacted diagnostic snapshot of the Brazier daemon,
hardware, configured runtimes, and host toolchain.

It does not contain conversations, prompts, model responses, attachments,
credentials, API keys, model files, or daemon logs. User-home and Brazier data
directory prefixes are replaced with placeholders. Please still review
diagnostics.json before sharing it, because model names, remote hostnames, and
other configuration choices may be visible.
";

/// Build a ZIP support bundle entirely in memory.
pub async fn create_bundle(state: &AppState) -> anyhow::Result<Vec<u8>> {
    let active = state.runtime.active_runtimes().await;
    let data_dir = state.data_dir.clone();
    let inventory_data_dir = data_dir.clone();
    let inventory_active = active.clone();
    let inventory = tokio::task::spawn_blocking(move || {
        runtimes::list(&inventory_data_dir, &inventory_active, None, false)
    })
    .await
    .context("collect runtime inventory")?;

    let remote_connections = remote::load(&data_dir)
        .iter()
        .map(remote::PublicConnection::from)
        .collect::<Vec<_>>();
    let generated_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut report = json!({
        "format_version": 1,
        "generated_at_unix": generated_at,
        "application": {
            "name": "Brazier",
            "version": env!("CARGO_PKG_VERSION"),
        },
        "hardware": hardware::detect(),
        "engine": state
            .runtime
            .engine_status(crate::engine::EngineStatusOptions { probe: false })
            .await,
        "runtimes": inventory,
        "toolchain": toolchain_hints::toolchain_status(),
        "remote_connections": remote_connections,
        "privacy": {
            "conversations_included": false,
            "attachments_included": false,
            "logs_included": false,
            "credentials_included": false,
        }
    });
    redact_report(&mut report, &data_dir, home_dir().as_deref());
    archive(&serde_json::to_vec_pretty(&report)?)
}

fn home_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("USERPROFILE").map(PathBuf::from)
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

fn archive(report: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    zip.start_file("README.txt", options)?;
    zip.write_all(README.as_bytes())?;
    zip.start_file("diagnostics.json", options)?;
    zip.write_all(report)?;
    Ok(zip.finish()?.into_inner())
}

fn redact_report(value: &mut Value, data_dir: &Path, home: Option<&Path>) {
    redact_value(value, None, data_dir, home);
}

fn redact_value(value: &mut Value, key: Option<&str>, data_dir: &Path, home: Option<&Path>) {
    match value {
        Value::Object(object) => redact_object(object, data_dir, home),
        Value::Array(values) => {
            for value in values {
                redact_value(value, key, data_dir, home);
            }
        }
        Value::String(text) => {
            if key.is_some_and(is_secret_key) || key == Some("default_voice_persona") {
                *text = "<redacted>".to_owned();
            } else {
                *text = redact_string(text, data_dir, home);
            }
        }
        _ => {}
    }
}

fn redact_object(object: &mut Map<String, Value>, data_dir: &Path, home: Option<&Path>) {
    for (key, value) in object {
        if is_secret_key(key) {
            *value = Value::String("<redacted>".to_owned());
        } else {
            redact_value(value, Some(key), data_dir, home);
        }
    }
}

fn is_secret_key(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase();
    if normalized.starts_with("has_") {
        return false;
    }
    normalized == "authorization"
        || normalized == "cookie"
        || normalized == "set_cookie"
        || normalized == "password"
        || normalized == "api_key"
        || normalized.ends_with("_api_key")
        || normalized == "token"
        || normalized.ends_with("_token")
        || normalized == "secret"
        || normalized.ends_with("_secret")
}

fn redact_string(value: &str, data_dir: &Path, home: Option<&Path>) -> String {
    let redacted_path = redact_path(value, data_dir, home);
    redact_url_credentials_and_query(&redacted_path)
}

fn redact_path(value: &str, data_dir: &Path, home: Option<&Path>) -> String {
    let path = Path::new(value);
    if let Ok(relative) = path.strip_prefix(data_dir) {
        return placeholder_path("<data-dir>", relative);
    }
    if let Some(home) = home
        && let Ok(relative) = path.strip_prefix(home)
    {
        return placeholder_path("~", relative);
    }
    value.to_owned()
}

fn placeholder_path(prefix: &str, relative: &Path) -> String {
    if relative.as_os_str().is_empty() {
        prefix.to_owned()
    } else {
        format!("{prefix}/{}", relative.display())
    }
}

/// Query strings and fragments commonly carry temporary credentials. They are
/// never useful in a support snapshot, even for otherwise public URLs.
fn redact_url_credentials_and_query(value: &str) -> String {
    let scheme_end = if value.starts_with("http://") {
        "http://".len()
    } else if value.starts_with("https://") {
        "https://".len()
    } else {
        return value.to_owned();
    };
    let end = value.find(['?', '#']).unwrap_or(value.len());
    let without_query = &value[..end];
    let authority_end = without_query[scheme_end..]
        .find('/')
        .map(|offset| scheme_end + offset)
        .unwrap_or(without_query.len());
    let authority = &without_query[scheme_end..authority_end];
    if let Some(userinfo_end) = authority.rfind('@') {
        format!(
            "{}{}",
            &without_query[..scheme_end],
            &without_query[scheme_end + userinfo_end + 1..]
        )
    } else {
        without_query.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use serde_json::json;
    use zip::ZipArchive;

    use super::*;

    #[test]
    fn redacts_secrets_paths_personal_text_and_url_queries() {
        let mut report = json!({
            "api_key": "sk-live",
            "nested": {
                "access_token": "token-value",
                "has_api_key": true,
                "default_voice_persona": "Call me Dylan",
                "binary": "/Users/person/Library/Application Support/Brazier/engines/llama",
                "model_path": "/Users/person/models/private/model.gguf",
                "endpoint": "https://person:password@example.test/v1?key=signed#fragment",
                "ordinary": "keep me",
            }
        });
        redact_report(
            &mut report,
            Path::new("/Users/person/Library/Application Support/Brazier"),
            Some(Path::new("/Users/person")),
        );

        assert_eq!(report["api_key"], "<redacted>");
        assert_eq!(report["nested"]["access_token"], "<redacted>");
        assert_eq!(report["nested"]["has_api_key"], true);
        assert_eq!(report["nested"]["default_voice_persona"], "<redacted>");
        assert_eq!(report["nested"]["binary"], "<data-dir>/engines/llama");
        assert_eq!(
            report["nested"]["model_path"],
            "~/models/private/model.gguf"
        );
        assert_eq!(report["nested"]["endpoint"], "https://example.test/v1");
        assert_eq!(report["nested"]["ordinary"], "keep me");
        let serialized = serde_json::to_string(&report).unwrap();
        assert!(!serialized.contains("sk-live"));
        assert!(!serialized.contains("/Users/person"));
        assert!(!serialized.contains("password"));
        assert!(!serialized.contains("signed"));
    }

    #[test]
    fn archive_is_reviewable_and_describes_its_privacy_boundary() {
        let bytes = archive(br#"{"status":"ok"}"#).unwrap();
        let mut archive = ZipArchive::new(Cursor::new(bytes)).unwrap();
        assert_eq!(archive.len(), 2);

        let mut diagnostics = String::new();
        archive
            .by_name("diagnostics.json")
            .unwrap()
            .read_to_string(&mut diagnostics)
            .unwrap();
        assert_eq!(diagnostics, r#"{"status":"ok"}"#);

        let mut readme = String::new();
        archive
            .by_name("README.txt")
            .unwrap()
            .read_to_string(&mut readme)
            .unwrap();
        assert!(readme.contains("does not contain conversations"));
        assert!(readme.contains("review"));
    }
}

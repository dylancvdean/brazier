//! Remote OpenAI-compatible connections.
//!
//! Brazier runs models locally, and that is the point of it. But an
//! OpenAI-compatible server someone already has — vLLM on a workstation down the
//! hall, llama-server on another machine, an Ollama box — is the same wire
//! protocol the local engines speak, and refusing to talk to it means keeping a
//! second application open for the same conversation.
//!
//! Connections are explicit and named: a base URL, an optional key, and a label.
//! Nothing is discovered, nothing is contacted until it is configured, and a
//! remote model's id says where it came from, so a conversation's run history
//! records that the answer was produced somewhere else.
//!
//! Keys are stored beside the connection list, with the file mode restricted on
//! Unix, and are never included in what the API returns — the UI is told whether
//! a key is set, never what it is.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::types::{ModelCapabilities, ModelDescriptor};

pub const ENGINE: &str = "remote";

/// `remote:{connection}/{model}` — the model id a conversation records.
const PREFIX: &str = "remote:";

/// A configured server, as stored. The key never leaves the daemon.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredConnection {
    pub id: String,
    pub label: String,
    pub base_url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

/// A connection as reported over the API: same fields, minus the secret.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PublicConnection {
    pub id: String,
    pub label: String,
    pub base_url: String,
    pub enabled: bool,
    /// Whether a key is stored. Never the key itself.
    pub has_api_key: bool,
}

impl From<&StoredConnection> for PublicConnection {
    fn from(connection: &StoredConnection) -> Self {
        Self {
            id: connection.id.clone(),
            label: connection.label.clone(),
            base_url: connection.base_url.clone(),
            enabled: connection.enabled,
            has_api_key: connection.api_key.is_some(),
        }
    }
}

pub fn connections_file(data_dir: &Path) -> PathBuf {
    data_dir.join("remote").join("connections.json")
}

pub fn load(data_dir: &Path) -> Vec<StoredConnection> {
    let path = connections_file(data_dir);
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

pub async fn save(data_dir: &Path, connections: &[StoredConnection]) -> anyhow::Result<()> {
    let path = connections_file(data_dir);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("create remote connection directory")?;
    }
    let text = serde_json::to_string_pretty(connections).context("encode remote connections")?;
    tokio::fs::write(&path, text)
        .await
        .context("write remote connections")?;
    // The file holds API keys; on Unix it is nobody else's business.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(&path)?.permissions();
        permissions.set_mode(0o600);
        std::fs::set_permissions(&path, permissions)?;
    }
    Ok(())
}

/// Add or replace one connection, keeping a stored key when none is supplied.
///
/// Editing a label or a URL should not require retyping a secret, and a UI that
/// cannot read the key back cannot resend it.
pub async fn upsert(data_dir: &Path, connection: StoredConnection) -> anyhow::Result<()> {
    let mut connection = connection;
    connection.id = normalize_id(&connection.id)?;
    connection.base_url = normalize_base_url(&connection.base_url)?;
    if connection.label.trim().is_empty() {
        connection.label = connection.id.clone();
    }
    let mut connections = load(data_dir);
    if let Some(existing) = connections
        .iter()
        .find(|entry| entry.id == connection.id)
        .cloned()
        && connection.api_key.is_none()
    {
        connection.api_key = existing.api_key;
    }
    connection.api_key = connection
        .api_key
        .map(|key| key.trim().to_owned())
        .filter(|key| !key.is_empty());
    connections.retain(|entry| entry.id != connection.id);
    connections.push(connection);
    connections.sort_by(|a, b| a.id.cmp(&b.id));
    save(data_dir, &connections).await
}

pub async fn remove(data_dir: &Path, id: &str) -> anyhow::Result<()> {
    let mut connections = load(data_dir);
    let before = connections.len();
    connections.retain(|entry| entry.id != id);
    anyhow::ensure!(connections.len() < before, "no remote connection `{id}`");
    save(data_dir, &connections).await
}

pub fn find(data_dir: &Path, id: &str) -> Option<StoredConnection> {
    load(data_dir).into_iter().find(|entry| entry.id == id)
}

/// Ids appear in model ids and in URLs, so they are kept to a boring alphabet.
pub fn normalize_id(id: &str) -> anyhow::Result<String> {
    let id = id.trim().to_ascii_lowercase();
    anyhow::ensure!(!id.is_empty(), "a connection id is required");
    anyhow::ensure!(
        id.len() <= 64
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
        "a connection id may only contain letters, digits, `-`, and `_`"
    );
    Ok(id)
}

/// Accept what someone would paste, and refuse what cannot be a server.
///
/// A trailing `/v1` is removed because every call site appends its own path;
/// leaving it produced `/v1/v1/chat/completions`, which fails with a 404 that
/// says nothing about the cause.
pub fn normalize_base_url(url: &str) -> anyhow::Result<String> {
    let url = url.trim().trim_end_matches('/');
    anyhow::ensure!(!url.is_empty(), "a base URL is required");
    anyhow::ensure!(
        url.starts_with("http://") || url.starts_with("https://"),
        "a base URL must start with http:// or https://"
    );
    let parsed = reqwest::Url::parse(url).context("that base URL could not be parsed")?;
    anyhow::ensure!(parsed.host_str().is_some(), "that base URL has no host");
    Ok(url.trim_end_matches("/v1").trim_end_matches('/').to_owned())
}

/// The model id a conversation stores for a model served remotely.
pub fn model_id(connection_id: &str, model: &str) -> String {
    format!("{PREFIX}{connection_id}/{model}")
}

/// Split a remote model id back into its connection and the server's own name.
pub fn parse_model_id(model_id: &str) -> Option<(String, String)> {
    let rest = model_id.strip_prefix(PREFIX)?;
    let (connection, model) = rest.split_once('/')?;
    if connection.is_empty() || model.is_empty() {
        return None;
    }
    Some((connection.to_owned(), model.to_owned()))
}

pub fn is_remote_model(model_id: &str) -> bool {
    parse_model_id(model_id).is_some()
}

/// Capabilities claimed for a remote model.
///
/// Deliberately plain: the protocol says nothing about what a server supports,
/// so this advertises what an OpenAI-compatible endpoint is expected to do and
/// nothing more. Claiming vision here would send image parts to a server that
/// may reject them, and the honest failure is better than a silent one.
pub fn capabilities() -> ModelCapabilities {
    ModelCapabilities {
        input_modalities: vec!["text".into()],
        output_modalities: vec!["text".into()],
        streaming: true,
        tools: true,
        reasoning: false,
        max_context_length: None,
        reasoning_modes: Vec::new(),
        harmony: false,
        audio_input: None,
    }
}

/// Model names a server reports, from `GET {base}/v1/models`.
pub async fn fetch_model_names(
    http: &reqwest::Client,
    connection: &StoredConnection,
) -> anyhow::Result<Vec<String>> {
    let mut request = http
        .get(format!("{}/v1/models", connection.base_url))
        .timeout(std::time::Duration::from_secs(15));
    if let Some(key) = connection.api_key.as_deref() {
        request = request.bearer_auth(key);
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("could not reach {}", connection.base_url))?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    anyhow::ensure!(
        status.is_success(),
        "{} returned {status}: {}",
        connection.base_url,
        body.chars().take(200).collect::<String>()
    );
    let payload: serde_json::Value =
        serde_json::from_str(&body).context("that server's model list was not JSON")?;
    let names = payload
        .get("data")
        .and_then(|data| data.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry.get("id").and_then(|id| id.as_str()))
                .map(|id| id.to_owned())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok(names)
}

/// Every enabled connection's models, as descriptors.
///
/// One unreachable server must not empty the model list: a machine that is
/// asleep is a normal state for a remote, and the local models are still there.
pub async fn list_models(http: &reqwest::Client, data_dir: &Path) -> Vec<ModelDescriptor> {
    let mut models = Vec::new();
    for connection in load(data_dir).into_iter().filter(|entry| entry.enabled) {
        match fetch_model_names(http, &connection).await {
            Ok(names) => {
                for name in names {
                    models.push(ModelDescriptor {
                        id: model_id(&connection.id, &name),
                        name: format!("{} · {name}", connection.label),
                        engine: ENGINE.to_owned(),
                        capabilities: capabilities(),
                        size_bytes: None,
                        read_only: true,
                        library_label: Some(connection.label.clone()),
                    });
                }
            }
            Err(error) => {
                tracing::debug!(connection = %connection.id, %error, "remote model list failed");
            }
        }
    }
    models.sort_by(|a, b| a.id.cmp(&b.id));
    models
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_ids_round_trip() {
        let id = model_id("workstation", "Qwen/Qwen3-8B");
        assert_eq!(id, "remote:workstation/Qwen/Qwen3-8B");
        let (connection, model) = parse_model_id(&id).unwrap();
        assert_eq!(connection, "workstation");
        // Only the first slash separates them: model names contain slashes.
        assert_eq!(model, "Qwen/Qwen3-8B");
        assert!(parse_model_id("gguf:acme/demo.gguf").is_none());
        assert!(parse_model_id("remote:workstation").is_none());
    }

    #[test]
    fn base_urls_are_normalized_not_guessed() {
        assert_eq!(
            normalize_base_url("http://10.0.0.4:8000/v1/").unwrap(),
            "http://10.0.0.4:8000"
        );
        assert_eq!(
            normalize_base_url(" https://api.example.com ").unwrap(),
            "https://api.example.com"
        );
        for bad in ["", "example.com:8000", "ftp://example.com", "http://"] {
            assert!(normalize_base_url(bad).is_err(), "{bad} must be refused");
        }
    }

    #[test]
    fn ids_stay_boring() {
        assert_eq!(normalize_id(" Workstation ").unwrap(), "workstation");
        for bad in ["", "has space", "has/slash", "has:colon"] {
            assert!(normalize_id(bad).is_err(), "{bad} must be refused");
        }
    }

    #[tokio::test]
    async fn editing_a_connection_keeps_the_key_it_was_not_told() {
        let dir = tempfile::tempdir().unwrap();
        upsert(
            dir.path(),
            StoredConnection {
                id: "work".into(),
                label: "Workstation".into(),
                base_url: "http://10.0.0.4:8000".into(),
                api_key: Some("secret".into()),
                enabled: true,
            },
        )
        .await
        .unwrap();
        upsert(
            dir.path(),
            StoredConnection {
                id: "work".into(),
                label: "Renamed".into(),
                base_url: "http://10.0.0.5:8000/v1".into(),
                api_key: None,
                enabled: false,
            },
        )
        .await
        .unwrap();

        let stored = find(dir.path(), "work").unwrap();
        assert_eq!(stored.label, "Renamed");
        assert_eq!(stored.base_url, "http://10.0.0.5:8000");
        assert!(!stored.enabled);
        assert_eq!(stored.api_key.as_deref(), Some("secret"));
        // What the API would report carries no secret.
        let public = PublicConnection::from(&stored);
        assert!(public.has_api_key);
        let json = serde_json::to_string(&public).unwrap();
        assert!(!json.contains("secret"), "{json}");
    }

    #[tokio::test]
    async fn clearing_a_key_is_possible() {
        let dir = tempfile::tempdir().unwrap();
        let connection = StoredConnection {
            id: "work".into(),
            label: "Workstation".into(),
            base_url: "http://10.0.0.4:8000".into(),
            api_key: Some("secret".into()),
            enabled: true,
        };
        upsert(dir.path(), connection.clone()).await.unwrap();
        upsert(
            dir.path(),
            StoredConnection {
                api_key: Some("   ".into()),
                ..connection
            },
        )
        .await
        .unwrap();
        assert_eq!(find(dir.path(), "work").unwrap().api_key, None);
    }

    #[tokio::test]
    async fn removing_something_that_is_not_there_says_so() {
        let dir = tempfile::tempdir().unwrap();
        assert!(remove(dir.path(), "ghost").await.is_err());
    }

    /// A stub OpenAI-compatible server that records the Authorization header.
    async fn stub_server(
        body: &'static str,
        status: axum::http::StatusCode,
    ) -> (String, std::sync::Arc<std::sync::Mutex<Option<String>>>) {
        use axum::{extract::State, routing::get};
        type Seen = std::sync::Arc<std::sync::Mutex<Option<String>>>;
        let seen: Seen = std::sync::Arc::new(std::sync::Mutex::new(None));
        let router = axum::Router::new()
            .route(
                "/v1/models",
                get(
                    move |State(seen): State<Seen>, headers: axum::http::HeaderMap| async move {
                        *seen.lock().unwrap() = headers
                            .get("authorization")
                            .and_then(|value| value.to_str().ok())
                            .map(|value| value.to_owned());
                        (status, body)
                    },
                ),
            )
            .with_state(seen.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        (format!("http://127.0.0.1:{port}"), seen)
    }

    #[tokio::test]
    async fn lists_what_a_server_says_it_serves_and_sends_the_key() {
        let (base_url, seen) = stub_server(
            r#"{"object":"list","data":[{"id":"qwen3-8b"},{"id":"llama-3.1-8b"}]}"#,
            axum::http::StatusCode::OK,
        )
        .await;
        let dir = tempfile::tempdir().unwrap();
        upsert(
            dir.path(),
            StoredConnection {
                id: "work".into(),
                label: "Workstation".into(),
                base_url,
                api_key: Some("sk-test".into()),
                enabled: true,
            },
        )
        .await
        .unwrap();

        let models = list_models(&reqwest::Client::new(), dir.path()).await;
        assert_eq!(
            models.iter().map(|model| &model.id).collect::<Vec<_>>(),
            vec!["remote:work/llama-3.1-8b", "remote:work/qwen3-8b"]
        );
        assert!(models.iter().all(|model| model.engine == ENGINE));
        assert_eq!(seen.lock().unwrap().as_deref(), Some("Bearer sk-test"));
    }

    /// A machine that is asleep is a normal state for a remote. It must cost the
    /// model list nothing but its own entries.
    #[tokio::test]
    async fn a_server_that_refuses_does_not_empty_the_list() {
        let (base_url, _) =
            stub_server(r#"{"error":"nope"}"#, axum::http::StatusCode::FORBIDDEN).await;
        let dir = tempfile::tempdir().unwrap();
        upsert(
            dir.path(),
            StoredConnection {
                id: "work".into(),
                label: "Workstation".into(),
                base_url: base_url.clone(),
                api_key: None,
                enabled: true,
            },
        )
        .await
        .unwrap();
        assert!(
            list_models(&reqwest::Client::new(), dir.path())
                .await
                .is_empty()
        );

        let connection = find(dir.path(), "work").unwrap();
        let error = fetch_model_names(&reqwest::Client::new(), &connection)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("403"), "{error}");
    }

    #[tokio::test]
    async fn a_connection_switched_off_is_not_contacted() {
        let (base_url, seen) =
            stub_server(r#"{"data":[{"id":"a"}]}"#, axum::http::StatusCode::OK).await;
        let dir = tempfile::tempdir().unwrap();
        upsert(
            dir.path(),
            StoredConnection {
                id: "work".into(),
                label: "Workstation".into(),
                base_url,
                api_key: None,
                enabled: false,
            },
        )
        .await
        .unwrap();
        assert!(
            list_models(&reqwest::Client::new(), dir.path())
                .await
                .is_empty()
        );
        assert!(seen.lock().unwrap().is_none(), "it must not be contacted");
    }
}

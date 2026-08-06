use std::{path::Path, time::Duration};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{
    FromRow, SqlitePool,
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous},
};
use std::collections::HashMap;
use uuid::Uuid;

/// Fields needed to create a durable queued-download record.
pub struct QueuedDownloadJobInput<'a> {
    pub repo_id: &'a str,
    pub filename: &'a str,
    pub revision: &'a str,
    pub kind: &'a str,
    pub payload: Option<&'a str>,
    pub label: Option<&'a str>,
    pub status: &'a str,
}

use crate::blob_store;
use brazier_protocol::types::{
    Conversation, CreateMessage, Message, Role, UpdateConversation, UpdateMessage,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunSnapshot {
    pub id: String,
    pub conversation_id: String,
    pub parent_message_id: Option<String>,
    pub assistant_message_id: Option<String>,
    pub model: String,
    pub settings: serde_json::Value,
    pub tool_calls: Option<serde_json::Value>,
    pub response_text: Option<String>,
    pub error: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CreateRunSnapshot {
    pub parent_message_id: Option<String>,
    pub assistant_message_id: Option<String>,
    pub model: String,
    pub settings: serde_json::Value,
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
    pub response_text: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportBlob {
    pub sha256: String,
    pub mime_type: String,
    pub data_base64: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationExport {
    pub schema_version: u32,
    pub exported_at: String,
    pub conversation: Conversation,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blobs: Vec<ExportBlob>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub run_snapshots: Vec<RunSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadJob {
    pub id: String,
    pub repo_id: String,
    pub filename: String,
    pub revision: String,
    /// `pending`, `downloading`, `paused`, `completed`, `failed`, `cancelled`.
    pub status: String,
    /// Which downloader handles this job (`gguf`, `mlx`, `sdcpp-bundle`, …).
    pub kind: String,
    /// Serialized request, so a paused job can be resumed after a restart.
    #[serde(skip_serializing)]
    pub payload: Option<String>,
    /// Human name for the queue UI.
    pub label: Option<String>,
    pub bytes_downloaded: Option<i64>,
    pub total_bytes: Option<i64>,
    pub sha256: Option<String>,
    pub error: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(FromRow)]
struct DownloadJobRow {
    id: String,
    repo_id: String,
    filename: String,
    revision: String,
    status: String,
    kind: String,
    payload: Option<String>,
    label: Option<String>,
    bytes_downloaded: Option<i64>,
    total_bytes: Option<i64>,
    sha256: Option<String>,
    error: Option<String>,
    created_at: String,
    updated_at: String,
}

impl From<DownloadJobRow> for DownloadJob {
    fn from(row: DownloadJobRow) -> Self {
        Self {
            id: row.id,
            repo_id: row.repo_id,
            filename: row.filename,
            revision: row.revision,
            status: row.status,
            kind: row.kind,
            payload: row.payload,
            label: row.label,
            bytes_downloaded: row.bytes_downloaded,
            total_bytes: row.total_bytes,
            sha256: row.sha256,
            error: row.error,
            created_at: row.created_at,
            updated_at: row.updated_at,
        }
    }
}

#[derive(Clone)]
pub struct Database {
    /// Visible to sibling modules that own their own tables (see
    /// `agent_store`), so those queries stay out of this file.
    pub(crate) pool: SqlitePool,
}

#[derive(FromRow)]
struct MessageRow {
    id: String,
    conversation_id: String,
    parent_id: Option<String>,
    role: String,
    content_json: String,
    model: Option<String>,
    tool_calls_json: Option<String>,
    tool_call_id: Option<String>,
    source: Option<String>,
    correlation_id: Option<String>,
    status: Option<String>,
    metadata_json: Option<String>,
    created_at: String,
}

/// Columns every message read selects, in `MessageRow` order.
const MESSAGE_COLUMNS: &str = "id, conversation_id, parent_id, role, content_json, model,
     tool_calls_json, tool_call_id, source, correlation_id, status, metadata_json, created_at";

/// Columns every conversation read selects, in `Conversation` field order.
const CONVERSATION_COLUMNS: &str =
    "id, title, created_at, updated_at, agent_session_id, summary, summary_updated_at, incognito";

/// [`CONVERSATION_COLUMNS`] qualified for the search join, where `messages` is
/// also in scope.
const CONVERSATION_COLUMNS_QUALIFIED: &str = "c.id, c.title, c.created_at, c.updated_at,
     c.agent_session_id, c.summary, c.summary_updated_at, c.incognito";

#[derive(FromRow)]
struct RunSnapshotRow {
    id: String,
    conversation_id: String,
    parent_message_id: Option<String>,
    assistant_message_id: Option<String>,
    model: String,
    settings_json: String,
    tool_calls_json: Option<String>,
    response_text: Option<String>,
    error: Option<String>,
    created_at: String,
}

impl TryFrom<RunSnapshotRow> for RunSnapshot {
    type Error = anyhow::Error;

    fn try_from(row: RunSnapshotRow) -> Result<Self, Self::Error> {
        Ok(Self {
            id: row.id,
            conversation_id: row.conversation_id,
            parent_message_id: row.parent_message_id,
            assistant_message_id: row.assistant_message_id,
            model: row.model,
            settings: serde_json::from_str(&row.settings_json)?,
            tool_calls: row
                .tool_calls_json
                .as_deref()
                .map(serde_json::from_str)
                .transpose()?,
            response_text: row.response_text,
            error: row.error,
            created_at: row.created_at,
        })
    }
}

impl TryFrom<MessageRow> for Message {
    type Error = anyhow::Error;

    fn try_from(row: MessageRow) -> Result<Self, Self::Error> {
        let role = match row.role.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            value => anyhow::bail!("unknown message role {value}"),
        };
        Ok(Self {
            id: row.id,
            conversation_id: row.conversation_id,
            parent_id: row.parent_id,
            role,
            content: serde_json::from_str(&row.content_json)?,
            model: row.model,
            tool_calls: row
                .tool_calls_json
                .as_deref()
                .map(serde_json::from_str)
                .transpose()?,
            tool_call_id: row.tool_call_id,
            source: row.source,
            correlation_id: row.correlation_id,
            status: row.status,
            metadata: row
                .metadata_json
                .as_deref()
                .map(serde_json::from_str)
                .transpose()?,
            created_at: row.created_at,
        })
    }
}

impl Database {
    pub async fn open(path: &Path) -> anyhow::Result<Self> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .context("create data directory")?;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
            if !path.exists() {
                std::fs::OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .mode(0o600)
                    .open(path)
                    .context("create sqlite database file")?;
            }
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .foreign_keys(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .busy_timeout(Duration::from_secs(5));
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
            .context("open sqlite database")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
            for suffix in ["-wal", "-shm"] {
                let sibling: std::path::PathBuf =
                    format!("{}{suffix}", path.display()).into();
                if sibling.exists() {
                    let _ =
                        std::fs::set_permissions(&sibling, std::fs::Permissions::from_mode(0o600));
                }
            }
        }
        let db = Self { pool };
        db.migrate().await?;
        Ok(db)
    }

    async fn migrate(&self) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        let mut version: i64 =
            sqlx::query_scalar("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
                .fetch_one(&self.pool)
                .await?;
        let existing_installation = version > 0;

        if version < 1 {
            let mut tx = self.pool.begin().await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    parent_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
                    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
                    content_json TEXT NOT NULL,
                    model TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "CREATE INDEX IF NOT EXISTS messages_conversation_created ON messages(conversation_id, created_at)",
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query("INSERT OR IGNORE INTO schema_migrations(version) VALUES (1)")
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
            version = 1;
        }

        if version < 2 {
            let mut tx = self.pool.begin().await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS attachments (
                    sha256 TEXT PRIMARY KEY,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    original_name TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS message_attachments (
                    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    sha256 TEXT NOT NULL REFERENCES attachments(sha256),
                    PRIMARY KEY (message_id, sha256)
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query("INSERT OR IGNORE INTO schema_migrations(version) VALUES (2)")
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
            version = 2;
        }

        if version < 3 {
            let mut tx = self.pool.begin().await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS run_snapshots (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    parent_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
                    assistant_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
                    model TEXT NOT NULL,
                    settings_json TEXT NOT NULL,
                    tool_calls_json TEXT,
                    response_text TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "CREATE INDEX IF NOT EXISTS run_snapshots_conversation_created ON run_snapshots(conversation_id, created_at DESC)",
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query("INSERT OR IGNORE INTO schema_migrations(version) VALUES (3)")
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
            version = 3;
        }

        if version < 4 {
            let mut tx = self.pool.begin().await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS download_jobs (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed')),
                    bytes_downloaded INTEGER,
                    total_bytes INTEGER,
                    sha256 TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "CREATE INDEX IF NOT EXISTS download_jobs_updated ON download_jobs(updated_at DESC)",
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query("INSERT OR IGNORE INTO schema_migrations(version) VALUES (4)")
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
            version = 4;
        }

        if version < 5 {
            let mut tx = self.pool.begin().await?;
            sqlx::query("ALTER TABLE messages ADD COLUMN tool_calls_json TEXT")
                .execute(&mut *tx)
                .await?;
            sqlx::query("ALTER TABLE messages ADD COLUMN tool_call_id TEXT")
                .execute(&mut *tx)
                .await?;
            sqlx::query("INSERT OR IGNORE INTO schema_migrations(version) VALUES (5)")
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
        }

        if version < 6 {
            // Queued work needs to survive a restart, and jobs can now be
            // paused, so the status CHECK from migration 4 is replaced. The
            // rebuild runs in one transaction on one connection: the pool
            // hands out several, and a DROP on one is not visible to another
            // mid-migration, which made the rename fail.
            let mut tx = self.pool.begin().await?;
            sqlx::query("ALTER TABLE download_jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'gguf'")
                .execute(&mut *tx)
                .await?;
            sqlx::query("ALTER TABLE download_jobs ADD COLUMN payload TEXT")
                .execute(&mut *tx)
                .await?;
            sqlx::query("ALTER TABLE download_jobs ADD COLUMN label TEXT")
                .execute(&mut *tx)
                .await?;
            sqlx::query(
                r#"
                CREATE TABLE download_jobs_v6 (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN (
                        'pending', 'downloading', 'paused', 'completed', 'failed', 'cancelled'
                    )),
                    kind TEXT NOT NULL DEFAULT 'gguf',
                    payload TEXT,
                    label TEXT,
                    bytes_downloaded INTEGER,
                    total_bytes INTEGER,
                    sha256 TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                r#"INSERT INTO download_jobs_v6(
                       id, repo_id, filename, revision, status, kind, payload, label,
                       bytes_downloaded, total_bytes, sha256, error, created_at, updated_at)
                   SELECT id, repo_id, filename, revision,
                          CASE status WHEN 'running' THEN 'downloading' ELSE status END,
                          kind, payload, label,
                          bytes_downloaded, total_bytes, sha256, error, created_at, updated_at
                   FROM download_jobs"#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query("DROP TABLE download_jobs")
                .execute(&mut *tx)
                .await?;
            sqlx::query("ALTER TABLE download_jobs_v6 RENAME TO download_jobs")
                .execute(&mut *tx)
                .await?;
            sqlx::query(
                "CREATE INDEX IF NOT EXISTS download_jobs_updated ON download_jobs(updated_at DESC)",
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query("INSERT OR IGNORE INTO schema_migrations(version) VALUES (6)")
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
            version = 6;
        }

        if version < 7 {
            // Agent mode state. Sessions are independent of chat conversations:
            // they carry a workspace, a permission mode, a tool-execution
            // ledger, and the approvals the user granted.
            let mut tx = self.pool.begin().await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    workspace_path TEXT,
                    model TEXT NOT NULL,
                    runtime_id TEXT NOT NULL,
                    permission_mode TEXT NOT NULL CHECK (
                        permission_mode IN ('ask', 'sandbox-only', 'skip-permissions')
                    ),
                    permission_settings_json TEXT NOT NULL,
                    enabled_tools_json TEXT,
                    last_run_status TEXT NOT NULL DEFAULT 'idle',
                    compaction_json TEXT,
                    runtime_metadata_json TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS agent_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE (session_id, seq)
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS agent_tool_executions (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
                    run_id TEXT,
                    tool_call_id TEXT,
                    tool TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    status TEXT NOT NULL,
                    exit_code INTEGER,
                    output_preview TEXT,
                    artifact_id TEXT,
                    truncated INTEGER NOT NULL DEFAULT 0,
                    changed_paths_json TEXT,
                    sandbox_json TEXT,
                    approval_id TEXT,
                    error TEXT,
                    duration_ms INTEGER,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "CREATE INDEX IF NOT EXISTS agent_tool_executions_session
                 ON agent_tool_executions(session_id, created_at)",
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS agent_approvals (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
                    tool TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    arguments_hash TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    allow_session_scope INTEGER NOT NULL DEFAULT 0,
                    elevation_json TEXT NOT NULL,
                    sandbox_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'approved', 'denied', 'expired', 'consumed')
                    ),
                    scope TEXT,
                    note TEXT,
                    decided_at TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "CREATE INDEX IF NOT EXISTS agent_approvals_session_status
                 ON agent_approvals(session_id, status)",
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS agent_grants (
                    session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
                    grant_key TEXT NOT NULL,
                    approval_id TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (session_id, grant_key)
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS agent_artifacts (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
                    tool_execution_id TEXT,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mime_type TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query("INSERT OR IGNORE INTO schema_migrations(version) VALUES (7)")
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
            version = 7;
        }

        if version < 8 {
            // Shared-conversation integration: voice, chat, and the agent write
            // into one message graph, so a message records which surface
            // produced it, which turn it belongs to, and whether it is still
            // the live answer. A conversation records the agent session it is
            // bound to plus the compact summary the voice session is seeded
            // with. Voice runtime state is deliberately not persisted.
            let mut tx = self.pool.begin().await?;
            for statement in [
                "ALTER TABLE messages ADD COLUMN source TEXT",
                "ALTER TABLE messages ADD COLUMN correlation_id TEXT",
                "ALTER TABLE messages ADD COLUMN status TEXT",
                "ALTER TABLE messages ADD COLUMN metadata_json TEXT",
                "ALTER TABLE conversations ADD COLUMN agent_session_id TEXT",
                "ALTER TABLE conversations ADD COLUMN summary TEXT",
                "ALTER TABLE conversations ADD COLUMN summary_updated_at TEXT",
            ] {
                sqlx::query(statement).execute(&mut *tx).await?;
            }
            sqlx::query(
                "CREATE INDEX IF NOT EXISTS messages_correlation
                 ON messages(conversation_id, correlation_id)",
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query("INSERT OR IGNORE INTO schema_migrations(version) VALUES (8)")
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
            version = 8;
        }

        if version < 9 {
            // Workspace-scoped Agent settings outlive individual tasks. Session
            // rows remain the task history; this table is the durable project
            // record they share.
            let mut tx = self.pool.begin().await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS agent_workspaces (
                    workspace_path TEXT PRIMARY KEY,
                    system_prompt TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                r#"INSERT OR IGNORE INTO agent_workspaces(workspace_path)
                   SELECT DISTINCT workspace_path
                   FROM agent_sessions
                   WHERE workspace_path IS NOT NULL"#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query("INSERT OR IGNORE INTO schema_migrations(version) VALUES (9)")
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
            version = 9;
        }

        if version < 10 {
            // Small application preferences that must survive renderer origin
            // changes (development HTTP versus packaged file URLs).
            let mut tx = self.pool.begin().await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS application_preferences (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            if existing_installation {
                // Earlier releases stored this in origin-scoped localStorage.
                // An existing database is sufficient evidence that this is an
                // upgrade, and avoids replaying onboarding when switching from
                // the packaged file:// renderer to the development HTTP one.
                sqlx::query(
                    r#"INSERT OR IGNORE INTO application_preferences(key, value_json)
                       VALUES ('welcome', '{"completed":true}')"#,
                )
                .execute(&mut *tx)
                .await?;
            }
            sqlx::query("INSERT OR IGNORE INTO schema_migrations(version) VALUES (10)")
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
        }

        if version < 11 {
            // Durable acceptances of model license agreements. Some curated
            // models (MiniMax-H3) are only licensed in part of the world
            // without a separate grant from the publisher, so whether the
            // person agreed — and to which version of the terms — must survive
            // restarts rather than living in renderer memory.
            let mut tx = self.pool.begin().await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS license_consents (
                    license_id TEXT PRIMARY KEY,
                    license_version TEXT NOT NULL,
                    accepted_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query("INSERT OR IGNORE INTO schema_migrations(version) VALUES (11)")
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
        }

        if version < 12 {
            // Chat memory: durable facts the model saves across conversations,
            // edited in Settings, and consolidated by dreaming. Incognito
            // conversations are ephemeral, so the flag is a daemon-side gate
            // that refuses message writes and memory sources for them.
            let mut tx = self.pool.begin().await?;
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'fact',
                    pinned INTEGER NOT NULL DEFAULT 0,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    source_conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
                    source_message_id TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                "#,
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query("CREATE INDEX IF NOT EXISTS memories_updated ON memories(updated_at)")
                .execute(&mut *tx)
                .await?;
            sqlx::query(
                "ALTER TABLE conversations ADD COLUMN incognito INTEGER NOT NULL DEFAULT 0",
            )
            .execute(&mut *tx)
            .await?;
            sqlx::query("INSERT OR IGNORE INTO schema_migrations(version) VALUES (12)")
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
        }

        Ok(())
    }

    pub(crate) fn escape_like(query: &str) -> String {
        query
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_")
    }

    pub async fn list_conversations(
        &self,
        query: Option<&str>,
    ) -> anyhow::Result<Vec<Conversation>> {
        let Some(raw) = query.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(sqlx::query_as::<_, Conversation>(&format!(
                "SELECT {CONVERSATION_COLUMNS} FROM conversations ORDER BY updated_at DESC"
            ))
            .fetch_all(&self.pool)
            .await?);
        };
        let pattern = format!("%{}%", Self::escape_like(raw));
        Ok(sqlx::query_as::<_, Conversation>(&format!(
            r#"SELECT DISTINCT {CONVERSATION_COLUMNS_QUALIFIED}
               FROM conversations c
               LEFT JOIN messages m ON m.conversation_id = c.id
               WHERE c.title LIKE ?1 ESCAPE '\'
                  OR m.content_json LIKE ?1 ESCAPE '\'
               ORDER BY c.updated_at DESC
               LIMIT 80"#
        ))
        .bind(&pattern)
        .fetch_all(&self.pool)
        .await?)
    }

    pub async fn create_conversation(&self, title: &str) -> anyhow::Result<Conversation> {
        let id = Uuid::new_v4().to_string();
        sqlx::query("INSERT INTO conversations(id, title) VALUES(?, ?)")
            .bind(&id)
            .bind(title)
            .execute(&self.pool)
            .await?;
        self.get_conversation(&id).await
    }

    pub async fn get_conversation(&self, id: &str) -> anyhow::Result<Conversation> {
        Ok(sqlx::query_as::<_, Conversation>(&format!(
            "SELECT {CONVERSATION_COLUMNS} FROM conversations WHERE id = ?"
        ))
        .bind(id)
        .fetch_one(&self.pool)
        .await?)
    }

    /// Remove a conversation and its message graph. Message rows cascade from
    /// the conversation foreign key; blobs remain available to any other turn
    /// that references them.
    pub async fn delete_conversation(&self, id: &str) -> anyhow::Result<()> {
        let result = sqlx::query("DELETE FROM conversations WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        if result.rows_affected() == 0 {
            anyhow::bail!("conversation not found")
        }
        Ok(())
    }

    /// Apply a partial conversation update. Absent fields are left alone; an
    /// explicit `agent_session_id: null` unbinds the agent session.
    pub async fn update_conversation(
        &self,
        id: &str,
        update: UpdateConversation,
    ) -> anyhow::Result<Conversation> {
        // Confirm the conversation exists so an unknown id is not a silent no-op.
        self.get_conversation(id).await?;
        let mut tx = self.pool.begin().await?;
        if let Some(title) = update
            .title
            .as_deref()
            .map(str::trim)
            .filter(|title| !title.is_empty())
        {
            sqlx::query("UPDATE conversations SET title = ? WHERE id = ?")
                .bind(title)
                .bind(id)
                .execute(&mut *tx)
                .await?;
        }
        if let Some(agent_session_id) = update.agent_session_id {
            sqlx::query("UPDATE conversations SET agent_session_id = ? WHERE id = ?")
                .bind(&agent_session_id)
                .bind(id)
                .execute(&mut *tx)
                .await?;
        }
        if let Some(summary) = update.summary {
            sqlx::query(
                "UPDATE conversations
                 SET summary = ?, summary_updated_at = datetime('now')
                 WHERE id = ?",
            )
            .bind(&summary)
            .bind(id)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        self.get_conversation(id).await
    }

    pub async fn list_messages(&self, conversation_id: &str) -> anyhow::Result<Vec<Message>> {
        let rows = sqlx::query_as::<_, MessageRow>(&format!(
            "SELECT {MESSAGE_COLUMNS}
             FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid"
        ))
        .bind(conversation_id)
        .fetch_all(&self.pool)
        .await?;
        let mut messages = Vec::with_capacity(rows.len());
        for row in rows {
            let message_id = row.id.clone();
            match Message::try_from(row) {
                Ok(message) => messages.push(message),
                Err(error) => tracing::warn!(
                    conversation_id = conversation_id,
                    message_id = %message_id,
                    error = %error,
                    "dropping malformed message row from conversation"
                ),
            }
        }
        Ok(messages)
    }

    pub async fn create_message(
        &self,
        conversation_id: &str,
        message: CreateMessage,
    ) -> anyhow::Result<Message> {
        if let Some(parent_id) = &message.parent_id {
            let parent_conversation: Option<String> =
                sqlx::query_scalar("SELECT conversation_id FROM messages WHERE id = ?")
                    .bind(parent_id)
                    .fetch_optional(&self.pool)
                    .await?;
            anyhow::ensure!(
                parent_conversation.as_deref() == Some(conversation_id),
                "parent message must belong to this conversation"
            );
        }

        let id = Uuid::new_v4().to_string();
        let content_json = serde_json::to_string(&message.content)?;
        let tool_calls_json = message
            .tool_calls
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let metadata_json = message
            .metadata
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let blob_refs = blob_store::blob_refs_in_content(&message.content);
        for sha256 in &blob_refs {
            blob_store::validate_sha256(sha256)?;
        }
        let mut tx = self.pool.begin().await?;
        sqlx::query(
            r#"INSERT INTO messages(id, conversation_id, parent_id, role, content_json, model,
                                   tool_calls_json, tool_call_id, source, correlation_id,
                                   status, metadata_json)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(&id)
        .bind(conversation_id)
        .bind(&message.parent_id)
        .bind(message.role.as_str())
        .bind(content_json)
        .bind(&message.model)
        .bind(tool_calls_json)
        .bind(&message.tool_call_id)
        .bind(&message.source)
        .bind(&message.correlation_id)
        .bind(&message.status)
        .bind(metadata_json)
        .execute(&mut *tx)
        .await?;
        sqlx::query("UPDATE conversations SET updated_at = datetime('now') WHERE id = ?")
            .bind(conversation_id)
            .execute(&mut *tx)
            .await?;
        for sha256 in blob_refs {
            sqlx::query("INSERT INTO message_attachments(message_id, sha256) VALUES(?, ?)")
                .bind(&id)
                .bind(sha256)
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        self.get_message(&id).await
    }

    /// Edit a stored message in place: finalize streamed content, or mark the
    /// turn cancelled, superseded, or failed. The message is never deleted, so
    /// cancelling spoken delivery cannot remove an answer from the chat.
    pub async fn update_message(
        &self,
        conversation_id: &str,
        message_id: &str,
        update: UpdateMessage,
    ) -> anyhow::Result<Message> {
        let blob_refs = update
            .content
            .as_ref()
            .map(blob_store::blob_refs_in_content)
            .unwrap_or_default();
        for sha256 in &blob_refs {
            blob_store::validate_sha256(sha256)?;
        }
        let mut tx = self.pool.begin().await?;
        // Take the write reservation before reading ownership so two edits do
        // not both enter a deferred read transaction and then fail to upgrade.
        let owner: Option<String> = sqlx::query_scalar(
            "UPDATE messages SET id = id WHERE id = ? RETURNING conversation_id",
        )
        .bind(message_id)
        .fetch_optional(&mut *tx)
        .await?;
        anyhow::ensure!(
            owner.as_deref() == Some(conversation_id),
            "message does not belong to this conversation"
        );
        if let Some(content) = update.content.as_ref() {
            sqlx::query("UPDATE messages SET content_json = ? WHERE id = ?")
                .bind(serde_json::to_string(content)?)
                .bind(message_id)
                .execute(&mut *tx)
                .await?;
            sqlx::query("DELETE FROM message_attachments WHERE message_id = ?")
                .bind(message_id)
                .execute(&mut *tx)
                .await?;
            for sha256 in blob_refs {
                sqlx::query("INSERT INTO message_attachments(message_id, sha256) VALUES(?, ?)")
                    .bind(message_id)
                    .bind(sha256)
                    .execute(&mut *tx)
                    .await?;
            }
        }
        if let Some(status) = update.status.as_deref() {
            sqlx::query("UPDATE messages SET status = ? WHERE id = ?")
                .bind(status)
                .bind(message_id)
                .execute(&mut *tx)
                .await?;
        }
        if let Some(metadata) = update.metadata.as_ref() {
            sqlx::query("UPDATE messages SET metadata_json = ? WHERE id = ?")
                .bind(serde_json::to_string(metadata)?)
                .bind(message_id)
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        self.get_message(message_id).await
    }

    pub async fn upsert_attachment(
        &self,
        sha256: &str,
        mime_type: &str,
        size_bytes: i64,
        original_name: Option<&str>,
    ) -> anyhow::Result<()> {
        blob_store::validate_sha256(sha256)?;
        sqlx::query(
            r#"INSERT INTO attachments(sha256, mime_type, size_bytes, original_name)
               VALUES(?, ?, ?, ?)
               ON CONFLICT(sha256) DO UPDATE SET
                 mime_type = excluded.mime_type,
                 size_bytes = excluded.size_bytes,
                 original_name = COALESCE(excluded.original_name, attachments.original_name)"#,
        )
        .bind(sha256)
        .bind(mime_type)
        .bind(size_bytes)
        .bind(original_name)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn create_run_snapshot(
        &self,
        conversation_id: &str,
        snapshot: CreateRunSnapshot,
    ) -> anyhow::Result<RunSnapshot> {
        let id = Uuid::new_v4().to_string();
        let settings_json = serde_json::to_string(&snapshot.settings)?;
        let tool_calls_json = snapshot
            .tool_calls
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        sqlx::query(
            r#"INSERT INTO run_snapshots(
                   id, conversation_id, parent_message_id, assistant_message_id,
                   model, settings_json, tool_calls_json, response_text, error
               ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(&id)
        .bind(conversation_id)
        .bind(&snapshot.parent_message_id)
        .bind(&snapshot.assistant_message_id)
        .bind(&snapshot.model)
        .bind(settings_json)
        .bind(tool_calls_json)
        .bind(&snapshot.response_text)
        .bind(&snapshot.error)
        .execute(&self.pool)
        .await?;
        self.get_run_snapshot(&id).await
    }

    pub async fn list_run_snapshots(
        &self,
        conversation_id: &str,
    ) -> anyhow::Result<Vec<RunSnapshot>> {
        let rows = sqlx::query_as::<_, RunSnapshotRow>(
            r#"SELECT id, conversation_id, parent_message_id, assistant_message_id,
                      model, settings_json, tool_calls_json, response_text, error, created_at
               FROM run_snapshots WHERE conversation_id = ? ORDER BY created_at DESC"#,
        )
        .bind(conversation_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(TryInto::try_into).collect()
    }

    async fn get_run_snapshot(&self, id: &str) -> anyhow::Result<RunSnapshot> {
        let row = sqlx::query_as::<_, RunSnapshotRow>(
            r#"SELECT id, conversation_id, parent_message_id, assistant_message_id,
                      model, settings_json, tool_calls_json, response_text, error, created_at
               FROM run_snapshots WHERE id = ?"#,
        )
        .bind(id)
        .fetch_one(&self.pool)
        .await?;
        row.try_into()
    }

    async fn get_message(&self, id: &str) -> anyhow::Result<Message> {
        let row = sqlx::query_as::<_, MessageRow>(&format!(
            "SELECT {MESSAGE_COLUMNS} FROM messages WHERE id = ?"
        ))
        .bind(id)
        .fetch_one(&self.pool)
        .await?;
        row.try_into()
    }

    pub async fn export_conversation(
        &self,
        data_dir: &Path,
        id: &str,
    ) -> anyhow::Result<ConversationExport> {
        let conversation = self.get_conversation(id).await?;
        let messages = self.list_messages(id).await?;
        let run_snapshots = self.list_run_snapshots(id).await?;
        let mut blob_ids = Vec::new();
        for message in &messages {
            blob_ids.extend(blob_store::blob_refs_in_content(&message.content));
        }
        blob_ids.sort();
        blob_ids.dedup();
        let mut blobs = Vec::new();
        for sha256 in blob_ids {
            let (bytes, mime_type) = blob_store::read_blob(data_dir, &sha256).await?;
            blobs.push(ExportBlob {
                sha256: sha256.clone(),
                mime_type,
                data_base64: base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    &bytes,
                ),
                original_name: None,
            });
        }
        Ok(ConversationExport {
            schema_version: 2,
            exported_at: format!(
                "{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ),
            conversation,
            messages,
            blobs,
            run_snapshots,
        })
    }

    pub async fn import_conversation(
        &self,
        data_dir: &Path,
        export: ConversationExport,
    ) -> anyhow::Result<Conversation> {
        anyhow::ensure!(
            export.schema_version == 1 || export.schema_version == 2,
            "unsupported export schema version {}",
            export.schema_version
        );
        for message in &export.messages {
            for sha256 in blob_store::blob_refs_in_content(&message.content) {
                blob_store::validate_sha256(&sha256)?;
            }
        }
        for blob in &export.blobs {
            blob_store::validate_sha256(&blob.sha256)?;
            let bytes = base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                blob.data_base64.trim(),
            )
            .context("decode export blob")?;
            anyhow::ensure!(
                blob_store::sha256_hex(&bytes) == blob.sha256,
                "export blob {} does not match its declared digest",
                blob.sha256
            );
            let stored = blob_store::store_bytes(
                data_dir,
                &bytes,
                &blob.mime_type,
                blob.original_name.as_deref(),
            )
            .await?;
            self.upsert_attachment(
                &stored.sha256,
                &stored.mime_type,
                stored.size_bytes as i64,
                stored.original_name.as_deref(),
            )
            .await?;
        }
        let title = export.conversation.title.trim();
        let title = if title.is_empty() {
            "Imported conversation"
        } else {
            title
        };
        let conversation_id = Uuid::new_v4().to_string();
        let mut id_map: HashMap<String, String> = HashMap::new();
        for message in &export.messages {
            anyhow::ensure!(
                id_map
                    .insert(message.id.clone(), Uuid::new_v4().to_string())
                    .is_none(),
                "duplicate message id in import"
            );
        }
        let mut tx = self.pool.begin().await?;
        sqlx::query("INSERT INTO conversations(id, title) VALUES(?, ?)")
            .bind(&conversation_id)
            .bind(title)
            .execute(&mut *tx)
            .await?;
        for message in export.messages {
            let new_id = id_map
                .get(&message.id)
                .context("missing remapped message id")?;
            let parent_id = match message.parent_id.as_ref() {
                Some(parent) => Some(
                    id_map
                        .get(parent)
                        .cloned()
                        .with_context(|| format!("missing parent message {parent}"))?,
                ),
                None => None,
            };
            let content_json = serde_json::to_string(&message.content)?;
            let tool_calls_json = message
                .tool_calls
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?;
            let metadata_json = message
                .metadata
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?;
            sqlx::query(
                r#"INSERT INTO messages(id, conversation_id, parent_id, role, content_json, model,
                                       tool_calls_json, tool_call_id, source, correlation_id,
                                       status, metadata_json, created_at)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
            )
            .bind(new_id)
            .bind(&conversation_id)
            .bind(parent_id)
            .bind(message.role.as_str())
            .bind(content_json)
            .bind(&message.model)
            .bind(tool_calls_json)
            .bind(&message.tool_call_id)
            .bind(&message.source)
            // Correlation ids scope live work, not history: importing keeps the
            // source and status labels but not another conversation's turn ids.
            .bind(Option::<String>::None)
            .bind(&message.status)
            .bind(metadata_json)
            .bind(&message.created_at)
            .execute(&mut *tx)
            .await?;
            for sha256 in blob_store::blob_refs_in_content(&message.content) {
                sqlx::query("INSERT INTO message_attachments(message_id, sha256) VALUES(?, ?)")
                    .bind(new_id)
                    .bind(sha256)
                    .execute(&mut *tx)
                    .await?;
            }
        }
        for snapshot in export.run_snapshots {
            let parent_message_id = match snapshot.parent_message_id.as_ref() {
                Some(id) => Some(
                    id_map
                        .get(id)
                        .cloned()
                        .with_context(|| format!("missing run parent message {id}"))?,
                ),
                None => None,
            };
            let assistant_message_id = match snapshot.assistant_message_id.as_ref() {
                Some(id) => Some(
                    id_map
                        .get(id)
                        .cloned()
                        .with_context(|| format!("missing run assistant message {id}"))?,
                ),
                None => None,
            };
            sqlx::query(
                r#"INSERT INTO run_snapshots(
                       id, conversation_id, parent_message_id, assistant_message_id, model,
                       settings_json, tool_calls_json, response_text, error, created_at)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
            )
            .bind(Uuid::new_v4().to_string())
            .bind(&conversation_id)
            .bind(parent_message_id)
            .bind(assistant_message_id)
            .bind(snapshot.model)
            .bind(serde_json::to_string(&snapshot.settings)?)
            .bind(
                snapshot
                    .tool_calls
                    .as_ref()
                    .map(serde_json::to_string)
                    .transpose()?,
            )
            .bind(snapshot.response_text)
            .bind(snapshot.error)
            .bind(snapshot.created_at)
            .execute(&mut *tx)
            .await?;
        }
        sqlx::query("UPDATE conversations SET updated_at = datetime('now') WHERE id = ?")
            .bind(&conversation_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        self.get_conversation(&conversation_id).await
    }

    pub async fn create_download_job(
        &self,
        repo_id: &str,
        filename: &str,
        revision: &str,
    ) -> anyhow::Result<DownloadJob> {
        self.create_queued_download_job(QueuedDownloadJobInput {
            repo_id,
            filename,
            revision,
            kind: "gguf",
            payload: None,
            label: None,
            status: "downloading",
        })
        .await
    }

    /// Record a job the queue will run later, keeping enough detail to
    /// reconstruct the work after a pause or a restart.
    pub async fn create_queued_download_job(
        &self,
        input: QueuedDownloadJobInput<'_>,
    ) -> anyhow::Result<DownloadJob> {
        let QueuedDownloadJobInput {
            repo_id,
            filename,
            revision,
            kind,
            payload,
            label,
            status,
        } = input;
        let id = Uuid::new_v4().to_string();
        sqlx::query(
            r#"INSERT INTO download_jobs(id, repo_id, filename, revision, status, kind, payload, label)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(&id)
        .bind(repo_id)
        .bind(filename)
        .bind(revision)
        .bind(status)
        .bind(kind)
        .bind(payload)
        .bind(label)
        .execute(&self.pool)
        .await?;
        self.get_download_job(&id).await
    }

    /// Mark a job paused, keeping its partial file for a later resume.
    pub async fn pause_download_job(&self, job_id: &str) -> anyhow::Result<()> {
        let result = sqlx::query(
            r#"UPDATE download_jobs
               SET status = 'paused', updated_at = datetime('now')
               WHERE id = ? AND status IN ('pending', 'downloading')"#,
        )
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        anyhow::ensure!(
            result.rows_affected() == 1,
            "that download is no longer queued or running"
        );
        Ok(())
    }

    /// Put a paused or failed job back in line.
    pub async fn requeue_download_job(&self, job_id: &str) -> anyhow::Result<()> {
        let result = sqlx::query(
            r#"UPDATE download_jobs
               SET status = 'pending', error = NULL, updated_at = datetime('now')
               WHERE id = ? AND status IN ('paused', 'failed', 'cancelled')"#,
        )
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        anyhow::ensure!(
            result.rows_affected() == 1,
            "that download is already queued or running"
        );
        Ok(())
    }

    pub async fn get_download_job_public(&self, id: &str) -> anyhow::Result<DownloadJob> {
        self.get_download_job(id).await
    }

    /// Jobs that were mid-flight when the daemon stopped. Transfers can be
    /// paused and resumed from their partial files; source builds cannot, so
    /// their tray rows become failed rather than appearing stuck forever.
    pub async fn interrupt_running_download_jobs(&self) -> anyhow::Result<()> {
        let mut tx = self.pool.begin().await?;
        sqlx::query(
            r#"UPDATE download_jobs
               SET status = 'failed', error = 'Brazier restarted before this build completed',
                   updated_at = datetime('now')
               WHERE kind = 'runtime-build' AND status = 'downloading'"#,
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            r#"UPDATE download_jobs
               SET status = 'paused', updated_at = datetime('now')
               WHERE kind != 'runtime-build' AND status IN ('pending', 'downloading')"#,
        )
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn update_download_job_progress(
        &self,
        job_id: &str,
        bytes_downloaded: u64,
        total_bytes: Option<u64>,
    ) -> anyhow::Result<()> {
        sqlx::query(
            r#"UPDATE download_jobs
               SET bytes_downloaded = ?, total_bytes = ?, updated_at = datetime('now')
               WHERE id = ? AND status IN ('pending', 'downloading')"#,
        )
        .bind(bytes_downloaded as i64)
        .bind(total_bytes.map(|value| value as i64))
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Store a human-readable status for non-transfer work shown beside
    /// downloads, such as a runtime source build.
    pub async fn update_download_job_message(
        &self,
        job_id: &str,
        message: &str,
    ) -> anyhow::Result<()> {
        sqlx::query(
            r#"UPDATE download_jobs
               SET error = ?, updated_at = datetime('now')
               WHERE id = ? AND status IN ('pending', 'downloading')"#,
        )
        .bind(message)
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Associate an in-flight runtime build with the durable tray row so it
    /// can be cancelled from anywhere in the application.
    pub async fn set_download_job_payload(
        &self,
        job_id: &str,
        payload: &str,
    ) -> anyhow::Result<()> {
        sqlx::query(
            r#"UPDATE download_jobs
               SET payload = ?, updated_at = datetime('now')
               WHERE id = ? AND status = 'downloading'"#,
        )
        .bind(payload)
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn complete_download_job(
        &self,
        job_id: &str,
        sha256: &str,
        bytes: u64,
    ) -> anyhow::Result<()> {
        let result = sqlx::query(
            r#"UPDATE download_jobs
               SET status = 'completed', sha256 = ?, bytes_downloaded = ?,
                   total_bytes = ?, error = NULL, updated_at = datetime('now')
               WHERE id = ? AND status IN ('pending', 'downloading')"#,
        )
        .bind(sha256)
        .bind(bytes as i64)
        .bind(bytes as i64)
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        anyhow::ensure!(
            result.rows_affected() == 1,
            "that download is no longer running"
        );
        Ok(())
    }

    pub async fn start_download_job(&self, job_id: &str) -> anyhow::Result<()> {
        let result = sqlx::query(
            r#"UPDATE download_jobs
               SET status = 'downloading', updated_at = datetime('now')
               WHERE id = ? AND status IN ('pending', 'downloading')"#,
        )
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        anyhow::ensure!(
            result.rows_affected() == 1,
            "that download is no longer queued or running"
        );
        Ok(())
    }

    pub async fn cancel_download_job(&self, job_id: &str) -> anyhow::Result<()> {
        let result = sqlx::query(
            r#"UPDATE download_jobs
               SET status = 'cancelled', error = 'cancelled by user', updated_at = datetime('now')
               WHERE id = ? AND status IN ('pending', 'downloading', 'paused')"#,
        )
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        anyhow::ensure!(
            result.rows_affected() == 1,
            "that download is already settled or does not exist"
        );
        Ok(())
    }

    pub async fn fail_download_job(&self, job_id: &str, error: &str) -> anyhow::Result<()> {
        let result = sqlx::query(
            r#"UPDATE download_jobs
               SET status = 'failed', error = ?, updated_at = datetime('now')
               WHERE id = ? AND status IN ('pending', 'downloading')"#,
        )
        .bind(error)
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        anyhow::ensure!(
            result.rows_affected() == 1,
            "that download is no longer queued or running"
        );
        Ok(())
    }

    /// Statuses a job never leaves on its own, and so can be dismissed.
    const SETTLED_STATUSES: &'static str = "('completed', 'failed', 'cancelled')";

    /// Forget one settled job.
    ///
    /// Refuses anything still running or queued, so dismissing a row cannot
    /// silently orphan a transfer that is still writing to disk.
    pub async fn dismiss_download_job(&self, job_id: &str) -> anyhow::Result<()> {
        let affected = sqlx::query(&format!(
            "DELETE FROM download_jobs WHERE id = ? AND status IN {}",
            Self::SETTLED_STATUSES
        ))
        .bind(job_id)
        .execute(&self.pool)
        .await?
        .rows_affected();
        anyhow::ensure!(
            affected > 0,
            "that download is still active; cancel it before dismissing it"
        );
        Ok(())
    }

    /// Forget every settled job, returning how many were cleared.
    pub async fn dismiss_finished_download_jobs(&self) -> anyhow::Result<u64> {
        let affected = sqlx::query(&format!(
            "DELETE FROM download_jobs WHERE status IN {}",
            Self::SETTLED_STATUSES
        ))
        .execute(&self.pool)
        .await?
        .rows_affected();
        Ok(affected)
    }

    pub async fn list_download_jobs(&self, limit: i64) -> anyhow::Result<Vec<DownloadJob>> {
        let rows = sqlx::query_as::<_, DownloadJobRow>(
            r#"SELECT id, repo_id, filename, revision, status, kind, payload, label, bytes_downloaded, total_bytes,
                      sha256, error, created_at, updated_at
               FROM download_jobs ORDER BY updated_at DESC LIMIT ?"#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(Into::into).collect())
    }

    async fn get_download_job(&self, id: &str) -> anyhow::Result<DownloadJob> {
        let row = sqlx::query_as::<_, DownloadJobRow>(
            r#"SELECT id, repo_id, filename, revision, status, kind, payload, label, bytes_downloaded, total_bytes,
                      sha256, error, created_at, updated_at
               FROM download_jobs WHERE id = ?"#,
        )
        .bind(id)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.into())
    }

    pub async fn ping(&self) -> anyhow::Result<()> {
        let _: i64 = sqlx::query_scalar("SELECT 1").fetch_one(&self.pool).await?;
        Ok(())
    }

    pub async fn application_preference(&self, key: &str) -> anyhow::Result<Option<Value>> {
        let value: Option<String> =
            sqlx::query_scalar("SELECT value_json FROM application_preferences WHERE key = ?")
                .bind(key)
                .fetch_optional(&self.pool)
                .await?;
        value
            .map(|value| serde_json::from_str(&value).context("decode application preference"))
            .transpose()
    }

    pub async fn set_application_preference(&self, key: &str, value: &Value) -> anyhow::Result<()> {
        let encoded = serde_json::to_string(value)?;
        sqlx::query(
            r#"INSERT INTO application_preferences(key, value_json)
               VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   value_json = excluded.value_json,
                   updated_at = datetime('now')"#,
        )
        .bind(key)
        .bind(encoded)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

pub fn message_text(message: &Message) -> String {
    match &message.content {
        Value::String(text) => text.clone(),
        value => brazier_protocol::types::text_from_content(value),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tempfile::tempdir;

    use super::*;
    use brazier_protocol::types::CreateMessage;

    fn message(parent_id: Option<&str>, role: Role, content: Value) -> CreateMessage {
        CreateMessage {
            parent_id: parent_id.map(str::to_owned),
            role,
            content,
            model: None,
            tool_calls: None,
            tool_call_id: None,
            source: None,
            correlation_id: None,
            status: None,
            metadata: None,
        }
    }

    async fn open() -> (tempfile::TempDir, Database) {
        let dir = tempdir().unwrap();
        let db = Database::open(&dir.path().join("test.sqlite"))
            .await
            .unwrap();
        (dir, db)
    }

    #[tokio::test]
    async fn opens_for_concurrent_readers_and_writers() {
        let (_dir, db) = open().await;
        let journal_mode: String = sqlx::query_scalar("PRAGMA journal_mode")
            .fetch_one(&db.pool)
            .await
            .unwrap();
        let busy_timeout: i64 = sqlx::query_scalar("PRAGMA busy_timeout")
            .fetch_one(&db.pool)
            .await
            .unwrap();

        assert_eq!(journal_mode, "wal");
        assert_eq!(busy_timeout, 5_000);
    }

    #[tokio::test]
    async fn application_preferences_round_trip_json_values() {
        let (_dir, db) = open().await;
        assert_eq!(db.application_preference("welcome").await.unwrap(), None);

        db.set_application_preference("welcome", &json!({ "completed": true }))
            .await
            .unwrap();

        assert_eq!(
            db.application_preference("welcome").await.unwrap(),
            Some(json!({ "completed": true }))
        );
    }

    #[tokio::test]
    async fn migration_marks_existing_installations_as_onboarded() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("existing.sqlite");
        let options = SqliteConnectOptions::new()
            .filename(&path)
            .create_if_missing(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .unwrap();
        sqlx::query("CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT OR IGNORE INTO schema_migrations(version) VALUES (9)")
            .execute(&pool)
            .await
            .unwrap();
        // A real v9 install has the tables earlier migrations created. The
        // migration ladder under test reuses them (v12 alters conversations).
        sqlx::query("CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')), updated_at TEXT NOT NULL DEFAULT (datetime('now')))")
            .execute(&pool)
            .await
            .unwrap();
        pool.close().await;

        let db = Database::open(&path).await.unwrap();
        assert_eq!(
            db.application_preference("welcome").await.unwrap(),
            Some(json!({ "completed": true }))
        );
    }

    #[tokio::test]
    async fn creates_a_branched_message_graph() {
        let (_dir, db) = open().await;
        let conversation = db.create_conversation("Test").await.unwrap();
        let root = db
            .create_message(&conversation.id, message(None, Role::User, json!("Hello")))
            .await
            .unwrap();
        for text in ["First", "Alternative"] {
            let mut reply = message(Some(&root.id), Role::Assistant, json!(text));
            reply.model = Some("gguf:acme/demo/model.gguf".into());
            db.create_message(&conversation.id, reply).await.unwrap();
        }
        let messages = db.list_messages(&conversation.id).await.unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].parent_id, messages[2].parent_id);
    }

    #[tokio::test]
    async fn deletes_a_conversation_and_its_messages() {
        let (_dir, db) = open().await;
        let conversation = db.create_conversation("Disposable").await.unwrap();
        db.create_message(&conversation.id, message(None, Role::User, json!("Hello")))
            .await
            .unwrap();

        db.delete_conversation(&conversation.id).await.unwrap();

        assert!(db.get_conversation(&conversation.id).await.is_err());
        let count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM messages WHERE conversation_id = ?")
                .bind(&conversation.id)
                .fetch_one(&db.pool)
                .await
                .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn records_the_surface_and_turn_a_message_came_from() {
        let (_dir, db) = open().await;
        let conversation = db.create_conversation("Voice and text").await.unwrap();
        let mut spoken = message(None, Role::User, json!("What broke the build?"));
        spoken.source = Some("user_voice".into());
        spoken.correlation_id = Some("turn-1".into());
        spoken.status = Some("final".into());
        spoken.metadata = Some(json!({ "utterance_id": "utt-7" }));
        let stored = db.create_message(&conversation.id, spoken).await.unwrap();

        assert_eq!(stored.source.as_deref(), Some("user_voice"));
        assert_eq!(stored.correlation_id.as_deref(), Some("turn-1"));
        assert_eq!(
            stored
                .metadata
                .as_ref()
                .and_then(|value| value.get("utterance_id")),
            Some(&json!("utt-7"))
        );

        // The same turn's agent answer shares the correlation id, so the spoken
        // rendering can be linked to it instead of stored as a second answer.
        let mut answer = message(
            Some(&stored.id),
            Role::Assistant,
            json!("A test timed out."),
        );
        answer.source = Some("assistant_agent".into());
        answer.correlation_id = Some("turn-1".into());
        answer.status = Some("final".into());
        db.create_message(&conversation.id, answer).await.unwrap();

        let messages = db.list_messages(&conversation.id).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert!(
            messages
                .iter()
                .all(|entry| entry.correlation_id.as_deref() == Some("turn-1"))
        );
    }

    #[tokio::test]
    async fn plain_messages_keep_null_integration_fields() {
        let (_dir, db) = open().await;
        let conversation = db.create_conversation("Text only").await.unwrap();
        let stored = db
            .create_message(&conversation.id, message(None, Role::User, json!("Hi")))
            .await
            .unwrap();
        assert!(stored.source.is_none());
        assert!(stored.correlation_id.is_none());
        assert!(stored.status.is_none());
        assert!(stored.metadata.is_none());
        assert!(conversation.agent_session_id.is_none());
        assert!(conversation.summary.is_none());
    }

    #[tokio::test]
    async fn updating_a_message_relabels_it_without_deleting_it() {
        let (_dir, db) = open().await;
        let conversation = db.create_conversation("Interrupted").await.unwrap();
        let stored = db
            .create_message(
                &conversation.id,
                message(None, Role::Assistant, json!("partial…")),
            )
            .await
            .unwrap();
        let updated = db
            .update_message(
                &conversation.id,
                &stored.id,
                UpdateMessage {
                    content: Some(json!("The build passed.")),
                    status: Some("final".into()),
                    metadata: Some(json!({ "spoken": false })),
                },
            )
            .await
            .unwrap();
        assert_eq!(updated.content, json!("The build passed."));
        assert_eq!(updated.status.as_deref(), Some("final"));
        assert_eq!(db.list_messages(&conversation.id).await.unwrap().len(), 1);

        // A message from another conversation is refused rather than moved.
        let other = db.create_conversation("Other").await.unwrap();
        assert!(
            db.update_message(&other.id, &stored.id, UpdateMessage::default())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn message_writes_are_atomic_with_their_attachment_index() {
        let (_dir, db) = open().await;
        let conversation = db.create_conversation("Attachments").await.unwrap();
        let first_blob = "1".repeat(64);
        let second_blob = "2".repeat(64);
        let missing_blob = "3".repeat(64);
        for sha256 in [&first_blob, &second_blob] {
            db.upsert_attachment(sha256, "image/png", 12, None)
                .await
                .unwrap();
        }
        let content_for = |sha256: &str| {
            json!([{
                "type": "image_url",
                "brazier_blob": { "sha256": sha256 }
            }])
        };

        let stored = db
            .create_message(
                &conversation.id,
                message(None, Role::User, content_for(&first_blob)),
            )
            .await
            .unwrap();
        db.update_message(
            &conversation.id,
            &stored.id,
            UpdateMessage {
                content: Some(content_for(&second_blob)),
                status: Some("final".into()),
                metadata: Some(json!({ "edited": true })),
            },
        )
        .await
        .unwrap();
        let indexed: Vec<String> =
            sqlx::query_scalar("SELECT sha256 FROM message_attachments WHERE message_id = ?")
                .bind(&stored.id)
                .fetch_all(&db.pool)
                .await
                .unwrap();
        assert_eq!(indexed, vec![second_blob.clone()]);

        assert!(
            db.update_message(
                &conversation.id,
                &stored.id,
                UpdateMessage {
                    content: Some(content_for(&missing_blob)),
                    status: Some("failed".into()),
                    metadata: Some(json!({ "edited": false })),
                },
            )
            .await
            .is_err(),
            "a missing attachment must reject the whole edit"
        );
        let unchanged = db.get_message(&stored.id).await.unwrap();
        assert_eq!(unchanged.content, content_for(&second_blob));
        assert_eq!(unchanged.status.as_deref(), Some("final"));
        assert_eq!(unchanged.metadata, Some(json!({ "edited": true })));
        let indexed: Vec<String> =
            sqlx::query_scalar("SELECT sha256 FROM message_attachments WHERE message_id = ?")
                .bind(&stored.id)
                .fetch_all(&db.pool)
                .await
                .unwrap();
        assert_eq!(indexed, vec![second_blob]);

        assert!(
            db.create_message(
                &conversation.id,
                message(None, Role::User, content_for(&missing_blob)),
            )
            .await
            .is_err()
        );
        assert_eq!(db.list_messages(&conversation.id).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn conversations_bind_and_unbind_an_agent_session() {
        let (_dir, db) = open().await;
        let conversation = db.create_conversation("Shared").await.unwrap();
        let bound = db
            .update_conversation(
                &conversation.id,
                UpdateConversation {
                    title: None,
                    agent_session_id: Some(Some("agent-1".into())),
                    summary: Some("Investigating a failing test.".into()),
                },
            )
            .await
            .unwrap();
        assert_eq!(bound.agent_session_id.as_deref(), Some("agent-1"));
        assert_eq!(
            bound.summary.as_deref(),
            Some("Investigating a failing test.")
        );
        assert!(bound.summary_updated_at.is_some());

        // An absent key leaves the binding alone; an explicit null clears it.
        let retitled = db
            .update_conversation(
                &conversation.id,
                UpdateConversation {
                    title: Some("Renamed".into()),
                    ..UpdateConversation::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(retitled.title, "Renamed");
        assert_eq!(retitled.agent_session_id.as_deref(), Some("agent-1"));

        let unbound = db
            .update_conversation(
                &conversation.id,
                UpdateConversation {
                    agent_session_id: Some(None),
                    ..UpdateConversation::default()
                },
            )
            .await
            .unwrap();
        assert!(unbound.agent_session_id.is_none());
        assert!(unbound.summary.is_some());

        assert!(
            db.update_conversation("missing", UpdateConversation::default())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn an_absent_agent_session_key_is_distinct_from_an_explicit_null() {
        let absent: UpdateConversation = serde_json::from_value(json!({ "title": "T" })).unwrap();
        assert!(absent.agent_session_id.is_none());
        let cleared: UpdateConversation =
            serde_json::from_value(json!({ "agent_session_id": null })).unwrap();
        assert_eq!(cleared.agent_session_id, Some(None));
    }

    /// Dismissing is for tidying a list, so it must never reach a job that is
    /// still writing to disk.
    #[tokio::test]
    async fn only_settled_downloads_can_be_dismissed() {
        let (_dir, db) = open().await;
        let running = db
            .create_download_job("acme/models", "big.gguf", "main")
            .await
            .unwrap();
        let failed = db
            .create_download_job("acme/models", "broken.gguf", "main")
            .await
            .unwrap();
        db.fail_download_job(&failed.id, "connection reset")
            .await
            .unwrap();

        assert!(
            db.dismiss_download_job(&running.id).await.is_err(),
            "a queued transfer must survive a dismiss"
        );
        db.dismiss_download_job(&failed.id).await.unwrap();

        let remaining = db.list_download_jobs(30).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, running.id);
    }

    #[tokio::test]
    async fn clearing_finished_downloads_leaves_the_active_ones() {
        let (_dir, db) = open().await;
        let running = db
            .create_download_job("acme/models", "live.gguf", "main")
            .await
            .unwrap();
        for (name, settle) in [("done.gguf", "complete"), ("gone.gguf", "cancel")] {
            let job = db
                .create_download_job("acme/models", name, "main")
                .await
                .unwrap();
            match settle {
                "complete" => db.complete_download_job(&job.id, "abc", 10).await.unwrap(),
                _ => db.cancel_download_job(&job.id).await.unwrap(),
            }
        }

        assert_eq!(db.dismiss_finished_download_jobs().await.unwrap(), 2);
        let remaining = db.list_download_jobs(30).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, running.id);
        // Clearing an already-clean list is not an error.
        assert_eq!(db.dismiss_finished_download_jobs().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn cancellation_wins_races_with_completion_and_resume() {
        let (_dir, db) = open().await;
        let job = db
            .create_download_job("acme/models", "cancelled.gguf", "main")
            .await
            .unwrap();
        db.cancel_download_job(&job.id).await.unwrap();

        assert!(db.start_download_job(&job.id).await.is_err());
        db.update_download_job_progress(&job.id, 9, Some(10))
            .await
            .unwrap();
        assert!(db.fail_download_job(&job.id, "late failure").await.is_err());
        assert!(
            db.complete_download_job(&job.id, "abc", 10).await.is_err(),
            "late completion must not resurrect a cancelled download"
        );
        let cancelled = db.get_download_job_public(&job.id).await.unwrap();
        assert_eq!(cancelled.status, "cancelled");
        assert_eq!(cancelled.bytes_downloaded, None);
        assert_eq!(cancelled.error.as_deref(), Some("cancelled by user"));

        db.requeue_download_job(&job.id).await.unwrap();
        assert!(
            db.requeue_download_job(&job.id).await.is_err(),
            "the same job must not be enqueued by two resume requests"
        );
        assert_eq!(
            db.get_download_job_public(&job.id).await.unwrap().status,
            "pending"
        );
    }

    #[tokio::test]
    async fn paused_downloads_can_be_cancelled_but_settled_jobs_cannot() {
        let (_dir, db) = open().await;
        let job = db
            .create_download_job("acme/models", "paused.gguf", "main")
            .await
            .unwrap();
        db.pause_download_job(&job.id).await.unwrap();
        db.cancel_download_job(&job.id).await.unwrap();
        assert_eq!(
            db.get_download_job_public(&job.id).await.unwrap().status,
            "cancelled"
        );
        assert!(db.cancel_download_job(&job.id).await.is_err());
        assert!(db.cancel_download_job("missing").await.is_err());
        assert!(db.pause_download_job("missing").await.is_err());
    }

    #[tokio::test]
    async fn importing_keeps_source_labels_but_not_live_turn_ids() {
        let (dir, db) = open().await;
        let conversation = db.create_conversation("Exported").await.unwrap();
        let mut spoken = message(None, Role::User, json!("Say that again"));
        spoken.source = Some("user_voice".into());
        spoken.correlation_id = Some("turn-9".into());
        spoken.status = Some("final".into());
        let spoken = db.create_message(&conversation.id, spoken).await.unwrap();
        db.create_run_snapshot(
            &conversation.id,
            CreateRunSnapshot {
                parent_message_id: Some(spoken.id.clone()),
                assistant_message_id: None,
                model: "gguf:demo".into(),
                settings: json!({ "temperature": 0.4 }),
                tool_calls: Some(json!([{ "name": "calculator" }])),
                response_text: Some("Four".into()),
                error: None,
            },
        )
        .await
        .unwrap();

        let export = db
            .export_conversation(dir.path(), &conversation.id)
            .await
            .unwrap();
        let imported = db.import_conversation(dir.path(), export).await.unwrap();
        let messages = db.list_messages(&imported.id).await.unwrap();
        assert_eq!(messages[0].source.as_deref(), Some("user_voice"));
        assert_eq!(messages[0].status.as_deref(), Some("final"));
        assert!(messages[0].correlation_id.is_none());
        let snapshots = db.list_run_snapshots(&imported.id).await.unwrap();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].model, "gguf:demo");
        assert_eq!(snapshots[0].parent_message_id, Some(messages[0].id.clone()));
        assert_eq!(snapshots[0].settings, json!({ "temperature": 0.4 }));
        assert_eq!(
            snapshots[0].tool_calls,
            Some(json!([{ "name": "calculator" }]))
        );
    }
}

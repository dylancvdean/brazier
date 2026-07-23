use std::path::Path;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use sqlx::{
    FromRow, SqlitePool,
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
};
use uuid::Uuid;

use crate::{
    blob_store,
    types::{Conversation, CreateMessage, Message, Role},
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
    pub status: String,
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
    pool: SqlitePool,
}

#[derive(FromRow)]
struct MessageRow {
    id: String,
    conversation_id: String,
    parent_id: Option<String>,
    role: String,
    content_json: String,
    model: Option<String>,
    created_at: String,
}

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
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
            .context("open sqlite database")?;
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

        if version < 1 {
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
            .execute(&self.pool)
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
            .execute(&self.pool)
            .await?;
            sqlx::query(
                "CREATE INDEX IF NOT EXISTS messages_conversation_created ON messages(conversation_id, created_at)",
            )
            .execute(&self.pool)
            .await?;
            sqlx::query("INSERT INTO schema_migrations(version) VALUES (1)")
                .execute(&self.pool)
                .await?;
            version = 1;
        }

        if version < 2 {
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
            .execute(&self.pool)
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
            .execute(&self.pool)
            .await?;
            sqlx::query("INSERT INTO schema_migrations(version) VALUES (2)")
                .execute(&self.pool)
                .await?;
            version = 2;
        }

        if version < 3 {
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
            .execute(&self.pool)
            .await?;
            sqlx::query(
                "CREATE INDEX IF NOT EXISTS run_snapshots_conversation_created ON run_snapshots(conversation_id, created_at DESC)",
            )
            .execute(&self.pool)
            .await?;
            sqlx::query("INSERT INTO schema_migrations(version) VALUES (3)")
                .execute(&self.pool)
                .await?;
            version = 3;
        }

        if version < 4 {
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
            .execute(&self.pool)
            .await?;
            sqlx::query(
                "CREATE INDEX IF NOT EXISTS download_jobs_updated ON download_jobs(updated_at DESC)",
            )
            .execute(&self.pool)
            .await?;
            sqlx::query("INSERT INTO schema_migrations(version) VALUES (4)")
                .execute(&self.pool)
                .await?;
        }

        Ok(())
    }

    fn escape_like(query: &str) -> String {
        query
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_")
    }

    pub async fn list_conversations(&self, query: Option<&str>) -> anyhow::Result<Vec<Conversation>> {
        let Some(raw) = query.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(sqlx::query_as::<_, Conversation>(
                r#"SELECT id, title, created_at, updated_at
                   FROM conversations ORDER BY updated_at DESC"#,
            )
            .fetch_all(&self.pool)
            .await?);
        };
        let pattern = format!("%{}%", Self::escape_like(raw));
        Ok(sqlx::query_as::<_, Conversation>(
            r#"SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at
               FROM conversations c
               LEFT JOIN messages m ON m.conversation_id = c.id
               WHERE c.title LIKE ?1 ESCAPE '\'
                  OR m.content_json LIKE ?1 ESCAPE '\'
               ORDER BY c.updated_at DESC
               LIMIT 80"#,
        )
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

    async fn get_conversation(&self, id: &str) -> anyhow::Result<Conversation> {
        Ok(sqlx::query_as::<_, Conversation>(
            r#"SELECT id, title, created_at, updated_at
               FROM conversations WHERE id = ?"#,
        )
        .bind(id)
        .fetch_one(&self.pool)
        .await?)
    }

    pub async fn list_messages(&self, conversation_id: &str) -> anyhow::Result<Vec<Message>> {
        let rows = sqlx::query_as::<_, MessageRow>(
            r#"SELECT id, conversation_id, parent_id, role, content_json, model, created_at
               FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid"#,
        )
        .bind(conversation_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(TryInto::try_into).collect()
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
        sqlx::query(
            r#"INSERT INTO messages(id, conversation_id, parent_id, role, content_json, model)
               VALUES(?, ?, ?, ?, ?, ?)"#,
        )
        .bind(&id)
        .bind(conversation_id)
        .bind(&message.parent_id)
        .bind(message.role.as_str())
        .bind(content_json)
        .bind(&message.model)
        .execute(&self.pool)
        .await?;
        sqlx::query("UPDATE conversations SET updated_at = datetime('now') WHERE id = ?")
            .bind(conversation_id)
            .execute(&self.pool)
            .await?;
        self.index_message_blobs(&id, &message.content).await?;
        self.get_message(&id).await
    }

    async fn index_message_blobs(&self, message_id: &str, content: &Value) -> anyhow::Result<()> {
        for sha256 in blob_store::blob_refs_in_content(content) {
            blob_store::validate_sha256(&sha256)?;
            sqlx::query(
                "INSERT OR IGNORE INTO message_attachments(message_id, sha256) VALUES(?, ?)",
            )
            .bind(message_id)
            .bind(&sha256)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
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
        let row = sqlx::query_as::<_, MessageRow>(
            r#"SELECT id, conversation_id, parent_id, role, content_json, model, created_at
               FROM messages WHERE id = ?"#,
        )
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
        for blob in &export.blobs {
            blob_store::validate_sha256(&blob.sha256)?;
            let bytes = base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                blob.data_base64.trim(),
            )
            .context("decode export blob")?;
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
        let conversation = self.create_conversation(title).await?;
        let mut id_map: HashMap<String, String> = HashMap::new();
        for message in &export.messages {
            id_map.insert(message.id.clone(), Uuid::new_v4().to_string());
        }
        for message in export.messages {
            let new_id = id_map
                .get(&message.id)
                .context("missing remapped message id")?;
            let parent_id = message
                .parent_id
                .as_ref()
                .and_then(|parent| id_map.get(parent))
                .cloned();
            let content_json = serde_json::to_string(&message.content)?;
            sqlx::query(
                r#"INSERT INTO messages(id, conversation_id, parent_id, role, content_json, model, created_at)
                   VALUES(?, ?, ?, ?, ?, ?, ?)"#,
            )
            .bind(new_id)
            .bind(&conversation.id)
            .bind(parent_id)
            .bind(message.role.as_str())
            .bind(content_json)
            .bind(&message.model)
            .bind(&message.created_at)
            .execute(&self.pool)
            .await?;
            self.index_message_blobs(new_id, &message.content).await?;
        }
        sqlx::query("UPDATE conversations SET updated_at = datetime('now') WHERE id = ?")
            .bind(&conversation.id)
            .execute(&self.pool)
            .await?;
        self.get_conversation(&conversation.id).await
    }

    pub async fn create_download_job(
        &self,
        repo_id: &str,
        filename: &str,
        revision: &str,
    ) -> anyhow::Result<DownloadJob> {
        let id = Uuid::new_v4().to_string();
        sqlx::query(
            r#"INSERT INTO download_jobs(id, repo_id, filename, revision, status)
               VALUES(?, ?, ?, ?, 'running')"#,
        )
        .bind(&id)
        .bind(repo_id)
        .bind(filename)
        .bind(revision)
        .execute(&self.pool)
        .await?;
        self.get_download_job(&id).await
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
               WHERE id = ?"#,
        )
        .bind(bytes_downloaded as i64)
        .bind(total_bytes.map(|value| value as i64))
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
        sqlx::query(
            r#"UPDATE download_jobs
               SET status = 'completed', sha256 = ?, bytes_downloaded = ?,
                   total_bytes = ?, error = NULL, updated_at = datetime('now')
               WHERE id = ?"#,
        )
        .bind(sha256)
        .bind(bytes as i64)
        .bind(bytes as i64)
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn start_download_job(&self, job_id: &str) -> anyhow::Result<()> {
        sqlx::query(
            r#"UPDATE download_jobs
               SET status = 'downloading', updated_at = datetime('now')
               WHERE id = ?"#,
        )
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn cancel_download_job(&self, job_id: &str) -> anyhow::Result<()> {
        sqlx::query(
            r#"UPDATE download_jobs
               SET status = 'cancelled', error = 'cancelled by user', updated_at = datetime('now')
               WHERE id = ? AND status IN ('pending', 'downloading')"#,
        )
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn fail_download_job(&self, job_id: &str, error: &str) -> anyhow::Result<()> {
        sqlx::query(
            r#"UPDATE download_jobs
               SET status = 'failed', error = ?, updated_at = datetime('now')
               WHERE id = ?"#,
        )
        .bind(error)
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_download_jobs(&self, limit: i64) -> anyhow::Result<Vec<DownloadJob>> {
        let rows = sqlx::query_as::<_, DownloadJobRow>(
            r#"SELECT id, repo_id, filename, revision, status, bytes_downloaded, total_bytes,
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
            r#"SELECT id, repo_id, filename, revision, status, bytes_downloaded, total_bytes,
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
}

pub fn message_text(message: &Message) -> String {
    match &message.content {
        Value::String(text) => text.clone(),
        value => crate::types::text_from_content(value),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tempfile::tempdir;

    use super::*;
    use crate::types::CreateMessage;

    #[tokio::test]
    async fn creates_a_branched_message_graph() {
        let dir = tempdir().unwrap();
        let db = Database::open(&dir.path().join("test.sqlite"))
            .await
            .unwrap();
        let conversation = db.create_conversation("Test").await.unwrap();
        let root = db
            .create_message(
                &conversation.id,
                CreateMessage {
                    parent_id: None,
                    role: Role::User,
                    content: json!("Hello"),
                    model: None,
                },
            )
            .await
            .unwrap();
        for text in ["First", "Alternative"] {
            db.create_message(
                &conversation.id,
                CreateMessage {
                    parent_id: Some(root.id.clone()),
                    role: Role::Assistant,
                    content: json!(text),
                    model: Some("gguf:acme/demo/model.gguf".into()),
                },
            )
            .await
            .unwrap();
        }
        let messages = db.list_messages(&conversation.id).await.unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].parent_id, messages[2].parent_id);
    }
}

use std::path::Path;

use anyhow::Context;
use serde_json::Value;
use sqlx::{
    FromRow, SqlitePool,
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
};
use uuid::Uuid;

use crate::types::{Conversation, CreateMessage, Message, Role};

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
        Ok(())
    }

    pub async fn list_conversations(&self) -> anyhow::Result<Vec<Conversation>> {
        Ok(sqlx::query_as::<_, Conversation>(
            r#"SELECT id, title, created_at, updated_at
               FROM conversations ORDER BY updated_at DESC"#,
        )
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
        self.get_message(&id).await
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
                    model: Some("brazier/mock".into()),
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

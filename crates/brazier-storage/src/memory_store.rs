//! SQLite persistence for the chat memory system.
//!
//! Memories are durable facts the model saves across conversations, edited in
//! the Settings Chat section, and consolidated by dreaming. The store is
//! global: a memory does not belong to a conversation, though the row records
//! which turn produced it so it can be traced and pruned.

use anyhow::Context;
use sqlx::FromRow;
use uuid::Uuid;

use crate::db::Database;
use brazier_protocol::types::{CreateMemory, Memory, UpdateMemory};

#[derive(FromRow)]
struct MemoryRow {
    id: String,
    text: String,
    kind: String,
    pinned: i64,
    tags_json: String,
    source_conversation_id: Option<String>,
    source_message_id: Option<String>,
    created_at: String,
    updated_at: String,
}

impl TryFrom<MemoryRow> for Memory {
    type Error = anyhow::Error;

    fn try_from(row: MemoryRow) -> Result<Self, Self::Error> {
        let tags: Vec<String> = serde_json::from_str(&row.tags_json)
            .with_context(|| format!("decode memory {} tags", row.id))?;
        Ok(Self {
            id: row.id,
            text: row.text,
            kind: row.kind,
            pinned: row.pinned != 0,
            tags,
            source_conversation_id: row.source_conversation_id,
            source_message_id: row.source_message_id,
            created_at: row.created_at,
            updated_at: row.updated_at,
        })
    }
}

const MEMORY_COLUMNS: &str = "id, text, kind, pinned, tags_json, source_conversation_id,
     source_message_id, created_at, updated_at";

impl Database {
    /// All memories, pinned first then most recently updated, bounded to a sane
    /// editor payload. Use [`Self::search_memories`] for recall.
    pub async fn list_memories(&self) -> anyhow::Result<Vec<Memory>> {
        let rows = sqlx::query_as::<_, MemoryRow>(&format!(
            "SELECT {MEMORY_COLUMNS} FROM memories
             ORDER BY pinned DESC, updated_at DESC LIMIT 2000"
        ))
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(TryInto::try_into).collect()
    }

    /// Keyword search for recall injection. Matches are bounded to `limit` so
    /// the renderer can fit them into a context budget without loading the
    /// whole store. Pinned memories are returned first.
    pub async fn search_memories(
        &self,
        query: Option<&str>,
        limit: i64,
    ) -> anyhow::Result<Vec<Memory>> {
        let Some(raw) = query.map(str::trim).filter(|value| !value.is_empty()) else {
            return self.query_memories_limit(limit).await;
        };
        let pattern = format!("%{}%", Database::escape_like(raw));
        let rows = sqlx::query_as::<_, MemoryRow>(&format!(
            r#"SELECT {MEMORY_COLUMNS} FROM memories
               WHERE text LIKE ? ESCAPE '\' OR tags_json LIKE ? ESCAPE '\'
               ORDER BY pinned DESC, updated_at DESC LIMIT ?"#
        ))
        .bind(&pattern)
        .bind(&pattern)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(TryInto::try_into).collect()
    }

    /// Memories referencing a conversation, so dreaming can consolidate what a
    /// conversation's turns saved and the UI can show provenance.
    pub async fn list_memories_for_conversation(
        &self,
        conversation_id: &str,
    ) -> anyhow::Result<Vec<Memory>> {
        let rows = sqlx::query_as::<_, MemoryRow>(&format!(
            "SELECT {MEMORY_COLUMNS} FROM memories WHERE source_conversation_id = ?
             ORDER BY created_at ASC"
        ))
        .bind(conversation_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(TryInto::try_into).collect()
    }

    pub async fn get_memory(&self, id: &str) -> anyhow::Result<Memory> {
        let row = sqlx::query_as::<_, MemoryRow>(&format!(
            "SELECT {MEMORY_COLUMNS} FROM memories WHERE id = ?"
        ))
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        let Some(row) = row else {
            anyhow::bail!("memory {id} not found");
        };
        row.try_into()
    }

    /// Insert a memory. An empty or whitespace-only body is rejected so the
    /// model cannot litter the store with blanks.
    pub async fn create_memory(&self, input: CreateMemory) -> anyhow::Result<Memory> {
        let text = input.text.trim().to_owned();
        anyhow::ensure!(!text.is_empty(), "memory text must not be empty");
        let id = Uuid::new_v4().to_string();
        let kind = input
            .kind
            .as_deref()
            .map(str::trim)
            .filter(|kind| !kind.is_empty())
            .unwrap_or("fact");
        let tags = input.tags.unwrap_or_default();
        let tags_json = serde_json::to_string(&tags)?;
        sqlx::query(
            r#"INSERT INTO memories(id, text, kind, pinned, tags_json, source_conversation_id,
                                    source_message_id)
               VALUES(?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(&id)
        .bind(&text)
        .bind(kind)
        .bind(i64::from(input.pinned.unwrap_or(false)))
        .bind(tags_json)
        .bind(&input.source_conversation_id)
        .bind(&input.source_message_id)
        .execute(&self.pool)
        .await?;
        self.get_memory(&id).await
    }

    /// Apply a partial update to a memory. Absent fields are left alone.
    pub async fn update_memory(&self, id: &str, update: UpdateMemory) -> anyhow::Result<Memory> {
        let existing = self.get_memory(id).await?;
        let mut next = existing.clone();
        if let Some(text) = update
            .text
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
        {
            next.text = text.to_owned();
        }
        if let Some(kind) = update
            .kind
            .as_deref()
            .map(str::trim)
            .filter(|k| !k.is_empty())
        {
            next.kind = kind.to_owned();
        }
        if let Some(pinned) = update.pinned {
            next.pinned = pinned;
        }
        if let Some(tags) = update.tags {
            next.tags = tags;
        }
        let tags_json = serde_json::to_string(&next.tags)?;
        sqlx::query(
            r#"UPDATE memories SET text = ?, kind = ?, pinned = ?, tags_json = ?,
                   updated_at = datetime('now')
               WHERE id = ?"#,
        )
        .bind(&next.text)
        .bind(&next.kind)
        .bind(i64::from(next.pinned))
        .bind(tags_json)
        .bind(id)
        .execute(&self.pool)
        .await?;
        self.get_memory(id).await
    }

    /// Delete a memory, returning whether a row was actually removed.
    pub async fn delete_memory(&self, id: &str) -> anyhow::Result<bool> {
        let result = sqlx::query("DELETE FROM memories WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }

    /// Delete every memory a conversation produced, used when an incognito
    /// conversation is discarded defensively.
    pub async fn delete_memories_for_conversation(
        &self,
        conversation_id: &str,
    ) -> anyhow::Result<u64> {
        let result = sqlx::query("DELETE FROM memories WHERE source_conversation_id = ?")
            .bind(conversation_id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    /// The most recently updated `limit` memories, for recall when there is no
    /// search query.
    async fn query_memories_limit(&self, limit: i64) -> anyhow::Result<Vec<Memory>> {
        let rows = sqlx::query_as::<_, MemoryRow>(&format!(
            "SELECT {MEMORY_COLUMNS} FROM memories
             ORDER BY pinned DESC, updated_at DESC LIMIT ?"
        ))
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(TryInto::try_into).collect()
    }
}

#[cfg(test)]
mod tests {
    use brazier_protocol::types::{CreateMemory, UpdateMemory};
    use tempfile::tempdir;

    use super::*;

    fn memory(text: &str) -> CreateMemory {
        CreateMemory {
            text: text.to_owned(),
            kind: None,
            pinned: None,
            tags: None,
            source_conversation_id: None,
            source_message_id: None,
        }
    }

    async fn open() -> (tempfile::TempDir, Database) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("memories.sqlite");
        let db = Database::open(&path).await.unwrap();
        (dir, db)
    }

    #[tokio::test]
    async fn creates_lists_and_searches_memories() {
        let (_dir, db) = open().await;
        db.create_memory(memory("User prefers concise answers over verbosity."))
            .await
            .unwrap();
        db.create_memory(memory("User is allergic to peanuts."))
            .await
            .unwrap();
        db.create_memory(memory("User works at a weather startup."))
            .await
            .unwrap();

        let all = db.list_memories().await.unwrap();
        assert_eq!(all.len(), 3);

        let hits = db.search_memories(Some("peanuts"), 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("allergic"));

        let capped = db.search_memories(None, 2).await.unwrap();
        assert_eq!(capped.len(), 2);
    }

    #[tokio::test]
    async fn rejects_blank_memories() {
        let (_dir, db) = open().await;
        assert!(db.create_memory(memory("   ")).await.is_err());
        assert!(db.create_memory(memory("")).await.is_err());
    }

    #[tokio::test]
    async fn updates_and_deletes_memories() {
        let (_dir, db) = open().await;
        let created = db.create_memory(memory("User likes tea.")).await.unwrap();

        let updated = db
            .update_memory(
                &created.id,
                UpdateMemory {
                    text: Some("User prefers coffee now.".to_owned()),
                    pinned: Some(true),
                    tags: Some(vec!["drinks".to_owned()]),
                    ..UpdateMemory::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(updated.text, "User prefers coffee now.");
        assert!(updated.pinned);
        assert_eq!(updated.tags, vec!["drinks"]);

        assert!(db.delete_memory(&created.id).await.unwrap());
        assert!(!db.delete_memory(&created.id).await.unwrap());
        assert!(db.get_memory(&created.id).await.is_err());
    }

    #[tokio::test]
    async fn search_matches_tags_too() {
        let (_dir, db) = open().await;
        db.create_memory(CreateMemory {
            text: "User runs a marathon every spring.".to_owned(),
            kind: None,
            pinned: None,
            tags: Some(vec!["health".to_owned()]),
            source_conversation_id: None,
            source_message_id: None,
        })
        .await
        .unwrap();
        let hits = db.search_memories(Some("health"), 10).await.unwrap();
        assert_eq!(hits.len(), 1);
    }
}

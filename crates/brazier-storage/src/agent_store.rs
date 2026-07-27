//! SQLite persistence for Agent mode.
//!
//! The application owns this state, not the runtime. A session restored here
//! must be resumable after the app restarts, after a runtime upgrade, and
//! without replaying any command.

use anyhow::Context;
use serde_json::Value;
use sqlx::{FromRow, Row};
use uuid::Uuid;

use crate::db::Database;
use brazier_protocol::agent_types::{
    AgentApproval, AgentElevationRequest, AgentEnvironment, AgentMessageRecord,
    AgentPermissionMode, AgentSessionRecord, AppendAgentMessage, ApprovalScope, ApprovalStatus,
    CreateAgentSession, SandboxDescription, ToolExecutionRecord, ToolRiskLevel, UpdateAgentSession,
    grant_key,
};

/// How long a pending approval stays answerable.
pub const APPROVAL_TTL_SECONDS: i64 = 900;

#[derive(FromRow)]
struct SessionRow {
    id: String,
    title: String,
    workspace_path: Option<String>,
    model: String,
    runtime_id: String,
    permission_mode: String,
    permission_settings_json: String,
    enabled_tools_json: Option<String>,
    last_run_status: String,
    compaction_json: Option<String>,
    runtime_metadata_json: Option<String>,
    created_at: String,
    updated_at: String,
}

impl TryFrom<SessionRow> for AgentSessionRecord {
    type Error = anyhow::Error;

    fn try_from(row: SessionRow) -> anyhow::Result<Self> {
        Ok(Self {
            id: row.id,
            title: row.title,
            workspace_path: row.workspace_path,
            model: row.model,
            runtime_id: row.runtime_id,
            permission_mode: parse_permission_mode(&row.permission_mode),
            permission_settings: serde_json::from_str(&row.permission_settings_json)
                .unwrap_or_default(),
            enabled_tools: row
                .enabled_tools_json
                .as_deref()
                .and_then(|raw| serde_json::from_str(raw).ok()),
            last_run_status: row.last_run_status,
            compaction: row
                .compaction_json
                .as_deref()
                .and_then(|raw| serde_json::from_str(raw).ok()),
            runtime_metadata: row
                .runtime_metadata_json
                .as_deref()
                .and_then(|raw| serde_json::from_str(raw).ok()),
            created_at: row.created_at,
            updated_at: row.updated_at,
        })
    }
}

fn parse_permission_mode(value: &str) -> AgentPermissionMode {
    match value {
        "sandbox-only" => AgentPermissionMode::SandboxOnly,
        "skip-permissions" => AgentPermissionMode::SkipPermissions,
        _ => AgentPermissionMode::Ask,
    }
}

fn parse_environment(value: &str) -> AgentEnvironment {
    match value {
        "host" => AgentEnvironment::Host,
        _ => AgentEnvironment::Sandbox,
    }
}

fn parse_risk(value: &str) -> ToolRiskLevel {
    match value {
        "safe" => ToolRiskLevel::Safe,
        "write" => ToolRiskLevel::Write,
        "execute" => ToolRiskLevel::Execute,
        "destructive" => ToolRiskLevel::Destructive,
        _ => ToolRiskLevel::Read,
    }
}

/// A new tool execution row.
pub struct NewToolExecution {
    pub session_id: String,
    pub run_id: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool: String,
    pub arguments: Value,
    pub environment: AgentEnvironment,
    pub risk: ToolRiskLevel,
    pub status: String,
    pub exit_code: Option<i32>,
    pub output_preview: Option<String>,
    pub artifact_id: Option<String>,
    pub truncated: bool,
    pub changed_paths: Vec<String>,
    pub sandbox: Option<SandboxDescription>,
    pub approval_id: Option<String>,
    pub error: Option<String>,
    pub duration_ms: Option<u64>,
}

/// A new approval request awaiting the user.
pub struct NewApproval {
    pub session_id: String,
    pub tool: String,
    pub arguments: Value,
    pub arguments_hash: String,
    pub environment: AgentEnvironment,
    pub risk: ToolRiskLevel,
    pub scope_key: String,
    pub allow_session_scope: bool,
    pub elevation: AgentElevationRequest,
    pub sandbox: SandboxDescription,
    pub summary: String,
}

impl Database {
    pub async fn create_agent_session(
        &self,
        request: CreateAgentSession,
    ) -> anyhow::Result<AgentSessionRecord> {
        let id = Uuid::new_v4().to_string();
        let settings = request.permission_settings.unwrap_or_default();
        let mode = request.permission_mode.unwrap_or(AgentPermissionMode::Ask);
        sqlx::query(
            r#"INSERT INTO agent_sessions(
                   id, title, workspace_path, model, runtime_id, permission_mode,
                   permission_settings_json, enabled_tools_json, last_run_status)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'idle')"#,
        )
        .bind(&id)
        .bind(request.title.unwrap_or_else(|| "Agent task".to_owned()))
        .bind(&request.workspace_path)
        .bind(&request.model)
        .bind(&request.runtime_id)
        .bind(mode.as_str())
        .bind(serde_json::to_string(&settings)?)
        .bind(
            request
                .enabled_tools
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?,
        )
        .execute(&self.pool)
        .await?;
        self.agent_session(&id).await
    }

    pub async fn agent_session(&self, id: &str) -> anyhow::Result<AgentSessionRecord> {
        let row = sqlx::query_as::<_, SessionRow>(
            r#"SELECT id, title, workspace_path, model, runtime_id, permission_mode,
                      permission_settings_json, enabled_tools_json, last_run_status,
                      compaction_json, runtime_metadata_json, created_at, updated_at
               FROM agent_sessions WHERE id = ?"#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?
        .with_context(|| format!("agent session {id} does not exist"))?;
        row.try_into()
    }

    pub async fn list_agent_sessions(&self) -> anyhow::Result<Vec<AgentSessionRecord>> {
        let rows = sqlx::query_as::<_, SessionRow>(
            r#"SELECT id, title, workspace_path, model, runtime_id, permission_mode,
                      permission_settings_json, enabled_tools_json, last_run_status,
                      compaction_json, runtime_metadata_json, created_at, updated_at
               FROM agent_sessions ORDER BY updated_at DESC LIMIT 100"#,
        )
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(TryInto::try_into).collect()
    }

    pub async fn update_agent_session(
        &self,
        id: &str,
        update: UpdateAgentSession,
    ) -> anyhow::Result<AgentSessionRecord> {
        // One statement per supplied field keeps the absent-versus-null
        // distinction that a single dynamic UPDATE would lose.
        if let Some(title) = update.title {
            sqlx::query("UPDATE agent_sessions SET title = ? WHERE id = ?")
                .bind(title)
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        if let Some(workspace) = update.workspace_path {
            sqlx::query("UPDATE agent_sessions SET workspace_path = ? WHERE id = ?")
                .bind(workspace)
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        if let Some(model) = update.model {
            sqlx::query("UPDATE agent_sessions SET model = ? WHERE id = ?")
                .bind(model)
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        if let Some(mode) = update.permission_mode {
            sqlx::query("UPDATE agent_sessions SET permission_mode = ? WHERE id = ?")
                .bind(mode.as_str())
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        if let Some(settings) = update.permission_settings {
            sqlx::query("UPDATE agent_sessions SET permission_settings_json = ? WHERE id = ?")
                .bind(serde_json::to_string(&settings)?)
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        if let Some(tools) = update.enabled_tools {
            sqlx::query("UPDATE agent_sessions SET enabled_tools_json = ? WHERE id = ?")
                .bind(serde_json::to_string(&tools)?)
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        if let Some(status) = update.last_run_status {
            sqlx::query("UPDATE agent_sessions SET last_run_status = ? WHERE id = ?")
                .bind(status)
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        if let Some(compaction) = update.compaction {
            sqlx::query("UPDATE agent_sessions SET compaction_json = ? WHERE id = ?")
                .bind(serde_json::to_string(&compaction)?)
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        if let Some(metadata) = update.runtime_metadata {
            sqlx::query("UPDATE agent_sessions SET runtime_metadata_json = ? WHERE id = ?")
                .bind(serde_json::to_string(&metadata)?)
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        sqlx::query("UPDATE agent_sessions SET updated_at = datetime('now') WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        self.agent_session(id).await
    }

    pub async fn delete_agent_session(&self, id: &str) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM agent_sessions WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn agent_messages(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Vec<AgentMessageRecord>> {
        let rows = sqlx::query(
            r#"SELECT id, session_id, seq, role, payload_json, created_at
               FROM agent_messages WHERE session_id = ? ORDER BY seq"#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| {
                let payload_json: String = row.try_get("payload_json")?;
                Ok(AgentMessageRecord {
                    id: row.try_get("id")?,
                    session_id: row.try_get("session_id")?,
                    seq: row.try_get("seq")?,
                    role: row.try_get("role")?,
                    payload: serde_json::from_str(&payload_json)?,
                    created_at: row.try_get("created_at")?,
                })
            })
            .collect()
    }

    /// Append transcript entries, or replace the transcript wholesale when
    /// compaction rewrites history.
    pub async fn append_agent_messages(
        &self,
        session_id: &str,
        messages: &[AppendAgentMessage],
        replace: bool,
    ) -> anyhow::Result<Vec<AgentMessageRecord>> {
        let mut tx = self.pool.begin().await?;
        if replace {
            sqlx::query("DELETE FROM agent_messages WHERE session_id = ?")
                .bind(session_id)
                .execute(&mut *tx)
                .await?;
        }
        let mut seq: i64 = sqlx::query_scalar(
            "SELECT COALESCE(MAX(seq), 0) FROM agent_messages WHERE session_id = ?",
        )
        .bind(session_id)
        .fetch_one(&mut *tx)
        .await?;
        for message in messages {
            seq += 1;
            sqlx::query(
                r#"INSERT INTO agent_messages(id, session_id, seq, role, payload_json)
                   VALUES(?, ?, ?, ?, ?)"#,
            )
            .bind(Uuid::new_v4().to_string())
            .bind(session_id)
            .bind(seq)
            .bind(&message.role)
            .bind(serde_json::to_string(&message.payload)?)
            .execute(&mut *tx)
            .await?;
        }
        sqlx::query("UPDATE agent_sessions SET updated_at = datetime('now') WHERE id = ?")
            .bind(session_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        self.agent_messages(session_id).await
    }

    pub async fn record_tool_execution(
        &self,
        execution: NewToolExecution,
    ) -> anyhow::Result<ToolExecutionRecord> {
        let id = Uuid::new_v4().to_string();
        sqlx::query(
            r#"INSERT INTO agent_tool_executions(
                   id, session_id, run_id, tool_call_id, tool, arguments_json, environment,
                   risk, status, exit_code, output_preview, artifact_id, truncated,
                   changed_paths_json, sandbox_json, approval_id, error, duration_ms)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(&id)
        .bind(&execution.session_id)
        .bind(&execution.run_id)
        .bind(&execution.tool_call_id)
        .bind(&execution.tool)
        .bind(serde_json::to_string(&execution.arguments)?)
        .bind(execution.environment.as_str())
        .bind(execution.risk.as_str())
        .bind(&execution.status)
        .bind(execution.exit_code)
        .bind(&execution.output_preview)
        .bind(&execution.artifact_id)
        .bind(i64::from(execution.truncated))
        .bind(serde_json::to_string(&execution.changed_paths)?)
        .bind(
            execution
                .sandbox
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?,
        )
        .bind(&execution.approval_id)
        .bind(&execution.error)
        .bind(execution.duration_ms.map(|value| value as i64))
        .execute(&self.pool)
        .await?;
        self.agent_tool_execution(&id).await
    }

    async fn agent_tool_execution(&self, id: &str) -> anyhow::Result<ToolExecutionRecord> {
        let row = sqlx::query(
            r#"SELECT id, session_id, run_id, tool_call_id, tool, arguments_json, environment,
                      risk, status, exit_code, output_preview, artifact_id, truncated,
                      changed_paths_json, sandbox_json, approval_id, error, duration_ms, created_at
               FROM agent_tool_executions WHERE id = ?"#,
        )
        .bind(id)
        .fetch_one(&self.pool)
        .await?;
        tool_execution_from_row(&row)
    }

    pub async fn list_tool_executions(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Vec<ToolExecutionRecord>> {
        let rows = sqlx::query(
            r#"SELECT id, session_id, run_id, tool_call_id, tool, arguments_json, environment,
                      risk, status, exit_code, output_preview, artifact_id, truncated,
                      changed_paths_json, sandbox_json, approval_id, error, duration_ms, created_at
               FROM agent_tool_executions WHERE session_id = ?
               ORDER BY created_at, rowid"#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(tool_execution_from_row).collect()
    }

    pub async fn create_approval(&self, approval: NewApproval) -> anyhow::Result<AgentApproval> {
        let id = Uuid::new_v4().to_string();
        sqlx::query(
            r#"INSERT INTO agent_approvals(
                   id, session_id, tool, arguments_json, arguments_hash, environment, risk,
                   scope_key, allow_session_scope, elevation_json, sandbox_json, summary, status)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')"#,
        )
        .bind(&id)
        .bind(&approval.session_id)
        .bind(&approval.tool)
        .bind(serde_json::to_string(&approval.arguments)?)
        .bind(&approval.arguments_hash)
        .bind(approval.environment.as_str())
        .bind(approval.risk.as_str())
        .bind(&approval.scope_key)
        .bind(i64::from(approval.allow_session_scope))
        .bind(serde_json::to_string(&approval.elevation)?)
        .bind(serde_json::to_string(&approval.sandbox)?)
        .bind(&approval.summary)
        .execute(&self.pool)
        .await?;
        self.approval(&id).await
    }

    pub async fn approval(&self, id: &str) -> anyhow::Result<AgentApproval> {
        let row = sqlx::query(
            r#"SELECT id, session_id, tool, arguments_json, arguments_hash, environment, risk,
                      scope_key, allow_session_scope, elevation_json, sandbox_json, summary,
                      status, scope, note, decided_at, created_at
               FROM agent_approvals WHERE id = ?"#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?
        .with_context(|| format!("approval {id} does not exist"))?;
        approval_from_row(&row)
    }

    /// Whether a session-scope grant may be created for this approval.
    pub async fn approval_allows_session_scope(&self, id: &str) -> anyhow::Result<bool> {
        let flag: i64 =
            sqlx::query_scalar("SELECT allow_session_scope FROM agent_approvals WHERE id = ?")
                .bind(id)
                .fetch_one(&self.pool)
                .await?;
        Ok(flag != 0)
    }

    pub async fn pending_approvals(&self, session_id: &str) -> anyhow::Result<Vec<AgentApproval>> {
        let rows = sqlx::query(
            r#"SELECT id, session_id, tool, arguments_json, arguments_hash, environment, risk,
                      scope_key, allow_session_scope, elevation_json, sandbox_json, summary,
                      status, scope, note, decided_at, created_at
               FROM agent_approvals WHERE session_id = ? AND status = 'pending'
               ORDER BY created_at"#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(approval_from_row).collect()
    }

    /// Record the user's decision. Returns the updated approval, or an error if
    /// it was already decided or has expired.
    pub async fn decide_approval(
        &self,
        id: &str,
        approved: bool,
        scope: Option<ApprovalScope>,
        note: Option<String>,
    ) -> anyhow::Result<AgentApproval> {
        let current = self.approval(id).await?;
        anyhow::ensure!(
            current.status == ApprovalStatus::Pending,
            "approval {id} was already {}",
            current.status.as_str()
        );
        let mut effective_scope = scope.unwrap_or(ApprovalScope::Once);
        if effective_scope == ApprovalScope::Session
            && !self.approval_allows_session_scope(id).await?
        {
            // Destructive and host actions never carry a standing grant, even
            // if the UI asks for one.
            effective_scope = ApprovalScope::Once;
        }
        let status = if approved {
            ApprovalStatus::Approved
        } else {
            ApprovalStatus::Denied
        };
        sqlx::query(
            r#"UPDATE agent_approvals
               SET status = ?, scope = ?, note = ?, decided_at = datetime('now')
               WHERE id = ? AND status = 'pending'"#,
        )
        .bind(status.as_str())
        .bind(effective_scope.as_str())
        .bind(&note)
        .bind(id)
        .execute(&self.pool)
        .await?;

        if approved && effective_scope == ApprovalScope::Session && !current.scope_key.is_empty() {
            let key = grant_key(current.environment, &current.scope_key);
            sqlx::query(
                r#"INSERT OR IGNORE INTO agent_grants(session_id, grant_key, approval_id)
                   VALUES(?, ?, ?)"#,
            )
            .bind(&current.session_id)
            .bind(&key)
            .bind(id)
            .execute(&self.pool)
            .await?;
        }
        self.approval(id).await
    }

    /// Spend a one-shot approval so it cannot authorize a second call.
    pub async fn consume_approval(&self, id: &str) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE agent_approvals SET status = 'consumed' WHERE id = ? AND status = 'approved'",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Expire stale pending approvals, and deny everything still pending for a
    /// session when the run is cancelled.
    pub async fn expire_pending_approvals(&self, session_id: Option<&str>) -> anyhow::Result<u64> {
        let result =
            match session_id {
                Some(session_id) => sqlx::query(
                    r#"UPDATE agent_approvals SET status = 'expired', decided_at = datetime('now')
                       WHERE session_id = ? AND status = 'pending'"#,
                )
                .bind(session_id)
                .execute(&self.pool)
                .await?,
                None => sqlx::query(
                    r#"UPDATE agent_approvals SET status = 'expired', decided_at = datetime('now')
                       WHERE status = 'pending'
                         AND created_at <= datetime('now', ?)"#,
                )
                .bind(format!("-{APPROVAL_TTL_SECONDS} seconds"))
                .execute(&self.pool)
                .await?,
            };
        Ok(result.rows_affected())
    }

    pub async fn session_grants(&self, session_id: &str) -> anyhow::Result<Vec<String>> {
        Ok(
            sqlx::query_scalar("SELECT grant_key FROM agent_grants WHERE session_id = ?")
                .bind(session_id)
                .fetch_all(&self.pool)
                .await?,
        )
    }

    pub async fn clear_session_grants(&self, session_id: &str) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM agent_grants WHERE session_id = ?")
            .bind(session_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn record_artifact(
        &self,
        session_id: &str,
        kind: &str,
        path: &str,
        size_bytes: u64,
        mime_type: Option<&str>,
    ) -> anyhow::Result<String> {
        let id = Uuid::new_v4().to_string();
        sqlx::query(
            r#"INSERT INTO agent_artifacts(id, session_id, kind, path, size_bytes, mime_type)
               VALUES(?, ?, ?, ?, ?, ?)"#,
        )
        .bind(&id)
        .bind(session_id)
        .bind(kind)
        .bind(path)
        .bind(size_bytes as i64)
        .bind(mime_type)
        .execute(&self.pool)
        .await?;
        Ok(id)
    }

    /// Filesystem path and size of a stored artifact.
    pub async fn artifact(&self, id: &str) -> anyhow::Result<(String, String, i64)> {
        let row =
            sqlx::query("SELECT session_id, path, size_bytes FROM agent_artifacts WHERE id = ?")
                .bind(id)
                .fetch_optional(&self.pool)
                .await?
                .with_context(|| format!("artifact {id} does not exist"))?;
        Ok((
            row.try_get("session_id")?,
            row.try_get("path")?,
            row.try_get("size_bytes")?,
        ))
    }
}

fn tool_execution_from_row(row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<ToolExecutionRecord> {
    let arguments_json: String = row.try_get("arguments_json")?;
    let changed_paths_json: Option<String> = row.try_get("changed_paths_json")?;
    let sandbox_json: Option<String> = row.try_get("sandbox_json")?;
    let truncated: i64 = row.try_get("truncated")?;
    Ok(ToolExecutionRecord {
        id: row.try_get("id")?,
        session_id: row.try_get("session_id")?,
        run_id: row.try_get("run_id")?,
        tool_call_id: row.try_get("tool_call_id")?,
        tool: row.try_get("tool")?,
        arguments: serde_json::from_str(&arguments_json).unwrap_or(Value::Null),
        environment: row.try_get("environment")?,
        risk: row.try_get("risk")?,
        status: row.try_get("status")?,
        exit_code: row.try_get("exit_code")?,
        output_preview: row.try_get("output_preview")?,
        artifact_id: row.try_get("artifact_id")?,
        truncated: truncated != 0,
        changed_paths: changed_paths_json
            .as_deref()
            .and_then(|raw| serde_json::from_str(raw).ok())
            .unwrap_or_default(),
        sandbox: sandbox_json
            .as_deref()
            .and_then(|raw| serde_json::from_str(raw).ok()),
        approval_id: row.try_get("approval_id")?,
        error: row.try_get("error")?,
        duration_ms: row.try_get("duration_ms")?,
        created_at: row.try_get("created_at")?,
    })
}

fn approval_from_row(row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<AgentApproval> {
    let arguments_json: String = row.try_get("arguments_json")?;
    let elevation_json: String = row.try_get("elevation_json")?;
    let sandbox_json: String = row.try_get("sandbox_json")?;
    let status: String = row.try_get("status")?;
    let environment: String = row.try_get("environment")?;
    let risk: String = row.try_get("risk")?;
    let scope: Option<String> = row.try_get("scope")?;
    let allow_session_scope: i64 = row.try_get("allow_session_scope")?;
    Ok(AgentApproval {
        id: row.try_get("id")?,
        session_id: row.try_get("session_id")?,
        tool: row.try_get("tool")?,
        arguments: serde_json::from_str(&arguments_json).unwrap_or(Value::Null),
        arguments_hash: row.try_get("arguments_hash")?,
        environment: parse_environment(&environment),
        risk: parse_risk(&risk),
        scope_key: row.try_get("scope_key")?,
        allow_session_scope: allow_session_scope != 0,
        elevation: serde_json::from_str(&elevation_json)?,
        summary: row.try_get("summary")?,
        sandbox: serde_json::from_str(&sandbox_json)?,
        status: ApprovalStatus::parse(&status),
        scope: scope.as_deref().and_then(|value| match value {
            "session" => Some(ApprovalScope::Session),
            "once" => Some(ApprovalScope::Once),
            _ => None,
        }),
        note: row.try_get("note")?,
        decided_at: row.try_get("decided_at")?,
        created_at: row.try_get("created_at")?,
    })
}

/// Default sandbox description used when a row predates one.
pub fn unknown_sandbox() -> SandboxDescription {
    SandboxDescription {
        backend: "none".to_owned(),
        profile: "workspace".to_owned(),
        isolated: false,
        network: false,
        workspace_path: None,
        detail: "Sandbox state was not recorded for this call.".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brazier_protocol::agent_types::{RequestedPathAccess, arguments_hash};
    use serde_json::json;
    use tempfile::tempdir;

    async fn database() -> (tempfile::TempDir, Database) {
        let dir = tempdir().expect("temp dir");
        let db = Database::open(&dir.path().join("brazier.sqlite"))
            .await
            .expect("open database");
        (dir, db)
    }

    fn session_request(workspace: Option<&str>) -> CreateAgentSession {
        CreateAgentSession {
            title: Some("Test task".to_owned()),
            workspace_path: workspace.map(str::to_owned),
            model: "gguf:test".to_owned(),
            runtime_id: "pi".to_owned(),
            permission_mode: None,
            permission_settings: None,
            enabled_tools: None,
            confine_to_worktree: false,
        }
    }

    fn elevation() -> AgentElevationRequest {
        AgentElevationRequest {
            reason: "Needs to build".to_owned(),
            proposed_command: Some("cargo test".to_owned()),
            requested_filesystem_paths: vec![RequestedPathAccess {
                path: "/etc/hosts".to_owned(),
                write: false,
            }],
            requested_network_access: false,
            requested_host_execution: false,
        }
    }

    fn new_approval(session_id: &str, allow_session_scope: bool) -> NewApproval {
        let arguments = json!({ "command": "cargo test" });
        NewApproval {
            session_id: session_id.to_owned(),
            tool: "shell_run".to_owned(),
            arguments_hash: arguments_hash("shell_run", &arguments),
            arguments,
            environment: AgentEnvironment::Sandbox,
            risk: ToolRiskLevel::Execute,
            scope_key: "run:cargo".to_owned(),
            allow_session_scope,
            elevation: elevation(),
            sandbox: unknown_sandbox(),
            summary: "Run `cargo test` in the sandbox".to_owned(),
        }
    }

    #[tokio::test]
    async fn sessions_round_trip_with_defaults() {
        let (_dir, db) = database().await;
        let session = db
            .create_agent_session(session_request(Some("/ws")))
            .await
            .expect("create session");
        assert_eq!(session.permission_mode, AgentPermissionMode::Ask);
        assert!(session.permission_settings.auto_approve_sandboxed_actions);
        assert!(!session.permission_settings.auto_approve_host_actions);
        assert_eq!(session.last_run_status, "idle");

        let listed = db.list_agent_sessions().await.expect("list");
        assert_eq!(listed.len(), 1);

        let updated = db
            .update_agent_session(
                &session.id,
                UpdateAgentSession {
                    permission_mode: Some(AgentPermissionMode::SandboxOnly),
                    last_run_status: Some("completed".to_owned()),
                    ..Default::default()
                },
            )
            .await
            .expect("update");
        assert_eq!(updated.permission_mode, AgentPermissionMode::SandboxOnly);
        assert_eq!(updated.last_run_status, "completed");
        // Untouched fields survive a partial update.
        assert_eq!(updated.model, "gguf:test");
        assert_eq!(updated.workspace_path.as_deref(), Some("/ws"));
    }

    #[tokio::test]
    async fn clearing_a_workspace_is_distinct_from_leaving_it_alone() {
        let (_dir, db) = database().await;
        let session = db
            .create_agent_session(session_request(Some("/ws")))
            .await
            .expect("create session");
        let untouched = db
            .update_agent_session(
                &session.id,
                UpdateAgentSession {
                    title: Some("Renamed".to_owned()),
                    ..Default::default()
                },
            )
            .await
            .expect("update");
        assert_eq!(untouched.workspace_path.as_deref(), Some("/ws"));

        let cleared = db
            .update_agent_session(
                &session.id,
                UpdateAgentSession {
                    workspace_path: Some(None),
                    ..Default::default()
                },
            )
            .await
            .expect("update");
        assert_eq!(cleared.workspace_path, None);
    }

    #[tokio::test]
    async fn messages_append_in_order_and_can_be_replaced_by_compaction() {
        let (_dir, db) = database().await;
        let session = db
            .create_agent_session(session_request(None))
            .await
            .expect("create session");
        db.append_agent_messages(
            &session.id,
            &[
                AppendAgentMessage {
                    role: "user".to_owned(),
                    payload: json!({ "text": "one" }),
                },
                AppendAgentMessage {
                    role: "assistant".to_owned(),
                    payload: json!({ "text": "two" }),
                },
            ],
            false,
        )
        .await
        .expect("append");
        let more = db
            .append_agent_messages(
                &session.id,
                &[AppendAgentMessage {
                    role: "user".to_owned(),
                    payload: json!({ "text": "three" }),
                }],
                false,
            )
            .await
            .expect("append");
        assert_eq!(more.len(), 3);
        assert_eq!(
            more.iter().map(|m| m.seq).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );

        let compacted = db
            .append_agent_messages(
                &session.id,
                &[AppendAgentMessage {
                    role: "user".to_owned(),
                    payload: json!({ "text": "summary" }),
                }],
                true,
            )
            .await
            .expect("replace");
        assert_eq!(compacted.len(), 1);
        assert_eq!(compacted[0].seq, 1);
    }

    #[tokio::test]
    async fn approving_with_session_scope_creates_a_grant() {
        let (_dir, db) = database().await;
        let session = db
            .create_agent_session(session_request(Some("/ws")))
            .await
            .expect("create session");
        let approval = db
            .create_approval(new_approval(&session.id, true))
            .await
            .expect("create approval");
        assert_eq!(approval.status, ApprovalStatus::Pending);

        let decided = db
            .decide_approval(&approval.id, true, Some(ApprovalScope::Session), None)
            .await
            .expect("decide");
        assert_eq!(decided.status, ApprovalStatus::Approved);
        assert_eq!(decided.scope, Some(ApprovalScope::Session));
        let grants = db.session_grants(&session.id).await.expect("grants");
        assert_eq!(grants, vec!["sandbox:run:cargo".to_owned()]);
    }

    #[tokio::test]
    async fn session_scope_is_downgraded_when_the_policy_forbids_it() {
        let (_dir, db) = database().await;
        let session = db
            .create_agent_session(session_request(Some("/ws")))
            .await
            .expect("create session");
        let approval = db
            .create_approval(new_approval(&session.id, false))
            .await
            .expect("create approval");
        let decided = db
            .decide_approval(&approval.id, true, Some(ApprovalScope::Session), None)
            .await
            .expect("decide");
        assert_eq!(decided.scope, Some(ApprovalScope::Once));
        assert!(
            db.session_grants(&session.id)
                .await
                .expect("grants")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn an_approval_cannot_be_decided_twice_or_reused() {
        let (_dir, db) = database().await;
        let session = db
            .create_agent_session(session_request(Some("/ws")))
            .await
            .expect("create session");
        let approval = db
            .create_approval(new_approval(&session.id, true))
            .await
            .expect("create approval");
        db.decide_approval(&approval.id, true, Some(ApprovalScope::Once), None)
            .await
            .expect("first decision");
        assert!(
            db.decide_approval(&approval.id, true, Some(ApprovalScope::Once), None)
                .await
                .is_err(),
            "a decided approval must not be re-decided"
        );
        db.consume_approval(&approval.id).await.expect("consume");
        let spent = db.approval(&approval.id).await.expect("read");
        assert_eq!(spent.status, ApprovalStatus::Consumed);
    }

    #[tokio::test]
    async fn cancelling_a_session_expires_pending_approvals() {
        let (_dir, db) = database().await;
        let session = db
            .create_agent_session(session_request(Some("/ws")))
            .await
            .expect("create session");
        db.create_approval(new_approval(&session.id, true))
            .await
            .expect("create approval");
        let expired = db
            .expire_pending_approvals(Some(&session.id))
            .await
            .expect("expire");
        assert_eq!(expired, 1);
        assert!(
            db.pending_approvals(&session.id)
                .await
                .expect("pending")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn tool_executions_are_recorded_for_the_timeline() {
        let (_dir, db) = database().await;
        let session = db
            .create_agent_session(session_request(Some("/ws")))
            .await
            .expect("create session");
        db.record_tool_execution(NewToolExecution {
            session_id: session.id.clone(),
            run_id: Some("run-1".to_owned()),
            tool_call_id: Some("call-1".to_owned()),
            tool: "shell_run".to_owned(),
            arguments: json!({ "command": "ls" }),
            environment: AgentEnvironment::Sandbox,
            risk: ToolRiskLevel::Execute,
            status: "completed".to_owned(),
            exit_code: Some(0),
            output_preview: Some("Cargo.toml".to_owned()),
            artifact_id: None,
            truncated: true,
            changed_paths: vec!["src/main.rs".to_owned()],
            sandbox: Some(unknown_sandbox()),
            approval_id: None,
            error: None,
            duration_ms: Some(42),
        })
        .await
        .expect("record");
        let records = db.list_tool_executions(&session.id).await.expect("list");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].tool, "shell_run");
        assert!(records[0].truncated);
        assert_eq!(records[0].changed_paths, vec!["src/main.rs".to_owned()]);
        assert_eq!(records[0].duration_ms, Some(42));
    }

    #[tokio::test]
    async fn deleting_a_session_removes_its_children() {
        let (_dir, db) = database().await;
        let session = db
            .create_agent_session(session_request(Some("/ws")))
            .await
            .expect("create session");
        db.append_agent_messages(
            &session.id,
            &[AppendAgentMessage {
                role: "user".to_owned(),
                payload: json!({ "text": "hi" }),
            }],
            false,
        )
        .await
        .expect("append");
        db.delete_agent_session(&session.id).await.expect("delete");
        assert!(db.agent_session(&session.id).await.is_err());
        assert!(
            db.agent_messages(&session.id)
                .await
                .expect("messages")
                .is_empty()
        );
    }
}

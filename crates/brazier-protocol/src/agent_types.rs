//! Shared wire types for Agent mode.
//!
//! These are the wire types shared by the daemon, the agent worker, and the
//! desktop UI. They are deliberately independent of any agent framework: a
//! runtime adapter translates its own representation into these, never the
//! other way around.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::execution_location::ExecutionLocation;

/// Where a tool call runs. The sandbox is the default; `host` requires an
/// approved elevation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentEnvironment {
    Sandbox,
    Host,
}

impl AgentEnvironment {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sandbox => "sandbox",
            Self::Host => "host",
        }
    }
}

/// Stable key used to look up a session-scoped permission grant.
pub fn grant_key(environment: AgentEnvironment, scope_key: &str) -> String {
    format!("{}:{scope_key}", environment.as_str())
}

impl std::fmt::Display for AgentEnvironment {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// How much damage a tool call can do. Risk drives the approval decision, not
/// the tool's own opinion of itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolRiskLevel {
    /// Metadata only: never touches user content.
    Safe,
    /// Reads workspace content.
    Read,
    /// Creates or modifies workspace content.
    Write,
    /// Runs a program.
    Execute,
    /// Removes or overwrites content, or leaves the workspace.
    Destructive,
}

impl ToolRiskLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Read => "read",
            Self::Write => "write",
            Self::Execute => "execute",
            Self::Destructive => "destructive",
        }
    }
}

/// Permission mode for a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AgentPermissionMode {
    /// Ask before anything that writes, executes, or leaves the sandbox.
    Ask,
    /// Sandboxed work proceeds without prompts; host execution is refused.
    SandboxOnly,
    /// Explicit user opt-out of prompting, bounded by the settings below.
    SkipPermissions,
}

impl AgentPermissionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::SandboxOnly => "sandbox-only",
            Self::SkipPermissions => "skip-permissions",
        }
    }
}

/// Bounds for `skip-permissions`. Host auto-approval is a separate, explicit
/// choice from sandbox auto-approval.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AgentPermissionSettings {
    #[serde(default = "default_true")]
    pub auto_approve_sandboxed_actions: bool,
    #[serde(default)]
    pub auto_approve_host_actions: bool,
}

fn default_true() -> bool {
    true
}

impl Default for AgentPermissionSettings {
    fn default() -> Self {
        Self {
            auto_approve_sandboxed_actions: true,
            auto_approve_host_actions: false,
        }
    }
}

/// Filesystem access an agent asks for when it wants to leave the workspace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestedPathAccess {
    pub path: String,
    #[serde(default)]
    pub write: bool,
}

/// A request for privileges the session does not currently hold. This is what
/// the approval dialog renders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentElevationRequest {
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposed_command: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requested_filesystem_paths: Vec<RequestedPathAccess>,
    #[serde(default)]
    pub requested_network_access: bool,
    #[serde(default)]
    pub requested_host_execution: bool,
}

/// How long a granted approval lasts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApprovalScope {
    /// One tool call with these exact arguments.
    Once,
    /// Every later call of this tool with the same scope key in this session.
    Session,
}

impl ApprovalScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Once => "once",
            Self::Session => "session",
        }
    }
}

/// Lifecycle of an approval record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Denied,
    Expired,
    /// Approved and already spent by a `once` grant.
    Consumed,
}

impl ApprovalStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Approved => "approved",
            Self::Denied => "denied",
            Self::Expired => "expired",
            Self::Consumed => "consumed",
        }
    }

    pub fn parse(value: &str) -> Self {
        match value {
            "approved" => Self::Approved,
            "denied" => Self::Denied,
            "expired" => Self::Expired,
            "consumed" => Self::Consumed,
            _ => Self::Pending,
        }
    }
}

/// A stored approval request plus its decision, if any.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentApproval {
    pub id: String,
    pub session_id: String,
    pub tool: String,
    pub arguments: serde_json::Value,
    pub arguments_hash: String,
    pub environment: AgentEnvironment,
    pub risk: ToolRiskLevel,
    pub scope_key: String,
    /// Whether the user may grant this for the whole session. False for
    /// destructive and host actions, which are always one-shot.
    pub allow_session_scope: bool,
    pub elevation: AgentElevationRequest,
    /// What the daemon will actually do if this is approved.
    pub summary: String,
    pub sandbox: SandboxDescription,
    /// Exact daemon host on which approving this request will run the action.
    pub execution_location: ExecutionLocation,
    pub status: ApprovalStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<ApprovalScope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decided_at: Option<String>,
    /// Paired-client id, or the `owner` sentinel for a bootstrap owner key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decided_by_client_id: Option<String>,
    pub created_at: String,
}

/// The user's answer to an approval request.
#[derive(Debug, Clone, Deserialize)]
pub struct ApprovalDecisionRequest {
    /// `approve` or `deny`.
    pub decision: String,
    #[serde(default)]
    pub scope: Option<ApprovalScope>,
    #[serde(default)]
    pub note: Option<String>,
    /// Optional optimistic trust-boundary check. When present, the daemon
    /// rejects a decision unless it names the exact execution host snapshot
    /// shown to the user for this approval.
    #[serde(default)]
    pub expected_execution_location: Option<ExecutionLocation>,
}

/// Honest description of the isolation a tool call actually got. The UI must
/// never claim more than this reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxDescription {
    /// `seatbelt`, `bubblewrap`, or `none`.
    pub backend: String,
    pub profile: String,
    /// False when no OS-level isolation was applied.
    pub isolated: bool,
    pub network: bool,
    /// Path the workspace is reachable at inside the sandbox.
    pub workspace_path: Option<String>,
    /// Human-readable caveat shown next to the sandbox badge.
    pub detail: String,
}

/// A tool execution request from the runtime adapter.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolExecRequest {
    pub session_id: String,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    pub tool: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
    /// Requested environment. Defaults to the sandbox.
    #[serde(default)]
    pub environment: Option<AgentEnvironment>,
    /// Why the agent wants this, shown in the approval dialog.
    #[serde(default)]
    pub reason: Option<String>,
    /// Approval granted for this exact call, from a previous round trip.
    #[serde(default)]
    pub approval_id: Option<String>,
}

/// Outcome of a broker call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecStatus {
    Completed,
    Failed,
    Denied,
    ApprovalRequired,
}

/// An image a tool produced for a vision-capable agent model.
#[derive(Debug, Clone, Serialize)]
pub struct ToolImage {
    pub mime_type: String,
    /// Raw base64 (no data-URL prefix).
    pub data: String,
}

/// Result of a broker call, including approval state when execution is held.
#[derive(Debug, Clone, Serialize)]
pub struct ToolExecResponse {
    pub status: ToolExecStatus,
    pub tool: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    pub environment: AgentEnvironment,
    pub risk: ToolRiskLevel,
    pub sandbox: SandboxDescription,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_id: Option<String>,
    pub output: String,
    #[serde(default)]
    pub truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub changed_paths: Vec<String>,
    /// Page renders and similar images for the model (e.g. `doc_read`).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub images: Vec<ToolImage>,
    pub duration_ms: u64,
    /// Present when `status` is `approval_required`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approval: Option<AgentApproval>,
    /// Set when the call was refused outright.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denied_reason: Option<String>,
    pub is_error: bool,
}

/// Session record as the UI and the runtime adapter see it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSessionRecord {
    pub id: String,
    /// Authorization principal that created this session. This is deliberately
    /// not returned over the API; callers only need the server-side ownership
    /// check, not another client's durable identifier.
    #[serde(default = "default_agent_session_owner", skip_serializing)]
    pub owner_client_id: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_path: Option<String>,
    pub model: String,
    pub runtime_id: String,
    pub permission_mode: AgentPermissionMode,
    pub permission_settings: AgentPermissionSettings,
    #[serde(default)]
    pub enabled_tools: Option<Vec<String>>,
    pub last_run_status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_metadata: Option<serde_json::Value>,
    pub created_at: String,
    pub updated_at: String,
}

fn default_agent_session_owner() -> String {
    "owner".to_owned()
}

#[derive(Debug, Clone, Deserialize)]
pub struct CreateAgentSession {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub workspace_path: Option<String>,
    pub model: String,
    /// Agent framework adapter id (`simple` or `powerful`). When omitted, the
    /// daemon resolves the saved default preference, then falls back to `simple`.
    #[serde(default)]
    pub runtime_id: Option<String>,
    #[serde(default)]
    pub permission_mode: Option<AgentPermissionMode>,
    #[serde(default)]
    pub permission_settings: Option<AgentPermissionSettings>,
    #[serde(default)]
    pub enabled_tools: Option<Vec<String>>,
    /// When true and `workspace_path` is a git repo, the session workspace is a
    /// fresh worktree so the agent cannot dirty the user's current checkout.
    #[serde(default)]
    pub confine_to_worktree: bool,
    /// Required when creating with skip-permissions or host auto-approval.
    /// The desktop UI sets this after an explicit mode choice; remote clients
    /// must confirm elevation the same way.
    #[serde(default)]
    pub confirm_elevated_permissions: bool,
}

/// Stock agent modes the daemon advertises.
pub const AGENT_RUNTIME_SIMPLE: &str = "simple";
pub const AGENT_RUNTIME_POWERFUL: &str = "powerful";
pub const DEFAULT_AGENT_RUNTIME_ID: &str = AGENT_RUNTIME_SIMPLE;

/// Partial session update. Absent fields keep their stored value.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct UpdateAgentSession {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(
        default,
        deserialize_with = "crate::serde_util::deserialize_double_option"
    )]
    pub workspace_path: Option<Option<String>>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub permission_mode: Option<AgentPermissionMode>,
    #[serde(default)]
    pub permission_settings: Option<AgentPermissionSettings>,
    #[serde(default)]
    pub enabled_tools: Option<Vec<String>>,
    #[serde(default)]
    pub last_run_status: Option<String>,
    #[serde(default)]
    pub compaction: Option<serde_json::Value>,
    #[serde(default)]
    pub runtime_metadata: Option<serde_json::Value>,
    /// Set to enable or disable worktree confinement for this session.
    #[serde(default)]
    pub confine_to_worktree: Option<bool>,
    /// When turning confinement off, discard uncommitted worktree changes that
    /// have not been applied to the source checkout. Requires an explicit UI
    /// confirmation; ignored when enabling confinement.
    #[serde(default)]
    pub discard_unapplied: Option<bool>,
    /// Required when elevating to skip-permissions or host auto-approval.
    #[serde(default)]
    pub confirm_elevated_permissions: Option<bool>,
}

/// One persisted transcript entry. The payload is the runtime-neutral message
/// the UI renders and the adapter restores from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessageRecord {
    pub id: String,
    pub session_id: String,
    pub seq: i64,
    pub role: String,
    pub payload: serde_json::Value,
    pub created_at: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AppendAgentMessages {
    pub messages: Vec<AppendAgentMessage>,
    /// Replace the whole transcript instead of appending (used by compaction).
    #[serde(default)]
    pub replace: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AppendAgentMessage {
    pub role: String,
    pub payload: serde_json::Value,
}

/// A recorded tool execution, for the activity timeline and run summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionRecord {
    pub id: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    pub tool: String,
    pub arguments: serde_json::Value,
    pub environment: String,
    pub risk: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_preview: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<String>,
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub changed_paths: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approval_id: Option<String>,
    /// Immutable daemon host recorded by the approval consumed for this call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_location: Option<ExecutionLocation>,
    /// Paired client (or owner sentinel) that approved the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decided_by_client_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<i64>,
    pub created_at: String,
}

/// Stable hash of tool arguments, so an approval cannot be replayed against a
/// different call.
pub fn arguments_hash(tool: &str, arguments: &serde_json::Value) -> String {
    let canonical = canonical_json(arguments);
    let mut hasher = Sha256::new();
    hasher.update(tool.as_bytes());
    hasher.update([0u8]);
    hasher.update(canonical.as_bytes());
    hex::encode(hasher.finalize())
}

/// Serialize with object keys sorted, so key order cannot change the hash.
fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let body = keys
                .into_iter()
                .map(|key| format!("{:?}:{}", key, canonical_json(&map[key])))
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
        serde_json::Value::Array(items) => {
            let body = items
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{body}]")
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn argument_hashes_ignore_key_order_but_not_values() {
        let a = json!({ "command": "ls", "cwd": "." });
        let b = json!({ "cwd": ".", "command": "ls" });
        let c = json!({ "command": "rm -rf .", "cwd": "." });
        assert_eq!(
            arguments_hash("shell_run", &a),
            arguments_hash("shell_run", &b)
        );
        assert_ne!(
            arguments_hash("shell_run", &a),
            arguments_hash("shell_run", &c)
        );
        // The tool name is part of the hash: an approval for one tool cannot be
        // spent on another.
        assert_ne!(
            arguments_hash("shell_run", &a),
            arguments_hash("fs_read", &a)
        );
    }

    #[test]
    fn nested_key_order_is_also_canonical() {
        let a = json!({ "outer": { "x": 1, "y": [2, { "a": 3, "b": 4 }] } });
        let b = json!({ "outer": { "y": [2, { "b": 4, "a": 3 }], "x": 1 } });
        assert_eq!(arguments_hash("t", &a), arguments_hash("t", &b));
    }

    #[test]
    fn skip_permissions_does_not_imply_host_auto_approval() {
        let settings = AgentPermissionSettings::default();
        assert!(settings.auto_approve_sandboxed_actions);
        assert!(!settings.auto_approve_host_actions);
    }

    #[test]
    fn session_update_distinguishes_a_cleared_workspace_from_an_absent_field() {
        let absent: UpdateAgentSession = serde_json::from_value(json!({})).unwrap();
        let cleared: UpdateAgentSession =
            serde_json::from_value(json!({ "workspace_path": null })).unwrap();
        let changed: UpdateAgentSession =
            serde_json::from_value(json!({ "workspace_path": "/work" })).unwrap();

        assert_eq!(absent.workspace_path, None);
        assert_eq!(cleared.workspace_path, Some(None));
        assert_eq!(changed.workspace_path, Some(Some("/work".to_owned())));
    }
}

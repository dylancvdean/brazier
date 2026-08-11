//! Execution broker for agent tools.
//!
//! Every agent tool call in the application funnels through [`execute`]. The
//! broker validates arguments, asks the policy layer what is allowed, runs the
//! work in the sandbox (or, once approved, on the host), bounds the output, and
//! records what happened. The agent runtime never touches the filesystem, a
//! shell, or any host API by itself.

use std::{
    collections::{BTreeMap, HashMap},
    fs::OpenOptions,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Context;
use serde_json::{Value, json};
use tokio::{
    io::{
        AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt,
        BufReader,
    },
    process::{Child, Command},
    sync::{Mutex, Notify, mpsc},
};
use uuid::Uuid;

use crate::{
    agent_policy::{self, PolicyDecision, PolicyRequest, is_inside, tool_spec},
    agent_sandbox::{
        SandboxBackend, SandboxBackendCapabilities, SandboxProfile, SandboxRequest, secret_paths,
    },
};
use brazier_protocol::agent_types::{
    AgentEnvironment, AgentSessionRecord, ApprovalScope, ApprovalStatus, SandboxDescription,
    ToolExecRequest, ToolExecResponse, ToolExecStatus, ToolImage, ToolRiskLevel, arguments_hash,
};
use brazier_storage::agent_store::{NewApproval, NewToolExecution};

/// Characters of tool output handed back to the model. Anything longer is
/// truncated in the middle and preserved in full as an artifact.
const MAX_MODEL_OUTPUT_CHARS: usize = 24_000;
/// Bytes of a file `fs_read` will return without an explicit range.
const MAX_READ_BYTES: usize = 512 * 1024;
/// Bytes of output retained per background process.
const MAX_PROCESS_BUFFER_BYTES: usize = 1024 * 1024;
const DEFAULT_SHELL_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_SHELL_TIMEOUT: Duration = Duration::from_secs(600);
const MAX_SEARCH_RESULTS: usize = 200;
/// Directories skipped by `fs_search` and deep `fs_list` walks.
const SKIPPED_DIRECTORIES: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "dist",
    "out",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".turbo",
    ".cargo-home",
    ".uv-cache",
];

/// Shared broker state: the detected sandbox, background processes, and the
/// wake-up signal that unblocks approval waiters.
pub struct AgentBroker {
    backend: SandboxBackend,
    processes: Mutex<HashMap<String, BackgroundProcess>>,
    /// Signalled whenever any approval is decided, so long-polls return fast.
    approvals_changed: Arc<Notify>,
}

struct BackgroundProcess {
    session_id: String,
    command: String,
    child: Arc<Mutex<Child>>,
    stdin: Arc<Mutex<Option<tokio::process::ChildStdin>>>,
    output: Arc<Mutex<String>>,
    started: Instant,
}

impl AgentBroker {
    pub fn new() -> Self {
        Self {
            backend: SandboxBackend::detect(),
            processes: Mutex::new(HashMap::new()),
            approvals_changed: Arc::new(Notify::new()),
        }
    }

    pub fn backend(&self) -> &SandboxBackend {
        &self.backend
    }

    pub fn capabilities(&self) -> SandboxBackendCapabilities {
        self.backend.capabilities()
    }

    /// Wake every approval waiter; called after a decision is recorded.
    pub fn notify_approvals(&self) {
        self.approvals_changed.notify_waiters();
    }

    pub fn approvals_notifier(&self) -> Arc<Notify> {
        Arc::clone(&self.approvals_changed)
    }

    /// Terminate every background process for a session.
    pub async fn terminate_session_processes(&self, session_id: &str) -> usize {
        let mut processes = self.processes.lock().await;
        let ids: Vec<String> = processes
            .iter()
            .filter(|(_, process)| process.session_id == session_id)
            .map(|(id, _)| id.clone())
            .collect();
        let mut terminated = 0;
        for id in ids {
            if let Some(process) = processes.remove(&id) {
                let mut child = process.child.lock().await;
                let _ = child.start_kill();
                terminated += 1;
            }
        }
        terminated
    }
}

impl Default for AgentBroker {
    fn default() -> Self {
        Self::new()
    }
}

/// Everything the broker needs that lives outside itself.
pub struct BrokerContext<'a> {
    pub broker: &'a AgentBroker,
    pub db: &'a brazier_storage::db::Database,
    pub data_dir: &'a Path,
    pub session: &'a AgentSessionRecord,
}

/// Resolved, validated environment for one call.
struct CallPlan {
    environment: AgentEnvironment,
    profile: SandboxProfile,
    sandbox: SandboxDescription,
    approval_id: Option<String>,
}

/// Run one tool call end to end: policy, execution, bookkeeping.
pub async fn execute(
    context: &BrokerContext<'_>,
    request: &ToolExecRequest,
) -> anyhow::Result<ToolExecResponse> {
    execute_inner(context, request, None).await
}

/// Execute a tool while forwarding incremental output when the tool supports it.
///
/// The final response remains authoritative and is still bounded, persisted,
/// and recorded exactly like a non-streamed execution. At present `shell_run`
/// is the only foreground tool that emits chunks.
pub async fn execute_streaming(
    context: &BrokerContext<'_>,
    request: &ToolExecRequest,
    output: mpsc::UnboundedSender<String>,
) -> anyhow::Result<ToolExecResponse> {
    execute_inner(context, request, Some(output)).await
}

async fn execute_inner(
    context: &BrokerContext<'_>,
    request: &ToolExecRequest,
    output: Option<mpsc::UnboundedSender<String>>,
) -> anyhow::Result<ToolExecResponse> {
    let started = Instant::now();
    let Some(spec) = tool_spec(&request.tool) else {
        return Ok(denied(
            request,
            context.broker.backend().describe_host(None),
            ToolRiskLevel::Safe,
            format!("Unknown tool `{}`.", request.tool),
            started,
        ));
    };

    let workspace = workspace_root(context.session)?;
    let requested_environment = request.environment.unwrap_or(AgentEnvironment::Sandbox);
    let grants = context.db.session_grants(&context.session.id).await?;

    // A previously granted approval short-circuits the prompt, but only for the
    // exact call it was issued for.
    let plan = if let Some(approval_id) = &request.approval_id {
        match validate_approval(context, request, approval_id).await? {
            Ok(plan) => plan,
            Err(reason) => {
                let sandbox = context.broker.backend().describe_host(workspace.as_deref());
                let response = denied(request, sandbox, spec.risk, reason, started);
                record(context, request, &response, spec.risk).await?;
                return Ok(response);
            }
        }
    } else {
        let decision = agent_policy::decide(&PolicyRequest {
            tool: &request.tool,
            arguments: &request.arguments,
            requested_environment,
            permission_mode: context.session.permission_mode,
            permission_settings: context.session.permission_settings,
            workspace: workspace.as_deref(),
            backend: context.broker.backend(),
            session_grants: &grants,
            reason: request.reason.as_deref(),
            data_dir: context.data_dir,
        });

        match decision {
            PolicyDecision::Deny { reason } => {
                let sandbox = context
                    .broker
                    .backend()
                    .describe(SandboxProfile::Workspace, workspace.as_deref());
                let response = denied(request, sandbox, spec.risk, reason, started);
                record(context, request, &response, spec.risk).await?;
                return Ok(response);
            }
            PolicyDecision::RequireApproval {
                environment,
                profile,
                elevation,
                scope_key,
                allow_session_scope,
                summary,
            } => {
                let sandbox = if environment == AgentEnvironment::Host {
                    context.broker.backend().describe_host(workspace.as_deref())
                } else {
                    context
                        .broker
                        .backend()
                        .describe(profile, workspace.as_deref())
                };
                let approval = context
                    .db
                    .create_approval(NewApproval {
                        session_id: context.session.id.clone(),
                        tool: request.tool.clone(),
                        arguments_hash: arguments_hash(&request.tool, &request.arguments),
                        arguments: request.arguments.clone(),
                        environment,
                        risk: spec.risk,
                        scope_key,
                        allow_session_scope,
                        elevation,
                        sandbox: sandbox.clone(),
                        summary,
                    })
                    .await?;
                return Ok(ToolExecResponse {
                    status: ToolExecStatus::ApprovalRequired,
                    tool: request.tool.clone(),
                    tool_call_id: request.tool_call_id.clone(),
                    environment,
                    risk: spec.risk,
                    sandbox,
                    execution_id: None,
                    output: String::new(),
                    truncated: false,
                    artifact_id: None,
                    exit_code: None,
                    changed_paths: Vec::new(),
                    images: Vec::new(),
                    duration_ms: started.elapsed().as_millis() as u64,
                    approval: Some(approval),
                    denied_reason: None,
                    is_error: false,
                });
            }
            PolicyDecision::Allow {
                environment,
                profile,
                ..
            } => {
                let sandbox = if environment == AgentEnvironment::Host {
                    context.broker.backend().describe_host(workspace.as_deref())
                } else {
                    context
                        .broker
                        .backend()
                        .describe(profile, workspace.as_deref())
                };
                CallPlan {
                    environment,
                    profile,
                    sandbox,
                    approval_id: None,
                }
            }
        }
    };

    // Claim a one-shot approval before starting the call. Validation and
    // execution are separate async operations, so consuming it afterwards
    // would let two concurrent requests both run on the same approval.
    if let Some(approval_id) = &plan.approval_id
        && context.db.consume_approval(approval_id).await.is_err()
    {
        let response = denied(
            request,
            plan.sandbox.clone(),
            spec.risk,
            "That approval was already used once.".to_owned(),
            started,
        );
        record(context, request, &response, spec.risk).await?;
        return Ok(response);
    }

    let outcome = run_tool(
        context,
        request,
        &plan,
        workspace.as_deref(),
        output.as_ref(),
    )
    .await;
    let mut response = match outcome {
        Ok(outcome) => {
            let (output, truncated, artifact_id) = bound_output(context, &outcome.output).await?;
            ToolExecResponse {
                status: if outcome.is_error {
                    ToolExecStatus::Failed
                } else {
                    ToolExecStatus::Completed
                },
                tool: request.tool.clone(),
                tool_call_id: request.tool_call_id.clone(),
                environment: plan.environment,
                risk: spec.risk,
                sandbox: plan.sandbox.clone(),
                execution_id: None,
                output,
                truncated,
                artifact_id,
                exit_code: outcome.exit_code,
                changed_paths: outcome.changed_paths,
                images: outcome.images,
                duration_ms: started.elapsed().as_millis() as u64,
                approval: None,
                denied_reason: None,
                is_error: outcome.is_error,
            }
        }
        Err(error) => ToolExecResponse {
            status: ToolExecStatus::Failed,
            tool: request.tool.clone(),
            tool_call_id: request.tool_call_id.clone(),
            environment: plan.environment,
            risk: spec.risk,
            sandbox: plan.sandbox.clone(),
            execution_id: None,
            output: format!("Error: {error}"),
            truncated: false,
            artifact_id: None,
            exit_code: None,
            changed_paths: Vec::new(),
            images: Vec::new(),
            duration_ms: started.elapsed().as_millis() as u64,
            approval: None,
            denied_reason: None,
            is_error: true,
        },
    };

    let record = record(context, request, &response, spec.risk).await?;
    response.execution_id = Some(record);
    Ok(response)
}

/// Check that a supplied approval authorizes exactly this call.
async fn validate_approval(
    context: &BrokerContext<'_>,
    request: &ToolExecRequest,
    approval_id: &str,
) -> anyhow::Result<Result<CallPlan, String>> {
    let approval = match context.db.approval(approval_id).await {
        Ok(approval) => approval,
        Err(_) => return Ok(Err("That approval no longer exists.".to_owned())),
    };
    if approval.session_id != context.session.id {
        return Ok(Err("That approval belongs to another session.".to_owned()));
    }
    match approval.status {
        ApprovalStatus::Approved => {}
        ApprovalStatus::Pending => {
            return Ok(Err("That approval has not been answered yet.".to_owned()));
        }
        ApprovalStatus::Denied => {
            return Ok(Err(format!(
                "The user denied this action.{}",
                approval
                    .note
                    .as_deref()
                    .map(|note| format!(" Note: {note}"))
                    .unwrap_or_default()
            )));
        }
        ApprovalStatus::Expired => {
            return Ok(Err("That approval expired before it was used.".to_owned()));
        }
        ApprovalStatus::Consumed => {
            return Ok(Err("That approval was already used once.".to_owned()));
        }
    }
    if approval.tool != request.tool
        || approval.arguments_hash != arguments_hash(&request.tool, &request.arguments)
    {
        return Ok(Err(
            "That approval was granted for a different call. Ask again.".to_owned(),
        ));
    }

    let workspace = workspace_root(context.session)?;
    let profile =
        SandboxProfile::parse(&approval.sandbox.profile).unwrap_or(SandboxProfile::Workspace);
    let sandbox = if approval.environment == AgentEnvironment::Host {
        context.broker.backend().describe_host(workspace.as_deref())
    } else {
        context
            .broker
            .backend()
            .describe(profile, workspace.as_deref())
    };
    // A session-scoped grant stays valid; a one-shot approval is consumed after
    // the call runs.
    let approval_id = (approval.scope != Some(ApprovalScope::Session)).then(|| approval.id.clone());
    Ok(Ok(CallPlan {
        environment: approval.environment,
        profile,
        sandbox,
        approval_id,
    }))
}

fn denied(
    request: &ToolExecRequest,
    sandbox: SandboxDescription,
    risk: ToolRiskLevel,
    reason: String,
    started: Instant,
) -> ToolExecResponse {
    ToolExecResponse {
        status: ToolExecStatus::Denied,
        tool: request.tool.clone(),
        tool_call_id: request.tool_call_id.clone(),
        environment: request.environment.unwrap_or(AgentEnvironment::Sandbox),
        risk,
        sandbox,
        execution_id: None,
        output: format!("Refused: {reason}"),
        truncated: false,
        artifact_id: None,
        exit_code: None,
        changed_paths: Vec::new(),
        images: Vec::new(),
        duration_ms: started.elapsed().as_millis() as u64,
        approval: None,
        denied_reason: Some(reason),
        is_error: true,
    }
}

async fn record(
    context: &BrokerContext<'_>,
    request: &ToolExecRequest,
    response: &ToolExecResponse,
    risk: ToolRiskLevel,
) -> anyhow::Result<String> {
    let status = match response.status {
        ToolExecStatus::Completed => "completed",
        ToolExecStatus::Failed => "failed",
        ToolExecStatus::Denied => "denied",
        ToolExecStatus::ApprovalRequired => "awaiting-approval",
    };
    let record = context
        .db
        .record_tool_execution(NewToolExecution {
            session_id: context.session.id.clone(),
            run_id: request.run_id.clone(),
            tool_call_id: request.tool_call_id.clone(),
            tool: request.tool.clone(),
            arguments: request.arguments.clone(),
            environment: response.environment,
            risk,
            status: status.to_owned(),
            exit_code: response.exit_code,
            output_preview: Some(preview(&response.output)),
            artifact_id: response.artifact_id.clone(),
            truncated: response.truncated,
            changed_paths: response.changed_paths.clone(),
            sandbox: Some(response.sandbox.clone()),
            approval_id: request.approval_id.clone(),
            error: response.denied_reason.clone(),
            duration_ms: Some(response.duration_ms),
        })
        .await?;
    Ok(record.id)
}

fn preview(output: &str) -> String {
    output.chars().take(600).collect()
}

/// Keep the model's view of output bounded, and store the full text as an
/// artifact so nothing is silently lost.
async fn bound_output(
    context: &BrokerContext<'_>,
    output: &str,
) -> anyhow::Result<(String, bool, Option<String>)> {
    let char_count = output.chars().count();
    if char_count <= MAX_MODEL_OUTPUT_CHARS {
        return Ok((output.to_owned(), false, None));
    }
    let artifacts = context.data_dir.join("agent").join("artifacts");
    tokio::fs::create_dir_all(&artifacts).await?;
    let file_name = format!("{}.txt", Uuid::new_v4());
    let path = artifacts.join(&file_name);
    tokio::fs::write(&path, output.as_bytes()).await?;
    let artifact_id = context
        .db
        .record_artifact(
            &context.session.id,
            "tool-output",
            &path.display().to_string(),
            output.len() as u64,
            Some("text/plain"),
        )
        .await?;

    let half = MAX_MODEL_OUTPUT_CHARS / 2;
    let tail_start = char_count.saturating_sub(half);
    let mut head = String::new();
    let mut tail = String::new();
    for (index, character) in output.chars().enumerate() {
        if index < half {
            head.push(character);
        } else if index >= tail_start {
            tail.push(character);
        }
    }
    let dropped = char_count - half - (char_count - tail_start);
    let truncated = format!(
        "{head}\n\n[... {dropped} characters truncated. Full output stored as artifact \
         {artifact_id} ...]\n\n{tail}"
    );
    Ok((truncated, true, Some(artifact_id)))
}

fn workspace_root(session: &AgentSessionRecord) -> anyhow::Result<Option<PathBuf>> {
    let Some(raw) = session.workspace_path.as_deref() else {
        return Ok(None);
    };
    let path = PathBuf::from(raw);
    // Canonicalize once here so every later comparison sees the real path and a
    // symlinked workspace root cannot widen its own scope.
    let resolved = std::fs::canonicalize(&path).unwrap_or_else(|_| agent_policy::normalize(&path));
    Ok(Some(resolved))
}

/// Raw result of a tool before output bounding.
struct ToolOutcome {
    output: String,
    is_error: bool,
    exit_code: Option<i32>,
    changed_paths: Vec<String>,
    images: Vec<ToolImage>,
}

impl ToolOutcome {
    fn text(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: false,
            exit_code: None,
            changed_paths: Vec::new(),
            images: Vec::new(),
        }
    }

    fn changed(output: impl Into<String>, paths: Vec<String>) -> Self {
        Self {
            output: output.into(),
            is_error: false,
            exit_code: None,
            changed_paths: paths,
            images: Vec::new(),
        }
    }

    fn with_images(output: impl Into<String>, images: Vec<ToolImage>) -> Self {
        Self {
            output: output.into(),
            is_error: false,
            exit_code: None,
            changed_paths: Vec::new(),
            images,
        }
    }
}

async fn run_tool(
    context: &BrokerContext<'_>,
    request: &ToolExecRequest,
    plan: &CallPlan,
    workspace: Option<&Path>,
    output: Option<&mpsc::UnboundedSender<String>>,
) -> anyhow::Result<ToolOutcome> {
    let arguments = &request.arguments;
    match request.tool.as_str() {
        "workspace_info" => workspace_info(context, workspace).await,
        "fs_list" => fs_list(context, plan, workspace, arguments).await,
        "fs_read" => fs_read(context, plan, workspace, arguments).await,
        "doc_read" => doc_read(context, plan, workspace, arguments).await,
        "fs_stat" => fs_stat(context, plan, workspace, arguments).await,
        "fs_search" => fs_search(context, plan, workspace, arguments).await,
        "fs_find" => fs_find(context, plan, workspace, arguments).await,
        "fs_read_many" => fs_read_many(context, plan, workspace, arguments).await,
        "fs_tree" => fs_tree(context, plan, workspace, arguments).await,
        "fs_write" => fs_write(context, plan, workspace, arguments).await,
        "fs_patch" => fs_patch(context, plan, workspace, arguments).await,
        "fs_mkdir" => fs_mkdir(context, plan, workspace, arguments).await,
        "fs_copy" => fs_copy(context, plan, workspace, arguments).await,
        "fs_move" => fs_move(context, plan, workspace, arguments).await,
        "fs_delete" => fs_delete(context, plan, workspace, arguments).await,
        "shell_run" => shell_run(context, plan, workspace, arguments, output).await,
        "shell_start" => shell_start(context, plan, workspace, arguments).await,
        "shell_output" => shell_output(context, arguments).await,
        "shell_input" => shell_input(context, arguments).await,
        "shell_terminate" => shell_terminate(context, arguments).await,
        "git_status" => {
            git(
                context,
                plan,
                workspace,
                arguments,
                &["status", "--porcelain=v1", "--branch"],
            )
            .await
        }
        "git_diff" => git_diff(context, plan, workspace, arguments).await,
        "git_log" => git_log(context, plan, workspace, arguments).await,
        "git_show" => git_show(context, plan, workspace, arguments).await,
        "git_blame" => git_blame(context, plan, workspace, arguments).await,
        "git_grep" => git_grep(context, plan, workspace, arguments).await,
        "git_branch" => {
            git(
                context,
                plan,
                workspace,
                arguments,
                &["--no-pager", "branch", "--all", "--verbose", "--no-color"],
            )
            .await
        }
        "git_tags" => git_tags(context, plan, workspace, arguments).await,
        "git_worktree" => {
            git(
                context,
                plan,
                workspace,
                arguments,
                &["--no-pager", "worktree", "list", "--porcelain"],
            )
            .await
        }
        "git_diff_check" => git_diff_check(context, plan, workspace, arguments).await,
        "git_remote" => git_remote(context, plan, workspace, arguments).await,
        "project_test" | "project_build" | "project_lint" | "project_typecheck"
        | "project_format" => {
            project_check(context, plan, workspace, request.tool.as_str(), arguments).await
        }
        "env_info" => env_info(context, plan, workspace, arguments).await,
        "process_list" => process_list(context, plan, arguments).await,
        "code_symbols" => code_symbols(context, plan, workspace, arguments).await,
        "request_permission" => request_permission(context).await,
        "spawn_subagent" => anyhow::bail!(
            "`spawn_subagent` runs in the agent worker, not through the daemon exec path"
        ),
        "web_search" => web_search(context, workspace, arguments).await,
        "web_fetch" => web_fetch(context, workspace, arguments).await,
        "lsp_diagnostics" => lsp_diagnostics(context, plan, workspace, arguments).await,
        other if agent_policy::is_mcp_tool_name(other) => {
            mcp_tool(context, plan, other, arguments).await
        }
        other => anyhow::bail!("no executor for tool `{other}`"),
    }
}

async fn mcp_tool(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    name: &str,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    anyhow::ensure!(
        plan.environment == AgentEnvironment::Host,
        "MCP servers must run as explicitly approved host processes"
    );
    let (server_id, tool_name) =
        brazier_runtime::mcp::parse_tool_name(name).context("invalid MCP tool name")?;
    let encoded = serde_json::to_string(arguments).context("encode MCP tool arguments")?;
    let invocation =
        brazier_runtime::mcp::call_tool(context.data_dir, &server_id, &tool_name, &encoded).await;
    Ok(ToolOutcome {
        output: invocation.output,
        is_error: invocation.is_error,
        exit_code: None,
        changed_paths: Vec::new(),
        images: Vec::new(),
    })
}

/// Validate a tool-supplied path for this call, including symlink escapes.
fn checked_path(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
    field: &str,
    write: bool,
) -> anyhow::Result<PathBuf> {
    let raw = arguments
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("`{field}` is required"))?;
    anyhow::ensure!(!raw.trim().is_empty(), "`{field}` must not be empty");
    validate_platform_path(Path::new(raw))?;
    let resolved = agent_policy::resolve_path(workspace, raw);
    validate_platform_path(&resolved)?;

    // Follow symlinks on the deepest existing ancestor: a link inside the
    // workspace must not point outside it. Every containment check below is on
    // this canonical form, never on the path the caller typed.
    let real = agent_policy::canonical_ancestor(&resolved);
    for secret in secret_paths(Some(context.data_dir)) {
        let secret_real = agent_policy::canonical_ancestor(&secret);
        anyhow::ensure!(
            !is_inside(&real, &secret)
                && !is_inside(&real, &secret_real)
                && !is_inside(&resolved, &secret)
                && !is_inside(&resolved, &secret_real),
            "`{raw}` resolves into a credential or Brazier-owned path"
        );
    }
    if plan.environment == AgentEnvironment::Sandbox {
        let root = workspace.context("this session has no workspace")?;
        let root_real = agent_policy::canonical_ancestor(root);
        anyhow::ensure!(
            is_inside(&real, root) || is_inside(&real, &root_real),
            "`{raw}` is outside the workspace; request host access for it"
        );
        if write && !plan.profile.allows_workspace_writes() {
            anyhow::bail!(
                "the {} profile does not allow writes",
                plan.profile.as_str()
            );
        }
    }
    Ok(resolved)
}

#[cfg(not(windows))]
fn validate_platform_path(_path: &Path) -> anyhow::Result<()> {
    Ok(())
}

/// Reject Win32 path spellings that can alias a different object than the
/// lexical path names. Canonical drive and UNC paths (including the verbatim
/// forms returned by `canonicalize`) remain valid; raw device namespaces,
/// alternate data streams, and reserved DOS device aliases do not.
#[cfg(windows)]
fn validate_platform_path(path: &Path) -> anyhow::Result<()> {
    use std::path::{Component, Prefix};

    for component in path.components() {
        match component {
            Component::Prefix(prefix) => match prefix.kind() {
                Prefix::DeviceNS(_) | Prefix::Verbatim(_) => {
                    anyhow::bail!("Windows device namespace paths are not supported")
                }
                Prefix::Disk(_) | Prefix::VerbatimDisk(_) => anyhow::ensure!(
                    path.has_root(),
                    "Windows drive-relative paths are not supported"
                ),
                Prefix::UNC(_, _) | Prefix::VerbatimUNC(_, _) => {}
            },
            Component::Normal(name) => {
                let name = name.to_string_lossy();
                anyhow::ensure!(
                    !name.contains(':'),
                    "Windows alternate data stream paths are not supported"
                );
                anyhow::ensure!(
                    !name.ends_with(['.', ' ']),
                    "Windows path components ending in a dot or space are not supported"
                );

                let base = name
                    .split('.')
                    .next()
                    .unwrap_or_default()
                    .to_ascii_uppercase();
                let numbered_device = base
                    .strip_prefix("COM")
                    .or_else(|| base.strip_prefix("LPT"))
                    .is_some_and(|suffix| {
                        matches!(suffix, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
                    });
                anyhow::ensure!(
                    !matches!(
                        base.as_str(),
                        "CON" | "PRN" | "AUX" | "NUL" | "CLOCK$" | "CONIN$" | "CONOUT$"
                    ) && !numbered_device,
                    "Windows reserved device path component `{name}` is not supported"
                );
            }
            Component::RootDir | Component::CurDir | Component::ParentDir => {}
        }
    }
    Ok(())
}

/// Workspace-relative label for a path, for output the model and the UI read.
fn relative_display(workspace: Option<&Path>, path: &Path) -> String {
    let real = agent_policy::canonical_ancestor(path);
    workspace
        .and_then(|root| {
            let root_real = agent_policy::canonical_ancestor(root);
            real.strip_prefix(root)
                .or_else(|_| real.strip_prefix(&root_real))
                .ok()
                .map(Path::to_path_buf)
        })
        .map(|relative| relative.display().to_string())
        .unwrap_or_else(|| path.display().to_string())
}

async fn workspace_info(
    context: &BrokerContext<'_>,
    workspace: Option<&Path>,
) -> anyhow::Result<ToolOutcome> {
    let sandbox = context.broker.capabilities();
    let mut info = json!({
        "workspace": workspace.map(|path| path.display().to_string()),
        "permission_mode": context.session.permission_mode.as_str(),
        "sandbox": {
            "backend": sandbox.backend,
            "isolated": sandbox.isolated,
            "sandboxed_execution": sandbox.sandboxed_execution,
            "detail": sandbox.detail,
        },
        "platform": std::env::consts::OS,
        "architecture": std::env::consts::ARCH,
    });
    if let Some(root) = workspace {
        let git_dir = root.join(".git");
        info["git_repository"] = Value::Bool(git_dir.exists());
        let mut entries = Vec::new();
        if let Ok(mut dir) = tokio::fs::read_dir(root).await {
            while let Ok(Some(entry)) = dir.next_entry().await {
                entries.push(entry.file_name().to_string_lossy().to_string());
                if entries.len() >= 100 {
                    break;
                }
            }
        }
        entries.sort();
        info["top_level_entries"] = json!(entries);
    }
    Ok(ToolOutcome::text(serde_json::to_string_pretty(&info)?))
}

async fn fs_list(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let path = if arguments.get("path").is_some() {
        checked_path(context, plan, workspace, arguments, "path", false)?
    } else {
        workspace
            .context("this session has no workspace")?
            .to_path_buf()
    };
    let depth = arguments
        .get("depth")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .min(4) as usize;
    let boundary = walk_boundary(&path, workspace);
    let lines = list_directory(&path, workspace, depth, boundary.as_deref()).await?;
    if lines.is_empty() {
        return Ok(ToolOutcome::text("(empty directory)"));
    }
    Ok(ToolOutcome::text(lines.join("\n")))
}

/// When the walk root sits inside the workspace, keep every followed symlink
/// inside that workspace too. Host-approved roots outside the workspace keep
/// normal follow semantics.
fn walk_boundary(root: &Path, workspace: Option<&Path>) -> Option<PathBuf> {
    let workspace = workspace?;
    let workspace_real = agent_policy::canonical_ancestor(workspace);
    let root_real = agent_policy::canonical_ancestor(root);
    if is_inside(&root_real, &workspace_real) {
        Some(workspace_real)
    } else {
        None
    }
}

/// Resolve an entry for walking. Symlinks are followed only when their target
/// remains inside `boundary` (when set); escapes are skipped.
fn walk_entry_target(path: &Path, boundary: Option<&Path>) -> Option<(PathBuf, std::fs::Metadata)> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    if meta.file_type().is_symlink() {
        let real = agent_policy::canonical_ancestor(path);
        if let Some(root) = boundary
            && !is_inside(&real, root)
        {
            return None;
        }
        let followed = std::fs::metadata(path).ok()?;
        Some((real, followed))
    } else {
        let real = agent_policy::canonical_ancestor(path);
        if let Some(root) = boundary
            && !is_inside(&real, root)
        {
            return None;
        }
        Some((real, meta))
    }
}

/// Depth-first listing, walked iteratively so the recursion stays off the async
/// stack. Children are pushed in reverse so output reads in sorted order.
async fn list_directory(
    root: &Path,
    workspace: Option<&Path>,
    depth: usize,
    boundary: Option<&Path>,
) -> anyhow::Result<Vec<String>> {
    let mut lines = Vec::new();
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    let mut visited_root = false;
    while let Some((directory, level)) = stack.pop() {
        let read = tokio::fs::read_dir(&directory).await;
        let mut dir = match read {
            Ok(dir) => dir,
            Err(error) if !visited_root => {
                return Err(anyhow::Error::from(error))
                    .with_context(|| format!("cannot list {}", directory.display()));
            }
            // A directory that vanished mid-walk is not worth failing the call.
            Err(_) => continue,
        };
        visited_root = true;
        let mut entries = Vec::new();
        while let Some(entry) = dir.next_entry().await? {
            entries.push(entry);
        }
        entries.sort_by_key(|entry| entry.file_name());
        let mut children = Vec::new();
        for entry in entries {
            let name = entry.file_name().to_string_lossy().to_string();
            let entry_path = entry.path();
            let indent = "  ".repeat(level);
            let Some((_real, metadata)) = walk_entry_target(&entry_path, boundary) else {
                // Symlink (or mount) that leaves the workspace: name it without
                // following so the model can see it exists without reading out.
                if entry
                    .file_type()
                    .await
                    .map(|kind| kind.is_symlink())
                    .unwrap_or(false)
                {
                    lines.push(format!(
                        "{indent}{} (symlink outside workspace)",
                        relative_display(workspace, &entry_path)
                    ));
                }
                continue;
            };
            if metadata.is_dir() {
                lines.push(format!(
                    "{indent}{}/",
                    relative_display(workspace, &entry_path)
                ));
                if level + 1 < depth && !SKIPPED_DIRECTORIES.contains(&name.as_str()) {
                    children.push((entry_path, level + 1));
                }
            } else {
                lines.push(format!(
                    "{indent}{} ({} bytes)",
                    relative_display(workspace, &entry_path),
                    metadata.len()
                ));
            }
            if lines.len() >= 2_000 {
                lines.push("[... listing truncated ...]".to_owned());
                return Ok(lines);
            }
        }
        children.reverse();
        stack.extend(children);
    }
    Ok(lines)
}

async fn fs_read(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let path = checked_path(context, plan, workspace, arguments, "path", false)?;
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    let canonical_parent = std::fs::canonicalize(parent)
        .with_context(|| format!("cannot canonicalize {}", parent.display()))?;
    let file_name = path
        .file_name()
        .with_context(|| format!("{} has no file name", path.display()))?;
    let opened_path = canonical_parent.join(file_name);
    let (file, actual_path) = open_read_without_final_link(&opened_path)
        .with_context(|| format!("cannot open {}", path.display()))?;
    validate_platform_path(&actual_path)?;
    for secret in secret_paths(Some(context.data_dir)) {
        let secret_real = agent_policy::canonical_ancestor(&secret);
        anyhow::ensure!(
            !is_inside(&actual_path, &secret) && !is_inside(&actual_path, &secret_real),
            "`{}` resolves into a credential or Brazier-owned path",
            path.display()
        );
    }
    if plan.environment == AgentEnvironment::Sandbox {
        let root = workspace.context("this session has no workspace")?;
        let root_real = agent_policy::canonical_ancestor(root);
        anyhow::ensure!(
            is_inside(&actual_path, root) || is_inside(&actual_path, &root_real),
            "`{}` is outside the workspace; request host access for it",
            path.display()
        );
    }
    let size = file
        .metadata()
        .with_context(|| format!("cannot inspect {}", path.display()))?
        .len();
    anyhow::ensure!(
        size <= MAX_READ_BYTES as u64,
        "{} is {} bytes, larger than the {MAX_READ_BYTES}-byte read limit. Pass \
         `start_line` and `line_count` to read a smaller range of a file that fits.",
        path.display(),
        size
    );
    let mut file = tokio::fs::File::from_std(file);
    let mut bytes = Vec::with_capacity(size as usize);
    file.read_to_end(&mut bytes).await?;
    let text = String::from_utf8_lossy(&bytes);
    let start_line = arguments
        .get("start_line")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1) as usize;
    let line_count = arguments
        .get("line_count")
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX) as usize;
    let numbered: Vec<String> = text
        .lines()
        .enumerate()
        .skip(start_line - 1)
        .take(line_count)
        .map(|(index, line)| format!("{:>6}\t{line}", index + 1))
        .collect();
    if numbered.is_empty() {
        return Ok(ToolOutcome::text("(no lines in that range)"));
    }
    Ok(ToolOutcome::text(numbered.join("\n")))
}

#[cfg(unix)]
fn open_read_without_final_link(path: &Path) -> std::io::Result<(std::fs::File, PathBuf)> {
    use std::os::unix::fs::OpenOptionsExt as _;

    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    Ok((file, path.to_path_buf()))
}

#[cfg(windows)]
fn open_read_without_final_link(path: &Path) -> std::io::Result<(std::fs::File, PathBuf)> {
    use std::{
        ffi::c_void, os::windows::fs::MetadataExt as _, os::windows::fs::OpenOptionsExt as _,
    };
    use std::{os::windows::io::AsRawHandle as _, ptr};

    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetFinalPathNameByHandleW(
            file: *mut c_void,
            path: *mut u16,
            path_len: u32,
            flags: u32,
        ) -> u32;
    }

    let file = OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)?;
    if file.metadata()?.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "refusing to read a Windows reparse point",
        ));
    }

    let handle = file.as_raw_handle();
    // A zero-length query returns the UTF-16 buffer length required for the
    // normalized DOS-volume path. The handle pins the opened object while the
    // caller validates that resolved path against the workspace boundary.
    let required = unsafe { GetFinalPathNameByHandleW(handle, ptr::null_mut(), 0, 0) };
    if required == 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mut buffer = vec![0_u16; required as usize + 1];
    let written =
        unsafe { GetFinalPathNameByHandleW(handle, buffer.as_mut_ptr(), buffer.len() as u32, 0) };
    if written == 0 {
        return Err(std::io::Error::last_os_error());
    }
    if written as usize >= buffer.len() {
        return Err(std::io::Error::other(
            "opened Windows path changed while it was being resolved",
        ));
    }

    use std::os::windows::ffi::OsStringExt as _;
    let resolved = std::ffi::OsString::from_wide(&buffer[..written as usize]);
    Ok((file, PathBuf::from(resolved)))
}

#[cfg(not(any(unix, windows)))]
fn open_read_without_final_link(_path: &Path) -> std::io::Result<(std::fs::File, PathBuf)> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "secure direct file reads are not supported on this platform",
    ))
}

async fn doc_read(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    // A `document` id refers to a PDF a previous web_fetch attached (stored as
    // a blob), mirroring the chat `doc_read` attachment flow. Otherwise the
    // path-based workspace form applies.
    let path = match arguments.get("document").and_then(Value::as_str) {
        Some(requested) => {
            let sha256 = requested
                .strip_prefix("brazier_blob:")
                .unwrap_or(requested)
                .trim();
            let path = brazier_runtime::blob_store::blob_path(context.data_dir, sha256)
                .context("invalid `document` id")?;
            anyhow::ensure!(
                path.is_file(),
                "that document is no longer stored locally — re-fetch it with web_fetch"
            );
            path
        }
        None => checked_path(context, plan, workspace, arguments, "path", false)?,
    };
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("document");
    let kind = match brazier_runtime::documents::kind_for_name(name) {
        Some(kind) => kind,
        None => {
            // Web-fetched blobs are stored without an extension; sniff the
            // magic bytes instead.
            let mut header = [0_u8; 8];
            let mut file = tokio::fs::File::open(&path)
                .await
                .context("open document to identify its kind")?;
            let read = file.read(&mut header).await?;
            if header[..read].starts_with(b"%PDF-") {
                brazier_runtime::documents::DocumentKind::Pdf
            } else {
                anyhow::bail!(
                    "{} is not a PDF, RTF, DOC, or DOCX — use fs_read for text files",
                    relative_display(workspace, &path)
                )
            }
        }
    };

    let render = arguments
        .get("render_pages")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if render {
        anyhow::ensure!(
            kind == brazier_runtime::documents::DocumentKind::Pdf,
            "render_pages only applies to PDFs"
        );
        let start = arguments
            .get("start_page")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .unwrap_or(1)
            .max(1);
        let end = arguments
            .get("end_page")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .unwrap_or(start + brazier_runtime::documents::DEFAULT_PAGE_COUNT - 1)
            .max(start);
        let count = (end - start + 1).min(brazier_runtime::documents::MAX_RENDER_PAGES);
        let rendered =
            brazier_runtime::documents::render_pages(context.data_dir, &path, start, count).await?;
        let pages: Vec<String> = rendered
            .iter()
            .map(|page| format!("page {}", page.page))
            .collect();
        let images = rendered
            .into_iter()
            .map(|page| ToolImage {
                data: page.base64_data(),
                mime_type: page.mime_type,
            })
            .collect::<Vec<_>>();
        return Ok(ToolOutcome::with_images(
            format!(
                "Rendered {} ({}) as images. The pages are included for a vision model.",
                relative_display(workspace, &path),
                pages.join(", ")
            ),
            images,
        ));
    }

    let pages = if kind == brazier_runtime::documents::DocumentKind::Pdf {
        let start = arguments
            .get("start_page")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .unwrap_or(1)
            .max(1);
        let end = arguments
            .get("end_page")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .unwrap_or(start + brazier_runtime::documents::DEFAULT_PAGE_COUNT - 1)
            .max(start);
        anyhow::ensure!(
            end - start < brazier_runtime::documents::MAX_TEXT_PAGES,
            "PDF text window is limited to {} pages; narrow start_page/end_page",
            brazier_runtime::documents::MAX_TEXT_PAGES
        );
        Some((start, end))
    } else {
        None
    };
    let lines = if kind == brazier_runtime::documents::DocumentKind::Pdf {
        None
    } else {
        match (
            arguments.get("start_line").and_then(Value::as_u64),
            arguments.get("end_line").and_then(Value::as_u64),
        ) {
            (None, None) => None,
            (start, end) => {
                let start = start.unwrap_or(1).max(1) as usize;
                let end = end
                    .map(|value| value as usize)
                    .unwrap_or(start.saturating_add(199))
                    .max(start);
                Some((start, end))
            }
        }
    };
    let extraction = brazier_runtime::documents::extract_text(
        &path,
        kind,
        pages,
        lines,
        brazier_runtime::documents::MAX_EXTRACTION_CHARS,
    )
    .await?;
    Ok(ToolOutcome::text(extraction.describe()))
}

async fn fs_stat(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let path = checked_path(context, plan, workspace, arguments, "path", false)?;
    let metadata = tokio::fs::symlink_metadata(&path)
        .await
        .with_context(|| format!("cannot stat {}", path.display()))?;
    let kind = if metadata.is_dir() {
        "directory"
    } else if metadata.file_type().is_symlink() {
        "symlink"
    } else {
        "file"
    };
    Ok(ToolOutcome::text(serde_json::to_string_pretty(&json!({
        "path": relative_display(workspace, &path),
        "kind": kind,
        "size_bytes": metadata.len(),
        "read_only": metadata.permissions().readonly(),
    }))?))
}

/// Literal, case-insensitive-optional content search. Deliberately not a regex
/// engine: the tool description tells the model to pass plain text.
async fn fs_search(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let query = arguments
        .get("query")
        .and_then(Value::as_str)
        .context("`query` is required")?;
    anyhow::ensure!(!query.is_empty(), "`query` must not be empty");
    let root = if arguments.get("path").is_some() {
        checked_path(context, plan, workspace, arguments, "path", false)?
    } else {
        workspace
            .context("this session has no workspace")?
            .to_path_buf()
    };
    let case_sensitive = arguments
        .get("case_sensitive")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let name_filter = arguments.get("name_glob").and_then(Value::as_str);
    let needle = if case_sensitive {
        query.to_owned()
    } else {
        query.to_lowercase()
    };

    let boundary = walk_boundary(&root, workspace);
    let mut matches = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(directory) = stack.pop() {
        let Ok(mut dir) = tokio::fs::read_dir(&directory).await else {
            continue;
        };
        while let Ok(Some(entry)) = dir.next_entry().await {
            let entry_path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            let Some((_real, metadata)) = walk_entry_target(&entry_path, boundary.as_deref())
            else {
                continue;
            };
            if metadata.is_dir() {
                if !SKIPPED_DIRECTORIES.contains(&name.as_str()) {
                    stack.push(entry_path);
                }
                continue;
            }
            if let Some(pattern) = name_filter
                && !glob_match(pattern, &name)
            {
                continue;
            }
            if metadata.len() > 4 * 1024 * 1024 {
                continue;
            }
            let Ok(bytes) = tokio::fs::read(&entry_path).await else {
                continue;
            };
            if bytes.contains(&0) {
                continue; // binary
            }
            let text = String::from_utf8_lossy(&bytes);
            for (index, line) in text.lines().enumerate() {
                let candidate = if case_sensitive {
                    line.to_owned()
                } else {
                    line.to_lowercase()
                };
                if candidate.contains(&needle) {
                    matches.push(format!(
                        "{}:{}: {}",
                        relative_display(workspace, &entry_path),
                        index + 1,
                        line.trim_end().chars().take(300).collect::<String>()
                    ));
                    if matches.len() >= MAX_SEARCH_RESULTS {
                        matches.push(format!(
                            "[... stopped at {MAX_SEARCH_RESULTS} matches; narrow the query ...]"
                        ));
                        return Ok(ToolOutcome::text(matches.join("\n")));
                    }
                }
            }
        }
    }
    if matches.is_empty() {
        return Ok(ToolOutcome::text(format!("No matches for `{query}`.")));
    }
    Ok(ToolOutcome::text(matches.join("\n")))
}

/// Find files and directories by a small, predictable glob language. Unlike a
/// shell find, this never executes a predicate and never follows a symlink
/// outside the workspace.
async fn fs_find(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let pattern = arguments
        .get("pattern")
        .and_then(Value::as_str)
        .context("pattern is required")?
        .trim();
    anyhow::ensure!(!pattern.is_empty(), "pattern must not be empty");
    let root = if arguments.get("path").is_some() {
        checked_path(context, plan, workspace, arguments, "path", false)?
    } else {
        workspace
            .context("this session has no workspace")?
            .to_path_buf()
    };
    let kind = arguments
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("any");
    anyhow::ensure!(
        matches!(kind, "any" | "file" | "directory"),
        "kind must be any, file, or directory"
    );
    let max_results = arguments
        .get("max_results")
        .and_then(Value::as_u64)
        .unwrap_or(200)
        .clamp(1, 500) as usize;
    let boundary = walk_boundary(&root, workspace);
    let mut results = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(directory) = stack.pop() {
        let Ok(mut entries) = tokio::fs::read_dir(&directory).await else {
            continue;
        };
        while let Some(entry) = entries.next_entry().await? {
            let entry_path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            let Some((_real, metadata)) = walk_entry_target(&entry_path, boundary.as_deref())
            else {
                continue;
            };
            let is_directory = metadata.is_dir();
            let relative = entry_path
                .strip_prefix(workspace.unwrap_or(&root))
                .unwrap_or(&entry_path)
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");
            let matches_kind = kind == "any"
                || (kind == "directory" && is_directory)
                || (kind == "file" && !is_directory);
            if matches_kind && (glob_match(pattern, &relative) || glob_match(pattern, &name)) {
                results.push(if is_directory {
                    format!("{relative}/")
                } else {
                    relative
                });
                if results.len() >= max_results {
                    results.push(format!("[... stopped at {max_results} results ...]"));
                    return Ok(ToolOutcome::text(results.join("\n")));
                }
            }
            if is_directory && !SKIPPED_DIRECTORIES.contains(&name.as_str()) {
                stack.push(entry_path);
            }
        }
    }
    results.sort();
    if results.is_empty() {
        return Ok(ToolOutcome::text(format!("No paths matched {pattern}.")));
    }
    Ok(ToolOutcome::text(results.join("\n")))
}

async fn fs_read_many(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let paths = arguments
        .get("paths")
        .and_then(Value::as_array)
        .context("paths is required")?;
    anyhow::ensure!(!paths.is_empty(), "paths must not be empty");
    anyhow::ensure!(paths.len() <= 16, "paths is limited to 16 files");
    let max_bytes = arguments
        .get("max_bytes_each")
        .and_then(Value::as_u64)
        .unwrap_or(128 * 1024)
        .clamp(1024, 512 * 1024) as usize;
    let mut output = String::new();
    for value in paths {
        let raw = value
            .as_str()
            .context("every entry in paths must be a string")?;
        let path_arguments = json!({ "path": raw });
        let path = checked_path(context, plan, workspace, &path_arguments, "path", false)?;
        let bytes = tokio::fs::read(&path)
            .await
            .with_context(|| format!("cannot read {raw}"))?;
        output.push_str(&format!("--- {} ---\n", relative_display(workspace, &path)));
        if bytes.len() > max_bytes {
            output.push_str(&format!(
                "[skipped: {} bytes exceeds max_bytes_each={max_bytes}]\n\n",
                bytes.len()
            ));
            continue;
        }
        let text = String::from_utf8_lossy(&bytes);
        for (index, line) in text.lines().enumerate() {
            output.push_str(&format!("{:>6}\t{line}\n", index + 1));
        }
        output.push('\n');
    }
    Ok(ToolOutcome::text(output))
}

async fn fs_tree(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let path = if arguments.get("path").is_some() {
        checked_path(context, plan, workspace, arguments, "path", false)?
    } else {
        workspace
            .context("this session has no workspace")?
            .to_path_buf()
    };
    let depth = arguments
        .get("depth")
        .and_then(Value::as_u64)
        .unwrap_or(3)
        .clamp(1, 6) as usize;
    let max_entries = arguments
        .get("max_entries")
        .and_then(Value::as_u64)
        .unwrap_or(500)
        .clamp(1, 2_000) as usize;
    let boundary = walk_boundary(&path, workspace);
    let mut lines = list_directory(&path, workspace, depth, boundary.as_deref()).await?;
    if lines.len() > max_entries {
        lines.truncate(max_entries);
        lines.push(format!("[... truncated at {max_entries} entries ...]"));
    }
    if lines.is_empty() {
        lines.push("(empty directory)".to_owned());
    }
    Ok(ToolOutcome::text(lines.join("\n")))
}

/// Minimal glob: `*` matches any run of characters, `?` one character.
fn glob_match(pattern: &str, value: &str) -> bool {
    fn matches(pattern: &[char], value: &[char]) -> bool {
        match pattern.first() {
            None => value.is_empty(),
            Some('*') => (0..=value.len()).any(|split| matches(&pattern[1..], &value[split..])),
            Some('?') => !value.is_empty() && matches(&pattern[1..], &value[1..]),
            Some(expected) => {
                value.first() == Some(expected) && matches(&pattern[1..], &value[1..])
            }
        }
    }
    let pattern: Vec<char> = pattern.chars().collect();
    let value: Vec<char> = value.chars().collect();
    matches(&pattern, &value)
}

async fn fs_write(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let path = checked_path(context, plan, workspace, arguments, "path", true)?;
    let content = arguments
        .get("content")
        .and_then(Value::as_str)
        .context("`content` is required")?;
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(&path, content.as_bytes())
        .await
        .with_context(|| format!("cannot write {}", path.display()))?;
    let display = relative_display(workspace, &path);
    Ok(ToolOutcome::changed(
        format!("Wrote {} bytes to {display}.", content.len()),
        vec![display],
    ))
}

async fn fs_patch(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let path = checked_path(context, plan, workspace, arguments, "path", true)?;
    let old = arguments
        .get("old_string")
        .and_then(Value::as_str)
        .context("`old_string` is required")?;
    let new = arguments
        .get("new_string")
        .and_then(Value::as_str)
        .context("`new_string` is required")?;
    anyhow::ensure!(!old.is_empty(), "`old_string` must not be empty");
    let replace_all = arguments
        .get("replace_all")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let original = tokio::fs::read_to_string(&path)
        .await
        .with_context(|| format!("cannot read {}", path.display()))?;
    let occurrences = original.matches(old).count();
    anyhow::ensure!(
        occurrences > 0,
        "`old_string` does not appear in {}",
        path.display()
    );
    anyhow::ensure!(
        replace_all || occurrences == 1,
        "`old_string` appears {occurrences} times in {}; pass more context or set replace_all",
        path.display()
    );
    let patched = if replace_all {
        original.replace(old, new)
    } else {
        original.replacen(old, new, 1)
    };
    tokio::fs::write(&path, patched.as_bytes()).await?;
    let display = relative_display(workspace, &path);
    let replaced = if replace_all { occurrences } else { 1 };
    Ok(ToolOutcome::changed(
        format!("Replaced {replaced} occurrence(s) in {display}."),
        vec![display],
    ))
}

async fn fs_mkdir(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let path = checked_path(context, plan, workspace, arguments, "path", true)?;
    tokio::fs::create_dir_all(&path)
        .await
        .with_context(|| format!("cannot create {}", path.display()))?;
    let display = relative_display(workspace, &path);
    Ok(ToolOutcome::changed(
        format!("Created directory {display}."),
        vec![display],
    ))
}

async fn fs_copy(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let from = checked_path(context, plan, workspace, arguments, "from", false)?;
    let to = checked_path(context, plan, workspace, arguments, "to", true)?;
    let metadata = tokio::fs::symlink_metadata(&from).await?;
    anyhow::ensure!(
        !metadata.is_dir(),
        "fs_copy handles files; copy directory contents file by file"
    );
    if let Some(parent) = to.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::copy(&from, &to).await?;
    let display = relative_display(workspace, &to);
    Ok(ToolOutcome::changed(
        format!(
            "Copied {} to {display}.",
            relative_display(workspace, &from)
        ),
        vec![display],
    ))
}

async fn fs_move(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let from = checked_path(context, plan, workspace, arguments, "from", true)?;
    let to = checked_path(context, plan, workspace, arguments, "to", true)?;
    if let Some(parent) = to.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::rename(&from, &to)
        .await
        .with_context(|| format!("cannot move {} to {}", from.display(), to.display()))?;
    let from_display = relative_display(workspace, &from);
    let to_display = relative_display(workspace, &to);
    Ok(ToolOutcome::changed(
        format!("Moved {from_display} to {to_display}."),
        vec![from_display, to_display],
    ))
}

async fn fs_delete(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let path = checked_path(context, plan, workspace, arguments, "path", true)?;
    // Deleting the workspace root itself is never what the user meant. Compare
    // canonical forms so a differently spelled path cannot slip past.
    if let Some(root) = workspace {
        anyhow::ensure!(
            agent_policy::canonical_ancestor(&path) != root,
            "refusing to delete the workspace root {}",
            root.display()
        );
    }
    let metadata = tokio::fs::symlink_metadata(&path)
        .await
        .with_context(|| format!("{} does not exist", path.display()))?;
    let recursive = arguments
        .get("recursive")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if metadata.is_dir() {
        anyhow::ensure!(
            recursive,
            "{} is a directory; pass recursive=true to delete it",
            path.display()
        );
        tokio::fs::remove_dir_all(&path).await?;
    } else {
        tokio::fs::remove_file(&path).await?;
    }
    let display = relative_display(workspace, &path);
    Ok(ToolOutcome::changed(
        format!("Deleted {display}."),
        vec![display],
    ))
}

/// Scratch directory for a session: writable inside the sandbox, and the
/// TMPDIR every wrapped command sees.
async fn scratch_directory(context: &BrokerContext<'_>) -> anyhow::Result<PathBuf> {
    let path = context
        .data_dir
        .join("agent")
        .join("scratch")
        .join(&context.session.id);
    tokio::fs::create_dir_all(&path).await?;
    Ok(path)
}

fn shell_program() -> (&'static str, &'static str) {
    if cfg!(target_os = "windows") {
        ("cmd", "/C")
    } else {
        ("/bin/sh", "-c")
    }
}

/// Resolve the working directory for a command, defaulting to the workspace.
fn command_cwd(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<PathBuf> {
    if arguments.get("cwd").and_then(Value::as_str).is_some() {
        return checked_path(context, plan, workspace, arguments, "cwd", false);
    }
    Ok(workspace
        .context("this session has no workspace")?
        .to_path_buf())
}

async fn build_command(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
    command_line: &str,
) -> anyhow::Result<(Command, SandboxDescription)> {
    let workspace_root = workspace.context("this session has no workspace")?;
    let scratch = scratch_directory(context).await?;
    let cwd = command_cwd(context, plan, workspace, arguments)?;
    let sandbox_request = SandboxRequest {
        profile: plan.profile,
        workspace: workspace_root,
        scratch: &scratch,
        cwd: &cwd,
        data_dir: Some(context.data_dir),
    };
    let (shell, flag) = shell_program();
    let args = vec![flag.to_owned(), command_line.to_owned()];
    let extra_env = BTreeMap::new();
    let wrapped = match plan.environment {
        AgentEnvironment::Sandbox => {
            context
                .broker
                .backend()
                .wrap(&sandbox_request, shell, &args, &extra_env)?
        }
        AgentEnvironment::Host => {
            context
                .broker
                .backend()
                .wrap_host(&sandbox_request, shell, &args, &extra_env)?
        }
    };
    let mut command = Command::new(&wrapped.program);
    command
        .args(&wrapped.args)
        .current_dir(&cwd)
        .env_clear()
        .envs(&wrapped.env)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    Ok((command, wrapped.description))
}

async fn shell_run(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
    output: Option<&mpsc::UnboundedSender<String>>,
) -> anyhow::Result<ToolOutcome> {
    let command_line = arguments
        .get("command")
        .and_then(Value::as_str)
        .context("`command` is required")?;
    let timeout = arguments
        .get("timeout_ms")
        .and_then(Value::as_u64)
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_SHELL_TIMEOUT)
        .min(MAX_SHELL_TIMEOUT);
    let (mut command, _) = build_command(context, plan, workspace, arguments, command_line).await?;
    let mut child = command
        .spawn()
        .with_context(|| format!("cannot start `{command_line}`"))?;
    drop(child.stdin.take());

    let stdout = child
        .stdout
        .take()
        .context("cannot capture command stdout")?;
    let stderr = child
        .stderr
        .take()
        .context("cannot capture command stderr")?;
    let stdout_task = tokio::spawn(read_shell_pipe(stdout, output.cloned(), None));
    let stderr_task = tokio::spawn(read_shell_pipe(stderr, output.cloned(), Some("[stderr]\n")));
    let status = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(result) => result?,
        Err(_) => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            let _ = stdout_task.await;
            let _ = stderr_task.await;
            return Ok(ToolOutcome {
                output: format!(
                    "`{command_line}` was still running after {} ms and was terminated.",
                    timeout.as_millis()
                ),
                is_error: true,
                exit_code: None,
                changed_paths: Vec::new(),
                images: Vec::new(),
            });
        }
    };
    let stdout = String::from_utf8_lossy(&stdout_task.await??).to_string();
    let stderr = String::from_utf8_lossy(&stderr_task.await??).to_string();
    let code = status.code();
    let mut body = String::new();
    if !stdout.is_empty() {
        body.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !body.is_empty() {
            body.push('\n');
        }
        body.push_str("[stderr]\n");
        body.push_str(&stderr);
    }
    if body.is_empty() {
        body.push_str("(no output)");
    }
    body.push_str(&format!(
        "\n\n[exit code {}]",
        code.map(|value| value.to_string())
            .unwrap_or_else(|| "signal".to_owned())
    ));
    Ok(ToolOutcome {
        output: body,
        is_error: code != Some(0),
        exit_code: code,
        changed_paths: Vec::new(),
        images: Vec::new(),
    })
}

/// Read a child pipe without waiting for process completion, forwarding chunks
/// to the HTTP stream while retaining the exact bytes for the final result.
async fn read_shell_pipe<R>(
    mut pipe: R,
    output: Option<mpsc::UnboundedSender<String>>,
    prefix: Option<&'static str>,
) -> anyhow::Result<Vec<u8>>
where
    R: AsyncRead + Unpin,
{
    let mut collected = Vec::new();
    let mut buffer = vec![0_u8; 8 * 1024];
    let mut sent_prefix = false;
    loop {
        let count = pipe.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        collected.extend_from_slice(&buffer[..count]);
        if let Some(sender) = &output {
            let mut chunk = String::new();
            if !sent_prefix {
                chunk.push_str(prefix.unwrap_or_default());
                sent_prefix = true;
            }
            chunk.push_str(&String::from_utf8_lossy(&buffer[..count]));
            let _ = sender.send(chunk);
        }
    }
    Ok(collected)
}

async fn shell_start(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let command_line = arguments
        .get("command")
        .and_then(Value::as_str)
        .context("`command` is required")?;
    let (mut command, _) = build_command(context, plan, workspace, arguments, command_line).await?;
    let mut child = command
        .spawn()
        .with_context(|| format!("cannot start `{command_line}`"))?;
    let stdin = child.stdin.take();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let buffer = Arc::new(Mutex::new(String::new()));
    for stream in [
        stdout.map(StreamKind::Stdout),
        stderr.map(StreamKind::Stderr),
    ]
    .into_iter()
    .flatten()
    {
        let buffer = Arc::clone(&buffer);
        tokio::spawn(async move {
            match stream {
                StreamKind::Stdout(pipe) => pump(pipe, buffer, "").await,
                StreamKind::Stderr(pipe) => pump(pipe, buffer, "[stderr] ").await,
            }
        });
    }
    let id = format!("proc-{}", Uuid::new_v4().simple());
    context.broker.processes.lock().await.insert(
        id.clone(),
        BackgroundProcess {
            session_id: context.session.id.clone(),
            command: command_line.to_owned(),
            child: Arc::new(Mutex::new(child)),
            stdin: Arc::new(Mutex::new(stdin)),
            output: buffer,
            started: Instant::now(),
        },
    );
    Ok(ToolOutcome::text(format!(
        "Started `{command_line}` as process {id}. Read its output with shell_output, send input \
         with shell_input, and stop it with shell_terminate."
    )))
}

enum StreamKind {
    Stdout(tokio::process::ChildStdout),
    Stderr(tokio::process::ChildStderr),
}

/// Copy a child pipe into the shared buffer, trimming the front when it grows
/// past the retention limit.
async fn pump<R>(pipe: R, buffer: Arc<Mutex<String>>, prefix: &str)
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut reader = BufReader::new(pipe).lines();
    while let Ok(Some(line)) = reader.next_line().await {
        let mut guard = buffer.lock().await;
        guard.push_str(prefix);
        guard.push_str(&line);
        guard.push('\n');
        if guard.len() > MAX_PROCESS_BUFFER_BYTES {
            let excess = guard.len() - MAX_PROCESS_BUFFER_BYTES;
            // Cut on a char boundary so the buffer stays valid UTF-8.
            let cut = guard
                .char_indices()
                .map(|(index, _)| index)
                .find(|index| *index >= excess)
                .unwrap_or(0);
            guard.replace_range(..cut, "");
        }
    }
}

async fn shell_output(
    context: &BrokerContext<'_>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let id = arguments
        .get("process_id")
        .and_then(Value::as_str)
        .context("`process_id` is required")?;
    let drain = arguments
        .get("drain")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let (command, started, output, child) = {
        let processes = context.broker.processes.lock().await;
        let process = processes
            .get(id)
            .with_context(|| format!("no background process {id}"))?;
        anyhow::ensure!(
            process.session_id == context.session.id,
            "process {id} belongs to another session"
        );
        (
            process.command.clone(),
            process.started,
            Arc::clone(&process.output),
            Arc::clone(&process.child),
        )
    };
    let mut buffer = output.lock().await;
    let text = buffer.clone();
    if drain {
        buffer.clear();
    }
    drop(buffer);
    let status = {
        let mut child_guard = child.lock().await;
        match child_guard.try_wait()? {
            Some(status) => format!("exited with {status}"),
            None => format!("running for {} s", started.elapsed().as_secs()),
        }
    };
    let body = if text.is_empty() {
        format!("(no new output) [{command}: {status}]")
    } else {
        format!("{text}\n[{command}: {status}]")
    };
    Ok(ToolOutcome::text(body))
}

async fn shell_input(
    context: &BrokerContext<'_>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let id = arguments
        .get("process_id")
        .and_then(Value::as_str)
        .context("`process_id` is required")?;
    let data = arguments
        .get("data")
        .and_then(Value::as_str)
        .context("`data` is required")?;
    let stdin = {
        let processes = context.broker.processes.lock().await;
        let process = processes
            .get(id)
            .with_context(|| format!("no background process {id}"))?;
        anyhow::ensure!(
            process.session_id == context.session.id,
            "process {id} belongs to another session"
        );
        Arc::clone(&process.stdin)
    };
    let mut stdin_guard = stdin.lock().await;
    let pipe = stdin_guard
        .as_mut()
        .context("that process no longer accepts input")?;
    pipe.write_all(data.as_bytes()).await?;
    if !data.ends_with('\n') {
        pipe.write_all(b"\n").await?;
    }
    pipe.flush().await?;
    Ok(ToolOutcome::text(format!(
        "Sent {} bytes to {id}.",
        data.len()
    )))
}

async fn shell_terminate(
    context: &BrokerContext<'_>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let id = arguments
        .get("process_id")
        .and_then(Value::as_str)
        .context("`process_id` is required")?;
    let mut processes = context.broker.processes.lock().await;
    let process = processes
        .remove(id)
        .with_context(|| format!("no background process {id}"))?;
    if process.session_id != context.session.id {
        // Put it back: another session owns it.
        processes.insert(id.to_owned(), process);
        anyhow::bail!("process {id} belongs to another session");
    }
    let mut child = process.child.lock().await;
    child.start_kill().ok();
    Ok(ToolOutcome::text(format!(
        "Terminated {id} (`{}`).",
        process.command
    )))
}

/// Report the access the session actually holds.
///
/// Reaching this executor means the user already reviewed the request, since the
/// policy always holds `request_permission` for approval. What matters to the
/// model now is what it can do, so this reports the standing grants rather than
/// restating the request.
async fn request_permission(context: &BrokerContext<'_>) -> anyhow::Result<ToolOutcome> {
    let grants = context.db.session_grants(&context.session.id).await?;
    if grants.is_empty() {
        return Ok(ToolOutcome::text(
            "The user reviewed the request but granted no standing access. Make the call you need: \
             it will be shown to them for approval on its own terms.",
        ));
    }
    Ok(ToolOutcome::text(format!(
        "Standing access for this session: {}. Anything outside that list is still asked about \
         each time.",
        grants.join(", ")
    )))
}

async fn git(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
    git_args: &[&str],
) -> anyhow::Result<ToolOutcome> {
    let quoted = git_args
        .iter()
        .map(|argument| shell_quote(argument))
        .collect::<Vec<_>>()
        .join(" ");
    let command_line = format!("git {quoted}");
    let mut merged = arguments.clone();
    if let Value::Object(map) = &mut merged {
        map.insert("command".to_owned(), Value::String(command_line.clone()));
    }
    shell_run(context, plan, workspace, &merged, None).await
}

async fn git_diff(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let staged = arguments
        .get("staged")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut git_args: Vec<&str> = vec!["--no-pager", "diff", "--no-color"];
    if staged {
        git_args.push("--staged");
    }
    let path = arguments.get("path").and_then(Value::as_str);
    let owned;
    if let Some(path) = path {
        // Validate the path before it reaches a command line.
        let checked = checked_path(context, plan, workspace, arguments, "path", false)?;
        owned = relative_display(workspace, &checked);
        git_args.push("--");
        git_args.push(&owned);
        let _ = path;
    }
    git(context, plan, workspace, arguments, &git_args).await
}

async fn git_owned(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
    git_args: &[String],
) -> anyhow::Result<ToolOutcome> {
    let command_line = std::iter::once("git".to_owned())
        .chain(git_args.iter().map(|argument| shell_quote(argument)))
        .collect::<Vec<_>>()
        .join(" ");
    let mut merged = arguments.clone();
    if let Value::Object(map) = &mut merged {
        map.insert("command".to_owned(), Value::String(command_line));
    }
    shell_run(context, plan, workspace, &merged, None).await
}

async fn git_log(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let max_count = arguments
        .get("max_count")
        .and_then(Value::as_u64)
        .unwrap_or(20)
        .clamp(1, 100);
    let mut args = vec![
        "--no-pager".to_owned(),
        "log".to_owned(),
        format!("-{max_count}"),
        "--decorate".to_owned(),
        "--date=short".to_owned(),
        "--format=%h %ad %an %d %s".to_owned(),
    ];
    if arguments.get("path").is_some() {
        let checked = checked_path(context, plan, workspace, arguments, "path", false)?;
        anyhow::ensure!(checked.is_file(), "git_log path must be a file");
        args.push("--".to_owned());
        args.push(relative_display(workspace, &checked));
    }
    git_owned(context, plan, workspace, arguments, &args).await
}

async fn git_show(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let revision = arguments
        .get("revision")
        .and_then(Value::as_str)
        .context("revision is required")?
        .trim();
    anyhow::ensure!(!revision.is_empty(), "revision must not be empty");
    anyhow::ensure!(
        !revision.starts_with('-'),
        "revision must not start with a dash"
    );
    let mut args = vec![
        "--no-pager".to_owned(),
        "show".to_owned(),
        "--no-ext-diff".to_owned(),
        "--no-color".to_owned(),
        revision.to_owned(),
    ];
    if arguments
        .get("stat_only")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        args.push("--stat".to_owned());
        args.push("--oneline".to_owned());
    }
    if arguments.get("path").is_some() {
        let checked = checked_path(context, plan, workspace, arguments, "path", false)?;
        args.push("--".to_owned());
        args.push(relative_display(workspace, &checked));
    }
    git_owned(context, plan, workspace, arguments, &args).await
}

async fn git_blame(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let checked = checked_path(context, plan, workspace, arguments, "path", false)?;
    anyhow::ensure!(checked.is_file(), "git_blame path must be a file");
    let mut args = vec![
        "--no-pager".to_owned(),
        "blame".to_owned(),
        "--line-porcelain".to_owned(),
    ];
    if let Some(start) = arguments.get("start_line").and_then(Value::as_u64) {
        let end = arguments
            .get("end_line")
            .and_then(Value::as_u64)
            .unwrap_or(start)
            .max(start);
        args.push("-L".to_owned());
        args.push(format!("{start},{end}"));
    }
    args.push("--".to_owned());
    args.push(relative_display(workspace, &checked));
    git_owned(context, plan, workspace, arguments, &args).await
}

async fn git_grep(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let query = arguments
        .get("query")
        .and_then(Value::as_str)
        .context("query is required")?;
    anyhow::ensure!(!query.is_empty(), "query must not be empty");
    let max_count = arguments
        .get("max_count")
        .and_then(Value::as_u64)
        .unwrap_or(20)
        .clamp(1, 100);
    let mut args = vec![
        "--no-pager".to_owned(),
        "grep".to_owned(),
        "-n".to_owned(),
        "-I".to_owned(),
        format!("-m{max_count}"),
        "-e".to_owned(),
        query.to_owned(),
        "--".to_owned(),
    ];
    if arguments.get("path").is_some() {
        let checked = checked_path(context, plan, workspace, arguments, "path", false)?;
        args.push(relative_display(workspace, &checked));
    } else {
        args.push(".".to_owned());
    }
    git_owned(context, plan, workspace, arguments, &args).await
}

async fn git_tags(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let max_count = arguments
        .get("max_count")
        .and_then(Value::as_u64)
        .unwrap_or(50)
        .clamp(1, 200);
    let args = vec![
        "--no-pager".to_owned(),
        "for-each-ref".to_owned(),
        format!("--count={max_count}"),
        "--sort=-creatordate".to_owned(),
        "--format=%(refname:short) %(objectname:short) %(creatordate:short) %(subject)".to_owned(),
        "refs/tags".to_owned(),
    ];
    git_owned(context, plan, workspace, arguments, &args).await
}

async fn git_diff_check(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let mut args = vec![
        "--no-pager".to_owned(),
        "diff".to_owned(),
        "--check".to_owned(),
    ];
    if arguments
        .get("staged")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        args.push("--cached".to_owned());
    }
    git_owned(context, plan, workspace, arguments, &args).await
}

async fn git_remote(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let mut outcome = git(
        context,
        plan,
        workspace,
        arguments,
        &["--no-pager", "remote", "-v"],
    )
    .await?;
    let output = std::mem::take(&mut outcome.output);
    outcome.output = output
        .lines()
        .map(redact_remote_credentials)
        .collect::<Vec<_>>()
        .join("\n");
    Ok(outcome)
}

fn redact_remote_credentials(line: &str) -> String {
    let mut redacted = line.to_owned();
    let mut cursor = 0;
    while let Some(relative_scheme) = redacted[cursor..].find("://") {
        let authority_start = cursor + relative_scheme + 3;
        let Some(relative_at) = redacted[authority_start..].find('@') else {
            break;
        };
        let at = authority_start + relative_at;
        redacted.replace_range(authority_start..=at, "***@");
        cursor = authority_start + 4;
    }
    redacted
}

/// Single-quote an argument for `sh -c`. Used only for arguments the daemon
/// itself supplies to git.
fn shell_quote(argument: &str) -> String {
    format!("'{}'", argument.replace('\'', "'\\''"))
}

async fn project_check(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    tool: &str,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let cwd = command_cwd(context, plan, workspace, arguments)?;
    let command = project_command(&cwd, tool).await?;
    let mut merged = arguments.clone();
    if let Value::Object(map) = &mut merged {
        map.insert("command".to_owned(), Value::String(command.clone()));
    }
    shell_run(context, plan, workspace, &merged, None)
        .await
        .with_context(|| format!("run {tool} command {command}"))
}

async fn project_command(cwd: &Path, tool: &str) -> anyhow::Result<String> {
    let has = |name: &str| cwd.join(name).is_file();
    if has("Cargo.toml") {
        return Ok(match tool {
            "project_test" => "cargo test --all-targets".to_owned(),
            "project_build" | "project_typecheck" => "cargo check --all-targets".to_owned(),
            "project_lint" => "cargo clippy --all-targets --all-features".to_owned(),
            "project_format" => "cargo fmt --all -- --check".to_owned(),
            _ => unreachable!("unknown project check"),
        });
    }
    if has("go.mod") {
        return Ok(match tool {
            "project_test" => "go test ./...".to_owned(),
            "project_build" | "project_typecheck" => "go build ./...".to_owned(),
            "project_lint" => "go vet ./...".to_owned(),
            "project_format" => "test -z \"$(gofmt -l .)\"".to_owned(),
            _ => unreachable!("unknown project check"),
        });
    }
    if has("package.json") {
        let contents = tokio::fs::read_to_string(cwd.join("package.json"))
            .await
            .context("read package.json")?;
        let package: Value = serde_json::from_str(&contents).context("parse package.json")?;
        let scripts = package.get("scripts").and_then(Value::as_object);
        let script = match tool {
            "project_format" if scripts.is_some_and(|s| s.contains_key("format:check")) => {
                Some("format:check")
            }
            "project_format" => None,
            "project_typecheck" => {
                scripts.and_then(|s| s.contains_key("typecheck").then_some("typecheck"))
            }
            "project_lint" => scripts.and_then(|s| s.contains_key("lint").then_some("lint")),
            "project_build" => scripts.and_then(|s| s.contains_key("build").then_some("build")),
            "project_test" => scripts.and_then(|s| s.contains_key("test").then_some("test")),
            _ => None,
        };
        let Some(script) = script else {
            anyhow::bail!(
                "package.json has no conventional script for {tool}; refusing to invent an install or build command"
            );
        };
        let runner = if has("pnpm-lock.yaml") {
            "pnpm"
        } else if has("yarn.lock") {
            "yarn"
        } else if has("bun.lockb") || has("bun.lock") {
            "bun"
        } else {
            "npm"
        };
        return Ok(if runner == "npm" {
            format!("npm run {script}")
        } else {
            format!("{runner} run {script}")
        });
    }
    if has("pyproject.toml") || has("pytest.ini") || has("setup.cfg") {
        return Ok(match tool {
            "project_test" => "python -m pytest".to_owned(),
            "project_build" => "python -m compileall -q .".to_owned(),
            "project_typecheck" => "pyright".to_owned(),
            "project_lint" => "ruff check .".to_owned(),
            "project_format" => "ruff format --check .".to_owned(),
            _ => unreachable!("unknown project check"),
        });
    }
    anyhow::bail!(
        "could not detect a Cargo, Go, Node, or Python project in {}",
        cwd.display()
    )
}

async fn env_info(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let cwd = command_cwd(context, plan, workspace, arguments)?;
    let mut manifests = Vec::new();
    for name in [
        "Cargo.toml",
        "package.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "go.mod",
        "pyproject.toml",
        "requirements.txt",
    ] {
        if cwd.join(name).is_file() {
            manifests.push(name);
        }
    }
    let mut command_arguments = arguments.clone();
    if let Value::Object(map) = &mut command_arguments {
        map.insert(
            "command".to_owned(),
            Value::String(
                "printf 'platform: '; uname -srm; printf 'shell: '; printf '%s\\n' \"$SHELL\"; \
                 git --version 2>/dev/null || true; cargo --version 2>/dev/null || true; \
                 node --version 2>/dev/null || true; python --version 2>/dev/null || true; \
                 go version 2>/dev/null || true"
                    .to_owned(),
            ),
        );
    }
    let versions = shell_run(context, plan, workspace, &command_arguments, None).await?;
    Ok(ToolOutcome::text(format!(
        "Workspace: {}\nManifests: {}\n\n{}",
        cwd.display(),
        if manifests.is_empty() {
            "(none)".to_owned()
        } else {
            manifests.join(", ")
        },
        versions.output
    )))
}

async fn process_list(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let mut command_arguments = arguments.clone();
    if let Value::Object(map) = &mut command_arguments {
        map.insert(
            "command".to_owned(),
            Value::String("ps -eo pid=,ppid=,pcpu=,pmem=,comm=,args=".to_owned()),
        );
        map.insert("timeout_ms".to_owned(), json!(10_000));
        map.insert(
            "cwd".to_owned(),
            Value::String(
                context
                    .session
                    .workspace_path
                    .clone()
                    .context("this session has no workspace")?,
            ),
        );
    }
    let output = shell_run(
        context,
        plan,
        context.session.workspace_path.as_deref().map(Path::new),
        &command_arguments,
        None,
    )
    .await?;
    let filter = arguments
        .get("match")
        .and_then(Value::as_str)
        .map(str::to_ascii_lowercase);
    if let Some(filter) = filter {
        let filtered = output
            .output
            .lines()
            .filter(|line| line.to_ascii_lowercase().contains(&filter))
            .collect::<Vec<_>>()
            .join("\n");
        return Ok(ToolOutcome::text(if filtered.is_empty() {
            "(no matching processes)".to_owned()
        } else {
            filtered
        }));
    }
    Ok(output)
}

async fn code_symbols(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let root = if arguments.get("path").is_some() {
        checked_path(context, plan, workspace, arguments, "path", false)?
    } else {
        workspace
            .context("this session has no workspace")?
            .to_path_buf()
    };
    let max_symbols = arguments
        .get("max_symbols")
        .and_then(Value::as_u64)
        .unwrap_or(500)
        .clamp(1, 1_000) as usize;
    let name_glob = arguments.get("name_glob").and_then(Value::as_str);
    let mut files = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(path) = stack.pop() {
        let metadata = tokio::fs::symlink_metadata(&path).await?;
        if metadata.is_dir() {
            let name = path.file_name().and_then(|v| v.to_str()).unwrap_or("");
            if SKIPPED_DIRECTORIES.contains(&name) {
                continue;
            }
            let mut entries = tokio::fs::read_dir(&path).await?;
            while let Some(entry) = entries.next_entry().await? {
                stack.push(entry.path());
            }
        } else if metadata.is_file() {
            let name = path.file_name().and_then(|v| v.to_str()).unwrap_or("");
            if name_glob
                .map(|pattern| glob_match(pattern, name))
                .unwrap_or(true)
            {
                files.push(path);
            }
        }
    }
    files.sort();
    let mut symbols = Vec::new();
    for path in files {
        let bytes = tokio::fs::read(&path).await?;
        if bytes.len() > 2 * 1024 * 1024 || bytes.contains(&0) {
            continue;
        }
        let extension = path.extension().and_then(|v| v.to_str()).unwrap_or("");
        if !matches!(
            extension,
            "rs" | "ts" | "tsx" | "js" | "jsx" | "py" | "go" | "java" | "c" | "cpp" | "h"
        ) {
            continue;
        }
        for (line_number, line) in String::from_utf8_lossy(&bytes).lines().enumerate() {
            let trimmed = line.trim_start();
            let declaration = trimmed
                .strip_prefix("pub ")
                .or_else(|| trimmed.strip_prefix("export "))
                .or_else(|| trimmed.strip_prefix("async "))
                .unwrap_or(trimmed);
            let kind = [
                ("fn ", "function"),
                ("function ", "function"),
                ("def ", "function"),
                ("func ", "function"),
                ("class ", "class"),
                ("struct ", "struct"),
                ("enum ", "enum"),
                ("trait ", "trait"),
                ("interface ", "interface"),
                ("type ", "type"),
                ("const ", "constant"),
                ("mod ", "module"),
            ]
            .iter()
            .find_map(|(prefix, kind)| declaration.strip_prefix(prefix).map(|rest| (*kind, rest)));
            let Some((kind, rest)) = kind else {
                continue;
            };
            let name = rest
                .split(|character: char| {
                    matches!(character, '(' | '<' | ':' | '=' | '{' | ' ' | '\t')
                })
                .next()
                .unwrap_or("")
                .trim();
            if name.is_empty() {
                continue;
            }
            symbols.push(format!(
                "{}:{}: {kind} {name}",
                relative_display(workspace, &path),
                line_number + 1
            ));
            if symbols.len() >= max_symbols {
                symbols.push(format!("[... stopped at {max_symbols} symbols ...]"));
                return Ok(ToolOutcome::text(symbols.join("\n")));
            }
        }
    }
    if symbols.is_empty() {
        return Ok(ToolOutcome::text("(no symbols found)"));
    }
    Ok(ToolOutcome::text(symbols.join("\n")))
}

// ---------------------------------------------------------------------------
// Power tools (Powerful mode): web search, web fetch, and LSP diagnostics.
// ---------------------------------------------------------------------------

const WEB_REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

fn web_http_client() -> anyhow::Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(brazier_runtime::web::WEB_USER_AGENT)
        .redirect(reqwest::redirect::Policy::limited(5))
        .timeout(WEB_REQUEST_TIMEOUT)
        .build()
        .context("build HTTP client")
}

async fn web_search(
    context: &BrokerContext<'_>,
    _workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let query = arguments
        .get("query")
        .and_then(Value::as_str)
        .context("`query` is required")?
        .trim();
    anyhow::ensure!(!query.is_empty(), "`query` must not be empty");
    let max_results = arguments
        .get("max_results")
        .and_then(Value::as_u64)
        .unwrap_or(5)
        .clamp(1, 10) as usize;
    let region = arguments.get("region").and_then(Value::as_str);
    let safesearch = arguments.get("safesearch").and_then(Value::as_str);
    let settings = brazier_runtime::runtime_settings::load(context.data_dir);
    let results = brazier_runtime::web::search(
        &web_http_client()?,
        query,
        max_results,
        region,
        safesearch,
        &settings,
    )
    .await?;
    Ok(ToolOutcome::text(brazier_runtime::web::format_results(
        &results,
    )))
}

async fn web_fetch(
    context: &BrokerContext<'_>,
    _workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let raw_url = arguments
        .get("url")
        .and_then(Value::as_str)
        .context("`url` is required")?
        .trim();
    let url = reqwest::Url::parse(raw_url).context("`url` is not a valid URL")?;
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https"),
        "only http(s) URLs can be fetched"
    );
    let max_chars = arguments
        .get("max_chars")
        .and_then(Value::as_u64)
        .unwrap_or(12_000)
        .clamp(500, 50_000) as usize;
    let start = arguments.get("start").and_then(Value::as_u64).unwrap_or(0) as usize;
    // Same guarded downloader as the chat `fetch_url` tool: private and local
    // addresses are refused, every redirect hop is re-vetted, and the shared
    // fetch rate budget is consumed here too.
    let download = brazier_runtime::web::download_url(raw_url).await?;
    if brazier_runtime::web::downloaded_is_pdf(&download) {
        return web_fetch_pdf(context, download).await;
    }
    Ok(ToolOutcome::text(brazier_runtime::web::fetch_content_text(
        &download, start, max_chars,
    )?))
}

/// Store a fetched PDF as a blob and hand the model a `doc_read` document id
/// plus its page count for paging — the same attach-and-read flow as the chat
/// `fetch_url` tool. The contents are deliberately not dumped here; the model
/// chooses a page range with `doc_read`.
async fn web_fetch_pdf(
    context: &BrokerContext<'_>,
    download: brazier_runtime::web::DownloadedUrl,
) -> anyhow::Result<ToolOutcome> {
    let name = download
        .final_url
        .path_segments()
        .and_then(|mut parts| parts.next_back())
        .filter(|part| !part.is_empty())
        .unwrap_or("download.pdf");
    let blob = brazier_runtime::blob_store::store_bytes(
        context.data_dir,
        &download.bytes,
        "application/pdf",
        Some(name),
    )
    .await
    .context("store fetched PDF")?;
    let path = brazier_runtime::blob_store::blob_path(context.data_dir, &blob.sha256)?;
    let pages = if brazier_runtime::documents::missing_poppler_tools().is_empty() {
        brazier_runtime::documents::page_count(&path)
            .await
            .unwrap_or(None)
    } else {
        None
    };
    let document_id = format!("brazier_blob:{}", blob.sha256);
    let mut text = format!("Fetched PDF {name}. It was stored for doc_read.");
    if let Some(count) = pages {
        text.push_str(&format!(" It is {count} pages long."));
    }
    text.push_str(&format!(
        " Its contents are not included here. Use the doc_read tool with document \
         `{document_id}` to read it, choosing a page range{}. If it is a scan with no text \
         layer, set render_pages to receive page images.",
        if pages.is_some() {
            " from the page count"
        } else {
            " (page count could not be determined)"
        }
    ));
    Ok(ToolOutcome::text(text))
}

const LSP_DIAGNOSTIC_TIMEOUT: Duration = Duration::from_secs(12);
const LSP_MAX_DIAGNOSTICS: usize = 200;

/// Which language server to run for a file extension:
/// `(binary, extra args, languageId)`.
fn lsp_server_for(
    extension: &str,
) -> Option<(&'static str, &'static [&'static str], &'static str)> {
    match extension {
        "ts" | "tsx" => Some(("typescript-language-server", &["--stdio"][..], "typescript")),
        "js" | "jsx" | "mjs" | "cjs" => {
            Some(("typescript-language-server", &["--stdio"][..], "javascript"))
        }
        "py" => Some(("pylsp", &[][..], "python")),
        "rs" => Some(("rust-analyzer", &[][..], "rust")),
        "c" | "h" => Some(("clangd", &[][..], "c")),
        "cpp" | "hpp" | "cc" | "cxx" => Some(("clangd", &[][..], "cpp")),
        "go" => Some(("gopls", &[][..], "go")),
        "json" => Some(("vscode-json-languageserver", &["--stdio"][..], "json")),
        "yaml" | "yml" => Some(("yaml-language-server", &["--stdio"][..], "yaml")),
        _ => None,
    }
}

fn find_on_path(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|directory| directory.join(program))
        .find(|candidate| candidate.is_file())
}

fn path_uri(root: &Path, path: &Path) -> String {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let normalized = absolute.to_string_lossy().replace('\\', "/");
    if normalized.starts_with('/') {
        format!("file://{normalized}")
    } else {
        format!("file:///{normalized}")
    }
}

async fn lsp_send_frame<W>(writer: &mut W, message: &Value) -> anyhow::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let body = serde_json::to_vec(message)?;
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    writer.write_all(header.as_bytes()).await?;
    writer.write_all(&body).await?;
    writer.flush().await?;
    Ok(())
}

async fn lsp_read_frame<R>(reader: &mut R) -> anyhow::Result<Option<Value>>
where
    R: AsyncBufRead + Unpin,
{
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        match tokio::time::timeout(Duration::from_secs(5), reader.read_line(&mut line)).await {
            Ok(Ok(0)) => return Ok(None),
            Ok(Ok(_)) => {}
            Ok(Err(error)) => return Err(error.into()),
            // A quiet server is not a failure; the caller's overall deadline
            // decides how long to keep waiting.
            Err(_) => return Ok(None),
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some(value) = trimmed.strip_prefix("Content-Length:") {
            content_length = value.trim().parse::<usize>().ok();
        }
    }
    let length = content_length.context("language server frame had no Content-Length")?;
    let mut body = vec![0u8; length];
    tokio::time::timeout(Duration::from_secs(5), reader.read_exact(&mut body))
        .await
        .context("language server timed out sending a frame")??;
    Ok(serde_json::from_slice(&body).ok())
}

/// Run the workspace's language server over stdio and report diagnostics for a
/// file. The server runs on the host (see [`agent_policy::is_host_tool`]), so it
/// can reach user toolchain installs; the call itself is approval-gated.
async fn lsp_diagnostics(
    context: &BrokerContext<'_>,
    plan: &CallPlan,
    workspace: Option<&Path>,
    arguments: &Value,
) -> anyhow::Result<ToolOutcome> {
    let checked = checked_path(context, plan, workspace, arguments, "path", false)?;
    let workspace_root = workspace.context("this session has no workspace")?;
    let label = relative_display(workspace, &checked);
    let text = tokio::fs::read_to_string(&checked)
        .await
        .with_context(|| format!("cannot read `{label}`"))?;
    let extension = checked
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let Some((binary, args, language_id)) = lsp_server_for(&extension) else {
        return Ok(ToolOutcome::text(format!(
            "No language server is configured for `.{extension}` files. Supported extensions: \
             ts, tsx, js, jsx, mjs, cjs, py, rs, c, h, cpp, hpp, cc, go, json, yaml, yml."
        )));
    };
    let Some(binary_path) = find_on_path(binary) else {
        return Ok(ToolOutcome::text(format!(
            "The `{binary}` language server is not installed, so `{label}` cannot be analyzed. \
             Install it and try again."
        )));
    };
    let include_warnings = arguments
        .get("include_warnings")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let mut command_line = shell_quote(&binary_path.display().to_string());
    for argument in args {
        command_line.push(' ');
        command_line.push_str(&shell_quote(argument));
    }
    let (mut command, _) = build_command(context, plan, workspace, arguments, &command_line)
        .await
        .with_context(|| format!("cannot prepare `{binary}`"))?;
    let mut child = command
        .spawn()
        .with_context(|| format!("cannot start `{binary}`"))?;
    let mut stdin = child.stdin.take().context("language server stdin")?;
    let stdout = child.stdout.take().context("language server stdout")?;
    let mut reader = BufReader::new(stdout);

    let uri = path_uri(workspace_root, &checked);
    let root_uri = path_uri(workspace_root, workspace_root);
    let collected = lsp_collect(&mut stdin, &mut reader, &uri, &root_uri, language_id, &text).await;
    let _ = child.start_kill();
    let _ = child.wait().await;

    let diagnostics = match collected {
        Ok(items) => items,
        Err(error) => {
            return Ok(ToolOutcome::text(format!("`{binary}` reported: {error:#}")));
        }
    };
    if diagnostics.is_empty() {
        return Ok(ToolOutcome::text(format!(
            "`{binary}` reported no diagnostics for `{label}`."
        )));
    }
    let mut output = String::new();
    let mut shown = 0usize;
    let mut total = 0usize;
    for item in diagnostics.iter() {
        total += 1;
        if total > LSP_MAX_DIAGNOSTICS {
            break;
        }
        let message = item.get("message").and_then(Value::as_str).unwrap_or("");
        let message = message.trim();
        if message.is_empty() {
            continue;
        }
        let severity = item.get("severity").and_then(Value::as_u64).unwrap_or(0);
        // 1=error, 2=warning, 3=info, 4=hint.
        if !include_warnings && severity >= 2 {
            continue;
        }
        if severity >= 3 {
            continue;
        }
        let level = if severity == 1 { "error" } else { "warning" };
        let location = item
            .get("range")
            .and_then(|range| range.get("start"))
            .map(|start| {
                let line = start.get("line").and_then(Value::as_u64).unwrap_or(0) + 1;
                let column = start.get("character").and_then(Value::as_u64).unwrap_or(0) + 1;
                format!("{label}:{line}:{column}")
            })
            .unwrap_or_else(|| label.clone());
        let code = item
            .get("code")
            .and_then(Value::as_str)
            .map(|code| format!(" [{code}]"))
            .unwrap_or_default();
        output.push_str(&format!("{level}: {location}{code}: {message}\n"));
        shown += 1;
    }
    if shown == 0 {
        return Ok(ToolOutcome::text(format!(
            "`{binary}` reported no {}-level diagnostics for `{label}`.",
            if include_warnings {
                "error or warning"
            } else {
                "error"
            }
        )));
    }
    output.push_str(&format!("\n{shown} diagnostic(s) for `{label}`."));
    Ok(ToolOutcome::text(output))
}

/// Drive a language server over stdio: initialize, open the file, collect
/// `textDocument/publishDiagnostics` until the first batch for `uri` or the
/// timeout, then shut the server down. Reused by the tool and by tests.
async fn lsp_collect<W, R>(
    stdin: &mut W,
    reader: &mut R,
    uri: &str,
    root_uri: &str,
    language_id: &str,
    text: &str,
) -> anyhow::Result<Vec<Value>>
where
    W: AsyncWrite + Unpin,
    R: AsyncBufRead + Unpin,
{
    lsp_send_frame(
        stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "processId": null,
                "rootUri": root_uri,
                "workspaceFolders": [{ "uri": root_uri, "name": "workspace" }],
                "capabilities": {}
            }
        }),
    )
    .await?;
    let mut initialized = false;
    let mut diagnostics: Vec<Value> = Vec::new();
    let deadline = tokio::time::Instant::now() + LSP_DIAGNOSTIC_TIMEOUT;
    loop {
        if tokio::time::Instant::now() >= deadline {
            break;
        }
        let remaining = deadline - tokio::time::Instant::now();
        let frame = tokio::time::timeout(remaining, lsp_read_frame(reader))
            .await
            .context("language server did not answer in time")??;
        let Some(frame) = frame else { break };
        let method = frame.get("method").and_then(Value::as_str).unwrap_or("");
        if method == "textDocument/publishDiagnostics" {
            let params = frame.get("params");
            if params
                .and_then(|params| params.get("uri"))
                .and_then(Value::as_str)
                .is_some_and(|frame_uri| frame_uri == uri)
                && let Some(items) = params
                    .and_then(|params| params.get("diagnostics"))
                    .and_then(Value::as_array)
            {
                diagnostics.extend(items.iter().cloned());
            }
            if !diagnostics.is_empty() {
                break;
            }
            continue;
        }
        if !initialized && frame.get("id").and_then(Value::as_u64) == Some(1) {
            lsp_send_frame(
                stdin,
                &json!({ "jsonrpc": "2.0", "method": "initialized", "params": {} }),
            )
            .await?;
            lsp_send_frame(
                stdin,
                &json!({
                    "jsonrpc": "2.0",
                    "method": "textDocument/didOpen",
                    "params": {
                        "textDocument": {
                            "uri": uri,
                            "languageId": language_id,
                            "version": 1,
                            "text": text
                        }
                    }
                }),
            )
            .await?;
            initialized = true;
        }
    }
    let _ = lsp_send_frame(
        stdin,
        &json!({ "jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": null }),
    )
    .await;
    let _ = lsp_send_frame(
        stdin,
        &json!({ "jsonrpc": "2.0", "method": "exit", "params": null }),
    )
    .await;
    Ok(diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use brazier_protocol::agent_types::{
        AgentPermissionMode, AgentPermissionSettings, CreateAgentSession,
    };
    use brazier_storage::db::Database;
    use serde_json::json;
    use tempfile::TempDir;

    #[cfg(windows)]
    #[test]
    fn windows_path_validation_accepts_canonical_drive_and_unc_paths() {
        for path in [
            r"C:\workspace\file.txt",
            r"\\server\share\workspace\file.txt",
            r"\\?\C:\workspace\file.txt",
            r"\\?\UNC\server\share\workspace\file.txt",
        ] {
            validate_platform_path(Path::new(path)).unwrap_or_else(|error| {
                panic!("expected canonical Windows path `{path}` to be valid: {error}")
            });
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_path_validation_rejects_ambiguous_and_device_paths() {
        for path in [
            r"C:drive-relative.txt",
            r"C:\workspace\file.txt:secret-stream",
            r"C:\workspace\NUL.txt",
            r"C:\workspace\COM1",
            r"C:\workspace\trailing-dot.",
            r"\\.\PhysicalDrive0",
            r"\\?\GLOBALROOT\Device\HarddiskVolume1\secret.txt",
        ] {
            assert!(
                validate_platform_path(Path::new(path)).is_err(),
                "ambiguous or device path `{path}` must be rejected"
            );
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_safe_open_refuses_a_final_file_reparse_point_when_available() {
        let directory = TempDir::new().expect("temp dir");
        let target = directory.path().join("target.txt");
        let link = directory.path().join("link.txt");
        std::fs::write(&target, "secret").expect("write target");
        if let Err(error) = std::os::windows::fs::symlink_file(&target, &link) {
            if error.kind() == std::io::ErrorKind::PermissionDenied {
                // Developer Mode or SeCreateSymbolicLinkPrivilege is not
                // guaranteed on downstream Windows test hosts.
                return;
            }
            panic!("create file symlink: {error}");
        }

        let error = open_read_without_final_link(&link)
            .expect_err("the final reparse point must not be followed");
        assert!(error.to_string().contains("reparse point"), "{error}");
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_junction_out_of_the_workspace_does_not_widen_access() {
        let harness = Harness::new(AgentPermissionMode::SandboxOnly).await;
        let outside = TempDir::new().expect("outside dir");
        std::fs::write(outside.path().join("secret.txt"), "junction-secret").expect("write secret");
        let junction = harness.workspace.path().join("escape");
        let status = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(&junction)
            .arg(outside.path())
            .status()
            .expect("run mklink");
        assert!(status.success(), "mklink /J must create the test junction");

        let response = harness
            .call("fs_read", json!({ "path": "escape/secret.txt" }))
            .await;
        std::fs::remove_dir(&junction).expect("remove junction");
        assert_eq!(
            response.status,
            ToolExecStatus::Denied,
            "{}",
            response.output
        );
        assert!(!response.output.contains("junction-secret"));
    }

    struct Harness {
        _data: TempDir,
        workspace: TempDir,
        db: Database,
        broker: AgentBroker,
        session: AgentSessionRecord,
        data_dir: PathBuf,
    }

    impl Harness {
        async fn new(mode: AgentPermissionMode) -> Self {
            Self::new_with(mode, false).await
        }

        /// A session that auto-approves host actions, so host-routed tools
        /// (the Powerful mode tools) run end to end in tests.
        async fn new_host(mode: AgentPermissionMode) -> Self {
            Self::new_with(mode, true).await
        }

        async fn new_with(mode: AgentPermissionMode, auto_approve_host_actions: bool) -> Self {
            let data = TempDir::new().expect("data dir");
            let workspace = TempDir::new().expect("workspace");
            let db = Database::open(&data.path().join("brazier.sqlite"))
                .await
                .expect("database");
            let session = db
                .create_agent_session(CreateAgentSession {
                    title: Some("test".to_owned()),
                    workspace_path: Some(workspace.path().display().to_string()),
                    model: "gguf:test".to_owned(),
                    runtime_id: Some("simple".to_owned()),
                    permission_mode: Some(mode),
                    permission_settings: Some(AgentPermissionSettings {
                        auto_approve_sandboxed_actions: true,
                        auto_approve_host_actions,
                    }),
                    enabled_tools: None,
                    confine_to_worktree: false,
                    confirm_elevated_permissions: matches!(
                        mode,
                        AgentPermissionMode::SkipPermissions
                    ),
                })
                .await
                .expect("session");
            let data_dir = data.path().to_path_buf();
            Self {
                _data: data,
                workspace,
                db,
                broker: AgentBroker::new(),
                session,
                data_dir,
            }
        }

        fn context(&self) -> BrokerContext<'_> {
            BrokerContext {
                broker: &self.broker,
                db: &self.db,
                data_dir: &self.data_dir,
                session: &self.session,
            }
        }

        async fn call(&self, tool: &str, arguments: Value) -> ToolExecResponse {
            let request = ToolExecRequest {
                session_id: self.session.id.clone(),
                run_id: Some("run-1".to_owned()),
                tool_call_id: Some("call-1".to_owned()),
                tool: tool.to_owned(),
                arguments,
                environment: None,
                reason: None,
                approval_id: None,
            };
            execute(&self.context(), &request)
                .await
                .expect("broker call")
        }
    }

    #[tokio::test]
    async fn writes_and_reads_round_trip_inside_the_workspace() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        let write = harness
            .call(
                "fs_write",
                json!({ "path": "notes/todo.txt", "content": "first line\nsecond line\n" }),
            )
            .await;
        assert_eq!(write.status, ToolExecStatus::Completed, "{}", write.output);
        assert_eq!(write.changed_paths, vec!["notes/todo.txt".to_owned()]);

        let read = harness
            .call("fs_read", json!({ "path": "notes/todo.txt" }))
            .await;
        assert!(read.output.contains("first line"));
        assert!(read.output.contains("     2\t"), "lines are numbered");

        let patch = harness
            .call(
                "fs_patch",
                json!({
                    "path": "notes/todo.txt",
                    "old_string": "second line",
                    "new_string": "changed line"
                }),
            )
            .await;
        assert_eq!(patch.status, ToolExecStatus::Completed, "{}", patch.output);
        let after = tokio::fs::read_to_string(harness.workspace.path().join("notes/todo.txt"))
            .await
            .expect("read file");
        assert!(after.contains("changed line"));
    }

    #[tokio::test]
    async fn doc_read_extracts_rtf_by_line_range() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        let rtf = br"{\rtf1\ansi Alpha\par Beta\par Gamma\par}";
        tokio::fs::write(harness.workspace.path().join("letter.rtf"), rtf)
            .await
            .expect("write rtf");

        let response = harness
            .call(
                "doc_read",
                json!({ "path": "letter.rtf", "start_line": 1, "end_line": 2 }),
            )
            .await;
        assert_eq!(
            response.status,
            ToolExecStatus::Completed,
            "{}",
            response.output
        );
        assert!(response.output.contains("Alpha"), "{}", response.output);
        assert!(response.output.contains("Beta"), "{}", response.output);
        assert!(response.output.contains("lines 1–2"), "{}", response.output);
        assert!(!response.output.contains("Gamma"), "{}", response.output);
    }

    /// A tiny single-page PDF with a known line, so the document-id path can
    /// run end to end when poppler is installed.
    const MINIMAL_PDF: &[u8] = br#"%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 48 >>
stream
BT /F1 24 Tf 72 720 Td (Hello from doc_read) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
trailer
<< /Size 6 /Root 1 0 R >>
startxref
455
%%EOF
"#;

    #[tokio::test]
    async fn doc_read_resolves_web_fetched_document_ids() {
        if !brazier_runtime::documents::missing_poppler_tools().is_empty() {
            return; // pdftotext not installed; the sniffing still works
        }
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        // A blob stored without an extension, exactly what web_fetch produces.
        let blob = brazier_runtime::blob_store::store_bytes(
            &harness.data_dir,
            MINIMAL_PDF,
            "application/pdf",
            Some("paper.pdf"),
        )
        .await
        .expect("store pdf");

        let response = harness
            .call(
                "doc_read",
                json!({ "document": format!("brazier_blob:{}", blob.sha256) }),
            )
            .await;
        assert_eq!(
            response.status,
            ToolExecStatus::Completed,
            "{}",
            response.output
        );
        assert!(
            response.output.contains("Hello from doc_read"),
            "{}",
            response.output
        );
    }

    #[tokio::test]
    async fn doc_read_document_id_refuses_non_pdf_blobs() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        let blob = brazier_runtime::blob_store::store_bytes(
            &harness.data_dir,
            br"{\rtf1\ansi Alpha\par}",
            "application/rtf",
            Some("letter.rtf"),
        )
        .await
        .expect("store rtf");

        let response = harness
            .call("doc_read", json!({ "document": blob.sha256 }))
            .await;
        assert!(response.is_error, "{}", response.output);
        assert!(response.output.contains("not a PDF"), "{}", response.output);
    }

    #[tokio::test]
    async fn doc_read_document_id_rejects_unknown_or_missing_blobs() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        let invalid = harness
            .call("doc_read", json!({ "document": "not-a-sha" }))
            .await;
        assert!(invalid.is_error, "{}", invalid.output);
        assert!(
            invalid.output.contains("invalid `document` id"),
            "{}",
            invalid.output
        );

        let missing = harness
            .call("doc_read", json!({ "document": "ab".repeat(32) }))
            .await;
        assert!(missing.is_error, "{}", missing.output);
        assert!(
            missing.output.contains("no longer stored locally"),
            "{}",
            missing.output
        );
    }

    #[tokio::test]
    async fn ambiguous_patches_are_refused() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        harness
            .call("fs_write", json!({ "path": "a.txt", "content": "x\nx\n" }))
            .await;
        let patch = harness
            .call(
                "fs_patch",
                json!({ "path": "a.txt", "old_string": "x", "new_string": "y" }),
            )
            .await;
        assert!(patch.is_error);
        assert!(patch.output.contains("appears 2 times"), "{}", patch.output);
    }

    #[tokio::test]
    async fn paths_outside_the_workspace_are_refused_in_the_sandbox() {
        let harness = Harness::new(AgentPermissionMode::SandboxOnly).await;
        let response = harness
            .call("fs_read", json!({ "path": "/etc/hosts" }))
            .await;
        assert_eq!(response.status, ToolExecStatus::Denied);
        assert!(response.is_error);
    }

    #[tokio::test]
    async fn a_symlink_out_of_the_workspace_does_not_widen_access() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        let outside = TempDir::new().expect("outside dir");
        let secret = outside.path().join("secret.txt");
        tokio::fs::write(&secret, "classified")
            .await
            .expect("write");
        let link = harness.workspace.path().join("link.txt");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&secret, &link).expect("symlink");
        #[cfg(not(unix))]
        return;

        // Following the link leaves the workspace, so it is treated as host
        // access and held for approval instead of being read.
        let response = harness.call("fs_read", json!({ "path": "link.txt" })).await;
        assert_eq!(
            response.status,
            ToolExecStatus::ApprovalRequired,
            "symlink escape must not read through: {}",
            response.output
        );
        assert!(!response.output.contains("classified"));
        let approval = response.approval.expect("approval");
        assert_eq!(approval.environment, AgentEnvironment::Host);
        assert_eq!(
            approval.elevation.requested_filesystem_paths.len(),
            1,
            "the target outside the workspace is named in the request"
        );
        let _ = link;
    }

    #[tokio::test]
    async fn fs_search_does_not_follow_symlinks_out_of_the_workspace() {
        let harness = Harness::new(AgentPermissionMode::SandboxOnly).await;
        let outside = TempDir::new().expect("outside dir");
        let secret = outside.path().join("secret.txt");
        tokio::fs::write(&secret, "classified-search-marker")
            .await
            .expect("write");
        let link = harness.workspace.path().join("escape.txt");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&secret, &link).expect("symlink");
        #[cfg(not(unix))]
        return;

        let response = harness
            .call("fs_search", json!({ "query": "classified-search-marker" }))
            .await;
        assert_eq!(
            response.status,
            ToolExecStatus::Completed,
            "{}",
            response.output
        );
        assert!(
            response.output.starts_with("No matches for"),
            "fs_search must not read through an escaping symlink: {}",
            response.output
        );
        assert!(
            !response.output.contains("escape.txt"),
            "fs_search must not report a hit via an escaping symlink: {}",
            response.output
        );
    }

    #[tokio::test]
    async fn fs_list_names_but_does_not_descend_escaping_symlinks() {
        let harness = Harness::new(AgentPermissionMode::SandboxOnly).await;
        let outside = TempDir::new().expect("outside dir");
        let nested = outside.path().join("nested");
        tokio::fs::create_dir_all(&nested).await.expect("mkdir");
        tokio::fs::write(nested.join("secret.txt"), "listed-escape-marker")
            .await
            .expect("write");
        let link = harness.workspace.path().join("out");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&nested, &link).expect("symlink");
        #[cfg(not(unix))]
        return;

        let response = harness
            .call("fs_list", json!({ "path": ".", "depth": 3 }))
            .await;
        assert_eq!(
            response.status,
            ToolExecStatus::Completed,
            "{}",
            response.output
        );
        assert!(
            response.output.contains("symlink outside workspace"),
            "escaping symlink should be named without following: {}",
            response.output
        );
        assert!(
            !response.output.contains("secret.txt")
                && !response.output.contains("listed-escape-marker"),
            "fs_list must not descend an escaping directory symlink: {}",
            response.output
        );
    }

    #[tokio::test]
    async fn fs_search_follows_symlinks_that_stay_in_the_workspace() {
        let harness = Harness::new(AgentPermissionMode::SandboxOnly).await;
        let target_dir = harness.workspace.path().join("real");
        tokio::fs::create_dir_all(&target_dir).await.expect("mkdir");
        tokio::fs::write(target_dir.join("inside.txt"), "workspace-link-marker")
            .await
            .expect("write");
        let link = harness.workspace.path().join("alias");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target_dir, &link).expect("symlink");
        #[cfg(not(unix))]
        return;

        let response = harness
            .call("fs_search", json!({ "query": "workspace-link-marker" }))
            .await;
        assert_eq!(
            response.status,
            ToolExecStatus::Completed,
            "{}",
            response.output
        );
        assert!(
            response.output.contains("workspace-link-marker"),
            "in-workspace symlinks should still be searchable: {}",
            response.output
        );
    }

    #[tokio::test]
    async fn ask_mode_requires_approval_for_writes_and_then_honours_it() {
        let harness = Harness::new(AgentPermissionMode::Ask).await;
        let first = harness
            .call("fs_write", json!({ "path": "x.txt", "content": "hi" }))
            .await;
        assert_eq!(first.status, ToolExecStatus::ApprovalRequired);
        let approval = first.approval.expect("approval");
        assert!(!harness.workspace.path().join("x.txt").exists());

        harness
            .db
            .decide_approval(
                &approval.id,
                true,
                Some(ApprovalScope::Once),
                None,
                Some("test-client"),
            )
            .await
            .expect("approve");

        let request = ToolExecRequest {
            session_id: harness.session.id.clone(),
            run_id: None,
            tool_call_id: None,
            tool: "fs_write".to_owned(),
            arguments: json!({ "path": "x.txt", "content": "hi" }),
            environment: None,
            reason: None,
            approval_id: Some(approval.id.clone()),
        };
        let second = execute(&harness.context(), &request)
            .await
            .expect("execute");
        assert_eq!(
            second.status,
            ToolExecStatus::Completed,
            "{}",
            second.output
        );
        assert!(harness.workspace.path().join("x.txt").exists());

        // The one-shot grant is spent.
        let replay = execute(&harness.context(), &request)
            .await
            .expect("execute");
        assert_eq!(replay.status, ToolExecStatus::Denied);
        assert!(replay.output.contains("already used"));
    }

    #[tokio::test]
    async fn an_approval_cannot_be_spent_on_different_arguments() {
        let harness = Harness::new(AgentPermissionMode::Ask).await;
        let first = harness
            .call("fs_write", json!({ "path": "safe.txt", "content": "hi" }))
            .await;
        let approval = first.approval.expect("approval");
        harness
            .db
            .decide_approval(
                &approval.id,
                true,
                Some(ApprovalScope::Once),
                None,
                Some("test-client"),
            )
            .await
            .expect("approve");

        let request = ToolExecRequest {
            session_id: harness.session.id.clone(),
            run_id: None,
            tool_call_id: None,
            tool: "fs_write".to_owned(),
            arguments: json!({ "path": "other.txt", "content": "hi" }),
            environment: None,
            reason: None,
            approval_id: Some(approval.id.clone()),
        };
        let response = execute(&harness.context(), &request)
            .await
            .expect("execute");
        assert_eq!(response.status, ToolExecStatus::Denied);
        assert!(response.output.contains("different call"));
        assert!(!harness.workspace.path().join("other.txt").exists());
    }

    #[tokio::test]
    async fn denied_approvals_report_the_users_reason() {
        let harness = Harness::new(AgentPermissionMode::Ask).await;
        let first = harness
            .call("fs_delete", json!({ "path": "gone.txt" }))
            .await;
        let approval = first.approval.expect("approval");
        harness
            .db
            .decide_approval(
                &approval.id,
                false,
                None,
                Some("not that file".to_owned()),
                Some("test-client"),
            )
            .await
            .expect("deny");
        let request = ToolExecRequest {
            session_id: harness.session.id.clone(),
            run_id: None,
            tool_call_id: None,
            tool: "fs_delete".to_owned(),
            arguments: json!({ "path": "gone.txt" }),
            environment: None,
            reason: None,
            approval_id: Some(approval.id.clone()),
        };
        let response = execute(&harness.context(), &request)
            .await
            .expect("execute");
        assert_eq!(response.status, ToolExecStatus::Denied);
        assert!(response.output.contains("not that file"));
    }

    #[tokio::test]
    async fn deleting_the_workspace_root_is_refused() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        let response = harness
            .call(
                "fs_delete",
                json!({ "path": harness.workspace.path().display().to_string(), "recursive": true }),
            )
            .await;
        assert!(response.is_error, "{}", response.output);
        assert!(harness.workspace.path().exists());
    }

    #[tokio::test]
    async fn search_finds_matches_and_skips_heavy_directories() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        harness
            .call(
                "fs_write",
                json!({ "path": "src/lib.rs", "content": "fn marker() {}\n" }),
            )
            .await;
        harness
            .call(
                "fs_write",
                json!({ "path": "node_modules/pkg/index.js", "content": "marker\n" }),
            )
            .await;
        let response = harness
            .call("fs_search", json!({ "query": "marker" }))
            .await;
        assert!(
            response.output.contains("src/lib.rs:1"),
            "{}",
            response.output
        );
        assert!(
            !response.output.contains("node_modules"),
            "vendored directories are skipped: {}",
            response.output
        );
    }

    #[tokio::test]
    async fn search_can_filter_by_name() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        harness
            .call("fs_write", json!({ "path": "a.rs", "content": "needle\n" }))
            .await;
        harness
            .call(
                "fs_write",
                json!({ "path": "b.txt", "content": "needle\n" }),
            )
            .await;
        let response = harness
            .call(
                "fs_search",
                json!({ "query": "needle", "name_glob": "*.rs" }),
            )
            .await;
        assert!(response.output.contains("a.rs"));
        assert!(!response.output.contains("b.txt"));
    }

    #[tokio::test]
    async fn unknown_tools_are_refused_before_execution() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        let response = harness.call("exec_anything", json!({})).await;
        assert_eq!(response.status, ToolExecStatus::Denied);
    }

    #[tokio::test]
    async fn every_call_is_recorded_for_the_timeline() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        harness.call("workspace_info", json!({})).await;
        harness
            .call("fs_read", json!({ "path": "missing.txt" }))
            .await;
        let records = harness
            .db
            .list_tool_executions(&harness.session.id)
            .await
            .expect("records");
        assert_eq!(records.len(), 2);
        assert!(records.iter().any(|record| record.status == "failed"));
        assert!(records.iter().all(|record| record.sandbox.is_some()));
    }

    #[tokio::test]
    async fn large_output_is_truncated_and_stored_as_an_artifact() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        let big = "x".repeat(MAX_MODEL_OUTPUT_CHARS + 5_000);
        harness
            .call("fs_write", json!({ "path": "big.txt", "content": big }))
            .await;
        let response = harness.call("fs_read", json!({ "path": "big.txt" })).await;
        assert!(response.truncated, "output should be truncated");
        let artifact_id = response.artifact_id.expect("artifact id");
        let (session_id, path, size) = harness.db.artifact(&artifact_id).await.expect("artifact");
        assert_eq!(session_id, harness.session.id);
        assert!(size > MAX_MODEL_OUTPUT_CHARS as i64);
        assert!(PathBuf::from(path).exists());
        assert!(response.output.contains("characters truncated"));
    }

    #[cfg(unix)]
    fn require_usable_sandbox(broker: &AgentBroker) -> bool {
        if broker.backend().isolated() {
            return true;
        }
        eprintln!(
            "skipping sandboxed shell test: {}",
            broker.backend().capabilities().detail
        );
        false
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shell_run_captures_output_and_exit_code() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        if !require_usable_sandbox(&harness.broker) {
            return;
        }
        let response = harness
            .call("shell_run", json!({ "command": "echo agent-mode-ok" }))
            .await;
        assert!(
            response.output.contains("agent-mode-ok"),
            "unexpected output: {}",
            response.output
        );
        assert_eq!(response.exit_code, Some(0));

        let failure = harness
            .call("shell_run", json!({ "command": "exit 3" }))
            .await;
        assert_eq!(failure.exit_code, Some(3));
        assert!(failure.is_error);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shell_run_streams_output_before_returning_the_final_result() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        let context = harness.context();
        let workspace = workspace_root(&harness.session)
            .expect("workspace")
            .expect("workspace path");
        // This test verifies streaming, not Seatbelt itself. Host execution
        // avoids nesting macOS sandbox-exec under a sandboxed test runner.
        let plan = CallPlan {
            environment: AgentEnvironment::Host,
            profile: SandboxProfile::Workspace,
            sandbox: harness.broker.backend().describe_host(Some(&workspace)),
            approval_id: None,
        };
        let arguments = json!({
            "command": "printf first; sleep 0.2; printf second; printf problem >&2"
        });
        let (tx, mut rx) = mpsc::unbounded_channel();
        let execution = shell_run(&context, &plan, Some(&workspace), &arguments, Some(&tx));
        tokio::pin!(execution);

        let first = tokio::select! {
            chunk = rx.recv() => chunk.expect("first streamed chunk"),
            _ = &mut execution => panic!("execution completed before streaming output"),
        };
        assert!(first.contains("first"), "{first}");

        let outcome = execution.await.expect("streamed execution");
        assert!(!outcome.is_error);
        assert!(outcome.output.contains("firstsecond"), "{}", outcome.output);
        assert!(outcome.output.contains("problem"), "{}", outcome.output);
        assert_eq!(outcome.exit_code, Some(0));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shell_run_does_not_leak_secret_environment_variables() {
        // SAFETY: single-threaded test mutating its own environment.
        unsafe { std::env::set_var("AGENT_EXEC_TEST_TOKEN", "leaked-value") };
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        if !require_usable_sandbox(&harness.broker) {
            unsafe { std::env::remove_var("AGENT_EXEC_TEST_TOKEN") };
            return;
        }
        let response = harness
            .call(
                "shell_run",
                json!({ "command": "echo \"[${AGENT_EXEC_TEST_TOKEN:-absent}]\"" }),
            )
            .await;
        unsafe { std::env::remove_var("AGENT_EXEC_TEST_TOKEN") };
        assert!(
            response.output.contains("[absent]"),
            "token must not reach the tool: {}",
            response.output
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shell_run_times_out_instead_of_hanging() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        if !require_usable_sandbox(&harness.broker) {
            return;
        }
        let response = harness
            .call(
                "shell_run",
                json!({ "command": "sleep 5", "timeout_ms": 300 }),
            )
            .await;
        assert!(response.is_error);
        assert!(
            response.output.contains("terminated"),
            "{}",
            response.output
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn background_processes_stream_output_and_stop_on_request() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        if !require_usable_sandbox(&harness.broker) {
            return;
        }
        let started = harness
            .call(
                "shell_start",
                json!({ "command": "for i in 1 2 3; do echo tick-$i; sleep 0.1; done; sleep 30" }),
            )
            .await;
        let id = started
            .output
            .split_whitespace()
            .find(|token| token.starts_with("proc-"))
            .expect("process id")
            .trim_end_matches('.')
            .to_owned();

        let mut seen = String::new();
        for _ in 0..40 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let output = harness
                .call("shell_output", json!({ "process_id": id }))
                .await;
            seen.push_str(&output.output);
            if seen.contains("tick-3") {
                break;
            }
        }
        assert!(seen.contains("tick-1"), "streamed output: {seen}");
        assert!(seen.contains("tick-3"), "streamed output: {seen}");

        let terminated = harness
            .call("shell_terminate", json!({ "process_id": id }))
            .await;
        assert!(terminated.output.contains("Terminated"));
        assert_eq!(
            harness
                .broker
                .terminate_session_processes(&harness.session.id)
                .await,
            0
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_a_session_kills_its_processes() {
        let harness = Harness::new(AgentPermissionMode::SkipPermissions).await;
        if !require_usable_sandbox(&harness.broker) {
            return;
        }
        harness
            .call("shell_start", json!({ "command": "sleep 30" }))
            .await;
        let killed = harness
            .broker
            .terminate_session_processes(&harness.session.id)
            .await;
        assert_eq!(killed, 1);
    }

    #[test]
    fn glob_matching_handles_stars_and_question_marks() {
        assert!(glob_match("*.rs", "main.rs"));
        assert!(!glob_match("*.rs", "main.rss"));
        assert!(glob_match("a?c", "abc"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("src/*.ts", "src/api.ts"));
    }

    #[test]
    fn shell_quoting_survives_embedded_quotes() {
        assert_eq!(shell_quote("simple"), "'simple'");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
    }

    // -----------------------------------------------------------------------
    // Power tools: advanced workspace, project, web, and language-server tools.
    // -----------------------------------------------------------------------

    #[test]
    fn remote_credentials_are_redacted() {
        assert_eq!(
            redact_remote_credentials("origin https://alice:secret@example.com/repo.git (fetch)"),
            "origin https://***@example.com/repo.git (fetch)"
        );
        assert_eq!(
            redact_remote_credentials("origin git@example.com:repo.git (push)"),
            "origin git@example.com:repo.git (push)"
        );
    }

    #[test]
    fn html_text_is_cleaned_for_the_model() {
        let page = "<html><head><style>p{color:red}</style><script>alert(1)</script></head>\
                    <body><h1>Hello</h1><p>First line.</p><p></p><p>Second line.</p></body></html>";
        let text = brazier_runtime::web::html_to_text(page);
        assert!(text.contains("Hello"));
        assert!(text.contains("First line."));
        assert!(!text.contains("<style>"));
        assert!(!text.contains("alert"));
        assert!(!text.contains("<h1>"));
    }

    #[test]
    fn search_results_parse_duckduckgo_html() {
        let html = r#"<div class="result">
          <h2 class="result__title">
            <a rel="nofollow" class="result__a"
               href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs&amp;rut=abc">
               Rust <b>docs</b> &amp; guides
            </a>
          </h2>
          <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs">
            Learn Rust with <em>examples</em>.
          </a>
        </div>
        <div class="result">
          <a rel="nofollow" class="result__a" href="https://crates.io">crates.io</a>
        </div>"#;
        let results = brazier_runtime::web::parse_html_results(html, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Rust docs & guides");
        assert_eq!(results[0].url, "https://example.com/docs");
        assert!(results[0].snippet.contains("Learn Rust"));
        assert_eq!(results[1].title, "crates.io");
        assert_eq!(results[1].url, "https://crates.io");
        // max_results is honored.
        assert_eq!(brazier_runtime::web::parse_html_results(html, 1).len(), 1);
    }

    #[tokio::test]
    async fn web_fetch_rejects_non_http_schemes() {
        let harness = Harness::new_host(AgentPermissionMode::SkipPermissions).await;
        let response = harness
            .call("web_fetch", json!({ "url": "file:///etc/passwd" }))
            .await;
        assert!(response.is_error, "{}", response.output);
        assert!(
            response.output.contains("only http(s)"),
            "{}",
            response.output
        );
    }

    #[test]
    fn lsp_server_for_maps_extensions() {
        assert_eq!(
            lsp_server_for("ts").map(|entry| entry.0),
            Some("typescript-language-server")
        );
        assert_eq!(lsp_server_for("py").map(|entry| entry.0), Some("pylsp"));
        assert_eq!(
            lsp_server_for("rs").map(|entry| entry.0),
            Some("rust-analyzer")
        );
        assert_eq!(lsp_server_for("c").map(|entry| entry.0), Some("clangd"));
        assert_eq!(lsp_server_for("cpp").map(|entry| entry.0), Some("clangd"));
        assert_eq!(lsp_server_for("go").map(|entry| entry.0), Some("gopls"));
        assert_eq!(lsp_server_for("rb"), None);
    }

    #[tokio::test]
    async fn lsp_framing_round_trips_over_an_in_memory_duplex() {
        let (mut client, mut server) = tokio::io::duplex(4096);
        let message = json!({ "jsonrpc": "2.0", "method": "x", "params": { "n": 1 } });
        let send = lsp_send_frame(&mut client, &message);
        let read = async {
            let mut reader = BufReader::new(&mut server);
            lsp_read_frame(&mut reader).await
        };
        let (send_result, read_result) = tokio::join!(send, read);
        send_result.expect("write frame");
        let received = read_result.expect("read frame").expect("a frame arrived");
        assert_eq!(received, message);
    }

    #[tokio::test]
    async fn lsp_diagnostics_refuses_unknown_extensions() {
        let harness = Harness::new_host(AgentPermissionMode::SkipPermissions).await;
        std::fs::write(harness.workspace.path().join("script.rb"), "puts 1\n").expect("write file");
        let response = harness
            .call("lsp_diagnostics", json!({ "path": "script.rb" }))
            .await;
        assert!(
            response.output.contains("No language server is configured"),
            "{}",
            response.output
        );
    }

    #[tokio::test]
    async fn lsp_diagnostics_reports_errors_from_clangd_when_installed() {
        // clangd is system-installed on this machine; elsewhere the test skips.
        if find_on_path("clangd").is_none() {
            return;
        }
        let harness = Harness::new_host(AgentPermissionMode::SkipPermissions).await;
        std::fs::write(
            harness.workspace.path().join("broken.c"),
            "int main() { return nonexistent_thing; }\n",
        )
        .expect("write file");
        let response = harness
            .call("lsp_diagnostics", json!({ "path": "broken.c" }))
            .await;
        assert!(
            response.output.contains("diagnostic") && !response.is_error,
            "expected diagnostics from clangd, got: {}",
            response.output
        );
    }

    #[test]
    fn power_tools_are_host_routed_and_refused_in_sandbox_only() {
        let backend = SandboxBackend::detect();
        let permissions = AgentPermissionSettings::default();
        fn request<'a>(
            tool: &'a str,
            arguments: &'a Value,
            mode: AgentPermissionMode,
            permissions: AgentPermissionSettings,
            backend: &'a SandboxBackend,
        ) -> agent_policy::PolicyRequest<'a> {
            agent_policy::PolicyRequest {
                tool,
                arguments,
                requested_environment: AgentEnvironment::Sandbox,
                permission_mode: mode,
                permission_settings: permissions,
                workspace: Some(std::path::Path::new("/ws")),
                backend,
                session_grants: &[],
                reason: None,
                data_dir: std::path::Path::new("/tmp"),
            }
        }
        for tool in crate::agent_tools::POWER_TOOLS {
            let arguments = json!({});
            let decision = agent_policy::decide(&request(
                tool,
                &arguments,
                AgentPermissionMode::Ask,
                permissions,
                &backend,
            ));
            match decision {
                agent_policy::PolicyDecision::RequireApproval { environment, .. } => {
                    assert_eq!(
                        environment,
                        AgentEnvironment::Host,
                        "{tool} must run as a host action"
                    );
                }
                other => panic!("{tool} should require host approval in ask mode, got {other:?}"),
            }
            let denied = agent_policy::decide(&request(
                tool,
                &arguments,
                AgentPermissionMode::SandboxOnly,
                permissions,
                &backend,
            ));
            assert!(
                matches!(denied, agent_policy::PolicyDecision::Deny { .. }),
                "{tool} must be refused in sandbox-only mode"
            );
        }
    }
}

//! Policy broker for agent tool calls.
//!
//! The runtime adapter never decides what is allowed. Every call arrives here
//! first with the session's permission mode, the detected sandbox backend, and
//! the grants the user has already given. The result is one of: run it, ask the
//! user, or refuse.

use std::path::{Component, Path, PathBuf};

use crate::agent_sandbox::{SandboxBackend, SandboxProfile, secret_paths};
use brazier_protocol::agent_types::{
    AgentElevationRequest, AgentEnvironment, AgentPermissionMode, AgentPermissionSettings,
    RequestedPathAccess, ToolRiskLevel,
};

/// Static facts about a tool the daemon can execute.
#[derive(Debug, Clone, Copy)]
pub struct ToolSpec {
    pub name: &'static str,
    pub risk: ToolRiskLevel,
    /// Runs another program.
    pub executes: bool,
    /// Cannot work without a workspace.
    pub needs_workspace: bool,
}

/// Every tool the execution broker implements. A call for anything else is
/// refused before it reaches an executor.
#[rustfmt::skip]
pub const TOOL_SPECS: &[ToolSpec] = &[
    ToolSpec { name: "workspace_info", risk: ToolRiskLevel::Safe, executes: false, needs_workspace: false },
    ToolSpec { name: "fs_list", risk: ToolRiskLevel::Read, executes: false, needs_workspace: true },
    ToolSpec { name: "fs_read", risk: ToolRiskLevel::Read, executes: false, needs_workspace: true },
    ToolSpec { name: "doc_read", risk: ToolRiskLevel::Read, executes: false, needs_workspace: true },
    ToolSpec { name: "fs_stat", risk: ToolRiskLevel::Read, executes: false, needs_workspace: true },
    ToolSpec { name: "fs_search", risk: ToolRiskLevel::Read, executes: false, needs_workspace: true },
    ToolSpec { name: "fs_write", risk: ToolRiskLevel::Write, executes: false, needs_workspace: true },
    ToolSpec { name: "fs_patch", risk: ToolRiskLevel::Write, executes: false, needs_workspace: true },
    ToolSpec { name: "fs_mkdir", risk: ToolRiskLevel::Write, executes: false, needs_workspace: true },
    ToolSpec { name: "fs_copy", risk: ToolRiskLevel::Write, executes: false, needs_workspace: true },
    ToolSpec { name: "fs_move", risk: ToolRiskLevel::Destructive, executes: false, needs_workspace: true },
    ToolSpec { name: "fs_delete", risk: ToolRiskLevel::Destructive, executes: false, needs_workspace: true },
    ToolSpec { name: "shell_run", risk: ToolRiskLevel::Execute, executes: true, needs_workspace: true },
    ToolSpec { name: "shell_start", risk: ToolRiskLevel::Execute, executes: true, needs_workspace: true },
    ToolSpec { name: "shell_input", risk: ToolRiskLevel::Execute, executes: true, needs_workspace: true },
    ToolSpec { name: "shell_output", risk: ToolRiskLevel::Read, executes: false, needs_workspace: true },
    ToolSpec { name: "shell_terminate", risk: ToolRiskLevel::Safe, executes: false, needs_workspace: true },
    ToolSpec { name: "git_status", risk: ToolRiskLevel::Read, executes: true, needs_workspace: true },
    ToolSpec { name: "git_diff", risk: ToolRiskLevel::Read, executes: true, needs_workspace: true },
    ToolSpec { name: "request_permission", risk: ToolRiskLevel::Safe, executes: false, needs_workspace: false },
    ToolSpec { name: "spawn_subagent", risk: ToolRiskLevel::Execute, executes: false, needs_workspace: false },
];

const MCP_TOOL_SPEC: ToolSpec = ToolSpec {
    name: "mcp/*",
    risk: ToolRiskLevel::Execute,
    executes: true,
    needs_workspace: false,
};

/// MCP names have a server and tool component. The server process itself is a
/// host executable, so MCP calls use a conservative dynamic policy spec.
pub fn is_mcp_tool_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("mcp/") else {
        return false;
    };
    rest.split_once('/')
        .is_some_and(|(server, tool)| !server.is_empty() && !tool.is_empty())
}

pub fn tool_spec(name: &str) -> Option<&'static ToolSpec> {
    TOOL_SPECS
        .iter()
        .find(|spec| spec.name == name)
        .or_else(|| is_mcp_tool_name(name).then_some(&MCP_TOOL_SPEC))
}

/// A path the call wants to touch, already resolved against the workspace.
#[derive(Debug, Clone)]
pub struct RequestedPath {
    pub raw: String,
    /// Lexically resolved path, used for filesystem operations and messages.
    pub resolved: PathBuf,
    /// Canonical form of `resolved`, with symlinks followed as far as the path
    /// exists. Every containment decision uses this, never `resolved`.
    pub real: PathBuf,
    pub write: bool,
    pub inside_workspace: bool,
    pub secret: bool,
}

/// What the policy broker decided.
#[derive(Debug, Clone)]
pub enum PolicyDecision {
    /// Run it now in this environment and profile.
    Allow {
        environment: AgentEnvironment,
        profile: SandboxProfile,
        /// Why this was allowed without a prompt, for the audit record.
        rationale: String,
    },
    /// Ask the user first.
    RequireApproval {
        environment: AgentEnvironment,
        profile: SandboxProfile,
        elevation: AgentElevationRequest,
        /// Identity for a session-wide grant; empty when only `once` applies.
        scope_key: String,
        /// Whether the user may grant this for the whole session.
        allow_session_scope: bool,
        summary: String,
    },
    /// Refuse, with a reason the model and the user both see.
    Deny { reason: String },
}

/// Inputs to a policy decision.
pub struct PolicyRequest<'a> {
    pub tool: &'a str,
    pub arguments: &'a serde_json::Value,
    pub requested_environment: AgentEnvironment,
    pub permission_mode: AgentPermissionMode,
    pub permission_settings: AgentPermissionSettings,
    pub workspace: Option<&'a Path>,
    pub backend: &'a SandboxBackend,
    /// Scope keys already granted for this session, as `environment:key`.
    pub session_grants: &'a [String],
    /// Free-text reason supplied by the agent.
    pub reason: Option<&'a str>,
    /// Daemon data directory, always off limits to agent tools.
    pub data_dir: &'a Path,
}

pub use brazier_protocol::agent_types::grant_key;

/// Decide what happens to one tool call.
pub fn decide(request: &PolicyRequest<'_>) -> PolicyDecision {
    let Some(spec) = tool_spec(request.tool) else {
        return PolicyDecision::Deny {
            reason: format!("Unknown tool `{}`.", request.tool),
        };
    };

    if spec.needs_workspace && request.workspace.is_none() {
        return PolicyDecision::Deny {
            reason: "This session has no workspace. Choose a folder before running tools."
                .to_owned(),
        };
    }

    // Configured MCP servers run as host processes and may make outbound
    // connections. Treat that capability as requested even if a particular
    // tool schema has no `network` argument.
    let mcp_tool = is_mcp_tool_name(request.tool);
    let wants_network = mcp_tool || argument_bool(request.arguments, "network");
    let paths = requested_paths(
        request.tool,
        request.arguments,
        request.workspace,
        request.data_dir,
    );

    // Credential and daemon-owned paths are never reachable, in any mode.
    if let Some(secret) = paths.iter().find(|path| path.secret) {
        return PolicyDecision::Deny {
            reason: format!(
                "`{}` is a credential or Brazier-owned path. Agent tools never read or write it.",
                secret.raw
            ),
        };
    }

    let escapes_workspace = paths.iter().any(|path| !path.inside_workspace);
    let mut environment = request.requested_environment;
    if mcp_tool {
        environment = AgentEnvironment::Host;
    }
    // Touching anything outside the workspace is host access by definition.
    if escapes_workspace {
        environment = AgentEnvironment::Host;
    }

    let profile = if environment == AgentEnvironment::Host {
        SandboxProfile::Workspace
    } else if wants_network {
        SandboxProfile::WorkspaceNetwork
    } else if matches!(spec.risk, ToolRiskLevel::Safe | ToolRiskLevel::Read) && !spec.executes {
        SandboxProfile::ReadOnly
    } else {
        SandboxProfile::Workspace
    };

    // An unsandboxed host cannot honestly offer sandboxed execution. Running a
    // program then has host reach, so it is judged as host execution.
    let unsandboxed_execution =
        spec.executes && environment == AgentEnvironment::Sandbox && !request.backend.isolated();

    let scope_key = scope_key(request.tool, spec, request.arguments, &paths, wants_network);
    let summary = summarize(
        request.tool,
        request.arguments,
        environment,
        &paths,
        wants_network,
    );
    let elevation = AgentElevationRequest {
        // A caller-supplied reason wins; `request_permission` carries its own in
        // an argument, which is the point of the tool.
        reason: request
            .reason
            .map(str::to_owned)
            .or_else(|| argument_str(request.arguments, "reason").map(str::to_owned))
            .unwrap_or_else(|| summary.clone()),
        proposed_command: proposed_command(request.tool, request.arguments),
        requested_filesystem_paths: paths
            .iter()
            .filter(|path| !path.inside_workspace)
            .map(|path| RequestedPathAccess {
                path: path.resolved.display().to_string(),
                write: path.write,
            })
            .collect(),
        requested_network_access: wants_network,
        requested_host_execution: environment == AgentEnvironment::Host
            || unsandboxed_execution
            // `request_permission` states this outright instead of implying it
            // through a path or a command.
            || (request.tool == "request_permission"
                && argument_bool(request.arguments, "host_execution")),
    };

    let needs_prompt_by_risk = match spec.risk {
        ToolRiskLevel::Safe => false,
        ToolRiskLevel::Read => escapes_workspace,
        ToolRiskLevel::Write | ToolRiskLevel::Execute | ToolRiskLevel::Destructive => true,
    }
        || wants_network
        // Asking for access is the one tool whose entire purpose is to prompt.
        || request.tool == "request_permission";

    // Destructive work is never covered by a standing grant, and neither is
    // anything that writes or executes outside the sandbox. Reading one named
    // path outside the workspace may be granted for the session, because the
    // scope key names that exact path.
    //
    // `request_permission` is judged by what it asks for rather than by its own
    // risk level, so approving a read request pre-authorizes the read that
    // follows — its scope key is the one that read will compute.
    let read_only_request = request.tool == "request_permission"
        && !paths.is_empty()
        && paths.iter().all(|path| !path.write)
        && !argument_bool(request.arguments, "host_execution");
    let allow_session_scope = spec.risk != ToolRiskLevel::Destructive
        && !unsandboxed_execution
        && (environment == AgentEnvironment::Sandbox
            || spec.risk == ToolRiskLevel::Read
            || read_only_request);

    let host_action = environment == AgentEnvironment::Host || unsandboxed_execution;

    // A standing grant from this session covers a repeat of the same shape.
    if allow_session_scope
        && !scope_key.is_empty()
        && request
            .session_grants
            .iter()
            .any(|granted| granted == &grant_key(environment, &scope_key))
    {
        return PolicyDecision::Allow {
            environment,
            profile,
            rationale: format!("Session grant for `{scope_key}`."),
        };
    }

    match request.permission_mode {
        AgentPermissionMode::Ask => {
            if !needs_prompt_by_risk && !host_action {
                return PolicyDecision::Allow {
                    environment,
                    profile,
                    rationale: "Sandboxed read inside the workspace.".to_owned(),
                };
            }
            PolicyDecision::RequireApproval {
                environment,
                profile,
                elevation,
                scope_key,
                allow_session_scope,
                summary,
            }
        }
        AgentPermissionMode::SandboxOnly => {
            if host_action {
                let reason = if unsandboxed_execution {
                    format!(
                        "Sandbox-only mode refuses `{}`: {} Running programs here would have full \
                         host access.",
                        request.tool,
                        request.backend.capabilities().detail
                    )
                } else {
                    format!(
                        "Sandbox-only mode refuses host access. `{}` needs to leave the workspace.",
                        request.tool
                    )
                };
                return PolicyDecision::Deny { reason };
            }
            PolicyDecision::Allow {
                environment,
                profile,
                rationale: "Sandbox-only mode auto-approves sandboxed actions.".to_owned(),
            }
        }
        AgentPermissionMode::SkipPermissions => {
            if host_action {
                if request.permission_settings.auto_approve_host_actions {
                    return PolicyDecision::Allow {
                        environment,
                        profile,
                        rationale: "Skip-permissions mode with host actions auto-approved."
                            .to_owned(),
                    };
                }
                return PolicyDecision::RequireApproval {
                    environment,
                    profile,
                    elevation,
                    scope_key,
                    allow_session_scope,
                    summary,
                };
            }
            if request.permission_settings.auto_approve_sandboxed_actions {
                return PolicyDecision::Allow {
                    environment,
                    profile,
                    rationale: "Skip-permissions mode with sandboxed actions auto-approved."
                        .to_owned(),
                };
            }
            PolicyDecision::RequireApproval {
                environment,
                profile,
                elevation,
                scope_key,
                allow_session_scope,
                summary,
            }
        }
    }
}

fn argument_bool(arguments: &serde_json::Value, key: &str) -> bool {
    arguments
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn argument_str<'a>(arguments: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    arguments.get(key).and_then(serde_json::Value::as_str)
}

/// Identity used for session-wide grants: the shape of the action, not its
/// exact arguments. Approving `cargo` once should not re-prompt per command.
///
/// Paths outside the workspace are the exception: their key names the resolved
/// path, so granting a read of one file grants that file and nothing else.
fn scope_key(
    tool: &str,
    spec: &ToolSpec,
    arguments: &serde_json::Value,
    paths: &[RequestedPath],
    wants_network: bool,
) -> String {
    // stdin injection is scoped to the specific background process that was
    // approved, not to every process in the session.
    if tool == "shell_input" {
        let process = argument_str(arguments, "process_id").unwrap_or("unknown");
        return format!("shell-input:{process}");
    }
    if spec.executes {
        let program = argument_str(arguments, "command")
            .and_then(|command| command.split_whitespace().next())
            .unwrap_or(tool);
        let suffix = if wants_network { "+network" } else { "" };
        return format!("run:{program}{suffix}");
    }
    if let Some(outside) = paths.iter().find(|path| !path.inside_workspace) {
        let verb = if outside.write { "fs-write" } else { "fs-read" };
        return format!("{verb}:{}", outside.real.display());
    }
    if paths.iter().any(|path| path.write) {
        return "fs-write:workspace".to_owned();
    }
    format!("tool:{tool}")
}

fn proposed_command(tool: &str, arguments: &serde_json::Value) -> Option<String> {
    if !tool_spec(tool).is_some_and(|spec| spec.executes) {
        return None;
    }
    argument_str(arguments, "command").map(str::to_owned)
}

/// One-line description of what will happen, shown in the approval dialog.
fn summarize(
    tool: &str,
    arguments: &serde_json::Value,
    environment: AgentEnvironment,
    paths: &[RequestedPath],
    wants_network: bool,
) -> String {
    let where_ = match environment {
        AgentEnvironment::Sandbox => "in the sandbox",
        AgentEnvironment::Host => "on the host",
    };
    // Asking for access describes itself completely, so it returns early rather
    // than collecting the generic suffixes below.
    if tool == "request_permission" {
        let mut summary = "Grant access the agent does not have:".to_owned();
        let wanted: Vec<String> = paths
            .iter()
            .map(|path| {
                format!(
                    "{} ({})",
                    path.resolved.display(),
                    if path.write { "write" } else { "read" }
                )
            })
            .collect();
        if wanted.is_empty() {
            summary.push_str(" no paths named");
        } else {
            summary.push(' ');
            summary.push_str(&wanted.join(", "));
        }
        if wants_network {
            summary.push_str(" · outbound network");
        }
        if argument_bool(arguments, "host_execution") {
            summary.push_str(" · commands outside the sandbox");
        }
        return summary;
    }

    let mut summary = match tool {
        "shell_run" | "shell_start" => format!(
            "Run `{}` {where_}",
            argument_str(arguments, "command").unwrap_or("(no command)")
        ),
        "fs_write" => format!(
            "Write {} {where_}",
            argument_str(arguments, "path").unwrap_or("a file")
        ),
        "fs_patch" => format!(
            "Patch {} {where_}",
            argument_str(arguments, "path").unwrap_or("a file")
        ),
        "fs_delete" => format!(
            "Delete {} {where_}",
            argument_str(arguments, "path").unwrap_or("a path")
        ),
        "fs_move" => format!(
            "Move {} to {} {where_}",
            argument_str(arguments, "from").unwrap_or("a path"),
            argument_str(arguments, "to").unwrap_or("a path")
        ),
        "fs_copy" => format!(
            "Copy {} to {} {where_}",
            argument_str(arguments, "from").unwrap_or("a path"),
            argument_str(arguments, "to").unwrap_or("a path")
        ),
        "spawn_subagent" => {
            let detail = argument_str(arguments, "prompt")
                .map(str::to_owned)
                .or_else(|| {
                    arguments
                        .get("prompts")
                        .and_then(|value| value.as_array())
                        .map(|entries| format!("{} tasks", entries.len()))
                })
                .unwrap_or_else(|| "(no prompt)".to_owned());
            let trimmed = detail.trim();
            let preview = if trimmed.len() > 80 {
                format!("{}…", &trimmed[..80])
            } else {
                trimmed.to_owned()
            };
            format!("Spawn subagent(s) {where_}: {preview}")
        }
        other => format!("Run `{other}` {where_}"),
    };
    let outside: Vec<String> = paths
        .iter()
        .filter(|path| !path.inside_workspace)
        .map(|path| path.resolved.display().to_string())
        .collect();
    if !outside.is_empty() {
        summary.push_str(&format!(" · outside the workspace: {}", outside.join(", ")));
    }
    if wants_network {
        summary.push_str(" · network enabled");
    }
    summary
}

/// Which paths a call touches, and whether each one is a write.
pub fn requested_paths(
    tool: &str,
    arguments: &serde_json::Value,
    workspace: Option<&Path>,
    data_dir: &Path,
) -> Vec<RequestedPath> {
    let fields: &[(&str, bool)] = match tool {
        "fs_read" | "doc_read" | "fs_list" | "fs_stat" | "fs_search" => &[("path", false)],
        "fs_write" | "fs_patch" | "fs_mkdir" => &[("path", true)],
        "fs_delete" => &[("path", true)],
        "fs_move" => &[("from", true), ("to", true)],
        "fs_copy" => &[("from", false), ("to", true)],
        "shell_run" | "shell_start" | "git_status" | "git_diff" => &[("cwd", false)],
        _ => &[],
    };
    let secrets = secret_paths(Some(data_dir));
    let workspace_real = workspace.map(canonical_ancestor);

    // `request_permission` carries its paths in an array rather than in named
    // fields, so the same checks apply to what it asks for.
    if tool == "request_permission" {
        let entries = arguments
            .get("paths")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        return entries
            .iter()
            .filter_map(|entry| {
                let raw = entry.get("path").and_then(serde_json::Value::as_str)?;
                if raw.trim().is_empty() {
                    return None;
                }
                let write = entry
                    .get("write")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let resolved = resolve_path(workspace, raw);
                let real = canonical_ancestor(&resolved);
                let inside_workspace = workspace_real
                    .as_deref()
                    .map(|root| is_inside(&real, root))
                    .unwrap_or(false);
                let secret = secrets
                    .iter()
                    .any(|secret| is_inside(&real, secret) || is_inside(&resolved, secret));
                Some(RequestedPath {
                    raw: raw.to_owned(),
                    resolved,
                    real,
                    write,
                    inside_workspace,
                    secret,
                })
            })
            .collect();
    }

    fields
        .iter()
        .filter_map(|(field, write)| {
            let raw = argument_str(arguments, field)?;
            if raw.is_empty() {
                return None;
            }
            let resolved = resolve_path(workspace, raw);
            let real = canonical_ancestor(&resolved);
            // Containment is judged on the canonical path: a symlink inside the
            // workspace that points out of it counts as outside.
            let inside_workspace = workspace_real
                .as_deref()
                .map(|root| is_inside(&real, root))
                .unwrap_or(false);
            let secret = secrets
                .iter()
                .any(|secret| is_inside(&real, secret) || is_inside(&resolved, secret));
            Some(RequestedPath {
                raw: raw.to_owned(),
                resolved,
                real,
                write: *write,
                inside_workspace,
                secret,
            })
        })
        .collect()
}

/// Canonicalize the deepest existing ancestor of a path and re-append the
/// remainder, so a not-yet-created file is still judged against real parents.
/// Paths that do not exist at all come back lexically normalized.
pub fn canonical_ancestor(path: &Path) -> PathBuf {
    let mut suffix: Vec<std::ffi::OsString> = Vec::new();
    let mut cursor = normalize(path);
    loop {
        if let Ok(canonical) = std::fs::canonicalize(&cursor) {
            let mut result = canonical;
            for part in suffix.iter().rev() {
                result.push(part);
            }
            return result;
        }
        let Some(name) = cursor.file_name().map(|name| name.to_os_string()) else {
            return normalize(path);
        };
        suffix.push(name);
        if !cursor.pop() {
            return normalize(path);
        }
    }
}

/// Resolve a tool-supplied path against the workspace and normalize it
/// lexically. `..` is collapsed here so traversal cannot hide behind it, and
/// the executor canonicalizes again before touching the filesystem.
pub fn resolve_path(workspace: Option<&Path>, raw: &str) -> PathBuf {
    let candidate = Path::new(raw);
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else if let Some(root) = workspace {
        root.join(candidate)
    } else {
        candidate.to_path_buf()
    };
    normalize(&joined)
}

/// Lexical normalization: drop `.`, resolve `..` against collected components.
pub fn normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    normalized.push("..");
                }
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

/// True when `path` is `root` or sits under it. Both sides are normalized
/// first; neither is required to exist.
pub fn is_inside(path: &Path, root: &Path) -> bool {
    let path = normalize(path);
    let root = normalize(root);
    path == root || path.starts_with(&root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn backend() -> SandboxBackend {
        SandboxBackend::detect()
    }

    fn base<'a>(
        tool: &'a str,
        arguments: &'a serde_json::Value,
        workspace: &'a Path,
        backend: &'a SandboxBackend,
        data_dir: &'a Path,
        grants: &'a [String],
    ) -> PolicyRequest<'a> {
        PolicyRequest {
            tool,
            arguments,
            requested_environment: AgentEnvironment::Sandbox,
            permission_mode: AgentPermissionMode::Ask,
            permission_settings: AgentPermissionSettings::default(),
            workspace: Some(workspace),
            backend,
            session_grants: grants,
            reason: None,
            data_dir,
        }
    }

    #[test]
    fn reads_inside_the_workspace_need_no_prompt() {
        let backend = backend();
        let arguments = json!({ "path": "src/main.rs" });
        let request = base(
            "fs_read",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        );
        assert!(matches!(decide(&request), PolicyDecision::Allow { .. }));
    }

    #[test]
    fn writes_inside_the_workspace_ask_first_and_can_be_granted_for_the_session() {
        let backend = backend();
        let arguments = json!({ "path": "src/main.rs", "content": "fn main() {}" });
        let request = base(
            "fs_write",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        );
        match decide(&request) {
            PolicyDecision::RequireApproval {
                scope_key,
                allow_session_scope,
                environment,
                ..
            } => {
                assert_eq!(scope_key, "fs-write:workspace");
                assert!(allow_session_scope);
                assert_eq!(environment, AgentEnvironment::Sandbox);
            }
            other => panic!("expected approval, got {other:?}"),
        }
    }

    #[test]
    fn a_session_grant_skips_the_second_prompt() {
        let backend = backend();
        let arguments = json!({ "path": "src/main.rs", "content": "fn main() {}" });
        let grants = vec![grant_key(AgentEnvironment::Sandbox, "fs-write:workspace")];
        let request = base(
            "fs_write",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &grants,
        );
        assert!(matches!(decide(&request), PolicyDecision::Allow { .. }));
    }

    #[test]
    fn destructive_calls_never_accept_a_session_grant() {
        let backend = backend();
        let arguments = json!({ "path": "src" });
        let grants = vec![grant_key(AgentEnvironment::Sandbox, "fs-write:workspace")];
        let request = base(
            "fs_delete",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &grants,
        );
        match decide(&request) {
            PolicyDecision::RequireApproval {
                allow_session_scope,
                ..
            } => assert!(!allow_session_scope),
            other => panic!("expected approval, got {other:?}"),
        }
    }

    #[test]
    fn leaving_the_workspace_becomes_a_host_action() {
        let backend = backend();
        let arguments = json!({ "path": "/etc/hosts" });
        let request = base(
            "fs_read",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        );
        match decide(&request) {
            PolicyDecision::RequireApproval {
                environment,
                elevation,
                allow_session_scope,
                scope_key,
                ..
            } => {
                assert_eq!(environment, AgentEnvironment::Host);
                assert!(elevation.requested_host_execution);
                assert_eq!(elevation.requested_filesystem_paths.len(), 1);
                // A standing grant is offered, but it names the one path, so it
                // cannot be spent on any other file outside the workspace.
                assert!(allow_session_scope);
                // The key uses the canonical path, so two spellings of the same
                // file cannot each earn their own grant.
                assert_eq!(
                    scope_key,
                    format!(
                        "fs-read:{}",
                        canonical_ancestor(Path::new("/etc/hosts")).display()
                    )
                );
            }
            other => panic!("expected approval, got {other:?}"),
        }
    }

    #[test]
    fn mcp_tools_are_one_shot_host_network_actions() {
        let backend = backend();
        let arguments = json!({ "query": "release notes" });
        let request = base(
            "mcp/search/web",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        );
        match decide(&request) {
            PolicyDecision::RequireApproval {
                environment,
                elevation,
                allow_session_scope,
                scope_key,
                summary,
                ..
            } => {
                assert_eq!(environment, AgentEnvironment::Host);
                assert!(elevation.requested_host_execution);
                assert!(elevation.requested_network_access);
                assert!(!allow_session_scope);
                assert_eq!(scope_key, "run:mcp/search/web+network");
                assert!(summary.contains("network enabled"), "{summary}");
            }
            other => panic!("expected approval, got {other:?}"),
        }

        assert!(matches!(
            decide(&base(
                "mcp/no-tool-component",
                &arguments,
                Path::new("/ws"),
                &backend,
                Path::new("/data"),
                &[]
            )),
            PolicyDecision::Deny { .. }
        ));
    }

    #[test]
    fn a_grant_for_one_outside_path_does_not_cover_another() {
        let backend = backend();
        let granted = json!({ "path": "/etc/hosts" });
        let other = json!({ "path": "/etc/passwd" });
        let grants = vec![grant_key(
            AgentEnvironment::Host,
            &format!(
                "fs-read:{}",
                canonical_ancestor(Path::new("/etc/hosts")).display()
            ),
        )];
        assert!(matches!(
            decide(&base(
                "fs_read",
                &granted,
                Path::new("/ws"),
                &backend,
                Path::new("/data"),
                &grants
            )),
            PolicyDecision::Allow { .. }
        ));
        assert!(matches!(
            decide(&base(
                "fs_read",
                &other,
                Path::new("/ws"),
                &backend,
                Path::new("/data"),
                &grants
            )),
            PolicyDecision::RequireApproval { .. }
        ));
    }

    #[test]
    fn writing_outside_the_workspace_is_never_a_standing_grant() {
        let backend = backend();
        let arguments = json!({ "path": "/etc/hosts", "content": "127.0.0.1 evil" });
        match decide(&base(
            "fs_write",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        )) {
            PolicyDecision::RequireApproval {
                allow_session_scope,
                environment,
                scope_key,
                ..
            } => {
                assert_eq!(environment, AgentEnvironment::Host);
                assert!(!allow_session_scope, "host writes stay one-shot");
                assert_eq!(
                    scope_key,
                    format!(
                        "fs-write:{}",
                        canonical_ancestor(Path::new("/etc/hosts")).display()
                    )
                );
            }
            other => panic!("expected approval, got {other:?}"),
        }
    }

    #[test]
    fn requesting_permission_always_asks_and_names_what_it_wants() {
        let backend = backend();
        let arguments = json!({
            "reason": "The build needs the system SDK headers.",
            "paths": [{ "path": "/opt/sdk", "write": false }],
            "network": true,
            "host_execution": true
        });
        let mut request = base(
            "request_permission",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        );
        request.reason = None;
        match decide(&request) {
            PolicyDecision::RequireApproval {
                elevation,
                summary,
                allow_session_scope,
                ..
            } => {
                // The reason comes from the tool's own argument, not from a
                // caller-supplied one, and the summary states the access asked
                // for without repeating itself.
                assert!(summary.contains("/opt/sdk (read)"), "{summary}");
                assert!(summary.contains("outbound network"), "{summary}");
                assert_eq!(summary.matches("network").count(), 1, "{summary}");
                // Host execution was requested, so no standing grant is offered.
                assert!(!allow_session_scope);
                assert_eq!(elevation.reason, "The build needs the system SDK headers.");
                assert!(elevation.requested_network_access);
                assert!(elevation.requested_host_execution);
                assert_eq!(
                    elevation
                        .requested_filesystem_paths
                        .iter()
                        .map(|entry| entry.path.as_str())
                        .collect::<Vec<_>>(),
                    ["/opt/sdk"]
                );
            }
            other => panic!("expected approval, got {other:?}"),
        }
    }

    #[test]
    fn a_read_only_access_request_can_pre_authorize_the_read_that_follows() {
        let backend = backend();
        let arguments = json!({
            "reason": "I need to check the system hosts file.",
            "paths": [{ "path": "/etc/hosts", "write": false }]
        });
        let expected_key = format!(
            "fs-read:{}",
            canonical_ancestor(Path::new("/etc/hosts")).display()
        );
        match decide(&base(
            "request_permission",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        )) {
            PolicyDecision::RequireApproval {
                allow_session_scope,
                scope_key,
                ..
            } => {
                assert!(allow_session_scope, "a read-only request may be granted");
                // The key is the one `fs_read` of that path computes, so the
                // grant covers the call the agent actually makes next.
                assert_eq!(scope_key, expected_key);
            }
            other => panic!("expected approval, got {other:?}"),
        }

        let grants = vec![grant_key(AgentEnvironment::Host, &expected_key)];
        let read = json!({ "path": "/etc/hosts" });
        assert!(matches!(
            decide(&base(
                "fs_read",
                &read,
                Path::new("/ws"),
                &backend,
                Path::new("/data"),
                &grants
            )),
            PolicyDecision::Allow { .. }
        ));
    }

    #[test]
    fn requesting_permission_for_a_credential_path_is_refused() {
        let backend = backend();
        let arguments = json!({
            "reason": "I need the deploy key.",
            "paths": [{ "path": "/data/brazier.sqlite", "write": false }]
        });
        assert!(matches!(
            decide(&base(
                "request_permission",
                &arguments,
                Path::new("/ws"),
                &backend,
                Path::new("/data"),
                &[]
            )),
            PolicyDecision::Deny { .. }
        ));
    }

    #[test]
    fn traversal_out_of_the_workspace_is_detected() {
        let backend = backend();
        let arguments = json!({ "path": "../../etc/passwd" });
        let request = base(
            "fs_read",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        );
        match decide(&request) {
            PolicyDecision::RequireApproval { environment, .. } => {
                assert_eq!(environment, AgentEnvironment::Host);
            }
            other => panic!("expected approval, got {other:?}"),
        }
    }

    #[test]
    fn the_daemon_data_directory_is_always_refused() {
        let backend = backend();
        let arguments = json!({ "path": "/data/brazier.sqlite" });
        let request = base(
            "fs_read",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        );
        match decide(&request) {
            PolicyDecision::Deny { reason } => assert!(reason.contains("Brazier-owned")),
            other => panic!("expected denial, got {other:?}"),
        }
    }

    #[test]
    fn skip_permissions_still_refuses_credential_paths() {
        let backend = backend();
        let arguments = json!({ "path": "/data/brazier.sqlite" });
        let mut request = base(
            "fs_read",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        );
        request.permission_mode = AgentPermissionMode::SkipPermissions;
        request.permission_settings = AgentPermissionSettings {
            auto_approve_sandboxed_actions: true,
            auto_approve_host_actions: true,
        };
        assert!(matches!(decide(&request), PolicyDecision::Deny { .. }));
    }

    #[test]
    fn sandbox_only_mode_refuses_host_work_but_runs_sandboxed_work() {
        let backend = backend();
        let write_arguments = json!({ "path": "src/main.rs", "content": "x" });
        let mut request = base(
            "fs_write",
            &write_arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        );
        request.permission_mode = AgentPermissionMode::SandboxOnly;
        assert!(matches!(decide(&request), PolicyDecision::Allow { .. }));

        let host_arguments = json!({ "path": "/etc/hosts" });
        let mut host_request = base(
            "fs_read",
            &host_arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        );
        host_request.permission_mode = AgentPermissionMode::SandboxOnly;
        assert!(matches!(decide(&host_request), PolicyDecision::Deny { .. }));
    }

    #[test]
    fn skip_permissions_needs_the_host_flag_for_host_work() {
        let backend = backend();
        let arguments = json!({ "path": "/etc/hosts" });
        let mut request = base(
            "fs_read",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        );
        request.permission_mode = AgentPermissionMode::SkipPermissions;
        request.permission_settings = AgentPermissionSettings {
            auto_approve_sandboxed_actions: true,
            auto_approve_host_actions: false,
        };
        assert!(matches!(
            decide(&request),
            PolicyDecision::RequireApproval { .. }
        ));

        request.permission_settings.auto_approve_host_actions = true;
        assert!(matches!(decide(&request), PolicyDecision::Allow { .. }));
    }

    #[test]
    fn tools_that_need_a_workspace_refuse_without_one() {
        let backend = backend();
        let arguments = json!({ "command": "ls" });
        let mut request = base(
            "shell_run",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        );
        request.workspace = None;
        match decide(&request) {
            PolicyDecision::Deny { reason } => assert!(reason.contains("no workspace")),
            other => panic!("expected denial, got {other:?}"),
        }
    }

    #[test]
    fn unknown_tools_are_refused() {
        let backend = backend();
        let arguments = json!({});
        let request = base(
            "rm_minus_rf",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        );
        assert!(matches!(decide(&request), PolicyDecision::Deny { .. }));
    }

    #[test]
    fn network_requests_prompt_and_pick_the_network_profile() {
        let backend = backend();
        let arguments = json!({ "command": "curl https://example.com", "network": true });
        let request = base(
            "shell_run",
            &arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        );
        match decide(&request) {
            PolicyDecision::RequireApproval {
                profile,
                elevation,
                scope_key,
                ..
            } => {
                assert_eq!(profile, SandboxProfile::WorkspaceNetwork);
                assert!(elevation.requested_network_access);
                assert_eq!(scope_key, "run:curl+network");
            }
            other => panic!("expected approval, got {other:?}"),
        }
    }

    #[test]
    fn shell_scope_keys_group_by_program() {
        let backend = backend();
        let first = json!({ "command": "cargo test" });
        let second = json!({ "command": "cargo build --release" });
        let key_of = |arguments: &serde_json::Value| match decide(&base(
            "shell_run",
            arguments,
            Path::new("/ws"),
            &backend,
            Path::new("/data"),
            &[],
        )) {
            PolicyDecision::RequireApproval { scope_key, .. } => scope_key,
            other => panic!("expected approval, got {other:?}"),
        };
        assert_eq!(key_of(&first), "run:cargo");
        assert_eq!(key_of(&first), key_of(&second));
    }

    #[test]
    fn path_normalization_collapses_traversal() {
        assert_eq!(
            resolve_path(Some(Path::new("/ws")), "a/../b/./c"),
            PathBuf::from("/ws/b/c")
        );
        assert_eq!(
            resolve_path(Some(Path::new("/ws")), "../secret"),
            PathBuf::from("/secret")
        );
        assert!(is_inside(Path::new("/ws/a/b"), Path::new("/ws")));
        assert!(is_inside(Path::new("/ws"), Path::new("/ws")));
        assert!(!is_inside(Path::new("/wsx/a"), Path::new("/ws")));
    }
}

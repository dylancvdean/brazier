//! Agent tool definitions and the agent system prompt.
//!
//! The daemon owns both: the runtime adapter fetches this catalog and hands it
//! to whatever framework it drives, so adding a tool never requires a change
//! outside this file, and the schema a model sees always matches the executor
//! and policy that back it.

use serde_json::{Value, json};

use crate::{
    agent_policy::{TOOL_SPECS, tool_spec},
    agent_sandbox::SandboxBackendCapabilities,
    agent_types::AgentSessionRecord,
};

/// Full catalog: JSON Schema for the model, risk metadata for the UI.
pub fn definitions() -> Value {
    let entries: Vec<Value> = raw_definitions()
        .into_iter()
        .map(|(name, description, schema)| {
            let spec = tool_spec(name).expect("every definition has a policy spec");
            json!({
                "name": name,
                "label": label(name),
                "description": description,
                "input_schema": schema,
                "risk": spec.risk.as_str(),
                "executes": spec.executes,
                "needs_workspace": spec.needs_workspace,
                // The sandbox is always the default; leaving it is an approval.
                "default_environment": "sandbox",
            })
        })
        .collect();
    json!({ "data": entries })
}

/// Human label for the activity timeline.
fn label(name: &str) -> &'static str {
    match name {
        "workspace_info" => "Workspace",
        "fs_list" => "List",
        "fs_read" => "Read",
        "fs_stat" => "Stat",
        "fs_search" => "Search",
        "fs_write" => "Write",
        "fs_patch" => "Edit",
        "fs_mkdir" => "Make directory",
        "fs_copy" => "Copy",
        "fs_move" => "Move",
        "fs_delete" => "Delete",
        "shell_run" => "Run",
        "shell_start" => "Start process",
        "shell_output" => "Process output",
        "shell_input" => "Process input",
        "shell_terminate" => "Stop process",
        "git_status" => "Git status",
        "git_diff" => "Git diff",
        "request_permission" => "Request access",
        _ => "Tool",
    }
}

fn object(properties: Value, required: Vec<&str>) -> Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

/// Like [`object`], plus the optional `reason` every tool that can prompt should
/// accept. The approval dialog shows it, so the user sees why the agent wants
/// this rather than only what it is about to do.
fn object_with_reason(properties: Value, required: Vec<&str>) -> Value {
    let mut schema = object(properties, required);
    schema["properties"]["reason"] = json!({
        "type": "string",
        "description": "Why this call is needed, in one sentence. Shown to the user if they are \
                        asked to approve it."
    });
    schema
}

fn path_property(description: &str) -> Value {
    json!({ "type": "string", "description": description })
}

#[allow(clippy::type_complexity)]
fn raw_definitions() -> Vec<(&'static str, &'static str, Value)> {
    let workspace_relative =
        "Path relative to the workspace root. Absolute paths outside the workspace need approval.";
    vec![
        (
            "workspace_info",
            "Describe the workspace: root path, whether it is a git repository, top-level \
             entries, the sandbox in force, and the current permission mode. Call this first when \
             you do not know the layout.",
            object(json!({}), vec![]),
        ),
        (
            "fs_list",
            "List directory entries with sizes. Vendored directories such as node_modules, \
             target, and .git are skipped below the first level.",
            object(
                json!({
                    "path": path_property(workspace_relative),
                    "depth": {
                        "type": "integer",
                        "description": "Levels to descend, 1-4. Default 1.",
                        "minimum": 1,
                        "maximum": 4
                    }
                }),
                vec![],
            ),
        ),
        (
            "fs_read",
            "Read a text file with line numbers. Pass start_line and line_count to read part of \
             a large file.",
            object(
                json!({
                    "path": path_property(workspace_relative),
                    "start_line": { "type": "integer", "description": "First line to return, 1-based.", "minimum": 1 },
                    "line_count": { "type": "integer", "description": "How many lines to return.", "minimum": 1 }
                }),
                vec!["path"],
            ),
        ),
        (
            "fs_stat",
            "Report whether a path is a file, directory, or symlink, plus its size.",
            object(
                json!({ "path": path_property(workspace_relative) }),
                vec!["path"],
            ),
        ),
        (
            "fs_search",
            "Search file contents for literal text. This is not a regular-expression engine: \
             pass plain text. Filter filenames with name_glob (`*` and `?` wildcards).",
            object(
                json!({
                    "query": { "type": "string", "description": "Literal text to find." },
                    "path": path_property("Directory to search. Defaults to the workspace root."),
                    "name_glob": { "type": "string", "description": "Only search files whose name matches, e.g. `*.rs`." },
                    "case_sensitive": { "type": "boolean", "description": "Default false." }
                }),
                vec!["query"],
            ),
        ),
        (
            "fs_write",
            "Create or overwrite a file with the given content. Parent directories are created. \
             Prefer fs_patch when editing an existing file.",
            object_with_reason(
                json!({
                    "path": path_property(workspace_relative),
                    "content": { "type": "string", "description": "Complete new file content." }
                }),
                vec!["path", "content"],
            ),
        ),
        (
            "fs_patch",
            "Replace an exact string in a file. old_string must appear exactly once unless \
             replace_all is true; include surrounding context to make it unique.",
            object_with_reason(
                json!({
                    "path": path_property(workspace_relative),
                    "old_string": { "type": "string", "description": "Exact text to replace, including indentation." },
                    "new_string": { "type": "string", "description": "Replacement text." },
                    "replace_all": { "type": "boolean", "description": "Replace every occurrence. Default false." }
                }),
                vec!["path", "old_string", "new_string"],
            ),
        ),
        (
            "fs_mkdir",
            "Create a directory, including missing parents.",
            object_with_reason(
                json!({ "path": path_property(workspace_relative) }),
                vec!["path"],
            ),
        ),
        (
            "fs_copy",
            "Copy one file to another path.",
            object_with_reason(
                json!({
                    "from": path_property(workspace_relative),
                    "to": path_property(workspace_relative)
                }),
                vec!["from", "to"],
            ),
        ),
        (
            "fs_move",
            "Move or rename a file or directory.",
            object_with_reason(
                json!({
                    "from": path_property(workspace_relative),
                    "to": path_property(workspace_relative)
                }),
                vec!["from", "to"],
            ),
        ),
        (
            "fs_delete",
            "Delete a file, or a directory when recursive is true. Always destructive: the user \
             is asked every time.",
            object_with_reason(
                json!({
                    "path": path_property(workspace_relative),
                    "recursive": { "type": "boolean", "description": "Required to delete a directory." }
                }),
                vec!["path"],
            ),
        ),
        (
            "shell_run",
            "Run a shell command in the workspace and wait for it to finish. Output is captured \
             and truncated; the exit code is reported. Commands run sandboxed with no network \
             unless you set network to true, which needs approval. Do not use this for \
             interactive programs; use shell_start instead.",
            object_with_reason(
                json!({
                    "command": { "type": "string", "description": "Command line, run through the system shell." },
                    "cwd": path_property("Working directory. Defaults to the workspace root."),
                    "timeout_ms": { "type": "integer", "description": "Kill the command after this long. Default 120000, maximum 600000.", "minimum": 1000 },
                    "network": { "type": "boolean", "description": "Request outbound network access for this command." }
                }),
                vec!["command"],
            ),
        ),
        (
            "shell_start",
            "Start a long-running command in the background and return a process id. Use \
             shell_output to read what it printed, shell_input to send it stdin, and \
             shell_terminate to stop it.",
            object_with_reason(
                json!({
                    "command": { "type": "string", "description": "Command line to start." },
                    "cwd": path_property("Working directory. Defaults to the workspace root."),
                    "network": { "type": "boolean", "description": "Request outbound network access." }
                }),
                vec!["command"],
            ),
        ),
        (
            "shell_output",
            "Read buffered output from a background process. Output is consumed unless drain is \
             false.",
            object(
                json!({
                    "process_id": { "type": "string", "description": "Id returned by shell_start." },
                    "drain": { "type": "boolean", "description": "Clear the buffer after reading. Default true." }
                }),
                vec!["process_id"],
            ),
        ),
        (
            "shell_input",
            "Send a line of input to a background process's stdin.",
            object(
                json!({
                    "process_id": { "type": "string", "description": "Id returned by shell_start." },
                    "data": { "type": "string", "description": "Text to write. A newline is added if missing." }
                }),
                vec!["process_id", "data"],
            ),
        ),
        (
            "shell_terminate",
            "Stop a background process started with shell_start.",
            object(
                json!({ "process_id": { "type": "string", "description": "Id returned by shell_start." } }),
                vec!["process_id"],
            ),
        ),
        (
            "git_status",
            "Show `git status --porcelain` with branch information for the workspace.",
            object(
                json!({ "cwd": path_property("Repository directory. Defaults to the workspace root.") }),
                vec![],
            ),
        ),
        (
            "git_diff",
            "Show the working-tree diff, or the staged diff when staged is true.",
            object(
                json!({
                    "staged": { "type": "boolean", "description": "Diff the index instead of the working tree." },
                    "path": path_property("Limit the diff to one path."),
                    "cwd": path_property("Repository directory. Defaults to the workspace root.")
                }),
                vec![],
            ),
        ),
        (
            "request_permission",
            "Ask the user for access you do not have yet: a path outside the workspace, network \
             access, or host execution. Explain why in reason. Prefer this over guessing when a \
             call was refused.",
            object(
                json!({
                    "reason": { "type": "string", "description": "Why the access is needed, in one or two sentences." },
                    "paths": {
                        "type": "array",
                        "description": "Paths outside the workspace you need.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": { "type": "string" },
                                "write": { "type": "boolean" }
                            },
                            "required": ["path"],
                            "additionalProperties": false
                        }
                    },
                    "network": { "type": "boolean", "description": "Whether outbound network access is needed." },
                    "host_execution": { "type": "boolean", "description": "Whether a command must run outside the sandbox." }
                }),
                vec!["reason"],
            ),
        ),
    ]
}

/// System prompt for an agent session. The application owns the agent's
/// operating rules; the runtime only carries them to the model.
pub fn system_prompt(
    session: &AgentSessionRecord,
    sandbox: &SandboxBackendCapabilities,
    tool_names: &[String],
) -> String {
    let workspace = session
        .workspace_path
        .as_deref()
        .unwrap_or("(no workspace selected)");
    let sandbox_line = if sandbox.isolated {
        format!(
            "Commands run inside a {} sandbox: writes are limited to the workspace and \
             credential paths are unreadable.",
            sandbox.backend
        )
    } else {
        format!(
            "There is no OS sandbox on this host ({}). Every command therefore runs with the \
             user's full privileges, so the user is asked before anything executes.",
            sandbox.detail
        )
    };
    let permission_line = match session.permission_mode {
        crate::agent_types::AgentPermissionMode::Ask => {
            "The user approves writes, command execution, network access, and anything outside \
             the workspace. Expect some calls to come back refused; adapt instead of retrying \
             the same call."
        }
        crate::agent_types::AgentPermissionMode::SandboxOnly => {
            "Sandboxed work runs without prompts. Anything that would leave the workspace or the \
             sandbox is refused outright, so do not plan around host access."
        }
        crate::agent_types::AgentPermissionMode::SkipPermissions => {
            "The user disabled prompting for this session. Be correspondingly careful: prefer \
             reversible steps, and say what you are about to do before destructive changes."
        }
    };

    format!(
        "You are Brazier's coding and system agent, running locally on the user's machine.\n\n\
         Workspace: {workspace}\n\
         Platform: {} ({})\n\
         {sandbox_line}\n\
         {permission_line}\n\n\
         How to work:\n\
         - Investigate before changing anything. Read the files you are about to edit.\n\
         - Prefer fs_patch over fs_write for existing files, and keep edits minimal.\n\
         - Run the project's own checks when you change code, and report failures honestly \
           instead of claiming success.\n\
         - Tool output is truncated when long; narrow your query rather than re-reading \
           everything.\n\
         - Never try to read credential files, ssh keys, or Brazier's own data directory. Those \
           calls are refused and the attempt is recorded.\n\
         - When you need access you do not have, call request_permission and explain why.\n\
         - Stop and summarize when the task is done: what changed, what you ran, and what is \
           still open.\n\n\
         Available tools: {}.",
        std::env::consts::OS,
        std::env::consts::ARCH,
        tool_names.join(", ")
    )
}

/// Names of every tool the daemon can execute.
pub fn tool_names() -> Vec<String> {
    TOOL_SPECS.iter().map(|spec| spec.name.to_owned()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_types::{AgentPermissionMode, AgentPermissionSettings};

    fn session(mode: AgentPermissionMode) -> AgentSessionRecord {
        AgentSessionRecord {
            id: "s1".to_owned(),
            title: "task".to_owned(),
            workspace_path: Some("/ws".to_owned()),
            model: "gguf:test".to_owned(),
            runtime_id: "pi".to_owned(),
            permission_mode: mode,
            permission_settings: AgentPermissionSettings::default(),
            enabled_tools: None,
            last_run_status: "idle".to_owned(),
            compaction: None,
            runtime_metadata: None,
            created_at: "now".to_owned(),
            updated_at: "now".to_owned(),
        }
    }

    #[test]
    fn every_definition_has_a_policy_spec_and_vice_versa() {
        let catalog = definitions();
        let names: Vec<&str> = catalog["data"]
            .as_array()
            .expect("array")
            .iter()
            .map(|entry| entry["name"].as_str().expect("name"))
            .collect();
        for spec in TOOL_SPECS {
            assert!(
                names.contains(&spec.name),
                "{} has a policy spec but no definition",
                spec.name
            );
        }
        assert_eq!(
            names.len(),
            TOOL_SPECS.len(),
            "definitions and policy specs must stay in step"
        );
    }

    #[test]
    fn schemas_are_objects_with_declared_properties() {
        for entry in definitions()["data"].as_array().expect("array") {
            let schema = &entry["input_schema"];
            assert_eq!(schema["type"], "object", "{}", entry["name"]);
            assert!(schema["properties"].is_object(), "{}", entry["name"]);
            assert!(schema["required"].is_array(), "{}", entry["name"]);
            // Weak models invent arguments; refusing extras keeps repair honest.
            assert_eq!(schema["additionalProperties"], false, "{}", entry["name"]);
        }
    }

    #[test]
    fn destructive_tools_are_marked_as_such() {
        let catalog = definitions();
        let risk_of = |name: &str| {
            catalog["data"]
                .as_array()
                .unwrap()
                .iter()
                .find(|entry| entry["name"] == name)
                .map(|entry| entry["risk"].as_str().unwrap().to_owned())
                .expect("tool present")
        };
        assert_eq!(risk_of("fs_delete"), "destructive");
        assert_eq!(risk_of("fs_move"), "destructive");
        assert_eq!(risk_of("fs_read"), "read");
        assert_eq!(risk_of("shell_run"), "execute");
    }

    #[test]
    fn the_prompt_is_honest_when_no_sandbox_exists() {
        let capabilities = SandboxBackendCapabilities {
            backend: "none".to_owned(),
            isolated: false,
            filesystem_scoping: false,
            network_isolation: false,
            process_isolation: false,
            profiles: Vec::new(),
            detail: "No sandbox: install bubblewrap.".to_owned(),
            program: None,
        };
        let prompt = system_prompt(
            &session(AgentPermissionMode::Ask),
            &capabilities,
            &tool_names(),
        );
        assert!(prompt.contains("no OS sandbox"));
        assert!(prompt.contains("install bubblewrap"));
        assert!(prompt.contains("/ws"));
    }

    #[test]
    fn the_prompt_reflects_the_permission_mode() {
        let capabilities = SandboxBackendCapabilities {
            backend: "seatbelt".to_owned(),
            isolated: true,
            filesystem_scoping: true,
            network_isolation: true,
            process_isolation: false,
            profiles: Vec::new(),
            detail: "Seatbelt".to_owned(),
            program: None,
        };
        let names = tool_names();
        let sandbox_only = system_prompt(
            &session(AgentPermissionMode::SandboxOnly),
            &capabilities,
            &names,
        );
        assert!(sandbox_only.contains("refused outright"));
        let skip = system_prompt(
            &session(AgentPermissionMode::SkipPermissions),
            &capabilities,
            &names,
        );
        assert!(skip.contains("disabled prompting"));
    }
}

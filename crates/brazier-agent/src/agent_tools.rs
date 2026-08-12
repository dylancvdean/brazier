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
};
use brazier_protocol::agent_types::{AgentPermissionMode, AgentSessionRecord};

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
                // Power tools need real egress or user toolchain access, so the
                // policy runs them as host actions (see `agent_policy`).
                "default_environment": if POWER_TOOLS.contains(&name) {
                    "host"
                } else {
                    "sandbox"
                },
                // Power tools are the optional "Powerful" mode surface. Simple
                // mode never exposes them; the Manage → Agent page toggles them.
                "power_tool": POWER_TOOLS.contains(&name),
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
        "doc_read" => "Read document",
        "fs_stat" => "Stat",
        "fs_search" => "Search",
        "fs_find" => "Find paths",
        "fs_read_many" => "Read files",
        "fs_tree" => "Tree",
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
        "git_log" => "Git log",
        "git_show" => "Git show",
        "git_blame" => "Git blame",
        "git_grep" => "Git grep",
        "git_branch" => "Git branches",
        "git_tags" => "Git tags",
        "git_worktree" => "Git worktrees",
        "git_diff_check" => "Git diff check",
        "git_remote" => "Git remotes",
        "project_test" => "Run tests",
        "project_build" => "Build project",
        "project_lint" => "Lint project",
        "project_typecheck" => "Type-check project",
        "project_format" => "Format check",
        "env_info" => "Environment",
        "process_list" => "Processes",
        "code_symbols" => "Symbols",
        "request_permission" => "Request access",
        "spawn_subagent" => "Spawn subagent",
        "web_search" => "Search web",
        "web_fetch" => "Fetch URL",
        "lsp_diagnostics" => "LSP diagnostics",
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

type RawDefinition = (&'static str, &'static str, Value);

fn raw_definitions() -> Vec<RawDefinition> {
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
            "doc_read",
            "Read a PDF, RTF, DOC, or DOCX in the workspace — it does not accept URLs; for a \
             PDF link use web_fetch first, then pass the document id it returns. For PDFs, pass \
             start_page/end_page (default: first 3 pages) or set render_pages to true to get \
             page images for scanned documents or layout-sensitive reading. For RTF/DOC/DOCX, \
             pass start_line/end_line. Prefer this over shelling out to pdftotext.",
            object(
                json!({
                    "path": path_property(workspace_relative),
                    "start_page": { "type": "integer", "description": "First PDF page, 1-based. Default 1.", "minimum": 1 },
                    "end_page": { "type": "integer", "description": "Last PDF page, inclusive. Defaults to start_page + 2.", "minimum": 1 },
                    "start_line": { "type": "integer", "description": "First line for non-PDF documents, 1-based.", "minimum": 1 },
                    "end_line": { "type": "integer", "description": "Last line for non-PDF documents, inclusive.", "minimum": 1 },
                    "render_pages": { "type": "boolean", "description": "Render PDF pages as images instead of extracting text. Max 4 pages." }
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
            "fs_find",
            "Find files and directories by a glob pattern. Results are workspace-relative, \
             symlinks that leave the workspace are skipped, and common generated directories \
             are ignored. Prefer this over shelling out to find.",
            object(
                json!({
                    "pattern": { "type": "string", "description": "Glob such as `src/**/*.rs`, `*.toml`, or `tests` ." },
                    "path": path_property("Directory to search. Defaults to the workspace root."),
                    "kind": { "type": "string", "enum": ["any", "file", "directory"], "description": "Filter result type. Default any." },
                    "max_results": { "type": "integer", "description": "Maximum results. Default 200.", "minimum": 1, "maximum": 500 }
                }),
                vec!["pattern"],
            ),
        ),
        (
            "fs_read_many",
            "Read several small text files in one call. Each file is labeled and line-numbered; \
             use fs_read for a large file or a precise range.",
            object(
                json!({
                    "paths": { "type": "array", "description": "Workspace-relative files, up to 16.", "items": path_property("File to read."), "minItems": 1, "maxItems": 16 },
                    "max_bytes_each": { "type": "integer", "description": "Per-file byte limit. Default 131072.", "minimum": 1024, "maximum": 524288 }
                }),
                vec!["paths"],
            ),
        ),
        (
            "fs_tree",
            "Render a compact directory tree with files, directories, and sizes. Generated and \
             vendored directories are skipped; use fs_list when you need a single directory's \
             detailed listing.",
            object(
                json!({
                    "path": path_property("Directory to render. Defaults to the workspace root."),
                    "depth": { "type": "integer", "description": "Levels to descend, 1-6. Default 3.", "minimum": 1, "maximum": 6 },
                    "max_entries": { "type": "integer", "description": "Maximum entries. Default 500.", "minimum": 1, "maximum": 2000 }
                }),
                vec![],
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
            "git_log",
            "Show concise commit history, optionally limited to a path. This is read-only and \
             includes commit ids, decorations, authors, dates, and subjects.",
            object(
                json!({
                    "max_count": { "type": "integer", "description": "Number of commits. Default 20.", "minimum": 1, "maximum": 100 },
                    "path": path_property("Limit history to one path."),
                    "cwd": path_property("Repository directory. Defaults to the workspace root.")
                }),
                vec![],
            ),
        ),
        (
            "git_show",
            "Show a commit, tag, or object with a bounded patch. Use this to inspect history \
             without constructing an arbitrary shell command.",
            object(
                json!({
                    "revision": { "type": "string", "description": "Commit, tag, or object name, e.g. `HEAD~2`." },
                    "path": path_property("Limit the object diff to one path."),
                    "stat_only": { "type": "boolean", "description": "Show only the change summary. Default false." },
                    "cwd": path_property("Repository directory. Defaults to the workspace root.")
                }),
                vec!["revision"],
            ),
        ),
        (
            "git_blame",
            "Show line-by-line commit attribution for a file, optionally narrowed to a line \
             range.",
            object(
                json!({
                    "path": path_property("File to attribute."),
                    "start_line": { "type": "integer", "minimum": 1 },
                    "end_line": { "type": "integer", "minimum": 1 },
                    "cwd": path_property("Repository directory. Defaults to the workspace root.")
                }),
                vec!["path"],
            ),
        ),
        (
            "git_grep",
            "Search tracked files with Git's text-aware grep and return matching lines with \
             paths and line numbers.",
            object(
                json!({
                    "query": { "type": "string", "description": "Literal or regular-expression pattern accepted by git grep." },
                    "path": path_property("Limit the search to one path."),
                    "max_count": { "type": "integer", "description": "Maximum matches per file. Default 20.", "minimum": 1, "maximum": 100 },
                    "cwd": path_property("Repository directory. Defaults to the workspace root.")
                }),
                vec!["query"],
            ),
        ),
        (
            "git_branch",
            "List local and remote branches with their upstream and latest commit.",
            object(
                json!({ "cwd": path_property("Repository directory. Defaults to the workspace root.") }),
                vec![],
            ),
        ),
        (
            "git_tags",
            "List tags sorted by creation date, with the referenced object and subject.",
            object(
                json!({ "max_count": { "type": "integer", "minimum": 1, "maximum": 200 }, "cwd": path_property("Repository directory. Defaults to the workspace root.") }),
                vec![],
            ),
        ),
        (
            "git_worktree",
            "List worktrees and their branch or detached commit state.",
            object(
                json!({ "cwd": path_property("Repository directory. Defaults to the workspace root.") }),
                vec![],
            ),
        ),
        (
            "git_diff_check",
            "Check the working tree or index for whitespace errors and malformed patches.",
            object(
                json!({ "staged": { "type": "boolean", "description": "Check the index instead of the working tree." }, "cwd": path_property("Repository directory. Defaults to the workspace root.") }),
                vec![],
            ),
        ),
        (
            "git_remote",
            "List configured Git remotes and their fetch/push URLs without exposing credentials \
             embedded in URLs.",
            object(
                json!({ "cwd": path_property("Repository directory. Defaults to the workspace root.") }),
                vec![],
            ),
        ),
        (
            "project_test",
            "Run the repository's conventional test command, detected from Cargo, Node, Python, \
             or Go project files. This never installs dependencies.",
            object_with_reason(
                json!({ "cwd": path_property("Project directory. Defaults to the workspace root."), "timeout_ms": { "type": "integer", "minimum": 1000, "maximum": 600000 } }),
                vec![],
            ),
        ),
        (
            "project_build",
            "Run the repository's conventional build or compile check, detected from Cargo, \
             Node, Python, or Go project files. This never installs dependencies.",
            object_with_reason(
                json!({ "cwd": path_property("Project directory. Defaults to the workspace root."), "timeout_ms": { "type": "integer", "minimum": 1000, "maximum": 600000 } }),
                vec![],
            ),
        ),
        (
            "project_lint",
            "Run the repository's configured lint command when it can be detected from Cargo, \
             Node, Python, or Go project files. Missing linters are reported clearly.",
            object_with_reason(
                json!({ "cwd": path_property("Project directory. Defaults to the workspace root."), "timeout_ms": { "type": "integer", "minimum": 1000, "maximum": 600000 } }),
                vec![],
            ),
        ),
        (
            "project_typecheck",
            "Run a non-mutating type-check or compiler check detected from the project files.",
            object_with_reason(
                json!({ "cwd": path_property("Project directory. Defaults to the workspace root."), "timeout_ms": { "type": "integer", "minimum": 1000, "maximum": 600000 } }),
                vec![],
            ),
        ),
        (
            "project_format",
            "Check formatting without changing files, using the repository's conventional \
             formatter for Rust, Node, Python, or Go.",
            object_with_reason(
                json!({ "cwd": path_property("Project directory. Defaults to the workspace root."), "timeout_ms": { "type": "integer", "minimum": 1000, "maximum": 600000 } }),
                vec![],
            ),
        ),
        (
            "env_info",
            "Report non-secret host and workspace facts useful for choosing a build command: \
             platform, architecture, shell, tool versions, and detected project manifests. \
             Credential values are never returned.",
            object(
                json!({ "cwd": path_property("Project directory. Defaults to the workspace root.") }),
                vec![],
            ),
        ),
        (
            "process_list",
            "List processes owned by the current user with pid, parent pid, CPU, memory, and \
             command. Use this to diagnose a stuck build or server.",
            object_with_reason(
                json!({ "match": { "type": "string", "description": "Optional case-insensitive substring filter." } }),
                vec![],
            ),
        ),
        (
            "code_symbols",
            "Extract a compact symbol index from source files using common declaration forms. \
             This is a fast fallback when no language server is available; it is not a semantic \
             cross-reference engine.",
            object(
                json!({
                    "path": path_property("File or directory to index. Defaults to the workspace root."),
                    "name_glob": { "type": "string", "description": "Optional filename filter, e.g. `*.rs`." },
                    "max_symbols": { "type": "integer", "minimum": 1, "maximum": 1000 }
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
        (
            "spawn_subagent",
            "Start one or more sandboxed child agents and wait for their answers. For \
             independent work, pass several tasks in `prompts` in a single call (or emit \
             multiple spawn_subagent tool calls in the same turn) so they run concurrently — \
             do not wait for one child before starting the next. Nested subagents are not \
             allowed.",
            object_with_reason(
                json!({
                    "prompt": {
                        "type": "string",
                        "description": "Task for a single child agent. Prefer `prompts` when you have several independent tasks."
                    },
                    "prompts": {
                        "type": "array",
                        "description": "Independent tasks to run concurrently in this one call, up to the session max. Prefer this over calling spawn_subagent once per task across turns.",
                        "items": { "type": "string" },
                        "minItems": 1,
                        "maxItems": 8
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional Brazier model id. Defaults to the parent profile's subagent model, then this model."
                    }
                }),
                vec![],
            ),
        ),
        // Power tools (Powerful mode). They run as host actions under the
        // policy broker, so each call is approval-gated.
        (
            "web_search",
            "Search the web for up-to-date information and return a ranked list of results \
             with titles, URLs, and short snippets. Use it for facts that changed recently or \
             are outside the workspace. Runs on DuckDuckGo by default (keyless, rate-limited); \
             a Brave API key can be configured in Manage → Engine → Web search for a higher \
             rate limit.",
            object(
                json!({
                    "query": {
                        "type": "string",
                        "description": "The search query, phrased like you would type into a search engine."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return. Default 5.",
                        "minimum": 1,
                        "maximum": 10
                    },
                    "region": {
                        "type": "string",
                        "description": "Optional region/locale code such as `us-en`, `de-de`, or `wt-wt`. DuckDuckGo only."
                    },
                    "safesearch": {
                        "type": "string",
                        "enum": ["moderate", "strict", "off"],
                        "description": "Filtering level. Default moderate."
                    }
                }),
                vec!["query"],
            ),
        ),
        (
            "web_fetch",
            "Fetch a URL and return its text content. Useful for reading a page the model \
             cannot reach directly, such as documentation or an API reference. A PDF URL must \
             go through web_fetch, not doc_read: fetching a PDF stores it and returns a document \
             id you then pass to doc_read to read by page.",
            object(
                json!({
                    "url": {
                        "type": "string",
                        "description": "Absolute http(s) URL to fetch."
                    },
                    "start": {
                        "type": "integer",
                        "description": "Character offset to start from; use the count in the previous result to page through long pages. Default 0.",
                        "minimum": 0
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return. Default 12000.",
                        "minimum": 500,
                        "maximum": 50000
                    }
                }),
                vec!["url"],
            ),
        ),
        (
            "lsp_diagnostics",
            "Run the language server for a file in the workspace and return its diagnostics \
             (errors and warnings) with severity, source line, and message.",
            object(
                json!({
                    "path": path_property("File to analyze, relative to the workspace root."),
                    "include_warnings": {
                        "type": "boolean",
                        "description": "Include warnings, not just errors. Default true."
                    }
                }),
                vec!["path"],
            ),
        ),
    ]
}

/// Names of the optional "Powerful" mode tools. Simple mode never exposes them.
pub const POWER_TOOLS: &[&str] = &[
    "fs_find",
    "fs_read_many",
    "fs_tree",
    "git_log",
    "git_show",
    "git_blame",
    "git_grep",
    "git_branch",
    "git_tags",
    "git_worktree",
    "git_diff_check",
    "git_remote",
    "project_test",
    "project_build",
    "project_lint",
    "project_typecheck",
    "project_format",
    "env_info",
    "process_list",
    "code_symbols",
    "web_search",
    "web_fetch",
    "lsp_diagnostics",
];

/// Names of every power tool, for the daemon's mode-aware defaults.
pub fn power_tool_names() -> Vec<String> {
    POWER_TOOLS.iter().map(|name| (*name).to_owned()).collect()
}

/// The editable application default. Each shortcut is expanded from live
/// session state immediately before the prompt is handed to the runtime.
pub const DEFAULT_SYSTEM_PROMPT_TEMPLATE: &str = "{identity}\n\n\
{workspace}\n\
{system_info}\n\
{permissions}\n\n\
{working_rules}\n\n\
{tools}";

/// Generated, read-only pieces available to workspace prompt templates.
pub fn system_prompt_components(
    session: &AgentSessionRecord,
    sandbox: &SandboxBackendCapabilities,
    tool_names: &[String],
) -> Vec<(&'static str, String)> {
    let workspace = session
        .workspace_path
        .as_deref()
        .unwrap_or("(no workspace selected)");
    let worktree = crate::agent_worktree::worktree_from_metadata(session.runtime_metadata.as_ref());
    let workspace_line = if let Some(info) = &worktree {
        format!(
            "Workspace: {workspace}\n\
             This session is confined to a git worktree branched from {} on `{}`. \
             Edit and run commands here; the user's original checkout is untouched.",
            info.source_path, info.branch
        )
    } else {
        format!("Workspace: {workspace}")
    };
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
        AgentPermissionMode::Ask => {
            "The user approves writes, command execution, network access, and anything outside \
             the workspace. Expect some calls to come back refused; adapt instead of retrying \
             the same call."
        }
        AgentPermissionMode::SandboxOnly => {
            "Sandboxed work runs without prompts. Anything that would leave the workspace or the \
             sandbox is refused outright, so do not plan around host access."
        }
        AgentPermissionMode::SkipPermissions => {
            "The user disabled prompting for this session. Be correspondingly careful: prefer \
             reversible steps, and say what you are about to do before destructive changes."
        }
    };

    let spawn_line = if tool_names.iter().any(|name| name == "spawn_subagent") {
        "- Use spawn_subagent for focused parallel work. When tasks are independent, pass them \
           together in `prompts` (or issue several spawn_subagent calls in the same turn) so \
           children run concurrently — do not serialize one child after another across turns.\n"
    } else {
        ""
    };
    vec![
        (
            "identity",
            "You are Brazier's coding and system agent. Tools and paths belong to the Brazier \
             daemon host, which may be a different machine from the desktop client the user is \
             holding. Never describe a daemon path or action as client-local unless the supplied \
             execution location says it is."
                .to_owned(),
        ),
        ("workspace", workspace_line),
        (
            "system_info",
            format!(
                "Platform: {} ({})\n{sandbox_line}",
                std::env::consts::OS,
                std::env::consts::ARCH
            ),
        ),
        ("permissions", permission_line.to_owned()),
        (
            "working_rules",
            format!(
                "How to work:\n\
                 - Investigate before changing anything. Read the files you are about to edit.\n\
                 - Use doc_read for PDF/RTF/DOC/DOCX (by path, never a URL — fetch PDF links \
                    with web_fetch and pass the document id it returns); use fs_read for \
                    ordinary text files.\n\
                 - Prefer fs_patch over fs_write for existing files, and keep edits minimal.\n\
                 - Run the project's own checks when you change code, and report failures \
                   honestly instead of claiming success.\n\
                 - Tool output is truncated when long; narrow your query rather than re-reading \
                   everything.\n\
                 - Never try to read credential files, ssh keys, or Brazier's own data directory. \
                   Those calls are refused and the attempt is recorded.\n\
                 - When you need access you do not have, call request_permission and explain why.\n\
                 {spawn_line}\
                 - Stop and summarize when the task is done: what changed, what you ran, and what \
                   is still open."
            ),
        ),
        (
            "tools",
            format!("Available tools: {}.", tool_names.join(", ")),
        ),
    ]
}

/// Expand known `{shortcut}` values once. Unknown shortcuts remain visible so
/// a typo in an editable prompt is not silently discarded.
pub fn render_system_prompt(template: &str, components: &[(&str, String)]) -> String {
    let mut rendered = String::with_capacity(template.len());
    let mut cursor = 0;
    while cursor < template.len() {
        let remaining = &template[cursor..];
        let replacement = components.iter().find(|(name, _)| {
            remaining.starts_with('{')
                && remaining.strip_prefix('{').is_some_and(|tail| {
                    tail.starts_with(name) && tail[name.len()..].starts_with('}')
                })
        });
        if let Some((name, value)) = replacement {
            rendered.push_str(value);
            cursor += name.len() + 2;
        } else {
            let character = remaining
                .chars()
                .next()
                .expect("cursor is within the template");
            rendered.push(character);
            cursor += character.len_utf8();
        }
    }
    rendered
}

/// Effective system prompt for an agent session. The application owns the
/// operating rules; the runtime only carries them to the model.
pub fn system_prompt(
    session: &AgentSessionRecord,
    sandbox: &SandboxBackendCapabilities,
    tool_names: &[String],
) -> String {
    let components = system_prompt_components(session, sandbox, tool_names);
    render_system_prompt(DEFAULT_SYSTEM_PROMPT_TEMPLATE, &components)
}

/// Names of every tool the daemon can execute.
pub fn tool_names() -> Vec<String> {
    TOOL_SPECS.iter().map(|spec| spec.name.to_owned()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use brazier_protocol::agent_types::{AgentPermissionMode, AgentPermissionSettings};

    fn session(mode: AgentPermissionMode) -> AgentSessionRecord {
        AgentSessionRecord {
            id: "s1".to_owned(),
            owner_client_id: "owner".to_owned(),
            title: "task".to_owned(),
            workspace_path: Some("/ws".to_owned()),
            model: "gguf:test".to_owned(),
            runtime_id: brazier_protocol::agent_types::AGENT_RUNTIME_SIMPLE.to_owned(),
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
    fn powerful_tools_are_cataloged_as_host_actions() {
        let catalog = definitions();
        let entries = catalog["data"].as_array().expect("array");
        for name in POWER_TOOLS {
            let entry = entries
                .iter()
                .find(|entry| entry["name"] == *name)
                .unwrap_or_else(|| panic!("missing power tool {name}"));
            assert_eq!(entry["power_tool"], true, "{name}");
            assert_eq!(entry["default_environment"], "host", "{name}");
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
        assert_eq!(risk_of("spawn_subagent"), "execute");
    }

    #[test]
    fn the_prompt_is_honest_when_no_sandbox_exists() {
        let capabilities = SandboxBackendCapabilities {
            backend: "none".to_owned(),
            isolated: false,
            sandboxed_execution: false,
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
        assert!(prompt.contains("daemon host"));
        assert!(!prompt.contains("running locally on the user's machine"));
    }

    #[test]
    fn the_prompt_reflects_the_permission_mode() {
        let capabilities = SandboxBackendCapabilities {
            backend: "seatbelt".to_owned(),
            isolated: true,
            sandboxed_execution: true,
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

    #[test]
    fn prompt_templates_expand_known_components_and_preserve_unknown_ones() {
        let components = vec![
            ("workspace", "Workspace: /ws".to_owned()),
            ("tools", "Available tools: fs_read.".to_owned()),
        ];
        assert_eq!(
            render_system_prompt(
                "Work here:\n{workspace}\n{tools}\nKeep {project_rule}.\n\
                 JSON: {\"tools\": \"{tools}\"}",
                &components
            ),
            "Work here:\nWorkspace: /ws\nAvailable tools: fs_read.\nKeep {project_rule}.\n\
             JSON: {\"tools\": \"Available tools: fs_read.\"}"
        );
    }

    #[test]
    fn the_default_prompt_is_composed_from_every_shortcut() {
        let capabilities = SandboxBackendCapabilities {
            backend: "seatbelt".to_owned(),
            isolated: true,
            sandboxed_execution: true,
            filesystem_scoping: true,
            network_isolation: true,
            process_isolation: false,
            profiles: Vec::new(),
            detail: "Seatbelt".to_owned(),
            program: None,
        };
        let names = tool_names();
        let components =
            system_prompt_components(&session(AgentPermissionMode::Ask), &capabilities, &names);
        for (name, _) in &components {
            assert!(DEFAULT_SYSTEM_PROMPT_TEMPLATE.contains(&format!("{{{name}}}")));
        }
        let prompt = system_prompt(&session(AgentPermissionMode::Ask), &capabilities, &names);
        assert!(!prompt.contains("{identity}"));
        assert!(prompt.contains("You are Brazier's coding and system agent"));
        assert!(prompt.contains("Available tools:"));
    }
}

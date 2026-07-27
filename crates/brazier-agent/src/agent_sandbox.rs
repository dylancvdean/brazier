//! Sandbox backends for agent tool execution.
//!
//! A backend wraps a command so it can only reach the session workspace and a
//! scratch directory. Two real backends ship: macOS Seatbelt (`sandbox-exec`)
//! and Linux Bubblewrap (`bwrap`). When neither exists the backend reports
//! itself as `none` with `isolated: false`; the policy layer refuses to treat
//! that as a sandbox, and the UI shows the caveat verbatim. Nothing in this
//! module may claim isolation it did not apply.

use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
};

use brazier_protocol::agent_types::SandboxDescription;

/// Isolation profiles a session can run under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxProfile {
    /// Read the system, write only the workspace and scratch. No network.
    Workspace,
    /// `workspace` plus outbound network.
    WorkspaceNetwork,
    /// Read the workspace, write nothing.
    ReadOnly,
    /// Read-only plus network, for log and system inspection.
    Diagnostics,
}

impl SandboxProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Workspace => "workspace",
            Self::WorkspaceNetwork => "workspace-network",
            Self::ReadOnly => "read-only",
            Self::Diagnostics => "diagnostics",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "workspace" => Some(Self::Workspace),
            "workspace-network" => Some(Self::WorkspaceNetwork),
            "read-only" => Some(Self::ReadOnly),
            "diagnostics" => Some(Self::Diagnostics),
            _ => None,
        }
    }

    pub fn allows_network(self) -> bool {
        matches!(self, Self::WorkspaceNetwork | Self::Diagnostics)
    }

    pub fn allows_workspace_writes(self) -> bool {
        matches!(self, Self::Workspace | Self::WorkspaceNetwork)
    }
}

/// Which mechanism is available on this host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxBackendKind {
    /// macOS `sandbox-exec`.
    Seatbelt,
    /// Linux `bwrap`.
    Bubblewrap,
    /// No OS-level isolation available.
    None,
}

impl SandboxBackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Seatbelt => "seatbelt",
            Self::Bubblewrap => "bubblewrap",
            Self::None => "none",
        }
    }

    pub fn isolated(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// What the detected backend can enforce. Reported to the UI as-is.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SandboxBackendCapabilities {
    pub backend: String,
    pub isolated: bool,
    pub filesystem_scoping: bool,
    pub network_isolation: bool,
    pub process_isolation: bool,
    pub profiles: Vec<String>,
    pub detail: String,
    /// Program that provides the isolation, when there is one.
    pub program: Option<String>,
}

/// A configured sandbox for one session.
#[derive(Debug, Clone)]
pub struct SandboxBackend {
    kind: SandboxBackendKind,
    program: Option<PathBuf>,
    detail: String,
}

/// Everything a wrapped command needs to know about its jail.
#[derive(Debug, Clone)]
pub struct SandboxRequest<'a> {
    pub profile: SandboxProfile,
    /// Canonical workspace root on the host.
    pub workspace: &'a Path,
    /// Writable scratch directory (temp files, artifacts the tool produces).
    pub scratch: &'a Path,
    /// Working directory for the command; must be inside the workspace.
    pub cwd: &'a Path,
}

/// A command ready to spawn, with isolation and environment already applied.
#[derive(Debug, Clone)]
pub struct WrappedCommand {
    pub program: OsString,
    pub args: Vec<OsString>,
    pub env: BTreeMap<String, String>,
    pub description: SandboxDescription,
}

/// Environment variable names that never reach a tool, matched
/// case-insensitively as substrings.
const SECRET_ENV_PATTERNS: &[&str] = &[
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "APIKEY",
    "API_KEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
    "SESSION_KEY",
    "AUTH",
    "COOKIE",
    "LICENSE_KEY",
];

/// Variables dropped by exact name even though they do not match a pattern.
const DENIED_ENV_NAMES: &[&str] = &[
    "AWS_ACCESS_KEY_ID",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "SSH_AUTH_SOCK",
    "SSH_AGENT_PID",
    "GPG_AGENT_INFO",
    "NETRC",
    "npm_config_registry",
    "PIP_INDEX_URL",
];

/// Paths under the user's home that a sandboxed command must not read.
/// Kept in one place so both backends deny the same set.
const SECRET_HOME_PATHS: &[&str] = &[
    ".ssh",
    ".aws",
    ".gnupg",
    ".kube",
    ".netrc",
    ".git-credentials",
    ".npmrc",
    ".pypirc",
    ".config/gh",
    ".config/gcloud",
    ".config/op",
    ".docker/config.json",
    ".cargo/credentials.toml",
    ".cargo/credentials",
    "Library/Keychains",
    "Library/Application Support/Brazier",
    "Library/Cookies",
    ".local/share/keyrings",
    ".mozilla",
    ".config/google-chrome",
];

impl SandboxBackend {
    /// Detect the best backend for this host.
    pub fn detect() -> Self {
        if cfg!(target_os = "macos") {
            let program = PathBuf::from("/usr/bin/sandbox-exec");
            if program.exists() {
                return Self {
                    kind: SandboxBackendKind::Seatbelt,
                    program: Some(program),
                    detail: "macOS Seatbelt confines writes to the workspace and blocks reads of \
                             credential paths."
                        .to_owned(),
                };
            }
            return Self::unavailable(
                "macOS Seatbelt is unavailable: /usr/bin/sandbox-exec is missing.",
            );
        }
        if cfg!(target_os = "linux") {
            if let Some(program) = which("bwrap") {
                return Self {
                    kind: SandboxBackendKind::Bubblewrap,
                    program: Some(program),
                    detail: "Bubblewrap mounts the host read-only, hides the home directory, and \
                             binds the workspace read-write."
                        .to_owned(),
                };
            }
            return Self::unavailable(
                "No sandbox: install bubblewrap (`bwrap`) to isolate agent commands.",
            );
        }
        Self::unavailable("No sandbox backend exists for this platform yet.")
    }

    fn unavailable(detail: &str) -> Self {
        Self {
            kind: SandboxBackendKind::None,
            program: None,
            detail: detail.to_owned(),
        }
    }

    pub fn kind(&self) -> SandboxBackendKind {
        self.kind
    }

    /// True when the backend applies real OS-level isolation.
    pub fn isolated(&self) -> bool {
        self.kind.isolated()
    }

    pub fn capabilities(&self) -> SandboxBackendCapabilities {
        SandboxBackendCapabilities {
            backend: self.kind.as_str().to_owned(),
            isolated: self.isolated(),
            filesystem_scoping: self.isolated(),
            network_isolation: self.isolated(),
            process_isolation: matches!(self.kind, SandboxBackendKind::Bubblewrap),
            profiles: vec![
                SandboxProfile::Workspace.as_str().to_owned(),
                SandboxProfile::WorkspaceNetwork.as_str().to_owned(),
                SandboxProfile::ReadOnly.as_str().to_owned(),
                SandboxProfile::Diagnostics.as_str().to_owned(),
            ],
            detail: self.detail.clone(),
            program: self.program.as_ref().map(|path| path.display().to_string()),
        }
    }

    /// Describe what a sandboxed call would get, without running anything.
    pub fn describe(
        &self,
        profile: SandboxProfile,
        workspace: Option<&Path>,
    ) -> SandboxDescription {
        SandboxDescription {
            backend: self.kind.as_str().to_owned(),
            profile: profile.as_str().to_owned(),
            isolated: self.isolated(),
            network: profile.allows_network(),
            workspace_path: workspace.map(|path| path.display().to_string()),
            detail: self.detail.clone(),
        }
    }

    /// Describe a host (unsandboxed) call. Never claims isolation.
    pub fn describe_host(&self, workspace: Option<&Path>) -> SandboxDescription {
        SandboxDescription {
            backend: "none".to_owned(),
            profile: "host".to_owned(),
            isolated: false,
            network: true,
            workspace_path: workspace.map(|path| path.display().to_string()),
            detail: "Host execution: no sandbox, full user privileges.".to_owned(),
        }
    }

    /// Wrap a command for sandboxed execution.
    pub fn wrap(
        &self,
        request: &SandboxRequest<'_>,
        program: &str,
        args: &[String],
        extra_env: &BTreeMap<String, String>,
    ) -> WrappedCommand {
        let env = filtered_env(extra_env, request);
        let description = self.describe(request.profile, Some(request.workspace));
        match (self.kind, self.program.as_ref()) {
            (SandboxBackendKind::Seatbelt, Some(sandbox_exec)) => {
                let mut wrapped: Vec<OsString> = vec![
                    OsString::from("-p"),
                    OsString::from(seatbelt_profile(request)),
                    OsString::from(program),
                ];
                wrapped.extend(args.iter().map(OsString::from));
                WrappedCommand {
                    program: sandbox_exec.clone().into_os_string(),
                    args: wrapped,
                    env,
                    description,
                }
            }
            (SandboxBackendKind::Bubblewrap, Some(bwrap)) => {
                let mut wrapped: Vec<OsString> = bubblewrap_args(request)
                    .into_iter()
                    .map(OsString::from)
                    .collect();
                wrapped.push(OsString::from("--"));
                wrapped.push(OsString::from(program));
                wrapped.extend(args.iter().map(OsString::from));
                WrappedCommand {
                    program: bwrap.clone().into_os_string(),
                    args: wrapped,
                    env,
                    description,
                }
            }
            _ => WrappedCommand {
                program: OsString::from(program),
                args: args.iter().map(OsString::from).collect(),
                env,
                description,
            },
        }
    }

    /// Build a host command: no isolation, but still a filtered environment.
    pub fn wrap_host(
        &self,
        request: &SandboxRequest<'_>,
        program: &str,
        args: &[String],
        extra_env: &BTreeMap<String, String>,
    ) -> WrappedCommand {
        WrappedCommand {
            program: OsString::from(program),
            args: args.iter().map(OsString::from).collect(),
            env: filtered_env(extra_env, request),
            description: self.describe_host(Some(request.workspace)),
        }
    }
}

/// Look up an executable on `PATH`.
fn which(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|directory| directory.join(program))
        .find(|candidate| candidate.is_file())
}

/// Build the child environment: inherit, drop secrets, then apply the caller's
/// explicit additions. Explicit additions are the only way a secret-looking
/// name can reach a tool.
pub fn filtered_env(
    extra: &BTreeMap<String, String>,
    request: &SandboxRequest<'_>,
) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    for (key, value) in std::env::vars() {
        if is_secret_env(&key) {
            continue;
        }
        // Daemon-private configuration, including its bearer token.
        if key.starts_with("BRAZIER_") {
            continue;
        }
        env.insert(key, value);
    }
    env.insert("PWD".to_owned(), request.cwd.display().to_string());
    env.insert("TMPDIR".to_owned(), request.scratch.display().to_string());
    env.insert("TEMP".to_owned(), request.scratch.display().to_string());
    env.insert("TMP".to_owned(), request.scratch.display().to_string());
    // Interactive pagers and editors hang a captured pipe forever.
    env.insert("PAGER".to_owned(), "cat".to_owned());
    env.insert("GIT_PAGER".to_owned(), "cat".to_owned());
    env.insert("GIT_TERMINAL_PROMPT".to_owned(), "0".to_owned());
    env.insert("CI".to_owned(), "1".to_owned());
    env.insert("TERM".to_owned(), "dumb".to_owned());
    env.insert("BRAZIER_AGENT".to_owned(), "1".to_owned());
    for (key, value) in extra {
        env.insert(key.clone(), value.clone());
    }
    env
}

/// True when a variable name looks like it carries a credential.
pub fn is_secret_env(name: &str) -> bool {
    if DENIED_ENV_NAMES.iter().any(|denied| denied == &name) {
        return true;
    }
    let upper = name.to_ascii_uppercase();
    SECRET_ENV_PATTERNS
        .iter()
        .any(|pattern| upper.contains(pattern))
}

fn home_directory() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
}

/// Absolute paths a sandboxed command must not read.
///
/// Both the literal and the canonical form of each path are returned when they
/// differ, so a comparison against either spelling matches. On macOS the same
/// directory is reachable as `/var/...` and `/private/var/...`; missing the
/// second form would leave a hole.
pub fn secret_paths(data_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut push = |path: PathBuf| {
        if let Ok(canonical) = std::fs::canonicalize(&path)
            && canonical != path
        {
            paths.push(canonical);
        }
        paths.push(path);
    };
    if let Some(home) = home_directory() {
        for relative in SECRET_HOME_PATHS {
            push(home.join(relative));
        }
    }
    if let Some(data_dir) = data_dir {
        push(data_dir.to_path_buf());
    }
    paths
}

/// Quote a path for a Seatbelt profile string literal.
fn sbpl_quote(path: &Path) -> String {
    let raw = path.display().to_string();
    let escaped = raw.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

/// Generate a Seatbelt profile.
///
/// Rule order is the whole security argument, because the last matching SBPL
/// rule wins:
///
/// 1. deny everything,
/// 2. allow the broad reads a compiler needs,
/// 3. deny credential paths, overriding those reads,
/// 4. allow writes to the workspace and the session scratch directory last, so
///    they survive step 3 — the scratch directory lives under Brazier's data
///    directory, which step 3 denies wholesale.
///
/// Nothing else on the filesystem is writable. `/tmp` deliberately is not:
/// `TMPDIR` already points at the scratch directory, so a tool that honours it
/// works, and one that hardcodes `/tmp` fails visibly instead of writing
/// outside the jail.
pub fn seatbelt_profile(request: &SandboxRequest<'_>) -> String {
    let mut profile = String::from("(version 1)\n(deny default)\n");
    profile.push_str("(allow process-fork)\n(allow process-exec)\n");
    profile.push_str("(allow signal)\n(allow sysctl-read)\n");
    profile.push_str("(allow mach-lookup)\n(allow ipc-posix-shm)\n");
    profile.push_str("(allow file-read-metadata)\n(allow file-read*)\n");
    // /dev is needed for null, tty, and random; it holds no user content.
    profile.push_str("(allow file-write* (subpath \"/dev\"))\n");
    profile.push_str("(allow file-ioctl (subpath \"/dev\"))\n");

    if request.profile.allows_network() {
        profile.push_str("(allow network-outbound)\n(allow network-bind)\n");
        profile.push_str("(allow system-socket)\n");
    } else {
        profile.push_str("(deny network*)\n");
    }

    // Credential denials come after the broad read allowance.
    for secret in secret_paths(None) {
        profile.push_str(&format!(
            "(deny file-read* file-write* (subpath {}))\n",
            sbpl_quote(&secret)
        ));
    }

    // Writable paths come last so no earlier denial can shadow them.
    if request.profile.allows_workspace_writes() {
        profile.push_str(&format!(
            "(allow file-write* (subpath {}))\n",
            sbpl_quote(request.workspace)
        ));
    }
    profile.push_str(&format!(
        "(allow file-read* file-write* (subpath {}))\n",
        sbpl_quote(request.scratch)
    ));
    profile
}

/// Generate Bubblewrap arguments. Later binds override earlier ones, so the
/// workspace bind comes after the home tmpfs that hides credentials.
pub fn bubblewrap_args(request: &SandboxRequest<'_>) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "--die-with-parent".into(),
        "--new-session".into(),
        "--unshare-ipc".into(),
        "--unshare-uts".into(),
        "--unshare-pid".into(),
        "--ro-bind".into(),
        "/".into(),
        "/".into(),
        "--proc".into(),
        "/proc".into(),
        "--dev".into(),
        "/dev".into(),
        "--tmpfs".into(),
        "/tmp".into(),
    ];

    if let Some(home) = home_directory() {
        // A tmpfs over $HOME hides ssh keys, cloud credentials, and browser
        // profiles in one move.
        args.push("--tmpfs".into());
        args.push(home.display().to_string());
    }

    args.push("--bind".into());
    args.push(request.scratch.display().to_string());
    args.push(request.scratch.display().to_string());

    if request.profile.allows_workspace_writes() {
        args.push("--bind".into());
    } else {
        args.push("--ro-bind".into());
    }
    args.push(request.workspace.display().to_string());
    args.push(request.workspace.display().to_string());

    if !request.profile.allows_network() {
        args.push("--unshare-net".into());
    }

    args.push("--chdir".into());
    args.push(request.cwd.display().to_string());
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(
        profile: SandboxProfile,
        workspace: &'a Path,
        scratch: &'a Path,
    ) -> SandboxRequest<'a> {
        SandboxRequest {
            profile,
            workspace,
            scratch,
            cwd: workspace,
        }
    }

    #[test]
    fn secret_env_names_are_rejected_case_insensitively() {
        assert!(is_secret_env("HF_TOKEN"));
        assert!(is_secret_env("hf_token"));
        assert!(is_secret_env("OPENAI_API_KEY"));
        assert!(is_secret_env("MY_SECRET_THING"));
        assert!(is_secret_env("SSH_AUTH_SOCK"));
        assert!(!is_secret_env("PATH"));
        assert!(!is_secret_env("HOME"));
        assert!(!is_secret_env("CARGO_HOME"));
    }

    #[test]
    fn filtered_env_drops_secrets_and_daemon_config() {
        // SAFETY: single-threaded test process mutating its own environment.
        unsafe {
            std::env::set_var("BRAZIER_TEST_MARKER", "1");
            std::env::set_var("TEST_ACCESS_KEY", "shh");
            std::env::set_var("AGENT_SANDBOX_PLAIN", "fine");
        }
        let workspace = PathBuf::from("/tmp/ws");
        let scratch = PathBuf::from("/tmp/scratch");
        let env = filtered_env(
            &BTreeMap::new(),
            &request(SandboxProfile::Workspace, &workspace, &scratch),
        );
        assert!(!env.contains_key("BRAZIER_TEST_MARKER"));
        assert!(!env.contains_key("TEST_ACCESS_KEY"));
        assert_eq!(
            env.get("AGENT_SANDBOX_PLAIN").map(String::as_str),
            Some("fine")
        );
        // Scratch replaces the host temp directory.
        assert_eq!(env.get("TMPDIR").map(String::as_str), Some("/tmp/scratch"));
        assert_eq!(
            env.get("GIT_TERMINAL_PROMPT").map(String::as_str),
            Some("0")
        );
        unsafe {
            std::env::remove_var("BRAZIER_TEST_MARKER");
            std::env::remove_var("TEST_ACCESS_KEY");
            std::env::remove_var("AGENT_SANDBOX_PLAIN");
        }
    }

    #[test]
    fn explicit_env_overrides_the_secret_filter() {
        let workspace = PathBuf::from("/tmp/ws");
        let scratch = PathBuf::from("/tmp/scratch");
        let mut extra = BTreeMap::new();
        extra.insert("GITHUB_TOKEN".to_owned(), "granted".to_owned());
        let env = filtered_env(
            &extra,
            &request(SandboxProfile::Workspace, &workspace, &scratch),
        );
        assert_eq!(env.get("GITHUB_TOKEN").map(String::as_str), Some("granted"));
    }

    #[test]
    fn seatbelt_profile_denies_network_and_scopes_writes() {
        let workspace = PathBuf::from("/tmp/ws");
        let scratch = PathBuf::from("/tmp/scratch");
        let profile = seatbelt_profile(&request(SandboxProfile::Workspace, &workspace, &scratch));
        assert!(profile.starts_with("(version 1)\n(deny default)"));
        assert!(profile.contains("(deny network*)"));
        assert!(profile.contains("(allow file-write* (subpath \"/tmp/ws\"))"));
        // Credential denials must come after the broad read allowance,
        // because the last matching SBPL rule wins.
        let read_all = profile.find("(allow file-read*)").expect("read rule");
        let deny_secret = profile
            .find("(deny file-read* file-write* (subpath")
            .unwrap_or(usize::MAX);
        assert!(deny_secret > read_all, "secret denials must override reads");
    }

    #[test]
    fn read_only_profile_grants_no_workspace_writes() {
        let workspace = PathBuf::from("/tmp/ws");
        let scratch = PathBuf::from("/tmp/scratch");
        let profile = seatbelt_profile(&request(SandboxProfile::ReadOnly, &workspace, &scratch));
        assert!(!profile.contains("(allow file-write* (subpath \"/tmp/ws\"))"));
        // Scratch stays writable so tools can still produce artifacts.
        assert!(profile.contains("(allow file-read* file-write* (subpath \"/tmp/scratch\"))"));
    }

    #[test]
    fn the_profile_grants_no_blanket_temp_writes() {
        // TMPDIR points at the session scratch directory, so a broad /tmp grant
        // would only widen the jail. A live test caught exactly that escape.
        let workspace = PathBuf::from("/ws");
        let scratch = PathBuf::from("/data/agent/scratch/s1");
        let profile = seatbelt_profile(&request(SandboxProfile::Workspace, &workspace, &scratch));
        assert!(!profile.contains("(allow file-write* (subpath \"/tmp\"))"));
        assert!(!profile.contains("(allow file-write* (subpath \"/private/tmp\"))"));
        assert!(!profile.contains("(subpath \"/private/var/folders\")"));
    }

    #[test]
    fn the_scratch_grant_survives_the_data_directory_denial() {
        // The scratch directory lives inside Brazier's data directory, which is
        // denied wholesale. Because the last matching SBPL rule wins, the grant
        // has to come after that denial or the sandbox cannot write its own
        // temporary files.
        let workspace = PathBuf::from("/ws");
        let scratch = PathBuf::from("/data/agent/scratch/s1");
        let profile = seatbelt_profile(&request(SandboxProfile::Workspace, &workspace, &scratch));
        let last_deny = profile
            .rfind("(deny file-read* file-write* (subpath")
            .expect("credential denials present");
        let scratch_allow = profile
            .find("(allow file-read* file-write* (subpath \"/data/agent/scratch/s1\"))")
            .expect("scratch grant present");
        assert!(
            scratch_allow > last_deny,
            "the scratch grant must come after every denial"
        );
    }

    #[test]
    fn network_profile_allows_outbound() {
        let workspace = PathBuf::from("/tmp/ws");
        let scratch = PathBuf::from("/tmp/scratch");
        let profile = seatbelt_profile(&request(
            SandboxProfile::WorkspaceNetwork,
            &workspace,
            &scratch,
        ));
        assert!(profile.contains("(allow network-outbound)"));
        assert!(!profile.contains("(deny network*)"));
    }

    #[test]
    fn bubblewrap_binds_workspace_after_hiding_home() {
        let workspace = PathBuf::from("/tmp/ws");
        let scratch = PathBuf::from("/tmp/scratch");
        let args = bubblewrap_args(&request(SandboxProfile::Workspace, &workspace, &scratch));
        let joined = args.join(" ");
        assert!(joined.contains("--ro-bind / /"));
        assert!(joined.contains("--unshare-net"));
        assert!(joined.contains("--bind /tmp/ws /tmp/ws"));
        if let Some(home) = home_directory() {
            let tmpfs_home = format!("--tmpfs {}", home.display());
            let home_index = joined.find(&tmpfs_home).expect("home tmpfs");
            let bind_index = joined.find("--bind /tmp/ws").expect("workspace bind");
            assert!(
                bind_index > home_index,
                "workspace must survive the home tmpfs"
            );
        }
    }

    #[test]
    fn read_only_bubblewrap_binds_workspace_read_only() {
        let workspace = PathBuf::from("/tmp/ws");
        let scratch = PathBuf::from("/tmp/scratch");
        let args = bubblewrap_args(&request(SandboxProfile::ReadOnly, &workspace, &scratch));
        let joined = args.join(" ");
        assert!(joined.contains("--ro-bind /tmp/ws /tmp/ws"));
    }

    #[test]
    fn an_unavailable_backend_never_claims_isolation() {
        let backend = SandboxBackend::unavailable("nothing here");
        assert!(!backend.isolated());
        let described = backend.describe(SandboxProfile::Workspace, None);
        assert_eq!(described.backend, "none");
        assert!(!described.isolated);
        let capabilities = backend.capabilities();
        assert!(!capabilities.filesystem_scoping);
        assert!(!capabilities.network_isolation);
    }

    #[test]
    fn host_description_is_always_unisolated() {
        let backend = SandboxBackend::detect();
        let workspace = PathBuf::from("/tmp/ws");
        let described = backend.describe_host(Some(&workspace));
        assert!(!described.isolated);
        assert_eq!(described.profile, "host");
    }

    #[test]
    fn wrapping_without_a_backend_runs_the_command_directly() {
        let backend = SandboxBackend::unavailable("no backend");
        let workspace = PathBuf::from("/tmp/ws");
        let scratch = PathBuf::from("/tmp/scratch");
        let wrapped = backend.wrap(
            &request(SandboxProfile::Workspace, &workspace, &scratch),
            "echo",
            &["hello".to_owned()],
            &BTreeMap::new(),
        );
        assert_eq!(wrapped.program, OsString::from("echo"));
        assert_eq!(wrapped.args, vec![OsString::from("hello")]);
        assert!(!wrapped.description.isolated);
    }
}

//! Native Windows AppContainer launcher for agent commands.
//!
//! The daemon starts a short-lived copy of itself in launcher mode. That
//! trusted launcher creates a unique AppContainer profile, grants its package
//! SID access to the canonical workspace, scratch, and narrowly selected
//! read-only toolchain trees, creates the requested command suspended with
//! `SECURITY_CAPABILITIES`, assigns it to a kill-on-close Job Object, and only
//! then resumes it. Launcher failure never falls back to an ordinary process.

use std::{
    ffi::{OsStr, OsString, c_void},
    mem,
    os::windows::{ffi::OsStrExt as _, fs::MetadataExt as _},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    ptr,
    sync::OnceLock,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, bail};
use windows_sys::Win32::{
    Foundation::{
        CloseHandle, HANDLE, HANDLE_FLAG_INHERIT, INVALID_HANDLE_VALUE, LocalFree,
        SetHandleInformation, WAIT_OBJECT_0,
    },
    Security::{
        Authorization::ConvertSidToStringSidW,
        DeriveCapabilitySidsFromName, FreeSid,
        Isolation::{CreateAppContainerProfile, DeleteAppContainerProfile},
        PSID, SECURITY_CAPABILITIES, SID_AND_ATTRIBUTES,
    },
    System::{
        Console::{GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE},
        JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_ACTIVE_PROCESS,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JobObjectExtendedLimitInformation, SetInformationJobObject,
        },
        Memory::{GetProcessHeap, HeapAlloc, HeapFree},
        Threading::{
            CREATE_SUSPENDED, CreateProcessW, DeleteProcThreadAttributeList,
            EXTENDED_STARTUPINFO_PRESENT, GetExitCodeProcess, INFINITE,
            InitializeProcThreadAttributeList, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES, PROCESS_INFORMATION, ResumeThread,
            STARTF_USESTDHANDLES, STARTUPINFOEXW, TerminateProcess, UpdateProcThreadAttribute,
            WaitForSingleObject,
        },
    },
};

use crate::agent_sandbox::{SandboxProfile, SandboxRequest};

pub const LAUNCH_ARGUMENT: &str = "--brazier-windows-appcontainer-launch";
pub const PROBE_ARGUMENT: &str = "--brazier-windows-appcontainer-probe";
const MAX_JOB_PROCESSES: u32 = 64;
const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
const HELPER_PROBE_TIMEOUT: Duration = Duration::from_secs(15);
const INTERNET_CLIENT_CAPABILITY: &str = "internetClient";
const PRIVATE_NETWORK_CLIENT_SERVER_CAPABILITY: &str = "privateNetworkClientServer";

/// Return the current executable only after its hidden launcher mode has
/// demonstrated an actual AppContainer boundary. Cached because policy tests
/// and capability endpoints may construct several brokers in one process.
pub fn usable_launcher() -> Option<PathBuf> {
    static PROBE: OnceLock<Option<PathBuf>> = OnceLock::new();
    PROBE
        .get_or_init(|| {
            let executable = std::env::current_exe().ok()?;
            let mut child = Command::new(&executable)
                .arg(PROBE_ARGUMENT)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .ok()?;
            let deadline = Instant::now() + HELPER_PROBE_TIMEOUT;
            loop {
                match child.try_wait() {
                    Ok(Some(status)) if status.success() => return Some(executable),
                    Ok(Some(_)) | Err(_) => return None,
                    Ok(None) if Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(25));
                    }
                    Ok(None) => {
                        let _ = child.kill();
                        let _ = child.wait();
                        return None;
                    }
                }
            }
        })
        .clone()
}

pub fn launcher_args(
    request: &SandboxRequest<'_>,
    program: &str,
    args: &[String],
) -> Vec<OsString> {
    let mut result = vec![
        OsString::from(LAUNCH_ARGUMENT),
        OsString::from(request.profile.as_str()),
        request.workspace.as_os_str().to_owned(),
        request.scratch.as_os_str().to_owned(),
        request.cwd.as_os_str().to_owned(),
        request
            .data_dir
            .map(|path| path.as_os_str().to_owned())
            .unwrap_or_default(),
        OsString::from("--"),
        OsString::from(program),
    ];
    result.extend(args.iter().map(OsString::from));
    result
}

/// Intercept the daemon's private launcher/probe modes before its normal CLI
/// and services start. The returned code is suitable for `std::process::exit`.
pub fn maybe_run_helper() -> Option<i32> {
    let mut args = std::env::args_os().skip(1);
    let mode = args.next()?;
    if mode == PROBE_ARGUMENT {
        let result = if args.next().is_none() {
            probe_full_isolation()
        } else {
            Err(anyhow::anyhow!("unexpected AppContainer probe arguments"))
        };
        return Some(report_helper_result(result));
    }
    if mode != LAUNCH_ARGUMENT {
        return None;
    }

    let result = (|| {
        let profile = args
            .next()
            .and_then(|value| value.to_str().and_then(SandboxProfile::parse))
            .context("invalid or missing AppContainer profile")?;
        let workspace = PathBuf::from(args.next().context("missing AppContainer workspace")?);
        let scratch = PathBuf::from(args.next().context("missing AppContainer scratch path")?);
        let cwd = PathBuf::from(
            args.next()
                .context("missing AppContainer working directory")?,
        );
        let data_dir = args.next().context("missing AppContainer data directory")?;
        let data_dir = (!data_dir.is_empty()).then(|| PathBuf::from(data_dir));
        anyhow::ensure!(
            args.next().as_deref() == Some(OsStr::new("--")),
            "missing AppContainer command separator"
        );
        let program = args.next().context("missing AppContainer program")?;
        let child_args: Vec<OsString> = args.collect();
        run_isolated(
            profile,
            &workspace,
            &scratch,
            &cwd,
            data_dir.as_deref(),
            &program,
            &child_args,
        )
    })();
    Some(report_helper_result(result))
}

fn report_helper_result(result: anyhow::Result<i32>) -> i32 {
    match result {
        Ok(code) => code,
        Err(error) => {
            eprintln!("Brazier Windows sandbox failed closed: {error:#}");
            125
        }
    }
}

fn run_isolated(
    profile: SandboxProfile,
    workspace: &Path,
    scratch: &Path,
    cwd: &Path,
    data_dir: Option<&Path>,
    program: &OsStr,
    args: &[OsString],
) -> anyhow::Result<i32> {
    let workspace = std::fs::canonicalize(workspace)
        .with_context(|| format!("canonicalize workspace {}", workspace.display()))?;
    let scratch = std::fs::canonicalize(scratch)
        .with_context(|| format!("canonicalize scratch {}", scratch.display()))?;
    let cwd = std::fs::canonicalize(cwd)
        .with_context(|| format!("canonicalize working directory {}", cwd.display()))?;
    anyhow::ensure!(
        cwd == workspace || cwd.starts_with(&workspace),
        "AppContainer working directory is outside the workspace"
    );
    let executable = resolve_executable(program)?;
    let toolchain_roots = user_toolchain_read_roots(&executable, args, &workspace, &scratch);
    let mut allowed_reparse_targets = vec![workspace.clone(), scratch.clone()];
    allowed_reparse_targets.extend(toolchain_roots.iter().cloned());
    validate_reparse_tree(&workspace, &allowed_reparse_targets)?;
    validate_reparse_tree(&scratch, &allowed_reparse_targets)?;

    let capabilities = Capabilities::new(profile.allows_network())?;
    let container = AppContainerProfile::create(&capabilities)?;
    let workspace_grant = AclGrant::tree(
        &workspace,
        &container.sid_string,
        profile.allows_workspace_writes(),
    )?;
    let scratch_grant = AclGrant::tree(&scratch, &container.sid_string, true)?;
    let toolchain_grants = grant_toolchain_roots(&toolchain_roots, &container.sid_string)?;
    let mut granted_roots = vec![workspace.clone(), scratch.clone()];
    granted_roots.extend(toolchain_roots.iter().cloned());
    let secret_denials = deny_scoped_secrets(&granted_roots, data_dir, &container.sid_string)?;

    let exit_code = create_appcontainer_process(&container, &capabilities, &executable, args, &cwd);

    // Keep ACL grants and the profile alive until every process in the job has
    // exited or has been killed by closing the job handle.
    drop(secret_denials);
    drop(toolchain_grants);
    drop(scratch_grant);
    drop(workspace_grant);
    exit_code
}

/// Override workspace, scratch, and toolchain grants for credential paths
/// nested below them. Credentials outside granted trees remain inaccessible
/// through the AppContainer boundary without changing their host ACLs.
fn deny_scoped_secrets(
    granted_roots: &[PathBuf],
    data_dir: Option<&Path>,
    sid: &str,
) -> anyhow::Result<Vec<AclDeny>> {
    let mut paths = Vec::new();
    for path in crate::agent_sandbox::secret_paths(data_dir) {
        if !inside_any(&path, granted_roots) {
            continue;
        }
        match path.try_exists() {
            Ok(false) => continue,
            Ok(true) => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspect credential path {}", path.display()));
            }
        }
        let path = std::fs::canonicalize(&path)
            .with_context(|| format!("canonicalize credential path {}", path.display()))?;
        if inside_any(&path, granted_roots) && !paths.contains(&path) {
            paths.push(path);
        }
    }
    paths
        .into_iter()
        .map(|path| AclDeny::apply(&path, sid))
        .collect()
}

fn inside_any(path: &Path, roots: &[PathBuf]) -> bool {
    roots
        .iter()
        .any(|root| path == root || path.starts_with(root))
}

/// Reparse points are safe only when their final target is already covered by
/// one of the explicit workspace, scratch, or read-only toolchain grants. Do
/// not descend through them here: `icacls /L` operates on the link itself, and
/// the independently granted target tree supplies the target's effective ACL.
fn validate_reparse_tree(root: &Path, allowed_targets: &[PathBuf]) -> anyhow::Result<()> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        let metadata = std::fs::symlink_metadata(&path)
            .with_context(|| format!("inspect sandbox path {}", path.display()))?;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            let target = match std::fs::canonicalize(&path) {
                Ok(target) => target,
                Err(_) => {
                    let target = std::fs::read_link(&path).with_context(|| {
                        format!("resolve sandbox reparse point {}", path.display())
                    })?;
                    let target = if target.is_absolute() {
                        target
                    } else {
                        path.parent().unwrap_or(root).join(target)
                    };
                    crate::agent_policy::canonical_ancestor(&target)
                }
            };
            anyhow::ensure!(
                inside_any(&target, allowed_targets),
                "sandbox reparse point {} escapes to {}",
                path.display(),
                target.display()
            );
            continue;
        }
        if metadata.is_dir() {
            for entry in std::fs::read_dir(&path)
                .with_context(|| format!("scan sandbox directory {}", path.display()))?
            {
                stack.push(entry?.path());
            }
        }
    }
    Ok(())
}

/// Read-only trees that user-installed command-line tools need after `cmd.exe`
/// resolves them through PATH. Broad user/system roots are deliberately
/// excluded: only narrow PATH entries and known toolchain caches are opened.
fn user_toolchain_read_roots(
    executable: &Path,
    args: &[OsString],
    workspace: &Path,
    scratch: &Path,
) -> Vec<PathBuf> {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .and_then(|path| std::fs::canonicalize(path).ok());
    let user_scopes =
        canonical_environment_roots(&["USERPROFILE", "HOME", "LOCALAPPDATA", "APPDATA"]);
    let excluded_scopes = canonical_environment_roots(&[
        "SystemRoot",
        "ProgramFiles",
        "ProgramFiles(x86)",
        "ProgramData",
        "RUNNER_TOOL_CACHE",
        "ChocolateyInstall",
    ]);

    let mut candidates = Vec::new();
    if let Some(parent) = executable.parent() {
        candidates.push(parent.to_path_buf());
    }
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&path));
    }
    if let Some(home) = home {
        let mut invocation = executable
            .as_os_str()
            .to_string_lossy()
            .to_ascii_lowercase();
        for arg in args {
            invocation.push(' ');
            invocation.push_str(&arg.to_string_lossy().to_ascii_lowercase());
        }
        candidates.extend(
            crate::agent_sandbox::HOME_TOOLCHAIN_PATHS
                .iter()
                .filter(|relative| toolchain_path_is_requested(relative, &invocation))
                .map(|relative| home.join(relative)),
        );
    }

    normalize_toolchain_roots(
        candidates,
        &user_scopes,
        &excluded_scopes,
        workspace,
        scratch,
    )
}

fn toolchain_path_is_requested(relative: &str, invocation: &str) -> bool {
    let mentions_any = |names: &[&str]| {
        invocation
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '-')
            .any(|token| {
                names.iter().any(|name| {
                    token == *name
                        || token.strip_prefix(name).is_some_and(|suffix| {
                            suffix.starts_with('-')
                                || (!suffix.is_empty()
                                    && suffix.chars().all(|character| character.is_ascii_digit()))
                        })
                })
            })
    };
    match relative {
        ".cargo" | ".rustup" => mentions_any(&["cargo", "rustc", "rustup"]),
        ".local/share/uv" | ".cache/uv" | ".cache/pip" => {
            mentions_any(&["uv", "uvx", "python", "pip"])
        }
        ".cache/go-build" => mentions_any(&["go", "gofmt"]),
        ".cache/typescript" | ".npm" | ".node" | ".local/share/pnpm" => {
            mentions_any(&["node", "nodejs", "npm", "npx", "pnpm", "tsc", "typescript"])
        }
        ".bun" => mentions_any(&["bun", "bunx"]),
        _ => false,
    }
}

fn canonical_environment_roots(variables: &[&str]) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for variable in variables {
        if let Some(path) = std::env::var_os(*variable)
            .map(PathBuf::from)
            .and_then(|path| std::fs::canonicalize(path).ok())
            && !roots.contains(&path)
        {
            roots.push(path);
        }
    }
    roots
}

fn normalize_toolchain_roots(
    candidates: Vec<PathBuf>,
    user_scopes: &[PathBuf],
    excluded_scopes: &[PathBuf],
    workspace: &Path,
    scratch: &Path,
) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for candidate in candidates {
        let Ok(candidate) = std::fs::canonicalize(candidate) else {
            continue;
        };
        if !candidate.is_dir()
            || candidate.parent().is_none()
            || candidate == workspace
            || candidate.starts_with(workspace)
            || workspace.starts_with(&candidate)
            || candidate == scratch
            || candidate.starts_with(scratch)
            || scratch.starts_with(&candidate)
            || is_windows_app_alias_directory(&candidate)
            || excluded_scopes
                .iter()
                .any(|scope| candidate == *scope || candidate.starts_with(scope))
            || user_scopes.iter().any(|scope| candidate == *scope)
            || user_scopes
                .iter()
                .any(|scope| scope.starts_with(&candidate))
        {
            continue;
        }
        if !roots.contains(&candidate) {
            roots.push(candidate);
        }
    }
    roots.sort_by_key(|path| path.components().count());
    let mut compact: Vec<PathBuf> = Vec::new();
    for root in roots {
        if !compact.iter().any(|parent| root.starts_with(parent)) {
            compact.push(root);
        }
    }
    compact
}

fn is_windows_app_alias_directory(path: &Path) -> bool {
    let mut components = path
        .components()
        .rev()
        .filter_map(|component| match component {
            std::path::Component::Normal(value) => Some(value.to_string_lossy()),
            _ => None,
        });
    components
        .next()
        .is_some_and(|value| value.eq_ignore_ascii_case("WindowsApps"))
        && components
            .next()
            .is_some_and(|value| value.eq_ignore_ascii_case("Microsoft"))
}

fn grant_toolchain_roots(roots: &[PathBuf], sid: &str) -> anyhow::Result<Vec<AclGrant>> {
    let mut grants = Vec::new();
    let mut traversed = Vec::new();
    let user_scopes = canonical_environment_roots(&["USERPROFILE", "HOME"]);
    for root in roots {
        let stop = user_scopes
            .iter()
            .filter(|scope| root.starts_with(scope))
            .min_by_key(|scope| scope.components().count())
            .map(|path| path.as_path());
        let mut parent = root.parent();
        while let Some(path) = parent {
            if path.parent().is_none() {
                break;
            }
            if !traversed.contains(&path.to_path_buf()) {
                grants.push(AclGrant::traverse(path, sid)?);
                traversed.push(path.to_path_buf());
            }
            if stop.is_some_and(|stop| path == stop) {
                break;
            }
            parent = path.parent();
        }
        grants.push(AclGrant::tree(root, sid, false)?);
    }
    Ok(grants)
}

struct AclGrant {
    path: PathBuf,
    sid: String,
    recursive: bool,
}

impl AclGrant {
    fn tree(path: &Path, sid: &str, writable: bool) -> anyhow::Result<Self> {
        let permission = if writable {
            "(OI)(CI)(M)"
        } else {
            "(OI)(CI)(RX)"
        };
        Self::apply(path, sid, permission, true)
    }

    fn traverse(path: &Path, sid: &str) -> anyhow::Result<Self> {
        Self::apply(path, sid, "(X)", false)
    }

    fn apply(path: &Path, sid: &str, permission: &str, recursive: bool) -> anyhow::Result<Self> {
        let principal = format!("*{sid}:{permission}");
        let mut command = Command::new("icacls.exe");
        command.arg(path).args(["/grant", &principal]);
        if recursive {
            command.arg("/T");
        }
        let output = command
            .args(["/L", "/Q"])
            .output()
            .with_context(|| format!("start icacls for {}", path.display()))?;
        let grant = Self {
            path: path.to_path_buf(),
            sid: sid.to_owned(),
            recursive,
        };
        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            drop(grant); // Remove any ACEs applied before icacls failed.
            bail!(
                "icacls refused AppContainer access to {}: stdout={} stderr={}",
                path.display(),
                stdout,
                stderr
            );
        }
        Ok(grant)
    }
}

impl Drop for AclGrant {
    fn drop(&mut self) {
        let principal = format!("*{}", self.sid);
        let mut command = Command::new("icacls.exe");
        command.arg(&self.path).args(["/remove:g", &principal]);
        if self.recursive {
            command.arg("/T");
        }
        let _ = command
            .args(["/C", "/L", "/Q"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

struct AclDeny {
    path: PathBuf,
    sid: String,
    recursive: bool,
}

impl AclDeny {
    fn apply(path: &Path, sid: &str) -> anyhow::Result<Self> {
        let recursive = path.is_dir();
        let principal = if recursive {
            format!("*{sid}:(OI)(CI)(F)")
        } else {
            format!("*{sid}:(F)")
        };
        let mut command = Command::new("icacls.exe");
        command.arg(path).args(["/deny", &principal]);
        if recursive {
            command.arg("/T");
        }
        let output = command
            .args(["/L", "/Q"])
            .output()
            .with_context(|| format!("start icacls for credential path {}", path.display()))?;
        let denial = Self {
            path: path.to_path_buf(),
            sid: sid.to_owned(),
            recursive,
        };
        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            drop(denial); // Remove any ACEs applied before icacls failed.
            bail!(
                "icacls could not mask credential path {}: stdout={} stderr={}",
                path.display(),
                stdout,
                stderr
            );
        }
        Ok(denial)
    }
}

impl Drop for AclDeny {
    fn drop(&mut self) {
        let principal = format!("*{}", self.sid);
        let mut command = Command::new("icacls.exe");
        command.arg(&self.path).args(["/remove:d", &principal]);
        if self.recursive {
            command.arg("/T");
        }
        let _ = command
            .args(["/C", "/L", "/Q"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

struct Capabilities {
    entries: Vec<SID_AND_ATTRIBUTES>,
    owned_sids: Vec<PSID>,
}

impl Capabilities {
    fn new(network: bool) -> anyhow::Result<Self> {
        let mut result = Self {
            entries: Vec::new(),
            owned_sids: Vec::new(),
        };
        if network {
            for name in [
                INTERNET_CLIENT_CAPABILITY,
                PRIVATE_NETWORK_CLIENT_SERVER_CAPABILITY,
            ] {
                let sid = derive_capability_sid(name)?;
                result.entries.push(SID_AND_ATTRIBUTES {
                    Sid: sid,
                    Attributes: 4, // SE_GROUP_ENABLED
                });
                result.owned_sids.push(sid);
            }
        }
        Ok(result)
    }
}

impl Drop for Capabilities {
    fn drop(&mut self) {
        for sid in self.owned_sids.drain(..) {
            unsafe {
                LocalFree(sid);
            }
        }
    }
}

fn derive_capability_sid(name: &str) -> anyhow::Result<PSID> {
    let name_wide = wide(name);
    let mut group_sids: *mut PSID = ptr::null_mut();
    let mut group_count = 0_u32;
    let mut capability_sids: *mut PSID = ptr::null_mut();
    let mut capability_count = 0_u32;
    let success = unsafe {
        DeriveCapabilitySidsFromName(
            name_wide.as_ptr(),
            &mut group_sids,
            &mut group_count,
            &mut capability_sids,
            &mut capability_count,
        )
    };
    if success == 0 {
        return Err(std::io::Error::last_os_error()).context("derive Windows capability SID");
    }

    let result = (|| {
        anyhow::ensure!(
            capability_count == 1 && !capability_sids.is_null(),
            "Windows returned {capability_count} SIDs for capability `{name}`"
        );
        Ok(unsafe { *capability_sids })
    })();
    unsafe {
        if !group_sids.is_null() {
            for index in 0..group_count as usize {
                LocalFree(*group_sids.add(index));
            }
            LocalFree(group_sids.cast());
        }
        if !capability_sids.is_null() {
            if result.is_err() {
                for index in 0..capability_count as usize {
                    LocalFree(*capability_sids.add(index));
                }
            } else {
                for index in 1..capability_count as usize {
                    LocalFree(*capability_sids.add(index));
                }
            }
            LocalFree(capability_sids.cast());
        }
    }
    result
}

struct AppContainerProfile {
    name: Vec<u16>,
    sid: PSID,
    sid_string: String,
}

impl AppContainerProfile {
    fn create(capabilities: &Capabilities) -> anyhow::Result<Self> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let name = format!("Brazier.Agent.{}.{unique:x}", std::process::id());
        let name_wide = wide(&name);
        let display = wide("Brazier isolated agent command");
        let description = wide("Ephemeral AppContainer for one Brazier agent command");
        let mut sid: PSID = ptr::null_mut();
        let (capabilities_ptr, capability_count) = if capabilities.entries.is_empty() {
            (ptr::null(), 0)
        } else {
            (
                capabilities.entries.as_ptr(),
                capabilities.entries.len() as u32,
            )
        };
        let result = unsafe {
            CreateAppContainerProfile(
                name_wide.as_ptr(),
                display.as_ptr(),
                description.as_ptr(),
                capabilities_ptr,
                capability_count,
                &mut sid,
            )
        };
        anyhow::ensure!(
            result >= 0 && !sid.is_null(),
            "CreateAppContainerProfile failed with HRESULT 0x{:08x}",
            result as u32
        );
        let sid_string = match sid_to_string(sid) {
            Ok(value) => value,
            Err(error) => {
                unsafe {
                    FreeSid(sid);
                    DeleteAppContainerProfile(name_wide.as_ptr());
                }
                return Err(error);
            }
        };
        Ok(Self {
            name: name_wide,
            sid,
            sid_string,
        })
    }
}

impl Drop for AppContainerProfile {
    fn drop(&mut self) {
        unsafe {
            DeleteAppContainerProfile(self.name.as_ptr());
            FreeSid(self.sid);
        }
    }
}

fn sid_to_string(sid: PSID) -> anyhow::Result<String> {
    let mut value = ptr::null_mut();
    if unsafe { ConvertSidToStringSidW(sid, &mut value) } == 0 {
        return Err(std::io::Error::last_os_error()).context("format AppContainer SID");
    }
    let mut length = 0;
    unsafe {
        while *value.add(length) != 0 {
            length += 1;
        }
    }
    let string = String::from_utf16(unsafe { std::slice::from_raw_parts(value, length) });
    unsafe {
        LocalFree(value.cast());
    }
    string.context("AppContainer SID was not valid UTF-16")
}

fn create_appcontainer_process(
    container: &AppContainerProfile,
    capabilities: &Capabilities,
    executable: &Path,
    args: &[OsString],
    cwd: &Path,
) -> anyhow::Result<i32> {
    let job = OwnedHandle::new(unsafe { CreateJobObjectW(ptr::null(), ptr::null()) })
        .context("create Windows sandbox Job Object")?;
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    limits.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
    limits.BasicLimitInformation.ActiveProcessLimit = MAX_JOB_PROCESSES;
    if unsafe {
        SetInformationJobObject(
            job.0,
            JobObjectExtendedLimitInformation,
            (&raw const limits).cast(),
            mem::size_of_val(&limits) as u32,
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error())
            .context("configure Windows sandbox Job Object");
    }

    let std_handles = [
        unsafe { GetStdHandle(STD_INPUT_HANDLE) },
        unsafe { GetStdHandle(STD_OUTPUT_HANDLE) },
        unsafe { GetStdHandle(STD_ERROR_HANDLE) },
    ];
    anyhow::ensure!(
        std_handles
            .iter()
            .all(|handle| !handle.is_null() && *handle != INVALID_HANDLE_VALUE),
        "Windows sandbox launcher requires stdin, stdout, and stderr handles"
    );
    let mut inherited_handles = Vec::with_capacity(std_handles.len());
    for handle in std_handles {
        if !inherited_handles.contains(&handle) {
            if unsafe { SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) }
                == 0
            {
                return Err(std::io::Error::last_os_error())
                    .context("mark Windows sandbox stdio inheritable");
            }
            inherited_handles.push(handle);
        }
    }

    let attributes = AttributeList::new(2)?;
    let capability_entries = if capabilities.entries.is_empty() {
        ptr::null_mut()
    } else {
        capabilities.entries.as_ptr() as *mut SID_AND_ATTRIBUTES
    };
    let mut security_capabilities = SECURITY_CAPABILITIES {
        AppContainerSid: container.sid,
        Capabilities: capability_entries,
        CapabilityCount: capabilities.entries.len() as u32,
        Reserved: 0,
    };
    if unsafe {
        UpdateProcThreadAttribute(
            attributes.ptr,
            0,
            PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES as usize,
            (&raw mut security_capabilities).cast(),
            mem::size_of_val(&security_capabilities),
            ptr::null_mut(),
            ptr::null(),
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error())
            .context("attach AppContainer security capabilities");
    }
    if unsafe {
        UpdateProcThreadAttribute(
            attributes.ptr,
            0,
            PROC_THREAD_ATTRIBUTE_HANDLE_LIST as usize,
            inherited_handles.as_ptr().cast(),
            mem::size_of_val(inherited_handles.as_slice()),
            ptr::null_mut(),
            ptr::null(),
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error())
            .context("restrict AppContainer inherited handles");
    }

    let mut startup = STARTUPINFOEXW::default();
    startup.StartupInfo.cb = mem::size_of::<STARTUPINFOEXW>() as u32;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = std_handles[0];
    startup.StartupInfo.hStdOutput = std_handles[1];
    startup.StartupInfo.hStdError = std_handles[2];
    startup.lpAttributeList = attributes.ptr;

    let application = wide_os(executable.as_os_str());
    let mut command_line = encode_command_line(executable.as_os_str(), args);
    let cwd = wide_os(cwd.as_os_str());
    let mut process = PROCESS_INFORMATION::default();
    let created = unsafe {
        CreateProcessW(
            application.as_ptr(),
            command_line.as_mut_ptr(),
            ptr::null(),
            ptr::null(),
            1,
            EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED,
            ptr::null(),
            cwd.as_ptr(),
            (&raw const startup).cast(),
            &mut process,
        )
    };
    if created == 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("create AppContainer process for {}", executable.display()));
    }
    let process_handle = OwnedHandle(process.hProcess);
    let thread_handle = OwnedHandle(process.hThread);

    if unsafe { AssignProcessToJobObject(job.0, process_handle.0) } == 0 {
        unsafe {
            TerminateProcess(process_handle.0, 125);
        }
        return Err(std::io::Error::last_os_error())
            .context("assign AppContainer process to Job Object");
    }
    if unsafe { ResumeThread(thread_handle.0) } == u32::MAX {
        unsafe {
            TerminateProcess(process_handle.0, 125);
        }
        return Err(std::io::Error::last_os_error()).context("resume AppContainer process");
    }
    if unsafe { WaitForSingleObject(process_handle.0, INFINITE) } != WAIT_OBJECT_0 {
        return Err(std::io::Error::last_os_error()).context("wait for AppContainer process");
    }
    let mut exit_code = 0_u32;
    if unsafe { GetExitCodeProcess(process_handle.0, &mut exit_code) } == 0 {
        return Err(std::io::Error::last_os_error()).context("read AppContainer exit code");
    }
    drop(thread_handle);
    drop(process_handle);
    drop(job); // Kills any descendants still alive after the shell exits.
    Ok(exit_code as i32)
}

struct AttributeList {
    ptr: *mut c_void,
    heap: HANDLE,
}

impl AttributeList {
    fn new(count: u32) -> anyhow::Result<Self> {
        let mut size = 0_usize;
        unsafe {
            InitializeProcThreadAttributeList(ptr::null_mut(), count, 0, &mut size);
        }
        anyhow::ensure!(size > 0, "Windows returned an empty process attribute list");
        let heap = unsafe { GetProcessHeap() };
        let pointer = unsafe { HeapAlloc(heap, 0, size) };
        anyhow::ensure!(
            !pointer.is_null(),
            "allocate Windows process attribute list"
        );
        if unsafe { InitializeProcThreadAttributeList(pointer, count, 0, &mut size) } == 0 {
            unsafe {
                HeapFree(heap, 0, pointer);
            }
            return Err(std::io::Error::last_os_error())
                .context("initialize Windows process attribute list");
        }
        Ok(Self { ptr: pointer, heap })
    }
}

impl Drop for AttributeList {
    fn drop(&mut self) {
        unsafe {
            DeleteProcThreadAttributeList(self.ptr);
            HeapFree(self.heap, 0, self.ptr);
        }
    }
}

struct OwnedHandle(HANDLE);

impl OwnedHandle {
    fn new(handle: HANDLE) -> Option<Self> {
        (!handle.is_null() && handle != INVALID_HANDLE_VALUE).then_some(Self(handle))
    }
}

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}

fn resolve_executable(program: &OsStr) -> anyhow::Result<PathBuf> {
    let requested = Path::new(program);
    if requested.components().count() > 1 || requested.is_absolute() {
        anyhow::ensure!(
            requested.is_file(),
            "program {} does not exist",
            requested.display()
        );
        return std::fs::canonicalize(requested).context("canonicalize sandbox executable");
    }

    let extensions: Vec<OsString> = if requested.extension().is_some() {
        vec![OsString::new()]
    } else {
        std::env::var_os("PATHEXT")
            .map(|value| {
                value
                    .to_string_lossy()
                    .split(';')
                    .filter(|value| !value.is_empty())
                    .map(OsString::from)
                    .collect()
            })
            .unwrap_or_else(|| {
                [".COM", ".EXE", ".BAT", ".CMD"]
                    .map(OsString::from)
                    .to_vec()
            })
    };
    let path = std::env::var_os("PATH").context("PATH is unavailable to sandbox launcher")?;
    for directory in std::env::split_paths(&path) {
        for extension in &extensions {
            let mut name = program.to_owned();
            name.push(extension);
            let candidate = directory.join(name);
            if candidate.is_file() {
                return std::fs::canonicalize(&candidate)
                    .with_context(|| format!("canonicalize {}", candidate.display()));
            }
        }
    }
    bail!(
        "program `{}` was not found on PATH",
        program.to_string_lossy()
    )
}

fn encode_command_line(program: &OsStr, args: &[OsString]) -> Vec<u16> {
    let mut command = quote_windows_arg(program);
    for argument in args {
        command.push(' ' as u16);
        command.extend(quote_windows_arg(argument));
    }
    command.push(0);
    command
}

fn quote_windows_arg(argument: &OsStr) -> Vec<u16> {
    let value: Vec<u16> = argument.encode_wide().collect();
    if !value.is_empty()
        && !value
            .iter()
            .any(|character| matches!(*character, 0x20 | 0x09 | 0x22))
    {
        return value;
    }

    let mut result = vec!['"' as u16];
    let mut backslashes = 0;
    for character in value {
        if character == '\\' as u16 {
            backslashes += 1;
        } else if character == '"' as u16 {
            result.extend(std::iter::repeat_n('\\' as u16, backslashes * 2 + 1));
            result.push(character);
            backslashes = 0;
        } else {
            result.extend(std::iter::repeat_n('\\' as u16, backslashes));
            result.push(character);
            backslashes = 0;
        }
    }
    result.extend(std::iter::repeat_n('\\' as u16, backslashes * 2));
    result.push('"' as u16);
    result
}

fn wide(value: &str) -> Vec<u16> {
    wide_os(OsStr::new(value))
}

fn wide_os(value: &OsStr) -> Vec<u16> {
    value.encode_wide().chain(Some(0)).collect()
}

fn create_directory_junction(link: &Path, target: &Path) -> anyhow::Result<()> {
    let output = Command::new("cmd.exe")
        .args(["/D", "/C", "mklink", "/J"])
        .arg(link)
        .arg(target)
        .output()
        .with_context(|| format!("create test junction {}", link.display()))?;
    anyhow::ensure!(
        output.status.success(),
        "mklink /J failed for {} -> {}: stdout={} stderr={}",
        link.display(),
        target.display(),
        String::from_utf8_lossy(&output.stdout).trim(),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

fn probe_full_isolation() -> anyhow::Result<i32> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "brazier-appcontainer-probe-{}-{unique:x}",
        std::process::id()
    ));
    let workspace = root.join("workspace");
    let scratch = root.join("scratch");
    let outside = root.join("outside");
    std::fs::create_dir_all(&workspace)?;
    std::fs::create_dir_all(&scratch)?;
    std::fs::create_dir_all(&outside)?;
    std::fs::write(outside.join("secret.txt"), "appcontainer-probe-secret")?;
    let linked_target = workspace.join("linked-target");
    let linked_workspace = workspace.join("linked-workspace");
    std::fs::create_dir_all(&linked_target)?;
    std::fs::write(linked_target.join("source.txt"), "linked-workspace-ok")?;
    create_directory_junction(&linked_workspace, &linked_target)?;
    let leak = workspace.join("leak.txt");
    let inside = workspace.join("inside.txt");
    let linked_copy = workspace.join("linked-copy.txt");
    let escaped = outside.join("escaped.txt");
    let command = format!(
        "type \"{}\" > \"{}\" 2>nul & echo escaped > \"{}\" 2>nul & echo inside > \"{}\" & type \"{}\" > \"{}\"",
        outside.join("secret.txt").display(),
        leak.display(),
        escaped.display(),
        inside.display(),
        linked_workspace.join("source.txt").display(),
        linked_copy.display(),
    );
    let result = run_isolated(
        SandboxProfile::Workspace,
        &workspace,
        &scratch,
        &workspace,
        None,
        OsStr::new("cmd.exe"),
        &[
            OsString::from("/D"),
            OsString::from("/C"),
            OsString::from(command),
        ],
    );
    let verified = result.and_then(|code| {
        anyhow::ensure!(code == 0, "AppContainer probe command exited {code}");
        anyhow::ensure!(
            inside.is_file(),
            "AppContainer could not write its workspace"
        );
        anyhow::ensure!(
            !std::fs::read_to_string(&leak)
                .unwrap_or_default()
                .contains("appcontainer-probe-secret"),
            "AppContainer read outside its workspace"
        );
        anyhow::ensure!(
            !escaped.exists(),
            "AppContainer wrote outside its workspace"
        );
        anyhow::ensure!(
            std::fs::read_to_string(&linked_copy).unwrap_or_default() == "linked-workspace-ok",
            "AppContainer could not follow an in-workspace junction"
        );
        Ok(0)
    });
    let _ = std::fs::remove_dir(&linked_workspace);
    let _ = std::fs::remove_dir_all(&root);
    verified
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launcher_arguments_preserve_profile_paths_and_command() {
        let workspace = PathBuf::from(r"C:\work space");
        let scratch = PathBuf::from(r"C:\scratch");
        let cwd = workspace.join("crate");
        let request = SandboxRequest {
            profile: SandboxProfile::WorkspaceNetwork,
            workspace: &workspace,
            scratch: &scratch,
            cwd: &cwd,
            data_dir: None,
        };
        assert_eq!(
            launcher_args(&request, "cmd", &["/C".to_owned(), "cargo test".to_owned()]),
            vec![
                OsString::from(LAUNCH_ARGUMENT),
                OsString::from("workspace-network"),
                workspace.as_os_str().to_owned(),
                scratch.as_os_str().to_owned(),
                cwd.as_os_str().to_owned(),
                OsString::new(),
                OsString::from("--"),
                OsString::from("cmd"),
                OsString::from("/C"),
                OsString::from("cargo test"),
            ]
        );
    }

    #[test]
    fn windows_command_line_quoting_handles_empty_spaces_quotes_and_slashes() {
        let quoted =
            |value: &str| String::from_utf16(&quote_windows_arg(OsStr::new(value))).unwrap();
        assert_eq!(quoted("plain"), "plain");
        assert_eq!(quoted(""), "\"\"");
        assert_eq!(quoted("two words"), "\"two words\"");
        assert_eq!(quoted("say\"hi"), "\"say\\\"hi\"");
        assert_eq!(quoted(r"C:\path with space\"), r#""C:\path with space\\""#);
    }

    #[test]
    fn appcontainer_probe_enforces_workspace_boundary() {
        probe_full_isolation().expect("real AppContainer isolation must be usable");
    }

    #[test]
    fn reparse_validation_allows_internal_junctions_and_rejects_escapes() {
        let root = tempfile::tempdir().expect("temp root");
        let workspace = root.path().join("workspace");
        let scratch = root.path().join("scratch");
        let outside = root.path().join("outside");
        let internal_target = workspace.join("store");
        std::fs::create_dir_all(&internal_target).unwrap();
        std::fs::create_dir_all(&scratch).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let workspace = std::fs::canonicalize(workspace).unwrap();
        let scratch = std::fs::canonicalize(scratch).unwrap();
        let outside = std::fs::canonicalize(outside).unwrap();
        let internal = workspace.join("internal-link");
        create_directory_junction(&internal, &internal_target).unwrap();
        validate_reparse_tree(&workspace, &[workspace.clone(), scratch.clone()])
            .expect("an in-workspace junction is bounded by the workspace ACL");
        std::fs::remove_dir(&internal).unwrap();

        let escape = workspace.join("escape-link");
        create_directory_junction(&escape, &outside).unwrap();
        let error = validate_reparse_tree(&workspace, &[workspace.clone(), scratch])
            .expect_err("a junction outside every granted tree must fail closed");
        std::fs::remove_dir(&escape).unwrap();
        assert!(error.to_string().contains("escapes to"), "{error:#}");
    }

    #[test]
    fn toolchain_roots_keep_narrow_user_and_external_paths() {
        let root = tempfile::tempdir().expect("temp root");
        let user = root.path().join("user");
        let cargo = user.join(".cargo");
        let cargo_bin = cargo.join("bin");
        let system = root.path().join("system");
        let system_bin = system.join("bin");
        let external_bin = root.path().join("external-tools").join("bin");
        let workspace = root.path().join("workspace");
        let workspace_bin = workspace.join("bin");
        let scratch = root.path().join("scratch");
        for path in [
            &cargo_bin,
            &system_bin,
            &external_bin,
            &workspace_bin,
            &scratch,
        ] {
            std::fs::create_dir_all(path).unwrap();
        }
        let canonical = |path: &Path| std::fs::canonicalize(path).unwrap();
        let user = canonical(&user);
        let cargo = canonical(&cargo);
        let cargo_bin = canonical(&cargo_bin);
        let system = canonical(&system);
        let system_bin = canonical(&system_bin);
        let external_bin = canonical(&external_bin);
        let workspace = canonical(&workspace);
        let workspace_bin = canonical(&workspace_bin);
        let scratch = canonical(&scratch);
        let roots = normalize_toolchain_roots(
            vec![
                user.clone(),
                cargo_bin.clone(),
                cargo.clone(),
                system_bin.clone(),
                external_bin.clone(),
                workspace_bin,
            ],
            &[user],
            &[system],
            &workspace,
            &scratch,
        );
        assert!(roots.contains(&cargo));
        assert!(roots.contains(&external_bin));
        assert!(!roots.contains(&cargo_bin));
        assert!(!roots.contains(&system_bin));
        assert_eq!(roots.len(), 2, "{roots:?}");
    }

    #[test]
    fn toolchain_cache_selection_follows_the_requested_command_family() {
        assert!(toolchain_path_is_requested(".cargo", "cmd /c cargo test"));
        assert!(toolchain_path_is_requested(
            ".rustup",
            "cmd /c rustc --version"
        ));
        assert!(toolchain_path_is_requested(".cache/uv", "cmd /c uv run"));
        assert!(toolchain_path_is_requested(
            ".cache/uv",
            "cmd /c python3 script.py"
        ));
        assert!(toolchain_path_is_requested(
            ".local/share/pnpm",
            "cmd /c pnpm test"
        ));
        assert!(!toolchain_path_is_requested(".cargo", "cmd /c echo ok"));
        assert!(!toolchain_path_is_requested(".npm", "cmd /c echo ok"));
        assert!(!toolchain_path_is_requested(
            ".cache/go-build",
            "cmd /c cargo test"
        ));
    }

    #[test]
    fn windows_app_execution_alias_directory_is_not_acl_rewritten() {
        assert!(is_windows_app_alias_directory(Path::new(
            r"C:\Users\person\AppData\Local\Microsoft\WindowsApps"
        )));
        assert!(!is_windows_app_alias_directory(Path::new(
            r"C:\Users\person\AppData\Roaming\npm"
        )));
    }

    #[test]
    fn appcontainer_can_launch_a_user_toolchain_executable() {
        let root = tempfile::tempdir().expect("temp root");
        let workspace = root.path().join("workspace");
        let scratch = root.path().join("scratch");
        let toolchain_bin = root.path().join("user-toolchain").join("bin");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&scratch).unwrap();
        std::fs::create_dir_all(&toolchain_bin).unwrap();
        let copied_shell = toolchain_bin.join("toolchain-command.exe");
        std::fs::copy(
            resolve_executable(OsStr::new("cmd.exe")).unwrap(),
            &copied_shell,
        )
        .unwrap();
        let marker = workspace.join("toolchain-ran.txt");
        let code = run_isolated(
            SandboxProfile::Workspace,
            &workspace,
            &scratch,
            &workspace,
            None,
            copied_shell.as_os_str(),
            &[
                OsString::from("/D"),
                OsString::from("/C"),
                OsString::from(format!("echo ok>\"{}\"", marker.display())),
            ],
        )
        .expect("run copied user-toolchain executable in AppContainer");
        assert_eq!(code, 0);
        assert!(marker.is_file());
    }

    #[test]
    fn read_only_profile_cannot_modify_workspace_but_can_use_scratch() {
        let root = tempfile::tempdir().expect("temp root");
        let workspace = root.path().join("workspace");
        let scratch = root.path().join("scratch");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&scratch).unwrap();
        let protected = workspace.join("protected.txt");
        let scratch_file = scratch.join("scratch.txt");
        std::fs::write(&protected, "original").unwrap();
        let command = format!(
            "echo changed > \"{}\" 2>nul & echo scratch > \"{}\"",
            protected.display(),
            scratch_file.display()
        );
        let code = run_isolated(
            SandboxProfile::ReadOnly,
            &workspace,
            &scratch,
            &workspace,
            None,
            OsStr::new("cmd.exe"),
            &[
                OsString::from("/D"),
                OsString::from("/C"),
                OsString::from(command),
            ],
        )
        .expect("run read-only AppContainer");
        assert_eq!(code, 0);
        assert_eq!(std::fs::read_to_string(&protected).unwrap(), "original");
        assert!(scratch_file.is_file());
    }

    #[test]
    fn credential_directory_inside_workspace_remains_unreadable_and_immutable() {
        let root = tempfile::tempdir().expect("temp root");
        let workspace = root.path().join("workspace");
        let scratch = root.path().join("scratch");
        let data_dir = workspace.join(".brazier-data");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&scratch).unwrap();
        let secret = data_dir.join("credential.txt");
        let leak = workspace.join("credential-leak.txt");
        let marker = scratch.join("marker.txt");
        std::fs::write(&secret, "do-not-leak").unwrap();
        let command = format!(
            "type \"{}\" > \"{}\" 2>nul & echo changed > \"{}\" 2>nul & echo ok > \"{}\"",
            secret.display(),
            leak.display(),
            secret.display(),
            marker.display(),
        );
        let code = run_isolated(
            SandboxProfile::Workspace,
            &workspace,
            &scratch,
            &workspace,
            Some(&data_dir),
            OsStr::new("cmd.exe"),
            &[
                OsString::from("/D"),
                OsString::from("/C"),
                OsString::from(command),
            ],
        )
        .expect("run credential-boundary AppContainer");
        assert_eq!(code, 0);
        assert_eq!(std::fs::read_to_string(&secret).unwrap(), "do-not-leak");
        assert!(
            !std::fs::read_to_string(&leak)
                .unwrap_or_default()
                .contains("do-not-leak")
        );
        assert!(marker.is_file());
    }

    #[test]
    fn capability_set_adds_internet_and_private_network_for_network_profiles() {
        let offline = Capabilities::new(false).unwrap();
        assert!(offline.entries.is_empty());
        let online = Capabilities::new(true).unwrap();
        assert_eq!(online.entries.len(), 2);
        assert!(online.entries.iter().all(|entry| !entry.Sid.is_null()));
    }
}

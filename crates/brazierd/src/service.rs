//! Durable state for a deliberately long-lived daemon.
//!
//! The desktop app starts a disposable loopback daemon, whose random key can
//! disappear with the process. A headless service is different: a restart
//! must not strand its clients with a new endpoint or credential. This module
//! owns only that small service contract; pairing and scoped client credentials
//! remain a later remote-access layer.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use serde::Serialize;
use uuid::Uuid;

#[cfg(windows)]
mod windows_acl {
    use std::{
        ffi::OsStr,
        fs::File,
        mem::size_of,
        os::windows::{ffi::OsStrExt as _, io::FromRawHandle as _},
        path::Path,
        ptr::{self, addr_of},
    };

    use anyhow::Context as _;
    use windows_sys::Win32::{
        Foundation::{ERROR_SUCCESS, GENERIC_WRITE, INVALID_HANDLE_VALUE, LocalFree},
        Security::{
            ACCESS_ALLOWED_ACE, ACL,
            Authorization::{
                ConvertStringSecurityDescriptorToSecurityDescriptorW, GetNamedSecurityInfoW,
                SDDL_REVISION_1, SE_FILE_OBJECT, SetNamedSecurityInfoW,
            },
            CreateWellKnownSid, DACL_SECURITY_INFORMATION, EqualSid, GetAce,
            GetSecurityDescriptorControl, GetSecurityDescriptorDacl, INHERITED_ACE, IsValidSid,
            OWNER_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR,
            PSID, SE_DACL_PROTECTED, SECURITY_ATTRIBUTES, SECURITY_MAX_SID_SIZE,
            WinCreatorOwnerRightsSid,
        },
        Storage::FileSystem::{CREATE_NEW, CreateFileW, FILE_ATTRIBUTE_NORMAL},
        System::SystemServices::ACCESS_ALLOWED_ACE_TYPE,
    };

    // `P` prevents inherited directory ACLs from widening access. `OW` is the
    // Windows Owner Rights SID, resolved by the kernel against the new file's
    // owner, and `FA` is full file access.
    const OWNER_ONLY_SDDL: &str = "D:P(A;;FA;;;OW)";

    fn wide(value: &OsStr) -> Vec<u16> {
        value.encode_wide().chain(Some(0)).collect()
    }

    struct LocalSecurityDescriptor(PSECURITY_DESCRIPTOR);

    impl Drop for LocalSecurityDescriptor {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    LocalFree(self.0);
                }
            }
        }
    }

    fn descriptor_from_sddl(sddl: &str) -> anyhow::Result<LocalSecurityDescriptor> {
        let sddl = wide(OsStr::new(sddl));
        let mut descriptor = ptr::null_mut();
        if unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut descriptor,
                ptr::null_mut(),
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error())
                .context("construct owner-only Windows security descriptor");
        }
        Ok(LocalSecurityDescriptor(descriptor))
    }

    /// Create the file with its restrictive DACL already attached. Applying an
    /// ACL after ordinary creation would leave a race in which inherited users
    /// could open a newly written bearer key.
    pub(super) fn create_new_private(path: &Path) -> std::io::Result<File> {
        let descriptor = descriptor_from_sddl(OWNER_ONLY_SDDL)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        let attributes = SECURITY_ATTRIBUTES {
            nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: descriptor.0,
            bInheritHandle: 0,
        };
        let path = wide(path.as_os_str());
        let handle = unsafe {
            CreateFileW(
                path.as_ptr(),
                GENERIC_WRITE,
                0,
                &attributes,
                CREATE_NEW,
                FILE_ATTRIBUTE_NORMAL,
                ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(std::io::Error::last_os_error());
        }
        Ok(unsafe { File::from_raw_handle(handle) })
    }

    fn owner_rights_sid() -> anyhow::Result<[u8; SECURITY_MAX_SID_SIZE as usize]> {
        let mut sid = [0u8; SECURITY_MAX_SID_SIZE as usize];
        let mut length = sid.len() as u32;
        anyhow::ensure!(
            unsafe {
                CreateWellKnownSid(
                    WinCreatorOwnerRightsSid,
                    ptr::null_mut(),
                    sid.as_mut_ptr().cast(),
                    &mut length,
                )
            } != 0,
            "create Windows Owner Rights SID: {}",
            std::io::Error::last_os_error()
        );
        Ok(sid)
    }

    /// Require a protected DACL with exactly one ordinary allow ACE, addressed
    /// either to the file owner directly or to the special Owner Rights SID.
    /// Rejecting unfamiliar ACE forms is intentional: credentials fail closed.
    pub(super) fn ensure_owner_only(path: &Path) -> anyhow::Result<()> {
        let path_wide = wide(path.as_os_str());
        let mut owner: PSID = ptr::null_mut();
        let mut dacl: *mut ACL = ptr::null_mut();
        let mut descriptor: PSECURITY_DESCRIPTOR = ptr::null_mut();
        let status = unsafe {
            GetNamedSecurityInfoW(
                path_wide.as_ptr(),
                SE_FILE_OBJECT,
                OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
                &mut owner,
                ptr::null_mut(),
                &mut dacl,
                ptr::null_mut(),
                &mut descriptor,
            )
        };
        anyhow::ensure!(
            status == ERROR_SUCCESS,
            "inspect Windows ACL at {}: {}",
            path.display(),
            std::io::Error::from_raw_os_error(status as i32)
        );
        let _descriptor = LocalSecurityDescriptor(descriptor);
        anyhow::ensure!(
            !owner.is_null() && unsafe { IsValidSid(owner) } != 0,
            "refusing service file at {}: it has no valid Windows owner",
            path.display()
        );
        anyhow::ensure!(
            !dacl.is_null(),
            "refusing service file at {}: its Windows DACL grants unrestricted access",
            path.display()
        );

        let mut control = 0u16;
        let mut revision = 0u32;
        anyhow::ensure!(
            unsafe { GetSecurityDescriptorControl(descriptor, &mut control, &mut revision) } != 0,
            "inspect Windows ACL control at {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        );
        anyhow::ensure!(
            control & SE_DACL_PROTECTED != 0,
            "refusing service file at {}: its Windows DACL inherits access",
            path.display()
        );
        anyhow::ensure!(
            unsafe { (*dacl).AceCount } == 1,
            "refusing service file at {}: its Windows DACL is not owner-only",
            path.display()
        );

        let mut raw_ace = ptr::null_mut();
        anyhow::ensure!(
            unsafe { GetAce(dacl, 0, &mut raw_ace) } != 0 && !raw_ace.is_null(),
            "inspect Windows ACL entry at {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        );
        let ace = raw_ace.cast::<ACCESS_ALLOWED_ACE>();
        let header = unsafe { &(*ace).Header };
        anyhow::ensure!(
            header.AceType == ACCESS_ALLOWED_ACE_TYPE as u8
                && header.AceFlags as u32 & INHERITED_ACE == 0
                && header.AceSize as usize >= size_of::<ACCESS_ALLOWED_ACE>(),
            "refusing service file at {}: its Windows ACL entry is not a direct owner grant",
            path.display()
        );
        let granted_sid: PSID = unsafe { addr_of!((*ace).SidStart).cast_mut().cast() };
        anyhow::ensure!(
            unsafe { IsValidSid(granted_sid) } != 0,
            "refusing service file at {}: its Windows ACL has an invalid SID",
            path.display()
        );
        let owner_rights = owner_rights_sid()?;
        let owner_rights = owner_rights.as_ptr().cast_mut().cast();
        anyhow::ensure!(
            unsafe { EqualSid(granted_sid, owner) } != 0
                || unsafe { EqualSid(granted_sid, owner_rights) } != 0,
            "refusing service file at {}: its Windows ACL grants another principal access",
            path.display()
        );
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn set_acl_for_test(path: &Path, sddl: &str) -> anyhow::Result<()> {
        let descriptor = descriptor_from_sddl(sddl)?;
        let mut present = 0;
        let mut defaulted = 0;
        let mut dacl = ptr::null_mut();
        anyhow::ensure!(
            unsafe {
                GetSecurityDescriptorDacl(descriptor.0, &mut present, &mut dacl, &mut defaulted)
            } != 0
                && present != 0
                && !dacl.is_null(),
            "read test DACL: {}",
            std::io::Error::last_os_error()
        );
        let path = wide(path.as_os_str());
        let status = unsafe {
            SetNamedSecurityInfoW(
                path.as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                ptr::null_mut(),
                ptr::null_mut(),
                dacl,
                ptr::null_mut(),
            )
        };
        anyhow::ensure!(
            status == ERROR_SUCCESS,
            "set test DACL: {}",
            std::io::Error::from_raw_os_error(status as i32)
        );
        Ok(())
    }
}

/// The predictable port used by `brazierd --service` when `--port` is omitted.
pub const DEFAULT_SERVICE_PORT: u16 = 7614;

const KEY_FILE: &str = "service/api-key";
const READY_FILE: &str = "service/ready.json";

#[derive(Debug, Serialize)]
pub struct ReadyDescriptor<'a> {
    pub address: &'a str,
    pub pid: u32,
    pub product: &'static str,
    pub version: &'static str,
    pub management_api: ManagementApiVersion,
}

#[derive(Debug, Serialize)]
pub struct ManagementApiVersion {
    pub major: u8,
    pub minor: u8,
}

pub fn api_key_path(data_dir: &Path) -> PathBuf {
    data_dir.join(KEY_FILE)
}

pub fn ready_path(data_dir: &Path) -> PathBuf {
    data_dir.join(READY_FILE)
}

/// Return the service key already assigned to this data directory, or create
/// it with owner-only permissions. An explicitly supplied key is intentionally
/// not copied into this file: deployment tooling remains its source of truth.
pub fn service_api_key(data_dir: &Path, configured: Option<String>) -> anyhow::Result<String> {
    if let Some(key) = configured {
        anyhow::ensure!(!key.trim().is_empty(), "the API key must not be empty");
        return Ok(key);
    }

    let path = api_key_path(data_dir);
    if path.exists() {
        ensure_private(&path)?;
        let key = fs::read_to_string(&path)
            .with_context(|| format!("read service API key at {}", path.display()))?;
        let key = key.trim().to_owned();
        anyhow::ensure!(
            !key.is_empty(),
            "service API key at {} is empty",
            path.display()
        );
        return Ok(key);
    }

    let parent = path.parent().expect("service key has a parent");
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let key = format!("brazier_{}", Uuid::new_v4().simple());
    write_new_private(&path, key.as_bytes())?;
    // `create_new` may have lost a concurrent-start race. Re-read the winner
    // so every process uses the same durable credential.
    let stored = fs::read_to_string(&path)
        .with_context(|| format!("read service API key at {}", path.display()))?;
    let stored = stored.trim().to_owned();
    anyhow::ensure!(
        !stored.is_empty(),
        "service API key at {} is empty",
        path.display()
    );
    Ok(stored)
}

fn write_new_private(path: &Path, contents: &[u8]) -> anyhow::Result<()> {
    use std::io::Write as _;
    #[cfg(not(windows))]
    let mut options = fs::OpenOptions::new();
    #[cfg(not(windows))]
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    #[cfg(windows)]
    let opened = windows_acl::create_new_private(path);
    #[cfg(not(windows))]
    let opened = options.open(path);
    match opened {
        Ok(mut file) => {
            file.write_all(contents)
                .with_context(|| format!("write service API key at {}", path.display()))?;
            drop(file);
            ensure_private(path)?;
            Ok(())
        }
        // A concurrent service startup created it first. Treat its key as the
        // authoritative one instead of giving the two processes different keys.
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            ensure_private(path)?;
            Ok(())
        }
        Err(error) => {
            Err(error).with_context(|| format!("create service API key at {}", path.display()))
        }
    }
}

fn ensure_private(path: &Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = fs::metadata(path)
            .with_context(|| format!("inspect service API key at {}", path.display()))?
            .permissions()
            .mode()
            & 0o777;
        anyhow::ensure!(
            mode & 0o077 == 0,
            "refusing service API key at {}: it is readable by group or others",
            path.display()
        );
    }
    #[cfg(windows)]
    windows_acl::ensure_owner_only(path)?;
    Ok(())
}

/// Atomically publish the endpoint only after the listener has bound. It has
/// no bearer credential, but remains owner-only operational metadata.
pub fn write_ready_descriptor(path: &Path, address: &str) -> anyhow::Result<()> {
    let descriptor = ReadyDescriptor {
        address,
        pid: std::process::id(),
        product: "brazier",
        version: env!("CARGO_PKG_VERSION"),
        management_api: ManagementApiVersion { major: 1, minor: 0 },
    };
    let bytes =
        serde_json::to_vec_pretty(&descriptor).context("encode service ready descriptor")?;
    let parent = path
        .parent()
        .context("service ready descriptor path has no parent")?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let temporary = parent.join(format!(".ready-{}.tmp", Uuid::new_v4()));
    write_new_private(&temporary, &bytes)?;
    #[cfg(windows)]
    if path.exists() {
        fs::remove_file(path).with_context(|| format!("replace {}", path.display()))?;
    }
    fs::rename(&temporary, path).with_context(|| format!("publish {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_key_survives_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let first = service_api_key(dir.path(), None).unwrap();
        let second = service_api_key(dir.path(), None).unwrap();
        assert_eq!(first, second);
        assert!(first.starts_with("brazier_"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                fs::metadata(api_key_path(dir.path()))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn explicit_key_is_not_persisted() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            service_api_key(dir.path(), Some("from-secret-store".into())).unwrap(),
            "from-secret-store"
        );
        assert!(!api_key_path(dir.path()).exists());
    }

    #[test]
    fn ready_descriptor_has_no_credential_and_is_private() {
        let dir = tempfile::tempdir().unwrap();
        let path = ready_path(dir.path());
        write_ready_descriptor(&path, "http://127.0.0.1:7614").unwrap();
        let json: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(json["address"], "http://127.0.0.1:7614");
        assert_eq!(json["management_api"]["major"], 1);
        assert!(json.get("api_key").is_none());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn refuses_an_exposed_existing_key() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = tempfile::tempdir().unwrap();
        let path = api_key_path(dir.path());
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "secret").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(service_api_key(dir.path(), None).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn generated_service_files_have_protected_owner_only_acls() {
        let dir = tempfile::tempdir().unwrap();
        service_api_key(dir.path(), None).unwrap();
        windows_acl::ensure_owner_only(&api_key_path(dir.path())).unwrap();

        let ready = ready_path(dir.path());
        write_ready_descriptor(&ready, "http://127.0.0.1:7614").unwrap();
        windows_acl::ensure_owner_only(&ready).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn refuses_an_existing_key_with_a_broad_windows_acl() {
        let dir = tempfile::tempdir().unwrap();
        let path = api_key_path(dir.path());
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "exposed-secret").unwrap();
        windows_acl::set_acl_for_test(&path, "D:P(A;;FA;;;OW)(A;;GR;;;WD)").unwrap();

        let error = service_api_key(dir.path(), None).unwrap_err();
        assert!(error.to_string().contains("not owner-only"));
    }

    #[cfg(windows)]
    #[test]
    fn ready_descriptor_replacement_repairs_a_broad_windows_acl() {
        let dir = tempfile::tempdir().unwrap();
        let path = ready_path(dir.path());
        write_ready_descriptor(&path, "http://127.0.0.1:7614").unwrap();
        windows_acl::set_acl_for_test(&path, "D:P(A;;FA;;;OW)(A;;GR;;;WD)").unwrap();
        assert!(windows_acl::ensure_owner_only(&path).is_err());

        write_ready_descriptor(&path, "http://127.0.0.1:7615").unwrap();
        windows_acl::ensure_owner_only(&path).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(json["address"], "http://127.0.0.1:7615");
    }
}

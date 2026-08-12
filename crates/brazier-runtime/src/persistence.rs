use std::path::Path;

use anyhow::Context;
use serde::Serialize;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

#[cfg(windows)]
mod windows_acl {
    use std::{
        ffi::OsStr,
        fs::File,
        mem::size_of,
        os::windows::{ffi::OsStrExt as _, io::FromRawHandle as _},
        path::Path,
        ptr,
    };
    use windows_sys::Win32::{
        Foundation::{GENERIC_WRITE, INVALID_HANDLE_VALUE, LocalFree},
        Security::{
            Authorization::{
                ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
            },
            PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES,
        },
        Storage::FileSystem::{CREATE_NEW, CreateFileW, FILE_ATTRIBUTE_NORMAL},
    };

    fn wide(value: &OsStr) -> Vec<u16> {
        value.encode_wide().chain(Some(0)).collect()
    }

    pub(super) fn create_new_private(path: &Path) -> std::io::Result<File> {
        let sddl = wide(OsStr::new("D:P(A;;FA;;;OW)"));
        let mut descriptor: PSECURITY_DESCRIPTOR = ptr::null_mut();
        if unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut descriptor,
                ptr::null_mut(),
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        let attributes = SECURITY_ATTRIBUTES {
            nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: descriptor,
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
        unsafe { LocalFree(descriptor) };
        if handle == INVALID_HANDLE_VALUE {
            return Err(std::io::Error::last_os_error());
        }
        Ok(unsafe { File::from_raw_handle(handle) })
    }
}

#[cfg(windows)]
async fn open_private_temporary(path: &Path) -> std::io::Result<tokio::fs::File> {
    windows_acl::create_new_private(path).map(tokio::fs::File::from_std)
}

#[cfg(not(windows))]
async fn open_private_temporary(path: &Path) -> std::io::Result<tokio::fs::File> {
    let mut options = tokio::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    options.open(path).await
}

pub(crate) async fn write_json(
    path: &Path,
    value: &(impl Serialize + ?Sized),
    label: &str,
) -> anyhow::Result<()> {
    let bytes = serde_json::to_vec_pretty(value).with_context(|| format!("encode {label}"))?;
    write(path, &bytes, label).await
}

pub(crate) async fn write(path: &Path, bytes: &[u8], label: &str) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("{label} path has no parent"))?;
    tokio::fs::create_dir_all(parent)
        .await
        .with_context(|| format!("create {label} directory"))?;
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("state");
    let temporary = parent.join(format!(".{name}.{}.tmp", Uuid::new_v4()));

    let mut file = open_private_temporary(&temporary)
        .await
        .with_context(|| format!("create temporary {label}"))?;
    if let Err(error) = file.write_all(bytes).await {
        drop(file);
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(error).with_context(|| format!("write temporary {label}"));
    }
    file.flush()
        .await
        .with_context(|| format!("flush temporary {label}"))?;
    drop(file);

    #[cfg(windows)]
    if path.is_file() {
        tokio::fs::remove_file(path)
            .await
            .with_context(|| format!("replace {label}"))?;
    }
    if let Err(error) = tokio::fs::rename(&temporary, path).await {
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(error).with_context(|| format!("commit {label}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn concurrent_writers_do_not_share_a_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let writer_one = json!({ "writer": 1 });
        let writer_two = json!({ "writer": 2 });
        let (first, second) = tokio::join!(
            write_json(&path, &writer_one, "settings"),
            write_json(&path, &writer_two, "settings")
        );
        first.unwrap();
        second.unwrap();

        let value: serde_json::Value =
            serde_json::from_slice(&tokio::fs::read(&path).await.unwrap()).unwrap();
        assert!(value["writer"] == 1 || value["writer"] == 2);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
}

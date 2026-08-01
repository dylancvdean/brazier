//! Desktop computer-use target and OS permission probes.
//!
//! Capture and input injection are gated on platform permissions. Until those
//! drivers are fully wired, probing reports honest status and execution fails
//! closed.

use brazier_protocol::computer_types::{
    ComputerAction, ComputerActionResult, ComputerActionStatus, OsPermissionState,
    OsPermissionStatus,
};

/// Probe the host for screen capture and input-injection capability.
pub fn probe_os_permissions() -> OsPermissionStatus {
    #[cfg(target_os = "macos")]
    {
        probe_macos()
    }
    #[cfg(target_os = "linux")]
    {
        probe_linux()
    }
    #[cfg(target_os = "windows")]
    {
        OsPermissionStatus {
            platform: "windows".into(),
            display_server: "win32".into(),
            screen_capture: OsPermissionState::Unsupported,
            input_injection: OsPermissionState::Unsupported,
            detail: Some("Windows desktop computer use is not implemented yet.".into()),
            settings_hint: None,
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        OsPermissionStatus {
            platform: std::env::consts::OS.into(),
            display_server: "unknown".into(),
            screen_capture: OsPermissionState::Unsupported,
            input_injection: OsPermissionState::Unsupported,
            detail: Some("Desktop computer use is unsupported on this platform.".into()),
            settings_hint: None,
        }
    }
}

#[cfg(target_os = "macos")]
fn probe_macos() -> OsPermissionStatus {
    // TCC APIs are not linked yet; report Unknown so the UI can deep-link to
    // System Settings rather than falsely claiming a grant.
    OsPermissionStatus {
        platform: "macos".into(),
        display_server: "quartz".into(),
        screen_capture: OsPermissionState::Unknown,
        input_injection: OsPermissionState::Unknown,
        detail: Some(
            "macOS Screen Recording and Accessibility permissions are required for desktop control."
                .into(),
        ),
        settings_hint: Some(
            "Open System Settings → Privacy & Security → Screen Recording and Accessibility."
                .into(),
        ),
    }
}

#[cfg(target_os = "linux")]
fn probe_linux() -> OsPermissionStatus {
    let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
    let x11 = std::env::var_os("DISPLAY").is_some();
    if wayland {
        let portal = std::path::Path::new(
            "/usr/share/dbus-1/services/org.freedesktop.portal.Desktop.service",
        )
        .exists()
            || std::path::Path::new(
                "/usr/local/share/dbus-1/services/org.freedesktop.portal.Desktop.service",
            )
            .exists();
        OsPermissionStatus {
            platform: "linux".into(),
            display_server: "wayland".into(),
            screen_capture: if portal {
                OsPermissionState::Unknown
            } else {
                OsPermissionState::Missing
            },
            input_injection: if portal {
                OsPermissionState::Unknown
            } else {
                OsPermissionState::Missing
            },
            detail: Some(
                "Wayland desktop control uses xdg-desktop-portal ScreenCast and RemoteDesktop. Brazier fails closed until both grants exist."
                    .into(),
            ),
            settings_hint: Some(
                "Approve ScreenCast and RemoteDesktop prompts from your desktop portal when starting a desktop session."
                    .into(),
            ),
        }
    } else if x11 {
        OsPermissionStatus {
            platform: "linux".into(),
            display_server: "x11".into(),
            screen_capture: OsPermissionState::Unknown,
            input_injection: OsPermissionState::Unknown,
            detail: Some(
                "X11 capture (XShm/XGetImage) and input (XTest/uinput) are available in principle; the desktop driver is not enabled yet."
                    .into(),
            ),
            settings_hint: Some(
                "Prefer an isolated browser computer-use session until the X11 driver ships."
                    .into(),
            ),
        }
    } else {
        OsPermissionStatus {
            platform: "linux".into(),
            display_server: "none".into(),
            screen_capture: OsPermissionState::Missing,
            input_injection: OsPermissionState::Missing,
            detail: Some("No WAYLAND_DISPLAY or DISPLAY is set.".into()),
            settings_hint: None,
        }
    }
}

pub fn desktop_permitted(status: &OsPermissionStatus) -> bool {
    matches!(status.screen_capture, OsPermissionState::Granted)
        && matches!(status.input_injection, OsPermissionState::Granted)
}

/// Fail-closed desktop executor used until platform drivers land.
pub async fn execute_desktop_action(action: &ComputerAction) -> ComputerActionResult {
    let status = probe_os_permissions();
    if !desktop_permitted(&status) {
        return ComputerActionResult {
            status: ComputerActionStatus::Refused,
            message: Some(
                status
                    .detail
                    .clone()
                    .unwrap_or_else(|| "Desktop computer use is not permitted.".into()),
            ),
            screenshot_base64: None,
            mime_type: None,
            viewport: None,
            url: None,
            title: None,
            needs_approval: false,
            approval_id: None,
        };
    }

    ComputerActionResult {
        status: ComputerActionStatus::Error,
        message: Some(format!(
            "Desktop driver is not implemented for {} yet (action {}).",
            status.platform,
            action.kind()
        )),
        screenshot_base64: None,
        mime_type: None,
        viewport: None,
        url: None,
        title: None,
        needs_approval: false,
        approval_id: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_platform() {
        let status = probe_os_permissions();
        assert!(!status.platform.is_empty());
        assert!(!status.display_server.is_empty());
    }
}

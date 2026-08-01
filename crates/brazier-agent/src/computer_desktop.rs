//! Native desktop Computer Use drivers.
//!
//! These drivers intentionally use the OS' normal permission boundary: Quartz
//! prompts for Screen Recording/Accessibility, X11 trusts the logged-in X
//! client, and Wayland requires compositor-approved tools/portals.  Nothing is
//! silently emulated when the required integration is absent.

#[cfg(target_os = "macos")]
use std::process::Stdio;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use brazier_protocol::computer_types::{
    ComputerAction, ComputerActionResult, ComputerActionStatus, ComputerViewport,
    OsPermissionState, OsPermissionStatus,
};
#[cfg(target_os = "macos")]
use tokio::process::Command;

#[cfg(target_os = "macos")]
fn binary(name: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|path| path.join(name).is_file()))
}

fn result(status: ComputerActionStatus, message: Option<String>) -> ComputerActionResult {
    ComputerActionResult {
        status,
        message,
        screenshot_base64: None,
        mime_type: None,
        viewport: None,
        url: None,
        title: None,
        needs_approval: false,
        approval_id: None,
    }
}

/// Probe the host for capture and injection capability. `Granted` means the
/// integration is present; the first action may still trigger the OS consent UI.
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
    OsPermissionStatus { platform: "macos".into(), display_server: "quartz".into(),
        screen_capture: if binary("screencapture") { OsPermissionState::Granted } else { OsPermissionState::Missing },
        input_injection: if binary("osascript") { OsPermissionState::Granted } else { OsPermissionState::Missing },
        detail: Some("Brazier uses macOS Screen Recording for capture and Accessibility through System Events for input.".into()),
        settings_hint: Some("If an action is refused, enable Brazier in System Settings → Privacy & Security → Screen Recording and Accessibility.".into()) }
}

#[cfg(target_os = "linux")]
fn probe_linux() -> OsPermissionStatus {
    let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
    let x11 = std::env::var_os("DISPLAY").is_some();
    if wayland {
        let portal = std::env::var_os("DBUS_SESSION_BUS_ADDRESS").is_some();
        let pipewire = std::env::var_os("XDG_RUNTIME_DIR")
            .map(std::path::PathBuf::from)
            .is_some_and(|dir| dir.join("pipewire-0").exists());
        let ready = portal && pipewire;
        OsPermissionStatus { platform: "linux".into(), display_server: "wayland".into(),
            screen_capture: if ready { OsPermissionState::Granted } else { OsPermissionState::Missing },
            input_injection: if ready { OsPermissionState::Granted } else { OsPermissionState::Missing },
            detail: Some("Wayland uses Brazier's built-in XDG ScreenCast, Screenshot, and RemoteDesktop portal client. No root service or ydotoold is used.".into()),
            settings_hint: Some(if !portal { "Start Brazier from your graphical login session so DBUS_SESSION_BUS_ADDRESS is present." } else if !pipewire { "Start PipeWire and WirePlumber in this graphical session, then return here to request Screen Share and Remote Desktop access." } else { "Use Request access below, then approve Brazier in the compositor's Screen Share and Remote Desktop prompt." }.into()) }
    } else if x11 {
        OsPermissionStatus { platform: "linux".into(), display_server: "x11".into(),
            screen_capture: OsPermissionState::Granted,
            input_injection: OsPermissionState::Granted,
            detail: Some("X11 uses Brazier's direct X11 GetImage and XTEST client; no xdotool or ImageMagick is required.".into()),
            settings_hint: Some("Brazier must be started in the logged-in X11 session.".into()) }
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

/// Request all OS permissions that can be requested programmatically. This is
/// intentionally invoked from Settings, before a model is allowed to act.
pub async fn request_os_permissions() -> Result<OsPermissionStatus, String> {
    #[cfg(target_os = "linux")]
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        crate::computer_portal::request_permissions().await?;
    }
    #[cfg(target_os = "macos")]
    {
        let _ = screenshot().await?;
        command("osascript", &["-e".into(), "tell application \"System Events\" to get name of first process".into()]).await?;
    }
    Ok(probe_os_permissions())
}

#[cfg(target_os = "macos")]
async fn command(program: &str, args: &[String]) -> Result<Vec<u8>, String> {
    let mut command = Command::new(program);
    command.args(args);
    let output = command
        .output()
        .await
        .map_err(|e| format!("start {program}: {e}"))?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        let detail = if detail.is_empty() {
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        } else {
            detail
        };
        Err(if detail.is_empty() {
            format!("{program} exited with status {}", output.status)
        } else {
            detail
        })
    }
}

fn png_viewport(bytes: &[u8]) -> Option<ComputerViewport> {
    if bytes.len() < 24 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    Some(ComputerViewport {
        width: u32::from_be_bytes(bytes[16..20].try_into().ok()?),
        height: u32::from_be_bytes(bytes[20..24].try_into().ok()?),
        device_pixel_ratio: Some(1.0),
    })
}

async fn screenshot() -> Result<ComputerActionResult, String> {
    #[cfg(target_os = "macos")]
    let bytes = command(
        "screencapture",
        &["-x".into(), "-t".into(), "png".into(), "-".into()],
    )
    .await?;
    #[cfg(all(target_os = "linux"))]
    let bytes = if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        crate::computer_portal::screenshot().await?
    } else {
        tokio::task::spawn_blocking(crate::computer_x11::screenshot)
            .await.map_err(|error| error.to_string())??
    };
    Ok(ComputerActionResult {
        status: ComputerActionStatus::Ok,
        message: None,
        screenshot_base64: Some(STANDARD.encode(&bytes)),
        mime_type: Some("image/png".into()),
        viewport: png_viewport(&bytes),
        url: None,
        title: None,
        needs_approval: false,
        approval_id: None,
    })
}

#[cfg(target_os = "macos")]
async fn mac_action(action: &ComputerAction) -> Result<(), String> {
    let script = match action {
        ComputerAction::LeftClick { x, y } => {
            format!("tell application \"System Events\" to click at {{{x}, {y}}}")
        }
        ComputerAction::RightClick { x, y } => format!(
            "tell application \"System Events\" to key down control\nclick at {{{x}, {y}}}\nkey up control"
        ),
        ComputerAction::DoubleClick { x, y } => {
            format!("tell application \"System Events\" to click at {{{x}, {y}}} twice")
        }
        ComputerAction::TripleClick { x, y } => {
            format!("tell application \"System Events\" to click at {{{x}, {y}}} three times")
        }
        ComputerAction::MouseMove { x, y } => {
            format!("tell application \"System Events\" to move mouse to {{{x}, {y}}}")
        }
        ComputerAction::LeftClickDrag {
            start_x,
            start_y,
            end_x,
            end_y,
        } => format!(
            "tell application \"System Events\" to drag from {{{start_x}, {start_y}}} to {{{end_x}, {end_y}}}"
        ),
        ComputerAction::Keypress { keys } => format!(
            "tell application \"System Events\" to key code {}",
            mac_key(keys)?
        ),
        ComputerAction::Scroll { delta_y, .. } => format!(
            "tell application \"System Events\" to scroll {}",
            -delta_y.round() as i64
        ),
        ComputerAction::Type { text } => {
            let mut child = Command::new("pbcopy")
                .stdin(Stdio::piped())
                .spawn()
                .map_err(|e| e.to_string())?;
            use tokio::io::AsyncWriteExt;
            let Some(mut stdin) = child.stdin.take() else {
                return Err("pbcopy did not provide stdin".into());
            };
            stdin
                .write_all(text.as_bytes())
                .await
                .map_err(|e| e.to_string())?;
            drop(stdin);
            child.wait().await.map_err(|e| e.to_string())?;
            "tell application \"System Events\" to keystroke \"v\" using command down".into()
        }
        ComputerAction::Wait { milliseconds } => {
            tokio::time::sleep(std::time::Duration::from_millis(*milliseconds)).await;
            return Ok(());
        }
        _ => return Err("This action is not available on the desktop.".into()),
    };
    command("osascript", &["-e".into(), script])
        .await
        .map(|_| ())
}
#[cfg(target_os = "macos")]
fn mac_key(keys: &[String]) -> Result<u16, String> {
    let key = keys
        .last()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match key.as_str() {
        "escape" | "esc" => Ok(53),
        "enter" | "return" => Ok(36),
        "tab" => Ok(48),
        "space" => Ok(49),
        "backspace" => Ok(51),
        _ => Err(format!(
            "unsupported macOS key combination: {}",
            keys.join("+")
        )),
    }
}

#[cfg(target_os = "linux")]
async fn linux_action(action: &ComputerAction) -> Result<(), String> {
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        return wayland_action(action).await;
    }
    tokio::task::spawn_blocking({
        let action = action.clone();
        move || match action {
            ComputerAction::LeftClick { x, y } => crate::computer_x11::click(x, y, 1, 1),
            ComputerAction::RightClick { x, y } => crate::computer_x11::click(x, y, 3, 1),
            ComputerAction::DoubleClick { x, y } => crate::computer_x11::click(x, y, 1, 2),
            ComputerAction::TripleClick { x, y } => crate::computer_x11::click(x, y, 1, 3),
            ComputerAction::MouseMove { x, y } => crate::computer_x11::move_to(x, y),
            ComputerAction::LeftClickDrag { start_x, start_y, end_x, end_y } => crate::computer_x11::drag(start_x, start_y, end_x, end_y),
            ComputerAction::Type { text } => crate::computer_x11::type_text(&text),
            ComputerAction::Keypress { keys } => { for key in keys { crate::computer_x11::key(wayland_keysym(&key)?)?; } Ok(()) }
            ComputerAction::Scroll { delta_y, .. } => crate::computer_x11::scroll(delta_y),
            _ => Err("This action is not available on the desktop.".into()),
        }
    }).await.map_err(|error| error.to_string())?
}
#[cfg(target_os = "linux")]
async fn wayland_action(action: &ComputerAction) -> Result<(), String> {
    if let ComputerAction::Wait { milliseconds } = action {
        tokio::time::sleep(std::time::Duration::from_millis(*milliseconds)).await;
        return Ok(());
    }
    match action {
        ComputerAction::MouseMove { x, y } => crate::computer_portal::pointer_motion(*x, *y).await,
        ComputerAction::LeftClick { x, y } => crate::computer_portal::pointer_button(*x, *y, 272, 1).await,
        ComputerAction::RightClick { x, y } => crate::computer_portal::pointer_button(*x, *y, 273, 1).await,
        ComputerAction::DoubleClick { x, y } => crate::computer_portal::pointer_button(*x, *y, 272, 2).await,
        ComputerAction::TripleClick { x, y } => crate::computer_portal::pointer_button(*x, *y, 272, 3).await,
        ComputerAction::LeftClickDrag { start_x, start_y, end_x, end_y } => crate::computer_portal::pointer_drag(*start_x, *start_y, *end_x, *end_y).await,
        ComputerAction::Scroll { delta_x, delta_y, .. } => crate::computer_portal::scroll(*delta_x, *delta_y).await,
        ComputerAction::Type { text } => crate::computer_portal::type_text(text).await,
        ComputerAction::Keypress { keys } => {
            for key in keys { crate::computer_portal::key(wayland_keysym(key)?).await?; }
            Ok(())
        }
        _ => Err("This action is not available through the Wayland desktop portal.".into()),
    }
}

#[cfg(target_os = "linux")]
fn wayland_keysym(key: &str) -> Result<u32, String> {
    let key = key.to_ascii_lowercase();
    match key.as_str() {
        "escape" | "esc" => Ok(0xff1b),
        "enter" | "return" => Ok(0xff0d),
        "tab" => Ok(0xff09),
        "space" => Ok(0x20),
        "backspace" => Ok(0xff08),
        "delete" => Ok(0xffff),
        "arrowup" | "up" => Ok(0xff52),
        "arrowdown" | "down" => Ok(0xff54),
        "arrowleft" | "left" => Ok(0xff51),
        "arrowright" | "right" => Ok(0xff53),
        value if value.len() == 1 => Ok(value.as_bytes()[0] as u32),
        _ => Err(format!("unsupported Wayland key: {key}")),
    }
}

pub async fn execute_desktop_action(action: &ComputerAction) -> ComputerActionResult {
    let status = probe_os_permissions();
    if !desktop_permitted(&status) {
        return result(ComputerActionStatus::Refused, status.detail);
    }
    if matches!(action, ComputerAction::Screenshot) {
        return screenshot().await.unwrap_or_else(|e| {
            result(
                ComputerActionStatus::Error,
                Some(format!("Desktop screenshot failed: {e}")),
            )
        });
    }
    #[cfg(target_os = "macos")]
    let execution = mac_action(action).await;
    #[cfg(target_os = "linux")]
    let execution = linux_action(action).await;
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let execution: Result<(), String> = Err("unsupported platform".into());
    match execution {
        Ok(()) => screenshot().await.unwrap_or_else(|e| {
            result(
                ComputerActionStatus::Error,
                Some(format!("Action succeeded but screenshot failed: {e}")),
            )
        }),
        Err(e) => result(
            ComputerActionStatus::Error,
            Some(format!("Desktop {} failed: {e}", action.kind())),
        ),
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

    #[test]
    fn reads_png_dimensions_without_decoding_the_image() {
        let mut png = vec![0; 24];
        png[..8].copy_from_slice(b"\x89PNG\r\n\x1a\n");
        png[16..20].copy_from_slice(&1920_u32.to_be_bytes());
        png[20..24].copy_from_slice(&1080_u32.to_be_bytes());
        assert_eq!(
            png_viewport(&png).map(|size| (size.width, size.height)),
            Some((1920, 1080))
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn maps_common_portal_keysyms() {
        assert_eq!(wayland_keysym("Escape").unwrap(), 0xff1b);
        assert_eq!(wayland_keysym("ArrowLeft").unwrap(), 0xff51);
    }
}

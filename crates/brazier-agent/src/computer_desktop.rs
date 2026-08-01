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
use tokio::process::Command;

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
        let portal = binary("gdbus") || binary("busctl");
        OsPermissionStatus { platform: "linux".into(), display_server: "wayland".into(),
            screen_capture: if binary("grim") || binary("spectacle") || binary("gnome-screenshot") { OsPermissionState::Granted } else { OsPermissionState::Missing },
            input_injection: if binary("ydotool") && portal { OsPermissionState::Granted } else { OsPermissionState::Missing },
            detail: Some("Wayland uses grim when the compositor exposes screencopy, or portal-aware Spectacle/GNOME Screenshot otherwise; ydotool performs input.".into()),
            settings_hint: Some("Install grim or Spectacle/GNOME Screenshot plus ydotool, start ydotoold, and approve any ScreenCast/RemoteDesktop portal prompt.".into()) }
    } else if x11 {
        OsPermissionStatus { platform: "linux".into(), display_server: "x11".into(),
            screen_capture: if binary("magick") || binary("import") || binary("gnome-screenshot") { OsPermissionState::Granted } else { OsPermissionState::Missing },
            input_injection: if binary("xdotool") { OsPermissionState::Granted } else { OsPermissionState::Missing },
            detail: Some("X11 uses ImageMagick/gnome-screenshot for capture and XTest through xdotool for input.".into()),
            settings_hint: Some("Install xdotool plus ImageMagick (or gnome-screenshot).".into()) }
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

async fn command(program: &str, args: &[String]) -> Result<Vec<u8>, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .await
        .map_err(|e| format!("start {program}: {e}"))?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_owned())
    }
}

/// Portal-aware desktop utilities write to a file rather than stdout. Keep the
/// image only long enough to return it to the broker, then remove it.
async fn screenshot_file(program: &str, args: Vec<String>) -> Result<Vec<u8>, String> {
    let path = std::env::temp_dir().join(format!("brazier-computer-{}.png", uuid::Uuid::new_v4()));
    let mut args = args;
    args.push(path.to_string_lossy().into_owned());
    let captured = command(program, &args).await;
    let bytes = if captured.is_ok() {
        tokio::fs::read(&path).await.map_err(|e| e.to_string())
    } else {
        Err(captured.unwrap_err())
    };
    let _ = tokio::fs::remove_file(&path).await;
    bytes
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
        match command("grim", &["-".into()]).await {
            Ok(bytes) => bytes,
            Err(grim_error) if binary("spectacle") => {
                screenshot_file("spectacle", vec!["-b".into(), "-n".into(), "-o".into()])
                    .await
                    .map_err(|spectacle_error| {
                        format!("grim: {grim_error}; Spectacle portal fallback: {spectacle_error}")
                    })?
            }
            Err(grim_error) if binary("gnome-screenshot") => {
                screenshot_file("gnome-screenshot", vec!["-f".into()])
                    .await
                    .map_err(|gnome_error| {
                        format!(
                            "grim: {grim_error}; GNOME Screenshot portal fallback: {gnome_error}"
                        )
                    })?
            }
            Err(error) => return Err(error),
        }
    } else if binary("magick") {
        command(
            "magick",
            &[
                "import".into(),
                "-window".into(),
                "root".into(),
                "png:-".into(),
            ],
        )
        .await?
    } else if binary("import") {
        command("import", &["-window".into(), "root".into(), "png:-".into()]).await?
    } else {
        command("gnome-screenshot", &["-f".into(), "/dev/stdout".into()]).await?
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
    let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
    let (program, args): (&str, Vec<String>) = if wayland {
        ("ydotool", wayland_args(action)?)
    } else {
        ("xdotool", x11_args(action)?)
    };
    command(program, &args).await.map(|_| ())
}
#[cfg(target_os = "linux")]
fn x11_args(a: &ComputerAction) -> Result<Vec<String>, String> {
    Ok(match a {
        ComputerAction::LeftClick { x, y } => vec![
            "mousemove".into(),
            x.to_string(),
            y.to_string(),
            "click".into(),
            "1".into(),
        ],
        ComputerAction::RightClick { x, y } => vec![
            "mousemove".into(),
            x.to_string(),
            y.to_string(),
            "click".into(),
            "3".into(),
        ],
        ComputerAction::DoubleClick { x, y } => vec![
            "mousemove".into(),
            x.to_string(),
            y.to_string(),
            "click".into(),
            "--repeat".into(),
            "2".into(),
            "1".into(),
        ],
        ComputerAction::TripleClick { x, y } => vec![
            "mousemove".into(),
            x.to_string(),
            y.to_string(),
            "click".into(),
            "--repeat".into(),
            "3".into(),
            "1".into(),
        ],
        ComputerAction::MouseMove { x, y } => {
            vec!["mousemove".into(), x.to_string(), y.to_string()]
        }
        ComputerAction::LeftClickDrag {
            start_x,
            start_y,
            end_x,
            end_y,
        } => vec![
            "mousemove".into(),
            start_x.to_string(),
            start_y.to_string(),
            "mousedown".into(),
            "1".into(),
            "mousemove".into(),
            end_x.to_string(),
            end_y.to_string(),
            "mouseup".into(),
            "1".into(),
        ],
        ComputerAction::Type { text } => {
            vec!["type".into(), "--clearmodifiers".into(), text.clone()]
        }
        ComputerAction::Keypress { keys } => vec!["key".into(), keys.join("+")],
        ComputerAction::Scroll { delta_y, .. } => vec![
            "click".into(),
            "--repeat".into(),
            delta_y.abs().round().to_string(),
            if *delta_y < 0.0 { "4" } else { "5" }.into(),
        ],
        ComputerAction::Wait { milliseconds } => {
            vec!["sleep".into(), (*milliseconds as f64 / 1000.0).to_string()]
        }
        _ => return Err("This action is not available on the desktop.".into()),
    })
}
#[cfg(target_os = "linux")]
fn wayland_args(a: &ComputerAction) -> Result<Vec<String>, String> {
    let out = match a {
        ComputerAction::MouseMove { x, y } => vec![
            "mousemove".into(),
            x.round().to_string(),
            y.round().to_string(),
        ],
        ComputerAction::LeftClick { x, y } => vec![
            "mousemove".into(),
            x.round().to_string(),
            y.round().to_string(),
            "click".into(),
            "0xC0".into(),
        ],
        ComputerAction::RightClick { x, y } => vec![
            "mousemove".into(),
            x.round().to_string(),
            y.round().to_string(),
            "click".into(),
            "0xC1".into(),
        ],
        ComputerAction::Type { text } => vec!["type".into(), text.clone()],
        ComputerAction::Keypress { keys }
            if keys.len() == 1 && matches!(keys[0].as_str(), "Escape" | "Esc") =>
        {
            vec!["key".into(), "1:1".into(), "1:0".into()]
        }
        ComputerAction::Wait { milliseconds } => {
            vec!["sleep".into(), (*milliseconds as f64 / 1000.0).to_string()]
        }
        _ => return Err("This Wayland action is not supported by ydotool yet.".into()),
    };
    Ok(out)
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
}

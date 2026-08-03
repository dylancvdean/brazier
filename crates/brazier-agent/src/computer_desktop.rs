//! Native desktop Computer Use drivers.
//!
//! These drivers intentionally use the OS' normal permission boundary: Quartz
//! prompts for Screen Recording/Accessibility, X11 trusts the logged-in X
//! client, and Wayland requires compositor-approved tools/portals.  Nothing is
//! silently emulated when the required integration is absent.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use brazier_protocol::computer_types::{
    ComputerAction, ComputerActionResult, ComputerActionStatus, ComputerViewport,
    OsPermissionState, OsPermissionStatus,
};
#[cfg(target_os = "macos")]
use tokio::process::Command;
#[cfg(target_os = "macos")]
use uuid::Uuid;

#[cfg(target_os = "macos")]
fn binary(name: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|path| path.join(name).is_file()))
}

/// Query macOS TCC via the public preflight APIs. Binary presence alone must
/// not be reported as Granted — Screen Recording / Accessibility consent is
/// independent of `screencapture` / `osascript` being on PATH.
#[cfg(target_os = "macos")]
fn macos_tcc_states() -> (OsPermissionState, OsPermissionState) {
    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGPreflightScreenCaptureAccess() -> bool;
    }
    #[link(name = "ApplicationServices", kind = "framework")]
    unsafe extern "C" {
        fn AXIsProcessTrusted() -> bool;
    }
    let screen = if !binary("screencapture") {
        OsPermissionState::Missing
    } else if unsafe { CGPreflightScreenCaptureAccess() } {
        OsPermissionState::Granted
    } else {
        OsPermissionState::Denied
    };
    let input = if !binary("osascript") {
        OsPermissionState::Missing
    } else if unsafe { AXIsProcessTrusted() } {
        OsPermissionState::Granted
    } else {
        OsPermissionState::Denied
    };
    (screen, input)
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
    let (screen_capture, input_injection) = macos_tcc_states();
    let detail = match (&screen_capture, &input_injection) {
        (OsPermissionState::Granted, OsPermissionState::Granted) => {
            "Brazier has Screen Recording and Accessibility access for desktop computer use."
        }
        _ => {
            "Brazier needs macOS Screen Recording for capture and Accessibility (System Events) for input. Use Request access, then approve Brazier in System Settings if prompted."
        }
    };
    OsPermissionStatus {
        platform: "macos".into(),
        display_server: "quartz".into(),
        screen_capture,
        input_injection,
        detail: Some(detail.into()),
        settings_hint: Some("Enable Brazier in System Settings → Privacy & Security → Screen Recording and Accessibility.".into()),
    }
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
        #[link(name = "CoreGraphics", kind = "framework")]
        unsafe extern "C" {
            fn CGRequestScreenCaptureAccess() -> bool;
        }
        #[link(name = "ApplicationServices", kind = "framework")]
        unsafe extern "C" {
            fn AXIsProcessTrustedWithOptions(options: *const std::ffi::c_void) -> bool;
        }
        // Prompt for Screen Recording / Accessibility when missing, then probe.
        unsafe {
            let _ = CGRequestScreenCaptureAccess();
            let _ = AXIsProcessTrustedWithOptions(std::ptr::null());
        }
        let _ = screenshot().await;
        let _ = command(
            "osascript",
            &[
                "-e".into(),
                "tell application \"System Events\" to get name of first process".into(),
            ],
        )
        .await;
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
        // Retina screenshots contain physical pixels, while System Events
        // addresses the same display in logical points. `screencapture` puts
        // the display DPI in pHYs, so retain it with the viewport instead of
        // assuming that every Mac display is 1x.
        device_pixel_ratio: Some(png_device_pixel_ratio(bytes).unwrap_or(1.0)),
    })
}

/// Read the PNG pixels-per-metre metadata written by macOS `screencapture`.
/// The PNG header itself only reports the physical framebuffer size; pHYs is
/// what lets us convert that size back to the logical point space used by
/// System Events. A missing or nonsensical chunk is deliberately treated as
/// 1x so other screenshot providers keep their existing behavior.
fn png_device_pixel_ratio(bytes: &[u8]) -> Option<f32> {
    if bytes.len() < 8 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    let mut offset = 8_usize;
    while offset.checked_add(12)? <= bytes.len() {
        let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().ok()?) as usize;
        let data_start = offset.checked_add(8)?;
        let data_end = data_start.checked_add(length)?;
        let chunk_end = data_end.checked_add(4)?;
        if chunk_end > bytes.len() {
            return None;
        }
        if &bytes[offset + 4..offset + 8] == b"pHYs" && length >= 9 && bytes[data_start + 8] == 1 {
            let x_pixels_per_metre =
                u32::from_be_bytes(bytes[data_start..data_start + 4].try_into().ok()?);
            let y_pixels_per_metre =
                u32::from_be_bytes(bytes[data_start + 4..data_start + 8].try_into().ok()?);
            if x_pixels_per_metre == 0 || y_pixels_per_metre == 0 {
                return None;
            }
            // PNG stores pixels per metre; macOS logical points are 1/72 inch.
            let dpi =
                (f64::from(x_pixels_per_metre) + f64::from(y_pixels_per_metre)) * 0.0254 / 2.0;
            let ratio = dpi / 72.0;
            return (ratio.is_finite() && (0.5..=4.0).contains(&ratio)).then_some(ratio as f32);
        }
        offset = chunk_end;
    }
    None
}

async fn screenshot() -> Result<ComputerActionResult, String> {
    #[cfg(target_os = "macos")]
    let bytes = mac_screenshot_bytes().await?;
    #[cfg(all(target_os = "linux"))]
    let bytes = if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        crate::computer_portal::screenshot().await?
    } else {
        tokio::task::spawn_blocking(crate::computer_x11::screenshot)
            .await
            .map_err(|error| error.to_string())??
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
async fn mac_screenshot_bytes() -> Result<Vec<u8>, String> {
    // macOS `screencapture` writes to a file path; unlike the Linux capture
    // tools, a `-` argument is not stdout. Keep the file private and unique so
    // concurrent sessions cannot collide, then remove it after reading it.
    let path = std::env::temp_dir().join(format!(
        "brazier-computer-use-{}.png",
        Uuid::new_v4().simple()
    ));
    let path_arg = path.to_string_lossy().into_owned();
    let capture = command(
        "screencapture",
        &["-x".into(), "-t".into(), "png".into(), path_arg],
    )
    .await;
    let bytes = match capture {
        Ok(_) => tokio::fs::read(&path)
            .await
            .map_err(|error| format!("read macOS screenshot: {error}")),
        Err(error) => Err(error),
    };
    let _ = tokio::fs::remove_file(&path).await;
    let bytes = bytes?;
    if bytes.len() < 8 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" {
        return Err(format!(
            "macOS screencapture did not produce a PNG ({} bytes)",
            bytes.len()
        ));
    }
    Ok(bytes)
}

#[cfg(target_os = "macos")]
fn mac_logical_point(viewport: &ComputerViewport, x: f64, y: f64) -> (f64, f64) {
    let device_pixel_ratio = viewport
        .device_pixel_ratio
        .filter(|ratio| ratio.is_finite() && *ratio > 0.0)
        .map(f64::from)
        .unwrap_or(1.0);
    (x / device_pixel_ratio, y / device_pixel_ratio)
}

#[cfg(target_os = "macos")]
async fn mac_action(
    action: &ComputerAction,
    viewport: &ComputerViewport,
    cancel: Option<&crate::computer_browser::ActionCancel>,
) -> Result<(), String> {
    let script = match action {
        ComputerAction::LeftClick { x, y } => {
            let (x, y) = mac_logical_point(viewport, *x, *y);
            format!("tell application \"System Events\" to click at {{{x}, {y}}}")
        }
        ComputerAction::RightClick { x, y } => {
            let (x, y) = mac_logical_point(viewport, *x, *y);
            format!(
                "tell application \"System Events\" to key down control\nclick at {{{x}, {y}}}\nkey up control"
            )
        }
        ComputerAction::DoubleClick { x, y } => {
            let (x, y) = mac_logical_point(viewport, *x, *y);
            format!("tell application \"System Events\" to click at {{{x}, {y}}} twice")
        }
        ComputerAction::TripleClick { x, y } => {
            let (x, y) = mac_logical_point(viewport, *x, *y);
            format!("tell application \"System Events\" to click at {{{x}, {y}}} three times")
        }
        ComputerAction::MouseMove { x, y } => {
            let (x, y) = mac_logical_point(viewport, *x, *y);
            return mac_mouse_move(x, y).await;
        }
        ComputerAction::LeftClickDrag {
            start_x,
            start_y,
            end_x,
            end_y,
        } => {
            let (start_x, start_y) = mac_logical_point(viewport, *start_x, *start_y);
            let (end_x, end_y) = mac_logical_point(viewport, *end_x, *end_y);
            return mac_mouse_drag(start_x, start_y, end_x, end_y).await;
        }
        ComputerAction::Keypress { keys } => format!(
            "tell application \"System Events\" to key code {}",
            mac_key(keys)?
        ),
        ComputerAction::Scroll { delta_y, .. } => format!(
            "tell application \"System Events\" to scroll {}",
            -delta_y.round() as i64
        ),
        ComputerAction::Type { text } => {
            // Type via System Events keystroke — never pbcopy/Cmd+V, which
            // overwrites the user's clipboard as a side effect of agent input.
            return mac_type_text(text, cancel).await;
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

/// Type text through System Events without touching the pasteboard.
#[cfg(target_os = "macos")]
async fn mac_type_text(
    text: &str,
    cancel: Option<&crate::computer_browser::ActionCancel>,
) -> Result<(), String> {
    // AppleScript string literals cannot contain unescaped backslash or quote.
    // Chunk so a single huge keystroke call cannot stall the Esc hatch forever.
    for chunk in text
        .chars()
        .collect::<Vec<_>>()
        .chunks(32)
        .map(|chars| chars.iter().collect::<String>())
    {
        if cancel.is_some_and(|c| c.is_cancelled()) {
            return Err("computer action cancelled".into());
        }
        let escaped = chunk.replace('\\', "\\\\").replace('"', "\\\"");
        command(
            "osascript",
            &[
                "-e".into(),
                format!("tell application \"System Events\" to keystroke \"{escaped}\""),
            ],
        )
        .await
        .map(|_| ())?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
async fn mac_mouse_move(x: f64, y: f64) -> Result<(), String> {
    mac_post_mouse_events(format!("post($.kCGEventMouseMoved, {x}, {y});")).await
}

#[cfg(target_os = "macos")]
async fn mac_mouse_drag(start_x: f64, start_y: f64, end_x: f64, end_y: f64) -> Result<(), String> {
    mac_post_mouse_events(format!(
        "post($.kCGEventMouseMoved, {start_x}, {start_y});\n\
         post($.kCGEventLeftMouseDown, {start_x}, {start_y});\n\
         post($.kCGEventLeftMouseDragged, {end_x}, {end_y});\n\
         post($.kCGEventLeftMouseUp, {end_x}, {end_y});"
    ))
    .await
}

/// Post pointer events through CoreGraphics. System Events exposes `click at`
/// for process targets, but it has no `mouse` variable or `move mouse` command;
/// JXA is a system-provided bridge to the same Accessibility-authorized event
/// API without requiring a third-party `cliclick` installation.
#[cfg(target_os = "macos")]
async fn mac_post_mouse_events(body: String) -> Result<(), String> {
    let script = format!(
        "ObjC.import('CoreGraphics');\n\
         function post(type, x, y) {{\n\
             var event = $.CGEventCreateMouseEvent(null, type, $.CGPointMake(x, y), $.kCGMouseButtonLeft);\n\
             if (event === null) throw new Error('CoreGraphics could not create mouse event');\n\
             $.CGEventPost($.kCGHIDEventTap, event);\n\
         }}\n\
         {body}"
    );
    command(
        "osascript",
        &["-l".into(), "JavaScript".into(), "-e".into(), script],
    )
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
async fn linux_action(
    action: &ComputerAction,
    cancel: Option<&crate::computer_browser::ActionCancel>,
) -> Result<(), String> {
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        return wayland_action(action, cancel).await;
    }
    tokio::task::spawn_blocking({
        let action = action.clone();
        move || match action {
            ComputerAction::LeftClick { x, y } => crate::computer_x11::click(x, y, 1, 1),
            ComputerAction::RightClick { x, y } => crate::computer_x11::click(x, y, 3, 1),
            ComputerAction::DoubleClick { x, y } => crate::computer_x11::click(x, y, 1, 2),
            ComputerAction::TripleClick { x, y } => crate::computer_x11::click(x, y, 1, 3),
            ComputerAction::MouseMove { x, y } => crate::computer_x11::move_to(x, y),
            ComputerAction::LeftClickDrag {
                start_x,
                start_y,
                end_x,
                end_y,
            } => crate::computer_x11::drag(start_x, start_y, end_x, end_y),
            ComputerAction::Type { text } => crate::computer_x11::type_text(&text),
            ComputerAction::Keypress { keys } => {
                for key in keys {
                    crate::computer_x11::key(wayland_keysym(&key)?)?;
                }
                Ok(())
            }
            ComputerAction::Scroll { delta_y, .. } => crate::computer_x11::scroll(delta_y),
            _ => Err("This action is not available on the desktop.".into()),
        }
    })
    .await
    .map_err(|error| error.to_string())?
}
#[cfg(target_os = "linux")]
async fn wayland_action(
    action: &ComputerAction,
    cancel: Option<&crate::computer_browser::ActionCancel>,
) -> Result<(), String> {
    if let ComputerAction::Wait { milliseconds } = action {
        return cancellable_desktop_sleep(
            std::time::Duration::from_millis(*milliseconds),
            cancel,
        )
        .await;
    }
    match action {
        ComputerAction::MouseMove { x, y } => crate::computer_portal::pointer_motion(*x, *y).await,
        ComputerAction::LeftClick { x, y } => {
            crate::computer_portal::pointer_button(*x, *y, 272, 1).await
        }
        ComputerAction::RightClick { x, y } => {
            crate::computer_portal::pointer_button(*x, *y, 273, 1).await
        }
        ComputerAction::DoubleClick { x, y } => {
            crate::computer_portal::pointer_button(*x, *y, 272, 2).await
        }
        ComputerAction::TripleClick { x, y } => {
            crate::computer_portal::pointer_button(*x, *y, 272, 3).await
        }
        ComputerAction::LeftClickDrag {
            start_x,
            start_y,
            end_x,
            end_y,
        } => crate::computer_portal::pointer_drag(*start_x, *start_y, *end_x, *end_y).await,
        ComputerAction::Scroll {
            delta_x, delta_y, ..
        } => crate::computer_portal::scroll(*delta_x, *delta_y).await,
        ComputerAction::Type { text } => crate::computer_portal::type_text(text, cancel).await,
        ComputerAction::Keypress { keys } => {
            for key in keys {
                if cancel.is_some_and(|c| c.is_cancelled()) {
                    return Err("computer action cancelled".into());
                }
                crate::computer_portal::key(wayland_keysym(key)?).await?;
            }
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

pub async fn execute_desktop_action(
    action: &ComputerAction,
    viewport: &ComputerViewport,
    settle_delay_ms: u64,
    cancel: Option<&crate::computer_browser::ActionCancel>,
) -> ComputerActionResult {
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
    if let ComputerAction::Wait { milliseconds } = action {
        if let Err(error) = cancellable_desktop_sleep(
            std::time::Duration::from_millis(*milliseconds),
            cancel,
        )
        .await
        {
            return result(ComputerActionStatus::Error, Some(error));
        }
        return screenshot().await.unwrap_or_else(|e| {
            result(
                ComputerActionStatus::Error,
                Some(format!("Wait succeeded but screenshot failed: {e}")),
            )
        });
    }
    #[cfg(target_os = "macos")]
    let execution = mac_action(action, viewport, cancel).await;
    #[cfg(target_os = "linux")]
    let execution = {
        let _ = viewport;
        linux_action(action, cancel).await
    };
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let execution: Result<(), String> = Err("unsupported platform".into());
    match execution {
        Ok(()) => {
            if settle_delay_ms > 0
                && let Err(error) = cancellable_desktop_sleep(
                    std::time::Duration::from_millis(settle_delay_ms),
                    cancel,
                )
                .await
            {
                return result(ComputerActionStatus::Error, Some(error));
            }
            screenshot().await.unwrap_or_else(|e| {
                result(
                    ComputerActionStatus::Error,
                    Some(format!("Action succeeded but screenshot failed: {e}")),
                )
            })
        }
        Err(e) => result(
            ComputerActionStatus::Error,
            Some(format!("Desktop {} failed: {e}", action.kind())),
        ),
    }
}

async fn cancellable_desktop_sleep(
    duration: std::time::Duration,
    cancel: Option<&crate::computer_browser::ActionCancel>,
) -> Result<(), String> {
    let Some(cancel) = cancel else {
        tokio::time::sleep(duration).await;
        return Ok(());
    };
    tokio::select! {
        _ = tokio::time::sleep(duration) => Ok(()),
        _ = cancel.cancelled() => Err("computer action cancelled".into()),
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

    #[test]
    fn reads_retina_scale_from_png_phys_metadata() {
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&1920_u32.to_be_bytes());
        ihdr.extend_from_slice(&1080_u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        append_png_chunk(&mut png, b"IHDR", &ihdr);
        let mut phys = Vec::new();
        phys.extend_from_slice(&5669_u32.to_be_bytes());
        phys.extend_from_slice(&5669_u32.to_be_bytes());
        phys.push(1);
        append_png_chunk(&mut png, b"pHYs", &phys);

        let viewport = png_viewport(&png).expect("viewport");
        assert!((viewport.device_pixel_ratio.unwrap() - 2.0).abs() < 0.001);
    }

    fn append_png_chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        png.extend_from_slice(&(data.len() as u32).to_be_bytes());
        png.extend_from_slice(kind);
        png.extend_from_slice(data);
        png.extend_from_slice(&[0; 4]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn converts_retina_pixels_to_system_events_points() {
        let viewport = ComputerViewport {
            width: 2880,
            height: 1800,
            device_pixel_ratio: Some(2.0),
        };
        assert_eq!(mac_logical_point(&viewport, 1440.0, 900.0), (720.0, 450.0));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn maps_common_portal_keysyms() {
        assert_eq!(wayland_keysym("Escape").unwrap(), 0xff1b);
        assert_eq!(wayland_keysym("ArrowLeft").unwrap(), 0xff51);
    }
}

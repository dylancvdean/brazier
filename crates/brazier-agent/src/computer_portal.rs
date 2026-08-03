//! Native XDG desktop-portal client for Wayland Computer Use.
//!
//! It talks D-Bus directly: no `grim`, `ydotool`, `gdbus`, or privileged helper.

#![cfg(target_os = "linux")]

use std::{collections::HashMap, sync::OnceLock};

use futures::StreamExt;
use tokio::sync::Mutex;
use zbus::{
    Connection, Proxy,
    zvariant::{OwnedObjectPath, OwnedValue, Str},
};

const DESKTOP: &str = "org.freedesktop.portal.Desktop";
const ROOT: &str = "/org/freedesktop/portal/desktop";

struct PortalSession {
    connection: Connection,
    path: OwnedObjectPath,
    stream: u32,
}
static SESSION: OnceLock<Mutex<Option<PortalSession>>> = OnceLock::new();
fn session_slot() -> &'static Mutex<Option<PortalSession>> {
    SESSION.get_or_init(|| Mutex::new(None))
}

fn restore_token_path() -> Option<std::path::PathBuf> {
    let state_home = std::env::var_os("XDG_STATE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".local/state"))
        })?;
    Some(
        state_home
            .join("brazier")
            .join("wayland-remote-desktop-token"),
    )
}

fn load_restore_token() -> Option<String> {
    restore_token_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|token| token.trim().to_owned())
        .filter(|token| !token.is_empty())
}

fn save_restore_token(token: &str) {
    let Some(path) = restore_token_path() else {
        return;
    };
    let Some(parent) = path.parent() else { return };
    if std::fs::create_dir_all(parent).is_ok() {
        let _ = std::fs::write(path, token);
    }
}

/// Drop the persisted RemoteDesktop restore token so the next session must
/// obtain fresh compositor consent. Called from Esc / stop paths.
pub fn clear_restore_token() {
    if let Some(path) = restore_token_path() {
        let _ = std::fs::remove_file(path);
    }
}

async fn proxy<'a>(
    connection: &'a Connection,
    path: &'a str,
    interface: &'a str,
) -> Result<Proxy<'a>, String> {
    Proxy::new(connection, DESKTOP, path, interface)
        .await
        .map_err(|e| e.to_string())
}

async fn portal_connection() -> Result<Connection, String> {
    let connection = Connection::session()
        .await
        .map_err(|e| format!("connect to session D-Bus: {e}"))?;
    let registry = proxy(&connection, ROOT, "org.freedesktop.host.portal.Registry")
        .await
        .map_err(|error| format!("register Brazier portal identity: {error}"))?;
    let app_id = std::env::var("BRAZIER_PORTAL_APP_ID").unwrap_or_else(|_| "brazier".to_owned());
    registry
        .call::<_, _, ()>("Register", &(app_id, HashMap::<String, OwnedValue>::new()))
        .await
        .map_err(|error| format!("register Brazier portal identity: {error}"))?;
    Ok(connection)
}

async fn request_response(
    connection: &Connection,
    request: OwnedObjectPath,
) -> Result<HashMap<String, OwnedValue>, String> {
    let request = proxy(
        connection,
        request.as_str(),
        "org.freedesktop.portal.Request",
    )
    .await?;
    let mut responses = request
        .receive_signal("Response")
        .await
        .map_err(|e| e.to_string())?;
    let message = tokio::time::timeout(std::time::Duration::from_secs(20), responses.next())
        .await
        .map_err(|_| "Timed out waiting for the desktop portal. No consent dialog was presented; check that a ScreenCast/RemoteDesktop portal backend and PipeWire are running.".to_owned())?
        .ok_or_else(|| "desktop portal closed the request without a response".to_owned())?;
    let (code, results): (u32, HashMap<String, OwnedValue>) =
        message.body().deserialize().map_err(|e| e.to_string())?;
    match code {
        0 => Ok(results),
        1 => Err("Desktop portal request was cancelled by the user.".into()),
        _ => Err(format!(
            "Desktop portal request failed (response code {code})."
        )),
    }
}

async fn response_from(
    mut responses: zbus::proxy::SignalStream<'_>,
) -> Result<HashMap<String, OwnedValue>, String> {
    let message = tokio::time::timeout(std::time::Duration::from_secs(20), responses.next()).await
        .map_err(|_| "Timed out waiting for the desktop portal. No consent dialog was presented; check that a ScreenCast/RemoteDesktop portal backend and PipeWire are running.".to_owned())?
        .ok_or_else(|| "desktop portal closed the request without a response".to_owned())?;
    let (code, results): (u32, HashMap<String, OwnedValue>) =
        message.body().deserialize().map_err(|e| e.to_string())?;
    match code {
        0 => Ok(results),
        1 => Err("Desktop portal request was cancelled by the user.".into()),
        _ => Err(format!(
            "Desktop portal request failed (response code {code})."
        )),
    }
}

fn expected_request_path(connection: &Connection, token: &str) -> Result<String, String> {
    let sender = connection
        .unique_name()
        .ok_or("session D-Bus did not assign Brazier a unique name")?
        .as_str()
        .trim_start_matches(':')
        .replace('.', "_");
    Ok(format!("{ROOT}/request/{sender}/{token}"))
}

fn token(prefix: &str) -> String {
    // Portal object paths are derived from these tokens. Older portal releases
    // assert rather than return an error when a RemoteDesktop session lacks one.
    format!("{prefix}{}", uuid::Uuid::new_v4().simple())
}
fn string_value(value: &str) -> OwnedValue {
    OwnedValue::from(Str::from(value))
}
fn options() -> HashMap<String, OwnedValue> {
    let mut options = HashMap::new();
    let handle = token("brazier");
    options.insert("handle_token".into(), string_value(&handle));
    options
}
fn request_options() -> (HashMap<String, OwnedValue>, String) {
    let token = token("brazier");
    let mut options = HashMap::new();
    options.insert("handle_token".into(), string_value(&token));
    (options, token)
}
async fn response_listener<'a>(
    connection: &'a Connection,
    token: &str,
) -> Result<zbus::proxy::SignalStream<'a>, String> {
    let path = expected_request_path(connection, token)?;
    proxy(connection, &path, "org.freedesktop.portal.Request")
        .await?
        .receive_signal("Response")
        .await
        .map_err(|e| e.to_string())
}
fn value<T: TryFrom<OwnedValue>>(
    results: &mut HashMap<String, OwnedValue>,
    key: &str,
) -> Result<T, String> {
    results
        .remove(key)
        .ok_or_else(|| format!("desktop portal response omitted {key}"))?
        .try_into()
        .map_err(|_| format!("desktop portal response has an invalid {key}"))
}

/// Ask the compositor for a still image through the Screenshot portal.
pub async fn screenshot() -> Result<Vec<u8>, String> {
    let connection = portal_connection().await?;
    let screenshot = proxy(&connection, ROOT, "org.freedesktop.portal.Screenshot").await?;
    let request: OwnedObjectPath = screenshot
        .call("Screenshot", &("", options()))
        .await
        .map_err(|e| e.to_string())?;
    let mut results = request_response(&connection, request).await?;
    let uri: String = value(&mut results, "uri")?;
    let path = uri
        .strip_prefix("file://")
        .ok_or("desktop portal returned a non-file screenshot URI")?;
    tokio::fs::read(path)
        .await
        .map_err(|e| format!("read portal screenshot: {e}"))
}

async fn create_session_attempt(restore_token: Option<&str>) -> Result<PortalSession, String> {
    let connection = portal_connection().await?;
    let remote = proxy(&connection, ROOT, "org.freedesktop.portal.RemoteDesktop").await?;
    let session_token = token("brazier_session");
    let create_token = token("brazier");
    let mut create_options: HashMap<String, OwnedValue> = HashMap::new();
    create_options.insert("handle_token".into(), string_value(&create_token));
    create_options.insert("session_handle_token".into(), string_value(&session_token));
    // The portal API explicitly requires subscribing before the method call:
    // KDE may emit Response before the call reply is processed.
    let expected = expected_request_path(&connection, &create_token)?;
    let request_proxy = proxy(&connection, &expected, "org.freedesktop.portal.Request").await?;
    let responses = request_proxy
        .receive_signal("Response")
        .await
        .map_err(|e| e.to_string())?;
    let request: OwnedObjectPath = remote
        .call("CreateSession", &create_options)
        .await
        .map_err(|e| e.to_string())?;
    if request.as_str() != expected {
        return Err(format!(
            "desktop portal returned unexpected request handle {}",
            request.as_str()
        ));
    }
    let mut results = response_from(responses)
        .await
        .map_err(|error| format!("CreateSession: {error}"))?;
    // The portal specification preserves this result as a string for backward
    // compatibility, even though its content is an object path.
    let path: String = value(&mut results, "session_handle")?;
    let path = OwnedObjectPath::try_from(path)
        .map_err(|error| format!("invalid portal session handle: {error}"))?;

    // Sharing ScreenCast and RemoteDesktop gives absolute pointer coordinates
    // the same monitor mapping as the screenshot the model observed.
    let screencast = proxy(&connection, ROOT, "org.freedesktop.portal.ScreenCast").await?;
    let (mut source_options, source_token) = request_options();
    source_options.insert("types".into(), OwnedValue::from(1_u32));
    source_options.insert("multiple".into(), OwnedValue::from(false));
    let source_response = response_listener(&connection, &source_token).await?;
    let request: OwnedObjectPath = screencast
        .call("SelectSources", &(path.clone(), source_options))
        .await
        .map_err(|e| e.to_string())?;
    if request.as_str() != expected_request_path(&connection, &source_token)? {
        return Err("desktop portal returned an unexpected SelectSources handle".into());
    }
    response_from(source_response)
        .await
        .map_err(|error| format!("SelectSources: {error}"))?;

    let (mut device_options, device_token) = request_options();
    device_options.insert("types".into(), OwnedValue::from(3_u32));
    device_options.insert("persist_mode".into(), OwnedValue::from(2_u32));
    if let Some(restore_token) = restore_token {
        device_options.insert("restore_token".into(), string_value(&restore_token));
    }
    let device_response = response_listener(&connection, &device_token).await?;
    let request: OwnedObjectPath = remote
        .call("SelectDevices", &(path.clone(), device_options))
        .await
        .map_err(|e| e.to_string())?;
    if request.as_str() != expected_request_path(&connection, &device_token)? {
        return Err("desktop portal returned an unexpected SelectDevices handle".into());
    }
    response_from(device_response)
        .await
        .map_err(|error| format!("SelectDevices: {error}"))?;
    let (start_options, start_token) = request_options();
    let start_response = response_listener(&connection, &start_token).await?;
    let request: OwnedObjectPath = remote
        .call("Start", &(path.clone(), "", start_options))
        .await
        .map_err(|e| e.to_string())?;
    if request.as_str() != expected_request_path(&connection, &start_token)? {
        return Err("desktop portal returned an unexpected Start handle".into());
    }
    let mut results = response_from(start_response)
        .await
        .map_err(|error| format!("Start: {error}"))?;
    if let Ok(restore_token) = value::<String>(&mut results, "restore_token") {
        save_restore_token(&restore_token);
    }
    let streams: Vec<(u32, HashMap<String, OwnedValue>)> = value(&mut results, "streams")?;
    let stream = streams
        .first()
        .ok_or("no monitor was selected in the desktop portal")?
        .0;
    Ok(PortalSession {
        connection,
        path,
        stream,
    })
}

async fn create_session() -> Result<PortalSession, String> {
    let restore_token = load_restore_token();
    let first = create_session_attempt(restore_token.as_deref()).await;
    let Err(restored_error) = first else {
        return first;
    };

    // KDE can retain a restore token after the corresponding permission-store
    // entry has disappeared. Some portal versions then leave SelectDevices or
    // Start unanswered instead of rejecting the stale token. Retry once without
    // restoration so the compositor can present a fresh consent dialog. The
    // existing token is deliberately kept until a successful Start replaces it.
    if restore_token.is_some()
        && (restored_error.starts_with("SelectDevices:") || restored_error.starts_with("Start:"))
    {
        return create_session_attempt(None).await.map_err(|fresh_error| {
            format!(
                "Restoring the saved Wayland permission failed ({restored_error}). Retrying with a fresh consent request also failed ({fresh_error})"
            )
        });
    }

    Err(restored_error)
}

async fn with_session<T>(operation: impl FnOnce(&PortalSession) -> T) -> Result<T, String> {
    let mut guard = session_slot().lock().await;
    if guard.is_none() {
        *guard = Some(create_session().await?);
    }
    Ok(operation(guard.as_ref().expect("session inserted")))
}

/// Begin the compositor's combined screen-share and remote-desktop consent
/// flow. The session stays alive after approval, so the first computer action
/// does not need to surprise the user with a second prompt.
pub async fn request_permissions() -> Result<(), String> {
    with_session(|_| ()).await
}

async fn remote_call<T: serde::Serialize + zbus::zvariant::DynamicType>(
    method: &str,
    body: &T,
) -> Result<(), String> {
    let (connection, _) = with_session(|s| (s.connection.clone(), s.path.clone())).await?;
    let remote = proxy(&connection, ROOT, "org.freedesktop.portal.RemoteDesktop").await?;
    remote
        .call::<_, _, ()>(method, body)
        .await
        .map_err(|e| e.to_string())
}

pub async fn pointer_motion(x: f64, y: f64) -> Result<(), String> {
    let (path, stream) = with_session(|s| (s.path.clone(), s.stream)).await?;
    remote_call(
        "NotifyPointerMotionAbsolute",
        &(path, options(), stream, x, y),
    )
    .await
}
pub async fn pointer_button(x: f64, y: f64, button: u32, clicks: usize) -> Result<(), String> {
    pointer_motion(x, y).await?;
    let path = with_session(|s| s.path.clone()).await?;
    let button = i32::try_from(button).map_err(|_| "invalid pointer button")?;
    for _ in 0..clicks {
        remote_call(
            "NotifyPointerButton",
            &(path.clone(), options(), button, 1_u32),
        )
        .await?;
        remote_call(
            "NotifyPointerButton",
            &(path.clone(), options(), button, 0_u32),
        )
        .await?;
    }
    Ok(())
}
pub async fn pointer_drag(
    start_x: f64,
    start_y: f64,
    end_x: f64,
    end_y: f64,
) -> Result<(), String> {
    pointer_motion(start_x, start_y).await?;
    let path = with_session(|s| s.path.clone()).await?;
    remote_call(
        "NotifyPointerButton",
        &(path.clone(), options(), 272_i32, 1_u32),
    )
    .await?;
    pointer_motion(end_x, end_y).await?;
    remote_call("NotifyPointerButton", &(path, options(), 272_i32, 0_u32)).await
}
pub async fn scroll(delta_x: f64, delta_y: f64) -> Result<(), String> {
    let path = with_session(|s| s.path.clone()).await?;
    if delta_y != 0.0 {
        remote_call(
            "NotifyPointerAxis",
            &(path.clone(), options(), 0_u32, delta_y),
        )
        .await?;
    }
    if delta_x != 0.0 {
        remote_call("NotifyPointerAxis", &(path, options(), 1_u32, delta_x)).await?;
    }
    Ok(())
}
pub async fn key(keysym: u32) -> Result<(), String> {
    let path = with_session(|s| s.path.clone()).await?;
    let keysym = i32::try_from(keysym).map_err(|_| "invalid keyboard keysym")?;
    remote_call(
        "NotifyKeyboardKeysym",
        &(path.clone(), options(), keysym, 1_u32),
    )
    .await?;
    remote_call("NotifyKeyboardKeysym", &(path, options(), keysym, 0_u32)).await
}
pub async fn type_text(
    text: &str,
    cancel: Option<&crate::computer_browser::ActionCancel>,
) -> Result<(), String> {
    for character in text.chars() {
        if cancel.is_some_and(|c| c.is_cancelled()) {
            return Err("computer action cancelled".into());
        }
        key(if character.is_ascii() {
            character as u32
        } else {
            0x0100_0000 | character as u32
        })
        .await?;
    }
    Ok(())
}
pub async fn close_session() {
    let Some(session) = session_slot().lock().await.take() else {
        return;
    };
    if let Ok(portal_session) = proxy(
        &session.connection,
        session.path.as_str(),
        "org.freedesktop.portal.Session",
    )
    .await
    {
        let _: Result<(), _> = portal_session.call("Close", &()).await;
    }
    // Esc/stop closes the live session; keep the restore token until an
    // explicit clear so Settings "Request access" can reconnect without a
    // second prompt. Emergency stop paths call [`clear_restore_token`].
}

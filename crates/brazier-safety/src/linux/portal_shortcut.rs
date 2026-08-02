use std::{collections::HashMap, time::Duration};

use anyhow::{Context as _, Result, bail};
use futures::StreamExt as _;
use zbus::{
    Connection, Proxy,
    zvariant::{OwnedObjectPath, OwnedValue, Str},
};

const DESKTOP: &str = "org.freedesktop.portal.Desktop";
const ROOT: &str = "/org/freedesktop/portal/desktop";
const INTERFACE: &str = "org.freedesktop.portal.GlobalShortcuts";
const SHORTCUT_ID: &str = "brazier_computer_use_ctrl_shift_escape_v4";
type Shortcuts = Vec<(String, HashMap<String, OwnedValue>)>;

pub struct EscapeShortcut {
    connection: Connection,
    session: OwnedObjectPath,
    guard_kde_component: bool,
}

impl EscapeShortcut {
    pub async fn open() -> Result<Self> {
        let connection = Connection::session()
            .await
            .context("connect to the desktop session bus")?;
        register_host_app(&connection).await?;
        let shortcuts = proxy(&connection, ROOT, INTERFACE).await?;
        let version: u32 = shortcuts
            .get_property("version")
            .await
            .context("read GlobalShortcuts portal version")?;
        if version < 1 {
            bail!("the desktop portal does not provide GlobalShortcuts")
        }

        let create_token = token("brazier_safety");
        let session_token = token("brazier_safety_session");
        let mut create_options = HashMap::new();
        create_options.insert("handle_token".to_owned(), string_value(&create_token));
        create_options.insert(
            "session_handle_token".to_owned(),
            string_value(&session_token),
        );
        let create_response = response_listener(&connection, &create_token).await?;
        let request: OwnedObjectPath = shortcuts
            .call("CreateSession", &create_options)
            .await
            .context("create GlobalShortcuts portal session")?;
        verify_request(&connection, &create_token, &request)?;
        let mut results = response_from(create_response).await?;
        let session_text: String = value(&mut results, "session_handle")?;
        let session = OwnedObjectPath::try_from(session_text)
            .context("validate GlobalShortcuts session handle")?;

        // A persisted shortcut reported by ListShortcuts is not active in this
        // newly-created session. The portal contract requires every session to
        // call BindShortcuts once; the backend reuses the user's saved binding.
        let mut effective = bind_escape(&connection, &shortcuts, &session).await?;
        if !has_effective_escape(&effective) {
            effective = configure_shortcuts(&shortcuts, &session).await?;
        }
        if !has_effective_escape(&effective) {
            bail!(
                "the compositor did not assign Ctrl+Shift+Esc to Brazier's emergency stop shortcut"
            )
        }

        let guard_kde_component = verify_kde_active(&connection).await?;

        Ok(Self {
            connection,
            session,
            guard_kde_component,
        })
    }

    pub async fn wait(self) -> Result<()> {
        let shortcuts = proxy(&self.connection, ROOT, INTERFACE).await?;
        let mut activated = shortcuts
            .receive_signal("Activated")
            .await
            .context("subscribe to the global Escape shortcut")?;
        if self.guard_kde_component {
            let component = kde_component(&self.connection).await?;
            let mut watchdog = tokio::time::interval(Duration::from_secs(1));
            loop {
                tokio::select! {
                    message = activated.next() => {
                        if portal_activation(message, &self.session)? { return Ok(()); }
                    }
                    _ = watchdog.tick() => {
                        let active: bool = component
                            .call("isActive", &())
                            .await
                            .context("verify KDE's emergency shortcut remains active")?;
                        if !active {
                            bail!("KDE stopped Brazier's emergency shortcut; computer use was stopped");
                        }
                    }
                }
            }
        } else {
            while let Some(message) = activated.next().await {
                if portal_activation(Some(message), &self.session)? {
                    return Ok(());
                }
            }
        }
        bail!("the GlobalShortcuts portal stopped watching the emergency shortcut")
    }
}

async fn verify_kde_active(connection: &Connection) -> Result<bool> {
    if !std::env::var("XDG_CURRENT_DESKTOP")
        .unwrap_or_default()
        .split(':')
        .any(|desktop| desktop.eq_ignore_ascii_case("kde"))
    {
        return Ok(false);
    }

    let component = kde_component(connection).await?;
    for _ in 0..10 {
        let active: bool = component
            .call("isActive", &())
            .await
            .context("verify KDE's emergency shortcut is active")?;
        if active {
            return Ok(true);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    bail!(
        "KDE accepted Ctrl+Shift+Esc but left Brazier's shortcut inactive; computer use cannot start safely"
    )
}

async fn kde_component(connection: &Connection) -> Result<Proxy<'_>> {
    Proxy::new(
        connection,
        "org.kde.kglobalaccel",
        "/component/brazier",
        "org.kde.kglobalaccel.Component",
    )
    .await
    .context("open KDE's Brazier shortcut component")
}

fn portal_activation(
    message: Option<zbus::Message>,
    expected_session: &OwnedObjectPath,
) -> Result<bool> {
    let message =
        message.context("GlobalShortcuts portal stopped watching the emergency shortcut")?;
    let (session, shortcut_id, _timestamp, _options): (
        OwnedObjectPath,
        String,
        u64,
        HashMap<String, OwnedValue>,
    ) = message
        .body()
        .deserialize()
        .context("decode GlobalShortcuts activation")?;
    Ok(session == *expected_session && shortcut_id == SHORTCUT_ID)
}

async fn configure_shortcuts(proxy: &Proxy<'_>, session: &OwnedObjectPath) -> Result<Shortcuts> {
    let mut changes = proxy
        .receive_signal("ShortcutsChanged")
        .await
        .context("watch shortcut configuration changes")?;
    proxy
        .call::<_, _, ()>(
            "ConfigureShortcuts",
            &(session.clone(), "", HashMap::<String, OwnedValue>::new()),
        )
        .await
        .context("open the compositor's shortcut configuration")?;
    loop {
        let message = tokio::time::timeout(Duration::from_secs(90), changes.next())
            .await
            .context("timed out waiting for Ctrl+Shift+Esc to be assigned")?
            .context("the compositor closed shortcut configuration")?;
        let (changed_session, shortcuts): (OwnedObjectPath, Shortcuts) = message
            .body()
            .deserialize()
            .context("decode changed global shortcuts")?;
        if changed_session == *session {
            return Ok(shortcuts);
        }
    }
}

async fn bind_escape(
    connection: &Connection,
    proxy: &Proxy<'_>,
    session: &OwnedObjectPath,
) -> Result<Shortcuts> {
    let mut properties = HashMap::new();
    properties.insert(
        "description".to_owned(),
        string_value("Immediately stop Brazier Computer Use"),
    );
    properties.insert(
        "preferred_trigger".to_owned(),
        string_value("CTRL+SHIFT+Escape"),
    );
    let requested = vec![(SHORTCUT_ID.to_owned(), properties)];
    let (options, request_token) = request_options();
    let response = response_listener(connection, &request_token).await?;
    let request: OwnedObjectPath = proxy
        .call("BindShortcuts", &(session.clone(), requested, "", options))
        .await
        .context("request the global Escape shortcut")?;
    verify_request(connection, &request_token, &request)?;
    let mut results = response_from(response).await?;
    let bound: Shortcuts = value(&mut results, "shortcuts")?;
    if !bound.iter().any(|(id, _)| id == SHORTCUT_ID) {
        bail!("Ctrl+Shift+Esc was not granted as Brazier's emergency stop shortcut")
    }
    Ok(bound)
}

fn has_effective_escape(shortcuts: &Shortcuts) -> bool {
    shortcuts.iter().any(|(id, properties)| {
        if id != SHORTCUT_ID {
            return false;
        }
        properties
            .get("trigger_description")
            .and_then(|value| <&str>::try_from(value).ok())
            .is_some_and(|trigger| {
                let normalized = trigger.to_ascii_lowercase().replace(' ', "");
                matches!(
                    normalized.as_str(),
                    "ctrl+shift+esc" | "ctrl+shift+escape" | "shift+ctrl+esc" | "shift+ctrl+escape"
                )
            })
    })
}

async fn register_host_app(connection: &Connection) -> Result<()> {
    let registry = proxy(connection, ROOT, "org.freedesktop.host.portal.Registry")
        .await
        .context("the desktop portal cannot register Brazier's application identity")?;
    let app_id = std::env::var("BRAZIER_PORTAL_APP_ID").unwrap_or_else(|_| "brazier".to_owned());
    registry
        .call::<_, _, ()>("Register", &(app_id, HashMap::<String, OwnedValue>::new()))
        .await
        .context("register Brazier's application identity with the desktop portal")
}

async fn proxy<'a>(
    connection: &'a Connection,
    path: &'a str,
    interface: &'a str,
) -> Result<Proxy<'a>> {
    Proxy::new(connection, DESKTOP, path, interface)
        .await
        .context("create desktop portal proxy")
}

async fn response_listener<'a>(
    connection: &'a Connection,
    request_token: &str,
) -> Result<zbus::proxy::SignalStream<'a>> {
    let path = expected_request_path(connection, request_token)?;
    proxy(connection, &path, "org.freedesktop.portal.Request")
        .await?
        .receive_signal("Response")
        .await
        .context("subscribe to desktop portal response")
}

async fn response_from(
    mut responses: zbus::proxy::SignalStream<'_>,
) -> Result<HashMap<String, OwnedValue>> {
    let message = tokio::time::timeout(Duration::from_secs(30), responses.next())
        .await
        .context("timed out waiting for the desktop portal permission dialog")?
        .context("desktop portal closed the request without a response")?;
    let (code, results): (u32, HashMap<String, OwnedValue>) = message
        .body()
        .deserialize()
        .context("decode desktop portal response")?;
    match code {
        0 => Ok(results),
        1 => bail!("desktop shortcut permission was cancelled by the user"),
        other => bail!("desktop shortcut permission failed (response code {other})"),
    }
}

fn request_options() -> (HashMap<String, OwnedValue>, String) {
    let request_token = token("brazier_safety");
    let mut options = HashMap::new();
    options.insert("handle_token".to_owned(), string_value(&request_token));
    (options, request_token)
}

fn expected_request_path(connection: &Connection, request_token: &str) -> Result<String> {
    let sender = connection
        .unique_name()
        .context("session bus did not assign a unique name")?
        .as_str()
        .trim_start_matches(':')
        .replace('.', "_");
    Ok(format!("{ROOT}/request/{sender}/{request_token}"))
}

fn verify_request(
    connection: &Connection,
    request_token: &str,
    actual: &OwnedObjectPath,
) -> Result<()> {
    let expected = expected_request_path(connection, request_token)?;
    if actual.as_str() != expected {
        bail!("desktop portal returned an unexpected request handle")
    }
    Ok(())
}

fn token(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

fn string_value(value: &str) -> OwnedValue {
    OwnedValue::from(Str::from(value))
}

fn value<T: TryFrom<OwnedValue>>(
    results: &mut HashMap<String, OwnedValue>,
    key: &str,
) -> Result<T> {
    results
        .remove(key)
        .with_context(|| format!("desktop portal response omitted {key}"))?
        .try_into()
        .map_err(|_| anyhow::anyhow!("desktop portal response has an invalid {key}"))
}

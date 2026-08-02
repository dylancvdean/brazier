//! Isolated Chromium/CDP target for Computer Use.
//!
//! Browser actions are deliberately backed by a real Chromium process.  We do
//! not provide a synthetic fallback: reporting an unavailable browser is safer
//! than telling a model that a click or a screenshot happened when it did not.

use std::{collections::HashMap, net::TcpListener, path::PathBuf, process::Stdio, sync::Arc};

use anyhow::{Context, Result, bail};
use brazier_protocol::computer_types::{
    ComputerAction, ComputerActionResult, ComputerActionStatus, ComputerViewport,
};
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::{
    process::{Child, Command},
    sync::Mutex,
    time::{Duration, sleep, timeout},
};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use uuid::Uuid;

const DRIVER_UNAVAILABLE: &str =
    "Browser computer use requires a working Chromium installation; no action was performed.";
type CdpSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

struct CdpBrowserSession {
    id: String,
    viewport: ComputerViewport,
    debug_url: String,
    process: Child,
    profile_dir: PathBuf,
}

impl CdpBrowserSession {
    async fn launch(viewport: ComputerViewport, executable: Option<&str>) -> Result<Self> {
        let executable =
            find_chromium(executable).ok_or_else(|| anyhow::anyhow!(DRIVER_UNAVAILABLE))?;
        let listener = TcpListener::bind("127.0.0.1:0").context("reserve Chromium debug port")?;
        let port = listener.local_addr()?.port();
        drop(listener);
        let profile_dir = std::env::temp_dir().join(format!("brazier-computer-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&profile_dir).context("create isolated Chromium profile")?;
        let mut command = Command::new(&executable);
        command
            .arg("--headless=new")
            .arg("--disable-gpu")
            .arg("--no-first-run")
            .arg("--no-default-browser-check")
            .arg(format!("--remote-debugging-port={port}"))
            .arg(format!("--user-data-dir={}", profile_dir.display()))
            .arg(format!(
                "--window-size={},{}",
                viewport.width, viewport.height
            ))
            .arg("about:blank")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        command.kill_on_drop(true);
        let mut process = command
            .spawn()
            .with_context(|| format!("launch Chromium at {executable}"))?;
        let debug_url = format!("http://127.0.0.1:{port}");
        for _ in 0..40 {
            if let Ok(Ok(response)) = timeout(
                Duration::from_secs(1),
                reqwest::get(format!("{debug_url}/json/version")),
            )
            .await
                && response.status().is_success()
            {
                return Ok(Self {
                    id: Uuid::new_v4().to_string(),
                    viewport,
                    debug_url,
                    process,
                    profile_dir,
                });
            }
            if let Some(status) = process.try_wait().context("check Chromium process")? {
                let _ = std::fs::remove_dir_all(&profile_dir);
                bail!(
                    "Chromium exited during startup ({status}); no browser action was performed."
                );
            }
            sleep(Duration::from_millis(100)).await;
        }
        let _ = process.kill().await;
        let _ = process.wait().await;
        let _ = std::fs::remove_dir_all(&profile_dir);
        bail!("Chromium did not expose its DevTools endpoint; no browser action was performed.")
    }

    async fn close(&mut self) {
        let _ = self.process.kill().await;
        let _ = self.process.wait().await;
        let _ = std::fs::remove_dir_all(&self.profile_dir);
    }

    async fn execute(
        &mut self,
        action: &ComputerAction,
        settle_delay_ms: u64,
    ) -> Result<ComputerActionResult> {
        let ws_url = self.page_websocket().await?;
        let (mut socket, _) = timeout(Duration::from_secs(5), connect_async(&ws_url))
            .await
            .context("timed out connecting to Chromium DevTools")??;
        let mut next_id = 1_u64;
        cdp(
            &mut socket,
            &mut next_id,
            "Emulation.setDeviceMetricsOverride",
            json!({
                "width": self.viewport.width,
                "height": self.viewport.height,
                "deviceScaleFactor": self.viewport.device_pixel_ratio.unwrap_or(1.0),
                "mobile": false,
            }),
        )
        .await?;
        let message = match action {
            ComputerAction::Screenshot => "Captured browser viewport.".to_owned(),
            ComputerAction::VisitUrl { url } => {
                if !(url.starts_with("http://")
                    || url.starts_with("https://")
                    || url == "about:blank")
                {
                    bail!("Only http(s) URLs are allowed; no navigation was performed.");
                }
                cdp(
                    &mut socket,
                    &mut next_id,
                    "Page.navigate",
                    json!({"url": url}),
                )
                .await?;
                wait_for_page_ready(&mut socket, &mut next_id).await?;
                format!("Navigated to {url}")
            }
            ComputerAction::WebSearch { query } => {
                let url = format!("https://duckduckgo.com/?q={}", urlencoding_lite(query));
                cdp(
                    &mut socket,
                    &mut next_id,
                    "Page.navigate",
                    json!({"url": url}),
                )
                .await?;
                wait_for_page_ready(&mut socket, &mut next_id).await?;
                format!("Opened search for {query}")
            }
            ComputerAction::LeftClick { x, y } => {
                mouse_click(&mut socket, &mut next_id, *x, *y, 1, "left").await?;
                format!("Clicked at ({x:.0}, {y:.0})")
            }
            ComputerAction::RightClick { x, y } => {
                mouse_click(&mut socket, &mut next_id, *x, *y, 1, "right").await?;
                format!("Right-clicked at ({x:.0}, {y:.0})")
            }
            ComputerAction::DoubleClick { x, y } => {
                mouse_click(&mut socket, &mut next_id, *x, *y, 2, "left").await?;
                format!("Double-clicked at ({x:.0}, {y:.0})")
            }
            ComputerAction::TripleClick { x, y } => {
                mouse_click(&mut socket, &mut next_id, *x, *y, 3, "left").await?;
                format!("Triple-clicked at ({x:.0}, {y:.0})")
            }
            ComputerAction::MouseMove { x, y } => {
                cdp(
                    &mut socket,
                    &mut next_id,
                    "Input.dispatchMouseEvent",
                    json!({"type":"mouseMoved","x":x,"y":y}),
                )
                .await?;
                format!("Moved pointer to ({x:.0}, {y:.0})")
            }
            ComputerAction::LeftClickDrag {
                start_x,
                start_y,
                end_x,
                end_y,
            } => {
                cdp(&mut socket, &mut next_id, "Input.dispatchMouseEvent", json!({"type":"mousePressed","x":start_x,"y":start_y,"button":"left","clickCount":1})).await?;
                cdp(
                    &mut socket,
                    &mut next_id,
                    "Input.dispatchMouseEvent",
                    json!({"type":"mouseMoved","x":end_x,"y":end_y,"button":"left","buttons":1}),
                )
                .await?;
                cdp(&mut socket, &mut next_id, "Input.dispatchMouseEvent", json!({"type":"mouseReleased","x":end_x,"y":end_y,"button":"left","clickCount":1})).await?;
                format!("Dragged ({start_x:.0},{start_y:.0}) to ({end_x:.0},{end_y:.0})")
            }
            ComputerAction::Type { text } => {
                cdp(
                    &mut socket,
                    &mut next_id,
                    "Input.insertText",
                    json!({"text": text}),
                )
                .await?;
                format!("Typed {} chars", text.chars().count())
            }
            ComputerAction::Keypress { keys } => {
                dispatch_keys(&mut socket, &mut next_id, keys).await?;
                format!("Pressed {}", keys.join("+"))
            }
            ComputerAction::Scroll {
                x,
                y,
                delta_x,
                delta_y,
            } => {
                cdp(
                    &mut socket,
                    &mut next_id,
                    "Input.dispatchMouseEvent",
                    json!({"type":"mouseWheel","x":x,"y":y,"deltaX":delta_x,"deltaY":delta_y}),
                )
                .await?;
                format!("Scrolled at ({x:.0}, {y:.0})")
            }
            ComputerAction::Wait { milliseconds } => {
                sleep(Duration::from_millis((*milliseconds).min(10_000))).await;
                format!("Waited {milliseconds}ms")
            }
            ComputerAction::Memorize { fact } => format!("Memorized: {fact}"),
            ComputerAction::AskUser { question } => {
                return self
                    .result(
                        ComputerActionStatus::WaitingForUser,
                        Some(question.clone()),
                        true,
                        None,
                    )
                    .await;
            }
            ComputerAction::Terminate { response } => {
                return self
                    .result(
                        ComputerActionStatus::Finished,
                        response.clone().or_else(|| Some("Task finished.".into())),
                        false,
                        None,
                    )
                    .await;
            }
        };
        if !matches!(
            action,
            ComputerAction::Screenshot | ComputerAction::Wait { .. }
        ) && settle_delay_ms > 0
        {
            sleep(Duration::from_millis(settle_delay_ms)).await;
        }
        self.result(
            ComputerActionStatus::Ok,
            Some(message),
            false,
            Some(&mut socket),
        )
        .await
    }

    async fn page_websocket(&self) -> Result<String> {
        let pages: Vec<Value> = timeout(
            Duration::from_secs(5),
            reqwest::get(format!("{}/json", self.debug_url)),
        )
        .await
        .context("timed out querying Chromium pages")??
        .error_for_status()?
        .json()
        .await
        .context("decode Chromium pages")?;
        pages
            .into_iter()
            .find(|page| page.get("type").and_then(Value::as_str) == Some("page"))
            .and_then(|page| {
                page.get("webSocketDebuggerUrl")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .context("Chromium has no controllable page")
    }

    async fn result(
        &self,
        status: ComputerActionStatus,
        message: Option<String>,
        needs_approval: bool,
        socket: Option<&mut CdpSocket>,
    ) -> Result<ComputerActionResult> {
        if let Some(socket) = socket {
            return self
                .result_from_socket(status, message, needs_approval, socket)
                .await;
        }
        let ws_url = self.page_websocket().await?;
        let (mut socket, _) = timeout(Duration::from_secs(5), connect_async(&ws_url))
            .await
            .context("timed out connecting to Chromium for screenshot")??;
        self.result_from_socket(status, message, needs_approval, &mut socket)
            .await
    }

    async fn result_from_socket(
        &self,
        status: ComputerActionStatus,
        message: Option<String>,
        needs_approval: bool,
        socket: &mut CdpSocket,
    ) -> Result<ComputerActionResult> {
        let mut id = 10_000;
        let capture = cdp(
            socket,
            &mut id,
            "Page.captureScreenshot",
            json!({"format":"png"}),
        )
        .await?;
        let screenshot_base64 = capture
            .get("data")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .context("Chromium screenshot missing image data")?;
        let info = cdp(socket, &mut id, "Runtime.evaluate", json!({"expression":"JSON.stringify({url: location.href, title: document.title})","returnByValue":true})).await?;
        let metadata = info
            // `cdp` unwraps the protocol envelope and returns `result`.
            .pointer("/result/value")
            .and_then(Value::as_str)
            .and_then(|value| serde_json::from_str::<Value>(value).ok())
            .unwrap_or(Value::Null);
        Ok(ComputerActionResult {
            status,
            message,
            screenshot_base64: Some(screenshot_base64),
            mime_type: Some("image/png".into()),
            viewport: Some(self.viewport.clone()),
            url: metadata
                .get("url")
                .and_then(Value::as_str)
                .map(str::to_owned),
            title: metadata
                .get("title")
                .and_then(Value::as_str)
                .map(str::to_owned),
            needs_approval,
            approval_id: None,
        })
    }

    async fn metadata(&self) -> Result<(Option<String>, Option<String>)> {
        let ws_url = self.page_websocket().await?;
        let (mut socket, _) = timeout(Duration::from_secs(5), connect_async(&ws_url))
            .await
            .context("timed out connecting to Chromium for metadata")??;
        let mut id = 1;
        let response = cdp(&mut socket, &mut id, "Runtime.evaluate", json!({"expression":"JSON.stringify({url: location.href, title: document.title})","returnByValue":true})).await?;
        let metadata = response
            .pointer("/result/value")
            .and_then(Value::as_str)
            .and_then(|value| serde_json::from_str::<Value>(value).ok())
            .unwrap_or(Value::Null);
        Ok((
            metadata
                .get("url")
                .and_then(Value::as_str)
                .map(str::to_owned),
            metadata
                .get("title")
                .and_then(Value::as_str)
                .map(str::to_owned),
        ))
    }
}

impl Drop for CdpBrowserSession {
    fn drop(&mut self) {
        // `kill_on_drop` handles the child asynchronously; remove the profile
        // synchronously as a best-effort cleanup for broker/process shutdown.
        let _ = self.process.start_kill();
        let _ = std::fs::remove_dir_all(&self.profile_dir);
    }
}

async fn cdp(socket: &mut CdpSocket, id: &mut u64, method: &str, params: Value) -> Result<Value> {
    let request_id = *id;
    *id += 1;
    socket
        .send(Message::Text(
            json!({"id":request_id,"method":method,"params":params})
                .to_string()
                .into(),
        ))
        .await
        .context("send Chromium command")?;
    while let Some(frame) = timeout(Duration::from_secs(10), socket.next())
        .await
        .context("timed out waiting for Chromium response")?
    {
        let frame = frame.context("read Chromium response")?;
        if let Message::Text(text) = frame {
            let response: Value =
                serde_json::from_str(&text).context("decode Chromium response")?;
            if response.get("id").and_then(Value::as_u64) == Some(request_id) {
                if let Some(error) = response.get("error") {
                    bail!("Chromium {method} failed: {error}");
                }
                return Ok(response.get("result").cloned().unwrap_or(Value::Null));
            }
        }
    }
    bail!("Chromium DevTools closed while executing {method}")
}

async fn mouse_click(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    id: &mut u64,
    x: f64,
    y: f64,
    count: u8,
    button: &str,
) -> Result<()> {
    cdp(
        socket,
        id,
        "Input.dispatchMouseEvent",
        json!({"type":"mousePressed","x":x,"y":y,"button":button,"clickCount":count}),
    )
    .await?;
    cdp(
        socket,
        id,
        "Input.dispatchMouseEvent",
        json!({"type":"mouseReleased","x":x,"y":y,"button":button,"clickCount":count}),
    )
    .await?;
    Ok(())
}

async fn dispatch_keys(socket: &mut CdpSocket, id: &mut u64, keys: &[String]) -> Result<()> {
    if keys.is_empty() {
        bail!("Keypress requires at least one key.");
    }
    let keys: Vec<_> = keys.iter().map(|key| normalize_key(key)).collect();
    let modifiers = keys.iter().fold(0_u8, |bits, key| bits | key.modifier);
    let ordinary: Vec<_> = keys.iter().filter(|key| key.modifier == 0).collect();
    // Press modifiers first, keep their bitfield on ordinary keys, then release
    // in reverse. This makes Ctrl+A/Command+A real shortcuts rather than text.
    for key in keys.iter().filter(|key| key.modifier != 0) {
        cdp(
            socket,
            id,
            "Input.dispatchKeyEvent",
            json!({"type":"rawKeyDown","key":key.key,"code":key.code,"modifiers":key.modifier,"windowsVirtualKeyCode":key.virtual_key,"nativeVirtualKeyCode":key.virtual_key}),
        )
        .await?;
    }
    for key in &ordinary {
        cdp(
            socket,
            id,
            "Input.dispatchKeyEvent",
            json!({"type":if modifiers == 0 { "keyDown" } else { "rawKeyDown" },"key":key.key,"code":key.code,"modifiers":modifiers,"windowsVirtualKeyCode":key.virtual_key,"nativeVirtualKeyCode":key.virtual_key}),
        )
        .await?;
        cdp(
            socket,
            id,
            "Input.dispatchKeyEvent",
            json!({"type":"keyUp","key":key.key,"code":key.code,"modifiers":modifiers,"windowsVirtualKeyCode":key.virtual_key,"nativeVirtualKeyCode":key.virtual_key}),
        )
        .await?;
    }
    for key in keys.iter().rev().filter(|key| key.modifier != 0) {
        cdp(
            socket,
            id,
            "Input.dispatchKeyEvent",
            json!({"type":"keyUp","key":key.key,"code":key.code,"modifiers":0,"windowsVirtualKeyCode":key.virtual_key,"nativeVirtualKeyCode":key.virtual_key}),
        )
        .await?;
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct NormalizedKey {
    key: String,
    code: String,
    modifier: u8,
    virtual_key: u16,
}

fn normalize_key(raw: &str) -> NormalizedKey {
    let upper = raw.trim().to_ascii_uppercase();
    let (key, code, modifier, virtual_key) = match upper.as_str() {
        "CTRL" | "CONTROL" => ("Control".into(), "ControlLeft".into(), 2, 17),
        "CMD" | "COMMAND" | "META" => ("Meta".into(), "MetaLeft".into(), 4, 91),
        "ALT" | "OPTION" => ("Alt".into(), "AltLeft".into(), 1, 18),
        "SHIFT" => ("Shift".into(), "ShiftLeft".into(), 8, 16),
        "ENTER" | "RETURN" => ("Enter".into(), "Enter".into(), 0, 13),
        "TAB" => ("Tab".into(), "Tab".into(), 0, 9),
        "ESC" | "ESCAPE" => ("Escape".into(), "Escape".into(), 0, 27),
        "SPACE" | "SPACEBAR" => (" ".into(), "Space".into(), 0, 32),
        "ARROWUP" | "UP" => ("ArrowUp".into(), "ArrowUp".into(), 0, 38),
        "ARROWDOWN" | "DOWN" => ("ArrowDown".into(), "ArrowDown".into(), 0, 40),
        "ARROWLEFT" | "LEFT" => ("ArrowLeft".into(), "ArrowLeft".into(), 0, 37),
        "ARROWRIGHT" | "RIGHT" => ("ArrowRight".into(), "ArrowRight".into(), 0, 39),
        value if value.len() == 1 && value.as_bytes()[0].is_ascii_alphabetic() => (
            value.to_ascii_lowercase(),
            format!("Key{value}"),
            0,
            value.as_bytes()[0] as u16,
        ),
        value if value.len() == 1 && value.as_bytes()[0].is_ascii_digit() => (
            value.into(),
            format!("Digit{value}"),
            0,
            value.as_bytes()[0] as u16,
        ),
        _ => (raw.to_owned(), raw.to_owned(), 0, 0),
    };
    NormalizedKey {
        key,
        code,
        modifier,
        virtual_key,
    }
}

async fn wait_for_page_ready(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    id: &mut u64,
) -> Result<()> {
    for _ in 0..50 {
        let ready = cdp(
            socket,
            id,
            "Runtime.evaluate",
            json!({"expression":"document.readyState","returnByValue":true}),
        )
        .await?;
        if matches!(
            ready.pointer("/result/value").and_then(Value::as_str),
            Some("interactive" | "complete")
        ) {
            return Ok(());
        }
        sleep(Duration::from_millis(100)).await;
    }
    bail!("navigation did not become ready within 5 seconds")
}

/// Whether a runnable Chromium binary is discoverable for browser computer use.
/// This intentionally validates `--version`, rather than merely checking that
/// a file exists, so callers never advertise an unusable driver.
pub fn chromium_available() -> bool {
    find_chromium(None).is_some()
}

fn find_chromium(configured: Option<&str>) -> Option<String> {
    configured
        .map(str::to_owned)
        .or_else(|| std::env::var("BRAZIER_CHROMIUM_PATH").ok())
        .or_else(|| {
            [
                "chromium",
                "chromium-browser",
                "google-chrome",
                "google-chrome-stable",
            ]
            .iter()
            .find_map(|name| {
                std::process::Command::new(name)
                    .arg("--version")
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .ok()
                    .filter(|status| status.success())
                    .map(|_| (*name).to_owned())
            })
        })
}

fn urlencoding_lite(value: &str) -> String {
    value.bytes().fold(String::new(), |mut out, byte| {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{byte:02X}")),
        };
        out
    })
}

/// Registry keeps only short map lookups under its global mutex. Each browser
/// has its own mutex, so a long `wait` in one session never blocks another.
pub struct BrowserSessionRegistry {
    sessions: Mutex<HashMap<String, Arc<Mutex<CdpBrowserSession>>>>,
    executable: Option<String>,
}

impl Default for BrowserSessionRegistry {
    fn default() -> Self {
        Self::new()
    }
}
impl BrowserSessionRegistry {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            executable: None,
        }
    }
    #[cfg(test)]
    pub fn with_executable(executable: impl Into<String>) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            executable: Some(executable.into()),
        }
    }
    pub async fn open(&self, viewport: ComputerViewport) -> Result<String> {
        let session = CdpBrowserSession::launch(viewport, self.executable.as_deref()).await?;
        let id = session.id.clone();
        self.sessions
            .lock()
            .await
            .insert(id.clone(), Arc::new(Mutex::new(session)));
        Ok(id)
    }
    pub async fn close(&self, id: &str) {
        if let Some(session) = self.sessions.lock().await.remove(id) {
            session.lock().await.close().await;
        }
    }
    async fn session(&self, id: &str) -> Result<Arc<Mutex<CdpBrowserSession>>> {
        self.sessions
            .lock()
            .await
            .get(id)
            .cloned()
            .with_context(|| format!("unknown browser session {id}"))
    }
    pub async fn snapshot(
        &self,
        id: &str,
    ) -> Result<(ComputerViewport, Option<String>, Option<String>)> {
        let session = self.session(id).await?;
        let session = session.lock().await;
        let (url, title) = session.metadata().await?;
        Ok((session.viewport.clone(), url, title))
    }
    pub async fn execute(
        &self,
        id: &str,
        action: &ComputerAction,
        settle_delay_ms: u64,
    ) -> Result<ComputerActionResult> {
        self.session(id)
            .await?
            .lock()
            .await
            .execute(action, settle_delay_ms)
            .await
    }
}
pub type SharedBrowserRegistry = Arc<BrowserSessionRegistry>;

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    #[tokio::test]
    async fn unavailable_driver_never_fakes_a_screenshot() {
        let registry = BrowserSessionRegistry::with_executable("/definitely/missing/chromium");
        assert!(registry.open(ComputerViewport::default()).await.is_err());
    }
    #[test]
    fn normalizes_model_key_names_for_cdp() {
        assert_eq!(
            normalize_key("CTRL"),
            NormalizedKey {
                key: "Control".into(),
                code: "ControlLeft".into(),
                modifier: 2,
                virtual_key: 17
            }
        );
        assert_eq!(
            normalize_key("cmd"),
            NormalizedKey {
                key: "Meta".into(),
                code: "MetaLeft".into(),
                modifier: 4,
                virtual_key: 91
            }
        );
        assert_eq!(
            normalize_key("ENTER"),
            NormalizedKey {
                key: "Enter".into(),
                code: "Enter".into(),
                modifier: 0,
                virtual_key: 13
            }
        );
        assert_eq!(
            normalize_key("a"),
            NormalizedKey {
                key: "a".into(),
                code: "KeyA".into(),
                modifier: 0,
                virtual_key: 65
            }
        );
        assert_eq!(
            normalize_key("ArrowUp"),
            NormalizedKey {
                key: "ArrowUp".into(),
                code: "ArrowUp".into(),
                modifier: 0,
                virtual_key: 38
            }
        );
    }
    #[tokio::test]
    #[ignore = "requires a working Chromium and local TCP loopback"]
    async fn a_wait_in_one_session_does_not_block_another() {
        if !chromium_available() {
            return;
        }
        let registry = BrowserSessionRegistry::new();
        let first = registry.open(ComputerViewport::default()).await.unwrap();
        let second = registry.open(ComputerViewport::default()).await.unwrap();
        let registry = Arc::new(registry);
        let wait_registry = Arc::clone(&registry);
        let wait = tokio::spawn(async move {
            wait_registry
                .execute(&first, &ComputerAction::Wait { milliseconds: 400 }, 0)
                .await
                .unwrap();
        });
        sleep(Duration::from_millis(30)).await;
        let start = std::time::Instant::now();
        registry
            .execute(&second, &ComputerAction::Screenshot, 0)
            .await
            .unwrap();
        assert!(start.elapsed() < Duration::from_millis(300));
        wait.await.unwrap();
    }
    #[tokio::test]
    #[ignore = "requires a working Chromium and local TCP loopback"]
    async fn chromium_executes_real_dom_input_and_returns_a_viewport_screenshot() {
        if !chromium_available() {
            return;
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            // One document response is sufficient; Chromium may abandon its
            // favicon request when this small test server closes.
            if let Ok((mut stream, _)) = listener.accept().await {
                let mut request = [0_u8; 4096];
                let _ = stream.read(&mut request).await;
                let body = r#"<!doctype html><title>ready</title><input id=i style='position:absolute;left:10px;top:10px' oninput='document.title="typed:"+this.value'><button style='position:absolute;left:10px;top:60px' onclick='document.title="clicked"'>go</button>"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        let registry = BrowserSessionRegistry::new();
        let viewport = ComputerViewport {
            width: 640,
            height: 480,
            device_pixel_ratio: Some(1.0),
        };
        let id = registry.open(viewport.clone()).await.unwrap();
        let navigated = registry
            .execute(
                &id,
                &ComputerAction::VisitUrl {
                    url: format!("http://{address}"),
                },
                0,
            )
            .await
            .unwrap();
        assert_eq!(navigated.title.as_deref(), Some("ready"));
        registry
            .execute(&id, &ComputerAction::LeftClick { x: 20.0, y: 20.0 }, 0)
            .await
            .unwrap();
        let typed = registry
            .execute(
                &id,
                &ComputerAction::Type {
                    text: "hello".into(),
                },
                0,
            )
            .await
            .unwrap();
        assert_eq!(typed.title.as_deref(), Some("typed:hello"));
        registry
            .execute(
                &id,
                &ComputerAction::Keypress {
                    keys: vec!["CTRL".into(), "a".into()],
                },
                0,
            )
            .await
            .unwrap();
        let replaced = registry
            .execute(
                &id,
                &ComputerAction::Type {
                    text: "replaced".into(),
                },
                0,
            )
            .await
            .unwrap();
        assert_eq!(replaced.title.as_deref(), Some("typed:replaced"));
        let clicked = registry
            .execute(&id, &ComputerAction::LeftClick { x: 25.0, y: 75.0 }, 0)
            .await
            .unwrap();
        assert_eq!(clicked.title.as_deref(), Some("clicked"));
        let png = base64::engine::general_purpose::STANDARD
            .decode(clicked.screenshot_base64.unwrap())
            .unwrap();
        assert_eq!(&png[16..20], &640_u32.to_be_bytes());
        assert_eq!(&png[20..24], &480_u32.to_be_bytes());
        registry.close(&id).await;
    }
}

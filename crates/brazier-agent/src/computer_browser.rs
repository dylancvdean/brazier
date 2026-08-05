//! Isolated Chromium/CDP target for Computer Use.
//!
//! Browser actions are deliberately backed by a real Chromium process.  We do
//! not provide a synthetic fallback: reporting an unavailable browser is safer
//! than telling a model that a click or a screenshot happened when it did not.

use std::{
    collections::HashMap,
    net::IpAddr,
    os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd},
    path::PathBuf,
    process::Stdio,
    sync::Arc,
};

use anyhow::{Context, Result, bail};
use brazier_protocol::computer_types::{
    ComputerAction, ComputerActionResult, ComputerActionStatus, ComputerViewport,
};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child, Command},
    sync::{Mutex, broadcast, mpsc},
    time::{Duration, sleep, timeout},
};
use uuid::Uuid;

const DRIVER_UNAVAILABLE: &str =
    "Browser computer use requires a working Chromium installation; no action was performed.";

/// Cooperative cancel signal for an in-flight computer action (Esc / stop).
pub struct ActionCancel {
    cancelled: std::sync::atomic::AtomicBool,
    notify: tokio::sync::Notify,
}

impl ActionCancel {
    pub fn new() -> Self {
        Self {
            cancelled: std::sync::atomic::AtomicBool::new(false),
            notify: tokio::sync::Notify::new(),
        }
    }

    pub fn reset(&self) {
        self.cancelled
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn trip(&self) {
        self.cancelled
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub async fn cancelled(&self) {
        loop {
            let notified = self.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
            if self.is_cancelled() {
                return;
            }
        }
    }
}

impl Default for ActionCancel {
    fn default() -> Self {
        Self::new()
    }
}

/// A single CDP event (no `id`) arriving over the DevTools pipe, forwarded to
/// subscribers such as the live screencast relay.
#[derive(Clone)]
struct CdpEvent {
    method: String,
    params: Value,
}

/// Shared write handle the screencast relay uses to acknowledge frames without
/// serializing against the pipe's normal command/response flow.
#[derive(Clone)]
struct CdpWriter {
    writer: Arc<Mutex<tokio::fs::File>>,
    page_session_id: String,
}

impl CdpWriter {
    async fn send_ack(&self, frame_id: &Value) -> Result<()> {
        let mut message = json!({
            "method": "Page.screencastFrameAck",
            "params": { "sessionId": frame_id },
        });
        message["sessionId"] = Value::String(self.page_session_id.clone());
        let mut bytes = message.to_string().into_bytes();
        bytes.push(0);
        self.writer
            .lock()
            .await
            .write_all(&bytes)
            .await
            .context("acknowledge Chromium screencast frame")
    }
}

/// Private CDP transport over Chromium's `--remote-debugging-pipe`.
///
/// Unlike `--remote-debugging-port`, this never opens a TCP listener that any
/// local process could attach to for the session lifetime. A background task
/// owns the read end, routing command responses to their pending calls while
/// broadcasting events (screencast frames, etc.) to subscribers.
struct CdpPipe {
    writer: Arc<Mutex<tokio::fs::File>>,
    next_id: u64,
    page_session_id: String,
    /// Pending command responses keyed by request id.
    pending: Arc<Mutex<HashMap<u64, mpsc::UnboundedSender<anyhow::Result<Value>>>>>,
    events: broadcast::Sender<CdpEvent>,
}

impl CdpPipe {
    fn new(reader: tokio::fs::File, writer: tokio::fs::File) -> Self {
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let (events, _) = broadcast::channel(64);
        spawn_reader(reader, Arc::clone(&pending), events.clone());
        Self {
            writer: Arc::new(Mutex::new(writer)),
            next_id: 1,
            page_session_id: String::new(),
            pending,
            events,
        }
    }

    fn subscribe_events(&self) -> broadcast::Receiver<CdpEvent> {
        self.events.subscribe()
    }

    fn writer_handle(&self) -> CdpWriter {
        CdpWriter {
            writer: Arc::clone(&self.writer),
            page_session_id: self.page_session_id.clone(),
        }
    }

    async fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        self.call_raw(method, params, Some(self.page_session_id.clone()))
            .await
    }

    async fn call_browser(&mut self, method: &str, params: Value) -> Result<Value> {
        self.call_raw(method, params, None).await
    }

    async fn call_raw(
        &mut self,
        method: &str,
        params: Value,
        session_id: Option<String>,
    ) -> Result<Value> {
        let request_id = self.next_id;
        self.next_id += 1;
        let mut message = json!({"id": request_id, "method": method, "params": params});
        if let Some(session_id) = session_id {
            message["sessionId"] = Value::String(session_id);
        }
        let (tx, mut rx) = mpsc::unbounded_channel();
        self.pending.lock().await.insert(request_id, tx);
        let mut bytes = message.to_string().into_bytes();
        bytes.push(0);
        timeout(Duration::from_secs(10), self.writer.lock().await.write_all(&bytes))
            .await
            .context("timed out sending Chromium command")?
            .context("send Chromium command")?;
        match timeout(Duration::from_secs(10), rx.recv()).await {
            Ok(Some(result)) => result,
            Ok(None) => {
                self.pending.lock().await.remove(&request_id);
                bail!("Chromium DevTools channel closed while executing {method}");
            }
            Err(_) => {
                self.pending.lock().await.remove(&request_id);
                bail!("timed out waiting for Chromium response to {method}");
            }
        }
    }
}

/// Own the read end of the DevTools pipe and dispatch its messages. Frames are
/// null-delimited JSON: responses carry an `id`, events do not.
fn spawn_reader(
    mut reader: tokio::fs::File,
    pending: Arc<Mutex<HashMap<u64, mpsc::UnboundedSender<anyhow::Result<Value>>>>>,
    events: broadcast::Sender<CdpEvent>,
) {
    tokio::spawn(async move {
        let mut buffer = Vec::new();
        loop {
            let mut chunk = [0_u8; 8192];
            let read = match reader.read(&mut chunk).await {
                Ok(0) | Err(_) => break,
                Ok(read) => read,
            };
            buffer.extend_from_slice(&chunk[..read]);
            while let Some(end) = buffer.iter().position(|byte| *byte == 0) {
                let frame = String::from_utf8_lossy(&buffer[..end]).into_owned();
                buffer.drain(..=end);
                let message: Value = match serde_json::from_str(&frame) {
                    Ok(message) => message,
                    Err(_) => continue,
                };
                if let Some(id) = message.get("id").and_then(Value::as_u64) {
                    let reply = if let Some(error) = message.get("error") {
                        Err(anyhow::anyhow!("Chromium command failed: {error}"))
                    } else {
                        Ok(message.get("result").cloned().unwrap_or(Value::Null))
                    };
                    if let Some(tx) = pending.lock().await.remove(&id) {
                        let _ = tx.send(reply);
                    }
                } else if let (Some(method), Some(params)) = (
                    message.get("method").and_then(Value::as_str),
                    message.get("params"),
                ) {
                    let _ = events.send(CdpEvent {
                        method: method.to_owned(),
                        params: params.clone(),
                    });
                }
            }
        }
    });
}

struct CdpBrowserSession {
    id: String,
    viewport: ComputerViewport,
    process: Child,
    profile_dir: PathBuf,
    pipe: CdpPipe,
}

impl CdpBrowserSession {
    async fn launch(viewport: ComputerViewport, executable: Option<&str>) -> Result<Self> {
        let executable =
            find_chromium(executable).ok_or_else(|| anyhow::anyhow!(DRIVER_UNAVAILABLE))?;
        let profile_dir = std::env::temp_dir().join(format!("brazier-computer-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&profile_dir).context("create isolated Chromium profile")?;

        // Chromium reads CDP from FD 3 and writes responses to FD 4.
        let (to_chrome_read, to_chrome_write) =
            std::io::pipe().context("create Chromium CDP input pipe")?;
        let (from_chrome_read, from_chrome_write) =
            std::io::pipe().context("create Chromium CDP output pipe")?;
        let to_chrome_read_fd = to_chrome_read.as_raw_fd();
        let from_chrome_write_fd = from_chrome_write.as_raw_fd();

        let mut command = Command::new(&executable);
        command
            .arg("--headless=new")
            .arg("--disable-gpu")
            .arg("--no-first-run")
            .arg("--no-default-browser-check")
            .arg("--remote-debugging-pipe")
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
        unsafe {
            command.pre_exec(move || {
                if libc::dup2(to_chrome_read_fd, 3) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::dup2(from_chrome_write_fd, 4) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let process = command
            .spawn()
            .with_context(|| format!("launch Chromium at {executable}"))?;
        // Parent keeps the write end of the input pipe and the read end of the
        // output pipe; close the ends Chromium already inherited.
        drop(to_chrome_read);
        drop(from_chrome_write);

        let reader = tokio::fs::File::from_std(std::fs::File::from(unsafe {
            OwnedFd::from_raw_fd(from_chrome_read.into_raw_fd())
        }));
        let writer = tokio::fs::File::from_std(std::fs::File::from(unsafe {
            OwnedFd::from_raw_fd(to_chrome_write.into_raw_fd())
        }));
        let mut pipe = CdpPipe::new(reader, writer);
        let page_session_id = match timeout(Duration::from_secs(5), attach_page_session(&mut pipe))
            .await
        {
            Ok(Ok(session_id)) => session_id,
            Ok(Err(error)) => {
                let _ = std::fs::remove_dir_all(&profile_dir);
                return Err(error).context("attach Chromium page session");
            }
            Err(_) => {
                let _ = std::fs::remove_dir_all(&profile_dir);
                bail!("Chromium did not expose a page target over the DevTools pipe");
            }
        };
        pipe.page_session_id = page_session_id;
        Ok(Self {
            id: Uuid::new_v4().to_string(),
            viewport,
            process,
            profile_dir,
            pipe,
        })
    }

    async fn close(&mut self) {
        let _ = self.process.kill().await;
        let _ = self.process.wait().await;
        let _ = std::fs::remove_dir_all(&self.profile_dir);
    }

    /// Begin streaming the page as JPEG screencast frames into `frames`
    /// (base64 payloads). The relay acks each frame so Chromium keeps
    /// producing them and ends when the browser closes; a slow subscriber
    /// simply misses frames instead of backing up the pipe.
    async fn start_screencast(&mut self, frames: broadcast::Sender<String>) -> Result<()> {
        self.pipe
            .call(
                "Page.startScreencast",
                json!({
                    "format": "jpeg",
                    "quality": 60,
                    "maxWidth": self.viewport.width,
                    "maxHeight": self.viewport.height,
                    "everyNthFrame": 1,
                }),
            )
            .await?;
        let mut events = self.pipe.subscribe_events();
        let writer = self.pipe.writer_handle();
        tokio::spawn(async move {
            loop {
                let event = match events.recv().await {
                    Ok(event) => event,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => break,
                };
                if event.method != "Page.screencastFrame" {
                    continue;
                }
                let data = event
                    .params
                    .get("data")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let frame_id = event.params.get("sessionId").cloned();
                if let Some(frame_id) = frame_id {
                    let _ = writer.send_ack(&frame_id).await;
                }
                if let Some(data) = data {
                    let _ = frames.send(data);
                }
            }
        });
        Ok(())
    }

    async fn execute(
        &mut self,
        action: &ComputerAction,
        settle_delay_ms: u64,
        cancel: Option<&ActionCancel>,
    ) -> Result<ComputerActionResult> {
        self.pipe
            .call(
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
                ensure_public_navigation_url(url).await?;
                self.pipe
                    .call("Page.navigate", json!({"url": url}))
                    .await?;
                wait_for_page_ready(&mut self.pipe).await?;
                format!("Navigated to {url}")
            }
            ComputerAction::WebSearch { query } => {
                let url = format!("https://duckduckgo.com/?q={}", urlencoding_lite(query));
                self.pipe
                    .call("Page.navigate", json!({"url": url}))
                    .await?;
                wait_for_page_ready(&mut self.pipe).await?;
                format!("Opened search for {query}")
            }
            ComputerAction::LeftClick { x, y } => {
                mouse_click(&mut self.pipe, *x, *y, 1, "left").await?;
                format!("Clicked at ({x:.0}, {y:.0})")
            }
            ComputerAction::RightClick { x, y } => {
                mouse_click(&mut self.pipe, *x, *y, 1, "right").await?;
                format!("Right-clicked at ({x:.0}, {y:.0})")
            }
            ComputerAction::DoubleClick { x, y } => {
                mouse_click(&mut self.pipe, *x, *y, 2, "left").await?;
                format!("Double-clicked at ({x:.0}, {y:.0})")
            }
            ComputerAction::TripleClick { x, y } => {
                mouse_click(&mut self.pipe, *x, *y, 3, "left").await?;
                format!("Triple-clicked at ({x:.0}, {y:.0})")
            }
            ComputerAction::MouseMove { x, y } => {
                self.pipe
                    .call(
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
                self.pipe.call("Input.dispatchMouseEvent", json!({"type":"mousePressed","x":start_x,"y":start_y,"button":"left","clickCount":1})).await?;
                self.pipe
                    .call(
                        "Input.dispatchMouseEvent",
                        json!({"type":"mouseMoved","x":end_x,"y":end_y,"button":"left","buttons":1}),
                    )
                    .await?;
                self.pipe.call("Input.dispatchMouseEvent", json!({"type":"mouseReleased","x":end_x,"y":end_y,"button":"left","clickCount":1})).await?;
                format!("Dragged ({start_x:.0},{start_y:.0}) to ({end_x:.0},{end_y:.0})")
            }
            ComputerAction::Type { text } => {
                self.pipe
                    .call("Input.insertText", json!({"text": text}))
                    .await?;
                format!("Typed {} chars", text.chars().count())
            }
            ComputerAction::Keypress { keys } => {
                dispatch_keys(&mut self.pipe, keys).await?;
                format!("Pressed {}", keys.join("+"))
            }
            ComputerAction::Scroll {
                x,
                y,
                delta_x,
                delta_y,
            } => {
                self.pipe
                    .call(
                        "Input.dispatchMouseEvent",
                        json!({"type":"mouseWheel","x":x,"y":y,"deltaX":delta_x,"deltaY":delta_y}),
                    )
                    .await?;
                format!("Scrolled at ({x:.0}, {y:.0})")
            }
            ComputerAction::Wait { milliseconds } => {
                cancellable_sleep(Duration::from_millis((*milliseconds).min(10_000)), cancel)
                    .await?;
                format!("Waited {milliseconds}ms")
            }
            ComputerAction::Memorize { fact } => format!("Memorized: {fact}"),
            ComputerAction::AskUser { question } => {
                return self
                    .result(
                        ComputerActionStatus::WaitingForUser,
                        Some(question.clone()),
                        true,
                    )
                    .await;
            }
            ComputerAction::Terminate { response } => {
                return self
                    .result(
                        ComputerActionStatus::Finished,
                        response.clone().or_else(|| Some("Task finished.".into())),
                        false,
                    )
                    .await;
            }
        };
        if !matches!(
            action,
            ComputerAction::Screenshot | ComputerAction::Wait { .. }
        ) && settle_delay_ms > 0
        {
            cancellable_sleep(Duration::from_millis(settle_delay_ms), cancel).await?;
        }
        self.result(ComputerActionStatus::Ok, Some(message), false)
            .await
    }

    async fn result(
        &mut self,
        status: ComputerActionStatus,
        message: Option<String>,
        needs_approval: bool,
    ) -> Result<ComputerActionResult> {
        let capture = self
            .pipe
            .call("Page.captureScreenshot", json!({"format":"png"}))
            .await?;
        let screenshot_base64 = capture
            .get("data")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .context("Chromium screenshot missing image data")?;
        let info = self
            .pipe
            .call(
                "Runtime.evaluate",
                json!({"expression":"JSON.stringify({url: location.href, title: document.title})","returnByValue":true}),
            )
            .await?;
        let metadata = info
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

    async fn metadata(&mut self) -> Result<(Option<String>, Option<String>)> {
        let response = self
            .pipe
            .call(
                "Runtime.evaluate",
                json!({"expression":"JSON.stringify({url: location.href, title: document.title})","returnByValue":true}),
            )
            .await?;
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

async fn attach_page_session(pipe: &mut CdpPipe) -> Result<String> {
    // Poll Target.getTargets until the initial about:blank page appears.
    for _ in 0..50 {
        let targets = pipe
            .call_browser("Target.getTargets", json!({}))
            .await?;
        let page = targets
            .get("targetInfos")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|target| target.get("type").and_then(Value::as_str) == Some("page"));
        if let Some(page) = page {
            let target_id = page
                .get("targetId")
                .and_then(Value::as_str)
                .context("page target missing id")?;
            let attached = pipe
                .call_browser(
                    "Target.attachToTarget",
                    json!({"targetId": target_id, "flatten": true}),
                )
                .await?;
            return attached
                .get("sessionId")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .context("attachToTarget missing sessionId");
        }
        sleep(Duration::from_millis(100)).await;
    }
    bail!("Chromium has no controllable page")
}

async fn cancellable_sleep(duration: Duration, cancel: Option<&ActionCancel>) -> Result<()> {
    let Some(cancel) = cancel else {
        sleep(duration).await;
        return Ok(());
    };
    tokio::select! {
        _ = sleep(duration) => Ok(()),
        _ = cancel.cancelled() => bail!("computer action cancelled"),
    }
}

async fn mouse_click(
    pipe: &mut CdpPipe,
    x: f64,
    y: f64,
    count: u8,
    button: &str,
) -> Result<()> {
    pipe.call(
        "Input.dispatchMouseEvent",
        json!({"type":"mousePressed","x":x,"y":y,"button":button,"clickCount":count}),
    )
    .await?;
    pipe.call(
        "Input.dispatchMouseEvent",
        json!({"type":"mouseReleased","x":x,"y":y,"button":button,"clickCount":count}),
    )
    .await?;
    Ok(())
}

async fn dispatch_keys(pipe: &mut CdpPipe, keys: &[String]) -> Result<()> {
    if keys.is_empty() {
        bail!("Keypress requires at least one key.");
    }
    let keys: Vec<_> = keys.iter().map(|key| normalize_key(key)).collect();
    let modifiers = keys.iter().fold(0_u8, |bits, key| bits | key.modifier);
    let ordinary: Vec<_> = keys.iter().filter(|key| key.modifier == 0).collect();
    // Press modifiers first, keep their bitfield on ordinary keys, then release
    // in reverse. This makes Ctrl+A/Command+A real shortcuts rather than text.
    for key in keys.iter().filter(|key| key.modifier != 0) {
        pipe.call(
            "Input.dispatchKeyEvent",
            json!({"type":"rawKeyDown","key":key.key,"code":key.code,"modifiers":key.modifier,"windowsVirtualKeyCode":key.virtual_key,"nativeVirtualKeyCode":key.virtual_key}),
        )
        .await?;
    }
    for key in &ordinary {
        pipe.call(
            "Input.dispatchKeyEvent",
            json!({"type":if modifiers == 0 { "keyDown" } else { "rawKeyDown" },"key":key.key,"code":key.code,"modifiers":modifiers,"windowsVirtualKeyCode":key.virtual_key,"nativeVirtualKeyCode":key.virtual_key}),
        )
        .await?;
        pipe.call(
            "Input.dispatchKeyEvent",
            json!({"type":"keyUp","key":key.key,"code":key.code,"modifiers":modifiers,"windowsVirtualKeyCode":key.virtual_key,"nativeVirtualKeyCode":key.virtual_key}),
        )
        .await?;
    }
    for key in keys.iter().rev().filter(|key| key.modifier != 0) {
        pipe.call(
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

async fn wait_for_page_ready(pipe: &mut CdpPipe) -> Result<()> {
    for _ in 0..50 {
        let ready = pipe
            .call(
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

/// Block browser computer-use from navigating to private/link-local targets.
/// `about:blank` remains allowed as the isolated profile's initial page.
async fn ensure_public_navigation_url(url: &str) -> Result<()> {
    if url == "about:blank" {
        return Ok(());
    }
    let parsed = reqwest::Url::parse(url).context("invalid navigation URL")?;
    anyhow::ensure!(
        matches!(parsed.scheme(), "http" | "https"),
        "Only http(s) URLs are allowed; no navigation was performed."
    );
    let host = parsed.host_str().context("navigation URL has no host")?;
    // Some Url serializers keep brackets around IPv6 literals; strip so parse works.
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = host.parse::<IpAddr>() {
        #[cfg(test)]
        if ip.is_loopback() {
            // Integration tests drive an ephemeral loopback fixture page.
            return Ok(());
        }
        anyhow::ensure!(
            navigation_ip_is_public(ip),
            "Refusing to navigate to non-public address {ip}."
        );
        return Ok(());
    }
    #[cfg(not(test))]
    anyhow::ensure!(
        !host.eq_ignore_ascii_case("localhost") && !host.ends_with(".local"),
        "Refusing to navigate to local hostname {host}."
    );
    #[cfg(test)]
    if host.eq_ignore_ascii_case("localhost") {
        return Ok(());
    }
    anyhow::ensure!(
        !host.ends_with(".local"),
        "Refusing to navigate to local hostname {host}."
    );
    let port = parsed.port_or_known_default().unwrap_or(443);
    let mut saw_address = false;
    for address in tokio::net::lookup_host((host, port))
        .await
        .with_context(|| format!("resolve navigation host {host}"))?
    {
        saw_address = true;
        anyhow::ensure!(
            navigation_ip_is_public(address.ip()),
            "Refusing to navigate to {host}; it resolves to non-public address {}.",
            address.ip()
        );
    }
    anyhow::ensure!(
        saw_address,
        "Refusing to navigate to {host}; DNS returned no addresses."
    );
    Ok(())
}

fn navigation_ip_is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            !(v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_documentation()
                || (v4.octets()[0] == 100 && (64..=127).contains(&v4.octets()[1])))
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return navigation_ip_is_public(IpAddr::V4(v4));
            }
            let segments = v6.segments();
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (segments[0] & 0xfe00) == 0xfc00
                || (segments[0] & 0xffc0) == 0xfe80
                || (segments[0] == 0x2001 && segments[1] == 0xdb8))
        }
    }
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
        let mut session = session.lock().await;
        let (url, title) = session.metadata().await?;
        Ok((session.viewport.clone(), url, title))
    }
    pub async fn execute(
        &self,
        id: &str,
        action: &ComputerAction,
        settle_delay_ms: u64,
        cancel: Option<&ActionCancel>,
    ) -> Result<ComputerActionResult> {
        self.session(id)
            .await?
            .lock()
            .await
            .execute(action, settle_delay_ms, cancel)
            .await
    }
    /// Start streaming live frames for a browser session. Re-entrant: calling
    /// again for a browser already screencasting simply subscribes more senders.
    pub async fn start_screencast(&self, id: &str, frames: broadcast::Sender<String>) -> Result<()> {
        self.session(id)
            .await?
            .lock()
            .await
            .start_screencast(frames)
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
    async fn visit_url_refuses_private_and_link_local_targets() {
        for url in [
            "http://10.0.0.1/",
            "http://169.254.169.254/latest",
            "http://[::ffff:127.0.0.1]/",
            "http://[::ffff:169.254.169.254]/",
            "http://something.local/",
        ] {
            let error = ensure_public_navigation_url(url)
                .await
                .expect_err(url)
                .to_string();
            assert!(
                error.contains("Refusing") || error.contains("non-public") || error.contains("local"),
                "{url} => {error}"
            );
        }
        // Literal public address avoids a DNS dependency in unit tests.
        ensure_public_navigation_url("https://93.184.216.34/")
            .await
            .expect("public host");
        ensure_public_navigation_url("about:blank")
            .await
            .expect("blank");
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
                .execute(&first, &ComputerAction::Wait { milliseconds: 400 }, 0, None)
                .await
                .unwrap();
        });
        sleep(Duration::from_millis(30)).await;
        let start = std::time::Instant::now();
        registry
            .execute(&second, &ComputerAction::Screenshot, 0, None)
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
                None,
            )
            .await
            .unwrap();
        assert_eq!(navigated.title.as_deref(), Some("ready"));
        registry
            .execute(&id, &ComputerAction::LeftClick { x: 20.0, y: 20.0 }, 0, None)
            .await
            .unwrap();
        let typed = registry
            .execute(
                &id,
                &ComputerAction::Type {
                    text: "hello".into(),
                },
                0,
                None,
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
                None,
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
                None,
            )
            .await
            .unwrap();
        assert_eq!(replaced.title.as_deref(), Some("typed:replaced"));
        let clicked = registry
            .execute(&id, &ComputerAction::LeftClick { x: 25.0, y: 75.0 }, 0, None)
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
    #[tokio::test]
    #[ignore = "requires a working Chromium and local TCP loopback"]
    async fn screencast_streams_live_jpeg_frames() {
        if !chromium_available() {
            return;
        }
        let registry = BrowserSessionRegistry::new();
        let viewport = ComputerViewport {
            width: 640,
            height: 480,
            device_pixel_ratio: Some(1.0),
        };
        let id = registry.open(viewport).await.unwrap();
        let (frames, mut rx) = tokio::sync::broadcast::channel::<String>(8);
        registry.start_screencast(&id, frames).await.unwrap();
        let frame = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("screencast produced no frame within 5s")
            .expect("screencast stream closed before the first frame");
        let jpeg = base64::engine::general_purpose::STANDARD
            .decode(frame)
            .unwrap();
        assert_eq!(&jpeg[..3], &[0xFF, 0xD8, 0xFF], "expected a JPEG screencast frame");
        registry.close(&id).await;
    }
}

//! Isolated browser target for Computer Use.
//!
//! The first driver keeps an in-process session with a synthetic viewport so the
//! observe–act loop and policy path can be exercised without bundling Chromium.
//! When a system Chrome/Chromium with remote debugging is configured later, a
//! CDP-backed driver can replace [`SyntheticBrowserSession`] behind the same
//! trait.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use brazier_protocol::computer_types::{
    ComputerAction, ComputerActionResult, ComputerActionStatus, ComputerViewport,
};
use tokio::sync::Mutex;
use uuid::Uuid;

pub trait BrowserDriver: Send + Sync {
    fn id(&self) -> &str;
    fn viewport(&self) -> ComputerViewport;
    fn url(&self) -> Option<String>;
    fn title(&self) -> Option<String>;
    fn execute(
        &mut self,
        action: &ComputerAction,
    ) -> impl std::future::Future<Output = Result<ComputerActionResult>> + Send;
}

/// Lightweight browser stand-in used until a Chromium worker is wired.
pub struct SyntheticBrowserSession {
    id: String,
    viewport: ComputerViewport,
    url: Option<String>,
    title: Option<String>,
    memories: Vec<String>,
}

impl SyntheticBrowserSession {
    pub fn new(viewport: ComputerViewport) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            viewport,
            url: Some("about:blank".into()),
            title: Some("New tab".into()),
            memories: Vec::new(),
        }
    }

    fn render_frame(&self) -> String {
        // Minimal 1×1 PNG so adapters always receive an image payload.
        // Real Chromium capture will replace this with a viewport JPEG/PNG.
        const PIXEL_PNG: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08,
            0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x05, 0xFE,
            0xD4, 0xEF, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        BASE64.encode(PIXEL_PNG)
    }

    fn ok_frame(&self, message: impl Into<String>) -> ComputerActionResult {
        ComputerActionResult {
            status: ComputerActionStatus::Ok,
            message: Some(message.into()),
            screenshot_base64: Some(self.render_frame()),
            mime_type: Some("image/png".into()),
            viewport: Some(self.viewport.clone()),
            url: self.url.clone(),
            title: self.title.clone(),
            needs_approval: false,
            approval_id: None,
        }
    }
}

impl BrowserDriver for SyntheticBrowserSession {
    fn id(&self) -> &str {
        &self.id
    }

    fn viewport(&self) -> ComputerViewport {
        self.viewport.clone()
    }

    fn url(&self) -> Option<String> {
        self.url.clone()
    }

    fn title(&self) -> Option<String> {
        self.title.clone()
    }

    async fn execute(&mut self, action: &ComputerAction) -> Result<ComputerActionResult> {
        match action {
            ComputerAction::Screenshot => Ok(self.ok_frame("Captured browser viewport.")),
            ComputerAction::VisitUrl { url } => {
                if !(url.starts_with("http://")
                    || url.starts_with("https://")
                    || url == "about:blank")
                {
                    bail!("Only http(s) URLs are allowed.");
                }
                self.url = Some(url.clone());
                self.title = Some(url.clone());
                Ok(self.ok_frame(format!("Navigated to {url}")))
            }
            ComputerAction::WebSearch { query } => {
                let encoded = urlencoding_lite(query);
                let url = format!("https://duckduckgo.com/?q={encoded}");
                self.url = Some(url.clone());
                self.title = Some(format!("Search: {query}"));
                Ok(self.ok_frame(format!("Opened search for {query}")))
            }
            ComputerAction::LeftClick { x, y }
            | ComputerAction::RightClick { x, y }
            | ComputerAction::DoubleClick { x, y }
            | ComputerAction::TripleClick { x, y }
            | ComputerAction::MouseMove { x, y } => {
                Ok(self.ok_frame(format!("{} at ({x:.0}, {y:.0})", action.kind())))
            }
            ComputerAction::LeftClickDrag {
                start_x,
                start_y,
                end_x,
                end_y,
            } => Ok(self.ok_frame(format!(
                "Drag ({start_x:.0},{start_y:.0}) → ({end_x:.0},{end_y:.0})"
            ))),
            ComputerAction::Type { text } => Ok(self.ok_frame(format!("Typed {} chars", text.len()))),
            ComputerAction::Keypress { keys } => {
                Ok(self.ok_frame(format!("Pressed {}", keys.join("+"))))
            }
            ComputerAction::Scroll {
                x,
                y,
                delta_x,
                delta_y,
            } => Ok(self.ok_frame(format!(
                "Scroll at ({x:.0},{y:.0}) Δ({delta_x:.0},{delta_y:.0})"
            ))),
            ComputerAction::Wait { milliseconds } => {
                tokio::time::sleep(std::time::Duration::from_millis((*milliseconds).min(10_000)))
                    .await;
                Ok(self.ok_frame(format!("Waited {milliseconds}ms")))
            }
            ComputerAction::Memorize { fact } => {
                self.memories.push(fact.clone());
                Ok(self.ok_frame(format!("Memorized: {fact}")))
            }
            ComputerAction::AskUser { question } => Ok(ComputerActionResult {
                status: ComputerActionStatus::WaitingForUser,
                message: Some(question.clone()),
                screenshot_base64: Some(self.render_frame()),
                mime_type: Some("image/png".into()),
                viewport: Some(self.viewport.clone()),
                url: self.url.clone(),
                title: self.title.clone(),
                needs_approval: true,
                approval_id: None,
            }),
            ComputerAction::Terminate { response } => Ok(ComputerActionResult {
                status: ComputerActionStatus::Finished,
                message: response.clone().or_else(|| Some("Task finished.".into())),
                screenshot_base64: Some(self.render_frame()),
                mime_type: Some("image/png".into()),
                viewport: Some(self.viewport.clone()),
                url: self.url.clone(),
                title: self.title.clone(),
                needs_approval: false,
                approval_id: None,
            }),
        }
    }
}

fn urlencoding_lite(value: &str) -> String {
    let mut out = String::with_capacity(value.len() * 3);
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// Process-wide registry of live browser sessions.
#[derive(Default)]
pub struct BrowserSessionRegistry {
    sessions: Mutex<HashMap<String, SyntheticBrowserSession>>,
}

impl BrowserSessionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn open(&self, viewport: ComputerViewport) -> String {
        let session = SyntheticBrowserSession::new(viewport);
        let id = session.id.clone();
        self.sessions.lock().await.insert(id.clone(), session);
        id
    }

    pub async fn close(&self, id: &str) {
        self.sessions.lock().await.remove(id);
    }

    pub async fn snapshot(
        &self,
        id: &str,
    ) -> Result<(ComputerViewport, Option<String>, Option<String>)> {
        let sessions = self.sessions.lock().await;
        let session = sessions
            .get(id)
            .with_context(|| format!("unknown browser session {id}"))?;
        Ok((session.viewport(), session.url(), session.title()))
    }

    pub async fn execute(
        &self,
        id: &str,
        action: &ComputerAction,
    ) -> Result<ComputerActionResult> {
        let mut sessions = self.sessions.lock().await;
        let session = sessions
            .get_mut(id)
            .with_context(|| format!("unknown browser session {id}"))?;
        session.execute(action).await
    }

    pub async fn memories(&self, id: &str) -> Result<Vec<String>> {
        let sessions = self.sessions.lock().await;
        let session = sessions
            .get(id)
            .with_context(|| format!("unknown browser session {id}"))?;
        Ok(session.memories.clone())
    }
}

pub type SharedBrowserRegistry = Arc<BrowserSessionRegistry>;

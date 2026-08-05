//! Execution broker for Computer Use actions.
//!
//! The broker is the single writer for action records.  Renderer clients may
//! append conversational/model steps, but must not append a second tool step
//! after calling [`ComputerBroker::execute`].

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use anyhow::{Context, Result, bail};
use brazier_protocol::computer_types::{
    ComputerAction, ComputerActionResult, ComputerActionStatus, ComputerPermissionMode,
    ComputerSession, ComputerStep, ComputerTarget, ComputerViewport, OsPermissionStatus,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Notify};
use uuid::Uuid;

use crate::{
    computer_browser::{ActionCancel, BrowserSessionRegistry, SharedBrowserRegistry},
    computer_desktop::{self, desktop_permitted},
    computer_policy::{self, ComputerPolicyDecision, ComputerPolicyRequest},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputerExecRequest {
    pub session_id: String,
    pub action: ComputerAction,
    #[serde(default)]
    pub approval_id: Option<String>,
    /// Optional per-request settle delay, overriding the broker default. The
    /// renderer sends a short one for direct user input so the viewport
    /// responds almost immediately, then the live stream delivers the fully
    /// settled frame.
    #[serde(default)]
    pub settle_delay_ms: Option<u64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingComputerApproval {
    pub id: String,
    pub session_id: String,
    pub action: ComputerAction,
    pub created_at: String,
}

struct LiveSession {
    record: ComputerSession,
    browser_id: Option<String>,
    steps: Vec<ComputerStep>,
    pending: HashMap<String, PendingComputerApproval>,
    /// Serializes an individual browser/desktop session without blocking any
    /// other session while Chromium waits or performs input.
    action_gate: Arc<Mutex<()>>,
    /// Ephemeral, fail-closed authority established only after the desktop app
    /// has a visible native safety overlay and a working Esc watcher.
    desktop_authorized: bool,
    /// Esc/stop sets the flag and notifies waiters so in-flight Wait/settle
    /// delays abort instead of continuing after authority is revoked.
    cancel: Arc<ActionCancel>,
}

#[derive(Serialize, Deserialize)]
struct PersistedState {
    version: u32,
    sessions: Vec<PersistedSession>,
}
#[derive(Serialize, Deserialize)]
struct PersistedSession {
    record: ComputerSession,
    steps: Vec<ComputerStep>,
    pending: HashMap<String, PendingComputerApproval>,
}

pub struct ComputerBroker {
    browsers: SharedBrowserRegistry,
    sessions: Mutex<HashMap<String, LiveSession>>,
    /// Live browser frames keyed by session id. The relay keeps a sender here
    /// so every SSE subscriber shares one Chromium screencast per session.
    screencasts: Mutex<HashMap<String, tokio::sync::broadcast::Sender<String>>>,
    /// Serializes starting a screencast so concurrent subscribers cannot begin
    /// a second relay (double-acking) for the same browser.
    screencast_start: Mutex<()>,
    approvals_changed: Arc<Notify>,
    persist_path: Option<PathBuf>,
    /// Serializes durable snapshots. State is copied while this is held, then
    /// written after releasing the session mutex; snapshots cannot regress.
    persistence: Mutex<()>,
    action_settle_delay_ms: AtomicU64,
}

pub const DEFAULT_ACTION_SETTLE_DELAY_MS: u64 = 750;
pub const MAX_ACTION_SETTLE_DELAY_MS: u64 = 10_000;

impl ComputerBroker {
    /// Ephemeral broker intended for unit tests. Daemon code must use
    /// [`Self::open`] so task state survives restart.
    pub fn new() -> Self {
        Self::from_sessions(None, HashMap::new())
    }

    /// Restore computer sessions, steps, memories, and pending approvals from
    /// an atomically-written local file. Browser processes are intentionally
    /// not restored: a later action starts a fresh isolated Chromium process.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let sessions = match tokio::fs::read(&path).await {
            Ok(bytes) => {
                let state: PersistedState =
                    serde_json::from_slice(&bytes).context("decode computer session store")?;
                if state.version != 1 {
                    bail!(
                        "unsupported computer session store version {}",
                        state.version
                    );
                }
                state
                    .sessions
                    .into_iter()
                    .map(|mut item| {
                        // A process cannot still be executing after daemon restart.
                        item.record.running = false;
                        (
                            item.record.id.clone(),
                            LiveSession {
                                record: item.record,
                                browser_id: None,
                                steps: item.steps,
                                pending: item.pending,
                                action_gate: Arc::new(Mutex::new(())),
                                desktop_authorized: false,
                                cancel: Arc::new(ActionCancel::new()),
                            },
                        )
                    })
                    .collect()
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => HashMap::new(),
            Err(error) => return Err(error).context("read computer session store"),
        };
        Ok(Self::from_sessions(Some(path), sessions))
    }

    fn from_sessions(
        persist_path: Option<PathBuf>,
        sessions: HashMap<String, LiveSession>,
    ) -> Self {
        Self {
            browsers: Arc::new(BrowserSessionRegistry::new()),
            sessions: Mutex::new(sessions),
            screencasts: Mutex::new(HashMap::new()),
            screencast_start: Mutex::new(()),
            approvals_changed: Arc::new(Notify::new()),
            persist_path,
            persistence: Mutex::new(()),
            action_settle_delay_ms: AtomicU64::new(DEFAULT_ACTION_SETTLE_DELAY_MS),
        }
    }
    pub fn action_settle_delay_ms(&self) -> u64 {
        self.action_settle_delay_ms.load(Ordering::Relaxed)
    }

    pub fn set_action_settle_delay_ms(&self, milliseconds: u64) {
        self.action_settle_delay_ms.store(
            milliseconds.min(MAX_ACTION_SETTLE_DELAY_MS),
            Ordering::Relaxed,
        );
    }
    pub fn os_permissions(&self) -> OsPermissionStatus {
        computer_desktop::probe_os_permissions()
    }

    pub async fn request_os_permissions(&self) -> Result<OsPermissionStatus> {
        computer_desktop::request_os_permissions()
            .await
            .map_err(anyhow::Error::msg)
    }

    async fn persist(&self) -> Result<()> {
        let Some(path) = &self.persist_path else {
            return Ok(());
        };
        let _write = self.persistence.lock().await;
        let state = {
            let sessions = self.sessions.lock().await;
            PersistedState {
                version: 1,
                sessions: sessions
                    .values()
                    .map(|session| PersistedSession {
                        record: session.record.clone(),
                        steps: session.steps.clone(),
                        pending: session.pending.clone(),
                    })
                    .collect(),
            }
        };
        let bytes = serde_json::to_vec(&state).context("encode computer session store")?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .context("create computer session store directory")?;
        }
        let temporary = path.with_extension(format!("tmp-{}", Uuid::new_v4()));
        tokio::fs::write(&temporary, bytes)
            .await
            .context("write computer session store")?;
        tokio::fs::rename(&temporary, path)
            .await
            .context("commit computer session store")?;
        Ok(())
    }

    pub async fn list_sessions(&self) -> Vec<ComputerSession> {
        let sessions = self.sessions.lock().await;
        let mut out: Vec<_> = sessions.values().map(|s| s.record.clone()).collect();
        out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        out
    }
    pub async fn get_session(&self, id: &str) -> Result<ComputerSession> {
        self.sessions
            .lock()
            .await
            .get(id)
            .map(|s| s.record.clone())
            .with_context(|| format!("unknown computer session {id}"))
    }

    /// Update mutable session settings. The permission mode is changeable on a
    /// live session so the user can tighten or relax how future actions are
    /// judged without abandoning the current task; the target is not.
    pub async fn update_session(
        &self,
        id: &str,
        permission_mode: ComputerPermissionMode,
    ) -> Result<ComputerSession> {
        let record = {
            let mut sessions = self.sessions.lock().await;
            let session = sessions
                .get_mut(id)
                .with_context(|| format!("unknown computer session {id}"))?;
            session.record.permission_mode = permission_mode;
            session.record.updated_at = now_stamp();
            session.record.clone()
        };
        self.persist().await?;
        Ok(record)
    }

    pub async fn create_session(
        &self,
        title: Option<String>,
        target: ComputerTarget,
        model_id: Option<String>,
        permission_mode: ComputerPermissionMode,
        viewport: Option<ComputerViewport>,
    ) -> Result<ComputerSession> {
        let viewport = viewport.unwrap_or_default();
        let browser_id = if target == ComputerTarget::Browser {
            Some(self.browsers.open(viewport.clone()).await?)
        } else {
            None
        };
        let (url, title_page) = match &browser_id {
            Some(id) => {
                let (_, url, page) = self.browsers.snapshot(id).await?;
                (url, page)
            }
            None => (None, None),
        };
        let now = now_stamp();
        let id = Uuid::new_v4().to_string();
        let record = ComputerSession {
            id: id.clone(),
            title: title.unwrap_or_else(|| "Computer task".into()),
            target,
            model_id,
            permission_mode,
            viewport,
            created_at: now.clone(),
            updated_at: now,
            url,
            title_page,
            running: false,
            memories: Vec::new(),
        };
        self.sessions.lock().await.insert(
            id,
            LiveSession {
                record: record.clone(),
                browser_id,
                steps: Vec::new(),
                pending: HashMap::new(),
                action_gate: Arc::new(Mutex::new(())),
                desktop_authorized: false,
                cancel: Arc::new(ActionCancel::new()),
            },
        );
        self.persist().await?;
        Ok(record)
    }

    pub async fn delete_session(&self, id: &str) -> Result<()> {
        let session = self
            .sessions
            .lock()
            .await
            .remove(id)
            .with_context(|| format!("unknown computer session {id}"))?;
        if let Some(browser_id) = session.browser_id {
            self.browsers.close(&browser_id).await;
        }
        self.screencasts.lock().await.remove(id);
        self.persist().await
    }

    /// Subscribe to a browser session's live screencast frames (base64 JPEG).
    /// Starts Chromium's screencast on first use and fans every subscriber out
    /// from the single relay, so the daemon pushes one stream per session.
    pub async fn subscribe_screencast(
        &self,
        session_id: &str,
    ) -> Result<tokio::sync::broadcast::Receiver<String>> {
        let _start = self.screencast_start.lock().await;
        if let Some(frames) = self.screencasts.lock().await.get(session_id) {
            return Ok(frames.subscribe());
        }
        let browser_id = self.ensure_browser(session_id).await?;
        let (frames, _) = tokio::sync::broadcast::channel(16);
        self.browsers
            .start_screencast(&browser_id, frames.clone())
            .await?;
        self.screencasts
            .lock()
            .await
            .insert(session_id.to_owned(), frames.clone());
        Ok(frames.subscribe())
    }
    pub async fn list_steps(&self, id: &str) -> Result<Vec<ComputerStep>> {
        Ok(self
            .sessions
            .lock()
            .await
            .get(id)
            .with_context(|| format!("unknown computer session {id}"))?
            .steps
            .clone())
    }

    /// Only narrative/user/model steps should use this. Tool results are
    /// recorded by `execute`, including denied and refused actions.
    pub async fn append_step(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        thought: Option<String>,
        action: Option<ComputerAction>,
        result: Option<ComputerActionResult>,
    ) -> Result<ComputerStep> {
        if role == "tool" || action.is_some() || result.is_some() {
            bail!(
                "tool/action steps are recorded by the computer broker; append only narrative steps"
            );
        }
        let step = ComputerStep {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_owned(),
            role: role.to_owned(),
            content: content.to_owned(),
            thought,
            action: None,
            result: None,
            created_at: now_stamp(),
        };
        {
            let mut sessions = self.sessions.lock().await;
            let session = sessions
                .get_mut(session_id)
                .with_context(|| format!("unknown computer session {session_id}"))?;
            session.record.updated_at = step.created_at.clone();
            session.steps.push(step.clone());
        }
        self.persist().await?;
        Ok(step)
    }

    async fn action_gate(&self, session_id: &str) -> Result<Arc<Mutex<()>>> {
        self.sessions
            .lock()
            .await
            .get(session_id)
            .map(|s| Arc::clone(&s.action_gate))
            .with_context(|| format!("unknown computer session {session_id}"))
    }
    async fn ensure_browser(&self, session_id: &str) -> Result<String> {
        if let Some(id) = self
            .sessions
            .lock()
            .await
            .get(session_id)
            .and_then(|s| s.browser_id.clone())
        {
            return Ok(id);
        }
        let viewport = self
            .sessions
            .lock()
            .await
            .get(session_id)
            .with_context(|| format!("unknown computer session {session_id}"))?
            .record
            .viewport
            .clone();
        let id = self.browsers.open(viewport).await?;
        let mut sessions = self.sessions.lock().await;
        let session = sessions
            .get_mut(session_id)
            .context("session disappeared while starting browser")?;
        session.browser_id = Some(id.clone());
        Ok(id)
    }

    fn refusal(reason: impl Into<String>) -> ComputerActionResult {
        ComputerActionResult {
            status: ComputerActionStatus::Refused,
            message: Some(reason.into()),
            screenshot_base64: None,
            mime_type: None,
            viewport: None,
            url: None,
            title: None,
            needs_approval: false,
            approval_id: None,
        }
    }
    async fn record(
        &self,
        session_id: &str,
        action: ComputerAction,
        result: ComputerActionResult,
    ) -> Result<()> {
        {
            let mut sessions = self.sessions.lock().await;
            let session = sessions
                .get_mut(session_id)
                .with_context(|| format!("unknown computer session {session_id}"))?;
            if let Some(url) = &result.url {
                session.record.url = Some(url.clone());
            }
            if let Some(title) = &result.title {
                session.record.title_page = Some(title.clone());
            }
            if let Some(viewport) = &result.viewport {
                session.record.viewport = viewport.clone();
            }
            if let ComputerAction::Memorize { fact } = &action
                && !session.record.memories.contains(fact)
            {
                session.record.memories.push(fact.clone());
            }
            session.record.running = false;
            session.record.updated_at = now_stamp();
            session.steps.push(ComputerStep {
                id: Uuid::new_v4().to_string(),
                session_id: session_id.to_owned(),
                role: "tool".into(),
                content: result
                    .message
                    .clone()
                    .unwrap_or_else(|| action.kind().into()),
                thought: None,
                action: Some(action),
                result: Some(result),
                created_at: session.record.updated_at.clone(),
            });
        }
        self.persist().await
    }

    pub async fn execute(&self, request: ComputerExecRequest) -> Result<ComputerActionResult> {
        let gate = self.action_gate(&request.session_id).await?;
        let _gate = gate.lock().await;
        let (target, mode, viewport, desktop_authorized) = {
            let sessions = self.sessions.lock().await;
            let session = sessions
                .get(&request.session_id)
                .with_context(|| format!("unknown computer session {}", request.session_id))?;
            (
                session.record.target,
                session.record.permission_mode,
                session.record.viewport.clone(),
                session.desktop_authorized,
            )
        };
        let needs_desktop_authority = target == ComputerTarget::Desktop
            && !matches!(
                &request.action,
                ComputerAction::Memorize { .. }
                    | ComputerAction::AskUser { .. }
                    | ComputerAction::Terminate { .. }
            );
        if needs_desktop_authority && !desktop_authorized {
            let result = Self::refusal(
                "Desktop control is locked because the always-visible safety overlay and Esc emergency stop are not active.",
            );
            self.record(&request.session_id, request.action, result.clone())
                .await?;
            return Ok(result);
        }
        let decision = computer_policy::decide(&ComputerPolicyRequest {
            target,
            mode,
            action: &request.action,
            desktop_permitted: desktop_permitted(&self.os_permissions()),
        });
        let mut consumed_approval_id = None;
        match decision {
            ComputerPolicyDecision::Refuse(reason) => {
                let detail = if target == ComputerTarget::Desktop
                    && reason == "Desktop capture or input permission is missing."
                {
                    self.os_permissions()
                        .detail
                        .unwrap_or_else(|| reason.into())
                } else {
                    reason.into()
                };
                let result = Self::refusal(detail);
                self.record(&request.session_id, request.action, result.clone())
                    .await?;
                return Ok(result);
            }
            ComputerPolicyDecision::Ask if request.approval_id.is_none() => {
                let approval_id = Uuid::new_v4().to_string();
                let pending = PendingComputerApproval {
                    id: approval_id.clone(),
                    session_id: request.session_id.clone(),
                    action: request.action.clone(),
                    created_at: now_stamp(),
                };
                let result = {
                    let mut sessions = self.sessions.lock().await;
                    let session = sessions
                        .get_mut(&request.session_id)
                        .context("session disappeared")?;
                    session.pending.insert(approval_id.clone(), pending);
                    ComputerActionResult {
                        status: ComputerActionStatus::NeedsApproval,
                        message: Some(format!("Approval required for {}", request.action.kind())),
                        screenshot_base64: None,
                        mime_type: None,
                        viewport: Some(session.record.viewport.clone()),
                        url: session.record.url.clone(),
                        title: session.record.title_page.clone(),
                        needs_approval: true,
                        approval_id: Some(approval_id),
                    }
                };
                // The pending request is itself an authoritative tool record:
                // after a daemon/UI restart the renderer can recover its
                // approval id from this durable `needs_approval` step.
                self.record(&request.session_id, request.action, result.clone())
                    .await?;
                return Ok(result);
            }
            ComputerPolicyDecision::Ask => {
                let mut sessions = self.sessions.lock().await;
                let session = sessions
                    .get_mut(&request.session_id)
                    .context("session disappeared")?;
                let approval_id = request.approval_id.as_deref().unwrap_or("");
                let Some(pending) = session.pending.get(approval_id).cloned() else {
                    bail!("approval not found or already spent");
                };
                if pending.action != request.action {
                    bail!("approval does not match action");
                }
                session.pending.remove(approval_id);
                consumed_approval_id = Some(approval_id.to_owned());
            }
            ComputerPolicyDecision::Allow => {}
        }
        {
            let mut sessions = self.sessions.lock().await;
            sessions
                .get_mut(&request.session_id)
                .context("session disappeared")?
                .record
                .running = true;
        }
        self.persist().await?;
        // The durable trajectory retains Fara's model-space coordinates. The
        // final conversion is intentionally here, after a screenshot has
        // updated the session viewport and immediately before an OS/browser
        // driver receives the action.
        let driver_action = request.action.scaled_for_viewport(&viewport);
        let cancel = {
            let sessions = self.sessions.lock().await;
            let session = sessions
                .get(&request.session_id)
                .context("session disappeared")?;
            session.cancel.reset();
            session.cancel.clone()
        };
        let executed = match &request.action {
            ComputerAction::Memorize { fact } => Ok(ComputerActionResult {
                status: ComputerActionStatus::Ok,
                message: Some(format!("Memorized: {fact}")),
                screenshot_base64: None,
                mime_type: None,
                viewport: None,
                url: None,
                title: None,
                needs_approval: false,
                approval_id: None,
            }),
            ComputerAction::AskUser { question } => Ok(ComputerActionResult {
                status: ComputerActionStatus::WaitingForUser,
                message: Some(question.clone()),
                screenshot_base64: None,
                mime_type: None,
                viewport: None,
                url: None,
                title: None,
                needs_approval: false,
                approval_id: None,
            }),
            ComputerAction::Terminate { response } => Ok(ComputerActionResult {
                status: ComputerActionStatus::Finished,
                message: response.clone().or_else(|| Some("Task finished.".into())),
                screenshot_base64: None,
                mime_type: None,
                viewport: None,
                url: None,
                title: None,
                needs_approval: false,
                approval_id: None,
            }),
            _ => match target {
                ComputerTarget::Browser => match self.ensure_browser(&request.session_id).await {
                    Ok(id) => {
                        self.browsers
                            .execute(
                                &id,
                                &driver_action,
                                request.settle_delay_ms.unwrap_or_else(|| {
                                    self.action_settle_delay_ms()
                                }),
                                Some(cancel.as_ref()),
                            )
                            .await
                    }
                    Err(error) => Err(error),
                },
                ComputerTarget::Desktop => Ok(computer_desktop::execute_desktop_action(
                    &driver_action,
                    &viewport,
                    request
                        .settle_delay_ms
                        .unwrap_or_else(|| self.action_settle_delay_ms()),
                    Some(cancel.as_ref()),
                )
                .await),
            },
        };
        match executed {
            Ok(mut result) => {
                result.approval_id = consumed_approval_id.clone();
                self.record(&request.session_id, request.action, result.clone())
                    .await?;
                Ok(result)
            }
            Err(error) => {
                let result = ComputerActionResult {
                    status: ComputerActionStatus::Error,
                    message: Some(format!("{error}; no successful action was reported.")),
                    screenshot_base64: None,
                    mime_type: None,
                    viewport: None,
                    url: None,
                    title: None,
                    needs_approval: false,
                    approval_id: consumed_approval_id,
                };
                self.record(&request.session_id, request.action, result.clone())
                    .await?;
                Ok(result)
            }
        }
    }

    pub async fn decide_approval(
        &self,
        approval_id: &str,
        approve: bool,
    ) -> Result<Option<ComputerActionResult>> {
        let (session_id, action) = {
            let sessions = self.sessions.lock().await;
            sessions
                .values()
                .find_map(|session| {
                    session
                        .pending
                        .get(approval_id)
                        .map(|pending| (pending.session_id.clone(), pending.action.clone()))
                })
                .context("unknown approval")?
        };
        if !approve {
            {
                let mut sessions = self.sessions.lock().await;
                let session = sessions
                    .get_mut(&session_id)
                    .context("session disappeared")?;
                session.pending.remove(approval_id);
            }
            let mut result = Self::refusal("User denied the action.");
            result.approval_id = Some(approval_id.into());
            self.record(&session_id, action, result.clone()).await?;
            self.approvals_changed.notify_waiters();
            return Ok(Some(result));
        }
        let result = self
            .execute(ComputerExecRequest {
                session_id,
                action,
                approval_id: Some(approval_id.into()),
                settle_delay_ms: None,
            })
            .await?;
        self.approvals_changed.notify_waiters();
        Ok(Some(result))
    }
    pub async fn screenshot(&self, session_id: &str) -> Result<ComputerActionResult> {
        self.execute(ComputerExecRequest {
            session_id: session_id.into(),
            action: ComputerAction::Screenshot,
            approval_id: None,
            settle_delay_ms: None,
        })
        .await
    }

    /// Capture the current browser viewport without appending a durable step.
    ///
    /// The renderer polls this while a browser session is idle so the page
    /// updates live under the user's cursor (hover menus, form typing, streamed
    /// content) rather than only after an action. Unlike `screenshot`, nothing
    /// is recorded: live previews must not leak into the model's trajectory or
    /// flood the step log.
    pub async fn live_screenshot(&self, session_id: &str) -> Result<ComputerActionResult> {
        let gate = self.action_gate(session_id).await?;
        let _gate = gate.lock().await;
        let target = self
            .sessions
            .lock()
            .await
            .get(session_id)
            .with_context(|| format!("unknown computer session {session_id}"))?
            .record
            .target;
        if target != ComputerTarget::Browser {
            bail!("live preview is only available for browser sessions");
        }
        let id = self.ensure_browser(session_id).await?;
        let cancel = {
            let sessions = self.sessions.lock().await;
            let session = sessions
                .get(session_id)
                .context("session disappeared")?;
            session.cancel.reset();
            session.cancel.clone()
        };
        self.browsers
            .execute(&id, &ComputerAction::Screenshot, 0, Some(cancel.as_ref()))
            .await
    }

    /// Revoke host-desktop authority immediately. The renderer calls this for
    /// its global Escape hatch in addition to aborting the model request.
    pub async fn stop(&self, session_id: &str) -> Result<()> {
        let target = {
            let mut sessions = self.sessions.lock().await;
            let session = sessions
                .get_mut(session_id)
                .with_context(|| format!("unknown computer session {session_id}"))?;
            session.record.running = false;
            session.desktop_authorized = false;
            session.cancel.trip();
            session.record.target
        };
        #[cfg(target_os = "linux")]
        if target == ComputerTarget::Desktop && std::env::var_os("WAYLAND_DISPLAY").is_some() {
            crate::computer_portal::close_session().await;
            crate::computer_portal::clear_restore_token();
        }
        self.persist().await
    }

    /// Revoke desktop authority on every session. Used by the Electron main
    /// process Esc hatch so a wedged renderer cannot leave injection live.
    pub async fn revoke_all_desktop_authority(&self) -> Result<()> {
        let mut had_desktop = false;
        {
            let mut sessions = self.sessions.lock().await;
            for session in sessions.values_mut() {
                if session.record.target == ComputerTarget::Desktop || session.desktop_authorized {
                    session.desktop_authorized = false;
                    session.record.running = false;
                    session.cancel.trip();
                    had_desktop = true;
                }
            }
        }
        #[cfg(target_os = "linux")]
        if had_desktop && std::env::var_os("WAYLAND_DISPLAY").is_some() {
            crate::computer_portal::close_session().await;
            crate::computer_portal::clear_restore_token();
        }
        let _ = had_desktop;
        self.persist().await
    }

    /// Gate every host desktop action behind the independently established
    /// native safety indicator. This lease is deliberately never persisted.
    pub async fn set_desktop_authority(&self, session_id: &str, authorized: bool) -> Result<()> {
        let mut sessions = self.sessions.lock().await;
        let session = sessions
            .get_mut(session_id)
            .with_context(|| format!("unknown computer session {session_id}"))?;
        if authorized && session.record.target != ComputerTarget::Desktop {
            bail!("desktop safety authority can only be granted to a desktop session");
        }
        session.desktop_authorized = authorized;
        if !authorized {
            session.record.running = false;
        }
        Ok(())
    }
}

/// Marker file written by the desktop main process only after the native
/// overlay and Esc hatch report READY. Remote API clients cannot create this
/// file over HTTP, so granting desktop authority requires a live local safety
/// surface — not merely a bearer token.
pub fn safety_overlay_marker_path(data_dir: &Path) -> PathBuf {
    data_dir.join("computer-safety-overlay.ready")
}

pub fn safety_overlay_is_ready(data_dir: &Path) -> bool {
    safety_overlay_marker_path(data_dir).is_file()
}

pub fn write_safety_overlay_marker(data_dir: &Path) -> Result<()> {
    let path = safety_overlay_marker_path(data_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("create safety overlay marker directory {}", parent.display())
        })?;
    }
    std::fs::write(&path, b"ready\n")
        .with_context(|| format!("write safety overlay marker {}", path.display()))
}

pub fn clear_safety_overlay_marker(data_dir: &Path) {
    let path = safety_overlay_marker_path(data_dir);
    let _ = std::fs::remove_file(path);
}
impl Default for ComputerBroker {
    fn default() -> Self {
        Self::new()
    }
}
fn now_stamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_settle_delay_defaults_and_is_bounded() {
        let broker = ComputerBroker::new();
        assert_eq!(broker.action_settle_delay_ms(), 750);
        broker.set_action_settle_delay_ms(1_250);
        assert_eq!(broker.action_settle_delay_ms(), 1_250);
        broker.set_action_settle_delay_ms(MAX_ACTION_SETTLE_DELAY_MS + 1);
        assert_eq!(broker.action_settle_delay_ms(), MAX_ACTION_SETTLE_DELAY_MS);
    }

    #[tokio::test]
    async fn permission_mode_is_changeable_on_a_live_session() {
        let broker = ComputerBroker::new();
        let session = broker
            .create_session(
                None,
                ComputerTarget::Desktop,
                None,
                ComputerPermissionMode::Ask,
                None,
            )
            .await
            .unwrap();
        let updated = broker
            .update_session(&session.id, ComputerPermissionMode::BrowserOnly)
            .await
            .unwrap();
        assert_eq!(updated.permission_mode, ComputerPermissionMode::BrowserOnly);
        assert_eq!(
            broker
                .get_session(&session.id)
                .await
                .unwrap()
                .permission_mode,
            ComputerPermissionMode::BrowserOnly
        );
    }

    #[tokio::test]
    async fn persistent_session_keeps_steps_memories_and_pending_approvals() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("computer.json");
        let broker = ComputerBroker::open(&path).await.unwrap();
        let session = broker
            .create_session(
                None,
                ComputerTarget::Desktop,
                None,
                ComputerPermissionMode::Ask,
                None,
            )
            .await
            .unwrap();
        broker
            .execute(ComputerExecRequest {
                session_id: session.id.clone(),
                action: ComputerAction::Memorize {
                    fact: "persist me".into(),
                },
                approval_id: None,
                settle_delay_ms: None,
            })
            .await
            .unwrap();
        let approval_id = Uuid::new_v4().to_string();
        {
            let mut sessions = broker.sessions.lock().await;
            sessions.get_mut(&session.id).unwrap().pending.insert(
                approval_id.clone(),
                PendingComputerApproval {
                    id: approval_id.clone(),
                    session_id: session.id.clone(),
                    action: ComputerAction::LeftClick { x: 1.0, y: 1.0 },
                    created_at: now_stamp(),
                },
            );
        }
        broker.persist().await.unwrap();
        drop(broker);
        let restored = ComputerBroker::open(&path).await.unwrap();
        let restored_session = restored.get_session(&session.id).await.unwrap();
        assert!(!restored_session.running);
        assert_eq!(restored_session.memories, vec!["persist me"]);
        assert_eq!(restored.list_steps(&session.id).await.unwrap().len(), 1);
        assert!(
            restored
                .decide_approval(&approval_id, false)
                .await
                .unwrap()
                .is_some()
        );
    }

    #[tokio::test]
    async fn desktop_authority_is_ephemeral_and_stop_revokes_it() {
        let broker = ComputerBroker::new();
        let session = broker
            .create_session(
                None,
                ComputerTarget::Desktop,
                None,
                ComputerPermissionMode::AllowAll,
                None,
            )
            .await
            .unwrap();

        let refused = broker.screenshot(&session.id).await.unwrap();
        assert_eq!(refused.status, ComputerActionStatus::Refused);
        assert!(
            refused
                .message
                .as_deref()
                .unwrap_or_default()
                .contains("safety overlay")
        );

        broker
            .set_desktop_authority(&session.id, true)
            .await
            .unwrap();
        assert!(broker.sessions.lock().await[&session.id].desktop_authorized);
        broker.stop(&session.id).await.unwrap();
        assert!(!broker.sessions.lock().await[&session.id].desktop_authorized);
    }

    #[tokio::test]
    async fn revoke_all_clears_every_desktop_session() {
        let broker = ComputerBroker::new();
        let first = broker
            .create_session(
                None,
                ComputerTarget::Desktop,
                None,
                ComputerPermissionMode::AllowAll,
                None,
            )
            .await
            .unwrap();
        let second = broker
            .create_session(
                None,
                ComputerTarget::Desktop,
                None,
                ComputerPermissionMode::AllowAll,
                None,
            )
            .await
            .unwrap();
        broker
            .set_desktop_authority(&first.id, true)
            .await
            .unwrap();
        broker
            .set_desktop_authority(&second.id, true)
            .await
            .unwrap();
        broker.revoke_all_desktop_authority().await.unwrap();
        let sessions = broker.sessions.lock().await;
        assert!(!sessions[&first.id].desktop_authorized);
        assert!(!sessions[&second.id].desktop_authorized);
    }
    #[tokio::test]
    async fn refused_actions_are_recorded_once() {
        let broker = ComputerBroker::new();
        let session = broker
            .create_session(
                None,
                ComputerTarget::Desktop,
                None,
                ComputerPermissionMode::BrowserOnly,
                None,
            )
            .await
            .unwrap();
        let result = broker
            .execute(ComputerExecRequest {
                session_id: session.id.clone(),
                action: ComputerAction::LeftClick { x: 1.0, y: 1.0 },
                approval_id: None,
                settle_delay_ms: None,
            })
            .await
            .unwrap();
        assert_eq!(result.status, ComputerActionStatus::Refused);
        assert_eq!(broker.list_steps(&session.id).await.unwrap().len(), 1);
    }

    #[tokio::test]
    #[ignore = "requires a working Chromium and local TCP loopback"]
    async fn approved_result_references_the_consumed_approval() {
        if !crate::computer_browser::chromium_available() {
            return;
        }
        let broker = ComputerBroker::new();
        let session = broker
            .create_session(
                None,
                ComputerTarget::Browser,
                None,
                ComputerPermissionMode::Ask,
                None,
            )
            .await
            .unwrap();
        let pending = broker
            .execute(ComputerExecRequest {
                session_id: session.id,
                action: ComputerAction::LeftClick { x: 1.0, y: 1.0 },
                approval_id: None,
                settle_delay_ms: None,
            })
            .await
            .unwrap();
        let approval_id = pending.approval_id.unwrap();
        let approved = broker
            .decide_approval(&approval_id, true)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(approved.approval_id.as_deref(), Some(approval_id.as_str()));
    }
}

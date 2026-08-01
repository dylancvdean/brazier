//! Execution broker for Computer Use actions.
//!
//! The broker is the single writer for action records.  Renderer clients may
//! append conversational/model steps, but must not append a second tool step
//! after calling [`ComputerBroker::execute`].

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
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
    computer_browser::{BrowserSessionRegistry, SharedBrowserRegistry},
    computer_desktop::{self, desktop_permitted},
    computer_policy::{self, ComputerPolicyDecision, ComputerPolicyRequest},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputerExecRequest {
    pub session_id: String,
    pub action: ComputerAction,
    #[serde(default)]
    pub approval_id: Option<String>,
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
    approvals_changed: Arc<Notify>,
    persist_path: Option<PathBuf>,
    /// Serializes durable snapshots. State is copied while this is held, then
    /// written after releasing the session mutex; snapshots cannot regress.
    persistence: Mutex<()>,
}

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
            approvals_changed: Arc::new(Notify::new()),
            persist_path,
            persistence: Mutex::new(()),
        }
    }
    pub fn os_permissions(&self) -> OsPermissionStatus {
        computer_desktop::probe_os_permissions()
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
        self.persist().await
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
        let (target, mode) = {
            let sessions = self.sessions.lock().await;
            let session = sessions
                .get(&request.session_id)
                .with_context(|| format!("unknown computer session {}", request.session_id))?;
            (session.record.target, session.record.permission_mode)
        };
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
                    Ok(id) => self.browsers.execute(&id, &request.action).await,
                    Err(error) => Err(error),
                },
                ComputerTarget::Desktop => {
                    Ok(computer_desktop::execute_desktop_action(&request.action).await)
                }
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
        })
        .await
    }
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

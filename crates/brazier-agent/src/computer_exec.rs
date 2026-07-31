//! Execution broker for Computer Use actions.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
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
}

pub struct ComputerBroker {
    browsers: SharedBrowserRegistry,
    sessions: Mutex<HashMap<String, LiveSession>>,
    approvals_changed: Arc<Notify>,
}

impl ComputerBroker {
    pub fn new() -> Self {
        Self {
            browsers: Arc::new(BrowserSessionRegistry::new()),
            sessions: Mutex::new(HashMap::new()),
            approvals_changed: Arc::new(Notify::new()),
        }
    }

    pub fn os_permissions(&self) -> OsPermissionStatus {
        computer_desktop::probe_os_permissions()
    }

    pub async fn list_sessions(&self) -> Vec<ComputerSession> {
        let sessions = self.sessions.lock().await;
        let mut out: Vec<_> = sessions.values().map(|s| s.record.clone()).collect();
        out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        out
    }

    pub async fn get_session(&self, id: &str) -> Result<ComputerSession> {
        let sessions = self.sessions.lock().await;
        sessions
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
        let now = now_stamp();
        let id = Uuid::new_v4().to_string();
        let browser_id = if target == ComputerTarget::Browser {
            Some(self.browsers.open(viewport.clone()).await)
        } else {
            None
        };
        let (url, title_page) = if let Some(browser_id) = &browser_id {
            let (_vp, url, title) = self.browsers.snapshot(browser_id).await?;
            (url, title)
        } else {
            (None, None)
        };
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
            id.clone(),
            LiveSession {
                record: record.clone(),
                browser_id,
                steps: Vec::new(),
                pending: HashMap::new(),
            },
        );
        Ok(record)
    }

    pub async fn delete_session(&self, id: &str) -> Result<()> {
        let mut sessions = self.sessions.lock().await;
        let Some(session) = sessions.remove(id) else {
            bail!("unknown computer session {id}");
        };
        if let Some(browser_id) = session.browser_id {
            self.browsers.close(&browser_id).await;
        }
        Ok(())
    }

    pub async fn list_steps(&self, id: &str) -> Result<Vec<ComputerStep>> {
        let sessions = self.sessions.lock().await;
        let session = sessions
            .get(id)
            .with_context(|| format!("unknown computer session {id}"))?;
        Ok(session.steps.clone())
    }

    pub async fn append_step(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        thought: Option<String>,
        action: Option<ComputerAction>,
        result: Option<ComputerActionResult>,
    ) -> Result<ComputerStep> {
        let mut sessions = self.sessions.lock().await;
        let session = sessions
            .get_mut(session_id)
            .with_context(|| format!("unknown computer session {session_id}"))?;
        let step = ComputerStep {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_owned(),
            role: role.to_owned(),
            content: content.to_owned(),
            thought,
            action,
            result,
            created_at: now_stamp(),
        };
        session.steps.push(step.clone());
        session.record.updated_at = step.created_at.clone();
        Ok(step)
    }

    pub async fn execute(&self, request: ComputerExecRequest) -> Result<ComputerActionResult> {
        let (target, mode, browser_id) = {
            let sessions = self.sessions.lock().await;
            let session = sessions
                .get(&request.session_id)
                .with_context(|| format!("unknown computer session {}", request.session_id))?;
            (
                session.record.target,
                session.record.permission_mode,
                session.browser_id.clone(),
            )
        };

        let os = self.os_permissions();
        let decision = computer_policy::decide(&ComputerPolicyRequest {
            target,
            mode,
            action: &request.action,
            desktop_permitted: desktop_permitted(&os),
        });

        match decision {
            ComputerPolicyDecision::Refuse(reason) => {
                return Ok(ComputerActionResult {
                    status: ComputerActionStatus::Refused,
                    message: Some(reason.into()),
                    screenshot_base64: None,
                    mime_type: None,
                    viewport: None,
                    url: None,
                    title: None,
                    needs_approval: false,
                    approval_id: None,
                });
            }
            ComputerPolicyDecision::Ask => {
                if request.approval_id.is_none() {
                    let approval_id = Uuid::new_v4().to_string();
                    let pending = PendingComputerApproval {
                        id: approval_id.clone(),
                        session_id: request.session_id.clone(),
                        action: request.action.clone(),
                        created_at: now_stamp(),
                    };
                    let mut sessions = self.sessions.lock().await;
                    let session = sessions
                        .get_mut(&request.session_id)
                        .context("session disappeared")?;
                    session.pending.insert(approval_id.clone(), pending);
                    return Ok(ComputerActionResult {
                        status: ComputerActionStatus::NeedsApproval,
                        message: Some(format!(
                            "Approval required for {}",
                            request.action.kind()
                        )),
                        screenshot_base64: None,
                        mime_type: None,
                        viewport: Some(session.record.viewport.clone()),
                        url: session.record.url.clone(),
                        title: session.record.title_page.clone(),
                        needs_approval: true,
                        approval_id: Some(approval_id),
                    });
                }
                let mut sessions = self.sessions.lock().await;
                let session = sessions
                    .get_mut(&request.session_id)
                    .context("session disappeared")?;
                let Some(pending) = session.pending.remove(request.approval_id.as_deref().unwrap_or(""))
                else {
                    bail!("approval not found or already spent");
                };
                if pending.action != request.action {
                    bail!("approval does not match action");
                }
            }
            ComputerPolicyDecision::Allow => {}
        }

        let result = match target {
            ComputerTarget::Browser => {
                let browser_id = browser_id.context("browser session missing")?;
                self.browsers.execute(&browser_id, &request.action).await?
            }
            ComputerTarget::Desktop => {
                computer_desktop::execute_desktop_action(&request.action).await
            }
        };

        {
            let mut sessions = self.sessions.lock().await;
            if let Some(session) = sessions.get_mut(&request.session_id) {
                if let Some(url) = &result.url {
                    session.record.url = Some(url.clone());
                }
                if let Some(title) = &result.title {
                    session.record.title_page = Some(title.clone());
                }
                if let Some(viewport) = &result.viewport {
                    session.record.viewport = viewport.clone();
                }
                if let ComputerAction::Memorize { fact } = &request.action {
                    session.record.memories.push(fact.clone());
                }
                session.record.updated_at = now_stamp();
                let step = ComputerStep {
                    id: Uuid::new_v4().to_string(),
                    session_id: request.session_id.clone(),
                    role: "tool".into(),
                    content: result
                        .message
                        .clone()
                        .unwrap_or_else(|| request.action.kind().into()),
                    thought: None,
                    action: Some(request.action.clone()),
                    result: Some(result.clone()),
                    created_at: session.record.updated_at.clone(),
                };
                session.steps.push(step);
            }
        }

        Ok(result)
    }

    pub async fn decide_approval(
        &self,
        approval_id: &str,
        approve: bool,
    ) -> Result<Option<ComputerActionResult>> {
        let (session_id, action) = {
            let sessions = self.sessions.lock().await;
            let mut found = None;
            for session in sessions.values() {
                if let Some(pending) = session.pending.get(approval_id) {
                    found = Some((pending.session_id.clone(), pending.action.clone()));
                    break;
                }
            }
            found.context("unknown approval")?
        };
        if !approve {
            let mut sessions = self.sessions.lock().await;
            for session in sessions.values_mut() {
                session.pending.remove(approval_id);
            }
            self.approvals_changed.notify_waiters();
            return Ok(Some(ComputerActionResult {
                status: ComputerActionStatus::Refused,
                message: Some("User denied the action.".into()),
                screenshot_base64: None,
                mime_type: None,
                viewport: None,
                url: None,
                title: None,
                needs_approval: false,
                approval_id: Some(approval_id.into()),
            }));
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
    // RFC3339-ish UTC without pulling in chrono; good enough for session ordering.
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

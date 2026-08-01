//! Shared wire types for Computer Use mode.
//!
//! Models emit dialect-specific actions (Fara XML, OpenAI tools, …). Adapters
//! translate those into [`ComputerAction`], and drivers execute the normalized
//! form against a browser or desktop target.

use serde::{Deserialize, Serialize};

/// Where computer-use actions run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputerTarget {
    Browser,
    Desktop,
}

impl ComputerTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Browser => "browser",
            Self::Desktop => "desktop",
        }
    }
}

/// Permission mode for a computer-use session.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ComputerPermissionMode {
    /// Ask before navigate, type, click, and credentialed-site work.
    #[default]
    Ask,
    /// Only browser/sandbox targets; host desktop refused.
    BrowserOnly,
    /// Explicit opt-out of prompting for low-risk browser actions.
    SkipPermissions,
}

impl ComputerPermissionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::BrowserOnly => "browser-only",
            Self::SkipPermissions => "skip-permissions",
        }
    }
}

/// Viewport metadata so coordinate dialects can be scaled.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComputerViewport {
    pub width: u32,
    pub height: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_pixel_ratio: Option<f32>,
}

impl Default for ComputerViewport {
    fn default() -> Self {
        Self {
            // Fara1.5 is most commonly trained and evaluated at this viewport.
            width: 1440,
            height: 900,
            device_pixel_ratio: Some(1.0),
        }
    }
}

#[cfg(test)]
mod viewport_tests {
    use super::ComputerViewport;

    #[test]
    fn default_matches_fara_grounding_viewport() {
        let viewport = ComputerViewport::default();
        assert_eq!((viewport.width, viewport.height), (1440, 900));
        assert_eq!(viewport.device_pixel_ratio, Some(1.0));
    }
}

/// Normalized computer-use action set shared by every model adapter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ComputerAction {
    Screenshot,
    LeftClick {
        x: f64,
        y: f64,
    },
    RightClick {
        x: f64,
        y: f64,
    },
    DoubleClick {
        x: f64,
        y: f64,
    },
    TripleClick {
        x: f64,
        y: f64,
    },
    MouseMove {
        x: f64,
        y: f64,
    },
    LeftClickDrag {
        start_x: f64,
        start_y: f64,
        end_x: f64,
        end_y: f64,
    },
    Type {
        text: String,
    },
    Keypress {
        keys: Vec<String>,
    },
    Scroll {
        x: f64,
        y: f64,
        delta_x: f64,
        delta_y: f64,
    },
    Wait {
        #[serde(default = "default_wait_ms")]
        milliseconds: u64,
    },
    VisitUrl {
        url: String,
    },
    WebSearch {
        query: String,
    },
    Memorize {
        fact: String,
    },
    AskUser {
        question: String,
    },
    Terminate {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response: Option<String>,
    },
}

fn default_wait_ms() -> u64 {
    1000
}

impl ComputerAction {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Screenshot => "screenshot",
            Self::LeftClick { .. } => "left_click",
            Self::RightClick { .. } => "right_click",
            Self::DoubleClick { .. } => "double_click",
            Self::TripleClick { .. } => "triple_click",
            Self::MouseMove { .. } => "mouse_move",
            Self::LeftClickDrag { .. } => "left_click_drag",
            Self::Type { .. } => "type",
            Self::Keypress { .. } => "keypress",
            Self::Scroll { .. } => "scroll",
            Self::Wait { .. } => "wait",
            Self::VisitUrl { .. } => "visit_url",
            Self::WebSearch { .. } => "web_search",
            Self::Memorize { .. } => "memorize",
            Self::AskUser { .. } => "ask_user",
            Self::Terminate { .. } => "terminate",
        }
    }

    pub fn requires_approval(&self, mode: ComputerPermissionMode) -> bool {
        match mode {
            ComputerPermissionMode::SkipPermissions => matches!(
                self,
                Self::AskUser { .. }
                    | Self::Type { .. }
                    | Self::Keypress { .. }
                    | Self::VisitUrl { .. }
                    // Coordinates do not convey intent. A click can submit a
                    // purchase, consent, delete data, or send a message, so it
                    // is never safe to auto-approve based on its primitive.
                    | Self::LeftClick { .. }
                    | Self::RightClick { .. }
                    | Self::DoubleClick { .. }
                    | Self::TripleClick { .. }
                    | Self::LeftClickDrag { .. }
                    | Self::MouseMove { .. }
                    | Self::Scroll { .. }
                    | Self::WebSearch { .. }
            ),
            ComputerPermissionMode::BrowserOnly | ComputerPermissionMode::Ask => !matches!(
                self,
                Self::Screenshot
                    | Self::Wait { .. }
                    | Self::Memorize { .. }
                    | Self::Terminate { .. }
            ),
        }
    }

    /// Desktop target never accepts browser-only actions.
    pub fn allowed_on(&self, target: ComputerTarget) -> bool {
        !matches!(
            (target, self),
            (
                ComputerTarget::Desktop,
                Self::VisitUrl { .. } | Self::WebSearch { .. }
            )
        )
    }
}

/// Result of executing one normalized action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputerActionResult {
    pub status: ComputerActionStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screenshot_base64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub viewport: Option<ComputerViewport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default)]
    pub needs_approval: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputerActionStatus {
    Ok,
    NeedsApproval,
    Refused,
    Error,
    Finished,
    WaitingForUser,
}

/// OS permission probe status for desktop computer use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OsPermissionState {
    Granted,
    Missing,
    Unsupported,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsPermissionStatus {
    pub platform: String,
    pub display_server: String,
    pub screen_capture: OsPermissionState,
    pub input_injection: OsPermissionState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settings_hint: Option<String>,
}

/// In-memory / API-facing computer-use session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputerSession {
    pub id: String,
    pub title: String,
    pub target: ComputerTarget,
    pub model_id: Option<String>,
    pub permission_mode: ComputerPermissionMode,
    pub viewport: ComputerViewport,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title_page: Option<String>,
    #[serde(default)]
    pub running: bool,
    #[serde(default)]
    pub memories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputerStep {
    pub id: String,
    pub session_id: String,
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<ComputerAction>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<ComputerActionResult>,
    pub created_at: String,
}

/// Which workspace modes appear in the top bar.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceModesPreference {
    pub chat: bool,
    pub agent: bool,
    pub generate: bool,
    pub voice: bool,
    pub computer: bool,
}

impl Default for WorkspaceModesPreference {
    fn default() -> Self {
        Self {
            chat: true,
            agent: true,
            generate: true,
            voice: true,
            computer: false,
        }
    }
}

impl WorkspaceModesPreference {
    pub fn normalize(mut self) -> Self {
        if !self.chat && !self.agent && !self.generate && !self.voice && !self.computer {
            self.chat = true;
        }
        self
    }

    pub fn enabled_modes(&self) -> Vec<&'static str> {
        let mut modes = Vec::new();
        if self.chat {
            modes.push("chat");
        }
        if self.agent {
            modes.push("agent");
        }
        if self.generate {
            modes.push("generate");
        }
        if self.voice {
            modes.push("voice");
        }
        if self.computer {
            modes.push("computer");
        }
        modes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_modes_require_at_least_one() {
        let normalized = WorkspaceModesPreference {
            chat: false,
            agent: false,
            generate: false,
            voice: false,
            computer: false,
        }
        .normalize();
        assert!(normalized.chat);
    }

    #[test]
    fn browser_only_actions_rejected_on_desktop() {
        let action = ComputerAction::VisitUrl {
            url: "https://example.com".into(),
        };
        assert!(!action.allowed_on(ComputerTarget::Desktop));
        assert!(action.allowed_on(ComputerTarget::Browser));
    }
}

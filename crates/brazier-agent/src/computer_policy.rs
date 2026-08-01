//! Policy decisions for Computer Use actions.

use brazier_protocol::computer_types::{ComputerAction, ComputerPermissionMode, ComputerTarget};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputerPolicyDecision {
    Allow,
    Ask,
    Refuse(&'static str),
}

pub struct ComputerPolicyRequest<'a> {
    pub target: ComputerTarget,
    pub mode: ComputerPermissionMode,
    pub action: &'a ComputerAction,
    pub desktop_permitted: bool,
}

pub fn decide(request: &ComputerPolicyRequest<'_>) -> ComputerPolicyDecision {
    if !request.action.allowed_on(request.target) {
        return ComputerPolicyDecision::Refuse(
            "This action is only available for the browser target.",
        );
    }

    // These are broker-local control actions; they do not capture or inject
    // anything into the desktop and therefore do not depend on OS grants.
    if matches!(
        request.action,
        ComputerAction::Memorize { .. }
            | ComputerAction::Terminate { .. }
            | ComputerAction::AskUser { .. }
    ) {
        return ComputerPolicyDecision::Allow;
    }

    if request.target == ComputerTarget::Desktop {
        if request.mode == ComputerPermissionMode::BrowserOnly {
            return ComputerPolicyDecision::Refuse(
                "Desktop computer use is disabled in browser-only mode.",
            );
        }
        if !request.desktop_permitted {
            return ComputerPolicyDecision::Refuse(
                "Desktop capture or input permission is missing.",
            );
        }
    }

    if request.action.requires_approval(request.mode) {
        ComputerPolicyDecision::Ask
    } else {
        ComputerPolicyDecision::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screenshot_allowed_without_ask() {
        let action = ComputerAction::Screenshot;
        let decision = decide(&ComputerPolicyRequest {
            target: ComputerTarget::Browser,
            mode: ComputerPermissionMode::Ask,
            action: &action,
            desktop_permitted: false,
        });
        assert_eq!(decision, ComputerPolicyDecision::Allow);
    }

    #[test]
    fn click_asks_in_ask_mode() {
        let action = ComputerAction::LeftClick { x: 10.0, y: 10.0 };
        let decision = decide(&ComputerPolicyRequest {
            target: ComputerTarget::Browser,
            mode: ComputerPermissionMode::Ask,
            action: &action,
            desktop_permitted: false,
        });
        assert_eq!(decision, ComputerPolicyDecision::Ask);
    }

    #[test]
    fn skip_permissions_still_asks_for_ambiguous_interactive_actions() {
        for action in [
            ComputerAction::LeftClick { x: 10.0, y: 10.0 },
            ComputerAction::Scroll {
                x: 10.0,
                y: 10.0,
                delta_x: 0.0,
                delta_y: 500.0,
            },
            ComputerAction::WebSearch {
                query: "sensitive account".into(),
            },
        ] {
            let decision = decide(&ComputerPolicyRequest {
                target: ComputerTarget::Browser,
                mode: ComputerPermissionMode::SkipPermissions,
                action: &action,
                desktop_permitted: false,
            });
            assert_eq!(decision, ComputerPolicyDecision::Ask, "{}", action.kind());
        }
    }

    #[test]
    fn allow_all_does_not_pause_an_os_permitted_desktop_click() {
        let action = ComputerAction::LeftClick { x: 10.0, y: 10.0 };
        assert_eq!(
            decide(&ComputerPolicyRequest {
                target: ComputerTarget::Desktop,
                mode: ComputerPermissionMode::AllowAll,
                action: &action,
                desktop_permitted: true,
            }),
            ComputerPolicyDecision::Allow
        );
    }
}

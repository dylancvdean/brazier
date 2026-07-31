//! Policy decisions for Computer Use actions.

use brazier_protocol::computer_types::{
    ComputerAction, ComputerPermissionMode, ComputerTarget,
};

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

    if matches!(request.action, ComputerAction::AskUser { .. }) {
        return ComputerPolicyDecision::Ask;
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
}

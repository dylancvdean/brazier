//! A durable snapshot of where sensitive work is performed.
//!
//! This is deliberately a snapshot rather than a live connection reference:
//! approvals and results must continue to name the exact daemon after a
//! desktop reconnects or the daemon's display name later changes.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionLocationKind {
    Daemon,
}

/// Stable daemon identity and host characteristics captured with an action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionLocation {
    pub kind: ExecutionLocationKind,
    pub daemon_instance_id: String,
    pub daemon_display_name: String,
    pub platform: String,
    pub arch: String,
}

impl ExecutionLocation {
    pub fn daemon(instance_id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            kind: ExecutionLocationKind::Daemon,
            daemon_instance_id: instance_id.into(),
            daemon_display_name: display_name.into(),
            platform: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daemon_location_has_a_stable_wire_shape() {
        let value = serde_json::to_value(ExecutionLocation::daemon("daemon-1", "Studio Mac"))
            .expect("serialize execution location");
        assert_eq!(value["kind"], "daemon");
        assert_eq!(value["daemon_instance_id"], "daemon-1");
        assert_eq!(value["daemon_display_name"], "Studio Mac");
        assert_eq!(value["platform"], std::env::consts::OS);
        assert_eq!(value["arch"], std::env::consts::ARCH);
    }
}

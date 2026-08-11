//! Fail-closed browser Computer Use driver for unsupported platforms.
//!
//! Chromium's remote-debugging pipe uses inherited Unix file descriptors in
//! the current driver. Keep the public broker API available for portability,
//! but never advertise or emulate browser control where that transport has not
//! been implemented.

use std::sync::Arc;

use anyhow::{Result, bail};
use brazier_protocol::computer_types::{ComputerAction, ComputerActionResult, ComputerViewport};
use tokio::sync::{Notify, broadcast};

const DRIVER_UNAVAILABLE: &str =
    "Browser computer use is not implemented on this platform; no action was performed.";

/// Cooperative cancel signal shared with the desktop driver.
pub struct ActionCancel {
    cancelled: std::sync::atomic::AtomicBool,
    notify: Notify,
}

impl ActionCancel {
    pub fn new() -> Self {
        Self {
            cancelled: std::sync::atomic::AtomicBool::new(false),
            notify: Notify::new(),
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

/// Browser sessions are unavailable until this platform has a real CDP
/// transport. Every action fails instead of returning synthetic state.
pub struct BrowserSessionRegistry;

impl Default for BrowserSessionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl BrowserSessionRegistry {
    pub fn new() -> Self {
        Self
    }

    #[cfg(test)]
    pub fn with_executable(_executable: impl Into<String>) -> Self {
        Self
    }

    pub async fn open(&self, _viewport: ComputerViewport) -> Result<String> {
        bail!(DRIVER_UNAVAILABLE)
    }

    pub async fn close(&self, _id: &str) {}

    pub async fn snapshot(
        &self,
        _id: &str,
    ) -> Result<(ComputerViewport, Option<String>, Option<String>)> {
        bail!(DRIVER_UNAVAILABLE)
    }

    pub async fn execute(
        &self,
        _id: &str,
        _action: &ComputerAction,
        _settle_delay_ms: u64,
        _cancel: Option<&ActionCancel>,
    ) -> Result<ComputerActionResult> {
        bail!(DRIVER_UNAVAILABLE)
    }

    pub async fn start_screencast(
        &self,
        _id: &str,
        _frames: broadcast::Sender<String>,
    ) -> Result<()> {
        bail!(DRIVER_UNAVAILABLE)
    }
}

pub type SharedBrowserRegistry = Arc<BrowserSessionRegistry>;

/// Unsupported platforms must never advertise an available browser driver.
pub fn chromium_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unsupported_driver_is_fail_closed() {
        assert!(!chromium_available());
        let registry = BrowserSessionRegistry::new();
        let error = registry
            .open(ComputerViewport::default())
            .await
            .expect_err("unsupported driver must refuse browser sessions");
        assert!(error.to_string().contains("not implemented"));
    }
}

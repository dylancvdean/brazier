//! Minimal Linux input-device watcher for Brazier's emergency stop.
//!
//! This binary is intentionally separate from the desktop application and the
//! overlay helper. When installed root:input with mode 2755 it can read evdev
//! devices, but its only observable output is readiness and the fixed
//! Ctrl+Shift+Esc chord. It never reports individual keys or executes another
//! program.

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    linux::run()
}

#[cfg(not(target_os = "linux"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("the privileged input guard is available only on Linux")
}

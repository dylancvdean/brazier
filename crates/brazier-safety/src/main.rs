//! Native safety boundary for desktop Computer Use.
//!
//! The parent process must wait for `READY` before allowing any desktop
//! action. `ESC` means authority was revoked. Any other exit is also treated
//! as revocation by the parent.

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "linux")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    linux::run().await
}

#[cfg(not(target_os = "linux"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("the native safety helper is currently used only on Linux")
}

mod input_guard;
mod portal_shortcut;
mod wayland;
mod x11;

use std::io::Write as _;

use anyhow::{Context as _, Result, bail};

pub async fn run() -> Result<()> {
    let prepare = std::env::args().any(|argument| argument == "--prepare");
    if std::env::var_os("WAYLAND_DISPLAY").is_some()
        && std::env::var("XDG_SESSION_TYPE").as_deref() == Ok("wayland")
    {
        let emergency = match portal_shortcut::EscapeShortcut::open().await {
            Ok(shortcut) => EmergencyStop::Portal(shortcut),
            Err(portal_error) => match input_guard::InputGuard::open().await {
                Ok(guard) => {
                    eprintln!(
                        "Global shortcut portal unavailable ({portal_error:#}); using the privileged keyboard safety fallback."
                    );
                    EmergencyStop::InputGuard(Box::new(guard))
                }
                Err(guard_error) => bail!(
                    "the Wayland global shortcut could not be activated ({portal_error:#}), and the privileged keyboard safety fallback is unavailable ({guard_error:#}). Install it from Settings > Computer Use permissions"
                ),
            },
        };
        if prepare {
            ready()?;
            return Ok(());
        }
        let overlay = wayland::Overlay::open().context("create the Wayland safety overlay")?;
        ready()?;
        tokio::select! {
            result = emergency.wait() => result.context("watch the Wayland emergency shortcut")?,
            result = overlay.wait() => result.context("keep the Wayland safety overlay visible")?,
            _ = parent_closed() => bail!("safety parent closed its control pipe"),
        }
        escaped()?;
        return Ok(());
    }

    if std::env::var_os("DISPLAY").is_some() {
        if prepare {
            // XInput2 raw events need no standing user grant.
            ready()?;
            return Ok(());
        }
        return tokio::task::spawn_blocking(x11::run)
            .await
            .context("join X11 safety thread")?;
    }

    bail!("no Wayland or X11 desktop session is available")
}

enum EmergencyStop {
    Portal(portal_shortcut::EscapeShortcut),
    InputGuard(Box<input_guard::InputGuard>),
}

impl EmergencyStop {
    async fn wait(self) -> Result<()> {
        match self {
            Self::Portal(shortcut) => shortcut.wait().await,
            Self::InputGuard(guard) => guard.wait().await,
        }
    }
}

fn ready() -> Result<()> {
    println!("READY");
    std::io::stdout().flush().context("flush safety readiness")
}

fn escaped() -> Result<()> {
    println!("ESC");
    std::io::stdout().flush().context("flush Escape event")
}

async fn parent_closed() {
    use tokio::io::AsyncReadExt as _;
    let mut input = tokio::io::stdin();
    let mut byte = [0_u8; 1];
    loop {
        match input.read(&mut byte).await {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
    }
}

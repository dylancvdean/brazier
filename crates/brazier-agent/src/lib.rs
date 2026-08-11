//! Agent policy, isolation, worktree, and execution components.

#[cfg(feature = "execution")]
pub mod agent_exec;
pub mod agent_policy;
pub mod agent_sandbox;
#[cfg(windows)]
mod agent_sandbox_windows;
pub mod agent_tools;
pub mod agent_worktree;

#[cfg(unix)]
pub mod computer_browser;
#[cfg(not(unix))]
#[path = "computer_browser_unsupported.rs"]
pub mod computer_browser;
pub mod computer_desktop;
#[cfg(feature = "execution")]
pub mod computer_exec;
pub mod computer_fara;
pub mod computer_policy;
#[cfg(target_os = "linux")]
pub mod computer_portal;
#[cfg(target_os = "linux")]
pub mod computer_x11;

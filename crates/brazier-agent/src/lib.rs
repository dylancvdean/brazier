//! Agent policy, isolation, worktree, and execution components.

#[cfg(feature = "execution")]
pub mod agent_exec;
pub mod agent_policy;
pub mod agent_sandbox;
pub mod agent_tools;
pub mod agent_worktree;

pub mod computer_browser;
pub mod computer_desktop;
#[cfg(target_os = "linux")]
pub mod computer_portal;
#[cfg(target_os = "linux")]
pub mod computer_x11;
#[cfg(feature = "execution")]
pub mod computer_exec;
pub mod computer_fara;
pub mod computer_policy;

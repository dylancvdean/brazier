//! Shared generation job control for image/video backends.
//!
//! Both stable-diffusion.cpp (spawn-per-job CLI) and vLLM-Omni (HTTP server)
//! serialize GPU work through a single process-global lock and publish the same
//! [`ActiveGeneration`] snapshot for the UI and cancel path.

use std::{
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU32, Ordering as AtomicOrdering},
    },
    time::Instant,
};

use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex as AsyncMutex, Notify};

/// Image vs video for the active-job UI (backend-agnostic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Image,
    Video,
}

/// Returned when another generation job is already running.
#[derive(Debug)]
pub struct BusyError;

impl std::fmt::Display for BusyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "a generation job is already running; wait for it to finish or stop it"
        )
    }
}

impl std::error::Error for BusyError {}

/// Returned when the user stopped a generation from the interface.
///
/// A distinct type because this is not a failure: a model that asked for the
/// picture needs to hear that the person decided against it, not that the
/// engine broke.
#[derive(Debug)]
pub struct CancelledError;

impl std::fmt::Display for CancelledError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "generation was stopped by the user")
    }
}

impl std::error::Error for CancelledError {}

/// Who asked for a generation, so the interface can say whose prompt it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GenerationOrigin {
    /// Typed by the person, in Generate mode.
    #[default]
    User,
    /// Requested by a model through the generate tools.
    Model,
}

/// What a running generation is doing, for the interface to show and stop.
#[derive(Debug, Clone, Serialize)]
pub struct ActiveGeneration {
    pub id: String,
    pub modality: Modality,
    pub model_id: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    /// Blob the conditioning image came from, so the interface can show it.
    pub init_image_blob: Option<String>,
    pub origin: GenerationOrigin,
    /// How long it has been running, refreshed on every read.
    pub elapsed_secs: u64,
    /// When this job will be given up on, so a long render is not a mystery.
    pub timeout_secs: u64,
    /// Diffusion sampling progress when the backend reports it.
    pub current_step: u32,
    pub total_steps: u32,
}

/// Single-flight lock protecting the GPU: only one generation job may run at a time.
static JOB_LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();

pub fn job_lock() -> &'static AsyncMutex<()> {
    JOB_LOCK.get_or_init(|| AsyncMutex::new(()))
}

struct RunningJob {
    info: ActiveGeneration,
    started: Instant,
    cancel: Arc<AtomicBool>,
    notify: Arc<Notify>,
    current_step: Arc<AtomicU32>,
}

static RUNNING: OnceLock<Mutex<Option<RunningJob>>> = OnceLock::new();

fn running() -> &'static Mutex<Option<RunningJob>> {
    RUNNING.get_or_init(|| Mutex::new(None))
}

/// The generation in flight, if any, with its elapsed time brought up to date.
pub fn active_generation() -> Option<ActiveGeneration> {
    let guard = running().lock().expect("generation lock");
    guard.as_ref().map(|job| {
        let mut info = job.info.clone();
        info.elapsed_secs = job.started.elapsed().as_secs();
        info.current_step = job.current_step.load(AtomicOrdering::Relaxed);
        info
    })
}

/// Ask the running generation to stop. False when nothing is running.
pub fn cancel_active_generation() -> bool {
    let guard = running().lock().expect("generation lock");
    match guard.as_ref() {
        Some(job) => {
            job.cancel.store(true, AtomicOrdering::SeqCst);
            job.notify.notify_waiters();
            true
        }
        None => false,
    }
}

/// Registers a generation for the lifetime of the job and clears it on drop,
/// so a panic cannot leave the interface showing a job that is not running.
pub struct JobRegistration {
    pub cancel: Arc<AtomicBool>,
    pub notify: Arc<Notify>,
    pub current_step: Arc<AtomicU32>,
    pub total_steps: u32,
}

impl JobRegistration {
    pub fn open(info: ActiveGeneration) -> Self {
        let cancel = Arc::new(AtomicBool::new(false));
        let notify = Arc::new(Notify::new());
        let current_step = Arc::new(AtomicU32::new(info.current_step));
        let total_steps = info.total_steps;
        *running().lock().expect("generation lock") = Some(RunningJob {
            info,
            started: Instant::now(),
            cancel: Arc::clone(&cancel),
            notify: Arc::clone(&notify),
            current_step: Arc::clone(&current_step),
        });
        Self {
            cancel,
            notify,
            current_step,
            total_steps,
        }
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(AtomicOrdering::SeqCst)
    }

    /// Update progress when a backend reports step counts.
    pub fn set_step(&self, step: u32) {
        self.current_step.store(step, AtomicOrdering::Relaxed);
    }
}

impl Drop for JobRegistration {
    fn drop(&mut self) {
        *running().lock().expect("generation lock") = None;
    }
}

/// Acquire the single-flight generation lock or return [`BusyError`].
pub async fn try_acquire_job() -> Result<tokio::sync::MutexGuard<'static, ()>, BusyError> {
    job_lock().try_lock().map_err(|_| BusyError)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Process-global job state is shared across tests in this process.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn registration_surfaces_and_clears_active_job() {
        let _guard = TEST_LOCK.lock().expect("test lock");
        assert!(
            active_generation().is_none(),
            "another test left a generation running"
        );
        {
            let _job = JobRegistration::open(ActiveGeneration {
                id: "test".into(),
                modality: Modality::Image,
                model_id: "test:model".into(),
                prompt: "hi".into(),
                negative_prompt: None,
                init_image_blob: None,
                origin: GenerationOrigin::User,
                elapsed_secs: 0,
                timeout_secs: 60,
                current_step: 0,
                total_steps: 20,
            });
            let active = active_generation().expect("job registered");
            assert_eq!(active.model_id, "test:model");
            assert_eq!(active.total_steps, 20);
            assert!(cancel_active_generation());
        }
        assert!(active_generation().is_none());
        assert!(!cancel_active_generation());
    }
}

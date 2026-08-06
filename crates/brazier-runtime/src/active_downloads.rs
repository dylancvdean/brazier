//! In-flight model downloads that clients can stop.
//!
//! Stopping is cooperative: the downloader polls the flag between chunks. The
//! reason is recorded alongside it so the queue can tell a pause (keep the
//! partial file, stay in the list) from a cancel (give up on the job).

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, Ordering},
    },
};

fn lock<'a>(mutex: &'a Mutex<HashMap<String, Entry>>) -> std::sync::MutexGuard<'a, HashMap<String, Entry>> {
    mutex.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// Why a download was asked to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Pause,
    Cancel,
}

/// Shared stop signal polled by the downloader between chunks.
///
/// The reason travels with the flag because the two stops differ on disk: a
/// pause keeps the partial file so the transfer can pick up where it left off,
/// while a cancel discards it.
#[derive(Debug, Default)]
pub struct StopFlag(AtomicU8);

impl StopFlag {
    pub fn request(&self, reason: StopReason) {
        let requested = match reason {
            StopReason::Pause => 1,
            StopReason::Cancel => 2,
        };
        // Stop requests are monotonic. In particular, a pause arriving after
        // a cancel must not downgrade cancellation and cause the downloader to
        // retain a partial file the user asked it to discard.
        self.0.fetch_max(requested, Ordering::Relaxed);
    }

    /// The requested stop, if any.
    pub fn reason(&self) -> Option<StopReason> {
        match self.0.load(Ordering::Relaxed) {
            1 => Some(StopReason::Pause),
            2 => Some(StopReason::Cancel),
            _ => None,
        }
    }

    pub fn should_stop(&self) -> bool {
        self.reason().is_some()
    }
}

struct Entry {
    flag: Arc<StopFlag>,
}

#[derive(Default)]
pub struct ActiveDownloads {
    jobs: Mutex<HashMap<String, Entry>>,
}

impl ActiveDownloads {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, job_id: &str) -> Arc<StopFlag> {
        let flag = Arc::new(StopFlag::default());
        self.jobs
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(job_id.to_owned(), Entry { flag: flag.clone() });
        flag
    }

    /// Ask a running download to stop. Returns false when it is not running,
    /// in which case the caller updates the job row directly.
    pub fn stop(&self, job_id: &str, reason: StopReason) -> bool {
        let jobs = self
            .jobs
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(entry) = jobs.get(job_id) {
            entry.flag.request(reason);
            true
        } else {
            false
        }
    }

    pub fn cancel(&self, job_id: &str) -> bool {
        self.stop(job_id, StopReason::Cancel)
    }

    pub fn stop_reason(&self, job_id: &str) -> Option<StopReason> {
        self.jobs
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(job_id)
            .and_then(|entry| entry.flag.reason())
    }

    pub fn is_running(&self, job_id: &str) -> bool {
        self.jobs
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .contains_key(job_id)
    }

    pub fn finish(&self, job_id: &str) {
        self.jobs
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .remove(job_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pause_is_distinguishable_from_a_cancel() {
        let active = ActiveDownloads::new();
        let flag = active.register("job-1");
        assert!(!flag.should_stop());
        assert_eq!(active.stop_reason("job-1"), None);

        assert!(active.stop("job-1", StopReason::Pause));
        assert!(flag.should_stop(), "downloader sees the stop");
        assert_eq!(flag.reason(), Some(StopReason::Pause));
        assert_eq!(active.stop_reason("job-1"), Some(StopReason::Pause));

        active.finish("job-1");
        assert!(!active.is_running("job-1"));
        assert!(
            !active.stop("job-1", StopReason::Cancel),
            "no longer running"
        );
    }

    #[test]
    fn cancellation_cannot_be_downgraded_to_a_pause() {
        let flag = StopFlag::default();
        flag.request(StopReason::Cancel);
        flag.request(StopReason::Pause);
        assert_eq!(flag.reason(), Some(StopReason::Cancel));
    }
}

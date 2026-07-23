//! In-flight model downloads (streamed or queued) that clients can cancel.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Default)]
pub struct ActiveDownloads {
    jobs: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

impl ActiveDownloads {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, job_id: &str) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        self.jobs
            .lock()
            .expect("active downloads lock")
            .insert(job_id.to_owned(), flag.clone());
        flag
    }

    pub fn cancel(&self, job_id: &str) -> bool {
        let jobs = self.jobs.lock().expect("active downloads lock");
        if let Some(flag) = jobs.get(job_id) {
            flag.store(true, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    pub fn finish(&self, job_id: &str) {
        self.jobs
            .lock()
            .expect("active downloads lock")
            .remove(job_id);
    }
}

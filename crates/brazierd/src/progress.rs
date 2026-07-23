//! Shared progress events for long-running download / install jobs.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ProgressEvent {
    pub phase: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
}

impl ProgressEvent {
    pub fn phase(phase: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            phase: phase.into(),
            bytes: None,
            total: None,
            percent: None,
            message: Some(message.into()),
            done: None,
            error: None,
            result: None,
        }
    }

    pub fn download(bytes: u64, total: Option<u64>) -> Self {
        let percent = total
            .filter(|t| *t > 0)
            .map(|t| ((bytes as f64 / t as f64) * 100.0).clamp(0.0, 100.0));
        Self {
            phase: "download".into(),
            bytes: Some(bytes),
            total,
            percent,
            message: None,
            done: None,
            error: None,
            result: None,
        }
    }

    pub fn done(result: serde_json::Value) -> Self {
        Self {
            phase: "done".into(),
            bytes: None,
            total: None,
            percent: Some(100.0),
            message: Some("Complete".into()),
            done: Some(true),
            error: None,
            result: Some(result),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            phase: "error".into(),
            bytes: None,
            total: None,
            percent: None,
            message: Some(message.clone()),
            done: Some(true),
            error: Some(message),
            result: None,
        }
    }
}

pub type ProgressCallback = Box<dyn FnMut(ProgressEvent) + Send>;

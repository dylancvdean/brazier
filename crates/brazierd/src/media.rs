//! Daemon-side media preparation: hydrate blobs for vision, ASR, and video.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
};

use anyhow::Context;
use base64::Engine as _;
use serde_json::{Value, json};
use tokio::process::Command;

use crate::{
    blob_store,
    types::{ModelCapabilities, OpenAiMessage},
    whisper::{self, TranscribeRequest},
};

const MAX_VIDEO_FRAMES: usize = 8;
const FRAME_MAX_EDGE: u32 = 1024;

#[derive(Debug, Clone, Copy)]
pub struct PipelineFeatures {
    pub asr: bool,
    pub video_preprocess: bool,
}

pub fn detect_pipeline_features(data_dir: &Path, whisper_binary: Option<&str>, whisper_model: Option<&str>) -> PipelineFeatures {
    let binary = whisper::resolve_binary(data_dir, whisper_binary);
    let model = whisper::resolve_model_path(data_dir, whisper_model);
    PipelineFeatures {
        asr: binary.is_some() && model.is_some(),
        video_preprocess: ffmpeg_available(),
    }
}

pub fn ffmpeg_available() -> bool {
    command_on_path("ffmpeg") && command_on_path("ffprobe")
}

fn command_on_path(name: &str) -> bool {
    let Ok(path_env) = std::env::var("PATH") else {
        return false;
    };
    std::env::split_paths(&path_env).any(|dir| dir.join(name).is_file())
}

pub fn ffmpeg_missing_message() -> &'static str {
    "`ffmpeg` and `ffprobe` are required for audio conversion and video frame sampling. Install ffmpeg (for example `brew install ffmpeg`) and restart Brazier."
}

pub fn asr_missing_message() -> &'static str {
    "Audio transcription requires an active whisper.cpp runtime and a downloaded Whisper model. Build whisper.cpp under Runtimes and download a Whisper GGUF/ggml model from Discover."
}

#[derive(Clone)]
pub struct MediaContext<'a> {
    pub data_dir: &'a Path,
    pub model_caps: &'a ModelCapabilities,
    pub features: PipelineFeatures,
    pub whisper_binary: Option<PathBuf>,
    pub whisper_model: Option<PathBuf>,
}

pub type ProgressFn = Box<dyn Fn(String, String) + Send + Sync>;

/// Rewrite `brazier_blob` parts into engine-ready OpenAI content parts.
pub async fn prepare_messages(
    ctx: &MediaContext<'_>,
    messages: &mut [OpenAiMessage],
    progress: Option<ProgressFn>,
) -> anyhow::Result<()> {
    let emit = |phase: &str, message: &str| {
        if let Some(progress) = &progress {
            progress(phase.to_owned(), message.to_owned());
        }
    };

    for message in messages.iter_mut() {
        let Value::Array(parts) = &message.content else {
            continue;
        };
        let mut next_parts = Vec::with_capacity(parts.len());
        for part in parts {
            if part.get("type").and_then(Value::as_str) != Some("brazier_blob") {
                next_parts.push(part.clone());
                continue;
            }
            let blob = part
                .get("brazier_blob")
                .ok_or_else(|| anyhow::anyhow!("malformed brazier_blob part"))?;
            let sha256 = blob
                .get("sha256")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("brazier_blob missing sha256"))?;
            let mime = blob
                .get("mime_type")
                .and_then(Value::as_str)
                .unwrap_or("application/octet-stream");
            let name = blob
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("attachment");

            if mime.starts_with("image/") {
                anyhow::ensure!(
                    ctx.model_caps
                        .input_modalities
                        .iter()
                        .any(|modality| modality == "image"),
                    "Selected model does not support vision. Choose a model with an mmproj projector (llama.cpp) or an mlx-vlm snapshot."
                );
                emit("hydrate", &format!("Preparing image {name}…"));
                next_parts.push(image_part_from_blob(ctx.data_dir, sha256, mime).await?);
            } else if mime.starts_with("audio/") {
                let native = ctx
                    .model_caps
                    .audio_input
                    .as_deref()
                    .is_some_and(|mode| mode == "native");
                if native {
                    emit("hydrate", &format!("Preparing native audio for {name}…"));
                    next_parts.push(
                        native_audio_part_from_blob(ctx.data_dir, sha256, mime, name).await?,
                    );
                } else if ctx.features.asr {
                    emit("transcribe", &format!("Transcribing {name} via batch ASR…"));
                    let transcript = transcribe_blob(ctx, sha256, mime, name).await?;
                    next_parts.push(json!({
                        "type": "text",
                        "text": format!("[Transcript of {name}]\n{transcript}")
                    }));
                } else {
                    anyhow::bail!(
                        "No audio path available. Either select a native-audio chat model (audio_input=native), or enable batch ASR by building whisper.cpp and downloading a Whisper model. Realtime voice (PersonaPlex / speech-to-speech) is not implemented yet."
                    );
                }
            } else if mime.starts_with("video/") {
                anyhow::ensure!(
                    ctx.features.video_preprocess,
                    "{}",
                    ffmpeg_missing_message()
                );
                anyhow::ensure!(
                    ctx.model_caps
                        .input_modalities
                        .iter()
                        .any(|modality| modality == "image"),
                    "Video attachments need a vision-capable chat model (llama.cpp + mmproj or mlx-vlm) so sampled frames can be understood."
                );
                emit("extract_frames", &format!("Sampling frames from {name}…"));
                let prepared = prepare_video(ctx, sha256, mime, name, progress.as_ref()).await?;
                next_parts.extend(prepared);
            } else {
                anyhow::bail!("unsupported attachment type `{mime}`");
            }
        }
        message.content = Value::Array(next_parts);
    }
    Ok(())
}

async fn image_part_from_blob(
    data_dir: &Path,
    sha256: &str,
    mime: &str,
) -> anyhow::Result<Value> {
    let (bytes, stored_mime) = blob_store::read_blob(data_dir, sha256).await?;
    let mime = if stored_mime.starts_with("image/") {
        stored_mime
    } else {
        mime.to_owned()
    };
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(json!({
        "type": "image_url",
        "image_url": {
            "url": format!("data:{mime};base64,{encoded}")
        }
    }))
}

/// OpenAI-style `input_audio` part for models with `audio_input: native`.
/// Keeps `brazier_sha256` so chat can fall back to batch ASR without re-uploading.
async fn native_audio_part_from_blob(
    data_dir: &Path,
    sha256: &str,
    mime: &str,
    name: &str,
) -> anyhow::Result<Value> {
    let input = blob_to_temp_file(data_dir, sha256, extension_for_mime(mime)).await?;
    let wav = ensure_wav(&input).await?;
    let bytes = tokio::fs::read(&wav)
        .await
        .context("read converted wav for native audio")?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let _ = tokio::fs::remove_file(&input).await;
    if wav != input {
        let _ = tokio::fs::remove_file(&wav).await;
    }
    Ok(json!({
        "type": "input_audio",
        "input_audio": {
            "data": encoded,
            "format": "wav"
        },
        "brazier_sha256": sha256,
        "brazier_name": name,
        "brazier_mime_type": mime
    }))
}

pub fn messages_contain_input_audio(messages: &[OpenAiMessage]) -> bool {
    messages.iter().any(|message| match &message.content {
        Value::Array(parts) => parts.iter().any(|part| {
            part.get("type").and_then(Value::as_str) == Some("input_audio")
        }),
        _ => false,
    })
}

pub fn looks_like_audio_rejection(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    lower.contains("input_audio")
        || lower.contains("audio content")
        || lower.contains("unsupported content")
        || lower.contains("unknown content type")
        || lower.contains("modality")
        || lower.contains("multimodal")
        || (lower.contains("audio")
            && (lower.contains("not support")
                || lower.contains("unsupported")
                || lower.contains("invalid")
                || lower.contains("400")))
}

/// Replace native `input_audio` parts with Whisper transcripts when the chat
/// engine rejected audio. Requires batch ASR to be available.
pub async fn fallback_native_audio_to_asr(
    ctx: &MediaContext<'_>,
    messages: &mut [OpenAiMessage],
    progress: Option<&ProgressFn>,
) -> anyhow::Result<usize> {
    anyhow::ensure!(ctx.features.asr, "{}", asr_missing_message());
    let emit = |phase: &str, message: &str| {
        if let Some(progress) = progress {
            progress(phase.to_owned(), message.to_owned());
        }
    };
    let mut converted = 0_usize;
    for message in messages.iter_mut() {
        let Value::Array(parts) = &message.content else {
            continue;
        };
        let mut next_parts = Vec::with_capacity(parts.len());
        for part in parts {
            if part.get("type").and_then(Value::as_str) != Some("input_audio") {
                next_parts.push(part.clone());
                continue;
            }
            let sha256 = part
                .get("brazier_sha256")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "native audio part is missing brazier_sha256; cannot fall back to ASR"
                    )
                })?;
            let name = part
                .get("brazier_name")
                .and_then(Value::as_str)
                .unwrap_or("audio");
            let mime = part
                .get("brazier_mime_type")
                .and_then(Value::as_str)
                .unwrap_or("audio/wav");
            emit("transcribe", &format!("Engine rejected native audio; transcribing {name}…"));
            let transcript = transcribe_blob(ctx, sha256, mime, name).await?;
            next_parts.push(json!({
                "type": "text",
                "text": format!("[Transcript of {name} — native audio unsupported by engine]\n{transcript}")
            }));
            converted += 1;
        }
        message.content = Value::Array(next_parts);
    }
    Ok(converted)
}

/// Materialize a blob as a 16 kHz WAV temp file for ASR workers.
pub async fn materialize_wav_from_blob(
    data_dir: &Path,
    sha256: &str,
    mime: &str,
) -> anyhow::Result<PathBuf> {
    let input = blob_to_temp_file(data_dir, sha256, extension_for_mime(mime)).await?;
    ensure_wav(&input).await
}

async fn blob_to_temp_file(
    data_dir: &Path,
    sha256: &str,
    extension: &str,
) -> anyhow::Result<PathBuf> {
    let (bytes, _) = blob_store::read_blob(data_dir, sha256).await?;
    let dir = data_dir.join("tmp").join("media");
    tokio::fs::create_dir_all(&dir)
        .await
        .context("create media temp directory")?;
    let path = dir.join(format!("{sha256}.{extension}"));
    tokio::fs::write(&path, &bytes)
        .await
        .context("write media temp file")?;
    Ok(path)
}

fn extension_for_mime(mime: &str) -> &'static str {
    match mime {
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/wav" | "audio/wave" | "audio/x-wav" => "wav",
        "audio/flac" => "flac",
        "audio/ogg" => "ogg",
        "audio/webm" => "webm",
        "audio/mp4" | "audio/m4a" | "audio/x-m4a" => "m4a",
        "video/mp4" => "mp4",
        "video/webm" => "webm",
        "video/quicktime" => "mov",
        "video/x-matroska" => "mkv",
        _ if mime.starts_with("audio/") => "audio",
        _ if mime.starts_with("video/") => "video",
        _ => "bin",
    }
}

async fn ensure_wav(input: &Path) -> anyhow::Result<PathBuf> {
    if input
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
    {
        return Ok(input.to_path_buf());
    }
    anyhow::ensure!(ffmpeg_available(), "{}", ffmpeg_missing_message());
    let output = input.with_extension("16k.wav");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            &input.display().to_string(),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            &output.display().to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .context("run ffmpeg for audio conversion")?;
    anyhow::ensure!(status.success(), "ffmpeg failed to convert audio to WAV");
    Ok(output)
}

async fn transcribe_blob(
    ctx: &MediaContext<'_>,
    sha256: &str,
    mime: &str,
    name: &str,
) -> anyhow::Result<String> {
    let binary = ctx
        .whisper_binary
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("{}", asr_missing_message()))?;
    let model = ctx
        .whisper_model
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("{}", asr_missing_message()))?;
    let input = blob_to_temp_file(ctx.data_dir, sha256, extension_for_mime(mime)).await?;
    let wav = ensure_wav(&input).await?;
    let text = whisper::transcribe(TranscribeRequest {
        binary,
        model,
        audio: &wav,
    })
    .await
    .with_context(|| format!("transcribe {name}"))?;
    let _ = tokio::fs::remove_file(&input).await;
    if wav != input {
        let _ = tokio::fs::remove_file(&wav).await;
    }
    Ok(text)
}

async fn prepare_video(
    ctx: &MediaContext<'_>,
    sha256: &str,
    mime: &str,
    name: &str,
    progress: Option<&ProgressFn>,
) -> anyhow::Result<Vec<Value>> {
    let emit = |phase: &str, message: &str| {
        if let Some(progress) = progress {
            progress(phase.to_owned(), message.to_owned());
        }
    };
    let input = blob_to_temp_file(ctx.data_dir, sha256, extension_for_mime(mime)).await?;
    let duration = probe_duration_seconds(&input).await.unwrap_or(1.0).max(0.1);
    let frame_count = MAX_VIDEO_FRAMES.min(((duration / 2.0).ceil() as usize).max(1));
    let frame_dir = ctx.data_dir.join("tmp").join("media").join(format!("{sha256}-frames"));
    tokio::fs::create_dir_all(&frame_dir)
        .await
        .context("create frame directory")?;

    let fps = frame_count as f64 / duration;
    let pattern = frame_dir.join("frame-%03d.jpg");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            &input.display().to_string(),
            "-vf",
            &format!("fps={fps:.6},scale='min({FRAME_MAX_EDGE},iw)':-2"),
            "-frames:v",
            &frame_count.to_string(),
            "-q:v",
            "3",
            &pattern.display().to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .context("run ffmpeg for frame sampling")?;
    anyhow::ensure!(status.success(), "ffmpeg failed to sample video frames");

    let mut parts = Vec::new();
    parts.push(json!({
        "type": "text",
        "text": format!(
            "[Video attachment: {name}. Showing {frame_count} sampled frames from a {duration:.1}s clip.]"
        )
    }));

    if ctx.features.asr && ctx.whisper_binary.is_some() && ctx.whisper_model.is_some() {
        emit("transcribe", &format!("Transcribing audio from {name}…"));
        match extract_and_transcribe_audio(ctx, &input, name).await {
            Ok(transcript) if !transcript.trim().is_empty() => {
                parts.push(json!({
                    "type": "text",
                    "text": format!("[Audio transcript of {name}]\n{transcript}")
                }));
            }
            Ok(_) => {
                parts.push(json!({
                    "type": "text",
                    "text": format!("[No speech detected in audio track of {name}.]")
                }));
            }
            Err(error) => {
                parts.push(json!({
                    "type": "text",
                    "text": format!("[Could not transcribe audio from {name}: {error}]")
                }));
            }
        }
    } else {
        parts.push(json!({
            "type": "text",
            "text": "[Audio transcription skipped — whisper.cpp runtime/model not available.]"
        }));
    }

    let mut frames = tokio::fs::read_dir(&frame_dir)
        .await
        .context("read sampled frames")?;
    let mut frame_paths = Vec::new();
    while let Some(entry) = frames.next_entry().await? {
        let path = entry.path();
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("jpg"))
        {
            frame_paths.push(path);
        }
    }
    frame_paths.sort();
    for path in frame_paths {
        let bytes = tokio::fs::read(&path).await.context("read frame")?;
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        parts.push(json!({
            "type": "image_url",
            "image_url": {
                "url": format!("data:image/jpeg;base64,{encoded}")
            }
        }));
        let _ = tokio::fs::remove_file(&path).await;
    }

    let _ = tokio::fs::remove_dir_all(&frame_dir).await;
    let _ = tokio::fs::remove_file(&input).await;
    Ok(parts)
}

async fn extract_and_transcribe_audio(
    ctx: &MediaContext<'_>,
    video: &Path,
    name: &str,
) -> anyhow::Result<String> {
    let audio = video.with_extension("track.wav");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            &video.display().to_string(),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            &audio.display().to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .context("extract audio track")?;
    if !status.success() || !audio.is_file() {
        anyhow::bail!("no audio track");
    }
    let binary = ctx
        .whisper_binary
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("{}", asr_missing_message()))?;
    let model = ctx
        .whisper_model
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("{}", asr_missing_message()))?;
    let text = whisper::transcribe(TranscribeRequest {
        binary,
        model,
        audio: &audio,
    })
    .await
    .with_context(|| format!("transcribe audio from {name}"))?;
    let _ = tokio::fs::remove_file(&audio).await;
    Ok(text)
}

async fn probe_duration_seconds(path: &Path) -> anyhow::Result<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            &path.display().to_string(),
        ])
        .output()
        .await
        .context("run ffprobe")?;
    anyhow::ensure!(output.status.success(), "ffprobe failed");
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim()
        .parse::<f64>()
        .context("parse ffprobe duration")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelCapabilities;

    #[test]
    fn detects_audio_rejection_and_input_audio_parts() {
        assert!(looks_like_audio_rejection("unsupported content type input_audio"));
        assert!(looks_like_audio_rejection("Model does not support audio modality"));
        assert!(!looks_like_audio_rejection("context length exceeded"));
        let messages = vec![OpenAiMessage {
            role: "user".into(),
            content: json!([{
                "type": "input_audio",
                "input_audio": {"data": "AA", "format": "wav"},
                "brazier_sha256": "abc"
            }]),
            tool_calls: None,
            tool_call_id: None,
        }];
        assert!(messages_contain_input_audio(&messages));
    }

    #[tokio::test]
    async fn hydrates_image_blobs_to_data_urls() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = b"\x89PNG\r\n\x1a\nfake";
        let stored = blob_store::store_bytes(dir.path(), bytes, "image/png", Some("x.png"))
            .await
            .unwrap();
        let caps = ModelCapabilities {
            input_modalities: vec!["text".into(), "image".into()],
            output_modalities: vec!["text".into()],
            streaming: true,
            tools: false,
            reasoning: false,
            max_context_length: None,
            reasoning_modes: Vec::new(),
            harmony: false,
            audio_input: None,
        };
        let ctx = MediaContext {
            data_dir: dir.path(),
            model_caps: &caps,
            features: PipelineFeatures {
                asr: false,
                video_preprocess: false,
            },
            whisper_binary: None,
            whisper_model: None,
        };
        let mut messages = vec![OpenAiMessage {
            role: "user".into(),
            content: json!([
                {"type": "text", "text": "describe"},
                {"type": "brazier_blob", "brazier_blob": {
                    "sha256": stored.sha256,
                    "mime_type": "image/png",
                    "name": "x.png"
                }}
            ]),
            tool_calls: None,
            tool_call_id: None,
        }];
        prepare_messages(&ctx, &mut messages, None).await.unwrap();
        let parts = messages[0].content.as_array().unwrap();
        assert_eq!(parts[1]["type"], "image_url");
        let url = parts[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
    }

    #[tokio::test]
    async fn rejects_images_without_vision_capability() {
        let dir = tempfile::tempdir().unwrap();
        let stored = blob_store::store_bytes(dir.path(), b"img", "image/png", None)
            .await
            .unwrap();
        let caps = ModelCapabilities {
            input_modalities: vec!["text".into()],
            output_modalities: vec!["text".into()],
            streaming: true,
            tools: false,
            reasoning: false,
            max_context_length: None,
            reasoning_modes: Vec::new(),
            harmony: false,
            audio_input: None,
        };
        let ctx = MediaContext {
            data_dir: dir.path(),
            model_caps: &caps,
            features: PipelineFeatures {
                asr: false,
                video_preprocess: false,
            },
            whisper_binary: None,
            whisper_model: None,
        };
        let mut messages = vec![OpenAiMessage {
            role: "user".into(),
            content: json!([{
                "type": "brazier_blob",
                "brazier_blob": {
                    "sha256": stored.sha256,
                    "mime_type": "image/png",
                    "name": "x.png"
                }
            }]),
            tool_calls: None,
            tool_call_id: None,
        }];
        let err = prepare_messages(&ctx, &mut messages, None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("vision"));
    }
}

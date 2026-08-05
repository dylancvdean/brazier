use std::{convert::Infallible, net::SocketAddr, path::PathBuf, time::Duration};

use anyhow::Context as _;
use async_stream::stream;
use axum::http::header;
use axum::{
    Json, Router,
    extract::{ConnectInfo, DefaultBodyLimit, FromRequestParts, Path, Query, Request, State},
    http::{HeaderMap, HeaderValue, StatusCode, request::Parts},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{delete, get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use uuid::Uuid;

use crate::{
    AppState, adapters, blob_store,
    build_recipe::{self, BuildPlanRequest},
    builds,
    db::ConversationExport,
    db::CreateRunSnapshot,
    download::{self},
    engine::{Engine, StreamEvent},
    fork_hints::{self, ModelLoadError, RuntimeForkHint},
    hf::{self, SearchQuery},
    hf_auth, llama, mcp, media, model_bindings, model_settings, models_store,
    progress::ProgressEvent,
    recommendations, remote, runtimes, sdcpp, sdcpp_arch, sdcpp_catalog, streaming_asr, support,
    tool_registry, toolchain_hints,
    types::{
        ChatCompletionRequest, CreateConversation, CreateMessage, OpenAiMessage, ResponsesRequest,
        text_from_content,
    },
    voice, whisper, whisperkit,
};

type ApiResult<T> = Result<T, ApiError>;

/// A direct download still needs a durable job row: the HTTP response is only
/// how the current screen receives progress, while the download tray is what
/// lets the work remain visible after that screen goes away.
async fn track_resumable_download(
    state: &AppState,
    work: &crate::download_queue::QueuedWork,
) -> ApiResult<(String, std::sync::Arc<crate::active_downloads::StopFlag>)> {
    let payload = serde_json::to_string(work).map_err(ApiError::internal)?;
    let job = state
        .db
        .create_queued_download_job(crate::db::QueuedDownloadJobInput {
            repo_id: &work.repo_id(),
            filename: &work.filename(),
            revision: &work.revision(),
            kind: work.kind(),
            payload: Some(&payload),
            label: Some(&work.label()),
            status: "pending",
        })
        .await
        .map_err(ApiError::internal)?;
    let cancel = state.active_downloads.register(&job.id);
    Ok((job.id, cancel))
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
    fork_hints: Option<Vec<RuntimeForkHint>>,
}

impl ApiError {
    fn bad_request(error: impl ToString) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: error.to_string(),
            fork_hints: None,
        }
    }

    fn not_found(error: impl ToString) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: error.to_string(),
            fork_hints: None,
        }
    }

    fn internal(error: impl ToString) -> Self {
        tracing::error!(error = %error.to_string(), "request failed");
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "The request could not be completed.".to_owned(),
            fork_hints: None,
        }
    }

    /// A local engine failed and its own account of why is worth showing.
    ///
    /// Unlike [`Self::internal`], the message survives to the interface: an
    /// sd-cli job that ran out of time or died mid-render is diagnosable only
    /// if the person is told what it said, rather than being sent to the
    /// terminal to find out.
    fn engine_failure(error: impl ToString) -> Self {
        let message = error.to_string();
        tracing::error!(error = %message, "engine job failed");
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            message,
            fork_hints: None,
        }
    }

    /// The user stopped the work; not a failure, and told apart by its status.
    fn cancelled(error: impl ToString) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            message: error.to_string(),
            fork_hints: None,
        }
    }

    fn model_load(message: String, fork_hints: Vec<RuntimeForkHint>) -> Self {
        tracing::error!(error = %message, "model load failed");
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            message,
            fork_hints: (!fork_hints.is_empty()).then_some(fork_hints),
        }
    }

    fn from_anyhow(error: anyhow::Error) -> Self {
        if let Some(load) = error.downcast_ref::<ModelLoadError>() {
            return Self::model_load(load.cause.clone(), load.fork_hints.clone());
        }
        let message = error.to_string();
        // Local engine launch failures (OOM, bad flags, missing weights) are only
        // useful if the person sees what the server said.
        if crate::llama::startup_looks_like_oom(&message)
            || message.contains("exited during startup")
            || message.contains("ran out of memory while starting")
            || message.contains("health check timed out")
        {
            return Self::engine_failure(message);
        }
        Self::internal(error)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut body = json!({ "error": { "message": self.message } });
        if let Some(fork_hints) = self.fork_hints {
            body["brazier"] = json!({ "fork_hints": fork_hints });
        }
        (self.status, Json(body)).into_response()
    }
}

/// Origins the browser UI is served from during development.
///
/// `null` is the origin of a `file://` page, which is what the packaged
/// renderer is; the two localhost ports are `electron-vite dev`.
const DEFAULT_ORIGINS: [&str; 3] = ["null", "http://localhost:5173", "http://127.0.0.1:5173"];

/// Turn configured origin strings into header values, refusing what cannot work.
///
/// A wildcard is refused rather than translated: this daemon holds a machine's
/// conversations and can execute tools, and "any page may call it" is not a
/// thing to enable by typing `*` into a flag. Named origins are the only way to
/// widen it, so widening is always deliberate and always visible in the launch
/// command.
pub fn parse_origins(origins: &[String]) -> anyhow::Result<Vec<HeaderValue>> {
    let mut values = Vec::new();
    for origin in DEFAULT_ORIGINS
        .iter()
        .map(|origin| (*origin).to_owned())
        .chain(
            origins
                .iter()
                .flat_map(|value| value.split(','))
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty()),
        )
    {
        anyhow::ensure!(
            origin != "*",
            "a wildcard CORS origin is not accepted; name the origins that may call this daemon"
        );
        if origin != "null" {
            anyhow::ensure!(
                origin.starts_with("http://") || origin.starts_with("https://"),
                "`{origin}` is not an origin: it must start with http:// or https://"
            );
            anyhow::ensure!(
                !origin.ends_with('/') && origin.matches('/').count() == 2,
                "`{origin}` is not an origin: it must have no path or trailing slash"
            );
        }
        let value = HeaderValue::from_str(&origin)
            .with_context(|| format!("`{origin}` cannot be sent as a header"))?;
        if !values.contains(&value) {
            values.push(value);
        }
    }
    Ok(values)
}

pub fn router(state: AppState) -> Router {
    router_with_origins(
        state,
        parse_origins(&[]).expect("built-in origins are valid"),
    )
}

/// The router, with the set of browser origins allowed to call it.
pub fn router_with_origins(state: AppState, origins: Vec<HeaderValue>) -> Router {
    let protected = Router::new()
        .route("/api/v1/daemon/info", get(daemon_info))
        .route("/api/v1/capabilities", get(capabilities))
        .route("/api/v1/support/bundle", get(support_bundle))
        .route(
            "/api/v1/conversations",
            get(list_conversations).post(create_conversation),
        )
        .route(
            "/api/v1/conversations/{id}",
            get(get_conversation)
                .patch(update_conversation)
                .delete(delete_conversation),
        )
        .route(
            "/api/v1/conversations/{id}/messages",
            get(list_messages).post(create_message),
        )
        .route(
            "/api/v1/conversations/{id}/messages/{message_id}",
            axum::routing::patch(update_message),
        )
        .route(
            "/api/v1/conversations/{id}/export",
            get(export_conversation),
        )
        .route("/api/v1/conversations/import", post(import_conversation))
        .route(
            "/api/v1/conversations/{id}/runs",
            get(list_run_snapshots).post(create_run_snapshot),
        )
        // Attachments are sent as base64 JSON. Allow for the 4/3 encoding
        // overhead of the 50 MiB video limit while `blob_store` continues to
        // enforce the actual type-specific limits after decoding.
        .route(
            "/api/v1/blobs",
            post(upload_blob).layer(DefaultBodyLimit::max(70 * 1024 * 1024)),
        )
        .route("/api/v1/blobs/{sha256}", get(get_blob))
        .route(
            "/api/v1/huggingface/token",
            get(huggingface_token_status)
                .put(set_huggingface_token)
                .delete(clear_huggingface_token),
        )
        .route(
            "/api/v1/remote/connections",
            get(list_remote_connections).put(save_remote_connection),
        )
        .route(
            "/api/v1/remote/connections/{id}",
            delete(delete_remote_connection),
        )
        .route(
            "/api/v1/remote/connections/{id}/test",
            post(test_remote_connection),
        )
        .route("/api/v1/huggingface/models", get(search_hugging_face))
        .route(
            "/api/v1/huggingface/models/{repo_owner}/{repo_name}/files",
            get(list_hub_files),
        )
        .route(
            "/api/v1/huggingface/models/{repo_owner}/{repo_name}/trust",
            get(model_trust),
        )
        .route(
            "/api/v1/huggingface/models/{repo_owner}/{repo_name}/description",
            get(model_description),
        )
        .route(
            "/api/v1/huggingface/models/{repo_owner}/{repo_name}/fork-hints",
            get(model_fork_hints),
        )
        .route("/api/v1/engines/build-plan", post(build_plan))
        .route("/api/v1/engines", get(engine_status))
        .route(
            "/api/v1/runtime/settings",
            get(runtime_settings).put(update_runtime_settings),
        )
        .route(
            "/api/v1/preferences/welcome",
            get(welcome_preference).put(update_welcome_preference),
        )
        .route(
            "/api/v1/preferences/agent",
            get(agent_preference).put(update_agent_preference),
        )
        .route(
            "/api/v1/preferences/workspace",
            get(workspace_preference).put(update_workspace_preference),
        )
        .route(
            "/api/v1/preferences/computer",
            get(computer_preference).put(update_computer_preference),
        )
        .route(
            "/api/v1/computer/permissions",
            get(computer_os_permissions).post(request_computer_os_permissions),
        )
        .route(
            "/api/v1/computer/sessions",
            get(list_computer_sessions).post(create_computer_session),
        )
        .route(
            "/api/v1/computer/sessions/{id}",
            get(get_computer_session)
                .put(update_computer_session)
                .delete(delete_computer_session),
        )
        .route(
            "/api/v1/computer/sessions/{id}/steps",
            get(list_computer_steps).post(append_computer_step),
        )
        .route(
            "/api/v1/computer/sessions/{id}/screenshot",
            post(computer_screenshot),
        )
        .route(
            "/api/v1/computer/sessions/{id}/preview",
            post(computer_preview),
        )
        .route(
            "/api/v1/computer/sessions/{id}/stream",
            get(computer_stream),
        )
        .route(
            "/api/v1/computer/sessions/{id}/stop",
            post(stop_computer_session),
        )
        .route(
            "/api/v1/computer/sessions/{id}/safety-authority",
            post(set_computer_safety_authority),
        )
        .route(
            "/api/v1/computer/desktop-authority/revoke-all",
            post(revoke_all_desktop_authority),
        )
        .route("/api/v1/computer/exec", post(computer_exec_action))
        .route(
            "/api/v1/computer/approvals/{id}",
            post(decide_computer_approval),
        )
        .route("/api/v1/computer/parse-fara", post(parse_fara_output))
        .route("/api/v1/hardware", get(hardware))
        .route(
            "/api/v1/toolchain",
            get(toolchain_status).post(setup_toolchain),
        )
        .route("/api/v1/engines/llama.cpp/ensure", post(ensure_llama))
        .route(
            "/api/v1/engines/llama.cpp/managed-status",
            get(managed_llama_status),
        )
        .route("/api/v1/engines/whisper.cpp/ensure", post(ensure_whisper))
        .route(
            "/api/v1/engines/whisper.cpp/managed-status",
            get(managed_whisper_status),
        )
        .route(
            "/api/v1/engines/stable-diffusion.cpp/ensure",
            post(ensure_sdcpp),
        )
        .route(
            "/api/v1/engines/stable-diffusion.cpp/managed-status",
            get(managed_sdcpp_status),
        )
        .route("/api/v1/models/sdcpp/catalog", get(sdcpp_catalog))
        .route("/api/v1/models/sdcpp/assemble", post(assemble_sdcpp_bundle))
        .route(
            "/api/v1/models/sdcpp/bundles",
            axum::routing::put(save_sdcpp_bundle),
        )
        .route(
            "/api/v1/models/sdcpp/bundles/{id}",
            axum::routing::delete(delete_sdcpp_bundle),
        )
        .route("/api/v1/models/sdcpp/consent", post(accept_sdcpp_license))
        .route("/api/v1/models/sdcpp/install", post(install_sdcpp_bundle))
        .route("/api/v1/generate/image", post(generate_image))
        .route("/api/v1/generate/video", post(generate_video))
        .route("/api/v1/generate/active", get(active_generation))
        .route("/api/v1/generate/cancel", post(cancel_generation))
        .route(
            "/api/v1/voice/sessions",
            get(list_voice_session).post(create_voice_session),
        )
        .route(
            "/api/v1/voice/sessions/{id}",
            axum::routing::delete(end_voice_session),
        )
        .route("/api/v1/agent/capabilities", get(agent_capabilities))
        .route("/api/v1/agent/tools", get(agent_tool_catalog))
        .route(
            "/api/v1/agent/sessions",
            get(list_agent_sessions).post(create_agent_session),
        )
        .route(
            "/api/v1/agent/sessions/{id}",
            get(get_agent_session)
                .patch(patch_agent_session)
                .delete(delete_agent_session),
        )
        .route(
            "/api/v1/agent/sessions/{id}/messages",
            get(list_agent_messages).post(append_agent_messages),
        )
        .route(
            "/api/v1/agent/sessions/{id}/tool-executions",
            get(list_agent_tool_executions),
        )
        .route(
            "/api/v1/agent/sessions/{id}/approvals",
            get(list_agent_approvals),
        )
        .route("/api/v1/agent/sessions/{id}/cancel", post(cancel_agent_run))
        .route(
            "/api/v1/agent/sessions/{id}/apply-worktree",
            post(apply_agent_worktree),
        )
        .route(
            "/api/v1/agent/sessions/{id}/worktree",
            get(agent_worktree_status),
        )
        .route(
            "/api/v1/agent/sessions/{id}/prompt",
            get(agent_system_prompt),
        )
        .route("/api/v1/agent/exec", post(agent_exec_tool))
        .route("/api/v1/agent/exec/stream", post(agent_exec_tool_stream))
        .route(
            "/api/v1/agent/approvals/{id}",
            get(get_agent_approval).post(decide_agent_approval),
        )
        .route("/api/v1/agent/artifacts/{id}", get(get_agent_artifact))
        .route(
            "/api/v1/agent/workspaces/prompt",
            get(get_agent_workspace_prompt).put(put_agent_workspace_prompt),
        )
        .route("/api/v1/agent/workspace", post(validate_agent_workspace))
        .route("/api/v1/tools", get(list_tools))
        .route(
            "/api/v1/mcp/servers",
            get(list_mcp_servers).post(create_mcp_server),
        )
        .route(
            "/api/v1/mcp/servers/{id}",
            axum::routing::put(update_mcp_server).delete(delete_mcp_server),
        )
        .route("/api/v1/mcp/servers/{id}/refresh", post(refresh_mcp_server))
        .route(
            "/api/v1/runtimes",
            get(list_runtimes).delete(delete_runtime),
        )
        .route("/api/v1/runtimes/activate", post(activate_runtime))
        .route("/api/v1/runtimes/deactivate", post(deactivate_runtime))
        .route(
            "/api/v1/runtimes/check-updates",
            post(check_runtime_updates),
        )
        .route("/api/v1/runtimes/build", post(build_runtime))
        .route("/api/v1/runtimes/build/cancel", post(cancel_build))
        .route("/api/v1/runtimes/build/cancel-job", post(cancel_build_job))
        .route("/api/v1/models/download", post(download_model))
        .route("/api/v1/models/download/mlx", post(download_mlx_model))
        .route(
            "/api/v1/models/download/streaming-asr",
            post(download_streaming_asr_model),
        )
        .route(
            "/api/v1/models/download/personaplex",
            post(download_personaplex_model),
        )
        .route("/api/v1/models/download/queue", post(queue_model_download))
        .route(
            "/api/v1/models/download/queue/snapshot/{kind}",
            post(queue_snapshot_download),
        )
        .route(
            "/api/v1/models/sdcpp/install/queue",
            post(queue_sdcpp_install),
        )
        .route("/api/v1/models/download/pause", post(pause_model_download))
        .route(
            "/api/v1/models/download/resume",
            post(resume_model_download),
        )
        .route(
            "/api/v1/models/download/cancel",
            post(cancel_model_download),
        )
        .route("/api/v1/models/downloads", get(list_download_jobs))
        .route(
            "/api/v1/models/download/dismiss",
            post(dismiss_model_download),
        )
        .route(
            "/api/v1/models/downloads/finished",
            delete(dismiss_finished_model_downloads),
        )
        .route(
            "/api/v1/models/library-paths/suggestions",
            get(model_library_path_suggestions),
        )
        .route("/api/v1/models", axum::routing::delete(delete_local_model))
        .route(
            "/api/v1/models/bindings",
            get(model_bindings_list).put(update_model_binding),
        )
        .route(
            "/api/v1/models/settings",
            get(model_settings_list).put(update_model_settings),
        )
        .route("/api/v1/models/settings/reset", post(reset_model_settings))
        .route("/api/v1/models/chat-template", get(model_chat_template))
        .route("/api/v1/recommendations", get(model_recommendations))
        .route(
            "/api/v1/recommendations/setups",
            get(list_recommendation_setups).post(start_recommendation_setup),
        )
        .route(
            "/api/v1/recommendations/setups/{id}/cancel",
            post(cancel_recommendation_setup),
        )
        .route(
            "/api/v1/recommendations/state",
            get(recommendation_state).put(update_recommendation_state),
        )
        .route(
            "/api/v1/recommendations/installed",
            post(record_recommendation_install),
        )
        .route("/api/v1/adapters", get(list_adapters))
        .route("/api/v1/adapters/register", post(register_adapter))
        .route("/api/v1/adapters/forget", post(forget_adapter))
        .route("/api/v1/adapters/delete", post(delete_adapter))
        .route("/api/v1/adapters/download", post(download_adapter))
        .route("/api/v1/models/prepare", post(prepare_model))
        .route("/api/v1/models/loaded", delete(unload_model))
        .route("/v1/models", get(list_models))
        // Computer Use can include several full desktop screenshots in one
        // trajectory. Axum's 2 MiB default rejects those before the runtime
        // ever sees them, despite the image-history limit being valid.
        .route(
            "/v1/chat/completions",
            post(chat_completions).layer(DefaultBodyLimit::max(70 * 1024 * 1024)),
        )
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/responses", post(responses))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_auth));

    Router::new()
        .route("/health", get(health))
        .merge(protected)
        .with_state(state)
        .layer(
            CorsLayer::new()
                .allow_origin(origins)
                .allow_headers(Any)
                .allow_methods(Any),
        )
        .layer(TraceLayer::new_for_http())
}

async fn require_auth(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    if state.api_keys.is_empty() {
        return next.run(request).await;
    }
    let supplied = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    if supplied.is_some_and(|key| state.api_keys.iter().any(|expected| expected == key)) {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": { "message": "A valid API key is required." } })),
        )
            .into_response()
    }
}

async fn health(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    state.db.ping().await.map_err(ApiError::internal)?;
    let llama = state.runtime.llama_server_summary().await;
    Ok(Json(json!({
        "status": "healthy",
        "engine": state.runtime.id(),
        "version": env!("CARGO_PKG_VERSION"),
        "database": "ok",
        "llama_server": llama,
    })))
}

/// The small, authenticated handshake a remote desktop client needs before it
/// assumes a daemon speaks its management API. Keep this independent from the
/// much larger capabilities response: compatibility must be checkable even
/// while model discovery is slow or a remote engine is unavailable.
async fn daemon_info() -> Json<Value> {
    Json(json!({
        "product": "brazier",
        "version": env!("CARGO_PKG_VERSION"),
        "management_api": { "major": 1, "minor": 0 },
        "openai_api": { "chat_completions": "/v1/chat/completions", "responses": "/v1/responses" },
    }))
}

async fn capabilities(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let models = state.runtime.models().await.map_err(ApiError::internal)?;
    let settings = state.runtime.settings().await;
    let features_pipeline = media::detect_pipeline_features(
        &state.data_dir,
        settings.whisper_binary.as_deref(),
        settings.whisper_model.as_deref(),
    );
    let any_native_audio = models.iter().any(|model| {
        model
            .capabilities
            .audio_input
            .as_deref()
            .is_some_and(|mode| mode == "native")
    });
    let streaming_asr_available = streaming_asr::detect_available(
        &state.data_dir,
        settings.streaming_asr_python.as_deref(),
        settings.streaming_asr_model.as_deref(),
    );
    let sdcpp_binary = sdcpp::resolve_binary(&state.data_dir, settings.sdcpp_binary.as_deref());
    let image_gen_available = sdcpp_binary.is_some()
        && settings
            .default_image_gen_model
            .as_deref()
            .is_some_and(|id| sdcpp::path_for_model_id(&state.data_dir, id).is_ok());
    let video_gen_available = sdcpp_binary.is_some()
        && settings
            .default_video_gen_model
            .as_deref()
            .is_some_and(|id| sdcpp::path_for_model_id(&state.data_dir, id).is_ok());
    let voice_python = voice::resolve_python(&state.data_dir, settings.voice_python.as_deref());
    let voice_model =
        voice::resolve_model_path(&state.data_dir, settings.default_voice_model.as_deref());
    let realtime_voice_available =
        voice::realtime_voice_available(voice_python.as_deref(), voice_model.as_deref());
    Ok(Json(json!({
        "schema_version": 1,
        "models": models,
        "features": {
            "conversation_branches": true,
            "hugging_face_search": true,
            "model_download": true,
            "llama_cpp_engine": true,
            "whisper_cpp_engine": true,
            "streaming_asr_engine": true,
            "stable_diffusion_cpp_engine": true,
            "personaplex_engine": true,
            "openai_chat_completions": true,
            "openai_responses": true,
            "openai_audio_transcriptions": true,
            "conversation_search": true,
            "conversation_import_export": true,
            "model_download_jobs": true,
            "model_download_queue": true,
            "model_download_cancel": true,
            "model_trust_acknowledgement": true,
            // Legacy aliases — prefer audio_interfaces below.
            "asr": features_pipeline.asr,
            "video_preprocess": features_pipeline.video_preprocess,
            "audio_interfaces": {
                "batch_asr": {
                    "id": "batch_asr",
                    "available": features_pipeline.asr,
                    "engine": "whisper.cpp|whisperkit",
                    "summary": "File/blob transcription via whisper.cpp or WhisperKit (Argmax) before chat. Works with any text model."
                },
                "native_model_audio": {
                    "id": "native_model_audio",
                    "available": any_native_audio,
                    "summary": "Chat model consumes audio tokens directly (OpenAI input_audio). Detected on specific audio-LLM checkpoints; not Whisper ASR weights. Falls back to batch ASR when the chat engine rejects input_audio."
                },
                "streaming_asr": {
                    "id": "streaming_asr",
                    "available": streaming_asr_available,
                    "engine": "streaming-asr",
                    "summary": "Low-latency chunked transcription via NVIDIA Nemotron ASR Streaming (Transformers). POST /v1/audio/transcriptions with stream=true."
                },
                "realtime_voice": {
                    "id": "realtime_voice",
                    "available": realtime_voice_available,
                    "planned": !realtime_voice_available,
                    "engine": "personaplex",
                    "summary": "Full-duplex speech-to-speech via Moshi protocol (PersonaPlex). Dedicated Voice mode; not file-attach chat."
                }
            },
            "generation_interfaces": {
                "image_gen": {
                    "id": "image_gen",
                    "available": image_gen_available,
                    "engine": "stable-diffusion.cpp",
                    "summary": "Local image generation via sd-cli (SD/Flux/Qwen-Image). Chat tool or Generate mode."
                },
                "video_gen": {
                    "id": "video_gen",
                    "available": video_gen_available,
                    "engine": "stable-diffusion.cpp",
                    "summary": "Local video generation via sd-cli (Wan/LTX). Chat tool or Generate mode."
                }
            }
        }
    })))
}

async fn support_bundle(State(state): State<AppState>) -> ApiResult<Response> {
    let bytes = support::create_bundle(&state)
        .await
        .map_err(ApiError::internal)?;
    Ok((
        [
            (header::CONTENT_TYPE, "application/zip"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"brazier-support.zip\"",
            ),
            (header::CACHE_CONTROL, "no-store"),
        ],
        bytes,
    )
        .into_response())
}

async fn list_conversations(
    State(state): State<AppState>,
    Query(query): Query<ConversationListQuery>,
) -> ApiResult<Json<Value>> {
    let conversations = state
        .db
        .list_conversations(query.q.as_deref())
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "data": conversations })))
}

#[derive(Debug, Deserialize)]
struct ConversationListQuery {
    q: Option<String>,
}

async fn export_conversation(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<ConversationExport>> {
    state
        .db
        .export_conversation(&state.data_dir, &id)
        .await
        .map(Json)
        .map_err(ApiError::bad_request)
}

async fn import_conversation(
    State(state): State<AppState>,
    Json(export): Json<ConversationExport>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    let conversation = state
        .db
        .import_conversation(&state.data_dir, export)
        .await
        .map_err(ApiError::bad_request)?;
    Ok((StatusCode::CREATED, Json(json!(conversation))))
}

async fn create_conversation(
    State(state): State<AppState>,
    Json(request): Json<CreateConversation>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    let title = request
        .title
        .as_deref()
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .unwrap_or("New conversation");
    let conversation = state
        .db
        .create_conversation(title)
        .await
        .map_err(ApiError::internal)?;
    Ok((StatusCode::CREATED, Json(json!(conversation))))
}

async fn get_conversation(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let conversation = state
        .db
        .get_conversation(&id)
        .await
        .map_err(ApiError::not_found)?;
    Ok(Json(json!(conversation)))
}

/// Bind the conversation to an agent session, retitle it, or store the compact
/// summary a voice session is seeded with.
async fn update_conversation(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<crate::types::UpdateConversation>,
) -> ApiResult<Json<Value>> {
    let conversation = state
        .db
        .update_conversation(&id, request)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!(conversation)))
}

async fn delete_conversation(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    state
        .db
        .delete_conversation(&id)
        .await
        .map_err(ApiError::not_found)?;
    Ok(Json(json!({ "deleted": true })))
}

async fn list_messages(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let messages = state
        .db
        .list_messages(&id)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "data": messages })))
}

async fn create_message(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<CreateMessage>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    register_message_blobs(&state, &request.content)
        .await
        .map_err(ApiError::bad_request)?;
    let message = state
        .db
        .create_message(&id, request)
        .await
        .map_err(ApiError::bad_request)?;
    Ok((StatusCode::CREATED, Json(json!(message))))
}

/// Tool-generated media is already present in the content-addressed blob
/// store, but unlike an upload it has not passed through `upload_blob`, which
/// normally creates the corresponding attachment row. Register references
/// before message indexing so `message_attachments` never points at a missing
/// attachment.
async fn register_message_blobs(state: &AppState, content: &Value) -> anyhow::Result<()> {
    let Value::Array(parts) = content else {
        return Ok(());
    };
    for part in parts {
        let Some(blob) = part.get("brazier_blob") else {
            continue;
        };
        let sha256 = blob
            .get("sha256")
            .and_then(Value::as_str)
            .context("brazier_blob missing sha256")?;
        let mime_type = blob
            .get("mime_type")
            .and_then(Value::as_str)
            .context("brazier_blob missing mime_type")?;
        let original_name = blob.get("name").and_then(Value::as_str);
        let path = blob_store::blob_path(&state.data_dir, sha256)?;
        let metadata = tokio::fs::metadata(&path)
            .await
            .with_context(|| format!("blob not found: {sha256}"))?;
        state
            .db
            .upsert_attachment(
                sha256,
                mime_type,
                i64::try_from(metadata.len()).context("blob is too large to index")?,
                original_name,
            )
            .await?;
    }
    Ok(())
}

/// Finalize a streamed message or relabel its status. Never deletes: a spoken
/// answer that was interrupted is still an answer in the chat.
async fn update_message(
    State(state): State<AppState>,
    Path((id, message_id)): Path<(String, String)>,
    Json(request): Json<crate::types::UpdateMessage>,
) -> ApiResult<Json<Value>> {
    let message = state
        .db
        .update_message(&id, &message_id, request)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!(message)))
}

async fn list_run_snapshots(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let runs = state
        .db
        .list_run_snapshots(&id)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "data": runs })))
}

async fn create_run_snapshot(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<CreateRunSnapshot>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    let snapshot = state
        .db
        .create_run_snapshot(&id, request)
        .await
        .map_err(ApiError::bad_request)?;
    Ok((StatusCode::CREATED, Json(json!(snapshot))))
}

#[derive(Debug, Deserialize)]
struct UploadBlobRequest {
    mime_type: String,
    data_base64: String,
    #[serde(default)]
    filename: Option<String>,
}

async fn upload_blob(
    State(state): State<AppState>,
    Json(request): Json<UploadBlobRequest>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(request.data_base64.trim())
        .map_err(|error| ApiError::bad_request(format!("invalid base64 payload: {error}")))?;
    let stored = blob_store::store_bytes(
        &state.data_dir,
        &bytes,
        &request.mime_type,
        request.filename.as_deref(),
    )
    .await
    .map_err(ApiError::bad_request)?;
    state
        .db
        .upsert_attachment(
            &stored.sha256,
            &stored.mime_type,
            stored.size_bytes as i64,
            stored.original_name.as_deref(),
        )
        .await
        .map_err(ApiError::internal)?;
    Ok((StatusCode::CREATED, Json(json!(stored))))
}

async fn get_blob(
    State(state): State<AppState>,
    Path(sha256): Path<String>,
) -> ApiResult<Response> {
    blob_store::validate_sha256(&sha256).map_err(ApiError::bad_request)?;
    let (bytes, mime_type) = blob_store::read_blob(&state.data_dir, &sha256)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(([(header::CONTENT_TYPE, mime_type)], bytes).into_response())
}

async fn huggingface_token_status(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    Ok(Json(json!({
        "configured": hf_auth::token_configured(&state.data_dir),
        "source": if std::env::var("HF_TOKEN").is_ok() || std::env::var("HUGGING_FACE_HUB_TOKEN").is_ok() {
            "environment"
        } else if hf_auth::token_file(&state.data_dir).is_file() {
            "stored"
        } else {
            "none"
        }
    })))
}

#[derive(Debug, Deserialize)]
struct HuggingFaceTokenRequest {
    token: String,
}

async fn set_huggingface_token(
    State(state): State<AppState>,
    Json(request): Json<HuggingFaceTokenRequest>,
) -> ApiResult<Json<Value>> {
    hf_auth::save_token(&state.data_dir, &request.token)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!({ "configured": true })))
}

async fn clear_huggingface_token(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    hf_auth::clear_token(&state.data_dir)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(
        json!({ "configured": hf_auth::token_configured(&state.data_dir) }),
    ))
}

#[derive(Debug, Deserialize)]
struct RemoteConnectionRequest {
    id: String,
    #[serde(default)]
    label: String,
    base_url: String,
    /// Omitted keeps whatever key is stored; an empty string clears it.
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default = "default_enabled")]
    enabled: bool,
    #[serde(default)]
    llama_cpp_compatible: bool,
}

fn default_enabled() -> bool {
    true
}

async fn list_remote_connections(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let connections: Vec<remote::PublicConnection> = remote::load(&state.data_dir)
        .iter()
        .map(remote::PublicConnection::from)
        .collect();
    Ok(Json(json!({ "object": "list", "data": connections })))
}

async fn save_remote_connection(
    State(state): State<AppState>,
    Json(request): Json<RemoteConnectionRequest>,
) -> ApiResult<Json<Value>> {
    remote::upsert(
        &state.data_dir,
        remote::StoredConnection {
            id: request.id,
            label: request.label,
            base_url: request.base_url,
            api_key: request.api_key,
            enabled: request.enabled,
            llama_cpp_compatible: request.llama_cpp_compatible,
        },
    )
    .await
    .map_err(ApiError::bad_request)?;
    // The model list now has different contents; nothing else knows that.
    state.invalidate_models_cache().await;
    list_remote_connections(State(state)).await
}

async fn delete_remote_connection(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    remote::remove(&state.data_dir, &id)
        .await
        .map_err(ApiError::bad_request)?;
    state.invalidate_models_cache().await;
    list_remote_connections(State(state)).await
}

/// Contact a configured server and report what it says it can serve.
///
/// Separate from saving on purpose: a connection that cannot be reached is
/// still worth keeping — the machine may be off — so failing to reach it is a
/// report, not a refusal to store it.
async fn test_remote_connection(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let connection = remote::find(&state.data_dir, &id)
        .ok_or_else(|| ApiError::bad_request(format!("no remote connection `{id}`")))?;
    match remote::fetch_model_names(&state.http, &connection).await {
        Ok(models) => Ok(Json(json!({ "reachable": true, "models": models }))),
        Err(error) => Ok(Json(json!({
            "reachable": false,
            "error": error.to_string(),
            "models": [],
        }))),
    }
}

async fn search_hugging_face(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> ApiResult<Json<Value>> {
    let models = hf::search(&state.http, &state.data_dir, query)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "data": models })))
}

#[derive(Debug, Deserialize)]
struct HubFilesQuery {
    revision: Option<String>,
}

async fn list_hub_files(
    State(state): State<AppState>,
    Path((repo_owner, repo_name)): Path<(String, String)>,
    Query(query): Query<HubFilesQuery>,
) -> ApiResult<Json<Value>> {
    let repo_id = format!("{repo_owner}/{repo_name}");
    let (files, preferred) = hf::list_gguf_files(
        &state.http,
        &state.data_dir,
        &repo_id,
        query.revision.as_deref(),
    )
    .await
    .map_err(ApiError::internal)?;
    Ok(Json(json!({
        "repo_id": repo_id,
        "data": files,
        "preferred_filename": preferred
    })))
}

async fn model_trust(
    State(state): State<AppState>,
    Path((repo_owner, repo_name)): Path<(String, String)>,
) -> ApiResult<Json<Value>> {
    let repo_id = format!("{repo_owner}/{repo_name}");
    let trust = hf::model_trust(&state.http, &state.data_dir, &repo_id)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!(trust)))
}

async fn model_description(
    State(state): State<AppState>,
    Path((repo_owner, repo_name)): Path<(String, String)>,
) -> ApiResult<Json<Value>> {
    let repo_id = format!("{repo_owner}/{repo_name}");
    let description = hf::model_description(&state.http, &state.data_dir, &repo_id)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!({ "description": description })))
}

async fn model_fork_hints(
    State(state): State<AppState>,
    Path((repo_owner, repo_name)): Path<(String, String)>,
) -> ApiResult<Json<Value>> {
    let repo_id = format!("{repo_owner}/{repo_name}");
    let fork_hints = fork_hints::hints_for_repo(&state.http, &state.data_dir, &repo_id)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(
        json!({ "repo_id": repo_id, "fork_hints": fork_hints }),
    ))
}

async fn list_download_jobs(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let jobs = state
        .db
        .list_download_jobs(30)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "data": jobs })))
}

async fn build_plan(Json(request): Json<BuildPlanRequest>) -> ApiResult<Json<Value>> {
    let plan = build_recipe::plan(request).map_err(ApiError::bad_request)?;
    Ok(Json(json!(plan)))
}

async fn engine_status(
    State(state): State<AppState>,
    Query(query): Query<EngineStatusQuery>,
) -> ApiResult<Json<Value>> {
    Ok(Json(
        state
            .runtime
            .engine_status(crate::engine::EngineStatusOptions {
                probe: query.probe.unwrap_or(false),
            })
            .await,
    ))
}

#[derive(Debug, Deserialize)]
struct EngineStatusQuery {
    probe: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ListRuntimesQuery {
    include_system: Option<bool>,
}

async fn list_runtimes(
    State(state): State<AppState>,
    Query(query): Query<ListRuntimesQuery>,
) -> Json<Value> {
    let include_system = query.include_system.unwrap_or(false);
    let active = state.runtime.active_runtimes().await;
    let entries = if !include_system {
        if let Some(cached) = state.runtimes_cache.lock().await.clone() {
            apply_active_flags(cached, &active)
        } else {
            let loaded = load_runtimes(&state.data_dir, &active, false).await;
            *state.runtimes_cache.lock().await = Some(loaded.clone());
            loaded
        }
    } else {
        load_runtimes(&state.data_dir, &active, true).await
    };
    Json(json!({
        "data": entries,
        "active_binary": active.llama.map(|path| path.display().to_string()),
        "active_mlx_lm_python": active.mlx_lm.map(|path| path.display().to_string()),
        "active_mlx_vlm_python": active.mlx_vlm.map(|path| path.display().to_string()),
        "active_whisper_binary": active.whisper.map(|path| path.display().to_string()),
        "active_streaming_asr_python": active
            .streaming_asr
            .map(|path| path.display().to_string()),
    }))
}

fn apply_active_flags(
    mut entries: Vec<runtimes::RuntimeEntry>,
    active: &runtimes::ActiveRuntimes,
) -> Vec<runtimes::RuntimeEntry> {
    for entry in &mut entries {
        let selected = match entry.engine.as_str() {
            "mlx-lm" => &active.mlx_lm,
            "mlx-vlm" => &active.mlx_vlm,
            "whisper.cpp" | "whisperkit" => &active.whisper,
            "streaming-asr" => &active.streaming_asr,
            "stable-diffusion.cpp" => &active.sdcpp,
            "personaplex" | "personaplex-mlx" => &active.voice,
            _ => &active.llama,
        };
        entry.active = selected.as_ref().is_some_and(|active_path| {
            std::path::Path::new(&entry.path)
                .canonicalize()
                .ok()
                .zip(active_path.canonicalize().ok())
                .is_some_and(|(left, right)| left == right)
                || active_path == std::path::Path::new(&entry.path)
        });
    }
    entries
}

async fn load_runtimes(
    data_dir: &std::path::Path,
    active: &runtimes::ActiveRuntimes,
    include_system: bool,
) -> Vec<runtimes::RuntimeEntry> {
    let data_dir = data_dir.to_path_buf();
    let active = active.clone();
    tokio::task::spawn_blocking(move || {
        runtimes::list(
            &data_dir,
            &active,
            std::env::var("PATH").ok().as_deref(),
            include_system,
        )
    })
    .await
    .unwrap_or_default()
}

async fn hardware() -> Json<Value> {
    Json(json!(crate::hardware::detect()))
}

#[derive(Debug, Deserialize, Default)]
struct ToolchainNeedsQuery {
    custom_runtimes: Option<bool>,
    voice: Option<bool>,
    computer_use: Option<bool>,
    video: Option<bool>,
}

impl ToolchainNeedsQuery {
    fn into_needs(self) -> Option<toolchain_hints::ToolchainNeeds> {
        if self.custom_runtimes.is_none()
            && self.voice.is_none()
            && self.computer_use.is_none()
            && self.video.is_none()
        {
            return None;
        }
        Some(toolchain_hints::ToolchainNeeds {
            custom_runtimes: self.custom_runtimes.unwrap_or(false),
            voice: self.voice.unwrap_or(false),
            computer_use: self.computer_use.unwrap_or(false),
            video: self.video.unwrap_or(false),
        })
    }
}

async fn toolchain_status(Query(needs): Query<ToolchainNeedsQuery>) -> Json<Value> {
    Json(toolchain_hints::toolchain_status_for(needs.into_needs()))
}

#[derive(Debug, Deserialize)]
struct ToolchainSetupRequest {
    #[serde(default)]
    custom_runtimes: bool,
    #[serde(default)]
    voice: bool,
    #[serde(default)]
    computer_use: bool,
    #[serde(default)]
    video: bool,
}

/// Install the fixed, intent-selected Homebrew prerequisites on macOS.
///
/// This deliberately does not accept arbitrary commands or formulas. The
/// action is exposed behind an explicit first-run button and is macOS-only;
/// other platforms continue to receive distro-specific install hints.
async fn setup_toolchain(Json(request): Json<ToolchainSetupRequest>) -> ApiResult<Json<Value>> {
    let os = toolchain_hints::detect_os();
    if !matches!(os.family, toolchain_hints::OsFamily::Macos) {
        return Err(ApiError::bad_request(
            "automatic host setup is currently available on macOS only",
        ));
    }

    let needs = toolchain_hints::ToolchainNeeds {
        custom_runtimes: request.custom_runtimes,
        voice: request.voice,
        computer_use: request.computer_use,
        video: request.video,
    };
    let required = toolchain_hints::required_tool_ids(needs);
    let mut output = Vec::new();
    let mut brew = toolchain_hints::resolve_command("brew");

    if brew.is_none() {
        let install = tokio::process::Command::new("/bin/bash")
            .args([
                "-c",
                "/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"",
            ])
            .output()
            .await
            .map_err(|error| ApiError::internal(anyhow::anyhow!(error)))?;
        output.push(String::from_utf8_lossy(&install.stdout).into_owned());
        output.push(String::from_utf8_lossy(&install.stderr).into_owned());
        if !install.status.success() {
            return Err(ApiError::bad_request(format!(
                "Homebrew installation failed: {}",
                output.join("\n").trim()
            )));
        }
        brew = toolchain_hints::resolve_command("brew");
    }

    let Some(brew) = brew else {
        return Err(ApiError::bad_request(
            "Homebrew finished installing but Brazier could not find `brew`; restart Brazier and try again",
        ));
    };

    let initial_status = toolchain_hints::toolchain_status_for(Some(needs));
    let formulas: Vec<&str> = required
        .iter()
        .copied()
        .filter(|id| matches!(*id, "git" | "cmake" | "uv" | "ffmpeg"))
        .filter(|id| {
            initial_status["tools"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|tool| tool["id"].as_str() == Some(id) && tool["available"] == false)
        })
        .collect();

    if !formulas.is_empty() {
        let install = tokio::process::Command::new(&brew)
            .arg("install")
            .args(&formulas)
            .output()
            .await
            .map_err(|error| ApiError::internal(anyhow::anyhow!(error)))?;
        output.push(String::from_utf8_lossy(&install.stdout).into_owned());
        output.push(String::from_utf8_lossy(&install.stderr).into_owned());
        if !install.status.success() {
            return Err(ApiError::bad_request(format!(
                "Homebrew dependency setup failed: {}",
                output.join("\n").trim()
            )));
        }
    }

    // Apple ships the C/C++ compiler and headers through Command Line Tools,
    // not Homebrew. Opening Apple's installer completes this one non-formula
    // prerequisite when a source build was requested.
    if needs.custom_runtimes {
        let missing_cpp = initial_status["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|tool| tool["id"].as_str() == Some("cpp") && tool["available"] == false);
        if missing_cpp {
            let command = tokio::process::Command::new("/usr/bin/xcode-select")
                .arg("--install")
                .output()
                .await
                .map_err(|error| ApiError::internal(anyhow::anyhow!(error)))?;
            output.push(String::from_utf8_lossy(&command.stdout).into_owned());
            output.push(String::from_utf8_lossy(&command.stderr).into_owned());
        }
    }

    Ok(Json(json!({
        "status": toolchain_hints::toolchain_status_for(Some(needs)),
        "output": output.join("\n").trim(),
    })))
}

async fn runtime_settings(State(state): State<AppState>) -> Json<Value> {
    Json(json!(state.runtime.settings().await))
}

const WELCOME_PREFERENCE_KEY: &str = "welcome";

#[derive(Debug, Deserialize)]
struct UpdateWelcomePreference {
    completed: bool,
}

async fn welcome_preference(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let completed = state
        .db
        .application_preference(WELCOME_PREFERENCE_KEY)
        .await
        .map_err(ApiError::internal)?
        .and_then(|value| value["completed"].as_bool())
        .unwrap_or(false);
    Ok(Json(json!({ "completed": completed })))
}

async fn update_welcome_preference(
    State(state): State<AppState>,
    Json(preference): Json<UpdateWelcomePreference>,
) -> ApiResult<Json<Value>> {
    state
        .db
        .set_application_preference(
            WELCOME_PREFERENCE_KEY,
            &json!({ "completed": preference.completed }),
        )
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "completed": preference.completed })))
}

const AGENT_PREFERENCE_KEY: &str = "agent";

/// Keep preferences and sessions created by removed agent adapters usable.
/// OMP had the broadest tool surface, so Powerful is the closest current mode;
/// the old Pi id maps to the current Simple mode.
fn migrate_legacy_agent_runtime_id(runtime_id: &str) -> String {
    match runtime_id.trim() {
        "omp" => crate::agent_types::AGENT_RUNTIME_POWERFUL.to_owned(),
        "pi" => crate::agent_types::AGENT_RUNTIME_SIMPLE.to_owned(),
        value => value.to_owned(),
    }
}

#[derive(Debug, Deserialize)]
struct UpdateAgentPreference {
    default_runtime_id: String,
    #[serde(default)]
    power_tools: Vec<String>,
}

async fn load_default_agent_runtime_id(state: &AppState) -> ApiResult<String> {
    let stored = state
        .db
        .application_preference(AGENT_PREFERENCE_KEY)
        .await
        .map_err(ApiError::internal)?
        .and_then(|value| value["default_runtime_id"].as_str().map(str::to_owned))
        .unwrap_or_else(|| crate::agent_types::DEFAULT_AGENT_RUNTIME_ID.to_owned());
    let migrated = migrate_legacy_agent_runtime_id(&stored);
    let known = agent_runtime_catalog()
        .iter()
        .any(|entry| entry["id"].as_str() == Some(migrated.as_str()));
    Ok(if known {
        migrated
    } else {
        crate::agent_types::DEFAULT_AGENT_RUNTIME_ID.to_owned()
    })
}

async fn load_enabled_power_tools(state: &AppState) -> ApiResult<Vec<String>> {
    let stored = state
        .db
        .application_preference(AGENT_PREFERENCE_KEY)
        .await
        .map_err(ApiError::internal)?;
    // Nothing configured yet: Powerful mode starts with every power tool on.
    // An explicit list (even an empty one) is respected as-is.
    let names = match stored {
        None => crate::agent_tools::power_tool_names(),
        Some(value) => match value.get("power_tools") {
            Some(array) => array
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|entry| entry.as_str().map(str::to_owned))
                .collect(),
            None => crate::agent_tools::power_tool_names(),
        },
    };
    Ok(names)
}

async fn agent_preference(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let stored = state
        .db
        .application_preference(AGENT_PREFERENCE_KEY)
        .await
        .map_err(ApiError::internal)?;
    let default_runtime_id = stored
        .as_ref()
        .and_then(|value| value["default_runtime_id"].as_str())
        .map(migrate_legacy_agent_runtime_id)
        .filter(|runtime_id| {
            agent_runtime_catalog()
                .iter()
                .any(|entry| entry["id"].as_str() == Some(runtime_id.as_str()))
        })
        .unwrap_or_else(|| crate::agent_types::DEFAULT_AGENT_RUNTIME_ID.to_owned());
    let power_tools = load_enabled_power_tools(&state).await?;
    Ok(Json(json!({
        "default_runtime_id": default_runtime_id,
        "power_tools": power_tools,
    })))
}

async fn update_agent_preference(
    State(state): State<AppState>,
    Json(preference): Json<UpdateAgentPreference>,
) -> ApiResult<Json<Value>> {
    let runtime_id = migrate_legacy_agent_runtime_id(&preference.default_runtime_id);
    let catalog = agent_runtime_catalog();
    let entry = catalog
        .iter()
        .find(|entry| entry["id"].as_str() == Some(runtime_id.as_str()))
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "unknown agent runtime `{runtime_id}`. Available: {}",
                catalog
                    .iter()
                    .filter_map(|entry| entry["id"].as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
    if entry["available"].as_bool() == Some(false) {
        let reason = entry["unavailable_reason"]
            .as_str()
            .unwrap_or("that runtime is not available on this machine");
        return Err(ApiError::bad_request(format!(
            "cannot default to agent runtime `{runtime_id}`: {reason}"
        )));
    }
    let known_power: std::collections::HashSet<String> =
        crate::agent_tools::power_tool_names().into_iter().collect();
    let mut power_tools: Vec<String> = preference
        .power_tools
        .into_iter()
        .filter(|name| known_power.contains(name))
        .collect();
    power_tools.sort();
    power_tools.dedup();
    state
        .db
        .set_application_preference(
            AGENT_PREFERENCE_KEY,
            &json!({
                "default_runtime_id": runtime_id,
                "power_tools": power_tools,
            }),
        )
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(
        json!({ "default_runtime_id": runtime_id, "power_tools": power_tools }),
    ))
}

fn agent_runtime_catalog() -> Vec<Value> {
    vec![
        json!({
            "id": crate::agent_types::AGENT_RUNTIME_SIMPLE,
            "name": "Simple",
            "adapter_api_version": 1,
            "available": true,
            "trust": "broker",
            "capabilities": {
                "streaming": true,
                "tool_calls": true,
                "compaction": true,
                "cancellation": true,
                "session_restore": true,
            }
        }),
        json!({
            "id": crate::agent_types::AGENT_RUNTIME_POWERFUL,
            "name": "Powerful",
            "adapter_api_version": 1,
            "available": true,
            "trust": "broker",
            "capabilities": {
                "streaming": true,
                "tool_calls": true,
                "compaction": true,
                "cancellation": true,
                "session_restore": true,
            }
        }),
    ]
}

fn resolve_agent_runtime_id(requested: Option<String>, default_id: &str) -> String {
    requested
        .map(|value| migrate_legacy_agent_runtime_id(&value))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default_id.to_owned())
}

fn validate_agent_runtime_id(runtime_id: &str) -> ApiResult<Value> {
    let catalog = agent_runtime_catalog();
    let entry = catalog
        .into_iter()
        .find(|entry| entry["id"].as_str() == Some(runtime_id))
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "unknown agent runtime `{runtime_id}`. Available: simple, powerful"
            ))
        })?;
    if entry["available"].as_bool() == Some(false) {
        let reason = entry["unavailable_reason"]
            .as_str()
            .unwrap_or("that runtime is not available on this machine");
        return Err(ApiError::bad_request(format!(
            "agent runtime `{runtime_id}` is unavailable: {reason}"
        )));
    }
    Ok(entry)
}

const WORKSPACE_PREFERENCE_KEY: &str = "workspace";

async fn workspace_preference(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let modes = state
        .db
        .application_preference(WORKSPACE_PREFERENCE_KEY)
        .await
        .map_err(ApiError::internal)?
        .and_then(|value| {
            serde_json::from_value::<crate::computer_types::WorkspaceModesPreference>(
                value.get("modes").cloned().unwrap_or(value),
            )
            .ok()
        })
        .unwrap_or_default()
        .normalize();
    Ok(Json(json!({ "modes": modes })))
}

#[derive(Debug, Deserialize)]
struct UpdateWorkspacePreference {
    modes: crate::computer_types::WorkspaceModesPreference,
}

async fn update_workspace_preference(
    State(state): State<AppState>,
    Json(preference): Json<UpdateWorkspacePreference>,
) -> ApiResult<Json<Value>> {
    let modes = preference.modes.normalize();
    state
        .db
        .set_application_preference(WORKSPACE_PREFERENCE_KEY, &json!({ "modes": modes }))
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "modes": modes })))
}

pub const COMPUTER_PREFERENCE_KEY: &str = "computer";

/// How many screenshots a computer-use trajectory keeps by default. Fara's
/// reference agent retains the most recent three; more costs context fast.
pub const DEFAULT_MAX_SCREENSHOTS_KEPT: u32 = 3;
const MIN_SCREENSHOTS_KEPT: u32 = 1;
const MAX_SCREENSHOTS_KEPT: u32 = 20;

fn computer_settle_delay(value: Option<&Value>) -> u64 {
    value
        .and_then(|value| value["action_settle_delay_ms"].as_u64())
        .unwrap_or(crate::computer_exec::DEFAULT_ACTION_SETTLE_DELAY_MS)
        .min(crate::computer_exec::MAX_ACTION_SETTLE_DELAY_MS)
}

fn computer_screenshots_kept(value: Option<&Value>) -> u32 {
    value
        .and_then(|value| value["max_screenshots_kept"].as_u64())
        .unwrap_or(u64::from(DEFAULT_MAX_SCREENSHOTS_KEPT))
        .clamp(
            u64::from(MIN_SCREENSHOTS_KEPT),
            u64::from(MAX_SCREENSHOTS_KEPT),
        ) as u32
}

async fn computer_preference(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let stored = state
        .db
        .application_preference(COMPUTER_PREFERENCE_KEY)
        .await
        .map_err(ApiError::internal)?;
    let action_settle_delay_ms = computer_settle_delay(stored.as_ref());
    state
        .computer_broker
        .set_action_settle_delay_ms(action_settle_delay_ms);
    Ok(Json(json!({
        "action_settle_delay_ms": action_settle_delay_ms,
        "max_screenshots_kept": computer_screenshots_kept(stored.as_ref()),
    })))
}

#[derive(Debug, Deserialize)]
struct UpdateComputerPreference {
    action_settle_delay_ms: u64,
    #[serde(default = "default_max_screenshots_kept")]
    max_screenshots_kept: u32,
}

fn default_max_screenshots_kept() -> u32 {
    DEFAULT_MAX_SCREENSHOTS_KEPT
}

async fn update_computer_preference(
    State(state): State<AppState>,
    Json(preference): Json<UpdateComputerPreference>,
) -> ApiResult<Json<Value>> {
    if preference.action_settle_delay_ms > crate::computer_exec::MAX_ACTION_SETTLE_DELAY_MS {
        return Err(ApiError::bad_request(format!(
            "Computer action settle delay must be between 0 and {} milliseconds.",
            crate::computer_exec::MAX_ACTION_SETTLE_DELAY_MS
        )));
    }
    let max_screenshots_kept = preference
        .max_screenshots_kept
        .clamp(MIN_SCREENSHOTS_KEPT, MAX_SCREENSHOTS_KEPT);
    state
        .db
        .set_application_preference(
            COMPUTER_PREFERENCE_KEY,
            &json!({
                "action_settle_delay_ms": preference.action_settle_delay_ms,
                "max_screenshots_kept": max_screenshots_kept,
            }),
        )
        .await
        .map_err(ApiError::internal)?;
    state
        .computer_broker
        .set_action_settle_delay_ms(preference.action_settle_delay_ms);
    Ok(Json(json!({
        "action_settle_delay_ms": preference.action_settle_delay_ms,
        "max_screenshots_kept": max_screenshots_kept,
    })))
}

async fn computer_os_permissions(State(state): State<AppState>) -> Json<Value> {
    Json(json!(state.computer_broker.os_permissions()))
}

async fn request_computer_os_permissions(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let status = state
        .computer_broker
        .request_os_permissions()
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!(status)))
}

async fn list_computer_sessions(State(state): State<AppState>) -> Json<Value> {
    Json(json!({ "sessions": state.computer_broker.list_sessions().await }))
}

#[derive(Debug, Deserialize)]
struct CreateComputerSession {
    title: Option<String>,
    #[serde(default)]
    target: Option<String>,
    model_id: Option<String>,
    permission_mode: Option<String>,
    viewport: Option<crate::computer_types::ComputerViewport>,
    /// Required for skip-permissions / allow-all. Desktop UI sets this after an
    /// explicit mode choice so a bare bearer token cannot silently elevate.
    #[serde(default)]
    confirm_elevated_permissions: bool,
}

/// Peer address when the server was started with connect-info; absent in
/// `oneshot` unit tests, which are treated as loopback.
struct ClientAddr(Option<SocketAddr>);

impl<S> FromRequestParts<S> for ClientAddr
where
    S: Send + Sync,
{
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        Ok(Self(
            parts
                .extensions
                .get::<ConnectInfo<SocketAddr>>()
                .map(|ConnectInfo(addr)| *addr),
        ))
    }
}

fn client_is_loopback(client: &ClientAddr) -> bool {
    client.0.map(|addr| addr.ip().is_loopback()).unwrap_or(true)
}

fn require_elevated_permission_step_up(
    elevated: bool,
    confirmed: bool,
    loopback: bool,
) -> ApiResult<()> {
    if !elevated {
        return Ok(());
    }
    if !confirmed {
        return Err(ApiError::bad_request(
            "elevated permission modes require confirm_elevated_permissions=true",
        ));
    }
    if !loopback {
        return Err(ApiError::bad_request(
            "elevated permission modes can only be set by a loopback client",
        ));
    }
    Ok(())
}

async fn create_computer_session(
    State(state): State<AppState>,
    client: ClientAddr,
    Json(body): Json<CreateComputerSession>,
) -> ApiResult<Json<Value>> {
    let target = match body.target.as_deref().unwrap_or("browser") {
        "desktop" => crate::computer_types::ComputerTarget::Desktop,
        _ => crate::computer_types::ComputerTarget::Browser,
    };
    let permission_mode = match body.permission_mode.as_deref().unwrap_or("ask") {
        "browser-only" => crate::computer_types::ComputerPermissionMode::BrowserOnly,
        "skip-permissions" => crate::computer_types::ComputerPermissionMode::SkipPermissions,
        "allow-all" => crate::computer_types::ComputerPermissionMode::AllowAll,
        _ => crate::computer_types::ComputerPermissionMode::Ask,
    };
    let elevated = matches!(
        permission_mode,
        crate::computer_types::ComputerPermissionMode::SkipPermissions
            | crate::computer_types::ComputerPermissionMode::AllowAll
    );
    require_elevated_permission_step_up(
        elevated,
        body.confirm_elevated_permissions,
        client_is_loopback(&client),
    )?;
    let session = state
        .computer_broker
        .create_session(
            body.title,
            target,
            body.model_id,
            permission_mode,
            body.viewport,
        )
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!(session)))
}

async fn get_computer_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let session = state
        .computer_broker
        .get_session(&id)
        .await
        .map_err(ApiError::not_found)?;
    Ok(Json(json!(session)))
}

#[derive(Debug, Deserialize)]
struct UpdateComputerSession {
    permission_mode: String,
    /// Required for skip-permissions / allow-all, mirroring session creation so
    /// a bare bearer token cannot silently elevate a live session.
    #[serde(default)]
    confirm_elevated_permissions: bool,
}

async fn update_computer_session(
    State(state): State<AppState>,
    client: ClientAddr,
    Path(id): Path<String>,
    Json(body): Json<UpdateComputerSession>,
) -> ApiResult<Json<Value>> {
    let permission_mode = match body.permission_mode.as_str() {
        "browser-only" => crate::computer_types::ComputerPermissionMode::BrowserOnly,
        "skip-permissions" => crate::computer_types::ComputerPermissionMode::SkipPermissions,
        "allow-all" => crate::computer_types::ComputerPermissionMode::AllowAll,
        _ => crate::computer_types::ComputerPermissionMode::Ask,
    };
    let elevated = matches!(
        permission_mode,
        crate::computer_types::ComputerPermissionMode::SkipPermissions
            | crate::computer_types::ComputerPermissionMode::AllowAll
    );
    require_elevated_permission_step_up(
        elevated,
        body.confirm_elevated_permissions,
        client_is_loopback(&client),
    )?;
    let session = state
        .computer_broker
        .update_session(&id, permission_mode)
        .await
        .map_err(ApiError::not_found)?;
    Ok(Json(json!(session)))
}

async fn delete_computer_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<StatusCode> {
    state
        .computer_broker
        .delete_session(&id)
        .await
        .map_err(ApiError::not_found)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn stop_computer_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<StatusCode> {
    state
        .computer_broker
        .stop(&id)
        .await
        .map_err(ApiError::not_found)?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Deserialize)]
struct ComputerSafetyAuthority {
    active: bool,
}

async fn set_computer_safety_authority(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<ComputerSafetyAuthority>,
) -> ApiResult<StatusCode> {
    if body.active && !crate::computer_exec::safety_overlay_is_ready(&state.data_dir) {
        return Err(ApiError::bad_request(
            "desktop safety authority requires the always-visible overlay and Esc emergency stop to be READY",
        ));
    }
    state
        .computer_broker
        .set_desktop_authority(&id, body.active)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn revoke_all_desktop_authority(State(state): State<AppState>) -> ApiResult<StatusCode> {
    crate::computer_exec::clear_safety_overlay_marker(&state.data_dir);
    state
        .computer_broker
        .revoke_all_desktop_authority()
        .await
        .map_err(ApiError::internal)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn list_computer_steps(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let steps = state
        .computer_broker
        .list_steps(&id)
        .await
        .map_err(ApiError::not_found)?;
    Ok(Json(json!({ "steps": steps })))
}

#[derive(Debug, Deserialize)]
struct AppendComputerStep {
    role: String,
    content: String,
    thought: Option<String>,
    action: Option<crate::computer_types::ComputerAction>,
    result: Option<crate::computer_types::ComputerActionResult>,
}

async fn append_computer_step(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<AppendComputerStep>,
) -> ApiResult<Json<Value>> {
    let step = state
        .computer_broker
        .append_step(
            &id,
            &body.role,
            &body.content,
            body.thought,
            body.action,
            body.result,
        )
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!(step)))
}

async fn computer_screenshot(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let result = state
        .computer_broker
        .screenshot(&id)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!(result)))
}

/// Non-recording viewport capture for live polling. The renderer streams this
/// while a browser session is idle so the page appears to render in real time;
/// unlike `computer_screenshot` it never writes a step into the trajectory.
async fn computer_preview(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let result = state
        .computer_broker
        .live_screenshot(&id)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!(result)))
}

/// Live browser screencast. Each SSE `data:` line is a base64 JPEG frame of
/// the current page; the renderer paints the newest frame as it arrives, which
/// is what makes the viewport feel like a real browser rather than a series of
/// stills. Nothing here is recorded into the trajectory.
async fn computer_stream(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Response> {
    let mut frames = state
        .computer_broker
        .subscribe_screencast(&id)
        .await
        .map_err(ApiError::bad_request)?;
    let events = stream! {
        loop {
            match frames.recv().await {
                Ok(data) => yield Ok::<Event, Infallible>(Event::default().data(data)),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        }
    };
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
        .into_response())
}

async fn computer_exec_action(
    State(state): State<AppState>,
    Json(body): Json<crate::computer_exec::ComputerExecRequest>,
) -> ApiResult<Json<Value>> {
    let result = state
        .computer_broker
        .execute(body)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!(result)))
}

#[derive(Debug, Deserialize)]
struct DecideComputerApproval {
    approve: bool,
}

async fn decide_computer_approval(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<DecideComputerApproval>,
) -> ApiResult<Json<Value>> {
    let result = state
        .computer_broker
        .decide_approval(&id, body.approve)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!({ "result": result })))
}

#[derive(Debug, Deserialize)]
struct ParseFaraRequest {
    text: String,
}

async fn parse_fara_output(Json(body): Json<ParseFaraRequest>) -> ApiResult<Json<Value>> {
    let parsed =
        crate::computer_fara::parse_fara_output(&body.text).map_err(ApiError::bad_request)?;
    Ok(Json(json!({
        "thought": parsed.thought,
        "actions": parsed.actions,
        "raw_tool_calls": parsed.raw_tool_calls,
    })))
}

async fn update_runtime_settings(
    State(state): State<AppState>,
    Json(settings): Json<crate::runtime_settings::RuntimeSettings>,
) -> ApiResult<Json<Value>> {
    let settings = state
        .runtime
        .update_settings(settings)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!(settings)))
}

async fn list_tools(State(state): State<AppState>) -> Json<Value> {
    Json(tool_registry::combined_catalog(&state.data_dir))
}

#[derive(Debug, Deserialize)]
struct McpServerUpsert {
    id: String,
    name: String,
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: std::collections::HashMap<String, String>,
    #[serde(default = "default_true")]
    enabled: bool,
}

fn default_true() -> bool {
    true
}

async fn list_mcp_servers(State(state): State<AppState>) -> Json<Value> {
    Json(mcp::catalog(&state.data_dir))
}

async fn create_mcp_server(
    State(state): State<AppState>,
    Json(request): Json<McpServerUpsert>,
) -> ApiResult<Json<Value>> {
    let mut config = mcp::load(&state.data_dir);
    if config.servers.iter().any(|server| server.id == request.id) {
        return Err(ApiError::bad_request(format!(
            "MCP server `{}` already exists",
            request.id
        )));
    }
    config.servers.push(mcp::McpServerConfig {
        id: request.id.clone(),
        name: request.name,
        command: request.command,
        args: request.args,
        env: request.env,
        enabled: request.enabled,
        tools: Vec::new(),
    });
    mcp::save(&state.data_dir, &config)
        .await
        .map_err(ApiError::internal)?;
    if request.enabled {
        let _ = mcp::refresh_tools(&state.data_dir, &request.id).await;
    }
    Ok(Json(json!({ "id": request.id, "status": "created" })))
}

async fn update_mcp_server(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<McpServerUpsert>,
) -> ApiResult<Json<Value>> {
    let mut config = mcp::load(&state.data_dir);
    let Some(server) = config.find_mut(&id) else {
        return Err(ApiError::bad_request(format!("unknown MCP server `{id}`")));
    };
    server.name = request.name;
    server.command = request.command;
    server.args = request.args;
    server.env = request.env;
    server.enabled = request.enabled;
    mcp::save(&state.data_dir, &config)
        .await
        .map_err(ApiError::internal)?;
    if request.enabled {
        let _ = mcp::refresh_tools(&state.data_dir, &id).await;
    }
    Ok(Json(json!({ "id": id, "status": "updated" })))
}

async fn delete_mcp_server(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let mut config = mcp::load(&state.data_dir);
    let before = config.servers.len();
    config.servers.retain(|server| server.id != id);
    if config.servers.len() == before {
        return Err(ApiError::bad_request(format!("unknown MCP server `{id}`")));
    }
    mcp::save(&state.data_dir, &config)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "id": id, "status": "deleted" })))
}

async fn refresh_mcp_server(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let tools = mcp::refresh_tools(&state.data_dir, &id)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!({ "id": id, "tools": tools })))
}

#[derive(Debug, Deserialize)]
struct RuntimeIdRequest {
    id: String,
}

#[derive(Debug, Default, Deserialize)]
struct EnsureLlamaRequest {
    #[serde(default)]
    target: Option<crate::runtime_settings::RuntimeTarget>,
    /// When true, download the latest managed release even if one is installed.
    #[serde(default)]
    force: bool,
}

async fn check_runtime_updates(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let data_dir = state.data_dir.clone();
    let updates = tokio::task::spawn_blocking(move || runtimes::check_source_updates(&data_dir))
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "data": updates })))
}

async fn activate_runtime(
    State(state): State<AppState>,
    Json(request): Json<RuntimeIdRequest>,
) -> ApiResult<Json<Value>> {
    let path_env = std::env::var("PATH").ok();
    let active = state.runtime.active_runtimes().await;
    let entry = runtimes::find(
        &state.data_dir,
        path_env.as_deref(),
        &request.id,
        false,
        &active,
    )
    .ok_or_else(|| ApiError::bad_request(format!("unknown runtime `{}`", request.id)))?;
    // Dispatch lives in the runtime, so this endpoint cannot drift out of step
    // with it and file a voice interpreter under llama-server.
    let path = state
        .runtime
        .activate_runtime_entry(&entry)
        .await
        .map_err(ApiError::bad_request)?;
    state.invalidate_runtimes_cache().await;
    Ok(Json(json!({
        "active_binary": path.display().to_string(),
        "engine": entry.engine,
        "id": entry.id
    })))
}

async fn deactivate_runtime(
    State(state): State<AppState>,
    Json(request): Json<RuntimeIdRequest>,
) -> ApiResult<Json<Value>> {
    let path_env = std::env::var("PATH").ok();
    let active = state.runtime.active_runtimes().await;
    let entry = runtimes::find(
        &state.data_dir,
        path_env.as_deref(),
        &request.id,
        false,
        &active,
    )
    .ok_or_else(|| ApiError::bad_request(format!("unknown runtime `{}`", request.id)))?;
    if !entry.active {
        return Err(ApiError::bad_request(format!(
            "runtime `{}` is not active",
            request.id
        )));
    }
    state
        .runtime
        .deactivate_runtime_entry(&entry)
        .await
        .map_err(ApiError::bad_request)?;
    state.invalidate_runtimes_cache().await;
    Ok(Json(json!({ "id": entry.id, "deactivated": true })))
}

async fn delete_runtime(
    State(state): State<AppState>,
    Json(request): Json<RuntimeIdRequest>,
) -> ApiResult<Json<Value>> {
    let removed = runtimes::delete(&state.data_dir, &request.id).map_err(ApiError::bad_request)?;
    state
        .runtime
        .release_runtime(&removed)
        .await
        .map_err(ApiError::internal)?;
    state.invalidate_runtimes_cache().await;
    Ok(Json(json!({ "deleted": request.id })))
}

#[derive(Debug, Deserialize)]
struct DeleteModelRequest {
    model_id: String,
}

async fn delete_local_model(
    State(state): State<AppState>,
    Json(request): Json<DeleteModelRequest>,
) -> ApiResult<Json<Value>> {
    let settings = state.runtime.settings().await;
    let extra_paths: Vec<PathBuf> = settings
        .extra_model_library_paths
        .iter()
        .map(PathBuf::from)
        .collect();
    let _path = models_store::path_for_model_id(&state.data_dir, &request.model_id, &extra_paths)
        .map_err(ApiError::bad_request)?;
    state.runtime.release_model(&request.model_id).await;
    models_store::delete_model(&state.data_dir, &request.model_id, &extra_paths)
        .map_err(ApiError::bad_request)?;
    state.invalidate_models_cache().await;
    Ok(Json(json!({ "deleted": request.model_id })))
}

async fn model_bindings_list(State(state): State<AppState>) -> Json<Value> {
    let bindings = model_bindings::load(&state.data_dir);
    Json(json!({ "bindings": bindings.bindings }))
}

#[derive(Debug, Deserialize)]
struct UpdateModelBindingRequest {
    model_id: String,
    #[serde(default)]
    runtime_id: Option<String>,
}

async fn update_model_binding(
    State(state): State<AppState>,
    Json(request): Json<UpdateModelBindingRequest>,
) -> ApiResult<Json<Value>> {
    let bindings = if let Some(runtime_id) = request.runtime_id.filter(|id| !id.is_empty()) {
        state
            .runtime
            .activate_runtime_by_id(&runtime_id)
            .await
            .map_err(ApiError::bad_request)?;
        state.invalidate_runtimes_cache().await;
        model_bindings::set_binding(&state.data_dir, &request.model_id, &runtime_id)
            .await
            .map_err(ApiError::internal)?
    } else {
        model_bindings::clear_binding(&state.data_dir, &request.model_id)
            .await
            .map_err(ApiError::internal)?
    };
    Ok(Json(json!({
        "model_id": request.model_id,
        "bindings": bindings.bindings
    })))
}

/// Advanced configuration for every model that has any, plus the kind each
/// installed model takes so the interface knows which fields to offer.
async fn model_settings_list(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let store = model_settings::load(&state.data_dir);
    let kinds: std::collections::BTreeMap<String, &str> = state
        .runtime
        .models()
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|model| {
            let kind = model_settings::kind_for(&model.id);
            (model.id, kind.as_str())
        })
        .collect();
    Ok(Json(json!({ "models": store.models, "kinds": kinds })))
}

#[derive(Debug, Deserialize)]
struct UpdateModelSettingsRequest {
    model_id: String,
    profile: model_settings::ModelProfile,
}

/// Store one model's overrides.
///
/// A profile with nothing set is removed rather than saved, so "reset" and
/// "clear every field" reach the same state.
async fn update_model_settings(
    State(state): State<AppState>,
    Json(request): Json<UpdateModelSettingsRequest>,
) -> ApiResult<Json<Value>> {
    let store = model_settings::set_profile(&state.data_dir, &request.model_id, request.profile)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!({
        "model_id": request.model_id,
        "models": store.models,
    })))
}

#[derive(Debug, Deserialize)]
struct ResetModelSettingsRequest {
    model_id: String,
}

async fn reset_model_settings(
    State(state): State<AppState>,
    Json(request): Json<ResetModelSettingsRequest>,
) -> ApiResult<Json<Value>> {
    let store = model_settings::clear_profile(&state.data_dir, &request.model_id)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({
        "model_id": request.model_id,
        "models": store.models,
    })))
}

#[derive(Debug, Deserialize)]
struct ModelChatTemplateQuery {
    model_id: String,
}

/// Return the Jinja chat template embedded in a GGUF, when present.
async fn model_chat_template(
    State(state): State<AppState>,
    Query(query): Query<ModelChatTemplateQuery>,
) -> ApiResult<Json<Value>> {
    let settings = state.runtime.settings().await;
    let extra: Vec<PathBuf> = settings
        .extra_model_library_paths
        .iter()
        .map(PathBuf::from)
        .collect();
    let path = models_store::path_for_model_id(&state.data_dir, &query.model_id, &extra)
        .map_err(ApiError::bad_request)?;
    if !path.is_file() {
        return Err(ApiError::bad_request(format!(
            "model file not found for {}",
            query.model_id
        )));
    }
    // Only GGUFs carry tokenizer.chat_template metadata Brazier can read.
    if !path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        return Ok(Json(json!({
            "model_id": query.model_id,
            "chat_template": Value::Null,
            "source": "unsupported",
        })));
    }
    let template = crate::gguf_meta::read_chat_template(&path).map_err(ApiError::internal)?;
    Ok(Json(json!({
        "model_id": query.model_id,
        "chat_template": template,
        "source": if template.is_some() { "gguf" } else { "missing" },
    })))
}

/// Resolve one repository-named recommendation against the Hub.
///
/// The quantisation is chosen here rather than written into the catalogue
/// because it depends on both the machine and what the repository actually
/// publishes. A repository that cannot be reached is not an error — the
/// recommendation is still shown, with the reason it could not be sized.
async fn resolve_repo_recommendation(
    state: &AppState,
    entry: &recommendations::RepoRecommendation,
    memory_bytes: u64,
    weight_fraction: Option<f64>,
) -> Value {
    let mut resolved = json!({
        "id": entry.id,
        "label": entry.label,
        "repo_id": entry.repo_id,
        "summary": entry.summary,
    });
    if let Some(context) = entry.context_tokens {
        resolved["context_tokens"] = json!(context);
    }
    // A catalogue entry still waiting for a real repository should say so
    // rather than sending anyone to a 404.
    if entry.repo_id.starts_with("TODO") || entry.repo_id.contains("TODO") {
        resolved["unresolved"] = json!("This recommendation has no model set yet.");
        return resolved;
    }

    let files =
        match hf::list_repo_files(&state.http, &state.data_dir, &entry.repo_id, "main").await {
            Ok(files) => files,
            Err(error) => {
                resolved["unresolved"] = json!(format!(
                    "Could not reach Hugging Face to size this download: {error}"
                ));
                return resolved;
            }
        };
    let listing: Vec<(String, Option<u64>)> = files
        .into_iter()
        .map(|file| (file.path, file.size))
        .collect();

    let choice = match entry.quant.as_deref() {
        Some("by_memory") | None => recommendations::choose_quant_with_fraction(
            &listing,
            memory_bytes,
            weight_fraction.unwrap_or(recommendations::WEIGHT_MEMORY_FRACTION),
        ),
        Some(quant) => recommendations::find_quant(&listing, quant),
    };
    match choice {
        Some(choice) => {
            resolved["quant"] = json!(choice.quant);
            resolved["files"] = json!(choice.files);
            resolved["bytes"] = json!(choice.bytes);
            resolved["tight"] = json!(choice.tight);
        }
        None => {
            resolved["unresolved"] =
                json!("This repository publishes no GGUF weights Brazier can run.");
        }
    }

    // Recommendations need to surface a gated repository before the user
    // starts its download, particularly during first-run setup where Discover
    // (and its token field) has not been visited yet. Metadata failures should
    // not hide an otherwise usable recommendation.
    if let Ok(trust) = hf::model_trust(&state.http, &state.data_dir, &entry.repo_id).await {
        resolved["gated"] = json!(trust.gated);
    }

    // Vision projectors are optional companions, not part of the quant ladder.
    // Discover them for every recommendation instead of maintaining a fragile
    // per-model list. Prefer Q8 when a repository publishes several projectors.
    let mut discovered_companions: Vec<&str> = listing
        .iter()
        .map(|(path, _)| path.as_str())
        .filter(|path| {
            path.rsplit('/').next().is_some_and(|name| {
                let name = name.to_ascii_lowercase();
                name.ends_with(".gguf") && name.contains("mmproj")
            })
        })
        .collect();
    discovered_companions.sort_by_key(|path| (!path.to_ascii_lowercase().contains("q8"), *path));
    if let Some(companion) = discovered_companions.first() {
        resolved["companion_files"] = json!([companion]);
    }

    // Speculative draft weights are also optional companions, but they are
    // kept separate from projectors so clients can explain what they add.
    // Prefer the smallest same-repository draft: publishers commonly expose
    // both a bf16 reference and a much smaller runtime quant.
    let mut discovered_drafts: Vec<(&str, Option<u64>)> = listing
        .iter()
        .filter(|(path, _)| {
            path.rsplit('/').next().is_some_and(|name| {
                let name = name.to_ascii_lowercase();
                name.ends_with(".gguf")
                    && !name.contains("mmproj")
                    && (name.contains("dspark")
                        || name.contains("dflash")
                        || name.contains("draft"))
            })
        })
        .map(|(path, size)| (path.as_str(), *size))
        .collect();
    discovered_drafts.sort_by(|(left_path, left_size), (right_path, right_size)| {
        left_size
            .unwrap_or(u64::MAX)
            .cmp(&right_size.unwrap_or(u64::MAX))
            .then_with(|| left_path.cmp(right_path))
    });
    if let Some((draft, _)) = discovered_drafts.first() {
        resolved["draft_files"] = json!([draft]);
    }

    if !entry.draft_files.is_empty() && discovered_drafts.is_empty() {
        let published: std::collections::HashSet<&str> =
            listing.iter().map(|(path, _)| path.as_str()).collect();
        let mut drafts: Vec<&str> = Vec::new();
        let mut missing: Vec<&str> = Vec::new();
        for wanted in &entry.draft_files {
            if published.contains(wanted.as_str()) {
                drafts.push(wanted);
            } else {
                missing.push(wanted);
            }
        }
        if !drafts.is_empty() {
            resolved["draft_files"] = json!(drafts);
        }
        if !missing.is_empty() {
            resolved["unresolved_drafts"] = json!(format!(
                "Draft file(s) {} not published by this repository.",
                missing.join(", ")
            ));
        }
    }

    // Explicit companion files remain supported for catalogue overrides that
    // need a nonstandard projector name.
    if !entry.companion_files.is_empty() && discovered_companions.is_empty() {
        let published: std::collections::HashSet<&str> =
            listing.iter().map(|(path, _)| path.as_str()).collect();
        let mut companions: Vec<&str> = Vec::new();
        let mut missing: Vec<&str> = Vec::new();
        for wanted in &entry.companion_files {
            if published.contains(wanted.as_str()) {
                companions.push(wanted);
            } else {
                missing.push(wanted);
            }
        }
        if !companions.is_empty() {
            resolved["companion_files"] = json!(companions);
        }
        if !missing.is_empty() {
            resolved["unresolved_companions"] = json!(format!(
                "Companion file(s) {} not published by this repository.",
                missing.join(", ")
            ));
        }
    }
    resolved
}

/// Resolve a repository-backed recommendation against the available files.
async fn resolve_recommended_repo(
    state: &AppState,
    entry: &recommendations::RepoRecommendation,
    memory: u64,
    weight_fraction: Option<f64>,
) -> Value {
    resolve_repo_recommendation(state, entry, memory, weight_fraction).await
}

/// What to install on this machine, and whether any of it has changed.
async fn model_recommendations(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let hardware = crate::hardware::detect();
    let catalog = recommendations::catalog(&state.data_dir);
    let recorded = recommendations::load_state(&state.data_dir);

    let Some(memory) = crate::hardware::recommendation_memory_bytes(&hardware) else {
        return Ok(Json(json!({
            "memory_bytes": null,
            "tier_gb": null,
            "reason": "This machine did not report how much memory it has, so nothing can be recommended by size.",
            "categories": {},
            "voice": catalog.voice,
            "state": recorded,
            "swaps": [],
        })));
    };
    let Some(tier) = catalog.tier_for(memory) else {
        return Ok(Json(json!({
            "memory_bytes": memory,
            "tier_gb": null,
            "reason": format!(
                "{} GB is below the smallest tier Brazier has a recommendation for.",
                memory / (1024 * 1024 * 1024)
            ),
            "categories": {},
            "voice": catalog.voice,
            "state": recorded,
            "swaps": [],
        })));
    };

    let mut categories = serde_json::Map::new();
    if let Some(text) = tier.text.as_ref() {
        categories.insert(
            "text".into(),
            resolve_recommended_repo(&state, text, memory, None).await,
        );
    }
    if let Some(computer_use) = tier.computer_use.as_ref() {
        // Computer-use agents hold long screenshot trajectories; their quant
        // is chosen against a tighter weight budget that leaves room for the
        // KV cache the recommended context needs.
        categories.insert(
            "computer_use".into(),
            resolve_recommended_repo(
                &state,
                computer_use,
                memory,
                Some(recommendations::COMPUTER_USE_WEIGHT_FRACTION),
            )
            .await,
        );
    }
    if let Some(agent) = recommendations::resolved_agent(tier) {
        let mut resolved = resolve_repo_recommendation(&state, agent, memory, None).await;
        // When the tier's own agent model cannot run here, say why the chat
        // model is standing in rather than showing two identical cards with no
        // explanation.
        if let Some(note) = recommendations::agent_substitution_note(tier) {
            resolved["substituted"] = json!(note);
        }
        categories.insert("agent".into(), resolved);
    }
    let agent_options: Vec<Value> = futures::future::join_all(
        recommendations::resolved_agent_options(tier)
            .into_iter()
            .map(|agent| resolve_repo_recommendation(&state, agent, memory, None)),
    )
    .await;
    for (name, entry) in [
        ("image", tier.image.as_ref()),
        ("video", tier.video.as_ref()),
    ] {
        let Some(entry) = entry else { continue };
        let mut resolved = serde_json::to_value(entry).unwrap_or_else(|_| json!({}));
        // A bundle id that names nothing installable is a catalogue gap, and
        // showing it as a working button would waste a download attempt.
        let missing: Vec<String> = entry
            .parts
            .iter()
            .map(|part| part.bundle_id.clone())
            .chain(entry.bundle_id.clone())
            .filter(|id| sdcpp_catalog::find(&state.data_dir, id).is_none())
            .collect();
        if !missing.is_empty() {
            resolved["unresolved"] = json!(format!(
                "No installable bundle for {} yet.",
                missing.join(", ")
            ));
        }
        let mut bundle_ids = entry
            .parts
            .iter()
            .map(|part| part.bundle_id.as_str())
            .chain(entry.bundle_id.as_deref());
        resolved["gated"] = json!(bundle_ids.any(|id| {
            sdcpp_catalog::find(&state.data_dir, id).is_some_and(|bundle| bundle.gated())
        }));
        categories.insert(name.into(), resolved);
    }

    let swaps = recommendations::pending_swaps(&catalog, &recorded, tier);
    Ok(Json(json!({
        "memory_bytes": memory,
        "memory_source": if hardware.gpu_offload_memory_bytes.or(hardware.vram_bytes).is_some() { "vram" } else { "system" },
        "tier_gb": tier.min_gb,
        "categories": categories,
        "agent_options": agent_options,
        "voice": catalog.voice,
        "state": recorded,
        "swaps": swaps,
    })))
}

async fn recommendation_state(State(state): State<AppState>) -> Json<Value> {
    Json(json!(recommendations::load_state(&state.data_dir)))
}

#[derive(Debug, Deserialize)]
struct StartRecommendationSetupRequest {
    recommendation_id: String,
    categories: Vec<String>,
    works: Vec<crate::download_queue::QueuedWork>,
    required_bytes: u64,
    #[serde(default)]
    build: Option<builds::BuildRequest>,
}

fn available_disk_bytes(path: &std::path::Path) -> anyhow::Result<u64> {
    #[cfg(unix)]
    {
        let output = std::process::Command::new("df")
            .args(["-Pk", path.to_string_lossy().as_ref()])
            .output()
            .context("check free disk space")?;
        anyhow::ensure!(output.status.success(), "disk-space check failed");
        let listing = String::from_utf8_lossy(&output.stdout);
        let line = listing
            .lines()
            .nth(1)
            .context("disk-space check returned no filesystem")?;
        let fields: Vec<_> = line.split_whitespace().collect();
        let available_kib: u64 = fields
            .get(3)
            .context("disk-space check returned an unreadable filesystem")?
            .parse()
            .context("parse available disk space")?;
        Ok(available_kib.saturating_mul(1024))
    }
    #[cfg(not(unix))]
    anyhow::bail!("Brazier cannot verify free disk space on this platform")
}

/// Persist an onboarding setup before any transfer is enqueued. Repeating the
/// same request returns its live plan instead of creating duplicate downloads.
async fn start_recommendation_setup(
    State(state): State<AppState>,
    Json(request): Json<StartRecommendationSetupRequest>,
) -> ApiResult<Json<Value>> {
    if request.categories.is_empty() {
        return Err(ApiError::bad_request("choose at least one category"));
    }
    if request.works.is_empty() {
        return Err(ApiError::bad_request("setup has no installable work"));
    }
    if let Some(build) = request.build.as_ref()
        && build.engine == "llama.cpp"
    {
        builds::llama_cpp_build_preflight(
            build
                .target
                .unwrap_or(crate::runtime_settings::RuntimeTarget::Cpu),
        )
        .map_err(ApiError::bad_request)?;
    }
    let required_bytes = request.required_bytes.saturating_add(
        request
            .build
            .as_ref()
            .map(|_| 10 * 1024 * 1024 * 1024_u64)
            .unwrap_or(0),
    );
    let available = available_disk_bytes(&state.data_dir).map_err(ApiError::bad_request)?;
    if available < required_bytes {
        return Err(ApiError::bad_request(format!(
            "Not enough free disk space: this setup needs {} but only {} is available.",
            required_bytes, available
        )));
    }
    let mut recorded = recommendations::load_state(&state.data_dir);
    if let Some(existing) = recorded.setups.iter().find(|setup| {
        setup.recommendation_id == request.recommendation_id
            && setup.categories == request.categories
            && matches!(setup.status.as_str(), "pending" | "running" | "paused")
    }) {
        return Ok(Json(json!({ "setup": existing, "existing": true })));
    }
    let now = epoch_seconds();
    let mut steps: Vec<_> = request
        .works
        .iter()
        .map(|work| recommendations::RecommendationSetupStep {
            label: work.label(),
            kind: work.kind().to_owned(),
            payload: serde_json::to_value(work).unwrap_or(Value::Null),
            job_id: None,
            status: "pending".to_owned(),
        })
        .collect();
    if let Some(build) = &request.build {
        steps.push(recommendations::RecommendationSetupStep {
            label: format!("Build {}", build.engine),
            kind: "runtime-build".to_owned(),
            payload: serde_json::to_value(build).map_err(ApiError::internal)?,
            job_id: None,
            status: "pending".to_owned(),
        });
    }
    let mut setup = recommendations::RecommendationSetup {
        id: Uuid::new_v4().to_string(),
        recommendation_id: request.recommendation_id,
        categories: request.categories,
        status: "running".to_owned(),
        steps,
        error: None,
        created_at: now.clone(),
        updated_at: now,
    };
    for (step, work) in setup.steps.iter_mut().zip(request.works) {
        let payload = serde_json::to_string(&work).map_err(ApiError::internal)?;
        let job = state
            .db
            .create_queued_download_job(crate::db::QueuedDownloadJobInput {
                repo_id: &work.repo_id(),
                filename: &work.filename(),
                revision: &work.revision(),
                kind: work.kind(),
                payload: Some(&payload),
                label: Some(&work.label()),
                status: "pending",
            })
            .await
            .map_err(ApiError::internal)?;
        state
            .download_queue
            .enqueue(crate::download_queue::QueuedDownload {
                job_id: job.id.clone(),
                work,
            })
            .await
            .map_err(ApiError::internal)?;
        step.job_id = Some(job.id);
        step.status = "running".to_owned();
    }
    recorded.setups.push(setup.clone());
    recommendations::save_state(&state.data_dir, &recorded)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "setup": setup, "existing": false })))
}

/// Reconcile persisted plans from the durable activity rows. Polling this is
/// sufficient after a restart because downloads themselves already resume from
/// those rows and no browser-local state is involved.
async fn list_recommendation_setups(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let mut recorded = recommendations::load_state(&state.data_dir);
    let mut changed = false;
    let mut completed = Vec::new();
    let mut builds_to_start = Vec::new();
    for setup in &mut recorded.setups {
        if !matches!(setup.status.as_str(), "pending" | "running" | "paused") {
            continue;
        }
        let mut paused = false;
        let mut failed = None;
        let mut complete = true;
        for step in &mut setup.steps {
            let Some(job_id) = step.job_id.as_deref() else {
                complete = false;
                continue;
            };
            let Ok(job) = state.db.get_download_job_public(job_id).await else {
                complete = false;
                continue;
            };
            step.status = job.status.clone();
            match job.status.as_str() {
                "completed" => {}
                "paused" => {
                    paused = true;
                    complete = false;
                }
                "failed" | "cancelled" => {
                    failed = Some(job.error.unwrap_or(job.status));
                    complete = false;
                }
                _ => complete = false,
            }
        }
        let next = if let Some(error) = failed.as_ref() {
            setup.error = Some(error.clone());
            "failed"
        } else if paused {
            "paused"
        } else if complete {
            "completed"
        } else {
            "running"
        };
        if setup.status != next {
            setup.status = next.to_owned();
            setup.updated_at = epoch_seconds();
            changed = true;
        }
        if setup.status == "completed" {
            completed.push((setup.categories.clone(), setup.recommendation_id.clone()));
            changed = true;
        } else if failed.is_none()
            && !paused
            && setup
                .steps
                .iter()
                .filter(|step| step.kind != "runtime-build")
                .all(|step| step.status == "completed")
        {
            for (step_index, step) in setup.steps.iter().enumerate() {
                if step.kind == "runtime-build"
                    && step.job_id.is_none()
                    && let Ok(request) =
                        serde_json::from_value::<builds::BuildRequest>(step.payload.clone())
                {
                    builds_to_start.push((setup.id.clone(), step_index, request));
                }
            }
        }
    }
    for (setup_id, step_index, request) in builds_to_start {
        match start_setup_build(&state, request).await {
            Ok(job_id) => {
                if let Some(setup) = recorded
                    .setups
                    .iter_mut()
                    .find(|setup| setup.id == setup_id)
                    && let Some(step) = setup.steps.get_mut(step_index)
                {
                    step.job_id = Some(job_id);
                    step.status = "running".to_owned();
                    setup.updated_at = epoch_seconds();
                    changed = true;
                }
            }
            Err(error) => {
                if let Some(setup) = recorded
                    .setups
                    .iter_mut()
                    .find(|setup| setup.id == setup_id)
                {
                    setup.status = "failed".to_owned();
                    setup.error = Some(error.message);
                    changed = true;
                }
            }
        }
    }
    for (categories, recommendation_id) in completed {
        for category in categories {
            recommendations::record_install(
                &mut recorded,
                category,
                recommendation_id.clone(),
                None,
                epoch_seconds(),
            );
        }
    }
    if changed {
        recommendations::save_state(&state.data_dir, &recorded)
            .await
            .map_err(ApiError::internal)?;
    }
    Ok(Json(json!({ "data": recorded.setups })))
}

/// Cancel every unfinished child of a recommendation plan. Completed files are
/// deliberately retained so a later setup can reuse them safely.
async fn cancel_recommendation_setup(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let mut recorded = recommendations::load_state(&state.data_dir);
    let setup = recorded
        .setups
        .iter_mut()
        .find(|setup| setup.id == id)
        .ok_or_else(|| ApiError::bad_request("no such recommendation setup"))?;
    if !matches!(setup.status.as_str(), "pending" | "running" | "paused") {
        return Err(ApiError::bad_request(
            "that recommendation setup is already settled",
        ));
    }
    for step in &mut setup.steps {
        let Some(job_id) = step.job_id.as_deref() else {
            continue;
        };
        if let Ok(job) = state.db.get_download_job_public(job_id).await
            && matches!(job.status.as_str(), "pending" | "downloading" | "paused")
        {
            if job.kind == "runtime-build" {
                if let Some(build_id) = job.payload.as_deref() {
                    state.active_builds.cancel(build_id);
                }
            } else {
                state.active_downloads.cancel(job_id);
            }
            let _ = state.db.cancel_download_job(job_id).await;
            step.status = "cancelled".to_owned();
        }
    }
    setup.status = "cancelled".to_owned();
    setup.updated_at = epoch_seconds();
    recommendations::save_state(&state.data_dir, &recorded)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "cancelled": id })))
}

async fn start_setup_build(state: &AppState, request: builds::BuildRequest) -> ApiResult<String> {
    let label = format!("Build {}", request.engine);
    let job = state
        .db
        .create_queued_download_job(crate::db::QueuedDownloadJobInput {
            repo_id: &request.repository,
            filename: "runtime source build",
            revision: &request.revision,
            kind: "runtime-build",
            payload: None,
            label: Some(&label),
            status: "pending",
        })
        .await
        .map_err(ApiError::internal)?;
    let job_id = job.id.clone();
    let db = state.db.clone();
    let active_builds = state.active_builds.clone();
    let build_slots = state.build_slots.clone();
    let data_dir = state.data_dir.clone();
    let runtime = state.runtime.clone();
    tokio::spawn(async move {
        let _ = db
            .update_download_job_message(&job_id, "Waiting for the build slot")
            .await;
        let Ok(slot) = build_slots.acquire_owned().await else {
            let _ = db.fail_download_job(&job_id, "build queue is closed").await;
            return;
        };
        if db
            .get_download_job_public(&job_id)
            .await
            .is_ok_and(|job| job.status == "cancelled")
        {
            return;
        }
        if db.start_download_job(&job_id).await.is_err() {
            // Cancellation may win between the durable-state check above and
            // this transition. Never run native build commands after that.
            return;
        }
        let progress_db = db.clone();
        let progress_id = job_id.clone();
        let result = builds::run_build_with_progress(
            &data_dir,
            request,
            &active_builds,
            Box::new(move |event| {
                let db = progress_db.clone();
                let id = progress_id.clone();
                if let Some(build_id) = event
                    .result
                    .as_ref()
                    .and_then(|value| value.get("build_id"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                {
                    tokio::spawn(async move {
                        let _ = db.set_download_job_payload(&id, &build_id).await;
                    });
                } else if let Some(percent) = event.percent {
                    tokio::spawn(async move {
                        let _ = db
                            .update_download_job_progress(&id, percent.round() as u64, Some(100))
                            .await;
                    });
                } else if event.phase != "log"
                    && let Some(message) = event.message
                {
                    tokio::spawn(async move {
                        let _ = db.update_download_job_message(&id, &message).await;
                    });
                }
            }),
        )
        .await;
        match result {
            Ok(binary) => {
                if let Err(error) = db.complete_download_job(&job_id, "", 100).await {
                    tracing::warn!(
                        job_id = %job_id,
                        error = %error,
                        "build finished after its durable job was settled"
                    );
                    return;
                }
                let active = runtime.active_runtimes().await;
                if let Some(entry) = runtimes::list(&data_dir, &active, None, false)
                    .into_iter()
                    .find(|entry| entry.path == binary.display().to_string())
                {
                    let _ = runtime.activate_runtime_entry(&entry).await;
                }
            }
            Err(report) => {
                let _ = db.fail_download_job(&job_id, &report.message).await;
            }
        }
        drop(slot);
    });
    Ok(job.id)
}

#[derive(Debug, Deserialize)]
struct UpdateRecommendationStateRequest {
    /// Stop mentioning changed recommendations entirely.
    #[serde(default)]
    suppressed: Option<bool>,
    /// A recommendation id that was offered as a swap and declined.
    #[serde(default)]
    dismiss: Option<String>,
}

async fn update_recommendation_state(
    State(state): State<AppState>,
    Json(request): Json<UpdateRecommendationStateRequest>,
) -> ApiResult<Json<Value>> {
    let mut recorded = recommendations::load_state(&state.data_dir);
    if let Some(suppressed) = request.suppressed {
        recorded.suppressed = suppressed;
    }
    if let Some(dismiss) = request.dismiss
        && !recorded.dismissed.contains(&dismiss)
    {
        recorded.dismissed.push(dismiss);
    }
    recommendations::save_state(&state.data_dir, &recorded)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!(recorded)))
}

#[derive(Debug, Deserialize)]
struct RecordRecommendationInstallRequest {
    /// `text`, `agent`, `image`, `video`, or `voice`.
    category: String,
    recommendation_id: String,
    #[serde(default)]
    model_id: Option<String>,
}

/// Record that a category was set up from a recommendation.
///
/// Only categories recorded here are ever mentioned again when the
/// recommendation changes; a model chosen deliberately from Discover is nobody's
/// business to second-guess.
async fn record_recommendation_install(
    State(state): State<AppState>,
    Json(request): Json<RecordRecommendationInstallRequest>,
) -> ApiResult<Json<Value>> {
    let mut recorded = recommendations::load_state(&state.data_dir);
    recommendations::record_install(
        &mut recorded,
        request.category.clone(),
        request.recommendation_id.clone(),
        request.model_id,
        epoch_seconds(),
    );
    recommendations::save_state(&state.data_dir, &recorded)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!(recorded)))
}

/// Seconds since the Unix epoch, as a string.
///
/// The same shape conversation exports already use, so no date library is
/// pulled in for one field.
fn epoch_seconds() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format!("@{now}")
}

async fn list_adapters(State(state): State<AppState>) -> Json<Value> {
    Json(json!({ "data": adapters::list(&state.data_dir) }))
}

#[derive(Debug, Deserialize)]
struct RegisterAdapterRequest {
    kind: String,
    path: String,
    #[serde(default)]
    name: Option<String>,
}

/// Point Brazier at an adapter already on disk, without copying it.
async fn register_adapter(
    State(state): State<AppState>,
    Json(request): Json<RegisterAdapterRequest>,
) -> ApiResult<Json<Value>> {
    let kind = adapters::AdapterKind::parse(&request.kind)
        .ok_or_else(|| ApiError::bad_request(format!("unknown adapter kind `{}`", request.kind)))?;
    let adapter = adapters::register(
        &state.data_dir,
        kind,
        std::path::Path::new(&request.path),
        request.name,
    )
    .await
    .map_err(ApiError::bad_request)?;
    Ok(Json(json!({ "adapter": adapter })))
}

#[derive(Debug, Deserialize)]
struct AdapterIdRequest {
    id: String,
}

async fn forget_adapter(
    State(state): State<AppState>,
    Json(request): Json<AdapterIdRequest>,
) -> ApiResult<Json<Value>> {
    adapters::forget(&state.data_dir, &request.id)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!({ "id": request.id, "forgotten": true })))
}

async fn delete_adapter(
    State(state): State<AppState>,
    Json(request): Json<AdapterIdRequest>,
) -> ApiResult<Json<Value>> {
    adapters::delete(&state.data_dir, &request.id)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!({ "id": request.id, "deleted": true })))
}

#[derive(Debug, Deserialize)]
struct DownloadAdapterRequest {
    kind: String,
    repo_id: String,
    filename: String,
    #[serde(default = "default_revision")]
    revision: String,
}

fn default_revision() -> String {
    "main".to_owned()
}

async fn download_adapter(
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
    Json(request): Json<DownloadAdapterRequest>,
) -> Response {
    let Some(kind) = adapters::AdapterKind::parse(&request.kind) else {
        return ApiError::bad_request(format!("unknown adapter kind `{}`", request.kind))
            .into_response();
    };

    if !query.stream {
        let result = download::download_adapter_with_progress(
            &state.http,
            download::AdapterDownload {
                data_dir: &state.data_dir,
                kind,
                repo_id: &request.repo_id,
                revision: &request.revision,
                filename: &request.filename,
                cancel: None,
            },
            Box::new(|_| {}),
        )
        .await;
        return match result {
            Ok(result) => (StatusCode::OK, Json(json!(result))).into_response(),
            Err(error) => ApiError::bad_request(error).into_response(),
        };
    }

    let (tx, rx) = progress_channel();
    let http = state.http.clone();
    let data_dir = state.data_dir.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let result = download::download_adapter_with_progress(
            &http,
            download::AdapterDownload {
                data_dir: &data_dir,
                kind,
                repo_id: &request.repo_id,
                revision: &request.revision,
                filename: &request.filename,
                cancel: None,
            },
            Box::new(move |event| push_progress(&progress_tx, event)),
        )
        .await;
        if let Err(error) = result {
            push_progress(&tx, ProgressEvent::error(error.to_string()));
        }
    });
    progress_sse(rx)
}

#[derive(Debug, Deserialize)]
struct PrepareModelRequest {
    model_id: String,
    #[serde(default)]
    mode: Option<String>,
}

async fn unload_model(State(state): State<AppState>) -> ApiResult<Json<serde_json::Value>> {
    state.runtime.unload_chat_model().await;
    state.invalidate_runtimes_cache().await;
    Ok(Json(json!({ "status": "unloaded" })))
}

async fn prepare_model(
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
    Json(request): Json<PrepareModelRequest>,
) -> ApiResult<Response> {
    let agent_mode = request.mode.as_deref() == Some("agent");
    if !query.stream {
        match state
            .runtime
            .prepare_model_stream(&request.model_id, agent_mode)
            .await
        {
            Ok(mut rx) => {
                while let Some(item) = rx.recv().await {
                    match item {
                        Ok(StreamEvent::Load { .. }) => {}
                        Ok(_) => {}
                        Err(error) => return Err(ApiError::from_anyhow(error)),
                    }
                }
                state.invalidate_runtimes_cache().await;
                let residency = state
                    .runtime
                    .loaded_model_residency(&request.model_id)
                    .await;
                return Ok(Json(json!({
                    "status": "ready",
                    "model_id": request.model_id,
                    "residency": residency,
                }))
                .into_response());
            }
            Err(error) => return Err(ApiError::from_anyhow(error)),
        }
    }

    let mut event_rx = state
        .runtime
        .prepare_model_stream(&request.model_id, agent_mode)
        .await
        .map_err(ApiError::from_anyhow)?;
    let model_id = request.model_id.clone();
    let cache_state = state.clone();
    let events = stream! {
        while let Some(item) = event_rx.recv().await {
            match item {
                Ok(StreamEvent::Load { phase, message }) => {
                    let chunk = json!({
                        "model_id": model_id,
                        "phase": phase,
                        "message": message,
                    });
                    yield Ok::<Event, Infallible>(Event::default().data(chunk.to_string()));
                }
                Ok(_) => {}
                Err(error) => {
                    let fork_hints = error
                        .downcast_ref::<ModelLoadError>()
                        .map(|load| load.fork_hints.clone());
                    let mut body = json!({ "error": { "message": error.to_string() } });
                    if let Some(fork_hints) = fork_hints.filter(|hints| !hints.is_empty()) {
                        body["brazier"] = json!({ "fork_hints": fork_hints });
                    }
                    yield Ok::<Event, Infallible>(Event::default().data(body.to_string()));
                    return;
                }
            }
        }
        cache_state.invalidate_runtimes_cache().await;
        let residency = cache_state.runtime.loaded_model_residency(&model_id).await;
        let done = json!({
            "status": "ready",
            "model_id": model_id,
            "residency": residency,
        });
        yield Ok::<Event, Infallible>(Event::default().data(done.to_string()));
    };
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
        .into_response())
}

async fn model_library_path_suggestions(State(state): State<AppState>) -> Json<Value> {
    let settings = state.runtime.settings().await;
    Json(json!({
        "configured": settings.extra_model_library_paths,
        "suggestions": crate::model_library::library_path_suggestions(&settings.extra_model_library_paths),
    }))
}

#[derive(Debug, Deserialize)]
struct StreamQuery {
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct ManagedStatusQuery {
    /// Bypass the release-cache window and refresh upstream now.
    #[serde(default)]
    force: bool,
}

fn progress_channel() -> (
    mpsc::UnboundedSender<ProgressEvent>,
    mpsc::UnboundedReceiver<ProgressEvent>,
) {
    mpsc::unbounded_channel()
}

fn push_progress(tx: &mpsc::UnboundedSender<ProgressEvent>, event: ProgressEvent) {
    let _ = tx.send(event);
}

fn progress_sse(mut rx: mpsc::UnboundedReceiver<ProgressEvent>) -> Response {
    let events = stream! {
        while let Some(event) = rx.recv().await {
            let terminal = event.done == Some(true);
            let payload = serde_json::to_string(&event).unwrap_or_else(|_| "{}".into());
            yield Ok::<Event, Infallible>(Event::default().data(payload));
            if terminal {
                break;
            }
        }
    };
    Sse::new(events)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
        .into_response()
}

async fn managed_llama_status(
    State(state): State<AppState>,
    Query(query): Query<ManagedStatusQuery>,
) -> ApiResult<Json<Value>> {
    use crate::runtime_settings::RuntimeTarget;

    // Local install state answers immediately; the upstream tag is filled in
    // from cache, with `latest_pending` telling the UI a check is still running.
    let cached = llama::cached_release_tag(&state.http, query.force);
    let latest_tag = cached.release.map(|release| release.tag_name);

    let target_specs = [
        ("cpu", RuntimeTarget::Cpu),
        ("cuda", RuntimeTarget::Cuda),
        ("rocm", RuntimeTarget::Rocm),
        ("vulkan", RuntimeTarget::Vulkan),
    ];
    let targets: Vec<Value> = target_specs
        .iter()
        .map(|(id, target)| {
            let installed = llama::managed_is_installed(&state.data_dir, *target);
            let installed_version = llama::managed_installed_version(&state.data_dir, *target);
            let update_available = installed
                && latest_tag
                    .as_deref()
                    .is_some_and(|latest| Some(latest) != installed_version.as_deref());
            json!({
                "target": id,
                "installed": installed,
                "installed_version": installed_version,
                "latest_version": latest_tag,
                "update_available": update_available,
            })
        })
        .collect();

    Ok(Json(json!({
        "latest_version": latest_tag,
        "latest_pending": cached.refreshing,
        "targets": targets,
    })))
}

async fn ensure_llama(
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
    Json(request): Json<EnsureLlamaRequest>,
) -> Response {
    let target = request.target;
    let force = request.force;
    if !query.stream {
        return match state
            .runtime
            .ensure_llama_binary_with_progress(target, force, Box::new(|_| {}))
            .await
        {
            Ok(path) => {
                state.invalidate_runtimes_cache().await;
                (
                    StatusCode::OK,
                    Json(json!({
                        "binary": path.display().to_string(),
                        "status": "ready"
                    })),
                )
                    .into_response()
            }
            Err(error) => ApiError::internal(error).into_response(),
        };
    }

    let (tx, rx) = progress_channel();
    let runtime = state.runtime.clone();
    let cache_state = state.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let result = runtime
            .ensure_llama_binary_with_progress(
                target,
                force,
                Box::new(move |event| {
                    push_progress(&progress_tx, event);
                }),
            )
            .await;
        if let Ok(path) = &result {
            cache_state.invalidate_runtimes_cache().await;
            push_progress(
                &tx,
                ProgressEvent::done(json!({
                    "binary": path.display().to_string(),
                    "status": "ready"
                })),
            );
        }
        if let Err(error) = result {
            push_progress(&tx, ProgressEvent::error(error.to_string()));
        }
    });
    progress_sse(rx)
}

async fn managed_whisper_status(
    State(state): State<AppState>,
    Query(query): Query<ManagedStatusQuery>,
) -> ApiResult<Json<Value>> {
    use crate::runtime_settings::RuntimeTarget;

    let supported = whisper::managed_prebuilts_supported();
    let cached = if supported {
        whisper::cached_release_tag(&state.http, query.force)
    } else {
        crate::github_releases::CachedRelease {
            release: None,
            refreshing: false,
        }
    };
    let latest_tag = cached.release.map(|release| release.tag_name);
    let target_specs = [("cpu", RuntimeTarget::Cpu), ("cuda", RuntimeTarget::Cuda)];
    let targets: Vec<Value> = target_specs
        .iter()
        .map(|(id, target)| {
            let installed = whisper::managed_is_installed(&state.data_dir, *target);
            let installed_version = whisper::managed_installed_version(&state.data_dir, *target);
            let update_available = installed
                && latest_tag
                    .as_deref()
                    .is_some_and(|latest| Some(latest) != installed_version.as_deref());
            json!({
                "target": id,
                "installed": installed,
                "installed_version": installed_version,
                "latest_version": latest_tag,
                "update_available": update_available,
                "managed_supported": supported
                    && (*id == "cpu" || (*id == "cuda" && cfg!(windows))),
            })
        })
        .collect();
    Ok(Json(json!({
        "latest_version": latest_tag,
        "latest_pending": cached.refreshing,
        "managed_supported": supported,
        "targets": targets,
        "note": if supported {
            None
        } else {
            Some("Official whisper.cpp releases do not ship a macOS CLI binary (XCFramework only). Build from source on macOS.")
        },
    })))
}

async fn ensure_whisper(
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
    Json(request): Json<EnsureLlamaRequest>,
) -> Response {
    let target = request.target;
    let force = request.force;
    if !query.stream {
        return match state
            .runtime
            .ensure_whisper_binary_with_progress(target, force, Box::new(|_| {}))
            .await
        {
            Ok(path) => {
                state.invalidate_runtimes_cache().await;
                (
                    StatusCode::OK,
                    Json(json!({
                        "binary": path.display().to_string(),
                        "status": "ready"
                    })),
                )
                    .into_response()
            }
            Err(error) => ApiError::internal(error).into_response(),
        };
    }
    let (tx, rx) = progress_channel();
    let runtime = state.runtime.clone();
    let cache_state = state.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let result = runtime
            .ensure_whisper_binary_with_progress(
                target,
                force,
                Box::new(move |event| {
                    push_progress(&progress_tx, event);
                }),
            )
            .await;
        if let Ok(path) = &result {
            cache_state.invalidate_runtimes_cache().await;
            push_progress(
                &tx,
                ProgressEvent::done(json!({
                    "binary": path.display().to_string(),
                    "status": "ready"
                })),
            );
        }
        if let Err(error) = result {
            push_progress(&tx, ProgressEvent::error(error.to_string()));
        }
    });
    progress_sse(rx)
}

async fn managed_sdcpp_status(
    State(state): State<AppState>,
    Query(query): Query<ManagedStatusQuery>,
) -> ApiResult<Json<Value>> {
    use crate::runtime_settings::RuntimeTarget;

    let cached = sdcpp::cached_release_tag(&state.http, query.force);
    let latest_tag = cached.release.map(|release| release.tag_name);
    let target_specs = [
        ("cpu", RuntimeTarget::Cpu),
        ("cuda", RuntimeTarget::Cuda),
        ("rocm", RuntimeTarget::Rocm),
        ("vulkan", RuntimeTarget::Vulkan),
    ];
    let targets: Vec<Value> = target_specs
        .iter()
        .map(|(id, target)| {
            let installed = sdcpp::managed_is_installed(&state.data_dir, *target);
            let installed_version = sdcpp::managed_installed_version(&state.data_dir, *target);
            let update_available = installed
                && latest_tag
                    .as_deref()
                    .is_some_and(|latest| Some(latest) != installed_version.as_deref());
            json!({
                "target": id,
                "installed": installed,
                "installed_version": installed_version,
                "latest_version": latest_tag,
                "update_available": update_available,
            })
        })
        .collect();
    Ok(Json(json!({
        "latest_version": latest_tag,
        "latest_pending": cached.refreshing,
        "targets": targets,
    })))
}

async fn ensure_sdcpp(
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
    Json(request): Json<EnsureLlamaRequest>,
) -> Response {
    let target = request.target;
    let force = request.force;
    if !query.stream {
        return match state
            .runtime
            .ensure_sdcpp_binary_with_progress(target, force, Box::new(|_| {}))
            .await
        {
            Ok(path) => {
                state.invalidate_runtimes_cache().await;
                (
                    StatusCode::OK,
                    Json(json!({
                        "binary": path.display().to_string(),
                        "status": "ready"
                    })),
                )
                    .into_response()
            }
            Err(error) => ApiError::internal(error).into_response(),
        };
    }
    let (tx, rx) = progress_channel();
    let runtime = state.runtime.clone();
    let cache_state = state.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let result = runtime
            .ensure_sdcpp_binary_with_progress(
                target,
                force,
                Box::new(move |event| {
                    push_progress(&progress_tx, event);
                }),
            )
            .await;
        if let Ok(path) = &result {
            cache_state.invalidate_runtimes_cache().await;
            push_progress(
                &tx,
                ProgressEvent::done(json!({
                    "binary": path.display().to_string(),
                    "status": "ready"
                })),
            );
        }
        if let Err(error) = result {
            push_progress(&tx, ProgressEvent::error(error.to_string()));
        }
    });
    progress_sse(rx)
}

#[derive(Debug, Deserialize)]
struct GenerateImageApiRequest {
    prompt: String,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    steps: Option<u32>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    cfg_scale: Option<f32>,
    #[serde(default)]
    guidance: Option<f32>,
    #[serde(default)]
    init_image_blob: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GenerateVideoApiRequest {
    prompt: String,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    steps: Option<u32>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    cfg_scale: Option<f32>,
    #[serde(default)]
    guidance: Option<f32>,
    #[serde(default)]
    fps: Option<u32>,
    #[serde(default)]
    init_image_blob: Option<String>,
    #[serde(default)]
    video_frames: Option<u32>,
    /// Ending image for first/last-frame conditioning (`--end-img`).
    #[serde(default)]
    end_image_blob: Option<String>,
    /// Reference images for Ref2VA conditioning.
    #[serde(default)]
    ref_image_blobs: Vec<String>,
    /// Reference videos for Ref2VA conditioning; frames are sampled to 24 fps.
    #[serde(default)]
    ref_video_blobs: Vec<String>,
    /// WAV soundtracks paired by index with `ref_video_blobs`.
    #[serde(default)]
    ref_video_audio_blobs: Vec<String>,
    /// Standalone audio references for Ref2VA conditioning.
    #[serde(default)]
    ref_audio_blobs: Vec<String>,
}

/// On-disk size of a generation checkpoint, used as a memory proxy for
/// pre-generation arbitration. Returns 0 when the path cannot be resolved.
fn generation_model_bytes(data_dir: &std::path::Path, model_id: &str) -> u64 {
    sdcpp::path_for_model_id(data_dir, model_id)
        .ok()
        .and_then(|path| std::fs::metadata(path).ok())
        .map(|meta| meta.len())
        .unwrap_or(0)
}

async fn generate_image(
    State(state): State<AppState>,
    Json(request): Json<GenerateImageApiRequest>,
) -> ApiResult<Json<Value>> {
    let settings = state.runtime.settings().await;
    let model_id = request
        .model_id
        .or(settings.default_image_gen_model.clone())
        .ok_or_else(|| ApiError::bad_request("model_id or default_image_gen_model is required"))?;
    let init_image = if let Some(blob) = &request.init_image_blob {
        Some(
            blob_store::blob_path(&state.data_dir, blob)
                .map_err(|e| ApiError::bad_request(e.to_string()))?,
        )
    } else {
        None
    };
    let gen_bytes = generation_model_bytes(&state.data_dir, &model_id);
    let profiles = model_settings::load(&state.data_dir);
    let profile = profiles.diffusion(&model_id).cloned();
    let job = sdcpp::GenerateImageRequest {
        prompt: request.prompt,
        model_id,
        negative_prompt: request.negative_prompt,
        width: request.width,
        height: request.height,
        steps: request.steps,
        seed: request.seed,
        cfg_scale: request.cfg_scale,
        guidance: request.guidance,
        init_image,
        init_image_blob: request.init_image_blob.clone(),
        origin: sdcpp::GenerationOrigin::User,
        timeout_secs: Some(settings.generation_timeout_secs),
    };
    let memory_plan = state.runtime.prepare_generation_memory(gen_bytes).await;
    let generated = sdcpp::generate_image(
        &state.data_dir,
        settings.sdcpp_binary.as_deref(),
        &job,
        profile.as_ref(),
    )
    .await;
    state.runtime.restore_after_generation(memory_plan).await;
    let result = generated.map_err(|e| {
        if e.downcast_ref::<sdcpp::BusyError>().is_some() {
            ApiError::bad_request(e.to_string())
        } else if e.downcast_ref::<sdcpp::CancelledError>().is_some() {
            ApiError::cancelled(e)
        } else {
            ApiError::engine_failure(e)
        }
    })?;
    let bytes = tokio::fs::read(&result.output_path)
        .await
        .map_err(ApiError::internal)?;
    let blob = blob_store::store_bytes(&state.data_dir, &bytes, "image/png", Some("generated.png"))
        .await
        .map_err(ApiError::internal)?;
    let _ = tokio::fs::remove_file(&result.output_path).await;
    Ok(Json(json!({
        "blob": blob,
        "metadata": result.metadata,
        "engine": "stable-diffusion.cpp",
    })))
}

/// The generation running right now, if any.
///
/// Polled by the interface so a job a model started is visible while it runs —
/// prompt, conditioning image, and how long it has been going — rather than
/// only when it finally produces something.
async fn active_generation() -> Json<Value> {
    Json(json!({ "active": sdcpp::active_generation() }))
}

/// Stop the running generation.
async fn cancel_generation() -> Json<Value> {
    Json(json!({ "cancelled": sdcpp::cancel_active_generation() }))
}

async fn generate_video(
    State(state): State<AppState>,
    Json(request): Json<GenerateVideoApiRequest>,
) -> ApiResult<Json<Value>> {
    let settings = state.runtime.settings().await;
    let model_id = request
        .model_id
        .or(settings.default_video_gen_model.clone())
        .ok_or_else(|| ApiError::bad_request("model_id or default_video_gen_model is required"))?;

    // Ref2VA accepts at most 9 reference images, 3 videos, 3 audio clips, and
    // 12 files in total. Enforce here so a malformed request fails before
    // sd-cli spends minutes loading a 40 GB checkpoint.
    let ref_image_count = request.ref_image_blobs.len();
    let ref_video_count = request.ref_video_blobs.len();
    let ref_audio_count = request.ref_video_audio_blobs.len() + request.ref_audio_blobs.len();
    let total_refs = ref_image_count + ref_video_count + ref_audio_count;
    let limit = |allowed: bool, message: &str| {
        if allowed {
            Ok(())
        } else {
            Err(ApiError::bad_request(message))
        }
    };
    limit(
        ref_image_count <= 9,
        "MiniMax-H3 Ref2VA accepts at most 9 reference images",
    )?;
    limit(
        ref_video_count <= 3,
        "MiniMax-H3 Ref2VA accepts at most 3 reference videos",
    )?;
    limit(
        ref_audio_count <= 3,
        "MiniMax-H3 Ref2VA accepts at most 3 reference audio clips",
    )?;
    limit(
        total_refs <= 12,
        "MiniMax-H3 Ref2VA accepts at most 12 reference files in total",
    )?;

    let init_image = if let Some(blob) = &request.init_image_blob {
        Some(
            blob_store::blob_path(&state.data_dir, blob)
                .map_err(|e| ApiError::bad_request(e.to_string()))?,
        )
    } else {
        None
    };
    let end_image = if let Some(blob) = &request.end_image_blob {
        Some(
            blob_store::blob_path(&state.data_dir, blob)
                .map_err(|e| ApiError::bad_request(e.to_string()))?,
        )
    } else {
        None
    };
    let ref_images = {
        let mut paths = Vec::with_capacity(request.ref_image_blobs.len());
        for blob in &request.ref_image_blobs {
            paths.push(
                blob_store::blob_path(&state.data_dir, blob)
                    .map_err(|e| ApiError::bad_request(e.to_string()))?,
            );
        }
        paths
    };
    let ref_audios = {
        let mut paths = Vec::with_capacity(request.ref_audio_blobs.len());
        for blob in &request.ref_audio_blobs {
            paths.push(
                media::materialize_wav_from_blob(&state.data_dir, blob, "audio/wav")
                    .await
                    .map_err(|e| ApiError::bad_request(e.to_string()))?,
            );
        }
        paths
    };
    let ref_video_audios = {
        let mut paths = Vec::with_capacity(request.ref_video_audio_blobs.len());
        for blob in &request.ref_video_audio_blobs {
            paths.push(
                media::materialize_wav_from_blob(&state.data_dir, blob, "audio/wav")
                    .await
                    .map_err(|e| ApiError::bad_request(e.to_string()))?,
            );
        }
        paths
    };

    // Reference videos are handed to sd-cli as a directory of 24 fps frames,
    // so each source clip is sampled into its own temp directory first. The
    // directories are removed once the job has finished, whether it succeeded
    // or not.
    let mut ref_frame_dirs = Vec::new();
    let mut ref_videos = Vec::new();
    for blob in &request.ref_video_blobs {
        let input = blob_store::blob_path(&state.data_dir, blob)
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        let frame_dir = state
            .data_dir
            .join("tmp")
            .join("sdcpp")
            .join(format!("ref-{blob}-frames"));
        let extracted = media::extract_reference_video_frames(&input, &frame_dir)
            .await
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        if extracted == 0 {
            return Err(ApiError::bad_request("reference video produced no frames"));
        }
        ref_videos.push(frame_dir.clone());
        ref_frame_dirs.push(frame_dir);
    }

    let gen_bytes = generation_model_bytes(&state.data_dir, &model_id);
    let profiles = model_settings::load(&state.data_dir);
    let profile = profiles.diffusion(&model_id).cloned();
    let job = sdcpp::GenerateVideoRequest {
        prompt: request.prompt,
        model_id,
        negative_prompt: request.negative_prompt,
        width: request.width,
        height: request.height,
        steps: request.steps,
        seed: request.seed,
        cfg_scale: request.cfg_scale,
        guidance: request.guidance,
        init_image,
        init_image_blob: request.init_image_blob.clone(),
        origin: sdcpp::GenerationOrigin::User,
        timeout_secs: Some(settings.generation_timeout_secs),
        video_frames: request.video_frames,
        fps: request.fps,
        end_image,
        end_image_blob: request.end_image_blob.clone(),
        ref_images,
        ref_image_blobs: request.ref_image_blobs.clone(),
        ref_videos,
        ref_video_blobs: request.ref_video_blobs.clone(),
        ref_video_audios,
        ref_audios,
        ref_audio_blobs: request.ref_audio_blobs.clone(),
    };
    let memory_plan = state.runtime.prepare_generation_memory(gen_bytes).await;
    let generated = sdcpp::generate_video(
        &state.data_dir,
        settings.sdcpp_binary.as_deref(),
        &job,
        profile.as_ref(),
    )
    .await;
    state.runtime.restore_after_generation(memory_plan).await;
    for dir in ref_frame_dirs {
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }
    let result = generated.map_err(|e| {
        if e.downcast_ref::<sdcpp::BusyError>().is_some() {
            ApiError::bad_request(e.to_string())
        } else if e.downcast_ref::<sdcpp::CancelledError>().is_some() {
            ApiError::cancelled(e)
        } else {
            ApiError::engine_failure(e)
        }
    })?;
    let bytes = tokio::fs::read(&result.output_path)
        .await
        .map_err(ApiError::internal)?;
    let mime = if result
        .output_path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("webm"))
    {
        "video/webm"
    } else {
        "video/mp4"
    };
    let blob = blob_store::store_bytes(
        &state.data_dir,
        &bytes,
        mime,
        Some(
            result
                .output_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("generated.mp4"),
        ),
    )
    .await
    .map_err(ApiError::internal)?;
    let _ = tokio::fs::remove_file(&result.output_path).await;
    Ok(Json(json!({
        "blob": blob,
        "metadata": result.metadata,
        "engine": "stable-diffusion.cpp",
    })))
}

#[derive(Debug, Deserialize)]
struct CreateVoiceSessionRequest {
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    persona_text: Option<String>,
    #[serde(default)]
    voice_prompt_path: Option<String>,
}

async fn create_voice_session(
    State(state): State<AppState>,
    Json(request): Json<CreateVoiceSessionRequest>,
) -> ApiResult<Json<Value>> {
    let settings = state.runtime.settings().await;
    let python = voice::resolve_python(&state.data_dir, settings.voice_python.as_deref())
        .ok_or_else(|| {
            ApiError::bad_request(
                "PersonaPlex runtime is not installed. Build it from Manage → Runtimes.",
            )
        })?;
    let model_path = voice::resolve_model_path(
        &state.data_dir,
        request
            .model_id
            .as_deref()
            .or(settings.default_voice_model.as_deref()),
    );
    let persona = request
        .persona_text
        .or(settings.default_voice_persona.clone())
        .unwrap_or_else(|| "You are a helpful assistant.".into());
    let voice_prompt = request.voice_prompt_path.map(PathBuf::from);
    let hf_token = crate::hf_auth::load_token(&state.data_dir);
    // The voice model's own configuration fills in whatever the session did not
    // name — its persona, its voice, and how heavily it is quantised.
    let voice_profile = request
        .model_id
        .as_deref()
        .or(settings.default_voice_model.as_deref())
        .and_then(|model_id| {
            model_settings::load(&state.data_dir)
                .voice(model_id)
                .cloned()
        });
    let voice_state = state.runtime.voice_state().await;
    let session = voice_state
        .sessions
        .create_session(
            &python,
            model_path.as_deref(),
            persona.clone(),
            voice_prompt.clone(),
            hf_token,
            voice_profile.as_ref(),
        )
        .await
        .map_err(ApiError::internal)?;
    let ws_url = session.proxy_url().await;
    Ok(Json(json!({
        "id": session.id,
        "ws_url": ws_url,
        "persona_text": persona,
        "voice_prompt": voice_prompt.map(|p| p.display().to_string()),
        "protocol": {
            "handshake": voice::protocol::HANDSHAKE,
            "audio": voice::protocol::AUDIO,
            "text": voice::protocol::TEXT,
        },
        "engine": "personaplex",
    })))
}

async fn list_voice_session(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let voice_state = state.runtime.voice_state().await;
    if let Some(session) = voice_state.sessions.active_session().await {
        let ws_url = session.proxy_url().await;
        Ok(Json(json!({
            "session": {
                "id": session.id,
                "ws_url": ws_url,
                "persona_text": session.persona_text,
            }
        })))
    } else {
        Ok(Json(json!({ "session": null })))
    }
}

async fn end_voice_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let voice_state = state.runtime.voice_state().await;
    voice_state
        .sessions
        .end_session(&id)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!({ "ended": id })))
}

async fn build_runtime(
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
    Json(request): Json<builds::BuildRequest>,
) -> Response {
    if !query.stream {
        return match builds::run_build_with_progress(
            &state.data_dir,
            request,
            &state.active_builds,
            Box::new(|_| {}),
        )
        .await
        {
            Ok(binary) => {
                state.invalidate_runtimes_cache().await;
                (
                    StatusCode::OK,
                    Json(json!({
                        "binary": binary.display().to_string(),
                        "status": "ready"
                    })),
                )
                    .into_response()
            }
            Err(report) => ApiError::bad_request(format_build_failure(&report)).into_response(),
        };
    }
    let (tx, rx) = progress_channel();
    let label = format!("Build {}", request.engine);
    let job = match state
        .db
        .create_queued_download_job(crate::db::QueuedDownloadJobInput {
            repo_id: &request.repository,
            filename: "runtime source build",
            revision: &request.revision,
            kind: "runtime-build",
            payload: None,
            label: Some(&label),
            status: "downloading",
        })
        .await
    {
        Ok(job) => job,
        Err(error) => return ApiError::internal(error).into_response(),
    };
    let data_dir = state.data_dir.clone();
    let db = state.db.clone();
    let progress_db = db.clone();
    let active_builds = state.active_builds.clone();
    let build_slots = state.build_slots.clone();
    let cache_state = state.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let job_id = job.id;
        let progress_job_id = job_id.clone();
        let _ = db
            .update_download_job_message(&job_id, "Waiting for the build slot")
            .await;
        let Ok(slot) = build_slots.acquire_owned().await else {
            let _ = db.fail_download_job(&job_id, "build queue is closed").await;
            return;
        };
        if db
            .get_download_job_public(&job_id)
            .await
            .is_ok_and(|job| job.status == "cancelled")
        {
            return;
        }
        if let Err(error) = db.start_download_job(&job_id).await {
            push_progress(
                &tx,
                ProgressEvent::error(format!("build did not start: {error}")),
            );
            return;
        }
        let result = builds::run_build_with_progress(
            &data_dir,
            request,
            &active_builds,
            Box::new(move |event| {
                let db = progress_db.clone();
                let job_id = progress_job_id.clone();
                if let Some(build_id) = event
                    .result
                    .as_ref()
                    .and_then(|value| value.get("build_id"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                {
                    let task_db = db.clone();
                    let task_job_id = job_id.clone();
                    tokio::spawn(async move {
                        let _ = task_db
                            .set_download_job_payload(&task_job_id, &build_id)
                            .await;
                    });
                } else if let Some(percent) = event.percent {
                    let task_db = db.clone();
                    let job_id = job_id.clone();
                    tokio::spawn(async move {
                        let _ = task_db
                            .update_download_job_progress(
                                &job_id,
                                percent.round() as u64,
                                Some(100),
                            )
                            .await;
                    });
                }
                if event.phase != "log"
                    && let Some(message) = event.message.as_deref()
                {
                    let task_db = db.clone();
                    let job_id = job_id.clone();
                    let message = message.to_owned();
                    tokio::spawn(async move {
                        let _ = task_db.update_download_job_message(&job_id, &message).await;
                    });
                }
                push_progress(&progress_tx, event);
            }),
        )
        .await;
        match result {
            Ok(binary) => {
                if let Err(error) = db.complete_download_job(&job_id, "", 100).await {
                    push_progress(
                        &tx,
                        ProgressEvent::error(format!(
                            "build finished after its job was settled: {error}"
                        )),
                    );
                    return;
                }
                cache_state.invalidate_runtimes_cache().await;
                push_progress(
                    &tx,
                    ProgressEvent::done(json!({
                        "binary": binary.display().to_string(),
                        "status": "ready"
                    })),
                );
            }
            Err(report) => {
                let _ = db.fail_download_job(&job_id, &report.message).await;
                push_progress(
                    &tx,
                    ProgressEvent::build_failed(
                        &serde_json::to_value(&report)
                            .unwrap_or_else(|_| json!({ "message": report.message })),
                    ),
                );
            }
        }
        drop(slot);
    });
    progress_sse(rx)
}

#[derive(Debug, Deserialize)]
struct CancelBuildRequest {
    build_id: String,
}

#[derive(Debug, Deserialize)]
struct CancelBuildJobRequest {
    job_id: String,
}

async fn cancel_build(
    State(state): State<AppState>,
    Json(request): Json<CancelBuildRequest>,
) -> ApiResult<Json<Value>> {
    if state.active_builds.cancel(&request.build_id) {
        Ok(Json(json!({ "cancelled": request.build_id })))
    } else {
        Err(ApiError::bad_request(format!(
            "no in-progress build with id `{}`",
            request.build_id
        )))
    }
}

/// Cancel a runtime build from its durable tray row.
async fn cancel_build_job(
    State(state): State<AppState>,
    Json(request): Json<CancelBuildJobRequest>,
) -> ApiResult<Json<Value>> {
    let job = state
        .db
        .get_download_job_public(&request.job_id)
        .await
        .map_err(|_| ApiError::bad_request("no such build job"))?;
    if job.kind != "runtime-build" {
        return Err(ApiError::bad_request("that job is not a runtime build"));
    }
    if job.status == "pending" {
        state
            .db
            .cancel_download_job(&request.job_id)
            .await
            .map_err(ApiError::internal)?;
        return Ok(Json(
            json!({ "cancelled": request.job_id, "queued_only": true }),
        ));
    }
    let build_id = job
        .payload
        .as_deref()
        .ok_or_else(|| ApiError::bad_request("build is still starting; try again in a moment"))?;
    state
        .db
        .cancel_download_job(&request.job_id)
        .await
        .map_err(ApiError::bad_request)?;
    let signalled = state.active_builds.cancel(build_id);
    Ok(Json(json!({
        "cancelled": request.job_id,
        "signalled": signalled
    })))
}

fn format_build_failure(report: &builds::BuildFailureReport) -> String {
    if report.hints.is_empty() {
        report.message.clone()
    } else {
        format!(
            "{}\n\n{}",
            report.message,
            report
                .hints
                .iter()
                .map(|hint| format!("- {hint}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}

async fn download_model(
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
    Json(request): Json<download::DownloadRequest>,
) -> Response {
    if !query.stream {
        // Keep the original request on the job row even for the older
        // synchronous endpoint. The download tray can pause these jobs, and
        // resume needs the engine as well as the repository and filename.
        let tracked = match track_resumable_download(
            &state,
            &crate::download_queue::QueuedWork::Gguf(request.clone()),
        )
        .await
        {
            Ok(tracked) => tracked,
            Err(error) => return error.into_response(),
        };
        let (job_id, cancel) = tracked;
        let result = download::download_gguf_with_progress(
            &state.http,
            &state.data_dir,
            request,
            Box::new(|_| {}),
            Some((state.db.clone(), job_id.clone())),
            Some(cancel),
        )
        .await;
        let stop = state.active_downloads.stop_reason(&job_id);
        state.active_downloads.finish(&job_id);
        if let Err(error) = &result {
            match stop {
                Some(crate::active_downloads::StopReason::Pause) => {
                    let _ = state.db.pause_download_job(&job_id).await;
                }
                Some(crate::active_downloads::StopReason::Cancel) => {
                    let _ = state.db.cancel_download_job(&job_id).await;
                }
                None => {
                    let _ = state
                        .db
                        .fail_download_job(&job_id, &error.to_string())
                        .await;
                }
            }
        }
        return match result {
            Ok(result) => {
                state.invalidate_models_cache().await;
                (StatusCode::OK, Json(json!(result))).into_response()
            }
            Err(error) => ApiError::bad_request(error).into_response(),
        };
    }

    let (tx, rx) = progress_channel();
    let http = state.http.clone();
    let data_dir = state.data_dir.clone();
    let db = state.db.clone();
    let active_downloads = state.active_downloads.clone();
    let cache_state = state.clone();
    let tracked = match track_resumable_download(
        &state,
        &crate::download_queue::QueuedWork::Gguf(request.clone()),
    )
    .await
    {
        Ok(tracked) => tracked,
        Err(error) => return error.into_response(),
    };
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let (job_id, cancel) = tracked;
        let result = download::download_gguf_with_progress(
            &http,
            &data_dir,
            request,
            Box::new(move |event| {
                push_progress(&progress_tx, event);
            }),
            Some((db.clone(), job_id.clone())),
            Some(cancel),
        )
        .await;
        let stop = active_downloads.stop_reason(&job_id);
        active_downloads.finish(&job_id);
        if let Err(error) = &result {
            match stop {
                Some(crate::active_downloads::StopReason::Pause) => {
                    let _ = db.pause_download_job(&job_id).await;
                }
                Some(crate::active_downloads::StopReason::Cancel) => {
                    let _ = db.cancel_download_job(&job_id).await;
                }
                None => {
                    let _ = db.fail_download_job(&job_id, &error.to_string()).await;
                }
            }
        }
        match result {
            Ok(download_result) => {
                cache_state.invalidate_models_cache().await;
                push_progress(
                    &tx,
                    ProgressEvent::done(
                        serde_json::to_value(&download_result).unwrap_or_else(|_| json!({})),
                    ),
                );
            }
            Err(error) => {
                push_progress(&tx, ProgressEvent::error(error.to_string()));
            }
        }
    });
    progress_sse(rx)
}

async fn download_mlx_model(
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
    Json(request): Json<download::MlxDownloadRequest>,
) -> Response {
    if !query.stream {
        let result = download::download_mlx_snapshot_with_progress(
            &state.http,
            &state.data_dir,
            request,
            Box::new(|_| {}),
            None,
            None,
        )
        .await;
        return match result {
            Ok(result) => {
                state.invalidate_models_cache().await;
                (StatusCode::OK, Json(json!(result))).into_response()
            }
            Err(error) => ApiError::bad_request(error).into_response(),
        };
    }

    let (tx, rx) = progress_channel();
    let http = state.http.clone();
    let data_dir = state.data_dir.clone();
    let tracked = match track_resumable_download(
        &state,
        &crate::download_queue::QueuedWork::Mlx(request.clone()),
    )
    .await
    {
        Ok(tracked) => tracked,
        Err(error) => return error.into_response(),
    };
    let db = state.db.clone();
    let active_downloads = state.active_downloads.clone();
    let cache_state = state.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let (job_id, cancel) = tracked;
        let result = download::download_mlx_snapshot_with_progress(
            &http,
            &data_dir,
            request,
            Box::new(move |event| {
                push_progress(&progress_tx, event);
            }),
            Some((db.clone(), job_id.clone())),
            Some(cancel),
        )
        .await;
        let stop = active_downloads.stop_reason(&job_id);
        active_downloads.finish(&job_id);
        if let Err(error) = &result {
            match stop {
                Some(crate::active_downloads::StopReason::Pause) => {
                    let _ = db.pause_download_job(&job_id).await;
                }
                Some(crate::active_downloads::StopReason::Cancel) => {
                    let _ = db.cancel_download_job(&job_id).await;
                }
                None => {
                    let _ = db.fail_download_job(&job_id, &error.to_string()).await;
                }
            }
        }
        match result {
            Ok(download_result) => {
                cache_state.invalidate_models_cache().await;
                push_progress(
                    &tx,
                    ProgressEvent::done(
                        serde_json::to_value(&download_result).unwrap_or_else(|_| json!({})),
                    ),
                );
            }
            Err(error) => {
                push_progress(&tx, ProgressEvent::error(error.to_string()));
            }
        }
    });
    progress_sse(rx)
}

#[derive(Debug, Deserialize)]
struct CancelDownloadRequest {
    job_id: String,
}

async fn cancel_model_download(
    State(state): State<AppState>,
    Json(request): Json<CancelDownloadRequest>,
) -> ApiResult<Json<Value>> {
    state
        .db
        .cancel_download_job(&request.job_id)
        .await
        .map_err(ApiError::bad_request)?;
    let signalled = state.active_downloads.cancel(&request.job_id);
    Ok(Json(json!({
        "cancelled": request.job_id,
        "queued_only": !signalled
    })))
}

/// Add work to the download queue.
///
/// One transfer runs at a time, but anything can be queued while one is in
/// flight, and the job row keeps enough detail to resume after a pause.
async fn enqueue_work(
    state: &AppState,
    work: crate::download_queue::QueuedWork,
) -> ApiResult<Json<Value>> {
    let payload = serde_json::to_string(&work).map_err(ApiError::internal)?;
    let job = state
        .db
        .create_queued_download_job(crate::db::QueuedDownloadJobInput {
            repo_id: &work.repo_id(),
            filename: &work.filename(),
            revision: &work.revision(),
            kind: work.kind(),
            payload: Some(&payload),
            label: Some(&work.label()),
            status: "pending",
        })
        .await
        .map_err(ApiError::bad_request)?;
    state
        .download_queue
        .enqueue(crate::download_queue::QueuedDownload {
            job_id: job.id.clone(),
            work,
        })
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "job_id": job.id, "status": job.status })))
}

async fn queue_model_download(
    State(state): State<AppState>,
    Json(request): Json<download::DownloadRequest>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    let body = enqueue_work(&state, crate::download_queue::QueuedWork::Gguf(request)).await?;
    Ok((StatusCode::ACCEPTED, body))
}

#[derive(Debug, Deserialize)]
struct QueueSnapshotRequest {
    repo_id: String,
    #[serde(default)]
    engine: Option<String>,
    #[serde(default = "default_main_revision")]
    revision: String,
}

fn default_main_revision() -> String {
    "main".to_owned()
}

/// Queue a multi-file snapshot: MLX, PersonaPlex, or streaming ASR.
async fn queue_snapshot_download(
    State(state): State<AppState>,
    Path(kind): Path<String>,
    Json(request): Json<QueueSnapshotRequest>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    let engine = request.engine.unwrap_or_else(|| match kind.as_str() {
        "personaplex" => voice::ENGINE.to_owned(),
        "streaming-asr" => streaming_asr::ENGINE.to_owned(),
        _ => "mlx-lm".to_owned(),
    });
    let snapshot = download::MlxDownloadRequest {
        repo_id: request.repo_id,
        engine,
        revision: request.revision,
    };
    let work = match kind.as_str() {
        "mlx" => crate::download_queue::QueuedWork::Mlx(snapshot),
        "personaplex" => crate::download_queue::QueuedWork::Personaplex(snapshot),
        "streaming-asr" => crate::download_queue::QueuedWork::StreamingAsr(snapshot),
        other => {
            return Err(ApiError::bad_request(format!(
                "unknown snapshot kind `{other}`"
            )));
        }
    };
    let body = enqueue_work(&state, work).await?;
    Ok((StatusCode::ACCEPTED, body))
}

/// Queue a stable-diffusion.cpp bundle install.
async fn queue_sdcpp_install(
    State(state): State<AppState>,
    Json(request): Json<InstallSdcppBundleRequest>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    let bundle = match (request.bundle, request.id) {
        (Some(bundle), _) => {
            sdcpp_catalog::validate(&bundle).map_err(ApiError::bad_request)?;
            bundle
        }
        (None, Some(id)) => sdcpp_catalog::find(&state.data_dir, &id)
            .ok_or_else(|| ApiError::bad_request(format!("unknown model bundle `{id}`")))?,
        (None, None) => return Err(ApiError::bad_request("id or bundle is required")),
    };
    ensure_license_consent(&state, &bundle)
        .await
        .map_err(ApiError::bad_request)?;
    let body = enqueue_work(
        &state,
        crate::download_queue::QueuedWork::SdcppBundle(bundle),
    )
    .await?;
    Ok((StatusCode::ACCEPTED, body))
}

#[derive(Debug, Deserialize)]
struct DownloadJobRequest {
    job_id: String,
}

/// Pause a download, keeping its partial file for a later resume.
async fn pause_model_download(
    State(state): State<AppState>,
    Json(request): Json<DownloadJobRequest>,
) -> ApiResult<Json<Value>> {
    use crate::active_downloads::StopReason;
    // Persist the transition immediately even for an active transfer. This
    // keeps the tray honest while the worker cooperatively stops, and the
    // worker's later pause is deliberately allowed to observe "already
    // paused" without changing it again.
    state
        .db
        .pause_download_job(&request.job_id)
        .await
        .map_err(ApiError::bad_request)?;
    let signalled = state
        .active_downloads
        .stop(&request.job_id, StopReason::Pause);
    Ok(Json(json!({
        "paused": request.job_id,
        "queued_only": !signalled
    })))
}

/// Forget a finished, failed, or cancelled job so it leaves the list.
async fn dismiss_model_download(
    State(state): State<AppState>,
    Json(request): Json<DownloadJobRequest>,
) -> ApiResult<Json<Value>> {
    state
        .db
        .dismiss_download_job(&request.job_id)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!({ "dismissed": request.job_id })))
}

/// Clear every settled job at once, for a list that has built up over time.
async fn dismiss_finished_model_downloads(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let dismissed = state
        .db
        .dismiss_finished_download_jobs()
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "dismissed": dismissed })))
}

/// Put a paused, failed, or cancelled job back in line. The transfer resumes
/// from its partial file rather than starting over.
async fn resume_model_download(
    State(state): State<AppState>,
    Json(request): Json<DownloadJobRequest>,
) -> ApiResult<Json<Value>> {
    let job = state
        .db
        .get_download_job_public(&request.job_id)
        .await
        .map_err(|_| ApiError::bad_request("no such download job"))?;
    let payload = job.payload.as_deref().ok_or_else(|| {
        ApiError::bad_request("this job predates the queue and cannot be resumed")
    })?;
    let work: crate::download_queue::QueuedWork = serde_json::from_str(payload)
        .map_err(|_| ApiError::bad_request("unreadable job payload"))?;
    state
        .db
        .requeue_download_job(&request.job_id)
        .await
        .map_err(ApiError::bad_request)?;
    state
        .download_queue
        .enqueue(crate::download_queue::QueuedDownload {
            job_id: job.id.clone(),
            work,
        })
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "resumed": job.id })))
}

async fn list_models(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let models = state.runtime.models().await.map_err(ApiError::internal)?;
    let data = models
        .into_iter()
        .map(|model| {
            json!({
                "id": model.id,
                "object": "model",
                "owned_by": format!("brazier:{}", model.engine),
                "engine": model.engine,
                "capabilities": model.capabilities,
                "size_bytes": model.size_bytes,
                "read_only": model.read_only,
                "library_label": model.library_label,
            })
        })
        .collect::<Vec<_>>();
    Ok(Json(json!({ "object": "list", "data": data })))
}

#[derive(Debug, Deserialize)]
struct TranscriptionRequest {
    #[serde(default)]
    file_sha256: Option<String>,
    #[serde(default)]
    file_base64: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    stream: bool,
    /// Prefer `streaming-asr` or `whisper.cpp`. Default: streaming when stream=true.
    #[serde(default)]
    engine: Option<String>,
}

async fn resolve_transcription_blob(
    state: &AppState,
    request: &TranscriptionRequest,
) -> ApiResult<(String, String)> {
    let (bytes, mime) = if let Some(sha256) = request.file_sha256.as_deref() {
        let (bytes, stored_mime) = blob_store::read_blob(&state.data_dir, sha256)
            .await
            .map_err(ApiError::bad_request)?;
        (bytes, request.mime_type.clone().unwrap_or(stored_mime))
    } else if let Some(encoded) = request.file_base64.as_deref() {
        use base64::Engine as _;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|error| ApiError::bad_request(format!("invalid file_base64: {error}")))?;
        (
            bytes,
            request
                .mime_type
                .clone()
                .unwrap_or_else(|| "audio/wav".to_owned()),
        )
    } else {
        return Err(ApiError::bad_request(
            "provide file_sha256 or file_base64 for transcription",
        ));
    };
    let stored =
        blob_store::store_bytes(&state.data_dir, &bytes, &mime, Some("transcription-input"))
            .await
            .map_err(ApiError::bad_request)?;
    Ok((stored.sha256, mime))
}

async fn audio_transcriptions(
    State(state): State<AppState>,
    Json(request): Json<TranscriptionRequest>,
) -> ApiResult<Response> {
    // Timed and reported, because which interface should transcribe a spoken
    // turn is an open question with two plausible answers — one binary
    // invocation against a resident Python worker — and it cannot be settled by
    // reasoning about them. `duration_ms` covers everything the daemon does with
    // the audio: decode, convert, and run the engine.
    let started = std::time::Instant::now();
    let settings = state.runtime.settings().await;
    let prefer_streaming = request.stream
        || request
            .engine
            .as_deref()
            .is_some_and(|engine| engine == streaming_asr::ENGINE);
    if prefer_streaming {
        // Checked for the actionable message; the worker resolves its own copy.
        streaming_asr::resolve_python(
            &state.data_dir,
            settings.streaming_asr_python.as_deref(),
        )
        .ok_or_else(|| {
            ApiError::bad_request(
                "Streaming ASR requires a built streaming-asr Python environment. Install it under Runtimes.".to_owned(),
            )
        })?;
        let model_path = streaming_asr::resolve_model_path(
            &state.data_dir,
            request
                .model
                .as_deref()
                .or(settings.streaming_asr_model.as_deref()),
        )
        .ok_or_else(|| {
            ApiError::bad_request(
                "Streaming ASR requires a downloaded Nemotron ASR Streaming snapshot from Discover.".to_owned(),
            )
        })?;
        let (sha256, mime) = resolve_transcription_blob(&state, &request).await?;
        let wav = media::materialize_wav_from_blob(&state.data_dir, &sha256, &mime)
            .await
            .map_err(ApiError::bad_request)?;
        // The worker outlives the request, so the model is loaded once per
        // session rather than once per utterance. Events arrive on a channel
        // either way; a non-streaming caller just collects them.
        let (tx, mut events) = tokio::sync::mpsc::channel(64);
        let transcription = {
            let runtime = state.runtime.clone();
            let model_path = model_path.clone();
            let wav = wav.clone();
            tokio::spawn(async move {
                runtime
                    .transcribe_streaming(&model_path, &wav, None, tx)
                    .await
            })
        };
        if !request.stream {
            while events.recv().await.is_some() {}
            let text = transcription
                .await
                .map_err(ApiError::internal)?
                .map_err(ApiError::bad_request)?;
            let _ = tokio::fs::remove_file(&wav).await;
            return Ok(Json(json!({
                "text": text.trim(),
                "engine": streaming_asr::ENGINE,
                "duration_ms": started.elapsed().as_millis() as u64,
            }))
            .into_response());
        }
        let stream_events = stream! {
            while let Some(item) = events.recv().await {
                match item {
                    Ok(streaming_asr::WorkerEvent::Status { phase, message, latency_ms }) => {
                        yield Ok::<Event, Infallible>(Event::default()
                            .event("transcription.status")
                            .data(json!({
                                "type": "transcription.status",
                                "phase": phase,
                                "message": message,
                                "latency_ms": latency_ms,
                            }).to_string()));
                    }
                    Ok(streaming_asr::WorkerEvent::Delta { text }) => {
                        yield Ok::<Event, Infallible>(Event::default()
                            .event("transcription.delta")
                            .data(json!({
                                "type": "transcription.delta",
                                "text": text,
                            }).to_string()));
                    }
                    Ok(streaming_asr::WorkerEvent::Done { text }) => {
                        yield Ok::<Event, Infallible>(Event::default()
                            .event("transcription.done")
                            .data(json!({
                                "type": "transcription.done",
                                "text": text,
                                "engine": streaming_asr::ENGINE,
                                "duration_ms": started.elapsed().as_millis() as u64,
                            }).to_string()));
                    }
                    Ok(streaming_asr::WorkerEvent::Error { message }) => {
                        yield Ok::<Event, Infallible>(Event::default()
                            .event("error")
                            .data(json!({ "error": { "message": message } }).to_string()));
                        break;
                    }
                    Err(error) => {
                        yield Ok::<Event, Infallible>(Event::default()
                            .event("error")
                            .data(json!({ "error": { "message": error.to_string() } }).to_string()));
                        break;
                    }
                }
            }
            let _ = tokio::fs::remove_file(&wav).await;
        };
        return Ok(Sse::new(stream_events)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    let binary = whisper::resolve_binary(&state.data_dir, settings.whisper_binary.as_deref())
        .ok_or_else(|| ApiError::bad_request(media::asr_missing_message().to_owned()))?;
    let model_pref = request
        .model
        .as_deref()
        .or(settings.whisper_model.as_deref());
    let model_path = whisper::resolve_model_path(&state.data_dir, model_pref);
    if !whisperkit::is_whisperkit_binary(&binary) && model_path.is_none() {
        return Err(ApiError::bad_request(
            media::asr_missing_message().to_owned(),
        ));
    }

    let (sha256, mime) = resolve_transcription_blob(&state, &request).await?;

    let mut messages = vec![crate::types::OpenAiMessage {
        role: "user".into(),
        content: json!([{
            "type": "brazier_blob",
            "brazier_blob": {
                "sha256": sha256,
                "mime_type": mime,
                "name": "audio"
            }
        }]),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let caps = crate::types::ModelCapabilities {
        input_modalities: vec!["audio".into()],
        output_modalities: vec!["text".into()],
        streaming: false,
        tools: false,
        reasoning: false,
        max_context_length: None,
        reasoning_modes: Vec::new(),
        harmony: false,
        audio_input: None,
        computer_use: false,
    };
    let ctx = media::MediaContext {
        data_dir: &state.data_dir,
        model_caps: &caps,
        features: media::PipelineFeatures {
            asr: true,
            video_preprocess: media::ffmpeg_available(),
        },
        whisper_binary: Some(binary),
        whisper_model: model_path,
        whisper_model_pref: model_pref,
        whisper_profile: model_pref.and_then(|id| {
            model_settings::load(&state.data_dir)
                .transcription(id)
                .cloned()
        }),
    };
    media::prepare_messages(&ctx, &mut messages, None)
        .await
        .map_err(ApiError::bad_request)?;
    let text = messages
        .first()
        .and_then(|message| match &message.content {
            Value::Array(parts) => parts
                .iter()
                .find_map(|part| part.get("text").and_then(Value::as_str)),
            Value::String(text) => Some(text.as_str()),
            _ => None,
        })
        .unwrap_or("")
        .lines()
        .skip_while(|line| line.starts_with('['))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_owned();
    Ok(Json(json!({
        "text": text,
        "engine": whisper::ENGINE,
        "duration_ms": started.elapsed().as_millis() as u64,
    }))
    .into_response())
}

async fn download_streaming_asr_model(
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
    Json(request): Json<download::MlxDownloadRequest>,
) -> Response {
    let mut request = request;
    request.engine = streaming_asr::ENGINE.to_owned();
    if !query.stream {
        let result = download::download_streaming_asr_snapshot_with_progress(
            &state.http,
            &state.data_dir,
            request,
            Box::new(|_| {}),
            None,
            None,
        )
        .await;
        return match result {
            Ok(result) => {
                state.invalidate_models_cache().await;
                (StatusCode::OK, Json(json!(result))).into_response()
            }
            Err(error) => ApiError::bad_request(error).into_response(),
        };
    }

    let (tx, rx) = progress_channel();
    let http = state.http.clone();
    let data_dir = state.data_dir.clone();
    let tracked = match track_resumable_download(
        &state,
        &crate::download_queue::QueuedWork::StreamingAsr(request.clone()),
    )
    .await
    {
        Ok(tracked) => tracked,
        Err(error) => return error.into_response(),
    };
    let db = state.db.clone();
    let active_downloads = state.active_downloads.clone();
    let cache_state = state.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let (job_id, cancel) = tracked;
        let result = download::download_streaming_asr_snapshot_with_progress(
            &http,
            &data_dir,
            request,
            Box::new(move |event| {
                push_progress(&progress_tx, event);
            }),
            Some((db.clone(), job_id.clone())),
            Some(cancel),
        )
        .await;
        let stop = active_downloads.stop_reason(&job_id);
        active_downloads.finish(&job_id);
        if let Err(error) = &result {
            match stop {
                Some(crate::active_downloads::StopReason::Pause) => {
                    let _ = db.pause_download_job(&job_id).await;
                }
                Some(crate::active_downloads::StopReason::Cancel) => {
                    let _ = db.cancel_download_job(&job_id).await;
                }
                None => {
                    let _ = db.fail_download_job(&job_id, &error.to_string()).await;
                }
            }
        }
        match result {
            Ok(download_result) => {
                cache_state.invalidate_models_cache().await;
                push_progress(
                    &tx,
                    ProgressEvent::done(serde_json::to_value(&download_result).unwrap_or_default()),
                );
            }
            Err(error) => {
                push_progress(&tx, ProgressEvent::error(error.to_string()));
            }
        }
    });
    progress_sse(rx)
}

async fn download_personaplex_model(
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
    Json(request): Json<download::MlxDownloadRequest>,
) -> Response {
    let mut request = request;
    request.engine = voice::ENGINE.to_owned();
    if !query.stream {
        let result = download::download_personaplex_snapshot_with_progress(
            &state.http,
            &state.data_dir,
            request,
            Box::new(|_| {}),
            None,
            None,
        )
        .await;
        return match result {
            Ok(result) => {
                state.invalidate_models_cache().await;
                (StatusCode::OK, Json(json!(result))).into_response()
            }
            Err(error) => ApiError::bad_request(error).into_response(),
        };
    }

    let (tx, rx) = progress_channel();
    let http = state.http.clone();
    let data_dir = state.data_dir.clone();
    let tracked = match track_resumable_download(
        &state,
        &crate::download_queue::QueuedWork::Personaplex(request.clone()),
    )
    .await
    {
        Ok(tracked) => tracked,
        Err(error) => return error.into_response(),
    };
    let db = state.db.clone();
    let active_downloads = state.active_downloads.clone();
    let cache_state = state.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let (job_id, cancel) = tracked;
        let result = download::download_personaplex_snapshot_with_progress(
            &http,
            &data_dir,
            request,
            Box::new(move |event| {
                push_progress(&progress_tx, event);
            }),
            Some((db.clone(), job_id.clone())),
            Some(cancel),
        )
        .await;
        let stop = active_downloads.stop_reason(&job_id);
        active_downloads.finish(&job_id);
        if let Err(error) = &result {
            match stop {
                Some(crate::active_downloads::StopReason::Pause) => {
                    let _ = db.pause_download_job(&job_id).await;
                }
                Some(crate::active_downloads::StopReason::Cancel) => {
                    let _ = db.cancel_download_job(&job_id).await;
                }
                None => {
                    let _ = db.fail_download_job(&job_id, &error.to_string()).await;
                }
            }
        }
        match result {
            Ok(download_result) => {
                cache_state.invalidate_models_cache().await;
                push_progress(
                    &tx,
                    ProgressEvent::done(serde_json::to_value(&download_result).unwrap_or_default()),
                );
            }
            Err(error) => {
                push_progress(&tx, ProgressEvent::error(error.to_string()));
            }
        }
    });
    progress_sse(rx)
}

/// Every stable-diffusion.cpp bundle on offer, with what is already installed.
async fn sdcpp_catalog(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let mut data = Vec::new();
    for entry in sdcpp_catalog::catalog(&state.data_dir) {
        let mut value = bundle_json(&entry.bundle, entry.origin, &state.data_dir);
        // Fold the durable acceptance state into the bundle's consent
        // requirement, since `bundle_json` itself cannot touch the DB.
        if let Some(consent) = value.get_mut("consent").and_then(Value::as_object_mut)
            && let Some(requirement) = entry.bundle.license_requirement()
        {
            let accepted =
                crate::license_store::has_consent(&state.db, &requirement.id, &requirement.version)
                    .await
                    .unwrap_or(false);
            consent.insert("accepted".into(), Value::Bool(accepted));
        }
        data.push(value);
    }
    Ok(Json(json!({ "data": data })))
}

fn bundle_json(
    bundle: &sdcpp_catalog::Bundle,
    origin: sdcpp_catalog::Origin,
    data_dir: &std::path::Path,
) -> Value {
    let consent = bundle.license_requirement().map(|requirement| {
        json!({
            "id": requirement.id,
            "version": requirement.version,
            "url": requirement.url,
            "summary": requirement.summary,
        })
    });
    let conditioning =
        bundle
            .installed_conditioning(data_dir)
            .map(|conditioning| match conditioning {
                sdcpp::VideoConditioning::None => "text",
                sdcpp::VideoConditioning::InitImage => "init_image",
                sdcpp::VideoConditioning::FirstLastFrame => "first_last_frame",
                sdcpp::VideoConditioning::References => "references",
            });
    json!({
        "id": bundle.id,
        "label": bundle.label,
        "modality": bundle.modality,
        "key": bundle.key,
        "summary": bundle.summary,
        "license": bundle.license,
        // The full agreement fields travel with the bundle so an inline
        // round-trip (a variant install sends `{ bundle }` back) keeps what
        // validation and the consent gate need. `consent` carries the same
        // values for display; these mirror the daemon-side `Bundle` fields.
        "license_url": bundle.license_url,
        "license_version": bundle.license_version,
        "license_summary": bundle.license_summary,
        "requires_license_acceptance": bundle.requires_license_acceptance,
        "consent": consent,
        "model_id": bundle.model_id(),
        "installed": bundle.installed(data_dir),
        "conditioning": conditioning,
        "gated": bundle.gated(),
        "approx_bytes": bundle.approx_bytes(),
        "supports_init_image": bundle.supports_init_image,
        "featured": bundle.featured,
        "origin": origin,
        "defaults": bundle.defaults,
        "components": bundle.components.iter().map(|component| json!({
            "repo_id": component.repo_id,
            "path": component.path,
            "flag": component.flag,
            "role": component.role,
            "gated": component.gated,
            "approx_bytes": component.approx_bytes,
            "variants": component.variants,
        })).collect::<Vec<_>>(),
    })
}

/// Refuse to install a bundle whose license requires an acceptance the person
/// has not recorded.
///
/// This is enforced here in the daemon rather than only in the interface so a
/// direct API caller or the download queue cannot install a licensed model
/// past its agreement. The error names the license and is distinct enough for
/// the interface to route to the consent dialog.
async fn ensure_license_consent(
    state: &AppState,
    bundle: &sdcpp_catalog::Bundle,
) -> anyhow::Result<()> {
    let Some(requirement) = bundle.license_requirement() else {
        return Ok(());
    };
    let accepted =
        crate::license_store::has_consent(&state.db, &requirement.id, &requirement.version).await?;
    anyhow::ensure!(
        accepted,
        "{} is released under the {} license, which must be accepted before it can be installed. \
         Review the terms and agree to them to continue.",
        bundle.label,
        requirement.id,
    );
    Ok(())
}

#[derive(Debug, Deserialize)]
struct AssembleSdcppRequest {
    repo_id: String,
    path: String,
    #[serde(default)]
    modality: Option<sdcpp::Modality>,
}

/// Inspect a checkpoint on the Hub and propose a bundle for it.
///
/// Only the file's header is fetched, so this stays fast even for a 20 GB
/// model, and the proposal is returned for review rather than installed.
async fn assemble_sdcpp_bundle(
    State(state): State<AppState>,
    Json(request): Json<AssembleSdcppRequest>,
) -> ApiResult<Json<Value>> {
    let probe = sdcpp_arch::probe_hub_file(
        &state.http,
        &state.data_dir,
        &request.repo_id,
        &request.path,
    )
    .await
    .map_err(ApiError::bad_request)?;
    let proposal = sdcpp_arch::assemble(&request.repo_id, &request.path, &probe, request.modality);
    Ok(Json(json!({
        "bundle": bundle_json(&proposal.bundle, sdcpp_catalog::Origin::Custom, &state.data_dir),
        "architecture": proposal.architecture,
        "architecture_label": proposal.architecture_label,
        "variant": proposal.variant,
        "detected_by": proposal.detected_by,
        "self_contained": proposal.self_contained,
        "warnings": proposal.warnings,
    })))
}

/// Save a locally defined bundle, whether assembled or hand-written.
async fn save_sdcpp_bundle(
    State(state): State<AppState>,
    Json(bundle): Json<sdcpp_catalog::Bundle>,
) -> ApiResult<Json<Value>> {
    let saved =
        sdcpp_catalog::save_custom(&state.data_dir, bundle).map_err(ApiError::bad_request)?;
    Ok(Json(bundle_json(
        &saved,
        sdcpp_catalog::Origin::Custom,
        &state.data_dir,
    )))
}

async fn delete_sdcpp_bundle(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    sdcpp_catalog::delete_custom(&state.data_dir, &id).map_err(ApiError::bad_request)?;
    Ok(Json(json!({ "deleted": id })))
}

#[derive(Debug, Deserialize)]
struct SdcppConsentRequest {
    /// Installable id of the bundle whose license is being accepted.
    bundle_id: String,
}

/// Record acceptance of a bundle's license agreement.
///
/// Consent is durable and versioned: the daemon stores the version of the
/// terms the person saw, and installs require a matching record, so a license
/// that gets re-termed cannot be silently grandfathered.
async fn accept_sdcpp_license(
    State(state): State<AppState>,
    Json(request): Json<SdcppConsentRequest>,
) -> ApiResult<Json<Value>> {
    let bundle = sdcpp_catalog::find(&state.data_dir, &request.bundle_id).ok_or_else(|| {
        ApiError::bad_request(format!("unknown model bundle `{}`", request.bundle_id))
    })?;
    let requirement = bundle.license_requirement().ok_or_else(|| {
        ApiError::bad_request(format!(
            "{} does not require a license agreement",
            bundle.label
        ))
    })?;
    crate::license_store::record_consent(&state.db, &requirement.id, &requirement.version)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({
        "license_id": requirement.id,
        "version": requirement.version,
        "accepted": true,
        "bundle_id": bundle.id,
    })))
}

#[derive(Debug, Deserialize)]
struct InstallSdcppBundleRequest {
    #[serde(default)]
    id: Option<String>,
    /// A bundle to install directly, without saving it to the catalog first.
    #[serde(default)]
    bundle: Option<sdcpp_catalog::Bundle>,
}

async fn install_sdcpp_bundle(
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
    Json(request): Json<InstallSdcppBundleRequest>,
) -> Response {
    let bundle = match (request.bundle, request.id) {
        (Some(bundle), _) => match sdcpp_catalog::validate(&bundle) {
            Ok(()) => bundle,
            Err(error) => return ApiError::bad_request(error).into_response(),
        },
        (None, Some(id)) => match sdcpp_catalog::find(&state.data_dir, &id) {
            Some(bundle) => bundle,
            None => {
                return ApiError::bad_request(format!("unknown model bundle `{id}`"))
                    .into_response();
            }
        },
        (None, None) => {
            return ApiError::bad_request("id or bundle is required").into_response();
        }
    };

    if let Err(error) = ensure_license_consent(&state, &bundle).await {
        return ApiError::bad_request(error).into_response();
    }

    if !query.stream {
        let result = download::install_sdcpp_bundle_with_progress(
            &state.http,
            &state.data_dir,
            &bundle,
            Box::new(|_| {}),
            None,
            None,
        )
        .await;
        return match result {
            Ok(result) => {
                state.invalidate_models_cache().await;
                (StatusCode::OK, Json(json!(result))).into_response()
            }
            Err(error) => ApiError::bad_request(error).into_response(),
        };
    }

    let (tx, rx) = progress_channel();
    let http = state.http.clone();
    let data_dir = state.data_dir.clone();
    let tracked = match track_resumable_download(
        &state,
        &crate::download_queue::QueuedWork::SdcppBundle(bundle.clone()),
    )
    .await
    {
        Ok(tracked) => tracked,
        Err(error) => return error.into_response(),
    };
    let db = state.db.clone();
    let active_downloads = state.active_downloads.clone();
    let cache_state = state.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let (job_id, cancel) = tracked;
        let result = download::install_sdcpp_bundle_with_progress(
            &http,
            &data_dir,
            &bundle,
            Box::new(move |event| {
                push_progress(&progress_tx, event);
            }),
            Some((db.clone(), job_id.clone())),
            Some(cancel),
        )
        .await;
        let stop = active_downloads.stop_reason(&job_id);
        active_downloads.finish(&job_id);
        if let Err(error) = &result {
            match stop {
                Some(crate::active_downloads::StopReason::Pause) => {
                    let _ = db.pause_download_job(&job_id).await;
                }
                Some(crate::active_downloads::StopReason::Cancel) => {
                    let _ = db.cancel_download_job(&job_id).await;
                }
                None => {
                    let _ = db.fail_download_job(&job_id, &error.to_string()).await;
                }
            }
        }
        match result {
            Ok(install) => {
                cache_state.invalidate_models_cache().await;
                push_progress(
                    &tx,
                    ProgressEvent::done(serde_json::to_value(&install).unwrap_or_default()),
                );
            }
            Err(error) => {
                push_progress(&tx, ProgressEvent::error(error.to_string()));
            }
        }
    });
    progress_sse(rx)
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut request): Json<ChatCompletionRequest>,
) -> ApiResult<Response> {
    if headers
        .get("x-brazier-mode")
        .and_then(|value| value.to_str().ok())
        == Some("agent")
    {
        request.brazier_mode = Some("agent".to_owned());
    }
    if let Some(raw) = headers
        .get("x-brazier-slot")
        .and_then(|value| value.to_str().ok())
        && let Ok(slot) = raw.parse::<u32>()
    {
        request.llama_slot = Some(slot);
    }
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4().simple());

    if !request.stream {
        let generation = state
            .runtime
            .generate(&request)
            .await
            .map_err(ApiError::from_anyhow)?;
        let mut message = json!({
            "role": "assistant",
            "content": if generation.text.is_empty() {
                Value::Null
            } else {
                Value::String(generation.text.clone())
            },
            "reasoning_content": generation.reasoning
        });
        let finish_reason = if !generation.client_tool_calls.is_empty() {
            message["tool_calls"] = llama::tool_calls_to_json(&generation.client_tool_calls);
            "tool_calls"
        } else {
            "stop"
        };
        return Ok(Json(json!({
            "id": completion_id,
            "object": "chat.completion",
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "brazier": {
                "tool_calls": generation.tool_invocations,
                "transcript": generation.transcript,
            },
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }))
        .into_response());
    }

    let model = request.model.clone();
    let mut token_rx = state
        .runtime
        .generate_stream(&request)
        .await
        .map_err(ApiError::from_anyhow)?;
    let events = stream! {
        // Set once a terminal finish_reason has been sent, so the closing chunk
        // does not overwrite `tool_calls` with `stop`. Clients that map the last
        // finish_reason they see (including agent runtimes) would otherwise miss
        // the tool round trip.
        let mut finished = false;
        let mut saw_end = false;
        while let Some(item) = token_rx.recv().await {
            match item {
                Ok(StreamEvent::Load { phase, message }) => {
                    let chunk = json!({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": null
                        }],
                        "brazier": { "load": { "phase": phase, "message": message } }
                    });
                    yield Ok::<Event, Infallible>(Event::default().data(chunk.to_string()));
                }
                Ok(StreamEvent::PrefillProgress {
                    total,
                    cached,
                    processed,
                    elapsed_ms,
                    context_total,
                }) => {
                    let chunk = json!({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": null
                        }],
                        "brazier": { "prefill": {
                            "total": total,
                            "cached": cached,
                            "processed": processed,
                            "elapsed_ms": elapsed_ms,
                            "context_total": context_total
                        }}
                    });
                    yield Ok::<Event, Infallible>(Event::default().data(chunk.to_string()));
                }
                Ok(StreamEvent::Content(content)) => {
                    let chunk = json!({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": { "content": content },
                            "finish_reason": null
                        }]
                    });
                    yield Ok::<Event, Infallible>(Event::default().data(chunk.to_string()));
                }
                Ok(StreamEvent::Reasoning(reasoning)) => {
                    let chunk = json!({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": { "reasoning_content": reasoning },
                            "finish_reason": null
                        }]
                    });
                    yield Ok::<Event, Infallible>(Event::default().data(chunk.to_string()));
                }
                Ok(StreamEvent::Tool(invocation)) => {
                    // Brazier extension chunk: harmless to standard OpenAI clients.
                    let chunk = json!({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": null
                        }],
                        "brazier": { "tool_call": invocation }
                    });
                    yield Ok::<Event, Infallible>(Event::default().data(chunk.to_string()));
                }
                Ok(StreamEvent::ToolCallDelta(fragment)) => {
                    let chunk = json!({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [llama::tool_call_fragment_to_delta(&fragment)]
                            },
                            "finish_reason": null
                        }]
                    });
                    yield Ok::<Event, Infallible>(Event::default().data(chunk.to_string()));
                }
                Ok(StreamEvent::ClientToolCalls(calls)) => {
                    let chunk = json!({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": llama::tool_calls_to_json(&calls)
                            },
                            "finish_reason": "tool_calls"
                        }]
                    });
                    finished = true;
                    yield Ok::<Event, Infallible>(Event::default().data(chunk.to_string()));
                }
                Ok(StreamEvent::TranscriptMessage(message)) => {
                    let chunk = json!({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": null
                        }],
                        "brazier": { "transcript_message": message }
                    });
                    yield Ok::<Event, Infallible>(Event::default().data(chunk.to_string()));
                }
                Ok(StreamEvent::GenerationStats { prompt_tokens, completion_tokens, decode_duration_ms }) => {
                    let chunk = json!({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": null
                        }],
                        "brazier": { "generation": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "decode_duration_ms": decode_duration_ms
                        }}
                    });
                    yield Ok::<Event, Infallible>(Event::default().data(chunk.to_string()));
                }
                Ok(StreamEvent::End) => {
                    saw_end = true;
                    break;
                }
                Err(error) => {
                    tracing::error!(error = %error, "stream generation failed");
                    let fork_hints = error
                        .downcast_ref::<ModelLoadError>()
                        .map(|load| load.fork_hints.clone())
                        .filter(|hints| !hints.is_empty());
                    let mut chunk = json!({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "error"
                        }],
                        "error": { "message": error.to_string() }
                    });
                    if let Some(fork_hints) = fork_hints {
                        chunk["brazier"] = json!({ "fork_hints": fork_hints });
                    }
                    finished = true;
                    yield Ok(Event::default().data(chunk.to_string()));
                    break;
                }
            }
        }
        if !finished && saw_end {
            let final_chunk = json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            });
            yield Ok(Event::default().data(final_chunk.to_string()));
        } else if !finished {
            tracing::error!("chat completion stream closed without a terminal event");
            let chunk = json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error"
                }],
                "error": { "message": "model stream closed before completion" }
            });
            yield Ok(Event::default().data(chunk.to_string()));
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
        .into_response())
}

fn responses_input_to_messages(input: &Value) -> Vec<OpenAiMessage> {
    match input {
        Value::String(text) => vec![OpenAiMessage {
            role: "user".to_owned(),
            content: Value::String(text.clone()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }],
        Value::Array(items) => items
            .iter()
            .map(|item| OpenAiMessage {
                role: item
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("user")
                    .to_owned(),
                content: item.get("content").cloned().unwrap_or_else(|| item.clone()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            })
            .collect(),
        other => vec![OpenAiMessage {
            role: "user".to_owned(),
            content: Value::String(text_from_content(other)),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }],
    }
}

fn responses_failed_event(response_id: &str, error: &str) -> Value {
    json!({
        "type": "response.failed",
        "response": {
            "id": response_id,
            "status": "failed",
            "error": {
                "code": "server_error",
                "message": error
            }
        }
    })
}

async fn responses(
    State(state): State<AppState>,
    Json(request): Json<ResponsesRequest>,
) -> ApiResult<Response> {
    let chat_request = ChatCompletionRequest {
        model: request.model.clone(),
        messages: responses_input_to_messages(&request.input),
        stream: request.stream,
        tools: request.tools,
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_output_tokens,
        seed: request.seed,
        enable_reasoning: request.enable_reasoning,
        reasoning_budget_tokens: request.reasoning_budget_tokens,
        tool_choice: request.tool_choice,
        builtin_tools: request.builtin_tools,
        builtin_tool_names: None,
        llama_slot: None,
        brazier_mode: None,
    };
    let response_id = format!("resp_{}", Uuid::new_v4().simple());
    if !request.stream {
        let generation = state
            .runtime
            .generate(&chat_request)
            .await
            .map_err(ApiError::internal)?;
        return Ok(Json(json!({
            "id": response_id,
            "object": "response",
            "status": "completed",
            "model": request.model,
            "output": [{
                "id": format!("msg_{}", Uuid::new_v4().simple()),
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": generation.text,
                    "annotations": []
                }]
            }],
            "output_text": generation.text
        }))
        .into_response());
    }

    let mut token_rx = state
        .runtime
        .generate_stream(&chat_request)
        .await
        .map_err(ApiError::internal)?;
    let events = stream! {
        let mut failure = Some("model stream closed before completion".to_owned());
        yield Ok::<Event, Infallible>(Event::default()
            .event("response.created")
            .data(json!({"type": "response.created", "response": {"id": response_id, "status": "in_progress"}}).to_string()));
        while let Some(item) = token_rx.recv().await {
            match item {
                Ok(StreamEvent::Load { .. } | StreamEvent::PrefillProgress { .. }) => {}
                Ok(StreamEvent::Content(content)) => {
                    yield Ok(Event::default()
                        .event("response.output_text.delta")
                        .data(json!({"type": "response.output_text.delta", "delta": content}).to_string()));
                }
                Ok(StreamEvent::Reasoning(_)) => {}
                Ok(StreamEvent::Tool(invocation)) => {
                    yield Ok(Event::default()
                        .event("response.brazier.tool_call")
                        .data(json!({"type": "response.brazier.tool_call", "tool_call": invocation}).to_string()));
                }
                Ok(StreamEvent::ToolCallDelta(_)) => {}
                Ok(StreamEvent::ClientToolCalls(calls)) => {
                    yield Ok(Event::default()
                        .event("response.brazier.client_tool_calls")
                        .data(json!({
                            "type": "response.brazier.client_tool_calls",
                            "tool_calls": llama::tool_calls_to_json(&calls)
                        }).to_string()));
                }
                Ok(StreamEvent::TranscriptMessage(message)) => {
                    yield Ok(Event::default()
                        .event("response.brazier.transcript_message")
                        .data(json!({
                            "type": "response.brazier.transcript_message",
                            "message": message
                        }).to_string()));
                }
                Ok(StreamEvent::GenerationStats { .. }) => {}
                Ok(StreamEvent::End) => {
                    failure = None;
                    break;
                }
                Err(error) => {
                    tracing::error!(error = %error, "responses stream failed");
                    failure = Some(error.to_string());
                    break;
                }
            }
        }
        if let Some(error) = failure {
            yield Ok(Event::default()
                .event("response.failed")
                .data(responses_failed_event(&response_id, &error).to_string()));
        } else {
            yield Ok(Event::default()
                .event("response.completed")
                .data(json!({"type": "response.completed", "response": {"id": response_id, "status": "completed"}}).to_string()));
        }
    };
    Ok(Sse::new(events).into_response())
}

// ---------------------------------------------------------------------------
// Agent mode
//
// The agent runtime is a separate process with no host privileges. It reaches
// the machine only through these endpoints, which apply the policy broker,
// the sandbox, and the execution broker in that order.
// ---------------------------------------------------------------------------

/// Sandbox backend, permission modes, and runtime descriptors for Agent mode.
async fn agent_capabilities(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let sandbox = state.agent_broker.capabilities();
    let default_runtime_id = load_default_agent_runtime_id(&state).await?;
    Ok(Json(json!({
        "schema_version": 1,
        "sandbox": sandbox,
        "permission_modes": ["ask", "sandbox-only", "skip-permissions"],
        // The daemon advertises stock adapters and whether they can run here.
        // Package versions are reported by the agent worker when it loads them.
        "runtimes": agent_runtime_catalog(),
        "default_runtime_id": default_runtime_id,
        "tool_output_limit_chars": 24_000,
    })))
}

fn agent_tool_definitions(data_dir: &std::path::Path) -> Value {
    let mut definitions = crate::agent_tools::definitions()["data"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for server in mcp::enabled_servers(data_dir) {
        for tool in server.tools {
            let tool_name = tool.name;
            let name = mcp::openai_tool_name(&server.id, &tool_name);
            let description = tool.description.unwrap_or_else(|| {
                format!(
                    "Call {tool_name} on the configured {} MCP server.",
                    server.name
                )
            });
            let schema = if tool.input_schema.is_null() {
                json!({ "type": "object", "properties": {} })
            } else {
                tool.input_schema
            };
            definitions.push(json!({
                "name": name,
                "label": format!("{} · {}", server.name, tool_name),
                "description": description,
                "input_schema": schema,
                // MCP servers are configured host processes. Agent mode never
                // presents them as sandboxed calls.
                "risk": "execute",
                "executes": true,
                "needs_workspace": false,
                "default_environment": "host",
                "source": "mcp",
            }));
        }
    }
    json!({ "data": definitions })
}

fn agent_tool_names(data_dir: &std::path::Path) -> Vec<String> {
    agent_tool_definitions(data_dir)["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry["name"].as_str().map(str::to_owned))
        .collect()
}

fn validate_agent_enabled_tools(data_dir: &std::path::Path, enabled: &[String]) -> ApiResult<()> {
    let available: std::collections::HashSet<String> =
        agent_tool_names(data_dir).into_iter().collect();
    let unknown: Vec<&str> = enabled
        .iter()
        .map(String::as_str)
        .filter(|name| !available.contains(*name))
        .collect();
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(ApiError::bad_request(format!(
            "unknown agent tools: {}",
            unknown.join(", ")
        )))
    }
}

/// The tool names a session gets by default for a given mode: the base set
/// (everything but power tools) for `simple`, plus the operator-enabled power
/// tools for `powerful`.
fn default_enabled_tools(
    runtime_id: &str,
    enabled_power_tools: &[String],
    data_dir: &std::path::Path,
) -> Vec<String> {
    let power: std::collections::HashSet<String> =
        crate::agent_tools::power_tool_names().into_iter().collect();
    let base: Vec<String> = agent_tool_names(data_dir)
        .into_iter()
        .filter(|name| !power.contains(name))
        .collect();
    if runtime_id != crate::agent_types::AGENT_RUNTIME_POWERFUL {
        return base;
    }
    let mut tools = base;
    for name in enabled_power_tools {
        if power.contains(name) && !tools.contains(name) {
            tools.push(name.clone());
        }
    }
    tools
}

/// `simple` mode never exposes power tools, even through an explicit per-session
/// tool list. `powerful` accepts any known power tool.
fn validate_agent_tools_for_runtime(
    runtime_id: &str,
    enabled: &[String],
    data_dir: &std::path::Path,
) -> ApiResult<()> {
    validate_agent_enabled_tools(data_dir, enabled)?;
    if runtime_id == crate::agent_types::AGENT_RUNTIME_POWERFUL {
        return Ok(());
    }
    let power: std::collections::HashSet<String> =
        crate::agent_tools::power_tool_names().into_iter().collect();
    let power_in_simple: Vec<&str> = enabled
        .iter()
        .map(String::as_str)
        .filter(|name| power.contains(*name))
        .collect();
    if power_in_simple.is_empty() {
        Ok(())
    } else {
        Err(ApiError::bad_request(format!(
            "`simple` mode does not expose power tools ({}). Switch to Powerful mode to use them.",
            power_in_simple.join(", ")
        )))
    }
}

async fn agent_tool_catalog(State(state): State<AppState>) -> Json<Value> {
    Json(agent_tool_definitions(&state.data_dir))
}

async fn list_agent_sessions(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let sessions = state
        .db
        .list_agent_sessions()
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "data": sessions })))
}

async fn create_agent_session(
    State(state): State<AppState>,
    client: ClientAddr,
    Json(mut request): Json<crate::agent_types::CreateAgentSession>,
) -> ApiResult<Json<crate::agent_types::AgentSessionRecord>> {
    let confine = request.confine_to_worktree;
    request.confine_to_worktree = false;
    if let Some(workspace) = request.workspace_path.as_deref() {
        request.workspace_path = Some(
            validate_workspace_path(&state, workspace)?
                .display()
                .to_string(),
        );
    }
    if let Some(enabled) = request.enabled_tools.as_deref() {
        validate_agent_enabled_tools(&state.data_dir, enabled)?;
    }
    let default_runtime_id = load_default_agent_runtime_id(&state).await?;
    let runtime_id = resolve_agent_runtime_id(request.runtime_id.take(), &default_runtime_id);
    validate_agent_runtime_id(&runtime_id)?;
    let elevated = matches!(
        request.permission_mode,
        Some(crate::agent_types::AgentPermissionMode::SkipPermissions)
    ) || request
        .permission_settings
        .is_some_and(|settings| settings.auto_approve_host_actions);
    require_elevated_permission_step_up(
        elevated,
        request.confirm_elevated_permissions,
        client_is_loopback(&client),
    )?;
    // The mode decides the default tool set: `simple` gets the base tools,
    // `powerful` adds the power tools the operator enabled in Manage → Agent.
    // An explicit per-session list still wins, but must respect the mode.
    let enabled_power_tools = load_enabled_power_tools(&state).await?;
    match request.enabled_tools.as_deref() {
        Some(enabled) => validate_agent_tools_for_runtime(&runtime_id, enabled, &state.data_dir)?,
        None => {
            request.enabled_tools = Some(default_enabled_tools(
                &runtime_id,
                &enabled_power_tools,
                &state.data_dir,
            ));
        }
    }
    request.runtime_id = Some(runtime_id);
    let mut session = state
        .db
        .create_agent_session(request)
        .await
        .map_err(ApiError::internal)?;
    if confine {
        match set_worktree_confinement(&state, session.clone(), true, false).await {
            Ok(confined) => session = confined,
            Err(error) => {
                if let Err(cleanup) = state.db.delete_agent_session(&session.id).await {
                    tracing::error!(
                        session_id = %session.id,
                        error = %cleanup,
                        "failed to roll back agent session after worktree setup failed"
                    );
                }
                return Err(error);
            }
        }
    }
    Ok(Json(session))
}

async fn get_agent_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let session = state
        .db
        .agent_session(&id)
        .await
        .map_err(|error| ApiError::not_found(error.to_string()))?;
    let messages = state
        .db
        .agent_messages(&id)
        .await
        .map_err(ApiError::internal)?;
    let executions = state
        .db
        .list_tool_executions(&id)
        .await
        .map_err(ApiError::internal)?;
    let pending = state
        .db
        .pending_approvals(&id)
        .await
        .map_err(ApiError::internal)?;
    let grants = state
        .db
        .session_grants(&id)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({
        "session": session,
        "messages": messages,
        "tool_executions": executions,
        "pending_approvals": pending,
        "grants": grants,
        "sandbox": state.agent_broker.capabilities(),
    })))
}

async fn patch_agent_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
    client: ClientAddr,
    Json(mut update): Json<crate::agent_types::UpdateAgentSession>,
) -> ApiResult<Json<crate::agent_types::AgentSessionRecord>> {
    let confine = update.confine_to_worktree.take();
    let discard_unapplied = update.discard_unapplied.take().unwrap_or(false);
    let confirm_elevated = update.confirm_elevated_permissions.take().unwrap_or(false);
    let has_other_updates = update.title.is_some()
        || update.workspace_path.is_some()
        || update.model.is_some()
        || update.permission_mode.is_some()
        || update.permission_settings.is_some()
        || update.enabled_tools.is_some()
        || update.last_run_status.is_some()
        || update.compaction.is_some()
        || update.runtime_metadata.is_some();
    if confine.is_some() && has_other_updates {
        return Err(ApiError::bad_request(
            "worktree confinement must be changed in a separate request",
        ));
    }
    if discard_unapplied && confine != Some(false) {
        return Err(ApiError::bad_request(
            "discard_unapplied is only valid when turning worktree confinement off",
        ));
    }
    if let Some(Some(workspace)) = update.workspace_path.as_ref() {
        let resolved = validate_workspace_path(&state, workspace)?
            .display()
            .to_string();
        update.workspace_path = Some(Some(resolved));
    }
    if let Some(enabled) = update.enabled_tools.as_deref() {
        validate_agent_enabled_tools(&state.data_dir, enabled)?;
    }
    if let Some(enabled) = confine {
        let session = state
            .db
            .agent_session(&id)
            .await
            .map_err(|error| ApiError::not_found(error.to_string()))?;
        return Ok(Json(
            set_worktree_confinement(&state, session, enabled, discard_unapplied).await?,
        ));
    }
    let existing = state
        .db
        .agent_session(&id)
        .await
        .map_err(|error| ApiError::not_found(error.to_string()))?;
    if let Some(enabled) = update.enabled_tools.as_deref() {
        validate_agent_tools_for_runtime(&existing.runtime_id, enabled, &state.data_dir)?;
    }
    let elevating_mode = matches!(
        update.permission_mode,
        Some(crate::agent_types::AgentPermissionMode::SkipPermissions)
    ) && !matches!(
        existing.permission_mode,
        crate::agent_types::AgentPermissionMode::SkipPermissions
    );
    let elevating_host = update.permission_settings.is_some_and(|settings| {
        settings.auto_approve_host_actions
            && !existing.permission_settings.auto_approve_host_actions
    });
    require_elevated_permission_step_up(
        elevating_mode || elevating_host,
        confirm_elevated,
        client_is_loopback(&client),
    )?;
    let session = state
        .db
        .update_agent_session(&id, update)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(session))
}

#[derive(Debug, Deserialize)]
struct DeleteAgentSessionQuery {
    #[serde(default)]
    discard_unapplied: bool,
}

async fn delete_agent_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<DeleteAgentSessionQuery>,
) -> ApiResult<Json<Value>> {
    let session = state
        .db
        .agent_session(&id)
        .await
        .map_err(|error| ApiError::not_found(error.to_string()))?;
    // Refuse discardable worktrees before tearing down the agent, so a missing
    // discard flag cannot kill the run and then leave the session behind.
    if let Some(info) =
        crate::agent_worktree::worktree_from_metadata(session.runtime_metadata.as_ref())
    {
        let status = crate::agent_worktree::inspect_worktree(&info)
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?;
        if status.has_discardable_changes && !query.discard_unapplied {
            return Err(ApiError::bad_request(format!(
                "worktree {} has unapplied changes; apply, commit, or discard them before cleanup",
                info.path
            )));
        }
    }
    state.agent_broker.terminate_session_processes(&id).await;
    if let Some(info) =
        crate::agent_worktree::worktree_from_metadata(session.runtime_metadata.as_ref())
    {
        crate::agent_worktree::remove_worktree(&info, query.discard_unapplied)
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?;
    }
    state
        .db
        .delete_agent_session(&id)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "deleted": true })))
}

/// Inspect a session's managed worktree before delete/unconfine confirmation.
async fn agent_worktree_status(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let session = state
        .db
        .agent_session(&id)
        .await
        .map_err(|error| ApiError::not_found(error.to_string()))?;
    let Some(info) =
        crate::agent_worktree::worktree_from_metadata(session.runtime_metadata.as_ref())
    else {
        return Ok(Json(json!({ "worktree": null })));
    };
    let status = crate::agent_worktree::inspect_worktree(&info)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!({ "worktree": status })))
}

/// Copy the task's worktree delta into the source checkout for local testing.
async fn apply_agent_worktree(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let session = state
        .db
        .agent_session(&id)
        .await
        .map_err(|error| ApiError::not_found(error.to_string()))?;
    if session.last_run_status == "running" {
        return Err(ApiError::bad_request(
            "stop the agent before applying its worktree changes",
        ));
    }
    let info = crate::agent_worktree::worktree_from_metadata(session.runtime_metadata.as_ref())
        .ok_or_else(|| ApiError::bad_request("this task does not use a managed worktree"))?;
    let applied = crate::agent_worktree::apply_to_source(&info)
        .await
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let metadata = crate::agent_worktree::metadata_with_worktree(
        session.runtime_metadata.clone(),
        Some(applied.worktree),
    );
    let session = state
        .db
        .update_agent_session(
            &id,
            crate::agent_types::UpdateAgentSession {
                runtime_metadata: Some(metadata),
                ..Default::default()
            },
        )
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({
        "session": session,
        "changed_paths": applied.changed_paths,
        "already_up_to_date": applied.already_up_to_date,
    })))
}

async fn list_agent_messages(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let messages = state
        .db
        .agent_messages(&id)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "data": messages })))
}

async fn append_agent_messages(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<crate::agent_types::AppendAgentMessages>,
) -> ApiResult<Json<Value>> {
    // Confirm the session exists before writing rows against it.
    state
        .db
        .agent_session(&id)
        .await
        .map_err(|error| ApiError::not_found(error.to_string()))?;
    let messages = state
        .db
        .append_agent_messages(&id, &request.messages, request.replace)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "data": messages })))
}

async fn list_agent_tool_executions(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let executions = state
        .db
        .list_tool_executions(&id)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "data": executions })))
}

async fn list_agent_approvals(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let pending = state
        .db
        .pending_approvals(&id)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "data": pending })))
}

/// Stop a run: kill the session's processes and refuse anything still waiting
/// on the user, so no approved-after-the-fact command runs later.
async fn cancel_agent_run(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    state
        .db
        .agent_session(&id)
        .await
        .map_err(|error| ApiError::not_found(error.to_string()))?;
    let terminated = state.agent_broker.terminate_session_processes(&id).await;
    let expired = state
        .db
        .expire_pending_approvals(Some(&id))
        .await
        .map_err(ApiError::internal)?;
    state.agent_broker.notify_approvals();
    state
        .db
        .update_agent_session(
            &id,
            crate::agent_types::UpdateAgentSession {
                last_run_status: Some("cancelled".to_owned()),
                ..Default::default()
            },
        )
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({
        "terminated_processes": terminated,
        "expired_approvals": expired,
    })))
}

/// Effective system prompt for a session: the workspace override when one is
/// saved, otherwise the application default built from live session state.
async fn agent_system_prompt(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let session = state
        .db
        .agent_session(&id)
        .await
        .map_err(|error| ApiError::not_found(error.to_string()))?;
    let names = match session.enabled_tools.clone() {
        Some(tools) => tools,
        // Legacy sessions predate mode-aware defaults: derive the mode's set
        // from the stored runtime id so power tools never leak into Simple.
        None => {
            let enabled_power_tools = match session.runtime_id.as_str() {
                crate::agent_types::AGENT_RUNTIME_POWERFUL => {
                    load_enabled_power_tools(&state).await.unwrap_or_default()
                }
                _ => Vec::new(),
            };
            default_enabled_tools(&session.runtime_id, &enabled_power_tools, &state.data_dir)
        }
    };
    let workspace_key = agent_session_workspace_key(&state, &session)?;
    let custom = match workspace_key {
        Some(ref key) => state
            .db
            .agent_workspace_system_prompt(key)
            .await
            .map_err(ApiError::internal)?,
        None => None,
    };
    let customized = custom.is_some();
    let template = custom
        .as_deref()
        .unwrap_or(crate::agent_tools::DEFAULT_SYSTEM_PROMPT_TEMPLATE);
    let components = crate::agent_tools::system_prompt_components(
        &session,
        &state.agent_broker.capabilities(),
        &names,
    );
    let prompt = crate::agent_tools::render_system_prompt(template, &components);
    Ok(Json(json!({
        "system_prompt": prompt,
        "tools": names,
        "customized": customized,
    })))
}

const MAX_AGENT_SYSTEM_PROMPT_BYTES: usize = 64 * 1024;

#[derive(Debug, Deserialize)]
struct AgentWorkspacePromptQuery {
    workspace_path: String,
    #[serde(default)]
    session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UpdateAgentWorkspacePrompt {
    workspace_path: String,
    /// Null restores the generated application default.
    system_prompt: Option<String>,
}

fn agent_session_workspace_key(
    state: &AppState,
    session: &crate::agent_types::AgentSessionRecord,
) -> ApiResult<Option<String>> {
    let worktree = crate::agent_worktree::worktree_from_metadata(session.runtime_metadata.as_ref());
    let raw = worktree
        .as_ref()
        .map(|info| info.source_path.as_str())
        .or(session.workspace_path.as_deref());
    raw.map(|path| {
        validate_workspace_path(state, path).map(|resolved| resolved.display().to_string())
    })
    .transpose()
}

fn workspace_prompt_preview_session(
    workspace_path: String,
) -> crate::agent_types::AgentSessionRecord {
    crate::agent_types::AgentSessionRecord {
        id: "workspace-prompt-preview".to_owned(),
        title: "Prompt preview".to_owned(),
        workspace_path: Some(workspace_path),
        model: "preview".to_owned(),
        runtime_id: crate::agent_types::AGENT_RUNTIME_SIMPLE.to_owned(),
        permission_mode: crate::agent_types::AgentPermissionMode::Ask,
        permission_settings: crate::agent_types::AgentPermissionSettings::default(),
        enabled_tools: None,
        last_run_status: "idle".to_owned(),
        compaction: None,
        runtime_metadata: None,
        created_at: String::new(),
        updated_at: String::new(),
    }
}

fn prompt_components_json(components: &[(&str, String)]) -> Value {
    Value::Array(
        components
            .iter()
            .map(|(name, content)| {
                json!({
                    "name": name,
                    "placeholder": format!("{{{name}}}"),
                    "content": content,
                })
            })
            .collect(),
    )
}

async fn get_agent_workspace_prompt(
    State(state): State<AppState>,
    Query(query): Query<AgentWorkspacePromptQuery>,
) -> ApiResult<Json<Value>> {
    let workspace_path = validate_workspace_path(&state, &query.workspace_path)?
        .display()
        .to_string();
    let session = if let Some(id) = query.session_id {
        let session = state
            .db
            .agent_session(&id)
            .await
            .map_err(|error| ApiError::not_found(error.to_string()))?;
        let session_key = agent_session_workspace_key(&state, &session)?;
        if session_key.as_deref() != Some(workspace_path.as_str()) {
            return Err(ApiError::bad_request(
                "agent session does not belong to this workspace",
            ));
        }
        Some(session)
    } else {
        None
    };
    state
        .db
        .remember_agent_workspace(&workspace_path)
        .await
        .map_err(ApiError::internal)?;
    let custom = state
        .db
        .agent_workspace_system_prompt(&workspace_path)
        .await
        .map_err(ApiError::internal)?;
    let customized = custom.is_some();
    let template =
        custom.unwrap_or_else(|| crate::agent_tools::DEFAULT_SYSTEM_PROMPT_TEMPLATE.to_owned());
    let session =
        session.unwrap_or_else(|| workspace_prompt_preview_session(workspace_path.clone()));
    let names = session
        .enabled_tools
        .clone()
        .unwrap_or_else(|| agent_tool_names(&state.data_dir));
    let components = crate::agent_tools::system_prompt_components(
        &session,
        &state.agent_broker.capabilities(),
        &names,
    );
    let resolved_prompt = crate::agent_tools::render_system_prompt(&template, &components);
    Ok(Json(json!({
        "workspace_path": workspace_path,
        "system_prompt": template,
        "resolved_prompt": resolved_prompt,
        "components": prompt_components_json(&components),
        "customized": customized,
    })))
}

async fn put_agent_workspace_prompt(
    State(state): State<AppState>,
    Json(request): Json<UpdateAgentWorkspacePrompt>,
) -> ApiResult<Json<Value>> {
    let workspace_path = validate_workspace_path(&state, &request.workspace_path)?
        .display()
        .to_string();
    if let Some(prompt) = request.system_prompt.as_deref()
        && prompt.len() > MAX_AGENT_SYSTEM_PROMPT_BYTES
    {
        return Err(ApiError::bad_request(
            "agent system prompt cannot exceed 64 KiB",
        ));
    }
    state
        .db
        .set_agent_workspace_system_prompt(&workspace_path, request.system_prompt.as_deref())
        .await
        .map_err(ApiError::internal)?;
    let customized = request.system_prompt.is_some();
    let system_prompt = request
        .system_prompt
        .unwrap_or_else(|| crate::agent_tools::DEFAULT_SYSTEM_PROMPT_TEMPLATE.to_owned());
    let session = workspace_prompt_preview_session(workspace_path.clone());
    let names = agent_tool_names(&state.data_dir);
    let components = crate::agent_tools::system_prompt_components(
        &session,
        &state.agent_broker.capabilities(),
        &names,
    );
    let resolved_prompt = crate::agent_tools::render_system_prompt(&system_prompt, &components);
    Ok(Json(json!({
        "workspace_path": workspace_path,
        "system_prompt": system_prompt,
        "resolved_prompt": resolved_prompt,
        "components": prompt_components_json(&components),
        "customized": customized,
    })))
}

/// The only way an agent tool call reaches the machine.
async fn agent_exec_tool(
    State(state): State<AppState>,
    Json(request): Json<crate::agent_types::ToolExecRequest>,
) -> ApiResult<Json<crate::agent_types::ToolExecResponse>> {
    let session = state
        .db
        .agent_session(&request.session_id)
        .await
        .map_err(|error| ApiError::not_found(error.to_string()))?;
    if let Some(enabled) = &session.enabled_tools
        && !enabled.iter().any(|name| name == &request.tool)
    {
        return Err(ApiError::bad_request(format!(
            "tool `{}` is not enabled for this session",
            request.tool
        )));
    }
    if crate::agent_policy::is_mcp_tool_name(&request.tool)
        && !agent_tool_names(&state.data_dir).contains(&request.tool)
    {
        return Err(ApiError::bad_request(format!(
            "MCP tool `{}` is not enabled or advertised",
            request.tool
        )));
    }
    let context = crate::agent_exec::BrokerContext {
        broker: state.agent_broker.as_ref(),
        db: &state.db,
        data_dir: &state.data_dir,
        session: &session,
    };
    let response = crate::agent_exec::execute(&context, &request)
        .await
        .map_err(ApiError::from_anyhow)?;
    Ok(Json(response))
}

/// Stream foreground tool output while preserving the normal final response.
///
/// Each SSE `output` event contains one display chunk. The terminal `result`
/// event contains the same response `/exec` would return, including persisted
/// execution and artifact identifiers.
async fn agent_exec_tool_stream(
    State(state): State<AppState>,
    Json(request): Json<crate::agent_types::ToolExecRequest>,
) -> ApiResult<Sse<impl futures::Stream<Item = Result<Event, Infallible>>>> {
    let session = state
        .db
        .agent_session(&request.session_id)
        .await
        .map_err(|error| ApiError::not_found(error.to_string()))?;
    if let Some(enabled) = &session.enabled_tools
        && !enabled.iter().any(|name| name == &request.tool)
    {
        return Err(ApiError::bad_request(format!(
            "tool `{}` is not enabled for this session",
            request.tool
        )));
    }
    if crate::agent_policy::is_mcp_tool_name(&request.tool)
        && !agent_tool_names(&state.data_dir).contains(&request.tool)
    {
        return Err(ApiError::bad_request(format!(
            "MCP tool `{}` is not enabled or advertised",
            request.tool
        )));
    }

    let events = stream! {
        let context = crate::agent_exec::BrokerContext {
            broker: state.agent_broker.as_ref(),
            db: &state.db,
            data_dir: &state.data_dir,
            session: &session,
        };
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();
        let execution = crate::agent_exec::execute_streaming(&context, &request, output_tx);
        tokio::pin!(execution);
        loop {
            tokio::select! {
                biased;
                Some(chunk) = output_rx.recv() => {
                    yield Ok(Event::default()
                        .event("output")
                        .data(json!({ "chunk": chunk }).to_string()));
                }
                result = &mut execution => {
                    match result {
                        Ok(response) => {
                            yield Ok(Event::default()
                                .event("result")
                                .data(serde_json::to_string(&response).unwrap_or_else(|error| {
                                    json!({ "error": error.to_string() }).to_string()
                                })));
                        }
                        Err(error) => {
                            yield Ok(Event::default()
                                .event("error")
                                .data(json!({ "message": error.to_string() }).to_string()));
                        }
                    }
                    break;
                }
            }
        }
    };
    Ok(Sse::new(events).keep_alive(KeepAlive::default()))
}

#[derive(Debug, Deserialize)]
struct ApprovalWaitQuery {
    /// Block until the approval is decided, up to this many milliseconds.
    #[serde(default)]
    wait_ms: Option<u64>,
}

/// Read an approval. With `wait_ms`, block until the user answers, so the agent
/// worker does not have to poll.
async fn get_agent_approval(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<ApprovalWaitQuery>,
) -> ApiResult<Json<crate::agent_types::AgentApproval>> {
    use crate::agent_types::ApprovalStatus;

    let deadline =
        std::time::Instant::now() + Duration::from_millis(query.wait_ms.unwrap_or(0).min(600_000));
    let notifier = state.agent_broker.approvals_notifier();
    loop {
        // Time out stale requests before reporting, so a forgotten dialog does
        // not leave a run blocked forever.
        state
            .db
            .expire_pending_approvals(None)
            .await
            .map_err(ApiError::internal)?;
        let approval = state
            .db
            .approval(&id)
            .await
            .map_err(|error| ApiError::not_found(error.to_string()))?;
        if approval.status != ApprovalStatus::Pending {
            return Ok(Json(approval));
        }
        let now = std::time::Instant::now();
        if now >= deadline {
            return Ok(Json(approval));
        }
        // Woken by a decision, or re-checked periodically for expiry.
        let notified = notifier.notified();
        let _ = tokio::time::timeout((deadline - now).min(Duration::from_millis(1_000)), notified)
            .await;
    }
}

async fn decide_agent_approval(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<crate::agent_types::ApprovalDecisionRequest>,
) -> ApiResult<Json<crate::agent_types::AgentApproval>> {
    let approved = match request.decision.as_str() {
        "approve" => true,
        "deny" => false,
        other => {
            return Err(ApiError::bad_request(format!(
                "decision must be `approve` or `deny`, not `{other}`"
            )));
        }
    };
    let approval = state
        .db
        .decide_approval(&id, approved, request.scope, request.note)
        .await
        .map_err(ApiError::bad_request)?;
    // The decision is already durable. Wake the waiting call before doing
    // auxiliary timeline bookkeeping so a logging failure cannot strand it or
    // make the client retry a decision that has already been committed.
    state.agent_broker.notify_approvals();

    // An approved call records itself when it runs. A refused one never runs, so
    // record it here — otherwise the attempt would vanish from the activity
    // timeline as soon as the session is reloaded.
    if !approved {
        let note = approval.note.clone();
        if let Err(error) = state
            .db
            .record_tool_execution(crate::agent_store::NewToolExecution {
                session_id: approval.session_id.clone(),
                run_id: None,
                tool_call_id: None,
                tool: approval.tool.clone(),
                arguments: approval.arguments.clone(),
                environment: approval.environment,
                risk: approval.risk,
                status: "denied".to_owned(),
                exit_code: None,
                output_preview: Some(format!(
                    "The user denied this action.{}",
                    note.as_deref()
                        .map(|note| format!(" Note: {note}"))
                        .unwrap_or_default()
                )),
                artifact_id: None,
                truncated: false,
                changed_paths: Vec::new(),
                sandbox: Some(approval.sandbox.clone()),
                approval_id: Some(approval.id.clone()),
                error: Some(note.unwrap_or_else(|| "denied by the user".to_owned())),
                duration_ms: None,
            })
            .await
        {
            tracing::error!(
                approval_id = %approval.id,
                session_id = %approval.session_id,
                error = %error,
                "approval was denied but its timeline entry could not be recorded"
            );
        }
    }

    Ok(Json(approval))
}

/// Full text of a stored tool output. Truncated output is never lost.
async fn get_agent_artifact(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Response> {
    let (_, path, _) = state
        .db
        .artifact(&id)
        .await
        .map_err(|error| ApiError::not_found(error.to_string()))?;
    let body = tokio::fs::read(&path).await.map_err(ApiError::internal)?;
    Ok((
        [(header::CONTENT_TYPE, HeaderValue::from_static("text/plain"))],
        body,
    )
        .into_response())
}

#[derive(Debug, Deserialize)]
struct WorkspaceRequest {
    path: String,
}

/// Check that a folder can serve as an agent workspace before a session uses it.
async fn validate_agent_workspace(
    State(state): State<AppState>,
    Json(request): Json<WorkspaceRequest>,
) -> ApiResult<Json<Value>> {
    let resolved = validate_workspace_path(&state, &request.path)?;
    let git = crate::agent_worktree::is_git_repository(&resolved).await;
    Ok(Json(json!({
        "path": resolved.display().to_string(),
        "git_repository": git,
        "sandbox": state.agent_broker.capabilities(),
    })))
}

/// Point the session at a fresh git worktree, or restore the source checkout.
async fn set_worktree_confinement(
    state: &AppState,
    session: crate::agent_types::AgentSessionRecord,
    enabled: bool,
    discard_unapplied: bool,
) -> ApiResult<crate::agent_types::AgentSessionRecord> {
    let existing = crate::agent_worktree::worktree_from_metadata(session.runtime_metadata.as_ref());
    if session.last_run_status == "running" {
        return Err(ApiError::bad_request(
            "stop the agent before changing worktree confinement",
        ));
    }
    if enabled {
        if existing.is_some() {
            return Ok(session);
        }
        let source = session.workspace_path.as_deref().ok_or_else(|| {
            ApiError::bad_request("choose a workspace before confining to a worktree")
        })?;
        let source_path = validate_workspace_path(state, source)?;
        if !crate::agent_worktree::is_git_repository(&source_path).await {
            return Err(ApiError::bad_request(
                "worktree confinement needs a git repository workspace",
            ));
        }
        let info =
            crate::agent_worktree::create_worktree(&source_path, &session.id, &session.title)
                .await
                .map_err(|error| ApiError::bad_request(error.to_string()))?;
        let metadata = crate::agent_worktree::metadata_with_worktree(
            session.runtime_metadata.clone(),
            Some(info.clone()),
        );
        validate_workspace_path(state, &info.path)?;
        match state
            .db
            .update_agent_session(
                &session.id,
                crate::agent_types::UpdateAgentSession {
                    workspace_path: Some(Some(info.path.clone())),
                    runtime_metadata: Some(metadata),
                    ..Default::default()
                },
            )
            .await
        {
            Ok(session) => Ok(session),
            Err(error) => {
                if let Err(cleanup) = crate::agent_worktree::remove_worktree(&info, true).await {
                    tracing::error!(
                        session_id = %session.id,
                        error = %cleanup,
                        "failed to clean up worktree after session update failed"
                    );
                }
                Err(ApiError::internal(error))
            }
        }
    } else {
        let Some(info) = existing else {
            return Ok(session);
        };
        let metadata =
            crate::agent_worktree::metadata_with_worktree(session.runtime_metadata.clone(), None);
        let source = validate_workspace_path(state, &info.source_path)?;
        let updated = state
            .db
            .update_agent_session(
                &session.id,
                crate::agent_types::UpdateAgentSession {
                    workspace_path: Some(Some(source.display().to_string())),
                    runtime_metadata: Some(metadata),
                    ..Default::default()
                },
            )
            .await
            .map_err(ApiError::internal)?;
        if let Err(error) = crate::agent_worktree::remove_worktree(&info, discard_unapplied).await {
            let rollback = state
                .db
                .update_agent_session(
                    &session.id,
                    crate::agent_types::UpdateAgentSession {
                        workspace_path: Some(session.workspace_path.clone()),
                        runtime_metadata: session.runtime_metadata.clone(),
                        ..Default::default()
                    },
                )
                .await;
            if let Err(rollback) = rollback {
                tracing::error!(
                    session_id = %session.id,
                    error = %error,
                    rollback_error = %rollback,
                    "worktree removal and session rollback both failed"
                );
                return Err(ApiError::internal(rollback));
            }
            return Err(ApiError::bad_request(error.to_string()));
        }
        Ok(updated)
    }
}

/// A workspace must exist, be a directory, and sit outside Brazier's own data
/// and credential paths.
fn validate_workspace_path(state: &AppState, raw: &str) -> ApiResult<PathBuf> {
    let path = PathBuf::from(raw);
    if !path.is_absolute() {
        return Err(ApiError::bad_request("the workspace path must be absolute"));
    }
    let resolved = std::fs::canonicalize(&path)
        .map_err(|_| ApiError::bad_request(format!("{raw} does not exist")))?;
    if !resolved.is_dir() {
        return Err(ApiError::bad_request(format!("{raw} is not a directory")));
    }
    for secret in crate::agent_sandbox::secret_paths(Some(&state.data_dir)) {
        if crate::agent_policy::is_inside(&resolved, &secret) {
            return Err(ApiError::bad_request(format!(
                "{raw} is a credential or Brazier-owned path and cannot be an agent workspace"
            )));
        }
    }
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from)
        && resolved == home
    {
        return Err(ApiError::bad_request(
            "choose a project folder rather than the whole home directory",
        ));
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        active_downloads::ActiveDownloads,
        db::Database,
        download_queue::DownloadQueue,
        engine::Runtime,
        models_store::{self, download_destination},
    };
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use std::{io::Read, sync::Arc};
    use tempfile::tempdir;
    use tokio::sync::Mutex;
    use tower::ServiceExt;
    use zip::ZipArchive;

    #[test]
    fn allows_the_ui_and_whatever_else_was_named() {
        let origins = parse_origins(&["https://studio.example.com".into()]).unwrap();
        let rendered: Vec<&str> = origins
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect();
        // The packaged renderer is a file:// page, whose origin is `null`.
        assert!(rendered.contains(&"null"));
        assert!(rendered.contains(&"http://localhost:5173"));
        assert!(rendered.contains(&"https://studio.example.com"));

        // Comma-separated, for the environment variable form.
        let from_env = parse_origins(&["http://a.example:8080, http://b.example".into()]).unwrap();
        assert_eq!(from_env.len(), DEFAULT_ORIGINS.len() + 2);
        // Naming an origin twice does not repeat it in the header.
        assert_eq!(
            parse_origins(&["null".into(), "http://localhost:5173".into()])
                .unwrap()
                .len(),
            DEFAULT_ORIGINS.len()
        );
    }

    /// The daemon holds a machine's conversations and can execute tools. "Any
    /// page may call it" is not something a typo should be able to turn on.
    #[test]
    fn refuses_a_wildcard_and_things_that_are_not_origins() {
        for bad in [
            "*",
            "example.com",
            "https://example.com/",
            "https://example.com/path",
            "ftp://example.com",
        ] {
            assert!(
                parse_origins(&[bad.to_owned()]).is_err(),
                "{bad} must be refused"
            );
        }
    }

    #[test]
    fn converts_responses_string_input() {
        let messages = responses_input_to_messages(&json!("hello"));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }

    #[test]
    fn responses_failures_are_terminal_failures_not_completions() {
        let event = responses_failed_event("resp_test", "engine exited");
        assert_eq!(event["type"], "response.failed");
        assert_eq!(event["response"]["id"], "resp_test");
        assert_eq!(event["response"]["status"], "failed");
        assert_eq!(event["response"]["error"]["code"], "server_error");
        assert_eq!(event["response"]["error"]["message"], "engine exited");
    }

    async fn test_state(data_dir: &std::path::Path) -> AppState {
        let http = reqwest::Client::new();
        let runtime = Runtime::new(data_dir.to_path_buf(), http.clone());
        let active_downloads = Arc::new(ActiveDownloads::new());
        let db = Database::open(&data_dir.join("brazier.sqlite"))
            .await
            .unwrap();
        let download_queue = DownloadQueue::spawn(
            http.clone(),
            data_dir.to_path_buf(),
            db.clone(),
            Arc::clone(&active_downloads),
            Arc::clone(&runtime),
        );
        AppState {
            db,
            runtime,
            api_keys: Vec::new(),
            http,
            data_dir: data_dir.to_path_buf(),
            active_builds: Arc::new(builds::ActiveBuilds::new()),
            build_slots: Arc::new(tokio::sync::Semaphore::new(1)),
            active_downloads,
            download_queue,
            runtimes_cache: Arc::new(Mutex::new(None)),
            agent_broker: Arc::new(crate::agent_exec::AgentBroker::new()),
            computer_broker: Arc::new(crate::computer_exec::ComputerBroker::new()),
        }
    }

    #[tokio::test]
    async fn registers_tool_generated_blob_before_indexing_its_message() {
        let dir = tempdir().unwrap();
        let state = test_state(dir.path()).await;
        let stored = blob_store::store_bytes(
            dir.path(),
            b"\x89PNG\r\n\x1a\nfake image",
            "image/png",
            Some("generated.png"),
        )
        .await
        .unwrap();
        let content = json!([{
            "type": "brazier_blob",
            "brazier_blob": {
                "sha256": stored.sha256,
                "mime_type": "image/png",
                "name": "generated.png"
            }
        }]);

        register_message_blobs(&state, &content).await.unwrap();
        let conversation = state
            .db
            .create_conversation("Generated image")
            .await
            .unwrap();
        let message = state
            .db
            .create_message(
                &conversation.id,
                CreateMessage {
                    parent_id: None,
                    role: crate::types::Role::Assistant,
                    content,
                    model: None,
                    tool_calls: None,
                    tool_call_id: None,
                    source: Some("assistant_chat".into()),
                    correlation_id: None,
                    status: None,
                    metadata: Some(json!({ "generated_media_display": true })),
                },
            )
            .await
            .unwrap();

        let messages = state.db.list_messages(&conversation.id).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].id, message.id);
    }

    async fn json_request(
        app: &Router,
        method: &str,
        uri: &str,
        body: Value,
    ) -> (StatusCode, Value) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let parsed = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, parsed)
    }

    #[tokio::test]
    async fn download_controls_reject_false_success_and_cancel_paused_jobs() {
        let dir = tempdir().unwrap();
        let state = test_state(dir.path()).await;
        let job = state
            .db
            .create_download_job("acme/models", "queued.gguf", "main")
            .await
            .unwrap();
        let app = router(state.clone());

        let (status, body) = json_request(
            &app,
            "POST",
            "/api/v1/models/download/pause",
            json!({ "job_id": &job.id }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            state
                .db
                .get_download_job_public(&job.id)
                .await
                .unwrap()
                .status,
            "paused"
        );

        let (status, body) = json_request(
            &app,
            "POST",
            "/api/v1/models/download/cancel",
            json!({ "job_id": &job.id }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            state
                .db
                .get_download_job_public(&job.id)
                .await
                .unwrap()
                .status,
            "cancelled"
        );

        for route in [
            "/api/v1/models/download/pause",
            "/api/v1/models/download/cancel",
        ] {
            let (status, _) =
                json_request(&app, "POST", route, json!({ "job_id": "does-not-exist" })).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{route}");
        }
    }

    #[tokio::test]
    async fn agent_session_failures_do_not_leave_partial_state_or_report_success() {
        let dir = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);

        let (status, body) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test",
                "confine_to_worktree": true
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        let (status, sessions) = get_request(&app, "/api/v1/agent/sessions").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(sessions["data"], json!([]));

        let (status, _) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions/missing/cancel",
            json!({}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) =
            json_request(&app, "DELETE", "/api/v1/agent/sessions/missing", json!({})).await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        let (_, session) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test"
            }),
        )
        .await;
        let session_id = session["id"].as_str().unwrap();
        let (status, _) = json_request(
            &app,
            "PATCH",
            &format!("/api/v1/agent/sessions/{session_id}"),
            json!({
                "title": "Must not persist",
                "confine_to_worktree": true
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (_, reloaded) =
            get_request(&app, &format!("/api/v1/agent/sessions/{session_id}")).await;
        assert_eq!(reloaded["session"]["title"], "Agent task");
    }

    #[tokio::test]
    async fn workspace_prompt_starts_with_the_full_default_and_can_be_overridden() {
        let dir = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let workspace_path = workspace.path().display().to_string();
        let app = router(test_state(dir.path()).await);
        let uri = format!("/api/v1/agent/workspaces/prompt?workspace_path={workspace_path}");

        let (status, default) = get_request(&app, &uri).await;
        assert_eq!(status, StatusCode::OK, "{default}");
        assert_eq!(default["customized"], false);
        assert!(
            default["system_prompt"]
                .as_str()
                .unwrap()
                .contains("{identity}")
        );
        assert!(
            default["resolved_prompt"]
                .as_str()
                .unwrap()
                .contains("You are Brazier's coding and system agent")
        );
        assert_eq!(default["components"].as_array().unwrap().len(), 6);

        let (status, saved) = json_request(
            &app,
            "PUT",
            "/api/v1/agent/workspaces/prompt",
            json!({
                "workspace_path": workspace_path,
                "system_prompt": "Use the repository snapshot tests.\n\n{workspace}\n\n{tools}"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{saved}");
        assert_eq!(saved["customized"], true);
        assert!(
            saved["resolved_prompt"]
                .as_str()
                .unwrap()
                .contains("Workspace:")
        );

        let (_, session) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test"
            }),
        )
        .await;
        let session_id = session["id"].as_str().unwrap();
        let (_, prompt) =
            get_request(&app, &format!("/api/v1/agent/sessions/{session_id}/prompt")).await;
        let effective = prompt["system_prompt"].as_str().unwrap();
        assert!(
            effective.starts_with("Use the repository snapshot tests.\n\nWorkspace:"),
            "{effective}"
        );
        assert!(effective.contains("Available tools:"), "{effective}");
        assert_eq!(prompt["customized"], true);

        let (status, _) = json_request(
            &app,
            "PATCH",
            &format!("/api/v1/agent/sessions/{session_id}"),
            json!({ "enabled_tools": ["fs_read"] }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (_, restricted_prompt) =
            get_request(&app, &format!("/api/v1/agent/sessions/{session_id}/prompt")).await;
        assert!(
            restricted_prompt["system_prompt"]
                .as_str()
                .unwrap()
                .contains("Available tools: fs_read."),
            "{restricted_prompt}"
        );

        json_request(
            &app,
            "PUT",
            "/api/v1/agent/workspaces/prompt",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "system_prompt": null
            }),
        )
        .await;
        let (_, reset) =
            get_request(&app, &format!("/api/v1/agent/sessions/{session_id}/prompt")).await;
        assert_eq!(reset["customized"], false);
        assert!(
            reset["system_prompt"]
                .as_str()
                .unwrap()
                .contains("You are Brazier's coding and system agent")
        );
    }

    #[tokio::test]
    async fn welcome_completion_is_persisted_in_the_database() {
        let dir = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);

        let (status, initial) = get_request(&app, "/api/v1/preferences/welcome").await;
        assert_eq!(status, StatusCode::OK, "{initial}");
        assert_eq!(initial["completed"], false);

        let (status, saved) = json_request(
            &app,
            "PUT",
            "/api/v1/preferences/welcome",
            json!({ "completed": true }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{saved}");
        assert_eq!(saved["completed"], true);

        let (_, reloaded) = get_request(&app, "/api/v1/preferences/welcome").await;
        assert_eq!(reloaded["completed"], true);
    }

    #[tokio::test]
    async fn workspace_modes_preference_round_trips() {
        let dir = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);

        let (status, initial) = get_request(&app, "/api/v1/preferences/workspace").await;
        assert_eq!(status, StatusCode::OK, "{initial}");
        assert_eq!(initial["modes"]["chat"], true);
        assert_eq!(initial["modes"]["computer"], false);

        let (status, saved) = json_request(
            &app,
            "PUT",
            "/api/v1/preferences/workspace",
            json!({
                "modes": {
                    "chat": true,
                    "agent": false,
                    "generate": true,
                    "voice": false,
                    "computer": true
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{saved}");
        assert_eq!(saved["modes"]["computer"], true);
        assert_eq!(saved["modes"]["agent"], false);

        let (_, reloaded) = get_request(&app, "/api/v1/preferences/workspace").await;
        assert_eq!(reloaded["modes"]["computer"], true);
    }

    #[tokio::test]
    #[ignore = "requires a working Chromium and local TCP loopback"]
    async fn computer_session_executes_browser_screenshot() {
        let dir = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);

        let (status, session) = json_request(
            &app,
            "POST",
            "/api/v1/computer/sessions",
            json!({ "title": "Browse", "target": "browser" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{session}");
        let session_id = session["id"].as_str().unwrap();

        let (status, result) = json_request(
            &app,
            "POST",
            "/api/v1/computer/exec",
            json!({
                "session_id": session_id,
                "action": { "type": "screenshot" }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{result}");
        assert_eq!(result["status"], "ok");
        assert!(result["screenshot_base64"].as_str().unwrap().len() > 10);

        let (status, parsed) = json_request(
            &app,
            "POST",
            "/api/v1/computer/parse-fara",
            json!({
                "text": "<tool_call>{\"name\":\"computer_use\",\"arguments\":{\"action\":\"visit_url\",\"url\":\"https://example.com\"}}</tool_call>"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{parsed}");
        assert_eq!(parsed["actions"][0]["type"], "visit_url");
    }

    async fn get_request(app: &Router, uri: &str) -> (StatusCode, Value) {
        let response = app
            .clone()
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let parsed = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, parsed)
    }

    #[tokio::test]
    async fn daemon_info_is_authenticated_and_describes_the_versioned_boundary() {
        let dir = tempdir().unwrap();
        let mut state = test_state(dir.path()).await;
        state.api_keys = vec!["test-service-key".into()];
        let app = router(state);

        let (status, _) = get_request(&app, "/api/v1/daemon/info").await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/daemon/info")
                    .header("authorization", "Bearer test-service-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let info: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(info["product"], "brazier");
        assert_eq!(info["management_api"]["major"], 1);
        assert_eq!(info["openai_api"]["responses"], "/v1/responses");
    }

    #[tokio::test]
    async fn any_configured_api_key_authenticates_but_unknown_ones_do_not() {
        let dir = tempdir().unwrap();
        let mut state = test_state(dir.path()).await;
        state.api_keys = vec!["key-one".into(), "key-two".into()];
        let app = router(state);

        for key in ["key-one", "key-two"] {
            let (status, _) = get_request(&app, "/api/v1/daemon/info").await;
            assert_eq!(status, StatusCode::UNAUTHORIZED);
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/api/v1/daemon/info")
                        .header("authorization", format!("Bearer {key}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/daemon/info")
                    .header("authorization", "Bearer key-three")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn support_bundle_route_returns_a_reviewable_zip() {
        let dir = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/support/bundle")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/zip");
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let mut archive = ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
        let mut report = String::new();
        archive
            .by_name("diagnostics.json")
            .unwrap()
            .read_to_string(&mut report)
            .unwrap();
        let report: Value = serde_json::from_str(&report).unwrap();
        assert_eq!(report["format_version"], 1);
        assert_eq!(report["privacy"]["conversations_included"], false);
        assert_eq!(report["privacy"]["credentials_included"], false);
        assert!(report["engine"].is_object());
        assert!(report["runtimes"].is_array());
    }

    /// The approval round trip an agent worker performs: a held call, a user
    /// decision, then the same call with the approval attached.
    #[tokio::test]
    async fn agent_tool_calls_wait_for_approval_over_http() {
        let dir = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);

        let (status, session) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "title": "Add a test",
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{session}");
        let session_id = session["id"].as_str().unwrap().to_owned();
        assert_eq!(session["permission_mode"], "ask");

        let arguments = json!({ "path": "hello.txt", "content": "hi" });
        let (status, held) = json_request(
            &app,
            "POST",
            "/api/v1/agent/exec",
            json!({
                "session_id": session_id,
                "tool": "fs_write",
                "arguments": arguments,
                "tool_call_id": "call-1"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{held}");
        assert_eq!(held["status"], "approval_required");
        assert!(!workspace.path().join("hello.txt").exists());
        let approval_id = held["approval"]["id"].as_str().unwrap().to_owned();
        assert!(held["approval"]["summary"].as_str().is_some());

        // A pending approval is visible to the UI even if the run is restarted.
        let (_, pending) = get_request(
            &app,
            &format!("/api/v1/agent/sessions/{session_id}/approvals"),
        )
        .await;
        assert_eq!(pending["data"].as_array().unwrap().len(), 1);

        let (status, decided) = json_request(
            &app,
            "POST",
            &format!("/api/v1/agent/approvals/{approval_id}"),
            json!({ "decision": "approve", "scope": "once" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{decided}");
        assert_eq!(decided["status"], "approved");

        let (status, done) = json_request(
            &app,
            "POST",
            "/api/v1/agent/exec",
            json!({
                "session_id": session_id,
                "tool": "fs_write",
                "arguments": arguments,
                "tool_call_id": "call-1",
                "approval_id": approval_id
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{done}");
        assert_eq!(done["status"], "completed", "{done}");
        assert_eq!(
            std::fs::read_to_string(workspace.path().join("hello.txt")).unwrap(),
            "hi"
        );

        // The ledger holds one row for the call that ran, pointing at the
        // approval that authorized it. The held attempt lives in the approval
        // table rather than duplicating the timeline.
        let (_, executions) = get_request(
            &app,
            &format!("/api/v1/agent/sessions/{session_id}/tool-executions"),
        )
        .await;
        let rows = executions["data"].as_array().unwrap();
        assert_eq!(rows.len(), 1, "{executions}");
        assert_eq!(rows[0]["status"], "completed");
        assert_eq!(rows[0]["approval_id"], json!(approval_id));
        assert_eq!(rows[0]["changed_paths"], json!(["hello.txt"]));
    }

    #[tokio::test]
    async fn a_denied_approval_is_kept_in_the_tool_ledger() {
        let dir = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);
        let (_, session) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test"
            }),
        )
        .await;
        let session_id = session["id"].as_str().unwrap().to_owned();
        let (_, held) = json_request(
            &app,
            "POST",
            "/api/v1/agent/exec",
            json!({
                "session_id": session_id,
                "tool": "shell_run",
                "arguments": { "command": "rm -rf /" }
            }),
        )
        .await;
        let approval_id = held["approval"]["id"].as_str().unwrap().to_owned();
        json_request(
            &app,
            "POST",
            &format!("/api/v1/agent/approvals/{approval_id}"),
            json!({ "decision": "deny", "note": "absolutely not" }),
        )
        .await;

        let (_, executions) = get_request(
            &app,
            &format!("/api/v1/agent/sessions/{session_id}/tool-executions"),
        )
        .await;
        let rows = executions["data"].as_array().unwrap();
        assert_eq!(rows.len(), 1, "{executions}");
        assert_eq!(rows[0]["status"], "denied");
        assert_eq!(rows[0]["error"], "absolutely not");
        assert_eq!(rows[0]["tool"], "shell_run");
    }

    #[tokio::test]
    async fn agent_capabilities_never_overstate_the_sandbox() {
        let dir = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);
        let (status, capabilities) = get_request(&app, "/api/v1/agent/capabilities").await;
        assert_eq!(status, StatusCode::OK);
        let sandbox = &capabilities["sandbox"];
        let isolated = sandbox["isolated"].as_bool().unwrap();
        // Whatever the host offers, the claim and the detail must agree.
        if isolated {
            assert_ne!(sandbox["backend"], "none");
            assert_eq!(sandbox["sandboxed_execution"], json!(true));
        } else {
            assert_eq!(sandbox["backend"], "none");
            assert_eq!(sandbox["sandboxed_execution"], json!(false));
            assert!(sandbox["detail"].as_str().unwrap().len() > 10);
        }
        assert!(
            capabilities["permission_modes"]
                .as_array()
                .unwrap()
                .contains(&json!("skip-permissions"))
        );
        let runtimes = capabilities["runtimes"].as_array().unwrap();
        assert!(runtimes.iter().any(|entry| entry["id"] == "simple"));
        assert!(runtimes.iter().any(|entry| entry["id"] == "powerful"));
        assert_eq!(capabilities["default_runtime_id"], "simple");
        let simple = runtimes
            .iter()
            .find(|entry| entry["id"] == "simple")
            .unwrap();
        assert_eq!(simple["available"], true);
        assert_eq!(simple["trust"], "broker");
    }

    #[tokio::test]
    async fn agent_default_runtime_preference_is_persisted() {
        let dir = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);
        let (status, initial) = get_request(&app, "/api/v1/preferences/agent").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(initial["default_runtime_id"], "simple");
        // Nothing configured yet: Powerful mode defaults to every power tool on.
        let all_power_tools: Vec<String> = crate::agent_tools::power_tool_names();
        assert_eq!(
            initial["power_tools"],
            serde_json::to_value(all_power_tools).unwrap()
        );

        let (status, migrated) = json_request(
            &app,
            "PUT",
            "/api/v1/preferences/agent",
            json!({ "default_runtime_id": "omp" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{migrated}");
        assert_eq!(migrated["default_runtime_id"], "powerful");

        let (status, body) = json_request(
            &app,
            "PUT",
            "/api/v1/preferences/agent",
            json!({ "default_runtime_id": "powerful", "power_tools": ["web_search", "nope"] }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["default_runtime_id"], "powerful");
        // Unknown power tools are dropped, known ones survive.
        assert_eq!(body["power_tools"], json!(["web_search"]));

        let (status, rejected) = json_request(
            &app,
            "PUT",
            "/api/v1/preferences/agent",
            json!({ "default_runtime_id": "nope" }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{rejected}");
    }

    #[tokio::test]
    async fn power_tools_default_on_for_powerful_sessions_only() {
        let dir = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);
        json_request(
            &app,
            "PUT",
            "/api/v1/preferences/agent",
            json!({ "default_runtime_id": "powerful", "power_tools": ["web_search"] }),
        )
        .await;

        let (_, simple_session) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test",
                "runtime_id": "simple"
            }),
        )
        .await;
        let simple_tools = simple_session["enabled_tools"]
            .as_array()
            .expect("stored tool list");
        assert!(simple_tools.iter().any(|entry| entry == "shell_run"));
        assert!(
            !simple_tools.iter().any(|entry| entry == "web_search"),
            "simple mode must not expose power tools"
        );

        let (_, powerful_session) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test",
                "runtime_id": "powerful"
            }),
        )
        .await;
        let powerful_tools = powerful_session["enabled_tools"]
            .as_array()
            .expect("stored tool list");
        assert!(powerful_tools.iter().any(|entry| entry == "web_search"));
        assert!(
            !powerful_tools.iter().any(|entry| entry == "web_fetch"),
            "only the enabled power tools are on by default"
        );

        // Explicitly asking for a power tool on a simple session is refused.
        let (status, body) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test",
                "runtime_id": "simple",
                "enabled_tools": ["shell_run", "web_search"]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }

    #[tokio::test]
    async fn the_agent_tool_catalog_is_served_with_schemas() {
        let dir = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);
        let (status, catalog) = get_request(&app, "/api/v1/agent/tools").await;
        assert_eq!(status, StatusCode::OK);
        let tools = catalog["data"].as_array().unwrap();
        assert!(tools.iter().any(|tool| tool["name"] == "shell_run"));
        for tool in tools {
            assert!(tool["description"].as_str().unwrap().len() > 20);
            assert_eq!(tool["input_schema"]["type"], "object");
            assert!(tool["risk"].as_str().is_some());
        }
    }

    #[tokio::test]
    async fn enabled_mcp_tools_join_agent_catalog_and_policy() {
        let dir = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        mcp::save(
            dir.path(),
            &mcp::McpConfig {
                servers: vec![mcp::McpServerConfig {
                    id: "demo".into(),
                    name: "Demo server".into(),
                    command: "/definitely/not/run-during-this-test".into(),
                    args: Vec::new(),
                    env: std::collections::HashMap::new(),
                    enabled: true,
                    tools: vec![mcp::McpToolEntry {
                        name: "lookup".into(),
                        description: Some("Look up a value through the demo server.".into()),
                        input_schema: json!({
                            "type": "object",
                            "properties": { "query": { "type": "string" } },
                            "required": ["query"]
                        }),
                    }],
                }],
            },
        )
        .await
        .unwrap();
        let app = router(test_state(dir.path()).await);

        let (status, catalog) = get_request(&app, "/api/v1/agent/tools").await;
        assert_eq!(status, StatusCode::OK);
        let tool = catalog["data"]
            .as_array()
            .unwrap()
            .iter()
            .find(|tool| tool["name"] == "mcp/demo/lookup")
            .expect("MCP tool in agent catalog");
        assert_eq!(tool["default_environment"], "host");
        assert_eq!(tool["risk"], "execute");
        assert_eq!(tool["input_schema"]["required"], json!(["query"]));

        let (_, session) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test"
            }),
        )
        .await;
        let session_id = session["id"].as_str().unwrap();
        let (_, prompt) =
            get_request(&app, &format!("/api/v1/agent/sessions/{session_id}/prompt")).await;
        assert!(
            prompt["tools"]
                .as_array()
                .unwrap()
                .contains(&json!("mcp/demo/lookup"))
        );

        let (status, held) = json_request(
            &app,
            "POST",
            "/api/v1/agent/exec",
            json!({
                "session_id": session_id,
                "tool": "mcp/demo/lookup",
                "arguments": { "query": "value" }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{held}");
        assert_eq!(held["status"], "approval_required");
        assert_eq!(held["environment"], "host");
        assert_eq!(
            held["approval"]["elevation"]["requested_network_access"],
            true
        );
        assert_eq!(held["approval"]["allow_session_scope"], false);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn approved_agent_mcp_call_executes_and_is_recorded() {
        let dir = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let server = r#"
count=0
while IFS= read -r line; do
  count=$((count + 1))
  if [ "$count" -eq 1 ]; then
    printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"test","version":"1"}}}'
  elif [ "$count" -eq 3 ]; then
    printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"mcp-agent-ok"}]}}'
  fi
done
"#;
        mcp::save(
            dir.path(),
            &mcp::McpConfig {
                servers: vec![mcp::McpServerConfig {
                    id: "demo".into(),
                    name: "Demo server".into(),
                    command: "/bin/sh".into(),
                    args: vec!["-c".into(), server.into()],
                    env: std::collections::HashMap::new(),
                    enabled: true,
                    tools: vec![mcp::McpToolEntry {
                        name: "lookup".into(),
                        description: Some("Look up a test value.".into()),
                        input_schema: json!({ "type": "object", "properties": {} }),
                    }],
                }],
            },
        )
        .await
        .unwrap();
        let app = router(test_state(dir.path()).await);
        let (_, session) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test"
            }),
        )
        .await;
        let session_id = session["id"].as_str().unwrap();
        let arguments = json!({ "query": "value" });
        let (_, held) = json_request(
            &app,
            "POST",
            "/api/v1/agent/exec",
            json!({
                "session_id": session_id,
                "tool": "mcp/demo/lookup",
                "arguments": arguments,
                "tool_call_id": "mcp-call"
            }),
        )
        .await;
        let approval_id = held["approval"]["id"].as_str().unwrap();
        json_request(
            &app,
            "POST",
            &format!("/api/v1/agent/approvals/{approval_id}"),
            json!({ "decision": "approve", "scope": "once" }),
        )
        .await;
        let (status, done) = json_request(
            &app,
            "POST",
            "/api/v1/agent/exec",
            json!({
                "session_id": session_id,
                "tool": "mcp/demo/lookup",
                "arguments": arguments,
                "tool_call_id": "mcp-call",
                "approval_id": approval_id
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{done}");
        assert_eq!(done["status"], "completed", "{done}");
        assert_eq!(done["output"], "mcp-agent-ok");
        assert_eq!(done["environment"], "host");

        let (_, executions) = get_request(
            &app,
            &format!("/api/v1/agent/sessions/{session_id}/tool-executions"),
        )
        .await;
        let record = &executions["data"][0];
        assert_eq!(record["tool"], "mcp/demo/lookup");
        assert_eq!(record["approval_id"], approval_id);
        assert_eq!(record["status"], "completed");
    }

    #[tokio::test]
    async fn elevated_permission_modes_require_explicit_confirmation() {
        let dir = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);
        let (status, body) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test",
                "permission_mode": "skip-permissions"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("confirm_elevated_permissions"),
            "{body}"
        );

        let (status, body) = json_request(
            &app,
            "POST",
            "/api/v1/computer/sessions",
            json!({ "permission_mode": "allow-all" }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("confirm_elevated_permissions"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn a_session_restricted_to_some_tools_refuses_the_others() {
        let dir = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);
        let (_, session) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test",
                "permission_mode": "skip-permissions",
                "confirm_elevated_permissions": true,
                "enabled_tools": ["fs_read", "fs_list"]
            }),
        )
        .await;
        let session_id = session["id"].as_str().unwrap();
        let (status, refused) = json_request(
            &app,
            "POST",
            "/api/v1/agent/exec",
            json!({
                "session_id": session_id,
                "tool": "shell_run",
                "arguments": { "command": "echo nope" }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    }

    #[tokio::test]
    async fn agent_sessions_reject_unknown_enabled_tools() {
        let dir = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);
        let (status, refused) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test",
                "enabled_tools": ["fs_read", "imaginary_tool"]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
        assert!(
            refused["error"]["message"]
                .as_str()
                .unwrap()
                .contains("imaginary_tool")
        );

        let (_, session) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test"
            }),
        )
        .await;
        let session_id = session["id"].as_str().unwrap();
        let (status, refused) = json_request(
            &app,
            "PATCH",
            &format!("/api/v1/agent/sessions/{session_id}"),
            json!({ "enabled_tools": ["also_imaginary"] }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    }

    #[tokio::test]
    async fn a_workspace_must_be_an_existing_directory_outside_brazier_state() {
        let dir = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);
        let (status, _) = json_request(
            &app,
            "POST",
            "/api/v1/agent/workspace",
            json!({ "path": "/definitely/not/here" }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (status, _) = json_request(
            &app,
            "POST",
            "/api/v1/agent/workspace",
            json!({ "path": dir.path().display().to_string() }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "the daemon data directory cannot be a workspace"
        );

        let workspace = tempdir().unwrap();
        let (status, body) = json_request(
            &app,
            "POST",
            "/api/v1/agent/workspace",
            json!({ "path": workspace.path().display().to_string() }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["git_repository"], false);
    }

    #[tokio::test]
    async fn cancelling_a_run_expires_pending_approvals() {
        let dir = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let app = router(test_state(dir.path()).await);
        let (_, session) = json_request(
            &app,
            "POST",
            "/api/v1/agent/sessions",
            json!({
                "workspace_path": workspace.path().display().to_string(),
                "model": "gguf:test"
            }),
        )
        .await;
        let session_id = session["id"].as_str().unwrap().to_owned();
        let (_, held) = json_request(
            &app,
            "POST",
            "/api/v1/agent/exec",
            json!({
                "session_id": session_id,
                "tool": "fs_delete",
                "arguments": { "path": "anything.txt" }
            }),
        )
        .await;
        let approval_id = held["approval"]["id"].as_str().unwrap().to_owned();

        let (status, cancelled) = json_request(
            &app,
            "POST",
            &format!("/api/v1/agent/sessions/{session_id}/cancel"),
            json!({}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{cancelled}");
        assert_eq!(cancelled["expired_approvals"], 1);

        // An approval answered after cancellation must not authorize anything.
        let (status, _) = json_request(
            &app,
            "POST",
            &format!("/api/v1/agent/approvals/{approval_id}"),
            json!({ "decision": "approve" }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn lists_disk_gguf_and_rejects_unknown_model() {
        let dir = tempdir().unwrap();
        let gguf = download_destination(dir.path(), "acme/demo", "weights.gguf").unwrap();
        std::fs::create_dir_all(gguf.parent().unwrap()).unwrap();
        std::fs::write(&gguf, b"not-a-real-gguf").unwrap();
        let app = router(test_state(dir.path()).await);

        let models_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(models_response.status(), StatusCode::OK);
        let models_body = models_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let models: Value = serde_json::from_slice(&models_body).unwrap();
        let ids: Vec<&str> = models["data"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|model| model["id"].as_str())
            .collect();
        assert!(!ids.contains(&"brazier/mock"));
        assert!(ids.contains(&"gguf:acme/demo/weights.gguf"));
        assert_eq!(
            models_store::path_for_model_id(dir.path(), "gguf:acme/demo/weights.gguf", &[])
                .unwrap(),
            gguf
        );

        let chat_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "brazier/mock",
                            "messages": [{"role": "user", "content": "hello path"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(chat_response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}

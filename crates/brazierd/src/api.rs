use std::{convert::Infallible, path::PathBuf, time::Duration};

use async_stream::stream;
use axum::http::header;
use axum::{
    Json, Router,
    extract::{Path, Query, Request, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
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
    AppState, blob_store,
    build_recipe::{self, BuildPlanRequest},
    builds,
    db::ConversationExport,
    db::CreateRunSnapshot,
    download::{self},
    engine::{Engine, StreamEvent},
    fork_hints::{self, ModelLoadError, RuntimeForkHint},
    hf::{self, SearchQuery},
    hf_auth, llama, mcp, media, model_bindings, models_store,
    progress::ProgressEvent,
    runtimes, sdcpp, sdcpp_arch, sdcpp_catalog, streaming_asr, tool_registry, toolchain_hints,
    types::{
        ChatCompletionRequest, CreateConversation, CreateMessage, OpenAiMessage, ResponsesRequest,
        text_from_content,
    },
    voice, whisper, whisperkit,
};

type ApiResult<T> = Result<T, ApiError>;

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

pub fn router(state: AppState) -> Router {
    let protected = Router::new()
        .route("/api/v1/capabilities", get(capabilities))
        .route(
            "/api/v1/conversations",
            get(list_conversations).post(create_conversation),
        )
        .route(
            "/api/v1/conversations/{id}",
            get(get_conversation).patch(update_conversation),
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
        .route("/api/v1/blobs", post(upload_blob))
        .route("/api/v1/blobs/{sha256}", get(get_blob))
        .route(
            "/api/v1/huggingface/token",
            get(huggingface_token_status)
                .put(set_huggingface_token)
                .delete(clear_huggingface_token),
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
        .route("/api/v1/hardware", get(hardware))
        .route("/api/v1/toolchain", get(toolchain_status))
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
        .route("/api/v1/models/sdcpp/install", post(install_sdcpp_bundle))
        .route("/api/v1/generate/image", post(generate_image))
        .route("/api/v1/generate/video", post(generate_video))
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
            "/api/v1/agent/sessions/{id}/prompt",
            get(agent_system_prompt),
        )
        .route("/api/v1/agent/exec", post(agent_exec_tool))
        .route(
            "/api/v1/agent/approvals/{id}",
            get(get_agent_approval).post(decide_agent_approval),
        )
        .route("/api/v1/agent/artifacts/{id}", get(get_agent_artifact))
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
        .route(
            "/api/v1/runtimes/check-updates",
            post(check_runtime_updates),
        )
        .route("/api/v1/runtimes/build", post(build_runtime))
        .route("/api/v1/runtimes/build/cancel", post(cancel_build))
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
            "/api/v1/models/library-paths/suggestions",
            get(model_library_path_suggestions),
        )
        .route("/api/v1/models", axum::routing::delete(delete_local_model))
        .route(
            "/api/v1/models/bindings",
            get(model_bindings_list).put(update_model_binding),
        )
        .route("/api/v1/models/prepare", post(prepare_model))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/responses", post(responses))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_auth));

    Router::new()
        .route("/health", get(health))
        .merge(protected)
        .with_state(state)
        .layer(
            CorsLayer::new()
                .allow_origin([
                    HeaderValue::from_static("null"),
                    HeaderValue::from_static("http://localhost:5173"),
                    HeaderValue::from_static("http://127.0.0.1:5173"),
                ])
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
    let Some(expected) = &state.api_key else {
        return next.run(request).await;
    };
    let supplied = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    if supplied == Some(expected.as_str()) {
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
    let message = state
        .db
        .create_message(&id, request)
        .await
        .map_err(ApiError::bad_request)?;
    Ok((StatusCode::CREATED, Json(json!(message))))
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

async fn toolchain_status() -> Json<Value> {
    Json(toolchain_hints::toolchain_status())
}

async fn runtime_settings(State(state): State<AppState>) -> Json<Value> {
    Json(json!(state.runtime.settings().await))
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

#[derive(Debug, Deserialize)]
struct PrepareModelRequest {
    model_id: String,
}

async fn prepare_model(
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
    Json(request): Json<PrepareModelRequest>,
) -> ApiResult<Response> {
    if !query.stream {
        match state.runtime.prepare_model_stream(&request.model_id).await {
            Ok(mut rx) => {
                while let Some(item) = rx.recv().await {
                    match item {
                        Ok(StreamEvent::Load { .. }) => {}
                        Ok(_) => {}
                        Err(error) => return Err(ApiError::from_anyhow(error)),
                    }
                }
                state.invalidate_runtimes_cache().await;
                return Ok(
                    Json(json!({ "status": "ready", "model_id": request.model_id }))
                        .into_response(),
                );
            }
            Err(error) => return Err(ApiError::from_anyhow(error)),
        }
    }

    let mut event_rx = state
        .runtime
        .prepare_model_stream(&request.model_id)
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
        let done = json!({ "status": "ready", "model_id": model_id });
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

async fn managed_llama_status(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    use crate::runtime_settings::RuntimeTarget;

    // Local install state answers immediately; the upstream tag is filled in
    // from cache, with `latest_pending` telling the UI a check is still running.
    let cached = llama::cached_release_tag(&state.http);
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

async fn managed_whisper_status(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    use crate::runtime_settings::RuntimeTarget;

    let supported = whisper::managed_prebuilts_supported();
    let cached = if supported {
        whisper::cached_release_tag(&state.http)
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

async fn managed_sdcpp_status(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    use crate::runtime_settings::RuntimeTarget;

    let cached = sdcpp::cached_release_tag(&state.http);
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
    let job = sdcpp::GenerateImageRequest {
        prompt: request.prompt,
        model_id,
        negative_prompt: request.negative_prompt,
        width: request.width.unwrap_or(512),
        height: request.height.unwrap_or(512),
        steps: request.steps.unwrap_or(20),
        seed: request.seed,
        cfg_scale: request.cfg_scale,
        guidance: request.guidance,
        init_image,
    };
    let memory_plan = state.runtime.prepare_generation_memory(gen_bytes).await;
    let generated =
        sdcpp::generate_image(&state.data_dir, settings.sdcpp_binary.as_deref(), &job).await;
    state.runtime.restore_after_generation(memory_plan).await;
    let result = generated.map_err(|e| {
        if e.downcast_ref::<sdcpp::BusyError>().is_some() {
            ApiError::bad_request(e.to_string())
        } else {
            ApiError::internal(e)
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

async fn generate_video(
    State(state): State<AppState>,
    Json(request): Json<GenerateVideoApiRequest>,
) -> ApiResult<Json<Value>> {
    let settings = state.runtime.settings().await;
    let model_id = request
        .model_id
        .or(settings.default_video_gen_model.clone())
        .ok_or_else(|| ApiError::bad_request("model_id or default_video_gen_model is required"))?;
    let init_image = if let Some(blob) = &request.init_image_blob {
        Some(
            blob_store::blob_path(&state.data_dir, blob)
                .map_err(|e| ApiError::bad_request(e.to_string()))?,
        )
    } else {
        None
    };
    let gen_bytes = generation_model_bytes(&state.data_dir, &model_id);
    let job = sdcpp::GenerateVideoRequest {
        prompt: request.prompt,
        model_id,
        negative_prompt: request.negative_prompt,
        width: request.width.unwrap_or(512),
        height: request.height.unwrap_or(512),
        steps: request.steps.unwrap_or(20),
        seed: request.seed,
        cfg_scale: request.cfg_scale,
        guidance: request.guidance,
        init_image,
        video_frames: request.video_frames.unwrap_or(16),
        fps: request.fps,
    };
    let memory_plan = state.runtime.prepare_generation_memory(gen_bytes).await;
    let generated =
        sdcpp::generate_video(&state.data_dir, settings.sdcpp_binary.as_deref(), &job).await;
    state.runtime.restore_after_generation(memory_plan).await;
    let result = generated.map_err(|e| {
        if e.downcast_ref::<sdcpp::BusyError>().is_some() {
            ApiError::bad_request(e.to_string())
        } else {
            ApiError::internal(e)
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
    let voice_state = state.runtime.voice_state().await;
    let session = voice_state
        .sessions
        .create_session(
            &python,
            model_path.as_deref(),
            persona.clone(),
            voice_prompt.clone(),
            hf_token,
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
    let data_dir = state.data_dir.clone();
    let active_builds = state.active_builds.clone();
    let cache_state = state.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let result = builds::run_build_with_progress(
            &data_dir,
            request,
            &active_builds,
            Box::new(move |event| {
                push_progress(&progress_tx, event);
            }),
        )
        .await;
        match result {
            Ok(binary) => {
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
                push_progress(
                    &tx,
                    ProgressEvent::build_failed(
                        &serde_json::to_value(&report)
                            .unwrap_or_else(|_| json!({ "message": report.message })),
                    ),
                );
            }
        }
    });
    progress_sse(rx)
}

#[derive(Debug, Deserialize)]
struct CancelBuildRequest {
    build_id: String,
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
        let job_id = state
            .db
            .create_download_job(&request.repo_id, &request.filename, &request.revision)
            .await
            .ok()
            .map(|entry| entry.id);
        let cancel = job_id
            .as_ref()
            .map(|id| state.active_downloads.register(id));
        let job_handle = job_id.as_ref().map(|id| (state.db.clone(), id.clone()));
        let result = download::download_gguf_with_progress(
            &state.http,
            &state.data_dir,
            request,
            Box::new(|_| {}),
            job_handle,
            cancel.clone(),
        )
        .await;
        if let Some(id) = job_id.as_deref() {
            state.active_downloads.finish(id);
        }
        if let (Some(job_id), Err(error)) = (job_id.as_deref(), &result) {
            let message = error.to_string();
            if message.contains("cancelled") {
                let _ = state.db.cancel_download_job(job_id).await;
            } else {
                let _ = state.db.fail_download_job(job_id, &message).await;
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
    let repo_id = request.repo_id.clone();
    let filename = request.filename.clone();
    let revision = request.revision.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let job_id = db
            .create_download_job(&repo_id, &filename, &revision)
            .await
            .ok()
            .map(|entry| entry.id);
        let cancel = job_id.as_ref().map(|id| active_downloads.register(id));
        let job_handle = job_id.as_ref().map(|id| (db.clone(), id.clone()));
        let result = download::download_gguf_with_progress(
            &http,
            &data_dir,
            download::DownloadRequest {
                repo_id,
                filename,
                revision,
                engine: "llama.cpp".into(),
            },
            Box::new(move |event| {
                push_progress(&progress_tx, event);
            }),
            job_handle,
            cancel,
        )
        .await;
        if let Some(id) = job_id.as_deref() {
            active_downloads.finish(id);
        }
        if let (Some(job_id), Err(error)) = (job_id.as_deref(), &result) {
            let message = error.to_string();
            if message.contains("cancelled") {
                let _ = db.cancel_download_job(job_id).await;
            } else {
                let _ = db.fail_download_job(job_id, &message).await;
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
    let cache_state = state.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let result = download::download_mlx_snapshot_with_progress(
            &http,
            &data_dir,
            request,
            Box::new(move |event| {
                push_progress(&progress_tx, event);
            }),
            None,
            None,
        )
        .await;
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
    let signalled = state.active_downloads.cancel(&request.job_id);
    if signalled {
        let _ = state.db.cancel_download_job(&request.job_id).await;
        Ok(Json(json!({ "cancelled": request.job_id })))
    } else {
        let _ = state.db.cancel_download_job(&request.job_id).await;
        Ok(Json(
            json!({ "cancelled": request.job_id, "queued_only": true }),
        ))
    }
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
        .create_queued_download_job(
            &work.repo_id(),
            &work.filename(),
            &work.revision(),
            work.kind(),
            Some(&payload),
            Some(&work.label()),
            "pending",
        )
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
    let signalled = state
        .active_downloads
        .stop(&request.job_id, StopReason::Pause);
    // A job still waiting in line is paused directly; the queue skips it.
    if !signalled {
        state
            .db
            .pause_download_job(&request.job_id)
            .await
            .map_err(ApiError::bad_request)?;
    }
    Ok(Json(json!({ "paused": request.job_id })))
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
            return Ok(
                Json(json!({ "text": text.trim(), "engine": streaming_asr::ENGINE }))
                    .into_response(),
            );
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
    Ok(Json(json!({ "text": text, "engine": "whisper.cpp" })).into_response())
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
    let cache_state = state.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let result = download::download_streaming_asr_snapshot_with_progress(
            &http,
            &data_dir,
            request,
            Box::new(move |event| {
                push_progress(&progress_tx, event);
            }),
            None,
            None,
        )
        .await;
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
    let cache_state = state.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let result = download::download_personaplex_snapshot_with_progress(
            &http,
            &data_dir,
            request,
            Box::new(move |event| {
                push_progress(&progress_tx, event);
            }),
            None,
            None,
        )
        .await;
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
    let data: Vec<Value> = sdcpp_catalog::catalog(&state.data_dir)
        .into_iter()
        .map(|entry| bundle_json(&entry.bundle, entry.origin, &state.data_dir))
        .collect();
    Ok(Json(json!({ "data": data })))
}

fn bundle_json(
    bundle: &sdcpp_catalog::Bundle,
    origin: sdcpp_catalog::Origin,
    data_dir: &std::path::Path,
) -> Value {
    json!({
        "id": bundle.id,
        "label": bundle.label,
        "modality": bundle.modality,
        "key": bundle.key,
        "summary": bundle.summary,
        "license": bundle.license,
        "model_id": bundle.model_id(),
        "installed": bundle.installed(data_dir),
        "gated": bundle.gated(),
        "approx_bytes": bundle.approx_bytes(),
        "supports_init_image": bundle.supports_init_image,
        "origin": origin,
        "defaults": bundle.defaults,
        "components": bundle.components.iter().map(|component| json!({
            "repo_id": component.repo_id,
            "path": component.path,
            "flag": component.flag,
            "role": component.role,
            "gated": component.gated,
            "approx_bytes": component.approx_bytes,
        })).collect::<Vec<_>>(),
    })
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
    let cache_state = state.clone();
    tokio::spawn(async move {
        let progress_tx = tx.clone();
        let result = download::install_sdcpp_bundle_with_progress(
            &http,
            &data_dir,
            &bundle,
            Box::new(move |event| {
                push_progress(&progress_tx, event);
            }),
            None,
            None,
        )
        .await;
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
    Json(request): Json<ChatCompletionRequest>,
) -> ApiResult<Response> {
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4().simple());

    if !request.stream {
        let generation = state
            .runtime
            .generate(&request)
            .await
            .map_err(|error| ApiError::from_anyhow(error))?;
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
        .map_err(|error| ApiError::from_anyhow(error))?;
    let events = stream! {
        // Set once a terminal finish_reason has been sent, so the closing chunk
        // does not overwrite `tool_calls` with `stop`. Clients that map the last
        // finish_reason they see (including agent runtimes) would otherwise miss
        // the tool round trip.
        let mut finished = false;
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
                Ok(StreamEvent::End) => break,
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
        if !finished {
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
            })
            .collect(),
        other => vec![OpenAiMessage {
            role: "user".to_owned(),
            content: Value::String(text_from_content(other)),
            tool_calls: None,
            tool_call_id: None,
        }],
    }
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
        yield Ok::<Event, Infallible>(Event::default()
            .event("response.created")
            .data(json!({"type": "response.created", "response": {"id": response_id, "status": "in_progress"}}).to_string()));
        while let Some(item) = token_rx.recv().await {
            match item {
                Ok(StreamEvent::Load { .. }) => {}
                Ok(StreamEvent::Content(content)) => {
                    yield Ok(Event::default()
                        .event("response.output_text.delta")
                        .data(json!({"type": "response.output_text.delta", "delta": content}).to_string()));
                }
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
                Ok(StreamEvent::End) => break,
                Err(error) => {
                    tracing::error!(error = %error, "responses stream failed");
                    break;
                }
            }
        }
        yield Ok(Event::default()
            .event("response.completed")
            .data(json!({"type": "response.completed", "response": {"id": response_id, "status": "completed"}}).to_string()));
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
    Ok(Json(json!({
        "schema_version": 1,
        "sandbox": sandbox,
        "permission_modes": ["ask", "sandbox-only", "skip-permissions"],
        "runtimes": [{
            "id": "pi",
            "name": "Pi",
            // The daemon knows which adapter API it speaks, not which package
            // version the agent worker loaded; the worker reports that itself
            // when it comes up.
            "adapter_api_version": 1,
            "capabilities": {
                "streaming": true,
                "tool_calls": true,
                "compaction": true,
                "cancellation": true,
                "session_restore": true,
            }
        }],
        "tool_output_limit_chars": 24_000,
    })))
}

async fn agent_tool_catalog() -> Json<Value> {
    Json(crate::agent_tools::definitions())
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
    Json(request): Json<crate::agent_types::CreateAgentSession>,
) -> ApiResult<Json<crate::agent_types::AgentSessionRecord>> {
    if let Some(workspace) = &request.workspace_path {
        validate_workspace_path(&state, workspace)?;
    }
    let session = state
        .db
        .create_agent_session(request)
        .await
        .map_err(ApiError::internal)?;
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
    Json(update): Json<crate::agent_types::UpdateAgentSession>,
) -> ApiResult<Json<crate::agent_types::AgentSessionRecord>> {
    if let Some(Some(workspace)) = &update.workspace_path {
        validate_workspace_path(&state, workspace)?;
    }
    let session = state
        .db
        .update_agent_session(&id, update)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(session))
}

async fn delete_agent_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    state.agent_broker.terminate_session_processes(&id).await;
    state
        .db
        .delete_agent_session(&id)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "deleted": true })))
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

/// System prompt for a session, built by the application from the live sandbox
/// state and permission mode.
async fn agent_system_prompt(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let session = state
        .db
        .agent_session(&id)
        .await
        .map_err(|error| ApiError::not_found(error.to_string()))?;
    let names = session
        .enabled_tools
        .clone()
        .unwrap_or_else(crate::agent_tools::tool_names);
    let prompt =
        crate::agent_tools::system_prompt(&session, &state.agent_broker.capabilities(), &names);
    Ok(Json(json!({ "system_prompt": prompt, "tools": names })))
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
    if let Some(enabled) = &session.enabled_tools {
        if !enabled.iter().any(|name| name == &request.tool) {
            return Err(ApiError::bad_request(format!(
                "tool `{}` is not enabled for this session",
                request.tool
            )));
        }
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

    // An approved call records itself when it runs. A refused one never runs, so
    // record it here — otherwise the attempt would vanish from the activity
    // timeline as soon as the session is reloaded.
    if !approved {
        let note = approval.note.clone();
        state
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
            .map_err(ApiError::internal)?;
    }

    state.agent_broker.notify_approvals();
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
    let git = resolved.join(".git").exists();
    Ok(Json(json!({
        "path": resolved.display().to_string(),
        "git_repository": git,
        "sandbox": state.agent_broker.capabilities(),
    })))
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
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        if resolved == home {
            return Err(ApiError::bad_request(
                "choose a project folder rather than the whole home directory",
            ));
        }
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
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    #[test]
    fn converts_responses_string_input() {
        let messages = responses_input_to_messages(&json!("hello"));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
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
        );
        AppState {
            db,
            runtime,
            api_key: None,
            http,
            data_dir: data_dir.to_path_buf(),
            active_builds: Arc::new(builds::ActiveBuilds::new()),
            active_downloads,
            download_queue,
            runtimes_cache: Arc::new(Mutex::new(None)),
            agent_broker: Arc::new(crate::agent_exec::AgentBroker::new()),
        }
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
        assert_eq!(held["approval"]["summary"].as_str().is_some(), true);

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
        } else {
            assert_eq!(sandbox["backend"], "none");
            assert!(sandbox["detail"].as_str().unwrap().len() > 10);
        }
        assert!(
            capabilities["permission_modes"]
                .as_array()
                .unwrap()
                .contains(&json!("skip-permissions"))
        );
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

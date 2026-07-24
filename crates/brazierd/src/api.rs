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
    db::CreateRunSnapshot,
    db::ConversationExport,
    download::{self},
    engine::{Engine, StreamEvent},
    fork_hints::{self, ModelLoadError, RuntimeForkHint},
    hf::{self, SearchQuery},
    hf_auth, media, model_bindings, models_store,
    llama,
    mcp,
    progress::ProgressEvent,
    runtimes, streaming_asr, tool_registry, toolchain_hints, whisper,
    types::{
        ChatCompletionRequest, CreateConversation, CreateMessage, OpenAiMessage, ResponsesRequest,
        text_from_content,
    },
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
            "/api/v1/conversations/{id}/messages",
            get(list_messages).post(create_message),
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
        .route("/api/v1/tools", get(list_tools))
        .route("/api/v1/mcp/servers", get(list_mcp_servers).post(create_mcp_server))
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
        .route("/api/v1/runtimes/build", post(build_runtime))
        .route("/api/v1/runtimes/build/cancel", post(cancel_build))
        .route("/api/v1/models/download", post(download_model))
        .route("/api/v1/models/download/mlx", post(download_mlx_model))
        .route(
            "/api/v1/models/download/streaming-asr",
            post(download_streaming_asr_model),
        )
        .route("/api/v1/models/download/queue", post(queue_model_download))
        .route("/api/v1/models/download/cancel", post(cancel_model_download))
        .route("/api/v1/models/downloads", get(list_download_jobs))
        .route(
            "/api/v1/models/library-paths/suggestions",
            get(model_library_path_suggestions),
        )
        .route("/api/v1/models", axum::routing::delete(delete_local_model))
        .route("/api/v1/models/bindings", get(model_bindings_list).put(update_model_binding))
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
                    "engine": "whisper.cpp",
                    "summary": "File/blob transcription via whisper.cpp before chat. Works with any text model."
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
                    "available": false,
                    "planned": true,
                    "summary": "Full-duplex speech-to-speech (PersonaPlex / Nemotron VoiceChat class). Not shipped yet."
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

async fn model_fork_hints(
    State(state): State<AppState>,
    Path((repo_owner, repo_name)): Path<(String, String)>,
) -> ApiResult<Json<Value>> {
    let repo_id = format!("{repo_owner}/{repo_name}");
    let fork_hints = fork_hints::hints_for_repo(&state.http, &state.data_dir, &repo_id)
        .await
        .map_err(ApiError::bad_request)?;
    Ok(Json(json!({ "repo_id": repo_id, "fork_hints": fork_hints })))
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
            "whisper.cpp" => &active.whisper,
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
    let path = match entry.engine.as_str() {
        "mlx-lm" => state
            .runtime
            .activate_python(
                crate::mlx::MlxKind::Lm,
                std::path::PathBuf::from(&entry.path),
            )
            .await
            .map_err(ApiError::bad_request)?,
        "mlx-vlm" => state
            .runtime
            .activate_python(
                crate::mlx::MlxKind::Vlm,
                std::path::PathBuf::from(&entry.path),
            )
            .await
            .map_err(ApiError::bad_request)?,
        "whisper.cpp" => state
            .runtime
            .activate_whisper(std::path::PathBuf::from(&entry.path))
            .await
            .map_err(ApiError::bad_request)?,
        "streaming-asr" => state
            .runtime
            .activate_streaming_asr(std::path::PathBuf::from(&entry.path))
            .await
            .map_err(ApiError::bad_request)?,
        _ => state
            .runtime
            .activate_binary(std::path::PathBuf::from(&entry.path))
            .await
            .map_err(ApiError::bad_request)?,
    };
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
    let _path = models_store::path_for_model_id(
        &state.data_dir,
        &request.model_id,
        &extra_paths,
    )
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
        match state
            .runtime
            .prepare_model_stream(&request.model_id)
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
                return Ok(Json(json!({ "status": "ready", "model_id": request.model_id })).into_response());
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

    let latest_tag = llama::resolve_managed_release(&state.http, RuntimeTarget::Cpu)
        .await
        .ok()
        .map(|(tag, _)| tag);

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
            .ensure_llama_binary_with_progress(target, force, Box::new(move |event| {
                push_progress(&progress_tx, event);
            }))
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
                        &serde_json::to_value(&report).unwrap_or_else(|_| {
                            json!({ "message": report.message })
                        }),
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
        let job_handle = job_id
            .as_ref()
            .map(|id| (state.db.clone(), id.clone()));
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
        let cancel = job_id
            .as_ref()
            .map(|id| active_downloads.register(id));
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
        Ok(Json(json!({ "cancelled": request.job_id, "queued_only": true })))
    }
}

async fn queue_model_download(
    State(state): State<AppState>,
    Json(request): Json<download::DownloadRequest>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    let job = state
        .db
        .create_download_job(&request.repo_id, &request.filename, &request.revision)
        .await
        .map_err(ApiError::bad_request)?;
    state
        .download_queue
        .enqueue(crate::download_queue::QueuedDownload {
            job_id: job.id.clone(),
            request,
        })
        .await
        .map_err(ApiError::internal)?;
    Ok((StatusCode::ACCEPTED, Json(json!({ "job_id": job.id, "status": job.status }))))
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
    let stored = blob_store::store_bytes(
        &state.data_dir,
        &bytes,
        &mime,
        Some("transcription-input"),
    )
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
        let python = streaming_asr::resolve_python(
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
        let mut events = streaming_asr::transcribe_stream(streaming_asr::StreamTranscribeRequest {
            python: &python,
            model: &model_path,
            audio: &wav,
            lookahead: None,
        })
        .await
        .map_err(ApiError::bad_request)?;
        if !request.stream {
            let mut text = String::new();
            while let Some(item) = events.recv().await {
                match item.map_err(ApiError::bad_request)? {
                    streaming_asr::WorkerEvent::Delta { text: delta } => text.push_str(&delta),
                    streaming_asr::WorkerEvent::Done { text: final_text } => {
                        text = final_text;
                    }
                    streaming_asr::WorkerEvent::Error { message } => {
                        return Err(ApiError::bad_request(message));
                    }
                    streaming_asr::WorkerEvent::Status { .. } => {}
                }
            }
            let _ = tokio::fs::remove_file(&wav).await;
            return Ok(Json(json!({ "text": text.trim(), "engine": streaming_asr::ENGINE }))
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
    let model_path = whisper::resolve_model_path(
        &state.data_dir,
        request
            .model
            .as_deref()
            .or(settings.whisper_model.as_deref()),
    )
    .ok_or_else(|| ApiError::bad_request(media::asr_missing_message().to_owned()))?;

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
        whisper_model: Some(model_path),
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
                    yield Ok(Event::default().data(chunk.to_string()));
                    break;
                }
            }
        }
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
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tempfile::tempdir;
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
        }
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
            models_store::path_for_model_id(dir.path(), "gguf:acme/demo/weights.gguf", &[]).unwrap(),
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

use std::{convert::Infallible, time::Duration};

use async_stream::stream;
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
use serde_json::{Value, json};
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use uuid::Uuid;

use crate::{
    AppState,
    build_recipe::{self, BuildPlanRequest},
    hf::{self, SearchQuery},
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
}

impl ApiError {
    fn bad_request(error: impl ToString) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: error.to_string(),
        }
    }

    fn internal(error: impl ToString) -> Self {
        tracing::error!(error = %error.to_string(), "request failed");
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "The request could not be completed.".to_owned(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({ "error": { "message": self.message } })),
        )
            .into_response()
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
        .route("/api/v1/huggingface/models", get(search_hugging_face))
        .route("/api/v1/engines/build-plan", post(build_plan))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
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
    Ok(Json(json!({
        "status": "healthy",
        "engine": state.engine.id(),
        "version": env!("CARGO_PKG_VERSION")
    })))
}

async fn capabilities(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let models = state.engine.models().await.map_err(ApiError::internal)?;
    Ok(Json(json!({
        "schema_version": 1,
        "models": models,
        "features": {
            "conversation_branches": true,
            "hugging_face_search": true,
            "openai_chat_completions": true,
            "openai_responses": true
        }
    })))
}

async fn list_conversations(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let conversations = state
        .db
        .list_conversations()
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "data": conversations })))
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

async fn search_hugging_face(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> ApiResult<Json<Value>> {
    let models = hf::search(&state.http, query)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "data": models })))
}

async fn build_plan(Json(request): Json<BuildPlanRequest>) -> ApiResult<Json<Value>> {
    let plan = build_recipe::plan(request).map_err(ApiError::bad_request)?;
    Ok(Json(json!(plan)))
}

async fn list_models(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let models = state.engine.models().await.map_err(ApiError::internal)?;
    let data = models
        .into_iter()
        .map(|model| {
            json!({
                "id": model.id,
                "object": "model",
                "owned_by": format!("brazier:{}", model.engine),
                "capabilities": model.capabilities
            })
        })
        .collect::<Vec<_>>();
    Ok(Json(json!({ "object": "list", "data": data })))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> ApiResult<Response> {
    let generation = state
        .engine
        .generate(&request)
        .await
        .map_err(ApiError::internal)?;
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4().simple());

    if !request.stream {
        return Ok(Json(json!({
            "id": completion_id,
            "object": "chat.completion",
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generation.text,
                    "reasoning_content": generation.reasoning
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }))
        .into_response());
    }

    let model = request.model;
    let reasoning = generation.reasoning;
    let words = generation
        .text
        .split_inclusive(char::is_whitespace)
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let events = stream! {
        if let Some(reasoning) = reasoning {
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
        for word in words {
            let chunk = json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": { "content": word },
                    "finish_reason": null
                }]
            });
            yield Ok::<Event, Infallible>(Event::default().data(chunk.to_string()));
            tokio::time::sleep(Duration::from_millis(18)).await;
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
    };
    let generation = state
        .engine
        .generate(&chat_request)
        .await
        .map_err(ApiError::internal)?;
    let response_id = format!("resp_{}", Uuid::new_v4().simple());
    if !request.stream {
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

    let words = generation
        .text
        .split_inclusive(char::is_whitespace)
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let events = stream! {
        yield Ok::<Event, Infallible>(Event::default()
            .event("response.created")
            .data(json!({"type": "response.created", "response": {"id": response_id, "status": "in_progress"}}).to_string()));
        for word in words {
            yield Ok(Event::default()
                .event("response.output_text.delta")
                .data(json!({"type": "response.output_text.delta", "delta": word}).to_string()));
            tokio::time::sleep(Duration::from_millis(18)).await;
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

    #[test]
    fn converts_responses_string_input() {
        let messages = responses_input_to_messages(&json!("hello"));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }
}

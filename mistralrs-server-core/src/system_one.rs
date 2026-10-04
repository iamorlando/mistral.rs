use axum::{
    extract::{rejection::JsonRejection, State},
    response::{IntoResponse, Response},
    Json,
};
use mistralrs_core::{
    decision::{DecisionRequest, DecisionResponse, DecisionValidationError},
    DecisionInferenceRequest, MistralRs, Request,
};

use crate::{
    handler_core::{openai_error_from_error, ApiError, ApiErrorKind},
    types::ExtractedMistralRsState,
    util::validate_model_name,
};

#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
pub struct DecisionModelInfo {
    pub name: String,
    pub description: String,
    pub release_date: String,
}

#[utoipa::path(
    post,
    tag = "System One",
    path = "/v1/systemone",
    request_body = DecisionRequest,
    responses(
        (status = 200, description = "Typed decisions", body = DecisionResponse),
        (status = 400, description = "Invalid question, input, or model capability"),
        (status = 404, description = "Model not found"),
        (status = 500, description = "Inference failed")
    )
)]
pub async fn system_one(
    State(state): ExtractedMistralRsState,
    payload: Result<Json<DecisionRequest>, JsonRejection>,
) -> Response {
    let request = match payload {
        Ok(Json(request)) => request,
        Err(error) => {
            return openai_error_from_error(
                &ApiError::from_json_rejection(error),
                ApiErrorKind::InvalidRequest,
            )
        }
    };
    if let Err(error) = request.validate().and_then(|()| {
        validate_model_name(&request.model, state.clone()).map_err(anyhow::Error::from)
    }) {
        return openai_error_from_error(error.as_ref(), ApiErrorKind::InvalidRequest);
    }
    MistralRs::maybe_log_request(
        state.clone(),
        serde_json::to_string(&request).expect("decision request serializes"),
    );
    let (response, mut receiver) = tokio::sync::mpsc::channel(1);
    if let Err(error) = state
        .send_request_async(Request::Decision(Box::new(DecisionInferenceRequest {
            input: request,
            response,
        })))
        .await
    {
        return openai_error_from_error(&error, ApiErrorKind::Internal);
    }
    match receiver.recv().await {
        Some(Ok(response)) => {
            MistralRs::maybe_log_response(state, &response);
            Json(response).into_response()
        }
        Some(Err(error)) => {
            MistralRs::maybe_log_error(state, error.as_ref());
            let kind = if error.is::<DecisionValidationError>() {
                ApiErrorKind::InvalidRequest
            } else {
                ApiErrorKind::Internal
            };
            openai_error_from_error(error.as_ref(), kind)
        }
        None => openai_error_from_error(&ApiError::internal(), ApiErrorKind::Internal),
    }
}

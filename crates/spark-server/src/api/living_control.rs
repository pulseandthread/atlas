// SPDX-License-Identifier: AGPL-3.0-only

//! Living-weights steering control: `GET/POST /v1/living/steer`.
//!
//! `POST {"scale": d}` sets the dose of the middle-layer steering vector
//! (`spark_model::living::steer`): `d · rms · û` (û: unit RMS) is added to the residual
//! stream after the steering layer, `rms` being that layer's measured mean
//! residual RMS. `0` switches it off. Optional `"flush_prefix_cache": true`
//! drops the cached prefixes so a repeated prompt is recomputed under the new
//! dose (otherwise cached tokens keep the dose they were computed with).
//!
//! Like `POST /v1/lora/active`, the request travels the scheduler's rotation
//! channel and is applied at a QUIESCENT point (no in-flight step), on every
//! rank. `GET` returns the current state without touching the scheduler.

use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::Deserialize;

use crate::api::compact::openai_error_response;
use crate::main_modules::model_host::CurrentModel;

/// Largest |dose| accepted; the paper's useful range is a few units.
const MAX_DOSE: f32 = 64.0;

#[derive(Deserialize)]
pub struct SetSteerRequest {
    /// The dose `d`, in units of the residual RMS at the steering layer.
    pub scale: f32,
    #[serde(default)]
    pub flush_prefix_cache: bool,
}

/// GET /v1/living/steer
pub async fn get_living_steer() -> Response {
    Json(spark_model::living::steer::status()).into_response()
}

/// POST /v1/living/steer  `{"scale": d}`
pub async fn set_living_steer(
    CurrentModel(state): CurrentModel,
    body: Result<Json<SetSteerRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => {
            return openai_error_response(StatusCode::BAD_REQUEST, format!("invalid body: {e}"));
        }
    };
    if !req.scale.is_finite() || req.scale.abs() > MAX_DOSE {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("scale must be finite with |scale| <= {MAX_DOSE}"),
        );
    }
    if !spark_model::living::steer::loaded() {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "no steering vectors are loaded (start with ATLAS_STEER_DIR)".to_string(),
        );
    }
    let Some(ref tx) = state.rotation_tx else {
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler control channel is not armed".to_string(),
        );
    };
    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
    let cmd = crate::scheduler::LoraCommand::LivingSteer {
        dose: req.scale,
        flush_prefix_cache: req.flush_prefix_cache,
    };
    if tx.send((cmd, ack_tx)).await.is_err() {
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler control channel closed".to_string(),
        );
    }
    match ack_rx.await {
        Ok(Ok(_)) => Json(spark_model::living::steer::status()).into_response(),
        Ok(Err(reason)) => openai_error_response(StatusCode::BAD_REQUEST, reason),
        Err(_) => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler dropped the steer ack (shutting down?)".to_string(),
        ),
    }
}

# src/api/routes/predict.py
"""Credit risk prediction endpoint — multi-dataset support.

Accepts a dataset_id parameter to select which model to use.
Dynamically validates the input against the dataset's schema.
Falls back to the default dataset when no ID is specified.
"""

import time
from typing import Literal, cast

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError

from src.api.auth import verify_api_key
from src.api.dependencies import (
    get_drift_detector,
    get_model_artifacts,
    get_shap_engine,
    resolve_dataset_id,
)
from src.api.dynamic_schema import build_request_validator
from src.api.schemas import (
    PredictionResponse,
    ShapFactorResponse,
    XAIInterpretation,
)
from src.data.preprocessing import preprocess_input, preprocess_with_pipeline
from src.data.schema import load_dataset_schema
from src.monitoring.metrics import (
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    PREDICTION_PROBABILITY,
    SHAP_COMPUTATION_SECONDS,
)
from src.monitoring.tracing import get_tracer

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["prediction"])


@router.post("/predict_risk/", response_model=PredictionResponse)
async def predict_risk(
    request: dict,
    dataset_id: str | None = Query(
        default=None,
        description="Dataset/model to use. Defaults to the first loaded model.",
    ),
    _api_key: str | None = Depends(verify_api_key),
) -> PredictionResponse:
    """Predict credit risk using the specified dataset's model.

    Accepts raw JSON input, validates it against the dataset's schema,
    preprocesses, runs inference, and returns SHAP explanation.

    Query params:
        dataset_id: Which model to use (e.g., 'german_credit', 'lending_club').
    """
    # Resolve dataset
    ds_id = resolve_dataset_id(dataset_id)
    if ds_id is None:
        raise HTTPException(status_code=503, detail="No models loaded.")

    artifacts = get_model_artifacts(ds_id)
    shap_engine = get_shap_engine(ds_id)

    if artifacts is None or shap_engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model for dataset '{ds_id}' is not loaded. Train it first.",
        )

    tracer = get_tracer()

    try:
        # 1. Validate input against dataset schema
        try:
            schema = load_dataset_schema(ds_id)
            validator_cls = build_request_validator(schema)
            validated = validator_cls(**request)
            input_dict = validated.model_dump()
        except FileNotFoundError:
            # No schema file — accept raw dict (legacy mode)
            input_dict = request
        except ValidationError as ve:
            raise HTTPException(status_code=422, detail=ve.errors()) from None

        # 2. Preprocess
        if artifacts.pipeline is not None:
            X_processed = preprocess_with_pipeline(
                input_dict=input_dict,
                pipeline=artifacts.pipeline,
                feature_names=artifacts.feature_names,
            )
        else:
            # Legacy: use get_dummies fallback
            X_processed = preprocess_input(
                input_dict=input_dict,
                feature_names=artifacts.feature_names,
            )

        # 3. Inference
        _ctx = tracer.start_as_current_span("inference") if tracer else None
        if _ctx:
            with _ctx as span:
                proba = float(artifacts.model.predict_proba(X_processed)[:, 1][0])
                span.set_attribute("prediction.probability", proba)
                span.set_attribute("dataset_id", ds_id)
        else:
            proba = float(artifacts.model.predict_proba(X_processed)[:, 1][0])

        prediction_threshold = artifacts.decision_threshold
        prediction_label = (
            "High Risk (Default)" if proba >= prediction_threshold else "Low Risk (No Default)"
        )

        # 4. Metrics
        PREDICTION_PROBABILITY.observe(proba)
        PREDICTION_COUNT.labels(risk_label=prediction_label).inc()

        logger.info(
            "prediction_made",
            dataset_id=ds_id,
            probability=round(proba, 4),
            decision_threshold=round(prediction_threshold, 4),
            label=prediction_label,
        )

        # 5. SHAP explanation
        shap_start = time.perf_counter()
        explanation = shap_engine.explain(X_processed)
        SHAP_COMPUTATION_SECONDS.observe(time.perf_counter() - shap_start)

        # 6. Drift recording
        drift_detector = get_drift_detector(ds_id)
        if drift_detector is not None:
            drift_detector.record(
                input_vector=X_processed.values[0],
                predicted_proba=proba,
            )

        # 7. Build response
        factors = [
            ShapFactorResponse(
                factor=f.feature,
                risk_impact=cast("Literal['increases', 'reduces']", f.impact),
                shap_magnitude=f.shap_value,
                input_value=f.input_value,
            )
            for f in explanation.factors
        ]

        return PredictionResponse(
            prediction=prediction_label,
            probability_of_risk=round(proba, 4),
            xai_interpretation=XAIInterpretation(
                base_risk_score=explanation.base_value,
                detailed_explanation=factors,
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        PREDICTION_ERRORS.labels(error_type=type(exc).__name__).inc()
        logger.exception("prediction_failed", dataset_id=ds_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail="Internal error during prediction. Check server logs.",
        ) from exc

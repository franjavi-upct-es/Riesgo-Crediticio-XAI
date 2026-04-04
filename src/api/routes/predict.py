# src/api/routes/predict.py
"""Credit risk prediction endpoint.

Receives applicant data, preprocesses it, runs inference, computes SHAP
explanation, records inputs for drift detection, and instruments
everything with Prometheus metrics and OpenTelemetry spans.
"""

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import verify_api_key
from src.api.dependencies import (
    get_drift_detector,
    get_model_artifacts,
    get_shap_engine,
)
from src.api.schemas import (
    CreditDataRequest,
    PredictionResponse,
    ShapFactorResponse,
    XAIInterpretation,
)
from src.data.preprocessing import preprocess_input
from src.explain.shap_engine import ShapEngine
from src.model.registry import ModelArtifacts
from src.monitoring.drift import DriftDetector
from src.monitoring.metrics import (
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    PREDICTION_PROBABILITY,
    SHAP_COMPUTATION_SECONDS,
)
from src.monitoring.tracing import get_tracer

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["prediction"])

_RISK_THRESHOLD = 0.5


@router.post("/predict_risk/", response_model=PredictionResponse)
def predict_risk(
    data: CreditDataRequest,
    _api_key: str | None = Depends(verify_api_key),
    artifacts: ModelArtifacts | None = Depends(get_model_artifacts),
    shap_engine: ShapEngine | None = Depends(get_shap_engine),
    drift_detector: DriftDetector | None = Depends(get_drift_detector),
) -> PredictionResponse:
    """Predict credit risk and return a SHAP explanation."""
    if artifacts is None or shap_engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifacts are not loaded. Run the training pipeline first: credit-risk-train"
            ),
        )

    tracer = get_tracer()

    try:
        # 1. Preprocess
        _span_ctx = tracer.start_as_current_span("preprocess") if tracer else None
        if _span_ctx:
            with _span_ctx:
                X_processed = preprocess_input(
                    input_dict=data.model_dump(),
                    feature_names=artifacts.feature_names,
                )
        else:
            X_processed = preprocess_input(
                input_dict=data.model_dump(),
                feature_names=artifacts.feature_names,
            )

        # 2. Inference
        _span_ctx = tracer.start_as_current_span("inference") if tracer else None
        if _span_ctx:
            with _span_ctx as span:
                proba = float(artifacts.model.predict_proba(X_processed)[:, 1][0])
                span.set_attribute("prediction.probability", proba)
        else:
            proba = float(artifacts.model.predict_proba(X_processed)[:, 1][0])

        prediction_label = (
            "High Risk (Default)" if proba > _RISK_THRESHOLD else "Low Risk (No Default)"
        )

        # 3. Metrics
        PREDICTION_PROBABILITY.observe(proba)
        PREDICTION_COUNT.labels(risk_label=prediction_label).inc()

        logger.info(
            "prediction_made",
            probability=round(proba, 4),
            label=prediction_label,
        )

        # 4. SHAP explanation (timed)
        shap_start = time.perf_counter()

        _span_ctx = tracer.start_as_current_span("shap_explanation") if tracer else None
        if _span_ctx:
            with _span_ctx:
                explanation = shap_engine.explain(X_processed)
        else:
            explanation = shap_engine.explain(X_processed)

        shap_duration = time.perf_counter() - shap_start
        SHAP_COMPUTATION_SECONDS.observe(shap_duration)

        # 5. Record for drift detection
        if drift_detector is not None:
            drift_detector.record(
                input_vector=X_processed.values[0],
                predicted_proba=proba,
            )

        # 6. Build response
        factors = [
            ShapFactorResponse(
                factor=f.feature,
                risk_impact=f.impact,  # type: ignore[arg-type]
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
        logger.exception("prediction_failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail="Internal error during prediction. Check server logs.",
        ) from exc

# src/api/routes/evaluation.py
"""Evaluation data endpoints for the dashboard.

Serves pre-computed evaluation metrics (AUC, F1, confusion matrix,
ROC curve, SHAP importance) from the JSON artifact generated during
training. This avoids re-running expensive inference on every request.
"""

import json
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException

from src.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# Cache the evaluation data in memory after first load.
_cached_evaluation: dict[str, Any] | None = None


def _load_evaluation() -> dict[str, Any]:
    """Load evaluation metrics from the JSON artifact, with caching."""
    global _cached_evaluation
    if _cached_evaluation is not None:
        return _cached_evaluation

    path = settings.data.evaluation_metrics_path
    if not path.exists():
        raise FileNotFoundError(f"Evaluation metrics not found at {path}")

    with open(path) as f:
        _cached_evaluation = json.load(f)

    logger.info("evaluation_metrics_loaded", path=str(path))
    if _cached_evaluation is None:
        raise FileNotFoundError(f"Evaluation metrics file at {path} was empty")
    return _cached_evaluation


def clear_evaluation_cache() -> None:
    """Clear the cached evaluation data (e.g., after retraining)."""
    global _cached_evaluation
    _cached_evaluation = None


@router.get("/metrics")
def get_evaluation_metrics() -> dict[str, Any]:
    """Return classification metrics (AUC, F1, precision, recall).

    Returns:
        Dictionary with 'metrics' and 'dataset_info' keys.
    """
    try:
        data = _load_evaluation()
        return {
            "metrics": data["metrics"],
            "dataset_info": data["dataset_info"],
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Evaluation metrics not available. Run the training pipeline first.",
        ) from None


@router.get("/confusion_matrix")
def get_confusion_matrix() -> dict[str, Any]:
    """Return the confusion matrix with labels."""
    try:
        data = _load_evaluation()
        return data["confusion_matrix"]
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Evaluation data not available.") from None


@router.get("/roc_curve")
def get_roc_curve() -> dict[str, Any]:
    """Return ROC curve data points and AUC score."""
    try:
        data = _load_evaluation()
        return {
            "roc_curve": data["roc_curve"],
            "auc": data["metrics"]["auc"],
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Evaluation data not available.") from None


@router.get("/shap_importance")
def get_shap_importance() -> dict[str, Any]:
    """Return top-20 features by mean absolute SHAP value."""
    try:
        data = _load_evaluation()
        return {"shap_importance": data["shap_importance"]}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Evaluation data not available.") from None


@router.get("/prediction_distribution")
def get_prediction_distribution() -> dict[str, Any]:
    """Return histogram of predicted risk probabilities."""
    try:
        data = _load_evaluation()
        return {"prediction_distribution": data["prediction_distribution"]}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Evaluation data not available.") from None


@router.get("/full")
def get_full_evaluation() -> dict[str, Any]:
    """Return the complete evaluation artifact (all metrics and charts)."""
    try:
        return _load_evaluation()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Evaluation metrics not available. Run the training pipeline first.",
        ) from None

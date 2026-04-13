# src/api/routes/evaluation.py
"""Multi-dataset evaluation endpoints for the dashboard.

Serves pre-computed evaluation metrics from per-dataset JSON artifacts
generated during training. Accepts an optional dataset_id query
parameter; falls back to the default loaded dataset.
"""

import json
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query

from src.api.dependencies import resolve_dataset_id
from src.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# Cache: keyed by dataset_id
_cached_evaluations: dict[str, dict[str, Any]] = {}


def _get_eval_path(dataset_id: str) -> Path:
    """Resolve the evaluation JSON path for a dataset."""
    # Multi-dataset layout first
    multi = settings.data.dir / dataset_id / "evaluation_metrics.json"
    if multi.exists():
        return multi
    # Legacy layout fallback
    legacy = settings.data.evaluation_metrics_path
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"Evaluation metrics not found for '{dataset_id}'")


def _load_evaluation(dataset_id: str) -> dict[str, Any]:
    """Load evaluation metrics with per-dataset caching."""
    if dataset_id in _cached_evaluations:
        return _cached_evaluations[dataset_id]

    path = _get_eval_path(dataset_id)

    with open(path) as f:
        data = json.load(f)

    _cached_evaluations[dataset_id] = data
    logger.info("evaluation_metrics_loaded", dataset_id=dataset_id, path=str(path))
    return dict(data)


def clear_evaluation_cache(dataset_id: str | None = None) -> None:
    """Clear cached evaluation data."""
    if dataset_id:
        _cached_evaluations.pop(dataset_id, None)
    else:
        _cached_evaluations.clear()


def _resolve_dataset(dataset_id: str | None) -> str:
    """Resolve dataset ID with fallback to default."""
    ds_id = resolve_dataset_id(dataset_id)
    if ds_id is None:
        raise HTTPException(status_code=503, detail="No models loaded.")
    return ds_id


@router.get("/metrics")
def get_evaluation_metrics(
    dataset_id: str | None = Query(default=None, description="Dataset ID"),
) -> dict[str, Any]:
    """Return classification metrics (AUC, F1, precision, recall)."""
    ds = _resolve_dataset(dataset_id)
    try:
        data = _load_evaluation(ds)
        return {
            "metrics": data["metrics"],
            "dataset_info": data["dataset_info"],
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=503, detail=f"Evaluation not available for '{ds}'."
        ) from None


@router.get("/confusion_matrix")
def get_confusion_matrix(
    dataset_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return the confusion matrix with labels."""
    ds = _resolve_dataset(dataset_id)
    try:
        return dict(_load_evaluation(ds)["confusion_matrix"])
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Evaluation data not available.") from None


@router.get("/roc_curve")
def get_roc_curve(
    dataset_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return ROC curve data points and AUC score."""
    ds = _resolve_dataset(dataset_id)
    try:
        data = _load_evaluation(ds)
        return {"roc_curve": data["roc_curve"], "auc": data["metrics"]["auc"]}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Evaluation data not available.") from None


@router.get("/shap_importance")
def get_shap_importance(
    dataset_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return top-20 features by mean absolute SHAP value."""
    ds = _resolve_dataset(dataset_id)
    try:
        return {"shap_importance": _load_evaluation(ds)["shap_importance"]}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Evaluation data not available.") from None


@router.get("/prediction_distribution")
def get_prediction_distribution(
    dataset_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return histogram of predicted risk probabilities."""
    ds = _resolve_dataset(dataset_id)
    try:
        return {"prediction_distribution": _load_evaluation(ds)["prediction_distribution"]}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Evaluation data not available.") from None


@router.get("/full")
def get_full_evaluation(
    dataset_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return the complete evaluation artifact."""
    ds = _resolve_dataset(dataset_id)
    try:
        return _load_evaluation(ds)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"Evaluation metrics not available for '{ds}'. Run training first.",
        ) from None

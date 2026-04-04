# src/api/dependencies.py
"""FastAPI dependency injection providers.

Manages the lifecycle of shared resources (model, explainer, drift
detector) that are loaded once at startup and injected into route
handlers via Depends(). This decouples route logic from resource
management and makes testing straightforward (override the dependency).
"""

from functools import lru_cache

import pandas as pd
import structlog

from src.config import settings
from src.explain.shap_engine import ShapEngine
from src.model.registry import ModelArtifacts, load_model_artifacts
from src.monitoring.drift import DriftDetector

logger = structlog.get_logger(__name__)

# Module-level state — populated by lifespan, accessed by dependencies
_artifacts: ModelArtifacts | None = None
_shap_engine: ShapEngine | None = None
_drift_detector: DriftDetector | None = None


def initialize_resources() -> None:
    """Load model artifacts, SHAP engine, and drift detector.

    Called once during application startup (lifespan context).
    """
    global _artifacts, _shap_engine, _drift_detector

    try:
        _artifacts = load_model_artifacts()
        _shap_engine = ShapEngine(_artifacts.model)
        logger.info(
            "api_resources_initialized",
            n_features=len(_artifacts.feature_names),
        )
    except FileNotFoundError:
        logger.warning(
            "model_artifacts_not_found",
            detail="API will start but prediction endpoints will return 503.",
        )
        return

    # Initialize drift detector with reference data from synthetic test set
    try:
        ref_path = settings.data.synthetic_test_path
        if ref_path.exists():
            ref_df = pd.read_csv(ref_path)
            target_col = "risk_flag"
            X_ref = ref_df.drop(columns=[target_col]) if target_col in ref_df.columns else ref_df

            # Align reference columns to model's feature names
            X_ref = X_ref.reindex(columns=_artifacts.feature_names, fill_value=0)

            # Compute reference predictions for prediction drift detection
            ref_prediction = _artifacts.model.predict_proba(X_ref)[:, 1]

            _drift_detector = DriftDetector(
                reference_data=X_ref.values,
                feature_names=_artifacts.feature_names,
                reference_predictions=ref_prediction,
            )
        else:
            logger.warning("drift_reference_data_not_found", path=str(ref_path))
    except Exception:
        logger.execption("drif_detector_initalization_failed")


def shutdown_resources() -> None:
    """Clean up resources on application shutdown."""
    global _artifacts, _shap_engine, _drift_detector
    _artifacts = None
    _shap_engine = None
    _drift_detector = None
    logger.info("api_resources_released")


def get_model_artifacts() -> ModelArtifacts | None:
    """Dependency provider for model artifacts."""
    return _artifacts


def get_shap_engine() -> ShapEngine | None:
    """Dependency provider for the SHAP engine."""
    return _shap_engine


def get_drift_detector() -> DriftDetector | None:
    """Dependency provider for the drift detector."""
    return _drift_detector


@lru_cache(maxsize=1)
def get_feature_names() -> list[str] | None:
    """Cached access to feature names for preprocessing."""
    if _artifacts is None:
        return None
    return _artifacts.feature_names

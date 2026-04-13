# src/api/dependencies.py
"""FastAPI dependency injection providers.

Manages the lifecycle of shared resources (models, explainers, drift
detectors) for multiple datasets. Each dataset has its own model,
SHAP engine, and drift detector loaded at startup.
"""

import pandas as pd
import structlog

from src.config import settings
from src.explain.shap_engine import ShapEngine
from src.model.registry import (
    ModelArtifacts,
    list_trained_models,
    load_model_artifacts,
)
from src.monitoring.drift import DriftDetector

logger = structlog.get_logger(__name__)

# Module-level state: keyed by dataset_id
#
# Keep stable references to the managed containers so tests can safely patch
# the public module globals without startup repopulating them.
_MANAGED_MODELS: dict[str, ModelArtifacts] = {}
_MANAGED_SHAP_ENGINES: dict[str, ShapEngine] = {}
_MANAGED_DRIFT_DETECTORS: dict[str, DriftDetector] = {}

_models: dict[str, ModelArtifacts] = _MANAGED_MODELS
_shap_engines: dict[str, ShapEngine] = _MANAGED_SHAP_ENGINES
_drift_detectors: dict[str, DriftDetector] = _MANAGED_DRIFT_DETECTORS
_default_dataset_id: str | None = None


def _resources_overridden() -> bool:
    """Return whether callers replaced the managed resource containers."""
    return any(
        (
            _models is not _MANAGED_MODELS,
            _shap_engines is not _MANAGED_SHAP_ENGINES,
            _drift_detectors is not _MANAGED_DRIFT_DETECTORS,
        )
    )


def initialize_resources() -> None:
    """Load model artifacts for all trained datasets.

    Called once during application startup (lifespan context).
    """
    global _default_dataset_id

    if _resources_overridden():
        logger.info(
            "resource_initialization_skipped",
            reason="external_state_override",
            loaded_models=list(_models.keys()),
            default_dataset=_default_dataset_id,
        )
        return

    _models.clear()
    _shap_engines.clear()
    _drift_detectors.clear()
    _default_dataset_id = None

    trained = list_trained_models()
    if not trained:
        # Try legacy flat layout
        try:
            artifacts = load_model_artifacts(dataset_id=None)
            _models[artifacts.dataset_id] = artifacts
            _shap_engines[artifacts.dataset_id] = ShapEngine(artifacts.model)
            _default_dataset_id = artifacts.dataset_id
            logger.info("legacy_model_loaded", dataset_id=artifacts.dataset_id)
        except FileNotFoundError:
            logger.warning(
                "no_models_found",
                detail="API will start but predictions return 503.",
            )
        except Exception:
            logger.exception("resource_initialization_failed")
        return

    for ds_id in trained:
        try:
            artifacts = load_model_artifacts(dataset_id=ds_id)
            _models[ds_id] = artifacts
            _shap_engines[ds_id] = ShapEngine(artifacts.model)

            if _default_dataset_id is None:
                _default_dataset_id = ds_id

            # Initialize drift detector
            _init_drift_detector(ds_id, artifacts)

            logger.info(
                "dataset_model_loaded",
                dataset_id=ds_id,
                n_features=len(artifacts.feature_names),
            )
        except Exception:
            logger.exception("model_load_failed", dataset_id=ds_id)

    logger.info(
        "all_resources_initialized",
        loaded_models=list(_models.keys()),
        default_dataset=_default_dataset_id,
    )


def _init_drift_detector(ds_id: str, artifacts: ModelArtifacts) -> None:
    """Initialize drift detector for a dataset using its synthetic test set."""
    try:
        ref_path = settings.data.dir / ds_id / "synthetic_test_set.csv"
        if not ref_path.exists():
            # Fallback to legacy path
            ref_path = settings.data.synthetic_test_path

        if ref_path.exists():
            ref_df = pd.read_csv(ref_path)
            target_cols = ["target", "risk_flag", "Risk_Flag"]
            for tc in target_cols:
                if tc in ref_df.columns:
                    ref_df = ref_df.drop(columns=[tc])
                    break

            ref_df = ref_df.reindex(columns=artifacts.feature_names, fill_value=0)
            ref_predictions = artifacts.model.predict_proba(ref_df)[:, 1]

            _drift_detectors[ds_id] = DriftDetector(
                reference_data=ref_df.values,
                feature_names=artifacts.feature_names,
                reference_predictions=ref_predictions,
            )
    except Exception:
        logger.exception("drift_detector_init_failed", dataset_id=ds_id)


def shutdown_resources() -> None:
    """Clean up all resources on application shutdown."""
    global _default_dataset_id
    _models.clear()
    _shap_engines.clear()
    _drift_detectors.clear()
    _default_dataset_id = None
    logger.info("api_resources_released")


def get_model_artifacts(
    dataset_id: str | None = None,
) -> ModelArtifacts | None:
    """Get model artifacts for a dataset."""
    ds_id = dataset_id or _default_dataset_id
    if ds_id is None:
        return None
    return _models.get(ds_id)


def get_shap_engine(dataset_id: str | None = None) -> ShapEngine | None:
    """Get SHAP engine for a dataset."""
    ds_id = dataset_id or _default_dataset_id
    if ds_id is None:
        return None
    return _shap_engines.get(ds_id)


def get_drift_detector(dataset_id: str | None = None) -> DriftDetector | None:
    """Get drift detector for a dataset."""
    ds_id = dataset_id or _default_dataset_id
    if ds_id is None:
        return None
    return _drift_detectors.get(ds_id)


def get_loaded_datasets() -> list[str]:
    """Return list of dataset IDs with loaded models."""
    return list(_models.keys())


def get_default_dataset_id() -> str | None:
    """Return the default dataset ID."""
    return _default_dataset_id

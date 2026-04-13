# src/model/registry.py
"""Multi-model artifact loading and validation.

Supports loading trained model artifacts for multiple datasets.
Each dataset has its own model directory containing the trained
classifier, feature names, and preprocessing pipeline.

Directory layout:
    models/
    ├── german_credit/
    │   ├── model.pkl
    │   ├── feature_names.pkl
    │   ├── pipeline.pkl
    │   └── threshold.json
    ├── lending_club/
    │   ├── model.pkl
    │   ├── feature_names.pkl
    │   └── pipeline.pkl
    └── ...
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ModelArtifacts:
    """Immutable container for loaded model artifacts.

    Attributes:
        model: The trained classifier (XGBoost or any sklearn-compatible).
        feature_names: Ordered list of feature column names after encoding.
        pipeline: Fitted sklearn ColumnTransformer (or None for legacy).
        model_path: Filesystem path from which the model was loaded.
        dataset_id: Identifier of the dataset this model was trained on.
    """

    model: Any
    feature_names: list[str]
    pipeline: Any | None
    model_path: Path
    dataset_id: str
    decision_threshold: float = 0.5

    def validate(self) -> None:
        """Run basic sanity checks on loaded artifacts.

        Raises:
            ValueError: If artifacts are inconsistent or corrupted.
        """
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Loaded model does not support predict_proba.")

        if not self.feature_names:
            raise ValueError("feature_names is empty.")

        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be between 0 and 1.")

        n_model_features = self.model.n_features_in_
        n_saved_features = len(self.feature_names)
        if n_model_features != n_saved_features:
            raise ValueError(
                f"Model expects {n_model_features} features but "
                f"feature_names contains {n_saved_features}."
            )

        logger.info(
            "model_artifacts_validated",
            dataset_id=self.dataset_id,
            n_features=n_saved_features,
            model_path=str(self.model_path),
            has_pipeline=self.pipeline is not None,
        )


def load_model_artifacts(
    dataset_id: str | None = None,
    model_dir: Path | None = None,
) -> ModelArtifacts:
    """Load trained model and feature names from disk.

    Supports two directory layouts:
    1. Multi-dataset: models/{dataset_id}/model.pkl
    2. Legacy flat:   models/xgb_model.pkl (backward compat)

    Args:
        dataset_id: Dataset identifier. If None, uses legacy flat layout.
        model_dir: Root models directory. Defaults to settings.model.dir.

    Returns:
        Validated ModelArtifacts instance.

    Raises:
        FileNotFoundError: If required artifact files are missing.
        ValueError: If loaded artifacts fail validation.
    """
    base_dir = model_dir or settings.model.dir

    if dataset_id:
        # Multi-dataset layout
        ds_dir = base_dir / dataset_id
        model_path = ds_dir / "model.pkl"
        features_path = ds_dir / "feature_names.pkl"
        pipeline_path = ds_dir / "pipeline.pkl"
        threshold_path = ds_dir / "threshold.json"
    else:
        # Legacy flat layout (backward compat)
        model_path = base_dir / settings.model.filename
        features_path = base_dir / settings.model.feature_names_filename
        pipeline_path = base_dir / "pipeline.pkl"
        threshold_path = base_dir / "threshold.json"
        dataset_id = "german_credit"  # Default assumption

    for path, label in [
        (model_path, "model"),
        (features_path, "feature_names"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} artifact not found at {path}. "
                "Run the training pipeline first: credit-risk-train"
            )

    logger.info(
        "loading_model_artifacts",
        dataset_id=dataset_id,
        model_dir=str(base_dir),
    )

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)

    # Pipeline is optional (legacy models don't have it)
    pipeline = None
    if pipeline_path.exists():
        pipeline = joblib.load(pipeline_path)
        logger.info("preprocessing_pipeline_loaded", path=str(pipeline_path))

    decision_threshold = 0.5
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold_payload = json.load(f) or {}
        decision_threshold = float(threshold_payload.get("decision_threshold", 0.5))
        logger.info(
            "decision_threshold_loaded",
            path=str(threshold_path),
            decision_threshold=round(decision_threshold, 4),
        )

    artifacts = ModelArtifacts(
        model=model,
        feature_names=feature_names,
        pipeline=pipeline,
        model_path=model_path,
        dataset_id=dataset_id,
        decision_threshold=decision_threshold,
    )
    artifacts.validate()

    return artifacts


def list_trained_models(model_dir: Path | None = None) -> list[str]:
    """List all dataset IDs that have trained model artifacts.

    Args:
        model_dir: Root models directory. Defaults to settings.model.dir.

    Returns:
        Sorted list of dataset identifiers with available models.
    """
    base_dir = model_dir or settings.model.dir
    if not base_dir.exists():
        return []

    trained = []
    for sub in sorted(base_dir.iterdir()):
        if sub.is_dir() and (sub / "model.pkl").exists():
            trained.append(sub.name)

    # Also check for legacy flat layout
    if (base_dir / settings.model.filename).exists() and "german_credit" not in trained:
        trained.insert(0, "german_credit")

    return trained

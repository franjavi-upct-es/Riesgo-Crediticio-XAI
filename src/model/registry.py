# src/model/registry.py
"""Model artifact loading and validation.

Provides a clean interface for loading trained model artifacts with
proper error handling and validation. Designed to support future model
versioning and registry integration (e.g., MLflow).
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import structlog
import xgboost as xgb

from src.config import settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ModelArtifacts:
    """Inmmutable container for loaded model artifacts.

    Attributes:
        model: The trained XGBoost classifier.
        feature_names: Ordered list of feature column names after encoding.
        model_path: Filesystem path from which the model was loaded.
    """

    model: xgb.XGBClassifier
    feature_names: list[str]
    model_path: Path

    def validate(self) -> None:
        """Run basic sanity checks on loaded artifacts.

        Raises:
            ValueError: If artifacts are inconsistent or corrupted.
        """
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Loaded model does not support predict_proba.")

        if not self.feature_names:
            raise ValueError("feature_names is empty.")

        n_model_features = self.model.n_features_in_
        n_saved_features = len(self.feature_names)
        if n_model_features != n_saved_features:
            raise ValueError(
                f"Model expects {n_model_features} features but "
                f"feature_names contains {n_saved_features}."
            )

        logger.info(
            "model_artifacts_validated",
            n_features=n_saved_features,
            model_path=str(self.model_path),
        )


def load_model_artifacts(
    model_dir: Path | None = None,
) -> ModelArtifacts:
    """Load trained model and feature names from disk.

    Args:
        model_dir: Directory contianing model artifacts.
            Defaults to the path configured in settings.

        Returns:
            Validated ModelArtifacts instance.

        Raises:
            FileNotFoundError: If any required artifact file is missing.
            ValueError: If loaded artifacts fail validation.
    """
    base_dir = model_dir or settings.model.dir
    model_path = base_dir / settings.model.filename
    features_path = base_dir / settings.model.feature_names_filename

    for path, label in [
        (model_path, "model"),
        (features_path, "feature_names"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} artifact not found at {path}. "
                "Run the training pipeline first: credit-risk-train"
            )

    logger.info("loading_model_artifacts", model_dir=str(base_dir))

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)

    artifacts = ModelArtifacts(model=model, feature_names=feature_names, model_path=model_path)
    artifacts.validate()

    return artifacts

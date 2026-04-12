# src/model/training_config.py
"""Training configuration loader.

Reads hyperparameters from a YAML config file and merges with
environment variable overrides. Provides a typed dataclass for
consumption by the training pipeline.

The YAML file is the canonical reference for experiment reproducibility,
while env vars allow CI/CD overrides without file changes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/training.yml")


@dataclass(frozen=True)
class ModelHyperparams:
    """XGBoost hyperparameters."""

    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    def to_xgb_params(self) -> dict[str, Any]:
        """Convert to kwargs dict for XGBClassifier (excluding meta params)."""
        return {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }


@dataclass(frozen=True)
class TrainingConfig:
    """Complete training pipeline configuration."""

    test_size: float = 0.2
    random_state: int = 42
    model: ModelHyperparams = field(default_factory=ModelHyperparams)
    smote_strategy: str = "minority"
    smote_k_neighbors: int = 5
    mlflow_experiment_name: str = "credit-risk-xai"
    mlflow_run_name_prefix: str = "xgb"
    mlflow_tags: dict[str, str] = field(default_factory=dict)


def load_training_config(config_path: Path | None = None) -> TrainingConfig:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to the YAML config. Defaults to configs/training.yml.

    Returns:
        Parsed TrainingConfig dataclass.
    """
    path = config_path or DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning("training_config_not_found", path=str(path), using="defaults")
        return TrainingConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    logger.info("training_config_loaded", path=str(path))

    data_cfg = raw.get("data", {})
    model_cfg = raw.get("model", {})
    smote_cfg = raw.get("smote", {})
    mlflow_cfg = raw.get("mlflow", {})

    hyperparams = ModelHyperparams(
        objective=model_cfg.get("objective", "binary:logistic"),
        eval_metric=model_cfg.get("eval_metric", "logloss"),
        n_estimators=model_cfg.get("n_estimators", 100),
        learning_rate=model_cfg.get("learning_rate", 0.1),
        max_depth=model_cfg.get("max_depth", 6),
        subsample=model_cfg.get("subsample", 0.8),
        colsample_bytree=model_cfg.get("colsample_bytree", 0.8),
        min_child_weight=model_cfg.get("min_child_weight", 1),
        gamma=model_cfg.get("gamma", 0.0),
        reg_alpha=model_cfg.get("reg_alpha", 0.0),
        reg_lambda=model_cfg.get("reg_lambda", 1.0),
    )

    return TrainingConfig(
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
        model=hyperparams,
        smote_strategy=smote_cfg.get("sampling_strategy", "minority"),
        smote_k_neighbors=smote_cfg.get("k_neighbors", 5),
        mlflow_experiment_name=mlflow_cfg.get("experiment_name", "credit-risk-xai"),
        mlflow_run_name_prefix=mlflow_cfg.get("run_name_prefix", "xgb"),
        mlflow_tags=mlflow_cfg.get("tags", {}),
    )

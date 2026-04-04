# src/model/train.py
"""Model training pipeline.

Loads data, encodes features using the shared preprocessing module,
trains an XGBoost classifier, and saves all artifacts. Optionally
tracks the experiment in MLflow with full parameter/metric/artifact
logging. Hyperparameters are loaded from configs/training.yml.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import structlog
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src.config import settings
from src.data.loader import load_uci_dataset
from src.data.preprocessing import encode_features
from src.model.experiment_tracker import ExperimentTracker
from src.model.training_config import TrainingConfig, load_training_config

logger = structlog.get_logger(__name__)


def train_model(config_path: Path | None = None) -> None:
    """Execute full trianing pipeline.

    Steps:
        1. Load training config from YAML.
        2. Load and preprocess the UCI German Credit dataset.
        3. Split into train/test sets.
        4. Train an XGBoost classifier with class-weight balancing.
        5. Save model and feature_names artifacts.
        6. Generate a SMOTE-balanced synthetic test set.
        7. Compute and save evaluation metrics.
        8. Log everything to MLflow (if enabled).
    """
    # --- 1. Load config ---
    cfg = load_training_config(config_path)

    # --- 2. Load raw data ---
    X_raw, y = load_uci_dataset()

    # --- 3. Encode features (shared logic) ---
    X_encoded = encode_features(X_raw)
    feature_names = X_encoded.columns.tolist()

    logger.info("features_prepared", n_features=len(feature_names))

    # --- 4. Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    logger.info(
        "data_split",
        train_size=len(X_train),
        test_size=len(X_test),
        train_class_dist=y_train.value_counts().to_dict(),  # type: ignore[union-attr]
    )

    # --- 5. Train XGBoost ---
    class_counts = y_train.value_counts()  # type: ignore[union-attr]
    scale_pos_weight = class_counts[0] / class_counts[1]

    xgb_params = cfg.model.to_xgb_params()
    model = xgb.XGBClassifier(
        **xgb_params,
        random_state=cfg.random_state,
        scale_pos_weight=scale_pos_weight,
    )

    logger.info(
        "training_started",
        n_estimators=cfg.model.n_estimators,
        learning_rate=cfg.model.learning_rate,
        max_depth=cfg.model.max_depth,
        scale_pos_weight=round(scale_pos_weight, 4),
    )

    model.fit(X_train, y_train)
    logger.info("training_completed")

    # --- 6. Save artifacts ---
    model_dir = settings.model.dir
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = settings.model.model_path
    features_path = settings.model.feature_names_path

    joblib.dump(model, model_path)
    joblib.dump(feature_names, features_path)

    # --- 7. Generate synthetic balanced test set ---
    _generate_sythetic_test_set(X_test, y_test, cfg)  # type: ignore[arg-type]

    # --- 8. Compute and save evaluation metrics ---
    from src.model.evaluate import compute_and_save_evaluation

    eval_result = compute_and_save_evaluation(
        model=model,
        X_test=X_test,  # type: ignore[arg-type]
        y_test=y_test,  # type: ignore[arg-type]
        feature_names=feature_names,
    )

    # --- 9. MLflow tracking ---
    tracker = ExperimentTracker(experiment_name=cfg.mlflow_experiment_name)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.mlflow_experiment_name}-{ts}"

    with tracker.start_run(run_name=run_name, tags=cfg.mlflow_tags):
        # Log all hyperparameters
        tracker.log_params(
            {
                "test_size": cfg.test_size,
                "random_state": cfg.random_state,
                "scale_pos_weight": round(scale_pos_weight, 4),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "n_features": len(feature_names),
                **{f"xgb_{k}": v for k, v in xgb_params.items()},
            }
        )

        # Log evaluation metrics
        tracker.log_metrics(eval_result["metrics"])

        # Log model artifact
        tracker.log_model(model, artifact_path="xgb-model")

        # Log evaluation JSON
        tracker.log_dict(eval_result, "evaluation_metrics.json")

        # Log config and model artifacts
        if model_path.exists():
            tracker.log_artifact(model_path)
        if features_path.exists():
            tracker.log_artifact(features_path)

    logger.info("training_pipeline_complete")


def _generate_sythetic_test_set(
    X_test: pd.DataFrame, y_test: pd.Series, cfg: TrainingConfig
) -> None:
    """Create a SMOTE-balanced version of the test set."""
    smote = SMOTE(
        sampling_strategy=cfg.smote_strategy,
        k_neighbors=cfg.smote_k_neighbors,
        random_state=cfg.random_state,
    )
    X_synthetic, y_synthetic = smote.fit_resample(X_test, y_test)  # type: ignore[assignment]

    if isinstance(y_synthetic, pd.Series):
        y_synthetic_df = y_synthetic.to_frame(name="risk_flag")
    else:
        y_synthetic_df = pd.DataFrame(y_synthetic, columns=["risk_flag"])  # type: ignore[arg-type]

    synthetic_data = pd.concat([X_synthetic, y_synthetic_df], axis=1)

    data_dir = settings.data.dir
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.data.synthetic_test_path

    synthetic_data.to_csv(output_path, index=False)

    logger.info(
        "synthetic_test_set_saved",
        path=str(output_path),
        n_samples=len(synthetic_data),
        class_distribution=y_synthetic_df["risk_flag"].value_counts().to_dict(),
    )


def main() -> None:
    """CLI entry point for the training pipeline."""
    structlog.configure(
        processors=[
            structlog.dev.ConsoleRenderer(),
        ],
    )

    try:
        train_model()
    except Exception:
        logger.exception("training_pipeline_failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

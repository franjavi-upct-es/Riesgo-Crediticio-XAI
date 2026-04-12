# src/model/train.py
"""Multi-dataset model training pipeline.

Loads data via the dataset adapter, builds and fits a sklearn
preprocessing pipeline, trains an XGBoost classifier, generates
a SMOTE-balanced test set, computes evaluation metrics, and logs
everything to MLflow. All artifacts are saved under models/{dataset_id}/.

Usage::

    # Train a specific dataset
    credit-risk-train --dataset german_credit

    # Or from Python
    from src.model.train import train_model
    train_model(dataset_id="german_credit")
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
from src.data.adapter import load_dataset
from src.data.preprocessing import (
    build_preprocessing_pipeline,
    fit_and_save_pipeline,
)
from src.data.schema import load_dataset_schema
from src.model.experiment_tracker import ExperimentTracker
from src.model.training_config import TrainingConfig, load_training_config

logger = structlog.get_logger(__name__)


def train_model(
    dataset_id: str = "german_credit",
    config_path: Path | None = None,
) -> None:
    """Execute the full training pipeline for a dataset.

    Steps:
        1. Load training config from YAML.
        2. Load dataset via schema + adapter.
        3. Build and fit sklearn preprocessing pipeline.
        4. Split into train/test sets.
        5. Train an XGBoost classifier with class-weight balancing.
        6. Save all artifacts to models/{dataset_id}/.
        7. Generate a SMOTE-balanced synthetic test set.
        8. Compute and save evaluation metrics.
        9. Log everything to MLflow (if enabled).

    Args:
        dataset_id: Which dataset to train on (must have a YAML schema).
        config_path: Optional path to training config YAML.
    """
    # --- 1. Load config ---
    cfg = load_training_config(config_path)

    # --- 2. Load dataset ---
    schema = load_dataset_schema(dataset_id)
    X_raw, y = load_dataset(schema)

    logger.info(
        "dataset_loaded",
        dataset_id=dataset_id,
        n_samples=len(X_raw),
        n_features=X_raw.shape[1],
    )

    # --- 3. Build and fit preprocessing pipeline ---
    pipeline = build_preprocessing_pipeline(schema)

    # Output directories
    model_dir = settings.model.dir / dataset_id
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir = settings.data.dir / dataset_id
    data_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = model_dir / "pipeline.pkl"
    feature_names = fit_and_save_pipeline(pipeline, X_raw, pipeline_path)

    # Transform all data
    X_encoded = pd.DataFrame(
        pipeline.transform(X_raw),
        columns=feature_names,
        index=X_raw.index,
    )

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
        train_class_dist=y_train.value_counts().to_dict(),
    )

    # --- 5. Train XGBoost ---
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]

    xgb_params = cfg.model.to_xgb_params()
    model = xgb.XGBClassifier(
        **xgb_params,
        random_state=cfg.random_state,
        scale_pos_weight=scale_pos_weight,
    )

    logger.info(
        "training_started",
        dataset_id=dataset_id,
        n_estimators=cfg.model.n_estimators,
        learning_rate=cfg.model.learning_rate,
        max_depth=cfg.model.max_depth,
        scale_pos_weight=round(scale_pos_weight, 4),
    )

    model.fit(X_train, y_train)
    logger.info("training_completed")

    # --- 6. Save artifacts ---
    model_path = model_dir / "model.pkl"
    features_path = model_dir / "feature_names.pkl"

    joblib.dump(model, model_path)
    joblib.dump(feature_names, features_path)

    logger.info(
        "artifacts_saved",
        model_path=str(model_path),
        features_path=str(features_path),
        pipeline_path=str(pipeline_path),
    )

    # --- 7. Generate synthetic balanced test set ---
    _generate_synthetic_test_set(X_test, y_test, cfg, data_dir)

    # --- 8. Compute and save evaluation metrics ---
    from src.model.evaluate import compute_and_save_evaluation

    eval_output = data_dir / "evaluation_metrics.json"
    eval_result = compute_and_save_evaluation(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        output_path=eval_output,
    )

    # --- 9. MLflow tracking ---
    tracker = ExperimentTracker(experiment_name=cfg.mlflow_experiment_name)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.mlflow_run_name_prefix}-{dataset_id}-{ts}"

    with tracker.start_run(run_name=run_name, tags={**cfg.mlflow_tags, "dataset_id": dataset_id}):
        tracker.log_params(
            {
                "dataset_id": dataset_id,
                "test_size": cfg.test_size,
                "random_state": cfg.random_state,
                "scale_pos_weight": round(scale_pos_weight, 4),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "n_features": len(feature_names),
                **{f"xgb_{k}": v for k, v in xgb_params.items()},
            }
        )

        tracker.log_metrics(eval_result["metrics"])
        tracker.log_model(model, artifact_path="xgb-model")
        tracker.log_dict(eval_result, "evaluation_metrics.json")

        if model_path.exists():
            tracker.log_artifact(model_path)
        if features_path.exists():
            tracker.log_artifact(features_path)
        if pipeline_path.exists():
            tracker.log_artifact(pipeline_path)

    logger.info("training_pipeline_complete", dataset_id=dataset_id)


def _generate_synthetic_test_set(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: TrainingConfig,
    output_dir: Path,
) -> None:
    """Create a SMOTE-balanced version of the test set."""
    smote = SMOTE(
        sampling_strategy=cfg.smote_strategy,
        k_neighbors=cfg.smote_k_neighbors,
        random_state=cfg.random_state,
    )
    X_synthetic, y_synthetic = smote.fit_resample(X_test, y_test)

    if isinstance(y_synthetic, pd.Series):
        y_synthetic_df = y_synthetic.to_frame(name="target")
    else:
        y_synthetic_df = pd.DataFrame(y_synthetic, columns=["target"])

    synthetic_data = pd.concat([X_synthetic, y_synthetic_df], axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_test_set.csv"
    synthetic_data.to_csv(output_path, index=False)

    logger.info(
        "synthetic_test_set_saved",
        path=str(output_path),
        n_samples=len(synthetic_data),
        class_distribution=y_synthetic_df["target"].value_counts().to_dict(),
    )


def main() -> None:
    """CLI entry point for the training pipeline."""
    import argparse

    structlog.configure(processors=[structlog.dev.ConsoleRenderer()])

    parser = argparse.ArgumentParser(description="Train a credit risk model")
    parser.add_argument(
        "--dataset",
        default="german_credit",
        help="Dataset ID to train on (must have a YAML schema in configs/datasets/)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to training config YAML (defaults to configs/training.yml)",
    )
    args = parser.parse_args()

    try:
        config_path = Path(args.config) if args.config else None
        train_model(dataset_id=args.dataset, config_path=config_path)
    except Exception:
        logger.exception("training_pipeline_failed")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()

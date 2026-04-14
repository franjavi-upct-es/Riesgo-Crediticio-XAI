# src/model/train.py
"""Multi-dataset model training pipeline.

Loads data via the dataset adapter, builds and fits a sklearn
preprocessing pipeline, trains an XGBoost classifier with optional
Optuna hyperparameter tuning, generates a SMOTE-balanced test set,
computes evaluation metrics, and logs everything to MLflow.
All artifacts are saved under models/{dataset_id}/.

Usage::

    # Train a specific dataset (with tuning)
    credit-risk-train --dataset german_credit

    # Train without tuning (use config defaults)
    credit-risk-train --dataset german_credit --no-tune

    # Or from Python
    from src.model.train import train_model
    train_model(dataset_id="german_credit")
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

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


def _compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Compute the negative/positive class ratio for XGBoost."""
    class_counts = y_train.value_counts()
    negative_count = int(class_counts.get(0, 0))
    positive_count = int(class_counts.get(1, 0))

    if positive_count == 0:
        return 1.0

    return max(1.0, float(negative_count / positive_count))


def _get_smote(cfg: TrainingConfig, y_train: pd.Series) -> SMOTE | None:
    """Build a SMOTE instance adjusted to the available minority samples."""
    class_counts = y_train.value_counts()
    if len(class_counts) < 2:
        logger.warning("smote_skipped_single_class_training_data")
        return None

    minority_count = int(class_counts.min())
    if minority_count < 2:
        logger.warning(
            "smote_skipped_insufficient_minority_examples",
            minority_count=minority_count,
        )
        return None

    k_neighbors = min(cfg.smote_k_neighbors, minority_count - 1)
    if k_neighbors < 1:
        logger.warning(
            "smote_skipped_invalid_k_neighbors",
            requested_neighbors=cfg.smote_k_neighbors,
            minority_count=minority_count,
        )
        return None

    return SMOTE(
        sampling_strategy=cfg.smote_strategy,
        k_neighbors=k_neighbors,
        random_state=cfg.random_state,
    )


def _apply_smote_to_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: TrainingConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE oversampling to the training set for class balance.

    For datasets larger than cfg.smote_max_samples, SMOTE is skipped to
    avoid excessive memory usage. The caller should rely on
    scale_pos_weight for class balancing in those cases.
    """
    if len(X_train) > cfg.smote_max_samples:
        logger.info(
            "smote_skipped_large_dataset",
            n_samples=len(X_train),
            max_samples=cfg.smote_max_samples,
        )
        return X_train, y_train

    smote = _get_smote(cfg, y_train)
    if smote is None:
        return X_train, y_train

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    if not isinstance(X_resampled, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    if isinstance(y_resampled, pd.DataFrame):
        y_resampled = y_resampled.iloc[:, 0]
        y_resampled.name = y_train.name
    elif not isinstance(y_resampled, pd.Series):
        y_resampled = pd.Series(y_resampled, name=y_train.name)

    logger.info(
        "smote_applied_to_train",
        original_size=len(X_train),
        resampled_size=len(X_resampled),
        k_neighbors=smote.k_neighbors,
        class_distribution=y_resampled.value_counts().to_dict(),
    )
    return X_resampled, y_resampled


def _fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    cfg: TrainingConfig,
) -> xgb.XGBClassifier:
    """Fit XGBoost with validation-based early stopping."""
    model_params = dict(params)
    model_params.setdefault("random_state", cfg.random_state)
    model = xgb.XGBClassifier(
        **model_params,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _select_decision_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    """Choose a probability threshold from validation predictions.

    Uses a fixed grid of 199 candidates instead of iterating over every
    unique probability value, which caused O(n²) behaviour on large sets.
    """
    if metric != "f1":
        raise ValueError(f"Unsupported threshold metric: {metric}")

    candidate_thresholds = np.linspace(0.05, 0.95, 199)

    y_true_arr = np.asarray(y_true)
    scores = np.array(
        [
            float(f1_score(y_true_arr, (y_proba >= t).astype(int), zero_division=0))
            for t in candidate_thresholds
        ]
    )

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_threshold = float(candidate_thresholds[best_idx])

    # Among ties, prefer the threshold closest to 0.5
    tie_mask = np.isclose(scores, best_score)
    if tie_mask.sum() > 1:
        tie_thresholds = candidate_thresholds[tie_mask]
        best_threshold = float(tie_thresholds[np.argmin(np.abs(tie_thresholds - 0.5))])

    return round(best_threshold, 4), best_score


def _subsample_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    max_samples: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified subsample to cap dataset size for expensive operations."""
    if len(X) <= max_samples:
        return X, y
    _, X_sub, _, y_sub = train_test_split(
        X, y,
        test_size=max_samples,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "dataset_subsampled",
        original_size=len(X),
        subsampled_size=len(X_sub),
        max_samples=max_samples,
    )
    return X_sub, y_sub


def _tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: TrainingConfig,
    n_trials: int = 50,
) -> dict:
    """Run Optuna hyperparameter search with stratified cross-validation.

    For large datasets, a stratified subsample is used to keep memory
    and runtime manageable. Returns the best hyperparameter dict ready
    for XGBClassifier.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Subsample large datasets for tuning to reduce memory and time
    X_tune, y_tune = _subsample_stratified(
        X_train, y_train, cfg.tuning_max_samples, cfg.random_state,
    )

    base_scale_pos_weight = _compute_scale_pos_weight(y_tune)
    class_counts = y_tune.value_counts()
    minority_count = int(class_counts.min()) if len(class_counts) == 2 else 0
    n_splits = min(cfg.tuning_cv_folds, minority_count)

    if n_splits < 2:
        logger.warning(
            "optuna_tuning_skipped_insufficient_class_counts",
            minority_count=minority_count,
            requested_folds=cfg.tuning_cv_folds,
        )
        return {
            **cfg.model.to_xgb_params(),
            "scale_pos_weight": base_scale_pos_weight,
            "n_jobs": cfg.n_jobs,
            "random_state": cfg.random_state,
        }

    def objective(trial: optuna.Trial) -> float:
        max_scale_pos_weight = max(1.0, base_scale_pos_weight * 1.5)
        scale_pos_weight = (
            1.0
            if np.isclose(max_scale_pos_weight, 1.0)
            else trial.suggest_float(
                "scale_pos_weight",
                1.0,
                max_scale_pos_weight,
                log=True,
            )
        )
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "n_jobs": cfg.n_jobs,
            "random_state": cfg.random_state,
        }

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=cfg.random_state,
        )
        auc_scores = []

        for train_idx, val_idx in cv.split(X_tune, y_tune):
            X_fold_train = X_tune.iloc[train_idx]
            y_fold_train = y_tune.iloc[train_idx]
            X_fold_val = X_tune.iloc[val_idx]
            y_fold_val = y_tune.iloc[val_idx]

            X_fold_train_balanced, y_fold_train_balanced = _apply_smote_to_train(
                X_fold_train,
                y_fold_train,
                cfg,
            )
            model = _fit_model(
                X_train=X_fold_train_balanced,
                y_train=y_fold_train_balanced,
                X_val=X_fold_val,
                y_val=y_fold_val,
                params=params,
                cfg=cfg,
            )

            if y_fold_val.nunique() < 2:
                logger.warning(
                    "tuning_fold_skipped_single_class_validation",
                    class_distribution=y_fold_val.value_counts().to_dict(),
                )
                continue

            y_proba = model.predict_proba(X_fold_val)[:, 1]
            auc_scores.append(roc_auc_score(y_fold_val, y_proba))

        return float(np.mean(auc_scores)) if auc_scores else 0.0

    study = optuna.create_study(direction="maximize", study_name="xgb-credit-risk")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["objective"] = "binary:logistic"
    best["eval_metric"] = "logloss"
    best.setdefault("scale_pos_weight", base_scale_pos_weight)
    best["n_jobs"] = cfg.n_jobs
    best["random_state"] = cfg.random_state

    logger.info(
        "optuna_tuning_complete",
        best_auc_cv=round(study.best_value, 4),
        best_params={k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()},
        n_trials=n_trials,
    )

    return best


def train_model(
    dataset_id: str = "german_credit",
    config_path: Path | None = None,
    tune: bool = True,
    n_trials: int = 50,
) -> None:
    """Execute the full training pipeline for a dataset.

    Steps:
        1. Load training config from YAML.
        2. Load dataset via schema + adapter.
        3. Split raw data into train/test sets.
        4. Fit preprocessing only on the training split.
        5. Apply SMOTE only to training folds / final fit data.
        6. (Optional) Tune hyperparameters with Optuna cross-validation.
        7. Train an XGBoost classifier with validation-based early stopping.
        8. Learn an operating threshold from validation predictions.
        9. Save all artifacts to models/{dataset_id}/.
        10. Generate a SMOTE-balanced synthetic test set.
        11. Compute and save evaluation metrics.
        12. Log everything to MLflow (if enabled).

    Args:
        dataset_id: Which dataset to train on (must have a YAML schema).
        config_path: Optional path to training config YAML.
        tune: Whether to run Optuna hyperparameter tuning.
        n_trials: Number of Optuna trials (ignored if tune=False).
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

    # --- 3. Train/test split on raw data to avoid preprocessing leakage ---
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    logger.info(
        "raw_data_split",
        train_size=len(X_train_raw),
        test_size=len(X_test_raw),
        train_class_dist=y_train.value_counts().to_dict(),
    )

    # Output directories
    model_dir = settings.model.dir / dataset_id
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir = settings.data.dir / dataset_id
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- 4. Build and fit preprocessing pipeline on the training split ---
    pipeline = build_preprocessing_pipeline(schema)
    pipeline_path = model_dir / "pipeline.pkl"
    feature_names = fit_and_save_pipeline(pipeline, X_train_raw, pipeline_path)

    X_train = pd.DataFrame(
        pipeline.transform(X_train_raw),
        columns=feature_names,
        index=X_train_raw.index,
    )
    X_test = pd.DataFrame(
        pipeline.transform(X_test_raw),
        columns=feature_names,
        index=X_test_raw.index,
    )

    logger.info(
        "features_prepared",
        n_features=len(feature_names),
        scaler_fitted_on_training_only=True,
    )

    # --- 6. Hyperparameter tuning or defaults ---
    base_scale_pos_weight = _compute_scale_pos_weight(y_train)

    if tune:
        logger.info("optuna_tuning_started", n_trials=n_trials)
        xgb_params = _tune_hyperparameters(
            X_train,
            y_train,
            cfg,
            n_trials=n_trials,
        )
    else:
        xgb_params = {
            **cfg.model.to_xgb_params(),
            "scale_pos_weight": base_scale_pos_weight,
            "n_jobs": cfg.n_jobs,
        }

    # --- 7. Train final model with early stopping on an untouched validation split ---
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=cfg.validation_size,
        random_state=cfg.random_state,
        stratify=y_train,
    )

    X_fit_balanced, y_fit_balanced = _apply_smote_to_train(X_fit, y_fit, cfg)

    logger.info(
        "training_started",
        dataset_id=dataset_id,
        tuned=tune,
        n_estimators=xgb_params.get("n_estimators", cfg.model.n_estimators),
        learning_rate=xgb_params.get("learning_rate", cfg.model.learning_rate),
        max_depth=xgb_params.get("max_depth", cfg.model.max_depth),
        scale_pos_weight=round(float(xgb_params.get("scale_pos_weight", base_scale_pos_weight)), 4),
        early_stopping_rounds=cfg.early_stopping_rounds,
        validation_size=cfg.validation_size,
    )

    model = _fit_model(
        X_train=X_fit_balanced,
        y_train=y_fit_balanced,
        X_val=X_val,
        y_val=y_val,
        params=xgb_params,
        cfg=cfg,
    )

    best_iteration = getattr(model, "best_iteration", None)
    logger.info("training_completed", best_iteration=best_iteration)

    # --- 8. Learn a validation-based threshold ---
    y_val_proba = model.predict_proba(X_val)[:, 1]
    decision_threshold, threshold_score = _select_decision_threshold(
        y_true=y_val,
        y_proba=y_val_proba,
        metric=cfg.threshold_metric,
    )

    logger.info(
        "decision_threshold_selected",
        decision_threshold=decision_threshold,
        metric=cfg.threshold_metric,
        metric_score=round(threshold_score, 4),
    )

    # --- 9. Save artifacts ---
    model_path = model_dir / "model.pkl"
    features_path = model_dir / "feature_names.pkl"
    threshold_path = model_dir / "threshold.json"

    joblib.dump(model, model_path)
    joblib.dump(feature_names, features_path)
    with open(threshold_path, "w") as f:
        json.dump(
            {
                "decision_threshold": decision_threshold,
                "selection_metric": cfg.threshold_metric,
                "validation_score": round(threshold_score, 4),
            },
            f,
            indent=2,
        )

    logger.info(
        "artifacts_saved",
        model_path=str(model_path),
        features_path=str(features_path),
        pipeline_path=str(pipeline_path),
        threshold_path=str(threshold_path),
    )

    # --- 10. Generate synthetic balanced test set ---
    _generate_synthetic_test_set(X_test, y_test, cfg, data_dir)

    # --- 11. Compute and save evaluation metrics ---
    from src.model.evaluate import compute_and_save_evaluation

    eval_output = data_dir / "evaluation_metrics.json"
    eval_result = compute_and_save_evaluation(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        decision_threshold=decision_threshold,
        output_path=eval_output,
        shap_max_samples=cfg.shap_max_samples,
    )

    # --- 12. MLflow tracking ---
    tracker = ExperimentTracker(experiment_name=cfg.mlflow_experiment_name)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.mlflow_run_name_prefix}-{dataset_id}-{ts}"

    with tracker.start_run(run_name=run_name, tags={**cfg.mlflow_tags, "dataset_id": dataset_id}):
        tracker.log_params(
            {
                "dataset_id": dataset_id,
                "test_size": cfg.test_size,
                "validation_size": cfg.validation_size,
                "random_state": cfg.random_state,
                "base_scale_pos_weight": round(base_scale_pos_weight, 4),
                "train_samples": len(X_train),
                "fit_samples_after_smote": len(X_fit_balanced),
                "test_samples": len(X_test),
                "n_features": len(feature_names),
                "tuned": tune,
                "n_trials": n_trials if tune else 0,
                "early_stopping_rounds": cfg.early_stopping_rounds,
                "tuning_cv_folds": cfg.tuning_cv_folds,
                "decision_threshold": decision_threshold,
                "threshold_metric": cfg.threshold_metric,
                "threshold_metric_score": round(threshold_score, 4),
                "best_iteration": best_iteration,
                **{f"xgb_{k}": v for k, v in xgb_params.items() if k not in ("random_state",)},
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
        if threshold_path.exists():
            tracker.log_artifact(threshold_path)

    logger.info("training_pipeline_complete", dataset_id=dataset_id)


def _generate_synthetic_test_set(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: TrainingConfig,
    output_dir: Path,
) -> None:
    """Create a SMOTE-balanced version of the test set."""
    smote = _get_smote(cfg, y_test)
    if smote is None:
        logger.warning("synthetic_test_set_skipped", reason="smote_unavailable")
        return

    X_synthetic, y_synthetic = smote.fit_resample(X_test, y_test)

    if not isinstance(X_synthetic, pd.DataFrame):
        X_synthetic = pd.DataFrame(X_synthetic, columns=X_test.columns)

    if isinstance(y_synthetic, pd.Series):
        y_synthetic_df = y_synthetic.to_frame(name="target")
    elif isinstance(y_synthetic, pd.DataFrame):
        y_synthetic_df = y_synthetic.copy()
        y_synthetic_df.columns = ["target"]
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


def _discover_dataset_ids() -> list[str]:
    """Return all dataset IDs found in configs/datasets/."""
    datasets_dir = Path("configs/datasets")
    return sorted(p.stem for p in datasets_dir.glob("*.yml"))


def main() -> None:
    """CLI entry point for the training pipeline."""
    import argparse

    structlog.configure(processors=[structlog.dev.ConsoleRenderer()])

    parser = argparse.ArgumentParser(description="Train a credit risk model")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset ID to train on (omit to train all datasets in configs/datasets/)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to training config YAML (defaults to configs/training.yml)",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip Optuna hyperparameter tuning and use config defaults",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter search (default: 50)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    dataset_ids = [args.dataset] if args.dataset else _discover_dataset_ids()

    failed = []
    for dataset_id in dataset_ids:
        try:
            train_model(
                dataset_id=dataset_id,
                config_path=config_path,
                tune=not args.no_tune,
                n_trials=args.n_trials,
            )
        except Exception:
            logger.exception("training_pipeline_failed", dataset_id=dataset_id)
            failed.append(dataset_id)

    if failed:
        logger.error("some_datasets_failed", failed=failed)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()

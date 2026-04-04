# src/model/evaluate.py
"""Model evaluation and metrics computation.

Computes classification metrics (AUC, F1, confusion matrix, ROC curve)
and global SHAP feature importance on the synthetic test set, then
persists them as a JSON artifact. The evaluation API endpoint serves
this pre-computed data to the dashboard without re-running inference.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import structlog
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import settings

logger = structlog.get_logger(__name__)


def compute_and_save_evaluation(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    output_path: Path | None = None,
) -> dict:
    """Compute full evaluation metrics and persist as JSON.

    Args:
        model: Trained XGBoost classifier.
        X_test: Test features (encoded).
        y_test: True labels.
        feature_names: Ordered feature names.
        output_path: Where to save the JSON. Defaults to config path.

    Returns:
        Dictionary with all computed metrics.
    """
    out = output_path or settings.data.evaluation_metrics_path
    logger.info("evaluation_started", n_samples=len(X_test))

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    # --- Classification metrics ---
    auc = float(roc_auc_score(y_test, y_proba))
    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred).tolist()

    # --- ROC curve (sample to ~100 points for JSON size) ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    step = max(1, len(fpr) // 100)
    roc_data = [
        {"fpr": round(float(fpr[i]), 4), "top": round(float(tpr[i]), 4)}
        for i in range(0, len(fpr), step)
    ]

    # --- Prediction distribution ---
    hist_counts, hist_edges = np.histogram(y_proba, bins=20, range=(0, 1))
    distribution = [
        {
            "bin_start": round(float(hist_edges[i]), 3),
            "bin_end": round(float(hist_edges[i + 1]), 3),
            "count": int(hist_counts[i]),
        }
        for i in range(len(hist_counts))
    ]

    # --- Global SHAP importance ---
    logger.info("computing_shap_importance")
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_test)

    if isinstance(shap_values_raw, list):
        sv = np.asarray(shap_values_raw[1] if len(shap_values_raw) > 1 else shap_values_raw[0])
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        sv = shap_values_raw[1] if shap_values_raw.shape[0] > 1 else shap_values_raw[0]
    else:
        sv = np.asarray(shap_values_raw)

    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance = sorted(
        [
            {"feature": name, "importance": round(float(val), 6)}
            for name, val in zip(feature_names, mean_abs_shap)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # --- Assemble result ---
    result = {
        "metrics": {
            "auc": round(auc, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        },
        "confusion_matrix": {
            "matrix": cm,
            "labels": ["No Default (0)", "Default (1)"],
        },
        "roc_curve": roc_data,
        "prediction_distribution": distribution,
        "shap_importance": importance[:20],  # Top 20 features
        "dataset_info": {
            "n_samples": len(X_test),
            "n_features": len(feature_names),
            "class_distribution": y_test.value_counts().to_dict(),
        },
    }

    # Persist
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        "evaluation_saved",
        path=str(out),
        auc=result["metrics"]["auc"],
        f1=result["metrics"]["f1"],
    )

    return result

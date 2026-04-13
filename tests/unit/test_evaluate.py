# tests/unit/test_evaluate.py
"""Unit tests for src.model.evaluate.

Verifies metric computation, JSON persistence, and edge cases
using a mock model with deterministic predictions.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from src.model.evaluate import compute_and_save_evaluation


@pytest.fixture
def mock_model():
    """A model that returns predictable probabilities."""
    model = MagicMock()
    # Returns [1-p, p] for each sample
    model.predict_proba.return_value = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.4, 0.6],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.5, 0.5],
            [0.85, 0.15],
        ]
    )
    return model


@pytest.fixture
def test_data():
    """10-sample test set with 3 features."""
    X = pd.DataFrame(
        {
            "f1": np.random.randn(10),
            "f2": np.random.randn(10),
            "f3": np.random.randn(10),
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 0], name="target")
    return X, y


@pytest.fixture
def feature_names():
    return ["f1", "f2", "f3"]


class TestComputeAndSaveEvaluation:
    @patch("src.model.evaluate.shap.TreeExplainer")
    def test_returns_all_expected_keys(
        self,
        mock_explainer_cls,
        mock_model,
        test_data,
        feature_names,
        tmp_path,
    ):
        X, y = test_data
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.random.randn(10, 3)
        mock_explainer_cls.return_value = mock_explainer

        output = tmp_path / "eval.json"
        result = compute_and_save_evaluation(mock_model, X, y, feature_names, output)

        assert "metrics" in result
        assert "decision_threshold" in result
        assert "confusion_matrix" in result
        assert "roc_curve" in result
        assert "prediction_distribution" in result
        assert "shap_importance" in result
        assert "dataset_info" in result

    @patch("src.model.evaluate.shap.TreeExplainer")
    def test_metrics_have_expected_fields(
        self,
        mock_explainer_cls,
        mock_model,
        test_data,
        feature_names,
        tmp_path,
    ):
        X, y = test_data
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.random.randn(10, 3)
        mock_explainer_cls.return_value = mock_explainer

        result = compute_and_save_evaluation(mock_model, X, y, feature_names, tmp_path / "e.json")

        metrics = result["metrics"]
        assert "auc" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert 0 <= metrics["auc"] <= 1
        assert 0 <= metrics["f1"] <= 1

    @patch("src.model.evaluate.shap.TreeExplainer")
    def test_confusion_matrix_shape(
        self,
        mock_explainer_cls,
        mock_model,
        test_data,
        feature_names,
        tmp_path,
    ):
        X, y = test_data
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.random.randn(10, 3)
        mock_explainer_cls.return_value = mock_explainer

        result = compute_and_save_evaluation(mock_model, X, y, feature_names, tmp_path / "e.json")

        cm = result["confusion_matrix"]
        assert len(cm["matrix"]) == 2
        assert len(cm["matrix"][0]) == 2
        assert len(cm["labels"]) == 2

    @patch("src.model.evaluate.shap.TreeExplainer")
    def test_roc_curve_has_points(
        self,
        mock_explainer_cls,
        mock_model,
        test_data,
        feature_names,
        tmp_path,
    ):
        X, y = test_data
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.random.randn(10, 3)
        mock_explainer_cls.return_value = mock_explainer

        result = compute_and_save_evaluation(mock_model, X, y, feature_names, tmp_path / "e.json")

        roc = result["roc_curve"]
        assert len(roc) > 0
        assert "fpr" in roc[0]
        assert "tpr" in roc[0]

    @patch("src.model.evaluate.shap.TreeExplainer")
    def test_prediction_distribution_bins(
        self,
        mock_explainer_cls,
        mock_model,
        test_data,
        feature_names,
        tmp_path,
    ):
        X, y = test_data
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.random.randn(10, 3)
        mock_explainer_cls.return_value = mock_explainer

        result = compute_and_save_evaluation(mock_model, X, y, feature_names, tmp_path / "e.json")

        dist = result["prediction_distribution"]
        assert len(dist) == 20  # 20 bins
        assert "bin_start" in dist[0]
        assert "count" in dist[0]

    @patch("src.model.evaluate.shap.TreeExplainer")
    def test_shap_importance_sorted_descending(
        self,
        mock_explainer_cls,
        mock_model,
        test_data,
        feature_names,
        tmp_path,
    ):
        X, y = test_data
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.array(
            [
                [0.5, 0.1, 0.3],
            ]
            * 10
        )
        mock_explainer_cls.return_value = mock_explainer

        result = compute_and_save_evaluation(mock_model, X, y, feature_names, tmp_path / "e.json")

        importance = result["shap_importance"]
        values = [item["importance"] for item in importance]
        assert values == sorted(values, reverse=True)

    @patch("src.model.evaluate.shap.TreeExplainer")
    def test_saves_json_to_disk(
        self,
        mock_explainer_cls,
        mock_model,
        test_data,
        feature_names,
        tmp_path,
    ):
        X, y = test_data
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.random.randn(10, 3)
        mock_explainer_cls.return_value = mock_explainer

        output = tmp_path / "eval.json"
        compute_and_save_evaluation(mock_model, X, y, feature_names, output)

        assert output.exists()
        import json

        with open(output) as f:
            loaded = json.load(f)
        assert "metrics" in loaded

    @patch("src.model.evaluate.shap.TreeExplainer")
    def test_dataset_info_correct(
        self,
        mock_explainer_cls,
        mock_model,
        test_data,
        feature_names,
        tmp_path,
    ):
        X, y = test_data
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.random.randn(10, 3)
        mock_explainer_cls.return_value = mock_explainer

        result = compute_and_save_evaluation(mock_model, X, y, feature_names, tmp_path / "e.json")

        info = result["dataset_info"]
        assert info["n_samples"] == 10
        assert info["n_features"] == 3

    @patch("src.model.evaluate.shap.TreeExplainer")
    def test_uses_custom_decision_threshold(
        self,
        mock_explainer_cls,
        mock_model,
        test_data,
        feature_names,
        tmp_path,
    ):
        X, y = test_data
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.random.randn(10, 3)
        mock_explainer_cls.return_value = mock_explainer

        result = compute_and_save_evaluation(
            mock_model,
            X,
            y,
            feature_names,
            decision_threshold=0.8,
            output_path=tmp_path / "e.json",
        )

        assert result["decision_threshold"] == 0.8
        assert result["metrics"]["recall"] < 1.0

# tests/integration/test_evaluation_api.py
"""Integration tests for the /evaluation/* endpoints.

Verifies that evaluation data is served correctly when the metrics
JSON artifact exists, and returns 503 when it does not.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.auth import verify_api_key
from src.api.routes.evaluation import clear_evaluation_cache

# ---------------------------------------------------------------------------
# Sample evaluation data matching the schema from model/evaluate.py
# ---------------------------------------------------------------------------

SAMPLE_EVALUATION = {
    "metrics": {
        "auc": 0.8521,
        "f1": 0.7843,
        "precision": 0.8012,
        "recall": 0.7681,
    },
    "confusion_matrix": {
        "matrix": [[120, 30], [25, 125]],
        "labels": ["No Default (0)", "Default (1)"],
    },
    "roc_curve": [
        {"fpr": 0.0, "tpr": 0.0},
        {"fpr": 0.1, "tpr": 0.65},
        {"fpr": 0.3, "tpr": 0.85},
        {"fpr": 1.0, "tpr": 1.0},
    ],
    "prediction_distribution": [
        {"bin_start": 0.0, "bin_end": 0.05, "count": 40},
        {"bin_start": 0.45, "bin_end": 0.5, "count": 25},
        {"bin_start": 0.9, "bin_end": 0.95, "count": 15},
    ],
    "shap_importance": [
        {"feature": "checking_status_no_checking", "importance": 0.15432},
        {"feature": "duration", "importance": 0.09821},
        {"feature": "credit_amount", "importance": 0.08345},
    ],
    "dataset_info": {
        "n_samples": 300,
        "n_features": 48,
        "class_distribution": {"0": 150, "1": 150},
    },
}


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the evaluation cache before each test."""
    clear_evaluation_cache()
    yield
    clear_evaluation_cache()


@pytest.fixture
def client_with_eval(tmp_path: Path):
    """TestClient where the evaluation JSON artifact exists."""
    metrics_file = tmp_path / "evaluation_metrics.json"
    metrics_file.write_text(json.dumps(SAMPLE_EVALUATION))

    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: None

    with patch("src.api.routes.evaluation.settings") as mock_settings:
        mock_settings.data.evaluation_metrics_path = metrics_file
        with TestClient(app) as tc:
            yield tc

    app.dependency_overrides.clear()


@pytest.fixture
def client_no_eval(tmp_path: Path):
    """TestClient where the evaluation artifact does NOT exist."""
    missing_path = tmp_path / "nonexistent.json"

    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: None

    with patch("src.api.routes.evaluation.settings") as mock_settings:
        mock_settings.data.evaluation_metrics_path = missing_path
        with TestClient(app) as tc:
            yield tc

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


class TestFullEvaluation:
    def test_returns_200_with_all_keys(self, client_with_eval: TestClient):
        resp = client_with_eval.get("/evaluation/full")
        assert resp.status_code == 200
        body = resp.json()
        assert "metrics" in body
        assert "confusion_matrix" in body
        assert "roc_curve" in body
        assert "shap_importance" in body
        assert "prediction_distribution" in body
        assert "dataset_info" in body

    def test_returns_503_when_missing(self, client_no_eval: TestClient):
        resp = client_no_eval.get("/evaluation/full")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Individual endpoints
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_returns_metrics_and_dataset_info(self, client_with_eval: TestClient):
        resp = client_with_eval.get("/evaluation/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["metrics"]["auc"] == pytest.approx(0.8521)
        assert body["metrics"]["f1"] == pytest.approx(0.7843)
        assert "dataset_info" in body

    def test_returns_503_when_missing(self, client_no_eval: TestClient):
        resp = client_no_eval.get("/evaluation/metrics")
        assert resp.status_code == 503


class TestConfusionMatrix:
    def test_returns_matrix_and_labels(self, client_with_eval: TestClient):
        resp = client_with_eval.get("/evaluation/confusion_matrix")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["matrix"]) == 2
        assert len(body["labels"]) == 2

    def test_returns_503_when_missing(self, client_no_eval: TestClient):
        resp = client_no_eval.get("/evaluation/confusion_matrix")
        assert resp.status_code == 503


class TestRocCurve:
    def test_returns_curve_and_auc(self, client_with_eval: TestClient):
        resp = client_with_eval.get("/evaluation/roc_curve")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["roc_curve"]) == 4
        assert body["auc"] == pytest.approx(0.8521)

    def test_returns_503_when_missing(self, client_no_eval: TestClient):
        resp = client_no_eval.get("/evaluation/roc_curve")
        assert resp.status_code == 503


class TestShapImportance:
    def test_returns_sorted_features(self, client_with_eval: TestClient):
        resp = client_with_eval.get("/evaluation/shap_importance")
        assert resp.status_code == 200
        features = resp.json()["shap_importance"]
        assert len(features) == 3
        assert features[0]["feature"] == "checking_status_no_checking"

    def test_returns_503_when_missing(self, client_no_eval: TestClient):
        resp = client_no_eval.get("/evaluation/shap_importance")
        assert resp.status_code == 503


class TestPredictionDistribution:
    def test_returns_histogram_bins(self, client_with_eval: TestClient):
        resp = client_with_eval.get("/evaluation/prediction_distribution")
        assert resp.status_code == 200
        bins = resp.json()["prediction_distribution"]
        assert len(bins) == 3
        assert "bin_start" in bins[0]
        assert "count" in bins[0]

    def test_returns_503_when_missing(self, client_no_eval: TestClient):
        resp = client_no_eval.get("/evaluation/prediction_distribution")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Caching behavior
# ---------------------------------------------------------------------------


class TestEvaluationCaching:
    def test_second_request_uses_cache(self, client_with_eval: TestClient):
        """Two requests should return identical data (from cache)."""
        resp1 = client_with_eval.get("/evaluation/full")
        resp2 = client_with_eval.get("/evaluation/full")
        assert resp1.json() == resp2.json()

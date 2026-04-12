# tests/integration/test_evaluation_api.py
"""Integration tests for the multi-dataset /evaluation/* endpoints.

Writes a temporary evaluation JSON and tests the API serves it
correctly, with dataset_id routing and fallback.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from src.api import dependencies as deps
from src.api.app import create_app
from src.api.auth import verify_api_key
from src.api.routes.evaluation import clear_evaluation_cache

SAMPLE_EVALUATION = {
    "metrics": {"auc": 0.85, "f1": 0.72, "precision": 0.75, "recall": 0.69},
    "confusion_matrix": {
        "matrix": [[120, 30], [18, 82]],
        "labels": ["No Default (0)", "Default (1)"],
    },
    "roc_curve": [
        {"fpr": 0.0, "tpr": 0.0},
        {"fpr": 0.1, "tpr": 0.6},
        {"fpr": 1.0, "tpr": 1.0},
    ],
    "prediction_distribution": [
        {"bin_start": 0.0, "bin_end": 0.05, "count": 50},
        {"bin_start": 0.05, "bin_end": 0.1, "count": 30},
    ],
    "shap_importance": [
        {"feature": "age", "importance": 0.15},
        {"feature": "credit_amount", "importance": 0.12},
    ],
    "dataset_info": {
        "n_samples": 250,
        "n_features": 20,
        "class_distribution": {"0": 150, "1": 100},
    },
}


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear evaluation cache before each test."""
    clear_evaluation_cache()
    yield
    clear_evaluation_cache()


@pytest.fixture
def eval_dir(tmp_path: Path):
    """Create temp eval JSON for german_credit."""
    ds_dir = tmp_path / "german_credit"
    ds_dir.mkdir()
    path = ds_dir / "evaluation_metrics.json"
    path.write_text(json.dumps(SAMPLE_EVALUATION))
    return tmp_path


@pytest.fixture
def client(eval_dir: Path):
    """TestClient with patched data dir and default dataset."""
    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: None

    with (
        patch.object(deps, "_default_dataset_id", "german_credit"),
        patch("src.api.routes.evaluation.settings") as mock_settings,
    ):
        mock_settings.data.dir = eval_dir
        mock_settings.data.evaluation_metrics_path = eval_dir / "legacy_eval.json"

        with TestClient(app) as tc:
            yield tc

    app.dependency_overrides.clear()


class TestFullEvaluation:
    def test_returns_200_with_all_keys(self, client):
        resp = client.get("/evaluation/full?dataset_id=german_credit")
        assert resp.status_code == 200
        body = resp.json()
        assert "metrics" in body
        assert "confusion_matrix" in body
        assert "roc_curve" in body
        assert "shap_importance" in body

    def test_returns_503_when_missing(self, client):
        resp = client.get("/evaluation/full?dataset_id=nonexistent")
        assert resp.status_code == 503


class TestMetricsEndpoint:
    def test_returns_metrics_and_dataset_info(self, client):
        resp = client.get("/evaluation/metrics?dataset_id=german_credit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["metrics"]["auc"] == 0.85
        assert body["dataset_info"]["n_samples"] == 250

    def test_returns_503_when_missing(self, client):
        resp = client.get("/evaluation/metrics?dataset_id=nonexistent")
        assert resp.status_code == 503


class TestConfusionMatrix:
    def test_returns_matrix_and_labels(self, client):
        resp = client.get("/evaluation/confusion_matrix?dataset_id=german_credit")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["matrix"]) == 2
        assert len(body["labels"]) == 2

    def test_returns_503_when_missing(self, client):
        resp = client.get("/evaluation/confusion_matrix?dataset_id=nonexistent")
        assert resp.status_code == 503


class TestRocCurve:
    def test_returns_curve_and_auc(self, client):
        resp = client.get("/evaluation/roc_curve?dataset_id=german_credit")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["roc_curve"]) == 3
        assert body["auc"] == 0.85

    def test_returns_503_when_missing(self, client):
        resp = client.get("/evaluation/roc_curve?dataset_id=nonexistent")
        assert resp.status_code == 503


class TestShapImportance:
    def test_returns_sorted_features(self, client):
        resp = client.get("/evaluation/shap_importance?dataset_id=german_credit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["shap_importance"][0]["feature"] == "age"

    def test_returns_503_when_missing(self, client):
        resp = client.get("/evaluation/shap_importance?dataset_id=nonexistent")
        assert resp.status_code == 503


class TestPredictionDistribution:
    def test_returns_histogram_bins(self, client):
        resp = client.get("/evaluation/prediction_distribution?dataset_id=german_credit")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["prediction_distribution"]) == 2

    def test_returns_503_when_missing(self, client):
        resp = client.get("/evaluation/prediction_distribution?dataset_id=nonexistent")
        assert resp.status_code == 503


class TestDefaultDatasetFallback:
    def test_uses_default_dataset_when_no_param(self, client):
        resp = client.get("/evaluation/metrics")
        assert resp.status_code == 200
        assert resp.json()["metrics"]["auc"] == 0.85

    def test_caching_returns_same_data(self, client):
        resp1 = client.get("/evaluation/full?dataset_id=german_credit")
        resp2 = client.get("/evaluation/full?dataset_id=german_credit")
        assert resp1.json() == resp2.json()

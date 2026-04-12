# tests/integration/test_drift_api.py
"""Integration tests for /monitoring/drift endpoints."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.app import create_app
from src.api.auth import verify_api_key
from src.monitoring.drift import DriftDetector


@pytest.fixture
def mock_detector():
    """A DriftDetector with data ready for analysis."""
    ref_data = np.random.randn(50, 3)
    ref_preds = np.random.rand(50)
    with patch("src.monitoring.drift.settings") as mock_settings:
        mock_settings.drift.detection_threshold = 0.05
        mock_settings.drift.buffer_size = 5
        mock_settings.drift.reference_window_size = 200
        detector = DriftDetector(ref_data, ["f1", "f2", "f3"], ref_preds)

    for _ in range(10):
        detector.record(np.random.randn(3), np.random.rand())

    return detector


@pytest.fixture
def client_with_drift(mock_detector):
    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: None

    with (
        patch.object(deps, "_models", {"german_credit": MagicMock()}),
        patch.object(deps, "_shap_engines", {}),
        patch.object(
            deps, "_drift_detectors", {"german_credit": mock_detector}
        ),
        patch.object(deps, "_default_dataset_id", "german_credit"),
        TestClient(app) as tc,
    ):
        yield tc
    app.dependency_overrides.clear()


@pytest.fixture
def client_no_drift():
    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: None
    with (
        patch.object(deps, "_models", {}),
        patch.object(deps, "_shap_engines", {}),
        patch.object(deps, "_drift_detectors", {}),
        patch.object(deps, "_default_dataset_id", None),
        TestClient(app) as tc,
    ):
        yield tc
    app.dependency_overrides.clear()


class TestGetDriftReport:
    def test_returns_200(self, client_with_drift):
        resp = client_with_drift.get("/monitoring/drift")
        assert resp.status_code == 200
        body = resp.json()
        assert "buffer_count" in body
        assert "report" in body

    def test_returns_503_when_no_detector(self, client_no_drift):
        resp = client_no_drift.get("/monitoring/drift")
        assert resp.status_code == 503


class TestTriggerAnalysis:
    def test_returns_200_with_report(self, client_with_drift):
        resp = client_with_drift.post("/monitoring/drift/analyze")
        assert resp.status_code == 200
        body = resp.json()
        assert "report" in body
        assert body["report"]["features_total"] == 3

    def test_returns_503_when_no_detector(self, client_no_drift):
        resp = client_no_drift.post("/monitoring/drift/analyze")
        assert resp.status_code == 503

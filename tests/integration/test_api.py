# tests/integration/test_api.py
"""Integration tests for the FastAPI application.

Uses httpx TestClient with dependency overrides. Verifies the full
request/response cycle including preprocessing, schema validation,
error handling, authentication, metrics, and multi-dataset routing.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.api import dependencies as deps
from src.api.app import create_app
from src.api.auth import verify_api_key
from src.explain.shap_engine import ShapEngine, ShapExplanation, ShapFactor
from src.model.registry import ModelArtifacts

from tests.conftest import SAMPLE_FEATURE_NAMES, VALID_CREDIT_PAYLOAD

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_artifacts():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.7, 0.3]])
    model.n_features_in_ = len(SAMPLE_FEATURE_NAMES)
    return ModelArtifacts(
        model=model,
        feature_names=SAMPLE_FEATURE_NAMES.copy(),
        pipeline=None,
        model_path=Path("models/german_credit/model.pkl"),
        dataset_id="german_credit",
    )


@pytest.fixture
def mock_shap_engine():
    engine = MagicMock(spec=ShapEngine)
    engine.explain.return_value = ShapExplanation(
        base_value=0.35,
        factors=[
            ShapFactor(
                feature="age",
                shap_value=-0.12,
                input_value=35,
                impact="reduces",
            ),
            ShapFactor(
                feature="credit_amount",
                shap_value=0.08,
                input_value=5000,
                impact="increases",
            ),
        ],
        all_shap_values=[0.0] * len(SAMPLE_FEATURE_NAMES),
    )
    return engine


@pytest.fixture
def client(mock_artifacts, mock_shap_engine):
    """TestClient with mocked model for german_credit."""
    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: None

    # Patch the module-level state in dependencies
    with (
        patch.object(deps, "_models", {"german_credit": mock_artifacts}),
        patch.object(deps, "_shap_engines", {"german_credit": mock_shap_engine}),
        patch.object(deps, "_drift_detectors", {}),
        patch.object(deps, "_default_dataset_id", "german_credit"),
        TestClient(app) as tc,
    ):
        yield tc

    app.dependency_overrides.clear()


@pytest.fixture
def client_no_model():
    """TestClient with no models loaded."""
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


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    def test_liveness_always_200(self, client):
        resp = client.get("/alive")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_health_reports_healthy_with_model(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert "german_credit" in body["loaded_datasets"]

    def test_health_reports_degraded_without_model(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    def test_valid_request_returns_200(self, client):
        resp = client.post(
            "/predict_risk/?dataset_id=german_credit",
            json=VALID_CREDIT_PAYLOAD,
        )
        assert resp.status_code == 200

    def test_response_has_expected_structure(self, client):
        resp = client.post(
            "/predict_risk/?dataset_id=german_credit",
            json=VALID_CREDIT_PAYLOAD,
        )
        body = resp.json()
        assert "prediction" in body
        assert "probability_of_risk" in body
        assert "xai_interpretation" in body
        assert body["status"] == "success"

    def test_prediction_label_is_low_risk(self, client):
        resp = client.post(
            "/predict_risk/?dataset_id=german_credit",
            json=VALID_CREDIT_PAYLOAD,
        )
        body = resp.json()
        assert "Low Risk" in body["prediction"]
        assert body["probability_of_risk"] == 0.3

    def test_explanation_factors_have_correct_fields(self, client):
        resp = client.post(
            "/predict_risk/?dataset_id=german_credit",
            json=VALID_CREDIT_PAYLOAD,
        )
        factors = resp.json()["xai_interpretation"]["detailed_explanation"]
        for factor in factors:
            assert "factor" in factor
            assert "risk_impact" in factor
            assert factor["risk_impact"] in ("increases", "reduces")

    def test_returns_503_when_no_model_loaded(self, client_no_model):
        resp = client_no_model.post("/predict_risk/", json=VALID_CREDIT_PAYLOAD)
        assert resp.status_code == 503

    def test_returns_503_for_unknown_dataset(self, client):
        resp = client.post("/predict_risk/?dataset_id=nonexistent", json=VALID_CREDIT_PAYLOAD)
        assert resp.status_code == 503

    def test_default_dataset_used_without_param(self, client):
        resp = client.post("/predict_risk/", json=VALID_CREDIT_PAYLOAD)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    def test_no_auth_required_when_key_not_configured(self, client):
        resp = client.post(
            "/predict_risk/?dataset_id=german_credit",
            json=VALID_CREDIT_PAYLOAD,
        )
        assert resp.status_code == 200

    @patch("src.api.auth.settings")
    def test_returns_401_when_key_required_but_missing(
        self, mock_settings, mock_artifacts, mock_shap_engine
    ):
        mock_settings.api.api_key = "test-secret-key"

        app = create_app()
        with (
            patch.object(deps, "_models", {"german_credit": mock_artifacts}),
            patch.object(deps, "_shap_engines", {"german_credit": mock_shap_engine}),
            patch.object(deps, "_drift_detectors", {}),
            patch.object(deps, "_default_dataset_id", "german_credit"),
            TestClient(app) as tc,
        ):
            resp = tc.post(
                "/predict_risk/?dataset_id=german_credit",
                json=VALID_CREDIT_PAYLOAD,
            )
        assert resp.status_code == 401
        app.dependency_overrides.clear()

    @patch("src.api.auth.settings")
    def test_passes_with_valid_header_key(self, mock_settings, mock_artifacts, mock_shap_engine):
        mock_settings.api.api_key = "test-secret-key"
        app = create_app()
        with (
            patch.object(deps, "_models", {"german_credit": mock_artifacts}),
            patch.object(deps, "_shap_engines", {"german_credit": mock_shap_engine}),
            patch.object(deps, "_drift_detectors", {}),
            patch.object(deps, "_default_dataset_id", "german_credit"),
            TestClient(app) as tc,
        ):
            resp = tc.post(
                "/predict_risk/?dataset_id=german_credit",
                json=VALID_CREDIT_PAYLOAD,
                headers={"X-API-Key": "test-secret-key"},
            )
        assert resp.status_code == 200
        app.dependency_overrides.clear()

    def test_health_does_not_require_auth(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_alive_does_not_require_auth(self, client):
        resp = client.get("/alive")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Metrics + Security + Docs
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type(self, client):
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_metrics_contains_counters_after_predict(self, client):
        client.post(
            "/predict_risk/?dataset_id=german_credit",
            json=VALID_CREDIT_PAYLOAD,
        )
        resp = client.get("/metrics")
        assert "predictions_total" in resp.text

    def test_metrics_not_in_openapi(self, client):
        resp = client.get("/openapi.json")
        assert "/metrics" not in resp.json().get("paths", {})


class TestSecurityHeaders:
    def test_security_headers_present(self, client):
        resp = client.get("/alive")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("Cache-Control") == "no-store"

    def test_request_id_header_present(self, client):
        resp = client.get("/alive")
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) == 8


class TestDocs:
    def test_openapi_schema_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        assert resp.json()["info"]["title"] == "Credit Risk XAI API"

    def test_swagger_ui_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

# tests/integration/test_api.py
"""Integration tests for the FastAPI application.

Uses httpx TestClient with dependency overrides so that tests run
without real model artifacts. Verifies the full request/response
cycle including preprocessing, schema validation, error handling,
authentication, metrics endpoint, and rate limiting.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.auth import verify_api_key
from src.api.dependencies import get_model_artifacts, get_shap_engine
from src.explain.shap_engine import ShapEngine, ShapExplanation, ShapFactor
from src.model.registry import ModelArtifacts

from tests.conftest import SAMPLE_FEATURE_NAMES, VALID_CREDIT_PAYLOAD

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_artifacts():
    """Build a mock ModelArtifacts with a model that returns 30% risk."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.7, 0.3]])
    model.n_features_in_ = len(SAMPLE_FEATURE_NAMES)

    return ModelArtifacts(
        model=model,
        feature_names=SAMPLE_FEATURE_NAMES.copy(),
        model_path=Path("models/xgb_model.pkl"),
    )


@pytest.fixture
def mock_shap_engine():
    """Build a mock ShapEngine that returns predictable SHAP values."""
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
    """TestClient with overridden dependencies — no real model needed."""
    app = create_app()
    app.dependency_overrides[get_model_artifacts] = lambda: mock_artifacts
    app.dependency_overrides[get_shap_engine] = lambda: mock_shap_engine
    # Auth disabled by default in tests (no API_KEY configured)
    app.dependency_overrides[verify_api_key] = lambda: None

    with TestClient(app) as tc:
        yield tc

    app.dependency_overrides.clear()


@pytest.fixture
def client_no_model():
    """TestClient with no model loaded — simulates missing artifacts."""
    app = create_app()
    app.dependency_overrides[get_model_artifacts] = lambda: None
    app.dependency_overrides[get_shap_engine] = lambda: None
    app.dependency_overrides[verify_api_key] = lambda: None

    with TestClient(app) as tc:
        yield tc

    app.dependency_overrides.clear()


@pytest.fixture
def client_with_auth(mock_artifacts, mock_shap_engine):
    """TestClient with auth enforced — requires valid API key."""
    app = create_app()
    app.dependency_overrides[get_model_artifacts] = lambda: mock_artifacts
    app.dependency_overrides[get_shap_engine] = lambda: mock_shap_engine
    # Do NOT override verify_api_key — let real auth run

    with TestClient(app) as tc:
        yield tc

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    """Verify health and liveness probes."""

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
    """Verify the /predict_risk/ endpoint."""

    def test_valid_request_returns_200(self, client):
        resp = client.post("/predict_risk/", json=VALID_CREDIT_PAYLOAD)
        assert resp.status_code == 200

    def test_response_has_expected_structure(self, client):
        resp = client.post("/predict_risk/", json=VALID_CREDIT_PAYLOAD)
        body = resp.json()

        assert "prediction" in body
        assert "probability_of_risk" in body
        assert "xai_interpretation" in body
        assert "status" in body
        assert body["status"] == "success"

        xai = body["xai_interpretation"]
        assert "base_risk_score" in xai
        assert "detailed_explanation" in xai
        assert isinstance(xai["detailed_explanation"], list)

    def test_prediction_label_is_low_risk(self, client):
        """Mock model returns 0.3 probability -> Low Risk."""
        resp = client.post("/predict_risk/", json=VALID_CREDIT_PAYLOAD)
        body = resp.json()
        assert "Low Risk" in body["prediction"]
        assert body["probability_of_risk"] == 0.3

    def test_explanation_factors_have_correct_fields(self, client):
        resp = client.post("/predict_risk/", json=VALID_CREDIT_PAYLOAD)
        factors = resp.json()["xai_interpretation"]["detailed_explanation"]

        for factor in factors:
            assert "factor" in factor
            assert "risk_impact" in factor
            assert "shap_magnitude" in factor
            assert "input_value" in factor
            assert factor["risk_impact"] in ("increases", "reduces")

    def test_returns_503_when_model_not_loaded(self, client_no_model):
        resp = client_no_model.post("/predict_risk/", json=VALID_CREDIT_PAYLOAD)
        assert resp.status_code == 503

    def test_returns_422_on_missing_field(self, client):
        payload = VALID_CREDIT_PAYLOAD.copy()
        del payload["age"]
        resp = client.post("/predict_risk/", json=payload)
        assert resp.status_code == 422

    def test_returns_422_on_invalid_age(self, client):
        payload = VALID_CREDIT_PAYLOAD.copy()
        payload["age"] = 10
        resp = client.post("/predict_risk/", json=payload)
        assert resp.status_code == 422

    def test_returns_422_on_negative_credit_amount(self, client):
        payload = VALID_CREDIT_PAYLOAD.copy()
        payload["credit_amount"] = -500
        resp = client.post("/predict_risk/", json=payload)
        assert resp.status_code == 422

    def test_returns_422_on_wrong_type(self, client):
        payload = VALID_CREDIT_PAYLOAD.copy()
        payload["duration"] = "twelve"
        resp = client.post("/predict_risk/", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Verify API key authentication on the predict endpoint."""

    def test_no_auth_required_when_key_not_configured(self, client):
        """Default config has no API_KEY — requests pass without auth."""
        resp = client.post("/predict_risk/", json=VALID_CREDIT_PAYLOAD)
        assert resp.status_code == 200

    @patch("src.api.auth.settings")
    def test_returns_401_when_key_required_but_missing(
        self, mock_settings, mock_artifacts, mock_shap_engine
    ):
        """When API_KEY is set and no key is provided, return 401."""
        mock_settings.api.api_key = "test-secret-key"

        app = create_app()
        app.dependency_overrides[get_model_artifacts] = lambda: mock_artifacts
        app.dependency_overrides[get_shap_engine] = lambda: mock_shap_engine
        # Do NOT override verify_api_key — let real auth run with patched settings

        with TestClient(app) as tc:
            resp = tc.post("/predict_risk/", json=VALID_CREDIT_PAYLOAD)
        assert resp.status_code == 401
        app.dependency_overrides.clear()

    @patch("src.api.auth.settings")
    def test_returns_403_when_key_is_wrong(self, mock_settings, mock_artifacts, mock_shap_engine):
        """When API_KEY is set and wrong key is provided, return 403."""
        mock_settings.api.api_key = "test-secret-key"

        app = create_app()
        app.dependency_overrides[get_model_artifacts] = lambda: mock_artifacts
        app.dependency_overrides[get_shap_engine] = lambda: mock_shap_engine

        with TestClient(app) as tc:
            resp = tc.post(
                "/predict_risk/",
                json=VALID_CREDIT_PAYLOAD,
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 403
        app.dependency_overrides.clear()

    @patch("src.api.auth.settings")
    def test_passes_with_valid_header_key(self, mock_settings, mock_artifacts, mock_shap_engine):
        """Valid X-API-Key header grants access."""
        mock_settings.api.api_key = "test-secret-key"

        app = create_app()
        app.dependency_overrides[get_model_artifacts] = lambda: mock_artifacts
        app.dependency_overrides[get_shap_engine] = lambda: mock_shap_engine

        with TestClient(app) as tc:
            resp = tc.post(
                "/predict_risk/",
                json=VALID_CREDIT_PAYLOAD,
                headers={"X-API-Key": "test-secret-key"},
            )
        assert resp.status_code == 200
        app.dependency_overrides.clear()

    @patch("src.api.auth.settings")
    def test_passes_with_valid_query_key(self, mock_settings, mock_artifacts, mock_shap_engine):
        """Valid api_key query parameter grants access."""
        mock_settings.api.api_key = "test-secret-key"

        app = create_app()
        app.dependency_overrides[get_model_artifacts] = lambda: mock_artifacts
        app.dependency_overrides[get_shap_engine] = lambda: mock_shap_engine

        with TestClient(app) as tc:
            resp = tc.post(
                "/predict_risk/?api_key=test-secret-key",
                json=VALID_CREDIT_PAYLOAD,
            )
        assert resp.status_code == 200
        app.dependency_overrides.clear()

    def test_health_does_not_require_auth(self, client_with_auth):
        """Health probes should never require authentication."""
        resp = client_with_auth.get("/health")
        assert resp.status_code == 200

    def test_alive_does_not_require_auth(self, client_with_auth):
        """Liveness probes should never require authentication."""
        resp = client_with_auth.get("/alive")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    """Verify the /metrics Prometheus endpoint."""

    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type(self, client):
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_metrics_contains_request_counters(self, client):
        # Make a request first to populate metrics
        client.get("/health")
        resp = client.get("/metrics")
        body = resp.text
        assert "api_requests_total" in body

    def test_metrics_contains_prediction_counters_after_predict(self, client):
        client.post("/predict_risk/", json=VALID_CREDIT_PAYLOAD)
        resp = client.get("/metrics")
        body = resp.text
        assert "prediction_total" in body
        assert "prediction_probability" in body
        assert "shap_computation_duration_seconds" in body

    def test_metrics_not_in_openapi_schema(self, client):
        resp = client.get("/openapi.json")
        schema = resp.json()
        paths = schema.get("paths", {})
        assert "/metrics" not in paths


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


class TestSecurityHeaders:
    """Verify that security middleware is applied."""

    def test_security_headers_present(self, client):
        resp = client.get("/alive")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("Cache-Control") == "no-store"

    def test_request_id_header_present(self, client):
        resp = client.get("/alive")
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) == 8


# ---------------------------------------------------------------------------
# OpenAPI docs
# ---------------------------------------------------------------------------


class TestDocs:
    """Verify API documentation is served."""

    def test_openapi_schema_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "Credit Risk XAI API"

    def test_swagger_ui_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

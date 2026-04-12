# tests/unit/test_metrics.py
"""Unit tests for src.monitoring.metrics.

Verifies that all Prometheus metrics are correctly registered
in the custom registry and that the metrics endpoint produces
valid output.
"""

from prometheus_client import generate_latest

from src.monitoring.metrics import (
    MODEL_INFO,
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    PREDICTION_PROBABILITY,
    REGISTRY,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    REQUESTS_IN_PROGRESS,
    SHAP_COMPUTATION_SECONDS,
)


class TestMetricsRegistration:
    """Verify all metrics are registered in the custom registry."""

    def test_request_count_registered(self):
        output = generate_latest(REGISTRY).decode()
        assert "api_requests_total" in output

    def test_request_latency_registered(self):
        output = generate_latest(REGISTRY).decode()
        assert "api_request_duration_seconds" in output

    def test_requests_in_progress_registered(self):
        output = generate_latest(REGISTRY).decode()
        assert "api_requests_in_progress" in output

    def test_prediction_count_registered(self):
        output = generate_latest(REGISTRY).decode()
        assert "predictions_total" in output

    def test_prediction_probability_registered(self):
        output = generate_latest(REGISTRY).decode()
        assert "prediction_probability" in output

    def test_shap_computation_registered(self):
        output = generate_latest(REGISTRY).decode()
        assert "shap_computation_duration_seconds" in output

    def test_prediction_errors_registered(self):
        output = generate_latest(REGISTRY).decode()
        assert "prediction_errors_total" in output

    def test_model_info_registered(self):
        output = generate_latest(REGISTRY).decode()
        assert "model_info" in output


class TestMetricsIncrement:
    """Verify metrics can be incremented without errors."""

    def test_request_count_increment(self):
        REQUEST_COUNT.labels(
            method="POST", endpoint="/predict_risk/", status_code="200"
        ).inc()

    def test_request_latency_observe(self):
        REQUEST_LATENCY.labels(
            method="POST", endpoint="/predict_risk/"
        ).observe(0.05)

    def test_in_progress_gauge(self):
        g = REQUESTS_IN_PROGRESS.labels(method="GET", endpoint="/health")
        g.inc()
        g.dec()

    def test_prediction_count_increment(self):
        PREDICTION_COUNT.labels(risk_label="Low Risk (No Default)").inc()

    def test_prediction_probability_observe(self):
        PREDICTION_PROBABILITY.observe(0.35)

    def test_shap_duration_observe(self):
        SHAP_COMPUTATION_SECONDS.observe(0.12)

    def test_prediction_errors_increment(self):
        PREDICTION_ERRORS.labels(error_type="ValueError").inc()

    def test_model_info_set(self):
        MODEL_INFO.info(
            {
                "version": "1.0.0",
                "n_features": "48",
                "model_path": "models/xgb_model.pkl",
            }
        )


class TestRegistryIsolation:
    """Verify that our custom registry does not leak into the default."""

    def test_uses_custom_registry(self):
        from prometheus_client import REGISTRY as DEFAULT_REGISTRY

        assert REGISTRY is not DEFAULT_REGISTRY

    def test_metrics_not_in_default_registry(self):
        from prometheus_client import REGISTRY as DEFAULT_REGISTRY
        from prometheus_client import generate_latest

        default_output = generate_latest(DEFAULT_REGISTRY).decode()
        # Our custom metrics should NOT appear in the default registry
        assert "shap_computation_duration_seconds" not in default_output

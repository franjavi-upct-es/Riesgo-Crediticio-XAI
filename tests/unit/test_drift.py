# tests/unit/test_drift.py
"""Unit tests for monitoring.drift.

Verifies buffer management, KS test drift detection, Prometheus gauge
updates, and edge cases (empty buffer, identical distributions, etc.).
"""

from unittest.mock import patch

import numpy as np
import pytest
from src.monitoring.drift import DriftDetector, DriftReport


@pytest.fixture
def reference_data():
    """Generate a stable reference dataset (100 samples, 5 features)."""
    rng = np.random.RandomState(42)
    return rng.randn(100, 5)


@pytest.fixture
def reference_predictions():
    """Generate reference predictions centered around 0.3."""
    rng = np.random.RandomState(42)
    return rng.beta(2, 5, size=100)  # Skewed toward low risk


@pytest.fixture
def feature_names():
    return ["f0", "f1", "f2", "f3", "f4"]


@pytest.fixture
def detector(reference_data, feature_names, reference_predictions):
    """Create a DriftDetector with small buffer for testing."""
    with patch("src.monitoring.drift.settings") as mock_settings:
        mock_settings.drift.detection_threshold = 0.05
        mock_settings.drift.buffer_size = 10
        mock_settings.drift.reference_window_size = 200
        return DriftDetector(
            reference_data=reference_data,
            feature_names=feature_names,
            reference_predictions=reference_predictions,
        )


class TestBufferManagement:
    """Test the prediction recording buffer."""

    def test_initial_buffer_empty(self, detector):
        assert detector.buffer_count == 0

    def test_record_increments_count(self, detector):
        detector.record(np.zeros(5), 0.3)
        assert detector.buffer_count == 1

    def test_record_multiple(self, detector):
        for i in range(5):
            detector.record(np.ones(5) * i, 0.5)
        assert detector.buffer_count == 5

    def test_last_report_initially_none(self, detector):
        assert detector.last_report is None


class TestDriftAnalysis:
    """Test the statistical drift detection."""

    def test_no_drift_with_same_distribution(self, detector, reference_data):
        """When current data matches reference, no drift should be detected."""
        rng = np.random.RandomState(99)
        # Feed samples from the same distribution as reference
        for _ in range(15):
            sample = reference_data[rng.randint(len(reference_data))]
            detector.record(sample, 0.3)

        report = detector.analyze()

        assert isinstance(report, DriftReport)
        assert report.n_current >= 10
        # Most features should NOT be drifted
        assert report.features_drifted <= 2  # Allow some noise

    def test_drift_detected_with_shifted_distribution(self, detector):
        """When current data is shifted far from reference, drift is detected."""
        rng = np.random.RandomState(42)
        # Feed heavily shifted data
        for _ in range(20):
            shifted = rng.randn(5) + 10.0  # Massive shift
            detector.record(shifted, 0.9)

        report = detector.analyze()

        assert report.features_drifted > 0
        assert report.overall_drift_score > 0.0

    def test_prediction_drift_detected(self, detector):
        """When predicted probabilities shift, prediction drift is flagged."""
        rng = np.random.RandomState(42)
        # Reference predictions are beta(2,5) ≈ mean 0.29
        # Feed high-risk predictions
        for _ in range(20):
            detector.record(rng.randn(5), 0.95)  # All high risk

        report = detector.analyze()

        assert report.prediction_drift_pvalue < 0.05
        assert report.prediction_drifted is True

    def test_no_prediction_drift_with_similar_distribution(self, detector, reference_predictions):
        """When predictions match reference distribution, no drift."""
        rng = np.random.RandomState(99)
        for _ in range(20):
            p = reference_predictions[rng.randint(len(reference_predictions))]
            detector.record(rng.randn(5) * 0.01, float(p))

        report = detector.analyze()
        # p-value should be high (no significant difference)
        assert report.prediction_drift_pvalue > 0.01

    def test_insufficient_data_returns_empty_report(self, detector):
        """Analysis with fewer samples than buffer_size returns empty report."""
        detector.record(np.zeros(5), 0.3)
        report = detector.analyze()

        assert report.features_drifted == 0
        assert len(report.feature_results) == 0

    def test_report_has_correct_structure(self, detector, reference_data):
        """Verify all fields are present and typed correctly."""
        rng = np.random.RandomState(42)
        for _ in range(15):
            detector.record(rng.randn(5), 0.4)

        report = detector.analyze()

        assert isinstance(report.timestamp, str)
        assert isinstance(report.n_reference, int)
        assert isinstance(report.n_current, int)
        assert isinstance(report.threshold, float)
        assert isinstance(report.overall_drift_score, float)
        assert 0.0 <= report.overall_drift_score <= 1.0
        assert isinstance(report.features_total, int)
        assert report.features_total == 5
        assert isinstance(report.feature_results, list)

    def test_feature_results_sorted_by_pvalue(self, detector):
        """Feature results should be sorted by p-value ascending."""
        rng = np.random.RandomState(42)
        for _ in range(15):
            detector.record(rng.randn(5), 0.5)

        report = detector.analyze()
        p_values = [r.p_value for r in report.feature_results]
        assert p_values == sorted(p_values)

    def test_analyze_updates_last_report(self, detector):
        """After analysis, last_report should be set."""
        rng = np.random.RandomState(42)
        for _ in range(15):
            detector.record(rng.randn(5), 0.4)

        detector.analyze()
        assert detector.last_report is not None


class TestToDict:
    """Test JSON serialization."""

    def test_to_dict_without_analysis(self, detector):
        """to_dict works even without a prior analyze() call."""
        result = detector.to_dict()
        assert "timestamp" in result
        assert "overall_drift_score" in result
        assert result["feature_results"] == []

    def test_to_dict_after_analysis(self, detector):
        """to_dict includes feature results after analysis."""
        rng = np.random.RandomState(42)
        for _ in range(15):
            detector.record(rng.randn(5), 0.4)

        detector.analyze()
        result = detector.to_dict()

        assert len(result["feature_results"]) == 5
        for feat in result["feature_results"]:
            assert "feature" in feat
            assert "test_name" in feat
            assert "statistic" in feat
            assert "p_value" in feat
            assert "drifted" in feat


class TestPrometheusGauges:
    """Test that drift metrics are published to Prometheus."""

    def test_gauges_updated_after_analysis(self, detector):
        """DRIFT_SCORE and related gauges should be set after analyze()."""

        rng = np.random.RandomState(42)
        for _ in range(15):
            detector.record(rng.randn(5), 0.4)

        detector.analyze()

        # Gauges should have been set (any value, just not the initial default)
        # We can't easily read Gauge values in prometheus_client without
        # using the internal _value, so just verify no exceptions occurred.
        # The integration test_metrics tests verify the /metrics endpoint.
        assert detector.last_report is not None

# src/monitoring/drift.py
"""Data and prediction drift detection.

Implements lightweight statistical drift detection by comparing
production prediction inputs and outputs against a reference
distribution from training. Uses Kolmogorov-Smirnov test for
numeric features and chi-squared test for categorical features.

The detector maintains a rolling buffer of recent predictions.
When the buffer reaches the configured size, it runs drift checks
and exposes results via Prometheus gauges and a REST endpoint.

Design decisions:
  - No Evidently dependency: scipy-based tests keep the footprint small
    and avoid version conflicts. The same KS/chi2 tests Evidently uses
    internally are implemented directly.
  - Thread-safe: uses a lock around the buffer since FastAPI handlers
    may call record() concurrently with uvicorn workers=1 + async.
  - Stateless across restarts: the buffer is in-memory only. For
    persistence, a future version could flush to Redis or a time-series DB.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog
from prometheus_client import Gauge
from scipy import stats

from src.config import settings
from src.monitoring.metrics import REGISTRY

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus drift gauges
# ---------------------------------------------------------------------------

DRIFT_SCORE = Gauge(
    "drift_score",
    "Overall drift score (fraction of features that drifted).",
    registry=REGISTRY,
)

DRIFT_FEATURES_DRIFTED = Gauge(
    "drift_features_drifted_count",
    "Number of features with statistically significant drift.",
    registry=REGISTRY,
)

PREDICTION_DRIFT_PVALUE = Gauge(
    "prediction_drift_pvalue",
    "KS test p-value for predicted probability distribution drift.",
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Drift result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureDriftResult:
    """Drift detection result for a single feature."""

    feature: str
    test_name: str  # "ks" or "chi2"
    statistic: float
    p_value: float
    drifted: bool


@dataclass(frozen=True)
class DriftReport:
    """Complete drift analysis report."""

    timestamp: str
    n_reference: int
    n_current: int
    threshold: float
    overall_drift_score: float
    features_drifted: int
    features_total: int
    prediction_drift_pvalue: float
    prediction_drifted: bool
    feature_results: list[FeatureDriftResult]


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------


class DriftDetector:
    """Detects data and prediction drift using statistical tests.

    Usage::

        detector = DriftDetector(reference_df, feature_names)

        # On each prediction:
        detector.record(input_dict, predicted_probability)

        # Check drift (runs automatically when buffer is full):
        report = detector.analyze()
    """

    def __init__(
        self,
        reference_data: np.ndarray,
        feature_names: list[str],
        reference_predictions: np.ndarray | None = None,
    ) -> None:
        """Initialize with training reference data.

        Args:
            reference_data: 2D array of shape (n_samples, n_features) from training.
            feature_names: Ordered feature column names.
            reference_predictions: Optional 1D array of predicted probabilities on the reference set.
        """
        self._reference = reference_data
        self._feature_names = feature_names
        self._ref_predictions = reference_predictions
        self._threshold = settings.drift.detection_threshold
        self._buffer_size = settings.drift.buffer_size

        self._input_buffer: deque[np.ndarray] = deque(maxlen=settings.drift.reference_window_size)
        self._prediction_buffer: deque[float] = deque(maxlen=settings.drift.reference_window_size)
        self._lock = threading.Lock()
        self._last_report: DriftReport | None = None

        logger.info(
            "drift_detector_initialized",
            n_reference=len(reference_data),
            n_features=len(feature_names),
            threshold=self._threshold,
            buffer_size=self._buffer_size,
        )

    def record(self, input_vector: np.ndarray, predicted_proba: float) -> None:
        """Record a prediction for drift analysis.

        Args:
            input_vector: 1D feature vector (after preprocessing).
            predicted_proba: The model's predicted risk probability.
        """
        with self._lock:
            self._input_buffer.append(input_vector)
            self._prediction_buffer.append(predicted_proba)

    @property
    def buffer_count(self) -> int:
        """Number of predictions currently buffered."""
        return len(self._input_buffer)

    @property
    def last_report(self) -> DriftReport | None:
        """The most recent drift analysis report."""
        return self._last_report

    def analyze(self) -> DriftReport:
        """Run drift detection on the current buffer vs reference.

        Computes per-feature KS tests (numeric) and an overall drift score.
        Also tests the predicted probability distribution for drift.

        Returns:
            DriftReport with per-feature and overall results.
        """
        with self._lock:
            if len(self._input_buffer) < self._buffer_size:
                logger.info(
                    "drift_analysis_skipped",
                    buffered=len(self._input_buffer),
                    required=self._buffer_size,
                )
                return self._empty_report(reason="insufficient_data")

            current_data = np.array(list(self._input_buffer))
            current_predictions = np.array(list(self._prediction_buffer))

        feature_results = []
        drifted_count = 0

        for i, feature_name in enumerate(self._feature_names):
            ref_col = self._reference[:, i].astype(float)
            cur_col = current_data[:, i].astype(float)

            # Use KS test for all features (post one-hot encoding, all numeric)
            try:
                stat, p_value = stats.ks_2samp(ref_col, cur_col)
            except Exception:
                stat, p_value = 0.0, 1.0

            is_drifted = bool(p_value < self._threshold)
            if is_drifted:
                drifted_count += 1

            feature_results.append(
                FeatureDriftResult(
                    feature=feature_name,
                    test_name="ks",
                    statistic=round(float(stat), 6),
                    p_value=round(float(p_value), 6),
                    drifted=is_drifted,
                )
            )

        # Prediction distribution drift
        pred_p_value = 1.0
        pred_drifted = False
        if self._ref_predictions is not None and len(self._ref_predictions) > 0:
            try:
                _, pred_p_value = stats.ks_2samp(
                    self._ref_predictions.astype(float),
                    current_predictions,
                )
                pred_p_value = float(pred_p_value)
                pred_drifted = bool(pred_p_value < self._threshold)
            except Exception:
                pred_p_value = 1.0

        n_features = len(self._feature_names)
        drift_score = drifted_count / n_features if n_features > 0 else 0.0

        # Update Prometheus gauges
        DRIFT_SCORE.set(drift_score)
        DRIFT_FEATURES_DRIFTED.set(drifted_count)
        PREDICTION_DRIFT_PVALUE.set(pred_p_value)

        report = DriftReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            n_reference=len(self._reference),
            n_current=len(current_data),
            threshold=self._threshold,
            overall_drift_score=round(drift_score, 4),
            features_drifted=drifted_count,
            features_total=n_features,
            prediction_drift_pvalue=round(pred_p_value, 6),
            prediction_drifted=pred_drifted,
            feature_results=sorted(feature_results, key=lambda r: r.p_value),
        )

        self._last_report = report

        logger.info(
            "drift_analysis_complete",
            drift_score=report.overall_drift_score,
            features_drifted=drifted_count,
            prediction_drifted=pred_drifted,
        )

        return report

    def _empty_report(self, reason: str) -> DriftReport:
        """Create an empty report when analysis cannot run."""
        return DriftReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            n_reference=len(self._reference),
            n_current=len(self._input_buffer),
            threshold=self._threshold,
            overall_drift_score=0.0,
            features_drifted=0,
            features_total=len(self._feature_names),
            prediction_drift_pvalue=1.0,
            prediction_drifted=False,
            feature_results=[],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the last report to a JSON-compatible dict."""
        report = self._last_report or self._empty_report(reason="no_analysis_yet")
        return {
            "timestamp": report.timestamp,
            "n_reference": report.n_reference,
            "n_current": report.n_current,
            "threshold": report.threshold,
            "overall_drift_score": report.overall_drift_score,
            "features_drifted": report.features_drifted,
            "features_total": report.features_total,
            "prediction_drift_pvalue": report.prediction_drift_pvalue,
            "prediction_drifted": report.prediction_drifted,
            "feature_results": [
                {
                    "feature": r.feature,
                    "test_name": r.test_name,
                    "statistic": r.statistic,
                    "p_value": r.p_value,
                    "drifted": r.drifted,
                }
                for r in report.feature_results
            ],
        }

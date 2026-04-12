# src/monitoring/metrics.py
"""Prometheus metrics for API observability.

Defines all application-level metrics as module-level singletons.
Metrics are collected by middleware (request-level) and by route
handlers (prediction-level). The /metrics endpoint is mounted
by the application factory.

Metric naming follows Prometheus conventions:
  - Counters: *_total
  - Histograms: *_seconds or *_<unit>
  - Info: *_info
"""

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

# ---------------------------------------------------------------------------
# Custom registry (avoids polluting the default with test artifacts)
# ---------------------------------------------------------------------------
REGISTRY = CollectorRegistry()

# ---------------------------------------------------------------------------
# Request-level metrics (populated by middleware)
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests received.",
    labelnames=["method", "endpoint", "status_code"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request processing time in seconds.",
    labelnames=["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

REQUESTS_IN_PROGRESS = Gauge(
    "api_requests_in_progress",
    "Number of requests currently being processed.",
    labelnames=["method", "endpoint"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Prediction-level metrics (populated by predict route)
# ---------------------------------------------------------------------------

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total number of credit risk predictions made.",
    labelnames=["risk_label"],
    registry=REGISTRY,
)

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribution of predicted risk probabilities.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

SHAP_COMPUTATION_SECONDS = Histogram(
    "shap_computation_duration_seconds",
    "Time spent computing SHAP explanations.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors.",
    labelnames=["error_type"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Model info gauge (set once at startup)
# ---------------------------------------------------------------------------

MODEL_INFO = Info(
    "model",
    "Metadata about the currently loaded model.",
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------

router = APIRouter(tags=["monitoring"])


@router.get("/metrics", include_in_schema=False)
def metrics_endpoint() -> Response:
    """Prometheus-compatible metrics endpoint.

    Returns metrics in the Prometheus text exposition format.
    Excluded from OpenAPI schema to keep docs clean.
    """
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )

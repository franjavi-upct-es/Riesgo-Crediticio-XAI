# src/api/routes/drift.py
"""Drift monitoring endpoints.

Exposes drift analysis results via REST. The detector accumulates
prediction inputs in a rolling buffer and runs KS tests against
the training reference when analyzed.
"""

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException

from src.api.dependencies import get_drift_detector

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/drift")
def get_drift_report() -> dict[str, Any]:
    """Return the most recent drift analysis report.

    If no analysis has been run yet, returns an empty report
    with current buffer status.
    """
    detector = get_drift_detector()
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Drift detector not initialized. Model artifacts may not be loaded.",
        )

    return {
        "buffer_count": detector.buffer_count,
        "buffer_size_required": detector._buffer_size,
        "report": detector.to_dict(),
    }


@router.post("/drift/analyze")
def trigger_drift_analysis() -> dict[str, Any]:
    """Manually trigger drift analysis on the current buffer.

    Returns the fresh drift report. Requires at least `buffer_size`
    predictions to have been recorded.
    """
    detector = get_drift_detector()
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Drift detector not initialized.",
        )

    detector.analyze()

    return {
        "buffer_count": detector.buffer_count,
        "report": detector.to_dict(),
    }

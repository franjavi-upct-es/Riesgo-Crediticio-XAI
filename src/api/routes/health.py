# src/api/routes/health.py
"""Health check endpoints for liveness and readiness probes.

These are essential for Kubernetes deployments and load balancer
health checks. The readiness probe verifies that model artifacts
are loaded and the system can serve predictions.
"""

from fastapi import APIRouter, Depends

from src import __version__
from src.api.dependencies import get_model_artifacts, get_shap_engine
from src.api.schemas import HealthResponse
from src.model.registry import ModelArtifacts

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(
    artifacts: ModelArtifacts | None = Depends(get_model_artifacts),
    engine: object | None = Depends(get_shap_engine),
) -> HealthResponse:
    """Readiness probe: reports whether the API can serve predictions.

    Returns 200 with status="healthy" if model and explainer are loaded,
    or status="degraded" if they are missing.
    """
    model_ok = artifacts is not None and engine is not None

    return HealthResponse(
        status="healthy" if model_ok else "degraded",
        model_loaded=model_ok,
        version=__version__,
    )


@router.get("/alive")
def liveness() -> dict[str, str]:
    """Liveness probe: confirms the process is running.

    Always returns 200. Does not check dependencies — that is
    the readiness probe's job.
    """
    return {"status": "alive"}

# src/api/routes/health.py
"""Health check endpoints — multi-dataset aware."""

from fastapi import APIRouter

from src import __version__
from src.api.dependencies import get_loaded_datasets
from src.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Readiness probe: reports whether any models are loaded."""
    loaded = get_loaded_datasets()
    model_ok = len(loaded) > 0

    return HealthResponse(
        status="healthy" if model_ok else "degraded",
        model_loaded=model_ok,
        version=__version__,
        loaded_datasets=loaded,
    )


@router.get("/alive")
def liveness() -> dict[str, str]:
    """Liveness probe: confirms the process is running."""
    return {"status": "alive"}

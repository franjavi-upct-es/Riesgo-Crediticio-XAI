# src/api/app.py
"""FastAPI application factory — multi-dataset architecture.

Supports multiple credit risk datasets, each with its own model,
preprocessing pipeline, SHAP engine, and drift detector. Datasets
are discovered from YAML schema files and trained models.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI

from src import __version__
from src.api.dependencies import (
    get_loaded_datasets,
    get_model_artifacts,
    initialize_resources,
    shutdown_resources,
)
from src.api.middleware import register_middleware
from src.api.routes.datasets import router as datasets_router
from src.api.routes.drift import router as drift_router
from src.api.routes.evaluation import router as evaluation_router
from src.api.routes.health import router as health_router
from src.api.routes.predict import router as predict_router
from src.config import settings
from src.monitoring.metrics import MODEL_INFO
from src.monitoring.metrics import router as metrics_router
from src.monitoring.tracing import setup_tracing

logger = structlog.get_logger(__name__)


def configure_logging() -> None:
    """Configure structlog with the appropriate renderer."""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.api.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(settings.api.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: load all models on startup."""
    logger.info("application_starting", version=__version__)
    initialize_resources()

    loaded = get_loaded_datasets()
    if loaded:
        first = get_model_artifacts(loaded[0])
        if first:
            MODEL_INFO.info(
                {
                    "version": __version__,
                    "loaded_datasets": ",".join(loaded),
                    "n_features": str(len(first.feature_names)),
                }
            )

    yield
    shutdown_resources()
    logger.info("application_stopped")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Credit Risk XAI API",
        description="Multi-dataset credit risk prediction with SHAP explanations.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    register_middleware(app)

    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(datasets_router)
    app.include_router(evaluation_router)
    app.include_router(drift_router)
    app.include_router(metrics_router)

    setup_tracing(app)

    return app


app = create_app()


def run_server() -> None:
    """CLI entry point for running the API server."""
    configure_logging()
    logger.info(
        "server_starting",
        host=settings.api.host,
        port=settings.api.port,
        otel_enabled=settings.otel.enabled,
    )

    uvicorn.run(
        "src.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=settings.api.reload,
        log_level=settings.api.log_level,
    )


if __name__ == "__main__":  # pragma: no cover
    run_server()

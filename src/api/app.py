# src/api/app.py
"""FastAPI application factory.

Uses the factory pattern so the app can be configured differently for
testing (override dependencies) vs production. The lifespan context
manager handles resource initialization and cleanup. Integrates
OpenTelemetry tracing when enabled.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI

from src import __version__
from src.api.dependencies import initialize_resources, shutdown_resources
from src.api.middleware import register_middleware
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
            logging.getLevelNamesMapping()[settings.api.log_level.upper()]
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: load model on startup, release on shutdown."""
    logger.info("application_starting", version=__version__)
    initialize_resources()

    # Publish model metadata to Prometheus
    from src.api.dependencies import get_model_artifacts

    artifacts = get_model_artifacts()
    if artifacts is not None:
        MODEL_INFO.info(
            {
                "version": __version__,
                "n_features": str(len(artifacts.feature_names)),
                "model_path": str(artifacts.model_path),
            }
        )

    yield
    shutdown_resources()
    logger.info("application_stopped")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Returns:
        A fully configured FastAPI instance with routes, middleware,
        authentication, rate limiting, Prometheus metrics, drift
        detection, and OpenTelemetry tracing.
    """
    app = FastAPI(
        title="Credit Risk XAI API",
        description="Production-grade credit risk prediction with SHAP explanations.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Register middleware stack
    register_middleware(app)

    # Mount route modules
    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(evaluation_router)
    app.include_router(drift_router)
    app.include_router(metrics_router)

    # OpenTelemetry auto-instrumentation (no-op when disabled)
    setup_tracing(app)

    return app


# Module-level app instance for uvicorn CLI
app = create_app()


def run_server() -> None:
    """CLI entry point for running the API server."""
    configure_logging()

    logger.info(
        "server_starting",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        auth_enabled=settings.api.api_key is not None,
        rate_limit=settings.api.rate_limit,
        log_format=settings.api.log_format,
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


if __name__ == "__main__":
    run_server()

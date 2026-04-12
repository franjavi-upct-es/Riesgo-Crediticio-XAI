# src/api/middleware.py
"""API middleware stack.

Applies CORS, request-ID injection, security headers, Prometheus
metrics collection, and rate limiting to the FastAPI application.

Middleware executes in reverse registration order (last registered
runs first on request). Registration order matters.
"""

import time
import uuid

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)

from src.config import settings
from src.monitoring.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    REQUESTS_IN_PROGRESS,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter (module-level singleton, shared by middleware and routes)
# ---------------------------------------------------------------------------

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.api.rate_limit],
    enabled=settings.api.rate_limit_enabled,
)


def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Custom handler for rate limit violations."""
    logger.warning(
        "rate_limit_exceeded",
        client=request.client.host if request.client else "unknown",
        path=request.url.path,
    )
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded: {exc.detail}",
            "retry_after": str(exc.detail),
        },
        headers={"Retry-After": str(exc.detail)},
    )


# ---------------------------------------------------------------------------
# Middleware registration
# ---------------------------------------------------------------------------


def register_middleware(app: FastAPI) -> None:
    """Attach all middleware to the FastAPI app.

    Order (outermost to innermost on request):
      1. CORS
      2. Security headers
      3. Prometheus metrics collection
      4. Request logging with ID
      5. Rate limiting (via slowapi state, not BaseHTTPMiddleware)
    """
    # 1. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # 3. Prometheus metrics
    app.add_middleware(PrometheusMiddleware)

    # 4. Request logging with ID
    app.add_middleware(RequestLoggingMiddleware)

    # 5. Rate limiting (attached to app state, not a BaseHTTPMiddleware)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Middleware implementations
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Injects a unique request ID and logs request/response metadata."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Collects request-level Prometheus metrics.

    Tracks total request count (by method, endpoint, status), latency
    histogram, and in-progress gauge. Skips /metrics and /alive to
    avoid self-referential inflation.
    """

    _SKIP_PATHS: frozenset[str] = frozenset({"/metrics", "/alive"})

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        method = request.method

        if path in self._SKIP_PATHS:
            return await call_next(request)

        endpoint = self._normalize_path(path)

        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:  # pragma: no cover
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code="500").inc()
            REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
            raise

        duration = time.perf_counter() - start

        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(response.status_code),
        ).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()

        return response

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Collapse numeric path segments to prevent high-cardinality labels."""
        parts = path.strip("/").split("/")
        normalized = []
        for part in parts:
            if part.isdigit():
                normalized.append("{id}")
            else:
                normalized.append(part)
        return "/" + "/".join(normalized) if normalized else "/"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds standard security headers to all responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        return response

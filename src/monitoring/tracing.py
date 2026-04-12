# src/monitoring/tracing.py
"""OpenTelemetry distributed tracing.

Configures OTel tracing with OTLP gRPC exporter and auto-instruments
the FastAPI application. When disabled (OTEL_ENABLED=false), all setup
is skipped and no overhead is added.

Traces capture the full request lifecycle including SHAP computation
spans, preprocessing spans, and model inference spans. Custom spans
can be created via the get_tracer() function.

Integration:
  - FastAPI auto-instrumentation adds spans for every HTTP request.
  - Manual spans are added in predict route for preprocessing, inference, SHAP.
  - Traces are exported to an OTLP-compatible collector (Jaeger, Tempo, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from src.config import settings

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = structlog.get_logger(__name__)

_tracer = None


def setup_tracing(app: FastAPI) -> None:
    """Configure OpenTelemetry tracing and instrument the FastAPI app.

    Skips entirely when OTEL_ENABLED is false. Falls back gracefully
    if OTel packages are not installed or the collector is unreachable.

    Args:
        app: The FastAPI application to instrument.
    """
    if not settings.otel.enabled:
        logger.info("otel_tracing_disabled")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": settings.otel.service_name,
                "service.version": "1.0.0",
                "deployment.environment": "production",
            }
        )

        provider = TracerProvider(resource=resource)

        exporter = OTLPSpanExporter(
            endpoint=settings.otel.exporter_endpoint,
            insecure=True,
        )
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)

        # Auto-instrument FastAPI
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="alive,metrics",
        )

        global _tracer
        _tracer = trace.get_tracer(settings.otel.service_name)

        logger.info(
            "otel_tracing_configured",
            service_name=settings.otel.service_name,
            endpoint=settings.otel.exporter_endpoint,
        )

    except ImportError as exc:
        logger.warning("otel_packages_not_installed", error=str(exc))
    except Exception as exc:
        logger.warning("otel_setup_failed", error=str(exc))


def get_tracer() -> Any | None:
    """Return the configured OTel tracer, or None if tracing is disabled.

    Usage::

        tracer = get_tracer()
        if tracer:
            with tracer.start_as_current_span("my_operation") as span:
                span.set_attribute("key", "value")
                do_work()
    """
    return _tracer

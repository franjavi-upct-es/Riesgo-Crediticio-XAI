# tests/unit/test_tracing.py
"""Unit tests for src.monitoring.tracing.

Tests the OTel setup in both enabled and disabled modes, and the
graceful fallback when OTel packages are unavailable.
"""

from unittest.mock import MagicMock, patch

import src.monitoring.tracing as tracing_module


class TestSetupTracingDisabled:
    def test_skips_when_disabled(self):
        app = MagicMock()
        with patch.object(tracing_module, "settings") as mock_settings:
            mock_settings.otel.enabled = False
            tracing_module.setup_tracing(app)
        # No instrumentation calls should happen
        app.assert_not_called()


class TestSetupTracingEnabled:
    @patch("src.monitoring.tracing.settings")
    def test_configures_otel_when_enabled(self, mock_settings):
        mock_settings.otel.enabled = True
        mock_settings.otel.service_name = "test-service"
        mock_settings.otel.exporter_endpoint = "http://localhost:4317"

        app = MagicMock()

        # This will import the real OTel packages (installed in our env)
        tracing_module.setup_tracing(app)

        # Tracer should be set
        assert tracing_module._tracer is not None

        # Clean up global state
        tracing_module._tracer = None

    @patch("src.monitoring.tracing.settings")
    def test_graceful_fallback_on_import_error(self, mock_settings):
        mock_settings.otel.enabled = True

        app = MagicMock()

        # Simulate missing OTel packages
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "opentelemetry" in name:
                raise ImportError("No module named 'opentelemetry'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            tracing_module.setup_tracing(app)

        # Should not raise, tracer stays None

    @patch("src.monitoring.tracing.settings")
    def test_graceful_fallback_on_general_exception(self, mock_settings):
        mock_settings.otel.enabled = True
        mock_settings.otel.service_name = "test"
        mock_settings.otel.exporter_endpoint = "http://localhost:4317"

        app = MagicMock()

        # Force a RuntimeError inside setup
        with patch(
            "opentelemetry.sdk.resources.Resource.create",
            side_effect=RuntimeError("boom"),
        ):
            tracing_module.setup_tracing(app)
        # Should not raise


class TestGetTracer:
    def test_returns_none_when_not_configured(self):
        tracing_module._tracer = None
        assert tracing_module.get_tracer() is None

    def test_returns_tracer_when_set(self):
        mock_tracer = MagicMock()
        tracing_module._tracer = mock_tracer
        assert tracing_module.get_tracer() is mock_tracer
        tracing_module._tracer = None  # clean up

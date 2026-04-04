# src/model/experiment_tracker.py
"""MLflow experiment tracking integration.

Provides a context manager that handles the full MLflow run lifecycle:
experiment creation, parameter logging, metric logging, model artifact
registration, and run cleanup. Falls back gracefully to no-op when
MLflow is disabled.
"""

from __future__ import annotations

import hashlib
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class ExperimentTracker:
    """Tracks training experiments via MLflow.

    When MLflow is disabled (settings.mlflow.enabled = False), all
    methods are no-ops. This allows the training pipeline to run
    identically in environments without MLflow infrastructure.
    """

    def __init__(self, experiment_name: str | None = None) -> None:
        self._enabled = settings.mlflow.enabled
        self._run: Any = None

        if self._enabled:
            try:
                import mlflow

                mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
                mlflow.set_experiment(experiment_name or settings.mlflow.experiment_name)
                self._mlflow = mlflow
                logger.info(
                    "mlflow_initialized",
                    tracking_uri=settings.mlflow.tracking_uri,
                    experiment=experiment_name or settings.mlflow.experiment_name,
                )
            except Exception as exc:
                logger.warning("mlflow_init_failed", error=str(exc))
                self._enabled = False

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator[ExperimentTracker, None, None]:
        """Context manager for an MLflow run.

        Args:
            run_name: Human-readable run identifier.
            tags: Key-value tags for the run.

        Yields:
            self, for chaining log calls.
        """
        if not self._enabled:
            yield self
            return

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = run_name or f"run-{ts}"

        self._run = self._mlflow.start_run(run_name=name)
        if tags:
            self._mlflow.set_tags(tags)

        logger.info("mlflow_run_started", run_name=run_name, run_id=self._run.info.run_id)

        try:
            yield self
        except Exception:
            self._mlflow.set_tag("run_status", "failed")
            raise
        finally:
            self._mlflow.end_run()
            logger.info("mlflow_run_ended", run_id=self._run.info.run_id)
            self._run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dictionary of parameters."""
        if not self._enabled:
            return
        # MLflow requires string values for params
        flat = {k: str(v) for k, v in params.items()}
        self._mlflow.log_params(flat)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log a dictionary of numeric metrics."""
        if not self._enabled:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Log a trained model as an MLflow artifact."""
        if not self._enabled or not settings.mlflow.log_models:
            return
        try:
            self._mlflow.sklearn.log_model(model, artifact_path=artifact_path)  # type: ignore[attr-defined]
            logger.info("mlflow_model_logged", artifact_path=artifact_path)
        except Exception as exc:
            logger.warning("mlflow_model_log_failed", error=str(exc))

    def log_artifact(self, local_path: str | Path) -> None:
        """Log a file as an MLflow artifact."""
        if not self._enabled:
            return
        self._mlflow.log_artifact(str(local_path))

    def log_dict(self, data: dict, filename: str) -> None:
        """Log a dictionary as a JSON artifact."""
        if not self._enabled:
            return
        self._mlflow.log_dict(data, filename)

    def log_dataset_hash(self, data_path: Path) -> None:
        """Compute and log a SHA-256 hash of a data file for traceability."""
        if not self._enabled or not data_path.exists():
            return
        sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
        self._mlflow.log_param("dataset_sha256", sha[:16])
        logger.info("dataset_hash_logged", sha256_prefix=sha[:16])

    @property
    def run_id(self) -> str | None:
        """The current MLflow run ID, or None."""
        if self._run:
            return self._run.info.run_id
        return None

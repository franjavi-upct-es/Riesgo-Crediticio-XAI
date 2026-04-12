# tests/unit/test_experiment_tracker.py
"""Unit tests for src.model.experiment_tracker.

Verifies that the tracker behaves as a no-op when disabled, and
correctly delegates to MLflow when enabled.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.model.experiment_tracker import ExperimentTracker


class TestTrackerDisabled:
    """When mlflow.enabled is False, all methods should be no-ops."""

    @patch("src.model.experiment_tracker.settings")
    def test_disabled_tracker_does_nothing(self, mock_settings):
        mock_settings.mlflow.enabled = False
        tracker = ExperimentTracker()

        # All methods should run without error
        with tracker.start_run(run_name="test"):
            tracker.log_params({"a": 1, "b": "hello"})
            tracker.log_metrics({"auc": 0.85, "f1": 0.78})
            tracker.log_model(MagicMock(), artifact_path="model")
            tracker.log_dict({"key": "value"}, "test.json")

        assert tracker.run_id is None

    @patch("src.model.experiment_tracker.settings")
    def test_disabled_log_artifact(self, mock_settings, tmp_path: Path):
        mock_settings.mlflow.enabled = False
        tracker = ExperimentTracker()
        # Should not raise even with a real file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        tracker.log_artifact(test_file)

    @patch("src.model.experiment_tracker.settings")
    def test_disabled_log_dataset_hash(self, mock_settings, tmp_path: Path):
        mock_settings.mlflow.enabled = False
        tracker = ExperimentTracker()
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2")
        tracker.log_dataset_hash(test_file)


class TestTrackerEnabled:
    """When mlflow.enabled is True, calls should be delegated to MLflow."""

    @patch("src.model.experiment_tracker.settings")
    def test_enabled_tracker_starts_and_ends_run(self, mock_settings):
        mock_settings.mlflow.enabled = True
        mock_settings.mlflow.tracking_uri = "mlruns"
        mock_settings.mlflow.experiment_name = "test"
        mock_settings.mlflow.log_models = True

        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "abc123"
        mock_mlflow.start_run.return_value = mock_run

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker = ExperimentTracker(experiment_name="test-exp")

            with tracker.start_run(run_name="test-run", tags={"env": "test"}):
                assert tracker.run_id == "abc123"

                tracker.log_params({"lr": 0.1})
                mock_mlflow.log_params.assert_called_once()

                tracker.log_metrics({"auc": 0.85})
                mock_mlflow.log_metrics.assert_called_once()

            # Run should be ended
            mock_mlflow.end_run.assert_called_once()

    @patch("src.model.experiment_tracker.settings")
    def test_enabled_tracker_logs_dict(self, mock_settings):
        mock_settings.mlflow.enabled = True
        mock_settings.mlflow.tracking_uri = "mlruns"
        mock_settings.mlflow.experiment_name = "test"
        mock_settings.mlflow.log_models = True

        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "def456"
        mock_mlflow.start_run.return_value = mock_run

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker = ExperimentTracker()

            with tracker.start_run():
                tracker.log_dict({"metrics": {"auc": 0.85}}, "eval.json")
                mock_mlflow.log_dict.assert_called_once_with(
                    {"metrics": {"auc": 0.85}}, "eval.json"
                )

    @patch("src.model.experiment_tracker.settings")
    def test_enabled_tracker_sets_failed_tag_on_exception(self, mock_settings):
        mock_settings.mlflow.enabled = True
        mock_settings.mlflow.tracking_uri = "mlruns"
        mock_settings.mlflow.experiment_name = "test"

        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "fail789"
        mock_mlflow.start_run.return_value = mock_run

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker = ExperimentTracker()

            with pytest.raises(ValueError, match="boom"), tracker.start_run():
                raise ValueError("boom")

            mock_mlflow.set_tag.assert_any_call("run_status", "failed")
            mock_mlflow.end_run.assert_called_once()

    @patch("src.model.experiment_tracker.settings")
    def test_dataset_hash_logged(self, mock_settings, tmp_path: Path):
        mock_settings.mlflow.enabled = True
        mock_settings.mlflow.tracking_uri = "mlruns"
        mock_settings.mlflow.experiment_name = "test"

        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "hash000"
        mock_mlflow.start_run.return_value = mock_run

        data_file = tmp_path / "data.csv"
        data_file.write_text("col1,col2\n1,2\n3,4")

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker = ExperimentTracker()

            with tracker.start_run():
                tracker.log_dataset_hash(data_file)
                mock_mlflow.log_param.assert_called_once()
                call_args = mock_mlflow.log_param.call_args
                assert call_args[0][0] == "dataset_sha256"
                assert len(call_args[0][1]) == 16  # SHA prefix


class TestTrackerInitFailure:
    """When MLflow import or init fails, tracker should degrade gracefully."""

    @patch("src.model.experiment_tracker.settings")
    def test_graceful_degradation_on_import_error(self, mock_settings):
        mock_settings.mlflow.enabled = True
        mock_settings.mlflow.tracking_uri = "mlruns"
        mock_settings.mlflow.experiment_name = "test"

        with patch.dict("sys.modules", {"mlflow": None}):
            # Importing None will cause an error inside __init__
            tracker = ExperimentTracker()
            # Should fall back to disabled mode
            assert not tracker._enabled

            # All methods should still work as no-ops
            with tracker.start_run():
                tracker.log_params({"x": 1})
                tracker.log_metrics({"y": 2.0})

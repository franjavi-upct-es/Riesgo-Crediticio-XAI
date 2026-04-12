# tests/unit/test_dependencies.py
"""Unit tests for src.api.dependencies.

Tests initialization flows, multi-model loading, drift detector setup,
and shutdown.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import src.api.dependencies as deps_module


class TestInitializeResources:
    @patch("src.api.dependencies.ShapEngine")
    @patch("src.api.dependencies.list_trained_models", return_value=[])
    @patch("src.api.dependencies.load_model_artifacts")
    def test_loads_legacy_when_no_trained_models(self, mock_load, mock_list, mock_shap):
        mock_artifacts = MagicMock()
        mock_artifacts.dataset_id = "german_credit"
        mock_artifacts.model = MagicMock()
        mock_load.return_value = mock_artifacts

        deps_module._models.clear()
        deps_module._shap_engines.clear()
        deps_module._drift_detectors.clear()
        deps_module._default_dataset_id = None

        deps_module.initialize_resources()

        assert "german_credit" in deps_module._models
        assert deps_module._default_dataset_id == "german_credit"

        deps_module.shutdown_resources()

    @patch("src.api.dependencies.list_trained_models", return_value=[])
    @patch(
        "src.api.dependencies.load_model_artifacts",
        side_effect=FileNotFoundError,
    )
    def test_handles_no_models_at_all(self, mock_load, mock_list):
        deps_module._models.clear()
        deps_module._default_dataset_id = None

        deps_module.initialize_resources()

        assert len(deps_module._models) == 0
        assert deps_module._default_dataset_id is None

    @patch("src.api.dependencies.list_trained_models", return_value=[])
    @patch(
        "src.api.dependencies.load_model_artifacts",
        side_effect=RuntimeError("bad"),
    )
    def test_handles_initialization_exception(self, mock_load, mock_list):
        deps_module._models.clear()
        deps_module._default_dataset_id = None

        deps_module.initialize_resources()
        assert len(deps_module._models) == 0

    @patch("src.api.dependencies._init_drift_detector")
    @patch("src.api.dependencies.ShapEngine")
    @patch("src.api.dependencies.load_model_artifacts")
    @patch(
        "src.api.dependencies.list_trained_models",
        return_value=["ds_a", "ds_b"],
    )
    def test_loads_multiple_datasets(self, mock_list, mock_load, mock_shap, mock_drift):
        artifacts_a = MagicMock()
        artifacts_a.model = MagicMock()
        artifacts_a.feature_names = ["f1"]
        artifacts_b = MagicMock()
        artifacts_b.model = MagicMock()
        artifacts_b.feature_names = ["f1", "f2"]
        mock_load.side_effect = [artifacts_a, artifacts_b]

        deps_module._models.clear()
        deps_module._shap_engines.clear()
        deps_module._drift_detectors.clear()
        deps_module._default_dataset_id = None

        deps_module.initialize_resources()

        assert "ds_a" in deps_module._models
        assert "ds_b" in deps_module._models
        assert deps_module._default_dataset_id == "ds_a"

        deps_module.shutdown_resources()

    @patch("src.api.dependencies.ShapEngine")
    @patch(
        "src.api.dependencies.load_model_artifacts",
        side_effect=Exception("fail"),
    )
    @patch("src.api.dependencies.list_trained_models", return_value=["bad_ds"])
    def test_skips_failing_dataset(self, mock_list, mock_load, mock_shap):
        deps_module._models.clear()
        deps_module._default_dataset_id = None

        deps_module.initialize_resources()
        assert "bad_ds" not in deps_module._models

        deps_module.shutdown_resources()


class TestInitDriftDetector:
    @patch("src.api.dependencies.settings")
    def test_initializes_from_synthetic_test_set(self, mock_settings, tmp_path):
        ds_dir = tmp_path / "ds_a"
        ds_dir.mkdir()
        df = pd.DataFrame(
            {
                "f1": np.random.randn(20),
                "f2": np.random.randn(20),
                "target": [0] * 10 + [1] * 10,
            }
        )
        df.to_csv(ds_dir / "synthetic_test_set.csv", index=False)

        mock_settings.data.dir = tmp_path
        mock_settings.data.synthetic_test_path = tmp_path / "legacy.csv"

        artifacts = MagicMock()
        artifacts.feature_names = ["f1", "f2"]
        artifacts.model = MagicMock()
        artifacts.model.predict_proba.return_value = np.random.rand(20, 2)

        deps_module._drift_detectors.clear()
        deps_module._init_drift_detector("ds_a", artifacts)
        assert "ds_a" in deps_module._drift_detectors

        deps_module._drift_detectors.clear()

    @patch("src.api.dependencies.settings")
    def test_handles_missing_reference_data(self, mock_settings, tmp_path):
        mock_settings.data.dir = tmp_path
        mock_settings.data.synthetic_test_path = tmp_path / "legacy.csv"

        artifacts = MagicMock()
        artifacts.feature_names = ["f1"]
        deps_module._drift_detectors.clear()
        deps_module._init_drift_detector("ds_x", artifacts)
        assert "ds_x" not in deps_module._drift_detectors

    @patch("src.api.dependencies.settings")
    def test_handles_exception_during_init(self, mock_settings, tmp_path):
        ds_dir = tmp_path / "ds_err"
        ds_dir.mkdir()
        # Write invalid CSV
        (ds_dir / "synthetic_test_set.csv").write_text("bad,data\n")

        mock_settings.data.dir = tmp_path
        mock_settings.data.synthetic_test_path = tmp_path / "legacy.csv"

        artifacts = MagicMock()
        artifacts.feature_names = ["f1"]
        artifacts.model.predict_proba.side_effect = Exception("crash")

        deps_module._drift_detectors.clear()
        deps_module._init_drift_detector("ds_err", artifacts)
        # Should not raise


class TestShutdown:
    def test_clears_all_state(self):
        deps_module._models["test"] = MagicMock()
        deps_module._shap_engines["test"] = MagicMock()
        deps_module._drift_detectors["test"] = MagicMock()
        deps_module._default_dataset_id = "test"

        deps_module.shutdown_resources()

        assert len(deps_module._models) == 0
        assert len(deps_module._shap_engines) == 0
        assert len(deps_module._drift_detectors) == 0
        assert deps_module._default_dataset_id is None


class TestGetters:
    def test_get_model_artifacts_default(self):
        deps_module._models["ds"] = "model"
        deps_module._default_dataset_id = "ds"
        assert deps_module.get_model_artifacts() == "model"
        deps_module.shutdown_resources()

    def test_get_model_artifacts_specific(self):
        deps_module._models["ds"] = "model"
        assert deps_module.get_model_artifacts("ds") == "model"
        deps_module.shutdown_resources()

    def test_get_model_artifacts_none(self):
        deps_module._default_dataset_id = None
        assert deps_module.get_model_artifacts() is None

    def test_get_loaded_datasets(self):
        deps_module._models["a"] = "m"
        deps_module._models["b"] = "m"
        assert deps_module.get_loaded_datasets() == ["a", "b"]
        deps_module.shutdown_resources()

    def test_get_default_dataset_id(self):
        deps_module._default_dataset_id = "test"
        assert deps_module.get_default_dataset_id() == "test"
        deps_module._default_dataset_id = None

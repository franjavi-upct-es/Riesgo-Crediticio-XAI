# tests/unit/test_registry.py
"""Unit tests for model.registry.

Tests artifact loading, validation checks, and proper error handling
when artifacts are missing or corrupted.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.model.registry import ModelArtifacts, load_model_artifacts


class TestModelArtifacts:
    """Test the ModelArtifacts dataclass and its validation."""

    def test_validate_passes_with_consistent_artifacts(self, mock_xgb_model, sample_feature_names):
        artifacts = ModelArtifacts(
            model=mock_xgb_model,
            feature_names=sample_feature_names,
            model_path=Path("models/xgb_model.pkl"),
        )
        # Should not raise
        artifacts.validate()

    def test_validate_fails_on_empty_feature_names(self, mock_xgb_model):
        artifacts = ModelArtifacts(
            model=mock_xgb_model,
            feature_names=[],
            model_path=Path("models/xgb_model.pkl"),
        )
        with pytest.raises(ValueError, match="empty"):
            artifacts.validate()

    def test_validate_fails_on_feature_count_mismatch(self, mock_xgb_model):
        mock_xgb_model.n_features_in_ = 5  # Model expects 5
        artifacts = ModelArtifacts(
            model=mock_xgb_model,
            feature_names=["a", "b", "c"],  # But only 3 names
            model_path=Path("models/xgb_model.pkl"),
        )
        with pytest.raises(ValueError, match="feature"):
            artifacts.validate()

    def test_validate_fails_on_model_without_predict_proba(self, sample_feature_names):
        model = MagicMock(spec=[])  # No predict_proba
        model.n_features_in_ = len(sample_feature_names)
        artifacts = ModelArtifacts(
            model=model,
            feature_names=sample_feature_names,
            model_path=Path("models/xgb_model.pkl"),
        )
        with pytest.raises(ValueError, match="predict_proba"):
            artifacts.validate()

    def test_is_frozen(self, mock_xgb_model, sample_feature_names):
        artifacts = ModelArtifacts(
            model=mock_xgb_model,
            feature_names=sample_feature_names,
            model_path=Path("models/xgb_model.pkl"),
        )
        with pytest.raises(AttributeError):
            artifacts.model = None  # type: ignore[misc]


class TestLoadModelArtifacts:
    """Test the artifact loading function."""

    def test_raises_file_not_found_when_model_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="model"):
            load_model_artifacts(model_dir=tmp_path)

    def test_raises_file_not_found_when_features_missing(self, tmp_path):
        # Create the model file but not features
        (tmp_path / "xgb_model.pkl").touch()
        with pytest.raises(FileNotFoundError, match="feature_names"):
            load_model_artifacts(model_dir=tmp_path)

    @patch("src.model.registry.joblib")
    def test_loads_and_validates(self, mock_joblib, tmp_path, mock_xgb_model, sample_feature_names):
        # Create the artifact files
        (tmp_path / "xgb_model.pkl").touch()
        (tmp_path / "feature_names.pkl").touch()

        mock_joblib.load.side_effect = [mock_xgb_model, sample_feature_names]

        artifacts = load_model_artifacts(model_dir=tmp_path)

        assert artifacts.model is mock_xgb_model
        assert artifacts.feature_names == sample_feature_names
        assert mock_joblib.load.call_count == 2

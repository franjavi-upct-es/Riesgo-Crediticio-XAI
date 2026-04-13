# tests/unit/test_registry.py
"""Unit tests for src.model.registry (multi-dataset)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.model.registry import (
    ModelArtifacts,
    list_trained_models,
    load_model_artifacts,
)


class TestModelArtifacts:
    def test_validate_passes_with_consistent_artifacts(self, mock_xgb_model, sample_feature_names):
        artifacts = ModelArtifacts(
            model=mock_xgb_model,
            feature_names=sample_feature_names,
            pipeline=None,
            model_path=Path("models/german_credit/model.pkl"),
            dataset_id="german_credit",
        )
        artifacts.validate()

    def test_validate_fails_on_empty_feature_names(self, mock_xgb_model):
        artifacts = ModelArtifacts(
            model=mock_xgb_model,
            feature_names=[],
            pipeline=None,
            model_path=Path("models/model.pkl"),
            dataset_id="test",
        )
        with pytest.raises(ValueError, match="empty"):
            artifacts.validate()

    def test_validate_fails_on_feature_count_mismatch(self, mock_xgb_model):
        mock_xgb_model.n_features_in_ = 5
        artifacts = ModelArtifacts(
            model=mock_xgb_model,
            feature_names=["a", "b", "c"],
            pipeline=None,
            model_path=Path("models/model.pkl"),
            dataset_id="test",
        )
        with pytest.raises(ValueError, match="feature"):
            artifacts.validate()

    def test_validate_fails_on_model_without_predict_proba(self, sample_feature_names):
        model = MagicMock(spec=[])
        model.n_features_in_ = len(sample_feature_names)
        artifacts = ModelArtifacts(
            model=model,
            feature_names=sample_feature_names,
            pipeline=None,
            model_path=Path("models/model.pkl"),
            dataset_id="test",
        )
        with pytest.raises(ValueError, match="predict_proba"):
            artifacts.validate()

    def test_is_frozen(self, mock_xgb_model, sample_feature_names):
        artifacts = ModelArtifacts(
            model=mock_xgb_model,
            feature_names=sample_feature_names,
            pipeline=None,
            model_path=Path("models/model.pkl"),
            dataset_id="test",
        )
        with pytest.raises(AttributeError):
            artifacts.model = None

    def test_dataset_id_stored(self, mock_xgb_model, sample_feature_names):
        artifacts = ModelArtifacts(
            model=mock_xgb_model,
            feature_names=sample_feature_names,
            pipeline=None,
            model_path=Path("models/lending_club/model.pkl"),
            dataset_id="lending_club",
        )
        assert artifacts.dataset_id == "lending_club"

    def test_validate_fails_on_invalid_threshold(self, mock_xgb_model, sample_feature_names):
        artifacts = ModelArtifacts(
            model=mock_xgb_model,
            feature_names=sample_feature_names,
            pipeline=None,
            model_path=Path("models/model.pkl"),
            dataset_id="test",
            decision_threshold=1.5,
        )
        with pytest.raises(ValueError, match="decision_threshold"):
            artifacts.validate()


class TestLoadModelArtifacts:
    def test_raises_file_not_found_when_model_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="model"):
            load_model_artifacts(dataset_id="test", model_dir=tmp_path)

    def test_raises_file_not_found_when_features_missing(self, tmp_path):
        ds_dir = tmp_path / "test"
        ds_dir.mkdir()
        (ds_dir / "model.pkl").touch()
        with pytest.raises(FileNotFoundError, match="feature_names"):
            load_model_artifacts(dataset_id="test", model_dir=tmp_path)

    @patch("src.model.registry.joblib")
    def test_loads_and_validates(self, mock_joblib, tmp_path, mock_xgb_model, sample_feature_names):
        ds_dir = tmp_path / "test"
        ds_dir.mkdir()
        (ds_dir / "model.pkl").touch()
        (ds_dir / "feature_names.pkl").touch()

        mock_joblib.load.side_effect = [mock_xgb_model, sample_feature_names]

        artifacts = load_model_artifacts(dataset_id="test", model_dir=tmp_path)
        assert artifacts.model is mock_xgb_model
        assert artifacts.feature_names == sample_feature_names
        assert artifacts.dataset_id == "test"
        assert mock_joblib.load.call_count == 2

    @patch("src.model.registry.joblib")
    def test_loads_decision_threshold(
        self, mock_joblib, tmp_path, mock_xgb_model, sample_feature_names
    ):
        ds_dir = tmp_path / "test"
        ds_dir.mkdir()
        (ds_dir / "model.pkl").touch()
        (ds_dir / "feature_names.pkl").touch()
        (ds_dir / "threshold.json").write_text(json.dumps({"decision_threshold": 0.37}))

        mock_joblib.load.side_effect = [mock_xgb_model, sample_feature_names]

        artifacts = load_model_artifacts(dataset_id="test", model_dir=tmp_path)
        assert artifacts.decision_threshold == 0.37


class TestListTrainedModels:
    def test_empty_directory(self, tmp_path):
        assert list_trained_models(model_dir=tmp_path) == []

    def test_finds_trained_datasets(self, tmp_path):
        (tmp_path / "ds_a").mkdir()
        (tmp_path / "ds_a" / "model.pkl").touch()
        (tmp_path / "ds_b").mkdir()
        (tmp_path / "ds_b" / "model.pkl").touch()
        (tmp_path / "ds_c").mkdir()  # No model.pkl → not listed

        result = list_trained_models(model_dir=tmp_path)
        assert result == ["ds_a", "ds_b"]

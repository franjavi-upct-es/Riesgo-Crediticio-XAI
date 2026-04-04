# tests/unit/test_config.py
"""Unit tests for config.

Verifies default values, path construction, and validation rules
for the centralized settings system.
"""

from pathlib import Path

import pytest
from src.config import Settings


class TestDefaultSettings:
    """Verify that default settings are sensible and consistent."""

    def test_api_defaults(self):
        s = Settings()
        assert s.api.host == "127.0.0.1"
        assert s.api.port == 8000
        assert s.api.workers == 1

    def test_model_path_construction(self):
        s = Settings()
        assert s.model.model_path == Path("models/xgb_model.pkl")
        assert s.model.feature_names_path == Path("models/feature_names.pkl")

    def test_data_path_construction(self):
        s = Settings()
        assert s.data.synthetic_test_path == Path("data/synthetic_test_set.csv")

    def test_training_defaults(self):
        s = Settings()
        assert s.train.test_size == 0.2
        assert s.train.random_state == 42
        assert s.train.xgb_n_estimators == 100
        assert s.train.xgb_learning_rate == 0.1

    def test_shap_defaults(self):
        s = Settings()
        assert s.shap.significance_threshold == 0.001


class TestSettingsValidation:
    """Verify validation rules prevent misconfiguration."""

    def test_rejects_test_size_zero(self):
        with pytest.raises(Exception):
            Settings(train={"test_size": 0.0})  # type: ignore[arg-type]

    def test_rejects_test_size_one(self):
        with pytest.raises(Exception):
            Settings(train={"test_size": 1.0})  # type: ignore[arg-type]

    def test_accepts_valid_test_size(self):
        s = Settings(train={"test_size": 0.3})  # type: ignore[arg-type]
        assert s.train.test_size == 0.3

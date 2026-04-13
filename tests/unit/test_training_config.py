# tests/unit/test_training_config.py
"""Unit tests for src.model.training_config.

Verifies YAML loading, default fallbacks, and hyperparameter
dataclass construction.
"""

import textwrap
from pathlib import Path

import pytest
from src.model.training_config import (
    ModelHyperparams,
    TrainingConfig,
    load_training_config,
)


class TestModelHyperparams:
    """Test the hyperparameters dataclass."""

    def test_defaults(self):
        hp = ModelHyperparams()
        assert hp.n_estimators == 100
        assert hp.learning_rate == 0.1
        assert hp.max_depth == 6
        assert hp.objective == "binary:logistic"

    def test_to_xgb_params_returns_dict(self):
        hp = ModelHyperparams(n_estimators=200, learning_rate=0.05)
        params = hp.to_xgb_params()
        assert params["n_estimators"] == 200
        assert params["learning_rate"] == 0.05
        assert "objective" in params
        assert "max_depth" in params

    def test_is_frozen(self):
        hp = ModelHyperparams()
        with pytest.raises(AttributeError):
            hp.n_estimators = 999


class TestTrainingConfig:
    """Test the complete training config dataclass."""

    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.test_size == 0.2
        assert cfg.validation_size == 0.15
        assert cfg.random_state == 42
        assert cfg.early_stopping_rounds == 30
        assert cfg.tuning_cv_folds == 5
        assert cfg.threshold_metric == "f1"
        assert cfg.smote_strategy == "minority"
        assert cfg.mlflow_experiment_name == "credit-risk-xai"

    def test_is_frozen(self):
        cfg = TrainingConfig()
        with pytest.raises(AttributeError):
            cfg.test_size = 0.5


class TestLoadTrainingConfig:
    """Test YAML config loading."""

    def test_returns_defaults_when_file_missing(self, tmp_path: Path):
        cfg = load_training_config(tmp_path / "nonexistent.yml")
        assert cfg.test_size == 0.2
        assert cfg.model.n_estimators == 100

    def test_loads_from_yaml(self, tmp_path: Path):
        config_file = tmp_path / "test_config.yml"
        config_file.write_text(
            textwrap.dedent("""\
            data:
              test_size: 0.3
              random_state: 99
            validation:
              size: 0.25
              early_stopping_rounds: 40
            tuning:
              cv_folds: 4
            threshold:
              metric: f1
            model:
              n_estimators: 250
              learning_rate: 0.05
              max_depth: 8
            smote:
              sampling_strategy: auto
              k_neighbors: 3
            mlflow:
              experiment_name: test-experiment
              run_name_prefix: test
              tags:
                env: ci
        """)
        )

        cfg = load_training_config(config_file)

        assert cfg.test_size == 0.3
        assert cfg.validation_size == 0.25
        assert cfg.random_state == 99
        assert cfg.early_stopping_rounds == 40
        assert cfg.tuning_cv_folds == 4
        assert cfg.threshold_metric == "f1"
        assert cfg.model.n_estimators == 250
        assert cfg.model.learning_rate == 0.05
        assert cfg.model.max_depth == 8
        assert cfg.smote_strategy == "auto"
        assert cfg.smote_k_neighbors == 3
        assert cfg.mlflow_experiment_name == "test-experiment"
        assert cfg.mlflow_run_name_prefix == "test"
        assert cfg.mlflow_tags == {"env": "ci"}

    def test_partial_yaml_fills_defaults(self, tmp_path: Path):
        config_file = tmp_path / "partial.yml"
        config_file.write_text(
            textwrap.dedent("""\
            model:
              n_estimators: 50
        """)
        )

        cfg = load_training_config(config_file)

        assert cfg.model.n_estimators == 50
        # Everything else defaults
        assert cfg.test_size == 0.2
        assert cfg.validation_size == 0.15
        assert cfg.model.learning_rate == 0.1
        assert cfg.random_state == 42

    def test_empty_yaml_returns_defaults(self, tmp_path: Path):
        config_file = tmp_path / "empty.yml"
        config_file.write_text("")

        cfg = load_training_config(config_file)
        assert cfg.test_size == 0.2
        assert cfg.model.n_estimators == 100

    def test_loads_actual_project_config(self):
        """Verify the real configs/training.yml loads without error."""
        project_config = Path("configs/training.yml")
        if project_config.exists():
            cfg = load_training_config(project_config)
            assert cfg.model.n_estimators > 0
            assert cfg.test_size > 0

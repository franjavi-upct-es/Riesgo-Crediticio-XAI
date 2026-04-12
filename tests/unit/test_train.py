# tests/unit/test_train.py
"""Unit tests for src.model.train.

Mocks all external dependencies (dataset loading, pipeline fitting,
XGBoost, SMOTE, MLflow, evaluation) to verify the pipeline orchestration.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import src.model.train as train_module
from src.model.train import _generate_synthetic_test_set, train_model
from src.model.training_config import ModelHyperparams, TrainingConfig


@pytest.fixture
def mock_training_config():
    return TrainingConfig(
        test_size=0.2,
        random_state=42,
        model=ModelHyperparams(n_estimators=10, learning_rate=0.1),
        smote_strategy="minority",
        smote_k_neighbors=3,
        mlflow_experiment_name="test",
        mlflow_run_name_prefix="test-run",
        mlflow_tags={"env": "test"},
    )


@pytest.fixture
def mock_dataset():
    """Simple 20-row dataset."""
    X = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 20),
            "income": np.random.randint(1000, 100000, 20),
            "grade": np.random.choice(["A", "B", "C"], 20),
        }
    )
    y = pd.Series(np.random.choice([0, 1], 20, p=[0.7, 0.3]), name="target")
    return X, y


@pytest.fixture
def mock_schema():
    from src.data.schema import (
        DatasetSchema,
        FeatureSchema,
        SourceSchema,
        TargetSchema,
    )

    return DatasetSchema(
        id="test_ds",
        name="Test",
        description="",
        source=SourceSchema(type="csv"),
        target=TargetSchema(column="target"),
        features=[
            FeatureSchema(name="age", type="numerical"),
            FeatureSchema(name="income", type="numerical"),
            FeatureSchema(
                name="grade", type="categorical", options=["A", "B", "C"]
            ),
        ],
    )


class TestTrainModel:
    @patch("src.model.train.joblib")
    @patch("src.model.train.SMOTE")
    @patch("src.model.train.ExperimentTracker")
    @patch("src.model.evaluate.compute_and_save_evaluation")
    @patch("src.model.train.xgb.XGBClassifier")
    @patch("src.model.train.fit_and_save_pipeline")
    @patch("src.model.train.build_preprocessing_pipeline")
    @patch("src.model.train.load_dataset")
    @patch("src.model.train.load_dataset_schema")
    @patch("src.model.train.load_training_config")
    def test_full_pipeline_runs(
        self,
        mock_load_config,
        mock_load_schema,
        mock_load_ds,
        mock_build_pipe,
        mock_fit_pipe,
        mock_xgb_cls,
        mock_eval,
        mock_tracker_cls,
        mock_smote_cls,
        mock_joblib,
        mock_training_config,
        mock_dataset,
        mock_schema,
        tmp_path,
    ):
        mock_load_config.return_value = mock_training_config
        mock_load_schema.return_value = mock_schema

        X, y = mock_dataset
        mock_load_ds.return_value = (X, y)

        # Pipeline mock
        pipe = MagicMock()
        transformed = np.random.randn(len(X), 4)
        pipe.transform.return_value = transformed
        mock_build_pipe.return_value = pipe
        mock_fit_pipe.return_value = ["age", "income", "grade_B", "grade_C"]

        # XGBoost mock
        model = MagicMock()
        model.fit.return_value = None
        mock_xgb_cls.return_value = model

        # SMOTE mock
        smote = MagicMock()
        smote.fit_resample.return_value = (
            pd.DataFrame(
                np.random.randn(10, 4),
                columns=["age", "income", "grade_B", "grade_C"],
            ),
            pd.Series([0] * 5 + [1] * 5, name="target"),
        )
        mock_smote_cls.return_value = smote

        # Eval mock
        mock_eval.return_value = {"metrics": {"auc": 0.8, "f1": 0.7}}

        # Tracker mock
        tracker = MagicMock()
        tracker.start_run.return_value.__enter__ = MagicMock()
        tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_tracker_cls.return_value = tracker

        with patch("src.model.train.settings") as mock_settings:
            mock_settings.model.dir = tmp_path / "models"
            mock_settings.data.dir = tmp_path / "data"

            train_model(dataset_id="test_ds")

        mock_build_pipe.assert_called_once_with(mock_schema)
        mock_fit_pipe.assert_called_once()
        model.fit.assert_called_once()
        mock_eval.assert_called_once()
        tracker.start_run.assert_called_once()
        tracker.log_params.assert_called_once()
        tracker.log_metrics.assert_called_once()


class TestGenerateSyntheticTestSet:
    def test_generates_balanced_set(self, mock_training_config, tmp_path):
        X_test = pd.DataFrame(
            {
                "f1": np.random.randn(50),
                "f2": np.random.randn(50),
            }
        )
        y_test = pd.Series([0] * 40 + [1] * 10, name="target")

        _generate_synthetic_test_set(
            X_test, y_test, mock_training_config, tmp_path
        )

        output_path = tmp_path / "synthetic_test_set.csv"
        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert "target" in df.columns
        # SMOTE should have balanced the classes
        assert df["target"].value_counts()[0] == df["target"].value_counts()[1]


class TestMainCLI:
    @patch("src.model.train.train_model")
    @patch("src.model.train.structlog")
    def test_main_calls_train(self, mock_structlog, mock_train):
        with patch("sys.argv", ["train", "--dataset", "german_credit"]):
            train_module.main()
        mock_train.assert_called_once_with(
            dataset_id="german_credit", config_path=None
        )

    @patch("src.model.train.train_model", side_effect=Exception("boom"))
    @patch("src.model.train.structlog")
    def test_main_exits_on_failure(self, mock_structlog, mock_train):
        with (
            patch("sys.argv", ["train", "--dataset", "test"]),
            pytest.raises(SystemExit),
        ):
            train_module.main()

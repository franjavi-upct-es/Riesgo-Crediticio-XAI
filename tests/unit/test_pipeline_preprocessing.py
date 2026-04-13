# tests/unit/test_pipeline_preprocessing.py
"""Unit tests for the sklearn Pipeline preprocessing functions.

Verifies pipeline construction, fitting, serialization, and
single-input transformation for the multi-dataset architecture.
"""

import pandas as pd
import pytest
from src.data.preprocessing import (
    build_preprocessing_pipeline,
    fit_and_save_pipeline,
    load_pipeline,
    preprocess_with_pipeline,
)
from src.data.schema import (
    DatasetSchema,
    FeatureSchema,
    SourceSchema,
    TargetSchema,
)


@pytest.fixture
def simple_schema() -> DatasetSchema:
    """A minimal schema with 2 numerical + 2 categorical features."""
    return DatasetSchema(
        id="test",
        name="Test",
        description="Test dataset",
        source=SourceSchema(type="csv"),
        target=TargetSchema(column="target"),
        features=[
            FeatureSchema(name="age", type="numerical", min=18, max=99),
            FeatureSchema(name="income", type="numerical", min=0, max=1000000),
            FeatureSchema(name="grade", type="categorical", options=["A", "B", "C"]),
            FeatureSchema(
                name="status",
                type="categorical",
                options=["active", "inactive"],
            ),
        ],
    )


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small DataFrame matching the simple_schema."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 30],
            "income": [30000, 50000, 70000, 90000, 40000],
            "grade": ["A", "B", "C", "A", "B"],
            "status": ["active", "inactive", "active", "active", "inactive"],
        }
    )


class TestBuildPreprocessingPipeline:
    def test_returns_column_transformer(self, simple_schema):
        pipeline = build_preprocessing_pipeline(simple_schema)
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "transform")

    def test_pipeline_has_numerical_and_categorical_transformers(self, simple_schema):
        pipeline = build_preprocessing_pipeline(simple_schema)
        names = [name for name, _, _ in pipeline.transformers]
        assert "num" in names
        assert "cat" in names


class TestFitAndSavePipeline:
    def test_fits_and_returns_feature_names(self, simple_schema, sample_df, tmp_path):
        pipeline = build_preprocessing_pipeline(simple_schema)
        path = tmp_path / "pipeline.pkl"
        feature_names = fit_and_save_pipeline(pipeline, sample_df, path)

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        # Should have 2 numerical + (3-1) + (2-1) = 5 features (drop_first)
        assert len(feature_names) == 5
        assert path.exists()

    def test_numerical_columns_preserved(self, simple_schema, sample_df, tmp_path):
        pipeline = build_preprocessing_pipeline(simple_schema)
        path = tmp_path / "pipeline.pkl"
        feature_names = fit_and_save_pipeline(pipeline, sample_df, path)

        assert "age" in feature_names
        assert "income" in feature_names

    def test_categorical_columns_one_hot_encoded(self, simple_schema, sample_df, tmp_path):
        pipeline = build_preprocessing_pipeline(simple_schema)
        path = tmp_path / "pipeline.pkl"
        feature_names = fit_and_save_pipeline(pipeline, sample_df, path)

        # drop_first=True: 'A' dropped for grade, 'active' dropped for status
        cat_features = [f for f in feature_names if f not in ("age", "income")]
        assert len(cat_features) == 3  # B, C from grade + inactive from status


class TestLoadPipeline:
    def test_loads_saved_pipeline(self, simple_schema, sample_df, tmp_path):
        pipeline = build_preprocessing_pipeline(simple_schema)
        path = tmp_path / "pipeline.pkl"
        fit_and_save_pipeline(pipeline, sample_df, path)

        loaded = load_pipeline(path)
        assert hasattr(loaded, "transform")

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline(tmp_path / "nonexistent.pkl")


class TestPreprocessWithPipeline:
    def test_transforms_single_input(self, simple_schema, sample_df, tmp_path):
        pipeline = build_preprocessing_pipeline(simple_schema)
        path = tmp_path / "pipeline.pkl"
        feature_names = fit_and_save_pipeline(pipeline, sample_df, path)

        input_dict = {
            "age": 30,
            "income": 50000,
            "grade": "B",
            "status": "active",
        }
        result = preprocess_with_pipeline(input_dict, pipeline, feature_names)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, len(feature_names))
        assert list(result.columns) == feature_names

    def test_handles_unknown_category(self, simple_schema, sample_df, tmp_path):
        pipeline = build_preprocessing_pipeline(simple_schema)
        path = tmp_path / "pipeline.pkl"
        feature_names = fit_and_save_pipeline(pipeline, sample_df, path)

        # "D" was not seen during fit — handle_unknown='ignore' → all zeros
        input_dict = {
            "age": 30,
            "income": 50000,
            "grade": "D",
            "status": "active",
        }
        result = preprocess_with_pipeline(input_dict, pipeline, feature_names)

        assert result.shape == (1, len(feature_names))
        # Grade columns should be all zeros for unknown "D"
        grade_cols = [c for c in feature_names if c.startswith("grade")]
        assert all(result[c].iloc[0] == 0 for c in grade_cols)

    def test_numerical_values_are_scaled(self, simple_schema, sample_df, tmp_path):
        pipeline = build_preprocessing_pipeline(simple_schema)
        path = tmp_path / "pipeline.pkl"
        feature_names = fit_and_save_pipeline(pipeline, sample_df, path)

        input_dict = {
            "age": 42,
            "income": 75000,
            "grade": "A",
            "status": "active",
        }
        result = preprocess_with_pipeline(input_dict, pipeline, feature_names)

        assert result["age"].iloc[0] != 42
        assert result["income"].iloc[0] != 75000
        assert abs(result["age"].iloc[0]) < 3
        assert abs(result["income"].iloc[0]) < 3

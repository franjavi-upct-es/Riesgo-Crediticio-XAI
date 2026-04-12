# tests/unit/test_schema.py
"""Unit tests for src.data.schema.

Verifies YAML loading, typed dataclass construction, feature lookups,
serialization to API format, and edge cases.
"""

import textwrap
from pathlib import Path

import pytest

from src.data.schema import (
    DatasetSchema,
    FeatureSchema,
    list_available_datasets,
    load_dataset_schema,
)


class TestFeatureSchema:
    """Test feature-level schema behavior."""

    def test_categorical_feature(self):
        f = FeatureSchema(
            name="color", type="categorical", options=["red", "blue"]
        )
        assert f.is_categorical
        assert not f.is_numerical
        assert f.default_value == "red"

    def test_numerical_feature(self):
        f = FeatureSchema(name="age", type="numerical", min=18, max=120)
        assert f.is_numerical
        assert not f.is_categorical
        assert f.default_value == 69  # (18 + 120) / 2

    def test_protected_flag(self):
        f = FeatureSchema(name="gender", type="categorical", protected=True)
        assert f.protected

    def test_numerical_no_bounds(self):
        f = FeatureSchema(name="x", type="numerical")
        assert f.default_value == 0


class TestDatasetSchema:
    """Test dataset-level schema methods."""

    @pytest.fixture
    def sample_schema(self) -> DatasetSchema:
        from src.data.schema import SourceSchema, TargetSchema

        return DatasetSchema(
            id="test_ds",
            name="Test Dataset",
            description="A test",
            source=SourceSchema(type="csv", filename="test.csv"),
            target=TargetSchema(column="target", labels={0: "Good", 1: "Bad"}),
            features=[
                FeatureSchema(name="age", type="numerical", min=18, max=99),
                FeatureSchema(
                    name="color", type="categorical", options=["r", "g"]
                ),
                FeatureSchema(
                    name="gender", type="categorical", protected=True
                ),
            ],
        )

    def test_feature_names(self, sample_schema: DatasetSchema):
        assert sample_schema.feature_names == ["age", "color", "gender"]

    def test_categorical_features(self, sample_schema: DatasetSchema):
        assert sample_schema.categorical_features == ["color", "gender"]

    def test_numerical_features(self, sample_schema: DatasetSchema):
        assert sample_schema.numerical_features == ["age"]

    def test_protected_features(self, sample_schema: DatasetSchema):
        assert sample_schema.protected_features == ["gender"]

    def test_get_feature(self, sample_schema: DatasetSchema):
        assert sample_schema.get_feature("age") is not None
        assert sample_schema.get_feature("nonexistent") is None

    def test_to_api_schema(self, sample_schema: DatasetSchema):
        api = sample_schema.to_api_schema()
        assert api["id"] == "test_ds"
        assert len(api["features"]) == 3
        assert api["features"][0]["name"] == "age"
        assert api["features"][2]["protected"] is True


class TestLoadDatasetSchema:
    """Test YAML loading."""

    def test_loads_valid_yaml(self, tmp_path: Path):
        yml = tmp_path / "test.yml"
        yml.write_text(
            textwrap.dedent("""\
            id: test
            name: Test
            description: A test dataset
            source:
              type: csv
              filename: test.csv
            target:
              column: label
              mapping:
                0: 0
                1: 1
              positive_label: 1
              labels:
                0: Good
                1: Bad
            features:
              - name: age
                type: numerical
                min: 18
                max: 99
              - name: color
                type: categorical
                options: ["red", "blue"]
        """)
        )

        schema = load_dataset_schema("test", datasets_dir=tmp_path)
        assert schema.id == "test"
        assert len(schema.features) == 2
        assert schema.target.column == "label"
        assert schema.target.labels[0] == "Good"

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_dataset_schema("nonexistent", datasets_dir=tmp_path)

    def test_raises_on_empty_yaml(self, tmp_path: Path):
        yml = tmp_path / "empty.yml"
        yml.write_text("")
        with pytest.raises(ValueError, match="Empty"):
            load_dataset_schema("empty", datasets_dir=tmp_path)

    def test_loads_project_german_credit(self):
        """Verify the real German Credit schema loads."""
        schema = load_dataset_schema("german_credit")
        assert schema.id == "german_credit"
        assert len(schema.features) == 20
        assert "checking_status" in schema.feature_names
        assert len(schema.protected_features) >= 2

    def test_loads_project_lending_club(self):
        """Verify the real Lending Club schema loads."""
        schema = load_dataset_schema("lending_club")
        assert schema.id == "lending_club"
        assert len(schema.features) >= 15

    def test_loads_project_taiwan_credit(self):
        """Verify the real Taiwan Credit schema loads."""
        schema = load_dataset_schema("taiwan_credit")
        assert schema.id == "taiwan_credit"
        assert len(schema.features) >= 20


class TestListAvailableDatasets:
    """Test dataset discovery."""

    def test_lists_from_directory(self, tmp_path: Path):
        (tmp_path / "a.yml").touch()
        (tmp_path / "b.yml").touch()
        (tmp_path / "not_yaml.txt").touch()

        result = list_available_datasets(datasets_dir=tmp_path)
        assert result == ["a", "b"]

    def test_empty_directory(self, tmp_path: Path):
        assert list_available_datasets(datasets_dir=tmp_path) == []

    def test_nonexistent_directory(self, tmp_path: Path):
        assert list_available_datasets(datasets_dir=tmp_path / "nope") == []

    def test_lists_project_datasets(self):
        """Verify the real configs/datasets/ directory has entries."""
        datasets = list_available_datasets()
        assert "german_credit" in datasets
        assert "lending_club" in datasets
        assert "taiwan_credit" in datasets
